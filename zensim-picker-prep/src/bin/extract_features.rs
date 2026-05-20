//! extract_features — extract zenanalyze SUPPORTED features per source.
//!
//! Walks a sources dir, decodes each image to RGB8, runs
//! `zenanalyze::analyze_features_rgb8` with `FeatureSet::SUPPORTED`,
//! and emits one parquet row per source: `ref_basename` + one column
//! per feature (named `feat_<id>`).
//!
//! Used as the per-source content cache that joins into every codec
//! parquet via `ref_basename`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

#[derive(Parser, Debug)]
#[command(name = "extract_features")]
struct Args {
    /// Directory of source images (PNG / JPEG).
    #[arg(long)]
    sources: PathBuf,

    /// Output parquet path.
    #[arg(long)]
    output: PathBuf,

    /// Max sources to process (0 = all).
    #[arg(long, default_value_t = 0)]
    max_sources: usize,
}

fn load_rgb8(p: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(p).with_context(|| format!("decoding {:?}", p))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), w, h))
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("extract_features");
    eprintln!("  sources: {:?}", args.sources);
    eprintln!("  output:  {:?}", args.output);

    let mut src_paths: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&args.sources)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_file() {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg") {
                src_paths.push(p);
            }
        }
    }
    src_paths.sort();
    if args.max_sources > 0 && src_paths.len() > args.max_sources {
        src_paths.truncate(args.max_sources);
    }
    let n = src_paths.len();
    eprintln!("  sources: {} files", n);
    if n == 0 {
        return Err(anyhow!("no sources found"));
    }

    // Build the SUPPORTED feature list once, ordered by feature id.
    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let features: Vec<AnalysisFeature> = FeatureSet::SUPPORTED.iter().collect();
    eprintln!("  features: {} (SUPPORTED)", features.len());

    let t_start = Instant::now();
    let progress = AtomicUsize::new(0);

    // Per-source: (basename, Vec<Option<f32>> indexed by features[i])
    let rows: Vec<(String, Vec<Option<f32>>)> = src_paths
        .par_iter()
        .map(|p| {
            let basename = p
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string());

            let (rgb, w, h) = match load_rgb8(p) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("  WARN load {:?}: {e}", p);
                    let n_done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if n_done % 50 == 0 {
                        eprintln!(
                            "  {}/{} ({:.1}%)",
                            n_done,
                            n,
                            100.0 * n_done as f64 / n as f64
                        );
                    }
                    return (basename, vec![None; features.len()]);
                }
            };

            let res = zenanalyze::analyze_features_rgb8(&rgb, w, h, &query);
            let vals: Vec<Option<f32>> = features.iter().map(|&f| res.get_f32(f)).collect();

            let n_done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if n_done % 50 == 0 || n_done == n {
                let elapsed = t_start.elapsed().as_secs_f64();
                let rate = n_done as f64 / elapsed;
                let eta = if rate > 0.0 {
                    (n - n_done) as f64 / rate
                } else {
                    0.0
                };
                eprintln!(
                    "  {}/{} ({:.1}%) {:.1}s rate={:.2}/s eta={:.0}s",
                    n_done,
                    n,
                    100.0 * n_done as f64 / n as f64,
                    elapsed,
                    rate,
                    eta
                );
            }
            (basename, vals)
        })
        .collect();

    eprintln!(
        "  extracted {} sources in {:.1}s",
        rows.len(),
        t_start.elapsed().as_secs_f64()
    );

    // ── parquet output ─────────────────────────────────────────────
    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).ok();
    }

    // Schema: ref_basename + feat_<id> per feature
    let mut fields: Vec<Arc<Field>> = Vec::with_capacity(features.len() + 1);
    fields.push(Arc::new(Field::new("ref_basename", DataType::Utf8, false)));
    for &f in &features {
        fields.push(Arc::new(Field::new(
            format!("feat_{}", f as u32),
            DataType::Float32,
            true,
        )));
    }
    let schema = Arc::new(Schema::new(fields));

    let basenames: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(features.len() + 1);
    arrays.push(Arc::new(StringArray::from(basenames)));
    for (col_idx, _) in features.iter().enumerate() {
        let col: Vec<Option<f32>> = rows.iter().map(|r| r.1[col_idx]).collect();
        arrays.push(Arc::new(Float32Array::from(col)));
    }

    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let out_file =
        fs::File::create(&args.output).with_context(|| format!("creating {:?}", args.output))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(out_file, schema.clone(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    eprintln!(
        "  wrote {} rows × {} cols → {:?}",
        rows.len(),
        features.len() + 1,
        args.output
    );
    eprintln!("  total wall: {:.1}s", t_start.elapsed().as_secs_f64());
    Ok(())
}
