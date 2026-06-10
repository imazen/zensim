//! score_pairs_tuner — score (ref, dist) PNG pairs with `PreviewV0_5Tuner`
//!
//! Reads a TSV of pairs (header: `image_path codec q knob_tuple_json
//! ref_path dist_path`) and emits a parquet with columns
//! `image_path codec q knob_tuple_json ref_basename achieved_zensim_tuner`.
//!
//! Built specifically for the per-codec picker data-prep pipeline
//! (see ~/.claude/projects/-home-lilith-work-zen/memory/project_per_codec_picker_design.md).
//! The training input shape is `ref_basename | feat_0..feat_101 | q |
//! achieved_zensim_tuner`; this binary writes the `achieved_zensim_tuner`
//! half, joined downstream by `ref_basename` to a zenanalyze 102-feature
//! per-source parquet.
//!
//! Usage:
//!   score_pairs_tuner --pairs <pairs.tsv> --output <out.parquet>
//!                     [--threads N] [--cache-refs]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use zensim::{PixelFormat, StridedBytes, Zensim};

#[derive(Parser, Debug)]
#[command(name = "score_pairs_tuner")]
struct Args {
    /// TSV with `image_path<TAB>codec<TAB>q<TAB>knob_tuple_json<TAB>ref_path<TAB>dist_path` (header required).
    #[arg(long)]
    pairs: PathBuf,

    /// Output parquet path (one row per input pair).
    #[arg(long)]
    output: PathBuf,

    /// Number of rayon threads (default: auto).
    #[arg(long)]
    threads: Option<usize>,

    /// Cache reference images (saves I/O when many rows share a ref —
    /// always true for picker data prep where each ref maps to 19 q values).
    #[arg(long, default_value_t = true)]
    cache_refs: bool,
}

#[derive(Clone, Debug)]
struct PairRow {
    image_path: String,
    codec: String,
    q: i64,
    knob_tuple_json: String,
    ref_path: String,
    dist_path: String,
}

fn ref_basename(ref_path: &str) -> String {
    Path::new(ref_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| ref_path.to_string())
}

fn load_rgb8(p: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(p).with_context(|| format!("decoding {:?}", p))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), w, h))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .map_err(|e| anyhow!("rayon init: {e}"))?;
    }

    eprintln!("score_pairs_tuner");
    eprintln!("  pairs:  {:?}", args.pairs);
    eprintln!("  output: {:?}", args.output);
    eprintln!("  profile: PreviewV0_5Tuner");

    let t_start = Instant::now();

    // --- read pairs TSV
    let file = File::open(&args.pairs).with_context(|| format!("opening {:?}", args.pairs))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().ok_or_else(|| anyhow!("empty pairs TSV"))??;
    let cols: Vec<&str> = header.split('\t').collect();
    let pos = |name: &str| -> Result<usize> {
        cols.iter()
            .position(|c| *c == name)
            .ok_or_else(|| anyhow!("missing '{}' column in header (got {:?})", name, cols))
    };
    let image_path_idx = pos("image_path")?;
    let codec_idx = pos("codec")?;
    let q_idx = pos("q")?;
    let knob_idx = pos("knob_tuple_json")?;
    let ref_idx = pos("ref_path")?;
    let dist_idx = pos("dist_path")?;

    let mut pairs: Vec<PairRow> = Vec::new();
    for (i, ln) in lines.enumerate() {
        let ln = ln?;
        let parts: Vec<&str> = ln.split('\t').collect();
        let max_idx = [
            image_path_idx,
            codec_idx,
            q_idx,
            knob_idx,
            ref_idx,
            dist_idx,
        ]
        .iter()
        .copied()
        .max()
        .unwrap();
        if parts.len() <= max_idx {
            return Err(anyhow!("malformed line {}: {:?}", i + 2, ln));
        }
        let q: i64 = parts[q_idx]
            .parse()
            .with_context(|| format!("parsing q on line {}: {:?}", i + 2, parts[q_idx]))?;
        pairs.push(PairRow {
            image_path: parts[image_path_idx].to_string(),
            codec: parts[codec_idx].to_string(),
            q,
            knob_tuple_json: parts[knob_idx].to_string(),
            ref_path: parts[ref_idx].to_string(),
            dist_path: parts[dist_idx].to_string(),
        });
    }
    let total = pairs.len();
    eprintln!(
        "  read {} pairs in {:.2}s",
        total,
        t_start.elapsed().as_secs_f64()
    );

    // --- dedupe refs (essential — for 19 q × N sources, each ref repeated 19×)
    let mut seen: HashMap<String, u32> = HashMap::new();
    let mut unique_refs: Vec<String> = Vec::new();
    let mut row_ref_idx: Vec<u32> = Vec::with_capacity(total);
    for p in &pairs {
        if let Some(&idx) = seen.get(&p.ref_path) {
            row_ref_idx.push(idx);
        } else {
            let idx = unique_refs.len() as u32;
            seen.insert(p.ref_path.clone(), idx);
            unique_refs.push(p.ref_path.clone());
            row_ref_idx.push(idx);
        }
    }
    eprintln!(
        "  unique refs: {} (deduplicated from {} rows)",
        unique_refs.len(),
        total
    );

    // --- load each unique ref into RAM (parallel)
    let t_ref_load = Instant::now();
    let n_refs = unique_refs.len();
    let progress = AtomicUsize::new(0);
    let ref_cache: Vec<Option<(Vec<u8>, u32, u32)>> = unique_refs
        .par_iter()
        .map(|p| {
            let r = load_rgb8(Path::new(p)).ok();
            let n = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(100) || n == n_refs {
                eprintln!("  ref-load: {}/{}", n, n_refs);
            }
            r
        })
        .collect();
    eprintln!(
        "  ref-load: done in {:.2}s",
        t_ref_load.elapsed().as_secs_f64()
    );

    // --- score each pair with PreviewV0_5Tuner
    let t_score = Instant::now();
    let zensim = Zensim::new(zensim_experimental::preview_v0_5_tuner()).with_parallel(false);
    let progress = AtomicUsize::new(0);
    let scores: Vec<f64> = (0..total)
        .into_par_iter()
        .map(|i| {
            let row = &pairs[i];
            let ref_opt = ref_cache[row_ref_idx[i] as usize].as_ref();
            let dist_opt = load_rgb8(Path::new(&row.dist_path)).ok();
            let result: f64 = match (ref_opt, dist_opt) {
                (Some((rpx, rw, rh)), Some((dpx, dw, dh))) => {
                    if *rw != dw || *rh != dh || rpx.len() != dpx.len() {
                        eprintln!(
                            "  WARN: dim mismatch on row {} (ref={}x{} dist={}x{}) — skipping",
                            i, rw, rh, dw, dh
                        );
                        f64::NAN
                    } else {
                        let w = *rw as usize;
                        let h = *rh as usize;
                        let stride = w * 3;
                        let src =
                            match StridedBytes::try_new(rpx, w, h, stride, PixelFormat::Srgb8Rgb) {
                                Ok(s) => s,
                                Err(e) => {
                                    eprintln!("  WARN: ref slice fail row {}: {:?}", i, e);
                                    return f64::NAN;
                                }
                            };
                        let dst = match StridedBytes::try_new(
                            &dpx,
                            w,
                            h,
                            stride,
                            PixelFormat::Srgb8Rgb,
                        ) {
                            Ok(s) => s,
                            Err(e) => {
                                eprintln!("  WARN: dist slice fail row {}: {:?}", i, e);
                                return f64::NAN;
                            }
                        };
                        match zensim.compute(&src, &dst) {
                            Ok(r) => r.score(),
                            Err(e) => {
                                eprintln!("  WARN: zensim compute fail row {}: {:?}", i, e);
                                f64::NAN
                            }
                        }
                    }
                }
                _ => {
                    eprintln!(
                        "  WARN: load failure for row {} (ref={:?} dist={:?})",
                        i, row.ref_path, row.dist_path
                    );
                    f64::NAN
                }
            };
            let n = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(500) || n == total {
                let elapsed = t_score.elapsed().as_secs_f64();
                let rate = n as f64 / elapsed;
                let eta = (total - n) as f64 / rate;
                eprintln!(
                    "  score: {}/{} ({:.1}%) {:.1}s rate={:.1}/s eta={:.0}s",
                    n,
                    total,
                    100.0 * n as f64 / total as f64,
                    elapsed,
                    rate,
                    eta
                );
            }
            result
        })
        .collect();
    eprintln!(
        "  score: {} rows in {:.2}s",
        total,
        t_score.elapsed().as_secs_f64()
    );

    // --- write parquet
    let fields: Vec<Arc<Field>> = vec![
        Arc::new(Field::new("image_path", DataType::Utf8, false)),
        Arc::new(Field::new("codec", DataType::Utf8, false)),
        Arc::new(Field::new("q", DataType::Int64, false)),
        Arc::new(Field::new("knob_tuple_json", DataType::Utf8, false)),
        Arc::new(Field::new("ref_basename", DataType::Utf8, false)),
        Arc::new(Field::new("achieved_zensim_tuner", DataType::Float64, true)),
    ];
    let out_schema = Arc::new(Schema::new(fields));

    let image_paths: Vec<String> = pairs.iter().map(|p| p.image_path.clone()).collect();
    let codecs: Vec<String> = pairs.iter().map(|p| p.codec.clone()).collect();
    let qs: Vec<i64> = pairs.iter().map(|p| p.q).collect();
    let knobs: Vec<String> = pairs.iter().map(|p| p.knob_tuple_json.clone()).collect();
    let basenames: Vec<String> = pairs.iter().map(|p| ref_basename(&p.ref_path)).collect();

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(image_paths)),
        Arc::new(StringArray::from(codecs)),
        Arc::new(Int64Array::from(qs)),
        Arc::new(StringArray::from(knobs)),
        Arc::new(StringArray::from(basenames)),
        Arc::new(Float64Array::from(scores.clone())),
    ];
    let batch = RecordBatch::try_new(out_schema.clone(), arrays)?;

    let out_file =
        File::create(&args.output).with_context(|| format!("creating {:?}", args.output))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(out_file, out_schema.clone(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    let n_nan = scores.iter().filter(|s| s.is_nan()).count();
    let valid: Vec<f64> = scores.iter().copied().filter(|s| s.is_finite()).collect();
    let (mn, mx, mean) = if valid.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        let mn = valid.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean: f64 = valid.iter().sum::<f64>() / valid.len() as f64;
        (mn, mx, mean)
    };
    eprintln!(
        "  wrote {} rows → {:?}; achieved_zensim_tuner: min={:.2} mean={:.2} max={:.2} n_nan={}",
        total, args.output, mn, mean, mx, n_nan
    );

    Ok(())
}
