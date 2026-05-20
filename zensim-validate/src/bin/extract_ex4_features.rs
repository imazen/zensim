//! EX-4 corpus rebuild: add XYB/LMS-biased-log (24 per-ref) features +
//! optionally CVVDP-shape (19 per-pair) features to an existing
//! mix_targets / cvvdp_iwssim_large parquet.
//!
//! Input parquet schema:
//!   ref_basename, [target columns...], f0..f<N-1>
//!
//! Output parquet schema:
//!   <all input columns> + f<N>..f<N+23> (XYB+LMS per-ref)
//!   + optionally f<N+24>..f<N+42> (CVVDP-shape per-pair)
//!
//! Per-ref features are computed ONCE per unique `ref_basename` and
//! reused across all rows referencing it (huge speed-up — safesyn has
//! ~3200 unique refs across 196k rows; cvvdp_iwssim_large has 200 refs
//! for 73k rows).
//!
//! CVVDP per-pair features require both ref AND dist images. In this
//! initial rebuild the dist images are NOT regenerated; rows where dist
//! image is unavailable receive zero-filled per-pair features. The
//! per-ref 24 features remain valid for all rows.
//!
//! Usage:
//!   extract_ex4_features \
//!     --input <in.parquet> \
//!     --output <out.parquet> \
//!     --refs-root <dir>           (where ref_basename resolves)
//!     [--refs-suffix .png]        (append to basename if no extension)
//!     [--pair-features]           (compute 19 per-pair features)
//!     [--dist-grid <pattern>]     (e.g. /mnt/v/dataset/kadid10k/images/{base}_{q}.png)

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, StringArray, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use zensim::cvvdp_features::{CVVDP_FEATURE_COUNT, extract_cvvdp_features};
use zensim::xyb_lms_features::{XYB_LMS_FEATURE_COUNT, extract_xyb_lms_features};

#[derive(Parser, Debug)]
#[command(name = "extract_ex4_features")]
struct Args {
    /// Input parquet (ref_basename + targets + f0..)
    #[arg(long)]
    input: PathBuf,

    /// Output parquet (input + per-ref + optionally per-pair features)
    #[arg(long)]
    output: PathBuf,

    /// Directory containing reference images.
    #[arg(long)]
    refs_root: PathBuf,

    /// Suffix to append to ref_basename if it has no extension (e.g. ".png").
    #[arg(long, default_value = "")]
    refs_suffix: String,

    /// Also compute per-pair CVVDP-shape features (requires --dist-resolver).
    #[arg(long)]
    pair_features: bool,

    /// Pattern for resolving dist image path (placeholders TBD per corpus).
    #[arg(long)]
    dist_grid: Option<String>,

    /// Maximum parallel threads (default: rayon default).
    #[arg(long)]
    threads: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .map_err(|e| anyhow!("rayon init: {e}"))?;
    }

    eprintln!("EX-4 feature extractor");
    eprintln!("  input:  {:?}", args.input);
    eprintln!("  output: {:?}", args.output);
    eprintln!(
        "  refs:   {:?} (suffix {:?})",
        args.refs_root, args.refs_suffix
    );
    eprintln!("  per-pair: {}", args.pair_features);

    let t_start = Instant::now();

    // ------------------------------------------------------------------
    // Phase 1 — read input parquet entirely into memory (Arrow batches).
    // ------------------------------------------------------------------
    let file = File::open(&args.input).with_context(|| format!("opening {:?}", args.input))?;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).context("creating parquet reader")?;
    let in_schema = builder.schema().clone();
    let reader = builder
        .with_batch_size(16384)
        .build()
        .context("building parquet reader")?;

    let mut in_batches: Vec<RecordBatch> = Vec::new();
    let mut total_rows = 0usize;
    for b in reader {
        let b = b.context("reading parquet batch")?;
        total_rows += b.num_rows();
        in_batches.push(b);
    }
    eprintln!(
        "  read {} rows × {} cols in {:.2}s",
        total_rows,
        in_schema.fields().len(),
        t_start.elapsed().as_secs_f64()
    );

    // ------------------------------------------------------------------
    // Phase 2 — find ref_basename column + collect unique refs.
    // ------------------------------------------------------------------
    let ref_col_idx = in_schema
        .fields()
        .iter()
        .position(|f| f.name() == "ref_basename")
        .ok_or_else(|| anyhow!("input missing 'ref_basename' column"))?;

    let mut unique_refs: Vec<String> = Vec::new();
    let mut row_ref_idx: Vec<u32> = Vec::with_capacity(total_rows);
    {
        let mut seen: HashMap<String, u32> = HashMap::new();
        for batch in &in_batches {
            let col = batch.column(ref_col_idx);
            let arr = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                anyhow!("ref_basename not StringArray (got {:?})", col.data_type())
            })?;
            for i in 0..arr.len() {
                let s = arr.value(i);
                if let Some(&idx) = seen.get(s) {
                    row_ref_idx.push(idx);
                } else {
                    let idx = unique_refs.len() as u32;
                    seen.insert(s.to_string(), idx);
                    unique_refs.push(s.to_string());
                    row_ref_idx.push(idx);
                }
            }
        }
    }
    eprintln!("  unique refs: {}", unique_refs.len());

    // ------------------------------------------------------------------
    // Phase 3 — extract 24 XYB+LMS features per unique ref (parallel).
    // ------------------------------------------------------------------
    let progress = AtomicUsize::new(0);
    let total_refs = unique_refs.len();
    let t_extract = Instant::now();

    let per_ref_features: Vec<Vec<f32>> = unique_refs
        .par_iter()
        .map(|basename| {
            let p = resolve_ref_path(&args.refs_root, basename, &args.refs_suffix);
            let result = match load_rgb8(&p) {
                Ok((pixels, _w, _h)) => extract_xyb_lms_features(&pixels),
                Err(e) => {
                    eprintln!("  WARN: failed to load ref {:?}: {e}", p);
                    vec![f32::NAN; XYB_LMS_FEATURE_COUNT]
                }
            };
            let n = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 25 == 0 || n == total_refs {
                eprintln!(
                    "  per-ref: {}/{} ({:.1}%)",
                    n,
                    total_refs,
                    100.0 * n as f64 / total_refs as f64
                );
            }
            result
        })
        .collect();

    eprintln!(
        "  XYB+LMS extract: {} refs in {:.2}s",
        unique_refs.len(),
        t_extract.elapsed().as_secs_f64()
    );

    // ------------------------------------------------------------------
    // Phase 4 — optionally extract per-pair CVVDP features.
    // For initial rebuild we emit zeros if dist resolution not configured.
    // ------------------------------------------------------------------
    let pair_features: Vec<Vec<f32>> = if args.pair_features {
        // TODO: dist resolution per-corpus. For now emit zeros.
        eprintln!("  WARN: --pair-features set but dist resolver not implemented in this build");
        eprintln!("        emitting NaN-filled per-pair feature block (extractor stub)");
        (0..total_rows)
            .map(|_| vec![f32::NAN; CVVDP_FEATURE_COUNT])
            .collect()
    } else {
        Vec::new()
    };

    // ------------------------------------------------------------------
    // Phase 5 — broadcast per-ref features to per-row + assemble output.
    // ------------------------------------------------------------------
    let mut new_per_row_xyb: Vec<Vec<f32>> = Vec::with_capacity(total_rows);
    for &ri in &row_ref_idx {
        new_per_row_xyb.push(per_ref_features[ri as usize].clone());
    }

    // Determine current max fN in input schema so we append after.
    let max_existing_f: i64 = in_schema
        .fields()
        .iter()
        .filter_map(|f| {
            f.name()
                .strip_prefix('f')
                .and_then(|s| s.parse::<i64>().ok())
        })
        .max()
        .unwrap_or(-1);
    let first_new_f = (max_existing_f + 1) as usize;
    eprintln!("  appending new features starting at f{}", first_new_f);

    // Build output Arrow schema = input fields + new fN..fN+23 [+24..+42].
    let mut new_fields: Vec<Arc<Field>> = in_schema.fields().iter().cloned().collect();
    for i in 0..XYB_LMS_FEATURE_COUNT {
        new_fields.push(Arc::new(Field::new(
            format!("f{}", first_new_f + i),
            DataType::Float32,
            false,
        )));
    }
    let pair_start = first_new_f + XYB_LMS_FEATURE_COUNT;
    if args.pair_features {
        for i in 0..CVVDP_FEATURE_COUNT {
            new_fields.push(Arc::new(Field::new(
                format!("f{}", pair_start + i),
                DataType::Float32,
                false,
            )));
        }
    }
    let out_schema = Arc::new(Schema::new(new_fields));

    // Build columns for the new features (single concatenated batch).
    // Per-ref: XYB_LMS_FEATURE_COUNT columns; per-pair: CVVDP_FEATURE_COUNT.
    let mut new_xyb_cols: Vec<Vec<f32>> =
        vec![Vec::with_capacity(total_rows); XYB_LMS_FEATURE_COUNT];
    for row in &new_per_row_xyb {
        for (i, &v) in row.iter().enumerate() {
            new_xyb_cols[i].push(v);
        }
    }
    let mut new_pair_cols: Vec<Vec<f32>> = if args.pair_features {
        vec![Vec::with_capacity(total_rows); CVVDP_FEATURE_COUNT]
    } else {
        Vec::new()
    };
    if args.pair_features {
        for row in &pair_features {
            for (i, &v) in row.iter().enumerate() {
                new_pair_cols[i].push(v);
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 6 — write output parquet.
    // ------------------------------------------------------------------
    let out_file =
        File::create(&args.output).with_context(|| format!("creating {:?}", args.output))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(out_file, out_schema.clone(), Some(props))
        .context("creating ArrowWriter")?;

    let mut row_offset = 0usize;
    for batch in &in_batches {
        let n = batch.num_rows();
        // Copy input columns.
        let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
        // Add new XYB+LMS columns (slice from full arrays).
        for col_data in &new_xyb_cols {
            let slice = &col_data[row_offset..row_offset + n];
            cols.push(Arc::new(Float32Array::from_iter_values(
                slice.iter().copied(),
            )));
        }
        if args.pair_features {
            for col_data in &new_pair_cols {
                let slice = &col_data[row_offset..row_offset + n];
                cols.push(Arc::new(Float32Array::from_iter_values(
                    slice.iter().copied(),
                )));
            }
        }
        let out_batch = RecordBatch::try_new(out_schema.clone(), cols)
            .context("creating output RecordBatch")?;
        writer.write(&out_batch).context("writing batch")?;
        row_offset += n;
    }
    writer.close().context("closing writer")?;

    eprintln!(
        "  wrote {} rows × {} cols → {:?}",
        total_rows,
        out_schema.fields().len(),
        args.output
    );
    eprintln!("  total wall time: {:.2}s", t_start.elapsed().as_secs_f64());

    // ------------------------------------------------------------------
    // Phase 7 — sanity stats on new feature columns (per-ref only).
    // ------------------------------------------------------------------
    eprintln!("\n=== New feature distribution sanity ===");
    for i in 0..XYB_LMS_FEATURE_COUNT {
        let col = &new_xyb_cols[i];
        let n_nan = col.iter().filter(|x| x.is_nan()).count();
        let valid: Vec<f32> = col.iter().copied().filter(|x| x.is_finite()).collect();
        if valid.is_empty() {
            eprintln!("  f{} (xyb_lms[{}]): ALL NaN", first_new_f + i, i);
            continue;
        }
        let n = valid.len() as f64;
        let mean = valid.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = valid
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = var.sqrt();
        let min = valid.iter().copied().fold(f32::INFINITY, f32::min);
        let max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let n_zero = col.iter().filter(|&&x| x == 0.0).count();
        eprintln!(
            "  f{} (xyb_lms[{:2}]): min={:.4} max={:.4} mean={:.4} std={:.4} nan={} zero={}",
            first_new_f + i,
            i,
            min,
            max,
            mean,
            std,
            n_nan,
            n_zero
        );
    }

    Ok(())
}

fn resolve_ref_path(refs_root: &Path, basename: &str, suffix: &str) -> PathBuf {
    let has_ext = basename.contains('.');
    if has_ext || suffix.is_empty() {
        refs_root.join(basename)
    } else {
        refs_root.join(format!("{}{}", basename, suffix))
    }
}

fn load_rgb8(p: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(p).with_context(|| format!("decoding {:?}", p))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), w, h))
}

#[allow(dead_code)]
fn extract_cvvdp_pair(
    ref_pixels: &[u8],
    dist_pixels: &[u8],
    width: usize,
    height: usize,
) -> Vec<f32> {
    extract_cvvdp_features(ref_pixels, dist_pixels, width, height)
}
