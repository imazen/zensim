//! Per-pair CVVDP-shape feature extractor (EX-4 Chunk C).
//!
//! Reads a TSV of `(ref_path, dist_path)` pairs (one row per training-corpus
//! row, in the same order as the corresponding parquet), computes the 19
//! CVVDP-shape per-pair features for each, and emits a Float32 parquet with
//! columns `f0..f18` aligned 1:1 with the input TSV.
//!
//! The resulting parquet is then merged into the corpus parquet downstream
//! (Python pandas join on row index) to produce the final 343-feature
//! training corpus.
//!
//! Why a separate binary: the per-pair extraction is the only step that
//! needs both ref and dist pixels, and we want to be able to run it
//! independently per-corpus without re-reading the 73k input parquet.
//!
//! Usage:
//!   extract_pair_features \
//!     --pairs <pairs.tsv>           (header: ref_path<TAB>dist_path)
//!     --output <out.parquet>        (f0..f18 columns, one row per pair)
//!     [--threads N]
//!     [--cache-refs]                (deduplicate ref loads — speeds up
//!                                    corpora with few unique refs)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use arrow::array::{ArrayRef, Float32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use zensim::cvvdp_features::{CVVDP_FEATURE_COUNT, extract_cvvdp_features};

#[derive(Parser, Debug)]
#[command(name = "extract_pair_features")]
struct Args {
    /// TSV with `ref_path<TAB>dist_path` per row (header required).
    #[arg(long)]
    pairs: PathBuf,

    /// Output parquet path (f0..f18 columns).
    #[arg(long)]
    output: PathBuf,

    /// Number of rayon threads (default: auto).
    #[arg(long)]
    threads: Option<usize>,

    /// Cache reference images (saves I/O when many rows share a ref).
    #[arg(long, default_value_t = true)]
    cache_refs: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .map_err(|e| anyhow!("rayon init: {e}"))?;
    }

    eprintln!("extract_pair_features (EX-4 Chunk C)");
    eprintln!("  pairs:  {:?}", args.pairs);
    eprintln!("  output: {:?}", args.output);

    let t_start = Instant::now();

    // --- read pairs TSV
    let file = File::open(&args.pairs).with_context(|| format!("opening {:?}", args.pairs))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().ok_or_else(|| anyhow!("empty pairs TSV"))??;
    let cols: Vec<&str> = header.split('\t').collect();
    let ref_idx = cols
        .iter()
        .position(|c| *c == "ref_path")
        .ok_or_else(|| anyhow!("missing 'ref_path' column in header"))?;
    let dist_idx = cols
        .iter()
        .position(|c| *c == "dist_path")
        .ok_or_else(|| anyhow!("missing 'dist_path' column in header"))?;

    let mut pairs: Vec<(String, String)> = Vec::new();
    for (i, ln) in lines.enumerate() {
        let ln = ln?;
        let parts: Vec<&str> = ln.split('\t').collect();
        if parts.len() <= ref_idx.max(dist_idx) {
            return Err(anyhow!("malformed line {}: {:?}", i + 2, ln));
        }
        pairs.push((parts[ref_idx].to_string(), parts[dist_idx].to_string()));
    }
    let total = pairs.len();
    eprintln!(
        "  read {} pairs in {:.2}s",
        total,
        t_start.elapsed().as_secs_f64()
    );

    // --- precompute unique ref index for the cache (when enabled)
    let (unique_refs, row_ref_idx): (Vec<String>, Vec<u32>) = if args.cache_refs {
        let mut seen: HashMap<String, u32> = HashMap::new();
        let mut unique_refs: Vec<String> = Vec::new();
        let mut row_ref_idx: Vec<u32> = Vec::with_capacity(total);
        for (r, _) in &pairs {
            if let Some(&idx) = seen.get(r) {
                row_ref_idx.push(idx);
            } else {
                let idx = unique_refs.len() as u32;
                seen.insert(r.clone(), idx);
                unique_refs.push(r.clone());
                row_ref_idx.push(idx);
            }
        }
        (unique_refs, row_ref_idx)
    } else {
        (Vec::new(), Vec::new())
    };

    if args.cache_refs {
        eprintln!("  unique refs: {} (caching enabled)", unique_refs.len());
    }

    // --- load each unique ref into RAM (parallel)
    let t_ref_load = Instant::now();
    let ref_cache: Vec<Option<(Vec<u8>, u32, u32)>> = if args.cache_refs {
        let progress = AtomicUsize::new(0);
        let n_refs = unique_refs.len();
        let cache: Vec<Option<(Vec<u8>, u32, u32)>> = unique_refs
            .par_iter()
            .map(|p| {
                let r = load_rgb8(Path::new(p)).ok();
                let n = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(50) || n == n_refs {
                    eprintln!("  ref-load: {}/{}", n, n_refs);
                }
                r
            })
            .collect();
        cache
    } else {
        Vec::new()
    };
    if args.cache_refs {
        eprintln!(
            "  ref-load: done in {:.2}s",
            t_ref_load.elapsed().as_secs_f64()
        );
    }

    // --- extract features per pair (parallel)
    let t_pair = Instant::now();
    let progress = AtomicUsize::new(0);
    let pair_features: Vec<Vec<f32>> = (0..total)
        .into_par_iter()
        .map(|i| {
            let (ref_path, dist_path) = &pairs[i];

            // load ref (from cache if enabled)
            let ref_loaded: Option<(Vec<u8>, u32, u32)> = if args.cache_refs {
                ref_cache[row_ref_idx[i] as usize].clone()
            } else {
                load_rgb8(Path::new(ref_path)).ok()
            };

            let result = match (ref_loaded, load_rgb8(Path::new(dist_path)).ok()) {
                (Some((rpx, rw, rh)), Some((dpx, dw, dh))) => {
                    if rw == dw && rh == dh && rpx.len() == dpx.len() {
                        extract_cvvdp_features(&rpx, &dpx, rw as usize, rh as usize)
                    } else {
                        eprintln!(
                            "  WARN: dim mismatch ref={}x{} dist={}x{} for row {}",
                            rw, rh, dw, dh, i
                        );
                        vec![f32::NAN; CVVDP_FEATURE_COUNT]
                    }
                }
                _ => {
                    eprintln!(
                        "  WARN: load failure for row {} (ref={:?} dist={:?})",
                        i, ref_path, dist_path
                    );
                    vec![f32::NAN; CVVDP_FEATURE_COUNT]
                }
            };
            let n = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(500) || n == total {
                eprintln!(
                    "  pair-features: {}/{} ({:.1}%) {:.1}s",
                    n,
                    total,
                    100.0 * n as f64 / total as f64,
                    t_pair.elapsed().as_secs_f64()
                );
            }
            result
        })
        .collect();
    eprintln!(
        "  pair-feature extract: {} rows in {:.2}s",
        total,
        t_pair.elapsed().as_secs_f64()
    );

    // --- write output parquet (f0..f<N-1>)
    let fields: Vec<Arc<Field>> = (0..CVVDP_FEATURE_COUNT)
        .map(|i| Arc::new(Field::new(format!("f{}", i), DataType::Float32, false)))
        .collect();
    let out_schema = Arc::new(Schema::new(fields));

    // transpose pair_features into per-column vectors
    let mut cols: Vec<Vec<f32>> = (0..CVVDP_FEATURE_COUNT)
        .map(|_| Vec::with_capacity(total))
        .collect();
    for row in &pair_features {
        for (i, &v) in row.iter().enumerate() {
            cols[i].push(v);
        }
    }

    let arrays: Vec<ArrayRef> = cols
        .iter()
        .map(|c| Arc::new(Float32Array::from_iter_values(c.iter().copied())) as ArrayRef)
        .collect();
    let batch = RecordBatch::try_new(out_schema.clone(), arrays)?;

    let out_file =
        File::create(&args.output).with_context(|| format!("creating {:?}", args.output))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(out_file, out_schema.clone(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    eprintln!(
        "  wrote {} rows × {} cols → {:?}",
        total, CVVDP_FEATURE_COUNT, args.output
    );

    // --- sanity stats
    eprintln!("\n=== per-pair feature distribution sanity ===");
    for (i, col) in cols.iter().enumerate() {
        let n_nan = col.iter().filter(|x| x.is_nan()).count();
        let valid: Vec<f32> = col.iter().copied().filter(|x| x.is_finite()).collect();
        if valid.is_empty() {
            eprintln!("  f{}: ALL NaN", i);
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
        eprintln!(
            "  f{:2}: min={:.4} max={:.4} mean={:.4} std={:.4} nan={}",
            i, min, max, mean, std, n_nan
        );
    }

    eprintln!("  total wall time: {:.2}s", t_start.elapsed().as_secs_f64());
    Ok(())
}

fn load_rgb8(p: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(p).with_context(|| format!("decoding {:?}", p))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), w, h))
}
