//! Stage-1 perceptual-hash overlap detector for the synth-corpus
//! holdout audit (Goal 5 of `docs/PARITY_AND_METHODOLOGY_PLAN_2026-05-11.md`).
//!
//! Compares the **49 CID22 validation reference images** against every
//! distinct **training source** found in a given CSV. Reports the
//! minimum Hamming distance between each training source's dHash and
//! the nearest CID22 reference's dHash. Anything below the threshold
//! is flagged as a potential leak.
//!
//! Algorithm: standard dHash-64 (resize-to-9x8 grayscale, compare
//! adjacent horizontal pixel pairs, set bit if left > right). Robust
//! to small color/brightness shifts and codec compression. Catches
//! exact and resized matches. Does NOT catch cropped variants —
//! that's stage 2 (sliding-window) which lands in a follow-up binary.
//!
//! Usage:
//! ```text
//! cargo run --release --bin check_holdout_overlap -- \
//!   --cid22-refs /mnt/v/dataset/cid22/CID22_validation_set/original \
//!   --training-csv /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv \
//!   --threshold 10 \
//!   --out-tsv benchmarks/holdout_overlap_2026-05-11_stage1.tsv
//! ```

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(version, about = "Stage-1 dHash-64 overlap detector")]
struct Args {
    /// Directory of HOLDOUT reference images to protect. Named for its
    /// original use (the 49 CID22 validation references) and aliased
    /// `--holdout-refs` for any other holdout set — the audit is the same
    /// one either way, and other eval corpora need protecting too
    /// (KADID / TID / CSIQ / LIVE / KonJND references, 2026-09-01).
    #[arg(long, alias = "holdout-refs")]
    cid22_refs: PathBuf,

    /// Path to a training CSV whose first column lists source images
    /// (e.g. training_safe_synthetic.csv).
    #[arg(long)]
    training_csv: PathBuf,

    /// Hamming-distance threshold below (or equal to) which a match
    /// is flagged as a likely leak. Standard dHash literature uses 10
    /// for "very likely the same image" and 16 for "possibly the same"
    /// — we use 10 as the strict default.
    #[arg(long, default_value_t = 10)]
    threshold: u32,

    /// Where to write the per-training-source TSV report (one row
    /// per training source).
    #[arg(long)]
    out_tsv: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1. Enumerate CID22 references.
    let cid22_paths: Vec<PathBuf> = walk_image_dir(&args.cid22_refs)?
        .into_iter()
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|e| {
                    matches!(
                        e.to_ascii_lowercase().as_str(),
                        "png" | "jpg" | "jpeg" | "bmp" | "tif" | "tiff" | "webp"
                    )
                })
                .unwrap_or(false)
        })
        .collect();
    eprintln!("CID22 refs: {} files", cid22_paths.len());
    if cid22_paths.len() != 49 {
        eprintln!(
            "NOTE: {} reference images (49 = the CID22 validation set; any other \
             count means a different holdout set — sanity-check the path)",
            cid22_paths.len()
        );
    }

    // 2. Compute dHash for each CID22 ref. Tag with filename for
    //    reporting.
    let cid22_hashes: Vec<(String, u64)> = cid22_paths
        .par_iter()
        .filter_map(|p| match dhash_64_path(p) {
            Ok(h) => Some((
                p.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string(),
                h,
            )),
            Err(e) => {
                eprintln!("WARN: cid22 ref {}: {}", p.display(), e);
                None
            }
        })
        .collect();
    eprintln!("CID22 refs hashed: {}", cid22_hashes.len());

    // 3. Extract distinct source paths from the training CSV (column 0).
    let mut distinct_sources: BTreeSet<String> = BTreeSet::new();
    let f = File::open(&args.training_csv)
        .with_context(|| format!("opening {}", args.training_csv.display()))?;
    let r = BufReader::new(f);
    for (i, line) in r.lines().enumerate() {
        let line = line?;
        if i == 0 {
            // header
            continue;
        }
        if let Some(first_col) = line.split(',').next()
            && !first_col.is_empty()
        {
            distinct_sources.insert(first_col.to_string());
        }
    }
    let sources: Vec<String> = distinct_sources.into_iter().collect();
    eprintln!("training sources (distinct): {}", sources.len());

    // 4. Compute dHash for each training source.
    let total = sources.len();
    let chunk_progress = (total / 20).max(1);
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let hashes: Vec<(String, Option<u64>)> = sources
        .par_iter()
        .map(|path| {
            let n = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n.is_multiple_of(chunk_progress) {
                eprintln!("  hashing source {} / {}", n, total);
            }
            let h = match dhash_64_path(Path::new(path)) {
                Ok(h) => Some(h),
                Err(e) => {
                    eprintln!("WARN: source {}: {}", path, e);
                    None
                }
            };
            (path.clone(), h)
        })
        .collect();
    eprintln!(
        "training sources hashed: {} ({} failed)",
        hashes.iter().filter(|(_, h)| h.is_some()).count(),
        hashes.iter().filter(|(_, h)| h.is_none()).count()
    );

    // 5. For each training source, find the nearest CID22 ref.
    let out = File::create(&args.out_tsv)
        .with_context(|| format!("creating {}", args.out_tsv.display()))?;
    let mut w = BufWriter::new(out);
    writeln!(
        w,
        "training_source\tdhash\tnearest_cid22_ref\tnearest_dhash\thamming"
    )?;
    let mut flagged: Vec<(String, String, u32)> = Vec::new();
    let mut hist = [0u64; 65];
    for (path, hash_opt) in &hashes {
        let Some(h) = hash_opt else {
            writeln!(w, "{}\t\t\t\t", path)?;
            continue;
        };
        let (best_name, best_h, best_dist) = cid22_hashes
            .iter()
            .map(|(name, ch)| (name.as_str(), *ch, (ch ^ h).count_ones()))
            .min_by_key(|(_, _, d)| *d)
            .ok_or_else(|| anyhow!("no CID22 hashes computed"))?;
        writeln!(
            w,
            "{}\t{:016x}\t{}\t{:016x}\t{}",
            path, h, best_name, best_h, best_dist
        )?;
        if (best_dist as usize) < hist.len() {
            hist[best_dist as usize] += 1;
        }
        if best_dist <= args.threshold {
            flagged.push((path.clone(), best_name.to_string(), best_dist));
        }
    }
    w.flush()?;

    // 6. Print summary to stderr.
    eprintln!();
    eprintln!("=== Hamming distribution (training source ↔ nearest CID22 ref) ===");
    for (d, count) in hist.iter().enumerate() {
        if *count > 0 {
            eprintln!("  d={:>2}  n={}", d, count);
        }
    }
    eprintln!();
    eprintln!("=== FLAGGED (Hamming <= {}) ===", args.threshold);
    if flagged.is_empty() {
        eprintln!("  (none — clean holdout)");
    } else {
        for (src, ref_name, dist) in &flagged {
            eprintln!("  d={} src={} cid22_ref={}", dist, src, ref_name);
        }
    }
    eprintln!();
    eprintln!("Wrote {}", args.out_tsv.display());
    Ok(())
}

/// dHash-64 of an image file (decode + the ONE shared primitive,
/// `zensim_validate::content_clusters::dhash_64`). Two hashes with
/// Hamming distance ≤ 10 are very likely the same image.
fn dhash_64_path(path: &Path) -> Result<u64> {
    let img = image::open(path).with_context(|| format!("decoding {}", path.display()))?;
    Ok(zensim_validate::content_clusters::dhash_64(&img))
}

/// One-level shallow directory listing (no recursion).
fn walk_image_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let e = entry?;
        let p = e.path();
        if p.is_file() {
            paths.push(p);
        }
    }
    paths.sort();
    Ok(paths)
}
