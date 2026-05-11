//! Stage-2 sliding-window perceptual-hash overlap detector (Goal 5).
//!
//! Catches **cropped-variant** leaks that stage 1 misses: a training
//! source that's much larger than a CID22 ref but contains a region
//! near-identical to the ref. Algorithm:
//!
//! 1. dHash-64 each CID22 reference (assumed 512×512 or similar aspect).
//! 2. For each training source, slide an aspect-matched window over it
//!    at multiple scales; dHash each window; find the minimum Hamming
//!    distance to any CID22 ref.
//! 3. Anything below the threshold is a stage-2 hit (cropped overlap).
//!
//! Sliding strategy:
//! - Take the smallest dimension `s = min(W_src, H_src)` as the largest
//!   square window. Generate windows at sizes `s`, `s/2`, `s/4`
//!   (subject to a 96px floor — too small and dHash is meaningless).
//! - Stride = window_size / 4 (so ~16 windows per source for s=512).
//! - All windows are aspect-1:1, matching the typical CID22 512×512.
//!
//! Stage-1 hits at d≤16 are already known; this stage finds **additional**
//! leaks where the whole-image dHash distance was >16 but a sub-region
//! matches at ≤16.

use anyhow::{Context, Result};
use clap::Parser;
use image::{DynamicImage, GenericImageView};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(version, about = "Stage-2 sliding-window cropped-variant overlap detector")]
struct Args {
    /// Directory containing CID22 validation references.
    #[arg(long)]
    cid22_refs: PathBuf,

    /// Training CSV whose first column lists source images.
    #[arg(long)]
    training_csv: PathBuf,

    /// Hamming-distance threshold (best-window). 10 = very likely the
    /// same content; 16 = possibly the same. Default 10 (strict).
    #[arg(long, default_value_t = 10)]
    threshold: u32,

    /// Minimum window edge in pixels. Smaller windows hash unreliably.
    #[arg(long, default_value_t = 96)]
    min_window: u32,

    /// Where to write the per-training-source TSV report.
    #[arg(long)]
    out_tsv: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1. Hash CID22 refs (whole-image dHash; the refs are 512×512).
    let cid22_paths = walk_image_dir(&args.cid22_refs)?;
    let cid22_hashes: Vec<(String, u64)> = cid22_paths
        .par_iter()
        .filter_map(|p| {
            let img = image::open(p).ok()?;
            Some((
                p.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string(),
                dhash_64(&img),
            ))
        })
        .collect();
    eprintln!("CID22 refs hashed: {}", cid22_hashes.len());

    // 2. Distinct training sources.
    let mut distinct_sources: BTreeSet<String> = BTreeSet::new();
    let f = File::open(&args.training_csv)
        .with_context(|| format!("opening {}", args.training_csv.display()))?;
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue;
        }
        if let Some(first) = line.split(',').next() {
            if !first.is_empty() {
                distinct_sources.insert(first.to_string());
            }
        }
    }
    let sources: Vec<String> = distinct_sources.into_iter().collect();
    eprintln!("training sources: {}", sources.len());

    // 3. For each training source, slide windows and find min-distance ref.
    let total = sources.len();
    let chunk = (total / 20).max(1);
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<(String, Option<(String, u32, u32, u32, u32, u32)>)> = sources
        .par_iter()
        .map(|src_path| {
            let n = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n % chunk == 0 {
                eprintln!("  scanning {} / {}", n, total);
            }
            let img = match image::open(Path::new(src_path)) {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("WARN: {}: {}", src_path, e);
                    return (src_path.clone(), None);
                }
            };
            let best = scan_windows(&img, &cid22_hashes, args.min_window);
            (src_path.clone(), best)
        })
        .collect();

    // 4. Write TSV and summarize.
    let out = File::create(&args.out_tsv)
        .with_context(|| format!("creating {}", args.out_tsv.display()))?;
    let mut w = BufWriter::new(out);
    writeln!(
        w,
        "training_source\tnearest_cid22_ref\tbest_window_hamming\tbest_window_x\tbest_window_y\tbest_window_size"
    )?;
    let mut hist = [0u64; 65];
    let mut flagged: Vec<(String, String, u32, u32, u32, u32)> = Vec::new();
    for (src, best) in &results {
        match best {
            Some((ref_name, dist, x, y, ws, _)) => {
                writeln!(w, "{}\t{}\t{}\t{}\t{}\t{}", src, ref_name, dist, x, y, ws)?;
                if (*dist as usize) < hist.len() {
                    hist[*dist as usize] += 1;
                }
                if *dist <= args.threshold {
                    flagged.push((src.clone(), ref_name.clone(), *dist, *x, *y, *ws));
                }
            }
            None => {
                writeln!(w, "{}\t\t\t\t\t", src)?;
            }
        }
    }
    w.flush()?;

    eprintln!();
    eprintln!("=== Best-window Hamming distribution ===");
    for (d, count) in hist.iter().enumerate() {
        if *count > 0 {
            eprintln!("  d={:>2}  n={}", d, count);
        }
    }
    eprintln!();
    eprintln!("=== STAGE-2 FLAGGED (best-window Hamming <= {}) ===", args.threshold);
    if flagged.is_empty() {
        eprintln!("  (none)");
    } else {
        for (src, ref_name, dist, x, y, ws) in &flagged {
            eprintln!(
                "  d={} src={} ref={} window=(x={}, y={}, size={})",
                dist, src, ref_name, x, y, ws
            );
        }
    }
    eprintln!();
    eprintln!("Wrote {}", args.out_tsv.display());
    Ok(())
}

/// dHash-64 of a `DynamicImage`. Resize to 9×8 grayscale Lanczos3, set
/// bit per row-adjacent pixel pair if `left > right`.
fn dhash_64(img: &DynamicImage) -> u64 {
    let small = image::imageops::resize(
        &img.to_luma8(),
        9,
        8,
        image::imageops::FilterType::Lanczos3,
    );
    let mut hash = 0u64;
    let mut bit = 0u32;
    for y in 0..8 {
        for x in 0..8 {
            let left = small.get_pixel(x, y).0[0];
            let right = small.get_pixel(x + 1, y).0[0];
            if left > right {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }
    hash
}

/// Slide aspect-1:1 windows of decreasing size over `img`; dHash each
/// window; return the (ref_name, hamming, x, y, window_size, w_count)
/// of the best match. Window sizes: `s`, `s/2`, `s/4`, …, capped at
/// `min_window`. Stride per scale = window_size / 4.
fn scan_windows(
    img: &DynamicImage,
    cid22: &[(String, u64)],
    min_window: u32,
) -> Option<(String, u32, u32, u32, u32, u32)> {
    let (w, h) = img.dimensions();
    let max_window = w.min(h);
    if max_window < min_window {
        // Tiny image: just hash the whole thing.
        let h64 = dhash_64(img);
        return cid22
            .iter()
            .map(|(n, ch)| (n.clone(), (h64 ^ ch).count_ones(), 0u32, 0u32, max_window))
            .min_by_key(|(_, d, _, _, _)| *d)
            .map(|(n, d, x, y, s)| (n, d, x, y, s, 1));
    }
    let mut best: Option<(String, u32, u32, u32, u32, u32)> = None;
    let mut window_count = 0u32;
    let mut window_size = max_window;
    while window_size >= min_window {
        let stride = (window_size / 4).max(1);
        let mut x = 0u32;
        while x + window_size <= w {
            let mut y = 0u32;
            while y + window_size <= h {
                let crop = img.crop_imm(x, y, window_size, window_size);
                let ch = dhash_64(&crop);
                window_count += 1;
                for (name, rh) in cid22 {
                    let d = (ch ^ rh).count_ones();
                    if best.as_ref().map(|b| d < b.1).unwrap_or(true) {
                        best = Some((name.clone(), d, x, y, window_size, window_count));
                    }
                }
                y += stride;
            }
            x += stride;
        }
        // Avoid infinite loop at small sizes.
        let next = window_size / 2;
        if next < min_window || next == window_size {
            break;
        }
        window_size = next;
    }
    best
}

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
