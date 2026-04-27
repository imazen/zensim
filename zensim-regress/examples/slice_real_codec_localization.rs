//! Real-codec validation of Option D: does per-window canonical-via-slice
//! correctly localize spatially-varying damage that a real codec produces?
//!
//! Methodology: for each (source, decoded) pair from the real-codec corpus:
//!   1. Compute full-image diffmap via Zensim::compute_with_ref_and_diffmap.
//!      The full-pyramid diffmap is the **ground truth** for "where is the
//!      damage spatially" — it sees full context at all 4 scales, no
//!      truncation, no edge-padding.
//!   2. Integrate diffmap intensity over each K=64-row window → per-window
//!      ground-truth damage.
//!   3. Compute per-window canonical-via-slice scores (Option D).
//!   4. Rank windows by each. Spearman SROCC tells us how well Option D's
//!      ranking matches ground truth.
//!
//! The controller targets AQ at the most-damaged windows. If Option D's
//! ranking matches ground truth, the controller targets correctly. If not,
//! the controller is steered by edge-pad-noise into the wrong windows.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;
use zensim::{
    DiffmapOptions, DiffmapWeighting, PrecomputedReference, RgbSlice, Zensim, ZensimProfile,
    ZensimScratch, profile::WEIGHTS_PREVIEW_V0_1,
};

const CSV_PATH: &str = "/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv";
const SCALE0_BASIC_LEN: usize = 13 * 3;
const WINDOW_ROWS: usize = 64;

/// Read up to `n_per_codec` (source, decoded, q) triples per (codec, q-band)
/// for the codecs we care about, q60-q90.
fn read_pairs(n_per_cell: usize) -> Vec<(String, String, String, u32)> {
    let codecs = [
        "mozjpeg-rs-420-e4-v0.5.4",
        "zenjpeg-420-e2-v0.3.1",
        "zenjpeg-420-xyb-e2-v0.3.1",
        "zenavif-s5-e6",
    ];
    let bands = [(60u32, 70u32), (70, 80), (80, 90)];
    let file = File::open(CSV_PATH).expect("ledger csv");
    let mut by_cell: BTreeMap<(String, (u32, u32)), Vec<(String, String, u32)>> = BTreeMap::new();

    for (i, line) in BufReader::new(file).lines().enumerate() {
        if i == 0 { continue; }
        let line = line.expect("read");
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 13 { continue; }
        let codec = cols[2];
        if !codecs.contains(&codec) { continue; }
        let q: u32 = cols[3].parse().unwrap_or(0);
        let Some(band) = bands.iter().find(|(lo, hi)| q >= *lo && q < *hi) else { continue };
        let key = (codec.to_string(), *band);
        let entry = by_cell.entry(key).or_default();
        if entry.len() >= n_per_cell { continue; }
        entry.push((cols[0].to_string(), cols[1].to_string(), q));
    }

    let mut out = vec![];
    for ((codec, _), v) in by_cell {
        for (s, d, q) in v {
            out.push((codec.clone(), s, d, q));
        }
    }
    out
}

fn load_rgb(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    Some((img.into_raw(), w, h))
}

fn rgb_to_pixels(buf: &[u8]) -> Vec<[u8; 3]> {
    buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn scale0_proxy(features: &[f64]) -> f64 {
    features.iter().take(SCALE0_BASIC_LEN)
        .zip(WEIGHTS_PREVIEW_V0_1.iter().take(SCALE0_BASIC_LEN))
        .map(|(f, w)| f * w)
        .sum()
}

/// Spearman rank correlation
fn srocc(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 { return f64::NAN; }
    let mut rx: Vec<usize> = (0..n).collect();
    rx.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks_x = vec![0.0; n];
    for (i, &idx) in rx.iter().enumerate() { ranks_x[idx] = i as f64; }
    let mut ry: Vec<usize> = (0..n).collect();
    ry.sort_by(|&a, &b| ys[a].partial_cmp(&ys[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks_y = vec![0.0; n];
    for (i, &idx) in ry.iter().enumerate() { ranks_y[idx] = i as f64; }
    let mean_x = ranks_x.iter().sum::<f64>() / n as f64;
    let mean_y = ranks_y.iter().sum::<f64>() / n as f64;
    let mut num = 0.0; let mut dx = 0.0; let mut dy = 0.0;
    for i in 0..n {
        let a = ranks_x[i] - mean_x;
        let b = ranks_y[i] - mean_y;
        num += a * b; dx += a * a; dy += b * b;
    }
    if dx == 0.0 || dy == 0.0 { return f64::NAN; }
    num / (dx * dy).sqrt()
}

fn main() {
    let pairs = read_pairs(8); // 8 per (codec × band) ≈ ~80 total
    eprintln!("Loaded {} pairs across codec×band stratification", pairs.len());

    let z = Zensim::new(ZensimProfile::latest());
    let mut scratch = ZensimScratch::new();

    println!(
        "codec,quality,source,n_windows,top1_match,top3_overlap,\
         srocc_canon_vs_diffmap,srocc_proxy_vs_diffmap,srocc_canon_vs_proxy"
    );

    let mut summary_canon = vec![];
    let mut summary_proxy = vec![];
    let mut top1_canon_correct = 0;
    let mut top1_proxy_correct = 0;
    let mut top3_canon_overlap_total = 0;
    let mut top3_proxy_overlap_total = 0;
    let mut total = 0;

    let t0 = Instant::now();

    for (codec, src_path, dec_path, q) in &pairs {
        let Some((src_raw, sw, sh)) = load_rgb(src_path) else {
            eprintln!("skip src {}", src_path); continue;
        };
        let Some((dst_raw, dw, dh)) = load_rgb(dec_path) else {
            eprintln!("skip dst {}", dec_path); continue;
        };
        if sw != dw || sh != dh { continue; }

        let width = sw as usize;
        let height = sh as usize;
        let src_px = rgb_to_pixels(&src_raw);
        let dst_px = rgb_to_pixels(&dst_raw);

        let src_slice = RgbSlice::new(&src_px, width, height);
        let dst_slice = RgbSlice::new(&dst_px, width, height);

        // Ground truth: full-image diffmap with TRAINED weighting (matches what
        // the canonical metric weighs as damage, all 4 scales fully resolved).
        let pre_full = z.precompute_reference(&src_slice).unwrap();
        let opts = DiffmapOptions::from(DiffmapWeighting::Trained);
        let dm = z.compute_with_ref_and_diffmap(&pre_full, &dst_slice, opts).unwrap();
        let diffmap = dm.diffmap();
        // diffmap is width × height row-major; integrate per K-row window.
        let mut window_y = 0;
        let mut gt_per_window = Vec::new();
        let mut win_y_list = Vec::new();
        let mut win_h_list = Vec::new();
        while window_y < height {
            let h = (height - window_y).min(WINDOW_ROWS);
            if h < 8 { break; }
            let mut sum = 0.0f64;
            for y in window_y..(window_y + h) {
                for x in 0..width {
                    sum += diffmap[y * width + x] as f64;
                }
            }
            // Mean diffmap intensity per pixel
            gt_per_window.push(sum / (h * width) as f64);
            win_y_list.push(window_y);
            win_h_list.push(h);
            window_y += WINDOW_ROWS;
        }
        let n_windows = gt_per_window.len();

        // Per-window canonical (Option D) and per-window scale-0 proxy.
        let mut window_canon_neg = Vec::with_capacity(n_windows);
        let mut window_proxy = Vec::with_capacity(n_windows);
        for (idx, &y0) in win_y_list.iter().enumerate() {
            let h = win_h_list[idx];
            let start = y0 * width;
            let end = start + h * width;
            let win_src_slice = RgbSlice::new(&src_px[start..end], width, h);
            let win_dst_slice = RgbSlice::new(&dst_px[start..end], width, h);
            let pre_win = z.precompute_reference(&win_src_slice).unwrap();
            let r = z.compute_with_ref_into(&pre_win, &win_dst_slice, &mut scratch).unwrap();
            // Canonical score: HIGHER = better quality. Damage = -score so it
            // sorts the same direction as diffmap (higher = more damaged).
            window_canon_neg.push(-r.score());
            // Scale-0 proxy: HIGHER = more damaged (raw_distance-style).
            window_proxy.push(scale0_proxy(r.features()));
        }

        // Rankings
        let s_canon_dm = srocc(&window_canon_neg, &gt_per_window);
        let s_proxy_dm = srocc(&window_proxy, &gt_per_window);
        let s_canon_proxy = srocc(&window_canon_neg, &window_proxy);

        // Top-1: did each candidate's worst-scoring window match diffmap's
        // worst window?
        let argmax = |v: &[f64]| {
            v.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i)
        };
        let gt_top1 = argmax(&gt_per_window);
        let canon_top1 = argmax(&window_canon_neg);
        let proxy_top1 = argmax(&window_proxy);
        let canon_match = canon_top1 == gt_top1;
        let proxy_match = proxy_top1 == gt_top1;

        // Top-3: how many of canonical's top-3 windows match diffmap's top-3?
        let topk = |v: &[f64], k: usize| {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
            idx.truncate(k);
            idx
        };
        let gt_top3: std::collections::HashSet<_> = topk(&gt_per_window, 3).into_iter().collect();
        let canon_top3: std::collections::HashSet<_> = topk(&window_canon_neg, 3).into_iter().collect();
        let proxy_top3: std::collections::HashSet<_> = topk(&window_proxy, 3).into_iter().collect();
        let canon_overlap = canon_top3.intersection(&gt_top3).count();
        let proxy_overlap = proxy_top3.intersection(&gt_top3).count();

        let stem = PathBuf::from(src_path).file_stem().unwrap().to_string_lossy().into_owned();
        println!("{},{},{},{},{},{},{:.4},{:.4},{:.4}",
            codec, q, stem, n_windows,
            canon_match as u32, canon_overlap, s_canon_dm, s_proxy_dm, s_canon_proxy);

        if !s_canon_dm.is_nan() { summary_canon.push(s_canon_dm); }
        if !s_proxy_dm.is_nan() { summary_proxy.push(s_proxy_dm); }
        if canon_match { top1_canon_correct += 1; }
        if proxy_match { top1_proxy_correct += 1; }
        top3_canon_overlap_total += canon_overlap;
        top3_proxy_overlap_total += proxy_overlap;
        total += 1;
    }

    let dt = t0.elapsed();
    summary_canon.sort_by(|a, b| a.partial_cmp(b).unwrap());
    summary_proxy.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    let median = |v: &[f64]| if v.is_empty() { 0.0 } else { v[v.len()/2] };
    let p10 = |v: &[f64]| if v.is_empty() { 0.0 } else { v[v.len()/10] };

    eprintln!("\n# Per-pair SROCC vs ground-truth diffmap-aggregated damage (n={})", total);
    eprintln!("# Option D (per-window canonical):  mean={:.4}  median={:.4}  p10={:.4}",
        mean(&summary_canon), median(&summary_canon), p10(&summary_canon));
    eprintln!("# Scale-0 proxy:                     mean={:.4}  median={:.4}  p10={:.4}",
        mean(&summary_proxy), median(&summary_proxy), p10(&summary_proxy));
    eprintln!("\n# Top-1 spatial match (worst-scoring window matches ground-truth worst):");
    eprintln!("#   canonical: {}/{} = {:.0}%", top1_canon_correct, total,
        top1_canon_correct as f64 / total as f64 * 100.0);
    eprintln!("#   scale-0:   {}/{} = {:.0}%", top1_proxy_correct, total,
        top1_proxy_correct as f64 / total as f64 * 100.0);
    eprintln!("\n# Top-3 overlap (out of 3, average):");
    eprintln!("#   canonical: {:.2}/3", top3_canon_overlap_total as f64 / total as f64);
    eprintln!("#   scale-0:   {:.2}/3", top3_proxy_overlap_total as f64 / total as f64);
    eprintln!("\nDone in {:.1}s", dt.as_secs_f64());
}
