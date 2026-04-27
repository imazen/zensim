//! Compare three signal candidates for zenjpeg #113's per-strip AQ controller:
//!
//!   (1) Scale-0-only proxy: dot(features[0..39], WEIGHTS_PREVIEW_V0_1[0..39])
//!       on the full (source, decoded) pair.
//!   (2) Per-window scale-0 proxy: same proxy but computed independently
//!       per K-row window of the image.
//!   (3) Per-window CANONICAL (multi-scale, all features): full
//!       Zensim::compute_with_ref_into on each window slice, reusing
//!       ZensimScratch across windows.
//!
//! For each candidate we compute (a) summed-windows-vs-canonical-full SROCC
//! across (image, q) pairs, and (b) the per-window noise floor — variance
//! of per-window scores within a single image — which determines the
//! signal-to-noise ratio the controller actually has to work with.
//!
//! The headline question: at small q-deltas (the controller's actual operating
//! regime, not the q-sweep we measured before), does per-window CANONICAL
//! resolve quality differences that per-window scale-0 cannot?

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;
use zensim::{
    PrecomputedReference, RgbSlice, Zensim, ZensimProfile, ZensimScratch,
    profile::WEIGHTS_PREVIEW_V0_1,
};

const CSV_PATH: &str = "/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv";
const SCALE0_BASIC_LEN: usize = 13 * 3; // 39
/// Window height in rows. Matches zenjpeg #113's K=4 iMCU at 4:2:0 (64 rows).
const WINDOW_ROWS: usize = 64;

#[derive(Clone, Copy)]
struct PairRef<'a> {
    source: &'a str,
    decoded: &'a str,
    quality: u32,
}

fn read_pairs() -> Vec<(String, String, u32)> {
    let file = File::open(CSV_PATH).expect("ledger csv");
    let mut out = vec![];
    for (i, line) in BufReader::new(file).lines().enumerate() {
        if i == 0 {
            continue;
        }
        let line = line.expect("read line");
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 13 {
            continue;
        }
        // Filter to a manageable corpus size — small images, a handful of codecs,
        // a tight q-band that's near the controller's actual operating point.
        let codec = cols[2];
        let q: u32 = cols[3].parse().unwrap_or(0);
        if codec != "zenjpeg-420-e2-v0.3.1" {
            continue;
        }
        // Focus on the q-band where target-zq actually operates (q60–q90).
        if !(60..=90).contains(&q) {
            continue;
        }
        out.push((cols[0].to_string(), cols[1].to_string(), q));
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

/// Slice an RGB plane into row-windows. Returns Vec of (offset_y, rows, slice).
fn windows<'a>(buf: &'a [[u8; 3]], width: usize, height: usize, k: usize) -> Vec<(usize, usize, &'a [[u8; 3]])> {
    let mut out = vec![];
    let mut y = 0;
    while y < height {
        let h = (height - y).min(k);
        // Skip windows too small for zensim's 8x8 minimum
        if h < 8 || width < 8 {
            break;
        }
        let start = y * width;
        let end = start + h * width;
        out.push((y, h, &buf[start..end]));
        y += k;
    }
    out
}

fn scale0_proxy(features: &[f64]) -> f64 {
    features
        .iter()
        .take(SCALE0_BASIC_LEN)
        .zip(WEIGHTS_PREVIEW_V0_1.iter().take(SCALE0_BASIC_LEN))
        .map(|(f, w)| f * w)
        .sum()
}

/// Spearman rank correlation
fn srocc(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut rx: Vec<usize> = (0..n).collect();
    rx.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks_x = vec![0.0; n];
    for (i, &idx) in rx.iter().enumerate() {
        ranks_x[idx] = i as f64;
    }
    let mut ry: Vec<usize> = (0..n).collect();
    ry.sort_by(|&a, &b| ys[a].partial_cmp(&ys[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks_y = vec![0.0; n];
    for (i, &idx) in ry.iter().enumerate() {
        ranks_y[idx] = i as f64;
    }
    let mean_x = ranks_x.iter().sum::<f64>() / n as f64;
    let mean_y = ranks_y.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..n {
        let dx = ranks_x[i] - mean_x;
        let dy = ranks_y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    if den_x == 0.0 || den_y == 0.0 {
        return f64::NAN;
    }
    num / (den_x * den_y).sqrt()
}

fn main() {
    let pairs = read_pairs();
    eprintln!("ledger pairs in q60-q90 band: {}", pairs.len());

    // Group by source
    let mut by_source: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
    for (s, d, q) in &pairs {
        by_source.entry(s.clone()).or_default().push((d.clone(), *q));
    }
    // Pick first 30 sources with ≥4 q-levels for intra-image controller-deltas.
    let mut sources: Vec<_> = by_source
        .iter()
        .filter(|(_, vs)| vs.len() >= 4)
        .take(30)
        .collect();
    sources.sort_by_key(|(s, _)| s.clone());
    eprintln!("sources with ≥4 q-levels in band: {}", sources.len());

    let z = Zensim::new(ZensimProfile::latest());

    println!(
        "source,quality,full_canonical_score,full_canonical_raw_dist,\
         full_scale0_proxy,sum_windows_canonical,sum_windows_scale0,\
         var_windows_canonical,var_windows_scale0,n_windows"
    );

    let t0 = Instant::now();
    for (src_path, distorted_list) in &sources {
        // Load source once
        let Some((src_raw, sw, sh)) = load_rgb(src_path) else {
            eprintln!("skip source {}", src_path);
            continue;
        };
        let src_px = rgb_to_pixels(&src_raw);
        let src_slice = RgbSlice::new(&src_px, sw as usize, sh as usize);

        // Full-image precomputed reference (for full canonical + per-window scale-0)
        let pre_full = z.precompute_reference(&src_slice).unwrap();

        // Per-window precomputed references (for per-window canonical)
        let win_specs = windows(&src_px, sw as usize, sh as usize, WINDOW_ROWS);
        let mut window_refs: Vec<PrecomputedReference> = Vec::with_capacity(win_specs.len());
        for &(_, h, slice) in &win_specs {
            let ws = RgbSlice::new(slice, sw as usize, h);
            window_refs.push(z.precompute_reference(&ws).unwrap());
        }

        let mut scratch = ZensimScratch::new();

        for (decoded_path, q) in distorted_list.iter() {
            let Some((dst_raw, dw, dh)) = load_rgb(decoded_path) else {
                continue;
            };
            if dw != sw || dh != sh {
                continue;
            }
            let dst_px = rgb_to_pixels(&dst_raw);
            let dst_slice = RgbSlice::new(&dst_px, sw as usize, sh as usize);

            // (1) full canonical
            let full = z.compute(&src_slice, &dst_slice).unwrap();
            let full_score = full.score();
            let full_raw = full.raw_distance();

            // (2) full scale-0 proxy
            let full_features = full.features();
            let full_proxy = scale0_proxy(full_features);

            // Per-window: canonical + scale-0
            let dst_windows = windows(&dst_px, sw as usize, sh as usize, WINDOW_ROWS);
            let mut win_canon = Vec::with_capacity(dst_windows.len());
            let mut win_scale0 = Vec::with_capacity(dst_windows.len());

            for ((_, h, dst_w), pre_w) in dst_windows.iter().zip(window_refs.iter()) {
                let ws = RgbSlice::new(*dst_w, sw as usize, *h);
                let res = z.compute_with_ref_into(pre_w, &ws, &mut scratch).unwrap();
                win_canon.push(res.score() as f64);
                win_scale0.push(scale0_proxy(res.features()));
            }

            let n = win_canon.len() as f64;
            let mean_c: f64 = win_canon.iter().sum::<f64>() / n;
            let mean_s: f64 = win_scale0.iter().sum::<f64>() / n;
            let var_c: f64 =
                win_canon.iter().map(|x| (x - mean_c).powi(2)).sum::<f64>() / n;
            let var_s: f64 =
                win_scale0.iter().map(|x| (x - mean_s).powi(2)).sum::<f64>() / n;

            println!(
                "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
                Path::new(src_path).file_stem().unwrap().to_string_lossy(),
                q,
                full_score,
                full_raw,
                full_proxy,
                win_canon.iter().sum::<f64>(),
                win_scale0.iter().sum::<f64>(),
                var_c,
                var_s,
                win_canon.len(),
            );
        }
    }
    let dt = t0.elapsed();
    eprintln!("\ndone in {:.1}s", dt.as_secs_f64());
}
