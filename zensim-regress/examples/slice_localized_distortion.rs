//! Does Option D's per-window canonical *actually* localize spatial damage?
//!
//! Open question from issue #16: scale-2/3 features in a 64-row window have
//! ≤6 valid SSIM rows after the 5-row blur radius, so per-window canonical
//! is dominated by mirror-pad edge artifacts at the trained weight mass
//! (~94%) above scale 0. This experiment checks whether the per-window
//! canonical signal still correctly identifies which window is damaged
//! when distortion is injected into ONE specific window of a clean image.
//!
//! For each test:
//!   1. Load a clean source image, copy it as a clean "distorted" baseline.
//!   2. Inject heavy quantization-style noise into ONE specific row-band of
//!      the "distorted" copy (the target window).
//!   3. Compute per-window canonical scores for ALL windows.
//!   4. Compute per-window scale-0 proxy for ALL windows.
//!   5. Report: did the target window score worst on each signal?
//!
//! If per-window canonical correctly flags the target window even though
//! ~94% of its weight mass is edge-pad noise at scales 2-3, Option D works.
//! If it gets fooled by the pad noise into flagging a different window,
//! Option D is broken for the controller's spatial-targeting use case.

use std::path::PathBuf;
use zensim::{
    PrecomputedReference, RgbSlice, Zensim, ZensimProfile, ZensimScratch,
    profile::WEIGHTS_PREVIEW_V0_1,
};

const SCALE0_BASIC_LEN: usize = 13 * 3;
const WINDOW_ROWS: usize = 64;

fn load_rgb(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    Some((img.into_raw(), w, h))
}

fn rgb_to_pixels(buf: &[u8]) -> Vec<[u8; 3]> {
    buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Inject heavy block-quantization-style noise into rows [y0, y1) only.
/// The rest of the image is left identical to the source.
fn inject_distortion_in_band(
    src: &[[u8; 3]],
    width: usize,
    y0: usize,
    y1: usize,
    step: u8,
) -> Vec<[u8; 3]> {
    let mut out = src.to_vec();
    let step = step.max(1);
    let s = step as i32;
    for y in y0..y1 {
        for x in 0..width {
            let i = y * width + x;
            // Block-quantize each channel to multiples of `step`. Add a
            // mild per-block offset to avoid degenerate cases.
            let off = ((x / 8) ^ (y / 8)) as i32 % 11;
            for c in 0..3 {
                let v = src[i][c] as i32;
                let q = ((v + off) / s) * s;
                out[i][c] = q.clamp(0, 255) as u8;
            }
        }
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

fn main() {
    // Pick a few real source photos of varying content.
    let candidates = [
        "/mnt/v/input/zensim/sources/00b13be94a4867dd_1024sq.png",
        "/mnt/v/input/zensim/sources/3a695d7842d66b4d_1024sq.png",
        "/mnt/v/input/zensim/sources/0364e65f5cf278d0_818x1022.png",
        "/mnt/v/input/zensim/sources/473ee51327816001_818x1022.png",
        "/mnt/v/input/zensim/sources/03484c8be395c40d_1024sq.png",
    ];

    let z = Zensim::new(ZensimProfile::latest());
    let mut scratch = ZensimScratch::new();

    println!(
        "source,target_window,distortion_step,window_idx,window_y0,window_h,\
         per_window_canon_score,per_window_canon_raw_dist,per_window_scale0_proxy"
    );

    let distortion_steps: &[u8] = &[16, 32, 64];

    let mut summary: Vec<(String, u8, usize, Option<usize>, Option<usize>, usize)> = Vec::new();

    for src_path in &candidates {
        let path = PathBuf::from(src_path);
        if !path.exists() {
            eprintln!("skip {src_path} (missing)");
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();

        let Some((src_raw, w, h)) = load_rgb(src_path) else {
            eprintln!("skip {src_path}");
            continue;
        };
        let src_px = rgb_to_pixels(&src_raw);
        let width = w as usize;
        let height = h as usize;

        // Build per-window precomputed refs once for this source.
        let mut window_refs: Vec<(usize, usize, PrecomputedReference)> = Vec::new();
        let mut y = 0;
        while y < height {
            let win_h = (height - y).min(WINDOW_ROWS);
            if win_h < 8 {
                break;
            }
            let start = y * width;
            let end = start + win_h * width;
            let win_slice = RgbSlice::new(&src_px[start..end], width, win_h);
            window_refs.push((y, win_h, z.precompute_reference(&win_slice).unwrap()));
            y += WINDOW_ROWS;
        }
        let n_windows = window_refs.len();

        // Try injecting distortion into a few different target windows.
        let target_window_indices = [
            n_windows / 4,
            n_windows / 2,
            3 * n_windows / 4,
        ];

        for &target_w in &target_window_indices {
            let (target_y0, target_h, _) = window_refs[target_w];
            let target_y1 = target_y0 + target_h;
            for &step in distortion_steps {
                let distorted_px =
                    inject_distortion_in_band(&src_px, width, target_y0, target_y1, step);

                // Compute per-window canonical scores for all windows.
                let mut window_canon_scores: Vec<f64> = Vec::with_capacity(n_windows);
                let mut window_proxy_scores: Vec<f64> = Vec::with_capacity(n_windows);

                for (y0, win_h, pre) in &window_refs {
                    let start = y0 * width;
                    let end = start + win_h * width;
                    let dist_slice = RgbSlice::new(&distorted_px[start..end], width, *win_h);
                    let result = z.compute_with_ref_into(pre, &dist_slice, &mut scratch).unwrap();
                    window_canon_scores.push(result.score());
                    window_proxy_scores.push(scale0_proxy(result.features()));

                    println!(
                        "{},{},{},{},{},{},{:.6},{:.6},{:.6}",
                        stem,
                        target_w,
                        step,
                        window_canon_scores.len() - 1,
                        y0,
                        win_h,
                        result.score(),
                        result.raw_distance(),
                        scale0_proxy(result.features()),
                    );
                }

                // Find the window with the WORST score (lowest canonical score
                // = lowest perceived quality = most damage).
                let canon_min = window_canon_scores
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i);
                // Higher proxy = more damage (raw distance), so MAX proxy = worst.
                let proxy_max = window_proxy_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i);

                summary.push((stem.clone(), step, target_w, canon_min, proxy_max, n_windows));
            }
        }
    }

    eprintln!("\n# Spatial localization summary");
    eprintln!(
        "# {:>32} {:>5} {:>10} {:>14} {:>14} {:>10}",
        "source", "step", "target_w", "canon_pick", "proxy_pick", "n_win"
    );
    let mut canon_correct = 0usize;
    let mut proxy_correct = 0usize;
    let mut total = 0usize;
    for (stem, step, target_w, canon_pick, proxy_pick, n_win) in &summary {
        let cp = canon_pick.map_or("-".to_string(), |i| i.to_string());
        let pp = proxy_pick.map_or("-".to_string(), |i| i.to_string());
        let canon_ok = canon_pick == &Some(*target_w);
        let proxy_ok = proxy_pick == &Some(*target_w);
        eprintln!(
            "  {:>32} {:>5} {:>10} {:>14} {:>14} {:>10}  canon={} proxy={}",
            stem,
            step,
            target_w,
            cp,
            pp,
            n_win,
            if canon_ok { "✓" } else { "✗" },
            if proxy_ok { "✓" } else { "✗" },
        );
        if canon_ok {
            canon_correct += 1;
        }
        if proxy_ok {
            proxy_correct += 1;
        }
        total += 1;
    }
    eprintln!(
        "\n# canonical correctly localized: {}/{} = {:.0}%",
        canon_correct,
        total,
        canon_correct as f64 / total as f64 * 100.0
    );
    eprintln!(
        "# scale-0 proxy correctly localized: {}/{} = {:.0}%",
        proxy_correct,
        total,
        proxy_correct as f64 / total as f64 * 100.0
    );
}
