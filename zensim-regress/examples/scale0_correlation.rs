//! Validate whether scale-0 SSIM alone is a useful proxy for the canonical
//! multi-scale zensim score for use as a streaming AQ-controller signal.
//!
//! Methodology
//! -----------
//! For a corpus of (synthetic + real) images:
//!   * Run several distortion levels (block-quantization at varying step).
//!   * Compute canonical zensim raw_distance (multi-scale, all features, weights).
//!   * Compute scale-0-only proxy: dot(features[0..39], weights[0..39]).
//!   * Report:
//!       (a) intra-image SROCC scale-0 proxy vs canonical raw_distance
//!           — does the proxy track the gradient direction?
//!       (b) cross-image SROCC at fixed quality
//!           — is the proxy useful as an absolute level signal?
//!
//! Sub-Q2: For one large image and one quality level, slice the distorted
//! image into K-row windows, compute scale-0 proxy on each window
//! (re-running compute over the strip in isolation), and look at the
//! window-to-window noise relative to the change-from-distortion-strength.

use std::path::PathBuf;
use zensim::{RgbaSlice, Zensim, ZensimProfile, profile::WEIGHTS_PREVIEW_V0_1};

const SCALE0_BASIC_LEN: usize = 13 * 3; // 39 — first 39 entries of feature/weight vectors

/// Scale-0 proxy: dot(features[0..39], weights[0..39]) / num_scales_normalizer.
/// We don't divide by 4 because we want an unnormalized proxy magnitude — the
/// scaling is irrelevant for SROCC.
fn scale0_proxy(features: &[f64]) -> f64 {
    let n = SCALE0_BASIC_LEN.min(features.len());
    features[..n]
        .iter()
        .zip(WEIGHTS_PREVIEW_V0_1[..n].iter())
        .map(|(f, w)| f * w)
        .sum()
}

/// Block-quantize an RGBA image with a step of `step` per channel.
/// Larger step = stronger distortion. Used as a stand-in for codec output.
fn block_quantize(src: &[[u8; 4]], w: usize, h: usize, block: usize, step: u8) -> Vec<[u8; 4]> {
    let mut out = src.to_vec();
    let step = step.max(1) as u32;
    for by in 0..h.div_ceil(block) {
        for bx in 0..w.div_ceil(block) {
            // Average the block, then quantize each channel.
            let mut sums = [0u32; 3];
            let mut count = 0u32;
            let y0 = by * block;
            let x0 = bx * block;
            let y1 = (y0 + block).min(h);
            let x1 = (x0 + block).min(w);
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = src[y * w + x];
                    sums[0] += p[0] as u32;
                    sums[1] += p[1] as u32;
                    sums[2] += p[2] as u32;
                    count += 1;
                }
            }
            let mean = [sums[0] / count, sums[1] / count, sums[2] / count];
            // Snap pixels toward the block mean by `step`-quantization
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = y * w + x;
                    let p = src[i];
                    let mut q = [0u8; 4];
                    for c in 0..3 {
                        let m = mean[c] as i32;
                        let v = p[c] as i32;
                        let d = v - m;
                        let qd = (d / step as i32) * step as i32;
                        q[c] = (m + qd).clamp(0, 255) as u8;
                    }
                    q[3] = 255;
                    out[i] = q;
                }
            }
        }
    }
    out
}

// --- Synthetic image generators (deterministic, no I/O) ---

fn gradient_image(w: usize, h: usize, seed: u32) -> Vec<[u8; 4]> {
    let mut s = seed.wrapping_mul(2654435761);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let r = ((x * 251 ^ s as usize) & 0xff) as u8;
            let g = ((y * 173 ^ (s >> 8) as usize) & 0xff) as u8;
            let b = (((x ^ y) * 47 ^ (s >> 16) as usize) & 0xff) as u8;
            out.push([r, g, b, 255]);
        }
    }
    out
}

fn mandelbrot_image(w: usize, h: usize, zoom: f64) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(w * h);
    let cx = -0.743643887037151;
    let cy = 0.131825904205330;
    let scale = 3.0 / zoom / w as f64;
    for y in 0..h {
        for x in 0..w {
            let zr0 = (x as f64 - w as f64 / 2.0) * scale + cx;
            let zi0 = (y as f64 - h as f64 / 2.0) * scale + cy;
            let mut zr = 0.0;
            let mut zi = 0.0;
            let mut iter = 0;
            let max_iter = 80;
            while iter < max_iter && zr * zr + zi * zi < 4.0 {
                let nzr = zr * zr - zi * zi + zr0;
                zi = 2.0 * zr * zi + zi0;
                zr = nzr;
                iter += 1;
            }
            let v = (iter as f64 * 255.0 / max_iter as f64) as u8;
            // Some color modulation
            let r = v;
            let g = v.wrapping_add((iter * 5) as u8);
            let b = v.wrapping_mul(2);
            out.push([r, g, b, 255]);
        }
    }
    out
}

fn color_blocks(w: usize, h: usize, n: usize) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let bx = (x * n / w) as u32;
            let by = (y * n / h) as u32;
            let r = ((bx.wrapping_mul(73)) & 0xff) as u8;
            let g = ((by.wrapping_mul(151)) & 0xff) as u8;
            let b = (((bx + by).wrapping_mul(211)) & 0xff) as u8;
            out.push([r, g, b, 255]);
        }
    }
    out
}

// --- Real image loader ---
fn load_png(path: &str, max_dim: u32) -> Option<(Vec<[u8; 4]>, u32, u32)> {
    let img = image::open(path).ok()?;
    let img = if img.width().max(img.height()) > max_dim {
        let scale = max_dim as f32 / img.width().max(img.height()) as f32;
        img.resize(
            (img.width() as f32 * scale) as u32,
            (img.height() as f32 * scale) as u32,
            image::imageops::FilterType::Triangle,
        )
    } else {
        img
    };
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let pixels = rgba
        .pixels()
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect();
    Some((pixels, w, h))
}

// --- Spearman rank correlation ---
fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        let avg = ((i + j) as f64) / 2.0 + 1.0; // 1-based mean rank
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }
    ranks
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        num += da * db;
        va += da * da;
        vb += db * db;
    }
    if va == 0.0 || vb == 0.0 {
        return f64::NAN;
    }
    num / (va.sqrt() * vb.sqrt())
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let ra = rank(a);
    let rb = rank(b);
    pearson(&ra, &rb)
}

// --- Main experiment ---

struct Image {
    name: String,
    w: usize,
    h: usize,
    pixels: Vec<[u8; 4]>,
}

fn collect_corpus() -> Vec<Image> {
    let mut corpus = Vec::new();

    // Synthetic
    let synthetic_specs: Vec<(&str, fn(usize, usize) -> Vec<[u8; 4]>, usize, usize)> = vec![
        ("grad_512", |w, h| gradient_image(w, h, 1), 512, 512),
        ("grad_768", |w, h| gradient_image(w, h, 7), 768, 512),
        ("mandel_512", |w, h| mandelbrot_image(w, h, 1.0), 512, 512),
        ("mandel_768z", |w, h| mandelbrot_image(w, h, 8.0), 768, 512),
        ("blocks8_512", |w, h| color_blocks(w, h, 8), 512, 512),
        ("blocks16_768", |w, h| color_blocks(w, h, 16), 768, 512),
    ];
    for (name, gen_fn, w, h) in synthetic_specs {
        let pixels = gen_fn(w, h);
        corpus.push(Image { name: name.to_string(), w, h, pixels });
    }

    // Real photos: pull a few from /mnt/v/input/zensim/sources
    let dir = "/mnt/v/input/zensim/sources";
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|s| s.to_str()) == Some("png")
                    && !p
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .contains("512sq_512sq")
            })
            .collect();
        paths.sort();
        // Pick a stable spread
        let take = 24usize;
        let stride = (paths.len() / take).max(1);
        for p in paths.iter().step_by(stride).take(take) {
            if let Some((px, w, h)) = load_png(p.to_str().unwrap(), 768) {
                let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("?")
                    .chars()
                    .take(20)
                    .collect::<String>();
                corpus.push(Image {
                    name: format!("real_{}", name),
                    w: w as usize,
                    h: h as usize,
                    pixels: px,
                });
            }
        }
    }
    corpus
}

fn main() {
    let z = Zensim::new(ZensimProfile::PreviewV0_1);

    let corpus = collect_corpus();
    eprintln!("Corpus: {} images", corpus.len());

    // Distortion strengths to sweep — block sizes 8 (JPEG-like), step varies.
    let distortions: &[(&str, usize, u8)] = &[
        ("d8s4", 8, 4),
        ("d8s12", 8, 12),
        ("d8s32", 8, 32),
        ("d8s64", 8, 64),
        ("d8s128", 8, 128),
    ];

    // ---- Sub-Q1: Per-image rows, all distortions per row ----
    // For each (image, distortion): compute canonical raw_distance and scale-0 proxy.
    // Then compute SROCC within-image (across distortions) and across-image (per fixed distortion).

    println!("# ---------- Sub-Q1: scale-0 vs canonical correlation ----------");
    println!("# image,w,h,d_name,canonical_raw,canonical_score,scale0_proxy");
    let mut rows: Vec<(String, &'static str, f64, f64, f64)> = Vec::new();
    for img in &corpus {
        let src = RgbaSlice::new(&img.pixels, img.w, img.h);
        for (d_name, block, step) in distortions {
            let dst_pixels = block_quantize(&img.pixels, img.w, img.h, *block, *step);
            let dst = RgbaSlice::new(&dst_pixels, img.w, img.h);
            // Use compute_with_params with extended_features=false; default profile.
            let res = z.compute(&src, &dst).unwrap();
            let proxy = scale0_proxy(res.features());
            println!(
                "{},{},{},{},{:.6e},{:.4},{:.6e}",
                img.name,
                img.w,
                img.h,
                d_name,
                res.raw_distance(),
                res.score(),
                proxy
            );
            rows.push((img.name.clone(), d_name, res.raw_distance(), res.score(), proxy));
        }
    }

    // Intra-image SROCC: per image, rank distortions by canonical and proxy, compute SROCC.
    println!();
    println!("# ---------- Intra-image SROCC (gradient direction) ----------");
    println!("# image,n_distortions,srocc_raw,srocc_score");
    let mut intra_sroccs = Vec::new();
    for img in &corpus {
        let r: Vec<&_> = rows.iter().filter(|r| r.0 == img.name).collect();
        if r.len() < 3 {
            continue;
        }
        let raws: Vec<f64> = r.iter().map(|x| x.2).collect();
        let scores: Vec<f64> = r.iter().map(|x| x.3).collect();
        let proxies: Vec<f64> = r.iter().map(|x| x.4).collect();
        let s_raw = spearman(&raws, &proxies);
        let s_score = spearman(&scores, &proxies);
        intra_sroccs.push((s_raw, s_score));
        println!("{},{},{:.4},{:.4}", img.name, r.len(), s_raw, s_score);
    }
    let mean_intra_raw =
        intra_sroccs.iter().map(|x| x.0).sum::<f64>() / intra_sroccs.len() as f64;
    let mean_intra_score =
        intra_sroccs.iter().map(|x| x.1).sum::<f64>() / intra_sroccs.len() as f64;
    let n_high = intra_sroccs.iter().filter(|x| x.0 >= 0.9).count();
    println!(
        "# MEAN intra SROCC (raw_dist) = {:.4} ; n>=0.9: {}/{}",
        mean_intra_raw,
        n_high,
        intra_sroccs.len()
    );
    println!("# MEAN intra SROCC (score) = {:.4}", mean_intra_score);

    // Across-image SROCC: at fixed distortion, rank images.
    println!();
    println!("# ---------- Cross-image SROCC (calibration) ----------");
    println!("# distortion,n_images,srocc_raw,srocc_score");
    let mut all_d: Vec<(&str, f64, f64)> = Vec::new();
    for (d_name, _, _) in distortions {
        let r: Vec<&_> = rows.iter().filter(|r| r.1 == *d_name).collect();
        if r.len() < 3 {
            continue;
        }
        let raws: Vec<f64> = r.iter().map(|x| x.2).collect();
        let scores: Vec<f64> = r.iter().map(|x| x.3).collect();
        let proxies: Vec<f64> = r.iter().map(|x| x.4).collect();
        let s_raw = spearman(&raws, &proxies);
        let s_score = spearman(&scores, &proxies);
        all_d.push((d_name, s_raw, s_score));
        println!("{},{},{:.4},{:.4}", d_name, r.len(), s_raw, s_score);
    }
    let mean_cross_raw = all_d.iter().map(|x| x.1).sum::<f64>() / all_d.len() as f64;
    println!("# MEAN cross-image SROCC (raw_dist) = {:.4}", mean_cross_raw);

    // Pooled (every image at every distortion in one big rank)
    let raws: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let proxies: Vec<f64> = rows.iter().map(|r| r.4).collect();
    let pooled = spearman(&raws, &proxies);
    let pearson_pooled = pearson(&raws, &proxies);
    println!(
        "# POOLED SROCC = {:.4} ; Pearson = {:.4} ; n = {}",
        pooled,
        pearson_pooled,
        rows.len()
    );

    // ---- Sub-Q2: Per-window stability ----
    println!();
    println!("# ---------- Sub-Q2: per-K-row window scale-0 proxy ----------");
    // Use a 1024x1024 mandelbrot for richer content.
    let w = 1024usize;
    let h = 1024usize;
    let src = mandelbrot_image(w, h, 1.0);
    let src_slice = RgbaSlice::new(&src, w, h);
    // Two distortion strengths: a and b. We'll look at per-window proxy delta b - a.
    let dst_low = block_quantize(&src, w, h, 8, 12);
    let dst_high = block_quantize(&src, w, h, 8, 48);
    println!("# window_rows,strip_y,proxy_low,proxy_high,delta,canonical_raw_low,canonical_raw_high");
    for &k in &[32usize, 64, 128] {
        let mut deltas = Vec::new();
        let mut lows = Vec::new();
        let mut highs = Vec::new();
        let mut y = 0;
        // Need windows that are at least 16 rows for zensim min-size and divisible by 16
        // for scale 0 strip alignment. 32, 64, 128 all qualify.
        while y + k <= h {
            // Slice the SOURCE and DISTORTED buffer to a y..y+k strip.
            // Build per-strip RgbaSlice by copying — not efficient, but this is a
            // one-shot validation script.
            let strip_src: Vec<[u8; 4]> = (0..k)
                .flat_map(|r| src[(y + r) * w..(y + r) * w + w].iter().copied())
                .collect();
            let strip_low: Vec<[u8; 4]> = (0..k)
                .flat_map(|r| dst_low[(y + r) * w..(y + r) * w + w].iter().copied())
                .collect();
            let strip_high: Vec<[u8; 4]> = (0..k)
                .flat_map(|r| dst_high[(y + r) * w..(y + r) * w + w].iter().copied())
                .collect();
            let ss = RgbaSlice::new(&strip_src, w, k);
            let sl = RgbaSlice::new(&strip_low, w, k);
            let sh = RgbaSlice::new(&strip_high, w, k);
            let r_low = z.compute(&ss, &sl).unwrap();
            let r_high = z.compute(&ss, &sh).unwrap();
            let p_low = scale0_proxy(r_low.features());
            let p_high = scale0_proxy(r_high.features());
            println!(
                "{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                k,
                y,
                p_low,
                p_high,
                p_high - p_low,
                r_low.raw_distance(),
                r_high.raw_distance()
            );
            deltas.push(p_high - p_low);
            lows.push(p_low);
            highs.push(p_high);
            y += k;
        }
        // Stability: sign-consistency of (high - low). All windows should have positive delta
        // for the proxy to be a useful local AQ signal.
        let n = deltas.len();
        let n_pos = deltas.iter().filter(|d| **d > 0.0).count();
        let mean_delta = deltas.iter().sum::<f64>() / n as f64;
        let var_delta =
            deltas.iter().map(|d| (d - mean_delta).powi(2)).sum::<f64>() / n as f64;
        let mean_low = lows.iter().sum::<f64>() / n as f64;
        let var_low = lows.iter().map(|d| (d - mean_low).powi(2)).sum::<f64>() / n as f64;
        // Signal-to-noise: |mean(delta)| / stddev(low). If SNR > 1, delta dominates noise.
        let snr = if var_low > 0.0 {
            mean_delta.abs() / var_low.sqrt()
        } else {
            f64::INFINITY
        };
        // Also compute the "monotonic agreement": fraction of windows whose proxy delta
        // matches the canonical-raw direction.
        println!(
            "# K={} windows={} sign_agree={}/{} mean_delta={:.4e} var_low={:.4e} SNR={:.3}",
            k, n, n_pos, n, mean_delta, var_low, snr
        );
    }
}
