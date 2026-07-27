// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! V3 gate harness (`benchmarks/hdr_streaming_gates_2026-07-27.md`):
//! the SAME content presented as sRGB (SDR route) vs as
//! linear-100-nit-declared-HDR (PU route) must rank a distortion ladder
//! identically and drift boundedly per feature.
//!
//! Mapping: sRGB code v → `srgb_eotf(v) · 100` cd/m² — diffuse white at
//! 100 cd/m², the PU21 anchor (`PU21(100)/PU_WHITE = 1.0` exactly where
//! the cube-root path puts sRGB white), maximizing route agreement by
//! construction.
//!
//! Usage: `hdr_sdr_consistency <pairs.tsv> [n_refs]` — takes the ref
//! images from the TSV's unique `ref_path`s (sRGB PNGs), builds a
//! deterministic 9-step distortion ladder per ref, extracts 924 both
//! routes, and prints: per-pair readout scores (PreviewV0_2 weights over
//! the folded 228 prefix, rank-only use), route SROCC, and per-feature
//! drift stats.

#[path = "support/zen_io.rs"]
mod zen_io;

use zensim::feature_v2::{V2NewFeatureToggles, V2Scratch};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Absolute-linear cd/m² source (declared HDR).
struct NitsImage {
    data: Vec<[f32; 4]>,
    w: usize,
    h: usize,
}

impl ImageSource for NitsImage {
    fn width(&self) -> usize {
        self.w
    }
    fn height(&self) -> usize {
        self.h
    }
    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::LinearF32Rgba
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
    fn is_hdr(&self) -> bool {
        true
    }
}

fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.040_449_936 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn to_nits(px: &[[u8; 3]], w: usize, h: usize, white_nits: f32) -> NitsImage {
    NitsImage {
        data: px
            .iter()
            .map(|&[r, g, b]| {
                [
                    srgb_eotf(r as f32 / 255.0) * white_nits,
                    srgb_eotf(g as f32 / 255.0) * white_nits,
                    srgb_eotf(b as f32 / 255.0) * white_nits,
                    1.0,
                ]
            })
            .collect(),
        w,
        h,
    }
}

/// Deterministic distortion ladder: 3 types × 3 severities.
fn distort(src: &[[u8; 3]], w: usize, h: usize, kind: usize, level: usize) -> Vec<[u8; 3]> {
    match kind {
        // Posterize (quantization).
        0 => {
            let bits = [5, 4, 3][level];
            let mask = !((1u16 << (8 - bits)) - 1) as u8;
            let half = ((1u16 << (8 - bits)) / 2) as u8;
            src.iter()
                .map(|&[r, g, b]| [(r & mask) | half, (g & mask) | half, (b & mask) | half])
                .collect()
        }
        // Box blur, radius 1..=3.
        1 => {
            let rad = (level + 1) as isize;
            let mut out = vec![[0u8; 3]; w * h];
            for y in 0..h as isize {
                for x in 0..w as isize {
                    let mut acc = [0u32; 3];
                    let mut n = 0u32;
                    for dy in -rad..=rad {
                        for dx in -rad..=rad {
                            let yy = (y + dy).clamp(0, h as isize - 1) as usize;
                            let xx = (x + dx).clamp(0, w as isize - 1) as usize;
                            let p = src[yy * w + xx];
                            acc[0] += p[0] as u32;
                            acc[1] += p[1] as u32;
                            acc[2] += p[2] as u32;
                            n += 1;
                        }
                    }
                    out[(y as usize) * w + x as usize] =
                        [(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8];
                }
            }
            out
        }
        // Deterministic noise.
        _ => {
            let amp = [8i16, 16, 32][level];
            let mut state = 0x9E37_79B9u32;
            src.iter()
                .map(|&[r, g, b]| {
                    let mut n = || {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        ((state & 0xFF) as i16 - 128) * amp / 128
                    };
                    [
                        (r as i16 + n()).clamp(0, 255) as u8,
                        (g as i16 + n()).clamp(0, 255) as u8,
                        (b as i16 + n()).clamp(0, 255) as u8,
                    ]
                })
                .collect()
        }
    }
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    fn ranks(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0 + 1.0;
            for &k in &idx[i..=j] {
                r[k] = avg;
            }
            i = j + 1;
        }
        r
    }
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..a.len() {
        let xa = ra[i] - ma;
        let xb = rb[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    num / (da * db).sqrt().max(1e-12)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pairs_tsv = args.first().expect("usage: hdr_sdr_consistency <pairs.tsv> [n_refs]");
    let n_refs: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(8);

    let text = std::fs::read_to_string(pairs_tsv).expect("read pairs tsv");
    let mut refs: Vec<String> = Vec::new();
    for (i, line) in text.lines().enumerate().skip(1) {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() >= 2 && !refs.contains(&c[0].to_string()) {
            refs.push(c[0].to_string());
        }
        let _ = i;
        if refs.len() >= n_refs {
            break;
        }
    }
    eprintln!("{} refs", refs.len());

    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
    let mut scratch = V2Scratch::new();
    let weights = zensim::WEIGHTS;

    let mut sdr_scores = Vec::new();
    let mut hdr_scores = Vec::new();
    // Per-feature values across all pairs, both routes.
    let mut sdr_feats: Vec<Vec<f64>> = Vec::new();
    let mut hdr_feats: Vec<Vec<f64>> = Vec::new();

    for rp in &refs {
        let (r_px, rw, rh) = zen_io::decode_rgb8(std::path::Path::new(rp));
        let r_nits = to_nits(&r_px, rw, rh, 100.0);
        for kind in 0..3 {
            for level in 0..3 {
                let d_px = distort(&r_px, rw, rh, kind, level);
                let d_nits = to_nits(&d_px, rw, rh, 100.0);

                let sdr = z
                    .compute_folded720_append_features(
                        &RgbSlice::new(&r_px, rw, rh),
                        &RgbSlice::new(&d_px, rw, rh),
                    )
                    .unwrap();
                let hdr = z
                    .compute_folded720_append_features_hdr(
                        &r_nits,
                        &d_nits,
                        zensim::feature_v2::HdrEncoding::Linear,
                        V2NewFeatureToggles::default(),
                        &mut scratch,
                    )
                    .unwrap();

                let (ss, _) =
                    zensim::try_score_from_features(&sdr.features()[..228], weights).unwrap();
                let (hs, _) =
                    zensim::try_score_from_features(&hdr.features()[..228], weights).unwrap();
                sdr_scores.push(ss);
                hdr_scores.push(hs);
                sdr_feats.push(sdr.features().to_vec());
                hdr_feats.push(hdr.features().to_vec());
            }
        }
    }

    let n_pairs = sdr_scores.len();
    println!("pairs: {n_pairs}");
    println!("score SROCC (routes): {:.6}", spearman(&sdr_scores, &hdr_scores));
    // Within-ref SROCC: rank the 9-step ladder of ONE ref through both
    // routes (the pooled stat above additionally mixes cross-content
    // difficulty ordering, which no single-domain metric pins at 0.99
    // either — report both).
    let n_r = n_pairs / 9;
    let mut wr = Vec::new();
    for r in 0..n_r {
        let a = &sdr_scores[r * 9..(r + 1) * 9];
        let b = &hdr_scores[r * 9..(r + 1) * 9];
        wr.push(spearman(a, b));
    }
    wr.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!(
        "within-ref ladder SROCC: mean {:.4}  min {:.4}  (n={} refs)",
        wr.iter().sum::<f64>() / n_r as f64,
        wr[0],
        n_r
    );
    // Within-type SROCC (pairs are emitted kind-major per ref: idx % 9 / 3 = kind).
    for kind in 0..3 {
        let a: Vec<f64> = sdr_scores
            .iter()
            .enumerate()
            .filter(|(i, _)| (i % 9) / 3 == kind)
            .map(|(_, &v)| v)
            .collect();
        let b: Vec<f64> = hdr_scores
            .iter()
            .enumerate()
            .filter(|(i, _)| (i % 9) / 3 == kind)
            .map(|(_, &v)| v)
            .collect();
        println!(
            "  within-type {} SROCC: {:.6}",
            ["posterize", "blur", "noise"][kind],
            spearman(&a, &b)
        );
    }

    // Per-feature: SROCC across pairs (features with variance on both
    // routes) + drift stats.
    let nf = sdr_feats[0].len();
    let mut sroccs = Vec::new();
    let mut drift_abs = Vec::new();
    for f in 0..nf {
        let a: Vec<f64> = sdr_feats.iter().map(|v| v[f]).collect();
        let b: Vec<f64> = hdr_feats.iter().map(|v| v[f]).collect();
        let va = a.iter().any(|&x| (x - a[0]).abs() > 1e-12);
        let vb = b.iter().any(|&x| (x - b[0]).abs() > 1e-12);
        if va && vb {
            sroccs.push((spearman(&a, &b), f));
        }
        for i in 0..n_pairs {
            drift_abs.push((a[i] - b[i]).abs());
        }
    }
    sroccs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    drift_abs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let nn = sroccs.len();
    println!("per-feature SROCC over {nn} varying features:");
    println!(
        "  min {:.4} (f{})  p5 {:.4}  median {:.4}",
        sroccs[0].0,
        sroccs[0].1,
        sroccs[nn / 20].0,
        sroccs[nn / 2].0
    );
    println!("  worst 8: {:?}", &sroccs[..8.min(nn)]);
    println!(
        "per-feature |drift|: median {:.3e}  p95 {:.3e}  max {:.3e}",
        drift_abs[drift_abs.len() / 2],
        drift_abs[(drift_abs.len() as f64 * 0.95) as usize],
        drift_abs[drift_abs.len() - 1]
    );
    println!(
        "score |drift|: mean {:.3}  max {:.3} (dial points)",
        sdr_scores
            .iter()
            .zip(&hdr_scores)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / n_pairs as f64,
        sdr_scores
            .iter()
            .zip(&hdr_scores)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max)
    );
}
