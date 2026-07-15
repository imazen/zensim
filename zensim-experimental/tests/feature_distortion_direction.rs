//! Per-feature distortion-direction analysis for the 372-feature zensim
//! vector — the empirical basis for a monotone-by-construction encoder.
//!
//! For each of the 372 features we measure how it correlates with
//! INCREASING distortion across multiple content types and distortion
//! ladders (blur, posterize/banding, additive noise). A feature is:
//!   - **monotone-up**   if it consistently RISES with distortion (the
//!     SSIM/artifact/HF-loss error features) → encoder weight may be
//!     pinned `≥ 0`,
//!   - **monotone-down** if it consistently FALLS with distortion →
//!     weight pinned `≤ 0`,
//!   - **ambiguous**     if its sign flips across content or distortion
//!     type (the structurally-modulated IW-weighted / masked features)
//!     → must stay UNCONSTRAINED (constraining it the wrong way is the
//!     source of the V46 dial collapse).
//!
//! Also diagnoses the V39 inversion directly: scores the shipped
//! `ZensimProfile::A` and `LinearBounded` along the blur ladder and
//! reports where A is non-monotone (degraded scoring above less-degraded
//! / above identity).
//!
//! Run: `cargo test --release -p zensim --test feature_distortion_direction -- --nocapture`
//! Writes the per-feature classification TSV to
//! `benchmarks/feature_distortion_direction_2026-05-26.tsv`.

// Uses the training-feature-gated config API (ZensimConfig /
// compute_zensim_with_config). Compile-time gate so `cargo test -p zensim`
// (default features) builds cleanly; run via `cargo test -p zensim --features training`.
#![cfg(feature = "training")]
#![allow(deprecated)]
// exercises the deprecated `ZensimProfile::A` (shipped behind the default-on `deprecated-profiles` feature)

mod common;

use common::generators::*;
use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

const W: usize = 128;
const H: usize = 128;
const NF: usize = 372;

fn extract_372(src: &[[u8; 3]], dst: &[[u8; 3]]) -> Vec<f64> {
    let mut config = ZensimConfig::default();
    config.extended_features = true;
    config.compute_iw_features = true;
    let result = compute_zensim_with_config(src, dst, W, H, config).expect("feature extract");
    let f = result.features().to_vec();
    assert_eq!(f.len(), NF, "expected 372 features, got {}", f.len());
    f
}

/// Posterize to `2^bits` levels/channel (banding / blocking distortion).
fn distort_posterize(src: &[[u8; 3]], bits: u32) -> Vec<[u8; 3]> {
    let levels = (1u32 << bits).max(2);
    let step = 255.0 / (levels - 1) as f32;
    let q = |v: u8| ((v as f32 / step).round() * step).round().clamp(0.0, 255.0) as u8;
    src.iter().map(|p| [q(p[0]), q(p[1]), q(p[2])]).collect()
}

/// Additive value-noise perturbation at amplitude `amp` (added detail).
fn distort_noise(src: &[[u8; 3]], amp: i32, seed: u32) -> Vec<[u8; 3]> {
    let noise = gen_value_noise(W, H, seed);
    let off = |v: u8, nv: u8| {
        let d = ((nv as i32 - 128) * amp) / 128;
        (v as i32 + d).clamp(0, 255) as u8
    };
    src.iter()
        .zip(noise.iter())
        .map(|(p, n)| [off(p[0], n[0]), off(p[1], n[1]), off(p[2], n[2])])
        .collect()
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && v[idx[j + 1]] == v[idx[i]] {
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

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let x = a[i] - ma;
        let y = b[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da == 0.0 || db == 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    pearson(&ranks(xs), &ranks(ys))
}

#[test]
fn feature_distortion_direction_analysis() {
    let contents: Vec<(&str, Vec<[u8; 3]>)> = vec![
        ("color_blocks", gen_color_blocks(W, H)),
        ("checker", gen_checkerboard(W, H, 8)),
        ("mandelbrot", gen_mandelbrot(W, H)),
        ("value_noise", gen_value_noise(W, H, 42)),
    ];

    // Each entry: per-feature correlation with distortion level, one
    // value per (content × ladder). NF features × (4 contents × 3 ladders).
    let mut per_feat_corrs: Vec<Vec<f64>> = vec![Vec::new(); NF];
    let mut identity_feats: Vec<Vec<f64>> = vec![Vec::new(); NF];

    for (cname, src) in &contents {
        // Blur ladder (radius 0 = identity → 8). "detail loss".
        let blur_lvls: Vec<usize> = (0..=8).collect();
        let blur_xs: Vec<f64> = blur_lvls.iter().map(|&r| r as f64).collect();
        let mut blur_feats: Vec<Vec<f64>> = vec![Vec::new(); NF];
        // Per-level full feature rows (level → 372 feats), emitted in the
        // predict_features_with_bake wire format so any bake can be scored
        // along this blur ladder to check degradation-monotonicity.
        let mut blur_rows: Vec<Vec<f64>> = Vec::with_capacity(blur_lvls.len());
        for &r in &blur_lvls {
            let dst = if r == 0 {
                src.clone()
            } else {
                distort_blur(src, W, H, r)
            };
            let f = extract_372(src, &dst);
            if r == 0 {
                for j in 0..NF {
                    identity_feats[j].push(f[j]);
                }
            }
            for j in 0..NF {
                blur_feats[j].push(f[j]);
            }
            blur_rows.push(f);
        }
        for j in 0..NF {
            per_feat_corrs[j].push(spearman(&blur_xs, &blur_feats[j]));
        }
        // Emit wire format: u32 LE n_features, u32 LE n_rows, f32 matrix.
        let mut wire: Vec<u8> = Vec::new();
        wire.extend_from_slice(&(NF as u32).to_le_bytes());
        wire.extend_from_slice(&(blur_rows.len() as u32).to_le_bytes());
        for row in &blur_rows {
            for &v in row {
                wire.extend_from_slice(&(v as f32).to_le_bytes());
            }
        }
        let _ = std::fs::write(format!("/tmp/blur_ladder_{cname}.featmat"), &wire);

        // Posterize ladder (bits 8→1; distortion level = 8 − bits). "banding".
        let post_bits: Vec<u32> = vec![8, 7, 6, 5, 4, 3, 2, 1];
        let post_xs: Vec<f64> = post_bits.iter().map(|&b| (8 - b) as f64).collect();
        let mut post_feats: Vec<Vec<f64>> = vec![Vec::new(); NF];
        for &b in &post_bits {
            let dst = distort_posterize(src, b);
            let f = extract_372(src, &dst);
            for j in 0..NF {
                post_feats[j].push(f[j]);
            }
        }
        for j in 0..NF {
            per_feat_corrs[j].push(spearman(&post_xs, &post_feats[j]));
        }

        // Noise ladder (amp 0→64). "added detail".
        let noise_amps: Vec<i32> = vec![0, 8, 16, 24, 32, 48, 64];
        let noise_xs: Vec<f64> = noise_amps.iter().map(|&a| a as f64).collect();
        let mut noise_feats: Vec<Vec<f64>> = vec![Vec::new(); NF];
        for &a in &noise_amps {
            let dst = if a == 0 {
                src.clone()
            } else {
                distort_noise(src, a, 7)
            };
            let f = extract_372(src, &dst);
            for j in 0..NF {
                noise_feats[j].push(f[j]);
            }
        }
        for j in 0..NF {
            per_feat_corrs[j].push(spearman(&noise_xs, &noise_feats[j]));
        }
    }

    // Classify each feature by sign-consistency of its distortion correlation.
    let (mut n_up, mut n_down, mut n_ambig) = (0usize, 0usize, 0usize);
    let mut class = vec![0i8; NF];
    for j in 0..NF {
        let cs = &per_feat_corrs[j];
        let min = cs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = cs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = cs.iter().sum::<f64>() / cs.len() as f64;
        if min > 0.3 && mean > 0.5 {
            class[j] = 1;
            n_up += 1;
        } else if max < -0.3 && mean < -0.5 {
            class[j] = -1;
            n_down += 1;
        } else {
            class[j] = 0;
            n_ambig += 1;
        }
    }

    // Identity-feature magnitude: error features should be ≈0 at identity.
    let id_mean = identity_feats
        .iter()
        .flatten()
        .map(|v| v.abs())
        .sum::<f64>()
        / (NF * contents.len()) as f64;
    let id_max = identity_feats
        .iter()
        .flatten()
        .cloned()
        .fold(0.0f64, |a, b| a.max(b.abs()));

    // V39 inversion diagnosis: blur-ladder scores for A (V39) vs LinearBounded.
    let z_a = Zensim::new(ZensimProfile::A);
    let z_lb = Zensim::new(zensim_experimental::linear_bounded());
    let mut a_inversions = 0usize;
    let mut a_above_identity = 0usize;
    let mut lb_inversions = 0usize;
    let mut diag = String::new();
    for (cname, src) in &contents {
        let mut a_prev = f64::INFINITY;
        let mut lb_prev = f64::INFINITY;
        let mut a_id = 0.0;
        for r in 0..=6usize {
            let dst = if r == 0 {
                src.clone()
            } else {
                distort_blur(src, W, H, r)
            };
            let sa = z_a
                .compute(&RgbSlice::new(src, W, H), &RgbSlice::new(&dst, W, H))
                .unwrap()
                .score();
            let slb = z_lb
                .compute(&RgbSlice::new(src, W, H), &RgbSlice::new(&dst, W, H))
                .unwrap()
                .score();
            if r == 0 {
                a_id = sa;
            } else {
                if sa > a_prev + 1e-6 {
                    a_inversions += 1;
                }
                if sa > a_id + 1e-6 {
                    a_above_identity += 1;
                }
                if slb > lb_prev + 1e-6 {
                    lb_inversions += 1;
                }
            }
            diag.push_str(&format!("{cname} blur{r}: A={sa:.2} LB={slb:.2}\n"));
            a_prev = sa;
            lb_prev = slb;
        }
    }

    // Persist the per-feature classification.
    let mut tsv = String::from("feat_idx\tclass\tmean_corr\tmin_corr\tmax_corr\n");
    for j in 0..NF {
        let cs = &per_feat_corrs[j];
        let mean = cs.iter().sum::<f64>() / cs.len() as f64;
        let min = cs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = cs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let cl = match class[j] {
            1 => "up",
            -1 => "down",
            _ => "ambiguous",
        };
        tsv.push_str(&format!("{j}\t{cl}\t{mean:.4}\t{min:.4}\t{max:.4}\n"));
    }
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../benchmarks/feature_distortion_direction_2026-05-26.tsv");
    let _ = std::fs::write(&out, &tsv);

    eprintln!(
        "\n=== Per-feature distortion-direction (NF={NF}, 4 contents × 3 ladders = 12 corrs/feat) ==="
    );
    eprintln!("  monotone-UP   (pin W1 ≥ 0): {n_up}");
    eprintln!("  monotone-DOWN (pin W1 ≤ 0): {n_down}");
    eprintln!("  AMBIGUOUS (leave unconstrained / route to offset): {n_ambig}");
    eprintln!(
        "  identity-feature |value|: mean={id_mean:.4} max={id_max:.4} (error features → ~0 at identity)"
    );
    eprintln!("\n=== V39 (Profile::A) inversion diagnosis on blur ladder ===");
    eprint!("{diag}");
    eprintln!(
        "  A (V39): {a_inversions} adjacent-step inversions, {a_above_identity} scores ABOVE identity"
    );
    eprintln!("  LinearBounded: {lb_inversions} adjacent-step inversions (should be 0)");
    eprintln!("  classification TSV → {}", out.display());

    // Sanity assertions (the analysis itself, not a quality gate).
    assert!(n_up > 0, "expected some monotone-up error features");
    assert!(
        n_up + n_down + n_ambig == NF,
        "classification must cover all features"
    );
}
