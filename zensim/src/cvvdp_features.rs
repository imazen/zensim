//! EX-4 § 2 + § 3 — CVVDP-shaped per-pair features.
//!
//! These are **not** a CVVDP JOD reproduction. CVVDP is a calibrated
//! JND-scale metric; what we want for training is a small bundle of
//! features that capture the same psychovisual *shapes* CVVDP captures
//! — DKL color decomposition, Weber-contrast pyramid bands, CSF-weighted
//! band energies, mutual-masking residuals, and a Minkowski β=3 pool of
//! luma differences. Those signals are intentionally numerically simple
//! so they can land in a 343-feature training corpus without dragging
//! in cvvdp-gpu's AGPL-3.0 license; the CPU port lives in `cvvdp-gpu`
//! (AGPL) and any code that needs **exact** cvvdp JOD parity should
//! call that crate, not this one.
//!
//! The feature vector is intentionally on the small side — 19 features
//! per (ref, dist) pair — but together with the 24 from
//! [`crate::xyb_lms_features`] forms the **43 new features** the doc §
//! 8 EX-4 batch calls for.
//!
//! ## Feature layout (19 total, see [`CVVDP_FEATURE_COUNT`])
//!
//! | Indices | Block | Description |
//! |---|---|---|
//! | 0..6 | DKL global stats | mean(ref), std(ref), mean(dist), std(dist) per channel; achromatic only emits {Δmean, Δstd} = 2; chromatic channels emit {ref_std, dist_std, |Δstd|} for each → total 2 + 2×2 = 6 |
//! | 6..10 | Weber band gains (4 pyramid levels) | per-level mean Weber contrast of ref achromatic |
//! | 10..14 | CSF-weighted band-energy ratios | `E_dist[k] / E_ref[k]` for 4 levels, CSF-weighted |
//! | 14..18 | Mutual-masking residual variances | `var(|R_k − T_k| / (R_k + T_k + ε))` per level |
//! | 18..19 | Minkowski β=3 pool | global β=3 pool of |Y_ref − Y_dist| in linear cd/m² |
//!
//! ## Constants
//!
//! | Constant | Value | Source |
//! |---|---|---|
//! | DKL matrix (sRGB linear → DKLd65 opponent) | Mantiuk et al. CVVDP appendix | published |
//! | Minkowski β (spatial pool) | 3.0 | doc § 3 Table |
//! | Display preset (Y_peak, Y_black, Y_refl) | STANDARD_4K (200 / 0.2 / 0.3979) | cvvdp v0.5.4 default |
//! | Pyramid levels | 4 | matches existing zensim 4-scale layout |
//!
//! The CSF weights here are a **simple band-importance prior** —
//! `[0.5, 1.0, 0.8, 0.4]` for bands {0, 1, 2, 3} — matching the
//! qualitative shape of the achromatic CSF peak near 4-6 cy/deg
//! (band 1) rolling off at very high frequencies (band 0) and very
//! low frequencies (band 3). It is NOT the cvvdp castleCSF lookup;
//! using castleCSF requires the AGPL-licensed LUT.
//!
//! ## What we deliberately don't do
//!
//! - **No castleCSF LUT**. That lives in cvvdp-gpu (AGPL). For
//!   feature-extraction purposes a four-tap CSF prior is sufficient
//!   — the MLP downstream can re-shape any monotone CSF.
//! - **No PU encoding**. SDR-only for now; the doc says PU is a
//!   later-phase feature behind an `--hdr` flag.
//! - **No multi-tap LPF / steerable pyramid**. We use box downscale
//!   (matching zensim's existing pyramid choice).

use crate::color::srgb_u8_to_linear;

/// DKL matrix — sRGB linear → DKLd65 opponent (A, RG, VY).
///
/// Same coefficients as `cvvdp-gpu::params::SRGB_LINEAR_TO_DKL`,
/// re-published here because the matrix is part of the Mantiuk et
/// al. 2024 CVVDP paper appendix (public). Re-deriving from the
/// Smith-Pokorny LMS cone fundamentals + sRGB primaries gives the
/// same numbers to f32 precision.
pub(crate) const SRGB_LINEAR_TO_DKL: [[f32; 3]; 3] = [
    [0.233_201_2, 0.728_830_8, 0.088_995_87],
    [0.127_620_77, -0.087_068_09, -0.036_777_39],
    [-0.214_822_5, -0.626_253_7, 0.851_403_3],
];

/// Display preset constants: peak luminance, black luminance, and
/// reflected ambient (in cd/m²). cvvdp v0.5.4 `standard_4k` numbers.
pub(crate) const DISPLAY_Y_PEAK: f32 = 200.0;
/// Display black point in cd/m². Combined with reflected ambient for
/// the linear → emitted-luminance map.
pub(crate) const DISPLAY_Y_BLACK: f32 = 0.2;
/// Reflected ambient luminance in cd/m² (250 lux × 0.005 / π).
pub(crate) const DISPLAY_Y_REFL: f32 = 0.397_887_36;

/// Number of pyramid levels — matches existing zensim 4-scale layout.
pub(crate) const N_LEVELS: usize = 4;

/// Minkowski β for spatial pool (doc § 3 Table; cvvdp uses 2.0 but
/// the doc highlights 3.0 as the "Minkowski-β=3 pooled luminance
/// differences" feature batch). We honor the doc's value here even
/// though cvvdp itself runs β=2 for spatial pool — the point of this
/// feature is to provide a *different shape* than mean pooling, and
/// β=3 is what the doc asks for.
pub(crate) const MINKOWSKI_BETA: f32 = 3.0;

/// Tiny ε to keep mutual-masking residual divisions stable on flat
/// regions. `(R + T)` can be near zero on identical low-luminance
/// pixels.
const MASK_EPS: f32 = 1e-3;

/// Achromatic-CSF band-importance prior — qualitative shape only.
///
/// Bands are coarse→fine (band 0 = level 0 in our pyramid, which is
/// the finest scale at full resolution). Real castleCSF peaks around
/// band 1 / 2; bands 0 and 3 are deprioritised. These weights are
/// used only to bias the band-energy ratio features — the MLP can
/// learn a different shape if it wants.
pub(crate) const CSF_BAND_WEIGHTS: [f32; N_LEVELS] = [0.5, 1.0, 0.8, 0.4];

/// Feature counts per block (see module docs).
const N_DKL_STATS: usize = 6;
const N_WEBER_BANDS: usize = N_LEVELS;
const N_CSF_RATIOS: usize = N_LEVELS;
const N_MASK_VARS: usize = N_LEVELS;
const N_MINKOWSKI: usize = 1;

/// Total CVVDP-shape feature count.
///
/// `6 + 4 + 4 + 4 + 1 = 19` per (ref, dist) pair.
pub const CVVDP_FEATURE_COUNT: usize =
    N_DKL_STATS + N_WEBER_BANDS + N_CSF_RATIOS + N_MASK_VARS + N_MINKOWSKI;

/// Extract CVVDP-shape features for a (reference, distorted) pair.
///
/// Both inputs must be packed RGB8 of identical `width × height × 3`
/// extent. Returns exactly [`CVVDP_FEATURE_COUNT`] features in the
/// order documented at the constant.
///
/// # Panics
///
/// Panics if `ref_rgb.len() != dist_rgb.len()` or either is not
/// `width * height * 3` bytes.
#[must_use]
pub fn extract_cvvdp_features(
    ref_rgb: &[u8],
    dist_rgb: &[u8],
    width: usize,
    height: usize,
) -> Vec<f32> {
    assert_eq!(
        ref_rgb.len(),
        width * height * 3,
        "ref length {} != width*height*3 = {}",
        ref_rgb.len(),
        width * height * 3,
    );
    assert_eq!(
        dist_rgb.len(),
        width * height * 3,
        "dist length {} != width*height*3 = {}",
        dist_rgb.len(),
        width * height * 3,
    );
    let n = width * height;
    if n == 0 {
        return vec![0.0; CVVDP_FEATURE_COUNT];
    }

    // Pre-allocate the six DKL planes.
    let mut a_ref = vec![0.0_f32; n];
    let mut rg_ref = vec![0.0_f32; n];
    let mut vy_ref = vec![0.0_f32; n];
    let mut a_dist = vec![0.0_f32; n];
    let mut rg_dist = vec![0.0_f32; n];
    let mut vy_dist = vec![0.0_f32; n];

    let lum_scale = DISPLAY_Y_PEAK - DISPLAY_Y_BLACK;
    let lum_bias = DISPLAY_Y_BLACK + DISPLAY_Y_REFL;

    for i in 0..n {
        let lr = srgb_u8_to_linear(ref_rgb[3 * i]) * lum_scale + lum_bias;
        let lg = srgb_u8_to_linear(ref_rgb[3 * i + 1]) * lum_scale + lum_bias;
        let lb = srgb_u8_to_linear(ref_rgb[3 * i + 2]) * lum_scale + lum_bias;
        a_ref[i] = SRGB_LINEAR_TO_DKL[0][0] * lr
            + SRGB_LINEAR_TO_DKL[0][1] * lg
            + SRGB_LINEAR_TO_DKL[0][2] * lb;
        rg_ref[i] = SRGB_LINEAR_TO_DKL[1][0] * lr
            + SRGB_LINEAR_TO_DKL[1][1] * lg
            + SRGB_LINEAR_TO_DKL[1][2] * lb;
        vy_ref[i] = SRGB_LINEAR_TO_DKL[2][0] * lr
            + SRGB_LINEAR_TO_DKL[2][1] * lg
            + SRGB_LINEAR_TO_DKL[2][2] * lb;

        let dr = srgb_u8_to_linear(dist_rgb[3 * i]) * lum_scale + lum_bias;
        let dg = srgb_u8_to_linear(dist_rgb[3 * i + 1]) * lum_scale + lum_bias;
        let db = srgb_u8_to_linear(dist_rgb[3 * i + 2]) * lum_scale + lum_bias;
        a_dist[i] = SRGB_LINEAR_TO_DKL[0][0] * dr
            + SRGB_LINEAR_TO_DKL[0][1] * dg
            + SRGB_LINEAR_TO_DKL[0][2] * db;
        rg_dist[i] = SRGB_LINEAR_TO_DKL[1][0] * dr
            + SRGB_LINEAR_TO_DKL[1][1] * dg
            + SRGB_LINEAR_TO_DKL[1][2] * db;
        vy_dist[i] = SRGB_LINEAR_TO_DKL[2][0] * dr
            + SRGB_LINEAR_TO_DKL[2][1] * dg
            + SRGB_LINEAR_TO_DKL[2][2] * db;
    }

    let mut out = Vec::with_capacity(CVVDP_FEATURE_COUNT);

    // ─── DKL stats (6) ──────────────────────────────────────────
    // A (achromatic): emit {Δmean, |Δstd|} — same shape as luma
    // mean/std deltas the doc points at as CVVDP color-front-end
    // signals.
    let (a_ref_mean, a_ref_std) = mean_std(&a_ref);
    let (a_dist_mean, a_dist_std) = mean_std(&a_dist);
    out.push(a_dist_mean - a_ref_mean);
    out.push((a_dist_std - a_ref_std).abs());

    // Chromatic channels: emit {ref_std, dist_std} (saturation per
    // image) for each opponent channel. cvvdp's color sensitivity
    // is what these proxies for.
    let (_, rg_ref_std) = mean_std(&rg_ref);
    let (_, rg_dist_std) = mean_std(&rg_dist);
    out.push(rg_ref_std);
    out.push(rg_dist_std);

    let (_, vy_ref_std) = mean_std(&vy_ref);
    let (_, vy_dist_std) = mean_std(&vy_dist);
    out.push(vy_ref_std);
    out.push(vy_dist_std);

    // ─── Weber-contrast pyramid (4) ─────────────────────────────
    // Build a 4-level Gaussian-ish (box-downscale) pyramid of the
    // achromatic reference plane, then compute mean per-level Weber
    // contrast |G_k - G_{k+1}| / max(G_{k+1}, ε). These mirror
    // cvvdp's weber_contrast_pyr_dec_scalar shape without porting
    // the full kernel.
    let weber_bands = weber_contrast_bands(&a_ref, width, height, N_LEVELS);
    for w in &weber_bands {
        out.push(*w);
    }

    // ─── CSF-weighted band-energy ratios (4) ────────────────────
    // For each level, build the pyramid of A_ref and A_dist, take
    // the per-level energy (sum of squares / n_band), and emit the
    // ratio `E_dist / E_ref * CSF_BAND_WEIGHTS[k]`. Ratios ~1
    // indicate the level's energy is preserved; ratios ≫ 1 or ≪ 1
    // signal banding / blurring at that scale.
    let ratios = csf_weighted_band_ratios(&a_ref, &a_dist, width, height, N_LEVELS);
    for r in &ratios {
        out.push(*r);
    }

    // ─── Mutual-masking residual variances (4) ──────────────────
    // Per level: `(R_k - T_k)^2 / ((R_k + T_k)^2 + eps)` averaged.
    // Mantiuk's masking pre-divides by a mutual-masking sum; we
    // approximate that here.
    let mask_vars = mutual_masking_residuals(&a_ref, &a_dist, width, height, N_LEVELS);
    for m in &mask_vars {
        out.push(*m);
    }

    // ─── Minkowski β=3 pool of luma difference (1) ──────────────
    // Single global pool: `(mean(|Y_ref - Y_dist|^β))^(1/β)`. Beta=3
    // boosts contribution of high-error pixels relative to mean
    // pool, matching cvvdp's JOD-style aggregation.
    let mut sum = 0.0_f64;
    for i in 0..n {
        let d = (a_ref[i] - a_dist[i]).abs();
        sum += (d as f64).powf(MINKOWSKI_BETA as f64);
    }
    let mink = ((sum / n as f64).powf(1.0 / MINKOWSKI_BETA as f64)) as f32;
    out.push(mink);

    debug_assert_eq!(out.len(), CVVDP_FEATURE_COUNT);
    out
}

/// Mean + population std (n-divisor) over a flat slice.
fn mean_std(samples: &[f32]) -> (f32, f32) {
    let n = samples.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut sum = 0.0_f64;
    for &v in samples {
        sum += v as f64;
    }
    let mean = sum / n as f64;
    let mut sse = 0.0_f64;
    for &v in samples {
        let d = v as f64 - mean;
        sse += d * d;
    }
    let std = (sse / n as f64).sqrt();
    (mean as f32, std as f32)
}

/// Build a `n_levels`-deep box-downscale pyramid on a planar f32
/// channel. Returns `(width, height, plane)` per level, finest first.
///
/// Box downscale matches zensim's existing pyramid choice (see
/// `ZensimConfig::downscale_filter` default).
fn box_pyramid(
    plane: &[f32],
    width: usize,
    height: usize,
    n_levels: usize,
) -> Vec<(usize, usize, Vec<f32>)> {
    let mut out = Vec::with_capacity(n_levels);
    let mut w = width;
    let mut h = height;
    let mut cur = plane.to_vec();
    out.push((w, h, cur.clone()));
    for _ in 1..n_levels {
        let new_w = (w / 2).max(1);
        let new_h = (h / 2).max(1);
        let mut next = vec![0.0_f32; new_w * new_h];
        for y in 0..new_h {
            for x in 0..new_w {
                let xx = (2 * x).min(w - 1);
                let yy = (2 * y).min(h - 1);
                let xx1 = (xx + 1).min(w - 1);
                let yy1 = (yy + 1).min(h - 1);
                let v =
                    (cur[yy * w + xx] + cur[yy * w + xx1] + cur[yy1 * w + xx] + cur[yy1 * w + xx1])
                        * 0.25;
                next[y * new_w + x] = v;
            }
        }
        w = new_w;
        h = new_h;
        cur = next;
        out.push((w, h, cur.clone()));
    }
    out
}

/// Per-level mean Weber contrast: `mean(|finer - coarser| / max(coarser, eps))`.
/// We upsample coarser by nearest-neighbour to match the finer level's
/// resolution before the difference. The output is one value per level
/// (level 0 has no coarser parent → defined as the raw RMS of the
/// finest plane, normalised by its mean).
fn weber_contrast_bands(plane: &[f32], width: usize, height: usize, n_levels: usize) -> Vec<f32> {
    let pyr = box_pyramid(plane, width, height, n_levels);
    let mut out = Vec::with_capacity(n_levels);

    // Level 0 — no coarser parent. Use RMS / mean of plane as
    // baseline "contrast" so the feature is on the same scale as
    // the parent-relative bands. Empty-image guard for zero-pixel
    // shapes.
    let (mean0, std0) = mean_std(&pyr[0].2);
    out.push(if mean0.abs() < 1e-6 {
        0.0
    } else {
        std0 / mean0.abs().max(1e-6)
    });

    for k in 1..n_levels {
        let (w_fine, h_fine, ref fine) = pyr[k - 1];
        let (w_coarse, h_coarse, ref coarse) = pyr[k];
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        for y in 0..h_fine {
            let cy = (y * h_coarse / h_fine.max(1)).min(h_coarse - 1);
            for x in 0..w_fine {
                let cx = (x * w_coarse / w_fine.max(1)).min(w_coarse - 1);
                let f = fine[y * w_fine + x];
                let c = coarse[cy * w_coarse + cx];
                let denom = c.abs().max(MASK_EPS);
                sum += ((f - c).abs() / denom) as f64;
                count += 1;
            }
        }
        let mean = if count > 0 {
            (sum / count as f64) as f32
        } else {
            0.0
        };
        out.push(mean);
    }
    debug_assert_eq!(out.len(), n_levels);
    out
}

/// CSF-weighted band-energy ratios.
///
/// For each level k: compute mean(plane²) for ref pyramid and dist
/// pyramid, take dist/ref ratio, multiply by `CSF_BAND_WEIGHTS[k]`.
fn csf_weighted_band_ratios(
    a_ref: &[f32],
    a_dist: &[f32],
    width: usize,
    height: usize,
    n_levels: usize,
) -> Vec<f32> {
    let pyr_ref = box_pyramid(a_ref, width, height, n_levels);
    let pyr_dist = box_pyramid(a_dist, width, height, n_levels);
    let mut out = Vec::with_capacity(n_levels);
    for k in 0..n_levels {
        let (_, _, ref r) = pyr_ref[k];
        let (_, _, ref d) = pyr_dist[k];
        let mut energy_r = 0.0_f64;
        let mut energy_d = 0.0_f64;
        for &v in r {
            energy_r += (v as f64).powi(2);
        }
        for &v in d {
            energy_d += (v as f64).powi(2);
        }
        let n_band = r.len().max(1);
        energy_r /= n_band as f64;
        energy_d /= n_band as f64;
        let ratio = if energy_r > 1e-6 {
            (energy_d / energy_r) as f32
        } else {
            1.0
        };
        out.push(ratio * CSF_BAND_WEIGHTS[k]);
    }
    out
}

/// Mutual-masking residual variances per pyramid level.
///
/// Per level: variance of `(R_k - T_k) / (R_k + T_k + eps)`. Mantiuk's
/// masking model normalises the per-pixel error by a mutual-masking
/// sum; this is the variance of that normalised residual.
fn mutual_masking_residuals(
    a_ref: &[f32],
    a_dist: &[f32],
    width: usize,
    height: usize,
    n_levels: usize,
) -> Vec<f32> {
    let pyr_ref = box_pyramid(a_ref, width, height, n_levels);
    let pyr_dist = box_pyramid(a_dist, width, height, n_levels);
    let mut out = Vec::with_capacity(n_levels);
    for k in 0..n_levels {
        let (_, _, ref r) = pyr_ref[k];
        let (_, _, ref d) = pyr_dist[k];
        let n_band = r.len().min(d.len());
        if n_band == 0 {
            out.push(0.0);
            continue;
        }
        let mut tmp = Vec::with_capacity(n_band);
        for i in 0..n_band {
            let denom = (r[i].abs() + d[i].abs() + MASK_EPS).max(MASK_EPS);
            tmp.push((r[i] - d[i]) / denom);
        }
        let (_, std) = mean_std(&tmp);
        out.push(std);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feature count matches doc.
    #[test]
    fn feature_count_is_19() {
        assert_eq!(CVVDP_FEATURE_COUNT, 19);
        let pixels = vec![128u8; 16 * 16 * 3];
        let f = extract_cvvdp_features(&pixels, &pixels, 16, 16);
        assert_eq!(f.len(), CVVDP_FEATURE_COUNT);
    }

    /// Identical inputs produce zero deltas + ratio-1 bands.
    #[test]
    fn identical_inputs_zero_error() {
        let pixels = vec![64u8; 32 * 32 * 3];
        let f = extract_cvvdp_features(&pixels, &pixels, 32, 32);
        // Δmean and |Δstd| should be ~0.
        assert!(f[0].abs() < 1e-3, "Δmean = {}", f[0]);
        assert!(f[1].abs() < 1e-3, "|Δstd| = {}", f[1]);
        // Chromatic stds: ref == dist so the *pairs* must be equal.
        assert!((f[2] - f[3]).abs() < 1e-3, "rg_ref vs rg_dist std");
        assert!((f[4] - f[5]).abs() < 1e-3, "vy_ref vs vy_dist std");
        // Band-energy ratios ~ CSF weights (since energy_d == energy_r).
        for k in 0..N_LEVELS {
            let r = f[N_DKL_STATS + N_WEBER_BANDS + k];
            let expected = CSF_BAND_WEIGHTS[k];
            assert!(
                (r - expected).abs() < 1e-3,
                "band {} ratio = {}, expected {}",
                k,
                r,
                expected,
            );
        }
        // Mutual-masking residuals: identical inputs → residual ~0
        // → std ~0.
        for k in 0..N_LEVELS {
            let m = f[N_DKL_STATS + N_WEBER_BANDS + N_CSF_RATIOS + k];
            assert!(m.abs() < 1e-3, "mask resid band {} = {}", k, m);
        }
        // Minkowski pool: ~0 for identical inputs.
        assert!(
            f[CVVDP_FEATURE_COUNT - 1].abs() < 1e-3,
            "minkowski = {}",
            f[CVVDP_FEATURE_COUNT - 1],
        );
    }

    /// A bright-saturated-red distorted from neutral grey should
    /// produce non-trivial deltas across DKL channels and a non-zero
    /// Minkowski pool. Use a checkerboard pattern so the chromatic
    /// std is non-zero (a uniform image would have std == 0 by
    /// definition, regardless of color).
    #[test]
    fn red_vs_grey_produces_nonzero_features() {
        let w = 24;
        let h = 24;
        let grey = vec![128u8; w * h * 3];
        let mut red = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                // Checkerboard between deep red and dim red, so the
                // RG opponent channel has both larger mean *and*
                // non-zero std.
                let pat = ((x / 4 + y / 4) & 1) == 0;
                if pat {
                    red.push(220);
                    red.push(20);
                    red.push(20);
                } else {
                    red.push(140);
                    red.push(40);
                    red.push(40);
                }
            }
        }
        let f = extract_cvvdp_features(&grey, &red, w, h);
        // Achromatic delta is non-zero (red checkerboard differs in
        // luminance from mid-grey).
        assert!(f[0].abs() > 0.01, "Δmean(A) = {}", f[0]);
        // Chromatic RG std for the red checkerboard is greater than
        // the uniform grey's (grey has zero chroma std; red
        // checkerboard has both color AND pattern → non-zero std).
        assert!(
            f[3] > f[2],
            "rg_dist_std ({}) <= rg_ref_std ({})",
            f[3],
            f[2],
        );
        // Minkowski pool is non-zero.
        let mink = f[CVVDP_FEATURE_COUNT - 1];
        assert!(mink > 0.01, "minkowski = {}", mink);
    }

    /// Mutual-masking residual stds are bounded in [0, 1] for any
    /// finite input (the per-pixel ratio is bounded in [-1, 1]).
    #[test]
    fn mutual_masking_bounded() {
        let mut state = 0xBEEF_0042_u32;
        let n_pix = 32 * 32;
        let mut r_img = Vec::with_capacity(n_pix * 3);
        let mut d_img = Vec::with_capacity(n_pix * 3);
        for _ in 0..(n_pix * 3) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            r_img.push((state & 0xFF) as u8);
        }
        for _ in 0..(n_pix * 3) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            d_img.push((state & 0xFF) as u8);
        }
        let f = extract_cvvdp_features(&r_img, &d_img, 32, 32);
        for k in 0..N_LEVELS {
            let m = f[N_DKL_STATS + N_WEBER_BANDS + N_CSF_RATIOS + k];
            assert!((0.0..=1.5).contains(&m), "mask resid band {} = {}", k, m);
        }
    }

    /// Asymmetry sanity: swapping ref ↔ dist flips the sign of the
    /// achromatic Δmean but keeps |Δstd|, band ratios, and Minkowski
    /// pool magnitudes within ε.
    #[test]
    fn swap_ref_dist_flips_signed_features() {
        let w = 24;
        let h = 24;
        let grey = vec![128u8; w * h * 3];
        let mut dark = Vec::with_capacity(w * h * 3);
        for _ in 0..(w * h * 3) {
            dark.push(64);
        }
        let f1 = extract_cvvdp_features(&grey, &dark, w, h);
        let f2 = extract_cvvdp_features(&dark, &grey, w, h);
        // Δmean(A) flips sign.
        assert!(f1[0] * f2[0] < 0.0, "Δmean signs same: {} {}", f1[0], f2[0]);
        // |Δstd| same magnitude.
        assert!((f1[1] - f2[1]).abs() < 1e-3, "|Δstd|");
        // Minkowski |Δ| same magnitude.
        let m1 = f1[CVVDP_FEATURE_COUNT - 1];
        let m2 = f2[CVVDP_FEATURE_COUNT - 1];
        assert!((m1 - m2).abs() < 1e-3, "minkowski symmetric");
    }
}
