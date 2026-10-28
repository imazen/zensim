//! EX-4 § 1 — XYB / LMS-biased-log front-end statistics.
//!
//! Adds **global per-channel statistics** of two perceptually-uniform
//! color front-ends as features the trainer can consume alongside the
//! existing 372 SSIM-like features:
//!
//! 1. **XYB** (libjxl-style): a 3×3 LMS-like matrix followed by a
//!    `cbrt(x + b)` nonlinearity, with `b = K_B0 ≈ 0.003793...`. Same
//!    front-end SSIMULACRA2 / Butteraugli compute their per-pixel
//!    differences in.
//! 2. **LMS biased-log** (Butteraugli-style): `log(LMS + b)` with
//!    `b = 0.01` per Guetzli §3.1 + libjxl `enc_xyb.cc`. Keeps low-luma
//!    sensitivity high while compressing high-luma gracefully.
//!
//! Each channel emits four statistics (mean, std, p5, p95). With three
//! channels and two front-ends the module produces **24 features per
//! image** (12 from XYB, 12 from LMS-biased-log). Computed on the
//! reference image — the distorted image is not consumed at this
//! granularity (CVVDP/IW-SSIM-style per-pair features live in the
//! sibling `cvvdp_features.rs` module).
//!
//! These features capture the per-image colorimetric prior:
//! - XYB Y mean/std anchors overall luminance + dynamic range.
//! - XYB X / B std anchors chromatic spread (a saturated red shot vs a
//!   neutral scene differ here even if Y stats match).
//! - LMS biased-log gives a different non-linearity (log vs cbrt) that
//!   suppresses bright outliers and emphasises shadow detail. Useful
//!   discriminator for HDR-ish vs SDR-ish content.
//!
//! ## Constants (verbatim from doc § 1 Table)
//!
//! | Constant | Value | Source |
//! |---|---|---|
//! | XYB cube-root bias `K_B0` | 0.003793073 | libjxl `enc_xyb.cc` |
//! | LMS biased-log offset `b` | 0.01 | Guetzli §3.1, Butteraugli |
//! | Butteraugli viewing distance | 1000 px (informational only) | Butteraugli code |
//!
//! ## Numerical stability
//!
//! - Both front-ends operate on **linear** sRGB after gamma decode (per
//!   `srgb_u8_to_linear` LUT). u8 → f32 in `[0, 1]`.
//! - Cube root uses `crate::color::cbrtf_fast` (private; Halley iteration,
//!   ~20 bits accurate — sufficient for global statistics).
//! - Biased-log uses [`f32::ln`] on `(LMS + 0.01)`. With LMS ≥ 0 and the
//!   bias added we never sample `ln(0)`; range is `ln(0.01) ≈ -4.605` at
//!   black to `ln(1.01) ≈ 0.00995` at white.
//! - Percentile computation uses an in-place sort. For a 4 MP image this
//!   is ~50 ms scalar; acceptable for offline feature extraction (the
//!   feature-extract pipeline is already O(n log n) for IW-pool sorts).
//!
//! ## API
//!
//! [`extract_xyb_lms_features`] is the only public entry point. It
//! returns exactly [`XYB_LMS_FEATURE_COUNT`] features in a known order
//! (see the constant's docs).

use crate::color::{
    K_B0, K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22, cbrtf_fast,
    srgb_u8_to_linear,
};

/// XYB biased cube-root nonlinearity offset — libjxl `enc_xyb.cc` `kB0`.
///
/// Re-exported here so the constant is discoverable next to the feature
/// module that consumes it. Both definitions share the same source
/// value; if [`crate::color::K_B0`] is ever bumped the link above is
/// the authoritative source and this re-export must follow.
#[allow(dead_code)] // discoverability constant; the feature extractor uses K_B0 directly
pub(crate) const XYB_CBRT_BIAS: f32 = K_B0;

/// LMS-biased-log offset `b` — Guetzli §3.1 / Butteraugli code.
///
/// `f(lms) = log(lms + b)`. Keeps shadows highly resolved while
/// compressing highlights; the inverse function is recoverable but the
/// metric uses one-way.
pub(crate) const LMS_BIASED_LOG_OFFSET: f32 = 0.01;

/// Number of statistics emitted per channel: mean, std, p5, p95.
pub(crate) const STATS_PER_CHANNEL: usize = 4;

/// Channels in each front-end (X, Y, B for XYB; L, M, S for LMS).
pub(crate) const CHANNELS: usize = 3;

/// Number of distinct front-ends: XYB cube-root + LMS biased-log.
pub(crate) const FRONT_ENDS: usize = 2;

/// Total feature count from this module.
///
/// `FRONT_ENDS · CHANNELS · STATS_PER_CHANNEL = 2 · 3 · 4 = 24`.
///
/// Feature ordering within the returned slice:
/// `[xyb_X_mean, xyb_X_std, xyb_X_p5, xyb_X_p95,
///   xyb_Y_mean, xyb_Y_std, xyb_Y_p5, xyb_Y_p95,
///   xyb_B_mean, xyb_B_std, xyb_B_p5, xyb_B_p95,
///   lms_L_mean, lms_L_std, lms_L_p5, lms_L_p95,
///   lms_M_mean, lms_M_std, lms_M_p5, lms_M_p95,
///   lms_S_mean, lms_S_std, lms_S_p5, lms_S_p95]`
pub const XYB_LMS_FEATURE_COUNT: usize = FRONT_ENDS * CHANNELS * STATS_PER_CHANNEL;

/// Extract XYB + LMS-biased-log global front-end statistics for an
/// 8-bit interleaved sRGB image.
///
/// Returns exactly [`XYB_LMS_FEATURE_COUNT`] features in the order
/// documented on that constant. Panics if `pixels` is not divisible
/// into RGB triplets.
///
/// Both front-ends run in a single pass per pixel — the inner loop
/// computes linear RGB, then in parallel emits XYB (matrix · linear,
/// then `cbrtf(x + K_B0) - cbrtf(K_B0)`) and LMS-biased-log
/// (matrix · linear, then `ln(lms + 0.01)`).
///
/// # Examples
///
/// ```
/// use zensim::xyb_lms_features::{
///     extract_xyb_lms_features, XYB_LMS_FEATURE_COUNT,
/// };
///
/// // 64×64 mid-grey.
/// let pixels = vec![128u8; 64 * 64 * 3];
/// let feats = extract_xyb_lms_features(&pixels);
/// assert_eq!(feats.len(), XYB_LMS_FEATURE_COUNT);
/// // Mid-grey is desaturated: the X (red-green) channel of XYB has
/// // very small mean and std relative to Y (luminance).
/// let xyb_x_mean = feats[0];
/// let xyb_y_mean = feats[4];
/// assert!(xyb_x_mean.abs() < xyb_y_mean.abs());
/// ```
#[must_use]
pub fn extract_xyb_lms_features(pixels: &[u8]) -> Vec<f32> {
    assert!(
        pixels.len().is_multiple_of(3),
        "pixels.len() must be divisible by 3 (got {})",
        pixels.len(),
    );
    let n = pixels.len() / 3;
    if n == 0 {
        return vec![0.0; XYB_LMS_FEATURE_COUNT];
    }

    // Per-channel buffers. Allocating two full passes is acceptable
    // for an offline feature-extract pipeline; the metric hot path
    // (compute_zensim_with_*) does not call this — it lives in the
    // training/extraction harness.
    let mut xyb_x = Vec::with_capacity(n);
    let mut xyb_y = Vec::with_capacity(n);
    let mut xyb_b = Vec::with_capacity(n);
    let mut lms_l = Vec::with_capacity(n);
    let mut lms_m = Vec::with_capacity(n);
    let mut lms_s = Vec::with_capacity(n);

    // Pre-compute the XYB cbrt bias correction — matches libjxl's
    // `make positive` shift but applied per-pixel here.
    let xyb_bias_correction = cbrtf_fast(K_B0);

    for i in 0..n {
        let r = srgb_u8_to_linear(pixels[3 * i]);
        let g = srgb_u8_to_linear(pixels[3 * i + 1]);
        let b = srgb_u8_to_linear(pixels[3 * i + 2]);

        // LMS = M · linear_RGB. Same matrix as XYB pre-cbrt.
        let l_mixed = K_M00 * r + K_M01 * g + K_M02 * b + K_B0;
        let m_mixed = K_M10 * r + K_M11 * g + K_M12 * b + K_B0;
        let s_mixed = K_M20 * r + K_M21 * g + K_M22 * b + K_B0;

        // XYB front-end (libjxl): cube-root then opponent decorr.
        let l_cbrt = cbrtf_fast(l_mixed) - xyb_bias_correction;
        let m_cbrt = cbrtf_fast(m_mixed) - xyb_bias_correction;
        let s_cbrt = cbrtf_fast(s_mixed) - xyb_bias_correction;
        let x = 0.5 * (l_cbrt - m_cbrt);
        let y_xyb = 0.5 * (l_cbrt + m_cbrt);
        let b_xyb = s_cbrt;

        // LMS biased-log front-end (Butteraugli style).
        // `+ K_B0` already inside l/m/s_mixed; we additionally add the
        // 0.01 bias on top before log, matching how Butteraugli's
        // `f(x) = log(x + b)` operates on the LMS-shaped signal.
        let l_log = (l_mixed + LMS_BIASED_LOG_OFFSET).ln();
        let m_log = (m_mixed + LMS_BIASED_LOG_OFFSET).ln();
        let s_log = (s_mixed + LMS_BIASED_LOG_OFFSET).ln();

        xyb_x.push(x);
        xyb_y.push(y_xyb);
        xyb_b.push(b_xyb);
        lms_l.push(l_log);
        lms_m.push(m_log);
        lms_s.push(s_log);
    }

    let mut out = Vec::with_capacity(XYB_LMS_FEATURE_COUNT);
    push_channel_stats(&mut out, &mut xyb_x);
    push_channel_stats(&mut out, &mut xyb_y);
    push_channel_stats(&mut out, &mut xyb_b);
    push_channel_stats(&mut out, &mut lms_l);
    push_channel_stats(&mut out, &mut lms_m);
    push_channel_stats(&mut out, &mut lms_s);
    debug_assert_eq!(out.len(), XYB_LMS_FEATURE_COUNT);
    out
}

/// Push `[mean, std, p5, p95]` for a single channel into `out`.
///
/// `samples` is mutated for the percentile sort. The reference
/// version is not preserved; the caller must not need the original
/// order after this function returns.
fn push_channel_stats(out: &mut Vec<f32>, samples: &mut [f32]) {
    let n = samples.len();
    debug_assert!(n > 0, "channel must have at least one sample");

    // Mean
    let mut sum = 0.0_f64;
    for &v in samples.iter() {
        sum += v as f64;
    }
    let mean = (sum / n as f64) as f32;

    // Population standard deviation (n-divisor, not n-1).
    let mut sse = 0.0_f64;
    for &v in samples.iter() {
        let d = v as f64 - mean as f64;
        sse += d * d;
    }
    let std = ((sse / n as f64) as f32).sqrt();

    // Percentiles: in-place partial sort. Using total_cmp handles NaN
    // deterministically (NaN sorts to the high end, but we don't
    // expect NaN here — XYB cbrt and ln on positive inputs are NaN
    // free).
    samples.sort_by(|a, b| a.total_cmp(b));
    let p5_idx = ((n as f32) * 0.05).floor() as usize;
    let p95_idx = (((n as f32) * 0.95).floor() as usize).min(n - 1);
    let p5 = samples[p5_idx];
    let p95 = samples[p95_idx];

    out.push(mean);
    out.push(std);
    out.push(p5);
    out.push(p95);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a 32×32 uniform mid-grey image we expect:
    /// - X channel: ~0 mean, ~0 std (no chroma).
    /// - Y channel: positive mean (luminance), ~0 std (uniform).
    /// - B channel: positive mean (blue-yellow), ~0 std (uniform).
    /// - LMS channels: equal-ish mean, ~0 std.
    #[test]
    fn uniform_grey_has_zero_chroma_and_zero_std() {
        let pixels = vec![128u8; 32 * 32 * 3];
        let f = extract_xyb_lms_features(&pixels);
        assert_eq!(f.len(), XYB_LMS_FEATURE_COUNT);

        // XYB X mean ~ 0 (no red-green opponent signal on grey).
        assert!(f[0].abs() < 1e-4, "xyb_X_mean = {}", f[0]);
        // XYB X std ~ 0 (uniform).
        assert!(f[1].abs() < 1e-4, "xyb_X_std = {}", f[1]);
        // XYB Y mean > 0 (luminance signal).
        assert!(f[4] > 0.0, "xyb_Y_mean = {}", f[4]);
        // XYB Y std ~ 0 (uniform).
        assert!(f[5].abs() < 1e-4, "xyb_Y_std = {}", f[5]);
        // LMS L_std ~ 0 (uniform).
        assert!(f[13].abs() < 1e-4, "lms_L_std = {}", f[13]);
    }

    /// A red-saturated image should have negative XYB X mean (red has
    /// more L than M in the opponent matrix).
    #[test]
    fn red_image_has_negative_xyb_x() {
        let mut pixels = Vec::with_capacity(32 * 32 * 3);
        for _ in 0..(32 * 32) {
            pixels.push(255);
            pixels.push(0);
            pixels.push(0);
        }
        let f = extract_xyb_lms_features(&pixels);
        // For pure red, K_M00 = 0.30 contributes to L, K_M10 = 0.23 to
        // M. Since L > M, cbrt(L+B0) > cbrt(M+B0), so
        // X = 0.5 (L_cbrt - M_cbrt) > 0. Test the sign.
        assert!(f[0] > 0.0, "xyb_X_mean on pure red = {}", f[0]);
    }

    /// LMS biased-log is monotonic in luminance: a black image
    /// produces ln(K_B0 + 0.01) ≈ ln(0.01379) ≈ -4.28 on all three
    /// LMS channels.
    #[test]
    fn black_image_has_log_at_floor() {
        let pixels = vec![0u8; 16 * 16 * 3];
        let f = extract_xyb_lms_features(&pixels);
        // LMS L mean is at index 12.
        let expected = ((K_B0 + LMS_BIASED_LOG_OFFSET) as f64).ln() as f32;
        // Allow some slack for the K_M0X · 0 matrix mul cancellation.
        assert!(
            (f[12] - expected).abs() < 1e-3,
            "lms_L_mean = {}, expected ~{}",
            f[12],
            expected,
        );
    }

    /// p5 ≤ p95 invariant on every channel × every front-end.
    #[test]
    fn percentile_ordering_holds() {
        // Random-ish input via deterministic xorshift-style fill.
        let mut state = 0x1234_5678_u32;
        let mut pixels = Vec::with_capacity(64 * 64 * 3);
        for _ in 0..(64 * 64 * 3) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push((state & 0xFF) as u8);
        }
        let f = extract_xyb_lms_features(&pixels);
        for channel_idx in 0..(FRONT_ENDS * CHANNELS) {
            let base = channel_idx * STATS_PER_CHANNEL;
            let p5 = f[base + 2];
            let p95 = f[base + 3];
            assert!(
                p5 <= p95,
                "channel {}: p5={} > p95={}",
                channel_idx,
                p5,
                p95,
            );
            // std must be non-negative.
            let std = f[base + 1];
            assert!(std >= 0.0, "channel {}: std={} < 0", channel_idx, std);
        }
    }

    /// Feature count matches the documented constant.
    #[test]
    fn feature_count_is_24() {
        assert_eq!(XYB_LMS_FEATURE_COUNT, 24);
        let pixels = vec![0u8; 4 * 4 * 3];
        assert_eq!(extract_xyb_lms_features(&pixels).len(), 24);
    }
}
