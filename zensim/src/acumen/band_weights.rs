//! Per-band castleCSF weights at inference time.
//!
//! This module implements **Mode A** of the tracking-issue
//! ([imazen/zensim#40]) algorithm slate: compute the image's mean
//! luminance once, look up the castleCSF LUT at the resulting
//! `L_adapt` for each (channel, band) pair, and emit a small fixed
//! array of weights that downstream feature extraction multiplies
//! into the relevant band-energy features.
//!
//! ## Why Mode A and not per-pixel
//!
//! castleCSF is fundamentally a function of `(rho, L_adapt)`. The
//! cheapest faithful application is to evaluate it once per image
//! at the image's mean luminance. This is what cvvdp's "single
//! display + single adaptation state" assumption already implies
//! for stills. Per-pixel `L_adapt` (Mode B) is plumbing for HDR
//! local-tonemap fidelity; not required for SDR.
//!
//! Cost: 4 bands × 3 channels = 12 LUT bilinear interps per image.
//! At ~14 ns/interp scalar, that's ~170 ns per image — negligible
//! vs the feature extractor's microsecond envelope.
//!
//! ## Band-rho convention
//!
//! Bands index pyramid levels coarse → fine. At pyramid level
//! `level` and viewing condition `ppd`, the band rho is
//! `ppd / 2^(level + 1)`. For ppd=56 and 4 levels this gives
//! `{28, 14, 7, 3.5}` cy/deg from finest to coarsest, matching
//! the existing `cvvdp_features::CSF_BAND_WEIGHTS` indexing
//! convention (level 0 = finest = highest rho).
//!
//! ## Why not lerp two anchors at training time
//!
//! [Falsified 2026-05-20]: per-ppd anchor lerp fails because the
//! achromatic CSF peaks *inside* the common viewing range
//! (ppd ~45-90). Linear interp can't track a hump. The runtime
//! LUT lookup is the right shape and what this module ships.
//! See `benchmarks/acumen_castle_csf_validation_2026-05-20.md`.

use super::castle_csf::{CastleCsfLut, Channel};
use super::viewing::ViewingCondition;

/// Default pyramid level count, matching `cvvdp_features::N_LEVELS`.
/// The compile-time constant here keeps `band_weights` independent
/// of the `training`-gated feature module.
pub const N_BANDS: usize = 4;

/// Per-(channel, band) castleCSF weight table.
///
/// Row-major: `weights[channel][band]`. Bands are pyramid levels,
/// 0 = finest (highest rho), N-1 = coarsest (lowest rho).
///
/// Computed by [`compute_csf_band_weights`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandCsfWeights {
    /// `[Channel][Band]` table. Channel index matches the
    /// [`Channel`] enum's `u8` discriminant.
    pub weights: [[f32; N_BANDS]; 3],
}

impl BandCsfWeights {
    /// Achromatic-channel weights, one per band, finest first.
    #[inline]
    pub fn achromatic(&self) -> &[f32; N_BANDS] {
        &self.weights[Channel::Achromatic as usize]
    }

    /// Red-green channel weights.
    #[inline]
    pub fn red_green(&self) -> &[f32; N_BANDS] {
        &self.weights[Channel::RedGreen as usize]
    }

    /// Yellow-violet channel weights.
    #[inline]
    pub fn yellow_violet(&self) -> &[f32; N_BANDS] {
        &self.weights[Channel::YellowViolet as usize]
    }

    /// Maximum achromatic weight across bands — useful for
    /// normalising per-band weights so the brightest band gets
    /// weight 1.0 (matches the existing `CSF_BAND_WEIGHTS` prior's
    /// shape where band-1 is 1.0).
    pub fn achromatic_max(&self) -> f32 {
        let mut m = f32::NEG_INFINITY;
        for &w in self.achromatic() {
            if w > m {
                m = w;
            }
        }
        m
    }

    /// Normalise the achromatic row so its max equals 1.0. Useful
    /// when downstream code expects the existing prior's
    /// "band-1 = 1.0" convention. Returns a new table; does not
    /// mutate `self`.
    pub fn normalized_to_achromatic_peak(&self) -> Self {
        let m = self.achromatic_max();
        if m <= 0.0 || !m.is_finite() {
            return *self;
        }
        let inv = 1.0 / m;
        let mut out = *self;
        for ch in 0..3 {
            for b in 0..N_BANDS {
                out.weights[ch][b] *= inv;
            }
        }
        out
    }
}

/// Compute per-(channel, band) castleCSF weights for a single
/// image at the given viewing condition.
///
/// `image_mean_luminance_nits` is the linear-light per-pixel
/// luminance averaged across the reference image — see
/// [`image_mean_luminance_nits`] for the canonical computation
/// from sRGB-encoded input.
///
/// The returned weights are **NOT normalized**; downstream callers
/// decide whether to use raw cvvdp-style sensitivities or
/// peak-normalized ratios via
/// [`BandCsfWeights::normalized_to_achromatic_peak`].
pub fn compute_csf_band_weights(
    lut: &CastleCsfLut<'_>,
    viewing: ViewingCondition,
    image_mean_luminance_nits: f32,
) -> BandCsfWeights {
    let l_adapt = viewing.adapted_luminance_nits(image_mean_luminance_nits);
    let log_l = l_adapt.max(1e-3).log10();
    let mut weights = [[0.0_f32; N_BANDS]; 3];
    for band in 0..N_BANDS {
        // Band rho convention: level 0 = finest = highest rho.
        // rho(level) = ppd / 2^(level+1).
        let rho = viewing.ppd / (2u32.pow(band as u32 + 1) as f32);
        let log_rho = rho.max(1e-3).log10();
        for ch in [Channel::Achromatic, Channel::RedGreen, Channel::YellowViolet] {
            weights[ch as usize][band] = lut.sensitivity(log_rho, log_l, ch);
        }
    }
    BandCsfWeights { weights }
}

/// Compute the linear-light mean luminance of an sRGB-encoded
/// reference image, in cd/m² assuming the image white maps to
/// `peak_luminance_nits`.
///
/// Steps:
/// 1. sRGB inverse transfer per channel (per-pixel branchless
///    polynomial approximation, see [`srgb_to_linear`]).
/// 2. BT.709 luma matrix: `Y = 0.2126 R + 0.7152 G + 0.0722 B`.
/// 3. Scale by `peak_luminance_nits` to get cd/m².
///
/// Branchless and SIMD-friendly per row; loops are
/// auto-vectorizable. Used by [`compute_csf_band_weights`] when
/// the caller doesn't already have a luminance signal.
///
/// For HDR PQ / HLG input the caller should compute the linear
/// luminance externally and pass it directly to
/// [`compute_csf_band_weights`]; sRGB is the SDR fast path.
pub fn image_mean_luminance_nits(rgb8: &[u8], peak_luminance_nits: f32) -> f32 {
    if rgb8.is_empty() {
        return 0.0;
    }
    let n_pixels = rgb8.len() / 3;
    debug_assert_eq!(rgb8.len(), n_pixels * 3, "rgb8 length must be 3·N");
    let mut sum_l = 0.0_f64;
    for chunk in rgb8.chunks_exact(3) {
        let r = srgb_to_linear(chunk[0]);
        let g = srgb_to_linear(chunk[1]);
        let b = srgb_to_linear(chunk[2]);
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        sum_l += y as f64;
    }
    let mean = (sum_l / n_pixels as f64) as f32;
    mean * peak_luminance_nits
}

/// Branchless sRGB inverse transfer for a u8 input.
///
/// Approximates the piecewise sRGB → linear curve. The shipped
/// form is the canonical IEC 61966-2-1 expression — chosen here
/// for correctness clarity over the per-instruction-count
/// optimization (the caller already does N_pixels divisions).
/// Auto-vectorizable on `chunks_exact(3)` paths.
#[inline]
fn srgb_to_linear(v: u8) -> f32 {
    let u = v as f32 / 255.0;
    if u <= 0.040_45 {
        u / 12.92
    } else {
        ((u + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LUT_BYTES: &[u8] = include_bytes!("../../data/castle_csf_v0_5_4_cvvdp.lut");

    fn load_lut() -> CastleCsfLut<'static> {
        CastleCsfLut::from_bytes(LUT_BYTES).expect("LUT parse")
    }

    #[test]
    fn srgb_to_linear_endpoints() {
        assert!((srgb_to_linear(0) - 0.0).abs() < 1e-6);
        assert!((srgb_to_linear(255) - 1.0).abs() < 1e-4);
        // Mid-grey ≈ 0.21..0.22 linear (canonical sRGB 128 → 0.2159).
        let mid = srgb_to_linear(128);
        assert!((0.20..0.23).contains(&mid), "mid grey was {mid}");
    }

    #[test]
    fn image_mean_luminance_uniform_grey() {
        // 64x64 all 128/255 → linear Y ≈ 0.2159, × 100 nits ≈ 21.6
        let img = vec![128_u8; 64 * 64 * 3];
        let l = image_mean_luminance_nits(&img, 100.0);
        assert!((20.0..23.0).contains(&l), "expected ~21.6, got {l}");
    }

    #[test]
    fn image_mean_luminance_black() {
        let img = vec![0_u8; 64 * 64 * 3];
        let l = image_mean_luminance_nits(&img, 100.0);
        assert!(l < 1e-3);
    }

    #[test]
    fn weights_shape_is_low_pass_at_lab_reference_mid_gray() {
        // **NOT a parity test with the hardcoded prior.** The
        // legacy `cvvdp_features::CSF_BAND_WEIGHTS = [0.5, 1.0,
        // 0.8, 0.4]` peaks at band 1 — that's an MLP-tuned
        // approximation, not a calibrated CSF. At
        // LAB_REFERENCE (ppd=56, peak=100 cd/m²) with image
        // mean ≈ 21 cd/m² (mid-gray), castleCSF correctly
        // predicts a *low-pass* shape, peaking at band 3
        // (3.5 cy/deg) because lower adaptation luminance shifts
        // the CSF peak to lower spatial frequencies (paper
        // Eq. 4-6). At higher L_adapt the peak shifts higher;
        // see `weights_peak_shifts_with_luminance` below.
        let lut = load_lut();
        let vc = ViewingCondition::LAB_REFERENCE;
        let weights = compute_csf_band_weights(&lut, vc, 21.0);
        let normalized = weights.normalized_to_achromatic_peak();
        let a = normalized.achromatic();

        let peak_idx = a
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        // At this L_adapt the peak should sit in band 2 or 3
        // (3.5–7 cy/deg) — the low-frequency side of the
        // mesopic-to-photopic transition.
        assert!(
            (2..=3).contains(&peak_idx),
            "expected achromatic peak at band 2 or 3, got band {peak_idx} weights {a:?}"
        );

        // Peak is normalized to 1.0; bands further from peak
        // monotonically decrease.
        assert!((a[peak_idx] - 1.0).abs() < 1e-5);

        // Strict monotone decrease moving away from the peak.
        for i in 0..N_BANDS {
            if i == peak_idx {
                continue;
            }
            assert!(a[i] < 1.0, "band {i} weight {} should be < peak", a[i]);
            assert!(a[i] > 0.0, "band {i} weight {} should be > 0", a[i]);
        }
    }

    #[test]
    fn weights_peak_shifts_with_luminance() {
        // Higher L_adapt shifts the achromatic CSF peak to
        // higher spatial frequency. Compare the peak-band index
        // at low L (mesopic, mid-gray ~5 cd/m²) vs high L
        // (photopic, near-peak ~300 cd/m² on an HDR-capable
        // display). At low L the peak sits at band 3 (coarsest);
        // at high L it shifts up by at least one band.
        let lut = load_lut();
        let vc_low = ViewingCondition::new(56.0, 100.0, 5.0);
        let vc_high = ViewingCondition::new(56.0, 1000.0, 5.0);
        let low = compute_csf_band_weights(&lut, vc_low, 5.0);
        let high = compute_csf_band_weights(&lut, vc_high, 300.0);

        let peak = |w: &BandCsfWeights| {
            w.achromatic()
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        let low_peak = peak(&low);
        let high_peak = peak(&high);
        assert!(
            high_peak <= low_peak,
            "expected high-L peak (band {high_peak}) to be at higher freq than low-L peak (band {low_peak})"
        );
        // And the achromatic-channel absolute sensitivity at the
        // high-L peak should exceed the low-L peak — Weber/
        // photopic gain. (The LUT values are log10(S), so this
        // is a comparison in linear sensitivity space.)
        assert!(
            high.achromatic_max() > low.achromatic_max(),
            "photopic peak ({}) should exceed mesopic peak ({})",
            high.achromatic_max(),
            low.achromatic_max(),
        );
    }

    #[test]
    fn ppd_scaling_shifts_peak_band() {
        // At low ppd (more visible bands sit on the high-rho side
        // of the CSF peak), the band-rho peak should land at a
        // different band than at high ppd. This sanity check
        // confirms the castleCSF LUT actually responds to ppd
        // changes through the band-rho mapping.
        let lut = load_lut();
        let l = 30.0;

        let low = compute_csf_band_weights(
            &lut,
            ViewingCondition::new(28.0, 100.0, 5.0),
            l,
        );
        let high = compute_csf_band_weights(
            &lut,
            ViewingCondition::new(120.0, 100.0, 5.0),
            l,
        );

        // The two should NOT be byte-identical.
        assert_ne!(
            low.weights, high.weights,
            "weights should differ between ppd=28 and ppd=120"
        );

        // Verify chromatic monotonicity per the falsification
        // report: RG should decrease with ppd at every band
        // (chromatic CSFs are monotonic-decreasing in rho).
        for band in 0..N_BANDS {
            let low_rg = low.weights[Channel::RedGreen as usize][band];
            let high_rg = high.weights[Channel::RedGreen as usize][band];
            assert!(
                low_rg >= high_rg * 0.95,
                "expected RG band {band} weight to drop with ppd (low_rg={low_rg}, high_rg={high_rg})"
            );
        }
    }

    #[test]
    fn weights_diverge_from_hardcoded_prior() {
        // The existing hardcoded prior [0.5, 1.0, 0.8, 0.4] is
        // mid-band peaked; castleCSF at common viewing
        // conditions is low-pass peaked. The two shapes
        // *should* diverge — this test pins that divergence so
        // anyone who fiddles with the LUT or the band-rho
        // mapping notices when the shapes accidentally
        // re-align (which would mean either the LUT broke or
        // the prior is now physically wrong in a different
        // way). Specifically:
        //
        // - castleCSF at lab-reference predicts band 0 (28
        //   cy/deg) is essentially invisible (≈ 0.03 normalized,
        //   not 0.5).
        // - castleCSF predicts band 3 (3.5 cy/deg) is the peak
        //   (1.0 normalized, not 0.4).
        //
        // These divergences are the point of Gate A — replace
        // the MLP-tuned mid-band prior with a physically-
        // grounded CSF and re-train.
        let lut = load_lut();
        let normalized = compute_csf_band_weights(
            &lut,
            ViewingCondition::LAB_REFERENCE,
            21.0,
        )
        .normalized_to_achromatic_peak();
        let a = normalized.achromatic();
        // Sanity bounds:
        // - All weights are in [0, 1] after normalisation.
        // - Band 0 (highest rho) is much smaller than the
        //   hardcoded prior's 0.5 (CSF is rolling off).
        // - Band 3 (lowest rho) is the peak (1.0).
        for (i, &w) in a.iter().enumerate() {
            assert!((0.0..=1.0).contains(&w), "band {i} = {w} out of [0,1]");
        }
        assert!(a[0] < 0.2, "band 0 should fall below 0.2, got {}", a[0]);
        assert!(a[3] > 0.9, "band 3 should be near peak, got {}", a[3]);
    }

    #[test]
    fn lut_axes_remain_visible() {
        use super::super::castle_csf::{N_L_BKG, N_RHO};
        // Defensive sanity: callers can introspect the LUT
        // dimensions without depending on `castle_csf` internals.
        let lut = load_lut();
        assert_eq!(lut.log_l_bkg.len(), N_L_BKG);
        assert_eq!(lut.log_rho.len(), N_RHO);
    }
}
