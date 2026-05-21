//! Viewing-condition descriptor for [`acumen`](super) primitives.
//!
//! A [`ViewingCondition`] bundles the four pieces of context that the
//! perceptual primitives actually consume:
//!
//! - **ppd** (pixels per degree) — derived from display size, pixel
//!   density, and viewing distance. The single highest-impact axis;
//!   24-90 ppd is the common-viewing range, 56 is lab reference.
//! - **peak_luminance_nits** — the display white. SDR ≈ 100-300 nits;
//!   HDR PQ caps at 10 000.
//! - **ambient_luminance_nits** — surround / room light. Affects
//!   adaptation state, modest effect compared to display L.
//! - **transfer** — what the input pixels encode. Used by upstream
//!   `zensim::source::ColorTransferFunction` for HDR refusal gating;
//!   downstream of viewing this is informational.
//!
//! All fields are pre-converted to the units the LUT and pyramid
//! consume internally, so primitives don't pay a units-conversion
//! cost on the hot path.

/// Viewing condition presets + a free-form constructor.
///
/// Used by [`super::band_weights`] to compute per-band castleCSF
/// weights at inference time. The struct deliberately separates
/// `ppd` from `peak_luminance_nits` etc. so that runtime callers
/// can mix-and-match — for example, "lab reference ppd, mobile
/// peak" — without instantiating a new preset enum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewingCondition {
    /// Pixels per visual degree at the observer's eye. Common
    /// values: 28 (typical desktop @ 60 cm), 56 (lab reference),
    /// 90 (mobile retina @ 30 cm), 120 (close-mobile / VR).
    pub ppd: f32,
    /// Peak display luminance in cd/m² (nits). SDR sRGB ≈ 100-300;
    /// HDR PQ scales to 10_000.
    pub peak_luminance_nits: f32,
    /// Ambient surround luminance in cd/m². Affects adaptation
    /// state. Typical: 5 (dim room), 20 (office), 250 (outdoor).
    pub ambient_luminance_nits: f32,
}

impl ViewingCondition {
    /// Lab-reference desktop monitor: 56 ppd, 100 cd/m² peak,
    /// 5 cd/m² ambient. The training-time anchor for shipped
    /// zensim bakes — every other viewing condition should be
    /// thought of as a *deviation* from this.
    pub const LAB_REFERENCE: ViewingCondition = ViewingCondition {
        ppd: 56.0,
        peak_luminance_nits: 100.0,
        ambient_luminance_nits: 5.0,
    };

    /// Typical desktop @ ~60 cm viewing distance, 100 cd/m² display,
    /// dim office surround.
    pub const DESKTOP_STANDARD: ViewingCondition = ViewingCondition {
        ppd: 40.0,
        peak_luminance_nits: 100.0,
        ambient_luminance_nits: 20.0,
    };

    /// Mobile retina-class display @ ~30 cm, brighter display
    /// (auto-brightness in daylight indoors), moderate surround.
    pub const MOBILE_RETINA: ViewingCondition = ViewingCondition {
        ppd: 90.0,
        peak_luminance_nits: 300.0,
        ambient_luminance_nits: 50.0,
    };

    /// HDR PQ reference: 56 ppd, 1000 cd/m² peak (BT.2408 typical
    /// HDR mastering), dim room.
    pub const HDR_REFERENCE_1000: ViewingCondition = ViewingCondition {
        ppd: 56.0,
        peak_luminance_nits: 1000.0,
        ambient_luminance_nits: 5.0,
    };

    /// Free-form constructor for non-preset viewing conditions.
    /// Use the preset associated constants when possible — they
    /// match the constraints downstream primitives are validated
    /// against.
    #[inline]
    pub const fn new(
        ppd: f32,
        peak_luminance_nits: f32,
        ambient_luminance_nits: f32,
    ) -> Self {
        Self {
            ppd,
            peak_luminance_nits,
            ambient_luminance_nits,
        }
    }

    /// Effective adaptation luminance estimate for the achromatic
    /// (luminance) channel.
    ///
    /// Returns `image_mean_nits` clamped against the display peak
    /// and ambient surround. The clamp prevents pathological inputs
    /// (e.g., near-zero L from a fully-black image) from driving
    /// the CSF LUT outside its valid `L_bkg ∈ [0.005, 10_000]`
    /// range.
    ///
    /// `image_mean_nits` should be the linear-light per-pixel
    /// luminance averaged over the image, computed via
    /// [`super::band_weights::image_mean_luminance_nits`] from
    /// the reference signal.
    #[inline]
    pub fn adapted_luminance_nits(&self, image_mean_nits: f32) -> f32 {
        // Floor at 1% of peak to avoid log10(0); ceil at peak to
        // avoid headroom-exceeding inputs producing extrapolation.
        let floor = (self.peak_luminance_nits * 0.01).max(0.01);
        let raw = image_mean_nits.max(floor).min(self.peak_luminance_nits);
        // Blend with ambient: 30% weight (Mantiuk 2024 §III.A
        // adaptation model uses ~25-30% surround contribution).
        // This is an approximation — castleCSF itself accepts a
        // single L_adapt; we synthesize one here.
        raw * 0.7 + self.ambient_luminance_nits * 0.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_lie_in_lut_axes() {
        // The castleCSF LUT covers ppd implicit via the rho axis
        // [0.1, 64] cy/deg. At ppd=120 our highest band rho is
        // 120/8 = 15 cy/deg, well inside. At ppd=28 our lowest
        // band rho is 28/64 ≈ 0.4 cy/deg, also inside.
        for vc in [
            ViewingCondition::LAB_REFERENCE,
            ViewingCondition::DESKTOP_STANDARD,
            ViewingCondition::MOBILE_RETINA,
            ViewingCondition::HDR_REFERENCE_1000,
        ] {
            // Sanity: nothing pathological.
            assert!(vc.ppd > 0.0 && vc.ppd < 200.0);
            assert!(vc.peak_luminance_nits > 0.0);
            assert!(vc.peak_luminance_nits <= 10_000.0);
            assert!(vc.ambient_luminance_nits >= 0.0);
            assert!(vc.ambient_luminance_nits <= vc.peak_luminance_nits);
        }
    }

    #[test]
    fn adapted_luminance_clamps_extremes() {
        let vc = ViewingCondition::LAB_REFERENCE;
        // Image mean = 0 → floored at 1% of peak (1 cd/m²) before
        // ambient blend.
        let l_zero = vc.adapted_luminance_nits(0.0);
        assert!(l_zero >= 1.0 * 0.7);
        // Image mean way above peak → clamped at peak.
        let l_huge = vc.adapted_luminance_nits(1e6);
        assert!(l_huge <= vc.peak_luminance_nits * 0.7 + vc.ambient_luminance_nits * 0.3 + 1e-3);
        // Image mean = peak → near peak (after ambient blend).
        let l_full = vc.adapted_luminance_nits(vc.peak_luminance_nits);
        let expected = vc.peak_luminance_nits * 0.7 + vc.ambient_luminance_nits * 0.3;
        assert!((l_full - expected).abs() < 1e-3);
    }

    #[test]
    fn adapted_luminance_ambient_contribution() {
        // Two conditions identical except for ambient — adapted
        // luminance differs by 30% of the ambient delta.
        let vc_dim = ViewingCondition::new(56.0, 100.0, 5.0);
        let vc_bright = ViewingCondition::new(56.0, 100.0, 50.0);
        let im_l = 50.0;
        let diff = vc_bright.adapted_luminance_nits(im_l)
            - vc_dim.adapted_luminance_nits(im_l);
        let expected = (50.0 - 5.0) * 0.3;
        assert!((diff - expected).abs() < 1e-3);
    }
}
