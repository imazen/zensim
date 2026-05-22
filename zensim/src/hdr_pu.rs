//! HDR / PU (perceptually-uniform) primitives.
//!
//! Single useful artifact salvaged from the abandoned 2026-05
//! acumen / castleCSF Mode B exploration: a typed
//! [`ViewingCondition`] describing display + surround, plus a
//! simple adaptation-luminance estimator suitable for HDR-aware
//! preprocessing (e.g., PQ→PU encoding, peak-luminance-aware
//! tonemapping).
//!
//! The full castleCSF LUT + Mode B preprocessor lived on
//! `abandoned/feat-acumen-foundation` and `abandoned/feat-acumen-gpu`
//! between 2026-05-21 and 2026-05-22. They are preserved on those
//! branches for forensic reference and are documented in
//! `zensim/benchmarks/acumen_mode_b_production_recipe_2026-05-22.md`
//! (NEGATIVE result — Mode B-lite catastrophically drops CID22
//! 0.90 → 0.57 at V_24 production scale due to target/feature
//! mismatch).
//!
//! What stays here is the small, generally-useful subset for
//! future HDR work — no castleCSF dependency, no LUT, no
//! production runtime cost when unused.

/// Viewing condition presets + a free-form constructor.
///
/// Bundles display + surround parameters in their canonical units
/// (pixels-per-degree, cd/m²) so future HDR-aware code paths can
/// thread one well-typed value instead of separate floats.
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
    /// 5 cd/m² ambient.
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
    /// Returns `image_mean_nits` clamped against the display peak,
    /// blended with the ambient surround. The clamp prevents
    /// pathological inputs (e.g., near-zero L from a fully-black
    /// image) from driving downstream HDR math into log-space
    /// singularities.
    #[inline]
    pub fn adapted_luminance_nits(&self, image_mean_nits: f32) -> f32 {
        // Floor at 1% of peak to avoid log10(0); ceil at peak to
        // avoid headroom-exceeding inputs producing extrapolation.
        let floor = (self.peak_luminance_nits * 0.01).max(0.01);
        let raw = image_mean_nits.max(floor).min(self.peak_luminance_nits);
        // Blend with ambient: 30% weight (Mantiuk 2024 §III.A
        // adaptation model uses ~25-30% surround contribution).
        raw * 0.7 + self.ambient_luminance_nits * 0.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_lie_in_sane_range() {
        for vc in [
            ViewingCondition::LAB_REFERENCE,
            ViewingCondition::DESKTOP_STANDARD,
            ViewingCondition::MOBILE_RETINA,
            ViewingCondition::HDR_REFERENCE_1000,
        ] {
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
        let l_zero = vc.adapted_luminance_nits(0.0);
        assert!(l_zero >= 1.0 * 0.7);
        let l_huge = vc.adapted_luminance_nits(1e6);
        assert!(l_huge <= vc.peak_luminance_nits * 0.7 + vc.ambient_luminance_nits * 0.3 + 1e-3);
        let l_full = vc.adapted_luminance_nits(vc.peak_luminance_nits);
        let expected = vc.peak_luminance_nits * 0.7 + vc.ambient_luminance_nits * 0.3;
        assert!((l_full - expected).abs() < 1e-3);
    }

    #[test]
    fn adapted_luminance_ambient_contribution() {
        let vc_dim = ViewingCondition::new(56.0, 100.0, 5.0);
        let vc_bright = ViewingCondition::new(56.0, 100.0, 50.0);
        let im_l = 50.0;
        let diff = vc_bright.adapted_luminance_nits(im_l)
            - vc_dim.adapted_luminance_nits(im_l);
        let expected = (50.0 - 5.0) * 0.3;
        assert!((diff - expected).abs() < 1e-3);
    }
}
