//! Transfer functions and the display model: mapping pixel code values to
//! **absolute emitted display luminance** (cd/m²).
//!
//! This is the enabling front-end for HDR-aware scoring. Today zensim scores
//! in relative linear-sRGB `[0, 1]`; an HDR-correct metric must first map code
//! values to absolute luminance so that luminance-dependent contrast
//! sensitivity (and, on the HDR path, PU encoding) can be applied. The full
//! roadmap and citations live in `docs/HDR_PLAN.md`.
//!
//! Every constant here is a public ITU-R / SMPTE / IEC specification value
//! (ITU-R BT.2100, SMPTE ST 2084, IEC 61966-2-1); the functions are
//! reimplemented from those specs. The reference-parity tests at the bottom
//! pin each function to published golden values and to the display models in
//! the locally-verified `pycvvdp` 0.5.4 reference.
//!
//! This module is internal (`pub(crate)`); a public HDR API is added in a
//! later chunk once the scoring path that consumes it lands.
//!
//! Foundation-ahead-of-consumer: the scoring path that uses these does not
//! land until the PU front-end chunk, so the items are `allow(dead_code)` for
//! now. They are fully exercised by the reference-parity tests below.
#![allow(dead_code)]

/// IEC 61966-2-1 sRGB EOTF: sRGB-encoded `v ∈ [0, 1]` → relative linear
/// light `[0, 1]`.
#[inline]
pub(crate) fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.040_449_936 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// SMPTE ST 2084 (PQ) EOTF: PQ-encoded `v ∈ [0, 1]` → **absolute** luminance
/// in cd/m² over `[0, 10000]`.
///
/// Constants are the ST 2084 rational-polynomial coefficients
/// (`m1 = 2610/16384`, `m2 = 2523/4096·128`, `c1 = 3424/4096`,
/// `c2 = 2413/4096·32`, `c3 = 2392/4096·32`).
#[inline]
pub(crate) fn pq_eotf(v: f32) -> f32 {
    const L_MAX: f32 = 10000.0;
    const M1: f32 = 0.159_301_75; // 2610 / 16384
    const M2: f32 = 78.843_75; // 2523 / 4096 * 128
    const C1: f32 = 0.835_937_5; // 3424 / 4096
    const C2: f32 = 18.851_562; // 2413 / 4096 * 32
    const C3: f32 = 18.687_5; // 2392 / 4096 * 32

    let im = v.powf(1.0 / M2);
    let num = (im - C1).max(0.0);
    let den = C2 - C3 * im;
    L_MAX * (num / den).powf(1.0 / M1)
}

/// ITU-R BT.2100 HLG inverse-OETF: HLG-encoded `v ∈ [0, 1]` → scene-relative
/// linear `[0, 12]` **per channel**.
///
/// The HLG OOTF (system gamma, which converts scene-relative to
/// display-referred light) depends on the luminance of the whole RGB triple,
/// so it is applied at the color stage, not here. See [`hlg_system_gamma`].
#[inline]
pub(crate) fn hlg_inverse_oetf(v: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 1.0 - 4.0 * A; // 0.28466892
    // c = 0.5 − a·ln(4a) ≈ 0.55991073
    const C: f32 = 0.559_910_7;
    if v <= 0.5 {
        (v * v) / 3.0
    } else {
        (((v - C) / A).exp() + B) / 12.0
    }
}

/// HLG system gamma (ITU-R BT.2100 / BBC WHP 369): `1.2` at a 1000 cd/m² peak,
/// with a luminance term and an ambient-light correction above that.
#[inline]
pub(crate) fn hlg_system_gamma(y_peak: f32, e_ambient_lux: f32) -> f32 {
    if y_peak <= 1000.0 {
        1.2
    } else {
        let amb = if e_ambient_lux > 0.0 {
            e_ambient_lux
        } else {
            5.0
        };
        1.2 + 0.42 * (y_peak / 1000.0).log10() - 0.076_23 * (amb / 5.0).log10()
    }
}

/// The physical display the metric assumes the image is shown on: peak and
/// black emitted luminance plus the ambient light reflected off the screen,
/// all in cd/m². These three numbers turn relative pixel values into the
/// absolute luminance the eye actually adapts to.
///
/// The presets mirror `pycvvdp` 0.5.4 display models; `STANDARD_4K` is the
/// same SDR display zensim's CVVDP feature path already assumes
/// (`cvvdp_features.rs`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DisplayModel {
    /// Peak emitted luminance (white), cd/m².
    pub y_peak: f32,
    /// Black emitted luminance (leakage), cd/m².
    pub y_black: f32,
    /// Ambient light reflected off the screen, cd/m² (`E_ambient · k / π`).
    pub y_refl: f32,
}

impl DisplayModel {
    /// `pycvvdp` `standard_4k`: 200 cd/m² peak, 0.2 black, 250 lux ambient
    /// (`250 · 0.005 / π ≈ 0.3979`). The SDR reference display.
    pub(crate) const STANDARD_4K: Self = Self {
        y_peak: 200.0,
        y_black: 0.2,
        y_refl: 0.397_887_36,
    };

    /// A 1000 cd/m² PQ HDR reference display (BT.2100 grade-1000).
    pub(crate) const STANDARD_HDR_PQ_1000: Self = Self {
        y_peak: 1000.0,
        y_black: 0.005,
        y_refl: 0.397_887_36,
    };

    /// Map relative linear light `[0, 1]` (e.g. the output of [`srgb_eotf`]) to
    /// absolute emitted luminance:
    /// `L = (peak − black)·lin + black + reflection`.
    #[inline]
    pub(crate) fn sdr_linear_to_luminance(&self, lin: f32) -> f32 {
        (self.y_peak - self.y_black) * lin + self.y_black + self.y_refl
    }

    /// Map a PQ-encoded code value to absolute emitted luminance, clamped to
    /// what the display can physically reproduce and lifted by black +
    /// reflected ambient. A 1000 cd/m² display cannot show PQ's 10000 cd/m²
    /// peak, so the highlight clamps to `y_peak`.
    #[inline]
    pub(crate) fn pq_to_luminance(&self, v: f32) -> f32 {
        pq_eotf(v).min(self.y_peak) + self.y_black + self.y_refl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ── sRGB EOTF (IEC 61966-2-1) ──────────────────────────────────────────
    #[test]
    fn srgb_eotf_endpoints_and_midpoint() {
        assert_eq!(srgb_eotf(0.0), 0.0);
        assert!(close(srgb_eotf(1.0), 1.0, 1e-6));
        // 128/255 = 0.501961 → ~0.2159 linear (the classic "mid-gray" value).
        assert!(close(srgb_eotf(128.0 / 255.0), 0.215_861, 1e-4));
        // Linear segment below the 0.04045 break.
        assert!(close(srgb_eotf(0.02), 0.02 / 12.92, 1e-7));
    }

    // ── PQ EOTF (SMPTE ST 2084) — golden cd/m² values ──────────────────────
    #[test]
    fn pq_eotf_reference_values() {
        assert_eq!(pq_eotf(0.0), 0.0);
        // Published: PQ code 0.5 → ~92.25 cd/m².
        assert!(
            close(pq_eotf(0.5), 92.2466, 0.05),
            "pq_eotf(0.5) = {}",
            pq_eotf(0.5)
        );
        // Peak: code 1.0 → 10000 cd/m².
        assert!(
            close(pq_eotf(1.0), 10000.0, 1.0),
            "pq_eotf(1.0) = {}",
            pq_eotf(1.0)
        );
        // The code that encodes exactly 100 cd/m² (sRGB-ish reference white).
        assert!(
            close(pq_eotf(0.508_078), 100.0, 0.1),
            "pq_eotf(0.508078) = {}",
            pq_eotf(0.508_078)
        );
        // Monotone non-decreasing across the range.
        let mut prev = -1.0;
        for i in 0..=100 {
            let l = pq_eotf(i as f32 / 100.0);
            assert!(l >= prev, "PQ EOTF not monotone at v={}", i);
            prev = l;
        }
    }

    // ── HLG inverse-OETF (BT.2100) ─────────────────────────────────────────
    #[test]
    fn hlg_inverse_oetf_reference_values() {
        assert_eq!(hlg_inverse_oetf(0.0), 0.0);
        // The lower-segment join at v = 0.5 is 1/12.
        assert!(
            close(hlg_inverse_oetf(0.5), 1.0 / 12.0, 1e-5),
            "hlg(0.5) = {}",
            hlg_inverse_oetf(0.5)
        );
        // Upper segment reaches 1.0 (scene reference white) at v = 1.0.
        assert!(
            close(hlg_inverse_oetf(1.0), 1.0, 1e-4),
            "hlg(1.0) = {}",
            hlg_inverse_oetf(1.0)
        );
        // Continuity across the 0.5 segment break.
        let lo = hlg_inverse_oetf(0.499_999);
        let hi = hlg_inverse_oetf(0.500_001);
        assert!(
            close(lo, hi, 1e-4),
            "HLG discontinuous at 0.5: {lo} vs {hi}"
        );
    }

    #[test]
    fn hlg_system_gamma_values() {
        // 1.2 at and below a 1000 cd/m² peak.
        assert_eq!(hlg_system_gamma(1000.0, 200.0), 1.2);
        assert_eq!(hlg_system_gamma(600.0, 200.0), 1.2);
        // Brighter peak raises the gamma.
        assert!(hlg_system_gamma(4000.0, 200.0) > 1.2);
    }

    // ── Display model — SDR pixel → nits, zensim `standard_4k` convention ──
    #[test]
    fn sdr_display_model_golden_nits() {
        let d = DisplayModel::STANDARD_4K;
        // mid-gray 128/255 → 43.73 cd/m² (matches cvvdp_features.rs path).
        let mid = d.sdr_linear_to_luminance(srgb_eotf(128.0 / 255.0));
        assert!(close(mid, 43.73, 0.05), "mid-gray nits = {mid}");
        // White 255 → peak + black + reflection = 200.40 cd/m².
        let white = d.sdr_linear_to_luminance(srgb_eotf(1.0));
        assert!(close(white, 200.398, 0.05), "white nits = {white}");
        // Black 0 → black + reflection floor.
        let black = d.sdr_linear_to_luminance(srgb_eotf(0.0));
        assert!(close(black, 0.597_887, 1e-4), "black nits = {black}");
    }

    #[test]
    fn pq_display_model_clamps_to_peak() {
        let d = DisplayModel::STANDARD_HDR_PQ_1000;
        // PQ peak (10000) clamps to the display's 1000 cd/m² ceiling.
        let hi = d.pq_to_luminance(1.0);
        assert!(
            close(hi, 1000.0 + d.y_black + d.y_refl, 1e-3),
            "pq peak = {hi}"
        );
        // A mid PQ value below the ceiling passes through (+ floor).
        let mid = d.pq_to_luminance(0.5);
        assert!(
            close(mid, 92.2466 + d.y_black + d.y_refl, 0.05),
            "pq mid = {mid}"
        );
    }
}
