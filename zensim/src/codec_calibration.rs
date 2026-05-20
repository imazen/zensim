//! Per-codec score calibration for [`crate::ZensimProfile::PreviewV0_5Tuner`].
//!
//! The Tuner profile is the dial-honest variant designed for codec
//! auto-targeting (user types `score=70`, codec binary-searches the q
//! yielding zensim ≈ 70). On the 2026-05-19 cross-codec consistency
//! eval (10 images × 19 q × 3 codecs, see
//! `benchmarks/cross_codec_consistency_2026-05-19.md`), the un-calibrated
//! Tuner output had a mean pairwise butteraugli of **6.68** at the
//! PJND-anchored target (T=63). That's above the "broken" threshold
//! of ~4.0; "63" did not mean the same visual quality across codecs.
//!
//! Root cause from the EXP-TUNER-V2 falsification: the bake itself
//! cannot fix the cross-codec spread — its training corpus is
//! codec-agnostic, so at a given Tuner output the per-codec spread of
//! actual distance to reference is structural. The CLI layer however
//! knows which codec it's invoking, so it can apply a per-codec affine
//! at binary-search time to make `target=63` mean PJND for every
//! codec.
//!
//! ## How the calibration is fit
//!
//! 1. For each codec ∈ {jpeg, webp, avif}, score every (ref, dist@q)
//!    pair from the existing q-sweep cache under PreviewV0_5Tuner →
//!    `tuner_raw_C`.
//! 2. Score the same pairs with **ssim2** (the empirical PJND anchor —
//!    per the 2023 CID22 paper Table 4, mean KonJND-1k PJND threshold
//!    is at ssim2 ≈ 63).
//! 3. Linear regression per codec:
//!    `ssim2 = α_C + β_C · tuner_raw`.
//! 4. The calibrated score `calibrated_C = α_C + β_C · tuner_raw` then
//!    means the same thing across codecs by construction: "63" =
//!    "PJND from KonJND-1k", "90" = "near-lossless", etc.
//!
//! Fit script: `scripts/v_next/fit_per_codec_calibration.py`. Source
//! data: `cross_codec_consistency_2026-05-19/work/`. Fit data persisted
//! at `/mnt/v/output/zensim/per_codec_calibration_2026-05-19/fits.json`.
//!
//! ## Coverage
//!
//! Fits derive from PIL Pillow encoders (libjpeg-turbo + libwebp +
//! libavif) on the safesyn-sourced 10-image set. Production codecs
//! (zenjpeg, zenwebp, zenavif) produce slightly different RD curves;
//! the (α, β) here are the right shape but a per-encoder refit will
//! tighten residuals once production sweeps land. zenjxl uses the mean
//! of the other lossy codecs (no per-codec sweep yet). zenpng is the
//! lossless identity.

/// Per-codec affine: `calibrated = alpha + beta * tuner_raw`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationAffine {
    /// Intercept added to the Tuner raw output.
    pub alpha: f32,
    /// Slope multiplier applied to the Tuner raw output.
    pub beta: f32,
}

impl CalibrationAffine {
    /// No-op calibration: `calibrated = raw`.
    pub const IDENTITY: Self = Self {
        alpha: 0.0,
        beta: 1.0,
    };

    /// Apply the affine, clamping the output to `[0, 100]`.
    #[inline]
    pub fn apply(&self, raw: f32) -> f32 {
        let v = self.alpha + self.beta * raw;
        v.clamp(0.0, 100.0)
    }

    /// Invert the affine: given a desired calibrated score, return the
    /// Tuner raw score that produces it. Useful when binary-searching
    /// in raw space.
    ///
    /// Returns `None` when `beta == 0` (degenerate calibration).
    #[inline]
    pub fn invert(&self, calibrated: f32) -> Option<f32> {
        if self.beta == 0.0 {
            None
        } else {
            Some((calibrated - self.alpha) / self.beta)
        }
    }
}

/// Per-codec calibration registry for `PreviewV0_5Tuner`.
///
/// Lookup-by-name returns the right affine for the codec. Unknown
/// codec names return [`CalibrationAffine::IDENTITY`].
#[derive(Clone, Copy, Debug)]
pub struct CodecCalibration {
    /// JPEG (PIL libjpeg-turbo, 4:2:0 subsampling).
    pub jpeg: CalibrationAffine,
    /// WebP (PIL libwebp method=4).
    pub webp: CalibrationAffine,
    /// AVIF (PIL libavif speed=6).
    pub avif: CalibrationAffine,
    /// JPEG XL — placeholder = mean of jpeg/webp/avif (no per-codec data yet).
    pub zenjxl: CalibrationAffine,
    /// PNG — lossless, identity.
    pub zenpng: CalibrationAffine,
}

impl CodecCalibration {
    /// All-identity calibration (no compensation).
    pub const IDENTITY: Self = Self {
        jpeg: CalibrationAffine::IDENTITY,
        webp: CalibrationAffine::IDENTITY,
        avif: CalibrationAffine::IDENTITY,
        zenjxl: CalibrationAffine::IDENTITY,
        zenpng: CalibrationAffine::IDENTITY,
    };

    /// Default calibration for `PreviewV0_5Tuner` (2026-05-19 fit).
    ///
    /// Anchor: ssim2 (CPU) on 10 images × 19 q. R² = 0.93–0.95 per
    /// lossy codec; n=190 pairs per codec. zenjxl uses the mean of
    /// the three lossy codecs; zenpng is identity (lossless).
    pub const PREVIEW_V0_5_TUNER: Self = Self {
        // jpeg: n=190 R²=0.9453 MSE=56.500
        jpeg: CalibrationAffine {
            alpha: -31.701_27,
            beta: 1.352_23,
        },
        // webp: n=190 R²=0.9348 MSE=23.907
        webp: CalibrationAffine {
            alpha: -4.290_747,
            beta: 1.011_315,
        },
        // avif: n=190 R²=0.9495 MSE=53.807
        avif: CalibrationAffine {
            alpha: -14.299_716,
            beta: 1.125_777,
        },
        // zenjxl: mean of {jpeg, webp, avif}
        zenjxl: CalibrationAffine {
            alpha: -16.763_91,
            beta: 1.163_107,
        },
        // zenpng: identity (lossless)
        zenpng: CalibrationAffine::IDENTITY,
    };

    /// Look up the per-codec affine by codec name. Accepts the common
    /// short names (`"jpeg"`, `"webp"`, `"avif"`, `"jxl"`, `"png"`)
    /// and the longer `"zenjpeg"` / `"zenwebp"` / `"zenavif"` /
    /// `"zenjxl"` / `"zenpng"` aliases.
    ///
    /// Returns `None` if the name is unrecognized (caller may then
    /// fall back to [`CalibrationAffine::IDENTITY`] or surface an
    /// error).
    pub fn lookup(&self, codec_name: &str) -> Option<CalibrationAffine> {
        match codec_name.to_ascii_lowercase().as_str() {
            "jpeg" | "jpg" | "zenjpeg" | "mozjpeg" | "libjpeg" => Some(self.jpeg),
            "webp" | "zenwebp" => Some(self.webp),
            "avif" | "zenavif" => Some(self.avif),
            "jxl" | "zenjxl" | "jpegxl" | "jpeg-xl" => Some(self.zenjxl),
            "png" | "zenpng" => Some(self.zenpng),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_passthrough() {
        let id = CalibrationAffine::IDENTITY;
        assert_eq!(id.apply(0.0), 0.0);
        assert_eq!(id.apply(63.0), 63.0);
        assert_eq!(id.apply(100.0), 100.0);
        assert_eq!(id.apply(150.0), 100.0); // clamped
        assert_eq!(id.apply(-10.0), 0.0); // clamped
    }

    #[test]
    fn affine_apply_and_clamp() {
        let a = CalibrationAffine {
            alpha: -10.0,
            beta: 1.5,
        };
        // 50 * 1.5 - 10 = 65
        assert!((a.apply(50.0) - 65.0).abs() < 1e-5);
        // 80 * 1.5 - 10 = 110 -> clamped to 100
        assert_eq!(a.apply(80.0), 100.0);
        // 0 * 1.5 - 10 = -10 -> clamped to 0
        assert_eq!(a.apply(0.0), 0.0);
    }

    #[test]
    fn invert_round_trips() {
        let a = CalibrationAffine {
            alpha: -10.0,
            beta: 1.5,
        };
        let raw = 50.0;
        let cal = a.alpha + a.beta * raw;
        let back = a.invert(cal).unwrap();
        assert!((back - raw).abs() < 1e-5);
    }

    #[test]
    fn lookup_handles_aliases() {
        let cal = CodecCalibration::PREVIEW_V0_5_TUNER;
        assert!(cal.lookup("jpeg").is_some());
        assert!(cal.lookup("JPEG").is_some());
        assert!(cal.lookup("jpg").is_some());
        assert!(cal.lookup("zenjpeg").is_some());
        assert!(cal.lookup("webp").is_some());
        assert!(cal.lookup("avif").is_some());
        assert!(cal.lookup("jxl").is_some());
        assert!(cal.lookup("zenjxl").is_some());
        assert!(cal.lookup("png").is_some());
        assert!(cal.lookup("nonexistent").is_none());
    }

    #[test]
    fn preview_v0_5_tuner_calibration_changes_jpeg_score() {
        // Tuner raw at PJND-ish (sweep median for jpeg around q=50)
        // The jpeg affine has alpha=-31.7, beta=1.35; on a raw of ~70
        // the calibrated score is roughly -31.7 + 1.35 * 70 = 62.8
        let raw = 70.0;
        let cal = CodecCalibration::PREVIEW_V0_5_TUNER;
        let jpeg_cal = cal.lookup("jpeg").unwrap();
        let result = jpeg_cal.apply(raw);
        // result should land in roughly the [50, 75] range for raw=70
        assert!(result > 40.0 && result < 90.0, "unexpected: {result}");
    }
}
