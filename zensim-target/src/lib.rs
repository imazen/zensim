//! zensim-target — pick encode params to hit a user-specified zensim score.
//!
//! Given (image, target_score, codec), this crate runs a binary search over
//! the codec's quality knob, encodes + decodes the image at each probe,
//! scores the result with a zensim profile, and returns the encoded bytes
//! that landed closest to `target ± tolerance`.
//!
//! # Why this is a separate crate
//!
//! `zensim` itself is MIT/Apache and must stay free of AGPL deps. This
//! crate pulls in `zenjpeg` / `zenwebp` / `zenavif` / `zenjxl` / `zenpng`
//! (all AGPL-3.0-only) to drive their encoders, so it lives outside the
//! library tree as `publish = false`.
//!
//! # Algorithm
//!
//! Binary search over the codec's q range, capped at `max_iterations`
//! (default 8). Each iteration:
//!   1. Encode image at q_mid.
//!   2. Decode encoded bytes back to RGB.
//!   3. Compute zensim(reference, decoded).
//!   4. If achieved in `[target - tolerance, target + tolerance]`: done.
//!   5. If achieved > target: q_hi = q_mid (lower q → lower score).
//!   6. Else: q_lo = q_mid.
//!
//! For codecs where quality↑ → score↑ (zenjpeg / zenwebp / zenavif), the
//! search direction is monotonic. zenjxl uses distance (lower = higher
//! quality) so its search direction is inverted internally.
//!
//! Lossless codecs (PNG) skip the search entirely and return on first
//! probe with the lossless score.

#![forbid(unsafe_code)]

pub mod codec;
mod controller;
#[cfg(feature = "native-probe")]
pub mod native_probe;
mod seed_curve;
use controller::search;
#[cfg(feature = "native-probe")]
mod native_dial;
pub use seed_curve::SeedCurve;

use anyhow::{Result, bail};
#[cfg(test)]
use zensim::RgbSlice;
use zensim::{Zensim, ZensimProfile};

/// Selectable codec families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecKind {
    Jpeg,
    Webp,
    Avif,
    Jxl,
    Png,
}

impl CodecKind {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "zenjpeg" | "jpeg" | "jpg" => Ok(Self::Jpeg),
            "zenwebp" | "webp" => Ok(Self::Webp),
            "zenavif" | "avif" => Ok(Self::Avif),
            "zenjxl" | "jxl" => Ok(Self::Jxl),
            "zenpng" | "png" => Ok(Self::Png),
            other => bail!(
                "unknown codec '{other}'; expected one of: zenjpeg, zenwebp, zenavif, zenjxl, zenpng"
            ),
        }
    }

    /// Default output file extension for the codec.
    pub fn extension(self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Webp => "webp",
            Self::Avif => "avif",
            Self::Jxl => "jxl",
            Self::Png => "png",
        }
    }

    /// `true` if the codec has no quality knob (PNG).
    pub fn is_lossless_only(self) -> bool {
        matches!(self, Self::Png)
    }
}

/// User-facing target spec.
#[derive(Debug, Clone, Copy)]
pub struct TargetSpec {
    /// Desired finite zensim score; negative targets are allowed.
    pub target: f32,
    /// Convergence tolerance (`|achieved - target| <= tolerance` → ship).
    pub tolerance: f32,
    /// Maximum iterations of the binary search.
    pub max_iterations: u32,
    /// Profile to use for scoring.
    pub profile: ZensimProfile,
    /// Training-only starting knob and local slope. Never fit from this
    /// evaluation image's encodes or bound probes. `None` uses bisection.
    pub seed: Option<SeedEstimate>,
}

/// Codec/scorer-specific prediction fitted on separate training images.
#[derive(Debug, Clone, Copy)]
pub struct SeedEstimate {
    pub knob: f32,
    /// Change in zensim score per native knob unit (negative for distance).
    pub score_per_knob: f32,
}

impl Default for TargetSpec {
    fn default() -> Self {
        Self {
            target: 70.0,
            tolerance: 1.0,
            max_iterations: 8,
            // Follow the library's codec-target alias (B as of 2026-09-07).
            // This selects the scorer; convergence still depends on the
            // codec, content, target, and probe budget. The CLI's default and
            // --profile default follow the same alias. Current score contract:
            // docs/CODEC_TARGET_METRIC.md in the parent repository.
            profile: ZensimProfile::codec_target(),
            seed: None,
        }
    }
}

/// Per-iteration probe record.
#[derive(Debug, Clone)]
pub struct ProbeRecord {
    pub iteration: u32,
    /// Knob value in the codec's native scale (q for jpeg/webp/avif; distance for jxl).
    pub knob: f32,
    pub achieved_score: f32,
    pub byte_count: usize,
}

/// Result of one target-search run.
#[derive(Debug, Clone)]
pub struct TargetResult {
    pub codec: CodecKind,
    pub target: f32,
    pub tolerance: f32,
    pub profile: Option<ZensimProfile>,
    /// Encoded bytes (best probe).
    pub encoded: Vec<u8>,
    /// Final achieved zensim score.
    pub achieved_score: f32,
    /// Knob value that produced the final result (q or distance).
    pub final_knob: f32,
    pub iterations: u32,
    pub probes: Vec<ProbeRecord>,
    /// Width / height of the source.
    pub width: u32,
    pub height: u32,
    /// `true` if achieved is within `target ± tolerance`.
    pub converged: bool,
}

/// Drive the binary search.
///
/// `rgb` is a tightly-packed 24-bit RGB buffer of `width * height` pixels.
pub fn target_search(
    rgb: &[u8],
    width: u32,
    height: u32,
    codec: CodecKind,
    spec: TargetSpec,
) -> Result<TargetResult> {
    let scorer = build_zensim(spec.profile);
    let backend = enabled_backend(codec)?;
    search(
        rgb,
        width,
        height,
        codec,
        spec,
        Some(spec.profile),
        backend.as_ref(),
        |a, b| Ok(scorer.compute(a, b)?.score() as f32),
    )
}

/// Search with the complete Rust candidate scorer, including its heads and spline.
/// Input is tightly packed sRGB RGB8, with exactly `width * height * 3` bytes.
/// `spec.profile` is ignored; the result records `profile: None`.
pub fn target_search_with_bake(
    rgb: &[u8],
    width: u32,
    height: u32,
    codec: CodecKind,
    spec: TargetSpec,
    scorer: &mut zensim::BakeScorer<'_>,
) -> Result<TargetResult> {
    let backend = enabled_backend(codec)?;
    target_search_with_backend_and_bake(rgb, width, height, codec, spec, backend.as_ref(), scorer)
}

/// Use the shared candidate search with a codec's native encoder integration.
/// Input is tightly packed sRGB RGB8, exactly `width * height * 3` bytes.
/// `codec` identifies the model's codec hint and result label; `backend`
/// supplies the actual configuration, knob range and independent decode.
/// No built-in codec feature is required for an externally supplied backend.
#[allow(clippy::too_many_arguments)]
pub fn target_search_with_backend_and_bake(
    rgb: &[u8],
    width: u32,
    height: u32,
    codec: CodecKind,
    spec: TargetSpec,
    backend: &dyn codec::CodecBackend,
    scorer: &mut zensim::BakeScorer<'_>,
) -> Result<TargetResult> {
    search(rgb, width, height, codec, spec, None, backend, |a, b| {
        Ok(scorer.compute(a, b, Some(codec.extension()))?.score() as f32)
    })
}

fn enabled_backend(codec: CodecKind) -> Result<Box<dyn codec::CodecBackend>> {
    let enabled = match codec {
        CodecKind::Jpeg => cfg!(feature = "zenjpeg"),
        CodecKind::Webp => cfg!(feature = "zenwebp"),
        CodecKind::Avif => cfg!(feature = "zenavif"),
        CodecKind::Jxl => cfg!(feature = "zenjxl"),
        CodecKind::Png => cfg!(feature = "zenpng"),
    };
    if !enabled {
        bail!("codec {codec:?} is not enabled in this build");
    }
    Ok(codec::backend_for(codec))
}

fn build_zensim(profile: ZensimProfile) -> Zensim {
    Zensim::new(profile)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_backend_is_used_without_a_builtin_codec_feature() {
        use std::cell::RefCell;
        struct Native {
            calls: RefCell<Vec<f32>>,
            range: (f32, f32),
        }
        impl codec::CodecBackend for Native {
            fn quality_range(&self) -> (f32, f32) {
                self.range
            }
            fn lower_quality_means_higher_score(&self) -> bool {
                true
            }
            fn encode_decode(
                &self,
                rgb: &[u8],
                _: u32,
                _: u32,
                knob: f32,
            ) -> Result<(Vec<u8>, Vec<u8>)> {
                self.calls.borrow_mut().push(knob);
                Ok((knob.to_le_bytes().to_vec(), vec![0; rgb.len()]))
            }
        }
        let model = zenpredict::Model::from_bytes(include_bytes!(
            "../../zensim/weights/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin"
        ))
        .unwrap();
        let mut scorer = zensim::BakeScorer::new(&model).unwrap();
        let rgb: Vec<u8> = (0..32 * 32 * 3)
            .map(|i| ((i * 37 + i / 13) % 256) as u8)
            .collect();
        let black = vec![[0; 3]; 32 * 32];
        let source = RgbSlice::new(bytemuck::cast_slice(&rgb), 32, 32);
        let target = scorer
            .compute(&source, &RgbSlice::new(&black, 32, 32), Some("jxl"))
            .unwrap()
            .score() as f32;
        let spec = TargetSpec {
            target,
            tolerance: 0.,
            max_iterations: 3,
            seed: Some(SeedEstimate {
                knob: 2.25,
                score_per_knob: -10.,
            }),
            ..TargetSpec::default()
        };
        let backend = Native {
            calls: RefCell::new(Vec::new()),
            range: (0.01, 25.),
        };
        let result = target_search_with_backend_and_bake(
            &rgb,
            32,
            32,
            CodecKind::Jxl,
            spec,
            &backend,
            &mut scorer,
        )
        .unwrap();
        assert_eq!(*backend.calls.borrow(), [2.25]);
        assert_eq!(result.encoded, 2.25f32.to_le_bytes());
        assert_eq!(result.achieved_score, target);
        assert_eq!(result.profile, None);
        assert!(result.converged);
        let bad = Native {
            calls: RefCell::new(Vec::new()),
            range: (f32::NAN, 25.),
        };
        assert!(
            target_search_with_backend_and_bake(
                &rgb,
                32,
                32,
                CodecKind::Jxl,
                spec,
                &bad,
                &mut scorer
            )
            .is_err()
        );
        assert!(bad.calls.borrow().is_empty());
    }

    #[test]
    #[cfg(feature = "zenjpeg")]
    fn calibrated_seed_drives_the_real_first_encode() {
        let rgb: Vec<u8> = (0..32 * 32 * 3)
            .map(|i| ((i * 37 + i / 13) % 256) as u8)
            .collect();
        let backend = codec::backend_for(CodecKind::Jpeg);
        let (encoded, decoded) = backend.encode_decode(&rgb, 32, 32, 42.25).unwrap();
        let a = RgbSlice::try_new(bytemuck::cast_slice::<u8, [u8; 3]>(&rgb), 32, 32).unwrap();
        let b = RgbSlice::try_new(bytemuck::cast_slice::<u8, [u8; 3]>(&decoded), 32, 32).unwrap();
        let target = Zensim::new(ZensimProfile::D)
            .compute(&a, &b)
            .unwrap()
            .score() as f32;
        let spec = TargetSpec {
            target,
            max_iterations: 3,
            tolerance: 0.,
            profile: ZensimProfile::D,
            seed: Some(SeedEstimate {
                knob: 42.25,
                score_per_knob: 1.,
            }),
        };
        let r = target_search(&rgb, 32, 32, CodecKind::Jpeg, spec).unwrap();
        assert_eq!(r.encoded, encoded);
        assert_eq!(r.final_knob, 42.25);
        assert_eq!(r.iterations, 1);
        assert!(r.converged);
        let unseeded = target_search(
            &rgb,
            32,
            32,
            CodecKind::Jpeg,
            TargetSpec {
                seed: None,
                max_iterations: 1,
                ..spec
            },
        )
        .unwrap();
        assert_eq!(unseeded.final_knob, 50.);
        assert_ne!(unseeded.encoded, r.encoded);
        for seed in [
            SeedEstimate {
                knob: 101.,
                score_per_knob: 1.,
            },
            SeedEstimate {
                knob: 42.,
                score_per_knob: -1.,
            },
            SeedEstimate {
                knob: f32::NAN,
                score_per_knob: 1.,
            },
        ] {
            assert!(
                target_search(
                    &rgb,
                    32,
                    32,
                    CodecKind::Jpeg,
                    TargetSpec {
                        seed: Some(seed),
                        ..spec
                    }
                )
                .is_err()
            );
        }
    }
    #[test]
    fn invalid_requests_refuse_before_encoding() {
        for spec in [
            TargetSpec {
                max_iterations: 0,
                ..TargetSpec::default()
            },
            TargetSpec {
                target: f32::NAN,
                ..TargetSpec::default()
            },
            TargetSpec {
                tolerance: -1.0,
                ..TargetSpec::default()
            },
        ] {
            assert!(target_search(&[0; 12], 2, 2, CodecKind::Png, spec).is_err());
        }
        assert!(target_search(&[], 0, 0, CodecKind::Png, TargetSpec::default()).is_err());
    }
    #[cfg(feature = "zenpng")]
    #[test]
    fn candidate_and_named_loop_share_lossless_controller_and_report_unreachable_target() {
        let bytes = include_bytes!(
            "../../zensim/weights/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin"
        );
        let model = zenpredict::Model::from_bytes(bytes).unwrap();
        let mut scorer = zensim::BakeScorer::new(&model).unwrap();
        let rgb: Vec<u8> = (0..32 * 32 * 3).map(|i| (i * 31) as u8).collect();
        let spec = TargetSpec {
            target: -10.0,
            profile: ZensimProfile::D,
            ..TargetSpec::default()
        };
        let a = target_search(&rgb, 32, 32, CodecKind::Png, spec).unwrap();
        let b = target_search_with_bake(&rgb, 32, 32, CodecKind::Png, spec, &mut scorer).unwrap();
        assert_eq!(a.encoded, b.encoded);
        assert_eq!(a.achieved_score, b.achieved_score);
        assert_eq!(a.profile, Some(ZensimProfile::D));
        assert_eq!(b.profile, None);
        assert_eq!(b.iterations, 1);
        assert!(!b.converged);
    }
}
