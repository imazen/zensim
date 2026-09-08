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

use anyhow::{Context, Result, bail};
use zensim::{RgbSlice, Zensim, ZensimProfile};

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
    search(
        rgb,
        width,
        height,
        codec,
        spec,
        Some(spec.profile),
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
    search(rgb, width, height, codec, spec, None, |a, b| {
        Ok(scorer.compute(a, b, Some(codec.extension()))?.score() as f32)
    })
}

#[allow(clippy::too_many_arguments)]
fn search(
    rgb: &[u8],
    width: u32,
    height: u32,
    codec: CodecKind,
    spec: TargetSpec,
    profile: Option<ZensimProfile>,
    mut score: impl FnMut(&RgbSlice<'_>, &RgbSlice<'_>) -> Result<f32>,
) -> Result<TargetResult> {
    if width == 0
        || height == 0
        || !spec.target.is_finite()
        || !spec.tolerance.is_finite()
        || spec.tolerance < 0.0
        || spec.max_iterations == 0
    {
        bail!(
            "nonzero dimensions/pass budget, finite target, and finite nonnegative tolerance required"
        );
    }
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
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .context("image dimensions overflow")?;
    if rgb.len() != expected {
        bail!(
            "rgb buffer length {} != width*height*3 = {}",
            rgb.len(),
            expected
        );
    }

    let backend = codec::backend_for(codec);
    let (q_lo_native, q_hi_native) = backend.quality_range();
    let inverted = backend.lower_quality_means_higher_score();
    if let Some(seed) = spec.seed
        && (!seed.knob.is_finite()
            || seed.knob < q_lo_native
            || seed.knob > q_hi_native
            || !seed.score_per_knob.is_finite()
            || seed.score_per_knob.abs() < 1e-6
            || (seed.score_per_knob < 0.0) != inverted)
    {
        bail!("seed must be in the codec range with a finite, correctly oriented nonzero slope");
    }

    // Initial probe at the midpoint of the range.
    let src_pixels: &[[u8; 3]] = bytemuck::cast_slice(rgb);
    let scratch_src = RgbSlice::try_new(src_pixels, width as usize, height as usize)
        .map_err(|e| anyhow::anyhow!("rgb slice for reference image: {e:?}"))?;

    let mut probes: Vec<ProbeRecord> = Vec::new();
    let mut q_lo = q_lo_native;
    let mut q_hi = q_hi_native;
    let mut best_idx: Option<usize> = None;
    let mut best_encoded: Vec<u8> = Vec::new();

    // A seeded run uses the training slope for its second probe and measured
    // secants thereafter. Unseeded runs retain the historical environment
    // switch. Safeguarding keeps proposals inside the current interval; it
    // does not prove convergence or monotonicity of a real codec/scorer pair.
    let use_secant = spec.seed.is_some()
        || std::env::var("ZENSIM_TARGET_SECANT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    let mut sec_a: Option<(f32, f32)> = None; // (q, f=achieved−target), older
    let mut sec_b: Option<(f32, f32)> = None; // newer

    let budget = if codec.is_lossless_only() {
        1
    } else {
        spec.max_iterations
    };
    for iter in 0..budget {
        let q_bisect = (q_lo + q_hi) * 0.5;
        let q_mid = if let Some(seed) = spec.seed
            && iter == 0
        {
            seed.knob
        } else if let Some(seed) = spec.seed
            && iter == 1
        {
            let previous = &probes[0];
            let proposed =
                previous.knob + (spec.target - previous.achieved_score) / seed.score_per_knob;
            guarded_knob(proposed, q_lo, q_hi, q_bisect)
        } else if use_secant {
            if let (Some((qa, fa)), Some((qb, fb))) = (sec_a, sec_b) {
                let denom = fb - fa;
                let slope = denom / (qb - qa);
                if denom.abs() > 1e-6 && slope.is_finite() && (slope < 0.0) == inverted {
                    let qs = qb - fb * (qb - qa) / denom;
                    guarded_knob(qs, q_lo, q_hi, q_bisect)
                } else {
                    q_bisect
                }
            } else {
                q_bisect
            }
        } else {
            q_bisect
        };
        let (encoded, decoded_rgb) = backend
            .encode_decode(rgb, width, height, q_mid)
            .with_context(|| format!("codec {codec:?} encode/decode at knob {q_mid:.3}"))?;
        if decoded_rgb.len() != expected {
            bail!(
                "codec {codec:?} decode returned {} bytes; expected {}",
                decoded_rgb.len(),
                expected
            );
        }
        let dst_pixels: &[[u8; 3]] = bytemuck::cast_slice(&decoded_rgb);
        let dst = RgbSlice::try_new(dst_pixels, width as usize, height as usize)
            .map_err(|e| anyhow::anyhow!("rgb slice for decoded image: {e:?}"))?;
        let achieved =
            score(&scratch_src, &dst).with_context(|| format!("zensim compute on iter {iter}"))?;
        if !achieved.is_finite() {
            bail!("nonfinite score on iteration {iter}");
        }
        if use_secant {
            sec_a = sec_b;
            sec_b = Some((q_mid, achieved - spec.target));
        }

        probes.push(ProbeRecord {
            iteration: iter,
            knob: q_mid,
            achieved_score: achieved,
            byte_count: encoded.len(),
        });

        // Track best-so-far by |achieved - target|.
        let is_best = match best_idx {
            None => true,
            Some(i) => {
                (achieved - spec.target).abs() < (probes[i].achieved_score - spec.target).abs()
            }
        };
        if is_best {
            best_idx = Some(probes.len() - 1);
            best_encoded = encoded.clone();
        }

        if (achieved - spec.target).abs() <= spec.tolerance {
            // Converged.
            return Ok(finalize(
                codec,
                spec,
                profile,
                encoded,
                achieved,
                q_mid,
                iter + 1,
                probes,
                width,
                height,
                true,
            ));
        }

        // Move the search window. With inverted direction (jxl distance):
        // achieved > target means quality is too high → distance too low → increase q_lo.
        let too_high = achieved > spec.target;
        if inverted {
            if too_high {
                q_lo = q_mid;
            } else {
                q_hi = q_mid;
            }
        } else if too_high {
            q_hi = q_mid;
        } else {
            q_lo = q_mid;
        }
    }

    // Out of budget — return best-so-far.
    let best = best_idx.expect("at least one probe ran");
    let best_probe = probes[best].clone();
    Ok(finalize(
        codec,
        spec,
        profile,
        best_encoded,
        best_probe.achieved_score,
        best_probe.knob,
        budget,
        probes,
        width,
        height,
        false,
    ))
}

fn guarded_knob(proposed: f32, low: f32, high: f32, fallback: f32) -> f32 {
    if proposed.is_finite() && proposed > low && proposed < high {
        proposed
    } else {
        fallback
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize(
    codec: CodecKind,
    spec: TargetSpec,
    profile: Option<ZensimProfile>,
    encoded: Vec<u8>,
    achieved: f32,
    knob: f32,
    iterations: u32,
    probes: Vec<ProbeRecord>,
    width: u32,
    height: u32,
    converged: bool,
) -> TargetResult {
    TargetResult {
        codec,
        target: spec.target,
        tolerance: spec.tolerance,
        profile,
        encoded,
        achieved_score: achieved,
        final_knob: knob,
        iterations,
        probes,
        width,
        height,
        converged,
    }
}

fn build_zensim(profile: ZensimProfile) -> Zensim {
    Zensim::new(profile)
}

#[cfg(test)]
mod tests {
    use super::*;

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
