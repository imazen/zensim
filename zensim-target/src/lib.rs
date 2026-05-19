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
            other => bail!("unknown codec '{other}'; expected one of: zenjpeg, zenwebp, zenavif, zenjxl, zenpng"),
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
    /// Desired zensim score in `0..=100`.
    pub target: f32,
    /// Convergence tolerance (`|achieved - target| <= tolerance` → ship).
    pub tolerance: f32,
    /// Maximum iterations of the binary search.
    pub max_iterations: u32,
    /// Profile to use for scoring.
    pub profile: ZensimProfile,
}

impl Default for TargetSpec {
    fn default() -> Self {
        Self {
            target: 70.0,
            tolerance: 1.0,
            max_iterations: 8,
            // Default to V0_3 — V0_5* profiles currently return ~0 on
            // identity / near-identity inputs in this workspace
            // (their bake's raw output for zero features doesn't map
            // to 100, which breaks the search loop). Callers can still
            // request V0_5* explicitly via the CLI / API.
            profile: ZensimProfile::PreviewV0_3,
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
    pub profile: ZensimProfile,
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

    if codec.is_lossless_only() {
        return target_search_lossless(rgb, width, height, codec, spec);
    }

    let backend = codec::backend_for(codec);
    let (q_lo_native, q_hi_native) = backend.quality_range();
    let inverted = backend.lower_quality_means_higher_score();

    // Initial probe at the midpoint of the range.
    let zensim = build_zensim(spec.profile);
    let src_pixels: &[[u8; 3]] = bytemuck::cast_slice(rgb);
    let scratch_src = RgbSlice::try_new(src_pixels, width as usize, height as usize)
        .map_err(|e| anyhow::anyhow!("rgb slice for reference image: {e:?}"))?;

    let mut probes: Vec<ProbeRecord> = Vec::new();
    let mut q_lo = q_lo_native;
    let mut q_hi = q_hi_native;
    let mut best_idx: Option<usize> = None;
    let mut best_encoded: Vec<u8> = Vec::new();

    for iter in 0..spec.max_iterations {
        let q_mid = (q_lo + q_hi) * 0.5;
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
        let result = zensim
            .compute(&scratch_src, &dst)
            .with_context(|| format!("zensim compute on iter {iter}"))?;
        let achieved = result.score() as f32;

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
                codec, spec, encoded, achieved, q_mid, iter + 1, probes, width, height, true,
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
        best_encoded,
        best_probe.achieved_score,
        best_probe.knob,
        spec.max_iterations,
        probes,
        width,
        height,
        false,
    ))
}

fn target_search_lossless(
    rgb: &[u8],
    width: u32,
    height: u32,
    codec: CodecKind,
    spec: TargetSpec,
) -> Result<TargetResult> {
    let backend = codec::backend_for(codec);
    let (encoded, decoded_rgb) = backend
        .encode_decode(rgb, width, height, 100.0)
        .with_context(|| format!("lossless {codec:?} encode/decode"))?;
    let zensim = build_zensim(spec.profile);
    let src_pixels: &[[u8; 3]] = bytemuck::cast_slice(rgb);
    let scratch_src = RgbSlice::try_new(src_pixels, width as usize, height as usize)
        .map_err(|e| anyhow::anyhow!("rgb slice for reference image: {e:?}"))?;
    let dst_pixels: &[[u8; 3]] = bytemuck::cast_slice(&decoded_rgb);
    let dst = RgbSlice::try_new(dst_pixels, width as usize, height as usize)
        .map_err(|e| anyhow::anyhow!("rgb slice for decoded image: {e:?}"))?;
    let result = zensim.compute(&scratch_src, &dst)?;
    let achieved = result.score() as f32;
    let probes = vec![ProbeRecord {
        iteration: 0,
        knob: 100.0,
        achieved_score: achieved,
        byte_count: encoded.len(),
    }];
    let converged = (achieved - spec.target).abs() <= spec.tolerance;
    Ok(finalize(
        codec, spec, encoded, achieved, 100.0, 1, probes, width, height, converged,
    ))
}

#[allow(clippy::too_many_arguments)]
fn finalize(
    codec: CodecKind,
    spec: TargetSpec,
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
        profile: spec.profile,
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
