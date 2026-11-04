//! Core zensim metric computation.
//!
//! Multi-scale SSIM + edge + high-frequency features in XYB color space,
//! with trained weights per feature.
//!
//! # Feature extraction pipeline
//!
//! Both images are converted to the XYB perceptual color space (cube-root LMS,
//! same as ssimulacra2 and butteraugli), then processed at multiple scales.
//! Each scale halves resolution via 2× box downscale. At each scale, 19 features
//! are extracted per XYB channel (X, Y, B): 13 basic + 6 peak/diagnostic,
//! giving **228 features total** (4 scales × 3 channels × 19 features).
//!
//! ## SSIM features (3 per channel per scale)
//!
//! Uses the ssimulacra2 variant of SSIM, which differs from standard SSIM:
//!
//! ```text
//! mu1 = blur(src), mu2 = blur(dst)
//! sigma12 = blur(src * dst)
//! sum_sq  = blur(src² + dst²)     // one blur instead of two
//!
//! num_m   = 1 - (mu1 - mu2)²      // luminance (no C1, no denominator)
//! num_s   = 2·sigma12 - 2·mu1·mu2 + C2   // structure × contrast
//! denom_s = sum_sq - mu1² - mu2² + C2     // = sigma1² + sigma2² + C2
//!
//! d = max(0, 1 - num_m · num_s / denom_s) // per-pixel SSIM error
//! ```
//!
//! The luminance component drops the standard SSIM denominator
//! `(mu1² + mu2² + C1)` — ssimulacra2's reasoning is that the denominator
//! over-weights dark-region errors, which is wrong for perceptually uniform
//! values (XYB is already gamma-like). There is no C1 constant; C2 = 0.0009.
//!
//! The `sum_sq` optimization computes `blur(src² + dst²)` with one blur
//! instead of separate `blur(src²)` and `blur(dst²)`, because the SSIM
//! formula only needs `sigma1² + sigma2²`, not each individually.
//!
//! Three pooling norms capture different aspects of the error distribution:
//! - **ssim_mean** = `mean(d)` — average error
//! - **ssim_4th**  = `(mean(d⁴))^(1/4)` — L4 norm, emphasizes worst-case errors
//! - **ssim_2nd**  = `(mean(d²))^(1/2)` — L2 norm, intermediate sensitivity
//!
//! ## Edge features (6 per channel per scale)
//!
//! Edge detection compares local detail (pixel minus local mean) between
//! source and distorted:
//!
//! ```text
//! diff_src = |src - mu1|    // source edge magnitude
//! diff_dst = |dst - mu2|    // distorted edge magnitude
//!
//! d = (1 + diff_dst) / (1 + diff_src) - 1   // per-pixel edge ratio
//!
//! artifact    = max(0,  d)   // distorted has MORE edge than source
//! detail_lost = max(0, -d)   // distorted has LESS edge than source
//! ```
//!
//! The `1 +` offsets prevent division by zero and dampen sensitivity in flat
//! regions. The ratio formulation is scale-invariant. Splitting into artifact
//! (ringing, banding, blockiness) vs detail_lost (blur, smoothing) lets the
//! model weight them independently.
//!
//! Each is pooled with three norms (mean, L4, L2) = 6 features.
//!
//! ## MSE (1 per channel per scale)
//!
//! Plain mean squared error in XYB space: `mean((src - dst)²)`.
//! No blur dependency, computed directly from pixels.
//!
//! ## High-frequency features (3 per channel per scale)
//!
//! These measure changes in local detail energy by comparing `pixel - blur(pixel)`
//! (the high-frequency residual) between source and distorted. Despite their
//! former names ("variance_loss", "texture_loss", "contrast_increase"), they
//! do NOT measure image variance — they measure the ratio of high-pass energy.
//!
//! ```text
//! hf_src_L2 = Σ(src - mu1)²    // source HF energy (L2)
//! hf_dst_L2 = Σ(dst - mu2)²    // distorted HF energy (L2)
//! hf_src_L1 = Σ|src - mu1|     // source HF magnitude (L1)
//! hf_dst_L1 = Σ|dst - mu2|     // distorted HF magnitude (L1)
//! ```
//!
//! - **hf_energy_loss** = `max(0, 1 - hf_dst_L2 / hf_src_L2)` — detail smoothed away
//! - **hf_mag_loss**    = `max(0, 1 - hf_dst_L1 / hf_src_L1)` — same, L1 (robust to outliers)
//! - **hf_energy_gain** = `max(0, hf_dst_L2 / hf_src_L2 - 1)` — detail added (ringing/sharpening)
//!
//! `hf_energy_loss` and `hf_energy_gain` are the positive and negative halves
//! of the same signal, split by ReLU — this gives the linear model separate
//! knobs for blur vs ringing without needing signed weights.
//!
//! ## Peak features (6 per channel per scale)
//!
//! Computed during the fused V-blur kernel at no extra cost:
//! - **ssim_max**, **art_max**, **det_max** — per-pixel maximum of each error type
//! - **ssim_l8**, **art_l8**, **det_l8** — L8-pooled (near-worst-case) values
//!
//! These capture outlier sensitivity that mean/L2/L4 pooling may miss.
//!
//! ## Scoring
//!
//! All 228 features are multiplied by trained weights, summed, normalized by
//! scale count, then mapped to a 0–100 score via:
//! `score = 100 - a · distance^b` (default a=18.0, b=0.7).

use crate::bake_metadata::*;
use crate::det_math::{DetPow, active_pow_form};
use crate::error::ZensimError;

/// Configuration for zensim computation.
///
/// All computation uses the streaming path, which processes scale 0 in
/// horizontal strips with fused blur+feature extraction for minimal memory
/// traffic. When `blur_passes == 1` (the default), fused H-blur + V-blur+reduce
/// SIMD kernels are used for peak performance.
///
/// Blur kernel shape for local-mean computation.
///
/// Controls how `blur(src)` and `blur(dst)` are computed at each scale.
/// The default `Box` kernel uses iterated box blur, which is O(1) per pixel
/// regardless of radius and has full SIMD optimization.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlurKernel {
    /// Iterated box blur. `passes` controls the kernel shape:
    /// - 1 = rectangular (fastest, enables fused streaming kernels)
    /// - 2 = triangular (~1.5× slower at scale 0)
    /// - 3 = piecewise-quadratic ≈ Gaussian (~2× slower)
    Box {
        /// Number of passes (1 = rectangular, 2 = triangular, 3 ≈ Gaussian).
        passes: u8,
    },
}

impl Default for BlurKernel {
    fn default() -> Self {
        Self::Box { passes: 1 }
    }
}

/// Downscale filter for pyramid construction.
///
/// Controls how each pyramid level is produced from the previous one.
/// `Box2x2` (2×2 block averaging) is the only filter the trained
/// profiles are calibrated for and the only implemented variant; the
/// enum stays `#[non_exhaustive]` so alternative kernels can be added
/// without a breaking change. (Speculative `zenresize`-gated
/// Mitchell/Lanczos variants were removed pre-0.3.0 — imazen/zensimx#47 —
/// because no compute path ever dispatched on them.)
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum DownscaleFilter {
    /// 2×2 box averaging (fastest, current default).
    #[default]
    Box2x2,
}

/// Configuration for the zensim metric computation pipeline.
///
/// Controls blur kernel, pyramid construction, and feature extraction.
/// The defaults match the trained profile and give peak performance;
/// only change these for training or research.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ZensimConfig {
    /// Request-local arithmetic for model-bound serving.
    pub(crate) formula_revision: Option<crate::feature_defs::FormulaRevision>,
    pub(crate) local_only: bool,
    pub(crate) omit_edges: bool,
    pub(crate) attribution_channels: Option<[[bool; 3]; crate::NUM_SCALES]>,
    /// Box blur radius at scale 0 (default: 5, giving an 11-pixel kernel).
    ///
    /// The blur kernel width is `2 * blur_radius + 1`. Larger radii capture
    /// coarser structure but increase computation proportionally.
    /// Both streaming and full-image paths are SIMD-optimized for any radius.
    pub blur_radius: usize,

    /// Number of box blur passes (1, 2, or 3; default: 1).
    ///
    /// Controls the blur kernel shape:
    /// - **1 pass** — rectangular kernel. Enables fused blur+feature SIMD kernels
    ///   in the streaming path (fastest).
    /// - **2 passes** — triangular kernel. Falls back to separate blur+reduce in
    ///   the streaming path (~1.5× slower at scale 0).
    /// - **3 passes** — piecewise-quadratic ≈ Gaussian. Same fallback (~2× slower).
    ///
    /// All three variants have full SIMD optimization (AVX-512 + AVX2 dispatch).
    /// The performance difference comes from whether the fused streaming kernels
    /// can be used, not from the blur itself.
    pub blur_passes: u8,

    /// Blur kernel shape (default: `BlurKernel::Box { passes: 1 }`).
    ///
    /// Overrides `blur_passes` when set. The `blur_radius` field still controls
    /// the kernel width. Currently only the `Box` variant is implemented.
    pub blur_kernel: BlurKernel,

    /// Downscale filter for pyramid construction (default: `DownscaleFilter::Box2x2`).
    ///
    /// Controls how each pyramid level is produced. `Box2x2` is currently
    /// the only variant.
    pub downscale_filter: DownscaleFilter,

    /// Compute all 156 features even when their weights are zero (default: false).
    ///
    /// When false, channels/features with zero weight are skipped entirely.
    /// Enable for weight training to avoid circular dependency (need all features
    /// to determine which weights should be nonzero).
    pub compute_all_features: bool,

    /// Maximum number of downscale levels (default: 4).
    ///
    /// Each level halves resolution. 4 scales covers 1×, 2×, 4×, 8× — sufficient
    /// for most perceptual effects. The feature vector length scales linearly:
    /// `num_scales × 3 channels × 13 features`.
    ///
    /// Both paths are SIMD-optimized for any scale count.
    pub num_scales: usize,

    /// Score mapping scale factor (default: 18.0).
    ///
    /// Used in the final score formula: `score = 100 - a × d^b`, where `d` is
    /// the raw weighted distance. Larger values spread scores more aggressively.
    pub score_mapping_a: f64,

    /// Score mapping gamma exponent (default: 0.7).
    ///
    /// Used in the final score formula: `score = 100 - a × d^b`. Sub-linear
    /// gamma (< 1.0) compresses high distances, giving more resolution in the
    /// high-quality range.
    pub score_mapping_b: f64,

    /// Enable multi-threaded computation via rayon (default: true).
    pub allow_multithreading: bool,
}

impl ZensimConfig {
    /// The formula revision this computation runs: the request's own
    /// (`formula_revision`), or — when the request names none — the process
    /// revision, which is what `None` has always meant. The value every
    /// revision-dependent leaf on this config's walk receives (featcanon D1).
    pub(crate) fn revision(&self) -> crate::feature_defs::FormulaRevision {
        self.formula_revision
            .unwrap_or_else(crate::ssim_form::active_revision)
    }
}

impl Default for ZensimConfig {
    fn default() -> Self {
        Self {
            formula_revision: None,
            local_only: false,
            omit_edges: false,
            attribution_channels: None,
            blur_radius: 5,
            blur_passes: 1,
            blur_kernel: BlurKernel::default(),
            downscale_filter: DownscaleFilter::default(),
            compute_all_features: false,

            num_scales: crate::NUM_SCALES,
            score_mapping_a: 18.0,
            score_mapping_b: 0.7,
            allow_multithreading: true,
        }
    }
}

/// Map a raw weighted distance to the 0–100 quality score.
///
/// Uses the default power-law mapping: `score = 100 - 18 * d^0.7`, clamped to \[0, 100\].
/// Identical images (d = 0) score 100.
///
/// For profile-specific mapping, use [`Zensim::compute`] which applies the profile's
/// `score_mapping_a` and `score_mapping_b` automatically.
pub(crate) fn distance_to_score(raw_distance: f64) -> f64 {
    distance_to_score_mapped(raw_distance, 18.0, 0.7)
}

/// Map a raw weighted distance to the quality score with custom parameters.
///
/// `score = 100 - a * d^b`. Nominally 0–100 but can go negative for
/// extreme distortions (the magnitude below zero is informative —
/// it distinguishes "slightly wrong" from "completely wrong").
fn distance_to_score_mapped(raw_distance: f64, a: f64, b: f64) -> f64 {
    crate::score_math::distance_to_score_mapped(raw_distance, a, b, active_pow_form())
}

/// Compute score from raw features using custom weights.
///
/// `features`: raw features from `ZensimResult::features()`.
/// `weights`: one weight per feature; `weights.len()` must equal
/// `features.len()`.
///
/// Returns `(score, raw_distance)` on success, or
/// [`ZensimError::FeatureWeightsLengthMismatch`] if the slices have
/// different lengths.
pub fn try_score_from_features(
    features: &[f64],
    weights: &[f64],
) -> Result<(f64, f64), ZensimError> {
    if features.len() != weights.len() {
        return Err(ZensimError::FeatureWeightsLengthMismatch);
    }
    let raw_distance: f64 = features
        .iter()
        .zip(weights.iter())
        .map(|(&f, &w)| w * f)
        .sum();
    // Normalize by number of scales.
    // Layout: [scored × N_scales] [peaks × N_scales] [masked × N_scales]
    // 156 = 39×4, 228 = 57×4, 300 = 75×4 — all divide by 4 scales.
    let per_scale_candidates = [
        FEATURES_PER_CHANNEL_WITH_PEAKS * 3, // 57
        FEATURES_PER_CHANNEL_BASIC * 3,      // 39
    ];
    let features_per_scale = per_scale_candidates
        .iter()
        .copied()
        .find(|&ps| ps > 0 && features.len().is_multiple_of(ps))
        .unwrap_or(FEATURES_PER_CHANNEL_BASIC * 3);
    let n_scales = features.len() / features_per_scale;
    let raw_distance = raw_distance / n_scales.max(1) as f64;
    Ok((distance_to_score(raw_distance), raw_distance))
}

/// Compute score from raw features using custom weights.
/// `features`: raw features from ZensimResult.features
/// `weights`: one weight per feature (len must equal features.len())
/// Returns (score, raw_distance)
///
/// # Panics
///
/// Panics if the slices have different lengths; [`try_score_from_features`]
/// returns an error instead.
pub fn score_from_features(features: &[f64], weights: &[f64]) -> (f64, f64) {
    try_score_from_features(features, weights).expect("features and weights must have same length")
}

/// Pre-compute reference with a custom number of pyramid scales.
///
/// Use this when calling [`compute_zensim_with_ref_and_config`] with a non-default
/// `num_scales`. The precomputed data must have at least as many scales as the config
/// requests.
///
/// Feature-extraction API for custom scale counts. Changing the pyramid
/// changes the features; default trained weights need their original layout.
#[doc(hidden)]
pub fn precompute_reference_with_scales(
    source: &[[u8; 3]],
    width: usize,
    height: usize,
    num_scales: usize,
) -> Result<crate::streaming::PrecomputedReference, ZensimError> {
    if width < 8 || height < 8 {
        return Err(ZensimError::ImageTooSmall);
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(ZensimError::ImageTooLarge)?;
    // Reject any width/height combination whose padded-plane size would
    // overflow `usize` on the current target.
    check_within_max_pixels(width, height, None)?;
    if source.len() != pixels {
        return Err(ZensimError::InvalidDataLength);
    }
    let src_img = crate::source::RgbSlice::try_new(source, width, height)?;
    Ok(crate::streaming::PrecomputedReference::new(
        &src_img,
        num_scales,
        true,
        crate::ssim_form::active_revision(),
    ))
}

/// Compute zensim with a precomputed reference and custom configuration.
///
/// **Research / feature-extraction API; not stable.** The
/// `ZensimConfig` knobs that this entry exposes change metric
/// internals whose outputs are incomparable with the default trained
/// weights. The `config.num_scales` must not exceed the number of
/// scales in `precomputed`.
#[doc(hidden)]
#[cfg(feature = "training")]
pub fn compute_zensim_with_ref_and_config(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
    config: ZensimConfig,
) -> Result<ZensimResult, ZensimError> {
    if width < 8 || height < 8 {
        return Err(ZensimError::ImageTooSmall);
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(ZensimError::ImageTooLarge)?;
    if distorted.len() != pixels {
        return Err(ZensimError::InvalidDataLength);
    }
    if precomputed.width() != width || precomputed.height() != height {
        return Err(ZensimError::DimensionMismatch);
    }
    let dst_img = crate::source::RgbSlice::try_new(distorted, width, height)?;
    // The reference was reflect-padded inside `PrecomputedReference::new`;
    // pad the distorted to match or the scale walk asserts on the width
    // mismatch. Same rule as `Zensim::compute_with_ref`.
    let result = if needs_pyramid_pad(width, height, config.num_scales) {
        let d = reflect_pad_for_scales(&dst_img, config.num_scales);
        crate::streaming::compute_zensim_streaming_with_ref(precomputed, &d, &config, WEIGHTS, None)
    } else {
        crate::streaming::compute_zensim_streaming_with_ref(
            precomputed,
            &dst_img,
            &config,
            WEIGHTS,
            None,
        )
    };
    // This entry point scores via the plain WEIGHTS (V0_2 linear) distance,
    // never an MLP forward pass — tag honestly rather than inheriting
    // `combine_scores`'s internal `codec_target()` (`B`) placeholder tag
    // (which every `Zensim::compute*` caller overrides but this standalone
    // function does not go through).
    Ok(result.with_profile(crate::profile::ZensimProfile::LegacyLinearV0_2))
}

/// Per-scale statistics collected during computation.
#[derive(Default, Clone)]
pub(crate) struct ScaleStats {
    /// SSIM statistics: [mean_d, root4_d] per channel = 6 values
    pub(crate) ssim: [f64; 6],
    /// Edge features: [art_mean, art_4th, det_mean, det_4th] per channel = 12 values
    pub(crate) edge: [f64; 12],
    /// Per-channel MSE: mean((src - dst)²) for X, Y, B
    pub(crate) mse: [f64; 3],
    /// High-frequency energy loss (L2): max(0, 1 - Σ(dst-mu_dst)²/Σ(src-mu_src)²) per channel.
    /// Measures loss of local detail energy relative to source. Sensitive to blur/smoothing.
    pub(crate) hf_energy_loss: [f64; 3],
    /// High-frequency magnitude loss (L1): max(0, 1 - Σ|dst-mu_dst|/Σ|src-mu_src|) per channel.
    /// Like hf_energy_loss but with L1 norm — more robust to outliers.
    pub(crate) hf_mag_loss: [f64; 3],
    /// 2nd-power pooled SSIM: [root2_d] per channel = 3 values
    pub(crate) ssim_2nd: [f64; 3],
    /// Edge 2nd power: [art_2nd, det_2nd] per channel = 6 values
    pub(crate) edge_2nd: [f64; 6],
    /// High-frequency energy gain (L2): max(0, Σ(dst-mu_dst)²/Σ(src-mu_src)² - 1) per channel.
    /// Measures added local detail energy (ringing, sharpening artifacts).
    pub(crate) hf_energy_gain: [f64; 3],
    /// Max SSIM error per channel = 3 values
    pub(crate) ssim_max: [f64; 3],
    /// Max edge artifact per channel = 3 values
    pub(crate) art_max: [f64; 3],
    /// Max edge detail_lost per channel = 3 values
    pub(crate) det_max: [f64; 3],
    /// L8 power pool SSIM error per channel = 3 values: (Σd⁸/N)^(1/8)
    pub(crate) ssim_p95: [f64; 3],
    /// L8 power pool edge artifact per channel = 3 values: (Σd⁸/N)^(1/8)
    pub(crate) art_p95: [f64; 3],
    /// L8 power pool edge detail_lost per channel = 3 values: (Σd⁸/N)^(1/8)
    pub(crate) det_p95: [f64; 3],
}

/// Result from a zensim comparison.
///
/// Contains the final score, the raw distance used to derive it, and the
/// full per-scale feature vector (useful for diagnostics or weight training).
#[derive(Debug, Clone)]
pub struct ZensimResult {
    score: f64,
    raw_distance: f64,
    features: Vec<f64>,
    profile: crate::profile::ZensimProfile,
    mean_offset: [f64; 3],
    /// Set by the identity-image short-circuit in
    /// `compute_with_config_inner` / `compute_zensim_streaming`. When
    /// true, the post-network MLP scoring path in
    /// `apply_mlp_scoring_with_codec` MUST NOT overwrite the result —
    /// running the trained MLP on the all-zero feature vector
    /// produces garbage (the bake has no signal anchoring
    /// "zero feature vector → score 100").
    ///
    /// Replaces the pre-0.3.0 sentinel check
    /// `raw_distance == 0.0 && features.iter().all(== 0.0)` which was
    /// brittle (any future profile that legitimately produces
    /// raw_distance=0.0 + all-zero features on a real image would
    /// silently skip MLP scoring).
    is_identical: bool,
}

impl ZensimResult {
    /// Create a result from computed values. Internal use only.
    pub(crate) fn new(
        score: f64,
        raw_distance: f64,
        features: Vec<f64>,
        profile: crate::profile::ZensimProfile,
        mean_offset: [f64; 3],
    ) -> Self {
        Self {
            score,
            raw_distance,
            features,
            profile,
            mean_offset,
            is_identical: false,
        }
    }

    /// Mark this result as the byte-identical short-circuit output.
    /// `apply_mlp_scoring_with_codec` reads this flag to skip the
    /// MLP forward pass (which would overwrite the (100, 0, zeros)
    /// payload with garbage).
    #[inline]
    pub(crate) fn mark_identical(mut self) -> Self {
        self.is_identical = true;
        self
    }

    /// True iff this result was produced by the byte-identical
    /// short-circuit. Used by `apply_mlp_scoring_with_codec` to
    /// preserve the (score=100, raw_distance=0, zero-features)
    /// invariant regardless of which bake is loaded.
    #[inline]
    pub(crate) fn is_identical(&self) -> bool {
        self.is_identical
    }

    /// Set the profile on this result (builder pattern). Internal use only.
    pub(crate) fn with_profile(mut self, profile: crate::profile::ZensimProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Replace the raw distance and score with values from the MLP
    /// scoring path. Internal use only — called by
    /// [`apply_mlp_scoring`](crate::metric::apply_mlp_scoring).
    pub(crate) fn set_mlp_score(&mut self, raw_distance: f64, score: f64) {
        self.raw_distance = raw_distance;
        self.score = score;
    }

    /// Create a NaN sentinel result (for error/placeholder paths).
    pub fn nan() -> Self {
        Self {
            score: f64::NAN,
            raw_distance: f64::NAN,
            features: vec![],
            profile: crate::profile::ZensimProfile::PreviewV0_1,
            mean_offset: [f64::NAN; 3],
            is_identical: false,
        }
    }

    /// Quality score on a 0–100 scale. 100 = identical, 0 = maximally different.
    /// Derived from `raw_distance` via a power-law mapping.
    pub fn score(&self) -> f64 {
        self.score
    }

    /// Raw weighted feature distance before nonlinear mapping. Lower = more similar.
    /// Not bounded to a fixed range; depends on image content and weights.
    pub fn raw_distance(&self) -> f64 {
        self.raw_distance
    }

    /// Per-scale raw features as a slice.
    ///
    /// Layout: 4 scales × 3 channels (X, Y, B) × 19 features per channel = 228.
    /// See [`FeatureView`] for named access.
    pub fn features(&self) -> &[f64] {
        &self.features
    }

    /// Consume the result and return the owned feature vector.
    pub fn into_features(self) -> Vec<f64> {
        self.features
    }

    /// Which profile produced this score.
    pub fn profile(&self) -> crate::profile::ZensimProfile {
        self.profile
    }

    /// Per-channel XYB mean offset: `mean(src_xyb[c]) - mean(dst_xyb[c])`.
    ///
    /// Captures global color/luminance shifts (CMS errors, white balance changes).
    /// Channels: `[X, Y, B]`, signed. Positive = distorted is darker/less saturated.
    pub fn mean_offset(&self) -> [f64; 3] {
        self.mean_offset
    }

    /// Convert the score to a dissimilarity value.
    ///
    /// Dissimilarity is `(100 - score) / 100`: 0 = identical, higher = worse.
    /// This is the inverse of the 0–100 score scale, normalized to 0–1.
    ///
    /// See also [`score_to_dissimilarity`] for the standalone conversion.
    pub fn dissimilarity(&self) -> f64 {
        score_to_dissimilarity(self.score)
    }

    /// Approximate SSIMULACRA2 score from the raw distance.
    ///
    /// Direct power-law fit: `100 - 19.04 × d^0.598`, calibrated on 344k
    /// synthetic pairs. MAE: 4.4 SSIM2 points, Pearson r = 0.974.
    ///
    /// More accurate than `mapping::zensim_to_ssim2(score)` (MAE 4.9, r = 0.932)
    /// because it skips the intermediate score mapping.
    pub fn approx_ssim2(&self) -> f64 {
        if self.raw_distance <= 0.0 {
            return 100.0;
        }
        (100.0 - 19.0379 * self.raw_distance.det_powf(0.5979, active_pow_form())).max(-100.0)
    }

    /// Approximate DSSIM value from the raw distance.
    ///
    /// Direct power-law fit: `0.000922 × d^1.224`, calibrated on 344k
    /// synthetic pairs. MAE: 0.00129, Pearson r = 0.952.
    ///
    /// Significantly more accurate than `mapping::zensim_to_dssim(score)`
    /// (MAE 0.00213, r = 0.719) because DSSIM's natural exponent (1.22)
    /// differs from the score mapping exponent (0.70).
    pub fn approx_dssim(&self) -> f64 {
        if self.raw_distance <= 0.0 {
            return 0.0;
        }
        0.000922 * self.raw_distance.det_powf(1.2244, active_pow_form())
    }

    /// Approximate butteraugli distance from the raw distance.
    ///
    /// Direct power-law fit: `2.365 × d^0.613`, calibrated on 344k
    /// synthetic pairs. MAE: 1.65 distance units, Pearson r = 0.713.
    ///
    /// Butteraugli's weak correlation with our features (r = 0.71) limits
    /// approximation accuracy regardless of mapping choice.
    pub fn approx_butteraugli(&self) -> f64 {
        if self.raw_distance <= 0.0 {
            return 0.0;
        }
        2.365353 * self.raw_distance.det_powf(0.6130, active_pow_form())
    }
}

/// Convert a zensim score (0–100, 100 = identical) to a dissimilarity value
/// (0 = identical, higher = worse).
///
/// Linear conversion: `(100 - score) / 100`.
///
/// | score | dissimilarity |
/// |-------|---------------|
/// | 100.0 | 0.0           |
/// | 99.5  | 0.005         |
/// | 95.0  | 0.05          |
/// | 50.0  | 0.5           |
/// | 0.0   | 1.0           |
pub fn score_to_dissimilarity(score: f64) -> f64 {
    (100.0 - score) / 100.0
}

/// Convert a dissimilarity value (0 = identical, higher = worse) back to a
/// zensim score (0–100, 100 = identical).
///
/// Inverse of [`score_to_dissimilarity`]: `score = 100 * (1 - dissimilarity)`.
pub fn dissimilarity_to_score(dissimilarity: f64) -> f64 {
    (100.0 * (1.0 - dissimilarity)).clamp(0.0, 100.0)
}

/// What kind of perceptual difference dominates between source and distorted.
///
/// Only categories with provably defensible statistical signatures are offered.
/// If no category can be identified with high confidence, `Unclassified` is returned.
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Images are perceptually identical (score ≈ 100).
    Identical,
    /// Max delta ≤ N/255 — integer rounding, LUT precision, truncation.
    RoundingError,
    /// One channel zero-delta, others large — RGB↔BGR swap.
    ChannelSwap,
    /// Alpha compositing error (e.g. straight/premul confusion, wrong background).
    AlphaCompositing,
    /// Images differ but no category reached sufficient confidence.
    Unclassified,
}

/// Decomposed error classification for a source/distorted pair.
///
/// `dominant` is the category with the highest confidence (or `Identical`
/// if the overall score is ≈ 100).
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ErrorClassification {
    /// The dominant error category.
    pub dominant: ErrorCategory,
    /// Overall confidence in the classification (0.0–1.0).
    pub confidence: f64,
    /// Rounding bias analysis (only populated when `dominant == RoundingError`).
    ///
    /// Measures how balanced the rounding errors are across positive and negative
    /// directions. `None` when not a rounding error or insufficient data.
    pub rounding_bias: Option<RoundingBias>,
}

/// Analysis of whether rounding errors are balanced (+/-) or systematic.
///
/// A balanced distribution (roughly equal +1 and -1 counts) indicates normal
/// rounding mode differences — nothing to worry about. A heavily skewed
/// distribution (mostly one direction) suggests systematic truncation or
/// a floor/ceil bias that may indicate a pipeline bug.
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RoundingBias {
    /// Per-channel ratio of positive-to-total differing pixels.
    ///
    /// 0.5 = perfectly balanced, 0.0 = all negative, 1.0 = all positive.
    /// Channels: `[R, G, B]`.
    pub positive_fraction: [f64; 3],
    /// Whether the rounding appears balanced (within statistical norms).
    ///
    /// `true` means the +/- distribution is consistent with unbiased rounding
    /// and is likely nothing to worry about. `false` means systematic bias
    /// was detected (e.g., all errors in one direction = truncation).
    pub balanced: bool,
}

/// Pixel-level delta analysis for error classification.
///
/// All deltas are `src - dst` (positive = distorted is darker/lower).
/// Values normalized to [0.0, 1.0] regardless of input bit depth.
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DeltaStats {
    // --- Per-channel [R, G, B] summary stats ---
    /// Mean delta (signed). Positive = dst darker.
    pub mean_delta: [f64; 3],
    /// Standard deviation of delta.
    pub stddev_delta: [f64; 3],
    /// Maximum |delta|.
    pub max_abs_delta: [f64; 3],

    // --- Signed small-delta histogram ---
    /// Per-channel pixel counts for signed deltas -3 to +3 (in 1/native_max units).
    ///
    /// Index mapping: `[0]`=−3, `[1]`=−2, `[2]`=−1, `[3]`=0, `[4]`=+1, `[5]`=+2, `[6]`=+3.
    /// Delta convention: `src - dst`, so +1 means dst is 1 LSB lower than src.
    /// Only counts pixels whose per-channel delta falls in \[−3, +3\]; pixels
    /// outside this range are not tracked here.
    pub signed_small_histogram: [[u64; 7]; 3],

    /// Maximum representable value for the native pixel format.
    ///
    /// 255.0 for u8 formats, 65535.0 for u16, 1.0 for f32/f16.
    /// Used to interpret delta magnitudes at native precision.
    pub native_max: f64,

    // --- Pixel counts ---
    /// Total pixels compared.
    pub pixel_count: u64,
    /// Pixels where any channel differs.
    pub pixels_differing: u64,
    /// Pixels where any channel |delta| > 1/255.
    pub pixels_differing_by_more_than_1: u64,

    // --- Alpha channel ---
    /// Whether the input format has an alpha channel.
    pub has_alpha: bool,
    /// Max |src_alpha - dst_alpha| in 0-255 units. 0 for RGB-only formats.
    pub alpha_max_delta: u8,
    /// Pixels where alpha differs at all. 0 for RGB-only formats.
    pub alpha_pixels_differing: u64,

    // --- Per-channel value histograms (256 bins, quantized to 8-bit) ---
    /// Source image histogram. `[channel][value]`. R=0, G=1, B=2, A=3.
    pub src_histogram: [[u64; 256]; 4],
    /// Distorted image histogram. `[channel][value]`. R=0, G=1, B=2, A=3.
    pub dst_histogram: [[u64; 256]; 4],

    // --- Alpha-stratified stats (only for RGBA/BGRA inputs) ---
    /// Delta stats for fully opaque pixels (A = max).
    pub opaque_stats: Option<AlphaStratifiedStats>,
    /// Delta stats for semitransparent pixels (0 < A < max).
    pub semitransparent_stats: Option<AlphaStratifiedStats>,
    /// Pearson correlation between |delta| and (1 - alpha).
    /// High (> 0.8) = compositing/premul error. None if no alpha.
    pub alpha_error_correlation: Option<f64>,
}

/// Stats for a subset of pixels grouped by alpha.
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AlphaStratifiedStats {
    /// Number of pixels in this stratum.
    pub pixel_count: u64,
    /// Mean |delta| per channel in this alpha stratum.
    pub mean_abs_delta: [f64; 3],
    /// Max |delta| per channel.
    pub max_abs_delta: [f64; 3],
}

/// Result from `classify()`: the zensim score plus delta analysis and error classification.
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ClassifiedResult {
    /// The standard zensim result (score, features, etc.).
    pub result: ZensimResult,
    /// Error classification with per-category confidence scores.
    pub classification: ErrorClassification,
    /// Pixel-level delta statistics.
    pub delta_stats: DeltaStats,
}

// --- Zensim config struct (primary API) ---

use crate::feature_v2::HdrEncoding;
use crate::profile::{ProfileParams, ZensimProfile};
use crate::source::ImageSource;

mod bake;
pub(crate) use bake::BakeScorer;

/// Metric configuration. Methods on this struct are the primary API.
///
/// ```no_run
/// use zensim::{Zensim, ZensimProfile, RgbSlice};
/// # let (src, dst) = (vec![[0u8; 3]; 64], vec![[0u8; 3]; 64]);
/// let z = Zensim::new(ZensimProfile::codec_target());
/// let source = RgbSlice::new(&src, 8, 8);
/// let distorted = RgbSlice::new(&dst, 8, 8);
/// let result = z.compute(&source, &distorted).unwrap();
/// println!("{}: {:.2}", result.profile(), result.score());
/// ```
#[derive(Clone, Debug)]
pub struct Zensim {
    profile: ZensimProfile,
    parallel: bool,
    /// Optional cap on total pixels (`width * height`) per image. `None`
    /// disables the cap. Allocation grows roughly `~14 × pixels × 4 B` for
    /// the streaming pipeline, so callers feeding untrusted dimensions
    /// should set this. See [`Zensim::with_max_pixels`].
    max_pixels: Option<usize>,
    /// Optional cooperative-cancellation token, checked at row-band and
    /// scale boundaries inside the streaming walk. See [`Zensim::with_stop`].
    stop: Option<StopHandle>,
}

/// Shared cancellation token stored on [`Zensim`]. Wraps the caller's
/// [`enough::Stop`] in an `Arc` so `Zensim` stays `Clone` (clones share
/// the same token). Debug output elides the token (trait objects have no
/// useful Debug form).
#[derive(Clone)]
struct StopHandle(std::sync::Arc<dyn enough::Stop>);

impl core::fmt::Debug for StopHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("StopHandle(..)")
    }
}

/// Whether a computation that asked for internal parallel work runs it:
/// `requested`, and the current rayon pool has more than one thread. With a
/// single thread each parallel section would only hand its work to the
/// pool's one worker and block until it returns (most of a small image's
/// time); the sequential path computes the same bits.
#[inline]
pub(crate) fn effective_parallel(requested: bool) -> bool {
    #[cfg(feature = "threads")]
    {
        requested && rayon::current_num_threads() > 1
    }
    #[cfg(not(feature = "threads"))]
    {
        let _ = requested;
        false
    }
}

impl Zensim {
    /// Create a new `Zensim` with the given profile. Parallel by default.
    ///
    /// A default `max_pixels` cap of **120 million** (≈ 120 MP — large enough
    /// for common ~108 MP camera photos) is applied so that forwarding
    /// untrusted dimensions can't drive an unbounded allocation (memory is
    /// ~14×w×h×4 B). Call [`Zensim::with_max_pixels`] to tighten it, or
    /// `with_max_pixels(usize::MAX)` to opt out for trusted input.
    pub fn new(profile: ZensimProfile) -> Self {
        Self {
            profile,
            parallel: true,
            max_pixels: Some(120_000_000),
            stop: None,
        }
    }

    /// The serving PLAN for a SCORING entry: `None` (today's unconditional v1
    /// walk) unless the extraction must differ — because this instance opted
    /// into pool skipping, or because a bake declares a LAYOUT wider than the
    /// v1 walk emits.
    ///
    /// The second case is what makes a wide bake servable at all. Before it,
    /// a 944-declared bake reached `prep_bake_input_f32` with a 372-wide
    /// vector and was refused with `"bake declares more input features than
    /// the caller supplied"` — a true statement about a situation the runtime
    /// created by never asking for the width the bake declared.
    fn scoring_plan(
        &self,
        params: &ProfileParams,
        config: &ZensimConfig,
    ) -> Option<crate::feature_plan::Plan> {
        crate::fold_engine::score_plan(
            params,
            config,
            config.revision() >= crate::feature_defs::FormulaRevision::Rev5,
        )
    }

    /// Enable or disable multi-threaded computation (rayon).
    /// Default: `true`.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set a cap on total pixels (`width * height`) accepted by every
    /// `compute*` entry point. Images exceeding the cap are rejected with
    /// [`ZensimError::ImageTooLarge`] before any allocation runs.
    ///
    /// Default: **120 MP** (set by [`Zensim::new`]). Zensim allocates
    /// roughly `width × height × 4 bytes × ~14` for the streaming XYB
    /// pyramid plus destination planes, so a 4K (≈8.3 MP) image needs
    /// ~470 MB of scratch. A cap of e.g. 64 MP (8192×8192) keeps the
    /// worst case at ~3.6 GB.
    ///
    /// Pass [`usize::MAX`] explicitly to disable the cap on a previously
    /// configured `Zensim`.
    pub fn with_max_pixels(mut self, max_pixels: usize) -> Self {
        self.max_pixels = Some(max_pixels);
        self
    }

    /// Install a cooperative-cancellation token, checked at row-band and
    /// scale boundaries inside the streaming compute walk. When the token
    /// fires, the walk abandons remaining work promptly and the entry
    /// point returns [`ZensimError::Cancelled`].
    ///
    /// Accepts any [`enough::Stop`] implementation — typically a
    /// `Stopper` from the sibling
    /// [`almost-enough`](https://docs.rs/almost-enough) crate, cancelled
    /// from another thread (or wired to a timeout). Passing
    /// [`Unstoppable`](enough::Unstoppable) (or any token whose
    /// [`may_stop`](enough::Stop::may_stop) is `false`) stores nothing and
    /// keeps the zero-check fast path.
    ///
    /// Coverage: the scoring entry points (`compute`,
    /// `compute_with_codec_hint`, `compute_with_ref`, `compute_pu_linear`,
    /// `compute_pu_linear_planar`) and the diffmap entries
    /// (`compute_with_ref_and_diffmap`,
    /// `compute_with_ref_and_diffmap_linear_planar`). The v2
    /// feature-extraction walks and the caller-paced strip APIs
    /// (`compute_streaming_strips*`, where the caller already controls
    /// cadence between strips) do not check the token yet.
    ///
    /// At Rev5, canonical fold scoring checks the token between strips as
    /// well as before and after the fold. Earlier buffered paths retain
    /// the following band-level cancellation granularity.
    /// Cancellation granularity is one row band (a few dozen rows) at the
    /// current scale; a fired token is detected within roughly one band's
    /// worth of work per rayon worker.
    ///
    /// Below Rev5 a live token therefore keeps every scoring entry on the
    /// buffered walk. A profile whose bake reads past that walk's width
    /// (`C`, `CHdr`) is served through the fold only without a token; with
    /// one, `compute`, `compute_pu_linear` and `compute_pu_linear_planar` all
    /// refuse it with the same error, so one contract covers every entry.
    ///
    /// ```no_run
    /// use zensim::{Zensim, ZensimProfile};
    /// # struct S(std::sync::Arc<std::sync::atomic::AtomicBool>);
    /// # impl enough::Stop for S {
    /// #     fn check(&self) -> Result<(), enough::StopReason> {
    /// #         if self.0.load(std::sync::atomic::Ordering::Relaxed) {
    /// #             Err(enough::StopReason::Cancelled)
    /// #         } else { Ok(()) }
    /// #     }
    /// # }
    /// # let token = S(Default::default());
    /// let z = Zensim::new(ZensimProfile::codec_target()).with_stop(token);
    /// // z.compute(..) now returns Err(ZensimError::Cancelled { .. })
    /// // soon after the token fires.
    /// ```
    pub fn with_stop(mut self, stop: impl enough::Stop + 'static) -> Self {
        // may_stop() == false (e.g. Unstoppable) can never fire — skip
        // storing it so every check stays on the None fast path.
        self.stop = if stop.may_stop() {
            Some(StopHandle(std::sync::Arc::new(stop)))
        } else {
            None
        };
        self
    }

    /// Borrow the installed stop token for threading into the walkers.
    pub(crate) fn stop_ref(&self) -> Option<&dyn enough::Stop> {
        self.stop.as_ref().map(|h| &*h.0 as &dyn enough::Stop)
    }

    /// Return `Err(ZensimError::Cancelled)` if the installed token has
    /// fired. Entry points call this up front (cheap pre-flight) and after
    /// the walk returns (converting an abandoned walk into the error).
    pub(crate) fn check_stop(&self) -> Result<(), ZensimError> {
        match &self.stop {
            Some(h) => h.0.check().map_err(ZensimError::from),
            None => Ok(()),
        }
    }

    /// Current `max_pixels` cap, if any.
    pub fn max_pixels(&self) -> Option<usize> {
        self.max_pixels
    }

    /// Current profile.
    pub fn profile(&self) -> ZensimProfile {
        self.profile
    }

    /// Whether multi-threaded computation is enabled.
    pub fn parallel(&self) -> bool {
        self.parallel
    }

    /// Whether this call runs its internal parallel work: [`Self::parallel`]
    /// and a rayon pool of more than one thread ([`effective_parallel`]).
    #[inline]
    fn parallel_now(&self) -> bool {
        effective_parallel(self.parallel)
    }

    /// Compare source and distorted images.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too small.
    pub fn compute(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        check_within_max_pixels(source.width(), source.height(), self.max_pixels)?;
        self.check_stop()?;
        let config = config_from_params(params, self.parallel_now());
        let mut result = compute_with_config_inner(
            source,
            distorted,
            &config,
            params.weights,
            self.stop_ref(),
            self.scoring_plan(params, &config).as_ref(),
        );
        self.check_stop()?;
        apply_mlp_scoring_with_codec(
            &mut result,
            params,
            source.width() as u32,
            source.height() as u32,
        )?;
        Ok(result.with_profile(self.profile))
    }

    /// Pre-compute reference image data for batch comparison.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if either dimension is zero.
    pub fn precompute_reference(
        &self,
        source: &impl ImageSource,
    ) -> Result<crate::streaming::PrecomputedReference, ZensimError> {
        let params = self.profile.params();
        if source.width() == 0 || source.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        // Sub-64px sources are reflect-padded inside `PrecomputedReference::new`
        // (matching the buffered path), so no 8px floor here.
        check_within_max_pixels(source.width(), source.height(), self.max_pixels)?;
        Ok(crate::streaming::PrecomputedReference::new(
            source,
            params.num_scales,
            self.parallel_now(),
            config_from_params(params, self.parallel_now()).revision(),
        ))
    }

    /// The engine-routed body of [`Self::compute_with_ref`]: the fold-backed
    /// ref-cached compare when this instance asks for it AND the fold can
    /// serve the request bit-identically, the buffered `*_with_ref` walk
    /// otherwise. Both consume the SAME `PrecomputedReference` — the pyramid
    /// cache is not buffered-walk state (fold-engine lane stage 3).
    fn compare_against_ref(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
        config: &ZensimConfig,
        weights: &[f64],
    ) -> ZensimResult {
        self.compare_against_ref_into(precomputed, distorted, config, weights, None)
    }

    /// [`Self::compare_against_ref`] with an optional caller-owned scratch.
    /// `None` allocates a fresh one per call (what `compute_with_ref` does on
    /// both engines); `Some` is the ref-loop shape `compute_with_ref_into`
    /// exposes, and is what makes the fold's ref-cache saving match buffered's
    /// instead of being halved by a per-compare allocation (fold-MT lane).
    fn compare_against_ref_into(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
        config: &ZensimConfig,
        weights: &[f64],
        scratch: Option<&mut crate::streaming::ZensimScratch>,
    ) -> ZensimResult {
        let mut scratch = scratch;
        if config.revision() >= crate::feature_defs::FormulaRevision::Rev5
            && crate::fold_engine::is_fold_backable(config)
        {
            let mut owned;
            let v2 = match scratch.as_mut() {
                Some(s) => &mut s.v2,
                None => {
                    owned = crate::feature_v2::V2Scratch::new();
                    &mut owned
                }
            };
            if config.revision() >= crate::feature_defs::FormulaRevision::Rev5 {
                let plan = self.scoring_plan(self.profile.params(), config);
                if let Some(result) =
                    crate::fold_engine::compute_fold_backed_with_ref_stoppable_plan(
                        precomputed,
                        distorted,
                        config,
                        weights,
                        v2,
                        plan.as_ref().map(|p| p.compute.v1_pools),
                        plan.as_ref(),
                        self.stop_ref(),
                    )
                {
                    return result;
                }
            }
            // The ref-cached walk emits the v1 layout only; a wide bake is
            // not servable through it yet (phase 2 — registered, and the
            // servability census reports it as such rather than silently
            // scoring a truncated vector). Take just the pool axis.
            let pool_mode = self
                .scoring_plan(self.profile.params(), config)
                .map(|p| p.compute.v1_pools);
            if let Some(r) = crate::fold_engine::compute_fold_backed_with_ref(
                precomputed,
                distorted,
                config,
                weights,
                v2,
                pool_mode,
            ) {
                return r;
            }
        }
        crate::streaming::compute_zensim_streaming_with_ref(
            precomputed,
            distorted,
            config,
            weights,
            self.stop_ref(),
        )
    }

    /// Compare a distorted image against a precomputed reference.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if either dimension is zero.
    /// Returns [`ZensimError::DimensionMismatch`] if `distorted.width()` /
    /// `distorted.height()` differ from the precomputed reference's
    /// dimensions (see
    /// [`PrecomputedReference::width`](crate::PrecomputedReference::width)).
    /// Returns [`ZensimError::ImageTooLarge`] if dimensions exceed the
    /// configured `max_pixels` cap.
    pub fn compute_with_ref(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        if distorted.width() == 0 || distorted.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        // Contract: distorted must match the reference's ORIGINAL (unpadded)
        // dims (`PrecomputedReference` keeps those even when its pyramid was
        // built on a reflect-padded source).
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels)?;
        // The precomputed reference was built via `precompute_reference`,
        // which already rejected HDR — so only check the distorted side
        // here. Symmetry across the call path is preserved.
        let config = config_from_params(params, self.parallel_now());
        validate_cached_revision(precomputed, config.revision())?;
        let (ow, oh) = (distorted.width(), distorted.height());
        // Pad a sub-64px distorted to the pyramid minimum so it aligns with the
        // (also-padded) reference pyramid; score with the original dims. The
        // pad is SHARED — it runs before either engine, so both see the same
        // pixels against the same cached pyramid.
        let mut result = if needs_pyramid_pad(ow, oh, config.num_scales) {
            let d = reflect_pad_for_scales(distorted, config.num_scales);
            self.compare_against_ref(precomputed, &d, &config, params.weights)
        } else {
            self.compare_against_ref(precomputed, distorted, &config, params.weights)
        };
        self.check_stop()?;
        apply_mlp_scoring(&mut result, params, ow as u32, oh as u32)?;
        Ok(result.with_profile(self.profile))
    }

    /// Compare against a precomputed reference, reusing caller-owned scratch
    /// buffers. Designed for encoder quantization loops that compare many
    /// distorted candidates against the same reference: the per-call XYB
    /// plane allocation (which can be ~25 MB at 1080p, ~99 MB at 4K) is
    /// kept alive across calls instead of being freed and reallocated.
    ///
    /// The first call costs the same as [`Zensim::compute_with_ref`]; subsequent
    /// calls skip the allocation and the OS page-fault commit.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if either dimension is zero.
    pub fn compute_with_ref_into(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
        scratch: &mut crate::streaming::ZensimScratch,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        if distorted.width() == 0
            || distorted.height() == 0
            || (distorted.width() < 8 || distorted.height() < 8)
        {
            return Err(ZensimError::ImageTooSmall);
        }
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels)?;
        let config = config_from_params(params, self.parallel_now());
        validate_cached_revision(precomputed, config.revision())?;
        let (ow, oh) = (distorted.width(), distorted.height());
        // ENGINE ROUTING (fold-MT lane). `compute_with_ref` has routed to the
        // fold since the fold-engine lane; this entry did not, so a ref LOOP —
        // the one shape that exists to amortise work — was the one shape that
        // could not use the fold. It routes through the same
        // `compare_against_ref_into` and hands it `scratch`, so the fold's
        // per-compare `V2Scratch` is reused exactly like buffered's
        // `dst_planes`. Out-of-domain requests still degrade to the buffered
        // walk below, byte-for-byte.
        if config.revision() >= crate::feature_defs::FormulaRevision::Rev5
            && crate::fold_engine::is_fold_backable(&config)
        {
            // Same shared reflect-pad `compute_with_ref` applies, so the two
            // ref entries route identically rather than differing on sub-64.
            let mut result = if needs_pyramid_pad(ow, oh, config.num_scales) {
                let d = reflect_pad_for_scales(distorted, config.num_scales);
                self.compare_against_ref_into(
                    precomputed,
                    &d,
                    &config,
                    params.weights,
                    Some(scratch),
                )
            } else {
                self.compare_against_ref_into(
                    precomputed,
                    distorted,
                    &config,
                    params.weights,
                    Some(scratch),
                )
            };
            apply_mlp_scoring(&mut result, params, ow as u32, oh as u32)?;
            return Ok(result.with_profile(self.profile));
        }
        // Pad a sub-pyramid-minimum distorted so it aligns with the (also
        // padded) reference pyramid — the same rule `compute_with_ref`
        // applies. Without it this entry fed a 48-wide plane to a 64-wide
        // reference scale and tripped `scale 0 width mismatch`; scoring still
        // uses the ORIGINAL dims (2026-08-30).
        let (stats, mean_offset) = if needs_pyramid_pad(ow, oh, config.num_scales) {
            let d = reflect_pad_for_scales(distorted, config.num_scales);
            crate::streaming::compute_multiscale_stats_streaming_with_ref_borrowed(
                precomputed,
                &d,
                &mut scratch.dst_planes,
                &config,
                params.weights,
                self.stop_ref(),
            )
        } else {
            crate::streaming::compute_multiscale_stats_streaming_with_ref_borrowed(
                precomputed,
                distorted,
                &mut scratch.dst_planes,
                &config,
                params.weights,
                self.stop_ref(),
            )
        };
        self.check_stop()?;
        let mut result = combine_scores(&stats, params.weights, &config, mean_offset);
        apply_mlp_scoring(&mut result, params, ow as u32, oh as u32)?;
        Ok(result.with_profile(self.profile))
    }

    /// Precompute reference from planar linear RGB f32 data.
    ///
    /// `planes` are `[R, G, B]`, each with at least `stride * height` elements.
    /// `stride` is the number of f32 elements per row (≥ `width`; may be larger
    /// for padded buffers like the encoder's `padded_width`).
    ///
    /// This avoids the interleave-to-RGBA overhead when the caller already has
    /// separate channel buffers in linear light.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if either dimension is zero.
    pub fn precompute_reference_linear_planar(
        &self,
        planes: [&[f32]; 3],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<crate::streaming::PrecomputedReference, ZensimError> {
        let params = self.profile.params();
        if width < 8 || height < 8 {
            return Err(ZensimError::ImageTooSmall);
        }
        check_within_max_pixels(width, height, self.max_pixels)?;
        // `stride * height` must fit in usize on 32-bit; reject overflow up
        // front so downstream `y * stride + x` arithmetic cannot wrap.
        let row_capacity = stride
            .checked_mul(height)
            .ok_or(ZensimError::ImageTooLarge)?;
        for plane in &planes {
            if plane.len() < row_capacity {
                return Err(ZensimError::InvalidDataLength);
            }
        }
        Ok(crate::streaming::PrecomputedReference::from_linear_planar(
            planes,
            width,
            height,
            stride,
            params.num_scales,
            self.parallel_now(),
            config_from_params(params, self.parallel_now()).revision(),
        ))
    }

    /// Like `compute`, but always computes all features regardless of
    /// zero weights (forces every channel active). For training/research.
    /// At Rev5 this emits all supported basic, peak and v2 slots in the
    /// 720 layout; masked/IW slots remain outside the supported scope.
    #[cfg(feature = "training")]
    pub fn compute_all_features(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        let mut config = config_from_params(params, self.parallel_now());
        config.compute_all_features = true;
        let extraction_plan = if config.revision() >= crate::feature_defs::FormulaRevision::Rev5 {
            use crate::feature_defs::family_slots;
            use crate::feature_set_id::ComputeToken;
            let want = family_slots(ComputeToken::Basic, config.num_scales)
                .union(&family_slots(ComputeToken::Peaks, config.num_scales))
                .union(&family_slots(ComputeToken::V2, config.num_scales));
            // At the profile's revision, as `Plan::for_bake` serves it: a slot
            // request alone would take the shipped revision (1).
            Some(
                crate::feature_plan::Plan::derive_with_layout_at(
                    &want,
                    crate::feature_layout::Layout::identity(720),
                    config.revision(),
                )
                .map_err(|_| ZensimError::ModelLoadFailed {
                    reason: "cannot derive the full supported Rev5 extraction plan",
                })?,
            )
        } else {
            None
        };
        let result = compute_with_config_inner(
            source,
            distorted,
            &config,
            params.weights,
            self.stop_ref(),
            // Research extraction requests every supported slot, independently
            // of the profile's serving read set.
            extraction_plan.as_ref(),
        );
        self.check_stop()?;
        Ok(result.with_profile(self.profile))
    }

    /// Score an HDR pair. Pixel values map to absolute display light as
    /// `encoding` declares (already-linear cd/m², or PQ/HLG through a display
    /// model); the anchored cube-root front end then feeds the SDR feature walk,
    /// and this profile's model scores the result. BT.2408 reference white
    /// (203 cd/m², also the Ultra HDR SDR-base convention) maps to linear 1.0;
    /// light above that white is retained without clipping.
    ///
    /// Sources are [`PixelFormat::LinearF32Rgba`] (f32 light; any encoding) or
    /// [`PixelFormat::Srgb16Rgba`] (16-bit code values; [`HdrEncoding::Pq`] and
    /// [`HdrEncoding::Hlg`] only), with [`AlphaMode::Opaque`]. Declared
    /// primaries are honoured without clipping light to the SDR range.
    ///
    /// # Errors
    ///
    /// [`ZensimError::UnsupportedPixelFormat`] for any other format or alpha
    /// mode, invalid display parameters, or the size and cancellation errors of
    /// [`Self::compute`].
    ///
    /// [`PixelFormat::LinearF32Rgba`]: crate::PixelFormat::LinearF32Rgba
    /// [`PixelFormat::Srgb16Rgba`]: crate::PixelFormat::Srgb16Rgba
    /// [`AlphaMode::Opaque`]: crate::AlphaMode::Opaque
    pub fn compute_hdr(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        encoding: HdrEncoding,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        let config = config_from_params(params, self.parallel_now());
        let plan = if config.revision() >= crate::feature_defs::FormulaRevision::Rev5 {
            use crate::feature_defs::family_slots;
            use crate::feature_set_id::ComputeToken;
            let want = family_slots(ComputeToken::Basic, config.num_scales)
                .union(&family_slots(ComputeToken::Peaks, config.num_scales))
                .union(&family_slots(ComputeToken::V2, config.num_scales));
            Some(
                crate::feature_plan::Plan::derive_with_layout_at(
                    &want,
                    crate::feature_layout::Layout::identity(720),
                    config.revision(),
                )
                .map_err(|_| ZensimError::ModelLoadFailed {
                    reason: "cannot derive the full supported Rev5 extraction plan",
                })?,
            )
        } else {
            None
        };
        let toggles = plan.as_ref().map_or_else(
            || crate::feature_v2::V2NewFeatureToggles {
                v1_only: true,
                ..Default::default()
            },
            |p| p.toggles(),
        );
        let mut scratch = crate::feature_v2::V2Scratch::new();
        let mut mo = crate::feature_v2::MeanOffsetRows::at_revision(
            source.width(),
            source.height(),
            toggles.formula_revision,
        );
        let mut features = crate::feature_v2::compute_folded720_hdr_streaming_extras(
            source,
            distorted,
            encoding,
            crate::feature_v2::HdrFrontEnd::CubeRoot {
                unit_nits: crate::feature_v2::HDR_SDR_WHITE_NITS,
            },
            self.max_pixels,
            self.parallel_now(),
            toggles,
            &mut scratch,
            crate::feature_v2::FoldWalkExtras {
                compute: plan.as_ref().map(|p| p.compute),
                mean_offset: Some(&mut mo),
                ..Default::default()
            },
        )?
        .into_features();
        if let Some(p) = plan {
            p.check_emit_covered(features.len())
                .map_err(|_| ZensimError::ModelLoadFailed {
                    reason: "HDR fold did not materialize the bake's declared reads",
                })?;
            features.resize(
                p.walk_width()
                    .max(crate::fold_engine::v1_feature_width(&config)),
                0.0,
            );
        } else {
            features.truncate(crate::fold_engine::v1_feature_width(&config));
        }
        let (score, raw_distance) =
            score_v1_layout_features(&mut features, params.weights, &config, config.num_scales);
        let mut result =
            ZensimResult::new(score, raw_distance, features, self.profile, mo.finish());
        if images_byte_identical(source, distorted) {
            result.score = 100.0;
            result.raw_distance = 0.0;
            result = result.mark_identical();
        }
        self.check_stop()?;
        apply_mlp_scoring(
            &mut result,
            params,
            source.width() as u32,
            source.height() as u32,
        )?;
        Ok(result)
    }
}

#[cfg(feature = "classification")]
impl Zensim {
    /// Compare source and distorted images with full error classification.
    ///
    /// Returns a [`ClassifiedResult`] containing the standard zensim score,
    /// pixel-level delta statistics, and error type classification.
    ///
    /// The `result.score()` is identical to what `compute()` returns — classification
    /// is a separate analysis pass that doesn't affect the score.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too small.
    pub fn classify(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ClassifiedResult, ZensimError> {
        validate_pair(source, distorted)?;

        // Compute delta stats (pixel-level analysis in sRGB space).
        // Returns `ZensimError::UnsupportedPixelFormat` if either input
        // uses a `PixelFormat` the delta-stats extractor doesn't handle.
        let delta_stats = crate::streaming::compute_delta_stats(source, distorted)?;

        // Compute the standard zensim score
        let result = self.compute(source, distorted)?;

        // Derive classification from delta stats and zensim features
        let classification = derive_classification(&delta_stats, &result);

        Ok(ClassifiedResult {
            result,
            classification,
            delta_stats,
        })
    }
}

#[cfg(feature = "training")]
impl Zensim {
    /// Compute with explicit custom params (for training).
    pub fn compute_with_params(
        params: &ProfileParams,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        validate_pair(source, distorted)?;
        let config = config_from_params(params, true);
        let serving_plan = if config.revision() >= crate::feature_defs::FormulaRevision::Rev5 {
            crate::fold_engine::score_plan(params, &config, true)
        } else {
            None
        };
        let result = compute_with_config_inner(
            source,
            distorted,
            &config,
            params.weights,
            None,
            serving_plan.as_ref(),
        );
        Ok(result)
    }
}

/// Derive error classification from pixel-level delta statistics.
///
/// Uses only 3 provable detectors with mathematically defensible signatures:
/// 1. **RoundingError** — max delta ≤ 3/255, based on `pixels_differing_by_more_than_1`
/// 2. **ChannelSwap** — one zero-delta channel with large deltas in others
/// 3. **AlphaCompositing** — opaque unchanged, semitransparent changed (tightened)
///
/// No `Mixed` category — highest score wins, or `Unclassified`.
#[cfg(feature = "classification")]
fn derive_classification(delta_stats: &DeltaStats, _result: &ZensimResult) -> ErrorClassification {
    let mut rounding_bias: Option<RoundingBias> = None;

    // Track per-detector scores internally
    let mut score_rounding = 0.0f64;
    let mut score_swap = 0.0f64;
    let mut score_alpha = 0.0f64;

    // If images are identical, short circuit
    if delta_stats.pixels_differing == 0 {
        return ErrorClassification {
            dominant: ErrorCategory::Identical,
            confidence: 1.0,
            rounding_bias: None,
        };
    }

    let max_delta = delta_stats
        .max_abs_delta
        .iter()
        .copied()
        .fold(0.0f64, f64::max);

    // === 1. Rounding error: based on max_delta + pixels_differing_by_more_than_1 ===
    //
    // If no pixel in any channel exceeds 1/255 delta, this is provably off-by-1.
    // The only operations that produce max_delta ≤ 3/255 are: integer rounding
    // mode differences, sRGB LUT precision, float→int truncation.
    if delta_stats.pixels_differing_by_more_than_1 == 0 {
        score_rounding = 1.0;
    } else if max_delta <= 2.0 / 255.0 {
        score_rounding = 0.95;
    } else if max_delta <= 3.0 / 255.0 {
        score_rounding = 0.9;
    }

    // === 2. Channel swap: one zero-delta channel, others large ===
    //
    // The only way to get one channel with zero delta and others with large
    // deltas is a channel swap. No other operation produces this pattern.
    let mut zero_channels = 0u32;
    let mut hot_channels = 0u32;
    for ch in 0..3 {
        if delta_stats.max_abs_delta[ch] < 1.0 / 255.0 {
            zero_channels += 1;
        }
        if delta_stats.max_abs_delta[ch] > 0.1 {
            hot_channels += 1;
        }
    }
    if zero_channels == 1 && hot_channels >= 1 && max_delta > 0.05 {
        score_swap = 0.9;
    }

    // === 3. Alpha compositing: tightened thresholds ===
    //
    // Stratification: opaque pixels unchanged, semitransparent changed.
    // Tightened from 0.01→0.005 opaque threshold, 0.7→0.8 correlation threshold.
    if let Some(ref opaque) = delta_stats.opaque_stats
        && let Some(ref semi) = delta_stats.semitransparent_stats
    {
        let opaque_max = opaque.mean_abs_delta.iter().copied().fold(0.0f64, f64::max);
        let semi_mean = semi.mean_abs_delta.iter().copied().fold(0.0f64, f64::max);
        if opaque_max < 0.005 && semi_mean > 0.02 && semi.pixel_count > 100 {
            score_alpha = 0.9;
        }
    }
    if let Some(corr) = delta_stats.alpha_error_correlation
        && corr > 0.8
    {
        score_alpha = score_alpha.max(corr);
    }

    // === Determine dominant category ===
    // Highest score wins. No Mixed category.
    let scores = [
        (ErrorCategory::RoundingError, score_rounding),
        (ErrorCategory::ChannelSwap, score_swap),
        (ErrorCategory::AlphaCompositing, score_alpha),
    ];

    // NaN-safe: `total_cmp` orders all f64 values (including NaN) without
    // panicking. If any score is NaN, it sorts to the most-negative end
    // and is therefore never selected as the max — the `best_score > 0.0`
    // check below then routes to `Unclassified`.
    let (best_cat, best_score) = scores
        .iter()
        .copied()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("scores is a 3-element array — max is always defined");

    let (dominant, confidence) = if best_score > 0.0 {
        (best_cat, best_score)
    } else {
        (ErrorCategory::Unclassified, 0.0)
    };

    // === Rounding bias analysis ===
    // When RoundingError is detected, analyze the signed small-delta histogram
    // to determine if errors are balanced (+/-) or systematic (one direction).
    if dominant == ErrorCategory::RoundingError {
        rounding_bias = Some(compute_rounding_bias(delta_stats));
    }

    ErrorClassification {
        dominant,
        confidence,
        rounding_bias,
    }
}

/// Compute rounding bias from the signed small-delta histogram.
///
/// Examines the +1/-1, +2/-2, +3/-3 bins per channel to determine whether
/// errors are balanced (unbiased rounding) or systematic (truncation/floor).
#[cfg(feature = "classification")]
fn compute_rounding_bias(delta_stats: &DeltaStats) -> RoundingBias {
    let h = &delta_stats.signed_small_histogram;
    let mut positive_fraction = [0.5f64; 3];
    let mut all_balanced = true;

    for ch in 0..3 {
        // Count positive deltas (+1, +2, +3) and negative deltas (-1, -2, -3)
        let neg = h[ch][0] + h[ch][1] + h[ch][2]; // bins -3, -2, -1
        let pos = h[ch][4] + h[ch][5] + h[ch][6]; // bins +1, +2, +3
        let total_nonzero = neg + pos;

        if total_nonzero == 0 {
            // No differing pixels in this channel — perfectly balanced
            positive_fraction[ch] = 0.5;
            continue;
        }

        positive_fraction[ch] = pos as f64 / total_nonzero as f64;

        // Statistical test: for balanced rounding, we'd expect ~50% positive.
        // With N trials and p=0.5, the standard deviation is sqrt(N)/2.
        // Use a 3-sigma threshold: if |pos_frac - 0.5| > 3 * 0.5 / sqrt(N),
        // consider it unbalanced. But also require a minimum absolute skew
        // (at least 60/40 split) to avoid flagging trivially small deviations
        // in large samples.
        let n = total_nonzero as f64;
        let expected_std = 0.5 / n.sqrt();
        let deviation = (positive_fraction[ch] - 0.5).abs();
        if deviation > 3.0 * expected_std && deviation > 0.1 {
            all_balanced = false;
        }
    }

    RoundingBias {
        positive_fraction,
        balanced: all_balanced,
    }
}

pub(crate) fn validate_pair(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> Result<(), ZensimError> {
    validate_pair_dims(source, distorted)?;
    // Refuse HDR-flagged input on either side — a mixed SDR/HDR pair is
    // just as unscorable by the SDR pipeline as an all-HDR one.
    Ok(())
}

/// Validate matching, nonzero image dimensions.
pub(crate) fn validate_pair_dims(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> Result<(), ZensimError> {
    // Empty images are unscoreable; sub-64px images are now handled by
    // reflect-padding in `compute_with_config_inner` (scores down to 1×1),
    // so they are NOT rejected here anymore.
    if source.width() == 0 || source.height() == 0 {
        return Err(ZensimError::ImageTooSmall);
    }
    if source.width() != distorted.width() || source.height() != distorted.height() {
        return Err(ZensimError::DimensionMismatch);
    }
    Ok(())
}

/// Validate that `distorted` matches the dimensions baked into `precomputed`.
///
/// A precomputed reference is built from a specific source's `width × height`
/// (recorded on the struct via
/// [`PrecomputedReference::width`](crate::PrecomputedReference::width) /
/// [`height`](crate::PrecomputedReference::height)). Reusing it against a
/// distorted image with different dims would either silently produce garbage
/// scores (smaller padded width misaligns reference rows) or panic on slice
/// out-of-range. Reject such calls up front.
pub(crate) fn validate_ref_match(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
) -> Result<(), ZensimError> {
    validate_ref_dimensions(precomputed, distorted)
}

pub(crate) fn validate_cached_revision(
    precomputed: &crate::PrecomputedReference,
    revision: crate::feature_defs::FormulaRevision,
) -> Result<(), ZensimError> {
    if crate::featcanon::mode(precomputed.formula_revision) != crate::featcanon::mode(revision) {
        return Err(ZensimError::ModelLoadFailed {
            reason: "reference cache color conversion revision differs from scorer; precompute with this profile",
        });
    }
    Ok(())
}

pub(crate) fn validate_ref_dimensions(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
) -> Result<(), ZensimError> {
    if precomputed.width() != distorted.width() || precomputed.height() != distorted.height() {
        return Err(ZensimError::DimensionMismatch);
    }
    Ok(())
}

/// Reject `width × height` if it overflows `usize` on the current target,
/// exceeds the configured `max_pixels` cap, or if the padded plane size
/// (`pyramid_plane_stride(width) × height`) would overflow during downstream
/// allocation.
///
/// Centralizes the overflow guard so 32-bit / wasm32 builds (where
/// `usize = u32`) cannot wrap silently inside downstream allocation math.
pub(crate) fn check_within_max_pixels(
    width: usize,
    height: usize,
    max_pixels: Option<usize>,
) -> Result<(), ZensimError> {
    let pixels = width
        .checked_mul(height)
        .ok_or(ZensimError::ImageTooLarge)?;
    if let Some(cap) = max_pixels
        && pixels > cap
    {
        return Err(ZensimError::ImageTooLarge);
    }
    // Also confirm the padded plane size fits. `pyramid_plane_stride` rounds
    // up to a multiple of 16 (and may add another 16 above 512 wide), so
    // `padded_width * height` may overflow even when `width * height` did
    // not. Guarding here is cheaper than threading checked-arithmetic
    // through every internal allocation site.
    let padded = crate::blur::pyramid_plane_stride(width);
    crate::blur::checked_padded_plane_len(padded, height)?;
    Ok(())
}

/// Check if source and distorted images have byte-identical pixel data
/// and matching color interpretation (format + primaries).
pub(crate) fn images_byte_identical(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> bool {
    use crate::source::{AlphaMode, PixelFormat};

    let (w, h) = (source.width(), source.height());
    if w != distorted.width() || h != distorted.height() {
        return false;
    }
    if source.pixel_format() != distorted.pixel_format() {
        return false;
    }
    // Different primaries mean different perceptual colors even with identical bytes.
    if source.color_primaries() != distorted.color_primaries() {
        return false;
    }
    let fmt = source.pixel_format();
    let bpp = fmt.bytes_per_pixel();
    let row_len = w * bpp;

    // For RGBA formats with non-opaque alpha: pixels where both have A=0
    // composite to the same background, so they're visually identical
    // regardless of their RGB values.
    let alpha_aware = fmt.has_alpha()
        && !matches!(source.alpha_mode(), AlphaMode::Opaque)
        && !matches!(distorted.alpha_mode(), AlphaMode::Opaque);

    for y in 0..h {
        let sr = source.row_bytes(y);
        let dr = distorted.row_bytes(y);
        if sr[..row_len] == dr[..row_len] {
            continue; // fast path: row is byte-identical
        }
        if !alpha_aware {
            return false;
        }
        // Slow path: check pixel-by-pixel, skipping A=0 pairs
        match fmt {
            PixelFormat::Srgb8Rgba | PixelFormat::Srgb8Bgra => {
                for x in 0..w {
                    let o = x * 4;
                    if sr[o + 3] == 0 && dr[o + 3] == 0 {
                        continue;
                    }
                    if sr[o..o + 4] != dr[o..o + 4] {
                        return false;
                    }
                }
            }
            PixelFormat::Srgb16Rgba => {
                for x in 0..w {
                    let o = x * 8;
                    let sa = u16::from_ne_bytes([sr[o + 6], sr[o + 7]]);
                    let da = u16::from_ne_bytes([dr[o + 6], dr[o + 7]]);
                    if sa == 0 && da == 0 {
                        continue;
                    }
                    if sr[o..o + 8] != dr[o..o + 8] {
                        return false;
                    }
                }
            }
            PixelFormat::LinearF32Rgba => {
                for x in 0..w {
                    let o = x * 16;
                    let sa = f32::from_ne_bytes([sr[o + 12], sr[o + 13], sr[o + 14], sr[o + 15]]);
                    let da = f32::from_ne_bytes([dr[o + 12], dr[o + 13], dr[o + 14], dr[o + 15]]);
                    if sa <= 0.0 && da <= 0.0 {
                        continue;
                    }
                    if sr[o..o + 16] != dr[o..o + 16] {
                        return false;
                    }
                }
            }
            _ => return false,
        }
    }
    true
}

/// Minimum per-dimension size for the full 4-scale pyramid (8·2³ = 64).
pub(crate) const MIN_PYRAMID_DIM: usize = min_pyramid_dim_for_scales(crate::NUM_SCALES);

/// Smallest per-dimension size that still forms `num_scales` pyramid levels.
///
/// The scale walk (`streaming::compute_multiscale_stats_streaming` and every
/// `*_with_ref` sibling) halves `w`/`h` per level and stops at `w < 8 ||
/// h < 8`, so level `num_scales-1` needs `8 · 2^(num_scales-1)` at level 0.
/// At the shipped `NUM_SCALES = 4` this is 64 — the historical
/// [`MIN_PYRAMID_DIM`] constant, unchanged.
#[inline]
pub(crate) const fn min_pyramid_dim_for_scales(num_scales: usize) -> usize {
    // `num_scales` comes from a profile / `ZensimConfig` (1..=6 in practice);
    // saturate rather than shift-overflow on a nonsense value.
    match num_scales {
        0 | 1 => 8,
        n if n <= 32 => 8usize << (n - 1),
        _ => usize::MAX,
    }
}

/// **THE single owner of the "must reflect-pad before the scale walk"
/// decision.** Every v1 entry point that reaches the pyramid asks this.
///
/// A pair that answers `true` and is NOT padded produces a SHORT feature
/// vector (fewer `ScaleStats` → `n_scales · 3 · 31` instead of 372) or, on
/// the `*_with_ref` paths, trips the `scale 0 width mismatch` assertion
/// against an already-padded reference. Both were live defects until
/// 2026-08-30 — see `docs/DATASET_HISTORY.md` §3.26 and
/// `zensim/tests/v1_feature_width_pure_function.rs`.
///
/// Note the asymmetry this hides: the walk starts at
/// `pyramid_plane_stride(width)` but plain `height`, so before the padding was
/// applied `54x96` came out full-width (54 → 64 by SIMD alignment) while
/// `96x54` came out short. Callers must not re-derive the rule.
#[inline]
pub(crate) fn needs_pyramid_pad(width: usize, height: usize, num_scales: usize) -> bool {
    let min_dim = min_pyramid_dim_for_scales(num_scales);
    width > 0 && height > 0 && (width < min_dim || height < min_dim)
}

/// Owned, tightly-packed image presenting a reflect(mirror)-padded view of a
/// sub-64px input so the multi-scale pyramid always forms. This is the
/// bake-/metric-agnostic small-image fix: the pipeline computes genuine scales
/// over reflected content (a constant difference scores identically at every
/// size; scores down to 1×1 with no error) instead of synthesizing a feature
/// vector. NO-OP for images already ≥ 64px in both dims.
pub(crate) struct OwnedImage {
    data: Vec<u8>,
    width: usize,
    height: usize,
    format: crate::source::PixelFormat,
    alpha: crate::source::AlphaMode,
    primaries: crate::source::ColorPrimaries,
}

impl ImageSource for OwnedImage {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn pixel_format(&self) -> crate::source::PixelFormat {
        self.format
    }
    fn alpha_mode(&self) -> crate::source::AlphaMode {
        self.alpha
    }
    fn color_primaries(&self) -> crate::source::ColorPrimaries {
        self.primaries
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        let stride = self.width * self.format.bytes_per_pixel();
        &self.data[y * stride..y * stride + stride]
    }
}

/// reflect-101 index map (mirror without repeating the edge); `n == 1` → 0.
#[inline]
pub(crate) fn reflect_index(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut k = i % period;
    if k >= n {
        k = period - k;
    }
    k
}

/// Reflect(mirror)-pad `src` up to at least [`MIN_PYRAMID_DIM`] in each dim.
pub(crate) fn reflect_pad_to_min(src: &impl ImageSource) -> OwnedImage {
    reflect_pad_for_scales(src, crate::NUM_SCALES)
}

/// Reflect(mirror)-pad `src` up to [`min_pyramid_dim_for_scales`] in each dim.
pub(crate) fn reflect_pad_for_scales(src: &impl ImageSource, num_scales: usize) -> OwnedImage {
    reflect_pad_to_size(src, min_pyramid_dim_for_scales(num_scales))
}

pub(crate) fn reflect_pad_to_size(src: &impl ImageSource, min_dim: usize) -> OwnedImage {
    let (w, h) = (src.width(), src.height());
    let bpp = src.pixel_format().bytes_per_pixel();
    let (bw, bh) = (w.max(min_dim), h.max(min_dim));
    let mut data = vec![0u8; bw * bh * bpp];
    for y in 0..bh {
        let srow = src.row_bytes(reflect_index(y, h));
        for x in 0..bw {
            let sx = reflect_index(x, w) * bpp;
            let dx = (y * bw + x) * bpp;
            data[dx..dx + bpp].copy_from_slice(&srow[sx..sx + bpp]);
        }
    }
    OwnedImage {
        data,
        width: bw,
        height: bh,
        format: src.pixel_format(),
        alpha: src.alpha_mode(),
        primaries: src.color_primaries(),
    }
}

/// **The shared front of every SDR scoring entry.** The reflect-pad and the
/// byte-identical short-circuit run BEFORE either walk, so both walks see
/// exactly the same pixels and the same identical-pair payload.
fn compute_with_config_inner(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    stop: Option<&dyn enough::Stop>,
    // The serving plan, resolved by the CALLER (only `Zensim` knows the
    // profile's bakes). `None` = today's unconditional behaviour.
    // Deliberately a resolved plan rather than a flag: the decision belongs
    // to `fold_engine::score_plan`, and passing the answer keeps this shared
    // front ignorant of bakes.
    plan: Option<&crate::feature_plan::Plan>,
) -> ZensimResult {
    // Sub-64px inputs can't form the 4-scale pyramid. Reflect-pad both sides
    // to the pyramid minimum so every metric/bake gets 4 genuinely-computed
    // scales. NO-OP at ≥ 64px in both dims (the common path is untouched).
    let (w, h) = (source.width(), source.height());
    if needs_pyramid_pad(w, h, config.num_scales) {
        let s = reflect_pad_for_scales(source, config.num_scales);
        let d = reflect_pad_for_scales(distorted, config.num_scales);
        return compute_with_config_core(&s, &d, config, weights, stop, plan);
    }
    compute_with_config_core(source, distorted, config, weights, stop, plan)
}

fn compute_with_config_core(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    stop: Option<&dyn enough::Stop>,
    plan: Option<&crate::feature_plan::Plan>,
) -> ZensimResult {
    // Rev5 identity features are computed by the same canonical walk as
    // nonidentity pairs. The canonical walk checks a live stop token between
    // strips; public entries discard partial results when cancellation fires.
    if config.revision() >= crate::feature_defs::FormulaRevision::Rev5
        && crate::fold_engine::is_fold_backable(config)
    {
        let mut scratch = crate::feature_v2::V2Scratch::new();
        if let Ok(mut result) = crate::fold_engine::compute_fold_backed_stoppable(
            source,
            distorted,
            config,
            weights,
            &mut scratch,
            plan,
            stop,
        ) {
            if images_byte_identical(source, distorted) {
                result.score = 100.0;
                result.raw_distance = 0.0;
                result = result.mark_identical();
            }
            return result;
        }
    }

    // Identical images must score exactly 100.0 — short-circuit before
    // floating-point arithmetic introduces sub-ULP noise in SSIM/edge features.
    if images_byte_identical(source, distorted) {
        {
            // An identity result must be as wide as the bake that will read
            // it, or a wide bake is refused on a PERFECT COPY — the one pair
            // every metric must handle. The extra slots are zeros either way;
            // what changes is the length.
            return identical_result_at(config, plan.map_or(0, |p| p.walk_width()));
        }
    }

    // A plan whose layout is wider than the buffered walk can emit must take
    // the fold: falling back to buffered there would be a refusal. A live
    // cancellation token keeps the request on buffered, which checks it at
    // row-band and scale boundaries; this fold has no stop hook.
    let plan_needs_fold =
        plan.is_some_and(|p| p.walk_width() > crate::fold_engine::v1_feature_width(config));
    if plan_needs_fold && stop.is_none() && crate::fold_engine::is_fold_backable(config) {
        let mut scratch = crate::feature_v2::V2Scratch::new();
        // The fold's only failure modes here are the dimension/HDR checks
        // the caller already made; a surprise falls back rather than
        // erroring out of a path whose signature is infallible.
        if let Ok(r) = crate::fold_engine::compute_fold_backed_stoppable(
            source,
            distorted,
            config,
            weights,
            &mut scratch,
            plan,
            stop,
        ) {
            return r;
        }
    }

    crate::streaming::compute_zensim_streaming_stoppable(source, distorted, config, weights, stop)
}

/// [`identical_result`] widened to at least `min_width` slots.
///
/// A perfect copy is the one pair every metric must handle, and a wide bake
/// was refused on it for the same reason it was refused everywhere else: the
/// vector handed to `prep_bake_input_f32` was the v1 width. The extra slots
/// are zeros either way — what changes is the LENGTH, which is what the bake
/// checks. `min_width = 0` is exactly today's behaviour.
fn identical_result_at(config: &ZensimConfig, min_width: usize) -> ZensimResult {
    let fpc = FEATURES_PER_CHANNEL_WITH_PEAKS;
    let num_features = (config.num_scales * 3 * fpc).max(min_width);
    ZensimResult::new(
        100.0,
        0.0,
        vec![0.0; num_features],
        // Placeholder tag (overridden by the compute* caller); the
        // canonical codec-target profile.
        ZensimProfile::codec_target(),
        [0.0; 3],
    )
    .mark_identical()
}

pub(crate) fn config_from_params(params: &ProfileParams, parallel: bool) -> ZensimConfig {
    ZensimConfig {
        formula_revision: params
            .mlp_bytes
            .and_then(|load| cached_profile_model(load()).ok())
            .and_then(|model| crate::feature_layout::formula_revision(&model).ok())
            .filter(|revision| *revision >= crate::feature_defs::FormulaRevision::Rev5),
        local_only: false,
        omit_edges: false,
        attribution_channels: None,
        blur_radius: params.blur_radius,
        blur_passes: params.blur_passes,
        blur_kernel: BlurKernel::Box {
            passes: params.blur_passes,
        },
        downscale_filter: DownscaleFilter::default(),
        // MLP-scored profiles need the full feature vector populated;
        // the linear path can skip features it doesn't use.
        compute_all_features: params.mlp_bytes.is_some(),

        num_scales: params.num_scales,
        score_mapping_a: params.score_mapping_a,
        score_mapping_b: params.score_mapping_b,
        allow_multithreading: parallel,
    }
}

/// Replace `result.raw_distance` and `result.score` with the MLP forward
/// pass output when the profile uses an MLP scorer. No-op for linear
/// profiles.
///
/// When the loaded MLP's `n_inputs` exceeds `result.features().len()`
/// by exactly 4, this function appends size-axis features derived
/// from the source `(width, height)` before scoring. This matches
/// the V0_4 trainer's `--mlp-size-axes` mode (228 → 232 inputs):
/// log2(pixels), log2(min_dim), log2(max_dim), and signed
/// log2(max/min) by aspect orientation.
pub(crate) fn apply_mlp_scoring(
    result: &mut ZensimResult,
    params: &crate::profile::ProfileParams,
    width: u32,
    height: u32,
) -> Result<(), ZensimError> {
    apply_mlp_scoring_with_codec(result, params, width, height)
}

/// Same as [`apply_mlp_scoring`] but accepts an optional codec hint
/// that drives the per-codec post-spline affine calibration
/// (EXP-CROSS-CODEC-V11-E). The hint is threaded into every
/// `forward_one_bake_with_codec` call below — both the
/// ensemble-routing branches and the primary/b3 mix branch — so
/// every loaded bake gets a chance to apply per-codec affine if
/// it carries the metadata.
pub(crate) fn apply_mlp_scoring_with_codec(
    result: &mut ZensimResult,
    params: &crate::profile::ProfileParams,
    width: u32,
    height: u32,
) -> Result<(), ZensimError> {
    let Some(loader) = params.mlp_bytes else {
        return Ok(());
    };
    // **Identity-image short-circuit guard.** When the byte-identical
    // short-circuit in `compute_with_config_inner` /
    // `compute_zensim_with_config` fires, it produces
    // `(score=100.0, raw_distance=0.0, features=[0.0; N])`. Running
    // the MLP forward pass on the all-zero feature vector then
    // overwrites those values with garbage — typically ~0 on
    // V0_5Balanced (MSE bake on z-scored zeros), ~2 on
    // V0_5Compression / V0_5Ensemble — because the trained MLP has
    // no signal anchoring "zero feature vector → score 100". The
    // bake's biases dominate, producing an off-scale output that
    // `skip_score_mapping=true` returns verbatim.
    //
    // We detect the post-short-circuit state via the explicit
    // `is_identical` flag set by `mark_identical`. The pre-0.3.0 code
    // used a sentinel check (`raw_distance == 0.0 && features.all
    // (== 0.0)`); the flag is robust against any future profile that
    // legitimately produces an all-zero feature vector on a real
    // image.
    if result.is_identical() {
        return Ok(());
    }
    {
        let raw = forward_one_bake_with_codec(loader(), result.features(), width, height)?;
        let score = dispose_mlp_raw(raw, params);
        result.set_mlp_score(raw, score);
    }
    Ok(())
}

/// Post-forward score disposition shared by [`apply_mlp_scoring_with_codec`]
/// and the batched FD-gradient entry
/// ([`score_features_fd_gradient_with_profile`]) — extracted verbatim so the
/// two paths cannot drift (campaign appendix Y, L-Y1).
pub(crate) fn dispose_mlp_raw(raw: f64, params: &crate::profile::ProfileParams) -> f64 {
    let pre_bound = if params.skip_score_mapping {
        // The bake is already MCOS-calibrated (V0_8+); the raw
        // output IS the final score. Skipping the
        // `100 − A·d^B` transform avoids producing garbage
        // (e.g. raw=90 → mapped=-374).
        raw
    } else {
        distance_to_score_mapped(raw, params.score_mapping_a, params.score_mapping_b)
    };
    // Bound the score to [0, 100] before handing it back. Two
    // policies exist:
    //   - Hard clamp (default, legacy): out-of-range outputs pin
    //     to 0 or 100. Cheap, MCOS-natural. Tie blocks at the
    //     boundary collapse SROCC to 0 on affected bands when
    //     many pairs extrapolate (multi-bake V_20 IS regime).
    //   - Soft clamp (V_20+ multi-bake): logistic squash through
    //     `100 / (1 + exp(-(raw - 50) / 20))`. The output deviates
    //     from `raw` by < 1.5 units in the [5, 95] interior so
    //     interior calibration is unchanged; only the tails are
    //     reshaped. Preserves rank ordering at the extremes
    //     (no ties → SROCC stays defined). Costs one `exp` per
    //     score (~1 ns).
    // The choice is per-profile via `ProfileParams::soft_clamp_score`.
    // PreviewV0_4 (V_18 + V_20 IS multi-bake) sets `true`;
    // the V_18 ship single-bake and earlier set `false`.
    //
    // EXP-CROSS-CODEC-V10 (2026-05-20): `extrapolate_score` overrides
    // both hard- and soft-clamp paths. The PCHIP spline's linear
    // extrapolation past its endpoint knots flows through, so
    // "pathological" codec output (worst codec at q=0, butter > 12)
    // maps to a negative score instead of collapsing to a tie at 0.
    // V10 profiles (BalancedV3 / CompressionV3 / TunerV4) set this.
    if params.extrapolate_score {
        pre_bound
    } else {
        pre_bound.clamp(0.0, 100.0)
    }
}

fn apply_output_calibration_spline(x: f64, spline: &OutputCalibrationSpline) -> f64 {
    // ONE owner (`crate::score_math::pchip_eval_capped`), reached identically
    // from `zensim-validate::output_calibration_spline`. The mirror there had
    // the upper EXTRAPOLATION capped (2026-07-04) but not the INTERIOR, so a
    // spline declaring a knot above 100 — which the wire format permits —
    // could be published above 100 by the evaluation tooling while the product
    // runtime reported exactly 100. LATENT: 0 of the 49 bakes on disk declare
    // one. Not Hermite overshoot; Fritsch-Carlson forbids that.
    crate::score_math::pchip_eval_capped(x, &spline.xs, &spline.ys, &spline.derivs)
}

/// The parsed [`crate::mlp::Model`] of a `&'static` profile bake, interned by
/// the bytes' data pointer and length — the same cache shape (and stability
/// argument) as [`cached_bake_metadata`] and `fold_engine::cached_bake_pool_need`:
/// profile bake slices come from `&'static` slots via `ProfileParams`, so the
/// key is stable and unique per slot. The cache holds one entry per distinct
/// `&'static` bake slice: the built-in profiles, plus any loaders a caller
/// registers through `custom-profiles`. A loader that leaks a fresh slice on
/// every call (already a leak) adds one parsed, decoded model per call.
/// Parsing is deterministic, so a cached model is
/// the model a fresh parse would return; the scoring arithmetic is unchanged.
///
/// Without it every `Zensim::compute` of an MLP profile parsed its bake up to
/// three times (serving plan, disposition check, forward), which became the
/// dominant small-image cost once a parse decodes f16 layers.
///
/// A parse failure is returned, never cached. Dynamically loaded bakes do not
/// come through here; they own their `Model` through `BakeScorer`.
pub(crate) fn cached_profile_model(
    bytes: &'static [u8],
) -> Result<std::sync::Arc<crate::mlp::Model>, ZensimError> {
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock, RwLock};

    type Cache = RwLock<HashMap<(usize, usize), Arc<crate::mlp::Model>>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let key = (bytes.as_ptr() as usize, bytes.len());
    if let Ok(read) = cache.read()
        && let Some(model) = read.get(&key)
    {
        return Ok(Arc::clone(model));
    }
    let parsed = Arc::new(crate::mlp::Model::from_bytes(bytes).map_err(|_| {
        ZensimError::ModelLoadFailed {
            reason: "Model::from_bytes failed to parse the bake header or layer table",
        }
    })?);
    if let Ok(mut write) = cache.write() {
        // Another thread's parse of the same slot is equally valid; keep the first.
        return Ok(Arc::clone(write.entry(key).or_insert(parsed)));
    }
    // Poisoned lock: serve the fresh parse uncached.
    Ok(parsed)
}

/// Lookup the parsed metadata for `bytes` from the static interner,
/// or parse it on first sight and cache. Keyed by the bytes' data
/// pointer (bake byte slices always come from `&'static` slots via
/// `ProfileParams::mlp_bytes`, so the pointer is stable and unique
/// per slot).
///
/// Returns an `Arc<ScoreMetadata>` — the caller drops the cache
/// lock before doing the forward pass.
fn cached_bake_metadata(
    bytes: &[u8],
    model: &crate::mlp::Model,
) -> Result<std::sync::Arc<ScoreMetadata>, ZensimError> {
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock, RwLock};

    static CACHE: OnceLock<RwLock<HashMap<usize, Arc<ScoreMetadata>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let key = bytes.as_ptr() as usize;

    // Fast path: read-only lookup. Multiple readers can pass through
    // simultaneously — only the first-sight parse holds the write lock.
    if let Ok(read) = cache.read()
        && let Some(entry) = read.get(&key)
    {
        return Ok(Arc::clone(entry));
    }

    // Slow path: parse + insert. Re-check after acquiring the write
    // lock — another thread may have raced ahead of us.
    let parsed = std::sync::Arc::new(parse_bake_metadata(model)?);

    if let Ok(mut write) = cache.write() {
        // Idempotent: another thread's parse for the same key is also
        // valid; we just keep whichever Arc landed first.
        let entry = write.entry(key).or_insert_with(|| Arc::clone(&parsed));
        return Ok(Arc::clone(entry));
    }
    // Poisoned lock — return the freshly parsed bundle without
    // caching. Forward correctness is preserved at the cost of one
    // re-parse next call.
    Ok(parsed)
}

/// Same as [`forward_one_bake`] but accepts an optional codec hint
/// that drives the per-codec post-spline affine calibration
/// (EXP-CROSS-CODEC-V11-E). When `codec_hint` is `Some` and the
/// bake carries `zentrain.per_codec_calibration` metadata, the
/// matched codec's `(alpha, beta)` affine is applied AFTER the
/// PCHIP spline; otherwise the score is returned identical to the
/// hint-less path.
fn forward_one_bake_with_codec(
    bytes: &'static [u8],
    features: &[f64],
    width: u32,
    height: u32,
) -> Result<f64, ZensimError> {
    let model = cached_profile_model(bytes)?;
    let bundle = cached_bake_metadata(bytes, &model)?;
    let mut scorer = BakeScorer::with_metadata(&model, bundle)?;
    scorer.score_features(features, width, height)
}

// One inference dispatch for named profiles, dynamically loaded bakes and eval.
// The reusable surface owns these buffers; the named-profile adapter borrows them.
#[allow(clippy::too_many_arguments)]
fn forward_model_with_codec(
    model: &crate::mlp::Model,
    predictor: &mut crate::mlp::Predictor<'_>,
    bundle: &ScoreMetadata,
    layout: &crate::feature_layout::Layout,
    gathered: &mut Vec<f64>,
    f32_features: &mut Vec<f32>,
    features: &[f64],
    width: u32,
    height: u32,
) -> Result<f64, ZensimError> {
    let n_inputs = model.caller_input_width();
    let needs_transforms = model.has_nontrivial_feature_transforms();
    let output_spline = bundle.output_spline.as_deref();

    // Resolve dense slot gathering before selecting a model head.
    let feats: &[f64] = if layout.is_identity() {
        features
    } else {
        // **The row must REACH every declared id, or this is a refusal.**
        // `Layout::gather` writes the structural `0.0` for an id the walk did
        // not reach, which is correct for a DECLARED GAP and wrong for a
        // caller who simply handed over a short vector: every id a dense bake
        // declares is an id it reads, so filling one is exactly the "a
        // consumer cannot tell this zero from a measured zero" failure the
        // dense contract exists to end. Loud, named, and never a silent prefix.
        if features.len() < layout.walk_width() {
            return Err(ZensimError::ModelForwardFailed {
                reason: "the bake declares feature ids this feature vector does not reach",
            });
        }
        layout.gather(features, gathered);
        gathered.as_slice()
    };

    // E33 `fx1`: a derived-input bake's model row is built from its gathered
    // declared layout (direct ids and same-cell f32 products). Load refused
    // every head, transform and malformed declaration this cannot serve.
    if let Some(derived) = bundle.derived_inputs.as_deref() {
        if features.len() < layout.walk_width() {
            return Err(ZensimError::ModelForwardFailed {
                reason: "the bake declares feature ids this feature vector does not reach",
            });
        }
        if layout.is_identity() {
            layout.gather(features, gathered);
        }
        derived.fill_f32(gathered, f32_features);
        return bake_dispatch_one(predictor, f32_features, needs_transforms, output_spline);
    }

    // Min-max monotone head (Sill 1998): the score REPLACES the layer forward,
    // so bypass the Predictor entirely — read the bake's scaler + feature
    // transforms directly, apply transform→scale→clamp±8→min-max, then the
    // shared PCHIP spline + per-codec affine. The clamp mirrors the trainer's
    // standardize clamp so runtime scoring is bit-exact with the qualifying eval.

    let dispatch = |p: &mut crate::mlp::Predictor<'_>, x: &[f32]| -> Result<f64, ZensimError> {
        bake_dispatch_one(p, x, needs_transforms, output_spline)
    };
    prep_bake_input_f32(feats, n_inputs, width, height, f32_features)?;
    dispatch(predictor, f32_features)
}

/// Size a caller feature vector to a bake's declared input width and
/// convert to f32 into `out` (cleared first). Extracted from
/// [`forward_one_bake_with_codec`] so the batched FD-gradient entry
/// reuses the identical sizing semantics without a per-forward
/// allocation (campaign appendix Y, L-Y1). Branches:
/// exact width / +4 size-axes / prefix / loud mismatch.
fn prep_bake_input_f32(
    features: &[f64],
    n_inputs: usize,
    width: u32,
    height: u32,
    out: &mut Vec<f32>,
) -> Result<(), ZensimError> {
    out.clear();
    if n_inputs == features.len() {
        out.extend(features.iter().map(|&v| v as f32));
        Ok(())
    } else if n_inputs == features.len() + 4 {
        let mut augmented = features.to_vec();
        append_mlp_size_axes(&mut augmented, width, height);
        out.extend(augmented.iter().map(|&v| v as f32));
        Ok(())
    } else if n_inputs < features.len() {
        out.extend(features[..n_inputs].iter().map(|&v| v as f32));
        Ok(())
    } else {
        // n_inputs > features.len() + 4 — bake expects more features than
        // the caller supplied, including the optional 4 size axes. This
        // is a feature-shape mismatch on a successfully-loaded bake.
        Err(ZensimError::ModelForwardFailed {
            reason: "bake declares more input features than the caller supplied",
        })
    }
}

/// The per-forward post-network dispatch shared by
/// [`forward_one_bake_with_codec`] and the batched FD-gradient entry
/// ([`score_features_fd_gradient_with_profile`]) — extracted verbatim
/// from the former's closure so the two paths cannot drift (campaign
/// appendix Y, L-Y1). `x` must already be sized to the bake's input
/// width (see [`prep_bake_input_f32`]).
#[allow(clippy::too_many_arguments)]
fn bake_dispatch_one(
    p: &mut crate::mlp::Predictor<'_>,
    x: &[f32],
    needs_transforms: bool,
    output_spline: Option<&OutputCalibrationSpline>,
) -> Result<f64, ZensimError> {
    let out = if needs_transforms {
        p.predict_transformed(x)
            .map_err(|_| ZensimError::ModelForwardFailed {
                reason: "Predictor::predict_transformed failed",
            })?
    } else {
        p.predict(x).map_err(|_| ZensimError::ModelForwardFailed {
            reason: "Predictor::predict failed",
        })?
    };
    out.first()
        .copied()
        .map(|value| {
            output_spline.map_or(value as f64, |s| {
                apply_output_calibration_spline(value as f64, s)
            })
        })
        .ok_or(ZensimError::ModelForwardFailed {
            reason: "model has no score output",
        })
}

/// Extend features with log2 pixel count, minimum dimension, maximum dimension,
/// and signed aspect ratio, in the order expected by the model.
fn append_mlp_size_axes(features: &mut Vec<f64>, width: u32, height: u32) {
    if width == 0 || height == 0 {
        features.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        return;
    }
    let w = width as f64;
    let h = height as f64;
    let pixels = w * h;
    let min_dim = w.min(h);
    let max_dim = w.max(h);
    // F19: these four are MLP INPUTS, so they reach a shipped score through
    // the forward pass exactly as the mapping exponents do.
    let form = active_pow_form();
    let log2_pixels = pixels.det_log2(form);
    let log2_min = min_dim.det_log2(form);
    let log2_max = max_dim.det_log2(form);
    let log_aspect_signed = (max_dim / min_dim).det_log2(form) * if w >= h { 1.0 } else { -1.0 };
    features.extend_from_slice(&[log2_pixels, log2_min, log2_max, log_aspect_signed]);
}

/// Features per channel per scale: 19 features always emitted.
///
/// ```text
///  Index  Name             Pooling  Source
///  ─────  ───────────────  ───────  ──────────────────
///   0     ssim_mean        mean     SSIM error map
///   1     ssim_4th         L4       SSIM error map
///   2     ssim_2nd         L2       SSIM error map
///   3     art_mean         mean     edge artifact (ringing)
///   4     art_4th          L4       edge artifact
///   5     art_2nd          L2       edge artifact
///   6     det_mean         mean     edge detail lost (blur)
///   7     det_4th          L4       edge detail lost
///   8     det_2nd          L2       edge detail lost
///   9     mse              mean     (src - dst)²
///  10     hf_energy_loss   ratio    1 - Σ(dst-mu)²/Σ(src-mu)²
///  11     hf_mag_loss      ratio    1 - Σ|dst-mu|/Σ|src-mu|
///  12     hf_energy_gain   ratio    Σ(dst-mu)²/Σ(src-mu)² - 1
///  13     ssim_max         max      per-pixel SSIM error
///  14     art_max          max      per-pixel edge artifact
///  15     det_max          max      per-pixel edge detail_lost
///  16     ssim_l8          L8       (Σd⁸/N)^(1/8) SSIM error
///  17     art_l8           L8       (Σd⁸/N)^(1/8) edge artifact
///  18     det_l8           L8       (Σd⁸/N)^(1/8) edge detail_lost
/// ```
///
/// Total features = `num_scales × 3 channels × 13` = 156 at 4 scales.
///
/// Note: 6 additional "peak" features (max/l8) are always computed
/// but only included when `compute_all_features` is true. This keeps
/// the default feature vector compatible with existing profiles.
pub const FEATURES_PER_CHANNEL_BASIC: usize = 13;

/// Features per channel when `compute_all_features` is true: 19 features
/// (13 basic + 6 peak/l8). Peak features are always computed (near-zero cost)
/// but excluded from the default feature vector for profile compatibility.
pub const FEATURES_PER_CHANNEL_WITH_PEAKS: usize = 19;

/// Named view over a flat feature vector.
///
/// Provides ergonomic access to features by name, scale, and channel
/// without changing the underlying storage format.
///
/// ```ignore
/// let result = z.compute_all_features(&src, &dst)?;
/// let view = FeatureView::new(result.features(), 4)?;
/// let ssim_mean_s0_y = view.ssim_mean(0, 1);
/// let ssim_max_s2_x = view.ssim_max(0, 2).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FeatureView<'a> {
    features: &'a [f64],
    n_scales: usize,
    /// Number of features in the scored block
    scored_total: usize,
    /// Number of features in the peaks block (0 if not present)
    peaks_total: usize,
}

/// XYB channel index: X (red-green chrominance).
#[cfg(feature = "training")]
pub const CH_X: usize = 0;
/// XYB channel index: Y (luminance).
#[cfg(feature = "training")]
pub const CH_Y: usize = 1;
/// XYB channel index: B (blue-yellow chrominance).
#[cfg(feature = "training")]
pub const CH_B: usize = 2;

impl<'a> FeatureView<'a> {
    /// Create a view over a feature vector.
    ///
    /// Automatically detects the tier (peaks/extended) from length.
    /// Returns `None` if the length doesn't match any valid layout.
    /// Peaks are always present (basic-only 156-element vectors are no longer generated).
    pub fn new(features: &'a [f64], n_scales: usize) -> Option<Self> {
        let basic_total = n_scales * 3 * FEATURES_PER_CHANNEL_BASIC;
        let peaks_total = n_scales * 3 * 6;

        let (scored_total, peaks_total) = if features.len() == basic_total {
            // Legacy basic-only layout (backward compat)
            (basic_total, 0)
        } else if features.len() == basic_total + peaks_total {
            (basic_total, peaks_total)
        } else {
            return None;
        };

        Some(Self {
            features,
            n_scales,
            scored_total,
            peaks_total,
        })
    }

    /// Number of scales in this feature vector.
    pub fn n_scales(&self) -> usize {
        self.n_scales
    }

    /// Whether peak features (max/L8) are present.
    pub fn has_peaks(&self) -> bool {
        self.peaks_total > 0
    }

    // --- Scored features (always present) ---

    fn scored_idx(&self, scale: usize, ch: usize, offset: usize) -> usize {
        scale * 3 * FEATURES_PER_CHANNEL_BASIC + ch * FEATURES_PER_CHANNEL_BASIC + offset
    }

    /// SSIM error, mean pooling.
    pub fn ssim_mean(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 0)]
    }
    /// SSIM error, L4 norm.
    pub fn ssim_4th(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 1)]
    }
    /// SSIM error, L2 norm.
    pub fn ssim_2nd(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 2)]
    }
    /// Edge artifact (ringing), mean pooling.
    pub fn art_mean(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 3)]
    }
    /// Edge artifact, L4 norm.
    pub fn art_4th(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 4)]
    }
    /// Edge artifact, L2 norm.
    pub fn art_2nd(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 5)]
    }
    /// Edge detail lost (blur), mean pooling.
    pub fn det_mean(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 6)]
    }
    /// Edge detail lost, L4 norm.
    pub fn det_4th(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 7)]
    }
    /// Edge detail lost, L2 norm.
    pub fn det_2nd(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 8)]
    }
    /// Mean squared error.
    pub fn mse(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 9)]
    }
    /// High-frequency energy loss ratio.
    pub fn hf_energy_loss(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 10)]
    }
    /// High-frequency magnitude loss ratio.
    pub fn hf_mag_loss(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 11)]
    }
    /// High-frequency energy gain ratio.
    pub fn hf_energy_gain(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 12)]
    }

    // --- Peak features (always present) ---

    fn peak_idx(&self, scale: usize, ch: usize, offset: usize) -> Option<usize> {
        if self.peaks_total == 0 {
            return None;
        }
        Some(self.scored_total + scale * 3 * 6 + ch * 6 + offset)
    }

    /// SSIM error, pixel-wise max.
    pub fn ssim_max(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 0).map(|i| self.features[i])
    }
    /// Edge artifact, pixel-wise max.
    pub fn art_max(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 1).map(|i| self.features[i])
    }
    /// Edge detail lost, pixel-wise max.
    pub fn det_max(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 2).map(|i| self.features[i])
    }
    /// SSIM error, L8 norm `(Σd⁸/N)^(1/8)`.
    pub fn ssim_l8(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 3).map(|i| self.features[i])
    }
    /// Edge artifact, L8 norm.
    pub fn art_l8(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 4).map(|i| self.features[i])
    }
    /// Edge detail lost, L8 norm.
    pub fn det_l8(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 5).map(|i| self.features[i])
    }

    /// Get the scored features slice (first N features, WEIGHTS-compatible).
    pub fn scored_features(&self) -> &[f64] {
        &self.features[..self.scored_total]
    }

    /// Get the peak features slice, if present.
    pub fn peak_features(&self) -> Option<&[f64]> {
        if self.peaks_total == 0 {
            None
        } else {
            Some(&self.features[self.scored_total..self.scored_total + self.peaks_total])
        }
    }
}

/// Compute zensim with custom configuration (training API).
///
/// Uses the legacy v0.2 linear weights. Custom configuration changes metric
/// internals, so its scores are not comparable with default-profile scores.
#[doc(hidden)]
#[cfg(any(feature = "training", test))]
pub fn compute_zensim_with_config(
    source: &[[u8; 3]],
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
    config: ZensimConfig,
) -> Result<ZensimResult, ZensimError> {
    if width < 8 || height < 8 {
        return Err(ZensimError::ImageTooSmall);
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(ZensimError::ImageTooLarge)?;
    // Also reject overflow on the padded plane size (pyramid_plane_stride(width) * height).
    check_within_max_pixels(width, height, None)?;
    if source.len() != pixels {
        return Err(ZensimError::InvalidDataLength);
    }
    if distorted.len() != pixels {
        return Err(ZensimError::InvalidDataLength);
    }
    if source.len() != distorted.len() {
        return Err(ZensimError::DimensionMismatch);
    }

    // Identical images must score exactly 100.0 — short-circuit before
    // floating-point arithmetic introduces sub-ULP noise in SSIM/edge features.
    if source == distorted {
        // Match the feature width to the enabled config flags. See the
        // sister short-circuit above `compute_zensim_streaming` for the
        // bug history (the 372-input compute_iw_features path was
        // discovered when a 372-input bake failed with InvalidDataLength
        // because the short-circuit only counted basic+extended = 300).
        let fpc = FEATURES_PER_CHANNEL_WITH_PEAKS;
        let num_features = config.num_scales * 3 * fpc;
        // This entry point scores via the plain WEIGHTS (V0_2 linear)
        // distance, never an MLP forward pass — tag with the honest
        // legacy-linear profile rather than an MLP-scored profile (the
        // profile).
        return Ok(ZensimResult::new(
            100.0,
            0.0,
            vec![0.0; num_features],
            ZensimProfile::LegacyLinearV0_2,
            [0.0; 3],
        )
        .mark_identical());
    }

    let src_img = crate::source::RgbSlice::new(source, width, height);
    let dst_img = crate::source::RgbSlice::new(distorted, width, height);

    // Reflect-pad a sub-pyramid-minimum pair, exactly as
    // `compute_with_config_inner` (every `Zensim::compute*`) does. Without
    // this the scale walk stopped early and this entry returned a SHORT
    // feature vector — 93 / 186 / 279 wide instead of 372 — with no error.
    // Both v1-372 extractors call this, which is what made ~6.5 % of the
    // R1b eval-slice rows ragged (docs/DATASET_HISTORY.md §3.26).
    let result = if needs_pyramid_pad(width, height, config.num_scales) {
        let s = reflect_pad_for_scales(&src_img, config.num_scales);
        let d = reflect_pad_for_scales(&dst_img, config.num_scales);
        crate::streaming::compute_zensim_streaming(&s, &d, &config, WEIGHTS)
    } else {
        crate::streaming::compute_zensim_streaming(&src_img, &dst_img, &config, WEIGHTS)
    };
    // See the identical-shortcut branch above: this function scores via
    // plain linear WEIGHTS, not an MLP bake, so it does not inherit
    // `combine_scores`'s internal `codec_target()` (`B`) placeholder tag
    // (which every `Zensim::compute*` caller overrides but this standalone
    // function does not go through).
    Ok(result.with_profile(ZensimProfile::LegacyLinearV0_2))
}

/// Combine per-scale statistics into a final score.
///
/// Uses learned weights that balance:
/// - Per-channel sensitivity (Y > X > B, matching human vision)
/// - Per-scale importance (medium scales most important)
/// - SSIM vs edge features
/// - Mean vs 4th-power pooling
///
/// Weights are trained against synthetic quality scores (see `weights/` directory).
/// Features per scale for the default scoring profile (3 channels × 13 features = 39).
pub const FEATURES_PER_SCALE: usize = FEATURES_PER_CHANNEL_WITH_PEAKS * 3;

/// Default scoring weights — references the latest profile weights.
///
/// Layout: 4 scales × 3 channels (X,Y,B) × 13 basic features, then
///         4 scales × 3 channels × 6 peak features = 228 total.
#[cfg(any(feature = "training", test))]
pub const WEIGHTS: &[f64; 228] = &crate::profile::WEIGHTS_PREVIEW_V0_2;

/// **THE single owner of "turn a v1-layout feature vector into
/// `(score, raw_distance)`".** Extracted verbatim from [`combine_scores`]
/// (2026-08-30, the fold-engine lane) so the buffered walk and the
/// fold-backed engine share one implementation of the linear scoring tail
/// instead of each restating the dot product — per the repo's
/// NO-DUPLICATE-IMPLEMENTATIONS rule, where a second *Rust* site counts as a
/// duplicate exactly as much as a Python one.
///
/// Does three things, in this order and no other:
///
/// 1. **Sanitize non-finite values** (NaN/Inf → 0) in place, before scoring
///    *and* before the MLP forward reads the same slice. A single bad feature
///    would otherwise poison the MLP (NaN cascades through every weighted
///    sum) or the dot product (Inf swamps it). Replacing with 0 matches
///    `zensim_gpu`'s defensive behaviour on degenerate inputs (one corpus
///    pair, 2048×1365 JPEG q60, hit Inf at masked-ssim_mean B-channel).
/// 2. **Dot with the weights** over the scored prefix — basic + peak, i.e.
///    `n_scales · 3 · FEATURES_PER_CHANNEL_WITH_PEAKS` slots, truncated to
///    `weights.len()` — in **ascending index order**, then divide by
///    `max(n_scales, 1)`. The order is load-bearing: it is what makes the
///    two engines' `raw_distance` bit-identical rather than
///    equal-to-an-epsilon, so do not reassociate it.
/// 3. **Map distance to score**: the correct-by-construction bounded squash
///    (linear profiles that opt in) or the legacy unbounded `100 − a·d^b`.
///    The squash is a strictly-monotone transform of the same
///    `raw_distance`, so the *rank* is identical; only boundedness differs.
///    See [`bounded_score_squash`] and
///    `docs/METRIC_INVARIANTS_MECHANISM_AND_REDESIGN_2026-05-26.md`.
///
/// `features` may be any v1-layout width (228 / 300 / 372) — the scored
/// prefix is the same in all of them, and the tail is sanitized but not
/// weighted.
pub(crate) fn score_v1_layout_features(
    features: &mut [f64],
    weights: &[f64],
    config: &ZensimConfig,
    n_scales: usize,
) -> (f64, f64) {
    for f in features.iter_mut() {
        if !f.is_finite() {
            *f = 0.0;
        }
    }

    let scored_total = n_scales * 3 * FEATURES_PER_CHANNEL_WITH_PEAKS;
    let n_score = scored_total.min(weights.len()).min(features.len());
    let mut raw_distance = 0.0f64;
    for (i, &feat) in features[..n_score].iter().enumerate() {
        raw_distance += feat * weights[i];
    }
    raw_distance /= n_scales.max(1) as f64;

    let score =
        distance_to_score_mapped(raw_distance, config.score_mapping_a, config.score_mapping_b);
    (score, raw_distance)
}

pub(crate) fn combine_scores(
    scale_stats: &[ScaleStats],
    weights: &[f64],
    config: &ZensimConfig,
    mean_offset: [f64; 3],
) -> ZensimResult {
    let n_scales = scale_stats.len();
    let basic_per_ch = FEATURES_PER_CHANNEL_BASIC; // 13
    let basic_total = n_scales * basic_per_ch * 3;
    let peak_total = n_scales * 6 * 3;
    let mut features = Vec::with_capacity(basic_total + peak_total);

    // Pass 1: scored features (13/ch, weight-compatible order)
    for ss in scale_stats.iter() {
        for c in 0..3 {
            features.push(ss.ssim[c * 2].abs());
            features.push(ss.ssim[c * 2 + 1].abs());
            features.push(ss.ssim_2nd[c].abs());
            features.push(ss.edge[c * 4].abs());
            features.push(ss.edge[c * 4 + 1].abs());
            features.push(ss.edge_2nd[c * 2].abs());
            features.push(ss.edge[c * 4 + 2].abs());
            features.push(ss.edge[c * 4 + 3].abs());
            features.push(ss.edge_2nd[c * 2 + 1].abs());
            features.push(ss.mse[c]);
            features.push(ss.hf_energy_loss[c]);
            features.push(ss.hf_mag_loss[c]);
            features.push(ss.hf_energy_gain[c]);
        }
    }

    // Pass 2: peak features (6/ch — max + L8, always computed at near-zero cost)
    for ss in scale_stats.iter() {
        for c in 0..3 {
            features.push(ss.ssim_max[c]);
            features.push(ss.art_max[c]);
            features.push(ss.det_max[c]);
            features.push(ss.ssim_p95[c]);
            features.push(ss.art_p95[c]);
            features.push(ss.det_p95[c]);
        }
    }

    let (score, raw_distance) =
        score_v1_layout_features(&mut features, weights, config, scale_stats.len());
    debug_assert_eq!(
        basic_total + peak_total,
        n_scales * 3 * FEATURES_PER_CHANNEL_WITH_PEAKS
    );

    // Placeholder profile tag — every `Zensim::compute*` caller overrides
    // it via `with_profile(self.profile)` before returning to the user.
    // Uses the canonical codec-target profile (`A`).
    ZensimResult::new(
        score,
        raw_distance,
        features,
        ZensimProfile::codec_target(),
        mean_offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_spline_lower_extrapolation_floored_ood_safety_net() {
        // 2-knot identity-ish (0,0)-(100,100): floor = ys[0] - (ys[n-1]-ys[0])
        // = 0 - 100 = -100.
        let mut bytes = (2u32).to_le_bytes().to_vec();
        for (x, y) in [(0.0f32, 0.0f32), (100.0f32, 100.0f32)] {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
        }
        let spline = parse_output_calibration_spline(&bytes).expect("parse");
        // Modest negatives extrapolate linearly (neg-tail intact).
        assert!((apply_output_calibration_spline(-10.0, &spline) - (-10.0)).abs() < 1e-6);
        assert!((apply_output_calibration_spline(-8.63, &spline) - (-8.63)).abs() < 1e-6);
        // Pathological far-negative raw floored at -100, not wild.
        assert!(
            (apply_output_calibration_spline(-1.0e6, &spline) - (-100.0)).abs() < 1e-6,
            "OOD pathology must floor at -100, not extrapolate to a wild score"
        );
        // Monotone across the floor transition.
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=200 {
            let x = -200.0 + i as f64;
            let y = apply_output_calibration_spline(x, &spline);
            assert!(y >= prev - 1e-9, "non-monotone at x={x}");
            prev = y;
        }
    }

    #[test]
    #[allow(deprecated)]
    fn public_configs_preserve_revision_route_invariants() {
        use crate::feature_defs::FormulaRevision;
        for profile in [
            ZensimProfile::PreviewV0_1,
            ZensimProfile::PreviewV0_2,
            ZensimProfile::A,
            ZensimProfile::LegacyLinearV0_2,
        ] {
            let params = profile.params();
            let config = config_from_params(params, false);
            if config.revision() >= FormulaRevision::Rev5 {
                assert!(params.mlp_bytes.is_some(), "{profile:?}");
                assert!(crate::fold_engine::is_fold_backable(&config), "{profile:?}");
            }
        }
        let custom = ProfileParams::custom(&[], 3, 3, 2, 1.0, 1.0);
        assert_eq!(
            config_from_params(&custom, false).revision(),
            FormulaRevision::Rev1
        );
        assert_eq!(ZensimConfig::default().revision(), FormulaRevision::Rev1);
    }

    /// Verify compute_all_features produces same score as default (weight-skipped) path.
    /// This exercises the multi-SSIM channel code path where ssim_chs.len() > 1.
    #[test]
    fn compute_all_matches_default() {
        // Generate a simple test pattern: gradient source, slightly different distorted
        let w = 128;
        let h = 128;
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = 128;
                src[y * w + x] = [r, g, b];
                // Slight distortion
                dst[y * w + x] = [r.saturating_add(5), g, b.saturating_sub(3)];
            }
        }

        let default_result =
            compute_zensim_with_config(&src, &dst, w, h, ZensimConfig::default()).unwrap();
        let all_result = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Same score (default weights skip zero-weight channels; compute_all computes them
        // but zero weights still produce same weighted distance)
        assert!(
            (default_result.score - all_result.score).abs() < 0.01,
            "default {} vs all_features {}",
            default_result.score,
            all_result.score,
        );

        // Both default and compute_all now include peak features (228)
        assert_eq!(all_result.features.len(), 228);
        assert_eq!(default_result.features.len(), 228);
        // With compute_all, previously-skipped channels should now have nonzero features
        let all_nonzero = all_result
            .features
            .iter()
            .filter(|f| f.abs() > 1e-12)
            .count();
        let default_nonzero = default_result
            .features
            .iter()
            .filter(|f| f.abs() > 1e-12)
            .count();
        assert!(
            all_nonzero >= default_nonzero,
            "compute_all should have >= features: {} vs {}",
            all_nonzero,
            default_nonzero,
        );
    }

    /// `compute_zensim_with_config` scores via plain linear `WEIGHTS`
    /// (V0_2), never an MLP forward pass — it must NOT claim the
    /// MLP-scored profile tag.
    /// Covers both the general path and the identical-images
    /// short-circuit.
    #[test]
    fn compute_zensim_with_config_tags_legacy_linear_profile_not_a() {
        let w = 128;
        let h = 128;
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                src[y * w + x] = [r, 128, 128];
                dst[y * w + x] = [r.saturating_add(5), 128, 128];
            }
        }

        // General (non-identical) path.
        let result = compute_zensim_with_config(&src, &dst, w, h, ZensimConfig::default()).unwrap();
        assert_eq!(
            result.profile(),
            crate::profile::ZensimProfile::LegacyLinearV0_2,
            "linear-scored result must not be tagged as an MLP-scored profile"
        );

        // Identical-images short-circuit path (its own hardcoded tag).
        let identical =
            compute_zensim_with_config(&src, &src, w, h, ZensimConfig::default()).unwrap();
        assert_eq!(
            identical.profile(),
            crate::profile::ZensimProfile::LegacyLinearV0_2,
            "identical-images short-circuit must not be tagged as an MLP-scored profile"
        );
        assert_eq!(identical.score(), 100.0);
    }

    /// Same honesty check for `compute_zensim_with_ref_and_config`.
    #[cfg(feature = "training")]
    #[test]
    fn compute_zensim_with_ref_and_config_tags_legacy_linear_profile_not_a() {
        let w = 128;
        let h = 128;
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                src[y * w + x] = [r, 128, 128];
                dst[y * w + x] = [r.saturating_add(5), 128, 128];
            }
        }
        let config = ZensimConfig::default();
        let precomputed = precompute_reference_with_scales(&src, w, h, config.num_scales).unwrap();
        let result = compute_zensim_with_ref_and_config(&precomputed, &dst, w, h, config).unwrap();
        assert_eq!(
            result.profile(),
            crate::profile::ZensimProfile::LegacyLinearV0_2,
            "linear-scored result must not be tagged as an MLP-scored profile"
        );
    }

    /// Regression: `derive_classification` must not panic when any
    /// per-detector score is NaN. The pre-cleanup implementation used
    /// `partial_cmp(...).unwrap()` which would unwind on NaN.
    ///
    /// We feed `alpha_error_correlation = NaN` (the only path where a
    /// NaN can reach `score_alpha`). The function must return cleanly
    /// — the NaN score is sorted to the most-negative end by
    /// `total_cmp`, so the legitimate non-NaN scores still pick the
    /// dominant category.
    #[cfg(feature = "classification")]
    #[test]
    fn derive_classification_is_nan_safe() {
        let delta_stats = DeltaStats {
            mean_delta: [0.0; 3],
            stddev_delta: [0.0; 3],
            max_abs_delta: [0.5 / 255.0; 3],
            signed_small_histogram: [[0u64; 7]; 3],
            native_max: 255.0,
            pixel_count: 10_000,
            pixels_differing: 100,
            pixels_differing_by_more_than_1: 0,
            has_alpha: true,
            alpha_max_delta: 0,
            alpha_pixels_differing: 0,
            src_histogram: [[0u64; 256]; 4],
            dst_histogram: [[0u64; 256]; 4],
            opaque_stats: None,
            semitransparent_stats: None,
            // Inject NaN — the audit-flagged failure mode. The pre-fix
            // code called `partial_cmp(...).unwrap()` and panicked here.
            alpha_error_correlation: Some(f64::NAN),
        };
        let dummy_result = ZensimResult::nan();
        let cls = derive_classification(&delta_stats, &dummy_result);
        // Must NOT have selected a NaN-confidence category. With the
        // `total_cmp` ordering, NaN sorts low and never wins.
        assert!(
            cls.confidence.is_finite(),
            "confidence must be finite even when an internal score is NaN, got {}",
            cls.confidence
        );
    }
}
