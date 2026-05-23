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
/// The default `Box2x2` averages 2×2 pixel blocks, halving resolution.
/// Enable the `zenresize` feature for `Mitchell` and `Lanczos` variants.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum DownscaleFilter {
    /// 2×2 box averaging (fastest, current default).
    #[default]
    Box2x2,
    /// Mitchell-Netravali bicubic (B=1/3, C=1/3). Good balance of sharpness
    /// and ringing. Requires the `zenresize` feature.
    #[cfg(feature = "zenresize")]
    #[allow(dead_code)]
    Mitchell,
    /// Lanczos-3 windowed sinc. Sharper than Mitchell but may ring on edges.
    /// Requires the `zenresize` feature.
    #[cfg(feature = "zenresize")]
    #[allow(dead_code)]
    Lanczos,
    /// Mitchell-Netravali bicubic followed by a Gaussian blur with the given
    /// sigma. This anti-aliases the pyramid more aggressively than plain
    /// Mitchell, which may help metrics that are sensitive to high-frequency
    /// ringing. Requires the `zenresize` feature.
    #[cfg(feature = "zenresize")]
    #[allow(dead_code)]
    MitchellBlur(f32),
}

/// Configuration for the zensim metric computation pipeline.
///
/// Controls blur kernel, pyramid construction, and feature extraction.
/// The defaults match the trained profile and give peak performance;
/// only change these for training or research.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ZensimConfig {
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
    #[allow(dead_code)] // planned: not yet wired into blur dispatch
    pub blur_kernel: BlurKernel,

    /// Downscale filter for pyramid construction (default: `DownscaleFilter::Box2x2`).
    ///
    /// Controls how each pyramid level is produced. Enable the `zenresize`
    /// feature for `Mitchell` and `Lanczos` variants.
    #[allow(dead_code)] // planned: not yet wired into pyramid construction
    pub downscale_filter: DownscaleFilter,

    /// Compute all 156 features even when their weights are zero (default: false).
    ///
    /// When false, channels/features with zero weight are skipped entirely.
    /// Enable for weight training to avoid circular dependency (need all features
    /// to determine which weights should be nonzero).
    pub compute_all_features: bool,

    /// Compute extended features (25 per channel instead of 13; default: false).
    ///
    /// When true, adds 12 extra features per channel per scale:
    /// - 6 masked features (SSIM/edge/MSE weighted by source flatness)
    /// - 6 percentile/max features (worst-case SSIM/edge errors)
    ///
    /// The masking strength for extended features is controlled by
    /// `extended_masking_strength`.
    pub extended_features: bool,

    /// Masking strength for extended masked features (default: 4.0).
    ///
    /// Only used when `extended_features` is true. Controls the flatness mask:
    /// `mask[i] = 1 / (1 + k * blur(|src - mu|))`.
    ///
    /// Higher values = more aggressive masking of textured regions.
    /// Typical range: 2.0–8.0.
    pub extended_masking_strength: f32,

    /// Compute information-content-weighted (IW) features (Wang & Li
    /// 2011, IW-SSIM). When `true`, emits a per-scale per-channel block
    /// of 6 features (`iw_ssim_mean / iw_ssim_4th / iw_ssim_2nd /
    /// iw_art_4th / iw_det_4th / iw_mse`) pooled with weights derived
    /// from the source image's local activity — the OPPOSITE polarity
    /// of the `extended_features` masked block (which suppresses
    /// texture). Texture-rich regions get MORE weight, mirroring
    /// Wang's GSM-on-wavelet info-content estimator.
    ///
    /// At `num_scales = 4` and 3 channels, enabling this adds 72
    /// features (228 → 300 in the basic profile).
    ///
    /// **Zero overhead when `false`** — the per-pixel work is gated by
    /// a constant-per-call branch in the SIMD loop; the branch
    /// predictor sees a fixed direction.
    pub compute_iw_features: bool,

    /// IW weighting strength: `iw_weight[i] = 1.0 + k * blur(|src - mu|)`.
    /// Only used when `compute_iw_features` is true. Higher values
    /// concentrate the pool more aggressively on textured / edge
    /// regions. Default: 4.0 (mirrors `extended_masking_strength`).
    pub iw_strength: f32,

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

impl Default for ZensimConfig {
    fn default() -> Self {
        Self {
            blur_radius: 5,
            blur_passes: 1,
            blur_kernel: BlurKernel::default(),
            downscale_filter: DownscaleFilter::default(),
            compute_all_features: false,
            extended_features: false,
            extended_masking_strength: 4.0,
            compute_iw_features: false,
            iw_strength: 4.0,
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
    if raw_distance <= 0.0 {
        100.0
    } else {
        100.0 - a * raw_distance.powf(b)
    }
}

/// Compute score from raw features using custom weights.
///
/// # Panics
///
/// Panics if `features.len() != weights.len()`. Prefer
/// [`try_score_from_features`] for caller-supplied lengths — this thin
/// wrapper exists for backwards compatibility and will be removed in a
/// future major release.
#[cfg_attr(not(feature = "training"), allow(dead_code))]
#[deprecated(
    since = "0.2.9",
    note = "use `try_score_from_features` which returns a Result instead of panicking on length mismatch"
)]
pub fn score_from_features(features: &[f64], weights: &[f64]) -> (f64, f64) {
    try_score_from_features(features, weights)
        .expect("score_from_features: features and weights must have same length")
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
#[cfg_attr(not(feature = "training"), allow(dead_code))]
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
        FEATURES_PER_CHANNEL_EXTENDED * 3,   // 75
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

/// Pre-compute reference with a custom number of pyramid scales.
///
/// Use this when calling [`compute_zensim_with_ref_and_config`] with a non-default
/// `num_scales`. The precomputed data must have at least as many scales as the config
/// requests.
#[cfg_attr(not(feature = "training"), allow(dead_code))]
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
        &src_img, num_scales, true,
    ))
}

/// Compute zensim with a precomputed reference and custom configuration.
///
/// Training/research variant. The `config.num_scales`
/// must not exceed the number of scales in `precomputed`.
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
    let result = crate::streaming::compute_zensim_streaming_with_ref(
        precomputed,
        &dst_img,
        &config,
        WEIGHTS,
    );
    Ok(result)
}

/// Per-scale statistics collected during computation.
#[derive(Default)]
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
    // --- Extended features (only populated when extended_features=true) ---
    /// Masked SSIM: [mean, 4th, 2nd] per channel = 9 values
    pub(crate) masked_ssim: [f64; 9],
    /// Masked edge artifact L4 per channel = 3 values
    pub(crate) masked_art_4th: [f64; 3],
    /// Masked edge detail_lost L4 per channel = 3 values
    pub(crate) masked_det_4th: [f64; 3],
    /// Masked MSE per channel = 3 values
    pub(crate) masked_mse: [f64; 3],
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
    // --- IW (information-content-weighted) features (only populated when
    //     compute_iw_features=true). Texture-EMPHASISING counterpart to
    //     the masked_* block. Wang & Li 2011 IW-SSIM. ---
    /// IW SSIM: [mean, 4th, 2nd] per channel = 9 values
    pub(crate) iw_ssim: [f64; 9],
    /// IW edge artifact L4 per channel = 3 values
    pub(crate) iw_art_4th: [f64; 3],
    /// IW edge detail_lost L4 per channel = 3 values
    pub(crate) iw_det_4th: [f64; 3],
    /// IW MSE per channel = 3 values
    pub(crate) iw_mse: [f64; 3],
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
        }
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
        (100.0 - 19.0379 * self.raw_distance.powf(0.5979)).max(-100.0)
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
        0.000922 * self.raw_distance.powf(1.2244)
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
        2.365353 * self.raw_distance.powf(0.6130)
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

use crate::profile::{ProfileParams, ZensimProfile};
use crate::source::ImageSource;

/// Metric configuration. Methods on this struct are the primary API.
///
/// ```no_run
/// use zensim::{Zensim, ZensimProfile, RgbSlice};
/// # let (src, dst) = (vec![[0u8; 3]; 64], vec![[0u8; 3]; 64]);
/// let z = Zensim::new(ZensimProfile::latest());
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
}

impl Zensim {
    /// Create a new `Zensim` with the given profile. Parallel by default.
    /// No `max_pixels` cap is set by default — callers feeding
    /// untrusted dimensions should explicitly call
    /// [`Zensim::with_max_pixels`].
    pub fn new(profile: ZensimProfile) -> Self {
        Self {
            profile,
            parallel: true,
            max_pixels: None,
        }
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
    /// Default: `None` (no cap). Recommended for services accepting
    /// network-supplied dimensions: zensim allocates roughly
    /// `width × height × 4 bytes × ~14` for the streaming XYB pyramid plus
    /// destination planes, so a 4K (≈8.3 MP) image needs ~470 MB of
    /// scratch. A cap of e.g. 64 MP (8192×8192) keeps the worst case at
    /// ~3.6 GB.
    ///
    /// Pass [`usize::MAX`] explicitly to disable the cap on a previously
    /// configured `Zensim`.
    pub fn with_max_pixels(mut self, max_pixels: usize) -> Self {
        self.max_pixels = Some(max_pixels);
        self
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
        self.compute_with_codec_hint(source, distorted, None)
    }

    /// Compare source and distorted images, supplying an optional
    /// codec hint that drives **per-codec post-spline affine**
    /// calibration (EXP-CROSS-CODEC-V11-E, 2026-05-20).
    ///
    /// When the loaded profile's bake carries
    /// `zentrain.per_codec_calibration` metadata AND `codec_hint`
    /// names a registered codec, the runtime applies an extra
    /// `score = α_c + β_c · spline(raw)` affine after the standard
    /// scoring path. The affine is monotone within codec
    /// (β > 0 by construction) so within-codec rank ordering is
    /// bit-exact preserved; only the cross-codec systematic bias
    /// is pulled toward consensus at JND landmarks.
    ///
    /// Accepted codec hint strings (case-insensitive):
    /// `"jpeg"` / `"jpg"` / `"zenjpeg"` / `"mozjpeg"` / `"libjpeg"`,
    /// `"webp"` / `"zenwebp"`,
    /// `"avif"` / `"zenavif"`,
    /// `"jxl"` / `"zenjxl"` / `"jpegxl"`,
    /// `"png"` / `"zenpng"`. Any other hint is treated as unknown
    /// (no affine applied — identical to calling
    /// [`Self::compute`]).
    ///
    /// Profiles without the metadata silently ignore the hint
    /// (output is identical to [`Self::compute`]).
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too small.
    pub fn compute_with_codec_hint(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        codec_hint: Option<&str>,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        check_within_max_pixels(source.width(), source.height(), self.max_pixels)?;
        let config = config_from_params(params, self.parallel);
        let mut result = compute_with_config_inner(source, distorted, &config, params.weights);
        apply_mlp_scoring_with_codec(
            &mut result,
            params,
            source.width() as u32,
            source.height() as u32,
            codec_hint,
        )?;
        Ok(result.with_profile(self.profile))
    }

    /// Compare source and distorted images, returning the **extended** feature
    /// set (300 features at the default 4-scale, 3-channel layout) instead of
    /// the standard 228 weighted-only set.
    ///
    /// The extended set adds 6 masked features (SSIM/edge/MSE weighted by
    /// source flatness) per channel per scale on top of the 228 features
    /// returned by [`Zensim::compute`]. The score is identical to `compute()`
    /// — only the first 228 features have non-zero weights — but the extra 72
    /// features are useful inputs for retraining downstream models (selectors,
    /// regressors) without re-running the costly multi-scale stats pass.
    ///
    /// Layout (default profile, 4 scales × 3 channels):
    /// - 0..156   — basic features (13/channel/scale): SSIM/edge/HF errors
    /// - 156..228 — peak features (6/channel/scale): SSIM/edge max + p95
    /// - 228..300 — masked features (6/channel/scale): SSIM/edge/MSE flatness-weighted
    ///
    /// Use [`FeatureView`] for named access. Cost overhead vs `compute()` is
    /// modest — the masking pass reuses the already-computed flatness map.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too small.
    pub fn compute_extended_features(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        check_within_max_pixels(source.width(), source.height(), self.max_pixels)?;
        let mut config = config_from_params(params, self.parallel);
        config.extended_features = true;
        let result = compute_with_config_inner(source, distorted, &config, params.weights);
        Ok(result.with_profile(self.profile))
    }

    /// Pre-compute reference image data for batch comparison.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if dimensions < 8×8.
    pub fn precompute_reference(
        &self,
        source: &impl ImageSource,
    ) -> Result<crate::streaming::PrecomputedReference, ZensimError> {
        let params = self.profile.params();
        if source.width() < 8 || source.height() < 8 {
            return Err(ZensimError::ImageTooSmall);
        }
        check_within_max_pixels(source.width(), source.height(), self.max_pixels)?;
        Ok(crate::streaming::PrecomputedReference::new(
            source,
            params.num_scales,
            self.parallel,
        ))
    }

    /// Compare a distorted image against a precomputed reference.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if dimensions < 8×8.
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
        if distorted.width() < 8 || distorted.height() < 8 {
            return Err(ZensimError::ImageTooSmall);
        }
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels)?;
        let config = config_from_params(params, self.parallel);
        let mut result = crate::streaming::compute_zensim_streaming_with_ref(
            precomputed,
            distorted,
            &config,
            params.weights,
        );
        apply_mlp_scoring(
            &mut result,
            params,
            distorted.width() as u32,
            distorted.height() as u32,
        )?;
        Ok(result.with_profile(self.profile))
    }

    /// Compare source and distorted images using **strip-aggregating
    /// streaming** for very large images (e.g. 80 MP) where the
    /// standard [`compute`](Self::compute) path would OOM.
    ///
    /// Internally splits both images into horizontal Y-strips of
    /// `strip_inner` scale-0 rows (with `strip_margin` rows of overlap
    /// on each side for the blur stencil), runs the existing pipeline
    /// on each strip independently, and aggregates per-scale
    /// accumulators across strips. Memory peak per pair drops from
    /// `O(full_image × 1.33)` to `O(strip_height × 1.33)`.
    ///
    /// # When to use
    ///
    /// - **Image > 16 MP**: use this path. At 80 MP a `compute()` call
    ///   peaks at ~2.5 GB per pair; 16 rayon workers × 2.5 GB exceeds
    ///   reasonable RAM. This path's per-pair peak is ~125 MB with
    ///   `strip_inner = 256, strip_margin = 128`.
    /// - **Image ≤ 16 MP**: prefer [`compute`](Self::compute) — strip
    ///   aggregation has a small per-strip overhead and a minor
    ///   approximation in the strip-boundary blur context that the
    ///   full path avoids.
    ///
    /// # Default strip geometry
    ///
    /// `strip_inner = 256`, `strip_margin = 128` covers the default
    /// 4-scale pyramid with `blur_radius = 5`. For a 4× larger blur
    /// (`config.blur_radius = 20`), bump `strip_margin` proportionally.
    /// `strip_inner` must be at least `2 * blur_radius + 1` rows at
    /// every scale; with the default 4-scale pyramid that means
    /// `strip_inner >= 16 × (2 * blur_radius + 1) = 176` at default
    /// blur. The `256` default is safe for everything we ship.
    ///
    /// # Precision
    ///
    /// Byte-exact equivalent to the full-image [`compute`](Self::compute)
    /// path (within f64 machine epsilon, < 1e-13 rel). Strip-internal
    /// bands tile against the full-image plane layout so each band's
    /// V-blur running-sum init point and advance count match the
    /// full-image path exactly. Per-strip accumulators sum directly
    /// (raw `ScaleAccumulators::merge`) with no per-strip finalize
    /// precision loss.
    ///
    /// # Reference data: strip-per-strip (one-off pair) mode
    ///
    /// This entry builds the reference XYB pyramid per strip — each
    /// strip's `PrecomputedReference` covers JUST that strip's source
    /// rows + margin. Peak per-pair memory is `O(strip_h × width)`,
    /// at the cost of converting + downscaling the reference for each
    /// strip. **Best for one-off pairs** where the reference is not
    /// reused (each call sees a different source/distorted pair).
    ///
    /// For batch quantization loops where many distorted candidates
    /// are scored against the same reference, use
    /// [`Self::compute_with_ref_streaming_strips`] instead — that
    /// variant pre-builds the FULL reference pyramid once and reuses
    /// it across strips (and across calls).
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too
    /// small to support the strip geometry.
    pub fn compute_streaming_strips(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        strip_inner: usize,
        strip_margin: usize,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        check_within_max_pixels(source.width(), source.height(), self.max_pixels)?;
        let config = config_from_params(params, self.parallel);

        let (stats, mean_offset) = crate::streaming::compute_multiscale_stats_streaming_strips(
            source,
            distorted,
            &config,
            params.weights,
            strip_inner,
            strip_margin,
        );
        let mut result = combine_scores(&stats, params.weights, &config, mean_offset);
        apply_mlp_scoring(
            &mut result,
            params,
            source.width() as u32,
            source.height() as u32,
        )?;
        Ok(result.with_profile(self.profile))
    }

    /// Same as [`compute_streaming_strips`](Self::compute_streaming_strips)
    /// with sensible defaults: `strip_inner = 256`, `strip_margin = 128`.
    /// Use for 80 MP+ images.
    pub fn compute_streaming_strips_default(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        self.compute_streaming_strips(source, distorted, 256, 128)
    }

    /// Strip-aggregating variant that REUSES a full pre-built reference
    /// pyramid across strips — best for batch encoder loops where many
    /// distorted candidates are scored against the same source.
    ///
    /// Unlike [`Self::compute_streaming_strips`] (which builds a fresh
    /// per-strip [`crate::streaming::PrecomputedReference`] on each call,
    /// saving memory but redoing reference XYB conversion + pyramid
    /// downscale for every strip), this entry takes a caller-owned full
    /// `PrecomputedReference` and slices the appropriate Y-range out of
    /// it per strip. The distorted side is still processed per-strip so
    /// the distorted-side memory peak stays bounded.
    ///
    /// # Memory tradeoff
    ///
    /// - **`compute_streaming_strips`** (strip-per-strip ref): peak
    ///   `O(strip_h × width × 4 bytes × 3 channels × 2 sides × 1.33)`
    ///   per pair. ~125 MB at 80 MP with default strip geometry. Best
    ///   for memory-constrained one-off pairs.
    /// - **`compute_with_ref_streaming_strips`** (buffered ref): the
    ///   `PrecomputedReference` holds the FULL ref pyramid (~1.28 GB
    ///   at 80 MP). Per-call overhead is just the distorted side
    ///   (~960 MB at 80 MP), but amortized across N distorted
    ///   candidates the per-call cost drops to one distorted-side
    ///   pass. Best for encoder quantization loops.
    ///
    /// # Precision
    ///
    /// Byte-exact equivalent to [`Self::compute_with_ref`] (within f64
    /// machine epsilon).
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too
    /// small to support the strip geometry.
    pub fn compute_with_ref_streaming_strips(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
        strip_inner: usize,
        strip_margin: usize,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        if distorted.width() < 8 || distorted.height() < 8 {
            return Err(ZensimError::ImageTooSmall);
        }
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels)?;
        let config = config_from_params(params, self.parallel);

        let (stats, mean_offset) =
            crate::streaming::compute_multiscale_stats_streaming_strips_with_ref(
                precomputed,
                distorted,
                &config,
                params.weights,
                strip_inner,
                strip_margin,
            );
        let mut result = combine_scores(&stats, params.weights, &config, mean_offset);
        apply_mlp_scoring(
            &mut result,
            params,
            distorted.width() as u32,
            distorted.height() as u32,
        )?;
        Ok(result.with_profile(self.profile))
    }

    /// Same as [`Self::compute_with_ref_streaming_strips`] with default
    /// strip geometry (`strip_inner = 256`, `strip_margin = 128`).
    pub fn compute_with_ref_streaming_strips_default(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        self.compute_with_ref_streaming_strips(precomputed, distorted, 256, 128)
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
    /// Returns [`ZensimError::ImageTooSmall`] if dimensions < 8×8.
    pub fn compute_with_ref_into(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
        scratch: &mut crate::streaming::ZensimScratch,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        if distorted.width() < 8 || distorted.height() < 8 {
            return Err(ZensimError::ImageTooSmall);
        }
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels)?;
        let config = config_from_params(params, self.parallel);
        let (stats, mean_offset) =
            crate::streaming::compute_multiscale_stats_streaming_with_ref_borrowed(
                precomputed,
                distorted,
                &mut scratch.dst_planes,
                &config,
                params.weights,
            );
        let mut result = combine_scores(&stats, params.weights, &config, mean_offset);
        apply_mlp_scoring(
            &mut result,
            params,
            distorted.width() as u32,
            distorted.height() as u32,
        )?;
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
    /// Returns [`ZensimError::ImageTooSmall`] if dimensions < 8×8.
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
            self.parallel,
        ))
    }

    /// Like `compute`, but always computes all features regardless of
    /// zero weights (forces every channel active). For training/research.
    #[cfg(feature = "training")]
    pub fn compute_all_features(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        let mut config = config_from_params(params, self.parallel);
        config.compute_all_features = true;
        let result = compute_with_config_inner(source, distorted, &config, params.weights);
        Ok(result.with_profile(self.profile))
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
        let result = compute_with_config_inner(source, distorted, &config, params.weights);
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
    if source.width() < 8 || source.height() < 8 {
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
    if precomputed.width() != distorted.width() || precomputed.height() != distorted.height() {
        return Err(ZensimError::DimensionMismatch);
    }
    Ok(())
}

/// Reject `width × height` if it overflows `usize` on the current target,
/// exceeds the configured `max_pixels` cap, or if the padded plane size
/// (`simd_padded_width(width) × height`) would overflow during downstream
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
    // Also confirm the padded plane size fits. `simd_padded_width` rounds
    // up to a multiple of 16 (and may add another 16 above 512 wide), so
    // `padded_width * height` may overflow even when `width * height` did
    // not. Guarding here is cheaper than threading checked-arithmetic
    // through every internal allocation site.
    let padded = crate::blur::simd_padded_width(width);
    crate::blur::checked_padded_plane_len(padded, height)?;
    Ok(())
}

/// Check if source and distorted images have byte-identical pixel data
/// and matching color interpretation (format + primaries).
fn images_byte_identical(source: &impl ImageSource, distorted: &impl ImageSource) -> bool {
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

fn compute_with_config_inner(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
) -> ZensimResult {
    // Identical images must score exactly 100.0 — short-circuit before
    // floating-point arithmetic introduces sub-ULP noise in SSIM/edge features.
    if images_byte_identical(source, distorted) {
        // The feature width depends on which flags are enabled:
        //   - basic (228 = 4 × 3 × 19) is always present
        //   - extended adds 72 masked features (300 total)
        //   - compute_iw_features adds 72 IW-pool features (372 total)
        // On identical inputs every feature is zero (all error metrics
        // are 0). PreviewV0_5Compression (V_22-372feat) needs the full
        // 372-width vector; PreviewV0_5 / PreviewV0_5Balanced
        // (V_22-mix-LARGE+iwssim, 300-input) needs only the extended
        // 300-width vector. A missing block triggers InvalidDataLength
        // downstream when the bake's MLP runs.
        let fpc = match (config.extended_features, config.compute_iw_features) {
            (true, true) => FEATURES_PER_CHANNEL_EXTENDED + FEATURES_PER_CHANNEL_IW,
            (true, false) => FEATURES_PER_CHANNEL_EXTENDED,
            (false, _) => FEATURES_PER_CHANNEL_WITH_PEAKS,
        };
        let num_features = config.num_scales * 3 * fpc;
        return ZensimResult::new(
            100.0,
            0.0,
            vec![0.0; num_features],
            ZensimProfile::latest(),
            [0.0; 3],
        );
    }

    crate::streaming::compute_zensim_streaming(source, distorted, config, weights)
}

pub(crate) fn config_from_params(params: &ProfileParams, parallel: bool) -> ZensimConfig {
    ZensimConfig {
        blur_radius: params.blur_radius,
        blur_passes: params.blur_passes,
        blur_kernel: BlurKernel::Box {
            passes: params.blur_passes,
        },
        downscale_filter: DownscaleFilter::default(),
        // MLP-scored profiles need the full feature vector populated;
        // the linear path can skip features it doesn't use.
        compute_all_features: params.mlp_bytes.is_some(),
        extended_features: params.extended_features,
        extended_masking_strength: 4.0,
        compute_iw_features: params.compute_iw_features,
        iw_strength: 4.0,
        num_scales: params.num_scales,
        score_mapping_a: params.score_mapping_a,
        score_mapping_b: params.score_mapping_b,
        allow_multithreading: parallel,
    }
}

/// Logistic soft-clamp from raw score into the `[0, 100]` range.
///
/// `f(x) = 100 / (1 + exp(-(x - 50) / 20))`.
///
/// The scale parameter `20` gives a gentle squash centered at 50; at
/// the band centers the deviation from `x` is small but NON-ZERO:
///
/// | raw | f(raw) | Δ |
/// |---:|---:|---:|
/// | -1000 | 0.0     | -1000 |
/// |    0  | 7.586   | +7.6 |
/// |   25  | 28.110  | +3.1 |
/// |   50  | 50.0    | 0 |
/// |   75  | 71.890  | -3.1 |
/// |  100  | 92.414  | -7.6 |
/// | 1000  | 100.0   | -900 |
///
/// The interior shift is the cost of the formula: a profile that
/// enables `soft_clamp_score = true` ships a recalibration alongside
/// the rank-preservation fix. Downstream consumers that read a fixed
/// score-to-codec-parameter table will see a few-percentage-point
/// shift in the mapped codec settings.
///
/// **The function strictly preserves rank order globally** (monotone
/// increasing for all finite `x`), so SROCC / KROCC / PWRC are
/// invariant under the transform. PLCC and Z-RMSE shift modestly
/// because they're calibration-sensitive.
///
/// **Why the transform helps**: hard `raw.clamp(0, 100)` creates tie
/// blocks at exactly 0 or 100 when many out-of-range raw values
/// collapse to the boundary. The PreviewV0_4 multi-bake at α=0.4
/// raw-space sees this on heavy-distortion pairs (V_20 IS B3
/// specialist extrapolates past 100) and the resulting tie block
/// collapses SROCC to 0 on the affected band (observed on TID B0/B1
/// 2026-05-15). Soft-clamp distributes those raw values uniquely
/// into the high tail, so SROCC stays defined.
///
/// Cost: one `exp` call (~1 ns on modern hardware). Used when
/// `ProfileParams::soft_clamp_score` is `true`.
#[inline]
fn soft_clamp_score(raw: f64) -> f64 {
    100.0 / (1.0 + (-(raw - 50.0) / 20.0).exp())
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
    apply_mlp_scoring_with_codec(result, params, width, height, None)
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
    codec_hint: Option<&str>,
) -> Result<(), ZensimError> {
    let Some(loader) = params.mlp_bytes else {
        return Ok(());
    };
    // **Identity-image short-circuit guard.** When the byte-identical
    // short-circuit in `compute_with_config_inner` fires, it produces
    // `(score=100.0, raw_distance=0.0, features=[0.0; N])`. Running the
    // MLP forward pass on the all-zero feature vector then overwrites
    // those values with garbage — typically ~0 on V0_5Balanced (MSE bake
    // on z-scored zeros), ~2 on V0_5Compression / V0_5Ensemble — because
    // the trained MLP has no signal anchoring "zero feature vector →
    // score 100". The bake's biases dominate, producing an off-scale
    // output that `skip_score_mapping=true` returns verbatim.
    //
    // Detect the post-short-circuit state by checking the unique
    // signature: raw_distance is exactly 0.0 AND every feature is
    // exactly 0.0. Real (non-identical) images never produce this
    // signature because SSIM/edge/MSE on any pixel difference yields
    // non-zero values per-feature; `combine_scores` then derives a
    // non-zero `raw_distance` from the weighted feature vector.
    //
    // When detected, leave the result as `compute_with_config_inner`
    // set it (score=100.0, raw_distance=0.0). This preserves the
    // byte-identical invariant for every profile, regardless of which
    // bake is loaded or what `skip_score_mapping` does.
    if result.raw_distance() == 0.0 && result.features().iter().all(|&f| f == 0.0) {
        return Ok(());
    }
    {
        // **Ensemble routing path** (PreviewV0_5Ensemble) — when an
        // `ensemble_classifier_bytes` is present, run the classifier
        // first and route to either the primary (`mlp_bytes`,
        // balanced) or the alternative (`mlp_bytes_compression`)
        // based on the classifier's sign. The classifier output is a
        // pre-sigmoid logit; `logit > 0` ⇔ `sigmoid(logit) > 0.5` →
        // compression.
        //
        // Both target bakes MUST accept the same input feature
        // shape; the classifier runs over the same vector (its own
        // `n_inputs` selects the prefix). Documented in
        // `benchmarks/exp_ensemble_v05_eval_2026-05-18.md`.
        let raw = if let (Some(clf_loader), Some(cmp_loader)) = (
            params.ensemble_classifier_bytes,
            params.mlp_bytes_compression,
        ) {
            // Classifier dispatch — classifier bake is a router; it
            // does NOT receive the per-codec hint (its output is a
            // pre-sigmoid logit, not a score). The routed target bake
            // (primary or compression) receives the hint and may apply
            // the per-codec affine if its metadata is present.
            let logit = forward_one_bake(clf_loader(), result.features(), width, height)?;
            if logit > 0.0 {
                forward_one_bake_with_codec(
                    cmp_loader(),
                    result.features(),
                    width,
                    height,
                    codec_hint,
                )?
            } else {
                forward_one_bake_with_codec(
                    loader(),
                    result.features(),
                    width,
                    height,
                    codec_hint,
                )?
            }
        } else {
            let raw_primary = forward_one_bake_with_codec(
                loader(),
                result.features(),
                width,
                height,
                codec_hint,
            )?;
            // Optional secondary bake (D2 multi-output ensemble, e.g.
            // V_20 IS B3 specialist) — its forward path runs over the
            // SAME 228-feature vector but applies its own
            // `feature_transforms` metadata. Both bakes' raw outputs
            // are mixed linearly at `mlp_primary_mix` weight on the
            // primary. The codec hint is passed identically to both.
            if let Some(b3_loader) = params.mlp_bytes_b3 {
                let raw_b3 = forward_one_bake_with_codec(
                    b3_loader(),
                    result.features(),
                    width,
                    height,
                    codec_hint,
                )?;
                let a = params.mlp_primary_mix as f64;
                a * raw_primary + (1.0 - a) * raw_b3
            } else {
                raw_primary
            }
        };

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
        // PreviewV0_3 (V_18 ship single-bake) and earlier set `false`.
        //
        // EXP-CROSS-CODEC-V10 (2026-05-20): `extrapolate_score` overrides
        // both hard- and soft-clamp paths. The PCHIP spline's linear
        // extrapolation past its endpoint knots flows through, so
        // "pathological" codec output (worst codec at q=0, butter > 12)
        // maps to a negative score instead of collapsing to a tie at 0.
        // V10 profiles (BalancedV3 / CompressionV3 / TunerV4) set this.
        let score = if params.extrapolate_score {
            pre_bound
        } else if params.soft_clamp_score {
            soft_clamp_score(pre_bound)
        } else {
            pre_bound.clamp(0.0, 100.0)
        };
        result.set_mlp_score(raw, score);
    }
    Ok(())
}

/// Append `(log2(pixels), log2(min_dim), log2(max_dim), signed
/// log2(max/min))` to a feature vector. Mirrors the trainer-side
/// `append_size_axes` in `zensim-validate/src/main.rs` so a model
/// trained with `--mlp-size-axes` produces the same input layout
/// at runtime.
// ============================================================================
// Per-sample-α head runtime dispatch (V_24-per-sample-α bakes)
// ============================================================================
//
// Bakes trained with `train_mlp_per_sample_alpha_head_with_tv` in
// zensim-train-core attach a `zentrain.per_sample_alpha_head`
// metadata entry. The bake's final layer is a `n_hidden × n_hidden`
// identity matrix (passthrough), so the predictor's output IS the
// post-LeakyReLU hidden vector `h`. The runtime then mixes a rank
// head (`y_rank = h · rank_w + rank_b`) and a pool head (`y_pool =
// stats(h) · reducer_w + reducer_b`) via a per-sample gate
// `α = σ(h · w_α + b_α)`:
//
//     y_final = α · y_rank + (1 − α) · y_pool
//
// Payload layout (f32 little-endian):
//   [w_α[0..n_hidden]] [b_α] [rank_w[0..n_hidden]] [rank_b]
//   [reducer_w[0..4]] [reducer_b] [p_norm]
// Total size = (2·n_hidden + 8) × 4 bytes.
//
// Constants mirror `zensim-train-core::pool_head` (POOL_P_NORM,
// POOL_STD_FLOOR). Inlined here to keep zensim's dependency closure
// minimal (zensim runtime does not depend on zensim-train-core).

const PER_SAMPLE_ALPHA_HEAD_KEY: &str = "zentrain.per_sample_alpha_head";
const PER_SAMPLE_ALPHA_POOL_STD_FLOOR: f64 = 0.0026;

/// EXP-CROSS-CODEC-V4 (2026-05-19): tanh-pinned output head metadata
/// key. Payload is `[scale: f32 LE]` (4 bytes). When present, the
/// runtime wraps the per-sample-α head's raw output as
/// `y_score = 100 · σ(y_pre / scale)` — no post-hoc affine needed,
/// output is natively pinned to [0, 100].
const TANH_OUTPUT_HEAD_KEY: &str = "zentrain.tanh_output_head";

/// Parse the `zentrain.tanh_output_head` payload — a single f32 LE
/// (4 bytes) encoding the sigmoid pin scale. Returns `None` if the
/// payload length is wrong or the scale is non-positive / non-finite.
fn parse_tanh_output_head_scale(payload: &[u8]) -> Option<f64> {
    if payload.len() != 4 {
        return None;
    }
    let scale = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as f64;
    if scale.is_finite() && scale > 0.0 {
        Some(scale)
    } else {
        None
    }
}

/// Apply the tanh-pinned [0, 100] sigmoid wrap: `y_score = 100 · σ(y_pre / scale)`.
/// Matches `zensim_train_core::per_sample_alpha_head::bake_per_sample_alpha_head_v3_with_tanh`
/// at training time and bit-exact with the train-time pin (single-precision sigmoid
/// pin via clamp [−30, 30] in y_pre/scale, sigmoid in f64).
fn apply_tanh_output_pin(y_pre: f64, scale: f64) -> f64 {
    let xc = (y_pre / scale).clamp(-30.0, 30.0);
    let s = 1.0 / (1.0 + (-xc).exp());
    100.0 * s
}

/// EXP-CROSS-CODEC-V9 (2026-05-20): post-network PCHIP spline calibration
/// metadata key. Payload is `[n_knots: u32 LE, n_knots × (x: f32 LE, y: f32 LE)]`,
/// i.e. `4 + 8·n_knots` bytes. Knots must be sorted strictly increasing by x.
/// When present, the runtime applies a monotone cubic Hermite (PCHIP)
/// interpolation to the post-tanh-pin score: `y_calibrated = pchip(y_pinned)`.
const OUTPUT_CALIBRATION_SPLINE_KEY: &str = "zentrain.output_calibration_spline";

/// Parsed PCHIP spline payload: parallel `xs` and `ys` arrays plus the
/// precomputed monotone-Hermite slopes per knot (Fritsch–Carlson).
#[derive(Clone, Debug)]
struct OutputCalibrationSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Per-knot derivative (length == xs.len()).
    derivs: Vec<f64>,
}

/// Parse the `zentrain.output_calibration_spline` payload. Returns
/// `None` if the byte layout is wrong, knots are not strictly
/// increasing in x, or n_knots < 2.
fn parse_output_calibration_spline(payload: &[u8]) -> Option<OutputCalibrationSpline> {
    if payload.len() < 4 {
        return None;
    }
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if n < 2 {
        return None;
    }
    let expected = 4 + 8 * n;
    if payload.len() != expected {
        return None;
    }
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let off = 4 + i * 8;
        let x = f32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]) as f64;
        let y = f32::from_le_bytes([
            payload[off + 4],
            payload[off + 5],
            payload[off + 6],
            payload[off + 7],
        ]) as f64;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        xs.push(x);
        ys.push(y);
    }
    // Strictly increasing x.
    for i in 1..n {
        if !(xs[i] > xs[i - 1]) {
            return None;
        }
    }
    let derivs = pchip_compute_derivs(&xs, &ys);
    Some(OutputCalibrationSpline { xs, ys, derivs })
}

/// Compute Fritsch–Carlson monotone-preserving derivatives at each
/// knot. Standard PCHIP recipe: average of adjacent slopes harmonised
/// to prevent overshoot; endpoint derivatives use the one-sided slope
/// with a clamp to maintain monotonicity.
fn pchip_compute_derivs(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    debug_assert_eq!(ys.len(), n);
    debug_assert!(n >= 2);
    if n == 2 {
        let s = (ys[1] - ys[0]) / (xs[1] - xs[0]);
        return vec![s, s];
    }
    // Per-segment slopes h_k = (y_{k+1} - y_k) / (x_{k+1} - x_k).
    let mut h = Vec::with_capacity(n - 1);
    let mut s = Vec::with_capacity(n - 1);
    for k in 0..n - 1 {
        let hk = xs[k + 1] - xs[k];
        h.push(hk);
        s.push((ys[k + 1] - ys[k]) / hk);
    }
    let mut d = vec![0.0_f64; n];
    // Interior: weighted harmonic mean when adjacent slopes share sign,
    // else 0 (extremum).
    for k in 1..n - 1 {
        if s[k - 1] * s[k] <= 0.0 {
            d[k] = 0.0;
        } else {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            d[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k]);
        }
    }
    // Endpoints — three-point estimate, clamped to preserve mono.
    d[0] = pchip_endpoint(h[0], h[1], s[0], s[1]);
    d[n - 1] = pchip_endpoint(h[n - 2], h[n - 3], s[n - 2], s[n - 3]);
    d
}

fn pchip_endpoint(h0: f64, h1: f64, s0: f64, s1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * s0 - h0 * s1) / (h0 + h1);
    if d * s0 <= 0.0 {
        0.0
    } else if s0 * s1 <= 0.0 && d.abs() > 3.0 * s0.abs() {
        3.0 * s0
    } else {
        d
    }
}

/// Evaluate the PCHIP spline at `x`. Outside the knot range the
/// evaluation extrapolates linearly using the endpoint slope (so the
/// output stays monotone — crucial for the user-facing dial).
fn apply_output_calibration_spline(x: f64, spline: &OutputCalibrationSpline) -> f64 {
    let n = spline.xs.len();
    debug_assert!(n >= 2);
    let xs = spline.xs.as_slice();
    let ys = spline.ys.as_slice();
    let derivs = spline.derivs.as_slice();
    if !x.is_finite() {
        return x;
    }
    // Linear extrapolation outside the knot range.
    if x <= xs[0] {
        return ys[0] + derivs[0] * (x - xs[0]);
    }
    if x >= xs[n - 1] {
        return ys[n - 1] + derivs[n - 1] * (x - xs[n - 1]);
    }
    // Find the segment via binary search.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let h = xs[hi] - xs[lo];
    let t = (x - xs[lo]) / h;
    // Cubic Hermite basis on [0, 1].
    let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2);
    let h10 = t * (1.0 - t).powi(2);
    let h01 = t.powi(2) * (3.0 - 2.0 * t);
    let h11 = t.powi(2) * (t - 1.0);
    h00 * ys[lo] + h10 * h * derivs[lo] + h01 * ys[hi] + h11 * h * derivs[hi]
}

/// EXP-CROSS-CODEC-V11-E (2026-05-20): per-codec post-spline affine
/// calibration metadata key.
///
/// Payload layout (little-endian):
///   `[u32 n_codecs, n_codecs × (u32 name_len, name_len utf8 bytes, f32 alpha, f32 beta)]`
///
/// Applied AFTER the PCHIP spline (post all network forward + tanh-pin
/// + spline). For each entry, the runtime applies
/// `score_c = alpha_c + beta_c · spline(raw)` whenever the caller
/// supplies a matching codec name. Generic / unknown codec hint:
/// identity (alpha=0, beta=1).
///
/// The transform is monotone within codec (beta > 0 by construction),
/// so within-codec rank ordering is bit-exact preserved; only the
/// cross-codec systematic bias is adjusted toward consensus at JND
/// landmarks.
const PER_CODEC_CALIBRATION_KEY: &str = "zentrain.per_codec_calibration";

/// One per-codec affine entry parsed from the metadata payload.
#[derive(Clone, Debug)]
struct PerCodecAffineEntry {
    /// Lowercase ASCII codec name (matched case-insensitively).
    name: String,
    /// `score = alpha + beta · raw`. Beta is positive by construction.
    alpha: f32,
    beta: f32,
}

/// Parsed per-codec calibration payload.
#[derive(Clone, Debug)]
struct PerCodecCalibration {
    entries: Vec<PerCodecAffineEntry>,
}

/// Parse the `zentrain.per_codec_calibration` payload. Returns
/// `None` if the payload is malformed (truncated header, ragged
/// entry, non-utf8 name, beta ≤ 0, non-finite alpha/beta).
fn parse_per_codec_calibration(payload: &[u8]) -> Option<PerCodecCalibration> {
    if payload.len() < 4 {
        return None;
    }
    let n_codecs =
        u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let mut off = 4usize;
    let mut entries: Vec<PerCodecAffineEntry> = Vec::with_capacity(n_codecs);
    for _ in 0..n_codecs {
        if off + 4 > payload.len() {
            return None;
        }
        let name_len = u32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]) as usize;
        off += 4;
        if off + name_len + 8 > payload.len() {
            return None;
        }
        let name_bytes = &payload[off..off + name_len];
        let name = std::str::from_utf8(name_bytes).ok()?.to_ascii_lowercase();
        off += name_len;
        let alpha = f32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]);
        off += 4;
        let beta = f32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]);
        off += 4;
        if !alpha.is_finite() || !beta.is_finite() || beta <= 0.0 {
            return None;
        }
        entries.push(PerCodecAffineEntry { name, alpha, beta });
    }
    Some(PerCodecCalibration { entries })
}

/// Look up the per-codec affine for the given codec hint. Returns
/// `None` for unrecognized codecs (caller substitutes identity).
///
/// Match is case-insensitive and accepts common aliases (the table's
/// own name plus standard codec-family aliases). The metadata stores
/// canonical names (e.g. "jpeg", "webp", "avif", "jxl"); aliases like
/// "zenjpeg" / "mozjpeg" / "libjpeg" / "jpg" all map to "jpeg" etc.
fn lookup_per_codec_affine(
    cal: &PerCodecCalibration,
    codec_hint: &str,
) -> Option<(f32, f32)> {
    let lower = codec_hint.to_ascii_lowercase();
    let canon: &str = match lower.as_str() {
        "jpeg" | "jpg" | "zenjpeg" | "mozjpeg" | "libjpeg" => "jpeg",
        "webp" | "zenwebp" => "webp",
        "avif" | "zenavif" => "avif",
        "jxl" | "zenjxl" | "jpegxl" | "jpeg-xl" => "jxl",
        "png" | "zenpng" => "png",
        other => other,
    };
    for entry in &cal.entries {
        if entry.name == canon {
            return Some((entry.alpha, entry.beta));
        }
    }
    None
}

/// Parsed per-sample α head metadata payload.
struct PerSampleAlphaMeta {
    w_alpha: Vec<f32>,
    b_alpha: f32,
    rank_w: Vec<f32>,
    rank_b: f32,
    reducer_w: [f32; 4],
    reducer_b: f32,
    p_norm: f32,
}

/// Parse the `zentrain.per_sample_alpha_head` payload. Returns
/// `None` if the payload length doesn't match `(2·n_hidden + 8)·4`.
fn parse_per_sample_alpha_meta(payload: &[u8], n_hidden: usize) -> Option<PerSampleAlphaMeta> {
    let expected = (2 * n_hidden + 8) * 4;
    if payload.len() != expected {
        return None;
    }
    let mut floats: Vec<f32> = Vec::with_capacity(2 * n_hidden + 8);
    for chunk in payload.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let w_alpha: Vec<f32> = floats[..n_hidden].to_vec();
    let b_alpha = floats[n_hidden];
    let rank_w: Vec<f32> = floats[n_hidden + 1..2 * n_hidden + 1].to_vec();
    let rank_b = floats[2 * n_hidden + 1];
    let reducer_w = [
        floats[2 * n_hidden + 2],
        floats[2 * n_hidden + 3],
        floats[2 * n_hidden + 4],
        floats[2 * n_hidden + 5],
    ];
    let reducer_b = floats[2 * n_hidden + 6];
    let p_norm = floats[2 * n_hidden + 7];
    Some(PerSampleAlphaMeta {
        w_alpha,
        b_alpha,
        rank_w,
        rank_b,
        reducer_w,
        reducer_b,
        p_norm,
    })
}

/// Apply the per-sample-α runtime to a hidden vector `h`. Returns
/// the final mixed score `y`. Bit-exact match with
/// `zensim_train_core::per_sample_alpha_head::apply_per_sample_alpha_head_runtime`
/// (asserted by the canonical regression test in
/// `tests/per_sample_alpha_runtime.rs`).
fn apply_per_sample_alpha_runtime(h: &[f32], meta: &PerSampleAlphaMeta) -> f64 {
    debug_assert_eq!(meta.rank_w.len(), h.len());
    debug_assert_eq!(meta.w_alpha.len(), h.len());
    let n = h.len();
    debug_assert!(n > 0);

    let mut y_rank = meta.rank_b as f64;
    let mut alpha_logit = meta.b_alpha as f64;
    let mut sum = 0.0_f64;
    let mut max_v = f64::NEG_INFINITY;
    let mut sum_p = 0.0_f64;
    let p = meta.p_norm as f64;
    for (j, &hj) in h.iter().enumerate() {
        let hjf = hj as f64;
        y_rank += hjf * meta.rank_w[j] as f64;
        alpha_logit += hjf * meta.w_alpha[j] as f64;
        sum += hjf;
        if hjf > max_v {
            max_v = hjf;
        }
        sum_p += hjf.abs().powf(p);
    }
    let nf = n as f64;
    let mu = sum / nf;
    let mut var = 0.0_f64;
    for &hj in h.iter() {
        let d = hj as f64 - mu;
        var += d * d;
    }
    let sigma = (var / nf).sqrt().max(PER_SAMPLE_ALPHA_POOL_STD_FLOOR);
    let p_norm_stat = (sum_p / nf).powf(1.0 / p);

    let y_pool = mu * meta.reducer_w[0] as f64
        + sigma * meta.reducer_w[1] as f64
        + max_v * meta.reducer_w[2] as f64
        + p_norm_stat * meta.reducer_w[3] as f64
        + meta.reducer_b as f64;

    // sigmoid with clamp (matches trainer's `sigmoid` helper).
    let alpha = {
        let xc = alpha_logit.clamp(-20.0, 20.0);
        1.0 / (1.0 + (-xc).exp())
    };
    alpha * y_rank + (1.0 - alpha) * y_pool
}

// ============================================================================
// Hybrid-head runtime dispatch (V_24-hybrid bakes)
// ============================================================================
//
// Bakes trained with `train_mlp_hybrid_head_with_tv` in
// zensim-train-core attach a `zentrain.hybrid_head` metadata
// entry. Like the per-sample-α head, the bake's final layer is an
// `n_hidden × n_hidden` identity matrix (passthrough), so the
// predictor's output IS the post-LeakyReLU hidden vector `h`. The
// runtime then mixes a rank head (`y_rank = h · rank_w + rank_b`)
// and a pool head (`y_pool = stats(h) · reducer_w + reducer_b`)
// via a single LEARNED SCALAR gate `α = σ(α_logit)` (different
// from the per-sample-α head, where α is computed per-sample from
// the hidden vector):
//
//     y_final = α · y_rank + (1 − α) · y_pool
//
// Payload layout (f32 little-endian):
//   [rank_w[0..n_hidden]] [rank_b] [α_logit]
//   [reducer_w[0..4]] [reducer_b] [p_norm]
// Total size = (n_hidden + 8) × 4 bytes.
//
// Constants mirror `zensim-train-core::pool_head` (POOL_P_NORM,
// POOL_STD_FLOOR). Inlined to keep zensim's dependency closure
// minimal.

const HYBRID_HEAD_KEY: &str = "zentrain.hybrid_head";
const HYBRID_HEAD_POOL_STD_FLOOR: f64 = 0.0026;

/// Parsed hybrid-head metadata payload.
struct HybridHeadMeta {
    rank_w: Vec<f32>,
    rank_b: f32,
    alpha_logit: f32,
    reducer_w: [f32; 4],
    reducer_b: f32,
    p_norm: f32,
}

/// Parse the `zentrain.hybrid_head` payload. Returns `None` if the
/// payload length doesn't match `(n_hidden + 8) · 4`.
fn parse_hybrid_head_meta(payload: &[u8], n_hidden: usize) -> Option<HybridHeadMeta> {
    let expected = (n_hidden + 8) * 4;
    if payload.len() != expected {
        return None;
    }
    let mut floats: Vec<f32> = Vec::with_capacity(n_hidden + 8);
    for chunk in payload.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let rank_w: Vec<f32> = floats[..n_hidden].to_vec();
    let rank_b = floats[n_hidden];
    let alpha_logit = floats[n_hidden + 1];
    let reducer_w = [
        floats[n_hidden + 2],
        floats[n_hidden + 3],
        floats[n_hidden + 4],
        floats[n_hidden + 5],
    ];
    let reducer_b = floats[n_hidden + 6];
    let p_norm = floats[n_hidden + 7];
    Some(HybridHeadMeta {
        rank_w,
        rank_b,
        alpha_logit,
        reducer_w,
        reducer_b,
        p_norm,
    })
}

/// Apply the hybrid-head runtime to a hidden vector `h`. Returns the
/// final mixed score `y`. Bit-exact match with
/// `zensim_train_core::hybrid_head::apply_hybrid_head_runtime`
/// (asserted by the canonical regression test in
/// `tests/hybrid_head_runtime.rs`).
fn apply_hybrid_head_runtime(h: &[f32], meta: &HybridHeadMeta) -> f64 {
    debug_assert_eq!(meta.rank_w.len(), h.len());
    let n = h.len();
    debug_assert!(n > 0);

    let mut y_rank = meta.rank_b as f64;
    let mut sum = 0.0_f64;
    let mut max_v = f64::NEG_INFINITY;
    let mut sum_p = 0.0_f64;
    let p = meta.p_norm as f64;
    for (j, &hj) in h.iter().enumerate() {
        let hjf = hj as f64;
        y_rank += hjf * meta.rank_w[j] as f64;
        sum += hjf;
        if hjf > max_v {
            max_v = hjf;
        }
        sum_p += hjf.abs().powf(p);
    }
    let nf = n as f64;
    let mu = sum / nf;
    let mut var = 0.0_f64;
    for &hj in h.iter() {
        let d = hj as f64 - mu;
        var += d * d;
    }
    let sigma = (var / nf).sqrt().max(HYBRID_HEAD_POOL_STD_FLOOR);
    let p_norm_stat = (sum_p / nf).powf(1.0 / p);

    let y_pool = mu * meta.reducer_w[0] as f64
        + sigma * meta.reducer_w[1] as f64
        + max_v * meta.reducer_w[2] as f64
        + p_norm_stat * meta.reducer_w[3] as f64
        + meta.reducer_b as f64;

    // sigmoid with clamp (matches trainer's `sigmoid` helper).
    let alpha = {
        let xc = (meta.alpha_logit as f64).clamp(-20.0, 20.0);
        1.0 / (1.0 + (-xc).exp())
    };
    alpha * y_rank + (1.0 - alpha) * y_pool
}

/// Forward one bake over `features` and return the raw output scalar.
/// Handles the three accepted input-width shapes (n_inputs == features,
/// n_inputs == features+4 with size-axes, n_inputs < features prefix)
/// and dispatches to `predict_transformed` when the bake declares
/// non-trivial `feature_transforms` metadata (V_20+ input-shaping
/// bakes); otherwise uses the plain `predict` path with zero overhead.
///
/// Bakes carrying `zentrain.per_sample_alpha_head` metadata
/// (V_24-per-sample-α architecture) take a separate dispatch path:
/// the forward returns `h` (the post-LeakyReLU hidden vector, since
/// the bake's layer 2 is an identity passthrough) and the runtime
/// mixes a rank-head + pool-head pair via a per-sample sigmoid gate.
/// See `apply_per_sample_alpha_runtime` above for the formula and
/// payload layout.
///
/// Bakes carrying `zentrain.hybrid_head` metadata (V_24-hybrid
/// architecture) take a similar dispatch path, but the α gate is a
/// single learned SCALAR (`α = σ(α_logit)`) shared across all
/// samples rather than computed per-sample. See
/// `apply_hybrid_head_runtime` above for the formula and payload
/// layout.
fn forward_one_bake(
    bytes: &[u8],
    features: &[f64],
    width: u32,
    height: u32,
) -> Result<f64, ZensimError> {
    forward_one_bake_with_codec(bytes, features, width, height, None)
}

/// Same as [`forward_one_bake`] but accepts an optional codec hint
/// that drives the per-codec post-spline affine calibration
/// (EXP-CROSS-CODEC-V11-E). When `codec_hint` is `Some` and the
/// bake carries `zentrain.per_codec_calibration` metadata, the
/// matched codec's `(alpha, beta)` affine is applied AFTER the
/// PCHIP spline; otherwise the score is returned identical to the
/// hint-less path.
fn forward_one_bake_with_codec(
    bytes: &[u8],
    features: &[f64],
    width: u32,
    height: u32,
    codec_hint: Option<&str>,
) -> Result<f64, ZensimError> {
    let model = crate::mlp::Model::from_bytes(bytes).map_err(|_| ZensimError::InvalidDataLength)?;
    let n_inputs = model.n_inputs();
    let mut predictor = crate::mlp::Predictor::new(&model);
    let needs_transforms = model.has_nontrivial_feature_transforms();

    // Per-sample-α and hybrid-head metadata + parsed payload. When
    // present, the forward output is treated as the hidden vector h
    // (length = `n_hidden`); otherwise the legacy `out[0]` path is
    // taken. Per-sample-α takes precedence over hybrid-head if both
    // are somehow present (the two heads are alternative
    // architectures; a bake should only carry one).
    //
    // The metadata blob is owned by the model bytes — the returned
    // `MetadataEntry` borrows from `&model`. We copy the value bytes
    // into an owned Vec so the lifetime is independent of `predictor`
    // (which also borrows `&model`).
    let per_sample_alpha: Option<PerSampleAlphaMeta> = {
        let metadata = model.metadata();
        metadata
            .get(PER_SAMPLE_ALPHA_HEAD_KEY)
            .and_then(|entry| parse_per_sample_alpha_meta(entry.value, model.n_outputs()))
    };
    let hybrid_head: Option<HybridHeadMeta> = if per_sample_alpha.is_some() {
        None
    } else {
        let metadata = model.metadata();
        metadata
            .get(HYBRID_HEAD_KEY)
            .and_then(|entry| parse_hybrid_head_meta(entry.value, model.n_outputs()))
    };
    // EXP-CROSS-CODEC-V4 (2026-05-19): tanh-pinned [0, 100] output
    // head. Applies AFTER per-sample-α / hybrid-head mixing. Only
    // active when the bake carries `zentrain.tanh_output_head` metadata.
    let tanh_pin_scale: Option<f64> = {
        let metadata = model.metadata();
        metadata
            .get(TANH_OUTPUT_HEAD_KEY)
            .and_then(|entry| parse_tanh_output_head_scale(entry.value))
    };
    // EXP-CROSS-CODEC-V9 (2026-05-20): post-network monotone PCHIP
    // spline calibration. Applies AFTER tanh-pin. Only active when
    // the bake carries `zentrain.output_calibration_spline` metadata.
    let output_spline: Option<OutputCalibrationSpline> = {
        let metadata = model.metadata();
        metadata
            .get(OUTPUT_CALIBRATION_SPLINE_KEY)
            .and_then(|entry| parse_output_calibration_spline(entry.value))
    };
    // EXP-CROSS-CODEC-V11-E (2026-05-20): per-codec post-spline affine.
    // Applies AFTER the PCHIP spline. Only active when the bake carries
    // `zentrain.per_codec_calibration` metadata AND the caller supplied
    // a codec hint that maps to a registered codec entry.
    let per_codec_affine: Option<(f32, f32)> = {
        let cal = {
            let metadata = model.metadata();
            metadata
                .get(PER_CODEC_CALIBRATION_KEY)
                .and_then(|entry| parse_per_codec_calibration(entry.value))
        };
        match (cal, codec_hint) {
            (Some(cal), Some(hint)) => lookup_per_codec_affine(&cal, hint),
            _ => None,
        }
    };

    let dispatch = |p: &mut crate::mlp::Predictor<'_>, x: &[f32]| -> Result<f64, ZensimError> {
        let out = if needs_transforms {
            p.predict_transformed(x)
                .map_err(|_| ZensimError::InvalidDataLength)?
        } else {
            p.predict(x).map_err(|_| ZensimError::InvalidDataLength)?
        };
        let y_pre = if let Some(meta) = &per_sample_alpha {
            // `out` is the hidden vector h (n_hidden floats). Apply
            // the per-sample-α runtime formula.
            if out.len() != meta.rank_w.len() {
                // Shape mismatch between metadata's declared n_hidden
                // and bake's n_outputs — bake is malformed.
                return Err(ZensimError::InvalidDataLength);
            }
            apply_per_sample_alpha_runtime(out, meta)
        } else if let Some(meta) = &hybrid_head {
            // `out` is the hidden vector h. Apply the scalar-α
            // hybrid runtime formula.
            if out.len() != meta.rank_w.len() {
                return Err(ZensimError::InvalidDataLength);
            }
            apply_hybrid_head_runtime(out, meta)
        } else {
            out[0] as f64
        };
        let y_after_pin = if let Some(scale) = tanh_pin_scale {
            apply_tanh_output_pin(y_pre, scale)
        } else {
            y_pre
        };
        let y_after_spline = if let Some(spline) = &output_spline {
            apply_output_calibration_spline(y_after_pin, spline)
        } else {
            y_after_pin
        };
        // EXP-CROSS-CODEC-V11-E: per-codec post-spline affine. When
        // the metadata is present AND the caller supplied a known
        // codec hint, pull the per-codec output toward consensus.
        // Otherwise pass through unchanged. The affine is monotone
        // (beta > 0 by parse) so within-codec rank ordering is
        // preserved bit-exact.
        Ok(if let Some((alpha, beta)) = per_codec_affine {
            (alpha as f64) + (beta as f64) * y_after_spline
        } else {
            y_after_spline
        })
    };
    if n_inputs == features.len() {
        let f32_features: Vec<f32> = features.iter().map(|&v| v as f32).collect();
        dispatch(&mut predictor, &f32_features)
    } else if n_inputs == features.len() + 4 {
        let mut augmented = features.to_vec();
        append_mlp_size_axes(&mut augmented, width, height);
        let f32_features: Vec<f32> = augmented.iter().map(|&v| v as f32).collect();
        dispatch(&mut predictor, &f32_features)
    } else if n_inputs < features.len() {
        let f32_features: Vec<f32> = features[..n_inputs].iter().map(|&v| v as f32).collect();
        dispatch(&mut predictor, &f32_features)
    } else {
        Err(ZensimError::InvalidDataLength)
    }
}

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
    let log2_pixels = pixels.log2();
    let log2_min = min_dim.log2();
    let log2_max = max_dim.log2();
    let log_aspect_signed = (max_dim / min_dim).log2() * if w >= h { 1.0 } else { -1.0 };
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

/// Extended features per channel per scale: 25 features (19 with peaks + 6 masked).
///
/// ```text
///  Index  Name               Pooling  Source
///  ─────  ─────────────────  ───────  ──────────────────
///  0–12   (same as basic 13)
///  13–18  (same as peak features: max/l8)
///  19     masked_ssim_mean   mean     SSIM × flatness mask
///  20     masked_ssim_4th    L4       SSIM × flatness mask
///  21     masked_ssim_2nd    L2       SSIM × flatness mask
///  22     masked_art_4th     L4       edge artifact × flatness mask
///  23     masked_det_4th     L4       edge detail_lost × flatness mask
///  24     masked_mse         mean     (src-dst)² × flatness mask
/// ```
///
/// Total features = `num_scales × 3 channels × 25` = 300 at 4 scales.
pub const FEATURES_PER_CHANNEL_EXTENDED: usize = 25;

/// Information-content-weighted (IW) features per channel per scale.
/// Wang & Li 2011 — IW-SSIM. Same 6 feature slots as the `masked_*`
/// block but with the weight polarity flipped: texture-rich regions
/// get MORE weight.
///
/// ```text
///  Index  Name               Pooling  Source
///  ─────  ─────────────────  ───────  ──────────────────
///  0      iw_ssim_mean       mean     SSIM × iw_weight
///  1      iw_ssim_4th        L4       SSIM × iw_weight
///  2      iw_ssim_2nd        L2       SSIM × iw_weight
///  3      iw_art_4th         L4       edge artifact × iw_weight
///  4      iw_det_4th         L4       edge detail_lost × iw_weight
///  5      iw_mse             mean     (src-dst)² × iw_weight
/// ```
///
/// Enabled via `ZensimConfig::compute_iw_features`. At `num_scales = 4`
/// and 3 channels, adds 72 features → total profile becomes 300
/// (basic + peaks + IW).
pub const FEATURES_PER_CHANNEL_IW: usize = 6;

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
        let masked_total = n_scales * 3 * 6;

        let (scored_total, peaks_total) = if features.len() == basic_total {
            // Legacy basic-only layout (backward compat)
            (basic_total, 0)
        } else if features.len() == basic_total + peaks_total
            || features.len() == basic_total + peaks_total + masked_total
        {
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

    /// Whether masked features are present.
    pub fn has_masked(&self) -> bool {
        self.features.len() > self.scored_total + self.peaks_total
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

    // --- Masked features (require extended_features) ---

    fn masked_idx(&self, scale: usize, ch: usize, offset: usize) -> Option<usize> {
        if !self.has_masked() {
            return None;
        }
        Some(self.scored_total + self.peaks_total + scale * 3 * 6 + ch * 6 + offset)
    }

    /// Masked SSIM error, mean pooling.
    pub fn masked_ssim_mean(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 0).map(|i| self.features[i])
    }
    /// Masked SSIM error, L4 norm.
    pub fn masked_ssim_4th(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 1).map(|i| self.features[i])
    }
    /// Masked SSIM error, L2 norm.
    pub fn masked_ssim_2nd(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 2).map(|i| self.features[i])
    }
    /// Masked edge artifact, L4 norm.
    pub fn masked_art_4th(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 3).map(|i| self.features[i])
    }
    /// Masked edge detail lost, L4 norm.
    pub fn masked_det_4th(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 4).map(|i| self.features[i])
    }
    /// Masked MSE.
    pub fn masked_mse(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 5).map(|i| self.features[i])
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

    /// Get the masked features slice, if present.
    pub fn masked_features(&self) -> Option<&[f64]> {
        if !self.has_masked() {
            None
        } else {
            Some(&self.features[self.scored_total + self.peaks_total..])
        }
    }
}

/// Compute zensim with custom configuration (training API).
///
/// Uses the v0.2 weights (latest general-purpose profile).
#[cfg(any(feature = "training", test))]
pub fn compute_zensim_with_config(
    source: &[[u8; 3]],
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
    config: ZensimConfig,
) -> Result<ZensimResult, ZensimError> {
    // Validation
    if width < 8 || height < 8 {
        return Err(ZensimError::ImageTooSmall);
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(ZensimError::ImageTooLarge)?;
    // Also reject overflow on the padded plane size (simd_padded_width(width) * height).
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
        let fpc = match (config.extended_features, config.compute_iw_features) {
            (true, true) => FEATURES_PER_CHANNEL_EXTENDED + FEATURES_PER_CHANNEL_IW,
            (true, false) => FEATURES_PER_CHANNEL_EXTENDED,
            (false, _) => FEATURES_PER_CHANNEL_WITH_PEAKS,
        };
        let num_features = config.num_scales * 3 * fpc;
        return Ok(ZensimResult::new(
            100.0,
            0.0,
            vec![0.0; num_features],
            ZensimProfile::latest(),
            [0.0; 3],
        ));
    }

    let src_img = crate::source::RgbSlice::new(source, width, height);
    let dst_img = crate::source::RgbSlice::new(distorted, width, height);

    let result = crate::streaming::compute_zensim_streaming(&src_img, &dst_img, &config, WEIGHTS);
    Ok(result)
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
#[cfg_attr(not(feature = "training"), allow(dead_code))]
pub const FEATURES_PER_SCALE: usize = FEATURES_PER_CHANNEL_WITH_PEAKS * 3;

/// Default scoring weights — references the latest profile weights.
///
/// Layout: 4 scales × 3 channels (X,Y,B) × 13 basic features, then
///         4 scales × 3 channels × 6 peak features = 228 total.
#[cfg(any(feature = "training", test))]
pub const WEIGHTS: &[f64; 228] = &crate::profile::WEIGHTS_PREVIEW_V0_2;

pub(crate) fn combine_scores(
    scale_stats: &[ScaleStats],
    weights: &[f64],
    config: &ZensimConfig,
    mean_offset: [f64; 3],
) -> ZensimResult {
    let extended = config.extended_features;
    let iw = config.compute_iw_features;

    // Feature vector layout (in order they appear in the Vec):
    //   [0..N_basic)        — 13/ch × 3ch × n_scales (basic features)
    //   [N_basic..N_peaks)  — 6/ch × 3ch × n_scales peak features (always included)
    //   [N_peaks..N_masked) — 6/ch × 3ch × n_scales masked features (if extended)
    //   [N_masked..N_iw)    — 6/ch × 3ch × n_scales IW features (if compute_iw_features)
    //
    // Both basic and peak features are scored: features[0..WEIGHTS.len()]
    // produces the dot product used for the final score. Masked + IW are
    // training-only feature additions consumed by MLP-scored profiles.
    let n_scales = scale_stats.len();
    let basic_per_ch = FEATURES_PER_CHANNEL_BASIC; // 13
    let basic_total = n_scales * basic_per_ch * 3;
    let peak_total = n_scales * 6 * 3;
    let masked_total = if extended { n_scales * 6 * 3 } else { 0 };
    let iw_total = if iw {
        n_scales * FEATURES_PER_CHANNEL_IW * 3
    } else {
        0
    };
    let total = basic_total + peak_total + masked_total + iw_total;

    let mut features = Vec::with_capacity(total);
    let mut raw_distance = 0.0f64;

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

    // Pass 3: masked features (6/ch — expensive, training only)
    if extended {
        for ss in scale_stats.iter() {
            for c in 0..3 {
                features.push(ss.masked_ssim[c * 3].abs());
                features.push(ss.masked_ssim[c * 3 + 1].abs());
                features.push(ss.masked_ssim[c * 3 + 2].abs());
                features.push(ss.masked_art_4th[c].abs());
                features.push(ss.masked_det_4th[c].abs());
                features.push(ss.masked_mse[c]);
            }
        }
    }

    // Pass 4: IW (information-content-weighted) features (6/ch —
    // texture-emphasising counterpart to masked). Wang & Li 2011.
    if iw {
        for ss in scale_stats.iter() {
            for c in 0..3 {
                features.push(ss.iw_ssim[c * 3].abs());
                features.push(ss.iw_ssim[c * 3 + 1].abs());
                features.push(ss.iw_ssim[c * 3 + 2].abs());
                features.push(ss.iw_art_4th[c].abs());
                features.push(ss.iw_det_4th[c].abs());
                features.push(ss.iw_mse[c]);
            }
        }
    }

    // Apply weights — basic + peak features are scored
    let scored_total = basic_total + peak_total;
    let n_score = scored_total.min(weights.len());
    for (i, &feat) in features[..n_score].iter().enumerate() {
        raw_distance += feat * weights[i];
    }

    // Normalize by number of scales
    raw_distance /= scale_stats.len().max(1) as f64;

    let score =
        distance_to_score_mapped(raw_distance, config.score_mapping_a, config.score_mapping_b);

    ZensimResult::new(
        score,
        raw_distance,
        features,
        ZensimProfile::PreviewV0_1,
        mean_offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `soft_clamp_score` matches its documented boundary values and
    /// stays monotone strictly increasing — the rank-preservation
    /// property that the V_20+ multi-bake regime relies on.
    #[test]
    fn soft_clamp_score_boundary_values() {
        // Documented anchor values in the function docstring.
        assert!((soft_clamp_score(-1000.0) - 0.0).abs() < 1e-9);
        assert!((soft_clamp_score(0.0) - 7.585818002124355).abs() < 1e-9);
        assert!((soft_clamp_score(50.0) - 50.0).abs() < 1e-9);
        assert!((soft_clamp_score(100.0) - 92.41418199787566).abs() < 1e-9);
        assert!((soft_clamp_score(1000.0) - 100.0).abs() < 1e-9);
    }

    /// Soft-clamp is strictly monotone — adjacent raw scores keep
    /// their order, so SROCC (rank-based) is preserved through the
    /// transform. The hard-clamp pathology (tie blocks at 0/100)
    /// disappears: out-of-range raw values stay distinct.
    #[test]
    fn soft_clamp_score_monotone_across_full_range() {
        let mut prev = f64::NEG_INFINITY;
        // Sweep a wide grid: deep below 0, through the saturation band,
        // and far above 100. 1e6 raw is plausible for accidentally
        // distance-shaped bakes wired in by mistake — must still rank.
        for &raw in &[
            -1e6, -1000.0, -100.0, -50.0, -10.0, 0.0, 1.0, 25.0, 49.999, 50.0, 50.001, 75.0, 90.0,
            99.999, 100.0, 100.001, 110.0, 200.0, 1000.0, 1e6,
        ] {
            let s = soft_clamp_score(raw);
            assert!(
                s > prev || (s.is_finite() && prev.is_finite() && (s - prev).abs() < 1e-12),
                "soft_clamp not monotone: f({raw}) = {s}, prev = {prev}"
            );
            assert!(
                (0.0..=100.0).contains(&s),
                "soft_clamp out of [0, 100]: f({raw}) = {s}"
            );
            prev = s;
        }
    }

    /// The interior shift is non-zero but bounded. Inside `[25, 75]`
    /// the soft-clamp shift stays under ±3.2 units; at the boundary
    /// it grows to ±7.6 (raw = 0 → 7.586, raw = 100 → 92.414).
    /// Profiles that enable `soft_clamp_score` accept this
    /// recalibration as the price of rank preservation at the tails.
    #[test]
    fn soft_clamp_score_interior_shift_bounded() {
        for tenths in 250..=750 {
            let raw = tenths as f64 / 10.0;
            let s = soft_clamp_score(raw);
            assert!(
                (s - raw).abs() <= 3.2,
                "soft_clamp shift >3.2 in [25,75]: f({raw}) = {s} (Δ={})",
                (s - raw).abs()
            );
        }
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

    /// V0_20a IW pool integration: `compute_iw_features=true` must
    /// emit 72 additional features (228 → 300 at 4 scales × 3 ch × 6),
    /// AND those features must differ from the masked block when both
    /// are enabled (weight polarity is opposite).
    #[test]
    fn compute_iw_features_emits_300_when_enabled() {
        let w = 128;
        let h = 128;
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                // Add a textured patch so the IW weights actually vary.
                let noise = if (x % 16) < 8 && (y % 16) < 8 { 32 } else { 0 };
                src[y * w + x] = [r.saturating_add(noise), g, 128];
                dst[y * w + x] = [r.saturating_add(noise).saturating_add(5), g, 125];
            }
        }

        // Off → 228 features (basic + peaks, masked OFF, iw OFF)
        let off = compute_zensim_with_config(
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
        assert_eq!(off.features.len(), 228);

        // IW only → 228 + 72 = 300 features
        let iw_only = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                compute_all_features: true,
                compute_iw_features: true,
                iw_strength: 4.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(iw_only.features.len(), 300);

        // First 228 must match exactly (IW must not perturb basic/peaks).
        for i in 0..228 {
            let d = (off.features[i] - iw_only.features[i]).abs();
            assert!(
                d < 1e-9,
                "IW must not affect basic/peak features; index {} differs by {}",
                i,
                d,
            );
        }

        // IW block must produce non-zero output (we have textured patches).
        let iw_block_nonzero = iw_only.features[228..]
            .iter()
            .filter(|f| f.abs() > 1e-12)
            .count();
        assert!(iw_block_nonzero > 0, "IW block was all zeros");

        // When both extended_features and compute_iw_features are on,
        // total = 228 + 72 (masked) + 72 (IW) = 372. Masked and IW
        // pools have OPPOSITE weight polarity, so the two 72-feature
        // blocks must differ.
        let both = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                compute_all_features: true,
                extended_features: true,
                compute_iw_features: true,
                iw_strength: 4.0,
                extended_masking_strength: 4.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(both.features.len(), 372);

        let mut masked_iw_diff_count = 0usize;
        for i in 0..72 {
            let masked = both.features[228 + i];
            let iw = both.features[228 + 72 + i];
            if (masked - iw).abs() > 1e-9 {
                masked_iw_diff_count += 1;
            }
        }
        assert!(
            masked_iw_diff_count >= 60,
            "masked vs IW features should differ; only {}/72 differed",
            masked_iw_diff_count,
        );
    }

    /// `Zensim::compute_extended_features` returns 300 features at the
    /// default 4-scale, 3-channel layout, and produces a score that matches
    /// `Zensim::compute` to within a small numerical tolerance (the extra
    /// 72 features have zero weight, so the weighted score is unchanged).
    #[test]
    fn compute_extended_features_returns_300() {
        use crate::source::RgbSlice;
        let w = 64;
        let h = 64;
        let (src, dst) = make_gradient_pair(w, h);
        let src_img = RgbSlice::new(&src, w, h);
        let dst_img = RgbSlice::new(&dst, w, h);

        // Linear-profile-specific test: the score-equality assertion
        // only holds for profiles whose scoring is linear-weighted on
        // the 228 features (the extended 300 features just have
        // weight=0 in those slots). MLP profiles (PreviewV0_3 +) score
        // via a forward pass — extended features change the input
        // shape and break the equality. Pin to V0_2.
        let z = crate::Zensim::new(crate::ZensimProfile::PreviewV0_2);
        let standard = z.compute(&src_img, &dst_img).unwrap();
        let extended = z.compute_extended_features(&src_img, &dst_img).unwrap();

        assert_eq!(standard.features().len(), 228);
        assert_eq!(extended.features().len(), 300);
        // Score should match: extra features have zero weight in the
        // weighted-distance calculation.
        assert!(
            (standard.score() - extended.score()).abs() < 0.01,
            "standard score {} vs extended score {}",
            standard.score(),
            extended.score()
        );
        // Sanity: features 0..228 should agree wherever the standard path
        // populated them. The standard path skips channels whose weights are
        // all-zero (leaving 0.0 in those slots); the extended path forces
        // every channel/feature to be computed. So we only enforce equality
        // on slots the standard path actually filled.
        for i in 0..228 {
            let a = standard.features()[i];
            let b = extended.features()[i];
            if a != 0.0 {
                assert!(
                    (a - b).abs() < 1e-6,
                    "feature {i}: standard {a} vs extended {b}"
                );
            }
        }
        // The trailing 72 masked features should be non-negative.
        for (i, &f) in extended.features()[228..].iter().enumerate() {
            assert!(f >= 0.0, "extended feature {} is negative: {}", 228 + i, f);
        }
    }

    /// Helper: create a gradient test image pair.
    fn make_gradient_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = 128;
                src[y * w + x] = [r, g, b];
                dst[y * w + x] = [
                    r.saturating_add(10),
                    g.saturating_sub(5),
                    b.saturating_add(3),
                ];
            }
        }
        (src, dst)
    }

    /// Numerical equivalence: when both `extended_features` and
    /// `compute_iw_features` are enabled, the fused 2-mask SIMD path
    /// must produce the same masked block as running with
    /// `extended_features` alone, and the same IW block as running
    /// with `compute_iw_features` alone. This is the critical
    /// invariant for the 2026-05-15 perf optimization landing fused
    /// 2-mask SSIM/edge/MSE kernels.
    #[test]
    fn fused_2mask_matches_separate_paths() {
        let (w, h) = (128, 128);
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        // Build a textured image so weights actually vary.
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = (((x + y) * 255) / (2 * w)) as u8;
                let noise = if (x % 16) < 8 && (y % 16) < 8 { 32 } else { 0 };
                src[y * w + x] = [r.saturating_add(noise), g, b];
                dst[y * w + x] = [
                    r.saturating_add(noise).saturating_add(5),
                    g.saturating_sub(3),
                    b.saturating_add(2),
                ];
            }
        }

        let cfg_ext = ZensimConfig {
            extended_features: true,
            compute_iw_features: false,
            compute_all_features: true,
            extended_masking_strength: 4.0,
            iw_strength: 4.0,
            ..Default::default()
        };
        let cfg_iw = ZensimConfig {
            extended_features: false,
            compute_iw_features: true,
            compute_all_features: true,
            extended_masking_strength: 4.0,
            iw_strength: 4.0,
            ..Default::default()
        };
        let cfg_both = ZensimConfig {
            extended_features: true,
            compute_iw_features: true,
            compute_all_features: true,
            extended_masking_strength: 4.0,
            iw_strength: 4.0,
            ..Default::default()
        };
        let r_ext = compute_zensim_with_config(&src, &dst, w, h, cfg_ext).unwrap();
        let r_iw = compute_zensim_with_config(&src, &dst, w, h, cfg_iw).unwrap();
        let r_both = compute_zensim_with_config(&src, &dst, w, h, cfg_both).unwrap();

        assert_eq!(r_ext.features.len(), 300);
        assert_eq!(r_iw.features.len(), 300);
        assert_eq!(r_both.features.len(), 372);

        // First 228 features (basic + peaks) must agree across all 3.
        for i in 0..228 {
            let de = (r_ext.features[i] - r_both.features[i]).abs();
            let di = (r_iw.features[i] - r_both.features[i]).abs();
            assert!(
                de < 1e-9,
                "basic feature {} disagrees: ext_only {} vs both {}",
                i,
                r_ext.features[i],
                r_both.features[i]
            );
            assert!(
                di < 1e-9,
                "basic feature {} disagrees: iw_only {} vs both {}",
                i,
                r_iw.features[i],
                r_both.features[i]
            );
        }

        // Masked block (228..300 in r_ext, 228..300 in r_both) must agree
        // — same kernel called identically.
        let mut max_diff = 0.0f64;
        for i in 0..72 {
            let d = (r_ext.features[228 + i] - r_both.features[228 + i]).abs();
            let denom = r_ext.features[228 + i].abs().max(1e-9);
            // Use a relative tolerance — small numerical drift from
            // FMA reordering in the fused 2-mask vs single-mask kernels
            // is acceptable. The bound of 1e-4 relative captures any
            // structural correctness bug while permitting last-bit FMA
            // variability.
            let rel = d / denom;
            max_diff = max_diff.max(rel);
            assert!(
                rel < 1e-4,
                "masked feature {} differs: ext_only {} vs both {} (rel diff {:e})",
                i,
                r_ext.features[228 + i],
                r_both.features[228 + i],
                rel,
            );
        }
        // IW block (228..300 in r_iw, 300..372 in r_both) must agree.
        for i in 0..72 {
            let d = (r_iw.features[228 + i] - r_both.features[300 + i]).abs();
            let denom = r_iw.features[228 + i].abs().max(1e-9);
            let rel = d / denom;
            max_diff = max_diff.max(rel);
            assert!(
                rel < 1e-4,
                "iw feature {} differs: iw_only {} vs both {} (rel diff {:e})",
                i,
                r_iw.features[228 + i],
                r_both.features[300 + i],
                rel,
            );
        }
        // Sanity: not all zero
        assert!(
            max_diff < 1e-4 && max_diff >= 0.0,
            "max_diff out of range: {}",
            max_diff
        );
    }

    /// Extended features: default config produces same score as non-extended.
    #[test]
    fn extended_features_backward_compat() {
        let (w, h) = (64, 64);
        let (src, dst) = make_gradient_pair(w, h);

        let basic = compute_zensim_with_config(&src, &dst, w, h, ZensimConfig::default()).unwrap();

        let extended = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                extended_features: false,
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Both produce 228 features now (peaks always included)
        assert_eq!(basic.features.len(), 228);
        assert_eq!(extended.features.len(), 228);
        // Score should be the same — compute_all forces all channels active but result is same
        assert!(
            (basic.score - extended.score).abs() < 0.01,
            "basic {} vs compute_all {}",
            basic.score,
            extended.score,
        );
    }

    /// Extended features produce 300 values and all are non-negative.
    #[test]
    fn extended_features_count_and_nonneg() {
        let (w, h) = (64, 64);
        let (src, dst) = make_gradient_pair(w, h);

        let result = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                extended_features: true,
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(
            result.features.len(),
            300,
            "Expected 25 × 3 × 4 = 300 features"
        );
        for (i, &f) in result.features.iter().enumerate() {
            assert!(f >= 0.0, "Feature {} is negative: {}", i, f);
        }
    }

    /// ssim_max >= ssim_4th >= ssim_mean ordering.
    #[test]
    fn extended_features_ordering() {
        let (w, h) = (64, 64);
        let (src, dst) = make_gradient_pair(w, h);

        let result = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                extended_features: true,
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Feature layout (block-separated):
        //   [0..156)   scored: 13/ch × 3ch × 4 scales
        //   [156..228)  peaks: 6/ch × 3ch × 4 scales
        //   [228..300) masked: 6/ch × 3ch × 4 scales
        let scored_per_ch = FEATURES_PER_CHANNEL_BASIC; // 13
        let peaks_offset = 4 * scored_per_ch * 3; // 156
        let peaks_per_ch = 6;
        for scale in 0..4 {
            for ch in 0..3 {
                let scored_base = scale * scored_per_ch * 3 + ch * scored_per_ch;
                let peaks_base = peaks_offset + scale * peaks_per_ch * 3 + ch * peaks_per_ch;
                let ssim_mean = result.features[scored_base]; // scored[0]
                let ssim_4th = result.features[scored_base + 1]; // scored[1]
                let ssim_max = result.features[peaks_base]; // peaks[0]
                let ssim_p95 = result.features[peaks_base + 3]; // peaks[3]

                // max >= 4th >= mean (4th is L4 norm, always >= mean for non-negative values)
                assert!(
                    ssim_max >= ssim_4th - 1e-10,
                    "s{} c{}: max {:.6} < 4th {:.6}",
                    scale,
                    ch,
                    ssim_max,
                    ssim_4th,
                );
                assert!(
                    ssim_4th >= ssim_mean - 1e-10,
                    "s{} c{}: 4th {:.6} < mean {:.6}",
                    scale,
                    ch,
                    ssim_4th,
                    ssim_mean,
                );
                // p95 between 4th and max
                assert!(
                    ssim_p95 <= ssim_max + 1e-10,
                    "s{} c{}: p95 {:.6} > max {:.6}",
                    scale,
                    ch,
                    ssim_p95,
                    ssim_max,
                );
            }
        }
    }

    /// Identical images: all features zero.
    #[test]
    fn extended_features_identical_zero() {
        let (w, h) = (64, 64);
        let (src, _) = make_gradient_pair(w, h);

        let result = compute_zensim_with_config(
            &src,
            &src,
            w,
            h,
            ZensimConfig {
                extended_features: true,
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.score, 100.0);
        assert_eq!(result.features.len(), 300);
        for (i, &f) in result.features.iter().enumerate() {
            assert!(
                f.abs() < 1e-10,
                "Feature {} not zero for identical: {}",
                i,
                f
            );
        }
    }

    /// Masked features <= unmasked features (masking reduces).
    #[test]
    fn extended_masked_leq_unmasked() {
        let (w, h) = (64, 64);
        let (src, dst) = make_gradient_pair(w, h);

        let result = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                extended_features: true,
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Feature layout (block-separated):
        //   [0..156)   scored: 13/ch × 3ch × 4 scales
        //   [156..228)  peaks: 6/ch × 3ch × 4 scales
        //   [228..300) masked: 6/ch × 3ch × 4 scales
        let scored_per_ch = FEATURES_PER_CHANNEL_BASIC; // 13
        let masked_offset = 4 * scored_per_ch * 3 + 4 * 6 * 3; // 156 + 72 = 228
        let masked_per_ch = 6;
        for scale in 0..4 {
            for ch in 0..3 {
                let scored_base = scale * scored_per_ch * 3 + ch * scored_per_ch;
                let masked_base = masked_offset + scale * masked_per_ch * 3 + ch * masked_per_ch;
                let ssim_mean = result.features[scored_base]; // scored[0]
                let ssim_4th = result.features[scored_base + 1]; // scored[1]
                let ssim_2nd = result.features[scored_base + 2]; // scored[2]
                let masked_ssim_mean = result.features[masked_base]; // masked[0]
                let masked_ssim_4th = result.features[masked_base + 1]; // masked[1]
                let masked_ssim_2nd = result.features[masked_base + 2]; // masked[2]

                // Masked values should be <= unmasked (mask weights ∈ [0,1])
                assert!(
                    masked_ssim_mean <= ssim_mean + 1e-10,
                    "s{} c{}: masked_mean {:.6} > mean {:.6}",
                    scale,
                    ch,
                    masked_ssim_mean,
                    ssim_mean,
                );
                assert!(
                    masked_ssim_4th <= ssim_4th + 1e-10,
                    "s{} c{}: masked_4th {:.6} > 4th {:.6}",
                    scale,
                    ch,
                    masked_ssim_4th,
                    ssim_4th,
                );
                assert!(
                    masked_ssim_2nd <= ssim_2nd + 1e-10,
                    "s{} c{}: masked_2nd {:.6} > 2nd {:.6}",
                    scale,
                    ch,
                    masked_ssim_2nd,
                    ssim_2nd,
                );
            }
        }
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
