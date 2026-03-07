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
    Box { passes: u8 },
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

/// **Bottom line:** the defaults (`blur_passes=1`) give peak performance.
#[derive(Debug, Clone, Copy)]
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

/// Map a raw weighted distance to the 0–100 quality score with custom parameters.
///
/// `score = 100 - a * d^b`, clamped to \[0, 100\].
fn distance_to_score_mapped(raw_distance: f64, a: f64, b: f64) -> f64 {
    if raw_distance <= 0.0 {
        100.0
    } else {
        (100.0 - a * raw_distance.powf(b)).max(0.0)
    }
}

/// Compute score from raw features using custom weights.
/// `features`: raw features from ZensimResult.features
/// `weights`: one weight per feature (len must equal features.len())
/// Returns (score, raw_distance)
#[cfg_attr(not(feature = "training"), allow(dead_code))]
pub fn score_from_features(features: &[f64], weights: &[f64]) -> (f64, f64) {
    assert_eq!(
        features.len(),
        weights.len(),
        "features and weights must have same length"
    );
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
    (distance_to_score(raw_distance), raw_distance)
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
    if source.len() != width * height {
        return Err(ZensimError::InvalidDataLength);
    }
    let src_img = crate::source::RgbSlice::new(source, width, height);
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
    if distorted.len() != width * height {
        return Err(ZensimError::InvalidDataLength);
    }
    let dst_img = crate::source::RgbSlice::new(distorted, width, height);
    let result = crate::streaming::compute_zensim_streaming_with_ref(
        precomputed,
        &dst_img,
        &config,
        &WEIGHTS,
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
    ((100.0 - score) / 100.0).max(0.0)
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
}

impl Zensim {
    /// Create a new `Zensim` with the given profile. Parallel by default.
    pub fn new(profile: ZensimProfile) -> Self {
        Self {
            profile,
            parallel: true,
        }
    }

    /// Enable or disable multi-threaded computation (rayon).
    /// Default: `true`.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
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
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        let config = config_from_params(params, self.parallel);
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
    pub fn compute_with_ref(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        if distorted.width() < 8 || distorted.height() < 8 {
            return Err(ZensimError::ImageTooSmall);
        }
        let config = config_from_params(params, self.parallel);
        let result = crate::streaming::compute_zensim_streaming_with_ref(
            precomputed,
            distorted,
            &config,
            params.weights,
        );
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

        // Compute delta stats (pixel-level analysis in sRGB space)
        let delta_stats = crate::streaming::compute_delta_stats(source, distorted);

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

    let (best_cat, best_score) = scores
        .iter()
        .copied()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

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

/// Check if source and distorted images have byte-identical pixel data
/// and matching color interpretation (format + primaries).
fn images_byte_identical(source: &impl ImageSource, distorted: &impl ImageSource) -> bool {
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
    let bpp = source.pixel_format().bytes_per_pixel();
    let row_len = w * bpp;
    for y in 0..h {
        let src_row = source.row_bytes(y);
        let dst_row = distorted.row_bytes(y);
        if src_row[..row_len] != dst_row[..row_len] {
            return false;
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
        let fpc = if config.extended_features {
            FEATURES_PER_CHANNEL_EXTENDED
        } else {
            FEATURES_PER_CHANNEL_WITH_PEAKS
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
        compute_all_features: false,
        extended_features: false,
        extended_masking_strength: 4.0,
        num_scales: params.num_scales,
        score_mapping_a: params.score_mapping_a,
        score_mapping_b: params.score_mapping_b,
        allow_multithreading: parallel,
    }
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

/// XYB channel index constants for readability.
#[cfg(feature = "training")]
pub const CH_X: usize = 0;
#[cfg(feature = "training")]
pub const CH_Y: usize = 1;
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
    if source.len() != width * height {
        return Err(ZensimError::InvalidDataLength);
    }
    if distorted.len() != width * height {
        return Err(ZensimError::InvalidDataLength);
    }
    if source.len() != distorted.len() {
        return Err(ZensimError::DimensionMismatch);
    }

    // Identical images must score exactly 100.0 — short-circuit before
    // floating-point arithmetic introduces sub-ULP noise in SSIM/edge features.
    if source == distorted {
        let fpc = if config.extended_features {
            FEATURES_PER_CHANNEL_EXTENDED
        } else {
            FEATURES_PER_CHANNEL_WITH_PEAKS
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

    let result = crate::streaming::compute_zensim_streaming(&src_img, &dst_img, &config, &WEIGHTS);
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

/// Trained weights from v2 synthetic dataset (344k pairs, 5-fold CV SROCC=0.9936).
///
/// Optimized against GPU-accelerated SSIM2 scores on a diverse synthetic dataset
/// (4 codecs × 11 quality levels × 6 aspect ratios × 276 source images).
/// SROCC = 0.9941 on the full training set.
///
/// Layout: 4 scales × 3 channels (X,Y,B) × 13 basic features, then
///         4 scales × 3 channels × 6 peak features:
///   Basic: ssim_mean, ssim_4th, ssim_2nd, art_mean, art_4th, art_2nd,
///          det_mean, det_4th, det_2nd, mse, hf_energy_loss, hf_mag_loss, hf_energy_gain
///   Peaks: ssim_max, art_max, det_max, ssim_p95, art_p95, det_p95
#[cfg(any(feature = "training", test))]
#[allow(clippy::excessive_precision)]
pub const WEIGHTS: [f64; 228] = [
    // --- Basic features (13/ch × 3ch × 4 scales = 156) ---
    0.0000000000,
    0.1391674808,
    0.0000000000,
    0.0055172171,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0010650645,
    0.0071194723,
    69.6110793540,
    0.0106660235,
    0.0076379521,
    0.0051069220, // Scale 0 Channel X
    17.8445125125,
    1.9157888513,
    0.0109886875,
    0.0048996910,
    0.0000000000,
    0.0018418193,
    0.0000000000,
    1.5940983560,
    0.0072914879,
    0.0000000000,
    0.2695940535,
    0.5232582347,
    0.1101639205, // Scale 0 Channel Y
    0.0000000000,
    0.0097680540,
    0.0075408094,
    4.2314204599,
    0.0082993863,
    0.0060063585,
    0.0000000000,
    0.0000000000,
    0.0076442067,
    0.4127212154,
    0.0000000000,
    0.0000000000,
    0.0061137647, // Scale 0 Channel B
    0.0027028659,
    0.1421516497,
    0.0000000000,
    0.0000000000,
    0.0006394302,
    0.0004174259,
    0.0084670378,
    0.0000000000,
    0.0102579245,
    0.0000000000,
    0.0097535151,
    0.0000000000,
    0.0000000091, // Scale 1 Channel X
    22.0713261440,
    52.8548074123,
    87.4350424152,
    5.5343470971,
    8.5458130239,
    0.0026243365,
    0.0000000000,
    0.6444438326,
    0.0000000000,
    0.0000000000,
    0.4690274655,
    0.0111775837,
    0.0000000000, // Scale 1 Channel Y
    0.7853068895,
    0.5804301701,
    0.0000000000,
    241.7223774962,
    0.0852474584,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0046043128,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0092126667, // Scale 1 Channel B
    0.1907664071,
    1.1388072940,
    0.0069950673,
    0.0000000000,
    3.2949756637,
    0.0097480604,
    0.0114461871,
    0.0101092121,
    0.0120198795,
    0.0000000000,
    0.0102984460,
    0.0000000000,
    0.0003411392, // Scale 2 Channel X
    77.8638757528,
    4.9774136371,
    5.7998312546,
    0.0000000000,
    32.6107435348,
    0.0000000000,
    0.0000000000,
    7.3147158634,
    0.0000000000,
    112.3320506295,
    6.5803001760,
    0.9144713387,
    0.0800661074, // Scale 2 Channel Y
    0.6380873029,
    3.4344996615,
    0.0000000000,
    7.9969790535,
    4.0547889928,
    1.2673476404,
    7.9809497222,
    8.8252344733,
    0.0000000000,
    190.1707930678,
    0.0000000000,
    0.0042434316,
    0.0000117426, // Scale 2 Channel B
    42.4928921475,
    1.8499402382,
    18.0908263404,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0022710707,
    0.0000000000,
    0.0000000000,
    0.0068807271,
    0.1494089476,
    0.0001752242, // Scale 3 Channel X
    396.2394144642,
    33.6112684912,
    0.0053195470,
    331.9368790619,
    437.6418006190,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    15.5115983050,
    0.0052803584,
    0.0703659816, // Scale 3 Channel Y
    112.4036508580,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0073096632,
    0.0000000000,
    0.0091600012,
    0.0000000000,
    0.0000000000,
    0.0072861510,
    0.0493312705,
    0.0049937361, // Scale 3 Channel B
    // --- Peak features (6/ch × 3ch × 4 scales = 72) ---
    1.6405231709,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    1.8173590152,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    28.5681479205,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 0
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    1.7833707251,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    17.5252532711,
    0.0000000000, // Scale 1
    0.0000000000,
    31.1123311855,
    0.0000000000,
    0.0000000000,
    3.4969161675,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    3.4593661665,
    0.0000000000,
    56.7768222287,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    5.3758924006,
    0.0000000000, // Scale 2
    0.0000000000,
    1.6125342576,
    47.2133536610,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 3
];

pub(crate) fn combine_scores(
    scale_stats: &[ScaleStats],
    weights: &[f64],
    config: &ZensimConfig,
    mean_offset: [f64; 3],
) -> ZensimResult {
    let extended = config.extended_features;

    // Feature vector layout:
    //   [0..N_basic)        — 13/ch × 3ch × n_scales (basic features)
    //   [N_basic..N_peaks)  — 6/ch × 3ch × n_scales peak features (always included)
    //   [N_peaks..N_all)    — 6/ch × 3ch × n_scales masked features (if extended)
    //
    // Both basic and peak features are scored: features[0..WEIGHTS.len()]
    // produces the dot product used for the final score.
    let n_scales = scale_stats.len();
    let basic_per_ch = FEATURES_PER_CHANNEL_BASIC; // 13
    let basic_total = n_scales * basic_per_ch * 3;
    let peak_total = n_scales * 6 * 3;
    let masked_total = if extended { n_scales * 6 * 3 } else { 0 };
    let total = basic_total + peak_total + masked_total;

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
}
