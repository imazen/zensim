//! Core zensim metric computation.
//!
//! Multi-scale SSIM + edge + high-frequency features in XYB color space,
//! with trained weights per feature.
//!
//! # Feature extraction pipeline
//!
//! Both images are converted to the XYB perceptual color space (cube-root LMS,
//! same as ssimulacra2 and butteraugli), then processed at multiple scales.
//! Each scale halves resolution via 2× box downscale. At each scale, 13 features
//! are extracted per XYB channel (X, Y, B), giving **156 features total**
//! (4 scales × 3 channels × 13 features).
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
//! ## Scoring
//!
//! The 156 features are multiplied by trained weights, summed, normalized by
//! scale count, then mapped to a 0–100 score via:
//! `score = 100 - a · distance^b` (default a=18.0, b=0.7).

use crate::blur::{
    box_blur_1pass_into, box_blur_2pass_into, box_blur_3pass_into, box_blur_v_from_copy,
    downscale_2x_inplace, fused_blur_h_ssim, pad_plane_width, simd_padded_width,
};
use crate::color::srgb_to_positive_xyb_planar;
use crate::error::ZensimError;
use crate::pool::ScaleBuffers;
use crate::simd_ops::{
    abs_diff_into, abs_diff_sum, edge_diff_channel, edge_diff_channel_masked, mul_into,
    sq_diff_sum, sq_sum_into, ssim_channel, ssim_channel_masked,
};

/// Configuration for zensim computation.
///
/// # Performance paths
///
/// The metric has two internal computation strategies with different performance
/// characteristics. Which path is taken depends on these settings:
///
/// ## Streaming path (default, fastest)
///
/// Processes scale 0 in horizontal strips with fused blur+feature extraction,
/// minimizing memory traffic. Used when **all** of:
/// - `masking_strength == 0.0` (no per-pixel masking)
/// - `blur_passes == 1` (enables fused H-blur + V-blur+reduce kernels)
///
/// When `blur_passes != 1`, the streaming path still processes in strips but
/// falls back to separate blur + reduce passes per channel (slower but still
/// cache-friendly).
///
/// ## Full-image path (masking or legacy API)
///
/// Materializes full XYB planes in memory, then computes blur and features
/// per-scale. Required when `masking_strength > 0.0` because masking needs
/// the full blurred source plane to compute per-pixel activity weights.
/// Uses phased blur parallelism for large images (>512×512) when not masking.
///
/// **Bottom line:** the defaults (`blur_passes=1`, `masking_strength=0.0`) give
/// peak performance. Changing either has a measurable cost.
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

    /// Compute all 156 features even when their weights are zero (default: false).
    ///
    /// When false, channels/features with zero weight are skipped entirely.
    /// Enable for weight training to avoid circular dependency (need all features
    /// to determine which weights should be nonzero).
    pub compute_all_features: bool,

    /// Local contrast masking strength (default: 0.0 = disabled).
    ///
    /// When > 0, computes per-pixel activity from the source image:
    /// `mask[i] = 1 / (1 + masking_strength * blur(|src - mu|))`,
    /// then weights SSIM and edge distances by this mask. Textured/edge-heavy
    /// regions get down-weighted, modeling the human visual system's reduced
    /// sensitivity to distortion in busy areas.
    ///
    /// **Performance:** enabling masking forces the full-image path (no streaming)
    /// and adds one extra blur per channel per scale for the activity computation.
    /// Expect ~2× slower than the default streaming path.
    ///
    /// **Current profiles:** all ship with `masking_strength = 0.0` because the
    /// unmasked weights already achieve SROCC ≥ 0.98 on synthetic data. This
    /// parameter exists for training exploration — a future profile *could* use
    /// masking if it improves correlation on specific content types.
    ///
    /// Typical training range: 2.0–8.0.
    pub masking_strength: f32,

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
}

impl Default for ZensimConfig {
    fn default() -> Self {
        Self {
            blur_radius: 5,
            blur_passes: 1,
            compute_all_features: false,
            masking_strength: 0.0,
            num_scales: crate::NUM_SCALES,
            score_mapping_a: 18.0,
            score_mapping_b: 0.7,
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
pub fn distance_to_score(raw_distance: f64) -> f64 {
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
    // Normalize by number of scales
    let features_per_scale = FEATURES_PER_CHANNEL_BASIC * 3;
    let raw_distance = raw_distance / (features.len() as f64 / features_per_scale as f64).max(1.0);
    (distance_to_score(raw_distance), raw_distance)
}

/// Pre-compute reference image data for reuse across multiple distorted comparisons.
///
/// Converts sRGB pixels to XYB color space and builds the multi-scale downscale pyramid,
/// caching both for reuse with [`compute_zensim_with_ref`]. Uses the default 4 pyramid
/// scales.
///
/// # When to use
///
/// When comparing one reference against many distorted variants. Saves ~25% at 4K and
/// ~34% at 8K per comparison after amortizing the precompute cost (break-even: 3-7
/// distorted images per reference).
///
/// # Errors
///
/// Returns [`ZensimError::ImageTooSmall`] if width or height < 8, or
/// [`ZensimError::InvalidDataLength`] if `source.len() != width * height`.
pub fn precompute_reference(
    source: &[[u8; 3]],
    width: usize,
    height: usize,
) -> Result<crate::streaming::PrecomputedReference, ZensimError> {
    if width < 8 || height < 8 {
        return Err(ZensimError::ImageTooSmall);
    }
    if source.len() != width * height {
        return Err(ZensimError::InvalidDataLength);
    }
    let src_img = crate::source::RgbSlice::new(source, width, height);
    let config = ZensimConfig::default();
    Ok(crate::streaming::PrecomputedReference::new(
        &src_img,
        config.num_scales,
    ))
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
        &src_img, num_scales,
    ))
}

/// Compute zensim score using a [`PrecomputedReference`](crate::PrecomputedReference).
///
/// Equivalent to [`compute_zensim`] but skips the reference image's XYB conversion
/// and pyramid construction. The distorted image must have the same dimensions as
/// the reference that was precomputed.
///
/// # Errors
///
/// Returns [`ZensimError::ImageTooSmall`] if width or height < 8, or
/// [`ZensimError::InvalidDataLength`] if `distorted.len() != width * height`.
pub fn compute_zensim_with_ref(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
) -> Result<ZensimResult, ZensimError> {
    if width < 8 || height < 8 {
        return Err(ZensimError::ImageTooSmall);
    }
    if distorted.len() != width * height {
        return Err(ZensimError::InvalidDataLength);
    }
    let dst_img = crate::source::RgbSlice::new(distorted, width, height);
    let config = ZensimConfig::default();
    let result = crate::streaming::compute_zensim_streaming_with_ref(
        precomputed,
        &dst_img,
        &config,
        &WEIGHTS,
    );
    Ok(result)
}

/// Compute zensim with a precomputed reference and custom configuration.
///
/// Training/research variant of [`compute_zensim_with_ref`]. The `config.num_scales`
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
}

/// Result from a zensim comparison.
///
/// Contains the final score, the raw distance used to derive it, and the
/// full per-scale feature vector (useful for diagnostics or weight training).
#[derive(Debug, Clone)]
pub struct ZensimResult {
    /// Quality score on a 0–100 scale. 100 = identical, 0 = maximally different.
    /// Derived from `raw_distance` via a power-law mapping.
    pub score: f64,
    /// Raw weighted feature distance before nonlinear mapping. Lower = more similar.
    /// Not bounded to a fixed range; depends on image content and weights.
    pub raw_distance: f64,
    /// Per-scale raw features (156 values for the default 4-scale configuration).
    ///
    /// Layout: 4 scales × 3 channels (X, Y, B) × 13 features per channel:
    /// `ssim_mean, ssim_4th, ssim_2nd, art_mean, art_4th, art_2nd,
    ///  det_mean, det_4th, det_2nd, mse, var_loss, tex_loss, contrast_inc`
    pub features: Vec<f64>,
    /// Which profile produced this score.
    pub profile: crate::profile::ZensimProfile,
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
/// println!("{}: {:.2}", result.profile, result.score);
/// ```
#[derive(Clone, Debug)]
pub struct Zensim {
    profile: ZensimProfile,
}

impl Zensim {
    /// Create a new `Zensim` with the given profile.
    pub fn new(profile: ZensimProfile) -> Self {
        Self { profile }
    }

    /// Current profile.
    pub fn profile(&self) -> ZensimProfile {
        self.profile
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
        let config = config_from_params(params);
        let mut result = compute_with_config_inner(source, distorted, &config, params.weights);
        result.profile = self.profile;
        Ok(result)
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
        let config = config_from_params(params);
        let mut result = crate::streaming::compute_zensim_streaming_with_ref(
            precomputed,
            distorted,
            &config,
            params.weights,
        );
        result.profile = self.profile;
        Ok(result)
    }

    /// Like `compute`, but always computes all 156 features regardless of
    /// zero weights. For training/research.
    #[cfg(feature = "training")]
    pub fn compute_all_features(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        let mut config = config_from_params(params);
        config.compute_all_features = true;
        let mut result = compute_with_config_inner(source, distorted, &config, params.weights);
        result.profile = self.profile;
        Ok(result)
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
        let config = config_from_params(params);
        let result = compute_with_config_inner(source, distorted, &config, params.weights);
        Ok(result)
    }
}

fn validate_pair(
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

/// Core computation routing: streaming vs full-image based on config.
///
/// Streaming is used when masking is disabled (the common case).
/// Full-image is forced when masking_strength > 0.0 because masking
/// needs the full blurred source plane to compute per-pixel activity weights.
fn compute_with_config_inner(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64; 156],
) -> ZensimResult {
    let masked = config.masking_strength > 0.0;
    if !masked {
        return crate::streaming::compute_zensim_streaming(source, distorted, config, weights);
    }

    // Full-image path: materialize XYB planes, pad, compute stats
    let width = source.width();
    let height = source.height();
    let padded_width = simd_padded_width(width);

    let (mut src_xyb, mut dst_xyb) = std::thread::scope(|s| {
        let src_handle =
            s.spawn(|| crate::streaming::convert_source_to_xyb_parallel(source, padded_width));
        let dst_xyb = crate::streaming::convert_source_to_xyb_parallel(distorted, padded_width);
        let src_xyb = src_handle.join().unwrap();
        (src_xyb, dst_xyb)
    });

    // Pad if needed (convert_source_to_xyb_parallel already pads to padded_width)
    if padded_width != width {
        for c in 0..3 {
            pad_plane_width(&mut src_xyb[c], width, height, padded_width);
            pad_plane_width(&mut dst_xyb[c], width, height, padded_width);
        }
    }

    let scale_stats = compute_multiscale_stats(src_xyb, dst_xyb, padded_width, height, config);
    combine_scores(&scale_stats, masked, weights, config)
}

fn config_from_params(params: &ProfileParams) -> ZensimConfig {
    ZensimConfig {
        blur_radius: params.blur_radius,
        blur_passes: params.blur_passes,
        compute_all_features: false,
        masking_strength: params.masking_strength,
        num_scales: params.num_scales,
        score_mapping_a: params.score_mapping_a,
        score_mapping_b: params.score_mapping_b,
    }
}

/// Features per channel per scale: 13 features always emitted.
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
/// ```
///
/// Total features = `num_scales × 3 channels × 13` = 156 at 4 scales.
pub(crate) const FEATURES_PER_CHANNEL_BASIC: usize = 13;

/// Compute zensim score between two sRGB u8 images.
///
/// Uses the latest general-purpose profile. For explicit profile selection, use
/// [`Zensim::new`] with a [`ZensimProfile`].
///
/// Both images must be the same dimensions (at least 8×8). Pixels are sRGB
/// `[R, G, B]` with 8 bits per channel.
///
/// Returns a score on 0-100 scale where 100 = identical.
///
/// # Errors
///
/// Returns [`ZensimError::ImageTooSmall`] if width or height < 8,
/// [`ZensimError::InvalidDataLength`] if pixel count != width × height,
/// or [`ZensimError::DimensionMismatch`] if source and distorted differ in length.
pub fn compute_zensim(
    source: &[[u8; 3]],
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
) -> Result<ZensimResult, ZensimError> {
    compute_zensim_with_config(source, distorted, width, height, ZensimConfig::default())
}

/// Compute zensim score between two RGBA images with alpha compositing.
///
/// Both images are composited over the same 8×8 checkerboard (black/white)
/// before comparison, so alpha differences produce visible perceptual
/// distortion — fringing, incorrect transparency, and blending artifacts
/// all produce measurable scores.
///
/// Pixels are sRGB `[R, G, B, A]` with straight (non-premultiplied) alpha.
/// Fully opaque images (all alpha = 255) degrade to a simple channel strip
/// with no blending overhead.
///
/// # Errors
///
/// Same as [`compute_zensim`].
pub fn compute_zensim_rgba(
    source: &[[u8; 4]],
    distorted: &[[u8; 4]],
    width: usize,
    height: usize,
) -> Result<ZensimResult, ZensimError> {
    let src_rgb = crate::color::rgba_checkerboard_composite(source, width);
    let dst_rgb = crate::color::rgba_checkerboard_composite(distorted, width);
    compute_zensim(&src_rgb, &dst_rgb, width, height)
}

/// Pre-compute reference data from an RGBA image for batch comparisons.
///
/// RGBA variant of [`precompute_reference`]. The source is composited over
/// a checkerboard before building the pyramid. Use with
/// [`compute_zensim_with_ref_rgba`].
///
/// # Errors
///
/// Same as [`precompute_reference`].
pub fn precompute_reference_rgba(
    source: &[[u8; 4]],
    width: usize,
    height: usize,
) -> Result<crate::streaming::PrecomputedReference, ZensimError> {
    let src_rgb = crate::color::rgba_checkerboard_composite(source, width);
    precompute_reference(&src_rgb, width, height)
}

/// Compute zensim score using a precomputed RGBA reference.
///
/// RGBA variant of [`compute_zensim_with_ref`]. The distorted image is
/// composited over the same checkerboard used by [`precompute_reference_rgba`].
///
/// # Errors
///
/// Same as [`compute_zensim_with_ref`].
pub fn compute_zensim_with_ref_rgba(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &[[u8; 4]],
    width: usize,
    height: usize,
) -> Result<ZensimResult, ZensimError> {
    let dst_rgb = crate::color::rgba_checkerboard_composite(distorted, width);
    compute_zensim_with_ref(precomputed, &dst_rgb, width, height)
}

/// Compute zensim with custom configuration.
///
/// Uses the v0.2 weights (latest general-purpose profile).
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

    let src_img = crate::source::RgbSlice::new(source, width, height);
    let dst_img = crate::source::RgbSlice::new(distorted, width, height);

    // Use streaming path for non-masked images (faster at all sizes)
    let masked = config.masking_strength > 0.0;
    if !masked && crate::streaming::should_use_streaming(width, height) {
        let result =
            crate::streaming::compute_zensim_streaming(&src_img, &dst_img, &config, &WEIGHTS);
        return Ok(result);
    }

    // Full-image path for small images or masking mode
    // Convert both images to planar positive XYB in parallel
    let (mut src_xyb, mut dst_xyb) = std::thread::scope(|s| {
        let src_handle = s.spawn(|| srgb_to_positive_xyb_planar(source));
        let dst_xyb = srgb_to_positive_xyb_planar(distorted);
        let src_xyb = src_handle.join().unwrap();
        (src_xyb, dst_xyb)
    });

    // Pad plane widths to multiple of 16 for consistent SIMD utilization
    let padded_width = simd_padded_width(width);
    if padded_width != width {
        for c in 0..3 {
            pad_plane_width(&mut src_xyb[c], width, height, padded_width);
            pad_plane_width(&mut dst_xyb[c], width, height, padded_width);
        }
    }

    // Compute multi-scale statistics (take ownership to avoid clone)
    let scale_stats = compute_multiscale_stats(src_xyb, dst_xyb, padded_width, height, &config);

    // Combine with weights to produce final score
    let result = combine_scores(&scale_stats, masked, &WEIGHTS, &config);
    Ok(result)
}

/// Compute per-scale SSIM and edge statistics.
fn compute_multiscale_stats(
    src_xyb: [Vec<f32>; 3],
    dst_xyb: [Vec<f32>; 3],
    width: usize,
    height: usize,
    config: &ZensimConfig,
) -> Vec<ScaleStats> {
    let num_scales = config.num_scales;
    let mut stats = Vec::with_capacity(num_scales);

    let mut src_planes = src_xyb;
    let mut dst_planes = dst_xyb;
    let mut w = width;
    let mut h = height;

    // Pre-allocate two buffer sets: one for main thread, one for parallel thread.
    let max_n = width * height;
    let mut bufs = ScaleBuffers::new(max_n);
    let mut parallel_bufs = ScaleBuffers::new(max_n);

    for scale in 0..num_scales {
        if w < 8 || h < 8 {
            break;
        }

        let n = w * h;
        bufs.resize(n);
        parallel_bufs.resize(n);

        let scale_stat = compute_single_scale(
            &src_planes,
            &dst_planes,
            w,
            h,
            config,
            &mut bufs,
            &mut parallel_bufs,
            scale,
        );
        stats.push(scale_stat);

        // Downscale for next level (in-place, no allocations)
        if scale < num_scales - 1 {
            let mut nw = 0;
            let mut nh = 0;
            for c in 0..3 {
                let (sw, sh) = downscale_2x_inplace(&mut src_planes[c], w, h);
                let _ = downscale_2x_inplace(&mut dst_planes[c], w, h);
                nw = sw;
                nh = sh;
            }
            // Don't re-pad after downscale: padding is right-only, so padded
            // pixels participate in metric reductions and break left-right symmetry.
            // The SIMD cascade (v4→v3→scalar) handles arbitrary widths efficiently.
            w = nw;
            h = nh;
        }
    }

    stats
}

/// Per-channel result from compute_channel.
struct ChannelResult {
    ssim: [f64; 2],      // [mean_d, root4_d]
    edge: [f64; 4],      // [art_mean, art_4th, det_mean, det_4th]
    hf_energy_loss: f64, // max(0, 1 - hf_L2_dst / hf_L2_src)
    hf_mag_loss: f64,    // max(0, 1 - hf_L1_dst / hf_L1_src)
    hf_energy_gain: f64, // max(0, hf_L2_dst / hf_L2_src - 1)
    ssim_2nd: f64,       // root2 pooled SSIM
    edge_2nd: [f64; 2],  // [art_2nd, det_2nd]
}

/// Compute SSIM and/or edge features for a single channel.
/// Self-contained: allocates its own buffers to enable parallel execution.
#[allow(clippy::too_many_arguments)]
fn compute_channel(
    src_c: &[f32],
    dst_c: &[f32],
    width: usize,
    height: usize,
    config: &ZensimConfig,
    need_ssim: bool,
    need_edge: bool,
    bufs: &mut ScaleBuffers,
) -> ChannelResult {
    let n = width * height;
    let one_over_n = 1.0 / n as f64;
    let mut ssim = [0.0f64; 2];
    let mut edge = [0.0f64; 4];
    let mut ssim_2nd = 0.0f64;
    let mut edge_2nd = [0.0f64; 2];
    let masked = config.masking_strength > 0.0;

    #[allow(clippy::type_complexity)]
    let blur_fn: fn(&[f32], &mut [f32], &mut [f32], usize, usize, usize) = match config.blur_passes
    {
        1 => box_blur_1pass_into,
        2 => box_blur_2pass_into,
        _ => box_blur_3pass_into,
    };

    // Fused path: for 1-pass SSIM channels, compute all 4 H-blurs in one pass
    // then complete with 4 separate V-blurs. Saves 3 H-passes + 2 element-wise ops.
    let use_fused = need_ssim && config.blur_passes == 1;

    if use_fused {
        // Fused horizontal blur: reads src_c and dst_c once, produces H-blurred
        // mu1, mu2, sigma_sq, sigma12 in bufs.mu1/mu2/sigma1_sq/sigma12
        // (using mul_buf as temporary to hold H-blur output for mu1 before V-blur)
        //
        // We need 4 H-blur outputs to feed into 4 V-blurs.
        // Strategy: fused_blur_h writes to mu1, mu2, sigma1_sq, sigma12 (H-blur results)
        // Then we V-blur each in-place via temp_blur.
        // Fused H-blur outputs go to temp_blur, mul_buf, sigma1_sq, sigma12.
        // Then V-blur: temp_blur→mu1, mul_buf→mu2, sigma1_sq↔temp_blur, sigma12↔mul_buf.
        fused_blur_h_ssim(
            src_c,
            dst_c,
            &mut bufs.temp_blur, // H-blurred src → will become mu1 after V-blur
            &mut bufs.mul_buf,   // H-blurred dst → will become mu2 after V-blur
            &mut bufs.sigma1_sq, // H-blurred sq_sum → V-blur in-place via swap
            &mut bufs.sigma12,   // H-blurred product → V-blur in-place via swap
            width,
            height,
            config.blur_radius,
        );
        // V-blur temp_blur(H-blurred src) → mu1
        box_blur_v_from_copy(
            &bufs.temp_blur,
            &mut bufs.mu1,
            width,
            height,
            config.blur_radius,
        );
        // V-blur mul_buf(H-blurred dst) → mu2
        box_blur_v_from_copy(
            &bufs.mul_buf,
            &mut bufs.mu2,
            width,
            height,
            config.blur_radius,
        );
        // V-blur sigma1_sq → temp_blur, then swap back
        box_blur_v_from_copy(
            &bufs.sigma1_sq,
            &mut bufs.temp_blur,
            width,
            height,
            config.blur_radius,
        );
        std::mem::swap(&mut bufs.sigma1_sq, &mut bufs.temp_blur);
        // V-blur sigma12 → mul_buf, then swap back
        box_blur_v_from_copy(
            &bufs.sigma12,
            &mut bufs.mul_buf,
            width,
            height,
            config.blur_radius,
        );
        std::mem::swap(&mut bufs.sigma12, &mut bufs.mul_buf);
    } else {
        // Standard path: separate blur calls for mu1, mu2
        blur_fn(
            src_c,
            &mut bufs.mu1,
            &mut bufs.temp_blur,
            width,
            height,
            config.blur_radius,
        );
        blur_fn(
            dst_c,
            &mut bufs.mu2,
            &mut bufs.temp_blur,
            width,
            height,
            config.blur_radius,
        );
    }

    // Compute masking weights if enabled
    if masked {
        // mask[i] = 1 / (1 + k * blur(|src - mu1|))
        // Uses source-only activity to avoid biasing toward distorted image
        abs_diff_into(src_c, &bufs.mu1, &mut bufs.mul_buf);
        blur_fn(
            &bufs.mul_buf,
            &mut bufs.mask,
            &mut bufs.temp_blur,
            width,
            height,
            config.blur_radius,
        );
        let k = config.masking_strength;
        for i in 0..n {
            bufs.mask[i] = 1.0 / (1.0 + k * bufs.mask[i]);
        }
    }

    if need_ssim && !use_fused {
        // Standard SSIM path: separate element-wise ops + blur
        sq_sum_into(src_c, dst_c, &mut bufs.mul_buf);
        blur_fn(
            &bufs.mul_buf,
            &mut bufs.sigma1_sq,
            &mut bufs.temp_blur,
            width,
            height,
            config.blur_radius,
        );

        mul_into(src_c, dst_c, &mut bufs.mul_buf);
        blur_fn(
            &bufs.mul_buf,
            &mut bufs.sigma12,
            &mut bufs.temp_blur,
            width,
            height,
            config.blur_radius,
        );
    }

    if need_ssim {
        if masked {
            let (sum_d, sum_d4, sum_d2) = ssim_channel_masked(
                &bufs.mu1,
                &bufs.mu2,
                &bufs.sigma1_sq,
                &bufs.sigma12,
                &bufs.mask,
            );
            ssim[0] = sum_d * one_over_n;
            ssim[1] = (sum_d4 * one_over_n).powf(0.25);
            ssim_2nd = (sum_d2 * one_over_n).sqrt();
        } else {
            let (sum_d, sum_d4, sum_d2) =
                ssim_channel(&bufs.mu1, &bufs.mu2, &bufs.sigma1_sq, &bufs.sigma12);
            ssim[0] = sum_d * one_over_n;
            ssim[1] = (sum_d4 * one_over_n).powf(0.25);
            ssim_2nd = (sum_d2 * one_over_n).sqrt();
        }
    }

    if need_edge {
        if masked {
            let (art, art4, det, det4, art2, det2) =
                edge_diff_channel_masked(src_c, dst_c, &bufs.mu1, &bufs.mu2, &bufs.mask);
            edge[0] = art * one_over_n;
            edge[1] = (art4 * one_over_n).powf(0.25);
            edge[2] = det * one_over_n;
            edge[3] = (det4 * one_over_n).powf(0.25);
            edge_2nd[0] = (art2 * one_over_n).sqrt();
            edge_2nd[1] = (det2 * one_over_n).sqrt();
        } else {
            let (art, art4, det, det4, art2, det2) =
                edge_diff_channel(src_c, dst_c, &bufs.mu1, &bufs.mu2);
            edge[0] = art * one_over_n;
            edge[1] = (art4 * one_over_n).powf(0.25);
            edge[2] = det * one_over_n;
            edge[3] = (det4 * one_over_n).powf(0.25);
            edge_2nd[0] = (art2 * one_over_n).sqrt();
            edge_2nd[1] = (det2 * one_over_n).sqrt();
        }
    }

    // HF energy loss (L2): 1 - Σ(dst-mu)²_dst / Σ(src-mu)²_src
    let var_src = sq_diff_sum(&src_c[..n], &bufs.mu1[..n]) * one_over_n;
    let var_dst = sq_diff_sum(&dst_c[..n], &bufs.mu2[..n]) * one_over_n;
    let hf_energy_loss = if var_src > 1e-10 {
        (1.0 - var_dst / var_src).max(0.0)
    } else {
        0.0
    };

    // HF energy gain (L2): max(0, Σ(dst-mu)²_dst / Σ(src-mu)²_src - 1)
    let hf_energy_gain = if var_src > 1e-10 {
        (var_dst / var_src - 1.0).max(0.0)
    } else {
        0.0
    };

    // HF magnitude loss (L1): 1 - Σ|dst-mu_dst| / Σ|src-mu_src|
    let mad_src = abs_diff_sum(&src_c[..n], &bufs.mu1[..n]) * one_over_n;
    let mad_dst = abs_diff_sum(&dst_c[..n], &bufs.mu2[..n]) * one_over_n;
    let hf_mag_loss = if mad_src > 1e-10 {
        (1.0 - mad_dst / mad_src).max(0.0)
    } else {
        0.0
    };

    ChannelResult {
        ssim,
        edge,
        hf_energy_loss,
        hf_mag_loss,
        hf_energy_gain,
        ssim_2nd,
        edge_2nd,
    }
}

/// Minimum pixel count to justify phased parallel blur (2 sync points, 3 threads).
/// Below this, sequential is faster due to thread overhead.
const PARALLEL_THRESHOLD: usize = 100_000;

/// Compute SSIM and edge statistics for a single scale.
/// Uses phased blur parallelism for large scales (non-masking mode only).
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_single_scale(
    src: &[Vec<f32>; 3],
    dst: &[Vec<f32>; 3],
    width: usize,
    height: usize,
    config: &ZensimConfig,
    bufs: &mut ScaleBuffers,
    parallel_bufs: &mut ScaleBuffers,
    scale_idx: usize,
) -> ScaleStats {
    let mut ssim_vals = [0.0f64; 6];
    let mut edge_vals = [0.0f64; 12];
    let mut mse_vals = [0.0f64; 3];
    let mut hf_energy_loss_vals = [0.0f64; 3];
    let mut hf_mag_loss_vals = [0.0f64; 3];
    let mut hf_energy_gain_vals = [0.0f64; 3];
    let mut ssim_2nd_vals = [0.0f64; 3];
    let mut edge_2nd_vals = [0.0f64; 6];

    let compute_all = config.compute_all_features;
    let masked = config.masking_strength > 0.0;

    // For scales beyond WEIGHTS range, always compute all
    let fpc_basic = FEATURES_PER_CHANNEL_BASIC;

    // Check if any weight is nonzero for a given feature type at this scale+channel
    let has_weight = |base_idx: usize, count: usize| -> bool {
        (base_idx..base_idx + count).all(|i| i < WEIGHTS.len())
            && (base_idx..base_idx + count).any(|i| WEIGHTS[i].abs() > 0.001)
    };

    // Determine which channels need work
    // Feature layout per channel (13): ssim_mean(0), ssim_4th(1), ssim_2nd(2),
    //   art_mean(3), art_4th(4), art_2nd(5), det_mean(6), det_4th(7), det_2nd(8),
    //   mse(9), hf_energy_loss(10), hf_mag_loss(11), hf_energy_gain(12)
    let mut active_channels: Vec<(usize, bool, bool)> = Vec::new();
    let beyond_basic = scale_idx * (fpc_basic * 3) >= WEIGHTS.len();
    for c in 0..3 {
        if beyond_basic {
            if compute_all {
                active_channels.push((c, true, true));
            }
        } else {
            let base = scale_idx * (fpc_basic * 3) + c * fpc_basic;
            let need_ssim = compute_all || has_weight(base, 3); // positions 0-2
            let need_hf = has_weight(base + 10, 3); // positions 10-12
            // HF features need mu1/mu2 (same as edge), fold into need_edge
            let need_edge = compute_all || has_weight(base + 3, 6) || need_hf; // positions 3-8
            let need_mse = compute_all || has_weight(base + 9, 1); // position 9
            if need_ssim || need_edge || need_mse {
                active_channels.push((c, need_ssim, need_edge));
            }
        }
    }

    // Compute MSE for all active channels (no blur needed, just pixel differences)
    let n = width * height;
    let one_over_n = 1.0 / n as f64;
    for &(c, _, _) in &active_channels {
        mse_vals[c] = sq_diff_sum(&src[c][..n], &dst[c][..n]) * one_over_n;
    }

    // Use phased parallelism only for non-masking mode on large images
    if n >= PARALLEL_THRESHOLD && !masked {
        compute_single_scale_phased(
            src,
            dst,
            width,
            height,
            config.blur_radius,
            config.blur_passes,
            bufs,
            parallel_bufs,
            &active_channels,
            &mut ssim_vals,
            &mut edge_vals,
            &mut hf_energy_loss_vals,
            &mut hf_mag_loss_vals,
            &mut hf_energy_gain_vals,
            &mut ssim_2nd_vals,
            &mut edge_2nd_vals,
        );
    } else {
        // Sequential path (also used for masking since mask computation needs mu1)
        for &(c, need_ssim, need_edge) in &active_channels {
            let result = compute_channel(
                &src[c], &dst[c], width, height, config, need_ssim, need_edge, bufs,
            );
            store_channel_result(c, &result, &mut ssim_vals, &mut edge_vals);
            hf_energy_loss_vals[c] = result.hf_energy_loss;
            hf_mag_loss_vals[c] = result.hf_mag_loss;
            hf_energy_gain_vals[c] = result.hf_energy_gain;
            ssim_2nd_vals[c] = result.ssim_2nd;
            edge_2nd_vals[c * 2] = result.edge_2nd[0];
            edge_2nd_vals[c * 2 + 1] = result.edge_2nd[1];
        }
    }

    ScaleStats {
        ssim: ssim_vals,
        edge: edge_vals,
        mse: mse_vals,
        hf_energy_loss: hf_energy_loss_vals,
        hf_mag_loss: hf_mag_loss_vals,
        hf_energy_gain: hf_energy_gain_vals,
        ssim_2nd: ssim_2nd_vals,
        edge_2nd: edge_2nd_vals,
    }
}

fn store_channel_result(
    c: usize,
    result: &ChannelResult,
    ssim_vals: &mut [f64; 6],
    edge_vals: &mut [f64; 12],
) {
    ssim_vals[c * 2] = result.ssim[0];
    ssim_vals[c * 2 + 1] = result.ssim[1];
    edge_vals[c * 4] = result.edge[0];
    edge_vals[c * 4 + 1] = result.edge[1];
    edge_vals[c * 4 + 2] = result.edge[2];
    edge_vals[c * 4 + 3] = result.edge[3];
}

/// Compute high-frequency energy/magnitude features for a single channel.
/// Measures loss or gain of local detail by comparing (pixel - blur(pixel))
/// between source and distorted. mu1/mu2 must already contain blurred src/dst.
#[allow(clippy::too_many_arguments)]
fn compute_hf_features(
    src_c: &[f32],
    dst_c: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    one_over_n: f64,
    c: usize,
    hf_energy_loss_vals: &mut [f64; 3],
    hf_mag_loss_vals: &mut [f64; 3],
    hf_energy_gain_vals: &mut [f64; 3],
) {
    let n = (1.0 / one_over_n) as usize;
    let var_src = sq_diff_sum(&src_c[..n], &mu1[..n]) * one_over_n;
    let var_dst = sq_diff_sum(&dst_c[..n], &mu2[..n]) * one_over_n;
    hf_energy_loss_vals[c] = if var_src > 1e-10 {
        (1.0 - var_dst / var_src).max(0.0)
    } else {
        0.0
    };
    hf_energy_gain_vals[c] = if var_src > 1e-10 {
        (var_dst / var_src - 1.0).max(0.0)
    } else {
        0.0
    };

    let mad_src = abs_diff_sum(&src_c[..n], &mu1[..n]) * one_over_n;
    let mad_dst = abs_diff_sum(&dst_c[..n], &mu2[..n]) * one_over_n;
    hf_mag_loss_vals[c] = if mad_src > 1e-10 {
        (1.0 - mad_dst / mad_src).max(0.0)
    } else {
        0.0
    };
}

/// Compute SSIM (and optionally edge) for a single channel sequentially.
/// Used for additional SSIM channels beyond the first in the phased path.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn compute_ssim_channel_sequential(
    src_c: &[f32],
    dst_c: &[f32],
    width: usize,
    height: usize,
    blur_radius: usize,
    blur_fn: fn(&[f32], &mut [f32], &mut [f32], usize, usize, usize),
    need_edge: bool,
    one_over_n: f64,
    mu1: &mut [f32],
    mu2: &mut [f32],
    scratch: &mut [f32],
    sig_sq: &mut [f32],
    sig12: &mut [f32],
    temp: &mut [f32],
    c: usize,
    ssim_vals: &mut [f64; 6],
    edge_vals: &mut [f64; 12],
    hf_energy_loss_vals: &mut [f64; 3],
    hf_mag_loss_vals: &mut [f64; 3],
    hf_energy_gain_vals: &mut [f64; 3],
    ssim_2nd_vals: &mut [f64; 3],
    edge_2nd_vals: &mut [f64; 6],
) {
    // 4 sequential blurs for SSIM
    blur_fn(src_c, mu1, temp, width, height, blur_radius);
    blur_fn(dst_c, mu2, temp, width, height, blur_radius);
    sq_sum_into(src_c, dst_c, scratch);
    blur_fn(scratch, sig_sq, temp, width, height, blur_radius);
    mul_into(src_c, dst_c, scratch);
    blur_fn(scratch, sig12, temp, width, height, blur_radius);

    let (sum_d, sum_d4, sum_d2) = ssim_channel(mu1, mu2, sig_sq, sig12);
    ssim_vals[c * 2] = sum_d * one_over_n;
    ssim_vals[c * 2 + 1] = (sum_d4 * one_over_n).powf(0.25);
    ssim_2nd_vals[c] = (sum_d2 * one_over_n).sqrt();

    if need_edge {
        let (art, art4, det, det4, art2, det2) = edge_diff_channel(src_c, dst_c, mu1, mu2);
        edge_vals[c * 4] = art * one_over_n;
        edge_vals[c * 4 + 1] = (art4 * one_over_n).powf(0.25);
        edge_vals[c * 4 + 2] = det * one_over_n;
        edge_vals[c * 4 + 3] = (det4 * one_over_n).powf(0.25);
        edge_2nd_vals[c * 2] = (art2 * one_over_n).sqrt();
        edge_2nd_vals[c * 2 + 1] = (det2 * one_over_n).sqrt();
    }

    compute_hf_features(
        src_c,
        dst_c,
        mu1,
        mu2,
        one_over_n,
        c,
        hf_energy_loss_vals,
        hf_mag_loss_vals,
        hf_energy_gain_vals,
    );
}

/// Phased blur parallelism for large images.
///
/// Instead of parallelizing by channel (Y on thread 1, B on thread 2 — imbalanced),
/// we parallelize pairs of independent blur operations:
///
/// Phase 1: blur(src_ch0) || blur(dst_ch0) — 2 mu blurs in parallel
/// Phase 2: blur(sigma_sq) || blur(sigma12_or_ch1_src) — balanced work
/// Phase 3: blur(ch1_dst_if_needed) + all reductions
#[allow(clippy::too_many_arguments)]
fn compute_single_scale_phased(
    src: &[Vec<f32>; 3],
    dst: &[Vec<f32>; 3],
    width: usize,
    height: usize,
    blur_radius: usize,
    blur_passes: u8,
    bufs: &mut ScaleBuffers,
    parallel_bufs: &mut ScaleBuffers,
    active_channels: &[(usize, bool, bool)],
    ssim_vals: &mut [f64; 6],
    edge_vals: &mut [f64; 12],
    hf_energy_loss_vals: &mut [f64; 3],
    hf_mag_loss_vals: &mut [f64; 3],
    hf_energy_gain_vals: &mut [f64; 3],
    ssim_2nd_vals: &mut [f64; 3],
    edge_2nd_vals: &mut [f64; 6],
) {
    let n = width * height;
    let one_over_n = 1.0 / n as f64;

    #[allow(clippy::type_complexity)]
    let blur_fn: fn(&[f32], &mut [f32], &mut [f32], usize, usize, usize) = match blur_passes {
        1 => box_blur_1pass_into,
        2 => box_blur_2pass_into,
        _ => box_blur_3pass_into,
    };

    // Separate SSIM channels (heavy: 4 blurs) from edge-only (lighter: 2 blurs)
    let mut ssim_chs: Vec<(usize, bool)> = Vec::new(); // (channel_idx, need_edge)
    let mut edge_only_chs: Vec<usize> = Vec::new();

    for &(c, need_ssim, need_edge) in active_channels {
        if need_ssim {
            ssim_chs.push((c, need_edge));
        } else if need_edge {
            edge_only_chs.push(c);
        }
    }
    // Take the first SSIM channel for phased parallelism
    let ssim_ch = ssim_chs.first().copied();

    // Destructure bufs to allow split borrows across threads
    let ScaleBuffers {
        mul_buf: ref mut scratch1,
        mu1: ref mut mu1_a,
        mu2: ref mut mu2_a,
        sigma1_sq: ref mut sig_sq,
        sigma12: ref mut sig12,
        temp_blur: ref mut temp1,
        mask: _,
    } = *bufs;
    let ScaleBuffers {
        mul_buf: ref mut scratch2,
        mu1: ref mut mu1_b,
        mu2: ref mut mu2_b,
        sigma1_sq: ref mut temp3,
        sigma12: _,
        temp_blur: ref mut temp2,
        mask: _,
    } = *parallel_bufs;

    if let Some((sc, sc_need_edge)) = ssim_ch {
        // --- SSIM channel (4 blurs + reductions) ---

        if let Some(&edge_c) = edge_only_chs.first() {
            // 3-thread phased path: 2 phases instead of 3.
            // Phase 1 (3 threads): blur(src_Y) || blur(dst_Y) || blur(src_edge)
            std::thread::scope(|s| {
                s.spawn(|| blur_fn(&src[sc], mu1_a, temp1, width, height, blur_radius));
                s.spawn(|| blur_fn(&src[edge_c], mu1_b, temp3, width, height, blur_radius));
                blur_fn(&dst[sc], mu2_a, temp2, width, height, blur_radius);
            });

            // Phase 2 (3 threads): sq_sum+blur(sig_sq) || mul+blur(sig12) || blur(dst_edge)
            std::thread::scope(|s| {
                s.spawn(|| {
                    sq_sum_into(&src[sc], &dst[sc], scratch1);
                    blur_fn(scratch1, sig_sq, temp1, width, height, blur_radius);
                });
                s.spawn(|| blur_fn(&dst[edge_c], mu2_b, temp3, width, height, blur_radius));
                mul_into(&src[sc], &dst[sc], scratch2);
                blur_fn(scratch2, sig12, temp2, width, height, blur_radius);
            });

            // Phase 4: reductions (fast, sequential)
            let (sum_d, sum_d4, sum_d2) = ssim_channel(mu1_a, mu2_a, sig_sq, sig12);
            ssim_vals[sc * 2] = sum_d * one_over_n;
            ssim_vals[sc * 2 + 1] = (sum_d4 * one_over_n).powf(0.25);
            ssim_2nd_vals[sc] = (sum_d2 * one_over_n).sqrt();

            if sc_need_edge {
                let (art, art4, det, det4, art2, det2) =
                    edge_diff_channel(&src[sc], &dst[sc], mu1_a, mu2_a);
                edge_vals[sc * 4] = art * one_over_n;
                edge_vals[sc * 4 + 1] = (art4 * one_over_n).powf(0.25);
                edge_vals[sc * 4 + 2] = det * one_over_n;
                edge_vals[sc * 4 + 3] = (det4 * one_over_n).powf(0.25);
                edge_2nd_vals[sc * 2] = (art2 * one_over_n).sqrt();
                edge_2nd_vals[sc * 2 + 1] = (det2 * one_over_n).sqrt();
            }
            compute_hf_features(
                &src[sc],
                &dst[sc],
                mu1_a,
                mu2_a,
                one_over_n,
                sc,
                hf_energy_loss_vals,
                hf_mag_loss_vals,
                hf_energy_gain_vals,
            );

            let (art, art4, det, det4, art2, det2) =
                edge_diff_channel(&src[edge_c], &dst[edge_c], mu1_b, mu2_b);
            edge_vals[edge_c * 4] = art * one_over_n;
            edge_vals[edge_c * 4 + 1] = (art4 * one_over_n).powf(0.25);
            edge_vals[edge_c * 4 + 2] = det * one_over_n;
            edge_vals[edge_c * 4 + 3] = (det4 * one_over_n).powf(0.25);
            edge_2nd_vals[edge_c * 2] = (art2 * one_over_n).sqrt();
            edge_2nd_vals[edge_c * 2 + 1] = (det2 * one_over_n).sqrt();
            compute_hf_features(
                &src[edge_c],
                &dst[edge_c],
                mu1_b,
                mu2_b,
                one_over_n,
                edge_c,
                hf_energy_loss_vals,
                hf_mag_loss_vals,
                hf_energy_gain_vals,
            );

            // Handle additional edge-only channels (rare — usually only 1)
            for &edge_c2 in &edge_only_chs[1..] {
                blur_fn(&src[edge_c2], mu1_b, temp1, width, height, blur_radius);
                blur_fn(&dst[edge_c2], mu2_b, temp2, width, height, blur_radius);
                let (art, art4, det, det4, art2, det2) =
                    edge_diff_channel(&src[edge_c2], &dst[edge_c2], mu1_b, mu2_b);
                edge_vals[edge_c2 * 4] = art * one_over_n;
                edge_vals[edge_c2 * 4 + 1] = (art4 * one_over_n).powf(0.25);
                edge_vals[edge_c2 * 4 + 2] = det * one_over_n;
                edge_vals[edge_c2 * 4 + 3] = (det4 * one_over_n).powf(0.25);
                edge_2nd_vals[edge_c2 * 2] = (art2 * one_over_n).sqrt();
                edge_2nd_vals[edge_c2 * 2 + 1] = (det2 * one_over_n).sqrt();
                compute_hf_features(
                    &src[edge_c2],
                    &dst[edge_c2],
                    mu1_b,
                    mu2_b,
                    one_over_n,
                    edge_c2,
                    hf_energy_loss_vals,
                    hf_mag_loss_vals,
                    hf_energy_gain_vals,
                );
            }

            // Handle additional SSIM channels sequentially (reuse buffers)
            for &(extra_c, extra_edge) in &ssim_chs[1..] {
                compute_ssim_channel_sequential(
                    &src[extra_c],
                    &dst[extra_c],
                    width,
                    height,
                    blur_radius,
                    blur_fn,
                    extra_edge,
                    one_over_n,
                    mu1_b,
                    mu2_b,
                    scratch1,
                    sig_sq,
                    sig12,
                    temp1,
                    extra_c,
                    ssim_vals,
                    edge_vals,
                    hf_energy_loss_vals,
                    hf_mag_loss_vals,
                    hf_energy_gain_vals,
                    ssim_2nd_vals,
                    edge_2nd_vals,
                );
            }
        } else {
            // No edge-only channels — parallel mu blurs then parallel sigma blurs
            std::thread::scope(|s| {
                s.spawn(|| blur_fn(&src[sc], mu1_a, temp1, width, height, blur_radius));
                blur_fn(&dst[sc], mu2_a, temp2, width, height, blur_radius);
            });

            std::thread::scope(|s| {
                s.spawn(|| {
                    sq_sum_into(&src[sc], &dst[sc], scratch1);
                    blur_fn(scratch1, sig_sq, temp1, width, height, blur_radius);
                });
                mul_into(&src[sc], &dst[sc], scratch2);
                blur_fn(scratch2, sig12, temp2, width, height, blur_radius);
            });

            let (sum_d, sum_d4, sum_d2) = ssim_channel(mu1_a, mu2_a, sig_sq, sig12);
            ssim_vals[sc * 2] = sum_d * one_over_n;
            ssim_vals[sc * 2 + 1] = (sum_d4 * one_over_n).powf(0.25);
            ssim_2nd_vals[sc] = (sum_d2 * one_over_n).sqrt();

            if sc_need_edge {
                let (art, art4, det, det4, art2, det2) =
                    edge_diff_channel(&src[sc], &dst[sc], mu1_a, mu2_a);
                edge_vals[sc * 4] = art * one_over_n;
                edge_vals[sc * 4 + 1] = (art4 * one_over_n).powf(0.25);
                edge_vals[sc * 4 + 2] = det * one_over_n;
                edge_vals[sc * 4 + 3] = (det4 * one_over_n).powf(0.25);
                edge_2nd_vals[sc * 2] = (art2 * one_over_n).sqrt();
                edge_2nd_vals[sc * 2 + 1] = (det2 * one_over_n).sqrt();
            }
            compute_hf_features(
                &src[sc],
                &dst[sc],
                mu1_a,
                mu2_a,
                one_over_n,
                sc,
                hf_energy_loss_vals,
                hf_mag_loss_vals,
                hf_energy_gain_vals,
            );

            // Handle additional SSIM channels sequentially
            for &(extra_c, extra_edge) in &ssim_chs[1..] {
                compute_ssim_channel_sequential(
                    &src[extra_c],
                    &dst[extra_c],
                    width,
                    height,
                    blur_radius,
                    blur_fn,
                    extra_edge,
                    one_over_n,
                    mu1_a,
                    mu2_a,
                    scratch1,
                    sig_sq,
                    sig12,
                    temp1,
                    extra_c,
                    ssim_vals,
                    edge_vals,
                    hf_energy_loss_vals,
                    hf_mag_loss_vals,
                    hf_energy_gain_vals,
                    ssim_2nd_vals,
                    edge_2nd_vals,
                );
            }
        }
    } else {
        // No SSIM channels — just edge-only (process sequentially, they're light)
        for &edge_c in &edge_only_chs {
            std::thread::scope(|s| {
                s.spawn(|| blur_fn(&src[edge_c], mu1_a, temp1, width, height, blur_radius));
                blur_fn(&dst[edge_c], mu2_a, temp2, width, height, blur_radius);
            });
            let (art, art4, det, det4, art2, det2) =
                edge_diff_channel(&src[edge_c], &dst[edge_c], mu1_a, mu2_a);
            edge_vals[edge_c * 4] = art * one_over_n;
            edge_vals[edge_c * 4 + 1] = (art4 * one_over_n).powf(0.25);
            edge_vals[edge_c * 4 + 2] = det * one_over_n;
            edge_vals[edge_c * 4 + 3] = (det4 * one_over_n).powf(0.25);
            edge_2nd_vals[edge_c * 2] = (art2 * one_over_n).sqrt();
            edge_2nd_vals[edge_c * 2 + 1] = (det2 * one_over_n).sqrt();
            compute_hf_features(
                &src[edge_c],
                &dst[edge_c],
                mu1_a,
                mu2_a,
                one_over_n,
                edge_c,
                hf_energy_loss_vals,
                hf_mag_loss_vals,
                hf_energy_gain_vals,
            );
        }
    }
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
/// Total number of features per scale (3 channels × 13 features = 39)
#[cfg_attr(not(feature = "training"), allow(dead_code))]
pub const FEATURES_PER_SCALE: usize = 39;

/// Trained weights from v2 synthetic dataset (163k image pairs, 149.5k valid).
///
/// Optimized against GPU-accelerated SSIM2 scores on a diverse synthetic dataset
/// (4 codecs × 11 quality levels × 6 aspect ratios × 276 source images).
/// SROCC = 0.9857 on the training set.
///
/// Layout: 4 scales × 3 channels (X,Y,B) × 13 features:
///   ssim_mean, ssim_4th, ssim_2nd, art_mean, art_4th, art_2nd,
///   det_mean, det_4th, det_2nd, mse, var_loss, tex_loss, contrast_inc
#[allow(clippy::excessive_precision)]
pub const WEIGHTS: [f64; 156] = [
    0.0000000000,
    0.3054918030,
    0.0000000000,
    0.0120060829,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0025082274,
    0.0158347215,
    1.0232620956,
    0.0232095092,
    0.0157660229,
    0.0112574353, // Scale 0 Channel X
    2.6230901445,
    5.0412606075,
    0.0223917844,
    0.0098804550,
    0.0000000000,
    0.0038950338,
    0.0156205026,
    4.1215632793,
    0.0165163863,
    0.0000000000,
    0.6505034669,
    1.0661232328,
    0.2024220909, // Scale 0 Channel Y
    0.0000000000,
    0.0224667817,
    0.0214887709,
    8.6744486523,
    0.0212844299,
    0.0139801711,
    0.0000000000,
    0.0000000000,
    0.0167307326,
    1.0988427328,
    0.0000000000,
    0.0000000000,
    0.0147283435, // Scale 0 Channel B
    0.0056672813,
    0.3216560111,
    0.0000000000,
    0.0000000000,
    0.0013174343,
    0.0009618480,
    0.0235250311,
    0.0000000000,
    0.0233925065,
    51.1749644772,
    0.0240114888,
    0.0000000000,
    0.0000002541, // Scale 1 Channel X
    56.0466844974,
    13.7089560010,
    0.9639517783,
    11.6058052403,
    21.6136075666,
    0.0066979598,
    0.0000000000,
    1.8412885819,
    0.0000000000,
    0.0000000000,
    1.1198246434,
    0.0233930872,
    0.0010262427, // Scale 1 Channel Y
    1.7751266787,
    1.3741364948,
    0.0000000000,
    51.6748592180,
    1.9402960207,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0105201745,
    276.1886791157,
    0.0000000000,
    0.0000000000,
    0.0230782705, // Scale 1 Channel B
    0.5138116129,
    2.9679983985,
    0.0168215442,
    0.0000000000,
    9.4726552791,
    0.0227755446,
    0.0251548240,
    0.0231763042,
    0.0250751373,
    256.0304873419,
    0.0251168295,
    0.0000000000,
    0.0075593685, // Scale 2 Channel X
    17.1521567314,
    13.6143902459,
    15.2230380318,
    281.6839646804,
    67.7830673948,
    0.0224871201,
    0.0005901792,
    19.7594335859,
    0.0000000000,
    0.0000000000,
    2.4182112031,
    2.1322464656,
    0.0007368699, // Scale 2 Channel Y
    1.4831017806,
    0.0000000000,
    0.0000000000,
    38.8427093881,
    8.5955250320,
    2.7105025533,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0223146216,
    0.0000000000,
    0.0098977948,
    0.0000317606, // Scale 2 Channel B
    4.1642190683,
    13.5967841625,
    0.0000000000,
    0.0075158970,
    14.7858065617,
    0.0102946185,
    0.0047139742,
    0.0057291302,
    0.0000000000,
    1.0257943538,
    0.0146241140,
    0.3198369698,
    0.0000000000, // Scale 3 Channel X
    0.0582460099,
    0.0195498754,
    0.0114503725,
    869.4011573597,
    0.0001269371,
    0.3471799558,
    415.8779165902,
    82.2604764730,
    28.2928124905,
    0.2216564412,
    8.4103544255,
    0.0109363789,
    0.0151401628, // Scale 3 Channel Y
    8.2615006567,
    0.0069805260,
    0.0507613523,
    374.9053338534,
    69.6581630071,
    0.0150326812,
    0.0201076132,
    0.0246660472,
    0.0083943755,
    1.1493739991,
    0.0154112867,
    0.1279347879,
    0.0124929133, // Scale 3 Channel B
];

pub(crate) fn combine_scores(
    scale_stats: &[ScaleStats],
    _masked: bool,
    weights: &[f64; 156],
    config: &ZensimConfig,
) -> ZensimResult {
    let features_per_ch = FEATURES_PER_CHANNEL_BASIC;
    let features_per_scale = features_per_ch * 3;

    let mut features = Vec::with_capacity(scale_stats.len() * features_per_scale);
    let mut raw_distance = 0.0f64;

    for ss in scale_stats.iter() {
        for c in 0..3 {
            // .abs() is defensive — all these values are non-negative by construction
            // (SSIM errors are clamped ≥0, edge features use max(0,...), etc.)
            // but abs ensures no -0.0 or floating-point edge cases affect training.

            // ssim_mean, ssim_4th, ssim_2nd
            features.push(ss.ssim[c * 2].abs());
            features.push(ss.ssim[c * 2 + 1].abs());
            features.push(ss.ssim_2nd[c].abs());
            // art_mean, art_4th, art_2nd
            features.push(ss.edge[c * 4].abs());
            features.push(ss.edge[c * 4 + 1].abs());
            features.push(ss.edge_2nd[c * 2].abs());
            // det_mean, det_4th, det_2nd
            features.push(ss.edge[c * 4 + 2].abs());
            features.push(ss.edge[c * 4 + 3].abs());
            features.push(ss.edge_2nd[c * 2 + 1].abs());
            // mse, hf_energy_loss, hf_mag_loss, hf_energy_gain (also non-negative)
            features.push(ss.mse[c]);
            features.push(ss.hf_energy_loss[c]);
            features.push(ss.hf_mag_loss[c]);
            features.push(ss.hf_energy_gain[c]);
        }
    }

    // Apply weights — only up to weights.len(), extra features get weight 0
    for (i, &feat) in features.iter().enumerate() {
        if i < weights.len() {
            raw_distance += feat * weights[i];
        }
    }

    // Normalize by number of scales
    raw_distance /= scale_stats.len().max(1) as f64;

    let score =
        distance_to_score_mapped(raw_distance, config.score_mapping_a, config.score_mapping_b);

    ZensimResult {
        score,
        raw_distance,
        features,
        profile: ZensimProfile::GeneralV0_2,
    }
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

        let default_result = compute_zensim(&src, &dst, w, h).unwrap();
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

        // compute_all should have all features populated (nonzero for most)
        assert_eq!(all_result.features.len(), default_result.features.len());
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
}
