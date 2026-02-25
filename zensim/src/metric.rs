//! Core zensim metric computation.
//!
//! Multi-scale SSIM + edge features in XYB color space,
//! with contrast sensitivity weighting per scale.

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
#[derive(Debug, Clone, Copy)]
pub struct ZensimConfig {
    /// Box blur radius at scale 0 (default: 5, giving 11-pixel kernel)
    pub blur_radius: usize,
    /// Number of box blur passes (1, 2, or 3, default: 1).
    /// 1 pass = rectangular kernel, fastest. 2 = triangular. 3 = piecewise-quadratic ≈ Gaussian.
    pub blur_passes: u8,
    /// Compute all features even if their weights are zero.
    /// Enable this for weight training to avoid circular dependency.
    pub compute_all_features: bool,
    /// Local contrast masking strength (default: 0.0 = disabled).
    /// When > 0, computes local activity sigma and weights per-pixel distances by
    /// 1 / (1 + masking_strength * sigma). Higher values mask textured regions more.
    /// Typical range: 2.0-8.0.
    pub masking_strength: f32,
    /// Maximum number of downscale levels (default: 4).
    /// More scales capture larger structures but add features. Range: 2-6.
    pub num_scales: usize,
}

impl Default for ZensimConfig {
    fn default() -> Self {
        Self {
            blur_radius: 5,
            blur_passes: 1,
            compute_all_features: false,
            masking_strength: 0.0,
            num_scales: crate::NUM_SCALES,
        }
    }
}

/// Map raw weighted distance to 0-100 quality score.
pub fn distance_to_score(raw_distance: f64) -> f64 {
    if raw_distance <= 0.0 {
        100.0
    } else {
        // Power law mapping: score = 100 - scale * d^gamma
        // Calibrated for trained weights where d ∈ [0, ~10]:
        //   d=0.1 → ~97 (near-invisible distortion)
        //   d=0.5 → ~82 (subtle distortion)
        //   d=2.5 → ~52 (moderate distortion)
        //   d=8.0 → ~18 (severe distortion)
        //   d=16  → ~3  (heavy distortion)
        let scale = 18.0;
        let gamma = 0.7;
        (100.0 - scale * raw_distance.powf(gamma)).max(0.0)
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

/// Per-scale statistics collected during computation.
pub(crate) struct ScaleStats {
    /// SSIM statistics: [mean_d, root4_d] per channel = 6 values
    pub(crate) ssim: [f64; 6],
    /// Edge features: [art_mean, art_4th, det_mean, det_4th] per channel = 12 values
    pub(crate) edge: [f64; 12],
    /// Per-channel MSE: mean((src - dst)²) for X, Y, B
    pub(crate) mse: [f64; 3],
    /// Variance loss (L2): max(0, 1 - var_dst / var_src) per channel
    pub(crate) variance_loss: [f64; 3],
    /// Texture loss (L1): max(0, 1 - mad_dst / mad_src) per channel
    pub(crate) texture_loss: [f64; 3],
    /// 2nd-power pooled SSIM: [root2_d] per channel = 3 values
    pub(crate) ssim_2nd: [f64; 3],
    /// Edge 2nd power: [art_2nd, det_2nd] per channel = 6 values
    pub(crate) edge_2nd: [f64; 6],
    /// Contrast increase: max(0, var_dst/var_src - 1) per channel
    pub(crate) contrast_increase: [f64; 3],
}

/// Result from zensim comparison.
#[derive(Debug, Clone)]
pub struct ZensimResult {
    /// Score on 0-100 scale. 100 = identical, higher = better.
    pub score: f64,
    /// Raw weighted distance (before nonlinear mapping). Lower = more similar.
    pub raw_distance: f64,
    /// Per-scale raw features for weight training.
    /// 13 features per channel × 3 channels = 39 per scale:
    ///   ssim_mean, ssim_4th, ssim_2nd, edge_art_mean, edge_art_4th, edge_art_2nd,
    ///   edge_det_mean, edge_det_4th, edge_det_2nd, mse, variance_loss, texture_loss,
    ///   contrast_increase
    pub features: Vec<f64>,
}

/// Features per channel: 13 features always emitted.
/// ssim_mean, ssim_4th, ssim_2nd, art_mean, art_4th, art_2nd,
/// det_mean, det_4th, det_2nd, mse, variance_loss, texture_loss, contrast_increase
pub(crate) const FEATURES_PER_CHANNEL_BASIC: usize = 13;

/// Compute zensim score between two sRGB u8 images.
///
/// Returns a score on 0-100 scale where 100 = identical.
pub fn compute_zensim(
    source: &[[u8; 3]],
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
) -> Result<ZensimResult, ZensimError> {
    compute_zensim_with_config(source, distorted, width, height, ZensimConfig::default())
}

/// Compute zensim with custom configuration.
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

    // Use streaming path for large non-masked images to reduce peak memory
    let masked = config.masking_strength > 0.0;
    if !masked && crate::streaming::should_use_streaming(width, height) {
        let result =
            crate::streaming::compute_zensim_streaming(source, distorted, width, height, &config);
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
    let result = combine_scores(&scale_stats, masked);
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
    ssim: [f64; 2],        // [mean_d, root4_d]
    edge: [f64; 4],        // [art_mean, art_4th, det_mean, det_4th]
    variance_loss: f64,    // max(0, 1 - var_dst / var_src)
    texture_loss: f64,     // max(0, 1 - mad_dst / mad_src)
    contrast_increase: f64, // max(0, var_dst/var_src - 1)
    ssim_2nd: f64,         // root2 pooled SSIM
    edge_2nd: [f64; 2],   // [art_2nd, det_2nd]
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
            let (art, art4, det, det4, art2, det2) = edge_diff_channel(src_c, dst_c, &bufs.mu1, &bufs.mu2);
            edge[0] = art * one_over_n;
            edge[1] = (art4 * one_over_n).powf(0.25);
            edge[2] = det * one_over_n;
            edge[3] = (det4 * one_over_n).powf(0.25);
            edge_2nd[0] = (art2 * one_over_n).sqrt();
            edge_2nd[1] = (det2 * one_over_n).sqrt();
        }
    }

    // Variance loss (L2): 1 - var_dst / var_src
    let var_src = sq_diff_sum(&src_c[..n], &bufs.mu1[..n]) * one_over_n;
    let var_dst = sq_diff_sum(&dst_c[..n], &bufs.mu2[..n]) * one_over_n;
    let variance_loss = if var_src > 1e-10 {
        (1.0 - var_dst / var_src).max(0.0)
    } else {
        0.0
    };

    // Contrast increase: max(0, var_dst/var_src - 1)
    let contrast_increase = if var_src > 1e-10 {
        (var_dst / var_src - 1.0).max(0.0)
    } else {
        0.0
    };

    // Texture loss (L1): 1 - mad_dst / mad_src
    let mad_src = abs_diff_sum(&src_c[..n], &bufs.mu1[..n]) * one_over_n;
    let mad_dst = abs_diff_sum(&dst_c[..n], &bufs.mu2[..n]) * one_over_n;
    let texture_loss = if mad_src > 1e-10 {
        (1.0 - mad_dst / mad_src).max(0.0)
    } else {
        0.0
    };

    ChannelResult {
        ssim,
        edge,
        variance_loss,
        texture_loss,
        contrast_increase,
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
    let mut variance_loss_vals = [0.0f64; 3];
    let mut texture_loss_vals = [0.0f64; 3];
    let mut contrast_increase_vals = [0.0f64; 3];
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
    //   mse(9), variance_loss(10), texture_loss(11), contrast_increase(12)
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
            let need_blur_features = has_weight(base + 10, 3); // positions 10-12
            // Blur features need mu1/mu2 (same as edge), fold into need_edge
            let need_edge = compute_all || has_weight(base + 3, 6) || need_blur_features; // positions 3-8
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
            &mut variance_loss_vals,
            &mut texture_loss_vals,
            &mut contrast_increase_vals,
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
            variance_loss_vals[c] = result.variance_loss;
            texture_loss_vals[c] = result.texture_loss;
            contrast_increase_vals[c] = result.contrast_increase;
            ssim_2nd_vals[c] = result.ssim_2nd;
            edge_2nd_vals[c * 2] = result.edge_2nd[0];
            edge_2nd_vals[c * 2 + 1] = result.edge_2nd[1];
        }
    }

    ScaleStats {
        ssim: ssim_vals,
        edge: edge_vals,
        mse: mse_vals,
        variance_loss: variance_loss_vals,
        texture_loss: texture_loss_vals,
        contrast_increase: contrast_increase_vals,
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

/// Compute variance loss, texture loss, and contrast increase for a single channel.
/// mu1/mu2 must already contain blurred src/dst for this channel.
#[allow(clippy::too_many_arguments)]
fn compute_blur_features(
    src_c: &[f32],
    dst_c: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    one_over_n: f64,
    c: usize,
    variance_loss_vals: &mut [f64; 3],
    texture_loss_vals: &mut [f64; 3],
    contrast_increase_vals: &mut [f64; 3],
) {
    let n = (1.0 / one_over_n) as usize;
    let var_src = sq_diff_sum(&src_c[..n], &mu1[..n]) * one_over_n;
    let var_dst = sq_diff_sum(&dst_c[..n], &mu2[..n]) * one_over_n;
    variance_loss_vals[c] = if var_src > 1e-10 {
        (1.0 - var_dst / var_src).max(0.0)
    } else {
        0.0
    };
    contrast_increase_vals[c] = if var_src > 1e-10 {
        (var_dst / var_src - 1.0).max(0.0)
    } else {
        0.0
    };

    let mad_src = abs_diff_sum(&src_c[..n], &mu1[..n]) * one_over_n;
    let mad_dst = abs_diff_sum(&dst_c[..n], &mu2[..n]) * one_over_n;
    texture_loss_vals[c] = if mad_src > 1e-10 {
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
    variance_loss_vals: &mut [f64; 3],
    texture_loss_vals: &mut [f64; 3],
    contrast_increase_vals: &mut [f64; 3],
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

    compute_blur_features(
        src_c,
        dst_c,
        mu1,
        mu2,
        one_over_n,
        c,
        variance_loss_vals,
        texture_loss_vals,
        contrast_increase_vals,
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
    variance_loss_vals: &mut [f64; 3],
    texture_loss_vals: &mut [f64; 3],
    contrast_increase_vals: &mut [f64; 3],
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
                let (art, art4, det, det4, art2, det2) = edge_diff_channel(&src[sc], &dst[sc], mu1_a, mu2_a);
                edge_vals[sc * 4] = art * one_over_n;
                edge_vals[sc * 4 + 1] = (art4 * one_over_n).powf(0.25);
                edge_vals[sc * 4 + 2] = det * one_over_n;
                edge_vals[sc * 4 + 3] = (det4 * one_over_n).powf(0.25);
                edge_2nd_vals[sc * 2] = (art2 * one_over_n).sqrt();
                edge_2nd_vals[sc * 2 + 1] = (det2 * one_over_n).sqrt();
            }
            compute_blur_features(
                &src[sc],
                &dst[sc],
                mu1_a,
                mu2_a,
                one_over_n,
                sc,
                variance_loss_vals,
                texture_loss_vals,
                contrast_increase_vals,
            );

            let (art, art4, det, det4, art2, det2) =
                edge_diff_channel(&src[edge_c], &dst[edge_c], mu1_b, mu2_b);
            edge_vals[edge_c * 4] = art * one_over_n;
            edge_vals[edge_c * 4 + 1] = (art4 * one_over_n).powf(0.25);
            edge_vals[edge_c * 4 + 2] = det * one_over_n;
            edge_vals[edge_c * 4 + 3] = (det4 * one_over_n).powf(0.25);
            edge_2nd_vals[edge_c * 2] = (art2 * one_over_n).sqrt();
            edge_2nd_vals[edge_c * 2 + 1] = (det2 * one_over_n).sqrt();
            compute_blur_features(
                &src[edge_c],
                &dst[edge_c],
                mu1_b,
                mu2_b,
                one_over_n,
                edge_c,
                variance_loss_vals,
                texture_loss_vals,
                contrast_increase_vals,
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
                compute_blur_features(
                    &src[edge_c2],
                    &dst[edge_c2],
                    mu1_b,
                    mu2_b,
                    one_over_n,
                    edge_c2,
                    variance_loss_vals,
                    texture_loss_vals,
                    contrast_increase_vals,
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
                    variance_loss_vals,
                    texture_loss_vals,
                    contrast_increase_vals,
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
                let (art, art4, det, det4, art2, det2) = edge_diff_channel(&src[sc], &dst[sc], mu1_a, mu2_a);
                edge_vals[sc * 4] = art * one_over_n;
                edge_vals[sc * 4 + 1] = (art4 * one_over_n).powf(0.25);
                edge_vals[sc * 4 + 2] = det * one_over_n;
                edge_vals[sc * 4 + 3] = (det4 * one_over_n).powf(0.25);
                edge_2nd_vals[sc * 2] = (art2 * one_over_n).sqrt();
                edge_2nd_vals[sc * 2 + 1] = (det2 * one_over_n).sqrt();
            }
            compute_blur_features(
                &src[sc],
                &dst[sc],
                mu1_a,
                mu2_a,
                one_over_n,
                sc,
                variance_loss_vals,
                texture_loss_vals,
                contrast_increase_vals,
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
                    variance_loss_vals,
                    texture_loss_vals,
                    contrast_increase_vals,
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
            compute_blur_features(
                &src[edge_c],
                &dst[edge_c],
                mu1_a,
                mu2_a,
                one_over_n,
                edge_c,
                variance_loss_vals,
                texture_loss_vals,
                contrast_increase_vals,
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
/// These weights are initial values, to be optimized against human ratings.
/// Total number of features per scale (3 channels × 13 features = 39)
#[cfg_attr(not(feature = "training"), allow(dead_code))]
pub const FEATURES_PER_SCALE: usize = 39;

/// Trained weights from TID2013 optimization (3000 pairs).
/// Layout: 4 scales × 3 channels (X,Y,B) × 13 features (ssim_mean, ssim_4th, ssim_2nd,
///         edge_art_mean, edge_art_4th, edge_art_2nd, edge_det_mean, edge_det_4th, edge_det_2nd,
///         mse, variance_loss, texture_loss, contrast_increase)
/// New features (ssim_2nd, art_2nd, det_2nd, contrast_increase) start at zero weight.
#[allow(clippy::excessive_precision)]
pub const WEIGHTS: [f64; 156] = [
    // Scale 0 Channel X (13 features: ssim_mean, ssim_4th, ssim_2nd, art_mean, art_4th, art_2nd, det_mean, det_4th, det_2nd, mse, var_loss, tex_loss, contrast_inc)
    0.000000, 0.000000, 0.000000, 0.000000, 7.401262, 0.000000, 0.016026, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    // Scale 0 Channel Y
    7.011122, 8.391644, 0.375298, 0.000000, 15.370380, 0.000000, 30.384206, 28.121342, 0.779064, 0.000000, 5.627197, 2.054596, 0.000000,
    // Scale 0 Channel B
    0.000000, 0.000000, 0.000000, 0.000000, 21.938411, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    // Scale 1 Channel X
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.007212, 0.000000, 0.000000, 0.500001, 0.000000,
    // Scale 1 Channel Y
    22.740950, 15.800330, 3.894948, 5.885507, 0.000000, 9.946805, 0.000000, 18.979639, 11.218666, 0.000000, 0.500585, 0.000433, 0.000000,
    // Scale 1 Channel B
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.488601, 0.000000, 0.000000, 0.000000, 1.000000, 2.000001, 0.000000,
    // Scale 2 Channel X
    0.981918, 0.000000, 0.451594, 0.000000, 0.000000, 0.000000, 103.157958, 0.003606, 0.000000, 88.528548, 2.000000, 0.000000, 0.000000,
    // Scale 2 Channel Y
    0.000000, 2.944499, 0.000000, 0.000000, 0.000000, 32.418322, 0.000000, 0.000000, 0.000000, 0.000000, 0.000601, 0.000000, 0.000000,
    // Scale 2 Channel B
    17.770518, 0.000000, 0.380406, 56.722823, 0.009615, 0.018029, 65.149301, 0.000000, 8.388175, 0.000000, 1.000000, 0.108173, 0.000000,
    // Scale 3 Channel X
    17.800292, 7.734777, 1.568173, 0.000000, 0.000000, 1.201923, 0.000000, 0.000000, 0.000000, 11.872030, 0.500000, 0.000000, 0.000000,
    // Scale 3 Channel Y
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.001282, 0.000000, 0.005409, 0.000000, 0.000000, 0.000000,
    // Scale 3 Channel B
    7.391297, 3.614260, 0.182739, 158.737549, 0.000000, 28.485079, 0.000000, 0.000000, 0.000000, 0.004056, 0.192308, 0.000000, 0.004808,
];

pub(crate) fn combine_scores(scale_stats: &[ScaleStats], _masked: bool) -> ZensimResult {
    let features_per_ch = FEATURES_PER_CHANNEL_BASIC;
    let features_per_scale = features_per_ch * 3;

    let mut features = Vec::with_capacity(scale_stats.len() * features_per_scale);
    let mut raw_distance = 0.0f64;

    for ss in scale_stats.iter() {
        for c in 0..3 {
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
            // mse, variance_loss, texture_loss, contrast_increase
            features.push(ss.mse[c]);
            features.push(ss.variance_loss[c]);
            features.push(ss.texture_loss[c]);
            features.push(ss.contrast_increase[c]);
        }
    }

    // Apply weights — only up to WEIGHTS.len(), extra features get weight 0
    for (i, &feat) in features.iter().enumerate() {
        if i < WEIGHTS.len() {
            raw_distance += feat * WEIGHTS[i];
        }
    }

    // Normalize by number of scales
    raw_distance /= scale_stats.len().max(1) as f64;

    let score = distance_to_score(raw_distance);

    ZensimResult {
        score,
        raw_distance,
        features,
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
