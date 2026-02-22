//! Core zensim metric computation.
//!
//! Multi-scale SSIM + edge features in XYB color space,
//! with contrast sensitivity weighting per scale.

use crate::NUM_SCALES;
use crate::blur::{
    box_blur_1pass_into, box_blur_2pass_into, box_blur_3pass_into, downscale_2x_inplace,
};
use crate::color::srgb_to_positive_xyb_planar;
use crate::error::ZensimError;
use crate::pool::ScaleBuffers;
use crate::simd_ops::{edge_diff_channel, mul_into, sq_sum_into, ssim_channel};

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
}

impl Default for ZensimConfig {
    fn default() -> Self {
        Self {
            blur_radius: 5,
            blur_passes: 1,
            compute_all_features: false,
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
    let raw_distance = raw_distance / (features.len() as f64 / FEATURES_PER_SCALE as f64).max(1.0);
    (distance_to_score(raw_distance), raw_distance)
}

/// Per-scale statistics collected during computation.
struct ScaleStats {
    /// SSIM statistics: [mean_d, root4_d] per channel = 6 values
    ssim: [f64; 6],
    /// Edge features: [art_mean, art_4th, det_mean, det_4th] per channel = 12 values
    edge: [f64; 12],
}

/// Result from zensim comparison.
#[derive(Debug, Clone)]
pub struct ZensimResult {
    /// Score on 0-100 scale. 100 = identical, higher = better.
    pub score: f64,
    /// Raw weighted distance (before nonlinear mapping). Lower = more similar.
    pub raw_distance: f64,
    /// Per-scale raw features for weight training. Layout:
    /// For each scale: [ssim_x_mean, ssim_x_4th, ssim_y_mean, ssim_y_4th, ssim_b_mean, ssim_b_4th,
    ///                  edge_x_art_mean, edge_x_art_4th, edge_x_det_mean, edge_x_det_4th,
    ///                  edge_y_art_mean, edge_y_art_4th, edge_y_det_mean, edge_y_det_4th,
    ///                  edge_b_art_mean, edge_b_art_4th, edge_b_det_mean, edge_b_det_4th]
    /// = 18 features per scale × NUM_SCALES
    pub features: Vec<f64>,
}

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

    // Convert both images to planar positive XYB in parallel
    let (src_xyb, dst_xyb) = std::thread::scope(|s| {
        let src_handle = s.spawn(|| srgb_to_positive_xyb_planar(source));
        let dst_xyb = srgb_to_positive_xyb_planar(distorted);
        let src_xyb = src_handle.join().unwrap();
        (src_xyb, dst_xyb)
    });

    // Compute multi-scale statistics (take ownership to avoid clone)
    let scale_stats = compute_multiscale_stats(
        src_xyb,
        dst_xyb,
        width,
        height,
        config.blur_radius,
        config.blur_passes,
        config.compute_all_features,
    );

    // Combine with weights to produce final score
    let result = combine_scores(&scale_stats);
    Ok(result)
}

/// Compute per-scale SSIM and edge statistics.
fn compute_multiscale_stats(
    src_xyb: [Vec<f32>; 3],
    dst_xyb: [Vec<f32>; 3],
    width: usize,
    height: usize,
    blur_radius: usize,
    blur_passes: u8,
    compute_all: bool,
) -> Vec<ScaleStats> {
    let mut stats = Vec::with_capacity(NUM_SCALES);

    let mut src_planes = src_xyb;
    let mut dst_planes = dst_xyb;
    let mut w = width;
    let mut h = height;

    // Pre-allocate two buffer sets: one for main thread, one for parallel thread.
    let max_n = width * height;
    let mut bufs = ScaleBuffers::new(max_n);
    let mut parallel_bufs = ScaleBuffers::new(max_n);

    for scale in 0..NUM_SCALES {
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
            blur_radius,
            blur_passes,
            &mut bufs,
            &mut parallel_bufs,
            scale,
            compute_all,
        );
        stats.push(scale_stat);

        // Downscale for next level (in-place, no allocations)
        if scale < NUM_SCALES - 1 {
            let mut nw = 0;
            let mut nh = 0;
            for c in 0..3 {
                let (sw, sh) = downscale_2x_inplace(&mut src_planes[c], w, h);
                let _ = downscale_2x_inplace(&mut dst_planes[c], w, h);
                nw = sw;
                nh = sh;
            }
            w = nw;
            h = nh;
        }
    }

    stats
}

/// Per-channel result from compute_channel.
struct ChannelResult {
    ssim: [f64; 2], // [mean_d, root4_d]
    edge: [f64; 4], // [art_mean, art_4th, det_mean, det_4th]
}

/// Compute SSIM and/or edge features for a single channel.
/// Self-contained: allocates its own buffers to enable parallel execution.
#[allow(clippy::too_many_arguments)]
fn compute_channel(
    src_c: &[f32],
    dst_c: &[f32],
    width: usize,
    height: usize,
    blur_radius: usize,
    blur_passes: u8,
    need_ssim: bool,
    need_edge: bool,
    bufs: &mut ScaleBuffers,
) -> ChannelResult {
    let n = width * height;
    let one_over_n = 1.0 / n as f64;
    let mut ssim = [0.0f64; 2];
    let mut edge = [0.0f64; 4];

    #[allow(clippy::type_complexity)]
    let blur_fn: fn(&[f32], &mut [f32], &mut [f32], usize, usize, usize) = match blur_passes {
        1 => box_blur_1pass_into,
        2 => box_blur_2pass_into,
        _ => box_blur_3pass_into,
    };

    // mu1 and mu2 are needed for both SSIM and edge features
    blur_fn(
        src_c,
        &mut bufs.mu1,
        &mut bufs.temp_blur,
        width,
        height,
        blur_radius,
    );
    blur_fn(
        dst_c,
        &mut bufs.mu2,
        &mut bufs.temp_blur,
        width,
        height,
        blur_radius,
    );

    if need_ssim {
        // Compute blur(src² + dst²) — combined saves one blur vs separate sigma1_sq, sigma2_sq
        sq_sum_into(src_c, dst_c, &mut bufs.mul_buf);
        blur_fn(
            &bufs.mul_buf,
            &mut bufs.sigma1_sq,
            &mut bufs.temp_blur,
            width,
            height,
            blur_radius,
        );

        mul_into(src_c, dst_c, &mut bufs.mul_buf);
        blur_fn(
            &bufs.mul_buf,
            &mut bufs.sigma12,
            &mut bufs.temp_blur,
            width,
            height,
            blur_radius,
        );

        // sigma1_sq now holds blur(src² + dst²), ssim_channel uses combined formula
        let (sum_d, sum_d4) = ssim_channel(&bufs.mu1, &bufs.mu2, &bufs.sigma1_sq, &bufs.sigma12);
        ssim[0] = sum_d * one_over_n;
        ssim[1] = (sum_d4 * one_over_n).powf(0.25);
    }

    if need_edge {
        let (art, art4, det, det4) = edge_diff_channel(src_c, dst_c, &bufs.mu1, &bufs.mu2);
        edge[0] = art * one_over_n;
        edge[1] = (art4 * one_over_n).powf(0.25);
        edge[2] = det * one_over_n;
        edge[3] = (det4 * one_over_n).powf(0.25);
    }

    ChannelResult { ssim, edge }
}

/// Minimum pixel count to justify spawning a parallel thread.
const PARALLEL_THRESHOLD: usize = 50_000;

/// Compute SSIM and edge statistics for a single scale.
/// Uses parallel channel processing for large scales.
#[allow(clippy::too_many_arguments)]
fn compute_single_scale(
    src: &[Vec<f32>; 3],
    dst: &[Vec<f32>; 3],
    width: usize,
    height: usize,
    blur_radius: usize,
    blur_passes: u8,
    bufs: &mut ScaleBuffers,
    parallel_bufs: &mut ScaleBuffers,
    scale_idx: usize,
    compute_all: bool,
) -> ScaleStats {
    let mut ssim_vals = [0.0f64; 6];
    let mut edge_vals = [0.0f64; 12];

    // Check if any weight is nonzero for a given feature type at this scale+channel
    let has_weight = |base_idx: usize, count: usize| -> bool {
        (base_idx..base_idx + count).all(|i| i < WEIGHTS.len())
            && (base_idx..base_idx + count).any(|i| WEIGHTS[i].abs() > 0.001)
    };

    // Determine which channels need work
    let mut active_channels: Vec<(usize, bool, bool)> = Vec::new();
    for c in 0..3 {
        let base = scale_idx * FEATURES_PER_SCALE + c * 6;
        let need_ssim = compute_all || has_weight(base, 2);
        let need_edge = compute_all || has_weight(base + 2, 4);
        if need_ssim || need_edge {
            active_channels.push((c, need_ssim, need_edge));
        }
    }

    let n = width * height;

    if active_channels.len() >= 2 && n >= PARALLEL_THRESHOLD {
        // Parallel: process first channel on a spawned thread, rest on current thread.
        // Both use pre-allocated buffer sets (no per-call allocation).
        let (first_c, first_ssim, first_edge) = active_channels[0];
        let rest = &active_channels[1..];

        let first_result = std::thread::scope(|s| {
            let handle = s.spawn(|| {
                compute_channel(
                    &src[first_c],
                    &dst[first_c],
                    width,
                    height,
                    blur_radius,
                    blur_passes,
                    first_ssim,
                    first_edge,
                    parallel_bufs,
                )
            });

            // Process remaining channels on the current thread
            let mut rest_results: Vec<(usize, ChannelResult)> = Vec::new();
            for &(c, need_ssim, need_edge) in rest {
                let result = compute_channel(
                    &src[c],
                    &dst[c],
                    width,
                    height,
                    blur_radius,
                    blur_passes,
                    need_ssim,
                    need_edge,
                    bufs,
                );
                rest_results.push((c, result));
            }

            let first = handle.join().unwrap();
            (first, rest_results)
        });

        // Collect results
        let (first, rest_results) = first_result;
        ssim_vals[first_c * 2] = first.ssim[0];
        ssim_vals[first_c * 2 + 1] = first.ssim[1];
        edge_vals[first_c * 4] = first.edge[0];
        edge_vals[first_c * 4 + 1] = first.edge[1];
        edge_vals[first_c * 4 + 2] = first.edge[2];
        edge_vals[first_c * 4 + 3] = first.edge[3];
        for (c, result) in rest_results {
            ssim_vals[c * 2] = result.ssim[0];
            ssim_vals[c * 2 + 1] = result.ssim[1];
            edge_vals[c * 4] = result.edge[0];
            edge_vals[c * 4 + 1] = result.edge[1];
            edge_vals[c * 4 + 2] = result.edge[2];
            edge_vals[c * 4 + 3] = result.edge[3];
        }
    } else {
        // Sequential: use shared buffers
        for &(c, need_ssim, need_edge) in &active_channels {
            let result = compute_channel(
                &src[c],
                &dst[c],
                width,
                height,
                blur_radius,
                blur_passes,
                need_ssim,
                need_edge,
                bufs,
            );
            ssim_vals[c * 2] = result.ssim[0];
            ssim_vals[c * 2 + 1] = result.ssim[1];
            edge_vals[c * 4] = result.edge[0];
            edge_vals[c * 4 + 1] = result.edge[1];
            edge_vals[c * 4 + 2] = result.edge[2];
            edge_vals[c * 4 + 3] = result.edge[3];
        }
    }

    ScaleStats {
        ssim: ssim_vals,
        edge: edge_vals,
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
/// Total number of features per scale (3 channels × (2 SSIM + 4 edge) = 18)
pub const FEATURES_PER_SCALE: usize = 18;

/// Trained weights from TID2013 optimization (3000 pairs).
/// Layout: 4 scales × 3 channels (X,Y,B) × 6 features (ssim_mean, ssim_4th,
///         edge_art_mean, edge_art_4th, edge_det_mean, edge_det_4th)
#[allow(clippy::excessive_precision)]
const WEIGHTS: [f64; 72] = [
    // Scale 0 Channel X
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, // Scale 0 Channel Y
    0.117592, 8.367503, 0.000000, 0.000000, 25.587284, 0.000000, // Scale 0 Channel B
    0.000000, 0.000000, 0.000000, 22.157398, 55.113319, 0.000000,
    // Scale 1 Channel X
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, // Scale 1 Channel Y
    50.202655, 4.162088, 0.000000, 0.000000, 125.177307, 115.357675,
    // Scale 1 Channel B
    0.000000, 0.000000, 0.000000, 0.000000, 19.069615, 0.000000, // Scale 2 Channel X
    0.000000, 0.000000, 0.000000, 0.000000, 15.958061, 0.000000, // Scale 2 Channel Y
    0.000000, 1.245107, 0.000000, 0.000000, 0.000000, 0.000000, // Scale 2 Channel B
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, // Scale 3 Channel X
    24.891486, 8.001749, 0.000000, 0.000000, 374.130349, 0.000000,
    // Scale 3 Channel Y
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, // Scale 3 Channel B
    21.750148, 7.333710, 0.000000, 0.000000, 214.128879, 0.000000,
];

fn combine_scores(scale_stats: &[ScaleStats]) -> ZensimResult {
    // Collect raw features and compute weighted distance
    let mut features = Vec::with_capacity(scale_stats.len() * FEATURES_PER_SCALE);
    let mut raw_distance = 0.0f64;

    for ss in scale_stats.iter() {
        for c in 0..3 {
            let ssim_feats = [ss.ssim[c * 2].abs(), ss.ssim[c * 2 + 1].abs()];
            let edge_feats = [
                ss.edge[c * 4].abs(),
                ss.edge[c * 4 + 1].abs(),
                ss.edge[c * 4 + 2].abs(),
                ss.edge[c * 4 + 3].abs(),
            ];
            features.extend_from_slice(&ssim_feats);
            features.extend_from_slice(&edge_feats);
        }
    }

    // Apply weights (features and weights have same layout)
    for (i, (&feat, &weight)) in features.iter().zip(WEIGHTS.iter()).enumerate() {
        let _ = i; // suppress warning
        raw_distance += feat * weight;
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
