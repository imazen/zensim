//! Core zensim metric computation.
//!
//! Multi-scale SSIM + edge features in XYB color space,
//! with contrast sensitivity weighting per scale.

use crate::blur::{box_blur_3pass_into, downscale_2x};
use crate::color::{make_positive_xyb, srgb_to_xyb_planar};
use crate::error::ZensimError;
use crate::pool::ScaleBuffers;
use crate::simd_ops::{edge_diff_channel, mul_into, ssim_channel};
use crate::NUM_SCALES;

/// Configuration for zensim computation.
#[derive(Debug, Clone, Copy)]
pub struct ZensimConfig {
    /// Box blur radius at scale 0 (default: 2, giving 5-pixel kernel)
    pub blur_radius: usize,
    /// Compute all features even if their weights are zero.
    /// Enable this for weight training to avoid circular dependency.
    pub compute_all_features: bool,
}

impl Default for ZensimConfig {
    fn default() -> Self {
        Self {
            blur_radius: 2,
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
        //   d=1.5 → ~60 (moderate distortion)
        //   d=5.0 → ~20 (severe distortion)
        //   d=10  → ~5  (heavy distortion)
        let scale = 22.0;
        let gamma = 0.65;
        (100.0 - scale * raw_distance.powf(gamma)).max(0.0)
    }
}

/// Compute score from raw features using custom weights.
/// `features`: raw features from ZensimResult.features
/// `weights`: one weight per feature (len must equal features.len())
/// Returns (score, raw_distance)
pub fn score_from_features(features: &[f64], weights: &[f64]) -> (f64, f64) {
    assert_eq!(features.len(), weights.len(), "features and weights must have same length");
    let raw_distance: f64 = features.iter().zip(weights.iter())
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

    // Convert to planar XYB
    let mut src_xyb = srgb_to_xyb_planar(source);
    let mut dst_xyb = srgb_to_xyb_planar(distorted);

    // Make XYB positive
    {
        let [ref mut sx, ref mut sy, ref mut sb] = src_xyb;
        make_positive_xyb(sx, sy, sb);
    }
    {
        let [ref mut dx, ref mut dy, ref mut db] = dst_xyb;
        make_positive_xyb(dx, dy, db);
    }

    // Compute multi-scale statistics
    let scale_stats = compute_multiscale_stats(
        &src_xyb, &dst_xyb, width, height, config.blur_radius, config.compute_all_features,
    );

    // Combine with weights to produce final score
    let result = combine_scores(&scale_stats);
    Ok(result)
}

/// Compute per-scale SSIM and edge statistics.
fn compute_multiscale_stats(
    src_xyb: &[Vec<f32>; 3],
    dst_xyb: &[Vec<f32>; 3],
    width: usize,
    height: usize,
    blur_radius: usize,
    compute_all: bool,
) -> Vec<ScaleStats> {
    let mut stats = Vec::with_capacity(NUM_SCALES);

    // Start with the original planes (avoid clone for scale 0 by using references first)
    let mut src_planes: [Vec<f32>; 3] = [
        src_xyb[0].clone(),
        src_xyb[1].clone(),
        src_xyb[2].clone(),
    ];
    let mut dst_planes: [Vec<f32>; 3] = [
        dst_xyb[0].clone(),
        dst_xyb[1].clone(),
        dst_xyb[2].clone(),
    ];
    let mut w = width;
    let mut h = height;

    // Pre-allocate buffers at max size, reuse across scales
    let max_n = width * height;
    let mut bufs = ScaleBuffers::new(max_n);

    for scale in 0..NUM_SCALES {
        if w < 8 || h < 8 {
            break;
        }

        let n = w * h;
        bufs.resize(n);

        let scale_stat = compute_single_scale(
            &src_planes, &dst_planes, w, h, blur_radius, &mut bufs, scale, compute_all,
        );
        stats.push(scale_stat);

        // Downscale for next level
        if scale < NUM_SCALES - 1 {
            let mut new_src = [Vec::new(), Vec::new(), Vec::new()];
            let mut new_dst = [Vec::new(), Vec::new(), Vec::new()];
            let mut nw = 0;
            let mut nh = 0;
            for c in 0..3 {
                let (s, sw, sh) = downscale_2x(&src_planes[c], w, h);
                let (d, _, _) = downscale_2x(&dst_planes[c], w, h);
                new_src[c] = s;
                new_dst[c] = d;
                nw = sw;
                nh = sh;
            }
            src_planes = new_src;
            dst_planes = new_dst;
            w = nw;
            h = nh;
        }
    }

    stats
}

/// Compute SSIM and edge statistics for a single scale.
/// Only computes SSIM when needed (most trained weights are on edge features).
fn compute_single_scale(
    src: &[Vec<f32>; 3],
    dst: &[Vec<f32>; 3],
    width: usize,
    height: usize,
    blur_radius: usize,
    bufs: &mut ScaleBuffers,
    scale_idx: usize,
    compute_all: bool,
) -> ScaleStats {
    let n = width * height;
    let one_over_n = 1.0 / n as f64;

    let mut ssim_vals = [0.0f64; 6];
    let mut edge_vals = [0.0f64; 12];

    // Check if any weight is nonzero for a given feature type at this scale+channel
    let has_weight = |base_idx: usize, count: usize| -> bool {
        (base_idx..base_idx + count)
            .all(|i| i < WEIGHTS.len())
            && (base_idx..base_idx + count).any(|i| WEIGHTS[i].abs() > 0.001)
    };

    for c in 0..3 {
        let base = scale_idx * FEATURES_PER_SCALE + c * 6;
        let need_ssim = compute_all || has_weight(base, 2);
        let need_edge = compute_all || has_weight(base + 2, 4);

        // Skip channel entirely if no weights are active
        if !need_ssim && !need_edge {
            continue;
        }

        let src_c = &src[c];
        let dst_c = &dst[c];

        // mu1 and mu2 are needed for both SSIM and edge features
        box_blur_3pass_into(src_c, &mut bufs.mu1, &mut bufs.temp_blur, width, height, blur_radius);
        box_blur_3pass_into(dst_c, &mut bufs.mu2, &mut bufs.temp_blur, width, height, blur_radius);

        // Only compute variance/covariance if SSIM weight is nonzero
        if need_ssim {
            mul_into(src_c, src_c, &mut bufs.mul_buf);
            box_blur_3pass_into(&bufs.mul_buf, &mut bufs.sigma1_sq, &mut bufs.temp_blur, width, height, blur_radius);

            mul_into(dst_c, dst_c, &mut bufs.mul_buf);
            box_blur_3pass_into(&bufs.mul_buf, &mut bufs.sigma2_sq, &mut bufs.temp_blur, width, height, blur_radius);

            mul_into(src_c, dst_c, &mut bufs.mul_buf);
            box_blur_3pass_into(&bufs.mul_buf, &mut bufs.sigma12, &mut bufs.temp_blur, width, height, blur_radius);

            let (sum_d, sum_d4) = ssim_channel(&bufs.mu1, &bufs.mu2, &bufs.sigma1_sq, &bufs.sigma2_sq, &bufs.sigma12);
            ssim_vals[c * 2] = sum_d * one_over_n;
            ssim_vals[c * 2 + 1] = (sum_d4 * one_over_n).powf(0.25);
        }

        // Edge features
        if need_edge {
            let (art, art4, det, det4) = edge_diff_channel(src_c, dst_c, &bufs.mu1, &bufs.mu2);
            edge_vals[c * 4] = art * one_over_n;
            edge_vals[c * 4 + 1] = (art4 * one_over_n).powf(0.25);
            edge_vals[c * 4 + 2] = det * one_over_n;
            edge_vals[c * 4 + 3] = (det4 * one_over_n).powf(0.25);
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
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    // Scale 0 Channel Y
    0.000000, 1.255125, 0.000000, 10.645667, 0.000000, 0.000000,
    // Scale 0 Channel B
    0.000000, 0.000000, 0.000000, 24.879610, 0.000000, 0.000000,
    // Scale 1 Channel X
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    // Scale 1 Channel Y
    11.228827, 10.367977, 0.000000, 0.000000, 95.014995, 0.000000,
    // Scale 1 Channel B
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    // Scale 2 Channel X
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    // Scale 2 Channel Y
    0.000000, 4.648608, 0.000000, 0.000000, 429.911258, 0.000000,
    // Scale 2 Channel B
    0.000000, 0.000000, 0.000000, 0.000000, 237.902858, 0.000000,
    // Scale 3 Channel X
    0.000000, 5.370131, 0.000000, 0.000000, 547.073252, 0.000000,
    // Scale 3 Channel Y
    0.000000, 3.771915, 0.000000, 0.000000, 0.000000, 0.000000,
    // Scale 3 Channel B
    13.260158, 0.006120, 0.000000, 0.000000, 403.936001, 0.000000,
];

fn combine_scores(scale_stats: &[ScaleStats]) -> ZensimResult {
    // Collect raw features and compute weighted distance
    let mut features = Vec::with_capacity(scale_stats.len() * FEATURES_PER_SCALE);
    let mut raw_distance = 0.0f64;

    for ss in scale_stats.iter() {
        for c in 0..3 {
            let ssim_feats = [
                ss.ssim[c * 2].abs(),
                ss.ssim[c * 2 + 1].abs(),
            ];
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
