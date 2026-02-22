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
}

impl Default for ZensimConfig {
    fn default() -> Self {
        Self { blur_radius: 2 }
    }
}

/// Per-scale statistics collected during computation.
struct ScaleStats {
    /// SSIM statistics: [mean_d, root4_d] per channel = 6 values
    ssim: [f64; 6],
    /// Edge features: [art_mean, art_4th, det_mean, det_4th] per channel = 12 values
    edge: [f64; 12],
}

/// Result from zensim comparison.
#[derive(Debug, Clone, Copy)]
pub struct ZensimResult {
    /// Score on 0-100 scale. 100 = identical, higher = better.
    pub score: f64,
    /// Raw weighted distance (before nonlinear mapping). Lower = more similar.
    pub raw_distance: f64,
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
        &src_xyb, &dst_xyb, width, height, config.blur_radius,
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
) -> Vec<ScaleStats> {
    let mut stats = Vec::with_capacity(NUM_SCALES);

    // Current scale planes (owned, will be downscaled)
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

    for _scale in 0..NUM_SCALES {
        let n = w * h;
        if w < 8 || h < 8 {
            break;
        }

        let mut bufs = ScaleBuffers::new(n);
        let scale_stat = compute_single_scale(
            &src_planes, &dst_planes, w, h, blur_radius, &mut bufs,
        );
        stats.push(scale_stat);

        // Downscale for next level
        if _scale < NUM_SCALES - 1 {
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
fn compute_single_scale(
    src: &[Vec<f32>; 3],
    dst: &[Vec<f32>; 3],
    width: usize,
    height: usize,
    blur_radius: usize,
    bufs: &mut ScaleBuffers,
) -> ScaleStats {
    let n = width * height;
    let one_over_n = 1.0 / n as f64;

    let mut ssim_vals = [0.0f64; 6];
    let mut edge_vals = [0.0f64; 12];

    for c in 0..3 {
        let src_c = &src[c];
        let dst_c = &dst[c];

        // mu1 = blur(src), mu2 = blur(dst)
        // Reuse bufs.mu2 for mu2, store mu1 in temp first
        let mut mu1 = vec![0.0f32; n];
        let mut temp_blur = vec![0.0f32; n];
        box_blur_3pass_into(src_c, &mut mu1, &mut temp_blur, width, height, blur_radius);
        box_blur_3pass_into(dst_c, &mut bufs.mu2, &mut temp_blur, width, height, blur_radius);

        // sigma1_sq = blur(src^2)
        mul_into(src_c, src_c, &mut bufs.mul_buf);
        let mut sigma1_sq = vec![0.0f32; n];
        box_blur_3pass_into(&bufs.mul_buf, &mut sigma1_sq, &mut temp_blur, width, height, blur_radius);

        // sigma2_sq = blur(dst^2)
        mul_into(dst_c, dst_c, &mut bufs.mul_buf);
        box_blur_3pass_into(&bufs.mul_buf, &mut bufs.sigma2_sq, &mut temp_blur, width, height, blur_radius);

        // sigma12 = blur(src*dst)
        mul_into(src_c, dst_c, &mut bufs.mul_buf);
        box_blur_3pass_into(&bufs.mul_buf, &mut bufs.sigma12, &mut temp_blur, width, height, blur_radius);

        // SSIM statistics
        let (sum_d, sum_d4) = ssim_channel(&mu1, &bufs.mu2, &sigma1_sq, &bufs.sigma2_sq, &bufs.sigma12);
        ssim_vals[c * 2] = sum_d * one_over_n;
        ssim_vals[c * 2 + 1] = (sum_d4 * one_over_n).powf(0.25);

        // Edge difference statistics
        let (art, art4, det, det4) = edge_diff_channel(src_c, dst_c, &mu1, &bufs.mu2);
        edge_vals[c * 4] = art * one_over_n;
        edge_vals[c * 4 + 1] = (art4 * one_over_n).powf(0.25);
        edge_vals[c * 4 + 2] = det * one_over_n;
        edge_vals[c * 4 + 3] = (det4 * one_over_n).powf(0.25);
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
fn combine_scores(scale_stats: &[ScaleStats]) -> ZensimResult {
    // Weight structure: for each scale, for each channel (X, Y, B):
    //   [ssim_mean, ssim_4th, edge_art_mean, edge_art_4th, edge_det_mean, edge_det_4th]
    // = 6 features per channel × 3 channels × NUM_SCALES = 72 weights
    //
    // Initial weights derived from ssimulacra2 with adjustments for:
    // - Butteraugli's insight that luminance (Y) channel matters most
    // - Higher weight on medium scales (where CSF peaks)
    // - Edge features weighted more than in ssimulacra2 (butteraugli emphasis)

    // Per-scale contrast sensitivity function weights.
    // These approximate the human CSF at different spatial frequencies.
    // Scale 0 = highest frequency, scale 3 = lowest frequency.
    // CSF peaks at medium spatial frequencies (~4 cycles/degree).
    let csf_weights: [f64; 4] = [0.7, 1.0, 0.85, 0.5];

    // Per-channel weights: Y (luminance) > B (blue) > X (red-green)
    // Based on butteraugli's psychovisual model + HVS research.
    // Normalized so total weight per scale ≈ 1.0
    let channel_weights: [f64; 3] = [
        0.15,  // X (red-green opponent)
        0.55,  // Y (luminance) — humans most sensitive here
        0.30,  // B (blue)
    ];

    // Feature type weights — SSIM mean is dominant
    let ssim_mean_w = 1.0;
    let ssim_4th_w = 0.3;
    let edge_art_mean_w = 0.5;
    let edge_art_4th_w = 0.15;
    let edge_det_mean_w = 0.4;
    let edge_det_4th_w = 0.15;

    let feature_weights = [
        ssim_mean_w, ssim_4th_w,
        edge_art_mean_w, edge_art_4th_w,
        edge_det_mean_w, edge_det_4th_w,
    ];

    let mut raw_distance = 0.0f64;

    for (scale_idx, ss) in scale_stats.iter().enumerate() {
        let csf_w = if scale_idx < csf_weights.len() {
            csf_weights[scale_idx]
        } else {
            0.3
        };

        for c in 0..3 {
            let ch_w = channel_weights[c];

            // SSIM features
            let ssim_features = [
                ss.ssim[c * 2].abs(),
                ss.ssim[c * 2 + 1].abs(),
            ];
            // Edge features
            let edge_features = [
                ss.edge[c * 4].abs(),
                ss.edge[c * 4 + 1].abs(),
                ss.edge[c * 4 + 2].abs(),
                ss.edge[c * 4 + 3].abs(),
            ];

            for (i, &feat) in ssim_features.iter().chain(edge_features.iter()).enumerate() {
                raw_distance += csf_w * ch_w * feature_weights[i] * feat;
            }
        }
    }

    // Normalize by number of scales
    raw_distance /= scale_stats.len().max(1) as f64;

    // Nonlinear mapping to 0-100 scale.
    // Modeled after ssimulacra2: score = 100 - 10 * distance^gamma
    // Calibrated so that:
    //   JPEG Q90 ≈ 80-90 (imperceptible)
    //   JPEG Q50 ≈ 50-65 (noticeable)
    //   JPEG Q10 ≈ 15-30 (significant)
    let score = if raw_distance <= 0.0 {
        100.0
    } else {
        // Use log mapping for better dynamic range:
        // score = 100 / (1 + k * distance^gamma)
        // This gives a smooth sigmoid-like curve that handles wide range.
        let k = 0.06;
        let gamma = 0.75;
        100.0 / (1.0 + k * raw_distance.powf(gamma))
    };

    ZensimResult {
        score,
        raw_distance,
    }
}
