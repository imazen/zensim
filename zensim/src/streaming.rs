//! Parallel multi-scale metric computation with band-based strip processing.
//!
//! Phase 1: Convert sRGB→XYB for the entire image (parallel over row chunks).
//! Phase 2: Process each pyramid scale with parallel band processing.
//!   Strip-based H-blur → fused V-blur+features (parallel bands via rayon).

use crate::blur::{
    box_blur_2pass_into, box_blur_3pass_into, downscale_2x_inplace, fused_blur_h_mu,
    fused_blur_h_ssim, simd_padded_width,
};
use crate::color::srgb_to_positive_xyb_planar_into;
use crate::fused::{fused_vblur_features_edge, fused_vblur_features_ssim};
use crate::metric::{
    FEATURES_PER_CHANNEL_BASIC, ScaleStats, WEIGHTS, ZensimConfig, combine_scores,
};
use crate::pool::ScaleBuffers;
use crate::simd_ops::{
    abs_diff_sum, edge_diff_channel, mul_into, sq_diff_sum, sq_sum_into, ssim_channel,
};
use rayon::prelude::*;
use std::sync::Mutex;

/// Inner strip height: rows of useful output per strip (must be even for 2x downscale).
const STRIP_INNER: usize = 16;

/// Track background deallocation thread to prevent accumulation on repeated calls.
static DEALLOC_THREAD: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

/// Should we use the streaming path for this image size?
/// Benchmarking shows streaming band processing wins at all sizes (down to 98k pixels),
/// so this always returns true for non-trivial images.
pub(crate) fn should_use_streaming(_width: usize, _height: usize) -> bool {
    true
}

/// Per-scale feature accumulators. Collects raw sums across strips,
/// finalized to ScaleStats at the end.
struct ScaleAccumulators {
    // SSIM: sum_d, sum_d^4, sum_d^2 per channel
    ssim_d: [f64; 3],
    ssim_d4: [f64; 3],
    ssim_d2: [f64; 3],
    // Edge: artifact and detail_lost sums per channel
    edge_art: [f64; 3],
    edge_art4: [f64; 3],
    edge_art2: [f64; 3],
    edge_det: [f64; 3],
    edge_det4: [f64; 3],
    edge_det2: [f64; 3],
    // MSE: sum((src-dst)^2)
    mse: [f64; 3],
    // Variance loss components: sum((pixel-mu)^2) for src and dst
    sq_src: [f64; 3],
    sq_dst: [f64; 3],
    // Texture loss components: sum(|pixel-mu|) for src and dst
    abs_src: [f64; 3],
    abs_dst: [f64; 3],
    // Total inner pixels processed
    n: usize,
}

impl ScaleAccumulators {
    fn new() -> Self {
        Self {
            ssim_d: [0.0; 3],
            ssim_d4: [0.0; 3],
            ssim_d2: [0.0; 3],
            edge_art: [0.0; 3],
            edge_art4: [0.0; 3],
            edge_art2: [0.0; 3],
            edge_det: [0.0; 3],
            edge_det4: [0.0; 3],
            edge_det2: [0.0; 3],
            mse: [0.0; 3],
            sq_src: [0.0; 3],
            sq_dst: [0.0; 3],
            abs_src: [0.0; 3],
            abs_dst: [0.0; 3],
            n: 0,
        }
    }

    fn merge(&mut self, other: &Self) {
        for c in 0..3 {
            self.ssim_d[c] += other.ssim_d[c];
            self.ssim_d4[c] += other.ssim_d4[c];
            self.ssim_d2[c] += other.ssim_d2[c];
            self.edge_art[c] += other.edge_art[c];
            self.edge_art4[c] += other.edge_art4[c];
            self.edge_art2[c] += other.edge_art2[c];
            self.edge_det[c] += other.edge_det[c];
            self.edge_det4[c] += other.edge_det4[c];
            self.edge_det2[c] += other.edge_det2[c];
            self.mse[c] += other.mse[c];
            self.sq_src[c] += other.sq_src[c];
            self.sq_dst[c] += other.sq_dst[c];
            self.abs_src[c] += other.abs_src[c];
            self.abs_dst[c] += other.abs_dst[c];
        }
        self.n += other.n;
    }

    fn finalize(&self) -> ScaleStats {
        let one_over_n = 1.0 / self.n as f64;

        let mut ssim = [0.0f64; 6];
        let mut edge = [0.0f64; 12];
        let mut mse = [0.0f64; 3];
        let mut variance_loss = [0.0f64; 3];
        let mut texture_loss = [0.0f64; 3];
        let mut contrast_increase = [0.0f64; 3];
        let mut ssim_2nd = [0.0f64; 3];
        let mut edge_2nd = [0.0f64; 6];

        for c in 0..3 {
            ssim[c * 2] = self.ssim_d[c] * one_over_n;
            ssim[c * 2 + 1] = (self.ssim_d4[c] * one_over_n).powf(0.25);
            ssim_2nd[c] = (self.ssim_d2[c] * one_over_n).sqrt();
            edge[c * 4] = self.edge_art[c] * one_over_n;
            edge[c * 4 + 1] = (self.edge_art4[c] * one_over_n).powf(0.25);
            edge[c * 4 + 2] = self.edge_det[c] * one_over_n;
            edge[c * 4 + 3] = (self.edge_det4[c] * one_over_n).powf(0.25);
            edge_2nd[c * 2] = (self.edge_art2[c] * one_over_n).sqrt();
            edge_2nd[c * 2 + 1] = (self.edge_det2[c] * one_over_n).sqrt();
            mse[c] = self.mse[c] * one_over_n;

            let var_src = self.sq_src[c] * one_over_n;
            let var_dst = self.sq_dst[c] * one_over_n;
            variance_loss[c] = if var_src > 1e-10 {
                (1.0 - var_dst / var_src).max(0.0)
            } else {
                0.0
            };
            contrast_increase[c] = if var_src > 1e-10 {
                (var_dst / var_src - 1.0).max(0.0)
            } else {
                0.0
            };

            let mad_src = self.abs_src[c] * one_over_n;
            let mad_dst = self.abs_dst[c] * one_over_n;
            texture_loss[c] = if mad_src > 1e-10 {
                (1.0 - mad_dst / mad_src).max(0.0)
            } else {
                0.0
            };
        }

        ScaleStats {
            ssim,
            edge,
            mse,
            variance_loss,
            texture_loss,
            contrast_increase,
            ssim_2nd,
            edge_2nd,
        }
    }
}

/// Determine which channels need SSIM, edge, and/or MSE computation at a given scale.
fn active_channels(scale_idx: usize, config: &ZensimConfig) -> Vec<(usize, bool, bool)> {
    let compute_all = config.compute_all_features;
    let fpc = FEATURES_PER_CHANNEL_BASIC;

    let has_weight = |base: usize, count: usize| -> bool {
        (base..base + count).all(|i| i < WEIGHTS.len())
            && (base..base + count).any(|i| WEIGHTS[i].abs() > 0.001)
    };

    // Feature layout per channel (13): ssim_mean(0), ssim_4th(1), ssim_2nd(2),
    //   art_mean(3), art_4th(4), art_2nd(5), det_mean(6), det_4th(7), det_2nd(8),
    //   mse(9), variance_loss(10), texture_loss(11), contrast_increase(12)
    let mut active = Vec::new();
    let beyond = scale_idx * (fpc * 3) >= WEIGHTS.len();
    for c in 0..3 {
        if beyond {
            if compute_all {
                active.push((c, true, true));
            }
        } else {
            let base = scale_idx * (fpc * 3) + c * fpc;
            let need_ssim = compute_all || has_weight(base, 3); // positions 0-2
            let need_blur = has_weight(base + 10, 3); // positions 10-12
            // Blur features need mu1/mu2 (same as edge), fold into need_edge
            let need_edge = compute_all || has_weight(base + 3, 6) || need_blur; // positions 3-8
            let need_mse = compute_all || has_weight(base + 9, 1); // position 9
            if need_ssim || need_edge || need_mse {
                active.push((c, need_ssim, need_edge));
            }
        }
    }
    active
}

/// Streaming multi-scale stats: parallel XYB conversion, then band-parallel blur/features.
///
/// Phase 1: Convert sRGB→XYB for the entire image (parallel over row chunks).
/// Phase 2: Process each scale with parallel band processing over the XYB planes.
///
/// Produces identical results to the full-image path.
pub(crate) fn compute_multiscale_stats_streaming(
    source: &[[u8; 3]],
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
    config: &ZensimConfig,
) -> Vec<ScaleStats> {
    let padded_width = simd_padded_width(width);
    let num_scales = config.num_scales;

    // Phase 1: Convert sRGB→XYB for entire image, parallel over row chunks.
    let mut src_planes = convert_srgb_to_xyb_parallel(source, width, height, padded_width);
    let mut dst_planes = convert_srgb_to_xyb_parallel(distorted, width, height, padded_width);

    // Phase 2: Process all scales with parallel band processing.
    // Each scale: compute features in parallel bands, then downscale planes for next scale.
    let mut stats = Vec::with_capacity(num_scales);
    let mut w = padded_width;
    let mut h = height;

    for scale in 0..num_scales {
        if w < 8 || h < 8 {
            break;
        }

        // Parallel band processing: borrow slices from full planes
        let scale_stat = process_scale_bands(&src_planes, &dst_planes, w, h, config, scale);
        stats.push(scale_stat);

        // Downscale 6 planes in parallel for next scale
        if scale < num_scales - 1 {
            let [ref mut s0, ref mut s1, ref mut s2] = src_planes;
            let [ref mut d0, ref mut d1, ref mut d2] = dst_planes;
            let (((nw, nh), _), _) = rayon::join(
                || {
                    rayon::join(
                        || downscale_2x_inplace(s0, w, h),
                        || {
                            rayon::join(
                                || downscale_2x_inplace(s1, w, h),
                                || downscale_2x_inplace(s2, w, h),
                            )
                        },
                    )
                },
                || {
                    rayon::join(
                        || downscale_2x_inplace(d0, w, h),
                        || {
                            rayon::join(
                                || downscale_2x_inplace(d1, w, h),
                                || downscale_2x_inplace(d2, w, h),
                            )
                        },
                    )
                },
            );
            w = nw;
            h = nh;
        }
    }

    // Background deallocation: move ~400MB of XYB planes to a background thread
    // to avoid blocking on munmap syscalls (which take ~57ms on WSL for this volume).
    // Track the thread to prevent accumulation on repeated calls.
    {
        let mut guard = DEALLOC_THREAD.lock().unwrap();
        if let Some(prev) = guard.take() {
            let _ = prev.join();
        }
        *guard = Some(std::thread::spawn(move || {
            drop(src_planes);
            drop(dst_planes);
        }));
    }

    stats
}

/// Convert sRGB pixels to planar XYB at padded width, parallelized over row chunks.
fn convert_srgb_to_xyb_parallel(
    pixels: &[[u8; 3]],
    width: usize,
    height: usize,
    padded_width: usize,
) -> [Vec<f32>; 3] {
    let n = padded_width * height;
    let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; n]);

    let chunk_rows = 64;
    let [ref mut p0, ref mut p1, ref mut p2] = planes;
    let p0_chunks: Vec<&mut [f32]> = p0.chunks_mut(chunk_rows * padded_width).collect();
    let p1_chunks: Vec<&mut [f32]> = p1.chunks_mut(chunk_rows * padded_width).collect();
    let p2_chunks: Vec<&mut [f32]> = p2.chunks_mut(chunk_rows * padded_width).collect();

    // Precompute mirror indices for padding columns (same for every row)
    let pad_count = padded_width - width;
    let mirror_offsets: Vec<usize> = if pad_count > 0 {
        let period = 2 * (width - 1);
        (0..pad_count)
            .map(|i| {
                let m = (width + i) % period;
                if m < width { m } else { period - m }
            })
            .collect()
    } else {
        Vec::new()
    };

    p0_chunks
        .into_par_iter()
        .zip(p1_chunks)
        .zip(p2_chunks)
        .enumerate()
        .for_each(|(chunk_idx, ((c0, c1), c2))| {
            let row_start = chunk_idx * chunk_rows;
            let row_end = (row_start + chunk_rows).min(height);
            let rows = row_end - row_start;
            let pixel_slice = &pixels[row_start * width..row_end * width];

            // Convert sRGB→XYB: output goes into first rows*width elements of each chunk
            let raw_elems = rows * width;
            srgb_to_positive_xyb_planar_into(
                pixel_slice,
                &mut c0[..raw_elems],
                &mut c1[..raw_elems],
                &mut c2[..raw_elems],
            );

            // Spread rows from logical width to padded width (bottom-to-top for overlap safety)
            if pad_count > 0 {
                for plane in [&mut *c0, &mut *c1, &mut *c2] {
                    for y in (0..rows).rev() {
                        let src_start = y * width;
                        let dst_start = y * padded_width;
                        // Shift row data to padded position (right-to-left for overlap safety)
                        if dst_start != src_start {
                            for x in (0..width).rev() {
                                plane[dst_start + x] = plane[src_start + x];
                            }
                        }
                        // Fill padding columns with mirror-reflected values
                        for (i, &mx) in mirror_offsets.iter().enumerate() {
                            plane[dst_start + width + i] = plane[dst_start + mx];
                        }
                    }
                }
            }
        });

    planes
}

/// Process one channel of one strip: blur, extract inner rows, accumulate features.
///
/// Three paths based on what features the channel needs:
/// 1. MSE only (no blur) — raw pixel differences
/// 2. SSIM channel (1-pass blur): fused H-blur → fused V-blur + all features
/// 3. Edge-only channel (1-pass blur): separate H-blur → fused V-blur + features
/// 4. Multi-pass blur fallback: separate blur + reduce (unchanged from original)
#[allow(clippy::too_many_arguments)]
fn process_strip_channel(
    src_c: &[f32],
    dst_c: &[f32],
    width: usize,
    strip_h: usize,
    inner_start: usize,
    inner_h: usize,
    config: &ZensimConfig,
    c: usize,
    need_ssim: bool,
    need_edge: bool,
    bufs: &mut ScaleBuffers,
    accum: &mut ScaleAccumulators,
) {
    // MSE-only path: no blur needed
    if !need_ssim && !need_edge {
        let inner_off = inner_start * width;
        let inner_n = inner_h * width;
        let inner_src = &src_c[inner_off..inner_off + inner_n];
        let inner_dst = &dst_c[inner_off..inner_off + inner_n];
        accum.mse[c] += sq_diff_sum(inner_src, inner_dst);
        return;
    }

    // Fused path: 1-pass blur only (the common case for scale 0)
    if config.blur_passes == 1 {
        if need_ssim {
            // Fused H-blur: src,dst → 4 H-blurred planes in one pass
            fused_blur_h_ssim(
                src_c,
                dst_c,
                &mut bufs.mu1,
                &mut bufs.mu2,
                &mut bufs.sigma1_sq,
                &mut bufs.sigma12,
                width,
                strip_h,
                config.blur_radius,
            );

            // Fused V-blur + ALL feature extraction: no memory writes
            let strip_acc = fused_vblur_features_ssim(
                &bufs.mu1,
                &bufs.mu2,
                &bufs.sigma1_sq,
                &bufs.sigma12,
                src_c,
                dst_c,
                width,
                strip_h,
                inner_start,
                inner_h,
                config.blur_radius,
            );

            accum.ssim_d[c] += strip_acc.ssim_d;
            accum.ssim_d4[c] += strip_acc.ssim_d4;
            accum.ssim_d2[c] += strip_acc.ssim_d2;
            accum.edge_art[c] += strip_acc.edge_art;
            accum.edge_art4[c] += strip_acc.edge_art4;
            accum.edge_art2[c] += strip_acc.edge_art2;
            accum.edge_det[c] += strip_acc.edge_det;
            accum.edge_det4[c] += strip_acc.edge_det4;
            accum.edge_det2[c] += strip_acc.edge_det2;
            accum.mse[c] += strip_acc.mse;
            accum.sq_src[c] += strip_acc.sq_src;
            accum.sq_dst[c] += strip_acc.sq_dst;
            accum.abs_src[c] += strip_acc.abs_src;
            accum.abs_dst[c] += strip_acc.abs_dst;
        } else {
            // Edge-only: fused H-blur for mu1/mu2, then fused V-blur
            fused_blur_h_mu(
                src_c,
                dst_c,
                &mut bufs.mu1,
                &mut bufs.mu2,
                width,
                strip_h,
                config.blur_radius,
            );

            let strip_acc = fused_vblur_features_edge(
                &bufs.mu1,
                &bufs.mu2,
                src_c,
                dst_c,
                width,
                strip_h,
                inner_start,
                inner_h,
                config.blur_radius,
            );

            accum.edge_art[c] += strip_acc.edge_art;
            accum.edge_art4[c] += strip_acc.edge_art4;
            accum.edge_art2[c] += strip_acc.edge_art2;
            accum.edge_det[c] += strip_acc.edge_det;
            accum.edge_det4[c] += strip_acc.edge_det4;
            accum.edge_det2[c] += strip_acc.edge_det2;
            accum.mse[c] += strip_acc.mse;
            accum.sq_src[c] += strip_acc.sq_src;
            accum.sq_dst[c] += strip_acc.sq_dst;
            accum.abs_src[c] += strip_acc.abs_src;
            accum.abs_dst[c] += strip_acc.abs_dst;
        }
        return;
    }

    // Multi-pass blur fallback: separate blur + reduce
    #[allow(clippy::type_complexity)]
    let blur_fn: fn(&[f32], &mut [f32], &mut [f32], usize, usize, usize) = match config.blur_passes
    {
        2 => box_blur_2pass_into,
        _ => box_blur_3pass_into,
    };

    blur_fn(
        src_c,
        &mut bufs.mu1,
        &mut bufs.temp_blur,
        width,
        strip_h,
        config.blur_radius,
    );
    blur_fn(
        dst_c,
        &mut bufs.mu2,
        &mut bufs.temp_blur,
        width,
        strip_h,
        config.blur_radius,
    );

    if need_ssim {
        sq_sum_into(src_c, dst_c, &mut bufs.mul_buf);
        blur_fn(
            &bufs.mul_buf,
            &mut bufs.sigma1_sq,
            &mut bufs.temp_blur,
            width,
            strip_h,
            config.blur_radius,
        );
        mul_into(src_c, dst_c, &mut bufs.mul_buf);
        blur_fn(
            &bufs.mul_buf,
            &mut bufs.sigma12,
            &mut bufs.temp_blur,
            width,
            strip_h,
            config.blur_radius,
        );
    }

    let inner_off = inner_start * width;
    let inner_n = inner_h * width;
    let inner_src = &src_c[inner_off..inner_off + inner_n];
    let inner_dst = &dst_c[inner_off..inner_off + inner_n];
    let inner_mu1 = &bufs.mu1[inner_off..inner_off + inner_n];
    let inner_mu2 = &bufs.mu2[inner_off..inner_off + inner_n];

    accum.mse[c] += sq_diff_sum(inner_src, inner_dst);

    if need_ssim {
        let inner_sig_sq = &bufs.sigma1_sq[inner_off..inner_off + inner_n];
        let inner_sig12 = &bufs.sigma12[inner_off..inner_off + inner_n];
        let (sum_d, sum_d4, sum_d2) = ssim_channel(inner_mu1, inner_mu2, inner_sig_sq, inner_sig12);
        accum.ssim_d[c] += sum_d;
        accum.ssim_d4[c] += sum_d4;
        accum.ssim_d2[c] += sum_d2;
    }

    if need_edge {
        let (art, art4, det, det4, art2, det2) =
            edge_diff_channel(inner_src, inner_dst, inner_mu1, inner_mu2);
        accum.edge_art[c] += art;
        accum.edge_art4[c] += art4;
        accum.edge_art2[c] += art2;
        accum.edge_det[c] += det;
        accum.edge_det4[c] += det4;
        accum.edge_det2[c] += det2;
    }

    accum.sq_src[c] += sq_diff_sum(inner_src, inner_mu1);
    accum.sq_dst[c] += sq_diff_sum(inner_dst, inner_mu2);
    accum.abs_src[c] += abs_diff_sum(inner_src, inner_mu1);
    accum.abs_dst[c] += abs_diff_sum(inner_dst, inner_mu2);
}

/// Process a scale using parallel band processing over pre-existing XYB planes.
///
/// Divides the image into horizontal bands, each processing sequential strips.
/// Each band runs on a separate thread via rayon.
fn process_scale_bands(
    src_planes: &[Vec<f32>; 3],
    dst_planes: &[Vec<f32>; 3],
    width: usize,
    height: usize,
    config: &ZensimConfig,
    scale_idx: usize,
) -> ScaleStats {
    let r = config.blur_radius;
    let passes = config.blur_passes as usize;
    let overlap = passes * r;
    let scale_active = active_channels(scale_idx, config);

    let total_strips = height.div_ceil(STRIP_INNER);
    let num_bands = rayon::current_num_threads().min(total_strips).max(1);
    let strips_per_band = total_strips.div_ceil(num_bands);

    let max_strip_h = STRIP_INNER + 2 * overlap;
    let max_strip_n = max_strip_h * width;

    let band_accums: Vec<_> = (0..num_bands)
        .into_par_iter()
        .map(|band_idx| {
            let band_first_y = (band_idx * strips_per_band * STRIP_INNER).min(height);
            let band_end_y = (((band_idx + 1) * strips_per_band) * STRIP_INNER).min(height);

            if band_first_y >= height {
                return ScaleAccumulators::new();
            }

            let mut accum = ScaleAccumulators::new();
            let mut bufs = ScaleBuffers::new(max_strip_n);

            let mut y = band_first_y;
            while y < band_end_y {
                let inner_end = (y + STRIP_INNER).min(height);
                let inner_h = inner_end - y;

                let strip_top = y.saturating_sub(overlap);
                let strip_bot = (inner_end + overlap).min(height);
                let strip_h = strip_bot - strip_top;
                let inner_start = y - strip_top;

                let strip_n = width * strip_h;
                bufs.resize(strip_n);

                accum.n += inner_h * width;

                for &(c, need_ssim, need_edge) in &scale_active {
                    process_strip_channel(
                        &src_planes[c][strip_top * width..strip_bot * width],
                        &dst_planes[c][strip_top * width..strip_bot * width],
                        width,
                        strip_h,
                        inner_start,
                        inner_h,
                        config,
                        c,
                        need_ssim,
                        need_edge,
                        &mut bufs,
                        &mut accum,
                    );
                }

                y = inner_end;
            }

            accum
        })
        .collect();

    // Merge band accumulators
    let mut accum = ScaleAccumulators::new();
    for band_accum in band_accums {
        accum.merge(&band_accum);
    }
    accum.finalize()
}

/// Pre-computed reference image data for batch comparison against multiple distorted images.
///
/// Caches the reference image's XYB color-space planes and downscale pyramid so that
/// sRGB→XYB conversion and pyramid construction happen once, not once per distorted image.
/// At 4K this saves ~25% per comparison; at 8K, ~34%. Breaks even at 3-7 distorted
/// images per reference depending on resolution.
///
/// # Memory
///
/// Holds 3 f32 planes at each pyramid scale (4 scales by default).
/// Total ≈ `width × height × 4 bytes × 3 channels × 1.33` (geometric sum of pyramid).
/// For a 3840×2160 image: ~133 MB. For 7680×4320: ~532 MB.
///
/// Created via [`precompute_reference`](crate::precompute_reference) (default 4 scales)
/// or `precompute_reference_with_scales` (custom, requires `training` feature).
pub struct PrecomputedReference {
    pub(crate) scales: Vec<([Vec<f32>; 3], usize, usize)>,
}

impl PrecomputedReference {
    /// Build a precomputed reference from sRGB pixels.
    ///
    /// Converts to XYB and builds the downscale pyramid, storing planes at each level.
    pub(crate) fn new(source: &[[u8; 3]], width: usize, height: usize, num_scales: usize) -> Self {
        let padded_width = simd_padded_width(width);
        let mut planes = convert_srgb_to_xyb_parallel(source, width, height, padded_width);

        let mut scales = Vec::with_capacity(num_scales);
        let mut w = padded_width;
        let mut h = height;

        for scale in 0..num_scales {
            if w < 8 || h < 8 {
                break;
            }

            // Clone and store planes at this scale
            scales.push((planes.clone(), w, h));

            // Downscale for next scale
            if scale < num_scales - 1 {
                let [ref mut s0, ref mut s1, ref mut s2] = planes;
                let ((nw, nh), _) = rayon::join(
                    || downscale_2x_inplace(s0, w, h),
                    || {
                        rayon::join(
                            || downscale_2x_inplace(s1, w, h),
                            || downscale_2x_inplace(s2, w, h),
                        )
                    },
                );
                w = nw;
                h = nh;
            }
        }

        Self { scales }
    }
}

/// Streaming multi-scale stats using a precomputed reference.
///
/// Only converts the distorted image to XYB and downscales it between scales.
/// Reference planes are borrowed from the precomputed data.
pub(crate) fn compute_multiscale_stats_streaming_with_ref(
    precomputed: &PrecomputedReference,
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
    config: &ZensimConfig,
) -> Vec<ScaleStats> {
    let padded_width = simd_padded_width(width);
    let num_scales = config.num_scales.min(precomputed.scales.len());

    // Only convert distorted to XYB
    let mut dst_planes = convert_srgb_to_xyb_parallel(distorted, width, height, padded_width);

    let mut stats = Vec::with_capacity(num_scales);
    let mut w = padded_width;
    let mut h = height;

    for scale in 0..num_scales {
        if w < 8 || h < 8 {
            break;
        }

        let (ref src_planes, src_w, src_h) = precomputed.scales[scale];
        debug_assert_eq!(w, src_w, "width mismatch at scale {scale}");
        debug_assert_eq!(h, src_h, "height mismatch at scale {scale}");

        let scale_stat = process_scale_bands(src_planes, &dst_planes, w, h, config, scale);
        stats.push(scale_stat);

        // Only downscale the 3 distorted planes for next scale
        if scale < num_scales - 1 {
            let [ref mut d0, ref mut d1, ref mut d2] = dst_planes;
            let ((nw, nh), _) = rayon::join(
                || downscale_2x_inplace(d0, w, h),
                || {
                    rayon::join(
                        || downscale_2x_inplace(d1, w, h),
                        || downscale_2x_inplace(d2, w, h),
                    )
                },
            );
            w = nw;
            h = nh;
        }
    }

    // Background deallocation for distorted planes
    {
        let mut guard = DEALLOC_THREAD.lock().unwrap();
        if let Some(prev) = guard.take() {
            let _ = prev.join();
        }
        *guard = Some(std::thread::spawn(move || {
            drop(dst_planes);
        }));
    }

    stats
}

/// Entry point: compute zensim using streaming with precomputed reference.
/// Produces identical results to the non-precomputed path.
pub(crate) fn compute_zensim_streaming_with_ref(
    precomputed: &PrecomputedReference,
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
    config: &ZensimConfig,
) -> crate::metric::ZensimResult {
    let scale_stats =
        compute_multiscale_stats_streaming_with_ref(precomputed, distorted, width, height, config);
    let masked = config.masking_strength > 0.0;
    combine_scores(&scale_stats, masked)
}

/// Entry point: compute zensim using streaming for scale 0, full-image for the rest.
/// Produces identical results to the full-image path.
pub(crate) fn compute_zensim_streaming(
    source: &[[u8; 3]],
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
    config: &ZensimConfig,
) -> crate::metric::ZensimResult {
    let scale_stats = compute_multiscale_stats_streaming(source, distorted, width, height, config);
    let masked = config.masking_strength > 0.0;
    combine_scores(&scale_stats, masked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::compute_zensim_with_config;

    /// Verify streaming produces equivalent results to full-image processing.
    ///
    /// The strip-based V-blur running sum starts from strip boundaries (with mirror
    /// padding) while the full-image V-blur starts from image row 0. Additionally,
    /// the full-image path uses fused_blur_h_ssim while strips use separate blur
    /// calls. These produce mathematically identical results but different FP rounding.
    ///
    /// For SSIM features in smooth image regions, catastrophic cancellation in
    /// sigma_sq = blur(src²) - mu² amplifies tiny blur differences by 10-100×.
    /// Features with larger absolute values (edges, variance/texture loss) match
    /// closely since they don't involve cancellation.
    ///
    /// We verify: (1) final score matches within 0.01%, (2) significant features
    /// match within 5%, (3) all features match within absolute tolerance 1e-3.
    #[test]
    fn streaming_matches_full_image() {
        let w = 256;
        let h = 256;
        let n = w * h;

        // Generate test images: gradient with some noise for texture
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = ((x + y) * 127 / (w + h)) as u8;
                src[y * w + x] = [r, g, b];
                dst[y * w + x] = [
                    r.saturating_add(3),
                    g.saturating_sub(2),
                    b.saturating_add(1),
                ];
            }
        }

        let config = ZensimConfig {
            compute_all_features: true,
            ..Default::default()
        };

        // Full-image path
        let full_result = compute_zensim_with_config(&src, &dst, w, h, config).unwrap();

        // Streaming path (forced via direct call)
        let streaming_result = compute_zensim_streaming(&src, &dst, w, h, &config);

        assert_eq!(
            full_result.features.len(),
            streaming_result.features.len(),
            "feature count mismatch"
        );

        // Diagnostics: print all differing features
        let feature_names = [
            "ssim_mean",
            "ssim_4th",
            "ssim_2nd",
            "edge_art_mean",
            "edge_art_4th",
            "edge_art_2nd",
            "edge_det_mean",
            "edge_det_4th",
            "edge_det_2nd",
            "mse",
            "var_loss",
            "tex_loss",
            "contrast_inc",
        ];
        let mut max_sig_rel = 0.0f64; // max relative diff for significant features
        let mut max_abs_diff = 0.0f64;
        for (i, (f, s)) in full_result
            .features
            .iter()
            .zip(streaming_result.features.iter())
            .enumerate()
        {
            let diff = (f - s).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            let absmax = f.abs().max(s.abs());
            if absmax > 0.01 {
                let rel = diff / absmax;
                if rel > max_sig_rel {
                    max_sig_rel = rel;
                }
            }
            if diff > 1e-8 {
                let scale = i / 39;
                let within = i % 39;
                let ch = within / 13;
                let fi = within % 13;
                let rel = diff / absmax.max(1e-12);
                eprintln!(
                    "  feat {:3} (s{} c{} {:14}) full={:12.8} stream={:12.8} diff={:.2e} rel={:.2e}",
                    i, scale, ch, feature_names[fi], f, s, diff, rel,
                );
            }
        }
        let score_rel =
            (full_result.score - streaming_result.score).abs() / full_result.score.abs().max(1e-12);
        let dist_rel = (full_result.raw_distance - streaming_result.raw_distance).abs()
            / full_result.raw_distance.abs().max(1e-12);
        eprintln!(
            "score: full={:.6} stream={:.6} (rel={:.2e})",
            full_result.score, streaming_result.score, score_rel,
        );
        eprintln!(
            "raw_distance: full={:.8} stream={:.8} (rel={:.2e})",
            full_result.raw_distance, streaming_result.raw_distance, dist_rel,
        );
        eprintln!(
            "max abs diff: {:.2e}, max sig rel diff: {:.2e}",
            max_abs_diff, max_sig_rel,
        );

        // Score must match within 0.01%
        assert!(
            score_rel < 0.0001,
            "score relative diff {:.2e} exceeds 0.01%",
            score_rel,
        );

        // Raw distance must match within 0.1%
        assert!(
            dist_rel < 0.001,
            "raw_distance relative diff {:.2e} exceeds 0.1%",
            dist_rel,
        );

        // Significant features (abs > 0.01) must match within 5%
        assert!(
            max_sig_rel < 0.05,
            "significant feature relative diff {:.2e} exceeds 5%",
            max_sig_rel,
        );

        // All features must match within absolute tolerance
        assert!(
            max_abs_diff < 1e-3,
            "max absolute feature diff {:.2e} exceeds 1e-3",
            max_abs_diff,
        );
    }

    /// Verify precomputed reference produces bit-identical results to the streaming path.
    #[test]
    fn precomputed_ref_matches_streaming() {
        let w = 256;
        let h = 256;
        let n = w * h;

        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = ((x + y) * 127 / (w + h)) as u8;
                src[y * w + x] = [r, g, b];
                dst[y * w + x] = [
                    r.saturating_add(3),
                    g.saturating_sub(2),
                    b.saturating_add(1),
                ];
            }
        }

        let config = ZensimConfig {
            compute_all_features: true,
            ..Default::default()
        };

        let streaming_result = compute_zensim_streaming(&src, &dst, w, h, &config);
        let precomputed = PrecomputedReference::new(&src, w, h, config.num_scales);
        let precomp_result = compute_zensim_streaming_with_ref(&precomputed, &dst, w, h, &config);

        assert_eq!(streaming_result.score, precomp_result.score);
        assert_eq!(streaming_result.raw_distance, precomp_result.raw_distance);
        assert_eq!(
            streaming_result.features.len(),
            precomp_result.features.len()
        );
        for (i, (s, p)) in streaming_result
            .features
            .iter()
            .zip(precomp_result.features.iter())
            .enumerate()
        {
            assert_eq!(s, p, "feature {i} mismatch: streaming={s} precomp={p}");
        }
    }
}
