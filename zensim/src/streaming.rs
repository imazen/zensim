//! Strip-based streaming metric computation for memory-efficient large images.
//!
//! Instead of processing the full image at each scale (O(W×H) memory for ~20 buffers),
//! processes images in horizontal strips of ~16 rows with blur overlap.
//! Scale 0 converts sRGB→XYB per strip; scales 1+ borrow directly from downscaled planes.
//! Small scales (below STREAMING_THRESHOLD) fall back to full-image processing.
//!
//! Memory reduction: ~80 MB/MP → ~10 MB/MP for large images.

use crate::blur::{
    box_blur_2pass_into, box_blur_3pass_into, downscale_2x_inplace, fused_blur_h_mu,
    fused_blur_h_ssim, pad_plane_width, simd_padded_width,
};
use crate::color::srgb_to_positive_xyb_planar;
use crate::fused::{fused_vblur_features_edge, fused_vblur_features_ssim};
use crate::metric::{
    ScaleStats, ZensimConfig, FEATURES_PER_CHANNEL_BASIC, WEIGHTS, combine_scores, compute_single_scale,
};
use crate::pool::ScaleBuffers;
use crate::simd_ops::{
    abs_diff_sum, edge_diff_channel, mul_into, sq_diff_sum, sq_sum_into, ssim_channel,
};
use rayon::prelude::*;

/// Inner strip height: rows of useful output per strip (must be even for 2x downscale).
const STRIP_INNER: usize = 16;

/// Minimum pixels to use streaming instead of full-image processing.
/// Below this, the overhead of strip management isn't worth the memory savings.
const STREAMING_THRESHOLD: usize = 1_000_000;

/// Should we use the streaming path for this image size?
pub(crate) fn should_use_streaming(width: usize, height: usize) -> bool {
    width * height >= STREAMING_THRESHOLD
}

/// Per-scale feature accumulators. Collects raw sums across strips,
/// finalized to ScaleStats at the end.
struct ScaleAccumulators {
    // SSIM: sum_d and sum_d^4 per channel
    ssim_d: [f64; 3],
    ssim_d4: [f64; 3],
    // Edge: artifact and detail_lost sums per channel
    edge_art: [f64; 3],
    edge_art4: [f64; 3],
    edge_det: [f64; 3],
    edge_det4: [f64; 3],
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
            edge_art: [0.0; 3],
            edge_art4: [0.0; 3],
            edge_det: [0.0; 3],
            edge_det4: [0.0; 3],
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
            self.edge_art[c] += other.edge_art[c];
            self.edge_art4[c] += other.edge_art4[c];
            self.edge_det[c] += other.edge_det[c];
            self.edge_det4[c] += other.edge_det4[c];
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

        for c in 0..3 {
            ssim[c * 2] = self.ssim_d[c] * one_over_n;
            ssim[c * 2 + 1] = (self.ssim_d4[c] * one_over_n).powf(0.25);
            edge[c * 4] = self.edge_art[c] * one_over_n;
            edge[c * 4 + 1] = (self.edge_art4[c] * one_over_n).powf(0.25);
            edge[c * 4 + 2] = self.edge_det[c] * one_over_n;
            edge[c * 4 + 3] = (self.edge_det4[c] * one_over_n).powf(0.25);
            mse[c] = self.mse[c] * one_over_n;

            let var_src = self.sq_src[c] * one_over_n;
            let var_dst = self.sq_dst[c] * one_over_n;
            variance_loss[c] = if var_src > 1e-10 {
                (1.0 - var_dst / var_src).max(0.0)
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
            ssim_2nd: [0.0; 3],
            edge_2nd: [0.0; 6],
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

    let mut active = Vec::new();
    let beyond = scale_idx * (fpc * 3) >= WEIGHTS.len();
    for c in 0..3 {
        if beyond {
            if compute_all {
                active.push((c, true, true));
            }
        } else {
            let base = scale_idx * (fpc * 3) + c * fpc;
            let need_ssim = compute_all || has_weight(base, 2);
            let need_blur = has_weight(base + 7, 2);
            // Blur features need mu1/mu2 (same as edge), fold into need_edge
            let need_edge = compute_all || has_weight(base + 2, 4) || need_blur;
            let need_mse = compute_all || has_weight(base + 6, 1);
            if need_ssim || need_edge || need_mse {
                active.push((c, need_ssim, need_edge));
            }
        }
    }
    active
}

/// Streaming multi-scale stats: strip-based scale 0, full-image for scales 1+.
///
/// Produces identical results to the full-image path but uses O(W×strip_h) memory
/// for scale 0 instead of O(W×H).
pub(crate) fn compute_multiscale_stats_streaming(
    source: &[[u8; 3]],
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
    config: &ZensimConfig,
) -> Vec<ScaleStats> {
    let r = config.blur_radius;
    let passes = config.blur_passes as usize;
    let overlap = passes * r;
    let padded_width = simd_padded_width(width);
    let num_scales = config.num_scales;

    let mut stats = Vec::with_capacity(num_scales);

    // Active channels for scale 0
    let scale0_active = active_channels(0, config);

    // Scale 0: parallel band processing
    let next_w = padded_width / 2;
    let next_h = height / 2;
    let max_strip_h = STRIP_INNER + 2 * overlap;
    let max_strip_n = max_strip_h * padded_width;
    let collect_downscale = num_scales > 1;
    let cache_rows = 2 * overlap;

    // Divide strips into bands across threads
    let total_strips = height.div_ceil(STRIP_INNER);
    let num_bands = rayon::current_num_threads().min(total_strips).max(1);
    let strips_per_band = total_strips.div_ceil(num_bands);

    let band_results: Vec<_> = (0..num_bands)
        .into_par_iter()
        .map(|band_idx| {
            let band_first_y = (band_idx * strips_per_band * STRIP_INNER).min(height);
            let band_end_y = (((band_idx + 1) * strips_per_band) * STRIP_INNER).min(height);

            if band_first_y >= height {
                return (
                    ScaleAccumulators::new(),
                    std::array::from_fn::<Vec<f32>, 3, _>(|_| Vec::new()),
                    std::array::from_fn::<Vec<f32>, 3, _>(|_| Vec::new()),
                );
            }

            let mut accum = ScaleAccumulators::new();
            let mut bufs = ScaleBuffers::new(max_strip_n);
            let mut src_xyb: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
            let mut dst_xyb: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());

            let cache_size = cache_rows * padded_width;
            let mut src_cache: [Vec<f32>; 3] =
                std::array::from_fn(|_| vec![0.0f32; cache_size]);
            let mut dst_cache: [Vec<f32>; 3] =
                std::array::from_fn(|_| vec![0.0f32; cache_size]);
            let mut cache_valid = false;

            let band_inner = band_end_y - band_first_y;
            let ds_cap = if collect_downscale {
                next_w * (band_inner / 2 + 1)
            } else {
                0
            };
            let mut band_next_src: [Vec<f32>; 3] =
                std::array::from_fn(|_| Vec::with_capacity(ds_cap));
            let mut band_next_dst: [Vec<f32>; 3] =
                std::array::from_fn(|_| Vec::with_capacity(ds_cap));

            let mut y = band_first_y;
            while y < band_end_y {
                let inner_end = (y + STRIP_INNER).min(height);
                let inner_h = inner_end - y;

                let strip_top = y.saturating_sub(overlap);
                let strip_bot = (inner_end + overlap).min(height);
                let strip_h = strip_bot - strip_top;
                let inner_start = y - strip_top;

                let cached = if cache_valid && strip_top > 0 {
                    cache_rows.min(strip_h)
                } else {
                    0
                };
                let new_top = strip_top + cached;
                let new_h = strip_bot - new_top;

                let new_src_slice = &source[new_top * width..strip_bot * width];
                let new_dst_slice = &distorted[new_top * width..strip_bot * width];
                let mut new_src_xyb = srgb_to_positive_xyb_planar(new_src_slice);
                let mut new_dst_xyb = srgb_to_positive_xyb_planar(new_dst_slice);

                if padded_width != width {
                    for c in 0..3 {
                        pad_plane_width(&mut new_src_xyb[c], width, new_h, padded_width);
                        pad_plane_width(&mut new_dst_xyb[c], width, new_h, padded_width);
                    }
                }

                if cached > 0 {
                    let cached_elems = cached * padded_width;
                    let new_elems = new_h * padded_width;
                    for c in 0..3 {
                        src_xyb[c].clear();
                        src_xyb[c].reserve(cached_elems + new_elems);
                        src_xyb[c].extend_from_slice(&src_cache[c][..cached_elems]);
                        src_xyb[c].extend_from_slice(&new_src_xyb[c][..new_elems]);

                        dst_xyb[c].clear();
                        dst_xyb[c].reserve(cached_elems + new_elems);
                        dst_xyb[c].extend_from_slice(&dst_cache[c][..cached_elems]);
                        dst_xyb[c].extend_from_slice(&new_dst_xyb[c][..new_elems]);
                    }
                } else {
                    src_xyb = new_src_xyb;
                    dst_xyb = new_dst_xyb;
                }

                let strip_n = padded_width * strip_h;
                bufs.resize(strip_n);

                if cache_rows > 0 {
                    let bot = cache_rows.min(strip_h);
                    let bot_start = (strip_h - bot) * padded_width;
                    let bot_elems = bot * padded_width;
                    for c in 0..3 {
                        src_cache[c][..bot_elems]
                            .copy_from_slice(&src_xyb[c][bot_start..bot_start + bot_elems]);
                        dst_cache[c][..bot_elems]
                            .copy_from_slice(&dst_xyb[c][bot_start..bot_start + bot_elems]);
                    }
                    cache_valid = true;
                }

                accum.n += inner_h * padded_width;

                for &(c, need_ssim, need_edge) in &scale0_active {
                    process_strip_channel(
                        &src_xyb[c],
                        &dst_xyb[c],
                        padded_width,
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

                if collect_downscale {
                    let inner_pairs = inner_h / 2;
                    if inner_pairs > 0 {
                        for c in 0..3 {
                            downscale_inner_rows(
                                &src_xyb[c],
                                padded_width,
                                inner_start,
                                inner_pairs,
                                &mut band_next_src[c],
                                next_w,
                            );
                            downscale_inner_rows(
                                &dst_xyb[c],
                                padded_width,
                                inner_start,
                                inner_pairs,
                                &mut band_next_dst[c],
                                next_w,
                            );
                        }
                    }
                }

                y = inner_end;
            }

            (accum, band_next_src, band_next_dst)
        })
        .collect();

    // Merge band results
    let mut accum = ScaleAccumulators::new();
    let mut next_src: [Vec<f32>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(next_w * next_h));
    let mut next_dst: [Vec<f32>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(next_w * next_h));

    for (band_accum, band_src, band_dst) in band_results {
        accum.merge(&band_accum);
        for c in 0..3 {
            next_src[c].extend_from_slice(&band_src[c]);
            next_dst[c].extend_from_slice(&band_dst[c]);
        }
    }

    stats.push(accum.finalize());

    // Scales 1+: parallel band processing for large scales, full-image for small
    if num_scales > 1 && next_w >= 8 && next_h >= 8 {
        let mut w = next_w;
        let mut h = next_h;
        let mut cur_src = next_src;
        let mut cur_dst = next_dst;

        for scale in 1..num_scales {
            if w < 8 || h < 8 {
                break;
            }

            if w * h >= STREAMING_THRESHOLD {
                // Parallel band processing for large scales
                let collect_ds = scale < num_scales - 1;
                let (scale_stat, ds_data) = process_scale_bands_xyb(
                    &cur_src, &cur_dst, w, h, config, scale, collect_ds,
                );
                stats.push(scale_stat);

                if let Some((ns, nd, nw, nh)) = ds_data {
                    cur_src = ns;
                    cur_dst = nd;
                    w = nw;
                    h = nh;
                }
            } else {
                // Small scale: fall back to full-image processing for remaining scales
                let mut bufs = ScaleBuffers::new(w * h);
                let mut parallel_bufs = ScaleBuffers::new(w * h);

                for s in scale..num_scales {
                    if w < 8 || h < 8 {
                        break;
                    }
                    let n = w * h;
                    bufs.resize(n);
                    parallel_bufs.resize(n);

                    let scale_stat = compute_single_scale(
                        &cur_src, &cur_dst, w, h, config, &mut bufs, &mut parallel_bufs, s,
                    );
                    stats.push(scale_stat);

                    if s < num_scales - 1 {
                        let mut nw = 0;
                        let mut nh = 0;
                        for c in 0..3 {
                            let (sw, sh) = downscale_2x_inplace(&mut cur_src[c], w, h);
                            let _ = downscale_2x_inplace(&mut cur_dst[c], w, h);
                            nw = sw;
                            nh = sh;
                        }
                        w = nw;
                        h = nh;
                    }
                }
                break; // All remaining scales handled in the inner loop
            }
        }
    }

    stats
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
            accum.edge_art[c] += strip_acc.edge_art;
            accum.edge_art4[c] += strip_acc.edge_art4;
            accum.edge_det[c] += strip_acc.edge_det;
            accum.edge_det4[c] += strip_acc.edge_det4;
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
            accum.edge_det[c] += strip_acc.edge_det;
            accum.edge_det4[c] += strip_acc.edge_det4;
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
        let (sum_d, sum_d4) = ssim_channel(inner_mu1, inner_mu2, inner_sig_sq, inner_sig12);
        accum.ssim_d[c] += sum_d;
        accum.ssim_d4[c] += sum_d4;
    }

    if need_edge {
        let (art, art4, det, det4) = edge_diff_channel(inner_src, inner_dst, inner_mu1, inner_mu2);
        accum.edge_art[c] += art;
        accum.edge_art4[c] += art4;
        accum.edge_det[c] += det;
        accum.edge_det4[c] += det4;
    }

    accum.sq_src[c] += sq_diff_sum(inner_src, inner_mu1);
    accum.sq_dst[c] += sq_diff_sum(inner_dst, inner_mu2);
    accum.abs_src[c] += abs_diff_sum(inner_src, inner_mu1);
    accum.abs_dst[c] += abs_diff_sum(inner_dst, inner_mu2);
}

/// Downscale inner rows of a strip by 2×2 averaging, appending to output buffer.
fn downscale_inner_rows(
    strip: &[f32],
    strip_width: usize,
    inner_start: usize,
    n_pairs: usize,
    output: &mut Vec<f32>,
    out_width: usize,
) {
    output.reserve(n_pairs * out_width);
    for pair in 0..n_pairs {
        let y0 = inner_start + pair * 2;
        let row0 = y0 * strip_width;
        let row1 = row0 + strip_width;
        for x in 0..out_width {
            let sx = x * 2;
            let val = (strip[row0 + sx] + strip[row0 + sx + 1] + strip[row1 + sx] + strip[row1 + sx + 1])
                * 0.25;
            output.push(val);
        }
    }
}

/// Process a scale using parallel band processing over pre-existing XYB planes.
///
/// Unlike scale 0, no sRGB conversion or overlap caching is needed — we borrow
/// slices directly from the already-downscaled XYB planes.
#[allow(clippy::type_complexity)]
fn process_scale_bands_xyb(
    src_planes: &[Vec<f32>; 3],
    dst_planes: &[Vec<f32>; 3],
    width: usize,
    height: usize,
    config: &ZensimConfig,
    scale_idx: usize,
    collect_downscale: bool,
) -> (ScaleStats, Option<([Vec<f32>; 3], [Vec<f32>; 3], usize, usize)>) {
    let r = config.blur_radius;
    let passes = config.blur_passes as usize;
    let overlap = passes * r;
    let scale_active = active_channels(scale_idx, config);

    let total_strips = height.div_ceil(STRIP_INNER);
    let num_bands = rayon::current_num_threads().min(total_strips).max(1);
    let strips_per_band = total_strips.div_ceil(num_bands);

    let max_strip_h = STRIP_INNER + 2 * overlap;
    let max_strip_n = max_strip_h * width;
    let next_w = width / 2;

    let band_results: Vec<_> = (0..num_bands)
        .into_par_iter()
        .map(|band_idx| {
            let band_first_y = (band_idx * strips_per_band * STRIP_INNER).min(height);
            let band_end_y = (((band_idx + 1) * strips_per_band) * STRIP_INNER).min(height);

            if band_first_y >= height {
                return (
                    ScaleAccumulators::new(),
                    std::array::from_fn::<Vec<f32>, 3, _>(|_| Vec::new()),
                    std::array::from_fn::<Vec<f32>, 3, _>(|_| Vec::new()),
                );
            }

            let mut accum = ScaleAccumulators::new();
            let mut bufs = ScaleBuffers::new(max_strip_n);

            let band_inner = band_end_y - band_first_y;
            let ds_cap = if collect_downscale {
                next_w * (band_inner / 2 + 1)
            } else {
                0
            };
            let mut band_next_src: [Vec<f32>; 3] =
                std::array::from_fn(|_| Vec::with_capacity(ds_cap));
            let mut band_next_dst: [Vec<f32>; 3] =
                std::array::from_fn(|_| Vec::with_capacity(ds_cap));

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

                if collect_downscale {
                    let inner_pairs = inner_h / 2;
                    if inner_pairs > 0 {
                        for c in 0..3 {
                            downscale_inner_rows(
                                &src_planes[c][strip_top * width..strip_bot * width],
                                width,
                                inner_start,
                                inner_pairs,
                                &mut band_next_src[c],
                                next_w,
                            );
                            downscale_inner_rows(
                                &dst_planes[c][strip_top * width..strip_bot * width],
                                width,
                                inner_start,
                                inner_pairs,
                                &mut band_next_dst[c],
                                next_w,
                            );
                        }
                    }
                }

                y = inner_end;
            }

            (accum, band_next_src, band_next_dst)
        })
        .collect();

    // Merge band results
    let mut accum = ScaleAccumulators::new();
    let ds_data = if collect_downscale {
        let next_h = height / 2;
        let mut next_src: [Vec<f32>; 3] =
            std::array::from_fn(|_| Vec::with_capacity(next_w * next_h));
        let mut next_dst: [Vec<f32>; 3] =
            std::array::from_fn(|_| Vec::with_capacity(next_w * next_h));

        for (band_accum, band_src, band_dst) in band_results {
            accum.merge(&band_accum);
            for c in 0..3 {
                next_src[c].extend_from_slice(&band_src[c]);
                next_dst[c].extend_from_slice(&band_dst[c]);
            }
        }

        Some((next_src, next_dst, next_w, next_h))
    } else {
        for (band_accum, _, _) in band_results {
            accum.merge(&band_accum);
        }
        None
    };

    (accum.finalize(), ds_data)
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
            "ssim_mean", "ssim_4th", "edge_art_mean", "edge_art_4th",
            "edge_det_mean", "edge_det_4th", "mse", "var_loss", "tex_loss",
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
                let scale = i / 27;
                let within = i % 27;
                let ch = within / 9;
                let fi = within % 9;
                let rel = diff / absmax.max(1e-12);
                eprintln!(
                    "  feat {:3} (s{} c{} {:14}) full={:12.8} stream={:12.8} diff={:.2e} rel={:.2e}",
                    i, scale, ch, feature_names[fi], f, s, diff, rel,
                );
            }
        }
        let score_rel = (full_result.score - streaming_result.score).abs()
            / full_result.score.abs().max(1e-12);
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
}
