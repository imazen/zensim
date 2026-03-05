//! Parallel multi-scale metric computation with band-based strip processing.
//!
//! Phase 1: Convert sRGB→XYB for the entire image (parallel over row chunks).
//! Phase 2: Process each pyramid scale with parallel band processing.
//!   Strip-based H-blur → fused V-blur+features (parallel bands via rayon).

use crate::blur::{
    box_blur_1pass_into, downscale_2x_inplace, fused_blur_h_mu, fused_blur_h_ssim,
    simd_padded_width,
};
#[cfg(feature = "multiblur")]
use crate::blur::{box_blur_2pass_into, box_blur_3pass_into};
#[cfg(feature = "f16")]
use crate::color::composite_srgb_f16_rgba_to_linear;
use crate::color::{
    apply_gamut_matrix, composite_linear_f32_rgba, composite_srgb8_bgra_to_linear,
    composite_srgb8_rgba_to_linear, composite_srgb16_rgba_to_linear,
    linear_to_positive_xyb_planar_into, srgb_to_positive_xyb_planar_into,
};
use crate::fused::{fused_vblur_features_edge, fused_vblur_features_ssim};
use crate::metric::{
    FEATURES_PER_CHANNEL_BASIC, FEATURES_PER_CHANNEL_EXTENDED, ScaleStats, ZensimConfig,
    combine_scores,
};
use crate::pool::ScaleBuffers;
use crate::simd_ops::{
    abs_diff_into, abs_diff_sum, edge_diff_channel_extended, edge_diff_channel_masked, mul_into,
    sq_diff_sum, sq_sum_into, ssim_channel_extended, ssim_channel_masked,
};
use crate::source::{AlphaMode, ColorPrimaries, ImageSource, PixelFormat};
use rayon::prelude::*;
use std::sync::Mutex;

/// Inner strip height: rows of useful output per strip (must be even for 2x downscale).
const STRIP_INNER: usize = 16;

/// Track background deallocation thread to prevent accumulation on repeated calls.
static DEALLOC_THREAD: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

/// Should we use the streaming path for this image size?
/// Benchmarking shows streaming band processing wins at all sizes (down to 98k pixels),
/// so this always returns true for non-trivial images.
#[cfg(any(feature = "training", test))]
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
    // HF energy (L2) components: sum((pixel-mu)^2) for src and dst
    hf_sq_src: [f64; 3],
    hf_sq_dst: [f64; 3],
    // HF magnitude (L1) components: sum(|pixel-mu|) for src and dst
    hf_abs_src: [f64; 3],
    hf_abs_dst: [f64; 3],
    // Extended: L8 power pool (sum_d^8) for SSIM and edge
    ssim_d8: [f64; 3],
    edge_art8: [f64; 3],
    edge_det8: [f64; 3],
    // Extended: per-channel max values
    ssim_max: [f32; 3],
    edge_art_max: [f32; 3],
    edge_det_max: [f32; 3],
    // Extended: masked features
    masked_ssim_d: [f64; 3],
    masked_ssim_d4: [f64; 3],
    masked_ssim_d2: [f64; 3],
    masked_art4: [f64; 3],
    masked_det4: [f64; 3],
    masked_mse: [f64; 3],
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
            hf_sq_src: [0.0; 3],
            hf_sq_dst: [0.0; 3],
            hf_abs_src: [0.0; 3],
            hf_abs_dst: [0.0; 3],
            ssim_d8: [0.0; 3],
            edge_art8: [0.0; 3],
            edge_det8: [0.0; 3],
            ssim_max: [0.0; 3],
            edge_art_max: [0.0; 3],
            edge_det_max: [0.0; 3],
            masked_ssim_d: [0.0; 3],
            masked_ssim_d4: [0.0; 3],
            masked_ssim_d2: [0.0; 3],
            masked_art4: [0.0; 3],
            masked_det4: [0.0; 3],
            masked_mse: [0.0; 3],
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
            self.hf_sq_src[c] += other.hf_sq_src[c];
            self.hf_sq_dst[c] += other.hf_sq_dst[c];
            self.hf_abs_src[c] += other.hf_abs_src[c];
            self.hf_abs_dst[c] += other.hf_abs_dst[c];
            // Extended
            self.ssim_d8[c] += other.ssim_d8[c];
            self.edge_art8[c] += other.edge_art8[c];
            self.edge_det8[c] += other.edge_det8[c];
            self.ssim_max[c] = self.ssim_max[c].max(other.ssim_max[c]);
            self.edge_art_max[c] = self.edge_art_max[c].max(other.edge_art_max[c]);
            self.edge_det_max[c] = self.edge_det_max[c].max(other.edge_det_max[c]);
            self.masked_ssim_d[c] += other.masked_ssim_d[c];
            self.masked_ssim_d4[c] += other.masked_ssim_d4[c];
            self.masked_ssim_d2[c] += other.masked_ssim_d2[c];
            self.masked_art4[c] += other.masked_art4[c];
            self.masked_det4[c] += other.masked_det4[c];
            self.masked_mse[c] += other.masked_mse[c];
        }
        self.n += other.n;
    }

    fn finalize(&self) -> ScaleStats {
        let one_over_n = 1.0 / self.n as f64;

        let mut ssim = [0.0f64; 6];
        let mut edge = [0.0f64; 12];
        let mut mse = [0.0f64; 3];
        let mut hf_energy_loss = [0.0f64; 3];
        let mut hf_mag_loss = [0.0f64; 3];
        let mut hf_energy_gain = [0.0f64; 3];
        let mut ssim_2nd = [0.0f64; 3];
        let mut edge_2nd = [0.0f64; 6];
        let mut ssim_max = [0.0f64; 3];
        let mut art_max = [0.0f64; 3];
        let mut det_max = [0.0f64; 3];
        let mut ssim_l8 = [0.0f64; 3];
        let mut art_l8 = [0.0f64; 3];
        let mut det_l8 = [0.0f64; 3];
        let mut masked_ssim = [0.0f64; 9];
        let mut masked_art_4th = [0.0f64; 3];
        let mut masked_det_4th = [0.0f64; 3];
        let mut masked_mse = [0.0f64; 3];

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

            let var_src = self.hf_sq_src[c] * one_over_n;
            let var_dst = self.hf_sq_dst[c] * one_over_n;
            hf_energy_loss[c] = if var_src > 1e-10 {
                (1.0 - var_dst / var_src).max(0.0)
            } else {
                0.0
            };
            hf_energy_gain[c] = if var_src > 1e-10 {
                (var_dst / var_src - 1.0).max(0.0)
            } else {
                0.0
            };

            let mad_src = self.hf_abs_src[c] * one_over_n;
            let mad_dst = self.hf_abs_dst[c] * one_over_n;
            hf_mag_loss[c] = if mad_src > 1e-10 {
                (1.0 - mad_dst / mad_src).max(0.0)
            } else {
                0.0
            };

            // Extended: max and L8
            ssim_max[c] = self.ssim_max[c] as f64;
            art_max[c] = self.edge_art_max[c] as f64;
            det_max[c] = self.edge_det_max[c] as f64;
            ssim_l8[c] = (self.ssim_d8[c] * one_over_n).powf(0.125);
            art_l8[c] = (self.edge_art8[c] * one_over_n).powf(0.125);
            det_l8[c] = (self.edge_det8[c] * one_over_n).powf(0.125);

            // Extended: masked features (normalize by N, matching full-image path)
            masked_ssim[c * 3] = self.masked_ssim_d[c] * one_over_n;
            masked_ssim[c * 3 + 1] = (self.masked_ssim_d4[c] * one_over_n).powf(0.25);
            masked_ssim[c * 3 + 2] = (self.masked_ssim_d2[c] * one_over_n).sqrt();
            masked_art_4th[c] = (self.masked_art4[c] * one_over_n).powf(0.25);
            masked_det_4th[c] = (self.masked_det4[c] * one_over_n).powf(0.25);
            masked_mse[c] = self.masked_mse[c] * one_over_n;
        }

        ScaleStats {
            ssim,
            edge,
            mse,
            hf_energy_loss,
            hf_mag_loss,
            hf_energy_gain,
            ssim_2nd,
            edge_2nd,
            ssim_max,
            art_max,
            det_max,
            ssim_p95: ssim_l8,
            art_p95: art_l8,
            det_p95: det_l8,
            masked_ssim,
            masked_art_4th,
            masked_det_4th,
            masked_mse,
            ..Default::default()
        }
    }
}

/// Determine which channels need SSIM, edge, and/or MSE computation at a given scale.
fn active_channels(
    scale_idx: usize,
    config: &ZensimConfig,
    weights: &[f64],
) -> Vec<(usize, bool, bool)> {
    let compute_all = config.compute_all_features;
    let extended = config.extended_features;
    let fpc = if extended {
        FEATURES_PER_CHANNEL_EXTENDED
    } else {
        FEATURES_PER_CHANNEL_BASIC
    };

    let has_weight = |base: usize, count: usize| -> bool {
        (base..base + count).all(|i| i < weights.len())
            && (base..base + count).any(|i| weights[i].abs() > 0.001)
    };

    // Feature layout per channel (13 basic, 25 extended):
    //   0-2: ssim_mean, ssim_4th, ssim_2nd
    //   3-8: art_mean, art_4th, art_2nd, det_mean, det_4th, det_2nd
    //   9: mse, 10-12: hf_energy_loss, hf_mag_loss, hf_energy_gain
    //   13-15: masked_ssim_mean/4th/2nd, 16-17: masked_art_4th/det_4th, 18: masked_mse
    //   19-21: ssim_max/art_max/det_max, 22-24: ssim_l8/art_l8/det_l8
    let mut active = Vec::new();
    let beyond = scale_idx * (fpc * 3) >= weights.len();
    for c in 0..3 {
        if beyond {
            if compute_all || extended {
                active.push((c, true, true));
            }
        } else {
            let base = scale_idx * (fpc * 3) + c * fpc;
            let need_ssim = compute_all || extended || has_weight(base, 3);
            let need_hf = has_weight(base + 10, 3);
            let need_edge = compute_all || extended || has_weight(base + 3, 6) || need_hf;
            let need_mse = compute_all || extended || has_weight(base + 9, 1);
            if need_ssim || need_edge || need_mse {
                active.push((c, need_ssim, need_edge));
            }
        }
    }
    active
}

/// Compute per-channel XYB mean offset: `mean(src) - mean(dst)`.
///
/// Called after XYB conversion completes (planes are cache-hot).
/// Only iterates `width` pixels per row (skipping padding), so the count
/// is exactly `width * height`.
pub(crate) fn compute_xyb_mean_offset(
    src_planes: &[Vec<f32>; 3],
    dst_planes: &[Vec<f32>; 3],
    width: usize,
    height: usize,
    padded_width: usize,
) -> [f64; 3] {
    let mut offset = [0.0f64; 3];
    let n = (width * height) as f64;
    for c in 0..3 {
        let mut src_sum = 0.0f64;
        let mut dst_sum = 0.0f64;
        for y in 0..height {
            let row_start = y * padded_width;
            for x in 0..width {
                src_sum += src_planes[c][row_start + x] as f64;
                dst_sum += dst_planes[c][row_start + x] as f64;
            }
        }
        offset[c] = (src_sum - dst_sum) / n;
    }
    offset
}

/// Streaming multi-scale stats: parallel XYB conversion, then band-parallel blur/features.
///
/// Phase 1: Convert sRGB→XYB for the entire image (parallel over row chunks).
/// Phase 2: Process each scale with parallel band processing over the XYB planes.
///
/// Produces identical results to the full-image path.
pub(crate) fn compute_multiscale_stats_streaming(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
) -> (Vec<ScaleStats>, [f64; 3]) {
    let width = source.width();
    let height = source.height();
    let padded_width = simd_padded_width(width);
    let num_scales = config.num_scales;

    // Phase 1: Convert sRGB→XYB for entire image, parallel over row chunks.
    let mut src_planes = convert_source_to_xyb_parallel(source, padded_width);
    let mut dst_planes = convert_source_to_xyb_parallel(distorted, padded_width);

    // Compute mean_offset while XYB planes are cache-hot
    let mean_offset =
        compute_xyb_mean_offset(&src_planes, &dst_planes, width, height, padded_width);

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
        let scale_stat =
            process_scale_bands(&src_planes, &dst_planes, w, h, config, scale, weights);
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

    (stats, mean_offset)
}

/// Convert an ImageSource to planar XYB at padded width, parallelized over row chunks.
///
/// Handles both RGB and RGBA sources row-by-row. RGBA is composited over checkerboard.
pub(crate) fn convert_source_to_xyb_parallel(
    source: &impl ImageSource,
    padded_width: usize,
) -> [Vec<f32>; 3] {
    let width = source.width();
    let height = source.height();
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

    let pixel_format = source.pixel_format();
    let opaque = matches!(source.alpha_mode(), AlphaMode::Opaque);
    let primaries = source.color_primaries();
    let need_gamut = primaries != ColorPrimaries::Srgb;

    p0_chunks
        .into_par_iter()
        .zip(p1_chunks)
        .zip(p2_chunks)
        .enumerate()
        .for_each(|(chunk_idx, ((c0, c1), c2))| {
            let row_start = chunk_idx * chunk_rows;
            let row_end = (row_start + chunk_rows).min(height);
            let rows = row_end - row_start;

            // Helper: apply gamut matrix to every pixel in a linear row buffer.
            #[inline]
            fn gamut_convert_row(row: &mut [[f32; 3]], primaries: ColorPrimaries) {
                for px in row.iter_mut() {
                    apply_gamut_matrix(px, primaries);
                }
            }

            match pixel_format {
                PixelFormat::Srgb8Rgb => {
                    if need_gamut {
                        // Non-sRGB: linearize via LUT, apply gamut matrix, then XYB
                        let lut = crate::color::srgb_lut();
                        let mut linear_row = vec![[0.0f32; 3]; width];
                        for y in row_start..row_end {
                            let row_bytes = source.row_bytes(y);
                            let rgb_row: &[[u8; 3]] = bytemuck::cast_slice(row_bytes);
                            for (x, pixel) in linear_row.iter_mut().enumerate().take(width) {
                                let [r, g, b] = rgb_row[x];
                                *pixel = [lut[r as usize], lut[g as usize], lut[b as usize]];
                            }
                            gamut_convert_row(&mut linear_row[..width], primaries);
                            let row_offset = (y - row_start) * width;
                            linear_to_positive_xyb_planar_into(
                                &linear_row[..width],
                                &mut c0[row_offset..row_offset + width],
                                &mut c1[row_offset..row_offset + width],
                                &mut c2[row_offset..row_offset + width],
                            );
                        }
                    } else {
                        // sRGB fast path: bulk SIMD conversion
                        let raw_elems = rows * width;
                        let mut rgb_buf: Vec<[u8; 3]> = Vec::with_capacity(raw_elems);
                        for y in row_start..row_end {
                            let row_bytes = source.row_bytes(y);
                            let row: &[[u8; 3]] = bytemuck::cast_slice(row_bytes);
                            rgb_buf.extend_from_slice(&row[..width]);
                        }
                        srgb_to_positive_xyb_planar_into(
                            &rgb_buf,
                            &mut c0[..raw_elems],
                            &mut c1[..raw_elems],
                            &mut c2[..raw_elems],
                        );
                    }
                }
                PixelFormat::Srgb8Rgba => {
                    if opaque && !need_gamut {
                        // Opaque sRGB: ignore alpha byte, extract RGB directly
                        let raw_elems = rows * width;
                        let mut rgb_buf: Vec<[u8; 3]> = Vec::with_capacity(raw_elems);
                        for y in row_start..row_end {
                            let row_bytes = source.row_bytes(y);
                            let rgba_row: &[[u8; 4]] = bytemuck::cast_slice(row_bytes);
                            for &[r, g, b, _a] in &rgba_row[..width] {
                                rgb_buf.push([r, g, b]);
                            }
                        }
                        srgb_to_positive_xyb_planar_into(
                            &rgb_buf,
                            &mut c0[..raw_elems],
                            &mut c1[..raw_elems],
                            &mut c2[..raw_elems],
                        );
                    } else {
                        let mut linear_row = vec![[0.0f32; 3]; width];
                        for y in row_start..row_end {
                            let row_bytes = source.row_bytes(y);
                            let rgba_row: &[[u8; 4]] = bytemuck::cast_slice(row_bytes);
                            if opaque {
                                // Opaque non-sRGB: linearize + gamut
                                let lut = crate::color::srgb_lut();
                                for (x, pixel) in linear_row.iter_mut().enumerate().take(width) {
                                    let [r, g, b, _a] = rgba_row[x];
                                    *pixel = [lut[r as usize], lut[g as usize], lut[b as usize]];
                                }
                            } else {
                                composite_srgb8_rgba_to_linear(
                                    &rgba_row[..width],
                                    y,
                                    &mut linear_row,
                                );
                            }
                            if need_gamut {
                                gamut_convert_row(&mut linear_row[..width], primaries);
                            }
                            let row_offset = (y - row_start) * width;
                            linear_to_positive_xyb_planar_into(
                                &linear_row[..width],
                                &mut c0[row_offset..row_offset + width],
                                &mut c1[row_offset..row_offset + width],
                                &mut c2[row_offset..row_offset + width],
                            );
                        }
                    }
                }
                PixelFormat::Srgb8Bgra => {
                    if opaque && !need_gamut {
                        let raw_elems = rows * width;
                        let mut rgb_buf: Vec<[u8; 3]> = Vec::with_capacity(raw_elems);
                        for y in row_start..row_end {
                            let row_bytes = source.row_bytes(y);
                            let bgra_row: &[[u8; 4]] = bytemuck::cast_slice(row_bytes);
                            for &[b, g, r, _a] in &bgra_row[..width] {
                                rgb_buf.push([r, g, b]);
                            }
                        }
                        srgb_to_positive_xyb_planar_into(
                            &rgb_buf,
                            &mut c0[..raw_elems],
                            &mut c1[..raw_elems],
                            &mut c2[..raw_elems],
                        );
                    } else {
                        let mut linear_row = vec![[0.0f32; 3]; width];
                        for y in row_start..row_end {
                            let row_bytes = source.row_bytes(y);
                            let bgra_row: &[[u8; 4]] = bytemuck::cast_slice(row_bytes);
                            if opaque {
                                // Opaque non-sRGB: linearize + gamut
                                let lut = crate::color::srgb_lut();
                                for (x, pixel) in linear_row.iter_mut().enumerate().take(width) {
                                    let [b, g, r, _a] = bgra_row[x];
                                    *pixel = [lut[r as usize], lut[g as usize], lut[b as usize]];
                                }
                            } else {
                                composite_srgb8_bgra_to_linear(
                                    &bgra_row[..width],
                                    y,
                                    &mut linear_row,
                                );
                            }
                            if need_gamut {
                                gamut_convert_row(&mut linear_row[..width], primaries);
                            }
                            let row_offset = (y - row_start) * width;
                            linear_to_positive_xyb_planar_into(
                                &linear_row[..width],
                                &mut c0[row_offset..row_offset + width],
                                &mut c1[row_offset..row_offset + width],
                                &mut c2[row_offset..row_offset + width],
                            );
                        }
                    }
                }
                PixelFormat::Srgb16Rgba => {
                    let mut linear_row = vec![[0.0f32; 3]; width];
                    for y in row_start..row_end {
                        let row_bytes = source.row_bytes(y);
                        if opaque {
                            // Opaque: linearize RGB, ignore alpha
                            for (x, pixel) in linear_row.iter_mut().enumerate().take(width) {
                                let off = x * 8;
                                let r = u16::from_ne_bytes([row_bytes[off], row_bytes[off + 1]]);
                                let g =
                                    u16::from_ne_bytes([row_bytes[off + 2], row_bytes[off + 3]]);
                                let b =
                                    u16::from_ne_bytes([row_bytes[off + 4], row_bytes[off + 5]]);
                                *pixel = [
                                    crate::color::srgb_u16_to_linear(r),
                                    crate::color::srgb_u16_to_linear(g),
                                    crate::color::srgb_u16_to_linear(b),
                                ];
                            }
                        } else {
                            composite_srgb16_rgba_to_linear(row_bytes, width, y, &mut linear_row);
                        }
                        if need_gamut {
                            gamut_convert_row(&mut linear_row[..width], primaries);
                        }
                        let row_offset = (y - row_start) * width;
                        linear_to_positive_xyb_planar_into(
                            &linear_row,
                            &mut c0[row_offset..row_offset + width],
                            &mut c1[row_offset..row_offset + width],
                            &mut c2[row_offset..row_offset + width],
                        );
                    }
                }
                #[cfg(feature = "f16")]
                PixelFormat::SrgbF16Rgba => {
                    let mut linear_row = vec![[0.0f32; 3]; width];
                    for y in row_start..row_end {
                        let row_bytes = source.row_bytes(y);
                        if opaque {
                            for (x, pixel) in linear_row.iter_mut().enumerate().take(width) {
                                let off = x * 8;
                                let r =
                                    crate::color::srgb_f16_to_linear(half::f16::from_ne_bytes([
                                        row_bytes[off],
                                        row_bytes[off + 1],
                                    ]));
                                let g =
                                    crate::color::srgb_f16_to_linear(half::f16::from_ne_bytes([
                                        row_bytes[off + 2],
                                        row_bytes[off + 3],
                                    ]));
                                let b =
                                    crate::color::srgb_f16_to_linear(half::f16::from_ne_bytes([
                                        row_bytes[off + 4],
                                        row_bytes[off + 5],
                                    ]));
                                *pixel = [r, g, b];
                            }
                        } else {
                            composite_srgb_f16_rgba_to_linear(row_bytes, width, y, &mut linear_row);
                        }
                        if need_gamut {
                            gamut_convert_row(&mut linear_row[..width], primaries);
                        }
                        let row_offset = (y - row_start) * width;
                        linear_to_positive_xyb_planar_into(
                            &linear_row,
                            &mut c0[row_offset..row_offset + width],
                            &mut c1[row_offset..row_offset + width],
                            &mut c2[row_offset..row_offset + width],
                        );
                    }
                }
                PixelFormat::LinearF32Rgba => {
                    if opaque && !need_gamut {
                        // Opaque sRGB: extract RGB from f32 RGBA, skip alpha
                        let raw_elems = rows * width;
                        let mut rgb_buf: Vec<[f32; 3]> = Vec::with_capacity(raw_elems);
                        for y in row_start..row_end {
                            let row_bytes = source.row_bytes(y);
                            let rgba_row: &[[f32; 4]] = bytemuck::cast_slice(row_bytes);
                            for &[r, g, b, _a] in &rgba_row[..width] {
                                rgb_buf.push([r, g, b]);
                            }
                        }
                        linear_to_positive_xyb_planar_into(
                            &rgb_buf,
                            &mut c0[..raw_elems],
                            &mut c1[..raw_elems],
                            &mut c2[..raw_elems],
                        );
                    } else {
                        let mut linear_row = vec![[0.0f32; 3]; width];
                        for y in row_start..row_end {
                            let row_bytes = source.row_bytes(y);
                            let rgba_row: &[[f32; 4]] = bytemuck::cast_slice(row_bytes);
                            if opaque {
                                // Opaque non-sRGB: extract RGB + gamut
                                for (x, pixel) in linear_row.iter_mut().enumerate().take(width) {
                                    let [r, g, b, _a] = rgba_row[x];
                                    *pixel = [r, g, b];
                                }
                            } else {
                                composite_linear_f32_rgba(&rgba_row[..width], y, &mut linear_row);
                            }
                            if need_gamut {
                                gamut_convert_row(&mut linear_row[..width], primaries);
                            }
                            let row_offset = (y - row_start) * width;
                            linear_to_positive_xyb_planar_into(
                                &linear_row,
                                &mut c0[row_offset..row_offset + width],
                                &mut c1[row_offset..row_offset + width],
                                &mut c2[row_offset..row_offset + width],
                            );
                        }
                    }
                }
                // PixelFormat is #[non_exhaustive] so downstream crates need this arm,
                // but within this crate all variants are handled above.
                #[allow(unreachable_patterns)]
                other => panic!(
                    "zensim: unsupported pixel format {:?} in XYB conversion",
                    other
                ),
            }

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

    // Fused path: 1-pass blur (the common case for scale 0).
    // d8/max and mu1/mu2 are now computed inline by the fused kernels.
    if config.blur_passes == 1 {
        let strip_acc;

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

            // Fused V-blur + ALL feature extraction
            // mu1/mu2 outputs go to mask/mul_buf (mu1/mu2 still hold H-blurred values)
            strip_acc = fused_vblur_features_ssim(
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
                &mut bufs.mask,
                &mut bufs.mul_buf,
                config.extended_features,
            );

            accum.ssim_d[c] += strip_acc.ssim_d;
            accum.ssim_d4[c] += strip_acc.ssim_d4;
            accum.ssim_d2[c] += strip_acc.ssim_d2;
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

            strip_acc = fused_vblur_features_edge(
                &bufs.mu1,
                &bufs.mu2,
                src_c,
                dst_c,
                width,
                strip_h,
                inner_start,
                inner_h,
                config.blur_radius,
                &mut bufs.mask,
                &mut bufs.mul_buf,
                config.extended_features,
            );
        }

        // Swap: mu1/mu2 now hold V-blurred values, mask/mul_buf hold H-blurred garbage
        if config.extended_features {
            std::mem::swap(&mut bufs.mu1, &mut bufs.mask);
            std::mem::swap(&mut bufs.mu2, &mut bufs.mul_buf);
        }

        // Accumulate basic features
        accum.edge_art[c] += strip_acc.edge_art;
        accum.edge_art4[c] += strip_acc.edge_art4;
        accum.edge_art2[c] += strip_acc.edge_art2;
        accum.edge_det[c] += strip_acc.edge_det;
        accum.edge_det4[c] += strip_acc.edge_det4;
        accum.edge_det2[c] += strip_acc.edge_det2;
        accum.mse[c] += strip_acc.mse;
        accum.hf_sq_src[c] += strip_acc.hf_sq_src;
        accum.hf_sq_dst[c] += strip_acc.hf_sq_dst;
        accum.hf_abs_src[c] += strip_acc.hf_abs_src;
        accum.hf_abs_dst[c] += strip_acc.hf_abs_dst;

        // Extended: d8/max from fused kernel
        accum.ssim_d8[c] += strip_acc.ssim_d8;
        accum.ssim_max[c] = accum.ssim_max[c].max(strip_acc.ssim_max);
        accum.edge_art8[c] += strip_acc.edge_art8;
        accum.edge_art_max[c] = accum.edge_art_max[c].max(strip_acc.edge_art_max);
        accum.edge_det8[c] += strip_acc.edge_det8;
        accum.edge_det_max[c] = accum.edge_det_max[c].max(strip_acc.edge_det_max);

        // Extended: masked features using stored V-blurred mu1/mu2
        if config.extended_features {
            let inner_off = inner_start * width;
            let inner_n = inner_h * width;
            let strip_n = strip_h * width;
            let k = config.extended_masking_strength;
            let inner_src = &src_c[inner_off..inner_off + inner_n];
            let inner_dst = &dst_c[inner_off..inner_off + inner_n];

            // Step 1: |src - mu1| → mask (full strip for blur context)
            abs_diff_into(
                &src_c[..strip_n],
                &bufs.mu1[..strip_n],
                &mut bufs.mask[..strip_n],
            );

            // Step 2: blur the activity map → mul_buf
            box_blur_1pass_into(
                &bufs.mask[..strip_n],
                &mut bufs.mul_buf[..strip_n],
                &mut bufs.temp_blur[..strip_n],
                width,
                strip_h,
                config.blur_radius,
            );

            // Step 3: mask = 1/(1+k*blurred) for inner region
            for i in 0..inner_n {
                bufs.mask[inner_off + i] = 1.0 / (1.0 + k * bufs.mul_buf[inner_off + i]);
            }

            // Step 4: masked SSIM (needs sigma_sq and sigma12 — recompute from fused H-blur)
            if need_ssim {
                // sigma_sq and sigma12 are NOT stored from fused pass — recompute
                sq_sum_into(src_c, dst_c, &mut bufs.mul_buf);
                box_blur_1pass_into(
                    &bufs.mul_buf,
                    &mut bufs.sigma1_sq,
                    &mut bufs.temp_blur,
                    width,
                    strip_h,
                    config.blur_radius,
                );
                mul_into(src_c, dst_c, &mut bufs.mul_buf);
                box_blur_1pass_into(
                    &bufs.mul_buf,
                    &mut bufs.sigma12,
                    &mut bufs.temp_blur,
                    width,
                    strip_h,
                    config.blur_radius,
                );

                let inner_mask = &bufs.mask[inner_off..inner_off + inner_n];
                let inner_mu1 = &bufs.mu1[inner_off..inner_off + inner_n];
                let inner_mu2 = &bufs.mu2[inner_off..inner_off + inner_n];
                let inner_sig_sq = &bufs.sigma1_sq[inner_off..inner_off + inner_n];
                let inner_sig12 = &bufs.sigma12[inner_off..inner_off + inner_n];
                let (sum_d, sum_d4, sum_d2) = ssim_channel_masked(
                    inner_mu1,
                    inner_mu2,
                    inner_sig_sq,
                    inner_sig12,
                    inner_mask,
                );
                accum.masked_ssim_d[c] += sum_d;
                accum.masked_ssim_d4[c] += sum_d4;
                accum.masked_ssim_d2[c] += sum_d2;
            }

            // Masked edge (art_4th, det_4th)
            if need_edge {
                let inner_mask = &bufs.mask[inner_off..inner_off + inner_n];
                let inner_mu1 = &bufs.mu1[inner_off..inner_off + inner_n];
                let inner_mu2 = &bufs.mu2[inner_off..inner_off + inner_n];
                let (_art, art4, _det, det4, _art2, _det2) = edge_diff_channel_masked(
                    inner_src, inner_dst, inner_mu1, inner_mu2, inner_mask,
                );
                accum.masked_art4[c] += art4;
                accum.masked_det4[c] += det4;
            }

            // Masked MSE: Σ(src-dst)² × mask
            let mut masked_mse_sum = 0.0f64;
            for i in 0..inner_n {
                let d = inner_src[i] - inner_dst[i];
                masked_mse_sum += (d * d * bufs.mask[inner_off + i]) as f64;
            }
            accum.masked_mse[c] += masked_mse_sum;
        }
        return;
    }

    // Separate blur + reduce fallback (multi-pass only, requires 'multiblur' feature)
    #[allow(clippy::type_complexity)]
    let blur_fn: fn(&[f32], &mut [f32], &mut [f32], usize, usize, usize) = match config.blur_passes
    {
        1 => box_blur_1pass_into,
        #[cfg(feature = "multiblur")]
        2 => box_blur_2pass_into,
        #[cfg(feature = "multiblur")]
        _ => box_blur_3pass_into,
        #[cfg(not(feature = "multiblur"))]
        _ => {
            debug_assert_eq!(
                config.blur_passes, 1,
                "blur_passes > 1 requires 'multiblur' feature"
            );
            box_blur_1pass_into
        }
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
        let (sum_d, sum_d4, sum_d2, sum_d8, max_d) =
            ssim_channel_extended(inner_mu1, inner_mu2, inner_sig_sq, inner_sig12);
        accum.ssim_d[c] += sum_d;
        accum.ssim_d4[c] += sum_d4;
        accum.ssim_d2[c] += sum_d2;
        accum.ssim_d8[c] += sum_d8;
        accum.ssim_max[c] = accum.ssim_max[c].max(max_d);
    }

    if need_edge {
        let (art, art4, det, det4, art2, det2, art8, det8, max_art, max_det) =
            edge_diff_channel_extended(inner_src, inner_dst, inner_mu1, inner_mu2);
        accum.edge_art[c] += art;
        accum.edge_art4[c] += art4;
        accum.edge_art2[c] += art2;
        accum.edge_art8[c] += art8;
        accum.edge_det[c] += det;
        accum.edge_det4[c] += det4;
        accum.edge_det2[c] += det2;
        accum.edge_det8[c] += det8;
        accum.edge_art_max[c] = accum.edge_art_max[c].max(max_art);
        accum.edge_det_max[c] = accum.edge_det_max[c].max(max_det);
    }

    accum.hf_sq_src[c] += sq_diff_sum(inner_src, inner_mu1);
    accum.hf_sq_dst[c] += sq_diff_sum(inner_dst, inner_mu2);
    accum.hf_abs_src[c] += abs_diff_sum(inner_src, inner_mu1);
    accum.hf_abs_dst[c] += abs_diff_sum(inner_dst, inner_mu2);

    // Extended: masked features
    if config.extended_features {
        let strip_n = strip_h * width;
        let k = config.extended_masking_strength;

        abs_diff_into(
            &src_c[..strip_n],
            &bufs.mu1[..strip_n],
            &mut bufs.mask[..strip_n],
        );

        blur_fn(
            &bufs.mask[..strip_n],
            &mut bufs.mul_buf[..strip_n],
            &mut bufs.temp_blur[..strip_n],
            width,
            strip_h,
            config.blur_radius,
        );

        for i in 0..inner_n {
            bufs.mask[inner_off + i] = 1.0 / (1.0 + k * bufs.mul_buf[inner_off + i]);
        }

        if need_ssim {
            let inner_mask = &bufs.mask[inner_off..inner_off + inner_n];
            let inner_mu1 = &bufs.mu1[inner_off..inner_off + inner_n];
            let inner_mu2 = &bufs.mu2[inner_off..inner_off + inner_n];
            let inner_sig_sq = &bufs.sigma1_sq[inner_off..inner_off + inner_n];
            let inner_sig12 = &bufs.sigma12[inner_off..inner_off + inner_n];
            let (sum_d, sum_d4, sum_d2) =
                ssim_channel_masked(inner_mu1, inner_mu2, inner_sig_sq, inner_sig12, inner_mask);
            accum.masked_ssim_d[c] += sum_d;
            accum.masked_ssim_d4[c] += sum_d4;
            accum.masked_ssim_d2[c] += sum_d2;
        }

        if need_edge {
            let inner_mask = &bufs.mask[inner_off..inner_off + inner_n];
            let (_art, art4, _det, det4, _art2, _det2) =
                edge_diff_channel_masked(inner_src, inner_dst, inner_mu1, inner_mu2, inner_mask);
            accum.masked_art4[c] += art4;
            accum.masked_det4[c] += det4;
        }

        let mut masked_mse_sum = 0.0f64;
        for i in 0..inner_n {
            let d = inner_src[i] - inner_dst[i];
            masked_mse_sum += (d * d * bufs.mask[inner_off + i]) as f64;
        }
        accum.masked_mse[c] += masked_mse_sum;
    }
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
    weights: &[f64],
) -> ScaleStats {
    let r = config.blur_radius;
    let passes = config.blur_passes as usize;
    let overlap = passes * r;
    let scale_active = active_channels(scale_idx, config, weights);

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
/// Created via [`Zensim::precompute_reference`](crate::Zensim::precompute_reference).
pub struct PrecomputedReference {
    pub(crate) scales: Vec<([Vec<f32>; 3], usize, usize)>,
}

impl PrecomputedReference {
    /// Build a precomputed reference from an ImageSource.
    ///
    /// Converts to XYB and builds the downscale pyramid, storing planes at each level.
    pub(crate) fn new(source: &impl ImageSource, num_scales: usize) -> Self {
        let width = source.width();
        let height = source.height();
        let padded_width = simd_padded_width(width);
        let mut planes = convert_source_to_xyb_parallel(source, padded_width);

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
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
) -> (Vec<ScaleStats>, [f64; 3]) {
    let width = distorted.width();
    let height = distorted.height();
    let padded_width = simd_padded_width(width);
    let num_scales = config.num_scales.min(precomputed.scales.len());

    // Only convert distorted to XYB
    let mut dst_planes = convert_source_to_xyb_parallel(distorted, padded_width);

    // Compute mean_offset using scale-0 reference planes and fresh distorted planes
    let (ref src_planes_s0, _, _) = precomputed.scales[0];
    let mean_offset =
        compute_xyb_mean_offset(src_planes_s0, &dst_planes, width, height, padded_width);

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

        let scale_stat = process_scale_bands(src_planes, &dst_planes, w, h, config, scale, weights);
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

    (stats, mean_offset)
}

/// Entry point: compute zensim using streaming with precomputed reference.
/// Produces identical results to the non-precomputed path.
pub(crate) fn compute_zensim_streaming_with_ref(
    precomputed: &PrecomputedReference,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
) -> crate::metric::ZensimResult {
    let (scale_stats, mean_offset) =
        compute_multiscale_stats_streaming_with_ref(precomputed, distorted, config, weights);
    let masked = config.masking_strength > 0.0;
    combine_scores(&scale_stats, masked, weights, config, mean_offset)
}

/// Entry point: compute zensim using streaming for scale 0, full-image for the rest.
/// Produces identical results to the full-image path.
pub(crate) fn compute_zensim_streaming(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
) -> crate::metric::ZensimResult {
    let (scale_stats, mean_offset) =
        compute_multiscale_stats_streaming(source, distorted, config, weights);
    let masked = config.masking_strength > 0.0;
    combine_scores(&scale_stats, masked, weights, config, mean_offset)
}

// ─── Delta stats computation ─────────────────────────────────────────────

use crate::metric::{AlphaStratifiedStats, DeltaStats};

/// Per-chunk accumulator for delta stats (merged across parallel chunks).
struct DeltaAccum {
    // Per-channel accumulators
    sum_delta: [f64; 3],
    sum_delta_sq: [f64; 3],
    max_abs_delta: [f64; 3],
    // Signed small-delta histogram: bins -3..+3, index = delta + 3
    signed_small: [[u64; 7]; 3],
    // Pixel counts
    pixel_count: u64,
    pixels_differing: u64,
    pixels_differing_by_more_than_1: u64,
    // Alpha-stratified (RGBA only)
    opaque_count: u64,
    opaque_sum_abs: [f64; 3],
    opaque_max_abs: [f64; 3],
    semi_count: u64,
    semi_sum_abs: [f64; 3],
    semi_max_abs: [f64; 3],
    // Alpha channel delta tracking
    alpha_max_delta: u8,         // max |src_alpha - dst_alpha| in 0-255 units
    alpha_pixels_differing: u64, // pixels where alpha differs at all
    // Per-channel value histograms (256 bins, 4 channels: R, G, B, A)
    // Boxed to avoid 16KB on the stack per parallel chunk.
    src_histogram: Box<[[u64; 256]; 4]>,
    dst_histogram: Box<[[u64; 256]; 4]>,
    // For alpha-error correlation (Pearson)
    sum_delta_mag: f64,       // sum of per-pixel max(|delta[c]|)
    sum_one_minus_alpha: f64, // sum of (1 - alpha/255)
    sum_delta_alpha: f64,     // sum of max(|delta|) * (1 - alpha/255)
    sum_delta_mag_sq: f64,
    sum_one_minus_alpha_sq: f64,
    alpha_pixel_count: u64,
}

impl DeltaAccum {
    fn new() -> Self {
        Self {
            sum_delta: [0.0; 3],
            sum_delta_sq: [0.0; 3],
            max_abs_delta: [0.0; 3],
            signed_small: [[0u64; 7]; 3],
            pixel_count: 0,
            pixels_differing: 0,
            pixels_differing_by_more_than_1: 0,
            opaque_count: 0,
            opaque_sum_abs: [0.0; 3],
            opaque_max_abs: [0.0; 3],
            semi_count: 0,
            semi_sum_abs: [0.0; 3],
            semi_max_abs: [0.0; 3],
            alpha_max_delta: 0,
            alpha_pixels_differing: 0,
            src_histogram: Box::new([[0u64; 256]; 4]),
            dst_histogram: Box::new([[0u64; 256]; 4]),
            sum_delta_mag: 0.0,
            sum_one_minus_alpha: 0.0,
            sum_delta_alpha: 0.0,
            sum_delta_mag_sq: 0.0,
            sum_one_minus_alpha_sq: 0.0,
            alpha_pixel_count: 0,
        }
    }

    fn merge(&mut self, other: &Self) {
        for c in 0..3 {
            self.sum_delta[c] += other.sum_delta[c];
            self.sum_delta_sq[c] += other.sum_delta_sq[c];
            self.max_abs_delta[c] = self.max_abs_delta[c].max(other.max_abs_delta[c]);
            for b in 0..7 {
                self.signed_small[c][b] += other.signed_small[c][b];
            }
            self.opaque_sum_abs[c] += other.opaque_sum_abs[c];
            self.opaque_max_abs[c] = self.opaque_max_abs[c].max(other.opaque_max_abs[c]);
            self.semi_sum_abs[c] += other.semi_sum_abs[c];
            self.semi_max_abs[c] = self.semi_max_abs[c].max(other.semi_max_abs[c]);
        }
        self.pixel_count += other.pixel_count;
        self.pixels_differing += other.pixels_differing;
        self.pixels_differing_by_more_than_1 += other.pixels_differing_by_more_than_1;
        self.opaque_count += other.opaque_count;
        self.semi_count += other.semi_count;
        self.alpha_max_delta = self.alpha_max_delta.max(other.alpha_max_delta);
        self.alpha_pixels_differing += other.alpha_pixels_differing;
        for c in 0..4 {
            for b in 0..256 {
                self.src_histogram[c][b] += other.src_histogram[c][b];
                self.dst_histogram[c][b] += other.dst_histogram[c][b];
            }
        }
        self.sum_delta_mag += other.sum_delta_mag;
        self.sum_one_minus_alpha += other.sum_one_minus_alpha;
        self.sum_delta_alpha += other.sum_delta_alpha;
        self.sum_delta_mag_sq += other.sum_delta_mag_sq;
        self.sum_one_minus_alpha_sq += other.sum_one_minus_alpha_sq;
        self.alpha_pixel_count += other.alpha_pixel_count;
    }
}

/// Compute pixel-level delta statistics between two images.
///
/// Single parallel pass over both images. Operates in sRGB u8 space
/// (values normalized to [0, 1]) for all sRGB formats. Linear formats
/// are compared in linear space.
/// Derive the native maximum value for a pixel format.
///
/// 255.0 for u8 formats, 65535.0 for u16, 1.0 for f32/f16.
fn native_max_for_format(format: PixelFormat) -> f64 {
    match format {
        PixelFormat::Srgb8Rgb | PixelFormat::Srgb8Rgba | PixelFormat::Srgb8Bgra => 255.0,
        PixelFormat::Srgb16Rgba => 65535.0,
        PixelFormat::SrgbF16Rgba | PixelFormat::LinearF32Rgba => 1.0,
        #[allow(unreachable_patterns)]
        _ => 255.0,
    }
}

pub(crate) fn compute_delta_stats(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> DeltaStats {
    let width = source.width();
    let height = source.height();
    let src_format = source.pixel_format();
    let dst_format = distorted.pixel_format();
    let has_alpha = src_format.has_alpha() && dst_format.has_alpha();
    let native_max = native_max_for_format(src_format).max(native_max_for_format(dst_format));

    let chunk_rows = 64usize;
    let num_chunks = height.div_ceil(chunk_rows);

    // Parallel accumulation over row chunks
    let accum = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let mut acc = DeltaAccum::new();
            let row_start = chunk_idx * chunk_rows;
            let row_end = (row_start + chunk_rows).min(height);

            for y in row_start..row_end {
                let src_bytes = source.row_bytes(y);
                let dst_bytes = distorted.row_bytes(y);

                for x in 0..width {
                    // Extract normalized [0,1] RGB values per image format
                    let (src_rgb, src_alpha) = extract_pixel_normalized(src_bytes, x, src_format);
                    let (dst_rgb, dst_alpha) = extract_pixel_normalized(dst_bytes, x, dst_format);
                    let alpha = if has_alpha {
                        // Both formats have alpha — zip them
                        Some((src_alpha.unwrap_or(1.0), dst_alpha.unwrap_or(1.0)))
                    } else {
                        None
                    };

                    let mut any_diff = false;
                    let mut any_diff_gt1 = false;
                    let mut pixel_max_abs_delta = 0.0f64;

                    for c in 0..3 {
                        let delta = src_rgb[c] - dst_rgb[c];
                        let abs_delta = delta.abs();

                        acc.sum_delta[c] += delta;
                        acc.sum_delta_sq[c] += delta * delta;
                        if abs_delta > acc.max_abs_delta[c] {
                            acc.max_abs_delta[c] = abs_delta;
                        }

                        // Signed small-delta histogram: bins -3..+3 in 1/native_max units
                        let signed_delta = (delta * native_max).round() as i32;
                        if (-3..=3).contains(&signed_delta) {
                            acc.signed_small[c][(signed_delta + 3) as usize] += 1;
                        }

                        if abs_delta > 0.5 / native_max {
                            any_diff = true;
                        }
                        if abs_delta > 1.5 / native_max {
                            any_diff_gt1 = true;
                        }

                        if abs_delta > pixel_max_abs_delta {
                            pixel_max_abs_delta = abs_delta;
                        }
                    }

                    // Per-channel value histograms (always 256 bins)
                    for c in 0..3 {
                        let sb = (src_rgb[c] * 255.0).round().clamp(0.0, 255.0) as usize;
                        let db = (dst_rgb[c] * 255.0).round().clamp(0.0, 255.0) as usize;
                        acc.src_histogram[c][sb] += 1;
                        acc.dst_histogram[c][db] += 1;
                    }
                    if let Some((src_a, dst_a)) = alpha {
                        let sb = (src_a * 255.0).round().clamp(0.0, 255.0) as usize;
                        let db = (dst_a * 255.0).round().clamp(0.0, 255.0) as usize;
                        acc.src_histogram[3][sb] += 1;
                        acc.dst_histogram[3][db] += 1;
                    }

                    acc.pixel_count += 1;
                    if any_diff {
                        acc.pixels_differing += 1;
                    }
                    if any_diff_gt1 {
                        acc.pixels_differing_by_more_than_1 += 1;
                    }

                    // Alpha stratification and alpha delta tracking
                    if has_alpha && let Some((src_a, dst_a)) = alpha {
                        // Track alpha channel delta at native precision
                        let alpha_delta = ((src_a - dst_a).abs() * native_max).round() as u8;
                        if alpha_delta > acc.alpha_max_delta {
                            acc.alpha_max_delta = alpha_delta;
                        }
                        if alpha_delta > 0 {
                            acc.alpha_pixels_differing += 1;
                        }

                        let a = src_a;
                        let one_minus_a = 1.0 - a;
                        if a >= 1.0 - 0.5 / native_max {
                            // Opaque
                            acc.opaque_count += 1;
                            for c in 0..3 {
                                let ad = (src_rgb[c] - dst_rgb[c]).abs();
                                acc.opaque_sum_abs[c] += ad;
                                if ad > acc.opaque_max_abs[c] {
                                    acc.opaque_max_abs[c] = ad;
                                }
                            }
                        } else if a > 0.5 / native_max {
                            // Semitransparent
                            acc.semi_count += 1;
                            for c in 0..3 {
                                let ad = (src_rgb[c] - dst_rgb[c]).abs();
                                acc.semi_sum_abs[c] += ad;
                                if ad > acc.semi_max_abs[c] {
                                    acc.semi_max_abs[c] = ad;
                                }
                            }
                        }

                        // Pearson correlation accumulators
                        acc.sum_delta_mag += pixel_max_abs_delta;
                        acc.sum_one_minus_alpha += one_minus_a;
                        acc.sum_delta_alpha += pixel_max_abs_delta * one_minus_a;
                        acc.sum_delta_mag_sq += pixel_max_abs_delta * pixel_max_abs_delta;
                        acc.sum_one_minus_alpha_sq += one_minus_a * one_minus_a;
                        acc.alpha_pixel_count += 1;
                    }
                }
            }
            acc
        })
        .reduce(DeltaAccum::new, |mut a, b| {
            a.merge(&b);
            a
        });

    finalize_delta_stats(accum, has_alpha, native_max)
}

/// Extract normalized \[0,1\] RGB values and optional alpha from a single pixel
/// at position `x` in `row_bytes`, interpreting bytes according to `format`.
#[inline]
fn extract_pixel_normalized(
    row_bytes: &[u8],
    x: usize,
    format: PixelFormat,
) -> ([f64; 3], Option<f64>) {
    match format {
        PixelFormat::Srgb8Rgb => {
            let off = x * 3;
            let rgb = [
                row_bytes[off] as f64 / 255.0,
                row_bytes[off + 1] as f64 / 255.0,
                row_bytes[off + 2] as f64 / 255.0,
            ];
            (rgb, None)
        }
        PixelFormat::Srgb8Rgba => {
            let off = x * 4;
            let rgb = [
                row_bytes[off] as f64 / 255.0,
                row_bytes[off + 1] as f64 / 255.0,
                row_bytes[off + 2] as f64 / 255.0,
            ];
            let a = row_bytes[off + 3] as f64 / 255.0;
            (rgb, Some(a))
        }
        PixelFormat::Srgb8Bgra => {
            let off = x * 4;
            let rgb = [
                row_bytes[off + 2] as f64 / 255.0, // R
                row_bytes[off + 1] as f64 / 255.0, // G
                row_bytes[off] as f64 / 255.0,     // B
            ];
            let a = row_bytes[off + 3] as f64 / 255.0;
            (rgb, Some(a))
        }
        PixelFormat::Srgb16Rgba => {
            let off = x * 8;
            let rgb = [
                u16::from_ne_bytes([row_bytes[off], row_bytes[off + 1]]) as f64 / 65535.0,
                u16::from_ne_bytes([row_bytes[off + 2], row_bytes[off + 3]]) as f64 / 65535.0,
                u16::from_ne_bytes([row_bytes[off + 4], row_bytes[off + 5]]) as f64 / 65535.0,
            ];
            let a = u16::from_ne_bytes([row_bytes[off + 6], row_bytes[off + 7]]) as f64 / 65535.0;
            (rgb, Some(a))
        }
        #[cfg(feature = "f16")]
        PixelFormat::SrgbF16Rgba => {
            use half::f16;
            let off = x * 8;
            let rgb = [
                f16::from_ne_bytes([row_bytes[off], row_bytes[off + 1]]).to_f32() as f64,
                f16::from_ne_bytes([row_bytes[off + 2], row_bytes[off + 3]]).to_f32() as f64,
                f16::from_ne_bytes([row_bytes[off + 4], row_bytes[off + 5]]).to_f32() as f64,
            ];
            let a = f16::from_ne_bytes([row_bytes[off + 6], row_bytes[off + 7]]).to_f32() as f64;
            (rgb, Some(a))
        }
        PixelFormat::LinearF32Rgba => {
            let off = x * 16;
            let rgb = [
                f32::from_ne_bytes(row_bytes[off..off + 4].try_into().unwrap()) as f64,
                f32::from_ne_bytes(row_bytes[off + 4..off + 8].try_into().unwrap()) as f64,
                f32::from_ne_bytes(row_bytes[off + 8..off + 12].try_into().unwrap()) as f64,
            ];
            let a = f32::from_ne_bytes(row_bytes[off + 12..off + 16].try_into().unwrap()) as f64;
            (rgb, Some(a))
        }
        #[allow(unreachable_patterns)]
        _ => panic!("unsupported pixel format for delta stats: {:?}", format),
    }
}

/// Convert accumulated delta stats to the final DeltaStats struct.
fn finalize_delta_stats(acc: DeltaAccum, has_alpha: bool, native_max: f64) -> DeltaStats {
    let n = acc.pixel_count as f64;
    let inv_n = if n > 0.0 { 1.0 / n } else { 0.0 };

    let mut mean_delta = [0.0; 3];
    let mut stddev_delta = [0.0; 3];

    for c in 0..3 {
        mean_delta[c] = acc.sum_delta[c] * inv_n;
        let variance = (acc.sum_delta_sq[c] * inv_n) - (mean_delta[c] * mean_delta[c]);
        stddev_delta[c] = variance.max(0.0).sqrt();
    }

    // Alpha-stratified stats
    let opaque_stats = if has_alpha && acc.opaque_count > 0 {
        let oc = acc.opaque_count as f64;
        Some(AlphaStratifiedStats {
            pixel_count: acc.opaque_count,
            mean_abs_delta: [
                acc.opaque_sum_abs[0] / oc,
                acc.opaque_sum_abs[1] / oc,
                acc.opaque_sum_abs[2] / oc,
            ],
            max_abs_delta: acc.opaque_max_abs,
        })
    } else {
        None
    };

    let semitransparent_stats = if has_alpha && acc.semi_count > 0 {
        let sc = acc.semi_count as f64;
        Some(AlphaStratifiedStats {
            pixel_count: acc.semi_count,
            mean_abs_delta: [
                acc.semi_sum_abs[0] / sc,
                acc.semi_sum_abs[1] / sc,
                acc.semi_sum_abs[2] / sc,
            ],
            max_abs_delta: acc.semi_max_abs,
        })
    } else {
        None
    };

    // Pearson correlation between |delta| and (1 - alpha)
    let alpha_error_correlation = if has_alpha && acc.alpha_pixel_count > 1 {
        let n = acc.alpha_pixel_count as f64;
        let mean_d = acc.sum_delta_mag / n;
        let mean_a = acc.sum_one_minus_alpha / n;
        let cov = acc.sum_delta_alpha / n - mean_d * mean_a;
        let var_d = (acc.sum_delta_mag_sq / n - mean_d * mean_d).max(0.0);
        let var_a = (acc.sum_one_minus_alpha_sq / n - mean_a * mean_a).max(0.0);
        let denom = (var_d * var_a).sqrt();
        if denom > 1e-10 {
            Some((cov / denom).clamp(-1.0, 1.0))
        } else {
            Some(0.0)
        }
    } else {
        None
    };

    DeltaStats {
        mean_delta,
        stddev_delta,
        max_abs_delta: acc.max_abs_delta,
        signed_small_histogram: acc.signed_small,
        native_max,
        pixel_count: acc.pixel_count,
        pixels_differing: acc.pixels_differing,
        pixels_differing_by_more_than_1: acc.pixels_differing_by_more_than_1,
        has_alpha,
        alpha_max_delta: acc.alpha_max_delta,
        alpha_pixels_differing: acc.alpha_pixels_differing,
        src_histogram: *acc.src_histogram,
        dst_histogram: *acc.dst_histogram,
        opaque_stats,
        semitransparent_stats,
        alpha_error_correlation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::compute_zensim_with_config;
    use crate::profile::WEIGHTS_PREVIEW_V0_1;
    use crate::source::RgbSlice;

    /// Verify streaming produces equivalent results to full-image processing.
    ///
    /// The strip-based V-blur running sum starts from strip boundaries (with mirror
    /// padding) while the full-image V-blur starts from image row 0. Additionally,
    /// the full-image path uses fused_blur_h_ssim while strips use separate blur
    /// calls. These produce mathematically identical results but different FP rounding.
    ///
    /// For SSIM features in smooth image regions, catastrophic cancellation in
    /// sigma_sq = blur(src²) - mu² amplifies tiny blur differences by 10-100×.
    /// Features with larger absolute values (edges, hf energy/magnitude loss) match
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
        let src_img = RgbSlice::new(&src, w, h);
        let dst_img = RgbSlice::new(&dst, w, h);
        let streaming_result =
            compute_zensim_streaming(&src_img, &dst_img, &config, &WEIGHTS_PREVIEW_V0_1);

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

    /// Verify that linear f32 input produces equivalent results to sRGB u8 input.
    ///
    /// Given the same image content, the sRGB u8 and linear f32 paths should produce
    /// the same XYB values (within floating-point tolerance due to the LUT quantization
    /// in the sRGB u8 path vs direct float values in the linear path).
    #[test]
    fn linear_f32_matches_srgb_u8() {
        let w = 256;
        let h = 256;
        let n = w * h;

        let mut src_u8 = vec![[128u8, 128, 128]; n];
        let mut dst_u8 = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = ((x + y) * 127 / (w + h)) as u8;
                src_u8[y * w + x] = [r, g, b];
                dst_u8[y * w + x] = [
                    r.saturating_add(3),
                    g.saturating_sub(2),
                    b.saturating_add(1),
                ];
            }
        }

        // Convert u8 to linear f32 using the same LUT the sRGB path uses
        let src_f32: Vec<[f32; 4]> = src_u8
            .iter()
            .map(|&[r, g, b]| {
                [
                    crate::color::srgb_u8_to_linear(r),
                    crate::color::srgb_u8_to_linear(g),
                    crate::color::srgb_u8_to_linear(b),
                    1.0,
                ]
            })
            .collect();
        let dst_f32: Vec<[f32; 4]> = dst_u8
            .iter()
            .map(|&[r, g, b]| {
                [
                    crate::color::srgb_u8_to_linear(r),
                    crate::color::srgb_u8_to_linear(g),
                    crate::color::srgb_u8_to_linear(b),
                    1.0,
                ]
            })
            .collect();

        let config = ZensimConfig {
            compute_all_features: true,
            ..Default::default()
        };

        // sRGB u8 path
        let src_u8_img = RgbSlice::new(&src_u8, w, h);
        let dst_u8_img = RgbSlice::new(&dst_u8, w, h);
        let u8_result =
            compute_zensim_streaming(&src_u8_img, &dst_u8_img, &config, &WEIGHTS_PREVIEW_V0_1);

        // Linear f32 RGBA path via StridedBytes (opaque: alpha=1.0 ignored)
        let src_f32_bytes: &[u8] = bytemuck::cast_slice(&src_f32);
        let dst_f32_bytes: &[u8] = bytemuck::cast_slice(&dst_f32);
        let src_f32_img = crate::source::StridedBytes::with_alpha_mode(
            src_f32_bytes,
            w,
            h,
            w * 16,
            crate::source::PixelFormat::LinearF32Rgba,
            crate::source::AlphaMode::Opaque,
        );
        let dst_f32_img = crate::source::StridedBytes::with_alpha_mode(
            dst_f32_bytes,
            w,
            h,
            w * 16,
            crate::source::PixelFormat::LinearF32Rgba,
            crate::source::AlphaMode::Opaque,
        );
        let f32_result =
            compute_zensim_streaming(&src_f32_img, &dst_f32_img, &config, &WEIGHTS_PREVIEW_V0_1);

        // Score should match very closely (identical linear values → identical XYB → identical features)
        let score_rel =
            (u8_result.score - f32_result.score).abs() / u8_result.score.abs().max(1e-12);
        let dist_rel = (u8_result.raw_distance - f32_result.raw_distance).abs()
            / u8_result.raw_distance.abs().max(1e-12);

        eprintln!(
            "sRGB u8 score={:.10}  linear f32 score={:.10}  rel={:.2e}",
            u8_result.score, f32_result.score, score_rel,
        );
        eprintln!(
            "sRGB u8 dist={:.10}  linear f32 dist={:.10}  rel={:.2e}",
            u8_result.raw_distance, f32_result.raw_distance, dist_rel,
        );

        // When linear f32 values come from the same LUT, the results should be
        // very close (within FP rounding from different code paths).
        assert!(
            score_rel < 1e-6,
            "score relative diff {:.2e} exceeds 1e-6 (sRGB={:.10} vs linear={:.10})",
            score_rel,
            u8_result.score,
            f32_result.score,
        );
        assert!(
            dist_rel < 1e-5,
            "raw_distance relative diff {:.2e} exceeds 1e-5",
            dist_rel,
        );
    }

    /// Verify that BGRA u8 input produces equivalent results to RGB u8 (opaque).
    #[test]
    fn bgra_u8_matches_rgb_u8_opaque() {
        let w = 128;
        let h = 128;
        let n = w * h;

        let mut src_rgb = vec![[128u8, 128, 128]; n];
        let mut dst_rgb = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = ((x + y) * 127 / (w + h)) as u8;
                src_rgb[y * w + x] = [r, g, b];
                dst_rgb[y * w + x] = [
                    r.saturating_add(5),
                    g.saturating_sub(3),
                    b.saturating_add(2),
                ];
            }
        }

        // Convert RGB to BGRA (opaque, alpha=255)
        let src_bgra: Vec<[u8; 4]> = src_rgb.iter().map(|&[r, g, b]| [b, g, r, 255]).collect();
        let dst_bgra: Vec<[u8; 4]> = dst_rgb.iter().map(|&[r, g, b]| [b, g, r, 255]).collect();

        let config = ZensimConfig::default();

        // RGB u8 path
        let src_rgb_img = RgbSlice::new(&src_rgb, w, h);
        let dst_rgb_img = RgbSlice::new(&dst_rgb, w, h);
        let rgb_result =
            compute_zensim_streaming(&src_rgb_img, &dst_rgb_img, &config, &WEIGHTS_PREVIEW_V0_1);

        // BGRA u8 path via StridedBytes
        let src_bgra_bytes: &[u8] = bytemuck::cast_slice(&src_bgra);
        let dst_bgra_bytes: &[u8] = bytemuck::cast_slice(&dst_bgra);
        let src_bgra_img = crate::source::StridedBytes::new(
            src_bgra_bytes,
            w,
            h,
            w * 4,
            crate::source::PixelFormat::Srgb8Bgra,
        );
        let dst_bgra_img = crate::source::StridedBytes::new(
            dst_bgra_bytes,
            w,
            h,
            w * 4,
            crate::source::PixelFormat::Srgb8Bgra,
        );
        let bgra_result =
            compute_zensim_streaming(&src_bgra_img, &dst_bgra_img, &config, &WEIGHTS_PREVIEW_V0_1);

        // Opaque BGRA compositing in linear space should match sRGB u8 RGB
        // within a small tolerance (compositing detour adds FP rounding).
        let score_rel =
            (rgb_result.score - bgra_result.score).abs() / rgb_result.score.abs().max(1e-12);
        eprintln!(
            "RGB u8 score={:.10}  BGRA u8 score={:.10}  rel={:.2e}",
            rgb_result.score, bgra_result.score, score_rel,
        );

        // Note: BGRA path composites over checkerboard in linear space even for
        // opaque pixels (alpha=255 fast path skips blending but linearizes).
        // sRGB u8 RGB path uses the fused sRGB→XYB SIMD. The difference comes
        // from different code paths to the same opsin matrix. Should be very close.
        assert!(
            score_rel < 1e-4,
            "score relative diff {:.2e} exceeds 1e-4",
            score_rel,
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

        let src_img = RgbSlice::new(&src, w, h);
        let dst_img = RgbSlice::new(&dst, w, h);
        let streaming_result =
            compute_zensim_streaming(&src_img, &dst_img, &config, &WEIGHTS_PREVIEW_V0_1);
        let precomputed = PrecomputedReference::new(&src_img, config.num_scales);
        let precomp_result = compute_zensim_streaming_with_ref(
            &precomputed,
            &dst_img,
            &config,
            &WEIGHTS_PREVIEW_V0_1,
        );

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

    /// Identical P3 images through the streaming path should score very close to 100.
    ///
    /// Note: `compute_zensim_streaming` does not shortcircuit on byte-identical
    /// inputs (the public `Zensim::compute` API does). At small sizes like 64x64,
    /// the SSIM blur kernel covers most of the 8x8 scale-3 image, causing tiny
    /// numerical noise. We verify the score is ≥ 99.5.
    #[test]
    fn identical_p3_images_high_score() {
        let w = 64;
        let h = 64;
        let n = w * h;

        let mut pixels = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 200) / w + 30) as u8;
                let g = ((y * 200) / h + 30) as u8;
                let b = 128u8;
                pixels[y * w + x] = [r, g, b];
            }
        }

        let rgb_bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let src = crate::source::StridedBytes::new(
            rgb_bytes,
            w,
            h,
            w * 3,
            crate::source::PixelFormat::Srgb8Rgb,
        )
        .with_color_primaries(crate::source::ColorPrimaries::DisplayP3);

        let dst = crate::source::StridedBytes::new(
            rgb_bytes,
            w,
            h,
            w * 3,
            crate::source::PixelFormat::Srgb8Rgb,
        )
        .with_color_primaries(crate::source::ColorPrimaries::DisplayP3);

        let config = ZensimConfig::default();
        let p3_result = compute_zensim_streaming(&src, &dst, &config, &WEIGHTS_PREVIEW_V0_1);

        // Also run sRGB-identical to verify the gamut path gives similar behavior
        let src_srgb = RgbSlice::new(&pixels, w, h);
        let dst_srgb = RgbSlice::new(&pixels, w, h);
        let srgb_result =
            compute_zensim_streaming(&src_srgb, &dst_srgb, &config, &WEIGHTS_PREVIEW_V0_1);

        eprintln!(
            "P3 identical score: {:.6}, sRGB identical score: {:.6}",
            p3_result.score, srgb_result.score,
        );

        // Both should be very close to 100 (numerical noise at small sizes)
        assert!(
            p3_result.score >= 99.5,
            "P3 identical score should be >= 99.5, got {}",
            p3_result.score,
        );
        assert!(
            srgb_result.score >= 99.5,
            "sRGB identical score should be >= 99.5, got {}",
            srgb_result.score,
        );
    }

    /// P3 vs sRGB with same pixel values should produce different scores
    /// because gamut conversion changes the XYB values.
    #[test]
    fn p3_vs_srgb_same_pixels_differ() {
        let w = 64;
        let h = 64;
        let n = w * h;

        // Create a colorful gradient
        let mut pixels = vec![0u8; n * 16];
        for y in 0..h {
            for x in 0..w {
                let off = (y * w + x) * 16;
                let r = x as f32 / w as f32;
                let g = y as f32 / h as f32;
                let b = 0.5f32;
                let a = 1.0f32;
                pixels[off..off + 4].copy_from_slice(&r.to_ne_bytes());
                pixels[off + 4..off + 8].copy_from_slice(&g.to_ne_bytes());
                pixels[off + 8..off + 12].copy_from_slice(&b.to_ne_bytes());
                pixels[off + 12..off + 16].copy_from_slice(&a.to_ne_bytes());
            }
        }

        let src_p3 = crate::source::StridedBytes::with_alpha_mode(
            &pixels,
            w,
            h,
            w * 16,
            crate::source::PixelFormat::LinearF32Rgba,
            crate::source::AlphaMode::Opaque,
        )
        .with_color_primaries(crate::source::ColorPrimaries::DisplayP3);

        let dst_srgb = crate::source::StridedBytes::with_alpha_mode(
            &pixels,
            w,
            h,
            w * 16,
            crate::source::PixelFormat::LinearF32Rgba,
            crate::source::AlphaMode::Opaque,
        );
        // dst_srgb uses default Srgb primaries

        let config = ZensimConfig::default();
        let result = compute_zensim_streaming(&src_p3, &dst_srgb, &config, &WEIGHTS_PREVIEW_V0_1);

        assert!(
            result.score < 100.0,
            "P3 vs sRGB with same pixel values should score < 100, got {}",
            result.score,
        );
        eprintln!(
            "P3 vs sRGB same-pixels score: {:.4} (expected < 100)",
            result.score,
        );
    }

    /// u16 delta stats: 1-step difference should be detected with native precision.
    #[test]
    fn u16_delta_stats_native_precision() {
        let w = 16;
        let h = 16;
        let n = w * h;

        // Create two u16 RGBA images differing by 1 in the R channel
        let mut src_bytes = vec![0u8; n * 8];
        let mut dst_bytes = vec![0u8; n * 8];

        for i in 0..n {
            let off = i * 8;
            let r: u16 = 32768;
            let g: u16 = 32768;
            let b: u16 = 32768;
            let a: u16 = 65535;

            src_bytes[off..off + 2].copy_from_slice(&r.to_ne_bytes());
            src_bytes[off + 2..off + 4].copy_from_slice(&g.to_ne_bytes());
            src_bytes[off + 4..off + 6].copy_from_slice(&b.to_ne_bytes());
            src_bytes[off + 6..off + 8].copy_from_slice(&a.to_ne_bytes());

            let r_dst: u16 = 32769; // 1 step higher
            dst_bytes[off..off + 2].copy_from_slice(&r_dst.to_ne_bytes());
            dst_bytes[off + 2..off + 4].copy_from_slice(&g.to_ne_bytes());
            dst_bytes[off + 4..off + 6].copy_from_slice(&b.to_ne_bytes());
            dst_bytes[off + 6..off + 8].copy_from_slice(&a.to_ne_bytes());
        }

        let src = crate::source::StridedBytes::with_alpha_mode(
            &src_bytes,
            w,
            h,
            w * 8,
            crate::source::PixelFormat::Srgb16Rgba,
            crate::source::AlphaMode::Opaque,
        );
        let dst = crate::source::StridedBytes::with_alpha_mode(
            &dst_bytes,
            w,
            h,
            w * 8,
            crate::source::PixelFormat::Srgb16Rgba,
            crate::source::AlphaMode::Opaque,
        );

        let ds = compute_delta_stats(&src, &dst);

        assert_eq!(
            ds.native_max, 65535.0,
            "native_max should be 65535.0 for u16"
        );
        assert_eq!(
            ds.pixels_differing, n as u64,
            "all {n} pixels should differ by 1 step at native precision"
        );
        assert_eq!(
            ds.pixels_differing_by_more_than_1, 0,
            "no pixels should differ by more than 1 step"
        );
        // signed_small_histogram: R channel delta = src - dst = -1/65535, so signed_delta = -1
        // Index mapping: -3→0, -2→1, -1→2, 0→3, +1→4, +2→5, +3→6
        assert_eq!(
            ds.signed_small_histogram[0][2], n as u64,
            "R channel should have all pixels at signed delta -1 (index 2)"
        );
        eprintln!(
            "u16 1-step delta: max_abs_delta={:?}, native_max={}",
            ds.max_abs_delta, ds.native_max,
        );
    }

    /// f32 delta stats should have native_max == 1.0.
    #[test]
    fn f32_delta_stats_native_max() {
        let w = 16;
        let h = 16;
        let n = w * h;

        let mut src_bytes = vec![0u8; n * 16];
        let mut dst_bytes = vec![0u8; n * 16];

        for i in 0..n {
            let off = i * 16;
            let r = 0.5f32;
            let g = 0.5f32;
            let b = 0.5f32;
            let a = 1.0f32;

            src_bytes[off..off + 4].copy_from_slice(&r.to_ne_bytes());
            src_bytes[off + 4..off + 8].copy_from_slice(&g.to_ne_bytes());
            src_bytes[off + 8..off + 12].copy_from_slice(&b.to_ne_bytes());
            src_bytes[off + 12..off + 16].copy_from_slice(&a.to_ne_bytes());

            let r_dst = 0.501f32; // small difference
            dst_bytes[off..off + 4].copy_from_slice(&r_dst.to_ne_bytes());
            dst_bytes[off + 4..off + 8].copy_from_slice(&g.to_ne_bytes());
            dst_bytes[off + 8..off + 12].copy_from_slice(&b.to_ne_bytes());
            dst_bytes[off + 12..off + 16].copy_from_slice(&a.to_ne_bytes());
        }

        let src = crate::source::StridedBytes::with_alpha_mode(
            &src_bytes,
            w,
            h,
            w * 16,
            crate::source::PixelFormat::LinearF32Rgba,
            crate::source::AlphaMode::Opaque,
        );
        let dst = crate::source::StridedBytes::with_alpha_mode(
            &dst_bytes,
            w,
            h,
            w * 16,
            crate::source::PixelFormat::LinearF32Rgba,
            crate::source::AlphaMode::Opaque,
        );

        let ds = compute_delta_stats(&src, &dst);

        assert_eq!(ds.native_max, 1.0, "native_max should be 1.0 for f32");
        // With native_max=1.0, a delta of 0.001 is 0.001/1.0 = 0.001
        // The threshold is 0.5/1.0 = 0.5, so 0.001 < 0.5 means no pixels differ
        assert_eq!(
            ds.pixels_differing, 0,
            "delta of 0.001 should not exceed 0.5/1.0 threshold"
        );
        eprintln!(
            "f32 delta stats: max_abs_delta={:?}, native_max={}, pixels_differing={}",
            ds.max_abs_delta, ds.native_max, ds.pixels_differing,
        );
    }
}
