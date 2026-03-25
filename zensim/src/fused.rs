//! Fused V-blur + feature extraction for streaming strips.
//!
//! Instead of 4 separate V-blur passes (writing to memory) followed by 7 reduction
//! passes (reading from memory), this module fuses everything into a single column-wise
//! pass. V-blurred values stay in registers and all features are computed inline.
//!
//! Memory pass reduction: ~40 passes → ~12 passes per channel (with fused H-blur).
#![allow(
    clippy::assign_op_pattern,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::f32x16;

const C2: f32 = 0.0009;

/// Accumulated feature sums from a fused V-blur + feature extraction pass.
/// All values are raw sums (not yet divided by pixel count).
pub(crate) struct StripChannelAccum {
    pub ssim_d: f64,
    pub ssim_d4: f64,
    pub ssim_d2: f64,
    pub edge_art: f64,
    pub edge_art4: f64,
    pub edge_art2: f64,
    pub edge_det: f64,
    pub edge_det4: f64,
    pub edge_det2: f64,
    pub mse: f64,
    pub hf_sq_src: f64,
    pub hf_sq_dst: f64,
    pub hf_abs_src: f64,
    pub hf_abs_dst: f64,
    // Extended: L8 power pool and max
    pub ssim_d8: f64,
    pub edge_art8: f64,
    pub edge_det8: f64,
    pub ssim_max: f32,
    pub edge_art_max: f32,
    pub edge_det_max: f32,
}

impl StripChannelAccum {
    pub fn zero() -> Self {
        Self {
            ssim_d: 0.0,
            ssim_d4: 0.0,
            ssim_d2: 0.0,
            edge_art: 0.0,
            edge_art4: 0.0,
            edge_art2: 0.0,
            edge_det: 0.0,
            edge_det4: 0.0,
            edge_det2: 0.0,
            mse: 0.0,
            hf_sq_src: 0.0,
            hf_sq_dst: 0.0,
            hf_abs_src: 0.0,
            hf_abs_dst: 0.0,
            ssim_d8: 0.0,
            edge_art8: 0.0,
            edge_det8: 0.0,
            ssim_max: 0.0,
            edge_art_max: 0.0,
            edge_det_max: 0.0,
        }
    }
}

/// Fused V-blur + ALL feature extraction for SSIM channels.
///
/// Reads 6 inputs: 4 H-blurred planes (h_mu1, h_mu2, h_sigma_sq, h_sigma12) + raw src + dst.
/// Maintains 4 V-blur running sums per column group.
/// At each inner row, computes SSIM, edge, variance, texture, and MSE features
/// directly from register values — V-blur outputs never touch memory.
///
/// Returns accumulated feature sums for the inner rows of this strip.
pub(crate) fn fused_vblur_features_ssim(
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
) -> StripChannelAccum {
    incant!(
        fused_vblur_ssim_inner(
            h_mu1,
            h_mu2,
            h_sigma_sq,
            h_sigma12,
            src,
            dst,
            width,
            height,
            inner_start,
            inner_h,
            radius,
            mu1_out,
            mu2_out,
            store_mu,
            sd_out,
            store_sd
        ),
        [v4, v3, scalar]
    )
}

/// Fused V-blur + feature extraction for edge-only channels (no SSIM).
///
/// Reads 4 inputs: 2 H-blurred planes (h_mu1, h_mu2) + raw src + dst.
/// Maintains 2 V-blur running sums per column group.
/// Computes edge, variance, texture, and MSE features inline.
pub(crate) fn fused_vblur_features_edge(
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    incant!(
        fused_vblur_edge_inner(
            h_mu1,
            h_mu2,
            src,
            dst,
            width,
            height,
            inner_start,
            inner_h,
            radius,
            mu1_out,
            mu2_out,
            store_mu
        ),
        [v4, v3, scalar]
    )
}

// ============================================================
// Helper: mirror-reflect row index for V-blur boundary handling
// ============================================================

#[inline(always)]
fn mirror_idx(i: usize, r: usize, height: usize) -> usize {
    if i <= r {
        (r - i).min(height - 1)
    } else {
        (i - r).min(height - 1)
    }
}

#[inline(always)]
fn vblur_add_idx(y: usize, r: usize, height: usize) -> usize {
    let add_raw = y + r + 1;
    if add_raw < height {
        add_raw
    } else {
        // Mirror-reflect: fold back from the boundary.
        // Use signed math to avoid underflow when add_raw >> height.
        let reflected = 2 * (height as isize - 1) - add_raw as isize;
        reflected.unsigned_abs().min(height - 1)
    }
}

#[inline(always)]
fn vblur_rem_idx(y: usize, r: usize, height: usize) -> usize {
    let rem_i = y as isize - r as isize;
    let idx = if rem_i < 0 {
        rem_i.unsigned_abs()
    } else {
        rem_i as usize
    };
    idx.min(height - 1)
}

// ============================================================
// AVX-512 implementations
// ============================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_ssim_inner_v4(
    token: archmage::X64V4Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    // SSIM constants
    let c2v = f32x16::splat(token, C2);
    let one = f32x16::splat(token, 1.0);
    let two = f32x16::splat(token, 2.0);
    let zero = f32x16::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 16;

        // Initialize 4 running sums for this column group
        let mut sum_m1 = f32x16::zero(token);
        let mut sum_m2 = f32x16::zero(token);
        let mut sum_sq = f32x16::zero(token);
        let mut sum_s12 = f32x16::zero(token);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x16::from_array(token, h_mu1[base..][..16].try_into().unwrap());
            sum_m2 = sum_m2 + f32x16::from_array(token, h_mu2[base..][..16].try_into().unwrap());
            sum_sq =
                sum_sq + f32x16::from_array(token, h_sigma_sq[base..][..16].try_into().unwrap());
            sum_s12 =
                sum_s12 + f32x16::from_array(token, h_sigma12[base..][..16].try_into().unwrap());
        }

        for y in 0..height {
            // Only accumulate features for inner rows
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;

                // V-blurred values (still in registers)
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let ssq = sum_sq * inv_v;
                let s12 = sum_s12 * inv_v;

                // Load raw pixel values
                let s = f32x16::from_array(token, src[base..][..16].try_into().unwrap());
                let d = f32x16::from_array(token, dst[base..][..16].try_into().unwrap());

                // === SSIM ===
                let mu_diff = mu1 - mu2;
                let num_m = mu_diff.mul_add(-mu_diff, one);
                let num_s = two.mul_add((-mu1).mul_add(mu2, s12), c2v);
                let denom_s = (-mu2).mul_add(mu2, (-mu1).mul_add(mu1, ssq)) + c2v;
                let sd = (one - (num_m * num_s) / denom_s).max(zero);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                if store_sd {
                    sd_out[base..base + 16].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 16].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 16].copy_from_slice(&mu2.to_array());
                }

                // === Edge ===
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                // === HF energy (L2): (pixel - mu)² ===
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // === HF magnitude (L1): |pixel - mu| ===
                // diff1/diff2 already computed above
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // === MSE: (src - dst)² ===
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            // Slide V-blur window
            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;

            sum_m1 = sum_m1
                + f32x16::from_array(token, h_mu1[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu1[rem_base..][..16].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x16::from_array(token, h_mu2[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu2[rem_base..][..16].try_into().unwrap());
            sum_sq = sum_sq
                + f32x16::from_array(token, h_sigma_sq[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_sigma_sq[rem_base..][..16].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x16::from_array(token, h_sigma12[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_sigma12[rem_base..][..16].try_into().unwrap());
        }
    }

    // Remainder columns with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    let c2v8 = f32x8::splat(v3, C2);
    let one8 = f32x8::splat(v3, 1.0);
    let two8 = f32x8::splat(v3, 2.0);
    let zero8 = f32x8::zero(v3);

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum_m1 = f32x8::zero(v3);
        let mut sum_m2 = f32x8::zero(v3);
        let mut sum_sq = f32x8::zero(v3);
        let mut sum_s12 = f32x8::zero(v3);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[base..][..8].try_into().unwrap());
            sum_sq = sum_sq + f32x8::from_array(v3, h_sigma_sq[base..][..8].try_into().unwrap());
            sum_s12 = sum_s12 + f32x8::from_array(v3, h_sigma12[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v8;
                let mu2 = sum_m2 * inv_v8;
                let ssq = sum_sq * inv_v8;
                let s12 = sum_s12 * inv_v8;
                let s = f32x8::from_array(v3, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(v3, dst[base..][..8].try_into().unwrap());

                // SSIM
                let mu_diff = mu1 - mu2;
                let num_m = mu_diff.mul_add(-mu_diff, one8);
                let num_s = two8.mul_add((-mu1).mul_add(mu2, s12), c2v8);
                let denom_s = (-mu2).mul_add(mu2, (-mu1).mul_add(mu1, ssq)) + c2v8;
                let sd = (one8 - (num_m * num_s) / denom_s).max(zero8);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                if store_sd {
                    sd_out[base..base + 8].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }

                // Edge
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one8 + diff2) / (one8 + diff1) - one8;
                let artifact = ed.max(zero8);
                let detail_lost = (-ed).max(zero8);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // Texture
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu2[rem_base..][..8].try_into().unwrap());
            sum_sq = sum_sq
                + f32x8::from_array(v3, h_sigma_sq[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_sigma_sq[rem_base..][..8].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x8::from_array(v3, h_sigma12[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_sigma12[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_s12 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
            sum_sq += h_sigma_sq[idx * width + x];
            sum_s12 += h_sigma12[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let ssq = sum_sq * inv;
                let s12 = sum_s12 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                // SSIM (f32 to match SIMD paths)
                let mu_diff = mu1 - mu2;
                let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
                let num_s = 2.0f32.mul_add((-mu1).mul_add(mu2, s12), C2);
                let denom_s = (-mu2).mul_add(mu2, (-mu1).mul_add(mu1, ssq)) + C2;
                let sd = (1.0f32 - (num_m * num_s) / denom_s).max(0.0f32);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd as f64;
                acc.ssim_d4 += sd4 as f64;
                acc.ssim_d2 += sd2 as f64;
                acc.ssim_d8 += (sd4 * sd4) as f64;
                acc.ssim_max = acc.ssim_max.max(sd);
                if store_sd {
                    sd_out[y * width + x] = sd;
                }
                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge (f32 to match SIMD paths)
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact as f64;
                acc.edge_art4 += a4 as f64;
                acc.edge_art2 += a2 as f64;
                acc.edge_det += detail_lost as f64;
                acc.edge_det4 += dl4 as f64;
                acc.edge_det2 += dl2 as f64;
                acc.edge_art8 += (a4 * a4) as f64;
                acc.edge_det8 += (dl4 * dl4) as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact);
                acc.edge_det_max = acc.edge_det_max.max(detail_lost);

                // Variance
                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;

                // Texture
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                // MSE
                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
            sum_sq = sum_sq + h_sigma_sq[add_idx * width + x] - h_sigma_sq[rem_idx * width + x];
            sum_s12 = sum_s12 + h_sigma12[add_idx * width + x] - h_sigma12[rem_idx * width + x];
        }
    }

    acc
}

// ============================================================
// AVX2 implementations
// ============================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_ssim_inner_v3(
    token: archmage::X64V3Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    let c2v = f32x8::splat(token, C2);
    let one = f32x8::splat(token, 1.0);
    let two = f32x8::splat(token, 2.0);
    let zero = f32x8::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 8;
        let mut sum_m1 = f32x8::zero(token);
        let mut sum_m2 = f32x8::zero(token);
        let mut sum_sq = f32x8::zero(token);
        let mut sum_s12 = f32x8::zero(token);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(token, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(token, h_mu2[base..][..8].try_into().unwrap());
            sum_sq = sum_sq + f32x8::from_array(token, h_sigma_sq[base..][..8].try_into().unwrap());
            sum_s12 =
                sum_s12 + f32x8::from_array(token, h_sigma12[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let ssq = sum_sq * inv_v;
                let s12 = sum_s12 * inv_v;
                let s = f32x8::from_array(token, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(token, dst[base..][..8].try_into().unwrap());

                // SSIM
                let mu_diff = mu1 - mu2;
                let num_m = mu_diff.mul_add(-mu_diff, one);
                let num_s = two.mul_add((-mu1).mul_add(mu2, s12), c2v);
                let denom_s = (-mu2).mul_add(mu2, (-mu1).mul_add(mu1, ssq)) + c2v;
                let sd = (one - (num_m * num_s) / denom_s).max(zero);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                if store_sd {
                    sd_out[base..base + 8].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }

                // Edge
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // Texture
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(token, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(token, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu2[rem_base..][..8].try_into().unwrap());
            sum_sq = sum_sq
                + f32x8::from_array(token, h_sigma_sq[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_sigma_sq[rem_base..][..8].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x8::from_array(token, h_sigma12[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_sigma12[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_s12 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
            sum_sq += h_sigma_sq[idx * width + x];
            sum_s12 += h_sigma12[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let ssq = sum_sq * inv;
                let s12 = sum_s12 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                // SSIM
                let mu_diff = mu1 - mu2;
                let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
                let num_s = 2.0f32.mul_add((-mu1).mul_add(mu2, s12), C2);
                let denom_s = (-mu2).mul_add(mu2, (-mu1).mul_add(mu1, ssq)) + C2;
                let sd = (1.0f32 - (num_m * num_s) / denom_s).max(0.0f32);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd as f64;
                acc.ssim_d4 += sd4 as f64;
                acc.ssim_d2 += sd2 as f64;
                acc.ssim_d8 += (sd4 * sd4) as f64;
                acc.ssim_max = acc.ssim_max.max(sd);
                if store_sd {
                    sd_out[y * width + x] = sd;
                }
                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact as f64;
                acc.edge_art4 += a4 as f64;
                acc.edge_art2 += a2 as f64;
                acc.edge_det += detail_lost as f64;
                acc.edge_det4 += dl4 as f64;
                acc.edge_det2 += dl2 as f64;
                acc.edge_art8 += (a4 * a4) as f64;
                acc.edge_det8 += (dl4 * dl4) as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact);
                acc.edge_det_max = acc.edge_det_max.max(detail_lost);

                // Variance
                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;

                // Texture
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                // MSE
                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
            sum_sq = sum_sq + h_sigma_sq[add_idx * width + x] - h_sigma_sq[rem_idx * width + x];
            sum_s12 = sum_s12 + h_sigma12[add_idx * width + x] - h_sigma12[rem_idx * width + x];
        }
    }

    acc
}

// ============================================================
// Scalar implementations
// ============================================================

fn fused_vblur_ssim_inner_scalar(
    _token: archmage::ScalarToken,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv = 1.0 / diam as f32;
    let r = radius;

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for x in 0..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_s12 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
            sum_sq += h_sigma_sq[idx * width + x];
            sum_s12 += h_sigma12[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let ssq = sum_sq * inv;
                let s12 = sum_s12 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                // SSIM (f32 to match SIMD paths)
                let mu_diff = mu1 - mu2;
                let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
                let num_s = 2.0f32.mul_add((-mu1).mul_add(mu2, s12), C2);
                let denom_s = (-mu2).mul_add(mu2, (-mu1).mul_add(mu1, ssq)) + C2;
                let sd = (1.0f32 - (num_m * num_s) / denom_s).max(0.0f32);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd as f64;
                acc.ssim_d4 += sd4 as f64;
                acc.ssim_d2 += sd2 as f64;
                acc.ssim_d8 += (sd4 * sd4) as f64;
                acc.ssim_max = acc.ssim_max.max(sd);
                if store_sd {
                    sd_out[y * width + x] = sd;
                }
                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact as f64;
                acc.edge_art4 += a4 as f64;
                acc.edge_art2 += a2 as f64;
                acc.edge_det += detail_lost as f64;
                acc.edge_det4 += dl4 as f64;
                acc.edge_det2 += dl2 as f64;
                acc.edge_art8 += (a4 * a4) as f64;
                acc.edge_det8 += (dl4 * dl4) as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact);
                acc.edge_det_max = acc.edge_det_max.max(detail_lost);

                // Variance
                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;

                // Texture
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                // MSE
                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
            sum_sq = sum_sq + h_sigma_sq[add_idx * width + x] - h_sigma_sq[rem_idx * width + x];
            sum_s12 = sum_s12 + h_sigma12[add_idx * width + x] - h_sigma12[rem_idx * width + x];
        }
    }

    acc
}

// ============================================================
// Edge-only fused V-blur (no SSIM, only 2 running sums)
// ============================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_edge_inner_v4(
    token: archmage::X64V4Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    let one = f32x16::splat(token, 1.0);
    let zero = f32x16::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 16;
        let mut sum_m1 = f32x16::zero(token);
        let mut sum_m2 = f32x16::zero(token);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x16::from_array(token, h_mu1[base..][..16].try_into().unwrap());
            sum_m2 = sum_m2 + f32x16::from_array(token, h_mu2[base..][..16].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let s = f32x16::from_array(token, src[base..][..16].try_into().unwrap());
                let d = f32x16::from_array(token, dst[base..][..16].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 16].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 16].copy_from_slice(&mu2.to_array());
                }

                // Edge
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // Texture
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1
                + f32x16::from_array(token, h_mu1[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu1[rem_base..][..16].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x16::from_array(token, h_mu2[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu2[rem_base..][..16].try_into().unwrap());
        }
    }

    // Remainder with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    let one8 = f32x8::splat(v3, 1.0);
    let zero8 = f32x8::zero(v3);

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum_m1 = f32x8::zero(v3);
        let mut sum_m2 = f32x8::zero(v3);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v8;
                let mu2 = sum_m2 * inv_v8;
                let s = f32x8::from_array(v3, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(v3, dst[base..][..8].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }

                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one8 + diff2) / (one8 + diff1) - one8;
                let artifact = ed.max(zero8);
                let detail_lost = (-ed).max(zero8);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu2[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact as f64;
                acc.edge_art4 += a4 as f64;
                acc.edge_art2 += a2 as f64;
                acc.edge_det += detail_lost as f64;
                acc.edge_det4 += dl4 as f64;
                acc.edge_det2 += dl2 as f64;
                acc.edge_art8 += (a4 * a4) as f64;
                acc.edge_det8 += (dl4 * dl4) as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact);
                acc.edge_det_max = acc.edge_det_max.max(detail_lost);

                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
        }
    }

    acc
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_edge_inner_v3(
    token: archmage::X64V3Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 8;
        let mut sum_m1 = f32x8::zero(token);
        let mut sum_m2 = f32x8::zero(token);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(token, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(token, h_mu2[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let s = f32x8::from_array(token, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(token, dst[base..][..8].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }

                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(token, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(token, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu2[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact as f64;
                acc.edge_art4 += a4 as f64;
                acc.edge_art2 += a2 as f64;
                acc.edge_det += detail_lost as f64;
                acc.edge_det4 += dl4 as f64;
                acc.edge_det2 += dl2 as f64;
                acc.edge_art8 += (a4 * a4) as f64;
                acc.edge_det8 += (dl4 * dl4) as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact);
                acc.edge_det_max = acc.edge_det_max.max(detail_lost);

                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
        }
    }

    acc
}

fn fused_vblur_edge_inner_scalar(
    _token: archmage::ScalarToken,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv = 1.0 / diam as f32;
    let r = radius;

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for x in 0..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact as f64;
                acc.edge_art4 += a4 as f64;
                acc.edge_art2 += a2 as f64;
                acc.edge_det += detail_lost as f64;
                acc.edge_det4 += dl4 as f64;
                acc.edge_det2 += dl2 as f64;
                acc.edge_art8 += (a4 * a4) as f64;
                acc.edge_det8 += (dl4 * dl4) as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact);
                acc.edge_det_max = acc.edge_det_max.max(detail_lost);

                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
        }
    }

    acc
}
