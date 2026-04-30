//! Streaming/row-major SSIM kernel that eliminates full-strip σ plane storage.
//!
//! ## Background
//!
//! The default fused SSIM path (`fused_blur_h_ssim` → `fused_vblur_features_ssim`)
//! writes four full-strip H-blurred planes — `mu1`, `mu2`, `sigma_sq`,
//! `sigma_12` — and then reads them back in the V-blur. At 1080p with the
//! default `STRIP_INNER=32` the σ planes alone are ~640 KB; across all
//! scales × channels it's ~30–40 MB of σ store/load traffic per image.
//! Most of that doesn't fit in per-core L2.
//!
//! ## What this module does
//!
//! Same algorithm, smaller buffers. The σ planes are stored in a small
//! `(2r+1)`-row ring buffer (~84 KB per plane at 1080p, fits in L1) instead
//! of full-strip planes. To make that work the kernel runs row-major
//! instead of col-major: it pre-fills the ring with the first `2r+1` rows
//! of σ, computes features for inner rows in order, and slides the V-window
//! by H-blurring one new row per inner-row step (overwriting the oldest
//! ring slot).
//!
//! V-running sums are kept in per-column `Vec<f32>`s rather than registers
//! (since the row-major loop structure can't hold them in registers across
//! col-group iterations). At 1920 wide that's ~30 KB total for the four
//! V-running-sum arrays — fits in L1.
//!
//! ## What stays the same
//!
//! - μ planes (`mu1`, `mu2`) are still full-strip H-blurred buffers.
//!   They're the cheaper of the four planes (2 vs the 4 in the σ path)
//!   and the V-window slide for μ uses random-access reads from the full
//!   plane, which works fine with the existing `fused_blur_h_mu`.
//! - The actual SSIM math. We just feed it from differently-sourced
//!   running sums.
//!
//! ## Scope of this implementation
//!
//! Pure scalar fallback. Establishes correctness and measures the memory-
//! traffic-only win. SIMD specializations (v4/v3/neon) are followup work
//! once the design is validated.
//!
//! Edge / HF / MSE features still use the existing fused kernels — this
//! module only owns the SSIM-needed code path.

use crate::fused::{StripChannelAccum, mirror_idx};
#[cfg(target_arch = "x86_64")]
use archmage::{arcane, rite};
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::f32x16;

/// SSIM constant (matches `fused.rs`).
const C2: f32 = 0.0009;

/// Pre-allocated scratch for the streaming SSIM kernel.
///
/// Lives alongside `ScaleBuffers` so the allocations are reused across
/// scales. Sized once for the largest scale's width.
#[derive(Default)]
pub(crate) struct StreamingSsimScratch {
    /// Ring buffer of H-blurred (src² + dst²): `(2*radius+1) * width` floats.
    pub sigma_ring_sq: Vec<f32>,
    /// Ring buffer of H-blurred (src · dst): `(2*radius+1) * width` floats.
    pub sigma_ring_12: Vec<f32>,
    /// Per-column V-running sum of H-blurred mu1: `width` floats.
    pub col_v_mu1: Vec<f32>,
    /// Per-column V-running sum of H-blurred mu2: `width` floats.
    pub col_v_mu2: Vec<f32>,
    /// Per-column V-running sum of H-blurred σ²: `width` floats.
    pub col_v_sumsq: Vec<f32>,
    /// Per-column V-running sum of H-blurred σ₁₂: `width` floats.
    pub col_v_sumprod: Vec<f32>,
}

impl StreamingSsimScratch {
    pub fn ensure(&mut self, width: usize, ring_size: usize) {
        let ring_capacity = ring_size * width;
        if self.sigma_ring_sq.len() < ring_capacity {
            self.sigma_ring_sq.resize(ring_capacity, 0.0);
            self.sigma_ring_12.resize(ring_capacity, 0.0);
        }
        if self.col_v_mu1.len() < width {
            self.col_v_mu1.resize(width, 0.0);
            self.col_v_mu2.resize(width, 0.0);
            self.col_v_sumsq.resize(width, 0.0);
            self.col_v_sumprod.resize(width, 0.0);
        }
    }
}

/// H-blur a single row of (src² + dst²) and (src · dst) into the ring slots.
///
/// Mirror-reflects boundary columns. Dispatches to the AVX-512 inner kernel
/// when available; falls back to scalar otherwise.
#[inline]
fn h_blur_sigma_row(
    src_row: &[f32],
    dst_row: &[f32],
    out_sigma_sq: &mut [f32],
    out_sigma_12: &mut [f32],
    width: usize,
    radius: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = <archmage::X64V4Token as archmage::SimdToken>::summon() {
        h_blur_sigma_row_v4(token, src_row, dst_row, out_sigma_sq, out_sigma_12, width, radius);
        return;
    }
    h_blur_sigma_row_scalar(src_row, dst_row, out_sigma_sq, out_sigma_12, width, radius);
}

#[inline]
fn h_blur_sigma_row_scalar(
    src_row: &[f32],
    dst_row: &[f32],
    out_sigma_sq: &mut [f32],
    out_sigma_12: &mut [f32],
    width: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv = 1.0 / diam as f32;
    let r = radius as isize;
    for x in 0..width {
        let mut sumsq = 0.0f32;
        let mut sum12 = 0.0f32;
        for k in -r..=r {
            let mut xi = x as isize + k;
            if xi < 0 {
                xi = -xi;
            }
            if xi >= width as isize {
                xi = 2 * (width as isize - 1) - xi;
            }
            if xi < 0 {
                xi = 0;
            }
            let xi = xi as usize;
            let s = src_row[xi];
            let d = dst_row[xi];
            sumsq += s * s + d * d;
            sum12 += s * d;
        }
        out_sigma_sq[x] = sumsq * inv;
        out_sigma_12[x] = sum12 * inv;
    }
}

/// AVX-512 H-blur of a single row's σ for one 16-col chunk.
///
/// Used by the col-group-outer streaming kernel to stream σ-plane data
/// directly into a per-col-group ring buffer without writing a full
/// strip-wide σ plane to memory. Boundary case (col_base near image
/// edges) falls through to scalar with mirror-reflection.
// `#[rite]` so this inlines into the parent #[arcane] (run_inner_loop_v4)
// without crossing a target_feature boundary. Nested arcane = perf bug.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_blur_sigma_chunk_v4(
    token: archmage::X64V4Token,
    src: &[f32],
    dst: &[f32],
    row: usize,
    col_base: usize,
    width: usize,
    radius: usize,
    out_sq: &mut [f32; 16],
    out_12: &mut [f32; 16],
) {
    let r = radius;
    let diam = 2 * r + 1;
    let inv = f32x16::splat(token, 1.0 / diam as f32);
    let row_off = row * width;
    let body_start = r;
    let body_end = if width > 2 * r { width - r } else { r };

    if col_base >= body_start && col_base + 16 <= body_end {
        // Body case: straight-line SIMD, no boundary mirror.
        let mut sumsq = f32x16::zero(token);
        let mut sumprod = f32x16::zero(token);
        for k in 0..diam {
            let in_off = row_off + col_base + k - r;
            let s = f32x16::from_array(token, src[in_off..in_off + 16].try_into().unwrap());
            let d = f32x16::from_array(token, dst[in_off..in_off + 16].try_into().unwrap());
            sumsq = s.mul_add(s, d.mul_add(d, sumsq));
            sumprod = s.mul_add(d, sumprod);
        }
        let avg_sq = sumsq * inv;
        let avg_12 = sumprod * inv;
        *out_sq = avg_sq.to_array();
        *out_12 = avg_12.to_array();
    } else {
        // Boundary case: per-col scalar with mirror-reflection.
        for c in 0..16 {
            let x = col_base + c;
            let mut sumsq = 0.0f32;
            let mut sum12 = 0.0f32;
            for k in 0..diam {
                let mut xi = x as isize + k as isize - r as isize;
                if xi < 0 {
                    xi = -xi;
                }
                if xi >= width as isize {
                    xi = 2 * (width as isize - 1) - xi;
                }
                if xi < 0 {
                    xi = 0;
                }
                let xi = xi as usize;
                let s = src[row_off + xi];
                let d = dst[row_off + xi];
                sumsq += s * s + d * d;
                sum12 += s * d;
            }
            out_sq[c] = sumsq / diam as f32;
            out_12[c] = sum12 / diam as f32;
        }
    }
}

/// AVX-512 single-row H-blur for σ_sq + σ_12.
///
/// Strategy: maintain a sliding sum across (2r+1) column positions. At each
/// SIMD step, accumulate 16 lanes of (src²+dst²) and (src·dst) at offset
/// `[x..x+16)`. The "lane-aligned" position summed is `x - r .. x + r + 16`.
///
/// For the body (`r..width-r`) we can do this with straight-line SIMD; the
/// boundary edges (left r cols, right r cols) need mirror-reflection and
/// fall back to scalar — they're a tiny fraction of the work.
// `#[arcane]`: this is the per-row H-blur entry point called from the
// public `h_blur_sigma_row` dispatcher. Once-per-row calls amortise the
// target_feature boundary over 1920 cols of work.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn h_blur_sigma_row_v4(
    token: archmage::X64V4Token,
    src_row: &[f32],
    dst_row: &[f32],
    out_sigma_sq: &mut [f32],
    out_sigma_12: &mut [f32],
    width: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let body_end = if width > 2 * r { width - r } else { r };
    let body_start = r;

    // Boundary left cols: scalar (mirror-reflect).
    for x in 0..body_start.min(width) {
        let mut sumsq = 0.0f32;
        let mut sum12 = 0.0f32;
        for k in 0..diam {
            let mut xi = x as isize + k as isize - r as isize;
            if xi < 0 {
                xi = -xi;
            }
            if xi >= width as isize {
                xi = 2 * (width as isize - 1) - xi;
            }
            if xi < 0 {
                xi = 0;
            }
            let xi = xi as usize;
            let s = src_row[xi];
            let d = dst_row[xi];
            sumsq += s * s + d * d;
            sum12 += s * d;
        }
        out_sigma_sq[x] = sumsq / diam as f32;
        out_sigma_12[x] = sum12 / diam as f32;
    }

    // Body: SIMD over 16-col chunks. Output column `out_x` reads input
    // columns `[out_x-r .. out_x+r+1)`. We process 16 output cols at a time;
    // need 16 + 2r consecutive input columns.
    let mut out_x = body_start;
    while out_x + 16 <= body_end {
        let mut sumsq = f32x16::zero(token);
        let mut sumprod = f32x16::zero(token);
        for k in 0..diam {
            let in_off = out_x + k - r;
            let s = f32x16::from_array(
                token,
                src_row[in_off..in_off + 16].try_into().unwrap(),
            );
            let d = f32x16::from_array(
                token,
                dst_row[in_off..in_off + 16].try_into().unwrap(),
            );
            sumsq = s.mul_add(s, d.mul_add(d, sumsq));
            sumprod = s.mul_add(d, sumprod);
        }
        let avg_sq = sumsq * inv;
        let avg_pr = sumprod * inv;
        out_sigma_sq[out_x..out_x + 16].copy_from_slice(&avg_sq.to_array());
        out_sigma_12[out_x..out_x + 16].copy_from_slice(&avg_pr.to_array());
        out_x += 16;
    }

    // Body remainder: scalar.
    for x in out_x..body_end {
        let mut sumsq = 0.0f32;
        let mut sum12 = 0.0f32;
        for k in 0..diam {
            let xi = x + k - r;
            let s = src_row[xi];
            let d = dst_row[xi];
            sumsq += s * s + d * d;
            sum12 += s * d;
        }
        out_sigma_sq[x] = sumsq / diam as f32;
        out_sigma_12[x] = sum12 / diam as f32;
    }

    // Boundary right cols: scalar (mirror-reflect).
    for x in body_end..width {
        let mut sumsq = 0.0f32;
        let mut sum12 = 0.0f32;
        for k in 0..diam {
            let mut xi = x as isize + k as isize - r as isize;
            if xi < 0 {
                xi = -xi;
            }
            if xi >= width as isize {
                xi = 2 * (width as isize - 1) - xi;
            }
            if xi < 0 {
                xi = 0;
            }
            let xi = xi as usize;
            let s = src_row[xi];
            let d = dst_row[xi];
            sumsq += s * s + d * d;
            sum12 += s * d;
        }
        out_sigma_sq[x] = sumsq / diam as f32;
        out_sigma_12[x] = sum12 / diam as f32;
    }
}

/// Streaming SSIM features for an inner-row range of one strip.
///
/// Equivalent to `fused_vblur_features_ssim` but with σ planes stored in a
/// `(2*radius+1)`-row ring buffer instead of full-strip planes.
///
/// Caller-provided `mu1_full` / `mu2_full` are the full-strip H-blurred
/// μ planes from `fused_blur_h_mu`.
///
/// Outputs:
/// - `mu1_out` / `mu2_out`: filled with V-blurred μ values for inner rows
///   when `store_mu`. Same layout as the existing kernel writes.
/// - `sd_out`: filled with per-pixel SSIM diff `sd` for inner rows when
///   `store_sd`. Same layout as the existing kernel writes.
///
/// Returns `StripChannelAccum` with all SSIM/edge/HF/MSE accumulators
/// matching the existing kernel's outputs.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn streaming_features_ssim(
    mu1_full: &[f32],
    mu2_full: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    strip_h: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    scratch: &mut StreamingSsimScratch,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv_n = 1.0 / diam as f32;

    scratch.ensure(width, diam);

    // Resolve the i-th window position (0..diam) into a strip row index,
    // mirror-reflecting at boundaries — same convention as the existing
    // fused kernel (different `i`s can map to the same row near the top
    // edge, e.g. for r=5 both i=4 and i=6 map to row 1 when inner_start=0).
    let window_row = |i: usize| -> usize {
        if inner_start >= radius {
            let raw = inner_start as isize + i as isize - radius as isize;
            if raw < 0 {
                (-raw) as usize
            } else if raw >= strip_h as isize {
                (2 * (strip_h as isize - 1) - raw).max(0) as usize
            } else {
                raw as usize
            }
            .min(strip_h - 1)
        } else {
            mirror_idx(i, radius, strip_h)
        }
    };

    // Pre-fill ring slots keyed by ROW (slot = row % diam). Duplicate rows
    // (multiple i mapping to the same row) just rewrite the same data,
    // which is fine — slot content depends only on the row.
    let mut filled = [false; 32]; // diam fits in 32 for any reasonable radius
    for i in 0..diam {
        let row = window_row(i);
        let slot = row % diam;
        if filled[slot] {
            continue;
        }
        filled[slot] = true;
        let src_row = &src[row * width..row * width + width];
        let dst_row = &dst[row * width..row * width + width];
        let ring_sq = &mut scratch.sigma_ring_sq[slot * width..slot * width + width];
        let ring_12 = &mut scratch.sigma_ring_12[slot * width..slot * width + width];
        h_blur_sigma_row(src_row, dst_row, ring_sq, ring_12, width, radius);
    }

    // Initialize per-col V-running sums by iterating over `i` (so duplicate
    // rows contribute multiple times when the V-window mirror-reflects at
    // a boundary). σ values come from the ring keyed by row.
    for c in 0..width {
        let mut s_mu1 = 0.0f32;
        let mut s_mu2 = 0.0f32;
        let mut s_sq = 0.0f32;
        let mut s_12 = 0.0f32;
        for i in 0..diam {
            let row = window_row(i);
            s_mu1 += mu1_full[row * width + c];
            s_mu2 += mu2_full[row * width + c];
            let slot = row % diam;
            s_sq += scratch.sigma_ring_sq[slot * width + c];
            s_12 += scratch.sigma_ring_12[slot * width + c];
        }
        scratch.col_v_mu1[c] = s_mu1;
        scratch.col_v_mu2[c] = s_mu2;
        scratch.col_v_sumsq[c] = s_sq;
        scratch.col_v_sumprod[c] = s_12;
    }

    let mut acc = StripChannelAccum::zero();

    // SIMD-accelerate the row+slide inner loop on AVX-512 hosts.
    // Pre-fill and V-sum init above are scalar (run once per call —
    // cheap relative to the inner-row × col compute).
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = <archmage::X64V4Token as archmage::SimdToken>::summon() {
        run_inner_loop_v4(
            token, &mut acc, mu1_full, mu2_full, src, dst, width, strip_h, inner_start, inner_h,
            radius, mu1_out, mu2_out, store_mu, sd_out, store_sd, scratch,
        );
        return acc;
    }

    for step in 0..inner_h {
        let inner_y = inner_start + step;

        // Emit features for inner_y.
        for x in 0..width {
            let mu1 = scratch.col_v_mu1[x] * inv_n;
            let mu2 = scratch.col_v_mu2[x] * inv_n;
            let ssq = scratch.col_v_sumsq[x] * inv_n;
            let s12 = scratch.col_v_sumprod[x] * inv_n;

            let mu_diff = mu1 - mu2;
            let num_m = mu_diff.mul_add(-mu_diff, 1.0);
            let num_s = 2.0f32.mul_add((-mu1).mul_add(mu2, s12), C2);
            let denom_s = (-mu2).mul_add(mu2, (-mu1).mul_add(mu1, ssq)) + C2;
            let sd = (1.0f32 - (num_m * num_s) / denom_s).max(0.0);
            let sd2 = sd * sd;
            let sd4 = sd2 * sd2;

            acc.ssim_d += sd as f64;
            acc.ssim_d4 += sd4 as f64;
            acc.ssim_d2 += sd2 as f64;
            acc.ssim_d8 += (sd4 * sd4) as f64;
            acc.ssim_max = acc.ssim_max.max(sd);

            let pixel_off = inner_y * width + x;
            if store_sd {
                sd_out[pixel_off] = sd;
            }
            if store_mu {
                mu1_out[pixel_off] = mu1;
                mu2_out[pixel_off] = mu2;
            }

            // Edge features
            let sv = src[pixel_off];
            let dv = dst[pixel_off];
            let diff1 = (sv - mu1).abs();
            let diff2 = (dv - mu2).abs();
            let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0;
            let artifact = ed.max(0.0);
            let detail_lost = (-ed).max(0.0);
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

            // HF features
            let vs = sv - mu1;
            let vd = dv - mu2;
            acc.hf_sq_src += (vs * vs) as f64;
            acc.hf_sq_dst += (vd * vd) as f64;
            acc.hf_abs_src += diff1 as f64;
            acc.hf_abs_dst += diff2 as f64;

            // MSE
            let pd = sv - dv;
            acc.mse += (pd * pd) as f64;
        }

        // Slide V-window for the next inner row, unless this was the last.
        if step + 1 < inner_h {
            let next_y = inner_y + 1;
            // Row leaving: top of current window = inner_y - r.
            // Row entering: new bottom = inner_y + r + 1 (for next_y center).
            let row_leaving = if inner_y >= radius {
                inner_y - radius
            } else {
                radius - inner_y
            }
            .min(strip_h - 1);
            let row_entering = {
                let raw = next_y as isize + radius as isize;
                if raw >= strip_h as isize {
                    (2 * (strip_h as isize - 1) - raw).max(0) as usize
                } else {
                    raw as usize
                }
            }
            .min(strip_h - 1);

            // Subtract row_leaving from V-sums BEFORE overwriting in ring.
            // Slot is the same as row_entering's % diam, since we're sliding
            // by exactly 1 each step.
            let slot_old = row_leaving % diam;
            let mu1_lo = &mu1_full[row_leaving * width..row_leaving * width + width];
            let mu2_lo = &mu2_full[row_leaving * width..row_leaving * width + width];
            let ring_sq_old =
                &scratch.sigma_ring_sq[slot_old * width..slot_old * width + width];
            let ring_12_old =
                &scratch.sigma_ring_12[slot_old * width..slot_old * width + width];
            for x in 0..width {
                scratch.col_v_mu1[x] -= mu1_lo[x];
                scratch.col_v_mu2[x] -= mu2_lo[x];
                scratch.col_v_sumsq[x] -= ring_sq_old[x];
                scratch.col_v_sumprod[x] -= ring_12_old[x];
            }

            // H-blur new row into its slot. In the interior they're the
            // same slot (`row_entering - row_leaving == diam`); at top/
            // bottom boundaries the mirror reflection breaks that — the
            // entering row goes into a different slot, and the leaving
            // row's slot just stays put with stale-but-still-needed data
            // for upcoming iterations that still see that row.
            let slot_new = row_entering % diam;
            let src_row_new = &src[row_entering * width..row_entering * width + width];
            let dst_row_new = &dst[row_entering * width..row_entering * width + width];
            let ring_sq_new =
                &mut scratch.sigma_ring_sq[slot_new * width..slot_new * width + width];
            let ring_12_new =
                &mut scratch.sigma_ring_12[slot_new * width..slot_new * width + width];
            h_blur_sigma_row(
                src_row_new,
                dst_row_new,
                ring_sq_new,
                ring_12_new,
                width,
                radius,
            );

            // Add new row to V-sums.
            let mu1_hi = &mu1_full[row_entering * width..row_entering * width + width];
            let mu2_hi = &mu2_full[row_entering * width..row_entering * width + width];
            for x in 0..width {
                scratch.col_v_mu1[x] += mu1_hi[x];
                scratch.col_v_mu2[x] += mu2_hi[x];
                scratch.col_v_sumsq[x] += ring_sq_new[x];
                scratch.col_v_sumprod[x] += ring_12_new[x];
            }
        }
    }

    acc
}

/// AVX-512 SIMD inner loop: feature emission + V-window slide for the
/// streaming SSIM kernel. Same math as the scalar `for x in 0..width`
/// loop above, just 16 lanes wide. Falls back to scalar for the column
/// remainder past `width / 16 * 16`.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn run_inner_loop_v4(
    token: archmage::X64V4Token,
    acc: &mut StripChannelAccum,
    mu1_full: &[f32],
    mu2_full: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    strip_h: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    scratch: &mut StreamingSsimScratch,
) {
    let diam = 2 * radius + 1;
    let inv_n = 1.0 / diam as f32;
    let inv_n_v = f32x16::splat(token, inv_n);
    let one = f32x16::splat(token, 1.0);
    let two = f32x16::splat(token, 2.0);
    let c2v = f32x16::splat(token, C2);
    let zero = f32x16::zero(token);

    let col_chunks = width / 16;
    let tail_start = col_chunks * 16;

    for step in 0..inner_h {
        let inner_y = inner_start + step;
        let pixel_row = inner_y * width;

        // Determine slide boundary for this step. We process emit and slide
        // together in a single cg loop, so the H-blur of the new σ row must
        // happen BEFORE the cg loop (the slide section reads from the new
        // ring slot).
        let slide = if step + 1 < inner_h {
            let next_y = inner_y + 1;
            let row_leaving = if inner_y >= radius {
                inner_y - radius
            } else {
                radius - inner_y
            }
            .min(strip_h - 1);
            let row_entering = {
                let raw = next_y as isize + radius as isize;
                if raw >= strip_h as isize {
                    (2 * (strip_h as isize - 1) - raw).max(0) as usize
                } else {
                    raw as usize
                }
            }
            .min(strip_h - 1);
            let slot_old = row_leaving % diam;
            let slot_new = row_entering % diam;

            // H-blur new row's σ for the full width into ring slot_new.
            // (This OVERWRITES slot_new's data — but slot_new == slot_old
            // in the interior, so we must subtract from V-sums BEFORE this
            // overwrites. That happens inside the cg loop below.)
            //
            // We can't H-blur yet — the cg loop below subtracts old slot
            // values first. So we stash the slide info and H-blur after
            // all cgs have subtracted.
            Some((row_leaving, row_entering, slot_old, slot_new))
        } else {
            None
        };

        // Single cg loop: load v_*, emit, subtract row_leaving (still cached
        // in ring), store v_* back. After the cg loop we H-blur new row σ
        // and run a second short cg loop to add new row contributions.
        for cg in 0..col_chunks {
            let col_base = cg * 16;
            let off = col_base;

            // Load V-sums into registers (single round-trip per cg per step).
            let mut v_mu1 = f32x16::from_array(
                token,
                scratch.col_v_mu1[off..off + 16].try_into().unwrap(),
            );
            let mut v_mu2 = f32x16::from_array(
                token,
                scratch.col_v_mu2[off..off + 16].try_into().unwrap(),
            );
            let mut v_sumsq = f32x16::from_array(
                token,
                scratch.col_v_sumsq[off..off + 16].try_into().unwrap(),
            );
            let mut v_sumprod = f32x16::from_array(
                token,
                scratch.col_v_sumprod[off..off + 16].try_into().unwrap(),
            );

            // === Emit features ===
            let mu1 = v_mu1 * inv_n_v;
            let mu2 = v_mu2 * inv_n_v;
            let ssq = v_sumsq * inv_n_v;
            let s12 = v_sumprod * inv_n_v;

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

            let pixel_off = pixel_row + col_base;
            if store_sd {
                sd_out[pixel_off..pixel_off + 16].copy_from_slice(&sd.to_array());
            }
            if store_mu {
                mu1_out[pixel_off..pixel_off + 16].copy_from_slice(&mu1.to_array());
                mu2_out[pixel_off..pixel_off + 16].copy_from_slice(&mu2.to_array());
            }

            let s = f32x16::from_array(token, src[pixel_off..pixel_off + 16].try_into().unwrap());
            let d = f32x16::from_array(token, dst[pixel_off..pixel_off + 16].try_into().unwrap());
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

            // === Subtract row_leaving (kept in registers, then store) ===
            if let Some((row_leaving, _, slot_old, _)) = slide {
                let mu1_lo = f32x16::from_array(
                    token,
                    mu1_full[row_leaving * width + col_base..row_leaving * width + col_base + 16]
                        .try_into()
                        .unwrap(),
                );
                let mu2_lo = f32x16::from_array(
                    token,
                    mu2_full[row_leaving * width + col_base..row_leaving * width + col_base + 16]
                        .try_into()
                        .unwrap(),
                );
                let ring_sq_lo = f32x16::from_array(
                    token,
                    scratch.sigma_ring_sq[slot_old * width + col_base..slot_old * width + col_base + 16]
                        .try_into()
                        .unwrap(),
                );
                let ring_12_lo = f32x16::from_array(
                    token,
                    scratch.sigma_ring_12[slot_old * width + col_base..slot_old * width + col_base + 16]
                        .try_into()
                        .unwrap(),
                );
                v_mu1 = v_mu1 - mu1_lo;
                v_mu2 = v_mu2 - mu2_lo;
                v_sumsq = v_sumsq - ring_sq_lo;
                v_sumprod = v_sumprod - ring_12_lo;

                // Store back the post-subtract V-sums; the add pass below
                // re-loads and finalises after H-blur of the new σ row.
                scratch.col_v_mu1[off..off + 16].copy_from_slice(&v_mu1.to_array());
                scratch.col_v_mu2[off..off + 16].copy_from_slice(&v_mu2.to_array());
                scratch.col_v_sumsq[off..off + 16].copy_from_slice(&v_sumsq.to_array());
                scratch.col_v_sumprod[off..off + 16].copy_from_slice(&v_sumprod.to_array());
            }
        }

        // Scalar tail emission for cols past col_chunks * 16.
        for x in tail_start..width {
            let mu1 = scratch.col_v_mu1[x] * inv_n;
            let mu2 = scratch.col_v_mu2[x] * inv_n;
            let ssq = scratch.col_v_sumsq[x] * inv_n;
            let s12 = scratch.col_v_sumprod[x] * inv_n;
            let mu_diff = mu1 - mu2;
            let num_m = mu_diff.mul_add(-mu_diff, 1.0);
            let num_s = 2.0f32.mul_add((-mu1).mul_add(mu2, s12), C2);
            let denom_s = (-mu2).mul_add(mu2, (-mu1).mul_add(mu1, ssq)) + C2;
            let sd = (1.0f32 - (num_m * num_s) / denom_s).max(0.0);
            let sd2 = sd * sd;
            let sd4 = sd2 * sd2;
            acc.ssim_d += sd as f64;
            acc.ssim_d4 += sd4 as f64;
            acc.ssim_d2 += sd2 as f64;
            acc.ssim_d8 += (sd4 * sd4) as f64;
            acc.ssim_max = acc.ssim_max.max(sd);
            let pixel_off = pixel_row + x;
            if store_sd { sd_out[pixel_off] = sd; }
            if store_mu { mu1_out[pixel_off] = mu1; mu2_out[pixel_off] = mu2; }
            let sv = src[pixel_off];
            let dv = dst[pixel_off];
            let diff1 = (sv - mu1).abs();
            let diff2 = (dv - mu2).abs();
            let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0;
            let artifact = ed.max(0.0);
            let detail_lost = (-ed).max(0.0);
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

        // === Slide step 2: H-blur new σ row, then add ===
        if let Some((_, row_entering, slot_old, slot_new)) = slide {
            // Subtract σ for tail cols (separate loop; SIMD body already did).
            for x in tail_start..width {
                scratch.col_v_mu1[x] -= mu1_full[(if inner_y >= radius { inner_y - radius } else { radius - inner_y }).min(strip_h - 1) * width + x];
                scratch.col_v_mu2[x] -= mu2_full[(if inner_y >= radius { inner_y - radius } else { radius - inner_y }).min(strip_h - 1) * width + x];
                scratch.col_v_sumsq[x] -= scratch.sigma_ring_sq[slot_old * width + x];
                scratch.col_v_sumprod[x] -= scratch.sigma_ring_12[slot_old * width + x];
            }

            // H-blur the new σ row (full width, single call). We're already
            // inside the arcane region (run_inner_loop_v4), so call the v4
            // variant directly rather than going through the plain
            // `h_blur_sigma_row` dispatcher — that would cross two
            // target_feature boundaries per row (out and back in).
            let ring_sq_dst = &mut scratch.sigma_ring_sq[slot_new * width..slot_new * width + width];
            let ring_12_dst = &mut scratch.sigma_ring_12[slot_new * width..slot_new * width + width];
            h_blur_sigma_row_v4(
                token,
                &src[row_entering * width..row_entering * width + width],
                &dst[row_entering * width..row_entering * width + width],
                ring_sq_dst,
                ring_12_dst,
                width,
                radius,
            );

            // Add row_entering contributions (SIMD body + scalar tail).
            for cg in 0..col_chunks {
                let col_base = cg * 16;
                let off = col_base;
                let mut v_mu1 = f32x16::from_array(token, scratch.col_v_mu1[off..off + 16].try_into().unwrap());
                let mut v_mu2 = f32x16::from_array(token, scratch.col_v_mu2[off..off + 16].try_into().unwrap());
                let mut v_sumsq = f32x16::from_array(token, scratch.col_v_sumsq[off..off + 16].try_into().unwrap());
                let mut v_sumprod = f32x16::from_array(token, scratch.col_v_sumprod[off..off + 16].try_into().unwrap());
                let mu1_hi = f32x16::from_array(token, mu1_full[row_entering * width + col_base..row_entering * width + col_base + 16].try_into().unwrap());
                let mu2_hi = f32x16::from_array(token, mu2_full[row_entering * width + col_base..row_entering * width + col_base + 16].try_into().unwrap());
                let ring_sq_hi = f32x16::from_array(token, scratch.sigma_ring_sq[slot_new * width + col_base..slot_new * width + col_base + 16].try_into().unwrap());
                let ring_12_hi = f32x16::from_array(token, scratch.sigma_ring_12[slot_new * width + col_base..slot_new * width + col_base + 16].try_into().unwrap());
                v_mu1 = v_mu1 + mu1_hi;
                v_mu2 = v_mu2 + mu2_hi;
                v_sumsq = v_sumsq + ring_sq_hi;
                v_sumprod = v_sumprod + ring_12_hi;
                scratch.col_v_mu1[off..off + 16].copy_from_slice(&v_mu1.to_array());
                scratch.col_v_mu2[off..off + 16].copy_from_slice(&v_mu2.to_array());
                scratch.col_v_sumsq[off..off + 16].copy_from_slice(&v_sumsq.to_array());
                scratch.col_v_sumprod[off..off + 16].copy_from_slice(&v_sumprod.to_array());
            }
            for x in tail_start..width {
                scratch.col_v_mu1[x] += mu1_full[row_entering * width + x];
                scratch.col_v_mu2[x] += mu2_full[row_entering * width + x];
                scratch.col_v_sumsq[x] += scratch.sigma_ring_sq[slot_new * width + x];
                scratch.col_v_sumprod[x] += scratch.sigma_ring_12[slot_new * width + x];
            }
        }
    }
}

#[allow(dead_code)]
mod fbits {
    pub const _UNUSED: u32 = 0;
}
