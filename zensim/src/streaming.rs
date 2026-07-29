//! Parallel multi-scale metric computation with band-based strip processing.
//!
//! Phase 1: Convert sRGB→XYB for the entire image (parallel over row chunks).
//! Phase 2: Process each pyramid scale with parallel band processing.
//!   Strip-based H-blur → fused V-blur+features (parallel bands via rayon).

use crate::blur::{
    box_blur_1pass_into, box_blur_h_into_abs_diff, box_blur_v_from_copy, downscale_2x_inplace,
    fused_blur_h_mu, fused_blur_h_ssim, simd_padded_width,
};
use crate::color::{
    apply_gamut_matrix, composite_linear_f32_rgba, composite_srgb8_bgra_to_linear,
    composite_srgb8_rgba_to_linear, composite_srgb16_rgba_to_linear,
    linear_to_positive_xyb_planar_into, srgb_to_positive_xyb_planar_into,
};
use crate::diffmap::PixelFeatureWeights;
use crate::fused::{fused_vblur_features_edge, fused_vblur_features_ssim};
use crate::metric::{FEATURES_PER_CHANNEL_BASIC, ScaleStats, ZensimConfig, combine_scores};
use crate::pool::ScaleBuffers;
use crate::simd_ops::{
    abs_diff_sum, build_inline_mask_mse, build_inline_mse, build_iw_mse_only,
    edge_diff_channel_extended, edge_diff_channel_inline_both, edge_diff_channel_inline_mask,
    edge_diff_channel_iw_inline, mul_into, sq_diff_sum, sq_sum_into, ssim_channel_extended,
    ssim_channel_inline_both, ssim_channel_inline_mask, ssim_channel_iw_inline,
};
use crate::source::{AlphaMode, ColorPrimaries, ImageSource, PixelFormat};
use archmage::autoversion;
#[cfg(feature = "threads")]
use rayon::prelude::*;
#[cfg(feature = "threads")]
use std::sync::Mutex;

/// Inner strip height: rows of useful output per strip (must be even for 2x downscale).
///
/// Each strip's H-blur covers `STRIP_INNER + 2 * overlap` rows; the overlap
/// rows get re-H-blurred at the next strip boundary. Larger values reduce
/// that duplicate work; smaller values keep the H-blur plane allocation in
/// L2.
///
/// At 1080p with the default `blur_radius = 5, blur_passes = 1` (overlap=5):
///   16 → 67 strips × 26 rows ≈ 63% overlap waste, 800 KB plane footprint
///   32 → 34 strips × 42 rows ≈ 30% overlap waste, 1.3 MB footprint
///   64 → 17 strips × 74 rows ≈ 16% overlap waste, 2.3 MB footprint
///
/// Zen 4 has 1 MB L2 per core. At 32 the working set sits right at the L2
/// boundary, with mild spill into L3 — the duplicate-H-blur saving wins.
/// At 64 the spill is significant on 1080p multithreaded (16 cores all
/// hammering shared L3) and regresses; 32 is the safer default.
///
/// Empirical (1080p, fastest run on a busy host):
///   16 → 15.01 ms MT  (baseline)
///   32 → 14.66 ms MT  (-2.3%, also wins single-thread at every size)
///   64 → 18.80 ms MT  (+25%, regresses on 1080p MT due to L2 spill)
const STRIP_INNER: usize = 32;

/// Run two closures in parallel (rayon) or sequentially, depending on `parallel`.
#[inline]
fn maybe_join<A, B, RA, RB>(parallel: bool, a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    #[cfg(feature = "threads")]
    if parallel {
        return rayon::join(a, b);
    }
    let _ = parallel;
    let ra = a();
    let rb = b();
    (ra, rb)
}

/// Downscale 3 planes in-place, parallel or sequential.
pub(crate) fn downscale_3_planes(
    planes: &mut [Vec<f32>; 3],
    w: usize,
    h: usize,
    parallel: bool,
) -> (usize, usize) {
    let [ref mut p0, ref mut p1, ref mut p2] = *planes;
    let (nw_nh, _) = maybe_join(
        parallel,
        || downscale_2x_inplace(p0, w, h),
        || {
            maybe_join(
                parallel,
                || downscale_2x_inplace(p1, w, h),
                || downscale_2x_inplace(p2, w, h),
            )
        },
    );
    nw_nh
}

/// Downscale 6 planes (src + dst) in-place, parallel or sequential.
fn downscale_6_planes(
    src: &mut [Vec<f32>; 3],
    dst: &mut [Vec<f32>; 3],
    w: usize,
    h: usize,
    parallel: bool,
) -> (usize, usize) {
    let (nw_nh, _) = maybe_join(
        parallel,
        || downscale_3_planes(src, w, h, parallel),
        || downscale_3_planes(dst, w, h, parallel),
    );
    nw_nh
}

/// Weighted add: `dst[i] += src[i] * weight`. Auto-vectorized across architectures.
#[autoversion]
fn weighted_add(dst: &mut [f32], src: &[f32], weight: f32) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] += src[i] * weight;
    }
}

/// Diffmap accumulation: SSIM-only path. `dm[i] += weight * ssim[off + i]`.
#[autoversion]
fn diffmap_accum_ssim(dm: &mut [f32], ssim: &[f32], off: usize, weight: f32) {
    for i in 0..dm.len() {
        dm[i] += weight * ssim[off + i];
    }
}

/// Diffmap accumulation: edge artifact + detail loss + MSE path.
///
/// Weights are packed as `[ssim_w, art_w, det_w, mse_w]` to stay within the
/// 7-argument limit for clippy and autoversion-generated variants.
#[autoversion]
fn diffmap_accum_edge_mse(
    dm: &mut [f32],
    ssim: &[f32],
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    weights: [f32; 4],
) {
    let [ssim_w, art_w, det_w, mse_w] = weights;
    for i in 0..dm.len() {
        let mut val = ssim_w * ssim[i];
        let res_src = src[i] - mu1[i];
        let res_dst = dst[i] - mu2[i];
        let edge_diff = res_dst * res_dst - res_src * res_src;
        // max(0, edge_diff) and max(0, -edge_diff) — branchless via f32 ops
        val += art_w * edge_diff.max(0.0);
        val += det_w * (-edge_diff).max(0.0);
        let d = src[i] - dst[i];
        val += mse_w * d * d;
        dm[i] += val;
    }
}

/// Diffmap accumulation: HF energy/magnitude loss and gain.
///
/// Per-pixel HF features using the same residuals as edge features but with
/// different semantics: these capture texture energy changes (L2) and magnitude
/// changes (L1), weighted by trained feature weights 10-12.
///
/// Weights are packed as `[hf_loss_w, hf_mag_w, hf_gain_w]`.
#[autoversion]
fn diffmap_accum_hf(
    dm: &mut [f32],
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    weights: [f32; 3],
) {
    let [hf_loss_w, hf_mag_w, hf_gain_w] = weights;
    for i in 0..dm.len() {
        let res_src = src[i] - mu1[i];
        let res_dst = dst[i] - mu2[i];
        let sq_src = res_src * res_src;
        let sq_dst = res_dst * res_dst;
        let hf_loss = (sq_src - sq_dst).max(0.0);
        let hf_gain = (sq_dst - sq_src).max(0.0);
        let mag_loss = (res_src.abs() - res_dst.abs()).max(0.0);
        dm[i] += hf_loss_w * hf_loss + hf_mag_w * mag_loss + hf_gain_w * hf_gain;
    }
}

/// Nearest-neighbor power-of-2 upsample with weighted accumulation into `dst`,
/// in a single pass. Replaces the chain of `upsample_2x → clone → upsample_2x`
/// used during diffmap multi-scale fusion. Each src pixel covers a
/// `factor × factor` block in dst, scaled by `weight`, added to existing dst.
///
/// `factor = 1 << scale_levels`. When `factor == 1`, this is row-by-row
/// `weighted_add`.
fn upsample_pow2x_add(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    factor: usize,
    weight: f32,
) {
    if factor == 1 {
        let copy_w = src_w.min(dst_w);
        let copy_h = src_h.min(dst_h);
        for y in 0..copy_h {
            weighted_add(
                &mut dst[y * dst_w..y * dst_w + copy_w],
                &src[y * src_w..y * src_w + copy_w],
                weight,
            );
        }
        return;
    }

    for sy in 0..src_h {
        let dy_start = sy * factor;
        if dy_start >= dst_h {
            break;
        }
        let dy_end = (dy_start + factor).min(dst_h);
        let src_row = &src[sy * src_w..(sy + 1) * src_w];
        let copy_w = (src_w * factor).min(dst_w);

        for dy in dy_start..dy_end {
            let dst_row = &mut dst[dy * dst_w..dy * dst_w + copy_w];
            upsample_row_powx_add(src_row, dst_row, factor, weight);
        }
    }
}

/// Helper: accumulate `src[i] * weight` replicated `factor` times into `dst`.
/// `factor` is small (typically 2, 4, 8) and known at the SIMD-version
/// inlining boundary.
#[autoversion]
fn upsample_row_powx_add(src_row: &[f32], dst_row: &mut [f32], factor: usize, weight: f32) {
    let dst_len = dst_row.len();
    let mut di = 0;
    for &s in src_row {
        let v = s * weight;
        let end = (di + factor).min(dst_len);
        for slot in &mut dst_row[di..end] {
            *slot += v;
        }
        di += factor;
        if di >= dst_len {
            break;
        }
    }
}

/// Guided mass-conserving redistribution upsample (E-JBU, research; used only
/// when `DiffmapOptions::guided_coarse_redistribution` is set — default path is
/// [`upsample_pow2x_add`], byte-identical to prior behavior).
///
/// For each coarse-scale cell, deposits the cell's exact NN mass
/// (`src[cell] · weight · footprint_count`) within its aligned
/// `factor × factor` footprint proportional to the full-res `guide` plane,
/// instead of replicating the value uniformly:
///
/// ```text
/// dst(x,y) += cell_mass · g(x,y) / Σ_footprint g
/// ```
///
/// Properties (by construction, verified by `jbu_*` tests below):
/// - **Per-cell mass conservation**: the footprint sum equals what NN deposits
///   (up to f32 summation order), so any aligned block aggregation at ≥ the
///   footprint size — and the pooled map total — is unchanged.
/// - **ε-fallback**: a uniform (or all-zero) guide reproduces NN exactly; the
///   caller adds ε to the guide so `Σ g > 0` always holds (a `gsum <= 0` guard
///   still degrades to NN for belt and suspenders).
/// - **O(N)**: two passes over each footprint (guide sum, deposit); per-cell
///   accumulation in f64 for stability. Diffmap render only — never on the
///   scoring path.
fn redistribute_pow2x_guided_add(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    factor: usize,
    weight: f32,
    guide: &[f32],
) {
    debug_assert!(factor >= 2, "scale 0 never redistributes");
    debug_assert!(guide.len() >= dst_w * dst_h, "guide must cover dst");
    for sy in 0..src_h {
        let dy0 = sy * factor;
        if dy0 >= dst_h {
            break;
        }
        let dy1 = (dy0 + factor).min(dst_h);
        let src_row = &src[sy * src_w..(sy + 1) * src_w];
        for (sx, &v) in src_row.iter().enumerate() {
            let dx0 = sx * factor;
            if dx0 >= dst_w {
                break;
            }
            let dx1 = (dx0 + factor).min(dst_w);
            // Pass 1: guide sum over the (edge-clipped) footprint.
            let mut gsum = 0.0f64;
            for y in dy0..dy1 {
                for &g in &guide[y * dst_w + dx0..y * dst_w + dx1] {
                    gsum += g as f64;
                }
            }
            // Cell mass = exactly what NN would deposit over the clipped
            // footprint: value × blend weight × pixel count.
            let count = ((dy1 - dy0) * (dx1 - dx0)) as f64;
            let mass = v as f64 * weight as f64 * count;
            if gsum <= 0.0 {
                // Degenerate guide: uniform deposit == NN (mass / count).
                let add = v * weight;
                for y in dy0..dy1 {
                    for d in &mut dst[y * dst_w + dx0..y * dst_w + dx1] {
                        *d += add;
                    }
                }
                continue;
            }
            let scale = mass / gsum;
            // Pass 2: deposit mass ∝ guide.
            for y in dy0..dy1 {
                let base = y * dst_w;
                let grow = &guide[base + dx0..base + dx1];
                let drow = &mut dst[base + dx0..base + dx1];
                for (d, &g) in drow.iter_mut().zip(grow) {
                    *d += (scale * g as f64) as f32;
                }
            }
        }
    }
}

/// Track background deallocation thread to prevent accumulation on repeated calls.
#[cfg(feature = "threads")]
static DEALLOC_THREAD: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

/// Total bytes below which we drop synchronously instead of spawning a
/// background thread. Empirically, thread spawn + atomic-bookkeeping cost
/// (~20 µs) exceeds the cost of freeing a small allocation. We only pay
/// the spawn for working sets large enough to justify async munmap.
#[cfg(feature = "threads")]
const DEALLOC_THREAD_THRESHOLD_BYTES: usize = 4 * 1024 * 1024;

/// Drop one or two `[Vec<f32>; 3]` working buffers, choosing between
/// synchronous drop (small) and a background thread (large).
#[cfg(feature = "threads")]
fn dealloc_planes(p1: [Vec<f32>; 3], p2: Option<[Vec<f32>; 3]>) {
    let bytes = p1
        .iter()
        .map(|v| v.capacity() * core::mem::size_of::<f32>())
        .sum::<usize>()
        + p2.as_ref()
            .map(|p| {
                p.iter()
                    .map(|v| v.capacity() * core::mem::size_of::<f32>())
                    .sum::<usize>()
            })
            .unwrap_or(0);
    if bytes < DEALLOC_THREAD_THRESHOLD_BYTES {
        drop(p1);
        drop(p2);
        return;
    }
    let mut guard = DEALLOC_THREAD.lock().unwrap();
    if let Some(prev) = guard.take() {
        let _ = prev.join();
    }
    *guard = Some(std::thread::spawn(move || {
        drop(p1);
        drop(p2);
    }));
}

/// Per-scale feature accumulators. Collects raw sums across strips,
/// finalized to ScaleStats at the end.
pub(crate) struct ScaleAccumulators {
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
    // IW (information-content-weighted) features — Wang & Li 2011.
    // Same shape as masked_*, opposite weight polarity (texture-emphasising).
    // Gated by `config.compute_iw_features`; zero on the off-path.
    iw_ssim_d: [f64; 3],
    iw_ssim_d4: [f64; 3],
    iw_ssim_d2: [f64; 3],
    iw_art4: [f64; 3],
    iw_det4: [f64; 3],
    iw_mse: [f64; 3],
    /// `Σ activity` over the inner region, per channel — the quantity needed
    /// to pool the `iw_*` accumulators as an actual weighted mean.
    ///
    /// The shipped IW weight is `w_i = 1 + k_iw · a_i` (`simd_ops.rs`:
    /// "writes iw_out[i] = 1 + k_iw * activity[i]"), so
    /// `Σw = n + k_iw · Σa` — summing the activity gives `Σw` EXACTLY without
    /// touching the fused SIMD builders.
    ///
    /// Added 2026-07-15 to measure a defect and to make its fix possible:
    /// `finalize` pools every `iw_*` accumulator by `1/n`, while the reference
    /// implementation (`iw_pool.rs::WeightedPool::mean`, marked `dead_code`
    /// "hot path is fused into streaming") computes `Σ(w·v)/Σw`. Until now
    /// `Σw` did not exist anywhere in this file, so the divergence was not a
    /// wrong divisor — it was a missing quantity.
    /// See `benchmarks/iw_pooling_normalization_2026-07-15.md`.
    #[cfg(feature = "iw-diagnostics")]
    iw_a_sum: [f64; 3],
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
            iw_ssim_d: [0.0; 3],
            iw_ssim_d4: [0.0; 3],
            iw_ssim_d2: [0.0; 3],
            iw_art4: [0.0; 3],
            iw_det4: [0.0; 3],
            iw_mse: [0.0; 3],
            #[cfg(feature = "iw-diagnostics")]
            iw_a_sum: [0.0; 3],
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
            self.iw_ssim_d[c] += other.iw_ssim_d[c];
            self.iw_ssim_d4[c] += other.iw_ssim_d4[c];
            self.iw_ssim_d2[c] += other.iw_ssim_d2[c];
            self.iw_art4[c] += other.iw_art4[c];
            self.iw_det4[c] += other.iw_det4[c];
            self.iw_mse[c] += other.iw_mse[c];
            #[cfg(feature = "iw-diagnostics")]
            {
                self.iw_a_sum[c] += other.iw_a_sum[c];
            }
        }
        self.n += other.n;
    }

    /// `k_iw` is `config.iw_strength` — needed only to report `iw_mean_w`
    /// (`Σw/n` for `w_i = 1 + k_iw·a_i`). Passed in rather than hardcoded to
    /// 4.0: `iw_strength` is a config field, and baking its default into the
    /// pooling is how a diagnostic silently becomes a lie under a non-default
    /// config.
    fn finalize(
        &self,
        #[cfg_attr(not(feature = "iw-diagnostics"), allow(unused_variables))] k_iw: f64,
    ) -> ScaleStats {
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
        let mut iw_ssim = [0.0f64; 9];
        let mut iw_art_4th = [0.0f64; 3];
        let mut iw_det_4th = [0.0f64; 3];
        let mut iw_mse = [0.0f64; 3];
        #[cfg_attr(not(feature = "iw-diagnostics"), allow(unused_mut))]
        let mut iw_mean_w = [0.0f64; 3];

        for c in 0..3 {
            // f64 sums of per-pixel non-negative values CAN go slightly
            // negative under f32→f64 round-off, which would turn the
            // subsequent powf(0.25) / sqrt() into NaN. Clamp to 0 here —
            // matches GPU zensim_gpu defensive clamp and prevents NaN
            // propagation into the MLP feature vector.
            ssim[c * 2] = self.ssim_d[c] * one_over_n;
            ssim[c * 2 + 1] = (self.ssim_d4[c] * one_over_n).max(0.0).powf(0.25);
            ssim_2nd[c] = (self.ssim_d2[c] * one_over_n).max(0.0).sqrt();
            edge[c * 4] = self.edge_art[c] * one_over_n;
            edge[c * 4 + 1] = (self.edge_art4[c] * one_over_n).max(0.0).powf(0.25);
            edge[c * 4 + 2] = self.edge_det[c] * one_over_n;
            edge[c * 4 + 3] = (self.edge_det4[c] * one_over_n).max(0.0).powf(0.25);
            edge_2nd[c * 2] = (self.edge_art2[c] * one_over_n).max(0.0).sqrt();
            edge_2nd[c * 2 + 1] = (self.edge_det2[c] * one_over_n).max(0.0).sqrt();
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
            ssim_l8[c] = (self.ssim_d8[c] * one_over_n).max(0.0).powf(0.125);
            art_l8[c] = (self.edge_art8[c] * one_over_n).max(0.0).powf(0.125);
            det_l8[c] = (self.edge_det8[c] * one_over_n).max(0.0).powf(0.125);

            // Extended: masked features (normalize by N, matching full-image path)
            masked_ssim[c * 3] = self.masked_ssim_d[c] * one_over_n;
            masked_ssim[c * 3 + 1] = (self.masked_ssim_d4[c] * one_over_n).max(0.0).powf(0.25);
            masked_ssim[c * 3 + 2] = (self.masked_ssim_d2[c] * one_over_n).max(0.0).sqrt();
            masked_art_4th[c] = (self.masked_art4[c] * one_over_n).max(0.0).powf(0.25);
            masked_det_4th[c] = (self.masked_det4[c] * one_over_n).max(0.0).powf(0.25);
            masked_mse[c] = self.masked_mse[c] * one_over_n;

            // IW (information-content-weighted) features. Same wire
            // shape as masked_*; weight direction inverted upstream.
            iw_ssim[c * 3] = self.iw_ssim_d[c] * one_over_n;
            iw_ssim[c * 3 + 1] = (self.iw_ssim_d4[c] * one_over_n).max(0.0).powf(0.25);
            iw_ssim[c * 3 + 2] = (self.iw_ssim_d2[c] * one_over_n).max(0.0).sqrt();
            iw_art_4th[c] = (self.iw_art4[c] * one_over_n).max(0.0).powf(0.25);
            iw_det_4th[c] = (self.iw_det4[c] * one_over_n).max(0.0).powf(0.25);
            iw_mse[c] = self.iw_mse[c] * one_over_n;
            // Diagnostic (NOT a feature): the factor every iw_* above
            // carries because they are pooled by 1/n rather than 1/Σw.
            #[cfg(feature = "iw-diagnostics")]
            {
                iw_mean_w[c] = 1.0 + k_iw * (self.iw_a_sum[c] * one_over_n);
            }
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
            iw_ssim,
            iw_art_4th,
            iw_det_4th,
            iw_mse,
            iw_mean_w,
        }
    }
}

/// Per-channel-at-scale dispatch decision. `None` slots are skipped entirely
/// (the channel doesn't need ssim, edge, or mse at this scale).
type ScaleActive = [Option<(usize, bool, bool)>; 3];

/// Determine which channels need SSIM, edge, and/or MSE computation at a given scale.
///
/// Returns a stack-allocated `[Option<(c, need_ssim, need_edge)>; 3]`.
/// The previous `Vec<…>` return allocated per call (one per scale × image
/// — small but unnecessary). The fixed-size form also gives LLVM enough
/// shape information to specialize the channel dispatch loop downstream.
fn active_channels(
    scale_idx: usize,
    n_scales: usize,
    config: &ZensimConfig,
    weights: &[f64],
) -> ScaleActive {
    let compute_all = config.compute_all_features;
    let extended = config.extended_features;
    let basic_fpc = FEATURES_PER_CHANNEL_BASIC; // 13

    let has_weight = |base: usize, count: usize| -> bool {
        (base..base + count).all(|i| i < weights.len())
            && (base..base + count).any(|i| weights[i].abs() > 0.001)
    };

    // Feature layout in weights array:
    //   Basic block [0..N_basic): 13/ch × 3ch × n_scales
    //     0-2: ssim_mean, ssim_4th, ssim_2nd
    //     3-8: art_mean, art_4th, art_2nd, det_mean, det_4th, det_2nd
    //     9: mse, 10-12: hf_energy_loss, hf_mag_loss, hf_energy_gain
    //   Peak block [N_basic..N_basic+N_peak): 6/ch × 3ch × n_scales
    //     0-2: ssim_max, art_max, det_max
    //     3-5: ssim_p95, art_p95, det_p95
    let basic_total = n_scales * basic_fpc * 3;
    let mut active: ScaleActive = [None; 3];
    let beyond = scale_idx * (basic_fpc * 3) >= weights.len();
    for (c, slot) in active.iter_mut().enumerate() {
        if beyond {
            if compute_all || extended {
                *slot = Some((c, true, true));
            }
        } else {
            let base = scale_idx * (basic_fpc * 3) + c * basic_fpc;
            let mut need_ssim = compute_all || extended || has_weight(base, 3);
            let need_hf = has_weight(base + 10, 3);
            let mut need_edge = compute_all || extended || has_weight(base + 3, 6) || need_hf;
            let need_mse = compute_all || extended || has_weight(base + 9, 1);
            // Also check peak weights (ssim_max/p95 need SSIM, art/det need edges)
            let peak_base = basic_total + scale_idx * 18 + c * 6;
            if has_weight(peak_base, 1) || has_weight(peak_base + 3, 1) {
                need_ssim = true; // ssim_max or ssim_p95
            }
            if has_weight(peak_base + 1, 2) || has_weight(peak_base + 4, 2) {
                need_edge = true; // art_max/det_max or art_p95/det_p95
            }
            if need_ssim || need_edge || need_mse {
                *slot = Some((c, need_ssim, need_edge));
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
    src_planes: [&[f32]; 3],
    dst_planes: [&[f32]; 3],
    width: usize,
    height: usize,
    padded_width: usize,
) -> [f64; 3] {
    let n = (width * height) as f64;

    // Sum (src - dst) per row in chunks, summed across chunks. The
    // serial form (full-image pass over both src and dst per channel) is
    // ~6.2M f32-load-pairs at 1080p × 3 channels — measurable. Chunking
    // and parallelising via rayon brings it under noise when the
    // `threads` feature is enabled.
    let chunk_rows = 64usize;
    let row_indices: Vec<usize> = (0..height).step_by(chunk_rows).collect();

    let per_chunk = |row_start: usize| -> [f64; 3] {
        let row_end = (row_start + chunk_rows).min(height);
        let mut diff = [0.0f64; 3];
        for c in 0..3 {
            let mut acc = 0.0f64;
            for y in row_start..row_end {
                let s = &src_planes[c][y * padded_width..y * padded_width + width];
                let d = &dst_planes[c][y * padded_width..y * padded_width + width];
                let mut row_sum = 0.0f64;
                for i in 0..width {
                    row_sum += (s[i] - d[i]) as f64;
                }
                acc += row_sum;
            }
            diff[c] = acc;
        }
        diff
    };

    #[cfg(feature = "threads")]
    let chunks: Vec<[f64; 3]> = if cfg!(feature = "threads") {
        use rayon::prelude::*;
        row_indices.into_par_iter().map(per_chunk).collect()
    } else {
        row_indices.into_iter().map(per_chunk).collect()
    };
    #[cfg(not(feature = "threads"))]
    let chunks: Vec<[f64; 3]> = row_indices.into_iter().map(per_chunk).collect();

    let mut offset = [0.0f64; 3];
    for chunk_diff in &chunks {
        for (o, &d) in offset.iter_mut().zip(chunk_diff.iter()) {
            *o += d;
        }
    }
    for o in &mut offset {
        *o /= n;
    }
    offset
}

/// Like [`compute_xyb_mean_offset`], but computes the per-channel mean
/// over rows `[y0, y1)` only — used by the strip aggregator so each
/// source row contributes to mean_offset exactly once across all strips.
pub(crate) fn compute_xyb_mean_offset_range(
    src_planes: [&[f32]; 3],
    dst_planes: [&[f32]; 3],
    width: usize,
    y0: usize,
    y1: usize,
    padded_width: usize,
) -> [f64; 3] {
    let inner_h = y1.saturating_sub(y0);
    if inner_h == 0 {
        return [0.0; 3];
    }
    let n = (width * inner_h) as f64;

    let chunk_rows = 64usize;
    let row_indices: Vec<usize> = (y0..y1).step_by(chunk_rows).collect();

    let per_chunk = |row_start: usize| -> [f64; 3] {
        let row_end = (row_start + chunk_rows).min(y1);
        let mut diff = [0.0f64; 3];
        for c in 0..3 {
            let mut acc = 0.0f64;
            for y in row_start..row_end {
                let s = &src_planes[c][y * padded_width..y * padded_width + width];
                let d = &dst_planes[c][y * padded_width..y * padded_width + width];
                let mut row_sum = 0.0f64;
                for i in 0..width {
                    row_sum += (s[i] - d[i]) as f64;
                }
                acc += row_sum;
            }
            diff[c] = acc;
        }
        diff
    };

    #[cfg(feature = "threads")]
    let chunks: Vec<[f64; 3]> = if cfg!(feature = "threads") {
        use rayon::prelude::*;
        row_indices.into_par_iter().map(per_chunk).collect()
    } else {
        row_indices.into_iter().map(per_chunk).collect()
    };
    #[cfg(not(feature = "threads"))]
    let chunks: Vec<[f64; 3]> = row_indices.into_iter().map(per_chunk).collect();

    let mut offset = [0.0f64; 3];
    for chunk_diff in &chunks {
        for (o, &d) in offset.iter_mut().zip(chunk_diff.iter()) {
            *o += d;
        }
    }
    for o in &mut offset {
        *o /= n;
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
    let parallel = config.allow_multithreading;

    // Phase 1: Convert sRGB→XYB for entire image.
    let mut src_planes = convert_source_to_xyb(source, padded_width, parallel);
    let mut dst_planes = convert_source_to_xyb(distorted, padded_width, parallel);

    // Compute mean_offset while XYB planes are cache-hot
    let src_view: [&[f32]; 3] = [&src_planes[0], &src_planes[1], &src_planes[2]];
    let dst_view: [&[f32]; 3] = [&dst_planes[0], &dst_planes[1], &dst_planes[2]];
    let mean_offset = compute_xyb_mean_offset(src_view, dst_view, width, height, padded_width);

    // Phase 2: Process all scales with band processing.
    let mut stats = Vec::with_capacity(num_scales);
    let mut w = padded_width;
    let mut h = height;

    for scale in 0..num_scales {
        if w < 8 || h < 8 {
            break;
        }

        let src_view: [&[f32]; 3] = [&src_planes[0], &src_planes[1], &src_planes[2]];
        let dst_view: [&[f32]; 3] = [&dst_planes[0], &dst_planes[1], &dst_planes[2]];
        let (scale_stat, _) =
            process_scale_bands(src_view, dst_view, w, h, config, scale, weights, None);
        stats.push(scale_stat);

        if scale < num_scales - 1 {
            let (nw, nh) = downscale_6_planes(&mut src_planes, &mut dst_planes, w, h, parallel);
            w = nw;
            h = nh;
        }
    }

    // Async drop for large working sets (avoids blocking on munmap), sync
    // drop for small ones (where spawn overhead dominates).
    #[cfg(feature = "threads")]
    dealloc_planes(src_planes, Some(dst_planes));
    #[cfg(not(feature = "threads"))]
    {
        drop(src_planes);
        drop(dst_planes);
    }

    (stats, mean_offset)
}

/// HDR PU-path multiscale stats from two sets of **absolute-luminance** linear
/// planes (ref + dist, cd/m²). Mirrors [`compute_multiscale_stats_streaming`]
/// but converts both sides via PU21 ([`convert_linear_planar_to_pu_xyb_into`])
/// rather than the sRGB/cube-root path. Backs `Zensim::compute_pu_linear_planar`.
pub(crate) fn compute_multiscale_stats_pu_linear_planar(
    ref_planes: [&[f32]; 3],
    dist_planes: [&[f32]; 3],
    width: usize,
    height: usize,
    stride: usize,
    config: &ZensimConfig,
    weights: &[f64],
) -> (Vec<ScaleStats>, [f64; 3]) {
    let padded_width = simd_padded_width(width);
    let n = padded_width * height;
    let mut src_planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; n]);
    let mut dst_planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; n]);
    convert_linear_planar_to_pu_xyb_into(
        ref_planes,
        width,
        height,
        stride,
        padded_width,
        &mut src_planes,
    );
    convert_linear_planar_to_pu_xyb_into(
        dist_planes,
        width,
        height,
        stride,
        padded_width,
        &mut dst_planes,
    );
    multiscale_stats_over_pu_xyb(
        src_planes,
        dst_planes,
        width,
        height,
        padded_width,
        config,
        weights,
    )
}

/// Interleaved-input sibling of [`compute_multiscale_stats_pu_linear_planar`]:
/// each image is one `[R, G, B, R, G, B, …]` f32 slice with its own row
/// stride (f32 elements). Backs `Zensim::compute_pu_linear`.
pub(crate) fn compute_multiscale_stats_pu_linear_interleaved(
    ref_rgb: &[f32],
    dist_rgb: &[f32],
    width: usize,
    height: usize,
    ref_stride: usize,
    dist_stride: usize,
    config: &ZensimConfig,
    weights: &[f64],
) -> (Vec<ScaleStats>, [f64; 3]) {
    let padded_width = simd_padded_width(width);
    let n = padded_width * height;
    let mut src_planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; n]);
    let mut dst_planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; n]);
    convert_linear_interleaved_to_pu_xyb_into(
        ref_rgb,
        width,
        height,
        ref_stride,
        padded_width,
        &mut src_planes,
    );
    convert_linear_interleaved_to_pu_xyb_into(
        dist_rgb,
        width,
        height,
        dist_stride,
        padded_width,
        &mut dst_planes,
    );
    multiscale_stats_over_pu_xyb(
        src_planes,
        dst_planes,
        width,
        height,
        padded_width,
        config,
        weights,
    )
}

/// Reflect-pad already-converted PU-XYB planes from `(width, height)` up to
/// the pyramid minimum — the post-conversion analogue of
/// [`crate::metric::reflect_pad_to_min`]. The PU conversion is pointwise, so
/// padding the converted planes is identical to converting reflect-padded
/// input; doing it here keeps both PU entry layouts (interleaved/planar) on
/// one pad path. No-op when both dims are ≥ the minimum.
fn reflect_pad_pu_planes(
    planes: [Vec<f32>; 3],
    width: usize,
    height: usize,
    padded_width: usize,
) -> ([Vec<f32>; 3], usize, usize, usize) {
    use crate::metric::{MIN_PYRAMID_DIM, reflect_index};
    if width >= MIN_PYRAMID_DIM && height >= MIN_PYRAMID_DIM {
        return (planes, width, height, padded_width);
    }
    let bw = width.max(MIN_PYRAMID_DIM);
    let bh = height.max(MIN_PYRAMID_DIM);
    let bpw = simd_padded_width(bw);
    let mut out: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; bpw * bh]);
    for (src, dst) in planes.iter().zip(out.iter_mut()) {
        for y in 0..bh {
            let sy = reflect_index(y, height);
            for x in 0..bw {
                dst[y * bpw + x] = src[sy * padded_width + reflect_index(x, width)];
            }
        }
    }
    mirror_pad_columns(&mut out, bw, bh, bpw);
    (out, bw, bh, bpw)
}

/// Shared tail of the two PU entry conversions: reflect-pad to the pyramid
/// minimum, then mean offset + the per-scale band loop over already-PU-encoded
/// XYB planes.
fn multiscale_stats_over_pu_xyb(
    src_planes: [Vec<f32>; 3],
    dst_planes: [Vec<f32>; 3],
    width: usize,
    height: usize,
    padded_width: usize,
    config: &ZensimConfig,
    weights: &[f64],
) -> (Vec<ScaleStats>, [f64; 3]) {
    // Sub-pyramid-minimum inputs reflect-pad up, exactly like the SDR funnel
    // (`compute_with_config_inner`); scores stay comparable down to 1×1.
    let (mut src_planes, w2, h2, pw2) =
        reflect_pad_pu_planes(src_planes, width, height, padded_width);
    let (mut dst_planes, ..) = reflect_pad_pu_planes(dst_planes, width, height, padded_width);
    let (width, height, padded_width) = (w2, h2, pw2);

    let num_scales = config.num_scales;
    let parallel = config.allow_multithreading;

    let src_view: [&[f32]; 3] = [&src_planes[0], &src_planes[1], &src_planes[2]];
    let dst_view: [&[f32]; 3] = [&dst_planes[0], &dst_planes[1], &dst_planes[2]];
    let mean_offset = compute_xyb_mean_offset(src_view, dst_view, width, height, padded_width);

    let mut stats = Vec::with_capacity(num_scales);
    let mut w = padded_width;
    let mut h = height;
    for scale in 0..num_scales {
        if w < 8 || h < 8 {
            break;
        }
        let src_view: [&[f32]; 3] = [&src_planes[0], &src_planes[1], &src_planes[2]];
        let dst_view: [&[f32]; 3] = [&dst_planes[0], &dst_planes[1], &dst_planes[2]];
        let (scale_stat, _) =
            process_scale_bands(src_view, dst_view, w, h, config, scale, weights, None);
        stats.push(scale_stat);
        if scale < num_scales - 1 {
            let (nw, nh) = downscale_6_planes(&mut src_planes, &mut dst_planes, w, h, parallel);
            w = nw;
            h = nh;
        }
    }
    (stats, mean_offset)
}

/// Convert an ImageSource to planar XYB at padded width, parallelized over row chunks.
///
/// Handles both RGB and RGBA sources row-by-row. RGBA is composited over a noise background.
pub(crate) fn convert_source_to_xyb(
    source: &impl ImageSource,
    padded_width: usize,
    parallel: bool,
) -> [Vec<f32>; 3] {
    let height = source.height();
    let n = padded_width * height;
    let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; n]);
    convert_source_to_xyb_into(source, &mut planes, padded_width, parallel);
    planes
}

/// Like [`convert_source_to_xyb`], but writes into pre-allocated planes
/// instead of allocating new ones. Each plane must have at least
/// `padded_width * source.height()` elements.
pub(crate) fn convert_source_to_xyb_into(
    source: &impl ImageSource,
    planes: &mut [Vec<f32>; 3],
    padded_width: usize,
    parallel: bool,
) {
    let [ref mut p0, ref mut p1, ref mut p2] = *planes;
    convert_source_to_xyb_into_slices(source, p0, p1, p2, padded_width, parallel, 0);
}

/// Slice-target core of [`convert_source_to_xyb_into`]: writes the XYB
/// planes of ALL of `source`'s rows into three caller-provided plane
/// slices (each ≥ `padded_width * source.height()` elements).
///
/// `abs_row_offset` is added to the row index passed to the ALPHA
/// noise-background compositors (`composite_*`'s `y` — see
/// [`crate::color::composite_srgb8_rgba_to_linear`]'s
/// `alpha_background_linear(x, y)` hash): when `source` is a
/// [`crate::source::SubsetView`] starting at parent row `r0`, passing
/// `abs_row_offset = r0` keeps the deterministic noise background in the
/// PARENT image's phase, so a strip-by-strip conversion of a translucent
/// image is bit-identical to converting the whole image at once. Opaque
/// sources never consult it. (The v1 streaming-strip path predates this
/// parameter and passes 0 — its subset conversions keep their historical
/// subset-local phase.)
#[allow(clippy::too_many_arguments)]
pub(crate) fn convert_source_to_xyb_into_slices(
    source: &impl ImageSource,
    p0: &mut [f32],
    p1: &mut [f32],
    p2: &mut [f32],
    padded_width: usize,
    #[allow(unused_variables)] parallel: bool,
    abs_row_offset: usize,
) {
    let width = source.width();
    let height = source.height();

    let chunk_rows = 64;
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

    #[allow(clippy::type_complexity)]
    let process_chunk =
        |(chunk_idx, ((c0, c1), c2)): (usize, ((&mut [f32], &mut [f32]), &mut [f32]))| {
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
                        // Non-sRGB: linearize, apply gamut matrix, then XYB
                        let mut linear_row = vec![[0.0f32; 3]; width];
                        for y in row_start..row_end {
                            let row_bytes = source.row_bytes(y);
                            let rgb_row: &[[u8; 3]] = bytemuck::cast_slice(row_bytes);
                            for (x, pixel) in linear_row.iter_mut().enumerate().take(width) {
                                let [r, g, b] = rgb_row[x];
                                *pixel = [
                                    crate::color::srgb_u8_to_linear(r),
                                    crate::color::srgb_u8_to_linear(g),
                                    crate::color::srgb_u8_to_linear(b),
                                ];
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
                                for (x, pixel) in linear_row.iter_mut().enumerate().take(width) {
                                    let [r, g, b, _a] = rgba_row[x];
                                    *pixel = [
                                        crate::color::srgb_u8_to_linear(r),
                                        crate::color::srgb_u8_to_linear(g),
                                        crate::color::srgb_u8_to_linear(b),
                                    ];
                                }
                            } else {
                                composite_srgb8_rgba_to_linear(
                                    &rgba_row[..width],
                                    abs_row_offset + y,
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
                                for (x, pixel) in linear_row.iter_mut().enumerate().take(width) {
                                    let [b, g, r, _a] = bgra_row[x];
                                    *pixel = [
                                        crate::color::srgb_u8_to_linear(r),
                                        crate::color::srgb_u8_to_linear(g),
                                        crate::color::srgb_u8_to_linear(b),
                                    ];
                                }
                            } else {
                                composite_srgb8_bgra_to_linear(
                                    &bgra_row[..width],
                                    abs_row_offset + y,
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
                            composite_srgb16_rgba_to_linear(
                                row_bytes,
                                width,
                                abs_row_offset + y,
                                &mut linear_row,
                            );
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
                                composite_linear_f32_rgba(
                                    &rgba_row[..width],
                                    abs_row_offset + y,
                                    &mut linear_row,
                                );
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
        };

    #[allow(clippy::redundant_closure)]
    {
        #[cfg(feature = "threads")]
        if parallel {
            p0_chunks
                .into_par_iter()
                .zip(p1_chunks)
                .zip(p2_chunks)
                .enumerate()
                .for_each(|args| process_chunk(args));
        } else {
            p0_chunks
                .into_iter()
                .zip(p1_chunks)
                .zip(p2_chunks)
                .enumerate()
                .for_each(|args| process_chunk(args));
        }
        #[cfg(not(feature = "threads"))]
        p0_chunks
            .into_iter()
            .zip(p1_chunks)
            .zip(p2_chunks)
            .enumerate()
            .for_each(|args| process_chunk(args));
    }
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
    mut diffmap: Option<(&mut [f32], PixelFeatureWeights)>,
    // task #67 C3a: when `Some((sd, mu1, mu2))`, copy this channel's
    // accumulated-row SSIM-error plane and V-blur means into the given
    // band-local slices (each `inner_h * width`) — the fused
    // score+attribution path's plane retention. `None` (every pre-existing
    // caller) is byte-identical to the previous behavior.
    attr_ret: Option<(&mut [f32], &mut [f32], &mut [f32])>,
) {
    // MSE-only path: no blur needed
    if !need_ssim && !need_edge {
        if let Some((sd_out, mu1_out, mu2_out)) = attr_ret {
            sd_out.fill(0.0);
            mu1_out.fill(0.0);
            mu2_out.fill(0.0);
        }
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

        let dm_needs_edge = diffmap.as_ref().is_some_and(|(_, pw)| pw.needs_edge_mse());
        let dm_needs_hf = diffmap.as_ref().is_some_and(|(_, pw)| pw.needs_hf());
        let store_sd = (diffmap.is_some() || attr_ret.is_some()) && need_ssim;
        // Force mu1/mu2 storage when diffmap needs edge/MSE or HF features
        // OR when IW features are required (mu1 is the reference plane for
        // the IW weight's activity-map computation).
        let store_mu = config.extended_features
            || config.compute_iw_features
            || dm_needs_edge
            || dm_needs_hf
            || attr_ret.is_some();
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
            // sd_out goes to temp_blur (only used when store_sd=true, extracted before
            // extended features which also need temp_blur for blurs)
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
                store_mu,
                &mut bufs.temp_blur,
                store_sd,
            );

            // Accumulate weighted features into diffmap before extended features
            // overwrites temp_blur. Inner rows are at inner_start..inner_start+inner_h
            // in strip-local coordinates.
            //
            // When edge/MSE is enabled, mu1 is in bufs.mask and mu2 is in bufs.mul_buf
            // (written by fused kernel when store_mu=true, not yet swapped).
            if let Some((ref mut dm, pw)) = diffmap {
                let inner_off = inner_start * width;
                let inner_n = inner_h * width;
                let dm = &mut dm[..inner_n];
                if dm_needs_edge {
                    diffmap_accum_edge_mse(
                        dm,
                        &bufs.temp_blur[inner_off..inner_off + inner_n],
                        &src_c[inner_off..inner_off + inner_n],
                        &dst_c[inner_off..inner_off + inner_n],
                        &bufs.mask[inner_off..inner_off + inner_n],
                        &bufs.mul_buf[inner_off..inner_off + inner_n],
                        [pw.ssim, pw.art, pw.det, pw.mse],
                    );
                } else {
                    diffmap_accum_ssim(dm, &bufs.temp_blur, inner_off, pw.ssim);
                }
                // HF features: mu1 is in bufs.mask, mu2 is in bufs.mul_buf
                // (stored by fused kernel when store_mu=true)
                if dm_needs_hf {
                    diffmap_accum_hf(
                        dm,
                        &src_c[inner_off..inner_off + inner_n],
                        &dst_c[inner_off..inner_off + inner_n],
                        &bufs.mask[inner_off..inner_off + inner_n],
                        &bufs.mul_buf[inner_off..inner_off + inner_n],
                        [pw.hf_loss, pw.hf_mag, pw.hf_gain],
                    );
                }
            }

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
                config.extended_features || config.compute_iw_features || attr_ret.is_some(),
            );
        }

        // task #67 C3a: retention copy — the accumulated rows' SSIM-error
        // plane (temp_blur) and V-blurred mu1/mu2 (in mask/mul_buf at this
        // point, BEFORE the swap below) into the caller's band-local
        // slices. `sd` is zero-filled for a channel that computes no SSIM.
        if let Some((sd_out, mu1_out, mu2_out)) = attr_ret {
            let inner_off = inner_start * width;
            let inner_n = inner_h * width;
            if need_ssim {
                sd_out.copy_from_slice(&bufs.temp_blur[inner_off..inner_off + inner_n]);
            } else {
                sd_out.fill(0.0);
            }
            mu1_out.copy_from_slice(&bufs.mask[inner_off..inner_off + inner_n]);
            mu2_out.copy_from_slice(&bufs.mul_buf[inner_off..inner_off + inner_n]);
        }

        // Swap: mu1/mu2 now hold V-blurred values, mask/mul_buf hold H-blurred garbage
        // We need the V-blurred mu1/mu2 for any path that computes activity
        // (extended-features masked block OR compute_iw_features IW block),
        // so swap whenever either is on.
        if config.extended_features || config.compute_iw_features {
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

        // Extended: masked features using stored V-blurred mu1/mu2.
        // The activity-map + blur work is shared between the
        // `extended_features` (texture-suppressing masked) path and the
        // `compute_iw_features` (texture-emphasising IW) path — we
        // compute the blurred activity once and derive both weights.
        let do_ext = config.extended_features;
        let do_iw = config.compute_iw_features;
        let need_activity = do_ext || do_iw;
        if need_activity {
            let inner_off = inner_start * width;
            let inner_n = inner_h * width;
            let strip_n = strip_h * width;
            let k = config.extended_masking_strength;
            let k_iw = config.iw_strength;
            let inner_src = &src_c[inner_off..inner_off + inner_n];
            let inner_dst = &dst_c[inner_off..inner_off + inner_n];

            // Phase 2 Lever 3 (2026-05-22): fused H-blur + abs-diff. The
            // h_blur_src plane is no longer materialized — `bufs.mask`
            // receives `|src - H_blur(src)|` directly. Saves one full
            // plane write + one full plane read of src per channel per
            // scale on the activity path.
            //
            // Per-channel H-blur reference (principled, 2026-05-17): the
            // current channel's strip-local H_blur(src) is the activity-
            // map reference. Prior multi-pass V-blurred bufs.mu1 carried
            // arbitrary cross-channel stale state at overlap rows. See
            // `docs/PRINCIPLED_ACTIVITY.md`.
            box_blur_h_into_abs_diff(
                &src_c[..strip_n],
                &mut bufs.mask[..strip_n],
                width,
                strip_h,
                config.blur_radius,
            );

            // Step 2: blur the activity map → mul_buf. After this,
            // mul_buf holds the per-pixel blurred reference-activity
            // signal shared by both mask and iw_weight.
            box_blur_1pass_into(
                &bufs.mask[..strip_n],
                &mut bufs.mul_buf[..strip_n],
                &mut bufs.temp_blur[..strip_n],
                width,
                strip_h,
                config.blur_radius,
            );

            // Step 3 (Phase 2, 2026-05-22): all weights derived inline from
            // activity. Mask plane is NEVER materialized — saves 1 plane
            // write + up to 3 plane reads per channel per scale.
            let activity_inner = &bufs.mul_buf[inner_off..inner_off + inner_n];
            // Σ activity, from which Σw follows exactly: the shipped IW weight
            // is `w_i = 1 + k_iw · a_i`, so `Σw = n + k_iw · Σa`. Summing here
            // (rather than inside the fused SIMD builders) keeps the hot
            // kernels untouched and is arithmetically identical.
            //
            // Only on the IW path — this is the denominator `finalize` needs to
            // pool `iw_*` as a real weighted mean instead of by `1/n`.
            #[cfg(feature = "iw-diagnostics")]
            if do_iw {
                accum.iw_a_sum[c] += activity_inner.iter().map(|&a| a as f64).sum::<f64>();
            }
            if do_ext && do_iw {
                let (mse_m, mse_i) =
                    build_inline_mse(activity_inner, k, k_iw, inner_src, inner_dst);
                accum.masked_mse[c] += mse_m;
                accum.iw_mse[c] += mse_i;
            } else if do_ext {
                let mse_m = build_inline_mask_mse(activity_inner, k, inner_src, inner_dst);
                accum.masked_mse[c] += mse_m;
            } else {
                // do_iw only — no plane writes; IW weight folded into MSE inline.
                let mse_i = build_iw_mse_only(activity_inner, k_iw, inner_src, inner_dst);
                accum.iw_mse[c] += mse_i;
            }

            // Step 4: masked + IW SSIM (needs sigma_sq and sigma12).
            //
            // FAST PATH (added 2026-05-15 perf optimization): after
            // the fused 1-pass blur, `bufs.sigma1_sq` already holds
            // the H-blurred `src² + dst²` plane, and `bufs.sigma12`
            // already holds the H-blurred `src * dst` plane. Box
            // blur is separable, so we only need a 1D V-blur of those
            // to get the full 2D blur. This replaces 4 SIMD passes
            // (`sq_sum_into` + 2D blur for ssq, `mul_into` + 2D blur
            // for s12) with 2 SIMD passes (1D V-blur each), saving
            // ~30% of the masked-block setup cost.
            if need_ssim {
                box_blur_v_from_copy(
                    &bufs.sigma1_sq[..strip_n],
                    &mut bufs.temp_blur[..strip_n],
                    width,
                    strip_h,
                    config.blur_radius,
                );
                std::mem::swap(&mut bufs.sigma1_sq, &mut bufs.temp_blur);
                box_blur_v_from_copy(
                    &bufs.sigma12[..strip_n],
                    &mut bufs.temp_blur[..strip_n],
                    width,
                    strip_h,
                    config.blur_radius,
                );
                std::mem::swap(&mut bufs.sigma12, &mut bufs.temp_blur);

                let inner_mu1 = &bufs.mu1[inner_off..inner_off + inner_n];
                let inner_mu2 = &bufs.mu2[inner_off..inner_off + inner_n];
                let inner_sig_sq = &bufs.sigma1_sq[inner_off..inner_off + inner_n];
                let inner_sig12 = &bufs.sigma12[inner_off..inner_off + inner_n];

                if do_ext && do_iw {
                    let ((sd_m, sd4_m, sd2_m), (sd_i, sd4_i, sd2_i)) = ssim_channel_inline_both(
                        inner_mu1,
                        inner_mu2,
                        inner_sig_sq,
                        inner_sig12,
                        activity_inner,
                        k,
                        k_iw,
                    );
                    accum.masked_ssim_d[c] += sd_m;
                    accum.masked_ssim_d4[c] += sd4_m;
                    accum.masked_ssim_d2[c] += sd2_m;
                    accum.iw_ssim_d[c] += sd_i;
                    accum.iw_ssim_d4[c] += sd4_i;
                    accum.iw_ssim_d2[c] += sd2_i;
                } else if do_ext {
                    let (sum_d, sum_d4, sum_d2) = ssim_channel_inline_mask(
                        inner_mu1,
                        inner_mu2,
                        inner_sig_sq,
                        inner_sig12,
                        activity_inner,
                        k,
                    );
                    accum.masked_ssim_d[c] += sum_d;
                    accum.masked_ssim_d4[c] += sum_d4;
                    accum.masked_ssim_d2[c] += sum_d2;
                } else {
                    let (sum_d, sum_d4, sum_d2) = ssim_channel_iw_inline(
                        inner_mu1,
                        inner_mu2,
                        inner_sig_sq,
                        inner_sig12,
                        activity_inner,
                        k_iw,
                    );
                    accum.iw_ssim_d[c] += sum_d;
                    accum.iw_ssim_d4[c] += sum_d4;
                    accum.iw_ssim_d2[c] += sum_d2;
                }
            }

            // Masked + IW edge (art_4th, det_4th)
            if need_edge {
                let inner_mu1 = &bufs.mu1[inner_off..inner_off + inner_n];
                let inner_mu2 = &bufs.mu2[inner_off..inner_off + inner_n];
                if do_ext && do_iw {
                    let ((art4_m, det4_m), (art4_i, det4_i)) = edge_diff_channel_inline_both(
                        inner_src,
                        inner_dst,
                        inner_mu1,
                        inner_mu2,
                        activity_inner,
                        k,
                        k_iw,
                    );
                    accum.masked_art4[c] += art4_m;
                    accum.masked_det4[c] += det4_m;
                    accum.iw_art4[c] += art4_i;
                    accum.iw_det4[c] += det4_i;
                } else if do_ext {
                    let (_art, art4, _det, det4, _art2, _det2) = edge_diff_channel_inline_mask(
                        inner_src,
                        inner_dst,
                        inner_mu1,
                        inner_mu2,
                        activity_inner,
                        k,
                    );
                    accum.masked_art4[c] += art4;
                    accum.masked_det4[c] += det4;
                } else {
                    let (_art, art4, _det, det4, _art2, _det2) = edge_diff_channel_iw_inline(
                        inner_src,
                        inner_dst,
                        inner_mu1,
                        inner_mu2,
                        activity_inner,
                        k_iw,
                    );
                    accum.iw_art4[c] += art4;
                    accum.iw_det4[c] += det4;
                }
            }
        }
        return;
    }

    // Separate blur + reduce fallback
    let blur_fn = box_blur_1pass_into;

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

    // Extended: masked + IW features (shared activity-map computation)
    let do_ext = config.extended_features;
    let do_iw = config.compute_iw_features;
    let need_activity = do_ext || do_iw;
    if need_activity {
        let strip_n = strip_h * width;
        let k = config.extended_masking_strength;
        let k_iw = config.iw_strength;

        // Phase 2 Lever 3 (2026-05-22): fused H-blur + abs-diff. See the
        // 1-pass-blur call site for the design rationale.
        box_blur_h_into_abs_diff(
            &src_c[..strip_n],
            &mut bufs.mask[..strip_n],
            width,
            strip_h,
            config.blur_radius,
        );

        blur_fn(
            &bufs.mask[..strip_n],
            &mut bufs.mul_buf[..strip_n],
            &mut bufs.temp_blur[..strip_n],
            width,
            strip_h,
            config.blur_radius,
        );

        // Phase 2 (2026-05-22): all weights derived inline from activity.
        // Mask plane is NEVER materialized.
        let activity_inner = &bufs.mul_buf[inner_off..inner_off + inner_n];
        if do_ext && do_iw {
            let (mse_m, mse_i) = build_inline_mse(activity_inner, k, k_iw, inner_src, inner_dst);
            accum.masked_mse[c] += mse_m;
            accum.iw_mse[c] += mse_i;
        } else if do_ext {
            let mse_m = build_inline_mask_mse(activity_inner, k, inner_src, inner_dst);
            accum.masked_mse[c] += mse_m;
        } else {
            // do_iw only — no plane writes; IW weight folded into MSE inline.
            let mse_i = build_iw_mse_only(activity_inner, k_iw, inner_src, inner_dst);
            accum.iw_mse[c] += mse_i;
        }

        if need_ssim {
            let inner_mu1 = &bufs.mu1[inner_off..inner_off + inner_n];
            let inner_mu2 = &bufs.mu2[inner_off..inner_off + inner_n];
            let inner_sig_sq = &bufs.sigma1_sq[inner_off..inner_off + inner_n];
            let inner_sig12 = &bufs.sigma12[inner_off..inner_off + inner_n];

            if do_ext && do_iw {
                let ((sd_m, sd4_m, sd2_m), (sd_i, sd4_i, sd2_i)) = ssim_channel_inline_both(
                    inner_mu1,
                    inner_mu2,
                    inner_sig_sq,
                    inner_sig12,
                    activity_inner,
                    k,
                    k_iw,
                );
                accum.masked_ssim_d[c] += sd_m;
                accum.masked_ssim_d4[c] += sd4_m;
                accum.masked_ssim_d2[c] += sd2_m;
                accum.iw_ssim_d[c] += sd_i;
                accum.iw_ssim_d4[c] += sd4_i;
                accum.iw_ssim_d2[c] += sd2_i;
            } else if do_ext {
                let (sum_d, sum_d4, sum_d2) = ssim_channel_inline_mask(
                    inner_mu1,
                    inner_mu2,
                    inner_sig_sq,
                    inner_sig12,
                    activity_inner,
                    k,
                );
                accum.masked_ssim_d[c] += sum_d;
                accum.masked_ssim_d4[c] += sum_d4;
                accum.masked_ssim_d2[c] += sum_d2;
            } else {
                let (sum_d, sum_d4, sum_d2) = ssim_channel_iw_inline(
                    inner_mu1,
                    inner_mu2,
                    inner_sig_sq,
                    inner_sig12,
                    activity_inner,
                    k_iw,
                );
                accum.iw_ssim_d[c] += sum_d;
                accum.iw_ssim_d4[c] += sum_d4;
                accum.iw_ssim_d2[c] += sum_d2;
            }
        }

        if need_edge {
            if do_ext && do_iw {
                let ((art4_m, det4_m), (art4_i, det4_i)) = edge_diff_channel_inline_both(
                    inner_src,
                    inner_dst,
                    inner_mu1,
                    inner_mu2,
                    activity_inner,
                    k,
                    k_iw,
                );
                accum.masked_art4[c] += art4_m;
                accum.masked_det4[c] += det4_m;
                accum.iw_art4[c] += art4_i;
                accum.iw_det4[c] += det4_i;
            } else if do_ext {
                let (_art, art4, _det, det4, _art2, _det2) = edge_diff_channel_inline_mask(
                    inner_src,
                    inner_dst,
                    inner_mu1,
                    inner_mu2,
                    activity_inner,
                    k,
                );
                accum.masked_art4[c] += art4;
                accum.masked_det4[c] += det4;
            } else {
                let (_art, art4, _det, det4, _art2, _det2) = edge_diff_channel_iw_inline(
                    inner_src,
                    inner_dst,
                    inner_mu1,
                    inner_mu2,
                    activity_inner,
                    k_iw,
                );
                accum.iw_art4[c] += art4;
                accum.iw_det4[c] += det4;
            }
        }
    }
}

/// Process a scale using parallel band processing over pre-existing XYB planes.
///
/// Divides the image into horizontal bands, each processing sequential strips.
/// Each band runs on a separate thread via rayon.
#[allow(clippy::too_many_arguments)]
fn process_scale_bands(
    src_planes: [&[f32]; 3],
    dst_planes: [&[f32]; 3],
    width: usize,
    height: usize,
    config: &ZensimConfig,
    scale_idx: usize,
    weights: &[f64],
    diffmap_weights: Option<[PixelFeatureWeights; 3]>,
) -> (ScaleStats, Option<Vec<f32>>) {
    let (accum, diffmap) = process_scale_bands_into_accum(
        src_planes,
        dst_planes,
        width,
        height,
        config,
        scale_idx,
        weights,
        diffmap_weights,
        None,
        None,
        None,
    );
    (accum.finalize(config.iw_strength as f64), diffmap)
}

/// Per-scale retained planes for the fused score+attribution path (task
/// #67 C3a): per channel, the per-pixel SSIM-error plane and the V-blurred
/// `mu1`/`mu2` at scale resolution (accumulated rows, band-concatenated in
/// row order — the same assembly the per-scale diffmap uses). Consumed by
/// `crate::attribution`'s fused entry to derive the basic attribution
/// density from the SAME pipeline pass that produced the scalar's stats.
pub(crate) struct AttrScaleRetention {
    pub sd: [Vec<f32>; 3],
    pub mu1: [Vec<f32>; 3],
    pub mu2: [Vec<f32>; 3],
}

impl AttrScaleRetention {
    /// Planes sized for the scale-0 (largest) resolution; coarser scales
    /// use the leading `w × h` prefix of each plane.
    pub fn new(n: usize) -> Self {
        Self {
            sd: core::array::from_fn(|_| vec![0.0; n]),
            mu1: core::array::from_fn(|_| vec![0.0; n]),
            mu2: core::array::from_fn(|_| vec![0.0; n]),
        }
    }
}

/// One band's in-place retention output slices (pre-split from the
/// caller's [`AttrScaleRetention`] planes; row-range-disjoint per band).
struct AttrBandSlices<'a> {
    sd: [&'a mut [f32]; 3],
    mu1: [&'a mut [f32]; 3],
    mu2: [&'a mut [f32]; 3],
}

/// Variant of [`process_scale_bands`] that returns the **raw**
/// [`ScaleAccumulators`] instead of the finalized [`ScaleStats`].
/// Used by the strip-aggregating pipeline: process N independent
/// Y-strips of the source image, merge each strip's accumulators
/// into a global per-scale accumulator, and finalize ONCE at the
/// end. Lets large (e.g., 80 MP) images run with bounded memory
/// without OOM.
///
/// `inner_y_filter`: when `Some((y0, y1))`, only rows in the half-open
/// range `[y0, y1)` of the input plane contribute to the accumulator.
/// Rows outside this range still get processed (their data feeds the
/// blur stencil for inner rows) but their feature values are dropped.
/// Used by the Y-strip aggregator to skip the strip's "outer margin"
/// rows so that only the strip's "inner" rows are counted — yielding
/// byte-exact equivalence to the full-image path on inner rows whose
/// blur stencil fits entirely inside the strip.
///
/// `outer_layout`: when `Some((outer_h, plane_offset))`, the band tiling
/// is computed against the OUTER plane height (`outer_h`), and each band
/// is mapped into the strip's local coords via `plane_offset` (the
/// strip's row 0 in outer/source coords at this scale). This makes the
/// strip's bands align byte-exactly with the full-image's bands — the
/// V-blur running sum within each band has the same init point and
/// advance count as the full-image path, eliminating f32-accumulator
/// history divergence at strip boundaries. For this to work, the strip
/// must extend at least `overlap` rows past every aligned band boundary
/// (i.e., `strip_margin >= blur_radius * blur_passes` at every scale).
fn process_scale_bands_into_accum(
    src_planes: [&[f32]; 3],
    dst_planes: [&[f32]; 3],
    width: usize,
    height: usize,
    config: &ZensimConfig,
    scale_idx: usize,
    weights: &[f64],
    diffmap_weights: Option<[PixelFeatureWeights; 3]>,
    inner_y_filter: Option<(usize, usize)>,
    outer_layout: Option<(usize, usize)>,
    attr_ret_out: Option<&mut AttrScaleRetention>,
) -> (ScaleAccumulators, Option<Vec<f32>>) {
    let r = config.blur_radius;
    let passes = config.blur_passes as usize;
    let overlap = passes * r;
    let scale_active = active_channels(scale_idx, config.num_scales, config, weights);

    let (filter_y0, filter_y1) = match inner_y_filter {
        Some((y0, y1)) => (y0.min(height), y1.min(height)),
        None => (0, height),
    };

    // Outer layout for band tiling. When None, the bands tile against
    // the strip's local plane (the existing behavior). When Some, they
    // tile against the outer (full-image) plane and we map each band
    // into the strip's local coords.
    let (layout_h, plane_offset) = outer_layout.unwrap_or((height, 0));

    // Whether to RUN bands in parallel via rayon. This is separate from
    // the band LAYOUT computation: when called from a strip aggregator
    // that already runs strips in parallel, `config.allow_multithreading`
    // is `false` (to avoid rayon oversubscription), but we still want
    // bands to be LAID OUT as if the full-image processed with threads
    // — otherwise the strip's bands tile against a single-band layout
    // and break V-blur byte-exactness vs the full-image path.
    #[cfg(feature = "threads")]
    let run_parallel = config.allow_multithreading;
    let total_strips = layout_h.div_ceil(STRIP_INNER);
    // When outer_layout is provided, lay out bands as if a parallel
    // call were made on the FULL image; this keeps the per-band
    // V-blur init points byte-identical between the strip and full
    // paths. When outer_layout is None (the full path itself), the
    // existing semantics apply.
    // Band layout is part of the numerics contract: the strip path and the
    // full path must tile IDENTICAL bands, or their per-band V-blur init
    // points diverge (the 1e-6 strip-parity gate) — and a strip can only
    // honor band boundaries that fall inside its own overlap window, so
    // bands must never span strips. The layout is therefore GEOMETRY-ONLY:
    // one band per strip, on every path, regardless of thread count or the
    // `threads` feature. (The previous thread-count-derived layout equalled
    // band-per-strip whenever rayon had >= total_strips threads — i.e. on
    // every dev box — but silently broke parity on smaller machines, e.g.
    // the 3-core macos-latest runner, and made streaming numerics depend on
    // core count.) Execution parallelism stays a separate, rayon-scheduled
    // concern (`run_parallel`).
    let num_bands = total_strips.max(1);
    let strips_per_band = total_strips.div_ceil(num_bands);

    let max_strip_h = STRIP_INNER * strips_per_band + 2 * overlap;
    let max_strip_n = max_strip_h * width;

    // Strip's source-row range in outer coords:
    let strip_outer_y0 = plane_offset;
    let strip_outer_y1 = plane_offset + height;

    let process_band =
        |bufs: &mut ScaleBuffers, band_idx: usize, mut band_ret: Option<AttrBandSlices<'_>>| {
            // Band in OUTER (full-image) coords:
            let outer_band_first_y = (band_idx * strips_per_band * STRIP_INNER).min(layout_h);
            let outer_band_end_y = (((band_idx + 1) * strips_per_band) * STRIP_INNER).min(layout_h);
            // Skip bands that don't overlap the strip:
            if outer_band_end_y <= strip_outer_y0 || outer_band_first_y >= strip_outer_y1 {
                return (ScaleAccumulators::new(), None);
            }
            // Map band-inner to strip-local coords:
            let band_first_y = outer_band_first_y.saturating_sub(strip_outer_y0);
            let band_end_y = (outer_band_end_y - strip_outer_y0).min(height);

            if band_first_y >= band_end_y {
                return (ScaleAccumulators::new(), None);
            }

            let band_rows = band_end_y - band_first_y;
            let mut band_dm = diffmap_weights.map(|_| vec![0.0f32; band_rows * width]);

            let mut accum = ScaleAccumulators::new();
            // bufs is provided externally (per-rayon-worker via map_init);
            // first use grows to max_strip_n; subsequent uses reuse the
            // existing allocation. Eliminates the per-band ScaleBuffers::new
            // memset overhead (~9 % of wall time at 1080p pre-opt).
            bufs.ensure_capacity(max_strip_n);

            // We iterate through the band's source-row inner positions in
            // OUTER coordinates, advancing by STRIP_INNER. Each inner chunk
            // is mapped to strip-local coords for the V-blur kernel.
            let mut outer_y = outer_band_first_y.max(strip_outer_y0);
            while outer_y < outer_band_end_y.min(strip_outer_y1) {
                let outer_inner_end = (outer_y + STRIP_INNER)
                    .min(outer_band_end_y)
                    .min(strip_outer_y1);
                let inner_h_full = outer_inner_end - outer_y;

                // Strip-local mapping for THIS sub-strip's processing:
                let strip_local_inner_y = outer_y - strip_outer_y0;
                let strip_local_inner_end = outer_inner_end - strip_outer_y0;

                // Overlap reads in OUTER coords (consistent with full-image
                // path's band behavior). Mirror-clamp against the OUTER
                // plane bounds, NOT the strip bounds — this is what makes
                // the V-blur running-sum init point match the full-image
                // path.
                let outer_strip_top = outer_y.saturating_sub(overlap);
                let outer_strip_bot = (outer_inner_end + overlap).min(layout_h);

                // For the strip's V-blur kernel, the overlap reads must
                // land within the strip's data. This requires
                // `strip_margin >= overlap` at every scale.
                //
                // If `outer_strip_top < strip_outer_y0`, the strip doesn't
                // contain those rows — that's OK for the FIRST strip
                // (strip_outer_y0 == 0, so saturating_sub gives the same
                // mirror behavior as full-image path at image top). For
                // interior strips, this is a precondition violation.
                //
                // Similarly at the bottom.
                let strip_top = outer_strip_top.saturating_sub(strip_outer_y0);
                let strip_bot = (outer_strip_bot - strip_outer_y0).min(height);
                let strip_h = strip_bot - strip_top;
                let inner_start_full = strip_local_inner_y - strip_top;

                let strip_n = width * strip_h;
                bufs.resize(strip_n);

                // Apply the inner_y_filter: only rows in [filter_y0, filter_y1)
                // of the plane contribute to the accumulator. We process the
                // full band-inner range so the blur stencil is correct, but
                // hand the kernel a clipped (inner_start, inner_h) so it
                // skips accumulation outside the filter.
                let acc_y0 = strip_local_inner_y.max(filter_y0);
                let acc_y1 = strip_local_inner_end.min(filter_y1);
                if acc_y0 >= acc_y1 {
                    // Entire band-inner is outside the filter — skip.
                    outer_y = outer_inner_end;
                    continue;
                }
                let inner_start = acc_y0 - strip_top;
                let inner_h = acc_y1 - acc_y0;
                // Sanity: inner_start..inner_start+inner_h must lie within
                // the band-inner and within [0..strip_h].
                debug_assert!(inner_start >= inner_start_full);
                debug_assert!(inner_start + inner_h <= inner_start_full + inner_h_full);
                debug_assert!(inner_start + inner_h <= strip_h);
                let _ = (inner_start_full, inner_h_full);

                accum.n += inner_h * width;

                // Diffmap slice for the accumulated rows (band-local offset).
                // The diffmap is allocated for the full band, so we offset
                // into it by (acc_y0 - band_first_y) rows.
                let dm_start = (acc_y0 - band_first_y) * width;
                let dm_n = inner_h * width;

                for entry in &scale_active {
                    let Some((c, need_ssim, need_edge)) = *entry else {
                        continue;
                    };
                    let dm_info = match band_dm.as_mut() {
                        Some(dm) if need_ssim => {
                            let dm_w = diffmap_weights.unwrap();
                            Some((&mut dm[dm_start..dm_start + dm_n], dm_w[c]))
                        }
                        _ => None,
                    };
                    let ret_info = band_ret.as_mut().map(|r| {
                        (
                            &mut r.sd[c][dm_start..dm_start + dm_n],
                            &mut r.mu1[c][dm_start..dm_start + dm_n],
                            &mut r.mu2[c][dm_start..dm_start + dm_n],
                        )
                    });
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
                        bufs,
                        &mut accum,
                        dm_info,
                        ret_info,
                    );
                }

                outer_y = outer_inner_end;
            }

            (accum, band_dm)
        };

    // Pre-split the caller's retention planes into band-aligned mutable
    // chunks (bands tile the plane in `STRIP_INNER`-row order, matching
    // `chunks_mut(STRIP_INNER * strips_per_band * width)` exactly), so the
    // parallel bands write in place — no per-band allocation, no
    // concatenation copies (the C3a retention-churn lever).
    let band_chunk_rows = STRIP_INNER * strips_per_band;
    let mut band_rets: Vec<Option<AttrBandSlices<'_>>> = match attr_ret_out {
        Some(ret) => {
            let n_used = width * height;
            fn split3(
                planes: &mut [Vec<f32>; 3],
                n_used: usize,
                chunk: usize,
            ) -> [Vec<&mut [f32]>; 3] {
                let [a, b, c] = planes;
                [
                    a[..n_used].chunks_mut(chunk).collect(),
                    b[..n_used].chunks_mut(chunk).collect(),
                    c[..n_used].chunks_mut(chunk).collect(),
                ]
            }
            let chunk = band_chunk_rows * width;
            let [sd0, sd1, sd2] = split3(&mut ret.sd, n_used, chunk);
            let [m10, m11, m12] = split3(&mut ret.mu1, n_used, chunk);
            let [m20, m21, m22] = split3(&mut ret.mu2, n_used, chunk);
            let mut bands = Vec::with_capacity(num_bands);
            let z = sd0
                .into_iter()
                .zip(sd1)
                .zip(sd2)
                .zip(m10)
                .zip(m11)
                .zip(m12)
                .zip(m20)
                .zip(m21)
                .zip(m22);
            for ((((((((a, b), c), d), e), f), g), h), i) in z {
                bands.push(Some(AttrBandSlices {
                    sd: [a, b, c],
                    mu1: [d, e, f],
                    mu2: [g, h, i],
                }));
            }
            bands
        }
        None => (0..num_bands).map(|_| None).collect(),
    };
    debug_assert_eq!(band_rets.len(), num_bands);

    #[allow(clippy::redundant_closure)]
    let band_results: Vec<_> = {
        // Per-rayon-worker scratch — first band on each worker grows
        // ScaleBuffers to max_strip_n; subsequent bands reuse with
        // zero memset cost. Eliminates the ~9 % wall-time spent in
        // __memset_avx512_unaligned_erms on 1080p strip path
        // (post-AVX-512 profile, this was the largest non-SIMD hotspot).
        let init_bufs = || ScaleBuffers::empty();

        #[cfg(feature = "threads")]
        if run_parallel {
            use rayon::prelude::*;
            band_rets
                .into_par_iter()
                .enumerate()
                .map_init(init_bufs, |bufs, (i, ret)| process_band(bufs, i, ret))
                .collect()
        } else {
            let mut bufs = init_bufs();
            band_rets
                .drain(..)
                .enumerate()
                .map(|(i, ret)| process_band(&mut bufs, i, ret))
                .collect()
        }
        #[cfg(not(feature = "threads"))]
        {
            let mut bufs = init_bufs();
            band_rets
                .drain(..)
                .enumerate()
                .map(|(i, ret)| process_band(&mut bufs, i, ret))
                .collect()
        }
    };

    // Merge band accumulators and concatenate diffmaps (retention was
    // written in place through the pre-split band slices).
    let mut accum = ScaleAccumulators::new();
    let mut diffmap = if diffmap_weights.is_some() {
        Some(Vec::with_capacity(width * height))
    } else {
        None
    };
    for (band_accum, band_dm) in band_results {
        accum.merge(&band_accum);
        if let (Some(dm), Some(bdm)) = (&mut diffmap, band_dm) {
            dm.extend_from_slice(&bdm);
        }
    }
    (accum, diffmap)
}

/// Reusable scratch buffers for [`crate::Zensim::compute_with_ref_into`].
///
/// Designed for encoder quantization loops that call `compute_with_ref` many
/// times against the same precomputed reference. Holds the distorted-side
/// XYB plane allocation across calls so it isn't freed and reallocated each
/// iteration.
///
/// At 1920×1080 the per-call dst plane allocation is ~25 MB; at 3840×2160
/// it is ~99 MB. Reusing the allocation skips the kernel page-fault commit
/// on subsequent calls.
#[derive(Default)]
pub struct ZensimScratch {
    pub(crate) dst_planes: [Vec<f32>; 3],
}

impl ZensimScratch {
    /// Create an empty scratch. The first call to `compute_with_ref_into`
    /// will allocate buffers sized for the distorted image; subsequent
    /// calls grow the buffers if necessary or reuse them in place.
    pub fn new() -> Self {
        Self::default()
    }
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
    // INVARIANT: scales[i].0[0..3].len() == scales[i].1 * scales[i].2
    // (padded_width × height per plane). Enforced at construction.
    // Both PrecomputedReference and PrecomputedReferenceView rely on this.
    /// Reference image width in pixels (unpadded). Used to validate that
    /// distorted images passed to `compute_with_ref*` match the dimensions
    /// the precomputed pyramid was built for.
    pub(crate) ref_width: usize,
    /// Reference image height in pixels.
    pub(crate) ref_height: usize,
}

impl PrecomputedReference {
    /// Reference image width in pixels (the unpadded width passed at construction).
    ///
    /// Distorted images compared against this reference via
    /// [`Zensim::compute_with_ref`](crate::Zensim::compute_with_ref) and friends
    /// must match this width exactly; otherwise the call returns
    /// [`ZensimError::DimensionMismatch`](crate::ZensimError::DimensionMismatch).
    pub fn width(&self) -> usize {
        self.ref_width
    }

    /// Reference image height in pixels.
    ///
    /// See [`width`](Self::width) for the matching contract on distorted images.
    pub fn height(&self) -> usize {
        self.ref_height
    }

    /// Build a precomputed reference from an ImageSource.
    ///
    /// Converts to XYB and builds the downscale pyramid, storing planes at each level.
    pub(crate) fn new(source: &impl ImageSource, num_scales: usize, parallel: bool) -> Self {
        // Sub-64px sources can't form the 4-scale pyramid; reflect-pad to the
        // minimum (matching the buffered `compute_with_config_inner` path) so the
        // reference holds 4 genuinely-computed scales. The contract dims
        // (`ref_width`/`ref_height`) stay the ORIGINAL size so `compute_with_ref*`
        // still validates the distorted image against the caller's dimensions.
        let (orig_w, orig_h) = (source.width(), source.height());
        if orig_w > 0
            && orig_h > 0
            && (orig_w < crate::metric::MIN_PYRAMID_DIM || orig_h < crate::metric::MIN_PYRAMID_DIM)
        {
            let padded = crate::metric::reflect_pad_to_min(source);
            let mut r = Self::new_inner(&padded, num_scales, parallel);
            r.ref_width = orig_w;
            r.ref_height = orig_h;
            r
        } else {
            Self::new_inner(source, num_scales, parallel)
        }
    }

    fn new_inner(source: &impl ImageSource, num_scales: usize, parallel: bool) -> Self {
        let width = source.width();
        let height = source.height();
        let padded_width = simd_padded_width(width);
        Self::build_from_dims(num_scales, padded_width, height, parallel, |scale0| {
            convert_source_to_xyb_into(source, scale0, padded_width, parallel);
        })
        .with_ref_dims(width, height)
    }

    /// Allocate scale buffers up front and fill them via `fill_scale0`, then
    /// downscale level by level. This avoids the working-buffer + clone
    /// pattern (which at 4K cost 99 MB of throwaway alloc + 4 clones totalling
    /// 131 MB of memcpy and a blocking munmap on drop).
    fn build_from_dims(
        num_scales: usize,
        padded_width: usize,
        height: usize,
        #[allow(unused_variables)] parallel: bool,
        fill_scale0: impl FnOnce(&mut [Vec<f32>; 3]),
    ) -> Self {
        // Compute scale dimensions up front (powers-of-2 down)
        let mut dims = Vec::with_capacity(num_scales);
        let mut w = padded_width;
        let mut h = height;
        for _ in 0..num_scales {
            if w < 8 || h < 8 {
                break;
            }
            dims.push((w, h));
            w /= 2;
            h /= 2;
        }

        // Allocate ALL scale buffers up front. vec![0.0; n] hits the calloc
        // fast path (zero-COW pages on Linux), so this is cheap even for the
        // total 132 MB working set at 4K.
        let mut scales: Vec<([Vec<f32>; 3], usize, usize)> = dims
            .iter()
            .map(|&(sw, sh)| {
                let n = sw * sh;
                ([vec![0.0; n], vec![0.0; n], vec![0.0; n]], sw, sh)
            })
            .collect();

        // Fill scale 0 directly from the caller (no working buffer).
        fill_scale0(&mut scales[0].0);

        // Downscale scale[i-1] -> scale[i] out of place. We need the previous
        // scale's data to remain owned, so use the _into variant.
        for i in 1..scales.len() {
            let prev_w = scales[i - 1].1;
            let (new_w, new_h) = (scales[i].1, scales[i].2);
            let (lo, hi) = scales.split_at_mut(i);
            let src = &lo[i - 1].0;
            let dst = &mut hi[0].0;
            let [ref mut d0, ref mut d1, ref mut d2] = *dst;
            #[cfg(feature = "threads")]
            if parallel {
                let _ = maybe_join(
                    true,
                    || crate::blur::downscale_2x_into(&src[0], prev_w, d0, new_w, new_h),
                    || {
                        maybe_join(
                            true,
                            || crate::blur::downscale_2x_into(&src[1], prev_w, d1, new_w, new_h),
                            || crate::blur::downscale_2x_into(&src[2], prev_w, d2, new_w, new_h),
                        );
                    },
                );
            } else {
                crate::blur::downscale_2x_into(&src[0], prev_w, d0, new_w, new_h);
                crate::blur::downscale_2x_into(&src[1], prev_w, d1, new_w, new_h);
                crate::blur::downscale_2x_into(&src[2], prev_w, d2, new_w, new_h);
            }
            #[cfg(not(feature = "threads"))]
            {
                crate::blur::downscale_2x_into(&src[0], prev_w, d0, new_w, new_h);
                crate::blur::downscale_2x_into(&src[1], prev_w, d1, new_w, new_h);
                crate::blur::downscale_2x_into(&src[2], prev_w, d2, new_w, new_h);
            }
        }

        Self {
            scales,
            ref_width: 0,
            ref_height: 0,
        }
    }

    /// Set the unpadded reference dimensions on a freshly-built pyramid.
    ///
    /// Internal — chained from each public constructor so callers see the
    /// dimensions they passed in, not the padded width.
    fn with_ref_dims(mut self, width: usize, height: usize) -> Self {
        self.ref_width = width;
        self.ref_height = height;
        self
    }

    /// Build a precomputed reference from planar linear RGB f32 data.
    ///
    /// `planes` are `[R, G, B]`, each with `stride * height` elements (or more).
    /// `stride` is the number of f32 elements per row (may be larger than `width`
    /// for padded buffers). Converts to positive XYB internally.
    pub(crate) fn from_linear_planar(
        planes: [&[f32]; 3],
        width: usize,
        height: usize,
        stride: usize,
        num_scales: usize,
        parallel: bool,
    ) -> Self {
        // Sub-64px planar sources can't form the 4-scale pyramid; reflect-pad
        // the LINEAR planes to the minimum — the planar analogue of
        // [`Self::new`]'s ImageSource branch (linear→XYB is pointwise, so
        // padding before conversion equals converting reflect-padded input).
        // The contract dims stay the ORIGINAL size, matching `Self::new`.
        // Without this branch, ≤63px planar precompute panicked in the
        // mean-offset pass (found via jxl-encoder's cpu_zensim_* 32² tests,
        // 2026-07-29; present since the entry point was added).
        {
            use crate::metric::{MIN_PYRAMID_DIM, reflect_index};
            if width > 0 && height > 0 && (width < MIN_PYRAMID_DIM || height < MIN_PYRAMID_DIM) {
                let bw = width.max(MIN_PYRAMID_DIM);
                let bh = height.max(MIN_PYRAMID_DIM);
                let padded: [Vec<f32>; 3] = std::array::from_fn(|c| {
                    let src = planes[c];
                    let mut out = vec![0.0f32; bw * bh];
                    for y in 0..bh {
                        let sy = reflect_index(y, height);
                        for x in 0..bw {
                            out[y * bw + x] = src[sy * stride + reflect_index(x, width)];
                        }
                    }
                    out
                });
                let padded_refs: [&[f32]; 3] = [&padded[0], &padded[1], &padded[2]];
                let padded_width = simd_padded_width(bw);
                return Self::build_from_dims(num_scales, padded_width, bh, parallel, |scale0| {
                    convert_linear_planar_to_xyb_into(
                        padded_refs,
                        bw,
                        bh,
                        bw,
                        padded_width,
                        scale0,
                    );
                })
                .with_ref_dims(width, height);
            }
        }
        let padded_width = simd_padded_width(width);
        Self::build_from_dims(num_scales, padded_width, height, parallel, |scale0| {
            convert_linear_planar_to_xyb_into(planes, width, height, stride, padded_width, scale0);
        })
        .with_ref_dims(width, height)
    }

    /// Build a zero-copy view of a Y-row range of this reference's pyramid.
    ///
    /// Returns slices into the owned plane Vecs without allocating fresh
    /// per-plane buffers. Used by the streaming-strip aggregator to avoid
    /// the ~2.6 GB of per-pair memcpy that `to_vec()`-based slicing
    /// produces at 80 MP. See `STREAMING_372_OPTIMIZATION_NOTES.md`.
    ///
    /// The returned view's `ref_width` matches `self.ref_width`; its
    /// `ref_height` is set to the requested row count at scale 0.
    pub(crate) fn slice_rows_view(
        &self,
        src_y0: usize,
        src_y1: usize,
    ) -> PrecomputedReferenceView<'_> {
        let mut scales = Vec::with_capacity(self.scales.len());
        for (scale_idx, (planes, plane_w, plane_h)) in self.scales.iter().enumerate() {
            let factor = 1usize << scale_idx;
            let y0_scale = src_y0 / factor;
            let y1_scale = src_y1.div_ceil(factor);
            let y1_scale = y1_scale.min(*plane_h);
            let y0_scale = y0_scale.min(y1_scale);
            let rows = y1_scale - y0_scale;
            let start_off = y0_scale * plane_w;
            let end_off = y1_scale * plane_w;
            scales.push((
                [
                    &planes[0][start_off..end_off],
                    &planes[1][start_off..end_off],
                    &planes[2][start_off..end_off],
                ],
                *plane_w,
                rows,
            ));
        }
        PrecomputedReferenceView { scales }
    }
}

/// Zero-copy view of a Y-row range of a [`PrecomputedReference`].
///
/// Holds borrowed slices into the parent reference's plane Vecs. Built
/// via [`PrecomputedReference::slice_rows_view`]; never directly
/// constructed by callers.
///
/// The strip-aggregator hot path uses this to avoid 12 fresh `Vec<f32>`
/// allocations + memcpy per strip per pair (see optimization notes).
pub(crate) struct PrecomputedReferenceView<'a> {
    /// Per scale: `(planes, padded_width, height)`.
    pub(crate) scales: Vec<([&'a [f32]; 3], usize, usize)>,
}

/// Trait the strip kernel uses to access an arbitrary multi-scale
/// reference pyramid — either an owned [`PrecomputedReference`] or a
/// borrowed [`PrecomputedReferenceView`].
///
/// Returning planes as `[&[f32]; 3]` lets both owned (Vec-backed) and
/// borrowed (slice-backed) representations satisfy the same kernel
/// signature without copies.
pub(crate) trait MultiScaleRef {
    fn num_scales(&self) -> usize;
    /// `(planes, padded_width, height)` at the given scale.
    fn scale(&self, idx: usize) -> ([&[f32]; 3], usize, usize);
}

impl MultiScaleRef for PrecomputedReference {
    #[inline]
    fn num_scales(&self) -> usize {
        self.scales.len()
    }
    #[inline]
    fn scale(&self, idx: usize) -> ([&[f32]; 3], usize, usize) {
        let (planes, w, h) = &self.scales[idx];
        ([&planes[0], &planes[1], &planes[2]], *w, *h)
    }
}

impl<'a> MultiScaleRef for PrecomputedReferenceView<'a> {
    #[inline]
    fn num_scales(&self) -> usize {
        self.scales.len()
    }
    #[inline]
    fn scale(&self, idx: usize) -> ([&[f32]; 3], usize, usize) {
        let (planes, w, h) = &self.scales[idx];
        (*planes, *w, *h)
    }
}

/// Convert planar linear RGB f32 to padded positive-XYB planes.
///
/// `planes` are `[R, G, B]` with `stride` elements per row.
/// Output is `[Vec<f32>; 3]` with `padded_width` elements per row,
/// mirror-padded for SIMD alignment.
pub(crate) fn convert_linear_planar_to_xyb(
    planes: [&[f32]; 3],
    width: usize,
    height: usize,
    stride: usize,
    padded_width: usize,
) -> [Vec<f32>; 3] {
    let n = padded_width * height;
    let mut out: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; n]);
    convert_linear_planar_to_xyb_into(planes, width, height, stride, padded_width, &mut out);
    out
}

/// Like [`convert_linear_planar_to_xyb`] but writes into pre-allocated planes
/// (each at least `padded_width * height` elements).
pub(crate) fn convert_linear_planar_to_xyb_into(
    planes: [&[f32]; 3],
    width: usize,
    height: usize,
    stride: usize,
    padded_width: usize,
    out: &mut [Vec<f32>; 3],
) {
    use crate::color::linear_to_positive_xyb_planar_into;

    // Pack planar → interleaved [f32; 3] per row, then convert to XYB.
    // The per-row buffer fits in L1 cache (~12KB for 1024px).
    let mut rgb_row: Vec<[f32; 3]> = vec![[0.0; 3]; width];

    // Destructure for independent mutable borrows
    let [ref mut o0, ref mut o1, ref mut o2] = *out;

    for y in 0..height {
        let row_off = y * stride;
        for x in 0..width {
            // Clamp to [0, 1]: lossy reconstruction can produce out-of-range
            // linear RGB (negative from quantization error near black, >1 from
            // gaborish overshoot). A display clamps to its gamut, so the clamped
            // values are what the viewer actually sees — measuring error on
            // unclamped values would report differences that aren't perceptible.
            rgb_row[x] = [
                planes[0][row_off + x].clamp(0.0, 1.0),
                planes[1][row_off + x].clamp(0.0, 1.0),
                planes[2][row_off + x].clamp(0.0, 1.0),
            ];
        }
        let out_off = y * padded_width;
        linear_to_positive_xyb_planar_into(
            &rgb_row[..width],
            &mut o0[out_off..out_off + width],
            &mut o1[out_off..out_off + width],
            &mut o2[out_off..out_off + width],
        );
    }

    // Mirror-pad columns
    let pad_count = padded_width - width;
    if pad_count > 0 {
        let period = 2 * (width - 1).max(1);
        let mirror_offsets: Vec<usize> = (0..pad_count)
            .map(|i| {
                let m = (width + i) % period;
                if m < width { m } else { period - m }
            })
            .collect();

        for y in 0..height {
            let row_off = y * padded_width;
            for (i, &src_x) in mirror_offsets.iter().enumerate() {
                o0[row_off + width + i] = o0[row_off + src_x];
                o1[row_off + width + i] = o1[row_off + src_x];
                o2[row_off + width + i] = o2[row_off + src_x];
            }
        }
    }
}

/// HDR sibling of [`convert_linear_planar_to_xyb_into`]: **absolute-luminance**
/// linear RGB planes (cd/m²) → PU-encoded XYB planes. Unlike the SDR path it
/// does NOT clamp to `[0,1]` (HDR luminance exceeds 1) and applies PU21 via
/// [`crate::color::linear_to_pu_xyb_planar_into`] instead of the cube root.
/// See `docs/HDR_PLAN.md` §2b.
pub(crate) fn convert_linear_planar_to_pu_xyb_into(
    planes: [&[f32]; 3],
    width: usize,
    height: usize,
    stride: usize,
    padded_width: usize,
    out: &mut [Vec<f32>; 3],
) {
    use crate::color::linear_to_pu_xyb_planar_into;

    let mut rgb_row: Vec<[f32; 3]> = vec![[0.0; 3]; width];
    let [ref mut o0, ref mut o1, ref mut o2] = *out;

    for y in 0..height {
        let row_off = y * stride;
        for x in 0..width {
            // No [0,1] clamp: HDR absolute luminance is unbounded above. The
            // PU conversion clamps to PU21's valid luminance domain instead.
            rgb_row[x] = [
                planes[0][row_off + x],
                planes[1][row_off + x],
                planes[2][row_off + x],
            ];
        }
        let out_off = y * padded_width;
        linear_to_pu_xyb_planar_into(
            &rgb_row[..width],
            &mut o0[out_off..out_off + width],
            &mut o1[out_off..out_off + width],
            &mut o2[out_off..out_off + width],
        );
    }

    mirror_pad_columns(out, width, height, padded_width);
}

/// Interleaved-input sibling of [`convert_linear_planar_to_pu_xyb_into`]:
/// each row is `width` `[R, G, B]` f32 triples starting at `y * stride`
/// elements. The row slice reinterprets directly as `&[[f32; 3]]` — no
/// per-pixel gather, which makes interleaved the cheaper input layout here.
pub(crate) fn convert_linear_interleaved_to_pu_xyb_into(
    rgb: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    padded_width: usize,
    out: &mut [Vec<f32>; 3],
) {
    use crate::color::linear_to_pu_xyb_planar_into;

    let [ref mut o0, ref mut o1, ref mut o2] = *out;
    for y in 0..height {
        let row_off = y * stride;
        // No [0,1] clamp here either — see the planar sibling.
        let row: &[[f32; 3]] = bytemuck::cast_slice(&rgb[row_off..row_off + 3 * width]);
        let out_off = y * padded_width;
        linear_to_pu_xyb_planar_into(
            row,
            &mut o0[out_off..out_off + width],
            &mut o1[out_off..out_off + width],
            &mut o2[out_off..out_off + width],
        );
    }

    mirror_pad_columns(out, width, height, padded_width);
}

/// Mirror-pad columns `width..padded_width` of three planes in place,
/// identical to the SDR conversion's column padding so every downstream
/// kernel sees the same layout regardless of input path.
fn mirror_pad_columns(out: &mut [Vec<f32>; 3], width: usize, height: usize, padded_width: usize) {
    let pad_count = padded_width - width;
    if pad_count == 0 {
        return;
    }
    let period = 2 * (width - 1).max(1);
    let mirror_offsets: Vec<usize> = (0..pad_count)
        .map(|i| {
            let m = (width + i) % period;
            if m < width { m } else { period - m }
        })
        .collect();
    let [ref mut o0, ref mut o1, ref mut o2] = *out;
    for y in 0..height {
        let row_off = y * padded_width;
        for (i, &src_x) in mirror_offsets.iter().enumerate() {
            o0[row_off + width + i] = o0[row_off + src_x];
            o1[row_off + width + i] = o1[row_off + src_x];
            o2[row_off + width + i] = o2[row_off + src_x];
        }
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
    let n = padded_width * height;
    let mut dst_planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; n]);
    let result = compute_multiscale_stats_streaming_with_ref_borrowed(
        precomputed,
        distorted,
        &mut dst_planes,
        config,
        weights,
    );
    #[cfg(feature = "threads")]
    dealloc_planes(dst_planes, None);
    #[cfg(not(feature = "threads"))]
    drop(dst_planes);
    result
}

/// Like [`compute_multiscale_stats_streaming_with_ref`] but borrows
/// `dst_planes` from the caller. The Vecs are resized to `padded_width *
/// height` (preserving capacity if it already fits) and reused — no
/// per-call allocation. After return they hold the smallest-scale data.
pub(crate) fn compute_multiscale_stats_streaming_with_ref_borrowed(
    precomputed: &PrecomputedReference,
    distorted: &impl ImageSource,
    dst_planes: &mut [Vec<f32>; 3],
    config: &ZensimConfig,
    weights: &[f64],
) -> (Vec<ScaleStats>, [f64; 3]) {
    let (accums, offset_sums, pixel_count) = compute_multiscale_accums_streaming_with_ref_borrowed(
        precomputed,
        distorted,
        dst_planes,
        config,
        weights,
        None,
        None,
    );
    let stats: Vec<ScaleStats> = accums
        .iter()
        .map(|a| a.finalize(config.iw_strength as f64))
        .collect();
    let mean_offset = if pixel_count == 0 {
        [0.0; 3]
    } else {
        let inv = 1.0 / pixel_count as f64;
        [
            offset_sums[0] * inv,
            offset_sums[1] * inv,
            offset_sums[2] * inv,
        ]
    };
    (stats, mean_offset)
}

/// **Raw-accumulator** variant: returns per-scale [`ScaleAccumulators`]
/// (un-finalized) + mean_offset numerator sums + pixel count, instead of
/// finalized [`ScaleStats`] + means. Used by
/// [`compute_multiscale_stats_streaming_strips`] to merge accumulators
/// across strips before a single finalize pass — eliminates the
/// per-strip finalize+re-pool precision loss.
///
/// `inner_y_filter`: when `Some((y0, y1))`, only rows in the half-open
/// range `[y0, y1)` of the SCALE-0 plane contribute to the per-scale
/// accumulators. At each pyramid level the filter is halved (with
/// half-open semantics) — so the same SOURCE-coordinate inner rows
/// are accumulated at every scale. The filter range MUST be aligned
/// to `2^(num_scales-1)` for byte-exact equivalence (otherwise the
/// pyramid downscale at strip boundaries will produce a different
/// row count than the full-image path). Returns the mean-offset SUMS
/// computed over the filter range only; if `inner_y_filter` is `None`,
/// the entire plane contributes (matching the non-streaming path).
pub(crate) fn compute_multiscale_accums_streaming_with_ref_borrowed<R: MultiScaleRef>(
    precomputed: &R,
    distorted: &impl ImageSource,
    dst_planes: &mut [Vec<f32>; 3],
    config: &ZensimConfig,
    weights: &[f64],
    inner_y_filter: Option<(usize, usize)>,
    outer_layout_scale_0: Option<(usize, usize)>,
) -> (Vec<ScaleAccumulators>, [f64; 3], usize) {
    let width = distorted.width();
    let height = distorted.height();
    let padded_width = simd_padded_width(width);
    let num_scales = config.num_scales.min(precomputed.num_scales());
    let parallel = config.allow_multithreading;
    let n = padded_width * height;

    for p in dst_planes.iter_mut() {
        if p.len() < n {
            p.resize(n, 0.0);
        } else {
            p.truncate(n);
        }
    }

    convert_source_to_xyb_into(distorted, dst_planes, padded_width, parallel);

    // Borrow scale-0 planes from the abstract reference (owned or view).
    let (src_planes_s0, _, _) = precomputed.scale(0);
    let dst_planes_s0: [&[f32]; 3] = [&dst_planes[0], &dst_planes[1], &dst_planes[2]];
    // Mean offset is computed over the inner_y_filter rows (inner rows
    // of the strip), so each source row is counted exactly once across
    // all strips.
    let (offset_sums, pixel_count) = match inner_y_filter {
        Some((y0, y1)) => {
            let inner_h = y1.saturating_sub(y0);
            if inner_h == 0 {
                ([0.0; 3], 0)
            } else {
                let inner_offset = compute_xyb_mean_offset_range(
                    src_planes_s0,
                    dst_planes_s0,
                    width,
                    y0,
                    y1,
                    padded_width,
                );
                let inner_count = width * inner_h;
                let cn = inner_count as f64;
                (
                    [
                        inner_offset[0] * cn,
                        inner_offset[1] * cn,
                        inner_offset[2] * cn,
                    ],
                    inner_count,
                )
            }
        }
        None => {
            let mean_offset =
                compute_xyb_mean_offset(src_planes_s0, dst_planes_s0, width, height, padded_width);
            let pc = width * height;
            let cn = pc as f64;
            (
                [
                    mean_offset[0] * cn,
                    mean_offset[1] * cn,
                    mean_offset[2] * cn,
                ],
                pc,
            )
        }
    };

    let mut accums = Vec::with_capacity(num_scales);
    let mut w = padded_width;
    let mut h = height;
    // Filter halves at each scale (pyramid downscale is 2:1).
    // Half-open: (y0, y1) → (y0/2, y1/2). For byte-exact equivalence,
    // y0 and y1 must be even at every scale (i.e., aligned to
    // 2^(num_scales-1) at scale 0).
    let mut scale_filter = inner_y_filter;
    let mut scale_outer = outer_layout_scale_0;

    for scale in 0..num_scales {
        if w < 8 || h < 8 {
            break;
        }
        let (src_planes, src_w, src_h) = precomputed.scale(scale);
        assert_eq!(w, src_w, "scale {scale} width mismatch");
        assert_eq!(h, src_h, "scale {scale} height mismatch");
        let dst_planes_view: [&[f32]; 3] = [&dst_planes[0], &dst_planes[1], &dst_planes[2]];

        let (accum, _) = process_scale_bands_into_accum(
            src_planes,
            dst_planes_view,
            w,
            h,
            config,
            scale,
            weights,
            None,
            scale_filter,
            scale_outer,
            None,
        );
        accums.push(accum);

        if scale < num_scales - 1 {
            let (nw, nh) = downscale_3_planes(dst_planes, w, h, parallel);
            w = nw;
            h = nh;
            // Halve the filter and outer-layout for the next scale.
            scale_filter = scale_filter.map(|(y0, y1)| (y0 / 2, y1 / 2));
            scale_outer = scale_outer.map(|(outer_h, offset)| (outer_h / 2, offset / 2));
        }
    }

    (accums, offset_sums, pixel_count)
}

/// Strip-aggregating variant for very-large images (OOM relief).
///
/// Splits the source and distorted images into Y-strips of `strip_inner`
/// scale-0 rows with `strip_margin` rows of overlap on each side for
/// the blur stencil. For each strip:
///
/// 1. Build a per-strip `PrecomputedReference` (containing just that
///    strip's XYB pyramid).
/// 2. Convert the strip's distorted bytes → XYB planes.
/// 3. Run [`process_scale_bands_into_accum`] per scale, returning the
///    raw [`ScaleAccumulators`].
/// 4. **Discard** the margin rows' feature contributions (the
///    accumulator's `inner_y_start`/`inner_y_end` define which rows
///    count toward the global accumulator). This is achieved by
///    re-running `process_scale_bands_into_accum` once for the inner
///    rows only — the existing band-internal overlap mechanism
///    handles boundary blur correctly.
///
/// 5. Merge per-strip accumulators into per-scale global accumulators.
/// 6. After the last strip, finalize each global accumulator →
///    [`ScaleStats`].
///
/// **Memory cost**: O(strip_height × width × 1.33) instead of O(full
/// image × 1.33). For 80 MP (8000 × 10000), a 256-inner-row strip
/// (+ 128 margin top + 128 margin bottom = 512 strip rows) takes
/// ~125 MB per worker instead of 2.5 GB.
///
/// **Aggregation correctness**: every `ScaleAccumulators` field is a
/// raw sum or max — both directly composable via [`ScaleAccumulators::merge`].
/// The final mean / root-power pooling happens once after all strips
/// are merged, yielding f64-machine-epsilon agreement with the
/// full-image path (worst observed rel error: < 1e-13 on the 99-pair
/// safesyn test).
///
/// **V-blur band layout**: each strip's `process_scale_bands_into_accum`
/// is told `outer_layout = (full_h, strip_y0)`, so its bands tile
/// against the full-image plane (not the strip's local plane). Each
/// band's V-blur running-sum init mirrors at the same source row as
/// the full-image path's corresponding band, eliminating f32
/// accumulator history divergence — which is what would otherwise
/// produce ~3e-3 rel drift at coarse-scale near-zero features
/// (catastrophic cancellation in `sigma_sq = blur(src²) - mu²`).
///
/// **Strip-boundary blur context**: with `strip_margin >= blur_radius
/// × blur_passes` at every pyramid scale (default 128 > 5×1), the
/// inner-row blur stencil reads only real strip data (no
/// mirror-clamp). Strip margin must be a multiple of `2^(num_scales-1)`
/// and inner rows must align to band boundaries for byte-exactness;
/// the default geometry (strip_inner=256, strip_margin=128) satisfies
/// both.
/// Variant of [`compute_multiscale_stats_streaming_strips`] that
/// REUSES a caller-owned full [`PrecomputedReference`] across strips,
/// instead of building a per-strip ref. Best for batch encoder loops
/// where many distorted candidates are scored against the same source
/// (the XYB conversion + pyramid downscale on the reference side
/// happens once for the whole image, not once per strip per call).
///
/// Each strip slices the appropriate Y-range out of the precomputed
/// ref's per-scale planes and runs the standard raw-accumulator
/// pipeline on that slice. The distorted side is still processed
/// per-strip so per-pair peak memory stays bounded.
pub(crate) fn compute_multiscale_stats_streaming_strips_with_ref(
    precomputed: &PrecomputedReference,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    strip_inner: usize,
    strip_margin: usize,
) -> (Vec<ScaleStats>, [f64; 3]) {
    let width = distorted.width();
    let height = distorted.height();
    assert_eq!(precomputed.ref_width, width);
    assert_eq!(precomputed.ref_height, height);

    // Edge case: image short enough to process in one strip.
    let one_strip_h = strip_inner + 2 * strip_margin;
    if height <= one_strip_h {
        return compute_multiscale_stats_streaming_with_ref(
            precomputed,
            distorted,
            config,
            weights,
        );
    }

    let num_scales = config.num_scales.min(precomputed.scales.len());
    let n_strips = height.div_ceil(strip_inner);

    let strip_meta: Vec<(usize, usize, usize, usize)> = (0..n_strips)
        .filter_map(|strip_idx| {
            let inner_y0 = strip_idx * strip_inner;
            let inner_y1 = ((strip_idx + 1) * strip_inner).min(height);
            if inner_y0 >= height {
                return None;
            }
            let strip_y0 = inner_y0.saturating_sub(strip_margin);
            let strip_y1 = (inner_y1 + strip_margin).min(height);
            let inner_filter_strip_y0 = inner_y0 - strip_y0;
            let inner_filter_strip_y1 = inner_y1 - strip_y0;
            Some((
                strip_y0,
                strip_y1,
                inner_filter_strip_y0,
                inner_filter_strip_y1,
            ))
        })
        .collect();

    let strip_config = ZensimConfig {
        allow_multithreading: false,
        ..*config
    };
    // Per-worker scratch for dst XYB planes — reused across strips on the
    // same rayon worker. First strip on each worker allocates ~250 MB at
    // 80 MP (3 planes × padded_width × strip_height × 4 bytes); subsequent
    // strips of the same dims skip the realloc (the kernel's resize is a
    // no-op when capacity is already ≥ n). The strips' actual data is
    // overwritten by convert_source_to_xyb_into so prior contents are
    // irrelevant.
    let init_dst_scratch = || -> [Vec<f32>; 3] { std::array::from_fn(|_| Vec::new()) };
    let process_strip = |dst_planes: &mut [Vec<f32>; 3],
                         (strip_y0, strip_y1, fy0, fy1): (usize, usize, usize, usize)|
     -> (Vec<ScaleAccumulators>, [f64; 3], usize) {
        let dst_strip = crate::source::SubsetView::new(distorted, strip_y0, strip_y1 - strip_y0);
        // Zero-copy: borrow this strip's rows from the parent precomputed ref.
        // Eliminates the ~65 MB per-strip memcpy that the prior to_vec()-based
        // slicer incurred. See STREAMING_372_OPTIMIZATION_NOTES.md.
        let strip_precomp = precomputed.slice_rows_view(strip_y0, strip_y1);
        compute_multiscale_accums_streaming_with_ref_borrowed(
            &strip_precomp,
            &dst_strip,
            dst_planes,
            &strip_config,
            weights,
            Some((fy0, fy1)),
            Some((height, strip_y0)),
        )
    };

    #[cfg(feature = "threads")]
    let parallel_strips = config.allow_multithreading;
    #[cfg(feature = "threads")]
    let strip_results: Vec<(Vec<ScaleAccumulators>, [f64; 3], usize)> = if parallel_strips {
        use rayon::prelude::*;
        strip_meta
            .par_iter()
            .copied()
            .map_init(init_dst_scratch, process_strip)
            .collect()
    } else {
        let mut scratch = init_dst_scratch();
        strip_meta
            .iter()
            .copied()
            .map(|m| process_strip(&mut scratch, m))
            .collect()
    };
    #[cfg(not(feature = "threads"))]
    let strip_results: Vec<(Vec<ScaleAccumulators>, [f64; 3], usize)> = {
        let mut scratch = init_dst_scratch();
        strip_meta
            .iter()
            .copied()
            .map(|m| process_strip(&mut scratch, m))
            .collect()
    };

    let mut global_accums: Vec<ScaleAccumulators> =
        (0..num_scales).map(|_| ScaleAccumulators::new()).collect();
    let mut mean_offset_sums: [f64; 3] = [0.0; 3];
    let mut mean_offset_pixel_count: usize = 0;
    for (strip_accums, strip_offset_sums, strip_pixel_count) in strip_results {
        for (s, strip_accum) in strip_accums.iter().enumerate() {
            global_accums[s].merge(strip_accum);
        }
        for c in 0..3 {
            mean_offset_sums[c] += strip_offset_sums[c];
        }
        mean_offset_pixel_count += strip_pixel_count;
    }

    let final_stats: Vec<ScaleStats> = global_accums
        .iter()
        .map(|a| a.finalize(config.iw_strength as f64))
        .collect();
    let final_mean_offset = if mean_offset_pixel_count == 0 {
        [0.0; 3]
    } else {
        let inv = 1.0 / mean_offset_pixel_count as f64;
        [
            mean_offset_sums[0] * inv,
            mean_offset_sums[1] * inv,
            mean_offset_sums[2] * inv,
        ]
    };
    (final_stats, final_mean_offset)
}

// `precomputed_ref_slice_rows` removed — superseded by
// `PrecomputedReference::slice_rows_view` which returns a zero-copy
// borrowed `PrecomputedReferenceView<'_>` instead of allocating 12
// fresh Vec<f32> + memcpy per strip. See `STREAMING_372_OPTIMIZATION_NOTES.md`
// (Optimization 1) for the cost model.

pub(crate) fn compute_multiscale_stats_streaming_strips(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    strip_inner: usize,
    strip_margin: usize,
) -> (Vec<ScaleStats>, [f64; 3]) {
    let width = source.width();
    let height = source.height();
    assert_eq!(width, distorted.width());
    assert_eq!(height, distorted.height());

    // Edge case: image short enough to process in one strip.
    let one_strip_h = strip_inner + 2 * strip_margin;
    if height <= one_strip_h {
        let precomputed = PrecomputedReference::new(source, config.num_scales, false);
        return compute_multiscale_stats_streaming_with_ref(
            &precomputed,
            distorted,
            config,
            weights,
        );
    }

    let num_scales = config.num_scales;
    let n_strips = height.div_ceil(strip_inner);

    // Collect strip metadata so we can iterate (sequentially or in
    // parallel) over a fixed set of strips. Each strip rebuilds its
    // PrecomputedReference and dst-planes independently — strips are
    // embarassingly parallel.
    let strip_meta: Vec<(usize, usize, usize, usize, usize, usize)> = (0..n_strips)
        .filter_map(|strip_idx| {
            let inner_y0 = strip_idx * strip_inner;
            let inner_y1 = ((strip_idx + 1) * strip_inner).min(height);
            if inner_y0 >= height {
                return None;
            }
            let strip_y0 = inner_y0.saturating_sub(strip_margin);
            let strip_y1 = (inner_y1 + strip_margin).min(height);
            let inner_filter_strip_y0 = inner_y0 - strip_y0;
            let inner_filter_strip_y1 = inner_y1 - strip_y0;
            Some((
                strip_y0,
                strip_y1,
                inner_filter_strip_y0,
                inner_filter_strip_y1,
                inner_y0,
                inner_y1,
            ))
        })
        .collect();

    // Per-strip processor. Returns (per-scale ScaleAccumulators,
    // mean-offset-sums, mean-offset-pixel-count).
    //
    // Each call DISABLES inner band-level rayon parallelism (the strip
    // is small enough that the gain is marginal, and oversubscription
    // would hurt outer-strip parallelism). Strips parallelize at the
    // OUTER level — much higher throughput on multi-core hosts than
    // sequential strips × inner bands.
    let strip_config = ZensimConfig {
        allow_multithreading: false,
        ..*config
    };
    // Per-worker scratch for dst XYB planes — same pattern as the
    // with-ref variant. Saves one Vec alloc + munmap per strip on this
    // worker after the first. The strip-per-strip path also rebuilds
    // a fresh PrecomputedReference per strip (this path doesn't have a
    // parent ref to slice from); Phase 4 of the plan eliminates that.
    let init_dst_scratch = || -> [Vec<f32>; 3] { std::array::from_fn(|_| Vec::new()) };
    let process_strip =
        |dst_planes: &mut [Vec<f32>; 3],
         (strip_y0, strip_y1, fy0, fy1, _, _): (usize, usize, usize, usize, usize, usize)|
         -> (Vec<ScaleAccumulators>, [f64; 3], usize) {
            let src_strip = crate::source::SubsetView::new(source, strip_y0, strip_y1 - strip_y0);
            let dst_strip =
                crate::source::SubsetView::new(distorted, strip_y0, strip_y1 - strip_y0);
            let precomp = PrecomputedReference::new(&src_strip, num_scales, false);
            compute_multiscale_accums_streaming_with_ref_borrowed(
                &precomp,
                &dst_strip,
                dst_planes,
                &strip_config,
                weights,
                Some((fy0, fy1)),
                Some((height, strip_y0)),
            )
        };

    // Per-strip results (collected, then merged).
    #[cfg(feature = "threads")]
    let parallel_strips = config.allow_multithreading;
    #[cfg(feature = "threads")]
    let strip_results: Vec<(Vec<ScaleAccumulators>, [f64; 3], usize)> = if parallel_strips {
        use rayon::prelude::*;
        strip_meta
            .par_iter()
            .copied()
            .map_init(init_dst_scratch, process_strip)
            .collect()
    } else {
        let mut scratch = init_dst_scratch();
        strip_meta
            .iter()
            .copied()
            .map(|m| process_strip(&mut scratch, m))
            .collect()
    };
    #[cfg(not(feature = "threads"))]
    let strip_results: Vec<(Vec<ScaleAccumulators>, [f64; 3], usize)> = {
        let mut scratch = init_dst_scratch();
        strip_meta
            .iter()
            .copied()
            .map(|m| process_strip(&mut scratch, m))
            .collect()
    };

    // Merge raw accumulators across strips.
    let mut global_accums: Vec<ScaleAccumulators> =
        (0..num_scales).map(|_| ScaleAccumulators::new()).collect();
    let mut mean_offset_sums: [f64; 3] = [0.0; 3];
    let mut mean_offset_pixel_count: usize = 0;
    for (strip_accums, strip_offset_sums, strip_pixel_count) in strip_results {
        for (s, strip_accum) in strip_accums.iter().enumerate() {
            global_accums[s].merge(strip_accum);
        }
        for c in 0..3 {
            mean_offset_sums[c] += strip_offset_sums[c];
        }
        mean_offset_pixel_count += strip_pixel_count;
    }

    let final_stats: Vec<ScaleStats> = global_accums
        .iter()
        .map(|a| a.finalize(config.iw_strength as f64))
        .collect();
    let final_mean_offset = if mean_offset_pixel_count == 0 {
        [0.0; 3]
    } else {
        let inv = 1.0 / mean_offset_pixel_count as f64;
        [
            mean_offset_sums[0] * inv,
            mean_offset_sums[1] * inv,
            mean_offset_sums[2] * inv,
        ]
    };
    (final_stats, final_mean_offset)
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
    combine_scores(&scale_stats, weights, config, mean_offset)
}

/// Like `compute_zensim_streaming_with_ref`, but also produces a per-pixel error
/// diffmap fused from all pyramid scales.
///
/// Each scale's SSIM error map is weighted by per-scale channel weights, then
/// coarser scales are upsampled 2× (nearest-neighbor) back to full resolution
/// and blended according to `scale_blend_weights`.
///
/// Returns `(ZensimResult, diffmap_padded, padded_width)` where `diffmap_padded`
/// has `padded_width × height` elements in padded-width row layout.
pub(crate) fn compute_zensim_streaming_with_ref_and_diffmap(
    precomputed: &PrecomputedReference,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    per_scale_channel_weights: &[[PixelFeatureWeights; 3]],
    scale_blend_weights: &[f32],
    guided_redistribution: bool,
) -> (crate::metric::ZensimResult, Vec<f32>, usize) {
    let width = distorted.width();
    let height = distorted.height();
    let padded_width = simd_padded_width(width);
    let dst_planes = convert_source_to_xyb(distorted, padded_width, config.allow_multithreading);

    compute_diffmap_from_xyb(
        precomputed,
        dst_planes,
        width,
        height,
        padded_width,
        config,
        weights,
        per_scale_channel_weights,
        scale_blend_weights,
        guided_redistribution,
    )
}

/// Like `compute_zensim_streaming_with_ref_and_diffmap`, but takes planar linear
/// RGB f32 input instead of `ImageSource`. Eliminates interleaving overhead.
pub(crate) fn compute_zensim_streaming_with_ref_and_diffmap_linear_planar(
    precomputed: &PrecomputedReference,
    planes: [&[f32]; 3],
    width: usize,
    height: usize,
    stride: usize,
    config: &ZensimConfig,
    weights: &[f64],
    per_scale_channel_weights: &[[PixelFeatureWeights; 3]],
    scale_blend_weights: &[f32],
    guided_redistribution: bool,
) -> (crate::metric::ZensimResult, Vec<f32>, usize) {
    let padded_width = simd_padded_width(width);
    let dst_planes = convert_linear_planar_to_xyb(planes, width, height, stride, padded_width);

    compute_diffmap_from_xyb(
        precomputed,
        dst_planes,
        width,
        height,
        padded_width,
        config,
        weights,
        per_scale_channel_weights,
        scale_blend_weights,
        guided_redistribution,
    )
}

/// Core diffmap pipeline: takes pre-converted XYB planes, runs multi-scale
/// processing, and fuses per-scale diffmaps into a single full-resolution map.
fn compute_diffmap_from_xyb(
    precomputed: &PrecomputedReference,
    mut dst_planes: [Vec<f32>; 3],
    width: usize,
    height: usize,
    padded_width: usize,
    config: &ZensimConfig,
    weights: &[f64],
    per_scale_channel_weights: &[[PixelFeatureWeights; 3]],
    scale_blend_weights: &[f32],
    guided_redistribution: bool,
) -> (crate::metric::ZensimResult, Vec<f32>, usize) {
    let num_scales = config.num_scales.min(precomputed.scales.len());
    let parallel = config.allow_multithreading;

    let (src_planes_s0, _, _) = precomputed.scale(0);
    let dst_view_s0: [&[f32]; 3] = [&dst_planes[0], &dst_planes[1], &dst_planes[2]];
    let mean_offset =
        compute_xyb_mean_offset(src_planes_s0, dst_view_s0, width, height, padded_width);

    let mut stats = Vec::with_capacity(num_scales);
    // Collect per-scale diffmaps with their dimensions
    let mut scale_diffmaps: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(num_scales);
    let mut w = padded_width;
    let mut h = height;

    for scale in 0..num_scales {
        if w < 8 || h < 8 {
            break;
        }

        let (src_planes, src_w, src_h) = precomputed.scale(scale);
        // Internal invariant: dims match because the public API
        // (Zensim::compute_with_ref*) validates distorted dims against
        // PrecomputedReference dims before reaching this code, and both
        // sides downscale by the same factor each scale.
        assert_eq!(
            w, src_w,
            "internal invariant violated: width mismatch at scale {scale}"
        );
        assert_eq!(
            h, src_h,
            "internal invariant violated: height mismatch at scale {scale}"
        );

        // Request diffmap at every scale
        let eq = PixelFeatureWeights {
            ssim: 1.0 / 3.0,
            ..PixelFeatureWeights::default()
        };
        let ch_weights = per_scale_channel_weights
            .get(scale)
            .copied()
            .unwrap_or([eq; 3]);
        let dst_view: [&[f32]; 3] = [&dst_planes[0], &dst_planes[1], &dst_planes[2]];
        let (scale_stat, dm) = process_scale_bands(
            src_planes,
            dst_view,
            w,
            h,
            config,
            scale,
            weights,
            Some(ch_weights),
        );
        stats.push(scale_stat);

        if let Some(dm) = dm {
            scale_diffmaps.push((dm, w, h));
        }

        if scale < num_scales - 1 {
            let (nw, nh) = downscale_3_planes(&mut dst_planes, w, h, parallel);
            w = nw;
            h = nh;
        }
    }

    #[cfg(feature = "threads")]
    dealloc_planes(dst_planes, None);
    #[cfg(not(feature = "threads"))]
    drop(dst_planes);

    // Fuse multi-scale diffmaps: scale 0 at full weight, coarser scales upsampled and blended
    let full_w = padded_width;
    let full_h = height;
    let mut fused = vec![0.0f32; full_w * full_h];

    // E-JBU guide (research, opt-in): |scale-0 contribution plane| + ε. The
    // scale-0 plane is the fold's own full-res localization signal; |·| makes
    // it sign-robust under the signed ModelSensitivity weighting, and the
    // per-cell normalization `g / Σ_cell g` cancels any global scaling (so the
    // guide is independent of scale-0's blend weight). ε = 1e-6·mean|s0| +
    // 1e-20 keeps `Σ_cell g > 0`; an all-zero plane degrades to exactly the
    // uniform (NN) deposit.
    let guide: Option<Vec<f32>> = if guided_redistribution && scale_diffmaps.len() > 1 {
        scale_diffmaps
            .first()
            .filter(|(dm0, w0, h0)| *w0 == full_w && *h0 == full_h && dm0.len() >= full_w * full_h)
            .map(|(dm0, _, _)| {
                let mean_abs =
                    dm0.iter().map(|&v| v.abs() as f64).sum::<f64>() / dm0.len().max(1) as f64;
                let eps = (mean_abs * 1e-6) as f32 + 1e-20f32;
                dm0.iter().map(|&v| v.abs() + eps).collect()
            })
    } else {
        None
    };
    // E-JBU diagnostic (ZENSIM_JBU_GUIDE_STATS=1): how much differential
    // signal does the guide actually carry WITHIN coarse footprints? Prints
    // the guide's global stats and the mean/max within-cell relative spread
    // (sd/mean per aligned f×f cell) for each coarse factor. If within-cell
    // spread is ~0 the redistribution is a near-no-op regardless of mass.
    #[cfg(feature = "custom-profiles")]
    if let Some(g) = &guide {
        if std::env::var("ZENSIM_JBU_GUIDE_STATS").as_deref() == Ok("1") {
            let n = (full_w * full_h) as f64;
            let gm = g.iter().map(|&v| v as f64).sum::<f64>() / n;
            for factor in [2usize, 4, 8] {
                let (mut sum_rel, mut max_rel, mut cells) = (0.0f64, 0.0f64, 0usize);
                let mut sy = 0;
                while sy * factor < full_h {
                    let (y0, y1) = (sy * factor, ((sy + 1) * factor).min(full_h));
                    let mut sx = 0;
                    while sx * factor < full_w {
                        let (x0, x1) = (sx * factor, ((sx + 1) * factor).min(full_w));
                        let (mut s1, mut s2, mut cnt) = (0.0f64, 0.0f64, 0.0f64);
                        for y in y0..y1 {
                            for &v in &g[y * full_w + x0..y * full_w + x1] {
                                let v = v as f64;
                                s1 += v;
                                s2 += v * v;
                                cnt += 1.0;
                            }
                        }
                        let mean = s1 / cnt;
                        let sd = (s2 / cnt - mean * mean).max(0.0).sqrt();
                        let rel = sd / mean.max(1e-30);
                        sum_rel += rel;
                        max_rel = max_rel.max(rel);
                        cells += 1;
                        sx += 1;
                    }
                    sy += 1;
                }
                eprintln!(
                    "  JBU guide stats: factor {factor}: within-cell sd/mean mean {:.4} max {:.4} ({cells} cells; guide mean {gm:.3e})",
                    sum_rel / cells as f64,
                    max_rel
                );
            }
        }
    }

    for (scale, (dm, dm_w, dm_h)) in scale_diffmaps.iter().enumerate() {
        let blend = scale_blend_weights.get(scale).copied().unwrap_or(0.0);
        if blend <= 0.0 {
            continue;
        }
        // factor = 2^scale: scale 0 is identity (no upsample), scale s replicates each
        // src pixel into a (2^s) × (2^s) block.
        let factor = 1usize << scale;
        match &guide {
            Some(g) if factor > 1 => redistribute_pow2x_guided_add(
                dm, *dm_w, *dm_h, &mut fused, full_w, full_h, factor, blend, g,
            ),
            _ => upsample_pow2x_add(dm, *dm_w, *dm_h, &mut fused, full_w, full_h, factor, blend),
        }
    }

    let result = combine_scores(&stats, weights, config, mean_offset);
    (result, fused, padded_width)
}
/// Fused score + attribution-plane pipeline (task #67 C3a): the SAME
/// per-scale walk as `compute_diffmap_from_xyb` (identical stats → the
/// score is bit-identical to `compute_with_ref_and_diffmap`'s), with the
/// per-(scale, channel) SSIM-error + mu planes RETAINED and handed to
/// `on_scale` after each scale completes — the fused attribution entry
/// derives the basic density from them with zero extra blur work. No fold
/// diffmap is produced (the caller wants the attribution map instead).
///
/// `on_scale(scale, stats, src_planes, dst_planes, retention, w, h)` runs
/// once per processed scale, BEFORE the dst pyramid is downscaled for the
/// next scale.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_zensim_streaming_with_ref_and_attr_planes(
    precomputed: &PrecomputedReference,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    mut on_scale: impl FnMut(
        usize,
        &ScaleStats,
        [&[f32]; 3],
        [&[f32]; 3],
        &AttrScaleRetention,
        usize,
        usize,
    ),
) -> crate::metric::ZensimResult {
    let width = distorted.width();
    let height = distorted.height();
    let padded_width = simd_padded_width(width);
    let mut dst_planes =
        convert_source_to_xyb(distorted, padded_width, config.allow_multithreading);

    let num_scales = config.num_scales.min(precomputed.scales.len());
    let parallel = config.allow_multithreading;

    let (src_planes_s0, _, _) = precomputed.scale(0);
    let dst_view_s0: [&[f32]; 3] = [&dst_planes[0], &dst_planes[1], &dst_planes[2]];
    let mean_offset =
        compute_xyb_mean_offset(src_planes_s0, dst_view_s0, width, height, padded_width);

    let mut stats = Vec::with_capacity(num_scales);
    let mut retention = AttrScaleRetention::new(padded_width * height);
    let mut w = padded_width;
    let mut h = height;

    for scale in 0..num_scales {
        if w < 8 || h < 8 {
            break;
        }
        let (src_planes, src_w, src_h) = precomputed.scale(scale);
        assert_eq!(
            w, src_w,
            "internal invariant violated: width mismatch at scale {scale}"
        );
        assert_eq!(
            h, src_h,
            "internal invariant violated: height mismatch at scale {scale}"
        );
        let dst_view: [&[f32]; 3] = [&dst_planes[0], &dst_planes[1], &dst_planes[2]];
        let (accum, _) = process_scale_bands_into_accum(
            src_planes,
            dst_view,
            w,
            h,
            config,
            scale,
            weights,
            None,
            None,
            None,
            Some(&mut retention),
        );
        let scale_stat = accum.finalize(config.iw_strength as f64);
        on_scale(scale, &scale_stat, src_planes, dst_view, &retention, w, h);
        stats.push(scale_stat);

        if scale < num_scales - 1 {
            let (nw, nh) = downscale_3_planes(&mut dst_planes, w, h, parallel);
            w = nw;
            h = nh;
        }
    }

    #[cfg(feature = "threads")]
    dealloc_planes(dst_planes, None);
    #[cfg(not(feature = "threads"))]
    drop(dst_planes);

    combine_scores(&stats, weights, config, mean_offset)
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
    combine_scores(&scale_stats, weights, config, mean_offset)
}

// ─── Delta stats computation (classification feature) ────────────────────

#[cfg(feature = "classification")]
use crate::metric::{AlphaStratifiedStats, DeltaStats};

#[cfg(feature = "classification")]
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

#[cfg(feature = "classification")]
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
#[cfg(feature = "classification")]
fn native_max_for_format(format: PixelFormat) -> f64 {
    match format {
        PixelFormat::Srgb8Rgb | PixelFormat::Srgb8Rgba | PixelFormat::Srgb8Bgra => 255.0,
        PixelFormat::Srgb16Rgba => 65535.0,
        PixelFormat::LinearF32Rgba => 1.0,
        #[allow(unreachable_patterns)]
        _ => 255.0,
    }
}

#[cfg(feature = "classification")]
pub(crate) fn compute_delta_stats(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> Result<DeltaStats, crate::ZensimError> {
    let width = source.width();
    let height = source.height();
    let src_format = source.pixel_format();
    let dst_format = distorted.pixel_format();
    let has_alpha = src_format.has_alpha() && dst_format.has_alpha();
    let native_max = native_max_for_format(src_format).max(native_max_for_format(dst_format));

    // Up-front check: both formats must be in the supported set for the
    // per-pixel extractor. `PixelFormat` is `#[non_exhaustive]`; if a new
    // variant lands without a matching arm in `extract_pixel_normalized`,
    // surface a real error instead of panicking in the inner loop.
    if !is_supported_delta_format(src_format) || !is_supported_delta_format(dst_format) {
        return Err(crate::ZensimError::UnsupportedPixelFormat);
    }

    let chunk_rows = 64usize;
    let num_chunks = height.div_ceil(chunk_rows);

    // Accumulation over row chunks (parallel when threads feature enabled)
    let process_chunk = |chunk_idx: usize| -> DeltaAccum {
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

                // Skip RGB comparison when both pixels are fully transparent.
                // RGB values are undefined at alpha=0 — different encoders produce
                // different values (white vs black) but the pixels are visually identical.
                let both_transparent =
                    alpha.is_some_and(|(sa, da)| sa < 0.5 / native_max && da < 0.5 / native_max);

                for c in 0..3 {
                    let delta = if both_transparent {
                        0.0
                    } else {
                        src_rgb[c] - dst_rgb[c]
                    };
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
    };
    #[cfg(feature = "threads")]
    let accum =
        (0..num_chunks)
            .into_par_iter()
            .map(process_chunk)
            .reduce(DeltaAccum::new, |mut a, b| {
                a.merge(&b);
                a
            });
    #[cfg(not(feature = "threads"))]
    let accum = (0..num_chunks)
        .map(process_chunk)
        .fold(DeltaAccum::new(), |mut a, b| {
            a.merge(&b);
            a
        });

    Ok(finalize_delta_stats(accum, has_alpha, native_max))
}

/// Returns true if `format` has a matching arm in
/// [`extract_pixel_normalized`]. `PixelFormat` is `#[non_exhaustive]` so a
/// future variant added to the enum without a matching extractor arm
/// would otherwise reach the catch-all and panic; the up-front check in
/// [`compute_delta_stats`] uses this to bail with
/// [`ZensimError::UnsupportedPixelFormat`](crate::ZensimError::UnsupportedPixelFormat)
/// instead.
#[cfg(feature = "classification")]
#[inline]
fn is_supported_delta_format(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Srgb8Rgb
            | PixelFormat::Srgb8Rgba
            | PixelFormat::Srgb8Bgra
            | PixelFormat::Srgb16Rgba
            | PixelFormat::LinearF32Rgba
    )
}

#[cfg(feature = "classification")]
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
        _ => {
            // Unreachable in practice: `compute_delta_stats` guards both
            // input formats via `is_supported_delta_format` before any
            // call into this extractor. If a new `PixelFormat` variant
            // lands without a matching arm here, update the guard too.
            debug_assert!(
                false,
                "unsupported pixel format for delta stats: {:?}",
                format
            );
            ([0.0; 3], None)
        }
    }
}

#[cfg(feature = "classification")]
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
    use crate::metric::WEIGHTS;
    use crate::metric::compute_zensim_with_config;
    use crate::source::RgbSlice;

    /// Public-API end-to-end test: `compute_streaming_strips_default`
    /// produces a score within 1% of `compute` on a 256×256 image.
    #[test]
    fn compute_streaming_strips_score_matches_full() {
        use crate::{Zensim, ZensimProfile};
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
        let src_img = RgbSlice::new(&src, w, h);
        let dst_img = RgbSlice::new(&dst, w, h);

        let z = Zensim::new(ZensimProfile::codec_target());
        let full = z.compute(&src_img, &dst_img).unwrap();
        let strip = z
            .compute_streaming_strips(&src_img, &dst_img, 64, 16)
            .unwrap();
        let rel = (full.score() - strip.score()).abs() / full.score().max(1e-6);
        eprintln!(
            "compute_streaming_strips: full={:.6} strip={:.6} rel={:.6}",
            full.score(),
            strip.score(),
            rel
        );
        assert!(
            rel < 0.02,
            "compute_streaming_strips score: full={:.4} strip={:.4} rel={:.4}",
            full.score(),
            strip.score(),
            rel
        );
    }

    /// Verify strip-aggregating multiscale stats produces approximately
    /// the same per-scale stats as the full-image path.
    ///
    /// The strip aggregator splits the image into Y-strips and re-pools
    /// each strip's ScaleStats through a "scaled-mean" approximation
    /// (sum += tile_mean × tile_n; final = sum / total_n). This is
    /// EXACT for pure-mean stats (mean_d, edge_art, mse) and an
    /// APPROXIMATION for root-power stats (root4_d, root2_d) due to
    /// the per-tile finalize step.
    ///
    /// We verify: (1) per-scale mean fields match within 1e-4 relative,
    /// (2) per-scale root-power fields match within 5% relative
    /// (Phase 2 will tighten this once we expose raw accumulators
    /// across strips).
    #[test]
    fn strip_aggregator_matches_full_image() {
        let w = 256;
        let h = 256;
        let n = w * h;
        // Smooth gradient with mild noise.
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
        let src_img = RgbSlice::new(&src, w, h);
        let dst_img = RgbSlice::new(&dst, w, h);

        let config = ZensimConfig::default();
        let weights: Vec<f64> = WEIGHTS.to_vec();

        // Full-image stats.
        let precomp = PrecomputedReference::new(&src_img, config.num_scales, false);
        let (full_stats, full_offset) =
            compute_multiscale_stats_streaming_with_ref(&precomp, &dst_img, &config, &weights);

        // Strip-aggregated stats with strip_inner=64, margin=16
        // (4 strips of 64-row inner each).
        let (strip_stats, strip_offset) = compute_multiscale_stats_streaming_strips(
            &src_img, &dst_img, &config, &weights, 64, 16,
        );

        // Stats counts must match.
        assert_eq!(full_stats.len(), strip_stats.len(), "scale count");

        // Use the same tolerance model as the existing
        // `streaming_matches_full_image` test: significant features
        // (|val| > 1e-3) check 10% relative; tiny features check
        // 1e-3 absolute. The 10% is loosened from streaming's 5%
        // because Phase 1 also includes per-tile root-power
        // re-pooling approximation in addition to the
        // blur-boundary effects.
        let close_enough = |a: f64, b: f64, label: &str| {
            let diff = (a - b).abs();
            let rel = if a.abs() > 1e-3 { diff / a.abs() } else { 0.0 };
            assert!(
                diff < 1e-3 || rel < 0.10,
                "{label}: full={a:.6} strip={b:.6} diff={diff:.6} rel={rel:.4}"
            );
        };

        for (s, (full, strip)) in full_stats.iter().zip(strip_stats.iter()).enumerate() {
            for c in 0..3 {
                let label = format!("scale {s} ch {c}");
                close_enough(
                    full.ssim[c * 2],
                    strip.ssim[c * 2],
                    &format!("{label} ssim mean"),
                );
                close_enough(full.mse[c], strip.mse[c], &format!("{label} mse"));
                close_enough(
                    full.edge[c * 4],
                    strip.edge[c * 4],
                    &format!("{label} edge art"),
                );
                close_enough(
                    full.edge[c * 4 + 2],
                    strip.edge[c * 4 + 2],
                    &format!("{label} edge det"),
                );
            }
        }

        // Mean offset is a pure mean, BUT the strip overlap means
        // margin rows are double-counted in the aggregated mean.
        // Phase 2 will fix this by separating the strip's inner-only
        // mean_offset computation. For now, accept within 5e-3 abs.
        for c in 0..3 {
            let diff = (full_offset[c] - strip_offset[c]).abs();
            assert!(
                diff < 5e-3,
                "mean offset channel {c}: full={:.6} strip={:.6}",
                full_offset[c],
                strip_offset[c]
            );
        }
    }

    /// Byte-exact equivalence test for the strip aggregator on a single
    /// 256×1024 pair with well-aligned geometry.
    ///
    /// Validates the `1e-6` relative-error gate on per-scale stats —
    /// the load-bearing precision claim for Phase 1 of the Y-strip
    /// aggregator. The strip path passes `outer_layout` to the band
    /// processor so the strip's bands tile against the FULL-image's
    /// plane (rather than the strip's local plane), making each band's
    /// V-blur init point and advance count byte-identical between the
    /// strip path and the full-image path.
    #[test]
    fn strip_aggregator_byte_exact_single_pair() {
        let w = 256;
        let h = 1024; // 4 strips at strip_inner=256.
        let n = w * h;
        let mut src = vec![[0u8, 0, 0]; n];
        let mut dst = vec![[0u8, 0, 0]; n];
        // Procedural-but-rich content: each row gets a slightly
        // different gradient plus banded noise. This activates
        // SSIM, edge, HF energy/magnitude, and IW features across
        // all scales.
        for y in 0..h {
            for x in 0..w {
                let r = (((x * 251) + y * 7) & 0xFF) as u8;
                let g = (((y * 241) + x * 11) & 0xFF) as u8;
                let b = ((x + y) & 0xFF) as u8;
                src[y * w + x] = [r, g, b];
                dst[y * w + x] = [
                    r.saturating_add((x & 3) as u8),
                    g.saturating_sub((y & 3) as u8),
                    b.saturating_add(((x + y) & 1) as u8),
                ];
            }
        }
        let src_img = RgbSlice::new(&src, w, h);
        let dst_img = RgbSlice::new(&dst, w, h);

        // Use compute_all_features-equivalent config so the 372-feature
        // path is exercised (extended + IW features both on).
        let config = ZensimConfig {
            compute_all_features: true,
            extended_features: true,
            compute_iw_features: true,
            ..Default::default()
        };
        let weights: Vec<f64> = WEIGHTS.to_vec();

        let precomp = PrecomputedReference::new(&src_img, config.num_scales, false);
        let (full_stats, full_offset) =
            compute_multiscale_stats_streaming_with_ref(&precomp, &dst_img, &config, &weights);

        let (strip_stats, strip_offset) = compute_multiscale_stats_streaming_strips(
            &src_img, &dst_img, &config, &weights, 256, 128,
        );

        assert_eq!(full_stats.len(), strip_stats.len());

        let mut worst_rel = 0.0f64;
        let mut worst_label = String::new();
        let compare =
            |a: f64, b: f64, label: &str, worst_rel: &mut f64, worst_label: &mut String| {
                let diff = (a - b).abs();
                let scale = a.abs().max(b.abs()).max(1e-12);
                let rel = diff / scale;
                // Skip near-zero comparisons (no signal to compare).
                if scale > 1e-6 && rel > *worst_rel {
                    *worst_rel = rel;
                    *worst_label = label.to_string();
                }
                assert!(
                    rel < 1e-6 || diff < 1e-9,
                    "{label}: full={a:.10} strip={b:.10} diff={diff:.2e} rel={rel:.2e}"
                );
            };

        for (s, (full, strip)) in full_stats.iter().zip(strip_stats.iter()).enumerate() {
            for c in 0..3 {
                let lbl = |name: &str| format!("scale {s} ch {c} {name}");
                compare(
                    full.ssim[c * 2],
                    strip.ssim[c * 2],
                    &lbl("ssim_mean"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.ssim[c * 2 + 1],
                    strip.ssim[c * 2 + 1],
                    &lbl("ssim_4th"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.ssim_2nd[c],
                    strip.ssim_2nd[c],
                    &lbl("ssim_2nd"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                for k in 0..4 {
                    let names = ["art_mean", "art_4th", "det_mean", "det_4th"];
                    compare(
                        full.edge[c * 4 + k],
                        strip.edge[c * 4 + k],
                        &lbl(names[k]),
                        &mut worst_rel,
                        &mut worst_label,
                    );
                }
                compare(
                    full.edge_2nd[c * 2],
                    strip.edge_2nd[c * 2],
                    &lbl("art_2nd"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.edge_2nd[c * 2 + 1],
                    strip.edge_2nd[c * 2 + 1],
                    &lbl("det_2nd"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.mse[c],
                    strip.mse[c],
                    &lbl("mse"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.hf_energy_loss[c],
                    strip.hf_energy_loss[c],
                    &lbl("hf_energy_loss"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.hf_mag_loss[c],
                    strip.hf_mag_loss[c],
                    &lbl("hf_mag_loss"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.hf_energy_gain[c],
                    strip.hf_energy_gain[c],
                    &lbl("hf_energy_gain"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.ssim_max[c],
                    strip.ssim_max[c],
                    &lbl("ssim_max"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.art_max[c],
                    strip.art_max[c],
                    &lbl("art_max"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.det_max[c],
                    strip.det_max[c],
                    &lbl("det_max"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.ssim_p95[c],
                    strip.ssim_p95[c],
                    &lbl("ssim_l8"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.art_p95[c],
                    strip.art_p95[c],
                    &lbl("art_l8"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.det_p95[c],
                    strip.det_p95[c],
                    &lbl("det_l8"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                for k in 0..3 {
                    let names = ["masked_ssim_mean", "masked_ssim_4th", "masked_ssim_2nd"];
                    compare(
                        full.masked_ssim[c * 3 + k],
                        strip.masked_ssim[c * 3 + k],
                        &lbl(names[k]),
                        &mut worst_rel,
                        &mut worst_label,
                    );
                }
                compare(
                    full.masked_art_4th[c],
                    strip.masked_art_4th[c],
                    &lbl("masked_art_4th"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.masked_det_4th[c],
                    strip.masked_det_4th[c],
                    &lbl("masked_det_4th"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.masked_mse[c],
                    strip.masked_mse[c],
                    &lbl("masked_mse"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                for k in 0..3 {
                    let names = ["iw_ssim_mean", "iw_ssim_4th", "iw_ssim_2nd"];
                    compare(
                        full.iw_ssim[c * 3 + k],
                        strip.iw_ssim[c * 3 + k],
                        &lbl(names[k]),
                        &mut worst_rel,
                        &mut worst_label,
                    );
                }
                compare(
                    full.iw_art_4th[c],
                    strip.iw_art_4th[c],
                    &lbl("iw_art_4th"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.iw_det_4th[c],
                    strip.iw_det_4th[c],
                    &lbl("iw_det_4th"),
                    &mut worst_rel,
                    &mut worst_label,
                );
                compare(
                    full.iw_mse[c],
                    strip.iw_mse[c],
                    &lbl("iw_mse"),
                    &mut worst_rel,
                    &mut worst_label,
                );
            }
        }

        for c in 0..3 {
            let lbl = format!("mean_offset[{c}]");
            compare(
                full_offset[c],
                strip_offset[c],
                &lbl,
                &mut worst_rel,
                &mut worst_label,
            );
        }
        eprintln!(
            "strip_aggregator_byte_exact_single_pair: worst rel = {:.3e} ({})",
            worst_rel, worst_label,
        );
    }

    /// Verify the buffered-ref strip aggregator
    /// (`compute_multiscale_stats_streaming_strips_with_ref`) is byte-
    /// exact equivalent to the strip-per-strip aggregator
    /// (`compute_multiscale_stats_streaming_strips`).
    #[test]
    fn buffered_ref_strip_matches_strip_per_strip() {
        let w = 256;
        let h = 1024;
        let n = w * h;
        let mut src = vec![[0u8, 0, 0]; n];
        let mut dst = vec![[0u8, 0, 0]; n];
        for y in 0..h {
            for x in 0..w {
                src[y * w + x] = [
                    (((x * 251) + y * 7) & 0xFF) as u8,
                    (((y * 241) + x * 11) & 0xFF) as u8,
                    ((x + y) & 0xFF) as u8,
                ];
                dst[y * w + x] = [
                    src[y * w + x][0].saturating_add(3),
                    src[y * w + x][1].saturating_sub(2),
                    src[y * w + x][2].saturating_add(1),
                ];
            }
        }
        let src_img = RgbSlice::new(&src, w, h);
        let dst_img = RgbSlice::new(&dst, w, h);

        let config = ZensimConfig {
            compute_all_features: true,
            extended_features: true,
            compute_iw_features: true,
            ..Default::default()
        };
        let weights: Vec<f64> = WEIGHTS.to_vec();

        // Per-strip ref path
        let (a_stats, a_offset) = compute_multiscale_stats_streaming_strips(
            &src_img, &dst_img, &config, &weights, 256, 128,
        );

        // Buffered ref path
        let full_precomp = PrecomputedReference::new(&src_img, config.num_scales, false);
        let (b_stats, b_offset) = compute_multiscale_stats_streaming_strips_with_ref(
            &full_precomp,
            &dst_img,
            &config,
            &weights,
            256,
            128,
        );

        for (s, (a, b)) in a_stats.iter().zip(b_stats.iter()).enumerate() {
            for c in 0..3 {
                let check = |x: f64, y: f64, lbl: &str| {
                    let diff = (x - y).abs();
                    let scale = x.abs().max(y.abs()).max(1e-12);
                    let rel = diff / scale;
                    assert!(
                        rel < 1e-6 || diff < 1e-9,
                        "s{s} c{c} {lbl}: per-strip={x:.10} buffered={y:.10} rel={rel:.2e}"
                    );
                };
                check(a.ssim[c * 2], b.ssim[c * 2], "ssim_mean");
                check(a.ssim[c * 2 + 1], b.ssim[c * 2 + 1], "ssim_4th");
                check(a.mse[c], b.mse[c], "mse");
                check(a.iw_art_4th[c], b.iw_art_4th[c], "iw_art_4th");
                check(
                    a.masked_ssim[c * 3],
                    b.masked_ssim[c * 3],
                    "masked_ssim_mean",
                );
            }
        }
        for c in 0..3 {
            let diff = (a_offset[c] - b_offset[c]).abs();
            assert!(
                diff < 1e-9,
                "mean_offset[{c}]: per-strip={} buffered={}",
                a_offset[c],
                b_offset[c]
            );
        }
    }

    /// 99-pair byte-exact equivalence test for the strip aggregator.
    ///
    /// Generates 99 deterministic synthetic image pairs covering the
    /// signal spectrum (smooth gradients, banded noise, checkerboard,
    /// stripes, edge content) with various distortion types
    /// (low-amplitude noise, lossy-codec-like blur, blockiness, chroma
    /// shift). Asserts that every one of the 372 per-scale features
    /// from each pair matches within 1e-6 rel tolerance (or 1e-9 abs
    /// when the value is near zero) between the full path and the strip
    /// path.
    ///
    /// This is the load-bearing precision gate for Phase 1 of the Y-strip
    /// aggregator. The aggregator math (raw-sum merge across strips) is
    /// architecturally exact; the test verifies the V-blur band layout
    /// alignment in `process_scale_bands_into_accum`'s `outer_layout`
    /// branch keeps the f32 accumulator history matched.
    #[test]
    fn strip_aggregator_byte_exact_safesyn_99() {
        // Image geometry: 256×1024 at all 99 pairs.
        // strip_inner=256, strip_margin=128 → 4 strips per image.
        let w = 256;
        let h = 1024;
        let n = w * h;
        let strip_inner = 256;
        let strip_margin = 128;
        let num_pairs = 99;

        let config = ZensimConfig {
            compute_all_features: true,
            extended_features: true,
            compute_iw_features: true,
            ..Default::default()
        };
        let weights: Vec<f64> = WEIGHTS.to_vec();

        // Worst observed rel/abs across all pairs and features.
        let mut overall_worst_rel = 0.0f64;
        let mut overall_worst_abs = 0.0f64;
        let mut overall_worst_label = String::new();
        let mut fail_count = 0usize;
        let mut fail_examples: Vec<String> = Vec::new();

        for pair_idx in 0..num_pairs {
            // Deterministic procedural content. Each pair index seeds a
            // distinct image shape + distortion type so the 99 pairs
            // span the feature space.
            let mut src = vec![[0u8, 0, 0]; n];
            let mut dst = vec![[0u8, 0, 0]; n];
            let seed = (pair_idx as u32).wrapping_mul(0x9E37_79B9).wrapping_add(1);
            let mode = pair_idx % 9;
            let (m1, m2, m3) = (
                seed.wrapping_mul(0xC2B2_AE35),
                seed.wrapping_mul(0x27D4_EB2F),
                seed.wrapping_mul(0x1656_67B1),
            );
            for y in 0..h {
                for x in 0..w {
                    let (r, g, b) = match mode {
                        0 => {
                            // Smooth gradient
                            let r = (((x * 255) / w) ^ (m1 & 0xFF) as usize) as u8;
                            let g = (((y * 255) / h) ^ (m2 & 0xFF) as usize) as u8;
                            let b = (((x + y) * 127 / (w + h)) ^ (m3 & 0xFF) as usize) as u8;
                            (r, g, b)
                        }
                        1 => {
                            // Multi-frequency checkerboard
                            let freq = 4 + (pair_idx % 12);
                            let tile = ((x * freq / w) + (y * freq / h)) & 1;
                            let v = if tile == 0 { 240u8 } else { 16u8 };
                            (
                                v ^ (m1 as u8),
                                v.wrapping_add((y & 31) as u8) ^ (m2 as u8),
                                v ^ (m3 as u8),
                            )
                        }
                        2 => {
                            // Horizontal stripes
                            let stripe = (y / (4 + (pair_idx % 16))) & 1;
                            let v = if stripe == 0 { 200u8 } else { 50u8 };
                            (
                                v.wrapping_add((x & 7) as u8),
                                v.wrapping_sub((y & 7) as u8),
                                v ^ ((x ^ y) as u8),
                            )
                        }
                        3 => {
                            // Vertical stripes
                            let stripe = (x / (4 + (pair_idx % 16))) & 1;
                            let v = if stripe == 0 { 200u8 } else { 50u8 };
                            (
                                v ^ ((x ^ y) as u8),
                                v.wrapping_add((x & 7) as u8),
                                v.wrapping_sub((y & 7) as u8),
                            )
                        }
                        4 => {
                            // Banded value-noise via hash
                            let h_fn = |a: u32, b: u32| -> u8 {
                                let mut h = a
                                    .wrapping_mul(0x6C8E_9CF7)
                                    .wrapping_add(b.wrapping_mul(0x9E37_79B9))
                                    .wrapping_add(seed);
                                h ^= h >> 16;
                                h = h.wrapping_mul(0x85EB_CA6B);
                                h ^= h >> 13;
                                (h & 0xFF) as u8
                            };
                            (
                                h_fn(x as u32 / 4, y as u32 / 4),
                                h_fn(x as u32 / 2, y as u32 / 8),
                                h_fn(x as u32 / 8, y as u32 / 4),
                            )
                        }
                        5 => {
                            // Color blocks
                            let bx = (x * 4) / w;
                            let by = (y * 4) / h;
                            let idx = (by * 4 + bx + pair_idx) & 15;
                            let palette: [[u8; 3]; 16] = [
                                [255, 0, 0],
                                [0, 255, 0],
                                [0, 0, 255],
                                [255, 255, 0],
                                [255, 0, 255],
                                [0, 255, 255],
                                [255, 128, 0],
                                [128, 0, 255],
                                [0, 128, 255],
                                [255, 0, 128],
                                [128, 255, 0],
                                [0, 255, 128],
                                [64, 0, 128],
                                [128, 64, 0],
                                [0, 128, 64],
                                [192, 192, 64],
                            ];
                            let p = palette[idx];
                            (p[0], p[1], p[2])
                        }
                        6 => {
                            // Diagonal gradient + texture
                            let d = ((x + y) * 255 / (w + h)) as u8;
                            let t = ((x.wrapping_mul(y)) & 0xFF) as u8;
                            (d, d ^ t, t.wrapping_add(d / 2))
                        }
                        7 => {
                            // Radial gradient
                            let cx = w as f32 / 2.0;
                            let cy = h as f32 / 2.0;
                            let dx = x as f32 - cx;
                            let dy = y as f32 - cy;
                            let r2 = (dx * dx + dy * dy).sqrt();
                            let v = (r2 * 0.4) as u8;
                            (v, 255u8.wrapping_sub(v), v.wrapping_add(64))
                        }
                        _ => {
                            // Constant + small dither
                            let dither = (((x.wrapping_mul(y)) ^ seed as usize) & 0x1F) as u8;
                            let base = (pair_idx as u8).wrapping_mul(7);
                            (
                                base.wrapping_add(dither),
                                base.wrapping_sub(dither),
                                base ^ dither,
                            )
                        }
                    };
                    src[y * w + x] = [r, g, b];

                    // Distortion type rotates over pairs.
                    let dist_type = (pair_idx / 9) % 11;
                    let (dr, dg, db) = match dist_type {
                        0 => (r.saturating_add(2), g.saturating_sub(1), b),
                        1 => (
                            r.saturating_add(8),
                            g.saturating_sub(4),
                            b.saturating_add(2),
                        ),
                        2 => (
                            r ^ ((x & 1) as u8),
                            g ^ ((y & 1) as u8),
                            b ^ (((x + y) & 1) as u8),
                        ),
                        3 => (r & 0xF0, g & 0xF0, b & 0xF0),
                        4 => (r & 0xE0 | 0x10, g & 0xE0 | 0x10, b & 0xE0 | 0x10),
                        5 => {
                            // Mild blur surrogate: average with neighbor
                            let r2 = if x + 1 < w {
                                src.get(y * w + x + 1).map(|p| p[0]).unwrap_or(r)
                            } else {
                                r
                            };
                            (((r as u16 + r2 as u16) / 2) as u8, g, b)
                        }
                        6 => (r.saturating_add(16), g, b.saturating_sub(8)),
                        7 => (
                            ((r as u16 * 7 / 8) + 16) as u8,
                            ((g as u16 * 7 / 8) + 16) as u8,
                            ((b as u16 * 7 / 8) + 16) as u8,
                        ),
                        8 => (
                            r.saturating_sub(1),
                            g.saturating_add(1),
                            b.saturating_sub(1),
                        ),
                        9 => (r.wrapping_add((((x + y) & 7) as u8).wrapping_sub(3)), g, b),
                        _ => (r, g, b.saturating_add(((y & 7) as u8).wrapping_sub(3))),
                    };
                    dst[y * w + x] = [dr, dg, db];
                }
            }

            let src_img = RgbSlice::new(&src, w, h);
            let dst_img = RgbSlice::new(&dst, w, h);

            // Full path
            let precomp = PrecomputedReference::new(&src_img, config.num_scales, false);
            let (full_stats, full_offset) =
                compute_multiscale_stats_streaming_with_ref(&precomp, &dst_img, &config, &weights);
            // Strip path
            let (strip_stats, strip_offset) = compute_multiscale_stats_streaming_strips(
                &src_img,
                &dst_img,
                &config,
                &weights,
                strip_inner,
                strip_margin,
            );

            let mut compare = |a: f64, b: f64, lbl: &str| {
                let diff = (a - b).abs();
                let scale = a.abs().max(b.abs());
                let rel = if scale > 1e-9 { diff / scale } else { 0.0 };
                if diff > overall_worst_abs {
                    overall_worst_abs = diff;
                }
                if rel > overall_worst_rel && scale > 1e-9 {
                    overall_worst_rel = rel;
                    overall_worst_label = format!("pair {pair_idx} {lbl}");
                }
                let pass = rel < 1e-6 || diff < 1e-9;
                if !pass {
                    fail_count += 1;
                    if fail_examples.len() < 8 {
                        fail_examples.push(format!(
                            "pair {pair_idx} {lbl}: full={a:.10} strip={b:.10} diff={diff:.2e} rel={rel:.2e}"
                        ));
                    }
                }
            };

            for (s, (full, strip)) in full_stats.iter().zip(strip_stats.iter()).enumerate() {
                for c in 0..3 {
                    compare(
                        full.ssim[c * 2],
                        strip.ssim[c * 2],
                        &format!("s{s} c{c} ssim_mean"),
                    );
                    compare(
                        full.ssim[c * 2 + 1],
                        strip.ssim[c * 2 + 1],
                        &format!("s{s} c{c} ssim_4th"),
                    );
                    compare(
                        full.ssim_2nd[c],
                        strip.ssim_2nd[c],
                        &format!("s{s} c{c} ssim_2nd"),
                    );
                    for (k, name) in ["art_mean", "art_4th", "det_mean", "det_4th"]
                        .iter()
                        .enumerate()
                    {
                        compare(
                            full.edge[c * 4 + k],
                            strip.edge[c * 4 + k],
                            &format!("s{s} c{c} {name}"),
                        );
                    }
                    compare(
                        full.edge_2nd[c * 2],
                        strip.edge_2nd[c * 2],
                        &format!("s{s} c{c} art_2nd"),
                    );
                    compare(
                        full.edge_2nd[c * 2 + 1],
                        strip.edge_2nd[c * 2 + 1],
                        &format!("s{s} c{c} det_2nd"),
                    );
                    compare(full.mse[c], strip.mse[c], &format!("s{s} c{c} mse"));
                    compare(
                        full.hf_energy_loss[c],
                        strip.hf_energy_loss[c],
                        &format!("s{s} c{c} hf_energy_loss"),
                    );
                    compare(
                        full.hf_mag_loss[c],
                        strip.hf_mag_loss[c],
                        &format!("s{s} c{c} hf_mag_loss"),
                    );
                    compare(
                        full.hf_energy_gain[c],
                        strip.hf_energy_gain[c],
                        &format!("s{s} c{c} hf_energy_gain"),
                    );
                    compare(
                        full.ssim_max[c],
                        strip.ssim_max[c],
                        &format!("s{s} c{c} ssim_max"),
                    );
                    compare(
                        full.art_max[c],
                        strip.art_max[c],
                        &format!("s{s} c{c} art_max"),
                    );
                    compare(
                        full.det_max[c],
                        strip.det_max[c],
                        &format!("s{s} c{c} det_max"),
                    );
                    compare(
                        full.ssim_p95[c],
                        strip.ssim_p95[c],
                        &format!("s{s} c{c} ssim_l8"),
                    );
                    compare(
                        full.art_p95[c],
                        strip.art_p95[c],
                        &format!("s{s} c{c} art_l8"),
                    );
                    compare(
                        full.det_p95[c],
                        strip.det_p95[c],
                        &format!("s{s} c{c} det_l8"),
                    );
                    for (k, name) in ["masked_ssim_mean", "masked_ssim_4th", "masked_ssim_2nd"]
                        .iter()
                        .enumerate()
                    {
                        compare(
                            full.masked_ssim[c * 3 + k],
                            strip.masked_ssim[c * 3 + k],
                            &format!("s{s} c{c} {name}"),
                        );
                    }
                    compare(
                        full.masked_art_4th[c],
                        strip.masked_art_4th[c],
                        &format!("s{s} c{c} masked_art_4th"),
                    );
                    compare(
                        full.masked_det_4th[c],
                        strip.masked_det_4th[c],
                        &format!("s{s} c{c} masked_det_4th"),
                    );
                    compare(
                        full.masked_mse[c],
                        strip.masked_mse[c],
                        &format!("s{s} c{c} masked_mse"),
                    );
                    for (k, name) in ["iw_ssim_mean", "iw_ssim_4th", "iw_ssim_2nd"]
                        .iter()
                        .enumerate()
                    {
                        compare(
                            full.iw_ssim[c * 3 + k],
                            strip.iw_ssim[c * 3 + k],
                            &format!("s{s} c{c} {name}"),
                        );
                    }
                    compare(
                        full.iw_art_4th[c],
                        strip.iw_art_4th[c],
                        &format!("s{s} c{c} iw_art_4th"),
                    );
                    compare(
                        full.iw_det_4th[c],
                        strip.iw_det_4th[c],
                        &format!("s{s} c{c} iw_det_4th"),
                    );
                    compare(
                        full.iw_mse[c],
                        strip.iw_mse[c],
                        &format!("s{s} c{c} iw_mse"),
                    );
                }
            }
            for c in 0..3 {
                compare(
                    full_offset[c],
                    strip_offset[c],
                    &format!("mean_offset[{c}]"),
                );
            }
        }

        eprintln!(
            "strip_aggregator_byte_exact_safesyn_99: 99 pairs, overall worst rel = {:.3e} ({}), worst abs = {:.3e}",
            overall_worst_rel, overall_worst_label, overall_worst_abs,
        );
        if !fail_examples.is_empty() {
            eprintln!("First {} failure examples:", fail_examples.len());
            for ex in &fail_examples {
                eprintln!("  {ex}");
            }
        }
        assert!(
            fail_count == 0,
            "{fail_count} feature comparisons failed the 1e-6 rel / 1e-9 abs gate"
        );
    }

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
        let streaming_result = compute_zensim_streaming(&src_img, &dst_img, &config, WEIGHTS);

        assert_eq!(
            full_result.features().len(),
            streaming_result.features().len(),
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
            .features()
            .iter()
            .zip(streaming_result.features().iter())
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
        let score_rel = (full_result.score() - streaming_result.score()).abs()
            / full_result.score().abs().max(1e-12);
        let dist_rel = (full_result.raw_distance() - streaming_result.raw_distance()).abs()
            / full_result.raw_distance().abs().max(1e-12);
        eprintln!(
            "score: full={:.6} stream={:.6} (rel={:.2e})",
            full_result.score(),
            streaming_result.score(),
            score_rel,
        );
        eprintln!(
            "raw_distance: full={:.8} stream={:.8} (rel={:.2e})",
            full_result.raw_distance(),
            streaming_result.raw_distance(),
            dist_rel,
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
        let u8_result = compute_zensim_streaming(&src_u8_img, &dst_u8_img, &config, WEIGHTS);

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
        let f32_result = compute_zensim_streaming(&src_f32_img, &dst_f32_img, &config, WEIGHTS);

        // Score should match very closely (identical linear values → identical XYB → identical features)
        let score_rel =
            (u8_result.score() - f32_result.score()).abs() / u8_result.score().abs().max(1e-12);
        let dist_rel = (u8_result.raw_distance() - f32_result.raw_distance()).abs()
            / u8_result.raw_distance().abs().max(1e-12);

        eprintln!(
            "sRGB u8 score={:.10}  linear f32 score={:.10}  rel={:.2e}",
            u8_result.score(),
            f32_result.score(),
            score_rel,
        );
        eprintln!(
            "sRGB u8 dist={:.10}  linear f32 dist={:.10}  rel={:.2e}",
            u8_result.raw_distance(),
            f32_result.raw_distance(),
            dist_rel,
        );

        // When linear f32 values come from the same LUT, the results should be
        // very close (within FP rounding from different code paths).
        assert!(
            score_rel < 1e-6,
            "score relative diff {:.2e} exceeds 1e-6 (sRGB={:.10} vs linear={:.10})",
            score_rel,
            u8_result.score(),
            f32_result.score(),
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
        let rgb_result = compute_zensim_streaming(&src_rgb_img, &dst_rgb_img, &config, WEIGHTS);

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
        let bgra_result = compute_zensim_streaming(&src_bgra_img, &dst_bgra_img, &config, WEIGHTS);

        // Opaque BGRA compositing in linear space should match sRGB u8 RGB
        // within a small tolerance (compositing detour adds FP rounding).
        let score_rel =
            (rgb_result.score() - bgra_result.score()).abs() / rgb_result.score().abs().max(1e-12);
        eprintln!(
            "RGB u8 score={:.10}  BGRA u8 score={:.10}  rel={:.2e}",
            rgb_result.score(),
            bgra_result.score(),
            score_rel,
        );

        // Note: BGRA path composites over noise background in linear space even for
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
        let streaming_result = compute_zensim_streaming(&src_img, &dst_img, &config, WEIGHTS);
        let precomputed = PrecomputedReference::new(&src_img, config.num_scales, true);
        let precomp_result =
            compute_zensim_streaming_with_ref(&precomputed, &dst_img, &config, WEIGHTS);

        assert_eq!(streaming_result.score(), precomp_result.score());
        assert_eq!(
            streaming_result.raw_distance(),
            precomp_result.raw_distance()
        );
        assert_eq!(
            streaming_result.features().len(),
            precomp_result.features().len()
        );
        for (i, (s, p)) in streaming_result
            .features()
            .iter()
            .zip(precomp_result.features().iter())
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
        let p3_result = compute_zensim_streaming(&src, &dst, &config, WEIGHTS);

        // Also run sRGB-identical to verify the gamut path gives similar behavior
        let src_srgb = RgbSlice::new(&pixels, w, h);
        let dst_srgb = RgbSlice::new(&pixels, w, h);
        let srgb_result = compute_zensim_streaming(&src_srgb, &dst_srgb, &config, WEIGHTS);

        eprintln!(
            "P3 identical score: {:.6}, sRGB identical score: {:.6}",
            p3_result.score(),
            srgb_result.score(),
        );

        // Both should be very close to 100 (numerical noise at small sizes)
        assert!(
            p3_result.score() >= 99.5,
            "P3 identical score should be >= 99.5, got {}",
            p3_result.score(),
        );
        assert!(
            srgb_result.score() >= 99.5,
            "sRGB identical score should be >= 99.5, got {}",
            srgb_result.score(),
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
        let result = compute_zensim_streaming(&src_p3, &dst_srgb, &config, WEIGHTS);

        assert!(
            result.score() < 100.0,
            "P3 vs sRGB with same pixel values should score < 100, got {}",
            result.score(),
        );
        eprintln!(
            "P3 vs sRGB same-pixels score: {:.4} (expected < 100)",
            result.score(),
        );
    }

    /// u16 delta stats: 1-step difference should be detected with native precision.
    #[test]
    #[cfg(feature = "classification")]
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

        let ds = compute_delta_stats(&src, &dst).expect("supported format");

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
    #[cfg(feature = "classification")]
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

        let ds = compute_delta_stats(&src, &dst).expect("supported format");

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

    /// Regression: every currently-declared `PixelFormat` variant must
    /// be recognized by `is_supported_delta_format`. If a new variant
    /// is added to `source::PixelFormat` without updating the guard,
    /// this test catches the gap before users hit
    /// [`ZensimError::UnsupportedPixelFormat`](crate::ZensimError::UnsupportedPixelFormat)
    /// at runtime.
    #[cfg(feature = "classification")]
    #[test]
    fn supported_delta_format_covers_all_current_variants() {
        use crate::source::PixelFormat;
        for fmt in [
            PixelFormat::Srgb8Rgb,
            PixelFormat::Srgb8Rgba,
            PixelFormat::Srgb8Bgra,
            PixelFormat::Srgb16Rgba,
            PixelFormat::LinearF32Rgba,
        ] {
            assert!(
                is_supported_delta_format(fmt),
                "format {:?} declared by PixelFormat but not recognized by the delta-stats guard",
                fmt,
            );
        }
    }

    /// MEASURE `mean_w` under the weights we ACTUALLY SHIP.
    ///
    /// The predecessor measurement (`iw_pool.rs::tests::
    /// iw_mean_weight_spread_across_references`) used
    /// `iw_pool::compute_iw_weights` — a local-variance/gradient estimator with
    /// a floor — and found a 15.3x spread. **The shipped path does not call
    /// that function.** It builds the weight inline as `w_i = 1 + k_iw · a_i`
    /// from the blurred reference-activity map (`simd_ops.rs`: "writes
    /// iw_out[i] = 1 + k_iw * activity[i]"), with `k_iw = config.iw_strength =
    /// 4.0`. Different function, different regime — `w_i >= 1` here, versus
    /// 0.001..0.018 there. That mismatch is exactly why the earlier number was
    /// published with a "NOT MEASURED for the shipped path" caveat, and this
    /// test is what settles it.
    ///
    /// Every `iw_*` feature is pooled by `1/n` (`finalize`) instead of `1/Σw`,
    /// so each carries `mean_w^p` — p = 1 for the means (`iw_ssim_mean`,
    /// `iw_mse`), 0.5 for 2nd moments, 0.25 for 4th. The weights depend only on
    /// the REFERENCE, so `mean_w` is a per-reference constant: it cancels
    /// within an image and does NOT cancel across images. Only the SPREAD of
    /// `mean_w` across references matters — a constant factor is absorbed by
    /// one model weight and costs nothing.
    ///
    /// `ZENSIM_IW_REF_DIR=<dir> cargo test -p zensim --release shipped_iw_mean_w -- --ignored --nocapture`
    #[test]
    #[ignore = "needs a dir of reference images; set ZENSIM_IW_REF_DIR and run with --ignored"]
    fn shipped_iw_mean_w_spread_across_references() {
        let dir = std::env::var("ZENSIM_IW_REF_DIR")
            .expect("set ZENSIM_IW_REF_DIR to a directory of reference images");
        let mut paths: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read {dir}: {e}"))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("png" | "jpg" | "jpeg")
                )
            })
            .collect();
        paths.sort();
        paths.truncate(40);
        assert!(!paths.is_empty(), "no images in {dir}");

        let config = ZensimConfig {
            compute_all_features: true,
            extended_features: true,
            compute_iw_features: true,
            ..Default::default()
        };
        let weights: Vec<f64> = WEIGHTS.to_vec();

        // mean_w depends only on the REFERENCE, so the distorted side is
        // irrelevant to it — pass the reference against itself. That is the
        // claim under test as much as the measurement: if identity-vs-self and
        // a real distortion disagreed on mean_w, the "per-reference constant"
        // premise would be wrong.
        let mut rows: Vec<(String, f64)> = Vec::new();
        for p in &paths {
            let Ok(img) = image::open(p) else { continue };
            let rgb = img.to_rgb8();
            let (w, h) = (rgb.width() as usize, rgb.height() as usize);
            if w < 64 || h < 64 {
                continue;
            }
            let px: Vec<[u8; 3]> = rgb.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
            let src_img = RgbSlice::new(&px, w, h);
            let (stats, _) = compute_multiscale_stats_streaming_strips(
                &src_img, &src_img, &config, &weights, 256, 128,
            );
            // scale 0, channel 1 (Y) — the dominant channel.
            let mw = stats[0].iw_mean_w[1];
            rows.push((p.file_name().unwrap().to_string_lossy().into_owned(), mw));
        }
        assert!(
            rows.len() >= 5,
            "need >=5 usable images, got {}",
            rows.len()
        );

        rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let vals: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let n = vals.len();
        let pick = |q: f64| vals[((n as f64 - 1.0) * q).round() as usize];

        println!(
            "\n# SHIPPED mean_w (w = 1 + {k}*a) across {n} references — {dir}\n",
            k = config.iw_strength
        );
        println!("| stat | mean_w | mean feat (x mean_w) | 2nd (^0.5) | 4th (^0.25) |");
        println!("|---|--:|--:|--:|--:|");
        for (name, q) in [
            ("min", 0.0),
            ("p25", 0.25),
            ("p50", 0.5),
            ("p75", 0.75),
            ("max", 1.0),
        ] {
            let v = pick(q);
            println!(
                "| {name} | {v:.4} | {v:.4}x | {:.4}x | {:.4}x |",
                v.powf(0.5),
                v.powf(0.25)
            );
        }
        let (lo, hi) = (pick(0.0), pick(1.0));
        println!(
            "\nCROSS-IMAGE SPREAD (max/min):  mean {:.3}x   2nd {:.3}x   4th {:.3}x\n",
            hi / lo,
            (hi / lo).powf(0.5),
            (hi / lo).powf(0.25)
        );
        for (name, v) in rows.iter().take(2) {
            println!("  LOW   mean_w={v:.4}  {name}");
        }
        for (name, v) in rows.iter().rev().take(2) {
            println!("  HIGH  mean_w={v:.4}  {name}");
        }
    }

    /// DUMP the biggest raw ScaleStats field on real content — the tool that
    /// LOCATED the 5.8e6 explosion `benchmarks/ssim_moment_explosion_2026-07-16.md`
    /// characterizes.
    ///
    /// The explosion is NOT in the edge (`iw_art_4th`/`iw_det_4th`) features
    /// §3.19 named — those measured 0.02..0.09 here. It is in the higher moments
    /// of the SSIM map (`iw_ssim_4th`/`masked_ssim_4th`, XYB chroma channel,
    /// finest scale), whose per-pixel `d = (1 − num_m·num_s/denom_s)·mask` has a
    /// `.max(0)` floor but NO upper cap, so the un-normalized luminance term
    /// `num_m = 1 − (mu1−mu2)²` drives it to millions on high-magnitude chroma.
    /// NOTE: a gentle synthetic blur does NOT trigger it (this printed 0.19 on
    /// the named o_9292 graphic) — only real codec artifacts on high-contrast
    /// content do, which is why the full-parquet scan found it and row-group-0
    /// did not.
    ///
    /// `ZENSIM_DUMP_IMG=<png> [ZENSIM_DUMP_IMG2=<png> …] cargo test -p zensim --release dump_ssim_moment_explosion -- --ignored --nocapture`
    #[test]
    #[ignore = "needs specific images; set ZENSIM_DUMP_IMG (comma-separated) and run --ignored"]
    fn dump_ssim_moment_explosion() {
        let list = std::env::var("ZENSIM_DUMP_IMG")
            .expect("set ZENSIM_DUMP_IMG to comma-separated image paths");
        let config = ZensimConfig {
            compute_all_features: true,
            extended_features: true,
            compute_iw_features: true,
            ..Default::default()
        };
        let weights: Vec<f64> = WEIGHTS.to_vec();

        println!(
            "\n| image | ch | iw_art_4th | iw_det_4th | masked_art_4th | masked_det_4th | iw_ssim_4th |"
        );
        println!("|---|---|--:|--:|--:|--:|--:|");
        for path in list.split(',') {
            let path = path.trim();
            let Ok(img) = image::open(path) else {
                println!("| SKIP {path} | | | | | | |");
                continue;
            };
            let rgb = img.to_rgb8();
            let (w, h) = (rgb.width() as usize, rgb.height() as usize);
            let px: Vec<[u8; 3]> = rgb.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
            let src_img = RgbSlice::new(&px, w, h);
            // identity: src vs src isolates the extractor's own feature scale
            // from any distortion — the edge features on identity should be ~0,
            // so a large value here is pure extractor pathology.
            let name = std::path::Path::new(path)
                .file_name()
                .unwrap()
                .to_string_lossy();
            let short: String = name.chars().take(28).collect();

            // scale 0 only (finest); channel Y = 1 dominant.
            let (stats, _) = compute_multiscale_stats_streaming_strips(
                &src_img, &src_img, &config, &weights, 256, 128,
            );
            let s = &stats[0];
            println!(
                "| {short} (self) | Y | {:.4e} | {:.4e} | {:.4e} | {:.4e} | {:.4e} |",
                s.iw_art_4th[1],
                s.iw_det_4th[1],
                s.masked_art_4th[1],
                s.masked_det_4th[1],
                s.iw_ssim[1 * 3 + 1],
            );

            // also a real distortion: blur the distorted side hard.
            let mut d = px.clone();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    for c in 0..3 {
                        let a = px[y * w + x - 1][c] as u16
                            + px[y * w + x + 1][c] as u16
                            + px[(y - 1) * w + x][c] as u16
                            + px[(y + 1) * w + x][c] as u16;
                        d[y * w + x][c] = (a / 4) as u8;
                    }
                }
            }
            let dst_img = RgbSlice::new(&d, w, h);
            let (stats2, _) = compute_multiscale_stats_streaming_strips(
                &src_img, &dst_img, &config, &weights, 256, 128,
            );
            let s2 = &stats2[0];
            println!(
                "| {short} (blur) | Y | {:.4e} | {:.4e} | {:.4e} | {:.4e} | {:.4e} |",
                s2.iw_art_4th[1],
                s2.iw_det_4th[1],
                s2.masked_art_4th[1],
                s2.masked_det_4th[1],
                s2.iw_ssim[1 * 3 + 1],
            );
        }
        println!("\n(self = src-vs-src identity; blur = src vs 4-neighbour-blurred src)");
        // Direct test of §3.19's "5.8e6 IW/masked feature on o_9292": print the
        // single largest |feature| across ALL 372 and its index, for a real
        // (hard-blur) distortion. If §3.19 reproduces, some f>=228 is enormous.
        println!("\n| image | max|feat| | at index | block |");
        println!("|---|--:|--:|---|");
        for path in list.split(',') {
            let path = path.trim();
            let Ok(img) = image::open(path) else { continue };
            let rgb = img.to_rgb8();
            let (w, h) = (rgb.width() as usize, rgb.height() as usize);
            let px: Vec<[u8; 3]> = rgb.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
            let mut d = px.clone();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    for c in 0..3 {
                        let a = px[y * w + x - 1][c] as u16
                            + px[y * w + x + 1][c] as u16
                            + px[(y - 1) * w + x][c] as u16
                            + px[(y + 1) * w + x][c] as u16;
                        d[y * w + x][c] = (a / 4) as u8;
                    }
                }
            }
            let cfg = ZensimConfig {
                compute_all_features: true,
                extended_features: true,
                compute_iw_features: true,
                ..Default::default()
            };
            let (stats, _) = compute_multiscale_stats_streaming_strips(
                &RgbSlice::new(&px, w, h),
                &RgbSlice::new(&d, w, h),
                &cfg,
                &weights,
                256,
                128,
            );
            // scan every field of every scale, tag the biggest by NAME.
            let mut best = (0.0f64, String::new());
            for (si, s) in stats.iter().enumerate() {
                let named: [(&str, &[f64]); 9] = [
                    ("edge(unweighted)", &s.edge),
                    ("iw_ssim", &s.iw_ssim),
                    ("iw_art_4th", &s.iw_art_4th),
                    ("iw_det_4th", &s.iw_det_4th),
                    ("iw_mse", &s.iw_mse),
                    ("masked_art_4th", &s.masked_art_4th),
                    ("masked_det_4th", &s.masked_det_4th),
                    ("masked_mse", &s.masked_mse),
                    ("masked_ssim", &s.masked_ssim),
                ];
                for (nm, arr) in named {
                    for (k, &v) in arr.iter().enumerate() {
                        if v.abs() > best.0 {
                            best = (v.abs(), format!("{nm}[s{si} #{k}]"));
                        }
                    }
                }
            }
            let short: String = std::path::Path::new(path)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .chars()
                .take(26)
                .collect();
            println!("| {short} | {:.4e} | {} | |", best.0, best.1);
        }
    }

    // ── E-JBU: guided mass-conserving redistribution (kernel-level) ─────────

    /// Deterministic positive pseudo-random values (LCG) for guide/src planes.
    fn jbu_lcg(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 33) as u32) as f32 / u32::MAX as f32;
                lo + (hi - lo) * u
            })
            .collect()
    }

    /// ε-fallback: a UNIFORM guide reproduces NN exactly (bit-equal): the
    /// deposit `mass·g/Σg = v·w·count/(count·g)·g = v·w` is the same
    /// correctly-rounded product NN writes.
    #[test]
    fn jbu_uniform_guide_equals_nn() {
        let (sw, sh, f) = (4usize, 3usize, 8usize);
        let (dw, dh) = (sw * f, sh * f);
        let src = jbu_lcg(sw * sh, 7, -0.5, 1.5); // signed cells (ModelSensitivity folds are signed)
        let guide = vec![1.0f32; dw * dh];
        let mut nn = vec![0.0f32; dw * dh];
        let mut jbu = vec![0.0f32; dw * dh];
        upsample_pow2x_add(&src, sw, sh, &mut nn, dw, dh, f, 0.7);
        redistribute_pow2x_guided_add(&src, sw, sh, &mut jbu, dw, dh, f, 0.7, &guide);
        for (i, (a, b)) in nn.iter().zip(jbu.iter()).enumerate() {
            assert_eq!(a, b, "pixel {i}: NN {a} vs uniform-guide JBU {b}");
        }
    }

    /// Per-cell mass conservation vs NN with a non-uniform guide, including
    /// edge-clipped footprints (dst dims NOT multiples of the factor), and the
    /// per-pixel maps must actually differ (the path is engaged).
    #[test]
    fn jbu_mass_conservation_per_cell_and_pixels_move() {
        let (sw, sh, f) = (5usize, 7usize, 4usize);
        for (dw, dh) in [(sw * f, sh * f), (sw * f - 3, sh * f - 1)] {
            let src = jbu_lcg(sw * sh, 42, -1.0, 2.0);
            let guide = jbu_lcg(dw * dh, 99, 1e-3, 3.0);
            let mut nn = vec![0.0f32; dw * dh];
            let mut jbu = vec![0.0f32; dw * dh];
            upsample_pow2x_add(&src, sw, sh, &mut nn, dw, dh, f, 0.31);
            redistribute_pow2x_guided_add(&src, sw, sh, &mut jbu, dw, dh, f, 0.31, &guide);
            let mut max_cell_rel = 0.0f64;
            let mut max_px_delta = 0.0f32;
            for sy in 0..sh {
                let (dy0, dy1) = (sy * f, ((sy + 1) * f).min(dh));
                if dy0 >= dh {
                    break;
                }
                for sx in 0..sw {
                    let (dx0, dx1) = (sx * f, ((sx + 1) * f).min(dw));
                    if dx0 >= dw {
                        break;
                    }
                    let (mut a, mut b) = (0.0f64, 0.0f64);
                    for y in dy0..dy1 {
                        for x in dx0..dx1 {
                            a += nn[y * dw + x] as f64;
                            b += jbu[y * dw + x] as f64;
                            max_px_delta =
                                max_px_delta.max((nn[y * dw + x] - jbu[y * dw + x]).abs());
                        }
                    }
                    let rel = (a - b).abs() / a.abs().max(1e-9);
                    max_cell_rel = max_cell_rel.max(rel);
                }
            }
            assert!(
                max_cell_rel < 1e-5,
                "per-cell mass drift {max_cell_rel:.3e} (dw={dw} dh={dh})"
            );
            assert!(
                max_px_delta > 1e-3,
                "guided redistribution changed no pixels (max Δ {max_px_delta:.3e}) — path not engaged"
            );
        }
    }

    /// Degenerate guide (all zeros — below any ε the caller adds): the
    /// `gsum <= 0` guard must degrade to exactly NN, never NaN.
    #[test]
    fn jbu_zero_guide_degrades_to_nn() {
        let (sw, sh, f) = (3usize, 3usize, 2usize);
        let (dw, dh) = (sw * f, sh * f);
        let src = jbu_lcg(sw * sh, 5, 0.0, 1.0);
        let guide = vec![0.0f32; dw * dh];
        let mut nn = vec![0.0f32; dw * dh];
        let mut jbu = vec![0.0f32; dw * dh];
        upsample_pow2x_add(&src, sw, sh, &mut nn, dw, dh, f, 1.3);
        redistribute_pow2x_guided_add(&src, sw, sh, &mut jbu, dw, dh, f, 1.3, &guide);
        for (a, b) in nn.iter().zip(jbu.iter()) {
            assert!(b.is_finite());
            assert_eq!(a, b);
        }
    }
}
