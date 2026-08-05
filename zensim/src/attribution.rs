//! Attribution-density steering map (task #67, C1) — `custom-profiles` research API.
//!
//! Builds a per-pixel **attribution density** `D(x, y)` for a scalar model's
//! gradient `s_k = ∂score/∂f_k` over the BASIC feature block (f0-155: 13 slots
//! × 3 channels × 4 scales, scale-major `scale*39 + ch*13 + slot`), plus a
//! summed-area table for O(1) arbitrary-rectangle queries.
//!
//! # Why this exists (E-M6..E-M9, `benchmarks/coherent089_seeded_frontier_2026-07-27.md`)
//!
//! The per-pixel *signal fold* (`DiffmapWeighting::ModelSensitivity`) blends
//! scales by gradient mass and renormalizes per scale, so a model whose mass
//! sits on coarse-scale MSE degrades to a 1/8-resolution map — block ranking
//! (M3) collapses to 0.1-0.3 and inverts at 128 px, while the per-block
//! gradient attribution `Σ_k s_k·Δf_k(block)` (M2) stays 0.999-1.000 at every
//! block size. This module productizes the M2 mechanism: each feature's
//! per-pixel contribution is laid down with its **TRUE integrand and ABSOLUTE
//! normalization** (no per-scale renormalization, no mass blend — `s_k`
//! carries the weight), so a rectangle sum over any block is the first-order
//! prediction of the score gain from refining that block.
//!
//! # The contract
//!
//! [`AttributionResult::query_rect`] over block `B` approximates
//! `Σ_k s_k · Δf_k(B)` — the linearized scalar-score gain from re-encoding
//! `B` at reference quality — for ANY axis-aligned rectangle, at O(1) cost.
//! This is the codec-facing steering surface for variable partitions (AV1
//! 4-128 px recursive, JXL variable DCT, HEVC CTU splits): the codec queries
//! its own partition geometry directly instead of consuming a fixed-grid map.
//!
//! # Integrands (verified against `finalize()` / `fused_vblur_features_ssim`)
//!
//! Per (scale, channel), the 13 basic slots pool per-pixel signals as:
//!
//! | slot | feature | pooling | per-pixel signal `v` |
//! |---|---|---|---|
//! | 0/1/2 | ssim mean/4th/2nd | mean / p4 / p2 | `sd` (SSIM error) |
//! | 3/4/5 | art mean/4th/2nd | mean / p4 / p2 | `max(0, ed)`, `ed = (1+|d−μ2|)/(1+|s−μ1|) − 1` |
//! | 6/7/8 | det mean/4th/2nd | mean / p4 / p2 | `max(0, −ed)` |
//! | 9 | mse | mean | `(s − d)²` |
//! | 10 | hf_energy_loss | `max(0, 1 − varDst/varSrc)` | `(s−μ1)²`, `(d−μ2)²` |
//! | 11 | hf_mag_loss | `max(0, 1 − madDst/madSrc)` | `\|s−μ1\|`, `\|d−μ2\|` |
//! | 12 | hf_energy_gain | `max(0, varDst/varSrc − 1)` | same as 10 |
//!
//! - **Mean-pooled** (slots 0,3,6,9): `d_k(i) = v_i / N` — the pixel's exact
//!   share of the feature. Full-plane sum = the feature value exactly.
//! - **p-pooled** (slots 1,2,4,5,7,8; `f = (mean(v^p))^{1/p}`):
//!   `d_k(i) = (1/p) · (v_i^p / N) · M^{(1−p)/p}` with `M = mean(v^p)`.
//!   The `1/p` factor makes the block sum the **removal-consistent first
//!   order**: zeroing block `B`'s members changes `M` by `−ΔM_B` linearly, so
//!   `Δf ≈ −(1/p)·ΔM_B·M^{1/p−1}`. The raw value-weighted gradient
//!   `v_i·∂f/∂v_i = (v_i^p/N)·M^{(1−p)/p}` (Euler form, sums to `f`) is the
//!   `t→0` path derivative of *scaling* the block and over-weights p-pooled
//!   slots by ×p relative to their true block-removal effect — wrong relative
//!   weighting against mean slots, so the `1/p` form is used. Per-slot block
//!   RANKING is identical either way (same `M` for all blocks); only the
//!   cross-slot mix differs.
//! - **hf ratio slots** (10-12): exact first-order integrands of the clamped
//!   ratio, gated on the clamp state (`varSrc > 1e-10` matching `finalize`,
//!   loss XOR gain active). With `e_i = (s_i−μ1_i)² − (d_i−μ2_i)²`:
//!   slot 10 active ⇒ `d_10(i) = e_i / Σe_src²`; slot 12 active ⇒
//!   `d_12(i) = −e_i / Σe_src²`; slot 11 active ⇒
//!   `d_11(i) = (|s_i−μ1_i| − |d_i−μ2_i|) / Σ|s−μ1|`. These are SIGNED —
//!   e.g. a ringing block inside a globally-blurred image genuinely
//!   *increases* `hf_energy_loss` when refined, and the density carries that.
//!   Full-plane sums equal the (unclamped-side) feature values exactly.
//!
//! Sign convention matches `ModelSensitivity`: the map weight is `−s_k`
//! (score-oriented models have negative sensitivity on error signals →
//! positive map weight = "refining here gains score"). The map is **signed**.
//!
//! Coarse scales are upsampled **sum-preservingly**: a scale-`s` pixel's
//! density spreads uniformly over its `2^s × 2^s` full-resolution footprint
//! divided by the footprint area (nearest-neighbor ÷ area, NOT bilinear), so
//! rectangle sums reproduce coarse-plane sums exactly.
//!
//! # Honest approximations / blind spots
//!
//! 1. **f156-371 (peak/masked/iw) and any block beyond are NOT spatialized**
//!    — same structural blind spot as the signal fold. The harness prints the
//!    dropped `|s_k|` mass per bake.
//! 2. **Blur bleed** (C2b MEASURED): refining block `B` also changes signals
//!    within the blur radius outside `B`. The pure-window-supported signals
//!    (ssim `d`; v2 contrast/texture) ARE spread over their blur window via
//!    the sum-preserving box spread (`blur::box_spread_sum_preserving`,
//!    clipped-window per-source-normalized convention) — measured NEUTRAL on
//!    the 8-cell gate (±0.003). Residual-form signals (art/det/hf/mscn) stay
//!    pixel-allocated: the 50/50 pixel/window split was measured and
//!    REGRESSED all 8 cells (−0.01..−0.08), and the pure `I − K` adjoint
//!    allocates zero net mass (wrong for removal semantics). The remaining
//!    fine-block residual is the finite-removal floor, not an allocation
//!    fix — see `benchmarks/attribution_map_c1_2026-07-29.md` §C2b.
//! 3. **SIMD-padding columns** (padded width − width) carry feature mass that
//!    the trimmed map cannot attribute (≤ ~3 % of columns; near-zero signal
//!    since both planes zero-pad identically).
//! 4. **p-root curvature** beyond first order for large removals, and **hf
//!    clamp state** assumed fixed under refinement.
//! 5. The internal V-blur banding replicates the production 32-row band
//!    layout, so per-pixel signals and pooled scalars are bit-compatible with
//!    the production feature extractor (`sum-preservation` tests below).

use crate::ZensimError;
use crate::blur::fused_blur_h_ssim;
use crate::fused::{StripChannelAccum, fused_vblur_features_ssim};
use crate::metric::{
    FEATURES_PER_CHANNEL_BASIC, MIN_PYRAMID_DIM, check_within_max_pixels, config_from_params,
    reflect_pad_to_min, validate_pair, validate_ref_match,
};
use crate::source::ImageSource;
use crate::streaming::{
    MultiScaleRef, PrecomputedReference, convert_source_to_xyb, downscale_3_planes,
};
use archmage::autoversion;

/// Production band height for the V-blur running-sum chains (must equal
/// `streaming::STRIP_INNER` so the per-band accumulator init points — and
/// therefore every per-pixel signal and pooled scalar — are bit-compatible
/// with the production feature extractor).
const BAND_ROWS: usize = 32;

// ── Feature-layout block boundaries (half-open ends) ─────────────────────
//
// These name the `s`-slice bounds that
// [`crate::Zensim::compute_attribution_density_full`] cuts the model
// gradient on. They exist because the append2 block (`f924-943`) was
// silently dropped for ten months by a hard-coded `min(len, 924)` bound —
// found 2026-08-04 by the coherence study, and the reason the block was
// classified "non-decomposable" when 8 of its 20 slots are exactly
// decomposable. A named bound plus the per-width coverage test
// (`attribution_covers_expected_slots_per_width`) is the anti-recurrence
// guard: adding a block means adding a constant AND a row to that test.

#[cfg(feature = "feature-regime-v2")]
mod layout_ends {
    /// End of the basic block (`f0-155`) = start of the v1 pooled block.
    /// Not referenced by the slicing itself (the basic path bounds-checks its
    /// own lookups), but it completes the layout description and is exercised
    /// by `attribution_covers_expected_slots_per_width`.
    #[allow(dead_code)]
    pub(crate) const BLOCK_END_BASIC: usize = 156;
    /// End of the v1 peak/masked/IW pooled block (`f156-371`) = start of v2.
    /// Structurally NOT spatialized (module blind spot 1).
    pub(crate) const BLOCK_END_V1_POOLS: usize = 372;
    /// End of the v2 block (`f372-719`) = start of the append block.
    pub(crate) const BLOCK_END_V2: usize = 720;
    /// End of the append block (`f720-923`) = start of append2.
    pub(crate) const BLOCK_END_APPEND: usize = 924;
    /// End of the append2 block (`f924-943`). Anything beyond (the f944+ CSFW
    /// block) has no attribution integrand yet and is deliberately not sliced —
    /// the coverage test pins that too.
    pub(crate) const BLOCK_END_APPEND2: usize = 944;
}
#[cfg(feature = "feature-regime-v2")]
use layout_ends::*;

/// Per-pixel attribution density + summed-area table.
///
/// Produced by [`crate::Zensim::compute_attribution_density`]. See the
/// [module docs](self) for the mechanism and the steering contract.
///
/// The stored density is the `f32` view (visualization / export); the SAT is
/// built from the internal `f64` accumulation *before* `f32` rounding, so
/// [`query_rect`](Self::query_rect) sums are `f64`-accurate. A result built
/// via [`from_density`](Self::from_density) instead derives the SAT from the
/// given `f32` values.
#[non_exhaustive]
pub struct AttributionResult {
    density: Vec<f32>,
    /// Summed-area table, `(width+1) × (height+1)`, `sat[y*(w+1)+x]` =
    /// Σ density over `[0,x) × [0,y)`.
    sat: Vec<f64>,
    width: usize,
    height: usize,
}

impl AttributionResult {
    /// Build from an externally assembled density plane (row-major
    /// `width × height`), e.g. the basic-block density plus a v2 fold map.
    /// The SAT is derived from the given `f32` values.
    ///
    /// # Panics
    ///
    /// Panics if `density.len() != width * height`.
    pub fn from_density(density: Vec<f32>, width: usize, height: usize) -> Self {
        Self::from_density_with_sat_scratch(density, width, height, Vec::new())
    }

    /// Internal `from_density` variant reusing a caller-provided SAT
    /// buffer's capacity (the stale single-pass session's recycle path —
    /// avoids a fresh multi-MB allocation + page-fault storm per codec
    /// iteration). An empty `sat_scratch` reproduces
    /// [`from_density`](Self::from_density) exactly.
    fn from_density_with_sat_scratch(
        density: Vec<f32>,
        width: usize,
        height: usize,
        mut sat_scratch: Vec<f64>,
    ) -> Self {
        assert_eq!(density.len(), width * height, "density len != width*height");
        build_sat_into(|i| density[i] as f64, width, height, &mut sat_scratch);
        Self {
            density,
            sat: sat_scratch,
            width,
            height,
        }
    }

    /// Internal: build from the f64 accumulation canvas (SAT keeps f64 truth).
    fn from_f64_canvas(canvas: Vec<f64>, width: usize, height: usize) -> Self {
        debug_assert_eq!(canvas.len(), width * height);
        let sat = build_sat(|i| canvas[i], width, height);
        let density = canvas.iter().map(|&v| v as f32).collect();
        Self {
            density,
            sat,
            width,
            height,
        }
    }

    /// The signed per-pixel density, row-major `width × height`.
    /// Positive = refining here is predicted to raise the scalar score.
    pub fn density(&self) -> &[f32] {
        &self.density
    }

    /// Image width in pixels (actual, not SIMD-padded).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Sum of the density over the half-open rectangle `[x0, x1) × [y0, y1)`,
    /// in O(1) via the summed-area table. Coordinates are clamped to the
    /// image; empty or inverted rectangles return `0.0`.
    ///
    /// For a codec partition block `B`, this is the first-order prediction of
    /// the scalar-score gain from re-encoding `B` at reference quality (see
    /// the [module docs](self) for the exact semantics and approximations).
    pub fn query_rect(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let x1 = x1.min(self.width);
        let y1 = y1.min(self.height);
        let x0 = x0.min(x1);
        let y0 = y0.min(y1);
        let w1 = self.width + 1;
        self.sat[y1 * w1 + x1] - self.sat[y0 * w1 + x1] - self.sat[y1 * w1 + x0]
            + self.sat[y0 * w1 + x0]
    }

    /// Per-block density sums on a fixed `block × block` grid (row-major,
    /// `ceil(w/block) × ceil(h/block)`, edge blocks clipped) — the same
    /// layout as the `diffmap_block_coherence` harness. Convenience over
    /// [`query_rect`](Self::query_rect); arbitrary partitions should query
    /// their own rectangles.
    ///
    /// # Panics
    ///
    /// Panics if `block == 0`.
    pub fn block_sums(&self, block: usize) -> Vec<f64> {
        assert!(block > 0, "block size must be non-zero");
        let bx = self.width.div_ceil(block);
        let by = self.height.div_ceil(block);
        let mut out = Vec::with_capacity(bx * by);
        for byi in 0..by {
            for bxi in 0..bx {
                let x0 = bxi * block;
                let y0 = byi * block;
                out.push(self.query_rect(x0, y0, x0 + block, y0 + block));
            }
        }
        out
    }
}

/// Standard SAT recurrence from an index-addressed f64 source.
fn build_sat(get: impl Fn(usize) -> f64, width: usize, height: usize) -> Vec<f64> {
    let mut sat = Vec::new();
    build_sat_into(get, width, height, &mut sat);
    sat
}

/// [`build_sat`] into a reusable buffer: only the guard row and guard
/// column are re-zeroed — every other element is written by the
/// recurrence before any read, so recycled contents are never observable.
/// Bitwise-identical to a fresh build.
fn build_sat_into(get: impl Fn(usize) -> f64, width: usize, height: usize, sat: &mut Vec<f64>) {
    let w1 = width + 1;
    let len = w1 * (height + 1);
    if sat.len() != len {
        sat.clear();
        sat.resize(len, 0.0);
    } else {
        sat[..w1].fill(0.0);
        for y in 1..=height {
            sat[y * w1] = 0.0;
        }
    }
    for y in 0..height {
        let row = y * width;
        let sat_prev = y * w1;
        let sat_row = (y + 1) * w1;
        let mut run = 0.0f64;
        for x in 0..width {
            run += get(row + x);
            sat[sat_row + x + 1] = run + sat[sat_prev + x + 1];
        }
    }
}

/// Per-(scale, channel) combine coefficients — everything deferred until the
/// channel's pooled sums are known. Each field multiplies the per-pixel term
/// shown in the module-docs table; the `−s_k` sign fold and `1/N` are baked in.
#[derive(Clone, Copy, Default)]
struct SlotCoeffs {
    c_sd: f64,
    c_sd4: f64,
    c_sd2: f64,
    c_art: f64,
    c_art4: f64,
    c_art2: f64,
    c_det: f64,
    c_det4: f64,
    c_det2: f64,
    c_mse: f64,
    /// Coefficient on the SIGNED plane `e = (s−μ1)² − (d−μ2)²` (slots 10/12,
    /// mutually exclusive by the global clamp).
    c_hfe: f64,
    /// Coefficient on the SIGNED plane `|s−μ1| − |d−μ2|` (slot 11).
    c_hfm: f64,
}

impl SlotCoeffs {
    /// Derive coefficients from the model gradient `s` (basic layout, global
    /// index `base_k + slot`), the channel's raw pooled sums, and the scale's
    /// pixel count `n`.
    fn derive(s: &[f64], base_k: usize, acc: &StripChannelAccum, n: f64) -> Self {
        // Gradient lookup: 0 beyond the slice (short/basic-only s vectors).
        let g = |slot: usize| s.get(base_k + slot).copied().unwrap_or(0.0);
        let inv_n = 1.0 / n;
        // p-pooled deferred scalar: (1/p) · M^{(1−p)/p} / N, zero-guarded like
        // `finalize` (M = 0 ⇒ feature 0 ⇒ no gradient; x^{1/p} has an infinite
        // slope at 0 which the removal-consistent form never evaluates).
        let p_coeff = |sk: f64, sum_pow: f64, p: f64| -> f64 {
            let m = sum_pow * inv_n;
            if m > 0.0 {
                -sk * inv_n * (1.0 / p) * m.powf((1.0 - p) / p)
            } else {
                0.0
            }
        };
        let var_src = acc.hf_sq_src * inv_n;
        let var_dst = acc.hf_sq_dst * inv_n;
        let mad_src = acc.hf_abs_src * inv_n;
        let mad_dst = acc.hf_abs_dst * inv_n;
        // hf energy: slot 10 (loss) active when varDst < varSrc, slot 12
        // (gain) when varDst > varSrc — never both (`finalize` clamps the
        // other side to 0). d10 = e/Σsrc², d12 = −e/Σsrc²; fold −s_k.
        let c_hfe = if var_src > 1e-10 {
            if var_dst < var_src {
                -g(10) / acc.hf_sq_src
            } else if var_dst > var_src {
                g(12) / acc.hf_sq_src
            } else {
                0.0
            }
        } else {
            0.0
        };
        let c_hfm = if mad_src > 1e-10 && mad_dst < mad_src {
            -g(11) / acc.hf_abs_src
        } else {
            0.0
        };
        Self {
            c_sd: -g(0) * inv_n,
            c_sd4: p_coeff(g(1), acc.ssim_d4, 4.0),
            c_sd2: p_coeff(g(2), acc.ssim_d2, 2.0),
            c_art: -g(3) * inv_n,
            c_art4: p_coeff(g(4), acc.edge_art4, 4.0),
            c_art2: p_coeff(g(5), acc.edge_art2, 2.0),
            c_det: -g(6) * inv_n,
            c_det4: p_coeff(g(7), acc.edge_det4, 4.0),
            c_det2: p_coeff(g(8), acc.edge_det2, 2.0),
            c_mse: -g(9) * inv_n,
            c_hfe,
            c_hfm,
        }
    }
}

/// Mirror of `ScaleAccumulators::finalize` for the 13 basic slots of one
/// channel — used by the sum-preservation tests to compare the density's
/// full-plane sums against the exact feature values its planes pooled to.
fn basic13_from_acc(a: &StripChannelAccum, n: f64) -> [f64; 13] {
    let inv = 1.0 / n;
    let var_src = a.hf_sq_src * inv;
    let var_dst = a.hf_sq_dst * inv;
    let mad_src = a.hf_abs_src * inv;
    let mad_dst = a.hf_abs_dst * inv;
    [
        a.ssim_d * inv,
        (a.ssim_d4 * inv).max(0.0).powf(0.25),
        (a.ssim_d2 * inv).max(0.0).sqrt(),
        a.edge_art * inv,
        (a.edge_art4 * inv).max(0.0).powf(0.25),
        (a.edge_art2 * inv).max(0.0).sqrt(),
        a.edge_det * inv,
        (a.edge_det4 * inv).max(0.0).powf(0.25),
        (a.edge_det2 * inv).max(0.0).sqrt(),
        a.mse * inv,
        if var_src > 1e-10 {
            (1.0 - var_dst / var_src).max(0.0)
        } else {
            0.0
        },
        if mad_src > 1e-10 {
            (1.0 - mad_dst / mad_src).max(0.0)
        } else {
            0.0
        },
        if var_src > 1e-10 {
            (var_dst / var_src - 1.0).max(0.0)
        } else {
            0.0
        },
    ]
}

/// Field-wise merge for per-band accumulators (the fused kernel returns one
/// per band; sums add, maxes max). Band-index order matches the production
/// sequential merge, keeping f64 addition order identical.
fn merge_acc(a: &mut StripChannelAccum, b: &StripChannelAccum) {
    a.ssim_d += b.ssim_d;
    a.ssim_d4 += b.ssim_d4;
    a.ssim_d2 += b.ssim_d2;
    a.edge_art += b.edge_art;
    a.edge_art4 += b.edge_art4;
    a.edge_art2 += b.edge_art2;
    a.edge_det += b.edge_det;
    a.edge_det4 += b.edge_det4;
    a.edge_det2 += b.edge_det2;
    a.mse += b.mse;
    a.hf_sq_src += b.hf_sq_src;
    a.hf_sq_dst += b.hf_sq_dst;
    a.hf_abs_src += b.hf_abs_src;
    a.hf_abs_dst += b.hf_abs_dst;
    a.ssim_d8 += b.ssim_d8;
    a.edge_art8 += b.edge_art8;
    a.edge_det8 += b.edge_det8;
    a.ssim_max = a.ssim_max.max(b.ssim_max);
    a.edge_art_max = a.edge_art_max.max(b.edge_art_max);
    a.edge_det_max = a.edge_det_max.max(b.edge_det_max);
}

/// Sum-preserving power-of-2 upsample-add: each scale pixel's value spreads
/// uniformly over its `factor × factor` footprint ÷ area. With floor-halved
/// pyramid dims the footprint never exceeds the canvas, so rectangle sums
/// over the canvas reproduce coarse-plane sums exactly (÷ by powers of two
/// is exact in f64); the `min` clamps are defensive only.
fn upsample_add_sum_preserving(
    scale_plane: &[f64],
    sw: usize,
    sh: usize,
    canvas: &mut [f64],
    cw: usize,
    ch: usize,
    factor: usize,
) {
    if factor == 1 {
        debug_assert_eq!(sw, cw);
        for (c, &v) in canvas.iter_mut().zip(scale_plane.iter()) {
            *c += v;
        }
        return;
    }
    let inv_area = 1.0 / ((factor * factor) as f64);
    for sy in 0..sh {
        let y0 = sy * factor;
        let y1 = (y0 + factor).min(ch);
        let src_row = sy * sw;
        for sx in 0..sw {
            let v = scale_plane[src_row + sx] * inv_area;
            let x0 = sx * factor;
            let x1 = (x0 + factor).min(cw);
            for row in canvas[y0 * cw..].chunks_mut(cw).take(y1.saturating_sub(y0)) {
                for slot in &mut row[x0..x1] {
                    *slot += v;
                }
            }
        }
    }
}

/// One channel of one scale, banded exactly like the production pipeline:
/// whole-plane fused H-blur, then per-32-row-band fused V-blur + feature
/// accumulation with `±blur_radius` overlap slices. Fills `sd`/`mu1`/`mu2`
/// (full scale planes) and returns the merged accumulator.
#[allow(clippy::too_many_arguments)]
fn process_channel_banded(
    src_c: &[f32],
    dst_c: &[f32],
    w: usize,
    h: usize,
    radius: usize,
    bufs: &mut ChannelBuffers,
) -> StripChannelAccum {
    let n = w * h;
    fused_blur_h_ssim(
        &src_c[..n],
        &dst_c[..n],
        &mut bufs.h_mu1[..n],
        &mut bufs.h_mu2[..n],
        &mut bufs.h_ssq[..n],
        &mut bufs.h_s12[..n],
        w,
        h,
        radius,
    );

    let mut acc = StripChannelAccum::zero();
    let overlap = radius; // blur_passes == 1 (validated by the caller)
    let mut y = 0usize;
    while y < h {
        let inner_end = (y + BAND_ROWS).min(h);
        let top = y.saturating_sub(overlap);
        let bot = (inner_end + overlap).min(h);
        let off = top * w;
        let band_n = (bot - top) * w;
        let band = fused_vblur_features_ssim(
            &bufs.h_mu1[off..off + band_n],
            &bufs.h_mu2[off..off + band_n],
            &bufs.h_ssq[off..off + band_n],
            &bufs.h_s12[off..off + band_n],
            &src_c[off..off + band_n],
            &dst_c[off..off + band_n],
            w,
            bot - top,
            y - top,
            inner_end - y,
            radius,
            &mut bufs.mu1[off..off + band_n],
            &mut bufs.mu2[off..off + band_n],
            true,
            &mut bufs.sd[off..off + band_n],
            true,
        );
        merge_acc(&mut acc, &band);
        y = inner_end;
    }
    acc
}

/// Reusable per-scale scratch (sized once at scale-0 dims, sliced per scale).
struct ChannelBuffers {
    h_mu1: Vec<f32>,
    h_mu2: Vec<f32>,
    h_ssq: Vec<f32>,
    h_s12: Vec<f32>,
    mu1: Vec<f32>,
    mu2: Vec<f32>,
    sd: Vec<f32>,
}

impl ChannelBuffers {
    fn new(n: usize) -> Self {
        Self {
            h_mu1: vec![0.0; n],
            h_mu2: vec![0.0; n],
            h_ssq: vec![0.0; n],
            h_s12: vec![0.0; n],
            mu1: vec![0.0; n],
            mu2: vec![0.0; n],
            sd: vec![0.0; n],
        }
    }
}

/// Exact-integrand combine (f64) for one channel: all seven signals
/// re-derived from the SAME planes the pooled features consumed. Mass is
/// routed by SUPPORT class (C2b bleed allocation, MEASURED variant:
/// window-only spread):
///  - ssim terms → the blur-window plane (per-pixel value is a pure
///    function of K-blurred planes);
///  - art/det/hf terms → the pixel plane (residual form `d_i − (K∗d)_i`;
///    the 50/50 pixel/window split was measured and REGRESSED all 8 gate
///    cells, and the pure `I − K` adjoint allocates zero net mass — both
///    recorded in the C2b benchmark section);
///  - mse → the pixel plane (no blur in its signal).
fn basic_combine_channel(
    bufs: &ChannelBuffers,
    src_c: &[f32],
    dst_c: &[f32],
    n: usize,
    co: &SlotCoeffs,
    id_slice: &mut [f64],
    bl_slice: &mut [f64],
) {
    for i in 0..n {
        let sd = bufs.sd[i] as f64;
        let sv = src_c[i] as f64;
        let dv = dst_c[i] as f64;
        let m1 = bufs.mu1[i] as f64;
        let m2 = bufs.mu2[i] as f64;
        let d1 = (sv - m1).abs();
        let d2 = (dv - m2).abs();
        let ed = (1.0 + d2) / (1.0 + d1) - 1.0;
        let art = ed.max(0.0);
        let det = (-ed).max(0.0);
        let sd2 = sd * sd;
        let a2 = art * art;
        let dt2 = det * det;
        let pd = sv - dv;
        let e_hf = d1 * d1 - d2 * d2;
        let win_term = co.c_sd * sd + co.c_sd4 * (sd2 * sd2) + co.c_sd2 * sd2;
        let res_term = co.c_art * art
            + co.c_art4 * (a2 * a2)
            + co.c_art2 * a2
            + co.c_det * det
            + co.c_det4 * (dt2 * dt2)
            + co.c_det2 * dt2
            + co.c_hfe * e_hf
            + co.c_hfm * (d1 - d2);
        let px_term = co.c_mse * (pd * pd);
        id_slice[i] += px_term + res_term;
        bl_slice[i] += win_term;
    }
}

/// Core builder: multi-scale pyramid walk over (reference pyramid, distorted
/// XYB planes), exact-integrand combine per (scale, channel), sum-preserving
/// upsample into a full-resolution f64 canvas at padded-compute dims.
///
/// Returns `(canvas, basic_features)` where `basic_features` mirrors the
/// production basic block (13 × 3 × scales, scale-major) — consumed by the
/// sum-preservation tests.
fn build_attribution_canvas(
    pyramid: &PrecomputedReference,
    mut dst_planes: [Vec<f32>; 3],
    comp_pw: usize,
    comp_h: usize,
    num_scales: usize,
    radius: usize,
    parallel: bool,
    s: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    const FPC: usize = FEATURES_PER_CHANNEL_BASIC;
    let mut canvas = vec![0.0f64; comp_pw * comp_h];
    let mut own_features = Vec::with_capacity(num_scales * FPC * 3);
    let n0 = comp_pw * comp_h;
    let mut bufs3: [ChannelBuffers; 3] = [
        ChannelBuffers::new(n0),
        ChannelBuffers::new(n0),
        ChannelBuffers::new(n0),
    ];
    let mut chan_id: [Vec<f64>; 3] = [vec![0.0; n0], vec![0.0; n0], vec![0.0; n0]];
    let mut chan_win: [Vec<f64>; 3] = [vec![0.0; n0], vec![0.0; n0], vec![0.0; n0]];
    let mut scale_density = vec![0.0f64; comp_pw * comp_h];
    let mut spread_plane = vec![0.0f64; comp_pw * comp_h];
    let mut spread_tmp: Vec<f64> = Vec::new();

    let mut w = comp_pw;
    let mut h = comp_h;
    for scale in 0..num_scales {
        if w < 8 || h < 8 {
            break;
        }
        let (src_planes, sw, sh) = pyramid.scale(scale);
        assert_eq!(w, sw, "attribution: width mismatch at scale {scale}");
        assert_eq!(h, sh, "attribution: height mismatch at scale {scale}");
        let n = sw * sh;
        let n_f = n as f64;
        scale_density[..n].fill(0.0);
        spread_plane[..n].fill(0.0);

        // Per-channel work is independent (own blur buffers, own output
        // planes) — channel-parallel under `threads` (C2b Part 2).
        let mut accs: [Option<StripChannelAccum>; 3] = [None, None, None];
        {
            let [b0, b1, b2] = &mut bufs3;
            let [i0, i1, i2] = &mut chan_id;
            let [w0p, w1p, w2p] = &mut chan_win;
            let run = |c: usize,
                       buf: &mut ChannelBuffers,
                       idp: &mut Vec<f64>,
                       winp: &mut Vec<f64>|
             -> StripChannelAccum {
                idp[..n].fill(0.0);
                winp[..n].fill(0.0);
                let src_c = &src_planes[c][..n];
                let dst_c = &dst_planes[c][..n];
                let acc = process_channel_banded(src_c, dst_c, sw, sh, radius, buf);
                let base_k = scale * FPC * 3 + c * FPC;
                let co = SlotCoeffs::derive(s, base_k, &acc, n as f64);
                basic_combine_channel(buf, src_c, dst_c, n, &co, &mut idp[..n], &mut winp[..n]);
                acc
            };
            #[cfg(feature = "threads")]
            if parallel {
                let ((a0, (a1, a2)), ()) = rayon::join(
                    || {
                        rayon::join(
                            || run(0, b0, i0, w0p),
                            || rayon::join(|| run(1, b1, i1, w1p), || run(2, b2, i2, w2p)),
                        )
                    },
                    || (),
                );
                accs = [Some(a0), Some(a1), Some(a2)];
            }
            if accs[0].is_none() {
                accs = [
                    Some(run(0, b0, i0, w0p)),
                    Some(run(1, b1, i1, w1p)),
                    Some(run(2, b2, i2, w2p)),
                ];
            }
        }
        for acc in accs.iter().flatten() {
            own_features.extend_from_slice(&basic13_from_acc(acc, n_f));
        }
        for c in 0..3 {
            for i in 0..n {
                scale_density[i] += chan_id[c][i];
                spread_plane[i] += chan_win[c][i];
            }
        }

        // One sum-preserving spread per scale for the window-supported mass,
        // then fold into the pixel plane.
        crate::blur::box_spread_sum_preserving(
            &mut spread_plane[..n],
            sw,
            sh,
            radius,
            &mut spread_tmp,
        );
        for (d, s) in scale_density[..n].iter_mut().zip(spread_plane[..n].iter()) {
            *d += *s;
        }

        upsample_add_sum_preserving(
            &scale_density[..n],
            sw,
            sh,
            &mut canvas,
            comp_pw,
            comp_h,
            1usize << scale,
        );

        if scale + 1 < num_scales {
            let (nw, nh) = downscale_3_planes(&mut dst_planes, w, h, parallel);
            w = nw;
            h = nh;
        }
    }
    (canvas, own_features)
}

impl crate::metric::Zensim {
    /// Attribution-density steering map for a scalar model's gradient
    /// (`custom-profiles` research API — see the [module docs](self) for the
    /// mechanism, the exact per-slot integrands, and the honest
    /// approximations list).
    ///
    /// `s` is the model's sensitivity `∂score/∂f_k` over the BASIC feature
    /// block, laid out scale-major (`scale*39 + ch*13 + slot`, i.e. `f0-155`
    /// of the production vector; entries beyond index 155 are ignored, so
    /// passing a full 372/720-wide gradient slice is safe). Gradients are the
    /// caller's job (e.g. central differences through the bake runtime, as in
    /// `examples/diffmap_block_coherence.rs`) — no gradient computation
    /// happens here.
    ///
    /// The result's [`query_rect`](AttributionResult::query_rect) over a
    /// block approximates the first-order score gain from refining that
    /// block; [`block_sums`](AttributionResult::block_sums) gives the fixed
    /// grid the coherence harness scores as `M3a`.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] for zero-dimension inputs,
    /// [`ZensimError::DimensionMismatch`] / [`ZensimError::ImageTooLarge`]
    /// per the usual pair validations, and
    /// [`ZensimError::ModelForwardFailed`] if the active profile uses
    /// `blur_passes != 1` (the banded fused kernels this density reuses are
    /// single-pass; every shipped profile is).
    pub fn compute_attribution_density(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        s: &[f64],
    ) -> Result<AttributionResult, ZensimError> {
        validate_pair(source, distorted)?;
        let precomputed = self.precompute_reference(source)?;
        self.compute_attribution_density_with_ref(&precomputed, distorted, s)
    }

    /// [`compute_attribution_density`](Self::compute_attribution_density)
    /// against a precomputed reference — the encoder-loop form (precompute
    /// once, re-query per candidate encode).
    ///
    /// # Errors
    ///
    /// Same as [`compute_attribution_density`](Self::compute_attribution_density).
    pub fn compute_attribution_density_with_ref(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
    ) -> Result<AttributionResult, ZensimError> {
        let (canvas, width, height) = self.basic_canvas_trimmed(precomputed, distorted, s)?;
        Ok(AttributionResult::from_f64_canvas(canvas, width, height))
    }

    /// FULL-coverage attribution density (task #67 C2a): the BASIC block
    /// density plus exact-integrand densities for the v2 (`f372-719`),
    /// append (`f720-923`) and append2 (`f924-943`) blocks — each included
    /// when `s` extends that far — built by
    /// [`crate::feature_v2::compute_v2_append_attribution`]'s replication of
    /// the production kernels. `s` is the raw full-layout gradient
    /// (`∂score/∂f_k`; 372-, 720-, 924- or 944-wide). The `f156-371`
    /// peak/masked/iw block is not spatializable and is ignored, as in the
    /// harness's structural-zero handling.
    ///
    /// Block bounds are the named [`BLOCK_END_*`](BLOCK_END_BASIC) constants
    /// so a future regime bump cannot silently drop a block the way the
    /// hard-coded `924` bound dropped append2 (found 2026-08-04; campaign
    /// appendix E). `attribution_covers_expected_slots_per_width` is the
    /// guard that fails if the covered slot set ever drifts.
    ///
    /// Remaining documented approximations: first-order integrands
    /// throughout, blur bleed unmodeled, finalize clamps treated as inert,
    /// blockiness steps split 50/50 across their pixel pair, reference-only
    /// slots (fragility, grad-src-mean, append2 luma-mean-ref) exactly zero,
    /// and the append2 HDR highlight bins structurally zero on this SDR
    /// route. See the module docs here and the `feature_v2` attribution
    /// section.
    ///
    /// # Errors
    ///
    /// Same as [`compute_attribution_density`](Self::compute_attribution_density).
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_attribution_density_full(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        s: &[f64],
    ) -> Result<AttributionResult, ZensimError> {
        validate_pair(source, distorted)?;
        let precomputed = self.precompute_reference(source)?;
        let (mut canvas, width, height) = self.basic_canvas_trimmed(&precomputed, distorted, s)?;
        // Half-open [start, end) slice of `s` for one block, empty when `s`
        // does not reach the block. One helper for all three so a new block
        // is one more line, not one more chance to mistype a bound.
        let block = |start: usize, end: usize| -> Option<&[f64]> {
            (s.len() > start).then(|| &s[start..s.len().min(end)])
        };
        // NB: the v2 slice starts at the END of the v1 pooled block (372),
        // not at the end of the basic block — `f156-371` is skipped, not
        // spatialized.
        let s_v2: &[f64] = block(BLOCK_END_V1_POOLS, BLOCK_END_V2).unwrap_or(&[]);
        let s_append: Option<&[f64]> = block(BLOCK_END_V2, BLOCK_END_APPEND);
        let s_append2: Option<&[f64]> = block(BLOCK_END_APPEND, BLOCK_END_APPEND2);
        if !s_v2.is_empty() || s_append.is_some() || s_append2.is_some() {
            let v2a = crate::feature_v2::compute_v2_append_attribution(
                source,
                distorted,
                s_v2,
                s_append,
                s_append2,
                self.max_pixels(),
                self.parallel(),
            )?;
            debug_assert_eq!((v2a.width, v2a.height), (width, height));
            for (c, v) in canvas.iter_mut().zip(v2a.density.iter()) {
                *c += *v;
            }
        }
        Ok(AttributionResult::from_f64_canvas(canvas, width, height))
    }

    /// Shared basic-block canvas builder (f64, trimmed to the logical
    /// image): the C1 path and the full-coverage path both start here.
    fn basic_canvas_trimmed(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
    ) -> Result<(Vec<f64>, usize, usize), ZensimError> {
        let params = self.profile().params();
        if distorted.width() == 0 || distorted.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels())?;
        let config = config_from_params(params, self.parallel());
        if config.blur_passes != 1 {
            return Err(ZensimError::ModelForwardFailed {
                reason: "attribution density requires blur_passes == 1 (all shipped profiles)",
            });
        }

        let width = distorted.width();
        let height = distorted.height();
        // Compute dims come from the (possibly reflect-padded) reference
        // pyramid; a sub-64px distorted is reflect-padded to match, and the
        // original image stays in the top-left for the trim below.
        let (comp_pw, comp_h) = (precomputed.scales[0].1, precomputed.scales[0].2);
        let dst_planes = if width < MIN_PYRAMID_DIM || height < MIN_PYRAMID_DIM {
            let padded = reflect_pad_to_min(distorted);
            convert_source_to_xyb(&padded, comp_pw, self.parallel())
        } else {
            convert_source_to_xyb(distorted, comp_pw, self.parallel())
        };

        let num_scales = config.num_scales.min(precomputed.scales.len());
        let (canvas, _own_features) = build_attribution_canvas(
            precomputed,
            dst_planes,
            comp_pw,
            comp_h,
            num_scales,
            config.blur_radius,
            self.parallel(),
            s,
        );

        // Trim the padded-compute canvas to the logical image (top-left).
        let trimmed = if comp_pw == width && comp_h == height {
            canvas
        } else {
            let mut out = Vec::with_capacity(width * height);
            for y in 0..height {
                out.extend_from_slice(&canvas[y * comp_pw..y * comp_pw + width]);
            }
            out
        };
        Ok((trimmed, width, height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::ProfileParams;
    use crate::{RgbSlice, Zensim, ZensimProfile};
    use std::sync::OnceLock;

    /// Extended-features test profile (all channels/scales active, default
    /// blur/scale config — the same knobs every shipped profile uses).
    fn test_zensim() -> Zensim {
        static PARAMS: OnceLock<ProfileParams> = OnceLock::new();
        Zensim::new(ZensimProfile::Custom {
            params: PARAMS.get_or_init(|| ProfileParams::builder().extended_features(true).build()),
            name: "attribution-test",
        })
    }

    /// Deterministic textured test pair (blocky JPEG-ish distortion).
    fn test_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                // Mixed content: gradient + texture + edges.
                let base = ((x * 255) / w) as u8;
                let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
                let edge = if (y / 16) % 2 == 0 { 40 } else { 0 };
                let px = [
                    base.wrapping_add(tex),
                    base.wrapping_add(edge),
                    (255 - base).wrapping_add(tex / 2),
                ];
                src.push(px);
                // Distortion: quantize + local shift in one quadrant.
                let q = |v: u8| (v / 12) * 12;
                let mut d = [q(px[0]), q(px[1]), q(px[2])];
                if x < w / 2 && y < h / 2 {
                    d[0] = d[0].saturating_add(18);
                }
                dst.push(d);
            }
        }
        (src, dst)
    }

    /// SAT queries must equal naive density sums over random rectangles.
    #[test]
    fn sat_matches_naive_rect_sums() {
        let (w, h) = (97, 61); // deliberately non-aligned dims
        // Deterministic pseudo-random signed density.
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let density: Vec<f32> = (0..w * h).map(|_| next() as f32).collect();
        let attr = AttributionResult::from_density(density.clone(), w, h);
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut rnd = move |m: usize| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 33) as usize % m
        };
        for _ in 0..200 {
            let x0 = rnd(w);
            let x1 = x0 + rnd(w - x0) + 1;
            let y0 = rnd(h);
            let y1 = y0 + rnd(h - y0) + 1;
            let mut naive = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    naive += density[y * w + x] as f64;
                }
            }
            let q = attr.query_rect(x0, y0, x1, y1);
            let tol = 1e-6 * naive.abs().max(1e-9) + 1e-9;
            assert!(
                (q - naive).abs() <= tol,
                "rect ({x0},{y0})..({x1},{y1}): sat {q} vs naive {naive}"
            );
        }
        // Degenerate / clamped rectangles.
        assert_eq!(attr.query_rect(5, 5, 5, 9), 0.0);
        assert_eq!(attr.query_rect(30, 20, 10, 25), 0.0);
        let full = attr.query_rect(0, 0, usize::MAX, usize::MAX);
        let total: f64 = density.iter().map(|&v| v as f64).sum();
        assert!((full - total).abs() <= 1e-6 * total.abs().max(1.0));
    }

    /// block_sums must equal the harness-style naive block partition.
    #[test]
    fn block_sums_match_naive_partition() {
        let (w, h) = (70, 50);
        let density: Vec<f32> = (0..w * h).map(|i| ((i % 17) as f32) - 4.0).collect();
        let attr = AttributionResult::from_density(density.clone(), w, h);
        for block in [16usize, 32, 33] {
            let bx = w.div_ceil(block);
            let by = h.div_ceil(block);
            let mut naive = vec![0.0f64; bx * by];
            for y in 0..h {
                for x in 0..w {
                    naive[(y / block) * bx + (x / block)] += density[y * w + x] as f64;
                }
            }
            let got = attr.block_sums(block);
            assert_eq!(got.len(), naive.len());
            for (b, (g, n)) in got.iter().zip(naive.iter()).enumerate() {
                assert!(
                    (g - n).abs() <= 1e-6 * n.abs().max(1.0),
                    "block {b} ({block}px): {g} vs {n}"
                );
            }
        }
    }

    /// Sum preservation, mean-pooled slots: with `s` = −1 on a single mean
    /// slot, the full-image query equals that feature's value exactly
    /// (density = plane/N; the f64 SAT keeps the identity to summation-order
    /// precision). Covers scale 0 AND scale 1 (exercises the sum-preserving
    /// upsample) on both the ssim (stored-plane) and mse (recomputed) slots.
    /// 64×64: SIMD-padded width == width, so no padding-column loss.
    #[test]
    fn sum_preservation_mean_slots() {
        let (w, h) = (64, 64);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let feats = z.compute_extended_features(&rs, &ds).unwrap();
        let feats = feats.features();
        // (k, tolerance): ssim-mean slots compare against the identical
        // stored f32 plane (1e-9); mse recomputes (s−d)² in f64 vs the
        // kernel's f32 product (1e-6).
        let cases = [
            (13usize, 1e-9),     // scale 0, ch Y, slot 0 (ssim mean)
            (39 + 13, 1e-9),     // scale 1, ch Y, slot 0 — upsample path
            (2 * 39 + 13, 1e-9), // scale 2, ch Y, slot 0
            (9, 1e-6),           // scale 0, ch X, slot 9 (mse)
            (39 + 26 + 9, 1e-6), // scale 1, ch B, slot 9
        ];
        for &(k, tol) in &cases {
            let mut s = vec![0.0f64; 156];
            s[k] = -1.0;
            let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
            let full = attr.query_rect(0, 0, w, h);
            let expect = feats[k];
            assert!(
                (full - expect).abs() <= tol * expect.abs().max(1e-12),
                "k={k}: full-image query {full} vs production feature {expect} (rel {})",
                ((full - expect) / expect.abs().max(1e-30)).abs()
            );
        }
    }

    /// p-pooled and hf slots: full-image sums land on the documented
    /// identities — f/p for p-pooled (removal-consistent 1/p integrand),
    /// exactly f for the active hf ratio slots.
    #[test]
    fn sum_preservation_ppool_and_hf_slots() {
        let (w, h) = (64, 64);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let feats = z.compute_extended_features(&rs, &ds).unwrap();
        let feats = feats.features();
        // (k, divisor): slot 1 = ssim p4 → f/4; slot 2 = ssim p2 → f/2;
        // slot 5 = art p2 → f/2 (scale 0, ch Y).
        for &(k, p) in &[(14usize, 4.0), (15, 2.0), (18, 2.0)] {
            let mut s = vec![0.0f64; 156];
            s[k] = -1.0;
            let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
            let full = attr.query_rect(0, 0, w, h);
            let expect = feats[k] / p;
            assert!(
                (full - expect).abs() <= 1e-5 * expect.abs().max(1e-12),
                "k={k}: full query {full} vs f/p {expect}"
            );
        }
        // hf slots: whichever of loss/gain is active sums to its feature; the
        // clamped side must contribute an all-zero density. Quantization
        // mostly removes texture → expect energy-loss active on Y at scale 0.
        for &(k_loss, k_gain) in &[(23usize, 25usize)] {
            for &(k, other) in &[(k_loss, k_gain), (k_gain, k_loss)] {
                let mut s = vec![0.0f64; 156];
                s[k] = -1.0;
                let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
                let full = attr.query_rect(0, 0, w, h);
                if feats[k] > 0.0 {
                    assert!(
                        (full - feats[k]).abs() <= 1e-5 * feats[k].max(1e-12),
                        "hf k={k}: full query {full} vs feature {}",
                        feats[k]
                    );
                    assert!(feats[other] == 0.0, "hf clamp pair both active?");
                } else {
                    assert!(
                        full.abs() <= 1e-12,
                        "hf k={k} clamped (feature 0) but density sums to {full}"
                    );
                }
            }
        }
        // hf_mag_loss (slot 11, ch Y scale 0 → k=24).
        let k = 24;
        let mut s = vec![0.0f64; 156];
        s[k] = -1.0;
        let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
        let full = attr.query_rect(0, 0, w, h);
        if feats[k] > 0.0 {
            assert!(
                (full - feats[k]).abs() <= 1e-5 * feats[k].max(1e-12),
                "hf_mag k={k}: {full} vs {}",
                feats[k]
            );
        } else {
            assert!(full.abs() <= 1e-12);
        }
    }

    /// Identical images: every signal plane is ~0, so the density is ~0
    /// everywhere regardless of `s`. The fused blur path carries ~3e-5
    /// float-precision noise on identical pairs (same floor the diffmap
    /// identical-images test documents), which the hf ratio slots divide by
    /// a near-cancelling Σ — 1e-4 is the established noise bound, not a
    /// correctness tolerance.
    #[test]
    fn identical_images_zero_density() {
        let (w, h) = (64, 64);
        let (src, _) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let s = vec![-1.0f64; 156];
        let attr = z.compute_attribution_density(&rs, &rs, &s).unwrap();
        let max = attr.density().iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max < 1e-4, "identical-pair density max {max}");
        assert_eq!(attr.width(), w);
        assert_eq!(attr.height(), h);
        assert_eq!(attr.density().len(), w * h);
    }

    /// `compute_attribution_density_full` wiring: with a gradient that is
    /// zero beyond the basic block it must equal the basic-only path
    /// exactly, and with v2/append weights present the result must equal
    /// basic + the feature_v2 density (same trimmed canvas addition).
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn full_density_is_basic_plus_v2_append() {
        let (w, h) = (96, 80);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let mut s = vec![0.0f64; 924];
        s[13] = -0.7; // basic: scale 0, ch Y, ssim mean
        let full_basic_only = z.compute_attribution_density_full(&rs, &ds, &s).unwrap();
        let basic = z.compute_attribution_density(&rs, &ds, &s[..156]).unwrap();
        for (a, b) in full_basic_only.density().iter().zip(basic.density().iter()) {
            assert!((a - b).abs() <= 1e-12, "basic-only full != basic path");
        }
        s[372 + 29 + 3] = -0.4; // v2 ART, scale 0, ch Y
        s[720 + 17 + 9] = -0.3; // append TEXTURE_DISSIM, scale 0, ch Y
        let full = z.compute_attribution_density_full(&rs, &ds, &s).unwrap();
        let v2a = crate::feature_v2::compute_v2_append_attribution(
            &rs,
            &ds,
            &s[372..720],
            Some(&s[720..924]),
            None,
            None,
            false,
        )
        .unwrap();
        for i in 0..w * h {
            let expect = basic.density()[i] as f64 + v2a.density[i];
            let got = full.density()[i] as f64;
            assert!(
                (got - expect).abs() <= 1e-6 * expect.abs().max(1e-9) + 1e-9,
                "pixel {i}: full {got} vs basic+v2app {expect}"
            );
        }
    }

    /// ANTI-RECURRENCE GUARD (campaign appendix E.2): the full-coverage
    /// density must cover exactly the intended slot set at every supported
    /// gradient width. The append2 block (`f924-943`) was silently dropped
    /// for the whole 944 era by a hard-coded `min(len, 924)` bound with no
    /// test standing between the regime bump and the loss — this is that
    /// test. Adding a block means adding a `BLOCK_END_*` constant AND a row
    /// here.
    ///
    /// Each probe sets ONE slot's gradient and asserts the density is
    /// non-zero (block reached, slot decomposable) or identically zero
    /// (block not reached at this width, or the slot is registered class N).
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn attribution_covers_expected_slots_per_width() {
        use crate::feature_v2::{APPEND2_PER_SCALE, idx_append2};
        let (w, h) = (96usize, 80usize);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);

        // |density| mass for a unit gradient on exactly one slot.
        let mass_at = |width: usize, k: usize| -> f64 {
            let mut s = vec![0.0f64; width];
            s[k] = -1.0;
            let d = z.compute_attribution_density_full(&rs, &ds, &s).unwrap();
            d.density().iter().map(|v| v.abs() as f64).sum()
        };
        // Same probe through the FUSED folded-944 entry (G-N3, appendix N):
        // the fused map must cover exactly the same slot set.
        let pre = z.precompute_reference(&rs).unwrap();
        let mut sess = Fused944Session::new();
        let mut mass_at_fused = |width: usize, k: usize| -> f64 {
            let mut s = vec![0.0f64; width];
            s[k] = -1.0;
            let (_, _, d) = z
                .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess)
                .unwrap();
            d.density().iter().map(|v| v.abs() as f64).sum()
        };

        // Representative decomposable slot per block, and the registered
        // class-N probes. (scale 0, ch Y where the block has a channel axis.)
        let basic_y = 13; // scale 0, ch Y, ssim mean
        let v1_pool = BLOCK_END_BASIC + 3; // f156-371: never spatialized
        let v2_y = BLOCK_END_V1_POOLS + 29 + 3; // v2 ART, scale 0, ch Y
        let append_y = BLOCK_END_V2 + 17 + 9; // append TEXTURE_DISSIM, scale 0, Y
        // append2 is Y-only with no channel axis; BANDVIS is a COARSE-scale
        // detector, so probe the coarsest scale (scale 3) where it fires.
        let app2 =
            |scale: usize, local: usize| BLOCK_END_APPEND + scale * APPEND2_PER_SCALE + local;

        // (width, slot, must_be_nonzero, why)
        let cases: &[(usize, usize, bool, &str)] = &[
            // 372: basic reached; v1 pools never; nothing above exists.
            (372, basic_y, true, "372: basic block"),
            (372, v1_pool, false, "372: f156-371 is never spatialized"),
            // 720: v2 reached, append does not exist at this width.
            (720, basic_y, true, "720: basic block"),
            (720, v1_pool, false, "720: f156-371 is never spatialized"),
            (720, v2_y, true, "720: v2 block"),
            // 924: append reached, append2 does not exist at this width.
            (924, v2_y, true, "924: v2 block"),
            (924, append_y, true, "924: append block"),
            // 944: append2 reached — the block the old slice dropped.
            (944, append_y, true, "944: append block still reached"),
            (
                944,
                app2(3, idx_append2::BANDVIS_GAIN),
                true,
                "944: append2 BANDVIS_GAIN is class E and MUST be covered",
            ),
            (
                944,
                app2(3, idx_append2::BANDVIS_LOSS),
                true,
                "944: append2 BANDVIS_LOSS is class E and MUST be covered",
            ),
            (
                944,
                app2(3, idx_append2::LUMA_MEAN_REF),
                false,
                "944: LUMA_MEAN_REF is reference-only — class N by definition",
            ),
            (
                944,
                app2(3, idx_append2::HL_BIN1),
                false,
                "944: HL_BIN1 is HDR-gated — structural zero on the SDR route",
            ),
            (
                944,
                app2(3, idx_append2::HL_BIN2),
                false,
                "944: HL_BIN2 is HDR-gated — structural zero on the SDR route",
            ),
        ];
        for &(width, k, want_nonzero, why) in cases {
            let m = mass_at(width, k);
            let m_f = mass_at_fused(width, k);
            if want_nonzero {
                assert!(m > 0.0, "{why}: expected non-zero density, got |mass| {m}");
                assert!(
                    m_f > 0.0,
                    "{why} (FUSED): expected non-zero density, got |mass| {m_f}"
                );
            } else {
                assert_eq!(
                    m, 0.0,
                    "{why}: expected identically-zero density, got |mass| {m}"
                );
                assert_eq!(
                    m_f, 0.0,
                    "{why} (FUSED): expected identically-zero density, got |mass| {m_f}"
                );
            }
        }

        // Beyond append2 (the f944+ CSFW block) there is no integrand yet —
        // deliberately not sliced. Pinned so a CSFW regime bump has to come
        // here and decide, rather than silently inheriting zero.
        let m = mass_at(BLOCK_END_APPEND2 + 12, BLOCK_END_APPEND2 + 3);
        assert_eq!(
            m, 0.0,
            "f944+ (CSFW) has no attribution integrand yet — must be exactly 0, got {m}"
        );
        let m_f = mass_at_fused(BLOCK_END_APPEND2 + 12, BLOCK_END_APPEND2 + 3);
        assert_eq!(
            m_f, 0.0,
            "f944+ (CSFW) has no FUSED attribution integrand yet — must be exactly 0, got {m_f}"
        );
    }

    /// C3a golden gate 1: the fused compare's SCORE is bit-identical to the
    /// fold-diffmap path's score (same pipeline, same stats, same
    /// apply_mlp_scoring) on a bake profile.
    #[test]
    fn fused_score_bit_matches_diffmap_path() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target());
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let fold = z
            .compute_with_ref_and_diffmap(&pre, &ds, crate::DiffmapWeighting::Trained)
            .unwrap();
        let s = vec![-1.0f64; 156];
        let (res, attr) = z
            .compute_with_ref_score_and_attribution(&pre, &ds, &s)
            .unwrap();
        assert_eq!(
            res.score().to_bits(),
            fold.score().to_bits(),
            "fused score {} != fold-path score {}",
            res.score(),
            fold.score()
        );
        assert_eq!(attr.width(), w);
        assert_eq!(attr.height(), h);
        assert!(attr.density().iter().all(|v| v.is_finite()));
    }

    /// C3a golden gate 2: the fused attribution equals the standalone f64
    /// density to f32-combine precision (identical retained planes by the
    /// C1 banding-parity construction; the deltas are the f32 kernel, the
    /// f32 canvas, and stats-derived vs sum-derived coefficients).
    #[test]
    fn fused_matches_standalone_attribution() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target());
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let mut s = vec![0.0f64; 156];
        for (k, v) in s.iter_mut().enumerate() {
            // Mixed-sign, all-slot gradient exercising every integrand.
            *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
        }
        let std_attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
        let (_, fused_attr) = z
            .compute_with_ref_score_and_attribution(&pre, &ds, &s)
            .unwrap();
        let max_abs = std_attr
            .density()
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs > 0.0);
        for (i, (a, b)) in fused_attr
            .density()
            .iter()
            .zip(std_attr.density().iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() <= 3e-5 * max_abs + 1e-9,
                "pixel {i}: fused {a} vs standalone {b} (max_abs {max_abs})"
            );
        }
        // Block sums through the SAT agree at the same class.
        let bs_f = fused_attr.block_sums(16);
        let bs_s = std_attr.block_sums(16);
        let bmax = bs_s.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (a, b)) in bs_f.iter().zip(bs_s.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * bmax.max(1e-12),
                "block {i}: fused {a} vs standalone {b}"
            );
        }
    }

    /// Appendix N G-N1: the fused folded-944 entry's features are BITWISE
    /// the canonical 944 extraction's (the retention hooks only copy —
    /// accumulation is untouched), so the standalone 944 score path
    /// (`score_features_with_profile` over the features) is bit-identical
    /// too — asserted through a shipped profile's forward.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn fused944_features_bitwise_and_score_match_standalone() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let std_v2 = z.compute_folded720_append2_features(&rs, &ds).unwrap();
        let mut s = vec![0.0f64; 944];
        for (k, v) in s.iter_mut().enumerate() {
            *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
        }
        let mut sess = Fused944Session::new();
        let (_res, fused_v2, attr) = z
            .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess)
            .unwrap();
        assert_eq!(fused_v2.features().len(), 944);
        assert_eq!(std_v2.features().len(), 944);
        for (k, (a, b)) in fused_v2
            .features()
            .iter()
            .zip(std_v2.features().iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "feature {k}: fused {a} vs standalone {b}"
            );
        }
        // The forward over the features (the loop's score step) — the
        // caller-width prefix branch accepts the 944 vector.
        let sf = crate::score_features_with_profile(
            crate::ZensimProfile::B,
            fused_v2.features(),
            w as u32,
            h as u32,
        )
        .unwrap();
        let ss = crate::score_features_with_profile(
            crate::ZensimProfile::B,
            std_v2.features(),
            w as u32,
            h as u32,
        )
        .unwrap();
        assert_eq!(
            sf.to_bits(),
            ss.to_bits(),
            "forward score: fused {sf} vs standalone {ss}"
        );
        assert_eq!((attr.width(), attr.height()), (w, h));
        assert!(attr.density().iter().all(|v| v.is_finite()));
    }

    /// Appendix N G-N2: the fused folded-944 density equals the standalone
    /// `compute_attribution_density_full` within the C3a tolerance class —
    /// the deltas are the f32 basic combine (C3a-gated) and the coefficient
    /// inputs (exact walk accumulators vs the standalone's 1e-9-parity
    /// pass-A replication); the retained planes are bitwise-equal.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn fused944_density_matches_standalone_full() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let mut s = vec![0.0f64; 944];
        for (k, v) in s.iter_mut().enumerate() {
            // Mixed-sign, all-slot gradient exercising every integrand in
            // every block (the C3a gate's pattern, extended to 944).
            *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
        }
        let std_attr = z.compute_attribution_density_full(&rs, &ds, &s).unwrap();
        let mut sess = Fused944Session::new();
        let (_res, _v2, fused_attr) = z
            .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess)
            .unwrap();
        let max_abs = std_attr
            .density()
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs > 0.0);
        for (i, (a, b)) in fused_attr
            .density()
            .iter()
            .zip(std_attr.density().iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() <= 3e-5 * max_abs + 1e-9,
                "pixel {i}: fused {a} vs standalone {b} (max_abs {max_abs})"
            );
        }
        let bs_f = fused_attr.block_sums(16);
        let bs_s = std_attr.block_sums(16);
        let bmax = bs_s.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (a, b)) in bs_f.iter().zip(bs_s.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * bmax.max(1e-12),
                "block {i}: fused {a} vs standalone {b}"
            );
        }
    }

    /// Fused-944 session reuse is pure buffer reuse: a second call with the
    /// same inputs through the SAME session reproduces features and density
    /// bitwise (no cross-compare numeric state, unlike AttributionSession).
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn fused944_session_reuse_is_deterministic() {
        let (w, h) = (150, 96);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let mut s = vec![0.0f64; 944];
        for (k, v) in s.iter_mut().enumerate() {
            *v = -0.5 - (k % 5) as f64 * 0.1;
        }
        let mut sess = Fused944Session::new();
        let (_r1, v1, a1) = z
            .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess)
            .unwrap();
        let (_r2, v2, a2) = z
            .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess)
            .unwrap();
        for (a, b) in v1.features().iter().zip(v2.features().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "session reuse changed features");
        }
        for (a, b) in a1.density().iter().zip(a2.density().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "session reuse changed density");
        }
    }

    /// The density must be finite everywhere on adversarial flat/extreme
    /// content (guards the pooled-scalar divisions).
    #[test]
    fn no_nan_on_flat_and_extreme_pairs() {
        let z = test_zensim();
        let s = vec![-0.5f64; 156];
        #[allow(clippy::type_complexity)]
        let cases: Vec<(Vec<[u8; 3]>, Vec<[u8; 3]>)> = vec![
            (vec![[0, 0, 0]; 64 * 64], vec![[255, 255, 255]; 64 * 64]),
            (
                vec![[128, 128, 128]; 64 * 64],
                vec![[128, 128, 128]; 64 * 64],
            ),
            (
                vec![[255, 255, 255]; 64 * 64],
                vec![[254, 255, 255]; 64 * 64],
            ),
        ];
        for (src, dst) in &cases {
            let rs = RgbSlice::new(src, 64, 64);
            let ds = RgbSlice::new(dst, 64, 64);
            let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
            assert!(
                attr.density().iter().all(|v| v.is_finite()),
                "non-finite density value"
            );
        }
    }

    /// #70 stale-scalar single-pass, exactness gate 1: with the SAME pair
    /// presented twice, the second (stale single-pass) call's map equals
    /// the fresh fused map of the previous input pair BITWISE — the
    /// in-strip fold with matching coefficient packs reproduces the
    /// retained-plane combine exactly (per-pixel elementwise kernel,
    /// identical spread/merge/upsample tail). Scores stay bit-identical on
    /// every call.
    #[test]
    fn stale_same_pair_second_call_equals_fresh_map_exactly() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target());
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let mut s = vec![0.0f64; 156];
        for (k, v) in s.iter_mut().enumerate() {
            *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
        }
        let (res_fresh, attr_fresh) = z
            .compute_with_ref_score_and_attribution(&pre, &ds, &s)
            .unwrap();
        let mut sess = AttributionSession::new();
        let (res1, attr1) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sess)
            .unwrap();
        // The priming call IS the fresh path.
        assert_eq!(res1.score().to_bits(), res_fresh.score().to_bits());
        for (a, b) in attr1.density().iter().zip(attr_fresh.density().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "priming map != fresh map");
        }
        // Second call: single-pass with the previous (same-pair) scalars —
        // must equal the fresh map of the previous input pair EXACTLY.
        let (res2, attr2) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sess)
            .unwrap();
        assert_eq!(
            res2.score().to_bits(),
            res_fresh.score().to_bits(),
            "stale-path score must stay bit-identical"
        );
        for (i, (a, b)) in attr2
            .density()
            .iter()
            .zip(attr_fresh.density().iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "pixel {i}: stale map {a} != fresh-previous-pair map {b}"
            );
        }
    }

    /// #70 exactness gate 2 (the stale semantics on DISTINCT pairs): after
    /// priming on pair A, the call on pair B returns score(B) bit-identical
    /// and a map equal — bitwise — to the reference construction "pair B's
    /// planes combined with pair A's coefficient packs" built through the
    /// retained-plane walk. Also proves staleness engages (differs from
    /// fresh(B)'s map).
    #[test]
    fn stale_call_on_new_pair_combines_current_planes_with_previous_scalars() {
        let (w, h) = (150, 170);
        let (src, dst_a) = test_pair(w, h);
        // A second, visibly different distortion for pair B.
        let dst_b: Vec<[u8; 3]> = dst_a
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let d = ((i % 13) as i16) - 6;
                [
                    (p[0] as i16 + d).clamp(0, 255) as u8,
                    (p[1] as i16 - d).clamp(0, 255) as u8,
                    (p[2] as i16 + d / 2).clamp(0, 255) as u8,
                ]
            })
            .collect();
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target());
        let rs = RgbSlice::new(&src, w, h);
        let dsa = RgbSlice::new(&dst_a, w, h);
        let dsb = RgbSlice::new(&dst_b, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let s = vec![-1.0f64; 156];

        let mut sess = AttributionSession::new();
        let _ = z
            .compute_with_ref_score_and_attribution_stale(&pre, &dsa, &s, &mut sess)
            .unwrap();
        let coeffs_a = sess.coeffs.clone();
        let (res_b, attr_ab) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &dsb, &s, &mut sess)
            .unwrap();

        let (res_b_fresh, attr_b_fresh) = z
            .compute_with_ref_score_and_attribution(&pre, &dsb, &s)
            .unwrap();
        assert_eq!(res_b.score().to_bits(), res_b_fresh.score().to_bits());

        // Reference construction: planes of B × coefficient packs of A,
        // through the retained-plane walk + the identical tail.
        let params = z.profile().params();
        let config = config_from_params(params, z.parallel());
        let (comp_pw, comp_h) = (pre.scales[0].1, pre.scales[0].2);
        let mut canvas = vec![0.0f32; comp_pw * comp_h];
        let mut id_plane = vec![0.0f32; comp_pw * comp_h];
        let mut win_plane = vec![0.0f32; comp_pw * comp_h];
        let mut spread_tmp: Vec<f32> = Vec::new();
        let mut spread_out: Vec<f32> = Vec::new();
        let on_scale = |scale: usize,
                        _stats: &crate::metric::ScaleStats,
                        src_planes: [&[f32]; 3],
                        dst_planes: [&[f32]; 3],
                        ret: &crate::streaming::AttrScaleRetention,
                        sw: usize,
                        sh: usize| {
            let n = sw * sh;
            id_plane[..n].fill(0.0);
            win_plane[..n].fill(0.0);
            let co32 = coeffs_a[scale];
            for c in 0..3 {
                fused_combine_plane_f32(
                    &ret.sd[c][..n],
                    &src_planes[c][..n],
                    &dst_planes[c][..n],
                    &ret.mu1[c][..n],
                    &ret.mu2[c][..n],
                    co32[c],
                    &mut id_plane[..n],
                    &mut win_plane[..n],
                );
            }
            crate::blur::box_spread_merge_f32(
                &mut win_plane[..n],
                &mut id_plane[..n],
                sw,
                sh,
                config.blur_radius,
                &mut spread_tmp,
                &mut spread_out,
                config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
            );
            upsample_add_sum_preserving_f32(
                &id_plane[..n],
                sw,
                sh,
                &mut canvas,
                comp_pw,
                comp_h,
                1usize << scale,
            );
        };
        let _ = crate::streaming::compute_zensim_streaming_with_ref_and_attr_planes(
            &pre,
            &dsb,
            &config,
            params.weights,
            on_scale,
        );
        let expected: Vec<f32> = if comp_pw == w && comp_h == h {
            canvas
        } else {
            let mut out = Vec::with_capacity(w * h);
            for y in 0..h.min(comp_h) {
                out.extend_from_slice(&canvas[y * comp_pw..y * comp_pw + w.min(comp_pw)]);
            }
            out
        };
        for (i, (a, b)) in attr_ab.density().iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "pixel {i}: stale map {a} != planes(B)×coeffs(A) reference {b}"
            );
        }
        // Staleness engages: the returned map differs from fresh(B).
        assert!(
            attr_ab
                .density()
                .iter()
                .zip(attr_b_fresh.density().iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "stale map unexpectedly identical to fresh(B) — staleness did not engage"
        );
    }

    /// #70: a gradient change re-primes (fresh path for the new gradient)
    /// instead of silently mixing semantics.
    #[test]
    fn stale_reprimes_on_gradient_change() {
        let (w, h) = (96, 80);
        let (src, dst) = test_pair(w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target());
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let s1 = vec![-1.0f64; 156];
        let mut s2 = vec![-0.5f64; 156];
        s2[13] = -2.0;
        let mut sess = AttributionSession::new();
        let _ = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s1, &mut sess)
            .unwrap();
        let (_, attr_s2) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s2, &mut sess)
            .unwrap();
        let (_, attr_s2_fresh) = z
            .compute_with_ref_score_and_attribution(&pre, &ds, &s2)
            .unwrap();
        for (a, b) in attr_s2.density().iter().zip(attr_s2_fresh.density().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "re-primed map != fresh map");
        }
        assert_eq!(sess.s_saved, s2);
    }

    /// #70: recycling spent results into the session is a pure allocation
    /// optimization — density AND SAT (guard-row/column re-zero logic in
    /// `build_sat_into`) must be bitwise-identical to the never-recycled
    /// arm on every subsequent call.
    #[test]
    fn stale_recycled_buffers_produce_identical_maps() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target());
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let s = vec![-1.0f64; 156];
        let rects = [
            (0usize, 0usize, w, h),
            (0, 0, 17, 23),
            (5, 7, 64, 64),
            (31, 100, 150, 170),
        ];
        // Arm A: never recycles.
        let mut sa = AttributionSession::new();
        let _ = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sa)
            .unwrap();
        let (_, a2) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sa)
            .unwrap();
        let (_, a3) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sa)
            .unwrap();
        // Arm B: recycles the spent result before every call.
        let mut sb = AttributionSession::new();
        let (_, b1) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sb)
            .unwrap();
        sb.recycle(b1);
        let (_, b2) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sb)
            .unwrap();
        for (i, (x, y)) in b2.density().iter().zip(a2.density().iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "px {i}: recycled density != fresh-alloc"
            );
        }
        for &(x0, y0, x1, y1) in &rects {
            assert_eq!(
                b2.query_rect(x0, y0, x1, y1).to_bits(),
                a2.query_rect(x0, y0, x1, y1).to_bits(),
                "SAT rect ({x0},{y0})..({x1},{y1}) differs after recycle"
            );
        }
        sb.recycle(b2);
        let (_, b3) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sb)
            .unwrap();
        for (i, (x, y)) in b3.density().iter().zip(a3.density().iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "px {i}: 2nd recycled density != fresh-alloc"
            );
        }
        for &(x0, y0, x1, y1) in &rects {
            assert_eq!(
                b3.query_rect(x0, y0, x1, y1).to_bits(),
                a3.query_rect(x0, y0, x1, y1).to_bits(),
                "SAT rect ({x0},{y0})..({x1},{y1}) differs after 2nd recycle"
            );
        }
    }
}

// ============================================================================
// C3a: the FUSED compare — scalar score + attribution map from ONE pipeline
// ============================================================================

/// Coefficient pack for the f32 fused combine, ordered
/// `[sd, sd4, sd2, art, art4, art2, det, det4, det2, mse, hfe, hfm]`.
fn coeffs_to_f32(co: &SlotCoeffs) -> [f32; 12] {
    [
        co.c_sd as f32,
        co.c_sd4 as f32,
        co.c_sd2 as f32,
        co.c_art as f32,
        co.c_art4 as f32,
        co.c_art2 as f32,
        co.c_det as f32,
        co.c_det4 as f32,
        co.c_det2 as f32,
        co.c_mse as f32,
        co.c_hfe as f32,
        co.c_hfm as f32,
    ]
}

/// f32 fused-combine kernel (C3a perf lever): the same integrand formulas as
/// [`basic_combine_channel`], computed in f32 and auto-vectorized per ISA.
/// Precision class: the density-sum identities move from the f64 path's
/// 1e-9/1e-6 to ~1e-5 relative (measured; the standalone f64 path and its
/// strict tests are unchanged — this kernel serves the fused entry only).
#[autoversion]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_combine_plane_f32(
    sd: &[f32],
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    co: [f32; 12],
    id_plane: &mut [f32],
    win_plane: &mut [f32],
) {
    let n = id_plane.len();
    for i in 0..n {
        let sdv = sd[i];
        let sv = src[i];
        let dv = dst[i];
        let m1 = mu1[i];
        let m2 = mu2[i];
        let d1 = (sv - m1).abs();
        let d2 = (dv - m2).abs();
        let ed = (1.0 + d2) / (1.0 + d1) - 1.0;
        let art = ed.max(0.0);
        let det = (-ed).max(0.0);
        let sd2 = sdv * sdv;
        let a2 = art * art;
        let dt2 = det * det;
        let pd = sv - dv;
        let e_hf = d1 * d1 - d2 * d2;
        let win_term = co[0] * sdv + co[1] * (sd2 * sd2) + co[2] * sd2;
        let res_term = co[3] * art
            + co[4] * (a2 * a2)
            + co[5] * a2
            + co[6] * det
            + co[7] * (dt2 * dt2)
            + co[8] * dt2
            + co[10] * e_hf
            + co[11] * (d1 - d2);
        let px_term = co[9] * (pd * pd);
        id_plane[i] += px_term + res_term;
        win_plane[i] += win_term;
    }
}

/// `Σ (src−mu1)²` and `Σ |src−mu1|` over a plane — the two reference-side
/// pooled sums the hf coefficients need that `ScaleStats` does not carry.
#[autoversion]
fn hf_src_sums(src: &[f32], mu1: &[f32]) -> (f64, f64) {
    let mut sq = 0.0f64;
    let mut ab = 0.0f64;
    for i in 0..src.len().min(mu1.len()) {
        let r = src[i] - mu1[i];
        sq += (r * r) as f64;
        ab += r.abs() as f64;
    }
    (sq, ab)
}

/// f32 twin of [`upsample_add_sum_preserving`] for the fused canvas.
/// `pub(crate)`: also the fused folded-944 pass-B upsample (appendix P
/// lever 1, `feature_v2::attr_pass_b_for_scale_f32`).
pub(crate) fn upsample_add_sum_preserving_f32(
    scale_plane: &[f32],
    sw: usize,
    sh: usize,
    canvas: &mut [f32],
    cw: usize,
    ch: usize,
    factor: usize,
) {
    if factor == 1 {
        debug_assert_eq!(sw, cw);
        for (c, &v) in canvas.iter_mut().zip(scale_plane.iter()) {
            *c += v;
        }
        return;
    }
    // Destination-row-major replicate-add via the diffmap fusion's SIMD
    // kernel (#70: the former source-major footprint loop revisited each
    // canvas row `factor` times with scattered block writes — same op
    // count, far worse locality). Bitwise-identical: each canvas element
    // still receives exactly one add of the identically-formed product
    // `src · inv_area`.
    let inv_area = 1.0 / ((factor * factor) as f32);
    let copy_w = (sw * factor).min(cw);
    let dh = (sh * factor).min(ch);
    for dy in 0..dh {
        let sy = dy / factor;
        let src_row = &scale_plane[sy * sw..(sy + 1) * sw];
        let dst_row = &mut canvas[dy * cw..dy * cw + copy_w];
        crate::streaming::upsample_row_powx_add(src_row, dst_row, factor, inv_area);
    }
}

impl SlotCoeffs {
    /// Derive the combine coefficients from a finalized [`ScaleStats`]
    /// (the fused path — the pooled roots are inverted back to raw-moment
    /// means: `M4 = f4⁴`, `M2 = f2²`) plus the two reference-side hf sums
    /// the stats do not carry. Matches [`SlotCoeffs::derive`]'s math; the
    /// clamp-side gating for the hf ratio slots reads the stats' CLAMPED
    /// feature values directly (`f10 > 0` ⇔ loss side active, etc.).
    fn from_scale_stats(
        s: &[f64],
        base_k: usize,
        stats: &crate::metric::ScaleStats,
        c: usize,
        n_f: f64,
        hf_sq_src_sum: f64,
        hf_abs_src_sum: f64,
    ) -> Self {
        let g = |slot: usize| s.get(base_k + slot).copied().unwrap_or(0.0);
        let inv_n = 1.0 / n_f;
        let p_coeff = |sk: f64, m: f64, p: f64| -> f64 {
            if m > 0.0 {
                -sk * inv_n * (1.0 / p) * m.powf((1.0 - p) / p)
            } else {
                0.0
            }
        };
        let m4_ssim = stats.ssim[c * 2 + 1].powi(4);
        let m2_ssim = stats.ssim_2nd[c].powi(2);
        let m4_art = stats.edge[c * 4 + 1].powi(4);
        let m2_art = stats.edge_2nd[c * 2].powi(2);
        let m4_det = stats.edge[c * 4 + 3].powi(4);
        let m2_det = stats.edge_2nd[c * 2 + 1].powi(2);
        let c_hfe = if hf_sq_src_sum > 1e-10 * n_f {
            if stats.hf_energy_loss[c] > 0.0 {
                -g(10) / hf_sq_src_sum
            } else if stats.hf_energy_gain[c] > 0.0 {
                g(12) / hf_sq_src_sum
            } else {
                0.0
            }
        } else {
            0.0
        };
        let c_hfm = if hf_abs_src_sum > 1e-10 * n_f && stats.hf_mag_loss[c] > 0.0 {
            -g(11) / hf_abs_src_sum
        } else {
            0.0
        };
        Self {
            c_sd: -g(0) * inv_n,
            c_sd4: p_coeff(g(1), m4_ssim, 4.0),
            c_sd2: p_coeff(g(2), m2_ssim, 2.0),
            c_art: -g(3) * inv_n,
            c_art4: p_coeff(g(4), m4_art, 4.0),
            c_art2: p_coeff(g(5), m2_art, 2.0),
            c_det: -g(6) * inv_n,
            c_det4: p_coeff(g(7), m4_det, 4.0),
            c_det2: p_coeff(g(8), m2_det, 2.0),
            c_mse: -g(9) * inv_n,
            c_hfe,
            c_hfm,
        }
    }
}

/// Cross-compare state for the **stale-scalar single-pass** fused compare
/// (task #70, C3a ranked lever 3):
/// [`Zensim::compute_with_ref_score_and_attribution_stale`](crate::Zensim::compute_with_ref_score_and_attribution_stale).
///
/// Holds, from the previous compare against the same reference: the
/// per-(scale, channel) fused-combine coefficient packs (derived from that
/// compare's pooled scalars) and the reference-side hf sums (distortion-
/// independent, cached at the priming call). One session serves ONE
/// (reference, image-size, gradient) triple — a codec loop creates one per
/// encode. Reuse across a *different* reference of the same size is a
/// caller error the session cannot detect (the map would mix references);
/// call [`reset`](Self::reset) when the reference changes. Size or
/// gradient changes are detected and re-prime automatically.
#[derive(Default)]
pub struct AttributionSession {
    /// Per visited scale: per-channel combine coefficient packs from the
    /// previous compare's pooled scalars.
    coeffs: Vec<[[f32; 12]; 3]>,
    /// Per visited scale: per-channel reference-side hf sums
    /// `(Σ(src−μ1)², Σ|src−μ1|)` — constant for a fixed reference.
    hf_src: Vec<[(f64, f64); 3]>,
    /// The gradient + dims the session was primed with (identity guard).
    s_saved: Vec<f64>,
    width: usize,
    height: usize,
    /// Reusable per-call scratch (comp-plane sized) — kept across calls so
    /// the single-pass path pays no allocation / page-fault cost per
    /// iteration.
    canvas: Vec<f32>,
    id_plane: Vec<f32>,
    win_plane: Vec<f32>,
    spread_tmp: Vec<f32>,
    spread_out: Vec<f32>,
    /// Recycled result buffers (see [`recycle`](Self::recycle)); empty
    /// unless the caller opts in.
    density_scratch: Vec<f32>,
    sat_scratch: Vec<f64>,
}

impl AttributionSession {
    /// Empty (unprimed) session — the first compare through it runs the
    /// fresh fused path and primes the cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all cached state; the next compare re-primes (fresh path).
    /// Call when switching to a different reference image.
    pub fn reset(&mut self) {
        self.coeffs.clear();
        self.hf_src.clear();
        self.s_saved.clear();
        self.width = 0;
        self.height = 0;
    }

    /// Return a spent [`AttributionResult`] to the session so the next
    /// stale compare reuses its buffers (density plane + f64 SAT) instead
    /// of allocating and page-faulting fresh multi-MB ones every
    /// iteration — the codec-loop shape drops one result per compare.
    /// Purely an allocation optimization: opting out (never calling this)
    /// changes nothing but the per-call allocations, and the produced
    /// maps are bitwise-identical either way.
    pub fn recycle(&mut self, spent: AttributionResult) {
        self.density_scratch = spent.density;
        self.sat_scratch = spent.sat;
    }

    fn primed_for(&self, width: usize, height: usize, s: &[f64]) -> bool {
        !self.coeffs.is_empty() && self.width == width && self.height == height && self.s_saved == s
    }
}

/// The fused v1 walk core's output (appendix N refactor): the compare
/// result + the UNTRIMMED f32 basic-block attribution canvas at compute
/// dims, with the walk timings the ATTRPERF diagnostic reports.
struct FusedBasicCanvas {
    result: crate::metric::ZensimResult,
    canvas: Vec<f32>,
    comp_pw: usize,
    comp_h: usize,
    t_pipe_ms: f64,
    combine_ms: f64,
}

impl FusedBasicCanvas {
    /// Trim the compute-dims canvas to the logical image (top-left).
    fn trim_to(&self, width: usize, height: usize) -> Vec<f32> {
        if self.comp_pw == width && self.comp_h == height {
            self.canvas.clone()
        } else {
            let mut out = Vec::with_capacity(width * height);
            for y in 0..height.min(self.comp_h) {
                out.extend_from_slice(
                    &self.canvas[y * self.comp_pw..y * self.comp_pw + width.min(self.comp_pw)],
                );
            }
            out
        }
    }
}

/// Reusable state for the fused folded-944 compare
/// ([`crate::Zensim::compute_folded944_score_and_attribution`]): the
/// streaming extraction's scratch + the walk retention (planes, pyramid
/// copies, exact accumulators). One session per encode loop; reuse across
/// compares avoids re-allocating ~42 MB (at 576²) of retained planes per
/// call. Unlike [`AttributionSession`] there is no cross-compare numeric
/// state — every retained value is rewritten by each call — so reuse
/// cannot change any output (gated by
/// `fused944_session_reuse_is_deterministic`).
#[cfg(feature = "feature-regime-v2")]
#[derive(Default)]
pub struct Fused944Session {
    scratch: crate::feature_v2::V2Scratch,
    retention: crate::feature_v2::FoldRetention,
    /// f32 pass-B scratch (appendix P lever 1) — plane-sized buffers
    /// reused across compares so per-iteration callers pay no
    /// allocation / page-fault cost.
    pass_b: crate::feature_v2::PassBScratchF32,
}

#[cfg(feature = "feature-regime-v2")]
impl Fused944Session {
    /// Empty session — buffers grow on first use.
    pub fn new() -> Self {
        Self::default()
    }
}

impl crate::metric::Zensim {
    /// **The fused compare (task #67 C3a)**: scalar score + attribution
    /// steering map from ONE plane pipeline — the codec-loop call shape.
    ///
    /// Runs the SAME per-scale walk as
    /// [`compute_with_ref_and_diffmap`](Self::compute_with_ref_and_diffmap)
    /// (the score is bit-identical to that path's), retaining each scale's
    /// SSIM-error and mu planes and deriving the BASIC-block (f0-155)
    /// attribution density from them with no additional blur work. The
    /// marginal cost over a score-only compare is the f32 combine + spread
    /// + SAT.
    ///
    /// `s` is the caller-supplied model gradient (`∂score/∂f_k`, basic
    /// layout; entries past 155 ignored). The density uses the C2b-final
    /// allocation (window-only spread) and the f32 fused-combine kernel —
    /// vs the standalone f64 [`compute_attribution_density`]
    /// (Self::compute_attribution_density) the density agrees to f32
    /// recompute precision (~2e-5 relative, gated by
    /// `fused_matches_standalone_attribution`), while the standalone path
    /// and its strict f64 identities are unchanged.
    ///
    /// # Errors
    ///
    /// Besides the usual pair validations: the profile must compute the
    /// full basic feature set on every channel (an MLP/bake profile —
    /// `compute_all_features` — or `extended_features`), and
    /// `blur_passes == 1`; otherwise
    /// [`ZensimError::ModelForwardFailed`] with a static reason.
    pub fn compute_with_ref_score_and_attribution(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
    ) -> Result<(crate::metric::ZensimResult, AttributionResult), ZensimError> {
        self.fused_score_attr_fresh(precomputed, distorted, s, None)
    }

    /// The fresh fused pipeline (the pre-#70 body of
    /// [`compute_with_ref_score_and_attribution`](Self::compute_with_ref_score_and_attribution),
    /// unchanged numerically). When `prime` is `Some`, additionally record
    /// each scale's coefficient packs + reference-side hf sums into the
    /// session — the stale entry's priming path.
    fn fused_score_attr_fresh(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        prime: Option<&mut AttributionSession>,
    ) -> Result<(crate::metric::ZensimResult, AttributionResult), ZensimError> {
        let fb = self.fused_basic_canvas(precomputed, distorted, s, prime)?;
        let width = distorted.width();
        let height = distorted.height();
        let perf_log = std::env::var("ZENSIM_ATTR_PERF").as_deref() == Ok("1");
        let t_sat0 = std::time::Instant::now();
        let trimmed = fb.trim_to(width, height);
        let attr = AttributionResult::from_density(trimmed, width, height);
        if perf_log {
            eprintln!(
                "ATTRPERF fused: pipeline {:.1} ms (combine/spread/upsample {:.1} ms → retention+stats {:.1} ms) | trim+SAT {:.1} ms",
                fb.t_pipe_ms,
                fb.combine_ms,
                fb.t_pipe_ms - fb.combine_ms,
                t_sat0.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok((fb.result, attr))
    }

    /// The fused v1 walk core (appendix N refactor of the pre-existing
    /// fused pipeline, numerically unchanged): one score walk with
    /// attribution-plane retention → the compare's [`crate::metric::ZensimResult`]
    /// plus the UNTRIMMED f32 basic-block canvas at compute dims. The C3a
    /// entry trims + SATs it directly; the fused folded-944 entry sums the
    /// v2/append/append2 retention density into it first.
    fn fused_basic_canvas(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        mut prime: Option<&mut AttributionSession>,
    ) -> Result<FusedBasicCanvas, ZensimError> {
        const FPC: usize = FEATURES_PER_CHANNEL_BASIC;
        let params = self.profile().params();
        if distorted.width() == 0 || distorted.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels())?;
        let config = config_from_params(params, self.parallel());
        if config.blur_passes != 1 {
            return Err(ZensimError::ModelForwardFailed {
                reason: "fused attribution requires blur_passes == 1 (all shipped profiles)",
            });
        }
        if !(config.compute_all_features || config.extended_features) {
            return Err(ZensimError::ModelForwardFailed {
                reason: "fused attribution requires a profile computing all basic features \
                         (an MLP/bake profile or extended_features)",
            });
        }

        let width = distorted.width();
        let height = distorted.height();
        let (comp_pw, comp_h) = (precomputed.scales[0].1, precomputed.scales[0].2);

        let t_all = std::time::Instant::now();
        let combine_ms = std::cell::Cell::new(0.0f64);
        let mut canvas = vec![0.0f32; comp_pw * comp_h];
        let mut id_plane = vec![0.0f32; comp_pw * comp_h];
        let mut win_plane = vec![0.0f32; comp_pw * comp_h];
        let mut spread_tmp: Vec<f32> = Vec::new();
        let mut spread_out: Vec<f32> = Vec::new();

        let on_scale = |scale: usize,
                        stats: &crate::metric::ScaleStats,
                        src_planes: [&[f32]; 3],
                        dst_planes: [&[f32]; 3],
                        ret: &crate::streaming::AttrScaleRetention,
                        sw: usize,
                        sh: usize| {
            let t_c = std::time::Instant::now();
            let n = sw * sh;
            let n_f = n as f64;
            id_plane[..n].fill(0.0);
            win_plane[..n].fill(0.0);
            let hf: [(f64, f64); 3] =
                core::array::from_fn(|c| hf_src_sums(&src_planes[c][..n], &ret.mu1[c][..n]));
            let co32: [[f32; 12]; 3] = core::array::from_fn(|c| {
                let base_k = scale * FPC * 3 + c * FPC;
                coeffs_to_f32(&SlotCoeffs::from_scale_stats(
                    s, base_k, stats, c, n_f, hf[c].0, hf[c].1,
                ))
            });
            // task #70: prime the stale session — THIS compare's coefficient
            // packs become the NEXT stale call's fold input; the hf sums are
            // reference-side constants cached once here.
            if let Some(sess) = prime.as_deref_mut() {
                sess.coeffs.push(co32);
                sess.hf_src.push(hf);
            }
            // Row-banded parallel combine (disjoint id/win rows per band).
            let run_band = |band: usize, idc: &mut [f32], winc: &mut [f32]| {
                let off = band * 64 * sw;
                let len = idc.len();
                for c in 0..3 {
                    fused_combine_plane_f32(
                        &ret.sd[c][off..off + len],
                        &src_planes[c][off..off + len],
                        &dst_planes[c][off..off + len],
                        &ret.mu1[c][off..off + len],
                        &ret.mu2[c][off..off + len],
                        co32[c],
                        idc,
                        winc,
                    );
                }
            };
            #[cfg(feature = "threads")]
            let banded = if config.allow_multithreading && sh > 64 {
                use rayon::prelude::*;
                id_plane[..n]
                    .par_chunks_mut(64 * sw)
                    .zip(win_plane[..n].par_chunks_mut(64 * sw))
                    .enumerate()
                    .for_each(|(band, (idc, winc))| run_band(band, idc, winc));
                true
            } else {
                false
            };
            #[cfg(not(feature = "threads"))]
            let banded = false;
            if !banded {
                run_band(0, &mut id_plane[..n], &mut win_plane[..n]);
            }
            // Window-class spread (C2b-final allocation) fused with the
            // window→identity merge, f32 fast path (#70 lever 1:
            // parallel, bitwise-invariant to banding). At scale 0 the
            // upsample is elementwise, so the spread merges directly into
            // the canvas — skipping a full id-plane store+reload; this is
            // value-exact: `(0+a)+b` and `0+(a+b)` round identically
            // (the first add is exact, signed zeros included).
            if scale == 0 {
                upsample_add_sum_preserving_f32(
                    &id_plane[..n],
                    sw,
                    sh,
                    &mut canvas,
                    comp_pw,
                    comp_h,
                    1,
                );
                crate::blur::box_spread_merge_f32(
                    &mut win_plane[..n],
                    &mut canvas[..n],
                    sw,
                    sh,
                    config.blur_radius,
                    &mut spread_tmp,
                    &mut spread_out,
                    config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
                );
            } else {
                crate::blur::box_spread_merge_f32(
                    &mut win_plane[..n],
                    &mut id_plane[..n],
                    sw,
                    sh,
                    config.blur_radius,
                    &mut spread_tmp,
                    &mut spread_out,
                    config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
                );
                upsample_add_sum_preserving_f32(
                    &id_plane[..n],
                    sw,
                    sh,
                    &mut canvas,
                    comp_pw,
                    comp_h,
                    1usize << scale,
                );
            }
            combine_ms.set(combine_ms.get() + t_c.elapsed().as_secs_f64() * 1e3);
        };

        let result = crate::streaming::compute_zensim_streaming_with_ref_and_attr_planes(
            precomputed,
            distorted,
            &config,
            params.weights,
            on_scale,
        );
        let mut result = result.with_profile(self.profile());
        // Same real-scoring step as `compute_with_ref_and_diffmap` — the
        // scalar golden gate (`fused_score_bit_matches_diffmap_path`) holds
        // this path bit-identical to the fold-diffmap call's score.
        crate::metric::apply_mlp_scoring_with_codec(
            &mut result,
            params,
            width as u32,
            height as u32,
            None,
        )?;

        let t_pipe = t_all.elapsed().as_secs_f64() * 1e3;
        Ok(FusedBasicCanvas {
            result,
            canvas,
            comp_pw,
            comp_h,
            t_pipe_ms: t_pipe,
            combine_ms: combine_ms.get(),
        })
    }

    /// **Stale-scalar single-pass fused compare (task #70; C3a ranked
    /// lever 3)**: scalar score + attribution steering map from ONE
    /// pipeline pass with NO second sweep — the attribution combine runs
    /// in-strip on cache-hot planes using the coefficient packs derived
    /// from the PREVIOUS compare through the same `session`.
    ///
    /// **Semantics — the map lags one iterate.** The returned map combines
    /// THIS compare's per-pixel signal planes with the PREVIOUS compare's
    /// pooled scalars (`SlotCoeffs` — the stats-derived coefficient
    /// packs). One-iterate staleness of the steering signal is the
    /// proven-free semantics in the codec loop (C3b `attr-stale`, #69 G4,
    /// and the metric-matrix study all measured it free); this variant is
    /// strictly fresher than those arms' fully-stale map (planes are
    /// current, only the pooled scalars lag). The **score is bit-identical
    /// to every other compare path** (the fold never touches the stats
    /// pipeline) — only the map lags.
    ///
    /// The FIRST call through an unprimed `session` (and any call after
    /// [`AttributionSession::reset`], or when the image size or `s`
    /// changes) runs the fresh fused path
    /// ([`compute_with_ref_score_and_attribution`](Self::compute_with_ref_score_and_attribution))
    /// and primes the session — that call returns the FRESH map at fresh
    /// cost; subsequent calls are single-pass. The session is valid for
    /// ONE precomputed reference; switching references without `reset()`
    /// is a caller error the session cannot detect.
    ///
    /// # Errors
    ///
    /// Same contract as the fresh fused entry (all-basic-features profile,
    /// `blur_passes == 1`, matching pair dims).
    pub fn compute_with_ref_score_and_attribution_stale(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        session: &mut AttributionSession,
    ) -> Result<(crate::metric::ZensimResult, AttributionResult), ZensimError> {
        const FPC: usize = FEATURES_PER_CHANNEL_BASIC;
        let params = self.profile().params();
        if distorted.width() == 0 || distorted.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels())?;
        let config = config_from_params(params, self.parallel());
        if config.blur_passes != 1 {
            return Err(ZensimError::ModelForwardFailed {
                reason: "fused attribution requires blur_passes == 1 (all shipped profiles)",
            });
        }
        if !(config.compute_all_features || config.extended_features) {
            return Err(ZensimError::ModelForwardFailed {
                reason: "fused attribution requires a profile computing all basic features \
                         (an MLP/bake profile or extended_features)",
            });
        }

        let width = distorted.width();
        let height = distorted.height();

        // Unprimed, or the size/gradient changed: fresh fused path, prime.
        if !session.primed_for(width, height, s) {
            session.reset();
            session.width = width;
            session.height = height;
            session.s_saved = s.to_vec();
            return self.fused_score_attr_fresh(precomputed, distorted, s, Some(session));
        }

        let (comp_pw, comp_h) = (precomputed.scales[0].1, precomputed.scales[0].2);
        let perf_log = std::env::var("ZENSIM_ATTR_PERF").as_deref() == Ok("1");
        let t_all = std::time::Instant::now();
        let tail_ms = std::cell::Cell::new(0.0f64);
        // Session-owned scratch: no per-iteration allocation/page-fault
        // cost on the single-pass path. The canvas accumulates across
        // scales, so it is re-zeroed here.
        let comp_n = comp_pw * comp_h;
        session.canvas.resize(comp_n, 0.0);
        session.canvas.fill(0.0);
        session.id_plane.resize(comp_n, 0.0);
        session.win_plane.resize(comp_n, 0.0);
        let mut canvas = std::mem::take(&mut session.canvas);
        let mut id_plane = std::mem::take(&mut session.id_plane);
        let mut win_plane = std::mem::take(&mut session.win_plane);
        let mut spread_tmp = std::mem::take(&mut session.spread_tmp);
        let mut spread_out = std::mem::take(&mut session.spread_out);
        // The previous compare's packs drive THIS pass's in-strip fold;
        // this pass's stats produce the packs for the NEXT call. (On an
        // error return below the session is left unprimed — the next call
        // re-primes through the fresh path.)
        let coeffs_prev = std::mem::take(&mut session.coeffs);
        let mut next_coeffs: Vec<[[f32; 12]; 3]> = Vec::with_capacity(coeffs_prev.len());
        let hf_cached = &session.hf_src;

        let on_scale = |scale: usize,
                        stats: &crate::metric::ScaleStats,
                        idp: &mut [f32],
                        winp: &mut [f32],
                        sw: usize,
                        sh: usize| {
            let t_c = std::time::Instant::now();
            let n = sw * sh;
            let n_f = n as f64;
            // Post-scale tail — IDENTICAL code to the fresh path (same
            // fused spread+merge, same scale-0 canvas-target fusion, same
            // upsample ⇒ bitwise-equal maps when the coefficient packs
            // coincide).
            if scale == 0 {
                upsample_add_sum_preserving_f32(&idp[..n], sw, sh, &mut canvas, comp_pw, comp_h, 1);
                crate::blur::box_spread_merge_f32(
                    &mut winp[..n],
                    &mut canvas[..n],
                    sw,
                    sh,
                    config.blur_radius,
                    &mut spread_tmp,
                    &mut spread_out,
                    config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
                );
            } else {
                crate::blur::box_spread_merge_f32(
                    &mut winp[..n],
                    &mut idp[..n],
                    sw,
                    sh,
                    config.blur_radius,
                    &mut spread_tmp,
                    &mut spread_out,
                    config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
                );
                upsample_add_sum_preserving_f32(
                    &idp[..n],
                    sw,
                    sh,
                    &mut canvas,
                    comp_pw,
                    comp_h,
                    1usize << scale,
                );
            }
            // Derive the NEXT call's packs from THIS compare's pooled
            // scalars + the cached reference-side hf sums.
            let hf = hf_cached.get(scale).copied().unwrap_or([(0.0, 0.0); 3]);
            next_coeffs.push(core::array::from_fn(|c| {
                let base_k = scale * FPC * 3 + c * FPC;
                coeffs_to_f32(&SlotCoeffs::from_scale_stats(
                    s, base_k, stats, c, n_f, hf[c].0, hf[c].1,
                ))
            }));
            tail_ms.set(tail_ms.get() + t_c.elapsed().as_secs_f64() * 1e3);
        };

        let result = crate::streaming::compute_zensim_streaming_with_ref_and_attr_fold(
            precomputed,
            distorted,
            &config,
            params.weights,
            &coeffs_prev,
            &mut id_plane,
            &mut win_plane,
            on_scale,
        );
        let mut result = result.with_profile(self.profile());
        // Same real-scoring step as every compare path — bit-identical
        // score (the in-strip fold reads planes, never writes stats).
        crate::metric::apply_mlp_scoring_with_codec(
            &mut result,
            params,
            width as u32,
            height as u32,
            None,
        )?;
        session.coeffs = next_coeffs;

        let t_pipe = t_all.elapsed().as_secs_f64() * 1e3;
        let t_sat0 = std::time::Instant::now();
        // Trim into the recycled density buffer (fresh alloc when the
        // caller never recycles — identical behavior, just re-allocating).
        let mut trimmed = std::mem::take(&mut session.density_scratch);
        trimmed.clear();
        if comp_pw == width && comp_h == height {
            trimmed.extend_from_slice(&canvas);
        } else {
            trimmed.reserve(width * height);
            for y in 0..height.min(comp_h) {
                trimmed.extend_from_slice(&canvas[y * comp_pw..y * comp_pw + width.min(comp_pw)]);
            }
        }
        let sat_scratch = std::mem::take(&mut session.sat_scratch);
        // Return the scratch to the session for the next iteration.
        session.canvas = canvas;
        session.id_plane = id_plane;
        session.win_plane = win_plane;
        session.spread_tmp = spread_tmp;
        session.spread_out = spread_out;
        let attr =
            AttributionResult::from_density_with_sat_scratch(trimmed, width, height, sat_scratch);
        if perf_log {
            eprintln!(
                "ATTRPERF stale: pipeline {:.1} ms (spread/upsample/derive tail {:.1} ms; in-strip fold inside walk) | trim+SAT {:.1} ms",
                t_pipe,
                tail_ms.get(),
                t_sat0.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok((result, attr))
    }

    /// **The fused folded-944 compare (campaign appendix N)**: canonical
    /// streaming folded-944 extraction + FULL-coverage attribution
    /// steering map from ONE extraction pass plus one v1 walk — the
    /// 944-class codec-loop call shape.
    ///
    /// Returns, in order:
    /// 1. the v1 walk's [`crate::metric::ZensimResult`] under `self`'s
    ///    profile (the loop reads `approx_butteraugli` from it — its score
    ///    is NOT the 944 model's score);
    /// 2. the [`crate::feature_v2::ZensimV2Result`] carrying the 944
    ///    features, BITWISE identical to
    ///    [`compute_folded720_append2_features`](Self::compute_folded720_append2_features)
    ///    (G-N1 — the retention hooks only copy; accumulation is
    ///    untouched). The caller forwards `features()` through
    ///    [`crate::score_features_with_profile`] exactly as the unfused
    ///    score path does — a PRUNED bake (caller 944 / internal 667)
    ///    consumes the full caller-width vector;
    /// 3. the [`AttributionResult`]: basic block (f0-155) from the C3a
    ///    fused v1 machinery + v2/append/append2 (f372-943) from pass-B
    ///    over the walk retention, coefficients derived from the EXACT
    ///    extraction accumulators. Matches the standalone
    ///    [`compute_attribution_density_full`](Self::compute_attribution_density_full)
    ///    within the C3a tolerance class (G-N2); per-width slot coverage
    ///    is pinned by the shared coverage gate (G-N3).
    ///
    /// `source` must be the SAME image `precomputed` was built from (the
    /// usual `*_with_ref` caller contract; dims are validated, content
    /// cannot be). `s` is the raw caller-layout gradient (944-wide for the
    /// 944 class; shorter widths slice per the named `BLOCK_END_*`
    /// bounds). SDR route only: HDR-declared inputs get
    /// [`ZensimError::HdrInputRequiresPuPath`].
    ///
    /// # Errors
    ///
    /// The C3a fused contract (all-basic-features profile,
    /// `blur_passes == 1`, matching pair dims) plus the folded
    /// extraction's own validations.
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded944_score_and_attribution(
        &self,
        source: &impl ImageSource,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        session: &mut Fused944Session,
    ) -> Result<
        (
            crate::metric::ZensimResult,
            crate::feature_v2::ZensimV2Result,
            AttributionResult,
        ),
        ZensimError,
    > {
        validate_pair(source, distorted)?;
        // ZENSIM_ATTR_PERF=1: coarse section timing (perf lever triage).
        let perf_log = std::env::var("ZENSIM_ATTR_PERF").as_deref() == Ok("1");
        let t0 = std::time::Instant::now();
        // 1) Folded-944 extraction with retention — exact features.
        let v2res = crate::feature_v2::compute_folded944_streaming_with_retention(
            source,
            distorted,
            self.max_pixels(),
            self.parallel(),
            &mut session.scratch,
            &mut session.retention,
        )?;
        let t_extract = t0.elapsed();
        // 2) Fused v1 walk — basic-block canvas + the map-profile result.
        let t1 = std::time::Instant::now();
        let fb = self.fused_basic_canvas(precomputed, distorted, s, None)?;
        let t_v1 = t1.elapsed();
        let width = distorted.width();
        let height = distorted.height();
        // 3) v2/append/append2 density from retention (same block slicing
        //    as `compute_attribution_density_full`).
        let block = |start: usize, end: usize| -> Option<&[f64]> {
            (s.len() > start).then(|| &s[start..s.len().min(end)])
        };
        let s_v2: &[f64] = block(BLOCK_END_V1_POOLS, BLOCK_END_V2).unwrap_or(&[]);
        let s_append: Option<&[f64]> = block(BLOCK_END_V2, BLOCK_END_APPEND);
        let s_append2: Option<&[f64]> = block(BLOCK_END_APPEND, BLOCK_END_APPEND2);
        // Appendix P lever 1: fully-f32 canvas assembly — the f32 basic
        // canvas plus the f32 pass-B retention density, one f64 conversion
        // at the SAT build (`from_density`, the C3a entry's own shape).
        let t2 = std::time::Instant::now();
        let mut canvas: Vec<f32> = fb.trim_to(width, height);
        if !s_v2.is_empty() || s_append.is_some() || s_append2.is_some() {
            let v2a = crate::feature_v2::compute_v2_append_attribution_from_retention(
                &session.retention,
                s_v2,
                s_append,
                s_append2,
                self.parallel(),
                width,
                height,
                &mut session.pass_b,
            );
            debug_assert_eq!(v2a.len(), width * height);
            for (c, v) in canvas.iter_mut().zip(v2a.iter()) {
                *c += *v;
            }
        }
        let t_pass_b = t2.elapsed();
        let t3 = std::time::Instant::now();
        let out = AttributionResult::from_density(canvas, width, height);
        if perf_log {
            eprintln!(
                "ATTRPERF fused944: extraction+retention {:.1} ms | v1 walk+basic {:.1} ms | pass-B f32 {:.1} ms | trim+SAT {:.1} ms",
                t_extract.as_secs_f64() * 1e3,
                t_v1.as_secs_f64() * 1e3,
                t_pass_b.as_secs_f64() * 1e3,
                t3.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok((fb.result, v2res, out))
    }
}
