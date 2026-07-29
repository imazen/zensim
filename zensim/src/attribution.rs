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
//! 2. **Blur bleed**: refining block `B` also changes signals within the blur
//!    radius outside `B` (at every scale); the density attributes a signal
//!    wholly to its own pixel. This is the residual between block sums and
//!    M2's exact feature deltas.
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

/// Production band height for the V-blur running-sum chains (must equal
/// `streaming::STRIP_INNER` so the per-band accumulator init points — and
/// therefore every per-pixel signal and pooled scalar — are bit-compatible
/// with the production feature extractor).
const BAND_ROWS: usize = 32;

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
        assert_eq!(density.len(), width * height, "density len != width*height");
        let sat = build_sat(|i| density[i] as f64, width, height);
        Self {
            density,
            sat,
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
    let w1 = width + 1;
    let mut sat = vec![0.0f64; w1 * (height + 1)];
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
    sat
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
    let mut bufs = ChannelBuffers::new(comp_pw * comp_h);
    let mut scale_density = vec![0.0f64; comp_pw * comp_h];

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
        let sd_slice = &mut scale_density[..n];
        sd_slice.fill(0.0);

        for c in 0..3 {
            let src_c = &src_planes[c][..n];
            let dst_c = &dst_planes[c][..n];
            let acc = process_channel_banded(src_c, dst_c, sw, sh, radius, &mut bufs);
            let base_k = scale * FPC * 3 + c * FPC;
            let co = SlotCoeffs::derive(s, base_k, &acc, n_f);
            own_features.extend_from_slice(&basic13_from_acc(&acc, n_f));

            // Exact-integrand combine (f64): all seven signals re-derived
            // from the SAME planes the pooled features consumed.
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
                sd_slice[i] += co.c_sd * sd
                    + co.c_sd4 * (sd2 * sd2)
                    + co.c_sd2 * sd2
                    + co.c_art * art
                    + co.c_art4 * (a2 * a2)
                    + co.c_art2 * a2
                    + co.c_det * det
                    + co.c_det4 * (dt2 * dt2)
                    + co.c_det2 * dt2
                    + co.c_mse * (pd * pd)
                    + co.c_hfe * e_hf
                    + co.c_hfm * (d1 - d2);
            }
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
    /// density plus exact-integrand densities for the v2 (`f372-719`) and —
    /// when `s` extends past 720 — append (`f720-923`) blocks, built by
    /// [`crate::feature_v2::compute_v2_append_attribution`]'s replication of
    /// the production kernels. `s` is the raw full-layout gradient
    /// (`∂score/∂f_k`, 720- or 924-wide; the `f156-371` peak/masked/iw block
    /// is not spatializable and is ignored, as in the harness's
    /// structural-zero handling).
    ///
    /// Remaining documented approximations: first-order integrands
    /// throughout, blur bleed unmodeled, finalize clamps treated as inert,
    /// blockiness steps split 50/50 across their pixel pair, reference-only
    /// slots (fragility, grad-src-mean) exactly zero. See the module docs
    /// here and the `feature_v2` attribution section.
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
        let s_v2: &[f64] = if s.len() > 372 {
            &s[372..s.len().min(720)]
        } else {
            &[]
        };
        let s_append: Option<&[f64]> = if s.len() > 720 {
            Some(&s[720..s.len().min(924)])
        } else {
            None
        };
        if !s_v2.is_empty() || s_append.is_some() {
            let v2a = crate::feature_v2::compute_v2_append_attribution(
                source,
                distorted,
                s_v2,
                s_append,
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
}
