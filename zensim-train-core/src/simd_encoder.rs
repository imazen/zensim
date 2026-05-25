//! SIMD-accelerated encoder + dot-product kernels for the
//! per-sample α head (and any future trainer head that needs the same
//! `h_pre = b1 + Σ_i x[i]·W[i,:]` → LeakyReLU → linear-head pattern).
//!
//! The kernels are bit-identical to the scalar reference modulo
//! FMA fusion (one rounding instead of two, drift < 1 ULP per fused
//! op). Cumulative drift on a 372-feature × 128-hidden encoder forward
//! is bounded by ~1e-12 relative — well inside the trainer's RankNet
//! loss precision budget.
//!
//! ## Dispatch tree
//!
//! All four entry points use [`archmage::incant!`] with the default
//! tier ladder (with `+v4` forcing AVX-512 dispatch on Zen 4 / Sapphire
//! Rapids regardless of whether this crate's transitive features turn
//! on `avx512` for `magetypes`):
//!
//! - `X64V4Token` (AVX-512F) → `f64x8` lane width
//! - `X64V3Token` (AVX2 + FMA) → `f64x4` lane width
//! - `NeonToken` (aarch64) → `f64x2` polyfilled to `GenericF64x4`
//! - `Wasm128Token` (wasm32 + simd128) → `f64x2` polyfilled
//! - `ScalarToken` → scalar fallback (also the bit-identity oracle)
//!
//! ## Hot dimension
//!
//! `n_hidden = 128` is the V_X production size, a clean multiple of
//! both 8 (AVX-512) and 4 (AVX2). The fast path never enters the tail
//! loop at production shape. The tail loop is present for unit-test
//! shapes (`n_hidden = 6, 8`) and any future architecture experiment
//! that drifts off the power-of-two grid.
//!
//! ## Sparse-x short-circuit
//!
//! Standardized feature vectors can carry exact zeros (e.g., quantile-
//! binned features with only some bins populated). Both `forward` and
//! `backprop_layer1` short-circuit on `x[i] == 0.0` to skip the inner
//! 128-wide FMA chain — preserves the scalar reference's behavior and
//! is a measurable win in practice.

// SIMD inner loops index `x` by `i` because the matching row slice in
// `w1` / `gw1` lives at `i * n_hidden..(i + 1) * n_hidden`; the
// `iter().enumerate()` form would require either parallel iteration of
// both slices (no early-zero short-circuit) or unsafe slice arithmetic.
// Mirrors the same allow in `zensim-validate/src/simd_mlp.rs`.
#![allow(clippy::needless_range_loop)]

use archmage::{incant, magetypes};
use magetypes::simd::generic::{f64x4 as GenericF64x4, f64x8 as GenericF64x8};

// =============================================================================
// Public dispatch entry points.
// =============================================================================

/// Encoder forward: `h_pre = b1 + Σ_i x[i] · w1[i*N..(i+1)*N]`, then
/// `h[j] = h_pre[j] >= 0 ? h_pre[j] : leaky_alpha · h_pre[j]`.
///
/// Returns `(h_pre, h)` — both length `n_hidden`. Both buffers are
/// fully written; no uninit memory escapes.
///
/// **Precondition:** `w1.len() == n_features * n_hidden`,
/// `b1.len() == n_hidden`, `x.len() == n_features`.
#[inline]
pub fn encoder_forward(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f64,
) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x.len(), n_features);
    debug_assert_eq!(b1.len(), n_hidden);
    debug_assert_eq!(w1.len(), n_features * n_hidden);

    let mut h_pre = b1.to_vec();
    incant!(accumulate_rows(x, w1, &mut h_pre, n_features, n_hidden), [+v4]);
    let h = incant!(apply_leaky_relu(&h_pre, leaky_alpha), [+v4]);
    (h_pre, h)
}

/// Encoder layer-1 backprop: for each `i` where `x[i] != 0.0`,
///   `gw1[i*N..(i+1)*N] += x[i] · dl_dh_pre`
/// then unconditionally `gb1 += dl_dh_pre` (one full add over
/// `n_hidden`).
///
/// **Precondition:** `gw1.len() == n_features * n_hidden`,
/// `gb1.len() == n_hidden`, `dl_dh_pre.len() == n_hidden`,
/// `x.len() == n_features`.
#[inline]
pub fn encoder_backprop_layer1(
    x: &[f64],
    dl_dh_pre: &[f64],
    gw1: &mut [f64],
    gb1: &mut [f64],
    n_features: usize,
    n_hidden: usize,
) {
    debug_assert_eq!(x.len(), n_features);
    debug_assert_eq!(dl_dh_pre.len(), n_hidden);
    debug_assert_eq!(gw1.len(), n_features * n_hidden);
    debug_assert_eq!(gb1.len(), n_hidden);

    incant!(
        scatter_outer_product(x, dl_dh_pre, gw1, n_features, n_hidden),
        [+v4]
    );
    incant!(accumulate_in_place(gb1, dl_dh_pre), [+v4]);
}

/// LeakyReLU back-route: `out[j] = h_pre[j] >= 0 ? dl_dh[j] : leaky · dl_dh[j]`.
///
/// Allocates and returns a fresh `Vec<f64>` of length `dl_dh.len()`.
#[inline]
pub fn leaky_relu_backward(dl_dh: &[f64], h_pre: &[f64], leaky_alpha: f64) -> Vec<f64> {
    debug_assert_eq!(dl_dh.len(), h_pre.len());
    incant!(leaky_relu_back(dl_dh, h_pre, leaky_alpha), [+v4])
}

/// Linear head: `b + Σ_j h[j] · w[j]`. The pattern used by the rank
/// head, the per-sample α-head logit, and any future scalar-output
/// linear-on-hidden head.
///
/// **Precondition:** `h.len() == w.len()`.
#[inline]
pub fn dot_bias(h: &[f64], w: &[f64], bias: f64) -> f64 {
    debug_assert_eq!(h.len(), w.len());
    bias + incant!(dot_product(h, w), [+v4])
}

// =============================================================================
// Scalar reference paths — bit-identity oracle for the test suite.
//
// These are compiled only under `#[cfg(test)]` so they don't add
// codegen weight to production builds. Production code paths the
// dispatched entry points exclusively.
// =============================================================================

#[cfg(test)]
/// Scalar reference for [`encoder_forward`]'s accumulation phase.
/// Equivalent to the original `for i in 0..n_features { ... }` loop.
#[inline]
pub(crate) fn accumulate_rows_ref(
    x: &[f64],
    w1: &[f64],
    h_pre: &mut [f64],
    n_features: usize,
    n_hidden: usize,
) {
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for (acc, &w) in h_pre.iter_mut().zip(row.iter()) {
            *acc += s * w;
        }
    }
}

#[cfg(test)]
/// Scalar reference for [`encoder_forward`]'s LeakyReLU phase.
#[inline]
pub(crate) fn apply_leaky_relu_ref(h_pre: &[f64], leaky_alpha: f64) -> Vec<f64> {
    h_pre
        .iter()
        .map(|&v| if v >= 0.0 { v } else { leaky_alpha * v })
        .collect()
}

#[cfg(test)]
/// Scalar reference for [`encoder_backprop_layer1`]'s gw1 update.
#[inline]
pub(crate) fn scatter_outer_product_ref(
    x: &[f64],
    dl_dh_pre: &[f64],
    gw1: &mut [f64],
    n_features: usize,
    n_hidden: usize,
) {
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &mut gw1[i * n_hidden..(i + 1) * n_hidden];
        for (g, &dh) in row.iter_mut().zip(dl_dh_pre.iter()) {
            *g += s * dh;
        }
    }
}

#[cfg(test)]
/// Scalar reference for `gb1 += dl_dh_pre`.
#[inline]
pub(crate) fn accumulate_in_place_ref(gb1: &mut [f64], dl_dh_pre: &[f64]) {
    for (g, &dh) in gb1.iter_mut().zip(dl_dh_pre.iter()) {
        *g += dh;
    }
}

#[cfg(test)]
/// Scalar reference for [`leaky_relu_backward`].
#[inline]
pub(crate) fn leaky_relu_back_ref(dl_dh: &[f64], h_pre: &[f64], leaky_alpha: f64) -> Vec<f64> {
    dl_dh
        .iter()
        .zip(h_pre.iter())
        .map(|(&dh, &hp)| if hp >= 0.0 { dh } else { leaky_alpha * dh })
        .collect()
}

#[cfg(test)]
/// Scalar reference for [`dot_bias`] (sans the bias term — caller adds).
#[inline]
pub(crate) fn dot_product_ref(h: &[f64], w: &[f64]) -> f64 {
    h.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum()
}

// =============================================================================
// AVX-512 path (Zen 4 / Sapphire Rapids) — f64x8.
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn accumulate_rows_v4(
    token: archmage::X64V4Token,
    x: &[f64],
    w1: &[f64],
    h_pre: &mut [f64],
    n_features: usize,
    n_hidden: usize,
) {
    let chunks = n_hidden / 8;
    let tail = chunks * 8;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let s_v = magetypes::simd::generic::f64x8::splat(token, s);
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for c in 0..chunks {
            let off = c * 8;
            let h_block: &mut [f64; 8] = (&mut h_pre[off..off + 8]).try_into().unwrap();
            let w_block: &[f64; 8] = (&row[off..off + 8]).try_into().unwrap();
            let h_v = GenericF64x8::load(token, h_block);
            let w_v = GenericF64x8::load(token, w_block);
            let new_v = s_v.mul_add(w_v, h_v);
            new_v.store(h_block);
        }
        // Tail (n_hidden % 8 != 0) — fixed shapes never hit this in
        // production; small test cases do.
        for j in tail..n_hidden {
            h_pre[j] += s * row[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn apply_leaky_relu_v4(token: archmage::X64V4Token, h_pre: &[f64], leaky_alpha: f64) -> Vec<f64> {
    let mut h = vec![0.0f64; h_pre.len()];
    let chunks = h_pre.len() / 8;
    let tail = chunks * 8;
    let leaky_v = GenericF64x8::splat(token, leaky_alpha);
    let zero_v = GenericF64x8::zero(token);
    for c in 0..chunks {
        let off = c * 8;
        let in_block: &[f64; 8] = (&h_pre[off..off + 8]).try_into().unwrap();
        let out_block: &mut [f64; 8] = (&mut h[off..off + 8]).try_into().unwrap();
        let v = GenericF64x8::load(token, in_block);
        let scaled = v * leaky_v;
        // mask: lane is set when v < 0 (negative side). `blend` picks
        // `scaled` on those lanes and `v` on non-negative lanes,
        // matching `if v >= 0 { v } else { alpha * v }`.
        let neg_mask = v.simd_lt(zero_v);
        GenericF64x8::blend(neg_mask, scaled, v).store(out_block);
    }
    for j in tail..h_pre.len() {
        let v = h_pre[j];
        h[j] = if v >= 0.0 { v } else { leaky_alpha * v };
    }
    h
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn scatter_outer_product_v4(
    token: archmage::X64V4Token,
    x: &[f64],
    dl_dh_pre: &[f64],
    gw1: &mut [f64],
    n_features: usize,
    n_hidden: usize,
) {
    let chunks = n_hidden / 8;
    let tail = chunks * 8;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let s_v = GenericF64x8::splat(token, s);
        let row_off = i * n_hidden;
        for c in 0..chunks {
            let off = row_off + c * 8;
            let dh_off = c * 8;
            let g_block: &mut [f64; 8] = (&mut gw1[off..off + 8]).try_into().unwrap();
            let dh_block: &[f64; 8] = (&dl_dh_pre[dh_off..dh_off + 8]).try_into().unwrap();
            let g_v = GenericF64x8::load(token, g_block);
            let dh_v = GenericF64x8::load(token, dh_block);
            let new_v = s_v.mul_add(dh_v, g_v);
            new_v.store(g_block);
        }
        for j in tail..n_hidden {
            gw1[row_off + j] += s * dl_dh_pre[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn accumulate_in_place_v4(token: archmage::X64V4Token, gb1: &mut [f64], dl_dh_pre: &[f64]) {
    let chunks = gb1.len() / 8;
    let tail = chunks * 8;
    for c in 0..chunks {
        let off = c * 8;
        let g_block: &mut [f64; 8] = (&mut gb1[off..off + 8]).try_into().unwrap();
        let dh_block: &[f64; 8] = (&dl_dh_pre[off..off + 8]).try_into().unwrap();
        let g_v = GenericF64x8::load(token, g_block);
        let dh_v = GenericF64x8::load(token, dh_block);
        (g_v + dh_v).store(g_block);
    }
    for j in tail..gb1.len() {
        gb1[j] += dl_dh_pre[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn leaky_relu_back_v4(
    token: archmage::X64V4Token,
    dl_dh: &[f64],
    h_pre: &[f64],
    leaky_alpha: f64,
) -> Vec<f64> {
    let n = dl_dh.len();
    let mut out = vec![0.0f64; n];
    let chunks = n / 8;
    let tail = chunks * 8;
    let leaky_v = GenericF64x8::splat(token, leaky_alpha);
    let zero_v = GenericF64x8::zero(token);
    for c in 0..chunks {
        let off = c * 8;
        let dh_block: &[f64; 8] = (&dl_dh[off..off + 8]).try_into().unwrap();
        let hp_block: &[f64; 8] = (&h_pre[off..off + 8]).try_into().unwrap();
        let out_block: &mut [f64; 8] = (&mut out[off..off + 8]).try_into().unwrap();
        let dh_v = GenericF64x8::load(token, dh_block);
        let hp_v = GenericF64x8::load(token, hp_block);
        let scaled = dh_v * leaky_v;
        let neg_mask = hp_v.simd_lt(zero_v);
        GenericF64x8::blend(neg_mask, scaled, dh_v).store(out_block);
    }
    for j in tail..n {
        out[j] = if h_pre[j] >= 0.0 {
            dl_dh[j]
        } else {
            leaky_alpha * dl_dh[j]
        };
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn dot_product_v4(token: archmage::X64V4Token, h: &[f64], w: &[f64]) -> f64 {
    let chunks = h.len() / 8;
    let tail = chunks * 8;
    let mut acc = GenericF64x8::zero(token);
    for c in 0..chunks {
        let off = c * 8;
        let h_block: &[f64; 8] = (&h[off..off + 8]).try_into().unwrap();
        let w_block: &[f64; 8] = (&w[off..off + 8]).try_into().unwrap();
        let h_v = GenericF64x8::load(token, h_block);
        let w_v = GenericF64x8::load(token, w_block);
        acc = h_v.mul_add(w_v, acc);
    }
    // Pairwise horizontal sum to keep the reduce order deterministic.
    let mut lanes = [0.0f64; 8];
    acc.store(&mut lanes);
    let lane_sum = (lanes[0] + lanes[1])
        + (lanes[2] + lanes[3])
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    let mut tail_sum = 0.0;
    for j in tail..h.len() {
        tail_sum += h[j] * w[j];
    }
    lane_sum + tail_sum
}

// =============================================================================
// AVX2 path (Haswell..Zen 3, also Sandy Bridge with fma3) — f64x4.
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn accumulate_rows_v3(
    token: archmage::X64V3Token,
    x: &[f64],
    w1: &[f64],
    h_pre: &mut [f64],
    n_features: usize,
    n_hidden: usize,
) {
    let chunks = n_hidden / 4;
    let tail = chunks * 4;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let s_v = GenericF64x4::splat(token, s);
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for c in 0..chunks {
            let off = c * 4;
            let h_block: &mut [f64; 4] = (&mut h_pre[off..off + 4]).try_into().unwrap();
            let w_block: &[f64; 4] = (&row[off..off + 4]).try_into().unwrap();
            let h_v = GenericF64x4::load(token, h_block);
            let w_v = GenericF64x4::load(token, w_block);
            s_v.mul_add(w_v, h_v).store(h_block);
        }
        for j in tail..n_hidden {
            h_pre[j] += s * row[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn apply_leaky_relu_v3(token: archmage::X64V3Token, h_pre: &[f64], leaky_alpha: f64) -> Vec<f64> {
    let mut h = vec![0.0f64; h_pre.len()];
    let chunks = h_pre.len() / 4;
    let tail = chunks * 4;
    let leaky_v = GenericF64x4::splat(token, leaky_alpha);
    let zero_v = GenericF64x4::zero(token);
    for c in 0..chunks {
        let off = c * 4;
        let in_block: &[f64; 4] = (&h_pre[off..off + 4]).try_into().unwrap();
        let out_block: &mut [f64; 4] = (&mut h[off..off + 4]).try_into().unwrap();
        let v = GenericF64x4::load(token, in_block);
        let scaled = v * leaky_v;
        let neg_mask = v.simd_lt(zero_v);
        GenericF64x4::blend(neg_mask, scaled, v).store(out_block);
    }
    for j in tail..h_pre.len() {
        let v = h_pre[j];
        h[j] = if v >= 0.0 { v } else { leaky_alpha * v };
    }
    h
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn scatter_outer_product_v3(
    token: archmage::X64V3Token,
    x: &[f64],
    dl_dh_pre: &[f64],
    gw1: &mut [f64],
    n_features: usize,
    n_hidden: usize,
) {
    let chunks = n_hidden / 4;
    let tail = chunks * 4;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let s_v = GenericF64x4::splat(token, s);
        let row_off = i * n_hidden;
        for c in 0..chunks {
            let off = row_off + c * 4;
            let dh_off = c * 4;
            let g_block: &mut [f64; 4] = (&mut gw1[off..off + 4]).try_into().unwrap();
            let dh_block: &[f64; 4] = (&dl_dh_pre[dh_off..dh_off + 4]).try_into().unwrap();
            let g_v = GenericF64x4::load(token, g_block);
            let dh_v = GenericF64x4::load(token, dh_block);
            s_v.mul_add(dh_v, g_v).store(g_block);
        }
        for j in tail..n_hidden {
            gw1[row_off + j] += s * dl_dh_pre[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn accumulate_in_place_v3(token: archmage::X64V3Token, gb1: &mut [f64], dl_dh_pre: &[f64]) {
    let chunks = gb1.len() / 4;
    let tail = chunks * 4;
    for c in 0..chunks {
        let off = c * 4;
        let g_block: &mut [f64; 4] = (&mut gb1[off..off + 4]).try_into().unwrap();
        let dh_block: &[f64; 4] = (&dl_dh_pre[off..off + 4]).try_into().unwrap();
        let g_v = GenericF64x4::load(token, g_block);
        let dh_v = GenericF64x4::load(token, dh_block);
        (g_v + dh_v).store(g_block);
    }
    for j in tail..gb1.len() {
        gb1[j] += dl_dh_pre[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn leaky_relu_back_v3(
    token: archmage::X64V3Token,
    dl_dh: &[f64],
    h_pre: &[f64],
    leaky_alpha: f64,
) -> Vec<f64> {
    let n = dl_dh.len();
    let mut out = vec![0.0f64; n];
    let chunks = n / 4;
    let tail = chunks * 4;
    let leaky_v = GenericF64x4::splat(token, leaky_alpha);
    let zero_v = GenericF64x4::zero(token);
    for c in 0..chunks {
        let off = c * 4;
        let dh_block: &[f64; 4] = (&dl_dh[off..off + 4]).try_into().unwrap();
        let hp_block: &[f64; 4] = (&h_pre[off..off + 4]).try_into().unwrap();
        let out_block: &mut [f64; 4] = (&mut out[off..off + 4]).try_into().unwrap();
        let dh_v = GenericF64x4::load(token, dh_block);
        let hp_v = GenericF64x4::load(token, hp_block);
        let scaled = dh_v * leaky_v;
        let neg_mask = hp_v.simd_lt(zero_v);
        GenericF64x4::blend(neg_mask, scaled, dh_v).store(out_block);
    }
    for j in tail..n {
        out[j] = if h_pre[j] >= 0.0 {
            dl_dh[j]
        } else {
            leaky_alpha * dl_dh[j]
        };
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn dot_product_v3(token: archmage::X64V3Token, h: &[f64], w: &[f64]) -> f64 {
    let chunks = h.len() / 4;
    let tail = chunks * 4;
    let mut acc = GenericF64x4::zero(token);
    for c in 0..chunks {
        let off = c * 4;
        let h_block: &[f64; 4] = (&h[off..off + 4]).try_into().unwrap();
        let w_block: &[f64; 4] = (&w[off..off + 4]).try_into().unwrap();
        let h_v = GenericF64x4::load(token, h_block);
        let w_v = GenericF64x4::load(token, w_block);
        acc = h_v.mul_add(w_v, acc);
    }
    let mut lanes = [0.0f64; 4];
    acc.store(&mut lanes);
    let lane_sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    let mut tail_sum = 0.0;
    for j in tail..h.len() {
        tail_sum += h[j] * w[j];
    }
    lane_sum + tail_sum
}

// =============================================================================
// NEON / WASM-SIMD / scalar polyfill — single `#[magetypes]` body
// shared across all three. The generic `GenericF64x4<Token>` polyfills
// to the platform's native width (2 lanes on NEON / wasm128, scalar on
// the no-SIMD fallback).
// =============================================================================

#[magetypes(neon, wasm128, scalar)]
fn accumulate_rows(
    token: Token,
    x: &[f64],
    w1: &[f64],
    h_pre: &mut [f64],
    n_features: usize,
    n_hidden: usize,
) {
    #[allow(non_camel_case_types)]
    type f64x4 = GenericF64x4<Token>;
    let chunks = n_hidden / 4;
    let tail = chunks * 4;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let s_v = f64x4::splat(token, s);
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for c in 0..chunks {
            let off = c * 4;
            let h_block: &mut [f64; 4] = (&mut h_pre[off..off + 4]).try_into().unwrap();
            let w_block: &[f64; 4] = (&row[off..off + 4]).try_into().unwrap();
            let h_v = f64x4::load(token, h_block);
            let w_v = f64x4::load(token, w_block);
            s_v.mul_add(w_v, h_v).store(h_block);
        }
        for j in tail..n_hidden {
            h_pre[j] += s * row[j];
        }
    }
}

#[magetypes(neon, wasm128, scalar)]
fn apply_leaky_relu(token: Token, h_pre: &[f64], leaky_alpha: f64) -> Vec<f64> {
    #[allow(non_camel_case_types)]
    type f64x4 = GenericF64x4<Token>;
    let mut h = vec![0.0f64; h_pre.len()];
    let chunks = h_pre.len() / 4;
    let tail = chunks * 4;
    let leaky_v = f64x4::splat(token, leaky_alpha);
    let zero_v = f64x4::zero(token);
    for c in 0..chunks {
        let off = c * 4;
        let in_block: &[f64; 4] = (&h_pre[off..off + 4]).try_into().unwrap();
        let out_block: &mut [f64; 4] = (&mut h[off..off + 4]).try_into().unwrap();
        let v = f64x4::load(token, in_block);
        let scaled = v * leaky_v;
        let neg_mask = v.simd_lt(zero_v);
        f64x4::blend(neg_mask, scaled, v).store(out_block);
    }
    for j in tail..h_pre.len() {
        let v = h_pre[j];
        h[j] = if v >= 0.0 { v } else { leaky_alpha * v };
    }
    h
}

#[magetypes(neon, wasm128, scalar)]
fn scatter_outer_product(
    token: Token,
    x: &[f64],
    dl_dh_pre: &[f64],
    gw1: &mut [f64],
    n_features: usize,
    n_hidden: usize,
) {
    #[allow(non_camel_case_types)]
    type f64x4 = GenericF64x4<Token>;
    let chunks = n_hidden / 4;
    let tail = chunks * 4;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let s_v = f64x4::splat(token, s);
        let row_off = i * n_hidden;
        for c in 0..chunks {
            let off = row_off + c * 4;
            let dh_off = c * 4;
            let g_block: &mut [f64; 4] = (&mut gw1[off..off + 4]).try_into().unwrap();
            let dh_block: &[f64; 4] = (&dl_dh_pre[dh_off..dh_off + 4]).try_into().unwrap();
            let g_v = f64x4::load(token, g_block);
            let dh_v = f64x4::load(token, dh_block);
            s_v.mul_add(dh_v, g_v).store(g_block);
        }
        for j in tail..n_hidden {
            gw1[row_off + j] += s * dl_dh_pre[j];
        }
    }
}

#[magetypes(neon, wasm128, scalar)]
fn accumulate_in_place(token: Token, gb1: &mut [f64], dl_dh_pre: &[f64]) {
    #[allow(non_camel_case_types)]
    type f64x4 = GenericF64x4<Token>;
    let chunks = gb1.len() / 4;
    let tail = chunks * 4;
    for c in 0..chunks {
        let off = c * 4;
        let g_block: &mut [f64; 4] = (&mut gb1[off..off + 4]).try_into().unwrap();
        let dh_block: &[f64; 4] = (&dl_dh_pre[off..off + 4]).try_into().unwrap();
        let g_v = f64x4::load(token, g_block);
        let dh_v = f64x4::load(token, dh_block);
        (g_v + dh_v).store(g_block);
    }
    for j in tail..gb1.len() {
        gb1[j] += dl_dh_pre[j];
    }
}

#[magetypes(neon, wasm128, scalar)]
fn leaky_relu_back(token: Token, dl_dh: &[f64], h_pre: &[f64], leaky_alpha: f64) -> Vec<f64> {
    #[allow(non_camel_case_types)]
    type f64x4 = GenericF64x4<Token>;
    let n = dl_dh.len();
    let mut out = vec![0.0f64; n];
    let chunks = n / 4;
    let tail = chunks * 4;
    let leaky_v = f64x4::splat(token, leaky_alpha);
    let zero_v = f64x4::zero(token);
    for c in 0..chunks {
        let off = c * 4;
        let dh_block: &[f64; 4] = (&dl_dh[off..off + 4]).try_into().unwrap();
        let hp_block: &[f64; 4] = (&h_pre[off..off + 4]).try_into().unwrap();
        let out_block: &mut [f64; 4] = (&mut out[off..off + 4]).try_into().unwrap();
        let dh_v = f64x4::load(token, dh_block);
        let hp_v = f64x4::load(token, hp_block);
        let scaled = dh_v * leaky_v;
        let neg_mask = hp_v.simd_lt(zero_v);
        f64x4::blend(neg_mask, scaled, dh_v).store(out_block);
    }
    for j in tail..n {
        out[j] = if h_pre[j] >= 0.0 {
            dl_dh[j]
        } else {
            leaky_alpha * dl_dh[j]
        };
    }
    out
}

#[magetypes(neon, wasm128, scalar)]
fn dot_product(token: Token, h: &[f64], w: &[f64]) -> f64 {
    #[allow(non_camel_case_types)]
    type f64x4 = GenericF64x4<Token>;
    let chunks = h.len() / 4;
    let tail = chunks * 4;
    let mut acc = f64x4::zero(token);
    for c in 0..chunks {
        let off = c * 4;
        let h_block: &[f64; 4] = (&h[off..off + 4]).try_into().unwrap();
        let w_block: &[f64; 4] = (&w[off..off + 4]).try_into().unwrap();
        let h_v = f64x4::load(token, h_block);
        let w_v = f64x4::load(token, w_block);
        acc = h_v.mul_add(w_v, acc);
    }
    let mut lanes = [0.0f64; 4];
    acc.store(&mut lanes);
    let lane_sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    let mut tail_sum = 0.0;
    for j in tail..h.len() {
        tail_sum += h[j] * w[j];
    }
    lane_sum + tail_sum
}

// =============================================================================
// Tests — bit-equivalence between dispatch entry points and the
// scalar references at production shape + tail-only shapes.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic xorshift PRNG. Kept self-contained so the test
    /// suite never depends on global RNG state.
    struct Xs64(u64);
    impl Xs64 {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_unit(&mut self) -> f64 {
            (self.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0
        }
    }

    fn random_vec(rng: &mut Xs64, n: usize) -> Vec<f64> {
        (0..n).map(|_| rng.next_unit()).collect()
    }

    fn random_sparse(rng: &mut Xs64, n: usize, zero_frac: f64) -> Vec<f64> {
        let thresh = 2.0 * zero_frac - 1.0;
        (0..n)
            .map(|_| {
                if rng.next_unit() < thresh {
                    0.0
                } else {
                    rng.next_unit()
                }
            })
            .collect()
    }

    fn assert_close(a: &[f64], b: &[f64], rel_tol: f64, label: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", label);
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            let denom = av.abs().max(bv.abs()).max(1.0);
            let rel = (av - bv).abs() / denom;
            assert!(
                rel < rel_tol,
                "{}[{}] mismatch: scalar={} simd={} rel={:e}",
                label,
                i,
                av,
                bv,
                rel
            );
        }
    }

    /// Production shape: 372 features × 128 hidden, sparse-x at 30%.
    /// Mirrors the trainer's hot path.
    #[test]
    fn encoder_forward_matches_scalar_372x128() {
        let n_features = 372;
        let n_hidden = 128;
        let alpha = 0.01;
        let mut rng = Xs64::new(0xCAFE_BEEF_1234_5678);
        let x = random_sparse(&mut rng, n_features, 0.3);
        let w1 = random_vec(&mut rng, n_features * n_hidden);
        let b1 = random_vec(&mut rng, n_hidden);

        let mut h_pre_ref = b1.clone();
        accumulate_rows_ref(&x, &w1, &mut h_pre_ref, n_features, n_hidden);
        let h_ref = apply_leaky_relu_ref(&h_pre_ref, alpha);

        let (h_pre, h) = encoder_forward(&x, &w1, &b1, n_features, n_hidden, alpha);

        // h_pre tolerates FMA fusion drift (up to 372 mul-adds per lane).
        assert_close(&h_pre_ref, &h_pre, 1e-12, "h_pre");
        assert_close(&h_ref, &h, 1e-12, "h");
    }

    /// Tail-only shapes: n_hidden < lane width forces 100% scalar mop-up.
    /// Exercises every dispatch path's tail branch.
    #[test]
    fn encoder_forward_matches_scalar_tail_only() {
        for &(n_features, n_hidden) in &[(8usize, 1usize), (8, 3), (8, 5), (8, 7)] {
            let alpha = 0.01;
            let seed = 0xDEAD_BEEF ^ (n_features * 1000 + n_hidden) as u64;
            let mut rng = Xs64::new(seed);
            let x = random_sparse(&mut rng, n_features, 0.5);
            let w1 = random_vec(&mut rng, n_features * n_hidden);
            let b1 = random_vec(&mut rng, n_hidden);

            let mut h_pre_ref = b1.clone();
            accumulate_rows_ref(&x, &w1, &mut h_pre_ref, n_features, n_hidden);
            let h_ref = apply_leaky_relu_ref(&h_pre_ref, alpha);

            let (h_pre, h) = encoder_forward(&x, &w1, &b1, n_features, n_hidden, alpha);

            assert_close(&h_pre_ref, &h_pre, 1e-12, "h_pre");
            assert_close(&h_ref, &h, 1e-12, "h");
        }
    }

    /// All-zeros x: forward returns b1 unchanged + LeakyReLU of b1.
    #[test]
    fn encoder_forward_all_zero_input() {
        let n_features = 32;
        let n_hidden = 16;
        let alpha = 0.01;
        let mut rng = Xs64::new(42);
        let x = vec![0.0; n_features];
        let w1 = random_vec(&mut rng, n_features * n_hidden);
        let b1 = random_vec(&mut rng, n_hidden);

        let (h_pre, h) = encoder_forward(&x, &w1, &b1, n_features, n_hidden, alpha);
        assert_close(&b1, &h_pre, 1e-15, "h_pre == b1");
        let h_ref = apply_leaky_relu_ref(&b1, alpha);
        assert_close(&h_ref, &h, 1e-15, "h == leaky(b1)");
    }

    /// Backprop layer-1: gw1 + gb1 update at production shape.
    #[test]
    fn encoder_backprop_layer1_matches_scalar_372x128() {
        let n_features = 372;
        let n_hidden = 128;
        let mut rng = Xs64::new(0x1234_5678_AABB_CCDD);
        let x = random_sparse(&mut rng, n_features, 0.3);
        let dl_dh_pre = random_vec(&mut rng, n_hidden);

        // Seed gw1 / gb1 with non-zero values to verify ACCUMULATION,
        // not initial write.
        let gw1_seed = random_vec(&mut rng, n_features * n_hidden);
        let gb1_seed = random_vec(&mut rng, n_hidden);
        let mut gw1_ref = gw1_seed.clone();
        let mut gb1_ref = gb1_seed.clone();
        scatter_outer_product_ref(&x, &dl_dh_pre, &mut gw1_ref, n_features, n_hidden);
        accumulate_in_place_ref(&mut gb1_ref, &dl_dh_pre);

        let mut gw1 = gw1_seed;
        let mut gb1 = gb1_seed;
        encoder_backprop_layer1(&x, &dl_dh_pre, &mut gw1, &mut gb1, n_features, n_hidden);

        assert_close(&gw1_ref, &gw1, 1e-12, "gw1");
        assert_close(&gb1_ref, &gb1, 1e-12, "gb1");
    }

    /// Backprop layer-1 with all-zero x: gw1 untouched, gb1 += dl_dh_pre.
    #[test]
    fn encoder_backprop_layer1_all_zero_input() {
        let n_features = 16;
        let n_hidden = 16;
        let mut rng = Xs64::new(7);
        let x = vec![0.0; n_features];
        let dl_dh_pre = random_vec(&mut rng, n_hidden);
        let gw1_seed = random_vec(&mut rng, n_features * n_hidden);
        let gb1_seed = random_vec(&mut rng, n_hidden);

        let mut gw1 = gw1_seed.clone();
        let mut gb1 = gb1_seed.clone();
        encoder_backprop_layer1(&x, &dl_dh_pre, &mut gw1, &mut gb1, n_features, n_hidden);

        // gw1 untouched; gb1 = seed + dl_dh_pre.
        assert_close(&gw1_seed, &gw1, 1e-15, "gw1 untouched");
        let gb1_ref: Vec<f64> = gb1_seed
            .iter()
            .zip(dl_dh_pre.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        assert_close(&gb1_ref, &gb1, 1e-15, "gb1 += dl_dh_pre");
    }

    /// LeakyReLU back-route at production hidden width.
    #[test]
    fn leaky_relu_back_matches_scalar_128() {
        let n = 128;
        let leaky = 0.01;
        let mut rng = Xs64::new(0xBABE_1010);
        let dl_dh = random_vec(&mut rng, n);
        let h_pre = random_vec(&mut rng, n);

        let scalar = leaky_relu_back_ref(&dl_dh, &h_pre, leaky);
        let simd = leaky_relu_backward(&dl_dh, &h_pre, leaky);
        assert_close(&scalar, &simd, 1e-15, "leaky_relu_back");
    }

    /// Dot product at production hidden width (rank / α-head pattern).
    #[test]
    fn dot_bias_matches_scalar_128() {
        let n = 128;
        let mut rng = Xs64::new(0xFEED_BEEF);
        let h = random_vec(&mut rng, n);
        let w = random_vec(&mut rng, n);
        let bias = rng.next_unit();

        let scalar = bias + dot_product_ref(&h, &w);
        let simd = dot_bias(&h, &w, bias);
        let denom = scalar.abs().max(simd.abs()).max(1.0);
        let rel = (scalar - simd).abs() / denom;
        assert!(
            rel < 1e-13,
            "dot_bias mismatch: scalar={} simd={} rel={:e}",
            scalar,
            simd,
            rel
        );
    }

    /// Dot product on uneven length exercises the scalar tail.
    #[test]
    fn dot_bias_matches_scalar_uneven() {
        for &n in &[1usize, 3, 5, 7, 9, 31, 127] {
            let mut rng = Xs64::new(0xABCD_1234 ^ n as u64);
            let h = random_vec(&mut rng, n);
            let w = random_vec(&mut rng, n);
            let scalar = dot_product_ref(&h, &w);
            let simd = dot_bias(&h, &w, 0.0);
            let denom = scalar.abs().max(simd.abs()).max(1.0);
            let rel = (scalar - simd).abs() / denom;
            assert!(rel < 1e-13, "n={}: scalar={} simd={}", n, scalar, simd);
        }
    }

    /// Microbenchmark — `#[ignore]` so it doesn't run on default `cargo
    /// test` (it spends ~1 s wall on a typical 7950X). Run with
    /// `cargo test --release simd_encoder::tests::encoder_speedup_vs_scalar -- --ignored --nocapture`
    /// to print measured throughput for both paths.
    ///
    /// Reports `(scalar µs/iter, simd µs/iter, speedup×)` so a future
    /// regression — say, an inadvertent inline-disable on the SIMD
    /// kernel — would show up as a < 1.5× ratio. We don't assert a
    /// specific ratio because the dispatch chooses based on runtime CPU
    /// detection (AVX-512 vs AVX2 vs polyfill) and CI runners vary.
    #[test]
    #[ignore = "performance microbench — run with --ignored --nocapture"]
    fn encoder_speedup_vs_scalar() {
        use std::time::Instant;
        let n_features = 372;
        let n_hidden = 128;
        let alpha = 0.01;
        let n_iters = 5_000;

        let mut rng = Xs64::new(0xBEEF_1234_5678);
        let x = random_sparse(&mut rng, n_features, 0.3);
        let w1 = random_vec(&mut rng, n_features * n_hidden);
        let b1 = random_vec(&mut rng, n_hidden);

        // Warm-up — fault-in caches + branch-predict, prevents bias.
        for _ in 0..200 {
            let mut h_pre = b1.clone();
            accumulate_rows_ref(&x, &w1, &mut h_pre, n_features, n_hidden);
            std::hint::black_box(&h_pre);
        }
        for _ in 0..200 {
            let (h_pre, _h) = encoder_forward(&x, &w1, &b1, n_features, n_hidden, alpha);
            std::hint::black_box(h_pre);
        }

        let t0 = Instant::now();
        for _ in 0..n_iters {
            let mut h_pre = b1.clone();
            accumulate_rows_ref(&x, &w1, &mut h_pre, n_features, n_hidden);
            std::hint::black_box(&h_pre);
        }
        let scalar_ns_per = t0.elapsed().as_nanos() as f64 / n_iters as f64;

        let t1 = Instant::now();
        for _ in 0..n_iters {
            let (h_pre, _h) = encoder_forward(&x, &w1, &b1, n_features, n_hidden, alpha);
            std::hint::black_box(h_pre);
        }
        let simd_ns_per = t1.elapsed().as_nanos() as f64 / n_iters as f64;

        let speedup = scalar_ns_per / simd_ns_per;
        eprintln!(
            "encoder_forward @ 372×128, 30% sparse-x:\n  \
             scalar (raw loop):   {:.2} µs/iter\n  \
             dispatched (SIMD):   {:.2} µs/iter\n  \
             speedup:             {:.2}×",
            scalar_ns_per / 1e3,
            simd_ns_per / 1e3,
            speedup,
        );
    }
}
