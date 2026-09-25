//! SIMD-accelerated MLP forward + backprop kernels.
//!
//! Drop-in replacements for the scalar `forward` and `backprop_step`
//! in `mlp_train.rs`. The dispatch tree, at function entry, picks the
//! best path supported by the current CPU:
//!
//! 1. **AVX-512F** (Zen 4, Sapphire Rapids, Ice Lake-SP) — f64x8
//!    inner loop with VFMADD231PD; ~16 mul-adds per cycle on Zen 4.
//! 2. **AVX2 + FMA** (Haswell and newer) — f64x4 inner loop with
//!    VFMADD231PD; ~8 mul-adds per cycle.
//! 3. **Scalar fallback** — identical algorithm, no intrinsics; used
//!    on non-x86_64 and pre-AVX2 boxes (i686 CI, aarch64 CI).
//!
//! Hot dimension: the 128-wide hidden layer. n_hidden=128 is the
//! V_X production size and is a clean multiple of 8 / 4 — no tail
//! handling needed on the fast path. The kernels still handle arbitrary
//! n_hidden via a scalar tail loop, for the small test cases
//! (n_hidden=6, 8) in `mlp_train::tests`.
//!
//! ## Bit-identity considerations
//!
//! The scalar kernels accumulate into `h_pre[j]` (forward) and
//! `gw1[i*N + j]` (backprop) in i-major order: the same set of
//! products is summed in the same order, just 8/4 lanes wide at a
//! time. Each lane's accumulation is bit-identical with the scalar
//! version unless FMA collapses a mul+add into a single rounding.
//!
//! We use `vfmadd231pd` (`a + b * c` with single rounding). This
//! differs from the scalar `*acc += s * w` (two roundings) by at
//! most 0.5 ULP per fused op. Cumulative drift across 372 mul-adds
//! at f64 is bounded by ~1e-12 relative — well inside the
//! `1e-9 relative` tolerance the caller accepts.
//!
//! The final `y = b2[0] + sum_o(h[o] * w2[o])` reduction in
//! `forward_simd` is kept in scalar form so the public output is
//! bit-identical (modulo the FMA used in the accumulation phase).
//!
//! ## Safety
//!
//! Every `unsafe` block in this module is gated by a runtime
//! `is_x86_feature_detected!` check at dispatch time, then guarded
//! by `#[target_feature]` on the implementation function. The
//! intrinsics themselves are `unsafe fn` in `std::arch` — the
//! `target_feature` annotation makes that contract explicit at the
//! Rust level.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]
// SIMD loops index parallel arrays
// Some intrinsics (set1_pd, setzero_pd) are marked safe; others
// (loadu_pd, storeu_pd) are unsafe. We wrap all intrinsic calls in
// `unsafe { ... }` blocks for visual uniformity at the cost of some
// redundant blocks the compiler would otherwise flag.
#![allow(unused_unsafe)]

/// Runtime feature dispatch for forward.
///
/// Returns `(y, h_pre, h)` with exactly the same shape as the scalar
/// `forward` in `mlp_train.rs`.
#[inline]
pub fn forward(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if crate::tier_cap::avx512_allowed() && archmage::X64V4Token::summon().is_some() {
            // SAFETY: dispatch gated by token summon (CPUID-checked).
            return unsafe { forward_avx512(x, w1, b1, w2, b2, n_features, n_hidden, alpha) };
        }
        if archmage::X64V3Token::summon().is_some() {
            // SAFETY: dispatch gated by token summon (CPUID-checked).
            return unsafe { forward_avx2(x, w1, b1, w2, b2, n_features, n_hidden, alpha) };
        }
    }
    forward_scalar(x, w1, b1, w2, b2, n_features, n_hidden, alpha)
}

/// Runtime feature dispatch for backprop_step.
///
/// Mutates `gw1`, `gb1`, `gw2`, `gb2` in place, matching the scalar
/// `backprop_step` in `mlp_train.rs`.
#[inline]
pub fn backprop_step(
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    dl_dy: f64,
    gw1: &mut [f64],
    gb1: &mut [f64],
    w2: &[f64],
    gw2: &mut [f64],
    gb2: &mut [f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if crate::tier_cap::avx512_allowed() && archmage::X64V4Token::summon().is_some() {
            // SAFETY: dispatch gated by token summon (CPUID-checked).
            unsafe {
                backprop_avx512(
                    x, h_pre, h, dl_dy, gw1, gb1, w2, gw2, gb2, n_features, n_hidden, alpha,
                );
            }
            return;
        }
        if archmage::X64V3Token::summon().is_some() {
            // SAFETY: dispatch gated by token summon (CPUID-checked).
            unsafe {
                backprop_avx2(
                    x, h_pre, h, dl_dy, gw1, gb1, w2, gw2, gb2, n_features, n_hidden, alpha,
                );
            }
            return;
        }
    }
    backprop_scalar(
        x, h_pre, h, dl_dy, gw1, gb1, w2, gw2, gb2, n_features, n_hidden, alpha,
    );
}

// =============================================================================
// SCALAR FALLBACK — bit-identical to the AVX2 (`forward_avx2` /
// `backprop_avx2`) kernels, which are the canonical arithmetic. That means
// the scalar path must reproduce three AVX2 details exactly:
//
//  1. Fused-op domain: lanes j < n_hidden - n_hidden%4 use FMA
//     (`f64::mul_add`, correctly-rounded single rounding on every target),
//     lanes j >= that boundary use mul+add (two roundings) — the same split
//     the AVX2 kernel's vector chunks / scalar tail produce.
//  2. The y-reduction's 4-lane accumulator: lane k sums h[4c+k]*w2[4c+k]
//     fused over all chunks, then pairwise (a0+a1)+(a2+a3), then the
//     sequential mul+add tail, then b2 + lane_sum + tail_sum.
//  3. Identical ascending-feature accumulation order (nonzero x only).
// =============================================================================

#[inline]
fn forward_scalar(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    let n4 = n_hidden - (n_hidden % 4);
    let mut h_pre = b1.to_vec();
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        // Fused domain — same fma chain the AVX2 lanes run.
        for (j, acc) in h_pre.iter_mut().enumerate().take(n4) {
            *acc = s.mul_add(row[j], *acc);
        }
        // Mul+add tail — the AVX2 kernel's scalar tail arithmetic.
        for (j, acc) in h_pre.iter_mut().enumerate().skip(n4) {
            *acc += s * row[j];
        }
    }
    let h: Vec<f64> = h_pre
        .iter()
        .map(|&v| if v >= 0.0 { v } else { alpha * v })
        .collect();
    // Emulate the AVX2 4-lane fused accumulator and its pairwise tree.
    let mut acc = [0.0f64; 4];
    for c in 0..n4 / 4 {
        for k in 0..4 {
            acc[k] = h[4 * c + k].mul_add(w2[4 * c + k], acc[k]);
        }
    }
    let lane_sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    let mut tail_sum = 0.0f64;
    for o in n4..n_hidden {
        tail_sum += h[o] * w2[o];
    }
    let y = b2[0] + lane_sum + tail_sum;
    (y, h_pre, h)
}

#[inline]
fn backprop_scalar(
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    dl_dy: f64,
    gw1: &mut [f64],
    gb1: &mut [f64],
    w2: &[f64],
    gw2: &mut [f64],
    gb2: &mut [f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) {
    let n4 = n_hidden - (n_hidden % 4);
    // gw2: fused on the canonical domain, mul+add on the AVX2 tail.
    for o in 0..n4 {
        gw2[o] = dl_dy.mul_add(h[o], gw2[o]);
    }
    for o in n4..n_hidden {
        gw2[o] += dl_dy * h[o];
    }
    gb2[0] += dl_dy;

    let mut dl_dh_pre = vec![0.0f64; n_hidden];
    for o in 0..n_hidden {
        let dh = dl_dy * w2[o];
        dl_dh_pre[o] = if h_pre[o] >= 0.0 { dh } else { alpha * dh };
    }

    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &mut gw1[i * n_hidden..(i + 1) * n_hidden];
        for (j, g) in row.iter_mut().enumerate().take(n4) {
            *g = s.mul_add(dl_dh_pre[j], *g);
        }
        for (j, g) in row.iter_mut().enumerate().skip(n4) {
            *g += s * dl_dh_pre[j];
        }
    }
    for (g, &dh) in gb1.iter_mut().zip(dl_dh_pre.iter()) {
        *g += dh;
    }
}

// =============================================================================
// AVX-512 (f64x8) — primary fast path on Zen 4 / Sapphire Rapids / Ice Lake
// =============================================================================
//
// BIT-PARITY RULE (tier-parity lane): this kernel must reproduce the AVX2
// (`forward_avx2`) arithmetic bit-for-bit. The AVX2 fused-op domain is
// lanes j < n4 = n_hidden - n_hidden%4; everything at/after n4 is scalar
// mul+add. So this kernel runs full 8-lane chunks, then — when
// n_hidden%8 >= 4 — ONE masked-8 fused group covering [n8*8, n4), then the
// mul+add tail [n4, n_hidden). The y reduction likewise emulates AVX2's
// 4-lane accumulator (lanes 0..3 via mask 0x0F) with the identical
// pairwise tree; an 8-lane accumulator would be a different summation
// order and is explicitly out of parity.

/// Mask covering the low 4 lanes of an f64x8 — used for the final
/// AVX2-boundary fused group when `n_hidden % 8 >= 4`, and for emulating
/// the AVX2 4-lane y accumulator.
#[cfg(target_arch = "x86_64")]
const LO4: u8 = 0x0F;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn forward_avx512(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    use std::arch::x86_64::{
        _CMP_GE_OQ, _mm512_cmp_pd_mask, _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_mask_blend_pd,
        _mm512_mask_fmadd_pd, _mm512_mask_storeu_pd, _mm512_maskz_loadu_pd, _mm512_mul_pd,
        _mm512_set1_pd, _mm512_setzero_pd, _mm512_storeu_pd,
    };

    // h_pre starts as a copy of b1 — matches scalar.
    let mut h_pre = b1.to_vec();
    debug_assert_eq!(h_pre.len(), n_hidden);

    let h_pre_ptr = h_pre.as_mut_ptr();
    let n_chunks = n_hidden / 8;
    let tail_start = n_chunks * 8;
    // Canonical fused-op boundary (AVX2 lane structure).
    let n4 = n_hidden - (n_hidden % 4);
    // Whether [n8*8, n4) is a nonempty 4-lane fused group.
    let mid_fused = tail_start + 4 <= n4;

    // For each input feature i: scale-and-accumulate w1's row into h_pre.
    // Preserves the sparse-x short-circuit.
    for (i, &s) in x[..n_features].iter().enumerate() {
        if s == 0.0 {
            continue;
        }
        let s_vec = unsafe { _mm512_set1_pd(s) };
        let row_ptr = unsafe { w1.as_ptr().add(i * n_hidden) };

        for c in 0..n_chunks {
            let off = c * 8;
            unsafe {
                let acc = _mm512_loadu_pd(h_pre_ptr.add(off));
                let row_v = _mm512_loadu_pd(row_ptr.add(off));
                // VFMADD231PD: acc = acc + row_v * s_vec
                let new_acc = _mm512_fmadd_pd(row_v, s_vec, acc);
                _mm512_storeu_pd(h_pre_ptr.add(off), new_acc);
            }
        }
        if mid_fused {
            // Fused lanes [tail_start, n4) — the AVX2 kernel covers these
            // with its last vector chunk; the masked form keeps the same
            // single-rounding fma per element.
            unsafe {
                let acc = _mm512_maskz_loadu_pd(LO4, h_pre_ptr.add(tail_start));
                let row_v = _mm512_maskz_loadu_pd(LO4, row_ptr.add(tail_start));
                let new_acc = _mm512_mask_fmadd_pd(row_v, LO4, s_vec, acc);
                _mm512_mask_storeu_pd(h_pre_ptr.add(tail_start), LO4, new_acc);
            }
        }
        // Mul+add tail [n4, n_hidden) — two roundings, matching AVX2's
        // scalar tail exactly.
        for (j, h) in h_pre.iter_mut().enumerate().take(n_hidden).skip(n4) {
            *h += s * unsafe { *row_ptr.add(j) };
        }
    }

    // LeakyReLU: h[o] = h_pre[o] >= 0 ? h_pre[o] : alpha * h_pre[o].
    // Elementwise — no accumulation structure, any width is bit-identical.
    let mut h = vec![0.0f64; n_hidden];
    let h_ptr = h.as_mut_ptr();
    let alpha_vec = unsafe { _mm512_set1_pd(alpha) };
    let zero_vec = unsafe { _mm512_setzero_pd() };
    for c in 0..n_chunks {
        let off = c * 8;
        unsafe {
            let pre = _mm512_loadu_pd(h_pre_ptr.add(off));
            let scaled = _mm512_mul_pd(pre, alpha_vec);
            // mask = lane i set when pre[i] >= 0.0
            let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(pre, zero_vec);
            // blend: mask=1 → pre, mask=0 → scaled
            let out = _mm512_mask_blend_pd(mask, scaled, pre);
            _mm512_storeu_pd(h_ptr.add(off), out);
        }
    }
    for o in tail_start..n_hidden {
        let v = h_pre[o];
        h[o] = if v >= 0.0 { v } else { alpha * v };
    }

    // Final reduction: y = b2[0] + sum_o(h[o] * w2[o]) — reproduces the
    // AVX2 4-lane accumulator exactly: lane k sums h[4c+k]*w2[4c+k] fused
    // over all chunks c, pairwise (a0+a1)+(a2+a3), then the sequential
    // mul+add tail.
    let mut acc_vec = unsafe { _mm512_setzero_pd() };
    for c in 0..n4 / 4 {
        let off = c * 4;
        unsafe {
            let h_v = _mm512_maskz_loadu_pd(LO4, h_ptr.add(off));
            let w2_v = _mm512_maskz_loadu_pd(LO4, w2.as_ptr().add(off));
            acc_vec = _mm512_mask_fmadd_pd(h_v, LO4, w2_v, acc_vec);
        }
    }
    let mut tail_sum = 0.0f64;
    for o in n4..n_hidden {
        tail_sum += h[o] * w2[o];
    }
    let mut acc_arr = [0.0f64; 8];
    unsafe { _mm512_storeu_pd(acc_arr.as_mut_ptr(), acc_vec) };
    let lane_sum = (acc_arr[0] + acc_arr[1]) + (acc_arr[2] + acc_arr[3]);
    let y = b2[0] + lane_sum + tail_sum;

    (y, h_pre, h)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn backprop_avx512(
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    dl_dy: f64,
    gw1: &mut [f64],
    gb1: &mut [f64],
    w2: &[f64],
    gw2: &mut [f64],
    gb2: &mut [f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) {
    use std::arch::x86_64::{
        _CMP_GE_OQ, _mm512_cmp_pd_mask, _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_mask_blend_pd,
        _mm512_mask_fmadd_pd, _mm512_mask_storeu_pd, _mm512_maskz_loadu_pd, _mm512_mul_pd,
        _mm512_set1_pd, _mm512_setzero_pd, _mm512_storeu_pd,
    };

    let n_chunks = n_hidden / 8;
    let tail_start = n_chunks * 8;
    let n4 = n_hidden - (n_hidden % 4);
    let mid_fused = tail_start + 4 <= n4;
    let dl_dy_vec = unsafe { _mm512_set1_pd(dl_dy) };
    let alpha_vec = unsafe { _mm512_set1_pd(alpha) };
    let zero_vec = unsafe { _mm512_setzero_pd() };

    // 1) gw2[o] += dl_dy * h[o] AND
    //    dl_dh_pre[o] = dl_dy * w2[o] * (h_pre[o] >= 0 ? 1 : alpha)
    // Fuse the two passes so we touch w2/h_pre once.
    let mut dl_dh_pre = vec![0.0f64; n_hidden];
    let dl_dh_pre_ptr = dl_dh_pre.as_mut_ptr();
    let gw2_ptr = gw2.as_mut_ptr();

    for c in 0..n_chunks {
        let off = c * 8;
        unsafe {
            // gw2 update
            let h_v = _mm512_loadu_pd(h.as_ptr().add(off));
            let gw2_v = _mm512_loadu_pd(gw2_ptr.add(off));
            let gw2_new = _mm512_fmadd_pd(dl_dy_vec, h_v, gw2_v);
            _mm512_storeu_pd(gw2_ptr.add(off), gw2_new);

            // dl_dh_pre
            let w2_v = _mm512_loadu_pd(w2.as_ptr().add(off));
            let dh = _mm512_mul_pd(dl_dy_vec, w2_v);
            let dh_scaled = _mm512_mul_pd(dh, alpha_vec);
            let pre_v = _mm512_loadu_pd(h_pre.as_ptr().add(off));
            let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(pre_v, zero_vec);
            let dh_gated = _mm512_mask_blend_pd(mask, dh_scaled, dh);
            _mm512_storeu_pd(dl_dh_pre_ptr.add(off), dh_gated);
        }
    }
    if mid_fused {
        // Fused lanes [tail_start, n4) — the AVX2 kernel covers these
        // with its last vector chunk.
        let off = tail_start;
        unsafe {
            let h_v = _mm512_maskz_loadu_pd(LO4, h.as_ptr().add(off));
            let gw2_v = _mm512_maskz_loadu_pd(LO4, gw2_ptr.add(off));
            let gw2_new = _mm512_mask_fmadd_pd(dl_dy_vec, LO4, h_v, gw2_v);
            _mm512_mask_storeu_pd(gw2_ptr.add(off), LO4, gw2_new);

            let w2_v = _mm512_maskz_loadu_pd(LO4, w2.as_ptr().add(off));
            let dh = _mm512_mul_pd(dl_dy_vec, w2_v);
            let dh_scaled = _mm512_mul_pd(dh, alpha_vec);
            let pre_v = _mm512_maskz_loadu_pd(LO4, h_pre.as_ptr().add(off));
            let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(pre_v, zero_vec);
            let dh_gated = _mm512_mask_blend_pd(mask, dh_scaled, dh);
            _mm512_mask_storeu_pd(dl_dh_pre_ptr.add(off), LO4, dh_gated);
        }
    }
    for o in n4..n_hidden {
        gw2[o] += dl_dy * h[o];
        let dh = dl_dy * w2[o];
        dl_dh_pre[o] = if h_pre[o] >= 0.0 { dh } else { alpha * dh };
    }

    gb2[0] += dl_dy;

    // 2) gw1 row update: for each i, if x[i] != 0,
    //    gw1[i*N + j] += x[i] * dl_dh_pre[j] for all j.
    //    Fused on j < n4, mul+add on the AVX2 tail.
    let gw1_ptr = gw1.as_mut_ptr();
    for (i, &s) in x[..n_features].iter().enumerate() {
        if s == 0.0 {
            continue;
        }
        let s_vec = unsafe { _mm512_set1_pd(s) };
        let row_off = i * n_hidden;
        for c in 0..n_chunks {
            let off = row_off + c * 8;
            unsafe {
                let g_v = _mm512_loadu_pd(gw1_ptr.add(off));
                let dh_v = _mm512_loadu_pd(dl_dh_pre_ptr.add(c * 8));
                let g_new = _mm512_fmadd_pd(s_vec, dh_v, g_v);
                _mm512_storeu_pd(gw1_ptr.add(off), g_new);
            }
        }
        if mid_fused {
            unsafe {
                let off = row_off + tail_start;
                let g_v = _mm512_maskz_loadu_pd(LO4, gw1_ptr.add(off));
                let dh_v = _mm512_maskz_loadu_pd(LO4, dl_dh_pre_ptr.add(tail_start));
                let g_new = _mm512_mask_fmadd_pd(s_vec, LO4, dh_v, g_v);
                _mm512_mask_storeu_pd(gw1_ptr.add(off), LO4, g_new);
            }
        }
        for j in n4..n_hidden {
            gw1[row_off + j] += s * dl_dh_pre[j];
        }
    }

    // 3) gb1[o] += dl_dh_pre[o] — a pure add per element, identical
    //    under any lane split.
    let gb1_ptr = gb1.as_mut_ptr();
    for c in 0..n_chunks {
        let off = c * 8;
        unsafe {
            let gb_v = _mm512_loadu_pd(gb1_ptr.add(off));
            let dh_v = _mm512_loadu_pd(dl_dh_pre_ptr.add(off));
            let one = _mm512_set1_pd(1.0);
            let gb_new = _mm512_fmadd_pd(dh_v, one, gb_v);
            _mm512_storeu_pd(gb1_ptr.add(off), gb_new);
        }
    }
    for o in tail_start..n_hidden {
        gb1[o] += dl_dh_pre[o];
    }
}

// =============================================================================
// AVX2 + FMA (f64x4) — secondary fast path for Haswell..Zen3
// =============================================================================

/// Register-blocked hidden-layer accumulate for `forward_avx2`: N YMM
/// accumulators (4N hidden units) are loaded once, kept in registers for the
/// entire nonzero-feature sweep, and stored once — the old loop paid a
/// load+store round-trip per (feature, chunk). `nz` is the ascending nonzero
/// index set, so the FMA order per accumulator is the same ascending feature
/// order as the `s == 0.0` skip loop it replaces; `h_pre` was seeded from
/// `b1`, so the first addend is unchanged.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fwd_acc_block<const N: usize>(
    h_pre_ptr: *mut f64,
    w1_ptr: *const f64,
    x_ptr: *const f64,
    nz: &[u32],
    n_hidden: usize,
    off: usize,
) {
    use std::arch::x86_64::{
        __m256d, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_setzero_pd,
        _mm256_storeu_pd,
    };
    unsafe {
        let mut acc: [__m256d; N] = [_mm256_setzero_pd(); N];
        for (k, a) in acc.iter_mut().enumerate() {
            *a = _mm256_loadu_pd(h_pre_ptr.add(off + 4 * k));
        }
        for &i in nz {
            let i = i as usize;
            let s_vec = _mm256_set1_pd(*x_ptr.add(i));
            let row = w1_ptr.add(i * n_hidden + off);
            for (k, a) in acc.iter_mut().enumerate() {
                *a = _mm256_fmadd_pd(_mm256_loadu_pd(row.add(4 * k)), s_vec, *a);
            }
        }
        for (k, a) in acc.iter().enumerate() {
            _mm256_storeu_pd(h_pre_ptr.add(off + 4 * k), *a);
        }
    }
}

/// Register-blocked `gw1` update for `backprop_avx2`: the N `dl_dh_pre`
/// vectors are loop-invariant across the feature sweep, so they load once and
/// stay in registers (the old loop reloaded them per feature). Per element
/// the arithmetic is the identical single `gw1 += x[i] * dh` FMA in ascending
/// feature order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn bwd_gw1_block<const N: usize>(
    gw1_ptr: *mut f64,
    dh_ptr: *const f64,
    x_ptr: *const f64,
    nz: &[u32],
    n_hidden: usize,
    off: usize,
) {
    use std::arch::x86_64::{
        __m256d, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_setzero_pd,
        _mm256_storeu_pd,
    };
    unsafe {
        let mut dh: [__m256d; N] = [_mm256_setzero_pd(); N];
        for (k, d) in dh.iter_mut().enumerate() {
            *d = _mm256_loadu_pd(dh_ptr.add(off + 4 * k));
        }
        for &i in nz {
            let i = i as usize;
            let s_vec = _mm256_set1_pd(*x_ptr.add(i));
            let row = gw1_ptr.add(i * n_hidden + off);
            for (k, d) in dh.iter().enumerate() {
                let g = _mm256_fmadd_pd(s_vec, *d, _mm256_loadu_pd(row.add(4 * k)));
                _mm256_storeu_pd(row.add(4 * k), g);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn forward_avx2(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    use std::arch::x86_64::{
        _CMP_LT_OQ, _mm256_blendv_pd, _mm256_cmp_pd, _mm256_fmadd_pd, _mm256_loadu_pd,
        _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };

    let mut h_pre = b1.to_vec();
    debug_assert_eq!(h_pre.len(), n_hidden);

    // Ascending nonzero-feature index set — the same features the old
    // `s == 0.0` continue skipped, in the same order.
    let mut nz: Vec<u32> = Vec::with_capacity(n_features);
    for (i, &s) in x[..n_features].iter().enumerate() {
        if s != 0.0 {
            nz.push(i as u32);
        }
    }

    let h_pre_ptr = h_pre.as_mut_ptr();
    let x_ptr = x.as_ptr();
    let w1_ptr = w1.as_ptr();
    let n_chunks = n_hidden / 4;
    let tail_start = n_chunks * 4;

    let mut c0 = 0usize;
    while c0 + 8 <= n_chunks {
        unsafe { fwd_acc_block::<8>(h_pre_ptr, w1_ptr, x_ptr, &nz, n_hidden, c0 * 4) };
        c0 += 8;
    }
    if n_chunks - c0 >= 4 {
        unsafe { fwd_acc_block::<4>(h_pre_ptr, w1_ptr, x_ptr, &nz, n_hidden, c0 * 4) };
        c0 += 4;
    }
    if n_chunks - c0 >= 2 {
        unsafe { fwd_acc_block::<2>(h_pre_ptr, w1_ptr, x_ptr, &nz, n_hidden, c0 * 4) };
        c0 += 2;
    }
    if n_chunks - c0 >= 1 {
        unsafe { fwd_acc_block::<1>(h_pre_ptr, w1_ptr, x_ptr, &nz, n_hidden, c0 * 4) };
    }
    for (o, h_pre_o) in h_pre.iter_mut().enumerate().skip(tail_start) {
        let mut acc = *h_pre_o;
        for &i in &nz {
            let i = i as usize;
            acc += unsafe { *x_ptr.add(i) * *w1_ptr.add(i * n_hidden + o) };
        }
        *h_pre_o = acc;
    }

    // LeakyReLU
    let mut h = vec![0.0f64; n_hidden];
    let h_ptr = h.as_mut_ptr();
    let alpha_vec = unsafe { _mm256_set1_pd(alpha) };
    let zero_vec = unsafe { _mm256_setzero_pd() };
    for c in 0..n_chunks {
        let off = c * 4;
        unsafe {
            let pre = _mm256_loadu_pd(h_pre_ptr.add(off));
            let scaled = _mm256_mul_pd(pre, alpha_vec);
            // mask = pre < 0.0  → all-ones lanes, 0 otherwise
            let mask = _mm256_cmp_pd::<_CMP_LT_OQ>(pre, zero_vec);
            // blendv: mask high bit set → take `scaled`, else `pre`
            let out = _mm256_blendv_pd(pre, scaled, mask);
            _mm256_storeu_pd(h_ptr.add(off), out);
        }
    }
    for o in tail_start..n_hidden {
        let v = h_pre[o];
        h[o] = if v >= 0.0 { v } else { alpha * v };
    }

    // Final reduction
    let mut acc_vec = unsafe { _mm256_setzero_pd() };
    for c in 0..n_chunks {
        let off = c * 4;
        unsafe {
            let h_v = _mm256_loadu_pd(h_ptr.add(off));
            let w2_v = _mm256_loadu_pd(w2.as_ptr().add(off));
            acc_vec = _mm256_fmadd_pd(h_v, w2_v, acc_vec);
        }
    }
    let mut tail_sum = 0.0f64;
    for o in tail_start..n_hidden {
        tail_sum += h[o] * w2[o];
    }
    let mut acc_arr = [0.0f64; 4];
    unsafe { _mm256_storeu_pd(acc_arr.as_mut_ptr(), acc_vec) };
    let lane_sum = (acc_arr[0] + acc_arr[1]) + (acc_arr[2] + acc_arr[3]);
    let y = b2[0] + lane_sum + tail_sum;

    (y, h_pre, h)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn backprop_avx2(
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    dl_dy: f64,
    gw1: &mut [f64],
    gb1: &mut [f64],
    w2: &[f64],
    gw2: &mut [f64],
    gb2: &mut [f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) {
    use std::arch::x86_64::{
        _CMP_LT_OQ, _mm256_add_pd, _mm256_blendv_pd, _mm256_cmp_pd, _mm256_fmadd_pd,
        _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };

    let n_chunks = n_hidden / 4;
    let tail_start = n_chunks * 4;
    let dl_dy_vec = unsafe { _mm256_set1_pd(dl_dy) };
    let alpha_vec = unsafe { _mm256_set1_pd(alpha) };
    let zero_vec = unsafe { _mm256_setzero_pd() };

    let mut dl_dh_pre = vec![0.0f64; n_hidden];
    let dl_dh_pre_ptr = dl_dh_pre.as_mut_ptr();
    let gw2_ptr = gw2.as_mut_ptr();
    let gb1_ptr = gb1.as_mut_ptr();

    // `gb1[o] += dl_dh_pre[o]` is folded into this pass: the addend is the
    // just-computed value, so the per-element arithmetic is unchanged.
    for c in 0..n_chunks {
        let off = c * 4;
        unsafe {
            let h_v = _mm256_loadu_pd(h.as_ptr().add(off));
            let gw2_v = _mm256_loadu_pd(gw2_ptr.add(off));
            let gw2_new = _mm256_fmadd_pd(dl_dy_vec, h_v, gw2_v);
            _mm256_storeu_pd(gw2_ptr.add(off), gw2_new);

            let w2_v = _mm256_loadu_pd(w2.as_ptr().add(off));
            let dh = _mm256_mul_pd(dl_dy_vec, w2_v);
            let dh_scaled = _mm256_mul_pd(dh, alpha_vec);
            let pre_v = _mm256_loadu_pd(h_pre.as_ptr().add(off));
            // mask = pre < 0  → take scaled, else take dh
            let mask = _mm256_cmp_pd::<_CMP_LT_OQ>(pre_v, zero_vec);
            let dh_gated = _mm256_blendv_pd(dh, dh_scaled, mask);
            _mm256_storeu_pd(dl_dh_pre_ptr.add(off), dh_gated);

            let gb_v = _mm256_loadu_pd(gb1_ptr.add(off));
            let gb_new = _mm256_add_pd(gb_v, dh_gated);
            _mm256_storeu_pd(gb1_ptr.add(off), gb_new);
        }
    }
    for o in tail_start..n_hidden {
        gw2[o] += dl_dy * h[o];
        let dh = dl_dy * w2[o];
        let dh_gated = if h_pre[o] >= 0.0 { dh } else { alpha * dh };
        dl_dh_pre[o] = dh_gated;
        gb1[o] += dh_gated;
    }

    gb2[0] += dl_dy;

    // Ascending nonzero-feature index set — same skip set as the old
    // `s == 0.0` continue.
    let mut nz: Vec<u32> = Vec::with_capacity(n_features);
    for (i, &s) in x[..n_features].iter().enumerate() {
        if s != 0.0 {
            nz.push(i as u32);
        }
    }

    let x_ptr = x.as_ptr();
    let gw1_ptr = gw1.as_mut_ptr();
    let mut c0 = 0usize;
    while c0 + 8 <= n_chunks {
        unsafe { bwd_gw1_block::<8>(gw1_ptr, dl_dh_pre_ptr, x_ptr, &nz, n_hidden, c0 * 4) };
        c0 += 8;
    }
    if n_chunks - c0 >= 4 {
        unsafe { bwd_gw1_block::<4>(gw1_ptr, dl_dh_pre_ptr, x_ptr, &nz, n_hidden, c0 * 4) };
        c0 += 4;
    }
    if n_chunks - c0 >= 2 {
        unsafe { bwd_gw1_block::<2>(gw1_ptr, dl_dh_pre_ptr, x_ptr, &nz, n_hidden, c0 * 4) };
        c0 += 2;
    }
    if n_chunks - c0 >= 1 {
        unsafe { bwd_gw1_block::<1>(gw1_ptr, dl_dh_pre_ptr, x_ptr, &nz, n_hidden, c0 * 4) };
    }
    for (o, &d) in dl_dh_pre.iter().enumerate().skip(tail_start) {
        for &i in &nz {
            let i = i as usize;
            unsafe {
                *gw1_ptr.add(i * n_hidden + o) += *x_ptr.add(i) * d;
            }
        }
    }
}

// =============================================================================
// Tests — bit-equivalence checks between scalar and SIMD paths.
// =============================================================================

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    /// Deterministic xorshift used to fill weights for the equivalence
    /// tests — keeps the test self-contained without pulling in the
    /// trainer's RNG.
    struct Xs64(u64);
    impl Xs64 {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_f64(&mut self) -> f64 {
            // Uniform in [-1, 1).
            let bits = self.next();
            (bits as f64 / u64::MAX as f64) * 2.0 - 1.0
        }
    }

    fn random_buf(rng: &mut Xs64, n: usize) -> Vec<f64> {
        (0..n).map(|_| rng.next_f64()).collect()
    }

    fn random_sparse_x(rng: &mut Xs64, n: usize, zero_frac: f64) -> Vec<f64> {
        // rng.next_f64() is uniform in [-1, 1). Threshold at
        // (2*zero_frac - 1.0) so a `zero_frac` fraction of draws land
        // in the zero bucket — preserves the sparse-x short-circuit.
        let thresh = 2.0 * zero_frac - 1.0;
        (0..n)
            .map(|_| {
                if rng.next_f64() < thresh {
                    0.0
                } else {
                    rng.next_f64()
                }
            })
            .collect()
    }

    /// Compare scalar vs the dispatched SIMD path for the production
    /// (372, 128) shape.
    #[test]
    fn forward_simd_matches_scalar_372x128() {
        let n_features = 372;
        let n_hidden = 128;
        let alpha = 0.01;
        let mut rng = Xs64::new(0xCAFE_BEEF_1234_5678);
        let x = random_sparse_x(&mut rng, n_features, 0.3);
        let w1 = random_buf(&mut rng, n_features * n_hidden);
        let b1 = random_buf(&mut rng, n_hidden);
        let w2 = random_buf(&mut rng, n_hidden);
        let b2 = vec![rng.next_f64()];

        let (y_scal, h_pre_scal, h_scal) =
            forward_scalar(&x, &w1, &b1, &w2, &b2, n_features, n_hidden, alpha);
        let (y_simd, h_pre_simd, h_simd) =
            forward(&x, &w1, &b1, &w2, &b2, n_features, n_hidden, alpha);

        // h_pre: per-lane bit-identical to scalar except for FMA fusion.
        // The longest accumulation chain is 372 mul-adds in f64 — drift
        // bounded well under 1e-12 relative.
        for (a, b) in h_pre_scal.iter().zip(h_pre_simd.iter()) {
            let denom = a.abs().max(b.abs()).max(1.0);
            let rel = (a - b).abs() / denom;
            assert!(
                rel < 1e-12,
                "h_pre mismatch: scalar={a} simd={b} rel={rel:e}",
            );
        }
        for (a, b) in h_scal.iter().zip(h_simd.iter()) {
            let denom = a.abs().max(b.abs()).max(1.0);
            let rel = (a - b).abs() / denom;
            assert!(rel < 1e-12, "h mismatch: scalar={a} simd={b} rel={rel:e}");
        }
        let denom = y_scal.abs().max(y_simd.abs()).max(1.0);
        let rel = (y_scal - y_simd).abs() / denom;
        assert!(
            rel < 1e-11,
            "y mismatch: scalar={y_scal} simd={y_simd} rel={rel:e}",
        );
    }

    /// Tiny shape exercises the scalar-tail path for n_hidden not a
    /// multiple of 8 / 4. Matches the test-suite shapes (n_hidden 6, 8).
    #[test]
    fn forward_simd_matches_scalar_tiny_shapes() {
        for &(n_features, n_hidden) in &[(16usize, 8usize), (16, 6), (8, 7), (4, 5)] {
            let alpha = 0.01;
            let mut rng = Xs64::new(0xDEAD_BEEF_0011 ^ (n_features * 31 + n_hidden) as u64);
            let x = random_sparse_x(&mut rng, n_features, 0.2);
            let w1 = random_buf(&mut rng, n_features * n_hidden);
            let b1 = random_buf(&mut rng, n_hidden);
            let w2 = random_buf(&mut rng, n_hidden);
            let b2 = vec![rng.next_f64()];

            let (y_scal, h_pre_scal, h_scal) =
                forward_scalar(&x, &w1, &b1, &w2, &b2, n_features, n_hidden, alpha);
            let (y_simd, h_pre_simd, h_simd) =
                forward(&x, &w1, &b1, &w2, &b2, n_features, n_hidden, alpha);
            for (a, b) in h_pre_scal.iter().zip(h_pre_simd.iter()) {
                let denom = a.abs().max(b.abs()).max(1.0);
                let rel = (a - b).abs() / denom;
                assert!(
                    rel < 1e-12,
                    "[{n_features},{n_hidden}] h_pre mismatch: \
                     scalar={a} simd={b} rel={rel:e}",
                );
            }
            for (a, b) in h_scal.iter().zip(h_simd.iter()) {
                let denom = a.abs().max(b.abs()).max(1.0);
                let rel = (a - b).abs() / denom;
                assert!(rel < 1e-12, "h mismatch ({n_features},{n_hidden})");
            }
            let denom = y_scal.abs().max(y_simd.abs()).max(1.0);
            let rel = (y_scal - y_simd).abs() / denom;
            assert!(
                rel < 1e-11,
                "[{n_features},{n_hidden}] y mismatch: \
                 scalar={y_scal} simd={y_simd} rel={rel:e}",
            );
        }
    }

    /// Backprop should produce per-element gradient arrays that match
    /// the scalar reference within FMA noise.
    #[test]
    fn backprop_simd_matches_scalar_372x128() {
        let n_features = 372;
        let n_hidden = 128;
        let alpha = 0.01;
        let mut rng = Xs64::new(0x1234_5678_AABB_CCDD);

        let x = random_sparse_x(&mut rng, n_features, 0.3);
        let w2 = random_buf(&mut rng, n_hidden);
        let h_pre = random_buf(&mut rng, n_hidden);
        let h: Vec<f64> = h_pre
            .iter()
            .map(|&v| if v >= 0.0 { v } else { alpha * v })
            .collect();
        let dl_dy = rng.next_f64();

        // Independent gradient buffers seeded with non-zero values so we
        // verify ACCUMULATION (a += b) and not just initial write.
        let mut gw1_a = random_buf(&mut rng, n_features * n_hidden);
        let mut gw1_b = gw1_a.clone();
        let mut gb1_a = random_buf(&mut rng, n_hidden);
        let mut gb1_b = gb1_a.clone();
        let mut gw2_a = random_buf(&mut rng, n_hidden);
        let mut gw2_b = gw2_a.clone();
        let mut gb2_a = vec![rng.next_f64()];
        let mut gb2_b = gb2_a.clone();

        backprop_scalar(
            &x, &h_pre, &h, dl_dy, &mut gw1_a, &mut gb1_a, &w2, &mut gw2_a, &mut gb2_a, n_features,
            n_hidden, alpha,
        );
        backprop_step(
            &x, &h_pre, &h, dl_dy, &mut gw1_b, &mut gb1_b, &w2, &mut gw2_b, &mut gb2_b, n_features,
            n_hidden, alpha,
        );

        for (i, (a, b)) in gw1_a.iter().zip(gw1_b.iter()).enumerate() {
            let denom = a.abs().max(b.abs()).max(1.0);
            let rel = (a - b).abs() / denom;
            assert!(
                rel < 1e-12,
                "gw1[{i}] mismatch: scalar={a} simd={b} rel={rel:e}",
            );
        }
        for (i, (a, b)) in gb1_a.iter().zip(gb1_b.iter()).enumerate() {
            let denom = a.abs().max(b.abs()).max(1.0);
            let rel = (a - b).abs() / denom;
            assert!(
                rel < 1e-12,
                "gb1[{i}] mismatch: scalar={a} simd={b} rel={rel:e}",
            );
        }
        for (i, (a, b)) in gw2_a.iter().zip(gw2_b.iter()).enumerate() {
            let denom = a.abs().max(b.abs()).max(1.0);
            let rel = (a - b).abs() / denom;
            assert!(
                rel < 1e-12,
                "gw2[{i}] mismatch: scalar={a} simd={b} rel={rel:e}",
            );
        }
        let denom = gb2_a[0].abs().max(gb2_b[0].abs()).max(1.0);
        let rel = (gb2_a[0] - gb2_b[0]).abs() / denom;
        assert!(rel < 1e-12, "gb2 mismatch");
    }

    /// Same shape sanity for tiny n_hidden.
    #[test]
    fn backprop_simd_matches_scalar_tiny_shapes() {
        for &(n_features, n_hidden) in &[(16usize, 8usize), (16, 6), (8, 7), (4, 5)] {
            let alpha = 0.01;
            let mut rng = Xs64::new(0xABCD_1234 ^ (n_features * 31 + n_hidden) as u64);

            let x = random_sparse_x(&mut rng, n_features, 0.2);
            let w2 = random_buf(&mut rng, n_hidden);
            let h_pre = random_buf(&mut rng, n_hidden);
            let h: Vec<f64> = h_pre
                .iter()
                .map(|&v| if v >= 0.0 { v } else { alpha * v })
                .collect();
            let dl_dy = rng.next_f64();

            let mut gw1_a = random_buf(&mut rng, n_features * n_hidden);
            let mut gw1_b = gw1_a.clone();
            let mut gb1_a = random_buf(&mut rng, n_hidden);
            let mut gb1_b = gb1_a.clone();
            let mut gw2_a = random_buf(&mut rng, n_hidden);
            let mut gw2_b = gw2_a.clone();
            let mut gb2_a = vec![rng.next_f64()];
            let mut gb2_b = gb2_a.clone();

            backprop_scalar(
                &x, &h_pre, &h, dl_dy, &mut gw1_a, &mut gb1_a, &w2, &mut gw2_a, &mut gb2_a,
                n_features, n_hidden, alpha,
            );
            backprop_step(
                &x, &h_pre, &h, dl_dy, &mut gw1_b, &mut gb1_b, &w2, &mut gw2_b, &mut gb2_b,
                n_features, n_hidden, alpha,
            );

            for (a, b) in gw1_a.iter().zip(gw1_b.iter()) {
                let denom = a.abs().max(b.abs()).max(1.0);
                let rel = (a - b).abs() / denom;
                assert!(rel < 1e-12, "gw1 mismatch [{n_features},{n_hidden}]",);
            }
            for (a, b) in gb1_a.iter().zip(gb1_b.iter()) {
                let denom = a.abs().max(b.abs()).max(1.0);
                let rel = (a - b).abs() / denom;
                assert!(rel < 1e-12, "gb1 mismatch [{n_features},{n_hidden}]");
            }
            for (a, b) in gw2_a.iter().zip(gw2_b.iter()) {
                let denom = a.abs().max(b.abs()).max(1.0);
                let rel = (a - b).abs() / denom;
                assert!(rel < 1e-12, "gw2 mismatch [{n_features},{n_hidden}]");
            }
            let denom = gb2_a[0].abs().max(gb2_b[0].abs()).max(1.0);
            let rel = (gb2_a[0] - gb2_b[0]).abs() / denom;
            assert!(rel < 1e-12, "gb2 mismatch [{n_features},{n_hidden}]");
        }
    }

    /// The dispatchers prefer AVX-512 on hosts that have it, which would
    /// leave the AVX2 kernels unexercised by the tests above. Call them
    /// directly, gated on the same runtime features the dispatch checks,
    /// across shapes that hit every cascade arm (8/4/2/1 chunk blocks plus
    /// scalar tails) and the 944×32 fold geometry.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernels_match_scalar_directly() {
        assert!(
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
            "this test needs an x86-64-v3 (AVX2+FMA) host: it is the canonical tier every other tier must reproduce"
        );
        let alpha = 0.01;
        for &(n_features, n_hidden) in &[
            (944usize, 32usize), // fold shape: 8 chunks → one <8> block
            (372, 128),          // production: 4× <8> blocks
            (64, 36),            // 9 chunks → 8 + 1
            (16, 20),            // 5 chunks → 4 + 1
            (16, 24),            // 6 chunks → 4 + 2
            (16, 28),            // 7 chunks → 4 + 2 + 1
            (16, 12),            // 3 chunks → 2 + 1
            (16, 8),
            (16, 6),
            (8, 7),
            (4, 5),
            (4, 3), // n_hidden < 4: scalar tail only
        ] {
            let mut rng = Xs64::new(0x51ED_1E55 ^ (n_features * 31 + n_hidden) as u64);
            let x = random_sparse_x(&mut rng, n_features, 0.35);
            let w1 = random_buf(&mut rng, n_features * n_hidden);
            let b1 = random_buf(&mut rng, n_hidden);
            let w2 = random_buf(&mut rng, n_hidden);
            let b2 = vec![rng.next_f64()];

            let (y_s, hpre_s, h_s) =
                forward_scalar(&x, &w1, &b1, &w2, &b2, n_features, n_hidden, alpha);
            let (y_v, hpre_v, h_v) =
                unsafe { forward_avx2(&x, &w1, &b1, &w2, &b2, n_features, n_hidden, alpha) };
            for (a, b) in hpre_s.iter().zip(hpre_v.iter()) {
                let rel = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
                assert!(rel < 1e-12, "[{n_features},{n_hidden}] h_pre {a} vs {b}");
            }
            for (a, b) in h_s.iter().zip(h_v.iter()) {
                let rel = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
                assert!(rel < 1e-12, "[{n_features},{n_hidden}] h {a} vs {b}");
            }
            let rel = (y_s - y_v).abs() / y_s.abs().max(y_v.abs()).max(1.0);
            assert!(rel < 1e-11, "[{n_features},{n_hidden}] y {y_s} vs {y_v}");

            let dl_dy = rng.next_f64();
            let mut gw1_s = random_buf(&mut rng, n_features * n_hidden);
            let mut gw1_v = gw1_s.clone();
            let mut gb1_s = random_buf(&mut rng, n_hidden);
            let mut gb1_v = gb1_s.clone();
            let mut gw2_s = random_buf(&mut rng, n_hidden);
            let mut gw2_v = gw2_s.clone();
            let mut gb2_s = vec![rng.next_f64()];
            let mut gb2_v = gb2_s.clone();

            backprop_scalar(
                &x, &hpre_s, &h_s, dl_dy, &mut gw1_s, &mut gb1_s, &w2, &mut gw2_s, &mut gb2_s,
                n_features, n_hidden, alpha,
            );
            unsafe {
                backprop_avx2(
                    &x, &hpre_s, &h_s, dl_dy, &mut gw1_v, &mut gb1_v, &w2, &mut gw2_v, &mut gb2_v,
                    n_features, n_hidden, alpha,
                );
            }
            for (a, b) in gw1_s.iter().zip(gw1_v.iter()) {
                let rel = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
                assert!(rel < 1e-12, "gw1 [{n_features},{n_hidden}] {a} vs {b}");
            }
            for (a, b) in gb1_s.iter().zip(gb1_v.iter()) {
                let rel = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
                assert!(rel < 1e-12, "gb1 [{n_features},{n_hidden}] {a} vs {b}");
            }
            for (a, b) in gw2_s.iter().zip(gw2_v.iter()) {
                let rel = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
                assert!(rel < 1e-12, "gw2 [{n_features},{n_hidden}] {a} vs {b}");
            }
            let rel = (gb2_s[0] - gb2_v[0]).abs() / gb2_s[0].abs().max(gb2_v[0].abs()).max(1.0);
            assert!(rel < 1e-12, "gb2 [{n_features},{n_hidden}]");
        }
    }

    // =====================================================================
    // Tier parity — the AVX2 (`_v3`) kernels are the canonical arithmetic;
    // every other tier the dispatcher can pick must reproduce it BIT FOR BIT.
    // =====================================================================

    /// Ordered-int ULP distance between two f64 values.
    fn ulp_diff_f64(a: f64, b: f64) -> u64 {
        fn ord(v: f64) -> i64 {
            let b = v.to_bits() as i64;
            if b < 0 { i64::MIN - b } else { b }
        }
        ord(a).abs_diff(ord(b))
    }

    fn max_ulp_slice(a: &[f64], b: &[f64]) -> (u64, usize) {
        let mut worst = (0u64, usize::MAX);
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let d = ulp_diff_f64(x, y);
            if d > worst.0 {
                worst = (d, i);
            }
        }
        worst
    }

    /// Tier-parity shapes: the fold (944×32) and production (372×128) plus
    /// every cascade-arm/tail combination so `n_hidden` exercises
    /// 8/4/2/1-chunk blocks and 0..7-element scalar tails.
    #[cfg(target_arch = "x86_64")]
    const PARITY_SHAPES: &[(usize, usize)] = &[
        (944, 32),
        (372, 128),
        (16, 8),
        (16, 6),
        (8, 7),
        (4, 5),
        (4, 3),
        (16, 20),
        (16, 24),
        (16, 28),
        (16, 12),
        (64, 36),
        (32, 40),
        (128, 24),
        (256, 16),
        (64, 64),
    ];

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn forward_all_tiers_bit_identical() {
        assert!(
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
            "this test needs an x86-64-v3 (AVX2+FMA) host: it is the canonical tier every other tier must reproduce"
        );
        let has512 = crate::tier_cap::avx512_allowed() && std::is_x86_feature_detected!("avx512f");
        let alpha = 0.01;
        let mut failures = Vec::new();
        for &(nf, nh) in PARITY_SHAPES {
            for &seed in &[0x51ED_1E55u64, 0xABCD_1234, 0x0000_00FF] {
                let mut rng = Xs64::new(seed ^ (nf * 31 + nh) as u64);
                let x = random_sparse_x(&mut rng, nf, 0.35);
                let w1 = random_buf(&mut rng, nf * nh);
                let b1 = random_buf(&mut rng, nh);
                let w2 = random_buf(&mut rng, nh);
                let b2 = vec![rng.next_f64()];

                let (y_v3, hp_v3, h_v3) =
                    unsafe { forward_avx2(&x, &w1, &b1, &w2, &b2, nf, nh, alpha) };

                // scalar tier vs canonical v3
                let (y_s, hp_s, h_s) = forward_scalar(&x, &w1, &b1, &w2, &b2, nf, nh, alpha);
                for (name, a, b) in [
                    ("y", ulp_diff_f64(y_s, y_v3), 0usize),
                    (
                        "h_pre",
                        max_ulp_slice(&hp_s, &hp_v3).0,
                        max_ulp_slice(&hp_s, &hp_v3).1,
                    ),
                    (
                        "h",
                        max_ulp_slice(&h_s, &h_v3).0,
                        max_ulp_slice(&h_s, &h_v3).1,
                    ),
                ] {
                    if a != 0 {
                        failures.push(format!(
                            "forward scalar-vs-v3 [{nf}x{nh} seed={seed:#x}] {name}: \
                             max_ulp={a} at {b} (scalar={} v3={})",
                            if name == "y" {
                                y_s
                            } else if name == "h_pre" {
                                hp_s[b]
                            } else {
                                h_s[b]
                            },
                            if name == "y" {
                                y_v3
                            } else if name == "h_pre" {
                                hp_v3[b]
                            } else {
                                h_v3[b]
                            },
                        ));
                    }
                }

                if has512 {
                    let (y_v4, hp_v4, h_v4) =
                        unsafe { forward_avx512(&x, &w1, &b1, &w2, &b2, nf, nh, alpha) };
                    for (name, a, b) in [
                        ("y", ulp_diff_f64(y_v4, y_v3), 0usize),
                        (
                            "h_pre",
                            max_ulp_slice(&hp_v4, &hp_v3).0,
                            max_ulp_slice(&hp_v4, &hp_v3).1,
                        ),
                        (
                            "h",
                            max_ulp_slice(&h_v4, &h_v3).0,
                            max_ulp_slice(&h_v4, &h_v3).1,
                        ),
                    ] {
                        if a != 0 {
                            failures.push(format!(
                                "forward v4-vs-v3 [{nf}x{nh} seed={seed:#x}] {name}: \
                                 max_ulp={a} at {b} (v4={} v3={})",
                                if name == "y" {
                                    y_v4
                                } else if name == "h_pre" {
                                    hp_v4[b]
                                } else {
                                    h_v4[b]
                                },
                                if name == "y" {
                                    y_v3
                                } else if name == "h_pre" {
                                    hp_v3[b]
                                } else {
                                    h_v3[b]
                                },
                            ));
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn backprop_all_tiers_bit_identical() {
        assert!(
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
            "this test needs an x86-64-v3 (AVX2+FMA) host: it is the canonical tier every other tier must reproduce"
        );
        let has512 = crate::tier_cap::avx512_allowed() && std::is_x86_feature_detected!("avx512f");
        let alpha = 0.01;
        let mut failures = Vec::new();
        for &(nf, nh) in PARITY_SHAPES {
            for &seed in &[0x51ED_1E55u64, 0xABCD_1234, 0x0000_00FF] {
                let mut rng = Xs64::new(seed ^ (nf * 31 + nh) as u64);
                let x = random_sparse_x(&mut rng, nf, 0.35);
                let w2 = random_buf(&mut rng, nh);
                let h_pre = random_buf(&mut rng, nh);
                let h: Vec<f64> = h_pre
                    .iter()
                    .map(|&v| if v >= 0.0 { v } else { alpha * v })
                    .collect();
                let dl_dy = rng.next_f64();
                let gw1_0 = random_buf(&mut rng, nf * nh);
                let gb1_0 = random_buf(&mut rng, nh);
                let gw2_0 = random_buf(&mut rng, nh);
                let gb2_0 = vec![rng.next_f64()];

                type Grads = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
                type RunFn<'a> = dyn FnMut(&mut [f64], &mut [f64], &mut [f64], &mut [f64]) + 'a;
                let run = |f: &mut RunFn<'_>| -> Grads {
                    let (mut gw1, mut gb1, mut gw2, mut gb2) =
                        (gw1_0.clone(), gb1_0.clone(), gw2_0.clone(), gb2_0.clone());
                    f(&mut gw1, &mut gb1, &mut gw2, &mut gb2);
                    (gw1, gb1, gw2, gb2)
                };

                let r_v3 = run(&mut |gw1, gb1, gw2, gb2| unsafe {
                    backprop_avx2(
                        &x, &h_pre, &h, dl_dy, gw1, gb1, &w2, gw2, gb2, nf, nh, alpha,
                    )
                });
                let r_s = run(&mut |gw1, gb1, gw2, gb2| {
                    backprop_scalar(
                        &x, &h_pre, &h, dl_dy, gw1, gb1, &w2, gw2, gb2, nf, nh, alpha,
                    )
                });
                for (name, (ulp, idx)) in [
                    ("gw1", max_ulp_slice(&r_s.0, &r_v3.0)),
                    ("gb1", max_ulp_slice(&r_s.1, &r_v3.1)),
                    ("gw2", max_ulp_slice(&r_s.2, &r_v3.2)),
                    ("gb2", max_ulp_slice(&r_s.3, &r_v3.3)),
                ] {
                    if ulp != 0 {
                        failures.push(format!(
                            "backprop scalar-vs-v3 [{nf}x{nh} seed={seed:#x}] {name}: \
                             max_ulp={ulp} at {idx}"
                        ));
                    }
                }

                if has512 {
                    let r_v4 = run(&mut |gw1, gb1, gw2, gb2| unsafe {
                        backprop_avx512(
                            &x, &h_pre, &h, dl_dy, gw1, gb1, &w2, gw2, gb2, nf, nh, alpha,
                        )
                    });
                    for (name, (ulp, idx)) in [
                        ("gw1", max_ulp_slice(&r_v4.0, &r_v3.0)),
                        ("gb1", max_ulp_slice(&r_v4.1, &r_v3.1)),
                        ("gw2", max_ulp_slice(&r_v4.2, &r_v3.2)),
                        ("gb2", max_ulp_slice(&r_v4.3, &r_v3.3)),
                    ] {
                        if ulp != 0 {
                            failures.push(format!(
                                "backprop v4-vs-v3 [{nf}x{nh} seed={seed:#x}] {name}: \
                                 max_ulp={ulp} at {idx}"
                            ));
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// The public dispatchers must select the bit-identical kernels under
    /// every token permutation the host offers: disabling the AVX-512 tokens
    /// must route to the AVX2 kernels (same bytes), and disabling those must
    /// reach the scalar replica (again same bytes).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dispatch_bit_identical_under_token_permutations() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        assert!(
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
            "this test needs an x86-64-v3 (AVX2+FMA) host: it is the canonical tier every other tier must reproduce"
        );
        let alpha = 0.01;
        let (nf, nh) = (944usize, 32usize);
        let mut rng = Xs64::new(0x600D_CAFE);
        let x = random_sparse_x(&mut rng, nf, 0.35);
        let w1 = random_buf(&mut rng, nf * nh);
        let b1 = random_buf(&mut rng, nh);
        let w2 = random_buf(&mut rng, nh);
        let b2 = vec![rng.next_f64()];
        let h_pre = random_buf(&mut rng, nh);
        let h: Vec<f64> = h_pre
            .iter()
            .map(|&v| if v >= 0.0 { v } else { alpha * v })
            .collect();
        let dl_dy = rng.next_f64();
        let gw1_0 = random_buf(&mut rng, nf * nh);
        let gb1_0 = random_buf(&mut rng, nh);
        let gw2_0 = random_buf(&mut rng, nh);
        let gb2_0 = vec![rng.next_f64()];

        fn bits_of(v: &[f64]) -> Vec<u64> {
            v.iter().map(|f| f.to_bits()).collect()
        }

        type PermOut = (f64, Vec<u64>, Vec<u64>, Vec<u64>);
        let mut baseline: Option<PermOut> = None;
        let mut failures = Vec::new();
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let (y, hp, hh) = forward(&x, &w1, &b1, &w2, &b2, nf, nh, alpha);
            let (mut gw1, mut gb1, mut gw2, mut gb2) =
                (gw1_0.clone(), gb1_0.clone(), gw2_0.clone(), gb2_0.clone());
            backprop_step(
                &x, &h_pre, &h, dl_dy, &mut gw1, &mut gb1, &w2, &mut gw2, &mut gb2, nf, nh, alpha,
            );
            let mut bits = Vec::new();
            for v in gw1.iter().chain(&gb1).chain(&gw2).chain(&gb2) {
                bits.push(v.to_bits());
            }
            let cur = (y, bits_of(&hp), bits_of(&hh), bits);
            if let Some(b) = &baseline {
                if cur.0.to_bits() != b.0.to_bits() || cur.1 != b.1 || cur.2 != b.2 || cur.3 != b.3
                {
                    failures.push(format!(
                        "dispatch diverged under permutation: {}",
                        perm.label
                    ));
                }
            } else {
                baseline = Some(cur);
            }
            eprintln!(
                "permutation [{}] -> y={:e} (bits {:#x})",
                perm.label,
                y,
                y.to_bits()
            );
        });
        eprintln!("permutations run: {}", report.permutations_run);
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
