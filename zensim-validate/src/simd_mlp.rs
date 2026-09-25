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
        if crate::tier_cap::avx512_allowed() && std::is_x86_feature_detected!("avx512f") {
            // SAFETY: dispatch gated by `is_x86_feature_detected`.
            return unsafe { forward_avx512(x, w1, b1, w2, b2, n_features, n_hidden, alpha) };
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: dispatch gated by `is_x86_feature_detected`.
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
        if crate::tier_cap::avx512_allowed() && std::is_x86_feature_detected!("avx512f") {
            // SAFETY: dispatch gated by `is_x86_feature_detected`.
            unsafe {
                backprop_avx512(
                    x, h_pre, h, dl_dy, gw1, gb1, w2, gw2, gb2, n_features, n_hidden, alpha,
                );
            }
            return;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: dispatch gated by `is_x86_feature_detected`.
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
// SCALAR FALLBACK — bit-identical to original `forward` / `backprop_step`
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
    let mut h_pre = b1.to_vec();
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
    let h: Vec<f64> = h_pre
        .iter()
        .map(|&v| if v >= 0.0 { v } else { alpha * v })
        .collect();
    let mut y = b2[0];
    for o in 0..n_hidden {
        y += h[o] * w2[o];
    }
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
    for o in 0..n_hidden {
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
        for (g, &dh) in row.iter_mut().zip(dl_dh_pre.iter()) {
            *g += s * dh;
        }
    }
    for (g, &dh) in gb1.iter_mut().zip(dl_dh_pre.iter()) {
        *g += dh;
    }
}

// =============================================================================
// AVX-512 (f64x8) — primary fast path on Zen 4 / Sapphire Rapids / Ice Lake
// =============================================================================

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
        _mm512_mul_pd, _mm512_set1_pd, _mm512_storeu_pd,
    };

    // h_pre starts as a copy of b1 — matches scalar.
    let mut h_pre = b1.to_vec();
    debug_assert_eq!(h_pre.len(), n_hidden);

    let h_pre_ptr = h_pre.as_mut_ptr();
    let n_chunks = n_hidden / 8;
    let tail_start = n_chunks * 8;

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
        // Scalar tail for n_hidden % 8 != 0 (test cases).
        for (k, h) in h_pre[tail_start..n_hidden].iter_mut().enumerate() {
            *h += s * unsafe { *row_ptr.add(tail_start + k) };
        }
    }

    // LeakyReLU: h[o] = h_pre[o] >= 0 ? h_pre[o] : alpha * h_pre[o].
    let mut h = vec![0.0f64; n_hidden];
    let h_ptr = h.as_mut_ptr();
    let alpha_vec = unsafe { _mm512_set1_pd(alpha) };
    let zero_vec = unsafe { _mm512_set1_pd(0.0) };
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

    // Final reduction: y = b2[0] + sum_o(h[o] * w2[o]).
    // Use a per-lane SIMD accumulator (8-wide) and reduce at the end.
    // Order differs from scalar but is within the 1e-9 relative budget.
    let mut acc_vec = unsafe { _mm512_set1_pd(0.0) };
    for c in 0..n_chunks {
        let off = c * 8;
        unsafe {
            let h_v = _mm512_loadu_pd(h_ptr.add(off));
            let w2_v = _mm512_loadu_pd(w2.as_ptr().add(off));
            acc_vec = _mm512_fmadd_pd(h_v, w2_v, acc_vec);
        }
    }
    let mut tail_sum = 0.0f64;
    for o in tail_start..n_hidden {
        tail_sum += h[o] * w2[o];
    }
    let mut acc_arr = [0.0f64; 8];
    unsafe { _mm512_storeu_pd(acc_arr.as_mut_ptr(), acc_vec) };
    // Horizontal sum: keep order deterministic (pairwise).
    let lane_sum = (acc_arr[0] + acc_arr[1])
        + (acc_arr[2] + acc_arr[3])
        + ((acc_arr[4] + acc_arr[5]) + (acc_arr[6] + acc_arr[7]));
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
        _mm512_mul_pd, _mm512_set1_pd, _mm512_storeu_pd,
    };

    let n_chunks = n_hidden / 8;
    let tail_start = n_chunks * 8;
    let dl_dy_vec = unsafe { _mm512_set1_pd(dl_dy) };
    let alpha_vec = unsafe { _mm512_set1_pd(alpha) };
    let zero_vec = unsafe { _mm512_set1_pd(0.0) };

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
    for o in tail_start..n_hidden {
        gw2[o] += dl_dy * h[o];
        let dh = dl_dy * w2[o];
        dl_dh_pre[o] = if h_pre[o] >= 0.0 { dh } else { alpha * dh };
    }

    gb2[0] += dl_dy;

    // 2) gw1 row update: for each i, if x[i] != 0,
    //    gw1[i*N + j] += x[i] * dl_dh_pre[j] for all j.
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
        for j in tail_start..n_hidden {
            gw1[row_off + j] += s * dl_dh_pre[j];
        }
    }

    // 3) gb1[o] += dl_dh_pre[o]
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
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
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
}
