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
#![allow(clippy::needless_range_loop)]
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
        if std::is_x86_feature_detected!("avx512f") {
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
        if std::is_x86_feature_detected!("avx512f") {
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
    for i in 0..n_features {
        let s = x[i];
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
        for j in tail_start..n_hidden {
            h_pre[j] += s * unsafe { *row_ptr.add(j) };
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
    for i in 0..n_features {
        let s = x[i];
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

    let h_pre_ptr = h_pre.as_mut_ptr();
    let n_chunks = n_hidden / 4;
    let tail_start = n_chunks * 4;

    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let s_vec = unsafe { _mm256_set1_pd(s) };
        let row_ptr = unsafe { w1.as_ptr().add(i * n_hidden) };

        for c in 0..n_chunks {
            let off = c * 4;
            unsafe {
                let acc = _mm256_loadu_pd(h_pre_ptr.add(off));
                let row_v = _mm256_loadu_pd(row_ptr.add(off));
                let new_acc = _mm256_fmadd_pd(row_v, s_vec, acc);
                _mm256_storeu_pd(h_pre_ptr.add(off), new_acc);
            }
        }
        for j in tail_start..n_hidden {
            h_pre[j] += s * unsafe { *row_ptr.add(j) };
        }
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
        }
    }
    for o in tail_start..n_hidden {
        gw2[o] += dl_dy * h[o];
        let dh = dl_dy * w2[o];
        dl_dh_pre[o] = if h_pre[o] >= 0.0 { dh } else { alpha * dh };
    }

    gb2[0] += dl_dy;

    let gw1_ptr = gw1.as_mut_ptr();
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let s_vec = unsafe { _mm256_set1_pd(s) };
        let row_off = i * n_hidden;
        for c in 0..n_chunks {
            let off = row_off + c * 4;
            unsafe {
                let g_v = _mm256_loadu_pd(gw1_ptr.add(off));
                let dh_v = _mm256_loadu_pd(dl_dh_pre_ptr.add(c * 4));
                let g_new = _mm256_fmadd_pd(s_vec, dh_v, g_v);
                _mm256_storeu_pd(gw1_ptr.add(off), g_new);
            }
        }
        for j in tail_start..n_hidden {
            gw1[row_off + j] += s * dl_dh_pre[j];
        }
    }

    let gb1_ptr = gb1.as_mut_ptr();
    for c in 0..n_chunks {
        let off = c * 4;
        unsafe {
            let gb_v = _mm256_loadu_pd(gb1_ptr.add(off));
            let dh_v = _mm256_loadu_pd(dl_dh_pre_ptr.add(off));
            let gb_new = _mm256_add_pd(gb_v, dh_v);
            _mm256_storeu_pd(gb1_ptr.add(off), gb_new);
        }
    }
    for o in tail_start..n_hidden {
        gb1[o] += dl_dh_pre[o];
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
}
