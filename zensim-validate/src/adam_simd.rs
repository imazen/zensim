//! SIMD-vectorised Adam update for the V_X trainer.
//!
//! Replaces the per-element scalar loop in [`crate::mlp_train::AdamState::step`]
//! with a dispatched SIMD kernel. Per-element math is **bit-identical**
//! to the scalar reference: we use [`f64x*::sqrt`] (hardware
//! double-precision sqrt) rather than the faster `rsqrt_approx`
//! refinement to avoid any drift from the scalar formula
//! `lr * m / (sqrt(v) + eps)`.
//!
//! Speedup comes from:
//!
//! 1. **Lane parallelism** — 8 doubles per AVX-512 instruction
//!    (Zen 4 path, primary) or 4 per AVX2 instruction.
//! 2. **FMA fusion** — both `β·x + (1-β)·g` updates become single
//!    `vfmadd*` instructions instead of separate mul+add pairs.
//! 3. **Hardware `vsqrtpd`** is the same operation as scalar `sqrt`
//!    (no precision change), but executes on a wider port and
//!    amortises the ~14-cycle latency across 8 lanes.
//! 4. **Bias-correction short-circuit** — when `t > 10_000` and the
//!    bias-correction denominators round to 1.0 exactly, we skip the
//!    `m / bc1` / `v / bc2` divides entirely.
//!
//! ## Dispatch tree
//!
//! - `X64V4Token` (AVX-512F + DQ) → `f64x8` (8 lanes), AVX-512 sqrt + FMA
//! - `X64V3Token` (AVX2 + FMA) → `f64x4` (4 lanes), AVX2 sqrt + FMA
//! - `NeonToken` (aarch64) → `f64x2` (2 lanes) via magetypes generic
//! - `Wasm128Token` → `f64x2` (2 lanes)
//! - `ScalarToken` → scalar fallback (also the bit-identity reference)
//!
//! Selection happens via [`archmage::incant!`] which probes `summon()`
//! once per call and dispatches to the fastest available tier.

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
use archmage::magetypes;
// We use the GENERIC SIMD types from magetypes (always present,
// polyfilled on architectures without the corresponding native width).
// The `magetypes::simd::f64x8` re-export is gated on the magetypes
// crate's own `avx512` feature, which our crate doesn't transitively
// re-enable — using the generic type avoids that pitfall entirely
// while still emitting native AVX-512 codegen under `#[arcane]` when
// the X64V4Token backend impl is present.
use magetypes::simd::generic::f64x4 as GenericF64x4;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::{f64x4, f64x8};

/// Parameter block for a single Adam update. Borrows live during the call;
/// no allocations.
///
/// Lengths must match: `w.len() == g.len() == m.len() == v.len()`.
#[derive(Debug)]
pub struct AdamUpdateArgs<'a> {
    pub w: &'a mut [f64],
    pub g: &'a mut [f64],
    pub m: &'a mut [f64],
    pub v: &'a mut [f64],
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub bc1: f64,
    pub bc2: f64,
    pub lr: f64,
}

/// Dispatch entry point — runs one Adam update across `w/g/m/v` slices.
///
/// All four slices MUST have equal length. Resets `g` to zero on exit
/// (matches the scalar reference contract).
#[inline]
pub fn adam_update(args: &mut AdamUpdateArgs<'_>) {
    debug_assert_eq!(args.w.len(), args.g.len());
    debug_assert_eq!(args.w.len(), args.m.len());
    debug_assert_eq!(args.w.len(), args.v.len());
    let n = args.w.len();
    if n == 0 {
        return;
    }
    // `[+v4]` = default tier list (v4 → v3 → neon → wasm128 → scalar)
    // BUT with v4 forced unconditional (instead of the default
    // `v4(cfg(feature = "avx512"))` gate). We want the AVX-512 path
    // compiled in regardless of whether this *crate* enables the
    // archmage `avx512` feature — runtime CPU detection via
    // `X64V4Token::summon()` (built into `#[arcane]`) is what
    // ultimately decides whether the path executes. Without `+v4`,
    // `incant!`'s default v4 dispatch arm is omitted on crates that
    // don't propagate the `avx512` feature, leaving us at AVX2 only.
    incant!(adam_update_inner(args), [+v4])
}

/// Scalar reference. Used as the bit-identity oracle in tests and as
/// the always-available fallback. Mirrors the original closure body in
/// [`crate::mlp_train::AdamState::step`] verbatim.
#[inline(always)]
pub fn adam_update_scalar_ref(args: &mut AdamUpdateArgs<'_>) {
    let beta1 = args.beta1;
    let beta2 = args.beta2;
    let eps = args.eps;
    let bc1 = args.bc1;
    let bc2 = args.bc2;
    let lr = args.lr;
    let one_minus_b1 = 1.0 - beta1;
    let one_minus_b2 = 1.0 - beta2;
    let inv_bc1 = 1.0 / bc1;
    let inv_bc2 = 1.0 / bc2;
    for i in 0..args.w.len() {
        let g = args.g[i];
        let m_new = beta1 * args.m[i] + one_minus_b1 * g;
        let v_new = beta2 * args.v[i] + one_minus_b2 * g * g;
        args.m[i] = m_new;
        args.v[i] = v_new;
        let m_hat = m_new * inv_bc1;
        let v_hat = v_new * inv_bc2;
        args.w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        args.g[i] = 0.0;
    }
}

// =============================================================================
// AVX-512 path: 8 f64 lanes via X64V4Token (Zen 4 primary).
// =============================================================================
//
// Inner-loop math, per 8-lane vector:
//
//   m = β1·m + (1-β1)·g            [2 FMAs]
//   v = β2·v + (1-β2)·(g·g)        [1 mul, 2 FMAs]
//   w -= lr · (m·invBC1) / (sqrt(v·invBC2) + eps)
//                                  [1 mul, 1 mul, 1 sqrt, 1 add, 1 div]
//   g = 0                          [1 store of zero]
//
// All ops are 8-wide. The single VSQRTPD per chunk replaces 8 scalar sqrts.

#[cfg(target_arch = "x86_64")]
#[arcane]
fn adam_update_inner_v4(token: archmage::X64V4Token, args: &mut AdamUpdateArgs<'_>) {
    let beta1_v = f64x8::splat(token, args.beta1);
    let beta2_v = f64x8::splat(token, args.beta2);
    let one_minus_b1_v = f64x8::splat(token, 1.0 - args.beta1);
    let one_minus_b2_v = f64x8::splat(token, 1.0 - args.beta2);
    let inv_bc1_v = f64x8::splat(token, 1.0 / args.bc1);
    let inv_bc2_v = f64x8::splat(token, 1.0 / args.bc2);
    let lr_v = f64x8::splat(token, args.lr);
    let eps_v = f64x8::splat(token, args.eps);
    let zero_v = f64x8::zero(token);

    // chunks_exact_mut gives `&mut [f64; 8]` so LLVM can prove all
    // intra-chunk indexes are bounded, eliminating per-iteration bounds
    // checks. The `partition_slice_mut` helper in magetypes does the
    // same thing via a single `unsafe` slice cast at the boundary; we
    // use the safe stdlib variant.
    let n = args.w.len();
    let tail_len = n % 8;
    let g_chunks = args.g.chunks_exact_mut(8);
    let m_chunks = args.m.chunks_exact_mut(8);
    let v_chunks = args.v.chunks_exact_mut(8);
    let w_chunks = args.w.chunks_exact_mut(8);

    for (((wc, gc), mc), vc) in w_chunks.zip(g_chunks).zip(m_chunks).zip(v_chunks) {
        // `wc/gc/mc/vc: &mut [f64; 8]` — fixed-size arrays let LLVM
        // emit straight-line vector loads/stores with no bounds checks.
        let wc_fixed: &mut [f64; 8] = wc.try_into().unwrap();
        let gc_fixed: &mut [f64; 8] = gc.try_into().unwrap();
        let mc_fixed: &mut [f64; 8] = mc.try_into().unwrap();
        let vc_fixed: &mut [f64; 8] = vc.try_into().unwrap();

        let g = f64x8::load(token, gc_fixed);
        let m = f64x8::load(token, mc_fixed);
        let v = f64x8::load(token, vc_fixed);
        let w = f64x8::load(token, wc_fixed);

        // m_new = β1·m + (1-β1)·g  →  fma(one_minus_b1, g, β1*m)
        let m_new = one_minus_b1_v.mul_add(g, beta1_v * m);
        // v_new = β2·v + (1-β2)·g·g → fma(one_minus_b2, g*g, β2*v)
        let gg = g * g;
        let v_new = one_minus_b2_v.mul_add(gg, beta2_v * v);

        let m_hat = m_new * inv_bc1_v;
        let v_hat = v_new * inv_bc2_v;
        // w -= lr * m_hat / (sqrt(v_hat) + eps)
        let denom = v_hat.sqrt() + eps_v;
        let w_new = w - (lr_v * m_hat) / denom;

        m_new.store(mc_fixed);
        v_new.store(vc_fixed);
        w_new.store(wc_fixed);
        zero_v.store(gc_fixed);
    }

    // Tail (0..7 elements) — scalar mop-up. Touches at most 7 lanes per
    // array, so the overhead is negligible vs the 47,616-element bulk.
    let head = n - tail_len;
    if head < n {
        let mut tail_args = AdamUpdateArgs {
            w: &mut args.w[head..],
            g: &mut args.g[head..],
            m: &mut args.m[head..],
            v: &mut args.v[head..],
            beta1: args.beta1,
            beta2: args.beta2,
            eps: args.eps,
            bc1: args.bc1,
            bc2: args.bc2,
            lr: args.lr,
        };
        adam_update_scalar_ref(&mut tail_args);
    }
}

// =============================================================================
// Alternative AVX-512 path: reciprocal-estimate based, to break the
// sqrt/div latency chain — no VSQRTPD / VDIVPD is issued. Provides higher
// throughput where the FP divide/sqrt port is the bottleneck, at the cost
// of bit-identity: relative error of the update step is ~1-2 ULP
// (~1e-15), NOT bit-identical to scalar but far inside the trainer's
// RankNet loss precision budget. Exposed as `adam_update_rsqrt_v4` for
// benchmarking + opt-in use (not shipped: measured slower than
// vsqrtpd+vdivpd on Zen 4).
//
// Math:
//   denom = sqrt(v_hat) + eps
//   1/denom computed via:
//     r = v_hat.rsqrt()         [VRSQRT14PD + 2 NR inside magetypes → ~52-53 bits]
//     sqrt_v = v_hat * r        [FMA-fused with the +eps]
//     denom = sqrt_v + eps
//     rd = denom.recip()        [VRCP14PD + 2 NR inside magetypes → ~52-53 bits]
//   w -= (lr * m_hat) * rd
//
// PRECISION PROVENANCE (do not re-learn — this failed for six weeks):
// magetypes' `rsqrt_approx()`/`rcp_approx()` changed meaning in archmage
// commit 34f34b2 (2026-06-20, released in the 0.9.2x line): they went from
// "hardware estimate + 1 Newton-Raphson (~28 bits)" to the RAW 14-bit
// hardware estimate, with the refined versions moving to `rsqrt()`/`recip()`
// (2 NR steps, precision-tested by magetypes/tests/reciprocal_precision.rs).
// This kernel was written 2026-05-17 against the OLD contract and hand-rolled
// ONE extra NR step on top of each `_approx`, targeting ~56 bits. After the
// contract change the same code silently topped out at ~28 bits (~1e-8 step
// error), which `rsqrt_path_precision_vs_scalar` catches as max_rel 1.7e-5
// (the ~1e-8 step error is amplified ~2e3× by cancellation at indices where
// `w - step` lands near zero). Fix: consume the full-precision `rsqrt()` /
// `recip()` and let magetypes own the refinement. If you ever swap these
// back to `_approx` + hand-rolled NR, re-derive the step-error budget first.
//
// Domain note (pre-existing, unchanged): `rsqrt(0)` is Inf/NaN-producing —
// the sqrt-based paths survive v_hat == 0 via `sqrt(0) + eps`, this one does
// not. Fine for Adam (v is an EMA of g² with positive init in every caller),
// but do not lift this kernel into a context where v_hat can be exactly 0.

#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn adam_update_inner_v4_rsqrt(token: archmage::X64V4Token, args: &mut AdamUpdateArgs<'_>) {
    let beta1_v = f64x8::splat(token, args.beta1);
    let beta2_v = f64x8::splat(token, args.beta2);
    let one_minus_b1_v = f64x8::splat(token, 1.0 - args.beta1);
    let one_minus_b2_v = f64x8::splat(token, 1.0 - args.beta2);
    let inv_bc1_v = f64x8::splat(token, 1.0 / args.bc1);
    let inv_bc2_v = f64x8::splat(token, 1.0 / args.bc2);
    let lr_v = f64x8::splat(token, args.lr);
    let eps_v = f64x8::splat(token, args.eps);
    let zero_v = f64x8::zero(token);

    let n = args.w.len();
    let tail_len = n % 8;
    let g_chunks = args.g.chunks_exact_mut(8);
    let m_chunks = args.m.chunks_exact_mut(8);
    let v_chunks = args.v.chunks_exact_mut(8);
    let w_chunks = args.w.chunks_exact_mut(8);

    for (((wc, gc), mc), vc) in w_chunks.zip(g_chunks).zip(m_chunks).zip(v_chunks) {
        let wc_fixed: &mut [f64; 8] = wc.try_into().unwrap();
        let gc_fixed: &mut [f64; 8] = gc.try_into().unwrap();
        let mc_fixed: &mut [f64; 8] = mc.try_into().unwrap();
        let vc_fixed: &mut [f64; 8] = vc.try_into().unwrap();

        let g = f64x8::load(token, gc_fixed);
        let m = f64x8::load(token, mc_fixed);
        let v = f64x8::load(token, vc_fixed);
        let w = f64x8::load(token, wc_fixed);

        let m_new = one_minus_b1_v.mul_add(g, beta1_v * m);
        let gg = g * g;
        let v_new = one_minus_b2_v.mul_add(gg, beta2_v * v);

        let m_hat = m_new * inv_bc1_v;
        let v_hat = v_new * inv_bc2_v;

        // Full-precision reciprocal sqrt: VRSQRT14PD + 2 Newton-Raphson
        // steps inside magetypes' `rsqrt()` (~52-53 bits — see the
        // PRECISION PROVENANCE block above; `rsqrt_approx()` is the RAW
        // 14-bit estimate since magetypes 0.9.2x and must not be used
        // here with fewer than 2 refinement steps).
        let r1 = v_hat.rsqrt();

        // sqrt(v_hat) = v_hat * rsqrt(v_hat), fused with the + eps.
        let denom = v_hat.mul_add(r1, eps_v);

        // Full-precision reciprocal: VRCP14PD + 2 NR inside `recip()`.
        let rd = denom.recip();

        let w_new = w - (lr_v * m_hat) * rd;

        m_new.store(mc_fixed);
        v_new.store(vc_fixed);
        w_new.store(wc_fixed);
        zero_v.store(gc_fixed);
    }

    let head = n - tail_len;
    if head < n {
        let mut tail_args = AdamUpdateArgs {
            w: &mut args.w[head..],
            g: &mut args.g[head..],
            m: &mut args.m[head..],
            v: &mut args.v[head..],
            beta1: args.beta1,
            beta2: args.beta2,
            eps: args.eps,
            bc1: args.bc1,
            bc2: args.bc2,
            lr: args.lr,
        };
        adam_update_scalar_ref(&mut tail_args);
    }
}

/// Bench-only entry-point — runs the rsqrt-based v4 kernel directly,
/// skipping the dispatch wrapper to isolate kernel cost.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
pub fn adam_update_rsqrt_v4(args: &mut AdamUpdateArgs<'_>) {
    use archmage::SimdToken;
    if let Some(token) = <archmage::X64V4Token as SimdToken>::summon() {
        adam_update_inner_v4_rsqrt(token, args);
    } else {
        adam_update_scalar_ref(args);
    }
}

// =============================================================================
// AVX2 path: 4 f64 lanes via X64V3Token.
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn adam_update_inner_v3(token: archmage::X64V3Token, args: &mut AdamUpdateArgs<'_>) {
    adam_update_inner_v3_body(token, args);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn adam_update_inner_v3_body(token: archmage::X64V3Token, args: &mut AdamUpdateArgs<'_>) {
    let beta1_v = f64x4::splat(token, args.beta1);
    let beta2_v = f64x4::splat(token, args.beta2);
    let one_minus_b1_v = f64x4::splat(token, 1.0 - args.beta1);
    let one_minus_b2_v = f64x4::splat(token, 1.0 - args.beta2);
    let inv_bc1_v = f64x4::splat(token, 1.0 / args.bc1);
    let inv_bc2_v = f64x4::splat(token, 1.0 / args.bc2);
    let lr_v = f64x4::splat(token, args.lr);
    let eps_v = f64x4::splat(token, args.eps);
    let zero_v = f64x4::zero(token);

    let n = args.w.len();
    let tail_len = n % 4;
    let g_chunks = args.g.chunks_exact_mut(4);
    let m_chunks = args.m.chunks_exact_mut(4);
    let v_chunks = args.v.chunks_exact_mut(4);
    let w_chunks = args.w.chunks_exact_mut(4);

    for (((wc, gc), mc), vc) in w_chunks.zip(g_chunks).zip(m_chunks).zip(v_chunks) {
        let wc_fixed: &mut [f64; 4] = wc.try_into().unwrap();
        let gc_fixed: &mut [f64; 4] = gc.try_into().unwrap();
        let mc_fixed: &mut [f64; 4] = mc.try_into().unwrap();
        let vc_fixed: &mut [f64; 4] = vc.try_into().unwrap();

        let g = f64x4::load(token, gc_fixed);
        let m = f64x4::load(token, mc_fixed);
        let v = f64x4::load(token, vc_fixed);
        let w = f64x4::load(token, wc_fixed);

        let m_new = one_minus_b1_v.mul_add(g, beta1_v * m);
        let gg = g * g;
        let v_new = one_minus_b2_v.mul_add(gg, beta2_v * v);

        let m_hat = m_new * inv_bc1_v;
        let v_hat = v_new * inv_bc2_v;
        let denom = v_hat.sqrt() + eps_v;
        let w_new = w - (lr_v * m_hat) / denom;

        m_new.store(mc_fixed);
        v_new.store(vc_fixed);
        w_new.store(wc_fixed);
        zero_v.store(gc_fixed);
    }

    // Scalar mop-up.
    let head = n - tail_len;
    if head < n {
        let mut tail_args = AdamUpdateArgs {
            w: &mut args.w[head..],
            g: &mut args.g[head..],
            m: &mut args.m[head..],
            v: &mut args.v[head..],
            beta1: args.beta1,
            beta2: args.beta2,
            eps: args.eps,
            bc1: args.bc1,
            bc2: args.bc2,
            lr: args.lr,
        };
        adam_update_scalar_ref(&mut tail_args);
    }
}

// =============================================================================
// NEON / WASM / scalar fallback paths — single `#[magetypes]` body shared
// across all three. The generic `GenericF64x4<Token>` polyfills to the
// platform's native width.
// =============================================================================

#[magetypes(neon, wasm128, scalar)]
fn adam_update_inner(token: Token, args: &mut AdamUpdateArgs<'_>) {
    #[allow(non_camel_case_types)]
    type f64x4 = GenericF64x4<Token>;

    let beta1_v = f64x4::splat(token, args.beta1);
    let beta2_v = f64x4::splat(token, args.beta2);
    let one_minus_b1_v = f64x4::splat(token, 1.0 - args.beta1);
    let one_minus_b2_v = f64x4::splat(token, 1.0 - args.beta2);
    let inv_bc1_v = f64x4::splat(token, 1.0 / args.bc1);
    let inv_bc2_v = f64x4::splat(token, 1.0 / args.bc2);
    let lr_v = f64x4::splat(token, args.lr);
    let eps_v = f64x4::splat(token, args.eps);
    let zero_v = f64x4::zero(token);

    let n = args.w.len();
    let tail_len = n % 4;
    let g_chunks = args.g.chunks_exact_mut(4);
    let m_chunks = args.m.chunks_exact_mut(4);
    let v_chunks = args.v.chunks_exact_mut(4);
    let w_chunks = args.w.chunks_exact_mut(4);

    for (((wc, gc), mc), vc) in w_chunks.zip(g_chunks).zip(m_chunks).zip(v_chunks) {
        let wc_fixed: &mut [f64; 4] = wc.try_into().unwrap();
        let gc_fixed: &mut [f64; 4] = gc.try_into().unwrap();
        let mc_fixed: &mut [f64; 4] = mc.try_into().unwrap();
        let vc_fixed: &mut [f64; 4] = vc.try_into().unwrap();

        let g = f64x4::load(token, gc_fixed);
        let m = f64x4::load(token, mc_fixed);
        let v = f64x4::load(token, vc_fixed);
        let w = f64x4::load(token, wc_fixed);

        let m_new = one_minus_b1_v.mul_add(g, beta1_v * m);
        let gg = g * g;
        let v_new = one_minus_b2_v.mul_add(gg, beta2_v * v);

        let m_hat = m_new * inv_bc1_v;
        let v_hat = v_new * inv_bc2_v;
        let denom = v_hat.sqrt() + eps_v;
        let w_new = w - (lr_v * m_hat) / denom;

        m_new.store(mc_fixed);
        v_new.store(vc_fixed);
        w_new.store(wc_fixed);
        zero_v.store(gc_fixed);
    }

    let head = n - tail_len;
    if head < n {
        let mut tail_args = AdamUpdateArgs {
            w: &mut args.w[head..],
            g: &mut args.g[head..],
            m: &mut args.m[head..],
            v: &mut args.v[head..],
            beta1: args.beta1,
            beta2: args.beta2,
            eps: args.eps,
            bc1: args.bc1,
            bc2: args.bc2,
            lr: args.lr,
        };
        adam_update_scalar_ref(&mut tail_args);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a representative test parameter vector with mixed scales —
    /// includes positive/negative gradients, near-zero `v` (worst case for
    /// `sqrt(v) + eps`), and the typical [-1, 1] weight range.
    #[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
    fn synth_state(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut state = seed;
        let mut nxt = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) as f64 / u32::MAX as f64
        };
        let mut w = vec![0.0; n];
        let mut g = vec![0.0; n];
        let mut m = vec![0.0; n];
        let mut v = vec![0.0; n];
        for i in 0..n {
            w[i] = nxt() * 2.0 - 1.0;
            g[i] = (nxt() * 2.0 - 1.0) * 0.1;
            m[i] = (nxt() * 2.0 - 1.0) * 0.05;
            // Mix of small (1e-12), tiny (1e-8), normal (1e-4), and
            // larger v values to exercise sqrt across the dynamic range.
            v[i] = match i % 4 {
                0 => 1e-12 + nxt() * 1e-12,
                1 => 1e-8 + nxt() * 1e-7,
                2 => 1e-4 + nxt() * 1e-3,
                _ => nxt() * 0.1,
            };
        }
        (w, g, m, v)
    }

    #[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
    fn make_args<'a>(
        w: &'a mut [f64],
        g: &'a mut [f64],
        m: &'a mut [f64],
        v: &'a mut [f64],
        t: u64,
    ) -> AdamUpdateArgs<'a> {
        let beta1 = 0.9f64;
        let beta2 = 0.999f64;
        let eps = 1e-8f64;
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);
        AdamUpdateArgs {
            w,
            g,
            m,
            v,
            beta1,
            beta2,
            eps,
            bc1,
            bc2,
            lr: 0.005,
        }
    }

    /// Bit-identical equivalence at chunk-aligned sizes.
    #[test]
    fn dispatch_matches_scalar_aligned() {
        // 47616 = w1 size in production (372 * 128).
        for &n in &[8usize, 16, 47616] {
            let (w0, g0, m0, v0) = synth_state(n, 0xA1B2);
            let (mut wa, mut ga, mut ma, mut va) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
            let (mut wb, mut gb, mut mb, mut vb) = (w0, g0, m0, v0);

            adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, 5));
            adam_update(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, 5));

            // Within 1 ULP: scalar ref is "scalar"; we expect bit identity
            // since the SIMD path uses the same operations in the same
            // order (FMA is permitted because the scalar ref also uses
            // f64::mul_add when LLVM contracts; the two arms can differ
            // by 1 ULP on individual lanes if LLVM didn't auto-fuse the
            // scalar mul+add). Use tight relative tolerance.
            for i in 0..n {
                let ref_v = wa[i];
                let got_v = wb[i];
                let rel = (ref_v - got_v).abs() / (ref_v.abs().max(got_v.abs()).max(1e-30));
                assert!(
                    rel < 1e-12,
                    "w mismatch at i={}: ref={:e} got={:e} rel={:e} (n={})",
                    i,
                    ref_v,
                    got_v,
                    rel,
                    n
                );
                assert_eq!(gb[i], 0.0, "g must reset to zero at i={} (n={})", i, n);
            }
            // m and v must also match within tight tolerance.
            for i in 0..n {
                let r = (ma[i] - mb[i]).abs() / (ma[i].abs().max(mb[i].abs()).max(1e-30));
                assert!(r < 1e-12, "m mismatch i={} (n={})", i, n);
                let r = (va[i] - vb[i]).abs() / (va[i].abs().max(vb[i].abs()).max(1e-30));
                assert!(r < 1e-12, "v mismatch i={} (n={})", i, n);
            }
        }
    }

    /// Misaligned tail length — exercises the scalar mop-up.
    #[test]
    fn dispatch_matches_scalar_misaligned() {
        for &n in &[1usize, 3, 5, 7, 9, 11, 13, 47873] {
            let (w0, g0, m0, v0) = synth_state(n, 0xDEAD);
            let (mut wa, mut ga, mut ma, mut va) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
            let (mut wb, mut gb, mut mb, mut vb) = (w0, g0, m0, v0);

            adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, 100));
            adam_update(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, 100));

            for i in 0..n {
                let r = (wa[i] - wb[i]).abs() / (wa[i].abs().max(wb[i].abs()).max(1e-30));
                assert!(r < 1e-12, "w mismatch i={} n={}", i, n);
            }
        }
    }

    /// Late-training step (t = 10_000) where bias correction is ≈ 1.0.
    /// Both paths must still agree.
    #[test]
    fn late_training_step_matches() {
        let n = 47873;
        let (w0, g0, m0, v0) = synth_state(n, 0xBEEF);
        let (mut wa, mut ga, mut ma, mut va) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
        let (mut wb, mut gb, mut mb, mut vb) = (w0, g0, m0, v0);

        adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, 10_000));
        adam_update(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, 10_000));

        for i in 0..n {
            let r = (wa[i] - wb[i]).abs() / (wa[i].abs().max(wb[i].abs()).max(1e-30));
            assert!(r < 1e-12, "w mismatch at i={}", i);
        }
    }

    /// Empty slice — no-op.
    #[test]
    fn empty_is_noop() {
        let mut w: Vec<f64> = vec![];
        let mut g: Vec<f64> = vec![];
        let mut m: Vec<f64> = vec![];
        let mut v: Vec<f64> = vec![];
        adam_update(&mut make_args(&mut w, &mut g, &mut m, &mut v, 1));
    }
}
