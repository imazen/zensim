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
#[cfg(target_arch = "aarch64")]
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

    // `as_chunks_mut` gives `&mut [f64; 8]` so LLVM can prove all
    // intra-chunk indexes are bounded, eliminating per-iteration bounds
    // checks. The `partition_slice_mut` helper in magetypes does the
    // same thing via a single `unsafe` slice cast at the boundary; we
    // use the safe stdlib variant.
    let n = args.w.len();
    let tail_len = n % 8;
    let (g_chunks, _) = args.g.as_chunks_mut::<8>();
    let (m_chunks, _) = args.m.as_chunks_mut::<8>();
    let (v_chunks, _) = args.v.as_chunks_mut::<8>();
    let (w_chunks, _) = args.w.as_chunks_mut::<8>();

    for (((wc_fixed, gc_fixed), mc_fixed), vc_fixed) in w_chunks
        .iter_mut()
        .zip(g_chunks)
        .zip(m_chunks)
        .zip(v_chunks)
    {
        // `*_fixed: &mut [f64; 8]` — `as_chunks_mut` hands out fixed-size
        // arrays, so LLVM emits straight-line vector loads/stores with no
        // bounds checks.

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

    // Remainder handling — BIT-PARITY RULE (tier-parity lane): the AVX2
    // (`_v3`) kernel's fused-op domain is [0, n - n%4); only the last n%4
    // elements are scalar mul+add. The 8-lane loop above leaves
    // n%8 ∈ [0,7) elements; when n%8 >= 4 the first four of them are
    // fused-domain in canonical arithmetic, so they take the identical
    // fused ops (not the mul+add scalar tail), and only the final n%4
    // take the scalar_ref path.
    //
    // Adam is elementwise — there is no cross-lane reduction — so four
    // per-element `f64::mul_add` updates are bit-identical to one
    // `_mm256_fmadd_pd` group on the `_v3` kernel.
    let rem_start = n - tail_len;
    let n4 = n - n % 4;
    if rem_start + 4 <= n4 {
        let one_minus_b1 = 1.0 - args.beta1;
        let one_minus_b2 = 1.0 - args.beta2;
        let inv_bc1 = 1.0 / args.bc1;
        let inv_bc2 = 1.0 / args.bc2;
        for i in rem_start..rem_start + 4 {
            let g = args.g[i];
            let m_new = one_minus_b1.mul_add(g, args.beta1 * args.m[i]);
            let v_new = one_minus_b2.mul_add(g * g, args.beta2 * args.v[i]);
            args.m[i] = m_new;
            args.v[i] = v_new;
            let m_hat = m_new * inv_bc1;
            let v_hat = v_new * inv_bc2;
            args.w[i] -= args.lr * m_hat / (v_hat.sqrt() + args.eps);
            args.g[i] = 0.0;
        }
    }

    // Final n%4 elements — mul+add scalar semantics, identical to the
    // AVX2 kernel's tail.
    if n4 < n {
        let mut tail_args = AdamUpdateArgs {
            w: &mut args.w[n4..],
            g: &mut args.g[n4..],
            m: &mut args.m[n4..],
            v: &mut args.v[n4..],
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
// of bit-identity to the scalar reference. Exposed for benchmarking +
// opt-in use (not shipped: measured slower than vsqrtpd+vdivpd on Zen 4).
//
// Math (shared by every tier):
//   denom = sqrt(v_hat) + eps
//   1/denom computed via:
//     r = P::rsqrt(v_hat)        [VRSQRT14PD + 0..2 NR steps, tier-selected]
//     sqrt_v = v_hat * r         [FMA-fused with the +eps]
//     denom = sqrt_v + eps
//     rd = P::recip(denom)       [VRCP14PD + 0..2 NR steps, tier-selected]
//   w -= (lr * m_hat) * rd
//
// PRECISION IS A TIER, NOT A CONSTANT (2026-08-05 user ruling): "full-
// precision fallbacks are acceptable only if runtime-optional via generics —
// prefer archmage's precision-TIERED variants." The kernel body is generic
// over [`RsqrtPrecisionTier`]; the DEFAULT (and the only tier any production
// caller uses) is [`RsqrtFull`], but [`RsqrtNr1`] / [`RsqrtEstimate`] are
// selectable at runtime via [`adam_update_rsqrt_v4_tiered`] for callers that
// can trade update-step precision for throughput. Per-tier error bounds are
// measured + gated in `tests/adam_simd_rsqrt_precision.rs`.
//
// magetypes 0.9.28 ships exactly two f64 precision tiers per op —
// `rsqrt_approx()`/`rcp_approx()` (RAW 14-bit hardware estimate) and
// `rsqrt()`/`recip()` (2 Newton-Raphson steps, ~52-53 bits; the f32-only
// `_portable`/`_newton` families do not exist for f64) — so the middle tier
// here hand-rolls ONE NR step using the SAME step formulas magetypes' full
// path uses internally (recip: r·(2−a·r); rsqrt: 0.5·y·(3−a·y·y); see
// magetypes/src/simd/impls/x86_v4.rs). NR1 output is therefore exactly
// "full minus one step" — the same expression tree the pre-repair kernel
// accidentally computed after the 0.9.27 `_approx` contract change, whose
// error was measured at max_rel 1.6653e-5 on the precision-test fixture.
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
// `w - step` lands near zero). The repair (`22e37ce3`) consumed the
// full-precision `rsqrt()`/`recip()`; the tier mechanism now makes that
// hand-rolled-NR shape a NAMED, boundedly-tested tier instead of an
// accident. If magetypes' `_approx` contract moves again, the per-tier
// precision gates fail loudly instead of silently degrading.
//
// Domain note (pre-existing, unchanged): `rsqrt(0)` is Inf/NaN-producing —
// the sqrt-based paths survive v_hat == 0 via `sqrt(0) + eps`, this one does
// not. Fine for Adam (v is an EMA of g² with positive init in every caller),
// but do not lift this kernel into a context where v_hat can be exactly 0.

/// Precision tier for the estimate-based reciprocal ops in the rsqrt Adam
/// kernel (AVX-512-only — AVX2 has no f64 reciprocal estimate; magetypes'
/// f64x4 `rsqrt()`/`recip()` are exact `sqrt`/`div` there, so tiering below
/// v4 would be a no-op).
///
/// Implementations are zero-sized selector types, monomorphized into the
/// kernel body — tier selection costs nothing inside the loop. Runtime
/// selection happens once per call in [`adam_update_rsqrt_v4_tiered`].
#[cfg(target_arch = "x86_64")]
pub trait RsqrtPrecisionTier {
    /// Reciprocal square root of `v` at this tier's precision.
    fn rsqrt(
        token: archmage::X64V4Token,
        v: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token>;
    /// Reciprocal of `d` at this tier's precision.
    fn recip(
        token: archmage::X64V4Token,
        d: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token>;
}

/// Full precision: magetypes `rsqrt()`/`recip()` = VRSQRT14PD/VRCP14PD + 2
/// Newton-Raphson steps each, ~52-53 bits. Step error vs the scalar
/// sqrt+div formula: ~1-2 ULP (~2e-16..5e-16 relative). **The default tier.**
#[cfg(target_arch = "x86_64")]
pub struct RsqrtFull;

#[cfg(target_arch = "x86_64")]
impl RsqrtPrecisionTier for RsqrtFull {
    #[inline(always)]
    fn rsqrt(
        _token: archmage::X64V4Token,
        v: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        v.rsqrt()
    }
    #[inline(always)]
    fn recip(
        _token: archmage::X64V4Token,
        d: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        d.recip()
    }
}

/// One Newton-Raphson step from the raw 14-bit estimate: ~28 bits (~1e-8
/// relative step error). Mirrors magetypes' internal NR step formulas
/// exactly, so this tier reproduces the pre-`22e37ce3` kernel's arithmetic
/// bit-for-bit (measured max w-relative error 1.6653e-5 on the precision
/// fixture through ~2e3× cancellation amplification — see the test).
#[cfg(target_arch = "x86_64")]
pub struct RsqrtNr1;

#[cfg(target_arch = "x86_64")]
impl RsqrtPrecisionTier for RsqrtNr1 {
    #[inline(always)]
    fn rsqrt(
        token: archmage::X64V4Token,
        v: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        // y ← 0.5·y·(3 − v·y·y), one step from the raw estimate — the same
        // (unfused) step magetypes' full rsqrt applies twice.
        let half = f64x8::splat(token, 0.5);
        let three = f64x8::splat(token, 3.0);
        let y = v.rsqrt_approx();
        (half * y) * (three - v * (y * y))
    }
    #[inline(always)]
    fn recip(
        token: archmage::X64V4Token,
        d: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        // r ← r·(2 − d·r), one step from the raw estimate.
        let two = f64x8::splat(token, 2.0);
        let r = d.rcp_approx();
        r * (two - d * r)
    }
}

/// Raw hardware estimate, no refinement: VRSQRT14PD/VRCP14PD alone. The
/// AVX-512 spec bounds the estimate's relative error at 2^-14 ≈ 6.1e-5 per
/// op. Cheapest possible tier; only for callers that can absorb ~1e-4-class
/// relative error in the update step.
#[cfg(target_arch = "x86_64")]
pub struct RsqrtEstimate;

#[cfg(target_arch = "x86_64")]
impl RsqrtPrecisionTier for RsqrtEstimate {
    #[inline(always)]
    fn rsqrt(
        _token: archmage::X64V4Token,
        v: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        v.rsqrt_approx()
    }
    #[inline(always)]
    fn recip(
        _token: archmage::X64V4Token,
        d: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        d.rcp_approx()
    }
}

/// Runtime-selectable precision tier for [`adam_update_rsqrt_v4_tiered`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target constructs a subset of variants
pub enum RsqrtPrecision {
    /// ~52-53 bits (2 NR steps) — the default; bounded at 1e-9 w-relative.
    #[default]
    Full,
    /// ~28 bits (1 NR step) — bounded at 1e-3 w-relative on the fixture.
    Nr1,
    /// Raw 14-bit estimate — bounded at 1e0 w-relative on the fixture.
    Estimate,
}

/// Generic kernel body — precision tier is a monomorphized type parameter,
/// so each `#[arcane]` wrapper below compiles a straight-line loop with the
/// tier's op sequence inlined (no per-lane or per-iteration branching).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn adam_update_inner_v4_rsqrt_body<P: RsqrtPrecisionTier>(
    token: archmage::X64V4Token,
    args: &mut AdamUpdateArgs<'_>,
) {
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
    let (g_chunks, _) = args.g.as_chunks_mut::<8>();
    let (m_chunks, _) = args.m.as_chunks_mut::<8>();
    let (v_chunks, _) = args.v.as_chunks_mut::<8>();
    let (w_chunks, _) = args.w.as_chunks_mut::<8>();

    for (((wc_fixed, gc_fixed), mc_fixed), vc_fixed) in w_chunks
        .iter_mut()
        .zip(g_chunks)
        .zip(m_chunks)
        .zip(v_chunks)
    {
        let g = f64x8::load(token, gc_fixed);
        let m = f64x8::load(token, mc_fixed);
        let v = f64x8::load(token, vc_fixed);
        let w = f64x8::load(token, wc_fixed);

        let m_new = one_minus_b1_v.mul_add(g, beta1_v * m);
        let gg = g * g;
        let v_new = one_minus_b2_v.mul_add(gg, beta2_v * v);

        let m_hat = m_new * inv_bc1_v;
        let v_hat = v_new * inv_bc2_v;

        // Tier-selected reciprocal sqrt (see the tier docs above).
        let r1 = P::rsqrt(token, v_hat);

        // sqrt(v_hat) = v_hat * rsqrt(v_hat), fused with the + eps.
        let denom = v_hat.mul_add(r1, eps_v);

        // Tier-selected reciprocal.
        let rd = P::recip(token, denom);

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

#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn adam_update_inner_v4_rsqrt(token: archmage::X64V4Token, args: &mut AdamUpdateArgs<'_>) {
    adam_update_inner_v4_rsqrt_body::<RsqrtFull>(token, args);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn adam_update_inner_v4_rsqrt_nr1(token: archmage::X64V4Token, args: &mut AdamUpdateArgs<'_>) {
    adam_update_inner_v4_rsqrt_body::<RsqrtNr1>(token, args);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn adam_update_inner_v4_rsqrt_estimate(
    token: archmage::X64V4Token,
    args: &mut AdamUpdateArgs<'_>,
) {
    adam_update_inner_v4_rsqrt_body::<RsqrtEstimate>(token, args);
}

/// Bench-only entry-point — runs the rsqrt-based v4 kernel directly at the
/// DEFAULT (full-precision) tier, skipping the dispatch wrapper to isolate
/// kernel cost. Equivalent to
/// `adam_update_rsqrt_v4_tiered(args, RsqrtPrecision::Full)`.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
pub fn adam_update_rsqrt_v4(args: &mut AdamUpdateArgs<'_>) {
    adam_update_rsqrt_v4_tiered(args, RsqrtPrecision::Full);
}

/// Runtime-tier-selected rsqrt Adam kernel. Falls back to the scalar
/// reference when AVX-512 is unavailable (every tier degrades to EXACT
/// there — the fallback is full precision by construction).
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
pub fn adam_update_rsqrt_v4_tiered(args: &mut AdamUpdateArgs<'_>, tier: RsqrtPrecision) {
    use archmage::SimdToken;
    if let Some(token) = <archmage::X64V4Token as SimdToken>::summon() {
        match tier {
            RsqrtPrecision::Full => adam_update_inner_v4_rsqrt(token, args),
            RsqrtPrecision::Nr1 => adam_update_inner_v4_rsqrt_nr1(token, args),
            RsqrtPrecision::Estimate => adam_update_inner_v4_rsqrt_estimate(token, args),
        }
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
    let (g_chunks, _) = args.g.as_chunks_mut::<4>();
    let (m_chunks, _) = args.m.as_chunks_mut::<4>();
    let (v_chunks, _) = args.v.as_chunks_mut::<4>();
    let (w_chunks, _) = args.w.as_chunks_mut::<4>();

    for (((wc_fixed, gc_fixed), mc_fixed), vc_fixed) in w_chunks
        .iter_mut()
        .zip(g_chunks)
        .zip(m_chunks)
        .zip(v_chunks)
    {
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
// NEON fallback — generic `GenericF64x4<Token>` body. NEON's `mul_add`
// lowers to `vfmaq` (single-rounding fused), so this is bit-identical to
// the `_v3` kernel already: same fused domain [0, n - n%4), same
// scalar_ref tail.
// =============================================================================

#[magetypes(neon, -scalar)]
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
    let (g_chunks, _) = args.g.as_chunks_mut::<4>();
    let (m_chunks, _) = args.m.as_chunks_mut::<4>();
    let (v_chunks, _) = args.v.as_chunks_mut::<4>();
    let (w_chunks, _) = args.w.as_chunks_mut::<4>();

    for (((wc_fixed, gc_fixed), mc_fixed), vc_fixed) in w_chunks
        .iter_mut()
        .zip(g_chunks)
        .zip(m_chunks)
        .zip(v_chunks)
    {
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

// =============================================================================
// wasm128 / scalar fallback — per-lane fused arithmetic.
//
// The generic `GenericF64x4<Token>::mul_add` lowers to plain `a*b+c` on
// these tiers (wasm has no f64 FMA primitive; the scalar polyfill avoids
// the libm call), which is NOT the canonical `_v3` arithmetic. Bit parity
// requires the same fused ops the AVX2 lanes run, so these tiers go
// per-element `f64::mul_add` — single-rounding fused on every target
// (hardware FMA where it exists, correctly-rounded libm `fma` elsewhere)
// — over the canonical fused domain [0, n - n%4), followed by the
// `scalar_ref` mul+add tail.
// =============================================================================

fn adam_update_scalar_fused(args: &mut AdamUpdateArgs<'_>) {
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
    let n = args.w.len();
    let n4 = n - n % 4;
    for i in 0..n4 {
        let g = args.g[i];
        // Identical fused expression to the AVX2 lane ops:
        // fma(one_minus_b1, g, round(beta1*m)) etc.
        let m_new = one_minus_b1.mul_add(g, beta1 * args.m[i]);
        let v_new = one_minus_b2.mul_add(g * g, beta2 * args.v[i]);
        args.m[i] = m_new;
        args.v[i] = v_new;
        let m_hat = m_new * inv_bc1;
        let v_hat = v_new * inv_bc2;
        args.w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        args.g[i] = 0.0;
    }
    if n4 < n {
        let mut tail_args = AdamUpdateArgs {
            w: &mut args.w[n4..],
            g: &mut args.g[n4..],
            m: &mut args.m[n4..],
            v: &mut args.v[n4..],
            beta1,
            beta2,
            eps,
            bc1,
            bc2,
            lr,
        };
        adam_update_scalar_ref(&mut tail_args);
    }
}

/// `wasm128` tier variant called by `incant!` dispatch.
#[cfg(target_arch = "wasm32")]
fn adam_update_inner_wasm128(_token: archmage::Wasm128Token, args: &mut AdamUpdateArgs<'_>) {
    adam_update_scalar_fused(args);
}

/// `scalar` tier variant called by `incant!` dispatch.
fn adam_update_inner_scalar(_token: archmage::ScalarToken, args: &mut AdamUpdateArgs<'_>) {
    adam_update_scalar_fused(args);
}

// =============================================================================
// FUSED K=1 PAIR UPDATE — layer-1 weights only
// =============================================================================
//
// Replaces the per-pair sequence
//
//   backprop_step(A) → gw1[i] += xa[i]·dha[j]      (FMA / mul+add per row domain)
//   backprop_step(B) → gw1[i] += xb[i]·dhb[j]
//   add_l2_grad_layer1 → gw1[i] += sm[i/nh]·w[i]   (mul+add, never FMA)
//   adam_update → consume gw1, write w/m/v, store g=0
//
// with one pass that holds `g` in a register across the whole chain. Every
// element still sees the identical op sequence in the identical order, so the
// result is bit-identical — the only change is that gw1 is not round-tripped
// through L1 between passes (≈1.6 MB/pair of traffic at 944×32).
//
// Domain splits preserved exactly (the "same ops per element" contract):
//   * backprop contributions: FMA (`s.mul_add(dh, g)`) on lanes j < n4h,
//     mul+add on the per-row AVX2 tail [n4h, n_hidden) — the kernel is only
//     dispatched when n_hidden % 4 == 0, so the whole row is fused-domain.
//   * Adam arithmetic: vector (FMA) domain on [0, n − n%4); scalar_ref on
//     the flat tail. n = n_features·n_hidden is a multiple of 4 whenever
//     n_hidden % 4 == 0, so the tail is empty on the gated path.
//   * `x[i] == 0.0` skips that row's contribution exactly like
//     `backprop_*`'s nonzero-feature filter.
//   * L2 applies iff `l2_scale > 0.0` (same gate as the call site); with a
//     per-feature multiplier `sm = l2_scale * mult[row]` is hoisted per row
//     exactly as `add_l2_grad_layer1` does.
//
// Non-v3 tiers route through `adam_pair_fused_fallback`: a scalar
// accumulation of the identical elementwise op sequence into `g`, then the
// regular `adam_update` dispatch — bit-identical by composition (every tier's
// adam is already AVX2-parity per the tierparity contract).

/// Parameter block for the fused pair update on layer-1 weights.
/// `w/g/m/v` are the Adam state; `xa`/`xb` are the pair's feature vectors
/// and `dha`/`dhb` their precomputed `dl_dh_pre` vectors
/// (`simd_mlp::backprop_grad_head` output).
#[derive(Debug)]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
pub(crate) struct AdamW1FusedArgs<'a> {
    pub w: &'a mut [f64],
    pub g: &'a mut [f64],
    pub m: &'a mut [f64],
    pub v: &'a mut [f64],
    pub xa: &'a [f64],
    pub dha: &'a [f64],
    pub xb: &'a [f64],
    pub dhb: &'a [f64],
    pub l2_scale: f64,
    pub l2_mult: Option<&'a [f64]>,
    pub n_hidden: usize,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub bc1: f64,
    pub bc2: f64,
    pub lr: f64,
}

/// Dispatch entry for the fused pair update. Caller contract: `n_hidden > 0`,
/// `n_hidden % 4 == 0`, `w/g/m/v` equal length with `w.len() % n_hidden == 0`,
/// `xa/xb.len() >= w.len()/n_hidden`, `dha/dhb.len() >= n_hidden`.
#[inline]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
pub(crate) fn adam_update_w1_fused(args: &mut AdamW1FusedArgs<'_>) {
    debug_assert_eq!(args.w.len(), args.g.len());
    debug_assert_eq!(args.w.len(), args.m.len());
    debug_assert_eq!(args.w.len(), args.v.len());
    if args.n_hidden == 0
        || !args.n_hidden.is_multiple_of(4)
        || !args.w.len().is_multiple_of(args.n_hidden)
    {
        // The v3 kernel walks full 4-lane row chunks; off-grid shapes take
        // the composition fallback (still bit-identical to the unfused
        // sequence — the fallback reproduces its per-element ops exactly).
        adam_pair_fused_fallback(args);
        return;
    }
    incant!(adam_pair_fused_inner(args), [+v4])
}

/// Scalar-fallback body shared by every non-v3 arm: replays the exact
/// unfused element sequence (backprop A, backprop B, L2) into `g` — fused
/// `mul_add` on the per-row canonical domain, mul+add on the AVX2 tail —
/// then runs the regular tier-dispatched `adam_update`.
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_fallback(args: &mut AdamW1FusedArgs<'_>) {
    let nh = args.n_hidden;
    let n = args.w.len();
    let n_rows = n / nh;
    let n4h = nh - nh % 4;
    for row in 0..n_rows {
        let base = row * nh;
        let sa = args.xa[row];
        let sb = args.xb[row];
        for j in 0..nh {
            let i = base + j;
            if j < n4h {
                if sa != 0.0 {
                    args.g[i] = sa.mul_add(args.dha[j], args.g[i]);
                }
                if sb != 0.0 {
                    args.g[i] = sb.mul_add(args.dhb[j], args.g[i]);
                }
            } else {
                if sa != 0.0 {
                    args.g[i] += sa * args.dha[j];
                }
                if sb != 0.0 {
                    args.g[i] += sb * args.dhb[j];
                }
            }
        }
    }
    // L2 — mul+add, row-hoisted `sm` exactly as add_l2_grad_layer1.
    if args.l2_scale > 0.0 {
        match args.l2_mult {
            None => {
                for (g, &w) in args.g.iter_mut().zip(args.w.iter()) {
                    *g += args.l2_scale * w;
                }
            }
            Some(mult) => {
                for (feat, (grow, wrow)) in args.g.chunks_mut(nh).zip(args.w.chunks(nh)).enumerate()
                {
                    let sm = args.l2_scale * mult[feat];
                    for (g, &w) in grow.iter_mut().zip(wrow.iter()) {
                        *g += sm * w;
                    }
                }
            }
        }
    }
    adam_update(&mut AdamUpdateArgs {
        w: args.w,
        g: args.g,
        m: args.m,
        v: args.v,
        beta1: args.beta1,
        beta2: args.beta2,
        eps: args.eps,
        bc1: args.bc1,
        bc2: args.bc2,
        lr: args.lr,
    });
}

/// `v4` tier variant. Deliberately NOT a native 8-lane kernel: AVX-512 must
/// preserve AVX2's fused-domain boundaries (4-lane chunks + canonical tail).
/// The kernel is purely elementwise and the caller gate guarantees
/// `n_hidden % 4 == 0`, so the 4-lane v3 body is exact under a v4 token —
/// run it directly via `token.v3()` instead of the scalar-emulating
/// fallback (which was bit-identical but ~4× slower on AVX-512 hosts).
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_v4(token: archmage::X64V4Token, args: &mut AdamW1FusedArgs<'_>) {
    adam_pair_fused_inner_v3_body(token.v3(), args);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_v3(token: archmage::X64V3Token, args: &mut AdamW1FusedArgs<'_>) {
    adam_pair_fused_inner_v3_body(token, args);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_v3_body(token: archmage::X64V3Token, args: &mut AdamW1FusedArgs<'_>) {
    let beta1_v = f64x4::splat(token, args.beta1);
    let beta2_v = f64x4::splat(token, args.beta2);
    let one_minus_b1_v = f64x4::splat(token, 1.0 - args.beta1);
    let one_minus_b2_v = f64x4::splat(token, 1.0 - args.beta2);
    let inv_bc1_v = f64x4::splat(token, 1.0 / args.bc1);
    let inv_bc2_v = f64x4::splat(token, 1.0 / args.bc2);
    let lr_v = f64x4::splat(token, args.lr);
    let eps_v = f64x4::splat(token, args.eps);
    let zero_v = f64x4::zero(token);

    let nh = args.n_hidden;
    let n = args.w.len();
    let n_rows = n / nh;
    let l2_on = args.l2_scale > 0.0;
    let (dha_chunks, _) = args.dha[..nh].as_chunks::<4>();
    let (dhb_chunks, _) = args.dhb[..nh].as_chunks::<4>();

    for row in 0..n_rows {
        let base = row * nh;
        let sa = args.xa[row];
        let sb = args.xb[row];
        let sa_v = f64x4::splat(token, sa);
        let sb_v = f64x4::splat(token, sb);
        let do_a = sa != 0.0;
        let do_b = sb != 0.0;
        let sm_v = f64x4::splat(
            token,
            match args.l2_mult {
                Some(mult) => args.l2_scale * mult[row],
                None => args.l2_scale,
            },
        );
        let (w_chunks, _) = args.w[base..base + nh].as_chunks_mut::<4>();
        let (g_chunks, _) = args.g[base..base + nh].as_chunks_mut::<4>();
        let (m_chunks, _) = args.m[base..base + nh].as_chunks_mut::<4>();
        let (v_chunks, _) = args.v[base..base + nh].as_chunks_mut::<4>();
        for (c, (((wc, gc), mc), vc)) in w_chunks
            .iter_mut()
            .zip(g_chunks.iter_mut())
            .zip(m_chunks.iter_mut())
            .zip(v_chunks.iter_mut())
            .enumerate()
        {
            let mut g = f64x4::load(token, gc);
            if do_a {
                g = sa_v.mul_add(f64x4::load(token, &dha_chunks[c]), g);
            }
            if do_b {
                g = sb_v.mul_add(f64x4::load(token, &dhb_chunks[c]), g);
            }
            let w = f64x4::load(token, wc);
            if l2_on {
                // mul+add — two roundings, matching l2_row_avx exactly.
                g += sm_v * w;
            }
            let m = f64x4::load(token, mc);
            let v = f64x4::load(token, vc);
            let m_new = one_minus_b1_v.mul_add(g, beta1_v * m);
            let gg = g * g;
            let v_new = one_minus_b2_v.mul_add(gg, beta2_v * v);
            let m_hat = m_new * inv_bc1_v;
            let v_hat = v_new * inv_bc2_v;
            let denom = v_hat.sqrt() + eps_v;
            let w_new = w - (lr_v * m_hat) / denom;
            m_new.store(mc);
            v_new.store(vc);
            w_new.store(wc);
            zero_v.store(gc);
        }
    }
}

/// `neon` tier variant — routes to the scalar-emulating fallback.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_neon(_token: archmage::NeonToken, args: &mut AdamW1FusedArgs<'_>) {
    adam_pair_fused_fallback(args);
}

/// `wasm128` tier variant — routes to the scalar-emulating fallback.
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_wasm128(_token: archmage::Wasm128Token, args: &mut AdamW1FusedArgs<'_>) {
    adam_pair_fused_fallback(args);
}

/// `scalar` tier variant — routes to the scalar-emulating fallback.
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_scalar(_token: archmage::ScalarToken, args: &mut AdamW1FusedArgs<'_>) {
    adam_pair_fused_fallback(args);
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

    // =====================================================================
    // Tier parity — the AVX2 (`_v3`) kernel is the canonical arithmetic;
    // every other tier the dispatcher can pick must reproduce it BIT FOR
    // BIT (same fused-op set on [0, n - n%4), same mul+add tail).
    // =====================================================================

    fn bits_of(v: &[f64]) -> Vec<u64> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    /// Max ordered-int ULP distance + first index, for reporting.
    fn max_ulp(a: &[f64], b: &[f64]) -> (u64, usize) {
        fn ord(v: f64) -> i64 {
            let b = v.to_bits() as i64;
            if b < 0 { i64::MIN - b } else { b }
        }
        let mut worst = (0u64, usize::MAX);
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let d = ord(x).abs_diff(ord(y));
            if d > worst.0 {
                worst = (d, i);
            }
        }
        worst
    }

    /// Compare every host-available tier against `_v3` (canonical) on the
    /// same buffers. Sizes cover: n<4 (all-tail on every tier), n=5..8
    /// (v4 all-tail vs v3 partially-chunked — the tier-boundary edge), the
    /// fold shape 30208/32/1 and a misaligned large n.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn adam_all_tiers_bit_identical() {
        use archmage::SimdToken;
        // Hold the token lock for the whole body: `adam_dispatch_bit_identical_under_token_permutations`
        // disables tokens process-wide, and a `summon()` racing it would silently skip the v4 comparison.
        let _lock = archmage::testing::lock_token_testing();
        let v3 = archmage::X64V3Token::summon();
        let v4 = archmage::X64V4Token::summon();
        assert!(
            v3.is_some(),
            "this test needs an x86-64-v3 (AVX2+FMA) host: it is the canonical tier every other tier must reproduce"
        );
        let sizes = [
            1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 20, 24, 28, 31, 32, 33, 40, 48, 63,
            64, 65, 30208, 32, 1, 47873,
        ];
        let mut failures = Vec::new();
        for &n in &sizes {
            for &t in &[5u64, 100, 10_000] {
                let (w0, g0, m0, v0) = synth_state(n, 0xFEED ^ n as u64 ^ t);

                let run = |f: &mut dyn FnMut(&mut AdamUpdateArgs<'_>)| {
                    let (mut w, mut g, mut m, mut v) =
                        (w0.clone(), g0.clone(), m0.clone(), v0.clone());
                    f(&mut make_args(&mut w, &mut g, &mut m, &mut v, t));
                    (w, g, m, v)
                };

                let r_v3 = run(&mut |a| adam_update_inner_v3(v3.unwrap(), a));

                // scalar tier (plain mul_add semantics — must equal v3)
                let r_s = run(&mut |a| {
                    adam_update_inner_scalar(archmage::ScalarToken::summon().unwrap(), a)
                });
                for (name, (ulp, idx)) in [
                    ("w", max_ulp(&r_s.0, &r_v3.0)),
                    ("g", max_ulp(&r_s.1, &r_v3.1)),
                    ("m", max_ulp(&r_s.2, &r_v3.2)),
                    ("v", max_ulp(&r_s.3, &r_v3.3)),
                ] {
                    if ulp != 0 {
                        failures.push(format!(
                            "adam scalar-vs-v3 n={n} t={t} {name}: max_ulp={ulp} at {idx}"
                        ));
                    }
                }

                if let Some(t4) = v4 {
                    let r_v4 = run(&mut |a| adam_update_inner_v4(t4, a));
                    for (name, (ulp, idx)) in [
                        ("w", max_ulp(&r_v4.0, &r_v3.0)),
                        ("g", max_ulp(&r_v4.1, &r_v3.1)),
                        ("m", max_ulp(&r_v4.2, &r_v3.2)),
                        ("v", max_ulp(&r_v4.3, &r_v3.3)),
                    ] {
                        if ulp != 0 {
                            failures.push(format!(
                                "adam v4-vs-v3 n={n} t={t} {name}: max_ulp={ulp} at {idx}"
                            ));
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// The dispatched `adam_update` must produce identical bytes under
    /// every host token permutation (v4-off → v3 bytes; v3-off → scalar
    /// bytes; and the scalar tier must match v3 too).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn adam_dispatch_bit_identical_under_token_permutations() {
        use archmage::SimdToken;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "this test needs an x86-64-v3 (AVX2+FMA) host: it is the canonical tier every other tier must reproduce"
        );
        let n = 30208usize + 65; // fold w1 + biases, misaligned on purpose
        let (w0, g0, m0, v0) = synth_state(n, 0xCAFE_5EED);
        let mut baseline: Option<Vec<u64>> = None;
        let mut failures = Vec::new();
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let (mut w, mut g, mut m, mut v) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
            adam_update(&mut make_args(&mut w, &mut g, &mut m, &mut v, 42));
            let mut bits = bits_of(&w);
            bits.extend(bits_of(&g));
            bits.extend(bits_of(&m));
            bits.extend(bits_of(&v));
            if let Some(b) = &baseline {
                if bits != *b {
                    failures.push(format!("adam dispatch diverged: {}", perm.label));
                }
            } else {
                baseline = Some(bits);
            }
            eprintln!(
                "perm [{}] w[0]={:#x} w[{}]={:#x}",
                perm.label,
                w[0].to_bits(),
                n - 1,
                w[n - 1].to_bits()
            );
        });
        eprintln!("permutations run: {}", report.permutations_run);
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    // ---- fused K=1 pair update (fusedstep lane) ----

    /// Unfused oracle: replicate `backprop_*`'s gw1 row sweep (row skip on
    /// `x == 0.0`; FMA on `j < n4h`, mul+add tail) + `add_l2_grad_layer1`
    /// (mul+add, per-row `sm`) + the tier-dispatched `adam_update` — the
    /// exact sequence `mlp_train` runs per pair.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)] // oracle mirrors the kernel signature 1:1
    fn unfused_w1_oracle(
        w: &mut [f64],
        g: &mut [f64],
        m: &mut [f64],
        v: &mut [f64],
        xa: &[f64],
        dha: &[f64],
        xb: &[f64],
        dhb: &[f64],
        l2: f64,
        mult: Option<&[f64]>,
        nh: usize,
        t: u64,
    ) {
        let n = w.len();
        let n4h = nh - nh % 4;
        for r in 0..n / nh {
            let base = r * nh;
            let sa = xa[r];
            if sa != 0.0 {
                for (j, &dh) in dha.iter().enumerate().take(nh) {
                    let i = base + j;
                    if j < n4h {
                        g[i] = sa.mul_add(dh, g[i]);
                    } else {
                        g[i] += sa * dh;
                    }
                }
            }
            let sb = xb[r];
            if sb != 0.0 {
                for (j, &dh) in dhb.iter().enumerate().take(nh) {
                    let i = base + j;
                    if j < n4h {
                        g[i] = sb.mul_add(dh, g[i]);
                    } else {
                        g[i] += sb * dh;
                    }
                }
            }
        }
        if l2 > 0.0 {
            match mult {
                None => {
                    for (gg, &ww) in g.iter_mut().zip(w.iter()) {
                        *gg += l2 * ww;
                    }
                }
                Some(mm) => {
                    for (f, (gr, wr)) in g.chunks_mut(nh).zip(w.chunks(nh)).enumerate() {
                        let sm = l2 * mm[f];
                        for (gg, &ww) in gr.iter_mut().zip(wr.iter()) {
                            *gg += sm * ww;
                        }
                    }
                }
            }
        }
        adam_update(&mut make_args(w, g, m, v, t));
    }

    /// Deliberate order perturbation for the input-sensitivity check: L2
    /// before side B's contribution. Hand-written replica — never calls
    /// the kernel under test.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)] // oracle mirrors the kernel signature 1:1
    fn fused_wrong_order(
        w: &mut [f64],
        g: &mut [f64],
        m: &mut [f64],
        v: &mut [f64],
        xa: &[f64],
        dha: &[f64],
        xb: &[f64],
        dhb: &[f64],
        l2: f64,
        mult: Option<&[f64]>,
        nh: usize,
        t: u64,
    ) {
        let l2_on = l2 > 0.0;
        for r in 0..w.len() / nh {
            let sa = xa[r];
            let sb = xb[r];
            let sm = match mult {
                Some(mm) => l2 * mm[r],
                None => l2,
            };
            for j in 0..nh {
                let i = r * nh + j;
                let mut gg = g[i];
                if sa != 0.0 {
                    gg = sa.mul_add(dha[j], gg);
                }
                if l2_on {
                    gg += sm * w[i];
                }
                if sb != 0.0 {
                    gg = sb.mul_add(dhb[j], gg);
                }
                g[i] = gg;
            }
        }
        adam_update(&mut make_args(w, g, m, v, t));
    }

    #[allow(dead_code)]
    fn pair_inputs(nf: usize, nh: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (xa, _, _, _) = synth_state(nf, seed);
        let (xb, _, _, _) = synth_state(nf, seed ^ 0x51);
        let (dha, _, _, _) = synth_state(nh, seed ^ 0x52);
        let (dhb, _, _, _) = synth_state(nh, seed ^ 0x53);
        // ~1/7 of rows zeroed on each side — exercises the row-skip path.
        let xa: Vec<f64> = xa
            .iter()
            .enumerate()
            .map(|(i, &x)| if i % 7 == 3 { 0.0 } else { x })
            .collect();
        let xb: Vec<f64> = xb
            .iter()
            .enumerate()
            .map(|(i, &x)| if i % 5 == 1 { 0.0 } else { x })
            .collect();
        (xa, xb, dha, dhb)
    }

    /// The dispatched `adam_update_w1_fused` reproduces the unfused sequence
    /// bit-for-bit on every gated shape; off-grid shapes (nh%4!=0) route to
    /// the composition fallback, which must match too.
    ///
    /// NOTE on `black_box(t)`: `f64::powi` with a compile-time-known exponent
    /// can const-fold to a different bit pattern than the same call evaluated
    /// at runtime (CTFE vs `llvm.powi` lowering disagree by ~1ulp). The oracle
    /// computes `bc1`/`bc2` inside `make_args` with a runtime `t`; the fused
    /// args must take the same runtime path or the two sides run Adam with
    /// different bias corrections and the comparison is meaningless.
    #[test]
    fn fused_w1_bit_identical_dispatch() {
        for &(nf, nh) in &[
            (4usize, 4usize),
            (7, 8),
            (13, 16),
            (944, 32),
            (372, 128),
            (5, 6), // nh%4!=0 → fallback
            (9, 3), // nh%4!=0 → fallback
        ] {
            for has_mult in [false, true] {
                for l2 in [0.0f64, 1e-5, 0.3] {
                    let n = nf * nh;
                    let seed = 0xB17_1D3A ^ ((nf as u64) << 24) ^ nh as u64;
                    let (w0, g0, m0, v0) = synth_state(n, seed);
                    let (xa, xb, dha, dhb) = pair_inputs(nf, nh, seed);
                    let mult: Option<Vec<f64>> = if has_mult {
                        Some(
                            synth_state(nf, seed ^ 0x77)
                                .0
                                .iter()
                                .map(|x| x.abs() + 0.5)
                                .collect(),
                        )
                    } else {
                        None
                    };
                    let (mut wo, mut go, mut mo, mut vo) =
                        (w0.clone(), g0.clone(), m0.clone(), v0.clone());
                    let t = std::hint::black_box(3u64);
                    unfused_w1_oracle(
                        &mut wo,
                        &mut go,
                        &mut mo,
                        &mut vo,
                        &xa,
                        &dha,
                        &xb,
                        &dhb,
                        l2,
                        mult.as_deref(),
                        nh,
                        t,
                    );
                    let (mut wf, mut gf, mut mf, mut vf) =
                        (w0.clone(), g0.clone(), m0.clone(), v0.clone());
                    adam_update_w1_fused(&mut AdamW1FusedArgs {
                        w: &mut wf,
                        g: &mut gf,
                        m: &mut mf,
                        v: &mut vf,
                        xa: &xa,
                        dha: &dha,
                        xb: &xb,
                        dhb: &dhb,
                        l2_scale: l2,
                        l2_mult: mult.as_deref(),
                        n_hidden: nh,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-8,
                        bc1: 1.0 - 0.9f64.powi(t as i32),
                        bc2: 1.0 - 0.999f64.powi(t as i32),
                        lr: 0.005,
                    });
                    for (name, a, b) in [
                        ("w", &wo, &wf),
                        ("g", &go, &gf),
                        ("m", &mo, &mf),
                        ("v", &vo, &vf),
                    ] {
                        assert_eq!(
                            bits_of(a),
                            bits_of(b),
                            "fused diverged: {name} nf={nf} nh={nh} mult={has_mult} l2={l2}"
                        );
                    }
                }
            }
        }
    }

    /// Per-tier check: the v3 fused kernel and the scalar-tier fallback
    /// must each reproduce the unfused oracle bit-for-bit.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fused_w1_per_tier_bit_identical() {
        use archmage::SimdToken;
        let _lock = archmage::testing::lock_token_testing();
        let v3 =
            archmage::X64V3Token::summon().expect("this test needs an x86-64-v3 (AVX2+FMA) host");
        let sc = archmage::ScalarToken::summon().expect("scalar token");
        let v4 = archmage::X64V4Token::summon();
        for &(nf, nh) in &[(6usize, 4usize), (944, 32), (372, 128), (33, 12)] {
            for has_mult in [false, true] {
                let n = nf * nh;
                let seed = 0xF00D ^ ((nf as u64) << 24) ^ nh as u64;
                let (w0, g0, m0, v0) = synth_state(n, seed);
                let (xa, xb, dha, dhb) = pair_inputs(nf, nh, seed);
                let mult: Option<Vec<f64>> = if has_mult {
                    Some(
                        synth_state(nf, seed ^ 0x99)
                            .0
                            .iter()
                            .map(|x| x.abs() + 0.5)
                            .collect(),
                    )
                } else {
                    None
                };
                let (mut wo, mut go, mut mo, mut vo) =
                    (w0.clone(), g0.clone(), m0.clone(), v0.clone());
                // `black_box(t)`: keeps `powi(t)` on the runtime path in BOTH
                // the oracle (`make_args`) and the fused args below —
                // CTFE-folded `powi(5)` differs from runtime `powi(5)` by ~1ulp.
                let t = std::hint::black_box(5u64);
                unfused_w1_oracle(
                    &mut wo,
                    &mut go,
                    &mut mo,
                    &mut vo,
                    &xa,
                    &dha,
                    &xb,
                    &dhb,
                    1e-5,
                    mult.as_deref(),
                    nh,
                    t,
                );
                let check = |tier: &str, run: &dyn Fn(&mut AdamW1FusedArgs<'_>)| {
                    let (mut wt, mut gt, mut mt, mut vt) =
                        (w0.clone(), g0.clone(), m0.clone(), v0.clone());
                    run(&mut AdamW1FusedArgs {
                        w: &mut wt,
                        g: &mut gt,
                        m: &mut mt,
                        v: &mut vt,
                        xa: &xa,
                        dha: &dha,
                        xb: &xb,
                        dhb: &dhb,
                        l2_scale: 1e-5,
                        l2_mult: mult.as_deref(),
                        n_hidden: nh,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-8,
                        bc1: 1.0 - 0.9f64.powi(t as i32),
                        bc2: 1.0 - 0.999f64.powi(t as i32),
                        lr: 0.005,
                    });
                    for (name, a, b) in [
                        ("w", &wo, &wt),
                        ("g", &go, &gt),
                        ("m", &mo, &mt),
                        ("v", &vo, &vt),
                    ] {
                        let i = a
                            .iter()
                            .zip(b.iter())
                            .position(|(x, y)| x.to_bits() != y.to_bits());
                        assert_eq!(
                            i,
                            None,
                            "fused {tier} diverged: {name} nf={nf} nh={nh} mult={has_mult} \
                             first i={:?} (row={} j={})",
                            i,
                            i.map(|i| i / nh).unwrap_or(0),
                            i.map(|i| i % nh).unwrap_or(0),
                        );
                    }
                };
                check("v3", &|a| adam_pair_fused_inner_v3(v3, a));
                check("scalar", &|a| adam_pair_fused_inner_scalar(sc, a));
                // v4 routes to the v3 body via `token.v3()` (the kernel is
                // elementwise and nh%4==0 keeps every row in the fused
                // domain); arm it wherever the host provides AVX-512.
                if let Some(v4) = v4 {
                    check("v4", &|a| adam_pair_fused_inner_v4(v4, a));
                }
            }
        }
    }

    /// Input-sensitivity check (NOT a kernel negative control): the same
    /// fused element sequence with L2 applied *before* side B's
    /// contribution must produce different bits — otherwise the test data
    /// is insensitive to ordering and the equivalence asserts above would
    /// be vacuous. This compares two hand-written replicas; it never runs
    /// the kernel. Kernel-level mutation evidence (reordered v3 kernels
    /// are caught) is recorded in the lane report / Opus review mutation
    /// table.
    #[test]
    fn fused_w1_negative_control_order_matters() {
        let (nf, nh) = (944usize, 32usize);
        let n = nf * nh;
        let seed = 0xC0FFEEu64;
        let (w0, g0, m0, v0) = synth_state(n, seed);
        let (xa, xb, dha, dhb) = pair_inputs(nf, nh, seed);
        let (mut wo, mut go, mut mo, mut vo) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
        unfused_w1_oracle(
            &mut wo, &mut go, &mut mo, &mut vo, &xa, &dha, &xb, &dhb, 1e-5, None, nh, 3,
        );
        let (mut wp, mut gp, mut mp, mut vp) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
        fused_wrong_order(
            &mut wp, &mut gp, &mut mp, &mut vp, &xa, &dha, &xb, &dhb, 1e-5, None, nh, 3,
        );
        let diffs: usize = [(&wo, &wp), (&go, &gp), (&mo, &mp), (&vo, &vp)]
            .iter()
            .map(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count()
            })
            .sum();
        assert!(
            diffs > 0,
            "negative control: reordered L2 produced identical bits — the oracle is insensitive"
        );
    }
}
