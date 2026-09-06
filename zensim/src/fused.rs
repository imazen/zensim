//! Fused V-blur + feature extraction for streaming strips.
//!
//! Instead of 4 separate V-blur passes (writing to memory) followed by 7 reduction
//! passes (reading from memory), this module fuses everything into a single column-wise
//! pass. V-blurred values stay in registers and all features are computed inline.
//!
//! Memory pass reduction: ~40 passes → ~12 passes per channel (with fused H-blur).
#![allow(
    clippy::assign_op_pattern,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use crate::ssim_form::{ssim_dissim_raw_scalar, ssim_dissim8, ssim_dissim16};
#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
use archmage::magetypes;
use magetypes::simd::backends::F32x8Backend;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::backends::F32x16Backend;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
use magetypes::simd::generic::f32x8 as GenericF32x8;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::f32x16;

// ============================================================
// Free raw-moments accumulation — shared across every SIMD tier
// (`benchmarks/free_features_2026-09-01.md`, `benchmarks/profile_d_notax_2026-09-01.md`)
// ============================================================
//
// The four FREE raw moments (`Σs, Σd, Σs², Σd²` — `V1FreeExtras::RawMoments`,
// `StripChannelAccum::sum_s` etc.) are the same two-line accumulate + one
// conditional four-line finish at every call site, independent of lane
// width or which concrete SIMD backend produced the row's `s`/`d` values.
// Before this pair of helpers the sequence was hand-duplicated at 6 vector
// sites (`_v4`'s and `_v4x`'s native f32x16 main loops, `_v4`'s and `_v4x`'s
// f32x8 REMAINDER loops via `token.v3()`, `_v3`'s native f32x8 main loop,
// and the `#[magetypes(neon, wasm128, scalar)]`-generated function's f32x8
// main loop) plus 4 scalar-tail sites — the free-features doc's own count.
// Consolidating the two WIDTHS into one generic definition each removes the
// vector-site duplication (6 -> 2 source definitions, still 6 call sites)
// and is also the "no waste, no effort duplication" extension point a
// future `V1FreeExtras` variant (a "class C" slot — see the free-features
// doc §4) can reuse without re-deriving or re-copying this arithmetic a
// 7th/8th/9th time: add the new lane-accumulate step here once, call it
// from wherever the new class needs it.
//
// `#[inline(always)]`, not `#[rite]`: both helpers are GENERIC over a
// backend TRAIT (`T: F32x8Backend` / `T: F32x16Backend`), not a concrete
// token, so there is no single `#[target_feature]` string to attach at the
// definition site the way `#[rite]` needs for a concrete-token function —
// each monomorphized instantiation inherits its caller's already-established
// feature region purely through ordinary generic inlining, exactly like
// every other `T: F32x8Backend`-generic kernel in this codebase
// (`feature_v2.rs`'s `dense_block_kernel_generic`, `ssim_d_local_v`, etc.).
// `dense_block_kernel_generic`'s own doc comment already carries the
// MEASURED reason forcing the inline is mandatory here too: an un-inlined
// generic SIMD helper compiles to a call into a `core::arch` shim OUTSIDE
// the `#[target_feature]` region, measured as a 5.3x whole-extraction
// regression on that kernel. Verified inlined away on this refactor: `nm -C`
// on the compiled `ssim2_speed_bar` release binary shows the tier entry
// points present (`zensim::fused::__arcane_fused_vblur_ssim_inner_{v3,v4,v4x}`)
// and zero occurrences of `raw_moments_accumulate`/`raw_moments_finish`
// anywhere in the binary — no un-inlined call site survives.

/// Accumulate one row's contribution to the four free raw-moment lane sums.
/// Bit-identical to the hand-inlined `fm_s = fm_s + s; …` sequence it
/// replaces — same operations, same order, same intermediate rounding
/// (`s * s` before adding, not any fused/reassociated form).
#[inline(always)]
fn raw_moments_accumulate8<T: F32x8Backend + Copy>(
    fm_s: &mut GenericF32x8<T>,
    fm_d: &mut GenericF32x8<T>,
    fm_s2: &mut GenericF32x8<T>,
    fm_d2: &mut GenericF32x8<T>,
    s: GenericF32x8<T>,
    d: GenericF32x8<T>,
) {
    *fm_s = *fm_s + s;
    *fm_d = *fm_d + d;
    *fm_s2 = *fm_s2 + s * s;
    *fm_d2 = *fm_d2 + d * d;
}

/// Reduce the four lane accumulators to scalars and add them into `acc`'s
/// running f64 sums — the band's-last-inner-row finish step. Bit-identical
/// to the hand-inlined `acc.sum_s += fm_s.reduce_add() as f64; …` it
/// replaces.
#[inline(always)]
fn raw_moments_finish8<T: F32x8Backend + Copy>(
    acc: &mut StripChannelAccum,
    fm_s: GenericF32x8<T>,
    fm_d: GenericF32x8<T>,
    fm_s2: GenericF32x8<T>,
    fm_d2: GenericF32x8<T>,
) {
    acc.sum_s += fm_s.reduce_add() as f64;
    acc.sum_d += fm_d.reduce_add() as f64;
    acc.sum_s2 += fm_s2.reduce_add() as f64;
    acc.sum_d2 += fm_d2.reduce_add() as f64;
}

/// 16-lane sibling of [`raw_moments_accumulate8`] — `_v4`'s and `_v4x`'s
/// native f32x16 main loops (x86-64 only, hence the arch gate matching
/// `f32x16`'s own import).
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
fn raw_moments_accumulate16<T: F32x16Backend + Copy>(
    fm_s: &mut f32x16<T>,
    fm_d: &mut f32x16<T>,
    fm_s2: &mut f32x16<T>,
    fm_d2: &mut f32x16<T>,
    s: f32x16<T>,
    d: f32x16<T>,
) {
    *fm_s = *fm_s + s;
    *fm_d = *fm_d + d;
    *fm_s2 = *fm_s2 + s * s;
    *fm_d2 = *fm_d2 + d * d;
}

/// 16-lane sibling of [`raw_moments_finish8`].
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
fn raw_moments_finish16<T: F32x16Backend + Copy>(
    acc: &mut StripChannelAccum,
    fm_s: f32x16<T>,
    fm_d: f32x16<T>,
    fm_s2: f32x16<T>,
    fm_d2: f32x16<T>,
) {
    acc.sum_s += fm_s.reduce_add() as f64;
    acc.sum_d += fm_d.reduce_add() as f64;
    acc.sum_s2 += fm_s2.reduce_add() as f64;
    acc.sum_d2 += fm_d2.reduce_add() as f64;
}

/// Scalar sibling of [`raw_moments_accumulate8`] / [`raw_moments_accumulate16`]
/// — every tier's remainder-columns tail (width not a multiple of its
/// vector width) falls back to plain `f32`.
#[inline(always)]
fn raw_moments_accumulate_scalar(
    fm_s: &mut f32,
    fm_d: &mut f32,
    fm_s2: &mut f32,
    fm_d2: &mut f32,
    s: f32,
    d: f32,
) {
    *fm_s += s;
    *fm_d += d;
    *fm_s2 += s * s;
    *fm_d2 += d * d;
}

/// Scalar sibling of [`raw_moments_finish8`] / [`raw_moments_finish16`].
#[inline(always)]
fn raw_moments_finish_scalar(
    acc: &mut StripChannelAccum,
    fm_s: f32,
    fm_d: f32,
    fm_s2: f32,
    fm_d2: f32,
) {
    acc.sum_s += fm_s as f64;
    acc.sum_d += fm_d as f64;
    acc.sum_s2 += fm_s2 as f64;
    acc.sum_d2 += fm_d2 as f64;
}

// ============================================================
// Free BOUNDED-ERROR accumulation (class C) — shared across every SIMD tier
// (`benchmarks/free_features_classC_2026-09-04.md`)
// ============================================================
//
// The SECOND free tranche this kernel can carry, and the extension point the
// raw-moments note above was written for: `Σ mse_i` and the two luminance-
// binned weighted sums of the same `mse_i`, where
//
//     mse_i = sat((s − d)², C_MSE)          [`feature_v2::C_MSE`]
//     t     = sat(ref_Y,     C_LUM_T)       [`feature_v2::C_LUM_T`]
//
// are per-pixel functions of values ALREADY LIVE in this kernel's registers:
// `pd = s − d` is computed one line above for `acc.mse`, and for the Y
// channel the reference-luma plane IS this channel's `src` (the append
// kernel keeps `ref_y` a separate argument only so the weight is explicitly
// a function of the reference — `feature_v2::csfw_block_kernel_generic`'s
// own note). So no new plane, no new load, no new pass — the class-C
// definition.
//
// They finalize four families of 944 slots the v1-only walk otherwise leaves
// at their structural zeros:
//   * v2-348 `MSE`      (12 cells) — `clamp01(Σ mse_i / n)`
//   * append `LUM_DARK_ERR` / `LUM_MID_ERR` / `LUM_BRIGHT_ERR` (Y, 4 scales
//     = 12 slots) — the Bernstein-partition dark/bright weighted means, with
//     mid DERIVED at finalize exactly as `finish_append` derives it.
// No slot is renumbered and none is invented: these are existing 944-layout
// positions, filled by a cheaper route (the append-only feature-numbering
// discipline).
//
// Same `#[inline(always)]`-generic-over-a-backend-trait shape, and the same
// MEASURED reason for it, as the raw-moments pair above — see that comment;
// `#[rite]` cannot apply to a function generic over a backend TRAIT.

/// `feature_v2::C_MSE` as `f32`. Duplicated (not imported) because
/// `feature_v2` is behind the `feature-regime-v2` feature while this module
/// is unconditional; `feature_v2::tests::class_c_kernel_constants_match`
/// fails the build if the two ever disagree.
pub(crate) const C_MSE_F32: f32 = 0.01;
/// `feature_v2::C_LUM_T` as `f32`. Same duplication contract as
/// [`C_MSE_F32`].
pub(crate) const C_LUM_T_F32: f32 = 0.35;

/// Which FREE extra accumulators the fused v1 kernel carries this call.
///
/// Replaces the previous bare boolean `raw_moments` parameter: the class-C
/// tranche adds two more independently-requestable accumulator groups, and
/// one `Copy` struct keeps every kernel signature at ONE free-extras
/// parameter instead of three parallel bools. All-false is the pre-existing
/// instruction sequence, unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FreeExtrasWork {
    /// Σs, Σd, Σs², Σd² — the `V1FreeExtras::RawMoments` set.
    pub raw_moments: bool,
    /// Σ `mse_i` — the bounded per-pixel error. Every channel.
    pub bounded_err: bool,
    /// The dark/bright luminance-binned weighted sums of the SAME `mse_i`.
    /// Y channel only (`feature_v2::APPEND2_CHANNEL`), because that is the
    /// only channel whose `src` plane IS the reference luma the weight
    /// reads. Requires `bounded_err` (the weights multiply its `mse_i`).
    pub lum_bins: bool,
}

/// Accumulate one row's bounded per-pixel error into the lane sum and return
/// it, so the luminance bins can weight the SAME value rather than
/// recomputing it. `sat(x, c) = max(x,0)/(max(x,0)+c)` — the vector form of
/// `feature_v2::saturate`, written out here for the same
/// module-independence reason as [`C_MSE_F32`]. The `max(0)` is a no-op for
/// a real square and is kept only so a NaN input maps the same way the f64
/// definition maps it.
#[inline(always)]
fn bounded_err_accumulate8<T: F32x8Backend + Copy>(
    token: T,
    be_m: &mut GenericF32x8<T>,
    pd: GenericF32x8<T>,
) -> GenericF32x8<T> {
    let sq = (pd * pd).max(GenericF32x8::<T>::zero(token));
    let m = sq / (sq + GenericF32x8::<T>::splat(token, C_MSE_F32));
    *be_m = *be_m + m;
    m
}

/// Accumulate the dark/bright Bernstein-weighted numerator + denominator
/// lane sums for one row. `s` is the Y channel's source row (= the reference
/// luma); `m` is [`bounded_err_accumulate8`]'s per-pixel value.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn lum_bins_accumulate8<T: F32x8Backend + Copy>(
    token: T,
    wd_num: &mut GenericF32x8<T>,
    wd_den: &mut GenericF32x8<T>,
    wb_num: &mut GenericF32x8<T>,
    wb_den: &mut GenericF32x8<T>,
    s: GenericF32x8<T>,
    m: GenericF32x8<T>,
) {
    let ry = s.max(GenericF32x8::<T>::zero(token));
    let t = ry / (ry + GenericF32x8::<T>::splat(token, C_LUM_T_F32));
    let one_mt = GenericF32x8::<T>::splat(token, 1.0) - t;
    let wd = one_mt * one_mt;
    let wb = t * t;
    *wd_num = *wd_num + wd * m;
    *wd_den = *wd_den + wd;
    *wb_num = *wb_num + wb * m;
    *wb_den = *wb_den + wb;
}

/// Band's-last-inner-row finish for [`bounded_err_accumulate8`].
#[inline(always)]
fn bounded_err_finish8<T: F32x8Backend + Copy>(acc: &mut StripChannelAccum, be_m: GenericF32x8<T>) {
    acc.sum_msat += be_m.reduce_add() as f64;
}

/// Band's-last-inner-row finish for [`lum_bins_accumulate8`].
#[inline(always)]
fn lum_bins_finish8<T: F32x8Backend + Copy>(
    acc: &mut StripChannelAccum,
    wd_num: GenericF32x8<T>,
    wd_den: GenericF32x8<T>,
    wb_num: GenericF32x8<T>,
    wb_den: GenericF32x8<T>,
) {
    acc.lum_wd_num += wd_num.reduce_add() as f64;
    acc.lum_wd_den += wd_den.reduce_add() as f64;
    acc.lum_wb_num += wb_num.reduce_add() as f64;
    acc.lum_wb_den += wb_den.reduce_add() as f64;
}

/// 16-lane sibling of [`bounded_err_accumulate8`].
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
fn bounded_err_accumulate16<T: F32x16Backend + Copy>(
    token: T,
    be_m: &mut f32x16<T>,
    pd: f32x16<T>,
) -> f32x16<T> {
    let sq = (pd * pd).max(f32x16::<T>::zero(token));
    let m = sq / (sq + f32x16::<T>::splat(token, C_MSE_F32));
    *be_m = *be_m + m;
    m
}

/// 16-lane sibling of [`lum_bins_accumulate8`].
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn lum_bins_accumulate16<T: F32x16Backend + Copy>(
    token: T,
    wd_num: &mut f32x16<T>,
    wd_den: &mut f32x16<T>,
    wb_num: &mut f32x16<T>,
    wb_den: &mut f32x16<T>,
    s: f32x16<T>,
    m: f32x16<T>,
) {
    let ry = s.max(f32x16::<T>::zero(token));
    let t = ry / (ry + f32x16::<T>::splat(token, C_LUM_T_F32));
    let one_mt = f32x16::<T>::splat(token, 1.0) - t;
    let wd = one_mt * one_mt;
    let wb = t * t;
    *wd_num = *wd_num + wd * m;
    *wd_den = *wd_den + wd;
    *wb_num = *wb_num + wb * m;
    *wb_den = *wb_den + wb;
}

/// 16-lane sibling of [`bounded_err_finish8`].
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
fn bounded_err_finish16<T: F32x16Backend + Copy>(acc: &mut StripChannelAccum, be_m: f32x16<T>) {
    acc.sum_msat += be_m.reduce_add() as f64;
}

/// 16-lane sibling of [`lum_bins_finish8`].
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
fn lum_bins_finish16<T: F32x16Backend + Copy>(
    acc: &mut StripChannelAccum,
    wd_num: f32x16<T>,
    wd_den: f32x16<T>,
    wb_num: f32x16<T>,
    wb_den: f32x16<T>,
) {
    acc.lum_wd_num += wd_num.reduce_add() as f64;
    acc.lum_wd_den += wd_den.reduce_add() as f64;
    acc.lum_wb_num += wb_num.reduce_add() as f64;
    acc.lum_wb_den += wb_den.reduce_add() as f64;
}

/// Scalar sibling of [`bounded_err_accumulate8`] — the remainder-columns tail.
#[inline(always)]
fn bounded_err_accumulate_scalar(be_m: &mut f32, pd: f32) -> f32 {
    let sq = (pd * pd).max(0.0);
    let m = sq / (sq + C_MSE_F32);
    *be_m += m;
    m
}

/// Scalar sibling of [`lum_bins_accumulate8`].
#[inline(always)]
fn lum_bins_accumulate_scalar(
    wd_num: &mut f32,
    wd_den: &mut f32,
    wb_num: &mut f32,
    wb_den: &mut f32,
    s: f32,
    m: f32,
) {
    let ry = s.max(0.0);
    let t = ry / (ry + C_LUM_T_F32);
    let one_mt = 1.0 - t;
    let wd = one_mt * one_mt;
    let wb = t * t;
    *wd_num += wd * m;
    *wd_den += wd;
    *wb_num += wb * m;
    *wb_den += wb;
}

/// Scalar sibling of [`bounded_err_finish8`].
#[inline(always)]
fn bounded_err_finish_scalar(acc: &mut StripChannelAccum, be_m: f32) {
    acc.sum_msat += be_m as f64;
}

/// Scalar sibling of [`lum_bins_finish8`].
#[inline(always)]
fn lum_bins_finish_scalar(
    acc: &mut StripChannelAccum,
    wd_num: f32,
    wd_den: f32,
    wb_num: f32,
    wb_den: f32,
) {
    acc.lum_wd_num += wd_num as f64;
    acc.lum_wd_den += wd_den as f64;
    acc.lum_wb_num += wb_num as f64;
    acc.lum_wb_den += wb_den as f64;
}

/// Test-only handles on the class-C scalar integrands, so
/// `feature_v2::tests::class_c_integrands_match_the_f64_scalar_oracle` can
/// gate the arithmetic itself against the f64 `saturate` the append kernel
/// calls — rather than only observing it end-to-end through a whole walk.
/// The vector tiers are gated end-to-end instead (the geometry list in
/// `class_c_extras_match_the_944_walk` covers every tier's main loop AND
/// its 8-lane / scalar remainder).
#[cfg(test)]
pub(crate) fn test_only_bounded_err_scalar(be_m: &mut f32, pd: f32) -> f32 {
    bounded_err_accumulate_scalar(be_m, pd)
}

/// Test-only handle on [`lum_bins_accumulate_scalar`]. See
/// [`test_only_bounded_err_scalar`].
#[cfg(test)]
pub(crate) fn test_only_lum_bins_scalar(
    wd_num: &mut f32,
    wd_den: &mut f32,
    wb_num: &mut f32,
    wb_den: &mut f32,
    s: f32,
    m: f32,
) {
    lum_bins_accumulate_scalar(wd_num, wd_den, wb_num, wb_den, s, m)
}

/// Accumulated feature sums from a fused V-blur + feature extraction pass.
/// All values are raw sums (not yet divided by pixel count).
pub(crate) struct StripChannelAccum {
    pub ssim_d: f64,
    pub ssim_d4: f64,
    pub ssim_d2: f64,
    pub edge_art: f64,
    pub edge_art4: f64,
    pub edge_art2: f64,
    pub edge_det: f64,
    pub edge_det4: f64,
    pub edge_det2: f64,
    pub mse: f64,
    pub hf_sq_src: f64,
    pub hf_sq_dst: f64,
    pub hf_abs_src: f64,
    pub hf_abs_dst: f64,
    // Extended: L8 power pool and max
    pub ssim_d8: f64,
    pub edge_art8: f64,
    pub edge_det8: f64,
    pub ssim_max: f32,
    pub edge_art_max: f32,
    pub edge_det_max: f32,
    // --- Free raw moments (`raw_moments`; zero and untouched otherwise) ---
    // Σsrc, Σdst, Σsrc², Σdst² over the inner rows. Written ONLY when the
    // caller asks; every existing field's value and accumulation order is
    // unchanged whether it asks or not.
    pub sum_s: f64,
    pub sum_d: f64,
    pub sum_s2: f64,
    pub sum_d2: f64,
    // --- Free bounded error (`FreeExtrasWork::bounded_err` / `lum_bins`;
    //     zero and untouched otherwise) ---
    /// Σ `sat((s−d)², C_MSE)` — the v2-348 `MSE` slot's numerator.
    pub sum_msat: f64,
    /// Σ `(1−t)²·mse_i` and Σ `(1−t)²` — `LUM_DARK_ERR`'s weighted mean.
    pub lum_wd_num: f64,
    pub lum_wd_den: f64,
    /// Σ `t²·mse_i` and Σ `t²` — `LUM_BRIGHT_ERR`'s weighted mean.
    /// (`LUM_MID_ERR` is DERIVED from these plus `sum_msat` and `n`, exactly
    /// as `feature_v2::finish_append` derives it.)
    pub lum_wb_num: f64,
    pub lum_wb_den: f64,
}

impl StripChannelAccum {
    pub fn zero() -> Self {
        Self {
            ssim_d: 0.0,
            ssim_d4: 0.0,
            ssim_d2: 0.0,
            edge_art: 0.0,
            edge_art4: 0.0,
            edge_art2: 0.0,
            edge_det: 0.0,
            edge_det4: 0.0,
            edge_det2: 0.0,
            mse: 0.0,
            hf_sq_src: 0.0,
            hf_sq_dst: 0.0,
            hf_abs_src: 0.0,
            hf_abs_dst: 0.0,
            ssim_d8: 0.0,
            edge_art8: 0.0,
            edge_det8: 0.0,
            ssim_max: 0.0,
            edge_art_max: 0.0,
            edge_det_max: 0.0,
            sum_s: 0.0,
            sum_d: 0.0,
            sum_s2: 0.0,
            sum_d2: 0.0,
            sum_msat: 0.0,
            lum_wd_num: 0.0,
            lum_wd_den: 0.0,
            lum_wb_num: 0.0,
            lum_wb_den: 0.0,
        }
    }
}

/// Fused V-blur + ALL feature extraction for SSIM channels.
///
/// Reads 6 inputs: 4 H-blurred planes (h_mu1, h_mu2, h_sigma_sq, h_sigma12) + raw src + dst.
/// Maintains 4 V-blur running sums per column group.
/// At each inner row, computes SSIM, edge, variance, texture, and MSE features
/// directly from register values — V-blur outputs never touch memory.
///
/// Returns accumulated feature sums for the inner rows of this strip.
pub(crate) fn fused_vblur_features_ssim(
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    // `ssq_out` / `s12_out`: the V-blurred `h_sigma_sq` / `h_sigma12` for the
    // inner rows — the exact planes `box_blur_v_from_copy` would produce from
    // the same inputs, taken straight out of the registers instead of a second
    // sweep. Used by the v1-pool replay's masked/IW SSIM slots (which is why
    // BIT-identity, not closeness, is required: `folded720_v1_pools_match_v1_path`
    // compares the pool slots to v1's 372 with `to_bits()`).
    //
    // Why they are bit-identical: each column's V-blur is an INDEPENDENT scalar
    // recurrence (`sum += src[add] - src[rem]`, then `sum * (1.0 / diam)`), so
    // lane width cannot change a value and only the index sequence can. Init
    // (`mirror_idx`) and `rem_idx` are written the same way in both kernels;
    // the ONE textual difference is the bottom-edge `add_idx` fold — this
    // kernel's `vblur_add_idx` takes `|2·(h−1) − add_raw|`, `box_blur_v`'s
    // takes `saturating_sub`. They differ only when `2·(h−1) < y + r + 1` for
    // some `y < h`, i.e. only when `h < r + 2` (= 7 at BLUR_RADIUS 5). The
    // folded walk reflect-pads to a 64px floor and runs 4 pyramid scales, so
    // the smallest plane it ever V-blurs is 8 rows and every band buffer is at
    // least that tall — the divergent branch is unreachable on this path.
    // (Not a general guarantee: a caller that V-blurs a <7-row plane must keep
    // using `box_blur_v_from_copy` if it needs v1 parity.)
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    // `free`: which FREE extra accumulators to carry alongside the existing
    // sums — the four raw moments (Σs, Σd, Σs², Σd²) and/or the class-C
    // bounded-error family. See [`FreeExtrasWork`] and
    // `StripChannelAccum::sum_s` / `sum_msat`.
    free: FreeExtrasWork,
) -> StripChannelAccum {
    incant!(
        fused_vblur_ssim_inner(
            h_mu1,
            h_mu2,
            h_sigma_sq,
            h_sigma12,
            src,
            dst,
            width,
            height,
            inner_start,
            inner_h,
            radius,
            mu1_out,
            mu2_out,
            store_mu,
            sd_out,
            store_sd,
            ssq_out,
            s12_out,
            store_sigma,
            free
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// Fused V-blur + feature extraction for edge-only channels (no SSIM).
///
/// Reads 4 inputs: 2 H-blurred planes (h_mu1, h_mu2) + raw src + dst.
/// Maintains 2 V-blur running sums per column group.
/// Computes edge, variance, texture, and MSE features inline.
pub(crate) fn fused_vblur_features_edge(
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    incant!(
        fused_vblur_edge_inner(
            h_mu1,
            h_mu2,
            src,
            dst,
            width,
            height,
            inner_start,
            inner_h,
            radius,
            mu1_out,
            mu2_out,
            store_mu
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

// ============================================================
// Helper: mirror-reflect row index for V-blur boundary handling
// ============================================================

#[inline(always)]
fn mirror_idx(i: usize, r: usize, height: usize) -> usize {
    if i <= r {
        (r - i).min(height - 1)
    } else {
        (i - r).min(height - 1)
    }
}

#[inline(always)]
fn vblur_add_idx(y: usize, r: usize, height: usize) -> usize {
    let add_raw = y + r + 1;
    if add_raw < height {
        add_raw
    } else {
        // Mirror-reflect: fold back from the boundary.
        // Use signed math to avoid underflow when add_raw >> height.
        let reflected = 2 * (height as isize - 1) - add_raw as isize;
        reflected.unsigned_abs().min(height - 1)
    }
}

#[inline(always)]
fn vblur_rem_idx(y: usize, r: usize, height: usize) -> usize {
    let rem_i = y as isize - r as isize;
    let idx = if rem_i < 0 {
        rem_i.unsigned_abs()
    } else {
        rem_i as usize
    };
    idx.min(height - 1)
}

// ============================================================
// AVX-512 implementations
// ============================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_ssim_inner_v4(
    token: archmage::X64V4Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    // `free`: which FREE extra accumulators to carry alongside the existing
    // sums — the four raw moments (Σs, Σd, Σs², Σd²) and/or the class-C
    // bounded-error family. See [`FreeExtrasWork`] and
    // `StripChannelAccum::sum_s` / `sum_msat`.
    free: FreeExtrasWork,
) -> StripChannelAccum {
    let form = crate::ssim_form::active_luma_form();
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    // SSIM constants
    let one = f32x16::splat(token, 1.0);
    let zero = f32x16::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 16;

        // Initialize 4 running sums for this column group
        let mut sum_m1 = f32x16::zero(token);
        let mut sum_m2 = f32x16::zero(token);
        let mut sum_sq = f32x16::zero(token);
        let mut sum_s12 = f32x16::zero(token);
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x16::zero(token),
            f32x16::zero(token),
            f32x16::zero(token),
            f32x16::zero(token),
        );
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x16::zero(token),
            f32x16::zero(token),
            f32x16::zero(token),
            f32x16::zero(token),
            f32x16::zero(token),
        );

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x16::from_array(token, h_mu1[base..][..16].try_into().unwrap());
            sum_m2 = sum_m2 + f32x16::from_array(token, h_mu2[base..][..16].try_into().unwrap());
            sum_sq =
                sum_sq + f32x16::from_array(token, h_sigma_sq[base..][..16].try_into().unwrap());
            sum_s12 =
                sum_s12 + f32x16::from_array(token, h_sigma12[base..][..16].try_into().unwrap());
        }

        for y in 0..height {
            // Only accumulate features for inner rows
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;

                // V-blurred values (still in registers)
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let ssq = sum_sq * inv_v;
                let s12 = sum_s12 * inv_v;

                // Load raw pixel values
                let s = f32x16::from_array(token, src[base..][..16].try_into().unwrap());
                let d = f32x16::from_array(token, dst[base..][..16].try_into().unwrap());

                // === SSIM ===
                let sd = (ssim_dissim16(token, form, mu1, mu2, ssq, s12)).max(zero);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                if store_sd {
                    sd_out[base..base + 16].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 16].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 16].copy_from_slice(&mu2.to_array());
                }
                if store_sigma {
                    ssq_out[base..base + 16].copy_from_slice(&ssq.to_array());
                    s12_out[base..base + 16].copy_from_slice(&s12.to_array());
                }

                // === Edge ===
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

                // === HF energy (L2): (pixel - mu)² ===
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // === HF magnitude (L1): |pixel - mu| ===
                // diff1/diff2 already computed above
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // === MSE: (src - dst)² ===
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate16(&mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, s, d);
                    if y + 1 == inner_end {
                        raw_moments_finish16(&mut acc, fm_s, fm_d, fm_s2, fm_d2);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate16(token, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate16(
                            token,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish16(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish16(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            // Slide V-blur window
            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;

            sum_m1 = sum_m1
                + f32x16::from_array(token, h_mu1[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu1[rem_base..][..16].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x16::from_array(token, h_mu2[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu2[rem_base..][..16].try_into().unwrap());
            sum_sq = sum_sq
                + f32x16::from_array(token, h_sigma_sq[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_sigma_sq[rem_base..][..16].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x16::from_array(token, h_sigma12[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_sigma12[rem_base..][..16].try_into().unwrap());
        }
    }

    // Remainder columns with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    let one8 = f32x8::splat(v3, 1.0);
    let zero8 = f32x8::zero(v3);

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum_m1 = f32x8::zero(v3);
        let mut sum_m2 = f32x8::zero(v3);
        let mut sum_sq = f32x8::zero(v3);
        let mut sum_s12 = f32x8::zero(v3);
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x8::zero(v3),
            f32x8::zero(v3),
            f32x8::zero(v3),
            f32x8::zero(v3),
        );
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x8::zero(v3),
            f32x8::zero(v3),
            f32x8::zero(v3),
            f32x8::zero(v3),
            f32x8::zero(v3),
        );

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[base..][..8].try_into().unwrap());
            sum_sq = sum_sq + f32x8::from_array(v3, h_sigma_sq[base..][..8].try_into().unwrap());
            sum_s12 = sum_s12 + f32x8::from_array(v3, h_sigma12[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v8;
                let mu2 = sum_m2 * inv_v8;
                let ssq = sum_sq * inv_v8;
                let s12 = sum_s12 * inv_v8;
                let s = f32x8::from_array(v3, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(v3, dst[base..][..8].try_into().unwrap());

                // SSIM
                let sd = (ssim_dissim8(v3, form, mu1, mu2, ssq, s12)).max(zero8);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                if store_sd {
                    sd_out[base..base + 8].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }
                if store_sigma {
                    ssq_out[base..base + 8].copy_from_slice(&ssq.to_array());
                    s12_out[base..base + 8].copy_from_slice(&s12.to_array());
                }

                // Edge
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one8 + diff2) / (one8 + diff1) - one8;
                let artifact = ed.max(zero8);
                let detail_lost = (-ed).max(zero8);
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

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // Texture
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate8(&mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, s, d);
                    if y + 1 == inner_end {
                        raw_moments_finish8(&mut acc, fm_s, fm_d, fm_s2, fm_d2);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate8(v3, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate8(
                            v3,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish8(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish8(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu2[rem_base..][..8].try_into().unwrap());
            sum_sq = sum_sq
                + f32x8::from_array(v3, h_sigma_sq[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_sigma_sq[rem_base..][..8].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x8::from_array(v3, h_sigma12[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_sigma12[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_s12 = 0.0f32;
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
            sum_sq += h_sigma_sq[idx * width + x];
            sum_s12 += h_sigma12[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let ssq = sum_sq * inv;
                let s12 = sum_s12 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                // SSIM (f32 to match SIMD paths)
                let sd = (ssim_dissim_raw_scalar(form, mu1, mu2, ssq, s12)).max(0.0f32);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd as f64;
                acc.ssim_d4 += sd4 as f64;
                acc.ssim_d2 += sd2 as f64;
                acc.ssim_d8 += (sd4 * sd4) as f64;
                acc.ssim_max = acc.ssim_max.max(sd);
                if store_sd {
                    sd_out[y * width + x] = sd;
                }
                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }
                if store_sigma {
                    ssq_out[y * width + x] = ssq;
                    s12_out[y * width + x] = s12;
                }

                // Edge (f32 to match SIMD paths)
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
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

                // Variance
                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;

                // Texture
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                // MSE
                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;

                // === Free raw moments (`raw_moments`) — scalar tail ===
                if free.raw_moments {
                    raw_moments_accumulate_scalar(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, sv, dv,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish_scalar(&mut acc, fm_s, fm_d, fm_s2, fm_d2);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate_scalar(&mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate_scalar(
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            sv,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish_scalar(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish_scalar(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
            sum_sq = sum_sq + h_sigma_sq[add_idx * width + x] - h_sigma_sq[rem_idx * width + x];
            sum_s12 = sum_s12 + h_sigma12[add_idx * width + x] - h_sigma12[rem_idx * width + x];
        }
    }

    acc
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_ssim_inner_v4x(
    token: archmage::X64V4xToken,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    // `free`: which FREE extra accumulators to carry alongside the existing
    // sums — the four raw moments (Σs, Σd, Σs², Σd²) and/or the class-C
    // bounded-error family. See [`FreeExtrasWork`] and
    // `StripChannelAccum::sum_s` / `sum_msat`.
    free: FreeExtrasWork,
) -> StripChannelAccum {
    let form = crate::ssim_form::active_luma_form();
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    // SSIM constants
    let one = f32x16::splat(token, 1.0);
    let zero = f32x16::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 16;

        // Initialize 4 running sums for this column group
        let mut sum_m1 = f32x16::zero(token);
        let mut sum_m2 = f32x16::zero(token);
        let mut sum_sq = f32x16::zero(token);
        let mut sum_s12 = f32x16::zero(token);
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x16::zero(token),
            f32x16::zero(token),
            f32x16::zero(token),
            f32x16::zero(token),
        );
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x16::zero(token),
            f32x16::zero(token),
            f32x16::zero(token),
            f32x16::zero(token),
            f32x16::zero(token),
        );

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x16::from_array(token, h_mu1[base..][..16].try_into().unwrap());
            sum_m2 = sum_m2 + f32x16::from_array(token, h_mu2[base..][..16].try_into().unwrap());
            sum_sq =
                sum_sq + f32x16::from_array(token, h_sigma_sq[base..][..16].try_into().unwrap());
            sum_s12 =
                sum_s12 + f32x16::from_array(token, h_sigma12[base..][..16].try_into().unwrap());
        }

        for y in 0..height {
            // Only accumulate features for inner rows
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;

                // V-blurred values (still in registers)
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let ssq = sum_sq * inv_v;
                let s12 = sum_s12 * inv_v;

                // Load raw pixel values
                let s = f32x16::from_array(token, src[base..][..16].try_into().unwrap());
                let d = f32x16::from_array(token, dst[base..][..16].try_into().unwrap());

                // === SSIM ===
                let sd = (ssim_dissim16(token, form, mu1, mu2, ssq, s12)).max(zero);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                if store_sd {
                    sd_out[base..base + 16].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 16].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 16].copy_from_slice(&mu2.to_array());
                }
                if store_sigma {
                    ssq_out[base..base + 16].copy_from_slice(&ssq.to_array());
                    s12_out[base..base + 16].copy_from_slice(&s12.to_array());
                }

                // === Edge ===
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

                // === HF energy (L2): (pixel - mu)² ===
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // === HF magnitude (L1): |pixel - mu| ===
                // diff1/diff2 already computed above
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // === MSE: (src - dst)² ===
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate16(&mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, s, d);
                    if y + 1 == inner_end {
                        raw_moments_finish16(&mut acc, fm_s, fm_d, fm_s2, fm_d2);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate16(token, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate16(
                            token,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish16(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish16(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            // Slide V-blur window
            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;

            sum_m1 = sum_m1
                + f32x16::from_array(token, h_mu1[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu1[rem_base..][..16].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x16::from_array(token, h_mu2[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu2[rem_base..][..16].try_into().unwrap());
            sum_sq = sum_sq
                + f32x16::from_array(token, h_sigma_sq[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_sigma_sq[rem_base..][..16].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x16::from_array(token, h_sigma12[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_sigma12[rem_base..][..16].try_into().unwrap());
        }
    }

    // Remainder columns with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    let one8 = f32x8::splat(v3, 1.0);
    let zero8 = f32x8::zero(v3);

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum_m1 = f32x8::zero(v3);
        let mut sum_m2 = f32x8::zero(v3);
        let mut sum_sq = f32x8::zero(v3);
        let mut sum_s12 = f32x8::zero(v3);
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x8::zero(v3),
            f32x8::zero(v3),
            f32x8::zero(v3),
            f32x8::zero(v3),
        );
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x8::zero(v3),
            f32x8::zero(v3),
            f32x8::zero(v3),
            f32x8::zero(v3),
            f32x8::zero(v3),
        );

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[base..][..8].try_into().unwrap());
            sum_sq = sum_sq + f32x8::from_array(v3, h_sigma_sq[base..][..8].try_into().unwrap());
            sum_s12 = sum_s12 + f32x8::from_array(v3, h_sigma12[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v8;
                let mu2 = sum_m2 * inv_v8;
                let ssq = sum_sq * inv_v8;
                let s12 = sum_s12 * inv_v8;
                let s = f32x8::from_array(v3, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(v3, dst[base..][..8].try_into().unwrap());

                // SSIM
                let sd = (ssim_dissim8(v3, form, mu1, mu2, ssq, s12)).max(zero8);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                if store_sd {
                    sd_out[base..base + 8].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }
                if store_sigma {
                    ssq_out[base..base + 8].copy_from_slice(&ssq.to_array());
                    s12_out[base..base + 8].copy_from_slice(&s12.to_array());
                }

                // Edge
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one8 + diff2) / (one8 + diff1) - one8;
                let artifact = ed.max(zero8);
                let detail_lost = (-ed).max(zero8);
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

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // Texture
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate8(&mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, s, d);
                    if y + 1 == inner_end {
                        raw_moments_finish8(&mut acc, fm_s, fm_d, fm_s2, fm_d2);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate8(v3, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate8(
                            v3,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish8(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish8(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu2[rem_base..][..8].try_into().unwrap());
            sum_sq = sum_sq
                + f32x8::from_array(v3, h_sigma_sq[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_sigma_sq[rem_base..][..8].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x8::from_array(v3, h_sigma12[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_sigma12[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_s12 = 0.0f32;
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
            sum_sq += h_sigma_sq[idx * width + x];
            sum_s12 += h_sigma12[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let ssq = sum_sq * inv;
                let s12 = sum_s12 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                // SSIM (f32 to match SIMD paths)
                let sd = (ssim_dissim_raw_scalar(form, mu1, mu2, ssq, s12)).max(0.0f32);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd as f64;
                acc.ssim_d4 += sd4 as f64;
                acc.ssim_d2 += sd2 as f64;
                acc.ssim_d8 += (sd4 * sd4) as f64;
                acc.ssim_max = acc.ssim_max.max(sd);
                if store_sd {
                    sd_out[y * width + x] = sd;
                }
                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }
                if store_sigma {
                    ssq_out[y * width + x] = ssq;
                    s12_out[y * width + x] = s12;
                }

                // Edge (f32 to match SIMD paths)
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
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

                // Variance
                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;

                // Texture
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                // MSE
                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;

                // === Free raw moments (`raw_moments`) — scalar tail ===
                if free.raw_moments {
                    raw_moments_accumulate_scalar(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, sv, dv,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish_scalar(&mut acc, fm_s, fm_d, fm_s2, fm_d2);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate_scalar(&mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate_scalar(
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            sv,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish_scalar(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish_scalar(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
            sum_sq = sum_sq + h_sigma_sq[add_idx * width + x] - h_sigma_sq[rem_idx * width + x];
            sum_s12 = sum_s12 + h_sigma12[add_idx * width + x] - h_sigma12[rem_idx * width + x];
        }
    }

    acc
}

// ============================================================
// AVX2 implementations
// ============================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_ssim_inner_v3(
    token: archmage::X64V3Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    // `free`: which FREE extra accumulators to carry alongside the existing
    // sums — the four raw moments (Σs, Σd, Σs², Σd²) and/or the class-C
    // bounded-error family. See [`FreeExtrasWork`] and
    // `StripChannelAccum::sum_s` / `sum_msat`.
    free: FreeExtrasWork,
) -> StripChannelAccum {
    let form = crate::ssim_form::active_luma_form();
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 8;
        let mut sum_m1 = f32x8::zero(token);
        let mut sum_m2 = f32x8::zero(token);
        let mut sum_sq = f32x8::zero(token);
        let mut sum_s12 = f32x8::zero(token);
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x8::zero(token),
            f32x8::zero(token),
            f32x8::zero(token),
            f32x8::zero(token),
        );
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x8::zero(token),
            f32x8::zero(token),
            f32x8::zero(token),
            f32x8::zero(token),
            f32x8::zero(token),
        );

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(token, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(token, h_mu2[base..][..8].try_into().unwrap());
            sum_sq = sum_sq + f32x8::from_array(token, h_sigma_sq[base..][..8].try_into().unwrap());
            sum_s12 =
                sum_s12 + f32x8::from_array(token, h_sigma12[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let ssq = sum_sq * inv_v;
                let s12 = sum_s12 * inv_v;
                let s = f32x8::from_array(token, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(token, dst[base..][..8].try_into().unwrap());

                // SSIM
                let sd = (ssim_dissim8(token, form, mu1, mu2, ssq, s12)).max(zero);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                if store_sd {
                    sd_out[base..base + 8].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }
                if store_sigma {
                    ssq_out[base..base + 8].copy_from_slice(&ssq.to_array());
                    s12_out[base..base + 8].copy_from_slice(&s12.to_array());
                }

                // Edge
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

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // Texture
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate8(&mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, s, d);
                    if y + 1 == inner_end {
                        raw_moments_finish8(&mut acc, fm_s, fm_d, fm_s2, fm_d2);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate8(token, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate8(
                            token,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish8(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish8(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(token, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(token, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu2[rem_base..][..8].try_into().unwrap());
            sum_sq = sum_sq
                + f32x8::from_array(token, h_sigma_sq[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_sigma_sq[rem_base..][..8].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x8::from_array(token, h_sigma12[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_sigma12[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_s12 = 0.0f32;
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
            sum_sq += h_sigma_sq[idx * width + x];
            sum_s12 += h_sigma12[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let ssq = sum_sq * inv;
                let s12 = sum_s12 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                // SSIM
                let sd = (ssim_dissim_raw_scalar(form, mu1, mu2, ssq, s12)).max(0.0f32);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd as f64;
                acc.ssim_d4 += sd4 as f64;
                acc.ssim_d2 += sd2 as f64;
                acc.ssim_d8 += (sd4 * sd4) as f64;
                acc.ssim_max = acc.ssim_max.max(sd);
                if store_sd {
                    sd_out[y * width + x] = sd;
                }
                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }
                if store_sigma {
                    ssq_out[y * width + x] = ssq;
                    s12_out[y * width + x] = s12;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
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

                // Variance
                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;

                // Texture
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                // MSE
                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;

                // === Free raw moments (`raw_moments`) — scalar tail ===
                if free.raw_moments {
                    raw_moments_accumulate_scalar(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, sv, dv,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish_scalar(&mut acc, fm_s, fm_d, fm_s2, fm_d2);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate_scalar(&mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate_scalar(
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            sv,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish_scalar(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish_scalar(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
            sum_sq = sum_sq + h_sigma_sq[add_idx * width + x] - h_sigma_sq[rem_idx * width + x];
            sum_s12 = sum_s12 + h_sigma12[add_idx * width + x] - h_sigma12[rem_idx * width + x];
        }
    }

    acc
}

// ============================================================
// Generic WASM128 + Scalar SSIM implementation (via magetypes)
// ============================================================

#[magetypes(neon, wasm128, scalar)]
fn fused_vblur_ssim_inner(
    token: Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    // `free`: which FREE extra accumulators to carry alongside the existing
    // sums — the four raw moments (Σs, Σd, Σs², Σd²) and/or the class-C
    // bounded-error family. See [`FreeExtrasWork`] and
    // `StripChannelAccum::sum_s` / `sum_msat`.
    free: FreeExtrasWork,
) -> StripChannelAccum {
    let form = crate::ssim_form::active_luma_form();
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 8;
        let mut sum_m1_a = [0.0f32; 8];
        let mut sum_m2_a = [0.0f32; 8];
        let mut sum_sq_a = [0.0f32; 8];
        let mut sum_s12_a = [0.0f32; 8];
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x8::zero(token),
            f32x8::zero(token),
            f32x8::zero(token),
            f32x8::zero(token),
        );
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x8::zero(token),
            f32x8::zero(token),
            f32x8::zero(token),
            f32x8::zero(token),
            f32x8::zero(token),
        );

        // Initialize running sums
        {
            let mut sm1 = f32x8::zero(token);
            let mut sm2 = f32x8::zero(token);
            let mut ssq = f32x8::zero(token);
            let mut ss12 = f32x8::zero(token);
            for i in 0..diam {
                let idx = mirror_idx(i, r, height);
                let base = idx * width + col_base;
                sm1 = sm1 + f32x8::from_array(token, h_mu1[base..][..8].try_into().unwrap());
                sm2 = sm2 + f32x8::from_array(token, h_mu2[base..][..8].try_into().unwrap());
                ssq = ssq + f32x8::from_array(token, h_sigma_sq[base..][..8].try_into().unwrap());
                ss12 = ss12 + f32x8::from_array(token, h_sigma12[base..][..8].try_into().unwrap());
            }
            sm1.store(&mut sum_m1_a);
            sm2.store(&mut sum_m2_a);
            ssq.store(&mut sum_sq_a);
            ss12.store(&mut sum_s12_a);
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let sum_m1 = f32x8::from_array(token, sum_m1_a);
                let sum_m2 = f32x8::from_array(token, sum_m2_a);
                let sum_sq = f32x8::from_array(token, sum_sq_a);
                let sum_s12 = f32x8::from_array(token, sum_s12_a);

                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let ssq = sum_sq * inv_v;
                let s12 = sum_s12 * inv_v;
                let s = f32x8::from_array(token, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(token, dst[base..][..8].try_into().unwrap());

                // SSIM
                let sd = (ssim_dissim8(token, form, mu1, mu2, ssq, s12)).max(zero);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                if store_sd {
                    sd.store((&mut sd_out[base..base + 8]).try_into().unwrap());
                }
                if store_mu {
                    mu1.store((&mut mu1_out[base..base + 8]).try_into().unwrap());
                    mu2.store((&mut mu2_out[base..base + 8]).try_into().unwrap());
                }
                if store_sigma {
                    ssq.store((&mut ssq_out[base..base + 8]).try_into().unwrap());
                    s12.store((&mut s12_out[base..base + 8]).try_into().unwrap());
                }

                // Edge
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

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // Texture
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate8(&mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, s, d);
                    if y + 1 == inner_end {
                        raw_moments_finish8(&mut acc, fm_s, fm_d, fm_s2, fm_d2);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate8(token, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate8(
                            token,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish8(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish8(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            // Slide V-blur window
            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let new_m1 = f32x8::from_array(token, sum_m1_a)
                + f32x8::from_array(token, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu1[rem_base..][..8].try_into().unwrap());
            let new_m2 = f32x8::from_array(token, sum_m2_a)
                + f32x8::from_array(token, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu2[rem_base..][..8].try_into().unwrap());
            let new_sq = f32x8::from_array(token, sum_sq_a)
                + f32x8::from_array(token, h_sigma_sq[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_sigma_sq[rem_base..][..8].try_into().unwrap());
            let new_s12 = f32x8::from_array(token, sum_s12_a)
                + f32x8::from_array(token, h_sigma12[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_sigma12[rem_base..][..8].try_into().unwrap());
            new_m1.store(&mut sum_m1_a);
            new_m2.store(&mut sum_m2_a);
            new_sq.store(&mut sum_sq_a);
            new_s12.store(&mut sum_s12_a);
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_s12 = 0.0f32;
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
            sum_sq += h_sigma_sq[idx * width + x];
            sum_s12 += h_sigma12[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let ssq = sum_sq * inv;
                let s12 = sum_s12 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                // SSIM
                let sd = (ssim_dissim_raw_scalar(form, mu1, mu2, ssq, s12)).max(0.0f32);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd as f64;
                acc.ssim_d4 += sd4 as f64;
                acc.ssim_d2 += sd2 as f64;
                acc.ssim_d8 += (sd4 * sd4) as f64;
                acc.ssim_max = acc.ssim_max.max(sd);
                if store_sd {
                    sd_out[y * width + x] = sd;
                }
                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }
                if store_sigma {
                    ssq_out[y * width + x] = ssq;
                    s12_out[y * width + x] = s12;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
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

                // Variance
                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;

                // Texture
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                // MSE
                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;

                // === Free raw moments (`raw_moments`) — scalar tail ===
                if free.raw_moments {
                    raw_moments_accumulate_scalar(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, sv, dv,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish_scalar(&mut acc, fm_s, fm_d, fm_s2, fm_d2);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate_scalar(&mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate_scalar(
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            sv,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish_scalar(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish_scalar(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
            sum_sq = sum_sq + h_sigma_sq[add_idx * width + x] - h_sigma_sq[rem_idx * width + x];
            sum_s12 = sum_s12 + h_sigma12[add_idx * width + x] - h_sigma12[rem_idx * width + x];
        }
    }

    acc
}

// ============================================================
// Edge-only fused V-blur (no SSIM, only 2 running sums)
// ============================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_edge_inner_v4(
    token: archmage::X64V4Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    let one = f32x16::splat(token, 1.0);
    let zero = f32x16::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 16;
        let mut sum_m1 = f32x16::zero(token);
        let mut sum_m2 = f32x16::zero(token);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x16::from_array(token, h_mu1[base..][..16].try_into().unwrap());
            sum_m2 = sum_m2 + f32x16::from_array(token, h_mu2[base..][..16].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let s = f32x16::from_array(token, src[base..][..16].try_into().unwrap());
                let d = f32x16::from_array(token, dst[base..][..16].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 16].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 16].copy_from_slice(&mu2.to_array());
                }

                // Edge
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

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // Texture
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1
                + f32x16::from_array(token, h_mu1[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu1[rem_base..][..16].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x16::from_array(token, h_mu2[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu2[rem_base..][..16].try_into().unwrap());
        }
    }

    // Remainder with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    let one8 = f32x8::splat(v3, 1.0);
    let zero8 = f32x8::zero(v3);

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum_m1 = f32x8::zero(v3);
        let mut sum_m2 = f32x8::zero(v3);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v8;
                let mu2 = sum_m2 * inv_v8;
                let s = f32x8::from_array(v3, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(v3, dst[base..][..8].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }

                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one8 + diff2) / (one8 + diff1) - one8;
                let artifact = ed.max(zero8);
                let detail_lost = (-ed).max(zero8);
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
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu2[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
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

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
        }
    }

    acc
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_edge_inner_v4x(
    token: archmage::X64V4xToken,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    let one = f32x16::splat(token, 1.0);
    let zero = f32x16::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 16;
        let mut sum_m1 = f32x16::zero(token);
        let mut sum_m2 = f32x16::zero(token);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x16::from_array(token, h_mu1[base..][..16].try_into().unwrap());
            sum_m2 = sum_m2 + f32x16::from_array(token, h_mu2[base..][..16].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let s = f32x16::from_array(token, src[base..][..16].try_into().unwrap());
                let d = f32x16::from_array(token, dst[base..][..16].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 16].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 16].copy_from_slice(&mu2.to_array());
                }

                // Edge
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

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // Texture
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1
                + f32x16::from_array(token, h_mu1[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu1[rem_base..][..16].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x16::from_array(token, h_mu2[add_base..][..16].try_into().unwrap())
                - f32x16::from_array(token, h_mu2[rem_base..][..16].try_into().unwrap());
        }
    }

    // Remainder with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    let one8 = f32x8::splat(v3, 1.0);
    let zero8 = f32x8::zero(v3);

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum_m1 = f32x8::zero(v3);
        let mut sum_m2 = f32x8::zero(v3);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v8;
                let mu2 = sum_m2 * inv_v8;
                let s = f32x8::from_array(v3, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(v3, dst[base..][..8].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }

                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one8 + diff2) / (one8 + diff1) - one8;
                let artifact = ed.max(zero8);
                let detail_lost = (-ed).max(zero8);
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
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(v3, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(v3, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, h_mu2[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
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

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
        }
    }

    acc
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_edge_inner_v3(
    token: archmage::X64V3Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 8;
        let mut sum_m1 = f32x8::zero(token);
        let mut sum_m2 = f32x8::zero(token);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(token, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(token, h_mu2[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let s = f32x8::from_array(token, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(token, dst[base..][..8].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }

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
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array(token, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array(token, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu2[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
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

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
        }
    }

    acc
}

#[magetypes(neon, wasm128, scalar)]
fn fused_vblur_edge_inner(
    token: Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 8;
        let mut sum_m1_a = [0.0f32; 8];
        let mut sum_m2_a = [0.0f32; 8];

        // Initialize running sums
        {
            let mut sm1 = f32x8::zero(token);
            let mut sm2 = f32x8::zero(token);
            for i in 0..diam {
                let idx = mirror_idx(i, r, height);
                let base = idx * width + col_base;
                sm1 = sm1 + f32x8::from_array(token, h_mu1[base..][..8].try_into().unwrap());
                sm2 = sm2 + f32x8::from_array(token, h_mu2[base..][..8].try_into().unwrap());
            }
            sm1.store(&mut sum_m1_a);
            sm2.store(&mut sum_m2_a);
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let sum_m1 = f32x8::from_array(token, sum_m1_a);
                let sum_m2 = f32x8::from_array(token, sum_m2_a);

                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let s = f32x8::from_array(token, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array(token, dst[base..][..8].try_into().unwrap());

                if store_mu {
                    mu1.store((&mut mu1_out[base..base + 8]).try_into().unwrap());
                    mu2.store((&mut mu2_out[base..base + 8]).try_into().unwrap());
                }

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
            }

            // Slide V-blur window
            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let new_m1 = f32x8::from_array(token, sum_m1_a)
                + f32x8::from_array(token, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu1[rem_base..][..8].try_into().unwrap());
            let new_m2 = f32x8::from_array(token, sum_m2_a)
                + f32x8::from_array(token, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(token, h_mu2[rem_base..][..8].try_into().unwrap());
            new_m1.store(&mut sum_m1_a);
            new_m2.store(&mut sum_m2_a);
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
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

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
        }
    }

    acc
}
