// ============================================================================
// Phase-4 (§A.14): magetypes-SIMD formula pass — f32x8-lane vectorized
// counterparts of the four scalar formulas above. Same algebra, f32
// precision (matches the source planes' own f32 storage), 8 pixels/call
// instead of 1. Generic over any token implementing `F32x8Backend` so
// `#[magetypes(v4x, v4, v3, neon, wasm128)]` (see
// `compute_channel_scale_v2_kernel_entry` below) monomorphizes one body
// per tier — this is the SAME pattern as `~/work/archmage/docs/site/
// content/magetypes/examples/plane-ops.md`/`gaussian-blur.md`.
// ============================================================================

type V8<T> = GenericF32x8<T>;

/// The element ops the vector term helpers use, at either vector width.
/// magetypes' 8- and 16-lane generic types implement each with the same
/// per-lane semantics (the same `max` operand order, `GT_OQ`/`LT_OQ`
/// compares, mask select; v3's 16-lane type is two 8-lane halves), so a
/// helper computes the same bits per lane at either width.
trait Vf32:
    Copy
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
{
    type Tok: Copy;
    fn splat_t(token: Self::Tok, v: f32) -> Self;
    fn zero_t(token: Self::Tok) -> Self;
    fn max(self, other: Self) -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn simd_gt(self, other: Self) -> Self;
    fn simd_lt(self, other: Self) -> Self;
    fn blend(mask: Self, if_true: Self, if_false: Self) -> Self;
}

macro_rules! impl_vf32 {
    ($ty:ident, $backend:ident) => {
        impl<T: $backend + Copy> Vf32 for $ty<T> {
            type Tok = T;
            #[inline(always)]
            fn splat_t(token: T, v: f32) -> Self {
                $ty::<T>::splat_t(token, v)
            }
            #[inline(always)]
            fn zero_t(token: T) -> Self {
                $ty::<T>::zero_t(token)
            }
            #[inline(always)]
            fn max(self, other: Self) -> Self {
                $ty::<T>::max(self, other)
            }
            #[inline(always)]
            fn abs(self) -> Self {
                $ty::<T>::abs(self)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                $ty::<T>::sqrt(self)
            }
            #[inline(always)]
            fn simd_gt(self, other: Self) -> Self {
                $ty::<T>::simd_gt(self, other)
            }
            #[inline(always)]
            fn simd_lt(self, other: Self) -> Self {
                $ty::<T>::simd_lt(self, other)
            }
            #[inline(always)]
            fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
                $ty::<T>::blend(mask, if_true, if_false)
            }
        }
    };
}
impl_vf32!(GenericF32x8, F32x8Backend);
impl_vf32!(GenericF32x16, F32x16Backend);

/// Whether a tier's kernels step sixteen f32 columns at a time: true where
/// `T`'s target features include AVX-512F (v4, v4x), so one register holds
/// sixteen lanes. v3's sixteen-lane type is two 256-bit halves, and the wider
/// step measured slower there (B at 2048², revision 7: 80.4 → 84.9 ms).
pub(crate) const fn steps_sixteen<T: archmage::SimdToken>() -> bool {
    let f = T::TARGET_FEATURES.as_bytes();
    let pat = b"avx512f";
    let mut i = 0;
    while i + pat.len() <= f.len() {
        let mut j = 0;
        while j < pat.len() && f[i + j] == pat[j] {
            j += 1;
        }
        if j == pat.len() {
            return true;
        }
        i += 1;
    }
    false
}

// ============================================================================
// Phase-6 (§A.16): division-cost reduction for the SIMD formula family.
//
// SHIPPED: ALGEBRAIC SHARING ONLY (exact, no precision tradeoff beyond
// ordinary FP reassociation already tolerated by this file's 5e-4 policy):
// a chained `(a/b)*(c/d)` becomes one division `(a*c)/(b*d)`; two calls
// that share a denominator by construction (`bounded_excess(a,b,c)` +
// `bounded_excess(b,a,c)`) compute it once; a chained ratio-then-saturate
// (`saturate(x/(1+k*act), c)`) collapses to one division by multiplying
// through by the inner denominator. See `ssim_d_local`/
// `bounded_excess_pair`/`pjnd_transducer` (scalar siblings) for the exact
// derivations — same algebra here, f32 SIMD lanes. This drops the
// per-pixel division count from 13 to 8 (with `transducer_bank` on, the
// default) for FREE — native division stays correctly-rounded throughout,
// so no bounds-safety concern beyond what already existed pre-phase-6.
//
// TRIED AND DROPPED: reciprocal-estimate + Newton-Raphson (`V8::<T>::
// recip`, magetypes `archmage/magetypes/src/simd/generic/generated/
// f32x8_impl.rs:371` — every dispatched backend overrides it with a
// hardware estimate refined by one NR step, full f32 precision). Two
// findings, both measured on this box (Ryzen 9 7950X, whatever tier
// `incant!` picks):
//   1. A per-pixel `.min(...)` clamp after every recip-based division
//      (the naive "guard the ~24-bit estimate against overshooting a
//      documented bound" approach) caused a SEVERE regression: 1024²
//      1-thread went 147.0ms -> 738.2ms (~5x worse). `cargo asm` showed
//      `dense_block_kernel_generic`'s AVX2 monomorphization bloated to
//      4268 instructions / 186 un-inlined `call`s — even `_mm256_mul_ps`/
//      `_mm256_add_ps` stopped inlining, 2112-byte stack frame. The extra
//      splat+min at every division site pushed the function past LLVM's
//      inliner threshold and it gave up wholesale.
//   2. Removing the per-pixel clamps (relying on the mean/weighted-mean
//      accumulation structure to keep any epsilon overshoot from
//      amplifying, plus one cheap clamp on the FINAL feature value)
//      recovered MOST but not all of the regression (524.9ms). The
//      remaining gap was `#[inline]` being too weak a hint once these
//      functions grew larger from the algebra fusion — switching to
//      `#[inline(always)]` (KEPT, see below) fixed it completely: 147.2ms,
//      matching plain-division's 144.3ms within noise (both single
//      4-round 1024² measurements, ~5ms MAD). Precision was never the
//      problem — measured max deviation from exact division was 3.246e-7
//      on real AVX2 hardware (`_mm256_rcp_ps` + 1 NR step), five orders of
//      magnitude under the smoke test's 1e-6 slack margin.
//   3. CONCLUSION: recip() is numerically safe and, once the inlining bug
//      is fixed, performance-NEUTRAL on this hardware — not a regression,
//      but not a measurable win either (147.2ms vs 144.3ms, within noise).
//      AMD Zen4's AVX2 divider unit is evidently fast enough that trading
//      1 division for ~1 estimate + 2-3 dependent multiply/subtract ops
//      does not pay for itself. Not shipped: it adds real complexity (the
//      bounds-safety story, an extra formula-precision test) for zero
//      measured benefit on the hardware this was tested on. Kept as a
//      documented, reproducible negative result rather than silently
//      dropped — a future session on different hardware (older x86 with
//      slower `vdivps`, or a tier where `rcp_approx` is a bigger win vs.
//      that tier's native divide) could reopen this with the inlining
//      fix already known. `#[inline(always)]` is KEPT on these six
//      functions regardless of the recip question — it is what actually
//      fixed the regression and is cheap, harmless insurance against the
//      same bloat recurring as the algebra-fusion body grew larger than
//      the pre-phase-6 originals.
// ============================================================================

/// Vectorized [`ssim_d_local`] — fused to one division via `(a*c)/(b*d)`.
/// Identical algebra, f32 lanes. Bounded `[0, 2]`.
#[inline(always)]
fn ssim_d_local_v<V: Vf32>(
    token: V::Tok,
    mu1: V,
    mu2: V,
    s12: V,
    ssq: V,
    c1: V,
    c2: V,
    direct: bool,
    // Revision 7: the luminance term without cancellation; see `ssim_d_local`.
    stable: bool,
) -> V {
    let two = V::splat_t(token, 2.0);
    let one = V::splat_t(token, 1.0);
    let zero = V::zero_t(token);
    let a = two * mu1 * mu2 + c1;
    let b = mu1 * mu1 + mu2 * mu2 + c1;
    let d = ssq - mu1 * mu1 - mu2 * mu2 + c2;
    if direct {
        let mu_diff = mu1 - mu2;
        let err_var = (s12 - mu_diff * mu_diff).max(zero);
        let lum = if stable { mu_diff * mu_diff } else { b - a };
        ((d * lum + a * err_var) / (b * d)).max(zero)
    } else {
        let cov = s12 - mu1 * mu2;
        let c = two * cov + c2;
        let local = (a * c) / (b * d);
        (one - local).max(zero)
    }
}

/// Vectorized [`bounded_sim`] — `(2ab+c)/(a²+b²+c)`. Bounded `(0, 1]`.
#[inline(always)]
fn bounded_sim_v<V: Vf32>(token: V::Tok, a: V, b: V, c: V) -> V {
    let two = V::splat_t(token, 2.0);
    (two * a * b + c) / (a * a + b * b + c)
}

/// Vectorized [`bounded_dissim`]: `one − bounded_sim_v`, or the exact
/// `(a−b)²/(a²+b²+c)` when `stable` (revision 7).
#[inline(always)]
fn bounded_dissim_v<V: Vf32>(
    token: V::Tok,
    a: V,
    b: V,
    c: V,
    stable: bool,
) -> V {
    if stable {
        let d = a - b;
        d * d / (a * a + b * b + c)
    } else {
        V::splat_t(token, 1.0) - bounded_sim_v(token, a, b, c)
    }
}

/// Vectorized [`bounded_excess`] — `max(0, a-b) / (a+b+c)`. Standalone (no
/// shared-denominator partner — see [`bounded_excess_pair_v`] for the
/// paired form). Bounded `[0, 1)`.
#[inline(always)]
fn bounded_excess_v<V: Vf32>(token: V::Tok, a: V, b: V, c: V) -> V {
    let zero = V::zero_t(token);
    (a - b).max(zero) / (a + b + c)
}

/// Vectorized [`bounded_excess_pair`] — `bounded_excess(a,b,c)` PAIRED with
/// `bounded_excess(b,a,c)`, one shared division instead of two (phase-6
/// §A.16: the two calls' denominators are the SAME value by commutativity
/// of addition). Returns `(bounded_excess(a,b,c), bounded_excess(b,a,c))`.
/// Bounded `[0, 1)` each.
#[inline(always)]
fn bounded_excess_pair_v<V: Vf32>(
    token: V::Tok,
    a: V,
    b: V,
    c: V,
) -> (V, V) {
    let zero = V::zero_t(token);
    let denom = a + b + c;
    ((a - b).max(zero) / denom, (b - a).max(zero) / denom)
}

/// [`bounded_excess_pair_v`] for non-negative `a` and `b` (squares), with one
/// division. At most one of `(a − b).max(0)` and `(b − a).max(0)` is nonzero,
/// and `b − a` is exactly `−(a − b)`, so `|a − b| / (a + b + c)` selected by
/// `a > b` (or `b > a`) is that quotient's bits; the other side is the `+0`
/// that `0 / (a + b + c)` gives for the positive denominator.
#[inline(always)]
fn bounded_excess_split_v<V: Vf32>(token: V::Tok, a: V, b: V, c: V) -> (V, V) {
    let zero = V::zero_t(token);
    let q = (a - b).abs() / (a + b + c);
    (V::blend(a.simd_gt(b), q, zero), V::blend(b.simd_gt(a), q, zero))
}

/// Vectorized [`saturate`] — `max(x,0)/(max(x,0)+c)`. Bounded `[0, 1)`.
#[inline(always)]
fn saturate_v<V: Vf32>(token: V::Tok, x: V, c: V) -> V {
    let zero = V::zero_t(token);
    let x = x.max(zero);
    x / (x + c)
}

/// Vectorized [`pjnd_transducer`] — fused `saturate(raw_abs_err/(1+k*act),
/// c)` collapsed to one division (phase-6 §A.16, exact — see the scalar
/// sibling's doc for the derivation). Bounded `[0, 1)`.
#[inline(always)]
fn pjnd_transducer_v<V: Vf32>(
    token: V::Tok,
    raw_abs_err: V,
    act: V,
    k: V,
    c: V,
) -> V {
    let one = V::splat_t(token, 1.0);
    let denom = raw_abs_err + c * (one + k * act);
    raw_abs_err / denom
}

/// Terriberry (2007) single-pass online moments: tracks `n`, `mean`, and
/// the 2nd/3rd/4th central-moment sums `M2/M3/M4 = Σ(x-mean)^k`. `M3` is
/// tracked only because `M4`'s incremental update formula depends on its
/// PRE-update value — not used as an output here. This is what made
/// `ssim_dev2`/`ssim_dev4` (the D8 GMSD-style deviation moments) genuinely
/// single-pass and O(1)-space in iteration 2, unlike iteration 1's
/// store-then-revisit design.
///
/// Phase-4 (§A.14): no longer on the hot path — `compute_channel_scale_v2`
/// now computes `ssim_dev2`/`ssim_dev4` from SIMD-accumulated RAW power
/// sums (Σd, Σd², Σd³, Σd⁴; see `dense_block_kernel_generic`'s doc), since
/// Terriberry's update is inherently sequential and doesn't vectorize
/// across lanes. Kept as the reference implementation and cross-checked
/// against the raw-moment path in
/// `tests::raw_moment_reformulation_matches_terriberry` — this is the
/// proof that the phase-4 reformulation is numerically equivalent, not
/// just "removed and hoped for the best".
#[derive(Default, Clone, Copy)]
pub(crate) struct OnlineMoments {
    n: f64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl OnlineMoments {
    /// Two-pass moments within a small block; no subtraction of raw powers.
    #[inline(never)]
    fn block(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let n = values.len() as f64;
        let origin = values[0];
        let mean = origin + values.iter().map(|v| v - origin).sum::<f64>() / n;
        let mut out = Self {
            n,
            mean,
            ..Self::default()
        };
        for &v in values {
            let d = v - mean;
            let d2 = d * d;
            out.m2 += d2;
            out.m3 += d2 * d;
            out.m4 += d2 * d2;
        }
        out
    }

    /// Chan/Pébay merge, always called in raster block/strip order.
    fn merge(&mut self, b: Self) {
        if b.n == 0.0 {
            return;
        }
        if self.n == 0.0 {
            *self = b;
            return;
        }
        let a = *self;
        let n = a.n + b.n;
        let d = b.mean - a.mean;
        let d2 = d * d;
        let d3 = d2 * d;
        let d4 = d2 * d2;
        self.mean = a.mean + d * (b.n / n);
        self.m4 = a.m4
            + b.m4
            + d4 * a.n * b.n * (a.n * a.n - a.n * b.n + b.n * b.n) / (n * n * n)
            + 6.0 * d2 * (a.n * a.n * b.m2 + b.n * b.n * a.m2) / (n * n)
            + 4.0 * d * (a.n * b.m3 - b.n * a.m3) / n;
        self.m3 = a.m3
            + b.m3
            + d3 * a.n * b.n * (a.n - b.n) / (n * n)
            + 3.0 * d * (a.n * b.m2 - b.n * a.m2) / n;
        self.m2 = a.m2 + b.m2 + d2 * a.n * b.n / n;
        self.n = n;
    }

}

/// Revision 7's per-row central moments of `d` (the `rowlanes` era): a
/// sixteen-lane f64 sum (lane `x & 15`, ascending `x`, closed by the
/// [`crate::featcanon::Lanes16F64`] tree) gives the mean, then the centred
/// `(d−μ)²`, `(d−μ)³`, `(d−μ)⁴` sums in sixteen f64 lanes the same way. The
/// canonical scalar form; the vector dense kernel runs the same ops per lane.
pub(crate) fn row_moments_canon(v: &[f64]) -> OnlineMoments {
    use crate::featcanon::{Lanes16F64, Pool as _};
    let n = v.len();
    if n == 0 {
        return OnlineMoments::default();
    }
    let mut s1 = [0.0f64; 16];
    for (x, &d) in v.iter().enumerate() {
        s1[x & 15] += d;
    }
    let mean = Lanes16F64(s1).fin() / n as f64;
    let (mut s2, mut s3, mut s4) = ([0.0f64; 16], [0.0f64; 16], [0.0f64; 16]);
    for (x, &v) in v.iter().enumerate() {
        let d = v - mean;
        let q = d * d;
        s2[x & 15] += q;
        s3[x & 15] += q * d;
        s4[x & 15] += q * q;
    }
    OnlineMoments {
        n: n as f64,
        mean,
        m2: Lanes16F64(s2).fin(),
        m3: Lanes16F64(s3).fin(),
        m4: Lanes16F64(s4).fin(),
    }
}

/// Running `(Σw·v, Σw)` accumulator for the canonical weighted-mean
/// pooling form `Σw·v/Σw`, computed incrementally instead of from
/// materialized weight/value arrays.
#[derive(Default, Clone, Copy)]
pub(crate) struct WeightedSum {
    pub(crate) num: f64,
    pub(crate) den: f64,
}
impl WeightedSum {
    #[inline]
    fn finish(&self) -> f64 {
        if self.den < 1e-12 {
            0.0
        } else {
            self.num / self.den
        }
    }
    /// Phase-5 (§A.15): fold another (partial-strip) accumulator's
    /// `(num, den)` in directly — `Σw·v` and `Σw` are both plain sums, so
    /// combining N strips' partial `WeightedSum`s is exactly equivalent to
    /// one whole-image `WeightedSum` (no reassociation beyond ordinary
    /// floating-point addition, same class of drift as any other
    /// SIMD-lane-then-strip reduction already tolerated in this file).
    #[inline]
    fn accumulate(&mut self, other: &WeightedSum) {
        self.num += other.num;
        self.den += other.den;
    }
}

// ============================================================================
// FeatureRegime + result + explicit-regime view
// ============================================================================

/// Result of [`compute_v2_features_impl`]/[`crate::Zensim::compute_v2_features`].
#[derive(Debug, Clone)]
pub struct ZensimV2Result {
    features: Vec<f64>,
}

impl ZensimV2Result {
    pub fn into_features(self) -> Vec<f64> {
        self.features
    }

}

// ============================================================================
// Per-channel-per-scale computation — SINGLE PASS, O(1) accumulators
// ============================================================================

/// Which FORMULA REVISION an extraction computes — re-exported so
/// [`V2NewFeatureToggles::formula_revision`] is nameable by the callers that
/// set it.
///
/// `#[doc(hidden)]`: the revision axis is research/extraction machinery, not
/// product surface. A default build's `Zensim::compute` resolves it per bake
/// and never asks a caller.
#[doc(hidden)]
pub use crate::feature_defs::FormulaRevision;
/// Runtime toggles for the phase-2 new-feature GROUPS (A.10 candidates),
/// grouped by shared per-pixel computation rather than 1:1 with feature
/// slots (per `docs/FEATURE_V2_SPEC_2026-07-18.md` §A.12's per-group
/// marginal-cost measurement):
///
/// - `gradient_features`: GMS + chroma-edge-GMS (free via channel axis) +
///   ringing + banding + edge-width-change. These all consume the same
///   `sqrt`-based gradient magnitude, so they are gated together — turning
///   this off skips the `sqrt` calls entirely, not just the 4 accumulator
///   adds.
/// - `transducer_bank`: the 2 EXTRA masking-transducer k values (the core
///   `k=4.0` transducer, `PJND_TRANSDUCER`, is always computed — it is not
///   a phase-2 addition, iteration 1 already shipped it).
/// - `blockiness`: oriented blockiness (shares neighbor loads with the
///   gradient group but not the `sqrt`, so toggled independently).
///
/// Default: all `true` (every candidate on). A group that fails the speed
/// gate is set `false` here — NOT deleted — per the phase-2 brief ("If a
/// feature blows the budget, keep it OFF by default and say so").
#[derive(Debug, Clone, Copy)]
pub struct V2NewFeatureToggles {
    /// Preserve the plan's reference-only feature selection.
    pub(crate) omit_pjnd_fragility: bool,
    pub gradient_features: bool,
    pub transducer_bank: bool,
    pub blockiness: bool,
    pub formula_revision: crate::feature_defs::FormulaRevision,
    pub v1_pools: V1PoolsMode,
    #[doc(hidden)]
    pub v1_only: bool,

}

/// Which of v1's pool slots (`f156..372`) the folded walk emits live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum V1PoolsMode {
    /// Structural zeros (the production folded regimes).
    #[default]
    Off,
    /// **The peak block only** (`f156..228`) — masked and IW stay 0.
    ///
    /// Costs the SAME as [`Self::Off`]: the peak accumulators (`ssim_d8`,
    /// `edge_art8`, `edge_det8` and the three running maxima) are produced
    /// UNCONDITIONALLY by `fused::fused_vblur_features_ssim` on every path
    /// and merged unconditionally by `V1BasicSums::accumulate`, so `Off` has
    /// been paying for them all along and simply declined to emit them. What
    /// this mode skips relative to [`Self::Full`] is the masked/IW pass
    /// group — the band activity chain (`abs_diff_into` +
    /// `box_blur_1pass_into`), the fused kernel's `store_mu`/`store_sigma`
    /// side-outputs, and the three `*_inline_both` kernels — which is the
    /// whole measurable cost of the pool block.
    ///
    /// The band scratch is still handed to the band so the memory-traffic
    /// self-blur shape (`FoldHSource::SelfBlur`) stays available; it owns
    /// only the four H planes there, never the pool planes.
    Peaks,
    /// All 216 slots: peaks (72) + masked (72) + IW (72), v1-exact.
    Full,
}

impl V1PoolsMode {
    /// The ten carrier slots (`Carriers`), in v1's 372 layout.
    pub const CARRIER_SLOTS: [usize; 10] = [178, 190, 196, 226, 231, 237, 243, 303, 321, 333];
}

impl Default for V2NewFeatureToggles {
    fn default() -> Self {
        Self {
            omit_pjnd_fragility: false,
            gradient_features: true,
            transducer_bank: true,
            blockiness: true,
            formula_revision: crate::ssim_form::active_revision(),
            v1_pools: V1PoolsMode::Off,
            v1_only: false,
        }
    }
}

/// Perf-pass (phase 3, `docs/FEATURE_V2_SPEC_2026-07-18.md` §A.13) reusable
/// scratch buffers for [`compute_channel_scale_v2`], sized once for the
/// LARGEST scale (scale 0) and reused (sliced down to the current scale's
/// `n`) across all `n_scales * 3` calls in
/// [`compute_v2_features_impl_with_toggles`]. Eliminates 9 `vec![0.0f32; n]`
/// heap allocations (+ their zero-fill) per call — 108 allocations per
/// `compute_v2_features_impl_with_toggles` invocation before this change, 9
/// after (one-time, at the scratch's construction). Pure memory-management
/// change: does not alter any arithmetic or its order, so it carries zero
/// numerical risk (and zero risk to v1's golden gate, since v1 never
/// calls into this module).
///
/// Phase-5 (§A.15) RESIZED: was sized for the FULL channel-scale plane
/// (`width*height`, up to 4.2M elements at 2048²); now sized for one
/// STRIP-plus-halo (`width*(STRIP_ROWS+2*HALO_P)`, ≤ ~172K elements even
/// at 2048² width) — the actual memory-traffic reduction the bandwidth
/// hypothesis predicts should matter.
///
/// Ref-reuse pass: the halo-gather targets (`src_wide`/`dst_wide`) moved IN
/// here (they were per-call locals). The old aliasing objection — a
/// `&mut ScratchV2Strip` borrow against the `&[f32]` slices `run_blur_pass`
/// needs — is dissolved by `run_blur_pass_strip`, which destructures the
/// struct ONCE into disjoint field borrows (src/dst read-only, moment
/// buffers `&mut`), so no cell tricks are needed. Moving them in makes the
/// whole scratch set reusable ACROSS pairs via [`V2Scratch`], eliminating
/// the last per-call `vec![0.0f32; ..]` traffic (2 buffers × 3 channels ×
/// 4 scales = 24 zero-filled allocations per pair).
struct ScratchV2Strip {
    /// Halo-gathered strip inputs (`gather_strip_halo` targets) — the
    /// mirror-padded local windows of the source/distorted planes that
    /// every downstream stage of one strip iteration reads.
    src_wide: Vec<f32>,
    dst_wide: Vec<f32>,
    /// H-blur-only intermediates from [`crate::blur::fused_blur_h_ssim`]
    /// (see below) — analogous to v2 iteration-2's old per-call `tmp`, but
    /// now 4-wide since the fused kernel produces all four H-blurred planes
    /// in one pass.
    mu1_h: Vec<f32>,
    mu2_h: Vec<f32>,
    ssq_h: Vec<f32>,
    s12_h: Vec<f32>,
    /// Fully (H+V) blurred planes — same meaning as iteration-2's original
    /// `mu1`/`mu2`/`ssq`/`s12` locals.
    mu1: Vec<f32>,
    mu2: Vec<f32>,
    ssq: Vec<f32>,
    s12: Vec<f32>,
    /// Activity path (unchanged from iteration 2: `abs(src - mu1)` then
    /// blurred) — kept as its own two-buffer pair since
    /// `box_blur_1pass_into` still needs a temp scratch distinct from the
    /// 4-wide fused-H outputs above.
    abs_src: Vec<f32>,
    activity_tmp: Vec<f32>,
    activity: Vec<f32>,
    /// Streamed-walk σ-split plane: this kernel strip's `blur(src²)` wide
    /// buffer, filled per strip by the streaming walk (square + H + V
    /// inside the wide window — the P2 tiling) after the activity chain
    /// frees `abs_src`/`activity_tmp` for reuse as its temps. Unused by
    /// the materialized walk (which reads cached/replayed full planes).
    bs2: Vec<f32>,
    activity_dst: Vec<f32>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ComputeSet {
    /// The formula revision this request computes — see
    /// [`V2NewFeatureToggles::formula_revision`].
    pub formula_revision: crate::feature_defs::FormulaRevision,
    /// The v1 basic fold (`f0..155`). Always on today; a field so that a
    /// v2-only request is expressible rather than impossible.
    pub v1_basic: bool,
    /// Full-resolution X/B moments. Plans may omit them for basic/peak subsets.
    /// Conversion and the coarser XYB pyramid always remain complete.
    pub full_res_xb: bool,
    /// Coarse scales whose basic/peak moments need only Y (bits 1..3).
    pub coarse_y_only_scales: u8,
    /// Only basic local SSIM/edge moments and MSE; no peaks or HF ratios.
    pub local_only: bool,
    pub omit_edges: bool,
    /// v1's masked/IW/soft-peak pool slots — the 13.6 % pass of item E.
    pub v1_pools: V1PoolsMode,
    /// Scales that run the shared masked/IW activity chain in Full mode.
    /// Public extraction retains all scales; declared-ID plans narrow this.
    pub v1_full_scales: u8,
    /// Scales at which any v1-era slot (basic, peaks, masked, IW) is read.
    /// Public extraction keeps all scales; declared-ID plans narrow it, and the
    /// walk skips a scale's strips whole when neither this nor `v2_scales`
    /// names it (the pyramid is still built through it).
    pub v1_scales: u8,
    /// Actual v2-era work, including adjacent-scale gradient dependencies.
    pub v2_scales: u8,
    /// The v2-era blocks as a group (`f372..`): false for a `v1_only`
    /// request, which then computes NOTHING v2-era.
    pub v2_blocks: bool,
    pub gradient: bool,
    pub blockiness: bool,
    pub transducer_bank: bool,
    /// Declared plans may omit the reference-only PJND fragility finalizer.
    pub omit_pjnd_fragility: bool,

}

impl ComputeSet {
    pub(crate) const ALL_SCALES: u8 = (1 << crate::NUM_SCALES) - 1;

    /// Narrow kernels without changing layout or the within-scale arithmetic.
    pub(crate) fn at_scale(self, scale: usize) -> Self {
        let live = self.v2_scales & (1 << scale) != 0;
        // A dead scale (see `scale_dead`) is skipped whole by the walk, so nothing (not even the v1 fold's free
        // extras) is computed there. Every scale the walk still runs keeps its view unchanged.
        let dead = self.scale_dead(scale);
        Self {
            v1_basic: self.v1_basic && !dead,
            v1_pools: if dead {
                V1PoolsMode::Off
            } else {
                self.pools_at(scale)
            },
            v2_blocks: self.v2_blocks && live,
            gradient: self.gradient && live,
            blockiness: self.blockiness && live,
            ..self
        }
    }

    /// Only Full has a per-scale restriction. Peaks keeps the same basic
    /// and peak accumulators while omitting the activity/weighted-pool chain.
    pub(crate) fn pools_at(&self, scale: usize) -> V1PoolsMode {
        if self.v1_pools == V1PoolsMode::Full && self.v1_full_scales & (1 << scale) == 0 {
            V1PoolsMode::Peaks
        } else {
            self.v1_pools
        }
    }

    pub(crate) fn full_pool_scales(&self) -> u8 {
        match self.v1_pools {
            V1PoolsMode::Full => self.v1_full_scales,
            _ => 0,
        }
    }

    /// The ONE derivation from the public request type. Every `&&
    /// v2_blocks` here was previously written out at a call site; the
    /// invariant is that a v1-only request forces every v2-era block off
    /// rather than asserting they are already off, so `v1_only` composes
    /// with any toggle set (which is what makes it a drop-in bench A/B).
    pub(crate) fn from_toggles(t: V2NewFeatureToggles) -> Self {
        let v2_blocks = !t.v1_only;
        let mut set = Self {
            formula_revision: t.formula_revision,
            v1_basic: true,
            full_res_xb: true,
            coarse_y_only_scales: 0,
            local_only: false,
            omit_edges: false,
            v1_pools: t.v1_pools,
            v1_full_scales: Self::ALL_SCALES,
            v1_scales: Self::ALL_SCALES,
            v2_scales: Self::ALL_SCALES,
            v2_blocks,
            gradient: t.gradient_features && v2_blocks,
            blockiness: t.blockiness && v2_blocks,
            transducer_bank: t.transducer_bank,
            omit_pjnd_fragility: t.omit_pjnd_fragility,
        };
        set.rev5_scope();
        set
    }

    pub(crate) fn rev5_scope(&mut self) {
        if self.formula_revision < crate::feature_defs::FormulaRevision::Rev5 {
            return;
        }
        if self.v1_pools != V1PoolsMode::Off {
            self.v1_pools = V1PoolsMode::Peaks;
        }
        self.v1_full_scales = Self::ALL_SCALES;
    }

    pub(crate) fn free_work(&self, ch: usize, scale: usize) -> crate::fused::FreeExtrasWork {
        if self.formula_revision >= FormulaRevision::Rev5 && scale == 0 && ch != 1 {
            // Count actual scale-zero chroma consumers, regardless of plan intent.
        }
        crate::fused::FreeExtrasWork {
            revision: Some(self.formula_revision),
            omit_peaks: self.formula_revision >= FormulaRevision::Rev5
                && self.v1_pools == V1PoolsMode::Off,
            local_only: self.local_only,
            omit_edges: self.omit_edges,
        }
    }

    /// Which plane GROUPS the strip scratch must hold, given whether the v1
    /// fold is running self-blur bands (in which case phase A is skipped
    /// whole and its four fused-H outputs are never written).
    pub(crate) fn plane_needs(&self, self_blur: bool) -> StripPlaneNeeds {
        StripPlaneNeeds {
            h: !self_blur,
            v2: self.v2_blocks,
        }
    }

    /// True when the fold can own its H planes band-locally — the shape that
    /// skips phase A entirely. Needs the band pool scratch, which
    /// [`V1PoolsMode::Full`] and [`V1PoolsMode::Peaks`] both provide
    /// (`Peaks` hands the band the scratch for the four H planes alone —
    /// [`BandPoolWork::HOnly`]); [`V1PoolsMode::Off`] hands the band no
    /// scratch at all, which is why `fold_engine::pools_mode_for_need` never
    /// returns it.
    ///
    /// **THE owner of this predicate.** The strip loop's `fuse_channels` arm
    /// reads it rather than re-deriving it — those were two separate
    /// derivations until 2026-09-05 and they DISAGREED on `Peaks`: this one
    /// said `Full` only, the loop said `Full | Peaks`. Since this one sizes
    /// the strip scratch (via [`Self::plane_needs`]) and the loop decides
    /// whether phase A runs, a `Peaks` walk reserved phase A's four fused-H
    /// planes and then skipped phase A whole — allocating and first-touching
    /// planes nothing ever wrote, on the one mode the fast class ships.
    /// `self_blur_sizing_predicate_matches_the_strip_loop_predicate` pins
    /// them together; it fails against the pre-fix definition.
    pub(crate) fn self_blur_eligible(&self) -> bool {
        !self.v2_blocks
            && self.v1_basic
            && matches!(self.v1_pools, V1PoolsMode::Full | V1PoolsMode::Peaks)
    }

    pub(crate) fn populated_slots(
        &self,
        n_scales: usize,
        layout_width: usize,
    ) -> crate::feature_set_id::SlotSet {
        use crate::feature_set_id::SlotSet;
        if self.v2_scales != Self::ALL_SCALES || self.v1_scales != Self::ALL_SCALES {
            return SlotSet::from_slots((0..n_scales).flat_map(|scale| {
                let local = Self {
                    v1_scales: Self::ALL_SCALES,
                    v2_scales: Self::ALL_SCALES,
                    ..self.at_scale(scale)
                };
                local
                    .populated_slots(n_scales, layout_width)
                    .iter_slots()
                    .filter(move |&id| {
                        crate::feature_defs::def_at(id, n_scales)
                            .is_some_and(|d| usize::from(d.scale) == scale)
                            && self.computes_edge_width_change(id, n_scales)
                    })
                    .collect::<Vec<_>>()
            }));
        }
        let basic = n_scales * 3 * crate::metric::FEATURES_PER_CHANNEL_BASIC;
        let peaks_end = n_scales * 3 * crate::metric::FEATURES_PER_CHANNEL_WITH_PEAKS;
        let masked_end = n_scales * 3 * crate::metric::FEATURES_PER_CHANNEL_EXTENDED;
        let v1_total = masked_end + n_scales * 3 * crate::metric::FEATURES_PER_CHANNEL_IW;
        let v2_end = v1_total + n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;

        let mut ranges: Vec<(usize, usize)> = Vec::new();
        if self.v1_basic {
            ranges.push((0, basic));
        }
        match self.v1_pools {
            V1PoolsMode::Off => {}
            V1PoolsMode::Peaks => ranges.push((basic, peaks_end)),
            V1PoolsMode::Full => ranges.push((basic, v1_total)),
        }
        if self.v2_blocks {
            ranges.push((v1_total, v2_end));
        }
        let slots = SlotSet::from_ranges(ranges)
            .clipped_to(layout_width);
        SlotSet::from_slots(slots.iter_slots().filter(|&id| {
            self.computes_edge_width_change(id, n_scales)
                && (!self.omit_pjnd_fragility
                || !(v1_total..v2_end).contains(&id)
                || (id - v1_total) % FEATURES_PER_CHANNEL_V2_TOTAL != idx::PJND_FRAGILITY)
                && (!self.local_only
                    || (id < basic && id % crate::metric::FEATURES_PER_CHANNEL_BASIC < 10))
                && (!self.omit_edges || (id < basic && matches!(id % 13, 0..=2 | 9)))
                && (self.full_res_xb || !Self::is_full_res_xb(id, n_scales))
                && !crate::feature_defs::def_at(id, n_scales).is_some_and(|d| {
                    d.scale > 0
                        && self.coarse_y_only_scales & (1 << d.scale) != 0
                        && matches!(
                            d.channel,
                            crate::feature_defs::Channel::X | crate::feature_defs::Channel::B
                        )
                })
                && !(self.v1_pools == V1PoolsMode::Full
                    && (peaks_end..v1_total).contains(&id)
                    && crate::feature_defs::def_at(id, n_scales)
                        .is_some_and(|d| self.pools_at(usize::from(d.scale)) != V1PoolsMode::Full))
        }))
    }

    /// An edge-width-change slot pairs two scales (the last scale's slot copies
    /// the one below), so the walk computes it only where it has gradient work
    /// and the slot's channel at both. Every other slot passes.
    fn computes_edge_width_change(&self, id: usize, n_scales: usize) -> bool {
        let Some(d) = crate::feature_defs::def_at(id, n_scales) else {
            return true;
        };
        if d.signal.name != "edge_width_change" {
            return true;
        }
        let ch = match d.channel {
            crate::feature_defs::Channel::X => 0,
            crate::feature_defs::Channel::Y => 1,
            crate::feature_defs::Channel::B => 2,
        };
        let s = usize::from(d.scale).min(n_scales - 2);
        [s, s + 1]
            .into_iter()
            .all(|x| self.at_scale(x).gradient && self.channel_active(x, ch))
    }

    /// The v1 activity chains are channel-local, so masked/IW subsets may
    /// also omit finest X/B. Cross-channel and free-extra families retain
    /// the complete walk until their dependencies have their own plan.
    pub(crate) fn allows_full_res_y_subset(&self) -> bool {
        (!self.v2_blocks || self.v2_scales & 1 == 0 || self.chroma_local_families())
            && matches!(
                self.v1_pools,
                V1PoolsMode::Off | V1PoolsMode::Peaks | V1PoolsMode::Full
            )
    }

    /// True when no family that reads across channels is computed: the v1 blocks and the v2 core are
    /// channel-local (each channel's slots read only that channel's planes), so X/B work may be skipped
    /// families keep the complete walk until they have their own channel plan.
    pub(crate) fn chroma_local_families(&self) -> bool {
        true
    }

    #[inline]
    pub(crate) fn channel_active(&self, scale: usize, channel: usize) -> bool {
        !self.scale_dead(scale)
            && (channel == 1
                || if scale == 0 {
                    self.full_res_xb
                } else {
                    self.coarse_y_only_scales & (1 << scale) == 0
                })
    }

    /// A scale at which nothing is read: no v1-era slot (`v1_scales`) and no v2-era work (`v2_blocks` at a
    /// `v2_scales` bit). Every walk computes no channel there; the pyramid is still built through it. Only
    /// declared-ID plans narrow `v1_scales`, so public extraction never has a dead scale.
    pub(crate) fn scale_dead(&self, scale: usize) -> bool {
        let bit = 1 << scale;
        self.v1_scales & bit == 0 && !(self.v2_blocks && self.v2_scales & bit != 0)
    }

    pub(crate) fn is_full_res_xb(id: usize, n_scales: usize) -> bool {
        use crate::feature_defs::Channel;
        crate::feature_defs::def_at(id, n_scales)
            .is_some_and(|d| d.scale == 0 && matches!(d.channel, Channel::X | Channel::B))
    }

}

/// Which groups of [`ScratchV2Strip`]'s planes a walk will actually write.
///
/// The set is a property of the REQUEST, and the fold-footprint lane measured
/// what ignoring it costs: a `v1_only` + [`V1PoolsMode::Full`] score — the
/// fold-backed scoring walk — runs self-blur bands, so phase A never executes
/// and the ONLY planes it writes are `src_wide` / `dst_wide`. The other twelve
/// were still allocated at `3 × 148 × width × 4` bytes apiece: **21,312·width
/// bytes untouched**, 24.6 MB at 1152² and 49.1 MB at 2304².
///
/// Under stock glibc those pages are demand-zero (`vec![0.0; n]` →
/// `alloc_zeroed` → `mmap`) and never fault, so they cost address space rather
/// than RSS — but that is an ALLOCATOR POLICY, not a property of the program.
/// MEASURED: with `MALLOC_ARENA_MAX=1` the same binary's `score_fold` peak RSS
/// at 1152²/1T goes 61,952 KiB → 71,768 KiB, because the allocation now comes
/// from the main heap and `calloc` has to memset it. Not asking for the planes
/// is the only way to not depend on that.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct StripPlaneNeeds {
    /// `mu1_h` / `mu2_h` / `ssq_h` / `s12_h` — phase A's fused-H outputs.
    /// False only when every v1 band blurs its own rows ([`FoldHSource::SelfBlur`]).
    h: bool,
    /// `mu1` / `mu2` / `ssq` / `s12` / `abs_src` / `activity_tmp` / `activity`
    /// / `bs2` — everything below `run_blur_pass_inner`'s `want_v2` early
    /// return, plus the σ-split and dst-activity buffers that reuse its temps.
    v2: bool,
}

impl StripPlaneNeeds {
    /// Every plane — what the materialized walks and the reference-moment
    /// cache ask for, and the safe default for any new caller.
    const ALL: Self = Self { h: true, v2: true };
    const NONE: Self = Self {
        h: false,
        v2: false,
    };
    fn union(self, o: Self) -> Self {
        Self {
            h: self.h || o.h,
            v2: self.v2 || o.v2,
        }
    }
}

impl ScratchV2Strip {
    fn new(max_n: usize) -> Self {
        Self::new_for(max_n, StripPlaneNeeds::ALL)
    }

    fn new_for(max_n: usize, needs: StripPlaneNeeds) -> Self {
        let hn = if needs.h { max_n } else { 0 };
        let vn = if needs.v2 { max_n } else { 0 };
        Self {
            src_wide: vec![0.0f32; max_n],
            dst_wide: vec![0.0f32; max_n],
            mu1_h: vec![0.0f32; hn],
            mu2_h: vec![0.0f32; hn],
            ssq_h: vec![0.0f32; hn],
            s12_h: vec![0.0f32; hn],
            mu1: vec![0.0f32; vn],
            mu2: vec![0.0f32; vn],
            ssq: vec![0.0f32; vn],
            s12: vec![0.0f32; vn],
            abs_src: vec![0.0f32; vn],
            activity_tmp: vec![0.0f32; vn],
            activity: vec![0.0f32; vn],
            bs2: vec![0.0f32; vn],
            activity_dst: Vec::new(),
        }
    }
}

/// Reusable cross-pair scratch for the v2 extraction kernels — all 13
/// strip-sized working buffers for all 3 channels. Create ONCE per worker
/// thread and pass to
/// [`Zensim::compute_v2_features_with_ref_and_scratch`](crate::Zensim::compute_v2_features_with_ref_and_scratch)
/// for every pair that thread scores: buffers are lazily (re)sized to the
/// largest image seen and reused verbatim afterwards, so steady-state
/// extraction performs ZERO scratch allocation. Contents are overwritten
/// before every read (each strip iteration fully writes the ranges it
/// consumes), so reuse across unrelated pairs cannot leak data between
/// them — guarded by `scratch_reuse_matches_fresh_scratch` in this file's
/// tests.
pub struct V2Scratch {
    strips: [ScratchV2Strip; 3],
    sized_for: usize,
    /// The UNION of every plane set asked for so far — grow-only, exactly like
    /// `sized_for`, so a driver that alternates a v1-only score with a 944
    /// extraction converges to [`StripPlaneNeeds::ALL`] and never thrashes.
    sized_needs: StripPlaneNeeds,
    /// Recycled rolling-plane buffers for the STREAMING walk's
    /// [`crate::feature_v2_stream::StripPlaneProducer`] (drained at
    /// construction, refilled by `recycle` at walk end) — steady-state
    /// batch extraction performs zero producer allocation.
    stream_pool: Vec<Vec<f32>>,
    #[cfg(feature = "threads")]
    rev5_jobs: Vec<Rev5StripJob>,
    /// Recycled row-block buffers for [`crate::feature_v2_blocks`].
    #[cfg(feature = "threads")]
    block_pool: std::sync::Arc<crate::feature_v2_blocks::BlockPool>,
    #[cfg(all(test, feature = "threads"))]
    rev5_job_budget: Option<usize>,
}

impl V2Scratch {
    /// New, empty scratch. Buffers are allocated on first use.
    pub fn new() -> Self {
        Self {
            strips: [
                ScratchV2Strip::new(0),
                ScratchV2Strip::new(0),
                ScratchV2Strip::new(0),
            ],
            sized_for: 0,
            sized_needs: StripPlaneNeeds::NONE,
            stream_pool: Vec::new(),
            #[cfg(feature = "threads")]
            rev5_jobs: Vec::new(),
            #[cfg(feature = "threads")]
            block_pool: Default::default(),
            #[cfg(all(test, feature = "threads"))]
            rev5_job_budget: None,
        }
    }

    /// [`Self::ensure`] restricted to the plane groups the caller will write
    /// (see [`StripPlaneNeeds`]). Both the size and the set are grow-only, so
    /// this can only ever allocate LESS than `ensure` — never differently.
    fn ensure_for(&mut self, strip_max_n: usize, needs: StripPlaneNeeds) {
        let want = self.sized_needs.union(needs);
        if strip_max_n > self.sized_for || want != self.sized_needs {
            let n = strip_max_n.max(self.sized_for);
            self.strips = [
                ScratchV2Strip::new_for(n, want),
                ScratchV2Strip::new_for(n, want),
                ScratchV2Strip::new_for(n, want),
            ];
            self.sized_for = n;
            self.sized_needs = want;
        }
    }
}

impl Default for V2Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Row-band-parallel wrapper for [`crate::blur::fused_blur_h_ssim`].
///
/// **Bit-exact by construction, and the construction is the whole argument.**
/// A horizontal box blur is an independent running-sum recurrence per row, and
/// the kernels walk rows in groups of [`H_BLUR_ROW_GROUP`]
/// (`fused_blur_h_ssim_*_inner`'s `run_group(rg * 8, 8)` + one partial tail).
/// Bands here start at multiples of that group size, so every band's internal
/// grouping is a **sub-sequence of the whole-plane call's** — each row lands in
/// a group of the same size, at the same offset within it, on the same SIMD
/// tier. Only the LAST band can hold a partial group, which is exactly the
/// partial group the whole-plane call's tail would have run.
/// `phase_a_blur_bands_are_bit_exact` pins that over every band size and
/// geometry; `box_blur_h_ring_matches_regathered_reference` independently pins
/// the per-row claim the construction rests on.
///
/// Predecessor context (`extraction_perf_and_buffered_removal_2026-08-30.md`
/// §11.1): this lever was implemented, measured NEUTRAL and reverted **in the
/// 944-full walk**, where phase A is a small share behind `dense_block_kernel`.
/// In the `v1_only` SCORING request that backs the fold engine there is no
/// dense kernel at all and phase A is 36 % of the walk at 16 threads with an
/// occupancy of 0.157 — a different measurement, not a re-run of that one.
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_banded(
    src: &[f32],
    dst: &[f32],
    mu1_h: &mut [f32],
    mu2_h: &mut [f32],
    ssq_h: &mut [f32],
    s12_h: &mut [f32],
    width: usize,
    height_local: usize,
    #[allow(unused_variables)] parallel: bool,
    revision: FormulaRevision,
) {
    #[cfg(feature = "threads")]
    if parallel && height_local > H_BLUR_BAND_ROWS && width > 0 {
        use rayon::prelude::*;
        let n = H_BLUR_BAND_ROWS * width;
        mu1_h
            .par_chunks_mut(n)
            .zip(mu2_h.par_chunks_mut(n))
            .zip(ssq_h.par_chunks_mut(n))
            .zip(s12_h.par_chunks_mut(n))
            .enumerate()
            .for_each(|(b, (((m1, m2), sq), s12))| {
                let y0 = b * H_BLUR_BAND_ROWS;
                let rows = m1.len() / width;
                let lo = y0 * width;
                let hi = lo + rows * width;
                // The tile lives on `fused_blur_h_ssim` itself, so it nests
                // inside this row band automatically: the band gives the
                // thread its rows, the tile keeps that band's 6-plane window
                // dense. Each worker owns its own thread-local tile arena.
                crate::blur::fused_blur_h_ssim_at_revision(
                    &src[lo..hi],
                    &dst[lo..hi],
                    m1,
                    m2,
                    sq,
                    s12,
                    width,
                    rows,
                    BLUR_RADIUS,
                    revision,
                );
            });
        return;
    }
    crate::blur::fused_blur_h_ssim_at_revision(
        src,
        dst,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );
}

/// Shared body for [`run_blur_pass`]/[`run_blur_pass_strip`] — the actual
/// fused H-blur + 4x V-blur + activity sequence, over explicit buffers.
#[allow(clippy::too_many_arguments)]
fn run_blur_pass_inner(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height_local: usize,
    mu1_h: &mut [f32],
    mu2_h: &mut [f32],
    ssq_h: &mut [f32],
    s12_h: &mut [f32],
    mu1: &mut [f32],
    mu2: &mut [f32],
    ssq: &mut [f32],
    s12: &mut [f32],
    abs_src: &mut [f32],
    activity_tmp: &mut [f32],
    activity: &mut [f32],
    want_v2: bool,
    parallel: bool,
    revision: FormulaRevision,
) {

    let n = width * height_local;
    let mu1_h = &mut mu1_h[..n];
    let mu2_h = &mut mu2_h[..n];
    let ssq_h = &mut ssq_h[..n];
    let s12_h = &mut s12_h[..n];
    fused_blur_h_ssim_banded(
        src,
        dst,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        width,
        height_local,
        parallel,
        revision,
    );

    // BLOCK-SKIPPING: everything below this line feeds v2-era kernels ONLY.
    // `fold_v1_basic_bands` reads the H-blurred planes above and computes its
    // own activity internally, so a v1-only request can stop here — four
    // whole-window `box_blur_v_from_copy` sweeps and the activity chain
    // (abs-diff + a full 1-pass blur) are skipped, not merely unread.
    if !want_v2 {
        return;
    }

    let mu1 = &mut mu1[..n];
    crate::blur::box_blur_v_from_copy_at_revision(
        mu1_h,
        mu1,
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );
    let mu2 = &mut mu2[..n];
    crate::blur::box_blur_v_from_copy_at_revision(
        mu2_h,
        mu2,
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );
    let ssq = &mut ssq[..n];
    crate::blur::box_blur_v_from_copy_at_revision(
        ssq_h,
        ssq,
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );
    let s12 = &mut s12[..n];
    crate::blur::box_blur_v_from_copy_at_revision(
        s12_h,
        s12,
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );

    let abs_src = &mut abs_src[..n];
    crate::simd_ops::abs_diff_into(src, mu1, abs_src);
    let activity = &mut activity[..n];
    crate::blur::box_blur_1pass_at_revision(
        abs_src,
        activity,
        &mut activity_tmp[..n],
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );
}

// ============================================================================
// Phase-4 (§A.14): magetypes-SIMD dense formula pass + gradient pass.
//
// Two separate `#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]` hot loops,
// matching the brief's "compute_channel_scale_v2's formula pass + the shared
// gradient pass" scope exactly:
//   - `dense_block_kernel`: SSIM raw moments, edge artifact/detail, bounded
//     MSE, HF gain/loss/mag-loss, PJND core + transducer-bank, and all 11
//     masked/IW/soft-peak weighted-pool pairs. ALWAYS runs (matches the
//     scalar loop's unconditional block).
//   - `gradient_block_kernel`: GMS, ringing, banding, grad_src/grad_dst sums.
//     Only invoked when `toggles.gradient_features` — the toggle check is
//     hoisted OUTSIDE the pixel loop entirely (was a per-pixel branch in the
//     scalar version; now a single runtime branch per `compute_channel_
//     scale_v2` call), which is itself a small, free improvement.
//
// Blockiness is NOT ported to SIMD this phase — it is inherently sparse
// (only x%8==0 or y%8==0 lattice positions contribute), so a dense 8-wide
// SIMD pass would spend 7/8 of its lanes on masked-out zero contributions.
// It is restructured into its own small SCALAR pass that visits ONLY
// lattice positions (§A.14 "side quest not taken" — noted, not pursued,
// per the phase-4 addendum's single-lever discipline) — this is still
// strictly cheaper than the OLD dense-scalar-with-modulo-branch version it
// replaces, since it no longer visits all `width*height` pixels to find
// the ~1/8+1/8 fraction that matter.
//
// SSIM raw-moment reformulation (deliberate v2-only numeric shift,
// documented per this file's own 5e-4-tolerance allowance): Terriberry's
// online update is inherently sequential (each sample's update depends on
// the running n), so it does not vectorize across lanes. Instead, each
// lane accumulates plain running sums of d, d^2, d^3, d^4 (trivially
// SIMD — no cross-lane dependency), reduced ONCE PER ROW (not once per
// whole image — bounds the f32 accumulator's magnitude to ~width/8
// increments of a small [0,2]-bounded value, avoiding "large-swallows-
// small" f32 summation error) into a running f64 total. At the very end,
// the standard raw-to-central moment identities recover (mean, M2, M4):
//   mean = Sum_d / n
//   M2/n = Sum_d2/n - mean^2
//   M4/n = Sum_d4/n - 4*mean*(Sum_d3/n) + 6*mean^2*(Sum_d2/n) - 3*mean^4
// Matches Terriberry's OWN output up to floating-point reassociation
// (verified within 5e-4 relative on the fixture pairs — see the phase-4
// test `simd_matches_scalar_within_tolerance`).
// ============================================================================

/// Per-row-reduced f64 accumulator for the dense (always-on) block.
#[derive(Default, Clone, Copy)]
pub(crate) struct DenseAccum {
    pub(crate) central: OnlineMoments,
    pub(crate) sum_d: f64,
    pub(crate) sum_d2: f64,
    pub(crate) sum_d3: f64,
    pub(crate) sum_d4: f64,
    pub(crate) sum_art: f64,
    pub(crate) sum_det: f64,
    pub(crate) sum_mse: f64,
    pub(crate) sum_hf_gain: f64,
    pub(crate) sum_hf_loss: f64,
    pub(crate) sum_hf_mag_loss: f64,
    pub(crate) sum_pjnd: f64,
    pub(crate) sum_pjnd_lo: f64,
    pub(crate) sum_pjnd_hi: f64,
    pub(crate) ws_peak_ssim: WeightedSum,
    pub(crate) ws_peak_art: WeightedSum,
    pub(crate) ws_peak_det: WeightedSum,
    pub(crate) ws_mask_ssim: WeightedSum,
    pub(crate) ws_mask_art: WeightedSum,
    pub(crate) ws_mask_det: WeightedSum,
    pub(crate) ws_mask_mse: WeightedSum,
    pub(crate) ws_iw_ssim: WeightedSum,
    pub(crate) ws_iw_art: WeightedSum,
    pub(crate) ws_iw_det: WeightedSum,
    pub(crate) ws_iw_mse: WeightedSum,
}

impl DenseAccum {
    /// Phase-5 (§A.15): fold one strip's `DenseAccum` into the running
    /// whole-image total. Every field is a plain sum (or a `WeightedSum`,
    /// itself two plain sums), so summing N strips' partials is exactly
    /// equivalent to accumulating over the whole image in one pass —
    /// strip order does not change WHICH additions happen, only their
    /// grouping (associativity-level float reassociation only, same class
    /// already tolerated for the phase-4 SIMD-lane-then-row reduction).
    #[inline]
    pub(crate) fn accumulate(&mut self, other: &DenseAccum) {
        self.central.merge(other.central);
        self.sum_d += other.sum_d;
        self.sum_d2 += other.sum_d2;
        self.sum_d3 += other.sum_d3;
        self.sum_d4 += other.sum_d4;
        self.sum_art += other.sum_art;
        self.sum_det += other.sum_det;
        self.sum_mse += other.sum_mse;
        self.sum_hf_gain += other.sum_hf_gain;
        self.sum_hf_loss += other.sum_hf_loss;
        self.sum_hf_mag_loss += other.sum_hf_mag_loss;
        self.sum_pjnd += other.sum_pjnd;
        self.sum_pjnd_lo += other.sum_pjnd_lo;
        self.sum_pjnd_hi += other.sum_pjnd_hi;
        self.ws_peak_ssim.accumulate(&other.ws_peak_ssim);
        self.ws_peak_art.accumulate(&other.ws_peak_art);
        self.ws_peak_det.accumulate(&other.ws_peak_det);
        self.ws_mask_ssim.accumulate(&other.ws_mask_ssim);
        self.ws_mask_art.accumulate(&other.ws_mask_art);
        self.ws_mask_det.accumulate(&other.ws_mask_det);
        self.ws_mask_mse.accumulate(&other.ws_mask_mse);
        self.ws_iw_ssim.accumulate(&other.ws_iw_ssim);
        self.ws_iw_art.accumulate(&other.ws_iw_art);
        self.ws_iw_det.accumulate(&other.ws_iw_det);
        self.ws_iw_mse.accumulate(&other.ws_iw_mse);
    }
}

#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_at_revision(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    revision: FormulaRevision,
) -> DenseAccum {
    // Revisions 5 and 7 run the 16-lane canonical body (revision 7 with row
    // lanes); revision 1 keeps the era-2 production body.
    if crate::featcanon::mode(revision).active() {
        let acc = if revision.row_lanes() {
            incant!(
                dense_block_kernel_canon64_vec::<crate::featcanon::Lanes16F64, true>(
                src,
                dst,
                mu1,
                mu2,
                ssq,
                s12,
                activity,
                width,
                height,
                transducer_bank,
                revision,
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        } else {
            incant!(
                dense_block_kernel_canon64_vec::<crate::featcanon::Lanes16F64, false>(
                src,
                dst,
                mu1,
                mu2,
                ssq,
                s12,
                activity,
                width,
                height,
                transducer_bank,
                revision,
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        };
        return acc;
    }
    dense_block_kernel_era2(
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
    )
}

/// Per-row-reduced f64 accumulator for the gradient block.
#[derive(Default, Clone, Copy)]
pub(crate) struct GradientAccum {
    pub(crate) sum_gms: f64,
    pub(crate) sum_gms2: f64,
    pub(crate) sum_ringing: f64,
    pub(crate) sum_banding: f64,
    pub(crate) sum_grad_src: f64,
    pub(crate) sum_grad_dst: f64,
    sum_bv_gain: f64,
    sum_bv_loss: f64,
}

impl GradientAccum {
    /// Phase-5 (§A.15): fold one strip's `GradientAccum` into the running
    /// whole-image total — same reasoning as `DenseAccum::accumulate`.
    #[inline]
    pub(crate) fn accumulate(&mut self, other: &GradientAccum) {
        self.sum_gms += other.sum_gms;
        self.sum_gms2 += other.sum_gms2;
        self.sum_ringing += other.sum_ringing;
        self.sum_banding += other.sum_banding;
        self.sum_grad_src += other.sum_grad_src;
        self.sum_grad_dst += other.sum_grad_dst;
        self.sum_bv_gain += other.sum_bv_gain;
        self.sum_bv_loss += other.sum_bv_loss;
    }
}

#[inline]
fn gradient_block_kernel_generic<T: F32x8Backend + Copy>(
    token: T,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    bv_delta_lo: f32,
    bv_delta_hi: f32,
) -> GradientAccum {
    // Only revision 1 reaches this kernel; revisions 5 and 7 use the canonical kernels.
    debug_assert_eq!(src_h.len(), width * (height + 2));
    debug_assert_eq!(dst_h.len(), width * (height + 2));
    debug_assert_eq!(activity.len(), width * height);

    let zero = V8::<T>::zero_t(token);
    let one = V8::<T>::splat_t(token, 1.0);
    let c_gms = V8::<T>::splat_t(token, C_GMS as f32);
    let c_ring_err = V8::<T>::splat_t(token, C_RING_ERR as f32);
    let c_activity = V8::<T>::splat_t(token, C_ACTIVITY as f32);
    let c_ring_edge = V8::<T>::splat_t(token, C_RING_EDGE as f32);
    let c_band_dst = V8::<T>::splat_t(token, C_BAND_DST as f32);
    let c_band_src = V8::<T>::splat_t(token, C_BAND_SRC as f32);
    let _bv_lo = V8::<T>::splat_t(token, bv_delta_lo);
    let _bv_hi = V8::<T>::splat_t(token, bv_delta_hi);
    let _c_bv = V8::<T>::splat_t(token, C_BV as f32);

    let mut acc = GradientAccum::default();

    // Scalar helper for one pixel — used for the first/last COLUMN of
    // every row (x-axis boundary only; the y-axis "first/last row"
    // special case from phase 4 is GONE, per this function's new halo
    // contract — every row `y` has valid `row_u`/`row_d` neighbor data
    // in `src_h`/`dst_h` by construction, either real interior-strip
    // rows or a caller-supplied reflected true-edge row).
    let scalar_pixel = |x: usize,
                        y: usize,
                        acc: &mut GradientAccum| {
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(width - 1);
        let row = (y + 1) * width; // +1: src_h/dst_h carry 1 halo row up front
        let row_u = y * width; // y-1, in halo-relative terms
        let row_d = (y + 2) * width; // y+1, in halo-relative terms
        let i = row + x;
        let s = src_h[i] as f64;
        let dd = dst_h[i] as f64;
        let act = activity[y * width + x] as f64;
        let sxl = src_h[row + xl] as f64;
        let sxr = src_h[row + xr] as f64;
        let syu = src_h[row_u + x] as f64;
        let syd = src_h[row_d + x] as f64;
        let dxl = dst_h[row + xl] as f64;
        let dxr = dst_h[row + xr] as f64;
        let dyu = dst_h[row_u + x] as f64;
        let dyd = dst_h[row_d + x] as f64;

        let gx_src = sxr - sxl;
        let gy_src = syd - syu;
        let grad_src_mag = (gx_src * gx_src + gy_src * gy_src).sqrt();
        let gx_dst = dxr - dxl;
        let gy_dst = dyd - dyu;
        let grad_dst_mag = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

        acc.sum_grad_src += grad_src_mag;
        acc.sum_grad_dst += grad_dst_mag;
        let g = bounded_dissim(grad_src_mag, grad_dst_mag, C_GMS, false);
        acc.sum_gms += g;
        acc.sum_gms2 += g * g;

        let raw_abs_err = (s - dd).abs();
        let err_b = saturate(raw_abs_err, C_RING_ERR);
        let act_b = saturate(act, C_ACTIVITY);
        let edge_r = saturate(grad_src_mag, C_RING_EDGE);
        let ring_i = err_b * act_b * (1.0 - edge_r);
        acc.sum_ringing += ring_i;

        let edge_excess = bounded_excess(grad_dst_mag, grad_src_mag, C_BAND_DST);
        let src_smooth_b = 1.0 - saturate(grad_src_mag, C_BAND_SRC);
        acc.sum_banding += edge_excess * src_smooth_b;

    };

    for y in 0..height {

        let row = (y + 1) * width;
        let row_u = y * width;
        let row_d = (y + 2) * width;
        let act_row = y * width;

        scalar_pixel(0, y, &mut acc);
        if width > 2 {
            let interior_end = width - 1;
            let interior_w = interior_end - 1; // pixels [1, width-2]
            let chunk_end = 1 + interior_w - (interior_w % 8);

            let (mut r_gms, mut r_gms2, mut r_ring, mut r_band) = (zero, zero, zero, zero);
            let (mut r_gsrc, mut r_gdst) = (zero, zero);
            let (mut _r_bv_gain, mut _r_bv_loss) = (zero, zero);

            let mut x = 1usize;
            while x < chunk_end {
                macro_rules! ld_at {
                    ($plane:expr, $off:expr) => {
                        V8::<T>::from_array_t(token, $plane[$off..$off + 8].try_into().unwrap())
                    };
                }
                let sxl = ld_at!(src_h, row + x - 1);
                let sxr = ld_at!(src_h, row + x + 1);
                let syu = ld_at!(src_h, row_u + x);
                let syd = ld_at!(src_h, row_d + x);
                let dxl = ld_at!(dst_h, row + x - 1);
                let dxr = ld_at!(dst_h, row + x + 1);
                let dyu = ld_at!(dst_h, row_u + x);
                let dyd = ld_at!(dst_h, row_d + x);
                let s = ld_at!(src_h, row + x);
                let dd = ld_at!(dst_h, row + x);
                let act = ld_at!(activity, act_row + x);

                let gx_src = sxr - sxl;
                let gy_src = syd - syu;
                let grad_src_mag = (gx_src * gx_src + gy_src * gy_src).sqrt();
                let gx_dst = dxr - dxl;
                let gy_dst = dyd - dyu;
                let grad_dst_mag = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

                r_gsrc += grad_src_mag;
                r_gdst += grad_dst_mag;
                let g = bounded_dissim_v(token, grad_src_mag, grad_dst_mag, c_gms, false);
                r_gms += g;
                r_gms2 += g * g;

                let raw_abs_err = (s - dd).abs();
                let err_b = saturate_v(token, raw_abs_err, c_ring_err);
                let act_b = saturate_v(token, act, c_activity);
                let edge_r = saturate_v(token, grad_src_mag, c_ring_edge);
                let ring_i = err_b * act_b * (one - edge_r);
                r_ring += ring_i;

                let edge_excess = bounded_excess_v(token, grad_dst_mag, grad_src_mag, c_band_dst);
                let src_smooth_b = one - saturate_v(token, grad_src_mag, c_band_src);
                r_band += edge_excess * src_smooth_b;

                x += 8;
            }

            acc.sum_gms += r_gms.reduce_add() as f64;
            acc.sum_gms2 += r_gms2.reduce_add() as f64;
            acc.sum_ringing += r_ring.reduce_add() as f64;
            acc.sum_banding += r_band.reduce_add() as f64;
            acc.sum_grad_src += r_gsrc.reduce_add() as f64;
            acc.sum_grad_dst += r_gdst.reduce_add() as f64;

            for x in chunk_end..=interior_end - 1 {
                scalar_pixel(x, y, &mut acc);
            }
        }
        scalar_pixel(
            width - 1,
            y,
            &mut acc,
        );
    }

    acc
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn gradient_block_kernel_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
) -> GradientAccum {
    gradient_block_kernel_generic::<_>(
        token,
        src,
        dst,
        activity,
        width,
        height,
        0.0,
        0.0,
    )
}

/// One gradient pixel's full term vector in f64 — the production
/// `scalar_pixel` element math verbatim (clamped x-neighbors, so it serves
/// both the border columns and the exact interior).
pub(crate) struct GradTerms64 {
    pub(crate) gsrc: f64,
    pub(crate) gdst: f64,
    pub(crate) g: f64,
    pub(crate) g2: f64,
    pub(crate) ring: f64,
    pub(crate) band: f64,
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn gradient_terms64(
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    width: usize,
    x: usize,
    y: usize,
    stable: bool,
) -> GradTerms64 {
    let xl = x.saturating_sub(1);
    let xr = (x + 1).min(width - 1);
    let row = (y + 1) * width;
    let row_u = y * width;
    let row_d = (y + 2) * width;
    let i = row + x;
    let s = src_h[i] as f64;
    let dd = dst_h[i] as f64;
    let act = activity[y * width + x] as f64;
    let sxl = src_h[row + xl] as f64;
    let sxr = src_h[row + xr] as f64;
    let syu = src_h[row_u + x] as f64;
    let syd = src_h[row_d + x] as f64;
    let dxl = dst_h[row + xl] as f64;
    let dxr = dst_h[row + xr] as f64;
    let dyu = dst_h[row_u + x] as f64;
    let dyd = dst_h[row_d + x] as f64;

    let gx_src = sxr - sxl;
    let gy_src = syd - syu;
    let gsrc = (gx_src * gx_src + gy_src * gy_src).sqrt();
    let gx_dst = dxr - dxl;
    let gy_dst = dyd - dyu;
    let gdst = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();
    let g = bounded_dissim(gsrc, gdst, C_GMS, stable);

    let raw_abs_err = (s - dd).abs();
    let err_b = saturate(raw_abs_err, C_RING_ERR);
    let act_b = saturate(act, C_ACTIVITY);
    let edge_r = saturate(gsrc, C_RING_EDGE);
    let ring = err_b * act_b * (1.0 - edge_r);

    let edge_excess = bounded_excess(gdst, gsrc, C_BAND_DST);
    let src_smooth_b = 1.0 - saturate(gsrc, C_BAND_SRC);
    let band = edge_excess * src_smooth_b;

    let (_bv_gain, _bv_loss) = (0.0, 0.0);

    GradTerms64 {
        gsrc,
        gdst,
        g,
        g2: g * g,
        ring,
        band,
    }
}

/// f32 scalar helpers mirroring the `*_v` SIMD ops' exact op order.
#[inline(always)]
fn sat32(x: f32, c: f32) -> f32 {
    let x = x.max(0.0);
    x / (x + c)
}
#[inline(always)]
fn bsim32(a: f32, b: f32, c: f32) -> f32 {
    (2.0f32 * a * b + c) / (a * a + b * b + c)
}
#[inline(always)]
fn bex32(a: f32, b: f32, c: f32) -> f32 {
    (a - b).max(0.0) / (a + b + c)
}

#[allow(clippy::too_many_arguments)]
fn gradient_border_pixel<
    const R7: bool,
>(
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    _act_dst: &[f32],
    _mu1: &[f32],
    _mu2: &[f32],
    width: usize,
    x: usize,
    y: usize,
    _bv_lo: f32,
    _bv_hi: f32,
    acc: &mut GradientAccum,
) {
    let t = gradient_terms64(
        src_h,
        dst_h,
        activity,
        width,
        x,
        y,
        R7,
    );
    acc.sum_grad_src += t.gsrc;
    acc.sum_grad_dst += t.gdst;
    acc.sum_gms += t.g;
    acc.sum_gms2 += t.g2;
    acc.sum_ringing += t.ring;
    acc.sum_banding += t.band;
}

/// One interior pixel's contribution to the gradient canonical pools —
/// the scalar element step shared by [`gradient_block_kernel_canon`]'s
/// `Pool`-generic body and the canon64 vector body's lane-1..8 head and
/// `interior_end % 8` tail. Extracted verbatim from the canonical body.
#[allow(clippy::too_many_arguments)]
fn gradient_interior_elem_canon<
    P: crate::featcanon::Pool,
    // The pools revision 7 keeps in f32 row lanes (`P` otherwise).
    Q: crate::featcanon::Pool,
    const R7: bool,
>(
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    _act_dst: &[f32],
    _mu1: &[f32],
    _mu2: &[f32],
    _width: usize,
    row: usize,
    row_u: usize,
    row_d: usize,
    act_row: usize,
    x: usize,
    _bv_lo: f32,
    _bv_hi: f32,
    r_gsrc: &mut Q,
    r_gdst: &mut Q,
    r_gms: &mut Q,
    r_gms2: &mut Q,
    r_ring: &mut Q,
    r_band: &mut Q,
    _r_bv_gain: &mut Q,
    _r_bv_loss: &mut Q,
    // Revision 7's f32 row lanes: `r_gdst` takes `gdst − gsrc` (exact when
    // the two are within a factor of 2), and the fold adds `r_gsrc`'s lanes to
    // both sums, so the edge-width decay sees the difference summed directly.
    gdiff: bool,
) {
    let lane = x & (P::LANES - 1);
    let sxl = src_h[row + x - 1];
    let sxr = src_h[row + x + 1];
    let syu = src_h[row_u + x];
    let syd = src_h[row_d + x];
    let dxl = dst_h[row + x - 1];
    let dxr = dst_h[row + x + 1];
    let dyu = dst_h[row_u + x];
    let dyd = dst_h[row_d + x];
    let s = src_h[row + x];
    let dd = dst_h[row + x];
    let act = activity[act_row + x];

    let gx_src = sxr - sxl;
    let gy_src = syd - syu;
    let gsrc = (gx_src * gx_src + gy_src * gy_src).sqrt();
    let gx_dst = dxr - dxl;
    let gy_dst = dyd - dyu;
    let gdst = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

    r_gsrc.add(lane, gsrc);
    r_gdst.add(lane, if gdiff { gdst - gsrc } else { gdst });
    // Revision 7: `bounded_dissim_v`'s exact `(a−b)²/(a²+b²+c)`, same op order.
    let g = if R7 {
        crate::feature_v2::bounded_dissim32(gsrc, gdst, C_GMS as f32, true)
    } else {
        1.0f32 - bsim32(gsrc, gdst, C_GMS as f32)
    };
    r_gms.add(lane, g);
    r_gms2.add(lane, g * g);

    let raw_abs_err = (s - dd).abs();
    let err_b = sat32(raw_abs_err, C_RING_ERR as f32);
    let act_b = sat32(act, C_ACTIVITY as f32);
    let edge_r = sat32(gsrc, C_RING_EDGE as f32);
    let ring_i = err_b * act_b * (1.0f32 - edge_r);
    r_ring.add(lane, ring_i);

    let edge_excess = bex32(gdst, gsrc, C_BAND_DST as f32);
    let src_smooth_b = 1.0f32 - sat32(gsrc, C_BAND_SRC as f32);
    r_band.add(lane, edge_excess * src_smooth_b);

}

/// Add the gradient kernel's eight interior pools' finished sums to `acc`
/// (`[gms, gms2, ring, band, gsrc, gdst, bv_gain, bv_loss]`), in the fixed
/// order: once per row, or at revision 7 once per call.
#[inline(always)]
fn gradient_pools_finish<P: crate::featcanon::Pool>(
    acc: &mut GradientAccum,
    pools: &[P; 8],
) {
    let [gms, gms2, ring, band, gsrc, gdst, _bv_gain, _bv_loss] = *pools;
    acc.sum_gms += gms.fin();
    acc.sum_gms2 += gms2.fin();
    acc.sum_ringing += ring.fin();
    acc.sum_banding += band.fin();
    acc.sum_grad_src += gsrc.fin();
    acc.sum_grad_dst += gdst.fin();
}

/// `Pool`-generic scalar gradient kernel — f32 elements in the SIMD op
/// order, per-row canonical pools, f64 borders.
#[allow(clippy::too_many_arguments)]
fn gradient_block_kernel_canon<
    P: crate::featcanon::Pool,
    const R7: bool,
>(
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    height: usize,
    bv_lo: f32,
    bv_hi: f32,
    _mode: crate::featcanon::Mode,
) -> GradientAccum {
    let mut acc = GradientAccum::default();

    // Revision 7: the interior pools' f64 lanes live for the whole call (each
    // row's f32 row lanes fold into them) and finish once, after the last row.
    let mut call_pools = [P::zero(); 8];
    for y in 0..height {
        let row = (y + 1) * width;
        let row_u = y * width;
        let row_d = (y + 2) * width;
        let act_row = y * width;

        gradient_border_pixel::<R7>(
            src_h,
            dst_h,
            activity,
            act_dst,
            mu1,
            mu2,
            width,
            0,
            y,
            bv_lo,
            bv_hi,
            &mut acc,
        );
        if width > 2 {
            let interior_end = width - 1;
            // `[gms, gms2, ring, band, gsrc, gdst, bv_gain, bv_loss]`.
            let mut fresh = [P::zero(); 8];
            let [
                r_gms,
                r_gms2,
                r_ring,
                r_band,
                r_gsrc,
                r_gdst,
                r_bv_gain,
                r_bv_loss,
            ] = if R7 { &mut call_pools } else { &mut fresh };
            // Revision 7: the vector kernel's chunk columns (`head_end` to
            // `chunk_end`) add the six non-cancelling pools in f32 row lanes
            // (`crate::featcanon::Lanes16F32`), folded every `ROW_LANES_FOLD`
            // columns and at the end of the chunks; head and tail stay f64.
            let head_end = interior_end.min(8);
            let chunk_end = head_end + (interior_end - head_end) / 8 * 8;
            let mut q32 = [crate::featcanon::Lanes16F32::default(); 8];
            for x in 1..interior_end {
                if R7 && (head_end..chunk_end).contains(&x) {
                    let [q_gms, q_gms2, q_ring, q_band, q_bvg, q_bvl, q_gsrc, q_gdst] = &mut q32;
                    gradient_interior_elem_canon::<
                        P,
                        crate::featcanon::Lanes16F32,
                        R7,
                    >(
                        src_h,
                        dst_h,
                        activity,
                        act_dst,
                        mu1,
                        mu2,
                        width,
                        row,
                        row_u,
                        row_d,
                        act_row,
                        x,
                        bv_lo,
                        bv_hi,
                        q_gsrc,
                        q_gdst,
                        q_gms,
                        q_gms2,
                        q_ring,
                        q_band,
                        q_bvg,
                        q_bvl,
                        true,
                    );
                    if (x + 1).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) || x + 1 == chunk_end {
                        let [q_gms, q_gms2, q_ring, q_band, q_bvg, q_bvl, q_gsrc, q_gdst] = &mut q32;
                        q_gms.fold_into(&mut *r_gms);
                        q_gms2.fold_into(&mut *r_gms2);
                        q_ring.fold_into(&mut *r_ring);
                        q_band.fold_into(&mut *r_band);
                        q_bvg.fold_into(&mut *r_bv_gain);
                        q_bvl.fold_into(&mut *r_bv_loss);
                        // The difference pair: the source lanes to both sums.
                        q_gsrc.add_into(&mut *r_gsrc);
                        q_gsrc.fold_into(&mut *r_gdst);
                        q_gdst.fold_into(&mut *r_gdst);
                    }
                    continue;
                }
                gradient_interior_elem_canon::<P, P, R7>(
                    src_h,
                    dst_h,
                    activity,
                    act_dst,
                    mu1,
                    mu2,
                    width,
                    row,
                    row_u,
                    row_d,
                    act_row,
                    x,
                    bv_lo,
                    bv_hi,
                    &mut *r_gsrc,
                    &mut *r_gdst,
                    &mut *r_gms,
                    &mut *r_gms2,
                    &mut *r_ring,
                    &mut *r_band,
                    &mut *r_bv_gain,
                    &mut *r_bv_loss,
                    false,
                );
            }
            if !R7 {
                gradient_pools_finish::<P>(&mut acc, &fresh);
            }
        }
        gradient_border_pixel::<R7>(
            src_h,
            dst_h,
            activity,
            act_dst,
            mu1,
            mu2,
            width,
            width - 1,
            y,
            bv_lo,
            bv_hi,
            &mut acc,
        );
    }
    if R7 && width > 2 {
        gradient_pools_finish::<P>(&mut acc, &call_pools);
    }
    acc
}

#[inline(always)]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn gradient_terms32_v<V: Vf32>(
    token: V::Tok,
    sxl: V,
    sxr: V,
    syu: V,
    syd: V,
    dxl: V,
    dxr: V,
    dyu: V,
    dyd: V,
    s: V,
    dd: V,
    act: V,
    bv_lo: f32,
    bv_hi: f32,
    stable: bool,
) -> (V, V, V, V, V, V, V, V) {
    let c_gms = V::splat_t(token, C_GMS as f32);
    let c_ring_err = V::splat_t(token, C_RING_ERR as f32);
    let c_act = V::splat_t(token, C_ACTIVITY as f32);
    let c_ring_edge = V::splat_t(token, C_RING_EDGE as f32);
    let c_band_dst = V::splat_t(token, C_BAND_DST as f32);
    let c_band_src = V::splat_t(token, C_BAND_SRC as f32);
    let _c_bv = V::splat_t(token, C_BV as f32);
    let _bv_lo_v = V::splat_t(token, bv_lo);
    let _bv_hi_v = V::splat_t(token, bv_hi);
    let one = V::splat_t(token, 1.0);
    let zero = V::zero_t(token);

    let gx_src = sxr - sxl;
    let gy_src = syd - syu;
    let gsrc = (gx_src * gx_src + gy_src * gy_src).sqrt();
    let gx_dst = dxr - dxl;
    let gy_dst = dyd - dyu;
    let gdst = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

    let g = bounded_dissim_v(token, gsrc, gdst, c_gms, stable);
    let g2 = g * g;

    let raw_abs_err = (s - dd).abs();
    let err_b = saturate_v(token, raw_abs_err, c_ring_err);
    let act_b = saturate_v(token, act, c_act);
    let edge_r = saturate_v(token, gsrc, c_ring_edge);
    let ring_i = err_b * act_b * (one - edge_r);

    let edge_excess = bounded_excess_v(token, gdst, gsrc, c_band_dst);
    let src_smooth_b = one - saturate_v(token, gsrc, c_band_src);
    let band = edge_excess * src_smooth_b;

    let (bv_gain, bv_loss) = (zero, zero);

    (gsrc, gdst, g, g2, ring_i, band, bv_gain, bv_loss)
}

#[magetypes(define(f32x8, f32x16), v4x, v4, v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn gradient_block_kernel_canon64_vec<
    P: crate::featcanon::Pool64,
    const R7: bool,
>(
    token: Token,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    height: usize,
    bv_lo: f32,
    bv_hi: f32,
    _mode: crate::featcanon::Mode,
) -> GradientAccum {
    let mut acc = GradientAccum::default();

    // Revision 7 (`rowlanes`): the chunks' adds of the six non-cancelling
    // pools go to f32 row lanes, folded into the f64 lanes every
    // `ROW_LANES_FOLD` columns and at the end of the chunks (the scalar body's
    // `head_end..chunk_end` region). Revisions 5 and 6 keep their bits.
    let rev7 = R7;
    macro_rules! gpool_add {
        ($pool:ident, $pool32:ident, $x0:expr, $v:expr) => {{
            let v: f32x8 = $v;
            if rev7 {
                // `$x0` is a multiple of 8: its bit 3 picks the lane half.
                let lanes = &mut $pool32.as_chunks_mut::<8>().0[($x0 >> 3) & 1];
                (f32x8::load_t(token, lanes) + v).store(lanes);
            } else {
                let a = v.to_array();
                let lanes = $pool.chunk8($x0 & (P::LANES - 1));
                for l in 0..8 {
                    lanes[l] += a[l] as f64;
                }
            }
        }};
    }
    // The gradient sums feed a ratio (edge-width change) and keep f64 lanes.
    macro_rules! gpool_add64 {
        ($pool:ident, $x0:expr, $v:expr) => {{
            let a = ($v).to_array();
            let lanes = $pool.chunk8($x0 & (P::LANES - 1));
            for l in 0..8 {
                lanes[l] += a[l] as f64;
            }
        }};
    }
    // `Lanes16F32::fold_into`'s op per lane (`lane += v as f64`); `clear`
    // empties the f32 lanes for the row's next columns.
    macro_rules! gpool_fold {
        (clear; $($pool:ident, $pool32:ident);*) => {{
            $(
                gpool_fold!($pool, $pool32);
                $pool32 = [0.0f32; 16];
            )*
        }};
        (last; $($pool:ident, $pool32:ident);*) => {{
            $(
                gpool_fold!($pool, $pool32);
            )*
        }};
        // The difference pair: the source lanes to both sums, then the
        // difference lanes to the distorted sum.
        (pair; $src:ident, $src32:ident, $dst:ident, $dst32:ident) => {{
            gpool_fold!($src, $src32);
            gpool_fold!($dst, $src32);
            gpool_fold!($dst, $dst32);
        }};
        ($pool:ident, $pool32:ident) => {{
            for off in [0usize, 8] {
                let lanes = $pool.chunk8(off);
                for l in 0..8 {
                    lanes[l] += $pool32[off + l] as f64;
                }
            }
        }};
    }
    // Revision 7: the interior pools' f64 lanes live for the whole call (each
    // row's f32 row lanes fold into them) and finish once, after the last row.
    let mut call_pools = [P::zero(); 8];
    for y in 0..height {
        let row = (y + 1) * width;
        let row_u = y * width;
        let row_d = (y + 2) * width;
        let act_row = y * width;

        gradient_border_pixel::<R7>(
            src_h,
            dst_h,
            activity,
            act_dst,
            mu1,
            mu2,
            width,
            0,
            y,
            bv_lo,
            bv_hi,
            &mut acc,
        );
        if width > 2 {
            let interior_end = width - 1;
            // `[gms, gms2, ring, band, gsrc, gdst, bv_gain, bv_loss]`.
            let mut fresh = [P::zero(); 8];
            let [
                r_gms,
                r_gms2,
                r_ring,
                r_band,
                r_gsrc,
                r_gdst,
                r_bv_gain,
                r_bv_loss,
            ] = if R7 { &mut call_pools } else { &mut fresh };
            let (mut r_gms32, mut r_gms232, mut r_ring32, mut r_band32) =
                ([0.0f32; 16], [0.0f32; 16], [0.0f32; 16], [0.0f32; 16]);
            let (mut r_bv_gain32, mut r_bv_loss32) = ([0.0f32; 16], [0.0f32; 16]);
            // Revision 7's difference pair: `gsrc` and `gdst − gsrc`.
            let (mut r_gsrc32, mut r_gdst32) = ([0.0f32; 16], [0.0f32; 16]);
            // Lane-aligned head: x = 1..8 (interior starts one pixel in — the
            // canonical lanes land on chunk lanes only for x ≡ 0 (mod 8) chunk
            // starts). Head and tail add in the f64 lanes; without the bank
            // paths and in rows of at least ten columns they are evaluated as
            // one eight-wide step each (`edge8!`), whose lanes are the scalar
            // element's values, added in ascending x.
            let head_end = interior_end.min(8);
            let vec_edges = interior_end >= 9;
            macro_rules! edge8 {
                ($xs:expr, $from:expr, $to:expr) => {{
                    let xs: usize = $xs;
                    macro_rules! ld8 {
                        ($pl:expr, $o:expr) => {{
                            let o: usize = $o;
                            f32x8::load_t(token, $pl[o..o + 8].try_into().unwrap())
                        }};
                    }
                    let _actd = f32x8::zero_t(token);
                    let (_m1, _m2) = (f32x8::zero_t(token), f32x8::zero_t(token));
                    let (gsrc, gdst, g, g2, ring_i, band, bv_gain, bv_loss) =
                        gradient_terms32_v::<_>(
                            token,
                            ld8!(src_h, row + xs - 1),
                            ld8!(src_h, row + xs + 1),
                            ld8!(src_h, row_u + xs),
                            ld8!(src_h, row_d + xs),
                            ld8!(dst_h, row + xs - 1),
                            ld8!(dst_h, row + xs + 1),
                            ld8!(dst_h, row_u + xs),
                            ld8!(dst_h, row_d + xs),
                            ld8!(src_h, row + xs),
                            ld8!(dst_h, row + xs),
                            ld8!(activity, act_row + xs),
                            bv_lo,
                            bv_hi,
                            R7,
                        );
                    let (gs, gd, ga, gb) = (gsrc.to_array(), gdst.to_array(), g.to_array(), g2.to_array());
                    let (ri, ba) = (ring_i.to_array(), band.to_array());
                    let (_bg, _bl) = (bv_gain.to_array(), bv_loss.to_array());
                    for x in $from..$to {
                        let (l, lane) = (x - xs, x & (P::LANES - 1));
                        r_gsrc.add(lane, gs[l]);
                        r_gdst.add(lane, gd[l]);
                        r_gms.add(lane, ga[l]);
                        r_gms2.add(lane, gb[l]);
                        r_ring.add(lane, ri[l]);
                        r_band.add(lane, ba[l]);
                    }
                }};
            }
            if vec_edges {
                edge8!(1, 1, head_end);
            }
            for x in 1..if vec_edges { 1 } else { head_end } {
                gradient_interior_elem_canon::<P, P, R7>(
                    src_h,
                    dst_h,
                    activity,
                    act_dst,
                    mu1,
                    mu2,
                    width,
                    row,
                    row_u,
                    row_d,
                    act_row,
                    x,
                    bv_lo,
                    bv_hi,
                    &mut *r_gsrc,
                    &mut *r_gdst,
                    &mut *r_gms,
                    &mut *r_gms2,
                    &mut *r_ring,
                    &mut *r_band,
                    &mut *r_bv_gain,
                    &mut *r_bv_loss,
                    false,
                );
            }
            // The chunks, x0 = head_end + 8k for k < n. Every plane's window
            // is cut to its exact span first, so the loads need no per-chunk
            // range checks (they were most of this loop's instructions).
            let n = (interior_end - head_end) / 8;
            let span = n * 8;
            let c0 = row + head_end;
            let a0 = act_row + head_end;
            macro_rules! win {
                ($pl:expr, $start:expr) => {{
                    let w: &[[f32; 8]] = $pl[$start..$start + span].as_chunks::<8>().0;
                    &w[..n]
                }};
            }
            let (w_sxl, w_sxr, w_syu, w_syd, w_s) = (
                win!(src_h, c0 - 1),
                win!(src_h, c0 + 1),
                win!(src_h, row_u + head_end),
                win!(src_h, row_d + head_end),
                win!(src_h, c0),
            );
            let (w_dxl, w_dxr, w_dyu, w_dyd, w_d) = (
                win!(dst_h, c0 - 1),
                win!(dst_h, c0 + 1),
                win!(dst_h, row_u + head_end),
                win!(dst_h, row_d + head_end),
                win!(dst_h, c0),
            );
            let w_act = win!(activity, a0);
            let (_w_m1, _w_m2): (&[[f32; 8]], &[[f32; 8]]) =
                { (&[], &[]) };
            let wide = rev7 && steps_sixteen::<Token>() && n >= 3;
            let m16 = if wide { (n - 1) / 2 } else { 0 };
            let mut k = 0usize;
            while k < n {
                if k == 1 && m16 > 0 {
                    let span16 = m16 * 16;
                    macro_rules! win16 {
                        ($pl:expr, $start:expr) => {{
                            let w: &[[f32; 16]] =
                                $pl[$start + 8..$start + 8 + span16].as_chunks::<16>().0;
                            &w[..m16]
                        }};
                    }
                    let (v_sxl, v_sxr, v_syu, v_syd, v_s) = (
                        win16!(src_h, c0 - 1),
                        win16!(src_h, c0 + 1),
                        win16!(src_h, row_u + head_end),
                        win16!(src_h, row_d + head_end),
                        win16!(src_h, c0),
                    );
                    let (v_dxl, v_dxr, v_dyu, v_dyd, v_d) = (
                        win16!(dst_h, c0 - 1),
                        win16!(dst_h, c0 + 1),
                        win16!(dst_h, row_u + head_end),
                        win16!(dst_h, row_d + head_end),
                        win16!(dst_h, c0),
                    );
                    let v_act = win16!(activity, a0);
                    let (_v_m1, _v_m2): (&[[f32; 16]], &[[f32; 16]]) =
                        { (&[], &[]) };
                    for j in 0..m16 {
                        let x0 = head_end + 8 + 16 * j;
                        macro_rules! ld {
                            ($w:expr) => {
                                f32x16::load_t(token, &$w[j])
                            };
                        }
                        let _actd = f32x16::zero_t(token);
                        let (_m1, _m2) = (f32x16::zero_t(token), f32x16::zero_t(token));
                        let (gsrc, gdst, g, g2, ring_i, band, _bv_gain, _bv_loss) =
                            gradient_terms32_v::<_>(
                                token,
                                ld!(v_sxl),
                                ld!(v_sxr),
                                ld!(v_syu),
                                ld!(v_syd),
                                ld!(v_dxl),
                                ld!(v_dxr),
                                ld!(v_dyu),
                                ld!(v_dyd),
                                ld!(v_s),
                                ld!(v_d),
                                ld!(v_act),
                                bv_lo,
                                bv_hi,
                                R7,
                            );
                        // f64 lanes: lanes 0..8 then 8..16, each lane one add.
                        macro_rules! add64 {
                            ($pool:ident, $v:expr) => {{
                                let a = ($v).to_array();
                                for (off, half) in [(0usize, &a[..8]), (8, &a[8..])] {
                                    let lanes = $pool.chunk8(off);
                                    for l in 0..8 {
                                        lanes[l] += half[l] as f64;
                                    }
                                }
                            }};
                        }
                        macro_rules! add {
                            ($pool:ident, $pool32:ident, $v:expr) => {{
                                if rev7 {
                                    (f32x16::load_t(token, &$pool32) + $v).store(&mut $pool32);
                                } else {
                                    add64!($pool, $v);
                                }
                            }};
                        }
                        add!(r_gsrc, r_gsrc32, gsrc);
                        add!(r_gdst, r_gdst32, gdst - gsrc);
                        add!(r_gms, r_gms32, g);
                        add!(r_gms2, r_gms232, g2);
                        add!(r_ring, r_ring32, ring_i);
                        add!(r_band, r_band32, band);
                        if rev7 && (x0 + 16).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                            gpool_fold!(
                                clear;
                                r_gms, r_gms32; r_gms2, r_gms232;
                                r_ring, r_ring32; r_band, r_band32; r_bv_gain, r_bv_gain32;
                                r_bv_loss, r_bv_loss32
                            );
                            gpool_fold!(pair; r_gsrc, r_gsrc32, r_gdst, r_gdst32);
                            r_gsrc32 = [0.0f32; 16];
                            r_gdst32 = [0.0f32; 16];
                        }
                    }
                    k = 1 + 2 * m16;
                    continue;
                }
                let x0 = head_end + 8 * k;
                let sxl = f32x8::load_t(token, &w_sxl[k]);
                let sxr = f32x8::load_t(token, &w_sxr[k]);
                let syu = f32x8::load_t(token, &w_syu[k]);
                let syd = f32x8::load_t(token, &w_syd[k]);
                let dxl = f32x8::load_t(token, &w_dxl[k]);
                let dxr = f32x8::load_t(token, &w_dxr[k]);
                let dyu = f32x8::load_t(token, &w_dyu[k]);
                let dyd = f32x8::load_t(token, &w_dyd[k]);
                let s = f32x8::load_t(token, &w_s[k]);
                let dd = f32x8::load_t(token, &w_d[k]);
                let act = f32x8::load_t(token, &w_act[k]);
                let _actd = f32x8::zero_t(token);
                let (_m1, _m2) = (f32x8::zero_t(token), f32x8::zero_t(token));
                let (gsrc, gdst, g, g2, ring_i, band, _bv_gain, _bv_loss) =
                    gradient_terms32_v::<_>(
                        token, sxl, sxr, syu, syd, dxl, dxr, dyu, dyd, s, dd, act,                           bv_lo, bv_hi, R7,
                    );

                if rev7 {
                    gpool_add!(r_gsrc, r_gsrc32, x0, gsrc);
                    gpool_add!(r_gdst, r_gdst32, x0, gdst - gsrc);
                } else {
                    gpool_add64!(r_gsrc, x0, gsrc);
                    gpool_add64!(r_gdst, x0, gdst);
                }
                gpool_add!(r_gms, r_gms32, x0, g);
                gpool_add!(r_gms2, r_gms232, x0, g2);
                gpool_add!(r_ring, r_ring32, x0, ring_i);
                gpool_add!(r_band, r_band32, x0, band);
                if rev7 && (x0 + 8).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                    gpool_fold!(
                        clear;
                        r_gms, r_gms32; r_gms2, r_gms232;
                        r_ring, r_ring32; r_band, r_band32; r_bv_gain, r_bv_gain32;
                        r_bv_loss, r_bv_loss32
                    );
                    gpool_fold!(pair; r_gsrc, r_gsrc32, r_gdst, r_gdst32);
                    r_gsrc32 = [0.0f32; 16];
                    r_gdst32 = [0.0f32; 16];
                }
                k += 1;
            }
            let x0 = head_end + span;
            // The end of the chunks, unless the last chunk already folded.
            if rev7 && n > 0 && !x0.is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                gpool_fold!(
                    last;
                    r_gms, r_gms32; r_gms2, r_gms232;
                    r_ring, r_ring32; r_band, r_band32; r_bv_gain, r_bv_gain32;
                    r_bv_loss, r_bv_loss32
                );
                gpool_fold!(pair; r_gsrc, r_gsrc32, r_gdst, r_gdst32);
            }
            // The tail: one eight-wide step ending at the interior's end, or
            // the canonical element step per column.
            if vec_edges && x0 < interior_end {
                edge8!(interior_end - 8, x0, interior_end);
            }
            for x in x0..if vec_edges { x0 } else { interior_end } {
                gradient_interior_elem_canon::<P, P, R7>(
                    src_h,
                    dst_h,
                    activity,
                    act_dst,
                    mu1,
                    mu2,
                    width,
                    row,
                    row_u,
                    row_d,
                    act_row,
                    x,
                    bv_lo,
                    bv_hi,
                    &mut *r_gsrc,
                    &mut *r_gdst,
                    &mut *r_gms,
                    &mut *r_gms2,
                    &mut *r_ring,
                    &mut *r_band,
                    &mut *r_bv_gain,
                    &mut *r_bv_loss,
                    false,
                );
            }
            if !R7 {
                gradient_pools_finish::<P>(&mut acc, &fresh);
            }
        }
        gradient_border_pixel::<R7>(
            src_h,
            dst_h,
            activity,
            act_dst,
            mu1,
            mu2,
            width,
            width - 1,
            y,
            bv_lo,
            bv_hi,
            &mut acc,
        );
    }
    if R7 && width > 2 {
        gradient_pools_finish::<P>(&mut acc, &call_pools);
    }
    acc
}

/// Scalar-tier sibling of [`gradient_block_kernel_canon64_vec`].
#[allow(clippy::too_many_arguments)]
fn gradient_block_kernel_canon64_vec_scalar<
    P: crate::featcanon::Pool64,
    const R7: bool,
>(
    _token: archmage::ScalarToken,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    height: usize,
    bv_lo: f32,
    bv_hi: f32,
    mode: crate::featcanon::Mode,
) -> GradientAccum {
    gradient_block_kernel_canon::<P, R7>(
        src_h, dst_h, activity, act_dst, mu1, mu2, width, height, bv_lo, bv_hi,  mode,
    )
}

/// Wasm128 sibling of [`gradient_block_kernel_canon64_vec_scalar`].
#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn gradient_block_kernel_canon64_vec_wasm128<
    P: crate::featcanon::Pool64,
    const R7: bool,
>(
    _token: archmage::Wasm128Token,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    height: usize,
    bv_lo: f32,
    bv_hi: f32,
    mode: crate::featcanon::Mode,
) -> GradientAccum {
    gradient_block_kernel_canon::<P, R7>(
        src_h, dst_h, activity, act_dst, mu1, mu2, width, height, bv_lo, bv_hi,  mode,
    )
}

/// NEON runs the scalar canonical body (2026-10-03): NEON vector `max`/`min` propagate NaN where the canon's `f32::max`/
/// `f64::max` drop it (the REV4VEC CI aarch64 failure), so only the x86_64 fused tiers run a lane-parallel body.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn gradient_block_kernel_canon64_vec_neon<
    P: crate::featcanon::Pool64,
    const R7: bool,
>(
    _token: archmage::NeonToken,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    height: usize,
    bv_lo: f32,
    bv_hi: f32,
    mode: crate::featcanon::Mode,
) -> GradientAccum {
    gradient_block_kernel_canon::<P, R7>(
        src_h, dst_h, activity, act_dst, mu1, mu2, width, height, bv_lo, bv_hi,  mode,
    )
}

/// The gradient block kernel at `revision`: the 16-lane canonical body at revision 5 and 7
/// (with revision 7's row lanes), the production SIMD bodies at revision 1.
#[allow(clippy::too_many_arguments)]
fn gradient_block_kernel_at_revision(
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    mu: Option<(&[f32], &[f32])>,
    width: usize,
    height: usize,
    revision: FormulaRevision,
) -> GradientAccum {
    let m = crate::featcanon::mode(revision);
    if m.active() {
        let (mu1, mu2) = mu.unwrap_or((&[], &[]));
        return if revision.row_lanes() {
            incant!(
                gradient_block_kernel_canon64_vec::<crate::featcanon::Lanes16F64, true>(
                    src, dst, activity, &[], mu1, mu2, width, height, 0.0, 0.0, m,
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        } else {
            incant!(
                gradient_block_kernel_canon64_vec::<crate::featcanon::Lanes16F64, false>(
                    src, dst, activity, &[], mu1, mu2, width, height, 0.0, 0.0, m,
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        };
    }
    incant!(
        gradient_block_kernel_entry(src, dst, activity, width, height),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// One row of the blockiness lattice, the shared step of every walk: the
/// vertical steps at the lattice columns `x = 8j` (`j ≥ 1`, `x < width`) of
/// the row starting at index `row`, and on a lattice row (`horizontal`) the
/// steps from the row above (`row − width`) at every column, each
/// `bounded_excess(|Δdst|, |Δsrc|, C_BLOCK)` in f64. Horizontal terms add to
/// lane `x & 7`; vertical terms to lane 0, or at revision 7 (`spread`) to
/// lane `(j − 1) & 7`. Each lane takes its terms in ascending `x`, so the
/// eight-lane accumulators get the same bits from the scalar steps and from
/// [`blockiness_row_l64`]'s eight-wide ones.
#[allow(clippy::too_many_arguments)]
pub(crate) fn blockiness_row(
    src: &[f32],
    dst: &[f32],
    row: usize,
    width: usize,
    horizontal: bool,
    spread: bool,
    sum_v: &mut crate::featcanon::SumVar,
    sum_h: &mut crate::featcanon::SumVar,
) {
    use crate::featcanon::SumVar;
    if let (SumVar::L64(v), SumVar::L64(h)) = (&mut *sum_v, &mut *sum_h) {
        incant!(
            blockiness_row_l64(src, dst, row, width, horizontal, spread, v, h),
            [v4x, v4, v3, scalar]
        );
        return;
    }
    let mut x = BLOCK_LATTICE;
    while x < width {
        let i = row + x;
        let step_dst = (dst[i] as f64 - dst[i - 1] as f64).abs();
        let step_src = (src[i] as f64 - src[i - 1] as f64).abs();
        let lane = if spread {
            (x / BLOCK_LATTICE - 1) & 7
        } else {
            x & 7
        };
        sum_v.add64(lane, bounded_excess(step_dst, step_src, C_BLOCK));
        x += BLOCK_LATTICE;
    }
    if horizontal {
        for x in 0..width {
            let i = row + x;
            let i_up = i - width;
            let step_dst = (dst[i] as f64 - dst[i_up] as f64).abs();
            let step_src = (src[i] as f64 - src[i_up] as f64).abs();
            sum_h.add64(x & 7, bounded_excess(step_dst, step_src, C_BLOCK));
        }
    }
}

/// [`blockiness_row`] for the eight-lane f64 accumulators: eight steps per
/// vector, each lane the scalar step's IEEE operations (exact widening,
/// subtraction, `abs`, `max`, the sum and the division), added to the lanes
/// in the scalar order. The vertical lattice columns are gathered eight at a
/// time; with `spread` their eight terms go to the eight lanes at once
/// (columns `8j … 8(j + 7)` with `j ≡ 1 (mod 8)` map to lanes 0–7), without
/// it to lane 0 in column order.
#[magetypes(define(f64x8), v4x, v4, v3, scalar)]
#[allow(clippy::too_many_arguments)]
fn blockiness_row_l64(
    token: Token,
    src: &[f32],
    dst: &[f32],
    row: usize,
    width: usize,
    horizontal: bool,
    spread: bool,
    v: &mut crate::featcanon::LanesF64,
    h: &mut crate::featcanon::LanesF64,
) {
    let c = f64x8::splat_t(token, C_BLOCK);
    let zero = f64x8::zero_t(token);
    macro_rules! terms {
        ($d1:expr, $d0:expr, $s1:expr, $s0:expr) => {{
            let sd = (f64x8::from_array_t(token, $d1) - f64x8::from_array_t(token, $d0)).abs();
            let ss = (f64x8::from_array_t(token, $s1) - f64x8::from_array_t(token, $s0)).abs();
            ((sd - ss).max(zero) / (sd + ss + c)).to_array()
        }};
    }
    let scalar = |i: usize, j: usize| {
        let step_dst = (dst[i] as f64 - dst[j] as f64).abs();
        let step_src = (src[i] as f64 - src[j] as f64).abs();
        bounded_excess(step_dst, step_src, C_BLOCK)
    };
    // Vertical steps: lattice columns `x = 8j`, `j = 1..=last`.
    let last = width.saturating_sub(1) / BLOCK_LATTICE;
    let mut j = 1;
    while j + 8 <= last + 1 {
        // One checked window per plane covers the eight columns and their
        // left neighbors: element `8l + 1` is column `8(j + l)`.
        let a = row + j * BLOCK_LATTICE - 1;
        let dw: &[f32; 58] = dst[a..a + 58].try_into().unwrap();
        let sw: &[f32; 58] = src[a..a + 58].try_into().unwrap();
        let d1: [f64; 8] = core::array::from_fn(|l| dw[8 * l + 1] as f64);
        let d0: [f64; 8] = core::array::from_fn(|l| dw[8 * l] as f64);
        let s1: [f64; 8] = core::array::from_fn(|l| sw[8 * l + 1] as f64);
        let s0: [f64; 8] = core::array::from_fn(|l| sw[8 * l] as f64);
        let t = terms!(d1, d0, s1, s0);
        if spread {
            for (lane, tl) in v.0.iter_mut().zip(t) {
                *lane += tl;
            }
        } else {
            for tl in t {
                v.0[0] += tl;
            }
        }
        j += 8;
    }
    while j <= last {
        let i = row + j * BLOCK_LATTICE;
        let lane = if spread { (j - 1) & 7 } else { 0 };
        v.0[lane] += scalar(i, i - 1);
        j += 1;
    }
    if horizontal {
        let up = row - width;
        let mut x = 0;
        while x + 8 <= width {
            let w = |p: &[f32], at: usize| -> [f64; 8] {
                let a: &[f32; 8] = p[at..at + 8].try_into().unwrap();
                core::array::from_fn(|l| a[l] as f64)
            };
            let t = terms!(
                w(dst, row + x),
                w(dst, up + x),
                w(src, row + x),
                w(src, up + x)
            );
            for (lane, tl) in h.0.iter_mut().zip(t) {
                *lane += tl;
            }
            x += 8;
        }
        while x < width {
            h.0[x & 7] += scalar(row + x, up + x);
            x += 1;
        }
    }
}

#[derive(Default, Clone, Copy)]
struct AppendAccum {
}

impl AppendAccum {
}

#[derive(Default, Clone, Copy)]
struct CsfwAccum {
}

impl CsfwAccum {
}
