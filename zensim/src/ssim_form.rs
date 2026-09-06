//! THE per-pixel SSIM dissimilarity — one owner for a form that was
//! hand-copied at 36 kernel sites.
//!
//! # Why this module exists
//!
//! Every SSIM-derived slot in the crate (`basic`'s `ssim_*`, the `masked`
//! block, the `IW` block — 3 signals x 2 blocks x 4 scales x 3 channels, plus
//! the basic block's own) is built from ONE per-pixel expression:
//!
//! ```text
//! num_m   = 1 - (mu1 - mu2)^2          // luminance
//! num_s   = 2*cov + C2                 // structure numerator
//! denom_s = var1 + var2 + C2           // structure denominator
//! d_raw   = 1 - num_m * num_s / denom_s
//! ```
//!
//! That expression was written out by hand **36 times** — 10 in
//! [`crate::fused`] and 26 in [`crate::simd_ops`], across four SIMD tiers and
//! their scalar tails, in vector and scalar spellings. `C2` itself had three
//! separate declarations. A form with 36 owners cannot be revised, which is
//! exactly the position the defect audit found the crate in: F4 (the one live
//! arithmetic defect) is a property of this expression, and there was nowhere
//! to fix it.
//!
//! Everything here is `#[inline(always)]` and generic over the backend trait,
//! following [`crate::fused`]'s `raw_moments_accumulate*` precedent. `nm` over
//! a release build shows zero `ssim_dissim_*` symbols surviving.
//!
//! # F4, and why the fix needs a form rather than a clamp
//!
//! `num_s/denom_s` is bounded in `[-1, 1]` by construction: `|2*cov| <= var1 +
//! var2` (Cauchy-Schwarz), so the structure term cannot run away. `num_m` is
//! `1 - D^2` for a mean difference `D`, which is **unbounded below**, so a
//! large local mean difference makes `d_raw` a large positive number. MEASURED
//! (`benchmarks/ssim_moment_explosion_2026-07-16.md`, 2,322,579 rows):
//! `f313 = iw_ssim_4th s0 ch2` reaches **5,814,302** against a photographic
//! p99.9 of **0.48**.
//!
//! The weights are NOT the amplifier, and the numbers on record prove it: the
//! `masked` weight is `1/(1 + k*a)`, bounded in `(0, 1]`, while the `IW`
//! weight is `1 + k*a` and unbounded — yet `f241` (masked, 5,797,029) and
//! `f313` (IW, 5,814,302) agree to **0.3 %**. A bounded weight cannot produce
//! 5.8e6 from a bounded `d_raw`, and an unbounded weight that mattered could
//! not land within 0.3 % of a bounded one. Both are ~1 there because the
//! pathology lives in flat regions where activity is ~0. The amplifier is
//! `num_m`.
//!
//! # Provenance: this is ssimulacra2's form, faithfully inherited
//!
//! `zensim/src/lib.rs` describes the no-`C1` luminance term as "ssimulacra2's
//! variant". VERIFIED against our own SSIMULACRA2 implementation rather than
//! assumed: `fast-ssim2`'s `simd_ops.rs`, `lib.rs` and `strip.rs` all compute
//! `num_m = mu_diff.mul_add(-mu_diff, 1.0)` with `C2 = 0.0009` and no `C1`.
//! So F4 is **inherited from the algorithm**, not a zensim coding slip, and
//! any bounded form here is a deliberate, measured DEVIATION from that
//! lineage — taken for a learned metric that must feed monotone linear heads,
//! and recorded as such.
//!
//! (`fast-ssim2` carries the same unbounded term. It is a different repo and
//! is NOT touched from here; the observation is reported, not acted on.)

use crate::feature_defs::FormulaRevision;
use magetypes::simd::backends::{F32x8Backend, F32x16Backend};
use magetypes::simd::generic::{f32x8 as GenericF32x8, f32x16};

/// SSIM structure/contrast stabiliser — ssimulacra2's value, and the ONE
/// declaration of it.
///
/// `0.0009 = 0.03^2 = (K2 * L)^2` at the textbook `K2 = 0.03`, `L = 1`. Before
/// this module it was declared three times (`fused.rs`, `simd_ops.rs`,
/// `feature_v2.rs`'s `C2_V2`), which is how [`C_SSIM_LUMA`] came to be missing
/// rather than merely omitted: there was no single place for it to be missing
/// FROM.
pub(crate) const C2: f32 = 0.0009;

/// SSIM luminance stabiliser — **derived, not chosen**, and used only by the
/// bounded forms.
///
/// Two independent derivations land on the same value:
///
/// 1. **From the family.** Every `bounded_sim(a, b, c) = (2ab+c)/(a^2+b^2+c)`
///    regularizer in this crate is `1e-4` — `C_EDGE`, `C_GMS`, `C_CONTRAST`,
///    `C_BV`. `bounded_sim` **is** the standard SSIM luminance term, already
///    present as a named, documented, shared primitive
///    (`feature_v2.rs`, "Bounded `(0, 1]`"). [`SsimLumaForm::SsimLumaC1`] is
///    that owner's form, applied to the one place in the crate that hand-rolls
///    an unbounded substitute for it.
/// 2. **From the constant already here.** [`C2`] is `(K2 * L)^2` at
///    `K2 = 0.03`, `L = 1`. The matching `C1 = (K1 * L)^2` at the textbook
///    `K1 = 0.01` is `1e-4`.
///
/// It regularises only as `mu1^2 + mu2^2 -> 0`; at photographic magnitudes it
/// is negligible, which is why the exact value is not load-bearing and the
/// agreement of the two derivations is what makes it defensible.
///
/// CORRECTION to `benchmarks/ssim_moment_explosion_2026-07-16.md` §7a, which
/// evaluated "C1(0.01)" — 100x this value. Its ordering conclusions are
/// unaffected; its photographic row moves from +0.0003 to +0.00004 here.
pub(crate) const C_SSIM_LUMA: f32 = 1e-4;

/// Which luminance term the per-pixel dissimilarity uses.
///
/// The revision axis is owned by [`crate::feature_defs`]; this enum is the
/// arithmetic that a revision SELECTS. Three bounded arms exist because the
/// choice between them is a measurement (`docs/PLAN_FEATURE_REV2_2026-09-05.md`
/// R6), not an argument — see each variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SsimLumaForm {
    /// `1 - D^2`. ssimulacra2's form, shipped, **unbounded below** (F4).
    #[default]
    Ssim2Legacy,
    /// `(2*mu1*mu2 + C1) / (mu1^2 + mu2^2 + C1)` — the standard SSIM
    /// luminance term, i.e. `bounded_sim(mu1, mu2, C_SSIM_LUMA)`.
    ///
    /// Bounded `(0, 1]`, so `d_raw` lands in `[0, 2]` and the `.max(0)` floor
    /// the call sites apply becomes provably redundant. Preserves severity
    /// ORDER where a clamp would saturate. Its cost is that it re-introduces
    /// the mean-dependent (Weber) normalisation `lib.rs` says was dropped on
    /// purpose for a perceptually-uniform space.
    SsimLumaC1,
    /// `1 / (1 + D^2)`, i.e. `1 - saturate(D^2, 1)` in the crate's own
    /// `saturate(x, c) = x/(x+c)` idiom.
    ///
    /// Bounded `(0, 1]`, **no Weber normalisation** (so it keeps the
    /// perceptual-uniformity intent), and `c = 1` is the unique scale at which
    /// it agrees with [`Self::Ssim2Legacy`] to first order in `D^2` — so it
    /// introduces no new constant. Compresses the extreme tail harder than
    /// `SsimLumaC1`.
    Lorentz,
    /// `max(0, 1 - D^2)`. **Exact** for `D^2 <= 1` — the overwhelming majority
    /// of pixels — and flat above it, so it bounds without touching the
    /// photographic regime at all, at the cost of flattening rank among the
    /// worst pixels.
    Clamp,
}

impl SsimLumaForm {
    /// The form a registered revision selects.
    pub(crate) const fn for_revision(rev: FormulaRevision) -> Self {
        match rev {
            FormulaRevision::Rev1 => Self::Ssim2Legacy,
            FormulaRevision::Rev2 => Self::REV2_LUMA,
        }
    }

    /// The bounded arm revision 2 ships — **`Clamp`, DECIDED BY MEASUREMENT**
    /// (`benchmarks/f4_arm_decision_2026-09-05.md`, R6).
    ///
    /// Named once so the R6 probe and the kernels cannot disagree about which
    /// arm "rev2" means.
    ///
    /// # Why `Clamp` and not [`Self::SsimLumaC1`]
    ///
    /// `docs/PLAN_FEATURE_REV2_2026-09-05.md` §1.4 registered a prior — *"if
    /// the probe cannot separate them, arm A (`SsimLumaC1`) ships"* — because
    /// it reuses `feature_v2`'s own `bounded_sim` owner and preserves severity
    /// ORDER among pathological pixels. **The probe DID separate them**, so
    /// the prior never fired. Four arms were extracted from ONE binary over
    /// 217,756 rows (the seven human eval corpora + the full 196,086-row
    /// safesyn training leg), fitted through the shipped Profile-D recipe at
    /// three slices x two solvers, and graded on rank, dial and cell deltas:
    ///
    /// * **F4's pathology occurs on NONE of it.** `Clamp` — which differs from
    ///   the shipped form only where `(mu1-mu2)^2 > 1` — moves **0 cells**, and
    ///   no slot anywhere reaches `|f| > 2` against the 5,814,302 on record
    ///   (which belongs to the bigcodec sweep, a population with no local
    ///   pixels).
    /// * **`Clamp` is therefore bit-identical to revision 1 on every row R6
    ///   fits or scores** — features, Gram, solve, spline and ZNPR bytes, all
    ///   six bakes sha-for-sha — so its rank delta is exactly 0 with a
    ///   degenerate CI.
    /// * `SsimLumaC1` moves **29.4 M** healthy cells (worst |delta| 0.771) and
    ///   `Lorentz` **24.0 M** (worst 0.0901), against a pre-registered 1e-4
    ///   bar, to buy at most `+0.0025` CID22 in one of six variants. Both fail
    ///   the healthy-cell gate; neither wins a rank majority in more than one
    ///   variant.
    ///
    /// So `Clamp` is the unique arm that is ONLY a fix: it changes the metric
    /// exactly where the metric was unbounded and nowhere else, which is what
    /// makes a rev2 flip cheap for every table whose content resembles those
    /// corpora.
    ///
    /// **Its known cost, recorded rather than glossed:** above `D^2 = 1` every
    /// pixel gets the same `num_m = 0`, so tail ORDER is flat there — the exact
    /// property §1.4's prior was protecting. `d` stays bounded in `[0, 2]`
    /// regardless. If a future population makes that order load-bearing,
    /// [`Self::Lorentz`] is the registered successor (bounded, no Weber term,
    /// monotone in `D^2`, and 4-6 orders of magnitude closer to revision 1 on
    /// healthy content than `SsimLumaC1`) — not `SsimLumaC1`.
    pub(crate) const REV2_LUMA: Self = Self::Clamp;

    /// Whether `d_raw` is bounded below by 0 by construction, making the
    /// call sites' `.max(0)` floor redundant rather than load-bearing.
    ///
    /// True for every arm whose `num_m` is bounded in `[0, 1]`: with
    /// `num_s/denom_s` in `[-1, 1]`, `d_raw = 1 - num_m*(num_s/denom_s)` lands
    /// in `[0, 2]`.
    pub(crate) const fn bounds_dissim(self) -> bool {
        !matches!(self, Self::Ssim2Legacy)
    }
}

/// **The revision this build ships.**
///
/// Changing this line is the era flip. Everything else — the registry entries,
/// the gates, the recalculation — hangs off it, and
/// `docs/PLAN_FEATURE_REV2_2026-09-05.md` R1 is the control that proves the
/// machinery around it is inert while it still reads `Rev1`.
pub(crate) const SHIPPED_REVISION: FormulaRevision = FormulaRevision::Rev1;

/// **The revision switch — one owner, read once per process.**
///
/// Modelled on [`crate::feature_v2`]'s `era2_dense_enabled`, deliberately and
/// for its stated reason: *"a switch that some call sites honour and others do
/// not is the same defect"*. Every SSIM kernel reads this ONCE, above its
/// loop, and passes the result down; no call site chooses its own form.
///
/// `ZENSIM_FORMULA_REV=1` pins revision 1 and `=2` pins revision 2, so a
/// research extraction can reproduce either era's semantics from one binary
/// (phase 3's G3.2). Anything else — including unset — is
/// [`SHIPPED_REVISION`]. The two accepted values are the SAME BYTE LENGTH on
/// purpose: this repo has measured an environment block's size shifting a
/// binary's layout by ~10 % at 2304²
/// (`benchmarks/era2_perf_break_2026-08-31.md` §22.5), so an A/B that varies
/// the value must not vary the length.
pub(crate) fn active_revision() -> FormulaRevision {
    use std::sync::OnceLock;
    static REV: OnceLock<FormulaRevision> = OnceLock::new();
    *REV.get_or_init(|| match std::env::var("ZENSIM_FORMULA_REV").as_deref() {
        Ok("1") => FormulaRevision::Rev1,
        Ok("2") => FormulaRevision::Rev2,
        _ => SHIPPED_REVISION,
    })
}

/// The luminance form the active revision selects.
///
/// Call this ONCE per kernel invocation, above the pixel loop — it reads a
/// `OnceLock`, which LLVM cannot hoist out of a loop for you.
#[inline]
pub(crate) fn active_luma_form() -> SsimLumaForm {
    use std::sync::OnceLock;
    static FORM: OnceLock<SsimLumaForm> = OnceLock::new();
    *FORM.get_or_init(|| {
        // `ZENSIM_SSIM_LUMA` is a MEASUREMENT override and nothing else. R6
        // has to fit the monotone linear class against all four luminance
        // arms on ONE set of pixels, from one binary, or the comparison is
        // confounded by a rebuild (this repo has measured a rebuild alone
        // moving a 2304^2 timing ~10 %). No shipping path sets it; the
        // revision is what a shipping path selects.
        match std::env::var("ZENSIM_SSIM_LUMA").as_deref() {
            Ok("ssim2") => SsimLumaForm::Ssim2Legacy,
            Ok("c1") => SsimLumaForm::SsimLumaC1,
            Ok("lorentz") => SsimLumaForm::Lorentz,
            Ok("clamp") => SsimLumaForm::Clamp,
            _ => SsimLumaForm::for_revision(active_revision()),
        }
    })
}

/// Hoisted splats + the selected form, built once per kernel invocation.
///
/// Bundling them keeps the per-site call short and keeps [`SsimLumaForm`]
/// loop-invariant, so the `match` is unswitched out of the inner loop after
/// inlining rather than evaluated per pixel.
#[derive(Clone, Copy)]
pub(crate) struct SsimSplats16<T: F32x16Backend + Copy> {
    form: SsimLumaForm,
    one: f32x16<T>,
    two: f32x16<T>,
    zero: f32x16<T>,
    c2: f32x16<T>,
    c1: f32x16<T>,
}

impl<T: F32x16Backend + Copy> SsimSplats16<T> {
    #[inline(always)]
    pub(crate) fn new(token: T, form: SsimLumaForm) -> Self {
        Self {
            form,
            one: f32x16::splat(token, 1.0),
            two: f32x16::splat(token, 2.0),
            zero: f32x16::zero(token),
            c2: f32x16::splat(token, C2),
            c1: f32x16::splat(token, C_SSIM_LUMA),
        }
    }

    /// `d_raw = 1 - num_m * num_s / denom_s`, BEFORE any weight multiply and
    /// before the call site's `.max(0)` floor — the two things that differ
    /// between the basic, masked and IW call sites.
    ///
    /// For [`SsimLumaForm::Ssim2Legacy`] the emitted operations are
    /// bit-identical to the hand-written form this replaced: same `mul_add`
    /// spellings, same order, same single divide.
    #[inline(always)]
    pub(crate) fn dissim(
        &self,
        m1: f32x16<T>,
        m2: f32x16<T>,
        ssq: f32x16<T>,
        s12: f32x16<T>,
    ) -> f32x16<T> {
        let num_s = self.two.mul_add((-m1).mul_add(m2, s12), self.c2);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + self.c2;
        // Every arm reduces to ONE divide by expressing the luminance term as
        // a (numerator, denominator) pair folded into the structure ratio.
        let (num_m, den_m) = match self.form {
            SsimLumaForm::Ssim2Legacy => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one), self.one)
            }
            SsimLumaForm::SsimLumaC1 => (
                self.two.mul_add(m1 * m2, self.c1),
                m1.mul_add(m1, m2.mul_add(m2, self.c1)),
            ),
            SsimLumaForm::Lorentz => {
                let mu_diff = m1 - m2;
                (self.one, mu_diff.mul_add(mu_diff, self.one))
            }
            SsimLumaForm::Clamp => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one).max(self.zero), self.one)
            }
        };
        // `den_m` is exactly 1.0 for the two arms that do not need it, and
        // `x * 1.0` is exact, so the legacy arm keeps its original rounding.
        self.one - (num_m * num_s) / (den_m * denom_s)
    }
}

/// 8-lane sibling of [`SsimSplats16`].
#[derive(Clone, Copy)]
pub(crate) struct SsimSplats8<T: F32x8Backend + Copy> {
    form: SsimLumaForm,
    one: GenericF32x8<T>,
    two: GenericF32x8<T>,
    zero: GenericF32x8<T>,
    c2: GenericF32x8<T>,
    c1: GenericF32x8<T>,
}

impl<T: F32x8Backend + Copy> SsimSplats8<T> {
    #[inline(always)]
    pub(crate) fn new(token: T, form: SsimLumaForm) -> Self {
        Self {
            form,
            one: GenericF32x8::splat(token, 1.0),
            two: GenericF32x8::splat(token, 2.0),
            zero: GenericF32x8::zero(token),
            c2: GenericF32x8::splat(token, C2),
            c1: GenericF32x8::splat(token, C_SSIM_LUMA),
        }
    }

    /// See [`SsimSplats16::dissim`].
    #[inline(always)]
    pub(crate) fn dissim(
        &self,
        m1: GenericF32x8<T>,
        m2: GenericF32x8<T>,
        ssq: GenericF32x8<T>,
        s12: GenericF32x8<T>,
    ) -> GenericF32x8<T> {
        let num_s = self.two.mul_add((-m1).mul_add(m2, s12), self.c2);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + self.c2;
        let (num_m, den_m) = match self.form {
            SsimLumaForm::Ssim2Legacy => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one), self.one)
            }
            SsimLumaForm::SsimLumaC1 => (
                self.two.mul_add(m1 * m2, self.c1),
                m1.mul_add(m1, m2.mul_add(m2, self.c1)),
            ),
            SsimLumaForm::Lorentz => {
                let mu_diff = m1 - m2;
                (self.one, mu_diff.mul_add(mu_diff, self.one))
            }
            SsimLumaForm::Clamp => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one).max(self.zero), self.one)
            }
        };
        self.one - (num_m * num_s) / (den_m * denom_s)
    }
}

/// Scalar sibling — the `width % LANES` tails every vector kernel carries.
///
/// Same operations, same order, same `mul_add` spellings as the vector arms,
/// so a tail row and a vector row agree to the extent f32 FMA allows.
#[inline(always)]
pub(crate) fn ssim_dissim_raw_scalar(
    form: SsimLumaForm,
    m1: f32,
    m2: f32,
    ssq: f32,
    s12: f32,
) -> f32 {
    let num_s = 2.0f32.mul_add((-m1).mul_add(m2, s12), C2);
    let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + C2;
    let (num_m, den_m) = match form {
        SsimLumaForm::Ssim2Legacy => {
            let mu_diff = m1 - m2;
            (mu_diff.mul_add(-mu_diff, 1.0f32), 1.0f32)
        }
        SsimLumaForm::SsimLumaC1 => (
            2.0f32.mul_add(m1 * m2, C_SSIM_LUMA),
            m1.mul_add(m1, m2.mul_add(m2, C_SSIM_LUMA)),
        ),
        SsimLumaForm::Lorentz => {
            let mu_diff = m1 - m2;
            (1.0f32, mu_diff.mul_add(mu_diff, 1.0f32))
        }
        SsimLumaForm::Clamp => {
            let mu_diff = m1 - m2;
            (mu_diff.mul_add(-mu_diff, 1.0f32).max(0.0f32), 1.0f32)
        }
    };
    let d = 1.0f32 - (num_m * num_s) / (den_m * denom_s);
    // The boundedness claim, checked rather than asserted in prose: a bounded
    // luminance term puts `d` in `[0, 2]`, which is what makes the call sites'
    // `.max(0)` floor redundant instead of load-bearing. The tolerance is for
    // f32 rounding at the `num_s/denom_s = 1` boundary, not for slack in the
    // claim. Debug-only: this is the innermost scalar tail.
    debug_assert!(
        !form.bounds_dissim() || (-1e-5..=2.0 + 1e-5).contains(&d),
        "bounded SSIM form {form:?} produced d = {d} outside [0, 2]"
    );
    d
}

/// One-call wrapper over [`SsimSplats16`] for the 16-lane kernels.
///
/// The splats are loop-invariant constants, so building them inside an
/// `#[inline(always)]` call is free — LLVM hoists them out of the loop
/// exactly as it did when they were hand-declared above it. The struct stays
/// public to this crate for any future site that wants to hoist explicitly.
#[inline(always)]
pub(crate) fn ssim_dissim16<T: F32x16Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: f32x16<T>,
    m2: f32x16<T>,
    ssq: f32x16<T>,
    s12: f32x16<T>,
) -> f32x16<T> {
    SsimSplats16::new(token, form).dissim(m1, m2, ssq, s12)
}

/// 8-lane sibling of [`ssim_dissim16`].
#[inline(always)]
pub(crate) fn ssim_dissim8<T: F32x8Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: GenericF32x8<T>,
    m2: GenericF32x8<T>,
    ssq: GenericF32x8<T>,
    s12: GenericF32x8<T>,
) -> GenericF32x8<T> {
    SsimSplats8::new(token, form).dissim(m1, m2, ssq, s12)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sweep that reaches both the healthy regime and F4's pathology.
    ///
    /// `ssq` is `E[s^2] + E[d^2]`, so `denom_s = ssq - mu1^2 - mu2^2 + C2` is
    /// the variance sum; the cases keep it positive, as the walk does.
    fn cases() -> Vec<(f32, f32, f32, f32)> {
        let mut v = Vec::new();
        for &(m1, m2) in &[
            (0.0f32, 0.0f32),
            (0.3, 0.3),
            (0.3, 0.31), // photographic
            (0.05, 0.9),
            (50.0, 60.0), // the 2026-07-16 analytic rows
            (100.0, 400.0),
            (50.0, 2450.0),
            (2450.0, 50.0),
        ] {
            for &var in &[0.0f32, 1e-6, 1e-3, 0.25, 10.0] {
                for &cov_frac in &[-1.0f32, -0.5, 0.0, 0.5, 1.0] {
                    let ssq = m1 * m1 + m2 * m2 + 2.0 * var;
                    let s12 = m1 * m2 + cov_frac * var;
                    v.push((m1, m2, ssq, s12));
                }
            }
        }
        v
    }

    /// The exact expression the 36 hand-copied sites carried, transcribed
    /// verbatim — the control for the extraction.
    fn legacy_scalar(m1: f32, m2: f32, ssq: f32, s12: f32) -> f32 {
        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-m1).mul_add(m2, s12), C2);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + C2;
        1.0f32 - (num_m * num_s) / denom_s
    }

    /// **The extraction gate.** `Ssim2Legacy` must reproduce the replaced
    /// expression BIT-for-BIT, not merely closely: this is what lets the
    /// 36-site rewrite claim to be inert.
    #[test]
    fn legacy_arm_is_bit_identical_to_the_expression_it_replaced() {
        for (m1, m2, ssq, s12) in cases() {
            let got = ssim_dissim_raw_scalar(SsimLumaForm::Ssim2Legacy, m1, m2, ssq, s12);
            let want = legacy_scalar(m1, m2, ssq, s12);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "legacy arm diverged at mu=({m1},{m2}) ssq={ssq} s12={s12}: {got} vs {want}"
            );
        }
    }

    /// **F4, stated as a failing property of the shipped form.** The legacy
    /// arm is unbounded; every other arm is not. If this ever stops failing
    /// for `Ssim2Legacy`, the pathology has been fixed somewhere else and
    /// this module's reason to exist has changed.
    #[test]
    fn only_the_legacy_arm_is_unbounded() {
        // mu1 = 50, mu2 = 2450 on a near-flat region: the configuration
        // `benchmarks/ssim_moment_explosion_2026-07-16.md` reproduces 5.8e6 from.
        let (m1, m2) = (50.0f32, 2450.0f32);
        let var = 1e-6f32;
        let ssq = m1 * m1 + m2 * m2 + 2.0 * var;
        let s12 = m1 * m2 + var;

        let legacy = ssim_dissim_raw_scalar(SsimLumaForm::Ssim2Legacy, m1, m2, ssq, s12);
        assert!(
            legacy > 1e6,
            "the F4 pathology should reproduce here, got {legacy}"
        );

        for form in [
            SsimLumaForm::SsimLumaC1,
            SsimLumaForm::Lorentz,
            SsimLumaForm::Clamp,
        ] {
            assert!(form.bounds_dissim(), "{form:?} claims not to bound");
            let d = ssim_dissim_raw_scalar(form, m1, m2, ssq, s12);
            assert!(
                (-1e-5..=2.0 + 1e-5).contains(&d),
                "{form:?} produced d = {d} outside [0, 2] on the pathological case"
            );
        }
    }

    /// Boundedness across the whole sweep, not just the headline case.
    #[test]
    fn bounded_arms_keep_d_in_zero_to_two_everywhere() {
        for form in [
            SsimLumaForm::SsimLumaC1,
            SsimLumaForm::Lorentz,
            SsimLumaForm::Clamp,
        ] {
            for (m1, m2, ssq, s12) in cases() {
                let d = ssim_dissim_raw_scalar(form, m1, m2, ssq, s12);
                assert!(
                    d.is_finite() && (-1e-4..=2.0 + 1e-4).contains(&d),
                    "{form:?} at mu=({m1},{m2}) ssq={ssq} s12={s12} gave {d}"
                );
            }
        }
    }

    /// **`Clamp` is exact where it matters.** It is defined to differ from the
    /// shipped form only where `(mu1-mu2)^2 > 1`, which is why it moved ZERO
    /// of 22,396 cells on the synthetic dump. Pinning that here means a future
    /// edit cannot quietly make it a general-purpose approximation.
    #[test]
    fn clamp_arm_is_bit_identical_to_legacy_below_the_knee() {
        for (m1, m2, ssq, s12) in cases() {
            if (m1 - m2) * (m1 - m2) > 1.0 {
                continue;
            }
            let c = ssim_dissim_raw_scalar(SsimLumaForm::Clamp, m1, m2, ssq, s12);
            let l = ssim_dissim_raw_scalar(SsimLumaForm::Ssim2Legacy, m1, m2, ssq, s12);
            assert_eq!(c.to_bits(), l.to_bits(), "clamp diverged below the knee");
        }
    }

    /// **The G-OWNER claim, checked.** `SsimLumaC1`'s luminance term is not
    /// merely *like* the crate's `bounded_sim` primitive — it IS it, and this
    /// compares the f32 kernel arm against the f64 owner rather than against a
    /// second transcription.
    ///
    /// `crate::feature_v2` only exists under `feature-regime-v2` — this test
    /// asserts parity with it, so it is unrunnable (not just untestable)
    /// without that feature.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn ssim_luma_arm_is_the_crate_s_own_bounded_sim() {
        for (m1, m2, ssq, s12) in cases() {
            let want_luma: f64 =
                crate::feature_v2::bounded_sim(m1 as f64, m2 as f64, C_SSIM_LUMA as f64);
            // Reconstruct the arm's luminance factor from its output:
            // d = 1 - luma * (num_s/denom_s).
            let num_s = 2.0f64 * ((s12 - m1 * m2) as f64) + C2 as f64;
            let denom_s = ((ssq - m1 * m1 - m2 * m2) as f64) + C2 as f64;
            let d = ssim_dissim_raw_scalar(SsimLumaForm::SsimLumaC1, m1, m2, ssq, s12) as f64;
            let want = 1.0 - want_luma * (num_s / denom_s);
            let scale = want.abs().max(1.0);
            assert!(
                (d - want).abs() <= 2e-3 * scale,
                "arm vs bounded_sim at mu=({m1},{m2}): {d} vs {want}"
            );
            assert!(
                want_luma > 0.0 && want_luma <= 1.0 + 1e-12,
                "bounded_sim escaped (0,1]: {want_luma}"
            );
        }
    }

    /// The constant is DERIVED, and both derivations are pinned so a future
    /// edit has to break an equation rather than a taste.
    ///
    /// The second derivation checks agreement with `crate::feature_v2`'s own
    /// regularizers, which only exist under `feature-regime-v2`.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn c1_is_derived_from_the_constants_already_present() {
        // 1. The SSIM relation to the C2 already in the kernel: C2 = (K2*L)^2
        //    at K2 = 0.03, L = 1 -> C1 = (K1*L)^2 at the textbook K1 = 0.01.
        let k2 = (C2 as f64).sqrt();
        assert!((k2 - 0.03).abs() < 1e-9, "C2 is not (0.03)^2: K2 = {k2}");
        // Tolerances are RELATIVE: both constants are f32, so an exact f64
        // comparison would only be testing f32's round-trip error (2.5e-12
        // here), not the derivation.
        let k1 = 0.01f64;
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
        assert!(
            rel(C_SSIM_LUMA as f64, k1 * k1) < 1e-6,
            "C_SSIM_LUMA is not (0.01)^2: {C_SSIM_LUMA}"
        );
        // 2. The family's own bounded_sim regularizers.
        for c in [
            crate::feature_v2::C_EDGE,
            crate::feature_v2::C_GMS,
            crate::feature_v2::C_CONTRAST,
        ] {
            assert!(
                rel(c, C_SSIM_LUMA as f64) < 1e-6,
                "family regularizer {c} disagrees with C_SSIM_LUMA"
            );
        }
    }

    /// **G3.2** — pinning the shipped revision is the same as not pinning it.
    #[test]
    fn selecting_the_shipped_revision_is_a_no_op() {
        assert_eq!(
            SsimLumaForm::for_revision(SHIPPED_REVISION),
            SsimLumaForm::for_revision(active_revision()),
            "the active revision is not the shipped one (is ZENSIM_FORMULA_REV set?)"
        );
        assert!(
            !SsimLumaForm::Ssim2Legacy.bounds_dissim(),
            "the legacy arm must not claim to bound"
        );
    }
}
