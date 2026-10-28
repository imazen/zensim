//! Scalar and SIMD forms of per-pixel SSIM dissimilarity.
//!
//! Revision 1 retains the legacy luminance term. Canonical revisions use the
//! clamped form, selected from the comparison's explicit revision.

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
            FormulaRevision::Rev5 | FormulaRevision::Rev7 => Self::REV2_LUMA,
        }
    }

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

/// The default for computations without a declared revision.
pub(crate) const fn active_revision() -> FormulaRevision {
    SHIPPED_REVISION
}

/// Explicit research cross-revision replay keeps using the process kernels.
/// Normal model-bound requests never consult mutable thread/global overrides.
pub(crate) fn effective_revision(revision: FormulaRevision) -> FormulaRevision {
    revision
}

#[inline]
pub(crate) fn active_luma_form() -> SsimLumaForm {
    SsimLumaForm::Ssim2Legacy
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
}

impl<T: F32x16Backend + Copy> SsimSplats16<T> {
    #[inline(always)]
    pub(crate) fn new(token: T, form: SsimLumaForm) -> Self {
        Self {
            form,
            one: f32x16::splat_t(token, 1.0),
            two: f32x16::splat_t(token, 2.0),
            zero: f32x16::zero_t(token),
            c2: f32x16::splat_t(token, C2),
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
            SsimLumaForm::Clamp => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one).max(self.zero), self.one)
            }
        };
        // `den_m` is exactly 1.0 for the two arms that do not need it, and
        // `x * 1.0` is exact, so the legacy arm keeps its original rounding.
        self.one - (num_m * num_s) / (den_m * denom_s)
    }

    #[inline(always)]
    pub(crate) fn direct(
        &self,
        m1: f32x16<T>,
        m2: f32x16<T>,
        ssq: f32x16<T>,
        err: f32x16<T>,
    ) -> f32x16<T> {
        let mu_diff = m1 - m2;
        let mean_error2 = mu_diff * mu_diff;
        let variance_sum = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)).max(self.zero);
        let error_variance = (err - mean_error2).max(self.zero);
        let luma_loss = match self.form {
            SsimLumaForm::Ssim2Legacy => mean_error2,
            SsimLumaForm::Clamp => mean_error2.min(self.one),
        };
        (self.one - luma_loss).mul_add(error_variance / (variance_sum + self.c2), luma_loss)
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
}

impl<T: F32x8Backend + Copy> SsimSplats8<T> {
    #[inline(always)]
    pub(crate) fn new(token: T, form: SsimLumaForm) -> Self {
        Self {
            form,
            one: GenericF32x8::splat_t(token, 1.0),
            two: GenericF32x8::splat_t(token, 2.0),
            zero: GenericF32x8::zero_t(token),
            c2: GenericF32x8::splat_t(token, C2),
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
            SsimLumaForm::Clamp => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one).max(self.zero), self.one)
            }
        };
        self.one - (num_m * num_s) / (den_m * denom_s)
    }

    /// 8-lane sibling of [`SsimSplats16::direct`].
    #[inline(always)]
    pub(crate) fn direct(
        &self,
        m1: GenericF32x8<T>,
        m2: GenericF32x8<T>,
        ssq: GenericF32x8<T>,
        err: GenericF32x8<T>,
    ) -> GenericF32x8<T> {
        let mu_diff = m1 - m2;
        let mean_error2 = mu_diff * mu_diff;
        let variance_sum = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)).max(self.zero);
        let error_variance = (err - mean_error2).max(self.zero);
        let luma_loss = match self.form {
            SsimLumaForm::Ssim2Legacy => mean_error2,
            SsimLumaForm::Clamp => mean_error2.min(self.one),
        };
        (self.one - luma_loss).mul_add(error_variance / (variance_sum + self.c2), luma_loss)
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

/// Scalar direct-error form — the tail sibling of [`SsimSplats16::direct`],
/// same spellings so vector and tail agree to their usual `mul_add` residual.
#[inline(always)]
pub(crate) fn ssim_direct_raw_scalar(
    form: SsimLumaForm,
    m1: f32,
    m2: f32,
    ssq: f32,
    err: f32,
) -> f32 {
    let mu_diff = m1 - m2;
    let mean_error2 = mu_diff * mu_diff;
    let variance_sum = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)).max(0.0f32);
    let error_variance = (err - mean_error2).max(0.0f32);
    let luma_loss = match form {
        SsimLumaForm::Ssim2Legacy => mean_error2,
        SsimLumaForm::Clamp => mean_error2.min(1.0f32),
    };
    (1.0f32 - luma_loss).mul_add(error_variance / (variance_sum + C2), luma_loss)
}

/// 16-lane direct-error form; see [`SsimSplats16::direct`].
#[inline(always)]
pub(crate) fn ssim_direct16<T: F32x16Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: f32x16<T>,
    m2: f32x16<T>,
    ssq: f32x16<T>,
    err: f32x16<T>,
) -> f32x16<T> {
    SsimSplats16::new(token, form).direct(m1, m2, ssq, err)
}

/// 8-lane direct-error form; see [`SsimSplats8::direct`].
#[inline(always)]
pub(crate) fn ssim_direct8<T: F32x8Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: GenericF32x8<T>,
    m2: GenericF32x8<T>,
    ssq: GenericF32x8<T>,
    err: GenericF32x8<T>,
) -> GenericF32x8<T> {
    SsimSplats8::new(token, form).direct(m1, m2, ssq, err)
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

    /// **G3.2** — pinning the shipped revision is the same as not pinning it.
    #[test]
    fn selecting_the_shipped_revision_is_a_no_op() {
        assert_eq!(
            SsimLumaForm::for_revision(SHIPPED_REVISION),
            SsimLumaForm::Ssim2Legacy,
            "the default arithmetic must use the shipped revision"
        );
        assert!(
            !SsimLumaForm::Ssim2Legacy.bounds_dissim(),
            "the legacy arm must not claim to bound"
        );
    }
}
