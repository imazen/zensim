//! THE v1 HF-energy ratio family — one owner for three expressions that were
//! hand-copied at three sites, and the home of the **F17** fix.
//!
//! # The family, and which member is broken
//!
//! v1's basic block ends with three ratio slots built from two pooled second
//! moments (`var_* = Σ(x − μ)²/n`) and one pooled first absolute moment
//! (`mad_* = Σ|x − μ|/n`):
//!
//! ```text
//! var_loss     = max(0, 1 − var_dst/var_src)      bounded [0, 1]
//! tex_loss     = max(0, 1 − mad_dst/mad_src)      bounded [0, 1]
//! contrast_inc = max(0, var_dst/var_src − 1)      UNBOUNDED ABOVE
//! ```
//!
//! All three are `max(0, ·)` of a difference over the **source** term. That
//! bounds the two `loss` members structurally — their numerator can never
//! exceed their denominator — and bounds nothing at all for the `gain` member,
//! whose numerator is the distorted term. A flat source drives `var_src → 0`
//! past the [`VAR_SRC_FLOOR`] gate while the distorted image still carries HF,
//! and the ratio runs away. **The gate is a threshold, not a stabiliser**: it
//! decides *whether* to divide, never *what* to divide by.
//!
//! # F17, measured
//!
//! Over the R6 rev1 tables — 216,756 real pairs, eight corpora, all 372 slots
//! (`docs/PLAN_FEATURE_REV2_2026-09-05.md` §11.2): the twelve `contrast_inc`
//! slots are the **top twelve of all 372 by maximum**, worst **36,465.74**, and
//! the thirteenth slot is **1.972**. The population separates with no overlap.
//! Against the gold photographic holdout's own p99.9 over those slots
//! (CID22, 0.34687) the worst value is **×105,127**. Its two siblings max at
//! exactly `1.000000`, as the algebra above says they must.
//!
//! This is **not** F4's shape. F4's 5.8e6 lives in a bigcodec sweep with no
//! local pixels and fires on ZERO of those 216,756 rows; F17 fires on five
//! distortion corpora and on the training leg, on 0.0198 % of cells.
//!
//! # The fixed form already exists one block over
//!
//! v2's `HF_GAIN` — the same quantity in the same units (squared XYB energy),
//! computed per pixel instead of per pool — is
//! `bounded_excess_pair(hf_dst², hf_src², C_HF)`, i.e.
//! `max(0, a−b)/(a+b+C_HF)`, with `C_HF` declared as *"stabilizer for the HF
//! gain/loss/mag-loss bounded-excess forms"*. So F17 is not a missing idea; it
//! is one block that did not use the idea. [`HfGainForm::BoundedExcess`] is
//! that owner's form applied here.
//!
//! # Why an arm and not just a clamp
//!
//! Because `contrast_inc` is nonzero on 12 % (CID22) to 52 % (KADID) of cells,
//! any bounded form necessarily moves healthy content — so *which* bounded form
//! is a measurement, pre-registered as §11 of the rev2 plan and decided by it,
//! exactly as `ssim_form` handled F4.

use crate::feature_defs::FormulaRevision;

/// The `var_src` / `mad_src` floor below which the ratio is not formed.
///
/// A THRESHOLD, not a stabiliser — it chooses whether to divide, and the value
/// it divides by is unchanged on the other side of it. Owned here so it cannot
/// drift between the three sites that used to spell it out.
pub(crate) const VAR_SRC_FLOOR: f64 = 1e-10;

/// Which form the HF *gain* member uses.
///
/// The revision axis is owned by [`crate::feature_defs`]; this enum is the
/// arithmetic a revision SELECTS — the same split [`crate::ssim_form`] uses.
/// Four bounded/compressing arms exist because the choice between them is a
/// measurement (`docs/PLAN_FEATURE_REV2_2026-09-05.md` §11), not an argument.
///
/// Throughout, `a = var_dst`, `b = var_src`, and `g = max(0, a/b − 1)`.
// Every variant ends in `Excess` because every arm IS an excess form —
// `max(0, dst - src)` under four different normalisations. Dropping the suffix
// would leave `Ratio`, `Bounded`, `Log1p`, `Saturating`, `Capped`, which read
// as unrelated ideas rather than as one family; the shared word is the point.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum HfGainForm {
    /// `g`. The shipped form, **unbounded above** (F17).
    #[default]
    RatioExcess,
    /// `g / (g + 1)`, i.e. the crate's own `saturate(g, 1)` idiom.
    ///
    /// Bounded `[0, 1)`, strictly increasing, agrees with
    /// [`Self::RatioExcess`] to **first** order (`g − g² + …`), and `c = 1` is
    /// the unique scale at which it does — so it introduces no constant, the
    /// same derivation [`crate::ssim_form::SsimLumaForm::Lorentz`] uses.
    ///
    /// It also has an identity the other arms do not:
    /// `g/(g+1) = max(0, 1 − var_src/var_dst)`, the exact src↔dst reflection of
    /// `var_loss = max(0, 1 − var_dst/var_src)`. So it makes the family
    /// symmetric by changing only the member that is broken.
    SaturatingExcess,
}

impl HfGainForm {
    /// Resolve a per-comparison revision while retaining explicit research defaults.
    pub(crate) fn at_revision(revision: Option<crate::feature_defs::FormulaRevision>) -> Self {
        let rev = crate::ssim_form::effective_revision(
            revision.unwrap_or_else(crate::ssim_form::active_revision),
        );
        if rev == crate::ssim_form::active_revision() {
            active_gain_form()
        } else {
            Self::for_revision(rev)
        }
    }

    /// The form a registered revision selects.
    pub(crate) const fn for_revision(rev: FormulaRevision) -> Self {
        match rev {
            FormulaRevision::Rev1 => Self::RatioExcess,
            FormulaRevision::Rev5
            | FormulaRevision::Rev7 => Self::REV2_HFGAIN,
        }
    }

    /// The arm revision 2 ships — **`SaturatingExcess`, DECIDED BY MEASUREMENT**
    /// (`benchmarks/feature_rev2_2026-09-05.md` §11, R6b).
    ///
    /// Named ONCE so the R6b probe and the kernels cannot disagree about what
    /// "rev2" means for this family.
    ///
    /// # How the pre-registered rule selected it
    ///
    /// Five arms were extracted from ONE binary over 216,756 rows (the seven
    /// human eval corpora + the full 196,086-row safesyn training leg), fitted
    /// through the shipped Profile-D recipe at two slices × two solvers, and
    /// graded on the gates `docs/PLAN_FEATURE_REV2_2026-09-05.md` §11.8 fixed
    /// before any of it ran. Rule 1 (H3/H4/H5) left exactly one arm standing:
    ///
    /// * [`Self::Log1pExcess`] fails **H3** — log growth is a compression, not
    ///   a bound, and it measures **10.504** where the other arms cannot exceed
    ///   1. Carried precisely so that cost is measured rather than assumed.
    /// * [`Self::BoundedExcess`] fails **H5** with **263,195** adjacent-pair
    ///   inversions: it reads the MAGNITUDE of `var_src`, so it does not bound
    ///   the shipped statistic, it replaces it with a scale-dependent one.
    /// * [`Self::CappedExcess`] fails **H5** with **67,224** new ties — F4's
    ///   `Clamp` analogue, which was free there (0 moved cells) and is not here
    ///   (2.59 % of cells exceed `g = 1`).
    /// * `SaturatingExcess` passes all three: max **0.99997** against its
    ///   declared bound of 1, **0** inversions, **0** new ties.
    ///
    /// Rule 2 then fired on its own terms — it is the sole survivor AND wins a
    /// strict majority of {CID22, KonJND, AIC-3} with CI-excluding paired
    /// bootstrap deltas in 2 of the 4 variants (CID22 **+0.0027…+0.0090**,
    /// AIC-3 **+0.0009…+0.0032**). Its two constant-free derivations survived
    /// the measurement: it agrees with revision 1 to FIRST order in `g`, and
    /// `g/(g+1) = max(0, 1 − var_src/var_dst)` makes the family symmetric by
    /// changing only the member that is broken.
    ///
    /// # What it buys, and what it costs — both measured
    ///
    /// **LIVE goes 0.7357 → 0.9500** (+0.214) and 0.7960 → 0.9503 on the
    /// `lasso` variants, with TID +0.033/+0.040 and KADID +0.021/+0.031: the
    /// unbounded slot was actively wrecking the fit on the corpora where it
    /// fires (LIVE holds 122 of 779 rows above 100).
    ///
    /// **KonJND regresses, on every arm and every variant**
    /// (`satexcess` −0.013…−0.080, CI-excluding on 3 of 4). Two structurally
    /// different bounded maps produce the same sign, so this is a property of
    /// BOUNDING `contrast_inc`, not of this arm's shape: the near-threshold
    /// corpus is using the unbounded magnitude and no bounded form returns it.
    /// If that axis is worth recovering it is an APPEND slot under the
    /// append-only numbering, not a reason to keep an unbounded one.
    ///
    /// The dial (H7) does not regress: monotonicity within 0.0011 of revision 1,
    /// tied rate 0.0000–0.0001, negative-tail fraction within 0.022 (and
    /// BETTER on `s228_bvls`), at a reach cost of 2.2–11.8 points.
    pub(crate) const REV2_HFGAIN: Self = Self::SaturatingExcess;

    /// Whether the arm has a **structural** upper bound (gate H3).
    pub(crate) const fn bounds_gain(self) -> bool {
        matches!(self, Self::SaturatingExcess)
    }

    /// The arm's least upper bound, or `None` when it has none.
    pub(crate) const fn upper_bound(self) -> Option<f64> {
        match self {
            Self::RatioExcess => None,
            Self::SaturatingExcess => Some(1.0),
        }
    }

    /// Stable lower-case token — the `ZENSIM_HF_GAIN` spelling, and the arm
    /// label in every R6b artefact.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::RatioExcess => "ratio",
            Self::SaturatingExcess => "satexcess",
        }
    }
}

/// **The form the active revision selects.**
///
/// Read ONCE per finalize, not per call site — modelled on
/// [`crate::ssim_form::active_luma_form`] and for its stated reason: *"a switch
/// that some call sites honour and others do not is the same defect"*.
///
/// `ZENSIM_HF_GAIN` is a MEASUREMENT override and nothing else. R6b has to fit
/// the monotone linear class against every arm on ONE set of pixels from ONE
/// binary, or the comparison is confounded by a rebuild. No shipping path sets
/// it; a shipping path selects a REVISION, through the equal-byte-length
/// `ZENSIM_FORMULA_REV` that [`crate::ssim_form::active_revision`] owns.
#[inline]
pub(crate) fn active_gain_form() -> HfGainForm {
    HfGainForm::for_revision(crate::ssim_form::active_revision())
}

/// **`contrast_inc` / `hf_energy_gain`** — the one member with an arm.
///
/// Returns exactly `0.0` whenever `var_src` is at or below [`VAR_SRC_FLOOR`],
/// or whenever `var_dst <= var_src`, **for every arm** — the zero-preservation
/// invariant (gate H4) that keeps the slot a `Difference` form and the identity
/// vector zero.
#[inline]
pub(crate) fn hf_energy_gain(form: HfGainForm, var_src: f64, var_dst: f64) -> f64 {
    // `!(x > y)` rather than `x <= y` DELIBERATELY: it is the negation of the
    // shipped `if var_src > 1e-10 { … } else { 0.0 }`, so a NaN `var_src` takes
    // the zero branch exactly as it always has. `x <= y` would take the other
    // one and divide.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(var_src > VAR_SRC_FLOOR) {
        return 0.0;
    }
    let g = match form {
        // Bit-identical to the expression this replaced: same subtraction of
        // 1.0 from the ratio, same `.max(0.0)`, same order.
        HfGainForm::RatioExcess => (var_dst / var_src - 1.0).max(0.0),
        HfGainForm::SaturatingExcess => {
            let g = (var_dst / var_src - 1.0).max(0.0);
            g / (g + 1.0)
        }
    };
    // The boundedness claim, checked rather than asserted in prose — the same
    // shape `ssim_form`'s scalar tail uses. Debug-only; this is a per-cell
    // finalize, not a per-pixel loop, so the cost is 12 checks per image.
    debug_assert!(
        match form.upper_bound() {
            Some(hi) => (0.0..=hi).contains(&g),
            None => g >= 0.0,
        },
        "{} produced {g} outside its declared bound {:?}",
        form.as_str(),
        form.upper_bound()
    );
    debug_assert_eq!(form.bounds_gain(), form.upper_bound().is_some());
    g
}

/// **`var_loss` / `hf_energy_loss`** — bounded `[0, 1]` by construction, no arm.
///
/// Here so the family's gate has one owner; the arithmetic is unchanged and
/// [`tests::loss_members_are_bounded_by_construction`] states the bound as a
/// property rather than a comment.
#[inline]
pub(crate) fn hf_energy_loss(var_src: f64, var_dst: f64) -> f64 {
    if var_src > VAR_SRC_FLOOR {
        (1.0 - var_dst / var_src).max(0.0)
    } else {
        0.0
    }
}

/// **`tex_loss` / `hf_mag_loss`** — the L1 sibling of [`hf_energy_loss`],
/// likewise bounded `[0, 1]` and likewise without an arm.
#[inline]
pub(crate) fn hf_mag_loss(mad_src: f64, mad_dst: f64) -> f64 {
    if mad_src > VAR_SRC_FLOOR {
        (1.0 - mad_dst / mad_src).max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(var_src, var_dst)` pairs reaching the healthy regime, the gate, and
    /// F17's pathology — the last row is the measured 36,465.7 configuration.
    fn cases() -> Vec<(f64, f64)> {
        let mut v = Vec::new();
        for &b in &[0.0, 1e-12, 1e-10, 1e-9, 1e-4, 1e-2, 0.25, 1.0, 100.0] {
            for &ratio in &[0.0, 0.5, 0.9999, 1.0, 1.0001, 1.5, 2.0, 10.0, 1e3, 36_466.7] {
                v.push((b, b * ratio));
            }
        }
        v.push((1e-9, 3.6e-5)); // var_src just past the gate, real HF in dst
        v
    }

    /// The exact expression the three sites carried, transcribed verbatim —
    /// the control for the extraction.
    fn legacy_gain(var_src: f64, var_dst: f64) -> f64 {
        if var_src > 1e-10 {
            (var_dst / var_src - 1.0).max(0.0)
        } else {
            0.0
        }
    }
    fn legacy_loss(var_src: f64, var_dst: f64) -> f64 {
        if var_src > 1e-10 {
            (1.0 - var_dst / var_src).max(0.0)
        } else {
            0.0
        }
    }

    /// **The extraction gate.** The legacy arm must reproduce the replaced
    /// expression BIT-for-BIT, not merely closely — that is what lets the
    /// three-site rewrite claim to be inert.
    #[test]
    fn legacy_arm_is_bit_identical_to_the_expression_it_replaced() {
        for (b, a) in cases() {
            let got = hf_energy_gain(HfGainForm::RatioExcess, b, a);
            let want = legacy_gain(b, a);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "gain diverged at var_src={b} var_dst={a}: {got} vs {want}"
            );
            let got_l = hf_energy_loss(b, a);
            let want_l = legacy_loss(b, a);
            assert_eq!(
                got_l.to_bits(),
                want_l.to_bits(),
                "loss diverged at var_src={b} var_dst={a}"
            );
            let got_m = hf_mag_loss(b, a);
            assert_eq!(got_m.to_bits(), legacy_loss(b, a).to_bits());
        }
    }

    /// The `loss` members' bound is a property of the algebra, not a clamp —
    /// stated as a test so a future edit that breaks it fails loudly.
    #[test]
    fn loss_members_are_bounded_by_construction() {
        for (b, a) in cases() {
            let l = hf_energy_loss(b, a);
            let m = hf_mag_loss(b, a);
            assert!((0.0..=1.0).contains(&l), "hf_energy_loss = {l}");
            assert!((0.0..=1.0).contains(&m), "hf_mag_loss = {m}");
        }
    }

}
