//! Pooled high-frequency loss and gain from variance and absolute-deviation ratios.
//!
//! Revision 1 uses ratio excess for gain; canonical revisions use saturating
//! excess. The source floor decides whether a ratio is formed.

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
        Self::for_revision(revision.unwrap_or(FormulaRevision::Rev1))
    }

    /// The form a registered revision selects.
    pub(crate) const fn for_revision(rev: FormulaRevision) -> Self {
        match rev {
            FormulaRevision::Rev1 => Self::RatioExcess,
            FormulaRevision::Rev5 | FormulaRevision::Rev7 => Self::REV2_HFGAIN,
        }
    }

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

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::RatioExcess => "ratio",
            Self::SaturatingExcess => "satexcess",
        }
    }
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
