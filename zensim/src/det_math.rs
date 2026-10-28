//! Root and power operations used by feature pooling and score mapping.
//!
//! Revision 1 retains platform `powf` roots. Canonical feature revisions use
//! nested square roots. Callers pass a form explicitly when the arithmetic
//! belongs to a particular revision.

use crate::feature_defs::FormulaRevision;

/// How a pooled 4th- or 8th-root moment is evaluated.
///
/// The revision axis is owned by [`crate::feature_defs`]; this enum is the
/// arithmetic that a revision SELECTS — the same shape as
/// [`crate::ssim_form::SsimLumaForm`] and for the same reason: a form with
/// many owners cannot be revised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum RootForm {
    /// `x.powf(0.25)` / `x.powf(0.125)` — what every stored table was built
    /// with, and **libc-dependent**. Shipped.
    #[default]
    LibmPowf,
    NestedSqrt,
}

impl RootForm {
    /// Resolve a per-comparison revision, defaulting to revision 1.
    pub(crate) fn at_revision(revision: Option<crate::feature_defs::FormulaRevision>) -> Self {
        Self::for_revision(revision.unwrap_or(FormulaRevision::Rev1))
    }

    /// The form a registered revision selects.
    pub(crate) const fn for_revision(rev: FormulaRevision) -> Self {
        match rev {
            FormulaRevision::Rev1 => Self::LibmPowf,
            FormulaRevision::Rev5 | FormulaRevision::Rev7 => Self::NestedSqrt,
        }
    }
}

pub(crate) trait DetRoots: Copy {
    /// `x^(1/4)` for a pooled 4th raw moment.
    ///
    /// The `x == 0.0` guard inside is not a numerical stabiliser — it is what
    /// keeps the two arms' difference PURELY a rounding question. IEEE
    /// `pow(±0, 0.25)` is `+0`, while `sqrt(-0.0)` is `-0.0`; without the
    /// guard [`RootForm::NestedSqrt`] would differ from
    /// [`RootForm::LibmPowf`] in a *sign bit* on negative zero, which is a
    /// semantic change masquerading as a rounding one. Every call site applies
    /// `.max(0.0)` first, so a negative zero can still arrive here — Rust's
    /// `max` is IEEE `maxNum`, which may return either zero when both are
    /// zero.
    fn quarter_root(self, form: RootForm) -> Self;
    /// `x^(1/8)` for a pooled 8th raw moment. See [`Self::quarter_root`].
    fn eighth_root(self, form: RootForm) -> Self;
}

impl DetRoots for f64 {
    #[inline(always)]
    fn quarter_root(self, form: RootForm) -> Self {
        match form {
            RootForm::LibmPowf => self.powf(0.25),
            RootForm::NestedSqrt if self == 0.0 => 0.0,
            RootForm::NestedSqrt => self.sqrt().sqrt(),
        }
    }
    #[inline(always)]
    fn eighth_root(self, form: RootForm) -> Self {
        match form {
            RootForm::LibmPowf => self.powf(0.125),
            RootForm::NestedSqrt if self == 0.0 => 0.0,
            RootForm::NestedSqrt => self.sqrt().sqrt().sqrt(),
        }
    }
}

impl DetRoots for f32 {
    #[inline(always)]
    fn quarter_root(self, form: RootForm) -> Self {
        match form {
            RootForm::LibmPowf => self.powf(0.25),
            RootForm::NestedSqrt if self == 0.0 => 0.0,
            RootForm::NestedSqrt => self.sqrt().sqrt(),
        }
    }
    #[inline(always)]
    fn eighth_root(self, form: RootForm) -> Self {
        match form {
            RootForm::LibmPowf => self.powf(0.125),
            RootForm::NestedSqrt if self == 0.0 => 0.0,
            RootForm::NestedSqrt => self.sqrt().sqrt().sqrt(),
        }
    }
}

// ============================================================================
// F19 — the SCORE path
// ============================================================================

/// How a non-dyadic power / exponential / logarithm on the **score** path is
/// evaluated.
///
/// [`RootForm`] fixed the FEATURE path, and its fix was *derived*: for the two
/// exponents `1/4` and `1/8` a `sqrt` composition is the unique libm-free
/// answer. **No such derivation exists here.** The score's exponents are
/// `0.7` (every shipped profile's `score_mapping_b`), `0.5979`, `1.2244`,
/// `0.6130` and a bake-supplied p-norm `p` — none of them dyadic, so there is
/// no finite chain of correctly-rounded operations that reproduces them.
/// A pow at these exponents must be an *algorithm*, and the only property we
/// can insist on is that it be **the same algorithm everywhere**.
///
/// # What was rejected, and why — measured, not assumed
///
/// The obvious reuse is `magetypes`' `log2_midp_precise` / `exp2_midp_precise`
/// pair, which the SIMD PU-XYB path already composes into `x^p`. It is not
/// available at the width this path needs, on two counts, both read from
/// magetypes 0.9.28's own source rather than inferred:
///
/// - the `*_midp_precise` family is defined **only** on the `f32x4` / `f32x8`
///   / `f32x16` vector types (`simd/generic/generated/transcendentals_f32x*.rs`).
///   There is no scalar form and no `f64` form. The score path is `f64` end to
///   end — `raw_distance`, `score_mapping_b`, the head reducers — so routing it
///   through an f32 kernel would discard ~29 bits to fix a 1-ULP problem.
/// - the `f64` scalars that *do* exist — `nostd_math::{log2_f64, exp2_f64,
///   powf_f64}` — are documented in that file as **lowp**, "~1 % max relative
///   error", and `powf_f64` is literally `exp2_f64(n * log2_f64(x))` over those
///   lowp pieces. MEASURED here by [`tests::the_lowp_f64_pow_is_far_too_coarse`]:
///   over the score's own domain it errs by up to ~1e13 ULP. A score is a
///   product number; moving it by a percent to stop it moving by an ULP is not
///   a fix.
///
/// # What was chosen — a property, not a polynomial
///
/// [`Self::PureRust`] is `libm::{pow, exp, log2}` — the `libm` crate, which is
/// the pure-Rust port of musl's (fdlibm's) implementations. This is **not a
/// new dependency**: `cargo tree -p zensim -e normal -i libm` already showed
/// it arriving twice before this lane (`num-traits` ← `linear-srgb`, and
/// `zenpredict`). Making it explicit is what changed.
///
/// The property that makes it the answer is structural, and it is checkable
/// rather than asserted:
///
/// - `libm::pow` and `libm::log2` contain **no** `select_implementation!` and
///   **no** `fma` — `pow` is the Sun `e_pow.c` algorithm expressed in
///   `+ - * /` and bit manipulation only. One Rust source, compiled for every
///   target, with no arch dispatch to diverge on. (Checked by grep against
///   libm 0.2.16's own source, not inferred.)
/// - `libm::exp` is the ONE of the three that carries a
///   `select_implementation!`, gated `use_arch_required: x86_no_sse` — the x87
///   80-bit path, reachable only on i586-class targets with no SSE2. Every
///   target this crate ships for (`x86_64`, `i686` — which enables SSE2 by
///   default —, `aarch64`, `wasm32`, macOS, Windows) takes the portable
///   path.
///
/// So the case for this arm is exactly the case [`RootForm`] makes and no
/// more: **determinism, and an error bound we measured**, not accuracy. It is
/// not free — it moves revision-1 score bytes — which is why it is an era item
/// and not a plain fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PowForm {
    /// `f64::powf` / `f64::exp` / `f64::log2` — the platform libm, and
    /// **libc-dependent** by the same mechanism as [`RootForm::LibmPowf`].
    /// What every published verdict and every stored dial value was read at.
    /// Shipped.
    #[default]
    LibmPowf,
    PureRust,
}

impl PowForm {
    /// The form a registered revision selects.
    pub const fn for_revision(rev: FormulaRevision) -> Self {
        match rev {
            FormulaRevision::Rev1 => Self::LibmPowf,
            FormulaRevision::Rev5 | FormulaRevision::Rev7 => Self::PureRust,
        }
    }
}

#[inline]
pub fn active_pow_form() -> PowForm {
    PowForm::for_revision(FormulaRevision::Rev1)
}

/// The score path's transcendentals, as a method chain so a call site reads
/// the same as the `.powf(b)` / `.exp()` / `.log2()` it replaces.
///
/// A trait rather than free functions for the same reason [`DetRoots`] is one:
/// keeping the call shape means the diff that introduced the owner is a
/// mechanical substitution that cannot silently reassociate the expression
/// feeding it.
pub trait DetPow: Copy {
    /// `self ^ n`, for the non-dyadic exponents the score mapping uses.
    fn det_powf(self, n: Self, form: PowForm) -> Self;
    /// `e ^ self` — the bounded squash and every head sigmoid.
    fn det_exp(self, form: PowForm) -> Self;
    /// `log2(self)` — the MLP size axes.
    fn det_log2(self, form: PowForm) -> Self;
}

impl DetPow for f64 {
    #[inline(always)]
    fn det_powf(self, n: Self, form: PowForm) -> Self {
        match form {
            PowForm::LibmPowf => self.powf(n),
            PowForm::PureRust => libm::pow(self, n),
        }
    }
    #[inline(always)]
    fn det_exp(self, form: PowForm) -> Self {
        match form {
            PowForm::LibmPowf => self.exp(),
            PowForm::PureRust => libm::exp(self),
        }
    }
    #[inline(always)]
    fn det_log2(self, form: PowForm) -> Self {
        match form {
            PowForm::LibmPowf => self.log2(),
            PowForm::PureRust => libm::log2(self),
        }
    }
}

/// The seed of [`cbrt_newton_f32`]: `MAGIC − bits(x)/3` approximates the bits
/// of `x^(−1/3)` to about five bits (the classic reciprocal-root bit trick).
pub(crate) const CBRT_NEWTON_MAGIC: i32 = 0x54a2_fa8c;

#[inline(always)]
pub(crate) fn cbrt_newton_f32(x: f32) -> f32 {
    let a = x.abs();
    let third = 1.0f32 / 3.0;
    let q = ((a.to_bits() as i32) as f32 * third) as i32;
    let r = f32::from_bits((CBRT_NEWTON_MAGIC - q) as u32);
    let t = a * r;
    let e = (-t).mul_add(r * r, 1.0);
    let r = (r * e).mul_add(third, r);
    let y = (a * r) * r;
    let y2 = y * y;
    let res = (-y2).mul_add(y, a);
    let y = y.mul_add(res / (y2 + y2).mul_add(y, a), y);
    if x == 0.0 {
        x
    } else {
        f32::from_bits(y.to_bits() | (x.to_bits() & 0x8000_0000))
    }
}

/// [`cbrt_newton_x16`] for inputs that are `+0` or positive (the opsin mix
/// after its `max(0)`): `|x|` is `x` and the sign bit is clear, so the absolute
/// value and the sign restore are dropped and every lane has
/// [`cbrt_newton_x16`]'s bits; `+0` still returns itself.
#[inline(always)]
pub(crate) fn cbrt_newton_nonneg_x16<T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    a: magetypes::simd::generic::f32x16<T>,
) -> magetypes::simd::generic::f32x16<T> {
    use magetypes::simd::generic::{f32x16, i32x16};
    let third = f32x16::splat_t(token, 1.0 / 3.0);
    let one = f32x16::splat_t(token, 1.0);
    let q = (f32x16::from_i32_t(token, a.bitcast_to_i32()) * third).to_i32();
    let r = f32x16::from_i32_bitcast_t(token, i32x16::splat_t(token, CBRT_NEWTON_MAGIC) - q);
    let t = a * r;
    let e = (-t).mul_add(r * r, one);
    let r = (r * e).mul_add(third, r);
    let y = (a * r) * r;
    let y2 = y * y;
    let res = (-y2).mul_add(y, a);
    let y = y.mul_add(res / (y2 + y2).mul_add(y, a), y);
    let zero = a.simd_eq(f32x16::zero_t(token));
    f32x16::blend(zero, a, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Revision 7's cube root: within 3 ulp of the correctly rounded value on
    /// every 101st positive normal bit pattern below `2^126` (about 21 million
    /// inputs), odd in its sign, exact on ±0; and over every input in `[1, 8)`
    /// correctly rounded at least as often as `cbrt_midp_f32`, the root it
    /// replaces.
    #[test]
    fn cbrt_newton_is_within_three_ulp() {
        let (lo, hi) = (f32::MIN_POSITIVE.to_bits(), 2.0f32.powi(126).to_bits());
        let mut b = lo;
        while b < hi {
            let x = f32::from_bits(b);
            let want = (x as f64).cbrt() as f32;
            let got = cbrt_newton_f32(x);
            assert!(
                got.to_bits().abs_diff(want.to_bits()) <= 3,
                "cbrt_newton({x:e}) = {got:e}, correctly rounded {want:e}"
            );
            assert_eq!(
                cbrt_newton_f32(-x).to_bits(),
                (-got).to_bits(),
                "odd at {x:e}"
            );
            b += 101;
        }
        assert_eq!(cbrt_newton_f32(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(cbrt_newton_f32(-0.0).to_bits(), (-0.0f32).to_bits());
        let (mut newton, mut midp) = (0u64, 0u64);
        for b in 1.0f32.to_bits()..8.0f32.to_bits() {
            let x = f32::from_bits(b);
            let want = ((x as f64).cbrt() as f32).to_bits();
            newton += u64::from(cbrt_newton_f32(x).to_bits() == want);
            midp += u64::from(magetypes::nostd_math::cbrt_midp_f32(x).to_bits() == want);
        }
        assert!(
            newton >= midp,
            "correctly rounded: newton {newton}, midp {midp}"
        );
    }

    /// **THE pinning gate.** Bit-exact expected values for
    /// [`RootForm::NestedSqrt`] on a fixed input set, so a toolchain, libc or
    /// platform swap that moved these would fail loud instead of silently
    /// re-erasing the property this module exists to create.
    ///
    /// It can be pinned at all only *because* the form is libm-free: IEEE-754
    /// requires `sqrt` to be correctly rounded, so every value below is a
    /// mathematical fact about the input, not a fact about the host. The
    /// equivalent table for [`RootForm::LibmPowf`] would be a table of
    /// glibc's answers and would fail on musl — which is the defect.
    ///
    /// `(input, quarter_root_bits, eighth_root_bits)`; the sixth row is the
    /// source record's own glibc-vs-musl witness
    /// (`57076.535008512925 ** 0.25`).
    const PINNED_F64: &[(f64, u64, u64)] = &[
        (0.0, 0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
        (1.0, 0x3ff0_0000_0000_0000, 0x3ff0_0000_0000_0000),
        (16.0, 0x4000_0000_0000_0000, 0x3ff6_a09e_667f_3bcd),
        (256.0, 0x4010_0000_0000_0000, 0x4000_0000_0000_0000),
        (65536.0, 0x4030_0000_0000_0000, 0x4010_0000_0000_0000),
        (
            57076.535008512925,
            0x402e_e9c9_7d99_4c28,
            0x400f_73b1_256f_4538,
        ),
        (1e-12, 0x3f50_624d_d2f1_a9fc, 0x3fa0_30dc_4ea0_3a72),
        (3.7e-9, 0x3f7f_f20f_9c0e_8dab, 0x3fb6_9bb0_3f29_3a50),
        (2.5e-6, 0x3fa4_5be2_e126_ed34, 0x3fc9_8633_3fc1_d857),
        (
            0.00010000000000000002,
            0x3fb9_9999_9999_999a,
            0x3fd4_3d13_6248_490f,
        ),
        (0.0009, 0x3fc6_2b95_86ad_0a22, 0x3fda_a2ad_da4b_4144),
        (
            0.1234567890123,
            0x3fe2_f7e3_6780_dfb5,
            0x3fe8_a318_1be1_c678,
        ),
        (0.48, 0x3fea_a2ad_da4b_4144, 0x3fed_31dc_c6bd_791b),
        (2.0, 0x3ff3_06fe_0a31_b715, 0x3ff1_72b8_3c7d_517b),
        (3.0, 0x3ff5_0ea3_9fcb_f166, 0x3ff2_5af1_3555_28e5),
        (
            1.9999999999999998,
            0x3ff3_06fe_0a31_b715,
            0x3ff1_72b8_3c7d_517b,
        ),
        (
            12345.678901234567,
            0x4025_14f4_303f_91af,
            0x4009_f932_eb34_957d,
        ),
        // the F4 record's worst scanned slot value
        (5814302.0, 0x4048_8d6b_c0d2_89bd, 0x401c_07aa_3d08_19e0),
        (
            1.7976931348623157e100,
            0x4523_27fc_58da_0f70,
            0x4288_c240_c4ae_cb14,
        ),
        (4.9e-300, 0x3065_88df_2a31_cca2, 0x382a_403a_0116_5831),
        (1e-8, 0x3f84_7ae1_47ae_147b, 0x3fb9_9999_9999_999a),
        (7.0, 0x3ffa_0675_250e_e9f3, 0x3ff4_67ed_a182_7ac6),
        (1e30, 0x417e_2867_89a0_7f2f, 0x40b5_f769_cae0_7281),
        (
            0.3333333333333333,
            0x3fe8_5092_ed86_a26b,
            0x3feb_e4d9_1460_f6ab,
        ),
    ];

    const PINNED_F32: &[(f32, u32, u32)] = &[
        (0.0, 0x0000_0000, 0x0000_0000),
        (1.0, 0x3f80_0000, 0x3f80_0000),
        (16.0, 0x4000_0000, 0x3fb5_04f3),
        (256.0, 0x4080_0000, 0x4000_0000),
        (65536.0, 0x4180_0000, 0x4080_0000),
        (0.0009, 0x3e31_5cac, 0x3ed5_156f),
        (0.48, 0x3f55_156f, 0x3f69_8ee6),
        (2.0, 0x3f98_37f0, 0x3f8b_95c2),
        (3.0, 0x3fa8_751d, 0x3f92_d78a),
        // shortened from 12345.6789 for clippy::excessive_precision; VERIFIED
        // to round to the same f32 (0x4640e6b7), so the pinned roots are unmoved.
        (12_345.679, 0x4128_a7a1, 0x404f_c997),
        (1e-12, 0x3a83_126f, 0x3d01_86e3),
        (5814302.0, 0x4244_6b5e, 0x40e0_3d52),
        (3.4e38, 0x4f7f_f267, 0x477f_f933),
        (1.2e-38, 0x2fb5_f49a, 0x3798_9c92),
    ];

    #[test]
    fn deterministic_roots_are_bit_pinned_f64() {
        for &(x, q, e) in PINNED_F64 {
            assert_eq!(
                x.quarter_root(RootForm::NestedSqrt).to_bits(),
                q,
                "quarter_root({x:?}) moved — a libm-free form must never change"
            );
            assert_eq!(
                x.eighth_root(RootForm::NestedSqrt).to_bits(),
                e,
                "eighth_root({x:?}) moved"
            );
        }
    }

    #[test]
    fn deterministic_roots_are_bit_pinned_f32() {
        for &(x, q, e) in PINNED_F32 {
            assert_eq!(
                x.quarter_root(RootForm::NestedSqrt).to_bits(),
                q,
                "f32 q({x:?})"
            );
            assert_eq!(
                x.eighth_root(RootForm::NestedSqrt).to_bits(),
                e,
                "f32 e({x:?})"
            );
        }
    }

    /// The form IS the composition — no lookup table, no polynomial, no
    /// libm. Stated as an identity so a future "optimisation" that replaced
    /// it with an approximation fails here.
    #[test]
    fn nested_sqrt_is_exactly_the_sqrt_composition() {
        for &(x, _, _) in PINNED_F64 {
            if x == 0.0 {
                continue;
            }
            assert_eq!(
                x.quarter_root(RootForm::NestedSqrt).to_bits(),
                x.sqrt().sqrt().to_bits()
            );
            assert_eq!(
                x.eighth_root(RootForm::NestedSqrt).to_bits(),
                x.sqrt().sqrt().sqrt().to_bits()
            );
        }
    }

    /// `sqrt` itself is not a second divergence source: the hardware
    /// instruction the compiler emits and magetypes' pure-Rust Goldschmidt
    /// `sqrtf` — which shares no code with any libm — agree to the bit.
    ///
    /// Without this, "we removed the libm call" would be an argument; with
    /// it, it is a measurement.
    #[test]
    fn sqrt_is_not_a_divergence_source() {
        for &(x, _, _) in PINNED_F32 {
            if x == 0.0 {
                continue;
            }
            let via_std = x.quarter_root(RootForm::NestedSqrt);
            let via_pure = magetypes::nostd_math::sqrtf(magetypes::nostd_math::sqrtf(x));
            assert_eq!(
                via_std.to_bits(),
                via_pure.to_bits(),
                "sqrt disagrees with the pure-Rust reference at {x:?}"
            );
        }
    }

    /// Negative zero is a SEMANTIC difference, not a rounding one, and the
    /// guard is what keeps it out of the era's blast radius.
    #[test]
    fn negative_zero_is_positive_zero_in_both_arms() {
        for form in [RootForm::LibmPowf, RootForm::NestedSqrt] {
            assert_eq!((-0.0f64).quarter_root(form).to_bits(), 0.0f64.to_bits());
            assert_eq!((-0.0f64).eighth_root(form).to_bits(), 0.0f64.to_bits());
            assert_eq!((-0.0f32).quarter_root(form).to_bits(), 0.0f32.to_bits());
        }
    }

    /// The era's blast radius has the shape the registry claims: the two arms
    /// differ by **at most one ULP**, never in sign, never in magnitude class.
    ///
    /// This asserts a BOUND, not a value, so it passes against any libm —
    /// which is the point: a test that pinned `powf`'s answers would encode
    /// one libc and fail on the next.
    #[test]
    fn the_two_arms_differ_by_at_most_one_ulp() {
        let mut differ = 0usize;
        for i in 0..20_000u64 {
            // deterministic log-uniform sweep over a pooled 4th moment's range
            let t = i as f64 / 20_000.0;
            let x = (1e-14f64).powf(1.0 - t) * (1e14f64).powf(t);
            let a = x.quarter_root(RootForm::LibmPowf);
            let b = x.quarter_root(RootForm::NestedSqrt);
            assert!(a.is_finite() && b.is_finite());
            assert!(a > 0.0 && b > 0.0);
            let gap = (a.to_bits() as i64 - b.to_bits() as i64).abs();
            assert!(gap <= 1, "x={x:e} a={a:?} b={b:?} gap={gap} ulp");
            differ += usize::from(gap != 0);
        }
        // Reported, not asserted: the rate is a property of the host's libm.
        // On glibc 2.43 this reads ~13 %; on musl it reads a different
        // number, and THAT is the defect this module exists for.
        println!("libm vs deterministic: {differ}/20000 differ by 1 ULP");
    }

    /// The measurement override's two values are the SAME BYTE LENGTH, so an
    /// A/B that varies it cannot also vary the binary's memory layout
    /// (`benchmarks/era2_perf_break_2026-08-31.md` §22.5).
    #[test]
    fn override_values_are_equal_length() {
        assert_eq!("libm".len(), "sqrt".len());
    }

    // ========================================================================
    // F19 — the score path
    // ========================================================================

    /// **The correctly-rounded truth**, independently derived — NOT read back
    /// out of the arm under test.
    ///
    /// Every entry is `float(Decimal(x) ** Decimal(b))` at 60 significant
    /// digits under CPython's `decimal`, which uses no libm at all, then
    /// rounded once to `f64` by CPython's own correctly-rounded conversion.
    /// So this table is a set of mathematical facts about its inputs, and a
    /// test that compares an implementation against it is measuring the
    /// implementation rather than agreeing with it. Generated by
    /// `scripts/det_pow_error_bound.py`'s reference path.
    ///
    /// `(x, b, correctly_rounded_bits_of_x_pow_b)`; the exponents are the
    /// score path's own — 0.7 is `score_mapping_b` on every shipped profile,
    /// then the three `approx_*` fits and a spread of head p-norms.
    ///
    /// One `x` (`3.14159265358979`) is a domain sample that happens to look
    /// like pi truncated to 15 digits — it is NOT `std::f64::consts::PI`
    /// (`3.141592653589793`), and its `correctly_rounded_bits_of_x_pow_b` was
    /// computed for THIS literal's exact bit pattern by the external Python
    /// reference. Swapping in the real constant would silently retarget the
    /// row at a different `x` than the recorded bits describe.
    #[allow(clippy::approx_constant)]
    const POW_TRUTH: &[(f64, f64, u64)] = &[
        (1e-09, 0.7, 0x3ea0_d12a_61d3_698f),
        (1e-06, 0.7, 0x3f10_8a48_76c1_3123),
        (0.001, 0.7, 0x3f80_4491_4f3c_02b2),
        (0.1, 0.7, 0x3fc9_8a13_577c_93c1),
        (0.5, 0.7, 0x3fe3_b2c4_7bff_8329),
        (1.0, 0.7, 0x3ff0_0000_0000_0000),
        (2.0, 0.7, 0x3ff9_fdf8_bcce_533d),
        (7.5, 0.7, 0x4010_6412_353b_355c),
        (30.0, 0.7, 0x4025_a0bf_c14c_ad7c),
        (1000.0, 0.7, 0x405f_791f_6509_fb63),
        (0.25, 0.5979, 0x3fdb_f057_8ae2_ba29),
        (3.14159265358979, 0.5979, 0x3fff_b8ee_8291_b44e),
        (42.0, 1.2244, 0x4058_4a82_cf41_ca80),
        (0.0009, 1.2244, 0x3f28_7347_d473_112f),
        (12.5, 0.613, 0x4012_d036_d662_74b5),
        (1e-12, 0.613, 0x3e67_a6f0_9fd3_2afe),
        (0.75, 2.0, 0x3fe2_0000_0000_0000),
        (1.5, 3.0, 0x400b_0000_0000_0000),
        (0.9, 6.0, 0x3fe1_0190_8e58_1cf8),
        (8.0, 0.3333333333333333, 0x4000_0000_0000_0000),
        (64.0, 0.16666666666666666, 0x4000_0000_0000_0000),
    ];

    /// `(x, correctly_rounded_bits_of_exp(x))` — the bounded squash's
    /// `exp(-x)`, `soft_clamp_score`, the tanh output pin (clamped ±30) and
    /// both head sigmoids (clamped ±20). Same derivation as [`POW_TRUTH`].
    const EXP_TRUTH: &[(f64, u64)] = &[
        (-40.0, 0x3c53_9792_499b_1a24),
        (-30.0, 0x3d3a_56e0_c2ac_7f75),
        (-20.0, 0x3e21_b486_55f3_7267),
        (-7.5, 0x3f42_1f9b_a40f_31d5),
        (-1.0, 0x3fd7_8b56_362c_ef38),
        (-0.25, 0x3fe8_ebef_9eac_820b),
        (0.0, 0x3ff0_0000_0000_0000),
        (0.25, 0x3ff4_8b5e_3c3e_8186),
        (1.0, 0x4005_bf0a_8b14_5769),
        (7.5, 0x409c_402b_6eb1_f6ad),
        (20.0, 0x41bc_eb08_8b68_e804),
        (30.0, 0x42a3_7047_0aec_28ed),
    ];

    /// `(x, correctly_rounded_bits_of_log2(x))` — the four `--mlp-size-axes`
    /// inputs. `2073600` is 1920x1080 and `33177600` is 7680x4320, so the
    /// table spans the real pixel-count range rather than only round powers
    /// of two (where every implementation trivially agrees).
    const LOG2_TRUTH: &[(f64, u64)] = &[
        (1.0, 0x0000_0000_0000_0000),
        (2.0, 0x3ff0_0000_0000_0000),
        (3.0, 0x3ff9_5c01_a39f_bd68),
        (255.0, 0x401f_fa37_c98f_e55f),
        (1024.0, 0x4024_0000_0000_0000),
        (4096.0, 0x4028_0000_0000_0000),
        (2073600.0, 0x4034_fbd4_2b46_5836),
        (33177600.0, 0x4038_fbd4_2b46_5836),
        (1000000000.0, 0x403d_e5b8_eaa8_d7e0),
        (1.5, 0x3fe2_b803_473f_7ad1),
        (0.7, 0xbfe0_7762_2896_7d13),
    ];

    fn ulp_gap(a: u64, b: u64) -> i64 {
        (a as i64 - b as i64).abs()
    }

    /// **THE F19 pinning gate.** The deterministic arm is bit-pinned against
    /// an independently derived correctly-rounded reference, and its error is
    /// bounded at 1 ULP.
    ///
    /// Two claims, deliberately separate:
    ///
    /// 1. **BOUND** — `libm::pow` is within 1 ULP of the truth on every entry.
    ///    That is the accuracy contract, and it is asserted against
    ///    [`POW_TRUTH`], never against `f64::powf`.
    /// 2. **PIN** — where the arm IS correctly rounded it must equal the truth
    ///    exactly, so a toolchain or `libm` bump that moved it fails loud.
    ///    Where it is one ULP off, the bound in (1) is the pin: pinning the
    ///    port's own answer as a magic constant would encode a bug as a
    ///    requirement.
    ///
    /// Unlike [`deterministic_roots_are_bit_pinned_f64`] this cannot pin every
    /// value to the truth, because `x^0.7` has no correctly-rounded closed
    /// form the way `x^(1/4)` does — which is exactly the difference between
    /// [`RootForm`] (derived) and [`PowForm`] (chosen).
    #[test]
    fn deterministic_pow_is_within_one_ulp_of_the_truth() {
        let mut exact = 0usize;
        for &(x, b, truth) in POW_TRUTH {
            let got = x.det_powf(b, PowForm::PureRust).to_bits();
            let gap = ulp_gap(got, truth);
            assert!(
                gap <= 1,
                "libm::pow({x:?}, {b:?}) is {gap} ULP from the 60-digit truth"
            );
            exact += usize::from(gap == 0);
        }
        for &(x, truth) in EXP_TRUTH {
            let gap = ulp_gap(x.det_exp(PowForm::PureRust).to_bits(), truth);
            assert!(gap <= 1, "libm::exp({x:?}) is {gap} ULP from the truth");
        }
        for &(x, truth) in LOG2_TRUTH {
            let gap = ulp_gap(x.det_log2(PowForm::PureRust).to_bits(), truth);
            assert!(gap <= 1, "libm::log2({x:?}) is {gap} ULP from the truth");
        }
        // Reported, not asserted: the exact-hit count is a property of the
        // port, and asserting it would make a future accuracy IMPROVEMENT in
        // `libm` fail this gate.
        println!(
            "libm::pow correctly rounded on {exact}/{} entries",
            POW_TRUTH.len()
        );
    }

    /// The SHIPPED arm is measured against the same truth, so the era's cost
    /// is stated rather than implied: **both arms are within 1 ULP**.
    ///
    /// This is the F19 analogue of [`the_two_arms_differ_by_at_most_one_ulp`]
    /// and it carries the same correction: the deterministic arm is NOT more
    /// accurate. MEASURED over the score's whole domain
    /// (`scripts/det_pow_error_bound.py` on 6,611 probe rows): the two arms
    /// disagree on 523 (7.911 %), and of those the platform libm is nearer
    /// the truth on 520 and the port on 3. The port buys sameness, not
    /// accuracy — and that is the whole case for it.
    #[test]
    fn both_pow_arms_are_within_one_ulp_and_libm_is_not_worse() {
        for &(x, b, truth) in POW_TRUTH {
            let gap = ulp_gap(x.det_powf(b, PowForm::LibmPowf).to_bits(), truth);
            assert!(gap <= 1, "the SHIPPED powf({x:?}, {b:?}) is {gap} ULP out");
        }
        for &(x, truth) in EXP_TRUTH {
            assert!(ulp_gap(x.det_exp(PowForm::LibmPowf).to_bits(), truth) <= 1);
        }
        for &(x, truth) in LOG2_TRUTH {
            assert!(ulp_gap(x.det_log2(PowForm::LibmPowf).to_bits(), truth) <= 1);
        }
    }

    /// **The rejection of the brief's suggested reuse, as a measurement.**
    ///
    /// `magetypes::nostd_math::powf_f64` is `exp2_f64(n * log2_f64(x))` over
    /// that module's own `lowp` pieces, which its source documents as "~1 %
    /// max relative error". A percent is ~1e14 ULP at f64. This asserts the
    /// arm is unusable on the score path rather than leaving the reader to
    /// trust a doc comment — and it asserts a FLOOR on the error, so a future
    /// magetypes release that improved the lowp tier would fail here and
    /// prompt a re-measurement instead of silently making the note stale.
    ///
    /// The single most legible cell: `log2_f64(1.0)` returns `-1.868e-6`
    /// where the answer is exactly `0`. A 1-pixel image dimension would enter
    /// the MLP's size axes as `-1.9e-6` instead of `0`.
    #[test]
    fn the_lowp_f64_pow_is_far_too_coarse() {
        let mut worst_rel = 0.0f64;
        for &(x, b, truth) in POW_TRUTH {
            let t = f64::from_bits(truth);
            let got = magetypes::nostd_math::powf_f64(x, b);
            if t != 0.0 {
                worst_rel = worst_rel.max(((got - t) / t).abs());
            }
        }
        assert!(
            worst_rel > 1e-9,
            "the lowp arm now agrees to {worst_rel:e} relative — magetypes may \
             have gained a precise f64 tier; re-measure before citing this"
        );
        assert_ne!(
            magetypes::nostd_math::log2_f64(1.0),
            0.0,
            "log2_f64(1.0) is now exact — re-measure the lowp rejection"
        );
        println!("magetypes lowp f64 pow: worst relative error {worst_rel:e}");
    }

    /// **The f32 route cannot carry an f64 score, and the failure is not
    /// subtle.** `log2_midp_precise` / `exp2_midp_precise` / `pow_midp_precise`
    /// exist only on `f32x4`/`x8`/`x16`, so any reuse of them would first cast
    /// the score path to f32. This prices the BEST CASE for that — a
    /// perfectly-rounded f32 `powf` — and finds a total loss: at the head
    /// p-norm's `p = 6`, `x = 1e-12` underflows f32 to exactly `0.0` where the
    /// true `x^6` is `1e-72`, which an f64 carries without effort.
    #[test]
    fn any_f32_route_destroys_the_pnorm_tail() {
        let x = 1e-12f64;
        let via_f32 = (x as f32).powf(6.0) as f64;
        assert_eq!(via_f32, 0.0, "f32 must underflow here — that is the point");
        let via_f64 = x.det_powf(6.0, PowForm::PureRust);
        assert!(
            via_f64 > 0.0 && via_f64 < 1e-71,
            "the f64 arm must carry it: got {via_f64:e}"
        );
    }
}

/// REV4VEC gate: the `f32x16` midp methods the canonical leaves now call
/// per lane must be **bit-identical** to the scalar `*_midp_f32` bodies
/// above. Two levels:
///
/// * [`midp_x16_matches_scalar_sampled`] — always-on: ~1.6×10⁷ inputs
///   spanning every f32 special class (±0, denormals, ±inf, NaN,
///   clamp bounds) on every compiled fused tier.
/// * [`midp_x16_matches_scalar_over_all_f32_bits`] — `#[ignore]`d:
///   **all 2³² bit patterns** per function per tier (the gate's "where
///   feasible" bar — ≈10 min under the 8-CPU heavy wrapper).
///
/// Compiled only where a lane-parallel canon tier exists (x86_64 v3/v4/v4x). Other targets — i686, wasm32 and, since
/// 2026-10-03, aarch64 NEON (its vector max/min propagate NaN where the canon's `f32::max`/`min` drop it) — run the scalar
/// canonical body for every element, so there is no vector path to compare.
#[cfg(all(test, target_arch = "x86_64"))]
mod rev4vec_midp_gate {
    use super::*;
    use archmage::SimdToken as _;
    use archmage::magetypes;

    /// [`cbrt_newton_nonneg_x16`] in one tier, for the test below.
    #[magetypes(define(f32x16), v4x, v4, v3, -scalar)]
    fn cbrt_nonneg_vec(token: Token, arr: [f32; 16]) -> [f32; 16] {
        cbrt_newton_nonneg_x16(token, f32x16::from_array_t(token, arr)).to_array()
    }

    /// The opsin cube root ([`cbrt_newton_nonneg_x16`]) has [`cbrt_newton_f32`]'s
    /// bits on every compiled tier for `+0`, subnormals, `[1, 8)` in full and
    /// every 97th positive bit pattern up to `+inf`.
    #[test]
    fn cbrt_newton_nonneg_matches_scalar() {
        type RootFn = Box<dyn Fn([f32; 16]) -> [f32; 16]>;
        let mut tiers: Vec<(&str, RootFn)> = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(t) = archmage::X64V3Token::summon() {
                tiers.push(("v3", Box::new(move |a| cbrt_nonneg_vec_v3(t, a))));
            }
            #[cfg(feature = "avx512")]
            {
                if let Some(t) = archmage::X64V4Token::summon() {
                    tiers.push(("v4", Box::new(move |a| cbrt_nonneg_vec_v4(t, a))));
                }
                if let Some(t) = archmage::X64V4xToken::summon() {
                    tiers.push(("v4x", Box::new(move |a| cbrt_nonneg_vec_v4x(t, a))));
                }
            }
        }
        if tiers.is_empty() {
            return;
        }
        let lo = 1.0f32.to_bits();
        let hi = 8.0f32.to_bits();
        let bits: Vec<u32> = [0u32, 1, 2, 0x007f_ffff]
            .into_iter()
            .chain(lo..hi)
            .chain((0..0x7f80_0000u32).step_by(97))
            .chain([0x7f80_0000])
            .collect();
        for (name, f) in &tiers {
            for chunk in bits.chunks(16) {
                let mut arr = [1.0f32; 16];
                for (a, b) in arr.iter_mut().zip(chunk) {
                    *a = f32::from_bits(*b);
                }
                let got = f(arr);
                for (x, g) in arr.iter().zip(got) {
                    let want = cbrt_newton_f32(*x);
                    assert!(
                        want.to_bits() == g.to_bits() || (want.is_nan() && g.is_nan()),
                        "{name}: cbrt({x:e}) = {g:e}, scalar {want:e}"
                    );
                }
            }
        }
    }
}
