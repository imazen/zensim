//! PU21 — perceptually-uniform encoding for adapting an SDR metric to HDR
//! (Mantiuk & Azimi, *PU21: A novel perceptually uniform encoding for adapting
//! existing quality metrics for HDR*, Picture Coding Symposium 2021).
//!
//! `pu21_encode` maps **absolute luminance** `Y` (cd/m², in `[0.005, 10000]`) to
//! a perceptually-uniform value `V`. The encoding is constrained so that
//! `100 cd/m² → ~256` — i.e. SDR content on a ~100-nit reference display lands
//! in roughly the same `[0, 255]` range the SDR metric was tuned on, so the
//! metric needs no retraining to run on HDR luminance. On zensim's pipeline,
//! PU21 is the HDR-path replacement for the cube-root nonlinearity (see
//! `docs/HDR_PLAN.md` §2 and `transfer.rs` for the display model that produces
//! the absolute-luminance input).
//!
//! Coefficients and the rational-polynomial form are the published values from
//! `gfxdisp/pu21` (BSD-3-Clause, © Rafal Mantiuk); reimplemented here from the
//! specification, not copied. The reference-parity tests pin the `100 → 256`
//! normalization, `encode∘decode` round-trip, and monotonicity.
//!
//! Internal (`pub(crate)`) until the HDR scoring path that consumes it lands.
#![allow(dead_code)]

/// Minimum absolute luminance the encoding is defined over (cd/m²).
pub(crate) const PU21_L_MIN: f32 = 0.005;
/// Maximum absolute luminance the encoding is defined over (cd/m²).
pub(crate) const PU21_L_MAX: f32 = 10000.0;

/// PU21 parameter sets. `BandingGlare` is the recommended default (it models
/// display glare and is least prone to banding artifacts inflating the score).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Pu21Variant {
    /// Minimizes visible banding; no glare model.
    Banding,
    /// Banding + display glare. **Recommended default** (gfxdisp/pu21 default).
    #[default]
    BandingGlare,
    /// Tuned to peak-sensitivity thresholds; no glare model.
    Peaks,
    /// Peaks + display glare.
    PeaksGlare,
}

impl Pu21Variant {
    /// The 7 fitted parameters `[p1..p7]` (gfxdisp/pu21, updated 2020-02-06).
    // Literals carry the full published gfxdisp precision for provenance; f32
    // rounds the trailing digits (the parity test pins the resulting values).
    #[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
    #[inline]
    const fn params(self) -> [f32; 7] {
        match self {
            Self::Banding => [
                1.070_275_3,
                0.408_827_4,
                0.153_224_3,
                0.252_032_6,
                1.063_512_9,
                1.141_150_5,
                521.452_75,
            ],
            Self::BandingGlare => [
                0.353_487_9,
                0.373_465_86,
                8.277_049e-5,
                0.906_256_26,
                0.091_503_03,
                0.909_951_7,
                596.314_8,
            ],
            Self::Peaks => [
                1.043_882_8,
                0.645_949_55,
                0.319_458_42,
                0.374_025_25,
                1.114_783_4,
                1.095_360_4,
                384.921_76,
            ],
            Self::PeaksGlare => [
                816.885_03,
                1479.464_0,
                0.001_253_215_6,
                0.932_963_7,
                0.067_466_44,
                1.573_435_4,
                419.600_64,
            ],
        }
    }
}

/// Encode absolute luminance `y` (cd/m²) to the PU21 perceptually-uniform
/// value. `y` is clamped to `[PU21_L_MIN, PU21_L_MAX]`.
///
/// `V = max( p7·( ((p1 + p2·Y^p4)/(1 + p3·Y^p4))^p5 − p6 ), 0 )`.
#[inline]
pub(crate) fn pu21_encode(y: f32, variant: Pu21Variant) -> f32 {
    let p = variant.params();
    let y = y.clamp(PU21_L_MIN, PU21_L_MAX);
    let yp = y.powf(p[3]);
    let inner = (p[0] + p[1] * yp) / (1.0 + p[2] * yp);
    (p[6] * (inner.powf(p[4]) - p[5])).max(0.0)
}

/// Inverse of [`pu21_encode`]: PU21 value `v` → absolute luminance (cd/m²).
///
/// `V_p = max(V/p7 + p6, 0)^(1/p5);  Y = (max(V_p − p1, 0)/(p2 − p3·V_p))^(1/p4)`.
#[inline]
pub(crate) fn pu21_decode(v: f32, variant: Pu21Variant) -> f32 {
    let p = variant.params();
    let v_p = (v / p[6] + p[5]).max(0.0).powf(1.0 / p[4]);
    let num = (v_p - p[0]).max(0.0);
    let den = p[1] - p[2] * v_p;
    (num / den).powf(1.0 / p[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_close(a: f32, b: f32, rel: f32) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-6)
    }

    #[test]
    fn banding_glare_is_default() {
        assert_eq!(Pu21Variant::default(), Pu21Variant::BandingGlare);
    }

    #[test]
    fn normalization_100_nits_maps_near_256() {
        // The defining constraint of PU21: 100 cd/m² → ~256, so SDR content
        // lands in the [0,255] range the SDR metric expects.
        let v = pu21_encode(100.0, Pu21Variant::BandingGlare);
        assert!((v - 256.0).abs() < 1.5, "encode(100) = {v}, expected ~256");
    }

    #[test]
    fn low_luminance_floors_near_zero() {
        // At the minimum luminance the encoded value is ~0 (clamped).
        let v = pu21_encode(PU21_L_MIN, Pu21Variant::BandingGlare);
        assert!((0.0..1.0).contains(&v), "encode(L_min) = {v}");
        // Sub-minimum clamps, never panics or goes negative.
        assert!(pu21_encode(0.0, Pu21Variant::BandingGlare) >= 0.0);
    }

    #[test]
    fn strictly_monotone_increasing() {
        for &variant in &[
            Pu21Variant::Banding,
            Pu21Variant::BandingGlare,
            Pu21Variant::Peaks,
            Pu21Variant::PeaksGlare,
        ] {
            let mut prev = pu21_encode(PU21_L_MIN, variant);
            // Log-spaced sweep across the full luminance range.
            for i in 1..=200 {
                let y = PU21_L_MIN * (PU21_L_MAX / PU21_L_MIN).powf(i as f32 / 200.0);
                let v = pu21_encode(y, variant);
                assert!(v >= prev, "{variant:?} not monotone at Y={y}: {v} < {prev}");
                prev = v;
            }
        }
    }

    #[test]
    fn encode_decode_round_trip() {
        // decode∘encode recovers luminance across the operating range. f32
        // through two power chains → ~1% relative tolerance is appropriate.
        for &variant in &[Pu21Variant::Banding, Pu21Variant::BandingGlare] {
            for i in 0..=40 {
                let y = 0.01 * (5000.0_f32 / 0.01).powf(i as f32 / 40.0);
                let v = pu21_encode(y, variant);
                let y2 = pu21_decode(v, variant);
                assert!(
                    rel_close(y2, y, 0.01),
                    "{variant:?} round-trip Y={y} → V={v} → {y2}"
                );
            }
        }
    }

    #[test]
    fn peak_luminance_is_bounded() {
        // The encoding tops out in the ~400–700 range at 10000 cd/m².
        let v = pu21_encode(PU21_L_MAX, Pu21Variant::BandingGlare);
        assert!((256.0..1000.0).contains(&v), "encode(L_max) = {v}");
        assert!(v > pu21_encode(100.0, Pu21Variant::BandingGlare));
    }

    /// Cross-crate reference-parity drift-guard. The golden `V` values below are
    /// computed *independently in float64* from the published gfxdisp/pu21
    /// coefficients + formula (not from this Rust code). The IDENTICAL table is
    /// asserted in `zenmetrics`'s `hdr.rs` tests, so the two PU21 copies cannot
    /// silently drift from each other or from the gfxdisp reference — a single
    /// changed coefficient breaks this test in whichever copy diverged. If the
    /// reference coefficients ever legitimately change, regenerate the table in
    /// BOTH places. Generator: `scripts/pu21_golden.py` (gfxdisp float64).
    #[test]
    fn reference_parity_gfxdisp_goldens() {
        // Y sample points (cd/m²), spanning PU21_L_MIN..PU21_L_MAX.
        const YS: [f32; 7] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0];
        // (variant, golden encoded V at each YS point) — gfxdisp float64 reference.
        let goldens: [(Pu21Variant, [f64; 7]); 4] = [
            (
                Pu21Variant::Banding,
                [
                    6.3053, 36.0057, 84.4045, 158.5061, 261.7517, 388.1423, 520.4673,
                ],
            ),
            (
                Pu21Variant::BandingGlare,
                [
                    0.3722, 5.7171, 36.5439, 123.6475, 256.3839, 420.0969, 595.3939,
                ],
            ),
            (
                Pu21Variant::Peaks,
                [
                    5.0060, 32.6568, 85.5420, 167.5246, 260.7250, 335.6947, 380.9853,
                ],
            ),
            (
                Pu21Variant::PeaksGlare,
                [
                    0.5133, 8.0104, 47.0090, 136.2603, 252.2985, 359.6225, 407.5066,
                ],
            ),
        ];
        for (variant, expected) in goldens {
            for (&y, &want) in YS.iter().zip(expected.iter()) {
                let got = pu21_encode(y, variant) as f64;
                // abs term covers f32 cancellation near the p6 subtraction at low
                // Y; rel term covers power-chain error at high Y. A coefficient
                // typo shifts values far beyond this — the guard still fires.
                let tol = 0.1 + 5e-3 * want;
                assert!(
                    (got - want).abs() <= tol,
                    "{variant:?} encode({y}) = {got}, gfxdisp ref {want} (tol {tol})"
                );
            }
        }
    }
}
