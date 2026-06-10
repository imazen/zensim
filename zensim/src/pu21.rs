//! PU21 — perceptually-uniform encoding for adapting an SDR metric to HDR
//! (Mantiuk & Azimi, *PU21*, Picture Coding Symposium 2021).
//!
//! `pu21_encode` maps **absolute luminance** `Y` (cd/m², `[0.005, 10000]`) to a
//! perceptually-uniform value `V`, constrained so `100 cd/m² → ~256` — SDR
//! content on a ~100-nit reference display lands in the `[0, 255]` range the
//! SDR metric was tuned on. On zensim's HDR path PU21 replaces the cube-root
//! nonlinearity (see `docs/HDR_PLAN.md` §2).
//!
//! Only the **banding_glare** parameter set ships: it is the gfxdisp-
//! recommended and paper-measured-best variant (banding model + glare), and
//! the only one any zen scoring path uses. The other three published sets
//! (banding, peaks, peaks_glare) remain available — with full 4-row float64
//! reference goldens — in `zenmetrics-api::hdr`, which is the drift-guard
//! home for the cross-crate coefficient table.
//!
//! Coefficients are the published `gfxdisp/pu21` values (BSD-3-Clause,
//! © Rafal Mantiuk); reimplemented from the specification. The reference-
//! parity test pins them to float64 goldens (generator:
//! `scripts/pu21_golden.py`).

/// Minimum absolute luminance the encoding is defined over (cd/m²).
pub(crate) const PU21_L_MIN: f32 = 0.005;
/// Maximum absolute luminance the encoding is defined over (cd/m²).
pub(crate) const PU21_L_MAX: f32 = 10000.0;

/// The 7 fitted `banding_glare` parameters `[p1..p7]` (gfxdisp/pu21,
/// updated 2020-02-06).
const P: [f32; 7] = [
    0.353_487_9,
    0.373_465_86,
    8.277_049e-5,
    0.906_256_26,
    0.091_503_03,
    0.909_951_7,
    596.314_8,
];

/// Encode absolute luminance `y` (cd/m²) to the PU21 perceptually-uniform
/// value (`banding_glare`). `y` is clamped to `[PU21_L_MIN, PU21_L_MAX]`.
///
/// `V = max( p7·( ((p1 + p2·Y^p4)/(1 + p3·Y^p4))^p5 − p6 ), 0 )`.
#[inline]
pub(crate) fn pu21_encode(y: f32) -> f32 {
    let y = y.clamp(PU21_L_MIN, PU21_L_MAX);
    let yp = y.powf(P[3]);
    let inner = (P[0] + P[1] * yp) / (1.0 + P[2] * yp);
    (P[6] * (inner.powf(P[4]) - P[5])).max(0.0)
}

/// Inverse of [`pu21_encode`]: PU21 value `v` → absolute luminance (cd/m²).
#[inline]
#[allow(dead_code)] // used by tests; kept for symmetry with the encoder
pub(crate) fn pu21_decode(v: f32) -> f32 {
    let v_p = (v / P[6] + P[5]).max(0.0).powf(1.0 / P[4]);
    let num = (v_p - P[0]).max(0.0);
    let den = P[1] - P[2] * v_p;
    (num / den).powf(1.0 / P[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-crate reference-parity drift-guard: float64 gfxdisp goldens
    /// (generator: `scripts/pu21_golden.py`), banding_glare row. The same
    /// table is asserted in zenmetrics-api, ssim2-gpu, and fast-ssim2.
    #[test]
    fn reference_parity_gfxdisp_goldens() {
        const YS: [f32; 7] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0];
        const WANT: [f64; 7] = [
            0.3722, 5.7171, 36.5439, 123.6475, 256.3839, 420.0969, 595.3939,
        ];
        for (&y, &want) in YS.iter().zip(WANT.iter()) {
            let got = pu21_encode(y) as f64;
            let tol = 0.1 + 5e-3 * want;
            assert!(
                (got - want).abs() <= tol,
                "pu21_encode({y}) = {got}, gfxdisp ref {want} (tol {tol})"
            );
        }
    }

    #[test]
    fn normalization_100_nits_maps_near_256() {
        let v = pu21_encode(100.0);
        assert!((v - 256.0).abs() < 1.5, "encode(100) = {v}, expected ~256");
    }

    #[test]
    fn strictly_monotone_increasing() {
        let mut prev = pu21_encode(PU21_L_MIN);
        for i in 1..=200 {
            let y = PU21_L_MIN * (PU21_L_MAX / PU21_L_MIN).powf(i as f32 / 200.0);
            let v = pu21_encode(y);
            assert!(v >= prev, "not monotone at Y={y}: {v} < {prev}");
            prev = v;
        }
    }

    #[test]
    fn encode_decode_round_trip() {
        for i in 0..=40 {
            let y = 0.01 * (5000.0_f32 / 0.01).powf(i as f32 / 40.0);
            let v = pu21_encode(y);
            let y2 = pu21_decode(v);
            let rel = (y2 - y).abs() / y.abs().max(1e-6);
            assert!(rel <= 0.01, "round-trip Y={y} -> V={v} -> {y2}");
        }
    }
}
