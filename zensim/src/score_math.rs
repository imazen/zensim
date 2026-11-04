//! Legacy linear score mapping, shared by the preview scorers.
use crate::det_math::{DetPow, PowForm};

pub fn distance_to_score_mapped(raw_distance: f64, a: f64, b: f64, form: PowForm) -> f64 {
    if raw_distance <= 0.0 {
        100.0
    } else {
        100.0 - a * raw_distance.det_powf(b, form)
    }
}

pub fn pchip_derivs(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    assert_eq!(ys.len(), n, "pchip_derivs: xs/ys length mismatch");
    assert!(n >= 2, "pchip_derivs: needs at least 2 knots");
    if n == 2 {
        let s = (ys[1] - ys[0]) / (xs[1] - xs[0]);
        return vec![s, s];
    }
    // Per-segment slopes h_k = (y_{k+1} - y_k) / (x_{k+1} - x_k).
    let mut h = Vec::with_capacity(n - 1);
    let mut s = Vec::with_capacity(n - 1);
    for k in 0..n - 1 {
        let hk = xs[k + 1] - xs[k];
        h.push(hk);
        s.push((ys[k + 1] - ys[k]) / hk);
    }
    let mut d = vec![0.0_f64; n];
    // Interior: weighted harmonic mean when adjacent slopes share sign,
    // else 0 (extremum).
    for k in 1..n - 1 {
        if s[k - 1] * s[k] <= 0.0 {
            d[k] = 0.0;
        } else {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            d[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k]);
        }
    }
    // Endpoints — three-point estimate, clamped to preserve mono.
    d[0] = pchip_endpoint(h[0], h[1], s[0], s[1]);
    d[n - 1] = pchip_endpoint(h[n - 2], h[n - 3], s[n - 2], s[n - 3]);
    d
}

fn pchip_endpoint(h0: f64, h1: f64, s0: f64, s1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * s0 - h0 * s1) / (h0 + h1);
    if d * s0 <= 0.0 {
        0.0
    } else if s0 * s1 <= 0.0 && d.abs() > 3.0 * s0.abs() {
        3.0 * s0
    } else {
        d
    }
}

/// Evaluate the output-calibration PCHIP spline at `x`, capped at `≤ 100` on
/// the upper side and FLOORED one calibrated-dial-range below the bottom knot.
///
/// A perceptual score can never exceed 100 (nothing is more similar than
/// identical), so **every** branch — lower extrapolation, interior segment,
/// upper extrapolation — is capped. The lower side is allowed to go NEGATIVE
/// (an input more dissimilar than the worst codec output is meaningful signal,
/// not a tie at zero) but is floored at `ys[0] − (ys[n−1] − ys[0])` as an OOD
/// safety net for a pathological raw that slipped past the winsor guard.
/// Monotone: `min`/`max` with a constant preserves rank.
///
/// **The interior cap was a divergence — real in code, latent in practice.**
/// The product runtime has always capped the interior; the validate-side
/// mirror capped only the upper extrapolation (the 2026-07-04 fix stopped one
/// branch short), so `bake_verdict` could report a score the shipped runtime
/// reports as exactly 100. The reachable trigger is a KNOT above 100 — which
/// the wire format permits and nothing rejects — not Hermite overshoot, which
/// the Fritsch-Carlson rule forbids. Measured: 0 of the 49 bakes on disk
/// declare one, so no published number moved.
///
/// **A known asymmetry, deliberately unchanged.** The lower branch's
/// `floor = ys[0] − (ys[n−1] − ys[0])` is a FLOOR only for an increasing
/// spline. On a DECREASING one it lands above `ys[0]` and the `.max` makes it
/// a hard value: seven `zensim-experimental` bakes return exactly 200.0 at
/// `x == xs[0]`. That behaviour is IDENTICAL in both implementations, so it
/// is not an owner divergence and changing it would move product numbers —
/// recorded here, not fixed here. No shipped profile has a decreasing spline.
///
/// # Panics
///
/// Panics if the three arrays disagree in length or hold fewer than 2 knots —
/// parse-time invariants of every caller.
pub fn pchip_eval_capped(x: f64, xs: &[f64], ys: &[f64], derivs: &[f64]) -> f64 {
    let n = xs.len();
    assert!(n >= 2, "pchip_eval_capped: needs at least 2 knots");
    assert_eq!(ys.len(), n, "pchip_eval_capped: xs/ys length mismatch");
    assert_eq!(
        derivs.len(),
        n,
        "pchip_eval_capped: xs/derivs length mismatch"
    );
    if !x.is_finite() {
        return x;
    }
    if x <= xs[0] {
        let floor = ys[0] - (ys[n - 1] - ys[0]);
        return (ys[0] + derivs[0] * (x - xs[0])).max(floor);
    }
    if x >= xs[n - 1] {
        return (ys[n - 1] + derivs[n - 1] * (x - xs[n - 1])).min(100.0);
    }
    // Find the segment via binary search.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let h = xs[hi] - xs[lo];
    let t = (x - xs[lo]) / h;
    // Cubic Hermite basis on [0, 1].
    let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2);
    let h10 = t * (1.0 - t).powi(2);
    let h01 = t.powi(2) * (3.0 - 2.0 * t);
    let h11 = t.powi(2) * (t - 1.0);
    (h00 * ys[lo] + h10 * h * derivs[lo] + h01 * ys[hi] + h11 * h * derivs[hi]).min(100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_mapping_is_exactly_a_hundred_at_or_below_zero() {
        for form in [PowForm::LibmPowf, PowForm::PureRust] {
            assert_eq!(distance_to_score_mapped(0.0, 18.0, 0.7, form), 100.0);
            assert_eq!(distance_to_score_mapped(-1.0, 18.0, 0.7, form), 100.0);
            assert_eq!(distance_to_score_mapped(-0.0, 18.0, 0.7, form), 100.0);
            // Unclamped below zero on purpose.
            assert!(distance_to_score_mapped(1e6, 18.0, 0.7, form) < 0.0);
        }
    }

    /// The interior cap that the validate-side mirror was missing.
    ///
    /// The reachable trigger is a KNOT above 100 — the wire format permits one
    /// and nothing rejects it. An earlier draft of this test tried to build a
    /// Hermite OVERSHOOT instead and failed its own vacuity guard (max 99.5 on
    /// knots 0/99/99.5), which is the Fritsch-Carlson monotonicity guarantee
    /// showing up as a test failure: the interpolant cannot leave its
    /// bracketing knots. The guard is kept below for the same reason.
    #[test]
    fn the_pchip_interior_is_capped_at_one_hundred() {
        let xs = [0.0_f64, 50.0, 100.0];
        let ys = [0.0_f64, 80.0, 130.0];
        let d = pchip_derivs(&xs, &ys);
        // The old validate-side interior branch, verbatim and uncapped.
        let uncapped = |x: f64, lo: usize, hi: usize| {
            let h = xs[hi] - xs[lo];
            let t = (x - xs[lo]) / h;
            let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2);
            let h10 = t * (1.0 - t).powi(2);
            let h01 = t.powi(2) * (3.0 - 2.0 * t);
            let h11 = t.powi(2) * (t - 1.0);
            h00 * ys[lo] + h10 * h * d[lo] + h01 * ys[hi] + h11 * h * d[hi]
        };
        let worst = (1..1000)
            .map(|i| uncapped(50.0 + i as f64 * 0.05, 1, 2))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            worst > 100.0,
            "fixture no longer exceeds 100 in the interior (max {worst}); \
             the cap test would be vacuous"
        );
        for i in 1..1000 {
            let x = 50.0 + i as f64 * 0.05;
            let y = pchip_eval_capped(x, &xs, &ys, &d);
            assert!(y <= 100.0, "interior above 100 at x={x}: {y}");
        }
    }

    /// The Fritsch-Carlson rule keeps the interpolant inside its bracketing
    /// knots — pinned, because the interior-cap test's shape depends on it and
    /// a future change to `pchip_derivs` that broke it would otherwise be
    /// invisible.
    #[test]
    fn pchip_never_leaves_its_bracketing_knots_on_monotone_data() {
        let xs = [0.0_f64, 10.0, 10.5, 40.0, 41.0, 90.0];
        let ys = [0.0_f64, 99.0, 99.2, 99.4, 99.5, 99.9];
        let d = pchip_derivs(&xs, &ys);
        for seg in 0..xs.len() - 1 {
            for i in 0..=200 {
                let x = xs[seg] + (xs[seg + 1] - xs[seg]) * (i as f64 / 200.0);
                let y = pchip_eval_capped(x, &xs, &ys, &d);
                assert!(
                    y >= ys[seg] - 1e-9 && y <= ys[seg + 1] + 1e-9,
                    "segment {seg} left [{}, {}] at x={x}: {y}",
                    ys[seg],
                    ys[seg + 1]
                );
            }
        }
    }

    #[test]
    fn the_pchip_lower_tail_stays_negative_but_floored() {
        let xs = [0.0_f64, 50.0, 100.0];
        let ys = [10.0_f64, 55.0, 100.0];
        let d = pchip_derivs(&xs, &ys);
        let floor = ys[0] - (ys[2] - ys[0]);
        let deep = pchip_eval_capped(-1.0e6, &xs, &ys, &d);
        assert_eq!(deep, floor, "lower tail is not floored at {floor}");
        // A modest OOD input still goes negative — the neg-tail is intact.
        assert!(pchip_eval_capped(-20.0, &xs, &ys, &d) < 0.0);
    }

    #[test]
    fn pchip_passes_through_its_knots() {
        let xs = [4.5_f64, 7.2, 15.3, 28.1, 38.7, 55.0, 68.4, 88.9];
        let ys = [0.0_f64, 10.0, 30.0, 50.0, 60.0, 80.0, 90.0, 100.0];
        let d = pchip_derivs(&xs, &ys);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let got = pchip_eval_capped(*x, &xs, &ys, &d);
            assert!((got - *y).abs() < 1e-9, "knot {x}: {got} != {y}");
        }
    }
}
