//! EXP-CROSS-CODEC-V9 (2026-05-20): shared PCHIP-spline parsing +
//! evaluation for the `zentrain.output_calibration_spline` metadata
//! key. Mirrors `zensim::metric` private helpers so every bake-aware
//! binary in `zensim-validate` can apply the calibration without
//! re-implementing the PCHIP math.
//!
//! Payload layout: `[n_knots: u32 LE, n_knots × (x: f32 LE, y: f32
//! LE)]`. Knots must be strictly increasing in x; n_knots >= 2.
//!
//! **Bit-exact with `zensim::metric::apply_output_calibration_spline` BY
//! CONSTRUCTION** since 2026-09-06: both the derivative solve and the
//! evaluation delegate to `zensim::score_math`, the one owner. Before that,
//! the claim on this line was prose with no test behind it, and it was FALSE
//! in the interior segment — see [`apply`]. What remains local is the wire
//! format (parse/extract) and [`fit_monotone_spline`], which the product
//! runtime has no counterpart for.

use zenpredict::Model;

const KEY: &str = "zentrain.output_calibration_spline";

/// Parsed PCHIP spline. `xs.len() == ys.len() == derivs.len()`.
#[derive(Clone, Debug)]
pub struct OutputCalibrationSpline {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub derivs: Vec<f64>,
}

/// Read + parse the spline from a bake's metadata. Returns `None` if
/// the key is absent or the payload is malformed.
pub fn extract(model: &Model) -> Option<OutputCalibrationSpline> {
    let md = model.metadata();
    let entry = md.get(KEY)?;
    parse_payload(entry.value)
}

/// Parse the raw payload bytes (used by tests + callers that already
/// have the metadata blob).
pub fn parse_payload(payload: &[u8]) -> Option<OutputCalibrationSpline> {
    if payload.len() < 4 {
        return None;
    }
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if n < 2 {
        return None;
    }
    let expected = 4 + 8 * n;
    if payload.len() != expected {
        return None;
    }
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let off = 4 + i * 8;
        let x = f32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]) as f64;
        let y = f32::from_le_bytes([
            payload[off + 4],
            payload[off + 5],
            payload[off + 6],
            payload[off + 7],
        ]) as f64;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        xs.push(x);
        ys.push(y);
    }
    // Strictly increasing x. (All values were verified finite above, so
    // `<=` is an exact rewrite of the previous NaN-aware `!(a > b)`.)
    for i in 1..n {
        if xs[i] <= xs[i - 1] {
            return None;
        }
    }
    let derivs = zensim::score_math::pchip_derivs(&xs, &ys);
    Some(OutputCalibrationSpline { xs, ys, derivs })
}

/// Apply the spline at `x`.
///
/// **Delegates to [`zensim::score_math::pchip_eval_capped`], the ONE owner** —
/// the same function `zensim::metric::apply_output_calibration_spline` calls.
/// The lower extrapolation is floored one calibrated-dial-range below the
/// bottom knot (negatives stay reachable — the neg-tail is intact); the upper
/// extrapolation AND the interior are both capped at 100.
///
/// # The interior cap was a divergence — REAL in code, LATENT in practice
///
/// The 2026-07-04 spline-extrapolation audit fixed the uncapped upper
/// *extrapolation* here (dial-p95 artifacts of 300-500 on linear bakes) and
/// stopped one branch short: the *interior* Hermite segment kept no cap while
/// the product runtime has always had one, so this crate's tooling could
/// publish a score `Zensim::compute` reports as exactly 100 — the same shape
/// of defect the audit closed, in the branch it did not reach.
///
/// **The reachable trigger is a KNOT whose `y` exceeds 100**, which
/// [`parse_payload`] permits (it bounds `x` strictly increasing and both
/// coordinates finite, and bounds `y` not at all). It is **not** Hermite
/// overshoot: the Fritsch-Carlson rule keeps the interpolant inside its
/// bracketing knots, which a draft of the gate discovered by building an
/// "overshoot" fixture and failing its own vacuity guard.
///
/// **MEASURED over all 49 bakes on disk** (`zensim/weights`, its `archive/`,
/// `zensim-experimental/weights`): **0 declare such a knot**, so no published
/// verdict ever took that branch. Delegating removes the class — there is no
/// second implementation left to stop short.
///
/// **A shared asymmetry, deliberately unchanged**: the lower branch's floor is
/// `ys[0] − (ys[n−1] − ys[0])`, which is a floor only for an INCREASING
/// spline. On a decreasing one the `.max` returns exactly 200.0 at
/// `x == xs[0]` — seven `zensim-experimental` bakes do. Identical in both
/// implementations, so not an owner divergence; changing it would move product
/// numbers, and no shipped profile has a decreasing spline.
pub fn apply(x: f64, spline: &OutputCalibrationSpline) -> f64 {
    zensim::score_math::pchip_eval_capped(x, &spline.xs, &spline.ys, &spline.derivs)
}

/// Fit a monotone PCHIP spline mapping `predictions` → `targets`.
///
/// The spline maps the model's raw output scale to a target scale
/// (e.g., AIC-3 JND units). Knots are placed at quantile bins of the
/// predictions; within each bin the median (prediction, target) pair
/// becomes a knot. Monotonicity is enforced by removing knots that
/// break the trend (determined by the sign of the overall correlation).
///
/// Returns the binary payload for the `zentrain.output_calibration_spline`
/// metadata entry, or `None` if there are fewer than 2 valid knots.
pub fn fit_monotone_spline(predictions: &[f64], targets: &[f64], n_bins: usize) -> Option<Vec<u8>> {
    let n = predictions.len().min(targets.len());
    if n < 4 || n_bins < 2 {
        return None;
    }

    // Determine direction: is the mapping increasing or decreasing?
    let mean_p: f64 = predictions.iter().take(n).sum::<f64>() / n as f64;
    let mean_t: f64 = targets.iter().take(n).sum::<f64>() / n as f64;
    let cov: f64 = predictions
        .iter()
        .zip(targets.iter())
        .take(n)
        .map(|(&p, &t)| (p - mean_p) * (t - mean_t))
        .sum();
    let decreasing = cov < 0.0;

    // Sort indices by prediction value.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| predictions[a].total_cmp(&predictions[b]));

    // Split into n_bins quantile bins.
    let bin_size = n.div_ceil(n_bins);
    let mut raw_knots: Vec<(f64, f64)> = Vec::with_capacity(n_bins);
    for start in (0..n).step_by(bin_size) {
        let end = (start + bin_size).min(n);
        let bin_indices = &indices[start..end];
        if bin_indices.is_empty() {
            continue;
        }
        // Median prediction and target within the bin.
        let mut preds_bin: Vec<f64> = bin_indices.iter().map(|&i| predictions[i]).collect();
        let mut targs_bin: Vec<f64> = bin_indices.iter().map(|&i| targets[i]).collect();
        preds_bin.sort_by(|a, b| a.total_cmp(b));
        targs_bin.sort_by(|a, b| a.total_cmp(b));
        let mid = preds_bin.len() / 2;
        raw_knots.push((preds_bin[mid], targs_bin[mid]));
    }

    // Enforce strict monotonicity by removing violating knots.
    // Keep only knots where x is strictly increasing AND y follows
    // the expected direction (decreasing for negative correlation,
    // increasing for positive).
    let mut knots: Vec<(f64, f64)> = Vec::with_capacity(raw_knots.len());
    knots.push(raw_knots[0]);
    for &(x, y) in &raw_knots[1..] {
        let (last_x, last_y) = *knots.last().unwrap();
        if x <= last_x + 1e-6 {
            continue;
        }
        let y_ok = if decreasing { y < last_y } else { y > last_y };
        if y_ok {
            knots.push((x, y));
        }
    }

    if knots.len() < 2 {
        return None;
    }

    // Serialize payload: u32 n_knots + n_knots × (f32 x, f32 y)
    let nk = knots.len();
    let mut payload = Vec::with_capacity(4 + 8 * nk);
    payload.extend_from_slice(&(nk as u32).to_le_bytes());
    for &(x, y) in &knots {
        payload.extend_from_slice(&(x as f32).to_le_bytes());
        payload.extend_from_slice(&(y as f32).to_le_bytes());
    }

    // Verify the payload parses correctly.
    parse_payload(&payload)?;

    Some(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trip_minimal() {
        // 2 knots: (0, 0), (100, 100) → identity-ish
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(2u32).to_le_bytes());
        bytes.extend_from_slice(&(0.0f32).to_le_bytes());
        bytes.extend_from_slice(&(0.0f32).to_le_bytes());
        bytes.extend_from_slice(&(100.0f32).to_le_bytes());
        bytes.extend_from_slice(&(100.0f32).to_le_bytes());
        let spline = parse_payload(&bytes).expect("parse");
        assert_eq!(spline.xs, vec![0.0, 100.0]);
        assert_eq!(spline.ys, vec![0.0, 100.0]);
        assert!((apply(50.0, &spline) - 50.0).abs() < 1e-6);
        // Linear extrapolation below 0:
        assert!((apply(-10.0, &spline) - (-10.0)).abs() < 1e-6);
        // Above the top knot: capped at 100 for parity with the product
        // runtime (zensim/src/metric.rs upper extrapolation). The previous
        // expectation (110.0, uncapped) enshrined a measured divergence
        // from the module's own "bit-exact with zensim::metric" contract —
        // corrected 2026-07-04 (spline extrapolation audit).
        assert!((apply(110.0, &spline) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn parse_rejects_non_monotonic_x() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(2u32).to_le_bytes());
        bytes.extend_from_slice(&(100.0f32).to_le_bytes());
        bytes.extend_from_slice(&(0.0f32).to_le_bytes());
        bytes.extend_from_slice(&(0.0f32).to_le_bytes());
        bytes.extend_from_slice(&(100.0f32).to_le_bytes());
        assert!(parse_payload(&bytes).is_none());
    }

    #[test]
    fn parse_rejects_truncated() {
        let bytes = vec![0u8; 3];
        assert!(parse_payload(&bytes).is_none());
    }

    #[test]
    fn pchip_three_knot_monotone() {
        // (0, 0), (50, 60), (100, 100) — checks monotonicity is preserved.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(3u32).to_le_bytes());
        for (x, y) in [(0.0f32, 0.0f32), (50.0, 60.0), (100.0, 100.0)] {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
        }
        let spline = parse_payload(&bytes).expect("parse");
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=200 {
            let x = i as f64 * 0.5;
            let y = apply(x, &spline);
            assert!(y >= prev - 1e-9, "non-monotone at x={x}: {prev} → {y}");
            prev = y;
        }
        // Knot exactness.
        assert!((apply(0.0, &spline) - 0.0).abs() < 1e-6);
        assert!((apply(50.0, &spline) - 60.0).abs() < 1e-6);
        assert!((apply(100.0, &spline) - 100.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod extrapolation_parity_tests {
    use super::*;

    #[test]
    fn upper_extrapolation_caps_at_100_like_product_runtime() {
        // 3-knot payload: count header + (x, y) f32 pairs, y topping at 90.
        let mut payload = (3u32).to_le_bytes().to_vec();
        for (x, y) in [(0.0f32, 10.0f32), (1.0, 50.0), (2.0, 90.0)] {
            payload.extend_from_slice(&x.to_le_bytes());
            payload.extend_from_slice(&y.to_le_bytes());
        }
        let sp = parse_payload(&payload).expect("valid payload");
        let far = apply(1000.0, &sp);
        assert!(
            far <= 100.0,
            "upper extrapolation must cap at 100, got {far}"
        );
        // Bottom is linear until the OOD floor (one dial-range below the
        // bottom knot); still well below 10 for a far-negative input, and
        // negatives above the floor are preserved (neg-tail intact).
        assert!(apply(-1000.0, &sp) < 10.0);
    }

    #[test]
    fn lower_extrapolation_floored_as_ood_safety_net() {
        // Identity-ish spline (0,0)-(100,100): floor = ys[0] - (ys[n-1]-ys[0])
        // = 0 - 100 = -100. Modest negatives extrapolate linearly (neg-tail),
        // but a pathological far-negative raw is floored, not wild.
        let mut bytes = (2u32).to_le_bytes().to_vec();
        for (x, y) in [(0.0f32, 0.0f32), (100.0, 100.0)] {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
        }
        let sp = parse_payload(&bytes).expect("parse");
        // Legit / modest negatives: linear, unchanged (above the -100 floor).
        assert!((apply(-10.0, &sp) - (-10.0)).abs() < 1e-6);
        assert!((apply(-90.0, &sp) - (-90.0)).abs() < 1e-6);
        // Pathological far-negative (an OOD raw that slipped the winsor guard):
        // floored at -100, NOT the wild linear value (-1e6).
        assert!(
            (apply(-1.0e6, &sp) - (-100.0)).abs() < 1e-6,
            "must floor at -100"
        );
        assert!((apply(-8.63, &sp) - (-8.63)).abs() < 1e-6); // real f155 offender raw stays linear
        // Monotone across the floor transition.
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=200 {
            let x = -200.0 + i as f64; // -200 .. 0
            let y = apply(x, &sp);
            assert!(y >= prev - 1e-9, "non-monotone at x={x}: {prev} → {y}");
            prev = y;
        }
    }
}
