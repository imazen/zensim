//! EXP-CROSS-CODEC-V9 (2026-05-20): shared PCHIP-spline parsing +
//! evaluation for the `zentrain.output_calibration_spline` metadata
//! key. Mirrors `zensim::metric` private helpers so every bake-aware
//! binary in `zensim-validate` can apply the calibration without
//! re-implementing the PCHIP math.
//!
//! Payload layout: `[n_knots: u32 LE, n_knots × (x: f32 LE, y: f32
//! LE)]`. Knots must be strictly increasing in x; n_knots >= 2.
//!
//! Bit-exact with `zensim::metric::apply_output_calibration_spline`
//! (asserted by integration test in zensim's regression suite).

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
    let derivs = pchip_compute_derivs(&xs, &ys);
    Some(OutputCalibrationSpline { xs, ys, derivs })
}

/// Apply the spline at the given y_after_pin score. Linear
/// extrapolation outside the knot range using the endpoint slope.
pub fn apply(x: f64, spline: &OutputCalibrationSpline) -> f64 {
    let n = spline.xs.len();
    debug_assert!(n >= 2);
    let xs = spline.xs.as_slice();
    let ys = spline.ys.as_slice();
    let derivs = spline.derivs.as_slice();
    if !x.is_finite() {
        return x;
    }
    if x <= xs[0] {
        // OOD safety net (2026-07-10): parity with zensim/src/metric.rs. Floor
        // the downward extrapolation one calibrated-dial-range below the bottom
        // knot. Legitimate content never reaches here (the all-quality raw
        // distribution sits at/above xs[0]); this only bounds a pathological OOD
        // raw that slipped past the winsor guard (raw −8.63 previously → a wild
        // ~−1131). Negatives are still allowed (neg-tail intact) — only the wild
        // extreme is clamped. Monotone: `.max(floor)` preserves rank.
        let floor = ys[0] - (ys[n - 1] - ys[0]);
        return (ys[0] + derivs[0] * (x - xs[0])).max(floor);
    }
    if x >= xs[n - 1] {
        // Parity with the product runtime (zensim/src/metric.rs upper
        // extrapolation): capped at <=100 — no score exceeds identical.
        // Uncapped linear here was a REAL divergence (dial p95 artifacts
        // of 300-500 on linear bakes, found 2026-07-04 by the spline
        // extrapolation audit); the "bit-exact" doc claim was false above
        // the top knot. The bottom now carries the OOD floor (see the
        // `x <= xs[0]` branch above) — negatives are still allowed (neg-tail
        // resolution intact), only the wild pathological extreme is bounded;
        // same as product.
        return (ys[n - 1] + derivs[n - 1] * (x - xs[n - 1])).min(100.0);
    }
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
    let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2);
    let h10 = t * (1.0 - t).powi(2);
    let h01 = t.powi(2) * (3.0 - 2.0 * t);
    let h11 = t.powi(2) * (t - 1.0);
    h00 * ys[lo] + h10 * h * derivs[lo] + h01 * ys[hi] + h11 * h * derivs[hi]
}

fn pchip_compute_derivs(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    debug_assert_eq!(ys.len(), n);
    debug_assert!(n >= 2);
    if n == 2 {
        let s = (ys[1] - ys[0]) / (xs[1] - xs[0]);
        return vec![s, s];
    }
    let mut h = Vec::with_capacity(n - 1);
    let mut s = Vec::with_capacity(n - 1);
    for k in 0..n - 1 {
        let hk = xs[k + 1] - xs[k];
        h.push(hk);
        s.push((ys[k + 1] - ys[k]) / hk);
    }
    let mut d = vec![0.0_f64; n];
    for k in 1..n - 1 {
        if s[k - 1] * s[k] <= 0.0 {
            d[k] = 0.0;
        } else {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            d[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k]);
        }
    }
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
