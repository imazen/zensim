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
    for i in 1..n {
        if !(xs[i] > xs[i - 1]) {
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
        return ys[0] + derivs[0] * (x - xs[0]);
    }
    if x >= xs[n - 1] {
        return ys[n - 1] + derivs[n - 1] * (x - xs[n - 1]);
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
        // Above 100:
        assert!((apply(110.0, &spline) - 110.0).abs() < 1e-6);
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
