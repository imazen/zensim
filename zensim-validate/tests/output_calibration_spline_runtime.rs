//! EXP-CROSS-CODEC-V9 (2026-05-20): integration regression for the
//! shared output_calibration_spline module.
//!
//! Verifies (a) parse round-trip + (b) PCHIP monotonicity + (c) the
//! known-good case of identity spline through 8 V9 anchor targets.

use zensim_validate::output_calibration_spline as ocs;

#[test]
fn identity_spline_passes_through() {
    let mut payload = Vec::new();
    payload.extend_from_slice(&2u32.to_le_bytes());
    payload.extend_from_slice(&0.0f32.to_le_bytes());
    payload.extend_from_slice(&0.0f32.to_le_bytes());
    payload.extend_from_slice(&100.0f32.to_le_bytes());
    payload.extend_from_slice(&100.0f32.to_le_bytes());
    let spline = ocs::parse_payload(&payload).expect("parse identity");
    for x_i in 0..=100 {
        let x = x_i as f64;
        let y = ocs::apply(x, &spline);
        assert!((x - y).abs() < 1e-6, "identity failed at x={x}: y={y}");
    }
}

#[test]
fn v9_eight_knot_spline_monotone() {
    // The realistic case the calibrator will produce: a network's
    // raw predicted-score distribution mapped to the 8 V9 anchor
    // bands.
    let knots = [
        (4.5f32, 0.0f32),    // network puts worstfloor at ~4.5
        (7.2, 10.0),
        (15.3, 30.0),
        (28.1, 50.0),
        (38.7, 60.0),
        (55.0, 80.0),
        (68.4, 90.0),
        (88.9, 100.0),
    ];
    let mut payload = Vec::new();
    payload.extend_from_slice(&(knots.len() as u32).to_le_bytes());
    for (x, y) in &knots {
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
    }
    let spline = ocs::parse_payload(&payload).expect("parse 8-knot");
    // Sample 1000 points across the full range + margin, verify mono.
    let mut prev = f64::NEG_INFINITY;
    for i in -100..=1100 {
        let x = i as f64 * 0.1;
        let y = ocs::apply(x, &spline);
        assert!(y >= prev - 1e-9, "non-monotone at x={x}: prev={prev} y={y}");
        prev = y;
    }
    // Knot exactness.
    for (x, target_y) in &knots {
        let y = ocs::apply(*x as f64, &spline);
        assert!(
            (y - *target_y as f64).abs() < 1e-4,
            "knot at x={x}: y={y}, expected {target_y}"
        );
    }
    // PCHIP linear extrapolation below first knot.
    let y_low = ocs::apply(0.0, &spline);
    assert!(y_low < 0.0, "extrapolation below first knot should be <0: got {y_low}");
    // Linear extrapolation above last knot.
    let y_high = ocs::apply(95.0, &spline);
    assert!(y_high > 100.0, "extrapolation above last knot should be >100: got {y_high}");
}

#[test]
fn spline_rejects_n_knots_lt_2() {
    let mut payload = Vec::new();
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&0.0f32.to_le_bytes());
    payload.extend_from_slice(&0.0f32.to_le_bytes());
    assert!(ocs::parse_payload(&payload).is_none(), "n=1 should reject");
}
