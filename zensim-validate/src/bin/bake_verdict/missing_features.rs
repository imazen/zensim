//! Equality for a cached feature layout with explicitly uncomputed NaN slots.
//! Consumed inputs must be finite before inference; this comparator preserves
//! missingness and the owner's existing finite-feature tolerance.
pub(super) fn near_equal_rows(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            (x.is_nan() && y.is_nan())
                || (x.is_finite() && y.is_finite() && (x - y).abs() <= epsilon)
        })
}

#[cfg(test)]
mod tests {
    use super::near_equal_rows;

    #[test]
    fn matching_uncomputed_slots_preserve_codec_saturation() {
        assert!(near_equal_rows(
            &[f64::NAN, 1.0, 2.0],
            &[f64::NAN, 1.0, 2.0],
            1e-5
        ));
        assert!(!near_equal_rows(&[f64::NAN, 1.0], &[0.0, 1.0], 1e-5));
        assert!(!near_equal_rows(&[0.0, 1.0], &[f64::NAN, 1.0], 1e-5));
        assert!(!near_equal_rows(&[f64::INFINITY], &[f64::INFINITY], 1e-5));
    }

    #[test]
    fn finite_tolerance_and_layout_remain_unchanged() {
        assert!(near_equal_rows(
            &[0.0, f64::NAN],
            &[0.999e-5, f64::NAN],
            1e-5
        ));
        assert!(!near_equal_rows(
            &[0.0, f64::NAN],
            &[1.001e-5, f64::NAN],
            1e-5
        ));
        assert!(!near_equal_rows(&[1.0], &[1.0, f64::NAN], 1e-5));
    }
}

#[cfg(test)]
mod review_negative_controls {
    use super::near_equal_rows as eq;
    const E: f64 = 1e-5;
    #[test]
    fn real_tie_with_shared_placeholders_is_not_saturation() {
        // Same uncomputed NaN slots, one consumed feature differs beyond eps: must stay a model tie.
        let a = [f64::NAN, 0.25, 1.0, f64::NAN];
        let b = [f64::NAN, 0.25, 1.0 + 2e-5, f64::NAN];
        assert!(!eq(&a, &b, E));
    }
    #[test]
    fn placeholder_position_mismatch_is_not_saturation() {
        assert!(!eq(&[f64::NAN, 1.0], &[1.0, f64::NAN], E));
    }
    #[test]
    fn identical_rows_with_placeholders_are_saturation() {
        assert!(eq(
            &[f64::NAN, 0.5, f64::NAN],
            &[f64::NAN, 0.5, f64::NAN],
            E
        ));
    }
    #[test]
    fn neg_infinity_and_mixed_inf_never_equal() {
        assert!(!eq(&[f64::NEG_INFINITY], &[f64::NEG_INFINITY], E));
        assert!(!eq(&[f64::INFINITY], &[1.0], E));
    }
    #[test]
    fn original_comparator_misses_placeholder_saturation() {
        let a = [f64::NAN, 0.5];
        let b = [f64::NAN, 0.5];
        let original = a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= E);
        assert!(!original && eq(&a, &b, E));
    }
}
