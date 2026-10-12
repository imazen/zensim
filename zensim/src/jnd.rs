//! Declared JND side output for a profile's served score.
//!
//! Phase A of `docs/JND_DIAL_PLAN_2026-10-11.md`: the 0–100 dial is unchanged,
//! and a profile may carry a declared map from its served score `s` to an
//! estimate in JND (just-noticeable differences). 0 JND is the source image;
//! 1 JND is a distortion an observer picks out in 75 % of forced-choice
//! comparisons. Above a few JND the unit stops being measurable, so the map
//! answers only inside the range its calibration data supports and says
//! "beyond JND range" everywhere else, never a number.
//!
//! The map family, its calibration data and its range rule were registered
//! before fitting (`benchmarks/jndcal_protocol_2026-10-11.md` §§3 and 7):
//! `J(s) = a · max(0, 100 − s)^c`, with the knee fixed at 100 because a score
//! of 100 is pixel identity. The coefficients and their out-of-fold error are
//! recorded in `benchmarks/jnd_phase_a_2026-10-11.md`.
//!
//! The maps are calibrated on SDR images viewed at native pixels, with no
//! display model. A score computed from absolute-luminance (HDR) input is
//! outside that calibration.

use core::fmt;

/// A declared map from one profile's served score to JND.
///
/// Get one from [`ZensimProfile::jnd_map`](crate::ZensimProfile::jnd_map). It
/// is only meaningful for scores served by that profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JndMap {
    /// Scale `a ≥ 0` of `J(s) = a · max(0, 100 − s)^c`.
    a: f64,
    /// Exponent `c > 0`.
    c: f64,
    /// The supported range, or `None` when the calibration supports no range
    /// above identity (the registered join-point rule found no supported bin).
    support: Option<Support>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Support {
    /// The largest JND the calibration supports (J*, protocol §7).
    max_jnd: f64,
    /// Out-of-fold RMSE in JND, pooled over the calibration stimuli whose
    /// out-of-fold prediction is at most `max_jnd`.
    rmse: f64,
}

/// One JND estimate from a [`JndMap`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum JndEstimate {
    /// The score maps inside the supported range.
    InRange {
        /// The JND estimate, `J(s)`.
        jnd: f64,
        /// One standard prediction error in JND: the map's out-of-fold RMSE
        /// over its supported range. 0 at a score of 100 or more, which the
        /// fixed knee maps to 0 JND by definition.
        uncertainty: f64,
    },
    /// The score maps above the supported maximum (or is not finite). The map
    /// reports "beyond JND range" here, never a number.
    BeyondRange,
}

impl JndMap {
    /// The JND estimate for a served score.
    ///
    /// A score of 100 or more maps to 0 JND (pixel identity scores exactly
    /// 100). Any other score is [`JndEstimate::BeyondRange`] when `J(s)` is
    /// above [`max_jnd`](Self::max_jnd), when the map has no supported range,
    /// or when the score is not finite.
    pub fn estimate(&self, score: f64) -> JndEstimate {
        if !score.is_finite() {
            return JndEstimate::BeyondRange;
        }
        let below_identity = 100.0 - score;
        if below_identity <= 0.0 {
            return JndEstimate::InRange {
                jnd: 0.0,
                uncertainty: 0.0,
            };
        }
        match self.support {
            Some(Support { max_jnd, rmse }) => {
                let jnd = self.a * below_identity.powf(self.c);
                if jnd <= max_jnd {
                    JndEstimate::InRange {
                        jnd,
                        uncertainty: rmse,
                    }
                } else {
                    JndEstimate::BeyondRange
                }
            }
            None => JndEstimate::BeyondRange,
        }
    }

    /// The largest JND the calibration supports, or `None` when it supports no
    /// range above identity.
    pub fn max_jnd(&self) -> Option<f64> {
        self.support.map(|s| s.max_jnd)
    }
}

impl JndEstimate {
    /// `true` inside the supported range.
    pub fn in_range(&self) -> bool {
        matches!(self, Self::InRange { .. })
    }

    /// The JND estimate, or `None` beyond the supported range.
    pub fn jnd(&self) -> Option<f64> {
        match *self {
            Self::InRange { jnd, .. } => Some(jnd),
            Self::BeyondRange => None,
        }
    }

    /// The estimate's uncertainty in JND, or `None` beyond the supported range.
    pub fn uncertainty(&self) -> Option<f64> {
        match *self {
            Self::InRange { uncertainty, .. } => Some(uncertainty),
            Self::BeyondRange => None,
        }
    }
}

impl fmt::Display for JndEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InRange { jnd, uncertainty } => write!(f, "{jnd:.2} ± {uncertainty:.2} JND"),
            Self::BeyondRange => f.write_str("beyond JND range"),
        }
    }
}

/// `ZensimProfile::B`'s map: fitted on B's served score over the KonFiG-IQA
/// F-scale TRAIN sources (SRC06, SRC28, SRC50; 252 stimuli), knee fixed at 100.
/// Out-of-fold RMSE 1.427 JND over the whole range. The registered join-point
/// rule found no supported bin, so the map supports no range above identity.
/// Provenance: `benchmarks/jnd_phase_a_2026-10-11.md`.
pub(crate) const PROFILE_B_JND: JndMap = JndMap {
    a: 0.012_653_289_786_345_816,
    c: 1.317_411_381_194_271_6,
    support: None,
};

#[cfg(test)]
mod tests {
    use super::*;

    const SUPPORTED: JndMap = JndMap {
        a: 0.01,
        c: 1.5,
        support: Some(Support {
            max_jnd: 1.5,
            rmse: 0.4,
        }),
    };

    #[test]
    fn identity_and_above_map_to_zero_with_no_error() {
        for map in [SUPPORTED, PROFILE_B_JND] {
            for s in [100.0, 100.5, 1e9] {
                assert_eq!(
                    map.estimate(s),
                    JndEstimate::InRange {
                        jnd: 0.0,
                        uncertainty: 0.0
                    }
                );
            }
        }
    }

    #[test]
    fn in_range_follows_the_fixed_knee_power() {
        // 0.01 · 10^1.5 = 0.316…, inside max 1.5
        let e = SUPPORTED.estimate(90.0);
        let want = 0.01 * 10f64.powf(1.5);
        assert!(e.in_range());
        assert!((e.jnd().unwrap() - want).abs() < 1e-15);
        assert_eq!(e.uncertainty(), Some(0.4));
        // exactly at the maximum is still in range: 0.01 · d^1.5 = 1.5
        let d_max = (1.5f64 / 0.01).powf(1.0 / 1.5);
        assert!(SUPPORTED.estimate(100.0 - d_max * (1.0 - 1e-12)).in_range());
    }

    #[test]
    fn above_the_maximum_reports_no_number() {
        // 0.01 · 40^1.5 = 2.53 > 1.5
        let e = SUPPORTED.estimate(60.0);
        assert_eq!(e, JndEstimate::BeyondRange);
        assert!(!e.in_range());
        assert_eq!(e.jnd(), None);
        assert_eq!(e.uncertainty(), None);
        assert_eq!(e.to_string(), "beyond JND range");
        // negative served scores are valid dial values and stay beyond range
        assert_eq!(SUPPORTED.estimate(-20.0), JndEstimate::BeyondRange);
    }

    #[test]
    fn non_finite_scores_report_no_number() {
        for s in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(SUPPORTED.estimate(s), JndEstimate::BeyondRange);
        }
    }

    #[test]
    fn a_map_without_support_is_beyond_range_short_of_identity() {
        assert_eq!(PROFILE_B_JND.max_jnd(), None);
        for s in [99.999, 95.0, 90.0, 80.0, 70.0, 0.0, -50.0] {
            assert_eq!(PROFILE_B_JND.estimate(s), JndEstimate::BeyondRange);
        }
    }

    #[test]
    fn display_carries_estimate_and_uncertainty() {
        let e = JndEstimate::InRange {
            jnd: 0.8412,
            uncertainty: 0.5,
        };
        assert_eq!(e.to_string(), "0.84 ± 0.50 JND");
    }

    #[test]
    fn profile_b_coefficients_match_the_recorded_fit() {
        // benchmarks/jnd_phase_a_2026-10-11.md: jnd_calibrate full_fit_params
        // [0.012653289786345816, 100.0, 1.3174113811942716]
        assert_eq!(PROFILE_B_JND.a.to_bits(), 0.012653289786345816f64.to_bits());
        assert_eq!(PROFILE_B_JND.c.to_bits(), 1.3174113811942716f64.to_bits());
    }
}
