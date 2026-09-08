//! Train-only median/envelope seed fitting, shared by native codec harnesses.
use crate::SeedEstimate;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

/// Frozen native-knob versus score curve. Fitting expects aligned training
/// ladders; callers must validate source-family and model provenance separately.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeedCurve {
    points: Vec<(f32, f32)>,
    adjusted_points: usize,
}

impl SeedCurve {
    /// Median training score at each knob, followed by the existing monotone
    /// running envelope. `inverted` means smaller knobs yield higher scores.
    pub fn fit(rows: &[Vec<(f32, f32)>], inverted: bool) -> Result<Self> {
        ensure!(
            !rows.is_empty() && rows[0].len() >= 2,
            "at least one two-point training ladder required"
        );
        let grid = &rows[0];
        for row in rows {
            ensure!(
                row.len() == grid.len(),
                "training ladders have different lengths"
            );
            ensure!(
                row.iter()
                    .zip(grid)
                    .all(|(&(q, s), &(x, _))| q.is_finite() && s.is_finite() && q == x),
                "nonfinite or misaligned training ladder"
            );
            ensure!(
                row.windows(2).all(|p| p[0].0 < p[1].0),
                "training knobs must strictly increase"
            );
        }
        let mut points: Vec<(f32, f32)> = (0..grid.len())
            .map(|i| {
                let mut values: Vec<_> = rows.iter().map(|r| r[i].1).collect();
                values.sort_by(f32::total_cmp);
                let n = values.len();
                // Preserve the original fitter's f32 arithmetic.
                (grid[i].0, (values[(n - 1) / 2] + values[n / 2]) * 0.5)
            })
            .collect();
        let mut adjusted_points = 0;
        for i in 1..points.len() {
            let old = points[i].1;
            points[i].1 = if inverted {
                old.min(points[i - 1].1)
            } else {
                old.max(points[i - 1].1)
            };
            adjusted_points += usize::from(points[i].1 != old);
        }
        let curve = Self {
            points,
            adjusted_points,
        };
        curve.validate()?;
        Ok(curve)
    }

    /// Invert the closest non-flat segment and clamp to the native range.
    /// Validation also covers curves loaded from a serialized artifact.
    pub fn estimate(&self, target: f32) -> Result<SeedEstimate> {
        self.validate()?;
        ensure!(target.is_finite(), "seed target must be finite");
        let pair = self
            .points
            .windows(2)
            .filter(|p| (p[1].1 - p[0].1).abs() > 1e-5)
            .min_by(|a, b| {
                let distance = |p: &[(f32, f32)]| {
                    let lo = p[0].1.min(p[1].1);
                    let hi = p[0].1.max(p[1].1);
                    (target - target.clamp(lo, hi)).abs()
                };
                distance(a).total_cmp(&distance(b))
            })
            .context("training curve is entirely flat; no usable seed slope")?;
        let slope = (pair[1].1 - pair[0].1) / (pair[1].0 - pair[0].0);
        let knob = (pair[0].0 + (target - pair[0].1) / slope)
            .clamp(self.points[0].0, self.points.last().unwrap().0);
        ensure!(
            slope.is_finite() && slope.abs() >= 1e-6 && knob.is_finite(),
            "unusable seed slope/proposal"
        );
        Ok(SeedEstimate {
            knob,
            score_per_knob: slope,
        })
    }

    /// Ordered (native knob, median/enveloped score) pairs.
    pub fn points(&self) -> &[(f32, f32)] {
        &self.points
    }

    /// Number of median points changed by the monotone envelope.
    pub fn adjusted_points(&self) -> usize {
        self.adjusted_points
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.points.len() >= 2
                && self
                    .points
                    .iter()
                    .all(|(q, s)| q.is_finite() && s.is_finite()),
            "finite two-point seed curve required"
        );
        ensure!(
            self.points.windows(2).all(|p| p[0].0 < p[1].0),
            "seed knobs must strictly increase"
        );
        let up = self.points.windows(2).all(|p| p[0].1 <= p[1].1);
        let down = self.points.windows(2).all(|p| p[0].1 >= p[1].1);
        ensure!(up || down, "seed scores must be monotone");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fit_uses_training_medians_and_envelope_with_negative_targets() {
        let curve = SeedCurve::fit(
            &[
                vec![(0., 90.), (1., 60.), (2., 70.), (10., -20.)],
                vec![(0., 100.), (1., 80.), (2., 90.), (10., -10.)],
                vec![(0., 100.), (1., 80.), (2., 80.), (10., -20.)],
            ],
            true,
        )
        .unwrap();
        assert_eq!(
            curve.points(),
            [(0., 100.), (1., 80.), (2., 80.), (10., -20.)]
        );
        let seed = curve.estimate(-10.).unwrap();
        assert!((seed.knob - 9.2).abs() < 1e-5);
        assert_eq!(seed.score_per_knob, -12.5);
        let envelope =
            SeedCurve::fit(&[vec![(0., 100.), (1., 60.), (2., 70.), (10., -20.)]], true).unwrap();
        assert_eq!(envelope.adjusted_points(), 1);
        assert_eq!(envelope.points()[2].1, 60.);
    }
    #[test]
    fn invalid_training_or_serialized_curves_are_refused() {
        for rows in [
            vec![],
            vec![vec![]],
            vec![vec![(0., 0.)]],
            vec![vec![(0., 0.), (0., 1.)]],
            vec![vec![(0., 0.), (1., f32::NAN)]],
            vec![vec![(0., 0.), (1., 1.)], vec![(0., 0.), (2., 1.)]],
        ] {
            assert!(SeedCurve::fit(&rows, false).is_err());
        }
        let bad: SeedCurve =
            serde_json::from_str(r#"{"points":[[0,0],[1,1],[2,0]],"adjusted_points":0}"#).unwrap();
        assert!(bad.estimate(0.5).is_err());
        let flat = SeedCurve::fit(&[vec![(0., 1.), (1., 1.)]], false).unwrap();
        assert!(flat.estimate(1.).is_err());
    }
}
