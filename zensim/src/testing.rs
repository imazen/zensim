//! Visual regression testing utilities.
//!
//! Compare expected vs actual images against configurable tolerances,
//! producing structured pass/fail reports with human-readable output.
//!
//! # Quick start
//!
//! ```no_run
//! use zensim::{Zensim, ZensimProfile, RgbSlice};
//! use zensim::testing::{RegressionTolerance, RegressionReport};
//! # let (expected_px, actual_px) = (vec![[0u8; 3]; 64], vec![[0u8; 3]; 64]);
//!
//! let z = Zensim::new(ZensimProfile::latest());
//! let expected = RgbSlice::new(&expected_px, 8, 8);
//! let actual = RgbSlice::new(&actual_px, 8, 8);
//!
//! let report = z.check_regression(
//!     &expected, &actual,
//!     &RegressionTolerance::off_by_one(),
//! ).unwrap();
//! assert!(report.passed(), "{report}");
//! ```

use crate::error::ZensimError;
use crate::metric::{ErrorCategory, RoundingBias, Zensim};
use crate::source::ImageSource;

/// Tolerance for regression checking. All constraints must pass.
///
/// Use presets ([`exact()`](Self::exact), [`off_by_one()`](Self::off_by_one))
/// or build custom tolerances with the builder methods.
///
/// # Examples
///
/// ```
/// use zensim::testing::RegressionTolerance;
///
/// // Pixel-identical — no differences allowed
/// let t = RegressionTolerance::exact();
///
/// // Off-by-1, but at most 5% of pixels may differ
/// let t = RegressionTolerance::off_by_one()
///     .max_differing_pixel_fraction(0.05);
///
/// // Allow up to 3/255 delta, score must be >= 90
/// let t = RegressionTolerance::off_by_one()
///     .max_channel_delta(3)
///     .min_score(90.0);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RegressionTolerance {
    max_channel_delta: u8,
    max_differing_pixel_fraction: f64,
    min_identical_channel_fraction: f64,
    min_score: f64,
}

impl RegressionTolerance {
    /// Pixel-identical. No differences allowed.
    pub fn exact() -> Self {
        Self {
            max_channel_delta: 0,
            max_differing_pixel_fraction: 0.0,
            min_identical_channel_fraction: 1.0,
            min_score: 100.0,
        }
    }

    /// Allow off-by-1 rounding. Max delta 1/255, score >= 95.
    /// Any fraction of pixels may be affected.
    pub fn off_by_one() -> Self {
        Self {
            max_channel_delta: 1,
            max_differing_pixel_fraction: 1.0,
            min_identical_channel_fraction: 0.0,
            min_score: 95.0,
        }
    }

    /// Set the maximum per-channel delta (in 1/255 units).
    pub fn max_channel_delta(mut self, n: u8) -> Self {
        self.max_channel_delta = n;
        self
    }

    /// Set the maximum fraction of pixels where any channel differs.
    pub fn max_differing_pixel_fraction(mut self, f: f64) -> Self {
        self.max_differing_pixel_fraction = f;
        self
    }

    /// Set the minimum fraction of (pixel, channel) values that must be byte-identical.
    pub fn min_identical_channel_fraction(mut self, f: f64) -> Self {
        self.min_identical_channel_fraction = f;
        self
    }

    /// Set the minimum acceptable zensim score (0–100).
    pub fn min_score(mut self, s: f64) -> Self {
        self.min_score = s;
        self
    }
}

/// Per-constraint pass/fail detail (drives Display output).
#[derive(Clone)]
struct ConstraintResult {
    name: &'static str,
    passed: bool,
    actual: String,
    limit: String,
}

/// Result of comparing expected vs actual images against a tolerance.
///
/// Use [`Zensim::check_regression()`] to produce this.
///
/// The [`Display`](std::fmt::Display) impl produces human-readable prose
/// suitable for test output. The [`Debug`] impl shows structured key-value
/// data suitable for CI parsing.
#[non_exhaustive]
#[derive(Clone)]
pub struct RegressionReport {
    passed: bool,
    score: f64,
    category: ErrorCategory,
    confidence: f64,
    max_channel_delta: [u8; 3],
    pixel_count: u64,
    pixels_differing: u64,
    pixels_failing: u64,
    identical_channel_fraction: f64,
    rounding_bias: Option<RoundingBias>,
    constraint_results: Vec<ConstraintResult>,
}

impl RegressionReport {
    /// Whether all constraints passed.
    pub fn passed(&self) -> bool {
        self.passed
    }

    /// Zensim similarity score (0–100).
    pub fn score(&self) -> f64 {
        self.score
    }

    /// Detected error category.
    pub fn category(&self) -> ErrorCategory {
        self.category
    }

    /// Confidence in the category (0.0–1.0).
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Max per-channel delta in 1/255 units `[R, G, B]`.
    pub fn max_channel_delta(&self) -> [u8; 3] {
        self.max_channel_delta
    }

    /// Total pixels compared.
    pub fn pixel_count(&self) -> u64 {
        self.pixel_count
    }

    /// Pixels where any channel differs.
    pub fn pixels_differing(&self) -> u64 {
        self.pixels_differing
    }

    /// Pixels exceeding tolerance's max_channel_delta.
    pub fn pixels_failing(&self) -> u64 {
        self.pixels_failing
    }

    /// Fraction of (pixel, channel) values that are byte-identical.
    pub fn identical_channel_fraction(&self) -> f64 {
        self.identical_channel_fraction
    }

    /// Rounding bias (only when category is RoundingError).
    pub fn rounding_bias(&self) -> Option<&RoundingBias> {
        self.rounding_bias.as_ref()
    }
}

impl std::fmt::Debug for RegressionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegressionReport")
            .field("passed", &self.passed)
            .field("score", &self.score)
            .field("category", &self.category)
            .field("confidence", &self.confidence)
            .field("max_channel_delta", &self.max_channel_delta)
            .field("pixel_count", &self.pixel_count)
            .field("pixels_differing", &self.pixels_differing)
            .field("pixels_failing", &self.pixels_failing)
            .field(
                "identical_channel_fraction",
                &self.identical_channel_fraction,
            )
            .field("rounding_bias", &self.rounding_bias)
            .finish()
    }
}

impl std::fmt::Display for RegressionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };

        // Special case: pixel-identical
        if self.pixels_differing == 0 && self.passed {
            return write!(f, "PASS: Images are pixel-identical. Score: 100.0.");
        }

        // Header line
        match self.category {
            ErrorCategory::Identical => {
                writeln!(f, "{status}: Images are pixel-identical. Score: 100.0.")?;
            }
            ErrorCategory::RoundingError => {
                writeln!(
                    f,
                    "{status}: Off-by-{} rounding, {}/{} pixels ({:.1}%).",
                    self.max_channel_delta.iter().max().unwrap_or(&0),
                    self.pixels_differing,
                    self.pixel_count,
                    self.pixels_differing as f64 / self.pixel_count.max(1) as f64 * 100.0,
                )?;
            }
            ErrorCategory::ChannelSwap => {
                writeln!(f, "{status}: Channel swap detected (RGB/BGR).")?;
            }
            ErrorCategory::AlphaCompositing => {
                writeln!(f, "{status}: Alpha compositing error detected.")?;
            }
            ErrorCategory::Unclassified => {
                if self.passed {
                    writeln!(f, "{status}: Images differ within tolerance.")?;
                } else {
                    writeln!(f, "{status}: Large differences detected.")?;
                }
            }
        }

        // Constraint details
        for cr in &self.constraint_results {
            let mark = if cr.passed { " " } else { "x" };
            writeln!(
                f,
                "  {mark} {}: {} (limit: {}).",
                cr.name, cr.actual, cr.limit
            )?;
        }

        // Rounding bias
        if let Some(ref bias) = self.rounding_bias {
            if bias.balanced {
                writeln!(f, "  Direction: balanced rounding.")?;
            } else {
                let all_pos = bias.positive_fraction.iter().all(|&frac| frac > 0.8);
                let all_neg = bias.positive_fraction.iter().all(|&frac| frac < 0.2);
                if all_pos {
                    write!(f, "  Direction: all positive — systematic truncation.")?;
                } else if all_neg {
                    write!(f, "  Direction: all negative — systematic ceiling.")?;
                } else {
                    write!(
                        f,
                        "  Direction: biased (R={:.0}%+ G={:.0}%+ B={:.0}%+).",
                        bias.positive_fraction[0] * 100.0,
                        bias.positive_fraction[1] * 100.0,
                        bias.positive_fraction[2] * 100.0,
                    )?;
                }
            }
        }

        Ok(())
    }
}

// ─── Zensim::check_regression() ──────────────────────────────────────────

impl Zensim {
    /// Compare expected vs actual images against a tolerance.
    ///
    /// Returns a [`RegressionReport`] with pass/fail, score, classification,
    /// and per-constraint details. Use the report's [`Display`](std::fmt::Display)
    /// impl for human-readable output in test assertions.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use zensim::{Zensim, ZensimProfile, RgbSlice};
    /// use zensim::testing::RegressionTolerance;
    /// # let (expected_px, actual_px) = (vec![[0u8; 3]; 64], vec![[0u8; 3]; 64]);
    /// let z = Zensim::new(ZensimProfile::latest());
    /// let expected = RgbSlice::new(&expected_px, 8, 8);
    /// let actual = RgbSlice::new(&actual_px, 8, 8);
    ///
    /// let report = z.check_regression(
    ///     &expected, &actual,
    ///     &RegressionTolerance::off_by_one(),
    /// ).unwrap();
    /// assert!(report.passed(), "{report}");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too small.
    pub fn check_regression(
        &self,
        expected: &impl ImageSource,
        actual: &impl ImageSource,
        tolerance: &RegressionTolerance,
    ) -> Result<RegressionReport, ZensimError> {
        let cr = self.classify(expected, actual)?;
        let ds = &cr.delta_stats;

        // Convert max_abs_delta from f64 (0-1 range) to u8 (0-255 units)
        let max_channel_delta: [u8; 3] =
            std::array::from_fn(|c| (ds.max_abs_delta[c] * 255.0).round().min(255.0) as u8);

        // Compute identical_channel_fraction from signed_small_histogram
        let identical_values: u64 = (0..3).map(|ch| ds.signed_small_histogram[ch][3]).sum();
        let total_values = ds.pixel_count * 3;
        let identical_channel_fraction = if total_values > 0 {
            identical_values as f64 / total_values as f64
        } else {
            1.0
        };

        // Compute pixels_failing based on tolerance threshold
        let pixels_failing = if tolerance.max_channel_delta == 0 {
            ds.pixels_differing
        } else if tolerance.max_channel_delta == 1 {
            ds.pixels_differing_by_more_than_1
        } else {
            // For higher thresholds: conservative upper bound
            let max_u8 = *max_channel_delta.iter().max().unwrap_or(&0);
            if max_u8 <= tolerance.max_channel_delta {
                0
            } else {
                ds.pixels_differing // conservative: all differing pixels
            }
        };

        let differing_fraction = if ds.pixel_count > 0 {
            ds.pixels_differing as f64 / ds.pixel_count as f64
        } else {
            0.0
        };

        // Evaluate constraints
        let mut constraint_results = Vec::new();

        // 1. Max channel delta
        let max_delta_actual = *max_channel_delta.iter().max().unwrap_or(&0);
        let delta_pass = max_delta_actual <= tolerance.max_channel_delta;
        constraint_results.push(ConstraintResult {
            name: "Max delta",
            passed: delta_pass,
            actual: format!(
                "R={} G={} B={}",
                max_channel_delta[0], max_channel_delta[1], max_channel_delta[2],
            ),
            limit: format!("{}/255", tolerance.max_channel_delta),
        });

        // 2. Score
        let score_pass = cr.result.score >= tolerance.min_score;
        constraint_results.push(ConstraintResult {
            name: "Score",
            passed: score_pass,
            actual: format!("{:.1}", cr.result.score),
            limit: format!(">={:.1}", tolerance.min_score),
        });

        // 3. Differing pixel fraction
        let diff_pass = differing_fraction <= tolerance.max_differing_pixel_fraction;
        constraint_results.push(ConstraintResult {
            name: "Pixels differing",
            passed: diff_pass,
            actual: format!("{:.1}%", differing_fraction * 100.0),
            limit: format!("<={:.1}%", tolerance.max_differing_pixel_fraction * 100.0),
        });

        // 4. Identical channel fraction
        let ident_pass = identical_channel_fraction >= tolerance.min_identical_channel_fraction;
        constraint_results.push(ConstraintResult {
            name: "Identical channels",
            passed: ident_pass,
            actual: format!("{:.1}%", identical_channel_fraction * 100.0),
            limit: format!(">={:.1}%", tolerance.min_identical_channel_fraction * 100.0),
        });

        let passed = delta_pass && score_pass && diff_pass && ident_pass;

        Ok(RegressionReport {
            passed,
            score: cr.result.score,
            category: cr.classification.dominant,
            confidence: cr.classification.confidence,
            max_channel_delta,
            pixel_count: ds.pixel_count,
            pixels_differing: ds.pixels_differing,
            pixels_failing,
            identical_channel_fraction,
            rounding_bias: cr.classification.rounding_bias,
            constraint_results,
        })
    }
}
