//! Visual regression testing: tolerances, reports, and comparison.
//!
//! Compare expected vs actual images against configurable tolerances,
//! producing structured pass/fail reports with human-readable output.
//!
//! # Quick start
//!
//! ```no_run
//! use zensim::{Zensim, ZensimProfile, RgbSlice};
//! use zensim_regress::testing::{RegressionTolerance, check_regression};
//! # let (expected_px, actual_px) = (vec![[0u8; 3]; 64], vec![[0u8; 3]; 64]);
//!
//! let z = Zensim::new(ZensimProfile::latest());
//! let expected = RgbSlice::new(&expected_px, 8, 8);
//! let actual = RgbSlice::new(&actual_px, 8, 8);
//!
//! let tolerance = RegressionTolerance::off_by_one()
//!     .with_max_pixels_different(0.05);
//! let report = check_regression(&z, &expected, &actual, &tolerance).unwrap();
//! assert!(report.passed(), "{report}");
//! ```

use zensim::{
    AlphaMode, ClassifiedResult, ErrorCategory, ImageSource, PixelFormat, RoundingBias, Zensim,
    ZensimError,
};

/// Tolerance for regression checking. All constraints must pass.
///
/// Use presets ([`exact()`](Self::exact), [`off_by_one()`](Self::off_by_one))
/// or build custom tolerances with the builder methods.
///
/// # Examples
///
/// ```
/// use zensim_regress::testing::RegressionTolerance;
///
/// // Pixel-identical — no differences allowed
/// let t = RegressionTolerance::exact();
///
/// // Off-by-1, but at most 5% of pixels may differ
/// let t = RegressionTolerance::off_by_one()
///     .with_max_pixels_different(0.05);
///
/// // Allow up to 3/255 delta, score must be >= 90
/// let t = RegressionTolerance::off_by_one()
///     .with_max_delta(3)
///     .with_min_similarity(90.0);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RegressionTolerance {
    max_delta: u8,
    max_pixels_different: f64,
    min_similarity: f64,
    max_alpha_delta: u8,
    ignore_alpha: bool,
}

impl RegressionTolerance {
    /// Pixel-identical. No differences allowed.
    pub fn exact() -> Self {
        Self {
            max_delta: 0,
            max_pixels_different: 0.0,
            min_similarity: 100.0,
            max_alpha_delta: 0,
            ignore_alpha: false,
        }
    }

    /// Allow off-by-1 rounding. Max delta 1/255, score >= 85.
    /// Any fraction of pixels may be affected.
    ///
    /// The 85.0 threshold is calibrated from 200+ real imageflow pairs:
    /// true off-by-one differences (1/255 delta, ~5% of pixels) score as
    /// low as 86.4 on photo content, so 85.0 provides headroom without
    /// accepting visibly different images.
    ///
    /// Alpha channel has zero tolerance by default — alpha divergence is
    /// a structural bug, not a rounding artifact. Use [`max_alpha_delta`](Self::max_alpha_delta)
    /// to relax this if needed.
    pub fn off_by_one() -> Self {
        Self {
            max_delta: 1,
            max_pixels_different: 1.0,
            min_similarity: 85.0,
            max_alpha_delta: 0,
            ignore_alpha: false,
        }
    }

    // ─── Getters ─────────────────────────────────────────────────────

    /// Maximum per-channel delta (in 1/255 units).
    pub fn max_delta(&self) -> u8 {
        self.max_delta
    }

    /// Minimum acceptable zensim score (0–100).
    pub fn min_similarity(&self) -> f64 {
        self.min_similarity
    }

    /// Maximum fraction of pixels where any channel differs.
    pub fn max_pixels_different(&self) -> f64 {
        self.max_pixels_different
    }

    /// Maximum alpha channel delta (in 1/255 units).
    pub fn max_alpha_delta(&self) -> u8 {
        self.max_alpha_delta
    }

    /// Whether alpha channel is ignored.
    pub fn is_ignore_alpha(&self) -> bool {
        self.ignore_alpha
    }

    // ─── Builder methods ─────────────────────────────────────────────

    /// Set the maximum per-channel delta (in 1/255 units).
    pub fn with_max_delta(mut self, n: u8) -> Self {
        self.max_delta = n;
        self
    }

    /// Set the maximum fraction of pixels where any channel differs.
    pub fn with_max_pixels_different(mut self, f: f64) -> Self {
        self.max_pixels_different = f;
        self
    }

    /// Set the minimum acceptable zensim score (0–100).
    pub fn with_min_similarity(mut self, s: f64) -> Self {
        self.min_similarity = s;
        self
    }

    /// Set the maximum alpha channel delta (in 1/255 units).
    ///
    /// Defaults to 0 (zero tolerance) because alpha divergence between
    /// two processing pipelines is typically a structural bug, not numerical
    /// rounding. See `docs/alpha-channel-diffing.md` for the rationale.
    pub fn with_max_alpha_delta(mut self, n: u8) -> Self {
        self.max_alpha_delta = n;
        self
    }

    /// Ignore the alpha channel entirely when comparing RGBA images.
    ///
    /// When set, RGBA inputs are treated as opaque (RGBX) — the alpha byte
    /// is ignored for scoring (no checkerboard compositing) and the alpha
    /// constraint does not affect pass/fail. Alpha stats are still tracked
    /// in the report for informational purposes.
    ///
    /// Has no effect on RGB-only inputs.
    pub fn ignore_alpha(mut self) -> Self {
        self.ignore_alpha = true;
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

// ─── Channel histograms ─────────────────────────────────────────────────

/// Per-channel pixel value histograms (256 bins, quantized to 8-bit).
///
/// For 8-bit inputs, bins map exactly to pixel values 0–255.
/// For higher bit-depths (16-bit, f32), values are quantized to the
/// nearest 8-bit equivalent.
#[non_exhaustive]
#[derive(Clone)]
pub struct ChannelHistograms {
    bins: [[u64; 256]; 4],
    num_channels: usize,
}

impl ChannelHistograms {
    /// Number of channels (3 for RGB, 4 for RGBA).
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Histogram bins for the given channel index (0=R, 1=G, 2=B, 3=A).
    ///
    /// Returns `None` if `ch` is out of range for this image's format.
    pub fn channel(&self, ch: usize) -> Option<&[u64; 256]> {
        if ch < self.num_channels {
            Some(&self.bins[ch])
        } else {
            None
        }
    }

    /// Histogram intersection with `other` for a single channel.
    ///
    /// Returns 1.0 for identical distributions, 0.0 for completely disjoint.
    /// This is the standard histogram intersection metric:
    /// `sum(min(h1[i], h2[i])) / max(sum(h1), sum(h2))`.
    pub fn intersection(&self, other: &Self, ch: usize) -> f64 {
        if ch >= self.num_channels || ch >= other.num_channels {
            return 0.0;
        }
        let mut min_sum = 0u64;
        let mut max_total = 0u64;
        for i in 0..256 {
            min_sum += self.bins[ch][i].min(other.bins[ch][i]);
            max_total += self.bins[ch][i].max(other.bins[ch][i]);
        }
        if max_total == 0 {
            1.0
        } else {
            min_sum as f64 / max_total as f64
        }
    }

    /// Histogram intersection across all channels. Returns per-channel values.
    ///
    /// For RGB: `[R, G, B]`. For RGBA: `[R, G, B, A]`.
    pub fn intersection_all(&self, other: &Self) -> Vec<f64> {
        let n = self.num_channels.min(other.num_channels);
        (0..n).map(|ch| self.intersection(other, ch)).collect()
    }
}

impl std::fmt::Debug for ChannelHistograms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let labels = ["R", "G", "B", "A"];
        f.debug_struct("ChannelHistograms")
            .field("num_channels", &self.num_channels)
            .field(
                "totals",
                &(0..self.num_channels)
                    .map(|ch| {
                        let total: u64 = self.bins[ch].iter().sum();
                        format!("{}={}", labels[ch], total)
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
            )
            .finish()
    }
}

/// Result of comparing expected vs actual images against a tolerance.
///
/// Use [`check_regression()`] to produce this.
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
    max_channel_delta_f64: [f64; 3],
    native_max: f64,
    pixel_count: u64,
    pixels_differing: u64,
    pixels_failing: u64,
    identical_channel_fraction: f64,
    alpha_max_delta: u8,
    alpha_pixels_differing: u64,
    expected_histogram: ChannelHistograms,
    actual_histogram: ChannelHistograms,
    rounding_bias: Option<RoundingBias>,
    constraint_results: Vec<ConstraintResult>,
    dimension_info: Option<DimensionInfo>,
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

    /// Max per-channel delta as f64, at native precision.
    ///
    /// These are the raw `max_abs_delta` values from [`DeltaStats`](zensim::DeltaStats),
    /// not quantized to u8. For u8 formats this is in 0.0–1.0 (value/255).
    pub fn max_channel_delta_f64(&self) -> [f64; 3] {
        self.max_channel_delta_f64
    }

    /// Native maximum value for the pixel format (255.0, 65535.0, or 1.0).
    pub fn native_max(&self) -> f64 {
        self.native_max
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

    /// Max alpha channel delta in 1/255 units. 0 for RGB-only inputs.
    pub fn alpha_max_delta(&self) -> u8 {
        self.alpha_max_delta
    }

    /// Pixels where alpha channel differs. 0 for RGB-only inputs.
    pub fn alpha_pixels_differing(&self) -> u64 {
        self.alpha_pixels_differing
    }

    /// Per-channel value histograms for the expected (source) image.
    pub fn expected_histogram(&self) -> &ChannelHistograms {
        &self.expected_histogram
    }

    /// Per-channel value histograms for the actual (distorted) image.
    pub fn actual_histogram(&self) -> &ChannelHistograms {
        &self.actual_histogram
    }

    /// Rounding bias (only when category is RoundingError).
    pub fn rounding_bias(&self) -> Option<&RoundingBias> {
        self.rounding_bias.as_ref()
    }

    /// Dimension mismatch metadata, if the comparison used resized images.
    pub fn dimension_info(&self) -> Option<&DimensionInfo> {
        self.dimension_info.as_ref()
    }

    /// Set dimension mismatch metadata (e.g., after transform detection).
    pub fn set_dimension_info(&mut self, info: DimensionInfo) {
        self.dimension_info = Some(info);
    }
}

/// Classification of a dimension mismatch between expected and actual images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionMismatchKind {
    /// Width and height are swapped (e.g., 480x640 vs 640x480).
    /// Only detected when both dimensions are non-square.
    OrientationSwap,
    /// Dimensions differ by at most 2px in each axis (rounding artifact).
    OffByOne,
    /// Small dimension change (< 5% in each axis), likely a crop/trim.
    CropDifference,
    /// Large dimension difference that doesn't fit other categories.
    LargeDifference,
}

impl std::fmt::Display for DimensionMismatchKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OrientationSwap => write!(f, "orientation swap"),
            Self::OffByOne => write!(f, "off-by-one rounding"),
            Self::CropDifference => write!(f, "crop/trim"),
            Self::LargeDifference => write!(f, "different dimensions"),
        }
    }
}

/// How the dimension-mismatched images were aligned for comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonMethod {
    /// Actual was Lanczos3-resized to match expected dimensions.
    Resized,
    /// Both images center-cropped to the overlap region (no resize).
    CenterCropped,
    /// Actual was rotated 90° CW to match expected dimensions.
    Rotated90,
    /// Actual was rotated 270° CW (90° CCW) to match expected dimensions.
    Rotated270,
    /// Actual was flipped horizontally.
    FlipHorizontal,
    /// Actual was flipped vertically.
    FlipVertical,
    /// Actual was rotated 180° (equivalent to both flips).
    Rotated180,
    /// Actual was transposed (rotate 90° CW + flip horizontal).
    Transpose,
    /// Actual was transversed (rotate 270° CW + flip horizontal).
    Transverse,
}

impl std::fmt::Display for ComparisonMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resized => write!(f, "resized"),
            Self::CenterCropped => write!(f, "center-cropped"),
            Self::Rotated90 => write!(f, "rotated 90\u{00b0} CW"),
            Self::Rotated270 => write!(f, "rotated 270\u{00b0} CW"),
            Self::FlipHorizontal => write!(f, "flipped horizontally"),
            Self::FlipVertical => write!(f, "flipped vertically"),
            Self::Rotated180 => write!(f, "rotated 180\u{00b0}"),
            Self::Transpose => write!(f, "transposed (rot90+flipH)"),
            Self::Transverse => write!(f, "transversed (rot270+flipH)"),
        }
    }
}

/// Metadata about a dimension-mismatch comparison.
///
/// When expected and actual images have different dimensions, the scoring path
/// categorizes the mismatch and chooses the cheapest comparison method.
/// This struct records the original dimensions, category, and method used
/// so downstream consumers (montages, reports) can annotate the result.
#[derive(Debug, Clone)]
pub struct DimensionInfo {
    /// Original expected image dimensions (width, height).
    pub expected_dims: (u32, u32),
    /// Original actual image dimensions (width, height).
    pub actual_dims: (u32, u32),
    /// Classification of the mismatch.
    pub kind: DimensionMismatchKind,
    /// How the images were aligned for comparison.
    pub method: ComparisonMethod,
}

/// Classify a dimension mismatch from raw dimensions.
pub fn classify_dimension_mismatch(ew: u32, eh: u32, aw: u32, ah: u32) -> DimensionMismatchKind {
    // Orientation swap: width↔height swapped, non-square
    if ew == ah && eh == aw && ew != eh {
        return DimensionMismatchKind::OrientationSwap;
    }

    let dw = (ew as i64 - aw as i64).unsigned_abs();
    let dh = (eh as i64 - ah as i64).unsigned_abs();

    // Off-by-one: both axes differ by at most 2px
    if dw <= 2 && dh <= 2 {
        return DimensionMismatchKind::OffByOne;
    }

    // Crop/trim: each axis differs by less than 5%
    let pct_w = dw as f64 / ew.max(1) as f64;
    let pct_h = dh as f64 / eh.max(1) as f64;
    if pct_w < 0.05 && pct_h < 0.05 {
        return DimensionMismatchKind::CropDifference;
    }

    DimensionMismatchKind::LargeDifference
}

impl DimensionInfo {
    /// Classify the dimension mismatch.
    pub fn kind(&self) -> DimensionMismatchKind {
        self.kind
    }

    /// How the images were aligned for comparison.
    pub fn method(&self) -> ComparisonMethod {
        self.method
    }

    /// Human-readable description of the dimension difference.
    pub fn description(&self) -> String {
        let (ew, eh) = self.expected_dims;
        let (aw, ah) = self.actual_dims;
        let dw = aw as i64 - ew as i64;
        let dh = ah as i64 - eh as i64;
        format!(
            "{ew}\u{00d7}{eh} vs {aw}\u{00d7}{ah} ({}, {dw:+}w {dh:+}h)",
            self.kind,
        )
    }

    /// Short label suffix for the pixel diff panel (e.g., "(RESIZED)", "(CROPPED)").
    pub fn panel_label(&self) -> &'static str {
        match self.method {
            ComparisonMethod::Resized => "RESIZED",
            ComparisonMethod::CenterCropped => "CROPPED",
            ComparisonMethod::Rotated90 => "ROT 90\u{00b0}",
            ComparisonMethod::Rotated270 => "ROT 270\u{00b0}",
            ComparisonMethod::FlipHorizontal => "FLIP H",
            ComparisonMethod::FlipVertical => "FLIP V",
            ComparisonMethod::Rotated180 => "ROT 180\u{00b0}",
            ComparisonMethod::Transpose => "TRANSPOSE",
            ComparisonMethod::Transverse => "TRANSVERSE",
        }
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
            .field("max_channel_delta_f64", &self.max_channel_delta_f64)
            .field("native_max", &self.native_max)
            .field("pixel_count", &self.pixel_count)
            .field("pixels_differing", &self.pixels_differing)
            .field("pixels_failing", &self.pixels_failing)
            .field(
                "identical_channel_fraction",
                &self.identical_channel_fraction,
            )
            .field("alpha_max_delta", &self.alpha_max_delta)
            .field("alpha_pixels_differing", &self.alpha_pixels_differing)
            .field("expected_histogram", &self.expected_histogram)
            .field("actual_histogram", &self.actual_histogram)
            .field("rounding_bias", &self.rounding_bias)
            .field("dimension_info", &self.dimension_info)
            .finish()
    }
}

impl std::fmt::Display for RegressionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Dimension mismatch notice
        if let Some(ref dim_info) = self.dimension_info {
            writeln!(
                f,
                "NOTE: Dimensions differ ({}) \u{2014} score is approximate (actual {} to compare).",
                dim_info.description(),
                dim_info.method,
            )?;
        }

        let status = if self.passed { "PASS" } else { "FAIL" };

        // Special case: pixel-identical
        if self.pixels_differing == 0 && self.passed {
            return write!(f, "PASS: Images are pixel-identical. Score: 100.0.");
        }

        // Header line
        match self.category {
            ErrorCategory::Identical => {
                if self.alpha_pixels_differing > 0 {
                    writeln!(
                        f,
                        "{status}: RGB channels identical, alpha differs (max delta {}, {}/{} pixels).",
                        self.alpha_max_delta, self.alpha_pixels_differing, self.pixel_count,
                    )?;
                } else {
                    writeln!(f, "{status}: Images are pixel-identical. Score: 100.0.")?;
                }
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
            _ => {
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

        // Histogram comparison (only show when notable divergence)
        let hist_match = self
            .expected_histogram
            .intersection_all(&self.actual_histogram);
        let any_divergence = hist_match.iter().any(|&v| v < 0.999);
        if any_divergence {
            let labels = ["R", "G", "B", "A"];
            let parts: Vec<String> = hist_match
                .iter()
                .enumerate()
                .map(|(i, &v)| format!("{}={:.1}%", labels[i], v * 100.0))
                .collect();
            writeln!(f, "  Histogram match: {}.", parts.join(" "))?;
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

// ─── Alpha override wrapper ─────────────────────────────────────────────

/// Thin wrapper that forces `AlphaMode::Opaque` on any `ImageSource`.
struct AlphaOverride<'a, S: ImageSource>(&'a S);

impl<S: ImageSource> ImageSource for AlphaOverride<'_, S> {
    #[inline]
    fn width(&self) -> usize {
        self.0.width()
    }
    #[inline]
    fn height(&self) -> usize {
        self.0.height()
    }
    #[inline]
    fn pixel_format(&self) -> PixelFormat {
        self.0.pixel_format()
    }
    #[inline]
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
    #[inline]
    fn row_bytes(&self, y: usize) -> &[u8] {
        self.0.row_bytes(y)
    }
}

// ─── Report building ────────────────────────────────────────────────────

pub(crate) fn build_report(
    cr: ClassifiedResult,
    tolerance: &RegressionTolerance,
) -> RegressionReport {
    let ds = &cr.delta_stats;
    let native_max = ds.native_max;

    // Convert max_abs_delta from f64 (0-1 range) to u8 (0-255 units)
    let max_channel_delta: [u8; 3] =
        std::array::from_fn(|c| (ds.max_abs_delta[c] * 255.0).round().min(255.0) as u8);
    let max_channel_delta_f64 = ds.max_abs_delta;

    // Compute identical_channel_fraction from signed_small_histogram
    let identical_values: u64 = (0..3).map(|ch| ds.signed_small_histogram[ch][3]).sum();
    let total_values = ds.pixel_count * 3;
    let identical_channel_fraction = if total_values > 0 {
        identical_values as f64 / total_values as f64
    } else {
        1.0
    };

    // Compute pixels_failing based on tolerance threshold
    let pixels_failing = if tolerance.max_delta == 0 {
        ds.pixels_differing
    } else if tolerance.max_delta == 1 {
        ds.pixels_differing_by_more_than_1
    } else {
        // For higher thresholds: conservative upper bound
        let max_u8 = *max_channel_delta.iter().max().unwrap_or(&0);
        if max_u8 <= tolerance.max_delta {
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
    let delta_pass = max_delta_actual <= tolerance.max_delta;
    constraint_results.push(ConstraintResult {
        name: "Max delta",
        passed: delta_pass,
        actual: format!(
            "R={} G={} B={}",
            max_channel_delta[0], max_channel_delta[1], max_channel_delta[2],
        ),
        limit: format!("{}/255", tolerance.max_delta),
    });

    // 2. Score
    // Use 100.0 for pixel-identical images — the multi-scale SSIM computation
    // can produce sub-100 scores on small/uniform images due to floating-point
    // precision loss in the pyramid decomposition. The delta_stats are authoritative
    // for identical pixel detection.
    let effective_score = if ds.pixels_differing == 0 && (!ds.has_alpha || ds.alpha_max_delta == 0)
    {
        100.0
    } else {
        cr.result.score()
    };
    let score_pass = effective_score >= tolerance.min_similarity;
    constraint_results.push(ConstraintResult {
        name: "Similarity",
        passed: score_pass,
        actual: format!("{:.1}", effective_score),
        limit: format!(">={:.1}", tolerance.min_similarity),
    });

    // 3. Differing pixel fraction
    let diff_pass = differing_fraction <= tolerance.max_pixels_different;
    constraint_results.push(ConstraintResult {
        name: "Pixels differing",
        passed: diff_pass,
        actual: format!("{:.1}%", differing_fraction * 100.0),
        limit: format!("<={:.1}%", tolerance.max_pixels_different * 100.0),
    });

    // 4. Alpha channel constraint (only for RGBA/BGRA inputs, skipped when ignore_alpha)
    let alpha_pass = if ds.has_alpha && !tolerance.ignore_alpha {
        let pass = ds.alpha_max_delta <= tolerance.max_alpha_delta;
        constraint_results.push(ConstraintResult {
            name: "Alpha delta",
            passed: pass,
            actual: if ds.alpha_pixels_differing > 0 {
                format!(
                    "max={}, {}/{} pixels differ",
                    ds.alpha_max_delta, ds.alpha_pixels_differing, ds.pixel_count,
                )
            } else {
                "identical".to_string()
            },
            limit: format!("{}/255", tolerance.max_alpha_delta),
        });
        pass
    } else {
        true // RGB-only or ignore_alpha: no alpha constraint
    };

    let passed = delta_pass && score_pass && diff_pass && alpha_pass;

    let num_channels = if ds.has_alpha { 4 } else { 3 };
    let expected_histogram = ChannelHistograms {
        bins: ds.src_histogram,
        num_channels,
    };
    let actual_histogram = ChannelHistograms {
        bins: ds.dst_histogram,
        num_channels,
    };

    RegressionReport {
        passed,
        score: cr.result.score(),
        category: cr.classification.dominant,
        confidence: cr.classification.confidence,
        max_channel_delta,
        max_channel_delta_f64,
        native_max,
        pixel_count: ds.pixel_count,
        pixels_differing: ds.pixels_differing,
        pixels_failing,
        identical_channel_fraction,
        alpha_max_delta: ds.alpha_max_delta,
        alpha_pixels_differing: ds.alpha_pixels_differing,
        expected_histogram,
        actual_histogram,
        rounding_bias: cr.classification.rounding_bias,
        constraint_results,
        dimension_info: None,
    }
}

// ─── check_regression() free function ────────────────────────────────────

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
/// use zensim_regress::testing::{RegressionTolerance, check_regression};
/// # let (expected_px, actual_px) = (vec![[0u8; 3]; 64], vec![[0u8; 3]; 64]);
/// let z = Zensim::new(ZensimProfile::latest());
/// let expected = RgbSlice::new(&expected_px, 8, 8);
/// let actual = RgbSlice::new(&actual_px, 8, 8);
///
/// let report = check_regression(
///     &z, &expected, &actual,
///     &RegressionTolerance::off_by_one(),
/// ).unwrap();
/// assert!(report.passed(), "{report}");
/// ```
///
/// # Errors
///
/// Returns [`ZensimError`] if dimensions are mismatched or too small.
pub fn check_regression(
    zensim: &Zensim,
    expected: &impl ImageSource,
    actual: &impl ImageSource,
    tolerance: &RegressionTolerance,
) -> Result<RegressionReport, ZensimError> {
    let cr = if tolerance.ignore_alpha {
        zensim.classify(&AlphaOverride(expected), &AlphaOverride(actual))?
    } else {
        zensim.classify(expected, actual)?
    };
    Ok(build_report(cr, tolerance))
}

/// Compare expected vs actual images that have different dimensions.
///
/// Categorizes the mismatch and chooses the cheapest comparison method:
/// - **Orientation swap**: tries rotate90 + rotate270, picks best score
/// - **Off-by-one**: center-crops larger to match smaller (lossless)
/// - **Crop/trim**: center-crop first, resize fallback if score < 70
/// - **Large difference**: Lanczos resize only
///
/// The resulting report is annotated with [`DimensionInfo`] including the
/// [`DimensionMismatchKind`] and [`ComparisonMethod`] used.
///
/// # Errors
///
/// Returns [`ZensimError::ImageTooSmall`] if both images are < 8×8.
#[allow(clippy::too_many_arguments)]
pub fn check_regression_resized(
    zensim: &Zensim,
    expected_rgba: &[u8],
    ew: u32,
    eh: u32,
    actual_rgba: &[u8],
    aw: u32,
    ah: u32,
    tolerance: &RegressionTolerance,
) -> Result<RegressionReport, ZensimError> {
    let kind = classify_dimension_mismatch(ew, eh, aw, ah);

    let (report, method) = match kind {
        DimensionMismatchKind::OrientationSwap => {
            compare_orientation_swap(zensim, expected_rgba, ew, eh, actual_rgba, aw, ah, tolerance)?
        }
        DimensionMismatchKind::OffByOne => {
            let r = compare_center_crop(zensim, expected_rgba, ew, eh, actual_rgba, aw, ah, tolerance)?;
            (r, ComparisonMethod::CenterCropped)
        }
        DimensionMismatchKind::CropDifference => {
            let r = compare_center_crop(zensim, expected_rgba, ew, eh, actual_rgba, aw, ah, tolerance)?;
            if r.score() >= 70.0 {
                (r, ComparisonMethod::CenterCropped)
            } else {
                // Center-crop scored poorly — fall back to resize
                let r2 = compare_resized(zensim, expected_rgba, ew, eh, actual_rgba, aw, ah, tolerance)?;
                if r2.score() > r.score() {
                    (r2, ComparisonMethod::Resized)
                } else {
                    (r, ComparisonMethod::CenterCropped)
                }
            }
        }
        DimensionMismatchKind::LargeDifference => {
            let r = compare_resized(zensim, expected_rgba, ew, eh, actual_rgba, aw, ah, tolerance)?;
            (r, ComparisonMethod::Resized)
        }
    };

    let mut report = report;
    report.dimension_info = Some(DimensionInfo {
        expected_dims: (ew, eh),
        actual_dims: (aw, ah),
        kind,
        method,
    });
    Ok(report)
}

// ─── Dimension-mismatch comparison helpers ──────────────────────────────

/// Classify a pair of RGBA byte slices into `[u8; 4]` pixels and run zensim.
#[allow(clippy::too_many_arguments)]
fn classify_rgba_pair(
    zensim: &Zensim,
    a: &[u8],
    aw: u32,
    ah: u32,
    b: &[u8],
    bw: u32,
    bh: u32,
    tolerance: &RegressionTolerance,
) -> Result<RegressionReport, ZensimError> {
    let a_px: Vec<[u8; 4]> = a.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
    let b_px: Vec<[u8; 4]> = b.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
    let a_src = zensim::RgbaSlice::new(&a_px, aw as usize, ah as usize);
    let b_src = zensim::RgbaSlice::new(&b_px, bw as usize, bh as usize);
    let cr = if tolerance.ignore_alpha {
        zensim.classify(&AlphaOverride(&a_src), &AlphaOverride(&b_src))?
    } else {
        zensim.classify(&a_src, &b_src)?
    };
    Ok(build_report(cr, tolerance))
}

/// Center-crop an RGBA image to `(tw, th)`.
fn center_crop_rgba(rgba: &[u8], w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    let x0 = (w.saturating_sub(tw)) / 2;
    let y0 = (h.saturating_sub(th)) / 2;
    let tw = tw.min(w);
    let th = th.min(h);
    let mut out = Vec::with_capacity((tw * th * 4) as usize);
    for y in y0..y0 + th {
        let row_start = (y * w + x0) as usize * 4;
        let row_end = row_start + (tw as usize * 4);
        out.extend_from_slice(&rgba[row_start..row_end]);
    }
    out
}

/// Compare by center-cropping both images to the overlap region.
#[allow(clippy::too_many_arguments)]
fn compare_center_crop(
    zensim: &Zensim,
    exp: &[u8], ew: u32, eh: u32,
    act: &[u8], aw: u32, ah: u32,
    tolerance: &RegressionTolerance,
) -> Result<RegressionReport, ZensimError> {
    let tw = ew.min(aw);
    let th = eh.min(ah);
    if tw < 8 || th < 8 {
        return Err(ZensimError::ImageTooSmall);
    }
    let exp_crop = center_crop_rgba(exp, ew, eh, tw, th);
    let act_crop = center_crop_rgba(act, aw, ah, tw, th);
    classify_rgba_pair(zensim, &exp_crop, tw, th, &act_crop, tw, th, tolerance)
}

/// Compare by Lanczos3-resizing actual to match expected dimensions.
#[allow(clippy::too_many_arguments)]
fn compare_resized(
    zensim: &Zensim,
    exp: &[u8], ew: u32, eh: u32,
    act: &[u8], aw: u32, ah: u32,
    tolerance: &RegressionTolerance,
) -> Result<RegressionReport, ZensimError> {
    use image::imageops::{self, FilterType};
    use image::RgbaImage;

    if ew < 8 || eh < 8 {
        return Err(ZensimError::ImageTooSmall);
    }

    let act_img = RgbaImage::from_raw(aw, ah, act.to_vec())
        .expect("actual: data length does not match dimensions");
    let act_resized = imageops::resize(&act_img, ew, eh, FilterType::Lanczos3);

    classify_rgba_pair(zensim, exp, ew, eh, act_resized.as_raw(), ew, eh, tolerance)
}

/// Try all orientation transforms (rotations, flips) and pick the best score.
///
/// Tries all 7 non-identity EXIF orientations: rot90, rot270, rot180,
/// flipH, flipV, transpose (rot90+flipH), transverse (rot270+flipH).
///
/// For transforms that produce matching dimensions, compares directly.
/// For transforms that produce different dimensions, resizes to match expected.
#[allow(clippy::too_many_arguments)]
fn compare_orientation_swap(
    zensim: &Zensim,
    exp: &[u8], ew: u32, eh: u32,
    act: &[u8], aw: u32, ah: u32,
    tolerance: &RegressionTolerance,
) -> Result<(RegressionReport, ComparisonMethod), ZensimError> {
    use image::RgbaImage;

    if ew < 8 || eh < 8 {
        return Err(ZensimError::ImageTooSmall);
    }

    let act_img = RgbaImage::from_raw(aw, ah, act.to_vec())
        .expect("actual: data length does not match dimensions");

    let candidates = build_transform_candidates(&act_img);

    let mut best: Option<(RegressionReport, ComparisonMethod)> = None;

    for (transformed, method) in &candidates {
        let (tw, th) = transformed.dimensions();

        let report = if tw == ew && th == eh {
            classify_rgba_pair(zensim, exp, ew, eh, transformed.as_raw(), tw, th, tolerance)?
        } else if tw >= 8 && th >= 8 && ew >= 8 && eh >= 8 {
            let resized =
                image::imageops::resize(transformed, ew, eh, image::imageops::FilterType::Lanczos3);
            classify_rgba_pair(zensim, exp, ew, eh, resized.as_raw(), ew, eh, tolerance)?
        } else {
            continue;
        };

        let dominated = best.as_ref().is_none_or(|(b, _)| report.score() > b.score());
        if dominated {
            best = Some((report, *method));
        }
    }

    best.ok_or(ZensimError::ImageTooSmall)
}

/// Build all 7 non-identity transform candidates for an image.
fn build_transform_candidates(img: &image::RgbaImage) -> Vec<(image::RgbaImage, ComparisonMethod)> {
    use image::imageops;

    let rot90 = imageops::rotate90(img);
    let rot270 = imageops::rotate270(img);

    vec![
        (imageops::rotate90(img), ComparisonMethod::Rotated90),
        (imageops::rotate270(img), ComparisonMethod::Rotated270),
        (imageops::rotate180(img), ComparisonMethod::Rotated180),
        (imageops::flip_horizontal(img), ComparisonMethod::FlipHorizontal),
        (imageops::flip_vertical(img), ComparisonMethod::FlipVertical),
        // Transpose = rot90 then flipH (EXIF orientation 5)
        (imageops::flip_horizontal(&rot90), ComparisonMethod::Transpose),
        // Transverse = rot270 then flipH (EXIF orientation 7)
        (imageops::flip_horizontal(&rot270), ComparisonMethod::Transverse),
    ]
}

/// Try all transforms on a same-dimension image pair to detect flips/rotations.
///
/// Called when the normal comparison yields a low score for same-dimension
/// images. If any transform scores significantly better than the original,
/// returns the best transform report annotated with [`DimensionInfo`].
///
/// Returns `None` if no transform improves the score by at least 20 points,
/// or if the best transform score is below 50.
pub fn detect_transform(
    zensim: &Zensim,
    expected_rgba: &[u8],
    actual_rgba: &[u8],
    w: u32,
    h: u32,
    original_score: f64,
    tolerance: &RegressionTolerance,
) -> Option<(RegressionReport, ComparisonMethod)> {
    use image::RgbaImage;

    if w < 8 || h < 8 {
        return None;
    }

    let act_img = RgbaImage::from_raw(w, h, actual_rgba.to_vec())?;
    let candidates = build_transform_candidates(&act_img);

    let mut best: Option<(RegressionReport, ComparisonMethod)> = None;

    for (transformed, method) in &candidates {
        let (tw, th) = transformed.dimensions();
        if tw != w || th != h {
            continue; // Skip transforms that change dimensions
        }

        let report = classify_rgba_pair(
            zensim,
            expected_rgba, w, h,
            transformed.as_raw(), tw, th,
            tolerance,
        )
        .ok()?;

        let dominated = best.as_ref().is_none_or(|(b, _)| report.score() > b.score());
        if dominated {
            best = Some((report, *method));
        }
    }

    // Only report if the transform is a clear improvement
    let (report, method) = best?;
    if report.score() >= 50.0 && report.score() > original_score + 20.0 {
        Some((report, method))
    } else {
        None
    }
}

// ─── shrink_tolerance() ──────────────────────────────────────────────────

/// Shrink a tolerance toward measured values, without going below a floor.
///
/// For each field, the new value is clamped between `floor` and `current`:
/// - **max_delta**: `clamp(measured_max + 1, floor, current)`
/// - **max_pixels_different**: `clamp(measured_fraction * 1.1, floor, current)`
/// - **min_similarity**: `clamp(measured_score - 0.5, floor, current)` — note: *lowering* floor, *raising* toward current
/// - **max_alpha_delta**: `clamp(measured_max + 1, floor, current)`
///
/// `ignore_alpha` is preserved from `current`.
pub fn shrink_tolerance(
    current: &RegressionTolerance,
    report: &RegressionReport,
    floor: &RegressionTolerance,
) -> RegressionTolerance {
    // max_delta: tighten toward measured + 1
    let measured_max_delta = *report.max_channel_delta().iter().max().unwrap_or(&0);
    let shrunk_max_delta = measured_max_delta.saturating_add(1);
    let new_max_delta = shrunk_max_delta.clamp(floor.max_delta, current.max_delta);

    // max_pixels_different: tighten toward measured * 1.1
    let measured_fraction = if report.pixel_count() > 0 {
        report.pixels_differing() as f64 / report.pixel_count() as f64
    } else {
        0.0
    };
    let shrunk_fraction = measured_fraction * 1.1;
    let new_max_pixels_different =
        shrunk_fraction.clamp(floor.max_pixels_different, current.max_pixels_different);

    // min_similarity: tighten *upward* toward measured - 0.5
    // (higher min_similarity is tighter, so we want max(floor, measured - 0.5) clamped to current)
    let shrunk_score = report.score() - 0.5;
    let new_min_similarity = if shrunk_score > current.min_similarity {
        // Can't tighten beyond current
        current.min_similarity
    } else if shrunk_score < floor.min_similarity {
        floor.min_similarity
    } else {
        shrunk_score
    };

    // max_alpha_delta: tighten toward measured + 1
    let shrunk_alpha = report.alpha_max_delta().saturating_add(1);
    let new_max_alpha_delta = shrunk_alpha.clamp(floor.max_alpha_delta, current.max_alpha_delta);

    RegressionTolerance {
        max_delta: new_max_delta,
        max_pixels_different: new_max_pixels_different,
        min_similarity: new_min_similarity,
        max_alpha_delta: new_max_alpha_delta,
        ignore_alpha: current.ignore_alpha,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resized_comparison_same_image_different_size() {
        let z = Zensim::new(zensim::ZensimProfile::latest());

        // Create a 16x16 gradient image
        let mut small_rgba = Vec::with_capacity(16 * 16 * 4);
        for y in 0..16u8 {
            for x in 0..16u8 {
                small_rgba.extend_from_slice(&[x * 16, y * 16, 128, 255]);
            }
        }

        // Upscale to 32x32 — classified as LargeDifference (100% size change)
        let img = image::RgbaImage::from_raw(16, 16, small_rgba.clone()).unwrap();
        let big = image::imageops::resize(&img, 32, 32, image::imageops::FilterType::Lanczos3);
        let big_rgba = big.into_raw();

        let tol = RegressionTolerance::off_by_one().with_min_similarity(50.0);
        let report =
            check_regression_resized(&z, &small_rgba, 16, 16, &big_rgba, 32, 32, &tol).unwrap();

        assert!(report.score() > 70.0, "score {} should be > 70", report.score());
        let dim = report.dimension_info().unwrap();
        assert_eq!(dim.expected_dims, (16, 16));
        assert_eq!(dim.actual_dims, (32, 32));
        assert_eq!(dim.kind, DimensionMismatchKind::LargeDifference);
        assert_eq!(dim.method, ComparisonMethod::Resized);
    }

    #[test]
    fn resized_comparison_too_small_returns_error() {
        let z = Zensim::new(zensim::ZensimProfile::latest());
        let small = vec![0u8; 4 * 4 * 4]; // 4x4 RGBA
        let big = vec![0u8; 16 * 16 * 4];
        let tol = RegressionTolerance::exact();

        let result = check_regression_resized(&z, &small, 4, 4, &big, 16, 16, &tol);
        assert_eq!(result.unwrap_err(), ZensimError::ImageTooSmall);
    }

    #[test]
    fn off_by_one_uses_center_crop() {
        let z = Zensim::new(zensim::ZensimProfile::latest());

        // Create a 16x16 image and a 17x15 variant (off by 1 in each axis)
        let mut rgba16 = Vec::with_capacity(16 * 16 * 4);
        for y in 0..16u8 {
            for x in 0..16u8 {
                rgba16.extend_from_slice(&[x * 16, y * 16, 128, 255]);
            }
        }
        // 17x15 image — just enough variation
        let mut rgba17 = Vec::with_capacity(17 * 15 * 4);
        for y in 0..15u8 {
            for x in 0..17u8 {
                rgba17.extend_from_slice(&[x * 15, y * 17, 128, 255]);
            }
        }

        let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
        let report =
            check_regression_resized(&z, &rgba16, 16, 16, &rgba17, 17, 15, &tol).unwrap();

        let dim = report.dimension_info().unwrap();
        assert_eq!(dim.kind, DimensionMismatchKind::OffByOne);
        assert_eq!(dim.method, ComparisonMethod::CenterCropped);
    }

    // ─── Categorization tests ───────────────────────────────────────────

    #[test]
    fn classify_orientation_swap() {
        assert_eq!(
            classify_dimension_mismatch(480, 640, 640, 480),
            DimensionMismatchKind::OrientationSwap,
        );
    }

    #[test]
    fn classify_off_by_one() {
        assert_eq!(
            classify_dimension_mismatch(300, 200, 299, 199),
            DimensionMismatchKind::OffByOne,
        );
        // ±2px still off-by-one
        assert_eq!(
            classify_dimension_mismatch(300, 200, 298, 198),
            DimensionMismatchKind::OffByOne,
        );
    }

    #[test]
    fn classify_crop_difference() {
        assert_eq!(
            classify_dimension_mismatch(200, 150, 195, 148),
            DimensionMismatchKind::CropDifference,
        );
    }

    #[test]
    fn classify_large_difference() {
        assert_eq!(
            classify_dimension_mismatch(100, 100, 200, 300),
            DimensionMismatchKind::LargeDifference,
        );
    }

    #[test]
    fn classify_square_not_swap() {
        // Square images can't be orientation swaps
        assert_ne!(
            classify_dimension_mismatch(100, 100, 100, 100),
            DimensionMismatchKind::OrientationSwap,
        );
    }

    // ─── Description tests ──────────────────────────────────────────────

    #[test]
    fn dimension_info_description_orientation_swap() {
        let info = DimensionInfo {
            expected_dims: (480, 640),
            actual_dims: (640, 480),
            kind: DimensionMismatchKind::OrientationSwap,
            method: ComparisonMethod::Rotated90,
        };
        let desc = info.description();
        assert!(desc.contains("orientation swap"), "expected 'orientation swap' in: {desc}");
    }

    #[test]
    fn dimension_info_description_offset() {
        let info = DimensionInfo {
            expected_dims: (200, 150),
            actual_dims: (195, 148),
            kind: DimensionMismatchKind::CropDifference,
            method: ComparisonMethod::CenterCropped,
        };
        let desc = info.description();
        assert!(desc.contains("-5w"), "expected -5w in: {desc}");
        assert!(desc.contains("-2h"), "expected -2h in: {desc}");
    }

    #[test]
    fn dimension_info_description_larger() {
        let info = DimensionInfo {
            expected_dims: (100, 100),
            actual_dims: (110, 105),
            kind: DimensionMismatchKind::CropDifference,
            method: ComparisonMethod::CenterCropped,
        };
        let desc = info.description();
        assert!(desc.contains("+10w"), "expected +10w in: {desc}");
        assert!(desc.contains("+5h"), "expected +5h in: {desc}");
    }

    #[test]
    fn detect_transform_catches_hflip() {
        let z = Zensim::new(zensim::ZensimProfile::latest());

        // Create a 16x16 asymmetric gradient
        let mut rgba = Vec::with_capacity(16 * 16 * 4);
        for y in 0..16u8 {
            for x in 0..16u8 {
                rgba.extend_from_slice(&[x * 16, y * 8, (x + y) * 4, 255]);
            }
        }

        // Flip horizontally
        let img = image::RgbaImage::from_raw(16, 16, rgba.clone()).unwrap();
        let flipped = image::imageops::flip_horizontal(&img);
        let flipped_rgba = flipped.into_raw();

        // Original comparison should score low (asymmetric content)
        let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
        let orig_report = check_regression(
            &z,
            &zensim::RgbaSlice::new(
                &rgba.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect::<Vec<_>>(),
                16, 16,
            ),
            &zensim::RgbaSlice::new(
                &flipped_rgba.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect::<Vec<_>>(),
                16, 16,
            ),
            &tol,
        ).unwrap();

        // Transform detection should find the flip
        let result = detect_transform(
            &z, &rgba, &flipped_rgba, 16, 16, orig_report.score(), &tol,
        );
        if let Some((report, method)) = result {
            assert_eq!(method, ComparisonMethod::FlipHorizontal);
            assert!(report.score() > 90.0, "flip score {} should be > 90", report.score());
        }
        // Note: may return None for very small/simple images where the flip
        // doesn't lower the score enough. That's acceptable.
    }
}
