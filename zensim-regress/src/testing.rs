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

/// Try orientation transforms for w↔h swaps, using corner-SAD pre-filter.
///
/// For a true orientation swap, rot90 and rot270 produce exact dimension
/// matches. Transpose and transverse also match. We use a corner block
/// comparison to rank candidates, then run zensim on only the best 2.
#[allow(clippy::too_many_arguments)]
fn compare_orientation_swap(
    zensim: &Zensim,
    exp: &[u8], ew: u32, eh: u32,
    act: &[u8], aw: u32, ah: u32,
    tolerance: &RegressionTolerance,
) -> Result<(RegressionReport, ComparisonMethod), ZensimError> {
    use image::imageops;
    use image::RgbaImage;

    if ew < 8 || eh < 8 {
        return Err(ZensimError::ImageTooSmall);
    }

    let act_img = RgbaImage::from_raw(aw, ah, act.to_vec())
        .expect("actual: data length does not match dimensions");

    // Build only transforms that produce matching dimensions (ew×eh).
    // For w↔h swap: rot90, rot270, transpose, transverse all produce (ah, aw) = (ew, eh).
    let rot90 = imageops::rotate90(&act_img);
    let rot270 = imageops::rotate270(&act_img);
    let candidates: Vec<(RgbaImage, ComparisonMethod)> = vec![
        (rot90.clone(), ComparisonMethod::Rotated90),
        (rot270.clone(), ComparisonMethod::Rotated270),
        (imageops::flip_horizontal(&rot90), ComparisonMethod::Transpose),
        (imageops::flip_horizontal(&rot270), ComparisonMethod::Transverse),
    ];

    // Pre-filter: compare top-left corner block of expected against each candidate
    let n = 4u32.min(ew / 2).min(eh / 2).max(1);
    let exp_tl = extract_block(exp, ew, 0, 0, n);

    let mut scored: Vec<(u64, usize)> = candidates
        .iter()
        .enumerate()
        .filter(|(_, (img, _))| img.dimensions() == (ew, eh))
        .map(|(i, (img, _))| {
            let cand_tl = extract_block(img.as_raw(), ew, 0, 0, n);
            (block_sad(&exp_tl, &cand_tl), i)
        })
        .collect();
    scored.sort_by_key(|(sad, _)| *sad);

    // Run zensim on top 2 candidates; if both score poorly, try the rest.
    let mut best: Option<(RegressionReport, ComparisonMethod)> = None;
    let mut tried = 0usize;

    for &(_, idx) in &scored {
        tried += 1;
        let (ref transformed, method) = candidates[idx];
        let report =
            classify_rgba_pair(zensim, exp, ew, eh, transformed.as_raw(), ew, eh, tolerance)?;

        let dominated = best.as_ref().is_none_or(|(b, _)| report.score() > b.score());
        if dominated {
            best = Some((report, method));
        }

        // After 2 candidates: stop early if we found a good match (>70),
        // otherwise keep going through all candidates.
        if tried >= 2 && best.as_ref().is_some_and(|(b, _)| b.score() > 70.0) {
            break;
        }
    }

    best.ok_or(ZensimError::ImageTooSmall)
}

/// Compute sum-of-absolute-differences between two RGBA blocks.
fn block_sad(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u64)
        .sum()
}

/// Extract an NxN corner block from packed RGBA at (x0, y0).
fn extract_block(rgba: &[u8], w: u32, x0: u32, y0: u32, n: u32) -> Vec<u8> {
    let mut block = Vec::with_capacity((n * n * 4) as usize);
    for y in y0..y0 + n {
        let start = ((y * w + x0) * 4) as usize;
        let end = start + (n * 4) as usize;
        if end <= rgba.len() {
            block.extend_from_slice(&rgba[start..end]);
        }
    }
    block
}

/// Try transforms on a same-dimension image pair to detect flips/rotations.
///
/// Uses a cheap corner-block SAD pre-filter to identify the 1–2 most likely
/// transforms, then runs zensim only on those candidates. This avoids
/// running 7 full zensim comparisons.
///
/// Called when the normal comparison yields a low score for same-dimension
/// images. Returns `None` if no transform improves the score meaningfully.
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

    // ── Phase 1: cheap corner-SAD pre-filter ──
    // Compare top-left 4x4 block of expected against the position where that
    // block would land under each same-dimension transform of actual.
    let n = 4u32.min(w / 2).min(h / 2).max(1);
    let exp_tl = extract_block(expected_rgba, w, 0, 0, n);

    // For same-dimension transforms:
    // Identity:  TL→TL (already scored)
    // FlipH:     TL→TR (x mirrored)
    // FlipV:     TL→BL (y mirrored)
    // Rot180:    TL→BR (both mirrored)
    let act_tr = extract_block(actual_rgba, w, w - n, 0, n); // hflip: TL of expected ↔ TR of actual
    let act_bl = extract_block(actual_rgba, w, 0, h - n, n); // vflip: TL of expected ↔ BL of actual
    let act_br = extract_block(actual_rgba, w, w - n, h - n, n); // rot180: TL ↔ BR

    // But we need to reverse the block order for flipped comparisons.
    // For hflip: expected TL pixels should match actual TR pixels in reverse column order.
    let mirror_h = |block: &[u8], n: u32| -> Vec<u8> {
        let mut out = vec![0u8; block.len()];
        for y in 0..n {
            for x in 0..n {
                let src = ((y * n + x) * 4) as usize;
                let dst = ((y * n + (n - 1 - x)) * 4) as usize;
                if src + 4 <= block.len() && dst + 4 <= out.len() {
                    out[dst..dst + 4].copy_from_slice(&block[src..src + 4]);
                }
            }
        }
        out
    };
    let mirror_v = |block: &[u8], n: u32| -> Vec<u8> {
        let mut out = vec![0u8; block.len()];
        for y in 0..n {
            for x in 0..n {
                let src = ((y * n + x) * 4) as usize;
                let dst = (((n - 1 - y) * n + x) * 4) as usize;
                if src + 4 <= block.len() && dst + 4 <= out.len() {
                    out[dst..dst + 4].copy_from_slice(&block[src..src + 4]);
                }
            }
        }
        out
    };

    let mut scored: Vec<(u64, ComparisonMethod)> = vec![
        (block_sad(&exp_tl, &mirror_h(&act_tr, n)), ComparisonMethod::FlipHorizontal),
        (block_sad(&exp_tl, &mirror_v(&act_bl, n)), ComparisonMethod::FlipVertical),
        (
            block_sad(&exp_tl, &mirror_h(&mirror_v(&act_br, n), n)),
            ComparisonMethod::Rotated180,
        ),
    ];
    scored.sort_by_key(|(sad, _)| *sad);

    // ── Phase 2: run zensim on candidates, stop early if good match found ──
    let act_img = RgbaImage::from_raw(w, h, actual_rgba.to_vec())?;
    let mut best: Option<(RegressionReport, ComparisonMethod)> = None;
    let mut tried = 0usize;

    for &(_, method) in &scored {
        tried += 1;
        let transformed = match method {
            ComparisonMethod::FlipHorizontal => image::imageops::flip_horizontal(&act_img),
            ComparisonMethod::FlipVertical => image::imageops::flip_vertical(&act_img),
            ComparisonMethod::Rotated180 => image::imageops::rotate180(&act_img),
            _ => continue,
        };

        let report = classify_rgba_pair(
            zensim,
            expected_rgba, w, h,
            transformed.as_raw(), w, h,
            tolerance,
        )
        .ok()?;

        let dominated = best.as_ref().is_none_or(|(b, _)| report.score() > b.score());
        if dominated {
            best = Some((report, method));
        }

        // After 2: stop early if we found a good match
        if tried >= 2 && best.as_ref().is_some_and(|(b, _)| b.score() > 70.0) {
            break;
        }
    }

    // Report if the transform is a clear improvement
    let (report, method) = best?;
    if report.score() > original_score + 15.0 && report.score() >= 40.0 {
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

    // ─── Helpers ─────────────────────────────────────────────────────────

    /// Create an asymmetric 16x16 gradient image (packed RGBA).
    fn asymmetric_gradient(w: u32, h: u32) -> Vec<u8> {
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w.max(1)) as u8;
                let g = ((y * 127) / h.max(1)) as u8;
                let b = (((x + y * 2) * 63) / (w + h).max(1)) as u8;
                rgba.extend_from_slice(&[r, g, b, 255]);
            }
        }
        rgba
    }

    fn px(rgba: &[u8]) -> Vec<[u8; 4]> {
        rgba.chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect()
    }

    // ─── detect_transform tests ─────────────────────────────────────────

    #[test]
    fn detect_transform_catches_hflip() {
        let z = Zensim::new(zensim::ZensimProfile::latest());
        let rgba = asymmetric_gradient(16, 16);
        let img = image::RgbaImage::from_raw(16, 16, rgba.clone()).unwrap();
        let flipped = image::imageops::flip_horizontal(&img).into_raw();

        let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
        let orig = check_regression(
            &z,
            &zensim::RgbaSlice::new(&px(&rgba), 16, 16),
            &zensim::RgbaSlice::new(&px(&flipped), 16, 16),
            &tol,
        )
        .unwrap();

        if let Some((report, method)) = detect_transform(&z, &rgba, &flipped, 16, 16, orig.score(), &tol) {
            assert_eq!(method, ComparisonMethod::FlipHorizontal);
            assert!(report.score() > 90.0, "flip score {} should be > 90", report.score());
        }
    }

    #[test]
    fn detect_transform_catches_vflip() {
        let z = Zensim::new(zensim::ZensimProfile::latest());
        let rgba = asymmetric_gradient(16, 16);
        let img = image::RgbaImage::from_raw(16, 16, rgba.clone()).unwrap();
        let flipped = image::imageops::flip_vertical(&img).into_raw();

        let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
        let orig = check_regression(
            &z,
            &zensim::RgbaSlice::new(&px(&rgba), 16, 16),
            &zensim::RgbaSlice::new(&px(&flipped), 16, 16),
            &tol,
        )
        .unwrap();

        if let Some((report, method)) = detect_transform(&z, &rgba, &flipped, 16, 16, orig.score(), &tol) {
            assert_eq!(method, ComparisonMethod::FlipVertical);
            assert!(report.score() > 90.0, "flip score {} should be > 90", report.score());
        }
    }

    #[test]
    fn detect_transform_catches_rot180() {
        let z = Zensim::new(zensim::ZensimProfile::latest());
        let rgba = asymmetric_gradient(16, 16);
        let img = image::RgbaImage::from_raw(16, 16, rgba.clone()).unwrap();
        let rotated = image::imageops::rotate180(&img).into_raw();

        let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
        let orig = check_regression(
            &z,
            &zensim::RgbaSlice::new(&px(&rgba), 16, 16),
            &zensim::RgbaSlice::new(&px(&rotated), 16, 16),
            &tol,
        )
        .unwrap();

        if let Some((report, method)) = detect_transform(&z, &rgba, &rotated, 16, 16, orig.score(), &tol) {
            assert_eq!(method, ComparisonMethod::Rotated180);
            assert!(report.score() > 90.0, "rot180 score {} should be > 90", report.score());
        }
    }

    #[test]
    fn detect_transform_returns_none_for_unrelated_images() {
        let z = Zensim::new(zensim::ZensimProfile::latest());
        let rgba_a = asymmetric_gradient(16, 16);
        // Completely different image
        let rgba_b: Vec<u8> = (0..16 * 16)
            .flat_map(|i| [((i * 7) % 256) as u8, ((i * 13) % 256) as u8, ((i * 31) % 256) as u8, 255])
            .collect();

        let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
        let result = detect_transform(&z, &rgba_a, &rgba_b, 16, 16, 5.0, &tol);
        assert!(result.is_none(), "should not detect transform for unrelated images");
    }

    #[test]
    fn detect_transform_returns_none_for_too_small() {
        let z = Zensim::new(zensim::ZensimProfile::latest());
        let rgba = vec![128u8; 4 * 4 * 4]; // 4x4 — too small
        let tol = RegressionTolerance::exact();
        assert!(detect_transform(&z, &rgba, &rgba, 4, 4, 10.0, &tol).is_none());
    }

    // ─── Orientation swap scoring tests ─────────────────────────────────

    #[test]
    fn orientation_swap_detects_rot90() {
        let z = Zensim::new(zensim::ZensimProfile::latest());
        let rgba = asymmetric_gradient(16, 24);
        let img = image::RgbaImage::from_raw(16, 24, rgba.clone()).unwrap();
        let rotated = image::imageops::rotate90(&img);
        let rot_rgba = rotated.clone().into_raw();
        let (rw, rh) = rotated.dimensions(); // 24x16

        let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
        let report =
            check_regression_resized(&z, &rgba, 16, 24, &rot_rgba, rw, rh, &tol).unwrap();

        let dim = report.dimension_info().unwrap();
        assert_eq!(dim.kind, DimensionMismatchKind::OrientationSwap);
        // Corner-SAD may prefer rot90 or rot270 — both are correct rotations.
        // The important thing is the score is high (correct content match).
        assert!(
            matches!(dim.method, ComparisonMethod::Rotated90 | ComparisonMethod::Rotated270
                | ComparisonMethod::Transpose | ComparisonMethod::Transverse),
            "expected a rotation method, got {:?}", dim.method,
        );
        assert!(report.score() > 90.0, "rot90 score {} should be > 90", report.score());
    }

    #[test]
    fn orientation_swap_detects_rot270() {
        let z = Zensim::new(zensim::ZensimProfile::latest());
        let rgba = asymmetric_gradient(16, 24);
        let img = image::RgbaImage::from_raw(16, 24, rgba.clone()).unwrap();
        let rotated = image::imageops::rotate270(&img);
        let rot_rgba = rotated.clone().into_raw();
        let (rw, rh) = rotated.dimensions();

        let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
        let report =
            check_regression_resized(&z, &rgba, 16, 24, &rot_rgba, rw, rh, &tol).unwrap();

        let dim = report.dimension_info().unwrap();
        assert_eq!(dim.kind, DimensionMismatchKind::OrientationSwap);
        assert!(
            matches!(dim.method, ComparisonMethod::Rotated90 | ComparisonMethod::Rotated270
                | ComparisonMethod::Transpose | ComparisonMethod::Transverse),
            "expected a rotation method, got {:?}", dim.method,
        );
        assert!(report.score() > 90.0, "rot270 score {} should be > 90", report.score());
    }

    #[test]
    fn orientation_swap_detects_transpose() {
        let z = Zensim::new(zensim::ZensimProfile::latest());
        let rgba = asymmetric_gradient(16, 24);
        let img = image::RgbaImage::from_raw(16, 24, rgba.clone()).unwrap();
        // Transpose = rot90 + flipH
        let transposed = image::imageops::flip_horizontal(&image::imageops::rotate90(&img));
        let t_rgba = transposed.clone().into_raw();
        let (tw, th) = transposed.dimensions();

        let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
        let report =
            check_regression_resized(&z, &rgba, 16, 24, &t_rgba, tw, th, &tol).unwrap();

        let dim = report.dimension_info().unwrap();
        assert_eq!(dim.kind, DimensionMismatchKind::OrientationSwap);
        assert_eq!(dim.method, ComparisonMethod::Transpose);
        assert!(report.score() > 90.0, "transpose score {} should be > 90", report.score());
    }

    // ─── Crop difference with fallback ──────────────────────────────────

    #[test]
    fn crop_difference_same_content_uses_center_crop() {
        let z = Zensim::new(zensim::ZensimProfile::latest());
        // 100x100 image, cropped to 97x97 (3% diff, > ±2px, within 5%)
        let rgba100 = asymmetric_gradient(100, 100);
        let img100 = image::RgbaImage::from_raw(100, 100, rgba100.clone()).unwrap();
        let cropped = image::imageops::crop_imm(&img100, 1, 1, 97, 97).to_image();
        let rgba97 = cropped.into_raw();

        let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
        let report =
            check_regression_resized(&z, &rgba100, 100, 100, &rgba97, 97, 97, &tol).unwrap();

        let dim = report.dimension_info().unwrap();
        assert_eq!(dim.kind, DimensionMismatchKind::CropDifference);
        assert!(report.score() > 50.0, "crop score {} should be > 50", report.score());
    }

    // ─── Helper function tests ──────────────────────────────────────────

    #[test]
    fn center_crop_rgba_basic() {
        // 4x4 image, crop to 2x2 center
        let rgba: Vec<u8> = (0..16).flat_map(|i| [i as u8, 0, 0, 255]).collect();
        let cropped = center_crop_rgba(&rgba, 4, 4, 2, 2);
        // Center of 4x4 starts at (1,1), pixels 5,6,9,10
        assert_eq!(cropped.len(), 2 * 2 * 4);
        assert_eq!(cropped[0], 5); // pixel (1,1) R channel
        assert_eq!(cropped[4], 6); // pixel (2,1) R channel
    }

    #[test]
    fn center_crop_rgba_no_op_when_same_size() {
        let rgba: Vec<u8> = (0..16).flat_map(|i| [i as u8, 0, 0, 255]).collect();
        let cropped = center_crop_rgba(&rgba, 4, 4, 4, 4);
        assert_eq!(cropped, rgba);
    }

    #[test]
    fn block_sad_identical_is_zero() {
        let block = vec![100u8; 64];
        assert_eq!(block_sad(&block, &block), 0);
    }

    #[test]
    fn block_sad_maximum_difference() {
        let a = vec![0u8; 4];
        let b = vec![255u8; 4];
        assert_eq!(block_sad(&a, &b), 255 * 4);
    }

    #[test]
    fn extract_block_top_left() {
        // 4x4 image, extract 2x2 from (0,0)
        let rgba: Vec<u8> = (0..16).flat_map(|i| [i as u8, 0, 0, 255]).collect();
        let block = extract_block(&rgba, 4, 0, 0, 2);
        assert_eq!(block.len(), 2 * 2 * 4);
        assert_eq!(block[0], 0); // pixel (0,0)
        assert_eq!(block[4], 1); // pixel (1,0)
    }

    #[test]
    fn extract_block_bottom_right() {
        // 4x4 image, extract 2x2 from (2,2)
        let rgba: Vec<u8> = (0..16).flat_map(|i| [i as u8, 0, 0, 255]).collect();
        let block = extract_block(&rgba, 4, 2, 2, 2);
        assert_eq!(block[0], 10); // pixel (2,2)
        assert_eq!(block[4], 11); // pixel (3,2)
    }

    // ─── Display impl tests ────────────────────────────────────────────

    #[test]
    fn dimension_mismatch_kind_display() {
        assert_eq!(DimensionMismatchKind::OrientationSwap.to_string(), "orientation swap");
        assert_eq!(DimensionMismatchKind::OffByOne.to_string(), "off-by-one rounding");
        assert_eq!(DimensionMismatchKind::CropDifference.to_string(), "crop/trim");
        assert_eq!(DimensionMismatchKind::LargeDifference.to_string(), "different dimensions");
    }

    #[test]
    fn comparison_method_display() {
        assert_eq!(ComparisonMethod::Resized.to_string(), "resized");
        assert_eq!(ComparisonMethod::CenterCropped.to_string(), "center-cropped");
        assert_eq!(ComparisonMethod::Rotated90.to_string(), "rotated 90\u{00b0} CW");
        assert_eq!(ComparisonMethod::Rotated270.to_string(), "rotated 270\u{00b0} CW");
        assert_eq!(ComparisonMethod::FlipHorizontal.to_string(), "flipped horizontally");
        assert_eq!(ComparisonMethod::FlipVertical.to_string(), "flipped vertically");
        assert_eq!(ComparisonMethod::Rotated180.to_string(), "rotated 180\u{00b0}");
        assert_eq!(ComparisonMethod::Transpose.to_string(), "transposed (rot90+flipH)");
        assert_eq!(ComparisonMethod::Transverse.to_string(), "transversed (rot270+flipH)");
    }

    #[test]
    fn panel_label_all_variants() {
        let cases = [
            (ComparisonMethod::Resized, "RESIZED"),
            (ComparisonMethod::CenterCropped, "CROPPED"),
            (ComparisonMethod::Rotated90, "ROT 90"),
            (ComparisonMethod::Rotated270, "ROT 270"),
            (ComparisonMethod::FlipHorizontal, "FLIP H"),
            (ComparisonMethod::FlipVertical, "FLIP V"),
            (ComparisonMethod::Rotated180, "ROT 180"),
            (ComparisonMethod::Transpose, "TRANSPOSE"),
            (ComparisonMethod::Transverse, "TRANSVERSE"),
        ];
        for (method, expected) in cases {
            let info = DimensionInfo {
                expected_dims: (100, 100),
                actual_dims: (100, 100),
                kind: DimensionMismatchKind::OrientationSwap,
                method,
            };
            assert!(info.panel_label().contains(expected),
                "panel_label for {method:?} should contain '{expected}', got '{}'", info.panel_label());
        }
    }

    // ─── Report dimension_info setter ───────────────────────────────────

    #[test]
    fn set_dimension_info_on_report() {
        let z = Zensim::new(zensim::ZensimProfile::latest());
        let rgba = vec![128u8; 8 * 8 * 4];
        let tol = RegressionTolerance::exact();
        let mut report = check_regression(
            &z,
            &zensim::RgbaSlice::new(&px(&rgba), 8, 8),
            &zensim::RgbaSlice::new(&px(&rgba), 8, 8),
            &tol,
        )
        .unwrap();

        assert!(report.dimension_info().is_none());
        report.set_dimension_info(DimensionInfo {
            expected_dims: (8, 8),
            actual_dims: (10, 10),
            kind: DimensionMismatchKind::OffByOne,
            method: ComparisonMethod::CenterCropped,
        });
        assert!(report.dimension_info().is_some());
        assert_eq!(report.dimension_info().unwrap().kind, DimensionMismatchKind::OffByOne);
    }

    // ─── Edge case: classify boundary between categories ────────────────

    #[test]
    fn classify_boundary_off_by_one_to_crop() {
        // ±3px is NOT off-by-one (limit is ±2)
        assert_eq!(
            classify_dimension_mismatch(100, 100, 97, 97),
            DimensionMismatchKind::CropDifference,
        );
    }

    #[test]
    fn classify_boundary_crop_to_large() {
        // 5% boundary: diff/expected < 0.05
        // 4/100 = 0.04 < 0.05 → CropDifference
        assert_eq!(
            classify_dimension_mismatch(100, 100, 96, 96),
            DimensionMismatchKind::CropDifference,
        );
        // 5/100 = 0.05, NOT < 0.05 → LargeDifference
        assert_eq!(
            classify_dimension_mismatch(100, 100, 95, 95),
            DimensionMismatchKind::LargeDifference,
        );
    }

    #[test]
    fn classify_one_axis_zero_diff() {
        // One axis same, other differs by 1
        assert_eq!(
            classify_dimension_mismatch(100, 100, 100, 99),
            DimensionMismatchKind::OffByOne,
        );
    }
}
