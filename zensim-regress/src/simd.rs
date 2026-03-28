//! SIMD consistency testing via archmage token permutations.
//!
//! Runs an image operation under every available SIMD tier (AVX-512, AVX2,
//! SSE2, NEON, scalar) and verifies all tiers produce equivalent output.
//! Catches vectorization bugs, accumulator ordering differences, and
//! NaN-handling divergence across SIMD implementations.
//!
//! Wraps [`archmage::testing::for_each_token_permutation()`] with
//! zensim-regress comparison, forensic diff output, and optional
//! `.checksums` integration.
//!
//! Requires the `archmage` feature.
//!
//! # Which zen crates should use this?
//!
//! Any crate using `#[arcane]`, `#[autoversion]`, or `#[rite]`:
//! zenresize, zenfilters, zenpixels-convert, linear-srgb, zenjpeg,
//! zenwebp, zengif, zenpng, zenjxl-decoder, fast-ssim2, zensim.
//!
//! # Example
//!
//! ```rust,ignore
//! use zensim_regress::simd::*;
//! use zensim_regress::RegressionTolerance;
//! use archmage::testing::CompileTimePolicy;
//!
//! #[test]
//! fn resize_simd_consistency() {
//!     let input = load_test_image();
//!
//!     let report = check_simd_consistency(
//!         || {
//!             let output = resize(&input, 256, 256, Filter::Lanczos3);
//!             (output.to_rgba8(), 256, 256)
//!         },
//!         &RegressionTolerance::off_by_one(),
//!         CompileTimePolicy::Warn,
//!     ).unwrap();
//!
//!     assert!(report.all_passed, "{report}");
//! }
//! ```
//!
//! # CI integration
//!
//! For full permutation coverage in CI, compile with the `testable_dispatch`
//! feature on archmage and use `CompileTimePolicy::Fail`:
//!
//! ```toml
//! [dev-dependencies]
//! archmage = { version = "0.9", features = ["testable_dispatch"] }
//! zensim-regress = { version = "0.2", features = ["archmage"] }
//! ```

use std::fmt;

use archmage::testing::{CompileTimePolicy, PermutationReport, for_each_token_permutation};
use zensim::{RgbaSlice, Zensim, ZensimProfile};

use crate::error::RegressError;
use crate::testing::{RegressionReport, RegressionTolerance, check_regression};

// ─── Report types ──────────────���────────────────────────────────────────

/// Comparison of one SIMD tier against the reference (highest-tier) output.
#[derive(Debug, Clone)]
pub struct TierComparison {
    /// Human-readable tier label (e.g., "x86-64-v3 disabled").
    pub tier_label: String,
    /// Regression report comparing this tier's output against the reference.
    pub report: RegressionReport,
}

/// Result of running all SIMD tier permutations.
#[derive(Debug, Clone)]
pub struct SimdConsistencyReport {
    /// Label of the reference tier (first permutation, usually "all enabled").
    pub reference_tier: String,
    /// Total number of permutations executed.
    pub permutation_count: usize,
    /// Comparison of each non-reference tier against the reference.
    pub comparisons: Vec<TierComparison>,
    /// Whether all tier comparisons passed the tolerance.
    pub all_passed: bool,
    /// Underlying archmage permutation report (warnings, count).
    pub permutation_report: PermutationReport,
}

impl SimdConsistencyReport {
    /// Create a report for the degenerate case of a single permutation.
    fn single_tier(perm_report: PermutationReport) -> Self {
        Self {
            reference_tier: "all enabled".to_string(),
            permutation_count: 1,
            comparisons: Vec::new(),
            all_passed: true,
            permutation_report: perm_report,
        }
    }
}

impl fmt::Display for SimdConsistencyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.all_passed { "PASS" } else { "FAIL" };

        writeln!(
            f,
            "{status}: SIMD consistency across {} permutations (ref: {}).",
            self.permutation_count, self.reference_tier,
        )?;

        for cmp in &self.comparisons {
            let tier_status = if cmp.report.passed() { " " } else { "x" };
            writeln!(
                f,
                "  {tier_status} {}: score={:.1}, max_delta={:?}, category={:?}",
                cmp.tier_label,
                cmp.report.score(),
                cmp.report.max_channel_delta(),
                cmp.report.category(),
            )?;
        }

        for w in &self.permutation_report.warnings {
            writeln!(f, "  warning: {w}")?;
        }

        Ok(())
    }
}

// ─── Public API ────────��──────────────────────���─────────────────────────

/// Run `operation` under every SIMD token permutation and compare outputs.
///
/// The operation closure is called once per permutation. It must return
/// `(rgba_pixels, width, height)` — RGBA u8 format.
///
/// All outputs are compared against the first permutation (highest SIMD
/// tier, typically "all enabled") using the provided tolerance.
///
/// # Arguments
///
/// * `operation` — Closure that performs the image operation and returns
///   `(Vec<u8>, u32, u32)` — RGBA pixels, width, height.
/// * `tolerance` — How closely non-reference tiers must match the reference.
///   [`RegressionTolerance::off_by_one()`] is a good default for rounding
///   differences across SIMD implementations.
/// * `policy` — What to do when tokens can't be disabled (compile-time
///   guaranteed). Use `Warn` locally, `Fail` in CI with `testable_dispatch`.
///
/// # Errors
///
/// Returns [`RegressError`] if zensim comparison fails (dimension mismatch).
pub fn check_simd_consistency(
    operation: impl Fn() -> (Vec<u8>, u32, u32),
    tolerance: &RegressionTolerance,
    policy: CompileTimePolicy,
) -> Result<SimdConsistencyReport, RegressError> {
    let zensim = Zensim::new(ZensimProfile::latest());
    let mut outputs: Vec<(String, Vec<u8>, u32, u32)> = Vec::new();

    let perm_report = for_each_token_permutation(policy, |perm| {
        let (pixels, w, h) = operation();
        outputs.push((perm.label.clone(), pixels, w, h));
    });

    if outputs.len() < 2 {
        return Ok(SimdConsistencyReport::single_tier(perm_report));
    }

    let ref_label = outputs[0].0.clone();
    let ref_w = outputs[0].2 as usize;
    let ref_h = outputs[0].3 as usize;
    let ref_pixels: Vec<[u8; 4]> = outputs[0]
        .1
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    let ref_img = RgbaSlice::new(&ref_pixels, ref_w, ref_h);

    let mut comparisons = Vec::new();
    let mut all_passed = true;

    for (label, px, w, h) in &outputs[1..] {
        let actual_pixels: Vec<[u8; 4]> = px
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let actual_img = RgbaSlice::new(&actual_pixels, *w as usize, *h as usize);

        let report = check_regression(&zensim, &ref_img, &actual_img, tolerance)
            .map_err(|e| RegressError::Io {
                path: std::path::PathBuf::from("<simd-comparison>"),
                source: std::io::Error::other(e.to_string()),
            })?;

        if !report.passed() {
            all_passed = false;
        }
        comparisons.push(TierComparison {
            tier_label: label.clone(),
            report,
        });
    }

    Ok(SimdConsistencyReport {
        reference_tier: ref_label,
        permutation_count: outputs.len(),
        comparisons,
        all_passed,
        permutation_report: perm_report,
    })
}

/// Run `operation` under every SIMD token permutation (f32 variant).
///
/// Same as [`check_simd_consistency`] but for float pixel data.
/// Output f32 pixels are quantized to u8 RGBA for zensim comparison.
///
/// The operation returns `(Vec<f32>, u32, u32, usize)` — pixel data,
/// width, height, channels (1–4).
pub fn check_simd_consistency_f32(
    operation: impl Fn() -> (Vec<f32>, u32, u32, usize),
    tolerance: &RegressionTolerance,
    policy: CompileTimePolicy,
) -> Result<SimdConsistencyReport, RegressError> {
    check_simd_consistency(
        || {
            let (data, w, h, ch) = operation();
            let rgba = f32_to_rgba8(&data, w, h, ch);
            (rgba, w, h)
        },
        tolerance,
        policy,
    )
}

/// Convert f32 pixel data to RGBA u8, clamping to 0–255.
fn f32_to_rgba8(data: &[f32], w: u32, h: u32, channels: usize) -> Vec<u8> {
    let pixel_count = w as usize * h as usize;
    let mut rgba = Vec::with_capacity(pixel_count * 4);

    for i in 0..pixel_count {
        let offset = i * channels;
        let r = (data.get(offset).copied().unwrap_or(0.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8;
        let g = if channels > 1 {
            (data[offset + 1] * 255.0).round().clamp(0.0, 255.0) as u8
        } else {
            r
        };
        let b = if channels > 2 {
            (data[offset + 2] * 255.0).round().clamp(0.0, 255.0) as u8
        } else {
            r
        };
        let a = if channels > 3 {
            (data[offset + 3] * 255.0).round().clamp(0.0, 255.0) as u8
        } else {
            255
        };
        rgba.extend_from_slice(&[r, g, b, a]);
    }

    rgba
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators;

    #[test]
    fn single_permutation_passes() {
        // On most dev machines without testable_dispatch, all tokens are
        // compile-time guaranteed → single permutation.
        let input = generators::gradient(64, 64);

        let report = check_simd_consistency(
            || (input.clone(), 64, 64),
            &RegressionTolerance::exact(),
            CompileTimePolicy::Warn,
        )
        .unwrap();

        assert!(report.all_passed, "{report}");
        // May have 1 or more permutations depending on build flags
        assert!(report.permutation_count >= 1);
    }

    #[test]
    fn f32_variant_works() {
        let w = 32u32;
        let h = 32u32;
        let data: Vec<f32> = (0..w * h * 3)
            .map(|i| (i as f32) / (w * h * 3) as f32)
            .collect();

        let report = check_simd_consistency_f32(
            || (data.clone(), w, h, 3),
            &RegressionTolerance::exact(),
            CompileTimePolicy::Warn,
        )
        .unwrap();

        assert!(report.all_passed, "{report}");
    }

    #[test]
    fn report_display_format() {
        let report = SimdConsistencyReport {
            reference_tier: "all enabled".to_string(),
            permutation_count: 3,
            comparisons: Vec::new(),
            all_passed: true,
            permutation_report: PermutationReport {
                warnings: Vec::new(),
                permutations_run: 3,
            },
        };
        let s = format!("{report}");
        assert!(s.contains("PASS"));
        assert!(s.contains("3 permutations"));
    }
}
