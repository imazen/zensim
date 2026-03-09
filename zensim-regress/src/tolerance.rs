//! Tolerance specification types for visual regression testing.
//!
//! `ToleranceSpec` defines the acceptable pixel-level and perceptual
//! differences between actual output and a baseline reference.
//! `ToleranceOverride` allows per-architecture tolerance adjustments.

use std::collections::BTreeMap;

// ─── Tolerance specification ─────────────────────────────────────────────

/// Tolerance configuration for visual regression tests.
///
/// Maps to [`crate::testing::RegressionTolerance`] at runtime, with optional
/// per-architecture overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct ToleranceSpec {
    /// Maximum per-channel delta (in 1/255 units). Default: 0.
    pub max_delta: u8,

    /// Minimum acceptable zensim score. Default: 100.0.
    pub min_similarity: f64,

    /// Maximum fraction of pixels where any channel differs. Default: 0.0.
    pub max_pixels_different: f64,

    /// Maximum alpha channel delta (in 1/255 units). Default: 0.
    pub max_alpha_delta: u8,

    /// Whether to ignore alpha channel entirely. Default: false.
    pub ignore_alpha: bool,

    /// Per-architecture tolerance overrides.
    /// Key is an architecture tag (e.g., `"aarch64"`, `"x86_64-avx2"`).
    pub overrides: BTreeMap<String, ToleranceOverride>,
}

impl ToleranceSpec {
    /// Pixel-identical: no differences allowed (d:0 s:100).
    pub fn exact() -> Self {
        Self {
            max_delta: 0,
            min_similarity: 100.0,
            max_pixels_different: 0.0,
            max_alpha_delta: 0,
            ignore_alpha: false,
            overrides: BTreeMap::new(),
        }
    }

    /// Off-by-one rounding: max delta 1/255, any fraction of pixels, score >= 85.
    ///
    /// Use struct update syntax to customize further fields.
    pub fn off_by_one() -> Self {
        Self {
            max_delta: 1,
            min_similarity: 85.0,
            max_pixels_different: 1.0,
            ..Self::exact()
        }
    }

    /// Convert to a `RegressionTolerance`, applying overrides for the given arch tag.
    pub fn to_regression_tolerance(&self, arch_tag: &str) -> crate::testing::RegressionTolerance {
        let mut t = crate::testing::RegressionTolerance::exact()
            .with_max_delta(self.max_delta)
            .with_min_similarity(self.min_similarity)
            .with_max_pixels_different(self.max_pixels_different)
            .with_max_alpha_delta(self.max_alpha_delta);

        if self.ignore_alpha {
            t = t.ignore_alpha();
        }

        // Apply the most specific matching override
        if let Some(ov) = self.best_override_for(arch_tag) {
            if let Some(v) = ov.max_delta {
                t = t.with_max_delta(v);
            }
            if let Some(v) = ov.min_similarity {
                t = t.with_min_similarity(v);
            }
            if let Some(v) = ov.max_pixels_different {
                t = t.with_max_pixels_different(v);
            }
            if let Some(v) = ov.max_alpha_delta {
                t = t.with_max_alpha_delta(v);
            }
        }

        t
    }

    /// Create a `ToleranceSpec` from a `RegressionTolerance`.
    pub fn from_tolerance(t: &crate::testing::RegressionTolerance) -> Self {
        Self {
            max_delta: t.max_delta(),
            min_similarity: t.min_similarity(),
            max_pixels_different: t.max_pixels_different(),
            max_alpha_delta: t.max_alpha_delta(),
            ignore_alpha: t.is_ignore_alpha(),
            overrides: BTreeMap::new(),
        }
    }

    /// Find the best (most specific) override for a given arch tag.
    ///
    /// Exact match beats prefix match. Among prefix matches, longer wins.
    fn best_override_for(&self, arch_tag: &str) -> Option<&ToleranceOverride> {
        // Exact match first
        if let Some(ov) = self.overrides.get(arch_tag) {
            return Some(ov);
        }

        // Prefix match: find the longest matching key
        self.overrides
            .iter()
            .filter(|(key, _)| crate::arch::arch_matches(key, arch_tag))
            .max_by_key(|(key, _)| key.len())
            .map(|(_, ov)| ov)
    }
}

impl Default for ToleranceSpec {
    fn default() -> Self {
        Self::exact()
    }
}

// ─── Per-architecture override ───────────────────────────────────────────

/// Per-architecture tolerance override.
///
/// Only fields that are `Some` override the base tolerance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToleranceOverride {
    /// Override maximum per-channel delta (1/255 units).
    pub max_delta: Option<u8>,
    /// Override minimum acceptable zensim score.
    pub min_similarity: Option<f64>,
    /// Override maximum fraction of differing pixels.
    pub max_pixels_different: Option<f64>,
    /// Override maximum alpha channel delta (1/255 units).
    pub max_alpha_delta: Option<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_to_regression_tolerance() {
        let spec = ToleranceSpec {
            max_delta: 2,
            min_similarity: 90.0,
            ..ToleranceSpec::exact()
        };
        let _t = spec.to_regression_tolerance("x86_64-avx2");
    }

    #[test]
    fn tolerance_override_applied() {
        let spec = ToleranceSpec {
            max_delta: 1,
            min_similarity: 95.0,
            overrides: BTreeMap::from([(
                "aarch64".to_string(),
                ToleranceOverride {
                    max_delta: Some(3),
                    min_similarity: Some(90.0),
                    ..Default::default()
                },
            )]),
            ..ToleranceSpec::exact()
        };

        // For x86_64, no override should apply
        let _t_x86 = spec.to_regression_tolerance("x86_64-avx2");
        // For aarch64, override should apply (no panic)
        let _t_arm = spec.to_regression_tolerance("aarch64");
    }
}
