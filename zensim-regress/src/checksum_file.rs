//! TOML checksum file types and persistence.
//!
//! Each test gets a `.toml` file at `{checksum-dir}/{sanitized_name}.toml`
//! containing tolerance, checksum entries (active + retired), and metadata.
//!
//! # File format
//!
//! ```toml
//! name = "resize_bicubic_200x200"
//!
//! [tolerance]
//! max_channel_delta = 1
//! min_score = 95.0
//! max_alpha_delta = 0
//!
//! [[checksum]]
//! id = "sea:a1b2c3d4e5f6789a"
//! confidence = 10
//! commit = "1540445a"
//! arch = ["x86_64-avx2"]
//! reason = "initial baseline"
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::RegressError;

// ─── Top-level file ──────────────────────────────────────────────────────

/// A per-test checksum file. Serialized as TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestChecksumFile {
    /// Human-readable test name (e.g., `"resize_bicubic_200x200"`).
    pub name: String,

    /// Tolerance configuration for this test.
    #[serde(default)]
    pub tolerance: ToleranceSpec,

    /// Checksum entries (active + retired).
    /// Active entries have `confidence > 0`.
    #[serde(default)]
    pub checksum: Vec<ChecksumEntry>,

    /// Optional image metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<ImageInfo>,
}

impl TestChecksumFile {
    /// Create a new file with just a name (empty checksums, default tolerance).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tolerance: ToleranceSpec::default(),
            checksum: Vec::new(),
            info: None,
        }
    }

    /// All active checksum entries (confidence > 0).
    pub fn active_checksums(&self) -> impl Iterator<Item = &ChecksumEntry> {
        self.checksum.iter().filter(|e| e.confidence > 0)
    }

    /// The authoritative checksum: active entry with highest confidence.
    /// If tied, the last one added wins (later in the Vec).
    pub fn authoritative(&self) -> Option<&ChecksumEntry> {
        self.active_checksums().max_by_key(|e| e.confidence)
    }

    /// Find a checksum entry by its ID.
    pub fn find_by_id(&self, id: &str) -> Option<&ChecksumEntry> {
        self.checksum.iter().find(|e| e.id == id)
    }

    /// Find a mutable checksum entry by its ID.
    pub fn find_by_id_mut(&mut self, id: &str) -> Option<&mut ChecksumEntry> {
        self.checksum.iter_mut().find(|e| e.id == id)
    }

    /// Read a checksum file from a path.
    pub fn read_from(path: &Path) -> Result<Self, RegressError> {
        let content = std::fs::read_to_string(path).map_err(|e| RegressError::io(path, e))?;
        let file: Self = toml::from_str(&content).map_err(|e| RegressError::toml_parse(path, e))?;
        Ok(file)
    }

    /// Write the checksum file to a path.
    ///
    /// Creates parent directories if needed.
    pub fn write_to(&self, path: &Path) -> Result<(), RegressError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| RegressError::io(parent, e))?;
        }
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content).map_err(|e| RegressError::io(path, e))?;
        Ok(())
    }
}

// ─── Tolerance specification ─────────────────────────────────────────────

/// Tolerance configuration stored in the TOML file.
///
/// Maps to `crate::testing::RegressionTolerance` at runtime, with optional
/// per-architecture overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceSpec {
    /// Maximum per-channel delta (in 1/255 units). Default: 0.
    #[serde(default)]
    pub max_channel_delta: u8,

    /// Minimum acceptable zensim score. Default: 100.0.
    #[serde(default = "default_score")]
    pub min_score: f64,

    /// Maximum fraction of pixels where any channel differs. Default: 0.0.
    #[serde(default)]
    pub max_differing_pixel_fraction: f64,

    /// Minimum fraction of (pixel, channel) values that must be identical. Default: 1.0.
    #[serde(default = "default_one")]
    pub min_identical_channel_fraction: f64,

    /// Maximum alpha channel delta (in 1/255 units). Default: 0.
    #[serde(default)]
    pub max_alpha_delta: u8,

    /// Whether to ignore alpha channel entirely. Default: false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub ignore_alpha: bool,

    /// Per-architecture tolerance overrides.
    /// Key is an architecture tag (e.g., `"aarch64"`, `"x86_64-avx2"`).
    #[serde(
        default,
        rename = "override",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub overrides: BTreeMap<String, ToleranceOverride>,
}

fn default_score() -> f64 {
    100.0
}
fn default_one() -> f64 {
    1.0
}
fn is_false(v: &bool) -> bool {
    !*v
}

impl Default for ToleranceSpec {
    fn default() -> Self {
        Self {
            max_channel_delta: 0,
            min_score: 100.0,
            max_differing_pixel_fraction: 0.0,
            min_identical_channel_fraction: 1.0,
            max_alpha_delta: 0,
            ignore_alpha: false,
            overrides: BTreeMap::new(),
        }
    }
}

impl ToleranceSpec {
    /// Convert to a `RegressionTolerance`, applying overrides for the given arch tag.
    pub fn to_regression_tolerance(&self, arch_tag: &str) -> crate::testing::RegressionTolerance {
        let mut t = crate::testing::RegressionTolerance::exact()
            .max_channel_delta(self.max_channel_delta)
            .min_score(self.min_score)
            .max_differing_pixel_fraction(self.max_differing_pixel_fraction)
            .min_identical_channel_fraction(self.min_identical_channel_fraction)
            .max_alpha_delta(self.max_alpha_delta);

        if self.ignore_alpha {
            t = t.ignore_alpha();
        }

        // Apply the most specific matching override
        if let Some(ov) = self.best_override_for(arch_tag) {
            if let Some(v) = ov.max_channel_delta {
                t = t.max_channel_delta(v);
            }
            if let Some(v) = ov.min_score {
                t = t.min_score(v);
            }
            if let Some(v) = ov.max_differing_pixel_fraction {
                t = t.max_differing_pixel_fraction(v);
            }
            if let Some(v) = ov.min_identical_channel_fraction {
                t = t.min_identical_channel_fraction(v);
            }
            if let Some(v) = ov.max_alpha_delta {
                t = t.max_alpha_delta(v);
            }
        }

        t
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

/// Per-architecture tolerance override.
///
/// Only fields that are `Some` override the base tolerance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToleranceOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_channel_delta: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_differing_pixel_fraction: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_identical_channel_fraction: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_alpha_delta: Option<u8>,
}

// ─── Checksum entry ──────────────────────────────────────────────────────

/// A single checksum entry in the file.
///
/// `confidence > 0` means the entry is active (matching this hash passes).
/// `confidence = 0` means retired (kept for forensics and bisecting).
/// The entry with the highest confidence is the authoritative reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumEntry {
    /// Hash ID (e.g., `"sea:a1b2c3d4e5f6789a"`).
    pub id: String,

    /// Confidence level. 0 = retired, >0 = active.
    /// Higher = more authoritative (used as the reference for diffs).
    #[serde(default = "default_confidence")]
    pub confidence: u32,

    /// Git commit hash where this checksum was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,

    /// Architecture tags that produce this exact output.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arch: Vec<String>,

    /// Human-readable reason for this entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,

    /// Status annotation (e.g., `"wrong"` for known-bad entries).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,

    /// Diff evidence against the authoritative reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<ChecksumDiff>,
}

fn default_confidence() -> u32 {
    10
}

impl ChecksumEntry {
    /// Create a new active entry with default confidence.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            confidence: 10,
            commit: None,
            arch: Vec::new(),
            reason: None,
            status: None,
            diff: None,
        }
    }

    /// Whether this entry is active (confidence > 0, no "wrong" status).
    pub fn is_active(&self) -> bool {
        self.confidence > 0 && self.status.as_deref() != Some("wrong")
    }
}

// ─── Checksum diff (chain-of-trust evidence) ────────────────────────────

/// Evidence comparing this checksum against the authoritative reference.
///
/// Populated from a `RegressionReport` — records what zensim found when
/// comparing this output vs the reference. This is the chain-of-trust:
/// instead of blindly accepting alternate checksums, we record exactly
/// how they differ and whether the difference is acceptable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumDiff {
    /// Hash ID of the reference this was compared against.
    pub vs: String,

    /// Zensim similarity score (0–100).
    pub zensim_score: f64,

    /// Error category name (e.g., `"RoundingError"`, `"Unclassified"`).
    pub category: String,

    /// Max per-channel delta in 1/255 units `[R, G, B]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_channel_delta: Option<[u8; 3]>,

    /// Percentage of pixels that differ.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixels_differing_pct: Option<f64>,

    /// Whether the rounding bias is balanced (only for RoundingError).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rounding_bias_balanced: Option<bool>,
}

impl ChecksumDiff {
    /// Create from a `RegressionReport` and the reference checksum ID.
    pub fn from_report(report: &crate::testing::RegressionReport, vs_id: &str) -> Self {
        let category = format!("{:?}", report.category());

        let pixels_differing_pct = if report.pixel_count() > 0 {
            Some(report.pixels_differing() as f64 / report.pixel_count() as f64 * 100.0)
        } else {
            None
        };

        Self {
            vs: vs_id.to_string(),
            zensim_score: report.score(),
            category,
            max_channel_delta: Some(report.max_channel_delta()),
            pixels_differing_pct,
            rounding_bias_balanced: report.rounding_bias().map(|b| b.balanced),
        }
    }
}

// ─── Image info metadata ─────────────────────────────────────────────────

/// Optional image metadata stored alongside checksums.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

// ─── Name sanitization ──────────────────────────────────────────────────

/// Sanitize a test name into a safe filename stem.
///
/// Rules:
/// - Lowercase
/// - Alphanumeric, `-`, `_` are kept
/// - Spaces, `/`, `\`, `::` become `_`
/// - Other characters are dropped
/// - Leading/trailing `_` and `-` are stripped
/// - Consecutive `_` collapsed to single `_`
/// - Empty result becomes `_unnamed`
pub fn sanitize_name(name: &str) -> String {
    let mut result = String::with_capacity(name.len());

    for ch in name.chars() {
        match ch {
            'A'..='Z' => result.push(ch.to_ascii_lowercase()),
            'a'..='z' | '0'..='9' | '-' => result.push(ch),
            ' ' | '/' | '\\' | ':' => result.push('_'),
            '_' => result.push('_'),
            _ => {} // drop other characters
        }
    }

    // Collapse consecutive underscores
    let mut collapsed = String::with_capacity(result.len());
    let mut prev_underscore = false;
    for ch in result.chars() {
        if ch == '_' {
            if !prev_underscore {
                collapsed.push('_');
            }
            prev_underscore = true;
        } else {
            collapsed.push(ch);
            prev_underscore = false;
        }
    }

    // Strip leading/trailing separators
    let trimmed = collapsed.trim_matches(|c: char| c == '_' || c == '-');

    if trimmed.is_empty() {
        "_unnamed".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Get the TOML file path for a test name within a checksum directory.
pub fn checksum_path(dir: &Path, test_name: &str) -> std::path::PathBuf {
    dir.join(format!("{}.toml", sanitize_name(test_name)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Name sanitization ──────────────────────────────────────────

    #[test]
    fn sanitize_basic() {
        assert_eq!(
            sanitize_name("resize_bicubic_200x200"),
            "resize_bicubic_200x200"
        );
    }

    #[test]
    fn sanitize_spaces() {
        assert_eq!(sanitize_name("Resize Bicubic 200"), "resize_bicubic_200");
    }

    #[test]
    fn sanitize_path_separators() {
        assert_eq!(sanitize_name("tests/visual/foo"), "tests_visual_foo");
    }

    #[test]
    fn sanitize_rust_paths() {
        assert_eq!(sanitize_name("module::test::case"), "module_test_case");
    }

    #[test]
    fn sanitize_special_chars() {
        assert_eq!(sanitize_name("test@#$%name"), "testname");
    }

    #[test]
    fn sanitize_leading_trailing() {
        assert_eq!(sanitize_name("__test__"), "test");
        assert_eq!(sanitize_name("--test--"), "test");
        assert_eq!(sanitize_name("_-test-_"), "test");
    }

    #[test]
    fn sanitize_consecutive_underscores() {
        assert_eq!(sanitize_name("test___name"), "test_name");
        assert_eq!(sanitize_name("a::b::c"), "a_b_c");
    }

    #[test]
    fn sanitize_empty() {
        assert_eq!(sanitize_name(""), "_unnamed");
        assert_eq!(sanitize_name("@#$"), "_unnamed");
    }

    #[test]
    fn sanitize_mixed_case() {
        assert_eq!(sanitize_name("ResizeBicubic"), "resizebicubic");
    }

    // ─── TOML round-trip ─────────────────────────────────────────────

    #[test]
    fn roundtrip_minimal() {
        let file = TestChecksumFile::new("test_case");
        let toml_str = toml::to_string_pretty(&file).unwrap();
        let parsed: TestChecksumFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.name, "test_case");
        assert!(parsed.checksum.is_empty());
    }

    #[test]
    fn roundtrip_full() {
        let file = TestChecksumFile {
            name: "resize_bicubic_200x200".to_string(),
            tolerance: ToleranceSpec {
                max_channel_delta: 1,
                min_score: 95.0,
                max_alpha_delta: 0,
                overrides: BTreeMap::from([(
                    "aarch64".to_string(),
                    ToleranceOverride {
                        max_channel_delta: Some(2),
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            },
            checksum: vec![
                ChecksumEntry {
                    id: "sea:a1b2c3d4e5f6789a".to_string(),
                    confidence: 10,
                    commit: Some("1540445a".to_string()),
                    arch: vec!["x86_64-avx2".to_string()],
                    reason: Some("initial baseline".to_string()),
                    status: None,
                    diff: None,
                },
                ChecksumEntry {
                    id: "sea:b2c3d4e5f6789abc".to_string(),
                    confidence: 10,
                    commit: Some("1540445a".to_string()),
                    arch: vec!["aarch64".to_string()],
                    reason: Some("ARM NEON rounding".to_string()),
                    status: None,
                    diff: Some(ChecksumDiff {
                        vs: "sea:a1b2c3d4e5f6789a".to_string(),
                        zensim_score: 99.9,
                        category: "RoundingError".to_string(),
                        max_channel_delta: Some([1, 1, 1]),
                        pixels_differing_pct: Some(0.1),
                        rounding_bias_balanced: Some(true),
                    }),
                },
                ChecksumEntry {
                    id: "sea:0000deadbeef1234".to_string(),
                    confidence: 0,
                    commit: None,
                    arch: Vec::new(),
                    reason: Some("pre-CICP fix".to_string()),
                    status: Some("wrong".to_string()),
                    diff: Some(ChecksumDiff {
                        vs: "sea:a1b2c3d4e5f6789a".to_string(),
                        zensim_score: 12.3,
                        category: "Unclassified".to_string(),
                        max_channel_delta: Some([224, 198, 210]),
                        pixels_differing_pct: Some(95.2),
                        rounding_bias_balanced: None,
                    }),
                },
            ],
            info: Some(ImageInfo {
                width: Some(200),
                height: Some(200),
                format: None,
            }),
        };

        let toml_str = toml::to_string_pretty(&file).unwrap();
        let parsed: TestChecksumFile = toml::from_str(&toml_str).unwrap();

        assert_eq!(parsed.name, file.name);
        assert_eq!(parsed.tolerance.max_channel_delta, 1);
        assert_eq!(parsed.tolerance.min_score, 95.0);
        assert_eq!(parsed.checksum.len(), 3);

        // Active checksums
        let active: Vec<_> = parsed.active_checksums().collect();
        assert_eq!(active.len(), 2);

        // Authoritative
        let auth = parsed.authoritative().unwrap();
        assert_eq!(auth.confidence, 10);

        // Diff evidence
        let second = &parsed.checksum[1];
        let diff = second.diff.as_ref().unwrap();
        assert_eq!(diff.vs, "sea:a1b2c3d4e5f6789a");
        assert_eq!(diff.category, "RoundingError");
        assert_eq!(diff.rounding_bias_balanced, Some(true));

        // Retired entry
        let retired = &parsed.checksum[2];
        assert_eq!(retired.confidence, 0);
        assert_eq!(retired.status.as_deref(), Some("wrong"));
        assert!(!retired.is_active());

        // Override
        assert!(parsed.tolerance.overrides.contains_key("aarch64"));
        let ov = &parsed.tolerance.overrides["aarch64"];
        assert_eq!(ov.max_channel_delta, Some(2));

        // Image info
        let info = parsed.info.as_ref().unwrap();
        assert_eq!(info.width, Some(200));
        assert_eq!(info.height, Some(200));
    }

    #[test]
    fn roundtrip_file_io() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");

        let file = TestChecksumFile {
            name: "io_test".to_string(),
            tolerance: ToleranceSpec::default(),
            checksum: vec![ChecksumEntry::new("sea:1234567890abcdef")],
            info: None,
        };

        file.write_to(&path).unwrap();
        let parsed = TestChecksumFile::read_from(&path).unwrap();

        assert_eq!(parsed.name, "io_test");
        assert_eq!(parsed.checksum.len(), 1);
        assert_eq!(parsed.checksum[0].id, "sea:1234567890abcdef");
    }

    #[test]
    fn tolerance_to_regression_tolerance() {
        let spec = ToleranceSpec {
            max_channel_delta: 2,
            min_score: 90.0,
            ..Default::default()
        };
        // Basic conversion works (we can't inspect private fields, but we
        // verify it doesn't panic)
        let _t = spec.to_regression_tolerance("x86_64-avx2");
    }

    #[test]
    fn tolerance_override_applied() {
        let spec = ToleranceSpec {
            max_channel_delta: 1,
            min_score: 95.0,
            overrides: BTreeMap::from([(
                "aarch64".to_string(),
                ToleranceOverride {
                    max_channel_delta: Some(3),
                    min_score: Some(90.0),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        };

        // For x86_64, no override should apply
        let _t_x86 = spec.to_regression_tolerance("x86_64-avx2");
        // For aarch64, override should apply (no panic)
        let _t_arm = spec.to_regression_tolerance("aarch64");
    }

    #[test]
    fn checksum_path_basic() {
        let dir = std::path::Path::new("/tmp/checksums");
        let path = checksum_path(dir, "resize::bicubic::200");
        assert_eq!(
            path,
            std::path::PathBuf::from("/tmp/checksums/resize_bicubic_200.toml")
        );
    }

    #[test]
    fn toml_output_readable() {
        // Verify the TOML output looks reasonable
        let file = TestChecksumFile {
            name: "example".to_string(),
            tolerance: ToleranceSpec {
                max_channel_delta: 1,
                min_score: 95.0,
                ..Default::default()
            },
            checksum: vec![ChecksumEntry {
                id: "sea:a1b2c3d4e5f6789a".to_string(),
                confidence: 10,
                commit: Some("abc123".to_string()),
                arch: vec!["x86_64-avx2".to_string()],
                reason: Some("initial baseline".to_string()),
                status: None,
                diff: None,
            }],
            info: None,
        };

        let toml_str = toml::to_string_pretty(&file).unwrap();
        // Should contain expected key sections
        assert!(toml_str.contains("name = \"example\""), "missing name");
        assert!(toml_str.contains("[tolerance]"), "missing [tolerance]");
        assert!(toml_str.contains("[[checksum]]"), "missing [[checksum]]");
        assert!(toml_str.contains("sea:a1b2c3d4e5f6789a"), "missing hash");
    }
}
