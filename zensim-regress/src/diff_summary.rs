//! Human-readable diff summary for `.checksums` entries.
//!
//! Produces the parenthesized diff string that appears after `vs` in auto-accepted entries:
//!
//! ```text
//! (zensim:0.0013, 2.1% pixels ±1, max-delta:[1,1,0], category:rounding, balanced)
//! ```
//!
//! The `zensim:` value is a dissimilarity number where 0 = identical and higher = worse.
//! Simple linear conversion from the zensim score: `(100 - score) / 100`.

use crate::testing::RegressionReport;

// ─── zdsim metric ────────────────────────────────────────────────────────

/// Convert a zensim score (0–100, 100 = identical) to a zdsim dissimilarity
/// value (0 = identical, higher = worse).
///
/// **zdsim-0.1**: linear conversion `(100 - score) / 100`.
///
/// | zensim | zdsim  |
/// |--------|--------|
/// | 100.0  | 0.0    |
/// | 99.5   | 0.005  |
/// | 95.0   | 0.05   |
/// | 50.0   | 0.5    |
/// | 0.0    | 1.0    |
pub fn zdsim(zensim_score: f64) -> f64 {
    ((100.0 - zensim_score) / 100.0).max(0.0)
}

// ─── Diff summary formatting ─────────────────────────────────────────────

/// Format a human-readable diff summary from a [`RegressionReport`].
///
/// Output is a parenthesized string suitable for `.checksums` `vs` clauses:
/// ```text
/// (zensim:0.0013, 2.1% pixels ±1, max-delta:[1,1,0], category:rounding, balanced)
/// ```
///
/// Only includes fields when they carry information (omits trivially obvious values).
pub fn format_diff_summary(report: &RegressionReport) -> String {
    let mut parts = Vec::new();

    // Zensim dissimilarity (always present, 0 = identical, higher = worse)
    let zd = zdsim(report.score());
    if zd == 0.0 {
        parts.push("zensim:0".to_string());
    } else if zd < 0.0001 {
        parts.push(format!("zensim:{zd:.6}"));
    } else if zd < 0.01 {
        parts.push(format!("zensim:{zd:.4}"));
    } else {
        parts.push(format!("zensim:{zd:.3}"));
    }

    // Pixels differing by ±N tiers (from RegressionReport pixel stats)
    let pixel_count = report.pixel_count();
    let native_max = report.native_max();
    if pixel_count > 0 {
        let differing = report.pixels_differing();
        if differing > 0 {
            let pct = differing as f64 / pixel_count as f64 * 100.0;
            let pct_str = if pct < 0.01 {
                format!("{pct:.3}%")
            } else if pct < 1.0 {
                format!("{pct:.2}%")
            } else {
                format!("{pct:.1}%")
            };
            if native_max != 255.0 {
                parts.push(format!("{pct_str} pixels \u{00b1}1 LSB"));
            } else {
                let max_delta = *report.max_channel_delta().iter().max().unwrap_or(&0);
                parts.push(format!("{pct_str} pixels \u{00b1}{max_delta}"));
            }
        }
    }

    // Max channel deltas [R,G,B] (only if non-zero)
    if native_max != 255.0 {
        // Show native-precision deltas
        let mcd = report.max_channel_delta_f64();
        let any_nonzero = mcd.iter().any(|&d| d > 0.0);
        if any_nonzero {
            let nm = native_max;
            let r = (mcd[0] * nm).round() as u64;
            let g = (mcd[1] * nm).round() as u64;
            let b = (mcd[2] * nm).round() as u64;
            parts.push(format!("max-delta:[{r},{g},{b}]/{}", nm as u64));
        }
    } else {
        let mcd = report.max_channel_delta();
        if mcd != [0, 0, 0] {
            parts.push(format!("max-delta:[{},{},{}]", mcd[0], mcd[1], mcd[2]));
        }
    }

    // Error category (always present when not Identical)
    let cat = report.category();
    let cat_str = match cat {
        zensim::ErrorCategory::Identical => None,
        zensim::ErrorCategory::RoundingError => Some("rounding"),
        zensim::ErrorCategory::ChannelSwap => Some("channel-swap"),
        zensim::ErrorCategory::AlphaCompositing => Some("alpha"),
        _ => Some("unclassified"),
    };
    if let Some(c) = cat_str {
        parts.push(format!("category:{c}"));
    }

    // Rounding bias (only for RoundingError)
    if let Some(bias) = report.rounding_bias() {
        parts.push(if bias.balanced {
            "balanced".to_string()
        } else {
            "biased".to_string()
        });
    }

    format!("({})", parts.join(", "))
}

/// Format the tolerance that was active when an entry was accepted.
///
/// Output: `"within identical"`, `"within off-by-one"`, `"within zensim:0.01"`,
/// or `"within max-delta:1 zensim:0.005 pixels-changed:1.0%"`.
pub fn format_tolerance_note(tolerance: &crate::testing::RegressionTolerance) -> String {
    use crate::testing::RegressionTolerance;

    let exact = RegressionTolerance::exact();
    let obo = RegressionTolerance::off_by_one();

    let is_exact = tolerance.min_similarity() == exact.min_similarity()
        && tolerance.max_pixels_different() == exact.max_pixels_different()
        && tolerance.max_alpha_delta() == exact.max_alpha_delta()
        && !tolerance.is_ignore_alpha();

    let is_obo = tolerance.max_delta() == obo.max_delta()
        && tolerance.min_similarity() == obo.min_similarity()
        && tolerance.max_pixels_different() == obo.max_pixels_different()
        && tolerance.max_alpha_delta() == obo.max_alpha_delta()
        && !tolerance.is_ignore_alpha();

    // Perceptual-only: max-delta:255 pixels-changed:>=100% means only zensim matters
    let is_perceptual_only = tolerance.max_delta() == 255
        && tolerance.max_pixels_different() >= 1.0
        && tolerance.max_alpha_delta() == 0
        && !tolerance.is_ignore_alpha()
        && tolerance.min_similarity() < 100.0;

    if is_exact {
        return "within identical".to_string();
    }
    if is_obo {
        return "within off-by-one".to_string();
    }
    if is_perceptual_only {
        let zd = zdsim(tolerance.min_similarity());
        return format!("within {}", format_zensim_token(zd));
    }

    let mut parts = Vec::new();
    parts.push(format!("max-delta:{}", tolerance.max_delta()));
    if tolerance.min_similarity() < 100.0 {
        let zd = zdsim(tolerance.min_similarity());
        parts.push(format_zensim_token(zd));
    }
    if tolerance.max_pixels_different() > 0.0 {
        let px = tolerance.max_pixels_different() * 100.0;
        parts.push(format!("pixels-changed:{px:.1}%"));
    }
    format!("within {}", parts.join(" "))
}

// ─── Tolerance shorthand ─────────────────────────────────────────────────

/// Format a tolerance as the shorthand used in `.checksums` files.
///
/// Recognizes named presets for readability:
/// - `identical` — pixel-identical
/// - `off-by-one` — rounding tolerance (max-delta:1 zensim:0.05 pixels-changed:100%)
///
/// Perceptual-only tolerances use standalone `zensim:X`:
/// ```text
/// zensim:0.01
/// zensim:0.05 [aarch64 zensim:0.1]
/// ```
///
/// Per-pixel tolerances use compound tokens:
/// ```text
/// max-delta:2 zensim:0.05 pixels-changed:1.0%
/// max-delta:1 zensim:0.01 alpha-delta:0 [aarch64 max-delta:3 zensim:0.1]
/// ```
pub fn format_tolerance_shorthand(tolerance: &crate::checksum_file::ToleranceSpec) -> String {
    use crate::checksum_file::ToleranceSpec;

    // Check for named presets (base fields only; overrides appended after).
    //
    // "identical" matches any spec where zensim:0 + pixels-changed:0 — the
    // max-delta value is irrelevant because a perfect score already implies zero pixel deltas.
    let base_matches_exact = tolerance.min_similarity == 100.0
        && tolerance.max_pixels_different == 0.0
        && tolerance.max_alpha_delta == 0
        && !tolerance.ignore_alpha;

    let base_matches_obo = {
        let obo = ToleranceSpec::off_by_one();
        tolerance.max_delta == obo.max_delta
            && tolerance.min_similarity == obo.min_similarity
            && tolerance.max_pixels_different == obo.max_pixels_different
            && tolerance.max_alpha_delta == obo.max_alpha_delta
            && !tolerance.ignore_alpha
    };

    // Perceptual-only: max-delta:255 pixels-changed:100% means only zensim matters.
    let is_perceptual_only = tolerance.max_delta == 255
        && tolerance.max_pixels_different >= 1.0
        && tolerance.max_alpha_delta == 0
        && !tolerance.ignore_alpha
        && tolerance.min_similarity < 100.0;

    let mut parts = Vec::new();

    if base_matches_exact {
        parts.push("identical".to_string());
    } else if base_matches_obo {
        parts.push("off-by-one".to_string());
    } else if is_perceptual_only {
        let zd = zdsim(tolerance.min_similarity);
        parts.push(format_zensim_token(zd));
    } else {
        parts.push(format!("max-delta:{}", tolerance.max_delta));

        if tolerance.min_similarity < 100.0 {
            let zd = zdsim(tolerance.min_similarity);
            parts.push(format_zensim_token(zd));
        }

        if tolerance.max_pixels_different > 0.0 {
            let px = tolerance.max_pixels_different * 100.0;
            parts.push(format!("pixels-changed:{px:.1}%"));
        }

        if tolerance.max_alpha_delta > 0 {
            parts.push(format!("alpha-delta:{}", tolerance.max_alpha_delta));
        }

        if tolerance.ignore_alpha {
            parts.push("ignore-alpha".to_string());
        }
    }

    // Per-arch overrides
    for (arch, ov) in &tolerance.overrides {
        let mut ov_parts = Vec::new();
        if let Some(d) = ov.max_delta {
            ov_parts.push(format!("max-delta:{d}"));
        }
        if let Some(s) = ov.min_similarity {
            ov_parts.push(format_zensim_token(zdsim(s)));
        }
        if let Some(px) = ov.max_pixels_different {
            ov_parts.push(format!("pixels-changed:{:.1}%", px * 100.0));
        }
        if let Some(a) = ov.max_alpha_delta {
            ov_parts.push(format!("alpha-delta:{a}"));
        }
        if !ov_parts.is_empty() {
            parts.push(format!("[{} {}]", arch, ov_parts.join(" ")));
        }
    }

    parts.join(" ")
}

/// Format a zensim dissimilarity value as a shorthand token with appropriate precision.
///
/// The `zensim:` prefix is used everywhere — standalone perceptual tolerance,
/// compound tolerance, diff summaries, and recommendations.
fn format_zensim_token(zd: f64) -> String {
    if zd == 0.0 {
        "zensim:0".to_string()
    } else if zd < 0.001 {
        format!("zensim:{zd:.4}")
    } else if zd < 0.01 {
        format!("zensim:{zd:.3}")
    } else if zd == (zd * 100.0).round() / 100.0 {
        // Clean two-decimal value like 0.01, 0.05
        format!("zensim:{zd:.2}")
    } else {
        format!("zensim:{zd:.4}")
    }
}

/// Parse a tolerance shorthand string back into a [`ToleranceSpec`].
///
/// Handles the format produced by [`format_tolerance_shorthand`], including
/// named presets:
/// - `"identical"` → `ToleranceSpec::exact()`
/// - `"off-by-one"` → `ToleranceSpec::off_by_one()`
///
/// Named presets can have per-arch overrides appended:
/// `"off-by-one [aarch64 d:3]"`
pub fn parse_tolerance_shorthand(s: &str) -> crate::checksum_file::ToleranceSpec {
    use crate::checksum_file::{ToleranceOverride, ToleranceSpec};
    use std::collections::BTreeMap;

    let input = s.trim();

    // Check for named presets (before any bracket)
    let before_bracket = input.split('[').next().unwrap_or(input).trim();
    let mut spec = match before_bracket {
        "identical" => ToleranceSpec::exact(),
        "off-by-one" => ToleranceSpec::off_by_one(),
        _ => {
            let mut spec = ToleranceSpec::exact();
            parse_main_tolerance_tokens(before_bracket, &mut spec);
            spec
        }
    };

    // Extract bracketed per-arch overrides
    let mut overrides: BTreeMap<String, ToleranceOverride> = BTreeMap::new();
    let mut pos = 0;
    while pos < input.len() {
        if let Some(bracket_start) = input[pos..].find('[') {
            let abs_start = pos + bracket_start;
            let bracket_end = input[abs_start..]
                .find(']')
                .map(|i| abs_start + i)
                .unwrap_or(input.len());

            let bracket_content = &input[abs_start + 1..bracket_end];
            let mut tokens = bracket_content.split_whitespace();
            if let Some(arch) = tokens.next() {
                let mut ov = ToleranceOverride::default();
                for token in tokens {
                    parse_tolerance_token(token, &mut ov);
                }
                overrides.insert(arch.to_string(), ov);
            }

            pos = (bracket_end + 1).min(input.len());
        } else {
            break;
        }
    }

    spec.overrides = overrides;
    spec
}

/// Parse a dissimilarity value from a token with any of the supported prefixes.
///
/// Accepts: `zensim:X`, `dissimilarity:X`, `zdsim:X` (all mean the same thing).
fn strip_dissimilarity_prefix(token: &str) -> Option<&str> {
    token
        .strip_prefix("zensim:")
        .or_else(|| token.strip_prefix("dissimilarity:"))
        .or_else(|| token.strip_prefix("zdsim:"))
}

fn parse_main_tolerance_tokens(s: &str, spec: &mut crate::checksum_file::ToleranceSpec) {
    let mut has_explicit_delta = false;
    let mut has_explicit_pixels = false;

    for token in s.split_whitespace() {
        // Accept zensim:, dissimilarity:, zdsim: — all set min_similarity from dissimilarity
        if let Some(v) = strip_dissimilarity_prefix(token) {
            let zd: f64 = v.parse().unwrap_or(0.0);
            spec.min_similarity = (100.0 * (1.0 - zd)).max(0.0);
        } else if let Some(v) = token
            .strip_prefix("max-delta:")
            .or_else(|| token.strip_prefix("d:"))
        {
            spec.max_delta = v.parse().unwrap_or(0);
            has_explicit_delta = true;
        } else if let Some(v) = token
            .strip_prefix("similarity:")
            .or_else(|| token.strip_prefix("s:"))
        {
            // Legacy: score-based similarity (0-100) → convert internally
            spec.min_similarity = v.parse().unwrap_or(100.0);
        } else if let Some(v) = token
            .strip_prefix("pixels-changed:")
            .or_else(|| token.strip_prefix("px:"))
        {
            let v = v.trim_end_matches('%');
            spec.max_pixels_different = v.parse::<f64>().unwrap_or(0.0) / 100.0;
            has_explicit_pixels = true;
        } else if let Some(v) = token
            .strip_prefix("alpha-delta:")
            .or_else(|| token.strip_prefix("a:"))
        {
            spec.max_alpha_delta = v.parse().unwrap_or(0);
        } else if token == "ignore-alpha" || token == "ia" {
            spec.ignore_alpha = true;
        }
    }

    // If zensim: was the only constraint (no explicit delta or pixel tokens),
    // default to perceptual-only mode: allow any pixel-level differences.
    if spec.min_similarity < 100.0 && !has_explicit_delta && !has_explicit_pixels {
        spec.max_delta = 255;
        spec.max_pixels_different = 1.0;
    }
}

fn parse_tolerance_token(token: &str, ov: &mut crate::checksum_file::ToleranceOverride) {
    if let Some(v) = strip_dissimilarity_prefix(token) {
        let zd: f64 = v.parse().unwrap_or(0.0);
        ov.min_similarity = Some((100.0 * (1.0 - zd)).max(0.0));
    } else if let Some(v) = token
        .strip_prefix("max-delta:")
        .or_else(|| token.strip_prefix("d:"))
    {
        ov.max_delta = v.parse().ok();
    } else if let Some(v) = token
        .strip_prefix("similarity:")
        .or_else(|| token.strip_prefix("s:"))
    {
        // Legacy: score-based similarity (0-100)
        ov.min_similarity = v.parse().ok();
    } else if let Some(v) = token
        .strip_prefix("pixels-changed:")
        .or_else(|| token.strip_prefix("px:"))
    {
        let v = v.trim_end_matches('%');
        ov.max_pixels_different = v.parse::<f64>().ok().map(|p| p / 100.0);
    } else if let Some(v) = token
        .strip_prefix("alpha-delta:")
        .or_else(|| token.strip_prefix("a:"))
    {
        ov.max_alpha_delta = v.parse().ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zdsim_identical() {
        assert_eq!(zdsim(100.0), 0.0);
    }

    #[test]
    fn zdsim_half() {
        assert!((zdsim(50.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn zdsim_zero() {
        assert!((zdsim(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn zdsim_typical_regression() {
        // Score 99.87 → zdsim 0.0013
        let zd = zdsim(99.87);
        assert!((zd - 0.0013).abs() < 1e-6, "zd={zd}");
    }

    #[test]
    fn zdsim_never_negative() {
        // Score slightly above 100 (shouldn't happen but be safe)
        assert_eq!(zdsim(100.5), 0.0);
    }

    #[test]
    fn tolerance_shorthand_roundtrip_exact() {
        let spec = crate::checksum_file::ToleranceSpec::exact();
        let s = format_tolerance_shorthand(&spec);
        assert_eq!(s, "identical");
        let parsed = parse_tolerance_shorthand(&s);
        assert_eq!(parsed.max_delta, 0);
        assert_eq!(parsed.min_similarity, 100.0);
        assert_eq!(parsed.max_pixels_different, 0.0);
    }

    #[test]
    fn tolerance_shorthand_roundtrip_off_by_one() {
        let spec = crate::checksum_file::ToleranceSpec::off_by_one();
        let s = format_tolerance_shorthand(&spec);
        assert_eq!(s, "off-by-one");
        let parsed = parse_tolerance_shorthand(&s);
        assert_eq!(parsed.max_delta, 1);
        assert_eq!(parsed.min_similarity, 95.0);
        assert_eq!(parsed.max_pixels_different, 1.0);
    }

    #[test]
    fn tolerance_shorthand_parse_legacy_tokens() {
        // Old format still parses correctly
        let parsed = parse_tolerance_shorthand("d:0 s:100");
        assert_eq!(parsed.max_delta, 0);
        assert_eq!(parsed.min_similarity, 100.0);

        // Legacy zdsim: token
        let parsed = parse_tolerance_shorthand("zdsim:0.005");
        assert_eq!(parsed.max_delta, 255);
        assert!((parsed.min_similarity - 99.5).abs() < 0.01);
        assert_eq!(parsed.max_pixels_different, 1.0);

        // Legacy with per-arch override
        let parsed = parse_tolerance_shorthand("d:1 s:99 px:1.0% [aarch64 d:3 s:90]");
        assert_eq!(parsed.max_delta, 1);
        assert_eq!(parsed.min_similarity, 99.0);
        let ov = &parsed.overrides["aarch64"];
        assert_eq!(ov.max_delta, Some(3));
        assert_eq!(ov.min_similarity, Some(90.0));

        // Legacy ignore-alpha
        let parsed = parse_tolerance_shorthand("d:0 s:100 ia");
        assert!(parsed.ignore_alpha);

        // Legacy alpha delta
        let parsed = parse_tolerance_shorthand("d:1 s:95 a:2");
        assert_eq!(parsed.max_alpha_delta, 2);
    }

    #[test]
    fn tolerance_shorthand_d1_s100_is_identical() {
        // max-delta:1 similarity:100 is effectively identical — similarity:100 makes delta irrelevant
        use crate::checksum_file::ToleranceSpec;
        let spec = ToleranceSpec {
            max_delta: 1,
            ..ToleranceSpec::exact()
        };
        let s = format_tolerance_shorthand(&spec);
        assert_eq!(s, "identical");
        // Roundtrip normalizes to max-delta:0
        let parsed = parse_tolerance_shorthand(&s);
        assert_eq!(parsed.max_delta, 0);
        assert_eq!(parsed.min_similarity, 100.0);
    }

    #[test]
    fn tolerance_shorthand_preset_with_overrides() {
        use crate::checksum_file::{ToleranceOverride, ToleranceSpec};
        use std::collections::BTreeMap;

        let spec = ToleranceSpec {
            overrides: BTreeMap::from([(
                "aarch64".to_string(),
                ToleranceOverride {
                    max_delta: Some(3),
                    ..Default::default()
                },
            )]),
            ..ToleranceSpec::off_by_one()
        };

        let s = format_tolerance_shorthand(&spec);
        assert_eq!(s, "off-by-one [aarch64 max-delta:3]");

        let parsed = parse_tolerance_shorthand(&s);
        assert_eq!(parsed.max_delta, 1);
        assert_eq!(parsed.min_similarity, 95.0);
        assert_eq!(parsed.max_pixels_different, 1.0);
        let ov = &parsed.overrides["aarch64"];
        assert_eq!(ov.max_delta, Some(3));
    }

    #[test]
    fn tolerance_shorthand_roundtrip_complex() {
        use crate::checksum_file::{ToleranceOverride, ToleranceSpec};
        use std::collections::BTreeMap;

        let spec = ToleranceSpec {
            max_delta: 2,
            min_similarity: 95.0,
            max_pixels_different: 0.01,
            max_alpha_delta: 0,
            ignore_alpha: false,
            overrides: BTreeMap::from([(
                "aarch64".to_string(),
                ToleranceOverride {
                    max_delta: Some(3),
                    min_similarity: Some(90.0),
                    ..Default::default()
                },
            )]),
        };

        let s = format_tolerance_shorthand(&spec);
        assert!(s.contains("max-delta:2"), "s={s}");
        assert!(s.contains("zensim:0.05"), "s={s}");
        assert!(s.contains("pixels-changed:1.0%"), "s={s}");
        assert!(s.contains("[aarch64 max-delta:3 zensim:0.10]"), "s={s}");

        let parsed = parse_tolerance_shorthand(&s);
        assert_eq!(parsed.max_delta, 2);
        assert_eq!(parsed.min_similarity, 95.0);
        assert!((parsed.max_pixels_different - 0.01).abs() < 1e-6);
        assert!(parsed.overrides.contains_key("aarch64"));
        let ov = &parsed.overrides["aarch64"];
        assert_eq!(ov.max_delta, Some(3));
        assert_eq!(ov.min_similarity, Some(90.0));
    }

    #[test]
    fn tolerance_shorthand_zensim_only() {
        use crate::checksum_file::ToleranceSpec;

        // Perceptual-only: max-delta:255 pixels-changed:100% means only zensim matters
        let spec = ToleranceSpec {
            max_delta: 255,
            min_similarity: 99.0,
            max_pixels_different: 1.0,
            ..ToleranceSpec::exact()
        };
        let s = format_tolerance_shorthand(&spec);
        assert_eq!(s, "zensim:0.01", "s={s}");

        let parsed = parse_tolerance_shorthand(&s);
        assert_eq!(parsed.max_delta, 255);
        assert_eq!(parsed.min_similarity, 99.0);
        assert_eq!(parsed.max_pixels_different, 1.0);
    }

    #[test]
    fn tolerance_shorthand_zensim_roundtrip_loose() {
        use crate::checksum_file::ToleranceSpec;

        let spec = ToleranceSpec {
            max_delta: 255,
            min_similarity: 95.0,
            max_pixels_different: 1.0,
            ..ToleranceSpec::exact()
        };
        let s = format_tolerance_shorthand(&spec);
        assert_eq!(s, "zensim:0.05", "s={s}");

        let parsed = parse_tolerance_shorthand(&s);
        assert_eq!(parsed.max_delta, 255);
        assert!((parsed.min_similarity - 95.0).abs() < 0.01);
    }

    #[test]
    fn tolerance_shorthand_compound_zensim() {
        // Compound: max-delta + zensim + pixels-changed
        let parsed = parse_tolerance_shorthand("max-delta:2 zensim:0.05 pixels-changed:1.0%");
        assert_eq!(parsed.max_delta, 2);
        assert!((parsed.min_similarity - 95.0).abs() < 0.01);
        assert!((parsed.max_pixels_different - 0.01).abs() < 1e-6);
    }

    #[test]
    fn tolerance_shorthand_ignore_alpha() {
        use crate::checksum_file::ToleranceSpec;

        // ignore-alpha prevents preset matching, falls back to explicit tokens
        let spec = ToleranceSpec {
            ignore_alpha: true,
            ..ToleranceSpec::exact()
        };
        let s = format_tolerance_shorthand(&spec);
        assert!(s.contains("ignore-alpha"), "s={s}");
        assert!(s.contains("max-delta:0"), "s={s}");

        let parsed = parse_tolerance_shorthand(&s);
        assert!(parsed.ignore_alpha);
    }
}
