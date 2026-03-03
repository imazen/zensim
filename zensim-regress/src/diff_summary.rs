//! Human-readable diff summary for `.checksums` entries.
//!
//! Produces the parenthesized diff string that appears after `vs` in auto-accepted entries:
//!
//! ```text
//! (zs:99.87, zd:0.0013, 2.1% px ±1, mean R:-0.23 G:+0.01 B:-0.15, maxΔ:[1,1,0], cat:rounding, balanced)
//! ```
//!
//! Also provides the **zdsim-0.1** metric: a DSSIM-style dissimilarity number where
//! 0 = identical and higher = worse. Simple linear conversion from the zensim score.

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
/// (zs:99.87, zd:0.0013, 2.1% px ±1, mean R:-0.23, maxΔ:[1,1,0], cat:rounding, balanced)
/// ```
///
/// Only includes fields when they carry information (omits trivially obvious values).
pub fn format_diff_summary(report: &RegressionReport) -> String {
    let mut parts = Vec::new();

    // Zensim score (always present)
    let score = report.score();
    parts.push(format!("zs:{score:.2}"));

    // Zdsim dissimilarity (always present)
    let zd = zdsim(score);
    if zd == 0.0 {
        parts.push("zd:0".to_string());
    } else if zd < 0.0001 {
        parts.push(format!("zd:{zd:.6}"));
    } else if zd < 0.01 {
        parts.push(format!("zd:{zd:.4}"));
    } else {
        parts.push(format!("zd:{zd:.3}"));
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
                parts.push(format!("{pct_str} px \u{00b1}1 LSB"));
            } else {
                let max_delta = *report.max_channel_delta().iter().max().unwrap_or(&0);
                parts.push(format!("{pct_str} px \u{00b1}{max_delta}"));
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
            parts.push(format!("max\u{0394}:[{r},{g},{b}]/{}", nm as u64));
        }
    } else {
        let mcd = report.max_channel_delta();
        if mcd != [0, 0, 0] {
            parts.push(format!("max\u{0394}:[{},{},{}]", mcd[0], mcd[1], mcd[2]));
        }
    }

    // Error category (always present when not Identical)
    let cat = report.category();
    let cat_str = match cat {
        zensim::ErrorCategory::Identical => None,
        zensim::ErrorCategory::RoundingError => Some("rounding"),
        zensim::ErrorCategory::ChannelSwap => Some("chanswap"),
        zensim::ErrorCategory::AlphaCompositing => Some("alpha"),
        _ => Some("unclassified"),
    };
    if let Some(c) = cat_str {
        parts.push(format!("cat:{c}"));
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
/// Output: `"within d:1 s:99.5"` or similar.
pub fn format_tolerance_note(tolerance: &crate::testing::RegressionTolerance) -> String {
    let mut parts = Vec::new();
    parts.push(format!("d:{}", tolerance.max_delta()));
    if tolerance.min_similarity() < 100.0 {
        let s = tolerance.min_similarity();
        if s == s.floor() {
            parts.push(format!("s:{s:.0}"));
        } else {
            parts.push(format!("s:{s}"));
        }
    }
    if tolerance.max_pixels_different() > 0.0 {
        let px = tolerance.max_pixels_different() * 100.0;
        parts.push(format!("px:{px:.1}%"));
    }
    format!("within {}", parts.join(" "))
}

// ─── Tolerance shorthand ─────────────────────────────────────────────────

/// Format a tolerance as the shorthand used in `.checksums` files.
///
/// ```text
/// d:0 s:100
/// d:2 s:95.0 px:1.0%
/// d:1 s:99.0 a:0 [aarch64 d:3 s:90.0]
/// ```
pub fn format_tolerance_shorthand(tolerance: &crate::checksum_file::ToleranceSpec) -> String {
    let mut parts = Vec::new();

    parts.push(format!("d:{}", tolerance.max_delta));

    let s = tolerance.min_similarity;
    if s == s.floor() {
        parts.push(format!("s:{s:.0}"));
    } else {
        parts.push(format!("s:{s}"));
    }

    if tolerance.max_pixels_different > 0.0 {
        let px = tolerance.max_pixels_different * 100.0;
        parts.push(format!("px:{px:.1}%"));
    }

    if tolerance.max_alpha_delta > 0 {
        parts.push(format!("a:{}", tolerance.max_alpha_delta));
    }

    if tolerance.ignore_alpha {
        parts.push("ia".to_string());
    }

    // Per-arch overrides
    for (arch, ov) in &tolerance.overrides {
        let mut ov_parts = Vec::new();
        if let Some(d) = ov.max_delta {
            ov_parts.push(format!("d:{d}"));
        }
        if let Some(s) = ov.min_similarity {
            if s == s.floor() {
                ov_parts.push(format!("s:{s:.0}"));
            } else {
                ov_parts.push(format!("s:{s}"));
            }
        }
        if let Some(px) = ov.max_pixels_different {
            ov_parts.push(format!("px:{:.1}%", px * 100.0));
        }
        if let Some(a) = ov.max_alpha_delta {
            ov_parts.push(format!("a:{a}"));
        }
        if !ov_parts.is_empty() {
            parts.push(format!("[{} {}]", arch, ov_parts.join(" ")));
        }
    }

    parts.join(" ")
}

/// Parse a tolerance shorthand string back into a [`ToleranceSpec`].
///
/// Handles the format produced by [`format_tolerance_shorthand`].
pub fn parse_tolerance_shorthand(s: &str) -> crate::checksum_file::ToleranceSpec {
    use crate::checksum_file::{ToleranceOverride, ToleranceSpec};
    use std::collections::BTreeMap;

    let mut spec = ToleranceSpec::default();
    let mut overrides: BTreeMap<String, ToleranceOverride> = BTreeMap::new();

    // Collect the non-bracket portion and extract bracketed overrides
    let input = s.trim();
    let mut main_parts = String::new();
    let mut pos = 0;

    while pos < input.len() {
        if let Some(bracket_start) = input[pos..].find('[') {
            let abs_start = pos + bracket_start;
            main_parts.push_str(&input[pos..abs_start]);

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
            main_parts.push_str(&input[pos..]);
            break;
        }
    }

    parse_main_tolerance_tokens(&main_parts, &mut spec);
    spec.overrides = overrides;
    spec
}

fn parse_main_tolerance_tokens(s: &str, spec: &mut crate::checksum_file::ToleranceSpec) {
    for token in s.split_whitespace() {
        if let Some(v) = token.strip_prefix("d:") {
            spec.max_delta = v.parse().unwrap_or(0);
        } else if let Some(v) = token.strip_prefix("s:") {
            spec.min_similarity = v.parse().unwrap_or(100.0);
        } else if let Some(v) = token.strip_prefix("px:") {
            let v = v.trim_end_matches('%');
            spec.max_pixels_different = v.parse::<f64>().unwrap_or(0.0) / 100.0;
        } else if let Some(v) = token.strip_prefix("a:") {
            spec.max_alpha_delta = v.parse().unwrap_or(0);
        } else if token == "ia" {
            spec.ignore_alpha = true;
        }
    }
}

fn parse_tolerance_token(token: &str, ov: &mut crate::checksum_file::ToleranceOverride) {
    if let Some(v) = token.strip_prefix("d:") {
        ov.max_delta = v.parse().ok();
    } else if let Some(v) = token.strip_prefix("s:") {
        ov.min_similarity = v.parse().ok();
    } else if let Some(v) = token.strip_prefix("px:") {
        let v = v.trim_end_matches('%');
        ov.max_pixels_different = v.parse::<f64>().ok().map(|p| p / 100.0);
    } else if let Some(v) = token.strip_prefix("a:") {
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
        let spec = crate::checksum_file::ToleranceSpec::default();
        let s = format_tolerance_shorthand(&spec);
        assert_eq!(s, "d:0 s:100");
        let parsed = parse_tolerance_shorthand(&s);
        assert_eq!(parsed.max_delta, 0);
        assert_eq!(parsed.min_similarity, 100.0);
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
        assert!(s.contains("d:2"), "s={s}");
        assert!(s.contains("s:95"), "s={s}");
        assert!(s.contains("px:1.0%"), "s={s}");
        assert!(s.contains("[aarch64 d:3 s:90]"), "s={s}");

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
    fn tolerance_shorthand_ignore_alpha() {
        use crate::checksum_file::ToleranceSpec;

        let spec = ToleranceSpec {
            ignore_alpha: true,
            ..Default::default()
        };
        let s = format_tolerance_shorthand(&spec);
        assert!(s.contains("ia"), "s={s}");

        let parsed = parse_tolerance_shorthand(&s);
        assert!(parsed.ignore_alpha);
    }
}
