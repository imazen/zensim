//! HTML report generation from test manifest TSV files.
//!
//! Reads the manifest TSV written during test execution and generates
//! a standalone HTML file showing:
//!
//! - Per-test status (match/novel/accepted/failed) with zensim dissimilarity scores
//! - Recommended tolerance for ratcheting down
//! - Diff images (embedded as base64 data URIs)
//! - Cross-platform merged view when multiple manifests are provided
//!
//! # Usage
//!
//! ```no_run
//! use zensim_regress::report::{parse_manifest, generate_html_report, Platform};
//!
//! let platform = Platform {
//!     name: "linux-x64".into(),
//!     manifest_path: "test-manifest.tsv".into(),
//!     diffs_dir: Some(".image-cache/diffs".into()),
//! };
//! let entries = parse_manifest(&platform.manifest_path).unwrap();
//! let diffs_dirs = std::collections::BTreeMap::new();
//! let html = generate_html_report(&[("linux-x64", &entries)], &diffs_dirs);
//! std::fs::write("report.html", html).unwrap();
//! ```

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};

use crate::diff_summary::{format_dissim as format_dissimilarity, format_score};

// ─── Manifest parsing ──────────────────────────────────────────────────

/// A parsed manifest entry from the TSV file.
#[derive(Debug, Clone)]
pub struct ParsedEntry {
    /// Fully qualified test name.
    pub test_name: String,
    /// Status string: `"match"`, `"novel"`, `"accepted"`, or `"failed"`.
    pub status: String,
    /// Measured dissimilarity (0.0 = identical).
    pub actual_zdsim: Option<f64>,
    /// Tolerance dissimilarity threshold.
    pub tolerance_zdsim: Option<f64>,
    /// Raw hash of actual output.
    pub actual_hash: String,
    /// Memorable name for the actual hash.
    pub actual_petname: String,
    /// File path for actual output image.
    pub actual_file: String,
    /// Raw hash of the baseline reference.
    pub baseline_hash: String,
    /// Memorable name for the baseline hash.
    pub baseline_petname: String,
    /// File path for the baseline reference image.
    pub baseline_file: String,
    /// Human-readable diff summary.
    pub diff_summary: String,
}

/// Parse a manifest TSV file into structured entries.
///
/// Skips the header line (starts with `#`). Returns entries in file order.
pub fn parse_manifest(path: &Path) -> Result<Vec<ParsedEntry>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    Ok(parse_manifest_content(&content))
}

/// Parse a manifest directory (from [`ManifestDir`]) into structured entries.
///
/// Reads all `*.tsv` files in the directory, sorted by filename (timestamp order).
/// Deduplicates by test name — latest timestamp wins.
///
/// [`ManifestDir`]: crate::manifest::ManifestDir
pub fn parse_manifest_dir(dir: &Path) -> Result<Vec<ParsedEntry>, std::io::Error> {
    use std::collections::BTreeMap;

    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "tsv"))
        .collect();
    files.sort_by_key(|e| e.file_name());

    let mut by_name: BTreeMap<String, ParsedEntry> = BTreeMap::new();
    for file in &files {
        let content = std::fs::read_to_string(file.path())?;
        for entry in parse_manifest_content(&content) {
            by_name.insert(entry.test_name.clone(), entry);
        }
    }

    Ok(by_name.into_values().collect())
}

/// Parse manifest TSV content (string) into entries.
fn parse_manifest_content(content: &str) -> Vec<ParsedEntry> {
    let mut entries = Vec::new();

    for line in content.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 11 {
            continue; // malformed line
        }

        entries.push(ParsedEntry {
            test_name: fields[0].to_string(),
            status: fields[1].to_string(),
            actual_zdsim: parse_opt_f64(fields[2]),
            tolerance_zdsim: parse_opt_f64(fields[3]),
            actual_hash: fields[4].to_string(),
            actual_petname: fields[5].to_string(),
            actual_file: fields[6].to_string(),
            baseline_hash: fields[7].to_string(),
            baseline_petname: fields[8].to_string(),
            baseline_file: fields[9].to_string(),
            diff_summary: fields[10].to_string(),
        });
    }

    entries
}

fn parse_opt_f64(s: &str) -> Option<f64> {
    if s == "-" { None } else { s.parse().ok() }
}

// ─── Recommended tolerance ─────────────────────────────────────────────

/// Headroom multiplier for recommended tolerance.
///
/// We add 50% headroom above the observed zdsim to account for
/// platform variance. The recommendation is `ceil(observed * 1.5, precision)`.
const HEADROOM: f64 = 1.5;

/// Compute a recommended zdsim tolerance from an observed value.
///
/// Adds headroom and rounds up to a clean value for the `.checksums` file.
/// Returns `None` for exact matches (zdsim=0).
pub fn recommend_tolerance(observed_zdsim: f64) -> Option<f64> {
    if observed_zdsim <= 0.0 {
        return None;
    }
    let with_headroom = observed_zdsim * HEADROOM;
    // Round up to 2 significant figures
    Some(ceil_sig2(with_headroom))
}

/// Ceiling to 2 significant figures.
fn ceil_sig2(v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    let exp = v.log10().floor() as i32;
    let factor = 10.0_f64.powi(exp - 1);
    // Round the intermediate result to avoid floating-point noise
    let scaled = (v / factor).ceil();
    // Reconstruct with integer arithmetic where possible
    scaled * factor
}

/// Format a recommended `.checksums` tolerance line using the dual-value format.
///
/// Takes a dissimilarity value and formats as `zensim:SCORE (dissim VALUE)`.
pub fn format_recommended_line(zdsim: f64) -> String {
    let score = zensim::dissimilarity_to_score(zdsim);
    format_recommended_from_score(score)
}

/// Format a recommended tolerance from a score value.
fn format_recommended_from_score(score: f64) -> String {
    if score >= 100.0 {
        return "zensim:100".to_string();
    }
    let score_str = format_score(score);
    let dissim = zensim::score_to_dissimilarity(score);
    let dissim_str = format_dissimilarity(dissim);
    format!("zensim:{score_str} (dissim {dissim_str})")
}

// ─── Diff image amplification ──────────────────────────────────────────

/// Compute the ideal diff amplification factor.
///
/// Uses `min(10, 255 / max_diff_pixel)` so that:
/// - Small diffs (≤25) get full 10x amplification
/// - Large diffs get scaled so the max pixel appears at 255
///
/// The result is always at least 1.
pub fn ideal_amplification(max_channel_delta: u8) -> u8 {
    if max_channel_delta == 0 {
        return 10;
    }
    let scaled = (255.0 / max_channel_delta as f64).floor() as u8;
    scaled.clamp(1, 10)
}

// ─── HTML report generation ────────────────────────────────────────────

/// A platform's test results for the merged report.
pub struct Platform {
    /// Platform identifier (e.g., `"ubuntu-latest"`, `"windows-11-arm"`).
    pub name: String,
    /// Path to a combined manifest TSV file, OR a directory of per-process
    /// manifest files (from [`ManifestDir`]). Both are auto-detected.
    ///
    /// [`ManifestDir`]: crate::manifest::ManifestDir
    pub manifest_path: PathBuf,
    /// Directory containing diff PNGs (e.g., `.image-cache/diffs/`).
    pub diffs_dir: Option<PathBuf>,
}

/// Generate a standalone HTML report from one or more platform manifests.
///
/// `platforms` is a slice of `(platform_name, entries)` pairs.
/// `diffs_dirs` maps platform names to their diff image directories.
///
/// The HTML is self-contained (no external CSS/JS dependencies) with
/// diff images embedded as base64 data URIs.
pub fn generate_html_report(
    platforms: &[(&str, &[ParsedEntry])],
    diffs_dirs: &BTreeMap<String, PathBuf>,
) -> String {
    // Merge by test name across platforms
    let mut merged: BTreeMap<String, BTreeMap<String, &ParsedEntry>> = BTreeMap::new();
    for &(platform, entries) in platforms {
        for entry in entries {
            merged
                .entry(entry.test_name.clone())
                .or_default()
                .insert(platform.to_string(), entry);
        }
    }

    let platform_names: Vec<&str> = platforms.iter().map(|p| p.0).collect();
    let multi_platform = platform_names.len() > 1;

    let mut html = String::with_capacity(64 * 1024);

    // ─── HTML head ─────────────────────────────────────────────
    write_html_head(&mut html, &platform_names);

    // ─── Summary stats ─────────────────────────────────────────
    write_summary_stats(&mut html, platforms);

    // ─── Non-match entries (interesting ones first) ────────────
    let _ = writeln!(html, "<h2>Non-Match Results</h2>");
    let _ = writeln!(
        html,
        "<p class=\"hint\">Tests that differed from baseline. \
         Sorted by status (failed first), then by zensim (highest first).</p>"
    );

    let mut non_match: Vec<(&str, &BTreeMap<String, &ParsedEntry>)> = merged
        .iter()
        .filter(|(_, plats)| plats.values().any(|e| e.status != "match"))
        .map(|(name, plats)| (name.as_str(), plats))
        .collect();

    // Sort: failed first, then by max zdsim descending
    non_match.sort_by(|a, b| {
        let a_failed = a.1.values().any(|e| e.status == "failed");
        let b_failed = b.1.values().any(|e| e.status == "failed");
        let a_zdsim =
            a.1.values()
                .filter_map(|e| e.actual_zdsim)
                .fold(0.0_f64, f64::max);
        let b_zdsim =
            b.1.values()
                .filter_map(|e| e.actual_zdsim)
                .fold(0.0_f64, f64::max);
        b_failed.cmp(&a_failed).then(
            b_zdsim
                .partial_cmp(&a_zdsim)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });

    if non_match.is_empty() {
        let _ = writeln!(html, "<p class=\"ok\">All tests matched exactly.</p>");
    } else {
        for (test_name, plats) in &non_match {
            write_test_entry(
                &mut html,
                test_name,
                plats,
                &platform_names,
                multi_platform,
                diffs_dirs,
            );
        }
    }

    // ─── All-match entries (collapsed) ─────────────────────────
    let match_count = merged.len() - non_match.len();
    if match_count > 0 {
        let _ = write!(
            html,
            "<details class=\"matches\">\n<summary>{match_count} exact matches (click to expand)</summary>\n<ul>\n"
        );
        for (test_name, plats) in &merged {
            if plats.values().all(|e| e.status == "match") {
                let _ = writeln!(html, "<li><code>{test_name}</code></li>");
            }
        }
        let _ = write!(html, "</ul>\n</details>\n");
    }

    // ─── Recommended tolerance block ───────────────────────────
    write_recommended_tolerances(&mut html, &non_match);

    // ─── Footer ────────────────────────────────────────────────
    let _ = write!(html, "</div>\n</body>\n</html>\n");

    html
}

// ─── HTML building helpers ─────────────────────────────────────────────

fn write_html_head(html: &mut String, platform_names: &[&str]) {
    let title = if platform_names.len() == 1 {
        format!("Regression Report — {}", platform_names[0])
    } else {
        format!("Regression Report — {} platforms", platform_names.len())
    };
    let _ = write!(
        html,
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
:root {{
    --bg: #1a1a2e;
    --card: #16213e;
    --border: #0f3460;
    --text: #e0e0e0;
    --accent: #e94560;
    --ok: #4ecca3;
    --warn: #f0a500;
    --code-bg: #0d1117;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 2rem;
}}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ color: var(--text); margin-bottom: 0.5rem; }}
h2 {{ color: var(--text); margin: 2rem 0 0.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.25rem; }}
.hint {{ color: #888; font-size: 0.9rem; margin-bottom: 1rem; }}
.ok {{ color: var(--ok); }}
.stats {{
    display: flex;
    gap: 1.5rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}}
.stat {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    min-width: 120px;
}}
.stat .label {{ font-size: 0.8rem; color: #888; text-transform: uppercase; }}
.stat .value {{ font-size: 1.5rem; font-weight: 600; }}
.stat .value.fail {{ color: var(--accent); }}
.stat .value.warn {{ color: var(--warn); }}
.stat .value.pass {{ color: var(--ok); }}
.test-entry {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin: 1rem 0;
    padding: 1rem 1.25rem;
}}
.test-entry.failed {{ border-left: 4px solid var(--accent); }}
.test-entry.accepted {{ border-left: 4px solid var(--warn); }}
.test-entry.novel {{ border-left: 4px solid #666; }}
.test-header {{
    display: flex;
    align-items: baseline;
    gap: 1rem;
    flex-wrap: wrap;
}}
.test-name {{ font-weight: 600; font-size: 1.1rem; font-family: monospace; }}
.badge {{
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
}}
.badge.match {{ background: var(--ok); color: #000; }}
.badge.accepted {{ background: var(--warn); color: #000; }}
.badge.failed {{ background: var(--accent); color: #fff; }}
.badge.novel {{ background: #555; color: #fff; }}
.metrics {{
    display: flex;
    gap: 1.5rem;
    margin: 0.5rem 0;
    font-family: monospace;
    font-size: 0.9rem;
    flex-wrap: wrap;
}}
.metrics .metric {{ }}
.metrics .metric .label {{ color: #888; font-size: 0.75rem; }}
.diff-summary {{ color: #aaa; font-size: 0.85rem; font-family: monospace; margin: 0.25rem 0; }}
.diff-img {{ margin: 0.75rem 0; max-width: 100%; border-radius: 4px; }}
.diff-img img {{ max-width: 100%; border-radius: 4px; image-rendering: pixelated; }}
.recommend {{
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    font-family: monospace;
    font-size: 0.85rem;
    margin: 0.5rem 0;
    white-space: pre;
    overflow-x: auto;
}}
.platform-tag {{
    font-size: 0.75rem;
    color: #aaa;
    background: var(--code-bg);
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    margin-right: 0.25rem;
}}
details summary {{ cursor: pointer; color: #888; margin: 1rem 0 0.5rem; }}
details summary:hover {{ color: var(--text); }}
.matches ul {{ list-style: none; padding: 0; columns: 3; }}
.matches li {{ font-family: monospace; font-size: 0.85rem; padding: 0.1rem 0; }}
pre.tolerances {{
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
    font-size: 0.85rem;
    line-height: 1.6;
}}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
"#
    );
}

fn write_summary_stats(html: &mut String, platforms: &[(&str, &[ParsedEntry])]) {
    let _ = writeln!(html, "<div class=\"stats\">");

    for &(name, entries) in platforms {
        let total = entries.len();
        let matched = entries.iter().filter(|e| e.status == "match").count();
        let accepted = entries.iter().filter(|e| e.status == "accepted").count();
        let failed = entries.iter().filter(|e| e.status == "failed").count();
        let novel = entries.iter().filter(|e| e.status == "novel").count();

        let _ = writeln!(
            html,
            "<div class=\"stat\"><div class=\"label\">{name}</div>\
             <div class=\"value pass\">{matched}</div> match</div>"
        );
        if accepted > 0 {
            let _ = writeln!(
                html,
                "<div class=\"stat\"><div class=\"label\">{name}</div>\
                 <div class=\"value warn\">{accepted}</div> accepted</div>"
            );
        }
        if failed > 0 {
            let _ = writeln!(
                html,
                "<div class=\"stat\"><div class=\"label\">{name}</div>\
                 <div class=\"value fail\">{failed}</div> failed</div>"
            );
        }
        if novel > 0 {
            let _ = writeln!(
                html,
                "<div class=\"stat\"><div class=\"label\">{name}</div>\
                 <div class=\"value\">{novel}</div> novel</div>"
            );
        }
        let _ = writeln!(
            html,
            "<div class=\"stat\"><div class=\"label\">{name} total</div>\
             <div class=\"value\">{total}</div></div>"
        );
    }

    let _ = writeln!(html, "</div>");
}

fn write_test_entry(
    html: &mut String,
    test_name: &str,
    platforms: &BTreeMap<String, &ParsedEntry>,
    platform_names: &[&str],
    multi_platform: bool,
    diffs_dirs: &BTreeMap<String, PathBuf>,
) {
    // Determine worst status across platforms
    let worst_status = if platforms.values().any(|e| e.status == "failed") {
        "failed"
    } else if platforms.values().any(|e| e.status == "accepted") {
        "accepted"
    } else if platforms.values().any(|e| e.status == "novel") {
        "novel"
    } else {
        "match"
    };

    let _ = write!(
        html,
        "<div class=\"test-entry {worst_status}\">\n<div class=\"test-header\">\n\
         <span class=\"test-name\">{}</span>\n\
         <span class=\"badge {worst_status}\">{worst_status}</span>\n\
         </div>\n",
        html_escape(test_name)
    );

    for &platform_name in platform_names {
        if let Some(entry) = platforms.get(platform_name) {
            if multi_platform {
                let _ = writeln!(html, "<span class=\"platform-tag\">{platform_name}</span>");
            }

            // Metrics row
            let _ = writeln!(html, "<div class=\"metrics\">");
            if let Some(zdsim) = entry.actual_zdsim {
                let score = zensim::dissimilarity_to_score(zdsim);
                let _ = writeln!(
                    html,
                    "<div class=\"metric\"><div class=\"label\">zensim score</div>{:.2} <span style=\"color:#888\">(dissim {})</span></div>",
                    score,
                    format_dissimilarity(zdsim)
                );
            }
            if let Some(tol) = entry.tolerance_zdsim {
                let tol_score = zensim::dissimilarity_to_score(tol);
                let _ = writeln!(
                    html,
                    "<div class=\"metric\"><div class=\"label\">tolerance</div>{:.1} <span style=\"color:#888\">(dissim {})</span></div>",
                    tol_score,
                    format_dissimilarity(tol)
                );
            }
            // Recommendation
            if let Some(zdsim) = entry.actual_zdsim
                && let Some(rec) = recommend_tolerance(zdsim)
            {
                let _ = writeln!(
                    html,
                    "<div class=\"metric\"><div class=\"label\">recommended</div>{}</div>",
                    format_recommended_line(rec)
                );
            }
            let _ = writeln!(html, "</div>");

            // Diff summary
            if entry.diff_summary != "-" && !entry.diff_summary.is_empty() {
                let _ = writeln!(
                    html,
                    "<div class=\"diff-summary\">{}</div>",
                    html_escape(&entry.diff_summary)
                );
            }

            // Diff image (try to embed from this platform's diffs directory)
            if (entry.status == "accepted" || entry.status == "failed")
                && let Some(base) = diffs_dirs.get(platform_name)
            {
                try_embed_diff_image(html, base, test_name);
            }
        }
    }

    let _ = writeln!(html, "</div>");
}

fn write_recommended_tolerances(
    html: &mut String,
    non_match: &[(&str, &BTreeMap<String, &ParsedEntry>)],
) {
    let accepted_or_failed: Vec<_> = non_match
        .iter()
        .filter(|(_, plats)| {
            plats
                .values()
                .any(|e| e.status == "accepted" || e.status == "failed")
        })
        .collect();

    if accepted_or_failed.is_empty() {
        return;
    }

    let _ = writeln!(html, "<h2>Recommended Tolerances</h2>");
    let _ = writeln!(
        html,
        "<p class=\"hint\">Copy these into your .checksums tolerance lines to ratchet down. \
         Values include {}% headroom above observed maximum across all platforms.</p>",
        ((HEADROOM - 1.0) * 100.0) as u32
    );

    let _ = write!(html, "<pre class=\"tolerances\">");
    for (test_name, plats) in &accepted_or_failed {
        // Find max observed zdsim across all platforms
        let max_zdsim = plats
            .values()
            .filter_map(|e| e.actual_zdsim)
            .fold(0.0_f64, f64::max);

        if max_zdsim > 0.0
            && let Some(rec) = recommend_tolerance(max_zdsim)
        {
            let observed_score = zensim::dissimilarity_to_score(max_zdsim);
            let _ = writeln!(
                html,
                "# {test_name}: observed zensim:{:.2} (dissim {}) → tolerance {}",
                observed_score,
                format_dissimilarity(max_zdsim),
                format_recommended_line(rec),
            );
        }
    }
    let _ = writeln!(html, "</pre>");
}

/// Try to find and embed a diff image as a base64 data URI.
fn try_embed_diff_image(html: &mut String, diffs_dir: &Path, test_name: &str) {
    // The diff image filename is the sanitized test name + .png
    let sanitized = test_name.replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '_', "_");
    let path = diffs_dir.join(format!("{sanitized}.png"));

    if let Ok(data) = std::fs::read(&path) {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
        let _ = writeln!(
            html,
            "<div class=\"diff-img\">\
             <img src=\"data:image/png;base64,{b64}\" alt=\"diff: {test_name}\">\
             </div>"
        );
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ─── Multi-platform merge ──────────────────────────────────────────────

/// Load manifests from multiple platforms and generate a merged HTML report.
///
/// Each `Platform` specifies a manifest path (file or directory) and optional
/// diffs directory. Auto-detects whether the path is a single TSV file or a
/// directory of per-process files (from [`ManifestDir`]).
///
/// [`ManifestDir`]: crate::manifest::ManifestDir
pub fn generate_merged_report(platforms: &[Platform]) -> Result<String, std::io::Error> {
    let mut all_entries: Vec<(String, Vec<ParsedEntry>)> = Vec::new();

    for p in platforms {
        let entries = if p.manifest_path.is_dir() {
            parse_manifest_dir(&p.manifest_path)?
        } else {
            parse_manifest(&p.manifest_path)?
        };
        all_entries.push((p.name.clone(), entries));
    }

    let refs: Vec<(&str, &[ParsedEntry])> = all_entries
        .iter()
        .map(|(name, entries)| (name.as_str(), entries.as_slice()))
        .collect();

    // Build per-platform diffs_dir map
    let diffs_dirs: BTreeMap<String, PathBuf> = platforms
        .iter()
        .filter_map(|p| p.diffs_dir.as_ref().map(|d| (p.name.clone(), d.clone())))
        .collect();

    Ok(generate_html_report(&refs, &diffs_dirs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_sig2_values() {
        // Use format_dissimilarity to compare — it rounds away float noise
        assert_eq!(format_dissimilarity(ceil_sig2(0.0036)), "0.0036");
        assert_eq!(format_dissimilarity(ceil_sig2(0.0153)), "0.016");
        assert_eq!(format_dissimilarity(ceil_sig2(0.0999)), "0.10");
        assert_eq!(format_dissimilarity(ceil_sig2(0.1)), "0.10");
        assert_eq!(format_dissimilarity(ceil_sig2(0.123)), "0.13");
        assert_eq!(format_dissimilarity(ceil_sig2(0.0012)), "0.0012");
    }

    #[test]
    fn recommend_tolerance_values() {
        // zdsim=0 → None (exact match)
        assert_eq!(recommend_tolerance(0.0), None);

        // Small zdsim gets headroom
        let rec = recommend_tolerance(0.001).unwrap();
        assert!(rec >= 0.001 * HEADROOM);
        assert!(rec <= 0.005); // reasonable upper bound

        // Medium zdsim
        let rec = recommend_tolerance(0.01).unwrap();
        assert!(rec >= 0.01 * HEADROOM);
        assert!(rec <= 0.02);
    }

    #[test]
    fn ideal_amplification_values() {
        // Small diff → full 10x
        assert_eq!(ideal_amplification(1), 10);
        assert_eq!(ideal_amplification(5), 10);
        assert_eq!(ideal_amplification(25), 10);

        // At boundary: 255/25 = 10.2 → 10
        assert_eq!(ideal_amplification(25), 10);

        // Larger diffs → reduced amplification
        assert_eq!(ideal_amplification(50), 5); // 255/50 = 5.1
        assert_eq!(ideal_amplification(100), 2); // 255/100 = 2.55
        assert_eq!(ideal_amplification(200), 1); // 255/200 = 1.275

        // Zero diff → default 10
        assert_eq!(ideal_amplification(0), 10);
    }

    #[test]
    fn format_dissimilarity_precision() {
        assert_eq!(format_dissimilarity(0.0), "0");
        assert_eq!(format_dissimilarity(0.000012), "0.000012"); // < 0.0001
        assert_eq!(format_dissimilarity(0.0001), "0.0001"); // < 0.001
        assert_eq!(format_dissimilarity(0.005), "0.005"); // clean 3-decimal
        assert_eq!(format_dissimilarity(0.0056), "0.0056"); // < 0.01
        assert_eq!(format_dissimilarity(0.05), "0.05"); // clean 2-decimal
        assert_eq!(format_dissimilarity(0.056), "0.056"); // < 0.1
        assert_eq!(format_dissimilarity(0.56), "0.56"); // >= 0.1
    }

    #[test]
    fn parse_manifest_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.tsv");

        // Write a manifest with the writer
        let writer = crate::manifest::ManifestWriter::create(&path).unwrap();
        writer.write_entry(&crate::manifest::ManifestEntry {
            test_name: "resize_test",
            status: crate::manifest::ManifestStatus::Accepted,
            actual_hash: "sea:a4839401fabae99c",
            baseline_hash: Some("sea:bbbb222233334444"),
            actual_zdsim: Some(0.0036),
            tolerance_zdsim: Some(0.05),
            diff_summary: Some("(zs:99.64, zd:0.0036)"),
        });
        drop(writer);

        // Parse it back
        let entries = parse_manifest(&path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].test_name, "resize_test");
        assert_eq!(entries[0].status, "accepted");
        assert!(entries[0].actual_zdsim.unwrap() > 0.003);
        assert!(entries[0].tolerance_zdsim.unwrap() > 0.04);
    }

    #[test]
    fn html_report_smoke() {
        let entries = vec![
            ParsedEntry {
                test_name: "exact_match_test".into(),
                status: "match".into(),
                actual_zdsim: Some(0.0),
                tolerance_zdsim: Some(0.01),
                actual_hash: "sea:aaaa".into(),
                actual_petname: "happy-cat-aaaa:sea".into(),
                actual_file: "sea_aaaa.png".into(),
                baseline_hash: "sea:aaaa".into(),
                baseline_petname: "happy-cat-aaaa:sea".into(),
                baseline_file: "sea_aaaa.png".into(),
                diff_summary: "-".into(),
            },
            ParsedEntry {
                test_name: "tolerance_test".into(),
                status: "accepted".into(),
                actual_zdsim: Some(0.0036),
                tolerance_zdsim: Some(0.05),
                actual_hash: "sea:bbbb".into(),
                actual_petname: "sad-dog-bbbb:sea".into(),
                actual_file: "sea_bbbb.png".into(),
                baseline_hash: "sea:cccc".into(),
                baseline_petname: "old-fox-cccc:sea".into(),
                baseline_file: "sea_cccc.png".into(),
                diff_summary: "(zs:99.64, zd:0.0036, 1.2% px ±1)".into(),
            },
            ParsedEntry {
                test_name: "broken_test".into(),
                status: "failed".into(),
                actual_zdsim: Some(0.15),
                tolerance_zdsim: Some(0.01),
                actual_hash: "sea:dddd".into(),
                actual_petname: "red-bear-dddd:sea".into(),
                actual_file: "sea_dddd.png".into(),
                baseline_hash: "sea:eeee".into(),
                baseline_petname: "blue-bird-eeee:sea".into(),
                baseline_file: "sea_eeee.png".into(),
                diff_summary: "(zs:85.0, zd:0.15)".into(),
            },
        ];

        let html = generate_html_report(&[("linux-x64", &entries)], &BTreeMap::new());

        // Basic structure checks
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Regression Report"));
        assert!(html.contains("exact_match_test"));
        assert!(html.contains("tolerance_test"));
        assert!(html.contains("broken_test"));
        assert!(html.contains("Recommended Tolerances"));
        assert!(html.contains("zensim:"));
        // Failed should appear before accepted
        let failed_pos = html.find("broken_test").unwrap();
        let accepted_pos = html.find("tolerance_test").unwrap();
        assert!(
            failed_pos < accepted_pos,
            "failed entries should appear before accepted"
        );
    }
}
