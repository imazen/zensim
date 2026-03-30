//! `.checksums` format: line-oriented checksum log files.
//!
//! Each test source file gets a sibling `.checksums` file containing sections
//! for each test function. Entries track human-verified baselines, auto-accepted
//! variants, and retired superseded hashes.
//!
//! # Format overview
//!
//! ```text
//! # trim.checksums
//!
//! ## test_trim_whitespace transparent_shirt
//! tolerance identical
//! = sunny-crab-a4839:sea  x86_64-avx512  @773c807  human-verified
//! ~ tidy-frog-b2c3d:sea   aarch64        @773c807  auto-accepted (within off-by-one) vs sunny-crab-a4839:sea (zensim:0.0013, ...)
//! ```
//!
//! See the plan document for full format specification.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use zensim::{ImageSource, PixelFormat, RgbaSlice, Zensim, ZensimProfile};

use crate::diff_image::{AnnotationText, MontageOptions};
use crate::diff_summary::{format_diff_summary, format_tolerance_shorthand};
use crate::error::RegressError;
use crate::hasher::{ChecksumHasher, SeaHasher};
use crate::manifest::{ManifestEntry, ManifestStatus, ManifestWriter};
use crate::remote::ReferenceStorage;
use crate::testing::{RegressionReport, RegressionTolerance, check_regression};
use crate::tolerance::ToleranceSpec;

// ─── Top-level file ──────────────────────────────────────────────────────

/// A `.checksums` file containing test sections.
#[derive(Debug, Clone)]
pub struct ChecksumsFile {
    /// Header comments (before any section).
    pub header_comments: Vec<String>,
    /// Test sections in file order.
    pub sections: Vec<TestSection>,
}

impl ChecksumsFile {
    /// Create an empty file with a version header comment.
    pub fn new(filename_stem: &str) -> Self {
        Self {
            header_comments: vec![format!("# {filename_stem}.checksums \u{2014} v1")],
            sections: Vec::new(),
        }
    }

    /// Find a section by test name and detail name.
    pub fn find_section(&self, test_name: &str, detail_name: &str) -> Option<&TestSection> {
        self.sections
            .iter()
            .find(|s| s.test_name == test_name && s.detail_name == detail_name)
    }

    /// Find a mutable section by test name and detail name.
    pub fn find_section_mut(
        &mut self,
        test_name: &str,
        detail_name: &str,
    ) -> Option<&mut TestSection> {
        self.sections
            .iter_mut()
            .find(|s| s.test_name == test_name && s.detail_name == detail_name)
    }

    /// Get or create a section.
    pub fn get_or_create_section(
        &mut self,
        test_name: &str,
        detail_name: &str,
    ) -> &mut TestSection {
        // Check if it exists
        let idx = self
            .sections
            .iter()
            .position(|s| s.test_name == test_name && s.detail_name == detail_name);

        if let Some(idx) = idx {
            &mut self.sections[idx]
        } else {
            self.sections.push(TestSection {
                test_name: test_name.to_string(),
                detail_name: detail_name.to_string(),
                tolerance: None,
                entries: Vec::new(),
            });
            self.sections.last_mut().unwrap()
        }
    }

    /// All section names as `(test_name, detail_name)` pairs.
    pub fn section_names(&self) -> Vec<(&str, &str)> {
        self.sections
            .iter()
            .map(|s| (s.test_name.as_str(), s.detail_name.as_str()))
            .collect()
    }

    /// Parse a `.checksums` file from a string.
    pub fn parse(content: &str) -> Self {
        let mut file = ChecksumsFile {
            header_comments: Vec::new(),
            sections: Vec::new(),
        };

        let mut current_section: Option<TestSection> = None;

        for line in content.lines() {
            let trimmed = line.trim();

            // Section header: ## test_name detail_name
            if let Some(header) = trimmed.strip_prefix("## ") {
                // Save previous section
                if let Some(section) = current_section.take() {
                    file.sections.push(section);
                }

                let (test_name, detail_name) = split_section_header(header);
                current_section = Some(TestSection {
                    test_name,
                    detail_name,
                    tolerance: None,
                    entries: Vec::new(),
                });
                continue;
            }

            // Tolerance line
            if let Some(tol_str) = trimmed.strip_prefix("tolerance ") {
                if let Some(ref mut section) = current_section {
                    section.tolerance =
                        Some(crate::diff_summary::parse_tolerance_shorthand(tol_str));
                }
                continue;
            }

            // Entry lines: = ~ x
            if let Some(entry) = parse_entry_line(trimmed) {
                if let Some(ref mut section) = current_section {
                    section.entries.push(entry);
                }
                continue;
            }

            // Comment or blank line before any section → header comment
            if current_section.is_none() && (trimmed.starts_with('#') || trimmed.is_empty()) {
                file.header_comments.push(line.to_string());
            }
            // Comments inside sections are silently dropped on parse
            // (they're not semantically meaningful)
        }

        // Save last section
        if let Some(section) = current_section {
            file.sections.push(section);
        }

        // Strip trailing blank lines from header comments to prevent
        // accumulation on read-modify-write cycles (format() adds its own
        // blank line between header and first section).
        while file
            .header_comments
            .last()
            .is_some_and(|l| l.trim().is_empty())
        {
            file.header_comments.pop();
        }

        file
    }

    /// Serialize to the `.checksums` text format.
    pub fn format(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        // Header comments
        for comment in &self.header_comments {
            writeln!(out, "{comment}").unwrap();
        }

        for (i, section) in self.sections.iter().enumerate() {
            // Blank line before section (after header or previous section)
            if i > 0 || !self.header_comments.is_empty() {
                out.push('\n');
            }

            // Section header
            writeln!(out, "## {} {}", section.test_name, section.detail_name).unwrap();

            // Tolerance
            if let Some(ref tol) = section.tolerance {
                writeln!(
                    out,
                    "tolerance {}",
                    crate::diff_summary::format_tolerance_shorthand(tol)
                )
                .unwrap();
            }

            // Entries
            for entry in &section.entries {
                writeln!(out, "{}", format_entry_line(entry)).unwrap();
            }
        }

        out
    }

    /// Read a `.checksums` file from disk.
    pub fn read_from(path: &Path) -> Result<Self, RegressError> {
        let content = std::fs::read_to_string(path).map_err(|e| RegressError::io(path, e))?;
        Ok(Self::parse(&content))
    }

    /// Write the `.checksums` file to disk.
    pub fn write_to(&self, path: &Path) -> Result<(), RegressError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| RegressError::io(parent, e))?;
        }
        let content = self.format();
        std::fs::write(path, content).map_err(|e| RegressError::io(path, e))?;
        Ok(())
    }
}

// ─── Test section ────────────────────────────────────────────────────────

/// A single test section within a `.checksums` file.
#[derive(Debug, Clone)]
pub struct TestSection {
    /// Test function name (e.g., `"test_fill_rect"`).
    pub test_name: String,
    /// Detail/variant name (e.g., `"eeccff_hermite_400x400"`).
    pub detail_name: String,
    /// Tolerance in effect for this test (informational).
    pub tolerance: Option<crate::tolerance::ToleranceSpec>,
    /// Checksum entries in file order.
    pub entries: Vec<ChecksumEntry>,
}

impl TestSection {
    /// The human-verified anchor entry (first `=` entry, or most recent).
    pub fn anchor(&self) -> Option<&ChecksumEntry> {
        self.entries
            .iter()
            .find(|e| e.kind == EntryKind::HumanVerified)
    }

    /// All active entries (human-verified + auto-accepted, not retired).
    pub fn active_entries(&self) -> impl Iterator<Item = &ChecksumEntry> {
        self.entries.iter().filter(|e| e.kind != EntryKind::Retired)
    }

    /// Find an entry by its memorable name+hash.
    pub fn find_by_name_hash(&self, name_hash: &str) -> Option<&ChecksumEntry> {
        self.entries.iter().find(|e| e.name_hash == name_hash)
    }

    /// Remove auto-accepted entries for a specific arch, keeping only the latest.
    ///
    /// Returns the number of entries removed.
    pub fn prune_auto_accepted(&mut self, arch: &str) -> usize {
        // Find all auto-accepted entries for this arch
        let mut indices: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.kind == EntryKind::AutoAccepted && e.arch == arch)
            .map(|(i, _)| i)
            .collect();

        if indices.len() <= 1 {
            return 0;
        }

        // Keep the last one (most recent), remove the rest
        indices.pop(); // keep last
        let removed = indices.len();
        // Remove in reverse order to preserve indices
        for &i in indices.iter().rev() {
            self.entries.remove(i);
        }
        removed
    }

    /// Retire the current anchor and all auto-accepted entries that reference it.
    ///
    /// Used when a new human-verified entry replaces the old anchor.
    pub fn retire_anchor(&mut self) {
        let anchor_name = self.anchor().map(|a| a.name_hash.clone());
        if let Some(anchor_name) = anchor_name {
            for entry in &mut self.entries {
                if entry.kind == EntryKind::HumanVerified && entry.name_hash == anchor_name {
                    entry.kind = EntryKind::Retired;
                } else if entry.kind == EntryKind::AutoAccepted {
                    // Remove auto-accepted entries referencing old anchor
                    if entry.vs_ref.as_ref().is_some_and(|vs| vs == &anchor_name) {
                        entry.kind = EntryKind::Retired;
                    }
                }
            }

            // Actually remove the auto-accepted that referenced old anchor
            // (they become noise — plan says "old ~ entries pruned")
            self.entries.retain(|e| {
                !(e.kind == EntryKind::Retired
                    && e.vs_ref.as_ref().is_some_and(|vs| vs == &anchor_name)
                    && e.name_hash != anchor_name)
            });
        }
    }
}

// ─── Checksum entry ──────────────────────────────────────────────────────

/// Kind of entry in a `.checksums` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// `=` Human-verified baseline. Trust anchor, never auto-pruned.
    HumanVerified,
    /// `~` Auto-accepted within tolerance. Pruned to one per arch.
    AutoAccepted,
    /// `x` Retired (superseded human-verified). Kept as history.
    Retired,
}

impl EntryKind {
    /// The prefix character for this entry kind.
    pub fn prefix_char(self) -> char {
        match self {
            Self::HumanVerified => '=',
            Self::AutoAccepted => '~',
            Self::Retired => 'x',
        }
    }

    /// Parse from a prefix character.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '=' => Some(Self::HumanVerified),
            '~' => Some(Self::AutoAccepted),
            'x' => Some(Self::Retired),
            _ => None,
        }
    }
}

/// A single checksum entry in a `.checksums` section.
#[derive(Debug, Clone)]
pub struct ChecksumEntry {
    /// Entry kind: `=` human-verified, `~` auto-accepted, `x` retired.
    pub kind: EntryKind,
    /// Memorable name with hash (e.g., `"sunny-crab-a4839:sea"`).
    pub name_hash: String,
    /// Architecture tag (e.g., `"x86_64-avx512"`).
    pub arch: String,
    /// Git commit hash (short form, e.g., `"773c807"`).
    pub commit: String,
    /// Reason string (e.g., `"human-verified"`, `"auto-accepted"`).
    pub reason: String,
    /// Tolerance note for auto-accepted entries (e.g., `"within d:1 s:99.5"`).
    pub tolerance_note: Option<String>,
    /// Reference entry's name_hash that this was compared against.
    pub vs_ref: Option<String>,
    /// Human-readable diff summary (the parenthesized string).
    pub diff_summary: Option<String>,
}

impl ChecksumEntry {
    /// Create a new human-verified entry.
    pub fn human_verified(name_hash: String, arch: String, commit: String) -> Self {
        Self {
            kind: EntryKind::HumanVerified,
            name_hash,
            arch,
            commit,
            reason: "human-verified".to_string(),
            tolerance_note: None,
            vs_ref: None,
            diff_summary: None,
        }
    }

    /// Create a new auto-accepted entry.
    pub fn auto_accepted(
        name_hash: String,
        arch: String,
        commit: String,
        tolerance_note: String,
        vs_ref: String,
        diff_summary: String,
    ) -> Self {
        Self {
            kind: EntryKind::AutoAccepted,
            name_hash,
            arch,
            commit,
            reason: "auto-accepted".to_string(),
            tolerance_note: Some(tolerance_note),
            vs_ref: Some(vs_ref),
            diff_summary: Some(diff_summary),
        }
    }
}

// ─── Parsing ─────────────────────────────────────────────────────────────

fn split_section_header(header: &str) -> (String, String) {
    let mut parts = header.splitn(2, ' ');
    let test_name = parts.next().unwrap_or("").to_string();
    let detail_name = parts.next().unwrap_or("").to_string();
    (test_name, detail_name)
}

/// Parse a single entry line (starting with `=`, `~`, or `x`).
fn parse_entry_line(line: &str) -> Option<ChecksumEntry> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    let first_char = trimmed.chars().next()?;
    let kind = EntryKind::from_char(first_char)?;

    // Rest after the prefix char and space
    let rest = trimmed[first_char.len_utf8()..].trim_start();

    // Tokenize: name_hash arch @commit reason [tolerance_note] [vs ref (diff)]
    let mut tokens = Tokenizer::new(rest);

    let name_hash = tokens.next_word()?.to_string();
    let arch = tokens.next_word()?.to_string();

    let commit_token = tokens.next_word()?;
    let commit = commit_token
        .strip_prefix('@')
        .unwrap_or(commit_token)
        .to_string();

    // The rest is the reason, possibly followed by "vs" clause
    let remaining = tokens.remaining();

    let (reason, tolerance_note, vs_ref, diff_summary) = parse_reason_and_vs(remaining);

    Some(ChecksumEntry {
        kind,
        name_hash,
        arch,
        commit,
        reason,
        tolerance_note,
        vs_ref,
        diff_summary,
    })
}

/// Parse the reason string and optional `vs` clause from the remainder of an entry line.
fn parse_reason_and_vs(s: &str) -> (String, Option<String>, Option<String>, Option<String>) {
    let s = s.trim();

    // Look for " vs " to split reason from diff reference
    if let Some(vs_pos) = s.find(" vs ") {
        let reason_part = s[..vs_pos].trim();
        let vs_part = &s[vs_pos + 4..]; // after " vs "

        // Extract tolerance note from reason if present: "auto-accepted (within d:1 s:99.5)"
        let (reason, tolerance_note) = extract_parenthesized(reason_part);

        // vs_part: "sunny-crab-a4839:sea (zs:99.87, ...)"
        let (vs_ref, diff_summary) = if let Some(paren_start) = vs_part.find(" (") {
            let ref_name = vs_part[..paren_start].trim().to_string();
            let diff = vs_part[paren_start + 1..].trim().to_string();
            (Some(ref_name), Some(diff))
        } else {
            (Some(vs_part.trim().to_string()), None)
        };

        (reason, tolerance_note, vs_ref, diff_summary)
    } else {
        // No vs clause — just reason (possibly with parenthesized note)
        let (reason, tolerance_note) = extract_parenthesized(s);
        (reason, tolerance_note, None, None)
    }
}

/// Extract a parenthesized suffix from a string.
///
/// `"auto-accepted (within d:1 s:99.5)"` → `("auto-accepted", Some("within d:1 s:99.5"))`
fn extract_parenthesized(s: &str) -> (String, Option<String>) {
    if let Some(paren_start) = s.find(" (") {
        let before = s[..paren_start].trim().to_string();
        let inside = s[paren_start + 2..]
            .trim_end_matches(')')
            .trim()
            .to_string();
        if inside.is_empty() {
            (before, None)
        } else {
            (before, Some(inside))
        }
    } else {
        (s.to_string(), None)
    }
}

/// Simple whitespace tokenizer that respects the remaining text.
struct Tokenizer<'a> {
    remaining: &'a str,
}

impl<'a> Tokenizer<'a> {
    fn new(s: &'a str) -> Self {
        Self { remaining: s }
    }

    fn next_word(&mut self) -> Option<&'a str> {
        let s = self.remaining.trim_start();
        if s.is_empty() {
            return None;
        }
        let end = s.find(char::is_whitespace).unwrap_or(s.len());
        let word = &s[..end];
        self.remaining = &s[end..];
        Some(word)
    }

    fn remaining(&self) -> &'a str {
        self.remaining.trim_start()
    }
}

// ─── Formatting ──────────────────────────────────────────────────────────

/// Format a single entry line.
fn format_entry_line(entry: &ChecksumEntry) -> String {
    let prefix = entry.kind.prefix_char();
    let mut line = format!(
        "{} {}  {}  @{}  {}",
        prefix, entry.name_hash, entry.arch, entry.commit, entry.reason,
    );

    if let Some(ref note) = entry.tolerance_note {
        line.push_str(&format!(" ({note})"));
    }

    if let Some(ref vs) = entry.vs_ref {
        line.push_str(&format!(" vs {vs}"));
        if let Some(ref diff) = entry.diff_summary {
            line.push_str(&format!(" {diff}"));
        }
    }

    line
}

// ─── Manager ─────────────────────────────────────────────────────────

/// Result of checking a hash against a `.checksums` file.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CheckResult {
    /// Actual hash matches an active entry.
    #[non_exhaustive]
    Match {
        /// Memorable name of the matched entry.
        entry_name: String,
    },
    /// Hash mismatch, but zensim comparison passes tolerance.
    ///
    /// Only returned by [`ChecksumManager::check_pixels`] and
    /// [`ChecksumManager::check_file`] — hash-only
    /// [`ChecksumManager::check_hash`] cannot produce this variant
    /// because it has no pixel data to compare.
    #[non_exhaustive]
    WithinTolerance {
        /// Zensim regression report.
        report: RegressionReport,
        /// Memorable name of the authoritative baseline.
        authoritative_name: String,
        /// Memorable name of the actual hash.
        actual_name: String,
        /// Raw hash of the actual output.
        actual_hash: String,
        /// Whether the new hash was auto-accepted (UPDATE mode).
        auto_accepted: bool,
    },
    /// No baseline exists for this test/detail. May have been auto-accepted.
    #[non_exhaustive]
    NoBaseline {
        /// Memorable name of the actual hash.
        actual_name: String,
        /// Raw hash of the actual output.
        actual_hash: String,
        /// Whether the entry was automatically accepted (REPLACE or UPDATE mode).
        auto_accepted: bool,
    },
    /// Baseline exists but actual hash does not match any active entry.
    #[non_exhaustive]
    Failed {
        /// Zensim regression report, if pixel comparison was possible.
        report: Option<RegressionReport>,
        /// Memorable name of the authoritative baseline.
        authoritative_name: String,
        /// Memorable name of the actual (non-matching) hash.
        actual_name: String,
        /// Raw hash of the actual output.
        actual_hash: String,
        /// Path to the diff montage image, if one was generated.
        montage_path: Option<std::path::PathBuf>,
    },
}

impl CheckResult {
    /// Whether this result is a pass (Match, WithinTolerance, or auto-accepted NoBaseline).
    pub fn passed(&self) -> bool {
        matches!(
            self,
            Self::Match { .. }
                | Self::WithinTolerance { .. }
                | Self::NoBaseline {
                    auto_accepted: true,
                    ..
                }
        )
    }

    /// For a `Failed` or `NoBaseline` result, format the `.checksums` line
    /// that would accept this hash. Returns `None` for `Match`/`WithinTolerance`.
    pub fn suggest_accept_line(&self) -> Option<String> {
        let arch = crate::arch::detect_arch_tag();
        let commit = current_commit_short().unwrap_or_default();
        match self {
            Self::Failed {
                report,
                authoritative_name,
                actual_name,
                ..
            } => {
                let diff_summary = report.as_ref().map(|r| {
                    let delta = r.max_channel_delta();
                    format!(
                        "score={:.1} maxΔ=[{},{},{}]",
                        r.score(),
                        delta[0],
                        delta[1],
                        delta[2],
                    )
                });
                let mut line = format!(
                    "~ {actual_name}  {arch}  @{commit}  auto-accepted vs {authoritative_name}"
                );
                if let Some(ref ds) = diff_summary {
                    line.push(' ');
                    line.push_str(ds);
                }
                Some(line)
            }
            Self::NoBaseline {
                actual_name,
                auto_accepted: false,
                ..
            } => Some(format!("= {actual_name}  {arch}  @{commit}  new-baseline")),
            _ => None,
        }
    }
}

impl std::fmt::Display for CheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Match { entry_name } => write!(f, "PASS (exact match, {entry_name})"),
            Self::WithinTolerance {
                report,
                authoritative_name,
                auto_accepted,
                ..
            } => {
                let delta = report.max_channel_delta();
                write!(
                    f,
                    "PASS (within tolerance, score={:.1}, max-delta=[{},{},{}], vs {authoritative_name}{})",
                    report.score(),
                    delta[0],
                    delta[1],
                    delta[2],
                    if *auto_accepted {
                        ", auto-accepted"
                    } else {
                        ""
                    },
                )
            }
            Self::Failed {
                report,
                authoritative_name,
                actual_name,
                montage_path,
                ..
            } => {
                match report {
                    Some(r) => {
                        let delta = r.max_channel_delta();
                        write!(
                            f,
                            "FAIL (score={:.1}, maxΔ=[{},{},{}], vs {authoritative_name})",
                            r.score(),
                            delta[0],
                            delta[1],
                            delta[2],
                        )?;
                    }
                    None => write!(f, "FAIL (no reference image, vs {authoritative_name})")?,
                }
                if let Some(path) = montage_path {
                    write!(f, "\n  Montage: {}", path.display())?;
                }
                // Suggest the line to add to the .checksums file
                let arch = crate::arch::detect_arch_tag();
                let commit = current_commit_short().unwrap_or_default();
                write!(
                    f,
                    "\n  To accept, add this line to the .checksums file:\n  ~ {actual_name}  {arch}  @{commit}  auto-accepted vs {authoritative_name}"
                )?;
                if let Some(r) = report {
                    let delta = r.max_channel_delta();
                    write!(
                        f,
                        " score={:.1} maxΔ=[{},{},{}]",
                        r.score(),
                        delta[0],
                        delta[1],
                        delta[2],
                    )?;
                }
                Ok(())
            }
            Self::NoBaseline {
                actual_name,
                auto_accepted,
                ..
            } => {
                if *auto_accepted {
                    write!(f, "NO BASELINE (auto-accepted)")
                } else {
                    let arch = crate::arch::detect_arch_tag();
                    let commit = current_commit_short().unwrap_or_default();
                    write!(
                        f,
                        "NO BASELINE\n  To add baseline, add this line to the .checksums file:\n  = {actual_name}  {arch}  @{commit}  new-baseline"
                    )
                }
            }
        }
    }
}

/// Manager for `.checksums` files.
///
/// Provides the main workflow for checksum validation:
/// 1. Load the module's `.checksums` file
/// 2. Find the section for the test function + detail
/// 3. Compare the actual hash against active entries
/// 4. In UPDATE mode, auto-accept new hashes
///
/// # Environment variables
///
/// - `UPDATE_CHECKSUMS=1` — auto-accept new checksums as `AutoAccepted`
pub struct ChecksumManager {
    checksums_dir: PathBuf,
    update_mode: bool,
    hasher: Box<dyn ChecksumHasher>,
    zensim: Zensim,
    remote: Option<ReferenceStorage>,
    diff_dir: Option<PathBuf>,
    manifest: Option<Arc<ManifestWriter>>,
}

impl ChecksumManager {
    /// Create a new manager pointing at a directory of `.checksums` files.
    ///
    /// Reads `UPDATE_CHECKSUMS` from the environment.
    pub fn new(checksums_dir: &Path) -> Self {
        let update_mode = std::env::var("UPDATE_CHECKSUMS").is_ok_and(|v| v == "1" || v == "true");
        Self {
            checksums_dir: checksums_dir.to_path_buf(),
            update_mode,
            hasher: Box::new(SeaHasher),
            zensim: Zensim::new(ZensimProfile::latest()),
            remote: None,
            diff_dir: None,
            manifest: None,
        }
    }

    /// Create a manager with explicit mode flags (for testing).
    pub fn with_modes(checksums_dir: &Path, update_mode: bool) -> Self {
        Self {
            checksums_dir: checksums_dir.to_path_buf(),
            update_mode,
            hasher: Box::new(SeaHasher),
            zensim: Zensim::new(ZensimProfile::latest()),
            remote: None,
            diff_dir: None,
            manifest: None,
        }
    }

    /// Override the hasher.
    pub fn with_hasher(mut self, hasher: impl ChecksumHasher + 'static) -> Self {
        self.hasher = Box::new(hasher);
        self
    }

    /// Set remote storage for reference image upload/download.
    pub fn with_remote_storage(mut self, remote: ReferenceStorage) -> Self {
        self.remote = Some(remote);
        self
    }

    /// Configure remote storage from environment variables.
    ///
    /// Reads `REGRESS_REFERENCE_URL`, `REGRESS_UPLOAD_PREFIX`, and
    /// `UPLOAD_REFERENCES`. No-op if `REGRESS_REFERENCE_URL` is not set.
    /// Downloads are cached in `{checksums_dir}/.remote-cache`.
    pub fn with_remote_storage_from_env(mut self) -> Self {
        let cache_dir = self.checksums_dir.join(".remote-cache");
        self.remote = ReferenceStorage::from_env(cache_dir);
        self
    }

    /// Set a directory for saving comparison montages on mismatch.
    pub fn with_diff_output(mut self, dir: impl Into<PathBuf>) -> Self {
        self.diff_dir = Some(dir.into());
        self
    }

    /// Set a shared manifest writer for recording test results as TSV.
    pub fn with_manifest(mut self, writer: Arc<ManifestWriter>) -> Self {
        self.manifest = Some(writer);
        self
    }

    /// Enable manifest writing from the `REGRESS_MANIFEST_PATH` environment
    /// variable. No-op if the variable is unset or empty.
    pub fn with_manifest_from_env(mut self) -> Self {
        use std::sync::OnceLock;
        static GLOBAL: OnceLock<Option<Arc<ManifestWriter>>> = OnceLock::new();
        let writer = GLOBAL.get_or_init(|| ManifestWriter::from_env().map(Arc::new));
        if let Some(w) = writer {
            self.manifest = Some(Arc::clone(w));
        }
        self
    }

    /// Path to the `.checksums` file for a given module.
    pub fn module_path(&self, module: &str) -> PathBuf {
        self.checksums_dir.join(format!("{module}.checksums"))
    }

    /// Check whether a `.checksums` file exists for the given module.
    pub fn has_module(&self, module: &str) -> bool {
        self.module_path(module).exists()
    }

    /// Check a hash against the `.checksums` file for a given module/test/detail.
    ///
    /// The `actual_hash` should be the raw hash ID (e.g., `"sea:a4839401fabae99c.png"`).
    /// File extensions are stripped before petname lookup but preserved in the entry.
    ///
    /// Uses advisory file locking to prevent races when tests run in parallel.
    pub fn check_hash(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        actual_hash: &str,
        tolerance: Option<&crate::tolerance::ToleranceSpec>,
    ) -> Result<CheckResult, RegressError> {
        let path = self.module_path(module);

        // Acquire advisory file lock for the checksums file.
        let lock_path = path.with_extension("checksums.lock");
        let _guard = crate::lock::FileLockGuard::acquire_and_cleanup(&lock_path)?;
        self.check_hash_locked(
            module,
            &path,
            test_name,
            detail_name,
            actual_hash,
            tolerance,
        )
    }

    /// Inner implementation of `check_hash`, called while holding the file lock.
    fn check_hash_locked(
        &self,
        module: &str,
        path: &Path,
        test_name: &str,
        detail_name: &str,
        actual_hash: &str,
        tolerance: Option<&crate::tolerance::ToleranceSpec>,
    ) -> Result<CheckResult, RegressError> {
        let actual_name = hash_to_memorable(actual_hash);
        let arch = crate::arch::detect_arch_tag().to_string();
        let commit = current_commit_short().unwrap_or_default();

        // Load or create the file
        let mut file = if path.exists() {
            ChecksumsFile::read_from(path)?
        } else {
            ChecksumsFile::new(module)
        };

        // Find or create the section
        let section = file.find_section(test_name, detail_name);

        // Check if actual matches any active entry
        if let Some(section) = section {
            let matched_name = section.active_entries().find_map(|entry| {
                names_match(&entry.name_hash, &actual_name).then(|| entry.name_hash.clone())
            });
            if let Some(entry_name) = matched_name {
                // Update tolerance if provided and different
                if let Some(tol) = tolerance {
                    let section = file.find_section_mut(test_name, detail_name).unwrap();
                    if section.tolerance.as_ref() != Some(tol) {
                        section.tolerance = Some(tol.clone());
                        file.write_to(path)?;
                    }
                }
                return Ok(CheckResult::Match { entry_name });
            }

            // No match — check if there's an anchor to report as the authoritative baseline
            let anchor_name: Option<String> = section
                .anchor()
                .or_else(|| section.active_entries().next())
                .map(|e| e.name_hash.clone());

            if let Some(ref authoritative) = anchor_name {
                if self.update_mode {
                    // UPDATE: auto-accept within tolerance (caller should verify tolerance)
                    let section = file.get_or_create_section(test_name, detail_name);
                    if let Some(tol) = tolerance {
                        section.tolerance = Some(tol.clone());
                    }
                    section.prune_auto_accepted(&arch);
                    section.entries.push(ChecksumEntry {
                        kind: EntryKind::AutoAccepted,
                        name_hash: actual_name.clone(),
                        arch: arch.clone(),
                        commit,
                        reason: "auto-accepted".to_string(),
                        tolerance_note: None,
                        vs_ref: Some(authoritative.clone()),
                        diff_summary: None,
                    });
                    file.write_to(path)?;
                    return Ok(CheckResult::NoBaseline {
                        actual_name,
                        actual_hash: actual_hash.to_string(),
                        auto_accepted: true,
                    });
                }

                return Ok(CheckResult::Failed {
                    report: None,
                    authoritative_name: authoritative.clone(),
                    actual_name,
                    actual_hash: actual_hash.to_string(),
                    montage_path: None,
                });
            }
        }

        // No section or no active entries — treat as no baseline
        if self.update_mode {
            let section = file.get_or_create_section(test_name, detail_name);
            if let Some(tol) = tolerance {
                section.tolerance = Some(tol.clone());
            }
            section.entries.push(ChecksumEntry {
                kind: EntryKind::AutoAccepted,
                name_hash: actual_name.clone(),
                arch,
                commit,
                reason: "auto-accepted".to_string(),
                tolerance_note: None,
                vs_ref: None,
                diff_summary: None,
            });
            file.write_to(path)?;
            return Ok(CheckResult::NoBaseline {
                actual_name,
                actual_hash: actual_hash.to_string(),
                auto_accepted: true,
            });
        }

        Ok(CheckResult::NoBaseline {
            actual_name,
            actual_hash: actual_hash.to_string(),
            auto_accepted: false,
        })
    }

    /// Accept a new checksum entry with diff evidence.
    ///
    /// Used by callers who have already performed pixel comparison and
    /// want to record the result with chain-of-trust evidence.
    #[allow(clippy::too_many_arguments)]
    pub fn accept(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        actual_hash: &str,
        vs_ref_hash: Option<&str>,
        tolerance_note: Option<&str>,
        diff_summary: Option<&str>,
        reason: &str,
    ) -> Result<(), RegressError> {
        let path = self.module_path(module);

        // Acquire advisory file lock
        let lock_path = path.with_extension("checksums.lock");
        let _guard = crate::lock::FileLockGuard::acquire_and_cleanup(&lock_path)?;

        let actual_name = hash_to_memorable(actual_hash);
        let arch = crate::arch::detect_arch_tag().to_string();
        let commit = current_commit_short().unwrap_or_default();

        let mut file = if path.exists() {
            ChecksumsFile::read_from(&path)?
        } else {
            ChecksumsFile::new(module)
        };

        let section = file.get_or_create_section(test_name, detail_name);
        section.prune_auto_accepted(&arch);

        let vs_ref = vs_ref_hash.map(hash_to_memorable);

        section.entries.push(ChecksumEntry {
            kind: EntryKind::AutoAccepted,
            name_hash: actual_name,
            arch,
            commit,
            reason: reason.to_string(),
            tolerance_note: tolerance_note.map(|s| s.to_string()),
            vs_ref,
            diff_summary: diff_summary.map(|s| s.to_string()),
        });

        file.write_to(&path)
    }

    // ─── Full comparison workflow ────────────────────────────────────────

    /// Check actual RGBA pixels against stored checksums with full comparison.
    ///
    /// This is the high-level entry point that does the complete workflow:
    /// 1. Hash actual pixels
    /// 2. Check hash against `.checksums` file
    /// 3. On mismatch, load reference image and run zensim comparison
    /// 4. Auto-accept within tolerance (in UPDATE mode)
    /// 5. Save diff montage (if configured)
    /// 6. Write manifest entry (if configured)
    #[allow(clippy::too_many_arguments)]
    pub fn check_pixels(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        actual_rgba: &[u8],
        width: u32,
        height: u32,
        tolerance: Option<&ToleranceSpec>,
    ) -> Result<CheckResult, RegressError> {
        let actual_hash = self.hasher.hash_pixels(actual_rgba, width, height);
        let actual_pixels = rgba_bytes_to_pixels(actual_rgba);
        let actual_source = RgbaSlice::new(&actual_pixels, width as usize, height as usize);
        let result = self.check_with_source_impl(
            module,
            test_name,
            detail_name,
            &actual_hash,
            &actual_source,
            tolerance,
        )?;
        self.write_manifest_for_result(module, test_name, detail_name, &result, tolerance);
        Ok(result)
    }

    /// Check an image file against stored checksums with full comparison.
    ///
    /// Decodes the file to RGBA for hashing and comparison.
    pub fn check_file(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        actual_path: impl AsRef<Path>,
        tolerance: Option<&ToleranceSpec>,
    ) -> Result<CheckResult, RegressError> {
        let actual_path = actual_path.as_ref();
        let img = image::open(actual_path)
            .map_err(|e| RegressError::image(actual_path, e))?
            .to_rgba8();
        let (w, h) = img.dimensions();
        self.check_pixels(
            module,
            test_name,
            detail_name,
            img.as_raw(),
            w,
            h,
            tolerance,
        )
    }

    /// Check an [`ImageSource`] against stored checksums with full comparison.
    ///
    /// This is the format-aware entry point that works with any pixel layout
    /// (strided BGRA, packed RGBA, etc.). The image is hashed by converting to
    /// packed RGBA internally, but comparison uses the original `ImageSource`
    /// directly — no unnecessary copies for BGRA or strided data.
    pub fn check_image(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        actual: &impl ImageSource,
        tolerance: Option<&ToleranceSpec>,
    ) -> Result<CheckResult, RegressError> {
        // Convert to packed RGBA for hashing (format-independent hashes)
        let (rgba, w, h) = image_source_to_packed_rgba(actual);
        let actual_hash = self.hasher.hash_pixels(&rgba, w, h);
        let result = self.check_with_source_impl(
            module,
            test_name,
            detail_name,
            &actual_hash,
            actual,
            tolerance,
        )?;
        self.write_manifest_for_result(module, test_name, detail_name, &result, tolerance);
        Ok(result)
    }

    /// Check a pre-hashed [`ImageSource`] against stored checksums.
    ///
    /// Use this when the hash is computed externally (e.g., from legacy BGRA
    /// scanlines) but comparison should use the original image data.
    /// The `ImageSource` is used directly for zensim comparison — no
    /// BGRA→RGBA conversion needed for the comparison step.
    pub fn check_hash_with_image(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        actual_hash: &str,
        actual: &impl ImageSource,
        tolerance: Option<&ToleranceSpec>,
    ) -> Result<CheckResult, RegressError> {
        let result = self.check_with_source_impl(
            module,
            test_name,
            detail_name,
            actual_hash,
            actual,
            tolerance,
        )?;
        self.write_manifest_for_result(module, test_name, detail_name, &result, tolerance);
        Ok(result)
    }

    /// Core implementation for check_pixels/check_file/check_image/check_hash_with_image.
    ///
    /// Unlike `check_hash_locked`, this method only auto-accepts after
    /// pixel comparison passes tolerance (not blindly on hash mismatch).
    ///
    /// The `ImageSource` is used directly for zensim comparison (preserving
    /// stride/format), and converted to packed RGBA only for diff montage
    /// and reference image saving.
    fn check_with_source_impl(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        actual_hash: &str,
        actual: &impl ImageSource,
        tolerance: Option<&ToleranceSpec>,
    ) -> Result<CheckResult, RegressError> {
        let path = self.module_path(module);

        // Acquire advisory file lock
        let lock_path = path.with_extension("checksums.lock");
        let _guard = crate::lock::FileLockGuard::acquire_and_cleanup(&lock_path)?;

        let actual_name = hash_to_memorable(actual_hash);
        let arch = crate::arch::detect_arch_tag().to_string();
        let commit = current_commit_short().unwrap_or_default();

        // Load or create the file
        let mut file = if path.exists() {
            ChecksumsFile::read_from(&path)?
        } else {
            ChecksumsFile::new(module)
        };

        let section = file.find_section(test_name, detail_name);

        // Check for hash match
        if let Some(section) = section {
            let matched_name = section.active_entries().find_map(|entry| {
                names_match(&entry.name_hash, &actual_name).then(|| entry.name_hash.clone())
            });

            if let Some(entry_name) = matched_name {
                // Update tolerance if provided and different
                if let Some(tol) = tolerance {
                    let section = file.find_section_mut(test_name, detail_name).unwrap();
                    if section.tolerance.as_ref() != Some(tol) {
                        section.tolerance = Some(tol.clone());
                        file.write_to(&path)?;
                    }
                }
                return Ok(CheckResult::Match { entry_name });
            }

            // No match — get authoritative baseline
            let anchor_name = section
                .anchor()
                .or_else(|| section.active_entries().next())
                .map(|e| e.name_hash.clone());

            if let Some(ref authoritative) = anchor_name {
                // Hash mismatch — load reference and compare
                let ref_path =
                    self.find_reference_image(module, test_name, detail_name, Some(authoritative));
                let (ref_rgba, rw, rh) = match ref_path.and_then(|p| {
                    decode_reference_png(&p)
                        .map_err(|e| eprintln!("Warning: failed to decode reference image: {e}"))
                        .ok()
                }) {
                    Some(decoded) => decoded,
                    None => {
                        // No reference image — can't do pixel comparison
                        if self.update_mode {
                            let section = file.get_or_create_section(test_name, detail_name);
                            if let Some(tol) = tolerance {
                                section.tolerance = Some(tol.clone());
                            }
                            section.prune_auto_accepted(&arch);
                            section.entries.push(ChecksumEntry {
                                kind: EntryKind::AutoAccepted,
                                name_hash: actual_name.clone(),
                                arch,
                                commit,
                                reason: "auto-accepted (no reference image)".to_string(),
                                tolerance_note: None,
                                vs_ref: Some(authoritative.clone()),
                                diff_summary: None,
                            });
                            file.write_to(&path)?;
                            return Ok(CheckResult::NoBaseline {
                                actual_name,
                                actual_hash: actual_hash.to_string(),
                                auto_accepted: true,
                            });
                        }
                        return Ok(CheckResult::Failed {
                            report: None,
                            authoritative_name: authoritative.clone(),
                            actual_name,
                            actual_hash: actual_hash.to_string(),
                            montage_path: None,
                        });
                    }
                };

                let reg_tolerance = tolerance
                    .map(|t| t.to_regression_tolerance(&arch))
                    .unwrap_or_else(RegressionTolerance::exact);

                let ref_pixels = rgba_bytes_to_pixels(&ref_rgba);
                let ref_source = RgbaSlice::new(&ref_pixels, rw as usize, rh as usize);

                // Compare directly — zensim handles per-image pixel formats.
                // Images smaller than 8×8 can't be scored; skip perceptual comparison.
                // Convert actual to packed RGBA for montage (needed regardless of comparison outcome).
                let (actual_rgba, aw, ah) = image_source_to_packed_rgba(actual);

                let (report, comparison_error) =
                    match check_regression(&self.zensim, &ref_source, actual, &reg_tolerance) {
                        Ok(r) => (Some(r), false),
                        Err(zensim::ZensimError::ImageTooSmall) => (None, false),
                        Err(e) => {
                            // Print dimensions on any zensim error (especially dimension mismatch).
                            eprintln!(
                                "[checksum] zensim error for {test_name}/{detail_name}: {e}\n  \
                                 reference: {rw}x{rh}, actual: {aw}x{ah}",
                            );
                            (None, true) // comparison failed — don't auto-accept
                        }
                    };
                let montage_path = self.save_diff_montage(
                    module,
                    test_name,
                    detail_name,
                    &ref_rgba,
                    rw,
                    rh,
                    &actual_rgba,
                    aw,
                    ah,
                    report.as_ref(),
                    &reg_tolerance,
                );

                let passed = !comparison_error && report.as_ref().is_none_or(|r| r.passed());

                if passed {
                    let auto_accepted = self.update_mode;
                    if auto_accepted {
                        let (tol_note, diff_summary_str) = if let Some(ref r) = report {
                            (
                                tolerance
                                    .map(|t| format!("within {}", format_tolerance_shorthand(t))),
                                Some(format_diff_summary(r)),
                            )
                        } else {
                            (None, None)
                        };

                        let reason = if report.is_some() {
                            "auto-accepted within tolerance"
                        } else {
                            "auto-accepted (image too small for zensim)"
                        };

                        let section = file.get_or_create_section(test_name, detail_name);
                        if let Some(tol) = tolerance {
                            section.tolerance = Some(tol.clone());
                        }
                        section.prune_auto_accepted(&arch);
                        section.entries.push(ChecksumEntry {
                            kind: EntryKind::AutoAccepted,
                            name_hash: actual_name.clone(),
                            arch,
                            commit,
                            reason: reason.to_string(),
                            tolerance_note: tol_note,
                            vs_ref: Some(authoritative.clone()),
                            diff_summary: diff_summary_str,
                        });
                        file.write_to(&path)?;
                    }

                    if let Some(report) = report {
                        return Ok(CheckResult::WithinTolerance {
                            report,
                            authoritative_name: authoritative.clone(),
                            actual_name,
                            actual_hash: actual_hash.to_string(),
                            auto_accepted,
                        });
                    } else {
                        // Too small for zensim — treat like no-baseline with auto-accept
                        return Ok(CheckResult::NoBaseline {
                            actual_name,
                            actual_hash: actual_hash.to_string(),
                            auto_accepted,
                        });
                    }
                }

                return Ok(CheckResult::Failed {
                    report,
                    authoritative_name: authoritative.clone(),
                    actual_name,
                    actual_hash: actual_hash.to_string(),
                    montage_path,
                });
            }
        }

        // No section or no active entries — no baseline
        if self.update_mode {
            let section = file.get_or_create_section(test_name, detail_name);
            if let Some(tol) = tolerance {
                section.tolerance = Some(tol.clone());
            }
            section.entries.push(ChecksumEntry {
                kind: EntryKind::AutoAccepted,
                name_hash: actual_name.clone(),
                arch,
                commit,
                reason: "auto-accepted".to_string(),
                tolerance_note: None,
                vs_ref: None,
                diff_summary: None,
            });
            file.write_to(&path)?;

            // Save reference image (needs packed RGBA)
            let (rgba, w, h) = image_source_to_packed_rgba(actual);
            let _ = self.save_reference_image(module, test_name, detail_name, &rgba, w, h);

            return Ok(CheckResult::NoBaseline {
                actual_name,
                actual_hash: actual_hash.to_string(),
                auto_accepted: true,
            });
        }

        Ok(CheckResult::NoBaseline {
            actual_name,
            actual_hash: actual_hash.to_string(),
            auto_accepted: false,
        })
    }

    // ─── Reference image helpers ────────────────────────────────────────

    /// Try to find a reference image for a test.
    ///
    /// 1. Looks locally at `{checksums_dir}/images/{module}/{test}_{detail}.png`
    /// 2. If remote storage is configured, tries downloading by authoritative petname
    fn find_reference_image(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        authoritative_name: Option<&str>,
    ) -> Option<PathBuf> {
        let images_dir = self.checksums_dir.join("images").join(module);
        let flat_name = flat_test_name(test_name, detail_name);
        let local_path = images_dir.join(format!("{flat_name}.png"));
        if local_path.exists() {
            return Some(local_path);
        }

        // Try remote download by authoritative petname
        if let (Some(remote), Some(auth_name)) = (&self.remote, authoritative_name) {
            match remote.download_reference(auth_name) {
                Ok(Some(cached_path)) => {
                    let _ = std::fs::create_dir_all(&images_dir);
                    let _ = std::fs::copy(&cached_path, &local_path);
                    return Some(local_path);
                }
                Ok(None) => {}
                Err(e) => eprintln!("Warning: remote download failed: {e}"),
            }
        }

        None
    }

    /// Save a reference image for future comparisons.
    pub fn save_reference_image(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<PathBuf, RegressError> {
        let images_dir = self.checksums_dir.join("images").join(module);
        std::fs::create_dir_all(&images_dir).map_err(|e| RegressError::io(&images_dir, e))?;

        let flat_name = flat_test_name(test_name, detail_name);
        let path = images_dir.join(format!("{flat_name}.png"));

        let img = image::RgbaImage::from_raw(width, height, rgba.to_vec()).ok_or_else(|| {
            RegressError::Io {
                path: path.clone(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "invalid dimensions {width}x{height} for {} bytes",
                        rgba.len()
                    ),
                ),
            }
        })?;
        img.save(&path).map_err(|e| RegressError::image(&path, e))?;
        Ok(path)
    }

    /// Save a reference image only if one doesn't already exist on disk.
    ///
    /// Useful for ensuring reference images are populated even when the hash
    /// matches an existing baseline (the normal code path skips saving in that case).
    pub fn save_reference_if_missing(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        actual: &impl ImageSource,
    ) {
        let images_dir = self.checksums_dir.join("images").join(module);
        let flat_name = flat_test_name(test_name, detail_name);
        let path = images_dir.join(format!("{flat_name}.png"));
        if path.exists() {
            return;
        }
        let (rgba, w, h) = image_source_to_packed_rgba(actual);
        let _ = self.save_reference_image(module, test_name, detail_name, &rgba, w, h);
    }

    /// Upload a reference image to remote storage (if configured).
    pub fn upload_reference_image(&self, checksum_name: &str, local_path: &Path) {
        if let Some(remote) = &self.remote
            && let Err(e) = remote.upload_reference(local_path, checksum_name)
        {
            eprintln!("Warning: failed to upload reference {checksum_name}: {e}");
        }
    }

    // ─── Diff montage ───────────────────────────────────────────────────

    /// Save annotated comparison montage if diff_dir is configured.
    ///
    /// Produces a 2×2 grid (Expected | Actual | Pixel Diff | Structural Diff)
    /// with colored constraint text and spatial heatmap. Tiny images are
    /// pixelate-upscaled so panels are always large enough to inspect.
    ///
    /// Returns the path to the saved montage, if successful.
    #[allow(clippy::too_many_arguments)]
    fn save_diff_montage(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        ref_rgba: &[u8],
        rw: u32,
        rh: u32,
        actual_rgba: &[u8],
        aw: u32,
        ah: u32,
        report: Option<&RegressionReport>,
        tolerance: &RegressionTolerance,
    ) -> Option<std::path::PathBuf> {
        let dir = self.diff_dir.as_ref()?;
        let diff_dir = dir.join(module);
        if let Err(e) = std::fs::create_dir_all(&diff_dir) {
            eprintln!(
                "Warning: failed to create diff dir {}: {e}",
                diff_dir.display()
            );
            return None;
        }
        let flat_name = flat_test_name(test_name, detail_name);
        let out_path = diff_dir.join(format!("{flat_name}.png"));

        if rw != aw || rh != ah {
            // Dimensions differ — resize smaller to match larger for visual comparison.
            let target_w = rw.max(aw);
            let target_h = rh.max(ah);
            let ref_img = image::RgbaImage::from_raw(rw, rh, ref_rgba.to_vec())
                .expect("ref: invalid dimensions");
            let act_img = image::RgbaImage::from_raw(aw, ah, actual_rgba.to_vec())
                .expect("actual: invalid dimensions");
            let ref_resized = image::imageops::resize(
                &ref_img,
                target_w,
                target_h,
                image::imageops::FilterType::Lanczos3,
            );
            let act_resized = image::imageops::resize(
                &act_img,
                target_w,
                target_h,
                image::imageops::FilterType::Lanczos3,
            );

            let dim_title = format!(
                "{} {} (ref {}x{}, actual {}x{})",
                test_name, detail_name, rw, rh, aw, ah
            );
            let dim_annotation = AnnotationText::empty().with_title(dim_title);
            let montage =
                MontageOptions::default().render(&ref_resized, &act_resized, &dim_annotation);
            let _ = montage.save(&out_path);
            return Some(out_path);
        }

        let title = format!("{} {}", test_name, detail_name).trim().to_string();
        let annotation = match report {
            Some(r) => AnnotationText::from_report(r, tolerance).with_title(title),
            None => AnnotationText::empty().with_title(title),
        };

        let ref_img =
            image::RgbaImage::from_raw(rw, rh, ref_rgba.to_vec()).expect("ref: invalid dimensions");
        let act_img = image::RgbaImage::from_raw(aw, ah, actual_rgba.to_vec())
            .expect("actual: invalid dimensions");
        let montage = MontageOptions::default().render(&ref_img, &act_img, &annotation);
        match montage.save(&out_path) {
            Ok(()) => Some(out_path),
            Err(e) => {
                eprintln!("Warning: failed to save diff image for {flat_name}: {e}");
                None
            }
        }
    }

    // ─── Manifest ───────────────────────────────────────────────────────

    /// Write a manifest entry for a check result (if manifest is configured).
    fn write_manifest_for_result(
        &self,
        module: &str,
        test_name: &str,
        detail_name: &str,
        result: &CheckResult,
        tolerance: Option<&ToleranceSpec>,
    ) {
        let Some(manifest) = &self.manifest else {
            return;
        };

        let flat_name = flat_test_name(test_name, detail_name);

        let (status, actual_hash, baseline_hash, actual_zdsim, diff_summary) = match result {
            CheckResult::Match { entry_name } => (
                ManifestStatus::Match,
                entry_name.as_str(),
                Some(entry_name.as_str()),
                Some(0.0),
                None,
            ),
            CheckResult::WithinTolerance {
                report,
                authoritative_name,
                actual_hash,
                ..
            } => {
                let zd = zensim::score_to_dissimilarity(report.score());
                let summary = format!("score:{:.1}", report.score());
                (
                    ManifestStatus::Accepted,
                    actual_hash.as_str(),
                    Some(authoritative_name.as_str()),
                    Some(zd),
                    Some(summary),
                )
            }
            CheckResult::Failed {
                actual_hash,
                authoritative_name,
                report,
                ..
            } => {
                let zd = report
                    .as_ref()
                    .map(|r| zensim::score_to_dissimilarity(r.score()));
                let summary = report.as_ref().map(|r| format!("score:{:.1}", r.score()));
                (
                    ManifestStatus::Failed,
                    actual_hash.as_str(),
                    Some(authoritative_name.as_str()),
                    zd,
                    summary,
                )
            }
            CheckResult::NoBaseline { actual_hash, .. } => (
                ManifestStatus::Novel,
                actual_hash.as_str(),
                None,
                None,
                None,
            ),
        };

        manifest.write_entry(&ManifestEntry {
            test_name: &format!("{module}/{flat_name}"),
            status,
            actual_hash,
            baseline_hash,
            actual_zdsim,
            tolerance_zdsim: tolerance.map(|t| zensim::score_to_dissimilarity(t.min_similarity)),
            diff_summary: diff_summary.as_deref(),
        });
    }
}

/// Flatten test_name + detail_name into a single identifier.
fn flat_test_name(test_name: &str, detail_name: &str) -> String {
    if detail_name.is_empty() {
        test_name.to_string()
    } else {
        format!("{test_name}_{detail_name}")
    }
}

/// Decode a PNG reference image to RGBA8 pixels.
fn decode_reference_png(path: &Path) -> Result<(Vec<u8>, u32, u32), RegressError> {
    let img = image::open(path)
        .map_err(|e| RegressError::image(path, e))?
        .to_rgba8();
    let (w, h) = img.dimensions();
    Ok((img.into_raw(), w, h))
}

/// Convert `&[u8]` to `Vec<[u8; 4]>` (RGBA pixels).
fn rgba_bytes_to_pixels(bytes: &[u8]) -> Vec<[u8; 4]> {
    assert!(
        bytes.len().is_multiple_of(4),
        "RGBA byte slice length {} is not a multiple of 4",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

/// Convert any [`ImageSource`] to packed RGBA8 bytes.
///
/// Handles stride (non-contiguous rows) and BGRA→RGBA channel swapping.
/// Used for diff montage saving and reference image saving, which require
/// flat packed RGBA.
fn image_source_to_packed_rgba(src: &dyn ImageSource) -> (Vec<u8>, u32, u32) {
    let w = src.width();
    let h = src.height();
    let mut rgba = Vec::with_capacity(w * h * 4);

    for y in 0..h {
        let row = src.row_bytes(y);
        match src.pixel_format() {
            PixelFormat::Srgb8Rgba => {
                rgba.extend_from_slice(&row[..w * 4]);
            }
            PixelFormat::Srgb8Bgra => {
                for pixel in row[..w * 4].chunks_exact(4) {
                    rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
                }
            }
            PixelFormat::Srgb8Rgb => {
                for pixel in row[..w * 3].chunks_exact(3) {
                    rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
                }
            }
            other => panic!("image_source_to_packed_rgba: unsupported pixel format {other:?}"),
        }
    }
    (rgba, w as u32, h as u32)
}

/// Convert a raw hash ID to a memorable name, stripping any file extension.
///
/// Delegates to [`crate::petname::try_memorable_name`].
fn hash_to_memorable(hash_id: &str) -> String {
    crate::petname::try_memorable_name(hash_id)
}

/// Strip the file extension from a hash ID.
///
/// Delegates to [`crate::petname::strip_hash_extension`].
#[cfg(test)]
fn strip_hash_extension(hash_id: &str) -> &str {
    crate::petname::strip_hash_extension(hash_id)
}

/// Compare two memorable names by their hex prefix (ignoring words).
///
/// Handles both old 5-char and new 10-char hex petnames by comparing
/// the shorter prefix length. This allows old `.checksums` entries to
/// match new petnames during the transition.
fn names_match(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let a_parts = crate::petname::parse_memorable_name(a);
    let b_parts = crate::petname::parse_memorable_name(b);
    match (a_parts, b_parts) {
        (Some(a), Some(b)) => {
            if a.algo != b.algo {
                return false;
            }
            let min_len = a.hex.len().min(b.hex.len());
            a.hex[..min_len] == b.hex[..min_len]
        }
        _ => false,
    }
}

/// Try to get the current git commit hash (short form).
fn current_commit_short() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Parse/format round-trip ────────────────────────────────────────

    #[test]
    fn parse_minimal_file() {
        let content = "\
# test.checksums — v1

## test_fill_rect eeccff_hermite_400x400
tolerance d:0 s:100
= jolly-pike-a4839:sea  x86_64-avx512  @773c807  human-verified
";

        let file = ChecksumsFile::parse(content);
        // Header comment only (trailing blank lines stripped to prevent accumulation)
        assert_eq!(file.header_comments.len(), 1);
        assert_eq!(file.sections.len(), 1);

        let section = &file.sections[0];
        assert_eq!(section.test_name, "test_fill_rect");
        assert_eq!(section.detail_name, "eeccff_hermite_400x400");
        assert!(section.tolerance.is_some());
        assert_eq!(section.entries.len(), 1);

        let entry = &section.entries[0];
        assert_eq!(entry.kind, EntryKind::HumanVerified);
        assert_eq!(entry.name_hash, "jolly-pike-a4839:sea");
        assert_eq!(entry.arch, "x86_64-avx512");
        assert_eq!(entry.commit, "773c807");
        assert_eq!(entry.reason, "human-verified");
    }

    #[test]
    fn parse_auto_accepted_with_vs() {
        let content = "\
## test_fill_rect blue_on_transparent
= calm-bear-c5d6e:sea  x86_64-avx512  @576ef37  human-verified
~ tidy-frog-b2c3d:sea  aarch64  @576ef37  auto-accepted (within d:1 s:99.5) vs calm-bear-c5d6e:sea (zs:99.87, maxΔ:[1,1,0], cat:rounding, balanced)
";

        let file = ChecksumsFile::parse(content);
        let section = &file.sections[0];
        assert_eq!(section.entries.len(), 2);

        let auto = &section.entries[1];
        assert_eq!(auto.kind, EntryKind::AutoAccepted);
        assert_eq!(auto.name_hash, "tidy-frog-b2c3d:sea");
        assert_eq!(auto.arch, "aarch64");
        assert_eq!(auto.tolerance_note.as_deref(), Some("within d:1 s:99.5"));
        assert_eq!(auto.vs_ref.as_deref(), Some("calm-bear-c5d6e:sea"));
        assert!(auto.diff_summary.is_some());
        let diff = auto.diff_summary.as_deref().unwrap();
        assert!(diff.contains("zs:99.87"), "diff={diff}");
        assert!(diff.contains("balanced"), "diff={diff}");
    }

    #[test]
    fn parse_retired_entry() {
        let content = "\
## test_foo bar
x old-hash-12345:sea  x86_64-avx512  @abc123  superseded
= new-hash-67890:sea  x86_64-avx512  @def456  human-verified
";

        let file = ChecksumsFile::parse(content);
        let section = &file.sections[0];
        assert_eq!(section.entries.len(), 2);
        assert_eq!(section.entries[0].kind, EntryKind::Retired);
        assert_eq!(section.entries[1].kind, EntryKind::HumanVerified);
    }

    #[test]
    fn parse_multiple_sections() {
        let content = "\
# test.checksums — v1

## test_a variant_1
tolerance d:0 s:100
= foo-bar-11111:sea  x86_64-avx512  @aaa  human-verified

## test_b variant_2
tolerance d:1 s:95
= baz-qux-22222:sea  aarch64  @bbb  human-verified
";

        let file = ChecksumsFile::parse(content);
        assert_eq!(file.sections.len(), 2);
        assert_eq!(file.sections[0].test_name, "test_a");
        assert_eq!(file.sections[1].test_name, "test_b");
    }

    #[test]
    fn roundtrip_format() {
        let mut file = ChecksumsFile::new("test");
        let section = file.get_or_create_section("test_foo", "bar_baz");
        section.tolerance = Some(crate::tolerance::ToleranceSpec::exact());
        section.entries.push(ChecksumEntry::human_verified(
            "sunny-crab-a4839:sea".to_string(),
            "x86_64-avx512".to_string(),
            "abc1234".to_string(),
        ));
        section.entries.push(ChecksumEntry::auto_accepted(
            "tidy-frog-b2c3d:sea".to_string(),
            "aarch64".to_string(),
            "abc1234".to_string(),
            "within d:1 s:99.5".to_string(),
            "sunny-crab-a4839:sea".to_string(),
            "(zs:99.87, cat:rounding, balanced)".to_string(),
        ));

        let text = file.format();
        let parsed = ChecksumsFile::parse(&text);

        assert_eq!(parsed.sections.len(), 1);
        let s = &parsed.sections[0];
        assert_eq!(s.test_name, "test_foo");
        assert_eq!(s.detail_name, "bar_baz");
        assert_eq!(s.entries.len(), 2);
        assert_eq!(s.entries[0].kind, EntryKind::HumanVerified);
        assert_eq!(s.entries[0].name_hash, "sunny-crab-a4839:sea");
        assert_eq!(s.entries[1].kind, EntryKind::AutoAccepted);
        assert_eq!(s.entries[1].vs_ref.as_deref(), Some("sunny-crab-a4839:sea"));
    }

    #[test]
    fn file_io_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.checksums");

        let mut file = ChecksumsFile::new("test");
        let section = file.get_or_create_section("test_io", "roundtrip");
        section.entries.push(ChecksumEntry::human_verified(
            "able-ace-00000:sea".to_string(),
            "x86_64-avx2".to_string(),
            "deadbeef".to_string(),
        ));

        file.write_to(&path).unwrap();
        let loaded = ChecksumsFile::read_from(&path).unwrap();

        assert_eq!(loaded.sections.len(), 1);
        assert_eq!(
            loaded.sections[0].entries[0].name_hash,
            "able-ace-00000:sea"
        );
    }

    // ─── Section operations ─────────────────────────────────────────────

    #[test]
    fn find_section() {
        let mut file = ChecksumsFile::new("test");
        file.get_or_create_section("test_a", "v1");
        file.get_or_create_section("test_b", "v2");

        assert!(file.find_section("test_a", "v1").is_some());
        assert!(file.find_section("test_b", "v2").is_some());
        assert!(file.find_section("test_c", "v3").is_none());
    }

    #[test]
    fn prune_auto_accepted() {
        let mut file = ChecksumsFile::new("test");
        let section = file.get_or_create_section("test_a", "v1");
        section.entries.push(ChecksumEntry::human_verified(
            "anchor-00000:sea".to_string(),
            "x86_64".to_string(),
            "aaa".to_string(),
        ));
        section.entries.push(ChecksumEntry::auto_accepted(
            "old-auto-11111:sea".to_string(),
            "aarch64".to_string(),
            "bbb".to_string(),
            "within d:1 s:99".to_string(),
            "anchor-00000:sea".to_string(),
            "(zs:99.5)".to_string(),
        ));
        section.entries.push(ChecksumEntry::auto_accepted(
            "new-auto-22222:sea".to_string(),
            "aarch64".to_string(),
            "ccc".to_string(),
            "within d:1 s:99".to_string(),
            "anchor-00000:sea".to_string(),
            "(zs:99.8)".to_string(),
        ));

        assert_eq!(section.entries.len(), 3);
        let removed = section.prune_auto_accepted("aarch64");
        assert_eq!(removed, 1);
        assert_eq!(section.entries.len(), 2);
        // The remaining auto-accepted should be the newer one
        let auto = section
            .entries
            .iter()
            .find(|e| e.kind == EntryKind::AutoAccepted)
            .unwrap();
        assert_eq!(auto.name_hash, "new-auto-22222:sea");
    }

    #[test]
    fn retire_anchor() {
        let mut file = ChecksumsFile::new("test");
        let section = file.get_or_create_section("test_a", "v1");
        section.entries.push(ChecksumEntry::human_verified(
            "old-anchor-00000:sea".to_string(),
            "x86_64".to_string(),
            "aaa".to_string(),
        ));
        section.entries.push(ChecksumEntry::auto_accepted(
            "auto-11111:sea".to_string(),
            "aarch64".to_string(),
            "bbb".to_string(),
            "within d:1 s:99".to_string(),
            "old-anchor-00000:sea".to_string(),
            "(zs:99.5)".to_string(),
        ));

        section.retire_anchor();

        // Old anchor should be retired
        let old = section
            .entries
            .iter()
            .find(|e| e.name_hash == "old-anchor-00000:sea")
            .unwrap();
        assert_eq!(old.kind, EntryKind::Retired);

        // Auto-accepted referencing old anchor should be pruned
        assert!(
            section
                .entries
                .iter()
                .all(|e| e.name_hash != "auto-11111:sea")
        );
    }

    // ─── Entry constructors ─────────────────────────────────────────────

    #[test]
    fn entry_kind_prefix_roundtrip() {
        for kind in [
            EntryKind::HumanVerified,
            EntryKind::AutoAccepted,
            EntryKind::Retired,
        ] {
            let c = kind.prefix_char();
            let parsed = EntryKind::from_char(c).unwrap();
            assert_eq!(parsed, kind);
        }
    }

    // ─── Hash helpers ───────────────────────────────────────────────────

    #[test]
    fn strip_hash_extension_png() {
        assert_eq!(
            strip_hash_extension("sea:a4839401fabae99c.png"),
            "sea:a4839401fabae99c"
        );
    }

    #[test]
    fn strip_hash_extension_none() {
        assert_eq!(
            strip_hash_extension("sea:a4839401fabae99c"),
            "sea:a4839401fabae99c"
        );
    }

    #[test]
    fn strip_hash_extension_jpg() {
        assert_eq!(
            strip_hash_extension("sea:a4839401fabae99c.jpg"),
            "sea:a4839401fabae99c"
        );
    }

    #[test]
    fn hash_to_memorable_with_extension() {
        let name = hash_to_memorable("sea:a4839401fabae99c.png");
        // Should produce same name as without extension
        let name2 = hash_to_memorable("sea:a4839401fabae99c");
        assert_eq!(name, name2);
        assert!(name.ends_with(":sea"), "name={name}");
    }

    #[test]
    fn names_match_identical() {
        assert!(names_match("sunny-crab-a4839:sea", "sunny-crab-a4839:sea"));
    }

    #[test]
    fn names_match_different() {
        assert!(!names_match("sunny-crab-a4839:sea", "tidy-frog-b2c3d:sea"));
    }

    // ─── ChecksumManager ─────────────────────────────────────────────

    #[test]
    fn manager_no_baseline_no_auto() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::with_modes(dir.path(), false);

        let result = mgr
            .check_hash(
                "trim",
                "test_trim",
                "transparent",
                "sea:a4839401fabae99c",
                None,
            )
            .unwrap();
        match result {
            CheckResult::NoBaseline {
                auto_accepted: false,
                ..
            } => {}
            other => panic!("expected NoBaseline(auto_accepted=false), got {other:?}"),
        }

        // No file should be created
        assert!(!mgr.module_path("trim").exists());
    }

    #[test]
    fn manager_no_baseline_update_mode() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::with_modes(dir.path(), true);

        let result = mgr
            .check_hash(
                "trim",
                "test_trim",
                "transparent",
                "sea:a4839401fabae99c",
                None,
            )
            .unwrap();
        match result {
            CheckResult::NoBaseline {
                auto_accepted: true,
                ..
            } => {}
            other => panic!("expected NoBaseline(auto_accepted=true), got {other:?}"),
        }

        // File should be created with the entry
        assert!(mgr.module_path("trim").exists());
        let file = ChecksumsFile::read_from(&mgr.module_path("trim")).unwrap();
        assert_eq!(file.sections.len(), 1);
        assert_eq!(file.sections[0].entries.len(), 1);
        assert_eq!(file.sections[0].entries[0].kind, EntryKind::AutoAccepted);
    }

    #[test]
    fn manager_match_existing() {
        let dir = tempfile::tempdir().unwrap();
        let hash = "sea:a4839401fabae99c";
        let name = hash_to_memorable(hash);

        // Pre-populate a checksums file
        let mut file = ChecksumsFile::new("trim");
        let section = file.get_or_create_section("test_trim", "transparent");
        section.entries.push(ChecksumEntry::human_verified(
            name.clone(),
            "x86_64-avx2".to_string(),
            "abc1234".to_string(),
        ));
        file.write_to(&dir.path().join("trim.checksums")).unwrap();

        let mgr = ChecksumManager::with_modes(dir.path(), false);
        let result = mgr
            .check_hash("trim", "test_trim", "transparent", hash, None)
            .unwrap();

        match result {
            CheckResult::Match { entry_name } => {
                assert_eq!(entry_name, name);
            }
            other => panic!("expected Match, got {other:?}"),
        }
    }

    #[test]
    fn manager_match_with_extension() {
        let dir = tempfile::tempdir().unwrap();
        let bare_hash = "sea:a4839401fabae99c";
        let ext_hash = "sea:a4839401fabae99c.png";
        let name = hash_to_memorable(bare_hash);

        let mut file = ChecksumsFile::new("trim");
        let section = file.get_or_create_section("test_trim", "transparent");
        section.entries.push(ChecksumEntry::human_verified(
            name.clone(),
            "x86_64-avx2".to_string(),
            "abc1234".to_string(),
        ));
        file.write_to(&dir.path().join("trim.checksums")).unwrap();

        let mgr = ChecksumManager::with_modes(dir.path(), false);
        // Check with extension — should still match
        let result = mgr
            .check_hash("trim", "test_trim", "transparent", ext_hash, None)
            .unwrap();

        match result {
            CheckResult::Match { .. } => {}
            other => panic!("expected Match for hash with extension, got {other:?}"),
        }
    }

    #[test]
    fn manager_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let stored_hash = "sea:a4839401fabae99c";
        let actual_hash = "sea:1111111111111111";
        let stored_name = hash_to_memorable(stored_hash);

        let mut file = ChecksumsFile::new("trim");
        let section = file.get_or_create_section("test_trim", "transparent");
        section.entries.push(ChecksumEntry::human_verified(
            stored_name.clone(),
            "x86_64-avx2".to_string(),
            "abc1234".to_string(),
        ));
        file.write_to(&dir.path().join("trim.checksums")).unwrap();

        let mgr = ChecksumManager::with_modes(dir.path(), false);
        let result = mgr
            .check_hash("trim", "test_trim", "transparent", actual_hash, None)
            .unwrap();

        match result {
            CheckResult::Failed {
                authoritative_name,
                actual_name,
                ..
            } => {
                assert_eq!(authoritative_name, stored_name);
                assert_ne!(actual_name, stored_name);
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn manager_mismatch_update_mode_auto_accepts() {
        let dir = tempfile::tempdir().unwrap();
        let stored_hash = "sea:a4839401fabae99c";
        let actual_hash = "sea:1111111111111111";
        let stored_name = hash_to_memorable(stored_hash);

        let mut file = ChecksumsFile::new("trim");
        let section = file.get_or_create_section("test_trim", "transparent");
        section.entries.push(ChecksumEntry::human_verified(
            stored_name.clone(),
            "x86_64-avx2".to_string(),
            "abc1234".to_string(),
        ));
        file.write_to(&dir.path().join("trim.checksums")).unwrap();

        let mgr = ChecksumManager::with_modes(dir.path(), true);
        let result = mgr
            .check_hash("trim", "test_trim", "transparent", actual_hash, None)
            .unwrap();

        match result {
            CheckResult::NoBaseline {
                auto_accepted: true,
                ..
            } => {}
            other => panic!("expected NoBaseline(auto_accepted=true), got {other:?}"),
        }

        // Verify the entry was written
        let file = ChecksumsFile::read_from(&dir.path().join("trim.checksums")).unwrap();
        let section = file.find_section("test_trim", "transparent").unwrap();
        assert_eq!(section.entries.len(), 2); // original + auto-accepted
        assert_eq!(section.entries[1].kind, EntryKind::AutoAccepted);
        assert_eq!(
            section.entries[1].vs_ref.as_deref(),
            Some(stored_name.as_str())
        );
    }

    #[test]
    fn manager_accept_with_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let stored_hash = "sea:a4839401fabae99c";
        let actual_hash = "sea:1111111111111111";
        let stored_name = hash_to_memorable(stored_hash);

        let mut file = ChecksumsFile::new("trim");
        let section = file.get_or_create_section("test_trim", "transparent");
        section.entries.push(ChecksumEntry::human_verified(
            stored_name.clone(),
            "x86_64-avx2".to_string(),
            "abc1234".to_string(),
        ));
        file.write_to(&dir.path().join("trim.checksums")).unwrap();

        let mgr = ChecksumManager::with_modes(dir.path(), false);
        mgr.accept(
            "trim",
            "test_trim",
            "transparent",
            actual_hash,
            Some(stored_hash),
            Some("within d:1 s:99.5"),
            Some("(zs:99.87, cat:rounding)"),
            "auto-accepted within tolerance",
        )
        .unwrap();

        let file = ChecksumsFile::read_from(&dir.path().join("trim.checksums")).unwrap();
        let section = file.find_section("test_trim", "transparent").unwrap();
        assert_eq!(section.entries.len(), 2);

        let entry = &section.entries[1];
        assert_eq!(entry.kind, EntryKind::AutoAccepted);
        assert_eq!(entry.tolerance_note.as_deref(), Some("within d:1 s:99.5"));
        assert_eq!(
            entry.diff_summary.as_deref(),
            Some("(zs:99.87, cat:rounding)")
        );
        assert_eq!(entry.vs_ref.as_deref(), Some(stored_name.as_str()));
    }
}
