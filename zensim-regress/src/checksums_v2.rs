//! `.checksums` v1 format: line-oriented checksum log files.
//!
//! Each test source file gets a sibling `.checksums` file containing sections
//! for each test function. Entries track human-verified baselines, auto-accepted
//! variants, and retired superseded hashes.
//!
//! # Format overview
//!
//! ```text
//! # trim.checksums — v1
//!
//! ## test_trim_whitespace transparent_shirt
//! tolerance identical
//! = sunny-crab-a4839:sea  x86_64-avx512  @773c807  human-verified
//! ~ tidy-frog-b2c3d:sea   aarch64        @773c807  auto-accepted (within off-by-one) vs sunny-crab-a4839:sea (zs:99.87, ...)
//! ```
//!
//! See the plan document for full format specification.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::RegressError;

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
    pub tolerance: Option<crate::checksum_file::ToleranceSpec>,
    /// Checksum entries in file order.
    pub entries: Vec<ChecksumEntry2>,
}

impl TestSection {
    /// The human-verified anchor entry (first `=` entry, or most recent).
    pub fn anchor(&self) -> Option<&ChecksumEntry2> {
        self.entries
            .iter()
            .find(|e| e.kind == EntryKind::HumanVerified)
    }

    /// All active entries (human-verified + auto-accepted, not retired).
    pub fn active_entries(&self) -> impl Iterator<Item = &ChecksumEntry2> {
        self.entries.iter().filter(|e| e.kind != EntryKind::Retired)
    }

    /// Find an entry by its memorable name+hash.
    pub fn find_by_name_hash(&self, name_hash: &str) -> Option<&ChecksumEntry2> {
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
pub struct ChecksumEntry2 {
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

impl ChecksumEntry2 {
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
fn parse_entry_line(line: &str) -> Option<ChecksumEntry2> {
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

    Some(ChecksumEntry2 {
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
fn format_entry_line(entry: &ChecksumEntry2) -> String {
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

// ─── Migration from TOML ─────────────────────────────────────────────────

/// Group key for TOML-to-v2 migration: maps test names to their module file.
///
/// Given a checksum directory full of TOML files, groups them by the test
/// module they likely belong to (based on name prefix before the first `_`
/// detail separator, or a provided mapping).
pub fn migrate_toml_dir(
    checksum_dir: &Path,
    module_mapping: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, ChecksumsFile>, RegressError> {
    use crate::checksum_file::TestChecksumFile;
    use crate::petname::memorable_name;

    let mut files: BTreeMap<String, ChecksumsFile> = BTreeMap::new();

    let entries = std::fs::read_dir(checksum_dir).map_err(|e| RegressError::io(checksum_dir, e))?;

    for entry in entries {
        let entry = entry.map_err(|e| RegressError::io(checksum_dir, e))?;
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }

        let toml_file = TestChecksumFile::read_from(&path)?;

        // Determine which module this test belongs to
        let module_name = module_mapping
            .get(&toml_file.name)
            .cloned()
            .unwrap_or_else(|| {
                // Default: use the test name as-is for section header
                "tests".to_string()
            });

        let checksums_file = files
            .entry(module_name.clone())
            .or_insert_with(|| ChecksumsFile::new(&module_name));

        // Split test name into test_name and detail_name
        // Convention: the TOML `name` field is the full qualified name
        let (test_name, detail_name) = split_test_detail(&toml_file.name);

        let section = checksums_file.get_or_create_section(&test_name, &detail_name);

        // Set tolerance
        section.tolerance = Some(toml_file.tolerance.clone());

        // Convert checksum entries
        for cs in &toml_file.checksum {
            // Legacy hashes are raw hex without algo prefix (no ':' separator).
            // Modern hashes are "algo:hex" format (e.g., "sea:a4839401fabae99c").
            let name_hash = if cs.id.contains(':') {
                memorable_name(&cs.id)
            } else {
                // Legacy: use truncated hex as the name_hash directly
                if cs.id.len() > 10 {
                    format!("legacy-{}", &cs.id[..10].to_lowercase())
                } else {
                    format!("legacy-{}", cs.id.to_lowercase())
                }
            };

            let kind = if cs.confidence > 0 && cs.status.as_deref() != Some("wrong") {
                EntryKind::HumanVerified
            } else {
                EntryKind::Retired
            };

            let mut entry = ChecksumEntry2 {
                kind,
                name_hash,
                arch: cs.arch.first().cloned().unwrap_or_default(),
                commit: cs.commit.clone().unwrap_or_default(),
                reason: cs.reason.clone().unwrap_or_else(|| {
                    if kind == EntryKind::HumanVerified {
                        "migrated".to_string()
                    } else {
                        "superseded".to_string()
                    }
                }),
                tolerance_note: None,
                vs_ref: None,
                diff_summary: None,
            };

            // Convert diff evidence
            if let Some(ref diff) = cs.diff {
                entry.vs_ref = Some(if diff.vs.contains(':') {
                    memorable_name(&diff.vs)
                } else if diff.vs.len() > 10 {
                    format!("legacy-{}", &diff.vs[..10].to_lowercase())
                } else {
                    format!("legacy-{}", diff.vs.to_lowercase())
                });
                // Build a simple diff summary from the stored data
                let mut parts = Vec::new();
                parts.push(format!("zs:{:.2}", diff.zensim_score));
                parts.push(format!(
                    "zd:{:.4}",
                    crate::diff_summary::zdsim(diff.zensim_score)
                ));
                if let Some(mcd) = diff.max_channel_delta
                    && mcd != [0, 0, 0]
                {
                    parts.push(format!("max\u{0394}:[{},{},{}]", mcd[0], mcd[1], mcd[2]));
                }
                if let Some(pct) = diff.pixels_differing_pct
                    && pct > 0.0
                {
                    parts.push(format!("{pct:.1}% px diff"));
                }
                parts.push(format!("cat:{}", diff.category.to_lowercase()));
                if let Some(balanced) = diff.rounding_bias_balanced {
                    parts.push(if balanced {
                        "balanced".to_string()
                    } else {
                        "biased".to_string()
                    });
                }
                entry.diff_summary = Some(format!("({})", parts.join(", ")));

                // If this entry had a diff, it was auto-accepted, not human-verified
                if kind == EntryKind::HumanVerified {
                    entry.kind = EntryKind::AutoAccepted;
                    entry.reason = "auto-accepted".to_string();
                }
            }

            section.entries.push(entry);
        }
    }

    Ok(files)
}

/// Split a test name into (test_name, detail_name) using a simple heuristic.
///
/// First tries space separator (most common): `"test_name detail_name"`.
/// Then tries slash separator (loop tests): `"test_name/Variant"`.
/// Otherwise, treat the whole string as the test name with empty detail.
fn split_test_detail(full_name: &str) -> (String, String) {
    if let Some((test, detail)) = full_name.split_once(' ') {
        (test.to_string(), detail.to_string())
    } else if let Some((test, detail)) = full_name.split_once('/') {
        (test.to_string(), detail.to_string())
    } else {
        (full_name.to_string(), String::new())
    }
}

// ─── Manager ─────────────────────────────────────────────────────────

/// Result of checking a hash against a `.checksums` file.
#[derive(Debug, Clone)]
pub enum CheckResultV2 {
    /// Actual hash matches an active entry.
    Match {
        /// Memorable name of the matched entry.
        entry_name: String,
    },
    /// No baseline exists for this test/detail. May have been auto-accepted.
    NoBaseline {
        /// Memorable name of the actual hash.
        actual_name: String,
        /// Whether the entry was automatically accepted (REPLACE or UPDATE mode).
        auto_accepted: bool,
    },
    /// Baseline exists but actual hash does not match any active entry.
    Failed {
        /// Memorable name of the authoritative baseline.
        authoritative_name: String,
        /// Memorable name of the actual (non-matching) hash.
        actual_name: String,
    },
}

/// Manager for `.checksums` v1 files.
///
/// Provides the main workflow for v1 checksum validation:
/// 1. Load the module's `.checksums` file
/// 2. Find the section for the test function + detail
/// 3. Compare the actual hash against active entries
/// 4. In UPDATE/REPLACE mode, auto-accept new hashes
///
/// # Environment variables
///
/// - `UPDATE_CHECKSUMS=1` — auto-accept new checksums as `AutoAccepted`
/// - `REPLACE_CHECKSUMS=1` — accept new checksums as `HumanVerified` (replaces baseline)
pub struct ChecksumManagerV2 {
    checksums_dir: PathBuf,
    update_mode: bool,
    replace_mode: bool,
}

impl ChecksumManagerV2 {
    /// Create a new manager pointing at a directory of `.checksums` files.
    ///
    /// Reads `UPDATE_CHECKSUMS` and `REPLACE_CHECKSUMS` from the environment.
    pub fn new(checksums_dir: &Path) -> Self {
        let update_mode = std::env::var("UPDATE_CHECKSUMS").is_ok_and(|v| v == "1" || v == "true");
        let replace_mode =
            std::env::var("REPLACE_CHECKSUMS").is_ok_and(|v| v == "1" || v == "true");
        Self {
            checksums_dir: checksums_dir.to_path_buf(),
            update_mode,
            replace_mode,
        }
    }

    /// Create a manager with explicit mode flags (for testing).
    pub fn with_modes(checksums_dir: &Path, update_mode: bool, replace_mode: bool) -> Self {
        Self {
            checksums_dir: checksums_dir.to_path_buf(),
            update_mode,
            replace_mode,
        }
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
        tolerance: Option<&crate::checksum_file::ToleranceSpec>,
    ) -> Result<CheckResultV2, RegressError> {
        let path = self.module_path(module);

        // Acquire advisory file lock for the checksums file.
        let lock_path = path.with_extension("checksums.lock");
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .map_err(|e| RegressError::io(&lock_path, e))?;
        use fs2::FileExt;
        lock_file
            .lock_exclusive()
            .map_err(|e| RegressError::io(&lock_path, e))?;
        let result = self.check_hash_locked(
            module,
            &path,
            test_name,
            detail_name,
            actual_hash,
            tolerance,
        );
        let _ = lock_file.unlock();
        let _ = std::fs::remove_file(&lock_path);
        result
    }

    /// Inner implementation of `check_hash`, called while holding the file lock.
    fn check_hash_locked(
        &self,
        module: &str,
        path: &Path,
        test_name: &str,
        detail_name: &str,
        actual_hash: &str,
        tolerance: Option<&crate::checksum_file::ToleranceSpec>,
    ) -> Result<CheckResultV2, RegressError> {
        let actual_name = hash_to_memorable(actual_hash);
        let arch = crate::arch::detect_arch_tag().to_string();
        let commit = current_commit_short().unwrap_or_default();

        // Load or create the file
        let mut file = if path.exists() {
            ChecksumsFile::read_from(&path)?
        } else {
            ChecksumsFile::new(module)
        };

        // Find or create the section
        let section = file.find_section(test_name, detail_name);

        // Check if actual matches any active entry
        // In replace mode, require exact match so old petnames get replaced.
        if let Some(section) = section {
            let matched_name = section.active_entries().find_map(|entry| {
                let is_match = if self.replace_mode {
                    entry.name_hash == actual_name
                } else {
                    names_match(&entry.name_hash, &actual_name)
                };
                is_match.then(|| entry.name_hash.clone())
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
                return Ok(CheckResultV2::Match { entry_name });
            }

            // No match — check if there's an anchor to report as the authoritative baseline
            let anchor_name: Option<String> = section
                .anchor()
                .or_else(|| section.active_entries().next())
                .map(|e| e.name_hash.clone());

            if let Some(ref authoritative) = anchor_name {
                // Mismatch with existing baseline
                if self.replace_mode {
                    // REPLACE: accept as new human-verified anchor
                    let section = file.get_or_create_section(test_name, detail_name);
                    if let Some(tol) = tolerance {
                        section.tolerance = Some(tol.clone());
                    }
                    section.retire_anchor();
                    section.entries.push(ChecksumEntry2::human_verified(
                        actual_name.clone(),
                        arch,
                        commit,
                    ));
                    file.write_to(&path)?;
                    return Ok(CheckResultV2::NoBaseline {
                        actual_name,
                        auto_accepted: true,
                    });
                }

                if self.update_mode {
                    // UPDATE: auto-accept within tolerance (caller should verify tolerance)
                    let section = file.get_or_create_section(test_name, detail_name);
                    if let Some(tol) = tolerance {
                        section.tolerance = Some(tol.clone());
                    }
                    section.prune_auto_accepted(&arch);
                    section.entries.push(ChecksumEntry2 {
                        kind: EntryKind::AutoAccepted,
                        name_hash: actual_name.clone(),
                        arch: arch.clone(),
                        commit,
                        reason: "auto-accepted".to_string(),
                        tolerance_note: None,
                        vs_ref: Some(authoritative.clone()),
                        diff_summary: None,
                    });
                    file.write_to(&path)?;
                    return Ok(CheckResultV2::NoBaseline {
                        actual_name,
                        auto_accepted: true,
                    });
                }

                return Ok(CheckResultV2::Failed {
                    authoritative_name: authoritative.clone(),
                    actual_name,
                });
            }
        }

        // No section or no active entries — treat as no baseline
        if self.replace_mode || self.update_mode {
            let section = file.get_or_create_section(test_name, detail_name);
            if let Some(tol) = tolerance {
                section.tolerance = Some(tol.clone());
            }
            let kind = if self.replace_mode {
                EntryKind::HumanVerified
            } else {
                EntryKind::AutoAccepted
            };
            section.entries.push(ChecksumEntry2 {
                kind,
                name_hash: actual_name.clone(),
                arch,
                commit,
                reason: if self.replace_mode {
                    "new-baseline".to_string()
                } else {
                    "auto-accepted".to_string()
                },
                tolerance_note: None,
                vs_ref: None,
                diff_summary: None,
            });
            file.write_to(&path)?;
            return Ok(CheckResultV2::NoBaseline {
                actual_name,
                auto_accepted: true,
            });
        }

        Ok(CheckResultV2::NoBaseline {
            actual_name,
            auto_accepted: false,
        })
    }

    /// Accept a new checksum entry with diff evidence.
    ///
    /// Used by callers who have already performed pixel comparison and
    /// want to record the result with chain-of-trust evidence.
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
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .map_err(|e| RegressError::io(&lock_path, e))?;
        use fs2::FileExt;
        lock_file
            .lock_exclusive()
            .map_err(|e| RegressError::io(&lock_path, e))?;

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

        section.entries.push(ChecksumEntry2 {
            kind: EntryKind::AutoAccepted,
            name_hash: actual_name,
            arch,
            commit,
            reason: reason.to_string(),
            tolerance_note: tolerance_note.map(|s| s.to_string()),
            vs_ref,
            diff_summary: diff_summary.map(|s| s.to_string()),
        });

        let result = file.write_to(&path);
        let _ = lock_file.unlock();
        let _ = std::fs::remove_file(&lock_path);
        result
    }
}

/// Convert a raw hash ID to a memorable name, stripping any file extension.
///
/// `"sea:a4839401fabae99c.png"` → `"sunny-crab-a4839:sea"`
fn hash_to_memorable(hash_id: &str) -> String {
    // Strip file extension if present (e.g., ".png", ".jpg")
    let bare = strip_hash_extension(hash_id);
    crate::petname::memorable_name(bare)
}

/// Strip the file extension from a hash ID.
///
/// `"sea:a4839401fabae99c.png"` → `"sea:a4839401fabae99c"`
fn strip_hash_extension(hash_id: &str) -> &str {
    // Only strip known image extensions at the end
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".unknown"] {
        if let Some(stripped) = hash_id.strip_suffix(ext) {
            return stripped;
        }
    }
    hash_id
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
        // Header comment + blank line before first section
        assert_eq!(file.header_comments.len(), 2);
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
        section.tolerance = Some(crate::checksum_file::ToleranceSpec::exact());
        section.entries.push(ChecksumEntry2::human_verified(
            "sunny-crab-a4839:sea".to_string(),
            "x86_64-avx512".to_string(),
            "abc1234".to_string(),
        ));
        section.entries.push(ChecksumEntry2::auto_accepted(
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
        section.entries.push(ChecksumEntry2::human_verified(
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
        section.entries.push(ChecksumEntry2::human_verified(
            "anchor-00000:sea".to_string(),
            "x86_64".to_string(),
            "aaa".to_string(),
        ));
        section.entries.push(ChecksumEntry2::auto_accepted(
            "old-auto-11111:sea".to_string(),
            "aarch64".to_string(),
            "bbb".to_string(),
            "within d:1 s:99".to_string(),
            "anchor-00000:sea".to_string(),
            "(zs:99.5)".to_string(),
        ));
        section.entries.push(ChecksumEntry2::auto_accepted(
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
        section.entries.push(ChecksumEntry2::human_verified(
            "old-anchor-00000:sea".to_string(),
            "x86_64".to_string(),
            "aaa".to_string(),
        ));
        section.entries.push(ChecksumEntry2::auto_accepted(
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

    #[test]
    fn split_test_detail_with_space() {
        let (t, d) = split_test_detail("test_trim_whitespace transparent_shirt");
        assert_eq!(t, "test_trim_whitespace");
        assert_eq!(d, "transparent_shirt");
    }

    #[test]
    fn split_test_detail_no_space() {
        let (t, d) = split_test_detail("test_simple");
        assert_eq!(t, "test_simple");
        assert_eq!(d, "");
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

    // ─── ChecksumManagerV2 ─────────────────────────────────────────────

    #[test]
    fn manager_no_baseline_no_auto() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManagerV2::with_modes(dir.path(), false, false);

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
            CheckResultV2::NoBaseline {
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
        let mgr = ChecksumManagerV2::with_modes(dir.path(), true, false);

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
            CheckResultV2::NoBaseline {
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
    fn manager_no_baseline_replace_mode() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManagerV2::with_modes(dir.path(), false, true);

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
            CheckResultV2::NoBaseline {
                auto_accepted: true,
                ..
            } => {}
            other => panic!("expected NoBaseline(auto_accepted=true), got {other:?}"),
        }

        let file = ChecksumsFile::read_from(&mgr.module_path("trim")).unwrap();
        assert_eq!(file.sections[0].entries[0].kind, EntryKind::HumanVerified);
    }

    #[test]
    fn manager_match_existing() {
        let dir = tempfile::tempdir().unwrap();
        let hash = "sea:a4839401fabae99c";
        let name = hash_to_memorable(hash);

        // Pre-populate a checksums file
        let mut file = ChecksumsFile::new("trim");
        let section = file.get_or_create_section("test_trim", "transparent");
        section.entries.push(ChecksumEntry2::human_verified(
            name.clone(),
            "x86_64-avx2".to_string(),
            "abc1234".to_string(),
        ));
        file.write_to(&dir.path().join("trim.checksums")).unwrap();

        let mgr = ChecksumManagerV2::with_modes(dir.path(), false, false);
        let result = mgr
            .check_hash("trim", "test_trim", "transparent", hash, None)
            .unwrap();

        match result {
            CheckResultV2::Match { entry_name } => {
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
        section.entries.push(ChecksumEntry2::human_verified(
            name.clone(),
            "x86_64-avx2".to_string(),
            "abc1234".to_string(),
        ));
        file.write_to(&dir.path().join("trim.checksums")).unwrap();

        let mgr = ChecksumManagerV2::with_modes(dir.path(), false, false);
        // Check with extension — should still match
        let result = mgr
            .check_hash("trim", "test_trim", "transparent", ext_hash, None)
            .unwrap();

        match result {
            CheckResultV2::Match { .. } => {}
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
        section.entries.push(ChecksumEntry2::human_verified(
            stored_name.clone(),
            "x86_64-avx2".to_string(),
            "abc1234".to_string(),
        ));
        file.write_to(&dir.path().join("trim.checksums")).unwrap();

        let mgr = ChecksumManagerV2::with_modes(dir.path(), false, false);
        let result = mgr
            .check_hash("trim", "test_trim", "transparent", actual_hash, None)
            .unwrap();

        match result {
            CheckResultV2::Failed {
                authoritative_name,
                actual_name,
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
        section.entries.push(ChecksumEntry2::human_verified(
            stored_name.clone(),
            "x86_64-avx2".to_string(),
            "abc1234".to_string(),
        ));
        file.write_to(&dir.path().join("trim.checksums")).unwrap();

        let mgr = ChecksumManagerV2::with_modes(dir.path(), true, false);
        let result = mgr
            .check_hash("trim", "test_trim", "transparent", actual_hash, None)
            .unwrap();

        match result {
            CheckResultV2::NoBaseline {
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
        section.entries.push(ChecksumEntry2::human_verified(
            stored_name.clone(),
            "x86_64-avx2".to_string(),
            "abc1234".to_string(),
        ));
        file.write_to(&dir.path().join("trim.checksums")).unwrap();

        let mgr = ChecksumManagerV2::with_modes(dir.path(), false, false);
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
