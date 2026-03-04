//! Test manifest writer for instant CI debugging.
//!
//! Writes a TSV file during test execution, one line per checksum check.
//! The manifest is uploaded as a CI artifact alongside output images,
//! letting you instantly see which tests matched, which were novel, which
//! were accepted within tolerance, and which failed — with hashes, petnames,
//! filenames, and scores all in one place.
//!
//! # Multi-process safety (nextest)
//!
//! The manifest is safe for concurrent writes from multiple processes
//! (as with `cargo-nextest`, which runs each test in a separate process)
//! and from multiple threads within each process. The file is opened in
//! append mode and each line is written as a single `write_all` call
//! under an advisory file lock (`fs2`).
//!
//! # Enabling
//!
//! Set the `REGRESS_MANIFEST_PATH` environment variable to a file path:
//!
//! ```bash
//! REGRESS_MANIFEST_PATH=test-manifest.tsv cargo test
//! ```
//!
//! Or configure programmatically via [`ChecksumManager::with_manifest`].
//!
//! # Format
//!
//! Tab-separated, one line per test. Header is prefixed with `#` for easy
//! `grep -v '^#'` filtering:
//!
//! ```text
//! # test_name\tstatus\tactual_hash\tactual_petname\tactual_file\tbaseline_hash\tbaseline_petname\tbaseline_file\tdiff_summary
//! ```
//!
//! [`ChecksumManager::with_manifest`]: crate::manager::ChecksumManager::with_manifest

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::lock::FileLockGuard;
use crate::petname::try_memorable_name;
use crate::remote::ReferenceStorage;

/// Environment variable to enable manifest writing (single lock-based file).
pub const MANIFEST_ENV_VAR: &str = "REGRESS_MANIFEST_PATH";

/// Environment variable to enable directory-based manifest writing (nextest-friendly).
///
/// When set, each process writes its own timestamped file to this directory.
/// After all tests complete, call [`combine_manifest_dir`] to produce a
/// unified `test-manifest.tsv`.
pub const MANIFEST_DIR_ENV_VAR: &str = "REGRESS_MANIFEST_DIR";

/// TSV header line (with trailing newline).
const HEADER: &str = "# test_name\tstatus\tactual_zdsim\ttolerance_zdsim\tactual_hash\tactual_petname\tactual_file\tbaseline_hash\tbaseline_petname\tbaseline_file\tdiff_summary\n";

/// Status recorded in the manifest for each test check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestStatus {
    /// Exact hash match against an active baseline entry.
    Match,
    /// No baseline existed; new test (first run).
    Novel,
    /// Hash mismatch, but zensim comparison passed tolerance.
    Accepted,
    /// Hash mismatch exceeding tolerance, or no comparison possible.
    Failed,
}

impl ManifestStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Novel => "novel",
            Self::Accepted => "accepted",
            Self::Failed => "failed",
        }
    }
}

/// One row in the test manifest.
pub struct ManifestEntry<'a> {
    pub test_name: &'a str,
    pub status: ManifestStatus,
    /// Measured zdsim (0.0 = identical, higher = worse). `None` if not computed.
    pub actual_zdsim: Option<f64>,
    /// Tolerance zdsim threshold. `None` if not applicable.
    pub tolerance_zdsim: Option<f64>,
    pub actual_hash: &'a str,
    pub baseline_hash: Option<&'a str>,
    pub diff_summary: Option<&'a str>,
}

/// Multi-process-safe TSV manifest writer.
///
/// Each line is written as a single `write_all` call under an advisory
/// file lock, so concurrent nextest processes and parallel threads within
/// each process can safely interleave entries without corruption.
///
/// The file is opened in append mode — multiple `ManifestWriter` instances
/// (in different processes) append to the same file without truncation.
pub struct ManifestWriter {
    /// The data file, opened in append mode. The Mutex serializes
    /// in-process threads; the advisory lock file serializes processes.
    inner: Mutex<std::fs::File>,
    /// Sibling `.lock` file for cross-process advisory locking.
    lock_path: PathBuf,
    path: PathBuf,
}

impl ManifestWriter {
    /// Open (or create) a manifest file for multi-process append.
    ///
    /// The header is written only if the file is newly created (empty).
    /// Safe for concurrent calls from multiple processes — uses advisory
    /// file locking around the empty-check + header-write.
    pub fn open(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let lock_path = manifest_lock_path(&path);

        // Acquire cross-process lock for the header-if-empty check.
        let _guard = FileLockGuard::acquire(&lock_path).map_err(|e| match e {
            crate::error::RegressError::Io { source, .. } => source,
            other => std::io::Error::other(other.to_string()),
        })?;

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        // Write header only if the file is empty (first opener wins).
        if file.metadata()?.len() == 0 {
            (&file).write_all(HEADER.as_bytes())?;
        }

        drop(_guard);

        Ok(Self {
            inner: Mutex::new(file),
            lock_path,
            path,
        })
    }

    /// Create a new manifest writer, truncating any existing file.
    ///
    /// Use this only when you own the file exclusively (e.g., in a
    /// single-process test). For multi-process use, prefer [`open`].
    ///
    /// [`open`]: ManifestWriter::open
    pub fn create(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        // Truncate + write header, then reopen in append mode.
        std::fs::write(&path, HEADER)?;

        let lock_path = manifest_lock_path(&path);
        let file = std::fs::OpenOptions::new().append(true).open(&path)?;

        Ok(Self {
            inner: Mutex::new(file),
            lock_path,
            path,
        })
    }

    /// Create from the `REGRESS_MANIFEST_PATH` environment variable.
    ///
    /// Uses [`open`] (append mode) so multiple processes can share the file.
    /// Returns `None` if the variable is unset or empty.
    ///
    /// [`open`]: ManifestWriter::open
    pub fn from_env() -> Option<Self> {
        let path = std::env::var(MANIFEST_ENV_VAR).ok()?;
        if path.is_empty() {
            return None;
        }
        match Self::open(&path) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("Warning: failed to open manifest at {path}: {e}");
                None
            }
        }
    }

    /// The path this manifest is being written to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a result line to the manifest.
    ///
    /// The entire line is formatted into memory first, then written as a
    /// single `write_all` call under an advisory file lock. This ensures
    /// no partial lines even with concurrent writers from other processes.
    ///
    /// Silently drops errors (manifest is best-effort diagnostic output).
    pub fn write_entry(&self, entry: &ManifestEntry<'_>) {
        let line = format_entry_line(entry);

        // Advisory lock → Mutex → write_all → drop guard.
        // The advisory lock serializes across processes (nextest);
        // the Mutex serializes across threads within this process.
        let _guard = FileLockGuard::try_acquire(&self.lock_path);

        if let Ok(f) = self.inner.lock() {
            let _ = (&*f).write_all(line.as_bytes());
        }

        drop(_guard);
    }
}

/// Format a manifest entry as a TSV line (with trailing newline).
///
/// Shared between [`ManifestWriter`] and [`ManifestDir`].
fn format_entry_line(entry: &ManifestEntry<'_>) -> String {
    let actual_petname = try_memorable_name(entry.actual_hash);
    let actual_file = ReferenceStorage::remote_filename(entry.actual_hash);

    let (bl_hash, bl_petname, bl_file) = match entry.baseline_hash {
        Some(h) => {
            let pet = try_memorable_name(h);
            let file = ReferenceStorage::remote_filename(h);
            (h.to_string(), pet, file)
        }
        None => ("-".to_string(), "-".to_string(), "-".to_string()),
    };

    let diff = entry.diff_summary.unwrap_or("-");

    let actual_zdsim = match entry.actual_zdsim {
        Some(0.0) => "0".to_string(),
        Some(z) if z < 0.001 => format!("{z:.6}"),
        Some(z) if z < 0.01 => format!("{z:.4}"),
        Some(z) => format!("{z:.4}"),
        None => "-".to_string(),
    };
    let tolerance_zdsim = match entry.tolerance_zdsim {
        Some(0.0) => "0".to_string(),
        Some(z) => format!("{z:.4}"),
        None => "-".to_string(),
    };

    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        entry.test_name,
        entry.status.as_str(),
        actual_zdsim,
        tolerance_zdsim,
        entry.actual_hash,
        actual_petname,
        actual_file,
        bl_hash,
        bl_petname,
        bl_file,
        diff,
    )
}

/// Compute the lock file path for a manifest path.
fn manifest_lock_path(manifest_path: &Path) -> PathBuf {
    let mut lock = manifest_path.as_os_str().to_owned();
    lock.push(".lock");
    PathBuf::from(lock)
}

// ─── Directory-based manifest (nextest-friendly) ───────────────────────

/// Lock-free directory-based manifest writer for nextest.
///
/// Each process writes entries to its own timestamped file in the manifest
/// directory. No advisory locking needed — each process owns its file.
///
/// After all tests complete, call [`combine_manifest_dir`] to read all
/// per-process files and produce a single combined `test-manifest.tsv`.
///
/// # File naming
///
/// `{unix_nanos}_{pid}.tsv` — timestamp ensures ordering, PID ensures
/// uniqueness if two processes start in the same nanosecond.
///
/// # Entries
///
/// Each file contains only data lines (no header). The combine step
/// adds the header to the output.
pub struct ManifestDir {
    /// Per-process append file. The Mutex serializes in-process threads.
    file: Mutex<std::fs::File>,
    /// Path of this process's file within the directory.
    file_path: PathBuf,
    /// The directory containing all per-process files.
    dir: PathBuf,
}

impl ManifestDir {
    /// Create a manifest directory writer.
    ///
    /// Creates the directory and opens a per-process file for append.
    /// The filename includes a nanosecond timestamp and PID for uniqueness.
    pub fn open(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;

        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let pid = std::process::id();
        let file_path = dir.join(format!("{nanos}_{pid}.tsv"));

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;

        Ok(Self {
            file: Mutex::new(file),
            file_path,
            dir,
        })
    }

    /// Create from the `REGRESS_MANIFEST_DIR` environment variable.
    ///
    /// Returns `None` if the variable is unset or empty.
    pub fn from_env() -> Option<Self> {
        let dir = std::env::var(MANIFEST_DIR_ENV_VAR).ok()?;
        if dir.is_empty() {
            return None;
        }
        match Self::open(&dir) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("Warning: failed to open manifest dir at {dir}: {e}");
                None
            }
        }
    }

    /// The directory containing all per-process files.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Write a result entry, appending to the per-process file immediately.
    ///
    /// Each entry is written to disk right away (no buffering) because
    /// static globals may not get their `Drop` called on process exit.
    /// Since there's no cross-process locking, the cost is just a file
    /// append — cheap compared to the test itself.
    pub fn write_entry(&self, entry: &ManifestEntry<'_>) {
        let line = format_entry_line(entry);
        if let Ok(mut f) = self.file.lock() {
            let _ = f.write_all(line.as_bytes());
        }
    }

    /// The path of this process's manifest file.
    pub fn path(&self) -> &Path {
        &self.file_path
    }
}

/// Combine all per-process manifest files in a directory into a single TSV.
///
/// Reads every `*.tsv` file in `dir`, concatenates them (deduplicating by
/// test name — latest timestamp wins), and writes the result to `output`.
///
/// Returns the number of entries written.
pub fn combine_manifest_dir(dir: &Path, output: &Path) -> std::io::Result<usize> {
    use std::collections::BTreeMap;

    let mut entries_by_name: BTreeMap<String, String> = BTreeMap::new();

    // Read all .tsv files, sorted by name (timestamp order)
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "tsv"))
        .collect();
    files.sort_by_key(|e| e.file_name());

    for file_entry in &files {
        let content = std::fs::read_to_string(file_entry.path())?;
        for line in content.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            // First field is test_name
            if let Some(name) = line.split('\t').next() {
                // Latest file wins (files sorted by timestamp)
                entries_by_name.insert(name.to_string(), line.to_string());
            }
        }
    }

    // Write combined output
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let mut out = std::fs::File::create(output)?;
    out.write_all(HEADER.as_bytes())?;
    for line in entries_by_name.values() {
        writeln!(out, "{line}")?;
    }

    Ok(entries_by_name.len())
}

/// Create a manifest writer from environment variables.
///
/// Checks `REGRESS_MANIFEST_DIR` first (preferred for nextest),
/// then falls back to `REGRESS_MANIFEST_PATH` (lock-based single file).
///
/// Returns a trait object that can write entries.
pub fn writer_from_env() -> Option<Box<dyn ManifestWrite + Send + Sync>> {
    if let Some(dir_writer) = ManifestDir::from_env() {
        return Some(Box::new(dir_writer));
    }
    if let Some(file_writer) = ManifestWriter::from_env() {
        return Some(Box::new(file_writer));
    }
    None
}

/// Trait for writing manifest entries.
///
/// Implemented by both [`ManifestWriter`] (lock-based) and [`ManifestDir`]
/// (lock-free per-process files).
pub trait ManifestWrite {
    /// Append a result entry to the manifest.
    fn write_entry(&self, entry: &ManifestEntry<'_>);
}

impl ManifestWrite for ManifestWriter {
    fn write_entry(&self, entry: &ManifestEntry<'_>) {
        ManifestWriter::write_entry(self, entry);
    }
}

impl ManifestWrite for ManifestDir {
    fn write_entry(&self, entry: &ManifestEntry<'_>) {
        ManifestDir::write_entry(self, entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_status_strings() {
        assert_eq!(ManifestStatus::Match.as_str(), "match");
        assert_eq!(ManifestStatus::Novel.as_str(), "novel");
        assert_eq!(ManifestStatus::Accepted.as_str(), "accepted");
        assert_eq!(ManifestStatus::Failed.as_str(), "failed");
    }

    #[test]
    fn manifest_write_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test-manifest.tsv");

        let writer = ManifestWriter::create(&path).unwrap();

        writer.write_entry(&ManifestEntry {
            test_name: "resize_bicubic",
            status: ManifestStatus::Match,
            actual_hash: "sea:a4839401fabae99c",
            baseline_hash: Some("sea:a4839401fabae99c"),
            actual_zdsim: Some(0.0),
            tolerance_zdsim: Some(0.01),
            diff_summary: None,
        });

        writer.write_entry(&ManifestEntry {
            test_name: "jpeg_quality_50",
            status: ManifestStatus::Accepted,
            actual_hash: "sea:1234567890abcdef",
            baseline_hash: Some("sea:fedcba0987654321"),
            actual_zdsim: Some(0.036),
            tolerance_zdsim: Some(0.05),
            diff_summary: Some("score:96.4"),
        });

        writer.write_entry(&ManifestEntry {
            test_name: "first_run_test",
            status: ManifestStatus::Novel,
            actual_hash: "sea:0000111122223333",
            baseline_hash: None,
            actual_zdsim: None,
            tolerance_zdsim: None,
            diff_summary: None,
        });

        drop(writer); // flush

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Header + 3 data lines
        assert_eq!(lines.len(), 4, "content:\n{content}");
        assert!(lines[0].starts_with("# test_name\t"));

        // Check first data line (fields shifted by 2 for zdsim columns)
        let fields: Vec<&str> = lines[1].split('\t').collect();
        assert_eq!(fields[0], "resize_bicubic");
        assert_eq!(fields[1], "match");
        assert_eq!(fields[2], "0"); // actual_zdsim
        assert_eq!(fields[3], "0.0100"); // tolerance_zdsim
        assert_eq!(fields[4], "sea:a4839401fabae99c");
        // petname should be deterministic
        assert!(fields[5].contains(":sea"), "petname: {}", fields[5]);
        assert_eq!(fields[6], "sea_a4839401fabae99c.png");
        // baseline same as actual for match
        assert_eq!(fields[7], "sea:a4839401fabae99c");
        assert_eq!(fields[10], "-"); // no diff summary

        // Check novel line has "-" for baseline
        let novel_fields: Vec<&str> = lines[3].split('\t').collect();
        assert_eq!(novel_fields[0], "first_run_test");
        assert_eq!(novel_fields[1], "novel");
        assert_eq!(novel_fields[2], "-"); // no zdsim
        assert_eq!(novel_fields[3], "-"); // no tolerance
        assert_eq!(novel_fields[7], "-"); // baseline hash
        assert_eq!(novel_fields[8], "-");
        assert_eq!(novel_fields[9], "-");
    }

    #[test]
    fn manifest_thread_safety() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("threaded-manifest.tsv");
        let writer = std::sync::Arc::new(ManifestWriter::create(&path).unwrap());

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let w = writer.clone();
                std::thread::spawn(move || {
                    for j in 0..10 {
                        w.write_entry(&ManifestEntry {
                            test_name: &format!("thread_{i}_test_{j}"),
                            status: ManifestStatus::Match,
                            actual_hash: "sea:a4839401fabae99c",
                            baseline_hash: Some("sea:a4839401fabae99c"),
                            actual_zdsim: Some(0.0),
                            tolerance_zdsim: None,
                            diff_summary: None,
                        });
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        drop(writer);
        let content = std::fs::read_to_string(&path).unwrap();
        let data_lines = content.lines().filter(|l| !l.starts_with('#')).count();
        assert_eq!(data_lines, 80); // 8 threads × 10 entries
    }

    #[test]
    fn open_multiple_times_single_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi-open.tsv");

        // Simulate multiple processes opening the same manifest.
        let w1 = ManifestWriter::open(&path).unwrap();
        let w2 = ManifestWriter::open(&path).unwrap();
        let w3 = ManifestWriter::open(&path).unwrap();

        w1.write_entry(&ManifestEntry {
            test_name: "from_w1",
            status: ManifestStatus::Match,
            actual_hash: "sea:a4839401fabae99c",
            baseline_hash: Some("sea:a4839401fabae99c"),
            actual_zdsim: Some(0.0),
            tolerance_zdsim: None,
            diff_summary: None,
        });
        w2.write_entry(&ManifestEntry {
            test_name: "from_w2",
            status: ManifestStatus::Novel,
            actual_hash: "sea:1111222233334444",
            baseline_hash: None,
            actual_zdsim: None,
            tolerance_zdsim: None,
            diff_summary: None,
        });
        w3.write_entry(&ManifestEntry {
            test_name: "from_w3",
            status: ManifestStatus::Failed,
            actual_hash: "sea:aaaabbbbccccdddd",
            baseline_hash: Some("sea:eeeeffff00001111"),
            actual_zdsim: Some(0.58),
            tolerance_zdsim: Some(0.05),
            diff_summary: Some("score:42.0"),
        });

        drop(w1);
        drop(w2);
        drop(w3);

        let content = std::fs::read_to_string(&path).unwrap();
        let header_count = content.lines().filter(|l| l.starts_with('#')).count();
        let data_lines: Vec<&str> = content.lines().filter(|l| !l.starts_with('#')).collect();

        assert_eq!(header_count, 1, "should have exactly one header\n{content}");
        assert_eq!(data_lines.len(), 3, "should have 3 data lines\n{content}");

        // All three writers contributed
        let test_names: Vec<&str> = data_lines
            .iter()
            .map(|l| l.split('\t').next().unwrap())
            .collect();
        assert!(test_names.contains(&"from_w1"));
        assert!(test_names.contains(&"from_w2"));
        assert!(test_names.contains(&"from_w3"));
    }

    #[test]
    fn multiprocess_simulation() {
        // Simulate nextest: multiple processes each open() the same file
        // and write concurrently from separate threads.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nextest-sim.tsv");

        let handles: Vec<_> = (0..4)
            .map(|proc_id| {
                let p = path.clone();
                std::thread::spawn(move || {
                    // Each "process" opens its own ManifestWriter
                    let w = ManifestWriter::open(&p).unwrap();
                    for i in 0..20 {
                        w.write_entry(&ManifestEntry {
                            test_name: &format!("proc{proc_id}_test{i}"),
                            status: ManifestStatus::Match,
                            actual_hash: "sea:a4839401fabae99c",
                            baseline_hash: Some("sea:a4839401fabae99c"),
                            actual_zdsim: Some(0.0),
                            tolerance_zdsim: None,
                            diff_summary: None,
                        });
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let content = std::fs::read_to_string(&path).unwrap();

        // Exactly one header
        let header_count = content.lines().filter(|l| l.starts_with('#')).count();
        assert_eq!(header_count, 1, "headers:\n{content}");

        // 4 × 20 = 80 data lines
        let data_count = content.lines().filter(|l| !l.starts_with('#')).count();
        assert_eq!(data_count, 80, "data lines:\n{content}");

        // Every line has exactly 11 tab-separated fields
        for (i, line) in content.lines().enumerate() {
            if line.starts_with('#') {
                continue;
            }
            let field_count = line.split('\t').count();
            assert_eq!(
                field_count, 11,
                "line {i} has {field_count} fields (expected 11): {line}"
            );
        }
    }

    #[test]
    fn manifest_dir_write_and_combine() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_dir = dir.path().join("manifests");
        let combined = dir.path().join("combined.tsv");

        let writer = ManifestDir::open(&manifest_dir).unwrap();
        writer.write_entry(&ManifestEntry {
            test_name: "test_alpha",
            status: ManifestStatus::Match,
            actual_hash: "sea:aaaa111122223333",
            baseline_hash: Some("sea:aaaa111122223333"),
            actual_zdsim: Some(0.0),
            tolerance_zdsim: Some(0.01),
            diff_summary: None,
        });
        writer.write_entry(&ManifestEntry {
            test_name: "test_beta",
            status: ManifestStatus::Accepted,
            actual_hash: "sea:bbbb444455556666",
            baseline_hash: Some("sea:cccc777788889999"),
            actual_zdsim: Some(0.003),
            tolerance_zdsim: Some(0.05),
            diff_summary: Some("(zs:99.7, zd:0.003)"),
        });

        // Entries written immediately — file should exist
        assert!(writer.path().exists());
        assert!(writer.path().extension().unwrap() == "tsv");

        // Combine
        let count = combine_manifest_dir(&manifest_dir, &combined).unwrap();
        assert_eq!(count, 2);

        let content = std::fs::read_to_string(&combined).unwrap();
        assert!(content.starts_with("# test_name\t"));
        let data: Vec<&str> = content.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(data.len(), 2);
        assert!(data.iter().any(|l| l.starts_with("test_alpha\t")));
        assert!(data.iter().any(|l| l.starts_with("test_beta\t")));
    }

    #[test]
    fn manifest_dir_multi_process_combine() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_dir = dir.path().join("manifests");
        let combined = dir.path().join("combined.tsv");

        // Simulate multiple nextest processes — each opens its own file
        for i in 0..5 {
            let w = ManifestDir::open(&manifest_dir).unwrap();
            w.write_entry(&ManifestEntry {
                test_name: &format!("proc{i}_test"),
                status: ManifestStatus::Match,
                actual_hash: "sea:aaaa111122223333",
                baseline_hash: Some("sea:aaaa111122223333"),
                actual_zdsim: Some(0.0),
                tolerance_zdsim: None,
                diff_summary: None,
            });
            // Each writer creates its own file (entries written immediately)
        }

        // Combine all
        let count = combine_manifest_dir(&manifest_dir, &combined).unwrap();
        assert_eq!(count, 5, "expected 5 unique entries");

        let content = std::fs::read_to_string(&combined).unwrap();
        let data: Vec<&str> = content.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(data.len(), 5);
    }

    #[test]
    fn manifest_dir_latest_wins_on_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_dir = dir.path().join("manifests");
        let combined = dir.path().join("combined.tsv");
        std::fs::create_dir_all(&manifest_dir).unwrap();

        // Write first file with "failed" status
        std::fs::write(
            manifest_dir.join("0000000001_1.tsv"),
            "my_test\tfailed\t0.1000\t0.0500\tsea:aaa\t-\t-\tsea:bbb\t-\t-\t-\n",
        )
        .unwrap();

        // Write second file with "accepted" status (later timestamp)
        std::fs::write(
            manifest_dir.join("0000000002_1.tsv"),
            "my_test\taccepted\t0.0300\t0.0500\tsea:ccc\t-\t-\tsea:bbb\t-\t-\t-\n",
        )
        .unwrap();

        let count = combine_manifest_dir(&manifest_dir, &combined).unwrap();
        assert_eq!(count, 1, "duplicate test_name should deduplicate");

        let content = std::fs::read_to_string(&combined).unwrap();
        let data: Vec<&str> = content.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(data.len(), 1);
        // Latest wins → "accepted"
        assert!(
            data[0].contains("accepted"),
            "latest should win: {}",
            data[0]
        );
    }

    #[test]
    fn manifest_dir_writes_immediately() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_dir = dir.path().join("manifests");

        let w = ManifestDir::open(&manifest_dir).unwrap();
        w.write_entry(&ManifestEntry {
            test_name: "immediate_test",
            status: ManifestStatus::Novel,
            actual_hash: "sea:dddd000011112222",
            baseline_hash: None,
            actual_zdsim: None,
            tolerance_zdsim: None,
            diff_summary: None,
        });

        // Should be on disk immediately (no need to drop)
        let content = std::fs::read_to_string(w.path()).unwrap();
        assert!(content.contains("immediate_test"));
    }
}
