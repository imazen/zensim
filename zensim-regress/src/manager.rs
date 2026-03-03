//! Checksum manager: the bridge between test harnesses and TOML persistence.
//!
//! [`ChecksumManager`] provides the main workflow for visual regression tests:
//!
//! 1. **Check** actual output against stored checksums (direct hash match → pass)
//! 2. If no match, run zensim comparison against the authoritative reference
//! 3. **Accept** new checksums with chain-of-trust evidence
//! 4. **Reject** broken checksums (retire with `confidence = 0`)
//!
//! # Environment variables
//!
//! - `UPDATE_CHECKSUMS=1` — auto-accept checksums within tolerance (CI use)
//! - `REPLACE_CHECKSUMS=1` — replace all checksums with the new output (reset baseline)
//!
//! # Example
//!
//! ```no_run
//! use zensim_regress::manager::{ChecksumManager, CheckResult};
//!
//! let mgr = ChecksumManager::new("tests/checksums");
//! let result = mgr.check_file("resize_bicubic", "output.png").unwrap();
//! match &result {
//!     CheckResult::Match { .. } => println!("exact match"),
//!     CheckResult::WithinTolerance { report, .. } => {
//!         println!("within tolerance: {report}");
//!     }
//!     CheckResult::Failed { report, .. } => {
//!         if let Some(r) = report {
//!             panic!("regression: {r}");
//!         } else {
//!             panic!("hash mismatch, no reference image for comparison");
//!         }
//!     }
//!     CheckResult::NoBaseline { .. } => println!("first run — no baseline"),
//! }
//! ```

use std::path::{Path, PathBuf};

use zensim::{RgbaSlice, Zensim, ZensimProfile};

use crate::arch::detect_arch_tag;
use crate::checksum_file::{ChecksumDiff, ChecksumEntry, TestChecksumFile, checksum_path};
use crate::diff_image::create_comparison_montage_raw;
use crate::error::RegressError;
use crate::hasher::{ChecksumHasher, SeaHasher};
use crate::remote::ReferenceStorage;
use crate::testing::{RegressionReport, RegressionTolerance, check_regression};

/// How to handle new checksums via environment variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpdateMode {
    /// Normal: fail on mismatch (default).
    Normal,
    /// Auto-accept checksums that pass tolerance.
    Update,
    /// Replace all entries with the new output.
    Replace,
}

impl UpdateMode {
    fn from_env() -> Self {
        if std::env::var("REPLACE_CHECKSUMS").is_ok_and(|v| v == "1") {
            Self::Replace
        } else if std::env::var("UPDATE_CHECKSUMS").is_ok_and(|v| v == "1") {
            Self::Update
        } else {
            Self::Normal
        }
    }
}

/// Result of checking actual output against stored checksums.
#[derive(Debug)]
pub enum CheckResult {
    /// Hash matched an active checksum entry.
    Match {
        /// ID of the matched entry.
        entry_id: String,
        /// Confidence of the matched entry.
        confidence: u32,
    },

    /// No hash match, but zensim comparison passes tolerance.
    WithinTolerance {
        /// Zensim regression report.
        report: RegressionReport,
        /// ID of the authoritative checksum this was compared against.
        authoritative_id: String,
        /// Hash of the actual output.
        actual_hash: String,
        /// Whether the new hash was auto-accepted (UPDATE_CHECKSUMS mode).
        auto_accepted: bool,
    },

    /// No hash match and comparison exceeds tolerance (or no comparison possible).
    Failed {
        /// Zensim regression report, if pixel comparison was possible.
        /// `None` when no reference image is available for comparison.
        report: Option<RegressionReport>,
        /// ID of the authoritative checksum this was compared against, if any.
        authoritative_id: Option<String>,
        /// Hash of the actual output.
        actual_hash: String,
    },

    /// No checksum file or no active entries — first run.
    NoBaseline {
        /// Hash of the actual output.
        actual_hash: String,
        /// Whether this was auto-accepted (UPDATE or REPLACE mode).
        auto_accepted: bool,
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
}

impl std::fmt::Display for CheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Match {
                entry_id,
                confidence,
            } => write!(f, "PASS (exact match, confidence={confidence}, {entry_id})"),

            Self::WithinTolerance {
                report,
                authoritative_id,
                auto_accepted,
                ..
            } => {
                let delta = report.max_channel_delta();
                write!(
                    f,
                    "PASS (within tolerance, score={:.1}, max_delta=[{},{},{}], vs {authoritative_id}{})",
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
                authoritative_id,
                ..
            } => match (report, authoritative_id) {
                (Some(r), Some(auth_id)) => {
                    let delta = r.max_channel_delta();
                    write!(
                        f,
                        "FAIL (score={:.1}, max_delta=[{},{},{}], vs {auth_id})",
                        r.score(),
                        delta[0],
                        delta[1],
                        delta[2],
                    )
                }
                (None, Some(auth_id)) => {
                    write!(f, "FAIL (no reference image, vs {auth_id})")
                }
                _ => write!(f, "FAIL (no reference image)"),
            },

            Self::NoBaseline { auto_accepted, .. } => {
                if *auto_accepted {
                    write!(f, "NO BASELINE (auto-accepted)")
                } else {
                    write!(f, "NO BASELINE (run with UPDATE_CHECKSUMS=1)")
                }
            }
        }
    }
}

/// Manager for visual regression test checksums.
///
/// Owns the checksum directory path, hasher, and zensim instance.
/// Provides check/accept/reject workflow for test harnesses.
pub struct ChecksumManager {
    checksum_dir: PathBuf,
    hasher: Box<dyn ChecksumHasher>,
    zensim: Zensim,
    update_mode: UpdateMode,
    arch_tag: String,
    remote: Option<ReferenceStorage>,
    diff_dir: Option<PathBuf>,
}

impl ChecksumManager {
    /// Create a new manager with default settings.
    ///
    /// Reads `UPDATE_CHECKSUMS` and `REPLACE_CHECKSUMS` environment variables.
    pub fn new(checksum_dir: impl Into<PathBuf>) -> Self {
        Self {
            checksum_dir: checksum_dir.into(),
            hasher: Box::new(SeaHasher),
            zensim: Zensim::new(ZensimProfile::latest()),
            update_mode: UpdateMode::from_env(),
            arch_tag: detect_arch_tag().to_string(),
            remote: None,
            diff_dir: None,
        }
    }

    /// Override the hasher.
    pub fn with_hasher(mut self, hasher: impl ChecksumHasher + 'static) -> Self {
        self.hasher = Box::new(hasher);
        self
    }

    /// Force Normal mode (fail on mismatch, ignoring environment variables).
    pub fn with_update_mode_normal(mut self) -> Self {
        self.update_mode = UpdateMode::Normal;
        self
    }

    /// Force UPDATE mode (auto-accept within tolerance).
    pub fn with_update_mode_update(mut self) -> Self {
        self.update_mode = UpdateMode::Update;
        self
    }

    /// Force REPLACE mode (replace all entries).
    pub fn with_update_mode_replace(mut self) -> Self {
        self.update_mode = UpdateMode::Replace;
        self
    }

    /// Override the architecture tag (for testing).
    pub fn with_arch_tag(mut self, tag: impl Into<String>) -> Self {
        self.arch_tag = tag.into();
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
    /// Downloads are cached in `{checksum_dir}/.remote-cache`.
    pub fn with_remote_storage_from_env(mut self) -> Self {
        let cache_dir = self.checksum_dir.join(".remote-cache");
        self.remote = ReferenceStorage::from_env(cache_dir);
        self
    }

    /// Set a directory for saving comparison montages on mismatch.
    ///
    /// When set, each `WithinTolerance` or `Failed` result (where pixel
    /// comparison was possible) saves an `Expected | Actual | Diff` montage
    /// PNG to `{dir}/{test_name}.png`. Useful as a CI artifact.
    pub fn with_diff_output(mut self, dir: impl Into<PathBuf>) -> Self {
        self.diff_dir = Some(dir.into());
        self
    }

    /// Get the checksum directory path.
    pub fn checksum_dir(&self) -> &Path {
        &self.checksum_dir
    }

    /// Get the current architecture tag.
    pub fn arch_tag(&self) -> &str {
        &self.arch_tag
    }

    /// Get the TOML file path for a test name.
    pub fn test_path(&self, test_name: &str) -> PathBuf {
        checksum_path(&self.checksum_dir, test_name)
    }

    /// Load the checksum file for a test, or return None if it doesn't exist.
    pub fn load_file(&self, test_name: &str) -> Result<Option<TestChecksumFile>, RegressError> {
        let path = self.test_path(test_name);
        if !path.exists() {
            return Ok(None);
        }
        TestChecksumFile::read_from(&path).map(Some)
    }

    /// Get the effective tolerance for a test on the current architecture.
    pub fn tolerance_for_test(&self, test_name: &str) -> Result<RegressionTolerance, RegressError> {
        match self.load_file(test_name)? {
            Some(file) => Ok(file.tolerance.to_regression_tolerance(&self.arch_tag)),
            None => Ok(RegressionTolerance::exact()),
        }
    }

    // ─── Check workflow ──────────────────────────────────────────────────

    /// Check actual RGBA pixels against stored checksums.
    ///
    /// This is the main entry point for test harnesses working with raw pixels.
    pub fn check_pixels(
        &self,
        test_name: &str,
        actual_rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<CheckResult, RegressError> {
        let actual_hash = self.hasher.hash_pixels(actual_rgba, width, height);
        self.check_hash_inner(test_name, &actual_hash, Some((actual_rgba, width, height)))
    }

    /// Check an image file against stored checksums.
    ///
    /// Decodes the file to RGBA for hashing and comparison.
    pub fn check_file(
        &self,
        test_name: &str,
        actual_path: impl AsRef<Path>,
    ) -> Result<CheckResult, RegressError> {
        let actual_path = actual_path.as_ref();
        let img = image::open(actual_path)
            .map_err(|e| RegressError::image(actual_path, e))?
            .to_rgba8();
        let (w, h) = img.dimensions();
        self.check_pixels(test_name, img.as_raw(), w, h)
    }

    /// Check an encoded image file using opaque (file-level) hashing.
    ///
    /// Unlike [`check_file`](Self::check_file) which decodes to RGBA first,
    /// this hashes the raw file bytes — so different encoders producing
    /// identical pixels will have different checksums. Use this for encoder
    /// regression testing where you want to detect any output change.
    ///
    /// Tolerance comparison still works via pixel decode on mismatch.
    pub fn check_file_opaque(
        &self,
        test_name: &str,
        actual_path: impl AsRef<Path>,
    ) -> Result<CheckResult, RegressError> {
        let actual_path = actual_path.as_ref();
        let actual_hash = self.hasher.hash_file_bytes(actual_path)?;
        // Decode to pixels for tolerance comparison (only used on mismatch)
        let img = image::open(actual_path)
            .map_err(|e| RegressError::image(actual_path, e))?
            .to_rgba8();
        let (w, h) = img.dimensions();
        self.check_hash_inner(test_name, &actual_hash, Some((img.as_raw(), w, h)))
    }

    /// Check pixel data in any interleaved format against stored checksums.
    ///
    /// Converts to RGBA8 for format-independent hashing, then wraps in
    /// [`ZenpixelsSource`](zensim::ZenpixelsSource) for full-precision zensim comparison.
    #[cfg(feature = "zenpixels")]
    pub fn check_pixels_described(
        &self,
        test_name: &str,
        data: &[u8],
        descriptor: zenpixels::PixelDescriptor,
        width: u32,
        height: u32,
    ) -> Result<CheckResult, RegressError> {
        // Hash via RGBA8 conversion for format-independent checksums
        let actual_hash = crate::hasher::hash_pixels_described(
            self.hasher.as_ref(),
            data,
            descriptor,
            width,
            height,
        );

        // For pixel comparison, convert to RGBA8 so we can use RgbaSlice
        // (ZenpixelsSource might target a non-RGBA8 format; RGBA8 is universal)
        let target = zenpixels::PixelDescriptor::RGBA8_SRGB;
        let converter = zenpixels::RowConverter::new(descriptor, target)?;

        let rgba8 = if converter.is_identity() {
            data.to_vec()
        } else {
            let src_bpp = descriptor.bytes_per_pixel();
            let dst_bpp = target.bytes_per_pixel();
            let src_stride = width as usize * src_bpp;
            let dst_stride = width as usize * dst_bpp;
            let mut buf = vec![0u8; height as usize * dst_stride];
            for y in 0..height as usize {
                let src_row = &data[y * src_stride..y * src_stride + src_stride];
                let dst_row = &mut buf[y * dst_stride..y * dst_stride + dst_stride];
                converter.convert_row(src_row, dst_row, width);
            }
            buf
        };

        self.check_hash_inner(test_name, &actual_hash, Some((&rgba8, width, height)))
    }

    /// Check a pre-computed hash against stored checksums.
    ///
    /// Cannot run zensim comparison (no pixel data), so WithinTolerance
    /// is not possible — returns Match, Failed (with no report), or NoBaseline.
    pub fn check_hash(
        &self,
        test_name: &str,
        actual_hash: &str,
    ) -> Result<CheckResult, RegressError> {
        self.check_hash_inner(test_name, actual_hash, None)
    }

    /// Core check logic.
    fn check_hash_inner(
        &self,
        test_name: &str,
        actual_hash: &str,
        pixels: Option<(&[u8], u32, u32)>,
    ) -> Result<CheckResult, RegressError> {
        let path = self.test_path(test_name);

        // Load or handle missing file
        let mut file = match self.load_file(test_name)? {
            Some(f) => f,
            None => return self.handle_no_baseline(test_name, actual_hash, &path, pixels),
        };

        // REPLACE mode: wipe and write new baseline
        if self.update_mode == UpdateMode::Replace {
            return self.replace_baseline(&mut file, actual_hash, &path);
        }

        // Check for direct hash match
        if let Some(entry) = file.find_by_id(actual_hash) {
            if entry.is_active() {
                let confidence = entry.confidence;

                // Update arch list if this arch isn't recorded yet
                if let Some(entry_mut) = file.find_by_id_mut(actual_hash) {
                    if !entry_mut.arch.iter().any(|a| a == &self.arch_tag) {
                        entry_mut.arch.push(self.arch_tag.clone());
                        file.write_to(&path)?;
                    }
                }

                return Ok(CheckResult::Match {
                    entry_id: actual_hash.to_string(),
                    confidence,
                });
            }
            // Matched a retired entry — still a mismatch, fall through to comparison
        }

        // No direct match — try zensim comparison if we have pixels
        let Some((rgba, w, h)) = pixels else {
            // Hash-only mode: can't compare pixels
            return Ok(CheckResult::Failed {
                report: None,
                authoritative_id: file.authoritative().map(|e| e.id.clone()),
                actual_hash: actual_hash.to_string(),
            });
        };

        // Need a reference image to compare against
        let authoritative = match file.authoritative() {
            Some(e) => e.clone(),
            None => {
                // No active entries → treat as no baseline
                return self.handle_no_baseline(test_name, actual_hash, &path, Some((rgba, w, h)));
            }
        };

        // Load reference image for comparison
        let ref_path = self.find_reference_image(test_name, Some(&authoritative.id));
        let (ref_rgba, rw, rh) = match ref_path {
            Some(p) => decode_reference_png(&p)?,
            None => {
                // No reference image file — can't do pixel comparison
                if self.update_mode == UpdateMode::Update {
                    self.accept_inner(
                        &mut file,
                        actual_hash,
                        None,
                        "auto-accepted (no reference image)",
                        &path,
                    )?;
                    return Ok(CheckResult::NoBaseline {
                        actual_hash: actual_hash.to_string(),
                        auto_accepted: true,
                    });
                }
                return Ok(CheckResult::Failed {
                    report: None,
                    authoritative_id: Some(authoritative.id.clone()),
                    actual_hash: actual_hash.to_string(),
                });
            }
        };

        // Run zensim comparison
        let tolerance = file.tolerance.to_regression_tolerance(&self.arch_tag);

        let ref_pixels = rgba_bytes_to_pixels(&ref_rgba);
        let actual_pixels = rgba_bytes_to_pixels(rgba);
        let ref_source = RgbaSlice::new(&ref_pixels, rw as usize, rh as usize);
        let actual_source = RgbaSlice::new(&actual_pixels, w as usize, h as usize);

        let report = check_regression(&self.zensim, &ref_source, &actual_source, &tolerance)?;

        // Save diff montage if diff_dir is set
        self.save_diff_montage(test_name, &ref_rgba, rw, rh, rgba, w, h);

        if report.passed() {
            let auto_accepted = self.update_mode == UpdateMode::Update;
            if auto_accepted {
                self.accept_inner(
                    &mut file,
                    actual_hash,
                    Some(&report),
                    "auto-accepted within tolerance",
                    &path,
                )?;
            }
            Ok(CheckResult::WithinTolerance {
                report,
                authoritative_id: authoritative.id.clone(),
                actual_hash: actual_hash.to_string(),
                auto_accepted,
            })
        } else {
            Ok(CheckResult::Failed {
                report: Some(report),
                authoritative_id: Some(authoritative.id.clone()),
                actual_hash: actual_hash.to_string(),
            })
        }
    }

    // ─── Accept / Reject ─────────────────────────────────────────────────

    /// Accept a new checksum for a test.
    ///
    /// Adds a new active entry with chain-of-trust diff evidence (if a report
    /// is provided). If the test file doesn't exist, creates it.
    pub fn accept(
        &self,
        test_name: &str,
        actual_hash: &str,
        report: Option<&RegressionReport>,
        reason: &str,
    ) -> Result<(), RegressError> {
        let path = self.test_path(test_name);
        let mut file = self
            .load_file(test_name)?
            .unwrap_or_else(|| TestChecksumFile::new(test_name));
        self.accept_inner(&mut file, actual_hash, report, reason, &path)
    }

    /// Inner accept: adds entry and writes file.
    fn accept_inner(
        &self,
        file: &mut TestChecksumFile,
        actual_hash: &str,
        report: Option<&RegressionReport>,
        reason: &str,
        path: &Path,
    ) -> Result<(), RegressError> {
        // Don't duplicate if already present
        if let Some(existing) = file.find_by_id_mut(actual_hash) {
            // Re-activate if retired
            if !existing.is_active() {
                existing.confidence = 10;
                existing.status = None;
            }
            existing.reason = Some(reason.to_string());
            if !existing.arch.iter().any(|a| a == &self.arch_tag) {
                existing.arch.push(self.arch_tag.clone());
            }
            file.write_to(path)?;
            return Ok(());
        }

        // Build diff evidence
        let diff = report.and_then(|r| {
            file.authoritative()
                .map(|auth| ChecksumDiff::from_report(r, &auth.id))
        });

        let commit = current_commit_short();

        let entry = ChecksumEntry {
            id: actual_hash.to_string(),
            confidence: 10,
            commit,
            arch: vec![self.arch_tag.clone()],
            reason: Some(reason.to_string()),
            status: None,
            diff,
        };

        file.checksum.push(entry);
        file.write_to(path)?;

        // Upload reference image to remote if we have one locally
        let sanitized = crate::checksum_file::sanitize_name(&file.name);
        let images_dir = path.parent().unwrap_or(Path::new(".")).join("images");
        let ref_path = images_dir.join(format!("{sanitized}.png"));
        if ref_path.exists() {
            self.upload_reference_image(actual_hash, &ref_path);
        }

        Ok(())
    }

    /// Reject a checksum: set confidence to 0 and mark as wrong.
    ///
    /// Returns `true` if the entry was found and rejected, `false` if not found.
    pub fn reject(
        &self,
        test_name: &str,
        checksum_id: &str,
        reason: &str,
    ) -> Result<bool, RegressError> {
        let path = self.test_path(test_name);
        let mut file = match self.load_file(test_name)? {
            Some(f) => f,
            None => return Ok(false),
        };

        let entry = match file.find_by_id_mut(checksum_id) {
            Some(e) => e,
            None => return Ok(false),
        };

        entry.confidence = 0;
        entry.status = Some("wrong".to_string());
        entry.reason = Some(reason.to_string());
        file.write_to(&path)?;
        Ok(true)
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    fn handle_no_baseline(
        &self,
        test_name: &str,
        actual_hash: &str,
        path: &Path,
        pixels: Option<(&[u8], u32, u32)>,
    ) -> Result<CheckResult, RegressError> {
        let auto_accepted = self.update_mode != UpdateMode::Normal;
        if auto_accepted {
            let mut file = TestChecksumFile::new(test_name);
            let commit = current_commit_short();
            file.checksum.push(ChecksumEntry {
                id: actual_hash.to_string(),
                confidence: 10,
                commit,
                arch: vec![self.arch_tag.clone()],
                reason: Some("initial baseline".to_string()),
                status: None,
                diff: None,
            });
            file.write_to(path)?;

            // Save reference image so the next run can do zensim comparison
            if let Some((rgba, w, h)) = pixels {
                let _ = self.save_reference_image(test_name, rgba, w, h);
            }
        }
        Ok(CheckResult::NoBaseline {
            actual_hash: actual_hash.to_string(),
            auto_accepted,
        })
    }

    fn replace_baseline(
        &self,
        file: &mut TestChecksumFile,
        actual_hash: &str,
        path: &Path,
    ) -> Result<CheckResult, RegressError> {
        // Retire all existing entries
        for entry in &mut file.checksum {
            if entry.is_active() {
                entry.confidence = 0;
                entry.status = Some("replaced".to_string());
            }
        }

        // Add new entry
        let commit = current_commit_short();
        file.checksum.push(ChecksumEntry {
            id: actual_hash.to_string(),
            confidence: 10,
            commit,
            arch: vec![self.arch_tag.clone()],
            reason: Some("replaced baseline".to_string()),
            status: None,
            diff: None,
        });
        file.write_to(path)?;

        Ok(CheckResult::Match {
            entry_id: actual_hash.to_string(),
            confidence: 10,
        })
    }

    /// Try to find a reference image for a given test name.
    ///
    /// 1. Looks locally at `{checksum_dir}/images/{sanitized_test_name}.png`
    /// 2. If remote storage is configured, tries downloading by `authoritative_id`
    fn find_reference_image(
        &self,
        test_name: &str,
        authoritative_id: Option<&str>,
    ) -> Option<PathBuf> {
        let images_dir = self.checksum_dir.join("images");
        let sanitized = crate::checksum_file::sanitize_name(test_name);
        let local_path = images_dir.join(format!("{sanitized}.png"));
        if local_path.exists() {
            return Some(local_path);
        }

        // Try remote download by authoritative checksum ID
        if let (Some(remote), Some(auth_id)) = (&self.remote, authoritative_id) {
            match remote.download_reference(auth_id) {
                Ok(Some(cached_path)) => {
                    // Copy to local images dir for future use
                    let _ = std::fs::create_dir_all(&images_dir);
                    let _ = std::fs::copy(&cached_path, &local_path);
                    return Some(local_path);
                }
                Ok(None) => {} // not available remotely
                Err(e) => eprintln!("Warning: remote download failed: {e}"),
            }
        }

        None
    }

    /// Save a reference image for future comparisons.
    ///
    /// Stores the image at `{checksum_dir}/images/{sanitized_name}.png`.
    pub fn save_reference_image(
        &self,
        test_name: &str,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<PathBuf, RegressError> {
        let images_dir = self.checksum_dir.join("images");
        std::fs::create_dir_all(&images_dir).map_err(|e| RegressError::io(&images_dir, e))?;

        let sanitized = crate::checksum_file::sanitize_name(test_name);
        let path = images_dir.join(format!("{sanitized}.png"));

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

    /// Upload a reference image to remote storage (if configured).
    ///
    /// Best-effort: logs warning on failure, does not return error.
    pub fn upload_reference_image(&self, checksum_id: &str, local_path: &Path) {
        if let Some(remote) = &self.remote {
            if let Err(e) = remote.upload_reference(local_path, checksum_id) {
                eprintln!("Warning: failed to upload reference {checksum_id}: {e}");
            }
        }
    }

    /// Save a comparison montage (Expected | Actual | Diff) as a PNG.
    ///
    /// The images must have the same dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn save_diff_image(
        &self,
        test_name: &str,
        expected: &[u8],
        expected_w: u32,
        expected_h: u32,
        actual: &[u8],
        actual_w: u32,
        actual_h: u32,
        output_dir: &Path,
    ) -> Result<PathBuf, RegressError> {
        std::fs::create_dir_all(output_dir).map_err(|e| RegressError::io(output_dir, e))?;
        let sanitized = crate::checksum_file::sanitize_name(test_name);
        let out_path = output_dir.join(format!("{sanitized}.png"));

        // If dimensions differ, just save actual as-is (can't diff)
        if expected_w != actual_w || expected_h != actual_h {
            let img = image::RgbaImage::from_raw(actual_w, actual_h, actual.to_vec()).ok_or_else(
                || RegressError::Io {
                    path: out_path.clone(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "invalid dimensions for pixel data",
                    ),
                },
            )?;
            img.save(&out_path)
                .map_err(|e| RegressError::image(&out_path, e))?;
            return Ok(out_path);
        }

        let montage =
            create_comparison_montage_raw(expected, actual, expected_w, expected_h, 10, 2);
        montage
            .save(&out_path)
            .map_err(|e| RegressError::image(&out_path, e))?;
        Ok(out_path)
    }

    /// Internal: save diff montage if diff_dir is configured.
    #[allow(clippy::too_many_arguments)]
    fn save_diff_montage(
        &self,
        test_name: &str,
        ref_rgba: &[u8],
        rw: u32,
        rh: u32,
        actual_rgba: &[u8],
        aw: u32,
        ah: u32,
    ) {
        let Some(dir) = &self.diff_dir else { return };
        if let Err(e) = self.save_diff_image(test_name, ref_rgba, rw, rh, actual_rgba, aw, ah, dir)
        {
            eprintln!("Warning: failed to save diff image for {test_name}: {e}");
        }
    }
}

/// Decode a PNG reference image to RGBA8 pixels.
///
/// Currently uses the `image` crate. The `zenpng` feature flag is reserved
/// for future faster decode via zenpng.
fn decode_reference_png(path: &Path) -> Result<(Vec<u8>, u32, u32), RegressError> {
    let img = image::open(path)
        .map_err(|e| RegressError::image(path, e))?
        .to_rgba8();
    let (w, h) = img.dimensions();
    Ok((img.into_raw(), w, h))
}

/// Collect `&[u8]` into a `Vec<[u8; 4]>` (RGBA pixels).
///
/// Panics if the length is not a multiple of 4.
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

    fn make_test_pixels(w: u32, h: u32, seed: u8) -> Vec<u8> {
        (0..w * h)
            .flat_map(|i| {
                let v = (i as u8).wrapping_mul(17).wrapping_add(seed);
                [v, v.wrapping_add(30), v.wrapping_add(60), 255u8]
            })
            .collect()
    }

    fn make_variant_pixels(base: &[u8], delta: u8) -> Vec<u8> {
        base.chunks(4)
            .flat_map(|px| [px[0].saturating_add(delta), px[1], px[2], px[3]])
            .collect()
    }

    // ─── No baseline ─────────────────────────────────────────────────

    #[test]
    fn check_no_baseline_normal_mode() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let px = make_test_pixels(16, 16, 0);
        let result = mgr.check_pixels("test_no_baseline", &px, 16, 16).unwrap();

        assert!(matches!(
            result,
            CheckResult::NoBaseline {
                auto_accepted: false,
                ..
            }
        ));
        assert!(!result.passed());
    }

    #[test]
    fn check_no_baseline_update_mode() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_update();

        let px = make_test_pixels(16, 16, 0);
        let result = mgr
            .check_pixels("test_update_baseline", &px, 16, 16)
            .unwrap();

        assert!(matches!(
            result,
            CheckResult::NoBaseline {
                auto_accepted: true,
                ..
            }
        ));
        assert!(result.passed());

        // File should exist now
        let path = mgr.test_path("test_update_baseline");
        assert!(path.exists());
        let file = TestChecksumFile::read_from(&path).unwrap();
        assert_eq!(file.checksum.len(), 1);
        assert_eq!(file.checksum[0].arch, vec![mgr.arch_tag()]);
    }

    #[test]
    fn check_no_baseline_replace_mode() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_replace();

        let px = make_test_pixels(16, 16, 0);
        let result = mgr
            .check_pixels("test_replace_baseline", &px, 16, 16)
            .unwrap();

        assert!(matches!(
            result,
            CheckResult::NoBaseline {
                auto_accepted: true,
                ..
            }
        ));
        assert!(result.passed());
    }

    // ─── Direct match ────────────────────────────────────────────────

    #[test]
    fn check_direct_match() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let px = make_test_pixels(16, 16, 42);
        let hash = mgr.hasher.hash_pixels(&px, 16, 16);

        // Create baseline file
        let mut file = TestChecksumFile::new("test_match");
        file.checksum.push(ChecksumEntry {
            id: hash.clone(),
            confidence: 10,
            commit: None,
            arch: vec![],
            reason: Some("test baseline".to_string()),
            status: None,
            diff: None,
        });
        file.write_to(&mgr.test_path("test_match")).unwrap();

        let result = mgr.check_pixels("test_match", &px, 16, 16).unwrap();
        assert!(matches!(result, CheckResult::Match { confidence: 10, .. }));
        assert!(result.passed());
    }

    #[test]
    fn check_match_updates_arch() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path())
            .with_update_mode_normal()
            .with_arch_tag("test-arch");

        let px = make_test_pixels(16, 16, 42);
        let hash = mgr.hasher.hash_pixels(&px, 16, 16);

        let mut file = TestChecksumFile::new("test_arch_update");
        file.checksum.push(ChecksumEntry::new(hash.clone()));
        file.write_to(&mgr.test_path("test_arch_update")).unwrap();

        let _ = mgr.check_pixels("test_arch_update", &px, 16, 16).unwrap();

        let updated = TestChecksumFile::read_from(&mgr.test_path("test_arch_update")).unwrap();
        assert!(updated.checksum[0].arch.contains(&"test-arch".to_string()));
    }

    // ─── Retired entry not matched ──────────────────────────────────

    #[test]
    fn check_retired_entry_not_matched() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let px = make_test_pixels(16, 16, 42);
        let hash = mgr.hasher.hash_pixels(&px, 16, 16);

        // Create file with only a retired entry
        let mut file = TestChecksumFile::new("test_retired");
        file.checksum.push(ChecksumEntry {
            id: hash.clone(),
            confidence: 0,
            commit: None,
            arch: vec![],
            reason: Some("retired".to_string()),
            status: Some("wrong".to_string()),
            diff: None,
        });
        file.write_to(&mgr.test_path("test_retired")).unwrap();

        // No active entries → handle_no_baseline path
        let result = mgr.check_pixels("test_retired", &px, 16, 16).unwrap();
        assert!(!result.passed());
    }

    // ─── Hash-only check ─────────────────────────────────────────────

    #[test]
    fn check_hash_match() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let hash = "sea:deadbeef12345678";
        let mut file = TestChecksumFile::new("test_hash");
        file.checksum.push(ChecksumEntry::new(hash));
        file.write_to(&mgr.test_path("test_hash")).unwrap();

        let result = mgr.check_hash("test_hash", hash).unwrap();
        assert!(matches!(result, CheckResult::Match { .. }));
    }

    #[test]
    fn check_hash_mismatch_no_pixels() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let mut file = TestChecksumFile::new("test_hash_miss");
        file.checksum
            .push(ChecksumEntry::new("sea:aaaa000000000000"));
        file.write_to(&mgr.test_path("test_hash_miss")).unwrap();

        let result = mgr
            .check_hash("test_hash_miss", "sea:bbbb000000000000")
            .unwrap();
        assert!(matches!(result, CheckResult::Failed { report: None, .. }));
    }

    // ─── Accept / Reject ─────────────────────────────────────────────

    #[test]
    fn accept_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path())
            .with_update_mode_normal()
            .with_arch_tag("test-arch");

        mgr.accept("test_accept", "sea:1234567890abcdef", None, "manual accept")
            .unwrap();

        let file = TestChecksumFile::read_from(&mgr.test_path("test_accept")).unwrap();
        assert_eq!(file.checksum.len(), 1);
        assert_eq!(file.checksum[0].id, "sea:1234567890abcdef");
        assert_eq!(file.checksum[0].reason.as_deref(), Some("manual accept"));
        assert!(file.checksum[0].arch.contains(&"test-arch".to_string()));
    }

    #[test]
    fn accept_does_not_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        mgr.accept("test_nodup", "sea:1111111111111111", None, "first")
            .unwrap();
        mgr.accept("test_nodup", "sea:1111111111111111", None, "second")
            .unwrap();

        let file = TestChecksumFile::read_from(&mgr.test_path("test_nodup")).unwrap();
        assert_eq!(file.checksum.len(), 1, "should not duplicate entry");
        assert_eq!(file.checksum[0].reason.as_deref(), Some("second"));
    }

    #[test]
    fn accept_reactivates_retired() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let mut file = TestChecksumFile::new("test_reactivate");
        file.checksum.push(ChecksumEntry {
            id: "sea:aaaa000000000000".to_string(),
            confidence: 0,
            commit: None,
            arch: vec![],
            reason: Some("was wrong".to_string()),
            status: Some("wrong".to_string()),
            diff: None,
        });
        file.write_to(&mgr.test_path("test_reactivate")).unwrap();

        mgr.accept(
            "test_reactivate",
            "sea:aaaa000000000000",
            None,
            "reactivated",
        )
        .unwrap();

        let updated = TestChecksumFile::read_from(&mgr.test_path("test_reactivate")).unwrap();
        assert_eq!(updated.checksum[0].confidence, 10);
        assert!(updated.checksum[0].status.is_none());
        assert_eq!(updated.checksum[0].reason.as_deref(), Some("reactivated"));
    }

    #[test]
    fn reject_marks_wrong() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        mgr.accept("test_reject", "sea:aaaa000000000000", None, "initial")
            .unwrap();
        let rejected = mgr
            .reject("test_reject", "sea:aaaa000000000000", "broken")
            .unwrap();
        assert!(rejected);

        let file = TestChecksumFile::read_from(&mgr.test_path("test_reject")).unwrap();
        assert_eq!(file.checksum[0].confidence, 0);
        assert_eq!(file.checksum[0].status.as_deref(), Some("wrong"));
        assert_eq!(file.checksum[0].reason.as_deref(), Some("broken"));
    }

    #[test]
    fn reject_nonexistent_returns_false() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let result = mgr
            .reject("nonexistent", "sea:0000000000000000", "reason")
            .unwrap();
        assert!(!result);
    }

    // ─── Replace mode ────────────────────────────────────────────────

    #[test]
    fn replace_retires_old_entries() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_replace();

        let mut file = TestChecksumFile::new("test_replace");
        file.checksum
            .push(ChecksumEntry::new("sea:old0000000000000"));
        file.write_to(&mgr.test_path("test_replace")).unwrap();

        let px = make_test_pixels(16, 16, 99);
        let new_hash = mgr.hasher.hash_pixels(&px, 16, 16);
        let result = mgr.check_pixels("test_replace", &px, 16, 16).unwrap();

        assert!(result.passed());

        let file = TestChecksumFile::read_from(&mgr.test_path("test_replace")).unwrap();
        let old = file.find_by_id("sea:old0000000000000").unwrap();
        assert_eq!(old.confidence, 0);
        assert_eq!(old.status.as_deref(), Some("replaced"));
        let new = file.find_by_id(&new_hash).unwrap();
        assert!(new.is_active());
    }

    // ─── Tolerance resolution ────────────────────────────────────────

    #[test]
    fn tolerance_for_missing_test_is_exact() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let _t = mgr.tolerance_for_test("nonexistent").unwrap();
    }

    // ─── Within tolerance (with reference image) ─────────────────────

    #[test]
    fn check_within_tolerance_with_reference() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let (w, h) = (16u32, 16u32);
        let base_px = make_test_pixels(w, h, 42);
        let base_hash = mgr.hasher.hash_pixels(&base_px, w, h);

        mgr.save_reference_image("test_tolerance", &base_px, w, h)
            .unwrap();

        let mut file = TestChecksumFile::new("test_tolerance");
        file.tolerance = crate::checksum_file::ToleranceSpec {
            max_delta: 1,
            min_similarity: 90.0,
            max_pixels_different: 1.0,
            ..crate::checksum_file::ToleranceSpec::exact()
        };
        file.checksum.push(ChecksumEntry::new(base_hash.clone()));
        file.write_to(&mgr.test_path("test_tolerance")).unwrap();

        let variant_px = make_variant_pixels(&base_px, 1);
        let result = mgr
            .check_pixels("test_tolerance", &variant_px, w, h)
            .unwrap();

        assert!(result.passed(), "should be within tolerance: {result:?}");
        assert!(matches!(
            result,
            CheckResult::WithinTolerance {
                auto_accepted: false,
                ..
            }
        ));
    }

    #[test]
    fn check_fails_tolerance_with_reference() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let (w, h) = (16u32, 16u32);
        let base_px = make_test_pixels(w, h, 42);
        let base_hash = mgr.hasher.hash_pixels(&base_px, w, h);

        mgr.save_reference_image("test_fail_tol", &base_px, w, h)
            .unwrap();

        let mut file = TestChecksumFile::new("test_fail_tol");
        file.checksum.push(ChecksumEntry::new(base_hash.clone()));
        file.write_to(&mgr.test_path("test_fail_tol")).unwrap();

        let variant_px = make_variant_pixels(&base_px, 50);
        let result = mgr
            .check_pixels("test_fail_tol", &variant_px, w, h)
            .unwrap();

        assert!(!result.passed(), "should fail tolerance");
        assert!(matches!(result, CheckResult::Failed { .. }));
    }

    #[test]
    fn check_within_tolerance_auto_accept() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_update();

        let (w, h) = (16u32, 16u32);
        let base_px = make_test_pixels(w, h, 42);
        let base_hash = mgr.hasher.hash_pixels(&base_px, w, h);

        mgr.save_reference_image("test_auto_accept", &base_px, w, h)
            .unwrap();

        let mut file = TestChecksumFile::new("test_auto_accept");
        file.tolerance = crate::checksum_file::ToleranceSpec {
            max_delta: 1,
            min_similarity: 90.0,
            max_pixels_different: 1.0,
            ..crate::checksum_file::ToleranceSpec::exact()
        };
        file.checksum.push(ChecksumEntry::new(base_hash.clone()));
        file.write_to(&mgr.test_path("test_auto_accept")).unwrap();

        let variant_px = make_variant_pixels(&base_px, 1);
        let variant_hash = mgr.hasher.hash_pixels(&variant_px, w, h);

        let result = mgr
            .check_pixels("test_auto_accept", &variant_px, w, h)
            .unwrap();
        assert!(result.passed());
        assert!(matches!(
            result,
            CheckResult::WithinTolerance {
                auto_accepted: true,
                ..
            }
        ));

        let updated = TestChecksumFile::read_from(&mgr.test_path("test_auto_accept")).unwrap();
        assert!(updated.find_by_id(&variant_hash).is_some());
        let new_entry = updated.find_by_id(&variant_hash).unwrap();
        assert!(new_entry.diff.is_some(), "should have chain-of-trust diff");
    }

    // ─── Reference image save/load ───────────────────────────────────

    #[test]
    fn save_reference_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let px = make_test_pixels(8, 8, 0);
        let path = mgr.save_reference_image("test_ref", &px, 8, 8).unwrap();

        assert!(path.exists());
        let img = image::open(&path).unwrap().to_rgba8();
        assert_eq!(img.dimensions(), (8, 8));
        assert_eq!(img.as_raw(), &px);
    }

    // ─── check_file_opaque ─────────────────────────────────────────

    #[test]
    fn check_file_opaque_different_hash_than_check_file() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_update();

        let px = make_test_pixels(8, 8, 77);
        let img_path = dir.path().join("test_output.png");
        let img = image::RgbaImage::from_raw(8, 8, px.clone()).unwrap();
        img.save(&img_path).unwrap();

        // check_file hashes decoded pixels, check_file_opaque hashes raw bytes
        let result_file = mgr.check_file("opaque_test_file", &img_path).unwrap();
        let result_opaque = mgr
            .check_file_opaque("opaque_test_opaque", &img_path)
            .unwrap();

        // Both should produce NoBaseline in update mode
        let hash_file = match &result_file {
            CheckResult::NoBaseline { actual_hash, .. } => actual_hash.clone(),
            other => panic!("expected NoBaseline, got {other:?}"),
        };
        let hash_opaque = match &result_opaque {
            CheckResult::NoBaseline { actual_hash, .. } => actual_hash.clone(),
            other => panic!("expected NoBaseline, got {other:?}"),
        };

        assert_ne!(
            hash_file, hash_opaque,
            "file-level hash should differ from pixel hash"
        );
    }

    #[test]
    fn check_file_opaque_matches_stored_hash() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

        let px = make_test_pixels(8, 8, 88);
        let img_path = dir.path().join("test_opaque_match.png");
        let img = image::RgbaImage::from_raw(8, 8, px).unwrap();
        img.save(&img_path).unwrap();

        // Compute the opaque hash and store it
        let opaque_hash = mgr.hasher.hash_file_bytes(&img_path).unwrap();
        let mut file = TestChecksumFile::new("opaque_match");
        file.checksum.push(ChecksumEntry::new(opaque_hash.clone()));
        file.write_to(&mgr.test_path("opaque_match")).unwrap();

        let result = mgr.check_file_opaque("opaque_match", &img_path).unwrap();
        assert!(
            matches!(result, CheckResult::Match { .. }),
            "should match stored opaque hash"
        );
    }

    // ─── Remote storage builder ─────────────────────────────────────

    #[test]
    fn with_remote_storage_sets_field() {
        let dir = tempfile::tempdir().unwrap();
        let remote = crate::remote::ReferenceStorage::new(
            "https://example.com/refs",
            None,
            false,
            dir.path().join("cache"),
        );
        let mgr = ChecksumManager::new(dir.path()).with_remote_storage(remote);
        // Just verify it doesn't panic; the remote field is private
        assert_eq!(mgr.checksum_dir(), dir.path());
    }

    #[test]
    fn with_remote_storage_from_env_builder() {
        // from_env is inherently env-dependent; just test the builder chain works.
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_remote_storage_from_env();
        assert_eq!(mgr.checksum_dir(), dir.path());
    }

    // ─── decode_reference_png ───────────────────────────────────────

    #[test]
    fn decode_reference_png_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let px = make_test_pixels(8, 8, 33);
        let path = dir.path().join("test.png");
        let img = image::RgbaImage::from_raw(8, 8, px.clone()).unwrap();
        img.save(&path).unwrap();

        let (decoded, w, h) = decode_reference_png(&path).unwrap();
        assert_eq!((w, h), (8, 8));
        assert_eq!(decoded, px);
    }

    // ─── Upload integration ─────────────────────────────────────────

    #[test]
    fn upload_reference_image_noop_without_remote() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();
        // Should not panic when no remote is configured
        mgr.upload_reference_image("sea:test123", Path::new("/nonexistent"));
    }
}
