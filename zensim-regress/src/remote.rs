//! Remote reference image storage (S3, R2, or any rclone-compatible remote).
//!
//! Coordinates download and upload of content-addressed reference images.
//! Downloads are lazy (on-demand when local reference is missing).
//! Uploads are opt-in (env var gated), and failures are warnings not errors.
//!
//! # Environment variables
//!
//! - `REGRESS_REFERENCE_URL` — base URL for downloads (required to enable remote)
//! - `REGRESS_UPLOAD_PREFIX` — rclone/s3 destination for uploads (optional)
//! - `UPLOAD_REFERENCES=1` — gate for upload (default off)
//!
//! # Filename convention
//!
//! Checksum IDs like `sea:abc123` become `sea_abc123.png` (`:` → `_`, append `.png`).

use std::path::{Path, PathBuf};

use crate::error::RegressError;
use crate::fetch::{CachedFetcher, ShellFetcher};
use crate::upload::{ResourceUploader, ShellUploader};

/// Configuration for remote reference image storage.
pub struct ReferenceStorage {
    /// Base URL for downloading (e.g., "https://s3-us-west-2.amazonaws.com/bucket/images")
    download_base_url: String,
    /// Remote destination prefix for uploads (e.g., "s3://bucket/images" or "r2:bucket/images")
    upload_prefix: Option<String>,
    /// Whether uploads are enabled (UPLOAD_REFERENCES=1)
    upload_enabled: bool,
    fetcher: CachedFetcher<ShellFetcher>,
    uploader: ShellUploader,
}

impl ReferenceStorage {
    /// Create from environment variables.
    ///
    /// Returns `None` if `REGRESS_REFERENCE_URL` is not set.
    /// `cache_dir` is where downloaded files are stored locally.
    pub fn from_env(cache_dir: impl Into<PathBuf>) -> Option<Self> {
        let download_base_url = std::env::var("REGRESS_REFERENCE_URL").ok()?;
        if download_base_url.is_empty() {
            return None;
        }

        let upload_prefix = std::env::var("REGRESS_UPLOAD_PREFIX")
            .ok()
            .and_then(|v| if v.is_empty() { None } else { Some(v) });
        let upload_enabled =
            std::env::var("UPLOAD_REFERENCES").is_ok_and(|v| v == "1" || v == "true");

        Some(Self {
            download_base_url,
            upload_prefix,
            upload_enabled,
            fetcher: CachedFetcher::new(ShellFetcher::new(), cache_dir),
            uploader: ShellUploader::new(),
        })
    }

    /// Create with explicit configuration (for testing).
    pub fn new(
        download_base_url: impl Into<String>,
        upload_prefix: Option<String>,
        upload_enabled: bool,
        cache_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            download_base_url: download_base_url.into(),
            upload_prefix,
            upload_enabled,
            fetcher: CachedFetcher::new(ShellFetcher::new(), cache_dir),
            uploader: ShellUploader::new(),
        }
    }

    /// Convert a checksum ID to a remote filename.
    ///
    /// `sea:abc123` → `sea_abc123.png`
    pub fn remote_filename(checksum_id: &str) -> String {
        let sanitized = checksum_id.replace(':', "_");
        format!("{sanitized}.png")
    }

    /// Download a reference image by checksum ID.
    ///
    /// Returns `Ok(Some(path))` if downloaded successfully, `Ok(None)` if not
    /// available remotely (404 or fetch error — not an error, just missing).
    pub fn download_reference(&self, checksum_id: &str) -> Result<Option<PathBuf>, RegressError> {
        let filename = Self::remote_filename(checksum_id);
        match self
            .fetcher
            .ensure_from_base(&self.download_base_url, &filename)
        {
            Ok(path) => Ok(Some(path)),
            Err(_) => Ok(None), // missing remote is not an error
        }
    }

    /// Upload a reference image to remote storage.
    ///
    /// No-op if uploads are disabled or no upload prefix is configured.
    /// Returns error only on actual upload failure when enabled.
    pub fn upload_reference(
        &self,
        local_path: &Path,
        checksum_id: &str,
    ) -> Result<(), RegressError> {
        if !self.upload_enabled {
            return Ok(());
        }
        let prefix = match &self.upload_prefix {
            Some(p) => p,
            None => return Ok(()),
        };

        let filename = Self::remote_filename(checksum_id);
        let remote_url = format!("{}/{}", prefix.trim_end_matches('/'), filename);
        self.uploader.upload(local_path, &remote_url)
    }

    /// Whether uploads are enabled and configured.
    pub fn uploads_configured(&self) -> bool {
        self.upload_enabled && self.upload_prefix.is_some()
    }

    /// The download base URL.
    pub fn download_base_url(&self) -> &str {
        &self.download_base_url
    }

    /// The upload prefix, if configured.
    pub fn upload_prefix(&self) -> Option<&str> {
        self.upload_prefix.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_filename_basic() {
        assert_eq!(
            ReferenceStorage::remote_filename("sea:abc123"),
            "sea_abc123.png"
        );
    }

    #[test]
    fn remote_filename_no_colon() {
        assert_eq!(
            ReferenceStorage::remote_filename("deadbeef"),
            "deadbeef.png"
        );
    }

    #[test]
    fn remote_filename_multiple_colons() {
        assert_eq!(
            ReferenceStorage::remote_filename("sea:abc:def"),
            "sea_abc_def.png"
        );
    }

    #[test]
    fn remote_filename_long_hash() {
        assert_eq!(
            ReferenceStorage::remote_filename("sea:a1b2c3d4e5f6789a"),
            "sea_a1b2c3d4e5f6789a.png"
        );
    }

    #[test]
    fn new_with_all_options() {
        let storage = ReferenceStorage::new(
            "https://example.com/refs",
            Some("s3://bucket/refs".to_string()),
            true,
            "/tmp/cache",
        );
        assert_eq!(storage.download_base_url(), "https://example.com/refs");
        assert_eq!(storage.upload_prefix(), Some("s3://bucket/refs"));
        assert!(storage.uploads_configured());
    }

    #[test]
    fn uploads_not_configured_without_prefix() {
        let storage = ReferenceStorage::new("https://example.com/refs", None, true, "/tmp/cache");
        assert!(!storage.uploads_configured());
    }

    #[test]
    fn uploads_not_configured_when_disabled() {
        let storage = ReferenceStorage::new(
            "https://example.com/refs",
            Some("s3://bucket/refs".to_string()),
            false,
            "/tmp/cache",
        );
        assert!(!storage.uploads_configured());
    }

    #[test]
    fn upload_noop_when_disabled() {
        let storage = ReferenceStorage::new(
            "https://example.com/refs",
            Some("s3://bucket/refs".to_string()),
            false, // uploads disabled
            "/tmp/cache",
        );
        // Should succeed (no-op) even with a non-existent file
        let result = storage.upload_reference(Path::new("/nonexistent"), "sea:test");
        assert!(result.is_ok());
    }

    #[test]
    fn upload_noop_when_no_prefix() {
        let storage = ReferenceStorage::new(
            "https://example.com/refs",
            None, // no upload prefix
            true,
            "/tmp/cache",
        );
        let result = storage.upload_reference(Path::new("/nonexistent"), "sea:test");
        assert!(result.is_ok());
    }
}
