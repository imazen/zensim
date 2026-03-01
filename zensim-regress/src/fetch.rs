//! Resource fetching with local caching.
//!
//! Provides URL → local file download with configurable backends and
//! disk caching. Used by [`ChecksumManager`](crate::manager::ChecksumManager)
//! for reference image download, and available to any test harness that
//! needs to fetch remote fixtures.
//!
//! # Backends
//!
//! - [`ShellFetcher`] — cross-platform, uses `curl`, `wget`, or PowerShell
//!   (whichever is available). No extra dependencies.
//! - Implement [`ResourceFetcher`] for custom backends (e.g., `ureq`, signed URLs).
//!
//! # Caching
//!
//! [`CachedFetcher`] wraps any fetcher with a local cache directory.
//! Files are stored by a caller-chosen filename. If the file already exists
//! on disk, no download occurs.
//!
//! ```no_run
//! use zensim_regress::fetch::{CachedFetcher, ShellFetcher};
//!
//! let fetcher = CachedFetcher::new(
//!     ShellFetcher::new(),
//!     "/tmp/image-cache",
//! );
//!
//! // Downloads on first call, returns cached path on subsequent calls.
//! let path = fetcher.ensure("https://example.com/ref.png", "ref.png").unwrap();
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::RegressError;

/// Trait for downloading a URL to a local file.
///
/// Implementations handle a single download attempt. For caching,
/// wrap with [`CachedFetcher`].
pub trait ResourceFetcher: Send + Sync {
    /// Download `url` to `dest`. Returns `Ok(())` on success.
    ///
    /// Implementations should clean up partial files on failure.
    fn fetch(&self, url: &str, dest: &Path) -> Result<(), RegressError>;
}

/// Downloads using whichever shell tool is available: curl, wget, or PowerShell.
///
/// Tries tools in order until one succeeds. No extra Rust dependencies required.
pub struct ShellFetcher {
    /// Maximum time in seconds for the download. 0 = no limit.
    pub timeout_secs: u32,
}

impl ShellFetcher {
    pub fn new() -> Self {
        Self { timeout_secs: 120 }
    }

    pub fn with_timeout(mut self, secs: u32) -> Self {
        self.timeout_secs = secs;
        self
    }

    fn try_curl(&self, url: &str, dest: &Path) -> Result<(), String> {
        let mut cmd = Command::new("curl");
        cmd.args(["-fSL", "-o"]).arg(dest).arg(url);
        if self.timeout_secs > 0 {
            cmd.arg("--max-time").arg(self.timeout_secs.to_string());
        }
        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());

        let output = cmd.output().map_err(|e| format!("curl not found: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("curl exit {}: {}", output.status, stderr.trim()))
        }
    }

    fn try_wget(&self, url: &str, dest: &Path) -> Result<(), String> {
        let mut cmd = Command::new("wget");
        cmd.args(["-q", "-O"]).arg(dest).arg(url);
        if self.timeout_secs > 0 {
            cmd.arg("--timeout").arg(self.timeout_secs.to_string());
        }
        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());

        let output = cmd.output().map_err(|e| format!("wget not found: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("wget exit {}: {}", output.status, stderr.trim()))
        }
    }

    fn try_powershell(&self, url: &str, dest: &Path) -> Result<(), String> {
        let script = format!(
            "Invoke-WebRequest -Uri '{}' -OutFile '{}'",
            url.replace('\'', "''"),
            dest.display().to_string().replace('\'', "''"),
        );

        let mut cmd = Command::new("powershell");
        cmd.args(["-NoProfile", "-Command", &script]);
        if self.timeout_secs > 0 {
            // PowerShell doesn't have a direct timeout flag; use -Command with
            // a timeout wrapper if needed — for now, rely on system defaults.
        }
        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());

        let output = cmd
            .output()
            .map_err(|e| format!("powershell not found: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "powershell exit {}: {}",
                output.status,
                stderr.trim()
            ))
        }
    }
}

impl Default for ShellFetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceFetcher for ShellFetcher {
    fn fetch(&self, url: &str, dest: &Path) -> Result<(), RegressError> {
        // Ensure parent directory exists
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| RegressError::io(parent, e))?;
        }

        let result = self
            .try_curl(url, dest)
            .or_else(|_| self.try_wget(url, dest))
            .or_else(|_| self.try_powershell(url, dest));

        match result {
            Ok(()) => Ok(()),
            Err(msg) => {
                // Clean up partial download
                let _ = std::fs::remove_file(dest);
                Err(RegressError::Fetch {
                    url: url.to_string(),
                    message: msg,
                })
            }
        }
    }
}

/// Wraps a [`ResourceFetcher`] with a local cache directory.
///
/// Files are stored at `{cache_dir}/{filename}`. If the file already exists,
/// the download is skipped. No expiration or invalidation — the cache is
/// append-only by design (regression test images don't change).
pub struct CachedFetcher<F: ResourceFetcher = ShellFetcher> {
    inner: F,
    cache_dir: PathBuf,
}

impl CachedFetcher<ShellFetcher> {
    /// Create a cached fetcher with the default [`ShellFetcher`] backend.
    pub fn new(fetcher: ShellFetcher, cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            inner: fetcher,
            cache_dir: cache_dir.into(),
        }
    }
}

impl<F: ResourceFetcher> CachedFetcher<F> {
    /// Create a cached fetcher with a custom backend.
    pub fn with_fetcher(fetcher: F, cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            inner: fetcher,
            cache_dir: cache_dir.into(),
        }
    }

    /// Get the cache directory path.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Ensure a file exists at `{cache_dir}/{filename}`, downloading from `url` if needed.
    ///
    /// Returns the path to the cached file.
    pub fn ensure(&self, url: &str, filename: &str) -> Result<PathBuf, RegressError> {
        let dest = self.cache_dir.join(filename);
        if dest.exists() {
            return Ok(dest);
        }

        std::fs::create_dir_all(&self.cache_dir)
            .map_err(|e| RegressError::io(&self.cache_dir, e))?;

        self.inner.fetch(url, &dest)?;
        Ok(dest)
    }

    /// Ensure a file exists, using a URL builder that derives the URL from the filename.
    ///
    /// Common pattern: `fetcher.ensure_from_base("https://s3.example.com/images", "ref.png")`
    /// downloads from `https://s3.example.com/images/ref.png`.
    pub fn ensure_from_base(
        &self,
        base_url: &str,
        filename: &str,
    ) -> Result<PathBuf, RegressError> {
        let url = format!("{}/{}", base_url.trim_end_matches('/'), filename);
        self.ensure(&url, filename)
    }

    /// Remove a cached file if it exists.
    pub fn remove(&self, filename: &str) -> Result<(), RegressError> {
        let path = self.cache_dir.join(filename);
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| RegressError::io(&path, e))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock fetcher that writes fixed content.
    struct MockFetcher {
        content: Vec<u8>,
    }

    impl MockFetcher {
        fn new(content: &[u8]) -> Self {
            Self {
                content: content.to_vec(),
            }
        }
    }

    impl ResourceFetcher for MockFetcher {
        fn fetch(&self, _url: &str, dest: &Path) -> Result<(), RegressError> {
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).map_err(|e| RegressError::io(parent, e))?;
            }
            std::fs::write(dest, &self.content).map_err(|e| RegressError::io(dest, e))
        }
    }

    /// A mock fetcher that always fails.
    struct FailFetcher;

    impl ResourceFetcher for FailFetcher {
        fn fetch(&self, url: &str, _dest: &Path) -> Result<(), RegressError> {
            Err(RegressError::Fetch {
                url: url.to_string(),
                message: "mock failure".to_string(),
            })
        }
    }

    #[test]
    fn cached_fetcher_downloads_once() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = CachedFetcher::with_fetcher(MockFetcher::new(b"hello world"), dir.path());

        let path = fetcher
            .ensure("https://example.com/test.txt", "test.txt")
            .unwrap();
        assert!(path.exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");

        // Second call should not re-download (mock would write same content anyway,
        // but we verify the path is returned immediately).
        let path2 = fetcher
            .ensure("https://example.com/test.txt", "test.txt")
            .unwrap();
        assert_eq!(path, path2);
    }

    #[test]
    fn cached_fetcher_ensure_from_base() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = CachedFetcher::with_fetcher(MockFetcher::new(b"image data"), dir.path());

        let path = fetcher
            .ensure_from_base("https://s3.example.com/images", "ref.png")
            .unwrap();
        assert_eq!(path, dir.path().join("ref.png"));
        assert_eq!(std::fs::read(&path).unwrap(), b"image data");
    }

    #[test]
    fn cached_fetcher_base_url_trailing_slash() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = CachedFetcher::with_fetcher(MockFetcher::new(b"data"), dir.path());

        // Trailing slash should be handled.
        let path = fetcher
            .ensure_from_base("https://s3.example.com/images/", "ref.png")
            .unwrap();
        assert!(path.exists());
    }

    #[test]
    fn cached_fetcher_remove() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = CachedFetcher::with_fetcher(MockFetcher::new(b"temp"), dir.path());

        let path = fetcher
            .ensure("https://example.com/temp.txt", "temp.txt")
            .unwrap();
        assert!(path.exists());

        fetcher.remove("temp.txt").unwrap();
        assert!(!path.exists());

        // Remove of nonexistent file should succeed.
        fetcher.remove("temp.txt").unwrap();
    }

    #[test]
    fn cached_fetcher_propagates_error() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = CachedFetcher::with_fetcher(FailFetcher, dir.path());

        let result = fetcher.ensure("https://example.com/fail.txt", "fail.txt");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("mock failure"));

        // File should not exist after failed download.
        assert!(!dir.path().join("fail.txt").exists());
    }

    #[test]
    fn shell_fetcher_default_timeout() {
        let f = ShellFetcher::new();
        assert_eq!(f.timeout_secs, 120);
    }

    #[test]
    fn shell_fetcher_custom_timeout() {
        let f = ShellFetcher::new().with_timeout(30);
        assert_eq!(f.timeout_secs, 30);
    }
}
