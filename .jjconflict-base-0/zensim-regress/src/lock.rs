//! RAII advisory file lock guard.
//!
//! Provides `FileLockGuard` which acquires an exclusive advisory lock on
//! creation and releases it on drop, preventing lock leaks on early returns
//! or panics. Used internally by [`checksums`] and [`manifest`] for
//! multi-process safety.
//!
//! On `wasm32` targets, locking is a no-op (single-threaded environment).
//!
//! [`checksums`]: crate::checksums
//! [`manifest`]: crate::manifest

use std::path::{Path, PathBuf};

use crate::error::RegressError;

/// RAII guard for an exclusive advisory file lock.
///
/// Acquires an exclusive lock on construction (via `fs2` on native targets);
/// releases (and optionally deletes the lock file) on drop.
///
/// On `wasm32` targets, locking is a no-op — the file is opened but not locked.
///
/// # Example
///
/// ```no_run
/// # use zensim_regress::lock::FileLockGuard;
/// let _guard = FileLockGuard::acquire("data.lock")?;
/// // ... critical section — lock held until _guard is dropped ...
/// # Ok::<(), zensim_regress::error::RegressError>(())
/// ```
pub struct FileLockGuard {
    file: std::fs::File,
    path: PathBuf,
    remove_on_drop: bool,
}

impl FileLockGuard {
    /// Acquire an exclusive advisory lock.
    ///
    /// Creates the lock file if it doesn't exist. Blocks until the lock
    /// is available. The lock is released when the guard is dropped.
    pub fn acquire(lock_path: impl Into<PathBuf>) -> Result<Self, RegressError> {
        let path = lock_path.into();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|e| RegressError::io(&path, e))?;
        #[cfg(not(target_arch = "wasm32"))]
        fs2::FileExt::lock_exclusive(&file).map_err(|e| RegressError::io(&path, e))?;
        Ok(Self {
            file,
            path,
            remove_on_drop: false,
        })
    }

    /// Like [`acquire`](Self::acquire), but deletes the lock file on drop.
    ///
    /// Use this when the lock file is ephemeral and shouldn't persist
    /// between runs (e.g., `.checksums.lock`).
    pub fn acquire_and_cleanup(lock_path: impl Into<PathBuf>) -> Result<Self, RegressError> {
        let mut guard = Self::acquire(lock_path)?;
        guard.remove_on_drop = true;
        Ok(guard)
    }

    /// Acquire without propagating errors — returns `None` on failure.
    ///
    /// Use this for best-effort locking where the operation should proceed
    /// even if locking fails (e.g., manifest line writes).
    pub fn try_acquire(lock_path: impl Into<PathBuf>) -> Option<Self> {
        let path = lock_path.into();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .ok()?;
        #[cfg(not(target_arch = "wasm32"))]
        fs2::FileExt::lock_exclusive(&file).ok()?;
        Some(Self {
            file,
            path,
            remove_on_drop: false,
        })
    }

    /// The path to the lock file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Access the underlying file (e.g., to use as a sentinel).
    pub fn file(&self) -> &std::fs::File {
        &self.file
    }
}

impl Drop for FileLockGuard {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = fs2::FileExt::unlock(&self.file);
        }
        if self.remove_on_drop {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_and_drop() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("test.lock");

        {
            let guard = FileLockGuard::acquire(&lock_path).unwrap();
            assert!(lock_path.exists());
            assert_eq!(guard.path(), lock_path);
        }
        // Lock file still exists after drop (no cleanup)
        assert!(lock_path.exists());
    }

    #[test]
    fn acquire_and_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("ephemeral.lock");

        {
            let _guard = FileLockGuard::acquire_and_cleanup(&lock_path).unwrap();
            assert!(lock_path.exists());
        }
        // Lock file removed after drop
        assert!(!lock_path.exists());
    }

    #[test]
    fn try_acquire_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("try.lock");
        let guard = FileLockGuard::try_acquire(&lock_path);
        assert!(guard.is_some());
    }

    #[test]
    fn try_acquire_bad_path_returns_none() {
        let guard = FileLockGuard::try_acquire("/nonexistent/dir/impossible.lock");
        assert!(guard.is_none());
    }

    #[test]
    fn sequential_acquire() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("seq.lock");

        // First acquire, then drop, then re-acquire
        {
            let _g1 = FileLockGuard::acquire(&lock_path).unwrap();
        }
        {
            let _g2 = FileLockGuard::acquire(&lock_path).unwrap();
        }
    }
}
