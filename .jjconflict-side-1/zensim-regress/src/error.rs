//! Error types for visual regression testing.

use std::path::PathBuf;

/// Errors from regression checksum operations.
#[derive(Debug, thiserror::Error)]
pub enum RegressError {
    /// IO error reading/writing checksum files or images.
    #[error("IO error at {path}: {source}")]
    Io {
        /// File path that caused the error.
        path: PathBuf,
        /// Underlying IO error.
        source: std::io::Error,
    },

    /// PNG decode/encode error.
    #[error("PNG error at {path}: {message}")]
    Png {
        /// File path that caused the error.
        path: PathBuf,
        /// Underlying PNG error message (string form, hides backend).
        message: String,
    },

    /// No checksum file found for this test.
    #[error("no checksum file for test {test_name:?} (expected at {path})")]
    NoChecksumFile {
        /// Test module name.
        test_name: String,
        /// Expected file path.
        path: PathBuf,
    },

    /// No active checksums in the file (all have confidence=0).
    #[error("no active checksums for test {test_name:?}")]
    NoActiveChecksums {
        /// Test module name.
        test_name: String,
    },

    /// Zensim metric error during comparison.
    #[error("zensim error: {0}")]
    Zensim(#[from] zensim::ZensimError),

    /// Failed to download a resource.
    #[error("fetch failed for {url}: {message}")]
    Fetch {
        /// URL that failed.
        url: String,
        /// Error description.
        message: String,
    },

    /// Failed to upload a resource.
    #[error("upload failed to {dest}: {message}")]
    Upload {
        /// Upload destination.
        dest: String,
        /// Error description.
        message: String,
    },
}

impl RegressError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn png(path: impl Into<PathBuf>, source: crate::pixel_ops::PngError) -> Self {
        Self::Png {
            path: path.into(),
            message: source.to_string(),
        }
    }
}
