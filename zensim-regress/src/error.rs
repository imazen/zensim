//! Error types for visual regression testing.

use std::path::PathBuf;

/// Errors from regression checksum operations.
#[derive(Debug, thiserror::Error)]
pub enum RegressError {
    /// IO error reading/writing checksum files or images.
    #[error("IO error at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    /// Failed to parse a TOML checksum file.
    #[error("TOML parse error in {path}: {source}")]
    TomlParse {
        path: PathBuf,
        source: toml::de::Error,
    },

    /// Failed to serialize to TOML.
    #[error("TOML serialize error: {source}")]
    TomlSerialize {
        #[from]
        source: toml::ser::Error,
    },

    /// Image decoding error.
    #[error("image error at {path}: {source}")]
    Image {
        path: PathBuf,
        source: image::ImageError,
    },

    /// No checksum file found for this test.
    #[error("no checksum file for test {test_name:?} (expected at {path})")]
    NoChecksumFile { test_name: String, path: PathBuf },

    /// No active checksums in the file (all have confidence=0).
    #[error("no active checksums for test {test_name:?}")]
    NoActiveChecksums { test_name: String },

    /// Zensim metric error during comparison.
    #[error("zensim error: {0}")]
    Zensim(#[from] zensim::ZensimError),
}

impl RegressError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn toml_parse(path: impl Into<PathBuf>, source: toml::de::Error) -> Self {
        Self::TomlParse {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn image(path: impl Into<PathBuf>, source: image::ImageError) -> Self {
        Self::Image {
            path: path.into(),
            source,
        }
    }
}
