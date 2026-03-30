//! Visual regression testing persistence and workflow for zensim.
//!
//! This crate provides checksum file management, architecture-aware
//! tolerance handling, and pixel hashing for visual regression testing. It
//! builds on `zensim`'s `RegressionReport` and `ErrorCategory` to create a
//! complete chain-of-trust system for tracking expected test outputs.
//!
//! # Format
//!
//! `.checksums` — line-oriented log files via [`checksums`]. Compact,
//! append-friendly, one file per test module with human-readable memorable names.
//!
//! # Quick start
//!
//! ```no_run
//! use zensim_regress::checksums::ChecksumManager;
//! use zensim_regress::hasher::SeaHasher;
//! use zensim_regress::hasher::ChecksumHasher;
//! use zensim_regress::arch::detect_arch_tag;
//!
//! // Hash an actual test output
//! let hasher = SeaHasher;
//! let hash = hasher.hash_file(std::path::Path::new("test_output.png")).unwrap();
//!
//! println!("arch: {}, hash: {}", detect_arch_tag(), hash);
//! ```

#![forbid(unsafe_code)]

/// CPU architecture detection and tag matching.
pub mod arch;
/// Checksum file management, hash comparison, and the primary `ChecksumManager` API.
pub mod checksums;
/// Amplified difference images and side-by-side comparison montages.
pub mod diff_image;
/// Human-readable diff formatting and tolerance shorthand parsing.
pub mod diff_summary;
/// Sixel terminal image rendering.
pub mod display;
/// Deterministic pixel distortions for testing tolerance boundaries.
pub mod distortions;
/// Error types.
pub mod error;
/// HTTP fetcher for downloading remote reference images.
pub mod fetch;
/// Minimal bitmap font for annotating diff images.
pub mod font;
/// Deterministic synthetic test image generators.
pub mod generators;
/// Pixel and file hashing (`ChecksumHasher` trait, `SeaHasher`).
pub mod hasher;
/// Advisory file locking for parallel test safety.
pub mod lock;
/// TSV manifest writer for CI result aggregation across platforms.
pub mod manifest;
/// Pixel oracle testing: compare image operations against scalar references.
pub mod oracle;
/// Memorable names from hashes (e.g., `sea:a1b2...` → `sunny-crab`).
pub mod petname;
/// S3/R2 remote reference image storage configuration.
pub mod remote;
/// HTML report generation from manifest data.
pub mod report;
/// SIMD consistency testing via archmage token permutations.
#[cfg(feature = "archmage")]
pub mod simd;
/// Tolerance-based image comparison, reporting, and the `check_regression` function.
pub mod testing;
/// Tolerance specification with per-architecture overrides.
pub mod tolerance;
/// Shell-based file uploader for remote storage.
pub mod upload;

// Re-export key types at crate root for convenience
pub use error::RegressError;
pub use testing::{
    ComparisonMethod, DimensionInfo, DimensionMismatchKind, RegressionReport,
    RegressionTolerance, check_regression, check_regression_resized, detect_transform,
    shrink_tolerance,
};
pub use tolerance::ToleranceSpec as Tolerance;
