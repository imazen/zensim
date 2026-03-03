//! Visual regression testing persistence and workflow for zensim.
//!
//! This crate provides checksum file management, architecture-aware
//! tolerance handling, and pixel hashing for visual regression testing. It
//! builds on `zensim`'s `RegressionReport` and `ErrorCategory` to create a
//! complete chain-of-trust system for tracking expected test outputs.
//!
//! # Formats
//!
//! - **TOML** (v0): Per-test `.toml` files via [`checksum_file`]. Original format.
//! - **`.checksums`** (v1): Line-oriented log files via [`checksums_v2`]. Compact,
//!   append-friendly, one file per test module with human-readable memorable names.
//!
//! # Quick start
//!
//! ```no_run
//! use zensim_regress::checksum_file::{TestChecksumFile, ChecksumEntry};
//! use zensim_regress::hasher::SeaHasher;
//! use zensim_regress::hasher::ChecksumHasher;
//! use zensim_regress::arch::detect_arch_tag;
//!
//! // Hash an actual test output
//! let hasher = SeaHasher;
//! let hash = hasher.hash_file(std::path::Path::new("test_output.png")).unwrap();
//!
//! // Check against stored checksums
//! let file = TestChecksumFile::read_from(
//!     std::path::Path::new("checksums/my_test.toml"),
//! ).unwrap();
//!
//! let matched = file.active_checksums().any(|e| e.id == hash);
//! println!("arch: {}, match: {}", detect_arch_tag(), matched);
//! ```

#![forbid(unsafe_code)]

pub mod arch;
pub mod checksum_file;
pub mod checksums_v2;
pub mod diff_image;
pub mod diff_summary;
pub mod display;
pub mod distortions;
pub mod error;
pub mod fetch;
pub mod generators;
pub mod hasher;
pub mod manager;
pub mod petname;
pub mod remote;
pub mod testing;
pub mod upload;

// Re-export key types at crate root for convenience
pub use checksum_file::ToleranceSpec as Tolerance;
pub use error::RegressError;
pub use testing::{RegressionReport, RegressionTolerance, check_regression, shrink_tolerance};

// Re-export toml so consumers can use `toml::Value` for the `meta` field
// without adding a separate `toml` dependency.
pub use toml;
