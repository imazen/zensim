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

pub mod arch;
pub mod checksums;
pub mod diff_image;
pub mod diff_summary;
pub mod display;
pub mod distortions;
pub mod error;
pub mod fetch;
pub mod generators;
pub mod hasher;
pub mod lock;
pub mod manifest;
pub mod petname;
pub mod remote;
pub mod report;
pub mod testing;
pub mod tolerance;
pub mod upload;

// Re-export key types at crate root for convenience
pub use error::RegressError;
pub use testing::{RegressionReport, RegressionTolerance, check_regression, shrink_tolerance};
pub use tolerance::ToleranceSpec as Tolerance;
