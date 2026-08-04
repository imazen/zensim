//! Library surface for `zensim-validate`.
//!
//! The crate is primarily a collection of binaries (see `src/main.rs`
//! and `src/bin/*.rs`). This lib exists only to expose modules that
//! tests and small helper binaries (e.g. `parquet_load_bench`) need to
//! share with the main trainer without requiring `#[path = "..."]`
//! gymnastics.
//!
//! Adding modules here MUST be additive — existing binaries declare
//! their own `mod foo;` via `#[path = "../foo.rs"]` and should not be
//! touched as part of changes that add library surface.

pub mod adam_simd;
pub mod bake_runtime;
pub mod dial_spline;
pub mod eval_report;
pub mod gram_lasso;
#[allow(clippy::all)]
pub mod mlp_train;
pub mod npz;
pub mod output_calibration_spline;
pub mod panel;
pub mod parallel;
pub mod parquet_loader;
pub mod perf_trace;
pub mod prune;
pub mod simd_mlp;
pub mod train_manifest;
