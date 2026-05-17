//! Integration test crate for running zensim under wasm32-wasip1 with SIMD128.
//!
//! This crate has no library code — all content is in integration tests.
//! See `tests/` for the actual test modules.
//!
//! # Running locally
//!
//! ```sh
//! # First run: generate checksum baselines
//! RUSTFLAGS='-C target-feature=+simd128' \
//!   UPDATE_CHECKSUMS=1 \
//!   cargo test --target wasm32-wasip1 -p zensim-wasm-tests -- --nocapture
//!
//! # Subsequent runs: verify against baselines
//! RUSTFLAGS='-C target-feature=+simd128' \
//!   cargo test --target wasm32-wasip1 -p zensim-wasm-tests -- --nocapture
//! ```
