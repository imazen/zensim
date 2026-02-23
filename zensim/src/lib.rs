//! # zensim
//!
//! Fast psychovisual image similarity metric combining ideas from
//! SSIMULACRA2 and butteraugli. Targets 10x faster than butteraugli
//! while maintaining strong correlation with human quality ratings.
//!
//! ## Design
//!
//! - XYB perceptual color space (cube root LMS, same as ssimulacra2/butteraugli)
//! - Multi-scale analysis (4 scales instead of ssimulacra2's 6)
//! - O(1)-per-pixel box blur cascade (3-pass approximates Gaussian)
//! - SSIM-based structural comparison with edge/texture features
//! - Contrast sensitivity function weighting per scale
//! - AVX2/AVX-512 SIMD throughout via archmage

#![forbid(unsafe_code)]

mod blur;
mod color;
mod error;
pub mod mapping;
mod metric;
mod pool;
mod simd_ops;

pub use error::ZensimError;
pub use metric::{ZensimResult, compute_zensim, distance_to_score};

/// Training/research API — requires `features = ["training"]`.
///
/// These items expose metric internals (blur kernel shape, scale count,
/// masking, weight vectors) that change metric behavior. Scores produced
/// with non-default `ZensimConfig` are **not comparable** to the default
/// trained weights or the 0-100 score scale.
#[cfg(feature = "training")]
pub use metric::{
    FEATURES_PER_SCALE, WEIGHTS, ZensimConfig, compute_zensim_with_config, score_from_features,
};

/// Number of downscale levels. Each level halves resolution.
/// 4 scales covers 1x, 2x, 4x, 8x — sufficient for most perceptual effects.
pub(crate) const NUM_SCALES: usize = 4;
