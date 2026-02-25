//! # zensim
//!
//! Fast psychovisual image similarity metric combining ideas from
//! SSIMULACRA2 and butteraugli. Targets 10x faster than butteraugli
//! while maintaining strong correlation with human quality ratings.
//!
//! ## Quick start
//!
//! ```no_run
//! # let (src_pixels, dst_pixels) = (vec![[0u8; 3]; 64], vec![[0u8; 3]; 64]);
//! # let (width, height) = (8, 8);
//! // Single comparison
//! let result = zensim::compute_zensim(&src_pixels, &dst_pixels, width, height)?;
//! println!("score: {:.2}", result.score); // 0-100, higher = more similar
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ## Batch comparison (one reference, many distorted)
//!
//! When comparing one reference image against many distorted variants, use
//! [`precompute_reference`] + [`compute_zensim_with_ref`] to avoid redundant
//! XYB conversion and pyramid construction on the reference side.
//! Saves ~25% at 4K and ~34% at 8K per comparison.
//!
//! ```no_run
//! # let (ref_pixels, width, height) = (vec![[0u8; 3]; 64], 8usize, 8usize);
//! # let distorted_images: Vec<Vec<[u8; 3]>> = vec![];
//! let precomputed = zensim::precompute_reference(&ref_pixels, width, height)?;
//! for dst_pixels in &distorted_images {
//!     let result = zensim::compute_zensim_with_ref(&precomputed, dst_pixels, width, height)?;
//!     println!("score: {:.2}", result.score);
//! }
//! # Ok::<(), zensim::ZensimError>(())
//! ```
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
mod fused;
pub mod mapping;
mod metric;
mod pool;
mod simd_ops;
mod streaming;

pub use error::ZensimError;
pub use metric::{
    ZensimResult, compute_zensim, compute_zensim_with_ref, distance_to_score, precompute_reference,
};
pub use streaming::PrecomputedReference;

/// Training/research API — requires `features = ["training"]`.
///
/// These items expose metric internals (blur kernel shape, scale count,
/// masking, weight vectors) that change metric behavior. Scores produced
/// with non-default `ZensimConfig` are **not comparable** to the default
/// trained weights or the 0-100 score scale.
#[cfg(feature = "training")]
pub use metric::{
    FEATURES_PER_SCALE, WEIGHTS, ZensimConfig, compute_zensim_with_config,
    compute_zensim_with_ref_and_config, precompute_reference_with_scales, score_from_features,
};

/// Number of downscale levels. Each level halves resolution.
/// 4 scales covers 1x, 2x, 4x, 8x — sufficient for most perceptual effects.
pub(crate) const NUM_SCALES: usize = 4;
