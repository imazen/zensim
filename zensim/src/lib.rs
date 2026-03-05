//! # zensim
//!
//! Fast psychovisual image similarity metric combining ideas from
//! SSIMULACRA2 and butteraugli. Targets 10x faster than butteraugli
//! while maintaining strong correlation with human quality ratings.
//!
//! ## Quick start
//!
//! ```no_run
//! use zensim::{Zensim, ZensimProfile, RgbSlice};
//! # let (src_pixels, dst_pixels) = (vec![[0u8; 3]; 64], vec![[0u8; 3]; 64]);
//! let z = Zensim::new(ZensimProfile::latest());
//! let source = RgbSlice::new(&src_pixels, 8, 8);
//! let distorted = RgbSlice::new(&dst_pixels, 8, 8);
//! let result = z.compute(&source, &distorted)?;
//! println!("{}: {:.2}", result.profile, result.score);
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ## Batch comparison (one reference, many distorted)
//!
//! ```no_run
//! use zensim::{Zensim, ZensimProfile, RgbSlice};
//! # let (ref_pixels, width, height) = (vec![[0u8; 3]; 64], 8usize, 8usize);
//! # let distorted_images: Vec<Vec<[u8; 3]>> = vec![];
//! let z = Zensim::new(ZensimProfile::latest());
//! let source = RgbSlice::new(&ref_pixels, width, height);
//! let precomputed = z.precompute_reference(&source)?;
//! for dst_pixels in &distorted_images {
//!     let dst = RgbSlice::new(dst_pixels, width, height);
//!     let result = z.compute_with_ref(&precomputed, &dst)?;
//!     println!("score: {:.2}", result.score);
//! }
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ## RGBA support
//!
//! ```no_run
//! use zensim::{Zensim, ZensimProfile, RgbaSlice};
//! # let (src_rgba, dst_rgba) = (vec![[0u8; 4]; 64], vec![[0u8; 4]; 64]);
//! let z = Zensim::new(ZensimProfile::latest());
//! let source = RgbaSlice::new(&src_rgba, 8, 8);
//! let distorted = RgbaSlice::new(&dst_rgba, 8, 8);
//! let result = z.compute(&source, &distorted)?;
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ## Input requirements
//!
//! - **Color space:** sRGB. Future versions may accept additional color spaces
//!   (linear RGB, Display P3, etc.).
//! - **Pixel format:** `[R, G, B]` 8-bit sRGB, or `[R, G, B, A]` 8-bit sRGB
//!   with straight alpha. RGBA inputs are composited over a checkerboard before
//!   comparison so alpha differences produce visible distortion.
//! - **Dimensions:** Both images must be the same width × height, minimum 8×8.
//!
//! ## Score semantics
//!
//! Scores range 0–100, higher = more similar. `ZensimResult::raw_distance` is the
//! weighted feature distance before nonlinear mapping (lower = more similar).
//!
//! ## Determinism
//!
//! Deterministic for the same input on the same architecture. Cross-architecture
//! results (e.g. AVX2 vs scalar vs AVX-512) may differ by small ULP due to
//! different FMA contraction behavior.
//!
//! ## Design
//!
//! - **XYB color space** — cube root LMS, same perceptual space as ssimulacra2/butteraugli
//! - **Modified SSIM** — ssimulacra2's variant: drops the luminance denominator
//!   (no C1), uses `1 - (mu1-mu2)²` directly. Correct for perceptually-uniform spaces.
//! - **13 features per channel per scale** — SSIM (3 pooling norms), edge artifact/detail
//!   loss (3 norms each), MSE, and 3 high-frequency energy/magnitude features
//! - **4-scale pyramid** — 1×, 2×, 4×, 8× via box downscale (ssimulacra2 uses 6)
//! - **O(1)-per-pixel box blur** — single-pass with fused SIMD kernel
//! - **156 trained weights** — optimized on 149.5k synthetic pairs across 4 codecs
//! - **AVX2/AVX-512 SIMD** throughout via [archmage](https://crates.io/crates/archmage)
//!
//! See the `metric` module source for the full feature extraction math.

#![forbid(unsafe_code)]

mod blur;
mod color;
mod error;
mod fused;
pub mod mapping;
mod metric;
mod pool;
pub mod profile;
mod simd_ops;
pub mod source;
mod streaming;

// --- Primary API ---
pub use error::ZensimError;
pub use metric::{
    ErrorCategory, FeatureView, RoundingBias, Zensim, ZensimResult, dissimilarity_to_score,
    score_to_dissimilarity,
};

// --- Classification API (used by zensim-regress for regression testing) ---
pub use metric::{AlphaStratifiedStats, ClassifiedResult, DeltaStats, ErrorClassification};
pub use profile::ZensimProfile;
pub use source::{
    AlphaMode, ColorPrimaries, ImageSource, PixelFormat, RgbSlice, RgbaSlice, StridedBytes,
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
    BlurKernel, CH_B, CH_X, CH_Y, DownscaleFilter, FEATURES_PER_CHANNEL_BASIC,
    FEATURES_PER_CHANNEL_EXTENDED, FEATURES_PER_CHANNEL_WITH_PEAKS, FEATURES_PER_SCALE, WEIGHTS,
    ZensimConfig, compute_zensim_with_config, compute_zensim_with_ref_and_config,
    precompute_reference_with_scales, score_from_features,
};

/// Number of downscale levels. Each level halves resolution.
/// 4 scales covers 1x, 2x, 4x, 8x — sufficient for most perceptual effects.
pub(crate) const NUM_SCALES: usize = 4;
