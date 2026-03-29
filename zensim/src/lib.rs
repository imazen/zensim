//! # zensim
//!
//! Fast psychovisual image similarity metric combining ideas from
//! SSIMULACRA2 and butteraugli. Multi-scale SSIM + edge + high-frequency
//! features in XYB color space, with trained weights and AVX2/AVX-512 SIMD.
//!
//! ## Quick start
//!
//! ```
//! use zensim::{Zensim, ZensimProfile, RgbSlice};
//! # let (src_pixels, dst_pixels) = (vec![[0u8; 3]; 64], vec![[0u8; 3]; 64]);
//! let z = Zensim::new(ZensimProfile::latest());
//! let source = RgbSlice::new(&src_pixels, 8, 8);
//! let distorted = RgbSlice::new(&dst_pixels, 8, 8);
//! let result = z.compute(&source, &distorted)?;
//! println!("{}: {:.2}", result.profile(), result.score());
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ## Batch comparison (one reference, many distorted)
//!
//! ```
//! use zensim::{Zensim, ZensimProfile, RgbSlice};
//! # let (ref_pixels, width, height) = (vec![[0u8; 3]; 64], 8usize, 8usize);
//! # let distorted_images: Vec<Vec<[u8; 3]>> = vec![];
//! let z = Zensim::new(ZensimProfile::latest());
//! let source = RgbSlice::new(&ref_pixels, width, height);
//! let precomputed = z.precompute_reference(&source)?;
//! for dst_pixels in &distorted_images {
//!     let dst = RgbSlice::new(dst_pixels, width, height);
//!     let result = z.compute_with_ref(&precomputed, &dst)?;
//!     println!("score: {:.2}", result.score());
//! }
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ## RGBA support
//!
//! ```
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
//! - **Color space:** All inputs must be **sRGB-encoded** (gamma ~2.2) — the
//!   standard output of JPEG, PNG, and WebP decoders. For linear-light data,
//!   use `PixelFormat::LinearF32Rgba` via [`StridedBytes`].
//! - **Wide gamut:** Display P3 and BT.2020 primaries are accepted via
//!   [`ColorPrimaries`] on [`StridedBytes`] — gamut-mapped to sRGB internally.
//!   Passing wide-gamut data as sRGB will produce incorrect scores.
//! - **Pixel formats:** [`RgbSlice`] (sRGB u8), [`RgbaSlice`] (sRGB u8 + alpha),
//!   `imgref::ImgRef` (sRGB u8, stride-aware, default feature),
//!   [`StridedBytes`] (any of `Srgb8Rgb`, `Srgb8Rgba`, `Srgb8Bgra`,
//!   `Srgb16Rgba`, `LinearF32Rgba`), or implement [`ImageSource`] directly.
//! - **Alpha:** RGBA inputs are composited over a deterministic noise
//!   background so alpha differences are detected without the structured-pattern
//!   amplification of a checkerboard. Supports `Straight` and `Opaque` alpha modes.
//! - **Dimensions:** Both images must be the same width × height, minimum 8×8.
//!
//! ## Score semantics
//!
//! 100 = identical, higher = more similar. Score mapping:
//! `100 - 18 × d^0.7` where `d` is the per-scale weighted feature distance.
//! Calibrated from 0–100 on 344k training pairs; extreme distortions can
//! score below 0 (uncalibrated outside the training range).
//!
//! [`ZensimResult`] also provides [`approx_ssim2()`](ZensimResult::approx_ssim2),
//! [`approx_dssim()`](ZensimResult::approx_dssim), and
//! [`approx_butteraugli()`](ZensimResult::approx_butteraugli) for direct
//! metric approximations. The [`mapping`] module has bidirectional interpolation
//! tables for score-level conversions.
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
//! - **19 features per channel per scale** — 13 basic (SSIM, edge artifact/detail
//!   loss, MSE, high-frequency) + 6 peak features, all scored
//! - **4-scale pyramid** — 1×, 2×, 4×, 8× via box downscale (ssimulacra2 uses 6)
//! - **O(1)-per-pixel box blur** — single-pass with fused SIMD kernel
//! - **228 trained weights** — optimized on 344k synthetic pairs across 6 codecs
//! - **AVX2/AVX-512 SIMD** throughout via [archmage](https://crates.io/crates/archmage)
//!
//! See the `metric` module source for the full feature extraction math.

#![forbid(unsafe_code)]

mod blur;
mod color;
mod diffmap;
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
    FeatureView, Zensim, ZensimResult, dissimilarity_to_score, score_to_dissimilarity,
};

/// Classification API — requires `features = ["classification"]`.
///
/// Exposes `classify()`, error categorization, and per-pixel delta statistics
/// for regression testing workflows.
#[cfg(feature = "classification")]
pub use metric::{
    AlphaStratifiedStats, ClassifiedResult, DeltaStats, ErrorCategory, ErrorClassification,
    RoundingBias,
};
pub use profile::ZensimProfile;
pub use source::{
    AlphaMode, ColorPrimaries, ImageSource, PixelFormat, RgbSlice, RgbaSlice, StridedBytes,
};

pub use diffmap::{DiffmapOptions, DiffmapResult, DiffmapWeighting};
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
