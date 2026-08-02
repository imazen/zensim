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
//! let z = Zensim::new(ZensimProfile::codec_target());
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
//! let z = Zensim::new(ZensimProfile::codec_target());
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
//! ## Encoder closed-loop pattern (per-window quality probe)
//!
//! For codec quantization controllers that want a per-window quality signal
//! mid-encode (rather than one global score post-encode), pre-slice the
//! source into row-windows that match the encoder's natural emission
//! cadence, build a [`PrecomputedReference`] per window once, then call
//! [`compute_with_ref_into`](crate::Zensim::compute_with_ref_into) per
//! window with a shared [`ZensimScratch`] to amortize allocations.
//!
//! ```
//! use zensim::{Zensim, ZensimProfile, RgbSlice, ZensimScratch};
//! # let (source_pixels, width, height) = (vec![[0u8; 3]; 16 * 8], 16usize, 8usize);
//! # const WINDOW_ROWS: usize = 4;
//! let z = Zensim::new(ZensimProfile::codec_target());
//!
//! // Up front: pre-slice the source into row-windows. Build one
//! // PrecomputedReference per window (each is small — only that
//! // window's pyramid).
//! let mut window_refs = Vec::new();
//! let mut y = 0;
//! while y < height {
//!     let h = (height - y).min(WINDOW_ROWS);
//!     if h < 8 { break; }  // skip tail slivers (they'd reflect-pad, costing accuracy)
//!     let start = y * width;
//!     let end = start + h * width;
//!     let win = RgbSlice::new(&source_pixels[start..end], width, h);
//!     window_refs.push((y, h, z.precompute_reference(&win)?));
//!     y += WINDOW_ROWS;
//! }
//!
//! // Per encode iMCU window i: compute canonical zensim on the slice,
//! // reusing scratch buffers across all calls.
//! let mut scratch = ZensimScratch::new();
//! for (y0, h, pre) in &window_refs {
//!     # let distorted_window: Vec<[u8; 3]> = vec![[0u8; 3]; (*h) * width];
//!     let dst = RgbSlice::new(&distorted_window, width, *h);
//!     let result = z.compute_with_ref_into(pre, &dst, &mut scratch)?;
//!     // Feed result.score() into your AQ controller…
//!     let _ = result.score();
//! }
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ### Caveats — read before deploying as a control signal
//!
//! Empirical validation against full-image diffmap as ground-truth (see
//! parked branch `explored/issue-16-option-d-slice-canonical-and-zenwebp-data`
//! and the `slice_localized_distortion.rs` / `slice_real_codec_localization.rs`
//! examples in `zensim-regress`):
//!
//! - **Truncated pyramid context.** A 64-row window's pyramid is 64→32→16→8
//!   rows at scales 0–3. SSIM uses an 11×11 window (radius 5), so scale 3
//!   has 0 valid SSIM rows in 8 input rows; scale 2 has ~6 valid rows of 16.
//!   The features at coarser scales are dominated by mirror-padded boundary
//!   data, not content. The trained weights at scales 1–3 carry ~94% of
//!   the weight mass.
//!
//! - **On *synthetic* injected-distortion** (one window heavily damaged,
//!   others clean): per-window canonical correctly identifies the damaged
//!   window 100% of the time. The clean-vs-damaged gap dwarfs any
//!   pad-noise contribution.
//!
//! - **On *real* codec output** (mozjpeg / zenjpeg sRGB / zenjpeg XYB /
//!   zenavif at q60–q90): per-window canonical's top-1 ranking matches
//!   ground-truth (full-image diffmap aggregated per window) only ~24%
//!   of the time. Top-3 overlap is 1.71/3 ≈ 57%. Mean SROCC = 0.57.
//!
//! - **Treat the per-window signal as a *guidance* signal, not a per-window
//!   precision oracle.** Pair with EMA smoothing across windows and an
//!   iteration-boundary canonical check ([`Zensim::compute_with_ref`])
//!   for the global score.
//!
//! - **Cost.** With K windows of ~`H/K` rows each, total per-iteration
//!   compute is roughly the same as one full-image canonical compute,
//!   distributed across the encoder's iMCU emissions. The
//!   [`PrecomputedReference`] builds amortize across all encoder
//!   iterations against the same source.
//!
//! ## RGBA support
//!
//! ```
//! use zensim::{Zensim, ZensimProfile, RgbaSlice};
//! # let (src_rgba, dst_rgba) = (vec![[0u8; 4]; 64], vec![[0u8; 4]; 64]);
//! let z = Zensim::new(ZensimProfile::codec_target());
//! let source = RgbaSlice::new(&src_rgba, 8, 8);
//! let distorted = RgbaSlice::new(&dst_rgba, 8, 8);
//! let result = z.compute(&source, &distorted)?;
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ## zenpixels support
//!
//! With the `zenpixels` feature, any `zenpixels::PixelSlice` or
//! `zenpixels::PixelBuffer` can be used directly via `ZenpixelsSource`:
//!
//! ```ignore
//! use zensim::{Zensim, ZensimProfile, ZenpixelsSource};
//!
//! let source = ZenpixelsSource::try_from_slice(&pixel_slice)?;
//! let distorted = ZenpixelsSource::try_from_slice(&other_slice)?;
//! let result = Zensim::new(ZensimProfile::codec_target()).compute(&source, &distorted)?;
//! ```
//!
//! Supported: Rgb8, Rgba8, Bgra8, Rgbx8, Bgrx8, Rgba16, RgbaF32 (sRGB/BT.709/linear).
//! Premultiplied alpha is un-premultiplied automatically. RGBX/BGRX padding bytes
//! are treated as opaque automatically. HDR (PQ, HLG) and grayscale are rejected
//! with `UnsupportedFormat` (zenpixels feature).
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
//!   `ZenpixelsSource` (zenpixels `PixelSlice`/`PixelBuffer`, `zenpixels` feature),
//!   [`StridedBytes`] (any of `Srgb8Rgb`, `Srgb8Rgba`, `Srgb8Bgra`,
//!   `Srgb16Rgba`, `LinearF32Rgba`), or implement [`ImageSource`] directly.
//! - **Alpha:** RGBA inputs are composited over a deterministic noise
//!   background so alpha differences are detected without the structured-pattern
//!   amplification of a checkerboard. Supports `Straight` and `Opaque` alpha modes.
//! - **Dimensions:** Both images must be the same width × height. Any
//!   non-zero size scores: sub-64px inputs (down to 1×1) are reflect-padded
//!   to the pyramid minimum internally.
//!
//! ## Score semantics
//!
//! 100 = identical, higher = more similar. How a raw result becomes a
//! score depends on the profile:
//!
//! - [`ZensimProfile::B`] (the canonical profile — what
//!   [`ZensimProfile::codec_target`] and [`ZensimProfile::latest_preview`]
//!   return): a deterministic LINEAR core over the 372 features + a monotone
//!   PCHIP dial spline. Byte-reproducible (no training seed). SDR content;
//!   HDR (absolute-nits) content routes to [`ZensimProfile::BHdr`]
//!   automatically. Pathological inputs can score below 0 (the spline
//!   extrapolates past its knots).
//! - [`ZensimProfile::A`] (**deprecated** — the prior canonical profile,
//!   the v47 MLP): MLP forward pass + monotone PCHIP dial spline, identity
//!   at ≈ 97.7. Behind the default-on `deprecated-profiles` feature; disable
//!   it to drop `A`. Superseded by `B`.
//! - [`ZensimProfile::PreviewV0_1`] / [`ZensimProfile::PreviewV0_2`]
//!   (linear profiles): `100 - 18 × d^0.7` where `d` is the per-scale
//!   weighted feature distance. Calibrated from 0–100 on 344k training
//!   pairs; extreme distortions can score below 0. Externally-defined
//!   linear/MLP profiles are also reachable via [`ZensimProfile::Custom`]
//!   (reconstructed in the `zensim-experimental` crate).
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

/// Attribution-density steering map (research; `custom-profiles`).
#[cfg(feature = "custom-profiles")]
mod attribution;
mod blur;
mod color;
mod diffmap;
mod error;
mod fused;
mod iw_pool;
pub mod mapping;
mod metric;
// MLP runtime is internal — V0_4+ dispatch is exposed only through
// `ZensimProfile::PreviewV0_*` variants. Consumers wanting to bake or
// load custom MLP weights should depend on the `zenpredict` crate
// directly. zenpredict is MIT/Apache-2.0 (matching zensim), so the
// runtime is an unconditional dependency in 0.3.0+.
pub(crate) mod mlp;
mod pool;
pub mod profile;
mod simd_ops;
pub mod source;
mod streaming;
// HDR foundation: transfer functions + display model (code values → absolute
// luminance). Still foundation-only — the PU entry points take already-linear
// cd/m², so code-value decoding stays with the caller until a code-value
// entry lands. See `docs/HDR_PLAN.md`.
mod transfer;
// PU21 perceptually-uniform encoding — the HDR-path replacement for the
// cube-root nonlinearity, consumed by `Zensim::compute_pu_linear{,_planar}`.
mod pu21;
// EX-4 extended feature modules — XYB/LMS front-end stats + CVVDP-shape
// per-pair signals (DKL, Weber-contrast band ratios, mutual-masking
// residuals, Minkowski β=3 pool). Gated behind the `training` feature
// because they're only meaningful inside the feature-extract pipeline;
// the metric hot path never calls them.
#[cfg(feature = "training")]
pub mod cvvdp_features;
#[cfg(feature = "training")]
pub mod xyb_lms_features;

// V2 "bounded" feature extraction — opt-in, strictly additive. See
// feature_v2.rs's module doc and docs/FEATURE_V2_SPEC_2026-07-18.md.
#[cfg(feature = "feature-regime-v2")]
pub mod feature_v2;

// Streaming strip-plane producer for the folded-720+append walk
// (docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md).
#[cfg(feature = "feature-regime-v2")]
pub(crate) mod feature_v2_stream;

// --- Primary API ---
/// Cooperative-cancellation vocabulary, re-exported from the
/// [`enough`](https://docs.rs/enough) crate for use with
/// [`Zensim::with_stop`]. Construct a real cancellable token with the
/// sibling [`almost-enough`](https://docs.rs/almost-enough) crate's
/// `Stopper` (any `impl Stop` works).
pub use enough::{Stop, StopReason, Unstoppable};
pub use error::ZensimError;
pub use metric::{
    FeatureView, Zensim, ZensimResult, dissimilarity_to_score, score_to_dissimilarity,
};

#[doc(hidden)]
pub use color::{bench_pu_xyb_dispatch, bench_pu_xyb_scalar};

/// Dev-only stage-level exports for per-stage SIMD benchmarking.
///
/// NOT part of the public API and NOT semver-covered — same status as the
/// `bench_pu_xyb_*` exports above, which set this precedent.
///
/// These exist because the whole-pipeline tier bench established that NEON is
/// worth only ~1.26x here while butteraugli — a comparable multi-scale
/// XYB+blur metric — gets 3.4x on the same host, and that the shortfall is
/// inside the image kernels rather than in scoring. Attributing it further
/// needs per-stage numbers, and on macOS the usual profilers (dtrace via
/// cargo-flamegraph) require SIP to be disabled. Exporting the stage entry
/// points lets `zensim-bench`'s `stage_isolation` bench get the same
/// attribution with an A/B instead of a profiler.
#[doc(hidden)]
pub mod __bench_stages {
    pub use crate::blur::{
        box_blur_1pass_into, box_spread_merge_f32, downscale_2x_into, fused_blur_h_ssim,
    };
    pub use crate::color::srgb_to_positive_xyb_planar_into;
    pub use crate::simd_ops::{abs_diff_sum, mul_into, sq_diff_sum, sq_sum_into};
}
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
    AlphaMode, ColorPrimaries, GamutMapping, ImageSource, PixelFormat, RgbSlice, RgbaSlice,
    StridedBytes,
};

#[cfg(feature = "custom-profiles")]
pub use attribution::{AttributionResult, AttributionSession};
pub use diffmap::{DiffmapOptions, DiffmapResult, DiffmapWeighting};
pub use streaming::{PrecomputedReference, ZensimScratch};

/// Score a precomputed feature vector under a [`ZensimProfile`] —
/// the entry point alternative feature backends (e.g. `zensim-gpu`)
/// use to produce a bit-exact CPU-equivalent `0..100` score.
///
/// Runs the full bake forward pass (per-sample-α / hybrid head,
/// tanh-pin, PCHIP spline, per-codec affine, clamp / soft-clamp /
/// extrapolate disposition) — same dispatch the canonical
/// `Zensim::compute(...)` flow applies after feature extraction.
pub use metric::{score_features_with_profile, score_features_with_profile_and_codec};

/// Training/research API — requires `features = ["training"]`.
///
/// These items expose metric internals (blur kernel shape, scale count,
/// masking, weight vectors) that change metric behavior. Scores produced
/// with non-default `ZensimConfig` are **not comparable** to the default
/// trained weights or the 0-100 score scale.
///
/// The historical `score_from_features` (the panicking variant) was
/// removed in 0.3.0; use [`try_score_from_features`] instead.
#[cfg(feature = "training")]
pub use metric::{
    BlurKernel, CH_B, CH_X, CH_Y, DownscaleFilter, FEATURES_PER_CHANNEL_BASIC,
    FEATURES_PER_CHANNEL_EXTENDED, FEATURES_PER_CHANNEL_WITH_PEAKS, FEATURES_PER_SCALE, WEIGHTS,
    ZensimConfig, compute_zensim_with_config, compute_zensim_with_ref_and_config,
    precompute_reference_with_scales, try_score_from_features,
};

/// IW-weight estimator types — requires `features = ["training"]`.
///
/// Exposes the Wang & Li 2011 info-content weight estimators
/// ([`iw_pool::IwWeightKind`], [`iw_pool::IwWeightConfig`]) and the
/// `compute_iw_weights` entry point used for research experiments.
/// Including the steerable-pyramid GSM approximation spike added
/// 2026-05-15 — see `benchmarks/iw_pyramid_spike_methodology_2026-05-15.md`.
///
/// `WeightedPool` and `IwSsimFeatures` (the offline-experiment pool +
/// pooled-feature struct) are crate-internal as of 0.3.0 — the
/// streaming hot path emits the same numbers directly via
/// `streaming::process_strip_into_accum`, and no external workspace
/// caller used them.
#[cfg(feature = "training")]
pub use iw_pool::{IwWeightConfig, IwWeightKind, compute_iw_weights};

#[cfg(feature = "zenpixels")]
mod zenpixels_compat;
#[cfg(feature = "zenpixels")]
pub use error::UnsupportedFormat;
#[cfg(feature = "zenpixels")]
pub use zenpixels_compat::ZenpixelsSource;

/// Number of downscale levels. Each level halves resolution.
/// 4 scales covers 1x, 2x, 4x, 8x — sufficient for most perceptual effects.
pub(crate) const NUM_SCALES: usize = 4;
