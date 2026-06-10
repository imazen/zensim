//! Error types for zensim image comparison.

/// Errors from zensim computation.
///
/// This enum is `#[non_exhaustive]` — additional variants may be added in
/// future minor releases without a major version bump. Match with a `_`
/// arm to remain forward-compatible.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ZensimError {
    /// Source and distorted images have different pixel counts.
    #[error("Source and distorted images must have the same dimensions")]
    DimensionMismatch,

    /// Width or height is less than 8. The multi-scale pyramid requires
    /// at least 8×8 pixels to produce meaningful results.
    #[error("Images must be at least 8x8 pixels")]
    ImageTooSmall,

    /// `pixels.len()` does not equal `width * height`.
    #[error("Image data length does not match width * height")]
    InvalidDataLength,

    /// Row stride is smaller than `width * bytes_per_pixel`.
    #[error("Row stride is smaller than width * bytes_per_pixel")]
    InvalidStride,

    /// Image dimensions exceed the configured `max_pixels` cap, or the
    /// notional pixel/byte count overflows `usize` on the current target
    /// (e.g. `width * height` wraps on 32-bit / wasm32). Use
    /// [`Zensim::with_max_pixels`](crate::Zensim::with_max_pixels) to
    /// raise or remove the cap.
    #[error("Image dimensions exceed the configured maximum or overflow usize")]
    ImageTooLarge,

    /// `features.len()` does not equal `weights.len()` in
    /// [`try_score_from_features`](crate::try_score_from_features). Both
    /// slices must have the same length (one weight per feature).
    #[error("features and weights must have the same length")]
    FeatureWeightsLengthMismatch,

    /// A pixel format reached a code path that does not yet handle it.
    ///
    /// This currently fires from the optional `classification` feature's
    /// delta-stats pass when a format other than sRGB-8 (RGB/RGBA/BGRA),
    /// sRGB-16 (RGBA), or linear-F32 (RGBA) is presented. The enum is
    /// `#[non_exhaustive]`, so this variant is non-breaking — match it
    /// alongside a `_` arm if you handle it.
    #[error("pixel format is not supported by this code path")]
    UnsupportedPixelFormat,

    /// Loading a trained MLP bake's bytes failed.
    ///
    /// Distinct from [`Self::InvalidDataLength`] — covers
    /// header / version / structural parse failures, not feature-length
    /// mismatches. `reason` is a `&'static str` describing the parse
    /// stage that failed (kept static so the error type stays `Copy`).
    #[error("failed to load MLP bake: {reason}")]
    ModelLoadFailed { reason: &'static str },

    /// Running an MLP forward pass on a loaded bake failed.
    ///
    /// Covers shape mismatches between the bake's declared layers, the
    /// supplied features, and any per-sample-α / hybrid-head metadata
    /// (e.g. metadata declares `n_hidden=32` but the bake's final layer
    /// outputs 16 values). Distinct from [`Self::ModelLoadFailed`] —
    /// the bake parsed cleanly but the forward call could not produce
    /// a score. `reason` is a `&'static str` describing the failure
    /// site.
    #[error("MLP forward failed: {reason}")]
    ModelForwardFailed { reason: &'static str },

    /// An [`ImageSource`](crate::ImageSource) signaled an HDR transfer
    /// function ([`TransferFunction::Pq`](crate::TransferFunction::Pq)
    /// or [`TransferFunction::Hlg`](crate::TransferFunction::Hlg)),
    /// but zensim does not yet ship a validated HDR scoring path. Running
    /// the SDR pipeline on HDR-coded values would silently produce
    /// meaningless scores — we refuse the input instead.
    ///
    /// See [imazen/zensim#38](https://github.com/imazen/zensim/issues/38)
    /// for the HDR roadmap (PU-encoded XYB front-end + trained HDR profile
    /// against UPIQ + AIC-HDR2025). Until that lands, callers wanting to
    /// score HDR pairs must invert the transfer themselves and pass
    /// linear-light pixels with `TransferFunction::Linear` (and accept
    /// that the score is still SDR-trained — values outside [0, 1] are
    /// clamped by downstream XYB math).
    #[error("HDR transfer functions (PQ/HLG) are not yet supported — see imazen/zensim#38")]
    HdrInputNotYetSupported,

    /// Source and distorted images signaled different transfer functions.
    /// Comparing across transfer spaces is undefined — caller must convert
    /// to a common transfer before scoring.
    #[error("Source and distorted transfer functions must match")]
    TransferFunctionMismatch,
}

/// Pixel format conversion error from the zenpixels adapter.
#[cfg(feature = "zenpixels")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("Unsupported pixel format: {0}")]
pub struct UnsupportedFormat(pub &'static str);
