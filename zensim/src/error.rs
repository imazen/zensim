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
}

/// Pixel format conversion error from the zenpixels adapter.
#[cfg(feature = "zenpixels")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("Unsupported pixel format: {0}")]
pub struct UnsupportedFormat(pub &'static str);
