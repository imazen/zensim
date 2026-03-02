/// Errors from zensim computation.
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
}
