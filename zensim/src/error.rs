#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ZensimError {
    #[error("Source and distorted images must have the same dimensions")]
    DimensionMismatch,

    #[error("Images must be at least 8x8 pixels")]
    ImageTooSmall,

    #[error("Image data length does not match width * height")]
    InvalidDataLength,
}
