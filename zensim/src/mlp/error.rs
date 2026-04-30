//! Errors from MLP model loading and inference.
//!
//! Vendored from `zenpicker::error` v0.1.0 (originally
//! AGPL-3.0-only OR LicenseRef-Imazen-Commercial), re-licensed under
//! zensim's MIT OR Apache-2.0 by the copyright holder. See
//! [`crate::mlp`] for the full provenance note.

use core::fmt;

/// Errors raised by [`Model::from_bytes`](super::Model::from_bytes)
/// and the forward pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum MlpError {
    /// Header magic bytes don't match `ZNPK`.
    #[error("mlp: bad magic, expected ZNPK, found {found:?}")]
    BadMagic { found: [u8; 4] },

    /// Format version not supported by this build.
    #[error("mlp: format version {version} not supported (max supported: {max_supported})")]
    UnsupportedVersion { version: u16, max_supported: u16 },

    /// Header advertises a `header_size` smaller than the v1 minimum.
    #[error("mlp: header_size {advertised} < minimum {min}")]
    HeaderTooSmall { advertised: u16, min: u16 },

    /// Bytes ran out before the model was fully parsed.
    #[error("mlp: truncated at offset {offset}, wanted {want} bytes, have {have}")]
    Truncated {
        offset: usize,
        want: usize,
        have: usize,
    },

    /// `weight_dtype` byte was not 0 (f32), 1 (f16), or 2 (i8).
    #[error("mlp: unknown weight dtype byte {byte:#x}")]
    UnknownWeightDtype { byte: u8 },

    /// `activation` byte was not a recognized variant.
    #[error("mlp: unknown activation byte {byte:#x}")]
    UnknownActivation { byte: u8 },

    /// Layer's `in_dim` doesn't match the prior layer's `out_dim`
    /// (or, for layer 0, doesn't match the model's `n_inputs`).
    #[error("mlp: layer {layer} expected in_dim {expected_in}, got {got_in}")]
    LayerDimMismatch {
        layer: usize,
        expected_in: usize,
        got_in: usize,
    },

    /// The final layer's `out_dim` doesn't match the header's
    /// `n_outputs`.
    #[error("mlp: final layer out_dim {got} != header n_outputs {expected}")]
    OutputDimMismatch { expected: usize, got: usize },

    /// A header dimension was zero where it must be positive.
    #[error("mlp: zero dimension in `{what}`")]
    ZeroDimension { what: &'static str },

    /// Caller passed a feature vector of the wrong length.
    #[error("mlp: feature vector length {got} != n_inputs {expected}")]
    FeatureLenMismatch { expected: usize, got: usize },
}

// thiserror::Error already provides the std::error::Error impl when
// std is in scope; we keep an explicit Display implementation only
// when callers want to format errors themselves without importing
// thiserror's prelude. The derive on the enum produces both Display
// and Error.
const _: fn() = || {
    fn _assert_send_sync<T: Send + Sync>() {}
    _assert_send_sync::<MlpError>();
    fn _assert_display<T: fmt::Display>() {}
    _assert_display::<MlpError>();
};
