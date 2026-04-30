//! MLP runtime — load + forward pass for small dense networks.
//!
//! This module is the inference path for [`ZensimProfile::PreviewV0_4`](crate::profile::ZensimProfile),
//! which scores feature vectors through a small MLP rather than the
//! linear dot product used by V0_2. The on-disk format is ZNPR v2 — a
//! packed binary shipped via `include_bytes!` from the trained weights
//! file.
//!
//! # Backed by [`zenpredict`]
//!
//! All types in this module are re-exports from the
//! [`zenpredict`](https://github.com/imazen/zenanalyze) crate, which
//! is the canonical MLP runtime for the imazen image-codec ecosystem
//! (zenjpeg / zenwebp / zenavif / zenjxl picker selection + zensim
//! V0_4 perceptual scoring). The format and dispatch math live there;
//! we only re-export the surface we use here so consumer code can
//! say `zensim::mlp::Predictor` without a separate `use zenpredict`.
//!
//! # License note
//!
//! `zenpredict` is `AGPL-3.0-only OR LicenseRef-Imazen-Commercial`.
//! Adding it as a non-optional dependency means binaries that link
//! zensim's V0_4 path acquire AGPL obligations (or need a commercial
//! license from Imazen). zensim itself remains MIT/Apache-2.0 — the
//! V0_2 linear path has no `zenpredict` dependency.
//!
//! # Usage sketch
//!
//! ```ignore
//! use zensim::mlp::{Model, Predictor};
//!
//! static MODEL_BYTES: &[u8] = include_bytes!("../../weights/v04.bin");
//!
//! let model = Model::from_bytes(MODEL_BYTES)?;
//! let mut p = Predictor::new(model);
//! let distance = p.predict(&features)?[0];
//! ```

pub use zenpredict::{
    Activation, FORMAT_VERSION, FeatureBound, Header, LEAKY_RELU_ALPHA, LayerEntry, LayerView,
    Metadata, MetadataEntry, MetadataType, Model, Predictor, Section, WeightDtype, WeightStorage,
    keys,
};

/// Errors raised by [`Model::from_bytes`] and the forward pass. Alias
/// of [`zenpredict::PredictError`] for source compatibility with
/// earlier zensim versions; new code should use the alias name.
pub type MlpError = zenpredict::PredictError;

/// ZNPR v2 byte-stream composer. Used by `zensim-validate`'s
/// `--algorithm mlp` arm to bake trained weights, and by zensim's
/// V0_4 placeholder.
pub mod bake {
    pub use zenpredict::bake::{BakeError, BakeLayer, BakeMetadataEntry, BakeRequest, bake_v2};
}
