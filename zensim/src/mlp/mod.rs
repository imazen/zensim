//! MLP runtime — load + forward pass for small dense networks.
//!
//! This module is the inference path for [`ZensimProfile::V0_4`](crate::profile::ZensimProfile),
//! which scores feature vectors through a small MLP rather than the
//! linear dot product used by `V0_2`. The on-disk format is a packed
//! binary (`ZNPK` magic) shipped via `include_bytes!` from the trained
//! weights file. See [`model`] for the full layout.
//!
//! # Provenance
//!
//! Vendored from [zenpicker](https://github.com/imazen/zenanalyze)
//! v0.1.0 (originally `AGPL-3.0-only OR LicenseRef-Imazen-Commercial`),
//! re-licensed under zensim's `MIT OR Apache-2.0` by the copyright
//! holder (Imazen / Lilith River). The on-disk format is intentionally
//! kept identical to zenpicker's so bake tooling and round-trip checks
//! can be shared between the two crates.
//!
//! Vendored modules: [`error`], [`model`], [`inference`]. The
//! `picker`/`mask`/`rescue` layers from zenpicker are codec-side
//! concerns and are not vendored here.
//!
//! # Usage sketch
//!
//! ```ignore
//! use zensim::mlp::{Model, forward};
//!
//! static MODEL_BYTES: &[u8] = include_bytes!("../../weights/v04.bin");
//!
//! let model = Model::from_bytes(MODEL_BYTES)?;
//! let mut scratch_a = vec![0f32; model.scratch_len()];
//! let mut scratch_b = vec![0f32; model.scratch_len()];
//! let mut output = vec![0f32; model.n_outputs()];
//!
//! forward(&model, &features, &mut scratch_a, &mut scratch_b, &mut output)?;
//! ```

pub mod bake;
mod error;
mod inference;
mod model;
mod scorer;

#[cfg(test)]
mod tests;

pub use error::MlpError;
pub use inference::forward;
pub use model::{Activation, FORMAT_VERSION, LEAKY_RELU_ALPHA, LayerView, Model, WeightDtype, WeightStorage};
pub use scorer::{MlpScorer, MlpScratch};
