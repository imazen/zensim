//! MLP runtime — load + forward pass for small dense networks.
//!
//! This module is the inference path for [`ZensimProfile::PreviewV0_3`](crate::profile::ZensimProfile),
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
//! V0_3 perceptual scoring). The format and dispatch math live there;
//! we only re-export the surface we use here so consumer code can
//! say `zensim::mlp::Predictor` without a separate `use zenpredict`.
//!
//! # License note
//!
//! `zenpredict` is `MIT OR Apache-2.0` — matches zensim. The runtime
//! is intentionally permissive so it can be embedded in any consumer.
//! Older notes claimed AGPL/commercial; that licensing plan never
//! shipped. zensim default builds remain MIT/Apache.
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

// Internal-only re-exports of the zenpredict types we use. The module
// is `pub(crate)` (see `lib.rs`), so consumer crates that want to bake
// or load MLP models should depend on `zenpredict` directly. We
// re-export here only what `zensim::metric` and `zensim::profile`
// actually consume.
pub(crate) use zenpredict::{Model, Predictor};
