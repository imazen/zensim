//! MLP runtime — load + forward pass for small dense networks.
//!
//! This module is the inference path for the deprecated
//! [`ZensimProfile::A`](crate::profile::ZensimProfile) (behind the default-on
//! `deprecated-profiles` feature), which scores feature vectors through a
//! small MLP rather than the linear dot product used by the current default
//! `B` and by V0_2. The on-disk format is ZNPR v3 — a
//! packed binary shipped via `include_bytes!` from the trained weights
//! file. (v2 bakes are still loadable for backwards compatibility, but
//! NEW bakes must be v3 per CLAUDE.md.)
//!
//! # Backed by [`zenpredict`]
//!
//! All types in this module are re-exports from the
//! [`zenpredict`](https://github.com/imazen/zenanalyze) crate, which
//! is the canonical MLP runtime for the imazen image-codec ecosystem
//! (zenjpeg / zenwebp / zenavif / zenjxl picker selection + zensim
//! profile-`A` perceptual scoring). The format and dispatch math live
//! there; this module re-exports only what `zensim::metric` /
//! `zensim::profile` consume. The module is `pub(crate)` — external
//! consumers wanting to bake or load MLP models depend on `zenpredict`
//! directly.
//!
//! # License note
//!
//! `zenpredict` is `MIT OR Apache-2.0` — matches zensim. The runtime
//! is intentionally permissive so it can be embedded in any consumer.
//! Older notes claimed AGPL/commercial; that licensing plan never
//! shipped. zensim default builds remain MIT/Apache.
//!
//! # Usage sketch (crate-internal)
//!
//! ```ignore
//! use crate::mlp::{Model, Predictor};
//!
//! static MODEL_BYTES: &[u8] = include_bytes!("../../weights/v47_strict_qat_native_2026-05-27.bin");
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
/// Layer-weight readers — used only by the fold engine's per-profile
/// weight-skipping (`feature-regime-v2`), so re-exported under that gate to
/// keep a default build free of unused imports.
#[cfg(feature = "feature-regime-v2")]
pub(crate) use zenpredict::{WeightStorage, f16_bits_to_f32};
