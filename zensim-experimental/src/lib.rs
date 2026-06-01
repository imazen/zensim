//! # zensim-experimental
//!
//! Historical and experimental [`zensim`] metric profiles, preserved for
//! long-term research reference.
//!
//! These profiles and their trained MLP bakes were moved out of the
//! published `zensim` crate (2026-06-01) to keep its download size and
//! permanent API surface minimal. The published `zensim` keeps only the
//! shipping profiles (`A`, `PreviewV0_2`, the deprecated `PreviewV0_3`
//! alias). Everything else — the `PreviewV0_4` D2 ensemble, the whole
//! `PreviewV0_5*` SOTA-trail matrix, `A_Phone`, `PreviewV0_1`,
//! `LinearBounded`, and `PreviewV0_5Linear` — lives here.
//!
//! Every profile is reconstructed through zensim's stable
//! [`zensim::profile::ProfileParams::builder`] +
//! [`zensim::ZensimProfile::Custom`] extension point, pointing at a bake
//! embedded in this crate via `include_bytes!`. Because all the head /
//! spline / per-codec-calibration behaviour lives in the bake metadata,
//! each profile here is **bit-identical** to when it lived in `zensim`.
//!
//! This crate is **never published** (`publish = false`). It exists so the
//! research lineage stays runnable from the workspace — training, eval, and
//! regression tooling construct these profiles for comparison — without
//! shipping ~7.7 MB of historical bakes to every `zensim` consumer.
//!
//! ## Usage
//!
//! ```ignore
//! use zensim::{Zensim, RgbSlice};
//! let z = Zensim::new(zensim_experimental::preview_v0_5_tuner_v4());
//! // ... z.compute(&reference, &distorted)? as with any built-in profile.
//! ```
//!
//! The free functions return ready-to-use [`zensim::ZensimProfile::Custom`]
//! values whose `name()` matches the original built-in variant's name
//! string (e.g. `"zensim-preview-v0.5-tuner-v4"`), so existing tooling that
//! keys reports/filenames on the profile name keeps working unchanged.

#![forbid(unsafe_code)]

// Profile reconstructions are added in the next commit.
