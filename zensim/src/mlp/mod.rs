//! MLP runtime re-exports for the adopted model. The packed format and
//! forward pass belong to `zenpredict`; this module is crate-private.

pub(crate) use zenpredict::{Model, Predictor};
/// Layer-weight readers used by per-model feature planning.
pub(crate) use zenpredict::{WeightStorage, f16_bits_to_f32};
