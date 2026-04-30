//! High-level scorer wrapper around [`Model`] and [`forward`].
//!
//! Owns the parsed model + per-call scratch budget and exposes a
//! single [`score`](MlpScorer::score) entry point that takes the
//! 228-dim feature vector zensim produces and returns the scalar
//! `raw_distance`.

use super::error::MlpError;
use super::inference::forward;
use super::model::Model;

/// Cached, ready-to-call MLP scorer. Construct once per profile via
/// [`MlpScorer::from_bytes`] then reuse for every comparison.
///
/// Each [`score`](Self::score) call allocates a small scratch
/// buffer (`2 × scratch_len` f32). For tight loops, callers can
/// share a [`MlpScratch`] across calls via [`score_into`](Self::score_into).
#[derive(Debug)]
pub struct MlpScorer {
    model: Model<'static>,
    scratch_len: usize,
    n_inputs: usize,
    n_outputs: usize,
}

/// Reusable scratch buffers for [`MlpScorer::score_into`]. Allocate
/// once via [`MlpScratch::for_scorer`] and pass into every
/// `score_into` call to amortize the allocation.
#[derive(Debug)]
pub struct MlpScratch {
    a: Vec<f32>,
    b: Vec<f32>,
    output: Vec<f32>,
    /// f64→f32 staging for the input feature vector.
    features_f32: Vec<f32>,
}

impl MlpScratch {
    pub fn for_scorer(scorer: &MlpScorer) -> Self {
        Self {
            a: vec![0.0; scorer.scratch_len],
            b: vec![0.0; scorer.scratch_len],
            output: vec![0.0; scorer.n_outputs],
            features_f32: vec![0.0; scorer.n_inputs],
        }
    }
}

impl MlpScorer {
    /// Parse the MLP from `bytes` and pre-compute scratch sizing.
    /// `bytes` must outlive the scorer (typically `&'static`, e.g.
    /// from `include_bytes!` or a leaked `LazyLock<Vec<u8>>`).
    pub fn from_bytes(bytes: &'static [u8]) -> Result<Self, MlpError> {
        let model = Model::from_bytes(bytes)?;
        let scratch_len = model.scratch_len();
        let n_inputs = model.n_inputs();
        let n_outputs = model.n_outputs();
        Ok(Self {
            model,
            scratch_len,
            n_inputs,
            n_outputs,
        })
    }

    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Score `features` through the MLP, allocating scratch on each
    /// call. Returns the first output (n_outputs >= 1 enforced at
    /// load time).
    ///
    /// Features are converted from f64 to f32 element-wise before
    /// feeding into the network. Length must match
    /// [`n_inputs`](Self::n_inputs).
    pub fn score(&self, features: &[f64]) -> Result<f32, MlpError> {
        let mut scratch = MlpScratch::for_scorer(self);
        self.score_into(features, &mut scratch)
    }

    /// Score `features` through the MLP, reusing caller-owned
    /// scratch. Same return shape as [`score`](Self::score) but with
    /// no per-call allocation.
    pub fn score_into(&self, features: &[f64], scratch: &mut MlpScratch) -> Result<f32, MlpError> {
        if features.len() != self.n_inputs {
            return Err(MlpError::FeatureLenMismatch {
                expected: self.n_inputs,
                got: features.len(),
            });
        }
        for (dst, &src) in scratch.features_f32.iter_mut().zip(features.iter()) {
            *dst = src as f32;
        }
        forward(
            &self.model,
            &scratch.features_f32,
            &mut scratch.a,
            &mut scratch.b,
            &mut scratch.output,
        )?;
        Ok(scratch.output[0])
    }
}
