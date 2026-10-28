//! Reusable candidate serving surface. Inference and metadata live in the
//! parent metric owner, shared with every named profile.

use super::*;
use std::sync::Arc;

/// A loaded bake's scoring state: the parsed model, its predictor and the
/// reusable input buffers. Named profiles create one per call through
/// `forward_one_bake_with_codec`.
pub(crate) struct BakeScorer<'a> {
    model: &'a crate::mlp::Model,
    predictor: crate::mlp::Predictor<'a>,
    metadata: Arc<ScoreMetadata>,
    layout: crate::feature_layout::Layout,
    gathered: Vec<f64>,
    inputs: Vec<f32>,
    standardized: Vec<f64>,
}

impl<'a> BakeScorer<'a> {
    pub(super) fn with_metadata(
        model: &'a crate::mlp::Model,
        metadata: Arc<ScoreMetadata>,
    ) -> Result<Self, ZensimError> {
        crate::feature_layout::formula_revision(model)?;
        crate::sampling::Sampling::from_model(model)?;
        // A derived-input bake declares its read set, not one id per input;
        // `metadata` already validated its entries against that read set and
        // the caller width (`derived_inputs::from_model`).
        let declared_width = match metadata.derived_inputs.as_deref() {
            Some(d) => d.gather_width(),
            None => model.caller_input_width(),
        };
        if model.metadata().get("zentrain.feature_ids").is_some()
            && crate::feature_layout::declared_ids(model)
                .is_none_or(|ids| ids.len() != declared_width)
        {
            return Err(ZensimError::ModelLoadFailed {
                reason: "invalid zentrain.feature_ids declaration or input count",
            });
        }
        if let Some(head) = metadata.minmax_head.as_ref()
            && (head.n != model.n_inputs() || model.caller_input_width() != model.n_inputs())
        {
            return Err(ZensimError::ModelLoadFailed {
                reason: "min-max head width differs from model input width",
            });
        }
        Ok(Self {
            model,
            predictor: crate::mlp::Predictor::new(model),
            metadata,
            layout: crate::feature_layout::declared_layout(model),
            gathered: Vec::new(),
            inputs: Vec::with_capacity(model.caller_input_width()),
            standardized: Vec::new(),
        })
    }

    /// Score an identity-layout feature row using the bake's declared IDs.
    ///
    /// The caller must supply features at the bake's extraction revision.
    /// Dimensions supply optional size axes; a codec hint selects any baked
    /// per-codec calibration.
    ///
    /// # Errors
    /// Returns a named model error if the row cannot supply the declared inputs
    /// or the predictor cannot execute the model.
    pub(crate) fn score_features(
        &mut self,
        features: &[f64],
        width: u32,
        height: u32,
        codec_hint: Option<&str>,
    ) -> Result<f64, ZensimError> {
        forward_model_with_codec(
            self.model,
            &mut self.predictor,
            &self.metadata,
            &self.layout,
            &mut self.gathered,
            &mut self.inputs,
            &mut self.standardized,
            features,
            width,
            height,
            codec_hint,
        )
    }
}
