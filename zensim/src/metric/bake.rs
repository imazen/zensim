//! Reusable candidate serving surface. Inference and metadata live in the
//! parent metric owner, shared with every named profile.

use super::*;
use std::sync::Arc;

/// A loaded bake's reusable scoring state.
///
/// The caller owns the parsed model; no bytes are leaked or installed into a
/// global loader. Create one scorer per worker and reuse it across image pairs
/// or feature rows. All baked heads, output splines and codec calibration run
/// through the same dispatch used by named profiles.
///
/// By default the bake's calibrated output is returned unchanged, including
/// negative scores. Model authors may select a different final disposition
/// with [`Self::with_score_disposition`]. This is model configuration; end users
/// still control one target score.
pub struct BakeScorer<'a> {
    model: &'a crate::mlp::Model,
    predictor: crate::mlp::Predictor<'a>,
    metadata: Arc<ScoreMetadata>,
    layout: crate::feature_layout::Layout,
    gathered: Vec<f64>,
    inputs: Vec<f32>,
    standardized: Vec<f64>,
    pixel_scratch: crate::feature_v2::V2Scratch,
    disposition: Option<&'a ProfileParams>,
    parallel: bool,
    finite_moments: bool,
    members: Vec<BakeScorer<'a>>,
    weights: Option<Vec<f64>>,
    /// `Plan::for_bake(self.model)`, built on first use. It depends only on
    /// the model (and the process formula revision, fixed per process), and a
    /// scorer's model never changes, so the cache needs no invalidation.
    /// Composition (members, weights, companion) stays per call in [`Self::plan`].
    model_plan: std::sync::OnceLock<crate::feature_plan::Plan>,
    /// [`Self::plan`]'s result, built on first use. Unlike `model_plan` it
    /// depends on the composition, so every change to `members`, `weights` or
    /// `corruption` resets it ([`Self::reset_plan`]).
    served_plan: std::sync::OnceLock<crate::feature_plan::Plan>,
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
            pixel_scratch: crate::feature_v2::V2Scratch::default(),
            disposition: None,
            parallel: true,
            finite_moments: false,
            members: Vec::new(),
            weights: None,
            model_plan: std::sync::OnceLock::new(),
            served_plan: std::sync::OnceLock::new(),
        })
    }

    /// Score an identity-layout feature row using the bake's declared IDs.
    ///
    /// The caller must supply features at the bake's extraction revision and
    /// decoder era. Table admission validates that contract before a batch.
    /// Dimensions supply optional size axes; a codec hint selects any baked
    /// per-codec calibration. A zero row is not evidence of identical pixels.
    ///
    /// # Errors
    /// Returns a named model error if the row cannot supply the declared inputs
    /// or the predictor cannot execute the model.
    pub fn score_features(
        &mut self,
        features: &[f64],
        width: u32,
        height: u32,
        codec_hint: Option<&str>,
    ) -> Result<f64, ZensimError> {
        let primary = if self.weights.as_ref().is_some_and(|w| w[0] == 0.0) {
            0.0
        } else {
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
            )?
        };
        let raw = if let Some(weights) = &self.weights {
            let mut total = 0.0;
            if weights[0] != 0.0 {
                total += weights[0] * primary;
            }
            for (member, &weight) in self.members.iter_mut().zip(&weights[1..]) {
                if weight != 0.0 {
                    total += weight * member.score_features(features, width, height, codec_hint)?;
                }
            }
            total
        } else if self.members.is_empty() {
            primary
        } else {
            let mut total = 0.0;
            total += primary;
            for member in &mut self.members {
                total += member.score_features(features, width, height, codec_hint)?;
            }
            total / (self.members.len() + 1) as f64
        };
        let score = self.disposition.map_or(raw, |p| dispose_mlp_raw(raw, p));
        Ok(score)
    }

}

#[cfg(test)]
mod revision_contract_tests {

    use crate::feature_defs::FormulaRevision;
    use crate::ssim_form::SsimLumaForm;
    // Everything that scores PIXELS through a bake needs `feature-regime-v2`:
    // `bake_declaring` fixes a v2 extraction requirement, so `BakeScorer::new`
    // refuses it without the feature and the pixel-path tests are gated.
    
    

    /// The premise the mismatch test below rests on, asserted rather than
    /// assumed: revisions 2 and 3 select the SAME luminance form. A refusal
    /// built on comparing luminance forms would wave this pair straight
    /// through, which is exactly why `check_pixel_revision` compares the
    /// REVISION and not only the form it selects.
    #[test]
    fn revisions_two_and_three_select_the_same_luminance_form() {
        assert_eq!(
            SsimLumaForm::for_revision(FormulaRevision::Rev2),
            SsimLumaForm::for_revision(FormulaRevision::Rev3),
            "the two-revisions-one-form case this contract exists for is gone; \
             re-derive the mismatch control against a pair that still shares a form"
        );
        assert_eq!(
            SsimLumaForm::for_revision(FormulaRevision::Rev3),
            SsimLumaForm::Clamp
        );
    }

}

