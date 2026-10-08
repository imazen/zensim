//! Explicit E32 cached-table diagnostic, never a serving/extraction route.
//! Network execution, gather, metadata parsing and arithmetic retain their owners.
use zenpredict::{Model, Predictor};
use zensim_validate::bake_runtime::CallerGather;

pub(super) struct PaletteScorer<'a> {
    model: &'a Model,
    predictor: Predictor<'a>,
    gather: CallerGather,
    input: Vec<f32>,
    metadata: zensim::bake_metadata::ScoreMetadata,
}

impl<'a> PaletteScorer<'a> {
    pub(super) fn new(model: &'a Model) -> Result<Self, String> {
        let repro: serde_json::Value = serde_json::from_str(
            model
                .metadata()
                .get_utf8("zentrain.repro")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let ids: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../benchmarks/e32_palette_feature_ids_2026-10-07.json"
        ))
        .map_err(|e| e.to_string())?;
        let expected: Vec<u16> =
            serde_json::from_value(ids["arm_ids"].clone()).map_err(|e| e.to_string())?;
        let declared = zensim::declared_feature_ids(model);
        let admission = &repro["table_admission"];
        if model.metadata().get_utf8("zentrain.feature_set_id").ok()
            != Some("basic+peaks+v2+palette@w1867/e32_palette_v2#2ec9dcda")
            || model.metadata().get_utf8("zentrain.formula_revision").ok() != Some("5")
            || admission["qualified_provenance"] != true
            || admission["research_family"] != "palette_v2"
            || admission["serving_allowed"] != false
            || admission["formula_revision"] != 5
            || !admission["sampling"].is_null()
            || !admission["historical_replay"].is_null()
            || repro["keep_features_n"] != 462
            || !repro["argv"]
                .as_array()
                .is_some_and(|args| args.iter().any(|a| a == "--nonneg-distance"))
            || model.n_outputs() != 1
            || (model.metadata().get("zentrain.feature_ids").is_some() && declared.is_none())
            || match declared.as_ref() {
                Some(actual) => actual != &expected || model.caller_input_width() != 462,
                None => model.caller_input_width() != 1867,
            }
        {
            return Err("cached palette research requires the registered Rev5, 462-ID, admitted scalar model".into());
        }
        let metadata =
            zensim::bake_metadata::parse_bake_metadata(model).map_err(|e| e.to_string())?;
        if metadata.per_sample_alpha.is_some()
            || metadata.hybrid_head.is_some()
            || metadata.minmax_head.is_some()
            || model
                .metadata()
                .get("zentrain.per_codec_calibration")
                .is_some()
        {
            return Err("cached palette research refuses alternate heads and compositions".into());
        }
        Ok(Self {
            model,
            predictor: Predictor::new(model),
            gather: CallerGather::for_model(model),
            input: vec![0.0; model.caller_input_width()],
            metadata,
        })
    }

    pub(super) fn score(&mut self, row: &[f64]) -> Result<f64, String> {
        if !self.gather.accepts_row_width(row.len(), self.input.len()) {
            return Err("cached palette row does not reach every declared input".into());
        }
        self.gather.fill(&mut self.input, row);
        let output = if self.model.has_nontrivial_feature_transforms() {
            self.predictor.predict_transformed(&self.input)
        } else {
            self.predictor.predict(&self.input)
        }
        .map_err(|e| e.to_string())?;
        Ok(scalar_tail(output[0] as f64, &self.metadata))
    }
}

pub(super) fn scalar_tail(mut score: f64, metadata: &zensim::bake_metadata::ScoreMetadata) -> f64 {
    if let Some(scale) = metadata.tanh_pin_scale {
        score =
            zensim::score_math::tanh_output_pin(score, scale, zensim::det_math::active_pow_form());
    }
    if let Some(spline) = &metadata.output_spline {
        score =
            zensim::score_math::pchip_eval_capped(score, &spline.xs, &spline.ys, &spline.derivs);
    }
    score
}
