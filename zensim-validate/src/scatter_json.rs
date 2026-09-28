//! Shared serialization of the canonical zenstats geometry; no statistic math.
pub(crate) fn assess(pred: &[f64], target: &[f64]) -> serde_json::Value {
    use serde_json::json;
    match zenstats::scatter::diagnose(pred, target) {
        Err(reason) => {
            json!({"schema":"scatter-v2", "status":"NOT_MEASURED", "reason":reason, "n":pred.len()})
        }
        Ok(s) => json!({
            "schema":"scatter-v2", "status":"MEASURED", "n":s.n,
            "normalized_pred":s.mapped_prediction,
            "geo":{
                "out4":s.outside_envelope,"p99d":s.shape_p99,"maxd":s.shape_max,
                "mad":s.shape_mad,"span":s.target_span,
                "cov":s.prediction_coverage,"clump":s.prediction_clump,
                "target_cov":s.target_coverage,"target_clump":s.target_clump,
                "clampLo":s.floor_mass,"clampHi":s.ceiling_mass
            },
            "raw":{
                "cov":s.raw_coverage,"clump":s.raw_clump,
                "p99":s.raw_chart_p99,"max":s.raw_chart_max,"mad":s.raw_mad,
                "residual_p99":s.raw_residual_p99,"residual_max":s.raw_residual_max,
                "pred_min":s.prediction_min,"pred_max":s.prediction_max,"slope":s.raw_slope
            }
        }),
    }
}
