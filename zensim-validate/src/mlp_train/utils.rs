use zenpredict::FeatureTransform;

// `spearman_correlation` was a byte-identical copy of `panel::spearman`
// (which now lives in zenstats via the panel re-export shim). Alias the
// panel function to preserve the historical `utils::spearman_correlation`
// name across 5+ mlp_train call sites without renaming any caller.
pub use crate::panel::spearman as spearman_correlation;

/// Post-transform NaN/inf sweep. Call after applying auto-transforms
/// to any feature pool (training groups, anchor parquet, pjnd
/// passthrough, konjnd aggregation). Returns `Err` listing every
/// poisoned `(feature_idx, count, transform_name)` triple so the
/// caller can exit with a clear diagnostic.
///
/// **Strict by default**: the trainer binary should propagate the error
/// and exit nonzero rather than silently clamping NaN to 0. A NaN in
/// post-transform features propagates to the scaler (poisoning mean/std),
/// the forward pass (NaN activations), and the loss (NaN gradients) —
/// the entire training run is unrecoverable.
pub fn sweep_nan_inf(
    rows: &[Vec<f64>],
    transforms: &[FeatureTransform],
    source_name: &str,
) -> Result<(), String> {
    let n_features = transforms.len();
    let mut poisoned: Vec<(usize, usize, &str)> = Vec::new();
    for fi in 0..n_features {
        let t = transforms[fi];
        if matches!(t, FeatureTransform::Identity) {
            continue;
        }
        let mut bad = 0usize;
        for row in rows {
            if fi < row.len() && !row[fi].is_finite() {
                bad += 1;
            }
        }
        if bad > 0 {
            poisoned.push((fi, bad, t.as_token()));
        }
    }
    if poisoned.is_empty() {
        Ok(())
    } else {
        let details: Vec<String> = poisoned
            .iter()
            .map(|&(fi, count, tname)| format!("f{fi} ({tname}): {count} NaN/inf"))
            .collect();
        Err(format!(
            "NaN/inf in {source_name} after auto-transforms: {}",
            details.join(", ")
        ))
    }
}
