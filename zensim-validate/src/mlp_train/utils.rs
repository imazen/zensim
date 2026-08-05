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
/// Rows are any iterator of row slices, so both storage shapes share this
/// one owner: per-row tables pass `rows.iter().map(|r| r.as_slice())`, the
/// trainer's flat per-group buffers pass `flat.chunks(n_features)`. Single
/// pass over the rows (the historical per-feature outer loop re-walked the
/// table once per non-identity transform); per-feature counts and their
/// report order are unchanged.
pub fn sweep_nan_inf<'r>(
    rows: impl IntoIterator<Item = &'r [f64]>,
    transforms: &[FeatureTransform],
    source_name: &str,
) -> Result<(), String> {
    let n_features = transforms.len();
    let checked: Vec<usize> = (0..n_features)
        .filter(|&fi| !matches!(transforms[fi], FeatureTransform::Identity))
        .collect();
    let mut bad = vec![0usize; n_features];
    if !checked.is_empty() {
        for row in rows {
            for &fi in &checked {
                if fi < row.len() && !row[fi].is_finite() {
                    bad[fi] += 1;
                }
            }
        }
    }
    let poisoned: Vec<(usize, usize, &str)> = checked
        .into_iter()
        .filter(|&fi| bad[fi] > 0)
        .map(|fi| (fi, bad[fi], transforms[fi].as_token()))
        .collect();
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
