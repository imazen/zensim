use zenpredict::FeatureTransform;

pub fn spearman_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let mean_a = (n as f64 - 1.0) / 2.0;
    let mean_b = mean_a;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = ra[i] - mean_a;
        let xb = rb[i] - mean_b;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

pub fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (v[idx[j]] - v[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let avg = (i + j - 1) as f64 / 2.0;
        for k in i..j {
            r[idx[k]] = avg;
        }
        i = j;
    }
    r
}

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
