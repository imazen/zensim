/// Tie-correct ranks (midrank averaging over equal values) — a verified-equivalent
/// mirror of `zenstats::panel::ranks`. This example lives in the `zensim` crate,
/// which does not depend on `zenmetrics`/`zenstats`, so the stat is mirrored here
/// rather than adding a cross-repo dev-dep for one diagnostic binary. The previous
/// body assigned the raw sort position, which is WRONG for ties: block sums tie
/// often (flat/clamped blocks share a diffmap value), and distinct ranks on tied
/// inputs bias the Spearman that M3 reports. Midrank matches the canonical panel.
fn rank(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (v[idx[j]] - v[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let avg = (i + j - 1) as f64 / 2.0; // midrank over the tie block [i, j)
        for &ix in &idx[i..j] {
            r[ix] = avg;
        }
        i = j;
    }
    r
}
pub(super) fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt() + 1e-12)
}
pub(super) fn spearman(a: &[f64], b: &[f64]) -> f64 {
    pearson(&rank(a), &rank(b))
}
