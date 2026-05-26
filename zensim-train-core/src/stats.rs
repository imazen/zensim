//! Bit-exact port of rank/correlation helpers from
//! `zensim-validate/src/mlp_train.rs`. Pure f64, no SIMD, no allocator
//! tricks — bit-identical to the validate-time implementation.
//!
//! Dedup-K (2026-05-26): the canonical Mohammadi-2025-paper-correct
//! panel lives in the `zenstats` crate (zenmetrics workspace). This
//! module is intentionally kept independent because `zensim-train-core`
//! must compile on `wasm32-unknown-unknown` for the in-browser trainer,
//! and the `zenstats` crate is not yet WASM-vetted. The math is
//! identical to `zenstats::{pearson, ranks, spearman}` (verified by
//! `tests/test_zen_stats_rust_python_parity.py` in
//! `scripts/canonical_corpus/` once shipped). If a future change
//! splits the algorithms — that is a bug; both impls must be kept in
//! lock-step.

/// Pearson correlation between two slices of equal length. Returns 0
/// if either has zero variance.
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let xa = xi - mx;
        let xb = yi - my;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

/// Convert a slice of f64 to fractional ranks with mid-rank tie handling.
pub fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        v[a].partial_cmp(&v[b])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
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

/// Spearman rank correlation. Equivalent to Pearson on ranks.
pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}
