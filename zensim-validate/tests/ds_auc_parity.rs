//! DS-AUC moved from zensim's `zensim-validate/src/difference_auc.rs` to
//! `zenstats::pairs::ds_auc` (zenmetrics-stats) on 2026-10-11. This test
//! proves the move is bit-exact: the deleted implementation, kept below
//! verbatim as the oracle (it is not called anywhere else), and the owner
//! return the same bits on the fixtures the two entry points used
//! (`bake_verdict`'s DS-AUC unit cases, `panel evaluate`'s synthetic table,
//! the owner's brute-force case) and on tables large enough to take the
//! subsampled (`stride > 1`) path, with ties and non-finite rows.
//!
//! The oracle is a frozen copy of zensim main `35162acd`'s
//! `difference_auc.rs` body; never edit it, and never call it outside this
//! test.

use rayon::prelude::*;

/// Verbatim body of the removed `zensim-validate/src/difference_auc.rs`
/// `ds_auc` (zensim main `35162acd`). Test oracle only.
fn removed_zensim_ds_auc(predicted: &[f64], human: &[f64], diff_threshold: f64) -> f64 {
    let n = predicted.len();
    if n < 4 || human.len() != n {
        return f64::NAN;
    }
    // Cap pair count: with n up to ~10k, full O(n²) is 100M pairs.
    // Subsample to ~200k pairs deterministically via stride.
    let max_pairs = 200_000usize;
    let total_pairs = n * (n - 1) / 2;
    let stride = (total_pairs / max_pairs).max(1);

    // Collect (metric_gap, is_different) labels.
    //
    // Parallel over `i` (the outer pair index). Each `i` owns the contiguous
    // pair-index run `[base_i, base_i + (n-1-i))`, so its stride hits are a
    // pure function of `i` — the same pairs are selected as the sequential
    // sweep, and concatenating the per-`i` outputs in ascending `i` rebuilds
    // the identical `samples` vector. (It is then sorted, so even the order
    // would not matter; keeping it identical costs nothing and keeps the
    // tie-averaging below reading the same runs.)
    let mut samples: Vec<(f64, bool)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // pair_idx of (i, i+1) = sum_{k<i} (n-1-k) = i*(2n-i-1)/2
            let base = i * (2 * n - i - 1) / 2;
            let mut local: Vec<(f64, bool)> = Vec::new();
            for j in (i + 1)..n {
                let pair_idx = base + (j - i - 1);
                if pair_idx.is_multiple_of(stride) {
                    let metric_gap = (predicted[i] - predicted[j]).abs();
                    let human_gap = (human[i] - human[j]).abs();
                    if metric_gap.is_finite() && human_gap.is_finite() {
                        local.push((metric_gap, human_gap > diff_threshold));
                    }
                }
            }
            local
        })
        // Collect the per-`i` runs first, then concatenate in ascending `i`.
        // Explicit rather than `.flatten()` so the ordering guarantee is
        // syntactic, not a property of rayon's collect.
        .collect::<Vec<Vec<(f64, bool)>>>()
        .into_iter()
        .flatten()
        .collect();
    if samples.len() < 2 {
        return f64::NAN;
    }
    // AUC via rank-sum (Mann-Whitney U). Sort by metric_gap, sum ranks
    // of the "different" class.
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_diff = samples.iter().filter(|s| s.1).count();
    let n_same = samples.len() - n_diff;
    if n_diff == 0 || n_same == 0 {
        return f64::NAN;
    }
    // Average-rank for ties, then U-statistic.
    let mut rank_sum_diff = 0.0f64;
    let mut k = 0usize;
    while k < samples.len() {
        let mut m = k;
        while m + 1 < samples.len() && samples[m + 1].0 == samples[k].0 {
            m += 1;
        }
        // ranks k+1 .. m+1 (1-based), average:
        let avg_rank = ((k + 1) + (m + 1)) as f64 / 2.0;
        for s in &samples[k..=m] {
            if s.1 {
                rank_sum_diff += avg_rank;
            }
        }
        k = m + 1;
    }
    let u = rank_sum_diff - (n_diff * (n_diff + 1)) as f64 / 2.0;
    u / (n_diff as f64 * n_same as f64)
}

fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn check(label: &str, p: &[f64], h: &[f64], thresholds: &[f64]) {
    for &thr in thresholds {
        let old = removed_zensim_ds_auc(p, h, thr);
        let new = zenstats::pairs::ds_auc(p, h, thr);
        assert!(
            same_bits(old, new),
            "{label} thr={thr}: removed zensim {old:?} ({:#018x}) vs zenstats {new:?} ({:#018x})",
            old.to_bits(),
            new.to_bits()
        );
    }
}

/// bake_verdict's threshold: one population standard deviation of the
/// human scores.
fn h_std(h: &[f64]) -> f64 {
    let m = h.iter().sum::<f64>() / h.len().max(1) as f64;
    (h.iter().map(|x| (x - m).powi(2)).sum::<f64>() / h.len().max(1) as f64).sqrt()
}

/// Deterministic xorshift values in [0, 1).
fn uniform(n: usize, mut s: u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

#[test]
fn bake_verdict_ds_auc_fixtures_are_bit_identical() {
    check(
        "perfect",
        &[0.0, 0.0, 1.0, 1.0],
        &[0.0, 0.0, 1.0, 1.0],
        &[0.5],
    );
    check(
        "constant metric",
        &[0.5; 6],
        &[0.0, 0.3, 0.6, 0.9, 0.2, 0.7],
        &[0.4],
    );
    check(
        "no different pairs",
        &[0.1, 0.2, 0.3, 0.4, 0.5],
        &[0.5; 5],
        &[0.4],
    );
    check("too few rows", &[0.1, 0.2, 0.3], &[0.0, 1.0, 2.0], &[0.4]);
}

#[test]
fn panel_evaluate_synthetic_table_is_bit_identical() {
    // metric_evaluate.rs's default fixture: score = target = 1..6.
    let t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    check("fixture", &t, &t, &[0.5, 1.0, 1.5, 2.5, h_std(&t)]);
    let reversed: Vec<f64> = t.iter().rev().copied().collect();
    check("fixture reversed", &reversed, &t, &[1.5, h_std(&t)]);
}

#[test]
fn owner_brute_force_case_is_bit_identical() {
    let p = [0.0, 0.1, 0.5, 0.55, 1.0, 1.4, 2.0, 2.05];
    let h = [0.0, 0.3, 0.2, 0.9, 1.1, 1.0, 2.2, 1.9];
    check("owner", &p, &h, &[0.0, 0.4, 1.0, 100.0]);
}

#[test]
fn subsampled_tables_with_ties_and_nonfinite_rows_are_bit_identical() {
    // stride = ⌊pairs / 200,000⌋: 633 rows → 200,028 pairs → 1; 900 → 2;
    // 1,200 → 3; 2,000 → 9; 4,292 (CID22's size) → 46. Scores are quantized so
    // gaps tie, and a few rows are NaN or infinite so the finite filter is
    // exercised.
    for (n, seed) in [(633usize, 1u64), (900, 2), (1200, 3), (2000, 4), (4292, 5)] {
        let u = uniform(n, 0x9E37_79B9_7F4A_7C15 ^ seed);
        let v = uniform(n, 0xD1B5_4A32_D192_ED03 ^ seed);
        let h: Vec<f64> = u.iter().map(|x| (x * 40.0).round() / 4.0).collect();
        let mut p: Vec<f64> = h
            .iter()
            .zip(&v)
            .map(|(a, b)| (a + 3.0 * (b - 0.5) * 10.0).round() / 10.0)
            .collect();
        p[n / 3] = f64::NAN;
        p[n / 2] = f64::INFINITY;
        let mut h2 = h.clone();
        h2[n / 5] = f64::NAN;
        let s = h_std(&h);
        check(&format!("n={n}"), &p, &h, &[0.0, 0.25, 1.0, s, 4.0]);
        check(&format!("n={n} nan target"), &p, &h2, &[1.0, s]);
    }
}
