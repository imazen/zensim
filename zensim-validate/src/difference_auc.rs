//! Difference-discrimination statistic shared by the two evaluation entry points.
use rayon::prelude::*;

/// DS-AUC (G9, Mohammadi 2025 § VII): Area Under the ROC curve for
/// classifying stimulus pairs as "same" vs "different" perceptual quality.
///
/// Without parsed 2AFC response data, we use a practical proxy: a pair
/// (i, j) is labeled "different" when |human[i] − human[j]| exceeds
/// `diff_threshold` (in human-score units), "same" otherwise. The metric's
/// |score[i] − score[j]| is the classifier score. AUC measures how well
/// the metric's score-gap separates the same/different pairs.
///
/// Returns AUC in [0, 1]; 0.5 = chance. Subsamples pairs when n is large
/// to keep the O(n²) pair enumeration tractable.
pub(crate) fn ds_auc(predicted: &[f64], human: &[f64], diff_threshold: f64) -> f64 {
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
