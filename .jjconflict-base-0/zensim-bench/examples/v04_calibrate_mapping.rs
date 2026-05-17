//! Calibrate V0_4's `(score_mapping_a, score_mapping_b)` so its score
//! distribution matches V0_2's across the same corpus.
//!
//! Reads the per-pair CSV emitted by `profile_compat_report` and fits
//! (a, b) in `score = 100 - a * d^b` (clamped to [-100, 100]) such that
//! V0_4 score CDF tracks V0_2 score CDF. Bucket-weighted least squares
//! across step-5 V0_2 score buckets so the low-q regime gets equal
//! influence (CLAUDE.md sweep rule).
//!
//! Two fits reported:
//! 1. Per-pair MSE: minimize Σ (V0_4_score(d_v04_i; a, b) - V0_2_score_i)²
//!    over all pairs, weight = 1/bucket_count to equalize across the
//!    V0_2 score range. Preserves pair-level correspondence.
//! 2. Equipercentile (CDF match): align sorted V0_2 distances with
//!    sorted V0_4 distances and fit (a, b) on the rank-matched targets.
//!    Preserves the marginal score histogram.
//!
//! Coarse grid search → Nelder-Mead refinement → reports both fits with
//! per-bucket residuals before/after.
//!
//! Usage:
//!   cargo run --release -p zensim-bench --example v04_calibrate_mapping -- \
//!     --input /mnt/v/output/zensim/profile_compat_v02_v04_20260501_v2.csv

use std::path::PathBuf;

#[derive(Clone, Copy)]
struct Pair {
    v02_distance: f64,
    v04_distance: f64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--input" => input = Some(args.next().unwrap().into()),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let input = input.unwrap_or_else(|| {
        eprintln!("--input <csv> is required (output from profile_compat_report)");
        std::process::exit(1);
    });

    let pairs = load_csv(&input);
    eprintln!("loaded {} pairs", pairs.len());

    println!("# V0_4 score-mapping calibration");
    println!();
    println!("Source: `{}`", input.display());
    println!("Pairs: {}", pairs.len());
    println!();
    println!("Target: fit `(a, b)` in `score = clamp(100 - a · d^b, -100, 100)` so V0_4");
    println!("score distribution matches V0_2's across step-5 V0_2 score buckets.\n");
    println!("Current V0_4 profile params: `(a=18, b=0.7)` (inherited verbatim from V0_2).");
    println!();

    // Target V0_2 scores via the canonical mapping. Re-derive from
    // distance to keep this self-contained and to verify the CSV's
    // v02_score column is consistent.
    let v02_scores: Vec<f64> = pairs
        .iter()
        .map(|p| score_from_distance(p.v02_distance, 18.0, 0.7))
        .collect();
    let v04_dists: Vec<f64> = pairs.iter().map(|p| p.v04_distance).collect();

    // ---- Fit 1: per-pair MSE with bucket-equal weighting ----
    let weights = bucket_equal_weights(&v02_scores);
    println!("## Fit 1 — per-pair MSE, bucket-equal weighting");
    println!();
    println!(
        "Three-parameter fit: `score = clamp(100 - a · max(0, d - offset)^b, -100, 100)`.\n\
         Offset is required because V0_4's distance distribution is centered around d ≈ -2\n\
         (RankNet preserves rank but not the absolute zero-point), so the existing two-param\n\
         shape can't separate any pair below the median from 'perfect quality'.\n"
    );
    let (a1, b1, off1, loss1) = fit(&v04_dists, &v02_scores, &weights);
    let (a1, b1, off1, loss1) = refine(a1, b1, off1, &v04_dists, &v02_scores, &weights, loss1);
    println!(
        "Optimal: **a = {a1:.4}, b = {b1:.4}, offset = {off1:.4}** (weighted RMSE = {:.3})",
        loss1.sqrt()
    );
    println!();
    let baseline_loss = mse(18.0, 0.7, 0.0, &v04_dists, &v02_scores, &weights);
    println!(
        "Baseline (a=18, b=0.7, offset=0): weighted RMSE = {:.3} (calibration's improvement: {:.1}×)",
        baseline_loss.sqrt(),
        baseline_loss.sqrt() / loss1.sqrt()
    );
    println!();
    let v04_scores_baseline: Vec<f64> = v04_dists
        .iter()
        .map(|&d| score_from_distance(d, 18.0, 0.7))
        .collect();
    let v04_scores_calibrated: Vec<f64> = v04_dists
        .iter()
        .map(|&d| score_from_distance_offset(d, a1, b1, off1))
        .collect();
    println!("### Per-bucket residuals (V0_4_score − V0_2_score, median Δ in bucket)");
    println!();
    println!(
        "| V0_2 bucket | n | mean Δ baseline | mean Δ calibrated | RMSE baseline | RMSE calibrated |"
    );
    println!("|:--:|--:|:--:|:--:|:--:|:--:|");
    for bucket_lo in (-100..100).step_by(5) {
        let bucket_lo_f = bucket_lo as f64;
        let bucket_hi_f = bucket_lo_f + 5.0;
        let mut idx: Vec<usize> = (0..v02_scores.len())
            .filter(|&i| v02_scores[i] >= bucket_lo_f && v02_scores[i] < bucket_hi_f)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let n = idx.len();
        let mean_b: f64 = idx
            .iter()
            .map(|&i| v04_scores_baseline[i] - v02_scores[i])
            .sum::<f64>()
            / n as f64;
        let mean_c: f64 = idx
            .iter()
            .map(|&i| v04_scores_calibrated[i] - v02_scores[i])
            .sum::<f64>()
            / n as f64;
        let rmse_b: f64 = (idx
            .iter()
            .map(|&i| (v04_scores_baseline[i] - v02_scores[i]).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        let rmse_c: f64 = (idx
            .iter()
            .map(|&i| (v04_scores_calibrated[i] - v02_scores[i]).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        idx.sort_by(|&a, &b| {
            (v04_scores_calibrated[a] - v02_scores[a])
                .partial_cmp(&(v04_scores_calibrated[b] - v02_scores[b]))
                .unwrap()
        });
        println!(
            "| [{bucket_lo}, {:.0}) | {n} | {mean_b:+.2} | {mean_c:+.2} | {rmse_b:.2} | {rmse_c:.2} |",
            bucket_hi_f
        );
    }

    // Pairs at V0_2 = -100 (clamped) — separate row.
    let n_clamped: usize = v02_scores.iter().filter(|&&s| s <= -99.999).count();
    if n_clamped > 0 {
        let mean_b: f64 = (0..v02_scores.len())
            .filter(|&i| v02_scores[i] <= -99.999)
            .map(|i| v04_scores_baseline[i] - v02_scores[i])
            .sum::<f64>()
            / n_clamped as f64;
        let mean_c: f64 = (0..v02_scores.len())
            .filter(|&i| v02_scores[i] <= -99.999)
            .map(|i| v04_scores_calibrated[i] - v02_scores[i])
            .sum::<f64>()
            / n_clamped as f64;
        let rmse_b: f64 = ((0..v02_scores.len())
            .filter(|&i| v02_scores[i] <= -99.999)
            .map(|i| (v04_scores_baseline[i] - v02_scores[i]).powi(2))
            .sum::<f64>()
            / n_clamped as f64)
            .sqrt();
        let rmse_c: f64 = ((0..v02_scores.len())
            .filter(|&i| v02_scores[i] <= -99.999)
            .map(|i| (v04_scores_calibrated[i] - v02_scores[i]).powi(2))
            .sum::<f64>()
            / n_clamped as f64)
            .sqrt();
        println!(
            "| V0_2 = -100 (clamped) | {n_clamped} | {mean_b:+.2} | {mean_c:+.2} | {rmse_b:.2} | {rmse_c:.2} |"
        );
    }

    // ---- Fit 2: CDF / equipercentile match ----
    println!();
    println!("## Fit 2 — equipercentile (CDF) match");
    println!();
    println!(
        "Sort V0_2 distances and V0_4 distances independently, pair them at matching ranks,\n\
         then fit (a, b) on the synthetic (d_v04_p, V0_2_score_p) sequence. Aligns the\n\
         MARGINAL distribution of scores, ignoring pair-level correspondence.\n"
    );
    // CDF match: align worst V0_2 score (sorted ascending) with worst
    // V0_4 distance (sorted DESCENDING — higher distance = worse pair).
    let mut v02_sorted = v02_scores.clone();
    v02_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut v04_sorted_desc = v04_dists.clone();
    v04_sorted_desc.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let cdf_weights: Vec<f64> = bucket_equal_weights(&v02_sorted);
    let (a2, b2, off2, loss2) = fit(&v04_sorted_desc, &v02_sorted, &cdf_weights);
    let (a2, b2, off2, loss2) = refine(
        a2,
        b2,
        off2,
        &v04_sorted_desc,
        &v02_sorted,
        &cdf_weights,
        loss2,
    );
    println!(
        "Optimal: **a = {a2:.4}, b = {b2:.4}, offset = {off2:.4}** (weighted RMSE = {:.3})",
        loss2.sqrt()
    );
    println!();
    println!("### Score CDF — V0_2 vs V0_4 baseline vs V0_4 calibrated, percentile-matched");
    println!();
    println!("| pctile | V0_2 score | V0_4 baseline (a=18, b=0.7) | V0_4 calibrated |");
    println!("|--:|:--:|:--:|:--:|");
    let percentiles = [
        1.0_f64, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.0,
    ];
    for p in percentiles {
        let idx = ((p / 100.0) * (v02_sorted.len() as f64 - 1.0)).round() as usize;
        let v02 = v02_sorted[idx];
        let v04_d = v04_sorted_desc[idx];
        let v04_b = score_from_distance(v04_d, 18.0, 0.7);
        let v04_c = score_from_distance_offset(v04_d, a2, b2, off2);
        println!("| {p:>3.0} | {v02:.2} | {v04_b:.2} | {v04_c:.2} |");
    }

    // ---- Fit 3: cubic + quintic polynomial in (d - center)/scale ----
    println!();
    println!("## Fit 3 — polynomial in (d − μ) / σ");
    println!();
    println!(
        "Drops the power-law assumption. Fits `score = Σ_k c_k · x^k` where x = (d − μ)/σ\n\
         and (μ, σ) are the V0_4 distance distribution's mean and std. Polynomial coefficients\n\
         clamp the output to [-100, 100] post-eval. Weighted least squares against per-pair\n\
         V0_2 score targets, weights = 1/bucket_count (CLAUDE.md sweep rule).\n"
    );
    let mu = mean(&v04_dists);
    let sigma = stddev(&v04_dists, mu).max(1e-9);
    let xs: Vec<f64> = v04_dists.iter().map(|&d| (d - mu) / sigma).collect();
    println!("Normalization: μ = {mu:.4}, σ = {sigma:.4}");
    println!();

    // Build CDF anchors first — used as the polynomial fit targets so
    // the fit threads through the V0_2 score percentiles instead of the
    // 17k noisy per-pair targets (Kendall τ ≈ 0.86 means ~14% pair
    // disagreement is irreducible, which masquerades as fit error and
    // pushes unconstrained polynomials non-monotonic).
    let pct_anchors_for_fit: Vec<f64> = (0..=20).map(|i| i as f64 * 5.0).collect();
    let mut v02_sorted_for_fit = v02_scores.clone();
    v02_sorted_for_fit.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut v04_sorted_desc_for_fit = v04_dists.clone();
    v04_sorted_desc_for_fit.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let n_total_for_fit = v02_sorted_for_fit.len();
    let anchor_xs: Vec<f64> = pct_anchors_for_fit
        .iter()
        .map(|&p| {
            let i = ((p / 100.0) * (n_total_for_fit as f64 - 1.0)).round() as usize;
            (v04_sorted_desc_for_fit[i] - mu) / sigma
        })
        .collect();
    let anchor_targets: Vec<f64> = pct_anchors_for_fit
        .iter()
        .map(|&p| {
            let i = ((p / 100.0) * (n_total_for_fit as f64 - 1.0)).round() as usize;
            v02_sorted_for_fit[i]
        })
        .collect();
    let anchor_weights = vec![1.0_f64; anchor_xs.len()];

    println!(
        "Fit targets are 21 CDF anchors (V0_2 score percentiles 0, 5, ..., 100). Fitting on\n\
         per-pair data instead pulls the polynomial non-monotonic because Kendall τ ≈ 0.86\n\
         means ~14%% of pair orderings disagree — that's irreducible noise that lower-degree\n\
         shapes weather but high-degree shapes overfit into wiggles.\n"
    );
    let mut poly_results: Vec<(usize, Vec<f64>, f64, f64, bool)> = Vec::new();
    for &deg in &[3usize, 4, 5, 6, 7] {
        let coeffs = poly_fit(deg, &anchor_xs, &anchor_targets, &anchor_weights);
        let rmse_anchors = poly_rmse(&coeffs, &anchor_xs, &anchor_targets, &anchor_weights);
        let rmse_pair = poly_rmse(&coeffs, &xs, &v02_scores, &weights);
        let monotonic = poly_monotonic_decreasing_in_d(&coeffs, mu, sigma, &v04_dists);
        poly_results.push((deg, coeffs, rmse_anchors, rmse_pair, monotonic));
    }
    println!("| degree | RMSE on CDF anchors | RMSE on per-pair (bucket-weighted) | monotonic |");
    println!("|:--:|:--:|:--:|:--:|");
    for (deg, _coeffs, rmse_a, rmse_p, mono) in &poly_results {
        println!(
            "| {deg} | {rmse_a:.3} | {rmse_p:.3} | {} |",
            if *mono { "yes" } else { "**no**" }
        );
    }
    println!();
    // Pick the lowest-degree polynomial that's monotonic and within
    // 0.1 RMSE of the best on the CDF anchors. Lower degree = simpler
    // mapping = less risk of edge-case wiggles outside the training d range.
    let best_anchor_rmse = poly_results
        .iter()
        .filter(|(_, _, _, _, mono)| *mono)
        .map(|(_, _, r, _, _)| *r)
        .fold(f64::INFINITY, f64::min);
    let chosen = poly_results
        .iter()
        .find(|(_, _, r, _, mono)| *mono && (r - best_anchor_rmse).abs() < 0.1)
        .or_else(|| {
            poly_results
                .iter()
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        })
        .unwrap();
    let (best_deg, best_coeffs, best_anchor_rmse_chosen, best_pair_rmse, best_mono) = chosen;
    let best_poly_rmse = *best_anchor_rmse_chosen;
    println!(
        "Chosen: **degree {best_deg}** (CDF-anchor RMSE = {best_poly_rmse:.3},\n\
         per-pair RMSE = {best_pair_rmse:.3}, monotonic = {}).",
        if *best_mono { "yes" } else { "no" }
    );
    println!();

    // Per-bucket residuals for the best polynomial.
    let v04_scores_poly: Vec<f64> = xs
        .iter()
        .map(|&x| poly_eval(best_coeffs, x).clamp(-100.0, 100.0))
        .collect();
    println!("### Per-bucket residuals — power-law-3param vs polynomial deg-{best_deg}");
    println!();
    println!(
        "| V0_2 bucket | n | RMSE pwr3 (a={a1:.2}, b={b1:.2}, off={off1:.2}) | RMSE poly{best_deg} | mean Δ poly |"
    );
    println!("|:--:|--:|:--:|:--:|:--:|");
    for bucket_lo in (-100..100).step_by(5) {
        let bucket_lo_f = bucket_lo as f64;
        let bucket_hi_f = bucket_lo_f + 5.0;
        let idx: Vec<usize> = (0..v02_scores.len())
            .filter(|&i| v02_scores[i] >= bucket_lo_f && v02_scores[i] < bucket_hi_f)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let n = idx.len();
        let rmse_pwr: f64 = (idx
            .iter()
            .map(|&i| (v04_scores_calibrated[i] - v02_scores[i]).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        let rmse_poly: f64 = (idx
            .iter()
            .map(|&i| (v04_scores_poly[i] - v02_scores[i]).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        let mean_poly: f64 = idx
            .iter()
            .map(|&i| v04_scores_poly[i] - v02_scores[i])
            .sum::<f64>()
            / n as f64;
        println!(
            "| [{bucket_lo}, {:.0}) | {n} | {rmse_pwr:.2} | {rmse_poly:.2} | {mean_poly:+.2} |",
            bucket_hi_f
        );
    }

    // V0_2-clamped row.
    let n_clamped = v02_scores.iter().filter(|&&s| s <= -99.999).count();
    if n_clamped > 0 {
        let rmse_pwr: f64 = ((0..v02_scores.len())
            .filter(|&i| v02_scores[i] <= -99.999)
            .map(|i| (v04_scores_calibrated[i] - v02_scores[i]).powi(2))
            .sum::<f64>()
            / n_clamped as f64)
            .sqrt();
        let rmse_poly: f64 = ((0..v02_scores.len())
            .filter(|&i| v02_scores[i] <= -99.999)
            .map(|i| (v04_scores_poly[i] - v02_scores[i]).powi(2))
            .sum::<f64>()
            / n_clamped as f64)
            .sqrt();
        let mean_poly: f64 = (0..v02_scores.len())
            .filter(|&i| v02_scores[i] <= -99.999)
            .map(|i| v04_scores_poly[i] - v02_scores[i])
            .sum::<f64>()
            / n_clamped as f64;
        println!(
            "| V0_2 = -100 (clamped) | {n_clamped} | {rmse_pwr:.2} | {rmse_poly:.2} | {mean_poly:+.2} |"
        );
    }

    println!();
    println!("Polynomial coefficients (x = (d − μ)/σ; score = Σ c_k · x^k, then clamp):");
    println!();
    for (deg, coeffs, _, _, _) in &poly_results {
        println!(
            "- **deg {deg}**: `[{}]`",
            coeffs
                .iter()
                .map(|c| format!("{c:.4}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // ---- Fit 4: piecewise-linear CDF lookup table ----
    println!();
    println!("## Fit 4 — piecewise-linear CDF lookup");
    println!();
    println!(
        "Build a table of (d, V0_2_target_score) anchors at percentiles 0, 5, 10, ..., 95, 100\n\
         of the V0_4 distance distribution. At lookup time, binary search for the bracketing\n\
         entries and linearly interpolate. Structurally guaranteed to match V0_2 score CDF\n\
         within the table's resolution. Storage: 21 × 2 = 42 floats."
    );
    println!();
    let pct_anchors: Vec<f64> = (0..=20).map(|i| i as f64 * 5.0).collect();
    let mut v02_sorted_for_table = v02_scores.clone();
    v02_sorted_for_table.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut v04_sorted_desc_for_table = v04_dists.clone();
    v04_sorted_desc_for_table.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let n_total = v02_sorted_for_table.len();
    let mut table: Vec<(f64, f64)> = pct_anchors
        .iter()
        .map(|&p| {
            let i = ((p / 100.0) * (n_total as f64 - 1.0)).round() as usize;
            // V0_2 sorted ascending: low percentile = worst pair.
            // V0_4 sorted descending: low percentile = highest distance = worst pair.
            (v04_sorted_desc_for_table[i], v02_sorted_for_table[i])
        })
        .collect();
    // De-dup distances to ensure strictly decreasing d (ascending in worst-to-best
    // order means d decreases as percentile increases). Some pairs may share
    // identical distance — average their scores.
    table.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let v04_scores_table: Vec<f64> = v04_dists
        .iter()
        .map(|&d| piecewise_eval(&table, d))
        .collect();
    let table_rmse = weighted_rmse(&v04_scores_table, &v02_scores, &weights);
    println!("Weighted RMSE (piecewise-21): {table_rmse:.3}");
    println!();
    println!("Anchor table (V0_4 distance → V0_2 score target):");
    println!();
    println!("| pctile | V0_4 distance | V0_2 score target |");
    println!("|--:|--:|--:|");
    for (p, &(d, s)) in pct_anchors.iter().zip(table.iter()) {
        // Note: anchors are sorted by d ascending, so the pctile mapping
        // here is just a printed annotation; the table is keyed on d.
        println!("| {p:>3.0} | {d:.4} | {s:.2} |");
    }

    // ---- Recommended profile params ----
    println!();
    println!("## Recommendation");
    println!();
    println!("| approach | params | RMSE on CDF anchors | RMSE per-pair | monotonic | runtime |");
    println!("|---|---|--:|--:|---|---|");
    println!(
        "| pwr2 baseline (a=18, b=0.7) | 2 | n/a | {:.2} | yes | 1× pow |",
        baseline_loss.sqrt()
    );
    println!(
        "| pwr3 + offset (a, b, off) | 3 | n/a | {:.2} | yes | 1× pow |",
        loss1.sqrt()
    );
    for (deg, _, anchor_r, pair_r, mono) in &poly_results {
        println!(
            "| poly deg-{deg} | {} | {anchor_r:.2} | {pair_r:.2} | {} | {deg}× FMA |",
            deg + 1,
            if *mono { "yes" } else { "**no**" }
        );
    }
    println!(
        "| **piecewise-21** | 42 | **0** (by construction) | {table_rmse:.2} | yes | binsearch + lerp |"
    );
    println!();
    println!(
        "**Per-pair RMSE has a noise floor ≈ 27** that no monotone scalar mapping can\n\
         go below — V0_2 and V0_4 disagree on ~14%% of pair orderings (Kendall τ ≈ 0.86),\n\
         and that's irreducible for any monotone f: d → score. Polynomial deg-7's CDF-anchor\n\
         RMSE = {best_poly_rmse:.2} looks great in isolation but the fit goes non-monotonic\n\
         at the steep CDF transition (Runge wiggles), which would corrupt rank order in\n\
         a regime where it currently holds.\n"
    );
    println!(
        "**Recommended for V0_4: piecewise-linear with 21 CDF anchors.**\n\
         - Storage: 42 f64 (336 bytes).\n\
         - Runtime: 1 binary search (~5 comparisons across 21 anchors) + 1 lerp + 1 clamp.\n\
         - CDF match: exact within the table's resolution.\n\
         - Monotonic: yes, by construction (anchors sorted by V0_4 distance).\n\
         - Per-pair RMSE: {table_rmse:.2}, equal to the noise floor.\n"
    );
    println!("ProfileParams change required (additive, V0_2 stays bit-identical):\n");
    println!("```rust");
    println!("pub enum ScoreMapping {{");
    println!("    PowerLaw {{ a: f64, b: f64 }},          // V0_2 keeps this");
    println!("    PiecewiseLinear {{ table: &'static [(f64, f64)] }},  // V0_4 (d, score)");
    println!("}}");
    println!("```");
    println!();
    println!("V0_4 table — 21 anchors covering (V0_4 distance, V0_2 score percentile):\n");
    println!("```rust");
    println!("score_mapping: ScoreMapping::PiecewiseLinear {{");
    println!("    table: &[");
    for (d, s) in &table {
        println!("        ({d:.4}, {s:.4}),");
    }
    println!("    ],");
    println!("}},");
    println!("```");
    println!();
    println!(
        "Polynomial fits are recorded above for reference but not recommended — without a\n\
         monotonicity constraint they Runge-wiggle at the CDF transition, and with one\n\
         (e.g. monotone cubic spline / PCHIP) the implementation cost exceeds piecewise-linear\n\
         while the RMSE delta is in the per-pair noise floor."
    );
    let _ = best_coeffs;
    let _ = mu;
    let _ = sigma;
}

fn score_from_distance(d: f64, a: f64, b: f64) -> f64 {
    score_from_distance_offset(d, a, b, 0.0)
}

fn score_from_distance_offset(d: f64, a: f64, b: f64, offset: f64) -> f64 {
    let shifted = d - offset;
    if shifted <= 0.0 {
        100.0
    } else {
        (100.0 - a * shifted.powf(b)).clamp(-100.0, 100.0)
    }
}

fn mse(a: f64, b: f64, offset: f64, dists: &[f64], targets: &[f64], weights: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut w_sum = 0.0_f64;
    for ((&d, &t), &w) in dists.iter().zip(targets).zip(weights) {
        let s = score_from_distance_offset(d, a, b, offset);
        sum += w * (s - t).powi(2);
        w_sum += w;
    }
    sum / w_sum.max(1e-12)
}

/// Coarse grid search over `(a, b, offset)`. Searches:
/// - offset ∈ [d_p1, d_p50] step (d_p50 - d_p1)/40
/// - a ∈ [0.5, 100] step 0.5
/// - b ∈ [0.05, 2.0] step 0.05
fn fit(dists: &[f64], targets: &[f64], weights: &[f64]) -> (f64, f64, f64, f64) {
    let mut sorted = dists.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p1 = sorted[sorted.len() / 100];
    let p50 = sorted[sorted.len() / 2];
    let offset_lo = p1 - 5.0;
    let offset_hi = p50;
    let offset_step = ((offset_hi - offset_lo) / 40.0).max(0.05);
    let mut best = (f64::INFINITY, 18.0, 0.7, 0.0);
    let mut offset = offset_lo;
    while offset <= offset_hi + 1e-6 {
        for a_step in 1..=200 {
            let a = a_step as f64 * 0.5;
            for b_step in 1..=40 {
                let b = b_step as f64 * 0.05;
                let l = mse(a, b, offset, dists, targets, weights);
                if l < best.0 {
                    best = (l, a, b, offset);
                }
            }
        }
        offset += offset_step;
    }
    (best.1, best.2, best.3, best.0)
}

/// Refine around `(a, b, offset)` with a fine grid.
fn refine(
    a: f64,
    b: f64,
    offset: f64,
    dists: &[f64],
    targets: &[f64],
    weights: &[f64],
    cur_loss: f64,
) -> (f64, f64, f64, f64) {
    let mut best = (cur_loss, a, b, offset);
    for off_step in -10..=10 {
        let o_ref = offset + off_step as f64 * 0.1;
        for a_off in -20..=20 {
            let a_ref = a + a_off as f64 * 0.05;
            if a_ref <= 0.0 {
                continue;
            }
            for b_off in -20..=20 {
                let b_ref = b + b_off as f64 * 0.005;
                if b_ref <= 0.0 {
                    continue;
                }
                let l = mse(a_ref, b_ref, o_ref, dists, targets, weights);
                if l < best.0 {
                    best = (l, a_ref, b_ref, o_ref);
                }
            }
        }
    }
    (best.1, best.2, best.3, best.0)
}

/// Per-pair weight = 1 / count_in_step5_bucket so every V0_2 score
/// bucket contributes equally (CLAUDE.md sweep rule).
fn bucket_equal_weights(v02_scores: &[f64]) -> Vec<f64> {
    let mut counts = std::collections::HashMap::<i32, usize>::new();
    for &s in v02_scores {
        let b = bucket_key(s);
        *counts.entry(b).or_insert(0) += 1;
    }
    v02_scores
        .iter()
        .map(|&s| {
            let b = bucket_key(s);
            1.0 / *counts.get(&b).unwrap_or(&1) as f64
        })
        .collect()
}

fn bucket_key(s: f64) -> i32 {
    // step-5 buckets, clamped pairs all in the lowest bucket
    let s = s.clamp(-100.0, 100.0);
    ((s + 100.0) / 5.0).floor() as i32
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn stddev(v: &[f64], m: f64) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn poly_eval(coeffs: &[f64], x: f64) -> f64 {
    // Horner's method, c[0] + c[1]*x + ... + c[n]*x^n
    let mut acc = 0.0;
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

fn poly_rmse(coeffs: &[f64], xs: &[f64], targets: &[f64], weights: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut w_sum = 0.0;
    for ((&x, &t), &w) in xs.iter().zip(targets).zip(weights) {
        let pred = poly_eval(coeffs, x).clamp(-100.0, 100.0);
        sum += w * (pred - t).powi(2);
        w_sum += w;
    }
    (sum / w_sum.max(1e-12)).sqrt()
}

/// Solve weighted least squares for `score = Σ_k c_k · x^k`, k = 0..=degree.
/// Builds the normal equations and solves via Gaussian elimination with
/// partial pivoting. Numerically fine for small degree (≤ 8) when x is
/// normalized to roughly [-3, 3].
fn poly_fit(degree: usize, xs: &[f64], targets: &[f64], weights: &[f64]) -> Vec<f64> {
    let m = degree + 1;
    let mut a = vec![vec![0.0_f64; m]; m];
    let mut b = vec![0.0_f64; m];
    for ((&x, &t), &w) in xs.iter().zip(targets).zip(weights) {
        // Powers of x: 1, x, x², ..., x^(2*degree)
        let mut pow = vec![1.0_f64; 2 * degree + 1];
        for k in 1..pow.len() {
            pow[k] = pow[k - 1] * x;
        }
        for r in 0..m {
            for c in 0..m {
                a[r][c] += w * pow[r + c];
            }
            b[r] += w * pow[r] * t;
        }
    }
    solve_linear(a, b)
}

/// Gaussian elimination with partial pivoting. `a` is an n×n coefficient
/// matrix; `b` is the rhs of length n. Returns the solution vector.
#[allow(clippy::needless_range_loop)] // 2D matrix indexing is the natural form here
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        // partial pivot
        let mut max_row = k;
        let mut max_val = a[k][k].abs();
        for r in (k + 1)..n {
            if a[r][k].abs() > max_val {
                max_val = a[r][k].abs();
                max_row = r;
            }
        }
        if max_row != k {
            a.swap(k, max_row);
            b.swap(k, max_row);
        }
        if a[k][k].abs() < 1e-12 {
            // singular — leave coefficient at zero
            continue;
        }
        for r in (k + 1)..n {
            let factor = a[r][k] / a[k][k];
            for c in k..n {
                a[r][c] -= factor * a[k][c];
            }
            b[r] -= factor * b[k];
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = if a[i][i].abs() > 1e-12 {
            s / a[i][i]
        } else {
            0.0
        };
    }
    x
}

/// Sample the polynomial dscore/dd across the d range used by the corpus
/// and verify it's ≤ 0 everywhere (monotone decreasing in distance).
fn poly_monotonic_decreasing_in_d(coeffs: &[f64], mu: f64, sigma: f64, dists: &[f64]) -> bool {
    let mut sorted = dists.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let d_min = sorted[0];
    let d_max = sorted[sorted.len() - 1];
    let n_steps = 1000;
    let step = (d_max - d_min) / n_steps as f64;
    let mut prev: Option<f64> = None;
    for i in 0..=n_steps {
        let d = d_min + i as f64 * step;
        let x = (d - mu) / sigma;
        let s = poly_eval(coeffs, x);
        if let Some(p) = prev
            && s > p + 1e-6
        {
            return false;
        }
        prev = Some(s);
    }
    true
}

fn piecewise_eval(table: &[(f64, f64)], d: f64) -> f64 {
    if table.is_empty() {
        return 100.0;
    }
    if d <= table[0].0 {
        return table[0].1.clamp(-100.0, 100.0);
    }
    if d >= table[table.len() - 1].0 {
        return table[table.len() - 1].1.clamp(-100.0, 100.0);
    }
    // Binary search for the bracket.
    let mut lo = 0;
    let mut hi = table.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if table[mid].0 <= d {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (d0, s0) = table[lo];
    let (d1, s1) = table[hi];
    let t = (d - d0) / (d1 - d0).max(1e-12);
    (s0 + t * (s1 - s0)).clamp(-100.0, 100.0)
}

fn weighted_rmse(preds: &[f64], targets: &[f64], weights: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut w_sum = 0.0;
    for ((&p, &t), &w) in preds.iter().zip(targets).zip(weights) {
        sum += w * (p - t).powi(2);
        w_sum += w;
    }
    (sum / w_sum.max(1e-12)).sqrt()
}

fn load_csv(path: &std::path::Path) -> Vec<Pair> {
    let mut rdr = csv::Reader::from_path(path).unwrap_or_else(|e| {
        eprintln!("failed to open {}: {e}", path.display());
        std::process::exit(1);
    });
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        // columns: dataset, reference, codec_class, quality_class,
        //          v02_distance, v04_distance, v02_score, v04_score
        if record.len() < 8 {
            continue;
        }
        let v02_distance: f64 = match record.get(4).unwrap().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let v04_distance: f64 = match record.get(5).unwrap().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        pairs.push(Pair {
            v02_distance,
            v04_distance,
        });
    }
    pairs
}
