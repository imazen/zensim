//! Profile compatibility report: how does the V0_4 MLP profile differ
//! from the V0_2 linear profile on the same pairs?
//!
//! Downstream consumers (imageflow, codec-eval, anything with a fixed
//! "ship at zensim ≥ N" gate) are calibrated against V0_2 score values,
//! not V0_2 SROCC. A new profile that improves human SROCC but moves
//! the absolute score distribution is a silent contract break.
//!
//! For each (reference, distorted) pair in KADIK10k / TID2013 / CID22:
//! 1. Extract features once via PreviewV0_4 (forces compute_all_features).
//! 2. Score V0_2 (manual dot product with V0_2 weights / 4 scales).
//! 3. Score V0_4 (Predictor over the trained ZNPR v2 bake).
//! 4. Map both raw distances to publishable scores with the standard
//!    `100 - 18 * d^0.7` mapping.
//!
//! Outputs:
//! - Markdown summary table to stdout
//! - Per-pair CSV (one row per pair) to --output
//! - V0_2 score distribution histogram (step-5 buckets across [-100, 100),
//!   with V0_4 percentiles per bucket — the score-equivalence calibration table)
//! - Per-(codec, quality_level) bias breakdown so distortion strength
//!   and codec are separable (CLAUDE.md sweep rule: low-q must be sampled
//!   with the same density as high-q)
//! - Codec-aggregated medians as a secondary view (collapses level)
//! - Top-K largest disagreements
//!
//! Usage:
//!   cargo run --release -p zensim-bench --example profile_compat_report -- \
//!     --kadid /mnt/v/dataset/kadid10k \
//!     --tid /mnt/v/dataset/tid2013 \
//!     --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
//!     --v04-bake /mnt/v/output/zensim/synthetic-v2/runs/v04_mlp_v5znpr2_20260430T044620.bin \
//!     --output /mnt/v/output/zensim/profile_compat_v02_v04.csv

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use zenpredict::{Model, Predictor};
use zensim::{RgbSlice, Zensim, ZensimProfile};

#[derive(Debug, Clone)]
struct Pair {
    reference: PathBuf,
    distorted: PathBuf,
    /// Codec / distortion class for breakdown reporting. e.g.
    /// "kadid:24" (distortion type 24) or "cid22:JPEGXL" (encoder name).
    codec_class: String,
    /// Quality bucket / level if extractable, else empty.
    quality_class: String,
    /// Normalized human MOS / DMOS / MCOS. Polarity varies per dataset
    /// — KADIK uses dmos (higher = worse), CID22 uses MCOS (higher =
    /// better), TID uses MOS (higher = better). For SROCC we take
    /// `.abs()` so polarity doesn't matter; for pairwise rank we still
    /// compute (humanA - humanB) and rely on |corr|.
    human_score: f64,
}

#[derive(Debug, Clone)]
struct DatasetSpec {
    name: &'static str,
    pairs: Vec<Pair>,
}

#[derive(Debug, Clone, Copy)]
struct Row {
    v02_distance: f64,
    v04_distance: f64,
    v02_score: f64,
    v04_score: f64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut kadid: Option<PathBuf> = None;
    let mut tid: Option<PathBuf> = None;
    let mut cid22: Option<PathBuf> = None;
    let mut konjnd: Option<PathBuf> = None;
    let mut v04_bake_path: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut max_pairs: usize = usize::MAX;
    let mut top_k: usize = 10;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--kadid" => kadid = Some(args.next().unwrap().into()),
            "--tid" => tid = Some(args.next().unwrap().into()),
            "--cid22" => cid22 = Some(args.next().unwrap().into()),
            "--konjnd" => konjnd = Some(args.next().unwrap().into()),
            "--v04-bake" => v04_bake_path = Some(args.next().unwrap().into()),
            "--output" => output = Some(args.next().unwrap().into()),
            "--max-pairs" => max_pairs = args.next().unwrap().parse().unwrap(),
            "--top-k" => top_k = args.next().unwrap().parse().unwrap(),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    let v04_bake_path = v04_bake_path.unwrap_or_else(|| {
        eprintln!("--v04-bake is required");
        std::process::exit(1);
    });
    let v04_bake_bytes = std::fs::read(&v04_bake_path).unwrap_or_else(|e| {
        eprintln!("failed to read v04 bake at {:?}: {e}", v04_bake_path);
        std::process::exit(1);
    });

    let mut datasets: Vec<DatasetSpec> = Vec::new();
    if let Some(p) = kadid {
        datasets.push(DatasetSpec {
            name: "KADIK10k",
            pairs: load_kadid(&p, max_pairs),
        });
    }
    if let Some(p) = tid {
        datasets.push(DatasetSpec {
            name: "TID2013",
            pairs: load_tid(&p, max_pairs),
        });
    }
    if let Some(p) = cid22 {
        datasets.push(DatasetSpec {
            name: "CID22",
            pairs: load_cid22(&p, max_pairs),
        });
    }
    if datasets.is_empty() && konjnd.is_none() {
        eprintln!("no datasets — pass at least one of --kadid, --tid, --cid22, --konjnd");
        std::process::exit(1);
    }

    println!("# Profile compatibility report — V0_2 vs V0_4");
    println!();
    println!(
        "Bake: `{}`\n",
        v04_bake_path.file_name().unwrap().to_string_lossy()
    );
    println!(
        "All scores use the standard `100 - 18·d^0.7` mapping, clamped at 0.\n\
         Distances below are profile-internal raw weighted distances, lower = more similar.\n"
    );

    let mut all_rows_csv: Vec<(String, String, String, String, f64, Row)> = Vec::new();

    println!("## Cross-profile correlation");
    println!();
    println!("Pearson + Kendall τ between V0_2 and V0_4 raw distances on the SAME pairs.");
    println!(
        "τ = 1 means rank-equivalent; τ < 1 means the profiles disagree on relative ordering."
    );
    println!();
    println!(
        "| Dataset | n | r(d_V02, d_V04) | τ(d_V02, d_V04) | r(s_V02, s_V04) | mean Δscore | median Δscore | σ Δscore |"
    );
    println!("|---------|--:|:----:|:----:|:----:|:----:|:----:|:----:|");

    let z_v04 = Zensim::new(ZensimProfile::PreviewV0_4);

    for ds in &datasets {
        let n = ds.pairs.len();
        eprintln!("=== {} (n={n}) ===", ds.name);
        let started = std::time::Instant::now();
        let progress = AtomicUsize::new(0);
        let log_every = (n / 20).max(1);

        let results: Vec<Option<Row>> = ds
            .pairs
            .par_iter()
            .map(|pair| {
                let p = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if p.is_multiple_of(log_every) {
                    let elapsed = started.elapsed().as_secs_f64();
                    let rate = p as f64 / elapsed;
                    let eta = (n - p) as f64 / rate;
                    eprintln!("  {p}/{n} ({rate:.1}/s, ETA {eta:.0}s)");
                }
                process_pair(pair, &z_v04, &v04_bake_bytes)
            })
            .collect();

        let valid: Vec<(usize, Row)> = results
            .into_iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|row| (i, row)))
            .collect();
        let n_valid = valid.len();
        if n_valid < 4 {
            println!(
                "| {} | {} | n/a (only {} valid) | | | | | |",
                ds.name, n, n_valid
            );
            continue;
        }

        let v02_d: Vec<f64> = valid.iter().map(|(_, r)| r.v02_distance).collect();
        let v04_d: Vec<f64> = valid.iter().map(|(_, r)| r.v04_distance).collect();
        let v02_s: Vec<f64> = valid.iter().map(|(_, r)| r.v02_score).collect();
        let v04_s: Vec<f64> = valid.iter().map(|(_, r)| r.v04_score).collect();

        let r_dist = pearson(&v02_d, &v04_d);
        let tau_dist = kendall_tau(&v02_d, &v04_d);
        let r_score = pearson(&v02_s, &v04_s);

        let deltas: Vec<f64> = v02_s.iter().zip(&v04_s).map(|(a, b)| b - a).collect();
        let mean_delta = mean(&deltas);
        let median_delta = median(&deltas);
        let sigma_delta = stddev(&deltas, mean_delta);

        println!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:+.2} | {:+.2} | {:.2} |",
            ds.name, n_valid, r_dist, tau_dist, r_score, mean_delta, median_delta, sigma_delta
        );

        for (i, row) in &valid {
            let pair = &ds.pairs[*i];
            all_rows_csv.push((
                ds.name.to_string(),
                pair.reference
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                pair.codec_class.clone(),
                pair.quality_class.clone(),
                pair.human_score,
                *row,
            ));
        }

        eprintln!(
            "  done {n_valid}/{n} valid in {:.1}s",
            started.elapsed().as_secs_f64()
        );
    }

    // Pairwise SROCC — for each (reference, A, B) triplet sharing a
    // reference, compute (human_A - human_B) vs (metric_A - metric_B).
    // This is what codec A/B gates care about — "is encoder A better
    // than encoder B at this q on this image?" — and the paper notes
    // metrics correlate BETTER with pairwise diffs than absolute MCOS
    // (Cloudinary CID22 paper, Table 6 vs Table 3).
    //
    // Important framing caveat: KADID and TID2013 contain very few
    // compression-relevant pairs (paper p. 2: <5% of KADID, similar
    // TID). Their numbers below measure V0_4 against synthetic
    // distortions like Gaussian noise, blur, color quantization — not
    // codec output. CID22 is the only included corpus where the
    // pairwise SROCC genuinely answers the codec-selection question.
    println!();
    println!("## Pairwise SROCC against human MOS");
    println!();
    println!(
        "For each (reference, A, B) triplet sharing a reference image, compute the\n\
         signed differences `(human_A − human_B)` and `(distance_A − distance_B)` and\n\
         report `|SROCC|` between them. This is the codec-A-vs-B prediction skill —\n\
         what downstream codec gates actually need.\n\
         \n\
         Per the Cloudinary CID22 paper (Table 6 in Sneyers et al. 2023), pairwise\n\
         correlation is generally HIGHER than absolute correlation (Table 3) since\n\
         relative comparisons within a reference cancel image-content variance.\n\
         \n\
         **CID22 is the relevant column** for codec evaluation. KADID and TID2013\n\
         contain mostly non-compression distortions (paper p. 2: <5%% of KADID images\n\
         are compression-relevant; TID2013 similar). High SROCC there shows the MLP\n\
         can rank synthetic distortions but doesn't validate codec performance.\n"
    );
    println!(
        "| Dataset | n triplets | abs SRCC V0_2 | abs SRCC V0_4 | **pairwise SRCC V0_2** | **pairwise SRCC V0_4** | pairwise τ V0_2-vs-V0_4 |"
    );
    println!("|---------|--:|:--:|:--:|:--:|:--:|:--:|");
    type RowRef<'a> = &'a (String, String, String, String, f64, Row);
    let mut by_ref: BTreeMap<(String, String), Vec<RowRef<'_>>> = BTreeMap::new();
    for row in &all_rows_csv {
        by_ref
            .entry((row.0.clone(), row.1.clone()))
            .or_default()
            .push(row);
    }
    for ds in &datasets {
        // Absolute SROCC against human MOS — per-pair, matches the
        // published numbers in dataset_metric_baseline.rs.
        let pairs: Vec<RowRef<'_>> = all_rows_csv.iter().filter(|row| row.0 == ds.name).collect();
        if pairs.len() < 4 {
            println!("| {} | n/a | | | | | |", ds.name);
            continue;
        }
        let humans: Vec<f64> = pairs.iter().map(|r| r.4).collect();
        let v02_d: Vec<f64> = pairs.iter().map(|r| r.5.v02_distance).collect();
        let v04_d: Vec<f64> = pairs.iter().map(|r| r.5.v04_distance).collect();
        let abs_v02 = spearman(&humans, &v02_d).abs();
        let abs_v04 = spearman(&humans, &v04_d).abs();

        // Pairwise SROCC — generate all within-reference triplets.
        let mut h_diffs: Vec<f64> = Vec::new();
        let mut v02_d_diffs: Vec<f64> = Vec::new();
        let mut v04_d_diffs: Vec<f64> = Vec::new();
        for ((d_name, _ref), group) in &by_ref {
            if d_name != ds.name {
                continue;
            }
            for i in 0..group.len() {
                for j in (i + 1)..group.len() {
                    let row_a = group[i];
                    let row_b = group[j];
                    h_diffs.push(row_a.4 - row_b.4);
                    v02_d_diffs.push(row_a.5.v02_distance - row_b.5.v02_distance);
                    v04_d_diffs.push(row_a.5.v04_distance - row_b.5.v04_distance);
                }
            }
        }
        let n_triplets = h_diffs.len();
        let pw_v02 = spearman(&h_diffs, &v02_d_diffs).abs();
        let pw_v04 = spearman(&h_diffs, &v04_d_diffs).abs();
        let pw_v04_v02 = spearman(&v02_d_diffs, &v04_d_diffs).abs();
        println!(
            "| {} | {n_triplets} | {abs_v02:.4} | {abs_v04:.4} | **{pw_v02:.4}** | **{pw_v04:.4}** | {pw_v04_v02:.4} |",
            ds.name
        );
    }

    // V0_2 score distribution — show where pairs actually live across
    // the score range so the threshold table has visible support.
    println!();
    println!("## V0_2 score distribution across the corpus");
    println!();
    println!(
        "Step-5 buckets across the full V0_2 score range. Distortion sweeps must cover\n\
         the low-q regime with the same density as high-q (CLAUDE.md sweep rule).\n\
         A bucket with n=0 means no pairs in that V0_2 score band — the corpus doesn't\n\
         exercise that regime, NOT that V0_4 has been calibrated there."
    );
    println!();
    println!("| V0_2 bucket | n | V0_4 score p10 | p25 | median | p75 | p90 | mean Δ |");
    println!("|:--:|--:|:--:|:--:|:--:|:--:|:--:|:--:|");
    let mut bucket_lo = -100.0_f64;
    while bucket_lo < 100.0 {
        let bucket_hi = bucket_lo + 5.0;
        let mut band: Vec<f64> = all_rows_csv
            .iter()
            .filter(|(_, _, _, _, _, r)| r.v02_score >= bucket_lo && r.v02_score < bucket_hi)
            .map(|(_, _, _, _, _, r)| r.v04_score)
            .collect();
        let band_n = band.len();
        let mean_delta_in_band = if band_n == 0 {
            0.0
        } else {
            let sum: f64 = all_rows_csv
                .iter()
                .filter(|(_, _, _, _, _, r)| r.v02_score >= bucket_lo && r.v02_score < bucket_hi)
                .map(|(_, _, _, _, _, r)| r.v04_score - r.v02_score)
                .sum();
            sum / band_n as f64
        };
        if band_n == 0 {
            println!("| [{bucket_lo:.0}, {bucket_hi:.0}) | 0 | — | — | — | — | — | — |");
        } else if band_n < 5 {
            band.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = band[band_n / 2];
            println!(
                "| [{bucket_lo:.0}, {bucket_hi:.0}) | {band_n} | — | — | {p50:.1} | — | — | {mean_delta_in_band:+.2} |"
            );
        } else {
            band.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p10 = band[band_n / 10];
            let p25 = band[band_n / 4];
            let p50 = band[band_n / 2];
            let p75 = band[band_n * 3 / 4];
            let p90 = band[band_n * 9 / 10];
            println!(
                "| [{bucket_lo:.0}, {bucket_hi:.0}) | {band_n} | {p10:.1} | {p25:.1} | {p50:.1} | {p75:.1} | {p90:.1} | {mean_delta_in_band:+.2} |"
            );
        }
        bucket_lo += 5.0;
    }
    let n_clamped: usize = all_rows_csv
        .iter()
        .filter(|(_, _, _, _, _, r)| r.v02_score <= -99.999)
        .count();
    if n_clamped > 0 {
        let mut clamped_v04: Vec<f64> = all_rows_csv
            .iter()
            .filter(|(_, _, _, _, _, r)| r.v02_score <= -99.999)
            .map(|(_, _, _, _, _, r)| r.v04_score)
            .collect();
        clamped_v04.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p10 = clamped_v04[clamped_v04.len() / 10];
        let p50 = clamped_v04[clamped_v04.len() / 2];
        let p90 = clamped_v04[clamped_v04.len() * 9 / 10];
        println!(
            "\n**{n_clamped} pairs at V0_2 = -100 (clamped)** — V0_4 score for these: p10 = {p10:.1}, median = {p50:.1}, p90 = {p90:.1}.\n\
             V0_2's score mapping has flattened these to a single value; V0_4's response on these\n\
             pairs is the only signal left for distinguishing 'bad' from 'completely broken'."
        );
    }

    // Per-(codec, quality_level) bias breakdown — keeps strength and codec
    // separable so we can see if V0_4 disagreement scales with distortion
    // strength (it does — heavily) rather than burying that under codec means.
    println!();
    println!("## Per-(codec, quality_level) score bias (V0_4 − V0_2)");
    println!();
    println!(
        "Median V0_4-minus-V0_2 score delta grouped by `dataset:codec:quality_level`.\n\
         For KADIK/TID, level = distortion strength 01..05 (01 = mildest, 05 = harshest).\n\
         For CID22, level = encoder quality. Sorted by |median Δ| so the most-disagreeing\n\
         (codec, level) cells surface first. Large positive Δ = V0_4 more lenient; negative = V0_4 harsher."
    );
    println!();
    let mut by_codec_q: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (ds, _ref_name, codec, q, _h, row) in &all_rows_csv {
        let key = format!("{ds}:{codec}:q={q}");
        by_codec_q
            .entry(key)
            .or_default()
            .push(row.v04_score - row.v02_score);
    }
    println!("| dataset:codec:level | n | median Δ | p25 Δ | p75 Δ |");
    println!("|---|--:|:----:|:----:|:----:|");
    let mut codec_q_lines: Vec<(String, usize, f64, f64, f64)> = by_codec_q
        .into_iter()
        .filter(|(_, v)| v.len() >= 20)
        .map(|(k, mut v)| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = v.len();
            (k, n, v[n / 4], v[n / 2], v[n * 3 / 4])
        })
        .collect();
    codec_q_lines.sort_by(|a, b| b.3.abs().partial_cmp(&a.3.abs()).unwrap());
    for (k, n, p25, p50, p75) in codec_q_lines.iter().take(40) {
        println!("| {k} | {n} | {p50:+.2} | {p25:+.2} | {p75:+.2} |");
    }

    // Also show codec-aggregated table, but with explicit breakdown of how
    // the median changes with distortion strength so it's clear it's not
    // a constant offset.
    println!();
    println!("### Codec-aggregated medians (collapses level — read with caution)");
    println!();
    let mut by_codec: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (ds, _ref_name, codec, _q, _h, row) in &all_rows_csv {
        let key = format!("{ds}:{codec}");
        by_codec
            .entry(key)
            .or_default()
            .push(row.v04_score - row.v02_score);
    }
    println!("| dataset:codec | n | median Δ | p25 Δ | p75 Δ |");
    println!("|---|--:|:----:|:----:|:----:|");
    let mut codec_lines: Vec<(String, usize, f64, f64, f64)> = by_codec
        .into_iter()
        .filter(|(_, v)| v.len() >= 20)
        .map(|(k, mut v)| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = v.len();
            (k, n, v[n / 4], v[n / 2], v[n * 3 / 4])
        })
        .collect();
    codec_lines.sort_by(|a, b| b.3.abs().partial_cmp(&a.3.abs()).unwrap());
    for (k, n, p25, p50, p75) in codec_lines.iter().take(20) {
        println!("| {k} | {n} | {p50:+.2} | {p25:+.2} | {p75:+.2} |");
    }

    // Top-K largest disagreements (in score units).
    println!();
    println!("## Top-{top_k} largest score disagreements");
    println!();
    println!(
        "Pairs where V0_2 and V0_4 score the SAME comparison most differently. Useful for spotting failure modes."
    );
    println!();
    let mut by_disagreement: Vec<RowRef<'_>> = all_rows_csv.iter().collect();
    by_disagreement.sort_by(|a, b| {
        let da = (a.5.v04_score - a.5.v02_score).abs();
        let db = (b.5.v04_score - b.5.v02_score).abs();
        db.partial_cmp(&da).unwrap()
    });
    println!("| dataset | reference | codec | quality | V0_2 score | V0_4 score | Δ |");
    println!("|---|---|---|---|--:|--:|--:|");
    for (ds, ref_name, codec, q, _h, row) in by_disagreement.iter().take(top_k) {
        let delta = row.v04_score - row.v02_score;
        println!(
            "| {ds} | {ref_name} | {codec} | {q} | {:.2} | {:.2} | {delta:+.2} |",
            row.v02_score, row.v04_score
        );
    }

    // ---- KonJND-1k visually-lossless calibration ----
    //
    // KonJND-1k pairs at the per-source mean PJND threshold are the
    // canonical "near visually lossless" anchors. The Cloudinary CID22
    // paper (Sneyers, Ben Baruch, Vaxman, 2023, Table 4) publishes
    // mean ± stdev for each metric on this corpus. We compute the same
    // for V0_2 and V0_4 so both can be located on the same external
    // calibration anchor.
    if let Some(p) = konjnd {
        let pairs = load_konjnd(&p, max_pairs);
        let n_total = pairs.len();
        eprintln!("=== KonJND-1k (n={n_total}) ===");
        if n_total < 4 {
            println!();
            println!("## Visually-lossless calibration (KonJND-1k)");
            println!();
            println!(
                "Loaded only {n_total} pairs from {} — skipping.",
                p.display()
            );
        } else {
            let started = std::time::Instant::now();
            let progress = AtomicUsize::new(0);
            let log_every = (n_total / 20).max(1);
            let results: Vec<Option<Row>> = pairs
                .par_iter()
                .map(|pair| {
                    let p = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if p.is_multiple_of(log_every) {
                        let elapsed = started.elapsed().as_secs_f64();
                        let rate = p as f64 / elapsed;
                        let eta = (n_total - p) as f64 / rate;
                        eprintln!("  konjnd {p}/{n_total} ({rate:.1}/s, ETA {eta:.0}s)");
                    }
                    process_pair(pair, &z_v04, &v04_bake_bytes)
                })
                .collect();
            let scored: Vec<(&Pair, Row)> = pairs
                .iter()
                .zip(results.iter())
                .filter_map(|(p, r)| r.as_ref().map(|row| (p, *row)))
                .collect();
            eprintln!(
                "  konjnd done {n_scored}/{n_total} valid in {:.1}s",
                started.elapsed().as_secs_f64(),
                n_scored = scored.len()
            );

            println!();
            println!("## Visually-lossless calibration (KonJND-1k)");
            println!();
            println!(
                "1008 source images split into 504 JPEG and 504 BPG (no overlap). For each\n\
                 source, the Probabilistic Just-Noticeable-Difference (PJND) threshold is the\n\
                 mean file index where observers report just noticing the compression artifact\n\
                 ([Lin, Hosu, Saupe, IEEE T-CSVT 2022](https://ieeexplore.ieee.org/document/9802742)).\n\
                 Pairs below are at `round(mean PJND)` per source — the canonical near\n\
                 visually-lossless anchor.\n\
                 \n\
                 Cloudinary CID22 paper Table 4 publishes the same anchor for nine reference\n\
                 metrics. Numbers below place V0_2 and V0_4 on the same external scale; mean ±\n\
                 stdev that's a tight band means the metric agrees with the human PJND notion\n\
                 of visually-lossless (low cross-source variance), regardless of where the mean\n\
                 lands on the 0-100 score scale."
            );
            println!();
            for codec in ["JPEG", "BPG"] {
                let key = format!("konjnd:{codec}");
                let subset: Vec<&(&Pair, Row)> = scored
                    .iter()
                    .filter(|(p, _)| p.codec_class == key)
                    .collect();
                let n = subset.len();
                if n == 0 {
                    continue;
                }
                let v02_d: Vec<f64> = subset.iter().map(|(_, r)| r.v02_distance).collect();
                let v04_d: Vec<f64> = subset.iter().map(|(_, r)| r.v04_distance).collect();
                let v02_s: Vec<f64> = subset.iter().map(|(_, r)| r.v02_score).collect();
                let v04_s: Vec<f64> = subset.iter().map(|(_, r)| r.v04_score).collect();
                println!("### {codec} subset (n = {n})");
                println!();
                let m_v02_d = mean(&v02_d);
                let s_v02_d = stddev(&v02_d, m_v02_d);
                let m_v04_d = mean(&v04_d);
                let s_v04_d = stddev(&v04_d, m_v04_d);
                let m_v02_s = mean(&v02_s);
                let s_v02_s = stddev(&v02_s, m_v02_s);
                let m_v04_s = mean(&v04_s);
                let s_v04_s = stddev(&v04_s, m_v04_s);
                println!("| metric | mean | stdev |");
                println!("|---|--:|--:|");
                println!("| V0_2 raw distance | {m_v02_d:.4} | {s_v02_d:.4} |");
                println!("| V0_4 raw distance | {m_v04_d:.4} | {s_v04_d:.4} |");
                println!("| V0_2 score | {m_v02_s:.2} | {s_v02_s:.2} |");
                println!("| V0_4 score | {m_v04_s:.2} | {s_v04_s:.2} |");
                println!();
                let table4 = match codec {
                    "BPG" => {
                        "Cloudinary Table 4 reference values for BPG at PJND:\n\
                         - SSIMULACRA 2: 65.38 ± 5.10\n\
                         - DSSIM ×1000: 3.357 ± 1.267\n\
                         - Butteraugli 3-norm: 1.528 ± 0.192\n\
                         - MS-SSIM ×100: 99.21 ± 0.40\n\
                         - VMAF: 90.05 ± 2.25\n\
                         - PSNR-Y: 39.61 ± 2.98\n\
                         - PSNR-HVS: 40.31 ± 1.78"
                    }
                    "JPEG" => {
                        "Cloudinary Table 4 reference values for JPEG at PJND:\n\
                         - SSIMULACRA 2: 63.10 ± 4.65\n\
                         - DSSIM ×1000: 3.817 ± 1.297\n\
                         - Butteraugli 3-norm: 1.699 ± 0.229\n\
                         - MS-SSIM ×100: 99.22 ± 0.38\n\
                         - VMAF: 91.86 ± 1.90\n\
                         - PSNR-Y: 36.70 ± 3.79\n\
                         - PSNR-HVS: 39.96 ± 1.79"
                    }
                    _ => "",
                };
                println!("{table4}");
                println!();
            }
        }
    }

    // Write the per-pair CSV for downstream analysis.
    if let Some(out) = output {
        eprintln!("writing {} rows to {}", all_rows_csv.len(), out.display());
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut wtr = csv::Writer::from_path(&out).unwrap_or_else(|e| {
            eprintln!("failed to open {} for write: {e}", out.display());
            std::process::exit(1);
        });
        wtr.write_record([
            "dataset",
            "reference",
            "codec_class",
            "quality_class",
            "v02_distance",
            "v04_distance",
            "v02_score",
            "v04_score",
        ])
        .unwrap();
        for (ds, ref_name, codec, q, _h, row) in &all_rows_csv {
            wtr.write_record([
                ds,
                ref_name,
                codec,
                q,
                &format!("{:.6}", row.v02_distance),
                &format!("{:.6}", row.v04_distance),
                &format!("{:.4}", row.v02_score),
                &format!("{:.4}", row.v04_score),
            ])
            .unwrap();
        }
        wtr.flush().unwrap();
    }
}

fn process_pair(pair: &Pair, z_v04: &Zensim, v04_bake: &[u8]) -> Option<Row> {
    let src_img = match image::open(&pair.reference) {
        Ok(img) => img.to_rgb8(),
        Err(_) => return None,
    };
    let dst_img = match image::open(&pair.distorted) {
        Ok(img) => img.to_rgb8(),
        Err(_) => return None,
    };
    let (w, h) = src_img.dimensions();
    let (dw, dh) = dst_img.dimensions();
    if w != dw || h != dh {
        return None;
    }
    let src_pixels: Vec<[u8; 3]> = src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dst_pixels: Vec<[u8; 3]> = dst_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let w_us = w as usize;
    let h_us = h as usize;
    if w_us < 8 || h_us < 8 {
        return None;
    }
    let s = RgbSlice::new(&src_pixels, w_us, h_us);
    let d = RgbSlice::new(&dst_pixels, w_us, h_us);

    let result = z_v04.compute(&s, &d).ok()?;
    let features = result.features();
    let n_v02 = zensim::profile::LINEAR_WEIGHTS_PREVIEW_V0_2.len();
    if features.len() < n_v02 {
        return None;
    }

    let v02_dot: f64 = features[..n_v02]
        .iter()
        .zip(zensim::profile::LINEAR_WEIGHTS_PREVIEW_V0_2.iter())
        .map(|(f, w)| f * w)
        .sum();
    let v02_distance = v02_dot / 4.0;

    let model = Model::from_bytes(v04_bake).ok()?;
    let n_inputs = model.n_inputs();
    if features.len() < n_inputs {
        return None;
    }
    let f32_features: Vec<f32> = features[..n_inputs].iter().map(|&v| v as f32).collect();
    let mut p = Predictor::new(model);
    let v04_distance = p.predict(&f32_features).ok()?[0] as f64;

    let v02_score = distance_to_score(v02_distance);
    let v04_score = distance_to_score(v04_distance);

    Some(Row {
        v02_distance,
        v04_distance,
        v02_score,
        v04_score,
    })
}

fn distance_to_score(d: f64) -> f64 {
    if d <= 0.0 {
        100.0
    } else {
        (100.0 - 18.0 * d.powf(0.7)).clamp(-100.0, 100.0)
    }
}

// ---- dataset loaders ----

fn load_kadid(base: &Path, max: usize) -> Vec<Pair> {
    let csv_path = base.join("dmos.csv");
    let mut rdr = match csv::Reader::from_path(&csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        if record.len() < 3 {
            continue;
        }
        let dist = record.get(0).unwrap();
        let r = record.get(1).unwrap();
        let dmos: f64 = match record.get(2).unwrap().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        // KADIK file pattern: `IXX_DD_LL.png`. DD = distortion type
        // 01..25, LL = level 01..05.
        let (codec, quality) = parse_kadid_name(dist);
        pairs.push(Pair {
            reference: base.join("images").join(r),
            distorted: base.join("images").join(dist),
            codec_class: codec,
            quality_class: quality,
            human_score: dmos,
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

fn parse_kadid_name(name: &str) -> (String, String) {
    let stem = name.trim_end_matches(".png");
    let parts: Vec<&str> = stem.split('_').collect();
    if parts.len() == 3 {
        (format!("kadid:{}", parts[1]), parts[2].to_string())
    } else {
        ("kadid:?".into(), "".into())
    }
}

fn load_tid(base: &Path, max: usize) -> Vec<Pair> {
    let mos_path = base.join("mos_with_names.txt");
    let content = match std::fs::read_to_string(&mos_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to read {}: {e}", mos_path.display());
            return Vec::new();
        }
    };
    let mut pairs = Vec::new();
    for line in content.lines() {
        let mut parts = line.split_whitespace();
        let mos_str = parts.next().unwrap_or("");
        let name = parts.next().unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let mos: f64 = match mos_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let prefix = name.split('_').next().unwrap_or("");
        let ref_name = format!("{}.BMP", prefix);
        let ref_path = base.join("reference_images").join(&ref_name);
        let ref_path = if ref_path.exists() {
            ref_path
        } else {
            base.join("reference_images").join(format!("{prefix}.bmp"))
        };
        // TID file pattern: `iXX_YY_Z.bmp` — YY = distortion type 01..24.
        let (codec, quality) = parse_tid_name(name);
        pairs.push(Pair {
            reference: ref_path,
            distorted: base.join("distorted_images").join(name),
            codec_class: codec,
            quality_class: quality,
            human_score: mos,
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

fn parse_tid_name(name: &str) -> (String, String) {
    let stem = name.trim_end_matches(".bmp").trim_end_matches(".BMP");
    let parts: Vec<&str> = stem.split('_').collect();
    if parts.len() == 3 {
        (format!("tid:{}", parts[1]), parts[2].to_string())
    } else {
        ("tid:?".into(), "".into())
    }
}

fn load_cid22(base: &Path, max: usize) -> Vec<Pair> {
    let csv_path = base.join("CID22_validation_set.csv");
    let mut rdr = match csv::Reader::from_path(&csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        if record.len() < 6 {
            continue;
        }
        let r = record.get(0).unwrap();
        let dist = record.get(1).unwrap();
        let encoder = record.get(2).unwrap();
        if encoder == "Reference" {
            continue;
        }
        // CID22 columns: reference_img, distorted_img, encoder, setting,
        // bpp, MCOS, RMOS, Elo, nb_pc_opinions
        let q = record.get(3).unwrap_or("").to_string();
        let mcos: f64 = match record.get(5).unwrap_or("").parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        pairs.push(Pair {
            reference: base.join(r),
            distorted: base.join(dist),
            codec_class: format!("cid22:{encoder}"),
            quality_class: q,
            human_score: mcos,
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// Load KonJND-1k pairs at the mean PJND threshold.
///
/// CSV format (`subjective_ratings.csv`):
///   `image_id,Compression type,No. of ratings,mean,std,ratings`
///
/// File layout under `base`:
///   `source_image/SRC{NNNN}.png`
///   `jpeg/SRC{NNNN}_JPEG_{NNN}.jpg`     — 504 sources, 100 levels each
///   `bpg/SRC{NNNN}_BPG_{NNN}.png`       — 504 sources, 51 levels each
///
/// `mean` is the average index where observers just barely notice the
/// distortion. We pair the source with the distorted image at
/// `round(mean)` — that's the just-noticeable-difference pair, the
/// canonical "near visually lossless" anchor for this corpus.
///
/// Sources are split disjointly between the JPEG and BPG subsets — each
/// SRC**** appears in exactly one of them.
///
/// Polarity note: `human_score` here is the PJND threshold value (1..N
/// where N is the number of distortion levels). Higher threshold = the
/// codec degrades more gracefully on this source. It is NOT a quality
/// MOS, so SROCC against it has different semantics than KADID/TID/CID22
/// — the visually-lossless calibration section treats it accordingly.
fn load_konjnd(base: &Path, max: usize) -> Vec<Pair> {
    let csv_path = base.join("subjective_ratings.csv");
    let mut rdr = match csv::Reader::from_path(&csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        if record.len() < 5 {
            continue;
        }
        let image_id = record.get(0).unwrap_or(""); // e.g. "SRC0001.png"
        let comp = record.get(1).unwrap_or(""); // "JPEG" or "BPG"
        let mean: f64 = match record.get(3).unwrap_or("").parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let stem = image_id.trim_end_matches(".png");
        if stem.is_empty() {
            continue;
        }
        let level = mean.round().clamp(1.0, 100.0) as u32;
        let (subdir, ext) = match comp {
            "JPEG" => ("jpeg", "jpg"),
            "BPG" => ("bpg", "png"),
            _ => continue,
        };
        let dist_name = format!("{stem}_{comp}_{level:03}.{ext}");
        let ref_path = base.join("source_image").join(image_id);
        let dist_path = base.join(subdir).join(&dist_name);
        if !dist_path.exists() {
            // out-of-range threshold (e.g., BPG level above 51) — skip
            continue;
        }
        pairs.push(Pair {
            reference: ref_path,
            distorted: dist_path,
            codec_class: format!("konjnd:{comp}"),
            quality_class: format!("{level:03}"),
            human_score: mean,
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

// ---- statistics ----

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    pearson(&ra, &rb)
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0_f64; n];
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

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ma = mean(a);
    let mb = mean(b);
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    // O(n²) — fine for n up to ~30k. Concordance over all pairs.
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    let mut tied_a: i64 = 0;
    let mut tied_b: i64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let da = a[j] - a[i];
            let db = b[j] - b[i];
            if da == 0.0 && db == 0.0 {
                continue;
            } else if da == 0.0 {
                tied_a += 1;
            } else if db == 0.0 {
                tied_b += 1;
            } else if da.signum() == db.signum() {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let denom_a = (concordant + discordant + tied_a) as f64;
    let denom_b = (concordant + discordant + tied_b) as f64;
    let den = (denom_a * denom_b).sqrt();
    if den < 1e-12 {
        0.0
    } else {
        (concordant - discordant) as f64 / den
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn stddev(v: &[f64], m: f64) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}
