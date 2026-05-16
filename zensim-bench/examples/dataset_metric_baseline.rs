//! Baseline-metric SROCC on the human-rated holdout datasets.
//!
//! For each of KADIK10k, TID2013, and CID22-validation, this binary:
//!
//! 1. Loads the (reference, distorted, human_score) pairs from CSV.
//! 2. Decodes images, runs zensim V0_2, V0_4 (trained bake), fast-ssim2
//!    and butteraugli on each pair.
//! 3. Computes SROCC of each metric's distance against `human_score`.
//! 4. Prints a comparison table.
//!
//! Caveats when interpreting these numbers:
//! - SSIMULACRA 2's weights were tuned on 201 of 250 CID22 reference
//!   images (Sneyers, Ben Baruch, Vaxman, 2023, *AIC-3 Contribution from
//!   Cloudinary: CID22*, p. 26). CID22 SROCC for SSIMULACRA 2 is therefore
//!   not a fair held-out evaluation; KADID / TID2013 are unbiased
//!   baselines for it.
//! - KADID and TID2013 contain mostly non-compression distortions (same
//!   paper, p. 2: <5% of KADID images are compression-relevant; TID2013
//!   similar). High SROCC there validates a metric's handling of
//!   synthetic distortions like blur and noise, not codec output.
//! - Absolute SROCC is what this binary reports. Pairwise
//!   correlation — `(metric(R,A) − metric(R,B))` vs `(MOS(A) − MOS(B))`
//!   for triplets sharing a reference — is what codec A/B selection
//!   actually needs, and metrics generally rank differently on it.
//!   See `profile_compat_report` for pairwise numbers.
//!
//! Image decoding is the dominant cost; pairs are processed in
//! parallel via rayon.
//!
//! Usage:
//!   cargo run --release -p zensim-bench --example dataset_metric_baseline -- \
//!     --kadid /mnt/v/dataset/kadid10k \
//!     --tid /mnt/v/dataset/tid2013 \
//!     --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
//!     --v04-bake /mnt/v/output/zensim/synthetic-v2/runs/v04_mlp_v5znpr2_20260430T044620.bin \
//!     --max-pairs 500

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use butteraugli::ButteraugliParams;
use imgref::Img;
use rayon::prelude::*;
use rgb::RGB8;
use zenpredict::{Model, Predictor};
use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

#[derive(Debug, Clone)]
struct Pair {
    reference: PathBuf,
    distorted: PathBuf,
    human_score: f64,
    /// Optional codec / encoder label (CID22 sets it; others may leave None).
    codec: Option<String>,
    /// Optional version / setting label (CID22's "setting" column; q30, e7_q30, …).
    version: Option<String>,
}

#[derive(Debug, Clone)]
struct DatasetSpec {
    name: &'static str,
    pairs: Vec<Pair>,
}

/// Calibration pair from KonJND-1k. Carries the codec subset (JPEG or
/// BPG) — not used for SROCC, only for grouping in the calibration
/// table.
#[derive(Debug, Clone)]
struct KonJndPair {
    pair: Pair,
    codec: String,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut kadid: Option<PathBuf> = None;
    let mut tid: Option<PathBuf> = None;
    let mut cid22: Option<PathBuf> = None;
    let mut csiq: Option<PathBuf> = None;
    let mut aic3: Option<PathBuf> = None;
    let mut konjnd: Option<PathBuf> = None;
    let mut v04_bake_path: Option<PathBuf> = None;
    /// When set, drive the V_04 column via `Zensim::compute()` using
    /// the named profile's live runtime — bypasses `--v04-bake`. Used
    /// to eval the D2 multi-bake ensemble (PreviewV0_4 = V_18 ship +
    /// V_20 IS calibrated mixed at α=0.7).
    let mut zensim_profile_override: Option<zensim::ZensimProfile> = None;
    let mut max_pairs: usize = 500;
    let mut per_pair_output: Option<PathBuf> = None;
    let mut konjnd_features_csv: Option<PathBuf> = None;
    let mut konjnd_anchor_target: f64 = 63.0;
    /// Generic `--pairs-tsv <NAME>:<PATH>` flag, repeatable.
    /// TSV columns: ref_path, dist_path, codec, [version|image_name|...], human_score
    /// Order: cols 0/1 are ref/dist; the LAST column is treated as human_score
    /// (a float in any scale; SROCC is rank-invariant). Optional codec column at
    /// index 2 and version column at index 3 (or 4 if there's an image_name col).
    let mut pairs_tsv: Vec<(String, PathBuf)> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--kadid" => kadid = Some(args.next().unwrap().into()),
            "--tid" => tid = Some(args.next().unwrap().into()),
            "--cid22" => cid22 = Some(args.next().unwrap().into()),
            "--csiq" => csiq = Some(args.next().unwrap().into()),
            "--aic3" => aic3 = Some(args.next().unwrap().into()),
            "--konjnd" => konjnd = Some(args.next().unwrap().into()),
            "--v04-bake" => v04_bake_path = Some(args.next().unwrap().into()),
            "--zensim-profile" => {
                let v = args.next().expect("--zensim-profile <v0-3|v0-4>");
                zensim_profile_override = Some(match v.as_str() {
                    "v0-3" | "v03" | "preview-v0.3" => zensim::ZensimProfile::PreviewV0_3,
                    "v0-4" | "v04" | "preview-v0.4" => zensim::ZensimProfile::PreviewV0_4,
                    other => {
                        eprintln!("--zensim-profile must be v0-3 or v0-4, got {other:?}");
                        std::process::exit(2);
                    }
                });
            }
            "--max-pairs" => max_pairs = args.next().unwrap().parse().unwrap(),
            "--per-pair-output" => per_pair_output = Some(args.next().unwrap().into()),
            "--konjnd-features-csv" => konjnd_features_csv = Some(args.next().unwrap().into()),
            "--konjnd-anchor-target" => {
                konjnd_anchor_target = args.next().unwrap().parse().unwrap()
            }
            "--pairs-tsv" => {
                let arg = args.next().expect("--pairs-tsv NAME:PATH");
                if let Some(idx) = arg.find(':') {
                    let (name, path) = arg.split_at(idx);
                    pairs_tsv.push((name.to_string(), PathBuf::from(&path[1..])));
                } else {
                    eprintln!("--pairs-tsv expects NAME:PATH (got {arg})");
                    std::process::exit(1);
                }
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    let mut datasets = Vec::new();
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
    // Generic TSV input — leak the name string to satisfy the &'static
    // lifetime on `DatasetSpec.name` (this is a one-shot CLI tool).
    for (name, path) in &pairs_tsv {
        let pairs = load_pairs_tsv(path, max_pairs);
        let leaked: &'static str = Box::leak(name.clone().into_boxed_str());
        datasets.push(DatasetSpec {
            name: leaked,
            pairs,
        });
    }
    if let Some(p) = csiq {
        datasets.push(DatasetSpec {
            name: "CSIQ",
            pairs: load_csiq(&p, max_pairs),
        });
    }
    if let Some(p) = aic3 {
        datasets.push(DatasetSpec {
            name: "AIC-3 CTC",
            pairs: load_aic3(&p, max_pairs),
        });
    }

    if datasets.is_empty() && konjnd.is_none() {
        eprintln!(
            "no datasets — pass at least one of --kadid, --tid, --cid22, --csiq, --aic3, --konjnd"
        );
        std::process::exit(1);
    }

    // Optionally load V0_4 trained bake.
    let v04_bake_bytes: Option<Vec<u8>> = v04_bake_path.as_ref().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("failed to read v04 bake at {:?}: {e}", p);
            std::process::exit(1);
        })
    });

    // Inspect the bake's input width ONCE to pick the feature regime
    // the per-pair compute must produce. 228 = standard, 300 = extended
    // (adds 72 masked), 372 = extended + IW (Wang & Li 2011). Without
    // this dispatch, IW bakes load fine but get fed truncated features
    // and produce noise (SROCC ≈ 0.01).
    let feature_regime: FeatureRegime = v04_bake_bytes
        .as_deref()
        .and_then(|bytes| Model::from_bytes(bytes).ok())
        .map(|model| {
            let n = model.n_inputs();
            let r = FeatureRegime::from_n_inputs(n);
            eprintln!("v04 bake: n_inputs={n} → feature regime {r:?}");
            r
        })
        .unwrap_or(FeatureRegime::Standard);

    println!(
        "# Baseline-metric SROCC vs human MOS\n\nMax {} pairs per dataset.",
        max_pairs
    );
    println!();

    println!("| Dataset | n | V0_2 | V0_4 (bake) | fast-ssim2 | butteraugli |");
    println!("|---------|--:|:----:|:-----------:|:----------:|:-----------:|");

    // Optional per-pair CSV: dataset, reference, codec, quality, human, v02_dist, v04_dist, ssim2, butter
    let mut per_pair_writer: Option<csv::Writer<std::fs::File>> =
        per_pair_output.as_ref().map(|p| {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let mut w = csv::Writer::from_path(p).expect("open per-pair csv");
            w.write_record([
                "dataset",
                "reference",
                "distorted",
                "codec",
                "version",
                "human_score",
                "v02_distance",
                "v04_distance",
                "fast_ssim2_score",
                "butter_3norm",
            ])
            .unwrap();
            w
        });

    for ds in &datasets {
        let n = ds.pairs.len();
        eprintln!("=== {} (n={n}) ===", ds.name);
        let started = std::time::Instant::now();
        let progress = AtomicUsize::new(0);
        let log_every = (n / 10).max(1);

        // PreviewV0_4 forces feature population (the MLP path needs them).
        // We use it to extract features once per pair, then score V0_2,
        // V0_4 (trained), SSIM2, and Butteraugli on the same data.
        let z_v04 = Zensim::new(zensim_profile_override.unwrap_or(ZensimProfile::PreviewV0_3));

        type MetricRow = (f64, f64, f64, f64, f64);
        /// (ref_path, dist_path, codec, version, human, v02, v04, ssim2, butter)
        type EnrichedRow = (String, String, String, String, f64, f64, f64, f64, f64);
        let results: Vec<Option<EnrichedRow>> = ds
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
                let metrics: MetricRow =
                    process_pair(pair, &z_v04, v04_bake_bytes.as_deref(), feature_regime)?;
                Some((
                    pair.reference.to_string_lossy().to_string(),
                    pair.distorted.to_string_lossy().to_string(),
                    pair.codec.clone().unwrap_or_default(),
                    pair.version.clone().unwrap_or_default(),
                    metrics.0,
                    metrics.1,
                    metrics.2,
                    metrics.3,
                    metrics.4,
                ))
            })
            .collect();

        let pairs_with: Vec<EnrichedRow> = results.into_iter().flatten().collect();
        let n_valid = pairs_with.len();
        if n_valid < 4 {
            println!(
                "| {} | {} | n/a (only {} valid) | | | |",
                ds.name, n, n_valid
            );
            continue;
        }
        // EnrichedRow indices: 0=ref, 1=dist, 2=codec, 3=version, 4=human, 5=v02, 6=v04, 7=ssim2, 8=butter
        let humans: Vec<f64> = pairs_with.iter().map(|t| t.4).collect();
        let v02: Vec<f64> = pairs_with.iter().map(|t| t.5).collect();
        let v04: Vec<f64> = pairs_with.iter().map(|t| t.6).collect();
        let ssim2: Vec<f64> = pairs_with.iter().map(|t| t.7).collect();
        let butter: Vec<f64> = pairs_with.iter().map(|t| t.8).collect();

        // Spearman against human MOS — abs since each metric has its
        // own polarity (zensim is distance, ssim2 is score, butteraugli
        // is distance). We compare correlation magnitudes.
        let srocc_v02 = spearman(&humans, &v02).abs();
        let srocc_v04 = spearman(&humans, &v04).abs();
        let srocc_ssim2 = spearman(&humans, &ssim2).abs();
        let srocc_butter = spearman(&humans, &butter).abs();

        println!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} |",
            ds.name, n_valid, srocc_v02, srocc_v04, srocc_ssim2, srocc_butter
        );

        // CLAUDE.md "Statistical rigor" mandate (2026-05-14): emit
        // PLCC, KROCC, OR, PWRC alongside SROCC. Z-RMSE, MRR p-value,
        // Wilcoxon are queued (need per-stimulus σ + paired-test
        // infrastructure).
        println!();
        println!(
            "### {} full statistical panel (CLAUDE.md rigor mandate)",
            ds.name
        );
        println!();
        println!("| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |");
        println!("|---|---:|---:|---:|---:|---:|---:|");
        let metrics_for_panel: &[(&str, &Vec<f64>)] = &[
            ("V0_2", &v02),
            ("V0_4 (bake)", &v04),
            ("fast-ssim2", &ssim2),
            ("butteraugli", &butter),
        ];
        for (name, vals) in metrics_for_panel {
            let s = spearman(&humans, vals).abs();
            let p = pearson(&humans, vals).abs();
            let k = kendall_tau(&humans, vals).abs();
            let or_ = outlier_ratio(vals, &humans);
            let pw = pwrc(&humans, vals).abs();
            // Z-RMSE: rescale predictions to MOS units via 4-parameter
            // logistic (Mohammadi 2025 convention, eq. matches
            // `verify_mohammadi_anchor.py`). Nonlinear metrics
            // (PSNR-Y, Butteraugli, distance-based) get garbage
            // Z-RMSE with affine rescale because the saturation
            // region dominates the residual. Logistic absorbs that
            // saturation into the rescale so Z-RMSE measures
            // prediction error after the metric's natural shape.
            let rescaled = rescale_logistic(vals, &humans);
            let z = z_rmse(&rescaled, &humans, None);
            println!(
                "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.3} |",
                name, s, p, k, or_, pw, z
            );
        }
        println!();
        println!(
            "_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable on this corpus). \
On AIC-3 / AIC-4 / CID22 with bootstrap σ available, this becomes the σ-normalized form \
recommended by Mohammadi et al. 2025 (arXiv:2509.13150). Z-RMSE rescale is 4-parameter \
logistic (Mohammadi 2025 convention), NOT affine — affine inflates Z-RMSE on nonlinear \
metrics (PSNR-Y, Butteraugli) by 30× because saturation regions dominate the residual._"
        );
        println!();

        // MRR pairwise + Wilcoxon: rigorously test whether V0_4 (the
        // bake under test) beats each baseline. Both stats are
        // per CLAUDE.md "Statistical rigor" mandate.
        let v04_srocc = spearman(&humans, &v04).abs();
        println!(
            "### {} significance vs V0_4 bake (MRR + Wilcoxon, two-tailed)",
            ds.name
        );
        println!();
        println!(
            "| Comparison | SROCC_other | SROCC_V0_4 | MRR z | MRR p | Wilcoxon z | Wilcoxon p | effect r |"
        );
        println!("|---|---:|---:|---:|---:|---:|---:|---:|");
        for (name, other) in &[
            ("V0_2", &v02),
            ("fast-ssim2", &ssim2),
            ("butteraugli", &butter),
        ] {
            let r_other = spearman(&humans, other).abs();
            let r_v04 = v04_srocc;
            let r12 = spearman(other, &v04).abs();
            let (mrr_z, mrr_p, _dir) = mrr_test(r_other, r_v04, r12, n_valid);
            let (w_z, w_p, w_r) = wilcoxon_signed_rank(other, &v04, &humans);
            println!(
                "| {} vs V0_4 | {:.4} | {:.4} | {:.3} | {:.4} | {:.3} | {:.4} | {:.3} |",
                name, r_other, r_v04, mrr_z, mrr_p, w_z, w_p, w_r
            );
        }
        println!();

        if let Some(w) = per_pair_writer.as_mut() {
            for row in &pairs_with {
                w.write_record([
                    ds.name,
                    &row.0,
                    &row.1,
                    &row.2,
                    &row.3,
                    &format!("{:.6}", row.4),
                    &format!("{:.6}", row.5),
                    &format!("{:.6}", row.6),
                    &format!("{:.4}", row.7),
                    &format!("{:.6}", row.8),
                ])
                .unwrap();
            }
            w.flush().unwrap();
        }

        eprintln!(
            "  done {n_valid}/{n} valid in {:.1}s",
            started.elapsed().as_secs_f64()
        );

        // ---- Per-band reporting (per zensim/CLAUDE.md "Per-band reporting rule")
        //
        // Bands aligned to CID22 MCOS / SSIMULACRA 2 score thresholds via
        // Table 5: medium=50, high=65, visually-lossless=90. Each dataset's
        // human_score is normalized to its own scale; the band cuts below
        // are in *that* normalized scale.
        //
        //   CID22:  MCOS / 100      → cuts at 0.50, 0.65, 0.90 ;
        //                              Near-PJND [0.58, 0.68]
        //   CSIQ:   (1 - DMOS)      → use CID22 cuts directly (1:1 score map heuristic)
        //   KADID:  (DMOS - 1) / 4  → DMOS thresholds 3.7/4.3/4.5 → 0.675/0.825/0.875
        //   TID:    MOS / 9         → MOS thresholds 4.5/5.5/6.0 → 0.500/0.611/0.667
        let bands: Option<[(&str, f64, f64); 5]> = match ds.name {
            "CID22" | "CSIQ" => Some([
                ("B0 below medium (<50)", -f64::INFINITY, 0.50),
                ("B1 medium [50,65)", 0.50, 0.65),
                ("B2 high [65,90)", 0.65, 0.90),
                ("B3 visually-lossless (≥90)", 0.90, f64::INFINITY),
                ("Near-PJND [58,68]", 0.58, 0.68),
            ]),
            "KADIK10k" => Some([
                ("B0 below medium (<3.7)", -f64::INFINITY, 0.675),
                ("B1 medium [3.7,4.3)", 0.675, 0.825),
                ("B2 high [4.3,4.5)", 0.825, 0.875),
                ("B3 visually-lossless (≥4.5)", 0.875, f64::INFINITY),
                ("Near-PJND [3.9,4.2]", 0.725, 0.800),
            ]),
            "TID2013" => Some([
                ("B0 below medium (<4.5)", -f64::INFINITY, 0.500),
                ("B1 medium [4.5,5.5)", 0.500, 0.611),
                ("B2 high [5.5,6.0)", 0.611, 0.667),
                ("B3 visually-lossless (≥6.0)", 0.667, f64::INFINITY),
                ("Near-PJND [4.8,5.2]", 0.533, 0.578),
            ]),
            _ => None,
        };
        if let Some(bands) = bands {
            println!();
            println!("### {} per-band SROCC (vs human MOS)", ds.name);
            println!();
            println!(
                "| Band | n | V0_2 | V0_4 (bake) | V0_4 95% CI | fast-ssim2 | butter | V0_4 MAE | V0_2 MAE |"
            );
            println!("|---|--:|:--:|:--:|:--:|:--:|:--:|--:|--:|");
            for (label, lo, hi) in &bands {
                let idxs: Vec<usize> = humans
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &h)| (h >= *lo && h < *hi).then_some(i))
                    .collect();
                if idxs.len() < 4 {
                    println!(
                        "| {label} | {} | n/a | n/a | n/a | n/a | n/a | n/a |",
                        idxs.len()
                    );
                    continue;
                }
                let h_b: Vec<f64> = idxs.iter().map(|&i| humans[i]).collect();
                let v02_b: Vec<f64> = idxs.iter().map(|&i| v02[i]).collect();
                let v04_b: Vec<f64> = idxs.iter().map(|&i| v04[i]).collect();
                let ssim_b: Vec<f64> = idxs.iter().map(|&i| ssim2[i]).collect();
                let ba_b: Vec<f64> = idxs.iter().map(|&i| butter[i]).collect();
                let s_v02 = spearman(&h_b, &v02_b).abs();
                let s_v04 = spearman(&h_b, &v04_b).abs();
                let s_ssim = spearman(&h_b, &ssim_b).abs();
                let s_ba = spearman(&h_b, &ba_b).abs();
                // 200-iteration bootstrap 95% CI for V0_4 SROCC. Uses
                // xorshift64 from a fixed seed for reproducibility (no
                // rand crate dep). Small n (e.g. B3 ~ 43) needs CI
                // reported because point estimates are noisy.
                let (ci_lo, ci_hi) = bootstrap_srocc_ci_95(&h_b, &v04_b, 200, 0xC0FFEE);
                // MAE: V0_4 outputs distance ≈ 100 - score. Compare predicted
                // *score* = 100 - V0_4 distance against human MCOS * 100.
                let mae_v04: f64 = idxs
                    .iter()
                    .map(|&i| ((100.0 - v04[i]) - humans[i] * 100.0).abs())
                    .sum::<f64>()
                    / idxs.len() as f64;
                // V0_2 outputs raw distance ~ 0..90. Skip score-mapping for
                // V0_2 MAE; just report mean(|distance - (100 - MCOS)|) as a
                // rough cross-scale anchor.
                let mae_v02: f64 = idxs
                    .iter()
                    .map(|&i| (v02[i] - (100.0 - humans[i] * 100.0)).abs())
                    .sum::<f64>()
                    / idxs.len() as f64;
                println!(
                    "| {label} | {} | {s_v02:.4} | {s_v04:.4} | [{ci_lo:.2}, {ci_hi:.2}] | {s_ssim:.4} | {s_ba:.4} | {mae_v04:.2} | {mae_v02:.2} |",
                    idxs.len()
                );
            }
            println!();

            // ---- 10-band width-10 reporting (primary release gate) ----
            //
            // The 10-band grid is the PRIMARY per-band table required by
            // zensim/CLAUDE.md (revised 2026-05-14: "10 bands not 4").
            // Width-10 on the 0-100 MCOS / SSIMULACRA 2 scale. For corpora
            // whose normalized scores use different cuts (KADID, TID) the
            // band edges below map directly to width-10 zones on the
            // dataset's normalized human-score scale.
            println!(
                "### {} 10-band SROCC (PRIMARY: B0..B9 width-10 on normalized score)",
                ds.name
            );
            println!();
            println!(
                "| Band | range | n | V0_2 | V0_4 (bake) | V0_4 95% CI | fast-ssim2 | butter | V0_4 MAE |"
            );
            println!("|---|---|--:|:--:|:--:|:--:|:--:|:--:|--:|");
            for band_idx in 0..10 {
                let lo = band_idx as f64 * 0.10;
                let hi = lo + 0.10;
                let label = format!("B{band_idx}");
                let range_label = if band_idx == 9 {
                    format!("[{:.2}, 1.00]", lo)
                } else {
                    format!("[{:.2}, {:.2})", lo, hi)
                };
                let idxs: Vec<usize> = humans
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &h)| {
                        if band_idx == 9 {
                            (h >= lo).then_some(i)
                        } else {
                            (h >= lo && h < hi).then_some(i)
                        }
                    })
                    .collect();
                if idxs.len() < 4 {
                    println!(
                        "| {label} | {range_label} | {} | n/a | n/a | n/a | n/a | n/a | n/a |",
                        idxs.len()
                    );
                    continue;
                }
                let h_b: Vec<f64> = idxs.iter().map(|&i| humans[i]).collect();
                let v02_b: Vec<f64> = idxs.iter().map(|&i| v02[i]).collect();
                let v04_b: Vec<f64> = idxs.iter().map(|&i| v04[i]).collect();
                let ssim_b: Vec<f64> = idxs.iter().map(|&i| ssim2[i]).collect();
                let ba_b: Vec<f64> = idxs.iter().map(|&i| butter[i]).collect();
                let s_v02 = spearman(&h_b, &v02_b).abs();
                let s_v04 = spearman(&h_b, &v04_b).abs();
                let s_ssim = spearman(&h_b, &ssim_b).abs();
                let s_ba = spearman(&h_b, &ba_b).abs();
                let (ci_lo, ci_hi) = bootstrap_srocc_ci_95(&h_b, &v04_b, 200, 0xC0FFEE);
                let mae_v04: f64 = idxs
                    .iter()
                    .map(|&i| ((100.0 - v04[i]) - humans[i] * 100.0).abs())
                    .sum::<f64>()
                    / idxs.len() as f64;
                let noisy = if idxs.len() < 30 { " ⚠" } else { "" };
                println!(
                    "| {label}{noisy} | {range_label} | {} | {s_v02:.4} | {s_v04:.4} | [{ci_lo:.2}, {ci_hi:.2}] | {s_ssim:.4} | {s_ba:.4} | {mae_v04:.2} |",
                    idxs.len()
                );
            }
            println!();
            println!("_⚠ marks bands with n < 30 — point estimate is noisy._");
            println!();

            // ---- Fine-grained step-5 (optional supplement) ----
            // 20 bins of width 0.05. Retained because high-density corpora
            // (n >= 1000) benefit from the finer grid; primary release gate
            // is the 10-band table above.
            println!(
                "### {} step-5 per-band SROCC (20 bins of width 0.05 on normalized score)",
                ds.name
            );
            println!();
            println!(
                "| Bin (normalized) | n | V0_2 | V0_4 (bake) | V0_4 95% CI | fast-ssim2 | butter |"
            );
            println!("|---|--:|:--:|:--:|:--:|:--:|:--:|");
            for bin in 0..20 {
                let lo = bin as f64 * 0.05;
                let hi = lo + 0.05;
                let label = format!("[{:.2}, {:.2})", lo, hi);
                let idxs: Vec<usize> = humans
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &h)| (h >= lo && h < hi).then_some(i))
                    .collect();
                if idxs.len() < 4 {
                    println!("| {label} | {} | n/a | n/a | n/a | n/a | n/a |", idxs.len());
                    continue;
                }
                let h_b: Vec<f64> = idxs.iter().map(|&i| humans[i]).collect();
                let v02_b: Vec<f64> = idxs.iter().map(|&i| v02[i]).collect();
                let v04_b: Vec<f64> = idxs.iter().map(|&i| v04[i]).collect();
                let ssim_b: Vec<f64> = idxs.iter().map(|&i| ssim2[i]).collect();
                let ba_b: Vec<f64> = idxs.iter().map(|&i| butter[i]).collect();
                let s_v02 = spearman(&h_b, &v02_b).abs();
                let s_v04 = spearman(&h_b, &v04_b).abs();
                let s_ssim = spearman(&h_b, &ssim_b).abs();
                let s_ba = spearman(&h_b, &ba_b).abs();
                let (ci_lo, ci_hi) = bootstrap_srocc_ci_95(&h_b, &v04_b, 200, 0xC0FFEE);
                println!(
                    "| {label} | {} | {s_v02:.4} | {s_v04:.4} | [{ci_lo:.2}, {ci_hi:.2}] | {s_ssim:.4} | {s_ba:.4} |",
                    idxs.len()
                );
            }
            println!();
        }
    }

    // ---- KonJND-1k visually-lossless calibration ----
    //
    // KonJND-1k pairs source images with the distorted version at each
    // source's mean PJND threshold ("just barely noticeable"). For every
    // metric we report mean ± stdev across the 504 JPEG and 504 BPG
    // sources. The Cloudinary CID22 paper Table 4 publishes these for
    // SSIMULACRA 2 / Butteraugli / VMAF / etc. — comparing our numbers
    // with that table cross-validates our pipeline implementation
    // against the published reference values.
    if let Some(p) = konjnd {
        let pairs = load_konjnd(&p, usize::MAX);
        let n_total = pairs.len();
        if n_total >= 4 {
            eprintln!("=== KonJND-1k (n={n_total}) ===");
            let z_v04 = Zensim::new(zensim_profile_override.unwrap_or(ZensimProfile::PreviewV0_3));
            let started = std::time::Instant::now();
            let progress = AtomicUsize::new(0);
            let log_every = (n_total / 20).max(1);
            type CalibRow = (f64, f64, f64, f64, Vec<f64>); // v02_dist, v04_dist, ssim2, butter, features
            let results: Vec<Option<(String, String, CalibRow)>> = pairs
                .par_iter()
                .map(|kp| {
                    let p = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if p.is_multiple_of(log_every) {
                        let elapsed = started.elapsed().as_secs_f64();
                        let rate = p as f64 / elapsed;
                        let eta = (n_total - p) as f64 / rate;
                        eprintln!("  konjnd {p}/{n_total} ({rate:.1}/s, ETA {eta:.0}s)");
                    }
                    let ref_basename = kp
                        .pair
                        .reference
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    process_konjnd_pair(kp, &z_v04, v04_bake_bytes.as_deref())
                        .map(|row| (kp.codec.clone(), ref_basename, row))
                })
                .collect();
            let scored: Vec<(String, String, CalibRow)> = results.into_iter().flatten().collect();

            // Optionally emit per-pair feature CSV for trainer anchor input.
            // Format matches trainer's --human-csv expectation:
            //   ref_basename, human_score (in [0,1]), f0..f227
            // human_score is set to konjnd_anchor_target/100 so the trainer's
            // load_human_csv multiplies it back to the absolute target (e.g.
            // 0.63 -> score_zensim=63 — the CID22 paper Table 4 anchor).
            if let Some(out_path) = konjnd_features_csv.as_ref() {
                use std::io::Write;
                let f = std::fs::File::create(out_path).expect("create konjnd-features-csv");
                let mut w = std::io::BufWriter::new(f);
                let n_feat = 228usize;
                write!(w, "ref_basename,human_score").unwrap();
                for i in 0..n_feat {
                    write!(w, ",f{i}").unwrap();
                }
                writeln!(w).unwrap();
                let target_normalized = konjnd_anchor_target / 100.0;
                let mut rows_written = 0usize;
                for (_codec, ref_name, row) in &scored {
                    if row.4.len() < n_feat {
                        continue;
                    }
                    write!(w, "{ref_name},{target_normalized:.6}").unwrap();
                    for v in &row.4[..n_feat] {
                        write!(w, ",{v:.6}").unwrap();
                    }
                    writeln!(w).unwrap();
                    rows_written += 1;
                }
                w.flush().unwrap();
                eprintln!(
                    "  konjnd features CSV: {} rows -> {}",
                    rows_written,
                    out_path.display()
                );
            }
            eprintln!(
                "  konjnd done {}/{n_total} valid in {:.1}s",
                scored.len(),
                started.elapsed().as_secs_f64()
            );

            println!();
            println!("## KonJND-1k visually-lossless calibration (Lin, Hosu, Saupe 2022)");
            println!();
            println!(
                "Pairs at the per-source mean PJND (Probabilistic Just-Noticeable-Difference)\n\
                 threshold. Each source's pair is the just-barely-perceptible distortion. The\n\
                 Cloudinary CID22 paper Table 4 publishes these mean ± stdev anchors for\n\
                 several metrics; comparing our SSIMULACRA 2 / Butteraugli numbers below with\n\
                 that table cross-validates the pipeline."
            );
            for codec in ["JPEG", "BPG"] {
                let subset: Vec<&CalibRow> = scored
                    .iter()
                    .filter_map(|(c, _ref_name, r)| (c == codec).then_some(r))
                    .collect();
                let n = subset.len();
                if n == 0 {
                    continue;
                }
                let v02_d: Vec<f64> = subset.iter().map(|r| r.0).collect();
                let v04_d: Vec<f64> = subset.iter().map(|r| r.1).collect();
                let ssim2: Vec<f64> = subset.iter().map(|r| r.2).collect();
                let butter: Vec<f64> = subset.iter().map(|r| r.3).collect();
                let m_v02 = mean(&v02_d);
                let s_v02 = stddev(&v02_d, m_v02);
                let m_v04 = mean(&v04_d);
                let s_v04 = stddev(&v04_d, m_v04);
                let m_ss = mean(&ssim2);
                let s_ss = stddev(&ssim2, m_ss);
                let m_ba = mean(&butter);
                let s_ba = stddev(&butter, m_ba);
                println!();
                println!("### {codec} subset (n = {n})");
                println!();
                println!("| metric | mean | stdev | Cloudinary Table 4 (paper) |");
                println!("|---|--:|--:|---|");
                println!("| V0_2 raw distance | {m_v02:.4} | {s_v02:.4} | — |");
                println!("| V0_4 raw distance | {m_v04:.4} | {s_v04:.4} | — |");
                let (ss_ref, ba_ref) = match codec {
                    "BPG" => ("65.38 ± 5.10", "1.528 ± 0.192"),
                    "JPEG" => ("63.10 ± 4.65", "1.699 ± 0.229"),
                    _ => ("—", "—"),
                };
                println!("| fast-ssim2 score | {m_ss:.2} | {s_ss:.2} | {ss_ref} |");
                println!("| butteraugli 3-norm | {m_ba:.4} | {s_ba:.4} | {ba_ref} |");
            }
        }
    }
}

fn process_konjnd_pair(
    kp: &KonJndPair,
    z_v04: &Zensim,
    v04_bake: Option<&[u8]>,
) -> Option<(f64, f64, f64, f64, Vec<f64>)> {
    let p = &kp.pair;
    let src_img = match image::open(&p.reference) {
        Ok(img) => img.to_rgb8(),
        Err(_) => return None,
    };
    let dst_img = match image::open(&p.distorted) {
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
    let v04_distance = if let Some(bytes) = v04_bake {
        let model = Model::from_bytes(bytes).ok()?;
        let n_inputs = model.n_inputs();
        if features.len() < n_inputs {
            f64::NAN
        } else {
            let f32_features: Vec<f32> = features[..n_inputs].iter().map(|&v| v as f32).collect();
            let mut p = Predictor::new(&model);
            // V_20+ bakes carry feature_transforms metadata; use the
            // transform-aware path. zenpredict's predict_transformed
            // is a zero-cost no-op when the bake has no transforms,
            // so this is safe for V_18 (no transforms) too.
            if model.has_nontrivial_feature_transforms() {
                p.predict_transformed(&f32_features).ok()?[0] as f64
            } else {
                p.predict(&f32_features).ok()?[0] as f64
            }
        }
    } else {
        // Fall back to the live Zensim runtime's score. For
        // PreviewV0_3 this is the V_18 ship single-bake forward; for
        // PreviewV0_4 it's the D2 α=0.7 multi-bake mix. Both return
        // the calibrated MCOS 0..100 score. Convert to "distance" via
        // 100 - score so downstream uses the same convention.
        100.0 - result.score()
    };
    let s_img = Img::new(src_pixels.as_slice(), w_us, h_us);
    let d_img = Img::new(dst_pixels.as_slice(), w_us, h_us);
    let ssim2 = fast_ssim2::compute_ssimulacra2(s_img, d_img).ok()?;
    let src_rgb8: &[RGB8] = bytemuck::cast_slice(&src_pixels);
    let dst_rgb8: &[RGB8] = bytemuck::cast_slice(&dst_pixels);
    let s_b = Img::new(src_rgb8, w_us, h_us);
    let d_b = Img::new(dst_rgb8, w_us, h_us);
    let bp = ButteraugliParams::default().with_compute_diffmap(true);
    let butter = butteraugli::butteraugli(s_b, d_b, &bp).ok()?;
    let three_norm = match &butter.diffmap {
        Some(dm) => libjxl_pnorm(dm.buf(), 3.0),
        None => f64::NAN,
    };
    let features_vec: Vec<f64> = features.to_vec();
    Some((v02_distance, v04_distance, ssim2, three_norm, features_vec))
}

/// libjxl's `ComputeDistanceP` for diffmap-of-floats. Source:
/// `libjxl/lib/extras/metrics.cc`, `ComputeDistanceP` (slow path).
///
/// Despite the name "p-norm", it is the average of three p-norms at p,
/// 2p, and 4p:
///
///   pnorm_libjxl(p) = ( (Σ d^p / n)^(1/p)
///                     + (Σ d^(2p) / n)^(1/(2p))
///                     + (Σ d^(4p) / n)^(1/(4p)) ) / 3
///
/// This is what the Cloudinary CID22 paper Table 4 reports as
/// "Butteraugli 3-norm" — a single p-norm doesn't reproduce the
/// published values.
fn libjxl_pnorm(diffmap: &[f32], p: f64) -> f64 {
    if diffmap.is_empty() {
        return f64::NAN;
    }
    let mut sum1 = [0.0_f64; 3];
    for &v in diffmap {
        let d = v as f64;
        let mut acc = d.powf(p);
        sum1[0] += acc;
        acc *= acc;
        sum1[1] += acc;
        acc *= acc;
        sum1[2] += acc;
    }
    let one_per_pixels = 1.0 / diffmap.len() as f64;
    let mut v = 0.0_f64;
    for (i, &s) in sum1.iter().enumerate() {
        let exponent = 1.0 / (p * (1u32 << i) as f64);
        v += (one_per_pixels * s).powf(exponent);
    }
    v / 3.0
}

fn load_konjnd(base: &Path, max: usize) -> Vec<KonJndPair> {
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
        let image_id = record.get(0).unwrap_or("");
        let comp = record.get(1).unwrap_or("");
        let mean_threshold: f64 = match record.get(3).unwrap_or("").parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let stem = image_id.trim_end_matches(".png");
        if stem.is_empty() {
            continue;
        }
        let level = mean_threshold.round().clamp(1.0, 100.0) as u32;
        let (subdir, ext) = match comp {
            "JPEG" => ("jpeg", "jpg"),
            "BPG" => ("bpg", "png"),
            _ => continue,
        };
        let dist_name = format!("{stem}_{comp}_{level:03}.{ext}");
        let ref_path = base.join("source_image").join(image_id);
        let dist_path = base.join(subdir).join(&dist_name);
        if !dist_path.exists() {
            continue;
        }
        pairs.push(KonJndPair {
            pair: Pair {
                reference: ref_path,
                distorted: dist_path,
                human_score: mean_threshold,
                codec: None,
                version: None,
            },
            codec: comp.to_string(),
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
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

/// Feature-width regime needed by the loaded bake. Inferred once from
/// `Model::n_inputs()` so per-pair compute can pick the cheapest path
/// that produces enough columns.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FeatureRegime {
    /// 228 features — basic + peaks. Default `Zensim::compute` path.
    Standard,
    /// 300 features — adds 72 masked (`extended_features=true`).
    Extended,
    /// 372 features — adds 72 masked + 72 IW
    /// (`extended_features=true` + `compute_iw_features=true`).
    /// Wang & Li 2011 IW-SSIM pool, used by V_20a IW bakes.
    ExtendedIw,
}

impl FeatureRegime {
    fn from_n_inputs(n_inputs: usize) -> Self {
        if n_inputs <= 228 {
            FeatureRegime::Standard
        } else if n_inputs <= 300 {
            FeatureRegime::Extended
        } else {
            FeatureRegime::ExtendedIw
        }
    }
}

fn process_pair(
    pair: &Pair,
    z_v04: &Zensim,
    v04_bake: Option<&[u8]>,
    regime: FeatureRegime,
) -> Option<(f64, f64, f64, f64, f64)> {
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

    // Single compute pass — picks the feature regime needed by the
    // loaded bake. 228 = basic + peaks (cheap); 300 = adds masked
    // (extended); 372 = adds masked + IW. V_20a IW-SSIM bakes use 372.
    let result = match regime {
        FeatureRegime::Standard => z_v04.compute(&s, &d).ok()?,
        FeatureRegime::Extended => z_v04.compute_extended_features(&s, &d).ok()?,
        FeatureRegime::ExtendedIw => {
            let mut cfg = ZensimConfig::default();
            cfg.extended_features = true;
            cfg.compute_iw_features = true;
            compute_zensim_with_config(&src_pixels, &dst_pixels, w_us, h_us, cfg).ok()?
        }
    };
    let features = result.features();
    if features.len() < zensim::profile::LINEAR_WEIGHTS_PREVIEW_V0_2.len() {
        return None;
    }

    // V0_2 raw distance: dot(features, weights) / num_scales.
    let n_scored = zensim::profile::LINEAR_WEIGHTS_PREVIEW_V0_2.len();
    let v02_dot: f64 = features[..n_scored]
        .iter()
        .zip(zensim::profile::LINEAR_WEIGHTS_PREVIEW_V0_2.iter())
        .map(|(f, w)| f * w)
        .sum();
    let v02 = v02_dot / 4.0; // V0_2 has num_scales = 4

    // V0_4 trained bake: load model, run Predictor on the features.
    let v04 = if let Some(bytes) = v04_bake {
        let model = Model::from_bytes(bytes).ok()?;
        let n_inputs = model.n_inputs();
        if features.len() < n_inputs {
            f64::NAN
        } else {
            let f32_features: Vec<f32> = features[..n_inputs].iter().map(|&v| v as f32).collect();
            let mut p = Predictor::new(&model);
            if model.has_nontrivial_feature_transforms() {
                p.predict_transformed(&f32_features).ok()?[0] as f64
            } else {
                p.predict(&f32_features).ok()?[0] as f64
            }
        }
    } else {
        // No external bake: emit the live Zensim runtime's score
        // converted to distance-convention (100 - score). For
        // PreviewV0_3 this is V_18 ship; for PreviewV0_4 it's the
        // D2 α=0.7 multi-bake mix. Pre-existing eval callers passed
        // --v04-bake explicitly so we never hit this branch; new
        // callers can omit --v04-bake to test the live runtime.
        100.0 - result.score()
    };

    // fast-ssim2 — score, higher = more similar.
    let s_img = Img::new(src_pixels.as_slice(), w_us, h_us);
    let d_img = Img::new(dst_pixels.as_slice(), w_us, h_us);
    let ssim2 = fast_ssim2::compute_ssimulacra2(s_img, d_img).ok()?;

    // butteraugli — score, higher = more different.
    let src_rgb8: &[RGB8] = bytemuck::cast_slice(&src_pixels);
    let dst_rgb8: &[RGB8] = bytemuck::cast_slice(&dst_pixels);
    let s_b = Img::new(src_rgb8, w_us, h_us);
    let d_b = Img::new(dst_rgb8, w_us, h_us);
    let butter = butteraugli::butteraugli(s_b, d_b, &ButteraugliParams::default()).ok()?;

    Some((pair.human_score, v02, v04, ssim2, butter.score))
}

// ---- dataset loaders (minimal — just enough for SROCC) ----

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
        pairs.push(Pair {
            reference: base.join("images").join(r),
            distorted: base.join("images").join(dist),
            human_score: (dmos - 1.0) / 4.0,
            codec: None,
            version: None,
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
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
        // TID2013 file names: `iXX_YY_Z.bmp` → reference `iXX.BMP`.
        let prefix = name.split('_').next().unwrap_or("");
        let ref_name = format!("{}.BMP", prefix);
        let ref_path = base.join("reference_images").join(&ref_name);
        let dist_path = base.join("distorted_images").join(name);
        // Some installations have lowercase reference names.
        let ref_path = if ref_path.exists() {
            ref_path
        } else {
            base.join("reference_images").join(format!("{prefix}.bmp"))
        };
        pairs.push(Pair {
            reference: ref_path,
            distorted: dist_path,
            human_score: mos / 9.0,
            codec: None,
            version: None,
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
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
        let mcos: f64 = match record.get(5).unwrap().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        pairs.push(Pair {
            reference: base.join(r),
            distorted: base.join(dist),
            human_score: mcos / 100.0,
            codec: Some(encoder.to_string()),
            version: Some(record.get(3).unwrap_or("").to_string()),
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// CSIQ compression subset loader. Reads pre-extracted CSV
/// `csiq_compression_pairs.csv` (jpeg + jpeg2000 only, 145 pairs).
/// CSV format: reference,distorted,distortion_type,distortion_level,dmos.
/// DMOS is in [0, ~0.5]; mapped to score-equivalent via
/// `human_score = 1 - dmos` so identical = 1.0 and worst ≈ 0.5.
///
/// Caveat: CSIQ JPEG/JPEG2000 are older codec versions; SROCC is still
/// rank-meaningful but band cuts at [50, 65, 90] are heuristic
/// (Table 5 alignment is for CID22 MCOS specifically).
fn load_csiq(base: &Path, max: usize) -> Vec<Pair> {
    let csv_path = base.join("csiq_compression_pairs.csv");
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
        let r = record.get(0).unwrap();
        let dist = record.get(1).unwrap();
        let dmos: f64 = match record.get(4).unwrap().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        pairs.push(Pair {
            reference: base.join(r),
            distorted: base.join(dist),
            human_score: 1.0 - dmos,
            codec: None,
            version: None,
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// AIC-3 CTC (EPFL) loader. Reads a paired CSV produced by
/// `scripts/v_next/aic3_pairs_csv.py` with columns:
/// `ref_path, dist_path, codec, quality_idx, quality_selected, score_jnd, method`.
///
/// `score_jnd` is a human-judged JND (just-noticeable-difference) score
/// where 0 = identical and more negative = more degraded. Empirically
/// the dataset spans ~[-3, 0]. We map to `human_score = (score_jnd + 3) / 3`
/// so that 1.0 = identical and 0.0 = worst, matching the "higher = better"
/// convention used by KADID/TID/CID22 loaders.
///
/// SROCC vs zensim/ssim2 distance should be NEGATIVE under this
/// convention (since distance grows as human_score shrinks).
///
/// Paths in the CSV are absolute; the `base` argument is the CSV path
/// itself (not a directory root).
/// Generic loader for `--pairs-tsv NAME:PATH`.
///
/// TSV header is required. Columns: `ref_path`, `dist_path`, `codec`,
/// `(version|image_name|quality|dlevel|...)`, ..., last col is human score.
/// The header row identifies columns by NAME — we look for `ref_path`,
/// `dist_path`, `codec`, `version` / `quality` (whichever exists),
/// and the score column (`score_jnd` / `human_jnd` / `human_score` / last col).
fn load_pairs_tsv(path: &Path, max: usize) -> Vec<Pair> {
    let mut rdr = match csv::ReaderBuilder::new().delimiter(b'\t').from_path(path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", path.display());
            return Vec::new();
        }
    };
    let headers = match rdr.headers() {
        Ok(h) => h.clone(),
        Err(e) => {
            eprintln!("--pairs-tsv {}: header read failed: {e}", path.display());
            return Vec::new();
        }
    };
    let find = |names: &[&str]| -> Option<usize> {
        for (i, h) in headers.iter().enumerate() {
            if names.contains(&h) {
                return Some(i);
            }
        }
        None
    };
    let i_ref = find(&["ref_path", "reference", "ref"]);
    let i_dist = find(&["dist_path", "distorted", "dist"]);
    let i_codec = find(&["codec", "encoder"]);
    let i_version = find(&["version", "setting", "quality", "dlevel", "quality_index"]);
    let i_score = find(&[
        "score_jnd",
        "human_jnd",
        "human_score",
        "human_mos",
        "human_dmos",
        "mcos",
        "dmos",
        "mos",
    ]);
    if i_ref.is_none() || i_dist.is_none() || i_score.is_none() {
        eprintln!(
            "--pairs-tsv {}: missing required column(s) ref_path/dist_path/<score>; header was {:?}",
            path.display(),
            headers
        );
        return Vec::new();
    }
    let i_ref = i_ref.unwrap();
    let i_dist = i_dist.unwrap();
    let i_score = i_score.unwrap();
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        let r = match record.get(i_ref) {
            Some(s) => s,
            None => continue,
        };
        let d = match record.get(i_dist) {
            Some(s) => s,
            None => continue,
        };
        let s = match record.get(i_score).and_then(|s| s.parse::<f64>().ok()) {
            Some(v) => v,
            None => continue,
        };
        pairs.push(Pair {
            reference: PathBuf::from(r),
            distorted: PathBuf::from(d),
            human_score: s,
            codec: i_codec.and_then(|i| record.get(i)).map(|s| s.to_string()),
            version: i_version.and_then(|i| record.get(i)).map(|s| s.to_string()),
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

fn load_aic3(csv_path: &Path, max: usize) -> Vec<Pair> {
    let mut rdr = match csv::Reader::from_path(csv_path) {
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
        let score_jnd: f64 = match record.get(5).unwrap().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        pairs.push(Pair {
            reference: PathBuf::from(r),
            distorted: PathBuf::from(dist),
            human_score: (score_jnd + 3.0) / 3.0,
            codec: None,
            version: None,
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

// ---- Statistical helpers (CLAUDE.md mandatory full-stat panel) ----
//
// Per zensim/CLAUDE.md "Statistical rigor" (2026-05-14): every eval
// that emits SROCC MUST also emit PLCC, KROCC, OR, PWRC, Z-RMSE, MRR
// p-value, and Wilcoxon signed-rank + effect size. Full mandate
// shipped here.
//
// Reference impls: Mohammadi et al. 2025 "Evaluation of Objective IQA
// Metrics for HF Image Compression" (arXiv:2509.13150) reference
// notebooks at github.com/shimamohammadi/EvaluationMetrics.

/// Z-RMSE: σ-normalized RMSE between metric predictions and subjective
/// scores. Penalizes errors LESS where humans disagreed (large σ) and
/// MORE where the JND is sharp (small σ). From Mohammadi 2025 eq. 5:
///
/// ```text
/// Z-RMSE = √( (1/n) · Σ ((Ŝ_i − μ_i) / σ_i)² )
/// ```
///
/// where μ_i and σ_i are the per-stimulus subjective mean and σ
/// (from bootstrap of human ratings). When per-stimulus σ is not
/// available, falls back to corpus-wide σ (less informative but still
/// captures the magnitude of metric error).
///
/// Predictions must be on the same scale as subjective (apply affine
/// rescale via least-squares OR Pearson-based fit before computing
/// Z-RMSE; this fn assumes already-rescaled inputs).
pub fn z_rmse(predicted: &[f64], target: &[f64], target_sigma: Option<&[f64]>) -> f64 {
    let n = predicted.len();
    if n < 2 || target.len() != n {
        return f64::NAN;
    }
    let sigma_global = {
        let mean: f64 = target.iter().sum::<f64>() / n as f64;
        let var: f64 = target.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        var.sqrt().max(1e-9)
    };
    let mut sum_sq = 0.0f64;
    let mut count = 0;
    for i in 0..n {
        let sig = match target_sigma {
            Some(s) if i < s.len() && s[i].is_finite() && s[i] > 0.0 => s[i],
            _ => sigma_global,
        };
        let z = (predicted[i] - target[i]) / sig;
        if z.is_finite() {
            sum_sq += z * z;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    (sum_sq / count as f64).sqrt()
}

/// Rescale `predicted` to `target`'s mean/scale via least-squares
/// affine fit `Ŝ = a + b · pred` (the only step where polarity matters).
/// Returns the rescaled predictions. Use BEFORE z_rmse so the units
/// match.
///
/// NOTE: this is the LEGACY rescaler kept for compatibility and for
/// metrics that are already linear in the MOS scale. The
/// paper-aligned (Mohammadi 2025) choice is `rescale_logistic` — use
/// that one by default before computing Z-RMSE.
pub fn rescale_to_match(predicted: &[f64], target: &[f64]) -> Vec<f64> {
    let n = predicted.len().min(target.len());
    if n < 2 {
        return predicted.to_vec();
    }
    let mean_p: f64 = predicted.iter().take(n).sum::<f64>() / n as f64;
    let mean_t: f64 = target.iter().take(n).sum::<f64>() / n as f64;
    let mut cov = 0.0f64;
    let mut var_p = 0.0f64;
    for i in 0..n {
        let dp = predicted[i] - mean_p;
        let dt = target[i] - mean_t;
        cov += dp * dt;
        var_p += dp * dp;
    }
    let b = if var_p.abs() < 1e-12 {
        0.0
    } else {
        cov / var_p
    };
    let a = mean_t - b * mean_p;
    predicted.iter().map(|p| a + b * p).collect()
}

/// 4-parameter logistic rescale, the Mohammadi 2025 convention for
/// before-Z-RMSE metric→MOS rescaling. Fits
///
/// ```text
/// logistic(x; b1, b2, b3, b4) = b2 + (b1 − b2) / (1 + exp(−(x − b3) / b4))
/// ```
///
/// minimizing `Σ (logistic(pred_i) − target_i)²` via Levenberg-
/// Marquardt (Marquardt 1963; equivalent to scipy.optimize.curve_fit
/// with `method='lm'` modulo numeric tolerances).
///
/// **Why this and not affine?** Non-linear metrics (PSNR-Y, Butteraugli,
/// distance-based metrics in general) have an S-shape on the MOS scale:
/// they saturate at high quality and have a long low-q tail. Linear
/// rescale puts the entire saturation region into the residual,
/// blowing up Z-RMSE by 30× on nonlinear metrics. The 4-parameter
/// logistic absorbs the saturation into the rescale so Z-RMSE measures
/// *prediction error after the metric's natural shape*, not the shape
/// itself. Affine residuals dominate on PSNR-Y (Z-RMSE 486 vs paper
/// 13.36); logistic recovers 13.36 ±0.5 in our tests against
/// `Anchor_assessment_on_PTC_full_resolution_Aug_3_2025.csv` from the
/// AIC-3 corpus.
///
/// Initial guesses (matching Mohammadi's `my_fit.py`):
///   b1 = max(target),  b2 = min(target),
///   b3 = mean(predicted),  b4 = max(std(predicted), 1e-3).
///
/// `b4 < 0` is permitted — distance metrics (lower = better) flip
/// naturally to a decreasing fit. We do NOT constrain its sign.
///
/// Falls back to affine `rescale_to_match` and logs a stderr warning
/// when the LM fit fails to converge, the input is degenerate (all
/// identical predictions, n < 4), or the Hessian is singular.
pub fn rescale_logistic(predicted: &[f64], target: &[f64]) -> Vec<f64> {
    let n = predicted.len().min(target.len());
    if n < 4 {
        return rescale_to_match(predicted, target);
    }

    // Degenerate guards: if predicted has zero variance, no nonlinear fit is
    // possible; affine = constant function.
    let mean_p: f64 = predicted.iter().take(n).sum::<f64>() / n as f64;
    let var_p: f64 = predicted
        .iter()
        .take(n)
        .map(|x| (x - mean_p).powi(2))
        .sum::<f64>()
        / n as f64;
    if !var_p.is_finite() || var_p < 1e-18 {
        return rescale_to_match(predicted, target);
    }
    if !predicted.iter().take(n).all(|x| x.is_finite())
        || !target.iter().take(n).all(|x| x.is_finite())
    {
        return rescale_to_match(predicted, target);
    }
    // Initial parameter guesses, identical to Mohammadi `my_fit.py` p0.
    let t_max = target
        .iter()
        .take(n)
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let t_min = target.iter().take(n).cloned().fold(f64::INFINITY, f64::min);
    let p_std = var_p.sqrt();
    // Also detect predicted vs target correlation to seed an
    // "anti-correlated" start for distance metrics.
    let p_corr = {
        let mean_t = target.iter().take(n).sum::<f64>() / n as f64;
        let mut cov = 0.0f64;
        let mut vp = 0.0f64;
        let mut vt = 0.0f64;
        for i in 0..n {
            let dp = predicted[i] - mean_p;
            let dt = target[i] - mean_t;
            cov += dp * dt;
            vp += dp * dp;
            vt += dt * dt;
        }
        let d = (vp * vt).sqrt();
        if d < 1e-12 { 0.0 } else { cov / d }
    };
    let b4_sign = if p_corr < 0.0 { -1.0 } else { 1.0 };

    // Mohammadi `my_fit.py` baseline init, plus alternative starts
    // for ill-conditioned narrow-range cases. With narrow input
    // ranges (e.g. IW-SSIM in [0.97, 0.99]) the 4-param logistic has
    // flat ridges in parameter space — scipy's LM converges to
    // "tail-anchor" minima where b1 or b2 are ±thousands and the
    // logistic operates in its near-linear tail. The starts below
    // cover both conventional (b1=t_max, b2=t_min) and tail-anchor
    // regimes so the multi-start matches scipy's effective behavior
    // within ±0.05 Z-RMSE on narrow-range metrics.
    //
    // Sign convention: b1/b2 are upper/lower asymptotes. For
    // distance-style decreasing metrics (b4 < 0), they swap roles.
    // The "extreme tail" starts include large opposite-sign variants
    // that cover the regime where the logistic degenerates to a
    // near-linear function in the data range.
    // For narrow-range metrics (IW-SSIM in [0.97, 0.99]) scipy converges
    // to "near-linear-tail" minima where b3 sits ~25σ OUTSIDE the data
    // range and b1 (or b2) is large enough that the logistic operates
    // entirely on one of its near-linear tails over the data span.
    // Reference IW-SSIM scipy fit: b3=1.16 (26·p_std above data center),
    // b1=-7079, b4=0.022. We seed several b3-outside-data starts so the
    // multi-start covers that regime.
    let p_max = predicted
        .iter()
        .take(n)
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let p_min = predicted
        .iter()
        .take(n)
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let t_span = (t_max - t_min).abs().max(1.0);
    let tail = 1000.0 * t_span; // extreme tail magnitude
    let b3_high = p_max + 25.0 * p_std; // ~scipy's IW-SSIM b3=1.16
    let b3_low = p_min - 25.0 * p_std;
    let starts: [[f64; 4]; 13] = [
        // Conventional starts (Mohammadi `my_fit.py` baseline + b4 sweep)
        [
            t_max,
            t_min,
            mean_p,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p,
            (p_std * 0.1 * b4_sign).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p,
            (p_std * 10.0 * b4_sign).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p + p_std,
            (p_std * b4_sign).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p - p_std,
            (p_std * b4_sign).copysign(b4_sign),
        ],
        // Tail-anchor starts: b1 or b2 at ±tail with b3 at data center.
        [
            -tail,
            t_max,
            mean_p,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_max,
            -tail,
            mean_p,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            tail,
            t_min,
            mean_p,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_min,
            tail,
            mean_p,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        // b3-outside-data starts (the regime scipy converges to for
        // narrow-range metrics). Pair with extreme-tail asymptotes and
        // small b4 so the logistic crosses through the data range as a
        // near-linear function with slope (b2-b1)/b4.
        [
            -tail,
            t_max,
            b3_high,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_max,
            -tail,
            b3_low,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            tail,
            t_min,
            b3_low,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_min,
            tail,
            b3_high,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
    ];
    let mut best_b: Option<[f64; 4]> = None;
    let mut best_cost = f64::INFINITY;
    for start in &starts {
        if let Some((b_fit, cost_fit)) = run_lm(predicted, target, n, *start) {
            if cost_fit < best_cost {
                best_cost = cost_fit;
                best_b = Some(b_fit);
            }
        }
    }
    let b: [f64; 4] = match best_b {
        Some(b) => b,
        None => return rescale_to_match(predicted, target),
    };

    let any_bad = predicted
        .iter()
        .take(n)
        .any(|&x| !logistic_eval(&b, x).is_finite());
    if any_bad {
        eprintln!(
            "warning: rescale_logistic — fitted model produces non-finite output (n={}), falling back to affine",
            n
        );
        return rescale_to_match(predicted, target);
    }
    predicted.iter().map(|&x| logistic_eval(&b, x)).collect()
}

/// Evaluate the 4-parameter logistic at `x` with parameters
/// `[b1, b2, b3, b4]` = `[max, min, center, slope_inverse]`. Includes
/// the same numerical guards as the Jacobian builder so model and
/// jacobian stay consistent.
fn logistic_eval(b: &[f64; 4], x: f64) -> f64 {
    let b4 = if b[3].abs() < 1e-12 {
        1e-12_f64.copysign(b[3].max(0.0).signum().max(1.0))
    } else {
        b[3]
    };
    let arg = -(x - b[2]) / b4;
    let e = if arg > 700.0 {
        f64::INFINITY
    } else if arg < -700.0 {
        0.0
    } else {
        arg.exp()
    };
    b[1] + (b[0] - b[1]) / (1.0 + e)
}

/// Single Levenberg-Marquardt run for the 4-parameter logistic fit.
///
/// Termination: parameter delta L∞ < 1e-10 relative OR cost relative
/// decrease < 1e-12 OR 500 iterations. The cost-decrease check
/// catches ill-conditioned cases where parameters wander on a flat
/// ridge but the fit is essentially stable. Returns `None` on
/// non-finite initial cost or persistent Hessian singularity.
///
/// Partial derivatives (computed analytically):
/// ```text
/// let E = exp(-(x-b3)/b4),  A = 1 + E
/// logistic = b2 + (b1 - b2) / A
/// d/db1 = 1/A
/// d/db2 = 1 - 1/A
/// d/db3 = -(b1-b2) · E / (b4 · A²)
/// d/db4 = -(b1-b2) · E · (x-b3) / (b4² · A²)
/// ```
fn run_lm(predicted: &[f64], target: &[f64], n: usize, b0: [f64; 4]) -> Option<([f64; 4], f64)> {
    let max_iters = 500usize;
    let tol = 1e-10f64;
    let cost_tol = 1e-12f64;
    let mut lambda = 1.0e-3f64;
    let mut b = b0;

    let jacobian_and_residuals = |b: &[f64; 4]| -> (Vec<[f64; 4]>, Vec<f64>) {
        let mut jac = Vec::with_capacity(n);
        let mut res = Vec::with_capacity(n);
        let b4 = if b[3].abs() < 1e-12 {
            1e-12_f64.copysign(b[3].max(0.0).signum().max(1.0))
        } else {
            b[3]
        };
        for i in 0..n {
            let x = predicted[i];
            let arg = -(x - b[2]) / b4;
            let e = if arg > 700.0 {
                f64::INFINITY
            } else if arg < -700.0 {
                0.0
            } else {
                arg.exp()
            };
            let a = 1.0 + e;
            let inv_a = 1.0 / a;
            let pred = b[1] + (b[0] - b[1]) * inv_a;
            let diff = pred - target[i];
            res.push(diff);
            let db1 = inv_a;
            let db2 = 1.0 - inv_a;
            // ∂(1/A)/∂b3 = -E / (b4·A²); ∂(1/A)/∂b4 = -E·(x-b3) / (b4²·A²)
            // → ∂logistic/∂b3 = -(b1-b2)·E/(b4·A²)
            //   ∂logistic/∂b4 = -(b1-b2)·E·(x-b3) / (b4²·A²)
            // Partials vanish when e overflows (logistic saturated).
            let (db3, db4_) = if e.is_finite() && a.is_finite() && a > 1e-300 {
                let inv_a2 = inv_a * inv_a;
                let amp = b[0] - b[1];
                (
                    -amp * e * inv_a2 / b4,
                    -amp * e * (x - b[2]) * inv_a2 / (b4 * b4),
                )
            } else {
                (0.0, 0.0)
            };
            jac.push([db1, db2, db3, db4_]);
        }
        (jac, res)
    };

    let sum_sq = |res: &[f64]| -> f64 { res.iter().map(|r| r * r).sum::<f64>() };

    let (mut jac, mut res) = jacobian_and_residuals(&b);
    let mut cost = sum_sq(&res);
    if !cost.is_finite() {
        return None;
    }

    for _iter in 0..max_iters {
        // Build J^T J (4x4 symmetric) and J^T r (length 4).
        let mut jtj = [[0.0f64; 4]; 4];
        let mut jtr = [0.0f64; 4];
        for i in 0..n {
            let row = &jac[i];
            let r = res[i];
            for a_ in 0..4 {
                jtr[a_] += row[a_] * r;
                for c_ in 0..4 {
                    jtj[a_][c_] += row[a_] * row[c_];
                }
            }
        }
        // Marquardt damping: add λ · diag(JᵀJ) to diagonal.
        let mut h = jtj;
        for d in 0..4 {
            h[d][d] += lambda * jtj[d][d].max(1e-12);
        }
        // Solve h · δ = -J^T r via gaussian elimination.
        let mut aug = [[0.0f64; 5]; 4];
        for r_ in 0..4 {
            for c in 0..4 {
                aug[r_][c] = h[r_][c];
            }
            aug[r_][4] = -jtr[r_];
        }
        let solved = solve_4x4_gauss(&mut aug);
        let delta = match solved {
            Some(d) => d,
            None => {
                lambda *= 10.0;
                if lambda > 1e10 {
                    return Some((b, cost));
                }
                continue;
            }
        };
        let b_try = [
            b[0] + delta[0],
            b[1] + delta[1],
            b[2] + delta[2],
            b[3] + delta[3],
        ];
        let (jac_try, res_try) = jacobian_and_residuals(&b_try);
        let cost_try = sum_sq(&res_try);
        if cost_try.is_finite() && cost_try < cost {
            let max_delta = delta.iter().map(|d| d.abs()).fold(0.0f64, f64::max);
            let max_b = b.iter().map(|x| x.abs()).fold(1.0f64, f64::max);
            let cost_decrease_rel = (cost - cost_try) / cost.max(1e-30);
            b = b_try;
            jac = jac_try;
            res = res_try;
            cost = cost_try;
            lambda = (lambda / 10.0).max(1e-12);
            if max_delta < tol * (1.0 + max_b) || cost_decrease_rel < cost_tol {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e10 {
                break;
            }
        }
    }
    Some((b, cost))
}

/// Solve a 4-variable linear system Ax = b given the augmented matrix
/// `[A | b]` of shape 4x5 (in-place). Returns Some(x) on success,
/// None if A is singular (zero pivot after partial pivoting). Used by
/// the Levenberg-Marquardt step in `rescale_logistic`; small enough to
/// hand-roll without pulling in nalgebra.
fn solve_4x4_gauss(aug: &mut [[f64; 5]; 4]) -> Option<[f64; 4]> {
    // Forward elimination with partial pivoting.
    for i in 0..4 {
        // Find pivot row (largest |aug[k][i]| for k in i..4).
        let mut max_row = i;
        let mut max_val = aug[i][i].abs();
        for k in (i + 1)..4 {
            let v = aug[k][i].abs();
            if v > max_val {
                max_val = v;
                max_row = k;
            }
        }
        if max_val < 1e-14 {
            return None; // Singular.
        }
        if max_row != i {
            aug.swap(i, max_row);
        }
        // Eliminate below.
        for k in (i + 1)..4 {
            let factor = aug[k][i] / aug[i][i];
            for c in i..5 {
                aug[k][c] -= factor * aug[i][c];
            }
        }
    }
    // Back-substitution.
    let mut x = [0.0f64; 4];
    for i in (0..4).rev() {
        let mut sum = aug[i][4];
        for c in (i + 1)..4 {
            sum -= aug[i][c] * x[c];
        }
        x[i] = sum / aug[i][i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// Meng-Rosenthal-Rubin paired SROCC test (1992). Tests H0: r1 = r2
/// where r1 = corr(metric_a, target) and r2 = corr(metric_b, target),
/// accounting for the correlation r12 = corr(metric_a, metric_b).
/// Returns (z_statistic, p_value, effect_direction).
///
/// p_value is two-tailed (Pr(|Z| > observed)). effect_direction is
/// +1 if r1 > r2, -1 if r2 > r1, 0 if tied.
///
/// Implements the Fisher-z form (eq. 11 in Mohammadi 2025):
/// ```text
/// f = (1 − r_bar²) ⁻¹ · ⟨r1², r2², r12 × (2 r_bar² − r12)⟩ correction
/// z = √(n − 3) · (atanh(r1) − atanh(r2)) / √(2 (1 − r12) · f)
/// ```
///
/// where r_bar = (r1 + r2) / 2.
///
/// This is the standard form used in IQA-metric evaluation
/// (Mohammadi 2025, also in Sneyers 2023 CID22 paper).
pub fn mrr_test(r1: f64, r2: f64, r12: f64, n: usize) -> (f64, f64, i8) {
    if n < 4 {
        return (f64::NAN, f64::NAN, 0);
    }
    let z1 = r1.atanh();
    let z2 = r2.atanh();
    let r_bar = (r1 + r2) / 2.0;
    // Steiger / Meng-Rosenthal-Rubin variance correction
    let denom1 = 1.0 - r_bar * r_bar;
    if denom1.abs() < 1e-12 {
        return (f64::NAN, f64::NAN, 0);
    }
    let f = (1.0 - r12) / (2.0 * denom1);
    let h = (1.0 - f * r_bar * r_bar) / (1.0 - r_bar * r_bar);
    let var_z_diff = 2.0 * (1.0 - r12) * h / (n as f64 - 3.0);
    if var_z_diff <= 0.0 {
        return (f64::NAN, f64::NAN, 0);
    }
    let z_stat = (z1 - z2) / var_z_diff.sqrt();
    // Two-tailed normal p-value: erfc(|z| / √2)
    let p = libm_erfc_half(z_stat.abs());
    let dir: i8 = if r1 > r2 {
        1
    } else if r2 > r1 {
        -1
    } else {
        0
    };
    (z_stat, p, dir)
}

/// Two-tailed p-value of a standard normal: `erfc(|z| / √2)`.
/// Inline erfc approximation (no libm dependency).
fn libm_erfc_half(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    // Abramowitz & Stegun 7.1.26 erfc approximation (max relative
    // error ~3e-7 for x ≥ 0).
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    poly * (-x * x).exp()
}

/// Wilcoxon signed-rank test on absolute residuals. Tests H0: the
/// distributions of |e_a| and |e_b| are equal, where
/// e_x = z(metric_x) − z(target). Returns (z_statistic, p_value,
/// effect_size r = Z / √N).
///
/// Sign convention: positive z_statistic ⇒ metric_a has LARGER errors
/// than metric_b on average (i.e. metric_b is better).
///
/// Non-parametric companion to MRR: doesn't assume normality of
/// Fisher-z transformed correlations. Mohammadi 2025 uses both as
/// confirmatory tests.
pub fn wilcoxon_signed_rank(metric_a: &[f64], metric_b: &[f64], target: &[f64]) -> (f64, f64, f64) {
    let n = metric_a.len().min(metric_b.len()).min(target.len());
    if n < 6 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    // z-normalize each input
    let z = |xs: &[f64]| -> Vec<f64> {
        let m: f64 = xs.iter().sum::<f64>() / n as f64;
        let v: f64 = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
        let sd = v.sqrt().max(1e-12);
        xs.iter().map(|x| (x - m) / sd).collect()
    };
    let za = z(metric_a);
    let zb = z(metric_b);
    let zt = z(target);

    let mut diffs: Vec<f64> = (0..n)
        .map(|i| {
            let ea = (za[i] - zt[i]).abs();
            let eb = (zb[i] - zt[i]).abs();
            ea - eb
        })
        .filter(|d| d.abs() > 1e-12)
        .collect();
    let n_nonzero = diffs.len();
    if n_nonzero < 6 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    // Sort by |diff|, assign ranks (with average-rank ties), apply
    // sign of diff.
    let mut idx: Vec<usize> = (0..n_nonzero).collect();
    idx.sort_by(|&a, &b| {
        diffs[a]
            .abs()
            .partial_cmp(&diffs[b].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0f64; n_nonzero];
    let mut i = 0;
    while i < n_nonzero {
        let mut j = i + 1;
        while j < n_nonzero && (diffs[idx[j]].abs() - diffs[idx[i]].abs()).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    let mut w_plus = 0.0f64;
    let mut w_minus = 0.0f64;
    for k in 0..n_nonzero {
        if diffs[k] > 0.0 {
            w_plus += ranks[k];
        } else {
            w_minus += ranks[k];
        }
    }
    let n_f = n_nonzero as f64;
    let mean_w = n_f * (n_f + 1.0) / 4.0;
    let var_w = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0;
    // Use the smaller of w_plus / w_minus for the z statistic
    let w = w_plus.min(w_minus);
    let z_stat = (w - mean_w) / var_w.sqrt();
    let p = libm_erfc_half(z_stat.abs());
    // Effect size r = Z / √N (Rosenthal 1991 convention; |r| ∈ [0, 1])
    let r = z_stat.abs() / n_f.sqrt();
    // Sign convention: positive ⇒ a has larger errors than b ⇒ b is better.
    let signed_z = if w_plus > w_minus {
        z_stat.abs()
    } else {
        -z_stat.abs()
    };
    (signed_z, p, r)
}

/// Pearson product-moment correlation. The dial-honesty stat — measures
/// linearity between predictor and target, not just rank order. Critical
/// for zensim because users type a target score and expect a linear
/// response.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mean_a: f64 = a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b.iter().sum::<f64>() / n as f64;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = a[i] - mean_a;
        let xb = b[i] - mean_b;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

/// Kendall's τ-b (with ties correction). O(n²) — fine for n up to a
/// few thousand; the eval datasets fit. Returns the absolute value
/// for consistency with the other metric stats (metric polarities
/// differ).
fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    let mut ties_a = 0i64;
    let mut ties_b = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let da = a[i] - a[j];
            let db = b[i] - b[j];
            if da.abs() < 1e-12 && db.abs() < 1e-12 {
                // tied on both — ignore
                continue;
            } else if da.abs() < 1e-12 {
                ties_a += 1;
            } else if db.abs() < 1e-12 {
                ties_b += 1;
            } else if (da * db) > 0.0 {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let total_a = (concordant + discordant + ties_a) as f64;
    let total_b = (concordant + discordant + ties_b) as f64;
    let den = (total_a * total_b).sqrt();
    if den < 1e-12 {
        0.0
    } else {
        ((concordant - discordant) as f64) / den
    }
}

/// Outlier Ratio. Following the VQEG / IQA convention, an "outlier" is
/// a prediction whose residual exceeds 2·σ where σ is estimated per
/// stimulus from the subjective spread. We don't have per-stimulus
/// subjective σ (it requires bootstrapping the human MOS), so we use
/// the simpler global-σ z-score variant: fraction of |z(residual)| > 2.
///
/// The signal is "what fraction of stimuli does the metric grossly
/// misrank?" — high OR means a metric mostly tracks the MOS but blows
/// up on a few stimuli, which is exactly the pathology a user-facing
/// dial cares about.
fn outlier_ratio(predicted: &[f64], target: &[f64]) -> f64 {
    let n = predicted.len();
    if n < 4 {
        return f64::NAN;
    }
    // First, rescale predicted into target's range via z-score so the
    // metric polarity doesn't matter (zensim is distance, ssim2 is
    // score). We compare z-scores, not absolute residuals.
    let mean_p: f64 = predicted.iter().sum::<f64>() / n as f64;
    let mean_t: f64 = target.iter().sum::<f64>() / n as f64;
    let var_p: f64 = predicted.iter().map(|x| (x - mean_p).powi(2)).sum::<f64>() / n as f64;
    let var_t: f64 = target.iter().map(|x| (x - mean_t).powi(2)).sum::<f64>() / n as f64;
    let sd_p = var_p.sqrt().max(1e-12);
    let sd_t = var_t.sqrt().max(1e-12);
    let mut residuals: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        // z-score residual: account for metric flipping by detecting
        // direction via the sign of correlation.
        let zp = (predicted[i] - mean_p) / sd_p;
        let zt = (target[i] - mean_t) / sd_t;
        residuals.push((zp - zt).abs());
    }
    // Use the residuals' own σ for the 2σ outlier cutoff. This is the
    // standard "robust" form when no per-stimulus σ is available.
    let mean_r: f64 = residuals.iter().sum::<f64>() / n as f64;
    let sd_r: f64 = (residuals.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / n as f64)
        .sqrt()
        .max(1e-12);
    let mut polarity = 1.0;
    // Detect polarity-flip: if predictor is anti-correlated with
    // target, the z-scores point opposite — the residual becomes
    // a sum, not a difference. Re-compute residuals with the flip.
    let s = pearson(predicted, target);
    if s < 0.0 {
        polarity = -1.0;
    }
    if polarity < 0.0 {
        let mut residuals: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let zp = -(predicted[i] - mean_p) / sd_p;
            let zt = (target[i] - mean_t) / sd_t;
            residuals.push((zp - zt).abs());
        }
        let mean_r: f64 = residuals.iter().sum::<f64>() / n as f64;
        let sd_r: f64 = (residuals.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / n as f64)
            .sqrt()
            .max(1e-12);
        return residuals
            .iter()
            .filter(|r| (**r - mean_r).abs() > 2.0 * sd_r)
            .count() as f64
            / n as f64;
    }
    residuals
        .iter()
        .filter(|r| (**r - mean_r).abs() > 2.0 * sd_r)
        .count() as f64
        / n as f64
}

/// Pearson Weighted Rank Correlation (PWRC). IQA-literature stat that
/// weights rank-Pearson by extremeness — emphasises correlation at the
/// tails of the rank distribution where compression-product decisions
/// live (extreme low quality or extreme high quality).
///
/// Definition used here: weighted Pearson on rank-transformed values
/// with `w_i = |R(x_i) − (n+1)/2| / ((n+1)/2)` so the median rank gets
/// weight 0 and the extremes get weight 1. Other definitions exist in
/// the literature (PWRC of Wang & Liu, weighted Kendall variants);
/// document the exact form in our methodology so we're comparable
/// across runs.
fn pwrc(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 4 {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let mid = (n as f64 - 1.0) / 2.0;
    let max_dev = mid.max(1e-12);
    let w: Vec<f64> = ra.iter().map(|r| (r - mid).abs() / max_dev).collect();
    let wsum: f64 = w.iter().sum();
    if wsum < 1e-12 {
        return 0.0;
    }
    let mean_a: f64 = w.iter().zip(&ra).map(|(w, r)| w * r).sum::<f64>() / wsum;
    let mean_b: f64 = w.iter().zip(&rb).map(|(w, r)| w * r).sum::<f64>() / wsum;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = ra[i] - mean_a;
        let xb = rb[i] - mean_b;
        num += w[i] * xa * xb;
        da += w[i] * xa * xa;
        db += w[i] * xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

// ---- SROCC ----

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let mean_a = (n as f64 - 1.0) / 2.0;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = ra[i] - mean_a;
        let xb = rb[i] - mean_a;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    // `total_cmp` is NaN-safe (NaN sorts deterministically), unlike
    // `partial_cmp(...).unwrap_or(Equal)` which violates total order
    // when NaN is present and panics in Rust 1.81+ sort. NaN
    // predictions arise when a transform-bearing bake produces flat
    // output for heavy-distortion pairs — keep the per-band SROCC
    // path robust against them.
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

/// Bootstrap 95% CI for |Spearman(a, b)| using `iters` resamples and a
/// deterministic xorshift64 seed. Returns (lo, hi) at the 2.5% / 97.5%
/// percentile of the empirical distribution.
fn bootstrap_srocc_ci_95(a: &[f64], b: &[f64], iters: usize, seed: u64) -> (f64, f64) {
    let n = a.len();
    if n < 4 || iters < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    let mut a_b = vec![0.0f64; n];
    let mut b_b = vec![0.0f64; n];
    for _ in 0..iters {
        for k in 0..n {
            let idx = (next() % (n as u64)) as usize;
            a_b[k] = a[idx];
            b_b[k] = b[idx];
        }
        samples.push(spearman(&a_b, &b_b).abs());
    }
    // total_cmp: NaN-safe deterministic ordering. See ranks() comment.
    samples.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = ((iters as f64) * 0.025).round() as usize;
    let hi_idx = (((iters as f64) * 0.975).round() as usize).min(iters - 1);
    (samples[lo_idx], samples[hi_idx])
}

#[cfg(test)]
mod tests {
    //! Unit tests for the statistical helpers — focused on the 4-parameter
    //! logistic rescale added 2026-05-14 to align Z-RMSE with Mohammadi
    //! 2025 (arXiv:2509.13150). The headline gate is the AIC-3 anchor
    //! reproduction at the bottom: SSIMULACRA2 Z-RMSE must reproduce
    //! 47.63 ±0.5 against the Mohammadi anchor CSV.
    //!
    //! Run only the tests in this file:
    //!     cargo test -p zensim-bench --release --example dataset_metric_baseline
    //!
    //! The anchor test is gated on the CSV being present at
    //! `/mnt/v/input/datasets/aic3/EvaluationMetrics/Anchor_assessment_on_PTC_full_resolution_Aug_3_2025.csv` —
    //! it skips with a printed note if missing, rather than failing CI.
    use super::*;

    /// Mohammadi-style fit: target is sigmoidal-in-predicted — i.e., the
    /// raw metric `predicted` maps via a logistic to the MOS `target`.
    /// This is the actual use case: PSNR/SSIMULACRA2 are roughly linear
    /// in the middle of their range but saturate at the extremes, and
    /// the saturation maps onto a finite MOS range via a logistic.
    ///
    /// Logistic must recover near-zero Z-RMSE; affine, with the
    /// saturating extremes pulling the fit, must be substantially worse.
    #[test]
    fn logistic_recovers_sigmoidal_metric() {
        // predicted spans a wide range (− saturation through + saturation).
        // target is the logistic of predicted (the ground-truth relation).
        let n = 40;
        let mut predicted = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        // Spread predicted from -6 to +6 so the logistic saturates at both ends.
        for k in 0..n {
            let x = -6.0 + 12.0 * (k as f64) / (n as f64 - 1.0);
            predicted.push(x);
            // MOS range 1..=10 saturating at the extremes.
            let t = 1.0 + 9.0 / (1.0 + (-x).exp());
            target.push(t);
        }

        let fit_aff = rescale_to_match(&predicted, &target);
        let fit_log = rescale_logistic(&predicted, &target);

        let z_aff = z_rmse(&fit_aff, &target, None);
        let z_log = z_rmse(&fit_log, &target, None);

        assert!(z_log.is_finite() && z_aff.is_finite(), "Z-RMSE not finite");
        // Logistic recovers the ground-truth mapping → near-zero residual.
        assert!(
            z_log < 0.01,
            "expected Z-RMSE near zero on clean logistic data, got {}",
            z_log
        );
        // Affine leaves the saturation regions as systematic residual.
        // The exact gap depends on n and shape; require ≥10× improvement.
        assert!(
            z_log < z_aff / 10.0,
            "logistic Z-RMSE {} should be ≥10× smaller than affine Z-RMSE {}",
            z_log,
            z_aff
        );
    }

    /// Anti-correlated case: distance metric where lower = better quality
    /// (e.g. PSNR is "score" but Butteraugli is "distance"). b4 should
    /// fit negative; logistic must still match the true relation.
    #[test]
    fn logistic_handles_decreasing_metric() {
        let n = 40;
        let mut predicted = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for k in 0..n {
            // Predicted is a "distance" — higher value = worse quality.
            // Sweep 0..=12 so the logistic saturates on both ends.
            let x = 12.0 * (k as f64) / (n as f64 - 1.0);
            predicted.push(x);
            // Target is MOS-like: high at low distance, low at high distance.
            // Use a decreasing logistic; b4 = -2 implicit.
            let t = 1.0 + 9.0 / (1.0 + ((x - 6.0) / 2.0).exp());
            target.push(t);
        }
        let fit_log = rescale_logistic(&predicted, &target);
        let z_log = z_rmse(&fit_log, &target, None);
        assert!(
            z_log.is_finite(),
            "Z-RMSE must be finite for anti-correlated fit"
        );
        // Decreasing logistic should recover with near-zero residual.
        assert!(
            z_log < 0.01,
            "expected logistic to flip cleanly on anti-correlated predictor; got Z-RMSE {}",
            z_log
        );
    }

    /// Linear predictor: logistic should still fit at least as well as
    /// affine (the logistic family contains the linear function in its
    /// limit). Tolerance accounts for LM convergence noise.
    #[test]
    fn logistic_matches_affine_on_linear_metric() {
        let target: Vec<f64> = (0..50)
            .map(|i| 10.0 + 1.5 * (i as f64) / 49.0 * 80.0)
            .collect();
        let predicted: Vec<f64> = (0..50).map(|i| -5.0 + 0.7 * i as f64).collect();
        let aff = rescale_to_match(&predicted, &target);
        let log = rescale_logistic(&predicted, &target);
        let z_aff = z_rmse(&aff, &target, None);
        let z_log = z_rmse(&log, &target, None);
        // Logistic shouldn't be drastically worse than affine on linear data.
        assert!(
            z_log < z_aff * 1.5 + 1e-3,
            "z_log={} > 1.5 * z_aff={}",
            z_log,
            z_aff
        );
    }

    /// Degenerate input (all predictions identical): logistic should
    /// fall back to affine without panicking.
    #[test]
    fn logistic_falls_back_on_degenerate_input() {
        let target: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let predicted = vec![3.14; 10];
        let fit = rescale_logistic(&predicted, &target);
        assert_eq!(fit.len(), 10);
        // After fallback to affine on zero-variance input, the rescale
        // is a constant function — every output equals mean(target).
        let mean_t = target.iter().sum::<f64>() / target.len() as f64;
        for v in fit {
            assert!(
                (v - mean_t).abs() < 1e-9 || v == 3.14,
                "fit value {} unexpected",
                v
            );
        }
    }

    /// Sub-minimum input (n < 4): falls back gracefully.
    #[test]
    fn logistic_falls_back_on_tiny_input() {
        let target = vec![1.0, 2.0, 3.0];
        let predicted = vec![0.5, 1.0, 1.5];
        let fit = rescale_logistic(&predicted, &target);
        assert_eq!(fit.len(), 3);
        for v in &fit {
            assert!(v.is_finite(), "fallback must produce finite values");
        }
    }

    /// AIC-3 anchor reproduction — the headline acceptance gate.
    ///
    /// Reads Mohammadi's anchor CSV (`Anchor_assessment_on_PTC_full_resolution_Aug_3_2025.csv`)
    /// and checks that our Rust logistic fit reproduces the paper's
    /// SSIMULACRA2 Z-RMSE of 47.63 within ±0.5.
    ///
    /// Gated by env var per CLAUDE.md "NO GRACEFUL SKIPS IN TESTS":
    /// the skip decision must be visible in the full chain (CI →
    /// justfile → test invocation), never made silently inside the
    /// test body.
    ///
    /// - **Default**: not run (skip with a loud notice).
    /// - **`ZENSIM_TEST_AIC3=1`**: enable. The CSV path is the standard
    ///   /mnt/v location (`Anchor_assessment_on_PTC_full_resolution_
    ///   Aug_3_2025.csv`) — fails LOUDLY if the file isn't there.
    /// - **`ZENSIM_AIC3_CSV=/path/to/custom.csv`**: override the path
    ///   (still requires `ZENSIM_TEST_AIC3=1`).
    #[test]
    fn anchor_csv_reproduces_mohammadi_zrmse() {
        if std::env::var("ZENSIM_TEST_AIC3").is_err() {
            eprintln!(
                "anchor_csv_reproduces_mohammadi_zrmse: skip — set ZENSIM_TEST_AIC3=1 to enable"
            );
            return;
        }
        let csv_path = std::env::var("ZENSIM_AIC3_CSV").unwrap_or_else(|_| {
            "/mnt/v/input/datasets/aic3/EvaluationMetrics/Anchor_assessment_on_PTC_full_resolution_Aug_3_2025.csv"
                .to_string()
        });
        assert!(
            std::path::Path::new(&csv_path).exists(),
            "ZENSIM_TEST_AIC3=1 but CSV missing at {csv_path}; \
             set ZENSIM_AIC3_CSV to override the path"
        );
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(csv_path)
            .expect("anchor CSV failed to open");
        let headers = reader.headers().expect("CSV headers required").clone();
        // Locate columns by name.
        let col_mos = headers
            .iter()
            .position(|h| h == "distortion")
            .expect("distortion (MOS) column missing");
        let col_sigma = headers
            .iter()
            .position(|h| h == "std_bootstrap")
            .expect("std_bootstrap column missing");
        let col_ssim2 = headers
            .iter()
            .position(|h| h == "SSIMULACRA2")
            .expect("SSIMULACRA2 column missing");
        let col_psnry = headers.iter().position(|h| h == "psnry");
        let col_iwssim = headers.iter().position(|h| h == "iw_ssim");
        let col_cvvdp = headers.iter().position(|h| h == "CVVDP");

        let mut mos = Vec::new();
        let mut sigma = Vec::new();
        let mut ssim2 = Vec::new();
        let mut psnry: Vec<f64> = Vec::new();
        let mut iwssim: Vec<f64> = Vec::new();
        let mut cvvdp: Vec<f64> = Vec::new();
        for rec in reader.records() {
            let rec = rec.expect("CSV row failed to parse");
            let m: f64 = rec[col_mos].parse().expect("MOS not parseable");
            let s: f64 = rec[col_sigma].parse().expect("sigma not parseable");
            let v: f64 = rec[col_ssim2].parse().expect("SSIMULACRA2 not parseable");
            mos.push(m);
            sigma.push(s);
            ssim2.push(v);
            if let Some(i) = col_psnry {
                psnry.push(rec[i].parse().unwrap_or(f64::NAN));
            }
            if let Some(i) = col_iwssim {
                iwssim.push(rec[i].parse().unwrap_or(f64::NAN));
            }
            if let Some(i) = col_cvvdp {
                cvvdp.push(rec[i].parse().unwrap_or(f64::NAN));
            }
        }

        let n = mos.len();
        assert!(n >= 100, "expected ≥100 stimuli in anchor CSV, got {}", n);
        eprintln!("anchor: loaded {} stimuli", n);

        // SSIMULACRA2 Z-RMSE — paper Table I value 47.63.
        let fit_ssim2 = rescale_logistic(&ssim2, &mos);
        let z_ssim2 = z_rmse(&fit_ssim2, &mos, Some(&sigma));
        eprintln!(
            "anchor SSIMULACRA2 Z-RMSE = {:.4} (paper 47.63, tol ±0.5)",
            z_ssim2
        );
        assert!(
            (z_ssim2 - 47.63).abs() <= 0.5,
            "SSIMULACRA2 Z-RMSE {:.4} deviates from Mohammadi 2025's 47.63 by > 0.5",
            z_ssim2
        );

        // Assert the other paper-listed metrics from Mohammadi Table I
        // also reproduce within ±0.5. These are the "load-bearing
        // four" referenced in the task setup: PSNR-Y, IW-SSIM, CVVDP,
        // SSIMULACRA2.
        let assert_within = |name: &str, vals: &[f64], paper: f64| {
            if vals.is_empty() {
                return;
            }
            let fit = rescale_logistic(vals, &mos);
            let z = z_rmse(&fit, &mos, Some(&sigma));
            eprintln!(
                "anchor {} Z-RMSE      = {:.4} (paper {:.2})",
                name, z, paper
            );
            assert!(
                (z - paper).abs() <= 0.5,
                "{} Z-RMSE {:.4} deviates from Mohammadi 2025's {:.2} by > 0.5",
                name,
                z,
                paper
            );
        };
        assert_within("PSNR-Y ", &psnry, 13.36);
        assert_within("IW-SSIM", &iwssim, 31.51);
        assert_within("CVVDP  ", &cvvdp, 9.45);
    }

    /// Dispatch-boundary test for `FeatureRegime::from_n_inputs`.
    ///
    /// The boundaries are load-bearing: 228 → Standard (`Zensim::compute`),
    /// 300 → Extended (adds masked), 372 → ExtendedIw (adds IW pool).
    /// Wrong dispatch leads to feeding truncated features into a network
    /// that expects more columns (silent NaN cascade — see commit
    /// 8baa8e48 for the V_20a IW debug that motivated this wiring).
    #[test]
    fn feature_regime_dispatch_boundaries() {
        use FeatureRegime::*;
        // 228 inclusive bound: standard
        assert_eq!(FeatureRegime::from_n_inputs(0), Standard);
        assert_eq!(FeatureRegime::from_n_inputs(1), Standard);
        assert_eq!(FeatureRegime::from_n_inputs(227), Standard);
        assert_eq!(FeatureRegime::from_n_inputs(228), Standard);
        // 229..=300 inclusive: extended
        assert_eq!(FeatureRegime::from_n_inputs(229), Extended);
        assert_eq!(FeatureRegime::from_n_inputs(299), Extended);
        assert_eq!(FeatureRegime::from_n_inputs(300), Extended);
        // 301..: extended+IW
        assert_eq!(FeatureRegime::from_n_inputs(301), ExtendedIw);
        assert_eq!(FeatureRegime::from_n_inputs(371), ExtendedIw);
        assert_eq!(FeatureRegime::from_n_inputs(372), ExtendedIw);
        // Future hypothetical wider bakes still get the largest regime,
        // not a separate variant — graceful for forward compatibility.
        assert_eq!(FeatureRegime::from_n_inputs(500), ExtendedIw);
        assert_eq!(FeatureRegime::from_n_inputs(usize::MAX), ExtendedIw);
    }

    /// Regression test: ranks() must not panic when inputs contain NaN.
    ///
    /// The original `partial_cmp(...).unwrap_or(Ordering::Equal)` pattern
    /// violated total order when NaN was present — `5.0 == NaN` AND
    /// `6.0 == NaN` but `5.0 != 6.0` is a transitivity break and Rust
    /// 1.81+ sort panics with "user-provided comparison function does
    /// not correctly implement a total order". The fix uses
    /// `f64::total_cmp` which is NaN-safe.
    ///
    /// This NaN is real, not synthetic: transform-bearing bakes
    /// (V_20+) on heavy-distortion pairs (TID B0/B1) sometimes produce
    /// flat scaler output → flat predictions → NaN in the resulting
    /// ranks. The eval harness must survive this rather than crashing
    /// halfway through a multi-corpus run.
    #[test]
    fn ranks_handles_nan_without_panic() {
        let v = [0.5_f64, f64::NAN, 0.3, 0.8, f64::NAN, 0.1];
        let r = ranks(&v);
        // 6 elements → ranks sum to 0+1+2+3+4+5 = 15 regardless of NaN.
        let sum: f64 = r.iter().sum();
        assert!(
            (sum - 15.0).abs() < 1e-9,
            "rank sum should be 15 (0..5 ranks); got {sum} (ranks {r:?})"
        );
        // Non-NaN values must be ranked correctly relative to each other.
        // total_cmp places NaN at the high end, so 0.1, 0.3, 0.5, 0.8
        // get ranks 0, 1, 2, 3 (in some order — the two NaNs land at 4, 5).
        let non_nan_indices = [5usize, 2, 0, 3]; // 0.1, 0.3, 0.5, 0.8
        for (rank_pos, &i) in non_nan_indices.iter().enumerate() {
            assert!(
                (r[i] - rank_pos as f64).abs() < 1e-9,
                "v[{i}]={} should have rank {rank_pos}, got {}",
                v[i],
                r[i]
            );
        }
    }
}
