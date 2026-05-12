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
use zensim::{RgbSlice, Zensim, ZensimProfile};

#[derive(Debug, Clone)]
struct Pair {
    reference: PathBuf,
    distorted: PathBuf,
    human_score: f64,
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
    let mut max_pairs: usize = 500;
    let mut per_pair_output: Option<PathBuf> = None;
    let mut konjnd_features_csv: Option<PathBuf> = None;
    let mut konjnd_anchor_target: f64 = 63.0;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--kadid" => kadid = Some(args.next().unwrap().into()),
            "--tid" => tid = Some(args.next().unwrap().into()),
            "--cid22" => cid22 = Some(args.next().unwrap().into()),
            "--csiq" => csiq = Some(args.next().unwrap().into()),
            "--aic3" => aic3 = Some(args.next().unwrap().into()),
            "--konjnd" => konjnd = Some(args.next().unwrap().into()),
            "--v04-bake" => v04_bake_path = Some(args.next().unwrap().into()),
            "--max-pairs" => max_pairs = args.next().unwrap().parse().unwrap(),
            "--per-pair-output" => per_pair_output = Some(args.next().unwrap().into()),
            "--konjnd-features-csv" => konjnd_features_csv = Some(args.next().unwrap().into()),
            "--konjnd-anchor-target" => konjnd_anchor_target = args.next().unwrap().parse().unwrap(),
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
        eprintln!("no datasets — pass at least one of --kadid, --tid, --cid22, --csiq, --aic3, --konjnd");
        std::process::exit(1);
    }

    // Optionally load V0_4 trained bake.
    let v04_bake_bytes: Option<Vec<u8>> = v04_bake_path.as_ref().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("failed to read v04 bake at {:?}: {e}", p);
            std::process::exit(1);
        })
    });

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
        let z_v04 = Zensim::new(ZensimProfile::PreviewV0_4);

        type MetricRow = (f64, f64, f64, f64, f64);
        let results: Vec<Option<MetricRow>> = ds
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
                process_pair(pair, &z_v04, v04_bake_bytes.as_deref())
            })
            .collect();

        let pairs_with: Vec<(f64, f64, f64, f64, f64)> = results.into_iter().flatten().collect();
        let n_valid = pairs_with.len();
        if n_valid < 4 {
            println!(
                "| {} | {} | n/a (only {} valid) | | | |",
                ds.name, n, n_valid
            );
            continue;
        }
        let humans: Vec<f64> = pairs_with.iter().map(|t| t.0).collect();
        let v02: Vec<f64> = pairs_with.iter().map(|t| t.1).collect();
        let v04: Vec<f64> = pairs_with.iter().map(|t| t.2).collect();
        let ssim2: Vec<f64> = pairs_with.iter().map(|t| t.3).collect();
        let butter: Vec<f64> = pairs_with.iter().map(|t| t.4).collect();

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

        if let Some(w) = per_pair_writer.as_mut() {
            for row in &pairs_with {
                w.write_record([
                    ds.name,
                    &format!("{:.6}", row.0),
                    &format!("{:.6}", row.1),
                    &format!("{:.6}", row.2),
                    &format!("{:.4}", row.3),
                    &format!("{:.6}", row.4),
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
                ("B1 medium [50,65)",      0.50,           0.65),
                ("B2 high [65,90)",        0.65,           0.90),
                ("B3 visually-lossless (≥90)", 0.90,       f64::INFINITY),
                ("Near-PJND [58,68]",      0.58,           0.68),
            ]),
            "KADIK10k" => Some([
                ("B0 below medium (<3.7)", -f64::INFINITY, 0.675),
                ("B1 medium [3.7,4.3)",    0.675,          0.825),
                ("B2 high [4.3,4.5)",      0.825,          0.875),
                ("B3 visually-lossless (≥4.5)", 0.875,     f64::INFINITY),
                ("Near-PJND [3.9,4.2]",    0.725,          0.800),
            ]),
            "TID2013" => Some([
                ("B0 below medium (<4.5)", -f64::INFINITY, 0.500),
                ("B1 medium [4.5,5.5)",    0.500,          0.611),
                ("B2 high [5.5,6.0)",      0.611,          0.667),
                ("B3 visually-lossless (≥6.0)", 0.667,     f64::INFINITY),
                ("Near-PJND [4.8,5.2]",    0.533,          0.578),
            ]),
            _ => None,
        };
        if let Some(bands) = bands {
            println!();
            println!("### {} per-band SROCC (vs human MOS)", ds.name);
            println!();
            println!("| Band | n | V0_2 | V0_4 (bake) | V0_4 95% CI | fast-ssim2 | butter | V0_4 MAE | V0_2 MAE |");
            println!("|---|--:|:--:|:--:|:--:|:--:|:--:|--:|--:|");
            for (label, lo, hi) in &bands {
                let idxs: Vec<usize> = humans
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &h)| (h >= *lo && h < *hi).then_some(i))
                    .collect();
                if idxs.len() < 4 {
                    println!("| {label} | {} | n/a | n/a | n/a | n/a | n/a | n/a |", idxs.len());
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
                let mae_v04: f64 = idxs.iter()
                    .map(|&i| ((100.0 - v04[i]) - humans[i] * 100.0).abs())
                    .sum::<f64>() / idxs.len() as f64;
                // V0_2 outputs raw distance ~ 0..90. Skip score-mapping for
                // V0_2 MAE; just report mean(|distance - (100 - MCOS)|) as a
                // rough cross-scale anchor.
                let mae_v02: f64 = idxs.iter()
                    .map(|&i| (v02[i] - (100.0 - humans[i] * 100.0)).abs())
                    .sum::<f64>() / idxs.len() as f64;
                println!(
                    "| {label} | {} | {s_v02:.4} | {s_v04:.4} | [{ci_lo:.2}, {ci_hi:.2}] | {s_ssim:.4} | {s_ba:.4} | {mae_v04:.2} | {mae_v02:.2} |",
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
            let z_v04 = Zensim::new(ZensimProfile::PreviewV0_4);
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
            let mut p = Predictor::new(model);
            p.predict(&f32_features).ok()?[0] as f64
        }
    } else {
        f64::NAN
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
    let features_vec: Vec<f64> = features.iter().copied().collect();
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

fn process_pair(
    pair: &Pair,
    z_v04: &Zensim,
    v04_bake: Option<&[u8]>,
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

    // Single compute pass via V0_4 profile — this populates the full
    // 228-feature vector (the MLP path forces compute_all_features=true).
    // We then score V0_2 manually (dot product with V0_2 weights) and
    // V0_4 manually (Predictor over the trained bake).
    let result = z_v04.compute(&s, &d).ok()?;
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
            let mut p = Predictor::new(model);
            p.predict(&f32_features).ok()?[0] as f64
        }
    } else {
        f64::NAN
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
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
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
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
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
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_idx = ((iters as f64) * 0.025).round() as usize;
    let hi_idx = (((iters as f64) * 0.975).round() as usize).min(iters - 1);
    (samples[lo_idx], samples[hi_idx])
}
