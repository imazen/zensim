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
    let mut konjnd: Option<PathBuf> = None;
    let mut v04_bake_path: Option<PathBuf> = None;
    let mut max_pairs: usize = 500;
    let mut per_pair_output: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--kadid" => kadid = Some(args.next().unwrap().into()),
            "--tid" => tid = Some(args.next().unwrap().into()),
            "--cid22" => cid22 = Some(args.next().unwrap().into()),
            "--konjnd" => konjnd = Some(args.next().unwrap().into()),
            "--v04-bake" => v04_bake_path = Some(args.next().unwrap().into()),
            "--max-pairs" => max_pairs = args.next().unwrap().parse().unwrap(),
            "--per-pair-output" => per_pair_output = Some(args.next().unwrap().into()),
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

    if datasets.is_empty() && konjnd.is_none() {
        eprintln!("no datasets — pass at least one of --kadid, --tid, --cid22, --konjnd");
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
            type CalibRow = (f64, f64, f64, f64); // v02_dist, v04_dist, ssim2, butter
            let results: Vec<Option<(String, CalibRow)>> = pairs
                .par_iter()
                .map(|kp| {
                    let p = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if p.is_multiple_of(log_every) {
                        let elapsed = started.elapsed().as_secs_f64();
                        let rate = p as f64 / elapsed;
                        let eta = (n_total - p) as f64 / rate;
                        eprintln!("  konjnd {p}/{n_total} ({rate:.1}/s, ETA {eta:.0}s)");
                    }
                    process_konjnd_pair(kp, &z_v04, v04_bake_bytes.as_deref())
                        .map(|row| (kp.codec.clone(), row))
                })
                .collect();
            let scored: Vec<(String, CalibRow)> = results.into_iter().flatten().collect();
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
                    .filter_map(|(c, r)| (c == codec).then_some(r))
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
) -> Option<(f64, f64, f64, f64)> {
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
    Some((v02_distance, v04_distance, ssim2, three_norm))
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
