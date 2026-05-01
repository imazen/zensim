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

fn main() {
    let mut args = std::env::args().skip(1);
    let mut kadid: Option<PathBuf> = None;
    let mut tid: Option<PathBuf> = None;
    let mut cid22: Option<PathBuf> = None;
    let mut v04_bake_path: Option<PathBuf> = None;
    let mut max_pairs: usize = 500;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--kadid" => kadid = Some(args.next().unwrap().into()),
            "--tid" => tid = Some(args.next().unwrap().into()),
            "--cid22" => cid22 = Some(args.next().unwrap().into()),
            "--v04-bake" => v04_bake_path = Some(args.next().unwrap().into()),
            "--max-pairs" => max_pairs = args.next().unwrap().parse().unwrap(),
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

    if datasets.is_empty() {
        eprintln!("no datasets — pass at least one of --kadid, --tid, --cid22");
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

        eprintln!(
            "  done {n_valid}/{n} valid in {:.1}s",
            started.elapsed().as_secs_f64()
        );
    }
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
