//! Late-fusion ensemble over multiple ZNPR bakes against a labeled
//! features CSV. For each bake, forward-pass to get predictions,
//! then sweep mixture weights and report per-band SROCC.
//!
//! Built for V0_20a: confirms "can the MLP combine best of all worlds"
//! is reachable via output-level ensembling, which sidesteps the
//! single-MLP capacity bottleneck.
//!
//! Usage:
//!   ensemble_mix --csv kadid:PATH --bake baseline=PATH --bake iw_k1=PATH --bake iw_k4=PATH
//!                 [--band-edges 50,65,90]
//!
//! For each named bake, predictions are produced. Then a coarse grid
//! search over mixture weights (α₁..αₖ summing to 1, step 0.1) finds
//! the mixture that maximises aggregate SROCC and the mixture that
//! maximises per-band SROCC. Output is a markdown table.

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::{Model, Predictor};

fn ranks(v: &[f64]) -> Vec<f64> {
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
        let avg = (i + j - 1) as f64 / 2.0;
        for k in i..j {
            r[idx[k]] = avg;
        }
        i = j;
    }
    r
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let mean = (n as f64 - 1.0) / 2.0;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let xa = ra[i] - mean;
        let xb = rb[i] - mean;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

fn load_csv(path: &PathBuf) -> std::io::Result<(Vec<f64>, Vec<Vec<f32>>)> {
    let f = BufReader::new(File::open(path)?);
    let mut lines = f.lines();
    let _header = lines.next().transpose()?.unwrap_or_default();
    let mut scores = Vec::new();
    let mut features = Vec::new();
    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 3 {
            continue;
        }
        let score: f64 = match cols[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let f32_features: Vec<f32> = cols[2..]
            .iter()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();
        scores.push(score);
        features.push(f32_features);
    }
    Ok((scores, features))
}

fn predict_all(bake_bytes: &[u8], features: &[Vec<f32>]) -> Result<Vec<f64>, String> {
    let model = Model::from_bytes(bake_bytes).map_err(|e| format!("{:?}", e))?;
    let n_inputs = model.n_inputs();
    // V_20+ bakes carry feature_transforms metadata; dispatch accordingly.
    // predict_transformed is a no-op overhead for bakes without transforms.
    let needs_transforms = model.has_nontrivial_feature_transforms();
    let mut predictor = Predictor::new(&model);
    let mut preds = Vec::with_capacity(features.len());
    for row in features {
        let input = if row.len() == n_inputs {
            &row[..]
        } else if row.len() > n_inputs {
            &row[..n_inputs]
        } else {
            return Err(format!(
                "row has {} features but bake expects {}",
                row.len(),
                n_inputs
            ));
        };
        let out = if needs_transforms {
            predictor
                .predict_transformed(input)
                .map_err(|e| format!("{:?}", e))?
        } else {
            predictor.predict(input).map_err(|e| format!("{:?}", e))?
        };
        preds.push(out[0] as f64);
    }
    Ok(preds)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let mut csvs: Vec<(String, PathBuf)> = Vec::new();
    let mut bakes: Vec<(String, PathBuf)> = Vec::new();
    let mut band_edges: Vec<f64> = vec![50.0, 65.0, 90.0];

    while let Some(a) = args.next() {
        match a.as_str() {
            "--csv" => {
                let s = args.next().expect("--csv NAME:PATH");
                if let Some((name, path)) = s.split_once(':') {
                    csvs.push((name.to_string(), path.into()));
                } else {
                    eprintln!("--csv expects NAME:PATH");
                    return ExitCode::from(2);
                }
            }
            "--bake" => {
                let s = args.next().expect("--bake NAME=PATH");
                if let Some((name, path)) = s.split_once('=') {
                    bakes.push((name.to_string(), path.into()));
                } else {
                    eprintln!("--bake expects NAME=PATH");
                    return ExitCode::from(2);
                }
            }
            "--band-edges" => {
                let s = args.next().expect("--band-edges comma-list");
                band_edges = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            }
            _ => {
                eprintln!("unknown arg: {}", a);
                return ExitCode::from(2);
            }
        }
    }

    if csvs.is_empty() || bakes.is_empty() {
        eprintln!("usage: ensemble_mix --csv NAME:PATH [--csv ...] --bake NAME=PATH [--bake ...]");
        return ExitCode::from(2);
    }

    // Z-score normalize per bake so output ranges are comparable
    let normalize = |xs: &[f64]| -> Vec<f64> {
        let n = xs.len();
        if n < 2 {
            return xs.to_vec();
        }
        let mean: f64 = xs.iter().sum::<f64>() / n as f64;
        let var: f64 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let sd = var.sqrt().max(1e-12);
        xs.iter().map(|x| (x - mean) / sd).collect()
    };

    for (name, csv) in &csvs {
        let (scores, features) = match load_csv(csv) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("load {}: {}", csv.display(), e);
                return ExitCode::from(1);
            }
        };
        let scores_pct: Vec<f64> = if scores.iter().copied().fold(0.0f64, f64::max) <= 1.5 {
            scores.iter().map(|s| s * 100.0).collect()
        } else {
            scores.clone()
        };

        let mut bake_preds: HashMap<String, Vec<f64>> = HashMap::new();
        for (bname, bpath) in &bakes {
            let bytes = match std::fs::read(bpath) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("read {}: {}", bpath.display(), e);
                    return ExitCode::from(1);
                }
            };
            let preds = match predict_all(&bytes, &features) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("predict {} on {}: {}", bname, csv.display(), e);
                    return ExitCode::from(1);
                }
            };
            bake_preds.insert(bname.clone(), normalize(&preds));
        }

        println!("# Ensemble mix on `{}` (n={})", name, scores.len());
        println!();

        // Singleton scores per band first
        println!("## Per-bake baseline (z-normalized predictions)");
        println!();
        println!("| Bake | agg SROCC | B0 | B1 | B2 | B3 |");
        println!("|---|---:|---:|---:|---:|---:|");
        let per_band_indices: Vec<Vec<usize>> = {
            let mut bs: Vec<Vec<usize>> = (0..=band_edges.len()).map(|_| vec![]).collect();
            for (i, &s) in scores_pct.iter().enumerate() {
                let mut b = band_edges.len();
                for (bi, e) in band_edges.iter().enumerate() {
                    if s < *e {
                        b = bi;
                        break;
                    }
                }
                bs[b].push(i);
            }
            bs
        };
        for (bname, _) in &bakes {
            let preds = &bake_preds[bname];
            let agg = spearman(preds, &scores_pct).abs();
            let bands: Vec<String> = per_band_indices
                .iter()
                .map(|idxs| {
                    if idxs.len() < 10 {
                        "n<10".to_string()
                    } else {
                        let p: Vec<f64> = idxs.iter().map(|&i| preds[i]).collect();
                        let s: Vec<f64> = idxs.iter().map(|&i| scores_pct[i]).collect();
                        format!("{:.4}", spearman(&p, &s).abs())
                    }
                })
                .collect();
            println!("| {} | {:.4} | {} |", bname, agg, bands.join(" | "));
        }
        println!();

        // 2-bake mix: sweep α from 0 to 1 in 0.1 steps for every pair
        let bake_names: Vec<&String> = bakes.iter().map(|(n, _)| n).collect();
        if bake_names.len() >= 2 {
            println!("## 2-bake mix sweep (α step 0.1)");
            println!();
            for i in 0..bake_names.len() {
                for j in (i + 1)..bake_names.len() {
                    let a = bake_names[i];
                    let b = bake_names[j];
                    let pa = &bake_preds[a];
                    let pb = &bake_preds[b];

                    println!("### `{}` + `{}`", a, b);
                    println!();
                    println!("| α (weight on {}) | agg | B0 | B1 | B2 | B3 |", a);
                    println!("|---:|---:|---:|---:|---:|---:|");
                    let mut best_agg = (0.0, 0.0);
                    let mut best_b3 = (0.0, 0.0);
                    for k in 0..=10 {
                        let alpha = k as f64 / 10.0;
                        let mix: Vec<f64> = pa
                            .iter()
                            .zip(pb)
                            .map(|(x, y)| alpha * x + (1.0 - alpha) * y)
                            .collect();
                        let agg = spearman(&mix, &scores_pct).abs();
                        if agg > best_agg.1 {
                            best_agg = (alpha, agg);
                        }
                        let bands: Vec<(String, f64)> = per_band_indices
                            .iter()
                            .map(|idxs| {
                                if idxs.len() < 10 {
                                    return ("n<10".to_string(), 0.0);
                                }
                                let p: Vec<f64> = idxs.iter().map(|&i| mix[i]).collect();
                                let s: Vec<f64> = idxs.iter().map(|&i| scores_pct[i]).collect();
                                let v = spearman(&p, &s).abs();
                                (format!("{:.4}", v), v)
                            })
                            .collect();
                        // Track best B3
                        if bands.len() >= 4 && bands[3].1 > best_b3.1 {
                            best_b3 = (alpha, bands[3].1);
                        }
                        println!(
                            "| {:.1} | {:.4} | {} |",
                            alpha,
                            agg,
                            bands.iter().map(|(s, _)| s.clone()).collect::<Vec<_>>().join(" | ")
                        );
                    }
                    println!();
                    println!(
                        "**best agg**: α={:.1} ({:.4}); **best B3**: α={:.1} ({:.4})",
                        best_agg.0, best_agg.1, best_b3.0, best_b3.1
                    );
                    println!();
                }
            }
        }

        // 3-bake mix: simplex grid for 3 bakes
        if bake_names.len() == 3 {
            let a = bake_names[0];
            let b = bake_names[1];
            let c = bake_names[2];
            let pa = &bake_preds[a];
            let pb = &bake_preds[b];
            let pc = &bake_preds[c];
            println!("## 3-bake mix sweep (simplex, step 0.1)");
            println!();
            let mut best_agg = (0.0, 0.0, 0.0, 0.0);
            let mut best_b3 = (0.0, 0.0, 0.0, 0.0);
            for ka in 0..=10 {
                for kb in 0..=(10 - ka) {
                    let alpha_a = ka as f64 / 10.0;
                    let alpha_b = kb as f64 / 10.0;
                    let alpha_c = 1.0 - alpha_a - alpha_b;
                    if alpha_c < 0.0 - 1e-9 {
                        continue;
                    }
                    let mix: Vec<f64> = pa
                        .iter()
                        .zip(pb)
                        .zip(pc)
                        .map(|((x, y), z)| alpha_a * x + alpha_b * y + alpha_c * z)
                        .collect();
                    let agg = spearman(&mix, &scores_pct).abs();
                    if agg > best_agg.3 {
                        best_agg = (alpha_a, alpha_b, alpha_c, agg);
                    }
                    if per_band_indices.len() >= 4 && per_band_indices[3].len() >= 10 {
                        let p: Vec<f64> = per_band_indices[3].iter().map(|&i| mix[i]).collect();
                        let s: Vec<f64> = per_band_indices[3].iter().map(|&i| scores_pct[i]).collect();
                        let v = spearman(&p, &s).abs();
                        if v > best_b3.3 {
                            best_b3 = (alpha_a, alpha_b, alpha_c, v);
                        }
                    }
                }
            }
            println!(
                "best agg: α_{}={:.1}, α_{}={:.1}, α_{}={:.1} → SROCC {:.4}",
                a, best_agg.0, b, best_agg.1, c, best_agg.2, best_agg.3
            );
            println!(
                "best B3: α_{}={:.1}, α_{}={:.1}, α_{}={:.1} → SROCC {:.4}",
                a, best_b3.0, b, best_b3.1, c, best_b3.2, best_b3.3
            );
            println!();
        }
    }

    ExitCode::SUCCESS
}
