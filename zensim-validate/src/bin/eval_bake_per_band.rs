//! Per-band evaluator: load a ZNPR v2/v3 bake + a features CSV,
//! forward-pass each row through the MLP, compute SROCC + per-band
//! SROCC against the `human_score` column.
//!
//! Built for V0_20a — bakes the trainer produces have 228 or 372 inputs;
//! the runtime can't load 372-input bakes yet, but we can eval them
//! directly via the trainer-side features CSV + zenpredict::Model.
//!
//! Usage:
//!   eval_bake_per_band --bake PATH --csv PATH [--csv NAME:PATH ...]
//!                       [--band-count N] [--band-edges 50,65,90]
//!                       [--out-md PATH]

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::{Model, Predictor};

// Canonical spearman / pearson live in `zenstats::panel` (re-exported
// via `zensim_validate::panel`). Pre-2026-05-26 this binary carried
// byte-identical inline copies; routed through the shared home now.
use zensim_validate::panel::{pearson, spearman};

fn load_csv(path: &PathBuf) -> std::io::Result<(Vec<f64>, Vec<Vec<f32>>)> {
    let f = BufReader::new(File::open(path)?);
    let mut lines = f.lines();
    // First line = header
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
        // col 0 = ref_basename, col 1 = human_score in [0, 1] or [0, 100]
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

fn predict_all(bake: &[u8], features: &[Vec<f32>]) -> Result<Vec<f64>, String> {
    let model = Model::from_bytes(bake).map_err(|e| format!("{:?}", e))?;
    let n_inputs = model.n_inputs();
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
        let out = predictor.predict(input).map_err(|e| format!("{:?}", e))?;
        preds.push(out[0] as f64);
    }
    Ok(preds)
}

fn band_label(idx: usize, edges: &[f64]) -> String {
    let lo = if idx == 0 { 0.0 } else { edges[idx - 1] };
    let hi = if idx >= edges.len() {
        100.0
    } else {
        edges[idx]
    };
    format!("[{:.0},{:.0})", lo, hi)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let mut bake_path: Option<PathBuf> = None;
    let mut csvs: Vec<(String, PathBuf)> = Vec::new();
    let mut band_edges: Vec<f64> = vec![50.0, 65.0, 90.0]; // V0_18 4-band CID22 cuts
    let mut out_md: Option<PathBuf> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => bake_path = Some(args.next().expect("--bake PATH").into()),
            "--csv" => {
                let s = args.next().expect("--csv NAME:PATH or PATH");
                if let Some((name, path)) = s.split_once(':') {
                    csvs.push((name.to_string(), path.into()));
                } else {
                    csvs.push(("csv".to_string(), s.into()));
                }
            }
            "--band-edges" => {
                let s = args.next().expect("--band-edges comma-list");
                band_edges = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            }
            "--out-md" => out_md = Some(args.next().expect("--out-md PATH").into()),
            other => {
                eprintln!("unknown arg: {}", other);
                return ExitCode::from(2);
            }
        }
    }
    let bake_path = match bake_path {
        Some(p) => p,
        None => {
            eprintln!(
                "usage: eval_bake_per_band --bake PATH --csv NAME:PATH [--csv ...] [--band-edges 50,65,90] [--out-md PATH]"
            );
            return ExitCode::from(2);
        }
    };
    if csvs.is_empty() {
        eprintln!("at least one --csv is required");
        return ExitCode::from(2);
    }

    let bake = match std::fs::read(&bake_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read bake {}: {}", bake_path.display(), e);
            return ExitCode::from(1);
        }
    };

    let mut writer: Box<dyn Write> = match &out_md {
        Some(p) => Box::new(File::create(p).expect("create out-md")),
        None => Box::new(std::io::stdout()),
    };

    let bake_name = bake_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("bake")
        .to_string();

    writeln!(writer, "# Per-band eval: `{}`", bake_name).ok();
    writeln!(writer).ok();
    writeln!(writer, "Bake bytes: {}", bake.len()).ok();
    writeln!(writer, "Band edges (human_score): {:?}", band_edges).ok();
    writeln!(writer).ok();

    for (name, csv) in &csvs {
        let (scores, features) = match load_csv(csv) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("load {}: {}", csv.display(), e);
                return ExitCode::from(1);
            }
        };
        let preds = match predict_all(&bake, &features) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("predict on {}: {}", csv.display(), e);
                return ExitCode::from(1);
            }
        };

        // Rescale human_score to [0, 100] if needed
        let scores_pct: Vec<f64> = if scores.iter().copied().fold(0.0_f64, f64::max) <= 1.5 {
            scores.iter().map(|s| s * 100.0).collect()
        } else {
            scores.clone()
        };

        // Aggregate SROCC + PLCC
        let s_all = spearman(&preds, &scores_pct).abs();
        let p_all = pearson(&preds, &scores_pct).abs();
        writeln!(
            writer,
            "## `{}` (n={}, {} features per row)",
            name,
            scores.len(),
            features.first().map(|f| f.len()).unwrap_or(0),
        )
        .ok();
        writeln!(writer).ok();
        writeln!(
            writer,
            "**Aggregate**: SROCC = {:.4}, PLCC = {:.4}",
            s_all, p_all
        )
        .ok();
        writeln!(writer).ok();

        // Per-band: bucket by human score
        let n_bands = band_edges.len() + 1;
        let mut bands: Vec<Vec<(f64, f64)>> = (0..n_bands).map(|_| Vec::new()).collect();
        for (i, &s) in scores_pct.iter().enumerate() {
            let mut b = n_bands - 1;
            for (bi, edge) in band_edges.iter().enumerate() {
                if s < *edge {
                    b = bi;
                    break;
                }
            }
            bands[b].push((preds[i], s));
        }
        writeln!(writer, "| Band | range | n | SROCC | PLCC |").ok();
        writeln!(writer, "|---|---|---:|---:|---:|").ok();
        for (i, b) in bands.iter().enumerate() {
            let n = b.len();
            if n < 10 {
                writeln!(
                    writer,
                    "| B{} | {} | {} | (n<10) | (n<10) |",
                    i,
                    band_label(i, &band_edges),
                    n
                )
                .ok();
                continue;
            }
            let p: Vec<f64> = b.iter().map(|(p, _)| *p).collect();
            let s: Vec<f64> = b.iter().map(|(_, s)| *s).collect();
            writeln!(
                writer,
                "| B{} | {} | {} | {:.4} | {:.4} |",
                i,
                band_label(i, &band_edges),
                n,
                spearman(&p, &s).abs(),
                pearson(&p, &s).abs(),
            )
            .ok();
        }
        writeln!(writer).ok();
    }

    ExitCode::SUCCESS
}
