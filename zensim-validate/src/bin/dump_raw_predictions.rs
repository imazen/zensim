//! Dump per-row RAW (uncalibrated) bake predictions for a corpus parquet.
//!
//! Used to fit affine α, β for V0_5 bakes that ship without
//! score-range calibration. Reads a parquet (with `human_score` +
//! `f0..fN-1` features), loads a ZNPR v3 bake, scores every row via
//! the SAME `score_row` dispatch as `bake_verdict`, and writes TSV
//! `human_score\traw_prediction` to stdout (or `--output <path>`).
//!
//! Usage:
//!     dump_raw_predictions --bake <bake.bin> --parquet <corpus.parquet>
//!         [--target-column human_score] [--output <path.tsv>]
//!
//! When fitting α, β: pair these raw outputs against the ssim2-
//! aligned `human_score` column on a calibrated corpus (e.g., the
//! safesyn training parquet's `mix_cv40_iw60` column).

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::{Model, Predictor};

use zensim_validate::parquet_loader;

// Copy the score_row + extract functions inline to avoid wiring through
// bake_verdict's module structure. These are bit-exact duplicates of
// the corresponding functions in bake_verdict.rs.

type PerSampleAlphaHeadDispatch = (Vec<f32>, f32, Vec<f32>, f32, [f32; 4], f32, f32);
type HybridHeadDispatch = (Vec<f32>, f32, f32, [f32; 4], f32, f32);

fn extract_per_sample_alpha_head(model: &Model) -> Option<PerSampleAlphaHeadDispatch> {
    let md = model.metadata();
    let entry = md.get("zentrain.per_sample_alpha_head")?;
    let n_hidden = model.n_outputs();
    let expected = (2 * n_hidden + 8) * 4;
    if entry.value.len() != expected {
        return None;
    }
    let mut floats = Vec::with_capacity(2 * n_hidden + 8);
    for chunk in entry.value.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let w_alpha = floats[..n_hidden].to_vec();
    let b_alpha = floats[n_hidden];
    let rank_w = floats[n_hidden + 1..2 * n_hidden + 1].to_vec();
    let rank_b = floats[2 * n_hidden + 1];
    let reducer_w = [
        floats[2 * n_hidden + 2],
        floats[2 * n_hidden + 3],
        floats[2 * n_hidden + 4],
        floats[2 * n_hidden + 5],
    ];
    let reducer_b = floats[2 * n_hidden + 6];
    let p_norm = floats[2 * n_hidden + 7];
    Some((
        w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm,
    ))
}

fn extract_hybrid_head(model: &Model) -> Option<HybridHeadDispatch> {
    let md = model.metadata();
    let entry = md.get("zentrain.hybrid_head")?;
    let n_hidden = model.n_outputs();
    let expected = (n_hidden + 8) * 4;
    if entry.value.len() != expected {
        return None;
    }
    let mut floats = Vec::with_capacity(n_hidden + 8);
    for chunk in entry.value.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let rank_w = floats[..n_hidden].to_vec();
    let rank_b = floats[n_hidden];
    let alpha_logit = floats[n_hidden + 1];
    let reducer_w = [
        floats[n_hidden + 2],
        floats[n_hidden + 3],
        floats[n_hidden + 4],
        floats[n_hidden + 5],
    ];
    let reducer_b = floats[n_hidden + 6];
    let p_norm = floats[n_hidden + 7];
    Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm))
}

fn score_row(
    predictor: &mut Predictor<'_>,
    has_transforms: bool,
    per_sample_alpha_head: Option<&PerSampleAlphaHeadDispatch>,
    hybrid_head: Option<&HybridHeadDispatch>,
    f32_features: &mut [f32],
    row: &[f64],
) -> f64 {
    let n_inputs = f32_features.len();
    let take = n_inputs.min(row.len());
    for i in 0..take {
        f32_features[i] = row[i] as f32;
    }
    for f in &mut f32_features[take..] {
        *f = 0.0;
    }
    let result = if has_transforms {
        predictor.predict_transformed(f32_features)
    } else {
        predictor.predict(f32_features)
    };
    match result {
        Ok(out) => {
            if let Some((w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm)) =
                per_sample_alpha_head
            {
                let n = out.len() as f64;
                if n <= 0.0 || out.len() != rank_w.len() || out.len() != w_alpha.len() {
                    return f64::NAN;
                }
                let mut y_rank = *rank_b as f64;
                let mut alpha_logit = *b_alpha as f64;
                let mut sum = 0.0f64;
                let mut max_v = f64::NEG_INFINITY;
                let mut sum_p = 0.0f64;
                let p = *p_norm as f64;
                for (j, &h) in out.iter().enumerate() {
                    let hf = h as f64;
                    y_rank += hf * rank_w[j] as f64;
                    alpha_logit += hf * w_alpha[j] as f64;
                    sum += hf;
                    if hf > max_v {
                        max_v = hf;
                    }
                    sum_p += hf.abs().powf(p);
                }
                let mu = sum / n;
                let mut var = 0.0f64;
                for &h in out.iter() {
                    let d = h as f64 - mu;
                    var += d * d;
                }
                let sigma = (var / n).sqrt().max(0.0026);
                let p_norm_stat = (sum_p / n).powf(1.0 / p);
                let y_pool = mu * reducer_w[0] as f64
                    + sigma * reducer_w[1] as f64
                    + max_v * reducer_w[2] as f64
                    + p_norm_stat * reducer_w[3] as f64
                    + *reducer_b as f64;
                let alpha = {
                    let xc = alpha_logit.clamp(-20.0, 20.0);
                    1.0 / (1.0 + (-xc).exp())
                };
                alpha * y_rank + (1.0 - alpha) * y_pool
            } else if let Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm)) =
                hybrid_head
            {
                let n = out.len() as f64;
                if n <= 0.0 || out.len() != rank_w.len() {
                    return f64::NAN;
                }
                let mut y_rank = *rank_b as f64;
                let mut sum = 0.0f64;
                let mut max_v = f64::NEG_INFINITY;
                let mut sum_p = 0.0f64;
                let p = *p_norm as f64;
                for (j, &h) in out.iter().enumerate() {
                    let hf = h as f64;
                    y_rank += hf * rank_w[j] as f64;
                    sum += hf;
                    if hf > max_v {
                        max_v = hf;
                    }
                    sum_p += hf.abs().powf(p);
                }
                let mu = sum / n;
                let mut var = 0.0f64;
                for &h in out.iter() {
                    let d = h as f64 - mu;
                    var += d * d;
                }
                let sigma = (var / n).sqrt().max(0.0026);
                let p_norm_stat = (sum_p / n).powf(1.0 / p);
                let y_pool = mu * reducer_w[0] as f64
                    + sigma * reducer_w[1] as f64
                    + max_v * reducer_w[2] as f64
                    + p_norm_stat * reducer_w[3] as f64
                    + *reducer_b as f64;
                let alpha = {
                    let xc = (*alpha_logit as f64).clamp(-20.0, 20.0);
                    1.0 / (1.0 + (-xc).exp())
                };
                alpha * y_rank + (1.0 - alpha) * y_pool
            } else if out.len() == 1 {
                out[0] as f64
            } else {
                out[0] as f64
            }
        }
        Err(_) => f64::NAN,
    }
}

struct Args {
    bake: PathBuf,
    parquet: PathBuf,
    target_column: String,
    output: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut bake: Option<PathBuf> = None;
    let mut parquet: Option<PathBuf> = None;
    let mut target_column: String = "human_score".to_string();
    let mut output: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bake" => {
                bake = Some(PathBuf::from(args.next().ok_or("--bake requires <path>")?));
            }
            "--parquet" => {
                parquet = Some(PathBuf::from(
                    args.next().ok_or("--parquet requires <path>")?,
                ));
            }
            "--target-column" => {
                target_column = args.next().ok_or("--target-column requires <name>")?;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().ok_or("--output requires <path>")?,
                ));
            }
            "-h" | "--help" => {
                eprintln!(
                    "dump_raw_predictions --bake <path> --parquet <path> \
[--target-column human_score] [--output <path.tsv>]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    let bake = bake.ok_or("--bake required")?;
    let parquet = parquet.ok_or("--parquet required")?;
    Ok(Args {
        bake,
        parquet,
        target_column,
        output,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("dump_raw_predictions: {e}");
            return ExitCode::from(2);
        }
    };
    let bake_bytes = match std::fs::read(&args.bake) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read bake: {e}");
            return ExitCode::from(1);
        }
    };
    let model = match Model::from_bytes(&bake_bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse bake: {e:?}");
            return ExitCode::from(1);
        }
    };
    let n_inputs = model.n_inputs();
    let has_transforms = model.has_nontrivial_feature_transforms();
    let per_sample_alpha_head = extract_per_sample_alpha_head(&model);
    let hybrid_head = extract_hybrid_head(&model);
    eprintln!(
        "dump_raw_predictions: bake n_inputs={n_inputs} has_transforms={has_transforms} \
psalpha={} hybrid={}",
        per_sample_alpha_head.is_some(),
        hybrid_head.is_some()
    );

    let g = match parquet_loader::load_parquet(&args.parquet, "corpus", &args.target_column, 1.0) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("load parquet: {e}");
            return ExitCode::from(1);
        }
    };

    let mut predictor = Predictor::new(&model);
    let mut scratch = vec![0.0f32; n_inputs];

    let mut out: Box<dyn Write> = match &args.output {
        Some(p) => match File::create(p) {
            Ok(f) => Box::new(f),
            Err(e) => {
                eprintln!("create output {}: {e}", p.display());
                return ExitCode::from(1);
            }
        },
        None => Box::new(std::io::stdout()),
    };
    let _ = writeln!(out, "target\traw");
    for (target, row) in g.human_scores.iter().zip(g.feature_rows.iter()) {
        let raw = score_row(
            &mut predictor,
            has_transforms,
            per_sample_alpha_head.as_ref(),
            hybrid_head.as_ref(),
            &mut scratch,
            row,
        );
        let _ = writeln!(out, "{target}\t{raw}");
    }
    eprintln!("dump_raw_predictions: {} rows", g.human_scores.len());
    ExitCode::SUCCESS
}
