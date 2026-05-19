//! Score pre-extracted feature rows against an arbitrary ZNPR v3 bake.
//!
//! This is the feature-cache fast path for `cross_codec_consistency.py`
//! (EVAL-ACCEL 2026-05-19). Instead of decoding images and recomputing
//! features per `measure(q)` call, the script reads the pre-extracted
//! 372-feature parquet sidecars at
//! `/mnt/v/zen/picker-training/2026-05-19/butter/<codec>.parquet`, packs
//! the relevant rows into a tiny binary blob, and shells to this binary
//! to get the scores. Skips the ~5-15 s per call that
//! `score_pair_with_bake` spent on image decode + feature extract.
//!
//! Wire format (input file via `--features-file <path>`):
//!     u32 LE n_features
//!     u32 LE n_rows
//!     f32 LE feature_matrix[n_rows][n_features]  (row-major)
//!
//! Smaller fast path (`--features <space-sep floats>`): a single row of
//! features as a CLI arg, identical semantics to a 1-row input file.
//!
//! Output: one score per row, one `%.6f` per line on stdout.
//!
//! Honors the same `--bake-post {raw|clamp|mapped[:a,b]}` semantics as
//! `score_pair_with_bake`, plus the full V_24 dispatch path
//! (per-sample-α head, hybrid head, tanh output pin) so the produced
//! score is bit-exact with the slow path on the same feature row.

use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::{Model, Predictor};

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
    Some((w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm))
}

fn extract_tanh_output_head_scale(model: &Model) -> Option<f64> {
    let md = model.metadata();
    let entry = md.get("zentrain.tanh_output_head")?;
    if entry.value.len() != 4 {
        return None;
    }
    let scale = f32::from_le_bytes([
        entry.value[0],
        entry.value[1],
        entry.value[2],
        entry.value[3],
    ]) as f64;
    if scale.is_finite() && scale > 0.0 {
        Some(scale)
    } else {
        None
    }
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

fn apply_post(raw: f64, mode: &str) -> f64 {
    if raw.is_nan() {
        return f64::NAN;
    }
    match mode {
        "raw" => raw,
        "clamp" => raw.clamp(0.0, 100.0),
        m if m.starts_with("mapped") => {
            let (a, b) = if let Some(rest) = m.strip_prefix("mapped:") {
                let mut it = rest.splitn(2, ',');
                let a: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(18.0);
                let b: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.7);
                (a, b)
            } else {
                (18.0, 0.7)
            };
            let d = raw.max(0.0);
            (100.0 - a * d.powf(b)).clamp(0.0, 100.0)
        }
        _ => raw.clamp(0.0, 100.0),
    }
}

#[allow(clippy::too_many_arguments)]
fn score_with_bake(
    predictor: &mut Predictor<'_>,
    has_transforms: bool,
    psa: Option<&PerSampleAlphaHeadDispatch>,
    hyb: Option<&HybridHeadDispatch>,
    tanh_pin_scale: Option<f64>,
    f32_scratch: &mut [f32],
    features_row: &[f32],
) -> f64 {
    let n_inputs = f32_scratch.len();
    let take = n_inputs.min(features_row.len());
    f32_scratch[..take].copy_from_slice(&features_row[..take]);
    for f in &mut f32_scratch[take..] {
        *f = 0.0;
    }
    let result = if has_transforms {
        predictor.predict_transformed(f32_scratch)
    } else {
        predictor.predict(f32_scratch)
    };
    let y_pre = match result {
        Ok(out) => {
            if let Some((w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm)) = psa {
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
            } else if let Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm)) = hyb {
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
            } else {
                out.first().copied().map(|v| v as f64).unwrap_or(f64::NAN)
            }
        }
        Err(_) => f64::NAN,
    };
    if let Some(scale) = tanh_pin_scale {
        if !y_pre.is_nan() {
            let xc = (y_pre / scale).clamp(-30.0, 30.0);
            let s = 1.0 / (1.0 + (-xc).exp());
            return 100.0 * s;
        }
    }
    y_pre
}

fn parse_features_arg(s: &str) -> Result<(usize, usize, Vec<f32>), String> {
    let vals: Result<Vec<f32>, _> = s.split_whitespace().map(|t| t.parse::<f32>()).collect();
    let vals = vals.map_err(|e| format!("--features parse: {e}"))?;
    if vals.is_empty() {
        return Err("--features is empty".into());
    }
    Ok((vals.len(), 1, vals))
}

fn read_features_file(path: &PathBuf) -> Result<(usize, usize, Vec<f32>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    if bytes.len() < 8 {
        return Err(format!("{path:?}: header too short ({} bytes)", bytes.len()));
    }
    let n_features = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let n_rows = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let expected_floats = n_rows.checked_mul(n_features).ok_or_else(|| {
        format!("{path:?}: n_rows*n_features overflow ({n_rows} * {n_features})")
    })?;
    let expected_bytes = 8 + expected_floats * 4;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "{path:?}: payload size mismatch: header says {n_rows} rows × {n_features} features = {expected_bytes} bytes, got {}",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(expected_floats);
    for i in 0..expected_floats {
        let off = 8 + i * 4;
        out.push(f32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]));
    }
    Ok((n_features, n_rows, out))
}

fn print_usage() {
    eprintln!(
        "predict_features_with_bake — bake forward pass over pre-extracted features\n\
\n\
USAGE:\n\
    predict_features_with_bake --bake <path> [--bake-post raw|clamp|mapped[:a,b]] \\\n\
        (--features 'f0 f1 f2 ...' | --features-file <path>)\n\
\n\
The --features-file format is u32 LE n_features, u32 LE n_rows, then\n\
n_rows*n_features f32 LE features (row-major). Output is one\n\
'%.6f'-formatted score per row, one per line, on stdout.\n"
    );
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut bake: Option<PathBuf> = None;
    let mut bake_post: String = "clamp".to_string();
    let mut features_arg: Option<String> = None;
    let mut features_file: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => {
                let v = match args.next() {
                    Some(v) => v,
                    None => {
                        eprintln!("--bake requires a value");
                        return ExitCode::FAILURE;
                    }
                };
                bake = Some(v.into());
            }
            "--bake-post" => {
                bake_post = match args.next() {
                    Some(v) => v,
                    None => {
                        eprintln!("--bake-post requires a value");
                        return ExitCode::FAILURE;
                    }
                };
            }
            "--features" => {
                features_arg = args.next();
                if features_arg.is_none() {
                    eprintln!("--features requires a value");
                    return ExitCode::FAILURE;
                }
            }
            "--features-file" => {
                let v = match args.next() {
                    Some(v) => v,
                    None => {
                        eprintln!("--features-file requires a value");
                        return ExitCode::FAILURE;
                    }
                };
                features_file = Some(v.into());
            }
            "-h" | "--help" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown arg: {other}");
                print_usage();
                return ExitCode::FAILURE;
            }
        }
    }
    let bake = match bake {
        Some(b) => b,
        None => {
            eprintln!("--bake is REQUIRED");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let (n_features_in, n_rows, feature_buf) = match (features_arg, features_file) {
        (Some(s), None) => match parse_features_arg(&s) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        (None, Some(p)) => match read_features_file(&p) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        (Some(_), Some(_)) => {
            eprintln!("specify --features OR --features-file, not both");
            return ExitCode::FAILURE;
        }
        (None, None) => {
            eprintln!("one of --features or --features-file is REQUIRED");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    let bake_bytes = match std::fs::read(&bake) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read bake {bake:?}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let model = match Model::from_bytes(&bake_bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse ZNPR bake: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let n_inputs = model.n_inputs();
    let has_transforms = model.has_nontrivial_feature_transforms();
    let psa = extract_per_sample_alpha_head(&model);
    let hyb = extract_hybrid_head(&model);
    let tanh_pin_scale = extract_tanh_output_head_scale(&model);

    let mut predictor = Predictor::new(&model);
    let mut scratch = vec![0.0f32; n_inputs];

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    use std::io::Write as _;

    for row_idx in 0..n_rows {
        let start = row_idx * n_features_in;
        let end = start + n_features_in;
        let row = &feature_buf[start..end];
        let raw = score_with_bake(
            &mut predictor,
            has_transforms,
            psa.as_ref(),
            hyb.as_ref(),
            tanh_pin_scale,
            &mut scratch,
            row,
        );
        let score = apply_post(raw, &bake_post);
        if writeln!(out, "{score:.6}").is_err() {
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
