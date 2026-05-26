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

// DEDUP-M (2026-05-26): per-row dispatch + extract_per_sample_alpha_head /
// extract_hybrid_head / extract_tanh_output_head_scale moved to
// `zensim_validate::bake_runtime`. The EXP-CROSS-CODEC-V11-E per-codec
// affine step (alpha + beta * y) is unique to this bin and stays here.
use zensim_validate::bake_runtime::{
    HybridHeadDispatch, PerSampleAlphaHeadDispatch, extract_hybrid_head,
    extract_per_sample_alpha_head, extract_tanh_output_head_scale, score_with_bake_alloc,
};

type PerCodecAffine = Option<(f32, f32)>;

/// Parse the `zentrain.per_codec_calibration` payload and return the
/// `(alpha, beta)` for the given codec name (case-insensitive,
/// alias-aware). Returns `None` when the metadata is absent, the
/// codec is unknown, or the payload is malformed.
fn extract_per_codec_affine(model: &Model, codec_hint: Option<&str>) -> PerCodecAffine {
    let hint = codec_hint?;
    let lower = hint.to_ascii_lowercase();
    let canon: &str = match lower.as_str() {
        "jpeg" | "jpg" | "zenjpeg" | "mozjpeg" | "libjpeg" => "jpeg",
        "webp" | "zenwebp" => "webp",
        "avif" | "zenavif" => "avif",
        "jxl" | "zenjxl" | "jpegxl" | "jpeg-xl" => "jxl",
        "png" | "zenpng" => "png",
        other => other,
    };
    let md = model.metadata();
    let entry = md.get("zentrain.per_codec_calibration")?;
    let payload = entry.value;
    if payload.len() < 4 {
        return None;
    }
    let n_codecs = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let mut off = 4usize;
    for _ in 0..n_codecs {
        if off + 4 > payload.len() {
            return None;
        }
        let name_len = u32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]) as usize;
        off += 4;
        if off + name_len + 8 > payload.len() {
            return None;
        }
        let name = std::str::from_utf8(&payload[off..off + name_len])
            .ok()?
            .to_ascii_lowercase();
        off += name_len;
        let alpha = f32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]);
        off += 4;
        let beta = f32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]);
        off += 4;
        if name == canon && alpha.is_finite() && beta.is_finite() && beta > 0.0 {
            return Some((alpha, beta));
        }
    }
    None
}

// DEDUP-M (2026-05-26): extract_per_sample_alpha_head, extract_hybrid_head,
// extract_tanh_output_head_scale now imported from bake_runtime above.

fn apply_post(raw: f64, mode: &str) -> f64 {
    if raw.is_nan() {
        return f64::NAN;
    }
    match mode {
        "raw" => raw,
        // EXP-CROSS-CODEC-V10 (2026-05-20): `extrapolate` returns the
        // (post-spline, post-pin, post-α-mix) value WITHOUT clamping
        // to [0, 100]. Identical to `raw` from the perspective of the
        // post-processing branch but kept as a separate name for clarity:
        // V10 callers ask for `extrapolate` to make the no-clamp policy
        // explicit at the call-site.
        "extrapolate" => raw,
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

/// DEDUP-M (2026-05-26): delegates head/pin/spline dispatch to the shared
/// `score_with_bake_alloc`, then applies the EXP-CROSS-CODEC-V11-E
/// per-codec affine (unique to this bin). Bit-exact f32 ±1e-6.
#[allow(clippy::too_many_arguments)]
fn score_with_bake(
    predictor: &mut Predictor<'_>,
    has_transforms: bool,
    psa: Option<&PerSampleAlphaHeadDispatch>,
    hyb: Option<&HybridHeadDispatch>,
    tanh_pin_scale: Option<f64>,
    output_spline: Option<&zensim_validate::output_calibration_spline::OutputCalibrationSpline>,
    per_codec_affine: PerCodecAffine,
    f32_scratch: &mut [f32],
    features_row: &[f32],
) -> f64 {
    // Replicate the per-row pre-fill (this bin's input is &[f32] not &[f64];
    // the shared helper takes &[f64], so widen one row at a time).
    let n_inputs = f32_scratch.len();
    let take = n_inputs.min(features_row.len());
    // Use the shared score_with_bake_alloc which allocates its own
    // f32 buffer; here we widen the source to f64 in-place once.
    let row_f64: Vec<f64> = features_row[..take].iter().map(|&v| v as f64).collect();
    let y_after_spline = score_with_bake_alloc(
        predictor,
        has_transforms,
        psa,
        hyb,
        tanh_pin_scale,
        output_spline,
        n_inputs,
        &row_f64,
    );
    // Suppress unused-scratch warning — kept in the signature for
    // call-site compatibility with the pre-DEDUP-M plumbing in main.
    let _ = f32_scratch;
    // EXP-CROSS-CODEC-V11-E (2026-05-20): per-codec post-spline affine.
    if let Some((alpha, beta)) = per_codec_affine
        && !y_after_spline.is_nan()
    {
        return (alpha as f64) + (beta as f64) * y_after_spline;
    }
    y_after_spline
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
        return Err(format!(
            "{path:?}: header too short ({} bytes)",
            bytes.len()
        ));
    }
    let n_features = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let n_rows = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let expected_floats = n_rows
        .checked_mul(n_features)
        .ok_or_else(|| format!("{path:?}: n_rows*n_features overflow ({n_rows} * {n_features})"))?;
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
    let mut codec_hint: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--codec" => {
                codec_hint = match args.next() {
                    Some(v) => Some(v),
                    None => {
                        eprintln!("--codec requires a value");
                        return ExitCode::FAILURE;
                    }
                };
            }
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
    let output_spline = zensim_validate::output_calibration_spline::extract(&model);
    let per_codec_affine = extract_per_codec_affine(&model, codec_hint.as_deref());

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
            output_spline.as_ref(),
            per_codec_affine,
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
