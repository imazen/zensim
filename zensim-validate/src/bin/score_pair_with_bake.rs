//! Score one (ref, dist) PNG pair against an arbitrary ZNPR v3 bake.
//!
//! Usage:
//!   score_pair_with_bake --bake PATH [--bake-post raw|clamp|mapped[:A,B]] \
//!                        --ref REF.png --dist DIST.png
//!
//! Prints one float on stdout — the scored zensim value with the
//! specified post-processing applied. Used by cross_codec_consistency.py
//! to binary-search for the q value that matches a target zensim score.

use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::{Model, Predictor};
use zensim::{ZensimConfig, compute_zensim_with_config};

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

fn score_with_bake(
    predictor: &mut Predictor<'_>,
    has_transforms: bool,
    psa: Option<&PerSampleAlphaHeadDispatch>,
    hyb: Option<&HybridHeadDispatch>,
    n_inputs: usize,
    features: &[f64],
) -> f64 {
    let mut buf = vec![0.0f32; n_inputs];
    let take = n_inputs.min(features.len());
    for i in 0..take {
        buf[i] = features[i] as f32;
    }
    let result = if has_transforms {
        predictor.predict_transformed(&mut buf[..])
    } else {
        predictor.predict(&mut buf[..])
    };
    match result {
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
    }
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut bake: Option<PathBuf> = None;
    let mut bake_post: String = "clamp".to_string();
    let mut ref_path: Option<PathBuf> = None;
    let mut dist_path: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => bake = Some(args.next().expect("--bake VALUE").into()),
            "--bake-post" => bake_post = args.next().expect("--bake-post VALUE"),
            "--ref" => ref_path = Some(args.next().expect("--ref VALUE").into()),
            "--dist" => dist_path = Some(args.next().expect("--dist VALUE").into()),
            other => {
                eprintln!("unknown arg: {other}");
                return ExitCode::FAILURE;
            }
        }
    }
    let bake = bake.expect("--bake REQUIRED");
    let ref_path = ref_path.expect("--ref REQUIRED");
    let dist_path = dist_path.expect("--dist REQUIRED");

    // Load images.
    let src = image::open(&ref_path).expect("open ref").to_rgb8();
    let dst = image::open(&dist_path).expect("open dist").to_rgb8();
    let w = src.width() as usize;
    let h = src.height() as usize;
    let src_pixels: Vec<[u8; 3]> = src.pixels().map(|p| p.0).collect();
    let dst_pixels: Vec<[u8; 3]> = dst.pixels().map(|p| p.0).collect();

    // Compute features with both extended + IW pools (372-feat superset).
    let mut config = ZensimConfig::default();
    config.extended_features = true;
    config.compute_iw_features = true;
    let result =
        compute_zensim_with_config(&src_pixels, &dst_pixels, w, h, config).expect("zensim compute");
    let features: Vec<f64> = result.features().to_vec();

    // Load bake.
    let bake_bytes = std::fs::read(&bake).expect("read bake");
    let model = Model::from_bytes(&bake_bytes).expect("parse ZNPR bake");
    let n_inputs = model.n_inputs();
    let has_transforms = model.has_nontrivial_feature_transforms();
    let psa = extract_per_sample_alpha_head(&model);
    let hyb = extract_hybrid_head(&model);

    let mut predictor = Predictor::new(&model);
    let raw = score_with_bake(
        &mut predictor,
        has_transforms,
        psa.as_ref(),
        hyb.as_ref(),
        n_inputs,
        &features,
    );
    let score = apply_post(raw, &bake_post);
    println!("{:.6}", score);
    ExitCode::SUCCESS
}
