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

// DEDUP-M (2026-05-26): per-row dispatch + extract_* helpers moved to
// `zensim_validate::bake_runtime`. Bit-exact f32 ±1e-6.
use zensim_validate::bake_runtime::{
    extract_hybrid_head, extract_per_sample_alpha_head, extract_tanh_output_head_scale,
    score_with_bake_alloc,
};

fn apply_post(raw: f64, mode: &str) -> f64 {
    if raw.is_nan() {
        return f64::NAN;
    }
    match mode {
        "raw" => raw,
        // EXP-CROSS-CODEC-V10 (2026-05-20): explicit no-clamp mode.
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

// DEDUP-M (2026-05-26): `score_with_bake` is now `score_with_bake_alloc`
// imported from `zensim_validate::bake_runtime`. Bit-exact f32 ±1e-6.

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut bake: Option<PathBuf> = None;
    let mut bake_post: String = "clamp".to_string();
    let mut ref_path: Option<PathBuf> = None;
    let mut dist_path: Option<PathBuf> = None;
    let mut dump_features_to: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => bake = Some(args.next().expect("--bake VALUE").into()),
            "--bake-post" => bake_post = args.next().expect("--bake-post VALUE"),
            "--ref" => ref_path = Some(args.next().expect("--ref VALUE").into()),
            "--dist" => dist_path = Some(args.next().expect("--dist VALUE").into()),
            "--dump-features-to" => {
                dump_features_to = Some(args.next().expect("--dump-features-to VALUE").into());
            }
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

    // EVAL-ACCEL bit-exact verification helper: optionally dump the
    // computed feature vector in the predict_features_with_bake wire
    // format so the two binaries can be cross-checked on the exact
    // same numeric input.
    if let Some(path) = &dump_features_to {
        let n_features = features.len();
        let mut buf = Vec::with_capacity(8 + n_features * 4);
        buf.extend_from_slice(&(n_features as u32).to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_rows
        for &v in &features {
            buf.extend_from_slice(&(v as f32).to_le_bytes());
        }
        std::fs::write(path, &buf).expect("write --dump-features-to");
        eprintln!(
            "dumped {} features ({} bytes) to {}",
            n_features,
            buf.len(),
            path.display()
        );
    }

    // Load bake.
    let bake_bytes = std::fs::read(&bake).expect("read bake");
    let model = Model::from_bytes(&bake_bytes).expect("parse ZNPR bake");
    let n_inputs = model.caller_input_width();
    let has_transforms = model.has_nontrivial_feature_transforms();
    let psa = extract_per_sample_alpha_head(&model);
    let hyb = extract_hybrid_head(&model);
    let tanh_pin_scale = extract_tanh_output_head_scale(&model);
    let output_spline = zensim_validate::output_calibration_spline::extract(&model);

    let mut predictor = Predictor::new(&model);
    let raw = score_with_bake_alloc(
        &mut predictor,
        has_transforms,
        psa.as_ref(),
        hyb.as_ref(),
        tanh_pin_scale,
        output_spline.as_ref(),
        n_inputs,
        &features,
    );
    let score = apply_post(raw, &bake_post);
    println!("{:.6}", score);
    ExitCode::SUCCESS
}
