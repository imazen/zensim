//! ensemble_score_rows — dump per-row bake scores from a validation parquet.
//!
//! Used by the EXP-ENSEMBLE-V05 experiment to score every row in each val
//! parquet through BOTH the balanced and compression ship bakes so the
//! Python evaluation script can route per-pair and compute the full
//! Mohammadi panel for the ensemble.
//!
//! Output is a tab-separated stream to stdout (or `--output <path>`):
//!
//!     idx<TAB>human<TAB>score
//!
//! One header line + one row per parquet pair, ordered as in the input.
//!
//! Dispatch matches `bake_verdict::score_row` bit-for-bit (per-sample-α
//! head and hybrid-head metadata are honored).

use std::path::PathBuf;

use zenpredict::{Model, Predictor};
use zensim_validate::parquet_loader;

const PER_SAMPLE_ALPHA_KEY: &str = "zentrain.per_sample_alpha_head";
const HYBRID_HEAD_KEY: &str = "zentrain.hybrid_head";
const TANH_OUTPUT_HEAD_KEY: &str = "zentrain.tanh_output_head";
const POOL_STD_FLOOR: f64 = 0.0026;

/// EXP-CROSS-CODEC-V4 tanh-pin scale extractor. Returns `Some(scale)` if
/// the bake declares `zentrain.tanh_output_head` with a positive f32.
fn extract_tanh_output_head_scale(model: &Model) -> Option<f64> {
    let md = model.metadata();
    let entry = md.get(TANH_OUTPUT_HEAD_KEY)?;
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

type PerSampleAlpha = (Vec<f32>, f32, Vec<f32>, f32, [f32; 4], f32, f32);
type HybridHead = (Vec<f32>, f32, f32, [f32; 4], f32, f32);

fn extract_per_sample_alpha(model: &Model) -> Option<PerSampleAlpha> {
    let md = model.metadata();
    let entry = md.get(PER_SAMPLE_ALPHA_KEY)?;
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

fn extract_hybrid_head(model: &Model) -> Option<HybridHead> {
    let md = model.metadata();
    let entry = md.get(HYBRID_HEAD_KEY)?;
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
    per_sample: Option<&PerSampleAlpha>,
    hybrid: Option<&HybridHead>,
    tanh_pin_scale: Option<f64>,
    f32_features: &mut [f32],
    row: &[f64],
) -> f64 {
    let n = f32_features.len();
    let take = n.min(row.len());
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
    let out = match result {
        Ok(o) => o,
        Err(_) => return f64::NAN,
    };
    // Wrap any non-NaN raw output with the optional tanh-pin.
    let pin = |y_pre: f64| -> f64 {
        if let Some(scale) = tanh_pin_scale {
            if !y_pre.is_nan() {
                let xc = (y_pre / scale).clamp(-30.0, 30.0);
                let s = 1.0 / (1.0 + (-xc).exp());
                return 100.0 * s;
            }
        }
        y_pre
    };
    if let Some((w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm)) = per_sample {
        let nh = out.len() as f64;
        if nh <= 0.0 || out.len() != rank_w.len() || out.len() != w_alpha.len() {
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
        let mu = sum / nh;
        let mut var = 0.0f64;
        for &h in out.iter() {
            let d = h as f64 - mu;
            var += d * d;
        }
        let sigma = (var / nh).sqrt().max(POOL_STD_FLOOR);
        let p_norm_stat = (sum_p / nh).powf(1.0 / p);
        let y_pool = mu * reducer_w[0] as f64
            + sigma * reducer_w[1] as f64
            + max_v * reducer_w[2] as f64
            + p_norm_stat * reducer_w[3] as f64
            + *reducer_b as f64;
        let alpha = {
            let xc = alpha_logit.clamp(-20.0, 20.0);
            1.0 / (1.0 + (-xc).exp())
        };
        return pin(alpha * y_rank + (1.0 - alpha) * y_pool);
    }
    if let Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm)) = hybrid {
        let nh = out.len() as f64;
        if nh <= 0.0 || out.len() != rank_w.len() {
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
        let mu = sum / nh;
        let mut var = 0.0f64;
        for &h in out.iter() {
            let d = h as f64 - mu;
            var += d * d;
        }
        let sigma = (var / nh).sqrt().max(POOL_STD_FLOOR);
        let p_norm_stat = (sum_p / nh).powf(1.0 / p);
        let y_pool = mu * reducer_w[0] as f64
            + sigma * reducer_w[1] as f64
            + max_v * reducer_w[2] as f64
            + p_norm_stat * reducer_w[3] as f64
            + *reducer_b as f64;
        let alpha = {
            let xc = (*alpha_logit as f64).clamp(-20.0, 20.0);
            1.0 / (1.0 + (-xc).exp())
        };
        return pin(alpha * y_rank + (1.0 - alpha) * y_pool);
    }
    pin(out.first().copied().map(|v| v as f64).unwrap_or(f64::NAN))
}

fn print_usage() {
    eprintln!(
        "ensemble_score_rows — per-row bake scoring for EXP-ENSEMBLE-V05\n\
\n\
USAGE:\n\
    ensemble_score_rows --bake <path> --parquet <path> [--output <path>]\n\
\n\
OUTPUT (TSV, stdout or --output):\n\
    idx\\thuman\\tscore\n"
    );
}

fn main() -> Result<(), String> {
    let mut bake: Option<PathBuf> = None;
    let mut parquet: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => bake = Some(PathBuf::from(args.next().ok_or("--bake needs value")?)),
            "--parquet" => {
                parquet = Some(PathBuf::from(args.next().ok_or("--parquet needs value")?))
            }
            "--output" => output = Some(PathBuf::from(args.next().ok_or("--output needs value")?)),
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    let bake = bake.ok_or("--bake required")?;
    let parquet = parquet.ok_or("--parquet required")?;
    let bytes = std::fs::read(&bake).map_err(|e| format!("read {bake:?}: {e}"))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("model parse: {e}"))?;
    let n_inputs = model.n_inputs();
    let has_transforms = model.has_nontrivial_feature_transforms();
    let per_sample = extract_per_sample_alpha(&model);
    let hybrid = if per_sample.is_some() {
        None
    } else {
        extract_hybrid_head(&model)
    };
    let tanh_pin_scale = extract_tanh_output_head_scale(&model);
    let g = parquet_loader::load_parquet(&parquet, "rows", "human_score", 1.0)?;
    let humans = g.human_scores;
    let mut predictor = Predictor::new(&model);
    let mut scratch = vec![0.0f32; n_inputs];

    let mut writer: Box<dyn std::io::Write> = match output {
        Some(p) => Box::new(std::fs::File::create(&p).map_err(|e| format!("create {p:?}: {e}"))?),
        None => Box::new(std::io::stdout()),
    };
    writeln!(writer, "idx\thuman\tscore").map_err(|e| format!("write header: {e}"))?;
    for (i, row) in g.feature_rows.iter().enumerate() {
        let score = score_row(
            &mut predictor,
            has_transforms,
            per_sample.as_ref(),
            hybrid.as_ref(),
            tanh_pin_scale,
            &mut scratch,
            row,
        );
        writeln!(writer, "{}\t{:.6}\t{:.6}", i, humans[i], score)
            .map_err(|e| format!("write row {i}: {e}"))?;
    }
    Ok(())
}
