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
use zensim_validate::bake_runtime::{
    CallerGather, extract_hybrid_head, extract_per_sample_alpha_head as extract_per_sample_alpha,
    extract_tanh_output_head_scale, score_row as score_row_shared,
};
use zensim_validate::parquet_loader;

// DEDUP-M (2026-05-26): extract_tanh_output_head_scale, extract_per_sample_alpha,
// extract_hybrid_head, and the per-row dispatch logic were moved to
// `zensim_validate::bake_runtime`. The local `score_row` here is a thin
// adapter to the shared `score_row_shared` that passes `output_spline = None`
// (ensemble_score_rows doesn't carry EXP-CROSS-CODEC-V9 spline plumbing).
// Bit-exact f32 ±1e-6 on representative inputs.

/// `(w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm)` payload.
type PerSampleHeadPayload = (Vec<f32>, f32, Vec<f32>, f32, [f32; 4], f32, f32);
/// `(rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm)` payload.
type HybridHeadPayload = (Vec<f32>, f32, f32, [f32; 4], f32, f32);

#[allow(clippy::too_many_arguments)]
fn score_row(
    predictor: &mut Predictor<'_>,
    has_transforms: bool,
    per_sample: Option<&PerSampleHeadPayload>,
    hybrid: Option<&HybridHeadPayload>,
    tanh_pin_scale: Option<f64>,
    gather: &CallerGather,
    f32_features: &mut [f32],
    row: &[f64],
) -> f64 {
    score_row_shared(
        predictor,
        has_transforms,
        per_sample,
        hybrid,
        tanh_pin_scale,
        None,
        gather,
        f32_features,
        row,
    )
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
    let n_inputs = model.caller_input_width();
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
    let gather = CallerGather::for_model(&model);
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
            &gather,
            &mut scratch,
            row,
        );
        writeln!(writer, "{}\t{:.6}\t{:.6}", i, humans[i], score)
            .map_err(|e| format!("write row {i}: {e}"))?;
    }
    Ok(())
}
