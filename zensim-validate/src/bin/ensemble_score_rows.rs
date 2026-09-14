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
//! Repeating `--bake` serves a uniform ensemble through the public BakeScorer.
//! Dispatch matches `bake_verdict::score_row` bit-for-bit (per-sample-α
//! head and hybrid-head metadata are honored).

use std::path::PathBuf;

use zenpredict::Model;
use zensim::BakeScorer;
use zensim_validate::parquet_loader;

fn print_usage() {
    eprintln!(
        "ensemble_score_rows — per-row bake scoring for EXP-ENSEMBLE-V05\n\
\n\
USAGE:\n\
    ensemble_score_rows --bake <path> [--bake <path> ...] [--weights <w,...>] --parquet <path> [--output <path>]\n\
\n\
OUTPUT (TSV, stdout or --output):\n\
    idx\\thuman\\tscore\n"
    );
}

fn main() -> Result<(), String> {
    let mut bakes: Vec<PathBuf> = Vec::new();
    let mut parquet: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut weights: Option<Vec<f64>> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => bakes.push(PathBuf::from(args.next().ok_or("--bake needs value")?)),
            "--weights" => {
                weights = Some(
                    args.next()
                        .ok_or("--weights needs value")?
                        .split(',')
                        .map(|x| x.parse::<f64>().map_err(|e| e.to_string()))
                        .collect::<Result<_, _>>()?,
                );
            }
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
    if bakes.is_empty() {
        return Err("--bake required".into());
    }
    let parquet = parquet.ok_or("--parquet required")?;
    let bytes: Vec<Vec<u8>> = bakes
        .iter()
        .map(|bake| std::fs::read(bake).map_err(|e| format!("read {bake:?}: {e}")))
        .collect::<Result<_, _>>()?;
    let models: Vec<Model> = bytes
        .iter()
        .map(|b| Model::from_bytes(b).map_err(|e| format!("model parse: {e}")))
        .collect::<Result<_, _>>()?;
    let mut scorer =
        BakeScorer::ensemble(&models, weights.as_deref()).map_err(|e| e.to_string())?;
    let g = parquet_loader::load_parquet(&parquet, "rows", "human_score", 1.0)?;
    let humans = g.human_scores;
    let mut writer: Box<dyn std::io::Write> = match output {
        Some(p) => Box::new(std::fs::File::create(&p).map_err(|e| format!("create {p:?}: {e}"))?),
        None => Box::new(std::io::stdout()),
    };
    writeln!(writer, "idx\thuman\tscore").map_err(|e| format!("write header: {e}"))?;
    for (i, row) in g.feature_rows.iter().enumerate() {
        let score = scorer
            .score_features(row, 0, 0, None)
            .map_err(|e| e.to_string())?;
        writeln!(writer, "{}\t{:.6}\t{:.6}", i, humans[i], score)
            .map_err(|e| format!("write row {i}: {e}"))?;
    }
    Ok(())
}
