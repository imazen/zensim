//! Explicit-vector adapter to the canonical source-clustered Spearman owner.
use anyhow::{Result, ensure};
use clap::Parser;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::PathBuf, process::ExitCode};
use zenstats::resample::{BootstrapConfig, Design, spearman_interval};

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    input: PathBuf,
    #[arg(long, default_value_t = 10_000)]
    resamples: usize,
    #[arg(long, default_value_t = 20261017)]
    seed: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    predicted: Vec<f64>,
    target: Vec<f64>,
    reference: Vec<u32>,
}
fn evaluate(input: &Input, resamples: usize, seed: u64) -> Result<Value> {
    let n = input.predicted.len();
    ensure!(
        n >= 3 && input.target.len() == n && input.reference.len() == n,
        "aligned finite rows required"
    );
    ensure!(
        input
            .predicted
            .iter()
            .chain(&input.target)
            .all(|v| v.is_finite()),
        "nonfinite row; no dropping"
    );
    let units = input.reference.iter().collect::<BTreeSet<_>>().len();
    ensure!(units >= 2, "at least two source clusters required");
    let config = BootstrapConfig {
        resamples,
        seed,
        level: 0.95,
    };
    let result = spearman_interval(
        &input.predicted,
        &input.target,
        &Design::OneWay(&input.reference),
        &config,
    )?;
    Ok(
        json!({"signed":{"estimate":result.estimate,"lo":result.lo,"hi":result.hi,"se":result.se,"n_finite":result.n_finite},"srocc_ci_method":{"design":"one-way: reference","units":units,"resamples":resamples,"seed":seed,"level":0.95,"percentile_rule":"Hyndman-Fan type 7","owner":"zenstats::resample::spearman_interval"},"limitation":"Descriptive interval; small source counts under-cover. PLCC and KROCC intervals not computed by this adapter."}),
    )
}
pub(super) fn main() -> ExitCode {
    let cli = Cli::parse_from(std::env::args().skip(1));
    let run = || -> Result<Value> {
        let input = serde_json::from_slice(&fs::read(cli.input)?)?;
        evaluate(&input, cli.resamples, cli.seed)
    };
    match run() {
        Ok(value) => {
            println!("{value}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("panel rank-interval: {error:#}");
            ExitCode::from(2)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owner_parity_and_source_design() {
        let input = Input {
            predicted: vec![1., 4., 2., 7., 5., 9.],
            target: vec![2., 3., 1., 8., 6., 7.],
            reference: vec![0, 0, 1, 1, 2, 2],
        };
        let got = evaluate(&input, 128, 72).unwrap();
        let expected = spearman_interval(
            &input.predicted,
            &input.target,
            &Design::OneWay(&input.reference),
            &BootstrapConfig {
                resamples: 128,
                seed: 72,
                level: 0.95,
            },
        )
        .unwrap();
        assert_eq!(got["signed"]["lo"], expected.lo);
        assert_eq!(got["signed"]["hi"], expected.hi);
        assert_eq!(got["signed"]["estimate"], expected.estimate);
        assert_eq!(got["srocc_ci_method"]["units"], 3);
    }
    #[test]
    fn missing_nonfinite_and_single_cluster_refuse() {
        let mut input = Input {
            predicted: vec![1., 2., 3.],
            target: vec![1., 3., 2.],
            reference: vec![0, 1],
        };
        assert!(evaluate(&input, 128, 72).is_err());
        input.reference = vec![0, 0, 0];
        assert!(evaluate(&input, 128, 72).is_err());
        input.reference = vec![0, 1, 2];
        input.predicted[0] = f64::NAN;
        assert!(evaluate(&input, 128, 72).is_err());
    }
}
