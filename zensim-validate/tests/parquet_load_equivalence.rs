//! Verify `load_parquet` produces bit-identical output to a sequential
//! CSV parse of the same source data.
//!
//! Test fixture: the TID feature corpus (3,000 rows x 372 features) at
//! `/mnt/v/zen/zensim-training/2026-05-16/v2/tid_features_iwssim_log_372col.{csv,parquet}`.
//! The parquet is produced by `scripts/convert_csv_to_parquet.py`.
//!
//! Skip semantics: marked `#[ignore]` so the test does NOT run in CI
//! environments that lack the fixtures. To run locally:
//!
//!     # 1. Convert the CSV if not already converted:
//!     python3 scripts/convert_csv_to_parquet.py \
//!         /mnt/v/zen/zensim-training/2026-05-16/v2/tid_features_iwssim_log_372col.csv
//!     # 2. Run the test explicitly:
//!     cargo test -p zensim-validate --test parquet_load_equivalence -- \
//!         --ignored --nocapture
//!
//! See `tid_fixture_present_means_parquet_must_exist` for the
//! fail-loud companion that runs in CI: it panics if the CSV is
//! present but the parquet was not generated (so a developer who
//! drops the CSV into a CI box and forgets to convert it sees the
//! error immediately, not "silently passing tests").

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use zensim_validate::parquet_loader::{OwnedLoadedGroup, load_parquet};

const CSV_PATH: &str =
    "/mnt/v/zen/zensim-training/2026-05-16/v2/tid_features_iwssim_log_372col.csv";
const PARQUET_PATH: &str =
    "/mnt/v/zen/zensim-training/2026-05-16/v2/tid_features_iwssim_log_372col.parquet";
const TARGET_COLUMN: &str = "iwssim_log_norm";
const TARGET_SCALE: f64 = 1.0;

/// Sequential CSV loader, reproduced here to avoid touching
/// `bin/zensim_mlp_train.rs`. Must match its sequential `load_csv`
/// reference path exactly (same UTF-8 / split / parse semantics).
fn load_csv_reference(
    path: &PathBuf,
    name: &str,
    target_column: &str,
    target_scale: f64,
) -> Result<OwnedLoadedGroup, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut rdr = BufReader::new(file);
    let mut header = String::new();
    rdr.read_line(&mut header)
        .map_err(|e| format!("header read {path:?}: {e}"))?;
    let cols: Vec<&str> = header.trim_end().split(',').collect();
    let score_idx = cols
        .iter()
        .position(|&c| c == target_column)
        .ok_or_else(|| format!("{path:?}: missing target column {target_column:?}"))?;
    let f0 = cols
        .iter()
        .position(|&c| c == "f0")
        .ok_or_else(|| format!("{path:?}: missing f0 column"))?;
    let mut n_features = 0usize;
    while f0 + n_features < cols.len() {
        let expected = format!("f{}", n_features);
        if cols[f0 + n_features] != expected {
            break;
        }
        n_features += 1;
    }
    if n_features == 0 {
        return Err(format!("{path:?}: no fN columns found"));
    }
    let mut human_scores = Vec::new();
    let mut feature_rows = Vec::new();
    for (lineno, line) in rdr.lines().enumerate() {
        let line = line.map_err(|e| format!("read line {}: {e}", lineno + 2))?;
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < f0 + n_features {
            return Err(format!(
                "{path:?} line {}: expected ≥{} fields, got {}",
                lineno + 2,
                f0 + n_features,
                fields.len()
            ));
        }
        let score: f64 = fields[score_idx]
            .parse::<f64>()
            .map_err(|e| {
                format!(
                    "{path:?} line {}: bad target column {target_column:?}: {e}",
                    lineno + 2
                )
            })?
            * target_scale;
        let mut row = Vec::with_capacity(n_features);
        for i in 0..n_features {
            row.push(
                fields[f0 + i]
                    .parse::<f64>()
                    .map_err(|e| format!("{path:?} line {}: bad f{i}: {e}", lineno + 2))?,
            );
        }
        human_scores.push(score);
        feature_rows.push(row);
    }
    Ok(OwnedLoadedGroup {
        name: name.to_string(),
        train_w: 0.0,
        val_w: 0.0,
        human_scores,
        feature_rows,
        n_features,
    })
}

/// Bit-identity check between the parquet and CSV loaders.
///
/// Ignored by default — run with `cargo test --ignored` after
/// generating the parquet fixture. Tolerance is 1e-12: parquet
/// stores Float64 losslessly and the CSV path uses stdlib
/// `f64::from_str`, which should reproduce the same f64 bits as the
/// f64 written by pyarrow on conversion (both are round-trip-exact
/// for finite f64s in canonical decimal form).
#[test]
#[ignore = "requires /mnt/v fixtures + scripts/convert_csv_to_parquet.py run first"]
fn parquet_load_matches_csv_load() {
    let csv = PathBuf::from(CSV_PATH);
    let pq = PathBuf::from(PARQUET_PATH);
    assert!(
        csv.exists(),
        "CSV fixture missing at {CSV_PATH}; the equivalence test requires it",
    );
    assert!(
        pq.exists(),
        "Parquet fixture missing at {PARQUET_PATH}; run `python3 scripts/convert_csv_to_parquet.py {CSV_PATH}` first",
    );

    let parq = load_parquet(&pq, "tid_pq", TARGET_COLUMN, TARGET_SCALE)
        .expect("parquet load failed");
    let csv_g = load_csv_reference(&csv, "tid_csv", TARGET_COLUMN, TARGET_SCALE)
        .expect("csv load failed");

    assert_eq!(
        parq.n_features, csv_g.n_features,
        "n_features mismatch: parquet={} csv={}",
        parq.n_features, csv_g.n_features
    );
    assert_eq!(
        parq.human_scores.len(),
        csv_g.human_scores.len(),
        "row count mismatch: parquet={} csv={}",
        parq.human_scores.len(),
        csv_g.human_scores.len()
    );

    for (i, (a, b)) in parq
        .human_scores
        .iter()
        .zip(csv_g.human_scores.iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-12,
            "row {i} target mismatch: parquet={a} csv={b} (delta {})",
            (a - b).abs()
        );
    }
    for (i, (a_row, b_row)) in parq
        .feature_rows
        .iter()
        .zip(csv_g.feature_rows.iter())
        .enumerate()
    {
        assert_eq!(
            a_row.len(),
            b_row.len(),
            "row {i} length mismatch: parquet={} csv={}",
            a_row.len(),
            b_row.len()
        );
        for (j, (a, b)) in a_row.iter().zip(b_row.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "row {i} feature {j} mismatch: parquet={a} csv={b} (delta {})",
                (a - b).abs()
            );
        }
    }
}

/// Guardrail that runs in CI: if the CSV fixture IS present but the
/// parquet was not generated, fail loudly. Prevents "silent skips"
/// when a developer drops the CSV onto a box and forgets to convert.
///
/// No fixtures → no failure (returns Ok). Both fixtures present →
/// returns Ok (the `parquet_load_matches_csv_load` test handles
/// equivalence). Only CSV present → panic with explicit instructions.
#[test]
fn tid_fixture_present_means_parquet_must_exist() {
    let csv = PathBuf::from(CSV_PATH);
    let pq = PathBuf::from(PARQUET_PATH);
    if csv.exists() && !pq.exists() {
        panic!(
            "TID CSV fixture is present at {CSV_PATH} but the parquet is not at {PARQUET_PATH}. \
             Run `python3 scripts/convert_csv_to_parquet.py {CSV_PATH}` before invoking \
             `cargo test --ignored parquet_load_matches_csv_load`.",
        );
    }
}
