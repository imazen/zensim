//! `panel` — the canonical IQA correlation-statistics entry point.
//!
//! This is THE tool for computing the full Mohammadi 2025 statistical
//! panel (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE, plus the 4-parameter
//! logistic rescale that PLCC / Z-RMSE depend on) on an arbitrary table
//! of `(predicted, target)` pairs — i.e. the case that does NOT involve
//! a bake or a canonical corpus (use `bake_verdict` for that).
//!
//! ## Why this binary exists
//!
//! Across the zen workspace there were ~14 scattered Python
//! reimplementations of SROCC / PLCC / KROCC / OR / PWRC / Z-RMSE and
//! the 4-param logistic fit (see the dedup ledger
//! `benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md` Tier-1 #2 and
//! `benchmarks/sweep_training_script_dedup_2026-05-26.md` Class 2). Each
//! hand-rolled `spearman` carried its own tie-handling and NaN-drop
//! policy; a metric verdict is only as trustworthy as its stat impl, and
//! CLAUDE.md mandates the *full* Mohammadi panel everywhere. This binary
//! collapses all of those onto the single canonical Rust home,
//! `zensim_validate::panel`, so every consumer computes bit-identical
//! stats.
//!
//! ## Zero new stat math
//!
//! Every number this tool prints comes from a direct call into
//! `panel.rs`. No statistic is reimplemented here. The wrapped functions
//! (with their definitions, all in `zensim-validate/src/panel.rs`):
//!
//! | Stat / op             | panel.rs fn                  | file:line |
//! |-----------------------|------------------------------|-----------|
//! | full 6-stat panel     | `compute_panel`              | panel.rs:656 |
//! | Spearman SROCC        | `spearman` (via compute_panel)| panel.rs:50 |
//! | Pearson PLCC          | `pearson` (via compute_panel) | panel.rs:72 |
//! | Kendall KROCC         | `kendall_tau` (via compute_panel) | panel.rs:93 |
//! | Outlier ratio OR      | `outlier_ratio` (via compute_panel) | panel.rs:129 |
//! | PWRC                  | `pwrc` (via compute_panel)    | panel.rs:163 |
//! | global Z-RMSE         | `z_rmse` (via compute_panel)  | panel.rs:193 |
//! | per-sample Z-RMSE     | `z_rmse_per_sample`          | panel.rs:234 |
//! | 4-param logistic      | `rescale_logistic`           | panel.rs:458 |
//! | rank vector           | `ranks`                      | panel.rs:30 |
//!
//! `compute_panel` (panel.rs:656) internally calls `spearman.abs()`,
//! `kendall_tau.abs()`, `pwrc.abs()`, `outlier_ratio`, then
//! `rescale_logistic` → `pearson.abs()` for PLCC and `z_rmse` on the
//! rescaled scores for the global Z-RMSE. The `--sigma` column, when
//! present, additionally feeds `z_rmse_per_sample` (panel.rs:234) on the
//! logistic-rescaled scores — matching the per-stimulus σ-normalized
//! Z-RMSE of Mohammadi 2025 Eq. 6.
//!
//! ## Input
//!
//! A TSV (`--input foo.tsv`) or Parquet (`--input foo.parquet`) with
//! named columns:
//! - `predicted` (required) — the metric / model output.
//! - `target`    (required) — the human MOS / reference quality.
//! - `sigma`     (optional) — per-stimulus observer σ (enables the
//!   per-sample Z-RMSE; the global Z-RMSE is always reported).
//! - `band`      (optional) — a grouping key. When present, the panel is
//!   reported per distinct band value (sorted) AND in aggregate.
//!
//! TSV: tab-separated, first row is the header, column order is free
//! (located by name). Rows with a non-finite `predicted` or `target`
//! are dropped (and counted in the report) — matching the NaN-drop
//! policy of the Python reference (`scipy.stats.spearmanr` drops NaN
//! pairwise; we drop the whole row, which is the conservative choice
//! and what every py reimpl in the ledger does via `np.isfinite` masks).
//!
//! ## Output
//!
//! Text panel by default; `--json` for a machine-readable object. The
//! JSON is emitted by hand (no serde dep in the workspace) with full
//! `{:.10}` float precision and explicit `null` for NaN.
//!
//! ## Examples
//!
//! ```sh
//! # Panel on a TSV of (predicted, target) pairs
//! panel --input scores.tsv
//!
//! # With per-stimulus sigma (enables per-sample Z-RMSE) + JSON
//! panel --input scores_with_sigma.tsv --json
//!
//! # Per-band breakdown from a parquet
//! panel --input eval.parquet --json > panel.json
//! ```

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use zensim_validate::panel::{self, PanelStats};

// ----------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------

struct Args {
    input: PathBuf,
    json: bool,
    /// Override default column names if a caller's table uses different
    /// headers. Defaults: predicted / target / sigma / band.
    col_predicted: String,
    col_target: String,
    col_sigma: String,
    col_band: String,
    /// Hidden debug flag (used by the parity cross-check
    /// `scripts/verify_panel_parity.py`): instead of the panel, print
    /// the 4-param-logistic-rescaled `predicted` column (one value per
    /// line, full precision) for the ALL group. This lets the Python
    /// reference feed panel.rs's EXACT rescaled scores into its PLCC /
    /// Z-RMSE computation so the logistic-optimizer difference is
    /// removed and the underlying stat math can be gated to 1e-9.
    emit_rescaled: bool,
}

fn print_usage() {
    eprintln!(
        "panel — canonical IQA statistical-panel entry point (Mohammadi 2025)\n\
         \n\
         USAGE:\n\
         \x20\x20panel --input <FILE.tsv|FILE.parquet> [--json] [column overrides]\n\
         \n\
         INPUT COLUMNS (located by name; order free):\n\
         \x20\x20predicted   required   metric / model output\n\
         \x20\x20target      required   human MOS / reference quality\n\
         \x20\x20sigma       optional   per-stimulus observer σ (enables per-sample Z-RMSE)\n\
         \x20\x20band        optional   grouping key (per-band + aggregate panel)\n\
         \n\
         OPTIONS:\n\
         \x20\x20--input <PATH>          input TSV or Parquet (required)\n\
         \x20\x20--json                  emit JSON instead of text\n\
         \x20\x20--col-predicted <NAME>  override the 'predicted' column name\n\
         \x20\x20--col-target <NAME>     override the 'target' column name\n\
         \x20\x20--col-sigma <NAME>      override the 'sigma' column name\n\
         \x20\x20--col-band <NAME>       override the 'band' column name\n\
         \n\
         All statistics come from zensim_validate::panel (no stat math is\n\
         reimplemented here). See the module docs for the file:line map.\n\
         For bake-on-canonical-corpus evaluation use `bake_verdict` instead."
    );
}

fn parse_args() -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut json = false;
    let mut col_predicted = "predicted".to_string();
    let mut col_target = "target".to_string();
    let mut col_sigma = "sigma".to_string();
    let mut col_band = "band".to_string();
    let mut emit_rescaled = false;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" | "-i" => {
                input = Some(PathBuf::from(it.next().ok_or("--input requires a value")?));
            }
            "--json" => json = true,
            // Hidden — see Args::emit_rescaled.
            "--emit-rescaled" => emit_rescaled = true,
            "--col-predicted" => {
                col_predicted = it.next().ok_or("--col-predicted requires a value")?
            }
            "--col-target" => col_target = it.next().ok_or("--col-target requires a value")?,
            "--col-sigma" => col_sigma = it.next().ok_or("--col-sigma requires a value")?,
            "--col-band" => col_band = it.next().ok_or("--col-band requires a value")?,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        input: input.ok_or("--input is required")?,
        json,
        col_predicted,
        col_target,
        col_sigma,
        col_band,
        emit_rescaled,
    })
}

// ----------------------------------------------------------------------
// Loaded columns
// ----------------------------------------------------------------------

/// Raw columns read from the input file. `sigma` / `band` are `None`
/// when the column is absent. All vectors are the same length (file row
/// count); row filtering happens later so band membership stays aligned.
struct Columns {
    predicted: Vec<f64>,
    target: Vec<f64>,
    sigma: Option<Vec<f64>>,
    band: Option<Vec<String>>,
}

fn load_columns(args: &Args) -> Result<Columns, String> {
    let path = &args.input;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "parquet" => load_parquet_columns(args),
        "tsv" | "csv" | "txt" | "" => load_tsv_columns(args),
        other => Err(format!(
            "unsupported input extension {other:?} (expected .tsv / .parquet)"
        )),
    }
}

fn load_tsv_columns(args: &Args) -> Result<Columns, String> {
    let path: &Path = &args.input;
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| format!("{path:?}: empty file (no header)"))?;
    // Tab-separated by default; fall back to comma for .csv.
    let sep = if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("csv"))
        .unwrap_or(false)
    {
        ','
    } else {
        '\t'
    };
    let cols: Vec<&str> = header.split(sep).map(|c| c.trim()).collect();
    let find = |name: &str| cols.iter().position(|c| *c == name);

    let pi = find(&args.col_predicted)
        .ok_or_else(|| format!("{path:?}: missing '{}' column", args.col_predicted))?;
    let ti = find(&args.col_target)
        .ok_or_else(|| format!("{path:?}: missing '{}' column", args.col_target))?;
    let si = find(&args.col_sigma);
    let bi = find(&args.col_band);

    let mut predicted = Vec::new();
    let mut target = Vec::new();
    let mut sigma = if si.is_some() { Some(Vec::new()) } else { None };
    let mut band = if bi.is_some() { Some(Vec::new()) } else { None };

    for (lineno, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(sep).collect();
        let get = |idx: usize| -> Result<&str, String> {
            fields
                .get(idx)
                .map(|s| s.trim())
                .ok_or_else(|| format!("{path:?}:{}: too few columns", lineno + 2))
        };
        let parse_f = |s: &str| -> f64 { s.parse::<f64>().unwrap_or(f64::NAN) };
        predicted.push(parse_f(get(pi)?));
        target.push(parse_f(get(ti)?));
        if let (Some(idx), Some(v)) = (si, sigma.as_mut()) {
            v.push(parse_f(get(idx)?));
        }
        if let (Some(idx), Some(v)) = (bi, band.as_mut()) {
            v.push(get(idx)?.to_string());
        }
    }

    Ok(Columns {
        predicted,
        target,
        sigma,
        band,
    })
}

fn load_parquet_columns(args: &Args) -> Result<Columns, String> {
    let path: &Path = &args.input;
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let names: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    let find = |name: &str| names.iter().position(|c| c == name);

    let pi = find(&args.col_predicted)
        .ok_or_else(|| format!("{path:?}: missing '{}' column", args.col_predicted))?;
    let ti = find(&args.col_target)
        .ok_or_else(|| format!("{path:?}: missing '{}' column", args.col_target))?;
    let si = find(&args.col_sigma);
    let bi = find(&args.col_band);

    let reader = builder
        .with_batch_size(16384)
        .build()
        .map_err(|e| format!("{path:?}: parquet build reader: {e}"))?;

    let mut predicted = Vec::new();
    let mut target = Vec::new();
    let mut sigma = if si.is_some() { Some(Vec::new()) } else { None };
    let mut band = if bi.is_some() { Some(Vec::new()) } else { None };

    // Pull a numeric column as f64 (Float32/Float64/Int32/Int64), NaN for null.
    fn numeric_col(col: &dyn Array, name: &str) -> Result<Vec<f64>, String> {
        let n = col.len();
        match col.data_type() {
            DataType::Float64 => {
                let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                Ok((0..n)
                    .map(|i| if a.is_null(i) { f64::NAN } else { a.value(i) })
                    .collect())
            }
            DataType::Float32 => {
                let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                Ok((0..n)
                    .map(|i| {
                        if a.is_null(i) {
                            f64::NAN
                        } else {
                            a.value(i) as f64
                        }
                    })
                    .collect())
            }
            DataType::Int64 => {
                let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
                Ok((0..n)
                    .map(|i| {
                        if a.is_null(i) {
                            f64::NAN
                        } else {
                            a.value(i) as f64
                        }
                    })
                    .collect())
            }
            DataType::Int32 => {
                let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
                Ok((0..n)
                    .map(|i| {
                        if a.is_null(i) {
                            f64::NAN
                        } else {
                            a.value(i) as f64
                        }
                    })
                    .collect())
            }
            other => Err(format!(
                "column {name:?} has unsupported numeric dtype {other:?}"
            )),
        }
    }

    fn string_col(col: &dyn Array, name: &str) -> Result<Vec<String>, String> {
        let n = col.len();
        match col.data_type() {
            DataType::Utf8 => {
                let a = col.as_any().downcast_ref::<StringArray>().unwrap();
                Ok((0..n)
                    .map(|i| {
                        if a.is_null(i) {
                            String::new()
                        } else {
                            a.value(i).to_string()
                        }
                    })
                    .collect())
            }
            // A numeric band column is fine — stringify it.
            DataType::Float64 | DataType::Float32 | DataType::Int64 | DataType::Int32 => {
                Ok(numeric_col(col, name)?
                    .into_iter()
                    .map(|v| format!("{v}"))
                    .collect())
            }
            other => Err(format!(
                "band column {name:?} has unsupported dtype {other:?}"
            )),
        }
    }

    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: read batch: {e}"))?;
        predicted.extend(numeric_col(batch.column(pi), &args.col_predicted)?);
        target.extend(numeric_col(batch.column(ti), &args.col_target)?);
        if let (Some(idx), Some(v)) = (si, sigma.as_mut()) {
            v.extend(numeric_col(batch.column(idx), &args.col_sigma)?);
        }
        if let (Some(idx), Some(v)) = (bi, band.as_mut()) {
            v.extend(string_col(batch.column(idx), &args.col_band)?);
        }
    }

    Ok(Columns {
        predicted,
        target,
        sigma,
        band,
    })
}

// ----------------------------------------------------------------------
// Panel computation on a (filtered) slice
// ----------------------------------------------------------------------

/// One reported group: the aggregate or a single band.
struct GroupReport {
    label: String,
    panel: PanelStats,
    /// Per-sample σ-normalized Z-RMSE (Mohammadi Eq. 6) — only when a
    /// `sigma` column was supplied. Computed on the logistic-rescaled
    /// scores (same rescale `compute_panel` uses), so it lines up with
    /// the global Z-RMSE in `panel.z_rmse`.
    z_rmse_per_sample: Option<f64>,
    /// Rows kept after dropping non-finite (predicted, target) pairs.
    n_kept: usize,
    /// Rows dropped because predicted or target was non-finite.
    n_dropped: usize,
}

/// Filter to finite (predicted, target) rows, then run the panel. When
/// `sigma` is present, also computes the per-sample Z-RMSE on the
/// logistic-rescaled scores via `panel::rescale_logistic` +
/// `panel::z_rmse_per_sample`.
fn report_group(
    label: &str,
    predicted: &[f64],
    target: &[f64],
    sigma: Option<&[f64]>,
) -> GroupReport {
    let n = predicted.len().min(target.len());
    let mut p = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    let mut s: Option<Vec<f64>> = sigma.map(|_| Vec::with_capacity(n));
    let mut n_dropped = 0usize;
    for i in 0..n {
        if predicted[i].is_finite() && target[i].is_finite() {
            p.push(predicted[i]);
            t.push(target[i]);
            if let (Some(sig), Some(out)) = (sigma, s.as_mut()) {
                out.push(sig[i]);
            }
        } else {
            n_dropped += 1;
        }
    }

    // compute_panel (panel.rs:656) — the full 6-stat Mohammadi panel.
    let panel = panel::compute_panel(&p, &t);

    // Per-sample Z-RMSE on the logistic-rescaled scores. We mirror what
    // compute_panel does internally for the global Z-RMSE: rescale the
    // predicted scores through the 4-param logistic (panel.rs:458), then
    // feed (rescaled, target, sigma) to z_rmse_per_sample (panel.rs:234).
    let z_rmse_per_sample = s.as_ref().map(|sig| {
        let rescaled = panel::rescale_logistic(&p, &t);
        panel::z_rmse_per_sample(&rescaled, &t, sig)
    });

    GroupReport {
        label: label.to_string(),
        panel,
        z_rmse_per_sample,
        n_kept: p.len(),
        n_dropped,
    }
}

fn build_reports(cols: &Columns) -> Vec<GroupReport> {
    let mut out = Vec::new();
    // Aggregate panel over all rows.
    out.push(report_group(
        "ALL",
        &cols.predicted,
        &cols.target,
        cols.sigma.as_deref(),
    ));

    // Per-band panels when a band column is present.
    if let Some(bands) = &cols.band {
        // Group row indices by band value, sorted lexicographically for
        // deterministic output.
        let mut by_band: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (i, b) in bands.iter().enumerate() {
            by_band.entry(b.as_str()).or_default().push(i);
        }
        for (band_val, idxs) in by_band {
            let p: Vec<f64> = idxs.iter().map(|&i| cols.predicted[i]).collect();
            let t: Vec<f64> = idxs.iter().map(|&i| cols.target[i]).collect();
            let s: Option<Vec<f64>> = cols
                .sigma
                .as_ref()
                .map(|sig| idxs.iter().map(|&i| sig[i]).collect());
            out.push(report_group(
                &format!("band={band_val}"),
                &p,
                &t,
                s.as_deref(),
            ));
        }
    }
    out
}

// ----------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------

fn fmt_stat(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.4}")
    } else {
        "  n/a ".to_string()
    }
}

fn render_text(reports: &[GroupReport], has_sigma: bool) -> String {
    let mut s = String::new();
    s.push_str("# panel — Mohammadi 2025 IQA statistical panel\n");
    s.push_str("# (all stats via zensim_validate::panel — see `panel --help`)\n\n");
    if has_sigma {
        s.push_str(&format!(
            "{:<24} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>12}\n",
            "group", "n", "SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE", "Z-RMSE/σ"
        ));
    } else {
        s.push_str(&format!(
            "{:<24} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}\n",
            "group", "n", "SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE"
        ));
    }
    for r in reports {
        let p = &r.panel;
        if has_sigma {
            s.push_str(&format!(
                "{:<24} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>12}\n",
                r.label,
                r.n_kept,
                fmt_stat(p.srocc),
                fmt_stat(p.plcc),
                fmt_stat(p.krocc),
                fmt_stat(p.or_ratio),
                fmt_stat(p.pwrc),
                fmt_stat(p.z_rmse),
                fmt_stat(r.z_rmse_per_sample.unwrap_or(f64::NAN)),
            ));
        } else {
            s.push_str(&format!(
                "{:<24} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}\n",
                r.label,
                r.n_kept,
                fmt_stat(p.srocc),
                fmt_stat(p.plcc),
                fmt_stat(p.krocc),
                fmt_stat(p.or_ratio),
                fmt_stat(p.pwrc),
                fmt_stat(p.z_rmse),
            ));
        }
    }
    let dropped: usize = reports.first().map(|r| r.n_dropped).unwrap_or(0);
    if dropped > 0 {
        s.push_str(&format!(
            "\n# {dropped} row(s) dropped (non-finite predicted/target) from the ALL group\n"
        ));
    }
    s
}

/// JSON helper: a finite f64 renders as `{:.10}`, non-finite as `null`.
fn json_f(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.10}")
    } else {
        "null".to_string()
    }
}

fn render_json(reports: &[GroupReport], has_sigma: bool) -> String {
    let mut s = String::from("{\n  \"groups\": [\n");
    for (gi, r) in reports.iter().enumerate() {
        let p = &r.panel;
        s.push_str("    {\n");
        s.push_str(&format!("      \"label\": {:?},\n", r.label));
        s.push_str(&format!("      \"n\": {},\n", r.n_kept));
        s.push_str(&format!("      \"n_dropped\": {},\n", r.n_dropped));
        s.push_str(&format!("      \"srocc\": {},\n", json_f(p.srocc)));
        s.push_str(&format!("      \"plcc\": {},\n", json_f(p.plcc)));
        s.push_str(&format!("      \"krocc\": {},\n", json_f(p.krocc)));
        s.push_str(&format!("      \"or\": {},\n", json_f(p.or_ratio)));
        s.push_str(&format!("      \"pwrc\": {},\n", json_f(p.pwrc)));
        if has_sigma {
            s.push_str(&format!("      \"z_rmse\": {},\n", json_f(p.z_rmse)));
            s.push_str(&format!(
                "      \"z_rmse_per_sample\": {}\n",
                json_f(r.z_rmse_per_sample.unwrap_or(f64::NAN))
            ));
        } else {
            s.push_str(&format!("      \"z_rmse\": {}\n", json_f(p.z_rmse)));
        }
        s.push_str("    }");
        if gi + 1 < reports.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n}\n");
    s
}

// ----------------------------------------------------------------------
// main
// ----------------------------------------------------------------------

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("panel: {e}\n");
            print_usage();
            return ExitCode::from(2);
        }
    };

    let cols = match load_columns(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("panel: {e}");
            return ExitCode::from(1);
        }
    };

    if cols.predicted.is_empty() {
        eprintln!("panel: no data rows in {:?}", args.input);
        return ExitCode::from(1);
    }

    // Hidden parity-gate path: emit the logistic-rescaled `predicted`
    // column for finite (predicted, target) rows of the ALL group, one
    // per line. Used by scripts/verify_panel_parity.py.
    if args.emit_rescaled {
        let n = cols.predicted.len().min(cols.target.len());
        let mut p = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            if cols.predicted[i].is_finite() && cols.target[i].is_finite() {
                p.push(cols.predicted[i]);
                t.push(cols.target[i]);
            }
        }
        let rescaled = panel::rescale_logistic(&p, &t);
        let mut out = String::new();
        for v in rescaled {
            out.push_str(&format!("{v:.17e}\n"));
        }
        print!("{out}");
        return ExitCode::SUCCESS;
    }

    let has_sigma = cols.sigma.is_some();
    let reports = build_reports(&cols);

    if args.json {
        print!("{}", render_json(&reports, has_sigma));
    } else {
        print!("{}", render_text(&reports, has_sigma));
    }
    ExitCode::SUCCESS
}

// ----------------------------------------------------------------------
// Tests — the panel subcommand's own plumbing (NOT the stat math, which
// is tested in panel.rs). The cross-language parity gate lives in
// `tests/panel_parity.rs`.
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_group_drops_nonfinite() {
        let predicted = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let target = vec![1.0, 2.0, 3.0, f64::INFINITY, 5.0];
        let r = report_group("t", &predicted, &target, None);
        assert_eq!(r.n_kept, 3, "rows 0,1,4 kept");
        assert_eq!(r.n_dropped, 2, "rows 2,3 dropped");
    }

    #[test]
    fn report_group_perfect_rank() {
        let predicted: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let target: Vec<f64> = (0..50).map(|i| i as f64 * 2.0 + 3.0).collect();
        let r = report_group("t", &predicted, &target, None);
        assert!(
            (r.panel.srocc - 1.0).abs() < 1e-9,
            "srocc={}",
            r.panel.srocc
        );
        assert!(
            (r.panel.krocc - 1.0).abs() < 1e-9,
            "krocc={}",
            r.panel.krocc
        );
    }

    #[test]
    fn per_sample_z_rmse_present_only_with_sigma() {
        let predicted: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let target: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let without = report_group("t", &predicted, &target, None);
        assert!(without.z_rmse_per_sample.is_none());
        let sigma = vec![1.0; 40];
        let with = report_group("t", &predicted, &target, Some(&sigma));
        assert!(with.z_rmse_per_sample.is_some());
    }

    #[test]
    fn json_renders_nan_as_null() {
        assert_eq!(json_f(f64::NAN), "null");
        assert_eq!(json_f(f64::INFINITY), "null");
        assert_eq!(json_f(0.5), "0.5000000000");
    }
}
