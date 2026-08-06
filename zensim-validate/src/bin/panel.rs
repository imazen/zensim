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
//!
//! # Batch mode: N (x, y) vector pairs in, N panel rows out, ONE process
//! panel --batch jobs.tsv --stats srocc > sroccs.tsv
//! ```
//!
//! ## Batch mode (`--batch`, decision-surface audit 2026-07-31 gap 4)
//!
//! Per-call shelling is fine for aggregates but prohibitive inside
//! bootstrap loops (10k resamples = 10k process spawns). `--batch`
//! reads a manifest of many (x, y) vector pairs and emits one stats row
//! per pair **in one process**, so a 10k-resample bootstrap is a single
//! invocation. This is the canonical replacement for every
//! `scipy.stats.spearmanr`-in-a-loop call site (the banned pattern);
//! Python callers use `scripts/lib/zen_stats.panel_batch` /
//! `panel_batch_indexed`.
//!
//! Input (`--batch <FILE|->`, `-` = stdin) is line-oriented, tab-separated:
//!
//! ```text
//! #def NAME<TAB>v1,v2,v3,...            define a named base vector
//! LABEL<TAB>x1,x2,...<TAB>y1,y2,...     explicit (x, y) pair
//! LABEL<TAB>@X:@Y<TAB>i1,i2,...         indexed pair: (X[i], Y[i]) over
//!                                       previously #def'd bases (the SAME
//!                                       index set applies to both — the
//!                                       paired-bootstrap resample shape)
//! LABEL<TAB>@X:@Y<TAB>*                 indexed pair over ALL rows
//! ```
//!
//! Blank lines and `#`-comments (other than `#def`) are skipped. The
//! indexed form exists so a caller resampling the same base vectors 10k
//! times ships ~n integers per job instead of ~2n floats; the caller
//! keeps ownership of the resampling RNG (deterministic seeding stays at
//! the call site, e.g. upiq_panel.py's `np.random.default_rng(20260714)`),
//! and this binary stays RNG-free — batch output is a pure deterministic
//! function of the input bytes.
//!
//! Output: TSV with a header, one row per job in input order. Columns:
//!
//! * `--stats srocc` (the bootstrap fast path):
//!   `label n n_dropped srocc srocc_signed`
//! * `--stats full` (default — everything the aggregate panel emits):
//!   `label n n_dropped srocc srocc_signed plcc plcc_raw krocc or pwrc z_rmse`
//!
//! Zero new stat math, matching this binary's charter: `srocc_signed` is
//! a direct `panel::spearman` call (tie-correct midrank, pre-`.abs()`),
//! `plcc_raw` a direct `panel::pearson` on the raw pair (no logistic
//! rescale — some registered instruments, e.g. `scripts/hdr/upiq_panel.py`,
//! report raw-Pearson |PLCC|), and everything else comes from
//! `compute_panel` exactly as in aggregate mode. Non-finite (x, y) rows
//! are dropped per job and counted in `n_dropped` (same policy as
//! aggregate mode). Floats print as `{:.17e}` (round-trip exact).
//! `--json` is not supported with `--batch` (the TSV is the contract).

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
    input: Option<PathBuf>,
    /// Batch mode: a manifest of many (x, y) vector pairs (`-` = stdin).
    /// Mutually exclusive with `--input`. See the module docs.
    batch: Option<PathBuf>,
    /// Batch column set: `full` (default) or `srocc` (bootstrap fast path).
    stats_srocc_only: bool,
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
         \x20\x20panel --batch <FILE.tsv|-> [--stats full|srocc]\n\
         \n\
         INPUT COLUMNS (located by name; order free):\n\
         \x20\x20predicted   required   metric / model output\n\
         \x20\x20target      required   human MOS / reference quality\n\
         \x20\x20sigma       optional   per-stimulus observer σ (enables per-sample Z-RMSE)\n\
         \x20\x20band        optional   grouping key (per-band + aggregate panel)\n\
         \n\
         OPTIONS:\n\
         \x20\x20--input <PATH>          input TSV or Parquet (aggregate mode)\n\
         \x20\x20--batch <PATH|->        batch manifest: N (x,y) vector pairs in,\n\
         \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20N stat rows out, ONE process (see docs;\n\
         \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20lines: '#def N<TAB>csv', 'L<TAB>x-csv<TAB>y-csv',\n\
         \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20'L<TAB>@X:@Y<TAB>idx-csv|*')\n\
         \x20\x20--stats <full|srocc>    batch column set (default full)\n\
         \x20\x20--json                  emit JSON instead of text (aggregate mode only)\n\
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
    let mut batch: Option<PathBuf> = None;
    let mut stats_srocc_only = false;
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
            "--batch" => {
                batch = Some(PathBuf::from(it.next().ok_or("--batch requires a value")?));
            }
            "--stats" => {
                let v = it.next().ok_or("--stats requires a value")?;
                stats_srocc_only = match v.as_str() {
                    "srocc" => true,
                    "full" => false,
                    other => return Err(format!("--stats must be 'full' or 'srocc', got {other}")),
                };
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

    if input.is_some() && batch.is_some() {
        return Err("--input and --batch are mutually exclusive".to_string());
    }
    if batch.is_some() && json {
        return Err("--json is not supported with --batch (the TSV is the contract)".to_string());
    }
    if input.is_none() && batch.is_none() {
        return Err("--input or --batch is required".to_string());
    }

    Ok(Args {
        input,
        batch,
        stats_srocc_only,
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
    let path = args
        .input
        .as_ref()
        .expect("aggregate mode requires --input");
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
    let path: &Path = args
        .input
        .as_ref()
        .expect("aggregate mode requires --input");
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
    let path: &Path = args
        .input
        .as_ref()
        .expect("aggregate mode requires --input");
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
// Batch mode — N (x, y) pairs in, N stat rows out, one process.
// (decision_surface_audit_2026-07-31.md gap 4; zero new stat math —
// every number is a direct `zensim_validate::panel` owner call.)
// ----------------------------------------------------------------------

/// One parsed batch job. `Indexed` references `#def`'d base vectors by
/// slot so 10k resample jobs share storage instead of cloning floats.
enum BatchJob {
    Explicit {
        x: Vec<f64>,
        y: Vec<f64>,
    },
    Indexed {
        x_base: usize,
        y_base: usize,
        /// `None` = all rows (the `*` form).
        idx: Option<Vec<usize>>,
    },
}

struct BatchInput {
    bases: Vec<Vec<f64>>,
    jobs: Vec<(String, BatchJob)>,
}

fn parse_float_csv(s: &str, lineno: usize, what: &str) -> Result<Vec<f64>, String> {
    s.split(',')
        .map(|t| {
            let t = t.trim();
            t.parse::<f64>()
                .map_err(|e| format!("batch line {lineno}: bad {what} float {t:?}: {e}"))
        })
        .collect()
}

fn parse_batch(text: &str) -> Result<BatchInput, String> {
    let mut bases: Vec<Vec<f64>> = Vec::new();
    let mut base_names: BTreeMap<String, usize> = BTreeMap::new();
    let mut jobs: Vec<(String, BatchJob)> = Vec::new();

    for (i, line) in text.lines().enumerate() {
        let lineno = i + 1;
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("#def") {
            let mut parts = rest.trim_start().splitn(2, '\t');
            let name = parts
                .next()
                .filter(|n| !n.is_empty())
                .ok_or_else(|| format!("batch line {lineno}: #def needs NAME<TAB>csv"))?;
            let csv = parts
                .next()
                .ok_or_else(|| format!("batch line {lineno}: #def {name} missing vector"))?;
            let vec = parse_float_csv(csv, lineno, &format!("#def {name}"))?;
            if base_names.insert(name.to_string(), bases.len()).is_some() {
                return Err(format!("batch line {lineno}: duplicate #def {name}"));
            }
            bases.push(vec);
            continue;
        }
        if line.starts_with('#') {
            continue; // plain comment
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err(format!(
                "batch line {lineno}: expected LABEL<TAB>x<TAB>y (3 fields), got {}",
                fields.len()
            ));
        }
        let label = fields[0].to_string();
        let job = if let Some(refs) = fields[1].strip_prefix('@') {
            // Indexed form: @X:@Y <TAB> idx-csv | *
            let (xn, yn) = refs
                .split_once(":@")
                .ok_or_else(|| format!("batch line {lineno}: indexed form needs @X:@Y"))?;
            let x_base = *base_names
                .get(xn)
                .ok_or_else(|| format!("batch line {lineno}: undefined base @{xn}"))?;
            let y_base = *base_names
                .get(yn)
                .ok_or_else(|| format!("batch line {lineno}: undefined base @{yn}"))?;
            let (xl, yl) = (bases[x_base].len(), bases[y_base].len());
            if xl != yl {
                return Err(format!(
                    "batch line {lineno}: base @{xn} (len {xl}) and @{yn} (len {yl}) differ — \
                     the shared index set requires equal-length bases"
                ));
            }
            let idx = if fields[2].trim() == "*" {
                None
            } else {
                let idx: Vec<usize> = fields[2]
                    .split(',')
                    .map(|t| {
                        let t = t.trim();
                        t.parse::<usize>()
                            .map_err(|e| format!("batch line {lineno}: bad index {t:?}: {e}"))
                    })
                    .collect::<Result<_, _>>()?;
                if let Some(&bad) = idx.iter().find(|&&v| v >= xl) {
                    return Err(format!(
                        "batch line {lineno}: index {bad} out of bounds for base len {xl}"
                    ));
                }
                Some(idx)
            };
            BatchJob::Indexed {
                x_base,
                y_base,
                idx,
            }
        } else {
            let x = parse_float_csv(fields[1], lineno, "x")?;
            let y = parse_float_csv(fields[2], lineno, "y")?;
            if x.len() != y.len() {
                return Err(format!(
                    "batch line {lineno}: x (len {}) and y (len {}) differ",
                    x.len(),
                    y.len()
                ));
            }
            BatchJob::Explicit { x, y }
        };
        jobs.push((label, job));
    }
    Ok(BatchInput { bases, jobs })
}

/// One computed batch row. Field meanings match the aggregate panel;
/// `srocc_signed` / `plcc_raw` are the direct owner calls documented in
/// the module docs (pre-`.abs()` midrank Spearman; raw un-rescaled Pearson).
struct BatchRow {
    n: usize,
    n_dropped: usize,
    srocc: f64,
    srocc_signed: f64,
    full: Option<(PanelStats, f64, f64)>, // (compute_panel stats, plcc_raw, mae)
}

fn compute_batch_row(x: &[f64], y: &[f64], srocc_only: bool) -> BatchRow {
    // Same finite-row drop policy as aggregate mode (report_group).
    let n_in = x.len().min(y.len());
    let mut xf = Vec::with_capacity(n_in);
    let mut yf = Vec::with_capacity(n_in);
    let mut n_dropped = 0usize;
    for i in 0..n_in {
        if x[i].is_finite() && y[i].is_finite() {
            xf.push(x[i]);
            yf.push(y[i]);
        } else {
            n_dropped += 1;
        }
    }
    // panel.rs:50 `spearman` — tie-correct midrank, signed.
    let srocc_signed = panel::spearman(&xf, &yf);
    let full = if srocc_only {
        None
    } else {
        // compute_panel (panel.rs:656) — the full 6-stat panel — plus the
        // raw (un-rescaled) Pearson via panel.rs:72 `pearson`.
        let stats = panel::compute_panel(&xf, &yf);
        let plcc_raw = panel::pearson(&xf, &yf);
        // MAE after the owner's 4-parameter logistic rescale (Mohammadi 2025
        // convention) — the same quantity `bake_verdict`'s per-band rows
        // publish, computed by the same owner call, so a band recomputed
        // through this binary carries every field the emitter does.
        let rescaled = panel::rescale_logistic(&xf, &yf);
        let mae = if rescaled.is_empty() {
            f64::NAN
        } else {
            rescaled
                .iter()
                .zip(yf.iter())
                .map(|(r, t)| (r - t).abs())
                .sum::<f64>()
                / rescaled.len() as f64
        };
        Some((stats, plcc_raw, mae))
    };
    BatchRow {
        n: xf.len(),
        n_dropped,
        srocc: srocc_signed.abs(),
        srocc_signed,
        full,
    }
}

/// Round-trip-exact float formatting for the batch TSV.
fn fmt_batch_f(v: f64) -> String {
    format!("{v:.17e}")
}

fn run_batch(input: &BatchInput, srocc_only: bool) -> String {
    use rayon::prelude::*;

    let rows: Vec<BatchRow> = input
        .jobs
        .par_iter()
        .map(|(_, job)| match job {
            BatchJob::Explicit { x, y } => compute_batch_row(x, y, srocc_only),
            BatchJob::Indexed {
                x_base,
                y_base,
                idx,
            } => {
                let (bx, by) = (&input.bases[*x_base], &input.bases[*y_base]);
                match idx {
                    None => compute_batch_row(bx, by, srocc_only),
                    Some(idx) => {
                        let x: Vec<f64> = idx.iter().map(|&i| bx[i]).collect();
                        let y: Vec<f64> = idx.iter().map(|&i| by[i]).collect();
                        compute_batch_row(&x, &y, srocc_only)
                    }
                }
            }
        })
        .collect();

    let mut out = String::new();
    if srocc_only {
        out.push_str("label\tn\tn_dropped\tsrocc\tsrocc_signed\n");
    } else {
        out.push_str(
            "label\tn\tn_dropped\tsrocc\tsrocc_signed\tplcc\tplcc_raw\tkrocc\tor\tpwrc\tz_rmse\tmae\n",
        );
    }
    for ((label, _), r) in input.jobs.iter().zip(&rows) {
        out.push_str(label);
        out.push_str(&format!("\t{}\t{}", r.n, r.n_dropped));
        out.push_str(&format!(
            "\t{}\t{}",
            fmt_batch_f(r.srocc),
            fmt_batch_f(r.srocc_signed)
        ));
        if let Some((p, plcc_raw, mae)) = &r.full {
            out.push_str(&format!(
                "\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                fmt_batch_f(p.plcc),
                fmt_batch_f(*plcc_raw),
                fmt_batch_f(p.krocc),
                fmt_batch_f(p.or_ratio),
                fmt_batch_f(p.pwrc),
                fmt_batch_f(p.z_rmse),
                fmt_batch_f(*mae)
            ));
        }
        out.push('\n');
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

    // Batch mode: read the manifest (file or stdin), emit the TSV, done.
    if let Some(batch_path) = &args.batch {
        let text = if batch_path.as_os_str() == "-" {
            use std::io::Read;
            let mut s = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut s) {
                eprintln!("panel: read stdin: {e}");
                return ExitCode::from(1);
            }
            s
        } else {
            match std::fs::read_to_string(batch_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("panel: read {batch_path:?}: {e}");
                    return ExitCode::from(1);
                }
            }
        };
        let input = match parse_batch(&text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("panel: {e}");
                return ExitCode::from(2);
            }
        };
        print!("{}", run_batch(&input, args.stats_srocc_only));
        return ExitCode::SUCCESS;
    }

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

    // ------------------------------------------------------------------
    // Batch-mode plumbing (the stat math itself is tested in panel.rs /
    // tests/panel_parity.rs; scipy cross-check in
    // scripts/verify_panel_batch_parity.py).
    // ------------------------------------------------------------------

    #[test]
    fn batch_parses_explicit_and_indexed_and_star() {
        let text = "#def X\t1,2,3,4\n#def Y\t10,20,30,40\n\
                    # a comment\n\
                    e1\t1,2,3\t3,2,1\n\
                    i1\t@X:@Y\t0,2,1\n\
                    all\t@X:@Y\t*\n";
        let b = parse_batch(text).unwrap();
        assert_eq!(b.bases.len(), 2);
        assert_eq!(b.jobs.len(), 3);
        match &b.jobs[0].1 {
            BatchJob::Explicit { x, y } => {
                assert_eq!(x, &[1.0, 2.0, 3.0]);
                assert_eq!(y, &[3.0, 2.0, 1.0]);
            }
            _ => panic!("job 0 should be explicit"),
        }
        match &b.jobs[1].1 {
            BatchJob::Indexed { idx: Some(idx), .. } => assert_eq!(idx, &[0, 2, 1]),
            _ => panic!("job 1 should be indexed"),
        }
        match &b.jobs[2].1 {
            BatchJob::Indexed { idx: None, .. } => {}
            _ => panic!("job 2 should be the * form"),
        }
    }

    #[test]
    fn batch_rejects_bad_input_loudly() {
        assert!(parse_batch("l\t1,2\t1,2,3\n").is_err(), "length mismatch");
        assert!(parse_batch("l\t@A:@B\t0\n").is_err(), "undefined base");
        assert!(
            parse_batch("#def A\t1,2\nl\t@A:@A\t5\n").is_err(),
            "index out of bounds"
        );
        assert!(parse_batch("l\t1,zz\t1,2\n").is_err(), "garbage float");
        assert!(
            parse_batch("#def A\t1,2\n#def A\t3,4\n").is_err(),
            "duplicate def"
        );
        assert!(
            parse_batch("#def A\t1,2\n#def B\t1,2,3\nl\t@A:@B\t*\n").is_err(),
            "unequal base lengths"
        );
    }

    #[test]
    fn batch_indexed_matches_explicit_and_is_deterministic() {
        // Same resample expressed both ways must produce identical rows,
        // and two runs must be byte-identical (no RNG anywhere).
        let text = "#def P\t12,9,30,25,5,40\n#def T\t80,85,40,55,92,20\n\
                    ex\t9,30,5,9\t85,40,92,85\n\
                    ix\t@P:@T\t1,2,4,1\n";
        let b = parse_batch(text).unwrap();
        let out1 = run_batch(&b, false);
        let out2 = run_batch(&b, false);
        assert_eq!(out1, out2, "batch output must be deterministic");
        let lines: Vec<&str> = out1.lines().collect();
        assert_eq!(lines.len(), 3);
        let ex_cols: Vec<&str> = lines[1].split('\t').collect();
        let ix_cols: Vec<&str> = lines[2].split('\t').collect();
        assert_eq!(ex_cols[1..], ix_cols[1..], "indexed == explicit");
    }

    #[test]
    fn batch_row_drops_nonfinite_and_counts() {
        let x = [1.0, f64::NAN, 3.0, 4.0];
        let y = [1.0, 2.0, f64::INFINITY, 4.0];
        let r = compute_batch_row(&x, &y, true);
        assert_eq!(r.n, 2);
        assert_eq!(r.n_dropped, 2);
    }

    #[test]
    fn batch_srocc_signed_carries_polarity() {
        // Anti-correlated pair: signed is negative, abs is 1.0.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 8.0, 6.0, 4.0, 2.0];
        let r = compute_batch_row(&x, &y, true);
        assert!((r.srocc - 1.0).abs() < 1e-12, "srocc={}", r.srocc);
        assert!(
            (r.srocc_signed + 1.0).abs() < 1e-12,
            "srocc_signed={}",
            r.srocc_signed
        );
    }

    #[test]
    fn batch_full_matches_aggregate_compute_panel() {
        // The full batch row must equal what aggregate mode's
        // report_group produces on the same pair.
        let pred: Vec<f64> = vec![12.0, 9.0, 30.0, 25.0, 5.0, 40.0, 22.0, 18.0];
        let tgt: Vec<f64> = vec![80.0, 85.0, 40.0, 55.0, 92.0, 20.0, 60.0, 70.0];
        let r = compute_batch_row(&pred, &tgt, false);
        let agg = report_group("t", &pred, &tgt, None);
        let (p, plcc_raw, _mae) = r.full.as_ref().unwrap();
        assert_eq!(p.srocc, agg.panel.srocc);
        assert_eq!(p.plcc, agg.panel.plcc);
        assert_eq!(p.krocc, agg.panel.krocc);
        assert_eq!(p.or_ratio, agg.panel.or_ratio);
        assert_eq!(p.pwrc, agg.panel.pwrc);
        assert_eq!(p.z_rmse, agg.panel.z_rmse);
        assert_eq!(r.srocc, agg.panel.srocc, "abs srocc column == panel srocc");
        // plcc_raw is signed raw Pearson (distance-shaped fixture → negative).
        assert!(*plcc_raw < 0.0, "raw pearson keeps polarity: {plcc_raw}");
    }

    #[test]
    fn batch_tie_heavy_midrank() {
        // Heavy exact ties; golden value from scipy.stats.spearmanr
        // (midrank) run on this exact fixture: 0.8846153846153847
        // (recomputed 2026-07-31; the full randomized sweep incl.
        // tie-heavy cases is scripts/verify_panel_batch_parity.py).
        let x = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        let y = [1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let r = compute_batch_row(&x, &y, true);
        assert!(
            (r.srocc_signed - 0.884_615_384_615_384_7).abs() < 1e-12,
            "tie-heavy midrank srocc={}",
            r.srocc_signed
        );
    }
}
