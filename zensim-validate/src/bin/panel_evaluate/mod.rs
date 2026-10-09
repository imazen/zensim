//! Dataset/implementation adapters for the existing `panel` statistical owner.
//! No default corpus discovery, model fitting, or production qualification.

mod adapter;
mod dataset;
mod measurements;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "panel evaluate",
    about = "Evaluate metric implementations on explicit datasets"
)]
struct Cli {
    #[arg(long)]
    manifest: PathBuf,
    /// New directory for reports, scored rows and retained adapter output.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Suite {
    schema: String,
    datasets: Vec<Dataset>,
    metrics: Vec<Metric>,
    #[serde(default)]
    instruments: Vec<Instrument>,
    #[serde(default)]
    requirements: Vec<Requirement>,
    #[serde(default)]
    comparisons: Vec<Comparison>,
    /// Full PWRC is quadratic. Refuse, never silently sample, larger panels.
    #[serde(default = "default_panel_limit")]
    max_panel_rows: usize,
}
fn default_panel_limit() -> usize {
    10_000
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    id: String,
    path: PathBuf,
    sha256: String,
    /// Caller-declared use; this is not a corpus-specific admission receipt.
    role: Role,
    /// Canonical name -> source column. Extra names may select metric columns.
    columns: BTreeMap<String, String>,
    #[serde(default = "higher")]
    target_direction: Direction,
    /// Explicit target range for canonical merged quality bands (no fitted range).
    target_range: Option<[f64; 2]>,
    /// Human judgments, objective teacher scores, or synthetic test values.
    target_kind: Option<TargetKind>,
    /// Direction of the quality-setting column (e.g. distortion level is lower).
    quality_direction: Option<Direction>,
    /// Optional difference-discrimination threshold in target units.
    difference_threshold: Option<f64>,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum TargetKind {
    Human,
    Objective,
    Synthetic,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    Synthetic,
    Train,
    Eval,
    PublicTest,
}
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Direction {
    Higher,
    Lower,
}
fn higher() -> Direction {
    Direction::Higher
}
impl Direction {
    fn orient(self, x: f64) -> f64 {
        match self {
            Self::Higher => x,
            Self::Lower => -x,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Metric {
    id: String,
    direction: Direction,
    /// Version/build/model and preprocessing/viewing-condition description.
    implementation: String,
    input_contract: String,
    /// Material rung change in this implementation's native score units.
    ladder_epsilon: Option<f64>,
    source: Source,
    /// Model/script/config dependencies in addition to the executable itself.
    #[serde(default)]
    artifacts: Vec<PinnedFile>,
}
#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Source {
    Column {
        column: String,
    },
    Table {
        path: PathBuf,
        sha256: String,
        id_column: String,
        score_column: String,
    },
    BatchCommand {
        command: CommandSpec,
    },
    Command {
        #[serde(flatten)]
        command: CommandSpec,
    },
    Zenmetrics {
        program: PathBuf,
        metric: String,
        /// Exact emitted score column; omission only allowed for one output.
        score_column: Option<String>,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default = "default_timeout")]
        timeout_seconds: u64,
    },
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CommandSpec {
    program: PathBuf,
    #[serde(default)]
    args: Vec<String>,
    /// RFC 6901 pointer in stdout JSON; empty means the root numeric value.
    #[serde(default)]
    json_pointer: String,
    #[serde(default = "default_timeout")]
    timeout_seconds: u64,
}
fn default_timeout() -> u64 {
    120
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PinnedFile {
    path: PathBuf,
    sha256: String,
}

#[derive(Deserialize, Serialize)]
struct Instrument {
    metric: String,
    dataset: String,
    criterion: String,
    #[serde(flatten)]
    source: InstrumentSource,
}
#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum InstrumentSource {
    Artifact { path: PathBuf, sha256: String },
    Command { command: CommandSpec },
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Requirement {
    metric: String,
    dataset: String,
    criterion: String,
    /// Pointer relative to the criterion's measurements (not its status).
    pointer: String,
    min: Option<f64>,
    max: Option<f64>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Comparison {
    dataset: String,
    a: String,
    b: String,
    bootstrap_resamples: usize,
    seed: u64,
}

const CRITERIA: &[&str] = &[
    "rank",
    "within_reference",
    "bands",
    "scatter",
    "pairwise",
    "dial",
    "severity_ramp",
    "corruption_ordering",
    "spatial",
    "rd",
    "targeting",
    "integrity",
    "hdr",
    "runtime",
    "correctness",
];
const INSTRUMENT_CRITERIA: &[&str] = &[
    "spatial",
    "rd",
    "targeting",
    "integrity",
    "hdr",
    "runtime",
    "correctness",
];

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn file_sha(path: &Path) -> Result<String> {
    zensim_validate::train_manifest::sha256_file(path).map_err(anyhow::Error::msg)
}
fn resolve(base: &Path, path: &Path) -> PathBuf {
    base.join(path)
}
fn verify_file(base: &Path, path: &Path, expected: &str) -> Result<PathBuf> {
    ensure!(
        expected.len() == 64 && expected.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid SHA256 for {}",
        path.display()
    );
    let path = resolve(base, path);
    ensure!(
        file_sha(&path)? == expected.to_ascii_lowercase(),
        "SHA256 mismatch: {}",
        path.display()
    );
    Ok(path)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)?;
    Ok(())
}
fn measured(values: Value) -> Value {
    json!({"state":"measured", "measurements":values})
}
fn missing(reason: impl Into<String>) -> Value {
    json!({"state":"not_measured", "reason":reason.into()})
}
fn failed(reason: impl Into<String>) -> Value {
    json!({"state":"failed", "reason":reason.into()})
}

fn validate(s: &Suite) -> Result<()> {
    ensure!(
        s.schema == "metric-evaluation-v1",
        "expected schema metric-evaluation-v1"
    );
    ensure!(
        !s.datasets.is_empty() && !s.metrics.is_empty(),
        "datasets and metrics must be nonempty"
    );
    ensure!(s.max_panel_rows >= 2, "max_panel_rows must be >=2");
    for ids in [
        s.datasets.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
        s.metrics.iter().map(|m| m.id.as_str()).collect(),
    ] {
        let mut seen = BTreeSet::new();
        for id in ids {
            ensure!(
                !id.is_empty()
                    && id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
                "IDs must contain only ASCII letters, digits, _ or -"
            );
            ensure!(seen.insert(id), "duplicate ID {id}");
        }
    }
    for d in &s.datasets {
        ensure!(
            !d.columns.contains_key("target") || d.target_kind.is_some(),
            "target_kind is required when a target column is declared"
        );
        ensure!(
            !d.columns.contains_key("quality") || d.quality_direction.is_some(),
            "quality_direction is required for ladders"
        );
        ensure!(
            d.difference_threshold
                .is_none_or(|v| v.is_finite() && v >= 0.0),
            "invalid difference_threshold"
        );
        ensure!(d.columns.contains_key("id"), "{} needs an id column", d.id);
        if let Some([lo, hi]) = d.target_range {
            ensure!(
                lo.is_finite() && hi.is_finite() && lo < hi,
                "invalid target_range"
            );
        }
    }
    for m in &s.metrics {
        ensure!(
            !m.implementation.trim().is_empty() && !m.input_contract.trim().is_empty(),
            "implementation and input_contract must be explicit"
        );
        ensure!(
            m.ladder_epsilon.is_none_or(|v| v.is_finite() && v >= 0.0),
            "ladder_epsilon must be finite and nonnegative"
        );
    }
    let exists = |m: &str, d: &str| {
        s.metrics.iter().any(|x| x.id == m) && s.datasets.iter().any(|x| x.id == d)
    };
    for c in &s.comparisons {
        ensure!(
            exists(&c.a, &c.dataset) && exists(&c.b, &c.dataset) && c.a != c.b,
            "invalid comparison IDs"
        );
        ensure!(
            c.bootstrap_resamples >= 4,
            "comparison requires at least four bootstrap resamples"
        );
    }
    let mut keys = BTreeSet::new();
    for i in &s.instruments {
        ensure!(
            exists(&i.metric, &i.dataset),
            "unknown instrument metric/dataset"
        );
        ensure!(
            INSTRUMENT_CRITERIA.contains(&i.criterion.as_str()),
            "unsupported instrument criterion {}",
            i.criterion
        );
        ensure!(
            keys.insert((&i.metric, &i.dataset, &i.criterion)),
            "duplicate instrument"
        );
    }
    for r in &s.requirements {
        ensure!(
            exists(&r.metric, &r.dataset) && CRITERIA.contains(&r.criterion.as_str()),
            "unknown requirement metric/dataset/criterion"
        );
        ensure!(
            r.min.is_some() || r.max.is_some(),
            "requirement needs min or max"
        );
        ensure!(
            !r.pointer.is_empty() && r.pointer.starts_with('/'),
            "requirement needs a JSON pointer"
        );
        ensure!(
            r.min.iter().chain(r.max.iter()).all(|v| v.is_finite()),
            "nonfinite threshold"
        );
        if let (Some(lo), Some(hi)) = (r.min, r.max) {
            ensure!(lo <= hi, "inverted thresholds");
        }
    }
    Ok(())
}

fn instrument_report(
    i: &Instrument,
    base: &Path,
    out: &Path,
    binding: &Value,
    scores: &Path,
    dataset: &Path,
) -> Result<Value> {
    let (value, provenance) = match &i.source {
        InstrumentSource::Artifact { path, sha256 } => {
            let path = verify_file(base, path, sha256)?;
            (
                serde_json::from_slice(&fs::read(&path)?)?,
                json!({"path":path,"sha256":sha256}),
            )
        }
        InstrumentSource::Command { command } => {
            let vars = BTreeMap::from([
                ("dataset".into(), dataset.display().to_string()),
                ("scores".into(), scores.display().to_string()),
                (
                    "metric_sha256".into(),
                    binding["metric_sha256"].as_str().unwrap().into(),
                ),
                (
                    "dataset_sha256".into(),
                    binding["dataset_sha256"].as_str().unwrap().into(),
                ),
                ("criterion".into(), i.criterion.clone()),
            ]);
            adapter::execute(command, base, &vars, out, "instrument")?
        }
    };
    ensure!(
        value["schema"] == "metric-instrument-v1" && value["criterion"] == i.criterion,
        "instrument schema/criterion mismatch"
    );
    for key in ["metric_sha256", "dataset_sha256"] {
        ensure!(value[key] == binding[key], "instrument {key} mismatch");
    }
    ensure!(
        value["n"].as_u64().is_some_and(|n| n > 0),
        "instrument needs a positive sample count"
    );
    ensure!(
        value["measurements"].is_object(),
        "instrument needs measurements object"
    );
    Ok(
        json!({"state":"measured", "measurements":value["measurements"], "n":value["n"], "provenance":provenance}),
    )
}

fn run(cli: Cli) -> Result<bool> {
    let manifest = fs::canonicalize(&cli.manifest)?;
    let base = manifest.parent().unwrap();
    let bytes = fs::read(&manifest)?;
    let suite: Suite = serde_json::from_slice(&bytes).context("parse evaluation manifest")?;
    validate(&suite)?;
    // Refuse accidental overwrite before any dataset read or adapter execution.
    fs::create_dir(&cli.output)
        .context("output must be a NEW directory with an existing parent")?;
    let out = fs::canonicalize(&cli.output)?;
    write_new(&out.join("manifest.json"), &bytes)?;
    let evaluator = std::env::current_exe()?;
    let evaluator_sha = file_sha(&evaluator)?;
    let mut runs = Vec::new();
    let mut comparisons = Vec::new();
    for d in &suite.datasets {
        let path = verify_file(base, &d.path, &d.sha256)?;
        let table = dataset::load(&path, &d.columns)?;
        let mut metric_scores = BTreeMap::new();
        for m in &suite.metrics {
            eprintln!(
                "panel evaluate: {} / {} ({} rows)",
                d.id,
                m.id,
                table.rows.len()
            );
            let dataset_dir = out.join(&d.id);
            if !dataset_dir.exists() {
                fs::create_dir(&dataset_dir)?;
            }
            let dir = dataset_dir.join(&m.id);
            fs::create_dir(&dir)?;
            let metric_identity = adapter::identity(m, base)?;
            let binding = json!({"metric_sha256":sha(&serde_json::to_vec(&metric_identity)?), "dataset_sha256":d.sha256});
            let scores = adapter::score(m, &table, base, &path, &dir)?;
            ensure!(
                adapter::identity(m, base)? == metric_identity,
                "metric executable/dependencies changed during scoring"
            );
            metric_scores.insert(
                m.id.clone(),
                scores
                    .iter()
                    .map(|s| s.value.map(|v| m.direction.orient(v)))
                    .collect::<Option<Vec<f64>>>(),
            );
            let scores_path = dir.join("scores.tsv");
            dataset::write_scores(&scores_path, &table, &scores)?;
            let mut criteria = measurements::evaluate(d, m, &table, &scores, suite.max_panel_rows)?;
            for i in suite
                .instruments
                .iter()
                .filter(|i| i.dataset == d.id && i.metric == m.id)
            {
                let instrument_dir = dir.join(&i.criterion);
                fs::create_dir(&instrument_dir)?;
                let result =
                    instrument_report(i, base, &instrument_dir, &binding, &scores_path, &path);
                criteria.insert(
                    i.criterion.clone(),
                    result.unwrap_or_else(|e| failed(format!("{e:#}"))),
                );
            }
            let errors = scores.iter().filter(|s| s.error.is_some()).count();
            let measured_count = criteria
                .values()
                .filter(|c| c["state"] == "measured")
                .count();
            runs.push(json!({"metric":m.id,"dataset":d.id,"role":d.role,"metric_identity":metric_identity,
                "binding":binding,"target_kind":d.target_kind,"rows":table.rows.len(),"score_failures":errors,"coverage":{"measured":measured_count,"total":CRITERIA.len()},"scores":{"path":scores_path,"sha256":file_sha(&scores_path)?},"criteria":criteria}));
        }
        ensure!(
            file_sha(&path)? == d.sha256.to_ascii_lowercase(),
            "dataset changed during evaluation"
        );
        for c in suite.comparisons.iter().filter(|c| c.dataset == d.id) {
            let report = if let (Some(a), Some(b)) = (&metric_scores[&c.a], &metric_scores[&c.b]) {
                if !table.has("target") {
                    missing("comparison needs a target column")
                } else if a.len() > suite.max_panel_rows {
                    missing("comparison exceeds max_panel_rows; no implicit sampling")
                } else {
                    let t: Vec<f64> = table
                        .numbers("target")?
                        .into_iter()
                        .map(|v| d.target_direction.orient(v))
                        .collect();
                    let ra = zensim_validate::panel::spearman(a, &t);
                    let rb = zensim_validate::panel::spearman(b, &t);
                    if !ra.is_finite() || !rb.is_finite() || ra < 0.0 || rb < 0.0 {
                        missing(
                            "undefined or inverted rank: inspect signed panels; legacy decisive rule auto-aligns polarity",
                        )
                    } else {
                        let v = zensim_validate::panel::decisive(
                            a,
                            b,
                            &t,
                            c.bootstrap_resamples,
                            c.seed,
                        );
                        measured(
                            json!({"n":v.n_band,"decision":v.decision.as_str(),"h_srocc":v.h_srocc,"p_srocc":v.p_srocc,"h_z_rmse":v.h_z_rmse,"p_z_rmse":v.p_z_rmse,"pwrc_diff":v.pwrc_diff,"ci_delta":v.ci_delta,"ci_order":["srocc","plcc","krocc","or","pwrc","z_rmse"],"bootstrap_unit":"row; not source-cluster confirmation","agreement_a":v.agreement_a,"agreement_b":v.agreement_b}),
                        )
                    }
                }
            } else {
                failed("comparison has failed score rows")
            };
            comparisons.push(json!({"spec":c,"report":report}));
        }
    }
    let mut checks = Vec::new();
    for r in &suite.requirements {
        let run = runs
            .iter()
            .find(|v| v["metric"] == r.metric && v["dataset"] == r.dataset)
            .unwrap();
        let criterion = &run["criteria"][&r.criterion];
        let value = criterion["measurements"]
            .pointer(&r.pointer)
            .and_then(Value::as_f64)
            .filter(|x| x.is_finite());
        let state = if criterion["state"] == "failed" {
            "fail"
        } else if criterion["state"] != "measured" || value.is_none() {
            "incomplete"
        } else if value
            .is_some_and(|v| r.min.is_none_or(|lo| v >= lo) && r.max.is_none_or(|hi| v <= hi))
        {
            "pass"
        } else {
            "fail"
        };
        checks.push(json!({"requirement":r,"value":value,"state":state}));
    }
    let has_errors = runs.iter().any(|r| {
        r["score_failures"].as_u64().unwrap() > 0
            || r["criteria"]
                .as_object()
                .unwrap()
                .values()
                .any(|c| c["state"] == "failed")
    });
    let pass = !has_errors && checks.iter().all(|c| c["state"] == "pass");
    let report = json!({"schema":"metric-evaluation-report-v1", "manifest_sha256":sha(&bytes),
        "evaluator":{"path":evaluator,"sha256":evaluator_sha}, "runs":runs,"checks":checks,"comparisons":comparisons,
        "status":if has_errors || checks.iter().any(|c| c["state"] == "fail") {"fail"} else if checks.iter().any(|c| c["state"] == "incomplete") {"incomplete"} else if checks.is_empty() {"measured"} else {"pass"},
        "scope":"Dataset diagnostics and explicitly declared requirements; not production qualification"});
    write_new(
        &out.join("report.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    let md = render(&report);
    write_new(&out.join("report.md"), md.as_bytes())?;
    write_new(
        &out.join("report.html"),
        zensim_validate::eval_report::markdown_to_html(&md, "Metric evaluation").as_bytes(),
    )?;
    println!("{}", out.join("report.json").display());
    Ok(pass)
}

fn render(report: &Value) -> String {
    let mut md = format!(
        "# Metric evaluation\n\nStatus: **{}**. {}.\n\n| Dataset | Metric | Criterion | State | Measurements / reason |\n|---|---|---|---|---|\n",
        report["status"].as_str().unwrap(),
        report["scope"].as_str().unwrap()
    );
    // Objective teachers and synthetic targets must stay visible in the human report.
    let labels: Vec<String> = report["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            format!(
                "{} / {}: target kind {}, measured {}/{} criteria",
                r["dataset"],
                r["metric"],
                r["target_kind"],
                r["coverage"]["measured"],
                r["coverage"]["total"]
            )
        })
        .collect();
    let table_header =
        "| Dataset | Metric | Criterion | State | Measurements / reason |\n|---|---|---|---|---|\n";
    md = md.replace(table_header, &format!("{}\n\nFull measurements and per-curve detail: [report.json](report.json).\n\n{table_header}", labels.join("; ")));
    for run in report["runs"].as_array().unwrap() {
        for (name, c) in run["criteria"].as_object().unwrap() {
            let summary = if c["state"] == "measured" {
                let compact: serde_json::Map<String, Value> = c["measurements"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| {
                        let value = match v {
                            Value::Array(a) => json!(format!("{} entries; see JSON", a.len())),
                            Value::Object(_) => json!("see JSON"),
                            _ => v.clone(),
                        };
                        (k.clone(), value)
                    })
                    .collect();
                Value::Object(compact).to_string()
            } else {
                c["reason"].as_str().unwrap_or("").into()
            };
            // Render untrusted dataset/adapter text as text, never raw HTML.
            let summary = summary
                .replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('|', "&#124;")
                .replace(['\n', '\r'], " ");
            md.push_str(&format!(
                "| {} | {} | {name} | {} | {summary} |\n",
                run["dataset"].as_str().unwrap(),
                run["metric"].as_str().unwrap(),
                c["state"].as_str().unwrap()
            ));
        }
    }
    md.push_str("\n## Requirements\n\n```json\n");
    for c in report["checks"].as_array().unwrap() {
        md.push_str(&format!(
            "{}\n",
            serde_json::to_string(c).unwrap().replace('`', "\\u0060")
        ));
    }
    md.push_str("```\n\n## Paired comparisons\n\n```json\n");
    for c in report["comparisons"].as_array().unwrap() {
        md.push_str(&format!(
            "{}\n",
            serde_json::to_string(c).unwrap().replace('`', "\\u0060")
        ));
    }
    md.push_str("```\n");
    md
}

pub(super) fn main() -> ExitCode {
    let cli = Cli::parse_from(
        std::iter::once("panel evaluate".to_string()).chain(std::env::args().skip(2)),
    );
    match run(cli) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("panel evaluate: {e:#}");
            ExitCode::from(2)
        }
    }
}
