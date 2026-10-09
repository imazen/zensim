use super::*;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub(super) struct Score {
    pub(super) value: Option<f64>,
    pub(super) error: Option<String>,
    pub(super) provenance: Value,
}

fn executable(base: &Path, program: &Path) -> Result<PathBuf> {
    // Explicit path: resolving PATH differently between identity and execution
    // would score a different implementation under a plausible hash.
    let path = fs::canonicalize(resolve(base, program)).context(
        "program must be an explicit executable path (relative to manifest or absolute)",
    )?;
    ensure!(path.is_file(), "program is not a file");
    Ok(path)
}

pub(super) fn identity(metric: &Metric, base: &Path) -> Result<Value> {
    let program = match &metric.source {
        Source::Command { command } | Source::BatchCommand { command } => Some(&command.program),
        Source::Zenmetrics { program, .. } => Some(program),
        Source::Column { .. } | Source::Table { .. } => None,
    };
    let executable = program
        .map(|p| -> Result<Value> {
            let path = executable(base, p)?;
            Ok(json!({"path":path,"sha256":file_sha(&path)?}))
        })
        .transpose()?;
    let mut artifacts = Vec::new();
    if let Source::Table { path, sha256, .. } = &metric.source {
        let path = verify_file(base, path, sha256)?;
        artifacts.push(json!({"path":path,"sha256":sha256}));
    }
    for a in &metric.artifacts {
        let path = verify_file(base, &a.path, &a.sha256)?;
        artifacts.push(json!({"path":path,"sha256":a.sha256}));
    }
    Ok(json!({"metric":metric,"executable":executable,"artifacts":artifacts}))
}

fn expand(arg: &str, vars: &BTreeMap<String, String>) -> Result<String> {
    let mut result = String::new();
    let mut rest = arg;
    while let Some(start) = rest.find('{') {
        result.push_str(&rest[..start]);
        let end = rest[start..]
            .find('}')
            .context("unclosed adapter placeholder")?
            + start;
        let name = &rest[start + 1..end];
        result.push_str(
            vars.get(name)
                .with_context(|| format!("unknown/unavailable adapter placeholder {{{name}}}"))?,
        );
        rest = &rest[end + 1..];
    }
    result.push_str(rest);
    Ok(result)
}

pub(super) fn execute(
    spec: &CommandSpec,
    base: &Path,
    vars: &BTreeMap<String, String>,
    dir: &Path,
    stem: &str,
) -> Result<(Value, Value)> {
    ensure!(spec.timeout_seconds > 0, "timeout_seconds must be positive");
    let program = executable(base, &spec.program)?;
    let program_sha = file_sha(&program)?;
    let args = spec
        .args
        .iter()
        .map(|s| expand(s, vars))
        .collect::<Result<Vec<_>>>()?;
    let stdout_path = dir.join(format!("{stem}.stdout"));
    let stderr_path = dir.join(format!("{stem}.stderr"));
    let stdout = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stdout_path)?;
    let stderr = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stderr_path)?;
    let mut child = Command::new(&program)
        .args(&args)
        .current_dir(base)
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .spawn()?;
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() >= Duration::from_secs(spec.timeout_seconds)
            || fs::metadata(&stdout_path)?.len() > 1_048_576
            || fs::metadata(&stderr_path)?.len() > 1_048_576
        {
            let _ = child.kill();
            let _ = child.wait();
            bail!(
                "adapter exceeded timeout or 1 MiB output limit; see {}",
                stderr_path.display()
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    ensure!(
        status.success(),
        "adapter exit {status}; see {}",
        stderr_path.display()
    );
    ensure!(
        fs::metadata(&stdout_path)?.len() <= 1_048_576
            && fs::metadata(&stderr_path)?.len() <= 1_048_576,
        "adapter output exceeds 1 MiB per stream"
    );
    ensure!(
        file_sha(&program)? == program_sha,
        "executable changed during scoring"
    );
    let value: Value = serde_json::from_slice(&fs::read(&stdout_path)?)
        .context("adapter stdout must be one JSON value")?;
    Ok((
        value,
        json!({"program":program,"program_sha256":program_sha,"args":args,"stdout":stdout_path,"stderr":stderr_path,
        "stdout_sha256":file_sha(&stdout_path)?,"wall_seconds":start.elapsed().as_secs_f64(),"timing_scope":"whole subprocess including startup/decode; not qualified runtime"}),
    ))
}

pub(super) fn score(
    metric: &Metric,
    table: &dataset::Table,
    base: &Path,
    dataset_path: &Path,
    out: &Path,
) -> Result<Vec<Score>> {
    let data_dir = dataset_path.parent().unwrap();
    let keyed = match &metric.source {
        Source::Table {
            path,
            sha256,
            id_column,
            score_column,
        } => {
            let path = verify_file(base, path, sha256)?;
            let table = dataset::load(
                &path,
                &BTreeMap::from([
                    ("id".into(), id_column.clone()),
                    ("score".into(), score_column.clone()),
                ]),
            )?;
            let pairs = table
                .rows
                .iter()
                .map(|r| (r["id"].clone(), r["score"].parse::<f64>().ok()))
                .collect::<Vec<_>>();
            Some((pairs, json!({"path":path,"sha256":sha256})))
        }
        Source::BatchCommand { command } => {
            let vars = BTreeMap::from([("dataset".into(), dataset_path.display().to_string())]);
            let (value, provenance) = execute(command, base, &vars, out, "batch")?;
            let rows = value["scores"]
                .as_array()
                .context("batch adapter needs scores array")?;
            let pairs = rows
                .iter()
                .map(|r| {
                    Ok((
                        r["id"]
                            .as_str()
                            .context("batch row needs string id")?
                            .into(),
                        r["score"].as_f64(),
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Some((pairs, provenance))
        }
        _ => None,
    };
    if let Some((pairs, provenance)) = keyed {
        let expected: BTreeSet<&str> = table.rows.iter().map(|r| r["id"].as_str()).collect();
        let mut keyed = BTreeMap::new();
        for (id, value) in pairs {
            ensure!(expected.contains(id.as_str()), "unknown score row id {id}");
            ensure!(
                keyed.insert(id.clone(), value).is_none(),
                "duplicate score row id {id}"
            );
        }
        return Ok(table
            .rows
            .iter()
            .map(|row| {
                let value = keyed
                    .get(&row["id"])
                    .copied()
                    .flatten()
                    .filter(|v| v.is_finite());
                Score {
                    value,
                    error: value
                        .is_none()
                        .then(|| format!("missing or nonfinite score for {}", row["id"])),
                    provenance: provenance.clone(),
                }
            })
            .collect());
    }
    let mut scores = Vec::new();
    for (i, row) in table.rows.iter().enumerate() {
        let mut provenance = Value::Null;
        let result = (|| -> Result<f64> {
            let spec = match &metric.source {
                Source::Table { .. } | Source::BatchCommand { .. } => {
                    unreachable!("keyed adapters returned above")
                }
                Source::Column { column } => {
                    let v = row
                        .get(column)
                        .with_context(|| format!("missing mapped score column {column}"))?
                        .parse::<f64>()?;
                    ensure!(v.is_finite(), "nonfinite score");
                    return Ok(v);
                }
                Source::Command { command } => CommandSpec {
                    program: command.program.clone(),
                    args: command.args.clone(),
                    json_pointer: command.json_pointer.clone(),
                    timeout_seconds: command.timeout_seconds,
                },
                Source::Zenmetrics {
                    program,
                    metric,
                    args,
                    timeout_seconds,
                    ..
                } => {
                    let mut command_args = vec![
                        "score".into(),
                        "--metric".into(),
                        metric.clone(),
                        "--reference".into(),
                        "{reference}".into(),
                        "--distorted".into(),
                        "{distorted}".into(),
                        "--output".into(),
                        "json".into(),
                    ];
                    command_args.extend(args.clone());
                    CommandSpec {
                        program: program.clone(),
                        args: command_args,
                        json_pointer: String::new(),
                        timeout_seconds: *timeout_seconds,
                    }
                }
            };
            let mut vars = BTreeMap::from([("id".into(), row["id"].clone())]);
            let mut inputs = Vec::new();
            for key in ["reference", "distorted"] {
                if let Some(path) = row.get(key) {
                    let path = fs::canonicalize(resolve(data_dir, Path::new(path)))
                        .with_context(|| format!("missing {key} input"))?;
                    inputs.push(json!({"role":key,"path":path,"sha256":file_sha(&path)?}));
                    vars.insert(key.into(), path.display().to_string());
                }
            }
            let (value, mut invocation) = execute(&spec, base, &vars, out, &format!("row-{i}"))?;
            for input in &inputs {
                ensure!(
                    file_sha(Path::new(input["path"].as_str().unwrap()))? == input["sha256"],
                    "image input changed during scoring"
                );
            }
            invocation["inputs"] = json!(inputs);
            provenance = invocation;
            let result = match &metric.source {
                Source::Zenmetrics { score_column, .. } => {
                    let values = value["scores"]
                        .as_object()
                        .context("zenmetrics output needs scores object")?;
                    if let Some(column) = score_column {
                        values
                            .get(column)
                            .with_context(|| format!("missing zenmetrics score column {column}"))?
                    } else {
                        ensure!(
                            values.len() == 1,
                            "multiple zenmetrics outputs: declare score_column"
                        );
                        values.values().next().unwrap()
                    }
                }
                _ => value
                    .pointer(&spec.json_pointer)
                    .context("missing score JSON pointer")?,
            };
            let score = result
                .as_f64()
                .context("score must be a finite JSON number")?;
            ensure!(score.is_finite(), "nonfinite score");
            Ok(score)
        })();
        scores.push(match result {
            Ok(value) => Score {
                value: Some(value),
                error: None,
                provenance,
            },
            Err(e) => Score {
                value: None,
                error: Some(format!("row {}: {e:#}", row["id"])),
                provenance,
            },
        });
    }
    Ok(scores)
}
