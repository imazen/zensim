//! E34 orchestration: statistics are evaluated by the pinned zenmetrics-stats.
//! Inputs are explicit paired ten-seed panels; no corpus discovery or fitting.
use anyhow::{Result, ensure};
use clap::Parser;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use zenstats::resample::{BootstrapConfig, Design, Draw, Interval, cluster_bootstrap};

const DATASETS: [&str; 4] = ["kadid", "tid2013", "konfig", "cid22_a25"];
#[derive(Parser)]
struct Cli {
    #[arg(value_parser = ["seed-parity", "noise", "evaluate"])]
    mode: String,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn save(path: &Path, value: &Value) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    writeln!(f, "{}", serde_json::to_string_pretty(value)?)?;
    Ok(())
}
fn seed_stat(matrix: &[Vec<f64>]) -> Result<Value> {
    ensure!(
        matrix.len() == 10
            && matrix
                .iter()
                .all(|r| r.len() == 4 && r.iter().all(|v| v.is_finite())),
        "ten complete four-source seed units required"
    );
    let seeds: Vec<_> = matrix
        .iter()
        .map(|r| zenstats::resample::weighted_mean(r, &[1.; 4]))
        .collect();
    let mean = seeds.iter().sum::<f64>() / 10.;
    let se = (seeds.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 90.).sqrt();
    ensure!(se.is_finite() && se > 0., "undefined seed SE");
    Ok(json!({"delta":mean,"se":se,"n":10,"seed_deltas":seeds}))
}
fn matrix(v: &Value) -> Result<Vec<Vec<f64>>> {
    Ok(serde_json::from_value(v.clone())?)
}
fn seed_parity(bytes: &[u8]) -> Result<Value> {
    ensure!(
        digest(bytes) == "4a3c58b53d6da2ff46c3ef504e78067bff55a8c7a8fb54a1d55c7aa045bf6bf2",
        "not pinned E33 summary"
    );
    let doc: Value = serde_json::from_slice(bytes)?;
    let mut out = json!({});
    for arm in ["a", "c"] {
        let old = &doc["decisions"][arm];
        let signed = seed_stat(&matrix(&old["source_seed_deltas"])?)?;
        let w2: Vec<Vec<f64>> = matrix(&old["w2_source_seed_deltas"])?;
        ensure!(
            w2.len() == 10 && w2.iter().all(|r| r.len() == 2),
            "W2 shape"
        );
        let w2 = seed_stat(
            &w2.iter()
                .map(|r| vec![(r[0] + r[1]) / 2.; 4])
                .collect::<Vec<_>>(),
        )?;
        out[arm] = json!({"signed":signed,"w2":w2});
    }
    let stat = seed_stat(&matrix(&doc["c_vs_a"]["source_seed_deltas"])?)?;
    let t = stat["delta"].as_f64().unwrap() / stat["se"].as_f64().unwrap();
    out["c_vs_a"] =
        json!({"signed":stat,"t":t,"df":9,"p_one_sided":zenstats::student_t_upper_tail(t,9.)});
    Ok(out)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    datasets: Vec<Dataset>,
    tests: Vec<Test>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    name: String,
    target: Vec<f64>,
    reference: Vec<u32>,
    codec: Vec<u32>,
    /// E21 fixed direction, never inferred from predictions.
    orientation: f64,
    predictions: BTreeMap<String, Vec<Vec<f64>>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Test {
    index: u64,
    arm: String,
    control: String,
    eligibility: bool,
}
struct Rows {
    dataset: Vec<usize>,
    sample: Vec<usize>,
    reference: Vec<u32>,
    seed: Vec<u32>,
}
impl Input {
    fn validate(&self, noise: bool) -> Result<()> {
        ensure!(self.schema == "e34-paired-seed-v1", "input schema");
        ensure!(
            self.datasets
                .iter()
                .map(|d| d.name.as_str())
                .collect::<Vec<_>>()
                == DATASETS,
            "four registered datasets in order"
        );
        ensure!(
            (noise && self.tests.len() == 2) || (!noise && self.tests.len() == 9),
            "fixed test inventory"
        );
        for (k, t) in self.tests.iter().enumerate() {
            ensure!(t.index == k as u64 + 1, "test offsets must be ordered");
        }
        if !noise {
            let expected = [
                ("w-a", "base-a", true),
                ("k-a", "base-a", true),
                ("l-a", "base-a", true),
                ("w-c", "base-c", true),
                ("k-c", "base-c", true),
                ("k-a", "w-a", false),
                ("l-a", "w-a", false),
                ("l-a", "k-a", false),
                ("k-c", "w-c", false),
            ];
            ensure!(
                self.tests
                    .iter()
                    .zip(expected)
                    .all(|(t, (a, c, e))| t.arm == a && t.control == c && t.eligibility == e),
                "registered nine-test order"
            );
        }
        for d in &self.datasets {
            let n = d.target.len();
            ensure!(
                n >= 3 && d.reference.len() == n && d.codec.len() == n,
                "aligned keys"
            );
            ensure!(
                d.orientation == 1. || d.orientation == -1.,
                "fixed orientation required"
            );
            ensure!(d.target.iter().all(|v| v.is_finite()), "finite targets");
            for p in d.predictions.values() {
                ensure!(
                    p.len() == 10
                        && p.iter()
                            .all(|v| v.len() == n && v.iter().all(|x| x.is_finite())),
                    "ten complete paired prediction rows"
                );
            }
        }
        Ok(())
    }
    fn rows(&self, subset: &[usize]) -> Rows {
        let mut r = Rows {
            dataset: vec![],
            sample: vec![],
            reference: vec![],
            seed: vec![],
        };
        let mut refs = BTreeMap::new();
        for &di in subset {
            let d = &self.datasets[di];
            for s in 0..10 {
                for (i, reference) in d.reference.iter().enumerate() {
                    let next = refs.len() as u32;
                    let reference = *refs.entry((di, *reference)).or_insert(next);
                    r.dataset.push(di);
                    r.sample.push(i);
                    r.reference.push(reference);
                    r.seed.push(s);
                }
            }
        }
        r
    }
    fn statistic(
        &self,
        rows: &Rows,
        draw: &Draw<'_>,
        t: &Test,
        w2: bool,
        subset: &[usize],
    ) -> Vec<f64> {
        let expanded = draw.expand();
        let mut groups: BTreeMap<(usize, u64), Vec<usize>> = BTreeMap::new();
        let mut refs: BTreeMap<usize, BTreeSet<u64>> = BTreeMap::new();
        for (j, &ri) in expanded.rows.iter().enumerate() {
            let di = rows.dataset[ri];
            // In OneWay(reference), seed indices remain ten fixed distinct levels.
            let seed = if expanded.groups.len() == 2 {
                expanded.groups[1][j]
            } else {
                u64::from(rows.seed[ri])
            };
            groups.entry((di, seed)).or_default().push(ri);
            refs.entry(di).or_default().insert(expanded.groups[0][j]);
        }
        if subset
            .iter()
            .any(|di| refs.get(di).is_none_or(|r| r.len() < 3))
        {
            return vec![f64::NAN];
        }
        let mut deltas = Vec::new();
        for &di in subset {
            let d = &self.datasets[di];
            let mut seed_deltas = Vec::new();
            for ((dataset, _), indices) in &groups {
                if *dataset != di {
                    continue;
                }
                let s = rows.seed[indices[0]] as usize;
                let score = |model: &str| {
                    if w2 {
                        let mut types: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
                        for &ri in indices {
                            types
                                .entry(d.codec[rows.sample[ri]])
                                .or_default()
                                .push(rows.sample[ri]);
                        }
                        let mut values: Vec<_> = types
                            .values()
                            .map(|idx| {
                                let p: Vec<_> =
                                    idx.iter().map(|&i| d.predictions[model][s][i]).collect();
                                let y: Vec<_> = idx.iter().map(|&i| d.target[i]).collect();
                                d.orientation * zenstats::spearman(&p, &y)
                            })
                            .collect();
                        if values.len() < 3 || values.iter().any(|v| !v.is_finite()) {
                            return f64::NAN;
                        }
                        values.sort_by(f64::total_cmp);
                        values[..3].iter().sum::<f64>() / 3.
                    } else {
                        let p: Vec<_> = indices
                            .iter()
                            .map(|&ri| d.predictions[model][s][rows.sample[ri]])
                            .collect();
                        let y: Vec<_> = indices
                            .iter()
                            .map(|&ri| d.target[rows.sample[ri]])
                            .collect();
                        d.orientation * zenstats::spearman(&p, &y)
                    }
                };
                seed_deltas.push(score(&t.arm) - score(&t.control));
            }
            if seed_deltas.is_empty() || seed_deltas.iter().any(|v| !v.is_finite()) {
                return vec![f64::NAN];
            }
            deltas.push(seed_deltas.iter().sum::<f64>() / seed_deltas.len() as f64);
        }
        vec![deltas.iter().sum::<f64>() / subset.len() as f64]
    }
    fn interval(
        &self,
        t: &Test,
        subset: &[usize],
        offset: u64,
        reference_only: bool,
        w2: bool,
        margin: f64,
    ) -> Result<Interval> {
        let rows = self.rows(subset);
        let design = if reference_only {
            Design::OneWay(&rows.reference)
        } else {
            Design::TwoWay(&rows.reference, &rows.seed)
        };
        let cfg = BootstrapConfig {
            resamples: 10000,
            seed: 20261011 + offset,
            level: 0.95,
        };
        let mut iv = cluster_bootstrap(rows.sample.len(), &design, &cfg, |draw| {
            let mut v = self.statistic(&rows, draw, t, w2, subset);
            v[0] += margin;
            v
        })?
        .remove(0);
        // Keep the interval on theta when a margin is supplied; callers label it.
        if !iv.estimate.is_finite() {
            iv.p_nonpositive = 1.;
        }
        Ok(iv)
    }
}
fn interval_json(iv: &Interval, groups: usize) -> Value {
    json!({"estimate":iv.estimate,"lo":iv.lo,"hi":iv.hi,"se":iv.se,"n_finite":iv.n_finite,
        "p_nonpositive":iv.p_nonpositive,"student_interval":iv.student(groups,0.95)})
}
fn decision(iv: &Interval, g: usize) -> Value {
    let corrected_se = iv.se * (g as f64 / (g - 1) as f64).sqrt();
    let complete =
        iv.estimate.is_finite() && corrected_se.is_finite() && corrected_se > 0. && iv.n_finite > 0;
    let p_student = if complete {
        zenstats::student_t_upper_tail(iv.estimate / corrected_se, (g - 1) as f64)
    } else {
        1.
    };
    let p_boot = if complete { iv.p_nonpositive } else { 1. };
    json!({"complete":complete,"bootstrap":p_boot,"student":p_student,"decision":p_boot.max(p_student),"df":g-1,"student_se":corrected_se})
}
fn seed_component(input: &Input, t: &Test) -> Result<Value> {
    let matrix: Vec<Vec<f64>> = (0..10)
        .map(|seed| {
            input
                .datasets
                .iter()
                .map(|d| {
                    d.orientation
                        * (zenstats::spearman(&d.predictions[&t.arm][seed], &d.target)
                            - zenstats::spearman(&d.predictions[&t.control][seed], &d.target))
                })
                .collect()
        })
        .collect();
    Ok(seed_stat(&matrix).unwrap_or_else(|e| json!({"state":"INCOMPLETE","reason":e.to_string()})))
}

fn pair_classes(input: &Input, t: &Test) -> Result<Value> {
    use zenstats::pairs::{PairClass, PairInput, PairTable};
    use zenstats::panel::Orientation;
    let mut output = Vec::new();
    for d in &input.datasets {
        let design = Design::OneWay(&d.reference);
        let orientation = if d.orientation > 0. {
            Orientation::HigherIsBetter
        } else {
            Orientation::LowerIsBetter
        };
        let tables: Vec<_> = (0..10)
            .map(|seed| {
                let make = |model: &str| {
                    PairTable::new(
                        &PairInput {
                            predicted: &d.predictions[model][seed],
                            target: &d.target,
                            source: &d.reference,
                            codec: &d.codec,
                            family: None,
                            draw: None,
                            orientation,
                        },
                        &design,
                        0.,
                        None,
                    )
                };
                Ok((make(&t.arm)?, make(&t.control)?))
            })
            .collect::<Result<_>>()?;
        for (c, class) in [
            (1, PairClass::SameSourceSameCodec),
            (2, PairClass::SameSourceCrossCodec),
            (3, PairClass::CrossSource),
        ] {
            let cfg = BootstrapConfig {
                resamples: 10000,
                seed: 20261011 + 1000 + 10 * t.index + c,
                level: 0.95,
            };
            let iv = cluster_bootstrap(d.target.len(), &design, &cfg, |draw| {
                let mut values = Vec::new();
                for (a, b) in &tables {
                    let a = a
                        .class_agreement(draw)
                        .expect("validated reference design")
                        .get(class);
                    let b = b
                        .class_agreement(draw)
                        .expect("validated reference design")
                        .get(class);
                    values.push(a.agreement() - b.agreement());
                }
                let mean = values.iter().sum::<f64>() / 10.;
                let se = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 90.).sqrt();
                let counts = tables[0]
                    .0
                    .class_agreement(draw)
                    .expect("validated reference design")
                    .get(class);
                vec![mean, se, counts.pairs, counts.target_ties]
            })?;
            output.push(json!({"dataset":d.name,"class":c,"interval":interval_json(&iv[0],d.reference.iter().collect::<BTreeSet<_>>().len()),
                "seed_unit_se":iv[1].estimate,"pairs":iv[2].estimate,"excluded_target_ties":iv[3].estimate,
                "target_tie":0,"orientation":d.orientation,"family_labels":false,
                "seed":cfg.seed,"resamples":cfg.resamples,"finite_draws":iv[0].n_finite}));
        }
    }
    Ok(json!(output))
}

fn evaluate(input: &Input, noise: bool) -> Result<Value> {
    if let Err(error) = input.validate(noise) {
        let count = if noise { 2 } else { 9 };
        let ps = vec![1.; count];
        return Ok(json!({"state":"INCOMPLETE","reason":error.to_string(),
            "tests":(1..=count).map(|i|json!({"test":i,"state":"INCOMPLETE","p":{"decision":1.}})).collect::<Vec<_>>(),
            "holm_adjusted":if noise {Value::Null}else{json!(zenstats::multiple::holm(&ps))},
            "holm_reject":if noise {Value::Null}else{json!(zenstats::multiple::holm_reject(&ps,0.05))}}));
    }
    let mut out = Vec::new();
    let mut ps = Vec::new();
    for t in &input.tests {
        if input.datasets.iter().any(|d| {
            [&t.arm, &t.control].iter().any(|m| {
                d.predictions.get(*m).is_none_or(|p| {
                    p.len() != 10
                        || p.iter()
                            .any(|v| v.len() != d.target.len() || v.iter().any(|x| !x.is_finite()))
                })
            })
        }) {
            ps.push(1.);
            out.push(json!({"test":t.index,"arm":t.arm,"control":t.control,
                "state":"INCOMPLETE","p":{"decision":1.},
                "reason":"ten complete paired seed panels required"}));
            continue;
        }
        eprintln!("E34 test {} {} vs {}", t.index, t.arm, t.control);
        let margin = if t.eligibility { 0.002 } else { 0. };
        let primary = input.interval(t, &[0, 1, 2, 3], 100 * t.index, false, false, margin)?;
        let reference = input.interval(t, &[0, 1, 2, 3], 100 * t.index + 6, true, false, 0.)?;
        let p = decision(&primary, 10);
        let seed_component = seed_component(input, t)?;
        let q9: Vec<_> = [(1, 0.05 / 9.), (5, 0.05 / 5.), (9, 0.05)]
            .into_iter()
            .map(|(step, alpha)| {
                let critical = zenstats::student_t_quantile(1. - alpha, 9.);
                json!({"holm_step":step,"alpha":alpha,"t_critical":critical,
                "equal_arm_max_two_way_se":0.002/(critical*(10_f64/9.).sqrt()),
                "required_delta_at_measured_se":critical*primary.se*(10_f64/9.).sqrt()-0.002})
            })
            .collect();
        ps.push(p["decision"].as_f64().unwrap());
        let mut guards = Vec::new();
        if !noise {
            for (di, name) in DATASETS.iter().enumerate() {
                let iv =
                    input.interval(t, &[di], 100 * t.index + di as u64 + 1, false, false, 0.)?;
                let n = input.datasets[di]
                    .reference
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    .min(10);
                guards.push(json!({"dataset":name,"interval":interval_json(&iv,n),"point_guard":iv.estimate>= -0.005}));
            }
        }
        let w2 = if noise {
            Value::Null
        } else {
            let iv = input.interval(t, &[0, 1], 100 * t.index + 5, false, true, 0.)?;
            json!({"interval":interval_json(&iv,10),"point_guard":iv.estimate> -2.*iv.se})
        };
        out.push(json!({"test":t.index,"arm":t.arm,"control":t.control,"margin":margin,
            "theta":interval_json(&primary,10),"seed_unit":seed_component,"q9":q9,"reference_only":interval_json(&reference,10),"p":p,"dataset_guards":guards,"w2":w2,
            "pair_classes":if noise || !t.eligibility {Value::Null}else{pair_classes(input,t)?},
            "improvement_point_guard":primary.estimate>0.002}));
    }
    Ok(
        json!({"schema":"e34-statistics-v1","report_only":noise,"resamples":10000,"level":0.95,
        "percentile":"Hyndman-Fan type 7","base_seed":20261011,"tests":out,
        "holm_adjusted":if noise {Value::Null}else{json!(zenstats::multiple::holm(&ps))},
        "holm_reject":if noise {Value::Null}else{json!(zenstats::multiple::holm_reject(&ps,0.05))}}),
    )
}
fn run(cli: Cli) -> Result<()> {
    ensure!(!cli.output.exists(), "fresh output required");
    let bytes = fs::read(&cli.input)?;
    let mut result = if cli.mode == "seed-parity" {
        seed_parity(&bytes)?
    } else {
        evaluate(&serde_json::from_slice(&bytes)?, cli.mode == "noise")?
    };
    result["input_sha256"] = json!(digest(&bytes));
    save(&cli.output, &result)
}
pub(super) fn main() -> ExitCode {
    let cli =
        Cli::parse_from(std::iter::once("panel e34".to_string()).chain(std::env::args().skip(2)));
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("panel e34: {e:#}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn toy() -> Input {
        Input {
            schema: "e34-paired-seed-v1".into(),
            datasets: DATASETS
                .iter()
                .map(|name| Dataset {
                    name: (*name).into(),
                    target: (0..21).map(f64::from).collect(),
                    reference: (0..21).map(|i| i / 3).collect(),
                    codec: (0..21).map(|i| i % 3).collect(),
                    orientation: 1.,
                    predictions: BTreeMap::from([
                        (
                            "a".into(),
                            (0..10)
                                .map(|s| {
                                    (0..21)
                                        .map(|i| {
                                            if s % 2 == 0 {
                                                f64::from(i)
                                            } else {
                                                f64::from(20 - i)
                                            }
                                        })
                                        .collect()
                                })
                                .collect(),
                        ),
                        (
                            "b".into(),
                            (0..10)
                                .map(|s| {
                                    (0..21)
                                        .map(|i| {
                                            if s % 2 == 0 {
                                                f64::from(i)
                                            } else {
                                                f64::from(20 - i)
                                            }
                                        })
                                        .collect()
                                })
                                .collect(),
                        ),
                    ]),
                })
                .collect(),
            tests: (1..=2)
                .map(|index| Test {
                    index,
                    arm: "a".into(),
                    control: "b".into(),
                    eligibility: true,
                })
                .collect(),
        }
    }
    #[test]
    fn same_seed_pairing_cancels_and_reference_ids_are_dataset_unique() {
        let input = toy();
        input.validate(true).unwrap();
        let rows = input.rows(&[0, 1, 2, 3]);
        assert_eq!(rows.reference.iter().collect::<BTreeSet<_>>().len(), 28);
        let cfg = BootstrapConfig {
            resamples: 128,
            seed: 42,
            level: 0.95,
        };
        let design = Design::TwoWay(&rows.reference, &rows.seed);
        let iv = cluster_bootstrap(rows.sample.len(), &design, &cfg, |d| {
            input.statistic(&rows, d, &input.tests[0], false, &[0, 1, 2, 3])
        })
        .unwrap();
        assert_eq!(iv[0].estimate, 0.);
        assert_eq!(iv[0].se, 0.);
        assert_eq!(iv[0].p_nonpositive, 1.);
        let mut shifted = input;
        for d in &mut shifted.datasets {
            d.predictions.get_mut("b").unwrap().rotate_left(1);
        }
        let iv = cluster_bootstrap(rows.sample.len(), &design, &cfg, |d| {
            shifted.statistic(&rows, d, &shifted.tests[0], false, &[0, 1, 2, 3])
        })
        .unwrap();
        assert!(iv[0].se > 0.1, "seed-pair shifts must be visible");
    }
    #[test]
    fn student_rule_and_incomplete_are_conservative() {
        let iv = Interval {
            estimate: 0.002,
            lo: 0.,
            hi: 0.004,
            se: 0.0007,
            n_finite: 10000,
            p_zero: 0.002,
            p_nonpositive: 0.001,
        };
        let p = decision(&iv, 10);
        let expected = zenstats::student_t_upper_tail(0.002 / (0.0007 * (10_f64 / 9.).sqrt()), 9.);
        assert_eq!(p["student"].as_f64().unwrap(), expected);
        assert_eq!(p["decision"].as_f64().unwrap(), expected.max(0.001));
        let mut missing = iv;
        missing.n_finite = 0;
        assert_eq!(decision(&missing, 10)["decision"], 1.);
        let nine = vec![0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 1.];
        assert!(!zenstats::multiple::holm_reject(&nine, 0.05)[8]);
    }
    #[test]
    fn sample_seed_se_closed_form_and_input_refusals() {
        let matrix: Vec<Vec<f64>> = (0..10).map(|i| vec![f64::from(i) - 4.5; 4]).collect();
        let got = seed_stat(&matrix).unwrap();
        assert_eq!(got["delta"], 0.);
        assert_eq!(got["se"].as_f64().unwrap(), (82.5_f64 / 90.).sqrt());
        assert!(seed_stat(&matrix[..9]).is_err());
        let mut input = toy();
        input.datasets[0].predictions.get_mut("a").unwrap().pop();
        assert!(input.validate(true).is_err());
        let result = evaluate(&input, true).unwrap();
        assert_eq!(result["state"], "INCOMPLETE");
        assert_eq!(result["tests"][0]["p"]["decision"], 1.);
    }
}
