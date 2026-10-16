//! Conditional reports over frozen source-held-out networks and maps.
//! No map fitting, model selection or acceptance reads.
use anyhow::{Result, ensure};
use clap::Parser;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};
use zenstats::{
    pairs::{PairInput, class_agreement},
    panel::Orientation,
    resample::{BootstrapConfig, Design, Draw, Interval, cluster_bootstrap},
};
const MODELS: [&str; 8] = [
    "base-a", "p-a", "h-a", "ph-a", "base-c", "p-c", "h-c", "ph-c",
];
const METRICS: [&str; 5] = [
    "pooled-rmse",
    "equal-source-rmse",
    "same-ladder-ge-half",
    "cross-codec-ge-half",
    "all-within-source",
];
const PAIRS: [(usize, usize); 12] = [
    (1, 0),
    (2, 0),
    (3, 0),
    (2, 1),
    (3, 1),
    (3, 2),
    (5, 4),
    (6, 4),
    (7, 4),
    (6, 5),
    (7, 5),
    (7, 6),
];
#[derive(Parser)]
struct Cli {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    input_sha256: String,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    sources: Vec<u32>,
    rows: Vec<Row>,
    predictions: BTreeMap<String, Vec<Vec<Prediction>>>,
    frozen: Vec<Frozen>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    observation: String,
    source: u32,
    codec: String,
    ladder: String,
    target: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Prediction {
    quality: f64,
    jnd: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frozen {
    model: String,
    source: u32,
    seed_index: usize,
    network_sha256: String,
    map_sha256: String,
    map_fit_sources: Vec<u32>,
}
fn hash(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
impl Input {
    fn validate(&self) -> Result<()> {
        let sources: BTreeSet<_> = self.sources.iter().copied().collect();
        ensure!(
            self.schema == "registered-frozen-loso-v1"
                && sources.len() == 28
                && self.sources.len() == 28,
            "fixed source population"
        );
        ensure!(
            !self.rows.is_empty()
                && self.rows.iter().all(|r| sources.contains(&r.source)
                    && r.target.is_finite()
                    && !r.observation.is_empty()
                    && !r.codec.is_empty()
                    && !r.ladder.is_empty()),
            "finite keyed observations"
        );
        ensure!(
            self.rows
                .iter()
                .map(|r| &r.observation)
                .collect::<BTreeSet<_>>()
                .len()
                == self.rows.len()
                && self.rows.iter().map(|r| r.source).collect::<BTreeSet<_>>() == sources,
            "complete unique observations"
        );
        ensure!(
            self.predictions
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                == MODELS.into_iter().collect(),
            "eight frozen models"
        );
        for p in self.predictions.values() {
            ensure!(
                p.len() == 3
                    && p.iter().all(|v| v.len() == self.rows.len()
                        && v.iter()
                            .all(|x| x.quality.is_finite() && x.jnd.is_finite() && x.jnd >= 0.)),
                "exactly three finite paired seeds"
            );
        }
        let mut seen = BTreeSet::new();
        let mut reused = BTreeMap::new();
        for f in &self.frozen {
            ensure!(
                MODELS.contains(&f.model.as_str())
                    && sources.contains(&f.source)
                    && f.seed_index < 3
                    && hash(&f.network_sha256)
                    && hash(&f.map_sha256)
                    && seen.insert((f.model.as_str(), f.source, f.seed_index)),
                "unique frozen network/map binding"
            );
            ensure!(
                f.map_fit_sources.len() == 27
                    && f.map_fit_sources.iter().copied().collect::<BTreeSet<_>>()
                        == sources.iter().copied().filter(|s| *s != f.source).collect(),
                "held-out source entered map fitting"
            );
            if f.model.starts_with("base-") || f.model.starts_with("p-") {
                let key = (f.model.as_str(), f.seed_index);
                ensure!(
                    *reused.entry(key).or_insert(&f.network_sha256) == &f.network_sha256,
                    "Control/P full-data reuse differs"
                );
            }
        }
        ensure!(seen.len() == 672, "all fold-local map bindings required");
        Ok(())
    }
    fn units(&self, indices: &[usize]) -> Result<Vec<Unit>> {
        let mut out = vec![];
        for source in &self.sources {
            let ids: Vec<_> = indices
                .iter()
                .copied()
                .filter(|&i| self.rows[i].source == *source)
                .collect();
            if ids.is_empty() {
                continue;
            } // Explicit population subset; source count is reported.
            let refs = vec![*source; ids.len()];
            let names: BTreeMap<_, _> = ids
                .iter()
                .map(|&i| (self.rows[i].ladder.as_str(), 0u32))
                .collect();
            let names: BTreeMap<_, _> = names
                .keys()
                .enumerate()
                .map(|(i, &n)| (n, i as u32))
                .collect();
            let families: BTreeMap<_, _> = ids
                .iter()
                .map(|&i| (self.rows[i].codec.as_str(), 0u32))
                .collect();
            let families: BTreeMap<_, _> = families
                .keys()
                .enumerate()
                .map(|(i, &n)| (n, i as u32))
                .collect();
            let codec: Vec<_> = ids
                .iter()
                .map(|&i| names[self.rows[i].ladder.as_str()])
                .collect();
            let family: Vec<_> = ids
                .iter()
                .map(|&i| families[self.rows[i].codec.as_str()])
                .collect();
            // A ladder is nested in one codec; otherwise pair-class semantics change.
            let mut nested = BTreeMap::new();
            for (&c, &f) in codec.iter().zip(&family) {
                ensure!(*nested.entry(c).or_insert(f) == f, "ladder crosses codecs");
            }
            let target: Vec<_> = ids.iter().map(|&i| self.rows[i].target).collect();
            for seed in 0..3 {
                let mut unit = Unit {
                    source: *source,
                    seed: seed as u32,
                    n: ids.len() as f64,
                    sse: [0.; 8],
                    pairs: [[[0.; 2]; 3]; 8],
                };
                for (mi, name) in MODELS.iter().enumerate() {
                    let p = &self.predictions[*name][seed];
                    unit.sse[mi] = ids
                        .iter()
                        .map(|&i| (p[i].jnd - self.rows[i].target).powi(2))
                        .sum();
                    let quality: Vec<_> = ids.iter().map(|&i| p[i].quality).collect();
                    let input = PairInput {
                        predicted: &quality,
                        target: &target,
                        source: &refs,
                        codec: &codec,
                        family: Some(&family),
                        draw: None,
                        orientation: Orientation::LowerIsBetter,
                    };
                    // The existing owner excludes <= threshold. One representable
                    // step below 0.5 retains pairs separated by exactly 0.5 JND.
                    let threshold = f64::from_bits(0.5f64.to_bits() - 1);
                    let separated = class_agreement(&input, threshold)?;
                    let all = class_agreement(&input, 0.)?;
                    let a = separated.same_source_same_codec;
                    let b = separated.same_source_cross_codec;
                    unit.pairs[mi] = [
                        [a.concordant, a.pairs],
                        [b.concordant, b.pairs],
                        [
                            all.pooled.concordant,
                            all.pooled.pairs + all.pooled.target_ties,
                        ],
                    ];
                }
                out.push(unit);
            }
        }
        Ok(out)
    }
}
struct Unit {
    source: u32,
    seed: u32,
    n: f64,
    sse: [f64; 8],
    pairs: [[[f64; 2]; 3]; 8],
}
fn reduce(units: &[Unit], expanded: &[(usize, u64)]) -> Vec<f64> {
    let mut by_seed: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for &(i, seed_copy) in expanded {
        by_seed.entry(seed_copy).or_default().push(i);
    }
    let mut result = vec![0.; 40];
    for indices in by_seed.values() {
        let n: f64 = indices.iter().map(|&i| units[i].n).sum();
        for mi in 0..8 {
            result[mi * 5] += (indices.iter().map(|&i| units[i].sse[mi]).sum::<f64>() / n).sqrt();
            result[mi * 5 + 1] += (indices
                .iter()
                .map(|&i| units[i].sse[mi] / units[i].n)
                .sum::<f64>()
                / indices.len() as f64)
                .sqrt();
            for k in 0..3 {
                let correct: f64 = indices.iter().map(|&i| units[i].pairs[mi][k][0]).sum();
                let pairs: f64 = indices.iter().map(|&i| units[i].pairs[mi][k][1]).sum();
                result[mi * 5 + k + 2] += correct / pairs;
            }
        }
    }
    for v in &mut result {
        *v /= by_seed.len() as f64;
    }
    result
}
fn expanded(draw: &Draw<'_>) -> Vec<(usize, u64)> {
    let e = draw.expand();
    e.rows
        .into_iter()
        .zip(e.groups[1].iter().copied())
        .collect()
}
fn interval(iv: &Interval, g: usize) -> Value {
    let student = iv.student(g, 0.95);
    let complete = iv.estimate.is_finite()
        && iv.lo.is_finite()
        && iv.hi.is_finite()
        && iv.se.is_finite()
        && student.is_some();
    json!({"state":if complete {"COMPLETE"} else {"INCOMPLETE"},"estimate":iv.estimate,"percentile":[iv.lo,iv.hi],
        "student":student,"se":iv.se,"corrected_se":if g>=2 {iv.se*(g as f64/(g-1) as f64).sqrt()} else {f64::NAN},
        "groups":g,"df":g.saturating_sub(1),"n_finite":iv.n_finite,"attempted":10000,"dropped":10000-iv.n_finite})
}
fn population(input: &Input, indices: &[usize], seed: u64) -> Result<Value> {
    let units = input.units(indices)?;
    let sources: Vec<_> = units.iter().map(|u| u.source).collect();
    let seeds: Vec<_> = units.iter().map(|u| u.seed).collect();
    let g = sources.iter().collect::<BTreeSet<_>>().len().min(3);
    let point = reduce(
        &units,
        &units
            .iter()
            .enumerate()
            .map(|(i, u)| (i, u.seed as u64))
            .collect::<Vec<_>>(),
    );
    let mut per_seed = vec![];
    for seed in 0..3 {
        let p = reduce(
            &units,
            &units
                .iter()
                .enumerate()
                .filter(|(_, u)| u.seed == seed)
                .map(|(i, u)| (i, u.seed as u64))
                .collect::<Vec<_>>(),
        );
        per_seed.push(json!({"seed_index":seed,"metrics":p,"signed_srocc":MODELS.iter().map(|m| {
            let q:Vec<_>=indices.iter().map(|&i|input.predictions[*m][seed as usize][i].quality).collect();
            let y:Vec<_>=indices.iter().map(|&i|input.rows[i].target).collect(); -zenstats::spearman(&q,&y)
        }).collect::<Vec<_>>()}));
    }
    let cfg = BootstrapConfig {
        resamples: 10000,
        seed,
        level: 0.95,
    };
    let intervals = if g >= 2 {
        Some(cluster_bootstrap(
            units.len(),
            &Design::TwoWay(&sources, &seeds),
            &cfg,
            |draw| {
                let value = reduce(&units, &expanded(draw));
                PAIRS
                    .iter()
                    .flat_map(|&(a, c)| {
                        (0..5).map({
                            let value = &value;
                            move |m| value[a * 5 + m] - value[c * 5 + m]
                        })
                    })
                    .collect()
            },
        )?)
    } else {
        None
    };
    let comparisons:Vec<_>=PAIRS.iter().enumerate().map(|(i,&(a,c))|json!({"arm":MODELS[a],"control":MODELS[c],
        "metrics":METRICS.iter().enumerate().map(|(k,name)|json!({"name":name,"interval":intervals.as_ref().map(|iv|interval(&iv[i*5+k],g)).unwrap_or_else(||json!({"state":"INCOMPLETE","groups":g,"reason":"fewer than two source levels"}))})).collect::<Vec<_>>()})).collect();
    Ok(
        json!({"sources":sources.iter().collect::<BTreeSet<_>>().len(),"rows":indices.len(),"point":point,"per_seed":per_seed,"comparisons":comparisons,"seed":seed}),
    )
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    let bytes = fs::read(&cli.input)?;
    ensure!(
        Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
            == cli.input_sha256,
        "frozen LOSO input changed"
    );
    let input: Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    let mut populations = BTreeMap::new();
    populations.insert(
        "all".to_string(),
        population(
            &input,
            &(0..input.rows.len()).collect::<Vec<_>>(),
            20261016 + 4000,
        )?,
    );
    let codecs: BTreeSet<_> = input.rows.iter().map(|r| r.codec.as_str()).collect();
    for (k, codec) in codecs.iter().enumerate() {
        let indices: Vec<_> = input
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.codec == *codec)
            .map(|(i, _)| i)
            .collect();
        populations.insert(
            format!("codec/{codec}"),
            population(&input, &indices, 20261016 + 4001 + k as u64)?,
        );
    }
    // Per-source errors use the same reduction but cannot support a source-cluster interval.
    for source in &input.sources {
        let indices: Vec<_> = input
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.source == *source)
            .map(|(i, _)| i)
            .collect();
        populations.insert(
            format!("source/{source}"),
            population(&input, &indices, 20261016 + 5000 + u64::from(*source))?,
        );
    }
    let out = json!({"schema":"registered-conditional-loso-report-v1","report_only":true,"selection_allowed":false,
        "input_sha256":cli.input_sha256,"seeds":[0,1,2],"resamples":10000,"level":0.95,"percentile":"Hyndman-Fan type 7",
        "conditioned_on":"frozen source-held-out networks and fold-local maps; no retraining or map refitting",
        "models":MODELS,"metrics":METRICS,"populations":populations});
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(cli.output)?;
    serde_json::to_writer_pretty(file, &out)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn seed_rmse_precedes_three_seed_average_and_source_equalization() {
        let mut units = vec![];
        for seed in 0..3 {
            for source in 0..2 {
                let n = if source == 0 { 1. } else { 3. };
                units.push(Unit {
                    source,
                    seed,
                    n,
                    sse: [n * (seed as f64 + source as f64).powi(2); 8],
                    pairs: [[[1., 2.]; 3]; 8],
                });
            }
        }
        let all = units
            .iter()
            .enumerate()
            .map(|(i, u)| (i, u.seed as u64))
            .collect::<Vec<_>>();
        let p = reduce(&units, &all);
        let expected = ((0.75f64).sqrt() + (3.25f64).sqrt() + (7.75f64).sqrt()) / 3.;
        assert_eq!(p[0], expected);
        assert_ne!(p[0], ((0.75 + 3.25 + 7.75) / 3.0f64).sqrt());
        assert_eq!(
            p[1],
            ((0.5f64).sqrt() + (2.5f64).sqrt() + (6.5f64).sqrt()) / 3.
        );
        assert_eq!(p[2], 0.5);
        let duplicated = vec![(0, 0), (0, 1), (2, 2)];
        assert_eq!(reduce(&units, &duplicated)[0], 1. / 3.); // Three distinct seed copies, even with repeated original.
        let iv = Interval {
            estimate: 1.,
            lo: 0.,
            hi: 2.,
            se: 0.2,
            n_finite: 10000,
            p_zero: 1.,
            p_nonpositive: 0.5,
        };
        let j = interval(&iv, 3);
        assert_eq!(j["df"], 2);
        assert_eq!(j["student"], json!(iv.student(3, 0.95)));
        assert_eq!(interval(&iv, 1)["state"], "INCOMPLETE");
    }
    fn fixture() -> Input {
        let sources: Vec<u32> = (0..28).collect();
        let rows: Vec<_> = sources
            .iter()
            .flat_map(|&source| {
                [0., 0.5, 1.5, 1.5]
                    .into_iter()
                    .enumerate()
                    .map(move |(i, target)| Row {
                        observation: format!("invented-{source}-{i}"),
                        source,
                        codec: format!("c{}", i / 2),
                        ladder: format!("l{}", i / 2),
                        target,
                    })
            })
            .collect();
        let predictions = MODELS
            .iter()
            .enumerate()
            .map(|(mi, &m)| {
                (
                    m.into(),
                    (0..3)
                        .map(|seed| {
                            rows.iter()
                                .map(|r| Prediction {
                                    quality: if mi == 1 { 0. } else { -r.target },
                                    jnd: r.target
                                        + (1. + r.source as f64 / 100. + seed as f64 / 10.)
                                            / (mi + 1) as f64,
                                })
                                .collect()
                        })
                        .collect(),
                )
            })
            .collect();
        let frozen = MODELS
            .iter()
            .enumerate()
            .flat_map(|(mi, &m)| {
                sources.iter().flat_map(move |&source| {
                    (0..3).map(move |seed_index| Frozen {
                        model: m.into(),
                        source,
                        seed_index,
                        network_sha256: format!("{:064x}", mi * 3 + seed_index + 1),
                        map_sha256: format!("{:064x}", source + 1),
                        map_fit_sources: (0..28).filter(|&s| s != source).collect(),
                    })
                })
            })
            .collect();
        Input {
            schema: "registered-frozen-loso-v1".into(),
            sources,
            rows,
            predictions,
            frozen,
        }
    }
    #[test]
    fn frozen_fold_binding_refuses_leaks_missing_maps_and_wrong_seeds() {
        let mut input = fixture();
        input.validate().unwrap();
        input.frozen[0].map_fit_sources[0] = 0;
        assert!(input.validate().is_err());
        let mut input = fixture();
        input.frozen.pop();
        assert!(input.validate().is_err());
        let mut input = fixture();
        input.predictions.get_mut("base-a").unwrap().pop();
        assert!(input.validate().is_err());
        let mut input = fixture();
        input.frozen[0].network_sha256 = "f".repeat(64);
        assert!(input.validate().is_err());
    }
    #[test]
    fn actual_source_seed_bootstrap_uses_df_two_and_pair_ties_disagree() {
        let input = fixture();
        input.validate().unwrap();
        let units = input
            .units(&(0..input.rows.len()).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(units[0].pairs[0], [[1., 1.], [4., 4.], [5., 6.]]);
        assert_eq!(units[0].pairs[1], [[0., 1.], [0., 4.], [0., 6.]]);
        let report =
            population(&input, &(0..input.rows.len()).collect::<Vec<_>>(), 20265016).unwrap();
        let iv = &report["comparisons"][0]["metrics"][0]["interval"];
        assert_eq!(iv["state"], "COMPLETE");
        assert_eq!(iv["df"], 2);
        assert_eq!(iv["groups"], 3);
        assert_eq!(iv["n_finite"], 10000);
        assert!(iv["estimate"].as_f64().unwrap() < 0.);
        assert_eq!(report["per_seed"].as_array().unwrap().len(), 3);
        let only_one: Vec<_> = (0..4).collect();
        let sparse = population(&input, &only_one, 20265016).unwrap();
        assert_eq!(
            sparse["comparisons"][0]["metrics"][0]["interval"]["state"],
            "INCOMPLETE"
        );
    }
}
