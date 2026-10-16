//! Neutral factorial inventory over E34's numerical owner. No fitting/selection.
use super::*;

const INVENTORY: [(&str, &str, bool); 12] = [
    ("p-a", "base-a", true),
    ("h-a", "base-a", true),
    ("ph-a", "base-a", true),
    ("h-a", "p-a", false),
    ("ph-a", "p-a", false),
    ("ph-a", "h-a", false),
    ("p-c", "base-c", true),
    ("h-c", "base-c", true),
    ("ph-c", "base-c", true),
    ("h-c", "p-c", false),
    ("ph-c", "p-c", false),
    ("ph-c", "h-c", false),
];
const OWNER: &str = "81a545a3ff3457c9fabae93bc9d262b19b50952e";
const HELPER: &str = "48665c8fb1bb3de5c99d792ae4ed4f3f1e854629";

fn failure(reason: &str) -> Value {
    json!({"schema":"registered-factorial-statistics-v1","state":"INCOMPLETE","decision_allowed":false,"reason":reason,
        "tests":INVENTORY.iter().enumerate().map(|(i,_)|json!({"test":i+1,"state":"INCOMPLETE","p":{"decision":1.}})).collect::<Vec<_>>(),
        "holm_adjusted":zenstats::multiple::holm(&[1.;12]),"holm_reject":zenstats::multiple::holm_reject(&[1.;12],0.05)})
}
fn validate(input: &Input) -> Result<()> {
    ensure!(
        input.schema == "registered-factorial-paired-v1",
        "registered paired schema"
    );
    ensure!(
        input
            .datasets
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>()
            == DATASETS,
        "four-source order"
    );
    ensure!(
        input.tests.len() == 12
            && input
                .tests
                .iter()
                .zip(INVENTORY)
                .enumerate()
                .all(|(i, (t, (a, c, e)))| t.index == i as u64 + 1
                    && t.arm == a
                    && t.control == c
                    && t.eligibility == e),
        "fixed twelve-test inventory"
    );
    input.validate_rows()
}
fn category(v: f64) -> &'static str {
    if v.is_nan() {
        "nan"
    } else if v == f64::INFINITY {
        "positive-infinity"
    } else if v == f64::NEG_INFINITY {
        "negative-infinity"
    } else {
        "finite"
    }
}
fn reported(iv: &Interval, g: usize) -> Value {
    let mut out = interval_json(iv, g);
    out["attempted"] = json!(10000);
    out["dropped"] = json!(10000 - iv.n_finite);
    out["p_zero"] = json!(iv.p_zero);
    out["categories"] = json!({"estimate":category(iv.estimate),"lo":category(iv.lo),"hi":category(iv.hi),"se":category(iv.se),
                            "p_nonpositive":category(iv.p_nonpositive)});
    out
}
fn w2_guard(delta: f64, se: f64) -> bool {
    delta > -2. * se
}
fn guarded_decision(primary: &Interval, guards: &[Interval], w2: &Interval) -> Value {
    let mut p = decision(primary, 10);
    if guards
        .iter()
        .chain(std::iter::once(w2))
        .any(|iv| !iv.estimate.is_finite() || !iv.se.is_finite())
    {
        // Preserve the primary's already computed component probabilities.
        p["complete"] = json!(false);
        p["decision"] = json!(1.);
    }
    p
}
fn slot(input: &Input, t: &Test, seed_shift: u64, reports: bool) -> Result<Value> {
    if input.datasets.iter().any(|d| {
        [&t.arm, &t.control].iter().any(|m| {
            d.predictions.get(*m).is_none_or(|p| {
                p.len() != 10
                    || p.iter()
                        .any(|v| v.len() != d.target.len() || v.iter().any(|x| !x.is_finite()))
            })
        })
    }) {
        return Ok(
            json!({"test":t.index,"arm":t.arm,"control":t.control,"state":"INCOMPLETE","p":{"decision":1.},
            "reason":"ten complete paired seed panels required"}),
        );
    }
    let base = 100 * t.index + seed_shift;
    let margin = if t.eligibility {
        ELIGIBILITY_MARGIN
    } else {
        0.
    };
    let primary = input.interval(t, &[0, 1, 2, 3], base, false, false, margin)?;
    let reference = input.interval(t, &[0, 1, 2, 3], base + 6, true, false, 0.)?;
    let mut guards = Vec::new();
    let mut guard_reports = Vec::new();
    for (di, name) in DATASETS.iter().enumerate() {
        let iv = input.interval(t, &[di], base + di as u64 + 1, false, false, 0.)?;
        let g = input.datasets[di]
            .reference
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            .min(10);
        guard_reports.push(
            json!({"dataset":name,"interval":reported(&iv,g),"point_guard":iv.estimate>= -0.005,
            "seed":20261011+base+di as u64+1}),
        );
        guards.push(iv);
    }
    let w2 = input.interval(t, &[0, 1], base + 5, false, true, 0.)?;
    let p = guarded_decision(&primary, &guards, &w2);
    let w2_seed = w2_seed_component(input, t)
        .unwrap_or_else(|e| json!({"state":"INCOMPLETE","reason":e.to_string()}));
    let old_w2_guard = w2_seed["delta"]
        .as_f64()
        .zip(w2_seed["se"].as_f64())
        .map(|(d, s)| w2_guard(d, s));
    Ok(
        json!({"test":t.index,"arm":t.arm,"control":t.control,"margin":margin,
        "state":if p["complete"]==true {"COMPLETE"} else {"INCOMPLETE"},"theta":reported(&primary,10),
        "seed":20261011+base,"delta":primary.estimate-margin,"p":p,"dataset_guards":guard_reports,
        "w2":{"interval":reported(&w2,10),"point_guard":w2_guard(w2.estimate,w2.se),"seed":20261011+base+5,"seed_unit":w2_seed,"e33_seed_unit_point_guard":old_w2_guard},
        "reference_only":reported(&reference,input.rows(&[0,1,2,3]).reference.iter().collect::<BTreeSet<_>>().len()),
        "reference_only_seed":20261011+base+6,"seed_unit":seed_component(input,t)?,
        "e33_mean_point_guard":primary.estimate-margin>= -0.002,"improvement_point_guard":primary.estimate>0.002,
        "pair_classes":if reports && t.eligibility {pair_classes_at(input,t,|c|1000+10*t.index+c+seed_shift)?} else {Value::Null}}),
    )
}
fn evaluate(input: &Input) -> Result<Value> {
    if let Err(e) = validate(input) {
        return Ok(failure(&e.to_string()));
    }
    let tests = input
        .tests
        .iter()
        .map(|t| slot(input, t, 5, true))
        .collect::<Result<Vec<_>>>()?;
    let ps: Vec<_> = tests
        .iter()
        .map(|t| t["p"]["decision"].as_f64().unwrap())
        .collect();
    let complete = tests.iter().all(|t| t["state"] == "COMPLETE");
    Ok(
        json!({"schema":"registered-factorial-statistics-v1","state":if complete {"COMPLETE"} else {"INCOMPLETE"},
        "decision_allowed":complete,"tests":tests,"holm_adjusted":zenstats::multiple::holm(&ps),
        "holm_reject":zenstats::multiple::holm_reject(&ps,0.05),"base_seed":20261016,"resamples":10000,"level":0.95,
        "percentile":"Hyndman-Fan type 7","reference_design":"globally-dataset-qualified unstratified copy-aware TwoWay",
        "numerical_owner":OWNER,"helper_commit":HELPER}),
    )
}
pub(super) fn run(cli: Cli) -> Result<()> {
    let input: Input = serde_json::from_slice(&fs::read(&cli.input)?)?;
    let output = if cli.mode == "registered-owner-parity" {
        // Same comparisons, offsets, margins and RNG as the pinned owner.
        // This mode cannot emit a registered scientific verdict.
        if let Err(e) = input.validate(false) {
            json!({"state":"INCOMPLETE","reason":e.to_string(),"tests":(1..=9).map(|i|
                json!({"test":i,"state":"INCOMPLETE","p":{"decision":1.}})).collect::<Vec<_>>()})
        } else {
            json!({"state":"PARITY_ONLY","tests":input.tests.iter().map(|t|slot(&input,t,0,false)).collect::<Result<Vec<_>>>()?})
        }
    } else {
        evaluate(&input)?
    };
    save(&cli.output, &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn iv(estimate: f64, se: f64, n: usize) -> Interval {
        Interval {
            estimate,
            lo: estimate,
            hi: estimate,
            se,
            n_finite: n,
            p_zero: 0.2,
            p_nonpositive: 0.1,
        }
    }
    #[test]
    fn registered_zero_one_finite_and_guard_failures_preserve_owner_decisions() {
        for primary in [
            iv(0.0065, 0., 10000),
            iv(f64::NAN, f64::NAN, 0),
            iv(0., f64::NAN, 1),
            iv(0.01, 0.02, 9999),
        ] {
            assert_eq!(
                guarded_decision(&primary, &[iv(0., 0., 10000)], &iv(0., 0., 10000)),
                decision(&primary, 10)
            );
        }
        let primary = iv(0.01, 0.02, 10000);
        let expected = decision(&primary, 10);
        let incomplete =
            guarded_decision(&primary, &[iv(f64::NAN, f64::NAN, 0)], &iv(0., 0., 10000));
        assert_eq!(incomplete["bootstrap"], expected["bootstrap"]);
        assert_eq!(incomplete["student"], expected["student"]);
        assert_eq!(incomplete["decision"], 1.);
        assert_eq!(incomplete["complete"], false);
        assert!(!w2_guard(0., 0.));
    }
    #[test]
    fn registered_twelve_inventory_and_holm_are_fixed() {
        assert_eq!(INVENTORY.len(), 12);
        assert_eq!(INVENTORY[6], ("p-c", "base-c", true));
        let ps = [
            0.0001, 0.0002, 0.0003, 0.004, 0.005, 0.008, 0.009, 0.010, 0.011, 0.012, 0.5, 1.,
        ];
        let reject = zenstats::multiple::holm_reject(&ps, 0.05);
        assert!(reject[0] && reject[1] && reject[2] && reject[3] && reject[4]);
        assert!(!reject[5] && !reject[11]);
        let input = Input {
            schema: "registered-factorial-paired-v1".into(),
            datasets: vec![],
            tests: INVENTORY
                .iter()
                .enumerate()
                .map(|(i, (a, c, e))| Test {
                    index: i as u64 + 1,
                    arm: (*a).into(),
                    control: (*c).into(),
                    eligibility: *e,
                })
                .collect(),
        };
        let output = evaluate(&input).unwrap();
        assert_eq!(output["state"], "INCOMPLETE");
        assert_eq!(output["decision_allowed"], false);
        assert_eq!(output["tests"].as_array().unwrap().len(), 12);
    }
    #[test]
    fn registered_real_finite_draw_reduction_attempts_ten_thousand_without_redraw() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let refs = [0, 1, 2];
        let design = Design::OneWay(&refs);
        let cfg = BootstrapConfig {
            resamples: 10000,
            seed: 20261016,
            level: 0.95,
        };
        for wanted in [0, 1] {
            let calls = AtomicUsize::new(0);
            let iv = cluster_bootstrap(3, &design, &cfg, |_| {
                let index = calls.fetch_add(1, Ordering::Relaxed);
                vec![if index <= wanted { 1. } else { f64::NAN }]
            })
            .unwrap()
            .remove(0);
            assert_eq!(calls.load(Ordering::Relaxed), 10001); // Point plus fixed attempted draws.
            assert_eq!(iv.n_finite, wanted);
            assert!(iv.se.is_nan());
            assert_eq!(
                guarded_decision(
                    &iv,
                    &[super::tests::iv(0., 0., 10000)],
                    &super::tests::iv(0., 0., 10000)
                ),
                decision(&iv, 10)
            );
            assert_eq!(decision(&iv, 10)["bootstrap"], 1.);
            assert_eq!(decision(&iv, 10)["student"], 1.);
        }
        assert!(cluster_bootstrap(0, &design, &cfg, |_| vec![0.]).is_err());
        let bad = BootstrapConfig {
            resamples: 0,
            ..cfg
        };
        assert!(cluster_bootstrap(3, &design, &bad, |_| vec![0.]).is_err());
        let calls = AtomicUsize::new(0);
        assert!(
            cluster_bootstrap(3, &design, &cfg, |_| {
                if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                    vec![0.]
                } else {
                    vec![]
                }
            })
            .is_err()
        );
    }
    #[test]
    fn registered_copy_expansion_qualifies_datasets_and_repeated_seed_copies() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let input = Input {
            schema: "e34-paired-seed-v1".into(),
            tests: vec![],
            datasets: DATASETS
                .iter()
                .map(|name| Dataset {
                    name: (*name).into(),
                    target: (0..15).map(f64::from).collect(),
                    reference: (0..15).map(|i| i / 5).collect(),
                    codec: (0..15).map(|i| i % 3).collect(),
                    orientation: 1.,
                    predictions: BTreeMap::from([
                        ("a".into(), vec![(0..15).map(f64::from).collect(); 10]),
                        (
                            "b".into(),
                            (0..10)
                                .map(|s| {
                                    (0..15)
                                        .map(|i| f64::from(if s % 2 == 0 { i } else { 14 - i }))
                                        .collect()
                                })
                                .collect(),
                        ),
                    ]),
                })
                .collect(),
        };
        let t = Test {
            index: 1,
            arm: "a".into(),
            control: "b".into(),
            eligibility: false,
        };
        let rows = input.rows(&[0, 1, 2, 3]);
        assert_eq!(
            rows.reference.iter().collect::<BTreeSet<_>>().len(),
            12,
            "local IDs cannot merge datasets"
        );
        let missing = AtomicUsize::new(0);
        let repeated = AtomicUsize::new(0);
        let seed_copies = AtomicUsize::new(0);
        let design = Design::TwoWay(&rows.reference, &rows.seed);
        let cfg = BootstrapConfig {
            resamples: 10000,
            seed: 20261016,
            level: 0.95,
        };
        let iv = cluster_bootstrap(rows.sample.len(), &design, &cfg, |draw| {
            let e = draw.expand();
            let value = input.statistic(&rows, draw, &t, false, &[0, 1, 2, 3]);
            let counts: Vec<_> = (0..4)
                .map(|di| {
                    e.rows
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| rows.dataset[**r] == di)
                        .map(|(j, _)| e.groups[0][j])
                        .collect::<BTreeSet<_>>()
                        .len()
                })
                .collect();
            if counts.iter().any(|&n| n < 3) {
                assert!(value[0].is_nan());
                missing.fetch_add(1, Ordering::Relaxed);
            } else {
                assert!(value[0].is_finite());
                if (0..4).any(|di| {
                    e.rows
                        .iter()
                        .filter(|&&r| rows.dataset[r] == di)
                        .map(|&r| rows.reference[r])
                        .collect::<BTreeSet<_>>()
                        .len()
                        < 3
                }) {
                    repeated.fetch_add(1, Ordering::Relaxed);
                }
            }
            if e.rows
                .iter()
                .map(|&r| rows.seed[r])
                .collect::<BTreeSet<_>>()
                .len()
                < 10
            {
                assert_eq!(e.groups[1].iter().collect::<BTreeSet<_>>().len(), 10);
                seed_copies.fetch_add(1, Ordering::Relaxed);
            }
            value
        })
        .unwrap()
        .remove(0);
        assert!(
            missing.load(Ordering::Relaxed) > 0
                && repeated.load(Ordering::Relaxed) > 0
                && seed_copies.load(Ordering::Relaxed) > 0
        );
        assert!(iv.n_finite > 0 && iv.n_finite < 10000);
    }
}
