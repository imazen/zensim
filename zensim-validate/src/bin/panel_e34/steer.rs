//! Versioned STEER inventory. E34's frozen evaluator stays unchanged.
use super::*;

const INVENTORY: [(&str, &str, bool); 12] = [
    ("s-a", "base-a", true),
    ("r-a", "base-a", true),
    ("sr-a", "base-a", true),
    ("s-c", "base-c", true),
    ("r-c", "base-c", true),
    ("sr-c", "base-c", true),
    ("r-a", "s-a", false),
    ("sr-a", "s-a", false),
    ("sr-a", "r-a", false),
    ("r-c", "s-c", false),
    ("sr-c", "s-c", false),
    ("sr-c", "r-c", false),
];
const IDENTICAL: [u64; 4] = [2, 5, 8, 11];
const DATA_SHA: &str = "9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Registration {
    schema: String,
    alias_policy: String,
    q12_approved: bool,
    identical_tests: Vec<u64>,
    fits: usize,
    tau: f64,
    data_sha256: String,
}
impl Registration {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "steer-registration-v1",
            "STEER registration schema"
        );
        ensure!(
            matches!(self.alias_policy.as_str(), "q12-a-v1" | "e34-strict-b-v1"),
            "unknown alias policy"
        );
        ensure!(
            self.identical_tests == IDENTICAL,
            "pre-fit identical comparison inventory"
        );
        ensure!(
            self.fits == 172 && self.tau == 0.125 && self.data_sha256 == DATA_SHA,
            "fixed STEER packet"
        );
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    registration_sha256: String,
    paired: Input,
}

fn failure(reason: &str, policy: &str) -> Value {
    let ps = vec![1.; 12];
    json!({"schema":"steer-statistics-v1","state":"INCOMPLETE","alias_policy":policy,
        "scope":if policy=="e34-strict-b-v1" {"DIAGNOSTIC_ONLY"} else {"S_ONLY"},"reason":reason,
        "tests":INVENTORY.iter().enumerate().map(|(i,(a,c,e))|json!({"test":i+1,"arm":a,"control":c,"eligibility":e,"state":"INCOMPLETE","p":{"decision":1.}})).collect::<Vec<_>>(),
        "holm_adjusted":zenstats::multiple::holm(&ps),"holm_reject":zenstats::multiple::holm_reject(&ps,0.05)})
}

fn validate(input: &Input) -> Result<()> {
    ensure!(
        input.schema == "steer-paired-seed-v1",
        "versioned STEER paired schema"
    );
    ensure!(input.tests.len() == 12, "twelve STEER hypotheses required");
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
        input
            .tests
            .iter()
            .zip(INVENTORY)
            .enumerate()
            .all(|(i, (t, (a, c, e)))| t.index == i as u64 + 1
                && t.arm == a
                && t.control == c
                && t.eligibility == e),
        "fixed STEER test order"
    );
    input.validate_rows()?;
    for d in &input.datasets {
        ensure!(
            d.reference.iter().collect::<BTreeSet<_>>().len() >= 3,
            "three source references required"
        );
        ensure!(
            d.target.iter().any(|x| *x != d.target[0]),
            "constant target cannot support a rank comparison"
        );
        for (a, c, _) in INVENTORY {
            for m in [a, c] {
                let p = d
                    .predictions
                    .get(m)
                    .ok_or_else(|| anyhow::anyhow!("missing {m}"))?;
                for seed in p {
                    ensure!(
                        seed.iter().any(|x| *x != seed[0]),
                        "constant prediction cannot support a rank comparison"
                    );
                    ensure!(
                        zenstats::spearman(seed, &d.target).is_finite(),
                        "undefined constituent SROCC"
                    );
                }
            }
        }
        for (a, c) in [
            ("r-a", "base-a"),
            ("r-c", "base-c"),
            ("sr-a", "s-a"),
            ("sr-c", "s-c"),
        ] {
            ensure!(
                d.predictions[a]
                    .iter()
                    .flatten()
                    .zip(d.predictions[c].iter().flatten())
                    .all(|(a, c)| a.to_bits() == c.to_bits()),
                "broken predeclared accuracy alias {a}/{c}"
            );
        }
    }
    // Validate the constituent type correlations before granting an alias state.
    // A constant/invalid type cannot be repaired by subtracting identical NaNs.
    for d in &input.datasets[..2] {
        let mut types: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, t) in d.codec.iter().enumerate() {
            types.entry(*t).or_default().push(i);
        }
        ensure!(types.len() >= 3, "three W2 types required");
        for p in d.predictions.values() {
            for seed in p {
                for rows in types.values() {
                    let y: Vec<_> = rows.iter().map(|i| d.target[*i]).collect();
                    let p: Vec<_> = rows.iter().map(|i| seed[*i]).collect();
                    ensure!(
                        zenstats::spearman(&p, &y).is_finite(),
                        "undefined constituent W2 type"
                    );
                }
            }
        }
    }
    Ok(())
}

fn w2_guard(delta: f64, se: f64) -> bool {
    delta > -2. * se
}

fn exact_seed() -> Value {
    json!({"state":"IDENTICAL_BY_CONSTRUCTION","delta":0.,"se":0.,"n":10,
        "seed_deltas":vec![0.;10],"student":null,"reason":"construction-declared zero variance"})
}
fn admitted_interval(iv: &Interval) -> Result<()> {
    ensure!(
        iv.n_finite > 0
            && [iv.estimate, iv.lo, iv.hi, iv.se]
                .into_iter()
                .all(f64::is_finite),
        "invalid registered interval"
    );
    Ok(())
}
fn reported(iv: &Interval, g: usize, alias: bool) -> Value {
    let mut v = interval_json(iv, g);
    if alias {
        v["student_interval"] = Value::Null;
        v["student_reason"] = json!("construction-declared zero variance");
    }
    v
}
fn slot(input: &Input, t: &Test, policy: &str) -> Result<Value> {
    let alias = IDENTICAL.contains(&t.index) && policy == "q12-a-v1";
    let margin = if t.eligibility {
        ELIGIBILITY_MARGIN
    } else {
        0.
    };
    let primary = input.interval(
        t,
        &[0, 1, 2, 3],
        100 * t.index,
        false,
        false,
        if alias { 0. } else { margin },
    )?;
    let reference = input.interval(t, &[0, 1, 2, 3], 100 * t.index + 6, true, false, 0.)?;
    admitted_interval(&primary)?;
    admitted_interval(&reference)?;
    let seed = if alias {
        exact_seed()
    } else {
        seed_component(input, t)?
    };
    let mut p = if alias {
        ensure!(
            primary.estimate == 0.
                && primary.se == 0.
                && reference.estimate == 0.
                && reference.se == 0.,
            "alias did not reduce to exact zero"
        );
        json!({"complete":true,"bootstrap":null,"student":null,"decision":1.,"reason":"identical by construction, p = 1"})
    } else {
        decision(&primary, 10)
    };
    // Seed-only refusals are nested report diagnostics, as in frozen E34.
    let complete = p["complete"] == true;
    let mut guards = Vec::new();
    let w2 = {
        for (di, name) in DATASETS.iter().enumerate() {
            let iv = input.interval(t, &[di], 100 * t.index + di as u64 + 1, false, false, 0.)?;
            admitted_interval(&iv)?;
            if alias {
                ensure!(
                    iv.estimate == 0. && iv.se == 0.,
                    "nonzero alias dataset guard"
                );
            }
            let g = input.datasets[di]
                .reference
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                .min(10);
            guards.push(json!({"dataset":name,"interval":reported(&iv,g,alias),"point_guard":iv.estimate>=-0.005}));
        }
        let iv = input.interval(t, &[0, 1], 100 * t.index + 5, false, true, 0.)?;
        admitted_interval(&iv)?;
        let ws = if alias {
            exact_seed()
        } else {
            w2_seed_component(input, t)
                .unwrap_or_else(|e| json!({"state":"INCOMPLETE","reason":e.to_string()}))
        };
        if alias {
            ensure!(iv.estimate == 0. && iv.se == 0., "nonzero alias W2 guard");
        }
        json!({"interval":reported(&iv,10,alias),"point_guard":w2_guard(iv.estimate, iv.se),"seed_unit":ws})
    };
    if !complete {
        p["complete"] = json!(false);
        p["decision"] = json!(1.);
    }
    let delta = if alias {
        primary.estimate
    } else {
        primary.estimate - margin
    };
    let mut theta = reported(&primary, 10, alias);
    if alias {
        for key in ["estimate", "lo", "hi"] {
            theta[key] = json!(margin);
        }
    }
    Ok(
        json!({"test":t.index,"arm":t.arm,"control":t.control,"eligibility":t.eligibility,
        "state":if !complete {"INCOMPLETE"} else if alias {"IDENTICAL_BY_CONSTRUCTION"} else {"COMPLETE"},
        "label":if alias {"identical by construction, p = 1"}else{"ordinary"},
        "theta":theta,"delta":delta,
        "seed_unit":seed,"reference_only":reported(&reference,input.rows(&[0,1,2,3]).reference.iter().collect::<BTreeSet<_>>().len(),alias),
        "p":p,"dataset_guards":guards,"w2":w2,
        "e33_mean_point_guard":delta>=-0.002,"improvement_point_guard":delta>0.002}),
    )
}

fn evaluate(input: &Input, reg: &Registration) -> Value {
    if let Err(e) = reg.validate() {
        return failure(&e.to_string(), &reg.alias_policy);
    }
    if !reg.q12_approved {
        return json!({"schema":"steer-statistics-v1","state":"PENDING_OWNER_Q12"});
    }
    if let Err(e) = validate(input) {
        return failure(&e.to_string(), &reg.alias_policy);
    }
    let mut tests = Vec::new();
    for t in &input.tests {
        let result=slot(input,t,&reg.alias_policy).unwrap_or_else(|e|json!({"test":t.index,"arm":t.arm,"control":t.control,"state":"INCOMPLETE","reason":e.to_string(),"p":{"decision":1.}}));
        tests.push(result);
    }
    finish_packet(tests, &reg.alias_policy)
}

// Completeness applies to every comparison's required numeric guard evidence.
// A finite false point guard is evidence, and seed-only diagnostic states are
// deliberately not read here. Only the S eligibility slots use guard booleans.
fn guard_evidence_complete(t: &Value) -> bool {
    let valid = |iv: &Value| {
        ["estimate", "se"]
            .iter()
            .all(|key| iv[*key].as_f64().is_some_and(f64::is_finite))
    };
    t["dataset_guards"].as_array().is_some_and(|guards| {
        guards.len() == DATASETS.len()
            && guards.iter().zip(DATASETS).all(|(g, name)| {
                g["dataset"] == name && valid(&g["interval"]) && g["point_guard"].is_boolean()
            })
    }) && valid(&t["w2"]["interval"])
        && t["w2"]["point_guard"].is_boolean()
}

fn finish_packet(mut tests: Vec<Value>, policy: &str) -> Value {
    let mut ps = Vec::new();
    let mut complete = tests.len() == INVENTORY.len();
    for result in &mut tests {
        if !guard_evidence_complete(result) {
            result["state"] = json!("INCOMPLETE");
            result["guard_evidence_reason"] =
                json!("missing/nonfinite required dataset or W2 evidence");
            result["p"]["complete"] = json!(false);
            result["p"]["decision"] = json!(1.);
        }
        complete &= matches!(
            result["state"].as_str(),
            Some("COMPLETE" | "IDENTICAL_BY_CONSTRUCTION")
        );
        ps.push(result["p"]["decision"].as_f64().unwrap_or(1.));
    }
    json!({"schema":"steer-statistics-v1","state":if complete {"COMPLETE_FOR_S"}else{"INCOMPLETE"},
        "alias_policy":policy,"scope":if policy=="q12-a-v1" {"S_ONLY"}else{"DIAGNOSTIC_ONLY"},
        "eligibility_margin":ELIGIBILITY_MARGIN,"resamples":10000,"level":0.95,"percentile":"Hyndman-Fan type 7","base_seed":20261011,
        "tests":tests,"holm_adjusted":zenstats::multiple::holm(&ps),"holm_reject":zenstats::multiple::holm_reject(&ps,0.05)})
}

fn validate_model_names(models: &BTreeMap<String, Value>) -> Result<()> {
    let expected: BTreeSet<_> = ["base-a", "base-c", "s-a", "s-c"]
        .into_iter()
        .flat_map(|a| (0..3).map(move |s| format!("{a}-s{s}")))
        .collect();
    ensure!(
        models.keys().cloned().collect::<BTreeSet<_>>() == expected,
        "twelve exact full-data model names required"
    );
    Ok(())
}

pub(super) fn run(cli: Cli) -> Result<()> {
    if cli.mode == "steer-pack-check" {
        let bytes = fs::read(&cli.input)?;
        let models: BTreeMap<String, Value> = serde_json::from_slice(&bytes)?;
        validate_model_names(&models)?;
        return save(
            &cli.output,
            &reports::pack_inventory(&bytes, 12, "steer-pack-checks-v1")?,
        );
    }
    let pin = cli
        .registration
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--registration is required for STEER"))?;
    let registration_bytes = fs::read(pin)?;
    let reg: Registration = serde_json::from_slice(&registration_bytes)?;
    reg.validate()?;
    // Refuse before opening a payload when the owner has not resolved Q12.
    if !reg.q12_approved {
        return save(
            &cli.output,
            &json!({"schema":"steer-statistics-v1","state":"PENDING_OWNER_Q12","registration_sha256":digest(&registration_bytes)}),
        );
    }
    let bytes = fs::read(&cli.input)?;
    let request: Request = serde_json::from_slice(&bytes)?;
    ensure!(
        request.schema == "steer-evaluation-v1"
            && request.registration_sha256 == digest(&registration_bytes),
        "registration/input binding mismatch"
    );
    let mut result = evaluate(&request.paired, &reg);
    result["registration_sha256"] = json!(digest(&registration_bytes));
    result["input_sha256"] = json!(digest(&bytes));
    save(&cli.output, &result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn registration(policy: &str) -> Registration {
        Registration {
            schema: "steer-registration-v1".into(),
            alias_policy: policy.into(),
            q12_approved: true,
            identical_tests: IDENTICAL.to_vec(),
            fits: 172,
            tau: 0.125,
            data_sha256: DATA_SHA.into(),
        }
    }

    #[test]
    fn steer_full_model_names_refuse_e34_and_serving_aliases() {
        let models: BTreeMap<_, _> = ["base-a", "base-c", "s-a", "s-c"]
            .into_iter()
            .flat_map(|a| (0..3).map(move |s| (format!("{a}-s{s}"), Value::Null)))
            .collect();
        validate_model_names(&models).unwrap();
        for name in models.keys() {
            let mut missing = models.clone();
            missing.remove(name);
            assert!(validate_model_names(&missing).is_err());
        }
        let mut renamed = models.clone();
        renamed.remove("s-a-s0");
        renamed.insert("r-a-s0".into(), Value::Null);
        assert!(validate_model_names(&renamed).is_err());
        let mut e34 = models;
        for i in 0..9 {
            e34.insert(format!("extra-{i}"), Value::Null);
        }
        assert!(validate_model_names(&e34).is_err());
    }

    #[test]
    fn steer_a7_pre_freeze_canonical_boundaries() {
        let input = invented();
        let reg = registration("q12-a-v1");
        let baseline = evaluate(&input, &reg);
        assert_eq!(baseline["state"], "COMPLETE_FOR_S");
        let tests = || baseline["tests"].as_array().unwrap().clone();
        let mut cases = BTreeMap::new();
        cases.insert("eligible_s_exact_aliases".to_string(), baseline.clone());
        // Mutations below are explicitly at the canonical completion boundary,
        // after real bootstrap/Student/guard computation, never scientific data.
        for index in 0..12 {
            for field in ["dataset", "w2"] {
                let mut missing = tests();
                if field == "dataset" {
                    missing[index]["dataset_guards"] = json!([]);
                } else {
                    missing[index]["w2"]["interval"]["se"] = Value::Null;
                }
                let result = finish_packet(missing, "q12-a-v1");
                assert_eq!(result["state"], "INCOMPLETE");
                assert_eq!(result["tests"][index]["p"]["decision"], 1.);
                assert_eq!(result["holm_reject"].as_array().unwrap().len(), 12);
                cases.insert(format!("missing_{field}_slot_{}", index + 1), result);
            }
        }
        let mut nonrejecting = tests();
        for t in &mut nonrejecting {
            if t["state"] == "COMPLETE" {
                t["p"]["decision"] = json!(0.2);
            }
        }
        let result = finish_packet(nonrejecting, "q12-a-v1");
        assert_eq!(result["state"], "COMPLETE_FOR_S");
        assert!(
            result["holm_reject"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v == false)
        );
        cases.insert("ordinary_nonrejecting".into(), result);
        let mut strict = tests();
        strict[0]["w2"]["interval"]["estimate"] = json!(-0.002);
        strict[0]["w2"]["interval"]["se"] = json!(0.001);
        strict[0]["w2"]["point_guard"] = json!(w2_guard(-0.002, 0.001));
        let result = finish_packet(strict, "q12-a-v1");
        assert_eq!(result["state"], "COMPLETE_FOR_S");
        assert_eq!(result["tests"][0]["w2"]["point_guard"], false);
        cases.insert("strict_w2_boundary".into(), result);
        assert!(seed_stat(&vec![vec![0.; 3]; 10]).is_err());
        for defect in ["declaration", "source", "type", "constant_prediction"] {
            let mut reg = registration("q12-a-v1");
            let mut bad = invented();
            match defect {
                "declaration" => reg.identical_tests.pop().map(|_| ()).unwrap(),
                "source" => bad.datasets[0].reference.fill(0),
                "type" => bad.datasets[0].codec.fill(0),
                "constant_prediction" => {
                    bad.datasets[0].predictions.get_mut("s-a").unwrap()[0].fill(0.)
                }
                _ => unreachable!(),
            }
            let result = evaluate(&bad, &reg);
            assert_eq!(result["state"], "INCOMPLETE", "{defect}");
            assert_eq!(result["tests"].as_array().unwrap().len(), 12);
            cases.insert(format!("invalid_{defect}"), result);
        }
        if let Ok(path) = std::env::var("STEER_A7_RESULT") {
            save(Path::new(&path), &json!({
                "scope":"INVENTED_CANONICAL_A7_BOUNDARIES",
                "boundary_mutations":["numeric guard omission", "decision p=0.2", "strict W2 equality"],
                "cases":cases
            })).unwrap();
        }
    }
    fn invented() -> Input {
        let datasets = DATASETS
            .iter()
            .enumerate()
            .map(|(di, name)| {
                let target: Vec<_> = (0..36)
                    .map(|i| (i / 3) as f64 + (i % 3) as f64 * 0.125)
                    .collect();
                let mut predictions = BTreeMap::new();
                for family in ["a", "c"] {
                    let baseline: Vec<Vec<f64>> = (0..10)
                        .map(|s| {
                            (0..36)
                                .map(|i| {
                                    let rotated = ((i / 3) + (s % 6) + 1 + (di % 2)) % 12;
                                    rotated as f64 + (i % 3) as f64 * 0.125
                                })
                                .collect()
                        })
                        .collect();
                    let smooth: Vec<Vec<f64>> = (0..10)
                        .map(|s| target.iter().map(|y| y + s as f64 * 0.0001).collect())
                        .collect();
                    predictions.insert(format!("base-{family}"), baseline.clone());
                    predictions.insert(format!("r-{family}"), baseline);
                    predictions.insert(format!("s-{family}"), smooth.clone());
                    predictions.insert(format!("sr-{family}"), smooth);
                }
                Dataset {
                    name: name.to_string(),
                    target,
                    reference: (0..36).map(|i| (di * 100 + i / 3) as u32).collect(),
                    codec: (0..36).map(|i| (i % 3) as u32).collect(),
                    orientation: 1.,
                    predictions,
                }
            })
            .collect();
        Input {
            schema: "steer-paired-seed-v1".into(),
            datasets,
            tests: INVENTORY
                .iter()
                .enumerate()
                .map(|(i, (a, c, e))| Test {
                    index: i as u64 + 1,
                    arm: a.to_string(),
                    control: c.to_string(),
                    eligibility: *e,
                })
                .collect(),
        }
    }
    #[test]
    fn steer_real_statistics_with_invented_predictions() {
        let input = invented();
        let reg = registration("q12-a-v1");
        let value = evaluate(&input, &reg);
        assert_eq!(value["state"], "COMPLETE_FOR_S", "{value}");
        assert_eq!(value["tests"].as_array().unwrap().len(), 12);
        for i in IDENTICAL {
            let t = &value["tests"][i as usize - 1];
            assert_eq!(t["state"], "IDENTICAL_BY_CONSTRUCTION");
            assert_eq!(t["label"], "identical by construction, p = 1");
            assert_eq!(t["p"]["decision"], 1.);
            assert_eq!(t["seed_unit"]["se"], 0.);
            assert!(!value["holm_reject"][i as usize - 1].as_bool().unwrap());
        }
        for i in [1usize, 4] {
            // R−Control slots2/5.
            assert_eq!(value["tests"][i]["w2"]["interval"]["estimate"], 0.);
            assert_eq!(value["tests"][i]["w2"]["interval"]["se"], 0.);
            assert_eq!(value["tests"][i]["w2"]["point_guard"], false);
        }
        for i in [0usize, 3] {
            assert_eq!(value["tests"][i]["state"], "COMPLETE");
            assert!(value["holm_reject"][i].as_bool().unwrap(), "{value}");
            assert_eq!(value["tests"][i]["w2"]["point_guard"], true);
            assert!(
                value["tests"][i]["dataset_guards"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|g| g["point_guard"] == true)
            );
        }
        // Optional caller-selected receipt, never external payload discovery.
        if let Ok(path) = std::env::var("STEER_INVENTED_RESULT") {
            save(Path::new(&path), &value).unwrap();
        }
    }
    #[test]
    fn steer_fix2_canonical_completeness_boundaries() {
        // Primary/guard numbers come from the actual 10,000-draw owners above,
        // not injected statistics. Only the four reviewed boundary mutations
        // below are injected after computation, at the real completion owner.
        let baseline = evaluate(&invented(), &registration("q12-a-v1"));
        assert_eq!(baseline["state"], "COMPLETE_FOR_S");
        let zero = vec![vec![0.; 4]; 10];
        let refusal = seed_stat(&zero).unwrap_err();
        assert_eq!(refusal.to_string(), "undefined seed SE");
        let nested = json!({"state":"INCOMPLETE","reason":refusal.to_string()});
        let tests = || baseline["tests"].as_array().unwrap().clone();
        let mut primary_seed = tests();
        primary_seed[0]["seed_unit"] = nested.clone();
        let primary_seed = finish_packet(primary_seed, "q12-a-v1");
        let mut w2_seed = tests();
        w2_seed[0]["w2"]["seed_unit"] = nested;
        let w2_seed = finish_packet(w2_seed, "q12-a-v1");
        for result in [&primary_seed, &w2_seed] {
            assert_eq!(result["state"], "COMPLETE_FOR_S");
            assert_eq!(result["tests"][0]["state"], "COMPLETE");
            assert_eq!(result["holm_reject"], baseline["holm_reject"]);
            assert_eq!(result["tests"][0]["p"], baseline["tests"][0]["p"]);
        }
        assert_eq!(primary_seed["tests"][0]["seed_unit"]["state"], "INCOMPLETE");
        assert_eq!(
            w2_seed["tests"][0]["w2"]["seed_unit"]["state"],
            "INCOMPLETE"
        );
        let mut invalid = tests();
        invalid[6]["dataset_guards"] = json!([]);
        // JSON represents a nonfinite estimate/SE as null. This must not be
        // confused with a finite false diagnostic point guard.
        invalid[6]["w2"]["interval"]["se"] = json!(f64::NAN);
        let invalid = finish_packet(invalid, "q12-a-v1");
        assert_eq!(invalid["state"], "INCOMPLETE");
        assert_eq!(invalid["tests"][6]["state"], "INCOMPLETE");
        assert_eq!(invalid["tests"][6]["p"]["decision"], 1.);
        assert_eq!(invalid["holm_reject"].as_array().unwrap().len(), 12);
        let mut finite_false = tests();
        for g in finite_false[6]["dataset_guards"].as_array_mut().unwrap() {
            g["interval"]["estimate"] = json!(-0.01);
            g["point_guard"] = json!(false);
        }
        finite_false[6]["w2"]["interval"]["estimate"] = json!(-0.01);
        finite_false[6]["w2"]["interval"]["se"] = json!(0.001);
        finite_false[6]["w2"]["point_guard"] = json!(false);
        let finite_false = finish_packet(finite_false, "q12-a-v1");
        assert_eq!(finite_false["state"], "COMPLETE_FOR_S");
        assert_eq!(finite_false["holm_reject"], baseline["holm_reject"]);
        for i in [7, 10] {
            let alias = &baseline["tests"][i];
            assert_eq!(alias["dataset_guards"].as_array().unwrap().len(), 4);
            assert_eq!(alias["w2"]["interval"]["estimate"], 0.);
            assert_eq!(alias["w2"]["interval"]["se"], 0.);
            assert_eq!(alias["w2"]["point_guard"], false);
            assert_eq!(
                alias["w2"]["seed_unit"]["state"],
                "IDENTICAL_BY_CONSTRUCTION"
            );
        }
        // Prove the actual guard interval admission rejects a nonfinite SE.
        let input = invented();
        let mut iv = input
            .interval(&input.tests[6], &[0, 1], 705, false, true, 0.)
            .unwrap();
        admitted_interval(&iv).unwrap();
        iv.se = f64::NAN;
        assert!(admitted_interval(&iv).is_err());
        if let Ok(path) = std::env::var("STEER_FIX2_RESULT") {
            save(
                Path::new(&path),
                &json!({"scope":"INVENTED_CANONICAL_OWNER_BOUNDARIES",
                "baseline":baseline,"primary_seed_zero":primary_seed,"w2_seed_zero":w2_seed,
                "test7_invalid_guards":invalid,"test7_finite_false_guards":finite_false}),
            )
            .unwrap();
        }
    }

    #[test]
    fn steer_fix2_actual_seed_diagnostic_refusals_are_nonfatal() {
        let mut primary = invented();
        let permutation = [
            31, 13, 25, 2, 12, 7, 4, 11, 32, 26, 18, 30, 17, 28, 33, 14, 6, 23, 9, 34, 0, 22, 3,
            24, 5, 16, 15, 27, 19, 21, 20, 8, 1, 29, 35, 10,
        ];
        for d in &mut primary.datasets {
            for family in ["a", "c"] {
                let control: Vec<f64> = permutation.iter().map(|x| f64::from(*x)).collect();
                let smooth: Vec<f64> = control.iter().map(|x| 35. - x).collect();
                assert_eq!(zenstats::spearman(&control, &d.target), 0.);
                assert_eq!(zenstats::spearman(&smooth, &d.target), 0.);
                d.predictions
                    .insert(format!("base-{family}"), vec![control.clone(); 10]);
                d.predictions
                    .insert(format!("r-{family}"), vec![control; 10]);
                d.predictions
                    .insert(format!("s-{family}"), vec![smooth.clone(); 10]);
                d.predictions
                    .insert(format!("sr-{family}"), vec![smooth; 10]);
            }
        }
        validate(&primary).unwrap();
        assert_eq!(
            seed_component(&primary, &primary.tests[0]).unwrap()["state"],
            "INCOMPLETE"
        );
        let value = evaluate(&primary, &registration("q12-a-v1"));
        assert_eq!(value["state"], "COMPLETE_FOR_S", "{value}");
        assert_eq!(value["tests"][0]["state"], "COMPLETE");
        assert!(value["tests"][0]["theta"]["se"].as_f64().unwrap() > 0.);
        let mut w2 = invented();
        for d in &mut w2.datasets {
            for family in ["a", "c"] {
                let control: Vec<Vec<_>> = (0..10)
                    .map(|seed| {
                        d.target
                            .iter()
                            .enumerate()
                            .map(|(i, y)| y + ((seed % 5) + 1) as f64 * (i % 3) as f64)
                            .collect()
                    })
                    .collect();
                d.predictions
                    .insert(format!("base-{family}"), control.clone());
                d.predictions.insert(format!("r-{family}"), control);
            }
        }
        validate(&w2).unwrap();
        assert!(
            seed_component(&w2, &w2.tests[0])
                .unwrap()
                .get("reason")
                .is_none()
        );
        assert_eq!(
            w2_seed_component(&w2, &w2.tests[0])
                .unwrap_err()
                .to_string(),
            "undefined seed SE"
        );
        let value = evaluate(&w2, &registration("q12-a-v1"));
        assert_eq!(value["state"], "COMPLETE_FOR_S", "{value}");
        assert_eq!(value["tests"][0]["state"], "COMPLETE");
        assert_eq!(value["tests"][0]["w2"]["seed_unit"]["state"], "INCOMPLETE");
        assert_eq!(value["tests"][0]["w2"]["point_guard"], false);
    }

    #[test]
    fn steer_invalid_admission_retains_all_slots() {
        let reg = registration("q12-a-v1");
        let mut cases = Vec::new();
        let mut v = invented();
        v.tests.pop();
        cases.push(v);
        let mut v = invented();
        v.tests[0].arm = "r-a".into();
        cases.push(v);
        let mut v = invented();
        v.datasets[0].predictions.get_mut("r-a").unwrap()[0][0] += 1.;
        cases.push(v);
        let mut v = invented();
        v.datasets[0].target.fill(1.);
        cases.push(v);
        let mut v = invented();
        v.datasets[0].predictions.get_mut("s-a").unwrap()[0].pop();
        cases.push(v);
        let mut v = invented();
        v.datasets[0].predictions.get_mut("s-a").unwrap()[0][0] = f64::NAN;
        cases.push(v);
        for v in cases {
            let result = evaluate(&v, &reg);
            assert_eq!(result["state"], "INCOMPLETE");
            assert_eq!(result["tests"].as_array().unwrap().len(), 12);
            assert!(
                result["tests"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|t| t["p"]["decision"] == 1.)
            );
        }
    }
    #[test]
    fn steer_policy_pending_and_seed_refusal_are_explicit() {
        let mut reg = registration("q12-a-v1");
        reg.q12_approved = false;
        assert_eq!(evaluate(&invented(), &reg)["state"], "PENDING_OWNER_Q12");
        assert!(seed_stat(&vec![vec![0.; 4]; 10]).is_err());
        let input = invented();
        // B's packet stays diagnostic-only through primary zero-SE refusal,
        // not through the nonfatal report-only seed state. Improvement aliases
        // have no margin addition, so this primary variance is exactly zero.
        let strict = evaluate(&input, &registration("e34-strict-b-v1"));
        assert_eq!(strict["state"], "INCOMPLETE");
        assert_eq!(strict["scope"], "DIAGNOSTIC_ONLY");
        for index in [7, 10] {
            assert_eq!(strict["tests"][index]["state"], "INCOMPLETE");
            assert_eq!(strict["tests"][index]["theta"]["se"], 0.);
            assert_eq!(strict["tests"][index]["p"]["decision"], 1.);
        }
        assert_eq!(strict["tests"][1]["w2"]["point_guard"], false);
        // An observed identity outside the declared list cannot acquire A status.
        let mut accidental = invented();
        for d in &mut accidental.datasets {
            let p = d.predictions["base-a"].clone();
            d.predictions.insert("s-a".into(), p.clone());
            d.predictions.insert("sr-a".into(), p);
        }
        // R−S (slot7) was not declared identical. Observed equality cannot
        // acquire alias status; its actual zero primary SE blocks the packet.
        let refused = slot(&accidental, &accidental.tests[6], "q12-a-v1").unwrap();
        assert_eq!(refused["theta"]["se"], 0.);
        assert_eq!(refused["state"], "INCOMPLETE");
        assert_eq!(
            evaluate(&accidental, &registration("q12-a-v1"))["state"],
            "INCOMPLETE"
        );
    }
    #[test]
    fn steer_strict_guard_and_nonzero_constant_seed_refusal() {
        assert!(!w2_guard(0., 0.));
        assert!(!w2_guard(-0.002, 0.001));
        assert!(w2_guard(-0.001999, 0.001));
        assert!(seed_stat(&vec![vec![0.5; 4]; 10]).is_err());
    }
    #[test]
    fn steer_pending_policy_does_not_open_input() {
        let base = std::env::var_os("TMPDIR")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .or_else(|| std::env::var_os("USERPROFILE"))
                    .map(|home| PathBuf::from(home).join("tmp"))
            })
            .expect("disk-backed scratch or user profile required");
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = base.join(format!(
            "steer-pending-policy-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let pin = root.join("registration.json");
        fs::write(&pin,serde_json::to_vec(&json!({"schema":"steer-registration-v1","alias_policy":"q12-a-v1",
            "q12_approved":false,"identical_tests":IDENTICAL,"fits":172,"tau":0.125,"data_sha256":DATA_SHA})).unwrap()).unwrap();
        let input = root.join("must-not-exist.json");
        assert!(!input.exists());
        let output = root.join("pending.json");
        run(Cli {
            mode: "steer-evaluate".into(),
            registration: Some(pin),
            input,
            output: output.clone(),
        })
        .unwrap();
        let result: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert_eq!(result["state"], "PENDING_OWNER_Q12");
    }
}
