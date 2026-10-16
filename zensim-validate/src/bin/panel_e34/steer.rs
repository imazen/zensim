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
    let mut complete = p["complete"] == true && seed["state"] != "INCOMPLETE";
    let mut guards = Vec::new();
    let w2 = if t.eligibility {
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
        complete &= ws["state"] != "INCOMPLETE";
        json!({"interval":reported(&iv,10,alias),"point_guard":w2_guard(iv.estimate, iv.se),"seed_unit":ws})
    } else {
        json!({"state":"NOT_REQUIRED_BY_MANIFEST"})
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
        "p":p,"dataset_guards":if t.eligibility {json!(guards)}else{json!({"state":"NOT_REQUIRED_BY_MANIFEST"})},"w2":w2,
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
    let mut ps = Vec::new();
    let mut complete = true;
    for t in &input.tests {
        let result=slot(input,t,&reg.alias_policy).unwrap_or_else(|e|json!({"test":t.index,"arm":t.arm,"control":t.control,"state":"INCOMPLETE","reason":e.to_string(),"p":{"decision":1.}}));
        complete &= matches!(
            result["state"].as_str(),
            Some("COMPLETE" | "IDENTICAL_BY_CONSTRUCTION")
        );
        ps.push(result["p"]["decision"].as_f64().unwrap_or(1.));
        tests.push(result);
    }
    json!({"schema":"steer-statistics-v1","state":if complete {"COMPLETE_FOR_S"}else{"INCOMPLETE"},
        "alias_policy":reg.alias_policy,"scope":if reg.alias_policy=="q12-a-v1" {"S_ONLY"}else{"DIAGNOSTIC_ONLY"},
        "eligibility_margin":ELIGIBILITY_MARGIN,"resamples":10000,"level":0.95,"percentile":"Hyndman-Fan type 7","base_seed":20261011,
        "tests":tests,"holm_adjusted":zenstats::multiple::holm(&ps),"holm_reject":zenstats::multiple::holm_reject(&ps,0.05)})
}

pub(super) fn run(cli: Cli) -> Result<()> {
    if cli.mode == "steer-pack-check" {
        let bytes = fs::read(&cli.input)?;
        let models: BTreeMap<String, Value> = serde_json::from_slice(&bytes)?;
        let expected: BTreeSet<_> = ["base-a", "base-c", "s-a", "s-c"]
            .into_iter()
            .flat_map(|a| (0..3).map(move |s| format!("{a}-s{s}")))
            .collect();
        ensure!(
            models.keys().cloned().collect::<BTreeSet<_>>() == expected,
            "twelve exact full-data model names required"
        );
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
        let refused = slot(&input, &input.tests[1], "e34-strict-b-v1").unwrap();
        assert_eq!(refused["state"], "INCOMPLETE");
        assert_eq!(refused["w2"]["point_guard"], false);
        // An observed identity outside the declared list cannot acquire A status.
        let mut accidental = invented();
        for d in &mut accidental.datasets {
            let p = d.predictions["base-a"].clone();
            d.predictions.insert("s-a".into(), p.clone());
            d.predictions.insert("sr-a".into(), p);
        }
        let refused = slot(&accidental, &accidental.tests[0], "q12-a-v1").unwrap();
        assert_eq!(refused["state"], "INCOMPLETE");
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
        let root =
            PathBuf::from(std::env::var_os("TMPDIR").expect("caller must set disk-backed TMPDIR"))
                .join(format!("steer-pending-policy-{}", std::process::id()));
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
