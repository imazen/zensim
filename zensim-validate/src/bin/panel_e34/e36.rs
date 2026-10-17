//! E36's fixed two-slot family over the unchanged E34 numerical owner.
use super::*;
const ARMS: [&str; 3] = ["tv0-c", "tv1-c", "tv10-c"];
fn failure(reason: &str) -> Value {
    json!({"schema":"e36-statistics-v1","state":"INCOMPLETE","decision_allowed":false,"reason":reason,
        "tests":[{"test":1,"arm":"tv1-c","control":"tv0-c","state":"INCOMPLETE","p":{"decision":1.}},
                 {"test":2,"arm":"tv10-c","control":"tv0-c","state":"INCOMPLETE","p":{"decision":1.}}],
        "holm_family_size":2,"holm_adjusted":zenstats::multiple::holm(&[1.,1.]),
        "holm_reject":zenstats::multiple::holm_reject(&[1.,1.],0.05)})
}
fn evaluate(input: &Input) -> Result<Value> {
    if input.schema != "e36-paired-seed-v1"
        || input
            .datasets
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>()
            != DATASETS
        || input.tests.len() != 2
        || !input.tests.iter().enumerate().all(|(i, t)| {
            t.index == i as u64 + 1 && t.arm == ARMS[i + 1] && t.control == ARMS[0] && t.eligibility
        })
    {
        return Ok(failure(
            "fixed C-family lambda1/lambda10 inventory required",
        ));
    }
    let mut admitted = input.clone();
    // Missing or malformed treatment panels occupy their original p=1 slot.
    // A complete other treatment remains assessable in the same size-two family.
    for arm in ARMS {
        let complete = input.datasets.iter().all(|d| {
            d.predictions.get(arm).is_some_and(|p| {
                p.len() == 10
                    && p.iter()
                        .all(|r| r.len() == d.target.len() && r.iter().all(|v| v.is_finite()))
            })
        });
        if !complete {
            for d in &mut admitted.datasets {
                d.predictions.remove(arm);
            }
        }
    }
    if admitted
        .datasets
        .iter()
        .any(|d| d.predictions.keys().any(|k| !ARMS.contains(&k.as_str())))
    {
        return Ok(failure("unregistered model in paired panels"));
    }
    if let Err(e) = admitted.validate_rows() {
        return Ok(failure(&e.to_string()));
    }
    let tests = admitted
        .tests
        .iter()
        .map(|t| registered::slot(&admitted, t, 0, false))
        .collect::<Result<Vec<_>>>()?;
    let ps: Vec<_> = tests
        .iter()
        .map(|t| t["p"]["decision"].as_f64().unwrap_or(1.))
        .collect();
    let complete = tests.iter().all(|t| t["state"] == "COMPLETE");
    Ok(
        json!({"schema":"e36-statistics-v1","state":if complete {"COMPLETE"} else {"INCOMPLETE"},
        "decision_allowed":complete,"tests":tests,"holm_family_size":2,
        "holm_adjusted":zenstats::multiple::holm(&ps),"holm_reject":zenstats::multiple::holm_reject(&ps,0.05),
        "eligibility_margin":ELIGIBILITY_MARGIN,"mean_point_guard":-0.002,"dataset_point_guard":-0.005,
        "w2_guard":"delta > -2*SE","resamples":10000,"level":0.95,"percentile":"Hyndman-Fan type 7",
        "base_seed":20261011,"numerical_owner":"81a545a3ff3457c9fabae93bc9d262b19b50952e",
        "helper_commit":"48665c8fb1bb3de5c99d792ae4ed4f3f1e854629"}),
    )
}
fn safe_metadata(path: &Path) -> Result<()> {
    for candidate in [path.to_path_buf(), fs::canonicalize(path)?] {
        ensure!(
            !candidate.components().any(|c| {
                let name = c.as_os_str().to_string_lossy().to_lowercase();
                name.starts_with("_sealed")
                    || name.contains("holdout")
                    || name.starts_with("labels__")
            }),
            "E36 protected label path refused"
        );
    }
    Ok(())
}
fn is_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Assessment {
    schema: String,
    program_sha256: String,
    fit_plan: PathBuf,
    fit_plan_sha256: String,
    paired: PathBuf,
    paired_sha256: String,
    bindings: PathBuf,
    bindings_sha256: String,
    train_selection: PathBuf,
    train_selection_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    arm: String,
    heldout: String,
    seed_index: usize,
    init_seed: u64,
    sample_seed: u64,
    bake_sha256: Option<String>,
    fit_receipt: Option<PathBuf>,
    fit_receipt_sha256: String,
    qualification: Option<PathBuf>,
    qualification_sha256: Option<String>,
    model_binding: Option<PathBuf>,
    model_binding_sha256: Option<String>,
    withheld_source_excluded: bool,
    state: String,
    final_epoch: usize,
    role: String,
}
fn bound_metadata(path: &Path, pin: &str) -> Result<Value> {
    safe_metadata(path)?;
    let bytes = fs::read(path)?;
    ensure!(
        is_sha(pin) && digest(&bytes) == pin,
        "E36 native metadata hash mismatch"
    );
    Ok(serde_json::from_slice(&bytes)?)
}
fn verify_native_records(
    c: &Binding,
    plan: &Value,
    fleet: &Value,
    q: &Value,
    b: &Value,
) -> Result<()> {
    let cell = format!("e36-{}__N/without_{}_s{}", c.arm, c.heldout, c.seed_index);
    let slots: Vec<_> = plan["cells"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("E36 frozen slots"))?
        .iter()
        .filter(|r| r["name"] == cell)
        .collect();
    ensure!(
        slots.len() == 1
            && slots[0]["kind"] == "lodo"
            && slots[0]["arm"] == c.arm
            && slots[0]["heldout"] == c.heldout
            && slots[0]["seed_index"] == c.seed_index,
        "E36 source-withheld slot absent from exact frozen fit plan"
    );
    let bake = c
        .bake_sha256
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("E36 complete bake"))?;
    ensure!(
        fleet["schema"] == "zenfleet-fit-cell-receipt-v1"
            && fleet["cell"] == cell
            && fleet["program_sha"] == plan["program_sha256"]
            && fleet["data_sha"] == plan["data_sha256"]
            && fleet["argv_sha"] == slots[0]["argv_sha256"]
            && fleet["selected_bake_sha"] == bake
            && fleet["files"]["e36-qualified-checkpoint.json"].as_str()
                == c.qualification_sha256.as_deref()
            && fleet["files"]["e36-model-binding.json"].as_str()
                == c.model_binding_sha256.as_deref(),
        "E36 source-withheld native fleet identity differs"
    );
    ensure!(
        q["schema"] == "e36-qualified-checkpoint-v1"
            && q["qualified"] == true
            && q["execution_contract"] == "registered-fit"
            && q["cell"] == cell
            && q["program_sha256"] == plan["program_sha256"]
            && q["data_sha256"] == plan["data_sha256"]
            && q["argv_sha256"] == slots[0]["argv_sha256"]
            && q["result_sha256"] == fleet["result_sha"]
            && q["raw_checkpoint_sha256"] == bake
            && q["packed_model_sha256"].is_null()
            && q["model_binding_sha256"].as_str() == c.model_binding_sha256.as_deref(),
        "E36 complete assessment model requires native checkpoint qualification"
    );
    for key in [
        "inspector_sha256",
        "qualification_owner_sha256",
        "e36_owner_sha256",
    ] {
        ensure!(
            plan["qualification_owners"][key]
                .as_str()
                .is_some_and(is_sha)
                && q[key] == plan["qualification_owners"][key],
            "E36 qualification owner differs"
        );
    }
    ensure!(
        b["schema"] == "e36-fit-model-binding-v1"
            && b["role"] == "source-withheld"
            && b["heldout"] == c.heldout
            && b["arm"] == c.arm
            && b["seed_index"] == c.seed_index
            && b["init_seed"] == c.init_seed
            && b["sample_seed"] == c.sample_seed
            && b["selected_epoch"] == 119
            && b["raw_checkpoint_sha256"] == bake
            && b["packed_model_sha256"].is_null()
            && b["program_sha256"] == plan["program_sha256"]
            && b["manifest_sha256"] == plan["fit_manifest_sha256"]
            && b["contract_sha256"] == plan["contract_sha256"]
            && b["base_filter_sha256"] == plan["base_filter_sha256"]
            && b["trainer_sha256"] == plan["trainer_sha256"],
        "E36 full-data/other-source/other-build model refused in primary assessment"
    );
    Ok(())
}
pub(super) fn run(cli: Cli) -> Result<()> {
    safe_metadata(&cli.input)?;
    let request: Assessment = serde_json::from_slice(&fs::read(&cli.input)?)?;
    ensure!(
        request.schema == "e36-assessment-job-v1" && is_sha(&request.program_sha256),
        "E36 assessment job schema"
    );
    ensure!(
        std::env::var("ZEN_FIT_PROGRAM_SHA").as_deref() == Ok(request.program_sha256.as_str()),
        "E36 human assessment must run inside the bound fit-cell-exec program"
    );
    // Lock the one TRAIN-only choice before opening any paired target payload.
    for path in [
        &request.train_selection,
        &request.bindings,
        &request.paired,
        &request.fit_plan,
    ] {
        safe_metadata(path)?;
    }
    let selection_bytes = fs::read(&request.train_selection)?;
    ensure!(
        digest(&selection_bytes) == request.train_selection_sha256,
        "E36 TRAIN selection hash"
    );
    let selection: Value = serde_json::from_slice(&selection_bytes)?;
    let plan = bound_metadata(&request.fit_plan, &request.fit_plan_sha256)?;
    ensure!(
        selection["schema"] == "e36-global-train-selection-v1"
            && selection["selection_locked_before_assessment"] == true
            && selection["labels_read"] == false
            && selection["fits"] == 0
            && selection["fit_plan_sha256"] == request.fit_plan_sha256
            && plan["schema"] == "e36-future-fit-plan-v1"
            && plan["program_sha256"] == request.program_sha256
            && plan["fit_release"] == false
            && (selection["selected_arm"].is_null()
                || ["tv1-c", "tv10-c"].contains(&selection["selected_arm"].as_str().unwrap_or(""))),
        "E36 frozen global TRAIN selection required"
    );
    let bindings = fs::read(&request.bindings)?;
    ensure!(
        digest(&bindings) == request.bindings_sha256,
        "E36 assessment input hash mismatch"
    );
    let cells: Vec<Binding> = serde_json::from_slice(&bindings)?;
    ensure!(
        cells.len() == 120,
        "exactly120 source-withheld assessment bindings; full-data models cannot enter primary assessment"
    );
    let init = [1101, 1103, 1107, 1109, 1117, 1123, 1129, 1151, 1153, 1163];
    let mut keys = BTreeSet::new();
    let mut unavailable = BTreeSet::new();
    for c in cells {
        let fold = DATASETS
            .iter()
            .position(|&s| s == c.heldout)
            .ok_or_else(|| anyhow::anyhow!("E36 withheld source"))?;
        if c.state != "complete" {
            unavailable.insert(c.arm.clone());
        }
        ensure!(
            ARMS.contains(&c.arm.as_str())
                && c.seed_index < 10
                && c.init_seed == init[c.seed_index]
                && c.sample_seed == 101 + ((c.seed_index + fold) % 10) as u64 * 100_000_000
                && c.withheld_source_excluded
                && c.final_epoch == 119
                && c.role == "source-withheld-primary"
                && is_sha(&c.fit_receipt_sha256)
                && match c.state.as_str() {
                    "complete" => c.bake_sha256.as_ref().is_some_and(|s| is_sha(s)),
                    "failed" | "unavailable" => c.bake_sha256.is_none(),
                    _ => false,
                }
                && keys.insert((c.arm.clone(), c.heldout.clone(), c.seed_index)),
            "E36 immutable matched seed/source/final-epoch binding"
        );
        if c.state == "complete" {
            let fleet = bound_metadata(
                c.fit_receipt
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("E36 native fleet receipt required"))?,
                &c.fit_receipt_sha256,
            )?;
            let qualification = bound_metadata(
                c.qualification
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("E36 native qualification required"))?,
                c.qualification_sha256.as_deref().unwrap_or(""),
            )?;
            let binding = bound_metadata(
                c.model_binding
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("E36 native model binding required"))?,
                c.model_binding_sha256.as_deref().unwrap_or(""),
            )?;
            verify_native_records(&c, &plan, &fleet, &qualification, &binding)?;
        }
    }
    let bytes = fs::read(&request.paired)?;
    ensure!(
        digest(&bytes) == request.paired_sha256,
        "E36 paired target payload hash mismatch"
    );
    let mut input: Input = serde_json::from_slice(&bytes)?;
    for dataset in &mut input.datasets {
        for arm in &unavailable {
            dataset.predictions.remove(arm);
        }
    }
    let mut result = evaluate(&input)?;
    let selected = selection["selected_arm"].as_str();
    let selected_index = selected.and_then(|s| ARMS[1..].iter().position(|&a| a == s));
    let decision_allowed =
        selected_index.is_some_and(|i| result["tests"][i]["state"] == "COMPLETE");
    let eligible = selected_index.is_some_and(|i| {
        let t = &result["tests"][i];
        decision_allowed
            && result["holm_reject"][i] == true
            && t["e33_mean_point_guard"] == true
            && t["w2"]["point_guard"] == true
            && t["dataset_guards"]
                .as_array()
                .is_some_and(|g| g.len() == 4 && g.iter().all(|v| v["point_guard"] == true))
    });
    result["decision_allowed"] = json!(decision_allowed);
    result["selected_arm"] = json!(selected);
    result["selected_arm_noninferior_and_guards"] = json!(eligible);
    result["fallback_arm_substitution_allowed"] = json!(false);
    result["train_selection_sha256"] = json!(request.train_selection_sha256);
    result["paired_sha256"] = json!(request.paired_sha256);
    result["bindings_sha256"] = json!(request.bindings_sha256);
    result["fit_program_sha256"] = json!(request.program_sha256);
    result["fit_plan_sha256"] = json!(request.fit_plan_sha256);
    result["native_source_withheld_qualification_verified"] = json!(true);
    save(&cli.output, &result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn e36_assessment_requires_qualified_withheld_models() {
        // Invented metadata only. No model, label payload or optimizer.
        let c: Binding = serde_json::from_value(json!({
            "arm":"tv1-c","heldout":"kadid","seed_index":0,"init_seed":1101,"sample_seed":101,
            "bake_sha256":"b".repeat(64),"fit_receipt_sha256":"f".repeat(64),
            "qualification_sha256":"c".repeat(64),"model_binding_sha256":"d".repeat(64),
            "withheld_source_excluded":true,"state":"complete","final_epoch":119,"role":"source-withheld-primary"
        })).unwrap();
        let cell = "e36-tv1-c__N/without_kadid_s0";
        let plan = json!({"program_sha256":"a".repeat(64),"data_sha256":"e".repeat(64),
            "fit_manifest_sha256":"1".repeat(64),"contract_sha256":"2".repeat(64),
            "base_filter_sha256":"3".repeat(64),"trainer_sha256":"4".repeat(64),
            "qualification_owners":{"inspector_sha256":"5".repeat(64),"qualification_owner_sha256":"6".repeat(64),"e36_owner_sha256":"7".repeat(64)},
            "cells":[{"name":cell,"kind":"lodo","arm":"tv1-c","heldout":"kadid","seed_index":0,"argv_sha256":"8".repeat(64)}]});
        let fleet = json!({"schema":"zenfleet-fit-cell-receipt-v1","cell":cell,
            "program_sha":"a".repeat(64),"data_sha":"e".repeat(64),"argv_sha":"8".repeat(64),
            "selected_bake_sha":"b".repeat(64),"result_sha":"9".repeat(64),
            "files":{"e36-qualified-checkpoint.json":"c".repeat(64),"e36-model-binding.json":"d".repeat(64)}});
        let q = json!({"schema":"e36-qualified-checkpoint-v1","qualified":true,"execution_contract":"registered-fit","cell":cell,
            "program_sha256":"a".repeat(64),"data_sha256":"e".repeat(64),"argv_sha256":"8".repeat(64),
            "result_sha256":"9".repeat(64),"raw_checkpoint_sha256":"b".repeat(64),"packed_model_sha256":null,
            "model_binding_sha256":"d".repeat(64),"inspector_sha256":"5".repeat(64),
            "qualification_owner_sha256":"6".repeat(64),"e36_owner_sha256":"7".repeat(64)});
        let b = json!({"schema":"e36-fit-model-binding-v1","role":"source-withheld","heldout":"kadid","arm":"tv1-c",
            "seed_index":0,"init_seed":1101,"sample_seed":101,"selected_epoch":119,
            "raw_checkpoint_sha256":"b".repeat(64),"packed_model_sha256":null,"program_sha256":"a".repeat(64),
            "manifest_sha256":"1".repeat(64),"contract_sha256":"2".repeat(64),"base_filter_sha256":"3".repeat(64),"trainer_sha256":"4".repeat(64)});
        verify_native_records(&c, &plan, &fleet, &q, &b).unwrap();
        for (key, value) in [
            ("role", json!("full-data")),
            ("heldout", json!("tid2013")),
            ("sample_seed", json!(102)),
            ("base_filter_sha256", json!("0".repeat(64))),
            ("raw_checkpoint_sha256", json!("0".repeat(64))),
        ] {
            let mut wrong = b.clone();
            wrong[key] = value;
            assert!(
                verify_native_records(&c, &plan, &fleet, &q, &wrong).is_err(),
                "{key}"
            );
        }
        let mut unqualified = q.clone();
        unqualified["qualified"] = json!(false);
        assert!(verify_native_records(&c, &plan, &fleet, &unqualified, &b).is_err());
    }
    #[test]
    fn e36_one_available_slot_reuses_exact_owner_and_keeps_missing_p_one() {
        let mut input = super::super::tests::toy();
        input.schema = "e36-paired-seed-v1".into();
        input.tests = (1..=2)
            .map(|i| Test {
                index: i,
                arm: ARMS[i as usize].into(),
                control: ARMS[0].into(),
                eligibility: true,
            })
            .collect();
        for d in &mut input.datasets {
            let arm = d.predictions.remove("a").unwrap();
            let mut control = d.predictions.remove("b").unwrap();
            control.rotate_left(1);
            d.predictions.insert("tv1-c".into(), arm);
            d.predictions.insert("tv0-c".into(), control);
        }
        let expected = registered::slot(&input, &input.tests[0], 0, false).unwrap();
        let result = evaluate(&input).unwrap();
        assert_eq!(result["tests"][0], expected);
        assert_eq!(result["tests"][1]["p"]["decision"], 1.);
        let ps = [expected["p"]["decision"].as_f64().unwrap(), 1.];
        assert_eq!(
            result["holm_adjusted"],
            json!(zenstats::multiple::holm(&ps))
        );
        assert_eq!(
            result["holm_reject"],
            json!(zenstats::multiple::holm_reject(&ps, 0.05))
        );
    }
    #[test]
    fn e36_unavailable_arms_keep_both_holm_slots() {
        let mut input = super::super::tests::toy();
        input.schema = "e36-paired-seed-v1".into();
        input.tests = (1..=2)
            .map(|i| Test {
                index: i,
                arm: ARMS[i as usize].into(),
                control: ARMS[0].into(),
                eligibility: true,
            })
            .collect();
        for d in &mut input.datasets {
            d.predictions.clear();
        }
        let result = evaluate(&input).unwrap();
        assert_eq!(result["holm_family_size"], 2);
        assert_eq!(result["tests"].as_array().unwrap().len(), 2);
        assert_eq!(result["holm_adjusted"], json!([1., 1.]));
        assert_eq!(result["decision_allowed"], false);
        input.tests.pop();
        let invalid = evaluate(&input).unwrap();
        assert_eq!(invalid["holm_family_size"], 2);
    }
}
