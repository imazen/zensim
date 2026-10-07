//! Read the deployed model through the canonical loader, including compressed metadata.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected model path")?;
    let bytes = std::fs::read(path)?;
    let model = zenpredict::Model::from_bytes(&bytes)?;
    let repro: serde_json::Value =
        serde_json::from_str(model.metadata().get_utf8("zentrain.repro")?)?;
    assert_eq!(repro["table_admission"]["qualified_provenance"], true);
    assert_eq!(repro["table_admission"]["formula_revision"], 5);
    assert!(repro["table_admission"]["historical_replay"].is_null());
    assert_eq!(model.metadata().get_utf8("zentrain.formula_revision")?, "5");
    let declared = zensim_validate::feature_set::bake_declared_training_set(&model)
        .ok_or("missing declared training feature set")?;
    assert_eq!(
        declared.to_string(),
        "basic+peaks+v2@w1825/rev5_localwin#36c3f3af"
    );
    println!(
        "{}",
        serde_json::json!({"status":"PASS", "formula_revision":5,
            "qualified_provenance":true, "feature_set_id":declared.to_string(),
            "checkpoint_epoch":repro["checkpoint_epoch"],
            "pair_sampling":repro["pair_sampling"],
            "admitted_tables":repro["table_admission"]["tables"].as_array().unwrap().len(),
            "repro": repro})
    );
    Ok(())
}
