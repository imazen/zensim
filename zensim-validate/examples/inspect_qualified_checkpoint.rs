//! Read the deployed model through the canonical loader, including compressed metadata.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected model path")?;
    let bytes = std::fs::read(path)?;
    let model = zenpredict::Model::from_bytes(&bytes)?;
    let repro: serde_json::Value =
        serde_json::from_str(model.metadata().get_utf8("zentrain.repro")?)?;
    if std::env::args().nth(2).as_deref() == Some("--e29-research") {
        assert_eq!(repro["hdr_consensus_research"], true);
        assert_eq!(repro["table_admission"]["qualified_provenance"], false);
        assert_eq!(repro["table_admission"]["formula_revision"], 5);
        assert!(repro["table_admission"]["historical_replay"].is_null());
        assert_eq!(model.metadata().get_utf8("zentrain.formula_revision")?, "5");
        assert!(zensim_validate::feature_set::bake_declared_training_set(&model).is_none());
        let hdr: Vec<_> = repro["table_admission"]["tables"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|t| t["stored_declarations"]["study"] == "E29")
            .collect();
        assert_eq!(hdr.len(), 1);
        assert_eq!(hdr[0]["stored_declarations"]["role"], "train");
        assert_eq!(hdr[0]["stored_declarations"]["rows"], 7390);
        println!(
            "{}",
            serde_json::json!({"status":"PASS", "formula_revision":5,
            "qualified_provenance":false, "feature_set_id":null, "repro":repro})
        );
        return Ok(());
    }
    if std::env::args().nth(2).as_deref() == Some("--e31-research") {
        let admission = &repro["table_admission"];
        assert_eq!(admission["qualified_provenance"], false);
        assert_eq!(admission["formula_revision"], 5);
        assert!(admission["historical_replay"].is_null());
        assert_eq!(model.metadata().get_utf8("zentrain.formula_revision")?, "5");
        assert!(zensim_validate::feature_set::bake_declared_training_set(&model).is_none());
        assert_eq!(
            admission["tables"]
                .as_array()
                .ok_or("missing SDR tables")?
                .len(),
            7
        );
        let native = &admission["upiq380"];
        assert_eq!(
            native["manifest_sha256"],
            "2da346bb17e08a4a63aae9ed89b159b36edd331199a9dd1c42b4a05b53ac939e"
        );
        assert_eq!(
            native["table_sha256"],
            "7f09debedc591e7dd3494846ada9fe0c93b01779918b8358e2cf8053f5f1a6c4"
        );
        assert_eq!(
            native["keys_sha256"],
            "c47ca1c12d1e8e206464938884ff9d05baa8274785826060604d16caf0bff49e"
        );
        assert_eq!(native["label_disposition"]["state"], "approved");
        assert_eq!(native["native_width"], 1825);
        assert_eq!(native["logical_width"], 1853);
        println!(
            "{}",
            serde_json::json!({"status":"PASS", "formula_revision":5,
            "qualified_provenance":false, "feature_set_id":null, "repro":repro})
        );
        return Ok(());
    }
    assert_eq!(repro["table_admission"]["qualified_provenance"], true);
    assert_eq!(repro["table_admission"]["formula_revision"], 5);
    assert!(repro["table_admission"]["historical_replay"].is_null());
    assert_eq!(model.metadata().get_utf8("zentrain.formula_revision")?, "5");
    let declared = zensim_validate::feature_set::bake_declared_training_set(&model)
        .ok_or("missing declared training feature set")?;
    if repro["table_admission"]["research_family"] == "palette_v2" {
        let transport: serde_json::Value = serde_json::from_str(include_str!(
            "../../benchmarks/e32_palette_training_transport_2026-10-07.json"
        ))?;
        assert_eq!(declared.to_string(), transport["feature_set_id"]);
        assert_eq!(repro["table_admission"]["serving_allowed"], false);
        assert_eq!(repro["keep_features_n"], 462);
    } else {
        assert_eq!(
            declared.to_string(),
            "basic+peaks+v2@w1825/rev5_localwin#36c3f3af"
        );
    }
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
