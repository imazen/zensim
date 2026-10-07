//! Explicit E32 research projection. Legacy numeric auxiliary columns keep
//! their physical names; palette inputs are read only from named columns.

use std::collections::BTreeSet;
use std::path::Path;

use arrow::array::{Array, LargeStringArray, StringArray};
use arrow::datatypes::{DataType, Fields};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use zensim::feature_set_id::{ComputeParts, FeatureSetId, SlotSet};

const BASE: &str = "basic+peaks+v2@w1825/rev5_localwin#36c3f3af";
const PALETTE: &str = "palette@w1867/palette_v2#30b09cd1";
const IDS_SHA: &str = "207d6d29ed86395166444c60fb4fc788856818801e6b64bcbc36ed7b81d1773d";
const BANK_SHA: &str = "46587338cc74ba59e38fe96e637776bfe74135fde29f3f70e22b51605fc50917";
const INSTRUMENT_SHA: &str = "9f7523bf7d3aaa32418d40d83adb44edccff9e70cc75acc32eb8d5711fe89934";
const BINARY_SHA: &str = "575f39b4a0bb04e224ceca15eed5e878d53273511b884a895a3630e606d4ccc6";
const BUILD: &str = "e60a6ad74a47981f93f969d63b605c7a88ab09b0";

fn fail(message: &str) -> String {
    format!("E32 palette admission refused: {message}")
}

pub(crate) fn primary_ids() -> Vec<usize> {
    let ids: Value = serde_json::from_str(include_str!(
        "../../benchmarks/e32_palette_feature_ids_2026-10-07.json"
    ))
    .expect("registered E32 IDs");
    serde_json::from_value(ids["arm_ids"].clone()).expect("registered integer IDs")
}

pub(crate) fn identity() -> FeatureSetId {
    FeatureSetId::from_slots_with_layout(
        ComputeParts::parse("basic+peaks+v2+palette").expect("research compute token"),
        1867,
        "e32_palette_v2",
        &slots(),
    )
    .expect("research identity")
}

fn slots() -> SlotSet {
    SlotSet::from_slots((0..228).chain(372..720).chain(1825..1867))
}

fn pin(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}

pub(crate) fn validate(metadata: &Value) -> Result<bool, String> {
    let Some(p) = metadata.get("research_palette") else {
        return Ok(false);
    };
    let expected_map: serde_json::Map<String, Value> = (1825..1867)
        .map(|id| (id.to_string(), Value::String(format!("palette_f{id}"))))
        .collect();
    if p["schema"] != "e32-palette-projection-v1"
        || p["inherited_feature_set_id"] != BASE
        || p["palette_feature_set_id"] != PALETTE
        || p["feature_ids_sha256"] != IDS_SHA
        || p["bank_manifest_sha256"] != BANK_SHA
        || p["instrument_manifest_sha256"] != INSTRUMENT_SHA
        || p["producer_binary_sha256"] != BINARY_SHA
        || p["build_commit"] != BUILD
        || p["serving_allowed"] != false
        || p["cast"] != "Float64-to-Float32; finite primary reads"
        || p["columns"] != Value::Object(expected_map)
        || metadata["feature_set_id"] != identity().to_string()
        || metadata["formula_revision"] != 5
        || !metadata["decoder_era"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
    {
        return Err(fail(
            "identity, era, producer pin, cast or named map mismatch",
        ));
    }
    for key in [
        "table_sha256",
        "keys_sha256",
        "row_keys_sha256",
        "row_selection_sha256",
    ] {
        if !pin(&metadata[key]) {
            return Err(fail("table/key/selection byte pins required"));
        }
    }
    if !pin(&p["inherited_table_sha256"]) {
        return Err(fail("inherited table byte pin required"));
    }
    let members = p["member_sets"]
        .as_array()
        .ok_or_else(|| fail("member allowlist required"))?;
    let allowed = [
        "kadid_train",
        "kadid_select",
        "tid2013",
        "konfig_train",
        "konfig_val",
        "cid22_a25",
        "safesyn",
        "cid22_train",
        "coverage_pool",
    ];
    let unique: BTreeSet<_> = members.iter().filter_map(Value::as_str).collect();
    if members.is_empty()
        || unique.len() != members.len()
        || unique.iter().any(|m| !allowed.contains(m))
    {
        return Err(fail("unapproved/AIC member"));
    }
    let human = [
        "kadid_train",
        "kadid_select",
        "tid2013",
        "konfig_train",
        "konfig_val",
        "cid22_a25",
    ];
    let permitted = match p["role"].as_str() {
        Some("D1-fit" | "D1-development") => {
            metadata["data_role"] == "design-released-human"
                && metadata["data_role_decision_required"] == "SHIPPATH-human-production-role"
                && unique.iter().all(|m| human.contains(m))
                && metadata["human_sources"].as_array().is_some_and(|sources| {
                    let declared: BTreeSet<_> = sources.iter().filter_map(Value::as_str).collect();
                    let expected: BTreeSet<_> = unique
                        .iter()
                        .map(|m| match *m {
                            "kadid_train" | "kadid_select" => "kadid",
                            "konfig_train" | "konfig_val" => "konfig",
                            other => other,
                        })
                        .collect();
                    declared.len() == sources.len() && declared == expected
                })
        }
        Some("TRAIN-oracle-fit" | "TRAIN-oracle-development") => {
            metadata["data_role"] == "TRAIN oracle teacher"
                && unique.len() == 1
                && unique
                    .iter()
                    .all(|m| ["safesyn", "cid22_train"].contains(m))
        }
        Some("TRAIN-ordinal") => {
            metadata["data_role"] == "TRAIN ordinal KADIS source_id%10<8; no human labels"
                && unique == BTreeSet::from(["coverage_pool"])
                && p["key_domain"] == "coverage-selection-ordinal"
        }
        _ => false,
    };
    if !permitted || (p["role"] != "TRAIN-ordinal" && p["key_domain"] != "member-pair-observation")
    {
        return Err(fail("original D1/TRAIN role and key domain required"));
    }
    Ok(true)
}

pub(crate) fn reference(
    metadata: &Value,
    source: String,
) -> Result<Option<crate::feature_set::FeatureSetRef>, String> {
    if !validate(metadata)? {
        return Ok(None);
    }
    Ok(Some(crate::feature_set::FeatureSetRef {
        id: identity(),
        slots: slots(),
        layout: Some(1867),
        source,
        inferred: false,
    }))
}

pub(crate) fn verify_keys(path: &Path, metadata: &Value) -> Result<(), String> {
    let keys = path.with_extension("keys.parquet");
    if crate::train_manifest::sha256_file(&keys).map_err(|e| e.to_string())?
        != metadata["keys_sha256"].as_str().unwrap()
    {
        return Err(fail("key bytes changed"));
    }
    let builder = ParquetRecordBatchReaderBuilder::try_new(
        std::fs::File::open(&keys).map_err(|e| fail(&e.to_string()))?,
    )
    .map_err(|e| fail(&e.to_string()))?;
    let ordinal = metadata["research_palette"]["role"] == "TRAIN-ordinal";
    let allowed = if ordinal {
        vec![
            "ladder",
            "source_filename",
            "type",
            "family",
            "severity_level",
            "severity",
            "sign",
            "__index_level_0__",
        ]
    } else {
        vec![
            "pair_key",
            "source_row_id",
            "row_id",
            "ref_basename",
            "member_set",
        ]
    };
    if builder
        .schema()
        .fields()
        .iter()
        .any(|f| !allowed.contains(&f.name().as_str()))
    {
        return Err(fail("label-bearing/unapproved key columns"));
    }
    if ordinal {
        if !builder
            .schema()
            .fields()
            .iter()
            .any(|f| f.name() == "__index_level_0__")
        {
            return Err(fail("coverage original selection ordinal required"));
        }
        return Ok(());
    }
    let mut actual = BTreeSet::new();
    for batch in builder.build().map_err(|e| fail(&e.to_string()))? {
        let batch = batch.map_err(|e| fail(&e.to_string()))?;
        let members = batch
            .column_by_name("member_set")
            .ok_or_else(|| fail("string member keys required"))?;
        if members.null_count() != 0 {
            return Err(fail("null member key"));
        }
        if let Some(strings) = members.as_any().downcast_ref::<StringArray>() {
            actual.extend(strings.iter().flatten().map(str::to_owned));
        } else if let Some(strings) = members.as_any().downcast_ref::<LargeStringArray>() {
            actual.extend(strings.iter().flatten().map(str::to_owned));
        } else {
            return Err(fail("string member keys required"));
        }
    }
    let declared: BTreeSet<_> = metadata["research_palette"]["member_sets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    if actual != declared {
        return Err(fail("key population differs from admitted member set"));
    }
    Ok(())
}

pub(crate) fn columns(path: &Path, fields: &Fields) -> Result<Option<Vec<usize>>, String> {
    let metadata = crate::feature_set::table_metadata(path)?;
    if !validate(&metadata)? {
        if fields.iter().any(|f| f.name().starts_with("palette_f")) {
            return Err(fail(
                "named palette columns require explicit research projection",
            ));
        }
        return Ok(None);
    }
    let (_, first, width) = crate::parquet_loader::feature_column_run(path, fields)?;
    if width != 1853 {
        return Err(fail("inherited physical width must remain 1853"));
    }
    let mut indices: Vec<_> = (first..first + 1825).collect();
    for id in 1825..1867 {
        let name = format!("palette_f{id}");
        let found: Vec<_> = fields
            .iter()
            .enumerate()
            .filter(|(_, f)| f.name() == &name)
            .collect();
        if found.len() != 1 || found[0].1.data_type() != &DataType::Float32 {
            return Err(fail(
                "each named palette column must exist exactly once as Float32",
            ));
        }
        indices.push(found[0].0);
    }
    Ok(Some(indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Float32Array};
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    fn metadata() -> Value {
        serde_json::json!({"feature_set_id": identity().to_string(), "formula_revision":5,
            "decoder_era":"synthetic-fixture", "data_role":"TRAIN oracle teacher",
            "table_sha256":"a".repeat(64), "keys_sha256":"a".repeat(64),
            "row_keys_sha256":"a".repeat(64), "row_selection_sha256":"a".repeat(64),
            "research_palette": {"schema":"e32-palette-projection-v1",
                "inherited_feature_set_id":BASE, "palette_feature_set_id":PALETTE,
                "feature_ids_sha256":IDS_SHA, "bank_manifest_sha256":BANK_SHA,
                "instrument_manifest_sha256":INSTRUMENT_SHA, "producer_binary_sha256":BINARY_SHA,
                "build_commit":BUILD,"serving_allowed":false,
                "cast":"Float64-to-Float32; finite primary reads",
                "columns":(1825..1867).map(|id|(id.to_string(),Value::String(format!("palette_f{id}"))))
                    .collect::<serde_json::Map<_,_>>(),
                "inherited_table_sha256":"b".repeat(64), "member_sets":["safesyn"],
                "role":"TRAIN-oracle-fit", "key_domain":"member-pair-observation"}})
    }

    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("e32-rust-{}-{name}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();
            Self(dir)
        }
        fn path(&self) -> std::path::PathBuf {
            self.0.join("table.parquet")
        }
        fn declaration(&self, value: &Value) {
            std::fs::write(
                format!("{}.manifest.json", self.path().display()),
                value.to_string(),
            )
            .unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn write(path: &Path, palette: bool, nonfinite: bool) {
        let mut fields = vec![
            Field::new("human_score", DataType::Float32, false),
            Field::new("ref_basename", DataType::Utf8, false),
        ];
        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(Float32Array::from(vec![1., 2., 3., 4.])),
            Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
        ];
        for id in 0..1853 {
            fields.push(Field::new(format!("f{id}"), DataType::Float32, false));
            arrays.push(Arc::new(Float32Array::from(vec![77. + id as f32; 4])));
        }
        if palette {
            // Reverse physical order proves the reader follows names, not offsets.
            for id in (1825..1867).rev() {
                fields.push(Field::new(
                    format!("palette_f{id}"),
                    DataType::Float32,
                    false,
                ));
                arrays.push(Arc::new(Float32Array::from(vec![
                    if nonfinite && id == 1866 {
                        f32::NAN
                    } else {
                        id as f32 / 512.
                    };
                    4
                ])));
            }
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(path).unwrap(), batch.schema(), None)
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn write_keys(path: &Path, member: &str) {
        let fields = vec![
            Field::new("member_set", DataType::Utf8, false),
            Field::new("pair_key", DataType::Utf8, false),
        ];
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            vec![
                Arc::new(StringArray::from(vec![member; 4])),
                Arc::new(StringArray::from(vec![
                    "duplicate",
                    "duplicate",
                    "p3",
                    "p4",
                ])),
            ],
        )
        .unwrap();
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(path).unwrap(), batch.schema(), None)
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn contract_identity_and_era_map_refusals() {
        println!("E32_TRANSPORT_ID={}", identity());
        let original = metadata();
        assert!(validate(&original).unwrap());
        for (key, bad) in [
            ("palette_feature_set_id", Value::String("palette_v1".into())),
            ("build_commit", Value::String("0".repeat(40))),
            ("instrument_manifest_sha256", Value::String("0".repeat(64))),
            ("role", Value::String("TERMINAL".into())),
            ("member_sets", serde_json::json!(["jpeg-aic-heldout"])),
        ] {
            let mut v = original.clone();
            v["research_palette"][key] = bad;
            assert!(validate(&v).is_err(), "{key}");
        }
        let mut swapped = original.clone();
        swapped["research_palette"]["columns"]["1825"] = serde_json::json!("palette_f1826");
        assert!(validate(&swapped).is_err());
    }

    #[test]
    fn late_forbidden_role_refuses_before_any_payload_open() {
        let f = Fixture::new("late-refusal");
        let mut good = metadata();
        f.declaration(&good);
        let last = f.0.join("forbidden.parquet");
        good["research_palette"]["member_sets"] = serde_json::json!(["kadid_terminal"]);
        std::fs::write(
            format!("{}.manifest.json", last.display()),
            good.to_string(),
        )
        .unwrap();
        // Neither label-bearing table nor key file exists. A premature open
        // would produce ENOENT rather than the required population refusal.
        let err = crate::feature_set::admit_training_tables(
            &[f.path(), last],
            None,
            Some(&primary_ids()),
            Some(1867),
        )
        .unwrap_err();
        assert!(err.contains("unapproved/AIC member"), "{err}");
    }

    #[test]
    fn named_projection_preserves_auxiliary_and_all_loader_shapes() {
        let f = Fixture::new("projection");
        write(&f.path(), true, false);
        let mut m = metadata();
        write_keys(&f.path().with_extension("keys.parquet"), "safesyn");
        m["table_sha256"] =
            serde_json::json!(crate::train_manifest::sha256_file(&f.path()).unwrap());
        m["keys_sha256"] = serde_json::json!(
            crate::train_manifest::sha256_file(&f.path().with_extension("keys.parquet")).unwrap()
        );
        f.declaration(&m);
        let admitted = crate::feature_set::admit_training_tables(
            &[f.path()],
            None,
            Some(&primary_ids()),
            Some(1867),
        )
        .unwrap();
        assert_eq!(admitted["qualified_provenance"], true);
        assert_eq!(admitted["serving_allowed"], false);
        let rows =
            crate::parquet_loader::load_parquet(&f.path(), "rows", "human_score", 1.).unwrap();
        let flat =
            crate::parquet_loader::load_parquet_flat(&f.path(), "flat", "human_score", 1.).unwrap();
        let ids: Vec<_> = primary_ids().into_iter().map(|id| id as u32).collect();
        let compact = crate::parquet_loader::load_parquet_flat_f32(
            &f.path(),
            "compact",
            "human_score",
            1.,
            &ids,
            1867,
        )
        .unwrap()
        .unwrap();
        assert_eq!(rows.n_features, 1867);
        assert_eq!(rows.feature_rows[0][1825], 1825. / 512.);
        assert_ne!(rows.feature_rows[0][1825], 77. + 1825.);
        for row in 0..4 {
            for id in 0..1867 {
                assert_eq!(
                    rows.feature_rows[row][id].to_bits(),
                    flat.features_flat[row * 1867 + id].to_bits()
                );
            }
            for (j, &id) in ids.iter().enumerate() {
                assert_eq!(
                    (compact.data[row * ids.len() + j] as f64).to_bits(),
                    rows.feature_rows[row][id as usize].to_bits()
                );
            }
        }
        assert_eq!(rows.ref_ids, compact.ref_ids);
        assert_eq!(rows.human_scores, compact.human_scores);
        let reader =
            ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(f.path()).unwrap())
                .unwrap()
                .build()
                .unwrap();
        let batch = reader.into_iter().next().unwrap().unwrap();
        assert_eq!(
            batch
                .column_by_name("f1825")
                .unwrap()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(0),
            77. + 1825.
        );
        write_keys(&f.path().with_extension("keys.parquet"), "kadid_terminal");
        m["keys_sha256"] = serde_json::json!(
            crate::train_manifest::sha256_file(&f.path().with_extension("keys.parquet")).unwrap()
        );
        f.declaration(&m);
        assert!(
            crate::feature_set::admit_training_tables(
                &[f.path()],
                None,
                Some(&primary_ids()),
                Some(1867)
            )
            .unwrap_err()
            .contains("population")
        );
    }

    #[test]
    fn nonfinite_primary_is_never_zero_filled() {
        let f = Fixture::new("nonfinite");
        write(&f.path(), true, true);
        f.declaration(&metadata());
        let ids: Vec<_> = primary_ids().into_iter().map(|id| id as u32).collect();
        assert!(
            crate::parquet_loader::load_parquet_flat_f32(
                &f.path(),
                "bad",
                "human_score",
                1.,
                &ids,
                1867
            )
            .unwrap_err()
            .contains("nonfinite primary")
        );
    }

    #[test]
    fn null_primary_is_never_zero_filled() {
        let f = Fixture::new("null");
        write(&f.path(), true, false);
        let mut reader =
            ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(f.path()).unwrap())
                .unwrap()
                .build()
                .unwrap();
        let batch = reader.next().unwrap().unwrap();
        drop(reader);
        let index = batch.schema().index_of("palette_f1866").unwrap();
        let mut fields: Vec<_> = batch.schema().fields().iter().cloned().collect();
        fields[index] = Arc::new(Field::new("palette_f1866", DataType::Float32, true));
        let mut arrays = batch.columns().to_vec();
        arrays[index] = Arc::new(Float32Array::from(vec![Some(1.), None, Some(3.), Some(4.)]));
        let nullable = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
        let mut writer = ArrowWriter::try_new(
            std::fs::File::create(f.path()).unwrap(),
            nullable.schema(),
            None,
        )
        .unwrap();
        writer.write(&nullable).unwrap();
        writer.close().unwrap();
        f.declaration(&metadata());
        let ids: Vec<_> = primary_ids().into_iter().map(|id| id as u32).collect();
        let error = crate::parquet_loader::load_parquet_flat_f32(
            &f.path(),
            "bad",
            "human_score",
            1.,
            &ids,
            1867,
        )
        .unwrap_err();
        assert!(error.contains("null primary feature f1866"), "{error}");
        assert!(
            crate::parquet_loader::load_parquet(&f.path(), "bad", "human_score", 1.)
                .unwrap_err()
                .contains("null primary feature f1866")
        );
    }

    #[test]
    fn legacy_auxiliary_and_admission_metadata_unchanged() {
        let f = Fixture::new("legacy");
        write(&f.path(), false, false);
        let legacy = serde_json::json!({"feature_set_id":BASE,"formula_revision":5,
            "decoder_era":"synthetic-fixture","data_role":"unrelated-original-field","table_sha256":"irrelevant"});
        f.declaration(&legacy);
        let rows =
            crate::parquet_loader::load_parquet(&f.path(), "legacy", "human_score", 1.).unwrap();
        assert_eq!(rows.n_features, 1853);
        assert_eq!(rows.feature_rows[0][1825], 77. + 1825.);
        let admitted = crate::feature_set::admit_training_tables(
            &[f.path()],
            None,
            Some(&[13, 719]),
            Some(1853),
        )
        .unwrap();
        assert_eq!(
            admitted["tables"][0]["stored_declarations"],
            serde_json::json!({
            "feature_set_id":BASE,"formula_revision":5,"decoder_era":"synthetic-fixture"})
        );
        assert!(admitted.get("research_family").is_none());
        assert!(admitted.get("serving_allowed").is_none());
    }
}
