//! Registered E31 ingress; the ordinary table admission path stays separate.
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use zensim_validate::mlp_train::GroupLossMode;
use zensim_validate::{feature_set, train_manifest};

pub(super) const MANIFEST_SHA: &str =
    "2da346bb17e08a4a63aae9ed89b159b36edd331199a9dd1c42b4a05b53ac939e";
pub(super) const TABLE_SHA: &str =
    "7f09debedc591e7dd3494846ada9fe0c93b01779918b8358e2cf8053f5f1a6c4";
pub(super) const KEYS_SHA: &str =
    "c47ca1c12d1e8e206464938884ff9d05baa8274785826060604d16caf0bff49e";

fn hash(path: &Path) -> Result<String, String> {
    train_manifest::sha256_file(path).map_err(|e| e.to_string())
}

fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

pub(super) fn disposition(value: &Value) -> Result<(), String> {
    if value["schema"] != "e31-upiq-label-disposition-v1"
        || value["state"] != "approved"
        || value["decision_id"] != "E31-legacy-HDR-label-producer-gap"
        || value["allowed_use"] != "registered-E31-research-training"
        || value["manifest_sha256"] != MANIFEST_SHA
        || value["legacy_label_sha256"]
            != "e0b23f539d46845f6cc4a65591474bafbf7eb4faf02da7175fb342caa5df80ac"
        || value["accept_unresolved_producer"] != true
        || value["decided_by"].as_str().is_none_or(str::is_empty)
    {
        return Err("E31 requires the owner's bound legacy-label disposition".into());
    }
    Ok(())
}

pub(super) fn native_group<'a>(
    groups: &'a [(String, PathBuf, f64, f64, bool, GroupLossMode)],
    target: &str,
    scale: f64,
) -> Result<&'a Path, String> {
    let native: Vec<_> = groups.iter().filter(|g| g.0 == "upiq380").collect();
    if native.len() != 1 {
        return Err("E31 requires exactly one named upiq380 fit group".into());
    }
    let g = native[0];
    if g.4
        || g.5 != GroupLossMode::Rank
        || g.3 != 0.0
        || g.2.to_bits() != 4.34410740924913_f64.to_bits()
        || target != "human_score"
        || scale != 1.0
    {
        return Err("E31 requires the registered pooled rank-only nominal-4 fit group".into());
    }
    Ok(&g.1)
}

pub(super) fn admit(
    paths: &[PathBuf],
    decision: &Path,
    selected: Option<&[usize]>,
    width: usize,
    native_path: &Path,
) -> Result<Value, String> {
    let approval = read_json(decision)?;
    disposition(&approval)?;
    let mut native = None;
    let mut ordinary = Vec::new();
    // Inspect every declaration before hashing any feature payload.
    for path in paths {
        let sidecar = format!("{}.manifest.json", path.display());
        let declaration = read_json(Path::new(&sidecar))?;
        if declaration["source"] == "UPIQ-380" {
            if path != native_path || native.is_some() || hash(Path::new(&sidecar))? != MANIFEST_SHA
            {
                return Err("E31 requires exactly the pinned UPIQ fit manifest".into());
            }
            let ids: Vec<usize> = serde_json::from_value(declaration["requested_ids"].clone())
                .map_err(|e| e.to_string())?;
            if selected != Some(ids.as_slice()) || width != 1853 {
                return Err("E31 requires the exact registered 420-slot projection".into());
            }
            native = Some((path, declaration));
        } else {
            // Resolve metadata here too, before the ordinary owner reads footers.
            let identity = feature_set::table_feature_set_ref(Path::new(path))?
                .ok_or("E31 ordinary leg has no registered feature identity")?;
            if identity.id.to_string() != "basic+peaks+v2@w1825/rev5_localwin#36c3f3af"
                || declaration["formula_revision"] != 5
                || declaration["human_sources"]
                    .as_array()
                    .is_some_and(|sources| {
                        sources.iter().any(|source| {
                            !matches!(
                                source.as_str(),
                                Some("kadid" | "tid2013" | "konfig" | "cid22_a25")
                            )
                        })
                    })
                || declaration["bank_manifest_sha256"]
                    .as_object()
                    .is_some_and(|banks| {
                        banks
                            .keys()
                            .any(|bank| bank.to_ascii_lowercase().contains("aic"))
                    })
            {
                return Err(
                    "E31 ordinary leg must retain the registered Rev5/D1 population".into(),
                );
            }
            ordinary.push(path.clone());
        }
    }
    let (path, declaration) = native.ok_or("E31 disposition supplied without UPIQ fit")?;
    if ordinary.is_empty() {
        return Err("E31 requires the inherited SDR groups".into());
    }
    let mut receipt = feature_set::admit_training_tables(&ordinary, None, selected, Some(width))?;
    let keys = Path::new(path).with_extension("keys.parquet");
    if hash(&keys)? != KEYS_SHA || hash(Path::new(path))? != TABLE_SHA {
        return Err("E31 UPIQ fit payload/key pin changed".into());
    }
    receipt["upiq380"] = json!({"path": path, "manifest_sha256": MANIFEST_SHA,
        "table_sha256": TABLE_SHA, "keys_sha256": KEYS_SHA,
        "input_contract": declaration["input_contract"],
        "extractor_build_commit": declaration["build_commit"],
        "extractor_binary_sha256": declaration["binary_sha256"],
        "selected_ids": selected, "native_width": 1825, "logical_width": width,
        "label_source": declaration["label_source"],
        "label_disposition": approval,
        "label_disposition_sha256": hash(decision)?});
    // Explicit research permission does not repair missing producer provenance.
    receipt["qualified_provenance"] = json!(false);
    Ok(receipt)
}

pub(super) fn pad_native(rows: &mut Vec<f64>, native: usize, logical: usize) -> Result<(), String> {
    if native != 1825 || logical < native || !rows.len().is_multiple_of(native) {
        return Err("E31 native feature width/row shape mismatch".into());
    }
    if logical != native {
        let mut padded = Vec::with_capacity(rows.len() / native * logical);
        for row in rows.chunks_exact(native) {
            padded.extend_from_slice(row);
            padded.resize(padded.len() + logical - native, f64::NAN);
        }
        *rows = padded;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float64_parquet_falls_back_to_native_loader_without_clipping_targets() {
        use arrow::array::{ArrayRef, Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let path = std::env::temp_dir().join(format!("e31-native-{}.parquet", std::process::id()));
        let mut fields = vec![
            Field::new("human_score", DataType::Float64, false),
            Field::new("ref_basename", DataType::Utf8, false),
        ];
        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(vec![-120.0, 155.0])),
            Arc::new(StringArray::from(vec!["r1", "r2"])),
        ];
        let exact = f64::from_bits(0x3ff0_0000_0000_0001);
        for i in 0..1825 {
            fields.push(Field::new(format!("f{i}"), DataType::Float64, false));
            arrays.push(Arc::new(Float64Array::from(vec![
                if i == 13 { exact } else { f64::NAN },
                if i == 13 { 3.0 } else { f64::NAN },
            ])));
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create_new(&path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let mut loaded = super::super::load_group_dispatch(
            &path,
            "upiq380",
            "human_score",
            1.0,
            Some((&[13], 1853)),
        )
        .unwrap();
        assert!(loaded.compact.is_none());
        assert_eq!(loaded.n_features, 1825);
        assert_eq!(loaded.human_scores, [-120.0, 155.0]);
        assert_eq!(loaded.feature_rows[13].to_bits(), exact.to_bits());
        pad_native(&mut loaded.feature_rows, loaded.n_features, 1853).unwrap();
        assert_eq!(loaded.feature_rows[13].to_bits(), exact.to_bits());
        assert_eq!(loaded.feature_rows[1853 + 13], 3.0);
        assert!(loaded.feature_rows[1825..1853].iter().all(|v| v.is_nan()));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn native_transport_preserves_f64_bits_and_absent_slots() {
        let mut rows = vec![f64::NAN; 2 * 1825];
        rows[13] = f64::from_bits(0x3ff0_0000_0000_0001);
        rows[1825 + 719] = -3.25;
        let before = rows.clone();
        pad_native(&mut rows, 1825, 1853).unwrap();
        for r in 0..2 {
            for c in 0..1825 {
                assert_eq!(rows[r * 1853 + c].to_bits(), before[r * 1825 + c].to_bits());
            }
            assert!(
                rows[r * 1853 + 1825..(r + 1) * 1853]
                    .iter()
                    .all(|v| v.is_nan())
            );
        }
        assert!(pad_native(&mut vec![0.0; 1825], 1825, 720).is_err());
        assert!(pad_native(&mut vec![0.0; 1824], 1825, 1853).is_err());
    }

    #[test]
    fn population_authority_does_not_dispose_of_the_label_gap() {
        assert!(disposition(&json!({"state":"approved", "authority":"D3-2026-10-07"})).is_err());
    }

    #[test]
    fn rank_recipe_is_checked_before_admission_or_payload_reads() {
        let group = (
            "upiq380".into(),
            PathBuf::from("unread.parquet"),
            4.34410740924913,
            0.0,
            false,
            GroupLossMode::Rank,
        );
        assert_eq!(
            native_group(std::slice::from_ref(&group), "human_score", 1.0).unwrap(),
            Path::new("unread.parquet")
        );
        for bad in [
            (
                "upiq380".into(),
                group.1.clone(),
                group.2,
                1.0,
                false,
                GroupLossMode::Rank,
            ),
            (
                "upiq380".into(),
                group.1.clone(),
                group.2,
                0.0,
                true,
                GroupLossMode::Rank,
            ),
            (
                "upiq380".into(),
                group.1.clone(),
                group.2,
                0.0,
                false,
                GroupLossMode::Both,
            ),
            (
                "upiq380".into(),
                group.1.clone(),
                4.0,
                0.0,
                false,
                GroupLossMode::Rank,
            ),
        ] {
            assert!(native_group(&[bad], "human_score", 1.0).is_err());
        }
        assert!(native_group(&[group.clone(), group], "human_score", 1.0).is_err());
    }
}
