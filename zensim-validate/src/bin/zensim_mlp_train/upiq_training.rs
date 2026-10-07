//! Registered E31 ingress; the ordinary table admission path stays separate.
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
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
    {
        return Err("E31 requires the owner's bound legacy-label disposition".into());
    }
    Ok(())
}

pub(super) fn admit(
    paths: &[PathBuf],
    decision: &Path,
    selected: Option<&[usize]>,
    width: usize,
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
            if native.is_some() || hash(Path::new(&sidecar))? != MANIFEST_SHA {
                return Err("E31 requires exactly the pinned UPIQ fit manifest".into());
            }
            let ids: Vec<usize> = serde_json::from_value(declaration["requested_ids"].clone())
                .map_err(|e| e.to_string())?;
            if selected != Some(ids.as_slice()) || width < 1825 {
                return Err("E31 requires the exact registered 420-slot projection".into());
            }
            native = Some((path, declaration));
        } else {
            // Resolve metadata here too, before the ordinary owner reads footers.
            feature_set::table_feature_set_ref(Path::new(path))?
                .ok_or("E31 ordinary leg has no registered feature identity")?;
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
}
