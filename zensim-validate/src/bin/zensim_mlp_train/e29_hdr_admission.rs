//! E29's native HDR research population gate. No target/features before keys.
use arrow::array::{
    Array, BooleanArray, Int32Array, Int64Array, LargeStringArray, StringArray, UInt32Array,
    UInt64Array,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

fn refusal(message: &str) -> String {
    format!("E29 HDR admission: {message}")
}

fn key_string(array: &dyn Array, index: usize) -> Result<String, String> {
    if array.is_null(index) {
        return Err(refusal("null label-free key"));
    }
    macro_rules! value {
        ($kind:ty) => {
            if let Some(values) = array.as_any().downcast_ref::<$kind>() {
                return Ok(values.value(index).to_string());
            }
        };
    }
    value!(StringArray);
    value!(LargeStringArray);
    value!(Int64Array);
    value!(UInt64Array);
    value!(Int32Array);
    value!(UInt32Array);
    Err(refusal("unsupported label-free key type"))
}

pub(super) fn admit(
    path: &Path,
    selected: &[usize],
    pair_specs: &[String],
) -> Result<Value, String> {
    let sidecar = std::fs::read(format!("{}.manifest.json", path.display()))
        .map_err(|e| refusal(&format!("manifest: {e}")))?;
    let d: Value =
        serde_json::from_slice(&sidecar).map_err(|e| refusal(&format!("metadata: {e}")))?;
    let recipe: Value = serde_json::from_str(include_str!(
        "../../../../benchmarks/costset2_2026-10-03.candidate_ids.json"
    ))
    .map_err(|e| refusal(&format!("registered IDs: {e}")))?;
    let ids = serde_json::json!(selected);
    let arm = d["arm"].as_str().unwrap_or_default();
    let transform = match arm {
        "hb4" => "pooled-midrank-Borda-[0,1]",
        "hc4" => "score=10*q_jod;no-clipping",
        _ => return Err(refusal("unknown arm")),
    };
    let sources = [
        (
            "teacher_sha256",
            "deb70e775b043a578c77e0c3ff27960ebfa936e9d901497f9d74389e6ce9fbce",
        ),
        (
            "source_table_sha256",
            "1d66d371b85733d7a4fde4964688284e7ac01d69707c62e0b7c4888af0a9b0c3",
        ),
        (
            "source_keys_sha256",
            "b55a4ec95dac7f4c060f910c7a6dabf2a1d7c6245efc81f6a36e90ed33d09553",
        ),
        (
            "source_manifest_sha256",
            "66154e1760fb9724a6021897b39e9950a811eeb1ad846daf2677a63a34d93b0e",
        ),
    ];
    if d["study"] != "E29"
        || d["role"] != "train"
        || d["rows"] != 7390
        || d["population"] != "agree-only"
        || d["formula_revision"] != 5
        || d["requested_ids"] != ids
        || ids != recipe["candidates"]["by_v2fy"]
        || !d["source_bank_feature_set_id"].is_null()
        || d["target_transform"] != transform
        || !d["build_commit"]
            .as_str()
            .is_some_and(|s| !s.trim().is_empty())
        || sources.iter().any(|(field, pin)| d[*field] != *pin)
    {
        return Err(refusal("unapproved TRAIN population/source/IDs/transform"));
    }
    for field in ["keys_sha256", "row_keys_sha256", "table_sha256"] {
        if !d[field].as_str().is_some_and(|s| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }) {
            return Err(refusal("missing/invalid content pin"));
        }
    }
    // All declaration checks above precede the only permitted key-file open.
    let key_path = path.with_extension("keys.parquet");
    let mut key_file =
        std::fs::File::open(&key_path).map_err(|e| refusal(&format!("keys: {e}")))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(
        key_file.try_clone().map_err(|e| refusal(&e.to_string()))?,
    )
    .map_err(|e| refusal(&format!("key metadata: {e}")))?;
    let columns = ["row_id", "role", "agree", "ref_basename"];
    if builder
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect::<Vec<_>>()
        != columns
    {
        return Err(refusal(
            "keys must contain exactly the label-free identity schema",
        ));
    }
    let reader = builder.build().map_err(|e| refusal(&e.to_string()))?;
    let mut rows = Vec::with_capacity(7390);
    let mut seen = HashSet::new();
    for batch in reader {
        let batch = batch.map_err(|e| refusal(&e.to_string()))?;
        let agree = batch
            .column(2)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| refusal("agreement key must be boolean"))?;
        for i in 0..batch.num_rows() {
            let id = key_string(batch.column(0).as_ref(), i)?;
            let role = key_string(batch.column(1).as_ref(), i)?;
            if role != "train"
                || agree.is_null(i)
                || !agree.value(i)
                || !seen.insert(id.clone())
                || rows.len() >= 7390
            {
                return Err(refusal(
                    "label-free keys are not unique TRAIN agreement rows",
                ));
            }
            rows.push([
                id,
                role,
                "True".to_owned(),
                key_string(batch.column(3).as_ref(), i)?,
            ]);
        }
    }
    if rows.len() != 7390 {
        return Err(refusal("label-free key population must have 7390 rows"));
    }
    // v2_teacher.row_keys_sha: compact UTF-8 JSON, Python str(True) = "True".
    // Serialize fields explicitly in columns/rows order, independent of map order.
    let mut hash = Sha256::new();
    hash.update(b"{\"columns\":");
    hash.update(serde_json::to_vec(&columns).map_err(|e| refusal(&e.to_string()))?);
    hash.update(b",\"rows\":");
    hash.update(serde_json::to_vec(&rows).map_err(|e| refusal(&e.to_string()))?);
    hash.update(b"}");
    if d["row_keys_sha256"]
        != hash
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    {
        return Err(refusal("ordered label-free key identity changed"));
    }
    // Hash the admitted key object using the original handle, not another name.
    key_file
        .seek(SeekFrom::Start(0))
        .map_err(|e| refusal(&e.to_string()))?;
    let mut hash = Sha256::new();
    let mut buf = [0_u8; 65536];
    loop {
        let n = key_file
            .read(&mut buf)
            .map_err(|e| refusal(&e.to_string()))?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    if d["keys_sha256"]
        != hash
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    {
        return Err(refusal("key file pin changed"));
    }
    // Pair-list path/content and table hashes are deferred until key admission.
    if arm == "hc4" {
        let name = d["pair_list"]
            .as_str()
            .ok_or_else(|| refusal("pair-list binding missing"))?;
        if Path::new(name).components().count() != 1 || name == "." || name == ".." {
            return Err(refusal("pair-list must be a same-directory filename"));
        }
        let pairs = path
            .parent()
            .ok_or_else(|| refusal("table directory missing"))?
            .join(name);
        let bound = pair_specs.first().and_then(|s| s.split_once(':'));
        if pair_specs.len() != 1 || bound != pairs.to_str().map(|p| ("hdr", p)) {
            return Err(refusal("pair-list CLI does not match declaration"));
        }
        if d["pair_list_sha256"]
            != zensim_validate::train_manifest::sha256_file(&pairs)
                .map_err(|e| refusal(&e.to_string()))?
        {
            return Err(refusal("pair-list pin changed"));
        }
    } else if !pair_specs.is_empty() {
        return Err(refusal("hb4 cannot use a pair list"));
    }
    if d["table_sha256"]
        != zensim_validate::train_manifest::sha256_file(path)
            .map_err(|e| refusal(&e.to_string()))?
    {
        return Err(refusal("HDR table pin changed"));
    }
    Ok(d)
}
