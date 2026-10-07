//! Label-free ordered observation admission, shared by library and trainer.
use arrow::array::*;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub(crate) struct Keys {
    pub(crate) columns: Vec<String>,
    pub(crate) types: Vec<arrow::datatypes::DataType>,
    pub(crate) rows: Vec<Vec<String>>,
    pub(crate) digest: String,
    pub(crate) file_digest: String,
}

fn python_float(value: f64) -> Result<String, String> {
    if !value.is_finite() {
        return Err("nonfinite identity value".into());
    }
    let text = serde_json::to_string(&value).map_err(|e| e.to_string())?;
    if let Some((mantissa, exponent)) = text.split_once('e') {
        let exponent: i32 = exponent.parse().map_err(|_| "invalid identity exponent")?;
        return Ok(format!(
            "{}e{exponent:+03}",
            mantissa.trim_end_matches(".0")
        ));
    }
    // Python switches to scientific notation below 1e-4, JSON below 1e-5.
    let (sign, positive) = text
        .strip_prefix('-')
        .map_or(("", text.as_str()), |s| ("-", s));
    if positive.starts_with("0.0000") && value != 0.0 {
        let digits = &positive[2..];
        let start = digits
            .find(|c| c != '0')
            .ok_or("invalid identity decimal")?;
        let significant = &digits[start..];
        let mantissa = if significant.len() == 1 {
            significant.to_owned()
        } else {
            format!("{}.{}", &significant[..1], &significant[1..])
        };
        return Ok(format!("{sign}{mantissa}e-{:02}", start + 1));
    }
    Ok(text)
}

fn cell(array: &dyn Array, i: usize) -> Result<String, String> {
    if array.is_null(i) {
        return Err("null label-free identity".into());
    }
    macro_rules! number {
        ($t:ty) => {
            if let Some(a) = array.as_any().downcast_ref::<$t>() {
                return Ok(a.value(i).to_string());
            }
        };
    }
    number!(StringArray);
    number!(LargeStringArray);
    number!(Int8Array);
    number!(UInt8Array);
    number!(Int16Array);
    number!(UInt16Array);
    number!(Int32Array);
    number!(UInt32Array);
    number!(Int64Array);
    number!(UInt64Array);
    if let Some(a) = array.as_any().downcast_ref::<BooleanArray>() {
        return Ok(if a.value(i) { "True" } else { "False" }.into());
    }
    if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
        return python_float(a.value(i));
    }
    if let Some(a) = array.as_any().downcast_ref::<Float32Array>() {
        return python_float(f64::from(a.value(i)));
    }
    Err("unsupported identity type".into())
}

pub(crate) fn read(path: &Path, required: &[&str], allowed: &[&str]) -> Result<Keys, String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let file_digest = hash.finalize().iter().map(|v| format!("{v:02x}")).collect();
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    let types = builder
        .schema()
        .fields()
        .iter()
        .map(|f| f.data_type().clone())
        .collect();
    let columns: Vec<String> = builder
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    if required.iter().any(|n| !columns.iter().any(|c| c == n))
        || columns.iter().any(|c| !allowed.contains(&c.as_str()))
    {
        return Err("missing observation identity or label-bearing key schema".into());
    }
    let mut rows = Vec::new();
    for batch in builder.build().map_err(|e| e.to_string())? {
        let batch = batch.map_err(|e| e.to_string())?;
        for i in 0..batch.num_rows() {
            let row = batch
                .columns()
                .iter()
                .map(|a| cell(a.as_ref(), i))
                .collect::<Result<Vec<_>, _>>()?;
            if row.iter().any(String::is_empty) {
                return Err("empty observation identity".into());
            }
            rows.push(row);
        }
    }
    if rows.is_empty() {
        return Err("empty observation population".into());
    }
    let mut hash = Sha256::new();
    hash.update(b"{\"columns\":");
    hash.update(serde_json::to_vec(&columns).map_err(|e| e.to_string())?);
    hash.update(b",\"rows\":[");
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            hash.update(b",");
        }
        hash.update(serde_json::to_vec(row).map_err(|e| e.to_string())?);
    }
    hash.update(b"]}");
    Ok(Keys {
        columns,
        types,
        rows,
        digest: hash.finalize().iter().map(|v| format!("{v:02x}")).collect(),
        file_digest,
    })
}

impl Keys {
    pub(crate) fn column(&self, name: &str) -> Result<usize, String> {
        self.columns
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("missing {name} identity"))
    }
}

pub(crate) fn expected_rows(metadata: &Value) -> Result<usize, String> {
    for name in ["rows_kept", "rows", "observations"] {
        if let Some(n) = metadata[name].as_u64() {
            return usize::try_from(n).map_err(|e| e.to_string());
        }
    }
    if let Some(selections) = metadata["row_selection"].as_array() {
        return selections.iter().try_fold(0_usize, |n, s| {
            n.checked_add(
                usize::try_from(s["rows"].as_u64().ok_or("missing selection count")?)
                    .map_err(|e| e.to_string())?,
            )
            .ok_or_else(|| "observation count overflow".into())
        });
    }
    Err("declared observation count required".into())
}

pub(crate) fn verify(path: &Path, metadata: &Value, count: usize) -> Result<Keys, String> {
    let ordinal = metadata["data_role"] == "TRAIN ordinal KADIS source_id%10<8; no human labels";
    let required = if ordinal {
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
        vec!["pair_key", "ref_basename", "member_set"]
    };
    let allowed = if ordinal {
        required.clone()
    } else {
        vec![
            "pair_key",
            "source_row_id",
            "row_id",
            "ref_basename",
            "member_set",
        ]
    };
    let keys = read(&path.with_extension("keys.parquet"), &required, &allowed)?;
    if keys.rows.len() != count
        || metadata["keys_sha256"] != keys.file_digest
        || metadata["row_keys_sha256"] != keys.digest
    {
        return Err("ordered observation count/order/file pin changed".into());
    }
    use arrow::datatypes::DataType;
    let strings: &[&str] = if ordinal {
        &["ladder", "source_filename", "type", "family"]
    } else {
        &["pair_key", "ref_basename", "member_set"]
    };
    for name in strings {
        if !matches!(
            keys.types[keys.column(name)?],
            DataType::Utf8 | DataType::LargeUtf8
        ) {
            return Err("string observation identity required".into());
        }
    }
    if ordinal {
        for name in ["severity", "sign"] {
            if !matches!(
                keys.types[keys.column(name)?],
                DataType::Float32 | DataType::Float64
            ) {
                return Err("numeric ordinal identity required".into());
            }
        }
        for name in ["__index_level_0__", "severity_level"] {
            if !matches!(
                keys.types[keys.column(name)?],
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
            ) {
                return Err("integer ordinal identity required".into());
            }
        }
        let ix = keys.column("__index_level_0__")?;
        let mut seen = std::collections::HashSet::new();
        for row in &keys.rows {
            let n: u64 = row[ix]
                .parse()
                .map_err(|_| "original selection ordinal must be an unsigned integer")?;
            if !seen.insert(n) {
                return Err("duplicate original selection ordinal".into());
            }
        }
    } else {
        let ix = keys
            .columns
            .iter()
            .position(|n| n == "source_row_id" || n == "row_id")
            .ok_or("original row/observation identity required")?;
        if !matches!(
            keys.types[ix],
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
        ) {
            return Err("integer original row identity required".into());
        }
        for row in &keys.rows {
            row[ix]
                .parse::<u64>()
                .map_err(|_| "invalid original row identity")?;
        }
        let member = keys.column("member_set")?;
        let actual: std::collections::BTreeSet<_> =
            keys.rows.iter().map(|r| r[member].as_str()).collect();
        let allowed = [
            "kadid_train",
            "kadid_select",
            "tid2013",
            "konfig_train",
            "konfig_val",
            "cid22_a25",
            "safesyn",
            "cid22_train",
        ];
        if actual.iter().any(|m| !allowed.contains(m)) {
            return Err("unapproved/AIC observation population member".into());
        }
        if let Some(members) = metadata["research_palette"]["member_sets"].as_array() {
            let expected: std::collections::BTreeSet<_> =
                members.iter().filter_map(Value::as_str).collect();
            if actual != expected {
                return Err("observation members differ from declaration".into());
            }
        }
    }
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn python_identity_float_spelling() {
        for (value, expected) in [
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (0.00001, "1e-05"),
            (1e-6, "1e-06"),
            (1e16, "1e+16"),
            (0.5, "0.5"),
        ] {
            assert_eq!(python_float(value).unwrap(), expected);
        }
        assert!(python_float(f64::NAN).is_err());
    }
}
