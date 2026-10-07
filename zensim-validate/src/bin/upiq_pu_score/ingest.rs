//! Label-free, pinned UPIQ-380 extraction through the existing HDR owner.
use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Admission {
    schema: String,
    role: String,
    tier: String,
    authority: String,
    input_contract: String,
    formula_revision: u8,
    image_root: PathBuf,
    requested_ids: Vec<usize>,
    rows: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    condition_id: String,
    dataset: String,
    content: usize,
    distortion: usize,
    level: usize,
    reference_rel: String,
    distorted_rel: String,
    reference_sha256: String,
    distorted_sha256: String,
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn validate(a: &Admission) -> Result<(), String> {
    if a.schema != "upiq380-extraction-admission-v1"
        || a.role != "train"
        || a.tier != "T2"
        || a.authority != "D3-2026-10-07"
        || a.input_contract != "upiq-exr-bt709-nits-v1"
        || a.formula_revision != 5
        || !a.image_root.is_absolute()
        || a.requested_ids.len() != 420
        || a.requested_ids.windows(2).any(|w| w[0] >= w[1])
        || a.requested_ids
            .iter()
            .any(|&x| x >= zensim::research::full_width())
        || a.rows.len() != 380
    {
        return Err("not the registered TRAIN-only Rev5 UPIQ-380 contract".into());
    }
    let mut seen = BTreeSet::new();
    let mut reference_hashes = HashMap::new();
    for row in &a.rows {
        let (prefix, nc, nd, nl) = match row.dataset.as_str() {
            "narwaria" => ('n', 10, 2, 7),
            "korshunov" => ('k', 20, 3, 4),
            _ => return Err("non-HDR UPIQ source refused".into()),
        };
        if !(1..=nc).contains(&row.content)
            || !(1..=nd).contains(&row.distortion)
            || !(1..=nl).contains(&row.level)
            || row.condition_id
                != format!(
                    "{prefix}-i{:02}-{prefix}-{:02}-{}",
                    row.content, row.distortion, row.level
                )
            || row.reference_rel
                != format!("{}/{:02}/i{:02}.exr", row.dataset, row.content, row.content)
            || row.distorted_rel
                != format!(
                    "{}/{:02}/i{:02}_{:02}_{}.exr",
                    row.dataset, row.content, row.content, row.distortion, row.level
                )
            || !seen.insert(row.condition_id.clone())
        {
            return Err("UPIQ-380 member/key/path mismatch".into());
        }
        for hash in [&row.reference_sha256, &row.distorted_sha256] {
            if hash.len() != 64
                || !hash
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            {
                return Err("invalid original-byte SHA256".into());
            }
        }
        if let Some(previous) = reference_hashes.insert(&row.reference_rel, &row.reference_sha256)
            && previous != &row.reference_sha256
        {
            return Err("reference hash is inconsistent".into());
        }
    }
    Ok(())
}

fn load(root: &Path, rel: &str, hash: &str, reads: &mut Vec<String>) -> Result<super::Rgb, String> {
    let path = root.join(rel);
    let mut component = root.to_owned();
    for part in Path::new(rel).components() {
        component.push(part);
        if std::fs::symlink_metadata(&component)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
        {
            return Err("symlink below admitted image root refused".into());
        }
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    reads.push(path.display().to_string());
    if digest(&bytes) != hash {
        return Err(format!("original bytes changed: {path:?}"));
    }
    super::decode_exr_rgb(&path, &bytes)
}

pub(super) fn run(args: &[String]) -> Result<(), String> {
    let path = super::arg(args, "--training-allowlist").ok_or("missing allowlist")?;
    let pin = super::arg(args, "--allowlist-sha256").ok_or("missing allowlist pin")?;
    let out = super::arg(args, "--out").ok_or("missing output")?;
    if Path::new(&out).exists() || Path::new(&format!("{out}.manifest.json")).exists() {
        return Err("fresh extraction output required".into());
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    if digest(&bytes) != pin {
        return Err("allowlist pin mismatch before image reads".into());
    }
    let a: Admission = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    validate(&a)?; // Complete metadata admission precedes every pixel payload.
    if zensim::feature_v2::active_formula_revision() != zensim::feature_v2::FormulaRevision::Rev5 {
        return Err("UPIQ extraction requires process FormulaRevision::Rev5".into());
    }
    let req = zensim::research::Request::for_slots(
        zensim::feature_set_id::SlotSet::from_slots(a.requested_ids.iter().copied()),
        zensim::research::full_width(),
    )
    .with_era_label("e31-upiq380-native-hdr-rev5")
    .with_parallel(false);
    let mut refs = HashMap::new();
    let mut reads = vec![path.clone()];
    let mut manifest = None;
    let mut output = String::from("condition_id");
    for id in 0..zensim::research::full_width() {
        output.push_str(&format!("\tf{id}"));
    }
    output.push('\n');
    for (i, row) in a.rows.iter().enumerate() {
        if !refs.contains_key(&row.reference_rel) {
            refs.insert(
                row.reference_rel.clone(),
                load(
                    &a.image_root,
                    &row.reference_rel,
                    &row.reference_sha256,
                    &mut reads,
                )?,
            );
        }
        let reference = &refs[&row.reference_rel];
        let distorted = load(
            &a.image_root,
            &row.distorted_rel,
            &row.distorted_sha256,
            &mut reads,
        )?;
        if (reference.w, reference.h) != (distorted.w, distorted.h) {
            return Err("reference/distorted dimensions differ".into());
        }
        let result = zensim::research::extract_hdr(
            &req,
            reference,
            &distorted,
            zensim::feature_v2::HdrEncoding::Linear,
        )
        .map_err(|e| e.to_string())?;
        let metadata = result.manifest_json();
        if let Some(previous) = &manifest {
            if previous != &metadata {
                return Err("research metadata changed between rows".into());
            }
        } else {
            manifest = Some(metadata);
        }
        output.push_str(&row.condition_id);
        for (id, value) in result.values().iter().enumerate() {
            let requested = a.requested_ids.binary_search(&id).is_ok();
            if requested && !value.is_finite() {
                return Err(format!("nonfinite f{id}"));
            }
            output.push_str(&format!(
                "\t{:?}",
                if requested { *value } else { f64::NAN }
            ));
        }
        output.push('\n');
        if (i + 1) % 10 == 0 {
            eprintln!("UPIQ-380 Rev5 extraction: {}/380", i + 1);
        }
    }
    let metadata = serde_json::json!({"schema":"upiq380-extraction-result-v1", "rows":a.rows.len(),
        "formula_revision":5,"input_contract":a.input_contract,"requested_ids":a.requested_ids,
        "allowlist_sha256":pin,"build_commit":zensim::research::BUILD_COMMIT,
        "research":serde_json::from_str::<serde_json::Value>(&manifest.ok_or("empty extraction")?).map_err(|e| e.to_string())?,
        "files_read":reads,"reference_count":refs.len(),"decoder":"zenexr; native BT.709 f32 absolute nits; finite opaque samples"});
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out)
        .map_err(|e| e.to_string())?;
    file.write_all(output.as_bytes())
        .map_err(|e| e.to_string())?;
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(format!(
            "{}.manifest.json",
            super::arg(args, "--out").unwrap()
        ))
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(file, &metadata).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        let mut rows = Vec::new();
        for (dataset, prefix, nc, nd, nl) in
            [("narwaria", 'n', 10, 2, 7), ("korshunov", 'k', 20, 3, 4)]
        {
            for c in 1..=nc {
                for d in 1..=nd {
                    for l in 1..=nl {
                        rows.push(serde_json::json!({"dataset":dataset,"content":c,"distortion":d,"level":l,
                            "condition_id":format!("{prefix}-i{c:02}-{prefix}-{d:02}-{l}"),
                            "reference_rel":format!("{dataset}/{c:02}/i{c:02}.exr"),
                            "distorted_rel":format!("{dataset}/{c:02}/i{c:02}_{d:02}_{l}.exr"),
                            "reference_sha256":"1".repeat(64),"distorted_sha256":"2".repeat(64)}));
                    }
                }
            }
        }
        serde_json::json!({"schema":"upiq380-extraction-admission-v1","role":"train","tier":"T2",
            "authority":"D3-2026-10-07","input_contract":"upiq-exr-bt709-nits-v1","formula_revision":5,
            "image_root":"/nonexistent-upiq-input-tripwire","requested_ids":(0..420).collect::<Vec<_>>(),"rows":rows})
    }

    fn checked(value: serde_json::Value) -> Result<(), String> {
        validate(&serde_json::from_value(value).unwrap())
    }

    #[test]
    fn complete_metadata_admits_without_opening_any_images() {
        assert!(checked(fixture()).is_ok());
    }

    #[test]
    fn late_foreign_member_and_traversal_are_refused() {
        for (field, value) in [
            ("dataset", "live"),
            ("distorted_rel", "../live/image.png"),
            ("reference_rel", "tid2013/01/i01.png"),
            ("condition_id", "l-i20-l-03-4"),
        ] {
            let mut v = fixture();
            v["rows"][379][field] = value.into();
            assert!(checked(v).is_err(), "{field}");
        }
    }

    #[test]
    fn duplicate_and_wrong_original_hash_are_refused() {
        let mut v = fixture();
        v["rows"][379] = v["rows"][0].clone();
        assert!(checked(v).is_err());
        let mut v = fixture();
        v["rows"][379]["distorted_sha256"] = "not-a-hash".into();
        assert!(checked(v).is_err());
    }

    #[test]
    fn foreign_role_and_revision_are_refused() {
        for (field, value) in [
            ("role", serde_json::json!("val")),
            ("tier", serde_json::json!("T0")),
            ("formula_revision", serde_json::json!(4)),
            ("authority", serde_json::json!("other")),
        ] {
            let mut v = fixture();
            v[field] = value;
            assert!(checked(v).is_err());
        }
    }
}
