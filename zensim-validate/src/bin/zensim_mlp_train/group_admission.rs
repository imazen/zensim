//! One metadata/key phase for all native trainer groups. No table reads here.
use super::{Args, e29_hdr_admission, training_keys, upiq_training};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use zensim_validate::mlp_train::GroupLossMode;
use zensim_validate::{feature_set, train_manifest};

type Group = (String, PathBuf, f64, f64, bool, GroupLossMode);
const SOURCES: [&str; 4] = ["kadid", "tid2013", "konfig", "cid22_a25"];
const BASE: &str = "basic+peaks+v2@w1825/rev5_localwin#36c3f3af";

fn safe(path: &Path) -> Result<(), String> {
    for p in [
        path.to_path_buf(),
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
    ] {
        if p.components().any(|c| {
            let s = c.as_os_str().to_string_lossy().to_ascii_lowercase();
            s.contains("_sealed")
                || s.contains("holdout")
                || s.starts_with("labels__")
                || s.contains("terminal")
        }) {
            return Err("protected training input ancestry".into());
        }
    }
    Ok(())
}
fn json(path: &Path) -> Result<Value, String> {
    safe(path)?;
    serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| e.to_string())
}
fn hash(path: &Path) -> Result<String, String> {
    safe(path)?;
    train_manifest::sha256_file(path).map_err(|e| e.to_string())
}
fn pin(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}
fn member_source(member: &str) -> Option<&str> {
    match member {
        "kadid_train" | "kadid_select" => Some("kadid"),
        "konfig_train" | "konfig_val" => Some("konfig"),
        "tid2013" | "cid22_a25" => Some(member),
        _ => None,
    }
}
fn decision(path: &Path, d: &Value) -> Result<(), String> {
    let r = json(path)?;
    if r["schema"] != "shippath-human-role-decision-v1"
        || r["decision_id"] != "SHIPPATH-human-production-role"
        || r["state"] != "approved"
        || r["allowed_use"] != "qualified-recipe-training"
        || r["sources"] != serde_json::json!(SOURCES)
        || r["ledger_commit"] != "1d3bf35a78a149f0925029d71014c6df8535f550"
        || r["decided_by"].as_str().is_none_or(str::is_empty)
        || !pin(&r["source_receipt_sha256"])
        || r["source_receipt_sha256"] != d["source_receipt_sha256"]
        || d["data_role_decision_required"] != "SHIPPATH-human-production-role"
        || d["data_role_decision_sha256"] != hash(path)?
    {
        return Err("receipt-bound four-source D1 human decision required".into());
    }
    let sources = d["human_sources"]
        .as_array()
        .ok_or("human source inventory required")?;
    let names: BTreeSet<_> = sources.iter().filter_map(Value::as_str).collect();
    if names.is_empty()
        || names.len() != sources.len()
        || names.iter().any(|n| !SOURCES.contains(n))
    {
        return Err("unapproved/AIC human source".into());
    }
    Ok(())
}

fn receipt_root(path: &Path) -> Result<&Path, String> {
    path.parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .ok_or_else(|| "recipe root missing".into())
}

fn role_from_receipts<'a>(
    d: &Value,
    receipts: &'a [(PathBuf, Value)],
    name: &str,
) -> Result<(&'a Path, &'a Value, &'static str), String> {
    let split = if name.ends_with("_development") {
        "dev"
    } else {
        "fit"
    };
    for (path, r) in receipts {
        let Some(legs) = r["legs"].as_object() else {
            continue;
        };
        for (leg, record) in legs {
            let family = if name.starts_with("human") {
                leg.starts_with("human_")
            } else {
                leg == name.trim_end_matches("_development")
            };
            if !family {
                continue;
            }
            let rec = &record[split];
            let direct = d["table_sha256"] == rec["sha256"] && pin(&rec["sha256"]);
            let curated = d["source_table_sha256"] == rec["sha256"]
                && pin(&rec["sha256"])
                && d["source_manifest_sha256"] == rec["manifest_sha256"];
            if direct || curated {
                return Ok((
                    path,
                    rec,
                    if split == "dev" { "development" } else { "fit" },
                ));
            }
        }
    }
    Err("fit/development declaration must bind a recipe receipt".into())
}

pub(super) fn preflight(
    groups: &[Group],
    args: &Args,
    selected: Option<&[usize]>,
) -> Result<(), String> {
    if args.historical_replay.is_some() {
        if args.hdr_consensus_research || args.upiq_label_disposition.is_some() {
            return Err("research extensions require strict admission".into());
        }
        return Ok(()); // Existing explicitly unqualified historical entry.
    }
    if args.upiq_label_disposition.is_some() {
        let expected: BTreeSet<_> = [
            "safesyn",
            "safesyn_development",
            "cid22",
            "cid22_development",
            "coverage",
            "human",
            "human_development",
            "upiq380",
        ]
        .into_iter()
        .collect();
        let actual: BTreeSet<_> = groups.iter().map(|g| g.0.as_str()).collect();
        if actual != expected || actual.len() != groups.len() {
            return Err("E31 requires exactly the inherited seven SDR groups and upiq380; no HDR companion; registered Rev5/D1 population, pinned UPIQ fit manifest, exact 420-slot projection and pooled rank-only groups required".into());
        }
        upiq_training::native_group(groups, &args.target_column, args.target_scale)?;
        upiq_training::disposition(&json(args.upiq_label_disposition.as_ref().unwrap())?)?;
    }
    if args.hdr_consensus_research
        && (args.target_column != "human_score" || args.target_scale != 1.0)
    {
        return Err("E29 declared target/CLI transform mismatch".into());
    }
    // Collect ALL declarations and label-free recipe bindings first.
    let mut declarations = Vec::new();
    let mut receipts = Vec::new();
    for g in groups {
        safe(&g.1)?;
        safe(&g.1.with_extension("keys.parquet"))?;
        let sp = PathBuf::from(format!("{}.manifest.json", g.1.display()));
        let d = json(&sp)?;
        if !g.2.is_finite()
            || !g.3.is_finite()
            || g.2 < 0.0
            || g.3 < 0.0
            || ["role", "split", "tier"].iter().any(|k| {
                d[*k].as_str().is_some_and(|v| {
                    matches!(
                        v.to_ascii_lowercase().as_str(),
                        "val" | "validation" | "terminal" | "t0" | "test"
                    )
                })
            })
        {
            return Err("unapproved role or training/development weight".into());
        }
        let rp =
            g.1.parent()
                .ok_or("group directory missing")?
                .join("receipt.json");
        safe(&rp)?;
        if rp.is_file() && !receipts.iter().any(|(p, _)| p == &rp) {
            receipts.push((rp.clone(), json(&rp)?));
        }
        declarations.push((g, sp, d));
    }
    // Every group's role/feature declaration must pass before ANY key or table.
    let mut phases = Vec::new();
    for (g, sp, d) in &declarations {
        if d["row_selection"].is_array() && !d["rows_kept"].is_number() {
            use sha2::{Digest, Sha256};
            let bytes = serde_json::to_vec(&d["row_selection"]).map_err(|e| e.to_string())?;
            let digest: String = Sha256::digest(&bytes)
                .iter()
                .map(|v| format!("{v:02x}"))
                .collect();
            if d["row_selection_sha256"] != digest {
                return Err("declared ordered row selection pin changed".into());
            }
        }
        if d["study"] == "E29" {
            if !args.hdr_consensus_research
                || g.0 != "hdr"
                || g.2 <= 0.0
                || g.3 != 0.0
                || g.4
                || g.5 != GroupLossMode::Rank
            {
                return Err("E29 requires its named fit-only pooled-rank group".into());
            }
            phases.push(("hdr", 7390, None));
            continue;
        }
        if d["source"] == "UPIQ-380" {
            if args.upiq_label_disposition.is_none()
                || g.0 != "upiq380"
                || hash(sp)? != upiq_training::MANIFEST_SHA
            {
                return Err("E31 pinned UPIQ fit manifest required; malformed feature_set_id for ordinary SDR ingress".into());
            }
            let ids: Vec<usize> =
                serde_json::from_value(d["requested_ids"].clone()).map_err(|e| e.to_string())?;
            if selected != Some(ids.as_slice()) || args.max_features != 1853 {
                return Err("E31 exact native feature IDs required".into());
            }
            phases.push(("upiq", 330, None));
            continue;
        }
        for metadata in [
            g.1.parent().unwrap().join("_MANIFEST.json"),
            PathBuf::from(format!("{}._MANIFEST.json", g.1.display())),
        ] {
            safe(&metadata)?;
        }
        let identity =
            feature_set::table_feature_set_ref(&g.1)?.ok_or("feature identity required")?;
        let palette = d.get("research_palette").is_some();
        let ids = selected.ok_or("strict groups require explicit feature IDs")?;
        if !identity
            .slots
            .covers(&zensim::feature_set_id::SlotSet::from_slots(
                ids.iter().copied(),
            ))
            || identity.layout != Some(if palette { 1867 } else { 1825 })
            || (palette && args.max_features != 1867)
            || (!palette
                && (args.max_features > 1853 || ids.iter().any(|id| *id >= args.max_features)))
        {
            return Err("feature IDs/layout disagree with declaration".into());
        }
        if palette {
            let registered: Value = serde_json::from_str(include_str!(
                "../../../../benchmarks/e32_palette_feature_ids_2026-10-07.json"
            ))
            .map_err(|e| e.to_string())?;
            let expected: Vec<usize> =
                serde_json::from_value(registered["arm_ids"].clone()).map_err(|e| e.to_string())?;
            if ids != expected {
                return Err("E32 requires exactly the registered 462 IDs".into());
            }
        }
        if d["bank_manifest_sha256"]
            .as_object()
            .is_some_and(|m| m.keys().any(|k| k.to_ascii_lowercase().contains("aic")))
        {
            return Err("unapproved/AIC bank source".into());
        }
        if (!palette && identity.id.to_string() != BASE)
            || d["formula_revision"] != 5
            || d["decoder_era"].as_str().is_none_or(str::is_empty)
            || [
                "table_sha256",
                "keys_sha256",
                "row_keys_sha256",
                "row_selection_sha256",
            ]
            .iter()
            .any(|k| !pin(&d[*k]))
        {
            return Err(
                "bound feature/revision/decoder/table/key/selection declaration required".into(),
            );
        }
        let (phase, count, root) = if palette {
            let role = d["research_palette"]["role"]
                .as_str()
                .ok_or("palette role required")?;
            (
                if role.ends_with("development") {
                    "development"
                } else {
                    "fit"
                },
                training_keys::expected_rows(d)?,
                None,
            )
        } else if d["data_role"] == "TRAIN ordinal KADIS source_id%10<8; no human labels" {
            ("fit", training_keys::expected_rows(d)?, None)
        } else {
            let (rp, rec, phase) = role_from_receipts(d, &receipts, &g.0)?;
            if d["table_sha256"] == rec["sha256"] && hash(sp)? != rec["manifest_sha256"] {
                return Err("manifest differs from original fit/development binding".into());
            }
            let count = if d["rows_kept"].is_number() {
                training_keys::expected_rows(d)?
            } else {
                usize::try_from(
                    rec["rows"]
                        .as_u64()
                        .ok_or("receipt observation count required")?,
                )
                .map_err(|e| e.to_string())?
            };
            (phase, count, Some(rp))
        };
        if palette
            && phase != "hdr"
            && d["data_role"] != "TRAIN ordinal KADIS source_id%10<8; no human labels"
        {
            let mut found = None;
            for (rp, receipt) in &receipts {
                if let Some(legs) = receipt["legs"].as_object() {
                    for leg in legs.values() {
                        for split in ["fit", "dev"] {
                            let rec = &leg[split];
                            if rec["rel"]
                                .as_str()
                                .is_some_and(|rel| Path::new(rel).file_name() == g.1.file_name())
                                && pin(&rec["sha256"])
                                && (d["table_sha256"] == rec["sha256"]
                                    || d["research_palette"]["inherited_table_sha256"]
                                        == rec["sha256"])
                            {
                                let expected = if split == "dev" { "development" } else { "fit" };
                                if found.is_some_and(|v| v != expected) {
                                    return Err("ambiguous palette split binding".into());
                                }
                                found = Some(expected);
                            }
                        }
                    }
                    let _ = rp;
                }
            }
            if found != Some(phase) {
                return Err(
                    "palette role must match its original receipt fit/development binding".into(),
                );
            }
        }
        if let Some(role) = d["role"].as_str() {
            if !matches!(
                (role.to_ascii_lowercase().as_str(), phase),
                ("train" | "fit", "fit") | ("development" | "dev", "development")
            ) {
                return Err("declared role disagrees with original fit/development binding".into());
            }
        }
        if (phase == "development" && (g.2 != 0.0 || g.3 <= 0.0))
            || (phase == "fit" && (g.2 <= 0.0 || g.3 != 0.0))
        {
            return Err("fit/development role disagrees with training/validation weights".into());
        }
        if d["data_role"] == "design-released-human" {
            let rp = root
                .or_else(|| receipts.first().map(|(p, _)| p.as_path()))
                .ok_or("D1 receipt binding required")?;
            let root = receipt_root(rp)?;
            decision(&root.join("human_role_decision.json"), d)?;
            let receipt = json(rp)?;
            if receipt["admission_view"]["source_receipt_sha256"] != d["source_receipt_sha256"] {
                return Err("D1 original source receipt binding changed".into());
            }
        } else if !matches!(
            d["data_role"].as_str(),
            Some("TRAIN oracle teacher" | "TRAIN ordinal KADIS source_id%10<8; no human labels")
        ) {
            return Err("explicit original human/oracle/ordinal data role required".into());
        }
        phases.push((phase, count, root));
    }
    // Only label-free keys below. No ordinary header/hash/read until ALL pass.
    for ((g, _, d), (phase, count, _)) in declarations.iter().zip(phases) {
        if phase == "hdr" {
            e29_hdr_admission::admit_keys(
                &g.1,
                selected.ok_or("E29 exact IDs required")?,
                &args.rank_pair_list,
            )?;
            continue;
        }
        if phase == "upiq" {
            let kp = g.1.with_extension("keys.parquet");
            if hash(&kp)? != upiq_training::KEYS_SHA {
                return Err("E31 native key pin changed".into());
            }
            let names = [
                "condition_id",
                "dataset",
                "content",
                "distortion",
                "level",
                "reference_rel",
                "distorted_rel",
                "reference_sha256",
                "distorted_sha256",
                "split",
                "pair_key",
                "row_id",
                "role",
                "source",
                "authority",
            ];
            let keys = training_keys::read(&kp, &names, &names)?;
            let condition = keys.column("condition_id")?;
            let role = keys.column("role")?;
            let split = keys.column("split")?;
            if keys.rows.len() != count
                || keys.rows.iter().enumerate().any(|(i, r)| {
                    r[role] != "train" || r[split] != "fit" || d["member_set"][i] != r[condition]
                })
            {
                return Err("E31 original ordered TRAIN-fit conditions required".into());
            }
            continue;
        }
        let keys = training_keys::verify(&g.1, d, count)?;
        if d["data_role"] == "design-released-human" {
            let ix = keys.column("member_set")?;
            let actual: BTreeSet<_> = keys
                .rows
                .iter()
                .filter_map(|r| member_source(&r[ix]))
                .collect();
            let declared: BTreeSet<_> = d["human_sources"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_str)
                .collect();
            if actual != declared {
                return Err("D1 source declaration differs from original member keys".into());
            }
        }
    }
    Ok(())
}
