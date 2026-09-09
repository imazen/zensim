//! Optional native pixel identity and complete-candidate serving audit.
use super::{Pair, zen_decode};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use zenpredict::Model;
use zensim::corruption_head::CorruptionHead;
use zensim::{BakeScorer, RgbSlice};

pub(super) fn take_path(slot: &mut Option<PathBuf>, value: Option<String>) {
    assert!(slot.is_none(), "duplicate audit option");
    let value = value.expect("missing audit option value");
    assert!(!value.starts_with("--"), "missing audit option value");
    *slot = Some(value.into());
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn file_sha(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|v| sha(&v))
        .map_err(|e| format!("{}: {e}", path.display()))
}
pub(super) fn file_hashes(pair: &Pair) -> Result<(String, String), String> {
    Ok((file_sha(&pair.reference)?, file_sha(&pair.distorted)?))
}

pub(super) fn validate_pairs_input(path: &Path, pairs: &[Pair], max: usize) -> Result<(), String> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .map_err(|e| e.to_string())?;
    let header = reader.headers().map_err(|e| e.to_string())?.clone();
    let column = |name| {
        header
            .iter()
            .position(|x| x == name)
            .ok_or_else(|| format!("audit input missing {name}"))
    };
    let (ri, di, hi) = (
        column("ref_path")?,
        column("dist_path")?,
        column("human_score")?,
    );
    let mut count = 0;
    for row in reader.records().take(max) {
        let row = row.map_err(|e| e.to_string())?;
        let value: f64 = row
            .get(hi)
            .ok_or("missing row key")?
            .parse()
            .map_err(|_| "invalid row key")?;
        let pair = pairs
            .get(count)
            .ok_or("audit input rows were silently dropped")?;
        if !value.is_finite()
            || value != pair.human_score
            || Path::new(row.get(ri).ok_or("missing reference")?) != pair.reference
            || Path::new(row.get(di).ok_or("missing distortion")?) != pair.distorted
        {
            return Err("audit input keys/paths disagree with loaded pairs".into());
        }
        count += 1;
    }
    if count != pairs.len() {
        return Err("audit input row coverage mismatch".into());
    }
    Ok(())
}

pub(super) struct Config {
    out: PathBuf,
    model: Option<Model>,
    head: Option<CorruptionHead>,
    inputs: Vec<(PathBuf, String)>,
}

impl Config {
    pub(super) fn load(
        out: Option<PathBuf>,
        bake: Option<PathBuf>,
        head: Option<PathBuf>,
        csv: &Path,
        failures: usize,
    ) -> Result<Option<Self>, String> {
        let Some(out) = out else {
            if bake.is_some() || head.is_some() {
                return Err("audit model requires --audit-jsonl".into());
            }
            return Ok(None);
        };
        if out.exists() || csv.exists() || out == csv || failures != 0 {
            return Err("audit requires distinct fresh outputs and zero allowed failures".into());
        }
        if head.is_some() && bake.is_none() {
            return Err("audit companion requires --audit-bake".into());
        }
        let mut inputs = Vec::new();
        let model = bake
            .map(|p| {
                let bytes = fs::read(&p).map_err(|e| e.to_string())?;
                let model = Model::from_bytes(&bytes).map_err(|e| e.to_string())?;
                inputs.push((p, sha(&bytes)));
                Ok::<_, String>(model)
            })
            .transpose()?;
        let head = head
            .map(|p| {
                let bytes = fs::read(&p).map_err(|e| e.to_string())?;
                let head = CorruptionHead::from_bytes(&bytes).map_err(|e| e.to_string())?;
                if head.caller_input_width() != 372 {
                    return Err("audit requires a canonical w372 companion".into());
                }
                inputs.push((p, sha(&bytes)));
                Ok::<_, String>(head)
            })
            .transpose()?;
        if let Some(model) = &model {
            let scorer = BakeScorer::new(model).map_err(|e| e.to_string())?;
            if let Some(head) = &head {
                scorer
                    .with_corruption_head(head, None)
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(Some(Self {
            out,
            model,
            head,
            inputs,
        }))
    }

    pub(super) fn score(
        &self,
        pair: &Pair,
        features: &[f64],
        hashes: &(String, String),
    ) -> Result<Value, String> {
        if features.len() != 372 || !features.iter().all(|v| v.is_finite()) {
            return Err("audit requires 372 finite canonical features".into());
        }
        // Reuse the native decoder owner; a changed file across either extraction
        // or this independent pixel-surface check invalidates the audit.
        let src = zen_decode::decode_rgb8_path(&pair.reference).map_err(|e| e.to_string())?;
        let dst = zen_decode::decode_rgb8_path(&pair.distorted).map_err(|e| e.to_string())?;
        if file_hashes(pair)? != *hashes || (src.width, src.height) != (dst.width, dst.height) {
            return Err("audit input bytes/dimensions changed".into());
        }
        let identical = src.pixels == dst.pixels;
        let mut record = json!({"schema":"canonical-feature-audit-v1", "reference":pair.reference,
            "distorted":pair.distorted,"ref_basename":pair.ref_basename,"human_score":pair.human_score,
            "extra_targets":pair.extra_targets,"width":src.width,"height":src.height,
            "reference_file_sha256":hashes.0,"distorted_file_sha256":hashes.1,
            "reference_pixels_sha256":sha(&src.pixels),"distorted_pixels_sha256":sha(&dst.pixels),
            "pixels_identical":identical,
            "canonical_extractions":1,"audit_decodes":2,"model_inputs":self.inputs});
        if let Some(model) = &self.model {
            let mut base = BakeScorer::new(model).map_err(|e| e.to_string())?;
            let base_score = base
                .score_features_with_identity(features, src.width, src.height, None, identical)
                .map_err(|e| e.to_string())?;
            let mut scorer = BakeScorer::new(model).map_err(|e| e.to_string())?;
            if let Some(head) = &self.head {
                scorer = scorer
                    .with_corruption_head(head, None)
                    .map_err(|e| e.to_string())?;
            }
            let computed = scorer
                .compute(
                    &RgbSlice::new(
                        src.pixels.as_chunks::<3>().0,
                        src.width as usize,
                        src.height as usize,
                    ),
                    &RgbSlice::new(
                        dst.pixels.as_chunks::<3>().0,
                        dst.width as usize,
                        dst.height as usize,
                    ),
                    None,
                )
                .map_err(|e| e.to_string())?;
            let score = computed.score();
            let literal_cached = scorer
                .score_features(features, src.width, src.height, None)
                .map_err(|e| e.to_string())?;
            let cached = scorer
                .score_features_with_identity(features, src.width, src.height, None, identical)
                .map_err(|e| e.to_string())?;
            let stored: Vec<f64> = features.iter().map(|&v| f64::from(v as f32)).collect();
            let cached_f32 = scorer
                .score_features_with_identity(&stored, src.width, src.height, None, identical)
                .map_err(|e| e.to_string())?;
            let mut max_abs: f64 = 0.0;
            if let Some(head) = &self.head {
                let mut pixel_features = vec![0.0; head.caller_input_width()];
                for &id in head.declared_feature_ids() {
                    let id = usize::from(id);
                    let actual = *computed
                        .features()
                        .get(id)
                        .ok_or("missing served feature")?;
                    let expected = features[id];
                    pixel_features[id] = actual;
                    let diff = (actual - expected).abs();
                    max_abs = max_abs.max(diff);
                    if !actual.is_finite() || diff > 1e-6 + 1e-5 * expected.abs() {
                        return Err(format!(
                            "consumed feature f{id}: canonical {expected} versus served {actual}"
                        ));
                    }
                }
                record["head_probability"] =
                    json!(head.probability_f64(features).map_err(|e| e.to_string())?);
                record["head_threshold"] = json!(head.deadband());
                let stored_p = head.probability_f64(&stored).map_err(|e| e.to_string())?;
                let pixel_p = head
                    .probability_f64(&pixel_features)
                    .map_err(|e| e.to_string())?;
                let raw_p = record["head_probability"]
                    .as_f64()
                    .ok_or("nonfinite head probability")?;
                if (stored_p > head.deadband()) != (raw_p > head.deadband()) {
                    return Err("stored-f32 versus pixel head fire mismatch".into());
                }
                if (pixel_p > head.deadband()) != (raw_p > head.deadband()) {
                    return Err("canonical versus served pixel head fire mismatch".into());
                }
                record["stored_f32_head_probability"] = json!(stored_p);
                record["served_pixel_head_probability"] = json!(pixel_p);
                record["canonical_head_raw"] = json!(
                    head.decision_function(features)
                        .map_err(|e| e.to_string())?
                );
                record["stored_f32_head_raw"] =
                    json!(head.decision_function(&stored).map_err(|e| e.to_string())?);
                record["served_pixel_head_raw"] = json!(
                    head.decision_function(&pixel_features)
                        .map_err(|e| e.to_string())?
                );
            }
            if ![score, cached, cached_f32, base_score]
                .iter()
                .all(|v| v.is_finite())
                || (score - cached).abs() > 1e-4
                || (score - cached_f32).abs() > 1e-4
            {
                return Err(format!(
                    "cached/pixel score disagreement: {cached} versus {score}"
                ));
            }
            record["base_score"] = json!(base_score);
            record["pixel_composed_score"] = json!(score);
            record["cached_composed_score"] = json!(cached);
            record["stored_f32_composed_score"] = json!(cached_f32);
            record["literal_feature_composed_score"] = json!(literal_cached);
            record["max_consumed_feature_abs_delta"] = json!(max_abs);
            record["candidate_pixel_comparisons"] = json!(1);
            record["cached_comparisons"] = json!(4);
            record["auxiliary_head_evaluations"] = json!(6 * usize::from(self.head.is_some()));
        }
        Ok(record)
    }

    pub(super) fn write(&self, rows: &[Value], expected: usize) -> Result<(), String> {
        if rows.len() != expected {
            return Err("missing audit rows".into());
        }
        for (path, hash) in &self.inputs {
            if file_sha(path)? != *hash {
                return Err("audit model changed during extraction".into());
            }
        }
        let f = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&self.out)
            .map_err(|e| e.to_string())?;
        let mut w = BufWriter::new(f);
        for row in rows {
            serde_json::to_writer(&mut w, row).map_err(|e| e.to_string())?;
            writeln!(w).map_err(|e| e.to_string())?;
        }
        w.flush().map_err(|e| e.to_string())
    }
}
