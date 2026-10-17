//! Feature-only raw selection forward through the existing production owner.
use anyhow::{Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::{BufRead, Write};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    bake: std::path::PathBuf,
    bake_sha256: String,
    feature_ids: Vec<u16>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    row_idx: u64,
    width: u32,
    height: u32,
    features: Vec<f32>,
}
fn main() -> Result<()> {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let request: Request = serde_json::from_str(
        &lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("request required"))??,
    )?;
    let bytes = std::fs::read(&request.bake)?;
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    ensure!(digest == request.bake_sha256, "bake digest changed");
    let model =
        zenpredict::Model::from_bytes(&bytes).map_err(|e| anyhow::anyhow!("parse bake: {e:?}"))?;
    let mut scorer = zensim::BakeScorer::new(&model)
        .map_err(|e| anyhow::anyhow!("raw scorer: {e:?}"))?
        .without_output_calibration()
        .with_parallel(false);
    ensure!(
        request.feature_ids.len() == 420
            && request.feature_ids.windows(2).all(|x| x[0] < x[1])
            && request.feature_ids.iter().all(|&x| x < 1853),
        "fixed consumed native layout required"
    );
    ensure!(
        scorer
            .consumed_feature_ids()
            .map_err(|e| anyhow::anyhow!("consumed IDs: {e:?}"))?
            == request.feature_ids,
        "model consumed IDs differ from registered native pool"
    );
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for (expected, line) in lines.enumerate() {
        let row: Row = serde_json::from_str(&line?)?;
        ensure!(
            row.row_idx == expected as u64
                && row.features.len() == 420
                && row.features.iter().all(|v| v.is_finite())
                && row.width > 0
                && row.height > 0,
            "ordered finite native feature row required"
        );
        // Only unconsumed slots are zero. Every model read is supplied above;
        // derived fx1 products and base scaling stay in BakeScorer.
        let mut raw = vec![0.; 1853];
        for (&id, &value) in request.feature_ids.iter().zip(&row.features) {
            raw[id as usize] = f64::from(value);
        }
        let distance = scorer
            .score_features(&raw, row.width, row.height, None)
            .map_err(|e| anyhow::anyhow!("raw forward: {e:?}"))?;
        ensure!(distance.is_finite(), "nonfinite raw distance");
        writeln!(
            output,
            "{}",
            serde_json::json!({"row_idx":row.row_idx,"raw_distance":distance})
        )?;
    }
    output.flush()?;
    Ok(())
}
