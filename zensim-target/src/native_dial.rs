//! Private evaluation dial. Peer metric packets use the frozen fixed-judge owner.
use crate::CodecKind;
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use zensim::{BakeScorer, ImageSource, RgbSlice};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Deserialize)]
struct PeerSpec {
    metric: String,
    alpha: f64,
    beta: f64,
    judge: PathBuf,
    judge_sha256: String,
}
pub(crate) struct DialScorer<'a> {
    bake: BakeScorer<'a>,
    peer: Option<PeerSpec>,
    root: PathBuf,
    cache: BTreeMap<String, f32>,
}
impl<'a> DialScorer<'a> {
    pub(crate) fn new(
        model: &'a zenpredict::Model,
        peer: Option<&Path>,
        root: &Path,
    ) -> Result<Self> {
        let peer: Option<PeerSpec> = peer
            .map(|p| -> Result<_> { Ok(serde_json::from_slice(&fs::read(p)?)?) })
            .transpose()?;
        if let Some(p) = &peer {
            ensure!(
                matches!(p.metric.as_str(), "ssim2" | "butter"),
                "unsupported peer dial"
            );
            ensure!(
                p.alpha.is_finite() && p.alpha > 0. && p.beta.is_finite(),
                "invalid TRAIN mapping"
            );
            ensure!(
                sha(&fs::read(&p.judge)?) == p.judge_sha256,
                "fixed judge changed"
            );
        }
        Ok(Self {
            bake: BakeScorer::new(model)?,
            peer,
            root: root.into(),
            cache: BTreeMap::new(),
        })
    }
    pub(crate) fn compute(
        &mut self,
        codec: CodecKind,
        a: &RgbSlice<'_>,
        b: &RgbSlice<'_>,
    ) -> Result<f32> {
        let Some(peer) = &self.peer else {
            return Ok(self.bake.compute(a, b, Some(codec.extension()))?.score() as f32);
        };
        ensure!(
            (a.width(), a.height()) == (b.width(), b.height()),
            "peer dimensions"
        );
        let reference: Vec<u8> = (0..a.height())
            .flat_map(|y| a.row_bytes(y).iter().copied())
            .collect();
        let distorted: Vec<u8> = (0..b.height())
            .flat_map(|y| b.row_bytes(y).iter().copied())
            .collect();
        let key = format!(
            "{}-{}-{}x{}",
            sha(&reference),
            sha(&distorted),
            a.width(),
            a.height()
        );
        if let Some(value) = self.cache.get(&key) {
            return Ok(*value);
        }
        let root = self.root.join(&key);
        fs::create_dir_all(&root)?;
        let rp = root.join("reference.png");
        let dp = root.join("distorted.png");
        image::RgbImage::from_raw(a.width() as u32, a.height() as u32, reference)
            .context("reference shape")?
            .save(&rp)?;
        image::RgbImage::from_raw(b.width() as u32, b.height() as u32, distorted)
            .context("distorted shape")?
            .save(&dp)?;
        let pairs = root.join("pairs.tsv");
        fs::write(
            &pairs,
            format!("ref_path\tdist_path\n{}\t{}\n", rp.display(), dp.display()),
        )?;
        let packet = root.join("packet");
        let output = Command::new(&peer.judge)
            .arg("--pairs")
            .arg(&pairs)
            .arg("--out-dir")
            .arg(&packet)
            .args([
                "--metrics",
                &peer.metric,
                "--ssim2-map-policy",
                "scalar-only",
            ])
            .env("ZENSIM_FORMULA_REV", "1")
            .env("RAYON_NUM_THREADS", "1")
            .output()?;
        fs::write(root.join("judge.stdout"), &output.stdout)?;
        fs::write(root.join("judge.stderr"), &output.stderr)?;
        ensure!(
            output.status.success(),
            "fixed judge failed: {}",
            root.display()
        );
        ensure!(
            packet.join("COMPLETE.json").is_file(),
            "incomplete peer packet"
        );
        let lines = fs::read_to_string(packet.join("judges.jsonl"))?;
        let rows: Vec<_> = lines.lines().collect();
        ensure!(rows.len() == 1, "peer packet row count");
        let row: serde_json::Value = serde_json::from_str(rows[0])?;
        let raw = if peer.metric == "ssim2" {
            row["ssim2"].as_f64()
        } else {
            row["butteraugli_pnorm3"].as_f64().map(|v| -v)
        }
        .context("missing peer score")?;
        let value = ((raw - peer.beta) / peer.alpha) as f32;
        ensure!(value.is_finite(), "nonfinite normalized peer score");
        fs::write(
            root.join("NORMALIZED.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"oriented_peer":raw,"alpha":peer.alpha,"beta":peer.beta,"normalized_f32":value}),
            )?,
        )?;
        self.cache.insert(key, value);
        Ok(value)
    }
}
