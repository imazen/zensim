//! Label-free synthetic geometry/cost receipt for the prepared exact mode.
//! This is an engineering probe, never a panel or qualification evaluator.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};
use zenpredict::Model;
use zensim::{BakeScorer, RgbSlice};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    bake: PathBuf,
    #[arg(long)]
    model_sha256: String,
    #[arg(long)]
    out: PathBuf,
}
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(!a.out.exists(), "fresh receipt required");
    ensure!(
        std::env::var("ZENSIM_STEER_EXACT").as_deref() == Ok("all-blocks-v1"),
        "exact policy required"
    );
    let bytes = fs::read(&a.bake)?;
    ensure!(hash(&bytes) == a.model_sha256, "model binding changed");
    let model = Model::from_bytes(&bytes).context("load pinned control")?;
    let mut cases = Vec::new();
    for (w, h) in [(65usize, 49usize), (128usize, 96usize)] {
        // Fixed procedural pixels: no source images, labels or panel inputs.
        let mut state = 0x5354454552u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        };
        let reference: Vec<[u8; 3]> = (0..w * h).map(|_| [next(), next(), next()]).collect();
        let distorted: Vec<[u8; 3]> = reference
            .iter()
            .map(|p| [p[0] / 2 + 37, p[1] / 2 + 23, p[2] / 2 + 11])
            .collect();
        let source = RgbSlice::new(&reference, w, h);
        let target = RgbSlice::new(&distorted, w, h);
        let mut ordinary = BakeScorer::new(&model)?.with_parallel(false);
        let start = Instant::now();
        let base = ordinary.compute(&source, &target, None)?;
        let ordinary_seconds = start.elapsed().as_secs_f64();
        let mut scorer = BakeScorer::new(&model)?.with_parallel(false);
        let start = Instant::now();
        let mut session = scorer.prepare_steering(&source, 1)?;
        let prepare_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mapped = session.compute(&target, None)?;
        let compute_seconds = start.elapsed().as_secs_f64();
        ensure!(
            mapped.result().score().to_bits() == base.score().to_bits(),
            "ordinary score changed"
        );
        ensure!(
            mapped.result().features() == base.features(),
            "ordinary features changed"
        );
        let start = Instant::now();
        let mut rectangles = Vec::new();
        for side in [8usize, 16, 32, 64] {
            for row in 0..h.div_ceil(side) {
                for column in 0..w.div_ceil(side) {
                    let [x0, y0, x1, y1] = [
                        column * side,
                        row * side,
                        ((column + 1) * side).min(w),
                        ((row + 1) * side).min(h),
                    ];
                    let mut repair = distorted.clone();
                    for y in y0..y1 {
                        repair[y * w + x0..y * w + x1]
                            .copy_from_slice(&reference[y * w + x0..y * w + x1]);
                    }
                    let truth = ordinary
                        .compute(&source, &RgbSlice::new(&repair, w, h), None)?
                        .score()
                        - base.score();
                    let gain = mapped.refinement_gain(x0, y0, x1, y1);
                    ensure!(
                        gain.is_finite() && gain.to_bits() == truth.to_bits(),
                        "native repair parity at {x0},{y0},{x1},{y1}"
                    );
                    rectangles.push(json!({"grid":side,"rectangle":[x0,y0,x1,y1],"gain_bits":gain.to_bits(),"native_gain_bits":truth.to_bits()}));
                }
            }
        }
        let oracle_seconds = start.elapsed().as_secs_f64();
        ensure!(
            rectangles.len()
                == [8usize, 16, 32, 64]
                    .iter()
                    .map(|s| w.div_ceil(*s) * h.div_ceil(*s))
                    .sum::<usize>(),
            "inventory incomplete"
        );
        cases.push(json!({"width":w,"height":h,"reference_pixels_sha256":hash(bytemuck::cast_slice(&reference)),
            "distorted_pixels_sha256":hash(bytemuck::cast_slice(&distorted)),"ordinary_seconds":ordinary_seconds,
            "prepare_seconds":prepare_seconds,"compute_all_exact_seconds":compute_seconds,"independent_oracle_seconds":oracle_seconds,
            "score_and_feature_bits_equal":true,"rectangles":rectangles}));
    }
    fs::write(
        a.out,
        serde_json::to_vec_pretty(
            &json!({"schema":"steer-exact-synthetic-smoke-v1","model_sha256":a.model_sha256,
        "scope":"SYNTHETIC_GEOMETRY_ONLY_NOT_QUALIFICATION","policy":"all-blocks-v1","cases":cases}),
        )?,
    )?;
    Ok(())
}
