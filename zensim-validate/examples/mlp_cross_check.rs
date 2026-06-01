//! Cross-validate `site/js/mlp.js` against the Rust `zenpredict` runtime.
//!
//! Loads V0_16 (`zensim-experimental/weights/archive/v0_16_2026-05-12.bin`, ZNPR v2), runs
//! the forward pass on two known feature vectors, and prints the
//! outputs. The JS `predict()` should produce identical numbers
//! (within float-precision noise) for the same inputs.
//!
//! Expected JS outputs (from `/tmp/mlp_test.mjs` smoke test):
//!   predict([0.5]×228) → 815.8024
//!   predict([0.0]×228) → 115.4504
//!
//! Run:
//!   cargo run -p zensim-validate --release --example mlp_cross_check

use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path =
        "/home/lilith/work/zen/zensim/zensim-experimental/weights/archive/v0_16_2026-05-12.bin";
    let bytes = fs::read(path)?;
    println!("loaded {path} ({} bytes)", bytes.len());

    let model = zenpredict::Model::from_bytes(&bytes)?;
    let mut p = zenpredict::Predictor::new(&model);

    let features_05 = vec![0.5f32; 228];
    let out_05 = p.predict(&features_05)?;
    println!("Rust predict([0.5]×228) = {:?}", out_05);

    let features_0 = vec![0.0f32; 228];
    let out_0 = p.predict(&features_0)?;
    println!("Rust predict([0.0]×228) = {:?}", out_0);

    // For a sanity test, also try a vector of distinct increasing values
    let features_seq: Vec<f32> = (0..228).map(|i| (i as f32) / 228.0).collect();
    let out_seq = p.predict(&features_seq)?;
    println!("Rust predict(i/228 for i in 0..228) = {:?}", out_seq);

    Ok(())
}
