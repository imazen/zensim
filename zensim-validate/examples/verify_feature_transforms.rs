//! V0_20 smoke test: verify that a bake with `feature_transforms`
//! metadata round-trips correctly. Loads a tiny bake produced by
//! `zensim_mlp_train --feature-transform <tok>:<idx>`, parses the
//! metadata via `Model::feature_transforms()`, and runs forward.

use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bake_path = env::args()
        .nth(1)
        .ok_or("usage: verify_feature_transforms <bake.bin>")?;
    let bytes = fs::read(&bake_path)?;
    let model = zenpredict::Model::from_bytes(&bytes)?;
    println!("loaded {} ({} bytes)", bake_path, bytes.len());
    println!("n_inputs:  {}", model.n_inputs());
    println!("n_outputs: {}", model.n_outputs());
    match model.feature_transforms() {
        Some(ts) => {
            println!("feature_transforms ({}):", ts.len());
            for (i, t) in ts.iter().enumerate() {
                if *t != zenpredict::FeatureTransform::Identity {
                    println!("  f{i} = {}", t.as_token());
                }
            }
        }
        None => println!("feature_transforms: none (all-identity)"),
    }
    let mut p = zenpredict::Predictor::new(&model);
    let n = model.n_inputs();
    let features: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let out_transformed = p.predict_transformed(&features)?.to_vec();
    let out_raw = p.predict(&features)?.to_vec();
    println!("input              = {:?}", features);
    println!("predict()          = {:?}", out_raw);
    println!("predict_transformed = {:?}", out_transformed);
    if model.feature_transforms().is_some() {
        println!(
            "transformed≠raw   = {}",
            (out_transformed[0] - out_raw[0]).abs() > 1e-6
        );
    }
    Ok(())
}
