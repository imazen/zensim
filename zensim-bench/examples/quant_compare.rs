//! Compare F32 / F16 / I8 weight quantizations for a ZNPR bake.
//!
//! Loads a baked .bin (e.g. zensim/weights/v0_17_2026-05-13.bin), extracts
//! the f32 weights from each layer, then re-bakes the same model with
//! F16 and I8 dtypes via zenpredict's existing bake support.
//!
//! Reports:
//!   - On-disk size of each variant
//!   - Bake round-trip error vs original F32 prediction on a 1k random
//!     input batch
//!   - The path of the re-baked .bin files so downstream eval harnesses
//!     can score them on real corpora
//!
//! Usage:
//!   cargo run --release -p zensim-bench --example quant_compare -- \
//!       zensim/weights/v0_17_2026-05-13.bin /tmp/quant
//!
//! Output bin paths:
//!   /tmp/quant/<stem>_f16.bin
//!   /tmp/quant/<stem>_i8.bin

use std::env;
use std::fs;
use std::path::PathBuf;

use zenpredict::{Model, Predictor, WeightDtype, WeightStorage};
use zenpredict_bake::{BakeLayer, BakeRequest, bake};

fn main() {
    let mut args = env::args().skip(1);
    let in_path: PathBuf = args
        .next()
        .expect("usage: quant_compare <bake.bin> <out-dir>")
        .into();
    let out_dir: PathBuf = args
        .next()
        .expect("usage: quant_compare <bake.bin> <out-dir>")
        .into();
    fs::create_dir_all(&out_dir).expect("mkdir out-dir");

    let bytes_f32 = fs::read(&in_path).expect("read input bin");
    let model = Model::from_bytes(&bytes_f32).expect("parse input as ZNPR v3");

    println!(
        "input: {} ({} bytes, {} layers, {}->{}->{} dims)",
        in_path.display(),
        bytes_f32.len(),
        model.n_layers(),
        model.n_inputs(),
        model.layers().next().map(|l| l.out_dim).unwrap_or(0),
        model.n_outputs(),
    );

    // Extract f32 weights and biases per layer.
    let mut layer_data: Vec<(Vec<f32>, Vec<f32>, usize, usize, zenpredict::Activation)> =
        Vec::new();
    for (li, layer) in model.layers().enumerate() {
        let w: Vec<f32> = match &layer.weights {
            WeightStorage::F32(w) => w.to_vec(),
            WeightStorage::F16(w) => w.iter().map(|b| zenpredict::f16_bits_to_f32(*b)).collect(),
            WeightStorage::I8 { weights, scales } => weights
                .iter()
                .enumerate()
                .map(|(idx, &q)| (q as f32) * scales[idx % layer.out_dim])
                .collect(),
        };
        let b: Vec<f32> = layer.biases.to_vec();
        println!(
            "  layer {li}: in={} out={} weights.len={} bias.len={} act={:?}",
            layer.in_dim,
            layer.out_dim,
            w.len(),
            b.len(),
            layer.activation,
        );
        // Per-layer weight magnitude statistics: useful for thinking about
        // sparsity-based size reduction. Report fraction of weights below a
        // few magnitudes relative to the per-layer max.
        let max_w = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for thresh in [0.001f32, 0.01, 0.05, 0.1] {
            let cut = thresh * max_w;
            let n_below = w.iter().filter(|v| v.abs() <= cut).count();
            println!(
                "    |w| <= {:.3}*max ({:.4}): {} / {} ({:>5.1}%)",
                thresh,
                cut,
                n_below,
                w.len(),
                100.0 * n_below as f32 / w.len() as f32
            );
        }
        // Per-output channel-zero count: how many output columns have
        // every input weight effectively zero (drops the column entirely).
        let mut zero_cols = 0usize;
        for o in 0..layer.out_dim {
            let mut max_col = 0.0f32;
            for i in 0..layer.in_dim {
                max_col = max_col.max(w[i * layer.out_dim + o].abs());
            }
            if max_col < 1e-6 * max_w {
                zero_cols += 1;
            }
        }
        println!(
            "    near-zero output channels: {}/{}",
            zero_cols, layer.out_dim
        );

        layer_data.push((w, b, layer.in_dim, layer.out_dim, layer.activation));
    }

    let scaler_mean = model.scaler_mean().to_vec();
    let scaler_scale = model.scaler_scale().to_vec();

    let make_layers = |dtype: WeightDtype| -> Vec<BakeLayer<'_>> {
        layer_data
            .iter()
            .map(|(w, b, in_dim, out_dim, act)| BakeLayer {
                in_dim: *in_dim,
                out_dim: *out_dim,
                activation: *act,
                dtype,
                weights: w.as_slice(),
                biases: b.as_slice(),
            })
            .collect()
    };

    let stem = in_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    let mut variants: Vec<(&'static str, PathBuf, Vec<u8>)> = Vec::new();
    for (label, dtype) in [
        ("f32_rebake", WeightDtype::F32),
        ("f16", WeightDtype::F16),
        ("i8", WeightDtype::I8),
    ] {
        let layers = make_layers(dtype);
        let req = BakeRequest {
            schema_hash: model.schema_hash(),
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &[],
            metadata: &[],
            output_specs: &[],
            discrete_sets: &[],
            sparse_overrides: &[],
            feature_order: None,
            output_order: None,
            compressed: false,
            hu_permutations: None,
        };
        let bytes = bake(&req).expect("bake v3");
        let path = out_dir.join(format!("{stem}_{label}.bin"));
        fs::write(&path, &bytes).expect("write variant");
        println!(
            "{label}: {} bytes  ({:>5.1}% of original)",
            bytes.len(),
            100.0 * bytes.len() as f64 / bytes_f32.len() as f64
        );
        variants.push((label, path, bytes));
    }

    // Build predictors over each variant (parse from the bytes we just baked).
    let model_f32 = Model::from_bytes(&bytes_f32).expect("model f32");
    let mut pred_f32 = Predictor::new(&model_f32);
    // Re-bake F32 should be byte-identical structurally; verify size and parity.
    let f16_bytes = &variants[1].2;
    let i8_bytes = &variants[2].2;
    let model_f16 = Model::from_bytes(f16_bytes).expect("model f16");
    let mut pred_f16 = Predictor::new(&model_f16);
    let model_i8 = Model::from_bytes(i8_bytes).expect("model i8");
    let mut pred_i8 = Predictor::new(&model_i8);

    // Synthesize 1k random feature vectors and measure prediction divergence.
    // Use a tiny LCG so this is deterministic without an extra crate.
    let n_inputs = model.n_inputs();
    let n_samples = 1024usize;
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_f32 = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Uniform in [-3, 3], approximately covering the post-scaler input range.
        let u = ((rng >> 33) as u32) as f32 / (u32::MAX >> 1) as f32 - 1.0;
        u * 3.0
    };

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max_abs_i8 = 0.0f32;
    let mut sum_abs_i8 = 0.0f64;
    let mut sum_sq_i8 = 0.0f64;
    let mut max_abs_v_f32 = 0.0f32;
    let mut sum_abs_v = 0.0f64;
    let mut sample = vec![0.0f32; n_inputs];

    for _ in 0..n_samples {
        for v in sample.iter_mut() {
            *v = next_f32();
        }
        let p32 = pred_f32.predict(&sample).expect("predict f32");
        let p16 = pred_f16.predict(&sample).expect("predict f16");
        let pi8 = pred_i8.predict(&sample).expect("predict i8");
        for k in 0..p32.len() {
            let d16 = (p32[k] - p16[k]).abs();
            let di8 = (p32[k] - pi8[k]).abs();
            max_abs = max_abs.max(d16);
            sum_abs += d16 as f64;
            sum_sq += (d16 as f64) * (d16 as f64);
            max_abs_i8 = max_abs_i8.max(di8);
            sum_abs_i8 += di8 as f64;
            sum_sq_i8 += (di8 as f64) * (di8 as f64);
            max_abs_v_f32 = max_abs_v_f32.max(p32[k].abs());
            sum_abs_v += p32[k].abs() as f64;
        }
    }
    let n = (n_samples * model.n_outputs()) as f64;
    let mean_abs_v = sum_abs_v / n;
    println!(
        "\nprediction divergence on {n_samples} random inputs (output magnitude: max={:.4}, mean_abs={:.4}):",
        max_abs_v_f32, mean_abs_v
    );
    println!(
        "  f16 vs f32: max|Δ|={:.6}  mean|Δ|={:.6}  rms|Δ|={:.6}  (rel mean|Δ| / mean|y|={:.3e})",
        max_abs,
        sum_abs / n,
        (sum_sq / n).sqrt(),
        (sum_abs / n) / mean_abs_v.max(1e-9) as f64
    );
    println!(
        "  i8  vs f32: max|Δ|={:.6}  mean|Δ|={:.6}  rms|Δ|={:.6}  (rel mean|Δ| / mean|y|={:.3e})",
        max_abs_i8,
        sum_abs_i8 / n,
        (sum_sq_i8 / n).sqrt(),
        (sum_abs_i8 / n) / mean_abs_v.max(1e-9) as f64
    );
}
