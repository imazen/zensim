//! Concat-three-way construction for V0_17 / V0_19-style ensembles.
//!
//! Reads three 228→128→1 single-MLP F32 bakes (V0_16-base equivalent +
//! cycle-14 seed=1 TV-regularized + cycle-14 seed=42 TV-regularized) and
//! stacks them into a single 228→384→1 ensemble bake. The
//! construction is mathematically equivalent to the output average
//!
//!   y = w_a * y_a + w_b * y_b + w_c * y_c
//!
//! where each y_i is a separate MLP's output. The wire layout uses
//! diagonal block structure: layer 0 weights are the three 228×128
//! sub-matrices side-by-side along the hidden dim; layer 1 weights
//! are the three 128-row vectors with the mix coefficients baked in.
//!
//! Per `benchmarks/v0_18_methodology_2026-05-13.md` §2.4 the V0_18
//! shipping coefficients are 0.65/0.30/0.05 (base / s1 / s42). V0_19
//! uses the same coefficients by default; override via `--coeffs A:B:C`.
//!
//! ## Output
//!
//! Writes a v3 F32 bake at `--out PATH`. Down-stream consumers can
//! either ship that directly OR pass it through `quant_compare` to
//! get the I8 variant.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release -p zensim-validate --bin concat_three_way -- \
//!   --base benchmarks/v0_19_base_seed1_2026-05-14.bin \
//!   --s1   benchmarks/v0_19_cycle14_s1_2026-05-14.bin \
//!   --s42  benchmarks/v0_19_cycle14_s42_2026-05-14.bin \
//!   --out  benchmarks/v0_19_concat_3way_2026-05-14.bin
//! ```

use std::fs;
use std::path::PathBuf;

use clap::Parser;
use zenpredict::{Model, WeightStorage};
use zenpredict_bake::{BakeLayer, BakeRequest, bake};

#[derive(Parser)]
#[command(about = "Concat three single-MLP bakes into one wide-MLP ensemble bake")]
struct Args {
    /// V0_16-base equivalent bake (228→128→1 F32, no TV).
    #[arg(long)]
    base: PathBuf,
    /// cycle-14 seed=1 TV-regularized bake (228→128→1 F32).
    #[arg(long)]
    s1: PathBuf,
    /// cycle-14 seed=42 TV-regularized bake.
    #[arg(long)]
    s42: PathBuf,
    /// Mix coefficients as A:B:C (default 0.65:0.30:0.05 — V0_18 ship recipe).
    #[arg(long, default_value = "0.65:0.30:0.05")]
    coeffs: String,
    /// Output ensemble bake path (228→384→1 F32).
    #[arg(long)]
    out: PathBuf,
}

fn parse_coeffs(s: &str) -> (f32, f32, f32) {
    let parts: Vec<&str> = s.split(':').collect();
    assert_eq!(parts.len(), 3, "--coeffs must be A:B:C");
    let a: f32 = parts[0].parse().expect("invalid A");
    let b: f32 = parts[1].parse().expect("invalid B");
    let c: f32 = parts[2].parse().expect("invalid C");
    let sum = a + b + c;
    if (sum - 1.0).abs() > 1e-3 {
        eprintln!("warning: coeffs sum to {sum} (not 1.0). Proceeding as-is.");
    }
    (a, b, c)
}

/// Returns (scaler_mean, scaler_scale, W0[n_in*128], b0[128], W1[128], b1[1]).
fn load_single_mlp(path: &PathBuf) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32) {
    let bytes = fs::read(path).expect("read bake");
    let model = Model::from_bytes(&bytes).expect("parse bake");
    assert_eq!(model.n_layers(), 2, "expected 2-layer MLP at {path:?}");
    // Input width is whatever the bake was trained at (228 for V0_18,
    // 372 for V0_20a IW-feature bakes). The concat just needs the
    // three inputs to MATCH each other — checked in the caller.
    assert_eq!(model.n_outputs(), 1, "expected 1 output at {path:?}");

    let scaler_mean = model.scaler_mean().to_vec();
    let scaler_scale = model.scaler_scale().to_vec();

    let l0 = model.layer(0);
    assert_eq!(l0.out_dim, 128, "expected 128-wide hidden at {path:?}");
    let w0: Vec<f32> = match &l0.weights {
        WeightStorage::F32(w) => w.to_vec(),
        WeightStorage::F16(w) => w.iter().map(|b| zenpredict::f16_bits_to_f32(*b)).collect(),
        WeightStorage::I8 { weights, scales } => {
            let mut out = Vec::with_capacity(weights.len());
            for (idx, w) in weights.iter().enumerate() {
                let o = idx % 128;
                out.push(*w as f32 * scales[o]);
            }
            out
        }
    };
    let b0 = l0.biases.to_vec();

    let l1 = model.layer(1);
    assert_eq!(l1.in_dim, 128, "expected 128-wide hidden→1 at {path:?}");
    let w1: Vec<f32> = match &l1.weights {
        WeightStorage::F32(w) => w.to_vec(),
        WeightStorage::F16(w) => w.iter().map(|b| zenpredict::f16_bits_to_f32(*b)).collect(),
        WeightStorage::I8 { weights, scales } => {
            let mut out = Vec::with_capacity(weights.len());
            for (idx, w) in weights.iter().enumerate() {
                let o = idx % 1;
                out.push(*w as f32 * scales[o]);
            }
            out
        }
    };
    let b1 = l1.biases[0];

    (scaler_mean, scaler_scale, w0, b0, w1, b1)
}

fn main() {
    let args = Args::parse();
    let (ca, cb, cc) = parse_coeffs(&args.coeffs);
    eprintln!(
        "concat coefficients: base={ca:.4} s1={cb:.4} s42={cc:.4} (sum={:.4})",
        ca + cb + cc
    );

    let (sm_a, ss_a, w0_a, b0_a, w1_a, b1_a) = load_single_mlp(&args.base);
    let (sm_b, ss_b, w0_b, b0_b, w1_b, b1_b) = load_single_mlp(&args.s1);
    let (sm_c, ss_c, w0_c, b0_c, w1_c, b1_c) = load_single_mlp(&args.s42);

    // Auto-detect input width from the scaler length (= n_inputs).
    let n_in = sm_a.len();
    assert_eq!(sm_b.len(), n_in, "scaler size mismatch: base={n_in}, s1={}", sm_b.len());
    assert_eq!(sm_c.len(), n_in, "scaler size mismatch: base={n_in}, s42={}", sm_c.len());
    eprintln!("concat input width: {n_in} (228=V0_18 basic+peaks, 372=V0_20a +IW)");

    // All three should share the same scaler (trained on the same
    // feature distribution). Verify with a tight tolerance; on mismatch,
    // use the base's scaler — the linear blend is dominated by it anyway.
    let mut warn_scaler = false;
    for i in 0..n_in {
        if (sm_a[i] - sm_b[i]).abs() > 1e-4
            || (sm_a[i] - sm_c[i]).abs() > 1e-4
            || (ss_a[i] - ss_b[i]).abs() > 1e-4
            || (ss_a[i] - ss_c[i]).abs() > 1e-4
        {
            warn_scaler = true;
            break;
        }
    }
    if warn_scaler {
        eprintln!("WARNING: scaler stats differ across sub-MLPs; using base's. Check trainer determinism.");
    }
    let scaler_mean = sm_a;
    let scaler_scale = ss_a;

    // Build layer 0: weights are [W_a | W_b | W_c] along the OUTPUT
    // (hidden) dim. Row-major layout means new[r, c_new] = old[r, c_local].
    let mut w0_concat = vec![0.0f32; n_in * 384];
    let mut b0_concat = vec![0.0f32; 384];
    for r in 0..n_in {
        for c in 0..128 {
            w0_concat[r * 384 + c + 0] = w0_a[r * 128 + c];
            w0_concat[r * 384 + c + 128] = w0_b[r * 128 + c];
            w0_concat[r * 384 + c + 256] = w0_c[r * 128 + c];
        }
    }
    for c in 0..128 {
        b0_concat[c + 0] = b0_a[c];
        b0_concat[c + 128] = b0_b[c];
        b0_concat[c + 256] = b0_c[c];
    }

    // Build layer 1: w1_concat = [ca*w1_a, cb*w1_b, cc*w1_c] as a single 384-vec.
    // Bias: ca*b1_a + cb*b1_b + cc*b1_c (the output of each sub-MLP gets its bias).
    let mut w1_concat = vec![0.0f32; 384];
    for c in 0..128 {
        w1_concat[c + 0] = ca * w1_a[c];
        w1_concat[c + 128] = cb * w1_b[c];
        w1_concat[c + 256] = cc * w1_c[c];
    }
    let b1_concat = vec![ca * b1_a + cb * b1_b + cc * b1_c; 1];

    eprintln!("concat sanity: l0 weights={} l0 biases={} l1 weights={} l1 bias={}",
        w0_concat.len(), b0_concat.len(), w1_concat.len(), b1_concat.len());

    let layers = [
        BakeLayer {
            in_dim: n_in,
            out_dim: 384,
            activation: zenpredict::Activation::LeakyRelu,
            dtype: zenpredict::WeightDtype::F32,
            weights: &w0_concat,
            biases: &b0_concat,
        },
        BakeLayer {
            in_dim: 384,
            out_dim: 1,
            activation: zenpredict::Activation::Identity,
            dtype: zenpredict::WeightDtype::F32,
            weights: &w1_concat,
            biases: &b1_concat,
        },
    ];

    let bytes = bake(&BakeRequest {
        schema_hash: 0,
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
    })
    .expect("ensemble bake");

    fs::write(&args.out, &bytes).expect("write ensemble bake");
    eprintln!("wrote {} ({} bytes)", args.out.display(), bytes.len());
}
