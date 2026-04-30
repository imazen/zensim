//! Per-feature importance analysis for a trained V0_4 MLP.
//!
//! Loads a v1 ZNPK bake and produces three importance signals per
//! input dimension:
//!
//! 1. **Architecture importance** — `sum_j |W1[i, j] * W2[j]|`. Weight-only,
//!    captures the model's "wired-in" sensitivity to feature `i`.
//!    Independent of data; bounded above by `|dy/dx_i|` over standardized
//!    inputs (since LeakyReLU(α=0.01) derivatives lie in [α, 1]).
//!
//! 2. **Zero-out impact (data-free)** — sample `N` standardized inputs
//!    from N(0, I), forward-pass each, then for each feature `i` set
//!    `x[i] = 0`, measure `RMS(y_orig - y_zeroed)`.
//!
//! 3. **Gradient×Input (data-conditional)** — same N samples; per
//!    sample compute `g_i = ∂y/∂x_i` analytically and aggregate
//!    `mean_n |g_i × x_n_i|`.
//!
//! Output: a markdown report sorted by zero-out impact, plus disagreement
//! summary vs `WEIGHTS_PREVIEW_V0_2`.

use std::path::PathBuf;

use clap::Parser;
use zensim::mlp::{Activation, Model, WeightStorage};
use zensim::profile::WEIGHTS_PREVIEW_V0_2;

#[derive(Parser, Debug)]
#[command(name = "mlp_importance")]
struct Args {
    /// Path to a v1 ZNPK MLP bake.
    #[arg(long)]
    bake: PathBuf,

    /// Number of synthetic standardized samples to draw.
    #[arg(long, default_value = "5000")]
    n_samples: usize,

    /// Random seed.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Print this many top + bottom features.
    #[arg(long, default_value = "20")]
    top_k: usize,
}

struct Row {
    idx: usize,
    name: String,
    v02: f64,
    arch: f64,
    zeroout: f64,
    grad_x_input: f64,
}

fn main() {
    let args = Args::parse();

    let bytes = std::fs::read(&args.bake).expect("read bake");
    let model = Model::from_bytes(&bytes).expect("parse bake");

    let n_inputs = model.n_inputs();
    let n_outputs = model.n_outputs();
    let layers = model.layers();
    if layers.len() != 2 || n_outputs != 1 {
        eprintln!(
            "Expected 2-layer single-output MLP (got {} layers, {} outputs).",
            layers.len(),
            n_outputs
        );
        std::process::exit(1);
    }

    let n_hidden = layers[0].out_dim;
    let alpha = 0.01f64;

    let w1 = match &layers[0].weights {
        WeightStorage::F32(w) => w,
        _ => {
            eprintln!("Expected f32 weights on layer 0");
            std::process::exit(1);
        }
    };
    let b1: Vec<f64> = layers[0].biases.iter().map(|&v| v as f64).collect();
    let w2 = match &layers[1].weights {
        WeightStorage::F32(w) => w,
        _ => {
            eprintln!("Expected f32 weights on layer 1");
            std::process::exit(1);
        }
    };
    let _b2: f64 = layers[1].biases[0] as f64;

    let w1_f64: Vec<f64> = w1.iter().map(|&v| v as f64).collect();
    let w2_f64: Vec<f64> = w2.iter().map(|&v| v as f64).collect();

    let activation_is_leaky = layers[0].activation == Activation::LeakyRelu;
    let activation_is_relu = layers[0].activation == Activation::Relu;

    eprintln!(
        "Loaded MLP: n_inputs={n_inputs}, n_hidden={n_hidden}, layer0_act={:?}, layer1_act={:?}",
        layers[0].activation, layers[1].activation
    );

    // 1. Architecture importance.
    let mut arch_importance = vec![0.0f64; n_inputs];
    for i in 0..n_inputs {
        let mut s = 0.0f64;
        for j in 0..n_hidden {
            s += w1_f64[i * n_hidden + j].abs() * w2_f64[j].abs();
        }
        arch_importance[i] = s;
    }

    // Synthetic samples in standardized input space.
    let mut rng = SplitMix64::new(args.seed);
    let xs: Vec<Vec<f64>> = (0..args.n_samples)
        .map(|_| (0..n_inputs).map(|_| rng.next_normal()).collect())
        .collect();

    // Cache forward-pass intermediates per sample.
    struct Cached {
        h_pre: Vec<f64>,
        h: Vec<f64>,
    }
    let cached: Vec<Cached> = xs
        .iter()
        .map(|x| {
            let mut h_pre = b1.clone();
            for i in 0..n_inputs {
                let s = x[i];
                if s == 0.0 {
                    continue;
                }
                let row = &w1_f64[i * n_hidden..(i + 1) * n_hidden];
                for (acc, &w) in h_pre.iter_mut().zip(row.iter()) {
                    *acc += s * w;
                }
            }
            let h: Vec<f64> = h_pre
                .iter()
                .map(|&v| activate(v, activation_is_leaky, activation_is_relu, alpha))
                .collect();
            Cached { h_pre, h }
        })
        .collect();

    // 2. Zero-out impact.
    let mut zeroout = vec![0.0f64; n_inputs];
    for i in 0..n_inputs {
        let mut sumsq = 0.0f64;
        for (sample_idx, x) in xs.iter().enumerate() {
            let xi = x[i];
            if xi == 0.0 {
                continue;
            }
            let cache = &cached[sample_idx];
            let row = &w1_f64[i * n_hidden..(i + 1) * n_hidden];
            let mut delta_y = 0.0f64;
            for j in 0..n_hidden {
                let h_pre_zero = cache.h_pre[j] - xi * row[j];
                let h_zero = activate(h_pre_zero, activation_is_leaky, activation_is_relu, alpha);
                delta_y += w2_f64[j] * (h_zero - cache.h[j]);
            }
            sumsq += delta_y * delta_y;
        }
        zeroout[i] = (sumsq / args.n_samples as f64).sqrt();
    }

    // 3. Gradient × input.
    let mut grad_x_input = vec![0.0f64; n_inputs];
    for x in xs.iter() {
        let cache = &cached[xs.iter().position(|y| std::ptr::eq(y, x)).unwrap_or(0)];
        let dmask: Vec<f64> = cache
            .h_pre
            .iter()
            .map(|&v| activation_derivative(v, activation_is_leaky, activation_is_relu, alpha))
            .collect();
        let layer2_weighted: Vec<f64> =
            (0..n_hidden).map(|j| w2_f64[j] * dmask[j]).collect();
        for i in 0..n_inputs {
            let row = &w1_f64[i * n_hidden..(i + 1) * n_hidden];
            let mut g = 0.0f64;
            for j in 0..n_hidden {
                g += layer2_weighted[j] * row[j];
            }
            grad_x_input[i] += (g * x[i]).abs();
        }
    }
    for v in &mut grad_x_input {
        *v /= args.n_samples as f64;
    }

    let v02_weights: &[f64; 228] = &WEIGHTS_PREVIEW_V0_2;

    let rows: Vec<Row> = (0..n_inputs)
        .map(|i| Row {
            idx: i,
            name: feature_name(i, n_inputs),
            v02: if i < v02_weights.len() {
                v02_weights[i].abs()
            } else {
                0.0
            },
            arch: arch_importance[i],
            zeroout: zeroout[i],
            grad_x_input: grad_x_input[i],
        })
        .collect();

    println!("# V0_4 MLP feature importance");
    println!();
    println!("- Bake: `{}`", args.bake.display());
    println!("- Architecture: {} → {} (LeakyReLU) → 1", n_inputs, n_hidden);
    println!("- Synthetic samples: {}", args.n_samples);
    println!();

    println!("## Top {} features by zero-out impact", args.top_k);
    println!();
    print_table(&rows, |r| r.zeroout, args.top_k, true);
    println!();

    println!("## Bottom {} features by zero-out impact (prune candidates)", args.top_k);
    println!();
    print_table(&rows, |r| r.zeroout, args.top_k, false);
    println!();

    let zo_max = rows.iter().map(|r| r.zeroout).fold(0.0_f64, f64::max).max(1e-12);

    let v02_q3 = quantile(rows.iter().map(|r| r.v02).collect(), 0.75);
    let zo_q1 = quantile(rows.iter().map(|r| r.zeroout).collect(), 0.25);
    let zo_q3 = quantile(rows.iter().map(|r| r.zeroout).collect(), 0.75);

    println!("## Where V0_4 and V0_2 disagree");
    println!();
    println!(
        "Quartile thresholds — V0_2 |W| Q3 = {:.3}, V0_4 zero-out Q1 = {:.4}, Q3 = {:.4}.",
        v02_q3, zo_q1, zo_q3
    );
    println!();
    println!("### Features V0_2 weighted heavily, V0_4 ignores");
    println!("(V0_2 in top quartile, V0_4 zero-out in bottom quartile.)");
    println!();
    let mut v02_only: Vec<&Row> = rows
        .iter()
        .filter(|r| r.v02 > v02_q3 && r.zeroout < zo_q1)
        .collect();
    v02_only.sort_by(|a, b| b.v02.partial_cmp(&a.v02).unwrap());
    print_subset(&v02_only);
    println!();

    println!("### Features V0_4 picked up that V0_2 was zero on");
    println!("(V0_2 weight = 0, V0_4 zero-out in top quartile.)");
    println!();
    let mut v04_only: Vec<&Row> = rows
        .iter()
        .filter(|r| r.v02 == 0.0 && r.zeroout > zo_q3)
        .collect();
    v04_only.sort_by(|a, b| b.zeroout.partial_cmp(&a.zeroout).unwrap());
    print_subset(&v04_only);
    println!();

    let n_v02_zero = rows.iter().filter(|r| r.v02 == 0.0).count();
    let prune_threshold = 0.001 * zo_max;
    let n_v04_negligible = rows.iter().filter(|r| r.zeroout < prune_threshold).count();
    let agreement = rows
        .iter()
        .filter(|r| r.v02 == 0.0 && r.zeroout < prune_threshold)
        .count();
    println!("## Summary");
    println!();
    println!("- V0_2 features at exactly 0 weight: {}/228", n_v02_zero);
    println!(
        "- V0_4 features below 0.1% of max zero-out impact: {}/{}",
        n_v04_negligible, n_inputs
    );
    println!(
        "- Both V0_2 and V0_4 agree feature is irrelevant: {} (V0_5 prune-safe set)",
        agreement
    );
}

fn activate(v: f64, leaky: bool, relu: bool, alpha: f64) -> f64 {
    if leaky {
        if v >= 0.0 { v } else { alpha * v }
    } else if relu {
        v.max(0.0)
    } else {
        v
    }
}

fn activation_derivative(v: f64, leaky: bool, relu: bool, alpha: f64) -> f64 {
    if leaky {
        if v >= 0.0 { 1.0 } else { alpha }
    } else if relu {
        if v >= 0.0 { 1.0 } else { 0.0 }
    } else {
        1.0
    }
}

fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return 0.0;
    }
    let pos = (v.len() as f64 - 1.0) * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        v[lo] + (pos - lo as f64) * (v[hi] - v[lo])
    }
}

fn print_table(rows: &[Row], key: impl Fn(&Row) -> f64, k: usize, top: bool) {
    let mut indexed: Vec<usize> = (0..rows.len()).collect();
    indexed.sort_by(|&a, &b| {
        let ka = key(&rows[a]);
        let kb = key(&rows[b]);
        if top { kb.partial_cmp(&ka).unwrap() } else { ka.partial_cmp(&kb).unwrap() }
    });
    println!("| idx | feature | V0_2 |W| | V0_4 arch | V0_4 zero-out | V0_4 grad×x |");
    println!("|----:|---------|----:|----:|----:|----:|");
    for &i in indexed.iter().take(k) {
        let r = &rows[i];
        println!(
            "| {} | `{}` | {:.3} | {:.3} | {:.4} | {:.4} |",
            r.idx, r.name, r.v02, r.arch, r.zeroout, r.grad_x_input,
        );
    }
}

fn print_subset(rows: &[&Row]) {
    if rows.is_empty() {
        println!("(none)");
        return;
    }
    println!("| idx | feature | V0_2 |W| | V0_4 arch | V0_4 zero-out | V0_4 grad×x |");
    println!("|----:|---------|----:|----:|----:|----:|");
    for r in rows {
        println!(
            "| {} | `{}` | {:.3} | {:.3} | {:.4} | {:.4} |",
            r.idx, r.name, r.v02, r.arch, r.zeroout, r.grad_x_input,
        );
    }
}

fn feature_name(i: usize, n_inputs: usize) -> String {
    if n_inputs >= 232 && i >= 228 {
        let names = ["log2_pixels", "log2_min_dim", "log2_max_dim", "signed_log2_aspect"];
        return format!("size.{}", names[(i - 228).min(3)]);
    }
    if i < 156 {
        let scale = i / (3 * 13);
        let channel = (i % (3 * 13)) / 13;
        let feat = i % 13;
        let feat_names = [
            "ssim_mean", "ssim_4th", "ssim_2nd",
            "art_mean", "art_4th", "art_2nd",
            "det_mean", "det_4th", "det_2nd",
            "mse",
            "hf_energy_loss", "hf_mag_loss", "hf_energy_gain",
        ];
        let chan_names = ["X", "Y", "B"];
        format!(
            "s{}.{}.{}",
            scale,
            chan_names[channel.min(2)],
            feat_names[feat.min(12)]
        )
    } else if i < 228 {
        let p = i - 156;
        let scale = p / (3 * 6);
        let channel = (p % (3 * 6)) / 6;
        let feat = p % 6;
        let peak_names = ["ssim_max", "art_max", "det_max", "ssim_p95", "art_p95", "det_p95"];
        let chan_names = ["X", "Y", "B"];
        format!(
            "s{}.{}.{}",
            scale,
            chan_names[channel.min(2)],
            peak_names[feat.min(5)]
        )
    } else {
        format!("idx{}", i)
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64_unit().max(1e-12);
        let u2 = self.next_f64_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
