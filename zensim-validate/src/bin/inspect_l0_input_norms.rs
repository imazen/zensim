//! Per-input column L2 norm inspector for a ZNPR v3 bake.
//!
//! For each input column `i` of layer 0, computes
//!   ‖w0[i, :]‖₂ = sqrt(sum_j w0[i, j]^2)
//! and reports:
//!   - The full per-input norm vector
//!   - Top-K and bottom-K inputs by norm
//!   - "Dead" input count (norm < 0.01 * max_norm) — these features the
//!     trainer chose to ignore via gradient descent
//!
//! This is the diagnostic that answers "did GD select the IW features"
//! for 372-input bakes, "did GD select the masked features" for 300-
//! input bakes, etc. A near-zero norm means the column's contribution
//! is effectively masked out post-training.
//!
//! ## Usage
//!
//! ```
//! cargo run --release -p zensim-validate --bin inspect_l0_input_norms -- \
//!   --bake benchmarks/v0_20_extended_seed1_2026-05-15.bin \
//!   [--top 20] [--regions]
//! ```
//!
//! When `--regions` is passed, also reports aggregate norms per
//! feature block per the standard zensim layout:
//!   [0..156)   basic (13/ch × 3ch × 4 scales)
//!   [156..228) peaks (6/ch × 3ch × 4 scales)
//!   [228..300) masked (6/ch × 3ch × 4 scales, extended)
//!   [300..372) IW (6/ch × 3ch × 4 scales)

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::{Model, WeightStorage};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let mut bake: Option<PathBuf> = None;
    let mut top: usize = 20;
    let mut regions = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bake" => bake = Some(args.next().expect("--bake PATH").into()),
            "--top" => top = args.next().expect("--top N").parse().expect("usize"),
            "--regions" => regions = true,
            other => {
                eprintln!("unknown arg: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let bake = match bake {
        Some(p) => p,
        None => {
            eprintln!("usage: inspect_l0_input_norms --bake PATH [--top N] [--regions]");
            return ExitCode::from(2);
        }
    };

    let bytes = match fs::read(&bake) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read {}: {e}", bake.display());
            return ExitCode::from(1);
        }
    };
    let model = match Model::from_bytes(&bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to parse bake {}: {e:?}", bake.display());
            return ExitCode::from(1);
        }
    };

    let n_in = model.n_inputs();
    let layer0 = match model.layers().next() {
        Some(l) => l,
        None => {
            eprintln!("bake has no layers");
            return ExitCode::from(1);
        }
    };
    let out_dim = layer0.out_dim;
    let in_dim = layer0.in_dim;
    if in_dim != n_in {
        eprintln!(
            "warning: layer0.in_dim ({in_dim}) != model.n_inputs() ({n_in}); using layer0.in_dim"
        );
    }

    // Extract layer 0 weights as f32, regardless of storage dtype.
    let w0_f32: Vec<f32> = match &layer0.weights {
        WeightStorage::F32(w) => w.to_vec(),
        WeightStorage::F16(w) => w.iter().map(|b| zenpredict::f16_bits_to_f32(*b)).collect(),
        WeightStorage::I8 { weights, scales } => weights
            .iter()
            .enumerate()
            .map(|(idx, &q)| (q as f32) * scales[idx % out_dim])
            .collect(),
    };
    assert_eq!(
        w0_f32.len(),
        in_dim * out_dim,
        "layer0 weight count {} != in_dim {} × out_dim {}",
        w0_f32.len(),
        in_dim,
        out_dim
    );

    // Compute per-input L2 norm. Layout: row-major (in_dim × out_dim),
    // so input i's row spans w0_f32[i * out_dim .. (i + 1) * out_dim].
    let mut norms: Vec<f32> = Vec::with_capacity(in_dim);
    for i in 0..in_dim {
        let row = &w0_f32[i * out_dim..(i + 1) * out_dim];
        let sq: f32 = row.iter().map(|w| w * w).sum();
        norms.push(sq.sqrt());
    }
    let max_norm = norms.iter().cloned().fold(0.0f32, f32::max);
    let dead_threshold = 0.01_f32 * max_norm;
    let dead_count = norms.iter().filter(|&&n| n < dead_threshold).count();

    println!("# L0 input-column norms — {}", bake.display());
    println!();
    println!(
        "bake: n_inputs={n_in}, layer0 in×out = {in_dim}×{out_dim}, dtype={:?}",
        match &layer0.weights {
            WeightStorage::F32(_) => "F32",
            WeightStorage::F16(_) => "F16",
            WeightStorage::I8 { .. } => "I8",
        }
    );
    println!();

    println!("## Summary");
    println!();
    println!("- max L2 norm: {max_norm:.4}");
    println!(
        "- dead inputs (norm < 1% of max, < {dead_threshold:.4}): {dead_count}/{in_dim} ({:.1} %)",
        100.0 * dead_count as f32 / in_dim as f32
    );
    println!(
        "- live inputs: {}/{in_dim}",
        in_dim - dead_count
    );

    // Top-K and bottom-K.
    let mut idx_sorted: Vec<usize> = (0..in_dim).collect();
    idx_sorted.sort_by(|&a, &b| norms[b].partial_cmp(&norms[a]).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!("## Top-{top} inputs by L2 norm");
    println!();
    println!("| idx | norm | rel-to-max |");
    println!("|---:|---:|---:|");
    for &i in idx_sorted.iter().take(top) {
        let n = norms[i];
        println!(
            "| {i} | {:.4} | {:.3} |",
            n,
            if max_norm > 0.0 { n / max_norm } else { 0.0 }
        );
    }

    println!();
    println!("## Bottom-{top} inputs by L2 norm");
    println!();
    println!("| idx | norm | rel-to-max |");
    println!("|---:|---:|---:|");
    for &i in idx_sorted.iter().rev().take(top) {
        let n = norms[i];
        println!(
            "| {i} | {:.4} | {:.3} |",
            n,
            if max_norm > 0.0 { n / max_norm } else { 0.0 }
        );
    }

    if regions {
        println!();
        println!("## Region-level aggregates (zensim standard layout)");
        println!();
        let blocks: &[(&str, usize, usize)] = &[
            ("basic  [0..156)", 0, 156),
            ("peaks  [156..228)", 156, 228),
            ("masked [228..300)", 228, 300),
            ("IW     [300..372)", 300, 372),
        ];
        println!("| Block | n | mean L2 | median L2 | max L2 | dead frac |");
        println!("|---|---:|---:|---:|---:|---:|");
        for (name, lo, hi) in blocks {
            let lo = *lo;
            let hi = (*hi).min(in_dim);
            if lo >= in_dim {
                continue;
            }
            let slice = &norms[lo..hi];
            let n = slice.len();
            if n == 0 {
                continue;
            }
            let mean: f32 = slice.iter().sum::<f32>() / n as f32;
            let mut sorted = slice.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted[n / 2];
            let max = sorted[n - 1];
            let dead = slice.iter().filter(|&&v| v < dead_threshold).count();
            println!(
                "| {name} | {n} | {mean:.4} | {median:.4} | {max:.4} | {:.1} % |",
                100.0 * dead as f32 / n as f32
            );
        }
    }

    ExitCode::SUCCESS
}
