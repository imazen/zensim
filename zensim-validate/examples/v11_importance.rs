use std::fs;
use zenpredict::{Model, WeightStorage};

fn f16_to_f32(_h: u16) -> f32 { 0.0 }

fn main() {
    let bytes = fs::read(
        "/home/lilith/work/zen/zensim/zensim/weights/v_tuner_v11_2026-05-24.bin",
    )
    .unwrap();
    let model = Model::from_bytes(&bytes).unwrap();
    let l0 = model.layer(0);
    let in_dim = l0.in_dim;
    let out_dim = l0.out_dim;
    println!("v11 L0: {in_dim} → {out_dim}, act={:?}", l0.activation);

    let mut sum_abs = vec![0.0_f64; in_dim];
    match &l0.weights {
        WeightStorage::F32(w) => {
            for i in 0..in_dim {
                let base = i * out_dim;
                for o in 0..out_dim {
                    sum_abs[i] += (w[base + o] as f64).abs();
                }
            }
        }
        WeightStorage::F16(w) => {
            for i in 0..in_dim {
                let base = i * out_dim;
                for o in 0..out_dim {
                    sum_abs[i] += (f16_to_f32(w[base + o]) as f64).abs();
                }
            }
        }
        WeightStorage::I8 { weights, scales } => {
            for i in 0..in_dim {
                let base = i * out_dim;
                for o in 0..out_dim {
                    sum_abs[i] += (weights[base + o] as f64 * scales[o] as f64).abs();
                }
            }
        }
    }

    let scaler = model.scaler_scale();
    let mut importance: Vec<(usize, f64)> = (0..in_dim)
        .map(|i| (i, sum_abs[i] * scaler[i] as f64))
        .collect();

    let mut sorted: Vec<f64> = importance.iter().map(|(_, v)| *v).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let q = |p: f64| sorted[(p * (n - 1) as f64).round() as usize];
    println!(
        "importance distribution: min={:.4} p10={:.4} p25={:.4} p50={:.4} p75={:.4} p90={:.4} max={:.4}",
        sorted[0], q(0.10), q(0.25), q(0.50), q(0.75), q(0.90), sorted[n - 1]
    );

    // Count features below various thresholds
    let total_mass: f64 = sorted.iter().sum();
    let thresholds = [0.001, 0.01, 0.05, 0.10];
    for &t in &thresholds {
        let cnt = sorted.iter().filter(|&&v| v < total_mass * t / n as f64).count();
        println!("  features < {:.1}× mean importance: {}", t * 100.0, cnt);
    }
    let mean = total_mass / n as f64;
    for &frac in &[0.01_f64, 0.05, 0.10, 0.50] {
        let cnt = sorted.iter().filter(|&&v| v < mean * frac).count();
        println!(
            "  features < {:.0}% of mean importance ({:.4}): {}",
            frac * 100.0,
            mean * frac,
            cnt
        );
    }

    // Per-block stats: basic (0..156), peaks (156..228), masked (228..300), iw-pool (300..372)
    let blocks = [
        ("basic    f0..155  ", 0usize, 156),
        ("peaks    f156..227", 156, 228),
        ("masked   f228..299", 228, 300),
        ("iw-pool  f300..371", 300, 372),
    ];
    println!("\nper-block L0 importance:");
    println!("  block              n  sum_importance    mean    median    max     %total_mass");
    let total: f64 = sum_abs.iter().zip(scaler.iter()).map(|(&s, &sc)| s * sc as f64).sum();
    for (name, lo, hi) in blocks {
        let vals: Vec<f64> = (lo..hi).map(|i| sum_abs[i] * scaler[i] as f64).collect();
        let sum: f64 = vals.iter().sum();
        let mean = sum / vals.len() as f64;
        let mut sorted_blk = vals.clone();
        sorted_blk.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted_blk[sorted_blk.len() / 2];
        let max = sorted_blk[sorted_blk.len() - 1];
        let pct = 100.0 * sum / total;
        println!(
            "  {name}  {:3}  {:14.2}  {:7.3}  {:8.3}  {:8.2}  {:6.2}%",
            vals.len(),
            sum,
            mean,
            median,
            max,
            pct
        );
    }

    importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\ntop 20 features by importance:");
    for (i, (idx, v)) in importance.iter().take(20).enumerate() {
        println!("  #{:2}: f{idx} = {v:.4}", i + 1);
    }
    println!("\nbottom 20 features by importance:");
    for (i, (idx, v)) in importance.iter().rev().take(20).enumerate() {
        println!("  #{:2}: f{idx} = {v:.6}", i + 1);
    }
}
