//! Dump per-input-feature importance from each shipped zensim bake's L0 layer.
//!
//! Reports `importance[i] = scaler_scale[i] * sum_h(|L0[h, i]|)` per input
//! feature `i`. Used to ablate-check whether max-pool features (the 36
//! `*_max` slots at peak block indices 156..=224 with stride 6) carry
//! significant signal in shipped models.
//!
//! Output: CSV at /tmp/zensim_l0_importance.csv with columns
//!   bake_name, n_inputs, layer_index_used, layer_in_dim, layer_out_dim,
//!   layer_activation, feature_index, scaler_scale, sum_abs_w, importance,
//!   is_maxpool
//!
//! Plus a per-bake summary table on stdout.
//!
//! Run:
//!   cargo run -p zensim-validate --release --example dump_l0_importance

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use zenpredict::{Model, WeightStorage};

const MAXPOOL_BASE: usize = 156;
const MAXPOOL_STRIDE: usize = 6;
const N_SCALES: usize = 4;
const N_CHANNELS: usize = 3;

// Per (scale, channel) the 3 max-pool indices are at offsets 0, 1, 2
// within the 6-slot (scale, channel) cell.
fn is_maxpool_index(idx: usize) -> bool {
    if idx < MAXPOOL_BASE {
        return false;
    }
    let rel = idx - MAXPOOL_BASE;
    let total = N_SCALES * N_CHANNELS * MAXPOOL_STRIDE; // 72
    if rel >= total {
        return false;
    }
    rel % MAXPOOL_STRIDE < 3
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (pos.ceil() as usize).min(n - 1);
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn analyze_layer_weights(layer: &zenpredict::LayerView<'_>) -> Vec<f64> {
    // Per-input sum of |w[h, i]|. Weights are stored row-major
    // input-major: w[i * out_dim + o].
    let in_dim = layer.in_dim;
    let out_dim = layer.out_dim;
    let mut sum_abs = vec![0.0_f64; in_dim];
    match &layer.weights {
        WeightStorage::F32(w) => {
            for i in 0..in_dim {
                let base = i * out_dim;
                let mut s = 0.0_f64;
                for o in 0..out_dim {
                    s += (w[base + o] as f64).abs();
                }
                sum_abs[i] = s;
            }
        }
        WeightStorage::F16(w) => {
            for i in 0..in_dim {
                let base = i * out_dim;
                let mut s = 0.0_f64;
                for o in 0..out_dim {
                    let f = zenpredict::f16_bits_to_f32(w[base + o]);
                    s += (f as f64).abs();
                }
                sum_abs[i] = s;
            }
        }
        WeightStorage::I8 { weights, scales } => {
            for i in 0..in_dim {
                let base = i * out_dim;
                let mut s = 0.0_f64;
                for o in 0..out_dim {
                    let f = weights[base + o] as f32 * scales[o];
                    s += (f as f64).abs();
                }
                sum_abs[i] = s;
            }
        }
    }
    sum_abs
}

// Pick the first layer with in_dim == n_inputs and activation != Identity
// (or just first if all are Identity). Falls back to layer 0.
fn pick_l0_index(model: &Model) -> usize {
    let n_in = model.n_inputs();
    let layers: Vec<_> = model.layers().collect();
    // Prefer the first layer where in_dim matches n_inputs (the "true" L0).
    for (idx, l) in layers.iter().enumerate() {
        if l.in_dim == n_in {
            // If it has identity activation AND out_dim equals n_inputs,
            // it might be a passthrough — skip to next non-passthrough.
            let is_passthrough = l.in_dim == l.out_dim
                && matches!(l.activation, zenpredict::Activation::Identity);
            if !is_passthrough {
                return idx;
            }
        }
    }
    // Fallback: first matching layer regardless.
    for (idx, l) in layers.iter().enumerate() {
        if l.in_dim == n_in {
            return idx;
        }
    }
    0
}

#[derive(Debug)]
struct BakeReport {
    bake_name: String,
    file_size: u64,
    sha256: String,
    n_inputs: usize,
    layer_idx_used: usize,
    layer_in_dim: usize,
    layer_out_dim: usize,
    layer_activation: String,
    mp_min: f64,
    mp_p25: f64,
    mp_median: f64,
    mp_p75: f64,
    mp_max: f64,
    nmp_median: f64,
    ratio: f64,
    verdict: String,
    top5_mp: Vec<(usize, f64)>,
}

fn sha256_file(bytes: &[u8]) -> String {
    use std::process::{Command, Stdio};
    use std::io::Write as _;
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("sha256sum");
    child.stdin.as_mut().unwrap().write_all(bytes).unwrap();
    let out = child.wait_with_output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    s.split_whitespace().next().unwrap_or("").to_string()
}

fn process_bake(path: &PathBuf, csv: &mut impl Write) -> Result<BakeReport, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let file_size = bytes.len() as u64;
    let sha = sha256_file(&bytes);
    let model = Model::from_bytes(&bytes)?;
    let n_inputs = model.n_inputs();
    let scaler_scale = model.scaler_scale().to_vec();

    let l_idx = pick_l0_index(&model);
    let layer = model.layer(l_idx);
    let sum_abs_w = analyze_layer_weights(&layer);
    // If the picked layer's in_dim != n_inputs, scale alignment is moot —
    // but we still report sum_abs_w. Importance uses scaler_scale only when
    // sizes match.
    let layer_in_dim = layer.in_dim;
    let layer_out_dim = layer.out_dim;
    let activation = format!("{:?}", layer.activation);
    let aligned = layer_in_dim == n_inputs;

    let mut importance = vec![0.0_f64; layer_in_dim];
    for i in 0..layer_in_dim {
        let s = if aligned && i < scaler_scale.len() {
            scaler_scale[i] as f64
        } else {
            1.0
        };
        importance[i] = s * sum_abs_w[i];
    }

    // For max-pool analysis we need 372-input layout. If the layer is
    // smaller (e.g. some bakes use a feature subset before L0), only
    // count max-pool indices that fall inside [0, layer_in_dim).
    let mut mp_imps: Vec<f64> = Vec::new();
    let mut nmp_imps: Vec<f64> = Vec::new();
    let mut mp_with_idx: Vec<(usize, f64)> = Vec::new();
    for i in 0..layer_in_dim {
        if is_maxpool_index(i) {
            mp_imps.push(importance[i]);
            mp_with_idx.push((i, importance[i]));
        } else {
            nmp_imps.push(importance[i]);
        }
        let scaler_s = if aligned && i < scaler_scale.len() {
            scaler_scale[i]
        } else {
            f32::NAN
        };
        writeln!(
            csv,
            "{},{},{},{},{},{:?},{},{},{},{},{}",
            path.file_name().unwrap().to_string_lossy(),
            n_inputs,
            l_idx,
            layer_in_dim,
            layer_out_dim,
            layer.activation,
            i,
            scaler_s,
            sum_abs_w[i],
            importance[i],
            is_maxpool_index(i) as u8
        )?;
    }

    let mut mp_sorted = mp_imps.clone();
    mp_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut nmp_sorted = nmp_imps.clone();
    nmp_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mp_median = quantile(&mp_sorted, 0.5);
    let nmp_median = quantile(&nmp_sorted, 0.5);
    let ratio = if nmp_median > 0.0 { mp_median / nmp_median } else { f64::NAN };

    let verdict = if ratio.is_nan() {
        "no-data".to_string()
    } else if ratio < 0.1 {
        "drop".to_string()
    } else if ratio < 0.5 {
        "marginal".to_string()
    } else {
        "load-bearing".to_string()
    };

    mp_with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top5 = mp_with_idx.iter().take(5).cloned().collect();

    Ok(BakeReport {
        bake_name: path.file_name().unwrap().to_string_lossy().to_string(),
        file_size,
        sha256: sha,
        n_inputs,
        layer_idx_used: l_idx,
        layer_in_dim,
        layer_out_dim,
        layer_activation: activation,
        mp_min: quantile(&mp_sorted, 0.0),
        mp_p25: quantile(&mp_sorted, 0.25),
        mp_median,
        mp_p75: quantile(&mp_sorted, 0.75),
        mp_max: quantile(&mp_sorted, 1.0),
        nmp_median,
        ratio,
        verdict,
        top5_mp: top5,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let weights_dir = PathBuf::from("/home/lilith/work/zen/zensim/zensim/weights");
    let archive_dir = weights_dir.join("archive");

    // Bakes referenced as include_bytes! in profile.rs (shipped):
    let shipped = [
        "v0_18_zerobiased_lz4_2026-05-13.bin",
        "v0_20_is_calibrated_2026-05-15.bin",
        "v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin",
        "v_compression_persample_2026-05-18.bin",
        "v05_ensemble_classifier_2026-05-18.bin",
        "v_tuner_2026-05-18.bin",
        "v_cross_codec_2026-05-19.bin",
        "v_tuner_v6_2026-05-19.bin",
        "v_tuner_v9_2026-05-20.bin",
        "v_balanced_v2_2026-05-20.bin",
        "v_compression_v2_2026-05-20.bin",
        "v_balanced_v3_2026-05-20.bin",
        "v_compression_v3_2026-05-20.bin",
        "v_tuner_v10_2026-05-20.bin",
        "v_tuner_v4_per_codec_2026-05-20.bin",
        "v_balanced_v3_per_codec_2026-05-20.bin",
        "v_compression_v3_per_codec_2026-05-20.bin",
    ];

    let csv_path = "/tmp/zensim_l0_importance.csv";
    let mut csv = fs::File::create(csv_path)?;
    writeln!(
        csv,
        "bake_name,n_inputs,layer_idx,layer_in_dim,layer_out_dim,activation,feature_index,scaler_scale,sum_abs_w,importance,is_maxpool"
    )?;

    let mut reports: Vec<BakeReport> = Vec::new();
    for name in shipped.iter() {
        let path = weights_dir.join(name);
        if !path.exists() {
            eprintln!("MISSING (shipped): {}", path.display());
            continue;
        }
        eprintln!("processing {}", path.display());
        match process_bake(&path, &mut csv) {
            Ok(r) => reports.push(r),
            Err(e) => eprintln!("  ERROR: {e}"),
        }
    }

    // Also examine a few historical archive bakes for context (read-only).
    let archive_bakes = [
        "v0_4_2026-04-30.bin",
        "v0_16_2026-05-12.bin",
        "v0_18_2026-05-13_ship.bin",
    ];
    for name in archive_bakes.iter() {
        let path = archive_dir.join(name);
        if !path.exists() {
            continue;
        }
        eprintln!("processing (archive) {}", path.display());
        match process_bake(&path, &mut csv) {
            Ok(r) => reports.push(r),
            Err(e) => eprintln!("  ERROR: {e}"),
        }
    }

    drop(csv);

    // Markdown summary printed to stdout.
    println!("\n# zensim L0 Max-Pool Importance — Shipped Bakes\n");
    println!("CSV written: {csv_path}\n");
    println!("| Bake | n_in | L0 dims | act | sha256(8) | size | mp_med | nmp_med | ratio | verdict |");
    println!("|---|---:|---|---|---|---:|---:|---:|---:|---|");
    for r in &reports {
        println!(
            "| `{}` | {} | {}×{} | {} | `{}` | {} | {:.4} | {:.4} | {:.3} | **{}** |",
            r.bake_name,
            r.n_inputs,
            r.layer_in_dim,
            r.layer_out_dim,
            r.layer_activation,
            &r.sha256[..r.sha256.len().min(8)],
            r.file_size,
            r.mp_median,
            r.nmp_median,
            r.ratio,
            r.verdict,
        );
    }

    println!("\n## Max-pool importance distribution per bake\n");
    println!("| Bake | min | p25 | median | p75 | max |");
    println!("|---|---:|---:|---:|---:|---:|");
    for r in &reports {
        println!(
            "| `{}` | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            r.bake_name, r.mp_min, r.mp_p25, r.mp_median, r.mp_p75, r.mp_max
        );
    }

    println!("\n## Top-5 max-pool indices (highest importance) per bake\n");
    for r in &reports {
        let s = r
            .top5_mp
            .iter()
            .map(|(i, v)| format!("f{i}={v:.4}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("- `{}`: {}", r.bake_name, s);
    }

    Ok(())
}
