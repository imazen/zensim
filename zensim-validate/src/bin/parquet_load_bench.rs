//! One-shot benchmark for the trainer's Parquet loader variants.
//!
//! Times one load in a fresh process. `flat64` is the trainer's ordinary
//! Parquet path; `compact32` is its eligible `--keep-features` path.
//!
//! Usage:
//!     cargo run --release -p zensim-validate --bin parquet_load_bench -- \
//!         <parquet_path> [<target_column>] [<target_scale>] [rows|flat64|compact32]
//!         [selected_feature_ids_csv] [max_feature_width]
//!
//! Defaults: target_column = "iwssim_log_norm", target_scale = 1.0, mode =
//! `rows` (the historical benchmark). `compact32` requires sorted, unique
//! selected feature IDs; it errors rather than silently falling back.

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: parquet_load_bench <path> [<target_column>] [<target_scale>]");
    let target = args.get(2).map(String::as_str).unwrap_or("iwssim_log_norm");
    let scale: f64 = args
        .get(3)
        .map(|s| s.parse::<f64>().expect("target_scale must be f64"))
        .unwrap_or(1.0);
    let mode = args.get(4).map(String::as_str).unwrap_or("rows");
    let compact = if mode == "compact32" {
        let selected = args
            .get(5)
            .expect("compact32 requires selected feature IDs");
        let ids: Vec<u32> = selected
            .split(',')
            .map(|id| id.parse().expect("selected feature ID must be u32"))
            .collect();
        assert!(
            !ids.is_empty() && ids.windows(2).all(|pair| pair[0] < pair[1]),
            "selected IDs must be nonempty, sorted and unique"
        );
        let max_width = args
            .get(6)
            .map(|s| s.parse::<usize>().expect("max_feature_width must be usize"))
            .unwrap_or(usize::MAX);
        Some((ids, max_width))
    } else {
        None
    };

    let pb = PathBuf::from(path.as_str());
    let bytes = std::fs::metadata(&pb)
        .unwrap_or_else(|e| panic!("stat {path}: {e}"))
        .len();

    let t0 = Instant::now();
    let (rows, features, stored_values, dt) = match mode {
        "rows" => {
            let g = zensim_validate::parquet_loader::load_parquet(&pb, "bench", target, scale)
                .unwrap_or_else(|e| panic!("load_parquet failed: {e}"));
            let dt = t0.elapsed();
            let shape = (
                g.human_scores.len(),
                g.n_features,
                g.feature_rows.iter().map(Vec::len).sum::<usize>(),
                dt,
            );
            std::hint::black_box(&g);
            shape
        }
        "flat64" => {
            let g = zensim_validate::parquet_loader::load_parquet_flat(&pb, "bench", target, scale)
                .unwrap_or_else(|e| panic!("load_parquet_flat failed: {e}"));
            let shape = (
                g.human_scores.len(),
                g.n_features,
                g.features_flat.len(),
                t0.elapsed(),
            );
            std::hint::black_box(&g);
            shape
        }
        "compact32" => {
            let (ids, max_width) = compact.as_ref().unwrap();
            let g = zensim_validate::parquet_loader::load_parquet_flat_f32(
                &pb, "bench", target, scale, ids, *max_width,
            )
            .unwrap_or_else(|e| panic!("load_parquet_flat_f32 failed: {e}"))
            .expect("compact32 declined this table; the trainer would use flat64");
            let shape = (
                g.human_scores.len(),
                g.n_features,
                g.data.len(),
                t0.elapsed(),
            );
            std::hint::black_box(&g);
            shape
        }
        _ => panic!("mode must be rows, flat64 or compact32"),
    };

    println!(
        "{}",
        serde_json::json!({
            "mode": mode,
            "rows": rows,
            "logical_features": features,
            "stored_values": stored_values,
            "file_bytes": bytes,
            "load_ns": dt.as_nanos(),
        })
    );
}
