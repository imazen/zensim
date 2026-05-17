//! One-shot loader benchmark for `parquet_loader::load_parquet`.
//!
//! Times a single full load of a parquet feature file and prints rate
//! / size. Compare against the CSV loader's wall time on the same
//! source to confirm the parquet path is materially faster.
//!
//! Usage:
//!     cargo run --release -p zensim-validate --bin parquet_load_bench -- \
//!         <parquet_path> [<target_column>] [<target_scale>]
//!
//! Defaults: target_column = "iwssim_log_norm", target_scale = 1.0.

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: parquet_load_bench <path> [<target_column>] [<target_scale>]");
    let target = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "iwssim_log_norm".to_string());
    let scale: f64 = std::env::args()
        .nth(3)
        .map(|s| s.parse::<f64>().expect("target_scale must be f64"))
        .unwrap_or(1.0);

    let pb = PathBuf::from(&path);
    let bytes = std::fs::metadata(&pb)
        .unwrap_or_else(|e| panic!("stat {path}: {e}"))
        .len();

    let t0 = Instant::now();
    let g = zensim_validate::parquet_loader::load_parquet(&pb, "bench", &target, scale)
        .unwrap_or_else(|e| panic!("load_parquet failed: {e}"));
    let dt = t0.elapsed();

    println!(
        "loaded {} rows x {} features in {:.2}s ({:.1} MB/s, file {:.1} MB)",
        g.human_scores.len(),
        g.n_features,
        dt.as_secs_f64(),
        bytes as f64 / 1e6 / dt.as_secs_f64(),
        bytes as f64 / 1e6
    );
}
