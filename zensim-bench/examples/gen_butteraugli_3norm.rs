//! Generate butteraugli 3-norm scores for a training CSV.
//!
//! Reads a CSV with `source_path,decoded_path` columns, computes butteraugli
//! 3-norm using the libjxl-style averaged formula
//!
//!   pnorm(p) = ((Σdᵖ/n)^(1/p) + (Σd^(2p)/n)^(1/(2p)) + (Σd^(4p)/n)^(1/(4p))) / 3
//!
//! and writes the same rows plus an appended `butteraugli_3norm` column.
//!
//! Used for the V0_4 retrain on butteraugli 3-norm target. The synthetic
//! training dataset has `cpu_butteraugli` = max-norm `.score` field; this
//! binary fills in the 3-norm column trainers can target.
//!
//! Usage:
//!   cargo run --release -p zensim-bench --example gen_butteraugli_3norm -- \
//!     --input  /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv \
//!     --output /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_3norm.csv

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use butteraugli::ButteraugliParams;
use imgref::Img;
use rayon::prelude::*;
use rgb::RGB8;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut max_pairs: usize = usize::MAX;
    let mut source_col_name: String = "source_path".to_string();
    let mut decoded_col_name: String = "decoded_path".to_string();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--input" => input = Some(args.next().unwrap().into()),
            "--output" => output = Some(args.next().unwrap().into()),
            "--max-pairs" => max_pairs = args.next().unwrap().parse().unwrap(),
            "--source-col" => source_col_name = args.next().unwrap(),
            "--decoded-col" => decoded_col_name = args.next().unwrap(),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let input = input.expect("--input is required");
    let output = output.expect("--output is required");

    // Read the CSV in full so we can index by row.
    let mut rdr = csv::Reader::from_path(&input).expect("open input CSV");
    let headers = rdr.headers().expect("read headers").clone();
    let src_idx = headers
        .iter()
        .position(|h| h == source_col_name)
        .unwrap_or_else(|| panic!("missing column {source_col_name} in {}", input.display()));
    let dst_idx = headers
        .iter()
        .position(|h| h == decoded_col_name)
        .unwrap_or_else(|| panic!("missing column {decoded_col_name} in {}", input.display()));
    let three_norm_col = headers.iter().position(|h| h == "butteraugli_3norm");
    let mut rows: Vec<Vec<String>> = Vec::new();
    for record in rdr.records().flatten() {
        rows.push(record.iter().map(|s| s.to_string()).collect());
        if rows.len() >= max_pairs {
            break;
        }
    }
    let n = rows.len();
    eprintln!("loaded {n} rows from {}", input.display());

    let started = std::time::Instant::now();
    let progress = AtomicUsize::new(0);
    let log_every = (n / 50).max(1);
    let scores: Vec<Option<f64>> = rows
        .par_iter()
        .map(|row| {
            let p = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if p.is_multiple_of(log_every) {
                let elapsed = started.elapsed().as_secs_f64();
                let rate = p as f64 / elapsed;
                let eta = (n - p) as f64 / rate;
                eprintln!("  {p}/{n} ({rate:.1}/s, ETA {eta:.0}s)");
            }
            let src_path = &row[src_idx];
            let dst_path = &row[dst_idx];
            compute_butter_3norm(src_path, dst_path)
        })
        .collect();
    let n_valid = scores.iter().filter(|s| s.is_some()).count();
    eprintln!(
        "computed {n_valid}/{n} valid 3-norm scores in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    // Write output CSV — preserve headers + add butteraugli_3norm.
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut wtr = csv::Writer::from_path(&output).expect("open output CSV");
    let mut out_headers: Vec<String> = headers.iter().map(|s| s.to_string()).collect();
    if three_norm_col.is_none() {
        out_headers.push("butteraugli_3norm".to_string());
    }
    wtr.write_record(&out_headers).expect("write headers");
    for (row, score) in rows.iter().zip(&scores) {
        let mut out_row: Vec<String> = row.clone();
        let score_str = match score {
            Some(v) => format!("{v:.6}"),
            None => "".to_string(),
        };
        if let Some(idx) = three_norm_col {
            if out_row.len() > idx {
                out_row[idx] = score_str;
            } else {
                out_row.push(score_str);
            }
        } else {
            out_row.push(score_str);
        }
        wtr.write_record(&out_row).expect("write row");
    }
    wtr.flush().expect("flush");
    eprintln!("wrote {} rows to {}", rows.len(), output.display());
}

fn compute_butter_3norm(src: &str, dst: &str) -> Option<f64> {
    let src_img = image::open(src).ok()?.to_rgb8();
    let dst_img = image::open(dst).ok()?.to_rgb8();
    let (w, h) = src_img.dimensions();
    let (dw, dh) = dst_img.dimensions();
    if w != dw || h != dh {
        return None;
    }
    let w_us = w as usize;
    let h_us = h as usize;
    if w_us < 8 || h_us < 8 {
        return None;
    }
    let src_pixels: Vec<[u8; 3]> = src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dst_pixels: Vec<[u8; 3]> = dst_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let src_rgb8: &[RGB8] = bytemuck::cast_slice(&src_pixels);
    let dst_rgb8: &[RGB8] = bytemuck::cast_slice(&dst_pixels);
    let s = Img::new(src_rgb8, w_us, h_us);
    let d = Img::new(dst_rgb8, w_us, h_us);
    let bp = ButteraugliParams::default().with_compute_diffmap(true);
    let result = butteraugli::butteraugli(s, d, &bp).ok()?;
    let dm = result.diffmap?;
    Some(libjxl_pnorm(dm.buf(), 3.0))
}

/// libjxl-style p-norm averaging at p, 2p, 4p. See
/// `lib/extras/metrics.cc` `ComputeDistanceP`.
fn libjxl_pnorm(diffmap: &[f32], p: f64) -> f64 {
    if diffmap.is_empty() {
        return f64::NAN;
    }
    let mut sum1 = [0.0_f64; 3];
    for &v in diffmap {
        let d = v as f64;
        let mut acc = d.powf(p);
        sum1[0] += acc;
        acc *= acc;
        sum1[1] += acc;
        acc *= acc;
        sum1[2] += acc;
    }
    let one_per_pixels = 1.0 / diffmap.len() as f64;
    let mut v = 0.0_f64;
    for (i, &s) in sum1.iter().enumerate() {
        let exponent = 1.0 / (p * (1u32 << i) as f64);
        v += (one_per_pixels * s).powf(exponent);
    }
    v / 3.0
}
