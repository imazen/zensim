//! Compare two directories of montage PNGs (baseline vs new) and report
//! per-pixel differences. Usage:
//!   cargo run -p zensim-regress --example montage_diff -- <baseline_dir> <new_dir>

use std::env;

use image::{ImageReader, Rgba, RgbaImage};

fn main() {
    let mut args = env::args().skip(1);
    let baseline_dir = args
        .next()
        .unwrap_or_else(|| "/tmp/montage_baseline".into());
    let new_dir = args.next().unwrap_or_else(|| "/tmp/montage_new".into());

    let scenes = [
        "01_samedim_plain",
        "02_samedim_annotated",
        "03_mismatched_plain",
        "04_mismatched_annotated",
        "05_tiny_pixelate",
        "06_custom_labels",
    ];

    println!(
        "{:<30} {:>10} {:>10} {:>8} {:>8} {:>10}",
        "scene", "dim_b", "dim_n", "px_diff%", "max_d", "avg_d"
    );
    println!("{}", "-".repeat(80));

    let mut all_clean = true;
    for s in &scenes {
        let a_path = format!("{}/{}.png", baseline_dir, s);
        let b_path = format!("{}/{}.png", new_dir, s);
        let a = match ImageReader::open(&a_path).map(|r| r.decode().unwrap().to_rgba8()) {
            Ok(img) => img,
            Err(e) => {
                println!("{:<30} ERROR: {}", s, e);
                continue;
            }
        };
        let b = match ImageReader::open(&b_path).map(|r| r.decode().unwrap().to_rgba8()) {
            Ok(img) => img,
            Err(e) => {
                println!("{:<30} ERROR: {}", s, e);
                continue;
            }
        };
        let dim_a = format!("{}x{}", a.width(), a.height());
        let dim_b = format!("{}x{}", b.width(), b.height());
        if a.dimensions() != b.dimensions() {
            println!("{:<30} {:>10} {:>10}  DIM-MISMATCH", s, dim_a, dim_b);
            all_clean = false;
            continue;
        }

        let mut max_d = 0u32;
        let mut sum_d = 0u64;
        let mut diff_pixels = 0u64;
        for (pa, pb) in a.pixels().zip(b.pixels()) {
            let d = (0..4)
                .map(|i| (pa.0[i] as i32 - pb.0[i] as i32).unsigned_abs())
                .max()
                .unwrap();
            if d > 0 {
                diff_pixels += 1;
            }
            max_d = max_d.max(d);
            sum_d += d as u64;
        }
        let total_px = (a.width() * a.height()) as u64;
        let pct = 100.0 * diff_pixels as f64 / total_px as f64;
        let avg = sum_d as f64 / (total_px * 4) as f64;
        let marker = if max_d == 0 {
            "✓"
        } else if pct < 1.0 {
            "≈"
        } else {
            "✗"
        };

        println!(
            "{:<30} {:>10} {:>10} {:>7.2}% {:>8} {:>10.4} {}",
            s, dim_a, dim_b, pct, max_d, avg, marker
        );
        if max_d > 0 {
            all_clean = false;
            // Save a binary diff visualization: white where different.
            let mut diff_img = RgbaImage::new(a.width(), a.height());
            for (x, y, pa) in a.enumerate_pixels() {
                let pb = b.get_pixel(x, y);
                let d = (0..4)
                    .map(|i| (pa.0[i] as i32 - pb.0[i] as i32).unsigned_abs())
                    .max()
                    .unwrap();
                let v = (d * 255 / 234).min(255) as u8;
                diff_img.put_pixel(x, y, Rgba([v, 0, 0, 255]));
            }
            let _ = diff_img.save(format!("/tmp/montage_diff_{}.png", s));
        }
    }

    println!();
    println!("Clean: {}", all_clean);
}
