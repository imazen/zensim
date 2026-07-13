//! Measure the scaled-run producer across run lengths to pick
//! `font::SCALE_BATCH`, and model expected per-size cost on
//! representative label texts.
//!
//! ```text
//! cargo run --release -p zensim-regress --features _internal_api \
//!     --example glyph_batch_bench
//! ```
//!
//! Per-run cost fits `overhead + per_glyph · count`; the workload model
//! charges each label the batches it touches. Example-grade `Instant`
//! timing for a build-time constant choice — not a CI perf gate.

use std::time::Instant;

const LABELS: &[&str] = &[
    "EXPECTED",
    "ACTUAL",
    "PIXEL DIFF",
    "STRUCTURAL DIFF",
    "FAILED - zdsim 0.132 > 0.010",
    "ok: zdsim 0.004 <= 0.010",
    "zenjpeg-420-e1 q35 vs q75",
    "max 18/255 mean 0.38 px>2 6.17%",
    "Rag7 Handgloves 0123456789",
    "sea:a1b2c3d4 sunny-crab",
];

fn distinct_glyphs(s: &str) -> Vec<u32> {
    let mut v: Vec<u32> = s
        .chars()
        .filter(|c| (' '..='~').contains(c))
        .map(|c| c as u32 - 0x20)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

fn batches_touched(glyphs: &[u32], batch: u32) -> u32 {
    let mut b: Vec<u32> = glyphs.iter().map(|g| g / batch).collect();
    b.dedup();
    b.len() as u32
}

fn main() {
    let sizes = [12u32, 27, 54];
    let counts = [1u32, 2, 4, 8, 16, 32, 96];

    println!(
        "{:>5} {:>7} {:>12} {:>14}",
        "size", "run", "ms/run", "us/glyph"
    );
    // (count, size) -> measured ms
    let mut measured = Vec::new();
    for &h in &sizes {
        let w = (26 * h + 27) / 54;
        for &c in &counts {
            let iters = (2000 / c).clamp(20, 400);
            let t = Instant::now();
            for i in 0..iters {
                // Vary start so cache effects can't help and every run
                // stays in-bounds.
                let start = (i * 7) % (96 - c.min(95));
                std::hint::black_box(zensim_regress::font::bench_build_scaled_run(start, c, w, h));
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            println!(
                "{:>4}px {:>7} {:>12.4} {:>14.2}",
                h,
                c,
                ms,
                ms * 1000.0 / c as f64
            );
            measured.push((h, c, ms));
        }
    }

    // Least-squares fit ms = a + b*count per size, then model workloads.
    println!("\n{:>5} {:>10} {:>12}", "size", "a (ms)", "b (ms/glyph)");
    let mut fits = Vec::new();
    for &h in &sizes {
        let pts: Vec<(f64, f64)> = measured
            .iter()
            .filter(|(s, _, _)| *s == h)
            .map(|(_, c, ms)| (*c as f64, *ms))
            .collect();
        let n = pts.len() as f64;
        let (sx, sy): (f64, f64) = pts.iter().fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
        let sxx: f64 = pts.iter().map(|(x, _)| x * x).sum();
        let sxy: f64 = pts.iter().map(|(x, y)| x * y).sum();
        let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let a = (sy - b * sx) / n;
        println!("{:>4}px {:>10.4} {:>12.4}", h, a, b);
        fits.push((h, a, b));
    }

    println!(
        "\nworkload model: mean cost (ms) to first-render all {} labels at a fresh size",
        LABELS.len()
    );
    print!("{:>5}", "size");
    for &c in &counts {
        print!(" {:>8}", format!("B={c}"));
    }
    println!();
    for &(h, a, b) in &fits {
        print!("{:>4}px", h);
        for &c in &counts {
            let total: f64 = LABELS
                .iter()
                .map(|s| {
                    let g = distinct_glyphs(s);
                    batches_touched(&g, c) as f64 * (a + b * c as f64)
                })
                .sum();
            print!(" {:>8.3}", total / LABELS.len() as f64);
        }
        println!();
    }
    println!(
        "\n(pick the B minimizing the row at your working sizes; ties -> larger B for fewer allocs)"
    );
}
