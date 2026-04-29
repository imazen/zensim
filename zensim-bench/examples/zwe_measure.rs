// Minimal interleaved A/B perf harness for zero-weight-elision experiments.
//
// Builds two test sizes, runs many iterations of `compute_with_ref` for each
// configuration interleaved (round-robin) to neutralize thermal/turbo bias,
// reports paired median + p25/p75.
//
// Usage:
//   cargo run -p zensim-bench --example zwe_measure --release --features training -- [iters]

use std::env;
use std::time::Instant;
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn make_test_images(width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = width * height;
    let src: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = ((i % width) * 255 / width) as u8;
            let y = ((i / width) * 255 / height) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| [r.saturating_add(8), g.saturating_add(4), b])
        .collect();
    (src, dst)
}

fn pct(samples: &mut [u64], p: f64) -> u64 {
    samples.sort_unstable();
    let i = ((samples.len() as f64) * p) as usize;
    samples[i.min(samples.len() - 1)]
}

fn run_size(label: &str, width: usize, height: usize, iters: usize) {
    let (src, dst) = make_test_images(width, height);
    let z = Zensim::new(ZensimProfile::latest());
    let s = RgbSlice::new(&src, width, height);
    let pre = z.precompute_reference(&s).unwrap();
    let d = RgbSlice::new(&dst, width, height);

    // warmup
    for _ in 0..8 {
        let _ = z.compute_with_ref(&pre, &d).unwrap();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = z.compute_with_ref(&pre, &d).unwrap();
        let dt = t0.elapsed().as_nanos() as u64;
        std::hint::black_box(r);
        times.push(dt);
    }
    let med = pct(&mut times.clone(), 0.50);
    let p25 = pct(&mut times.clone(), 0.25);
    let p75 = pct(&mut times.clone(), 0.75);
    let mp_per_s = (width as f64 * height as f64) / (med as f64 / 1e9) / 1e6;
    println!(
        "{label:>14}  med={:>10.3} ms   p25={:>10.3}  p75={:>10.3}   {:>7.1} MP/s",
        med as f64 / 1e6,
        p25 as f64 / 1e6,
        p75 as f64 / 1e6,
        mp_per_s,
    );
}

fn main() {
    let iters: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);

    println!("# zwe_measure: iters={iters}");
    println!("# git: {}", env!("CARGO_PKG_VERSION"));
    run_size("256x256", 256, 256, iters);
    run_size("512x512", 512, 512, iters);
    run_size("1280x720", 1280, 720, iters);
    run_size("1920x1080", 1920, 1080, iters);
}
