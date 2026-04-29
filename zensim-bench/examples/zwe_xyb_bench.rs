//! Microbench for zensim's sRGB→XYB conversion path.
//!
//! Build with:
//!   cargo build --release -p zensim-bench --example zwe_xyb_bench \
//!     --features zensim/zwe-pub-color
//!
//! Reports min ns/pixel to expose the per-pixel cost of the
//! linearize-LUT + SIMD matrix + cbrt + bias pipeline.

use std::time::Instant;
use zensim::color::srgb_to_positive_xyb_planar_into;

fn make_pixels(n: usize) -> Vec<[u8; 3]> {
    (0..n)
        .map(|i| {
            let v = (i.wrapping_mul(2654435761) & 0xFFFFFF) as u32;
            [(v & 0xFF) as u8, ((v >> 8) & 0xFF) as u8, ((v >> 16) & 0xFF) as u8]
        })
        .collect()
}

fn bench_size(label: &str, w: usize, h: usize, iters: usize) {
    let n = w * h;
    let src = make_pixels(n);
    let mut x = vec![0.0f32; n];
    let mut y = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];

    // Warmup
    for _ in 0..4 {
        srgb_to_positive_xyb_planar_into(&src, &mut x, &mut y, &mut b);
    }

    let mut times: Vec<u64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        srgb_to_positive_xyb_planar_into(&src, &mut x, &mut y, &mut b);
        times.push(t0.elapsed().as_nanos() as u64);
        std::hint::black_box((&x, &y, &b));
    }
    times.sort_unstable();
    let min = times[0];
    let p25 = times[iters / 4];
    let p50 = times[iters / 2];
    let ns_per_pixel_min = min as f64 / n as f64;
    let ns_per_pixel_p50 = p50 as f64 / n as f64;
    let mp_per_s_min = (n as f64 * 1e3) / min as f64;
    let bw_in = (n * 3) as f64 / (min as f64) * 1e9 / 1e9; // bytes / sec → GiB/s rough
    println!(
        "{label:>11}  ({w}x{h}, {n}px, {iters}iter)  min={min}ns p25={p25} p50={p50}  ns/px={ns_per_pixel_min:.2}min/{ns_per_pixel_p50:.2}p50  {mp_per_s_min:.0}MP/s  ~{bw_in:.2} GiB/s read",
    );
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    println!("# zwe_xyb_bench: srgb_to_positive_xyb_planar_into");
    bench_size("256x256", 256, 256, iters);
    bench_size("512x512", 512, 512, iters);
    bench_size("1280x720", 1280, 720, iters);
    bench_size("1920x1080", 1920, 1080, iters);
    bench_size("3840x2160", 3840, 2160, iters / 2);
}
