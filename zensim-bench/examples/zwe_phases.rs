// Per-phase timing: H-blur vs V-blur breakdown.
//
// Builds with `--features zensim/zwe-time-phases` to enable the in-tree
// AtomicU64 phase counters in zensim::streaming::phase_timing.
//
// Runs with multithreading DISABLED so the per-phase nanosecond counts
// are clean serial work, not parallel-scheduled deltas.

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

#[cfg(feature = "zwe-time-phases-driver")]
fn dump(label: &str, iters: usize, width: usize, height: usize) {
    use zensim::streaming::phase_timing;
    let (src, dst) = make_test_images(width, height);
    let z = Zensim::new(ZensimProfile::latest()).with_parallel(false);
    let s = RgbSlice::new(&src, width, height);
    let pre = z.precompute_reference(&s).unwrap();
    let d = RgbSlice::new(&dst, width, height);

    for _ in 0..4 {
        let _ = z.compute_with_ref(&pre, &d).unwrap();
    }

    phase_timing::reset();
    let t0 = Instant::now();
    for _ in 0..iters {
        let r = z.compute_with_ref(&pre, &d).unwrap();
        std::hint::black_box(r);
    }
    let total = t0.elapsed().as_nanos() as u64;
    let [hs, hm, vs, ve] = phase_timing::snapshot();
    let hb = hs + hm;
    let vb = vs + ve;
    let to_ms = |ns: u64| (ns / iters as u64) as f64 / 1e6;
    println!(
        "{label:>14}  total/iter={:>7.3} ms   H-ssim={:>6.3}  H-mu={:>6.3}  V-ssim={:>6.3}  V-edge={:>6.3}   H/iter={:>6.3} ({:>4.1}%)  V/iter={:>6.3} ({:>4.1}%)",
        total as f64 / iters as f64 / 1e6,
        to_ms(hs),
        to_ms(hm),
        to_ms(vs),
        to_ms(ve),
        to_ms(hb),
        100.0 * hb as f64 / total as f64,
        to_ms(vb),
        100.0 * vb as f64 / total as f64,
    );
}

#[cfg(not(feature = "zwe-time-phases-driver"))]
fn dump(_label: &str, _iters: usize, _width: usize, _height: usize) {
    eprintln!(
        "Build with --features zwe-time-phases-driver to enable phase timers (passes zensim/zwe-time-phases)."
    );
    std::process::exit(1);
}

fn main() {
    let iters: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    println!("# zwe_phases: iters={iters} (single-threaded)");
    dump("256x256", iters, 256, 256);
    dump("512x512", iters, 512, 512);
    dump("1280x720", iters, 1280, 720);
    dump("1920x1080", iters, 1920, 1080);
}
