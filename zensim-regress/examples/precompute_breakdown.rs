//! Probe what `precompute_reference` spends its time on.
use std::time::Instant;
use zensim::{RgbaSlice, Zensim, ZensimProfile};

fn gradient(w: u32, h: u32, seed: u32) -> Vec<[u8; 4]> {
    let mut s = seed;
    let mut out = Vec::with_capacity((w * h) as usize);
    for _ in 0..(w * h) {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let r = (s >> 16) as u8;
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let g = (s >> 16) as u8;
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let b = (s >> 16) as u8;
        out.push([r, g, b, 255]);
    }
    out
}

fn main() {
    let z = Zensim::new(ZensimProfile::latest());

    for &(w, h) in &[(1920u32, 1080u32), (3840, 2160)] {
        let src = gradient(w, h, 42);
        let dst = gradient(w, h, 99);

        // Warm up
        for _ in 0..3 {
            let s = RgbaSlice::new(&src, w as usize, h as usize);
            let _ = z.precompute_reference(&s).unwrap();
        }

        // Bench precompute_reference
        let n_iter = if w >= 3000 { 8 } else { 16 };
        let t0 = Instant::now();
        for _ in 0..n_iter {
            let s = RgbaSlice::new(&src, w as usize, h as usize);
            let r = z.precompute_reference(&s).unwrap();
            std::hint::black_box(r);
        }
        let pre_t = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

        // Bench compute() (e2e)
        let t0 = Instant::now();
        for _ in 0..n_iter {
            let s = RgbaSlice::new(&src, w as usize, h as usize);
            let d = RgbaSlice::new(&dst, w as usize, h as usize);
            let r = z.compute(&s, &d).unwrap();
            std::hint::black_box(r);
        }
        let e2e_t = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

        // Bench compute_with_ref (which does NOT precompute, just second image)
        let s = RgbaSlice::new(&src, w as usize, h as usize);
        let pre = z.precompute_reference(&s).unwrap();
        let t0 = Instant::now();
        for _ in 0..n_iter {
            let d = RgbaSlice::new(&dst, w as usize, h as usize);
            let r = z.compute_with_ref(&pre, &d).unwrap();
            std::hint::black_box(r);
        }
        let ref_t = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

        println!(
            "{}x{}: precompute={:.2}ms  e2e={:.2}ms  compute_with_ref={:.2}ms  Δ(pre-(e2e-compute))={:.2}ms",
            w, h, pre_t, e2e_t, ref_t,
            pre_t - (e2e_t - ref_t),
        );
    }
}
