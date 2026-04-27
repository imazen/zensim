//! Measure whether reusing the distorted-plane allocation across repeated
//! compute_with_ref calls would help. We do this by simulating a "warm pages"
//! scenario: allocate one big buffer, then in a loop call compute_with_ref
//! many times against the same precomputed reference.
//!
//! The loop's per-call cost includes a fresh 3×n alloc + 3×n write + drop.
//! Comparing to a hypothetical world where the alloc/drop is amortized,
//! the gap is the encoder-loop opportunity.
use std::time::Instant;
use zensim::{RgbaSlice, Zensim, ZensimProfile, ZensimScratch};

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
        // Generate several distorted variants so each iteration has slightly
        // different data (defeats any "image cache" effects).
        let dists: Vec<Vec<[u8; 4]>> = (0..8).map(|i| gradient(w, h, 100 + i)).collect();

        let s = RgbaSlice::new(&src, w as usize, h as usize);
        let pre = z.precompute_reference(&s).unwrap();

        // Warmup
        for d in &dists {
            let dd = RgbaSlice::new(d, w as usize, h as usize);
            let _ = z.compute_with_ref(&pre, &dd).unwrap();
        }

        let n_iter = if w >= 3000 { 32 } else { 64 };
        let t0 = Instant::now();
        for k in 0..n_iter {
            let d = &dists[k % dists.len()];
            let dd = RgbaSlice::new(d, w as usize, h as usize);
            let r = z.compute_with_ref(&pre, &dd).unwrap();
            std::hint::black_box(r);
        }
        let no_scratch_ms = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

        // Warmup scratch
        let mut scratch = ZensimScratch::new();
        for d in &dists {
            let dd = RgbaSlice::new(d, w as usize, h as usize);
            let _ = z.compute_with_ref_into(&pre, &dd, &mut scratch).unwrap();
        }

        let t0 = Instant::now();
        for k in 0..n_iter {
            let d = &dists[k % dists.len()];
            let dd = RgbaSlice::new(d, w as usize, h as usize);
            let r = z.compute_with_ref_into(&pre, &dd, &mut scratch).unwrap();
            std::hint::black_box(r);
        }
        let scratch_ms = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

        let delta = no_scratch_ms - scratch_ms;
        let pct = delta / no_scratch_ms * 100.0;
        println!(
            "{}x{}  no_scratch={:.2} ms/call  scratch={:.2} ms/call  Δ={:+.2} ms ({:+.1}%)",
            w, h, no_scratch_ms, scratch_ms, delta, pct,
        );
    }
}
