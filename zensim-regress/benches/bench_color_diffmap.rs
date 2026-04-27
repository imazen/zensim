//! End-to-end micro-benchmarks targeting the cbrt-heavy color path
//! (via `compute`) and the diffmap pipeline.

use zensim::{DiffmapOptions, DiffmapWeighting, RgbaSlice, Zensim, ZensimProfile};

fn gradient_rgba(w: u32, h: u32) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 127) / h.max(1)) as u8;
            let b = (((x + y * 2) * 63) / (w + h).max(1)) as u8;
            out.push([r, g, b, 255]);
        }
    }
    out
}

fn noise_rgba(w: u32, h: u32, seed: u32) -> Vec<[u8; 4]> {
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

/// End-to-end zensim score (color → blur → SSIM → pooling). cbrt is hot.
fn bench_zensim_score(suite: &mut zenbench::Suite) {
    suite.group("zensim_score_e2e", |group| {
        for &(label, w, h) in &[
            ("256x256", 256u32, 256u32),
            ("1920x1080", 1920, 1080),
            ("3840x2160", 3840, 2160),
        ] {
            let z = Zensim::new(ZensimProfile::latest());
            let src = gradient_rgba(w, h);
            let dst = noise_rgba(w, h, 7);
            group.bench(label, move |b| {
                b.iter(|| {
                    let s = RgbaSlice::new(std::hint::black_box(&src), w as usize, h as usize);
                    let d = RgbaSlice::new(std::hint::black_box(&dst), w as usize, h as usize);
                    z.compute(&s, &d).unwrap()
                })
            });
        }
    });
}

/// Precompute reference (color conversion + multi-scale pyramid).
/// This is the cleanest cbrt-only signal: no SSIM correlation work.
fn bench_precompute_reference(suite: &mut zenbench::Suite) {
    suite.group("precompute_reference", |group| {
        for &(label, w, h) in &[
            ("256x256", 256u32, 256u32),
            ("1920x1080", 1920, 1080),
            ("3840x2160", 3840, 2160),
        ] {
            let z = Zensim::new(ZensimProfile::latest());
            let src = gradient_rgba(w, h);
            group.bench(label, move |b| {
                b.iter(|| {
                    let s = RgbaSlice::new(std::hint::black_box(&src), w as usize, h as usize);
                    z.precompute_reference(&s).unwrap()
                })
            });
        }
    });
}

/// Diffmap: cheapest options (no masking, no sqrt, no edge_mse, no hf).
fn bench_diffmap_minimal(suite: &mut zenbench::Suite) {
    suite.group("diffmap_minimal", |group| {
        for &(label, w, h) in &[("256x256", 256u32, 256u32), ("1920x1080", 1920, 1080)] {
            let z = Zensim::new(ZensimProfile::latest());
            let src = gradient_rgba(w, h);
            let dst = noise_rgba(w, h, 11);
            let opts = DiffmapOptions {
                weighting: DiffmapWeighting::Balanced,
                masking_strength: None,
                sqrt: false,
                include_edge_mse: false,
                include_hf: false,
            };
            group.bench(label, move |b| {
                b.iter(|| {
                    let s = RgbaSlice::new(std::hint::black_box(&src), w as usize, h as usize);
                    let d = RgbaSlice::new(std::hint::black_box(&dst), w as usize, h as usize);
                    z.compute_with_diffmap(&s, &d, opts).unwrap()
                })
            });
        }
    });
}

/// Diffmap: full options (masking + sqrt + edge_mse + hf).
fn bench_diffmap_full(suite: &mut zenbench::Suite) {
    suite.group("diffmap_full", |group| {
        for &(label, w, h) in &[("256x256", 256u32, 256u32), ("1920x1080", 1920, 1080)] {
            let z = Zensim::new(ZensimProfile::latest());
            let src = gradient_rgba(w, h);
            let dst = noise_rgba(w, h, 11);
            let opts = DiffmapOptions {
                weighting: DiffmapWeighting::Balanced,
                masking_strength: Some(8.0),
                sqrt: true,
                include_edge_mse: true,
                include_hf: true,
            };
            group.bench(label, move |b| {
                b.iter(|| {
                    let s = RgbaSlice::new(std::hint::black_box(&src), w as usize, h as usize);
                    let d = RgbaSlice::new(std::hint::black_box(&dst), w as usize, h as usize);
                    z.compute_with_diffmap(&s, &d, opts).unwrap()
                })
            });
        }
    });
}

zenbench::main!(
    bench_zensim_score,
    bench_precompute_reference,
    bench_diffmap_minimal,
    bench_diffmap_full,
);
