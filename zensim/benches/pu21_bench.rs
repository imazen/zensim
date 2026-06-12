//! PU-XYB conversion throughput: SIMD (incant-dispatched) vs scalar powf.
//!
//! Measures the HDR front-end's absolute-nits → positive PU-XYB planar
//! conversion at 1080p (2.07 MP). Run:
//! `cargo bench -p zensim --bench pu21_bench`
//! Results doc: `benchmarks/pu21_simd_bench_2026-06-10.md`.

use zenbench::prelude::*;

const N: usize = 1920 * 1080;

fn make_pixels() -> Vec<[f32; 3]> {
    (0..N)
        .map(|i| {
            let t = i as f32 / (N - 1) as f32;
            let y = 0.01 * (4000.0f32 / 0.01).powf(t);
            [y * 1.2, y, y * 0.7]
        })
        .collect()
}

fn bench_pu_xyb(suite: &mut Suite) {
    suite.group("pu_xyb_1080p", |g| {
        g.throughput(Throughput::Elements(N as u64));
        g.bench("simd_dispatch", |b| {
            b.with_input(|| {
                (
                    make_pixels(),
                    vec![0.0f32; N],
                    vec![0.0f32; N],
                    vec![0.0f32; N],
                )
            })
            .run(|(px, mut xs, mut ys, mut bs)| {
                zensim::bench_pu_xyb_dispatch(&px, &mut xs, &mut ys, &mut bs);
                (px, xs, ys, bs)
            })
        });
        g.bench("scalar_powf", |b| {
            b.with_input(|| {
                (
                    make_pixels(),
                    vec![0.0f32; N],
                    vec![0.0f32; N],
                    vec![0.0f32; N],
                )
            })
            .run(|(px, mut xs, mut ys, mut bs)| {
                zensim::bench_pu_xyb_scalar(&px, &mut xs, &mut ys, &mut bs);
                (px, xs, ys, bs)
            })
        });
    });
}

zenbench::main!(bench_pu_xyb);
