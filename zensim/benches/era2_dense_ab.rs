// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! era-1 vs era-2 DENSE KERNEL, paired/interleaved.
//! `benchmarks/era2_perf_break_2026-08-31.md` — the break's justification is
//! speed, so the kernel that carries it is measured before it is flipped.
//!
//! Run: `cargo bench --bench era2_dense_ab -p zensim \
//!        --features custom-profiles,feature-regime-v2,threads,training,oracle`
use zensim::feature_v2::{bench_dense_era1, bench_dense_era2};

fn planes(w: usize, h: usize) -> Vec<Vec<f32>> {
    (0..7)
        .map(|p| {
            (0..w * h)
                .map(|i| {
                    let k = ((i + p * 7919) * 2654435761usize) % 65521;
                    (k as f32) * (1.0 / 65536.0) + 0.05
                })
                .collect()
        })
        .collect()
}

fn main() {
    let result = zenbench::run(|suite| {
        for &(w, h) in &[(576usize, 128usize), (1152, 128)] {
            let p = planes(w, h);
            let p: &'static Vec<Vec<f32>> = Box::leak(Box::new(p));
            suite.compare(format!("era2_dense_{w}x{h}"), |group| {
                group
                    .config()
                    .max_rounds(200)
                    .min_rounds(25)
                    .max_wall_time(std::time::Duration::from_secs(100));
                group.bench("era1_dispatched", move |b| {
                    b.iter(move || {
                        zenbench::black_box(bench_dense_era1(
                            &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6], w, h, true,
                        ))
                    })
                });
                group.bench("era2", move |b| {
                    b.iter(move || {
                        zenbench::black_box(bench_dense_era2(
                            &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6], w, h, true,
                        ))
                    })
                });
            });
        }
    });
    let _ = result;
}
