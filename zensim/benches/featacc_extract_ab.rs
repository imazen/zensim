// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! featacc COST MEASUREMENT: whole `research::extract(everything())` under
//! each candidate arithmetic, interleaved IN-PROCESS so a shared box's drift
//! hits every candidate alike. `bench_featcanon` rewrites the measurement
//! cells between bench fns; every iter then runs the full Rev4 walk.
//!
//! Run (single-threaded, pinned — see benchmarks/featacc_WORKLOG.md):
//!   `RAYON_NUM_THREADS=1 taskset -c <cpu> cargo bench --bench \
//!     featacc_extract_ab -p zensim --features custom-profiles,feature-regime-v2,threads,training,oracle`
//!
//! Synthetic pairs: a fixed LCG pattern, ref and dist differing by a
//! deterministic perturbation (extract requires non-identical inputs). The
//! walk's work is data-independent in extent (strip/block counts are a
//! function of dimensions), so pixel provenance does not bias the slope.
use zensim::RgbSlice;
use zensim::feature_v2::bench_featcanon;
use zensim::research::{self, Request};

fn pixels(w: usize, h: usize, salt: usize) -> Vec<[u8; 3]> {
    (0..w * h)
        .map(|i| {
            let k = (i.wrapping_add(salt * 7919)).wrapping_mul(2654435761) % 65521;
            let g = (k % 256) as u8;
            // mild channel spread so chroma planes are not constant
            [g, ((k / 3) % 256) as u8, ((k / 7) % 256) as u8]
        })
        .collect()
}

/// (label, mode, blur) — the candidate matrix. "prod" is env-unset shipped
/// Rev4; "off" the uncanonized baseline; exact runs its default `fresh`
/// blur axis (the oracle itself — its cost is reported, not a candidate).
const MODES: &[(&str, &str, Option<&str>)] = &[
    ("prod", "prod", None),
    ("off", "off", None),
    ("c32", "c32", None),
    ("c64", "c64", None),
    ("neum", "neum", None),
    ("exact", "exact", None),
    // blur axis under one fixed accumulation mode (c32) prices the window
    // evaluation alone: rec (default) vs f64-sliding vs fresh re-sum.
    ("c32+rec64", "c32", Some("rec64")),
    ("c32+fresh", "c32", Some("fresh")),
];

fn main() {
    // `FEATACC_SIZES=256,1024` / `FEATACC_MODES=prod,off,...` narrow the
    // matrix for focused reruns on a busy box.
    let sizes: Vec<usize> = std::env::var("FEATACC_SIZES")
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![64, 256, 1024, 4096]);
    let modes: Vec<_> = std::env::var("FEATACC_MODES")
        .map(|s| {
            MODES
                .iter()
                .filter(|(l, _, _)| s.split(',').any(|m| m == *l))
                .collect()
        })
        .unwrap_or_else(|_| MODES.iter().collect());
    let result = zenbench::run(|suite| {
        for &size in &sizes {
            let src = pixels(size, size, 1);
            let dst = pixels(size, size, 2);
            assert_ne!(src, dst);
            let s = RgbSlice::new(Box::leak(Box::new(src)).as_slice(), size, size);
            let d = RgbSlice::new(Box::leak(Box::new(dst)).as_slice(), size, size);
            suite.compare(format!("extract_{size}x{size}"), |group| {
                group
                    .config()
                    .min_rounds(if size >= 1024 { 5 } else { 15 })
                    .max_rounds(if size >= 1024 { 15 } else { 60 })
                    .max_wall_time(std::time::Duration::from_secs(if size >= 4096 {
                        240
                    } else {
                        120
                    }));
                for &&(label, mode, blur) in &modes {
                    group.bench(label, move |b| {
                        b.iter(move || {
                            assert!(bench_featcanon(mode, blur));
                            zenbench::black_box(
                                research::extract(&Request::everything(), &s, &d).expect("extract"),
                            )
                            .values()
                            .len()
                        })
                    });
                }
            });
        }
    });
    let _ = result;
}
