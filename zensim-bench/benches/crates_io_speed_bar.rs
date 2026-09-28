// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! **What does the current tree cost against the last crates.io release?**
//!
//! `zensim 0.2.7` (`PreviewV0_2`, the last published version) is compiled into
//! this binary under the name `zensim027` next to the current tree, so the two
//! run as INTERLEAVED arms of one zenbench group — the only way a cross-version
//! comparison on this box is attributable (a rebuild alone has moved a 2304²
//! timing ~10%; `benchmarks/era2_perf_break_2026-08-31.md` §22.5).
//!
//! Read the arms for what they are:
//!
//! | arm | what it computes | comparable to |
//! |---|---|---|
//! | `v027_latest` | 0.2.7 `Zensim::new(ZensimProfile::latest())` — the product entry THEN | `main_B` as a PRODUCT, not as an optimization: different metric, different feature width |
//! | `main_B` | current `Zensim::new(ZensimProfile::B)` — the product entry NOW | — |
//! | `v027_buf228` | 0.2.7 `compute_zensim_with_config`, v1 basic 228 features, buffered | `main_buf228`: the SAME extraction API and width on both versions |
//! | `main_buf228` | current `compute_zensim_with_config`, v1 basic 228, buffered | — |
//! | `fast_ssim2` | the external anchor, compiled once, version-independent | if it moves, the box moved |
//!
//! Single thread by construction: both `Zensim`s are `with_parallel(false)`,
//! both configs `allow_multithreading = false`; still run with
//! `RAYON_NUM_THREADS=1` and a pinned core so nothing else spins up a pool.
//!
//! Run:
//! ```text
//! RAYON_NUM_THREADS=1 ZEN_S2_SIZES=1024,2048 ZEN_S2_WALL_S=300 \
//!   taskset -c 8 nice -n19 ionice -c3 \
//!   cargo bench --manifest-path zensim-bench/Cargo.toml --bench crates_io_speed_bar \
//!   --features crates-io-0-2-7
//! ```
//! `ZEN_S2_SIZES` (default `576,1152,2304`), `ZEN_S2_ROUNDS`, `ZEN_S2_WALL_S`
//! mean what they mean in `ssim2_speed_bar`. Same `test_pair` fixture as
//! `zensim/benches/extract_paths_bench.rs`, so the `main_*` arms line up with
//! that record's `buf_v1_228` numbers.

use imgref::Img;
use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

/// Deterministic textured pair — byte-identical to `extract_paths_bench` and
/// `ssim2_speed_bar`, so the numbers are comparable across all three records.
fn test_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut src = Vec::with_capacity(w * h);
    let mut dst = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40 } else { 0 };
            let px = [
                base.wrapping_add(tex),
                base.wrapping_add(edge),
                (255 - base).wrapping_add(tex / 2),
            ];
            src.push(px);
            let q = |v: u8| (v / 12) * 12;
            let mut d = [q(px[0]), q(px[1]), q(px[2])];
            if x < w / 2 && y < h / 2 {
                d[0] = d[0].saturating_add(18);
            }
            dst.push(d);
        }
    }
    (src, dst)
}

fn env_usize(key: &str, dflt: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(dflt)
}

/// Current tree: the buffered v1 basic configuration (`buf_v1_228` in
/// `extract_paths_bench`), single-threaded.
fn main_cfg() -> ZensimConfig {
    let mut c = ZensimConfig::default();
    c.extended_features = false;
    c.compute_iw_features = false;
    c.allow_multithreading = false;
    c
}

/// 0.2.7: the same buffered v1 basic configuration. Its `ZensimConfig` has no
/// IW switch (that family did not exist yet), so the two configs agree on
/// everything both versions know about.
fn v027_cfg() -> zensim027::ZensimConfig {
    let mut c = zensim027::ZensimConfig::default();
    c.extended_features = false;
    c.allow_multithreading = false;
    c
}

fn main() {
    let sizes: Vec<usize> = std::env::var("ZEN_S2_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![576, 1152, 2304]);
    let max_r = env_usize("ZEN_S2_ROUNDS", 200);
    let min_r = max_r.min(env_usize("ZEN_S2_MIN_ROUNDS", 30));
    let wall_s = env_usize("ZEN_S2_WALL_S", 120) as u64;

    let zm: &'static Zensim =
        Box::leak(Box::new(Zensim::new(ZensimProfile::B).with_parallel(false)));
    let zo: &'static zensim027::Zensim = Box::leak(Box::new(
        zensim027::Zensim::new(zensim027::ZensimProfile::latest()).with_parallel(false),
    ));
    println!(
        "# crates_io_speed_bar: zensim027=0.2.7 ({:?}) vs current ({:?}); RAYON_NUM_THREADS={}",
        zensim027::ZensimProfile::latest(),
        ZensimProfile::B,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<unset>".into()),
    );

    let result = zenbench::run(|suite| {
        for &n in &sizes {
            let (src, dst) = test_pair(n, n);
            let src_s: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
            let dst_s: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
            // Both product entries must accept the fixture before anything is timed.
            let s_old = zensim027::RgbSlice::new(src_s, n, n);
            let d_old = zensim027::RgbSlice::new(dst_s, n, n);
            let old_score = zo.compute(&s_old, &d_old).unwrap().score();
            let new_score = zm
                .compute(&RgbSlice::new(src_s, n, n), &RgbSlice::new(dst_s, n, n))
                .unwrap()
                .score();
            println!(
                "# geometry={n} v027_latest_score={old_score:.4} main_B_score={new_score:.4} (different metrics; printed so nobody reads the timings as a scoring agreement)"
            );
            suite.compare(format!("crates_io_bar_{n}"), |group| {
                group
                    .config()
                    .max_rounds(max_r)
                    .min_rounds(min_r)
                    .max_wall_time(std::time::Duration::from_secs(wall_s));
                group.bench("fast_ssim2", move |b| {
                    b.iter(move || {
                        let s = Img::new(src_s, n, n);
                        let d = Img::new(dst_s, n, n);
                        zenbench::black_box(fast_ssim2::compute_ssimulacra2(s, d).unwrap())
                    })
                });
                group.bench("v027_latest", move |b| {
                    b.iter(move || {
                        let s = zensim027::RgbSlice::new(src_s, n, n);
                        let d = zensim027::RgbSlice::new(dst_s, n, n);
                        zenbench::black_box(zo.compute(&s, &d).unwrap().score())
                    })
                });
                group.bench("main_B", move |b| {
                    b.iter(move || {
                        let s = RgbSlice::new(src_s, n, n);
                        let d = RgbSlice::new(dst_s, n, n);
                        zenbench::black_box(zm.compute(&s, &d).unwrap().score())
                    })
                });
                group.bench("v027_buf228", move |b| {
                    b.iter(move || {
                        let r =
                            zensim027::compute_zensim_with_config(src_s, dst_s, n, n, v027_cfg())
                                .unwrap();
                        zenbench::black_box(r.features()[0])
                    })
                });
                group.bench("main_buf228", move |b| {
                    b.iter(move || {
                        let r = compute_zensim_with_config(src_s, dst_s, n, n, main_cfg()).unwrap();
                        zenbench::black_box(r.features()[0])
                    })
                });
            });
        }
    });
    if let Some(path) = std::env::var_os("ZENBENCH_RESULT_PATH").map(std::path::PathBuf::from) {
        result.save(path).expect("save benchmark evidence");
    }
}
