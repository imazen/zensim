// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! Head-to-head of the two feature-extraction FAMILIES, paired/interleaved
//! in one process so shared-box noise cancels
//! (`benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md`).
//!
//! Four arms per size, one interleaved group:
//!
//! | arm | path | features |
//! |---|---|---|
//! | `buf_v1_228`  | BUFFERED v1 (`compute_zensim_with_config`, whole-image pyramids) | 228 |
//! | `buf_v1_372`  | same, `extended_features` + `compute_iw_features` | 372 (156 basic + 216 pools) |
//! | `fold944_off` | STREAMING fold, `V1PoolsMode::Off` (`folded720append2`) | 944, f156..371 = structural zeros |
//! | `fold944_full`| STREAMING fold, `V1PoolsMode::Full` (`folded720append2pools`) | 944, all live |
//!
//! `buf_v1_372 − buf_v1_228` and `fold944_full − fold944_off` are the
//! marginal cost of the SAME 216 v1 pool features in each family — that is
//! the honest same-feature-set comparison, because no fold mode computes
//! v1-372 alone (`fold_v1` is hardcoded on and v2-348 always rides along).
//!
//! WIDTH NOTE, load-bearing for reading these numbers: the buffered path
//! walks `simd_padded_width(w)` columns, the fold walks `w`. At 576/1152/2304
//! that is +16 columns (592/1168/2320), so the buffered arm does 2.8 % / 1.4 %
//! / 0.7 % more column work — and its pool VALUES differ from the fold's
//! (`folded720_v1_pools_match_v1_path` is bit-exact only where
//! `simd_padded_width(w) == w`).
//!
//! Thread count comes from `RAYON_NUM_THREADS` (both families take
//! `parallel = true`); run the binary once per count.
//!
//! Run:
//! ```text
//! cargo bench --bench extract_paths_bench -p zensim \
//!   --features custom-profiles,feature-regime-v2,threads,training
//! ```
//! Peak-RSS mode (one arm, no zenbench, for `/usr/bin/time -v`):
//! `ZEN_XP_RSS=<arm> ZEN_XP_SIZE=1152 ZEN_XP_ITERS=20 <the bench binary>`
use zensim::profile::ProfileParams;
use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

/// Deterministic textured pair — the same content family the attribution
/// tests and `fold_pools_bench` use, so numbers are comparable across both.
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

fn v1_cfg(extended: bool, iw: bool) -> ZensimConfig {
    // `ZensimConfig` is `#[non_exhaustive]` — build then set.
    let mut c = ZensimConfig::default();
    c.extended_features = extended;
    c.compute_iw_features = iw;
    c.allow_multithreading = true;
    c
}

fn fold_zensim() -> &'static Zensim {
    let params = ProfileParams::builder()
        .skip_score_mapping(true)
        .extrapolate_score(true)
        .extended_features(true)
        .build();
    let params: &'static ProfileParams = Box::leak(Box::new(params));
    Box::leak(Box::new(
        Zensim::new(ZensimProfile::Custom {
            name: "extract_paths_bench",
            params,
        })
        .with_parallel(true),
    ))
}

fn toggles_off() -> zensim::feature_v2::V2NewFeatureToggles {
    zensim::feature_v2::V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        ..Default::default()
    }
}

fn toggles_full() -> zensim::feature_v2::V2NewFeatureToggles {
    zensim::feature_v2::V2NewFeatureToggles {
        v1_pools: zensim::feature_v2::V1PoolsMode::Full,
        ..toggles_off()
    }
}

/// One-arm loop for external peak-RSS measurement (`/usr/bin/time -v`).
fn rss_mode(arm: &str) {
    let size: usize = std::env::var("ZEN_XP_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1152);
    let iters: usize = std::env::var("ZEN_XP_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let (src, dst) = test_pair(size, size);
    let z = fold_zensim();
    let mut scratch = zensim::feature_v2::V2Scratch::new();
    let mut sink = 0.0f64;
    for _ in 0..iters {
        match arm {
            "buf_v1_228" => {
                let r = compute_zensim_with_config(&src, &dst, size, size, v1_cfg(false, false))
                    .expect("buf_v1_228");
                sink += r.features()[0];
            }
            "buf_v1_372" => {
                let r = compute_zensim_with_config(&src, &dst, size, size, v1_cfg(true, true))
                    .expect("buf_v1_372");
                sink += r.features()[371];
            }
            "fold944_off" | "fold944_full" => {
                let t = if arm == "fold944_full" {
                    toggles_full()
                } else {
                    toggles_off()
                };
                let rsv = RgbSlice::new(&src, size, size);
                let dsv = RgbSlice::new(&dst, size, size);
                let v2 = z
                    .compute_folded720_features_streaming(&rsv, &dsv, t, &mut scratch)
                    .expect("fold");
                sink += v2.features()[943] as f64;
            }
            other => panic!("unknown ZEN_XP_RSS arm: {other}"),
        }
    }
    println!("{arm} size={size} iters={iters} sink={sink:e}");
}

fn main() {
    if let Ok(arm) = std::env::var("ZEN_XP_RSS") {
        rss_mode(&arm);
        return;
    }
    let sizes: Vec<usize> = std::env::var("ZEN_XP_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![576, 1152, 2304]);
    let z = fold_zensim();
    let (off, full) = (toggles_off(), toggles_full());
    let result = zenbench::run(|suite| {
        for &n in &sizes {
            let (src, dst) = test_pair(n, n);
            let src_s: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
            let dst_s: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
            suite.compare(&format!("extract_paths_{n}"), |group| {
                // Same budget the pools lane raised fold_pools_bench to: the
                // default 120 s / 4-usable-rounds cannot resolve a few-percent
                // lever on a shared box.
                group
                    .config()
                    .max_rounds(200)
                    .min_rounds(25)
                    .max_wall_time(std::time::Duration::from_secs(600));
                group.bench("buf_v1_228", move |b| {
                    b.iter(move || {
                        let r =
                            compute_zensim_with_config(src_s, dst_s, n, n, v1_cfg(false, false))
                                .unwrap();
                        zenbench::black_box(r.features()[0]);
                    })
                });
                group.bench("buf_v1_372", move |b| {
                    b.iter(move || {
                        let r = compute_zensim_with_config(src_s, dst_s, n, n, v1_cfg(true, true))
                            .unwrap();
                        zenbench::black_box(r.features()[371]);
                    })
                });
                group.bench("fold944_off", move |b| {
                    let mut scratch = zensim::feature_v2::V2Scratch::new();
                    b.iter(move || {
                        let rsv = RgbSlice::new(src_s, n, n);
                        let dsv = RgbSlice::new(dst_s, n, n);
                        let v2 = z
                            .compute_folded720_features_streaming(&rsv, &dsv, off, &mut scratch)
                            .unwrap();
                        zenbench::black_box(v2.features()[943]);
                    })
                });
                group.bench("fold944_full", move |b| {
                    let mut scratch = zensim::feature_v2::V2Scratch::new();
                    b.iter(move || {
                        let rsv = RgbSlice::new(src_s, n, n);
                        let dsv = RgbSlice::new(dst_s, n, n);
                        let v2 = z
                            .compute_folded720_features_streaming(&rsv, &dsv, full, &mut scratch)
                            .unwrap();
                        zenbench::black_box((v2.features()[178], v2.features()[943]));
                    })
                });
            });
        }
    });
    let _ = result;
}
