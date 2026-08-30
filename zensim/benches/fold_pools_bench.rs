// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! The carrier lane's registered perf instrument (2026-08-30,
//! `benchmarks/balance_campaign_2026-08-28.md` "carriers named + costed" §4
//! bench honesty): a zenbench PAIRED / interleaved A/B of the streaming
//! folded-944 extraction with v1's pool blocks (`f156..372` — the carriers'
//! native slots) EMITTED — the ten carriers only, or the full 216 —
//! (`V2NewFeatureToggles::v1_pools`) vs ZEROED (the
//! production default), serial, on the same textured pair at 576² and
//! 1152². The earlier "+0.52 ms" figure was the old buffered v1 harness's
//! wall-clock loop, not this structure.
//!
//! Run: `cargo bench --bench fold_pools_bench -p zensim \
//!        --features custom-profiles,feature-regime-v2`
use zensim::profile::ProfileParams;
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Deterministic textured pair (the attribution tests' content family).
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

fn main() {
    let params = ProfileParams::builder()
        .skip_score_mapping(true)
        .extrapolate_score(true)
        .extended_features(true)
        .build();
    let params: &'static ProfileParams = Box::leak(Box::new(params));
    let z: &'static Zensim = Box::leak(Box::new(
        Zensim::new(ZensimProfile::Custom {
            name: "fold_pools_bench",
            params,
        })
        .with_parallel(false),
    ));
    let zeroed = zensim::feature_v2::V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        ..Default::default()
    };
    let carriers = zensim::feature_v2::V2NewFeatureToggles {
        v1_pools: zensim::feature_v2::V1PoolsMode::Carriers,
        ..zeroed
    };
    let emitted = zensim::feature_v2::V2NewFeatureToggles {
        v1_pools: zensim::feature_v2::V1PoolsMode::Full,
        ..zeroed
    };
    let result = zenbench::run(|suite| {
        for &(w, h) in &[(576usize, 576usize), (1152, 1152)] {
            let (src, dst) = test_pair(w, h);
            let src_static: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
            let dst_static: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
            suite.compare(&format!("folded944_pools_{w}"), |group| {
                // The default 120 s group budget yielded only 4 usable rounds
                // on a shared box (the paired CI then spans ±10 points, which
                // cannot resolve a few-percent lever). Raise the wall budget +
                // the round floor so the interleaved A/B has enough paired
                // samples; the gate still throttles on load.
                group
                    .config()
                    .max_rounds(200)
                    .min_rounds(25)
                    .max_wall_time(std::time::Duration::from_secs(600));
                group.bench("pools_zeroed", move |b| {
                    let mut scratch = zensim::feature_v2::V2Scratch::new();
                    b.iter(move || {
                        let rsv = RgbSlice::new(src_static, w, h);
                        let dsv = RgbSlice::new(dst_static, w, h);
                        let v2 = z
                            .compute_folded720_features_streaming(&rsv, &dsv, zeroed, &mut scratch)
                            .unwrap();
                        zenbench::black_box(v2.features()[943]);
                    })
                });
                group.bench("pools_carriers10", move |b| {
                    let mut scratch = zensim::feature_v2::V2Scratch::new();
                    b.iter(move || {
                        let rsv = RgbSlice::new(src_static, w, h);
                        let dsv = RgbSlice::new(dst_static, w, h);
                        let v2 = z
                            .compute_folded720_features_streaming(&rsv, &dsv, carriers, &mut scratch)
                            .unwrap();
                        zenbench::black_box((v2.features()[178], v2.features()[943]));
                    })
                });
                group.bench("pools_full216", move |b| {
                    let mut scratch = zensim::feature_v2::V2Scratch::new();
                    b.iter(move || {
                        let rsv = RgbSlice::new(src_static, w, h);
                        let dsv = RgbSlice::new(dst_static, w, h);
                        let v2 = z
                            .compute_folded720_features_streaming(&rsv, &dsv, emitted, &mut scratch)
                            .unwrap();
                        zenbench::black_box((v2.features()[178], v2.features()[943]));
                    })
                });
            });
        }
    });
    let _ = result;
}
