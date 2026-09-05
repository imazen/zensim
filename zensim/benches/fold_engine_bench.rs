// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! **Fold-backed vs buffered SCORING**, paired/interleaved in one process so
//! shared-box noise cancels (`benchmarks/fold_engine_2026-08-31.md` stage 2).
//!
//! This is a different question from `extract_paths_bench`'s. That bench
//! compares feature-extraction FAMILIES at different widths (buffered v1-372
//! vs the 944 fold), so its arms do different amounts of work by design. Here
//! both arms compute **the same `ZensimResult`, bit-for-bit** — same profile,
//! same 372 features, same score — and differ only in which walk produced it.
//! `fold_engine_parity` is the gate that makes that claim true; this bench is
//! what it costs.
//!
//! | arm | entry | walk |
//! |---|---|---|
//! | `score_buffered` | `Zensim::compute` (profile B) | buffered |
//! | `score_fold`     | `Zensim::compute` (profile B) | the fold (`v1_only` + `V1PoolsMode::Full`) |
//! | `feat_buffered`  | `Zensim::compute_extended_features` (B) | buffered |
//! | `feat_fold`      | same | the fold |
//! | `ref_buffered`   | `Zensim::compute_with_ref` (B), one precomputed reference | buffered |
//! | `ref_fold`       | same | the fold (source-side planes COPIED from the cache) |
//!
//! With `custom-profiles` two more arms price the RETIREMENT MIGRATION for
//! the fused compare — the one genuinely walk-bound attribution consumer:
//!
//! | arm | what |
//! |---|---|
//! | `fused_buffered` | `compute_with_ref_score_and_attribution` — score + map from ONE buffered pipeline (the C3a shape) |
//! | `split_fold`     | fold-backed `compute_with_ref` + standalone `compute_attribution_density_with_ref` |
//!
//! `fused_compare_splits_into_fold_score_plus_standalone_map` gates that the
//! split does not move the score; these arms are what it costs.
//!
//! The `ref_*` arms are the M1 precompute-once / compare-many shape: the
//! reference pyramid is built ONCE outside the timed loop and every iteration
//! scores one distorted candidate against it, so `ref_x − score_x` is what
//! amortising the reference buys that engine. On the fold that is the decode +
//! sRGB→XYB conversion + 3-level downscale of the source side; on buffered it
//! is the same, plus buffered additionally skips re-materialising its
//! reference pyramid.
//!
//! `score_* − feat_*` is the bake forward + output spline, which is SHARED
//! code running after both walks — so it should come out equal in both
//! families, and a large asymmetry there is a measurement problem, not a
//! finding.
//!
//! Thread count comes from `RAYON_NUM_THREADS` (both arms take
//! `parallel = true`); run the binary once per count. Budget matches
//! `extract_paths_bench`'s raised one — the default 120 s / 4-usable-rounds
//! cannot resolve a few-percent lever on a shared box.
//!
//! Run:
//! ```text
//! RAYON_NUM_THREADS=8 cargo bench --bench fold_engine_bench -p zensim \
//!   --features feature-regime-v2,threads
//! ```
use zensim::fold_engine::ScoringEngine;
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Deterministic textured pair — byte-for-byte the generator
/// `extract_paths_bench` and `fold_pools_bench` use, so numbers are
/// comparable across all three.
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

fn engines() -> (&'static Zensim, &'static Zensim) {
    let buffered = Box::leak(Box::new(
        Zensim::new(ZensimProfile::codec_target()).with_parallel(true),
    ));
    let fold = Box::leak(Box::new(
        Zensim::new(ZensimProfile::codec_target())
            .with_parallel(true)
            .with_engine(ScoringEngine::Fold),
    ));
    (buffered, fold)
}

/// One-arm loop for external peak-RSS measurement (`/usr/bin/time -v`),
/// mirroring `extract_paths_bench`'s `ZEN_XP_RSS` mode.
///
/// Arms: the six timed ones above, plus the two pool-block CONTROLS
/// `poolctl_full` / `poolctl_off` added by the fold-footprint lane
/// (`benchmarks/fold_footprint_2026-08-31.md`) — the scoring walk with the
/// f156..371 pool block live vs structurally zero, so peak RSS prices
/// `FoldPoolScratch`. `poolctl_off` is a measurement control, never a
/// shippable configuration.
fn rss_mode(arm: &str) {
    let size: usize = std::env::var("ZEN_FE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1152);
    let iters: usize = std::env::var("ZEN_FE_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let (src, dst) = test_pair(size, size);
    let (buffered, fold) = engines();
    let z = match arm {
        "score_buffered" | "feat_buffered" | "ref_buffered" | "refinto_buffered" => buffered,
        "score_fold" | "feat_fold" | "ref_fold" | "refinto_fold" => fold,
        // MEASUREMENT CONTROL, not a shippable configuration (see the loop).
        "poolctl_full" | "poolctl_off" => fold,
        other => panic!("unknown ZEN_FE_RSS arm: {other}"),
    };
    let pre = (arm.starts_with("ref_") || arm.starts_with("refinto_")).then(|| {
        z.precompute_reference(&RgbSlice::new(&src, size, size))
            .unwrap()
    });
    let mut sc = zensim::ZensimScratch::new();
    let mut sink = 0.0f64;
    for _ in 0..iters {
        // POOL-BLOCK CONTROL ARMS (`poolctl_full` / `poolctl_off`). Same walk
        // shape the fold-backed score takes — `v1_only`, SDR, one fresh
        // `V2Scratch` per call, exactly as `metric::compute_with_config_inner`
        // builds one per `Zensim::compute` — differing ONLY in whether the
        // f156..371 pool block (peaks / masked / IW) is live. The delta is
        // therefore the `FoldPoolScratch` footprint and nothing else.
        // `poolctl_off` emits structural zeros in f156..371 and is a
        // MEASUREMENT CONTROL only; the product mode is `Full`.
        if let Some(mode) = match arm {
            "poolctl_full" => Some(zensim::feature_v2::V1PoolsMode::Full),
            "poolctl_off" => Some(zensim::feature_v2::V1PoolsMode::Off),
            _ => None,
        } {
            let toggles = zensim::feature_v2::V2NewFeatureToggles {
                v1_only: true,
                v1_pools: mode,
                ..Default::default()
            };
            let mut sc2 = zensim::feature_v2::V2Scratch::new();
            let v2 = z
                .compute_folded720_features_streaming(
                    &RgbSlice::new(&src, size, size),
                    &RgbSlice::new(&dst, size, size),
                    toggles,
                    &mut sc2,
                )
                .expect("poolctl");
            sink += v2.features()[371] as f64;
            continue;
        }
        let rsv = RgbSlice::new(&src, size, size);
        let dsv = RgbSlice::new(&dst, size, size);
        let r = match &pre {
            Some(p) if arm.starts_with("refinto_") => z
                .compute_with_ref_into(p, &dsv, &mut sc)
                .expect("compute_with_ref_into"),
            Some(p) => z.compute_with_ref(p, &dsv).expect("compute_with_ref"),
            None if arm.starts_with("score_") => z.compute(&rsv, &dsv).expect("compute"),
            None => z
                .compute_extended_features(&rsv, &dsv)
                .expect("compute_extended_features"),
        };
        sink += r.score() + r.features()[371];
    }
    println!("{arm} size={size} iters={iters} sink={sink:e}");
}

fn main() {
    if let Ok(arm) = std::env::var("ZEN_FE_RSS") {
        rss_mode(&arm);
        return;
    }
    let sizes: Vec<usize> = std::env::var("ZEN_FE_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![576, 1152, 2304]);
    let (buffered, fold) = engines();
    let result = zenbench::run(|suite| {
        for &n in &sizes {
            let (src, dst) = test_pair(n, n);
            let src_s: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
            let dst_s: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
            suite.compare(format!("fold_engine_{n}"), |group| {
                group
                    .config()
                    .max_rounds(200)
                    .min_rounds(25)
                    .max_wall_time(std::time::Duration::from_secs(600));
                group.bench("score_buffered", move |b| {
                    b.iter(move || {
                        let r = buffered
                            .compute(&RgbSlice::new(src_s, n, n), &RgbSlice::new(dst_s, n, n))
                            .unwrap();
                        zenbench::black_box(r.score());
                    })
                });
                group.bench("score_fold", move |b| {
                    b.iter(move || {
                        let r = fold
                            .compute(&RgbSlice::new(src_s, n, n), &RgbSlice::new(dst_s, n, n))
                            .unwrap();
                        zenbench::black_box(r.score());
                    })
                });
                group.bench("feat_buffered", move |b| {
                    b.iter(move || {
                        let r = buffered
                            .compute_extended_features(
                                &RgbSlice::new(src_s, n, n),
                                &RgbSlice::new(dst_s, n, n),
                            )
                            .unwrap();
                        zenbench::black_box(r.features()[371]);
                    })
                });
                group.bench("feat_fold", move |b| {
                    b.iter(move || {
                        let r = fold
                            .compute_extended_features(
                                &RgbSlice::new(src_s, n, n),
                                &RgbSlice::new(dst_s, n, n),
                            )
                            .unwrap();
                        zenbench::black_box(r.features()[371]);
                    })
                });
                // Precomputed ONCE, outside the timed loop — the encoder-loop
                // shape. Both engines consume the SAME `PrecomputedReference`.
                let pre_b: &'static _ = Box::leak(Box::new(
                    buffered
                        .precompute_reference(&RgbSlice::new(src_s, n, n))
                        .unwrap(),
                ));
                let pre_f: &'static _ = Box::leak(Box::new(
                    fold.precompute_reference(&RgbSlice::new(src_s, n, n))
                        .unwrap(),
                ));
                group.bench("ref_buffered", move |b| {
                    b.iter(move || {
                        let r = buffered
                            .compute_with_ref(pre_b, &RgbSlice::new(dst_s, n, n))
                            .unwrap();
                        zenbench::black_box(r.score());
                    })
                });
                group.bench("ref_fold", move |b| {
                    b.iter(move || {
                        let r = fold
                            .compute_with_ref(pre_f, &RgbSlice::new(dst_s, n, n))
                            .unwrap();
                        zenbench::black_box(r.score());
                    })
                });
                // The REF-LOOP shape: one scratch kept alive across compares,
                // which is what an encoder quantisation loop actually does.
                // `ref_*` above pays a fresh allocation every call on both
                // engines; `refinto_* - ref_*` is what keeping it buys.
                // (fold-MT lane — `compute_with_ref_into` routes to the fold
                // and reuses its `V2Scratch`; before this lane the ONE entry
                // that exists to amortise work was the one entry the fold
                // could not serve.)
                group.bench("refinto_buffered", move |b| {
                    let mut sc = zensim::ZensimScratch::new();
                    b.iter(move || {
                        let r = buffered
                            .compute_with_ref_into(pre_b, &RgbSlice::new(dst_s, n, n), &mut sc)
                            .unwrap();
                        zenbench::black_box(r.score());
                    })
                });
                group.bench("refinto_fold", move |b| {
                    let mut sc = zensim::ZensimScratch::new();
                    b.iter(move || {
                        let r = fold
                            .compute_with_ref_into(pre_f, &RgbSlice::new(dst_s, n, n), &mut sc)
                            .unwrap();
                        zenbench::black_box(r.score());
                    })
                });
                #[cfg(feature = "custom-profiles")]
                {
                    // A mixed-sign, all-slot basic gradient — the same shape
                    // `fused_matches_standalone_attribution` uses.
                    let s_grad: &'static [f64] = Box::leak(
                        (0..156)
                            .map(|k| {
                                let sign = if k % 3 == 0 { -1.0 } else { -0.25 };
                                sign * (1.0 + (k % 7) as f64 * 0.1)
                            })
                            .collect::<Vec<f64>>()
                            .into_boxed_slice(),
                    );
                    group.bench("fused_buffered", move |b| {
                        b.iter(move || {
                            let (r, a) = buffered
                                .compute_with_ref_score_and_attribution(
                                    pre_b,
                                    &RgbSlice::new(dst_s, n, n),
                                    s_grad,
                                )
                                .unwrap();
                            zenbench::black_box((r.score(), a.density()[0]));
                        })
                    });
                    group.bench("split_fold", move |b| {
                        b.iter(move || {
                            let ds = RgbSlice::new(dst_s, n, n);
                            let r = fold.compute_with_ref(pre_f, &ds).unwrap();
                            let a = fold
                                .compute_attribution_density_with_ref(pre_f, &ds, s_grad)
                                .unwrap();
                            zenbench::black_box((r.score(), a.density()[0]));
                        })
                    });
                }
            });
        }
    });
    let _ = result;
}
