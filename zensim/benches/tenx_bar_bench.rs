//! Appendix Y Part-0 bar bench (campaign `benchmarks/sota944_campaign_2026-08-03.md`
//! REGISTERED APPENDIX Y): the honest butteraugli denominator vs the zensim
//! primitives, per size × thread config — the measurement that freezes the
//! "10× faster than butteraugli" bar as concrete ms numbers.
//!
//! Arms (Y.1, frozen):
//!   1. `butter_oneshot`  — `butteraugli()` default params (intensity 80,
//!      diffmap ON — the deployed jxl-AQ comparator shape), cold: the ref-side
//!      work is inside the timed region (the standalone-tool shape).
//!   2. `butter_warm`     — `ButteraugliReference::new` OUTSIDE timing,
//!      `.compare()` timed (the codec-loop shape).
//!   3. `z_extract944`    — canonical streaming folded-944 extraction.
//!   4. `z_score944`      — extraction + MLP forward of the wave-11 packed
//!      candidate (the honest score-only compare).
//!   5. `z_fused_score_map` — `compute_folded944_score_and_attribution` +
//!      forward + one `query_rect` (the steered-compare kernel).
//!   6. `z_v1_score`      — 372-class v1 walk score-only (v47A-class row).
//!   7. `z_v1_fused_score_map` — C3a 372-class fused score+map (+ forward of
//!      the s156 gradient's basic block is NOT run here: the 372-class loop
//!      forwards through its own mount; this arm times the zensim kernel).
//!
//! Thread configs (two invocations of this same binary):
//!   ST: RAYON_NUM_THREADS=1, zensim with_parallel(false)   [default]
//!   MT: ZENSIM_TENX_MT=1 + run-heavy --jobs 6 (RAYON_NUM_THREADS=6),
//!       zensim with_parallel(true)
//!
//! Sizes: 576×576 / 1024×1024 / 3840×2160. Plain release, NO target-cpu=native.
//! The MLP bake: `/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin`
//! (caller-944/internal-667), override via ZENSIM_BENCH_BAKE; FAILS LOUD if absent.
//!
//! Run (each config):
//!   RAYON_NUM_THREADS=1 cargo bench --bench tenx_bar_bench -p zensim \
//!       --features custom-profiles,feature-regime-v2
//!   ZENSIM_TENX_MT=1 run-heavy --jobs 6 -- cargo bench --bench tenx_bar_bench \
//!       -p zensim --features custom-profiles,feature-regime-v2

use butteraugli::{ButteraugliParams, ButteraugliReference, ImgVec, RGB8};
use zenpredict::{Model, Predictor};
use zensim::profile::ProfileParams;
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Deterministic textured pair (the attribution tests' content family —
/// identical generator to `fused944_bench` / `fused944_probe`).
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

fn to_img(px: &[[u8; 3]], w: usize, h: usize) -> ImgVec<RGB8> {
    ImgVec::new(
        px.iter()
            .map(|p| RGB8 {
                r: p[0],
                g: p[1],
                b: p[2],
            })
            .collect(),
        w,
        h,
    )
}

fn flat(px: &[[u8; 3]]) -> Vec<u8> {
    let mut v = Vec::with_capacity(px.len() * 3);
    for p in px {
        v.extend_from_slice(p);
    }
    v
}

fn bake_path() -> String {
    std::env::var("ZENSIM_BENCH_BAKE").unwrap_or_else(|_| {
        "/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin".to_string()
    })
}

fn main() {
    let mt = std::env::var("ZENSIM_TENX_MT").is_ok();
    let rayon_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into());
    println!(
        "tenx_bar_bench config: {} (zensim with_parallel({}), RAYON_NUM_THREADS={})",
        if mt { "MT" } else { "ST" },
        mt,
        rayon_threads
    );

    let bake_file = bake_path();
    let bake_bytes = std::fs::read(&bake_file).unwrap_or_else(|e| {
        panic!(
            "tenx_bar_bench REQUIRES the packed candidate bake at {bake_file} \
             (override with ZENSIM_BENCH_BAKE): {e}"
        )
    });
    let model: &'static Model = Box::leak(Box::new(
        Model::from_bytes(&bake_bytes).expect("parse ZNPR bake"),
    ));
    let caller_w = model.caller_input_width();
    assert_eq!(caller_w, 944, "expected a caller-944 bake, got {caller_w}");

    let params = ProfileParams::builder()
        .skip_score_mapping(true)
        .extrapolate_score(true)
        .extended_features(true)
        .build();
    let params: &'static ProfileParams = Box::leak(Box::new(params));
    let z: &'static Zensim = Box::leak(Box::new(
        Zensim::new(ZensimProfile::Custom {
            params,
            name: "tenx-bar-bench",
        })
        .with_parallel(mt),
    ));

    // (label, w, h, max_rounds, min_rounds, max_time_s)
    let sizes: &[(&str, usize, usize, usize, usize, u64)] = &[
        ("576", 576, 576, 200, 30, 150),
        ("1mp", 1024, 1024, 60, 15, 300),
        ("4k", 3840, 2160, 20, 8, 480),
    ];

    let result = zenbench::run(|suite| {
        for &(label, w, h, rounds, min_rounds, max_time_s) in sizes {
            let (src, dst) = test_pair(w, h);
            let src_static: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
            let dst_static: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
            let bsrc: &'static ImgVec<RGB8> = Box::leak(Box::new(to_img(src_static, w, h)));
            let bdst: &'static ImgVec<RGB8> = Box::leak(Box::new(to_img(dst_static, w, h)));
            let dst_flat: &'static [u8] = Box::leak(flat(dst_static).into_boxed_slice());
            let rs = RgbSlice::new(src_static, w, h);
            let pre: &'static zensim::PrecomputedReference =
                Box::leak(Box::new(z.precompute_reference(&rs).unwrap()));
            let bwarm: &'static ButteraugliReference = Box::leak(Box::new(
                ButteraugliReference::new(&flat(src_static), w, h, ButteraugliParams::default())
                    .unwrap(),
            ));
            // Synthetic 944 gradient (values do not affect kernel cost; the
            // fused entry's pass-B work is gradient-shape-independent).
            let mut s944 = vec![0.0f64; 944];
            for (k, v) in s944.iter_mut().enumerate() {
                *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
            }
            let s944: &'static [f64] = Box::leak(s944.into_boxed_slice());
            let s156: &'static [f64] = Box::leak(vec![-1.0f64; 156].into_boxed_slice());
            let toggles944 = zensim::feature_v2::V2NewFeatureToggles {
                append_block: true,
                append2_block: true,
                ..Default::default()
            };

            suite.compare(format!("tenx_bar_{label}"), move |group| {
                group
                    .config()
                    .max_rounds(rounds)
                    .min_rounds(min_rounds)
                    .max_time(std::time::Duration::from_secs(max_time_s));

                group.bench("butter_oneshot", move |b| {
                    b.iter(move || {
                        let r = butteraugli::butteraugli(
                            bsrc.as_ref(),
                            bdst.as_ref(),
                            &ButteraugliParams::default(),
                        )
                        .unwrap();
                        zenbench::black_box(r.score);
                    })
                });

                group.bench("butter_warm", move |b| {
                    b.iter(move || {
                        let r = bwarm.compare(dst_flat).unwrap();
                        zenbench::black_box(r.score);
                    })
                });

                group.bench("z_extract944", move |b| {
                    let mut scratch = zensim::feature_v2::V2Scratch::new();
                    b.iter(move || {
                        let rsv = RgbSlice::new(src_static, w, h);
                        let dsv = RgbSlice::new(dst_static, w, h);
                        let v2 = z
                            .compute_folded720_features_streaming(
                                &rsv,
                                &dsv,
                                toggles944,
                                &mut scratch,
                            )
                            .unwrap();
                        zenbench::black_box(v2.features()[943]);
                    })
                });

                group.bench("z_score944", move |b| {
                    let mut scratch = zensim::feature_v2::V2Scratch::new();
                    let mut predictor = Predictor::new(model);
                    let mut f32buf = vec![0.0f32; 944];
                    b.iter(move || {
                        let rsv = RgbSlice::new(src_static, w, h);
                        let dsv = RgbSlice::new(dst_static, w, h);
                        let v2 = z
                            .compute_folded720_features_streaming(
                                &rsv,
                                &dsv,
                                toggles944,
                                &mut scratch,
                            )
                            .unwrap();
                        for (o, i) in f32buf.iter_mut().zip(v2.features()) {
                            *o = *i as f32;
                        }
                        let out = predictor.predict_transformed(&f32buf).unwrap();
                        zenbench::black_box(out[0]);
                    })
                });

                group.bench("z_fused_score_map", move |b| {
                    let mut sess = zensim::Fused944Session::new();
                    let mut predictor = Predictor::new(model);
                    let mut f32buf = vec![0.0f32; 944];
                    b.iter(move || {
                        let rsv = RgbSlice::new(src_static, w, h);
                        let dsv = RgbSlice::new(dst_static, w, h);
                        let (r, v2, a) = z
                            .compute_folded944_score_and_attribution(
                                &rsv, pre, &dsv, s944, &mut sess,
                            )
                            .unwrap();
                        for (o, i) in f32buf.iter_mut().zip(v2.features()) {
                            *o = *i as f32;
                        }
                        let out = predictor.predict_transformed(&f32buf).unwrap();
                        zenbench::black_box((r.score(), out[0], a.query_rect(0, 0, 32, 32)));
                    })
                });

                group.bench("z_v1_score", move |b| {
                    b.iter(move || {
                        let dsv = RgbSlice::new(dst_static, w, h);
                        let r = z.compute_with_ref(pre, &dsv).unwrap();
                        zenbench::black_box(r.score());
                    })
                });

                group.bench("z_v1_fused_score_map", move |b| {
                    b.iter(move || {
                        let dsv = RgbSlice::new(dst_static, w, h);
                        let (r, a) = z
                            .compute_with_ref_score_and_attribution(pre, &dsv, s156)
                            .unwrap();
                        zenbench::black_box((r.score(), a.query_rect(0, 0, 32, 32)));
                    })
                });
            });
        }
    });
    let _ = result;
}
