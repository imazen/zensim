// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! **Does SSIMULACRA2 go faster when you give it threads?** — the one question
//! `zensim/benches/extract_paths_bench.rs`'s ssim2 arm cannot answer, because
//! `fast-ssim2`'s parallelism is a cargo feature and cargo features are
//! per-build, not per-arm (`benchmarks/ssim2_replacement_bar_2026-08-31.md`).
//!
//! fast-ssim2 parallelises its Gaussian blur — the dominant kernel — only under
//! its optional `rayon` feature, which is OFF in this crate's default dep line.
//! So: build twice, once with `--features ssim2-rayon`, and read the delta. The
//! `zensim_B` arm is present in BOTH builds purely as the cross-build anchor —
//! it is untouched by the feature, so if it moves, the box moved.
//!
//! zenbench rather than criterion because the two arms must be interleaved:
//! criterion's isolated back-to-back runs bake the box's thermal/neighbour
//! state into a head-to-head against an external opponent, which is precisely
//! the bias zenbench's randomized round-robin exists to remove.
//! `benches/bench_compare.rs` keeps the wider implementation matrix (C++ FFI,
//! rust-av port); this is the paired instrument for the speed row of the exam.
//!
//! Run:
//! ```text
//! for T in 1 8 16; do RAYON_NUM_THREADS=$T cargo bench --bench ssim2_speed_bar -p zensim-bench; done
//! for T in 1 8 16; do RAYON_NUM_THREADS=$T cargo bench --bench ssim2_speed_bar -p zensim-bench --features ssim2-rayon; done
//! ```
//! `ZEN_S2_SIZES` (default `576,1152,2304`), `ZEN_S2_ROUNDS`, `ZEN_S2_WALL_S`
//! keep a matrix run from holding zenbench's exclusive lock for hours.
//!
//! ## The amended-W4 arms (`benchmarks/hybrid_candidate_2026-09-01.md`, APPENDIX B)
//!
//! The exam's speed clause now binds at **both** 1 and 8 threads, and it prices
//! a candidate on **its own extraction regime plus its own forwards** rather
//! than on its feature width — two 944-wide models can read different regimes
//! (`folded720append2` with f156-371 zeroed vs `folded720append2pools` with it
//! live) whose walks differ materially. An ensemble is ONE compare: one
//! extraction of the regime that serves every member, plus every member's
//! forward.
//!
//! | arm | extraction | forwards |
//! |---|---|---|
//! | `add156_156basic`  | `V1PoolsMode::Off`, v1-only (f0..156)              | the additive head |
//! | `flagship_944off`  | `V1PoolsMode::Off`  (f156-371 = structural zeros) | the MLP |
//! | `q7b_944pools`     | `V1PoolsMode::Full` (all 944 live)                | the linear |
//! | `hybrid_944pools`  | `V1PoolsMode::Full`                               | MLP **and** linear |
//! | `free156_peaks_raw` | `V1PoolsMode::Peaks` + `V1FreeExtras::RawMoments`, v1-only, 944 layout | the 156+free MLP (A3b/A4b class) |
//! | `peaks156_no_raw`  | `V1PoolsMode::Peaks`, `V1FreeExtras::Off`, v1-only, 944 layout | the 156+peaks head — the ZERO-marginal-compute half of the free set |
//! | `add156_plus_corrhead` | identical to `add156_156basic` | the additive head **and** the companion corruption head (`ZEN_HY_CORRHEAD`) — the delta against `add156_156basic` prices attaching the head |
//!
//! Bake bytes come from the environment so none enter git:
//! `ZEN_HY_MLP` (the 944 MLP flagship), `ZEN_HY_LIN` (the 944 pools linear),
//! `ZEN_HY_PEAKS` (a 156+peaks head — omit it and the arm is simply absent),
//! `ZEN_HY_ADD` (the basic-only additive head — `bake_block_profile` says it
//! uses 28 of f0..155 and NONE of f156-371, so its true walk is the cheapest
//! v1-only fold, not the peaks fold the exam credited it with) and
//! `ZEN_HY_FREE` (the 156+free MLP — `A3b`/`A4b` class, `benchmarks/
//! wave_r4_2026-09-01.md` §23/§24 — a `--keep-features` model over 265 of the
//! 944 coordinates: f0..155 + 72 peaks + 37 raw-moment slots.
//! `bake_block_profile` on this class reports `v1_peaks` 72/72 used and
//! `f720_943` 37/224 used, i.e. `V1PoolsMode::Peaks` plus the raw-moments
//! extra — the "15f" walk `zensim/examples/foldapp_stream_bigpair.rs`
//! already validates in production, copied verbatim below so the two
//! instruments agree on what the free-set walk means).
//! When one is unset its arms are **skipped loudly** — never silently.
//! The forward is `Predictor::predict[_transformed]`, the same call
//! `zensim_validate::bake_runtime::score_row` dispatches; the output PCHIP
//! spline (one scalar eval) is NOT in the arm and that exclusion is stated
//! wherever the numbers are published.
use imgref::Img;
use zenpredict::{Model, Predictor};
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// One env-supplied bake, parsed once and reused across every round.
struct Head {
    model: Model,
    has_transforms: bool,
    width: usize,
}

impl Head {
    fn load(var: &str) -> Option<Head> {
        let path = std::env::var(var).ok()?;
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("# {var}={path}: unreadable ({e}) — amended-W4 arms SKIPPED");
                return None;
            }
        };
        let model = match Model::from_bytes(&bytes) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("# {var}={path}: not a loadable ZNPR ({e:?}) — amended-W4 arms SKIPPED");
                return None;
            }
        };
        let has_transforms = model.has_nontrivial_feature_transforms();
        let width = model.caller_input_width();
        eprintln!(
            "# {var}={path}: n_inputs={} caller_width={width}",
            model.n_inputs()
        );
        Some(Head {
            model,
            has_transforms,
            width,
        })
    }

    fn forward(&self, p: &mut Predictor<'_>, x: &mut Vec<f32>, feats: &[f64]) -> f64 {
        x.clear();
        x.extend(feats.iter().take(self.width).map(|v| *v as f32));
        let out = if self.has_transforms {
            p.predict_transformed(x).expect("forward")
        } else {
            p.predict(x).expect("forward")
        };
        out[0] as f64
    }
}

/// Byte-identical to `zensim/benches/extract_paths_bench.rs::test_pair` — the
/// content family the attribution tests and `fold_pools_bench` also use, so
/// this bench and that one feed their kernels the same pixels.
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

/// Cap the SIMD tier at `X64V3Token` (x86-64-v3 = AVX2+FMA+BMI2, Haswell
/// 2013 / Zen 1 2017 — see archmage's own token docs; this is NOT SSE4.2,
/// despite older notes in this repo's memory saying so) by disabling
/// `X64V4Token` process-wide, which cascades to `X64V4xToken` and
/// `Avx512Fp16Token` too (archmage `tokens::mod.rs`). `incant!` then resolves
/// to the `_v3` variant everywhere, on every arch this box runs (x86-64
/// only — a no-op returning `false` elsewhere). Requires the `testable_dispatch`
/// archmage feature (already a `[dependencies]` feature here) AND a build
/// without `-C target-cpu=native` (CLAUDE.md already bans that for
/// benchmarking) — `dangerously_disable_token_process_wide` returns `Err`
/// when the target features are compile-time-guaranteed, which this reports
/// rather than silently ignoring.
///
/// `ZEN_S2_CAP_V3=1` selects the capped (AVX2) tier for the WHOLE process;
/// unset/`=0` leaves the native top tier (AVX-512 on this box) — Cargo
/// features are per-build, so capping per-arm inside one process is not
/// possible; the honest way to price both tiers is two process runs with
/// this env flipped, exactly like `ssim2-rayon` above.
#[cfg(target_arch = "x86_64")]
fn cap_tier_v3(cap: bool) -> Result<(), String> {
    archmage::X64V4Token::dangerously_disable_token_process_wide(cap).map_err(|e| e.to_string())
}
#[cfg(not(target_arch = "x86_64"))]
fn cap_tier_v3(_cap: bool) -> Result<(), String> {
    Ok(())
}

fn main() {
    if env_usize("ZEN_S2_CAP_V3", 0) == 1 {
        match cap_tier_v3(true) {
            Ok(()) => {
                eprintln!("# ZEN_S2_CAP_V3=1: X64V4Token disabled process-wide, ceiling=v3(AVX2)")
            }
            Err(e) => {
                eprintln!(
                    "# ZEN_S2_CAP_V3=1 requested but could not disable X64V4Token: {e} \
                     — refusing to report a native-tier run mislabeled as capped."
                );
                std::process::exit(1);
            }
        }
    }
    let sizes: Vec<usize> = std::env::var("ZEN_S2_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![576, 1152, 2304]);
    let max_r = env_usize("ZEN_S2_ROUNDS", 200);
    let min_r = max_r.min(env_usize("ZEN_S2_MIN_ROUNDS", 8));
    let wall_s = env_usize("ZEN_S2_WALL_S", 120) as u64;

    let zb: &'static Zensim = Box::leak(Box::new(Zensim::new(ZensimProfile::B)));
    // The shipped FAST-CLASS profile. `candidate-profiles` is one of `zensim`'s
    // DEFAULT features, so this needs no gate here — a dependency's features are
    // not visible to this crate as `cfg(feature = ...)` anyway, which is the
    // trap an earlier draft of this arm fell into.
    let zd: &'static Zensim = Box::leak(Box::new(Zensim::new(ZensimProfile::D)));
    // Amended-W4 arms. `Box::leak` so the &'static the closures need is real;
    // both are parsed exactly once, outside every timed region.
    let mlp: Option<&'static Head> = Head::load("ZEN_HY_MLP").map(|h| &*Box::leak(Box::new(h)));
    let lin: Option<&'static Head> = Head::load("ZEN_HY_LIN").map(|h| &*Box::leak(Box::new(h)));
    let add: Option<&'static Head> = Head::load("ZEN_HY_ADD").map(|h| &*Box::leak(Box::new(h)));
    let free: Option<&'static Head> = Head::load("ZEN_HY_FREE").map(|h| &*Box::leak(Box::new(h)));
    // PEAKS-ONLY arm (`ZEN_HY_PEAKS`): the ZERO-marginal-compute half of the free
    // set. `benchmarks/free_features_2026-09-01.md` §2.1 measured the 944 LAYOUT
    // alone (peaks emitted, accumulators off) at ratio CIs that all straddle 1.0,
    // and its §2.2 priced the raw-moment accumulators separately at ~1 %/1T. This
    // arm is the first half without the second, so the two halves can be read
    // apart in ONE binary instead of across two runs.
    let peaks: Option<&'static Head> = Head::load("ZEN_HY_PEAKS").map(|h| &*Box::leak(Box::new(h)));
    // The companion CORRUPTION head. Priced as a MARGINAL cost, not as its own
    // arm: `add156_plus_corrhead` runs the identical extraction and the identical
    // profile forward as `add156_156basic`, plus one extra forward. The delta
    // between the two arms — interleaved, so the box's state is common-mode — is
    // what attaching the head actually costs.
    let corrhead: Option<&'static Head> =
        Head::load("ZEN_HY_CORRHEAD").map(|h| &*Box::leak(Box::new(h)));
    // The fold engine + the two pool modes the two regimes correspond to.
    let zf: &'static Zensim = Box::leak(Box::new(Zensim::new(ZensimProfile::B)));
    // Byte-identical to `zensim/benches/extract_paths_bench.rs`'s `toggles_off`
    // / `toggles_full`, so the two instruments request the same walks.
    let pools_off = zensim::feature_v2::V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        ..Default::default()
    };
    let pools_full = zensim::feature_v2::V2NewFeatureToggles {
        v1_pools: zensim::feature_v2::V1PoolsMode::Full,
        ..pools_off
    };
    // The cheapest fold that can serve a basic-only (f0..156) model — the walk
    // ADD156 actually needs. Matches `extract_paths_bench`'s `fold156_basic`.
    let v1_basic = zensim::feature_v2::V2NewFeatureToggles {
        v1_pools: zensim::feature_v2::V1PoolsMode::Off,
        v1_only: true,
        ..Default::default()
    };
    // The free-set walk (A3b/A4b class): basic + peaks + raw-moments, at the
    // 944 LAYOUT (append_block/append2_block true) so the scattered
    // `--keep-features` indices land at their true f156.. / f720.. positions
    // — copied verbatim from the "15f" arm in
    // `zensim/examples/foldapp_stream_bigpair.rs` (the free-features lane's
    // own validated reference), not re-derived.
    let v1_basic_free = zensim::feature_v2::V2NewFeatureToggles {
        v1_only: true,
        v1_pools: zensim::feature_v2::V1PoolsMode::Peaks,
        append_block: true,
        append2_block: true,
        free_extras: zensim::feature_v2::V1FreeExtras::RawMoments,
        ..Default::default()
    };
    // The peaks-only walk: byte-identical to `v1_basic_free` except the free
    // accumulators are OFF, so the two arms differ in exactly the one thing
    // whose cost is being priced.
    let v1_basic_peaks = zensim::feature_v2::V2NewFeatureToggles {
        free_extras: zensim::feature_v2::V1FreeExtras::Off,
        ..v1_basic_free
    };

    println!(
        "# ssim2_speed_bar: RAYON_NUM_THREADS={} ssim2_rayon_feature={}",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<unset>".into()),
        cfg!(feature = "ssim2-rayon")
    );

    let result = zenbench::run(|suite| {
        for &n in &sizes {
            let (src, dst) = test_pair(n, n);
            let src_s: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
            let dst_s: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
            suite.compare(format!("ssim2_bar_{n}"), |group| {
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
                group.bench("zensim_B", move |b| {
                    b.iter(move || {
                        let s = RgbSlice::new(src_s, n, n);
                        let d = RgbSlice::new(dst_s, n, n);
                        zenbench::black_box(zb.compute(&s, &d).unwrap().score())
                    })
                });
                // ---- the PRODUCT fast-class path, which this bench had no arm
                // for (added by the kernel lane, 2026-09-05).
                //
                // Every other fast-class arm here reaches the walk by handing
                // `compute_folded720_features_streaming` a hand-built
                // `V2NewFeatureToggles`. That is the right shape for pricing an
                // extraction regime, and the wrong shape for answering "how fast
                // is the thing a user gets", because it bypasses the whole
                // routing stack a real call goes through: `Zensim::new`'s
                // per-profile `fold_engine` / `skip_unread_pools` defaults,
                // `fold_engine::is_fold_backable`'s four-condition guard (which
                // can DEGRADE to the buffered walk), `score_pool_mode`'s
                // derivation of the pool mode from the bake's own layer-0 read
                // pattern, the 372 truncation, and the bake forward + PCHIP
                // spline. `ZensimProfile::D` is the shipped fast-class profile
                // and `Zensim::compute` is the API; this arm is both, so a
                // routing regression shows up as a speed regression here rather
                // than passing unnoticed because every arm hand-built its way
                // around the router.
                //
                // It is also the arm that makes the W4 comparison honest at the
                // product level: `add156_156basic` prices a *regime*, this
                // prices the *shipped path*, and the two should track. They are
                // not redundant — `add156_156basic` requests
                // `V1PoolsMode::Off`, which `fold_engine::pools_mode_for_need`
                // never returns, so no production call can produce that walk.
                group.bench("zensim_D", move |b| {
                    b.iter(move || {
                        let s = RgbSlice::new(src_s, n, n);
                        let d = RgbSlice::new(dst_s, n, n);
                        zenbench::black_box(zd.compute(&s, &d).unwrap().score())
                    })
                });
                // ---- amended-W4: candidate = its own regime + its own forwards
                if let Some(h) = add {
                    group.bench("add156_156basic", move |b| {
                        let mut scratch = zensim::feature_v2::V2Scratch::new();
                        let mut pred = Predictor::new(&h.model);
                        let mut x: Vec<f32> = Vec::new();
                        b.iter(move || {
                            let rs = RgbSlice::new(src_s, n, n);
                            let ds = RgbSlice::new(dst_s, n, n);
                            let v2 = zf
                                .compute_folded720_features_streaming(
                                    &rs,
                                    &ds,
                                    v1_basic,
                                    &mut scratch,
                                )
                                .unwrap();
                            zenbench::black_box(h.forward(&mut pred, &mut x, v2.features()))
                        })
                    });
                }
                if let (Some(h), Some(ch)) = (add, corrhead) {
                    group.bench("add156_plus_corrhead", move |b| {
                        let mut scratch = zensim::feature_v2::V2Scratch::new();
                        let mut pred = Predictor::new(&h.model);
                        let mut cpred = Predictor::new(&ch.model);
                        let mut x: Vec<f32> = Vec::new();
                        let mut cx: Vec<f32> = Vec::new();
                        // The corruption head reads a SUBSET of the profile's own
                        // read-set (basic, or basic+peaks — both are emitted by the
                        // walk `add156_156basic` already runs), so the extraction
                        // toggles are identical and only the second forward is new.
                        b.iter(move || {
                            let rs = RgbSlice::new(src_s, n, n);
                            let ds = RgbSlice::new(dst_s, n, n);
                            let v2 = zf
                                .compute_folded720_features_streaming(
                                    &rs,
                                    &ds,
                                    v1_basic,
                                    &mut scratch,
                                )
                                .unwrap();
                            let a = h.forward(&mut pred, &mut x, v2.features());
                            let c = ch.forward(&mut cpred, &mut cx, v2.features());
                            zenbench::black_box(a + c)
                        })
                    });
                }
                if let Some(pk) = peaks {
                    group.bench("peaks156_no_raw", move |b| {
                        let mut scratch = zensim::feature_v2::V2Scratch::new();
                        let mut pred = Predictor::new(&pk.model);
                        let mut x: Vec<f32> = Vec::new();
                        b.iter(move || {
                            let rs = RgbSlice::new(src_s, n, n);
                            let ds = RgbSlice::new(dst_s, n, n);
                            let v2 = zf
                                .compute_folded720_features_streaming(
                                    &rs,
                                    &ds,
                                    v1_basic_peaks,
                                    &mut scratch,
                                )
                                .unwrap();
                            zenbench::black_box(pk.forward(&mut pred, &mut x, v2.features()))
                        })
                    });
                }
                if let Some(f) = free {
                    group.bench("free156_peaks_raw", move |b| {
                        let mut scratch = zensim::feature_v2::V2Scratch::new();
                        let mut pred = Predictor::new(&f.model);
                        let mut x: Vec<f32> = Vec::new();
                        b.iter(move || {
                            let rs = RgbSlice::new(src_s, n, n);
                            let ds = RgbSlice::new(dst_s, n, n);
                            let v2 = zf
                                .compute_folded720_features_streaming(
                                    &rs,
                                    &ds,
                                    v1_basic_free,
                                    &mut scratch,
                                )
                                .unwrap();
                            zenbench::black_box(f.forward(&mut pred, &mut x, v2.features()))
                        })
                    });
                }
                // ---- diagnostic: EXTRACTION ONLY, same entry point + toggles as the
                // two arms above, forward pass excluded (`ZEN_S2_EXTRACT_ONLY=1`).
                // Off by default so the exam's arm set is unchanged; exists to answer
                // "is a W4 anomaly in the walk or in the bake's forward pass" without
                // a second, potentially-mismatched instrument
                // (`benchmarks/profile_d_notax_2026-09-01.md` §W4 diagnosis).
                if env_usize("ZEN_S2_EXTRACT_ONLY", 0) == 1 {
                    if add.is_some() {
                        group.bench("add156_extract_only", move |b| {
                            let mut scratch = zensim::feature_v2::V2Scratch::new();
                            b.iter(move || {
                                let rs = RgbSlice::new(src_s, n, n);
                                let ds = RgbSlice::new(dst_s, n, n);
                                let v2 = zf
                                    .compute_folded720_features_streaming(
                                        &rs,
                                        &ds,
                                        v1_basic,
                                        &mut scratch,
                                    )
                                    .unwrap();
                                zenbench::black_box(
                                    v2.features().iter().fold(0.0f64, |a, &b| a + b),
                                )
                            })
                        });
                    }
                    if peaks.is_some() {
                        group.bench("peaks156_extract_only", move |b| {
                            let mut scratch = zensim::feature_v2::V2Scratch::new();
                            b.iter(move || {
                                let rs = RgbSlice::new(src_s, n, n);
                                let ds = RgbSlice::new(dst_s, n, n);
                                let v2 = zf
                                    .compute_folded720_features_streaming(
                                        &rs,
                                        &ds,
                                        v1_basic_peaks,
                                        &mut scratch,
                                    )
                                    .unwrap();
                                zenbench::black_box(
                                    v2.features().iter().fold(0.0f64, |a, &b| a + b),
                                )
                            })
                        });
                    }
                    if free.is_some() {
                        group.bench("free156_extract_only", move |b| {
                            let mut scratch = zensim::feature_v2::V2Scratch::new();
                            b.iter(move || {
                                let rs = RgbSlice::new(src_s, n, n);
                                let ds = RgbSlice::new(dst_s, n, n);
                                let v2 = zf
                                    .compute_folded720_features_streaming(
                                        &rs,
                                        &ds,
                                        v1_basic_free,
                                        &mut scratch,
                                    )
                                    .unwrap();
                                zenbench::black_box(
                                    v2.features().iter().fold(0.0f64, |a, &b| a + b),
                                )
                            })
                        });
                    }
                }
                if let Some(m) = mlp {
                    group.bench("flagship_944off", move |b| {
                        let mut scratch = zensim::feature_v2::V2Scratch::new();
                        let mut pred = Predictor::new(&m.model);
                        let mut x: Vec<f32> = Vec::new();
                        b.iter(move || {
                            let rs = RgbSlice::new(src_s, n, n);
                            let ds = RgbSlice::new(dst_s, n, n);
                            let v2 = zf
                                .compute_folded720_features_streaming(
                                    &rs,
                                    &ds,
                                    pools_off,
                                    &mut scratch,
                                )
                                .unwrap();
                            zenbench::black_box(m.forward(&mut pred, &mut x, v2.features()))
                        })
                    });
                }
                if let Some(l) = lin {
                    group.bench("q7b_944pools", move |b| {
                        let mut scratch = zensim::feature_v2::V2Scratch::new();
                        let mut pred = Predictor::new(&l.model);
                        let mut x: Vec<f32> = Vec::new();
                        b.iter(move || {
                            let rs = RgbSlice::new(src_s, n, n);
                            let ds = RgbSlice::new(dst_s, n, n);
                            let v2 = zf
                                .compute_folded720_features_streaming(
                                    &rs,
                                    &ds,
                                    pools_full,
                                    &mut scratch,
                                )
                                .unwrap();
                            zenbench::black_box(l.forward(&mut pred, &mut x, v2.features()))
                        })
                    });
                }
                if let (Some(m), Some(l)) = (mlp, lin) {
                    group.bench("hybrid_944pools", move |b| {
                        let mut scratch = zensim::feature_v2::V2Scratch::new();
                        let mut pm = Predictor::new(&m.model);
                        let mut pl = Predictor::new(&l.model);
                        let (mut xm, mut xl): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
                        b.iter(move || {
                            let rs = RgbSlice::new(src_s, n, n);
                            let ds = RgbSlice::new(dst_s, n, n);
                            let v2 = zf
                                .compute_folded720_features_streaming(
                                    &rs,
                                    &ds,
                                    pools_full,
                                    &mut scratch,
                                )
                                .unwrap();
                            let f = v2.features();
                            let a = m.forward(&mut pm, &mut xm, f);
                            let b2 = l.forward(&mut pl, &mut xl, f);
                            zenbench::black_box(0.5 * a + 0.5 * b2)
                        })
                    });
                }
            });
        }
    });
    let _ = result;
}
