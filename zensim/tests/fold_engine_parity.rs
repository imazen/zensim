//! **The stage-2 gate of the fold-engine lane** (design note:
//! `benchmarks/fold_engine_2026-08-31.md`): every scoring entry, run through
//! the streaming fold, must return a `ZensimResult` **bit-identical** to the
//! buffered walk's.
//!
//! ## Why bit-identical and not "within the golden tolerance"
//!
//! The golden gate's `max(1e-6 abs, 1e-5·scale)` tolerance exists for
//! cross-*environment* drift — the measured 241-246-of-372 divergence between
//! CPU-vendor classes (`benchmarks/v1_golden_env_triage_2026-08-05.md`). It is
//! not a licence for two code paths on the SAME box to disagree. Since option
//! C (`blur::pyramid_plane_stride(w) == w`) the fold and the buffered walk
//! compute the same statistic over the same pixels, and
//! `feature_v2::tests::v1_372_bit_exact_to_fold_at_every_width` already proves
//! the feature block agrees to the bit at 19 geometries. This file extends
//! that from the FEATURE VECTOR to the whole `ZensimResult` — `score`,
//! `raw_distance`, every feature, and `mean_offset` — across the product
//! entry points, so a fold-backed `compute()` is a substitution rather than
//! an approximation.
//!
//! Anything the fold cannot serve bit-identically must FALL BACK to buffered
//! rather than answer differently; `fold_engine::is_fold_backable` owns that
//! predicate and `fold_falls_back_on_a_weight_skipping_profile` is its gate.

#![cfg(all(feature = "training", feature = "feature-regime-v2"))]

mod common;

use zensim::fold_engine::ScoringEngine;
use zensim::{RgbSlice, Zensim, ZensimProfile, ZensimResult};

fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let r = common::generators::gen_value_noise(w, h, 0xC0FFEE);
    let d = common::generators::distort_block_artifacts(&r, w, h);
    (r, d)
}

fn load_png_rgb8(path: &std::path::Path) -> (Vec<[u8; 3]>, usize, usize) {
    use zenpixels::ChannelType;
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let cfg = zenpng::PngDecodeConfig::default();
    let out = zenpng::decode(&bytes, &cfg, &enough::Unstoppable).expect("zenpng decode");
    let (w, h) = (out.info.width as usize, out.info.height as usize);
    let desc = out.pixels.descriptor();
    assert_eq!(desc.channel_type(), ChannelType::U8);
    let slice = out.pixels.as_slice();
    let (channels, has_alpha) = (desc.channels(), desc.has_alpha());
    let mut rgb = Vec::with_capacity(w * h);
    for y in 0..h as u32 {
        let row = slice.row(y);
        match (channels, has_alpha) {
            (4, true) => {
                for px in row.as_chunks::<4>().0.iter().take(w) {
                    rgb.push([px[0], px[1], px[2]]);
                }
            }
            (3, false) => {
                for px in row.as_chunks::<3>().0.iter().take(w) {
                    rgb.push([px[0], px[1], px[2]]);
                }
            }
            other => panic!("unsupported PNG channel layout {other:?}"),
        }
    }
    (rgb, w, h)
}

/// The real-photo golden fixture (96×96 gb82 `city.png` / `city_q50.jpg`
/// crop) — the same bytes `v1_golden_bytes::GOLDEN_REAL` pins.
fn golden_real() -> (Vec<[u8; 3]>, Vec<[u8; 3]>, usize, usize) {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let (r, rw, rh) = load_png_rgb8(&manifest.join("tests/fixtures/v1_golden_real_ref.png"));
    let (d, dw, dh) = load_png_rgb8(&manifest.join("tests/fixtures/v1_golden_real_dist.png"));
    assert_eq!((rw, rh), (dw, dh), "fixture dimension mismatch");
    (r, d, rw, rh)
}

/// Every field of a `ZensimResult`, to the bit. `to_bits()` rather than `==`
/// so a NaN can never pass as "equal" and a `-0.0`/`0.0` swap is caught.
#[track_caller]
fn assert_result_bit_identical(ctx: &str, buffered: &ZensimResult, fold: &ZensimResult) {
    assert_eq!(
        buffered.score().to_bits(),
        fold.score().to_bits(),
        "{ctx}: score buffered {:.17e} vs fold {:.17e}",
        buffered.score(),
        fold.score()
    );
    assert_eq!(
        buffered.raw_distance().to_bits(),
        fold.raw_distance().to_bits(),
        "{ctx}: raw_distance buffered {:.17e} vs fold {:.17e}",
        buffered.raw_distance(),
        fold.raw_distance()
    );
    let (bf, ff) = (buffered.features(), fold.features());
    assert_eq!(
        bf.len(),
        ff.len(),
        "{ctx}: feature width buffered {} vs fold {}",
        bf.len(),
        ff.len()
    );
    let mut bad = Vec::new();
    for (i, (&b, &f)) in bf.iter().zip(ff.iter()).enumerate() {
        if b.to_bits() != f.to_bits() {
            bad.push((i, b, f));
        }
    }
    assert!(
        bad.is_empty(),
        "{ctx}: {} of {} features differ between the buffered walk and the fold. \
         These paths compute the same statistic over the same pixels since option C — \
         a difference here means padding was reintroduced, a summation grouping moved, \
         or the fold was asked for a request it cannot serve. Do NOT widen this to a \
         tolerance. First 12: {:?}",
        bad.len(),
        bf.len(),
        &bad[..bad.len().min(12)]
    );
    for c in 0..3 {
        assert_eq!(
            buffered.mean_offset()[c].to_bits(),
            fold.mean_offset()[c].to_bits(),
            "{ctx}: mean_offset[{c}] buffered {:.17e} vs fold {:.17e} — the fold's \
             per-row decomposition must reproduce compute_xyb_mean_offset's 64-row \
             chunk reduction EXACTLY, not to an epsilon",
            buffered.mean_offset()[c],
            fold.mean_offset()[c]
        );
    }
}

/// The geometry matrix. `v1_372_bit_exact_to_fold_at_every_width`'s cells
/// (tight, non-tight even, non-tight odd, and the three `h = 93` cells that
/// were the last residual under the option-A pre-pad workaround) plus the two
/// procedural golden geometries and a sub-64 cell that exercises the shared
/// reflect-pad.
const CELLS: &[(usize, usize)] = &[
    // the procedural golden fixtures
    (64, 64),
    (200, 150),
    // formerly "tight"
    (96, 64),
    (208, 144),
    (592, 80),
    (128, 93),
    // formerly divergent — even, non-tight
    (200, 151),
    (576, 96),
    (100, 96),
    // formerly divergent — odd, non-tight
    (127, 64),
    (129, 96),
    (255, 96),
    (577, 80),
    // the h = 93 cells
    (126, 93),
    (127, 93),
    (255, 93),
    // sub-64: the SHARED reflect-pad runs before either walk
    (48, 40),
    (17, 96),
];

fn run_both<F>(profile: ZensimProfile, parallel: bool, f: F) -> (ZensimResult, ZensimResult)
where
    F: Fn(&Zensim) -> ZensimResult,
{
    let buffered = Zensim::new(profile).with_parallel(parallel);
    let fold = Zensim::new(profile)
        .with_parallel(parallel)
        .with_engine(ScoringEngine::Fold);
    assert_eq!(buffered.engine(), ScoringEngine::Buffered);
    assert_eq!(fold.engine(), ScoringEngine::Fold);
    (f(&buffered), f(&fold))
}

/// `Zensim::compute` — the product entry, profile B (372 inputs, MLP/linear
/// bake + PCHIP output spline + extrapolate disposition). The bake forward
/// and the spline are SHARED code that runs after both walks, so this test
/// covers the whole chain, not just extraction.
#[test]
fn compute_is_bit_identical_across_engines() {
    for &parallel in &[false, true] {
        for &(w, h) in CELLS {
            let (r, d) = pair(w, h);
            let (b, f) = run_both(ZensimProfile::codec_target(), parallel, |z| {
                z.compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                    .expect("compute")
            });
            assert_result_bit_identical(&format!("compute B {w}x{h} par={parallel}"), &b, &f);
        }
    }
}

/// `Zensim::compute_extended_features` at both widths the shipped profiles
/// produce: 372 (profile B — extended + IW) and 300 (`PreviewV0_2` —
/// extended, no IW). The 300 case is the §2.4 prefix claim under test: the
/// fold emits 372 and truncates, so if v1's narrow layout were NOT a prefix
/// of its wide one this fails here.
#[test]
fn extended_features_are_bit_identical_across_engines() {
    for &(profile, want) in &[
        (ZensimProfile::codec_target(), 372usize),
        (ZensimProfile::PreviewV0_2, 300usize),
    ] {
        for &parallel in &[false, true] {
            for &(w, h) in CELLS {
                let (r, d) = pair(w, h);
                let (b, f) = run_both(profile, parallel, |z| {
                    z.compute_extended_features(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                        .expect("compute_extended_features")
                });
                assert_eq!(b.features().len(), want, "{profile:?} {w}x{h}: buffered width");
                assert_result_bit_identical(
                    &format!("extended {profile:?} {w}x{h} par={parallel}"),
                    &b,
                    &f,
                );
            }
        }
    }
}

/// The REAL-photo golden fixture, both entries. Procedural noise and a real
/// JPEG-artefacted photo have very different statistics; the golden set keeps
/// both for that reason and so does this gate.
#[test]
fn golden_real_fixture_is_bit_identical_across_engines() {
    let (r, d, w, h) = golden_real();
    for &parallel in &[false, true] {
        let (b, f) = run_both(ZensimProfile::codec_target(), parallel, |z| {
            z.compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                .expect("compute")
        });
        assert_result_bit_identical(&format!("GOLDEN_REAL compute par={parallel}"), &b, &f);
        let (b, f) = run_both(ZensimProfile::codec_target(), parallel, |z| {
            z.compute_extended_features(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                .expect("compute_extended_features")
        });
        assert_result_bit_identical(&format!("GOLDEN_REAL extended par={parallel}"), &b, &f);
    }
}

/// The fold-backed result must not depend on how many threads rayon happens
/// to have — the same invariant `v1_372_is_bit_identical_across_rayon_pool_sizes`
/// states for the buffered path. Both engines are checked in the same loop so
/// a failure names which one moved.
#[cfg(feature = "threads")]
#[test]
fn both_engines_are_bit_identical_across_rayon_pool_sizes() {
    for &(w, h) in &[(256usize, 256usize), (96, 320), (320, 96), (200, 150)] {
        let (r, d) = pair(w, h);
        let mut first: Option<(ZensimResult, ZensimResult)> = None;
        for threads in [1usize, 2, 3, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build rayon pool");
            let (b, f) = pool.install(|| {
                run_both(ZensimProfile::codec_target(), true, |z| {
                    z.compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                        .expect("compute")
                })
            });
            // Cross-engine parity holds at every thread count.
            assert_result_bit_identical(&format!("{w}x{h} @ {threads}T"), &b, &f);
            match &first {
                None => first = Some((b, f)),
                Some((b0, f0)) => {
                    assert_result_bit_identical(
                        &format!("{w}x{h}: BUFFERED moved between 1 and {threads} threads"),
                        b0,
                        &b,
                    );
                    assert_result_bit_identical(
                        &format!("{w}x{h}: FOLD moved between 1 and {threads} threads"),
                        f0,
                        &f,
                    );
                }
            }
        }
    }
}

/// The fallback contract. `PreviewV0_2`'s plain `compute()` config is
/// weight-skipping (`compute_all_features == false`, `extended_features ==
/// false`), which `streaming::active_channels` honours by leaving
/// zero-weight channels' slots at their default while the fold always
/// computes all three. `is_fold_backable` rejects that config, so asking for
/// the fold must return the BUFFERED answer — identical, not merely
/// score-equal.
#[test]
fn fold_falls_back_on_a_weight_skipping_profile() {
    for &(w, h) in &[(200usize, 150usize), (256, 256)] {
        let (r, d) = pair(w, h);
        let (b, f) = run_both(ZensimProfile::PreviewV0_2, true, |z| {
            z.compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                .expect("compute")
        });
        assert_eq!(b.features().len(), 228, "{w}x{h}: PreviewV0_2 is 228-wide");
        assert_result_bit_identical(&format!("fallback PreviewV0_2 {w}x{h}"), &b, &f);
    }
}

/// The identical-pair short-circuit is SHARED (it runs in
/// `compute_with_config_core` before either walk), so both engines must
/// return the same `mark_identical` payload — score exactly 100, an all-zero
/// feature vector of the config's width, and a zero mean_offset.
#[test]
fn identical_pair_short_circuit_is_shared() {
    let (w, h) = (200usize, 150usize);
    let (r, _) = pair(w, h);
    let (b, f) = run_both(ZensimProfile::codec_target(), true, |z| {
        z.compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&r, w, h))
            .expect("compute")
    });
    assert_eq!(b.score(), 100.0);
    assert_result_bit_identical("identical pair", &b, &f);
}

/// Classification rides `Zensim::compute`, so it is fold-backed for free —
/// stated as its own gate so a future refactor that gives `classify` its own
/// walk cannot silently skip the parity check. The delta-stats half is a pure
/// per-pixel pass over the two sources with no pyramid, so it is
/// engine-independent by construction; asserted anyway.
#[cfg(feature = "classification")]
#[test]
fn classify_is_bit_identical_across_engines() {
    for &(w, h) in &[(200usize, 150usize), (96, 64), (127, 93)] {
        let (r, d) = pair(w, h);
        let buffered = Zensim::new(ZensimProfile::codec_target());
        let fold = Zensim::new(ZensimProfile::codec_target()).with_engine(ScoringEngine::Fold);
        let cb = buffered
            .classify(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
            .expect("classify");
        let cf = fold
            .classify(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
            .expect("classify");
        assert_result_bit_identical(&format!("classify {w}x{h}"), &cb.result, &cf.result);
        assert_eq!(
            format!("{:?}", cb.classification),
            format!("{:?}", cf.classification),
            "{w}x{h}: classification differs between engines"
        );
    }
}
