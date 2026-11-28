#![cfg(feature = "zenpsy-api")]
//! End-to-end contract tests for `zensim`'s `zenpsy-api` integration.
//!
//! Covers:
//! - Score-only default (zero diffmap allocation) vs opt-in `Native` / `"diffmap"`
//! - Unknown named map rejection (`ErrorKind::NoMap`) before computation
//! - Bit-exact `f64::to_bits()` parity across cold `score`, warm `prepare_reference`
//!   -> `score`, native `Zensim::compute`, SIMD tiers, and serial vs parallel
//! - Exact `100.0` identity parity on warm & cold paths, including ignored X lanes,
//!   RGB under zero alpha, and Opaque-vs-Straight alpha distinction
//! - Sub-8px (`1x1`, `3x5`, `7x7`) and odd strided inputs
//! - Cooperative cancellation during reference preparation and candidate scoring,
//!   plus clean reuse of `PreparedZensim` after a cancelled evaluation
//! - Strict SDR ingress refusals (HDR, DiffuseWhite, HdrMetadata, premultiplied,
//!   pending orientation, BT.709 transfer, unknown primaries, narrow range, NaN f32)

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use archmage::testing::{CompileTimePolicy, for_each_token_permutation, lock_token_testing};
use zenpixels::{
    AlphaMode as ZpAlpha, ColorSpace, ContentLightLevel, DiffuseWhite, HdrMetadata, Orientation,
    PixelFormat as ZpPixelFormat, PixelView, Primaries as ZpPrimaries, SampleLayout, SignalRange,
    Transfer, formats,
};
use zenpsy_api::how_far::{Stop, StopOnly, StopReason};
use zenpsy_api::{Direction, ErrorKind, EvalOpts, Evaluation, Metric, prepare_reference, score};
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn synth_rgb(w: u32, h: u32, distort: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 29 + y * 13 + 17) % 256) as u8;
            let g = ((x * 7 + y * 37 + 53) % 256) as u8;
            let b = ((x ^ y).wrapping_mul(19) % 256) as u8;
            if distort {
                out.push(r.saturating_add(4));
                out.push(g.saturating_sub(2));
                out.push(b.saturating_add(3));
            } else {
                out.push(r);
                out.push(g);
                out.push(b);
            }
        }
    }
    out
}

fn synth_strided_rgb(w: u32, h: u32, extra_pad: usize, distort: bool) -> (Vec<u8>, usize) {
    let packed = synth_rgb(w, h, distort);
    let row_len = w as usize * 3;
    let stride = row_len + extra_pad;
    let mut out = vec![0xE5u8; stride * h as usize];
    for y in 0..h as usize {
        out[y * stride..y * stride + row_len]
            .copy_from_slice(&packed[y * row_len..(y + 1) * row_len]);
    }
    (out, stride)
}

#[derive(Clone)]
struct CountingPulse {
    checks: Arc<AtomicUsize>,
    trip_after: usize,
}

impl CountingPulse {
    fn new(trip_after: usize) -> Self {
        Self {
            checks: Arc::new(AtomicUsize::new(0)),
            trip_after,
        }
    }

    fn checks(&self) -> usize {
        self.checks.load(Ordering::SeqCst)
    }
}

impl Stop for CountingPulse {
    fn check(&self) -> Result<(), StopReason> {
        let n = self.checks.fetch_add(1, Ordering::SeqCst);
        if n >= self.trip_after {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn metric_id_carries_zensim_a_rev7_provenance() {
    let z = Zensim::new(ZensimProfile::A).with_parallel(false);
    let id = z.id();
    assert_eq!(id.family(), "zensim");
    assert_eq!(id.variant(), Some("zensim-a"));
    assert_eq!(id.implementation().crate_name(), "zensim");
    assert_eq!(id.implementation().score_revision(), 7);
    assert_eq!(id.config_value("profile"), Some("zensim-a"));
    assert_eq!(id.config_value("formula_revision"), Some("rev7"));
    assert_eq!(
        id.config_value("bake"),
        Some("rev7_mix_h128_2026-11-03.bin")
    );
    assert_eq!(id.config_value("parallel"), Some("false"));
}

#[test]
fn score_only_default_and_opt_in_diffmap_contract() {
    let mut z = Zensim::new(ZensimProfile::A).with_parallel(false);
    let src_bytes = synth_rgb(65, 49, false);
    let dst_bytes = synth_rgb(65, 49, true);
    let src = PixelView::new(&src_bytes, 65, 49, 65 * 3, &formats::RGB8_SRGB).unwrap();
    let dst = PixelView::new(&dst_bytes, 65, 49, 65 * 3, &formats::RGB8_SRGB).unwrap();

    // Default EvalOpts::new() is strictly score-only.
    let eval = score(&mut z, &src, &dst, EvalOpts::new()).unwrap();
    assert!(
        eval.diffmap().is_none(),
        "score-only must not allocate diffmap"
    );
    assert_eq!(eval.maps().count(), 0);
    let scores: Vec<_> = eval.scores().collect();
    assert_eq!(scores.len(), 1);
    assert_eq!(scores[0].spec.key(), "zensim");
    assert_eq!(scores[0].spec.direction(), Direction::Higher);
    assert_eq!(scores[0].spec.perfect(), Some(100.0));
    assert_eq!(scores[0].value.to_bits(), eval.score().to_bits());

    // Native native & named "diffmap" return the exact same headline score.
    let eval_native = score(&mut z, &src, &dst, EvalOpts::new().with_native_map()).unwrap();
    let eval_named = score(&mut z, &src, &dst, EvalOpts::new().with_map("diffmap")).unwrap();
    assert_eq!(eval_native.score().to_bits(), eval.score().to_bits());
    assert_eq!(eval_named.score().to_bits(), eval.score().to_bits());
    let maps: Vec<_> = eval_named.maps().collect();
    assert_eq!(maps.len(), 1);
    assert_eq!(maps[0].spec.key(), "diffmap");
    assert_eq!(maps[0].spec.direction(), Direction::Lower);
    assert_eq!(maps[0].spec.perfect(), Some(0.0));
    assert_eq!(maps[0].values.width(), 65);
    assert_eq!(maps[0].values.height(), 49);
    assert_eq!(maps[0].grid.input_width(), 65);
    assert_eq!(maps[0].grid.input_height(), 49);

    // Unknown named map fails immediately with NoMap.
    let err = score(&mut z, &src, &dst, EvalOpts::new().with_map("nonexistent")).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::NoMap);
}

#[test]
fn cold_and_warm_bit_exact_parity_across_sizes_stride_and_tiers() {
    let _token = lock_token_testing();
    let report = for_each_token_permutation(CompileTimePolicy::Fail, |perm| {
        for &(w, h) in &[(1, 1), (3, 5), (7, 7), (31, 17), (65, 63)] {
            let (src_bytes, src_stride) = synth_strided_rgb(w, h, 11, false);
            let (dst_bytes, dst_stride) = synth_strided_rgb(w, h, 19, true);
            let src = PixelView::new(&src_bytes, w, h, src_stride, &formats::RGB8_SRGB).unwrap();
            let dst = PixelView::new(&dst_bytes, w, h, dst_stride, &formats::RGB8_SRGB).unwrap();

            let packed_src = synth_rgb(w, h, false);
            let packed_dst = synth_rgb(w, h, true);
            let native_src: Vec<[u8; 3]> = packed_src.as_chunks::<3>().0.to_vec();
            let native_dst: Vec<[u8; 3]> = packed_dst.as_chunks::<3>().0.to_vec();

            for parallel in [false, true] {
                let mut z = Zensim::new(ZensimProfile::A).with_parallel(parallel);
                let native = z
                    .compute(
                        &RgbSlice::new(&native_src, w as usize, h as usize),
                        &RgbSlice::new(&native_dst, w as usize, h as usize),
                    )
                    .unwrap();
                let cold_id = score(&mut z, &src, &src, EvalOpts::new()).unwrap();
                let cold_dst = score(&mut z, &src, &dst, EvalOpts::new()).unwrap();

                // Prepare owned reference (survives dropping factory `z`).
                let mut prepared = prepare_reference(&mut z, &src, EvalOpts::new()).unwrap();
                drop(z);

                let warm_id = prepared.score(&src, EvalOpts::new()).unwrap();
                let warm_dst = prepared.score(&dst, EvalOpts::new()).unwrap();
                let warm_dst_again = prepared.score(&dst, EvalOpts::new()).unwrap();

                let ctx = format!("{w}x{h} parallel={parallel} {}", perm.label);
                assert_eq!(
                    cold_id.score().to_bits(),
                    100.0f64.to_bits(),
                    "cold id {ctx}"
                );
                assert_eq!(
                    warm_id.score().to_bits(),
                    100.0f64.to_bits(),
                    "warm id {ctx}"
                );
                assert_eq!(
                    cold_id.raw_distance().to_bits(),
                    0.0f64.to_bits(),
                    "cold id raw {ctx}"
                );
                assert_eq!(
                    warm_id.raw_distance().to_bits(),
                    0.0f64.to_bits(),
                    "warm id raw {ctx}"
                );
                assert_eq!(
                    cold_dst.score().to_bits(),
                    native.score().to_bits(),
                    "cold vs native {ctx}"
                );
                assert_eq!(
                    warm_dst.score().to_bits(),
                    native.score().to_bits(),
                    "warm vs native {ctx}"
                );
                assert_eq!(
                    warm_dst_again.score().to_bits(),
                    native.score().to_bits(),
                    "reused scratch {ctx}"
                );
                assert_eq!(
                    warm_dst.raw_distance().to_bits(),
                    cold_dst.raw_distance().to_bits(),
                    "warm vs cold raw {ctx}"
                );
            }
        }
    });
    assert!(report.permutations_run > 0);
}

#[test]
fn identity_handles_x_padding_zero_alpha_and_opaque_vs_straight() {
    let mut z = Zensim::new(ZensimProfile::A).with_parallel(false);
    let (w, h) = (16u32, 16u32);

    // 1. RGBX8 with different padding bytes in 4th lane MUST score exactly 100.0 (cold & warm).
    let mut a_rgbx = vec![0u8; (w * h * 4) as usize];
    let mut b_rgbx = vec![0u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        a_rgbx[i * 4..i * 4 + 4].copy_from_slice(&[(i % 251) as u8, 80, 160, 0x00]);
        b_rgbx[i * 4..i * 4 + 4].copy_from_slice(&[(i % 251) as u8, 80, 160, 0xFF]);
    }
    let va = PixelView::new(&a_rgbx, w, h, w as usize * 4, &formats::RGBX8_SRGB).unwrap();
    let vb = PixelView::new(&b_rgbx, w, h, w as usize * 4, &formats::RGBX8_SRGB).unwrap();
    assert_eq!(
        score(&mut z, &va, &vb, EvalOpts::new()).unwrap().score(),
        100.0
    );
    let mut prep = prepare_reference(&mut z, &va, EvalOpts::new()).unwrap();
    assert_eq!(prep.score(&vb, EvalOpts::new()).unwrap().score(), 100.0);

    // 2. Straight RGBA where both sides have A=0 but different RGB MUST score exactly 100.0.
    let mut a_rgba = vec![0u8; (w * h * 4) as usize];
    let mut b_rgba = vec![0u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        if i % 2 == 0 {
            a_rgba[i * 4..i * 4 + 4].copy_from_slice(&[10, 20, 30, 0]);
            b_rgba[i * 4..i * 4 + 4].copy_from_slice(&[200, 150, 90, 0]);
        } else {
            a_rgba[i * 4..i * 4 + 4].copy_from_slice(&[100, 120, 140, 180]);
            b_rgba[i * 4..i * 4 + 4].copy_from_slice(&[100, 120, 140, 180]);
        }
    }
    let va = PixelView::new(&a_rgba, w, h, w as usize * 4, &formats::RGBA8_SRGB).unwrap();
    let vb = PixelView::new(&b_rgba, w, h, w as usize * 4, &formats::RGBA8_SRGB).unwrap();
    assert_eq!(
        score(&mut z, &va, &vb, EvalOpts::new()).unwrap().score(),
        100.0
    );
    let mut prep = prepare_reference(&mut z, &va, EvalOpts::new()).unwrap();
    assert_eq!(prep.score(&vb, EvalOpts::new()).unwrap().score(), 100.0);

    // 3. Opaque RGBA vs Straight RGBA with identical bytes (A=128) MUST NOT falsely score 100.0!
    let mut semi = vec![0u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        semi[i * 4..i * 4 + 4].copy_from_slice(&[240, 30, 10, 128]);
    }
    let opaque_fmt = ZpPixelFormat::new(
        SampleLayout::RGBA8,
        ColorSpace::Rgb {
            primaries: ZpPrimaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: ZpAlpha::Opaque,
            white: None,
        },
    );
    let v_opaque = PixelView::new(&semi, w, h, w as usize * 4, &opaque_fmt).unwrap();
    let v_straight = PixelView::new(&semi, w, h, w as usize * 4, &formats::RGBA8_SRGB).unwrap();
    let cold_mixed = score(&mut z, &v_opaque, &v_straight, EvalOpts::new()).unwrap();
    let mut prep_opaque = prepare_reference(&mut z, &v_opaque, EvalOpts::new()).unwrap();
    let warm_mixed = prep_opaque.score(&v_straight, EvalOpts::new()).unwrap();
    assert!(
        cold_mixed.score() < 95.0,
        "Opaque vs semi-transparent Straight must not falsely report 100.0, got {}",
        cold_mixed.score()
    );
    assert_eq!(cold_mixed.score().to_bits(), warm_mixed.score().to_bits());
}

/// Fires `StopReason::TimedOut` ONCE on the exact `trip_at` check, then
/// returns `Ok(())` on all subsequent checks. Proves that internal latching
/// preserves the first-observed `StopReason` even if downstream checks would
/// otherwise succeed or return a different reason.
#[derive(Clone)]
struct OneShotTimeoutStop {
    checks: Arc<AtomicUsize>,
    trip_at: usize,
}

impl OneShotTimeoutStop {
    fn new(trip_at: usize) -> Self {
        Self {
            checks: Arc::new(AtomicUsize::new(0)),
            trip_at,
        }
    }

    fn checks(&self) -> usize {
        self.checks.load(Ordering::SeqCst)
    }
}

impl Stop for OneShotTimeoutStop {
    fn check(&self) -> Result<(), StopReason> {
        let n = self.checks.fetch_add(1, Ordering::SeqCst);
        if n == self.trip_at {
            Err(StopReason::TimedOut)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_during_prepare_and_scoring_and_scratch_reuse() {
    let mut z = Zensim::new(ZensimProfile::A).with_parallel(false);
    let (w, h) = (256u32, 256u32);
    let src_bytes = synth_rgb(w, h, false);
    let dst_bytes = synth_rgb(w, h, true);
    let src = PixelView::new(&src_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();
    let dst = PixelView::new(&dst_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();

    // Baseline uncancelled prepare + score + diffmap counts.
    let baseline_prep_stop = CountingPulse::new(usize::MAX);
    let baseline_prep_pulse = StopOnly::new(baseline_prep_stop.clone());
    let mut prepared = prepare_reference(
        &mut z,
        &src,
        EvalOpts::new().with_pulse(&baseline_prep_pulse),
    )
    .unwrap();
    let full_prep_checks = baseline_prep_stop.checks();
    assert!(
        full_prep_checks > 4,
        "prepare_reference should poll cancellation across phases, saw {full_prep_checks}"
    );

    let baseline_score_stop = CountingPulse::new(usize::MAX);
    let baseline_score_pulse = StopOnly::new(baseline_score_stop.clone());
    let baseline_score = prepared
        .score(&dst, EvalOpts::new().with_pulse(&baseline_score_pulse))
        .unwrap()
        .score();
    let full_score_checks = baseline_score_stop.checks();
    assert!(
        full_score_checks > 4,
        "warm score should poll cancellation across candidate conversion + fold, saw {full_score_checks}"
    );

    let baseline_map_stop = CountingPulse::new(usize::MAX);
    let baseline_map_pulse = StopOnly::new(baseline_map_stop.clone());
    let baseline_map_eval = prepared
        .score(
            &dst,
            EvalOpts::new()
                .with_map("diffmap")
                .with_pulse(&baseline_map_pulse),
        )
        .unwrap();
    let full_map_checks = baseline_map_stop.checks();
    assert!(
        full_map_checks > full_score_checks,
        "diffmap path should additionally poll stop during diffmap fusion ({full_map_checks} vs {full_score_checks})"
    );
    assert_eq!(
        baseline_map_eval.score().to_bits(),
        baseline_score.to_bits()
    );

    // 1. Mid-preparation one-shot TimedOut must abort early and preserve TimedOut.
    let trip_prep_stop = OneShotTimeoutStop::new(full_prep_checks / 2);
    let trip_prep = StopOnly::new(trip_prep_stop.clone());
    let err = prepare_reference(&mut z, &src, EvalOpts::new().with_pulse(&trip_prep))
        .err()
        .expect("mid-prepare TimedOut must fail");
    assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));
    assert!(trip_prep_stop.checks() < full_prep_checks);

    // 2. Mid-fold one-shot TimedOut on warm reference must preserve TimedOut.
    let trip_score_stop = OneShotTimeoutStop::new(full_score_checks / 2);
    let trip_score = StopOnly::new(trip_score_stop.clone());
    let err = prepared
        .score(&dst, EvalOpts::new().with_pulse(&trip_score))
        .unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));
    assert!(trip_score_stop.checks() < full_score_checks);

    // 3. Mid-diffmap one-shot TimedOut (after score fold checks) must preserve TimedOut.
    let diffmap_trip = (full_score_checks + full_map_checks) / 2;
    let trip_map_stop = OneShotTimeoutStop::new(diffmap_trip);
    let trip_map = StopOnly::new(trip_map_stop.clone());
    let err = prepared
        .score(
            &dst,
            EvalOpts::new().with_map("diffmap").with_pulse(&trip_map),
        )
        .unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));
    assert!(trip_map_stop.checks() < full_map_checks);

    // 4. Cold score one-shot TimedOut must also abort early and preserve TimedOut.
    let cold_trip = OneShotTimeoutStop::new(2);
    let cold_pulse = StopOnly::new(cold_trip);
    let err = score(&mut z, &src, &dst, EvalOpts::new().with_pulse(&cold_pulse)).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));

    // 5. Reuse the SAME PreparedZensim & scratch after cancellation — must be bit-exact!
    let recovered = prepared.score(&dst, EvalOpts::new()).unwrap().score();
    assert_eq!(recovered.to_bits(), baseline_score.to_bits());
}

#[test]
fn reflect_padding_expansion_respects_max_pixels_budget() {
    // 1 × 2000 = 2000 pixels fits under max_pixels=10_000, but 4-scale reflect
    // padding expands to 64 × 2000 = 128_000 pixels which MUST be rejected
    // before allocation instead of silently exceeding max_pixels.
    let mut z = Zensim::new(ZensimProfile::A)
        .with_parallel(false)
        .with_max_pixels(10_000);
    let (w, h) = (1u32, 2000u32);
    let src_bytes = synth_rgb(w, h, false);
    let dst_bytes = synth_rgb(w, h, true);
    let src = PixelView::new(&src_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();
    let dst = PixelView::new(&dst_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();

    // Cold score path rejects expanded reflect pad.
    let err = score(&mut z, &src, &dst, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::Resource);

    // Warm prepare_reference path also rejects expanded reflect pad.
    let err = prepare_reference(&mut z, &src, EvalOpts::new())
        .err()
        .expect("reflect-padded prepare exceeding max_pixels must fail");
    assert_eq!(err.error().kind(), ErrorKind::Resource);
}

#[test]
fn eval_opts_max_pixels_counts_active_pixels_excluding_reflect_scratch() {
    // Per zenpsy-api contract, EvalOpts::with_max_pixels counts active input
    // pixels only, excluding internal 64x64 reflect-padding scratch.
    let mut z = Zensim::new(ZensimProfile::A).with_parallel(false);
    let one_px = synth_rgb(1, 1, false);
    let one = PixelView::new(&one_px, 1, 1, 3, &formats::RGB8_SRGB).unwrap();

    let cold = score(&mut z, &one, &one, EvalOpts::new().with_max_pixels(1)).unwrap();
    assert_eq!(cold.score(), 100.0);

    let mut prep = prepare_reference(&mut z, &one, EvalOpts::new().with_max_pixels(1)).unwrap();
    let warm = prep
        .score(&one, EvalOpts::new().with_max_pixels(1))
        .unwrap();
    assert_eq!(warm.score(), 100.0);

    // 2x1 exceeds active cap = 1 -> rejected as Resource.
    let two_px = synth_rgb(2, 1, false);
    let two = PixelView::new(&two_px, 2, 1, 6, &formats::RGB8_SRGB).unwrap();
    let err = score(&mut z, &two, &two, EvalOpts::new().with_max_pixels(1)).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::Resource);
    let err = prepare_reference(&mut z, &two, EvalOpts::new().with_max_pixels(1))
        .err()
        .expect("2x1 prepare with active max_pixels=1 must fail");
    assert_eq!(err.error().kind(), ErrorKind::Resource);
}

#[test]
fn configured_stop_cancels_strict_ingress_even_with_default_eval_opts() {
    // Even with EvalOpts::new() (no call-level pulse), a configured stop on
    // Zensim must fire BEFORE strict f32 ingress scans pixel rows.
    let f32_px = vec![[0.5f32, 0.5, 0.5, 1.0]; 128 * 128];
    let view = PixelView::from_typed(&f32_px, 128, 128, &formats::RGBAF32_LINEAR).unwrap();

    // Cold score
    let stop_cold = OneShotTimeoutStop::new(0);
    let mut z_cold = Zensim::new(ZensimProfile::A)
        .with_parallel(false)
        .with_stop(stop_cold.clone());
    let err = score(&mut z_cold, &view, &view, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));
    assert_eq!(
        stop_cold.checks(),
        1,
        "must abort at first check before f32 scan"
    );

    // Warm prepare_reference
    let stop_prep = OneShotTimeoutStop::new(0);
    let mut z_prep = Zensim::new(ZensimProfile::A)
        .with_parallel(false)
        .with_stop(stop_prep.clone());
    let err = prepare_reference(&mut z_prep, &view, EvalOpts::new())
        .err()
        .expect("cancelled configured stop must fail warm_reference");
    assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));
    assert_eq!(
        stop_prep.checks(),
        1,
        "must abort at first check before f32 scan"
    );

    // Warm score: build reference uncancelled with a shared AtomicBool stop, then
    // arm the configured stop before warm scoring with EvalOpts::new().
    let warm_armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let warm_checks = Arc::new(AtomicUsize::new(0));
    #[derive(Clone)]
    struct ArmedTimeoutStop(Arc<std::sync::atomic::AtomicBool>, Arc<AtomicUsize>);
    impl Stop for ArmedTimeoutStop {
        fn check(&self) -> Result<(), StopReason> {
            self.1.fetch_add(1, Ordering::SeqCst);
            if self.0.load(Ordering::SeqCst) {
                Err(StopReason::TimedOut)
            } else {
                Ok(())
            }
        }
    }
    let mut z_warm = Zensim::new(ZensimProfile::A)
        .with_parallel(false)
        .with_stop(ArmedTimeoutStop(warm_armed.clone(), warm_checks.clone()));
    let mut prep_warm = prepare_reference(&mut z_warm, &view, EvalOpts::new()).unwrap();
    warm_checks.store(0, Ordering::SeqCst);
    warm_armed.store(true, Ordering::SeqCst);
    let err = prep_warm.score(&view, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));
    assert_eq!(
        warm_checks.load(Ordering::SeqCst),
        1,
        "warm score must abort at first check before candidate f32 scan"
    );
}

#[test]
fn planar_diffmap_one_shot_timeout_at_fusion_does_not_return_empty_or_panic() {
    use zensim::{DiffmapOptions, ZensimError};

    for (w, h) in [(64usize, 64usize), (31, 65)] {
        let src_rgb: Vec<[u8; 3]> = synth_rgb(w as u32, h as u32, false)
            .as_chunks::<3>()
            .0
            .to_vec();
        let src = RgbSlice::new(&src_rgb, w, h);
        let plane_r: Vec<f32> = (0..w * h).map(|i| ((i * 7) % 256) as f32 / 255.0).collect();
        let plane_g: Vec<f32> = (0..w * h)
            .map(|i| ((i * 13) % 256) as f32 / 255.0)
            .collect();
        let plane_b: Vec<f32> = (0..w * h)
            .map(|i| ((i * 29) % 256) as f32 / 255.0)
            .collect();
        let planes = [plane_r.as_slice(), plane_g.as_slice(), plane_b.as_slice()];

        let z_base = Zensim::new(ZensimProfile::A).with_parallel(false);
        let precomputed = z_base.precompute_reference(&src).unwrap();

        // Count checks across a full uncancelled planar diffmap run.
        let counter = CountingPulse::new(usize::MAX);
        let z_count = Zensim::new(ZensimProfile::A)
            .with_parallel(false)
            .with_stop(counter.clone());
        let ok = z_count
            .compute_with_ref_and_diffmap_linear_planar(
                &precomputed,
                planes,
                w,
                h,
                w,
                DiffmapOptions::default(),
            )
            .unwrap();
        assert_eq!(ok.diffmap().len(), w * h);
        let total = counter.checks();
        assert!(total > 3, "expected multiple stop polls, saw {total}");

        // Fire a one-shot TimedOut at EVERY single poll index across the entire
        // planar diffmap pipeline (including scale fusion and finish crop) —
        // NONE may return Ok (empty diffmap) or panic!
        for trip in 0..total {
            let z_trip = Zensim::new(ZensimProfile::A)
                .with_parallel(false)
                .with_stop(OneShotTimeoutStop::new(trip));
            let err = z_trip
                .compute_with_ref_and_diffmap_linear_planar(
                    &precomputed,
                    planes,
                    w,
                    h,
                    w,
                    DiffmapOptions::default(),
                )
                .err()
                .expect("one-shot TimedOut must not return Ok with empty diffmap");
            assert!(
                matches!(
                    err,
                    ZensimError::Cancelled {
                        reason: StopReason::TimedOut
                    }
                ),
                "w={w} h={h} trip={trip}/{total}: expected TimedOut, got {err:?}"
            );
        }
    }
}

#[test]
fn strict_sdr_ingress_rejects_unsupported_inputs() {
    let mut z = Zensim::new(ZensimProfile::A).with_parallel(false);
    let rgb = synth_rgb(16, 16, false);
    let valid = PixelView::new(&rgb, 16, 16, 16 * 3, &formats::RGB8_SRGB).unwrap();

    // 1. Premultiplied alpha rejected by zenpsy-api strict ingress.
    let rgba = vec![128u8; 16 * 16 * 4];
    let premul_fmt = ZpPixelFormat::new(
        SampleLayout::RGBA8,
        ColorSpace::Rgb {
            primaries: ZpPrimaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: ZpAlpha::Premultiplied,
            white: None,
        },
    );
    let premul = PixelView::new(&rgba, 16, 16, 16 * 4, &premul_fmt).unwrap();
    let err = score(&mut z, &premul, &premul, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // 2. Pending orientation rejected.
    let rotated = valid.clone().with_orientation(Orientation::Rotate90);
    let err = score(&mut z, &rotated, &rotated, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // 3. HDR metadata rejected.
    let with_hdr = valid.clone().with_hdr(HdrMetadata {
        content_light_level: Some(ContentLightLevel::new(1000, 400)),
        mastering: None,
    });
    let err = score(&mut z, &with_hdr, &with_hdr, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // 4. DiffuseWhite on Linear rejected.
    let f32_px = vec![[0.5f32, 0.5, 0.5, 1.0]; 16 * 16];
    let white_fmt = ZpPixelFormat::new(
        SampleLayout::RGBAF32,
        ColorSpace::Rgb {
            primaries: ZpPrimaries::Bt709,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: ZpAlpha::Opaque,
            white: Some(DiffuseWhite::BT2408),
        },
    );
    let white_view = PixelView::from_typed(&f32_px, 16, 16, &white_fmt).unwrap();
    let err = score(&mut z, &white_view, &white_view, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // 5. NaN linear f32 rejected as InvalidInput.
    let mut nan_px = f32_px;
    nan_px[0][0] = f32::NAN;
    let nan_view = PixelView::from_typed(&nan_px, 16, 16, &formats::RGBAF32_LINEAR).unwrap();
    let err = score(&mut z, &nan_view, &nan_view, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::InvalidInput);
}
