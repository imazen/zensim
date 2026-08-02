//! Issue #48: cooperative cancellation via `enough::Stop`.
//!
//! Verifies that `Zensim::with_stop`:
//! - rejects work when the token is already fired (pre-flight),
//! - actually interrupts a walk mid-flight (fewer checkpoints executed
//!   than an uncancelled run — scales after the trip are abandoned),
//! - never perturbs results when the token doesn't fire,
//! - covers the with_ref / PU-linear / diffmap entry points too.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use zensim::{RgbSlice, Stop, StopReason, Unstoppable, Zensim, ZensimError, ZensimProfile};

/// Counting stop token: records every `check()`, trips permanently once
/// `trip_after` checks have happened (`usize::MAX` = never trips).
#[derive(Clone)]
struct CountingStop {
    checks: Arc<AtomicUsize>,
    trip_after: usize,
}

impl CountingStop {
    fn new(trip_after: usize) -> Self {
        Self {
            checks: Arc::new(AtomicUsize::new(0)),
            trip_after,
        }
    }
    fn checks_seen(&self) -> usize {
        self.checks.load(Ordering::SeqCst)
    }
}

impl Stop for CountingStop {
    fn check(&self) -> Result<(), StopReason> {
        let n = self.checks.fetch_add(1, Ordering::SeqCst);
        if n >= self.trip_after {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// A deterministic non-identical 256×256 pair (gradient + mild distortion).
fn test_pair() -> (Vec<[u8; 3]>, Vec<[u8; 3]>, usize, usize) {
    let (w, h) = (256usize, 256usize);
    let mut src = Vec::with_capacity(w * h);
    let mut dst = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let r = (x * 255 / (w - 1)) as u8;
            let g = (y * 255 / (h - 1)) as u8;
            let b = ((x + y) % 256) as u8;
            src.push([r, g, b]);
            // Mild deterministic distortion so the identical-pair
            // short-circuit doesn't trigger.
            dst.push([r.saturating_add(3), g, b.saturating_sub(2)]);
        }
    }
    (src, dst, w, h)
}

#[test]
fn pre_cancelled_token_rejects_compute() {
    let (src, dst, w, h) = test_pair();
    let z = Zensim::new(ZensimProfile::codec_target()).with_stop(CountingStop::new(0));
    let err = z
        .compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h))
        .unwrap_err();
    assert!(
        matches!(
            err,
            ZensimError::Cancelled {
                reason: StopReason::Cancelled
            }
        ),
        "expected Cancelled, got {err:?}"
    );
}

#[test]
fn cancellation_interrupts_mid_walk() {
    let (src, dst, w, h) = test_pair();
    // Serial mode so checkpoint counts are deterministic.
    let baseline_stop = CountingStop::new(usize::MAX);
    let z = Zensim::new(ZensimProfile::codec_target())
        .with_parallel(false)
        .with_stop(baseline_stop.clone());
    let ok = z.compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h));
    assert!(ok.is_ok(), "never-tripping token must not fail: {ok:?}");
    let full_checks = baseline_stop.checks_seen();
    // The walk must actually hit multiple checkpoints (pre-flight +
    // per-scale + per-band + post-walk).
    assert!(
        full_checks > 4,
        "expected several checkpoints on a 256x256 walk, saw {full_checks}"
    );

    // Trip after the first successful check: the walk must abandon the
    // remaining scales, so strictly fewer checkpoints run than in the
    // uncancelled walk.
    let tripping = CountingStop::new(1);
    let z = Zensim::new(ZensimProfile::codec_target())
        .with_parallel(false)
        .with_stop(tripping.clone());
    let err = z
        .compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h))
        .unwrap_err();
    assert!(matches!(err, ZensimError::Cancelled { .. }));
    let cancelled_checks = tripping.checks_seen();
    assert!(
        cancelled_checks < full_checks,
        "cancelled walk must exit early: saw {cancelled_checks} checks vs {full_checks} uncancelled"
    );
}

#[test]
fn never_tripping_token_scores_identically() {
    let (src, dst, w, h) = test_pair();
    let plain = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
    let stopped = Zensim::new(ZensimProfile::codec_target())
        .with_parallel(false)
        .with_stop(CountingStop::new(usize::MAX));
    let a = plain
        .compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h))
        .unwrap();
    let b = stopped
        .compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h))
        .unwrap();
    assert_eq!(
        a.score().to_bits(),
        b.score().to_bits(),
        "an unfired stop token must not perturb the score"
    );
}

#[test]
fn unstoppable_is_stored_as_no_op() {
    let (src, dst, w, h) = test_pair();
    let z = Zensim::new(ZensimProfile::codec_target()).with_stop(Unstoppable);
    // Must succeed (Unstoppable can never fire; may_stop() == false skips
    // storage entirely).
    let r = z.compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h));
    assert!(r.is_ok(), "Unstoppable must never cancel: {r:?}");
}

#[test]
fn with_ref_entry_is_cancellable() {
    let (src, dst, w, h) = test_pair();
    let z = Zensim::new(ZensimProfile::codec_target());
    let pre = z.precompute_reference(&RgbSlice::new(&src, w, h)).unwrap();
    let z = z.with_stop(CountingStop::new(0));
    let err = z
        .compute_with_ref(&pre, &RgbSlice::new(&dst, w, h))
        .unwrap_err();
    assert!(matches!(err, ZensimError::Cancelled { .. }), "{err:?}");
}

#[test]
fn pu_linear_entry_is_cancellable() {
    let (w, h) = (128usize, 128usize);
    // Absolute-luminance linear RGB (cd/m²), interleaved.
    let mut r = vec![0.0f32; w * h * 3];
    let mut d = vec![0.0f32; w * h * 3];
    for i in 0..w * h {
        let v = 5.0 + (i % 97) as f32;
        r[i * 3] = v;
        r[i * 3 + 1] = v * 0.9;
        r[i * 3 + 2] = v * 0.8;
        d[i * 3] = v * 1.05;
        d[i * 3 + 1] = v * 0.9;
        d[i * 3 + 2] = v * 0.78;
    }
    let z = Zensim::new(ZensimProfile::codec_target()).with_stop(CountingStop::new(0));
    let err = z.compute_pu_linear(&r, &d, w, h, w * 3, w * 3).unwrap_err();
    assert!(matches!(err, ZensimError::Cancelled { .. }), "{err:?}");
}

#[test]
fn diffmap_entry_is_cancellable() {
    let (src, dst, w, h) = test_pair();
    let z = Zensim::new(ZensimProfile::codec_target());
    let pre = z.precompute_reference(&RgbSlice::new(&src, w, h)).unwrap();
    let z = z.with_stop(CountingStop::new(0));
    let r = z.compute_with_ref_and_diffmap(
        &pre,
        &RgbSlice::new(&dst, w, h),
        zensim::DiffmapWeighting::Trained,
    );
    match r {
        Err(err) => assert!(matches!(err, ZensimError::Cancelled { .. }), "{err:?}"),
        Ok(_) => panic!("pre-cancelled diffmap compute must fail"),
    }
}
