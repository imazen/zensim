//! Smoke test for `PreviewV0_5CompressionV2` runtime wiring.
//!
//! Task #177 (2026-05-20) ships the Compression bake
//! (`v_compression_persample_2026-05-18.bin`, V_24-per-sample-α s4)
//! extended with a **post-network monotone PCHIP spline calibration**
//! in the new `zentrain.output_calibration_spline` metadata key. The
//! re-baked bytes ship at `zensim/weights/v_compression_v2_2026-05-20.bin`;
//! this test guards the `PreviewV0_5CompressionV2` variant +
//! `ProfileParams` wiring.
//!
//! What this test covers:
//!   1. `include_bytes!` resolves at build time (bake found).
//!   2. The variant's `name()` / `params()` dispatch works.
//!   3. The runtime `forward_one_bake` dispatches through
//!      `zentrain.output_calibration_spline` metadata on a
//!      per-sample-α-head MLP (the spline applies AFTER the α-mix)
//!      without panicking.
//!   4. The bake bytes differ from PreviewV0_5Compression (the
//!      underlying MLP weights are the same, but the spline metadata
//!      produces a different score; sanity check that `include_bytes!`
//!      is pointed at the correct file AND that the spline runtime
//!      dispatch is active for a per-sample-α-head bake).
//!   5. `zensim_experimental::preview_v0_5_compression_v2()` alias returns the V2 variant.
//!
//! What this test does NOT cover (lives elsewhere):
//!   * Cross-corpus SROCC — `bake_verdict` against canonical val parquets,
//!     persisted at `benchmarks/v_compression_v2_2026-05-20_verdict.md`.
//!   * JND/JOD anchor landing (60.000 / 30.000 exact) — verified at
//!     calibration time, persisted at the calibration log.
//!   * Cross-codec consistency — V9-style smoke demo at
//!     `benchmarks/v_compression_v2_2026-05-20_cross_codec.log`. The
//!     Compression trail was not trained with cross-codec equivalence
//!     pairs, so this is not a primary gate.

use zensim::{RgbSlice, Zensim};

fn make_pair_with_delta(w: usize, h: usize, delta: u8) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = w * h;
    let src: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = ((i % w) * 255 / w) as u8;
            let y = ((i / w) * 255 / h) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| {
            [
                r.saturating_add(delta),
                g.saturating_add(delta.saturating_sub(2)),
                b.saturating_add(delta / 2),
            ]
        })
        .collect();
    (src, dst)
}

#[test]
fn compression_v2_profile_name() {
    assert_eq!(
        zensim_experimental::preview_v0_5_compression_v2().name(),
        "zensim-preview-v0.5-compression-v2"
    );
    assert_eq!(
        zensim_experimental::preview_v0_5_compression_v2(),
        zensim_experimental::preview_v0_5_compression_v2()
    );
}

#[test]
fn compression_v2_score_in_range() {
    let (src, dst) = make_pair_with_delta(64, 64, 16);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z = Zensim::new(zensim_experimental::preview_v0_5_compression_v2()).with_parallel(false);
    let r = z.compute(&s, &d).unwrap();

    let score = r.score();
    assert!(
        score.is_finite(),
        "compression-v2 score is NaN/Inf: {score} \
         (PCHIP-spline runtime dispatch on a per-sample-α-head MLP \
          may be broken)"
    );
    assert!(
        (0.0..=100.0).contains(&score),
        "compression-v2 score out of range: {score}"
    );
    assert_eq!(
        r.profile(),
        zensim_experimental::preview_v0_5_compression_v2()
    );
}

#[test]
fn compression_v2_score_in_range_across_distortion_levels() {
    let z = Zensim::new(zensim_experimental::preview_v0_5_compression_v2()).with_parallel(false);
    for delta in (0..50).step_by(5) {
        let (src, dst) = make_pair_with_delta(64, 64, delta as u8);
        let s = RgbSlice::new(&src, 64, 64);
        let d = RgbSlice::new(&dst, 64, 64);
        let score = z.compute(&s, &d).unwrap().score();
        assert!(
            score.is_finite() && (0.0..=100.0).contains(&score),
            "compression-v2 score out of range or NaN at delta={delta}: {score}"
        );
    }
}

#[test]
fn compression_v2_differs_from_compression_base() {
    // PreviewV0_5CompressionV2 wraps the same underlying MLP +
    // per-sample-α-head metadata as PreviewV0_5Compression + a
    // `zentrain.output_calibration_spline` metadata entry. The
    // Compression base's per-sample-α-mixed output is
    // distance-shaped (high raw = low quality, range ≈ [-27, 20] on
    // the V9 anchor parquet) and the production `soft_clamp_score`
    // was squashing the dial into ≈ [2, 18]. The spline maps the
    // distance-shaped mixed output onto the dial-honest [0, 100]
    // scale, so the V2 score MUST differ from the base on any
    // non-trivial pair.
    let (src, dst) = make_pair_with_delta(64, 64, 16);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z_v2 = Zensim::new(zensim_experimental::preview_v0_5_compression_v2()).with_parallel(false);
    let z_base = Zensim::new(zensim_experimental::preview_v0_5_compression()).with_parallel(false);
    let s_v2 = z_v2.compute(&s, &d).unwrap().score();
    let s_base = z_base.compute(&s, &d).unwrap().score();
    assert!(
        s_v2.is_finite() && s_base.is_finite(),
        "non-finite score(s): compression-v2={s_v2} compression-base={s_base}"
    );
    assert!(
        (s_v2 - s_base).abs() > 0.01,
        "compression-v2 produced identical score to compression-base \
         (v2={s_v2}, base={s_base}); include_bytes! may point at the wrong file \
         or the PCHIP spline runtime dispatch is missing for per-sample-α-head bakes"
    );
}

#[test]
fn compression_v2_identity_short_circuit_preserved() {
    // The V0_5 identity-image short-circuit (fix at metric.rs:1634)
    // must still produce score=100 on byte-identical pairs even with
    // the spline metadata in the bake (the short-circuit fires
    // BEFORE apply_mlp_scoring, so this is a regression check on the
    // short-circuit path's interaction with the spline-aware
    // forward_one_bake dispatching through the per-sample-α head).
    let (src, _dst) = make_pair_with_delta(64, 64, 0);
    let s = RgbSlice::new(&src, 64, 64);
    let z = Zensim::new(zensim_experimental::preview_v0_5_compression_v2()).with_parallel(false);
    let r = z.compute(&s, &s).unwrap();
    assert_eq!(
        r.score(),
        100.0,
        "compression-v2 identity short-circuit broken: identical pair scored {}",
        r.score()
    );
}
