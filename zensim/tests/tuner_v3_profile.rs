//! Smoke test for `PreviewV0_5TunerV3` runtime wiring.
//!
//! EXP-CROSS-CODEC-V9 (2026-05-20) shipped the V_24-per-sample-α
//! architecture extended with **post-network monotone PCHIP spline
//! calibration** via the new `zentrain.output_calibration_spline`
//! metadata. The bake bytes ship at
//! `zensim/weights/v_tuner_v9_2026-05-20.bin` (md5
//! `b50e8ca4946c1ec5bf2f5e9cf96ffdb8`); this test guards the
//! `PreviewV0_5TunerV3` variant + `ProfileParams` wiring.
//!
//! What this test covers:
//!   1. `include_bytes!` resolves at build time (bake found).
//!   2. The variant's `name()` / `params()` dispatch works.
//!   3. The runtime `forward_one_bake` dispatches through
//!      `zentrain.per_sample_alpha_head`, `zentrain.tanh_output_head`,
//!      AND `zentrain.output_calibration_spline` metadata together
//!      (no panic, finite output, in [0, 100]).
//!   4. The bake bytes are distinct from PreviewV0_5TunerV2 — the V9
//!      bake carries an output_calibration_spline payload that V2 lacks,
//!      so per-pair scores differ.
//!   5. `ZensimProfile::tuner_v3()` alias returns the V3 variant.
//!
//! What this test does NOT cover (lives elsewhere):
//!   * Cross-corpus SROCC — `bake_verdict` against canonical val parquets.
//!   * JPEG q-sweep monotonicity — `qsweep_eval` or codec smoke demo.
//!   * Anchor-band landing (JND@60, JOD@30, range [0, 100]) — the
//!     methodology doc + the zensim-target smoke demo cover these.

use zensim::{RgbSlice, Zensim, ZensimProfile};

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
fn tuner_v3_profile_name() {
    assert_eq!(
        ZensimProfile::PreviewV0_5TunerV3.name(),
        "zensim-preview-v0.5-tuner-v3"
    );
    // The const alias should return the same variant.
    assert_eq!(
        ZensimProfile::tuner_v3(),
        ZensimProfile::PreviewV0_5TunerV3
    );
}

#[test]
fn tuner_v3_score_in_range() {
    let (src, dst) = make_pair_with_delta(64, 64, 16);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z = Zensim::new(ZensimProfile::PreviewV0_5TunerV3).with_parallel(false);
    let r = z.compute(&s, &d).unwrap();

    let score = r.score();
    assert!(
        score.is_finite(),
        "tuner-v3 score is NaN/Inf: {score} \
         (per-sample-α + tanh-output-head + PCHIP-spline dispatch may be broken)"
    );
    assert!(
        (0.0..=100.0).contains(&score),
        "tuner-v3 score out of range: {score}"
    );
    assert_eq!(r.profile(), ZensimProfile::PreviewV0_5TunerV3);
}

#[test]
fn tuner_v3_score_in_range_across_distortion_levels() {
    let z = Zensim::new(ZensimProfile::PreviewV0_5TunerV3).with_parallel(false);
    for delta in (0..50).step_by(5) {
        let (src, dst) = make_pair_with_delta(64, 64, delta as u8);
        let s = RgbSlice::new(&src, 64, 64);
        let d = RgbSlice::new(&dst, 64, 64);
        let score = z.compute(&s, &d).unwrap().score();
        assert!(
            score.is_finite() && (0.0..=100.0).contains(&score),
            "tuner-v3 score out of range or NaN at delta={delta}: {score}"
        );
    }
}

#[test]
fn tuner_v3_differs_from_tuner_v2_on_typical_pair() {
    // PreviewV0_5TunerV3 carries `zentrain.output_calibration_spline`
    // metadata (post-network PCHIP spline) that PreviewV0_5TunerV2
    // does NOT have. The two bakes were also trained with different
    // anchor sets (V3 used 8-band extended-range with score ∈ [0, 100];
    // V2 used 6-band with score ∈ [10, 90]). On a typical non-trivial
    // pair the spline + different weights produce measurably different
    // scores — sanity check that include_bytes! is pointed at the
    // correct file AND that the spline metadata is being applied.
    let (src, dst) = make_pair_with_delta(64, 64, 16);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z_v3 = Zensim::new(ZensimProfile::PreviewV0_5TunerV3).with_parallel(false);
    let z_v2 = Zensim::new(ZensimProfile::PreviewV0_5TunerV2).with_parallel(false);
    let s_v3 = z_v3.compute(&s, &d).unwrap().score();
    let s_v2 = z_v2.compute(&s, &d).unwrap().score();
    assert!(
        s_v3.is_finite() && s_v2.is_finite(),
        "non-finite score(s): tuner-v3={s_v3} tuner-v2={s_v2}"
    );
    assert!(
        (s_v3 - s_v2).abs() > 0.01,
        "tuner-v3 produced identical score to tuner-v2 \
         (v3={s_v3}, v2={s_v2}); include_bytes! may point at the wrong file \
         or the PCHIP spline runtime dispatch is missing"
    );
}
