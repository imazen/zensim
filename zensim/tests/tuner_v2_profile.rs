#![allow(deprecated)]
//! Smoke test for `PreviewV0_5TunerV2` runtime wiring.
//!
//! EXP-CROSS-CODEC-V6 (2026-05-19) shipped the V_24-per-sample-α
//! architecture with multi-band anchor pressure (W=1.0, step_p=0.30)
//! and tanh-output-head scale=15.0. The bake bytes ship at
//! `zensim/weights/v_tuner_v6_2026-05-19.bin`; this test guards the
//! `PreviewV0_5TunerV2` variant + `ProfileParams` wiring.
//!
//! What this test covers:
//!   1. `include_bytes!` resolves at build time.
//!   2. The variant's `name()` / `params()` dispatch works.
//!   3. The runtime `forward_one_bake` dispatches through both
//!      `zentrain.per_sample_alpha_head` AND `zentrain.tanh_output_head`
//!      metadata (no panic, finite output, in [0, 100]).
//!   4. The bake bytes are distinct from PreviewV0_5Tuner / CrossCodec.
//!
//! What this test does NOT cover (lives elsewhere):
//!   * Cross-corpus SROCC — `bake_verdict` against canonical val parquets.
//!   * Per-band cross-codec parity — `eval_v6_multi_band_check.py`.
//!   * JPEG q-sweep monotonicity — `qsweep_eval`.

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
fn tuner_v2_profile_name() {
    assert_eq!(
        ZensimProfile::PreviewV0_5TunerV2.name(),
        "zensim-preview-v0.5-tuner-v2"
    );
}

#[test]
fn tuner_v2_score_in_range() {
    let (src, dst) = make_pair_with_delta(64, 64, 16);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z = Zensim::new(ZensimProfile::PreviewV0_5TunerV2).with_parallel(false);
    let r = z.compute(&s, &d).unwrap();

    let score = r.score();
    assert!(
        score.is_finite(),
        "tuner-v2 score is NaN/Inf: {score} (per-sample-α + tanh-output-head dispatch may be broken)"
    );
    assert!(
        (0.0..=100.0).contains(&score),
        "tuner-v2 score out of range: {score}"
    );
    assert_eq!(r.profile(), ZensimProfile::PreviewV0_5TunerV2);
}

#[test]
fn tuner_v2_score_in_range_across_distortion_levels() {
    let z = Zensim::new(ZensimProfile::PreviewV0_5TunerV2).with_parallel(false);
    for delta in (0..50).step_by(5) {
        let (src, dst) = make_pair_with_delta(64, 64, delta as u8);
        let s = RgbSlice::new(&src, 64, 64);
        let d = RgbSlice::new(&dst, 64, 64);
        let score = z.compute(&s, &d).unwrap().score();
        assert!(
            score.is_finite() && (0.0..=100.0).contains(&score),
            "tuner-v2 score out of range or NaN at delta={delta}: {score}"
        );
    }
}

#[test]
fn tuner_v2_differs_from_tuner_and_cross_codec_on_typical_pair() {
    // PreviewV0_5TunerV2, PreviewV0_5Tuner, and PreviewV0_5CrossCodec all
    // share the per-sample-α 372→128→128 architecture but were trained
    // with different loss recipes. Weights and the tanh-output-head
    // metadata payload differ. On a non-trivial pair scores should
    // differ measurably from at least one of the other two bakes —
    // sanity check that the include_bytes! is pointed at the right file.
    let (src, dst) = make_pair_with_delta(64, 64, 16);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z_v2 = Zensim::new(ZensimProfile::PreviewV0_5TunerV2).with_parallel(false);
    let z_t = Zensim::new(ZensimProfile::PreviewV0_5Tuner).with_parallel(false);
    let z_cc = Zensim::new(ZensimProfile::PreviewV0_5CrossCodec).with_parallel(false);
    let s_v2 = z_v2.compute(&s, &d).unwrap().score();
    let s_t = z_t.compute(&s, &d).unwrap().score();
    let s_cc = z_cc.compute(&s, &d).unwrap().score();
    assert!(
        s_v2.is_finite() && s_t.is_finite() && s_cc.is_finite(),
        "non-finite score(s): tuner-v2={s_v2} tuner={s_t} cross-codec={s_cc}"
    );
    let differs_from_tuner = (s_v2 - s_t).abs() > 0.01;
    let differs_from_cc = (s_v2 - s_cc).abs() > 0.01;
    assert!(
        differs_from_tuner || differs_from_cc,
        "tuner-v2 produced identical scores to BOTH tuner and cross-codec \
         (v2={s_v2}, tuner={s_t}, cc={s_cc}); include_bytes! may point at \
         the wrong file or runtime dispatch is missing the tanh-pin path"
    );
}
