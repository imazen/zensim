//! Per-group loss mode (2026-07-15) — `GroupLossMode::{Rank, Mse, Both}`.
//!
//! **Why this exists.** A corpus's target column decides what it can teach.
//! `safesyn`/`bigcodec`/`kadis` carry an ssim2-derived score on a shared
//! cross-image scale, so an absolute regression target is meaningful. The
//! near-lossless HF corpus does not: its ssim2 ladder moves ~0.92 pts
//! within an image against ~6 pts between images, so an absolute term
//! there fits between-image noise — it can only be consumed as rank.
//!
//! Before this, the plain (non-α-head) path was RankNet-only for EVERY
//! group and `--mse-weight` was rejected outright, so "MSE on the main
//! groups, rank-only on HF" — the round-7 recipe — was not expressible in
//! Rust at all. That is not academic: it is the measured reason
//! `benchmarks/r7_hf_rust_reproduction_2026-07-15.md` reproduced round-7's
//! CID22 (−0.0047 vs −0.0041) and non-photo (−0.0016 vs −0.0017) deltas
//! while INVERTING KonJND (−0.035 vs +0.033). A rank-only objective has no
//! absolute dial for KonJND — the most calibration-sensitive corpus — to
//! track.
//!
//! The property under test is therefore the DIAL, not the ranking: both
//! modes should rank, but only a mode carrying the absolute term should
//! land on the target's actual scale.

use zenpredict::{Model, Predictor};
use zensim_validate::mlp_train::{
    GroupLossMode, MlpHyperparams, TrainingGroup, ValidationPolicy, train_mlp,
};
// Stat math lives in zenstats -- never re-rolled locally, not even in a test.
use zenstats::panel::spearman;

const N_FEATURES: usize = 8;
const N_ROWS: usize = 240;

/// Rows whose target is a known affine function of the features, on an
/// absolute 0..100 scale. `score = 100 − 95·t` for `t` swept over [0,1),
/// with the feature vector a deterministic function of `t`.
fn synth_rows() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut feats = Vec::with_capacity(N_ROWS);
    let mut scores = Vec::with_capacity(N_ROWS);
    for i in 0..N_ROWS {
        let t = i as f64 / N_ROWS as f64;
        feats.push(
            (0..N_FEATURES)
                .map(|d| t * (1.0 + (d as f64 * 0.13).sin().abs()))
                .collect::<Vec<f64>>(),
        );
        scores.push(100.0 - 95.0 * t);
    }
    (feats, scores)
}

fn train(mode: GroupLossMode, mse_weight: f64) -> Vec<u8> {
    let (feats, scores) = synth_rows();
    let refs: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &scores,
        features: &refs,
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: mode,
    };
    let hp = MlpHyperparams {
        n_hidden: 16,
        n_epochs: 400,
        pairs_per_epoch: 400,
        initial_lr: 5e-3,
        seed: 3,
        mse_weight,
        log_every: 10_000,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        parallel_batch: false,
        ..Default::default()
    };
    let mut log = Vec::new();
    train_mlp(std::slice::from_ref(&group), N_FEATURES, &hp, &mut log)
}

fn predictions(bake: &[u8]) -> Vec<f64> {
    let model = Model::from_bytes(bake).expect("bake parses");
    let mut p = Predictor::new(&model);
    let (feats, _) = synth_rows();
    feats
        .iter()
        .map(|f| {
            let f32s: Vec<f32> = f.iter().map(|&v| v as f32).collect();
            p.predict(&f32s).unwrap()[0] as f64
        })
        .collect()
}

/// Mean |prediction − target|, in score units.
fn mean_abs_err(preds: &[f64], targets: &[f64]) -> f64 {
    preds
        .iter()
        .zip(targets)
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / preds.len() as f64
}

/// THE contract: an `Mse` group puts the model on the target's absolute
/// scale. A `Rank` group cannot — RankNet is scale-free, so its output is
/// an arbitrary monotone transform of quality.
#[test]
fn mse_mode_learns_the_absolute_dial_and_rank_mode_does_not() {
    let (_, targets) = synth_rows();

    let mse_err = mean_abs_err(&predictions(&train(GroupLossMode::Mse, 1.0)), &targets);
    let rank_err = mean_abs_err(&predictions(&train(GroupLossMode::Rank, 0.0)), &targets);

    // Targets span 5..100. A model on the right scale should land within a
    // few score units; a scale-free one has no reason to be close at all.
    assert!(
        mse_err < 10.0,
        "GroupLossMode::Mse should track the absolute target; mean|err| = {mse_err:.2} score units"
    );
    assert!(
        rank_err > mse_err * 2.0,
        "GroupLossMode::Rank is scale-free and must NOT land on the absolute \
         scale — if it does, the MSE term is leaking into rank-only groups. \
         rank mean|err| = {rank_err:.2}, mse mean|err| = {mse_err:.2}"
    );
}

/// `Both` carries the absolute term too, so it also lands on-scale — at a
/// weight that balances against the rank term (~0.1; see the normalization
/// note at the plain path's MSE block).
#[test]
fn both_mode_also_learns_the_absolute_dial() {
    let (_, targets) = synth_rows();
    let err = mean_abs_err(&predictions(&train(GroupLossMode::Both, 0.1)), &targets);
    assert!(
        err < 10.0,
        "GroupLossMode::Both should track the absolute target; mean|err| = {err:.2} score units"
    );
}

/// Every mode must still RANK — the absolute term is additive to the rank
/// term, not a replacement for the ordering the metric exists to produce.
#[test]
fn every_mode_still_recovers_the_ranking() {
    let (_, targets) = synth_rows();
    for (mode, w) in [
        (GroupLossMode::Rank, 0.0),
        (GroupLossMode::Mse, 1.0),
        (GroupLossMode::Both, 0.1),
    ] {
        let preds = predictions(&train(mode, w));
        // Rank-only output is a distance (anti-correlated with quality);
        // MSE-bearing output is score-shaped. Compare on |correlation| so
        // the assertion is about ordering, not polarity.
        let rho = spearman(&preds, &targets).abs();
        assert!(
            rho > 0.85,
            "{mode:?} failed to recover the ranking: |SROCC| = {rho:.4}"
        );
    }
}

/// `--mse-weight` with no group opted in is always a mistake: on the plain
/// path the term is per-group, so such a run silently trains pure rank and
/// ignores the flag. It must fail loud instead.
#[test]
#[should_panic(expected = "no group opted into an absolute term")]
fn mse_weight_without_an_opted_in_group_panics() {
    let _ = train(GroupLossMode::Rank, 1.0);
}

/// POLARITY. The model has ONE output, so the rank term and the absolute
/// term must agree on what it means. They do not by default: the legacy
/// RankNet is distance-shaped (higher quality → LOWER y) while regression
/// onto `human_score` is score-shaped (higher quality → HIGHER y). Mixed
/// without reconciliation, the rank group's own corpus INVERTS.
///
/// MEASURED 2026-07-15 on the round-7 recipe before the fix: the HF group's
/// held-out per-ref SROCC went +0.6393 / 6% backwards (no HF group) to
/// −0.3454 / 75% backwards (HF group added) — adding rank supervision to a
/// corpus made that corpus rank BACKWARDS, which is only possible if the two
/// terms are fighting.
///
/// So: a `Both` group must rank the SAME DIRECTION as an `Mse` group. If this
/// fails with a negative correlation, the polarity reconciliation is gone.
#[test]
fn rank_and_absolute_terms_agree_on_polarity() {
    let (_, targets) = synth_rows();

    // Signed, not |·|: the sign IS the property under test.
    let mse_rho = spearman(&predictions(&train(GroupLossMode::Mse, 1.0)), &targets);
    let both_rho = spearman(&predictions(&train(GroupLossMode::Both, 0.1)), &targets);

    assert!(
        mse_rho > 0.85,
        "an Mse group is score-shaped: higher quality must give HIGHER output; \
         signed SROCC = {mse_rho:.4}"
    );
    assert!(
        both_rho > 0.85,
        "a Both group carries BOTH terms — if its rank term still used the \
         legacy distance polarity it would fight the absolute term and invert. \
         signed SROCC = {both_rho:.4} (expected same sign as Mse's {mse_rho:.4})"
    );
}

/// The reconciliation must NOT touch rank-only training: with no group
/// carrying an absolute term, the legacy distance convention stands (that is
/// what every existing recipe and every shipped bake's calibration assumes).
#[test]
fn rank_only_training_keeps_the_legacy_distance_polarity() {
    let (_, targets) = synth_rows();
    let rho = spearman(&predictions(&train(GroupLossMode::Rank, 0.0)), &targets);
    assert!(
        rho < -0.85,
        "rank-only output is DISTANCE-shaped: higher quality must give LOWER \
         output. signed SROCC = {rho:.4}. If this flipped positive, the \
         polarity reconciliation is leaking into rank-only runs and every \
         existing bake's calibration is now inverted."
    );
}
