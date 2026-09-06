//! The model's raw output has ONE sign convention per run, and every loss term
//! must speak it.
//!
//! Before 2026-09-06 the trainer derived that convention at exactly one of its
//! polarity-sensitive sites (`train_mlp_strategy`'s sequential RankNet term) and
//! nowhere else. The pool head, the hybrid head, the per-sample-α head and both
//! plain-path mini-batch helpers each recomputed a bare `signum(mos_a − mos_b)`;
//! the TV within-ladder hinge and the α head's monotonicity hinge each hard-coded
//! an *opposite* assumption in prose. On the α-head path the result is a model
//! whose ordering is intact and whose sign is backwards — the fastclass2 campaign
//! measured its 2-layer α-head arm as the campaign's single best CID22 *ordering*,
//! `|−0.8921|` against the plain path's `+0.8863`, arriving as a negative number
//! that `bake_dial_refit pack` then could not spline at all (the output
//! calibration spline is monotone increasing by construction).
//!
//! These tests pin the invariant that closes it: **whatever convention a run
//! declares, every path obeys it.** Each arm trains a tiny synthetic corpus whose
//! target is a known monotone function of the features, then reads the trained
//! bake back through the same runtime `bake_verdict` scores with, and asserts the
//! SIGN of `corr(raw_output, quality)`.
//!
//! Five of these arms fail on the parent commit.

use zenpredict::{Model, Predictor};
use zensim_validate::bake_runtime::{
    extract_hybrid_head, extract_per_sample_alpha_head, score_with_bake_alloc,
};
use zensim_validate::mlp_train::{
    FeatureRows, GroupLossMode, MlpHyperparams, OutputPolarity, TrainingGroup, ValidationPolicy,
    train_mlp_strategy,
};

/// Deterministic synthetic corpus: `quality = w · x + 0.1·x0² + noise`, so a
/// model with any capacity at all can order it. Returns `(features, quality)`.
fn synthetic(n_features: usize, n_rows: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    // A tiny self-contained SplitMix64 — the trainer's RNG is crate-internal and
    // this corpus only has to be deterministic, not identical to any run's.
    let mut state = seed;
    let mut next = move || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // uniform in [0,1) then shifted to roughly N(0,1)-ish spread
        let u = (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        (u - 0.5) * 3.0
    };
    let w: Vec<f64> = (0..n_features)
        .map(|i| (i as f64 - n_features as f64 / 2.0) * 0.3)
        .collect();
    let mut feats = Vec::with_capacity(n_rows);
    let mut quality = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let x: Vec<f64> = (0..n_features).map(|_| next()).collect();
        let mut y: f64 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        y += 0.1 * x[0] * x[0];
        y += next() * 0.02;
        feats.push(x);
        quality.push(y);
    }
    // Rescale quality into [0, 1] so an absolute term has a sane target range.
    let lo = quality.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = quality.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for q in quality.iter_mut() {
        *q = (*q - lo) / (hi - lo);
    }
    (feats, quality)
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    fn ranks(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = ((i + j) as f64) / 2.0 + 1.0;
            for &k in &idx[i..=j] {
                r[k] = avg;
            }
            i = j + 1;
        }
        r
    }
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let mut num = 0.0;
    let (mut da, mut db) = (0.0, 0.0);
    for i in 0..ra.len() {
        let (x, y) = (ra[i] - ma, rb[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt())
}

/// Train one arm and return `corr(raw_output, quality)` read back through the
/// SAME runtime `bake_verdict` scores with — heads included.
fn raw_output_correlation(hyper: MlpHyperparams, loss_mode: GroupLossMode) -> f64 {
    let n_features = 12;
    let (feats, quality) = synthetic(n_features, 240, 20260906);
    let feats_ref: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let mut groups = [TrainingGroup {
        name: "synth".to_string(),
        human_scores: &quality,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode,
    }];
    let mut log = Vec::new();
    let bytes = train_mlp_strategy(
        &mut groups,
        n_features,
        &hyper,
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let model = Model::from_bytes(leaked).expect("bake loads");
    let psa = extract_per_sample_alpha_head(&model);
    let hyb = extract_hybrid_head(&model);
    let n_inputs = model.caller_input_width();
    let mut predictor = Predictor::new(&model);
    let preds: Vec<f64> = feats
        .iter()
        .map(|f| {
            score_with_bake_alloc(
                &mut predictor,
                false,
                psa.as_ref(),
                hyb.as_ref(),
                None,
                None,
                n_inputs,
                f,
            )
        })
        .collect();
    spearman(&preds, &quality)
}

fn base_hyper() -> MlpHyperparams {
    MlpHyperparams {
        n_hidden: 12,
        n_epochs: 60,
        pairs_per_epoch: 1200,
        initial_lr: 0.005,
        log_every: 1000,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    }
}

/// The owner's decision table, independent of any training.
#[test]
fn output_polarity_owner_maps_conventions_to_signs() {
    assert_eq!(OutputPolarity::Distance.rank_target_sign(), 1.0);
    assert_eq!(OutputPolarity::Score.rank_target_sign(), -1.0);
    assert_eq!(OutputPolarity::Distance.ladder_sign(), 1.0);
    assert_eq!(OutputPolarity::Score.ladder_sign(), -1.0);
    assert!(OutputPolarity::Distance.label().contains("LOWER"));
    assert!(OutputPolarity::Score.label().contains("HIGHER"));
}

/// Rank-only recipes are DISTANCE-shaped on every path. This is the legacy
/// convention every board bake trained under, and the arm that proves the fix
/// did not move it.
#[test]
fn rank_only_recipes_are_distance_shaped_on_every_path() {
    // (label, hyperparams)
    let arms: Vec<(&str, MlpHyperparams)> = vec![
        ("plain K=1", base_hyper()),
        (
            "plain K=32 (run_parallel_minibatch)",
            MlpHyperparams {
                minibatch_size: 32,
                ..base_hyper()
            },
        ),
        (
            "α head depth 1",
            MlpHyperparams {
                per_sample_alpha_head: true,
                ..base_hyper()
            },
        ),
        (
            "α head depth 2",
            MlpHyperparams {
                per_sample_alpha_head: true,
                n_hidden_layers: 2,
                ..base_hyper()
            },
        ),
        (
            "hybrid head",
            MlpHyperparams {
                hybrid_head: true,
                ..base_hyper()
            },
        ),
    ];
    for (label, hyper) in arms {
        let r = raw_output_correlation(hyper, GroupLossMode::Rank);
        assert!(
            r < -0.5,
            "{label}: a rank-only recipe is DISTANCE-shaped, so the raw output must \
             be NEGATIVELY correlated with quality; got SROCC {r:.4}"
        );
    }
}

/// An absolute term makes the run SCORE-shaped, and the rank term must follow.
///
/// The α-head arms are the ones that fail on the parent commit: that path never
/// applied `rank_target_sign`, so its RankNet term stayed DISTANCE-shaped while
/// its MSE term regressed the same output onto a score-unit target. Depth 2 is
/// where the campaign measured it (`raw CID22 −0.8921`).
#[test]
fn absolute_term_makes_every_path_score_shaped() {
    let arms: Vec<(&str, MlpHyperparams, GroupLossMode)> = vec![
        (
            "plain K=1 :both",
            MlpHyperparams {
                mse_weight: 1.0,
                ..base_hyper()
            },
            GroupLossMode::Both,
        ),
        (
            "α head depth 1 --mse-weight",
            MlpHyperparams {
                per_sample_alpha_head: true,
                mse_weight: 1.0,
                ..base_hyper()
            },
            GroupLossMode::Rank,
        ),
        (
            "α head depth 2 --mse-weight",
            MlpHyperparams {
                per_sample_alpha_head: true,
                n_hidden_layers: 2,
                mse_weight: 1.0,
                ..base_hyper()
            },
            GroupLossMode::Rank,
        ),
    ];
    for (label, hyper, mode) in arms {
        let r = raw_output_correlation(hyper, mode);
        assert!(
            r > 0.5,
            "{label}: an absolute term makes the run SCORE-shaped, so the raw output \
             must be POSITIVELY correlated with quality; got SROCC {r:.4}. A large \
             NEGATIVE value here is the defect this test exists for: the ordering is \
             learned and the sign is backwards."
        );
    }
}

/// The brief's requested proof, stated in the codebase's own convention: the α
/// head must carry the SAME polarity as the plain path at the same settings, at
/// both depths.
#[test]
fn alpha_head_raw_output_polarity_matches_plain_path_at_depth_1_and_2() {
    for (label, mse_weight, mode, want_positive) in [
        ("rank-only", 0.0, GroupLossMode::Rank, false),
        ("absolute", 1.0, GroupLossMode::Both, true),
    ] {
        let plain = raw_output_correlation(
            MlpHyperparams {
                mse_weight,
                ..base_hyper()
            },
            mode,
        );
        for depth in [1usize, 2usize] {
            let alpha = raw_output_correlation(
                MlpHyperparams {
                    per_sample_alpha_head: true,
                    n_hidden_layers: depth,
                    mse_weight,
                    ..base_hyper()
                },
                mode,
            );
            assert_eq!(
                plain > 0.0,
                alpha > 0.0,
                "{label}: α head at depth {depth} disagrees with the plain path about \
                 which way the raw output points — plain {plain:.4}, α {alpha:.4}"
            );
            assert_eq!(
                alpha > 0.0,
                want_positive,
                "{label}: α head at depth {depth} carries the wrong convention \
                 (SROCC {alpha:.4})"
            );
        }
    }
}

/// `run_parallel_minibatch` and `run_minibatch_with_nin` implement RankNet only.
/// Routing an absolute term through them silently discarded it; refuse instead.
#[test]
#[should_panic(expected = "absolute (mse/both) term is active")]
fn absolute_term_with_parallel_minibatch_is_refused() {
    raw_output_correlation(
        MlpHyperparams {
            mse_weight: 1.0,
            minibatch_size: 32,
            ..base_hyper()
        },
        GroupLossMode::Both,
    );
}

#[test]
#[should_panic(expected = "absolute (mse/both) term is active")]
fn absolute_term_with_norm_in_norm_is_refused() {
    raw_output_correlation(
        MlpHyperparams {
            mse_weight: 1.0,
            minibatch_size: 16,
            norm_in_norm_weight: 0.1,
            ..base_hyper()
        },
        GroupLossMode::Both,
    );
}
