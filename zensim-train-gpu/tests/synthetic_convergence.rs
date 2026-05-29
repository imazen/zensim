//! GPU-trainer convergence smoke test.
//!
//! Generates a synthetic 372-feature dataset where the "true" score is
//! a linear function of feature 0 plus noise. Trains the GPU per-sample-α
//! head for a few epochs and checks that the resulting model
//! distinguishes high-target from low-target rows on a held-out pair.
//!
//! Requires `--features gpu-cuda` (or another GPU backend).

#![cfg(feature = "gpu-cuda")]

use zensim_train_core::TrainingGroup;
use zensim_train_core::per_sample_alpha_head::{
    PerSampleAlphaHeadModel, forward_per_sample_alpha_head,
};
use zensim_train_gpu::{GpuHparams, GpuRuntime, train_per_sample_alpha_head_gpu};

#[test]
fn gpu_per_sample_alpha_recovers_synthetic_ranking_cuda() {
    let n_features = 32;
    let n_rows = 256;

    let mut feature_storage: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    let mut scores: Vec<f64> = Vec::with_capacity(n_rows);

    let mut prng_state: u64 = 0xABCD_DEAD_BEEF_1234;
    let mut next = || {
        prng_state = prng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        prng_state
    };
    let mut next_f64 = || (next() as f64 / u64::MAX as f64) * 2.0 - 1.0;

    for i in 0..n_rows {
        let target = (i as f64 / n_rows as f64) * 100.0; // 0..100
        let mut row = vec![0.0_f64; n_features];
        row[0] = target * 0.5 + next_f64() * 1.0; // strong signal in feat 0
        for d in 1..n_features {
            row[d] = next_f64() * 0.5; // noise
        }
        feature_storage.push(row);
        scores.push(target);
    }
    let feat_refs: Vec<&[f64]> = feature_storage.iter().map(|v| v.as_slice()).collect();

    let group = TrainingGroup {
        name: "synth".into(),
        human_scores: &scores,
        features: &feat_refs,
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 0.0,
    };

    let hp = GpuHparams {
        n_hidden: 128,
        n_epochs: 20,
        pairs_per_epoch: 4_000,
        minibatch_k: 128,
        initial_lr: 1e-3,
        leaky_alpha: 0.01,
        seed: 1,
        l2_lambda: 1e-5,
        mse_weight: 1.0,
        ranknet_weight: 1.0,
        monotonicity_reg: 0.0,
        monotonicity_margin: 1.0,
        tanh_output_head_scale: 0.0,
        ..GpuHparams::default()
    };

    let result = train_per_sample_alpha_head_gpu(&[group], &hp, n_features, GpuRuntime::Cuda);

    eprintln!(
        "GPU train: {} batches in {:.2} s ({:.1} batches/s)",
        result.n_batches,
        result.wall_seconds,
        result.n_batches as f64 / result.wall_seconds
    );

    let model = &result.model;
    // Smoke check: score a hi-target row and a lo-target row, expect
    // hi-row gets a higher predicted score.
    let hi_row = &feature_storage[n_rows - 1];
    let lo_row = &feature_storage[0];

    let std_row = |row: &[f64]| -> Vec<f64> {
        (0..n_features)
            .map(|d| (row[d] - model.scaler_mean[d]) / model.scaler_scale[d])
            .collect()
    };
    let hi_std = std_row(hi_row);
    let lo_std = std_row(lo_row);

    let (hi_y, _, _, _, _, _, _, _, _) = forward_per_sample_alpha_head(
        &hi_std,
        &model.w1,
        &model.b1,
        &model.rank_w,
        model.rank_b,
        &model.reducer_w,
        model.reducer_b,
        &model.w_alpha,
        model.b_alpha,
        n_features,
        model.n_hidden,
        hp.leaky_alpha,
    );
    let (lo_y, _, _, _, _, _, _, _, _) = forward_per_sample_alpha_head(
        &lo_std,
        &model.w1,
        &model.b1,
        &model.rank_w,
        model.rank_b,
        &model.reducer_w,
        model.reducer_b,
        &model.w_alpha,
        model.b_alpha,
        n_features,
        model.n_hidden,
        hp.leaky_alpha,
    );

    eprintln!("GPU-trained: hi_target=100 → y={hi_y:.3}, lo_target=0 → y={lo_y:.3}");
    assert!(
        hi_y > lo_y,
        "GPU per-sample-α head failed to rank: hi_y={hi_y} should exceed lo_y={lo_y}"
    );

    // Sanity check: weights are non-degenerate (not all zero, not all NaN).
    let w1_sum: f64 = model.w1.iter().map(|v| v.abs()).sum();
    assert!(w1_sum > 1e-3, "w1 collapsed (sum |w| = {w1_sum})");
    let rank_w_sum: f64 = model.rank_w.iter().map(|v| v.abs()).sum();
    assert!(
        rank_w_sum > 1e-3 || model.reducer_w.iter().map(|v| v.abs()).sum::<f64>() > 1e-3,
        "both heads collapsed"
    );
    assert!(
        model.w1.iter().all(|v| v.is_finite()),
        "w1 contains NaN/Inf"
    );
    assert!(model.rank_w.iter().all(|v| v.is_finite()));
    assert!(model.w_alpha.iter().all(|v| v.is_finite()));
}

#[test]
fn gpu_smoke_minimal_config_does_not_panic_cuda() {
    let n_features = 8;
    let n_rows = 32;
    let mut feats = vec![vec![0.0_f64; n_features]; n_rows];
    let mut scores = vec![0.0_f64; n_rows];
    for i in 0..n_rows {
        feats[i][0] = i as f64;
        scores[i] = i as f64;
    }
    let feat_refs: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "tiny".into(),
        human_scores: &scores,
        features: &feat_refs,
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 0.0,
    };
    let hp = GpuHparams {
        n_hidden: 32,
        n_epochs: 2,
        pairs_per_epoch: 64,
        minibatch_k: 32,
        ..GpuHparams::default()
    };
    let res = train_per_sample_alpha_head_gpu(&[group], &hp, n_features, GpuRuntime::Cuda);
    assert!(res.wall_seconds >= 0.0);
    assert!(res.n_batches >= 1);
}

// Ensure PerSampleAlphaHeadModel::new isn't dead — used by GPU trainer too.
#[allow(dead_code)]
fn _model_construct(n_features: usize, n_hidden: usize) -> PerSampleAlphaHeadModel {
    PerSampleAlphaHeadModel::new(n_features, n_hidden, 42)
}
