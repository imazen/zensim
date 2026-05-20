//! Phase 2 aux-loss smoke tests.
//!
//! Each test exercises one of the new aux kernels (anchor / cross-codec-eq /
//! σ-floor) via the CubeCL CPU backend so it runs on any developer
//! machine without a GPU. The aim is to catch shape mistakes (wrong
//! `batch_rows`, mis-paired A/B sides, etc.) and verify the
//! training loop does not panic / NaN-cascade when aux pools are
//! supplied alongside the main pair loss.

#![cfg(any(feature = "gpu-cpu", feature = "gpu-cuda"))]

use zensim_train_core::TrainingGroup;
use zensim_train_gpu::{
    GpuAnchorRows, GpuEquivPairs, GpuHparams, GpuRuntime,
    train_per_sample_alpha_head_gpu_with_aux,
};

#[cfg(all(feature = "gpu-cpu", not(feature = "gpu-cuda")))]
const RUNTIME: GpuRuntime = GpuRuntime::Cpu;
#[cfg(feature = "gpu-cuda")]
const RUNTIME: GpuRuntime = GpuRuntime::Cuda;

fn build_synth_dataset(
    n_features: usize,
    n_rows: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut prng_state: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut next_f64 = || {
        prng_state = prng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (prng_state as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    let mut feats: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    let mut scores: Vec<f64> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let target = (i as f64 / n_rows as f64) * 100.0;
        let mut row = vec![0.0_f64; n_features];
        row[0] = target * 0.5 + next_f64() * 1.0;
        for d in 1..n_features {
            row[d] = next_f64() * 0.5;
        }
        feats.push(row);
        scores.push(target);
    }
    (feats, scores)
}

#[test]
fn aux_anchor_only_does_not_crash() {
    let n_features = 16;
    let (feats, scores) = build_synth_dataset(n_features, 128, 1);
    let feat_refs: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".into(),
        human_scores: &scores,
        features: &feat_refs,
        train_weight: 1.0,
        validation_weight: 0.0,
    };

    // Build anchor pool: 16 rows pulled from the synth dataset.
    let anchor_feats: Vec<Vec<f64>> = (0..16).map(|i| feats[i * 8].clone()).collect();
    let anchor_refs: Vec<&[f64]> = anchor_feats.iter().map(|v| v.as_slice()).collect();
    let anchor_weights: Vec<f64> = vec![1.0; 16];
    let anchor_targets: Vec<f64> = vec![50.0; 16];

    let anchor = GpuAnchorRows {
        name: "test_anchor".into(),
        features: &anchor_refs,
        row_weights: &anchor_weights,
        target_scores: &anchor_targets,
    };

    let hp = GpuHparams {
        n_hidden: 32,
        n_epochs: 2,
        pairs_per_epoch: 256,
        minibatch_k: 64,
        anchor_loss_weight: 0.5,
        anchor_step_p: 1.0, // always fire
        minibatch_k_aux: 8,
        ..GpuHparams::default()
    };

    let res = train_per_sample_alpha_head_gpu_with_aux(
        &[group],
        &hp,
        n_features,
        RUNTIME,
        Some(&anchor),
        None,
    );
    assert!(res.n_batches >= 1);
    assert!(res.model.w1.iter().all(|v| v.is_finite()), "w1 NaN/Inf after anchor aux");
    assert!(res.model.rank_w.iter().all(|v| v.is_finite()));
    assert!(res.model.w_alpha.iter().all(|v| v.is_finite()));
}

#[test]
fn aux_eq_only_does_not_crash() {
    let n_features = 16;
    let (feats, scores) = build_synth_dataset(n_features, 128, 2);
    let feat_refs: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".into(),
        human_scores: &scores,
        features: &feat_refs,
        train_weight: 1.0,
        validation_weight: 0.0,
    };

    // Build equiv pool: 12 pairs (A from low-q half, B from high-q half).
    let mut a_feats: Vec<Vec<f64>> = Vec::new();
    let mut b_feats: Vec<Vec<f64>> = Vec::new();
    for i in 0..12 {
        a_feats.push(feats[i * 5].clone());
        b_feats.push(feats[i * 5 + 60].clone());
    }
    let a_refs: Vec<&[f64]> = a_feats.iter().map(|v| v.as_slice()).collect();
    let b_refs: Vec<&[f64]> = b_feats.iter().map(|v| v.as_slice()).collect();
    let weights: Vec<f64> = vec![1.0; 12];
    let butter_diff: Vec<f64> = (0..12).map(|i| if i % 2 == 0 { 0.3 } else { -0.3 }).collect();

    let equiv = GpuEquivPairs {
        name: "test_eq".into(),
        features_a: &a_refs,
        features_b: &b_refs,
        row_weights: &weights,
        butter_diff: &butter_diff,
    };

    let hp = GpuHparams {
        n_hidden: 32,
        n_epochs: 2,
        pairs_per_epoch: 256,
        minibatch_k: 64,
        cross_codec_eq_weight: 0.5,
        cross_codec_eq_step_p: 1.0,
        cross_codec_rank_preserve_weight: 0.3,
        minibatch_k_aux: 8,
        ..GpuHparams::default()
    };
    let res = train_per_sample_alpha_head_gpu_with_aux(
        &[group],
        &hp,
        n_features,
        RUNTIME,
        None,
        Some(&equiv),
    );
    assert!(res.n_batches >= 1);
    assert!(res.model.w1.iter().all(|v| v.is_finite()));
    assert!(res.model.rank_w.iter().all(|v| v.is_finite()));
    assert!(res.model.w_alpha.iter().all(|v| v.is_finite()));
}

#[test]
fn aux_sigma_floor_only_does_not_crash() {
    let n_features = 16;
    let (feats, scores) = build_synth_dataset(n_features, 128, 3);
    let feat_refs: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".into(),
        human_scores: &scores,
        features: &feat_refs,
        train_weight: 1.0,
        validation_weight: 0.0,
    };
    // σ-floor needs an equiv pool to draw probe rows from; reuse 32 pairs.
    let mut a_feats: Vec<Vec<f64>> = Vec::new();
    let mut b_feats: Vec<Vec<f64>> = Vec::new();
    for i in 0..32 {
        a_feats.push(feats[i * 3].clone());
        b_feats.push(feats[i * 3 + 1].clone());
    }
    let a_refs: Vec<&[f64]> = a_feats.iter().map(|v| v.as_slice()).collect();
    let b_refs: Vec<&[f64]> = b_feats.iter().map(|v| v.as_slice()).collect();
    let weights: Vec<f64> = vec![1.0; 32];
    let butter_diff: Vec<f64> = vec![]; // rank-preserve disabled

    let equiv = GpuEquivPairs {
        name: "test_eq".into(),
        features_a: &a_refs,
        features_b: &b_refs,
        row_weights: &weights,
        butter_diff: &butter_diff,
    };

    let hp = GpuHparams {
        n_hidden: 32,
        n_epochs: 2,
        pairs_per_epoch: 256,
        minibatch_k: 64,
        cross_codec_eq_weight: 0.0, // disabled — only σ-floor active
        cross_codec_eq_step_p: 0.0,
        dynamic_range_floor_weight: 1.0,
        dynamic_range_step_p: 1.0,
        dynamic_range_probe_n: 16,
        dynamic_range_sigma_threshold: 30.0, // high → likely violated → exercises grad path
        minibatch_k_aux: 8,
        ..GpuHparams::default()
    };
    // NOTE: σ-floor is gated on `cross_codec_eq_weight > 0` AND
    // `dynamic_range_floor_weight > 0` in dispatch — keep eq weight 0
    // here only to confirm that gate combination (it won't fire).
    // Re-run with eq active to actually fire σ-floor.
    let res = train_per_sample_alpha_head_gpu_with_aux(
        &[group],
        &hp,
        n_features,
        RUNTIME,
        None,
        Some(&equiv),
    );
    assert!(res.n_batches >= 1);
    assert!(res.model.w1.iter().all(|v| v.is_finite()));

    // Second run: eq + σ-floor BOTH active so σ-floor actually fires.
    let group2 = TrainingGroup {
        name: "synth".into(),
        human_scores: &scores,
        features: &feat_refs,
        train_weight: 1.0,
        validation_weight: 0.0,
    };
    let hp2 = GpuHparams {
        cross_codec_eq_weight: 0.1,
        cross_codec_eq_step_p: 0.5,
        ..hp
    };
    let res2 = train_per_sample_alpha_head_gpu_with_aux(
        &[group2],
        &hp2,
        n_features,
        RUNTIME,
        None,
        Some(&equiv),
    );
    assert!(res2.n_batches >= 1);
    assert!(res2.model.w1.iter().all(|v| v.is_finite()));
    assert!(res2.model.rank_w.iter().all(|v| v.is_finite()));
    assert!(res2.model.w_alpha.iter().all(|v| v.is_finite()));
}

#[test]
fn aux_all_three_combined_does_not_crash() {
    let n_features = 16;
    let (feats, scores) = build_synth_dataset(n_features, 128, 4);
    let feat_refs: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".into(),
        human_scores: &scores,
        features: &feat_refs,
        train_weight: 1.0,
        validation_weight: 0.0,
    };

    let anchor_feats: Vec<Vec<f64>> = (0..16).map(|i| feats[i * 8].clone()).collect();
    let anchor_refs: Vec<&[f64]> = anchor_feats.iter().map(|v| v.as_slice()).collect();
    let anchor_weights: Vec<f64> = vec![1.0; 16];
    let anchor_targets: Vec<f64> = vec![50.0; 16];
    let anchor = GpuAnchorRows {
        name: "anch".into(),
        features: &anchor_refs,
        row_weights: &anchor_weights,
        target_scores: &anchor_targets,
    };

    let mut a_feats: Vec<Vec<f64>> = Vec::new();
    let mut b_feats: Vec<Vec<f64>> = Vec::new();
    for i in 0..32 {
        a_feats.push(feats[i * 3].clone());
        b_feats.push(feats[i * 3 + 1].clone());
    }
    let a_refs: Vec<&[f64]> = a_feats.iter().map(|v| v.as_slice()).collect();
    let b_refs: Vec<&[f64]> = b_feats.iter().map(|v| v.as_slice()).collect();
    let weights: Vec<f64> = vec![1.0; 32];
    let butter_diff: Vec<f64> = (0..32)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    let equiv = GpuEquivPairs {
        name: "eq".into(),
        features_a: &a_refs,
        features_b: &b_refs,
        row_weights: &weights,
        butter_diff: &butter_diff,
    };

    let hp = GpuHparams {
        n_hidden: 32,
        n_epochs: 2,
        pairs_per_epoch: 256,
        minibatch_k: 64,
        anchor_loss_weight: 0.3,
        anchor_step_p: 0.5,
        cross_codec_eq_weight: 0.3,
        cross_codec_eq_step_p: 0.5,
        cross_codec_rank_preserve_weight: 0.1,
        dynamic_range_floor_weight: 0.5,
        dynamic_range_step_p: 0.5,
        dynamic_range_probe_n: 16,
        dynamic_range_sigma_threshold: 20.0,
        minibatch_k_aux: 8,
        ..GpuHparams::default()
    };

    let res = train_per_sample_alpha_head_gpu_with_aux(
        &[group],
        &hp,
        n_features,
        RUNTIME,
        Some(&anchor),
        Some(&equiv),
    );
    assert!(res.n_batches >= 1);
    assert!(res.model.w1.iter().all(|v| v.is_finite()));
    assert!(res.model.rank_w.iter().all(|v| v.is_finite()));
    assert!(res.model.w_alpha.iter().all(|v| v.is_finite()));
    let w1_sum: f64 = res.model.w1.iter().map(|v| v.abs()).sum();
    assert!(w1_sum > 1e-3, "w1 collapsed with all aux losses active");
}
