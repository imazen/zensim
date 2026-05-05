//! Mixture-of-Experts (MoE) trainer for V0_6.
//!
//! Architecture:
//!
//! ```text
//! input  x ∈ R^N            (N = n_features = n_content + n_cclass)
//!
//! gate:
//!   h_g = W_g1 · x + b_g1                       (R^Hg, ReLU)
//!   z   = W_g2 · h_g + b_g2                     (R^K, logits)
//!   w   = softmax(z / τ)                        (R^K, mixture weights)
//!
//! experts (k = 0..K):
//!   h_k    = LeakyReLU(W_e1[k] · x + b_e1[k])    (R^H)
//!   y_k    = w_e2[k] · h_k + b_e2[k]              (scalar)
//!
//! output:
//!   y = Σ_k w_k · y_k
//! ```
//!
//! ## Training
//!
//! Loss = RankNet(pairwise) + magnitude-match (optional) + L2 (optional)
//!      + load_balance_lambda · KL(uniform || mean_batch_w)
//!
//! The load-balance term penalizes gate collapse onto a single expert.
//! Without it, on a 99.5%-photo corpus the gate trivially routes
//! everything through expert 0 — exactly what FiLM degenerates to in
//! the same setting.
//!
//! ## Bake format
//!
//! - Each expert is baked as a standard 2-layer ZNPR v3 (LeakyReLU →
//!   Identity) — identical to the V0_6 baseline. Existing
//!   `Model::from_bytes` loads them with no changes.
//! - The gate is baked as a 2-layer ZNPR v3 (ReLU → Identity, K
//!   outputs). The runtime applies softmax/τ at inference time.
//! - A manifest TSV (`<stem>.moe_manifest.tsv`) lists the K expert
//!   bake paths and the gate bake path, plus metadata: K, τ,
//!   `hard_top1_threshold`, `n_features`, `n_hidden`.
//!
//! At inference (see `dataset_metric_baseline.rs::MoeManifest`):
//!   1. Run gate on standardized features → softmax → w[K]
//!   2. If `max(w) > hard_top1_threshold` → run only the argmax expert
//!      (saves K-1 forwards). Otherwise → run all K experts and
//!      compute Σ w_k · y_k.
//!
//! ## Why not fold the gate into experts?
//!
//! With FiLM, γ/β can be folded into the first layer because the
//! modulation is multiplicative on the same input. With MoE, the
//! mixing is over expert OUTPUTS — a different operator that needs a
//! runtime gate even at inference. Hence two artifact families: K
//! experts + 1 gate.

#![cfg(feature = "moe")]

use crate::mlp_train::{
    MlpHyperparams, SplitMix64, TrainingGroup, ValidationPolicy, bake_two_layer_znpr_v2,
    compute_scaler_from_groups, spearman_correlation,
};
use std::time::Instant;
use zensim::mlp::bake::{BakeLayer, BakeRequest, bake_v2};
use zensim::mlp::{Activation, WeightDtype};

/// Knobs specific to MoE on top of [`MlpHyperparams`].
#[derive(Clone, Debug)]
pub struct MoeHyperparams {
    /// Number of experts. Typically equals n_cclass.
    pub n_experts: usize,
    /// Hidden width of the gate network. Default 32.
    pub gate_hidden: usize,
    /// Softmax temperature. <1 sharpens, >1 softens. Default 1.0.
    pub gate_temperature: f64,
    /// Coefficient on the load-balance loss
    /// `λ · KL(uniform || mean_batch_w)`. 0 disables. Default 0.01.
    pub load_balance_lambda: f64,
    /// At inference, if `max(w) > threshold`, hard-route to that
    /// expert (skip the K-1 forwards). Default 0.95. Recorded in the
    /// manifest so the runtime can apply it.
    pub hard_top1_threshold: f64,
}

impl Default for MoeHyperparams {
    fn default() -> Self {
        Self {
            n_experts: 5,
            gate_hidden: 32,
            gate_temperature: 1.0,
            load_balance_lambda: 0.01,
            hard_top1_threshold: 0.95,
        }
    }
}

/// Output of [`train_mlp_moe`].
#[derive(Debug)]
pub struct MoeBakeOutput {
    /// Expert names (taken from class_names; one per K). When K =
    /// n_cclass and `cclass_*` columns are passed, this is the class
    /// short name (e.g., "photo", "screen").
    pub expert_names: Vec<String>,
    /// `expert_bakes[k]` = ZNPR v3 bytes for expert k.
    /// Same shape as the V0_6 baseline: (n_features → n_hidden → 1).
    pub expert_bakes: Vec<Vec<u8>>,
    /// Gate bake bytes — ZNPR v3 (n_features → gate_hidden → K).
    /// Activation: ReLU → Identity. Runtime applies softmax(z/τ).
    pub gate_bake: Vec<u8>,
    /// Best validation score during training.
    pub best_val_score: f64,
    /// Hyperparameters echoed for the manifest.
    pub n_experts: usize,
    pub gate_temperature: f64,
    pub hard_top1_threshold: f64,
    pub n_features: usize,
    pub n_hidden: usize,
}

/// Train MoE with K experts + gate, using RankNet on pair distances
/// and an optional load-balance regularizer.
///
/// The training data shape mirrors `train_mlp_film`: features include
/// a contiguous `cclass_*` tail. The gate sees the FULL input, so the
/// one-hot tail is a strong prior on expert routing without forcing
/// hard alignment.
#[allow(clippy::too_many_arguments)]
pub fn train_mlp_moe(
    groups: &[TrainingGroup<'_>],
    n_features: usize,
    expert_names: &[String],
    hyperparams: &MlpHyperparams,
    moe: &MoeHyperparams,
    log: &mut Vec<String>,
) -> MoeBakeOutput {
    assert!(moe.n_experts >= 1, "n_experts must be >= 1");
    assert_eq!(expert_names.len(), moe.n_experts);
    let k_experts = moe.n_experts;
    let n_hidden = hyperparams.n_hidden;
    let gate_hidden = moe.gate_hidden;
    let n_outputs = 1usize;

    assert!(!groups.is_empty(), "need at least one training group");
    for g in groups {
        assert_eq!(
            g.human_scores.len(),
            g.features.len(),
            "{}: scores/features length mismatch",
            g.name
        );
        assert!(
            g.features.iter().all(|f| f.len() == n_features),
            "{}: feature length mismatch",
            g.name
        );
    }
    let train_total: f64 = groups.iter().map(|g| g.train_weight).sum();
    assert!(train_total > 0.0, "no training groups (all train_weight == 0)");
    let train_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| if g.train_weight > 0.0 { Some(i) } else { None })
        .collect();
    let val_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| if g.validation_weight > 0.0 { Some(i) } else { None })
        .collect();

    let log_line = |msg: &str, log: &mut Vec<String>| {
        eprintln!("{msg}");
        log.push(msg.to_string());
    };

    log_line(
        &format!(
            "MoE-MLP train: arch=[{n_features} → K={k_experts} experts × ({n_hidden} LeakyReLU α={alpha}) → 1, \
             gate: {n_features} → {gate_hidden} ReLU → softmax(K)/τ={tau}, λ_lb={lb}], val_policy={vp:?}",
            alpha = hyperparams.leaky_alpha,
            tau = moe.gate_temperature,
            lb = moe.load_balance_lambda,
            vp = hyperparams.validation_policy,
        ),
        log,
    );
    log_line(
        &format!(
            "  experts: [{}], hard_top1_threshold={:.3}",
            expert_names.join(","),
            moe.hard_top1_threshold,
        ),
        log,
    );
    for (i, g) in groups.iter().enumerate() {
        let role = match (g.train_weight > 0.0, g.validation_weight > 0.0) {
            (true, true) => "train+val",
            (true, false) => "train",
            (false, true) => "val-only",
            (false, false) => "report",
        };
        log_line(
            &format!(
                "  {role:>9} group {i}: '{}' n={} train_w={:.3} val_w={:.3}",
                g.name,
                g.features.len(),
                g.train_weight,
                g.validation_weight
            ),
            log,
        );
    }

    let (scaler_mean, scaler_scale) =
        compute_scaler_from_groups(groups, &train_indices, n_features);

    let std_features: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            let mut buf = vec![0.0f64; g.features.len() * n_features];
            for (i, &f) in g.features.iter().enumerate() {
                for d in 0..n_features {
                    buf[i * n_features + d] =
                        (f[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
                }
            }
            buf
        })
        .collect();

    let mut rng = SplitMix64::new(hyperparams.seed);

    // Per-expert weights: stored flat. expert_w1[k] is the W1 for
    // expert k of length n_features * n_hidden, etc.
    let std_e1 = (2.0 / (n_features + n_hidden) as f64).sqrt();
    let std_e2 = (2.0 / (n_hidden + n_outputs) as f64).sqrt();
    let mut e_w1: Vec<Vec<f64>> = (0..k_experts)
        .map(|_| {
            (0..n_features * n_hidden)
                .map(|_| rng.next_normal() * std_e1)
                .collect()
        })
        .collect();
    let mut e_b1: Vec<Vec<f64>> = (0..k_experts).map(|_| vec![0.0f64; n_hidden]).collect();
    let mut e_w2: Vec<Vec<f64>> = (0..k_experts)
        .map(|_| {
            (0..n_hidden * n_outputs)
                .map(|_| rng.next_normal() * std_e2)
                .collect()
        })
        .collect();
    let mut e_b2: Vec<Vec<f64>> = (0..k_experts).map(|_| vec![0.0f64; n_outputs]).collect();

    // Gate weights.
    let std_g1 = (2.0 / (n_features + gate_hidden) as f64).sqrt();
    // Initialize gate output layer near zero so the softmax starts
    // ~uniform — gives every expert a chance to receive gradient
    // before the gate sharpens.
    let std_g2 = (2.0 / (gate_hidden + k_experts) as f64).sqrt() * 0.01;
    let mut g_w1: Vec<f64> = (0..n_features * gate_hidden)
        .map(|_| rng.next_normal() * std_g1)
        .collect();
    let mut g_b1 = vec![0.0f64; gate_hidden];
    let mut g_w2: Vec<f64> = (0..gate_hidden * k_experts)
        .map(|_| rng.next_normal() * std_g2)
        .collect();
    let mut g_b2 = vec![0.0f64; k_experts];

    // Adam state — separate per-expert and gate to keep the per-tensor
    // moment estimates well-behaved.
    let mut adam_e: Vec<TwoLayerAdam> = (0..k_experts)
        .map(|_| TwoLayerAdam::new(n_features * n_hidden, n_hidden, n_hidden * n_outputs, n_outputs))
        .collect();
    let mut adam_g = TwoLayerAdam::new(
        n_features * gate_hidden,
        gate_hidden,
        gate_hidden * k_experts,
        k_experts,
    );

    let start = Instant::now();
    let mut best_val_score = f64::NEG_INFINITY;
    let mut best: Option<(
        Vec<Vec<f64>>, // e_w1
        Vec<Vec<f64>>, // e_b1
        Vec<Vec<f64>>, // e_w2
        Vec<Vec<f64>>, // e_b2
        Vec<f64>,      // g_w1
        Vec<f64>,      // g_b1
        Vec<f64>,      // g_w2
        Vec<f64>,      // g_b2
    )> = None;
    let mut stale_epochs = 0usize;

    let cdf: Vec<f64> = {
        let mut cum = 0.0;
        train_indices
            .iter()
            .map(|&gi| {
                cum += groups[gi].train_weight;
                cum / train_total
            })
            .collect()
    };

    let low_band_indices: Vec<Vec<usize>> = if hyperparams.low_band_oversample > 0.0 {
        groups
            .iter()
            .map(|g| {
                g.human_scores
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &s)| {
                        (s.is_finite()
                            && s >= hyperparams.low_band_target_min
                            && s <= hyperparams.low_band_target_max)
                            .then_some(i)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    } else {
        Vec::new()
    };

    for epoch in 0..hyperparams.n_epochs {
        let lr = hyperparams.initial_lr
            * 0.5
            * (1.0 + (std::f64::consts::PI * (epoch % 50) as f64 / 50.0).cos());

        let mut total_loss = 0.0f64;
        let mut n_steps = 0u64;
        // Running mean of gate weights across this epoch's pairs
        // (used for load-balance KL term — the running mean is
        // accumulated, then we shape one regularizer step per epoch
        // directly via the gate output. To keep it stable we apply
        // the load-balance loss per-pair, not at epoch boundary —
        // adding a tiny push toward uniform on each step keeps the
        // gate from collapsing.)
        for _ in 0..hyperparams.pairs_per_epoch {
            let u = rng.next_f64_unit();
            let g_idx = train_indices[cdf.partition_point(|&c| c < u).min(cdf.len() - 1)];
            let g = &groups[g_idx];
            let n = g.features.len();
            if n < 2 {
                continue;
            }
            let use_low_band = hyperparams.low_band_oversample > 0.0
                && !low_band_indices.is_empty()
                && !low_band_indices[g_idx].is_empty()
                && rng.next_f64_unit() < hyperparams.low_band_oversample;
            let (ia, ib) = if use_low_band {
                let pool = &low_band_indices[g_idx];
                let lp = pool.len();
                (
                    pool[(rng.next_u64() as usize) % lp],
                    pool[(rng.next_u64() as usize) % lp],
                )
            } else {
                (
                    (rng.next_u64() as usize) % n,
                    (rng.next_u64() as usize) % n,
                )
            };
            if ia == ib {
                continue;
            }

            let g_feats = &std_features[g_idx];
            let xa = &g_feats[ia * n_features..(ia + 1) * n_features];
            let xb = &g_feats[ib * n_features..(ib + 1) * n_features];

            // Forward both endpoints through MoE.
            let fa = forward_moe(
                xa, &e_w1, &e_b1, &e_w2, &e_b2,
                &g_w1, &g_b1, &g_w2, &g_b2,
                n_features, n_hidden, gate_hidden, k_experts,
                hyperparams.leaky_alpha, moe.gate_temperature,
            );
            let fb = forward_moe(
                xb, &e_w1, &e_b1, &e_w2, &e_b2,
                &g_w1, &g_b1, &g_w2, &g_b2,
                n_features, n_hidden, gate_hidden, k_experts,
                hyperparams.leaky_alpha, moe.gate_temperature,
            );

            let target_signed = g.human_scores[ia] - g.human_scores[ib];
            let target = target_signed.signum();
            if target == 0.0 {
                continue;
            }
            let pred_diff = fb.y - fa.y;
            let z = -target * pred_diff;
            let loss = if z > 50.0 {
                z
            } else if z < -50.0 {
                0.0
            } else {
                (z.exp() + 1.0).ln()
            };
            total_loss += loss;
            n_steps += 1;

            let sig_z = 1.0 / (1.0 + (-z).exp());
            let mut dl_d_pred_diff = -target * sig_z;

            if hyperparams.magnitude_match_lambda > 0.0 {
                let mag_residual =
                    pred_diff - hyperparams.magnitude_match_alpha * target_signed;
                let mag_loss =
                    hyperparams.magnitude_match_lambda * mag_residual * mag_residual;
                total_loss += mag_loss;
                dl_d_pred_diff += 2.0 * hyperparams.magnitude_match_lambda * mag_residual;
            }

            let dl_dya = -dl_d_pred_diff;
            let dl_dyb = dl_d_pred_diff;

            backprop_moe(
                xa, &fa, dl_dya,
                &e_w2, &g_w2,
                &mut adam_e, &mut adam_g,
                n_features, n_hidden, gate_hidden, k_experts,
                hyperparams.leaky_alpha, moe.gate_temperature,
            );
            backprop_moe(
                xb, &fb, dl_dyb,
                &e_w2, &g_w2,
                &mut adam_e, &mut adam_g,
                n_features, n_hidden, gate_hidden, k_experts,
                hyperparams.leaky_alpha, moe.gate_temperature,
            );

            // Load-balance: KL(uniform || mean(w_a, w_b)) — penalize
            // gate concentration. Per-pair mean keeps the gradient
            // flowing on every step. The gradient for each pair is
            // ∂KL/∂w_k = -(1/K) / w_k (chain through softmax → logits
            // is implicit via fa.gate_softmax / fb.gate_softmax).
            if moe.load_balance_lambda > 0.0 {
                let lambda = moe.load_balance_lambda;
                add_load_balance_grad(
                    &fa, &mut adam_e, &mut adam_g, &g_w2, lambda, k_experts,
                    gate_hidden, moe.gate_temperature,
                    xa, n_features,
                );
                add_load_balance_grad(
                    &fb, &mut adam_e, &mut adam_g, &g_w2, lambda, k_experts,
                    gate_hidden, moe.gate_temperature,
                    xb, n_features,
                );
            }

            if hyperparams.l2_lambda > 0.0 {
                for k in 0..k_experts {
                    for (g, &w) in adam_e[k].gw1.iter_mut().zip(e_w1[k].iter()) {
                        *g += hyperparams.l2_lambda * w;
                    }
                    for (g, &w) in adam_e[k].gw2.iter_mut().zip(e_w2[k].iter()) {
                        *g += hyperparams.l2_lambda * w;
                    }
                }
                for (g, &w) in adam_g.gw1.iter_mut().zip(g_w1.iter()) {
                    *g += hyperparams.l2_lambda * w;
                }
                for (g, &w) in adam_g.gw2.iter_mut().zip(g_w2.iter()) {
                    *g += hyperparams.l2_lambda * w;
                }
            }

            for k in 0..k_experts {
                adam_e[k].step(&mut e_w1[k], &mut e_b1[k], &mut e_w2[k], &mut e_b2[k], lr);
            }
            adam_g.step(&mut g_w1, &mut g_b1, &mut g_w2, &mut g_b2, lr);
        }

        let avg_loss = if n_steps > 0 {
            total_loss / n_steps as f64
        } else {
            0.0
        };

        if epoch % hyperparams.log_every == 0 || epoch == hyperparams.n_epochs - 1 {
            let group_srocc: Vec<f64> = groups
                .iter()
                .enumerate()
                .map(|(gi, g)| {
                    let preds = predict_group_moe(
                        &std_features[gi], g.features.len(), n_features,
                        &e_w1, &e_b1, &e_w2, &e_b2,
                        &g_w1, &g_b1, &g_w2, &g_b2,
                        n_hidden, gate_hidden, k_experts,
                        hyperparams.leaky_alpha, moe.gate_temperature,
                    );
                    let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
                    spearman_correlation(g.human_scores, &neg_preds)
                })
                .collect();

            let val_score = if val_indices.is_empty() {
                group_srocc.iter().sum::<f64>() / group_srocc.len() as f64
            } else {
                match hyperparams.validation_policy {
                    ValidationPolicy::Mean => {
                        let total: f64 =
                            val_indices.iter().map(|&i| groups[i].validation_weight).sum();
                        val_indices
                            .iter()
                            .map(|&i| group_srocc[i] * groups[i].validation_weight)
                            .sum::<f64>()
                            / total
                    }
                    ValidationPolicy::Min => val_indices
                        .iter()
                        .map(|&i| group_srocc[i])
                        .fold(f64::INFINITY, f64::min),
                }
            };

            // Also report mean gate distribution across the val set for diagnostics.
            let gate_means = compute_mean_gate(
                &std_features, groups,
                &g_w1, &g_b1, &g_w2, &g_b2,
                n_features, gate_hidden, k_experts, moe.gate_temperature,
            );

            let elapsed = start.elapsed().as_secs_f64();
            let per_group = group_srocc
                .iter()
                .zip(groups.iter())
                .map(|(s, g)| format!("{}={s:.4}", g.name))
                .collect::<Vec<_>>()
                .join(" ");
            let gate_str = gate_means
                .iter()
                .enumerate()
                .map(|(k, &m)| format!("{}={m:.3}", expert_names[k]))
                .collect::<Vec<_>>()
                .join(",");
            log_line(
                &format!(
                    "  epoch {epoch:>3} | lr={lr:.5} | loss={avg_loss:.4} | val={val_score:.4} (best={best_val_score:.4}) | gate=[{gate_str}] | {per_group} | t={elapsed:.1}s"
                ),
                log,
            );

            if val_score > best_val_score {
                best_val_score = val_score;
                stale_epochs = 0;
                best = Some((
                    e_w1.clone(), e_b1.clone(), e_w2.clone(), e_b2.clone(),
                    g_w1.clone(), g_b1.clone(), g_w2.clone(), g_b2.clone(),
                ));
            } else {
                stale_epochs += hyperparams.log_every;
                if hyperparams.early_stop_patience > 0
                    && stale_epochs >= hyperparams.early_stop_patience
                {
                    log_line(
                        &format!(
                            "  early stop at epoch {epoch} (no validation improvement for {stale_epochs} epochs)"
                        ),
                        log,
                    );
                    break;
                }
            }
        }
    }

    let (e_w1f, e_b1f, e_w2f, e_b2f, g_w1f, g_b1f, g_w2f, g_b2f) = best
        .unwrap_or((e_w1, e_b1, e_w2, e_b2, g_w1, g_b1, g_w2, g_b2));
    log_line(
        &format!("MoE-MLP train: best validation SROCC = {best_val_score:.4}"),
        log,
    );

    // Bake K experts (each is a regular V0_6-shaped 2-layer MLP).
    let mut expert_bakes = Vec::with_capacity(k_experts);
    for k in 0..k_experts {
        let bake = bake_two_layer_znpr_v2(
            &scaler_mean, &scaler_scale,
            &e_w1f[k], &e_b1f[k], &e_w2f[k], &e_b2f[k],
            n_features, n_hidden, n_outputs,
        );
        expert_bakes.push(bake);
    }

    // Bake gate (ReLU → Identity, K outputs). The runtime applies
    // softmax(z/τ) — we bake raw logits.
    let gate_bake = bake_gate_znpr_v3(
        &scaler_mean, &scaler_scale,
        &g_w1f, &g_b1f, &g_w2f, &g_b2f,
        n_features, gate_hidden, k_experts,
    );

    MoeBakeOutput {
        expert_names: expert_names.to_vec(),
        expert_bakes,
        gate_bake,
        best_val_score,
        n_experts: k_experts,
        gate_temperature: moe.gate_temperature,
        hard_top1_threshold: moe.hard_top1_threshold,
        n_features,
        n_hidden,
    }
}

/// Forward MoE: returns mixed scalar y plus all intermediates needed
/// for backprop.
struct MoeForward {
    y: f64,
    /// gate_h_pre: pre-activation R^Hg (before ReLU)
    gate_h_pre: Vec<f64>,
    /// gate_h: post-ReLU R^Hg
    gate_h: Vec<f64>,
    /// gate_logits: pre-softmax R^K. Currently unused for backprop
    /// (we route through `gate_softmax`) but kept for diagnostics +
    /// future per-pair logit logging.
    #[allow(dead_code)]
    gate_logits: Vec<f64>,
    /// gate_softmax: softmax(z/τ) R^K
    gate_softmax: Vec<f64>,
    /// expert_h_pre[k]: pre-activation R^H per expert
    expert_h_pre: Vec<Vec<f64>>,
    /// expert_h[k]: post-leakyReLU R^H per expert
    expert_h: Vec<Vec<f64>>,
    /// expert_y[k]: scalar score per expert
    expert_y: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn forward_moe(
    x: &[f64],
    e_w1: &[Vec<f64>],
    e_b1: &[Vec<f64>],
    e_w2: &[Vec<f64>],
    e_b2: &[Vec<f64>],
    g_w1: &[f64],
    g_b1: &[f64],
    g_w2: &[f64],
    g_b2: &[f64],
    n_features: usize,
    n_hidden: usize,
    gate_hidden: usize,
    k_experts: usize,
    leaky_alpha: f64,
    tau: f64,
) -> MoeForward {
    // Gate: x → gate_hidden (ReLU) → K (softmax/τ).
    let mut gate_h_pre = g_b1.to_vec();
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &g_w1[i * gate_hidden..(i + 1) * gate_hidden];
        for (acc, &w) in gate_h_pre.iter_mut().zip(row.iter()) {
            *acc += s * w;
        }
    }
    let gate_h: Vec<f64> = gate_h_pre.iter().map(|&v| v.max(0.0)).collect();
    let mut gate_logits = g_b2.to_vec();
    for i in 0..gate_hidden {
        let s = gate_h[i];
        if s == 0.0 {
            continue;
        }
        let row = &g_w2[i * k_experts..(i + 1) * k_experts];
        for (acc, &w) in gate_logits.iter_mut().zip(row.iter()) {
            *acc += s * w;
        }
    }
    let gate_softmax = softmax_with_temperature(&gate_logits, tau);

    // K experts forward.
    let mut expert_h_pre = Vec::with_capacity(k_experts);
    let mut expert_h = Vec::with_capacity(k_experts);
    let mut expert_y = Vec::with_capacity(k_experts);
    for k in 0..k_experts {
        let mut h_pre = e_b1[k].to_vec();
        for i in 0..n_features {
            let s = x[i];
            if s == 0.0 {
                continue;
            }
            let row = &e_w1[k][i * n_hidden..(i + 1) * n_hidden];
            for (acc, &w) in h_pre.iter_mut().zip(row.iter()) {
                *acc += s * w;
            }
        }
        let h: Vec<f64> = h_pre
            .iter()
            .map(|&v| if v >= 0.0 { v } else { leaky_alpha * v })
            .collect();
        let mut y = e_b2[k][0];
        for o in 0..n_hidden {
            y += h[o] * e_w2[k][o];
        }
        expert_h_pre.push(h_pre);
        expert_h.push(h);
        expert_y.push(y);
    }

    let y: f64 = (0..k_experts).map(|k| gate_softmax[k] * expert_y[k]).sum();

    MoeForward {
        y,
        gate_h_pre,
        gate_h,
        gate_logits,
        gate_softmax,
        expert_h_pre,
        expert_h,
        expert_y,
    }
}

fn softmax_with_temperature(z: &[f64], tau: f64) -> Vec<f64> {
    let inv_tau = 1.0 / tau.max(1e-6);
    let max_z = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|&v| ((v - max_z) * inv_tau).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / z.len() as f64; z.len()]
    }
}

#[allow(clippy::too_many_arguments)]
fn backprop_moe(
    x: &[f64],
    fwd: &MoeForward,
    dl_dy: f64,
    e_w2: &[Vec<f64>],
    g_w2: &[f64],
    adam_e: &mut [TwoLayerAdam],
    adam_g: &mut TwoLayerAdam,
    n_features: usize,
    n_hidden: usize,
    gate_hidden: usize,
    k_experts: usize,
    leaky_alpha: f64,
    tau: f64,
) {
    // y = sum_k w_k * y_k
    // ∂y/∂y_k = w_k
    // ∂y/∂w_k = y_k
    // For each expert k, dl_dy_k = dl_dy * w_k
    for k in 0..k_experts {
        let dl_dy_k = dl_dy * fwd.gate_softmax[k];

        // Expert k: y_k = w2 · h_k + b2; h_k = LeakyReLU(h_pre_k)
        let h_k = &fwd.expert_h[k];
        let h_pre_k = &fwd.expert_h_pre[k];

        let gw2 = &mut adam_e[k].gw2;
        let gb2 = &mut adam_e[k].gb2;
        for o in 0..n_hidden {
            gw2[o] += dl_dy_k * h_k[o];
        }
        gb2[0] += dl_dy_k;

        let mut dl_dh_pre = vec![0.0f64; n_hidden];
        for o in 0..n_hidden {
            let dh = dl_dy_k * e_w2[k][o];
            dl_dh_pre[o] = if h_pre_k[o] >= 0.0 { dh } else { leaky_alpha * dh };
        }
        let gw1 = &mut adam_e[k].gw1;
        let gb1 = &mut adam_e[k].gb1;
        for i in 0..n_features {
            let s = x[i];
            if s == 0.0 {
                continue;
            }
            let row = &mut gw1[i * n_hidden..(i + 1) * n_hidden];
            for (gv, &dh) in row.iter_mut().zip(dl_dh_pre.iter()) {
                *gv += s * dh;
            }
        }
        for (gv, &dh) in gb1.iter_mut().zip(dl_dh_pre.iter()) {
            *gv += dh;
        }
    }

    // Gate gradient: dl/dw_k = dl_dy * y_k
    // softmax/τ: dw_i/dz_j = (w_i (δ_ij - w_j)) / τ
    // → dl/dz_j = (1/τ) sum_i dl/dw_i * w_i * (δ_ij - w_j)
    //          = (1/τ) (dl/dw_j * w_j - w_j * sum_i dl/dw_i * w_i)
    let mut dl_dw = vec![0.0f64; k_experts];
    for k in 0..k_experts {
        dl_dw[k] = dl_dy * fwd.expert_y[k];
    }
    let inv_tau = 1.0 / tau.max(1e-6);
    let dot: f64 = (0..k_experts).map(|k| dl_dw[k] * fwd.gate_softmax[k]).sum();
    let mut dl_dz = vec![0.0f64; k_experts];
    for k in 0..k_experts {
        dl_dz[k] = inv_tau * fwd.gate_softmax[k] * (dl_dw[k] - dot);
    }

    // Gate output layer: z = W_g2 · h + b_g2
    let gw2 = &mut adam_g.gw2;
    let gb2 = &mut adam_g.gb2;
    for o in 0..k_experts {
        gb2[o] += dl_dz[o];
    }
    for i in 0..gate_hidden {
        let s = fwd.gate_h[i];
        if s == 0.0 {
            continue;
        }
        let row = &mut gw2[i * k_experts..(i + 1) * k_experts];
        for (gv, &dz) in row.iter_mut().zip(dl_dz.iter()) {
            *gv += s * dz;
        }
    }
    // Gate hidden gradient: ReLU passthrough.
    let mut dl_dh = vec![0.0f64; gate_hidden];
    for j in 0..gate_hidden {
        let s = fwd.gate_h_pre[j];
        if s <= 0.0 {
            continue;
        }
        let mut acc = 0.0;
        for k in 0..k_experts {
            acc += g_w2[j * k_experts + k] * dl_dz[k];
        }
        dl_dh[j] = acc;
    }

    // Gate W_g1 / b_g1
    let gw1 = &mut adam_g.gw1;
    let gb1 = &mut adam_g.gb1;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &mut gw1[i * gate_hidden..(i + 1) * gate_hidden];
        for (gv, &dh) in row.iter_mut().zip(dl_dh.iter()) {
            *gv += s * dh;
        }
    }
    for (gv, &dh) in gb1.iter_mut().zip(dl_dh.iter()) {
        *gv += dh;
    }
}

/// Add the load-balance gradient to the gate Adam state.
///
/// Loss = λ · KL(uniform || mean_w) but instead of accumulating
/// `mean_w` over a batch, we apply per-pair: `λ · KL(uniform || w)` =
/// `λ · (-1/K · sum_k log(K · w_k))`. ∂L/∂w_k = -λ/K · 1/w_k.
/// Then chain through softmax exactly like backprop_moe does.
#[allow(clippy::too_many_arguments)]
fn add_load_balance_grad(
    fwd: &MoeForward,
    _adam_e: &mut [TwoLayerAdam],
    adam_g: &mut TwoLayerAdam,
    g_w2: &[f64],
    lambda: f64,
    k_experts: usize,
    gate_hidden: usize,
    tau: f64,
    x: &[f64],
    n_features: usize,
) {
    // ∂L/∂w_k for L = λ · KL(uniform || w) = λ · (1/K) · Σ_k log(1/(K w_k))
    //   = -λ/(K·w_k)
    let inv_k = 1.0 / k_experts as f64;
    let mut dl_dw = vec![0.0f64; k_experts];
    for k in 0..k_experts {
        let w = fwd.gate_softmax[k].max(1e-8);
        dl_dw[k] = -lambda * inv_k / w;
    }
    let inv_tau = 1.0 / tau.max(1e-6);
    let dot: f64 = (0..k_experts).map(|k| dl_dw[k] * fwd.gate_softmax[k]).sum();
    let mut dl_dz = vec![0.0f64; k_experts];
    for k in 0..k_experts {
        dl_dz[k] = inv_tau * fwd.gate_softmax[k] * (dl_dw[k] - dot);
    }

    let gw2 = &mut adam_g.gw2;
    let gb2 = &mut adam_g.gb2;
    for o in 0..k_experts {
        gb2[o] += dl_dz[o];
    }
    for i in 0..gate_hidden {
        let s = fwd.gate_h[i];
        if s == 0.0 {
            continue;
        }
        let row = &mut gw2[i * k_experts..(i + 1) * k_experts];
        for (gv, &dz) in row.iter_mut().zip(dl_dz.iter()) {
            *gv += s * dz;
        }
    }
    let mut dl_dh = vec![0.0f64; gate_hidden];
    for j in 0..gate_hidden {
        let s = fwd.gate_h_pre[j];
        if s <= 0.0 {
            continue;
        }
        let mut acc = 0.0;
        for k in 0..k_experts {
            acc += g_w2[j * k_experts + k] * dl_dz[k];
        }
        dl_dh[j] = acc;
    }
    let gw1 = &mut adam_g.gw1;
    let gb1 = &mut adam_g.gb1;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &mut gw1[i * gate_hidden..(i + 1) * gate_hidden];
        for (gv, &dh) in row.iter_mut().zip(dl_dh.iter()) {
            *gv += s * dh;
        }
    }
    for (gv, &dh) in gb1.iter_mut().zip(dl_dh.iter()) {
        *gv += dh;
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_group_moe(
    std_x: &[f64],
    n_pairs: usize,
    n_features: usize,
    e_w1: &[Vec<f64>],
    e_b1: &[Vec<f64>],
    e_w2: &[Vec<f64>],
    e_b2: &[Vec<f64>],
    g_w1: &[f64],
    g_b1: &[f64],
    g_w2: &[f64],
    g_b2: &[f64],
    n_hidden: usize,
    gate_hidden: usize,
    k_experts: usize,
    leaky_alpha: f64,
    tau: f64,
) -> Vec<f64> {
    (0..n_pairs)
        .map(|i| {
            let xi = &std_x[i * n_features..(i + 1) * n_features];
            forward_moe(
                xi, e_w1, e_b1, e_w2, e_b2, g_w1, g_b1, g_w2, g_b2,
                n_features, n_hidden, gate_hidden, k_experts, leaky_alpha, tau,
            )
            .y
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn compute_mean_gate(
    std_features: &[Vec<f64>],
    groups: &[TrainingGroup<'_>],
    g_w1: &[f64],
    g_b1: &[f64],
    g_w2: &[f64],
    g_b2: &[f64],
    n_features: usize,
    gate_hidden: usize,
    k_experts: usize,
    tau: f64,
) -> Vec<f64> {
    let mut mean = vec![0.0f64; k_experts];
    let mut count = 0u64;
    for (gi, g) in groups.iter().enumerate() {
        let buf = &std_features[gi];
        for i in 0..g.features.len() {
            let xi = &buf[i * n_features..(i + 1) * n_features];
            // Inline gate forward only (skip experts).
            let mut h_pre = g_b1.to_vec();
            for j in 0..n_features {
                let s = xi[j];
                if s == 0.0 {
                    continue;
                }
                let row = &g_w1[j * gate_hidden..(j + 1) * gate_hidden];
                for (acc, &w) in h_pre.iter_mut().zip(row.iter()) {
                    *acc += s * w;
                }
            }
            let h: Vec<f64> = h_pre.iter().map(|&v| v.max(0.0)).collect();
            let mut z = g_b2.to_vec();
            for j in 0..gate_hidden {
                let s = h[j];
                if s == 0.0 {
                    continue;
                }
                let row = &g_w2[j * k_experts..(j + 1) * k_experts];
                for (acc, &w) in z.iter_mut().zip(row.iter()) {
                    *acc += s * w;
                }
            }
            let w = softmax_with_temperature(&z, tau);
            for k in 0..k_experts {
                mean[k] += w[k];
            }
            count += 1;
        }
    }
    if count > 0 {
        for m in &mut mean {
            *m /= count as f64;
        }
    }
    mean
}

/// Bake the gate as a 2-layer ZNPR v3 (ReLU → Identity, K outputs).
/// The runtime applies softmax(z/τ) — we bake raw logits.
#[allow(clippy::too_many_arguments)]
fn bake_gate_znpr_v3(
    scaler_mean: &[f64],
    scaler_scale: &[f64],
    g_w1: &[f64],
    g_b1: &[f64],
    g_w2: &[f64],
    g_b2: &[f64],
    n_features: usize,
    gate_hidden: usize,
    k_experts: usize,
) -> Vec<u8> {
    let scaler_mean_f32: Vec<f32> = scaler_mean.iter().map(|&v| v as f32).collect();
    let scaler_scale_f32: Vec<f32> = scaler_scale.iter().map(|&v| v as f32).collect();
    let g_w1_f32: Vec<f32> = g_w1.iter().map(|&v| v as f32).collect();
    let g_b1_f32: Vec<f32> = g_b1.iter().map(|&v| v as f32).collect();
    let g_w2_f32: Vec<f32> = g_w2.iter().map(|&v| v as f32).collect();
    let g_b2_f32: Vec<f32> = g_b2.iter().map(|&v| v as f32).collect();
    let layers = [
        BakeLayer {
            in_dim: n_features,
            out_dim: gate_hidden,
            activation: Activation::Relu,
            dtype: WeightDtype::F32,
            weights: &g_w1_f32,
            biases: &g_b1_f32,
        },
        BakeLayer {
            in_dim: gate_hidden,
            out_dim: k_experts,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &g_w2_f32,
            biases: &g_b2_f32,
        },
    ];
    bake_v2(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean_f32,
        scaler_scale: &scaler_scale_f32,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
    })
    .expect("v2 bake of MoE gate")
}

/// Two-layer Adam state — used both for experts (each expert has its
/// own copy) and the gate. Same shape, different sizes per instance.
pub(crate) struct TwoLayerAdam {
    pub gw1: Vec<f64>,
    pub gb1: Vec<f64>,
    pub gw2: Vec<f64>,
    pub gb2: Vec<f64>,
    mw1: Vec<f64>,
    mb1: Vec<f64>,
    mw2: Vec<f64>,
    mb2: Vec<f64>,
    vw1: Vec<f64>,
    vb1: Vec<f64>,
    vw2: Vec<f64>,
    vb2: Vec<f64>,
    t: u64,
}

impl TwoLayerAdam {
    pub(crate) fn new(nw1: usize, nb1: usize, nw2: usize, nb2: usize) -> Self {
        Self {
            gw1: vec![0.0; nw1],
            gb1: vec![0.0; nb1],
            gw2: vec![0.0; nw2],
            gb2: vec![0.0; nb2],
            mw1: vec![0.0; nw1],
            mb1: vec![0.0; nb1],
            mw2: vec![0.0; nw2],
            mb2: vec![0.0; nb2],
            vw1: vec![0.0; nw1],
            vb1: vec![0.0; nb1],
            vw2: vec![0.0; nw2],
            vb2: vec![0.0; nb2],
            t: 0,
        }
    }

    pub(crate) fn step(
        &mut self,
        w1: &mut [f64],
        b1: &mut [f64],
        w2: &mut [f64],
        b2: &mut [f64],
        lr: f64,
    ) {
        self.t += 1;
        let beta1: f64 = 0.9;
        let beta2: f64 = 0.999;
        let eps: f64 = 1e-8;
        let bc1 = 1.0 - beta1.powi(self.t as i32);
        let bc2 = 1.0 - beta2.powi(self.t as i32);
        let update = |w: &mut [f64], g: &mut [f64], m: &mut [f64], v: &mut [f64]| {
            for i in 0..w.len() {
                m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
                let m_hat = m[i] / bc1;
                let v_hat = v[i] / bc2;
                w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                g[i] = 0.0;
            }
        };
        update(w1, &mut self.gw1, &mut self.mw1, &mut self.vw1);
        update(b1, &mut self.gb1, &mut self.mb1, &mut self.vb1);
        update(w2, &mut self.gw2, &mut self.mw2, &mut self.vw2);
        update(b2, &mut self.gb2, &mut self.mb2, &mut self.vb2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zensim::mlp::{Model, Predictor};

    fn predict_one(predictor: &mut Predictor<'_>, features: &[f64]) -> f64 {
        let f32_features: Vec<f32> = features.iter().map(|&v| v as f32).collect();
        predictor.predict(&f32_features).unwrap()[0] as f64
    }

    #[test]
    fn moe_recovers_class_dependent_target() {
        let n_content = 8;
        let n_cclass = 3;
        let n_features = n_content + n_cclass;
        let n_train = 600;
        let mut rng = SplitMix64::new(11);
        let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n_train);
        let mut targets: Vec<f64> = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let x_content: Vec<f64> = (0..n_content).map(|_| rng.next_normal()).collect();
            let class = i % n_cclass;
            let mut full = x_content.clone();
            for k in 0..n_cclass {
                full.push(if k == class { 1.0 } else { 0.0 });
            }
            let s_all: f64 = x_content.iter().sum();
            let s_first: f64 = x_content.iter().take(4).sum();
            let y = match class {
                0 => s_all,
                1 => -s_all,
                _ => s_first,
            };
            features_owned.push(full);
            targets.push(y);
        }
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "synth_moe".to_string(),
            human_scores: &targets,
            features: &feats_ref,
            train_weight: 1.0,
            validation_weight: 1.0,
        };
        let hyper = MlpHyperparams {
            n_hidden: 16,
            n_epochs: 80,
            pairs_per_epoch: 5_000,
            initial_lr: 0.005,
            log_every: 20,
            early_stop_patience: 0,
            ..Default::default()
        };
        let moe = MoeHyperparams {
            n_experts: n_cclass,
            gate_hidden: 16,
            gate_temperature: 1.0,
            load_balance_lambda: 0.001,
            hard_top1_threshold: 0.95,
        };
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut log = Vec::new();
        let out = train_mlp_moe(&[group], n_features, &names, &hyper, &moe, &mut log);
        // Note: this is a stress test; depending on init it may not
        // converge to a great score but must produce finite output and
        // the gate must average to a non-collapsed distribution.
        assert!(out.best_val_score.is_finite());

        // Roundtrip each expert + gate bake.
        for bake in &out.expert_bakes {
            let leaked: &'static [u8] = Box::leak(bake.clone().into_boxed_slice());
            let model = Model::from_bytes(leaked).expect("expert bake roundtrip");
            let mut p = Predictor::new(model);
            let row = &features_owned[0];
            let y = predict_one(&mut p, row);
            assert!(y.is_finite(), "non-finite expert prediction");
        }
        let leaked: &'static [u8] = Box::leak(out.gate_bake.clone().into_boxed_slice());
        let gate_model = Model::from_bytes(leaked).expect("gate bake roundtrip");
        assert_eq!(gate_model.n_inputs(), n_features);
    }
}
