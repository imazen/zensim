//! Two-layer MLP trainer for V0_4.
//!
//! Architecture: `n_features → n_hidden (LeakyReLU) → 1 (Identity)`.
//! Loss: RankNet pairwise (sigmoid cross-entropy on signed distance
//! deltas). Optimizer: Adam with cosine annealing.
//!
//! Output: a v1 ZNPK byte stream that loads via
//! [`zensim::mlp::Model::from_bytes`].
//!
//! This is the runtime-side counterpart to zenpicker's Python
//! distillation pipeline (`tools/train_hybrid.py`). Pure Rust, no
//! external numerics deps — the network is small enough (228×32 +
//! 32 = ~7.3K weights) that hand-rolled Adam is plenty fast.
//!
//! ## Multi-dataset training (V0_4 lessons from V0_2 audit)
//!
//! V0_2 was trained on a single concordant synthetic dataset; CMA-ES
//! hit higher synthetic SROCC than the shipped NM weights but lost
//! on KADIK / TID human holdouts. The takeaway for V0_4: ruthlessly
//! gate on human-dataset SROCC, not synthetic.
//!
//! This trainer takes [`TrainingGroup`]s with explicit `train_weight`.
//! Groups with `train_weight > 0` contribute to RankNet pair sampling
//! in proportion to their weight; groups with `train_weight == 0` are
//! validation-only — their per-epoch SROCC is logged and the best
//! model is the one with the highest validation mean.

use std::time::Instant;
use zensim::mlp::bake::{BakeLayer, BakeRequest, bake_v2};
use zensim::mlp::{Activation, WeightDtype};

/// How to aggregate per-group SROCC into the single value used for
/// best-checkpoint selection.
///
/// `Min` is the right default when shipping a metric: a model whose
/// worst dataset is bad will be observably bad in production. V0_4 v1
/// used `Mean` and ended up with a TID regression masked by KADIK and
/// CID22 wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationPolicy {
    Mean,
    Min,
}

/// Knobs for [`train_mlp`]. Defaults match the V0_4 placeholder
/// architecture (228 → 32 → 1) with `Min` validation gating.
#[derive(Clone, Debug)]
pub struct MlpHyperparams {
    pub n_hidden: usize,
    pub n_epochs: usize,
    pub pairs_per_epoch: usize,
    pub initial_lr: f64,
    pub leaky_alpha: f64,
    pub seed: u64,
    pub log_every: usize,
    /// L2 regularization on layer weights (not biases). 0 disables.
    pub l2_lambda: f64,
    /// Stop after this many epochs of no validation improvement.
    /// 0 disables early stopping.
    pub early_stop_patience: usize,
    pub validation_policy: ValidationPolicy,
}

impl Default for MlpHyperparams {
    fn default() -> Self {
        Self {
            n_hidden: 32,
            n_epochs: 200,
            pairs_per_epoch: 50_000,
            initial_lr: 0.001,
            leaky_alpha: 0.01,
            seed: 42,
            log_every: 10,
            l2_lambda: 1e-5,
            early_stop_patience: 50,
            validation_policy: ValidationPolicy::Min,
        }
    }
}

/// One named slice of training/validation data.
///
/// Multi-group training resolves V0_2's "synthetic dominates by 15×"
/// imbalance: per-step sampling picks a group in proportion to
/// `train_weight`, then samples a pair within.
///
/// `train_weight` and `validation_weight` are independent: a group
/// can be in both pools (trained on AND gated against), in only one,
/// or in neither (per-epoch SROCC still logged for transparency).
#[derive(Debug)]
pub struct TrainingGroup<'a> {
    pub name: String,
    pub human_scores: &'a [f64],
    pub features: &'a [&'a [f64]],
    /// Weight in the per-step group selection distribution. The
    /// per-pair sampling probability is `train_weight / total_weight`,
    /// so doubling `train_weight` doubles the sampling rate.
    /// Set to `0.0` to exclude this group from training.
    pub train_weight: f64,
    /// Weight in the per-epoch validation aggregation. `0.0` excludes
    /// the group from best-checkpoint scoring (it's still reported in
    /// the log). For `ValidationPolicy::Min`, weights act as a soft
    /// inclusion mask — any group with `validation_weight > 0`
    /// participates in the min.
    pub validation_weight: f64,
}

/// Train a 2-layer MLP across multiple datasets via RankNet pairwise
/// loss + Adam, with per-dataset SROCC tracking and best-checkpoint
/// selection on the validation mean.
///
/// `human_scores[i]` is the human-rated quality for pair `i`; HIGHER
/// means MORE similar to the source. The MLP must produce LOWER
/// raw_distance for higher-scored pairs.
///
/// Returns the bytes of the best-validation checkpoint (ZNPK v1).
pub fn train_mlp(
    groups: &[TrainingGroup<'_>],
    n_features: usize,
    hyperparams: &MlpHyperparams,
    log: &mut Vec<String>,
) -> Vec<u8> {
    let n_outputs = 1usize;
    let n_hidden = hyperparams.n_hidden;

    assert!(!groups.is_empty(), "need at least one training group");
    for g in groups {
        assert_eq!(g.human_scores.len(), g.features.len(), "{}: scores/features length mismatch", g.name);
        assert!(g.features.iter().all(|f| f.len() == n_features), "{}: feature length mismatch", g.name);
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
            "MLP train: arch=[{n_features} → {n_hidden} (LeakyReLU α={alpha}) → 1], val_policy={:?}",
            hyperparams.validation_policy,
            alpha = hyperparams.leaky_alpha,
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
                g.validation_weight,
            ),
            log,
        );
    }

    // 1. Compute per-feature scaler (mean / std) using ALL training-group
    //    samples. Validation-only groups are excluded from the scaler so
    //    we never look at validation data during fit.
    let (scaler_mean, scaler_scale) = compute_scaler_from_groups(groups, &train_indices, n_features);

    // 2. Standardize features per group up-front. Standardizing now
    //    avoids redoing it inside the per-step inner loop and lets the
    //    inner loop just slice into a flat f64 buffer per group.
    //    Group g's standardized features live in std_features[g], shape
    //    (n_pairs[g] × n_features).
    let std_features: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            let mut buf = vec![0.0f64; g.features.len() * n_features];
            for (i, &f) in g.features.iter().enumerate() {
                for d in 0..n_features {
                    buf[i * n_features + d] = (f[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
                }
            }
            buf
        })
        .collect();

    // 3. Initialize weights (Xavier-Glorot for tanh/leaky-relu).
    let mut rng = SplitMix64::new(hyperparams.seed);
    let std1 = (2.0 / (n_features + n_hidden) as f64).sqrt();
    let std2 = (2.0 / (n_hidden + n_outputs) as f64).sqrt();
    let mut w1 = (0..n_features * n_hidden)
        .map(|_| rng.next_normal() * std1)
        .collect::<Vec<_>>();
    let mut b1 = vec![0.0f64; n_hidden];
    let mut w2 = (0..n_hidden * n_outputs)
        .map(|_| rng.next_normal() * std2)
        .collect::<Vec<_>>();
    let mut b2 = vec![0.0f64; n_outputs];

    let mut adam = AdamState::new(w1.len(), b1.len(), w2.len(), b2.len());

    // 4. Training loop.
    let start = Instant::now();
    // The "validation score" is the mean SROCC across val_indices.
    // If there are no validation-only groups, fall back to the mean
    // across all groups (= training SROCC mean).
    let mut best_val_score = f64::NEG_INFINITY;
    let mut best_bake: Option<Vec<u8>> = None;
    let mut stale_epochs = 0usize;

    // Pre-compute the cumulative-distribution table for group sampling.
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

    for epoch in 0..hyperparams.n_epochs {
        let lr = hyperparams.initial_lr
            * 0.5
            * (1.0 + (std::f64::consts::PI * (epoch % 50) as f64 / 50.0).cos());

        let mut total_loss = 0.0f64;
        let mut n_steps = 0u64;

        for _ in 0..hyperparams.pairs_per_epoch {
            // Pick a training group via inverse-CDF sampling, then a
            // pair (ia, ib) within that group.
            let u = rng.next_f64_unit();
            let g_idx = train_indices[cdf.partition_point(|&c| c < u).min(cdf.len() - 1)];
            let g = &groups[g_idx];
            let n = g.features.len();
            if n < 2 {
                continue;
            }
            let ia = (rng.next_u64() as usize) % n;
            let ib = (rng.next_u64() as usize) % n;
            if ia == ib {
                continue;
            }

            let g_feats = &std_features[g_idx];
            let xa = &g_feats[ia * n_features..(ia + 1) * n_features];
            let xb = &g_feats[ib * n_features..(ib + 1) * n_features];
            let (ya, ha_pre, ha) = forward(xa, &w1, &b1, &w2, &b2, n_features, n_hidden, hyperparams.leaky_alpha);
            let (yb, hb_pre, hb) = forward(xb, &w1, &b1, &w2, &b2, n_features, n_hidden, hyperparams.leaky_alpha);

            let target = (g.human_scores[ia] - g.human_scores[ib]).signum();
            if target == 0.0 {
                continue;
            }
            let pred_diff = yb - ya;
            let z = -target * pred_diff;
            let loss = if z > 50.0 { z } else if z < -50.0 { 0.0 } else { (z.exp() + 1.0).ln() };
            total_loss += loss;
            n_steps += 1;

            let sig_z = 1.0 / (1.0 + (-z).exp());
            let dl_d_pred_diff = -target * sig_z;
            let dl_dya = -dl_d_pred_diff;
            let dl_dyb = dl_d_pred_diff;

            backprop_step(
                xa, &ha_pre, &ha, dl_dya,
                &w1, &mut adam.gw1, &mut adam.gb1, &w2, &mut adam.gw2, &mut adam.gb2,
                n_features, n_hidden, hyperparams.leaky_alpha,
            );
            backprop_step(
                xb, &hb_pre, &hb, dl_dyb,
                &w1, &mut adam.gw1, &mut adam.gb1, &w2, &mut adam.gw2, &mut adam.gb2,
                n_features, n_hidden, hyperparams.leaky_alpha,
            );

            if hyperparams.l2_lambda > 0.0 {
                for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                    *g += hyperparams.l2_lambda * w;
                }
                for (g, &w) in adam.gw2.iter_mut().zip(w2.iter()) {
                    *g += hyperparams.l2_lambda * w;
                }
            }

            adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
        }

        let avg_loss = if n_steps > 0 { total_loss / n_steps as f64 } else { 0.0 };

        if epoch % hyperparams.log_every == 0 || epoch == hyperparams.n_epochs - 1 {
            // Per-group SROCC. The MLP outputs raw_distance (lower =
            // more similar); human_scores are quality (higher = more
            // similar). They're anti-correlated by design, so we
            // compute SROCC against `-predictions` to surface positive
            // numbers that match V0_2's reporting convention.
            let group_srocc: Vec<f64> = groups
                .iter()
                .enumerate()
                .map(|(gi, g)| {
                    let preds = predict_group(
                        &std_features[gi], g.features.len(), n_features,
                        &w1, &b1, &w2, &b2, n_hidden, hyperparams.leaky_alpha,
                    );
                    let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
                    spearman_correlation(g.human_scores, &neg_preds)
                })
                .collect();

            // Validation score across val groups (validation_weight > 0).
            // If no val groups configured, fall back to mean across
            // all groups so the trainer still has a checkpoint signal.
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

            let elapsed = start.elapsed().as_secs_f64();
            let per_group = group_srocc
                .iter()
                .zip(groups.iter())
                .map(|(s, g)| format!("{}={s:.4}", g.name))
                .collect::<Vec<_>>()
                .join(" ");
            log_line(
                &format!(
                    "  epoch {epoch:>3} | lr={lr:.5} | loss={avg_loss:.4} | val_mean={val_score:.4} (best={best_val_score:.4}) | {per_group} | t={elapsed:.1}s"
                ),
                log,
            );

            if val_score > best_val_score {
                best_val_score = val_score;
                stale_epochs = 0;
                best_bake = Some(bake_two_layer_znpr_v2(
                    &scaler_mean,
                    &scaler_scale,
                    &w1,
                    &b1,
                    &w2,
                    &b2,
                    n_features,
                    n_hidden,
                    n_outputs,
                ));
            } else {
                stale_epochs += hyperparams.log_every;
                if hyperparams.early_stop_patience > 0 && stale_epochs >= hyperparams.early_stop_patience {
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

    log_line(&format!("MLP train: best validation mean SROCC = {best_val_score:.4}"), log);
    best_bake.unwrap_or_else(|| {
        bake_two_layer_znpr_v2(
            &scaler_mean,
            &scaler_scale,
            &w1,
            &b1,
            &w2,
            &b2,
            n_features,
            n_hidden,
            n_outputs,
        )
    })
}

/// Bake a 2-layer MLP (LeakyReLU → Identity) into ZNPR v2 bytes.
/// Converts f64 weights to f32 once and feeds them to [`bake_v2`].
#[allow(clippy::too_many_arguments)]
fn bake_two_layer_znpr_v2(
    scaler_mean: &[f64],
    scaler_scale: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_inputs: usize,
    n_hidden: usize,
    n_outputs: usize,
) -> Vec<u8> {
    let scaler_mean_f32: Vec<f32> = scaler_mean.iter().map(|&v| v as f32).collect();
    let scaler_scale_f32: Vec<f32> = scaler_scale.iter().map(|&v| v as f32).collect();
    let w1_f32: Vec<f32> = w1.iter().map(|&v| v as f32).collect();
    let b1_f32: Vec<f32> = b1.iter().map(|&v| v as f32).collect();
    let w2_f32: Vec<f32> = w2.iter().map(|&v| v as f32).collect();
    let b2_f32: Vec<f32> = b2.iter().map(|&v| v as f32).collect();
    let layers = [
        BakeLayer {
            in_dim: n_inputs,
            out_dim: n_hidden,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F32,
            weights: &w1_f32,
            biases: &b1_f32,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_outputs,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w2_f32,
            biases: &b2_f32,
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
    })
    .expect("v2 bake of 2-layer MLP")
}

fn compute_scaler_from_groups(
    groups: &[TrainingGroup<'_>],
    train_indices: &[usize],
    n_features: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut count = 0u64;
    let mut mean = vec![0.0f64; n_features];
    for &gi in train_indices {
        for f in groups[gi].features {
            for d in 0..n_features {
                mean[d] += f[d];
            }
            count += 1;
        }
    }
    let n = count.max(1) as f64;
    for m in &mut mean {
        *m /= n;
    }
    let mut var = vec![0.0f64; n_features];
    for &gi in train_indices {
        for f in groups[gi].features {
            for d in 0..n_features {
                let dx = f[d] - mean[d];
                var[d] += dx * dx;
            }
        }
    }
    let std = var.iter().map(|&v| (v / n).sqrt().max(1e-8)).collect();
    (mean, std)
}

#[allow(clippy::too_many_arguments)]
fn forward(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    let mut h_pre = b1.to_vec();
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for (acc, &w) in h_pre.iter_mut().zip(row.iter()) {
            *acc += s * w;
        }
    }
    let h: Vec<f64> = h_pre
        .iter()
        .map(|&v| if v >= 0.0 { v } else { alpha * v })
        .collect();
    let mut y = b2[0];
    for o in 0..n_hidden {
        y += h[o] * w2[o];
    }
    (y, h_pre, h)
}

#[allow(clippy::too_many_arguments)]
fn backprop_step(
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    dl_dy: f64,
    _w1: &[f64],
    gw1: &mut [f64],
    gb1: &mut [f64],
    w2: &[f64],
    gw2: &mut [f64],
    gb2: &mut [f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) {
    for o in 0..n_hidden {
        gw2[o] += dl_dy * h[o];
    }
    gb2[0] += dl_dy;

    let mut dl_dh_pre = vec![0.0f64; n_hidden];
    for o in 0..n_hidden {
        let dh = dl_dy * w2[o];
        dl_dh_pre[o] = if h_pre[o] >= 0.0 { dh } else { alpha * dh };
    }

    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &mut gw1[i * n_hidden..(i + 1) * n_hidden];
        for (g, &dh) in row.iter_mut().zip(dl_dh_pre.iter()) {
            *g += s * dh;
        }
    }
    for (g, &dh) in gb1.iter_mut().zip(dl_dh_pre.iter()) {
        *g += dh;
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_group(
    std_x: &[f64],
    n_pairs: usize,
    n_features: usize,
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_hidden: usize,
    alpha: f64,
) -> Vec<f64> {
    (0..n_pairs)
        .map(|i| {
            let xi = &std_x[i * n_features..(i + 1) * n_features];
            let (y, _, _) = forward(xi, w1, b1, w2, b2, n_features, n_hidden, alpha);
            y
        })
        .collect()
}

fn spearman_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let mean_a = (n as f64 - 1.0) / 2.0;
    let mean_b = mean_a;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = ra[i] - mean_a;
        let xb = rb[i] - mean_b;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (v[idx[j]] - v[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let avg = (i + j - 1) as f64 / 2.0;
        for k in i..j {
            r[idx[k]] = avg;
        }
        i = j;
    }
    r
}

struct AdamState {
    gw1: Vec<f64>,
    gb1: Vec<f64>,
    gw2: Vec<f64>,
    gb2: Vec<f64>,
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

impl AdamState {
    fn new(nw1: usize, nb1: usize, nw2: usize, nb2: usize) -> Self {
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

    fn step(
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

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64_unit().max(1e-12);
        let u2 = self.next_f64_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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

    /// Generate a synthetic dataset where the target is a known
    /// nonlinear function of features. Train a small MLP and confirm
    /// it recovers the ranking.
    #[test]
    fn train_mlp_recovers_synthetic_ranking() {
        let n_features = 16;
        let n_train = 300;
        let mut rng = SplitMix64::new(7);
        let true_w: Vec<f64> = (0..n_features)
            .map(|i| (i as f64 - 8.0) * 0.3 + rng.next_normal() * 0.1)
            .collect();
        let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n_train);
        let mut targets: Vec<f64> = Vec::with_capacity(n_train);
        for _ in 0..n_train {
            let x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            let mut y: f64 = x.iter().zip(true_w.iter()).map(|(a, b)| a * b).sum();
            y += 0.1 * x[0] * x[0];
            y += rng.next_normal() * 0.05;
            features_owned.push(x);
            // Higher target = more similar (matches zensim convention).
            // The MLP must produce LOWER raw_distance for higher target.
            targets.push(y);
        }
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: &feats_ref,
            train_weight: 1.0,
            validation_weight: 1.0,
        };

        let hyper = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 60,
            pairs_per_epoch: 1500,
            initial_lr: 0.005,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            ..Default::default()
        };
        let mut log = Vec::new();
        let bytes = train_mlp(&[group], n_features, &hyper, &mut log);

        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let model = Model::from_bytes(leaked).expect("bake should load");
        let mut predictor = Predictor::new(model);

        let preds: Vec<f64> = features_owned
            .iter()
            .map(|f| predict_one(&mut predictor, f))
            .collect();

        // MLP output is raw_distance (lower = more similar); targets
        // are quality (higher = more similar). Anti-correlated → SROCC
        // computed against negated predictions.
        let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
        let srocc = spearman_correlation(&targets, &neg_preds);
        assert!(
            srocc > 0.85,
            "MLP failed to recover synthetic ranking: SROCC={srocc:.4}"
        );
    }

    /// A validation-only group should not contribute training pairs
    /// but should still be reported in per-epoch SROCC, and the best
    /// checkpoint should be selected on its score.
    #[test]
    fn train_mlp_uses_validation_for_best_checkpoint() {
        let n_features = 8;
        let mut rng = SplitMix64::new(11);

        // Train data: target = +sum_of_features (higher = more similar).
        let train_features: Vec<Vec<f64>> = (0..200)
            .map(|_| (0..n_features).map(|_| rng.next_normal()).collect())
            .collect();
        let train_scores: Vec<f64> = train_features
            .iter()
            .map(|f| f.iter().sum::<f64>())
            .collect();
        let train_refs: Vec<&[f64]> = train_features.iter().map(|v| v.as_slice()).collect();

        // Val data: same target function — model should generalize.
        let val_features: Vec<Vec<f64>> = (0..80)
            .map(|_| (0..n_features).map(|_| rng.next_normal()).collect())
            .collect();
        let val_scores: Vec<f64> = val_features
            .iter()
            .map(|f| f.iter().sum::<f64>())
            .collect();
        let val_refs: Vec<&[f64]> = val_features.iter().map(|v| v.as_slice()).collect();

        let groups = vec![
            TrainingGroup {
                name: "train".to_string(),
                human_scores: &train_scores,
                features: &train_refs,
                train_weight: 1.0,
                validation_weight: 0.0,
            },
            TrainingGroup {
                name: "val".to_string(),
                human_scores: &val_scores,
                features: &val_refs,
                train_weight: 0.0,
                validation_weight: 1.0,
            },
        ];

        let hyper = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 40,
            pairs_per_epoch: 800,
            initial_lr: 0.005,
            log_every: 10,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Min,
            ..Default::default()
        };
        let mut log = Vec::new();
        let bytes = train_mlp(&groups, n_features, &hyper, &mut log);

        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let model = Model::from_bytes(leaked).expect("bake should load");
        let mut predictor = Predictor::new(model);

        let val_preds: Vec<f64> = val_features
            .iter()
            .map(|f| predict_one(&mut predictor, f))
            .collect();
        // Higher target ⇒ more similar ⇒ lower MLP raw_distance.
        // SROCC(target, -prediction) should be positive and high.
        let neg_preds: Vec<f64> = val_preds.iter().map(|&p| -p).collect();
        let val_srocc = spearman_correlation(&val_scores, &neg_preds);
        assert!(
            val_srocc > 0.85,
            "validation-tracking trainer failed to generalize: val SROCC={val_srocc:.4}",
        );
        // Spot-check the log: should mention val= per group and report
        // val_mean as part of every epoch line.
        assert!(
            log.iter().any(|line| line.contains("val_mean=")),
            "log missing val_mean= reporting"
        );
        assert!(
            log.iter().any(|line| line.contains("val=")),
            "log missing per-group val= field"
        );
    }
}
