//! EX-2 std-pool head: replace the MLP's final scalar with 4 pooled stats
//! `[μ, σ, max, p_6]` of the implicit per-feature quality signal, then a
//! 4→1 reducer.
//!
//! Architecture:
//! ```text
//! x (n_inputs) → W1 (n_inputs × n_hidden) → LeakyReLU(α) → h (n_hidden)
//!              → pool: [μ, σ, max, p_6] (4 scalars over h)
//!              → reducer (4 × 1) → y (scalar)
//! ```
//!
//! Pool stats:
//! - **μ** = mean(h)
//! - **σ** = sqrt(mean((h − μ)²))  — population std (no Bessel correction)
//! - **max** = max(h)
//! - **p_6** = (mean(h^6))^(1/6) — Butteraugli-style 6-norm
//!
//! Per `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md §3` (GMSD std-pooling,
//! Butteraugli p-norm trick, IW-SSIM info weights). Doc constants
//! lifted verbatim: `c = 0.0026` for GMSD 8-bit normalized (used as the
//! σ-floor scale here), Butteraugli p-norm exponent `6` from `{3, 6, 12}`.
//!
//! Bake format: the existing two-layer MLP `bake_two_layer_znpr_v3`
//! emits the model with n_outputs = n_hidden (the second layer is an
//! identity passthrough acting as "store hidden vector verbatim"). The
//! pool + reducer math lives **outside** the bake: the 4 reducer
//! weights and bias are serialized as v3 metadata under
//! `zentrain.pool_head_reducer`. The runtime reads metadata, runs
//! `predictor.predict()` to get the hidden vector, computes the 4
//! pool stats, and applies the reducer.
//!
//! This module is **scalar only** (no SIMD). A SIMD path can be
//! grafted onto the hidden→pool computation later; the per-pixel cost
//! is one Linear + one LeakyReLU + four reductions per sample, which
//! is dwarfed by the upstream 228-feature extractor in zensim.

use crate::TrainingGroup;
use crate::adam::AdamState;
use crate::mlp::compute_scaler_from_groups;
use crate::rng::SplitMix64;
use zenpredict::{Activation, MetadataType, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

/// Per-bake constants for the pool-head. Pool stats are fixed-functional
/// so only the **p-norm exponent** is a parameter; we hard-code 6 per
/// the doc but expose it for future ablations.
pub const POOL_P_NORM: f64 = 6.0;

/// GMSD's 8-bit normalized stability constant. Used as the σ-floor
/// (per-sample) so that std doesn't drop to 0 on degenerate hidden
/// vectors. Doc §3 constants table: "GMSD constant c = 0.0026".
pub const POOL_STD_FLOOR: f64 = 0.0026;

/// 4 pooled stats `[μ, σ, max, p_6]` computed over the hidden vector.
///
/// Returns `(stats, max_index)`. `max_index` is the position of the
/// max element; the backprop path needs it to route ∂L/∂max to a
/// single hidden activation. The other three stats route gradient to
/// every hidden activation.
///
/// `h` is the post-LeakyReLU hidden vector. We accept LeakyReLU's
/// negative tail (it can produce negative outputs); `max` then picks
/// the most-positive activation and `p_6` operates on `|h|^6` so it
/// stays well-defined.
pub fn pool_stats(h: &[f64]) -> ([f64; 4], usize) {
    let n = h.len() as f64;
    debug_assert!(n > 0.0, "pool_stats requires non-empty hidden vector");
    let mean: f64 = h.iter().sum::<f64>() / n;
    let var: f64 = h.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std = var.sqrt().max(POOL_STD_FLOOR);
    let (max_idx, &max_val) = h
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
        .expect("non-empty hidden vector");
    // p_6 = (mean(|h|^6))^(1/6); operating on |h| keeps it well-defined
    // and matches Butteraugli's intent (sensitivity to peak error).
    // |v|^6 = (v^2)^3 — exact for the even POOL_P_NORM=6 and free of the
    // per-element pow/log/exp transcendental (12.8 M of them per epoch on
    // the 128-hidden × 50 k-pair × 2-forward shape). LLVM auto-vectorizes
    // the multiply form; powf does not.
    debug_assert!(
        (POOL_P_NORM - 6.0).abs() < 1e-9,
        "fast |h|^6 path assumes POOL_P_NORM == 6.0"
    );
    let sum_p: f64 = h
        .iter()
        .map(|&v| {
            let v2 = v * v;
            v2 * v2 * v2
        })
        .sum();
    let mean_p = sum_p / n;
    let p_norm = mean_p.powf(1.0 / POOL_P_NORM);
    ([mean, std, max_val, p_norm], max_idx)
}

/// Apply the 4→1 reducer to the pool stats.
///
/// `reducer_w` is `[w_μ, w_σ, w_max, w_p6]`; `reducer_b` is the scalar
/// bias. Output is `w · stats + b`. The reducer is a plain linear
/// layer (no activation) because the calling code interprets `y` as a
/// score-shaped scalar in the same way the legacy 2-layer MLP did.
#[inline]
pub fn apply_reducer(stats: &[f64; 4], reducer_w: &[f64; 4], reducer_b: f64) -> f64 {
    stats[0] * reducer_w[0]
        + stats[1] * reducer_w[1]
        + stats[2] * reducer_w[2]
        + stats[3] * reducer_w[3]
        + reducer_b
}

/// Full forward pass: `x → h → pool stats → y`.
///
/// Returns `(y, h_pre, h, stats, max_idx)` — every intermediate needed
/// by `backprop_step_pool_head`. `h_pre` is the pre-LeakyReLU state.
#[allow(clippy::too_many_arguments)]
pub fn forward_pool_head(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    reducer_w: &[f64; 4],
    reducer_b: f64,
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) -> (f64, Vec<f64>, Vec<f64>, [f64; 4], usize) {
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
    let (stats, max_idx) = pool_stats(&h);
    let y = apply_reducer(&stats, reducer_w, reducer_b);
    (y, h_pre, h, stats, max_idx)
}

/// Backprop ∂L/∂y through the std-pool head to gradients on (w1, b1,
/// reducer_w, reducer_b).
///
/// Gradient chain rule per pool stat (n = n_hidden):
///
/// - ∂μ/∂h_j   = 1/n
/// - ∂σ/∂h_j   = (h_j − μ) / (n · σ)     (with σ floored — zero gradient
///                                         contribution when σ == floor)
/// - ∂max/∂h_j = 1 if j == max_idx else 0
/// - ∂p_6/∂h_j = sign(h_j) · (|h_j|^5) / (n · p_6^5)   (with p_6 floored
///                                                      at 1e-12 to
///                                                      avoid div-by-0)
///
/// `dl_dy` is `∂L/∂y` from upstream (RankNet). Per stat:
/// `∂L/∂stat_k = dl_dy · reducer_w[k]`.
///
/// Per hidden unit j: ∂L/∂h_j = Σ_k (∂L/∂stat_k · ∂stat_k/∂h_j).
/// Then LeakyReLU's slope routes ∂L/∂h_j → ∂L/∂h_pre_j.
/// Finally accumulate into gw1, gb1, and reducer gradients.
#[allow(clippy::too_many_arguments)]
pub fn backprop_step_pool_head(
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    stats: &[f64; 4],
    max_idx: usize,
    dl_dy: f64,
    reducer_w: &[f64; 4],
    gw1: &mut [f64],
    gb1: &mut [f64],
    g_reducer_w: &mut [f64; 4],
    g_reducer_b: &mut f64,
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) {
    // Reducer gradients (4 weights + 1 bias).
    for k in 0..4 {
        g_reducer_w[k] += dl_dy * stats[k];
    }
    *g_reducer_b += dl_dy;

    // ∂L/∂stat_k (= dl_dy · reducer_w[k]).
    let dl_dstat: [f64; 4] = [
        dl_dy * reducer_w[0],
        dl_dy * reducer_w[1],
        dl_dy * reducer_w[2],
        dl_dy * reducer_w[3],
    ];

    let n = n_hidden as f64;
    let mu = stats[0];
    let sigma = stats[1];
    let p6 = stats[2 + 1]; // index 3 = p_6
    // Avoid div-by-0 in σ and p_6 gradients.
    let inv_sigma_n = if sigma > POOL_STD_FLOOR + 1e-12 {
        1.0 / (n * sigma)
    } else {
        0.0
    };
    let p6_floor = p6.max(1e-12);
    let inv_p6_pow5_n = 1.0 / (n * p6_floor.powi(5));

    // Per-hidden-unit ∂L/∂h_j accumulator.
    let mut dl_dh = vec![0.0f64; n_hidden];
    for j in 0..n_hidden {
        let hj = h[j];
        // ∂μ/∂h_j = 1/n.
        dl_dh[j] += dl_dstat[0] / n;
        // ∂σ/∂h_j = (h_j − μ) / (n · σ).
        dl_dh[j] += dl_dstat[1] * (hj - mu) * inv_sigma_n;
        // ∂max/∂h_j = δ(j, max_idx).
        if j == max_idx {
            dl_dh[j] += dl_dstat[2];
        }
        // ∂p_6/∂h_j = sign(h_j) · |h_j|^5 / (n · p_6^5).
        let abs_hj = hj.abs();
        let sign_hj = if hj >= 0.0 { 1.0 } else { -1.0 };
        let abs_pow5 = abs_hj * abs_hj * abs_hj * abs_hj * abs_hj;
        dl_dh[j] += dl_dstat[3] * sign_hj * abs_pow5 * inv_p6_pow5_n;
    }

    // LeakyReLU back-route ∂L/∂h_j → ∂L/∂h_pre_j.
    let dl_dh_pre: Vec<f64> = dl_dh
        .iter()
        .zip(h_pre.iter())
        .map(|(&dh, &hp)| if hp >= 0.0 { dh } else { alpha * dh })
        .collect();

    // Layer-1 gradients (gw1: n_features × n_hidden row-major; gb1: n_hidden).
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

/// Hyperparameters for the std-pool MLP trainer.
#[derive(Clone, Debug)]
pub struct PoolHeadHparams {
    /// Hidden vector width. Doc §3 EX-2: ~30 new parameters → matches a
    /// 4→1 reducer; doesn't constrain hidden width. Default 128
    /// matches V_22-mix-LARGE.
    pub n_hidden: usize,
    /// Number of epochs.
    pub n_epochs: usize,
    /// RankNet pair samples per epoch.
    pub pairs_per_epoch: usize,
    /// Adam initial learning rate.
    pub initial_lr: f64,
    /// LeakyReLU negative-side slope.
    pub leaky_alpha: f64,
    /// PRNG seed.
    pub seed: u64,
    /// L2 regularization (excludes biases). 0 disables.
    pub l2_lambda: f64,
}

impl Default for PoolHeadHparams {
    fn default() -> Self {
        Self {
            n_hidden: 128,
            n_epochs: 200,
            pairs_per_epoch: 50_000,
            initial_lr: 1e-3,
            leaky_alpha: 0.01,
            seed: 1,
            l2_lambda: 1e-5,
        }
    }
}

/// Output of [`train_pool_head`]: learned weights + bake metadata.
#[derive(Debug)]
pub struct PoolHeadModel {
    /// Per-feature mean (n_features).
    pub scaler_mean: Vec<f64>,
    /// Per-feature std (n_features), floored at 1e-8.
    pub scaler_scale: Vec<f64>,
    /// Layer-1 weights, row-major (n_features × n_hidden).
    pub w1: Vec<f64>,
    /// Layer-1 biases (n_hidden).
    pub b1: Vec<f64>,
    /// 4→1 reducer weights `[w_μ, w_σ, w_max, w_p6]`.
    pub reducer_w: [f64; 4],
    /// 4→1 reducer bias.
    pub reducer_b: f64,
    /// Hidden width.
    pub n_hidden: usize,
    /// Input dim.
    pub n_features: usize,
}

impl PoolHeadModel {
    /// Initialize a fresh model with He-ish init on w1 (scaled normal),
    /// zeros on b1, and reducer weights `[0.0, 1.0, 0.0, 0.0]` —
    /// std-pool is the doc's named winner (GMSD insight), so start
    /// there. Other reducer weights start at 0 + small jitter.
    pub fn new(n_features: usize, n_hidden: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let scale = (2.0 / n_features as f64).sqrt();
        let n_w1 = n_features * n_hidden;
        let mut w1 = Vec::with_capacity(n_w1);
        for _ in 0..n_w1 {
            // Box-Muller from two u64 → uniform → standard normal.
            let u1 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let u2 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let u1 = u1.max(1e-12);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();
            w1.push(z * scale);
        }
        // Reducer init: std-pool dominant; other channels small.
        let reducer_w: [f64; 4] = [0.05, 1.0, 0.05, 0.05];
        Self {
            scaler_mean: vec![0.0; n_features],
            scaler_scale: vec![1.0; n_features],
            w1,
            b1: vec![0.0; n_hidden],
            reducer_w,
            reducer_b: 0.0,
            n_hidden,
            n_features,
        }
    }
}

/// Train a std-pool-head MLP via RankNet.
///
/// Pair sampling per step: weighted-uniform across `groups` by
/// `train_weight`, then uniform within group. Loss per pair `(a, b)`:
///
/// ```text
/// y_a, y_b  ← forward_pool_head(x_a, ...), forward_pool_head(x_b, ...)
/// d = y_a − y_b
/// label = sign(score_a − score_b)
/// loss  = log(1 + exp(−label · d))
/// ∂L/∂d = −label · σ(−label · d)
/// ∂L/∂y_a = ∂L/∂d
/// ∂L/∂y_b = −∂L/∂d
/// ```
///
/// Notable simplifications vs the production trainer
/// (`zensim_mlp_train`):
/// - No TV regularizer (defer to V_25; baseline lift first).
/// - No multi-output / PWRC head.
/// - No mini-batch parallelism (single-thread for simplicity; the
///   pool-head per-sample backprop is fast enough at h=128).
///
/// Returns the trained [`PoolHeadModel`].
pub fn train_pool_head(
    groups: &[TrainingGroup<'_>],
    hparams: &PoolHeadHparams,
    n_features: usize,
) -> PoolHeadModel {
    let train_indices: Vec<usize> = (0..groups.len())
        .filter(|&i| groups[i].train_weight > 0.0)
        .collect();
    assert!(
        !train_indices.is_empty(),
        "train_pool_head: at least one group must have train_weight > 0"
    );

    let (mean, scale) = compute_scaler_from_groups(groups, &train_indices, n_features);

    let mut model = PoolHeadModel::new(n_features, hparams.n_hidden, hparams.seed);
    model.scaler_mean = mean;
    model.scaler_scale = scale;

    // Standardize all training rows up-front (allocations OK at h=128 + 200k pairs).
    let std_groups: Vec<Vec<Vec<f64>>> = train_indices
        .iter()
        .map(|&gi| {
            let g = &groups[gi];
            g.features
                .iter()
                .map(|row| {
                    (0..n_features)
                        .map(|d| (row[d] - model.scaler_mean[d]) / model.scaler_scale[d])
                        .collect::<Vec<f64>>()
                })
                .collect()
        })
        .collect();
    let train_scores: Vec<&[f64]> = train_indices
        .iter()
        .map(|&gi| groups[gi].human_scores)
        .collect();
    let train_weights: Vec<f64> = train_indices
        .iter()
        .map(|&gi| groups[gi].train_weight)
        .collect();
    let total_w: f64 = train_weights.iter().sum();
    let weight_cdf: Vec<f64> = train_weights
        .iter()
        .scan(0.0, |acc, &w| {
            *acc += w / total_w;
            Some(*acc)
        })
        .collect();

    let n_hidden = hparams.n_hidden;
    let nw1 = n_features * n_hidden;
    let nb1 = n_hidden;
    // Reducer gradients go in dedicated slots (4 weights + 1 bias).
    let mut adam = AdamState::new(nw1, nb1, 4, 1);
    let mut rng = SplitMix64::new(hparams.seed ^ 0xA5A5_A5A5_A5A5_A5A5);

    let l2 = hparams.l2_lambda;
    let alpha = hparams.leaky_alpha;
    let lr = hparams.initial_lr;
    let n_train_groups = train_indices.len();

    // RankNet loop.
    for _epoch in 0..hparams.n_epochs {
        for _pair in 0..hparams.pairs_per_epoch {
            // Pick a group by CDF.
            let r = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let mut gi = 0usize;
            for (k, &c) in weight_cdf.iter().enumerate() {
                if r < c {
                    gi = k;
                    break;
                }
                gi = k;
            }
            let _ = n_train_groups;
            let g_rows = &std_groups[gi];
            let g_scores = train_scores[gi];
            if g_rows.len() < 2 {
                continue;
            }
            // Pick two distinct indices.
            let len = g_rows.len();
            let ia = (rng.next_u64() as usize) % len;
            let mut ib = (rng.next_u64() as usize) % len;
            if ib == ia {
                ib = (ib + 1) % len;
            }
            if (g_scores[ia] - g_scores[ib]).abs() < 1e-9 {
                continue; // skip ties
            }
            let (ihi, ilo) = if g_scores[ia] > g_scores[ib] {
                (ia, ib)
            } else {
                (ib, ia)
            };
            let xhi = &g_rows[ihi];
            let xlo = &g_rows[ilo];

            let (yhi, hp_hi, h_hi, stats_hi, max_idx_hi) = forward_pool_head(
                xhi,
                &model.w1,
                &model.b1,
                &model.reducer_w,
                model.reducer_b,
                n_features,
                n_hidden,
                alpha,
            );
            let (ylo, hp_lo, h_lo, stats_lo, max_idx_lo) = forward_pool_head(
                xlo,
                &model.w1,
                &model.b1,
                &model.reducer_w,
                model.reducer_b,
                n_features,
                n_hidden,
                alpha,
            );
            // We want y_hi > y_lo (label=+1).
            let d = yhi - ylo;
            // L = log(1 + exp(-d)); ∂L/∂d = -1/(1+exp(d)) = -σ(-d).
            let sig_neg_d = 1.0 / (1.0 + d.exp());
            let dl_dd = -sig_neg_d;
            let dl_dyhi = dl_dd;
            let dl_dylo = -dl_dd;

            // Backprop into shared model.
            // Use placeholder reducer-grad arrays sized 4.
            let mut g_red_w: [f64; 4] = [0.0; 4];
            let mut g_red_b: f64 = 0.0;
            backprop_step_pool_head(
                xhi,
                &hp_hi,
                &h_hi,
                &stats_hi,
                max_idx_hi,
                dl_dyhi,
                &model.reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_red_w,
                &mut g_red_b,
                n_features,
                n_hidden,
                alpha,
            );
            backprop_step_pool_head(
                xlo,
                &hp_lo,
                &h_lo,
                &stats_lo,
                max_idx_lo,
                dl_dylo,
                &model.reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_red_w,
                &mut g_red_b,
                n_features,
                n_hidden,
                alpha,
            );
            // L2 on w1 and reducer_w (not biases). Apply lazily so it
            // matches the loss formulation `L_total = L_rank + λ · ||w||²`.
            if l2 > 0.0 {
                for (g, &w) in adam.gw1.iter_mut().zip(model.w1.iter()) {
                    *g += 2.0 * l2 * w;
                }
                for (g, &w) in g_red_w.iter_mut().zip(model.reducer_w.iter()) {
                    *g += 2.0 * l2 * w;
                }
            }
            // Accumulate reducer grads into Adam's "w2"/"b2" slots
            // (sized 4 + 1). The Adam step then updates reducer_w and
            // reducer_b alongside w1 and b1 in one call.
            for k in 0..4 {
                adam.gw2[k] += g_red_w[k];
            }
            adam.gb2[0] += g_red_b;

            // One Adam step per pair (matches the in-tree
            // mini-batch=1 reference policy).
            let mut r_w_vec: Vec<f64> = model.reducer_w.to_vec();
            let mut r_b_vec: Vec<f64> = vec![model.reducer_b];
            adam.step(&mut model.w1, &mut model.b1, &mut r_w_vec, &mut r_b_vec, lr);
            model.reducer_w = [r_w_vec[0], r_w_vec[1], r_w_vec[2], r_w_vec[3]];
            model.reducer_b = r_b_vec[0];
        }
    }

    model
}

/// Bake the std-pool-head model into ZNPR v3 bytes.
///
/// Wire format:
/// - **Layer 1**: `n_features → n_hidden`, activation `LeakyRelu`, dtype `F32`.
/// - **Layer 2 (passthrough)**: `n_hidden → n_hidden`, activation `Identity`,
///   identity matrix on the diagonal, zero biases. This makes
///   `predictor.predict()` return the LeakyReLU hidden activations,
///   which the runtime then feeds into pool + reducer.
/// - **Metadata**: one `zentrain.pool_head_reducer` entry, dtype
///   `Numeric` (f32 array), payload `[w_μ, w_σ, w_max, w_p6, b, p_norm]`.
///   6 floats = 24 bytes.
///
/// The runtime side recognizes the metadata key and routes through the
/// pool-head forward. Bakes without this metadata key remain
/// scalar-output MLPs (existing path).
pub fn bake_pool_head_v3(model: &PoolHeadModel) -> Vec<u8> {
    let n_features = model.n_features;
    let n_hidden = model.n_hidden;

    let scaler_mean_f32: Vec<f32> = model.scaler_mean.iter().map(|&v| v as f32).collect();
    let scaler_scale_f32: Vec<f32> = model.scaler_scale.iter().map(|&v| v as f32).collect();
    let w1_f32: Vec<f32> = model.w1.iter().map(|&v| v as f32).collect();
    let b1_f32: Vec<f32> = model.b1.iter().map(|&v| v as f32).collect();
    // Identity passthrough layer: `n_hidden × n_hidden` identity matrix,
    // zero biases. Row-major: `w2[i * n_hidden + j] = (i == j) ? 1 : 0`.
    let mut w2_f32 = vec![0.0f32; n_hidden * n_hidden];
    for i in 0..n_hidden {
        w2_f32[i * n_hidden + i] = 1.0;
    }
    let b2_f32 = vec![0.0f32; n_hidden];

    // Reducer payload: [w_μ, w_σ, w_max, w_p6, b, p_norm] as f32 little-endian.
    let mut reducer_payload = Vec::with_capacity(6 * 4);
    for v in &model.reducer_w {
        reducer_payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    reducer_payload.extend_from_slice(&(model.reducer_b as f32).to_le_bytes());
    reducer_payload.extend_from_slice(&(POOL_P_NORM as f32).to_le_bytes());

    let layers = [
        BakeLayer {
            in_dim: n_features,
            out_dim: n_hidden,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F32,
            weights: &w1_f32,
            biases: &b1_f32,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_hidden,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w2_f32,
            biases: &b2_f32,
        },
    ];
    let metadata = [BakeMetadataEntry {
        key: "zentrain.pool_head_reducer",
        kind: MetadataType::Numeric,
        value: &reducer_payload,
    }];
    bake(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean_f32,
        scaler_scale: &scaler_scale_f32,
        layers: &layers,
        feature_bounds: &[],
        metadata: &metadata,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("v3 bake of pool-head MLP")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_stats_uniform_hidden() {
        // All ones → mean=1, std=0 (floored), max=1, p_6=1.
        let h = vec![1.0f64; 16];
        let (s, idx) = pool_stats(&h);
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - POOL_STD_FLOOR).abs() < 1e-12);
        assert!((s[2] - 1.0).abs() < 1e-12);
        assert!((s[3] - 1.0).abs() < 1e-12);
        assert!(idx < 16);
    }

    #[test]
    fn pool_stats_known_vector() {
        // h = [0, 1, 2, 3] → mean=1.5, var=(2.25+0.25+0.25+2.25)/4=1.25,
        // std=sqrt(1.25)≈1.1180, max=3, p_6 = (mean(|h|^6))^(1/6)
        let h = vec![0.0, 1.0, 2.0, 3.0];
        let ([mu, sd, mx, p6], _) = pool_stats(&h);
        assert!((mu - 1.5).abs() < 1e-12, "mean = {mu}");
        assert!((sd - 1.25f64.sqrt()).abs() < 1e-9, "std = {sd}");
        assert!((mx - 3.0).abs() < 1e-12);
        let mean_p = (0f64.powf(6.0) + 1.0 + 64.0 + 729.0) / 4.0;
        let p_expect = mean_p.powf(1.0 / 6.0);
        assert!(
            (p6 - p_expect).abs() < 1e-9,
            "p_6 got {p6}, expect {p_expect}"
        );
    }

    #[test]
    fn forward_pool_head_zero_weights_is_bias_path() {
        // Zero w1, zero b1 → h_pre = 0 vector → h = 0 vector.
        // Pool: μ=0, σ=POOL_STD_FLOOR (floored), max=0, p_6=0.
        // Reducer [0.5, 1.0, 0.5, 0.5] · [0, 0.0026, 0, 0] + 0.0 = 0.0026.
        let x = vec![1.0; 4];
        let w1 = vec![0.0; 4 * 8];
        let b1 = vec![0.0; 8];
        let rw: [f64; 4] = [0.5, 1.0, 0.5, 0.5];
        let rb = 0.0;
        let (y, _, _, stats, _) = forward_pool_head(&x, &w1, &b1, &rw, rb, 4, 8, 0.01);
        assert!((stats[0]).abs() < 1e-12);
        assert!((stats[1] - POOL_STD_FLOOR).abs() < 1e-12);
        assert!((stats[2]).abs() < 1e-12);
        assert!((stats[3]).abs() < 1e-12);
        assert!((y - POOL_STD_FLOOR).abs() < 1e-12);
    }

    #[test]
    fn backprop_pool_head_finite_diff_w1() {
        // Numerical gradient check on a tiny config (n_features=2, n_hidden=3).
        let n_f = 2;
        let n_h = 3;
        let alpha = 0.01;
        let x = vec![0.7f64, -0.3];
        let w1: Vec<f64> = vec![0.1, -0.2, 0.3, -0.4, 0.5, 0.6];
        let b1 = vec![0.05, -0.05, 0.1];
        let rw: [f64; 4] = [0.2, 0.8, 0.3, 0.1];
        let rb = 0.05;

        let (y, hp, h, stats, max_idx) = forward_pool_head(&x, &w1, &b1, &rw, rb, n_f, n_h, alpha);
        let dl_dy = 2.0 * (y - 1.0); // ½(y − 1)² loss → ∂L/∂y = (y − 1) · 2

        let mut gw1 = vec![0.0; w1.len()];
        let mut gb1 = vec![0.0; b1.len()];
        let mut g_rw: [f64; 4] = [0.0; 4];
        let mut g_rb: f64 = 0.0;
        backprop_step_pool_head(
            &x, &hp, &h, &stats, max_idx, dl_dy, &rw, &mut gw1, &mut gb1, &mut g_rw, &mut g_rb,
            n_f, n_h, alpha,
        );

        let eps = 1e-5;
        // Check ∂L/∂w1[0].
        let mut w1_p = w1.clone();
        w1_p[0] += eps;
        let (y_p, _, _, _, _) = forward_pool_head(&x, &w1_p, &b1, &rw, rb, n_f, n_h, alpha);
        let mut w1_m = w1.clone();
        w1_m[0] -= eps;
        let (y_m, _, _, _, _) = forward_pool_head(&x, &w1_m, &b1, &rw, rb, n_f, n_h, alpha);
        let num_grad = ((y_p - 1.0).powi(2) - (y_m - 1.0).powi(2)) / (2.0 * eps);
        assert!(
            (gw1[0] - num_grad).abs() < 1e-3,
            "analytic gw1[0]={} vs numerical={} (diff {})",
            gw1[0],
            num_grad,
            (gw1[0] - num_grad).abs()
        );

        // Check ∂L/∂reducer_w[1] (sigma weight).
        let mut rw_p = rw;
        rw_p[1] += eps;
        let (yp, _, _, _, _) = forward_pool_head(&x, &w1, &b1, &rw_p, rb, n_f, n_h, alpha);
        let mut rw_m = rw;
        rw_m[1] -= eps;
        let (ym, _, _, _, _) = forward_pool_head(&x, &w1, &b1, &rw_m, rb, n_f, n_h, alpha);
        let num_g_rw1 = ((yp - 1.0).powi(2) - (ym - 1.0).powi(2)) / (2.0 * eps);
        assert!(
            (g_rw[1] - num_g_rw1).abs() < 1e-4,
            "analytic g_rw[1]={} vs numerical={}",
            g_rw[1],
            num_g_rw1
        );
    }

    #[test]
    fn bake_pool_head_v3_has_metadata_and_version() {
        let model = PoolHeadModel::new(8, 4, 7);
        let bytes = bake_pool_head_v3(&model);
        assert_eq!(&bytes[0..4], b"ZNPR", "expected ZNPR magic");
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 3, "expected v3 (v2 prohibited)");
        // The metadata key must be present in the byte stream.
        let needle = b"zentrain.pool_head_reducer";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(found, "expected pool_head_reducer metadata entry in bake");
    }

    #[test]
    fn train_pool_head_recovers_synthetic_ranking() {
        // Synthetic: 20 pairs where higher score ↔ larger first feature.
        let n_f = 4;
        let n_pairs = 40;
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut scores: Vec<f64> = Vec::new();
        let mut rng = SplitMix64::new(123);
        for i in 0..n_pairs {
            let t = i as f64 / (n_pairs - 1) as f64;
            let f0 = t;
            let f1 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let f2 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let f3 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            rows.push(vec![f0, f1, f2, f3]);
            scores.push(t * 100.0); // monotonic in f0
        }
        let row_refs: Vec<&[f64]> = rows.iter().map(|v| v.as_slice()).collect();
        let g = TrainingGroup {
            name: "synth".into(),
            human_scores: &scores,
            features: &row_refs,
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 0.0,
        };
        let hp = PoolHeadHparams {
            n_hidden: 16,
            n_epochs: 5,
            pairs_per_epoch: 200,
            initial_lr: 5e-3,
            l2_lambda: 0.0,
            ..Default::default()
        };
        let model = train_pool_head(&[g], &hp, n_f);
        // Score every row through the trained model; check Spearman ≥ 0.85.
        let std_rows: Vec<Vec<f64>> = rows
            .iter()
            .map(|r| {
                (0..n_f)
                    .map(|d| (r[d] - model.scaler_mean[d]) / model.scaler_scale[d])
                    .collect()
            })
            .collect();
        let preds: Vec<f64> = std_rows
            .iter()
            .map(|r| {
                let (y, _, _, _, _) = forward_pool_head(
                    r,
                    &model.w1,
                    &model.b1,
                    &model.reducer_w,
                    model.reducer_b,
                    n_f,
                    model.n_hidden,
                    hp.leaky_alpha,
                );
                y
            })
            .collect();
        let s = crate::spearman(&preds, &scores);
        assert!(
            s.abs() > 0.85,
            "train_pool_head: Spearman {s} < 0.85 (synthetic ranking should be easy)"
        );
    }
}
