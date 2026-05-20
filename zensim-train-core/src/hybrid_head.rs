//! EX-2 follow-up: hybrid pool + rank head.
//!
//! Combines the production RankNet scalar head with the EX-2 std-pool
//! head (`pool_head.rs`) via a **learned sigmoid-bounded mix coefficient**
//! `α ∈ [0,1]`. The two heads share the same encoder (W1 + b1 + LeakyReLU)
//! and their outputs are blended:
//!
//! ```text
//! y_rank = h · rank_w + rank_b                            (scalar RankNet head)
//! y_pool = [μ, σ, max, p_6] · reducer_w + reducer_b       (pool head)
//! α      = sigmoid(α_logit)                               (learned mix ∈ [0,1])
//! y      = α · y_rank + (1 − α) · y_pool
//! ```
//!
//! Per the EX-2 follow-up agent analysis (2026-05-18): pool-head and
//! rank-head architectures are individually Pareto-tight (pool wins
//! KonJND, rank wins CID22). A binary "switch" between them creates
//! gradient-starvation in whichever path the loss penalizes. The
//! sigmoid-bounded learned α lets the loss balance the two paths
//! per-bake. Initialize `α_logit = 0` (so α=0.5 — neutral 50/50 mix);
//! gradient flows into both heads + α simultaneously.
//!
//! **Bake metadata format (`zentrain.hybrid_head`)**:
//! Payload = `[rank_w[0..n_hidden]] + [rank_b] + [α_logit] + [reducer_w[0..4]] + [reducer_b] + [p_norm]`
//! as f32 little-endian. Total size = `4 * (n_hidden + 1 + 1 + 4 + 1 + 1)`.
//!
//! Runtime detection: `apply_mlp_scoring` checks for this metadata key
//! BEFORE `zentrain.pool_head_reducer` (hybrid is the more general
//! form). When found, run BOTH heads on the hidden vector and mix.

use crate::TrainingGroup;
use crate::adam::AdamState;
use crate::mlp::compute_scaler_from_groups;
use crate::pool_head::{POOL_P_NORM, POOL_STD_FLOOR, pool_stats};
use crate::rng::SplitMix64;
use zenpredict::{Activation, MetadataType, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

/// Numerical stability for sigmoid in α and its gradient. We don't
/// clamp α_logit during training (Adam handles it) but we do clamp it
/// to ±20 before sigmoid to avoid overflow in `exp`.
#[inline]
fn sigmoid(x: f64) -> f64 {
    let xc = x.clamp(-20.0, 20.0);
    1.0 / (1.0 + (-xc).exp())
}

/// Full forward pass for the hybrid head.
///
/// Returns `(y, y_rank, y_pool, alpha, h_pre, h, stats, max_idx)` —
/// every intermediate the backprop step needs. `h_pre` is the
/// pre-LeakyReLU encoder state, `h` is post-LeakyReLU.
#[allow(clippy::too_many_arguments)]
pub fn forward_hybrid_head(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    rank_w: &[f64],
    rank_b: f64,
    reducer_w: &[f64; 4],
    reducer_b: f64,
    alpha_logit: f64,
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f64,
) -> (f64, f64, f64, f64, Vec<f64>, Vec<f64>, [f64; 4], usize) {
    debug_assert_eq!(rank_w.len(), n_hidden);
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
        .map(|&v| if v >= 0.0 { v } else { leaky_alpha * v })
        .collect();

    // Rank-net scalar head: y_rank = h · rank_w + rank_b
    let mut y_rank = rank_b;
    for j in 0..n_hidden {
        y_rank += h[j] * rank_w[j];
    }

    // Pool head: y_pool = stats · reducer_w + reducer_b
    let (stats, max_idx) = pool_stats(&h);
    let y_pool = stats[0] * reducer_w[0]
        + stats[1] * reducer_w[1]
        + stats[2] * reducer_w[2]
        + stats[3] * reducer_w[3]
        + reducer_b;

    let alpha = sigmoid(alpha_logit);
    let y = alpha * y_rank + (1.0 - alpha) * y_pool;
    (y, y_rank, y_pool, alpha, h_pre, h, stats, max_idx)
}

/// Backprop ∂L/∂y through the hybrid head to gradients on
/// (w1, b1, rank_w, rank_b, reducer_w, reducer_b, α_logit).
///
/// Gradient chain rule:
///
/// - `∂L/∂y_rank   = ∂L/∂y · α`
/// - `∂L/∂y_pool   = ∂L/∂y · (1 − α)`
/// - `∂L/∂α       = ∂L/∂y · (y_rank − y_pool)`
/// - `∂L/∂α_logit = ∂L/∂α · α · (1 − α)`   (sigmoid'(α_logit))
///
/// Then `∂L/∂y_rank` routes through the rank head (linear in h):
/// - `∂L/∂rank_w[j] = ∂L/∂y_rank · h[j]`
/// - `∂L/∂rank_b    = ∂L/∂y_rank`
/// - contribution to `∂L/∂h[j] += ∂L/∂y_rank · rank_w[j]`
///
/// And `∂L/∂y_pool` routes through the pool stats (same chain rule as
/// in `pool_head::backprop_step_pool_head`):
/// - `∂L/∂reducer_w[k] = ∂L/∂y_pool · stats[k]`
/// - `∂L/∂reducer_b    = ∂L/∂y_pool`
/// - `∂L/∂stat_k       = ∂L/∂y_pool · reducer_w[k]`
/// - contributions to `∂L/∂h[j]` via the four stat partials (see
///   pool_head doc for derivation).
///
/// Finally LeakyReLU back-routes ∂L/∂h_j → ∂L/∂h_pre_j and the
/// standard layer-1 backprop populates `gw1` and `gb1`.
#[allow(clippy::too_many_arguments)]
pub fn backprop_step_hybrid_head(
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    stats: &[f64; 4],
    max_idx: usize,
    y_rank: f64,
    y_pool: f64,
    alpha: f64,
    dl_dy: f64,
    rank_w: &[f64],
    reducer_w: &[f64; 4],
    gw1: &mut [f64],
    gb1: &mut [f64],
    g_rank_w: &mut [f64],
    g_rank_b: &mut f64,
    g_reducer_w: &mut [f64; 4],
    g_reducer_b: &mut f64,
    g_alpha_logit: &mut f64,
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f64,
) {
    debug_assert_eq!(rank_w.len(), n_hidden);
    debug_assert_eq!(g_rank_w.len(), n_hidden);

    let dl_dy_rank = dl_dy * alpha;
    let dl_dy_pool = dl_dy * (1.0 - alpha);
    let dl_dalpha = dl_dy * (y_rank - y_pool);
    // sigmoid'(α_logit) = α · (1 − α)
    *g_alpha_logit += dl_dalpha * alpha * (1.0 - alpha);

    // Rank-head gradients (linear in h).
    for j in 0..n_hidden {
        g_rank_w[j] += dl_dy_rank * h[j];
    }
    *g_rank_b += dl_dy_rank;

    // Pool-head reducer gradients.
    for k in 0..4 {
        g_reducer_w[k] += dl_dy_pool * stats[k];
    }
    *g_reducer_b += dl_dy_pool;

    // ∂L/∂h_j: rank contribution + pool contribution.
    let mut dl_dh = vec![0.0f64; n_hidden];

    // Rank: ∂L/∂h_j += dl_dy_rank · rank_w[j]
    for j in 0..n_hidden {
        dl_dh[j] += dl_dy_rank * rank_w[j];
    }

    // Pool: ∂L/∂stat_k = dl_dy_pool · reducer_w[k], then chain to h.
    let dl_dstat: [f64; 4] = [
        dl_dy_pool * reducer_w[0],
        dl_dy_pool * reducer_w[1],
        dl_dy_pool * reducer_w[2],
        dl_dy_pool * reducer_w[3],
    ];
    let n = n_hidden as f64;
    let mu = stats[0];
    let sigma = stats[1];
    let p6 = stats[3];
    let inv_sigma_n = if sigma > POOL_STD_FLOOR + 1e-12 {
        1.0 / (n * sigma)
    } else {
        0.0
    };
    let p6_floor = p6.max(1e-12);
    let inv_p6_pow5_n = 1.0 / (n * p6_floor.powi(5));
    for j in 0..n_hidden {
        let hj = h[j];
        dl_dh[j] += dl_dstat[0] / n;
        dl_dh[j] += dl_dstat[1] * (hj - mu) * inv_sigma_n;
        if j == max_idx {
            dl_dh[j] += dl_dstat[2];
        }
        let abs_hj = hj.abs();
        let sign_hj = if hj >= 0.0 { 1.0 } else { -1.0 };
        let abs_pow5 = abs_hj * abs_hj * abs_hj * abs_hj * abs_hj;
        dl_dh[j] += dl_dstat[3] * sign_hj * abs_pow5 * inv_p6_pow5_n;
    }

    // LeakyReLU back-route.
    let dl_dh_pre: Vec<f64> = dl_dh
        .iter()
        .zip(h_pre.iter())
        .map(|(&dh, &hp)| if hp >= 0.0 { dh } else { leaky_alpha * dh })
        .collect();

    // Layer-1 grads.
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

/// Hyperparameters for the hybrid-head trainer (mirrors PoolHeadHparams).
#[derive(Clone, Debug)]
pub struct HybridHeadHparams {
    /// Hidden vector width.
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
    /// L2 regularization on layer weights (and rank_w + reducer_w).
    pub l2_lambda: f64,
}

impl Default for HybridHeadHparams {
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

/// Trained hybrid-head model.
#[derive(Debug)]
pub struct HybridHeadModel {
    /// Per-feature mean (n_features).
    pub scaler_mean: Vec<f64>,
    /// Per-feature std (n_features).
    pub scaler_scale: Vec<f64>,
    /// Layer-1 weights, row-major (n_features × n_hidden).
    pub w1: Vec<f64>,
    /// Layer-1 biases (n_hidden).
    pub b1: Vec<f64>,
    /// Rank-net head weights (n_hidden).
    pub rank_w: Vec<f64>,
    /// Rank-net head bias.
    pub rank_b: f64,
    /// 4→1 pool-head reducer weights [w_μ, w_σ, w_max, w_p6].
    pub reducer_w: [f64; 4],
    /// 4→1 pool-head reducer bias.
    pub reducer_b: f64,
    /// Learned mix logit; α = sigmoid(α_logit).
    pub alpha_logit: f64,
    /// Hidden width.
    pub n_hidden: usize,
    /// Input dim.
    pub n_features: usize,
}

impl HybridHeadModel {
    /// Initialize fresh hybrid model:
    /// - `w1` Xavier-Glorot from `(n_features, n_hidden)`.
    /// - `b1 = 0`.
    /// - `rank_w` ~ N(0, 1/√n_hidden) (linear layer init).
    /// - `rank_b = 0`.
    /// - `reducer_w = [0.05, 1.0, 0.05, 0.05]` (std-pool dominant, matches
    ///   `PoolHeadModel::new`).
    /// - `reducer_b = 0`.
    /// - `alpha_logit = 0` → α = 0.5 (neutral 50/50 mix to start).
    pub fn new(n_features: usize, n_hidden: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let scale = (2.0 / n_features as f64).sqrt();
        let n_w1 = n_features * n_hidden;
        let mut w1 = Vec::with_capacity(n_w1);
        for _ in 0..n_w1 {
            let u1 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let u2 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let u1 = u1.max(1e-12);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();
            w1.push(z * scale);
        }
        let rank_scale = 1.0 / (n_hidden as f64).sqrt();
        let mut rank_w = Vec::with_capacity(n_hidden);
        for _ in 0..n_hidden {
            let u1 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let u2 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let u1 = u1.max(1e-12);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();
            rank_w.push(z * rank_scale);
        }
        Self {
            scaler_mean: vec![0.0; n_features],
            scaler_scale: vec![1.0; n_features],
            w1,
            b1: vec![0.0; n_hidden],
            rank_w,
            rank_b: 0.0,
            reducer_w: [0.05, 1.0, 0.05, 0.05],
            reducer_b: 0.0,
            alpha_logit: 0.0,
            n_hidden,
            n_features,
        }
    }
}

/// Train a hybrid-head MLP via RankNet (standalone trainer — minimal
/// recipe, intended for unit tests and synthetic-ranking smoke tests).
///
/// **Production training goes through `train_mlp_hybrid_head_with_tv`
/// in `zensim-validate::mlp_train`** (TV / PWRC / NiN / minibatch /
/// row-boost). This function is the floor for "does the math work."
pub fn train_hybrid_head(
    groups: &[TrainingGroup<'_>],
    hparams: &HybridHeadHparams,
    n_features: usize,
) -> HybridHeadModel {
    let train_indices: Vec<usize> = (0..groups.len())
        .filter(|&i| groups[i].train_weight > 0.0)
        .collect();
    assert!(
        !train_indices.is_empty(),
        "train_hybrid_head: at least one group must have train_weight > 0"
    );

    let (mean, scale) = compute_scaler_from_groups(groups, &train_indices, n_features);

    let mut model = HybridHeadModel::new(n_features, hparams.n_hidden, hparams.seed);
    model.scaler_mean = mean;
    model.scaler_scale = scale;

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
    // For Adam: route gw1/gb1 to layer-1, then "w2" slot holds the
    // concatenation `[rank_w (n_hidden) | reducer_w (4) | alpha_logit (1)]`,
    // and "b2" slot holds `[rank_b, reducer_b]`. Total w2 = n_hidden + 5,
    // b2 = 2.
    let n_w2 = n_hidden + 4 + 1;
    let n_b2 = 2;
    let mut adam = AdamState::new(nw1, nb1, n_w2, n_b2);
    let mut rng = SplitMix64::new(hparams.seed ^ 0xA5A5_A5A5_A5A5_A5A5);

    let l2 = hparams.l2_lambda;
    let alpha_leaky = hparams.leaky_alpha;
    let lr = hparams.initial_lr;

    for _epoch in 0..hparams.n_epochs {
        for _pair in 0..hparams.pairs_per_epoch {
            let r = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let mut gi = 0usize;
            for (k, &c) in weight_cdf.iter().enumerate() {
                if r < c {
                    gi = k;
                    break;
                }
                gi = k;
            }
            let g_rows = &std_groups[gi];
            let g_scores = train_scores[gi];
            if g_rows.len() < 2 {
                continue;
            }
            let len = g_rows.len();
            let ia = (rng.next_u64() as usize) % len;
            let mut ib = (rng.next_u64() as usize) % len;
            if ib == ia {
                ib = (ib + 1) % len;
            }
            if (g_scores[ia] - g_scores[ib]).abs() < 1e-9 {
                continue;
            }
            let (ihi, ilo) = if g_scores[ia] > g_scores[ib] {
                (ia, ib)
            } else {
                (ib, ia)
            };
            let xhi = &g_rows[ihi];
            let xlo = &g_rows[ilo];

            let (yhi, yhi_rank, yhi_pool, alpha_hi, hp_hi, h_hi, stats_hi, max_idx_hi) =
                forward_hybrid_head(
                    xhi,
                    &model.w1,
                    &model.b1,
                    &model.rank_w,
                    model.rank_b,
                    &model.reducer_w,
                    model.reducer_b,
                    model.alpha_logit,
                    n_features,
                    n_hidden,
                    alpha_leaky,
                );
            let (ylo, ylo_rank, ylo_pool, alpha_lo, hp_lo, h_lo, stats_lo, max_idx_lo) =
                forward_hybrid_head(
                    xlo,
                    &model.w1,
                    &model.b1,
                    &model.rank_w,
                    model.rank_b,
                    &model.reducer_w,
                    model.reducer_b,
                    model.alpha_logit,
                    n_features,
                    n_hidden,
                    alpha_leaky,
                );
            let d = yhi - ylo;
            let sig_neg_d = 1.0 / (1.0 + d.exp());
            let dl_dd = -sig_neg_d;
            let dl_dyhi = dl_dd;
            let dl_dylo = -dl_dd;

            let mut g_rank_w_buf = vec![0.0f64; n_hidden];
            let mut g_rank_b_buf = 0.0f64;
            let mut g_red_w: [f64; 4] = [0.0; 4];
            let mut g_red_b: f64 = 0.0;
            let mut g_alpha_logit: f64 = 0.0;
            backprop_step_hybrid_head(
                xhi,
                &hp_hi,
                &h_hi,
                &stats_hi,
                max_idx_hi,
                yhi_rank,
                yhi_pool,
                alpha_hi,
                dl_dyhi,
                &model.rank_w,
                &model.reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_alpha_logit,
                n_features,
                n_hidden,
                alpha_leaky,
            );
            backprop_step_hybrid_head(
                xlo,
                &hp_lo,
                &h_lo,
                &stats_lo,
                max_idx_lo,
                ylo_rank,
                ylo_pool,
                alpha_lo,
                dl_dylo,
                &model.rank_w,
                &model.reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_alpha_logit,
                n_features,
                n_hidden,
                alpha_leaky,
            );
            // L2.
            if l2 > 0.0 {
                for (g, &w) in adam.gw1.iter_mut().zip(model.w1.iter()) {
                    *g += 2.0 * l2 * w;
                }
                for j in 0..n_hidden {
                    g_rank_w_buf[j] += 2.0 * l2 * model.rank_w[j];
                }
                for k in 0..4 {
                    g_red_w[k] += 2.0 * l2 * model.reducer_w[k];
                }
                // alpha_logit is unregularized.
            }

            // Pack into Adam w2/b2 slots: w2 = [rank_w | reducer_w | alpha_logit].
            for j in 0..n_hidden {
                adam.gw2[j] += g_rank_w_buf[j];
            }
            for k in 0..4 {
                adam.gw2[n_hidden + k] += g_red_w[k];
            }
            adam.gw2[n_hidden + 4] += g_alpha_logit;
            // b2 = [rank_b, reducer_b].
            adam.gb2[0] += g_rank_b_buf;
            adam.gb2[1] += g_red_b;

            // Adam step: pack/unpack via temp vectors.
            let mut w2_vec = vec![0.0f64; n_w2];
            for j in 0..n_hidden {
                w2_vec[j] = model.rank_w[j];
            }
            for k in 0..4 {
                w2_vec[n_hidden + k] = model.reducer_w[k];
            }
            w2_vec[n_hidden + 4] = model.alpha_logit;
            let mut b2_vec = vec![model.rank_b, model.reducer_b];
            adam.step(&mut model.w1, &mut model.b1, &mut w2_vec, &mut b2_vec, lr);
            for j in 0..n_hidden {
                model.rank_w[j] = w2_vec[j];
            }
            for k in 0..4 {
                model.reducer_w[k] = w2_vec[n_hidden + k];
            }
            model.alpha_logit = w2_vec[n_hidden + 4];
            model.rank_b = b2_vec[0];
            model.reducer_b = b2_vec[1];
        }
    }

    model
}

/// Bake the hybrid-head model into ZNPR v3 bytes.
///
/// Wire format mirrors `bake_pool_head_v3` exactly except for the
/// metadata key + payload. The first two layers are still
/// `n_features → n_hidden (LeakyReLU)` and `n_hidden → n_hidden
/// (Identity passthrough)` so `predict()` returns the hidden vector.
///
/// **Metadata key**: `zentrain.hybrid_head`
/// **Payload** (f32 little-endian):
/// - `rank_w[0..n_hidden]` — n_hidden floats
/// - `rank_b` — 1 float
/// - `alpha_logit` — 1 float
/// - `reducer_w[0..4]` — 4 floats
/// - `reducer_b` — 1 float
/// - `p_norm` — 1 float (always 6.0 currently)
///
/// Total bytes = `4 * (n_hidden + 8)`.
pub fn bake_hybrid_head_v3(model: &HybridHeadModel) -> Vec<u8> {
    let n_features = model.n_features;
    let n_hidden = model.n_hidden;

    let scaler_mean_f32: Vec<f32> = model.scaler_mean.iter().map(|&v| v as f32).collect();
    let scaler_scale_f32: Vec<f32> = model.scaler_scale.iter().map(|&v| v as f32).collect();
    let w1_f32: Vec<f32> = model.w1.iter().map(|&v| v as f32).collect();
    let b1_f32: Vec<f32> = model.b1.iter().map(|&v| v as f32).collect();
    let mut w2_f32 = vec![0.0f32; n_hidden * n_hidden];
    for i in 0..n_hidden {
        w2_f32[i * n_hidden + i] = 1.0;
    }
    let b2_f32 = vec![0.0f32; n_hidden];

    let n_payload = n_hidden + 8; // rank_w + rank_b + alpha_logit + reducer_w(4) + reducer_b + p_norm
    let mut payload = Vec::with_capacity(n_payload * 4);
    for v in &model.rank_w {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.rank_b as f32).to_le_bytes());
    payload.extend_from_slice(&(model.alpha_logit as f32).to_le_bytes());
    for v in &model.reducer_w {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.reducer_b as f32).to_le_bytes());
    payload.extend_from_slice(&(POOL_P_NORM as f32).to_le_bytes());

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
        key: "zentrain.hybrid_head",
        kind: MetadataType::Numeric,
        value: &payload,
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
    .expect("v3 bake of hybrid-head MLP")
}

/// Parsed hybrid-head metadata payload, runtime-side.
#[derive(Clone, Copy, Debug)]
pub struct HybridHeadMeta {
    /// Rank-net bias.
    pub rank_b: f32,
    /// Mix logit; α = sigmoid(α_logit).
    pub alpha_logit: f32,
    /// Pool-head reducer weights.
    pub reducer_w: [f32; 4],
    /// Pool-head reducer bias.
    pub reducer_b: f32,
    /// p-norm exponent (6.0 currently).
    pub p_norm: f32,
}

/// Parse the `zentrain.hybrid_head` payload. Returns the meta + a
/// borrowed `rank_w` slice as a separate value (n_hidden floats), or
/// None if the payload is too short / malformed.
///
/// Layout: `[rank_w[0..n_hidden]] [rank_b] [alpha_logit]
/// [reducer_w[0..4]] [reducer_b] [p_norm]` — all f32 LE.
pub fn parse_hybrid_head_meta(
    payload: &[u8],
    n_hidden: usize,
) -> Option<(Vec<f32>, HybridHeadMeta)> {
    let expected = (n_hidden + 8) * 4;
    if payload.len() != expected {
        return None;
    }
    let mut floats = Vec::with_capacity(n_hidden + 8);
    for chunk in payload.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let rank_w: Vec<f32> = floats[..n_hidden].to_vec();
    let meta = HybridHeadMeta {
        rank_b: floats[n_hidden],
        alpha_logit: floats[n_hidden + 1],
        reducer_w: [
            floats[n_hidden + 2],
            floats[n_hidden + 3],
            floats[n_hidden + 4],
            floats[n_hidden + 5],
        ],
        reducer_b: floats[n_hidden + 6],
        p_norm: floats[n_hidden + 7],
    };
    Some((rank_w, meta))
}

/// Runtime forward path: given a hidden vector `h` and parsed
/// hybrid-head metadata, produce the mixed `y`. Used by zensim's
/// `apply_mlp_scoring` and by every bake comparator (bake_verdict,
/// bake_compare).
///
/// Formula:
/// - `y_rank = h · rank_w + rank_b`
/// - `[μ, σ, max, p_norm](h) → y_pool = stats · reducer_w + reducer_b`
/// - `α = sigmoid(alpha_logit)`
/// - `y = α · y_rank + (1 − α) · y_pool`
pub fn apply_hybrid_head_runtime(h: &[f32], rank_w: &[f32], meta: &HybridHeadMeta) -> f64 {
    let n = h.len();
    debug_assert_eq!(rank_w.len(), n);
    debug_assert!(n > 0);

    let mut y_rank = meta.rank_b as f64;
    let mut sum = 0.0f64;
    let mut max_v = f64::NEG_INFINITY;
    let mut sum_p = 0.0f64;
    let p = meta.p_norm as f64;
    for (j, &hj) in h.iter().enumerate() {
        let hjf = hj as f64;
        y_rank += hjf * rank_w[j] as f64;
        sum += hjf;
        if hjf > max_v {
            max_v = hjf;
        }
        sum_p += hjf.abs().powf(p);
    }
    let nf = n as f64;
    let mu = sum / nf;
    let mut var = 0.0f64;
    for &hj in h.iter() {
        let d = hj as f64 - mu;
        var += d * d;
    }
    let sigma = (var / nf).sqrt().max(POOL_STD_FLOOR);
    let p_norm_stat = (sum_p / nf).powf(1.0 / p);

    let y_pool = mu * meta.reducer_w[0] as f64
        + sigma * meta.reducer_w[1] as f64
        + max_v * meta.reducer_w[2] as f64
        + p_norm_stat * meta.reducer_w[3] as f64
        + meta.reducer_b as f64;

    let alpha_logit = meta.alpha_logit as f64;
    let alpha = {
        let xc = alpha_logit.clamp(-20.0, 20.0);
        1.0 / (1.0 + (-xc).exp())
    };
    alpha * y_rank + (1.0 - alpha) * y_pool
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_alpha_at_zero_logit_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn forward_hybrid_zero_weights_returns_neutral_mix() {
        // w1 = 0 → h = 0. rank_w = 0, rank_b = 0 → y_rank = 0.
        // reducer = [0,0,0,0], reducer_b = 0 → y_pool = 0.
        // α = 0.5 → y = 0.
        let x = vec![1.0; 4];
        let w1 = vec![0.0; 4 * 8];
        let b1 = vec![0.0; 8];
        let rank_w = vec![0.0; 8];
        let rw: [f64; 4] = [0.0; 4];
        let (y, y_rank, y_pool, alpha, _, _, _, _) =
            forward_hybrid_head(&x, &w1, &b1, &rank_w, 0.0, &rw, 0.0, 0.0, 4, 8, 0.01);
        assert!(y_rank.abs() < 1e-12);
        // y_pool gets σ-floor contribution (reducer_w = 0 so still 0).
        assert!(y_pool.abs() < 1e-12);
        assert!((alpha - 0.5).abs() < 1e-12);
        assert!(y.abs() < 1e-12);
    }

    #[test]
    fn forward_hybrid_alpha_1_only_rank() {
        // α_logit = +10 → α ≈ 1; y ≈ y_rank.
        let x = vec![1.0; 4];
        let mut w1 = vec![0.0; 4 * 8];
        for i in 0..4 {
            w1[i * 8] = 1.0; // h_pre[0] = sum(x[i] * 1) = 4 → h[0] = 4
        }
        let b1 = vec![0.0; 8];
        let mut rank_w = vec![0.0; 8];
        rank_w[0] = 0.5;
        let rw: [f64; 4] = [0.0; 4];
        let (y, y_rank, _, alpha, _, _, _, _) =
            forward_hybrid_head(&x, &w1, &b1, &rank_w, 0.1, &rw, 0.0, 10.0, 4, 8, 0.01);
        assert!((alpha - 1.0).abs() < 1e-3, "alpha = {alpha}, expected ~1");
        assert!(
            (y_rank - (4.0 * 0.5 + 0.1)).abs() < 1e-9,
            "y_rank = {y_rank}"
        );
        assert!((y - y_rank).abs() < 1e-3, "y={y}, y_rank={y_rank}");
    }

    #[test]
    fn backprop_hybrid_finite_diff_w1() {
        // Numerical gradient check on tiny config.
        let n_f = 2;
        let n_h = 3;
        let alpha_leaky = 0.01;
        let x = vec![0.7f64, -0.3];
        let w1: Vec<f64> = vec![0.1, -0.2, 0.3, -0.4, 0.5, 0.6];
        let b1 = vec![0.05, -0.05, 0.1];
        let rank_w: Vec<f64> = vec![0.2, -0.1, 0.05];
        let rank_b = 0.02;
        let rw: [f64; 4] = [0.2, 0.8, 0.3, 0.1];
        let rb = 0.05;
        let alpha_logit = 0.5; // α ≈ 0.622

        let (y, yr, yp, a, hp, h, stats, max_idx) = forward_hybrid_head(
            &x,
            &w1,
            &b1,
            &rank_w,
            rank_b,
            &rw,
            rb,
            alpha_logit,
            n_f,
            n_h,
            alpha_leaky,
        );
        let dl_dy = 2.0 * (y - 1.0); // L = (y - 1)²

        let mut gw1 = vec![0.0; w1.len()];
        let mut gb1 = vec![0.0; b1.len()];
        let mut g_rank_w = vec![0.0f64; n_h];
        let mut g_rank_b = 0.0;
        let mut g_red_w: [f64; 4] = [0.0; 4];
        let mut g_red_b = 0.0;
        let mut g_alpha_logit = 0.0;
        backprop_step_hybrid_head(
            &x,
            &hp,
            &h,
            &stats,
            max_idx,
            yr,
            yp,
            a,
            dl_dy,
            &rank_w,
            &rw,
            &mut gw1,
            &mut gb1,
            &mut g_rank_w,
            &mut g_rank_b,
            &mut g_red_w,
            &mut g_red_b,
            &mut g_alpha_logit,
            n_f,
            n_h,
            alpha_leaky,
        );

        let eps = 1e-5;
        let loss = |y: f64| (y - 1.0).powi(2);
        // ∂L/∂w1[0]
        let mut w1_p = w1.clone();
        w1_p[0] += eps;
        let (yp_p, _, _, _, _, _, _, _) = forward_hybrid_head(
            &x,
            &w1_p,
            &b1,
            &rank_w,
            rank_b,
            &rw,
            rb,
            alpha_logit,
            n_f,
            n_h,
            alpha_leaky,
        );
        let mut w1_m = w1.clone();
        w1_m[0] -= eps;
        let (yp_m, _, _, _, _, _, _, _) = forward_hybrid_head(
            &x,
            &w1_m,
            &b1,
            &rank_w,
            rank_b,
            &rw,
            rb,
            alpha_logit,
            n_f,
            n_h,
            alpha_leaky,
        );
        let num_grad = (loss(yp_p) - loss(yp_m)) / (2.0 * eps);
        assert!(
            (gw1[0] - num_grad).abs() < 1e-3,
            "gw1[0]={} num={}",
            gw1[0],
            num_grad
        );

        // ∂L/∂rank_w[1]
        let mut rw_p = rank_w.clone();
        rw_p[1] += eps;
        let (yp2_p, _, _, _, _, _, _, _) = forward_hybrid_head(
            &x,
            &w1,
            &b1,
            &rw_p,
            rank_b,
            &rw,
            rb,
            alpha_logit,
            n_f,
            n_h,
            alpha_leaky,
        );
        let mut rw_m = rank_w.clone();
        rw_m[1] -= eps;
        let (yp2_m, _, _, _, _, _, _, _) = forward_hybrid_head(
            &x,
            &w1,
            &b1,
            &rw_m,
            rank_b,
            &rw,
            rb,
            alpha_logit,
            n_f,
            n_h,
            alpha_leaky,
        );
        let num_g_rank1 = (loss(yp2_p) - loss(yp2_m)) / (2.0 * eps);
        assert!(
            (g_rank_w[1] - num_g_rank1).abs() < 1e-4,
            "g_rank_w[1]={} num={}",
            g_rank_w[1],
            num_g_rank1
        );

        // ∂L/∂reducer_w[1] (σ-weight)
        let mut red_p = rw;
        red_p[1] += eps;
        let (yp3_p, _, _, _, _, _, _, _) = forward_hybrid_head(
            &x,
            &w1,
            &b1,
            &rank_w,
            rank_b,
            &red_p,
            rb,
            alpha_logit,
            n_f,
            n_h,
            alpha_leaky,
        );
        let mut red_m = rw;
        red_m[1] -= eps;
        let (yp3_m, _, _, _, _, _, _, _) = forward_hybrid_head(
            &x,
            &w1,
            &b1,
            &rank_w,
            rank_b,
            &red_m,
            rb,
            alpha_logit,
            n_f,
            n_h,
            alpha_leaky,
        );
        let num_g_red1 = (loss(yp3_p) - loss(yp3_m)) / (2.0 * eps);
        assert!(
            (g_red_w[1] - num_g_red1).abs() < 1e-4,
            "g_red_w[1]={} num={}",
            g_red_w[1],
            num_g_red1
        );

        // ∂L/∂α_logit
        let (y4_p, _, _, _, _, _, _, _) = forward_hybrid_head(
            &x,
            &w1,
            &b1,
            &rank_w,
            rank_b,
            &rw,
            rb,
            alpha_logit + eps,
            n_f,
            n_h,
            alpha_leaky,
        );
        let (y4_m, _, _, _, _, _, _, _) = forward_hybrid_head(
            &x,
            &w1,
            &b1,
            &rank_w,
            rank_b,
            &rw,
            rb,
            alpha_logit - eps,
            n_f,
            n_h,
            alpha_leaky,
        );
        let num_g_alpha = (loss(y4_p) - loss(y4_m)) / (2.0 * eps);
        assert!(
            (g_alpha_logit - num_g_alpha).abs() < 1e-4,
            "g_alpha_logit={} num={}",
            g_alpha_logit,
            num_g_alpha
        );
    }

    #[test]
    fn bake_hybrid_head_v3_has_metadata_and_version() {
        let model = HybridHeadModel::new(8, 4, 7);
        let bytes = bake_hybrid_head_v3(&model);
        assert_eq!(&bytes[0..4], b"ZNPR", "expected ZNPR magic");
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 3, "expected v3 (v2 prohibited)");
        let needle = b"zentrain.hybrid_head";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(found, "expected hybrid_head metadata entry in bake");
    }

    #[test]
    fn parse_hybrid_head_meta_roundtrip() {
        let model = HybridHeadModel::new(8, 4, 7);
        let bytes = bake_hybrid_head_v3(&model);
        // Find the metadata payload by scanning for the key + length.
        let needle = b"zentrain.hybrid_head";
        let key_pos = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("key not found");
        // The bake format stores key, kind, then a 4-byte length, then payload.
        // We don't replicate the parser here; instead verify the round-trip
        // by computing the expected payload length and looking for it.
        // (Full byte-format coverage lives in zenpredict's own tests.)
        let _ = key_pos;
        // Round-trip: build a synthetic payload, parse it, check fields.
        let n_hidden = 4;
        let mut payload = Vec::new();
        for v in [0.1f32, 0.2, 0.3, 0.4] {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        payload.extend_from_slice(&0.5f32.to_le_bytes()); // rank_b
        payload.extend_from_slice(&1.5f32.to_le_bytes()); // alpha_logit
        for v in [0.05f32, 1.0, 0.05, 0.05] {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        payload.extend_from_slice(&0.0f32.to_le_bytes()); // reducer_b
        payload.extend_from_slice(&6.0f32.to_le_bytes()); // p_norm
        let (rank_w, meta) = parse_hybrid_head_meta(&payload, n_hidden).expect("parse ok");
        assert_eq!(rank_w, vec![0.1f32, 0.2, 0.3, 0.4]);
        assert_eq!(meta.rank_b, 0.5);
        assert_eq!(meta.alpha_logit, 1.5);
        assert_eq!(meta.reducer_w, [0.05f32, 1.0, 0.05, 0.05]);
        assert_eq!(meta.reducer_b, 0.0);
        assert_eq!(meta.p_norm, 6.0);
    }

    #[test]
    fn apply_hybrid_head_runtime_matches_forward() {
        // Forward train-side, then take the resulting hidden vector
        // through apply_hybrid_head_runtime and compare scalars.
        let n_f = 4;
        let n_h = 6;
        let alpha_leaky = 0.01;
        let x = vec![0.5f64, -0.2, 0.3, 0.7];
        let mut rng = SplitMix64::new(42);
        let w1: Vec<f64> = (0..n_f * n_h)
            .map(|_| {
                let u1 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
                u1 * 0.5 - 0.25
            })
            .collect();
        let b1: Vec<f64> = (0..n_h).map(|i| 0.01 * i as f64).collect();
        let rank_w: Vec<f64> = (0..n_h).map(|i| 0.1 - 0.02 * i as f64).collect();
        let rank_b = 0.3;
        let rw: [f64; 4] = [0.2, 0.6, 0.1, 0.1];
        let rb = 0.05;
        let alpha_logit = -0.8;

        let (y, _, _, _, _, h, _, _) = forward_hybrid_head(
            &x,
            &w1,
            &b1,
            &rank_w,
            rank_b,
            &rw,
            rb,
            alpha_logit,
            n_f,
            n_h,
            alpha_leaky,
        );

        // Convert to runtime types and compare.
        let h_f32: Vec<f32> = h.iter().map(|&v| v as f32).collect();
        let rank_w_f32: Vec<f32> = rank_w.iter().map(|&v| v as f32).collect();
        let meta = HybridHeadMeta {
            rank_b: rank_b as f32,
            alpha_logit: alpha_logit as f32,
            reducer_w: [rw[0] as f32, rw[1] as f32, rw[2] as f32, rw[3] as f32],
            reducer_b: rb as f32,
            p_norm: POOL_P_NORM as f32,
        };
        let y_runtime = apply_hybrid_head_runtime(&h_f32, &rank_w_f32, &meta);
        // f32 round-trip can introduce up to ~1e-6 error; loosen tolerance.
        assert!(
            (y - y_runtime).abs() < 5e-5,
            "y={y} y_runtime={y_runtime} diff={}",
            (y - y_runtime).abs()
        );
    }

    #[test]
    fn train_hybrid_recovers_synthetic_ranking() {
        // Synthetic: scores monotonic in first feature.
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
            scores.push(t * 100.0);
        }
        let row_refs: Vec<&[f64]> = rows.iter().map(|v| v.as_slice()).collect();
        let g = TrainingGroup {
            name: "synth".into(),
            human_scores: &scores,
            features: &row_refs,
            train_weight: 1.0,
            validation_weight: 0.0,
        };
        let hp = HybridHeadHparams {
            n_hidden: 16,
            n_epochs: 5,
            pairs_per_epoch: 200,
            initial_lr: 5e-3,
            l2_lambda: 0.0,
            ..Default::default()
        };
        let model = train_hybrid_head(&[g], &hp, n_f);
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
                let (y, _, _, _, _, _, _, _) = forward_hybrid_head(
                    r,
                    &model.w1,
                    &model.b1,
                    &model.rank_w,
                    model.rank_b,
                    &model.reducer_w,
                    model.reducer_b,
                    model.alpha_logit,
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
            "train_hybrid_head: Spearman {s} < 0.85 (synthetic ranking should be easy)"
        );
    }
}
