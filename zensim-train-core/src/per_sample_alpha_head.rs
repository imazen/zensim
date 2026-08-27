//! Per-sample α head — EX-2 follow-up to the scalar-α hybrid head.
//!
//! Replaces the bake-level scalar `α_logit` in
//! `hybrid_head.rs` with a learned function of the encoder hidden
//! vector: `α(x) = sigmoid(W_α · h + b_α)`. Lets α vary per-pair so
//! photo-like inputs can pull α toward the rank-head while JND-step-
//! grid inputs (KonJND-shaped) pull α toward the pool-head — without
//! the scalar α getting stuck at a single global compromise (0.62
//! ± 0.014 in the prior V_24 hybrid bake, structurally pinned by the
//! rank head's CID22-shaped gradient).
//!
//! ```text
//! y_rank = h · rank_w + rank_b                              (rank head)
//! y_pool = [μ, σ, max, p_6] · reducer_w + reducer_b         (pool head)
//! α      = sigmoid(W_α · h + b_α)                           (per-sample α)
//! y      = α · y_rank + (1 − α) · y_pool
//! ```
//!
//! Architectural cost vs scalar-α hybrid: `+(n_hidden + 1)` weights.
//! At `h = 128` that's +129 floats (≈ +500 bytes packed) — bake size
//! moves from `4 · (n_hidden + 8) = 544 B` (n_hidden=128 hybrid
//! metadata payload) to `4 · (n_hidden + n_hidden + 8) = 1056 B`
//! (the additional `W_α[n_hidden]` block).
//!
//! Backprop is a mechanical sigmoid extension on top of
//! `hybrid_head::backprop_step_hybrid_head`:
//! - `∂L/∂α       = ∂L/∂y · (y_rank − y_pool)`
//! - `∂L/∂α_logit = ∂L/∂α · α · (1 − α)`           (σ'(α_logit))
//! - `∂L/∂W_α[j]  = ∂L/∂α_logit · h[j]`
//! - `∂L/∂b_α     = ∂L/∂α_logit`
//! - `∂L/∂h[j]   += ∂L/∂α_logit · W_α[j]`          (extra h-grad
//!   contribution beyond rank + pool partials)
//!
//! **Bake metadata format (`zentrain.per_sample_alpha_head`)**:
//! Payload = `[W_α[0..n_hidden]] [b_α] [rank_w[0..n_hidden]] [rank_b]
//! [reducer_w[0..4]] [reducer_b] [p_norm]` as f32 little-endian. Total
//! size = `4 · (2·n_hidden + 8)`.
//!
//! Runtime detection: `apply_mlp_scoring` checks for this metadata key
//! BEFORE `zentrain.hybrid_head` (per-sample is the more general form;
//! scalar-α hybrid is a special case where `W_α = 0` and the
//! `α_logit` lives in `b_α`).

use crate::TrainingGroup;
use crate::adam::AdamState;
use crate::mlp::compute_scaler_from_groups;
use crate::pool_head::{POOL_P_NORM, POOL_STD_FLOOR, pool_stats};
use crate::rng::SplitMix64;
use crate::simd_encoder;
use zenpredict::{Activation, FeatureTransform, MetadataType, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

#[inline]
fn sigmoid(x: f64) -> f64 {
    let xc = x.clamp(-20.0, 20.0);
    1.0 / (1.0 + (-xc).exp())
}

/// `(y, y_rank, y_pool, alpha, alpha_logit, h_pre, h, stats, max_idx)` —
/// the forward-pass intermediates of [`forward_per_sample_alpha_head`].
pub type PerSampleAlphaForward = (f64, f64, f64, f64, f64, Vec<f64>, Vec<f64>, [f64; 4], usize);

/// Full forward pass for the per-sample α head.
///
/// Returns `(y, y_rank, y_pool, alpha, alpha_logit, h_pre, h, stats,
/// max_idx)` — every intermediate the backprop step needs. `h_pre`
/// is the pre-LeakyReLU encoder state, `h` is post-LeakyReLU.
/// `alpha_logit` is the raw pre-sigmoid α prediction (used by the
/// backprop step to compute the sigmoid derivative).
#[allow(clippy::too_many_arguments)]
pub fn forward_per_sample_alpha_head(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    rank_w: &[f64],
    rank_b: f64,
    reducer_w: &[f64; 4],
    reducer_b: f64,
    w_alpha: &[f64],
    b_alpha: f64,
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f64,
) -> PerSampleAlphaForward {
    debug_assert_eq!(rank_w.len(), n_hidden);
    debug_assert_eq!(w_alpha.len(), n_hidden);

    // Encoder: h_pre = b1 + Σ_i x[i]·W1[i,:], then LeakyReLU → h.
    // The SIMD-dispatched path picks AVX-512 / AVX2 / NEON / wasm128
    // automatically and falls back to scalar on platforms without a
    // matching tier. Bit-identical to the scalar oracle modulo FMA
    // fusion (~1e-12 relative drift at the 372-feature × 128-hidden
    // production shape).
    let (h_pre, h) = simd_encoder::encoder_forward(x, w1, b1, n_features, n_hidden, leaky_alpha);

    // Rank head: y_rank = rank_b + h · rank_w (n_hidden-wide dot).
    let y_rank = simd_encoder::dot_bias(&h, rank_w, rank_b);

    // Pool head: 4-tuple statistic reducer (μ, σ, max, p_6).
    let (stats, max_idx) = pool_stats(&h);
    let y_pool = stats[0] * reducer_w[0]
        + stats[1] * reducer_w[1]
        + stats[2] * reducer_w[2]
        + stats[3] * reducer_w[3]
        + reducer_b;

    // Per-sample α logit + sigmoid gate, then linear combination of
    // the two heads.
    let alpha_logit = simd_encoder::dot_bias(&h, w_alpha, b_alpha);
    let alpha = sigmoid(alpha_logit);
    let y = alpha * y_rank + (1.0 - alpha) * y_pool;
    (
        y,
        y_rank,
        y_pool,
        alpha,
        alpha_logit,
        h_pre,
        h,
        stats,
        max_idx,
    )
}

/// Heads-only forward: given a pre-computed hidden vector `h` (from any
/// encoder — 1-layer, 2-layer, skip, etc.), compute the rank head, pool
/// head, per-sample α gate, and combined output.
///
/// This is the same math as `forward_per_sample_alpha_head` but without
/// the encoder forward — the caller provides `h` and `h_pre` directly.
/// Used by mlp_train.rs when it wants to choose the encoder externally
/// (e.g., 2-layer or skip variants).
#[allow(clippy::too_many_arguments)]
pub fn forward_heads(
    h: &[f64],
    rank_w: &[f64],
    rank_b: f64,
    reducer_w: &[f64; 4],
    reducer_b: f64,
    w_alpha: &[f64],
    b_alpha: f64,
    n_hidden: usize,
) -> (f64, f64, f64, f64, f64, [f64; 4], usize) {
    debug_assert_eq!(h.len(), n_hidden);
    debug_assert_eq!(rank_w.len(), n_hidden);
    debug_assert_eq!(w_alpha.len(), n_hidden);

    let y_rank = simd_encoder::dot_bias(h, rank_w, rank_b);
    let (stats, max_idx) = pool_stats(h);
    let y_pool = stats[0] * reducer_w[0]
        + stats[1] * reducer_w[1]
        + stats[2] * reducer_w[2]
        + stats[3] * reducer_w[3]
        + reducer_b;
    let alpha_logit = simd_encoder::dot_bias(h, w_alpha, b_alpha);
    let alpha = sigmoid(alpha_logit);
    let y = alpha * y_rank + (1.0 - alpha) * y_pool;
    (y, y_rank, y_pool, alpha, alpha_logit, stats, max_idx)
}

/// Heads-only backprop: given dl/dy and the forward intermediates,
/// compute gradients for the head weights AND dl/dh (the gradient
/// that flows back into the encoder). The caller then routes dl/dh
/// through whichever encoder backprop it used.
///
/// Returns `dl_dh` (length `n_hidden`) — the gradient to propagate
/// into the encoder backward pass.
#[allow(clippy::too_many_arguments)]
pub fn backprop_heads(
    h: &[f64],
    stats: &[f64; 4],
    max_idx: usize,
    y_rank: f64,
    y_pool: f64,
    alpha: f64,
    dl_dy: f64,
    rank_w: &[f64],
    reducer_w: &[f64; 4],
    w_alpha: &[f64],
    g_rank_w: &mut [f64],
    g_rank_b: &mut f64,
    g_reducer_w: &mut [f64; 4],
    g_reducer_b: &mut f64,
    g_w_alpha: &mut [f64],
    g_b_alpha: &mut f64,
    n_hidden: usize,
    leaky_alpha: f64,
) -> Vec<f64> {
    debug_assert_eq!(rank_w.len(), n_hidden);
    debug_assert_eq!(g_rank_w.len(), n_hidden);
    debug_assert_eq!(w_alpha.len(), n_hidden);
    debug_assert_eq!(g_w_alpha.len(), n_hidden);

    let dl_dy_rank = dl_dy * alpha;
    let dl_dy_pool = dl_dy * (1.0 - alpha);
    let dl_dalpha = dl_dy * (y_rank - y_pool);
    let dl_dalpha_logit = dl_dalpha * alpha * (1.0 - alpha);

    for j in 0..n_hidden {
        g_rank_w[j] += dl_dy_rank * h[j];
    }
    *g_rank_b += dl_dy_rank;

    for k in 0..4 {
        g_reducer_w[k] += dl_dy_pool * stats[k];
    }
    *g_reducer_b += dl_dy_pool;

    for j in 0..n_hidden {
        g_w_alpha[j] += dl_dalpha_logit * h[j];
    }
    *g_b_alpha += dl_dalpha_logit;

    // dl/dh = rank + pool + α-head contributions.
    let mut dl_dh = vec![0.0f64; n_hidden];
    for j in 0..n_hidden {
        dl_dh[j] += dl_dy_rank * rank_w[j];
        dl_dh[j] += dl_dalpha_logit * w_alpha[j];
    }

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
    let dl_dstat: [f64; 4] = [
        dl_dy_pool * reducer_w[0],
        dl_dy_pool * reducer_w[1],
        dl_dy_pool * reducer_w[2],
        dl_dy_pool * reducer_w[3],
    ];
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

    let _ = leaky_alpha; // used by the caller for encoder backprop
    dl_dh
}

/// Backprop ∂L/∂y through the per-sample α head to gradients on
/// (w1, b1, rank_w, rank_b, reducer_w, reducer_b, W_α, b_α).
///
/// Chain rule (extends hybrid_head with W_α path):
/// - `∂L/∂y_rank   = ∂L/∂y · α`
/// - `∂L/∂y_pool   = ∂L/∂y · (1 − α)`
/// - `∂L/∂α       = ∂L/∂y · (y_rank − y_pool)`
/// - `∂L/∂α_logit = ∂L/∂α · α · (1 − α)`     (σ'(α_logit))
/// - `∂L/∂W_α[j]  = ∂L/∂α_logit · h[j]`
/// - `∂L/∂b_α     = ∂L/∂α_logit`
/// - `∂L/∂h_j     += ∂L/∂α_logit · W_α[j]`   (W_α path into the
///   encoder)
///
/// The remaining rank + pool contributions to `∂L/∂h_j` and
/// `∂L/∂w1` are identical to `hybrid_head::backprop_step_hybrid_head`.
#[allow(clippy::too_many_arguments)]
pub fn backprop_step_per_sample_alpha_head(
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
    w_alpha: &[f64],
    gw1: &mut [f64],
    gb1: &mut [f64],
    g_rank_w: &mut [f64],
    g_rank_b: &mut f64,
    g_reducer_w: &mut [f64; 4],
    g_reducer_b: &mut f64,
    g_w_alpha: &mut [f64],
    g_b_alpha: &mut f64,
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f64,
) {
    debug_assert_eq!(rank_w.len(), n_hidden);
    debug_assert_eq!(g_rank_w.len(), n_hidden);
    debug_assert_eq!(w_alpha.len(), n_hidden);
    debug_assert_eq!(g_w_alpha.len(), n_hidden);

    let dl_dy_rank = dl_dy * alpha;
    let dl_dy_pool = dl_dy * (1.0 - alpha);
    let dl_dalpha = dl_dy * (y_rank - y_pool);
    // σ'(α_logit) = α · (1 − α)
    let dl_dalpha_logit = dl_dalpha * alpha * (1.0 - alpha);

    // Rank-head grads.
    for j in 0..n_hidden {
        g_rank_w[j] += dl_dy_rank * h[j];
    }
    *g_rank_b += dl_dy_rank;

    // Pool-head grads.
    for k in 0..4 {
        g_reducer_w[k] += dl_dy_pool * stats[k];
    }
    *g_reducer_b += dl_dy_pool;

    // Per-sample α-head grads.
    for j in 0..n_hidden {
        g_w_alpha[j] += dl_dalpha_logit * h[j];
    }
    *g_b_alpha += dl_dalpha_logit;

    // ∂L/∂h_j = rank + pool + α-head contributions.
    let mut dl_dh = vec![0.0f64; n_hidden];

    for j in 0..n_hidden {
        dl_dh[j] += dl_dy_rank * rank_w[j];
        dl_dh[j] += dl_dalpha_logit * w_alpha[j];
    }

    // Pool stats chain rule (same as hybrid_head).
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

    // LeakyReLU back-route: scale dl_dh by `leaky_alpha` on lanes
    // where the forward saw a negative pre-activation.
    let dl_dh_pre = simd_encoder::leaky_relu_backward(&dl_dh, h_pre, leaky_alpha);

    // Layer-1 grads: accumulate into gw1 / gb1 via the SIMD scatter +
    // in-place add kernels.
    simd_encoder::encoder_backprop_layer1(x, &dl_dh_pre, gw1, gb1, n_features, n_hidden);
}

/// Hyperparameters for the per-sample α head trainer.
#[derive(Clone, Debug)]
pub struct PerSampleAlphaHeadHparams {
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
    /// L2 regularization on layer weights (and rank_w + reducer_w +
    /// W_α). b_α unregularized.
    pub l2_lambda: f64,
}

impl Default for PerSampleAlphaHeadHparams {
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

/// Trained per-sample α head model.
#[derive(Debug)]
pub struct PerSampleAlphaHeadModel {
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
    /// Pool-head reducer weights [w_μ, w_σ, w_max, w_p6].
    pub reducer_w: [f64; 4],
    /// Pool-head reducer bias.
    pub reducer_b: f64,
    /// Per-sample α head weights (n_hidden).
    pub w_alpha: Vec<f64>,
    /// Per-sample α head bias (also the scalar-α fallback when
    /// W_α = 0).
    pub b_alpha: f64,
    /// Hidden width.
    pub n_hidden: usize,
    /// Input dim.
    pub n_features: usize,
    /// LeakyReLU α the model was TRAINED with. Bake-emit uses this to pick
    /// the activation in the bake: if α ≈ 1.0 we emit `Activation::Identity`
    /// (the runtime's `LeakyRelu` has a hardcoded α = 0.01, so emitting it
    /// would mismatch the trained forward pass — collapsing hidden=1
    /// configurations entirely; see #40). Default 0.01 keeps the existing
    /// behavior; the trainer sets this from `hparams.leaky_alpha`.
    pub leaky_alpha: f64,
}

/// Pick the bake-time activation that matches the trained `leaky_alpha`.
/// The runtime's LeakyRelu has a fixed α = 0.01, so emitting LeakyRelu when
/// the trainer used α = 1.0 gives a different forward at inference than at
/// training. Identity is the right wire-format activation for α = 1.0.
#[inline]
fn activation_for_leaky(leaky_alpha: f64) -> Activation {
    if (leaky_alpha - 1.0).abs() < 1e-9 {
        Activation::Identity
    } else {
        Activation::LeakyRelu
    }
}

impl PerSampleAlphaHeadModel {
    /// Initialize: same shape as `HybridHeadModel::new`, with
    /// additional `W_α` initialized to zeros (so α at init is
    /// `sigmoid(b_α)` for every input, matching the scalar-α
    /// behavior at start). `b_α = 0` → initial α = 0.5 for every
    /// pair. Pool-dominant reducer init (matches scalar-α hybrid).
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
            w_alpha: vec![0.0; n_hidden],
            b_alpha: 0.0,
            n_hidden,
            n_features,
            leaky_alpha: 0.01, // matches the runtime's hardcoded LEAKY_RELU_ALPHA; trainer overrides from hparams
        }
    }
}

/// Standalone trainer for the per-sample α head. Mirrors
/// `train_hybrid_head` (synthetic-rank smoke test). Production
/// training goes through `train_mlp_per_sample_alpha_head_with_tv`
/// in `zensim-validate::mlp_train`.
pub fn train_per_sample_alpha_head(
    groups: &[TrainingGroup<'_>],
    hparams: &PerSampleAlphaHeadHparams,
    n_features: usize,
) -> PerSampleAlphaHeadModel {
    let train_indices: Vec<usize> = (0..groups.len())
        .filter(|&i| groups[i].train_weight > 0.0)
        .collect();
    assert!(
        !train_indices.is_empty(),
        "train_per_sample_alpha_head: need at least one group with train_weight > 0"
    );

    let (mean, scale) = compute_scaler_from_groups(groups, &train_indices, n_features);

    let mut model = PerSampleAlphaHeadModel::new(n_features, hparams.n_hidden, hparams.seed);
    model.scaler_mean = mean;
    model.scaler_scale = scale;
    model.leaky_alpha = hparams.leaky_alpha;

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
    // Adam w2 slot holds: [rank_w (n_hidden) | reducer_w (4) | W_α
    // (n_hidden) | b_α (1)]. Adam b2 slot: [rank_b, reducer_b].
    let n_w2 = n_hidden + 4 + n_hidden + 1;
    let n_b2 = 2;
    let mut adam = AdamState::new(nw1, nb1, n_w2, n_b2);
    let mut rng = SplitMix64::new(hparams.seed ^ 0x5A5A_5A5A_5A5A_5A5A);

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

            let (yhi, yhi_rank, yhi_pool, alpha_hi, _aliphi, hp_hi, h_hi, stats_hi, max_idx_hi) =
                forward_per_sample_alpha_head(
                    xhi,
                    &model.w1,
                    &model.b1,
                    &model.rank_w,
                    model.rank_b,
                    &model.reducer_w,
                    model.reducer_b,
                    &model.w_alpha,
                    model.b_alpha,
                    n_features,
                    n_hidden,
                    alpha_leaky,
                );
            let (ylo, ylo_rank, ylo_pool, alpha_lo, _alipllo, hp_lo, h_lo, stats_lo, max_idx_lo) =
                forward_per_sample_alpha_head(
                    xlo,
                    &model.w1,
                    &model.b1,
                    &model.rank_w,
                    model.rank_b,
                    &model.reducer_w,
                    model.reducer_b,
                    &model.w_alpha,
                    model.b_alpha,
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
            let mut g_w_alpha_buf = vec![0.0f64; n_hidden];
            let mut g_b_alpha: f64 = 0.0;

            backprop_step_per_sample_alpha_head(
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
                &model.w_alpha,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_w_alpha_buf,
                &mut g_b_alpha,
                n_features,
                n_hidden,
                alpha_leaky,
            );
            backprop_step_per_sample_alpha_head(
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
                &model.w_alpha,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_w_alpha_buf,
                &mut g_b_alpha,
                n_features,
                n_hidden,
                alpha_leaky,
            );

            if l2 > 0.0 {
                for (g, &w) in adam.gw1.iter_mut().zip(model.w1.iter()) {
                    *g += 2.0 * l2 * w;
                }
                for j in 0..n_hidden {
                    g_rank_w_buf[j] += 2.0 * l2 * model.rank_w[j];
                    g_w_alpha_buf[j] += 2.0 * l2 * model.w_alpha[j];
                }
                for (g, &w) in g_red_w.iter_mut().zip(model.reducer_w.iter()) {
                    *g += 2.0 * l2 * w;
                }
                // b_α unregularized.
            }

            // Pack into Adam w2/b2 slots.
            for (g, &v) in adam.gw2.iter_mut().zip(g_rank_w_buf.iter()) {
                *g += v;
            }
            for (g, &v) in adam.gw2[n_hidden..n_hidden + 4]
                .iter_mut()
                .zip(g_red_w.iter())
            {
                *g += v;
            }
            for (g, &v) in adam.gw2[n_hidden + 4..n_hidden + 4 + n_hidden]
                .iter_mut()
                .zip(g_w_alpha_buf.iter())
            {
                *g += v;
            }
            adam.gw2[n_hidden + 4 + n_hidden] += g_b_alpha;
            adam.gb2[0] += g_rank_b_buf;
            adam.gb2[1] += g_red_b;

            // Adam step: pack/unpack into w2.
            let mut w2_vec = vec![0.0f64; n_w2];
            w2_vec[..n_hidden].copy_from_slice(&model.rank_w);
            w2_vec[n_hidden..n_hidden + 4].copy_from_slice(&model.reducer_w);
            w2_vec[n_hidden + 4..n_hidden + 4 + n_hidden].copy_from_slice(&model.w_alpha);
            w2_vec[n_hidden + 4 + n_hidden] = model.b_alpha;
            let mut b2_vec = vec![model.rank_b, model.reducer_b];
            adam.step(&mut model.w1, &mut model.b1, &mut w2_vec, &mut b2_vec, lr);
            model.rank_w.copy_from_slice(&w2_vec[..n_hidden]);
            model
                .reducer_w
                .copy_from_slice(&w2_vec[n_hidden..n_hidden + 4]);
            model
                .w_alpha
                .copy_from_slice(&w2_vec[n_hidden + 4..n_hidden + 4 + n_hidden]);
            model.b_alpha = w2_vec[n_hidden + 4 + n_hidden];
            model.rank_b = b2_vec[0];
            model.reducer_b = b2_vec[1];

            // Silence unused-binding warnings.
            let _ = (yhi, ylo);
        }
    }

    model
}

/// Bake the per-sample α head model into ZNPR v3 bytes. Wire format
/// mirrors `bake_hybrid_head_v3` exactly except for the metadata
/// key + payload. Passthrough second layer; runtime reads the hidden
/// vector and computes y_rank + y_pool + α_logit(h) + mix.
///
/// **Metadata key**: `zentrain.per_sample_alpha_head`
/// **Payload** (f32 LE):
/// - `W_α[0..n_hidden]` — n_hidden floats
/// - `b_α` — 1 float
/// - `rank_w[0..n_hidden]` — n_hidden floats
/// - `rank_b` — 1 float
/// - `reducer_w[0..4]` — 4 floats
/// - `reducer_b` — 1 float
/// - `p_norm` — 1 float (6.0 currently)
///
/// Total bytes = `4 · (2·n_hidden + 8)`.
pub fn bake_per_sample_alpha_head_v3(model: &PerSampleAlphaHeadModel) -> Vec<u8> {
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

    let n_payload = 2 * n_hidden + 8;
    let mut payload = Vec::with_capacity(n_payload * 4);
    for v in &model.w_alpha {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.b_alpha as f32).to_le_bytes());
    for v in &model.rank_w {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.rank_b as f32).to_le_bytes());
    for v in &model.reducer_w {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.reducer_b as f32).to_le_bytes());
    payload.extend_from_slice(&(POOL_P_NORM as f32).to_le_bytes());

    let layers = [
        BakeLayer {
            in_dim: n_features,
            out_dim: n_hidden,
            activation: activation_for_leaky(model.leaky_alpha),
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
        key: "zentrain.per_sample_alpha_head",
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
    .expect("v3 bake of per-sample α head MLP")
}

/// EXP-CROSS-CODEC-V4 (2026-05-19): bake a per-sample-α head with the
/// `zentrain.tanh_output_head` metadata entry, marking the bake as
/// score-pinned via `y_score = 100 · σ(y_pre / scale)`.
///
/// Equivalent to `bake_per_sample_alpha_head_v3` plus a second
/// metadata entry `zentrain.tanh_output_head` with payload `[scale: f32]`
/// (4 bytes, little-endian). The runtime in zensim's `apply_mlp_scoring`
/// recognizes the key and applies the matching sigmoid pin AT INFERENCE
/// — no post-hoc affine needed.
///
/// `scale` must be `> 0`. The recommended value is `10.0` (active
/// linear region `y_pre ∈ [−30, 30]` mapping to `[5, 95]` score units).
pub fn bake_per_sample_alpha_head_v3_with_tanh(
    model: &PerSampleAlphaHeadModel,
    scale: f64,
) -> Vec<u8> {
    assert!(
        scale > 0.0,
        "tanh_output_head scale must be > 0; got {scale}"
    );
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

    let n_payload = 2 * n_hidden + 8;
    let mut payload = Vec::with_capacity(n_payload * 4);
    for v in &model.w_alpha {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.b_alpha as f32).to_le_bytes());
    for v in &model.rank_w {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.rank_b as f32).to_le_bytes());
    for v in &model.reducer_w {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.reducer_b as f32).to_le_bytes());
    payload.extend_from_slice(&(POOL_P_NORM as f32).to_le_bytes());

    // Tanh-output-head payload: [scale: f32 LE].
    let tanh_payload: [u8; 4] = (scale as f32).to_le_bytes();

    let layers = [
        BakeLayer {
            in_dim: n_features,
            out_dim: n_hidden,
            activation: activation_for_leaky(model.leaky_alpha),
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
    let metadata = [
        BakeMetadataEntry {
            key: "zentrain.per_sample_alpha_head",
            kind: MetadataType::Numeric,
            value: &payload,
        },
        BakeMetadataEntry {
            key: "zentrain.tanh_output_head",
            kind: MetadataType::Numeric,
            value: &tanh_payload,
        },
    ];
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
    .expect("v3 bake of per-sample α head MLP with tanh output head")
}

/// Bake a per-sample-α head with the tanh-output-head metadata PLUS
/// `zentrain.feature_transforms` + (optional) `zentrain.feature_transform_params`.
///
/// Equivalent to [`bake_per_sample_alpha_head_v3_with_tanh`] but
/// additionally writes the feature-transform spec so runtime / verdict
/// tools can apply the same transforms the trainer applied. Without
/// this metadata, a bake trained via `--auto-transforms` produces
/// silently-wrong predictions on raw input features (verified
/// 2026-05-25, task #214 Phase 2: v11 + auto-transforms with no
/// metadata gave CID22 SROCC 0.215 vs the v11 ship's 0.860).
///
/// Empty `feature_transforms` (or all-Identity) emits no metadata
/// keys — matches the [`bake_per_sample_alpha_head_v3_with_tanh`]
/// behaviour. `feature_transform_params` may be `None` even when
/// `feature_transforms` is set, for the case where every transform
/// is unparameterized (the non-parameterized variants).
///
/// Added 2026-05-25 (task #214 Phase 2).
pub fn bake_per_sample_alpha_head_v3_with_tanh_and_transforms(
    model: &PerSampleAlphaHeadModel,
    scale: f64,
    feature_transforms: Option<&[FeatureTransform]>,
    feature_transform_params: Option<&[Vec<f32>]>,
    output_spline_payload: Option<&[u8]>,
) -> Vec<u8> {
    assert!(
        scale > 0.0,
        "tanh_output_head scale must be > 0; got {scale}"
    );
    let n_features = model.n_features;
    let n_hidden = model.n_hidden;

    // Build the feature_transforms text + transform_params text if
    // provided AND non-trivial. All-Identity transforms are equivalent
    // to absence, so skip the metadata in that case.
    let nontrivial =
        feature_transforms.is_some_and(|ts| ts.iter().any(|&t| t != FeatureTransform::Identity));

    let transforms_text = if nontrivial {
        let ts = feature_transforms.unwrap();
        assert_eq!(
            ts.len(),
            n_features,
            "feature_transforms length {} != n_features {}",
            ts.len(),
            n_features
        );
        let mut s = String::new();
        for (i, t) in ts.iter().enumerate() {
            if i > 0 {
                s.push('\n');
            }
            s.push_str(t.as_token());
        }
        Some(s)
    } else {
        None
    };

    // Only emit params if any transform actually has parameters.
    let params_text = if let Some(params) = feature_transform_params {
        if nontrivial && params.iter().any(|p| !p.is_empty()) {
            assert_eq!(
                params.len(),
                n_features,
                "feature_transform_params length {} != n_features {}",
                params.len(),
                n_features
            );
            let mut s = String::new();
            for (i, p) in params.iter().enumerate() {
                if i > 0 {
                    s.push('\n');
                }
                for (j, &v) in p.iter().enumerate() {
                    if j > 0 {
                        s.push(',');
                    }
                    // Use 9-digit precision — preserves f32 round-trip.
                    s.push_str(&format!("{v:.9}"));
                }
            }
            Some(s)
        } else {
            None
        }
    } else {
        None
    };

    let scaler_mean_f32: Vec<f32> = model.scaler_mean.iter().map(|&v| v as f32).collect();
    let scaler_scale_f32: Vec<f32> = model.scaler_scale.iter().map(|&v| v as f32).collect();
    let w1_f32: Vec<f32> = model.w1.iter().map(|&v| v as f32).collect();
    let b1_f32: Vec<f32> = model.b1.iter().map(|&v| v as f32).collect();
    let mut w2_f32 = vec![0.0f32; n_hidden * n_hidden];
    for i in 0..n_hidden {
        w2_f32[i * n_hidden + i] = 1.0;
    }
    let b2_f32 = vec![0.0f32; n_hidden];

    let n_payload = 2 * n_hidden + 8;
    let mut payload = Vec::with_capacity(n_payload * 4);
    for v in &model.w_alpha {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.b_alpha as f32).to_le_bytes());
    for v in &model.rank_w {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.rank_b as f32).to_le_bytes());
    for v in &model.reducer_w {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.reducer_b as f32).to_le_bytes());
    payload.extend_from_slice(&(POOL_P_NORM as f32).to_le_bytes());

    let tanh_payload: [u8; 4] = (scale as f32).to_le_bytes();

    let layers = [
        BakeLayer {
            in_dim: n_features,
            out_dim: n_hidden,
            activation: activation_for_leaky(model.leaky_alpha),
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

    // Build metadata array dynamically — required two entries plus
    // optional 0/1/2 entries for transforms + params.
    let mut metadata: Vec<BakeMetadataEntry<'_>> = Vec::with_capacity(4);
    metadata.push(BakeMetadataEntry {
        key: "zentrain.per_sample_alpha_head",
        kind: MetadataType::Numeric,
        value: &payload,
    });
    metadata.push(BakeMetadataEntry {
        key: "zentrain.tanh_output_head",
        kind: MetadataType::Numeric,
        value: &tanh_payload,
    });
    if let Some(t) = &transforms_text {
        metadata.push(BakeMetadataEntry {
            key: "zentrain.feature_transforms",
            kind: MetadataType::Utf8,
            value: t.as_bytes(),
        });
    }
    if let Some(p) = &params_text {
        metadata.push(BakeMetadataEntry {
            key: "zentrain.feature_transform_params",
            kind: MetadataType::Utf8,
            value: p.as_bytes(),
        });
    }

    if let Some(spline) = output_spline_payload {
        metadata.push(BakeMetadataEntry {
            key: "zentrain.output_calibration_spline",
            kind: MetadataType::Numeric,
            value: spline,
        });
    }

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
    .expect("v3 bake of per-sample α head MLP with tanh output head + feature transforms")
}

/// Parsed per-sample α head metadata payload (runtime-side).
#[derive(Clone, Debug)]
pub struct PerSampleAlphaHeadMeta {
    /// Per-sample α head weights (n_hidden).
    pub w_alpha: Vec<f32>,
    /// Per-sample α head bias.
    pub b_alpha: f32,
    /// Rank-net weights (n_hidden).
    pub rank_w: Vec<f32>,
    /// Rank-net bias.
    pub rank_b: f32,
    /// Pool-head reducer weights.
    pub reducer_w: [f32; 4],
    /// Pool-head reducer bias.
    pub reducer_b: f32,
    /// p-norm exponent (6.0).
    pub p_norm: f32,
}

/// Parse the `zentrain.per_sample_alpha_head` payload. Returns
/// `Some(meta)` or `None` if the payload is malformed.
///
/// Layout: `[W_α[0..n_hidden]] [b_α] [rank_w[0..n_hidden]] [rank_b]
/// [reducer_w[0..4]] [reducer_b] [p_norm]` — all f32 LE.
pub fn parse_per_sample_alpha_head_meta(
    payload: &[u8],
    n_hidden: usize,
) -> Option<PerSampleAlphaHeadMeta> {
    let expected = (2 * n_hidden + 8) * 4;
    if payload.len() != expected {
        return None;
    }
    let mut floats = Vec::with_capacity(2 * n_hidden + 8);
    for chunk in payload.as_chunks::<4>().0 {
        floats.push(f32::from_le_bytes(*chunk));
    }
    let w_alpha: Vec<f32> = floats[..n_hidden].to_vec();
    let b_alpha = floats[n_hidden];
    let rank_w: Vec<f32> = floats[n_hidden + 1..2 * n_hidden + 1].to_vec();
    let rank_b = floats[2 * n_hidden + 1];
    let reducer_w = [
        floats[2 * n_hidden + 2],
        floats[2 * n_hidden + 3],
        floats[2 * n_hidden + 4],
        floats[2 * n_hidden + 5],
    ];
    let reducer_b = floats[2 * n_hidden + 6];
    let p_norm = floats[2 * n_hidden + 7];
    Some(PerSampleAlphaHeadMeta {
        w_alpha,
        b_alpha,
        rank_w,
        rank_b,
        reducer_w,
        reducer_b,
        p_norm,
    })
}

/// Runtime forward path: given a hidden vector `h` and parsed
/// per-sample α head metadata, produce the mixed `y`. Also returns
/// the per-sample α it computed — diagnostic callers use this to
/// check α distributions per corpus (the mandatory engagement check).
///
/// Formula:
/// - `y_rank = h · rank_w + rank_b`
/// - `[μ, σ, max, p_norm](h) → y_pool = stats · reducer_w + reducer_b`
/// - `α_logit = h · W_α + b_α`
/// - `α = sigmoid(α_logit)`
/// - `y = α · y_rank + (1 − α) · y_pool`
pub fn apply_per_sample_alpha_head_runtime(h: &[f32], meta: &PerSampleAlphaHeadMeta) -> (f64, f64) {
    let n = h.len();
    debug_assert_eq!(meta.rank_w.len(), n);
    debug_assert_eq!(meta.w_alpha.len(), n);
    debug_assert!(n > 0);

    let mut y_rank = meta.rank_b as f64;
    let mut alpha_logit = meta.b_alpha as f64;
    let mut sum = 0.0f64;
    let mut max_v = f64::NEG_INFINITY;
    let mut sum_p = 0.0f64;
    let p = meta.p_norm as f64;
    for (j, &hj) in h.iter().enumerate() {
        let hjf = hj as f64;
        y_rank += hjf * meta.rank_w[j] as f64;
        alpha_logit += hjf * meta.w_alpha[j] as f64;
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

    let alpha = {
        let xc = alpha_logit.clamp(-20.0, 20.0);
        1.0 / (1.0 + (-xc).exp())
    };
    let y = alpha * y_rank + (1.0 - alpha) * y_pool;
    (y, alpha)
}

/// Bake a 2-layer per-sample α head with optional tanh pin and
/// feature transforms. Emits 3 BakeLayer entries:
///   1. n_features → n_hidden1 (LeakyReLU)
///   2. n_hidden1 → n_hidden_final (LeakyReLU)
///   3. n_hidden_final → n_hidden_final (Identity pass-through)
///
/// Head metadata payload uses n_hidden_final for weight dims.
#[allow(clippy::too_many_arguments)]
pub fn bake_per_sample_alpha_head_v3_2layer(
    model: &PerSampleAlphaHeadModel,
    w2_enc: &[f64],
    b2_enc: &[f64],
    n_hidden1: usize,
    n_hidden_final: usize,
    tanh_scale: Option<f64>,
    feature_transforms: Option<&[zenpredict::FeatureTransform]>,
    feature_transform_params: Option<&[Vec<f32>]>,
    output_spline_payload: Option<&[u8]>,
    out_dtype: WeightDtype,
) -> Vec<u8> {
    let n_features = model.n_features;

    let scaler_mean_f32: Vec<f32> = model.scaler_mean.iter().map(|&v| v as f32).collect();
    let scaler_scale_f32: Vec<f32> = model.scaler_scale.iter().map(|&v| v as f32).collect();
    let w1_f32: Vec<f32> = model.w1.iter().map(|&v| v as f32).collect();
    let b1_f32: Vec<f32> = model.b1.iter().map(|&v| v as f32).collect();
    let w2_enc_f32: Vec<f32> = w2_enc.iter().map(|&v| v as f32).collect();
    let b2_enc_f32: Vec<f32> = b2_enc.iter().map(|&v| v as f32).collect();

    // Identity pass-through for layer 3.
    let mut w3_f32 = vec![0.0f32; n_hidden_final * n_hidden_final];
    for i in 0..n_hidden_final {
        w3_f32[i * n_hidden_final + i] = 1.0;
    }
    let b3_f32 = vec![0.0f32; n_hidden_final];

    // Head metadata payload (same format as 1-layer, but sized for n_hidden_final).
    let n_payload = 2 * n_hidden_final + 8;
    let mut payload = Vec::with_capacity(n_payload * 4);
    for v in &model.w_alpha {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.b_alpha as f32).to_le_bytes());
    for v in &model.rank_w {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.rank_b as f32).to_le_bytes());
    for v in &model.reducer_w {
        payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    payload.extend_from_slice(&(model.reducer_b as f32).to_le_bytes());
    payload.extend_from_slice(&(POOL_P_NORM as f32).to_le_bytes());

    let layers = [
        BakeLayer {
            in_dim: n_features,
            out_dim: n_hidden1,
            activation: activation_for_leaky(model.leaky_alpha),
            dtype: out_dtype,
            weights: &w1_f32,
            biases: &b1_f32,
        },
        BakeLayer {
            in_dim: n_hidden1,
            out_dim: n_hidden_final,
            activation: activation_for_leaky(model.leaky_alpha),
            dtype: out_dtype,
            weights: &w2_enc_f32,
            biases: &b2_enc_f32,
        },
        // Identity passthrough kept F32: it's tiny (n_hidden_final², ~8 KB)
        // and the per-sample-α runtime dispatch reads it as the exact
        // identity — quantizing it buys ~nothing and risks the passthrough.
        BakeLayer {
            in_dim: n_hidden_final,
            out_dim: n_hidden_final,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w3_f32,
            biases: &b3_f32,
        },
    ];

    let mut metadata_entries = vec![BakeMetadataEntry {
        key: "zentrain.per_sample_alpha_head",
        kind: MetadataType::Numeric,
        value: &payload,
    }];

    let tanh_payload;
    if let Some(scale) = tanh_scale {
        tanh_payload = (scale as f32).to_le_bytes().to_vec();
        metadata_entries.push(BakeMetadataEntry {
            key: "zentrain.tanh_output_head",
            kind: MetadataType::Numeric,
            value: &tanh_payload,
        });
    }

    let ft_token_payload;
    let ft_param_payload;
    if let Some(transforms) = feature_transforms {
        let has_nontrivial = transforms
            .iter()
            .any(|t| !matches!(t, zenpredict::FeatureTransform::Identity));
        if has_nontrivial {
            let tokens: String = transforms
                .iter()
                .map(|t| t.as_token())
                .collect::<Vec<_>>()
                .join("\n");
            ft_token_payload = tokens.into_bytes();
            metadata_entries.push(BakeMetadataEntry {
                key: "zentrain.feature_transforms",
                kind: MetadataType::Utf8,
                value: &ft_token_payload,
            });
            if let Some(params) = feature_transform_params {
                let mut s = String::new();
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        s.push('\n');
                    }
                    for (j, &v) in p.iter().enumerate() {
                        if j > 0 {
                            s.push(',');
                        }
                        use std::fmt::Write;
                        let _ = write!(s, "{v:.9}");
                    }
                }
                ft_param_payload = s.into_bytes();
                metadata_entries.push(BakeMetadataEntry {
                    key: "zentrain.feature_transform_params",
                    kind: MetadataType::Utf8,
                    value: &ft_param_payload,
                });
            }
        }
    }

    if let Some(spline) = output_spline_payload {
        metadata_entries.push(BakeMetadataEntry {
            key: "zentrain.output_calibration_spline",
            kind: MetadataType::Numeric,
            value: spline,
        });
    }

    bake(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean_f32,
        scaler_scale: &scaler_scale_f32,
        layers: &layers,
        feature_bounds: &[],
        metadata: &metadata_entries,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        // Compress when quantizing (f16/i8) — lz4 over the zerobias zeros +
        // the smaller weights is where the size win lands.
        compressed: !matches!(out_dtype, WeightDtype::F32),
        hu_permutations: None,
    })
    .expect("v3 bake of 2-layer per-sample α head MLP")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_zero_weights_returns_neutral() {
        let x = vec![1.0; 4];
        let w1 = vec![0.0; 4 * 8];
        let b1 = vec![0.0; 8];
        let rank_w = vec![0.0; 8];
        let rw: [f64; 4] = [0.0; 4];
        let w_alpha = vec![0.0; 8];
        let (y, y_rank, y_pool, alpha, alpha_logit, _, _, _, _) = forward_per_sample_alpha_head(
            &x, &w1, &b1, &rank_w, 0.0, &rw, 0.0, &w_alpha, 0.0, 4, 8, 0.01,
        );
        assert!(y_rank.abs() < 1e-12);
        assert!(y_pool.abs() < 1e-12);
        assert!((alpha - 0.5).abs() < 1e-12);
        assert!(alpha_logit.abs() < 1e-12);
        assert!(y.abs() < 1e-12);
    }

    #[test]
    fn forward_w_alpha_drives_per_sample_alpha() {
        // w1 = identity-ish so h_pre[0] depends on x[0].
        // W_α[0] = +5 → α_logit ≈ +5·h[0] + 0 → α ≈ 1 when h[0] > 1.
        // → y ≈ y_rank.
        let n_f = 2;
        let n_h = 4;
        let mut w1 = vec![0.0; n_f * n_h];
        w1[0] = 1.0; // h_pre[0] = x[0]
        let b1 = vec![0.0; n_h];
        let mut rank_w = vec![0.0f64; n_h];
        rank_w[0] = 2.0; // y_rank ≈ 2 · h[0]
        let rw: [f64; 4] = [0.0; 4]; // y_pool ≈ 0
        let mut w_alpha = vec![0.0f64; n_h];
        w_alpha[0] = 10.0; // α_logit ≈ 10 · h[0]

        // x[0] = 1 → h[0] = 1 → α_logit = 10 → α ≈ 1 → y ≈ 2.
        let x = vec![1.0, 0.0];
        let (y, y_rank, _, alpha, _, _, _, _, _) = forward_per_sample_alpha_head(
            &x, &w1, &b1, &rank_w, 0.0, &rw, 0.0, &w_alpha, 0.0, n_f, n_h, 0.01,
        );
        assert!((alpha - 1.0).abs() < 1e-3, "alpha={alpha} expected≈1");
        assert!((y_rank - 2.0).abs() < 1e-6);
        assert!((y - 2.0).abs() < 1e-3, "y={y} expected≈2");

        // x[0] = -1 → h[0] = -0.01 (LeakyReLU) → α_logit ≈ -0.1 → α ≈ 0.475 → y ≈ 0.475·(-0.02) ≈ -0.0095.
        // Just confirm α < 0.5 (W_α path is engaging).
        let x_neg = vec![-1.0, 0.0];
        let (_, _, _, alpha_neg, _, _, _, _, _) = forward_per_sample_alpha_head(
            &x_neg, &w1, &b1, &rank_w, 0.0, &rw, 0.0, &w_alpha, 0.0, n_f, n_h, 0.01,
        );
        assert!(
            alpha_neg < 0.5,
            "alpha for x=-1 must be < 0.5: got {alpha_neg}"
        );
    }

    #[test]
    fn backprop_finite_diff_all_params() {
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
        let w_alpha: Vec<f64> = vec![0.4, -0.3, 0.2];
        let b_alpha = 0.5;

        let (y, yr, yp, a, _, hp, h, stats, max_idx) = forward_per_sample_alpha_head(
            &x,
            &w1,
            &b1,
            &rank_w,
            rank_b,
            &rw,
            rb,
            &w_alpha,
            b_alpha,
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
        let mut g_w_alpha = vec![0.0f64; n_h];
        let mut g_b_alpha = 0.0;

        backprop_step_per_sample_alpha_head(
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
            &w_alpha,
            &mut gw1,
            &mut gb1,
            &mut g_rank_w,
            &mut g_rank_b,
            &mut g_red_w,
            &mut g_red_b,
            &mut g_w_alpha,
            &mut g_b_alpha,
            n_f,
            n_h,
            alpha_leaky,
        );

        let eps = 1e-5;
        let loss = |y: f64| (y - 1.0).powi(2);
        let fwd = |w1: &[f64],
                   rank_w: &[f64],
                   rw: &[f64; 4],
                   rb: f64,
                   w_alpha: &[f64],
                   b_alpha: f64|
         -> f64 {
            let (y, _, _, _, _, _, _, _, _) = forward_per_sample_alpha_head(
                &x,
                w1,
                &b1,
                rank_w,
                rank_b,
                rw,
                rb,
                w_alpha,
                b_alpha,
                n_f,
                n_h,
                alpha_leaky,
            );
            y
        };

        // ∂L/∂w1[0]
        let mut w1_p = w1.clone();
        w1_p[0] += eps;
        let yp_p = fwd(&w1_p, &rank_w, &rw, rb, &w_alpha, b_alpha);
        let mut w1_m = w1.clone();
        w1_m[0] -= eps;
        let yp_m = fwd(&w1_m, &rank_w, &rw, rb, &w_alpha, b_alpha);
        let num_grad_w1_0 = (loss(yp_p) - loss(yp_m)) / (2.0 * eps);
        assert!(
            (gw1[0] - num_grad_w1_0).abs() < 1e-3,
            "gw1[0]={} num={num_grad_w1_0}",
            gw1[0]
        );

        // ∂L/∂rank_w[1]
        let mut rw_p = rank_w.clone();
        rw_p[1] += eps;
        let y_rw_p = fwd(&w1, &rw_p, &rw, rb, &w_alpha, b_alpha);
        let mut rw_m = rank_w.clone();
        rw_m[1] -= eps;
        let y_rw_m = fwd(&w1, &rw_m, &rw, rb, &w_alpha, b_alpha);
        let num_grad_rank1 = (loss(y_rw_p) - loss(y_rw_m)) / (2.0 * eps);
        // Tolerance 1e-3 matches the other gradient assertions in this
        // test (line 1557 etc.) and the f32 SIMD precision of the
        // consolidated encoder (analytic gradient runs through f32 SIMD;
        // numerical gradient is f64 finite-diff; expected drift ~1e-3 at
        // unit-magnitude gradients).
        assert!(
            (g_rank_w[1] - num_grad_rank1).abs() < 1e-3,
            "g_rank_w[1]={} num={num_grad_rank1}",
            g_rank_w[1]
        );

        // ∂L/∂reducer_w[1]
        let mut red_p = rw;
        red_p[1] += eps;
        let y_red_p = fwd(&w1, &rank_w, &red_p, rb, &w_alpha, b_alpha);
        let mut red_m = rw;
        red_m[1] -= eps;
        let y_red_m = fwd(&w1, &rank_w, &red_m, rb, &w_alpha, b_alpha);
        let num_grad_red1 = (loss(y_red_p) - loss(y_red_m)) / (2.0 * eps);
        assert!(
            (g_red_w[1] - num_grad_red1).abs() < 1e-4,
            "g_red_w[1]={} num={num_grad_red1}",
            g_red_w[1]
        );

        // ∂L/∂W_α[1] — the new path.
        let mut wa_p = w_alpha.clone();
        wa_p[1] += eps;
        let y_wa_p = fwd(&w1, &rank_w, &rw, rb, &wa_p, b_alpha);
        let mut wa_m = w_alpha.clone();
        wa_m[1] -= eps;
        let y_wa_m = fwd(&w1, &rank_w, &rw, rb, &wa_m, b_alpha);
        let num_grad_wa1 = (loss(y_wa_p) - loss(y_wa_m)) / (2.0 * eps);
        assert!(
            (g_w_alpha[1] - num_grad_wa1).abs() < 1e-4,
            "g_w_alpha[1]={} num={num_grad_wa1}",
            g_w_alpha[1]
        );

        // ∂L/∂b_α
        let y_ba_p = fwd(&w1, &rank_w, &rw, rb, &w_alpha, b_alpha + eps);
        let y_ba_m = fwd(&w1, &rank_w, &rw, rb, &w_alpha, b_alpha - eps);
        let num_grad_ba = (loss(y_ba_p) - loss(y_ba_m)) / (2.0 * eps);
        assert!(
            (g_b_alpha - num_grad_ba).abs() < 1e-4,
            "g_b_alpha={g_b_alpha} num={num_grad_ba}"
        );
    }

    #[test]
    fn bake_per_sample_v3_has_metadata_and_version() {
        let model = PerSampleAlphaHeadModel::new(8, 4, 7);
        let bytes = bake_per_sample_alpha_head_v3(&model);
        assert_eq!(&bytes[0..4], b"ZNPR", "expected ZNPR magic");
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 3, "expected v3 (v2 prohibited)");
        let needle = b"zentrain.per_sample_alpha_head";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(found, "expected per_sample_alpha_head metadata in bake");
        // V4: tanh metadata key must NOT appear in the non-tanh bake.
        let tanh_needle = b"zentrain.tanh_output_head";
        let tanh_found = bytes.windows(tanh_needle.len()).any(|w| w == tanh_needle);
        assert!(
            !tanh_found,
            "did not expect tanh_output_head metadata in non-tanh bake"
        );
    }

    #[test]
    fn bake_per_sample_v3_with_tanh_has_both_metadata_keys() {
        let model = PerSampleAlphaHeadModel::new(8, 4, 7);
        let bytes = bake_per_sample_alpha_head_v3_with_tanh(&model, 10.0);
        assert_eq!(&bytes[0..4], b"ZNPR", "expected ZNPR magic");
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 3, "expected v3");
        let needle = b"zentrain.per_sample_alpha_head";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(found, "expected per_sample_alpha_head metadata in bake");
        let tanh_needle = b"zentrain.tanh_output_head";
        let tanh_found = bytes.windows(tanh_needle.len()).any(|w| w == tanh_needle);
        assert!(
            tanh_found,
            "expected tanh_output_head metadata in tanh-wrapped bake"
        );
    }

    #[test]
    #[should_panic(expected = "tanh_output_head scale must be > 0")]
    fn bake_per_sample_v3_with_tanh_rejects_zero_scale() {
        let model = PerSampleAlphaHeadModel::new(8, 4, 7);
        let _ = bake_per_sample_alpha_head_v3_with_tanh(&model, 0.0);
    }

    #[test]
    fn parse_per_sample_alpha_head_meta_roundtrip() {
        // Synthetic payload of known content.
        let n_hidden = 4;
        let mut payload = Vec::new();
        for v in [0.5f32, -0.3, 0.2, 0.1] {
            payload.extend_from_slice(&v.to_le_bytes()); // W_α
        }
        payload.extend_from_slice(&0.7f32.to_le_bytes()); // b_α
        for v in [0.1f32, 0.2, 0.3, 0.4] {
            payload.extend_from_slice(&v.to_le_bytes()); // rank_w
        }
        payload.extend_from_slice(&0.5f32.to_le_bytes()); // rank_b
        for v in [0.05f32, 1.0, 0.05, 0.05] {
            payload.extend_from_slice(&v.to_le_bytes()); // reducer_w
        }
        payload.extend_from_slice(&0.0f32.to_le_bytes()); // reducer_b
        payload.extend_from_slice(&6.0f32.to_le_bytes()); // p_norm
        let meta = parse_per_sample_alpha_head_meta(&payload, n_hidden).expect("parse ok");
        assert_eq!(meta.w_alpha, vec![0.5f32, -0.3, 0.2, 0.1]);
        assert_eq!(meta.b_alpha, 0.7);
        assert_eq!(meta.rank_w, vec![0.1f32, 0.2, 0.3, 0.4]);
        assert_eq!(meta.rank_b, 0.5);
        assert_eq!(meta.reducer_w, [0.05f32, 1.0, 0.05, 0.05]);
        assert_eq!(meta.reducer_b, 0.0);
        assert_eq!(meta.p_norm, 6.0);
    }

    #[test]
    fn apply_runtime_matches_train_forward() {
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
        let w_alpha: Vec<f64> = (0..n_h).map(|i| 0.05 * (i as f64) - 0.1).collect();
        let b_alpha = -0.3;

        let (y, _, _, _, _, _, h, _, _) = forward_per_sample_alpha_head(
            &x,
            &w1,
            &b1,
            &rank_w,
            rank_b,
            &rw,
            rb,
            &w_alpha,
            b_alpha,
            n_f,
            n_h,
            alpha_leaky,
        );

        let h_f32: Vec<f32> = h.iter().map(|&v| v as f32).collect();
        let meta = PerSampleAlphaHeadMeta {
            w_alpha: w_alpha.iter().map(|&v| v as f32).collect(),
            b_alpha: b_alpha as f32,
            rank_w: rank_w.iter().map(|&v| v as f32).collect(),
            rank_b: rank_b as f32,
            reducer_w: [rw[0] as f32, rw[1] as f32, rw[2] as f32, rw[3] as f32],
            reducer_b: rb as f32,
            p_norm: POOL_P_NORM as f32,
        };
        let (y_runtime, _alpha_runtime) = apply_per_sample_alpha_head_runtime(&h_f32, &meta);
        assert!(
            (y - y_runtime).abs() < 5e-5,
            "train={y} runtime={y_runtime} diff={}",
            (y - y_runtime).abs()
        );
    }

    #[test]
    fn train_per_sample_alpha_recovers_synthetic_ranking() {
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
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 0.0,
        };
        let hp = PerSampleAlphaHeadHparams {
            n_hidden: 16,
            n_epochs: 5,
            pairs_per_epoch: 200,
            initial_lr: 5e-3,
            l2_lambda: 0.0,
            ..Default::default()
        };
        let model = train_per_sample_alpha_head(&[g], &hp, n_f);
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
                let (y, _, _, _, _, _, _, _, _) = forward_per_sample_alpha_head(
                    r,
                    &model.w1,
                    &model.b1,
                    &model.rank_w,
                    model.rank_b,
                    &model.reducer_w,
                    model.reducer_b,
                    &model.w_alpha,
                    model.b_alpha,
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
            "train_per_sample_alpha_head: Spearman {s} < 0.85"
        );
    }

    // Task #35 isolation: FD-check backprop_heads' dl_dh (the gradient that
    // flows into the encoder) against numerical dL/dh with L=y (dl_dy=1).
    // Splits the konjnd-agg 2-layer gradient failure into head-vs-encoder.
    // Small, non-saturating h so the tanh pin (absent here) / pool floors
    // don't confound the FD.
    #[test]
    fn backprop_heads_dl_dh_matches_finite_difference() {
        let nh = 8usize;
        let mut st = 0x2468_ace0_1357_9bdfu64;
        let mut nxt = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            ((st >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let h: Vec<f64> = (0..nh).map(|_| nxt() * 2.0).collect();
        let rank_w: Vec<f64> = (0..nh).map(|_| nxt()).collect();
        let rank_b = nxt();
        let reducer_w: [f64; 4] = [nxt(), nxt(), nxt(), nxt()];
        let reducer_b = nxt();
        let w_alpha: Vec<f64> = (0..nh).map(|_| nxt()).collect();
        let b_alpha = nxt();

        let fwd = |hv: &[f64]| -> f64 {
            forward_heads(
                hv, &rank_w, rank_b, &reducer_w, reducer_b, &w_alpha, b_alpha, nh,
            )
            .0
        };
        let (_, y_rank, y_pool, alpha, _, stats, max_idx) = forward_heads(
            &h, &rank_w, rank_b, &reducer_w, reducer_b, &w_alpha, b_alpha, nh,
        );
        let mut g_rank_w = vec![0.0; nh];
        let mut g_rank_b = 0.0;
        let mut g_red_w = [0.0; 4];
        let mut g_red_b = 0.0;
        let mut g_w_alpha = vec![0.0; nh];
        let mut g_b_alpha = 0.0;
        let dl_dh = backprop_heads(
            &h,
            &stats,
            max_idx,
            y_rank,
            y_pool,
            alpha,
            1.0,
            &rank_w,
            &reducer_w,
            &w_alpha,
            &mut g_rank_w,
            &mut g_rank_b,
            &mut g_red_w,
            &mut g_red_b,
            &mut g_w_alpha,
            &mut g_b_alpha,
            nh,
            0.01,
        );
        // The forward computes in f32 (dot_bias casts f64→f32). A central
        // difference of an f32-valued forward is floor-limited: the rounding
        // noise in (f₊−f₋) is ~ε_f32·|y| ≈ 1e-7, while the signal is
        // 2·ε·|∂y|, so the relative error of the numeric derivative is
        // ~1e-7/(2·ε·|∂y|). At ε=1e-6 that floor is O(1) — the original test's
        // "~2× gradient bug" was entirely this f32 floor, NOT a gradient bug.
        // ε=1e-2 pushes the floor to ~5e-5 while O(ε²) truncation stays ~1e-4;
        // 2e-3 is a safe gate above both error sources.
        let eps = 1e-2;
        for j in 0..nh {
            let mut hp = h.clone();
            hp[j] += eps;
            let mut hm = h.clone();
            hm[j] -= eps;
            let num = (fwd(&hp) - fwd(&hm)) / (2.0 * eps);
            let ana = dl_dh[j];
            // Combined atol+rtol (numpy/jax/pytorch gradcheck form) — robust for
            // near-zero gradient entries where a pure relative gate is unbounded
            // and the f32-forward FD floor (~1e-5 abs at ε=1e-2) dominates.
            let abs_err = (num - ana).abs();
            let tol = 2e-5 + 2e-3 * num.abs().max(ana.abs());
            assert!(
                abs_err < tol,
                "backprop_heads dl_dh[{j}] num={num:.8} ana={ana:.8} abs_err={abs_err:.2e} tol={tol:.2e}"
            );
        }
    }
}
