//! Dual-target multi-task head — EX-DUAL.
//!
//! Replaces the single-output scalar y = h · w + b with a dual-output
//! head: `y_quality` (per-pair rank-trained scalar) and `y_pjnd`
//! (per-source PJND regression scalar, AUXILIARY ONLY). The
//! y_pjnd column shares the encoder (W1, b1) with y_quality, so
//! the auxiliary regression forces the encoder to learn PJND-relevant
//! features without polluting the y_quality output that ships at
//! inference.
//!
//! Motivation: previous attempts (V_22-mix-LARGE+iwssim,
//! V_23-konjnd010, V_24-hybrid-NiN, V_24-per-sample-α-finetune,
//! ssim2-per-pair densification, PJND-broadcast densification)
//! all hit the same wall — a single scalar-output regression
//! cannot encode per-pair quality AND per-source PJND
//! simultaneously. The dual head decouples them: y_quality fits
//! RankNet on the 5-group mix; y_pjnd fits MSE on the
//! KonJND-PJND-broadcast group. Loss = ranknet(y_quality) +
//! λ_pjnd · mse(y_pjnd).
//!
//! ```text
//! h = LeakyReLU(W1 · x + b1)                  (shared encoder)
//! y_quality = h · w_qual + b_qual              (per-pair rank head)
//! y_pjnd    = h · w_pjnd + b_pjnd              (per-source PJND head)
//! L = Σ_(group≠konjnd_pjnd) w_g · ranknet(y_quality)
//!   + λ_pjnd · Σ_(group=konjnd_pjnd) (y_pjnd − pjnd_target)²
//! ```
//!
//! **Inference**: only `y_quality` is consumed. The bake stores the
//! standard scalar output (single final-layer linear) — y_pjnd
//! weights are training-only and DISCARDED at bake time. This
//! preserves runtime compatibility with the existing
//! `apply_mlp_scoring` path — the bake looks like any other
//! single-output v3 bake, except the encoder has been auxiliary-
//! supervised. The bake header gets a `zentrain.dual_target_head =
//! true` metadata flag (utf8) for provenance.
//!
//! Backprop math:
//! - `∂L/∂y_quality = ranknet partial (existing)`
//! - `∂L/∂y_pjnd   = 2 · λ_pjnd · (y_pjnd − target)` (when target present)
//! - `∂L/∂h[j]    += ∂L/∂y_quality · w_qual[j]
//!                  + ∂L/∂y_pjnd · w_pjnd[j]`
//! - `∂L/∂w_qual[j] = ∂L/∂y_quality · h[j]`
//! - `∂L/∂w_pjnd[j] = ∂L/∂y_pjnd · h[j]`
//!
//! Architectural cost vs single-head: +(n_hidden + 1) training-only
//! parameters. Zero inference cost (y_pjnd weights are dropped).

use crate::TrainingGroup;
use crate::adam::AdamState;
use crate::mlp::compute_scaler_from_groups;
use crate::rng::SplitMix64;
use zenpredict::{Activation, MetadataType, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

/// Full forward pass for the dual-target head.
///
/// Returns `(y_quality, y_pjnd, h_pre, h)`. `h_pre` is the
/// pre-LeakyReLU encoder activation, `h` post-LeakyReLU.
pub fn forward_dual_target_head(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w_qual: &[f64],
    b_qual: f64,
    w_pjnd: &[f64],
    b_pjnd: f64,
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f64,
) -> (f64, f64, Vec<f64>, Vec<f64>) {
    debug_assert_eq!(w_qual.len(), n_hidden);
    debug_assert_eq!(w_pjnd.len(), n_hidden);

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

    let mut y_quality = b_qual;
    let mut y_pjnd = b_pjnd;
    for j in 0..n_hidden {
        y_quality += h[j] * w_qual[j];
        y_pjnd += h[j] * w_pjnd[j];
    }
    (y_quality, y_pjnd, h_pre, h)
}

/// Backprop dL/dy_quality + dL/dy_pjnd through the dual-target
/// head into gradients on (w1, b1, w_qual, b_qual, w_pjnd, b_pjnd).
///
/// Passing `dl_dy_pjnd = 0.0` is the same as a single-head ranknet
/// step (the PJND head receives no gradient signal). This is the
/// path taken by non-PJND groups.
#[allow(clippy::too_many_arguments)]
pub fn backprop_step_dual_target_head(
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    dl_dy_quality: f64,
    dl_dy_pjnd: f64,
    w_qual: &[f64],
    w_pjnd: &[f64],
    gw1: &mut [f64],
    gb1: &mut [f64],
    g_w_qual: &mut [f64],
    g_b_qual: &mut f64,
    g_w_pjnd: &mut [f64],
    g_b_pjnd: &mut f64,
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f64,
) {
    debug_assert_eq!(w_qual.len(), n_hidden);
    debug_assert_eq!(w_pjnd.len(), n_hidden);
    debug_assert_eq!(g_w_qual.len(), n_hidden);
    debug_assert_eq!(g_w_pjnd.len(), n_hidden);

    // Output-head grads.
    for j in 0..n_hidden {
        g_w_qual[j] += dl_dy_quality * h[j];
        g_w_pjnd[j] += dl_dy_pjnd * h[j];
    }
    *g_b_qual += dl_dy_quality;
    *g_b_pjnd += dl_dy_pjnd;

    // ∂L/∂h[j] = quality contribution + pjnd contribution.
    let mut dl_dh = vec![0.0f64; n_hidden];
    for j in 0..n_hidden {
        dl_dh[j] = dl_dy_quality * w_qual[j] + dl_dy_pjnd * w_pjnd[j];
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

/// Hyperparameters for the dual-target trainer.
#[derive(Clone, Debug)]
pub struct DualTargetHeadHparams {
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
    /// L2 regularization on layer weights (and w_qual + w_pjnd).
    /// Biases unregularized.
    pub l2_lambda: f64,
    /// Auxiliary PJND-MSE loss weight. λ ∈ {0.01, 0.05, 0.1, 0.3,
    /// 1.0}. λ=0 collapses to single-head RankNet baseline.
    pub pjnd_loss_weight: f64,
}

impl Default for DualTargetHeadHparams {
    fn default() -> Self {
        Self {
            n_hidden: 128,
            n_epochs: 200,
            pairs_per_epoch: 50_000,
            initial_lr: 1e-3,
            leaky_alpha: 0.01,
            seed: 1,
            l2_lambda: 1e-5,
            pjnd_loss_weight: 0.1,
        }
    }
}

/// One named slice of training/validation data, extended with an
/// optional PJND broadcast target column. `pjnd_targets` is `None`
/// for the standard 5-group mix and `Some(per_row_value)` for the
/// KonJND-PJND-broadcast group.
#[derive(Debug)]
pub struct DualTargetGroup<'a> {
    /// Human-readable group name.
    pub name: String,
    /// Per-pair quality scores (the RankNet target).
    pub human_scores: &'a [f64],
    /// Per-pair feature vectors.
    pub features: &'a [&'a [f64]],
    /// Per-pair PJND target (broadcast from per-source PJND). `None`
    /// → this group doesn't contribute to the y_pjnd auxiliary loss.
    pub pjnd_targets: Option<&'a [f64]>,
    /// RankNet sampling weight (for y_quality).
    pub train_weight: f64,
    /// PJND sampling weight (for y_pjnd). Independent of
    /// `train_weight` so a group can contribute to one without
    /// contributing to the other. Set to 0 for non-PJND groups.
    pub pjnd_train_weight: f64,
    /// Validation weight (per-epoch SROCC on y_quality only —
    /// y_pjnd is never the validation signal, it's auxiliary).
    pub validation_weight: f64,
}

/// Trained dual-target head model.
#[derive(Debug)]
pub struct DualTargetHeadModel {
    /// Per-feature mean (n_features).
    pub scaler_mean: Vec<f64>,
    /// Per-feature std (n_features).
    pub scaler_scale: Vec<f64>,
    /// Layer-1 weights, row-major (n_features × n_hidden).
    pub w1: Vec<f64>,
    /// Layer-1 biases (n_hidden).
    pub b1: Vec<f64>,
    /// Quality-output head weights (n_hidden).
    pub w_qual: Vec<f64>,
    /// Quality-output head bias.
    pub b_qual: f64,
    /// PJND auxiliary head weights (n_hidden) — TRAINING ONLY.
    pub w_pjnd: Vec<f64>,
    /// PJND auxiliary head bias — TRAINING ONLY.
    pub b_pjnd: f64,
    /// Hidden width.
    pub n_hidden: usize,
    /// Input dim.
    pub n_features: usize,
    /// Effective auxiliary loss weight used at training time.
    /// Recorded for bake provenance (metadata) only.
    pub pjnd_loss_weight: f64,
}

impl DualTargetHeadModel {
    /// Initialize with He-normal layer-1 weights and small
    /// gaussian heads. w_pjnd is initialized small (0.01-scaled)
    /// so the y_pjnd signal doesn't dominate the encoder gradient
    /// at epoch 0 — it should grow only as the λ_pjnd · MSE loss
    /// finds a useful direction.
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
        let head_scale = 1.0 / (n_hidden as f64).sqrt();
        let mut w_qual = Vec::with_capacity(n_hidden);
        let mut w_pjnd = Vec::with_capacity(n_hidden);
        for _ in 0..n_hidden {
            let u1 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let u2 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let u1 = u1.max(1e-12);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();
            w_qual.push(z * head_scale);
        }
        for _ in 0..n_hidden {
            let u1 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let u2 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let u1 = u1.max(1e-12);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();
            // Smaller init for the PJND head — it's an auxiliary
            // task that should ramp up, not dominate at epoch 0.
            w_pjnd.push(z * head_scale * 0.1);
        }
        Self {
            scaler_mean: vec![0.0; n_features],
            scaler_scale: vec![1.0; n_features],
            w1,
            b1: vec![0.0; n_hidden],
            w_qual,
            b_qual: 0.0,
            w_pjnd,
            b_pjnd: 0.0,
            n_hidden,
            n_features,
            pjnd_loss_weight: 0.0,
        }
    }
}

/// Convert `DualTargetGroup` slice → `TrainingGroup` slice for
/// scaler computation (which only needs features, train_weight).
fn to_training_groups<'a, 'b>(
    groups: &'b [DualTargetGroup<'a>],
) -> Vec<TrainingGroup<'a>>
where
    'a: 'b,
{
    groups
        .iter()
        .map(|g| TrainingGroup {
            name: g.name.clone(),
            human_scores: g.human_scores,
            features: g.features,
            train_weight: g.train_weight,
            validation_weight: g.validation_weight,
        })
        .collect()
}

/// Standalone trainer for the dual-target head. Mirrors
/// `train_per_sample_alpha_head`: synthetic-rank smoke test.
/// Production training goes through
/// `train_mlp_dual_target_head` in `zensim-validate::mlp_train`.
pub fn train_dual_target_head(
    groups: &[DualTargetGroup<'_>],
    hparams: &DualTargetHeadHparams,
    n_features: usize,
) -> DualTargetHeadModel {
    let train_indices: Vec<usize> = (0..groups.len())
        .filter(|&i| groups[i].train_weight > 0.0)
        .collect();
    assert!(
        !train_indices.is_empty(),
        "train_dual_target_head: need at least one group with train_weight > 0"
    );

    let tgs = to_training_groups(groups);
    let (mean, scale) = compute_scaler_from_groups(&tgs, &train_indices, n_features);

    let mut model = DualTargetHeadModel::new(n_features, hparams.n_hidden, hparams.seed);
    model.scaler_mean = mean;
    model.scaler_scale = scale;
    model.pjnd_loss_weight = hparams.pjnd_loss_weight;

    // Pre-standardize features per group.
    let std_groups: Vec<Vec<Vec<f64>>> = train_indices
        .iter()
        .map(|&gi| {
            let g = &groups[gi];
            g.features
                .iter()
                .map(|row| {
                    (0..n_features)
                        .map(|d| (row[d] - model.scaler_mean[d]) / model.scaler_scale[d].max(1e-12))
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

    // PJND group sampling: separate CDF over groups with
    // pjnd_train_weight > 0 AND pjnd_targets present.
    let pjnd_indices: Vec<usize> = train_indices
        .iter()
        .enumerate()
        .filter_map(|(idx, &gi)| {
            if groups[gi].pjnd_train_weight > 0.0 && groups[gi].pjnd_targets.is_some() {
                Some(idx)
            } else {
                None
            }
        })
        .collect();
    let pjnd_total_w: f64 = pjnd_indices
        .iter()
        .map(|&idx| groups[train_indices[idx]].pjnd_train_weight)
        .sum();
    let pjnd_cdf: Vec<f64> = if pjnd_total_w > 0.0 {
        pjnd_indices
            .iter()
            .scan(0.0, |acc, &idx| {
                *acc += groups[train_indices[idx]].pjnd_train_weight / pjnd_total_w;
                Some(*acc)
            })
            .collect()
    } else {
        vec![]
    };

    let n_hidden = hparams.n_hidden;
    let nw1 = n_features * n_hidden;
    let nb1 = n_hidden;
    // Adam w2 slot: [w_qual (n_hidden) | w_pjnd (n_hidden)].
    // Adam b2 slot: [b_qual, b_pjnd].
    let n_w2 = 2 * n_hidden;
    let n_b2 = 2;
    let mut adam = AdamState::new(nw1, nb1, n_w2, n_b2);
    let mut rng = SplitMix64::new(hparams.seed ^ 0x5A5A_5A5A_5A5A_5A5A);

    let l2 = hparams.l2_lambda;
    let alpha_leaky = hparams.leaky_alpha;
    let lr = hparams.initial_lr;
    let lambda_pjnd = hparams.pjnd_loss_weight;

    for _epoch in 0..hparams.n_epochs {
        for _pair in 0..hparams.pairs_per_epoch {
            // Standard RankNet step on a y_quality-sampled group.
            let r = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let mut gi_local = 0usize;
            for (k, &c) in weight_cdf.iter().enumerate() {
                if r < c {
                    gi_local = k;
                    break;
                }
                gi_local = k;
            }
            let g_rows = &std_groups[gi_local];
            let g_scores = train_scores[gi_local];
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

            let (y_qhi, _y_phi, hp_hi, h_hi) = forward_dual_target_head(
                xhi,
                &model.w1,
                &model.b1,
                &model.w_qual,
                model.b_qual,
                &model.w_pjnd,
                model.b_pjnd,
                n_features,
                n_hidden,
                alpha_leaky,
            );
            let (y_qlo, _y_plo, hp_lo, h_lo) = forward_dual_target_head(
                xlo,
                &model.w1,
                &model.b1,
                &model.w_qual,
                model.b_qual,
                &model.w_pjnd,
                model.b_pjnd,
                n_features,
                n_hidden,
                alpha_leaky,
            );
            // RankNet on y_quality.
            let d = y_qhi - y_qlo;
            let sig_neg_d = 1.0 / (1.0 + d.exp());
            let dl_dd = -sig_neg_d;
            let dl_dyq_hi = dl_dd;
            let dl_dyq_lo = -dl_dd;

            let mut g_w_qual_buf = vec![0.0f64; n_hidden];
            let mut g_w_pjnd_buf = vec![0.0f64; n_hidden];
            let mut g_b_qual: f64 = 0.0;
            let mut g_b_pjnd: f64 = 0.0;

            backprop_step_dual_target_head(
                xhi,
                &hp_hi,
                &h_hi,
                dl_dyq_hi,
                0.0, // no PJND signal on this group's pair
                &model.w_qual,
                &model.w_pjnd,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_w_qual_buf,
                &mut g_b_qual,
                &mut g_w_pjnd_buf,
                &mut g_b_pjnd,
                n_features,
                n_hidden,
                alpha_leaky,
            );
            backprop_step_dual_target_head(
                xlo,
                &hp_lo,
                &h_lo,
                dl_dyq_lo,
                0.0,
                &model.w_qual,
                &model.w_pjnd,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_w_qual_buf,
                &mut g_b_qual,
                &mut g_w_pjnd_buf,
                &mut g_b_pjnd,
                n_features,
                n_hidden,
                alpha_leaky,
            );

            // PJND auxiliary step: sample a PJND-bearing pair (one
            // sample, not a pair — MSE is a per-sample loss).
            if lambda_pjnd > 0.0 && !pjnd_cdf.is_empty() {
                let r2 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
                let mut pjnd_local = 0usize;
                for (k, &c) in pjnd_cdf.iter().enumerate() {
                    if r2 < c {
                        pjnd_local = k;
                        break;
                    }
                    pjnd_local = k;
                }
                let group_local_idx = pjnd_indices[pjnd_local];
                let pj_rows = &std_groups[group_local_idx];
                let gi_outer = train_indices[group_local_idx];
                let pj_targets = groups[gi_outer]
                    .pjnd_targets
                    .expect("pjnd group must have pjnd_targets");
                let pj_len = pj_rows.len();
                if pj_len >= 1 {
                    let ip = (rng.next_u64() as usize) % pj_len;
                    let xp = &pj_rows[ip];
                    let target_pjnd = pj_targets[ip];

                    let (_y_qp, y_pp, hp_p, h_p) = forward_dual_target_head(
                        xp,
                        &model.w1,
                        &model.b1,
                        &model.w_qual,
                        model.b_qual,
                        &model.w_pjnd,
                        model.b_pjnd,
                        n_features,
                        n_hidden,
                        alpha_leaky,
                    );
                    // L_pjnd = λ · (y_pjnd − target)²
                    // dL/dy_pjnd = 2 · λ · (y_pjnd − target)
                    let dl_dyp = 2.0 * lambda_pjnd * (y_pp - target_pjnd);

                    backprop_step_dual_target_head(
                        xp,
                        &hp_p,
                        &h_p,
                        0.0, // no quality signal on this step
                        dl_dyp,
                        &model.w_qual,
                        &model.w_pjnd,
                        &mut adam.gw1,
                        &mut adam.gb1,
                        &mut g_w_qual_buf,
                        &mut g_b_qual,
                        &mut g_w_pjnd_buf,
                        &mut g_b_pjnd,
                        n_features,
                        n_hidden,
                        alpha_leaky,
                    );
                }
            }

            if l2 > 0.0 {
                for (g, &w) in adam.gw1.iter_mut().zip(model.w1.iter()) {
                    *g += 2.0 * l2 * w;
                }
                for j in 0..n_hidden {
                    g_w_qual_buf[j] += 2.0 * l2 * model.w_qual[j];
                    g_w_pjnd_buf[j] += 2.0 * l2 * model.w_pjnd[j];
                }
            }

            // Pack into Adam w2/b2 slots.
            for j in 0..n_hidden {
                adam.gw2[j] += g_w_qual_buf[j];
                adam.gw2[n_hidden + j] += g_w_pjnd_buf[j];
            }
            adam.gb2[0] += g_b_qual;
            adam.gb2[1] += g_b_pjnd;

            // Adam step: pack/unpack into w2.
            let mut w2_vec = vec![0.0f64; n_w2];
            for j in 0..n_hidden {
                w2_vec[j] = model.w_qual[j];
                w2_vec[n_hidden + j] = model.w_pjnd[j];
            }
            let mut b2_vec = vec![model.b_qual, model.b_pjnd];
            adam.step(&mut model.w1, &mut model.b1, &mut w2_vec, &mut b2_vec, lr);
            for j in 0..n_hidden {
                model.w_qual[j] = w2_vec[j];
                model.w_pjnd[j] = w2_vec[n_hidden + j];
            }
            model.b_qual = b2_vec[0];
            model.b_pjnd = b2_vec[1];
        }
    }
    model
}

/// Bake the dual-target head model into ZNPR v3 bytes. The bake
/// is structurally **single-output**: the y_pjnd auxiliary head
/// is TRAINING-ONLY and dropped. Inference produces y_quality
/// alone through the standard `apply_mlp_scoring` path.
///
/// Wire format: standard ZNPR v3 with two layers:
/// - Layer 1: n_features → n_hidden, LeakyReLU, F32 weights.
/// - Layer 2: n_hidden → 1, Identity, F32 weights = w_qual.
///
/// Metadata:
/// - `zentrain.dual_target_head` = "true" (utf8 flag for provenance)
/// - `zentrain.dual_target_pjnd_loss_weight` = `[λ]` (numeric, 1×f32)
/// - `zentrain.dual_target_pjnd_head` = w_pjnd ++ b_pjnd as numeric
///   payload (n_hidden+1 × f32). This is OPTIONAL diagnostic
///   metadata — runtime ignores it. Reserved for offline analysis
///   (e.g., "did the PJND head learn anything useful?").
pub fn bake_dual_target_head_v3(model: &DualTargetHeadModel) -> Vec<u8> {
    let n_features = model.n_features;
    let n_hidden = model.n_hidden;

    let scaler_mean_f32: Vec<f32> = model.scaler_mean.iter().map(|&v| v as f32).collect();
    let scaler_scale_f32: Vec<f32> = model.scaler_scale.iter().map(|&v| v as f32).collect();
    let w1_f32: Vec<f32> = model.w1.iter().map(|&v| v as f32).collect();
    let b1_f32: Vec<f32> = model.b1.iter().map(|&v| v as f32).collect();
    // Layer 2: hidden → 1 (only the quality head).
    let w2_f32: Vec<f32> = model.w_qual.iter().map(|&v| v as f32).collect();
    let b2_f32 = vec![model.b_qual as f32];

    let lambda_payload = (model.pjnd_loss_weight as f32).to_le_bytes().to_vec();
    let mut pjnd_payload = Vec::with_capacity((n_hidden + 1) * 4);
    for v in &model.w_pjnd {
        pjnd_payload.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    pjnd_payload.extend_from_slice(&(model.b_pjnd as f32).to_le_bytes());

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
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w2_f32,
            biases: &b2_f32,
        },
    ];
    let metadata = [
        BakeMetadataEntry {
            key: "zentrain.dual_target_head",
            kind: MetadataType::Utf8,
            value: b"true",
        },
        BakeMetadataEntry {
            key: "zentrain.dual_target_pjnd_loss_weight",
            kind: MetadataType::Numeric,
            value: &lambda_payload,
        },
        BakeMetadataEntry {
            key: "zentrain.dual_target_pjnd_head",
            kind: MetadataType::Numeric,
            value: &pjnd_payload,
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
    .expect("v3 bake of dual-target head MLP")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_zero_weights_returns_zero() {
        let x = vec![1.0; 4];
        let w1 = vec![0.0; 4 * 8];
        let b1 = vec![0.0; 8];
        let w_qual = vec![0.0; 8];
        let w_pjnd = vec![0.0; 8];
        let (yq, yp, _, _) =
            forward_dual_target_head(&x, &w1, &b1, &w_qual, 0.0, &w_pjnd, 0.0, 4, 8, 0.01);
        assert!(yq.abs() < 1e-12);
        assert!(yp.abs() < 1e-12);
    }

    #[test]
    fn forward_bias_only() {
        let x = vec![0.0; 4];
        let w1 = vec![0.0; 4 * 8];
        let b1 = vec![0.5; 8];
        let w_qual = vec![1.0; 8];
        let w_pjnd = vec![0.5; 8];
        let (yq, yp, _, _) =
            forward_dual_target_head(&x, &w1, &b1, &w_qual, 0.1, &w_pjnd, 0.2, 4, 8, 0.01);
        // h_pre = b1 = 0.5; h = 0.5 (positive); y_q = 0.5·1.0·8 + 0.1 = 4.1
        // y_p = 0.5·0.5·8 + 0.2 = 2.2
        assert!((yq - 4.1).abs() < 1e-10, "y_q={yq}");
        assert!((yp - 2.2).abs() < 1e-10, "y_p={yp}");
    }

    #[test]
    fn backprop_finite_diff_quality_path() {
        // Loss = (y_quality - 1)² (no PJND term).
        let n_f = 2;
        let n_h = 3;
        let alpha_leaky = 0.01;
        let x = vec![0.7f64, -0.3];
        let w1: Vec<f64> = vec![0.1, -0.2, 0.3, -0.4, 0.5, 0.6];
        let b1 = vec![0.05, -0.05, 0.1];
        let w_qual: Vec<f64> = vec![0.2, -0.1, 0.05];
        let b_qual = 0.02;
        let w_pjnd: Vec<f64> = vec![0.4, -0.3, 0.2];
        let b_pjnd = 0.5;

        let (yq, _yp, hp, h) = forward_dual_target_head(
            &x, &w1, &b1, &w_qual, b_qual, &w_pjnd, b_pjnd, n_f, n_h, alpha_leaky,
        );
        let dl_dyq = 2.0 * (yq - 1.0);
        let dl_dyp = 0.0;

        let mut gw1 = vec![0.0; w1.len()];
        let mut gb1 = vec![0.0; b1.len()];
        let mut g_w_qual = vec![0.0; n_h];
        let mut g_b_qual = 0.0;
        let mut g_w_pjnd = vec![0.0; n_h];
        let mut g_b_pjnd = 0.0;

        backprop_step_dual_target_head(
            &x,
            &hp,
            &h,
            dl_dyq,
            dl_dyp,
            &w_qual,
            &w_pjnd,
            &mut gw1,
            &mut gb1,
            &mut g_w_qual,
            &mut g_b_qual,
            &mut g_w_pjnd,
            &mut g_b_pjnd,
            n_f,
            n_h,
            alpha_leaky,
        );

        let eps = 1e-5;
        let loss = |yq: f64| (yq - 1.0).powi(2);
        let fwd = |w1: &[f64], w_qual: &[f64], b_qual: f64, w_pjnd: &[f64], b_pjnd: f64| -> f64 {
            let (yq, _, _, _) = forward_dual_target_head(
                &x, w1, &b1, w_qual, b_qual, w_pjnd, b_pjnd, n_f, n_h, alpha_leaky,
            );
            yq
        };

        // ∂L/∂w1[0]
        let mut w1_p = w1.clone();
        w1_p[0] += eps;
        let mut w1_m = w1.clone();
        w1_m[0] -= eps;
        let num = (loss(fwd(&w1_p, &w_qual, b_qual, &w_pjnd, b_pjnd))
            - loss(fwd(&w1_m, &w_qual, b_qual, &w_pjnd, b_pjnd)))
            / (2.0 * eps);
        assert!((gw1[0] - num).abs() < 1e-4, "gw1[0]={} num={num}", gw1[0]);

        // ∂L/∂w_qual[1]
        let mut wq_p = w_qual.clone();
        wq_p[1] += eps;
        let mut wq_m = w_qual.clone();
        wq_m[1] -= eps;
        let num = (loss(fwd(&w1, &wq_p, b_qual, &w_pjnd, b_pjnd))
            - loss(fwd(&w1, &wq_m, b_qual, &w_pjnd, b_pjnd)))
            / (2.0 * eps);
        assert!(
            (g_w_qual[1] - num).abs() < 1e-4,
            "g_w_qual[1]={} num={num}",
            g_w_qual[1]
        );

        // ∂L/∂b_qual
        let num = (loss(fwd(&w1, &w_qual, b_qual + eps, &w_pjnd, b_pjnd))
            - loss(fwd(&w1, &w_qual, b_qual - eps, &w_pjnd, b_pjnd)))
            / (2.0 * eps);
        assert!(
            (g_b_qual - num).abs() < 1e-4,
            "g_b_qual={g_b_qual} num={num}"
        );

        // ∂L/∂w_pjnd[1] — should be exactly 0 (no PJND signal).
        assert!(g_w_pjnd[1].abs() < 1e-12, "g_w_pjnd[1]={}", g_w_pjnd[1]);
        assert!(g_b_pjnd.abs() < 1e-12, "g_b_pjnd={g_b_pjnd}");
    }

    #[test]
    fn backprop_finite_diff_pjnd_path() {
        // Loss = (y_pjnd - 60)² (no quality term — pure PJND path).
        let n_f = 2;
        let n_h = 3;
        let alpha_leaky = 0.01;
        let x = vec![0.7f64, -0.3];
        let w1: Vec<f64> = vec![0.1, -0.2, 0.3, -0.4, 0.5, 0.6];
        let b1 = vec![0.05, -0.05, 0.1];
        let w_qual: Vec<f64> = vec![0.2, -0.1, 0.05];
        let b_qual = 0.02;
        let w_pjnd: Vec<f64> = vec![0.4, -0.3, 0.2];
        let b_pjnd = 0.5;
        let target = 60.0;

        let (_yq, yp, hp, h) = forward_dual_target_head(
            &x, &w1, &b1, &w_qual, b_qual, &w_pjnd, b_pjnd, n_f, n_h, alpha_leaky,
        );
        let dl_dyq = 0.0;
        let dl_dyp = 2.0 * (yp - target);

        let mut gw1 = vec![0.0; w1.len()];
        let mut gb1 = vec![0.0; b1.len()];
        let mut g_w_qual = vec![0.0; n_h];
        let mut g_b_qual = 0.0;
        let mut g_w_pjnd = vec![0.0; n_h];
        let mut g_b_pjnd = 0.0;

        backprop_step_dual_target_head(
            &x,
            &hp,
            &h,
            dl_dyq,
            dl_dyp,
            &w_qual,
            &w_pjnd,
            &mut gw1,
            &mut gb1,
            &mut g_w_qual,
            &mut g_b_qual,
            &mut g_w_pjnd,
            &mut g_b_pjnd,
            n_f,
            n_h,
            alpha_leaky,
        );

        let eps = 1e-5;
        let loss = |yp: f64| (yp - target).powi(2);
        let fwd = |w1: &[f64], w_pjnd: &[f64], b_pjnd: f64| -> f64 {
            let (_, yp, _, _) = forward_dual_target_head(
                &x, w1, &b1, &w_qual, b_qual, w_pjnd, b_pjnd, n_f, n_h, alpha_leaky,
            );
            yp
        };

        // ∂L/∂w1[2] — encoder gets gradient through both heads,
        // but here only PJND contributes.
        let mut w1_p = w1.clone();
        w1_p[2] += eps;
        let mut w1_m = w1.clone();
        w1_m[2] -= eps;
        let num =
            (loss(fwd(&w1_p, &w_pjnd, b_pjnd)) - loss(fwd(&w1_m, &w_pjnd, b_pjnd))) / (2.0 * eps);
        assert!((gw1[2] - num).abs() < 1e-4, "gw1[2]={} num={num}", gw1[2]);

        // ∂L/∂w_pjnd[0]
        let mut wp_p = w_pjnd.clone();
        wp_p[0] += eps;
        let mut wp_m = w_pjnd.clone();
        wp_m[0] -= eps;
        let num = (loss(fwd(&w1, &wp_p, b_pjnd)) - loss(fwd(&w1, &wp_m, b_pjnd))) / (2.0 * eps);
        assert!(
            (g_w_pjnd[0] - num).abs() < 1e-4,
            "g_w_pjnd[0]={} num={num}",
            g_w_pjnd[0]
        );

        // ∂L/∂b_pjnd
        let num = (loss(fwd(&w1, &w_pjnd, b_pjnd + eps)) - loss(fwd(&w1, &w_pjnd, b_pjnd - eps)))
            / (2.0 * eps);
        assert!(
            (g_b_pjnd - num).abs() < 1e-4,
            "g_b_pjnd={g_b_pjnd} num={num}"
        );

        // ∂L/∂w_qual[1] — should be exactly 0 (no quality signal).
        assert!(g_w_qual[1].abs() < 1e-12, "g_w_qual[1]={}", g_w_qual[1]);
        assert!(g_b_qual.abs() < 1e-12, "g_b_qual={g_b_qual}");
    }

    #[test]
    fn backprop_finite_diff_combined_paths() {
        // Loss = (y_quality - 1)² + 0.3 · (y_pjnd - 60)²
        let n_f = 2;
        let n_h = 3;
        let alpha_leaky = 0.01;
        let x = vec![0.7f64, -0.3];
        let w1: Vec<f64> = vec![0.1, -0.2, 0.3, -0.4, 0.5, 0.6];
        let b1 = vec![0.05, -0.05, 0.1];
        let w_qual: Vec<f64> = vec![0.2, -0.1, 0.05];
        let b_qual = 0.02;
        let w_pjnd: Vec<f64> = vec![0.4, -0.3, 0.2];
        let b_pjnd = 0.5;
        let target = 60.0;
        let lambda = 0.3;

        let (yq, yp, hp, h) = forward_dual_target_head(
            &x, &w1, &b1, &w_qual, b_qual, &w_pjnd, b_pjnd, n_f, n_h, alpha_leaky,
        );
        let dl_dyq = 2.0 * (yq - 1.0);
        let dl_dyp = 2.0 * lambda * (yp - target);

        let mut gw1 = vec![0.0; w1.len()];
        let mut gb1 = vec![0.0; b1.len()];
        let mut g_w_qual = vec![0.0; n_h];
        let mut g_b_qual = 0.0;
        let mut g_w_pjnd = vec![0.0; n_h];
        let mut g_b_pjnd = 0.0;

        backprop_step_dual_target_head(
            &x,
            &hp,
            &h,
            dl_dyq,
            dl_dyp,
            &w_qual,
            &w_pjnd,
            &mut gw1,
            &mut gb1,
            &mut g_w_qual,
            &mut g_b_qual,
            &mut g_w_pjnd,
            &mut g_b_pjnd,
            n_f,
            n_h,
            alpha_leaky,
        );

        let eps = 1e-5;
        let loss = |yq: f64, yp: f64| (yq - 1.0).powi(2) + lambda * (yp - target).powi(2);
        let fwd = |w1: &[f64]| -> (f64, f64) {
            let (yq, yp, _, _) = forward_dual_target_head(
                &x, w1, &b1, &w_qual, b_qual, &w_pjnd, b_pjnd, n_f, n_h, alpha_leaky,
            );
            (yq, yp)
        };

        // ∂L/∂w1[3] — encoder gets contributions from both heads.
        let mut w1_p = w1.clone();
        w1_p[3] += eps;
        let mut w1_m = w1.clone();
        w1_m[3] -= eps;
        let (yqp, ypp) = fwd(&w1_p);
        let (yqm, ypm) = fwd(&w1_m);
        let num = (loss(yqp, ypp) - loss(yqm, ypm)) / (2.0 * eps);
        assert!((gw1[3] - num).abs() < 1e-4, "gw1[3]={} num={num}", gw1[3]);
    }

    #[test]
    fn bake_dual_target_v3_has_metadata_and_version() {
        let model = DualTargetHeadModel::new(8, 4, 7);
        let bytes = bake_dual_target_head_v3(&model);
        assert_eq!(&bytes[0..4], b"ZNPR", "expected ZNPR magic");
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 3, "expected v3 (v2 prohibited)");
        let needle = b"zentrain.dual_target_head";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(found, "expected dual_target_head metadata in bake");
        let pjnd_needle = b"zentrain.dual_target_pjnd_head";
        let pj_found = bytes.windows(pjnd_needle.len()).any(|w| w == pjnd_needle);
        assert!(pj_found, "expected dual_target_pjnd_head metadata in bake");
    }

    #[test]
    fn train_dual_target_recovers_synthetic_ranking_no_pjnd() {
        // λ_pjnd = 0 → behaves like single-head RankNet.
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
        let g = DualTargetGroup {
            name: "synth".into(),
            human_scores: &scores,
            features: &row_refs,
            pjnd_targets: None,
            train_weight: 1.0,
            pjnd_train_weight: 0.0,
            validation_weight: 0.0,
        };
        let hp = DualTargetHeadHparams {
            n_hidden: 16,
            n_epochs: 5,
            pairs_per_epoch: 200,
            initial_lr: 5e-3,
            l2_lambda: 0.0,
            pjnd_loss_weight: 0.0,
            ..Default::default()
        };
        let model = train_dual_target_head(&[g], &hp, n_f);
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
                let (yq, _, _, _) = forward_dual_target_head(
                    r,
                    &model.w1,
                    &model.b1,
                    &model.w_qual,
                    model.b_qual,
                    &model.w_pjnd,
                    model.b_pjnd,
                    n_f,
                    model.n_hidden,
                    hp.leaky_alpha,
                );
                yq
            })
            .collect();
        let s = crate::spearman(&preds, &scores);
        assert!(
            s.abs() > 0.85,
            "train_dual_target_head (λ=0): Spearman {s} < 0.85"
        );
    }

    #[test]
    fn train_dual_target_with_pjnd_aux_learns_both() {
        // Two groups: a ranking group (f0 → score) and a PJND
        // group (different sources, target = 60 constant for some
        // and 30 for others). The aux head should learn to predict
        // PJND from features when given the chance.
        let n_f = 3;
        let n_pairs_rank = 30;
        let mut rank_rows: Vec<Vec<f64>> = Vec::new();
        let mut rank_scores: Vec<f64> = Vec::new();
        let mut rng = SplitMix64::new(456);
        for i in 0..n_pairs_rank {
            let t = i as f64 / (n_pairs_rank - 1) as f64;
            let f0 = t;
            let f1 = 0.0;
            let f2 = 0.0;
            rank_rows.push(vec![f0, f1, f2]);
            rank_scores.push(t * 100.0);
        }
        // PJND group: f1 = source-1 indicator (target 60), f2 =
        // source-2 indicator (target 30).
        let mut pj_rows: Vec<Vec<f64>> = Vec::new();
        let mut pj_scores: Vec<f64> = Vec::new();
        let mut pj_pjnd: Vec<f64> = Vec::new();
        for _ in 0..20 {
            pj_rows.push(vec![0.5, 1.0, 0.0]);
            pj_scores.push(50.0);
            pj_pjnd.push(60.0);
            pj_rows.push(vec![0.5, 0.0, 1.0]);
            pj_scores.push(50.0);
            pj_pjnd.push(30.0);
        }
        let _ = &mut rng;

        let rank_refs: Vec<&[f64]> = rank_rows.iter().map(|v| v.as_slice()).collect();
        let pj_refs: Vec<&[f64]> = pj_rows.iter().map(|v| v.as_slice()).collect();
        let groups = vec![
            DualTargetGroup {
                name: "rank".into(),
                human_scores: &rank_scores,
                features: &rank_refs,
                pjnd_targets: None,
                train_weight: 1.0,
                pjnd_train_weight: 0.0,
                validation_weight: 0.0,
            },
            DualTargetGroup {
                name: "pjnd".into(),
                human_scores: &pj_scores,
                features: &pj_refs,
                pjnd_targets: Some(&pj_pjnd),
                train_weight: 0.0,
                pjnd_train_weight: 1.0,
                validation_weight: 0.0,
            },
        ];
        let hp = DualTargetHeadHparams {
            n_hidden: 16,
            n_epochs: 8,
            pairs_per_epoch: 400,
            initial_lr: 5e-3,
            l2_lambda: 0.0,
            pjnd_loss_weight: 0.3,
            ..Default::default()
        };
        let model = train_dual_target_head(&groups, &hp, n_f);

        // Quality output should still rank-order rank rows.
        let std_rank: Vec<Vec<f64>> = rank_rows
            .iter()
            .map(|r| {
                (0..n_f)
                    .map(|d| (r[d] - model.scaler_mean[d]) / model.scaler_scale[d])
                    .collect()
            })
            .collect();
        let preds_q: Vec<f64> = std_rank
            .iter()
            .map(|r| {
                let (yq, _, _, _) = forward_dual_target_head(
                    r,
                    &model.w1,
                    &model.b1,
                    &model.w_qual,
                    model.b_qual,
                    &model.w_pjnd,
                    model.b_pjnd,
                    n_f,
                    model.n_hidden,
                    hp.leaky_alpha,
                );
                yq
            })
            .collect();
        let s = crate::spearman(&preds_q, &rank_scores);
        assert!(s.abs() > 0.7, "dual-target rank Spearman {s} < 0.7");

        // PJND head should distinguish source 1 vs source 2.
        let s1 = {
            let r = (0..n_f)
                .map(|d| (pj_rows[0][d] - model.scaler_mean[d]) / model.scaler_scale[d])
                .collect::<Vec<_>>();
            let (_, yp, _, _) = forward_dual_target_head(
                &r,
                &model.w1,
                &model.b1,
                &model.w_qual,
                model.b_qual,
                &model.w_pjnd,
                model.b_pjnd,
                n_f,
                model.n_hidden,
                hp.leaky_alpha,
            );
            yp
        };
        let s2 = {
            let r = (0..n_f)
                .map(|d| (pj_rows[1][d] - model.scaler_mean[d]) / model.scaler_scale[d])
                .collect::<Vec<_>>();
            let (_, yp, _, _) = forward_dual_target_head(
                &r,
                &model.w1,
                &model.b1,
                &model.w_qual,
                model.b_qual,
                &model.w_pjnd,
                model.b_pjnd,
                n_f,
                model.n_hidden,
                hp.leaky_alpha,
            );
            yp
        };
        // s1 → target 60, s2 → target 30; s1 should be higher.
        assert!(
            s1 > s2,
            "PJND aux head failed to separate source 1 (target 60) vs source 2 (target 30): s1={s1} s2={s2}"
        );
    }
}
