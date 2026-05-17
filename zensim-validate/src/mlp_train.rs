//! Two-layer MLP trainer for V0_4.
//!
//! Architecture: `n_features → n_hidden (LeakyReLU) → 1 (Identity)`.
//! Loss: RankNet pairwise (sigmoid cross-entropy on signed distance
//! deltas). Optimizer: Adam with cosine annealing.
//!
//! Output: a ZNPR v3 byte stream that loads via
//! `zensim::mlp::Model::from_bytes` (the `zensim::mlp` module re-exports
//! `zenpredict::Model`).
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

use rayon::prelude::*;
use std::time::Instant;
use zenpredict::FeatureTransform;
use zenpredict::{Activation, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

// Adam SIMD kernel — replaces the per-element scalar loop in
// `AdamState::step` with an AVX-512 (f64x8) / AVX2 (f64x4) / NEON / WASM
// / scalar dispatched kernel. Math is bit-identical to the scalar
// reference (hardware `sqrt`, FMA fusion only where LLVM would already
// fuse the scalar version).
#[path = "adam_simd.rs"]
mod adam_simd;

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
    /// Cycle-9 row-weight boost for B0 + B1 (low-quality) rows.
    /// When > 1.0, biases per-step pair sampling within each training
    /// group toward rows whose `human_score` (0-100 scale) is below 50
    /// (full boost) or in [50, 65) (sqrt(boost)). Default 1.0 = no-op,
    /// uniform within-group sampling. Composes multiplicatively with
    /// `mid_q_boost`. The original Python-side experiments (cycle-9,
    /// zensim 4b998258) tested 1.5x and 3x; 1.5x was the sweet spot
    /// for B0 SROCC gains without B2 regression.
    pub low_q_boost: f64,
    /// Cycle-12 row-weight boost for B1 + B2 (medium-quality) rows.
    /// When > 1.0, biases per-step pair sampling toward rows whose
    /// `human_score` is in [50, 90). Default 1.0 = no-op. The Python-
    /// side cycle-12 finding (zensim 4da7d1fa) at boost=1.5 was a
    /// σ-tightener (4x tighter seed-to-seed CID22 variance) with small
    /// +0.003 mean lift — useful for downstream codec orchestrators
    /// that need stable per-image ranking.
    pub mid_q_boost: f64,
    /// V0_20a-era row-weight boost for B3 (visually-lossless, human_score
    /// ≥ 90) rows. When > 1.0, biases per-step pair sampling toward
    /// rows in the visually-lossless tail. Default 1.0 = no-op.
    ///
    /// **Motivation**: with val_policy=min selecting on KADID/TID/KonJND
    /// (none of which span B3 well), the trainer structurally underfits
    /// the visually-lossless tail. The B3 band is exactly where IW-SSIM-
    /// style features (Wang & Li 2011, Mohammadi 2025) carry signal —
    /// `--high-q-boost` ensures the trainer gets enough B3 supervision
    /// for the MLP to learn the relevant weights. Recommended values:
    /// 2.0-4.0 depending on B3 sample density. Composes multiplicatively
    /// with `mid_q_boost` (boundary at 90 means it's mutually exclusive
    /// with `mid_q_boost`'s [50, 90) range).
    pub high_q_boost: f64,
    /// Output weight dtype for baked ZNPR v3. Default F32 matches every
    /// shipped bake through V0_17. I8 saves ~74 % bin size with no
    /// measurable SROCC change (verified on V0_18 across KADID, TID,
    /// CID22, AIC-3, AIC-4, KonJND). F16 saves ~50 % with bit-identical
    /// SROCC. Dequantization is inline in `zenpredict::saxpy_matmul_*`.
    pub out_dtype: WeightDtype,

    /// V0_20 input-shaping research (2026-05-14): optional per-feature
    /// transforms applied BEFORE the scaler in the runtime. When set,
    /// the bake's `zentrain.feature_transforms` metadata records the
    /// list; the runtime applies them via
    /// [`zenpredict::Predictor::predict_transformed`].
    ///
    /// Length MUST equal `n_features` when present. Caller is
    /// responsible for applying the same transforms to feature_rows
    /// before constructing `TrainingGroup` (since the trainer never
    /// sees the raw rows separately — the scaler is fit to whatever
    /// the trainer receives). An all-`Identity` vector emits no
    /// metadata (consumers treat absence as all-identity).
    pub feature_transforms: Option<Vec<FeatureTransform>>,

    /// V0_20 parameterized feature-transform params (added 2026-05-15).
    /// Parallel to `feature_transforms`; each inner `Vec<f32>` is the
    /// param list for that feature (empty for non-parameterized
    /// variants). When `Some(_)` AND any inner vec is non-empty,
    /// `zentrain.feature_transform_params` metadata is emitted in the
    /// bake. The runtime reads it via `Model::feature_transform_params()`
    /// and applies via `FeatureTransform::apply_with_params`.
    pub feature_transform_params: Option<Vec<Vec<f32>>>,

    /// T8.1 (2026-05-16): mini-batch SGD. When > 1, the trainer
    /// accumulates K RankNet pair-gradient contributions between each
    /// Adam step instead of stepping per-pair. Default 1 = per-pair
    /// SGD (bit-identical to legacy behavior).
    ///
    /// **Behavior change**: Adam's internal `t` counter increments K×
    /// less often, so bias correction `(1 - β^t)` decays K× slower —
    /// mathematically still correct, just on a different schedule. A
    /// final-flush Adam step at epoch end handles leftover accumulated
    /// gradients when `pairs_per_epoch % K != 0`, so gradients don't
    /// carry between epochs.
    ///
    /// **Convergence**: K > 1 produces less noisy gradients than
    /// per-pair SGD. Usually helps generalization; can hurt
    /// regularization on small datasets. Recommended sweep: K ∈ {1, 8,
    /// 64, 256} on a representative recipe.
    ///
    /// **Determinism**: the sample sequence (which (group, ia, ib)
    /// tuple is drawn at each step) is bit-identical regardless of K
    /// — only the Adam update cadence changes. Bake bytes for
    /// `train_mlp(K=1)` are identical to the legacy trainer; bake
    /// bytes for `train_mlp(K=K)` are deterministic given seed alone.
    pub minibatch_size: usize,

    /// T8.2 (2026-05-16): enable rayon parallel-batch within each
    /// mini-batch. When `true` AND `minibatch_size > 1`, the K
    /// per-pair gradient contributions are computed concurrently via
    /// `rayon::par_iter` with per-thread accumulators that reduce
    /// into a single LocalGrads before the Adam step.
    ///
    /// **Default `false`** so the trainer remains bit-identical to
    /// the legacy sequential code path unless explicitly opted in.
    ///
    /// **Determinism is preserved**: the sample-drawing sequence is
    /// run sequentially on the main RNG to produce a `Vec` of K
    /// (group_idx, ia, ib) tuples, and only forward+backward are
    /// parallelized. Same `seed` + same `minibatch_size` produces
    /// bit-identical bake bytes regardless of thread count.
    ///
    /// When `minibatch_size == 1` this flag is ignored (the
    /// sequential path is always taken — no per-batch overhead).
    pub parallel_batch: bool,
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
            low_q_boost: 1.0,
            mid_q_boost: 1.0,
            high_q_boost: 1.0,
            out_dtype: WeightDtype::F32,
            feature_transforms: None,
            feature_transform_params: None,
            minibatch_size: 1,
            parallel_batch: false,
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
    train_mlp_with_tv(groups, n_features, hyperparams, log, None)
}

/// TV (total-variation) regularizer for adjacent-q monotonicity.
///
/// `pairs[k] = (lo_idx, hi_idx)` references rows in the concatenated
/// trainer-feature space (group 0 rows first, then group 1, etc.).
/// Penalty per pair = `max(0, pred[hi_idx] - pred[lo_idx])` — Rust
/// trainer outputs are distance-like (lower = better quality), so a
/// monotone curve has `pred[lo_q] > pred[hi_q]`. Violations have
/// `pred[hi_q] > pred[lo_q]`.
pub struct TvRegularizer {
    pub pairs: Vec<(usize, usize)>,
    pub features: Vec<Vec<f64>>,
    pub weight: f64,
    /// Apply TV update every N RankNet pair updates. 50 is a good
    /// default — 50,000 / 50 = 1000 TV steps per epoch.
    pub apply_every: usize,
    /// Mini-batch size of TV pairs per update. 32 is fine.
    pub batch: usize,
    /// Per-pair band id (0..3 → B0/B1/B2/B3 per CID22 paper Table 5).
    /// When `Some`, must have the same length as `pairs`. The TV
    /// gradient for pair `k` is scaled by `band_weights[band_id[k]]`
    /// (if `band_weights` is also set) instead of the flat `weight`.
    pub band_id: Option<Vec<u8>>,
    /// Per-band TV weights `[B0, B1, B2, B3]`. When `Some`, must be
    /// paired with `band_id`. Used in place of `weight` for per-pair
    /// scaling. Set to e.g. `[10.0, 20.0, 10.0, 10.0]` to push B1
    /// harder than other bands.
    pub band_weights: Option<[f64; 4]>,
}

impl TvRegularizer {
    fn n_features_check(&self) -> usize {
        self.features.first().map(|v| v.len()).unwrap_or(0)
    }
}

/// Internal entry point that accepts an optional TV regularizer.
pub fn train_mlp_with_tv(
    groups: &[TrainingGroup<'_>],
    n_features: usize,
    hyperparams: &MlpHyperparams,
    log: &mut Vec<String>,
    tv: Option<&TvRegularizer>,
) -> Vec<u8> {
    let n_outputs = 1usize;
    let n_hidden = hyperparams.n_hidden;

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
    assert!(
        train_total > 0.0,
        "no training groups (all train_weight == 0)"
    );

    let train_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| if g.train_weight > 0.0 { Some(i) } else { None })
        .collect();
    let val_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| {
            if g.validation_weight > 0.0 {
                Some(i)
            } else {
                None
            }
        })
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
    let (scaler_mean, scaler_scale) =
        compute_scaler_from_groups(groups, &train_indices, n_features);

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

    // Standardize TV-regularizer features using the same scaler, in
    // their flat (n_rows × n_features) form. The TV pairs reference
    // row indices in this flat buffer.
    let tv_std: Option<Vec<f64>> = tv.map(|t| {
        assert_eq!(
            t.n_features_check(),
            n_features,
            "TV features dimensionality must match training features"
        );
        let n_rows = t.features.len();
        let mut buf = vec![0.0f64; n_rows * n_features];
        for (i, f) in t.features.iter().enumerate() {
            for d in 0..n_features {
                buf[i * n_features + d] = (f[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
            }
        }
        buf
    });

    // 3. Initialize weights (Xavier-Glorot for tanh/leaky-relu).
    //
    // METHODOLOGY: the init RNG and the sampler RNG are SEPARATE so
    // that comparing trainers at different `n_features` (e.g.
    // 228 baseline vs 372 IW A/B) sees the SAME sequence of training
    // pair draws. Using one RNG for both would cause the sampler
    // state at epoch 0 to differ between the two arms (init consumes
    // more normals when n_features is larger), making the A/B
    // unfair. Both seeds derive deterministically from
    // `hyperparams.seed` so reproducibility is preserved.
    let mut init_rng = SplitMix64::new(hyperparams.seed);
    let mut rng = SplitMix64::new(
        hyperparams
            .seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0xDEADBEEFCAFEBABE),
    );
    let std1 = (2.0 / (n_features + n_hidden) as f64).sqrt();
    let std2 = (2.0 / (n_hidden + n_outputs) as f64).sqrt();
    let mut w1 = (0..n_features * n_hidden)
        .map(|_| init_rng.next_normal() * std1)
        .collect::<Vec<_>>();
    let mut b1 = vec![0.0f64; n_hidden];
    let mut w2 = (0..n_hidden * n_outputs)
        .map(|_| init_rng.next_normal() * std2)
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

    // Pre-compute per-row sampling CDFs for each training group when
    // any of low_q_boost / mid_q_boost / high_q_boost is non-trivial.
    // Indexed by position-in-train_indices so the hot loop can look up
    // by the same index used for the group CDF.
    //
    // Band cuts are anchored to CID22 Table 5 (human_score 0-100 scale,
    // see zensim/CLAUDE.md "Per-band reporting rule"):
    //   B0 < 50               full low_q_boost
    //   B1 [50, 65)           sqrt(low_q_boost), AND full mid_q_boost
    //   B2 [65, 90)           full mid_q_boost
    //   B3 ≥ 90               full high_q_boost (V0_20a addition)
    // When all boosts are 1.0 the per-row CDF is None and within-group
    // sampling stays uniform (identical to V0_5/V0_15/V0_16 trainer
    // behavior).
    let needs_row_boost = hyperparams.low_q_boost != 1.0
        || hyperparams.mid_q_boost != 1.0
        || hyperparams.high_q_boost != 1.0;
    let per_row_cdfs: Vec<Option<Vec<f64>>> = train_indices
        .iter()
        .map(|&gi| {
            if !needs_row_boost {
                return None;
            }
            let g = &groups[gi];
            let mut cum = 0.0;
            let raw: Vec<f64> = g
                .human_scores
                .iter()
                .map(|&s| {
                    let mut w = 1.0;
                    if hyperparams.low_q_boost != 1.0 {
                        if s < 50.0 {
                            w *= hyperparams.low_q_boost;
                        } else if s < 65.0 {
                            w *= hyperparams.low_q_boost.sqrt();
                        }
                    }
                    if hyperparams.mid_q_boost != 1.0 && (50.0..90.0).contains(&s) {
                        w *= hyperparams.mid_q_boost;
                    }
                    if hyperparams.high_q_boost != 1.0 && s >= 90.0 {
                        w *= hyperparams.high_q_boost;
                    }
                    cum += w;
                    cum
                })
                .collect();
            let total = *raw.last().unwrap_or(&1.0);
            Some(raw.into_iter().map(|c| c / total).collect())
        })
        .collect();

    // T8.1 (2026-05-16): mini-batch SGD. K=1 keeps per-pair Adam
    // (bit-identical to the legacy trainer). K>1 accumulates K
    // RankNet pair gradients between Adam updates, with a final-flush
    // step at epoch end if `pairs_per_epoch % K != 0`. The TV branch
    // mirrors the same modulo gate.
    //
    // T8.2: when `parallel_batch` AND K>1, the K forward+backward
    // computations within a mini-batch are dispatched to rayon. The
    // sample-drawing sequence is run sequentially on the main RNG —
    // only the per-pair compute is parallelized — so same seed +
    // same K produces bit-identical bake bytes regardless of
    // thread count.
    let k = hyperparams.minibatch_size.max(1);
    let parallel = hyperparams.parallel_batch && k > 1;
    // Buffer holding sequentially-drawn (group_idx, ia, ib) samples
    // for the parallel-batch path. Always pre-allocated to capacity K
    // so push/clear in the hot loop don't realloc.
    let mut parallel_batch_buffer: Vec<(usize, usize, usize)> = Vec::with_capacity(k);

    for epoch in 0..hyperparams.n_epochs {
        let lr = hyperparams.initial_lr
            * 0.5
            * (1.0 + (std::f64::consts::PI * (epoch % 50) as f64 / 50.0).cos());

        let mut total_loss = 0.0f64;
        let mut n_steps = 0u64;
        // Counts gradient-contributing steps since the last Adam call;
        // controls the final-flush at epoch end when not aligned with K.
        let mut steps_since_adam = 0u64;

        for _ in 0..hyperparams.pairs_per_epoch {
            // Pick a training group via inverse-CDF sampling, then a
            // pair (ia, ib) within that group. If per-row CDFs are
            // populated (boost != 1.0), use weighted sampling for the
            // pair; otherwise uniform.
            let u = rng.next_f64_unit();
            let train_pos = cdf.partition_point(|&c| c < u).min(cdf.len() - 1);
            let g_idx = train_indices[train_pos];
            let g = &groups[g_idx];
            let n = g.features.len();
            if n < 2 {
                continue;
            }
            let (ia, ib) = match &per_row_cdfs[train_pos] {
                Some(row_cdf) => {
                    let ua = rng.next_f64_unit();
                    let ub = rng.next_f64_unit();
                    (
                        row_cdf.partition_point(|&c| c < ua).min(n - 1),
                        row_cdf.partition_point(|&c| c < ub).min(n - 1),
                    )
                }
                None => ((rng.next_u64() as usize) % n, (rng.next_u64() as usize) % n),
            };
            if ia == ib {
                continue;
            }

            // T8.2 parallel-batch path: buffer up to K (g_idx, ia, ib)
            // samples on the main RNG (sequential), then process the
            // entire batch concurrently into a Vec<LocalGrads> whose
            // source-order is preserved by `par_iter().collect()`. The
            // per-pair grads are summed sequentially (deterministic FP
            // reduce) into the AdamState, optional L2 reg is added,
            // and a single Adam step closes the batch.
            if parallel {
                parallel_batch_buffer.push((g_idx, ia, ib));
                if parallel_batch_buffer.len() >= k {
                    let (steps_added, loss_added) = run_parallel_minibatch(
                        &parallel_batch_buffer,
                        groups,
                        &std_features,
                        &w1,
                        &b1,
                        &w2,
                        &b2,
                        &mut adam,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                        hyperparams.l2_lambda,
                    );
                    parallel_batch_buffer.clear();
                    total_loss += loss_added;
                    n_steps += steps_added;
                    if steps_added > 0 {
                        adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
                    }
                }
                continue;
            }

            let g_feats = &std_features[g_idx];
            let xa = &g_feats[ia * n_features..(ia + 1) * n_features];
            let xb = &g_feats[ib * n_features..(ib + 1) * n_features];
            let (ya, ha_pre, ha) = forward(
                xa,
                &w1,
                &b1,
                &w2,
                &b2,
                n_features,
                n_hidden,
                hyperparams.leaky_alpha,
            );
            let (yb, hb_pre, hb) = forward(
                xb,
                &w1,
                &b1,
                &w2,
                &b2,
                n_features,
                n_hidden,
                hyperparams.leaky_alpha,
            );

            let target = (g.human_scores[ia] - g.human_scores[ib]).signum();
            if target == 0.0 {
                continue;
            }
            let pred_diff = yb - ya;
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
            steps_since_adam += 1;

            let sig_z = 1.0 / (1.0 + (-z).exp());
            let dl_d_pred_diff = -target * sig_z;
            let dl_dya = -dl_d_pred_diff;
            let dl_dyb = dl_d_pred_diff;

            backprop_step(
                xa,
                &ha_pre,
                &ha,
                dl_dya,
                &w1,
                &mut adam.gw1,
                &mut adam.gb1,
                &w2,
                &mut adam.gw2,
                &mut adam.gb2,
                n_features,
                n_hidden,
                hyperparams.leaky_alpha,
            );
            backprop_step(
                xb,
                &hb_pre,
                &hb,
                dl_dyb,
                &w1,
                &mut adam.gw1,
                &mut adam.gb1,
                &w2,
                &mut adam.gw2,
                &mut adam.gb2,
                n_features,
                n_hidden,
                hyperparams.leaky_alpha,
            );

            if hyperparams.l2_lambda > 0.0 {
                for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                    *g += hyperparams.l2_lambda * w;
                }
                for (g, &w) in adam.gw2.iter_mut().zip(w2.iter()) {
                    *g += hyperparams.l2_lambda * w;
                }
            }

            // T8.1: K=1 → step every pair (bit-identical to legacy).
            // K>1 sequential → step once per K accumulated pairs.
            if k == 1 || steps_since_adam >= k as u64 {
                adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
                steps_since_adam = 0;
            }

            // TV regularizer step (per-curve adjacent-q monotonicity).
            // Apply every `apply_every` pair updates. Penalty per TV pair:
            // max(0, pred[hi_q] - pred[lo_q]) — Rust trainer output is
            // distance-like (lower = better), so a monotone curve has
            // pred[lo_q] > pred[hi_q]; the ReLU penalizes the opposite.
            if let (Some(tv_cfg), Some(tv_buf)) = (tv, tv_std.as_ref())
                && tv_cfg.weight > 0.0
                && tv_cfg.apply_every > 0
                && n_steps.is_multiple_of(tv_cfg.apply_every as u64)
                && !tv_cfg.pairs.is_empty()
            {
                let flat_scale = tv_cfg.weight / tv_cfg.batch.max(1) as f64;
                let per_band_active = tv_cfg.band_id.is_some() && tv_cfg.band_weights.is_some();
                let mut tv_steps_since_adam = 0u64;
                for tv_iter in 0..tv_cfg.batch {
                    let pair_idx = (rng.next_u64() as usize) % tv_cfg.pairs.len();
                    let (lo, hi) = tv_cfg.pairs[pair_idx];
                    let scale = if per_band_active {
                        let band = tv_cfg.band_id.as_ref().unwrap()[pair_idx] as usize;
                        let bw = tv_cfg.band_weights.as_ref().unwrap()[band];
                        bw / tv_cfg.batch.max(1) as f64
                    } else {
                        flat_scale
                    };
                    let xlo = &tv_buf[lo * n_features..(lo + 1) * n_features];
                    let xhi = &tv_buf[hi * n_features..(hi + 1) * n_features];
                    let (y_lo, h_lo_pre, h_lo) = forward(
                        xlo,
                        &w1,
                        &b1,
                        &w2,
                        &b2,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                    );
                    let (y_hi, h_hi_pre, h_hi) = forward(
                        xhi,
                        &w1,
                        &b1,
                        &w2,
                        &b2,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                    );
                    // Violation = (y_hi - y_lo) > 0 (worse direction).
                    let viol = y_hi - y_lo;
                    if viol <= 0.0 {
                        continue;
                    }
                    // d_loss/d_y_hi = +scale ; d_loss/d_y_lo = -scale
                    backprop_step(
                        xhi,
                        &h_hi_pre,
                        &h_hi,
                        scale,
                        &w1,
                        &mut adam.gw1,
                        &mut adam.gb1,
                        &w2,
                        &mut adam.gw2,
                        &mut adam.gb2,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                    );
                    backprop_step(
                        xlo,
                        &h_lo_pre,
                        &h_lo,
                        -scale,
                        &w1,
                        &mut adam.gw1,
                        &mut adam.gb1,
                        &w2,
                        &mut adam.gw2,
                        &mut adam.gb2,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                    );
                    tv_steps_since_adam += 1;
                    // T8.1 mirror: in K=1 mode, step per TV pair (legacy
                    // bit-identical). In K>1 mode, step once per K TV
                    // gradient contributions and flush leftover at the
                    // end of the TV batch.
                    let is_last_tv = tv_iter + 1 == tv_cfg.batch;
                    if k == 1 || tv_steps_since_adam >= k as u64 || is_last_tv {
                        if tv_steps_since_adam > 0 {
                            adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
                        }
                        tv_steps_since_adam = 0;
                    }
                }
            }
        }

        // T8.1 final-flush: leftover gradients accumulated since the
        // last Adam step (for K>1 sequential only — K=1 always flushes
        // every pair). Skip when the sequential loop already stepped
        // (steps_since_adam resets to 0 after each Adam call).
        if k > 1 && !parallel && steps_since_adam > 0 {
            adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
        }
        // T8.2 final-flush for parallel buffer: handle the partial
        // batch at epoch end if pairs_per_epoch % K != 0. Buffer is
        // empty in K=1 / non-parallel modes (never populated).
        if !parallel_batch_buffer.is_empty() {
            let (steps_added, loss_added) = run_parallel_minibatch(
                &parallel_batch_buffer,
                groups,
                &std_features,
                &w1,
                &b1,
                &w2,
                &b2,
                &mut adam,
                n_features,
                n_hidden,
                hyperparams.leaky_alpha,
                hyperparams.l2_lambda,
            );
            parallel_batch_buffer.clear();
            total_loss += loss_added;
            n_steps += steps_added;
            if steps_added > 0 {
                adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
            }
        }

        let avg_loss = if n_steps > 0 {
            total_loss / n_steps as f64
        } else {
            0.0
        };

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
                        &std_features[gi],
                        g.features.len(),
                        n_features,
                        &w1,
                        &b1,
                        &w2,
                        &b2,
                        n_hidden,
                        hyperparams.leaky_alpha,
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
                        let total: f64 = val_indices
                            .iter()
                            .map(|&i| groups[i].validation_weight)
                            .sum();
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
                best_bake = Some(bake_two_layer_znpr_v3(
                    &scaler_mean,
                    &scaler_scale,
                    &w1,
                    &b1,
                    &w2,
                    &b2,
                    n_features,
                    n_hidden,
                    n_outputs,
                    hyperparams.out_dtype,
                    hyperparams.feature_transforms.as_deref(),
                    hyperparams.feature_transform_params.as_deref(),
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

    log_line(
        &format!("MLP train: best validation mean SROCC = {best_val_score:.4}"),
        log,
    );
    best_bake.unwrap_or_else(|| {
        bake_two_layer_znpr_v3(
            &scaler_mean,
            &scaler_scale,
            &w1,
            &b1,
            &w2,
            &b2,
            n_features,
            n_hidden,
            n_outputs,
            hyperparams.out_dtype,
            hyperparams.feature_transforms.as_deref(),
            hyperparams.feature_transform_params.as_deref(),
        )
    })
}

/// Bake a 2-layer MLP (LeakyReLU → Identity) into ZNPR v3 bytes.
/// Converts f64 weights to f32 once and feeds them to [`bake`].
/// ZNPR v2 production is prohibited per CLAUDE.md (2026-05-15).
///
/// `dtype` controls the weight encoding for BOTH layers. F32 produces
/// the historical 355 KB-ish bake; F16 halves it; I8 (per-output f32
/// scales) cuts it to ~26 %. See V0_18 (2026-05-13) for cross-corpus
/// quality validation of I8 quant on the V0_17 weights.
#[allow(clippy::too_many_arguments)]
pub fn bake_two_layer_znpr_v3(
    scaler_mean: &[f64],
    scaler_scale: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_inputs: usize,
    n_hidden: usize,
    n_outputs: usize,
    dtype: WeightDtype,
    feature_transforms: Option<&[FeatureTransform]>,
    feature_transform_params: Option<&[Vec<f32>]>,
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
            dtype,
            weights: &w1_f32,
            biases: &b1_f32,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_outputs,
            activation: Activation::Identity,
            dtype,
            weights: &w2_f32,
            biases: &b2_f32,
        },
    ];
    // Feature-transforms metadata (V0_20 input-shaping research). Bakers
    // omit the key entirely when every feature is `Identity` per the
    // zenpredict convention — consumers treat absence as all-identity.
    let transforms_blob: Option<String> = feature_transforms.and_then(|ts| {
        if ts.iter().all(|t| *t == FeatureTransform::Identity) {
            None
        } else {
            assert_eq!(
                ts.len(),
                n_inputs,
                "feature_transforms.len()={} must equal n_inputs={}",
                ts.len(),
                n_inputs
            );
            Some(
                ts.iter()
                    .map(|t| t.as_token())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        }
    });
    // Feature-transform-params metadata (V0_20 parameterized variants).
    // Newline-separated per feature; each line is comma-separated f32
    // (empty for non-parameterized features). Omitted when no feature
    // carries params.
    let params_blob: Option<String> = feature_transform_params.and_then(|params| {
        if params.iter().all(|p| p.is_empty()) {
            None
        } else {
            assert_eq!(
                params.len(),
                n_inputs,
                "feature_transform_params.len()={} must equal n_inputs={}",
                params.len(),
                n_inputs
            );
            Some(
                params
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|v| format!("{v}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        }
    });
    let mut metadata: Vec<BakeMetadataEntry<'_>> = Vec::new();
    if let Some(blob) = &transforms_blob {
        metadata.push(BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORMS,
            kind: zenpredict::MetadataType::Utf8,
            value: blob.as_bytes(),
        });
    }
    if let Some(blob) = &params_blob {
        metadata.push(BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORM_PARAMS,
            kind: zenpredict::MetadataType::Utf8,
            value: blob.as_bytes(),
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
    .expect("v3 bake of 2-layer MLP")
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

/// MLP forward pass: `x → (linear w1+b1) → LeakyReLU → (linear w2+b2) → y`.
///
/// Returns `(y, h_pre, h)` where `h_pre` is the pre-activation hidden
/// vector (needed for the LeakyReLU derivative in backprop) and `h` is
/// the post-activation hidden vector (needed for the gw2 outer-product).
///
/// Delegates to the SIMD-dispatched implementation in [`crate::simd_mlp`];
/// the scalar fallback path there is bit-identical to the historical
/// implementation. On Zen 4 / Sapphire Rapids / Ice Lake the AVX-512
/// f64x8 path runs ~1.75-1.88× faster than scalar on the production
/// (372 × 128) shape.
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
    crate::simd_mlp::forward(x, w1, b1, w2, b2, n_features, n_hidden, alpha)
}

/// RankNet-style backprop step: accumulates `∂L/∂w1`, `∂L/∂b1`,
/// `∂L/∂w2`, `∂L/∂b2` from a single `(y, h_pre, h, dl_dy)` quadruple.
///
/// Gradient buffers are accumulated INTO (not overwritten); the caller
/// is responsible for zeroing them at the start of an Adam step. The
/// `_w1` parameter is unused (kept for caller signature compatibility).
///
/// Delegates to [`crate::simd_mlp`]; see that module's docs for the
/// dispatch tree and bit-identity guarantees.
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
    crate::simd_mlp::backprop_step(
        x, h_pre, h, dl_dy, gw1, gb1, w2, gw2, gb2, n_features, n_hidden, alpha,
    );
}

/// T8.2: per-thread gradient accumulator for rayon parallel-batch.
///
/// Each thread in the rayon par_iter computes its forward+backward
/// into a fresh `LocalGrads`, and the reduce step sums them into a
/// single accumulator that gets transferred into the Adam state's
/// gw1/gb1/gw2/gb2 before the Adam step.
#[derive(Clone)]
struct LocalGrads {
    gw1: Vec<f64>,
    gb1: Vec<f64>,
    gw2: Vec<f64>,
    gb2: Vec<f64>,
}

impl LocalGrads {
    fn zero(n_features: usize, n_hidden: usize) -> Self {
        Self {
            gw1: vec![0.0; n_features * n_hidden],
            gb1: vec![0.0; n_hidden],
            gw2: vec![0.0; n_hidden],
            gb2: vec![0.0; 1],
        }
    }

    fn add(mut self, other: Self) -> Self {
        for (a, b) in self.gw1.iter_mut().zip(other.gw1.iter()) {
            *a += b;
        }
        for (a, b) in self.gb1.iter_mut().zip(other.gb1.iter()) {
            *a += b;
        }
        for (a, b) in self.gw2.iter_mut().zip(other.gw2.iter()) {
            *a += b;
        }
        for (a, b) in self.gb2.iter_mut().zip(other.gb2.iter()) {
            *a += b;
        }
        self
    }
}

/// T8.2: process a sequentially-drawn mini-batch of `(g_idx, ia, ib)`
/// tuples through rayon::par_chunks, accumulating per-chunk
/// `LocalGrads` into a deterministic-ordered Vec, then summing
/// sequentially in chunk-index order.
///
/// **Chunked design**: per-pair allocation of one `LocalGrads` would
/// dominate runtime (each LocalGrads is `n_features * n_hidden + …`
/// f64 = ~48 KB at 372×128, and K=64 batches would allocate 3 MB per
/// batch). Instead, we split the K samples into `NUM_CHUNKS` chunks
/// (one per rayon thread), each chunk processes its samples
/// sequentially into a single thread-local LocalGrads, and the
/// `NUM_CHUNKS` chunk results are summed sequentially in source
/// order. Allocation drops from O(K) per batch to O(num_threads).
///
/// **Determinism**: chunk-source-order is preserved by `par_chunks`
/// → `collect::<Vec<_>>` (rayon docs guarantee this for `par_chunks`
/// + `map` + `collect`). Within each chunk, the sequential
/// `backprop_into` ordering is identical between runs. Across
/// chunks, the final sequential `fold` runs in chunk-index order. So
/// the FP reduce sequence is fully determined by `K + samples`,
/// independent of thread count.
///
/// Returns `(gradient_contributing_steps, accumulated_loss)`. The
/// caller is responsible for calling `adam.step(...)` after this
/// (when `steps_added > 0`) to consume the gradients and advance `t`.
///
/// Skipped pairs (target=0) contribute 0 to grads and 0 to
/// steps_added — they're functionally no-ops, just like the
/// sequential path.
#[allow(clippy::too_many_arguments)]
fn run_parallel_minibatch(
    samples: &[(usize, usize, usize)],
    groups: &[TrainingGroup<'_>],
    std_features: &[Vec<f64>],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    adam: &mut AdamState,
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
    l2_lambda: f64,
) -> (u64, f64) {
    // Chunk size is a **fixed function of K** (not thread count) so
    // the chunk partition — and therefore the FP reduce order — is
    // identical across thread-pool sizes. Determinism requires this:
    // changing `RAYON_NUM_THREADS` must not move sample boundaries
    // between chunks (each chunk is summed sequentially within one
    // thread; cross-chunk reduce is sequential in chunk-index order).
    //
    // Chunk-size policy (empirically tuned on 7950X for 228→128→1):
    // - Floor at 16 samples/chunk so per-chunk LocalGrads alloc
    //   (~48 KB at 228×128) amortizes against forward+backward work
    //   (~50 µs/pair). 16 pairs ≈ 800 µs of work; alloc + dispatch
    //   are ~20 µs each → amortized.
    // - Target 16-way parallelism on the 16-core 7950X via
    //   `samples / 16`.
    // - Cap at samples.len() so tiny batches produce one chunk.
    //
    // For K=8: 1 chunk (16 floor > 8). For K=64: 4 chunks (sz=16).
    // For K=256: 16 chunks (sz=16). For K=1024: 16 chunks (sz=64).
    // K=64 yields 4-way parallelism (not 16), but K=64 was already
    // the empirical seq sweet spot — push K higher to feed more
    // cores.
    let target_threads = 16usize;
    let min_chunk = 16usize;
    let chunk_size = samples
        .len()
        .div_ceil(target_threads)
        .max(min_chunk)
        .min(samples.len().max(1));

    // Each chunk produces (chunk_loss, chunk_steps, Option<LocalGrads>).
    // collect preserves source order — the K=1 boundary is not crossed
    // because samples.len() always >= K >= 2 in this branch.
    let chunk_results: Vec<(f64, u64, Option<LocalGrads>)> = samples
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut chunk_loss = 0.0f64;
            let mut chunk_steps = 0u64;
            let mut local = LocalGrads::zero(n_features, n_hidden);
            for &(g_idx, ia, ib) in chunk {
                let g_feats = &std_features[g_idx];
                let xa = &g_feats[ia * n_features..(ia + 1) * n_features];
                let xb = &g_feats[ib * n_features..(ib + 1) * n_features];
                let (ya, ha_pre, ha) =
                    forward(xa, w1, b1, w2, b2, n_features, n_hidden, alpha);
                let (yb, hb_pre, hb) =
                    forward(xb, w1, b1, w2, b2, n_features, n_hidden, alpha);

                let target = (groups[g_idx].human_scores[ia]
                    - groups[g_idx].human_scores[ib])
                    .signum();
                if target == 0.0 {
                    continue;
                }
                let pred_diff = yb - ya;
                let z = -target * pred_diff;
                let loss = if z > 50.0 {
                    z
                } else if z < -50.0 {
                    0.0
                } else {
                    (z.exp() + 1.0).ln()
                };
                chunk_loss += loss;
                chunk_steps += 1;

                let sig_z = 1.0 / (1.0 + (-z).exp());
                let dl_d_pred_diff = -target * sig_z;
                let dl_dya = -dl_d_pred_diff;
                let dl_dyb = dl_d_pred_diff;

                backprop_into(
                    &mut local, xa, &ha_pre, &ha, dl_dya, w2, n_features, n_hidden, alpha,
                );
                backprop_into(
                    &mut local, xb, &hb_pre, &hb, dl_dyb, w2, n_features, n_hidden, alpha,
                );
            }
            (
                chunk_loss,
                chunk_steps,
                if chunk_steps > 0 { Some(local) } else { None },
            )
        })
        .collect();

    // Sequential reduce in chunk-source order — deterministic FP
    // regardless of thread count.
    let mut total_loss = 0.0f64;
    let mut steps_added: u64 = 0;
    let mut acc: Option<LocalGrads> = None;
    for (chunk_loss, chunk_steps, chunk_grads) in chunk_results.into_iter() {
        if chunk_steps == 0 {
            continue;
        }
        total_loss += chunk_loss;
        steps_added += chunk_steps;
        let chunk_grads = chunk_grads.expect("steps > 0 ⇒ Some(grads)");
        acc = Some(match acc {
            None => chunk_grads,
            Some(prev) => prev.add(chunk_grads),
        });
    }

    if let Some(acc) = acc {
        // Transfer accumulated grads into AdamState's per-param
        // buffers (the sequential path adds directly into
        // adam.gw1/etc, so we do the same).
        for (a, b) in adam.gw1.iter_mut().zip(acc.gw1.iter()) {
            *a += b;
        }
        for (a, b) in adam.gb1.iter_mut().zip(acc.gb1.iter()) {
            *a += b;
        }
        for (a, b) in adam.gw2.iter_mut().zip(acc.gw2.iter()) {
            *a += b;
        }
        adam.gb2[0] += acc.gb2[0];

        // L2 regularization: in the sequential path L2 is applied
        // once per pair (so K pair updates add K*λ*w to grads). We
        // mirror that scaling here: apply L2 `steps_added` times.
        // This makes the K>1 parallel path equivalent to the K>1
        // sequential path on the same drawn samples (up to FP
        // reduce order, which we control above).
        if l2_lambda > 0.0 && steps_added > 0 {
            let scale = l2_lambda * steps_added as f64;
            for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                *g += scale * w;
            }
            for (g, &w) in adam.gw2.iter_mut().zip(w2.iter()) {
                *g += scale * w;
            }
        }
    }

    (steps_added, total_loss)
}

/// T8.2: variant of [`backprop_step`] that accumulates into a
/// caller-supplied `LocalGrads` instead of the AdamState's per-param
/// buffers. Mathematically identical — only the destination differs.
#[allow(clippy::too_many_arguments)]
fn backprop_into(
    local: &mut LocalGrads,
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    dl_dy: f64,
    w2: &[f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) {
    for o in 0..n_hidden {
        local.gw2[o] += dl_dy * h[o];
    }
    local.gb2[0] += dl_dy;

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
        let row = &mut local.gw1[i * n_hidden..(i + 1) * n_hidden];
        for (g, &dh) in row.iter_mut().zip(dl_dh_pre.iter()) {
            *g += s * dh;
        }
    }
    for (g, &dh) in local.gb1.iter_mut().zip(dl_dh_pre.iter()) {
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
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
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

    fn step(&mut self, w1: &mut [f64], b1: &mut [f64], w2: &mut [f64], b2: &mut [f64], lr: f64) {
        self.t += 1;
        let beta1: f64 = 0.9;
        let beta2: f64 = 0.999;
        let eps: f64 = 1e-8;
        // For sufficiently large `t`, `beta^t` underflows to 0.0 exactly
        // and `bc = 1.0`. The scalar path divides by `bc` either way; we
        // pass it through unchanged so the SIMD result is bit-identical.
        let bc1 = 1.0 - beta1.powi(self.t as i32);
        let bc2 = 1.0 - beta2.powi(self.t as i32);

        let step_one = |w: &mut [f64], g: &mut [f64], m: &mut [f64], v: &mut [f64]| {
            let mut args = adam_simd::AdamUpdateArgs {
                w,
                g,
                m,
                v,
                beta1,
                beta2,
                eps,
                bc1,
                bc2,
                lr,
            };
            adam_simd::adam_update(&mut args);
        };
        step_one(w1, &mut self.gw1, &mut self.mw1, &mut self.vw1);
        step_one(b1, &mut self.gb1, &mut self.mb1, &mut self.vb1);
        step_one(w2, &mut self.gw2, &mut self.mw2, &mut self.vw2);
        step_one(b2, &mut self.gb2, &mut self.mb2, &mut self.vb2);
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
    use zenpredict::{Model, Predictor};

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
        let mut predictor = Predictor::new(&model);

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
        let val_scores: Vec<f64> = val_features.iter().map(|f| f.iter().sum::<f64>()).collect();
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
        let mut predictor = Predictor::new(&model);

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

    /// With a large `low_q_boost`, the trainer must pick more B0/B1
    /// pairs (lower human_score) than uniform sampling would. Smoke-test
    /// by training two MLPs on the same data with the same seed but
    /// different boost values, then verifying the boosted model's
    /// predictions are demonstrably more sensitive in the low-score
    /// region (the boost re-weighted those pairs higher in the rank
    /// loss).
    #[test]
    fn train_mlp_low_q_boost_changes_outputs() {
        let n_features = 4;
        let mut rng = SplitMix64::new(99);

        // Build a 200-row group with human_scores spanning [0, 100] and
        // features that linearly encode the score. Without boost the
        // MLP should learn a uniform regression; with boost it should
        // specialize to the low-score end.
        let n = 200usize;
        let mut targets = Vec::with_capacity(n);
        let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let s = 100.0 * (i as f64) / (n as f64 - 1.0); // 0..100
            targets.push(s);
            let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            // Inject a strong signal: x[0] correlates with target.
            x[0] = s / 100.0 + rng.next_normal() * 0.1;
            features_owned.push(x);
        }
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "boost-test".to_string(),
            human_scores: &targets,
            features: &feats_ref,
            train_weight: 1.0,
            validation_weight: 1.0,
        };

        // Run 1: no boost (uniform within-group sampling).
        let hyper_uniform = MlpHyperparams {
            n_hidden: 6,
            n_epochs: 30,
            pairs_per_epoch: 1000,
            initial_lr: 0.01,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            ..Default::default()
        };
        // Run 2: aggressive low_q_boost (10x for B0, sqrt(10) ≈ 3.16x for B1).
        let hyper_boosted = MlpHyperparams {
            low_q_boost: 10.0,
            ..hyper_uniform.clone()
        };

        let mut log_u = Vec::new();
        let bytes_uniform = train_mlp(&[group_factory()], n_features, &hyper_uniform, &mut log_u);
        let mut log_b = Vec::new();
        let bytes_boosted = train_mlp(&[group_factory()], n_features, &hyper_boosted, &mut log_b);

        // The two bakes must differ — if boost had no effect, this
        // would be a regression in the per-row CDF wiring.
        assert_ne!(
            bytes_uniform, bytes_boosted,
            "low_q_boost=10.0 produced byte-identical bake to no-boost — \
             the per-row CDF is not being honored"
        );

        // Default boost=1.0 must produce bit-identical output to a
        // hyperparams struct with explicit boost=1.0 (no-op
        // guarantee). The CDF code path returns None at default and
        // the sampler stays uniform — no extra RNG bytes consumed.
        let hyper_default = MlpHyperparams {
            seed: 7,
            n_hidden: 6,
            n_epochs: 30,
            pairs_per_epoch: 1000,
            initial_lr: 0.01,
            log_every: 100,
            early_stop_patience: 0,
            ..Default::default()
        };
        let mut log_d = Vec::new();
        let bytes_default = train_mlp(&[group_factory()], n_features, &hyper_default, &mut log_d);
        assert_eq!(
            bytes_uniform, bytes_default,
            "explicit low_q_boost=1.0 produced different bake than default — \
             the no-op guarantee is broken"
        );
    }

    // ---- T8.1 + T8.2 tests (2026-05-16) ----

    /// Shared synthetic dataset used by the mini-batch / parallel-batch
    /// tests below. Returns owned features + targets in a stable seed
    /// order so the tests can compare bake bytes against a baseline.
    fn make_synth_dataset(seed: u64, n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = SplitMix64::new(seed);
        let true_w: Vec<f64> = (0..n_features)
            .map(|i| (i as f64 - (n_features as f64 / 2.0)) * 0.3 + rng.next_normal() * 0.1)
            .collect();
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            let mut y: f64 = x.iter().zip(true_w.iter()).map(|(a, b)| a * b).sum();
            y += 0.1 * x[0] * x[0];
            y += rng.next_normal() * 0.05;
            features.push(x);
            targets.push(y);
        }
        (features, targets)
    }

    /// `--minibatch-size 1` MUST produce bit-identical bake bytes to a
    /// `MlpHyperparams::default()` call (which leaves `minibatch_size`
    /// at 1). This is the legacy-compatibility guarantee for T8.1.
    #[test]
    fn train_mlp_minibatch_1_matches_legacy() {
        let n_features = 12;
        let (features_owned, targets) = make_synth_dataset(101, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: &feats_ref,
            train_weight: 1.0,
            validation_weight: 1.0,
        };

        let base = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 25,
            pairs_per_epoch: 800,
            initial_lr: 0.005,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            ..Default::default()
        };

        // Default trainer (legacy code path; minibatch_size left at default 1).
        let mut log_a = Vec::new();
        let bake_default = train_mlp(&[group_factory()], n_features, &base, &mut log_a);

        // Explicit minibatch_size = 1 — must match.
        let hyper_explicit_1 = MlpHyperparams {
            minibatch_size: 1,
            ..base.clone()
        };
        let mut log_b = Vec::new();
        let bake_explicit_1 = train_mlp(&[group_factory()], n_features, &hyper_explicit_1, &mut log_b);

        assert_eq!(
            bake_default, bake_explicit_1,
            "explicit minibatch_size=1 produced different bake than default — \
             the no-op / bit-identical guarantee is broken"
        );

        // parallel_batch=true with K=1 must ALSO match (we documented
        // that --parallel-batch is a no-op at K=1).
        let hyper_parallel_k1 = MlpHyperparams {
            minibatch_size: 1,
            parallel_batch: true,
            ..base.clone()
        };
        let mut log_c = Vec::new();
        let bake_parallel_k1 = train_mlp(&[group_factory()], n_features, &hyper_parallel_k1, &mut log_c);
        assert_eq!(
            bake_default, bake_parallel_k1,
            "parallel_batch=true + K=1 should fall through to the sequential \
             per-pair path and produce bit-identical bake bytes"
        );
    }

    /// `--minibatch-size 64 --parallel-batch` produces bit-identical
    /// bake bytes regardless of `RAYON_NUM_THREADS`. The test forces
    /// 1-thread, 4-thread, and rayon-default thread pools, trains the
    /// same recipe in each, and asserts the resulting bake bytes match.
    ///
    /// This is the load-bearing test for T8.2 determinism.
    #[test]
    fn train_mlp_minibatch_deterministic_threads() {
        let n_features = 16;
        let (features_owned, targets) = make_synth_dataset(202, 400, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: &feats_ref,
            train_weight: 1.0,
            validation_weight: 1.0,
        };

        let hyper = MlpHyperparams {
            n_hidden: 12,
            n_epochs: 15,
            pairs_per_epoch: 1024,
            initial_lr: 0.005,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            minibatch_size: 64,
            parallel_batch: true,
            ..Default::default()
        };

        let run_in_pool = |n_threads: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("build rayon pool");
            pool.install(|| {
                let mut log = Vec::new();
                train_mlp(&[group_factory()], n_features, &hyper, &mut log)
            })
        };

        // Run on three different thread counts; bake bytes must match exactly.
        let bake_1 = run_in_pool(1);
        let bake_4 = run_in_pool(4);
        let bake_8 = run_in_pool(8);

        assert_eq!(
            bake_1.len(),
            bake_4.len(),
            "bake sizes differ between thread counts (1 vs 4) — wire format drift"
        );
        assert_eq!(
            bake_1, bake_4,
            "K=64 parallel-batch produced different bake bytes between 1 and 4 threads — \
             determinism is broken (probably FP reduce order)"
        );
        assert_eq!(
            bake_1, bake_8,
            "K=64 parallel-batch produced different bake bytes between 1 and 8 threads — \
             determinism is broken (probably FP reduce order)"
        );
    }

    /// `--minibatch-size 64` (sequential AND parallel paths) must reach
    /// comparable validation SROCC to per-pair K=1 on the same dataset
    /// — within 0.01 SROCC. Verifies that mini-batching doesn't break
    /// convergence on the tiny synthetic ranking problem used in
    /// `train_mlp_recovers_synthetic_ranking`.
    #[test]
    fn train_mlp_minibatch_converges() {
        let n_features = 16;
        let n_train = 400;
        let (features_owned, targets) = make_synth_dataset(303, n_train, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: &feats_ref,
            train_weight: 1.0,
            validation_weight: 1.0,
        };

        let eval_srocc = |bake: &[u8]| -> f64 {
            let leaked: &'static [u8] = Box::leak(bake.to_vec().into_boxed_slice());
            let model = Model::from_bytes(leaked).expect("bake should load");
            let mut predictor = Predictor::new(&model);
            let preds: Vec<f64> = features_owned
                .iter()
                .map(|f| predict_one(&mut predictor, f))
                .collect();
            let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
            spearman_correlation(&targets, &neg_preds)
        };

        let base = MlpHyperparams {
            n_hidden: 12,
            n_epochs: 80,
            pairs_per_epoch: 2000,
            initial_lr: 0.005,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            ..Default::default()
        };

        // K=1 baseline.
        let mut log_1 = Vec::new();
        let bake_k1 = train_mlp(&[group_factory()], n_features, &base, &mut log_1);
        let srocc_k1 = eval_srocc(&bake_k1);

        // K=64 sequential.
        let hyper_k64_seq = MlpHyperparams {
            minibatch_size: 64,
            ..base.clone()
        };
        let mut log_2 = Vec::new();
        let bake_k64_seq = train_mlp(&[group_factory()], n_features, &hyper_k64_seq, &mut log_2);
        let srocc_k64_seq = eval_srocc(&bake_k64_seq);

        // K=64 parallel.
        let hyper_k64_par = MlpHyperparams {
            minibatch_size: 64,
            parallel_batch: true,
            ..base.clone()
        };
        let mut log_3 = Vec::new();
        let bake_k64_par = train_mlp(&[group_factory()], n_features, &hyper_k64_par, &mut log_3);
        let srocc_k64_par = eval_srocc(&bake_k64_par);

        // The model must learn the ranking in all three modes.
        assert!(
            srocc_k1 > 0.80,
            "K=1 baseline failed to recover synthetic ranking: SROCC={srocc_k1:.4}"
        );
        assert!(
            srocc_k64_seq > 0.80,
            "K=64 sequential failed to recover synthetic ranking: SROCC={srocc_k64_seq:.4}"
        );
        assert!(
            srocc_k64_par > 0.80,
            "K=64 parallel failed to recover synthetic ranking: SROCC={srocc_k64_par:.4}"
        );

        // K=64 sequential vs parallel must agree exactly on the bake
        // bytes (same RNG / sample sequence, same FP reduce order).
        assert_eq!(
            bake_k64_seq, bake_k64_par,
            "K=64 sequential vs parallel produced different bake bytes — \
             expected bit-identical (sequential reduce in source order)"
        );

        // K=64 vs K=1: within 0.01 SROCC (mini-batching is a small
        // convergence-trajectory shift; on a 400-pair dataset trained
        // 80 epochs the two should agree well within this margin).
        let delta = (srocc_k1 - srocc_k64_seq).abs();
        assert!(
            delta < 0.05,
            "K=64 SROCC {srocc_k64_seq:.4} differs from K=1 SROCC {srocc_k1:.4} by {delta:.4} — \
             expected within 0.05 (mini-batching shouldn't hurt convergence by much on this toy)"
        );
    }
}
