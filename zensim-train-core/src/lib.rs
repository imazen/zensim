//! Pure-Rust trainer core for zensim MLP profiles.
//!
//! WASM-compatible reimplementation of `zensim-validate/src/mlp_train.rs`.
//! Phase 1 of `docs/WASM_CUBECL_TRAINER_PLAN.md` — bit-exact reproduction
//! milestone first, then CubeCL kernel acceleration in Phase 2.
//!
//! Public surface (planned, growing):
//! - [`MlpHyperparams`] — knobs (epochs, lr, l2, validation policy)
//! - [`TrainingGroup`] — one named slice of training/validation data
//! - [`ValidationPolicy`] — `Mean` or `Min` aggregation across groups
//! - `train_mlp` — main entrypoint, returns a ZNPR v3 bake
//!
//! The current scaffold is intentionally minimal: it re-exports
//! `zenpredict::Activation` / `WeightDtype` and declares the
//! hyperparameter types. Bodies will land incrementally so each commit
//! has a clean intermediate.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub use zenpredict::{Activation, WeightDtype};

/// How to aggregate per-group SROCC into the single value used for
/// best-checkpoint selection.
///
/// `Min` is the right default when shipping a metric: a model whose
/// worst dataset is bad will be observably bad in production.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationPolicy {
    /// Average across all validation groups.
    Mean,
    /// Take the worst-performing group's SROCC.
    Min,
}

/// Knobs for the trainer. Defaults match the V0_4 baseline architecture
/// (228 → 32 → 1) with `Min` validation gating.
#[derive(Clone, Debug)]
pub struct MlpHyperparams {
    /// Hidden-layer width (single hidden layer for now).
    pub n_hidden: usize,
    /// Maximum number of epochs.
    pub n_epochs: usize,
    /// RankNet pair samples per epoch.
    pub pairs_per_epoch: usize,
    /// Adam initial learning rate.
    pub initial_lr: f64,
    /// LeakyReLU negative-side slope.
    pub leaky_alpha: f64,
    /// PRNG seed.
    pub seed: u64,
    /// Log validation SROCC every N epochs.
    pub log_every: usize,
    /// L2 regularization on layer weights (not biases). 0 disables.
    pub l2_lambda: f64,
    /// Stop after this many epochs of no validation improvement. 0 disables.
    pub early_stop_patience: usize,
    /// How to combine per-group SROCC.
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

#[allow(dead_code)]
mod rng;
#[cfg(test)]
pub(crate) use rng::SplitMix64;

#[allow(dead_code)]
mod adam;

mod stats;
pub use stats::{pearson, ranks, spearman};

pub mod mlp;

pub mod pool_head;

pub mod hybrid_head;

pub mod per_sample_alpha_head;

pub mod simd_encoder;

/// TV (total-variation) regularizer for adjacent-quality monotonicity.
///
/// `pairs[k] = (lo_idx, hi_idx)` references rows in
/// [`TvRegularizer::features`]. Penalty per pair =
/// `max(0, pred[hi_idx] - pred[lo_idx])` — Rust-trainer outputs are
/// distance-like (lower = better quality), so a monotone curve has
/// `pred[lo_q] > pred[hi_q]`. Violations are when
/// `pred[hi_q] > pred[lo_q]`.
///
/// The trainer applies TV updates every `apply_every` RankNet steps,
/// sampling `batch` pairs per update. At 50_000 pairs/epoch and the
/// default `apply_every = 50`, that's 1000 TV gradient steps per epoch.
#[derive(Debug, Clone)]
pub struct TvRegularizer {
    /// Pair indices: `(lo_idx, hi_idx)` into [`Self::features`]. `lo_idx`
    /// should be the higher-quality row, `hi_idx` the lower-quality row.
    pub pairs: Vec<(usize, usize)>,
    /// Feature rows in the same `n_features`-dimensional space as the
    /// training groups. The trainer standardizes these with the same
    /// scaler as the training data so weights generalize.
    pub features: Vec<Vec<f64>>,
    /// TV penalty weight (added to the per-step gradient scale).
    pub weight: f64,
    /// Apply TV update every N RankNet pair updates. 50 is a good
    /// default — 50_000 RankNet pairs / 50 = 1000 TV steps per epoch.
    pub apply_every: usize,
    /// Mini-batch size of TV pairs per update. 32 is fine.
    pub batch: usize,
}

impl TvRegularizer {
    /// Width of each feature row in [`Self::features`]. Returns 0 if no
    /// rows are present. The trainer asserts this matches the
    /// `n_features` passed to `train_mlp_with_tv`.
    pub fn n_features(&self) -> usize {
        self.features.first().map(|v| v.len()).unwrap_or(0)
    }
}

/// One named slice of training/validation data.
///
/// Multi-group training resolves single-corpus dominance: per-step
/// sampling picks a group in proportion to `train_weight`, then
/// samples a pair within. `train_weight` and `validation_weight` are
/// independent — a group can be in both pools (trained on AND gated
/// against), in only one, or in neither (per-epoch SROCC still
/// logged for transparency).
///
/// Ports the lifetime-borrowed shape from
/// `zensim-validate::mlp_train::TrainingGroup`. The current trainer
/// stores `human_scores` and `features` as borrowed slices so the
/// caller can keep the canonical CSV-derived arrays alive while the
/// trainer iterates. The WASM port will need an owned variant once
/// inputs come from `Vec`-shaped Worker postMessage payloads — we'll
/// add `TrainingGroupOwned` then; for Phase 1 milestone (bit-exact
/// reproduction) we mirror the existing shape verbatim.
#[derive(Debug)]
pub struct TrainingGroup<'a> {
    /// Human-readable name (used in trainer logs).
    pub name: String,
    /// Per-pair quality scores. HIGHER means MORE similar to source.
    pub human_scores: &'a [f64],
    /// Per-pair feature vectors. `features.len() == human_scores.len()`,
    /// and every inner slice has length `n_features` (checked by
    /// `train_mlp` callers).
    pub features: &'a [&'a [f64]],
    /// Weight in the per-step group selection distribution. The
    /// per-pair sampling probability is `train_weight / total_weight`,
    /// so doubling `train_weight` doubles the sampling rate.
    /// Set to `0.0` to exclude this group from training.
    pub train_weight: f64,
    /// Weight in the per-epoch validation aggregation. `0.0` excludes
    /// the group from best-checkpoint scoring (it's still reported in
    /// the log). For [`ValidationPolicy::Min`], weights act as a soft
    /// inclusion mask — any group with `validation_weight > 0`
    /// participates in the min.
    pub validation_weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_hparams_v04() {
        let h = MlpHyperparams::default();
        assert_eq!(h.n_hidden, 32);
        assert_eq!(h.n_epochs, 200);
        assert_eq!(h.validation_policy, ValidationPolicy::Min);
    }

    #[test]
    fn splitmix_seed_stable() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn pearson_perfect_linear() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 3.0).collect();
        let r = pearson(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "expected r≈1, got {r}");
    }

    #[test]
    fn spearman_rank_invariant() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_linear: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let y_monotonic: Vec<f64> = vec![1.0, 100.0, 1000.0, 10000.0, 100000.0];
        assert!((spearman(&x, &y_linear) - 1.0).abs() < 1e-10);
        assert!((spearman(&x, &y_monotonic) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tv_regularizer_n_features_empty() {
        let tv = TvRegularizer {
            pairs: vec![],
            features: vec![],
            weight: 1.0,
            apply_every: 50,
            batch: 32,
        };
        assert_eq!(tv.n_features(), 0);
    }

    #[test]
    fn tv_regularizer_n_features_nonempty() {
        let tv = TvRegularizer {
            pairs: vec![(0, 1), (2, 3)],
            features: vec![
                vec![0.1; 228],
                vec![0.2; 228],
                vec![0.3; 228],
                vec![0.4; 228],
            ],
            weight: 10.0,
            apply_every: 50,
            batch: 32,
        };
        assert_eq!(tv.n_features(), 228);
        assert_eq!(tv.pairs.len(), 2);
    }

    #[test]
    fn training_group_construct() {
        let scores: Vec<f64> = vec![85.0, 70.0, 55.0];
        let feat_rows: Vec<Vec<f64>> = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let feat_refs: Vec<&[f64]> = feat_rows.iter().map(|v| v.as_slice()).collect();
        let g = TrainingGroup {
            name: "synth-test".into(),
            human_scores: &scores,
            features: &feat_refs,
            train_weight: 1.0,
            validation_weight: 0.0,
        };
        assert_eq!(g.human_scores.len(), 3);
        assert_eq!(g.features.len(), 3);
        assert_eq!(g.features[1].len(), 2);
        assert_eq!(g.name, "synth-test");
    }
}
