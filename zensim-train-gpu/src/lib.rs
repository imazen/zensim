//! GPU-accelerated per-sample-α head trainer (Phase 1).
//!
//! This crate is purely additive: when the parent crate is built without
//! any `gpu-*` feature, the public entry point [`train_per_sample_alpha_head_gpu`]
//! still exists but panics with a clear "no GPU backend compiled in" message.
//! The CPU code path in `zensim-validate::mlp_train` is unaffected.
//!
//! ## Scope (Phase 1 MVP per task #166, 2026-05-19)
//!
//! - Architecture: per-sample-α head, identical to
//!   [`zensim_train_core::per_sample_alpha_head::PerSampleAlphaHeadModel`].
//! - Losses: RankNet pair loss + optional MSE pair loss + optional
//!   monotonicity hinge. **No** auxiliary losses (anchor, cross-codec-eq,
//!   σ-floor probe, rank-preserve) — those are Phase 2.
//! - Adam optimizer: per-parameter (m, v) state, same beta1=0.9, beta2=0.999,
//!   eps=1e-8 as the CPU [`zensim_train_core::adam::AdamState`].
//! - K-batched: default K=512. Each minibatch step uploads 2K standardized
//!   feature rows, runs forward + backprop + Adam update on GPU.
//! - F32 throughout (CPU uses f64). Quality target: within ±0.005 final
//!   SROCC vs CPU baseline.
//! - Tanh-pin output head supported via `tanh_output_head_scale` hparam.
//!
//! ## Phase 2 (not in this crate yet)
//!
//! - Anchor loss kernel (per-anchor-row weighted MSE toward target_score)
//! - Cross-codec-eq loss kernel
//! - σ-floor probe (per-ref σ across K probe rows)
//! - Rank-preserve loss
//! - Norm-in-Norm K-batch path
//! - Bit-exact f64 path (currently f32 on GPU)
//!
//! ## CubeCL primitive notes (for future Phase 2 work)
//!
//! - `Atomic<f32>::fetch_add` works on CUDA, silently no-ops on wgpu
//!   Metal. We accumulate gradients via atomic-add into a single
//!   `Array<Atomic<f32>>` buffer per parameter group.
//! - cubecl 0.10 does not (yet) ship a built-in GEMM primitive at the
//!   Array level; we hand-roll the two matmuls (forward + backprop_w1)
//!   as element-wise kernels. cubecl-matmul exists but operates on
//!   TensorMap which is heavier than needed for batch sizes ≤ 1024.
//! - LeakyReLU + sigmoid/tanh are scalar element-wise; pure Rust math
//!   inside `#[cube]` works (.exp(), .sqrt(), .abs(), .powi() — but
//!   .powi requires explicit f32 cast).

// CubeCL's `ArrayArg::from_raw_parts` API is `unsafe` (it asserts the
// handle's length matches the declared element count). We localize the
// `unsafe` blocks to the launch sites in `backend.rs`; everywhere else
// stays safe.
#![allow(unsafe_code)]
#![warn(missing_docs)]

use zensim_train_core::TrainingGroup;
use zensim_train_core::per_sample_alpha_head::PerSampleAlphaHeadModel;

/// GPU runtime selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuRuntime {
    /// CUDA via cubecl-cuda. Requires `gpu-cuda` cargo feature.
    Cuda,
    /// WGPU (Vulkan / Metal / DX12). Requires `gpu-wgpu` cargo feature.
    /// Note: f32 atomics behave differently across vendors — CUDA OK,
    /// Metal silently drops. CUDA is the Phase 1 supported backend.
    Wgpu,
    /// CubeCL CPU backend (tracel-mlir). Requires `gpu-cpu` feature.
    /// Intended for parity testing only — actual perf will be CPU-bound.
    Cpu,
}

impl GpuRuntime {
    /// Parse from the `--gpu-runtime` CLI flag.
    pub fn from_cli(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "cuda" => Ok(GpuRuntime::Cuda),
            "wgpu" => Ok(GpuRuntime::Wgpu),
            "cpu" => Ok(GpuRuntime::Cpu),
            other => Err(format!(
                "--gpu-runtime must be one of cuda / wgpu / cpu, got {other:?}"
            )),
        }
    }
}

/// Hyperparameters for the GPU per-sample-α trainer.
///
/// A subset of the CPU [`zensim_train_core::per_sample_alpha_head::PerSampleAlphaHeadHparams`]
/// plus the auxiliary loss weights supported in Phase 1
/// (MSE pair loss, monotonicity hinge) and the **Phase 2** aux losses
/// (anchor, cross-codec-eq, σ-floor, rank-preserve).
#[derive(Clone, Debug)]
pub struct GpuHparams {
    /// Hidden vector width. Power-of-2 (128, 256) recommended for kernel layout.
    pub n_hidden: usize,
    /// Number of epochs.
    pub n_epochs: usize,
    /// RankNet pair samples per epoch. Will be rounded up to the next
    /// multiple of `minibatch_k`.
    pub pairs_per_epoch: usize,
    /// Pairs per GPU minibatch. Default 512. Higher = better GPU saturation
    /// but more VRAM. 128 × 372 × 512 ≈ 24 MB scratch.
    pub minibatch_k: usize,
    /// Adam initial learning rate.
    pub initial_lr: f64,
    /// LeakyReLU negative-side slope.
    pub leaky_alpha: f64,
    /// PRNG seed.
    pub seed: u64,
    /// L2 regularization on w1 + rank_w + reducer_w + W_α. b_α unregularized.
    pub l2_lambda: f64,
    /// MSE pair-loss weight (per pair: `mse_weight · (yhi − ylo − Δscore)²`).
    /// 0 disables.
    pub mse_weight: f64,
    /// RankNet pair-loss weight. Default 1.0.
    pub ranknet_weight: f64,
    /// Monotonicity hinge penalty (per pair: `λ · max(0, margin − (yhi − ylo))`).
    /// 0 disables.
    pub monotonicity_reg: f64,
    /// Monotonicity margin (in score units).
    pub monotonicity_margin: f64,
    /// Tanh-output-head pin scale. When > 0, wraps the raw pre-output in
    /// `y_score = 100 · σ(y_pre / scale)`. Same semantics as the CPU
    /// `tanh_output_head_scale` flag. 0 = linear output.
    pub tanh_output_head_scale: f64,

    // ---- Phase 2 aux loss hparams (task #169, 2026-05-19) ----
    /// Anchor MSE loss weight. Per anchor row: `w · row_w · (y − target)²`.
    /// 0 disables (anchor data is ignored even if supplied).
    pub anchor_loss_weight: f64,
    /// Probability per minibatch step that the anchor kernel fires.
    /// Each fire processes `minibatch_k_aux` rows.
    pub anchor_step_p: f64,
    /// Cross-codec equivalence loss weight (`w · row_w · (y_a − y_b)²`).
    /// 0 disables.
    pub cross_codec_eq_weight: f64,
    /// Probability per minibatch step that the eq kernel fires.
    pub cross_codec_eq_step_p: f64,
    /// Rank-preserve auxiliary on equivalence pairs (butter-weighted
    /// RankNet-style log-loss). 0 disables.
    pub cross_codec_rank_preserve_weight: f64,
    /// Dynamic-range floor (σ-floor) weight. Penalizes the network when
    /// the σ across `dynamic_range_probe_n` random equiv-A rows falls
    /// below `dynamic_range_sigma_threshold`. 0 disables.
    pub dynamic_range_floor_weight: f64,
    /// σ-floor probe count per fire. Default 40 — match CPU.
    pub dynamic_range_probe_n: usize,
    /// σ threshold for the dynamic-range probe (score units).
    pub dynamic_range_sigma_threshold: f64,
    /// Probability per minibatch step that the σ-floor probe fires.
    pub dynamic_range_step_p: f64,

    /// K samples per aux-loss fire. Default 32 — K-batched semantics on
    /// GPU. CPU is K=1 per fire; GPU amortizes across larger K so each
    /// kernel launch is worth the overhead.
    pub minibatch_k_aux: usize,

    // ---- EXP-V11-D-PJND-DOMINANT (2026-05-20, task #198) ----
    /// PJND-passthrough anchor loss weight (second anchor pool, distinct
    /// from `anchor_loss_weight`). Per row: `w · row_w · (y − target)²`.
    /// 0 disables (data ignored even if supplied).
    pub pjnd_passthrough_weight: f64,
    /// Probability per minibatch step that the PJND-passthrough kernel
    /// fires. Each fire processes `minibatch_k_aux` rows.
    pub pjnd_passthrough_step_p: f64,
}

impl Default for GpuHparams {
    fn default() -> Self {
        Self {
            n_hidden: 128,
            n_epochs: 200,
            pairs_per_epoch: 50_000,
            minibatch_k: 512,
            initial_lr: 1e-3,
            leaky_alpha: 0.01,
            seed: 1,
            l2_lambda: 1e-5,
            mse_weight: 1.0,
            ranknet_weight: 1.0,
            monotonicity_reg: 0.0,
            monotonicity_margin: 1.0,
            tanh_output_head_scale: 0.0,
            anchor_loss_weight: 0.0,
            anchor_step_p: 0.10,
            cross_codec_eq_weight: 0.0,
            cross_codec_eq_step_p: 0.10,
            cross_codec_rank_preserve_weight: 0.0,
            dynamic_range_floor_weight: 0.0,
            dynamic_range_probe_n: 40,
            dynamic_range_sigma_threshold: 15.0,
            dynamic_range_step_p: 0.10,
            minibatch_k_aux: 32,
            pjnd_passthrough_weight: 0.0,
            pjnd_passthrough_step_p: 0.30,
        }
    }
}

/// Per-row anchor pool — feature vectors + per-row weight + per-row
/// target score. Phase 2 GPU mirror of the CPU `AnchorRows` struct.
///
/// `features.len() == row_weights.len() == target_scores.len()` and
/// every inner feature slice has length `n_features`.
#[derive(Clone, Debug)]
pub struct GpuAnchorRows<'a> {
    /// Human-readable pool name (used in trainer logs).
    pub name: String,
    /// One slice per anchor row, each `n_features` long.
    pub features: &'a [&'a [f64]],
    /// Per-row weight in the anchor MSE step.
    pub row_weights: &'a [f64],
    /// Per-row target score. CPU uses a global fallback when its
    /// `target_scores` field is `None`; on GPU we always require a
    /// per-row target (callers can pass a constant vector).
    pub target_scores: &'a [f64],
}

/// Per-pair cross-codec equivalence pool — A-side + B-side features +
/// per-pair weight + per-pair butter_diff. Phase 2 GPU mirror of the
/// CPU `EquivPairs` struct.
///
/// `features_a.len() == features_b.len() == row_weights.len()`.
/// `butter_diff` may be empty (rank-preserve disabled) or the same
/// length as `features_a`.
#[derive(Clone, Debug)]
pub struct GpuEquivPairs<'a> {
    /// Pool name (used in trainer logs).
    pub name: String,
    /// One feature slice per pair (A-side), each `n_features` long.
    pub features_a: &'a [&'a [f64]],
    /// One feature slice per pair (B-side), each `n_features` long.
    pub features_b: &'a [&'a [f64]],
    /// Per-pair weight in the equiv MSE step.
    pub row_weights: &'a [f64],
    /// Per-pair `butter_a - butter_b` (butteraugli-pnorm3 units).
    /// Empty slice disables rank-preserve.
    pub butter_diff: &'a [f64],
}

/// Result of a GPU training run.
#[derive(Debug)]
pub struct GpuTrainResult {
    /// Final model with all weights, biases, and pre-computed standardizer.
    /// Ready for [`zensim_train_core::per_sample_alpha_head::bake_per_sample_alpha_head_v3`].
    pub model: PerSampleAlphaHeadModel,
    /// Wall-clock seconds for the training loop (excludes data load + bake).
    pub wall_seconds: f64,
    /// Number of K-batches actually run.
    pub n_batches: usize,
}

/// Main GPU trainer entry point.
///
/// Behavior depends on cargo features:
/// - With `gpu-cuda` (or `gpu-wgpu` / `gpu-cpu`): runs the full GPU
///   training loop, returns the final model + wall time.
/// - Without any `gpu-*` feature: panics with a clear error message.
///
/// `runtime` selects the cubecl backend; the requested backend must
/// be compiled in via its cargo feature.
///
/// `groups` follows the same shape as the CPU trainer — training data
/// is per-group, with per-pair sampling proportional to `train_weight`.
/// Per-row pair sampling within each group uses the same SplitMix64 seed
/// as the CPU path, but the actual pair sequence will differ because GPU
/// minibatches sample K pairs at a time (CPU samples one).
///
/// **Phase 1 entry**: auxiliary losses (anchor / cross-codec-eq /
/// σ-floor / rank-preserve) are NOT applied even if their hparam
/// values are set. Use [`train_per_sample_alpha_head_gpu_with_aux`]
/// to opt into Phase 2 aux loss kernels.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-cpu"))]
pub fn train_per_sample_alpha_head_gpu(
    groups: &[TrainingGroup<'_>],
    hp: &GpuHparams,
    n_features: usize,
    runtime: GpuRuntime,
) -> GpuTrainResult {
    backend::dispatch(groups, hp, n_features, runtime, None, None, None)
}

/// Phase 2 entry point — same as [`train_per_sample_alpha_head_gpu`]
/// but accepts optional anchor + cross-codec-equivalence pools.
///
/// The σ-floor probe shares its sample substrate with the equiv pool's
/// A-side (matching the CPU path's behavior), so it's gated on
/// `equiv.is_some()` AND `hp.dynamic_range_floor_weight > 0`.
///
/// Per the K-batched semantics described in the Phase 2 plan: each
/// minibatch step has a per-aux-kernel Bernoulli trial against
/// `*_step_p`; on a hit, `hp.minibatch_k_aux` samples are drawn
/// (`minibatch_k_aux × 2` for cross-codec-eq) and one forward +
/// backprop pass adds gradients into the shared parameter grad
/// buffers before the minibatch's single Adam step.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-cpu"))]
pub fn train_per_sample_alpha_head_gpu_with_aux(
    groups: &[TrainingGroup<'_>],
    hp: &GpuHparams,
    n_features: usize,
    runtime: GpuRuntime,
    anchor: Option<&GpuAnchorRows<'_>>,
    equiv: Option<&GpuEquivPairs<'_>>,
) -> GpuTrainResult {
    backend::dispatch(groups, hp, n_features, runtime, anchor, equiv, None)
}

/// EXP-V11-D-PJND-DOMINANT (2026-05-20, task #198) entry — same as
/// [`train_per_sample_alpha_head_gpu_with_aux`] but accepts a SECOND
/// anchor pool (`pjnd_anchor`) for the KonJND-PJND passthrough loss.
/// The pjnd pool fires independently with `hp.pjnd_passthrough_step_p`
/// and `hp.pjnd_passthrough_weight`; both anchor pools may fire on the
/// same minibatch step and contribute additively to the gradient buffer
/// before the Adam update.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-cpu"))]
pub fn train_per_sample_alpha_head_gpu_with_aux_pjnd(
    groups: &[TrainingGroup<'_>],
    hp: &GpuHparams,
    n_features: usize,
    runtime: GpuRuntime,
    anchor: Option<&GpuAnchorRows<'_>>,
    equiv: Option<&GpuEquivPairs<'_>>,
    pjnd_anchor: Option<&GpuAnchorRows<'_>>,
) -> GpuTrainResult {
    backend::dispatch(groups, hp, n_features, runtime, anchor, equiv, pjnd_anchor)
}

/// Stub for builds without any `gpu-*` feature.
#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-cpu")))]
pub fn train_per_sample_alpha_head_gpu(
    _groups: &[TrainingGroup<'_>],
    _hp: &GpuHparams,
    _n_features: usize,
    _runtime: GpuRuntime,
) -> GpuTrainResult {
    panic!(
        "zensim-train-gpu: no GPU backend compiled in. \
         Rebuild with --features gpu-cuda (or gpu-wgpu / gpu-cpu)."
    );
}

/// Stub for builds without any `gpu-*` feature (Phase 2 entry point).
#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-cpu")))]
pub fn train_per_sample_alpha_head_gpu_with_aux(
    _groups: &[TrainingGroup<'_>],
    _hp: &GpuHparams,
    _n_features: usize,
    _runtime: GpuRuntime,
    _anchor: Option<&GpuAnchorRows<'_>>,
    _equiv: Option<&GpuEquivPairs<'_>>,
) -> GpuTrainResult {
    panic!(
        "zensim-train-gpu: no GPU backend compiled in. \
         Rebuild with --features gpu-cuda (or gpu-wgpu / gpu-cpu)."
    );
}

/// Stub for builds without any `gpu-*` feature (PJND-passthrough entry).
#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-cpu")))]
pub fn train_per_sample_alpha_head_gpu_with_aux_pjnd(
    _groups: &[TrainingGroup<'_>],
    _hp: &GpuHparams,
    _n_features: usize,
    _runtime: GpuRuntime,
    _anchor: Option<&GpuAnchorRows<'_>>,
    _equiv: Option<&GpuEquivPairs<'_>>,
    _pjnd_anchor: Option<&GpuAnchorRows<'_>>,
) -> GpuTrainResult {
    panic!(
        "zensim-train-gpu: no GPU backend compiled in. \
         Rebuild with --features gpu-cuda (or gpu-wgpu / gpu-cpu)."
    );
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-cpu"))]
mod backend;

#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu", feature = "gpu-cpu"))]
mod kernels;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_hparams_match_cpu() {
        let h = GpuHparams::default();
        assert_eq!(h.n_hidden, 128);
        assert_eq!(h.n_epochs, 200);
        assert_eq!(h.pairs_per_epoch, 50_000);
        assert_eq!(h.minibatch_k, 512);
        assert!((h.initial_lr - 1e-3).abs() < 1e-12);
        assert!((h.leaky_alpha - 0.01).abs() < 1e-12);
        assert!((h.l2_lambda - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn parse_runtime_flag() {
        assert_eq!(GpuRuntime::from_cli("cuda"), Ok(GpuRuntime::Cuda));
        assert_eq!(GpuRuntime::from_cli("CUDA"), Ok(GpuRuntime::Cuda));
        assert_eq!(GpuRuntime::from_cli("wgpu"), Ok(GpuRuntime::Wgpu));
        assert_eq!(GpuRuntime::from_cli("cpu"), Ok(GpuRuntime::Cpu));
        assert!(GpuRuntime::from_cli("rocm").is_err());
    }
}
