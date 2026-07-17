# Diffmap↔scalar coherence for the closed loop — 2026-07-16

User directive: "diffmap absolutely needs to match so closed loops work." The
closed loop is: user picks a target zensim → codec binary-searches quality to hit
the SCALAR score, then uses the DIFFMAP to decide WHERE to spend bits. If the
diffmap points at different pixels than the scalar actually cares about, the
per-block refinement fights the target instead of serving it.

## The incoherence (confirmed in source)

- **Scalar** `ZensimResult::score()` for the shipped profiles (A/B/min-max) comes
  from the 372-feature model (`mlp_bytes` → MLP / linear / min-max forward).
- **Diffmap** `DiffmapResult::diffmap()` is built by
  `diffmap::compute_with_ref_and_diffmap` → `DiffmapWeighting::Trained.
  resolve_multiscale(params.weights, …)` → `trained_multiscale_weights`, i.e. it
  is weighted by **`params.weights`** — the static `WEIGHTS_PREVIEW_V0_2` SSIM
  vector, which `profile.rs` itself documents as "unused on the MLP path but
  kept". So the diffmap reflects the OLD V0_2 linear-SSIM model, NOT the shipped
  scalar. For B/min-max the two disagree by construction.

## What "coherent" means

The diffmap at pixel (x,y) should be `∂score/∂distortion(x,y)` — how much reducing
local distortion there raises the scalar. For a feature vector `f = [f_0..f_371]`
where each `f_k` is a spatially-pooled signal `f_k = pool_k(local_k(x,y))`:

```
∂score/∂distortion(x,y) = Σ_k (∂score/∂f_k) · (∂f_k/∂local_k) · local_k(x,y)
```

For a mean-pooled feature `∂f_k/∂local_k = 1/N` (uniform), so a coherent diffmap is
`Σ_k s_k · local_k(x,y)` where `s_k = ∂score/∂f_k` is the model's per-feature
sensitivity. The `local_k(x,y)` per-pixel signals are exactly what the diffmap
already computes (ssim error, edge artifact/detail, mse, hf loss/mag/gain per
scale per channel = the `PixelFeatureWeights` fields). **Only the weights `s_k`
are wrong** — it uses `params.weights` instead of the model's `s_k`.

## The fix

Replace the weight source fed to `trained_multiscale_weights` with the model's
per-feature sensitivity `s_k`, derived from the bake:

- **Linear bake (B, linear-projection family):** `s_k = w_k / scaler_scale_k`
  (the linear weight, un-standardized). FIXED per bake — compute once. Directly
  maps onto the basic-feature block that `trained_multiscale_weights` already
  indexes (scale × channel × 13). The IW-pool / peak / masked feature blocks
  (f156..f371) need their own per-pixel signal definitions OR are folded into the
  scale they summarize.
- **Min-max bake:** `s_k = w[g*][h*][k] / scaler_scale_k` where `(g*,h*)` is the
  active (argmin-group, argmax-piece) for THIS image's feature vector — so the
  diffmap weights are per-image (the forward already finds `(g*,h*)`; expose it).
- **MLP bake:** `s_k = ∂score/∂f_k` via one backward pass through the small MLP
  (per-image). Heavier but exact.

Implementation shape (additive, does not disturb existing `Trained`/`Balanced`/
`Custom`): a new `DiffmapWeighting::ModelCoherent` (or make `Trained` read the
bake's sensitivity when `mlp_bytes` is present) that:
1. loads the bake, computes `s_k` (linear: once; min-max/MLP: per-image),
2. maps `s_k` onto `PixelFeatureWeights` per (scale, channel, signal) via the
   SAME index arithmetic `trained_multiscale_weights` uses for the basic block,
3. defines per-pixel signals for the non-basic feature blocks (IW-pool etc.) or
   documents that they fold into their scale's ssim/hf channels.

## Validation gate (the coherence metric)

Build a diagnostic: for a q-sweep of representative images, compute the scalar
score AND `mean(diffmap)`; a coherent diffmap must satisfy
`pooled_diffmap ≈ affine(100 − score)` with high SROCC/PLCC across the sweep
(the pooled per-pixel sensitivity IS the total distance up to the pooling
constant). Report SROCC(pooled_diffmap, 100−score) per profile:
- Current `Trained` diffmap vs B/min-max scalar → expected LOW (the incoherence).
- `ModelCoherent` diffmap vs same scalar → expected ≈ 1.0.
This is the ship gate for the closed-loop feature and the falsification test for
the fix.

## Sequencing

1. Diagnostic first (quantify the current incoherence for B — the default).
2. Linear-B `ModelCoherent` (fixed `s_k`) — simplest, benefits the current ship.
3. Min-max per-image `s_k` (expose the active piece from the forward).
4. Non-basic feature-block per-pixel signals (IW-pool) if the linear/basic
   version doesn't already pool-match closely enough.

Blocks the closed-loop product until at least (2) lands with the validation gate.
