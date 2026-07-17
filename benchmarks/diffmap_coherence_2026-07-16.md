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

## MEASURED (2026-07-16) — the diffmap is DECENT, not broken (assumption corrected)

Two diagnostics built (`zensim/examples/diffmap_coherence.rs` pooled;
`diffmap_block_coherence.rs` spatial — refine each block to the ref, rescore,
correlate ΔS with the diffmap-block sum). On the shipped B profile:

- **Pooled coherence: SROCC 1.0000, PLCC ~0.98** across images/codecs. The
  diffmap MAGNITUDE tracks the scalar perfectly within a q-sweep — because SSIM
  dominates BOTH the diffmap and the 372-feature scalar, so both fall
  monotonically with q. So the incoherence is NOT in the pooled magnitude.
- **Spatial coherence: SROCC(diffmap_block, ΔS) = 0.20 (q25) / 0.66 (q50) / 0.41
  (q75)**, and the diffmap BEATS SSE-per-block (the codec PSNR default: 0.44 /
  0.18) at q50/q75. So the diffmap partially predicts where refining raises the
  scalar, and is the better spatial predictor than the codec default — but it is
  far from SROCC≈1.

**Key correction to the premise above:** the scalar is NON-ADDITIVE (it pools
features non-linearly across scales + IW-weighting), so refining one block changes
the pooled features non-linearly — NO per-pixel map can perfectly predict per-block
ΔS. The 0.66 ceiling is PARTLY intrinsic to the non-additive scalar, not only the
stale V0_2 weights. So the fix (feed the model's `s_k`) can sharpen the spatial
match but cannot reach 1.0; its marginal value over "already beats SSE" is the open
question. NEXT diagnostic before building the fix: compare `Balanced` (fixed
weights, no params.weights) vs `Trained` (V0_2 weights) spatial SROCC — if they
tie, the weight SOURCE isn't the bottleneck (non-additivity is) and the fix won't
help; if `Trained` differs, the model-sensitivity fix has headroom.

## REWEIGHTING FIX FALSIFIED (2026-07-16) — measured before building

Balanced-vs-Trained spatial SROCC on B @ q50: **Trained (V0_2 weights) 0.656 vs
Balanced (fixed Y-dominant, ignores params.weights) 0.675.** The weight SOURCE
barely matters (Δ0.02, Balanced marginally better). So feeding the model's
per-feature sensitivity `s_k` — a third weighting — would NOT meaningfully improve
spatial coherence. The ~0.66 ceiling is the scalar's NON-ADDITIVITY (non-linear IW
pooling), which no reweighting of the same per-pixel signals can overcome. The
originally-designed fix (§"The fix" above) is therefore NOT worth building.

**What this means for the closed loop:** the diffmap is already (a) magnitude-
coherent (pooled SROCC 1.0), and (b) the best available SPATIAL predictor — it
beats SSE (the codec PSNR default) at predicting where refining raises the scalar
(0.66 vs 0.44 @ q50, 0.41 vs 0.18 @ q75). So a closed loop steered by the diffmap
already outperforms the codec default. The diffmap "matches" as well as a per-pixel
map can for a non-additive scalar.

**If sharper spatial coherence is ever required**, the only lever that can beat the
non-additivity ceiling is the TRUE per-block gradient — literally the ΔS the
diagnostic computes (refine block → rescore) — used AS the map. That is exact but
O(n_blocks) rescores per image; viable as an offline oracle / calibration, not a
per-frame encoder signal. A cheaper middle path: make the scalar more additive
(linear pooling) so the existing diffmap becomes its exact gradient — a metric-
architecture change, not a diffmap change.

Tools: `zensim/examples/diffmap_coherence.rs` (pooled), `diffmap_block_coherence.rs`
(spatial, `--weighting trained|balanced`, `--block N`).

## Sequencing

1. Diagnostic first (quantify the current incoherence for B — the default).
2. Linear-B `ModelCoherent` (fixed `s_k`) — simplest, benefits the current ship.
3. Min-max per-image `s_k` (expose the active piece from the forward).
4. Non-basic feature-block per-pixel signals (IW-pool) if the linear/basic
   version doesn't already pool-match closely enough.

Blocks the closed-loop product until at least (2) lands with the validation gate.
