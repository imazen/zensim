# Min-max monotone dial architecture (Sill 1998) — 2026-07-16

## Why

Users type a target zensim score; codecs binary-search a quality that hits it,
using the diffmap to close the loop. That REQUIRES the metric be a **consistent
dial**: monotone in codec quality (so the search converges) and bounded [0,100]
(so the target is reachable). The tension we've hit repeatedly:

- **Positive-weight LeakyReLU MLP** (`monotone_cbc`): monotone, but the
  positive-weight constraint craters real-content ssim2-agreement —
  **imazen-26 SROCC 0.032**. Useless as a ranker.
- **Unconstrained MLP** (`base_tfm` 0.948, `depth_v2` 0.956): wins real content,
  but the dial is non-monotone / dead (base_tfm G3 mono 0.929, collapses in
  bands) — a codec can't target it.
- **Masked-monotone linear** (shipped `A`): monotone dial, imazen-26 **0.862** —
  the empirical monotone ceiling for a *linear* score.

## The architecture

Sill (1998) min-max networks are **universal approximators for monotone
functions** — the expressiveness `monotone_cbc` lost, recovered without giving
up monotonicity:

```
score(x) = min over K groups  max over J pieces  ( w[g][h] · x + b[g][h] )
```

with each weight sign-constrained per the feature-sign mask: `sign[f]·w[g][h][f] ≥ 0`.
Every linear piece is then monotone in codec quality; max/min of monotone
functions is monotone → **`score` is monotone BY CONSTRUCTION, for arbitrary
weights** (proved by `minmax_monotone::tests::monotone_by_construction`). No dial
regularizer, no monotonicity penalty, no post-hoc iso — the dial is monotone the
moment the bytes are projected.

Owner: `zensim-validate/src/mlp_train/minmax_monotone.rs` (`MinMaxMonotone`,
`train_ranknet`, `project`). Probe binary: `zensim-validate/src/bin/train_minmax.rs`.

### Sign convention (the bug that cost the first run)

The feature-sign mask `feature_sign_mask_2026-05-26.tsv` marks 300 features
`pin_geq0` = "W1 ≥ 0 in the trainer", which the trainer pairs with `rank_w ≤ 0`.
So those features **increase with DISTORTION** (decrease with quality). A
score-INCREASING min-max therefore needs `w ≤ 0` on them → `sign = −1`, not +1.
The first PoC used +1 and got NEGATIVE SROCC (CID22 −0.07). Fixed in
`train_minmax::load_sign` (`s[idx] = -1.0`). The 72 sign-0 features are dropped
(`w ≡ 0`) so monotonicity is exact.

## PoC result (the reason to pursue this)

`train_minmax --k 8 --j 4 --epochs 60 --pairs 50000 --lr 4e-3`, base_tfm groups +
2026-05-25 screen transforms + standardize, held-out:

| model | imazen-26 | CID22 | nonphoto | monotone? |
|---|---|---|---|---|
| monotone_cbc MLP | 0.032 | — | — | yes |
| A (masked-monotone linear) | 0.862 | 0.866 | — | yes |
| **min-max K8J4 (ep60)** | **0.874** | 0.848 | 0.889 | **yes, by construction** |
| base_tfm (unconstrained) | 0.948 | — | — | **no (broken dial)** |

First model with BOTH a consistent dial AND above-ceiling real-content. Untuned.

**Epoch sensitivity (measured):** the same K8J4 at ep80 dropped to imazen-26
0.863 / CID22 0.809 — more epochs OVERFIT the dominant safesyn pairs (pair
sampling is ∝ row count, safesyn 40k dominates). The real lever is the DATA MIX,
not more training: uncapping `bigcodec` shifts pair-sampling toward real-codec
cells, which is what imazen-26 measures. Sweep in progress:
`/mnt/v/output/zensim/minmax-sweep/summary.tsv`.

## Bake + runtime path (designed, not yet built)

The min-max REPLACES the MLP layers, so its runtime can't get a hidden vector
from `Predictor::predict`. Two options considered:

- **Identity-layer** (out = standardized features via a 372×372 identity, then a
  metadata head — exactly the `per_sample_alpha_head` pattern): zero zenpredict
  change, but 553 KB of identity bloat.
- **Bypass** (CHOSEN): `Model` already exposes `scaler_mean/scaler_scale/
  feature_transforms/feature_transform_params`. The min-max path reads those,
  applies transform→scale→clamp±8→min-max→spline directly, ignoring layers. Tiny
  bake (w is K·J·372 f32 ≈ 190 KB at K16J8), single source of truth for
  scaler/transforms.

To avoid re-implementing transform+scale (banned duplication), **extend zenpredict**
with `Predictor::transform_and_scale(features) -> &[f32]` — the transform+scale
prefix of `predict_transformed` (scalar path only; the screen transforms are all
scalar). zensim's min-max path then: `transform_and_scale` → clamp ±8 (the one
thing the PoC does that the runtime forward doesn't) → `apply_minmax_runtime` →
reuse `apply_output_calibration_spline` for the [0,100] dial.

Bake bytes (ZNPR v3, via `zenpredict_bake::bake` / JSON pipeline):
- `n_inputs = 372`, `scaler_mean[372]`, `scaler_scale[372]` = training mean/std.
- `feature_transforms[372]` + params = the 2026-05-25 screen transforms.
- one dummy 372→1 layer (satisfies the parser; the min-max path ignores it).
- metadata `zentrain.minmax_monotone_head` = `[K:u32, J:u32, N:u32,
  w[K·J·N]:f32, b[K·J]:f32]`.
- metadata `zentrain.output_calibration_spline` = the PCHIP [0,100] dial,
  fit on the PROJECTED+quantized net (the QUANTIZE-then-CALIBRATE rule).

Runtime dispatch site: `forward_one_bake_with_codec` in `zensim/src/metric.rs`,
a new branch alongside `per_sample_alpha` / `hybrid_head`, gated on the
`zentrain.minmax_monotone_head` metadata key (parsed + cached in
`CachedBakeMetadata`).

## Gate to ship (SOTA_TRAILS dial trail)

Monotone-by-construction → G1 (range) + G3 (mono ≥ 0.93) pass by design once the
spline bounds it. The bar vs shipped A/B: imazen-26 (ssim2 north-star, real +
nonphoto) at-or-above A's 0.862 AND CID22 not decisively worse. If a tuned config
holds imazen-26 ≥ 0.90 with CID22 ≥ 0.86, it dominates A as a dial. Ship decision
is user-gated per the profile-rotation policy.
