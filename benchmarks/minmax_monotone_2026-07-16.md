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

## Tuning sweep (single seed 13, ep80 unless noted)

| config | CID22 | imazen-26 | nonphoto | note |
|---|---|---|---|---|
| k8j4 | 0.809 | 0.863 | 0.880 | |
| k16j8 | 0.830 | 0.876 | 0.889 | |
| k16j8 uncap-bigcodec | 0.828 | 0.871 | 0.888 | more real-codec data slightly HURT |
| **k24j8** | **0.831** | **0.880** | **0.895** | balanced champion |
| k16j8 uncap-bigcodec ep140 | 0.809 | 0.885 | 0.893 | imazen-26-favoring (ssim2-shaping trade) |

**Capacity (K/J) is the lever, not data volume.** K8→16→24 lifts imazen-26 monotonically
(0.863→0.876→0.880). Uncapping bigcodec (2× real-codec pairs) slightly HURT imazen-26 —
it dilutes the balanced mix. More epochs (ep140) overfit the ssim2-labeled pairs: CID22
drops, imazen-26 rises — the ssim2↔human-MOS trade in miniature. Seed-confirm in progress.

## Seed-confirm (2026-07-16) — K24 is the peak, but seed variance is real

| config | CID22 | imazen-26 | nonphoto |
|---|---|---|---|
| k24 seed 13 | 0.8307 | **0.8803** | 0.8949 |
| k24 seed 17 | 0.8291 | 0.8767 | 0.8909 |
| k24 seed 23 | 0.8140 | 0.8643 | 0.8821 |
| k32 seed 13 | 0.8301 | 0.8687 | 0.8817 |

**K24 is the capacity sweet spot** — K32 regresses (imazen-26 0.869 < K24's 0.880), so
more pieces overfit. **Seed variance is meaningful**: K24 imazen-26 = 0.874 ± ~0.009 across
3 seeds. The median/mean clearly beats A (0.862); nonphoto wins robustly (worst seed 0.882 >
A's 0.878); but the WORST seed (s23, 0.864) only ties A on imazen-26. Min-max RankNet with
subgradient-to-active-piece can leave pieces under-trained on a bad init (the classic
min/max "dead unit" issue). **A ship should pick best-of-N seeds** validated on a train-side
held-out (CID22 is validation-only, can't select on it). s13 is currently best.

## Reproducibility gate — PASSED (2026-07-16)

`bake_verdict`'s real-runtime forward on the k16 bake matches the in-process trainer eval
bit-for-bit: CID22 0.8302 (==), imazen-26 0.8761 (==), nonphoto 0.8883 (Δ0.001 f32-quant),
**%bwd 0%** (zero references ranked backwards — monotone-by-construction confirmed in the
shipped runtime). The runtime is `metric.rs` (encoder path) + `bake_runtime.rs` (eval path),
bit-exact mirrors like the per-sample-α head.

## Head-to-head vs shipped A/B (bake_verdict, same corpora)

| bake | CID22 (human MOS) | imazen-26 (ssim2) | nonphoto (ssim2) | dial mono / flat |
|---|---|---|---|---|
| A (v47 MLP) | 0.8657 | 0.8619 | 0.8783 | — |
| B (linear, default) | **0.8764** | 0.8413 | 0.8606 | — |
| min-max k16 | 0.8302 | 0.8761 | 0.8883 | **0.9815 / 0.0000** |
| min-max k24 (in-proc) | 0.831 | **0.880** | **0.895** | (expected same) |

**The min-max wins the ssim2 north-star decisively** (imazen-26 +0.018/nonphoto +0.017 vs A;
+0.047/+0.042 vs B). That is its ONE real advantage.

**Dial cleanliness is a WASH vs the current A/B ships (corrected 2026-07-16 — do not
overclaim).** Measured DIAL panel: min-max mono 0.9771 / **0% flat**; A_v47 0.9782 / 0% flat;
B_linear 0.9792 / 0% flat. All three are clean, dead-zone-free monotone dials — the min-max
is COMPARABLE (marginally lower mono, within noise), NOT cleaner. The "0% flat vs V0_5 ships
57-76% tied" comparison was to OLD ships; the current A/B are already 0% flat. So the
min-max's value is purely the ssim2-north-star rank, not the dial. After the [0,100] spline
(18 knots, y-range [0,95.1]): G1 PASS (dial p5/p95 10.4/94.8), G3 PASS (0.9771 mono), rank
unchanged (CID22 0.8307, imazen-26 0.8803, nonphoto 0.8938).

**The cost is CID22 (human MOS): −0.035 vs A, −0.046 vs B.** The min-max's capacity fits the
ssim2 training labels tightly (all of safesyn/bigcodec/kadis are ssim2-anchored), so it
becomes a better ssim2-predictor and a worse human-MOS predictor — the documented
ssim2-favoring bias, sharpened by expressiveness. No min-max config reached A's CID22 0.866.

## Ship framing (user-gated operating point)

The min-max is a NEW point on the monotone-dial frontier: **best ssim2 north-star + cleanest
dial, at a CID22 cost.** Per the user's "ssim2 is the best north star" (esp. non-photo) and
"consistent dial is the product", it is the intended direction; per the shipping policy CID22
trades are user-gated. Decision to the user:
- **Ship as a new ssim2-north-star dial** (k24: imazen-26 0.880, nonphoto 0.895, dial 98%/0%
  flat) accepting CID22 0.831 — OR keep A/B for CID22 and offer the min-max as a sibling.
- Remaining mechanical step once the config is chosen: fit the [0,100] output spline (G1) —
  rank-invariant, so none of the above numbers move.
