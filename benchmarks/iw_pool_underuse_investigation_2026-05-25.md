# Why CVVDP/IW-pool features (f300..371) didn't dominate v11 L0 (task #214, 2026-05-25)

## TL;DR

**IW-pool features are structurally redundant with basic+peaks.** 71 of 72 IW-pool features are linear combinations of `basic+peaks` with R² ≥ 0.99 on safesyn (50k-row sample); the most "novel" IW-pool feature (f370) still has R² = 0.989. The trainer correctly notices and routes weight to the canonical basic block. This pattern is invariant across V_18 (ssim2-target), V_22-LARGE (mix_cv40_iw60), V10, and V11 — i.e. it is **not** caused by the v11 training target. Block dominance is a function of the input feature math, not the loss target.

## What the data says

### Per-feature correlation with `mix_cv40_iw60` (safesyn, n=196,086)

| block | n | mean &#124;r&#124; | median &#124;r&#124; | max &#124;r&#124; |
|---|---:|---:|---:|---:|
| basic   (f0..155)   | 156 | 0.3725 | 0.3848 | 0.6826 |
| peaks   (f156..227) |  72 | **0.4784** | **0.4614** | **0.7281** |
| masked  (f228..299) |  72 | 0.4014 | 0.4069 | 0.7215 |
| iw-pool (f300..371) |  72 | 0.3881 | 0.4011 | 0.6516 |

If raw univariate correlation drove L0 mass, **peaks** would win (highest mean and max |r|), not basic. It doesn't — peaks gets 3.25% of v11 L0 mass while basic gets 94.16%. So univariate signal is not the lever.

### Inter-block redundancy (R² of IW-pool features explained by basic+peaks)

Standardized columns, n=50,000 random rows, OLS:

| stat | value |
|---|---:|
| median R² | **0.9980** |
| p25 / p75 | 0.9945 / 0.9993 |
| min R² (most-novel IW feature, f370) | **0.9890** |
| IW features with R² ≥ 0.99 | **71 / 72** |
| IW features with R² ≥ 0.95 | 72 / 72 |
| IW features with R² ≥ 0.90 | 72 / 72 |

IW-pool is **structurally a linear function of basic+peaks**. Once basic+peaks are in the input, IW-pool's marginal information is the residual — which is ≤ 1% of variance per feature. The MLP correctly attaches near-zero L0 weight to features whose information is already on the wire.

Pairwise max-|r| max-out (Pearson, 50k sample): IW vs basic median 0.992; IW vs peaks median 0.974. Pairwise max-|r| also exceeds 1.0 on a few features due to mean/std numerical noise on near-constant columns — the load-bearing number is R² (full multi-regression), which is bounded ≤ 1 and lands at p50=0.998.

### Cross-bake basic-dominance is invariant under target shape

`dump_l0_importance` analog, % of total L0 importance mass per block:

| bake | target | basic | peaks | masked | iw-pool |
|---|---|---:|---:|---:|---:|
| V_18 (228-feat ssim2 ship)            | ssim2 only        | 89.35% | 10.65% | — | — |
| V_22-LARGE (300-feat compression)     | mix_cv40_iw60     | 95.42% |  3.36% | 1.22% | — |
| V_tuner_v10 (372-feat tuner ship)     | safesyn mix       | 96.46% |  1.86% | 0.71% | 0.97% |
| **V_tuner_v11 (372-feat tuner a8)**   | **mix_cv40_iw60** | **94.16%** | **3.25%** | **1.11%** | **1.49%** |

Same pattern at every n_in and every target. **The training target does not move block-share more than ±2 percentage points.** It is a property of how feature blocks are constructed, not the loss.

### Trainer regularization is not the cause

`zensim_mlp_train` has L2 only (no L1). v11 recipe used `--l2 1e-5` (very weak). After standardization the safeguard is `scaler_scale.max(1e-12)`, so all features enter the MLP with unit variance. Zero-variance features would still pass through near-constant — but on safesyn no feature has σ < 2.2e-4 in raw units, so this is not the explanation either.

## Why this answer makes sense

The basic block already encodes per-(scale, channel) SSIM mean/variance, artifact statistics, and detail statistics across 4 scales × 3 channels. The peak block adds max/p95 pooled summaries. The IW-pool block applies information-content weighting to the *same underlying SSIM maps*, and IW-weighting is a linear reweighting before reduction. With 50k+ training pairs, a linear regression can recover the IW-weighted reduction from the unweighted basic stats plus the peak summaries to within ~0.2% variance per feature.

The trainer is **not** ignoring an informative signal — there is nothing left to ignore once basic+peaks are present. It is doing exactly what gradient descent on the dual-regularized MSE objective should do.

## Is this a problem?

Partly yes. The IW block was added to expose **IW-SSIM**'s information-content geometry to the network, on the theory that the `mix_cv40_iw60` target would value features whose pooling matches IW-SSIM's pooling. But the IW-pool we ship is mathematically reducible to a reweighting of the same Gaussian SSIM maps the basic block already exposes. To genuinely surface *novel* IW signal, the block would have to use a **different decomposition** (e.g. an explicit local-information-content map separated from the SSIM map, frequency-band-localized energy ratios, or a non-linear IW transform) — not just a re-pooling of the same stats.

## One experiment that would test the hypothesis directly

**Drop the IW-pool block (f300..371) from the input and retrain v11 at otherwise-identical settings.** If, on the 5-corpus ship gate, the resulting 300-feat bake is within ±0.005 SROCC of v11 across CID22 / KADID / TID / KonJND / AIC-3, the IW-pool block carries no shippable information. If a corpus regresses by > 0.01 SROCC, the residual 1% does matter — and the redesign question becomes "what feature decomposition captures IW-SSIM's signal without the linear-redundancy collapse?" Companion: drop the **peaks** block instead and confirm the 0.989 R² floor predicts the regression direction. Cheap: one seed each, ~30 min wall, no calibration spline rework.

## Data the user should look at

- `/tmp/iw_pool_corr_analysis.log` — full per-feature correlation + redundancy log (this session).
- `/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet` columns `f300..f371` vs `f0..f227`.
- `zensim-experimental/weights/v_tuner_v11_2026-05-24.bin` L0 dump via `cargo run -p zensim-validate --release --example v11_importance`.
- Cross-bake L0 importance pattern in the table above — reproduce via the same example pointed at each bake.
