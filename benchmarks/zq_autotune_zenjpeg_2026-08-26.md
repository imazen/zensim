# Zq autotune — zenjpeg predictor fit (2026-08-26)

**Criterion-4 "zenpredict-baked Zq one-shot predictor (autotune)" — the MODEL, proven.**
Fit a Zq predictor for zenjpeg from the canonical 924 bigcodec data
(`ext924-canonical-2026-07-27/bigcodec/zenjpeg_lossy/{train,validate,test}_924.parquet`,
origin-split, 761,310 train rows). Input = the 924 zensim features (`f0..f923`) + the
**target ssim2** (`score_ssim2`); output = the **q** that achieved it. So at inference:
(image features, desired target-ssim2) → predicted q_start for the loop.

## Result (TEST split, held-out origins)
| predictor | RMSE(q) | within ±5 q | within ±10 q |
|---|--:|--:|--:|
| target-only anchor (no features) | 26.71 | — | — |
| **features + target (ridge λ=10)** | **9.74** | 44% | 74% |

**Features cut the q-prediction error 64%** vs a target-only anchor. A seed within ±10 q
74% of the time meaningfully shortens the bracketed secant loop (fewer encode→decode→score
iterations before it reaches the target band). This is the autotune's core value, proven on
real held-out data.

## Status + follow-on
- **DONE:** the model fit + validation (this doc; coefficients at
  `/mnt/v/zen/zensim-training/zq_zenjpeg_ridge_2026-08-26.npz`, 925×f32). Ridge is the proof;
  a `zensim_mlp_train` MLP would do better (nonlinear feature×target interactions, like
  zenavif's q0_head).
- **FOLLOW-ON (the bake + wire, feature-gated per [[feedback_no_zenpredict_in_codecs]]):**
  (1) train the production form via `zensim_mlp_train` (features+target → q) and bake ZNPR via
  `zenpredict-bake`; (2) wire into `zenjpeg::target_quality` as the `q_start` seed behind an
  `auto-tune` feature (zenjpeg CAN dep zenanalyze — no cycle, verified — to extract features at
  inference; zenavif's `q0_head` uses 8 CHEAP zenanalyze features for speed, the better production
  design vs the full 924 here). CID22 stays validation-only; train on the curated bigcodec sets.
- **Per-codec:** the same recipe applies to zenwebp/zenavif/jxl from their bigcodec views.

## PER-ENCODER generalization (all 4 main codecs, TEST split, 2026-08-26)
Same fit (924 features + target ssim2 → q, ridge λ=10) on each codec's bigcodec 924 view:

| codec | train rows | anchor RMSE(q) | full RMSE(q) | error cut | within ±10 q |
|---|--:|--:|--:|--:|--:|
| zenavif_lossy | 775,152 | 18.46 | **4.41** | **76%** | **96%** |
| zenwebp_lossy | 484,470 | 26.08 | **5.61** | **78%** | **93%** |
| zenjpeg_lossy | 761,310 | 26.71 | **9.74** | **64%** | **74%** |
| zenjxl_lossy  | 726,705 | 24.01 | **9.90** | **59%** | **74%** |

**The Zq autotune model generalizes to every main codec** — a feature-based one-shot q predictor
beats a target-only anchor by 59–78% RMSE, landing within ±10 q on 74–96% of held-out encodes.
zenavif/zenwebp are especially seedable (±10 q on 93–96%). This is criterion-4's "zenpredict-baked
Zq one-shot predictor (autotune), per encoder" — the MODEL, validated. The production form (MLP via
`zensim_mlp_train` → `zenpredict-bake` → wire behind each codec's `auto-tune` feature, using the
cheap 8-feature q0_head design for inference speed) is the mechanical follow-on.
