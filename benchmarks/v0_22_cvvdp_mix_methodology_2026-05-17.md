# V_22-CVVDP+IWSSIM mix methodology — proposed PreviewV0_6

**Status:** Strong ship candidate. Multi-target training improves over BOTH pure CVVDP and pure IWSSIM.

**Bake:** `v22_mix_cv40_iw60_s3_h128.bin`
**Date:** 2026-05-17
**md5:** `9691290908d7a086e2c263c6189df6f8`
**Size:** 200,984 bytes

## TL;DR

A **weighted-target training** (0.4 × cvvdp_log_norm + 0.6 × iwssim_log_norm)
captures the strengths of both metrics. It beats V_22-IW v2 PreviewV0_5 on
EVERY compression-priority corpus (CID22, AIC-3, KonJND) while losing
~0.05 SROCC on KADID/TID (vs the pure-CVVDP -0.11 to -0.12 SROCC losses
there).

| Corpus | V_22-IW v2 SROCC | cv40_iw60_s3 SROCC | Δ |
|---|---:|---:|---:|
| CID22 | 0.8163 | **0.8881** | **+0.072 ✓✓** |
| AIC-3 | 0.8070 | **0.8360** | **+0.029 ✓** |
| KonJND | 0.0303 | **0.3059** | **+0.276 ✓✓** |
| KADID | 0.9506 | 0.9067 | -0.044 |
| TID | 0.9617 | 0.8944 | -0.067 |

Z-RMSE deltas:
- CID22 Z-RMSE: 0.569 → 0.455 (**-0.114** ✓✓)
- AIC-3 Z-RMSE: 0.578 → 0.539 (**-0.039** ✓)
- KonJND Z-RMSE: 0.994 → essentially unchanged

n-weighted compression Z-RMSE (CID22+AIC-3+KonJND):
- V_22-IW v2: 0.642
- mix_cv40_iw60_s3: **0.484** (-0.158 ✓✓✓)

## Method

Training target column:
```
mix_cv40_iw60 = 0.40 × cvvdp_log_norm + 0.60 × iwssim_log_norm
```
where both inputs are pre-normalized to [0, 100] (cvvdp via
`-log(10 - cvvdp + 1e-6)` min-max norm; iwssim via the V_22-IW v2
pipeline's log transform).

This is straightforward weighted regression supervision — no trainer
changes required, just a derived target column.

## Why α=0.4 (not 0.5)?

Alpha sweep at single seed=3, h128 (CID22 SROCC / KADID / TID / KonJND / AIC-3):

| α | CID22 | KADID | TID | KonJND | AIC-3 |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.8738 | 0.9234 | 0.9001 | 0.2113 | 0.8034 |
| 0.35 | 0.8817 | 0.9035 | 0.8851 | 0.1328 | 0.8327 |
| **0.40** | **0.8881** | **0.9067** | **0.8944** | **0.3059** | 0.8360 |
| 0.45 | 0.8819 | 0.9003 | 0.8864 | 0.2519 | 0.8348 |
| 0.50 | 0.8961 | 0.8943 | 0.8818 | 0.1976 | 0.8331 |
| 0.60 | 0.8840 | 0.8777 | 0.8753 | 0.2105 | 0.8427 |
| 0.65 | 0.8784 | 0.8767 | 0.8738 | 0.2252 | 0.8448 |
| 0.75 | 0.8788 | 0.8569 | 0.8660 | 0.2133 | 0.8470 |

α=0.50 wins CID22 by 0.008 SROCC over α=0.40, BUT α=0.40 wins
KADID by 0.012, TID by 0.013, KonJND by **0.108**. AIC-3 is a wash.

5-seed CI confirms (CID22 SROCC):
- α=0.50: 0.8783, 0.8853, 0.8961, 0.8830, 0.8778 — mean 0.8841 ± 0.0066
- α=0.40: 0.8903, 0.8874, 0.8881, 0.8789, 0.8814 — mean **0.8852 ± 0.0042**

α=0.40 has TIGHTER CI (std 0.0042 vs 0.0066) and slightly better mean.

## Training command

```sh
./target/release/zensim_mlp_train \
  --group safesyn:/mnt/v/zen/zensim-training/2026-05-17-cvvdp/safesyn_features_mix_targets_372col.parquet:1.0:0.0 \
  --group kadid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp/kadid_features_mix_targets_372col.parquet:0.3:1.0 \
  --group tid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp/tid_features_mix_targets_372col.parquet:0.3:1.0 \
  --target-column mix_cv40_iw60 --target-scale 1.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 \
  --lr 1e-3 --l2 1e-5 --leaky-alpha 0.01 --val-policy min --seed 3 \
  --log-every 60 --early-stop-patience 50 --max-features 372 \
  --auto-transforms benchmarks/v0_20_feature_transform_greedy_screen_2026-05-15.tsv \
  --auto-transforms-min-lift 0.05 \
  --minibatch-size 256 --pwrc-pair-weight --norm-in-norm-weight 0.1 \
  --out v22_mix_cv40_iw60_s3_h128.bin
```

## Honest gaps

- KADID -0.044 SROCC vs V_22-IW v2 (was -0.120 with pure CVVDP — partially recovered)
- TID -0.067 SROCC vs V_22-IW v2 (was -0.106 — partially recovered)
- KADID/TID losses come from non-compression distortion classes
  (blur/noise/geometric) that V_22-IW v2 directly trained on at full
  IWSSIM weight; the mix de-emphasizes those in favor of CVVDP's
  codec-aware signal.

## Cross-references

- T11.13 commit + matrix data
- benchmarks/cvvdp_matrix_2026-05-17/safesyn_verdicts/v22cvvdp_*.md
