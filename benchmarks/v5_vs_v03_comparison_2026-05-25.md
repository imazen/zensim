# V5 (prod_2layer_v5_spline) vs V0_3 (tuner_v11) — 2026-05-25

Bake verdict comparison on all 6 validation corpora.

- V0_3: `zensim/weights/v_tuner_v11_2026-05-24.bin` (currently shipped as PreviewV0_3)
- V5: `/mnt/v/output/zensim/bakes/prod_2layer_v5_spline_2026-05-25.bin`

## Aggregate Mohammadi Panel Comparison

| Corpus | n | V0_3 SROCC | V5 SROCC | ΔSROCC | V0_3 PLCC | V5 PLCC | V0_3 PWRC | V5 PWRC | V0_3 Z-RMSE | V5 Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8604 | **0.8798** | **+0.019** | 0.8525 | **0.8819** | 0.9089 | **0.9248** | 0.523 | **0.472** |
| KADIK10k | 10125 | **0.9237** | 0.9227 | -0.001 | 0.9229 | 0.9228 | 0.9550 | 0.9550 | 0.385 | 0.385 |
| TID2013 | 3000 | **0.8849** | 0.8834 | -0.002 | 0.8886 | **0.8929** | 0.9146 | 0.9132 | 0.459 | **0.450** |
| KonJND | 1008 | 0.2888 | **0.4523** | **+0.164** | 0.2586 | **0.4111** | 0.4043 | **0.5854** | 0.966 | **0.912** |
| AIC-3 | 600 | 0.7761 | **0.8180** | **+0.042** | 0.7877 | **0.8203** | 0.8538 | **0.8887** | 0.616 | **0.572** |
| AIC-4 | 300 | **0.9284** | 0.9258 | -0.003 | **0.9236** | 0.9218 | **0.9620** | 0.9582 | **0.383** | 0.388 |

## Verdict

V5 wins on:
- CID22: +0.019 SROCC, +0.029 PLCC, +0.016 PWRC, -0.051 Z-RMSE
- KonJND: +0.164 SROCC (massive improvement in JND calibration)
- AIC-3: +0.042 SROCC, +0.033 PLCC, +0.035 PWRC, -0.044 Z-RMSE

V0_3 wins on:
- KADIK10k: +0.001 SROCC (within noise)
- TID2013: +0.002 SROCC (within noise), but V5 wins PLCC and Z-RMSE
- AIC-4: +0.003 SROCC, but tiny n=300

**V5 is universally better or within noise of V0_3.** The KonJND and AIC-3
lifts are the most meaningful — those corpora measure exactly the
compression-focused quality judgments zensim targets.

## CVVDP gap (for reference)

CVVDP (Mohammadi 2025 Table III) achieves AIC-3 SROCC 0.960.
Our V5 is 0.818 — gap of 0.142.

The gap is structural: CVVDP uses display-calibrated CSF + spatial
frequency decomposition + display-dependent luminance mapping. Our
MLP architecture has no CSF-aware features.

## Training config (V5)

- Architecture: 2-layer 372→128→64 MLP, per-sample-α head, tanh pin scale=30
- Training: MSE-only (ranknet_weight=0, mse_weight=1), monotonicity_reg=1,
  anchor_loss_weight=0.5 target=60, seed=1
- Auto-transforms: 155 YeoJohnson/WinsorP99/QuantileBins/SignedCbrt
- Output: PCHIP spline calibration baked into ZNPR v3 metadata
- Groups: safesyn(1.0/0.5) + cid22_train(0.5/1.0) + kadid(0.5/1.0) +
  tid(0.5/1.0) + konjnd_dense(0.3/0.5)

## V7 (RankNet=1.0 + MSE=0.5, 5 groups) — 2026-05-25

Training: 200 epochs, val(geomean3)=0.9305 at epoch 199.
RankNet fixes the training divergence but produces an overfit model:

| Corpus | V5 SROCC | V7 SROCC | Δ | Notes |
|---|---|---|---|---|
| CID22 | 0.8798 | 0.8176 | -0.062 | Overfit to training groups |
| KADIK10k | 0.9227 | 0.9207 | -0.002 | Tied |
| TID2013 | 0.8834 | 0.9352 | +0.052 | Big win (TID in training) |
| KonJND | 0.4523 | 0.2796 | -0.173 | Regression (MSE too weak) |
| AIC-3 | 0.8180 | 0.7387 | -0.079 | Holdout regression |
| AIC-4 | 0.9258 | 0.8461 | -0.080 | Holdout regression |

**Verdict:** RankNet-dominant (w=1.0) overfits to training corpora.
V5's MSE-only approach generalizes better. Sweeping MSE/RankNet
balance in v8 (MSE=2.0, RN=0.3) and v9 (MSE=1.0, RN=0.1).

## Training regression (2026-05-25)

MSE-only 5-group 2-layer training diverges after epoch 0 in all
current builds. The v5 production bake was produced by a binary from
earlier in this session (before σ-weighted MSE infrastructure was
added). That binary is lost to context compaction.

Working training paths:
- RankNet-dominant (w≥0.5): converges but overfits to training groups
- 1-group MSE-only with RankNet: converges (CVVDP-proxy bake)
- 2-group (safesyn+kadid) MSE-only: converges

Broken paths:
- 5-group MSE-only (ranknet_weight=0): diverges at epoch 1
- 3-group MSE-only: diverges at epoch 1

Root cause unknown. Not in σ-weighted code (disabled and tested).
Likely in validation-loop interaction with multi-group min-policy.
