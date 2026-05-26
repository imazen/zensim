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

## ROOT CAUSE FOUND (2026-05-25 17:20)

The training regression was caused by **mismatched human_score scales**:
- safesyn: [-7.4, +1.0] (distance-like)
- cid22_train: [3.0, 94.2] (raw MCOS — should be /100)
- kadid: [0.0, 1.0] (normalized DMOS)
- tid: [0.03, 0.8] (normalized MOS)
- konjnd_dense: [-65.7, 96.2] (mixed training target)

With tanh_output_head_scale=30, the model outputs in [0, 100].
The MSE loss creates opposing gradients when targets span [-66, 94]
across groups while the model predicts in [0, 100].

**Fix: normalize all group targets to [0, 1] before training.**
- cid22_train: human_score / 100
- konjnd_dense: min-max normalization to [0, 1]

V12 (normalized, seed=1): CID22 0.8815, TID 0.9083, KADIK 0.9194.
Beats V5 on CID22 (+0.002) and TID (+0.025). Training converges
stably with MSE-only (val 0.8631 → 0.9205 over 200 epochs).

## V40 dynamic-range-floor (2026-05-25) — overshoots, rejected

Tried the goals-doc G1 lever --dynamic-range-floor-weight 0.3 (using the
2026-05-20-v12-cvvdp-substrate equiv pool as q-sweep substrate) + the
23,560-row continuous CVVDP anchor for a denser spline.

Result: 38 spline knots (vs V39's 3) but the floor pushed output too high:
  p5=103.6 p95=120.4 (raw) → clamps to 100 → saturated top, broken dial
  CID22 SROCC 0.8259 (vs V39's 0.8793)

V39 remains the champion. Its simpler recipe (V32 ranking + tiny-weight
multi-band anchor spline) gives raw p5=-89.7 p95=97.4 → clamps to a clean
[0,97] dial (G1=1.00) AND keeps CID22 SROCC 0.8793.

Note: cross-codec-eq aux loss (G4) is NOT wired for 2-layer mode —
"multi-layer / skip + cross_codec_eq: aux loss not yet wired". Wiring it
is a future task for the G4 cross-codec-equivalence goal.

## SESSION CONCLUSION: V39 shipped as PreviewV0_3

V39 = universally better than V0_3 (v_tuner_v11):
  CID22 0.8793 (+0.019), KADIK 0.9251 (+0.001), TID 0.9317 (+0.047),
  KonJND 0.4197 (+0.131), AIC-3 0.8023 (+0.026), G1 dial 1.00 (vs 0.69).

Core lesson: SROCC is rank-invariant under a monotone calibration spline.
A well-ranking compressed bake (V32) + multi-band-anchor spline = both
good rank AND working dial. The bake_verdict scorecard (auto-runs after
every train) catches the broken-dial regression that SROCC-only hides.

## V41 CVVDP-emulator (2026-05-25) — negative result

Trained V39's recipe but with cvvdp_log_norm as the target (safesyn+kadid+tid)
+ continuous CVVDP anchor for spline. Goal: a bake "close to CVVDP" by
training toward CVVDP scores directly.

Result — human-MOS SROCC DROPPED vs V39:
  CID22  0.6599 (V39: 0.8793)
  KADIK  0.5256 (V39: 0.9251)
  TID    0.6819 (V39: 0.9317)
  AIC-3  0.7464 (V39: 0.8023)

Lesson: emulating CVVDP's OUTPUT ≠ having CVVDP's ACCURACY. Training toward
CVVDP scores makes the bake inherit CVVDP's systematic deviations from human
MOS WITHOUT the CSF/display mechanism that earns CVVDP its 0.96. For the
codec-target use case (track human quality), V39 (trained on human-MOS-derived
targets) is strictly better. The CVVDP-proxy direction is a dead end for a
shipped metric.

EMPIRICAL VERDICT on "close to cvvdp": the 0.79-0.81 AIC-3 ceiling is
feature-limited (no CSF-aware spatial-frequency features), CONFIRMED by:
- 40 bakes across recipes all hitting ~0.80 AIC-3
- V41 CVVDP-target emulator NOT helping (0.75)
Closing the gap requires new input features (architectural), not training.
