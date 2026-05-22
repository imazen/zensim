# Mode B at production scale — V_24 recipe FAILS

**Date**: 2026-05-22
**Tracking**: imazen/zensim#40 (Gate A acumen)
**Status**: NEGATIVE — Mode B-lite preprocessing does NOT improve over the
shipped V_24-per-sample-α at production scale.

## Setup

The user mandate was to test Mode B-lite at the FULL production
recipe to determine if the small-data Path B win (+0.05 CID22
SROCC) transfers. The production-recipe pipeline:

- **Train data**: safesyn 196k + KADID 10k + TID 3k (217k pairs)
- **Val data**: AIC-3 (600 pairs) + CID22 (4292 pairs, held-out)
- **Features**: 372 (Basic + Extended + IW-pool regime)
- **Mode B preprocessing**: σ=8, band_idx=3, clamp=[0.1, 4.0]
  (the small-data sweep winner from
  `acumen_mode_b_sweep_2026-05-21.md`)
- **Architecture**: 300 → 128 → 128(identity) + per-sample-α head
  (matches the V_24-per-sample-α s4 ship)
- **Target**: `mix_cv40_iw60` (0.4×cvvdp + 0.6×iwssim, the V_24
  ship's target). For val groups (CID22/AIC-3) where `mix_cv40_iw60`
  is unpopulated in canonical, substituted from `human_score`.
- **Hyperparams**: h=128, 300 epochs, pairs-per-epoch=60000,
  lr=0.001, seed=1, early-stop patience 50, val_policy=Min,
  per_sample_alpha_head=true

## Result

| Metric | This run (Mode B) | Shipped V_24-per-sample-α s4 | Δ |
|---|--:|--:|--:|
| Best val SROCC | **0.2337** | ~0.81 | **−0.58** |
| CID22 SROCC | **0.5745** | **0.8641** | **−0.2896** |
| AIC-3 SROCC | 0.2337 | 0.8183 | -0.5846 |
| safesyn SROCC (train) | 0.99 | 0.99 | tied |
| KADID SROCC (train) | 0.92 | 0.93 | tied |
| TID SROCC (train) | 0.37 | 0.89 | -0.52 |

Best CID22 = 0.5745 at epoch 100; trainer stopped at epoch 150
with patience=50 expired (no val improvement since e100). Network
fit safesyn perfectly (0.99) but generalized poorly to all val.

## Root cause: target/feature mismatch

The Path B small-data setup (CID22 0.7044 → 0.7543) used **human
MOS directly as the target** on KADID/TID/AIC-3 → eval on CID22
human MOS. That setup REWARDED Mode B's perceptual weighting
because the task was "predict human MOS from perceptually-weighted
features."

The V_24 production recipe trains on **ssim2-mix as target**
(`mix_cv40_iw60`). The mix targets were computed on UN-WEIGHTED
images (the canonical parquets are from the un-modified pipeline).
So the (features, target) pairs are MISMATCHED at production scale:

- Features: extracted from Mode-B-weighted RGB
- Target: ssim2-mix computed on un-weighted RGB

The network learned the safesyn surface perfectly (safesyn=0.99)
because that surface is consistent within safesyn (Mode B applied
the same way to every safesyn pair). But the learned mapping
doesn't transfer to CID22 human MOS because:
- ssim2-mix and CID22 human MOS already correlate (~0.86 SROCC
  via the V_24 ship's training)
- Mode B alters features → network learns DIFFERENT mapping that
  predicts ssim2-mix on weighted features
- That DIFFERENT mapping doesn't preserve the ssim2-mix ↔
  CID22-MOS correlation

## Why small-data Path B worked but production-scale doesn't

The Path B small-data win was honest in its setup: 13k train pairs
predicting human MOS, eval on held-out human MOS. Mode B helped
because the network had limited capacity to learn perceptual
weighting on its own; Mode B pre-pool weighting provided helpful
inductive bias.

At production scale, the network has 217k training pairs and is
trained against ssim2-mix. ssim2-mix is ITSELF a perceptual metric
that already encodes contrast sensitivity, masking, multi-scale
weighting, etc. Adding Mode B on top is REDUNDANT with what
ssim2-mix already does — and the redundancy actively HURTS because
it shifts the input distribution away from what ssim2-mix expects.

## Three paths forward

| Path | What | Cost | Expected outcome |
|---|---|--:|---|
| 1 | Recompute ssim2-mix on Mode-B-weighted images, then train V_24 on matched (features, target) pairs | ~20h compute | Could work, could regress. The (target_weighted, MOS_unweighted) correlation is uncertain. |
| 2 | Train V_24 recipe on HUMAN MOS DIRECTLY (no ssim2 mix) — match the small-data Path B setup. Drop safesyn (no human MOS) | ~5min retrain | Underpowered: ~14k human-MOS train pairs vs 217k ssim2-mix. Likely worse than V_24 ship. |
| 3 | Per-band Mode B kernel (in-pyramid CSF, not pre-pool RGB) | Multi-hour kernel work | Architecturally different problem; may not have the target-mismatch issue. |

## Honest conclusion

**Mode B-lite preprocessing does not help at the production V_24 recipe.**
The 0.05 CID22 lift from small-data Path B does not transfer to the
ssim2-mix-targeted training that V_24 ships use. The mechanism is
clear: Mode B's pre-pool perceptual weighting is REDUNDANT with
what ssim2-mix already encodes, and the (features, target)
mismatch shifts the input distribution away from where the target
was defined.

**Acumen Mode B-lite shipping**: NOT recommended at this time.

The architectural foundation (LUT, ViewingCondition, ModeBPreprocessor,
Cargo feature gating) remains preserved on `feat/acumen-foundation`
+ `feat/acumen-gpu` for future HDR / Mode B-per-band / parallel-head
explorations where the target-mismatch issue may not apply.

## Bake artifact (preserved for reproducibility)

- `/home/lilith/acumen-data/v24/v24_modeb_372.bin` (223 KB, F32, per-sample-α)
- Best epoch: 100, CID22=0.5745, KADID=0.9212, TID=0.3735
- Trained on Mode-B-weighted safesyn 196k + KADID + TID
- Target: mix_cv40_iw60, scaled ×100
- Stopped: epoch 150 (early-stop patience 50)

## Compute cost

- Safesyn 372-feature extraction (CPU, 32-thread): 60 min
- Val 372-feature extraction (CPU): 1 min
- V_24 training (CPU, single-thread): 10 min
- Total: ~75 min on local workstation, \$0
