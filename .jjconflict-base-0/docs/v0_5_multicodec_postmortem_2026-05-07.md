# V0_5 multi-codec retrain — postmortem (2026-05-07)

## Result: not shipping

Multi-codec V0_5 (228 → 64 LeakyReLU → 1, trained on
v15r+v15rc+v14+v12 unified parquets, 2.34M rows including 39k
non-jpeg) achieved val_srocc=0.9653, test_srocc=0.9662 on the
synthetic ssim2 holdout — superficially beats V0_4's 0.9547/0.9814.

But on the human-rated holdouts it tanks:

| Dataset | V0_2 | **V0_4 (2026-04-30, currently shipping)** | V0_5 (multi-codec) |
|---|--:|--:|--:|
| KADIK10k | 0.8192 | **0.8432** | 0.3697 |
| TID2013 | 0.8427 | **0.8401** | 0.6298 |
| CID22 | 0.8676 | **0.8893** | 0.8609 |

V0_5 is *worse* than V0_4 by -0.47 SROCC on KADIK10k.

## Diagnosis

The 2026-04-30 V0_4 bake's win on human datasets came from explicit
mixed-source training (per training-log, recovered today):

```
train group 0: 'Synthetic' n=218089 train_w=1.000 val_w=0.000
train group 1: 'Kadid10k_train' n=7125 train_w=0.300 val_w=0.000
train group 3: 'Tid2013_train' n=2160 train_w=0.300 val_w=0.000
val-only group 2: 'Kadid10k_val' n=3000 train_w=0.000 val_w=1.000
val-only group 4: 'Tid2013_val' n=840 train_w=0.000 val_w=1.000
val-only group 5: 'cid22' n=4292 train_w=0.000 val_w=1.000
```

V0_5 used only synthetic ssim2-targets across more codecs but no
human-MOS supervision. Result: it learns to predict ssim2 well across
codec types, but ssim2 itself doesn't track KADID's noise/blur/color
distortions, so the model has no signal for those.

**Cross-codec synthetic ≠ cross-distortion-type. Adding webp/avif/jxl
samples doesn't help with KADID's analytic distortions.**

## Decision

- V0_5 multi-codec bake archived to R2
  (`s3://zentrain/v_next-training/2026-05-07/bakes/v0_5_2026-05-07_multicodec.bin`)
  for reproducibility but NOT shipping into `zensim/weights/`.
- V0_4 (2026-04-30) stays as the experimental-feature default.
- V0_6 plan: extend `scripts/v_next/train_v_next_mlp.py` to load raw
  KADID10k + TID2013 image pairs, extract 228-dim features per pair,
  and mix them as additional training rows with `train_w=0.3` and
  per-dataset val splits. Non-trivial (~2 days), not started.

## What would be useful in V0_6

1. **Data**: KADID10k_train (~7k pairs) + TID2013_train (~2k pairs)
   features extracted via zensim's existing feature pipeline (228
   features), plus the existing 2.34M synthetic rows.
2. **Loss**: per-group weighting like the legacy in-tree trainer used
   (synthetic weight 1.0, human-MOS weight 0.3).
3. **Validation**: held-out KADID_val + TID_val + full CID22 as
   independent val-only splits.
4. **Bumpiness**: TV regularizer (already implemented as
   `--tv-weight` in the trainer) to address the V0_4 bumpiness audit
   (V0_4 had 8.26% non-monotone q-steps vs ssim2's 5.08%).
5. **Cross-codec**: keep the v15r+v15rc+v14+v12 unified parquets in
   training so the model sees zenwebp/avif/jxl/png distortion shapes
   too.
