# Multi-codec ssim2 profile probe — 2026-06-30

Can the `canonical-picker-2026-06-27` dataset (5.74M rows, 7 codec×modes,
372 zensim feat + score_ssim2, clean origin split) train a better Profile::A?
**Yes.** A bare probe already beats the shipped v47 on every codec/HF human-MOS
corpus and cracks the G5 KonJND wall.

## Probe setup
- Train: 1.2M rows stratified (300k each from zenjpeg/zenavif/zenjxl/zenwebp
  lossy `train.parquet`, origins {0,2,4,6,8}), ssim2-decile-stratified for low-q.
- Target: `score_ssim2` (--target-scale 1.0). Held-out val = origins {1,3,5}.
- `zensim_mlp_train --max-features 372 --hidden 128 --per-sample-alpha-head
  --mse-weight 0.6 --ranknet-weight 0.6 --epochs 60 --seed 17 --lr 1e-3
  --minibatch-size 256`. NO monotone-cbc, NO QAT (probe only).
- Held-out-origin ssim2-tracking SROCC = 0.9647.

## A/B vs shipped v47 (bake_verdict, identical harness + 372-col canonical val)

| Corpus | probe | A (v47) | Δ | kind |
|---|--:|--:|--:|---|
| CID22 | 0.8827 | 0.8657 | +0.017 | codec (gold) |
| AIC-3 | 0.7948 | 0.7680 | +0.027 | codec JND |
| AIC-4 | 0.8921 | 0.8854 | +0.007 | neural codec |
| KonJND | 0.5619 | 0.4185 | +0.143 | HF/near-lossless (G5 wall) |
| KADID | 0.6397 | 0.7933 | −0.154 | NON-codec analytic |
| TID | 0.7248 | 0.7927 | −0.068 | NON-codec analytic |

CID22 0.8827 within 0.007 of fast-ssim2 (0.8895). Probe wins all codec+HF
corpora; loses only the non-compression analytic guards (never trained on
blur/noise/geometric).

## Caveats
- Dial broken (G3 inversions 0.28) — no monotone-cbc. Not ship-ready as-is.
- ssim2-only target; 1.2M of 5.7M; 60 epochs. Full-scale + balanced should beat this.

## Next: balanced ship-candidate
multi-codec 5.7M + kadid/tid/konjnd train subsets (recover analytic guards) +
--monotone-cbc + QAT f16 (A's dial correctness) + 372. Scale on Hetzner.
Optional: score 5.7M variants with cvvdp+iwssim for A's proven mix target.

Bakes: /mnt/v/output/zensim-multicodec-probe/probe_372.bin (+verdict.md);
A baseline: A_baseline.verdict.md. Trainer ban-228 guard landed same day.
