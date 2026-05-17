# V0_26 KonJND-aligned co-supervision — partial recovery, tradeoff finding

**Trained 2026-05-12 23:37Z. Eval 23:43Z.**
**Bake**: `/tmp/zensim_loop/bakes/v0_26_konjnd_dssim0_2026-05-13.bin`

## Recipe

```bash
train_v_next_mlp.py \
  --sweeps NONE \
  --human-csv "safesyn:.../safe_synth_clean_features_with_dssim_qc.csv:1.0:0.0" \
  --human-csv "konjnd:/tmp/zensim_loop/konjnd_aligned_features.csv:1.0:0.0" \
  --target ssim2 --loss mse_rank --hidden 128 --epochs 300 \
  --batch-size 16384 --lr 3e-3 --weight-decay 1e-5 --rank-weight 0.5 \
  --tv-weight 20 --dssim-weight 0.0 --seed 1
```

Two-source supervision: 144k safesyn + 76k KonJND human-rated.
**No dssim head** (control for the V0_24 dssim experiment).

## Cross-corpus result

| | V0_26 | V0_16 (ship) | V0_25 (no KonJND) | Δ V0_26-V0_16 | Δ V0_26-V0_25 |
|---|---:|---:|---:|---:|---:|
| **AIC-3** | **0.8027** | 0.7990 | (V0_25 not eval'd on AIC) | **+0.004 (V0_26 wins)** | — |
| AIC-4    | 0.9097 | 0.9175 | — | -0.008 | — |
| **CID22** | 0.8639 | **0.8919** | 0.8505 | -0.028 | **+0.013 (KonJND adds value)** |

V0_26 reproduces ~67% of V0_16's CID22 SROCC (0.8639 / 0.8919 — closes
the 0.041 V0_25 gap by 32%). The remaining 0.028 gap is **still
unexplained** — V0_16 has additional ingredients beyond TV + safesyn +
KonJND that we haven't reconstructed. Candidates:
- KADID + TID human-MOS mixed supervision (V0_4's recipe; V0_16 may
  have inherited it)
- Concordance filter (`--concordance-filter ssim2_butter`)
- Different features.bin generation (the safe_synth_clean_features.csv
  V0_16 trained on may not be the exact one we used)

## The interesting finding — JPEG-AI on AIC-4

V0_26 per-codec AIC-4 SROCC vs V0_16:

| Codec | V0_26 | V0_16 | Δ |
|---|---:|---:|---:|
| **JPEG-AI** | **0.8387** | 0.7951 | **+0.044** ⚡ |
| AVIF       | 0.9551 | 0.9598 | -0.005 |
| JPEG-1     | 0.9283 | 0.9541 | -0.026 |
| JPEG-2000  | 0.9224 | 0.9357 | -0.013 |
| JPEG-XL    | 0.9717 | 0.9705 | +0.001 |
| VVC        | 0.9225 | 0.9375 | -0.015 |

**V0_26 closes the JPEG-AI gap (cycle-7's primary motivation) by
+0.044 SROCC vs V0_16 without using dssim**. KonJND alignment alone
moves the V_X weights toward transformer-codec friendliness.

Per the tick-477 AIC-4 table:
- V0_16 JPEG-AI 0.7951 was the biggest per-codec deficit on AIC-4.
- V0_26 brings JPEG-AI to 0.8387 — comparable to V0_16's per-codec
  AIC-4 average; still ~0.008 below fast-ssim2 (0.8459) but only ~0.08
  below dssim (0.9147).

## The tradeoff

V0_26 doesn't dominate V0_16 — it shifts the per-corpus emphasis:

- **CID22** (-0.028): hurts the canonical gold-standard the most.
- **AIC-4 aggregate** (-0.008): tiny.
- **AIC-3** (+0.004): slight gain.
- **AIC-4 JPEG-AI** (+0.044): substantial gain.
- AIC-4 non-JPEG-AI codecs (-0.01 to -0.03 each): some regression.

Net: V0_26 is more uniform across codecs but loses the V0_16
specific tuning that pushed CID22 to 0.8919.

## Decision

**V0_16 stays ship.** CID22 is the canonical gate per CLAUDE.md
goal #1; -0.028 there is a real regression.

V0_26 is preserved as a useful experiment showing:
1. The V0_16 0.041 gap toward V0_25 was ~32% from KonJND alignment.
2. KonJND alignment indirectly improves JPEG-AI tracking.
3. Future cycle-8 work could explore (a) finding the remaining 0.028
   CID22 ingredient, (b) ensemble of V0_16 + V0_26 to balance both,
   or (c) architectural extensions for JPEG-AI without CID22 cost.

## Cycle-7 cumulative findings

| Variant | CID22 | AIC-3 | AIC-4 agg | AIC-4 JPEG-AI |
|---|---:|---:|---:|---:|
| V0_16 (ship) | 0.8919 | 0.7990 | 0.9175 | 0.7951 |
| V0_24 v1 (no TV, dssim=0.3) | 0.8315 | — | — | — |
| V0_24 v2 (TV, dssim=0.3)    | 0.8254 | — | — | — |
| V0_25 (TV, dssim=0)         | 0.8505 | — | — | — |
| **V0_26 (TV+KonJND, dssim=0)** | **0.8639** | **0.8027** | **0.9097** | **0.8387** |

V0_26 is the new cycle-7 reference recipe; future experiments build
from it. V0_16 stays ship.
