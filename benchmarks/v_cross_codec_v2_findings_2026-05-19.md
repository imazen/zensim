# EXP-CROSS-CODEC-V2 — findings (in progress)

**Date:** 2026-05-19
**Status:** Trainers in flight; eval pending.
**Prior ship:** PreviewV0_5CrossCodec = cc4_s1_w1.0 (W=1.0 seed=1 on
original 57,972-pair equivalence parquet). T=63 6-img butter 4.82,
20-img 5.52. Cross-corpus: CID22 0.880 / KADID 0.800 / TID 0.822 /
KonJND 0.327 / AIC-3 0.806.

## Substrate changes vs v1 (METRIC ship)

1. **Tighter equivalence pool**: max butter gap 0.3 (was 0.5).
2. **Avif-pair oversample**: 2× row weight for any pair containing
   zenavif (compensates for ~50% smaller pool).
3. **30 butter levels** (was 20) from 0.3 to 12.0 — finer granularity
   at lower butter (near-lossless region).
4. **Pool size**: 68,788 pairs (vs 57,972) — about +18%.
5. **Pair distribution after avif-2x weighting**:
   ```
   zenavif ↔ zenjpeg     8,803 × 2.0 weight
   zenavif ↔ zenjxl      9,309 × 2.0 weight
   zenavif ↔ zenwebp     7,667 × 2.0 weight
   zenjpeg ↔ zenjxl     12,327 × 1.0 weight
   zenjpeg ↔ zenwebp    15,323 × 1.0 weight
   zenjxl  ↔ zenwebp    15,359 × 1.0 weight
   ```
6. **Mean gap**: 0.114 (vs 0.173 in v1).

## Mid-training observation — W=2.0 collapse (epoch ~70)

The W=2.0 trainer at seed=1 ENTERED RANK-DEGENERATE STATE at
epoch ~30 and never recovered. Symptom: val SROCC dropped from 0.95
at epoch 30 to 0.05 at epoch 70 — the metric outputs degenerated
into a near-constant flatline (rank-collapse mode). α(x) collapsed
from per-sample-variable to μ=0.017, reducer_w near zero. This
matches the seed-2 W=1.0 collapse documented in v1 ship findings.

**Conclusion: W=2.0 with the current architecture is unsafe.**
The tighter pool and avif-rebalance did NOT prevent W=2.0 collapse.
Adding the documented rank-preservation regularizer (deferred to V3)
is the principled fix; for V2 we focus on W ∈ {1.0, 1.5} where the
training stayed stable.

## Pending evaluation

Trainers in flight (will resolve ~14:22 local):

| Bake | Seed | W | Epoch (last seen) | val SROCC | α(x) status |
|---|--:|--:|---|---:|---|
| cc4v2_s1_w1_0 | 1 | 1.0 | 70/300 | 0.89 (best 0.96) | stable α=1.0 |
| cc4v2_s1_w1_5 | 1 | 1.5 | 70/300 | 0.87 (best 0.91) | stable α=1.0 |
| cc4v2_s1_w2_0 | 1 | 2.0 | 70/300 | **0.05 COLLAPSED** | α=0.017 (degenerate) |
| cc4v2_s2_w1_5 | 2 | 1.5 | 60/300 | 0.94 (best 0.96) | stable α=1.0 |
| cc4v2_s3_w1_5 | 3 | 1.5 | 60/300 | 0.84 (best 0.95) | per-sample α=0.97 |

(The val-min policy means best is the LOWEST validation epoch reached;
the bake retained is the best checkpoint, NOT the final epoch.)

## TBD — eval results

To be filled in after trainers complete:

- Mohammadi panel (bake_verdict) for each bake.
- T=63 cross-codec butter (n=20, jpeg/webp/avif binary-search).
- § A.9 bake_compare vs ship cc4_s1_w1.0.

## TBD — ship decision

Three possible outcomes:

1. **W=1.5 candidate hits relaxed gate (T=63 butter < 3.0) with
   secondary gates clean** → rotate `PreviewV0_5CrossCodec` to V2.
2. **W=1.5 candidate beats v1 ship but doesn't hit < 3.0** → keep
   v1 ship; document V2 as "promising, not decisive."
3. **No candidate decisively beats v1 ship** → V2 falsified;
   document and move on to V3 (rank-preservation regularizer).
