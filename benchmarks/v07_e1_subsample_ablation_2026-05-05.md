# V0_7 e1-fill subsampling ablation (2026-05-05)
Trained 6 V0_6-architecture MLP variants (dct_hf zenanalyze features, no sampler bias) on increasing fractions of zenjpeg-420-e1 fill rows on top of the 218k base. Each variant evaluated on KADID/TID/CID22/KonJND human-MOS holdouts (1500 random pairs each).

Goal: find a sweet-spot e1 fraction (if any) that improves human-MOS generalization vs V0_6 baseline (0pct).

## |SROCC| vs human-MOS, full holdout

| variant | KADID | TID | CID22 | KonJND |
|---------|------:|----:|------:|-------:|
| V0_6 published baseline | 0.8496 | 0.8416 | 0.8935 | — |
| ablation_0pct (V0_6 baseline reproduction) | 0.8496 | 0.8416 | 0.8935 | — |
| ablation_5pct | 0.8363 | 0.8229 | 0.8914 | — |
| ablation_10pct | 0.8405 | 0.8395 | 0.8985 | — |
| ablation_20pct | 0.8491 | 0.8267 | 0.8920 | — |
| ablation_50pct | 0.8353 | 0.8326 | 0.8878 | — |
| ablation_100pct (V0_7 reproduction) | 0.8486 | 0.8264 | 0.8911 | — |

## Per-band |SROCC| vs human-MOS (KADID + TID + CID22 pooled, banded by fast_ssim2_score)

| variant | 25-40 | 40-60 | 60-75 | 75-90 |
|---------|------:|------:|------:|------:|
| ablation_0pct (V0_6 baseline reproduction) | 0.2070 (335) | 0.0315 (684) | 0.5050 (969) | 0.5927 (843) |
| ablation_5pct | 0.3119 (335) | 0.0989 (684) | 0.5284 (969) | 0.5366 (843) |
| ablation_10pct | 0.1509 (335) | 0.0385 (684) | 0.5526 (969) | 0.5766 (843) |
| ablation_20pct | 0.2806 (335) | 0.0419 (684) | 0.5173 (969) | 0.5620 (843) |
| ablation_50pct | 0.2258 (335) | 0.1315 (684) | 0.4926 (969) | 0.5764 (843) |
| ablation_100pct (V0_7 reproduction) | 0.3037 (335) | 0.0101 (684) | 0.4528 (969) | 0.5748 (843) |

## Δ vs V0_6 baseline (0pct = reproduction)

| variant | KADID Δ | TID Δ | CID22 Δ | KonJND Δ | wins |
|---------|--------:|------:|--------:|---------:|-----:|
| 5pct | -0.0133 | -0.0187 | -0.0022 | — | 0/4 |
| 10pct | -0.0091 | -0.0022 | +0.0050 | — | 1/4 |
| 20pct | -0.0005 | -0.0149 | -0.0015 | — | 0/4 |
| 50pct | -0.0143 | -0.0091 | -0.0058 | — | 0/4 |
| 100pct | -0.0010 | -0.0152 | -0.0024 | — | 0/4 |

## Verdict

**No e1-fill fraction improves on V0_6 baseline across the human-MOS axes (KADID + TID + CID22 summed).** Every fraction tested is a regression. The least-bad variant is ablation_10pct (summed Δ = -0.0063, wins 1/3 datasets), but even it loses on TID and KADID.

**Recommendation: skip the e1 fill entirely.** Keep V0_6 (218k base) as the V0_7 candidate. The original V0_7 plan (100% e1 fill + sampler bias) was worse on every holdout; subsampling at 5/10/20/50% does not recover. The e1 fill content is fundamentally unhelpful for human-MOS generalization in this configuration. Consider:

- A different intervention axis (e.g., new content classes, different   zenanalyze features, codec-class sampling weights)
- Investigating WHY e1 hurts: hypothesis is that JPEG-family bias goes   from 56% (base) to 63% at 100% (per zenjpeg_e1_fill_plan_2026-05-01.md),   which over-fits the MLP to JPEG artifact statistics at the expense of   AVIF/JXL/WebP/general-distortion sensitivity
- Trying e1 at quality grids that hit the 60-75 SSIM2 band (where most   human-MOS pairs live) instead of the wide 0-90 spread the fill targeted
