# P12 residual-boost — 2026-07-27

Stage 1: BVLS ≥0 on the 504 config, ext_safesyn_full train (active 23/504, train R² 0.9450).
Stage 2: per-family LINEAR on residuals (E-class excluded: GRAD_SRC_MEAN, LUM_MID_ERR, ART_DEV2, DET_DEV2).
Holdouts are NEVER fit (CID22 MOS ban absolute): eval-only column = SROCC(holdout residual, safesyn-fitted family prediction).

## Ranked marginal value (mean CV-R² across the 4 train-legal corpora)

| rank | append family | mean CV-R² |
|---|---|---|
| 1 | MSCN_DIFF | 0.51523 |
| 2 | CONTRAST_GAIN/LOSS | 0.48665 |
| 3 | GMS_DEV2 | 0.42069 |
| 4 | LUM_BINS(dark+bright) | 0.37690 |
| 5 | LUM_TRANSDUCER | 0.24390 |
| 6 | GLOBAL(dmean+cgain+closs) | 0.24355 |
| 7 | XMASK_TRANSDUCER | 0.22667 |
| 8 | TEXTURE_DISSIM | 0.11375 |

Full per-corpus table: `p12_residual_boost_2026-07-27.csv` (holdout rows are eval-only).
