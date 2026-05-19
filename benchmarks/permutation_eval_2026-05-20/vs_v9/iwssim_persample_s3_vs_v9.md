# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/zen/zensim-eval/exp_iwssim_persample_2026-05-18/iwssim_persample_s3_h128.bin` (label: `iwssim_persample_s3_h128`)
- **B**: `/home/lilith/work/zen/zensim--v9-ship/zensim/weights/v_tuner_v9_2026-05-20.bin` (label: `v_tuner_v9_2026-05-20`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `500`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.2347 | 0.5274 | 0.962 | 0.847 | 0.2968 | 0.6150 | -16.001 | -6.643 | -0.000 | B>>A |
| KonJND-1k (full) | 1008 | 0.0097 | 0.0574 | 0.997 | 0.996 | 0.0780 | 0.0601 | -1.186 | -0.039 | +0.000 | tied |
| TID2013 | 3000 | 0.0257 | 0.3430 | 0.889 | 0.951 | 0.0089 | 0.3902 | -34.266 | 7.223 | +11.422 | promising |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 0 cells
- **BDecisivelyBeatsA**: 1 cells
- **PromisingNotDecisive**: 4 cells
- **Tied**: 12 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (0 A wins vs 1 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 4292 | 0.2347 | 0.2744 | 0.1590 | 0.0508 | 0.2968 | 0.962 |
| B: v_tuner_v9_2026-05-20 | 4292 | 0.5274 | 0.5319 | 0.3713 | 0.0485 | 0.6150 | 0.847 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | -0.1745 | -16.001 | 0.0000 | -6.643 | 0.0000 | -0.3182 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.2897 | 0.2701 | 0.1867 | 0.0526 | 0.2721 | 0.963 |
| B4 | [0.40, 0.50) | 266 | 0.0595 | 0.1318 | 0.0403 | 0.0489 | 0.0626 | 0.991 |
| B5 | [0.50, 0.60) | 615 | 0.0549 | 0.1225 | 0.0364 | 0.0488 | 0.0774 | 0.992 |
| B6 | [0.60, 0.70) | 836 | 0.0685 | 0.0659 | 0.0463 | 0.0431 | 0.1036 | 0.998 |
| B7 | [0.70, 0.80) | 1092 | 0.0116 | 0.0779 | 0.0067 | 0.0421 | 0.0172 | 0.997 |
| B8 | [0.80, 0.90) | 1382 | 0.0810 | 0.1303 | 0.0539 | 0.0398 | 0.1042 | 0.991 |
| B9 | [0.90, 1.00] | 43 | 0.1395 | 0.2660 | 0.0986 | 0.0233 | 0.1613 | 0.964 |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.2084 | 0.2195 | 0.1479 | 0.0526 | 0.1469 | 0.976 |
| B4 | [0.40, 0.50) | 266 | 0.0646 | 0.1316 | 0.0415 | 0.0414 | 0.0936 | 0.991 |
| B5 | [0.50, 0.60) | 615 | 0.1488 | 0.1579 | 0.1004 | 0.0439 | 0.1859 | 0.987 |
| B6 | [0.60, 0.70) | 836 | 0.1693 | 0.1736 | 0.1119 | 0.0419 | 0.2030 | 0.985 |
| B7 | [0.70, 0.80) | 1092 | 0.1703 | 0.1696 | 0.1148 | 0.0357 | 0.2070 | 0.986 |
| B8 | [0.80, 0.90) | 1382 | 0.2224 | 0.2152 | 0.1492 | 0.0369 | 0.2497 | 0.977 |
| B9 | [0.90, 1.00] | 43 | 0.3953 | 0.4582 | 0.2580 | 0.0698 | 0.5123 | 0.889 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B2 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B3 | 57 | 0.730 | 0.4655 | 0.118 | 0.9063 | +0.1252 | 0 | 0 | +0.000 | tied |
| B4 | 266 | -0.057 | 0.9546 | 0.000 | 0.9997 | -0.0310 | 0 | 0 | +0.000 | tied |
| B5 | 615 | -1.694 | 0.0903 | -0.091 | 0.9274 | -0.1085 | 0 | 0 | -0.000 | tied |
| B6 | 836 | -2.097 | 0.0360 | -0.272 | 0.7857 | -0.0994 | 0 | 0 | -0.000 | tied |
| B7 | 1092 | -8.270 | 0.0000 | -0.601 | 0.5478 | -0.1898 | 0 | 5 | -0.000 | promising |
| B8 | 1382 | -3.859 | 0.0001 | -0.412 | 0.6803 | -0.1455 | 0 | 5 | -0.000 | promising |
| B9 | 43 | -1.656 | 0.0978 | -0.511 | 0.6097 | -0.3510 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 1008 | 0.0097 | 0.0748 | 0.0081 | 0.0347 | 0.0780 | 0.997 |
| B: v_tuner_v9_2026-05-20 | 1008 | 0.0574 | 0.0934 | 0.0393 | 0.0010 | 0.0601 | 0.996 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | -0.2728 | -1.186 | 0.2356 | -0.039 | 0.9688 | +0.0180 | 0 | 0 | +0.000 | tied |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 3000 | 0.0257 | 0.4581 | 0.0269 | 0.0240 | 0.0089 | 0.889 |
| B: v_tuner_v9_2026-05-20 | 3000 | 0.3430 | 0.3078 | 0.2365 | 0.0277 | 0.3902 | 0.951 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.5022 | -34.266 | 0.0000 | 7.223 | 0.0000 | -0.3813 | 2 | 3 | +11.422 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0946 | 0.2563 | 0.0567 | 0.0690 | 0.1686 | 0.967 |
| B1 | [0.10, 0.20) | 34 | 0.3625 | 0.4929 | 0.2411 | 0.0294 | 0.3877 | 0.870 |
| B2 | [0.20, 0.30) | 185 | 0.3488 | 0.3605 | 0.2381 | 0.0541 | 0.3909 | 0.933 |
| B3 | [0.30, 0.40) | 493 | 0.0731 | 0.1624 | 0.0479 | 0.0325 | 0.0860 | 0.987 |
| B4 | [0.40, 0.50) | 677 | 0.0093 | 0.1283 | 0.0071 | 0.0251 | 0.0051 | 0.992 |
| B5 | [0.50, 0.60) | 705 | 0.0172 | 0.1255 | 0.0151 | 0.0284 | 0.0187 | 0.992 |
| B6 | [0.60, 0.70) | 809 | 0.1002 | 0.1548 | 0.0689 | 0.0247 | 0.1059 | 0.988 |
| B7 | [0.70, 0.80) | 67 | 0.4891 | 0.5071 | 0.3389 | 0.0448 | 0.5542 | 0.862 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0239 | 0.2398 | 0.0419 | 0.0345 | 0.0244 | 0.971 |
| B1 | [0.10, 0.20) | 34 | 0.4554 | 0.5633 | 0.3125 | 0.0882 | 0.5062 | 0.826 |
| B2 | [0.20, 0.30) | 185 | 0.1558 | 0.1731 | 0.1031 | 0.0216 | 0.2067 | 0.985 |
| B3 | [0.30, 0.40) | 493 | 0.0467 | 0.1604 | 0.0297 | 0.0284 | 0.0717 | 0.987 |
| B4 | [0.40, 0.50) | 677 | 0.1149 | 0.1569 | 0.0771 | 0.0162 | 0.1305 | 0.988 |
| B5 | [0.50, 0.60) | 705 | 0.0879 | 0.0571 | 0.0598 | 0.0199 | 0.1042 | 0.998 |
| B6 | [0.60, 0.70) | 809 | 0.0999 | 0.0999 | 0.0682 | 0.0124 | 0.1226 | 0.995 |
| B7 | [0.70, 0.80) | 67 | 0.2940 | 0.3602 | 0.1992 | 0.0448 | 0.3701 | 0.933 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 ⚠ | 29 | 0.413 | 0.6799 | 0.025 | 0.9798 | +0.1442 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | -1.389 | 0.1648 | -0.754 | 0.4508 | -0.1184 | 0 | 0 | -0.000 | tied |
| B2 | 185 | 4.181 | 0.0000 | 1.154 | 0.2483 | +0.1842 | 3 | 0 | +2.091 | promising |
| B3 | 493 | 1.298 | 0.1943 | 0.016 | 0.9871 | +0.0143 | 0 | 0 | +0.000 | tied |
| B4 | 677 | -7.214 | 0.0000 | -0.284 | 0.7764 | -0.1255 | 0 | 0 | -0.000 | tied |
| B5 | 705 | -5.365 | 0.0000 | 0.478 | 0.6330 | -0.0855 | 0 | 0 | +0.000 | tied |
| B6 | 809 | 0.017 | 0.9867 | 0.390 | 0.6965 | -0.0166 | 0 | 0 | -0.000 | tied |
| B7 | 67 | 3.290 | 0.0010 | 1.292 | 0.1964 | +0.1841 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

---
Wall time: 41.14s (8300 pair rows scored × 2 bakes across 3 corpora; 500 bootstrap resamples × bands).
