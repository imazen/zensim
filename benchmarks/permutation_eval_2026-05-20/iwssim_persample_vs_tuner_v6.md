# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/zen/zensim-eval/exp_iwssim_persample_2026-05-18/iwssim_persample_s3_h128.bin` (label: `iwssim_persample_s3_h128`)
- **B**: `/home/lilith/work/zen/zensim--v6-reship/zensim/weights/v_tuner_v6_2026-05-19.bin` (label: `v_tuner_v6_2026-05-19`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `500`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.2347 | 0.8216 | 0.962 | 0.680 | 0.2968 | 0.8821 | -43.337 | -24.446 | -0.000 | B>>A |
| KonJND-1k (full) | 1008 | 0.0097 | 0.3734 | 0.997 | 0.960 | 0.0780 | 0.5297 | -13.019 | -1.381 | -0.000 | promising |
| TID2013 | 3000 | 0.0257 | 0.4447 | 0.889 | 0.809 | 0.0089 | 0.5451 | -16.809 | -3.726 | -2.801 | B>>A |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 0 cells
- **BDecisivelyBeatsA**: 3 cells
- **PromisingNotDecisive**: 7 cells
- **Tied**: 7 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (0 A wins vs 3 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 4292 | 0.2347 | 0.2744 | 0.1590 | 0.0508 | 0.2968 | 0.962 |
| B: v_tuner_v6_2026-05-19 | 4292 | 0.8216 | 0.7332 | 0.6249 | 0.0531 | 0.8821 | 0.680 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.2274 | -43.337 | 0.0000 | -24.446 | 0.0000 | -0.5853 | 0 | 5 | -0.000 | B>>A |

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
| B3 | [0.30, 0.40) | 57 | 0.0442 | 0.1398 | 0.0251 | 0.0351 | 0.1018 | 0.990 |
| B4 | [0.40, 0.50) | 266 | 0.2052 | 0.2253 | 0.1387 | 0.0414 | 0.2696 | 0.974 |
| B5 | [0.50, 0.60) | 615 | 0.2724 | 0.2644 | 0.1842 | 0.0504 | 0.3347 | 0.964 |
| B6 | [0.60, 0.70) | 836 | 0.3013 | 0.3177 | 0.2016 | 0.0395 | 0.3590 | 0.948 |
| B7 | [0.70, 0.80) | 1092 | 0.3289 | 0.3257 | 0.2227 | 0.0394 | 0.4053 | 0.945 |
| B8 | [0.80, 0.90) | 1382 | 0.4089 | 0.4129 | 0.2732 | 0.0246 | 0.4957 | 0.911 |
| B9 | [0.90, 1.00] | 43 | 0.0193 | 0.2439 | 0.0122 | 0.0465 | 0.0841 | 0.970 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B2 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B3 | 57 | 1.546 | 0.1222 | 0.176 | 0.8605 | +0.1703 | 0 | 0 | +0.000 | tied |
| B4 | 266 | -1.920 | 0.0549 | -0.227 | 0.8207 | -0.2070 | 0 | 0 | -0.000 | tied |
| B5 | 615 | -5.406 | 0.0000 | -0.708 | 0.4788 | -0.2572 | 0 | 5 | -0.000 | promising |
| B6 | 836 | -7.407 | 0.0000 | -1.598 | 0.1101 | -0.2553 | 0 | 5 | -0.000 | promising |
| B7 | 1092 | -9.050 | 0.0000 | -1.499 | 0.1339 | -0.3880 | 0 | 5 | -0.000 | promising |
| B8 | 1382 | -13.669 | 0.0000 | -3.460 | 0.0005 | -0.3915 | 0 | 6 | -0.000 | B>>A |
| B9 | 43 | 0.729 | 0.4660 | 0.037 | 0.9708 | +0.0773 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 1008 | 0.0097 | 0.0748 | 0.0081 | 0.0347 | 0.0780 | 0.997 |
| B: v_tuner_v6_2026-05-19 | 1008 | 0.3734 | 0.2813 | 0.2584 | 0.0347 | 0.5297 | 0.960 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.1357 | -13.019 | 0.0000 | -1.381 | 0.1671 | -0.4516 | 0 | 5 | -0.000 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 3000 | 0.0257 | 0.4581 | 0.0269 | 0.0240 | 0.0089 | 0.889 |
| B: v_tuner_v6_2026-05-19 | 3000 | 0.4447 | 0.5878 | 0.3055 | 0.0430 | 0.5451 | 0.809 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | -0.3191 | -16.809 | 0.0000 | -3.726 | 0.0002 | -0.5361 | 1 | 5 | -2.801 | B>>A |

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
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0825 | 0.4124 | 0.0321 | 0.0000 | 0.0071 | 0.911 |
| B1 | [0.10, 0.20) | 34 | 0.6252 | 0.7559 | 0.4125 | 0.0000 | 0.7246 | 0.655 |
| B2 | [0.20, 0.30) | 185 | 0.2158 | 0.2735 | 0.1467 | 0.0324 | 0.3043 | 0.962 |
| B3 | [0.30, 0.40) | 493 | 0.2468 | 0.2716 | 0.1696 | 0.0446 | 0.3020 | 0.962 |
| B4 | [0.40, 0.50) | 677 | 0.2092 | 0.2786 | 0.1389 | 0.0473 | 0.2635 | 0.960 |
| B5 | [0.50, 0.60) | 705 | 0.1067 | 0.1662 | 0.0705 | 0.0383 | 0.1421 | 0.986 |
| B6 | [0.60, 0.70) | 809 | 0.0190 | 0.0851 | 0.0124 | 0.0334 | 0.0263 | 0.996 |
| B7 | [0.70, 0.80) | 67 | 0.4070 | 0.4746 | 0.2745 | 0.0448 | 0.4777 | 0.880 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 ⚠ | 29 | 0.078 | 0.9380 | -0.371 | 0.7109 | +0.1615 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -4.674 | 0.0000 | -4.449 | 0.0000 | -0.3368 | 0 | 2 | -0.000 | promising |
| B2 | 185 | 2.380 | 0.0173 | 0.543 | 0.5868 | +0.0866 | 0 | 0 | +0.000 | tied |
| B3 | 493 | -6.057 | 0.0000 | -0.864 | 0.3876 | -0.2159 | 0 | 5 | -0.000 | promising |
| B4 | 677 | -3.777 | 0.0002 | -0.601 | 0.5479 | -0.2584 | 1 | 5 | -0.630 | promising |
| B5 | 705 | -1.557 | 0.1194 | -0.105 | 0.9162 | -0.1234 | 0 | 0 | -0.000 | tied |
| B6 | 809 | 1.581 | 0.1139 | 0.165 | 0.8691 | +0.0797 | 0 | 0 | +0.000 | tied |
| B7 | 67 | 2.672 | 0.0075 | 0.674 | 0.5002 | +0.0765 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

---
Wall time: 43.27s (8300 pair rows scored × 2 bakes across 3 corpora; 500 bootstrap resamples × bands).
