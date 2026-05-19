# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/home/lilith/work/zen/zensim/zensim/weights/v_compression_2026-05-18.bin` (label: `v_compression_2026-05-18`)
- **B**: `/home/lilith/work/zen/zensim--v6-reship/zensim/weights/v_tuner_v6_2026-05-19.bin` (label: `v_tuner_v6_2026-05-19`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `500`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8580 | 0.8216 | 0.520 | 0.680 | 0.9126 | 0.8821 | 12.182 | 78.063 | +10.152 | A>>B |
| KonJND-1k (full) | 1008 | 0.8125 | 0.3734 | 0.498 | 0.960 | 0.8504 | 0.5297 | 14.029 | 16.073 | +11.691 | A>>B |
| TID2013 | 3000 | 0.8875 | 0.4447 | 0.436 | 0.809 | 0.9158 | 0.5451 | 52.567 | 60.496 | +43.805 | A>>B |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 7 cells
- **BDecisivelyBeatsA**: 0 cells
- **PromisingNotDecisive**: 2 cells
- **Tied**: 8 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `A`** (7 A wins vs 0 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_2026-05-18 | 4292 | 0.8580 | 0.8539 | 0.6646 | 0.0480 | 0.9126 | 0.520 |
| B: v_tuner_v6_2026-05-19 | 4292 | 0.8216 | 0.7332 | 0.6249 | 0.0531 | 0.8821 | 0.680 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.8050 | 12.182 | 0.0000 | 78.063 | 0.0000 | +0.0305 | 5 | 0 | +10.152 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0788 | 0.1584 | 0.0476 | 0.0702 | 0.0305 | 0.987 |
| B4 | [0.40, 0.50) | 266 | 0.2470 | 0.2433 | 0.1674 | 0.0451 | 0.3100 | 0.970 |
| B5 | [0.50, 0.60) | 615 | 0.2502 | 0.2640 | 0.1691 | 0.0439 | 0.3037 | 0.965 |
| B6 | [0.60, 0.70) | 836 | 0.2520 | 0.2533 | 0.1689 | 0.0383 | 0.2922 | 0.967 |
| B7 | [0.70, 0.80) | 1092 | 0.3645 | 0.3744 | 0.2491 | 0.0421 | 0.4262 | 0.927 |
| B8 | [0.80, 0.90) | 1382 | 0.4861 | 0.4924 | 0.3303 | 0.0449 | 0.5693 | 0.870 |
| B9 | [0.90, 1.00] | 43 | 0.1796 | 0.4739 | 0.1096 | 0.0698 | 0.3397 | 0.881 |

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
| B3 | 57 | 0.315 | 0.7529 | 0.026 | 0.9795 | -0.0713 | 0 | 0 | -0.000 | tied |
| B4 | 266 | 0.889 | 0.3742 | 0.095 | 0.9245 | +0.0405 | 0 | 0 | +0.000 | tied |
| B5 | 615 | -0.765 | 0.4441 | -0.003 | 0.9974 | -0.0309 | 0 | 0 | -0.000 | tied |
| B6 | 836 | -2.180 | 0.0293 | -0.878 | 0.3801 | -0.0668 | 0 | 0 | -0.000 | tied |
| B7 | 1092 | 2.209 | 0.0271 | 1.194 | 0.2325 | +0.0209 | 0 | 0 | +0.000 | tied |
| B8 | 1382 | 10.029 | 0.0000 | 5.768 | 0.0000 | +0.0737 | 5 | 1 | +8.357 | A>>B |
| B9 | 43 | 2.293 | 0.0219 | 1.320 | 0.1867 | +0.2557 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_2026-05-18 | 1008 | 0.8125 | 0.8672 | 0.6006 | 0.0417 | 0.8504 | 0.498 |
| B: v_tuner_v6_2026-05-19 | 1008 | 0.3734 | 0.2813 | 0.2584 | 0.0347 | 0.5297 | 0.960 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.0933 | 14.029 | 0.0000 | 16.073 | 0.0000 | +0.3208 | 5 | 0 | +11.691 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_2026-05-18 | 3000 | 0.8875 | 0.9000 | 0.7117 | 0.0473 | 0.9158 | 0.436 |
| B: v_tuner_v6_2026-05-19 | 3000 | 0.4447 | 0.5878 | 0.3055 | 0.0430 | 0.5451 | 0.809 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.5994 | 52.567 | 0.0000 | 60.496 | 0.0000 | +0.3708 | 5 | 0 | +43.805 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.2276 | 0.4001 | 0.1998 | 0.0345 | 0.2199 | 0.916 |
| B1 | [0.10, 0.20) | 34 | 0.4740 | 0.5906 | 0.3375 | 0.0294 | 0.6222 | 0.807 |
| B2 | [0.20, 0.30) | 185 | 0.3655 | 0.3907 | 0.2487 | 0.0324 | 0.4532 | 0.921 |
| B3 | [0.30, 0.40) | 493 | 0.4785 | 0.4883 | 0.3309 | 0.0406 | 0.5820 | 0.873 |
| B4 | [0.40, 0.50) | 677 | 0.5437 | 0.5496 | 0.3803 | 0.0443 | 0.6345 | 0.835 |
| B5 | [0.50, 0.60) | 705 | 0.4628 | 0.4876 | 0.3191 | 0.0539 | 0.5492 | 0.873 |
| B6 | [0.60, 0.70) | 809 | 0.1837 | 0.2224 | 0.1248 | 0.0420 | 0.2260 | 0.975 |
| B7 | [0.70, 0.80) | 67 | 0.3464 | 0.4803 | 0.2346 | 0.0597 | 0.3899 | 0.877 |
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
| B0 ⚠ | 29 | 0.706 | 0.4801 | -0.029 | 0.9767 | +0.2129 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -1.758 | 0.0787 | -2.199 | 0.0279 | -0.1024 | 0 | 0 | -0.000 | tied |
| B2 | 185 | 4.309 | 0.0000 | 1.243 | 0.2139 | +0.1489 | 1 | 0 | +0.718 | promising |
| B3 | 493 | 6.553 | 0.0000 | 2.666 | 0.0077 | +0.2800 | 5 | 0 | +5.461 | A>>B |
| B4 | 677 | 15.135 | 0.0000 | 6.027 | 0.0000 | +0.3710 | 5 | 0 | +12.612 | A>>B |
| B5 | 705 | 18.892 | 0.0000 | 6.225 | 0.0000 | +0.4071 | 5 | 1 | +15.744 | A>>B |
| B6 | 809 | 11.547 | 0.0000 | 1.513 | 0.1302 | +0.1998 | 5 | 0 | +9.623 | promising |
| B7 | 67 | -6.061 | 0.0000 | 0.351 | 0.7257 | -0.0878 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

---
Wall time: 34.16s (8300 pair rows scored × 2 bakes across 3 corpora; 500 bootstrap resamples × bands).
