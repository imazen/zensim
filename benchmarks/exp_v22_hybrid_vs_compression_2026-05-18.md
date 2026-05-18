# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/zen/zensim-eval/exp_v22_hybrid_2026-05-18/v22_hybrid_s3_h128_packed.bin` (label: `v22_hybrid_s3_h128_packed`)
- **B**: `/home/lilith/work/zen/zensim--exp-v22-hybrid/zensim/weights/v_compression_persample_2026-05-18.bin` (label: `v_compression_persample_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8657 | 0.8641 | 0.503 | 0.508 | 0.9173 | 0.9157 | 6.841 | 40.406 | +0.000 | tied |
| KADIK10k | 10125 | 0.9315 | 0.9316 | 0.362 | 0.362 | 0.9596 | 0.9602 | -3.337 | -24.943 | -0.000 | promising |
| TID2013 | 3000 | 0.8906 | 0.8893 | 0.431 | 0.432 | 0.9181 | 0.9173 | 40.200 | 54.364 | +13.400 | promising |
| KonJND-1k (full) | 1008 | 0.7814 | 0.8080 | 0.568 | 0.502 | 0.8284 | 0.8505 | -25.577 | -112.110 | -0.000 | B>>A |
| AIC-3 CTC | 600 | 0.8034 | 0.8183 | 0.583 | 0.565 | 0.8758 | 0.8856 | -79.916 | -161.376 | -0.000 | B>>A |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 1 cells
- **BDecisivelyBeatsA**: 4 cells
- **PromisingNotDecisive**: 4 cells
- **Tied**: 20 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (1 A wins vs 4 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_hybrid_s3_h128_packed | 4292 | 0.8657 | 0.8642 | 0.6775 | 0.0461 | 0.9173 | 0.503 |
| B: v_compression_persample_2026-05-18 | 4292 | 0.8641 | 0.8614 | 0.6742 | 0.0489 | 0.9157 | 0.508 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9845 | 6.841 | 0.0000 | 40.406 | 0.0000 | +0.0016 | 0 | 0 | +0.000 | tied |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1186 | 0.1607 | 0.0764 | 0.0526 | 0.1009 | 0.987 |
| B4 | [0.40, 0.50) | 266 | 0.2815 | 0.2774 | 0.1899 | 0.0526 | 0.3467 | 0.961 |
| B5 | [0.50, 0.60) | 615 | 0.2763 | 0.2875 | 0.1866 | 0.0439 | 0.3353 | 0.958 |
| B6 | [0.60, 0.70) | 836 | 0.2818 | 0.2826 | 0.1885 | 0.0443 | 0.3302 | 0.959 |
| B7 | [0.70, 0.80) | 1092 | 0.3707 | 0.3796 | 0.2545 | 0.0430 | 0.4302 | 0.925 |
| B8 | [0.80, 0.90) | 1382 | 0.5026 | 0.5068 | 0.3418 | 0.0478 | 0.5878 | 0.862 |
| B9 | [0.90, 1.00] | 43 | 0.1874 | 0.5350 | 0.1251 | 0.0465 | 0.3614 | 0.845 |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0825 | 0.2911 | 0.0564 | 0.0526 | 0.0373 | 0.957 |
| B4 | [0.40, 0.50) | 266 | 0.2699 | 0.2646 | 0.1801 | 0.0489 | 0.3326 | 0.964 |
| B5 | [0.50, 0.60) | 615 | 0.2656 | 0.2734 | 0.1803 | 0.0504 | 0.3170 | 0.962 |
| B6 | [0.60, 0.70) | 836 | 0.2729 | 0.2751 | 0.1817 | 0.0455 | 0.3195 | 0.961 |
| B7 | [0.70, 0.80) | 1092 | 0.3792 | 0.3864 | 0.2602 | 0.0467 | 0.4420 | 0.922 |
| B8 | [0.80, 0.90) | 1382 | 0.4971 | 0.5011 | 0.3389 | 0.0441 | 0.5801 | 0.865 |
| B9 | [0.90, 1.00] | 43 | 0.1401 | 0.4358 | 0.1030 | 0.0465 | 0.2949 | 0.900 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B2 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B3 | 57 | 4.939 | 0.0000 | -4.209 | 0.0000 | +0.0637 | 0 | 0 | -0.000 | tied |
| B4 | 266 | 3.091 | 0.0020 | 1.003 | 0.3158 | +0.0141 | 0 | 0 | +0.000 | tied |
| B5 | 615 | 3.756 | 0.0002 | 1.510 | 0.1310 | +0.0183 | 0 | 0 | +0.000 | tied |
| B6 | 836 | 3.121 | 0.0018 | 0.797 | 0.4253 | +0.0107 | 0 | 0 | +0.000 | tied |
| B7 | 1092 | -4.738 | 0.0000 | -1.693 | 0.0905 | -0.0117 | 0 | 0 | -0.000 | tied |
| B8 | 1382 | 6.871 | 0.0000 | 4.782 | 0.0000 | +0.0077 | 0 | 0 | +0.000 | tied |
| B9 | 43 | 6.596 | 0.0000 | 8.555 | 0.0000 | +0.0665 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_hybrid_s3_h128_packed | 10125 | 0.9315 | 0.9320 | 0.7675 | 0.0547 | 0.9596 | 0.362 |
| B: v_compression_persample_2026-05-18 | 10125 | 0.9316 | 0.9321 | 0.7684 | 0.0462 | 0.9602 | 0.362 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9966 | -3.337 | 0.0008 | -24.943 | 0.0000 | -0.0006 | 0 | 2 | -0.000 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### KADIK10k 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2507 | 0.2680 | 0.1751 | 0.0482 | 0.3189 | 0.963 |
| B1 | [0.10, 0.20) | 910 | 0.2469 | 0.2461 | 0.1711 | 0.0418 | 0.3080 | 0.969 |
| B2 | [0.20, 0.30) | 1111 | 0.1771 | 0.1862 | 0.1218 | 0.0360 | 0.2062 | 0.983 |
| B3 | [0.30, 0.40) | 1291 | 0.2292 | 0.2368 | 0.1595 | 0.0411 | 0.2716 | 0.972 |
| B4 | [0.40, 0.50) | 1013 | 0.2607 | 0.2605 | 0.1839 | 0.0385 | 0.3145 | 0.965 |
| B5 | [0.50, 0.60) | 919 | 0.2033 | 0.2306 | 0.1417 | 0.0435 | 0.2481 | 0.973 |
| B6 | [0.60, 0.70) | 936 | 0.2214 | 0.2404 | 0.1530 | 0.0438 | 0.2708 | 0.971 |
| B7 | [0.70, 0.80) | 985 | 0.2579 | 0.2600 | 0.1803 | 0.0416 | 0.3028 | 0.966 |
| B8 | [0.80, 0.90) | 1699 | 0.4430 | 0.4420 | 0.3110 | 0.0394 | 0.5228 | 0.897 |
| B9 | [0.90, 1.00] | 486 | 0.1623 | 0.1731 | 0.1107 | 0.0412 | 0.1921 | 0.985 |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2461 | 0.2628 | 0.1728 | 0.0539 | 0.3139 | 0.965 |
| B1 | [0.10, 0.20) | 910 | 0.2599 | 0.2621 | 0.1803 | 0.0429 | 0.3219 | 0.965 |
| B2 | [0.20, 0.30) | 1111 | 0.1918 | 0.2016 | 0.1324 | 0.0342 | 0.2230 | 0.979 |
| B3 | [0.30, 0.40) | 1291 | 0.2201 | 0.2308 | 0.1539 | 0.0442 | 0.2601 | 0.973 |
| B4 | [0.40, 0.50) | 1013 | 0.2516 | 0.2523 | 0.1779 | 0.0395 | 0.3000 | 0.968 |
| B5 | [0.50, 0.60) | 919 | 0.2035 | 0.2324 | 0.1421 | 0.0424 | 0.2513 | 0.973 |
| B6 | [0.60, 0.70) | 936 | 0.2290 | 0.2447 | 0.1583 | 0.0459 | 0.2803 | 0.970 |
| B7 | [0.70, 0.80) | 985 | 0.2718 | 0.2751 | 0.1900 | 0.0406 | 0.3177 | 0.961 |
| B8 | [0.80, 0.90) | 1699 | 0.4514 | 0.4510 | 0.3169 | 0.0400 | 0.5343 | 0.893 |
| B9 | [0.90, 1.00] | 486 | 0.1708 | 0.1816 | 0.1168 | 0.0391 | 0.2037 | 0.983 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 705 | 3.446 | 0.0006 | 1.106 | 0.2688 | +0.0050 | 0 | 0 | +0.000 | tied |
| B1 | 910 | -14.676 | 0.0000 | -4.881 | 0.0000 | -0.0139 | 0 | 0 | -0.000 | tied |
| B2 | 1111 | -19.826 | 0.0000 | -4.176 | 0.0000 | -0.0168 | 0 | 3 | -0.000 | promising |
| B3 | 1291 | 14.205 | 0.0000 | 2.321 | 0.0203 | +0.0115 | 0 | 0 | +0.000 | tied |
| B4 | 1013 | 11.038 | 0.0000 | 2.730 | 0.0063 | +0.0145 | 0 | 0 | +0.000 | tied |
| B5 | 919 | -0.255 | 0.7984 | -0.441 | 0.6592 | -0.0032 | 0 | 0 | -0.000 | tied |
| B6 | 936 | -8.237 | 0.0000 | -1.216 | 0.2241 | -0.0095 | 0 | 0 | -0.000 | tied |
| B7 | 985 | -17.316 | 0.0000 | -5.397 | 0.0000 | -0.0149 | 0 | 1 | -0.000 | promising |
| B8 | 1699 | -24.505 | 0.0000 | -14.573 | 0.0000 | -0.0115 | 0 | 5 | -0.000 | B>>A |
| B9 | 486 | -18.486 | 0.0000 | -3.383 | 0.0007 | -0.0116 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_hybrid_s3_h128_packed | 3000 | 0.8906 | 0.9021 | 0.7154 | 0.0457 | 0.9181 | 0.431 |
| B: v_compression_persample_2026-05-18 | 3000 | 0.8893 | 0.9018 | 0.7130 | 0.0457 | 0.9173 | 0.432 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9982 | 40.200 | 0.0000 | 54.364 | 0.0000 | +0.0009 | 2 | 0 | +13.400 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.2202 | 0.4530 | 0.1504 | 0.0345 | 0.2054 | 0.891 |
| B1 | [0.10, 0.20) | 34 | 0.4610 | 0.5636 | 0.3268 | 0.0588 | 0.6061 | 0.826 |
| B2 | [0.20, 0.30) | 185 | 0.3201 | 0.3549 | 0.2180 | 0.0324 | 0.3962 | 0.935 |
| B3 | [0.30, 0.40) | 493 | 0.4748 | 0.4844 | 0.3280 | 0.0365 | 0.5775 | 0.875 |
| B4 | [0.40, 0.50) | 677 | 0.5516 | 0.5568 | 0.3854 | 0.0458 | 0.6418 | 0.831 |
| B5 | [0.50, 0.60) | 705 | 0.4601 | 0.4843 | 0.3181 | 0.0482 | 0.5446 | 0.875 |
| B6 | [0.60, 0.70) | 809 | 0.1904 | 0.2160 | 0.1295 | 0.0445 | 0.2367 | 0.976 |
| B7 | [0.70, 0.80) | 67 | 0.3637 | 0.4792 | 0.2455 | 0.0746 | 0.4134 | 0.878 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1857 | 0.3541 | 0.1356 | 0.0345 | 0.2015 | 0.935 |
| B1 | [0.10, 0.20) | 34 | 0.4807 | 0.4927 | 0.3411 | 0.0294 | 0.6201 | 0.870 |
| B2 | [0.20, 0.30) | 185 | 0.3282 | 0.3478 | 0.2237 | 0.0378 | 0.4069 | 0.938 |
| B3 | [0.30, 0.40) | 493 | 0.4693 | 0.4819 | 0.3256 | 0.0365 | 0.5672 | 0.876 |
| B4 | [0.40, 0.50) | 677 | 0.5339 | 0.5398 | 0.3729 | 0.0443 | 0.6269 | 0.842 |
| B5 | [0.50, 0.60) | 705 | 0.4687 | 0.4924 | 0.3245 | 0.0525 | 0.5541 | 0.870 |
| B6 | [0.60, 0.70) | 809 | 0.1864 | 0.2186 | 0.1268 | 0.0420 | 0.2322 | 0.976 |
| B7 | [0.70, 0.80) | 67 | 0.3474 | 0.4519 | 0.2291 | 0.0746 | 0.4019 | 0.892 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 ⚠ | 29 | 1.724 | 0.0847 | 2.337 | 0.0194 | +0.0038 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | -2.700 | 0.0069 | 6.945 | 0.0000 | -0.0140 | 0 | 0 | +0.000 | tied |
| B2 | 185 | -1.870 | 0.0615 | 0.655 | 0.5122 | -0.0107 | 0 | 0 | +0.000 | tied |
| B3 | 493 | 6.560 | 0.0000 | 1.877 | 0.0605 | +0.0103 | 0 | 0 | +0.000 | tied |
| B4 | 677 | 39.851 | 0.0000 | 29.829 | 0.0000 | +0.0149 | 5 | 0 | +33.210 | A>>B |
| B5 | 705 | -30.579 | 0.0000 | -18.206 | 0.0000 | -0.0095 | 0 | 4 | -0.000 | B>>A |
| B6 | 809 | 26.296 | 0.0000 | -3.878 | 0.0001 | +0.0045 | 0 | 0 | -0.000 | tied |
| B7 | 67 | 18.317 | 0.0000 | 18.149 | 0.0000 | +0.0115 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_hybrid_s3_h128_packed | 1008 | 0.7814 | 0.8228 | 0.5686 | 0.0427 | 0.8284 | 0.568 |
| B: v_compression_persample_2026-05-18 | 1008 | 0.8080 | 0.8648 | 0.5935 | 0.0456 | 0.8505 | 0.502 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9672 | -25.577 | 0.0000 | -112.110 | 0.0000 | -0.0220 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_hybrid_s3_h128_packed | 600 | 0.8034 | 0.8125 | 0.6377 | 0.0567 | 0.8758 | 0.583 |
| B: v_compression_persample_2026-05-18 | 600 | 0.8183 | 0.8248 | 0.6527 | 0.0550 | 0.8856 | 0.565 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9955 | -79.916 | 0.0000 | -161.376 | 0.0000 | -0.0098 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 208.04s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
