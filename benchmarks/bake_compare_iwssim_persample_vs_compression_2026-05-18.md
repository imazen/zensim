# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/zen/zensim-eval/exp_iwssim_persample_2026-05-18/iwssim_persample_s3_h128.bin` (label: `iwssim_persample_s3_h128`)
- **B**: `zensim/weights/v_compression_persample_2026-05-18.bin` (label: `v_compression_persample_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8406 | 0.8641 | 0.548 | 0.508 | 0.8936 | 0.9157 | -52.856 | -164.161 | -8.809 | B>>A |
| KADIK10k | 10125 | 0.9671 | 0.9316 | 0.250 | 0.362 | 0.9805 | 0.9602 | 83.726 | 734.129 | +83.726 | A>>B |
| TID2013 | 3000 | 0.9814 | 0.8893 | 0.196 | 0.432 | 0.9888 | 0.9173 | 49.266 | 315.320 | +41.055 | A>>B |
| KonJND-1k (full) | 1008 | 0.8053 | 0.8080 | 0.529 | 0.502 | 0.8493 | 0.8505 | -2.259 | -43.500 | -0.000 | promising |
| AIC-3 CTC | 600 | 0.7929 | 0.8183 | 0.592 | 0.565 | 0.8662 | 0.8856 | -36.107 | -63.797 | -0.000 | B>>A |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 17 cells
- **BDecisivelyBeatsA**: 2 cells
- **PromisingNotDecisive**: 2 cells
- **Tied**: 8 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `A`** (17 A wins vs 2 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 4292 | 0.8406 | 0.8366 | 0.6542 | 0.0436 | 0.8936 | 0.548 |
| B: v_compression_persample_2026-05-18 | 4292 | 0.8641 | 0.8614 | 0.6742 | 0.0489 | 0.9157 | 0.508 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9710 | -52.856 | 0.0000 | -164.161 | 0.0000 | -0.0221 | 1 | 5 | -8.809 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1222 | 0.1683 | 0.0915 | 0.0351 | 0.0902 | 0.986 |
| B4 | [0.40, 0.50) | 266 | 0.2217 | 0.2420 | 0.1507 | 0.0376 | 0.2854 | 0.970 |
| B5 | [0.50, 0.60) | 615 | 0.2257 | 0.2587 | 0.1531 | 0.0407 | 0.2630 | 0.966 |
| B6 | [0.60, 0.70) | 836 | 0.2449 | 0.2590 | 0.1642 | 0.0419 | 0.2873 | 0.966 |
| B7 | [0.70, 0.80) | 1092 | 0.3979 | 0.4047 | 0.2725 | 0.0385 | 0.4666 | 0.914 |
| B8 | [0.80, 0.90) | 1382 | 0.5156 | 0.5185 | 0.3514 | 0.0376 | 0.6052 | 0.855 |
| B9 | [0.90, 1.00] | 43 | 0.0699 | 0.3150 | 0.0476 | 0.0233 | 0.2322 | 0.949 |

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
| B3 | 57 | 4.286 | 0.0000 | -3.184 | 0.0015 | +0.0529 | 0 | 0 | -0.000 | tied |
| B4 | 266 | -10.177 | 0.0000 | -1.287 | 0.1979 | -0.0472 | 0 | 0 | -0.000 | tied |
| B5 | 615 | -11.247 | 0.0000 | -1.183 | 0.2368 | -0.0539 | 0 | 3 | -0.000 | promising |
| B6 | 836 | -8.124 | 0.0000 | -1.332 | 0.1827 | -0.0322 | 0 | 0 | -0.000 | tied |
| B7 | 1092 | 6.476 | 0.0000 | 2.972 | 0.0030 | +0.0246 | 0 | 0 | +0.000 | tied |
| B8 | 1382 | 10.595 | 0.0000 | 6.837 | 0.0000 | +0.0252 | 5 | 0 | +8.829 | A>>B |
| B9 | 43 | -5.717 | 0.0000 | -4.210 | 0.0000 | -0.0627 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 10125 | 0.9671 | 0.9682 | 0.8421 | 0.0397 | 0.9805 | 0.250 |
| B: v_compression_persample_2026-05-18 | 10125 | 0.9316 | 0.9321 | 0.7684 | 0.0462 | 0.9602 | 0.362 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9583 | 83.726 | 0.0000 | 734.129 | 0.0000 | +0.0203 | 6 | 0 | +83.726 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### KADIK10k 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.4408 | 0.4481 | 0.3117 | 0.0411 | 0.5398 | 0.894 |
| B1 | [0.10, 0.20) | 910 | 0.4267 | 0.4297 | 0.3020 | 0.0429 | 0.4895 | 0.903 |
| B2 | [0.20, 0.30) | 1111 | 0.3990 | 0.4041 | 0.2792 | 0.0495 | 0.4716 | 0.915 |
| B3 | [0.30, 0.40) | 1291 | 0.3428 | 0.3420 | 0.2410 | 0.0395 | 0.4092 | 0.940 |
| B4 | [0.40, 0.50) | 1013 | 0.3361 | 0.3374 | 0.2379 | 0.0503 | 0.3997 | 0.941 |
| B5 | [0.50, 0.60) | 919 | 0.3123 | 0.3171 | 0.2186 | 0.0435 | 0.3767 | 0.948 |
| B6 | [0.60, 0.70) | 936 | 0.3806 | 0.3802 | 0.2659 | 0.0459 | 0.4627 | 0.925 |
| B7 | [0.70, 0.80) | 985 | 0.3669 | 0.3714 | 0.2586 | 0.0386 | 0.4543 | 0.928 |
| B8 | [0.80, 0.90) | 1699 | 0.5096 | 0.5111 | 0.3616 | 0.0424 | 0.5996 | 0.860 |
| B9 | [0.90, 1.00] | 486 | 0.1844 | 0.1905 | 0.1289 | 0.0329 | 0.2113 | 0.982 |

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
| B0 | 705 | 10.177 | 0.0000 | 3.869 | 0.0001 | +0.2259 | 5 | 0 | +8.481 | A>>B |
| B1 | 910 | 10.577 | 0.0000 | 4.100 | 0.0000 | +0.1676 | 5 | 0 | +8.814 | A>>B |
| B2 | 1111 | 13.466 | 0.0000 | 4.330 | 0.0000 | +0.2486 | 5 | 1 | +11.221 | A>>B |
| B3 | 1291 | 11.096 | 0.0000 | 3.101 | 0.0019 | +0.1491 | 5 | 0 | +9.247 | A>>B |
| B4 | 1013 | 7.948 | 0.0000 | 2.558 | 0.0105 | +0.0998 | 5 | 0 | +6.623 | A>>B |
| B5 | 919 | 8.666 | 0.0000 | 1.989 | 0.0467 | +0.1254 | 5 | 0 | +7.222 | A>>B |
| B6 | 936 | 11.200 | 0.0000 | 3.420 | 0.0006 | +0.1824 | 5 | 0 | +9.333 | A>>B |
| B7 | 985 | 9.222 | 0.0000 | 3.329 | 0.0009 | +0.1366 | 5 | 0 | +7.685 | A>>B |
| B8 | 1699 | 14.528 | 0.0000 | 9.232 | 0.0000 | +0.0653 | 5 | 0 | +12.107 | A>>B |
| B9 | 486 | 3.280 | 0.0010 | 0.412 | 0.6801 | +0.0076 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 3000 | 0.9814 | 0.9807 | 0.8821 | 0.0443 | 0.9888 | 0.196 |
| B: v_compression_persample_2026-05-18 | 3000 | 0.8893 | 0.9018 | 0.7130 | 0.0457 | 0.9173 | 0.432 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9107 | 49.266 | 0.0000 | 315.320 | 0.0000 | +0.0715 | 5 | 0 | +41.055 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.4727 | 0.5238 | 0.3132 | 0.0345 | 0.5460 | 0.852 |
| B1 | [0.10, 0.20) | 34 | 0.5440 | 0.7549 | 0.3446 | 0.0294 | 0.5979 | 0.656 |
| B2 | [0.20, 0.30) | 185 | 0.6788 | 0.7062 | 0.4854 | 0.0432 | 0.7509 | 0.708 |
| B3 | [0.30, 0.40) | 493 | 0.8125 | 0.8192 | 0.6179 | 0.0304 | 0.8744 | 0.573 |
| B4 | [0.40, 0.50) | 677 | 0.7940 | 0.7951 | 0.5926 | 0.0561 | 0.8571 | 0.607 |
| B5 | [0.50, 0.60) | 705 | 0.7562 | 0.7553 | 0.5571 | 0.0440 | 0.8293 | 0.655 |
| B6 | [0.60, 0.70) | 809 | 0.7168 | 0.7141 | 0.5220 | 0.0445 | 0.8005 | 0.700 |
| B7 | [0.70, 0.80) | 67 | 0.4361 | 0.5043 | 0.2854 | 0.0448 | 0.4944 | 0.864 |
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
| B0 ⚠ | 29 | 2.021 | 0.0433 | 0.638 | 0.5233 | +0.3445 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | 1.249 | 0.2118 | 4.817 | 0.0000 | -0.0222 | 0 | 0 | -0.000 | tied |
| B2 | 185 | 8.212 | 0.0000 | 5.904 | 0.0000 | +0.3441 | 5 | 0 | +6.844 | A>>B |
| B3 | 493 | 16.018 | 0.0000 | 16.830 | 0.0000 | +0.3072 | 5 | 0 | +13.349 | A>>B |
| B4 | 677 | 20.208 | 0.0000 | 22.271 | 0.0000 | +0.2302 | 5 | 0 | +16.840 | A>>B |
| B5 | 705 | 19.509 | 0.0000 | 17.221 | 0.0000 | +0.2752 | 5 | 0 | +16.258 | A>>B |
| B6 | 809 | 23.629 | 0.0000 | 13.493 | 0.0000 | +0.5683 | 5 | 0 | +19.691 | A>>B |
| B7 | 67 | 1.090 | 0.2758 | 0.393 | 0.6941 | +0.0925 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 1008 | 0.8053 | 0.8483 | 0.5946 | 0.0417 | 0.8493 | 0.529 |
| B: v_compression_persample_2026-05-18 | 1008 | 0.8080 | 0.8648 | 0.5935 | 0.0456 | 0.8505 | 0.502 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9624 | -2.259 | 0.0239 | -43.500 | 0.0000 | -0.0012 | 0 | 2 | -0.000 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 600 | 0.7929 | 0.8059 | 0.6282 | 0.0517 | 0.8662 | 0.592 |
| B: v_compression_persample_2026-05-18 | 600 | 0.8183 | 0.8248 | 0.6527 | 0.0550 | 0.8856 | 0.565 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9828 | -36.107 | 0.0000 | -63.797 | 0.0000 | -0.0194 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 90.56s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
