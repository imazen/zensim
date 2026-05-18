# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/tmp/v24_hybrid_s4_packed_f16.bin` (label: `v24_hybrid_s4_packed_f16`)
- **B**: `/home/lilith/work/zen/zensim--hybrid-runtime/zensim/weights/v_compression_persample_2026-05-18.bin` (label: `v_compression_persample_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8657 | 0.8641 | 0.504 | 0.508 | 0.9171 | 0.9157 | 5.538 | 24.182 | +0.000 | tied |
| KADIK10k | 10125 | 0.9285 | 0.9316 | 0.370 | 0.362 | 0.9580 | 0.9602 | -59.168 | -430.015 | -0.000 | B>>A |
| TID2013 | 3000 | 0.8890 | 0.8893 | 0.438 | 0.432 | 0.9182 | 0.9173 | -5.444 | -251.697 | +0.000 | promising |
| KonJND-1k (full) | 1008 | 0.7901 | 0.8080 | 0.558 | 0.502 | 0.8343 | 0.8505 | -20.690 | -114.889 | -0.000 | B>>A |
| AIC-3 CTC | 600 | 0.8061 | 0.8183 | 0.579 | 0.565 | 0.8776 | 0.8856 | -71.893 | -137.393 | -0.000 | B>>A |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 0 cells
- **BDecisivelyBeatsA**: 5 cells
- **PromisingNotDecisive**: 5 cells
- **Tied**: 19 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (0 A wins vs 5 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_s4_packed_f16 | 4292 | 0.8657 | 0.8635 | 0.6767 | 0.0485 | 0.9171 | 0.504 |
| B: v_compression_persample_2026-05-18 | 4292 | 0.8641 | 0.8614 | 0.6742 | 0.0489 | 0.9157 | 0.508 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9809 | 5.538 | 0.0000 | 24.182 | 0.0000 | +0.0013 | 0 | 0 | +0.000 | tied |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0854 | 0.1584 | 0.0551 | 0.0702 | 0.0470 | 0.987 |
| B4 | [0.40, 0.50) | 266 | 0.2469 | 0.2474 | 0.1664 | 0.0451 | 0.2996 | 0.969 |
| B5 | [0.50, 0.60) | 615 | 0.2550 | 0.2662 | 0.1720 | 0.0423 | 0.3016 | 0.964 |
| B6 | [0.60, 0.70) | 836 | 0.2897 | 0.2906 | 0.1936 | 0.0371 | 0.3377 | 0.957 |
| B7 | [0.70, 0.80) | 1092 | 0.3843 | 0.3913 | 0.2637 | 0.0485 | 0.4491 | 0.920 |
| B8 | [0.80, 0.90) | 1382 | 0.4964 | 0.4999 | 0.3373 | 0.0434 | 0.5808 | 0.866 |
| B9 | [0.90, 1.00] | 43 | 0.1915 | 0.5440 | 0.1251 | 0.0465 | 0.3600 | 0.839 |

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
| B3 | 57 | 0.527 | 0.5979 | -5.617 | 0.0000 | +0.0097 | 0 | 0 | -0.000 | tied |
| B4 | 266 | -6.365 | 0.0000 | -1.295 | 0.1953 | -0.0330 | 0 | 0 | -0.000 | tied |
| B5 | 615 | -3.538 | 0.0004 | -0.695 | 0.4872 | -0.0154 | 0 | 0 | -0.000 | tied |
| B6 | 836 | 4.231 | 0.0000 | 1.197 | 0.2312 | +0.0182 | 0 | 0 | +0.000 | tied |
| B7 | 1092 | 2.281 | 0.0226 | 1.015 | 0.3102 | +0.0071 | 0 | 0 | +0.000 | tied |
| B8 | 1382 | -0.723 | 0.4697 | -0.838 | 0.4019 | +0.0008 | 0 | 0 | +0.000 | tied |
| B9 | 43 | 6.553 | 0.0000 | 8.642 | 0.0000 | +0.0652 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_s4_packed_f16 | 10125 | 0.9285 | 0.9289 | 0.7629 | 0.0538 | 0.9580 | 0.370 |
| B: v_compression_persample_2026-05-18 | 10125 | 0.9316 | 0.9321 | 0.7684 | 0.0462 | 0.9602 | 0.362 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9948 | -59.168 | 0.0000 | -430.015 | 0.0000 | -0.0022 | 0 | 6 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### KADIK10k 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2288 | 0.2589 | 0.1613 | 0.0440 | 0.2899 | 0.966 |
| B1 | [0.10, 0.20) | 910 | 0.2363 | 0.2363 | 0.1640 | 0.0418 | 0.2981 | 0.972 |
| B2 | [0.20, 0.30) | 1111 | 0.1756 | 0.1842 | 0.1209 | 0.0378 | 0.2057 | 0.983 |
| B3 | [0.30, 0.40) | 1291 | 0.2292 | 0.2404 | 0.1595 | 0.0418 | 0.2712 | 0.971 |
| B4 | [0.40, 0.50) | 1013 | 0.2488 | 0.2492 | 0.1746 | 0.0355 | 0.2989 | 0.968 |
| B5 | [0.50, 0.60) | 919 | 0.2030 | 0.2405 | 0.1417 | 0.0403 | 0.2478 | 0.971 |
| B6 | [0.60, 0.70) | 936 | 0.2120 | 0.2390 | 0.1463 | 0.0449 | 0.2593 | 0.971 |
| B7 | [0.70, 0.80) | 985 | 0.2569 | 0.2595 | 0.1792 | 0.0396 | 0.3047 | 0.966 |
| B8 | [0.80, 0.90) | 1699 | 0.4493 | 0.4488 | 0.3157 | 0.0406 | 0.5317 | 0.894 |
| B9 | [0.90, 1.00] | 486 | 0.1662 | 0.1852 | 0.1138 | 0.0391 | 0.1992 | 0.983 |

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
| B0 | 705 | -10.280 | 0.0000 | -0.647 | 0.5177 | -0.0240 | 0 | 0 | -0.000 | tied |
| B1 | 910 | -17.440 | 0.0000 | -5.024 | 0.0000 | -0.0239 | 0 | 4 | -0.000 | B>>A |
| B2 | 1111 | -14.234 | 0.0000 | -3.074 | 0.0021 | -0.0172 | 0 | 2 | -0.000 | promising |
| B3 | 1291 | 8.241 | 0.0000 | 2.165 | 0.0304 | +0.0111 | 0 | 0 | +0.000 | tied |
| B4 | 1013 | -1.955 | 0.0506 | -0.582 | 0.5606 | -0.0011 | 0 | 0 | -0.000 | tied |
| B5 | 919 | -0.356 | 0.7219 | 1.470 | 0.1416 | -0.0034 | 0 | 0 | +0.000 | tied |
| B6 | 936 | -12.838 | 0.0000 | -1.111 | 0.2664 | -0.0209 | 0 | 1 | -0.000 | promising |
| B7 | 985 | -14.072 | 0.0000 | -4.241 | 0.0000 | -0.0130 | 0 | 2 | -0.000 | promising |
| B8 | 1699 | -5.302 | 0.0000 | -3.082 | 0.0021 | -0.0026 | 0 | 0 | -0.000 | tied |
| B9 | 486 | -8.820 | 0.0000 | 1.305 | 0.1918 | -0.0045 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_s4_packed_f16 | 3000 | 0.8890 | 0.8990 | 0.7108 | 0.0457 | 0.9182 | 0.438 |
| B: v_compression_persample_2026-05-18 | 3000 | 0.8893 | 0.9018 | 0.7130 | 0.0457 | 0.9173 | 0.432 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9971 | -5.444 | 0.0000 | -251.697 | 0.0000 | +0.0009 | 0 | 3 | +0.000 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1919 | 0.2992 | 0.1554 | 0.0345 | 0.1278 | 0.954 |
| B1 | [0.10, 0.20) | 34 | 0.4299 | 0.5906 | 0.2911 | 0.0882 | 0.5679 | 0.807 |
| B2 | [0.20, 0.30) | 185 | 0.2656 | 0.2824 | 0.1770 | 0.0378 | 0.3416 | 0.959 |
| B3 | [0.30, 0.40) | 493 | 0.4587 | 0.4646 | 0.3156 | 0.0365 | 0.5574 | 0.886 |
| B4 | [0.40, 0.50) | 677 | 0.5345 | 0.5400 | 0.3727 | 0.0502 | 0.6236 | 0.842 |
| B5 | [0.50, 0.60) | 705 | 0.4478 | 0.4732 | 0.3083 | 0.0553 | 0.5313 | 0.881 |
| B6 | [0.60, 0.70) | 809 | 0.1947 | 0.2225 | 0.1320 | 0.0358 | 0.2407 | 0.975 |
| B7 | [0.70, 0.80) | 67 | 0.3580 | 0.4792 | 0.2436 | 0.0597 | 0.4078 | 0.878 |
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
| B0 ⚠ | 29 | 0.265 | 0.7914 | -0.854 | 0.3932 | -0.0737 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | -10.529 | 0.0000 | 15.034 | 0.0000 | -0.0522 | 0 | 0 | +0.000 | tied |
| B2 | 185 | -11.688 | 0.0000 | -4.224 | 0.0000 | -0.0653 | 0 | 3 | -0.000 | promising |
| B3 | 493 | -9.217 | 0.0000 | -9.054 | 0.0000 | -0.0097 | 0 | 0 | -0.000 | tied |
| B4 | 677 | 0.960 | 0.3372 | 0.250 | 0.8027 | -0.0033 | 0 | 0 | -0.000 | tied |
| B5 | 705 | -53.137 | 0.0000 | -30.603 | 0.0000 | -0.0228 | 0 | 5 | -0.000 | B>>A |
| B6 | 809 | 38.275 | 0.0000 | 4.233 | 0.0000 | +0.0085 | 0 | 0 | +0.000 | tied |
| B7 | 67 | 11.608 | 0.0000 | 17.653 | 0.0000 | +0.0059 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_s4_packed_f16 | 1008 | 0.7901 | 0.8302 | 0.5791 | 0.0456 | 0.8343 | 0.558 |
| B: v_compression_persample_2026-05-18 | 1008 | 0.8080 | 0.8648 | 0.5935 | 0.0456 | 0.8505 | 0.502 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9726 | -20.690 | 0.0000 | -114.889 | 0.0000 | -0.0161 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_s4_packed_f16 | 600 | 0.8061 | 0.8154 | 0.6392 | 0.0567 | 0.8776 | 0.579 |
| B: v_compression_persample_2026-05-18 | 600 | 0.8183 | 0.8248 | 0.6527 | 0.0550 | 0.8856 | 0.565 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9959 | -71.893 | 0.0000 | -137.393 | 0.0000 | -0.0080 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 98.33s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
