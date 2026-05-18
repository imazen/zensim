# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/exmix3_cv30_iw40_sm30_s3_h128_packed.bin` (label: `exmix3_cv30_iw40_sm30_s3_h128_packed`)
- **B**: `/home/lilith/work/zen/zensim--persample-runtime/zensim/weights/v_compression_persample_2026-05-18.bin` (label: `v_compression_persample_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8642 | 0.8641 | 0.506 | 0.508 | 0.9191 | 0.9157 | 0.357 | 10.209 | +0.000 | tied |
| KADIK10k | 10125 | 0.9255 | 0.9316 | 0.378 | 0.362 | 0.9558 | 0.9602 | -103.043 | -689.747 | -0.000 | B>>A |
| TID2013 | 3000 | 0.8776 | 0.8893 | 0.461 | 0.432 | 0.9101 | 0.9173 | -91.881 | -489.429 | -0.000 | B>>A |
| KonJND-1k (full) | 1008 | 0.8424 | 0.8080 | 0.448 | 0.502 | 0.8775 | 0.8505 | 26.879 | 84.379 | +22.399 | A>>B |
| AIC-3 CTC | 600 | 0.8048 | 0.8183 | 0.580 | 0.565 | 0.8789 | 0.8856 | -37.391 | -69.358 | -0.000 | B>>A |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 1 cells
- **BDecisivelyBeatsA**: 11 cells
- **PromisingNotDecisive**: 1 cells
- **Tied**: 16 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (1 A wins vs 11 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: exmix3_cv30_iw40_sm30_s3_h128_packed | 4292 | 0.8642 | 0.8628 | 0.6719 | 0.0480 | 0.9191 | 0.506 |
| B: v_compression_persample_2026-05-18 | 4292 | 0.8641 | 0.8614 | 0.6742 | 0.0489 | 0.9157 | 0.508 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9705 | 0.357 | 0.7211 | 10.209 | 0.0000 | +0.0034 | 0 | 0 | +0.000 | tied |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0377 | 0.1600 | 0.0226 | 0.0351 | 0.0930 | 0.987 |
| B4 | [0.40, 0.50) | 266 | 0.2775 | 0.2681 | 0.1837 | 0.0451 | 0.3489 | 0.963 |
| B5 | [0.50, 0.60) | 615 | 0.2773 | 0.2903 | 0.1866 | 0.0390 | 0.3453 | 0.957 |
| B6 | [0.60, 0.70) | 836 | 0.2440 | 0.2466 | 0.1623 | 0.0347 | 0.2795 | 0.969 |
| B7 | [0.70, 0.80) | 1092 | 0.3820 | 0.3869 | 0.2619 | 0.0513 | 0.4475 | 0.922 |
| B8 | [0.80, 0.90) | 1382 | 0.4825 | 0.4858 | 0.3283 | 0.0427 | 0.5625 | 0.874 |
| B9 | [0.90, 1.00] | 43 | 0.1131 | 0.4226 | 0.0742 | 0.0698 | 0.2607 | 0.906 |

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
| B3 | 57 | -0.175 | 0.8613 | -0.120 | 0.9043 | +0.0557 | 0 | 0 | +0.000 | tied |
| B4 | 266 | 1.100 | 0.2712 | 0.149 | 0.8819 | +0.0163 | 0 | 0 | +0.000 | tied |
| B5 | 615 | 2.111 | 0.0347 | 0.928 | 0.3536 | +0.0283 | 0 | 0 | +0.000 | tied |
| B6 | 836 | -5.304 | 0.0000 | -1.456 | 0.1454 | -0.0399 | 0 | 0 | -0.000 | tied |
| B7 | 1092 | 0.933 | 0.3510 | 0.080 | 0.9360 | +0.0055 | 0 | 0 | +0.000 | tied |
| B8 | 1382 | -10.848 | 0.0000 | -7.377 | 0.0000 | -0.0175 | 0 | 5 | -0.000 | B>>A |
| B9 | 43 | -1.333 | 0.1824 | -0.341 | 0.7333 | -0.0342 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: exmix3_cv30_iw40_sm30_s3_h128_packed | 10125 | 0.9255 | 0.9260 | 0.7578 | 0.0527 | 0.9558 | 0.378 |
| B: v_compression_persample_2026-05-18 | 10125 | 0.9316 | 0.9321 | 0.7684 | 0.0462 | 0.9602 | 0.362 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9940 | -103.043 | 0.0000 | -689.747 | 0.0000 | -0.0044 | 0 | 6 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### KADIK10k 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2396 | 0.2681 | 0.1676 | 0.0525 | 0.3097 | 0.963 |
| B1 | [0.10, 0.20) | 910 | 0.2469 | 0.2530 | 0.1707 | 0.0385 | 0.3080 | 0.967 |
| B2 | [0.20, 0.30) | 1111 | 0.1484 | 0.1585 | 0.1025 | 0.0378 | 0.1711 | 0.987 |
| B3 | [0.30, 0.40) | 1291 | 0.2240 | 0.2296 | 0.1561 | 0.0426 | 0.2618 | 0.973 |
| B4 | [0.40, 0.50) | 1013 | 0.2582 | 0.2589 | 0.1820 | 0.0346 | 0.3060 | 0.966 |
| B5 | [0.50, 0.60) | 919 | 0.1866 | 0.2165 | 0.1299 | 0.0392 | 0.2291 | 0.976 |
| B6 | [0.60, 0.70) | 936 | 0.1902 | 0.2100 | 0.1312 | 0.0427 | 0.2388 | 0.978 |
| B7 | [0.70, 0.80) | 985 | 0.2473 | 0.2509 | 0.1735 | 0.0447 | 0.2905 | 0.968 |
| B8 | [0.80, 0.90) | 1699 | 0.4319 | 0.4318 | 0.3028 | 0.0406 | 0.5126 | 0.902 |
| B9 | [0.90, 1.00] | 486 | 0.1549 | 0.1825 | 0.1047 | 0.0391 | 0.1815 | 0.983 |

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
| B0 | 705 | -3.006 | 0.0026 | 0.701 | 0.4836 | -0.0042 | 0 | 0 | +0.000 | tied |
| B1 | 910 | -9.325 | 0.0000 | -1.777 | 0.0756 | -0.0139 | 0 | 0 | -0.000 | tied |
| B2 | 1111 | -30.794 | 0.0000 | -5.669 | 0.0000 | -0.0519 | 0 | 5 | -0.000 | B>>A |
| B3 | 1291 | 3.228 | 0.0012 | -0.244 | 0.8075 | +0.0016 | 0 | 0 | -0.000 | tied |
| B4 | 1013 | 4.419 | 0.0000 | 1.219 | 0.2229 | +0.0061 | 0 | 0 | +0.000 | tied |
| B5 | 919 | -11.016 | 0.0000 | -2.449 | 0.0143 | -0.0222 | 0 | 0 | -0.000 | tied |
| B6 | 936 | -27.381 | 0.0000 | -5.849 | 0.0000 | -0.0414 | 0 | 5 | -0.000 | B>>A |
| B7 | 985 | -21.755 | 0.0000 | -6.059 | 0.0000 | -0.0272 | 0 | 5 | -0.000 | B>>A |
| B8 | 1699 | -30.746 | 0.0000 | -16.480 | 0.0000 | -0.0217 | 0 | 5 | -0.000 | B>>A |
| B9 | 486 | -13.966 | 0.0000 | 0.144 | 0.8856 | -0.0222 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: exmix3_cv30_iw40_sm30_s3_h128_packed | 3000 | 0.8776 | 0.8874 | 0.6924 | 0.0450 | 0.9101 | 0.461 |
| B: v_compression_persample_2026-05-18 | 3000 | 0.8893 | 0.9018 | 0.7130 | 0.0457 | 0.9173 | 0.432 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9930 | -91.881 | 0.0000 | -489.429 | 0.0000 | -0.0072 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1951 | 0.3487 | 0.1307 | 0.0000 | 0.2466 | 0.937 |
| B1 | [0.10, 0.20) | 34 | 0.4684 | 0.5581 | 0.3304 | 0.0588 | 0.6265 | 0.830 |
| B2 | [0.20, 0.30) | 185 | 0.2683 | 0.3429 | 0.1832 | 0.0432 | 0.3356 | 0.939 |
| B3 | [0.30, 0.40) | 493 | 0.3755 | 0.3831 | 0.2559 | 0.0385 | 0.4706 | 0.924 |
| B4 | [0.40, 0.50) | 677 | 0.4826 | 0.4839 | 0.3314 | 0.0487 | 0.5712 | 0.875 |
| B5 | [0.50, 0.60) | 705 | 0.4433 | 0.4584 | 0.3038 | 0.0496 | 0.5276 | 0.889 |
| B6 | [0.60, 0.70) | 809 | 0.1499 | 0.1859 | 0.1015 | 0.0346 | 0.1830 | 0.983 |
| B7 | [0.70, 0.80) | 67 | 0.3541 | 0.4780 | 0.2400 | 0.0448 | 0.3950 | 0.878 |
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
| B0 ⚠ | 29 | 0.897 | 0.3696 | -0.206 | 0.8365 | +0.0451 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -1.206 | 0.2278 | 4.522 | 0.0000 | +0.0064 | 0 | 0 | -0.000 | tied |
| B2 | 185 | -5.832 | 0.0000 | -0.186 | 0.8527 | -0.0712 | 0 | 0 | -0.000 | tied |
| B3 | 493 | -30.836 | 0.0000 | -16.936 | 0.0000 | -0.0966 | 0 | 5 | -0.000 | B>>A |
| B4 | 677 | -30.915 | 0.0000 | -22.963 | 0.0000 | -0.0557 | 0 | 5 | -0.000 | B>>A |
| B5 | 705 | -26.467 | 0.0000 | -21.511 | 0.0000 | -0.0266 | 0 | 5 | -0.000 | B>>A |
| B6 | 809 | -63.061 | 0.0000 | -11.868 | 0.0000 | -0.0492 | 0 | 3 | -0.000 | promising |
| B7 | 67 | 3.503 | 0.0005 | 7.978 | 0.0000 | -0.0069 | 0 | 0 | -0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: exmix3_cv30_iw40_sm30_s3_h128_packed | 1008 | 0.8424 | 0.8941 | 0.6373 | 0.0456 | 0.8775 | 0.448 |
| B: v_compression_persample_2026-05-18 | 1008 | 0.8080 | 0.8648 | 0.5935 | 0.0456 | 0.8505 | 0.502 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9595 | 26.879 | 0.0000 | 84.379 | 0.0000 | +0.0270 | 5 | 0 | +22.399 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: exmix3_cv30_iw40_sm30_s3_h128_packed | 600 | 0.8048 | 0.8147 | 0.6385 | 0.0500 | 0.8789 | 0.580 |
| B: v_compression_persample_2026-05-18 | 600 | 0.8183 | 0.8248 | 0.6527 | 0.0550 | 0.8856 | 0.565 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9912 | -37.391 | 0.0000 | -69.358 | 0.0000 | -0.0067 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 209.68s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
