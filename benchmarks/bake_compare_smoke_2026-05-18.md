# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin` (label: `v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed`)
- **B**: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_s3_h128_packed.bin` (label: `v22_mix_cv40_konjnd_0_02_s3_h128_packed`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8324 | 0.8292 | 0.559 | 0.579 | 0.9006 | 0.8927 | 2.764 | 28.327 | +1.382 | promising |
| KADIK10k | 10125 | 0.9677 | 0.9059 | 0.249 | 0.416 | 0.9804 | 0.9440 | 95.356 | 633.850 | +79.464 | A>>B |
| TID2013 | 3000 | 0.9729 | 0.8860 | 0.236 | 0.448 | 0.9832 | 0.9146 | 54.463 | 310.395 | +54.463 | A>>B |
| KonJND-1k (full) | 1008 | 0.8927 | 0.8931 | 0.376 | 0.219 | 0.9178 | 0.9204 | -0.195 | -193.866 | -0.000 | promising |
| AIC-3 CTC | 600 | 0.7845 | 0.7907 | 0.606 | 0.598 | 0.8630 | 0.8592 | -2.781 | -6.329 | +0.000 | tied |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 16 cells
- **BDecisivelyBeatsA**: 0 cells
- **PromisingNotDecisive**: 4 cells
- **Tied**: 9 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `A`** (16 A wins vs 0 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed | 4292 | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |
| B: v22_mix_cv40_konjnd_0_02_s3_h128_packed | 4292 | 0.8292 | 0.8156 | 0.6301 | 0.0447 | 0.8927 | 0.579 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9236 | 2.764 | 0.0057 | 28.327 | 0.0000 | +0.0079 | 3 | 0 | +1.382 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0487 | 0.1900 | 0.0276 | 0.0351 | 0.0703 | 0.982 |
| B4 | [0.40, 0.50) | 266 | 0.2510 | 0.2452 | 0.1658 | 0.0489 | 0.3169 | 0.969 |
| B5 | [0.50, 0.60) | 615 | 0.2466 | 0.2584 | 0.1655 | 0.0423 | 0.3049 | 0.966 |
| B6 | [0.60, 0.70) | 836 | 0.2220 | 0.2317 | 0.1484 | 0.0347 | 0.2580 | 0.973 |
| B7 | [0.70, 0.80) | 1092 | 0.3192 | 0.3244 | 0.2173 | 0.0513 | 0.3766 | 0.946 |
| B8 | [0.80, 0.90) | 1382 | 0.4958 | 0.5026 | 0.3354 | 0.0347 | 0.5846 | 0.865 |
| B9 | [0.90, 1.00] | 43 | 0.1056 | 0.1831 | 0.0698 | 0.0233 | 0.2725 | 0.983 |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0806 | 0.2967 | 0.0589 | 0.0351 | 0.0758 | 0.955 |
| B4 | [0.40, 0.50) | 266 | 0.2036 | 0.2023 | 0.1361 | 0.0451 | 0.2601 | 0.979 |
| B5 | [0.50, 0.60) | 615 | 0.1555 | 0.2188 | 0.1058 | 0.0455 | 0.1863 | 0.976 |
| B6 | [0.60, 0.70) | 836 | 0.1485 | 0.1568 | 0.0984 | 0.0419 | 0.1701 | 0.988 |
| B7 | [0.70, 0.80) | 1092 | 0.3502 | 0.3519 | 0.2395 | 0.0485 | 0.4070 | 0.936 |
| B8 | [0.80, 0.90) | 1382 | 0.5041 | 0.5098 | 0.3433 | 0.0441 | 0.5890 | 0.860 |
| B9 | [0.90, 1.00] | 43 | 0.1104 | 0.3942 | 0.0653 | 0.0698 | 0.2578 | 0.919 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B2 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B3 | 57 | -0.128 | 0.8978 | -0.110 | 0.9124 | -0.0054 | 0 | 0 | -0.000 | tied |
| B4 | 266 | 3.133 | 0.0017 | 0.666 | 0.5057 | +0.0568 | 0 | 0 | +0.000 | tied |
| B5 | 615 | 6.275 | 0.0000 | 0.687 | 0.4919 | +0.1186 | 3 | 0 | +3.137 | promising |
| B6 | 836 | 6.556 | 0.0000 | 1.341 | 0.1799 | +0.0880 | 5 | 0 | +5.463 | promising |
| B7 | 1092 | -5.848 | 0.0000 | -1.970 | 0.0489 | -0.0305 | 0 | 0 | -0.000 | tied |
| B8 | 1382 | -2.810 | 0.0050 | -1.664 | 0.0961 | -0.0044 | 0 | 0 | -0.000 | tied |
| B9 | 43 | -0.226 | 0.8211 | -3.048 | 0.0023 | +0.0147 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed | 10125 | 0.9677 | 0.9686 | 0.8432 | 0.0433 | 0.9804 | 0.249 |
| B: v22_mix_cv40_konjnd_0_02_s3_h128_packed | 10125 | 0.9059 | 0.9092 | 0.7276 | 0.0297 | 0.9440 | 0.416 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9380 | 95.356 | 0.0000 | 633.850 | 0.0000 | +0.0364 | 5 | 1 | +79.464 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### KADIK10k 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.4150 | 0.4235 | 0.2939 | 0.0241 | 0.5015 | 0.906 |
| B1 | [0.10, 0.20) | 910 | 0.4149 | 0.4191 | 0.2922 | 0.0374 | 0.4853 | 0.908 |
| B2 | [0.20, 0.30) | 1111 | 0.3995 | 0.4047 | 0.2807 | 0.0387 | 0.4741 | 0.914 |
| B3 | [0.30, 0.40) | 1291 | 0.3342 | 0.3325 | 0.2352 | 0.0418 | 0.3953 | 0.943 |
| B4 | [0.40, 0.50) | 1013 | 0.3754 | 0.3801 | 0.2666 | 0.0444 | 0.4405 | 0.925 |
| B5 | [0.50, 0.60) | 919 | 0.3454 | 0.3527 | 0.2442 | 0.0457 | 0.4190 | 0.936 |
| B6 | [0.60, 0.70) | 936 | 0.3649 | 0.3643 | 0.2549 | 0.0438 | 0.4434 | 0.931 |
| B7 | [0.70, 0.80) | 985 | 0.3603 | 0.3669 | 0.2552 | 0.0396 | 0.4420 | 0.930 |
| B8 | [0.80, 0.90) | 1699 | 0.5019 | 0.5025 | 0.3554 | 0.0383 | 0.5871 | 0.865 |
| B9 | [0.90, 1.00] | 486 | 0.1818 | 0.2299 | 0.1248 | 0.0370 | 0.2158 | 0.973 |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2105 | 0.2187 | 0.1477 | 0.0326 | 0.2657 | 0.976 |
| B1 | [0.10, 0.20) | 910 | 0.1626 | 0.1784 | 0.1132 | 0.0363 | 0.2090 | 0.984 |
| B2 | [0.20, 0.30) | 1111 | 0.0895 | 0.1060 | 0.0612 | 0.0396 | 0.1007 | 0.994 |
| B3 | [0.30, 0.40) | 1291 | 0.2030 | 0.2234 | 0.1409 | 0.0457 | 0.2484 | 0.975 |
| B4 | [0.40, 0.50) | 1013 | 0.2128 | 0.2180 | 0.1496 | 0.0365 | 0.2540 | 0.976 |
| B5 | [0.50, 0.60) | 919 | 0.1786 | 0.2117 | 0.1242 | 0.0446 | 0.2266 | 0.977 |
| B6 | [0.60, 0.70) | 936 | 0.1972 | 0.2210 | 0.1359 | 0.0374 | 0.2433 | 0.975 |
| B7 | [0.70, 0.80) | 985 | 0.2655 | 0.2711 | 0.1858 | 0.0386 | 0.3117 | 0.963 |
| B8 | [0.80, 0.90) | 1699 | 0.4322 | 0.4335 | 0.3030 | 0.0365 | 0.5136 | 0.901 |
| B9 | [0.90, 1.00] | 486 | 0.1651 | 0.1997 | 0.1153 | 0.0350 | 0.1814 | 0.980 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 705 | 9.497 | 0.0000 | 3.352 | 0.0008 | +0.2358 | 5 | 0 | +7.914 | A>>B |
| B1 | 910 | 12.164 | 0.0000 | 3.770 | 0.0002 | +0.2763 | 5 | 0 | +10.136 | A>>B |
| B2 | 1111 | 14.932 | 0.0000 | 3.939 | 0.0001 | +0.3735 | 5 | 0 | +12.444 | A>>B |
| B3 | 1291 | 11.656 | 0.0000 | 2.891 | 0.0038 | +0.1469 | 5 | 0 | +9.713 | A>>B |
| B4 | 1013 | 14.290 | 0.0000 | 4.616 | 0.0000 | +0.1865 | 5 | 0 | +11.908 | A>>B |
| B5 | 919 | 11.632 | 0.0000 | 2.983 | 0.0029 | +0.1923 | 5 | 0 | +9.693 | A>>B |
| B6 | 936 | 11.614 | 0.0000 | 3.142 | 0.0017 | +0.2001 | 5 | 0 | +9.678 | A>>B |
| B7 | 985 | 10.260 | 0.0000 | 3.633 | 0.0003 | +0.1303 | 5 | 0 | +8.550 | A>>B |
| B8 | 1699 | 21.863 | 0.0000 | 12.741 | 0.0000 | +0.0736 | 5 | 0 | +18.219 | A>>B |
| B9 | 486 | 3.483 | 0.0005 | 1.415 | 0.1570 | +0.0344 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed | 3000 | 0.9729 | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |
| B: v22_mix_cv40_konjnd_0_02_s3_h128_packed | 3000 | 0.8860 | 0.8939 | 0.7062 | 0.0467 | 0.9146 | 0.448 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9201 | 54.463 | 0.0000 | 310.395 | 0.0000 | +0.0686 | 6 | 0 | +54.463 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0621 | 0.3495 | 0.0666 | 0.0345 | 0.0778 | 0.937 |
| B1 | [0.10, 0.20) | 34 | 0.5319 | 0.5997 | 0.3411 | 0.0294 | 0.7017 | 0.800 |
| B2 | [0.20, 0.30) | 185 | 0.6479 | 0.6523 | 0.4607 | 0.0378 | 0.7180 | 0.758 |
| B3 | [0.30, 0.40) | 493 | 0.7352 | 0.7356 | 0.5396 | 0.0446 | 0.8173 | 0.677 |
| B4 | [0.40, 0.50) | 677 | 0.7625 | 0.7625 | 0.5598 | 0.0502 | 0.8370 | 0.647 |
| B5 | [0.50, 0.60) | 705 | 0.7077 | 0.7068 | 0.5096 | 0.0496 | 0.7907 | 0.707 |
| B6 | [0.60, 0.70) | 809 | 0.5860 | 0.5854 | 0.4099 | 0.0346 | 0.6825 | 0.811 |
| B7 | [0.70, 0.80) | 67 | 0.2842 | 0.4937 | 0.1928 | 0.0746 | 0.2801 | 0.870 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.2611 | 0.4258 | 0.1850 | 0.0345 | 0.2536 | 0.905 |
| B1 | [0.10, 0.20) | 34 | 0.5348 | 0.5727 | 0.3589 | 0.0588 | 0.6461 | 0.820 |
| B2 | [0.20, 0.30) | 185 | 0.3002 | 0.4123 | 0.2032 | 0.0541 | 0.3682 | 0.911 |
| B3 | [0.30, 0.40) | 493 | 0.4354 | 0.4423 | 0.3015 | 0.0467 | 0.5327 | 0.897 |
| B4 | [0.40, 0.50) | 677 | 0.5334 | 0.5370 | 0.3729 | 0.0487 | 0.6157 | 0.844 |
| B5 | [0.50, 0.60) | 705 | 0.4624 | 0.4902 | 0.3194 | 0.0454 | 0.5484 | 0.872 |
| B6 | [0.60, 0.70) | 809 | 0.1763 | 0.2060 | 0.1189 | 0.0383 | 0.2185 | 0.979 |
| B7 | [0.70, 0.80) | 67 | 0.3673 | 0.4634 | 0.2427 | 0.0746 | 0.4211 | 0.886 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 ⚠ | 29 | -1.795 | 0.0727 | -0.311 | 0.7556 | -0.1759 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -0.090 | 0.9279 | 0.740 | 0.4593 | +0.0556 | 0 | 0 | -0.000 | tied |
| B2 | 185 | 10.548 | 0.0000 | 5.257 | 0.0000 | +0.3498 | 5 | 0 | +8.790 | A>>B |
| B3 | 493 | 16.009 | 0.0000 | 13.424 | 0.0000 | +0.2845 | 5 | 0 | +13.341 | A>>B |
| B4 | 677 | 20.279 | 0.0000 | 21.003 | 0.0000 | +0.2212 | 5 | 0 | +16.899 | A>>B |
| B5 | 705 | 18.684 | 0.0000 | 14.607 | 0.0000 | +0.2424 | 5 | 0 | +15.570 | A>>B |
| B6 | 809 | 23.907 | 0.0000 | 10.368 | 0.0000 | +0.4640 | 5 | 0 | +19.922 | A>>B |
| B7 | 67 | -0.687 | 0.4922 | 0.154 | 0.8774 | -0.1410 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed | 1008 | 0.8927 | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |
| B: v22_mix_cv40_konjnd_0_02_s3_h128_packed | 1008 | 0.8931 | 0.9756 | 0.7216 | 0.0357 | 0.9204 | 0.219 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9340 | -0.195 | 0.8452 | -193.866 | 0.0000 | -0.0026 | 0 | 2 | -0.000 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed | 600 | 0.7845 | 0.7953 | 0.6155 | 0.0433 | 0.8630 | 0.606 |
| B: v22_mix_cv40_konjnd_0_02_s3_h128_packed | 600 | 0.7907 | 0.8017 | 0.6244 | 0.0533 | 0.8592 | 0.598 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9457 | -2.781 | 0.0054 | -6.329 | 0.0000 | +0.0038 | 0 | 0 | +0.000 | tied |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 113.23s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
