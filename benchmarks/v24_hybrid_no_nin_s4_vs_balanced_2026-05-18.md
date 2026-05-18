# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/tmp/v24_hybrid_s4_packed_f16.bin` (label: `v24_hybrid_s4_packed_f16`)
- **B**: `/home/lilith/work/zen/zensim--hybrid-runtime/zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` (label: `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8657 | 0.8324 | 0.504 | 0.559 | 0.9171 | 0.9006 | 41.722 | 123.494 | +34.769 | A>>B |
| KADIK10k | 10125 | 0.9285 | 0.9677 | 0.370 | 0.249 | 0.9580 | 0.9804 | -90.670 | -766.955 | -0.000 | B>>A |
| TID2013 | 3000 | 0.8890 | 0.9729 | 0.438 | 0.236 | 0.9182 | 0.9832 | -54.113 | -311.042 | -0.000 | B>>A |
| KonJND-1k (full) | 1008 | 0.7901 | 0.8927 | 0.558 | 0.376 | 0.8343 | 0.9178 | -44.031 | -139.868 | -0.000 | B>>A |
| AIC-3 CTC | 600 | 0.8061 | 0.7845 | 0.579 | 0.606 | 0.8776 | 0.8630 | 13.935 | 29.097 | +11.612 | A>>B |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 4 cells
- **BDecisivelyBeatsA**: 17 cells
- **PromisingNotDecisive**: 1 cells
- **Tied**: 7 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (4 A wins vs 17 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_s4_packed_f16 | 4292 | 0.8657 | 0.8635 | 0.6767 | 0.0485 | 0.9171 | 0.504 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 4292 | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9479 | 41.722 | 0.0000 | 123.494 | 0.0000 | +0.0164 | 5 | 0 | +34.769 | A>>B |

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
| B3 | [0.30, 0.40) | 57 | 0.0487 | 0.1900 | 0.0276 | 0.0351 | 0.0703 | 0.982 |
| B4 | [0.40, 0.50) | 266 | 0.2510 | 0.2452 | 0.1658 | 0.0489 | 0.3169 | 0.969 |
| B5 | [0.50, 0.60) | 615 | 0.2466 | 0.2584 | 0.1655 | 0.0423 | 0.3049 | 0.966 |
| B6 | [0.60, 0.70) | 836 | 0.2220 | 0.2317 | 0.1484 | 0.0347 | 0.2580 | 0.973 |
| B7 | [0.70, 0.80) | 1092 | 0.3192 | 0.3244 | 0.2173 | 0.0513 | 0.3766 | 0.946 |
| B8 | [0.80, 0.90) | 1382 | 0.4958 | 0.5026 | 0.3354 | 0.0347 | 0.5846 | 0.865 |
| B9 | [0.90, 1.00] | 43 | 0.1056 | 0.1831 | 0.0698 | 0.0233 | 0.2725 | 0.983 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B2 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B3 | 57 | 0.157 | 0.8753 | -0.024 | 0.9807 | -0.0233 | 0 | 0 | +0.000 | tied |
| B4 | 266 | -0.231 | 0.8175 | 0.031 | 0.9749 | -0.0173 | 0 | 0 | +0.000 | tied |
| B5 | 615 | 0.843 | 0.3993 | 0.220 | 0.8257 | -0.0034 | 0 | 0 | -0.000 | tied |
| B6 | 836 | 8.547 | 0.0000 | 2.068 | 0.0386 | +0.0797 | 5 | 0 | +7.122 | A>>B |
| B7 | 1092 | 14.518 | 0.0000 | 6.058 | 0.0000 | +0.0726 | 5 | 0 | +12.098 | A>>B |
| B8 | 1382 | 0.343 | 0.7315 | -0.993 | 0.3205 | -0.0038 | 0 | 1 | +0.000 | promising |
| B9 | 43 | 3.844 | 0.0001 | 6.538 | 0.0000 | +0.0876 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_s4_packed_f16 | 10125 | 0.9285 | 0.9289 | 0.7629 | 0.0538 | 0.9580 | 0.370 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 10125 | 0.9677 | 0.9686 | 0.8432 | 0.0433 | 0.9804 | 0.249 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9577 | -90.670 | 0.0000 | -766.955 | 0.0000 | -0.0224 | 0 | 6 | -0.000 | B>>A |

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

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 705 | -9.682 | 0.0000 | -3.249 | 0.0012 | -0.2115 | 0 | 5 | -0.000 | B>>A |
| B1 | 910 | -10.257 | 0.0000 | -3.790 | 0.0002 | -0.1873 | 0 | 5 | -0.000 | B>>A |
| B2 | 1111 | -14.174 | 0.0000 | -4.449 | 0.0000 | -0.2684 | 0 | 5 | -0.000 | B>>A |
| B3 | 1291 | -10.534 | 0.0000 | -2.855 | 0.0043 | -0.1241 | 0 | 5 | -0.000 | B>>A |
| B4 | 1013 | -11.664 | 0.0000 | -4.154 | 0.0000 | -0.1416 | 0 | 5 | -0.000 | B>>A |
| B5 | 919 | -10.069 | 0.0000 | -2.552 | 0.0107 | -0.1711 | 0 | 5 | -0.000 | B>>A |
| B6 | 936 | -10.700 | 0.0000 | -2.878 | 0.0040 | -0.1840 | 0 | 5 | -0.000 | B>>A |
| B7 | 985 | -11.433 | 0.0000 | -4.067 | 0.0000 | -0.1373 | 0 | 5 | -0.000 | B>>A |
| B8 | 1699 | -16.632 | 0.0000 | -10.279 | 0.0000 | -0.0554 | 0 | 5 | -0.000 | B>>A |
| B9 | 486 | -7.051 | 0.0000 | -4.360 | 0.0000 | -0.0166 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_s4_packed_f16 | 3000 | 0.8890 | 0.8990 | 0.7108 | 0.0457 | 0.9182 | 0.438 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 3000 | 0.9729 | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9222 | -54.113 | 0.0000 | -311.042 | 0.0000 | -0.0650 | 0 | 6 | -0.000 | B>>A |

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

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 ⚠ | 29 | 1.608 | 0.1077 | -0.225 | 0.8221 | +0.0501 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -2.668 | 0.0076 | -0.218 | 0.8278 | -0.1337 | 0 | 0 | -0.000 | tied |
| B2 | 185 | -10.678 | 0.0000 | -6.063 | 0.0000 | -0.3764 | 0 | 5 | -0.000 | B>>A |
| B3 | 493 | -14.961 | 0.0000 | -13.014 | 0.0000 | -0.2598 | 0 | 5 | -0.000 | B>>A |
| B4 | 677 | -19.650 | 0.0000 | -20.295 | 0.0000 | -0.2133 | 0 | 5 | -0.000 | B>>A |
| B5 | 705 | -18.525 | 0.0000 | -14.323 | 0.0000 | -0.2594 | 0 | 5 | -0.000 | B>>A |
| B6 | 809 | -24.197 | 0.0000 | -10.758 | 0.0000 | -0.4418 | 0 | 5 | -0.000 | B>>A |
| B7 | 67 | 0.597 | 0.5507 | -0.074 | 0.9406 | +0.1277 | 0 | 0 | -0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_s4_packed_f16 | 1008 | 0.7901 | 0.8302 | 0.5791 | 0.0456 | 0.8343 | 0.558 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 1008 | 0.8927 | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9277 | -44.031 | 0.0000 | -139.868 | 0.0000 | -0.0834 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_s4_packed_f16 | 600 | 0.8061 | 0.8154 | 0.6392 | 0.0567 | 0.8776 | 0.579 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 600 | 0.7845 | 0.7953 | 0.6155 | 0.0433 | 0.8630 | 0.606 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9621 | 13.935 | 0.0000 | 29.097 | 0.0000 | +0.0146 | 5 | 1 | +11.612 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 89.47s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
