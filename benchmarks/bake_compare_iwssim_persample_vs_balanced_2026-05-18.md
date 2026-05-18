# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/zen/zensim-eval/exp_iwssim_persample_2026-05-18/iwssim_persample_s3_h128.bin` (label: `iwssim_persample_s3_h128`)
- **B**: `zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` (label: `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8406 | 0.8324 | 0.548 | 0.559 | 0.8936 | 0.9006 | 6.164 | 15.552 | -1.027 | promising |
| KADIK10k | 10125 | 0.9671 | 0.9677 | 0.250 | 0.249 | 0.9805 | 0.9804 | -6.685 | -63.300 | +1.114 | promising |
| TID2013 | 3000 | 0.9814 | 0.9729 | 0.196 | 0.236 | 0.9888 | 0.9832 | 53.881 | 1090.796 | +44.901 | A>>B |
| KonJND-1k (full) | 1008 | 0.8053 | 0.8927 | 0.529 | 0.376 | 0.8493 | 0.9178 | -38.435 | -127.454 | -0.000 | B>>A |
| AIC-3 CTC | 600 | 0.7929 | 0.7845 | 0.592 | 0.606 | 0.8662 | 0.8630 | 4.838 | 13.590 | +0.000 | tied |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 6 cells
- **BDecisivelyBeatsA**: 3 cells
- **PromisingNotDecisive**: 2 cells
- **Tied**: 18 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `A`** (6 A wins vs 3 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 4292 | 0.8406 | 0.8366 | 0.6542 | 0.0436 | 0.8936 | 0.548 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 4292 | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9128 | 6.164 | 0.0000 | 15.552 | 0.0000 | -0.0070 | 1 | 0 | -1.027 | promising |

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
| B3 | 57 | 0.305 | 0.7605 | -0.017 | 0.9867 | +0.0199 | 0 | 0 | -0.000 | tied |
| B4 | 266 | -1.631 | 0.1029 | -0.047 | 0.9625 | -0.0315 | 0 | 0 | -0.000 | tied |
| B5 | 615 | -1.652 | 0.0986 | 0.007 | 0.9947 | -0.0419 | 0 | 0 | +0.000 | tied |
| B6 | 836 | 2.224 | 0.0261 | 0.691 | 0.4893 | +0.0293 | 0 | 0 | +0.000 | tied |
| B7 | 1092 | 10.775 | 0.0000 | 4.564 | 0.0000 | +0.0900 | 6 | 0 | +10.775 | A>>B |
| B8 | 1382 | 4.589 | 0.0000 | 2.539 | 0.0111 | +0.0206 | 0 | 0 | +0.000 | tied |
| B9 | 43 | -0.774 | 0.4392 | 0.751 | 0.4529 | -0.0403 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 10125 | 0.9671 | 0.9682 | 0.8421 | 0.0397 | 0.9805 | 0.250 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 10125 | 0.9677 | 0.9686 | 0.8432 | 0.0433 | 0.9804 | 0.249 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9915 | -6.685 | 0.0000 | -63.300 | 0.0000 | +0.0001 | 1 | 0 | +1.114 | promising |

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
| B0 | 705 | 2.917 | 0.0035 | 1.495 | 0.1348 | +0.0383 | 0 | 0 | +0.000 | tied |
| B1 | 910 | 2.203 | 0.0276 | 1.012 | 0.3113 | +0.0042 | 0 | 0 | +0.000 | tied |
| B2 | 1111 | -0.095 | 0.9242 | -0.055 | 0.9563 | -0.0025 | 0 | 0 | -0.000 | tied |
| B3 | 1291 | 2.155 | 0.0312 | 0.897 | 0.3696 | +0.0139 | 0 | 0 | +0.000 | tied |
| B4 | 1013 | -8.943 | 0.0000 | -3.976 | 0.0001 | -0.0407 | 0 | 5 | -0.000 | B>>A |
| B5 | 919 | -6.920 | 0.0000 | -2.789 | 0.0053 | -0.0423 | 0 | 5 | -0.000 | B>>A |
| B6 | 936 | 2.873 | 0.0041 | 1.252 | 0.2105 | +0.0193 | 0 | 0 | +0.000 | tied |
| B7 | 985 | 2.125 | 0.0335 | 0.630 | 0.5290 | +0.0123 | 0 | 0 | +0.000 | tied |
| B8 | 1699 | 5.648 | 0.0000 | 4.273 | 0.0000 | +0.0125 | 0 | 0 | +0.000 | tied |
| B9 | 486 | 1.009 | 0.3130 | -3.441 | 0.0006 | -0.0045 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 3000 | 0.9814 | 0.9807 | 0.8821 | 0.0443 | 0.9888 | 0.196 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 3000 | 0.9729 | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9914 | 53.881 | 0.0000 | 1090.796 | 0.0000 | +0.0056 | 5 | 1 | +44.901 | A>>B |

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
| B0 ⚠ | 29 | 2.611 | 0.0090 | 0.596 | 0.5509 | +0.4682 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | 0.305 | 0.7606 | 4.529 | 0.0000 | -0.1037 | 0 | 0 | -0.000 | tied |
| B2 | 185 | 1.741 | 0.0818 | 3.714 | 0.0002 | +0.0329 | 0 | 0 | +0.000 | tied |
| B3 | 493 | 11.277 | 0.0000 | 22.407 | 0.0000 | +0.0571 | 5 | 0 | +9.398 | A>>B |
| B4 | 677 | 9.592 | 0.0000 | 19.065 | 0.0000 | +0.0202 | 4 | 0 | +6.395 | A>>B |
| B5 | 705 | 16.088 | 0.0000 | 24.404 | 0.0000 | +0.0385 | 5 | 0 | +13.406 | A>>B |
| B6 | 809 | 33.098 | 0.0000 | 34.739 | 0.0000 | +0.1180 | 5 | 0 | +27.581 | A>>B |
| B7 | 67 | 4.969 | 0.0000 | 0.229 | 0.8190 | +0.2143 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 1008 | 0.8053 | 0.8483 | 0.5946 | 0.0417 | 0.8493 | 0.529 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 1008 | 0.8927 | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9291 | -38.435 | 0.0000 | -127.454 | 0.0000 | -0.0685 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: iwssim_persample_s3_h128 | 600 | 0.7929 | 0.8059 | 0.6282 | 0.0517 | 0.8662 | 0.592 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 600 | 0.7845 | 0.7953 | 0.6155 | 0.0433 | 0.8630 | 0.606 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9578 | 4.838 | 0.0000 | 13.590 | 0.0000 | +0.0032 | 0 | 0 | +0.000 | tied |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 92.11s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
