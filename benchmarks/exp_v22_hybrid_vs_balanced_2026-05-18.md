# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/zen/zensim-eval/exp_v22_hybrid_2026-05-18/v22_hybrid_s3_h128_packed.bin` (label: `v22_hybrid_s3_h128_packed`)
- **B**: `/home/lilith/work/zen/zensim--exp-v22-hybrid/zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` (label: `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8657 | 0.8324 | 0.503 | 0.559 | 0.9173 | 0.9006 | 41.973 | 127.071 | +34.978 | A>>B |
| KADIK10k | 10125 | 0.9315 | 0.9677 | 0.362 | 0.249 | 0.9596 | 0.9804 | -90.810 | -792.667 | -0.000 | B>>A |
| TID2013 | 3000 | 0.8906 | 0.9729 | 0.431 | 0.236 | 0.9181 | 0.9832 | -54.098 | -311.269 | -0.000 | B>>A |
| KonJND-1k (full) | 1008 | 0.7814 | 0.8927 | 0.568 | 0.376 | 0.8284 | 0.9178 | -46.356 | -141.309 | -0.000 | B>>A |
| AIC-3 CTC | 600 | 0.8034 | 0.7845 | 0.583 | 0.606 | 0.8758 | 0.8630 | 17.444 | 35.483 | +14.537 | A>>B |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 3 cells
- **BDecisivelyBeatsA**: 17 cells
- **PromisingNotDecisive**: 2 cells
- **Tied**: 7 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (3 A wins vs 17 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_hybrid_s3_h128_packed | 4292 | 0.8657 | 0.8642 | 0.6775 | 0.0461 | 0.9173 | 0.503 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 4292 | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9482 | 41.973 | 0.0000 | 127.071 | 0.0000 | +0.0167 | 5 | 0 | +34.978 | A>>B |

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
| B3 | 57 | 0.298 | 0.7661 | -0.023 | 0.9820 | +0.0306 | 0 | 0 | -0.000 | tied |
| B4 | 266 | 2.134 | 0.0328 | 0.630 | 0.5288 | +0.0298 | 0 | 0 | +0.000 | tied |
| B5 | 615 | 2.963 | 0.0030 | 0.853 | 0.3934 | +0.0304 | 0 | 0 | +0.000 | tied |
| B6 | 836 | 8.230 | 0.0000 | 1.917 | 0.0553 | +0.0721 | 5 | 0 | +6.858 | promising |
| B7 | 1092 | 13.072 | 0.0000 | 5.577 | 0.0000 | +0.0537 | 5 | 0 | +10.893 | A>>B |
| B8 | 1382 | 3.266 | 0.0011 | 1.361 | 0.1735 | +0.0032 | 0 | 1 | +0.000 | promising |
| B9 | 43 | 3.324 | 0.0009 | 5.698 | 0.0000 | +0.0889 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_hybrid_s3_h128_packed | 10125 | 0.9315 | 0.9320 | 0.7675 | 0.0547 | 0.9596 | 0.362 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 10125 | 0.9677 | 0.9686 | 0.8432 | 0.0433 | 0.9804 | 0.249 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9609 | -90.810 | 0.0000 | -792.667 | 0.0000 | -0.0208 | 0 | 6 | -0.000 | B>>A |

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
| B0 | 705 | -9.354 | 0.0000 | -3.415 | 0.0006 | -0.1826 | 0 | 5 | -0.000 | B>>A |
| B1 | 910 | -10.140 | 0.0000 | -3.840 | 0.0001 | -0.1773 | 0 | 5 | -0.000 | B>>A |
| B2 | 1111 | -14.732 | 0.0000 | -4.630 | 0.0000 | -0.2680 | 0 | 5 | -0.000 | B>>A |
| B3 | 1291 | -10.569 | 0.0000 | -2.953 | 0.0032 | -0.1237 | 0 | 5 | -0.000 | B>>A |
| B4 | 1013 | -10.636 | 0.0000 | -3.903 | 0.0001 | -0.1260 | 0 | 5 | -0.000 | B>>A |
| B5 | 919 | -11.067 | 0.0000 | -2.995 | 0.0027 | -0.1709 | 0 | 5 | -0.000 | B>>A |
| B6 | 936 | -10.862 | 0.0000 | -3.086 | 0.0020 | -0.1726 | 0 | 5 | -0.000 | B>>A |
| B7 | 985 | -11.972 | 0.0000 | -4.283 | 0.0000 | -0.1392 | 0 | 5 | -0.000 | B>>A |
| B8 | 1699 | -18.064 | 0.0000 | -11.095 | 0.0000 | -0.0643 | 0 | 5 | -0.000 | B>>A |
| B9 | 486 | -8.197 | 0.0000 | -4.978 | 0.0000 | -0.0236 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_hybrid_s3_h128_packed | 3000 | 0.8906 | 0.9021 | 0.7154 | 0.0457 | 0.9181 | 0.431 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 3000 | 0.9729 | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9235 | -54.098 | 0.0000 | -311.269 | 0.0000 | -0.0650 | 0 | 6 | -0.000 | B>>A |

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
| B0 ⚠ | 29 | 1.897 | 0.0578 | 0.584 | 0.5592 | +0.1276 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | -2.051 | 0.0403 | -0.905 | 0.3656 | -0.0956 | 0 | 0 | -0.000 | tied |
| B2 | 185 | -11.188 | 0.0000 | -6.629 | 0.0000 | -0.3218 | 0 | 5 | -0.000 | B>>A |
| B3 | 493 | -15.073 | 0.0000 | -13.342 | 0.0000 | -0.2398 | 0 | 5 | -0.000 | B>>A |
| B4 | 677 | -19.053 | 0.0000 | -20.284 | 0.0000 | -0.1952 | 0 | 5 | -0.000 | B>>A |
| B5 | 705 | -18.503 | 0.0000 | -14.563 | 0.0000 | -0.2461 | 0 | 5 | -0.000 | B>>A |
| B6 | 809 | -23.848 | 0.0000 | -10.572 | 0.0000 | -0.4458 | 0 | 5 | -0.000 | B>>A |
| B7 | 67 | 0.652 | 0.5147 | -0.076 | 0.9398 | +0.1332 | 0 | 0 | -0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_hybrid_s3_h128_packed | 1008 | 0.7814 | 0.8228 | 0.5686 | 0.0427 | 0.8284 | 0.568 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 1008 | 0.8927 | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9258 | -46.356 | 0.0000 | -141.309 | 0.0000 | -0.0893 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_hybrid_s3_h128_packed | 600 | 0.8034 | 0.8125 | 0.6377 | 0.0567 | 0.8758 | 0.583 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 600 | 0.7845 | 0.7953 | 0.6155 | 0.0433 | 0.8630 | 0.606 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9735 | 17.444 | 0.0000 | 35.483 | 0.0000 | +0.0128 | 5 | 0 | +14.537 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 174.90s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
