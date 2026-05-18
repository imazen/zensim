# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/exmix3_cv30_iw40_sm30_s3_h128_packed.bin` (label: `exmix3_cv30_iw40_sm30_s3_h128_packed`)
- **B**: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128.bin` (label: `v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8642 | 0.8323 | 0.506 | 0.559 | 0.9191 | 0.9005 | 35.151 | 105.555 | +29.293 | A>>B |
| KADIK10k | 10125 | 0.9255 | 0.9677 | 0.378 | 0.249 | 0.9558 | 0.9804 | -88.704 | -724.145 | -0.000 | B>>A |
| TID2013 | 3000 | 0.8776 | 0.9729 | 0.461 | 0.236 | 0.9101 | 0.9833 | -54.551 | -294.357 | -0.000 | B>>A |
| KonJND-1k (full) | 1008 | 0.8424 | 0.8928 | 0.448 | 0.375 | 0.8775 | 0.9181 | -30.954 | -99.844 | -0.000 | B>>A |
| AIC-3 CTC | 600 | 0.8048 | 0.7831 | 0.580 | 0.608 | 0.8789 | 0.8619 | 25.590 | 53.866 | +21.325 | A>>B |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 3 cells
- **BDecisivelyBeatsA**: 17 cells
- **PromisingNotDecisive**: 1 cells
- **Tied**: 8 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (3 A wins vs 17 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: exmix3_cv30_iw40_sm30_s3_h128_packed | 4292 | 0.8642 | 0.8628 | 0.6719 | 0.0480 | 0.9191 | 0.506 |
| B: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128 | 4292 | 0.8323 | 0.8290 | 0.6339 | 0.0445 | 0.9005 | 0.559 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9406 | 35.151 | 0.0000 | 105.555 | 0.0000 | +0.0186 | 5 | 0 | +29.293 | A>>B |

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
| B3 | [0.30, 0.40) | 57 | 0.0635 | 0.1900 | 0.0363 | 0.0526 | 0.0885 | 0.982 |
| B4 | [0.40, 0.50) | 266 | 0.2520 | 0.2451 | 0.1666 | 0.0489 | 0.3181 | 0.970 |
| B5 | [0.50, 0.60) | 615 | 0.2508 | 0.2618 | 0.1684 | 0.0390 | 0.3097 | 0.965 |
| B6 | [0.60, 0.70) | 836 | 0.2238 | 0.2333 | 0.1495 | 0.0323 | 0.2602 | 0.972 |
| B7 | [0.70, 0.80) | 1092 | 0.3189 | 0.3242 | 0.2170 | 0.0513 | 0.3762 | 0.946 |
| B8 | [0.80, 0.90) | 1382 | 0.4955 | 0.5019 | 0.3350 | 0.0347 | 0.5843 | 0.865 |
| B9 | [0.90, 1.00] | 43 | 0.1045 | 0.1831 | 0.0653 | 0.0233 | 0.2696 | 0.983 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B2 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B3 | 57 | -0.920 | 0.3576 | -0.193 | 0.8473 | +0.0045 | 0 | 0 | +0.000 | tied |
| B4 | 266 | 2.815 | 0.0049 | 0.697 | 0.4858 | +0.0308 | 0 | 0 | +0.000 | tied |
| B5 | 615 | 2.748 | 0.0060 | 0.878 | 0.3797 | +0.0355 | 0 | 0 | +0.000 | tied |
| B6 | 836 | 2.464 | 0.0138 | 0.413 | 0.6797 | +0.0194 | 0 | 0 | +0.000 | tied |
| B7 | 1092 | 11.873 | 0.0000 | 4.756 | 0.0000 | +0.0713 | 5 | 0 | +9.894 | A>>B |
| B8 | 1382 | -5.237 | 0.0000 | -4.239 | 0.0000 | -0.0217 | 0 | 1 | -0.000 | promising |
| B9 | 43 | 0.311 | 0.7560 | 2.817 | 0.0048 | -0.0089 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: exmix3_cv30_iw40_sm30_s3_h128_packed | 10125 | 0.9255 | 0.9260 | 0.7578 | 0.0527 | 0.9558 | 0.378 |
| B: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128 | 10125 | 0.9677 | 0.9686 | 0.8432 | 0.0424 | 0.9804 | 0.249 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9536 | -88.704 | 0.0000 | -724.145 | 0.0000 | -0.0246 | 0 | 6 | -0.000 | B>>A |

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
| B0 | [0.00, 0.10) | 705 | 0.4139 | 0.4237 | 0.2932 | 0.0255 | 0.5006 | 0.906 |
| B1 | [0.10, 0.20) | 910 | 0.4144 | 0.4186 | 0.2918 | 0.0407 | 0.4850 | 0.908 |
| B2 | [0.20, 0.30) | 1111 | 0.3964 | 0.4018 | 0.2784 | 0.0378 | 0.4714 | 0.916 |
| B3 | [0.30, 0.40) | 1291 | 0.3367 | 0.3355 | 0.2370 | 0.0434 | 0.3980 | 0.942 |
| B4 | [0.40, 0.50) | 1013 | 0.3746 | 0.3809 | 0.2662 | 0.0444 | 0.4402 | 0.925 |
| B5 | [0.50, 0.60) | 919 | 0.3460 | 0.3534 | 0.2448 | 0.0479 | 0.4194 | 0.935 |
| B6 | [0.60, 0.70) | 936 | 0.3652 | 0.3641 | 0.2550 | 0.0449 | 0.4431 | 0.931 |
| B7 | [0.70, 0.80) | 985 | 0.3613 | 0.3676 | 0.2554 | 0.0416 | 0.4434 | 0.930 |
| B8 | [0.80, 0.90) | 1699 | 0.5027 | 0.5035 | 0.3560 | 0.0388 | 0.5881 | 0.864 |
| B9 | [0.90, 1.00] | 486 | 0.1821 | 0.2235 | 0.1251 | 0.0370 | 0.2163 | 0.975 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 705 | -9.478 | 0.0000 | -3.270 | 0.0011 | -0.1909 | 0 | 5 | -0.000 | B>>A |
| B1 | 910 | -9.785 | 0.0000 | -3.600 | 0.0003 | -0.1770 | 0 | 5 | -0.000 | B>>A |
| B2 | 1111 | -14.564 | 0.0000 | -4.309 | 0.0000 | -0.3003 | 0 | 5 | -0.000 | B>>A |
| B3 | 1291 | -10.423 | 0.0000 | -2.976 | 0.0029 | -0.1362 | 0 | 5 | -0.000 | B>>A |
| B4 | 1013 | -9.814 | 0.0000 | -3.613 | 0.0003 | -0.1342 | 0 | 5 | -0.000 | B>>A |
| B5 | 919 | -10.408 | 0.0000 | -2.743 | 0.0061 | -0.1903 | 0 | 5 | -0.000 | B>>A |
| B6 | 936 | -11.587 | 0.0000 | -3.156 | 0.0016 | -0.2043 | 0 | 5 | -0.000 | B>>A |
| B7 | 985 | -11.910 | 0.0000 | -4.116 | 0.0000 | -0.1529 | 0 | 5 | -0.000 | B>>A |
| B8 | 1699 | -18.065 | 0.0000 | -10.747 | 0.0000 | -0.0754 | 0 | 5 | -0.000 | B>>A |
| B9 | 486 | -7.821 | 0.0000 | -2.486 | 0.0129 | -0.0348 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: exmix3_cv30_iw40_sm30_s3_h128_packed | 3000 | 0.8776 | 0.8874 | 0.6924 | 0.0450 | 0.9101 | 0.461 |
| B: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128 | 3000 | 0.9729 | 0.9718 | 0.8573 | 0.0380 | 0.9833 | 0.236 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9134 | -54.551 | 0.0000 | -294.357 | 0.0000 | -0.0732 | 0 | 6 | -0.000 | B>>A |

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
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0559 | 0.3495 | 0.0567 | 0.0345 | 0.0759 | 0.937 |
| B1 | [0.10, 0.20) | 34 | 0.5396 | 0.5997 | 0.3482 | 0.0294 | 0.7026 | 0.800 |
| B2 | [0.20, 0.30) | 185 | 0.6423 | 0.6464 | 0.4552 | 0.0432 | 0.7132 | 0.763 |
| B3 | [0.30, 0.40) | 493 | 0.7360 | 0.7367 | 0.5403 | 0.0467 | 0.8178 | 0.676 |
| B4 | [0.40, 0.50) | 677 | 0.7622 | 0.7623 | 0.5597 | 0.0458 | 0.8362 | 0.647 |
| B5 | [0.50, 0.60) | 705 | 0.7075 | 0.7070 | 0.5096 | 0.0496 | 0.7900 | 0.707 |
| B6 | [0.60, 0.70) | 809 | 0.5871 | 0.5864 | 0.4106 | 0.0334 | 0.6836 | 0.810 |
| B7 | [0.70, 0.80) | 67 | 0.2857 | 0.4937 | 0.1928 | 0.0746 | 0.2808 | 0.870 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 ⚠ | 29 | 1.325 | 0.1853 | -0.003 | 0.9974 | +0.1707 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -2.253 | 0.0243 | -1.128 | 0.2595 | -0.0760 | 0 | 0 | -0.000 | tied |
| B2 | 185 | -11.448 | 0.0000 | -5.942 | 0.0000 | -0.3775 | 0 | 5 | -0.000 | B>>A |
| B3 | 493 | -17.509 | 0.0000 | -13.505 | 0.0000 | -0.3473 | 0 | 5 | -0.000 | B>>A |
| B4 | 677 | -20.710 | 0.0000 | -19.787 | 0.0000 | -0.2650 | 0 | 5 | -0.000 | B>>A |
| B5 | 705 | -18.850 | 0.0000 | -14.867 | 0.0000 | -0.2625 | 0 | 5 | -0.000 | B>>A |
| B6 | 809 | -24.994 | 0.0000 | -10.448 | 0.0000 | -0.5006 | 0 | 5 | -0.000 | B>>A |
| B7 | 67 | 0.523 | 0.6011 | -0.076 | 0.9392 | +0.1142 | 0 | 0 | -0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: exmix3_cv30_iw40_sm30_s3_h128_packed | 1008 | 0.8424 | 0.8941 | 0.6373 | 0.0456 | 0.8775 | 0.448 |
| B: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128 | 1008 | 0.8928 | 0.9270 | 0.7074 | 0.0446 | 0.9181 | 0.375 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9487 | -30.954 | 0.0000 | -99.844 | 0.0000 | -0.0406 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: exmix3_cv30_iw40_sm30_s3_h128_packed | 600 | 0.8048 | 0.8147 | 0.6385 | 0.0500 | 0.8789 | 0.580 |
| B: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128 | 600 | 0.7831 | 0.7942 | 0.6139 | 0.0433 | 0.8619 | 0.608 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9793 | 25.590 | 0.0000 | 53.866 | 0.0000 | +0.0170 | 5 | 0 | +21.325 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 198.79s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
