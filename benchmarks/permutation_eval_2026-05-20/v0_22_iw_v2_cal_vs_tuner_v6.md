# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/home/lilith/work/zen/zensim/zensim/weights/v0_22_iw_v2_calibrated_2026-05-16.bin` (label: `v0_22_iw_v2_calibrated_2026-05-16`)
- **B**: `/home/lilith/work/zen/zensim--v6-reship/zensim/weights/v_tuner_v6_2026-05-19.bin` (label: `v_tuner_v6_2026-05-19`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `500`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8163 | 0.8216 | 0.569 | 0.680 | 0.8754 | 0.8821 | -1.120 | 34.507 | +0.560 | promising |
| KonJND-1k (full) | 1008 | 0.0303 | 0.3734 | 0.994 | 0.960 | 0.0883 | 0.5297 | -18.880 | -1.964 | -0.000 | B>>A |
| TID2013 | 3000 | 0.9617 | 0.4447 | 0.272 | 0.809 | 0.9766 | 0.5451 | 37.683 | 61.603 | +31.403 | A>>B |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 6 cells
- **BDecisivelyBeatsA**: 1 cells
- **PromisingNotDecisive**: 4 cells
- **Tied**: 6 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `A`** (6 A wins vs 1 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v0_22_iw_v2_calibrated_2026-05-16 | 4292 | 0.8163 | 0.8226 | 0.6317 | 0.0473 | 0.8754 | 0.569 |
| B: v_tuner_v6_2026-05-19 | 4292 | 0.8216 | 0.7332 | 0.6249 | 0.0531 | 0.8821 | 0.680 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.6905 | -1.120 | 0.2626 | 34.507 | 0.0000 | -0.0066 | 3 | 0 | +0.560 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1180 | 0.1640 | 0.0677 | 0.0526 | 0.0399 | 0.986 |
| B4 | [0.40, 0.50) | 266 | 0.2613 | 0.2989 | 0.1743 | 0.0489 | 0.3169 | 0.954 |
| B5 | [0.50, 0.60) | 615 | 0.2566 | 0.2738 | 0.1741 | 0.0520 | 0.3058 | 0.962 |
| B6 | [0.60, 0.70) | 836 | 0.2351 | 0.2549 | 0.1562 | 0.0431 | 0.2807 | 0.967 |
| B7 | [0.70, 0.80) | 1092 | 0.3415 | 0.3507 | 0.2364 | 0.0449 | 0.3824 | 0.936 |
| B8 | [0.80, 0.90) | 1382 | 0.4522 | 0.4541 | 0.3091 | 0.0449 | 0.5303 | 0.891 |
| B9 | [0.90, 1.00] | 43 | 0.0590 | 0.4033 | 0.0188 | 0.0465 | 0.1713 | 0.915 |

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
| B3 | 57 | 0.587 | 0.5571 | 0.030 | 0.9762 | -0.0618 | 0 | 0 | -0.000 | tied |
| B4 | 266 | 0.956 | 0.3391 | 0.350 | 0.7263 | +0.0474 | 0 | 0 | +0.000 | tied |
| B5 | 615 | -0.457 | 0.6479 | 0.079 | 0.9367 | -0.0288 | 0 | 0 | +0.000 | tied |
| B6 | 836 | -2.363 | 0.0181 | -0.694 | 0.4879 | -0.0783 | 0 | 0 | -0.000 | tied |
| B7 | 1092 | 0.549 | 0.5827 | 0.415 | 0.6781 | -0.0229 | 0 | 0 | -0.000 | tied |
| B8 | 1382 | 2.778 | 0.0055 | 1.399 | 0.1619 | +0.0346 | 0 | 1 | +0.000 | promising |
| B9 | 43 | 0.244 | 0.8069 | 0.347 | 0.7283 | +0.0873 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v0_22_iw_v2_calibrated_2026-05-16 | 1008 | 0.0303 | 0.1059 | 0.0229 | 0.0387 | 0.0883 | 0.994 |
| B: v_tuner_v6_2026-05-19 | 1008 | 0.3734 | 0.2813 | 0.2584 | 0.0347 | 0.5297 | 0.960 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.4362 | -18.880 | 0.0000 | -1.964 | 0.0495 | -0.4414 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v0_22_iw_v2_calibrated_2026-05-16 | 3000 | 0.9617 | 0.9623 | 0.8280 | 0.0487 | 0.9766 | 0.272 |
| B: v_tuner_v6_2026-05-19 | 3000 | 0.4447 | 0.5878 | 0.3055 | 0.0430 | 0.5451 | 0.809 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.4685 | 37.683 | 0.0000 | 61.603 | 0.0000 | +0.4315 | 5 | 0 | +31.403 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.2035 | 0.3588 | 0.1603 | 0.0690 | 0.2138 | 0.933 |
| B1 | [0.10, 0.20) | 34 | 0.3807 | 0.3956 | 0.2375 | 0.0000 | 0.4635 | 0.918 |
| B2 | [0.20, 0.30) | 185 | 0.5630 | 0.5804 | 0.3902 | 0.0595 | 0.6570 | 0.814 |
| B3 | [0.30, 0.40) | 493 | 0.6408 | 0.6457 | 0.4623 | 0.0507 | 0.7113 | 0.764 |
| B4 | [0.40, 0.50) | 677 | 0.6732 | 0.6741 | 0.4859 | 0.0502 | 0.7479 | 0.739 |
| B5 | [0.50, 0.60) | 705 | 0.6229 | 0.6225 | 0.4415 | 0.0369 | 0.7122 | 0.783 |
| B6 | [0.60, 0.70) | 809 | 0.5505 | 0.5501 | 0.3813 | 0.0396 | 0.6543 | 0.835 |
| B7 | [0.70, 0.80) | 67 | 0.0348 | 0.1235 | 0.0195 | 0.0299 | 0.0172 | 0.992 |
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
| B0 ⚠ | 29 | 0.516 | 0.6057 | -0.103 | 0.9181 | +0.2068 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -2.881 | 0.0040 | -3.397 | 0.0007 | -0.2610 | 0 | 2 | -0.000 | promising |
| B2 | 185 | 7.964 | 0.0000 | 3.608 | 0.0003 | +0.3528 | 5 | 0 | +6.637 | A>>B |
| B3 | 493 | 12.352 | 0.0000 | 6.710 | 0.0000 | +0.4093 | 5 | 0 | +10.293 | A>>B |
| B4 | 677 | 15.157 | 0.0000 | 7.937 | 0.0000 | +0.4844 | 5 | 0 | +12.631 | A>>B |
| B5 | 705 | 14.503 | 0.0000 | 6.135 | 0.0000 | +0.5702 | 5 | 0 | +12.086 | A>>B |
| B6 | 809 | 16.361 | 0.0000 | 5.264 | 0.0000 | +0.6281 | 5 | 0 | +13.635 | A>>B |
| B7 | 67 | -2.484 | 0.0130 | -0.773 | 0.4397 | -0.4605 | 0 | 1 | -0.000 | promising |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

---
Wall time: 34.16s (8300 pair rows scored × 2 bakes across 3 corpora; 500 bootstrap resamples × bands).
