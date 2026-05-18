# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/zen/zensim-eval/v24_persample_konjnd_finetune_v2_2026-05-18/persample_konjnd_gentle_seed4_packed.bin` (label: `persample_konjnd_gentle_seed4_packed`)
- **B**: `/home/lilith/work/zen/zensim--hybrid-runtime/zensim/weights/v_compression_persample_2026-05-18.bin` (label: `v_compression_persample_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8451 | 0.8641 | 0.540 | 0.508 | 0.9025 | 0.9157 | -398.890 | -1253.185 | -0.000 | B>>A |
| KADIK10k | 10125 | 0.9321 | 0.9316 | 0.361 | 0.362 | 0.9603 | 0.9602 | 250.153 | 2066.353 | +208.460 | A>>B |
| TID2013 | 3000 | 0.8896 | 0.8893 | 0.431 | 0.432 | 0.9174 | 0.9173 | 102.892 | 1366.580 | +51.446 | promising |
| KonJND-1k (full) | 1008 | 0.8544 | 0.8080 | 0.425 | 0.502 | 0.8913 | 0.8505 | 146.893 | 485.097 | +122.411 | A>>B |
| AIC-3 CTC | 600 | 0.8131 | 0.8183 | 0.573 | 0.565 | 0.8812 | 0.8856 | -73.679 | -181.450 | -0.000 | B>>A |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 2 cells
- **BDecisivelyBeatsA**: 6 cells
- **PromisingNotDecisive**: 4 cells
- **Tied**: 17 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (2 A wins vs 6 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: persample_konjnd_gentle_seed4_packed | 4292 | 0.8451 | 0.8416 | 0.6519 | 0.0478 | 0.9025 | 0.540 |
| B: v_compression_persample_2026-05-18 | 4292 | 0.8641 | 0.8614 | 0.6742 | 0.0489 | 0.9157 | 0.508 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9969 | -398.890 | 0.0000 | -1253.185 | 0.0000 | -0.0132 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0762 | 0.2476 | 0.0564 | 0.0526 | 0.0280 | 0.969 |
| B4 | [0.40, 0.50) | 266 | 0.2542 | 0.2720 | 0.1690 | 0.0489 | 0.3111 | 0.962 |
| B5 | [0.50, 0.60) | 615 | 0.2332 | 0.2445 | 0.1587 | 0.0439 | 0.2788 | 0.970 |
| B6 | [0.60, 0.70) | 836 | 0.2362 | 0.2420 | 0.1570 | 0.0383 | 0.2742 | 0.970 |
| B7 | [0.70, 0.80) | 1092 | 0.3531 | 0.3610 | 0.2416 | 0.0430 | 0.4140 | 0.933 |
| B8 | [0.80, 0.90) | 1382 | 0.4859 | 0.4888 | 0.3310 | 0.0456 | 0.5670 | 0.872 |
| B9 | [0.90, 1.00] | 43 | 0.1780 | 0.4449 | 0.1185 | 0.0465 | 0.3404 | 0.896 |

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
| B3 | 57 | -20.004 | 0.0000 | -39.519 | 0.0000 | -0.0093 | 0 | 0 | -0.000 | tied |
| B4 | 266 | -54.302 | 0.0000 | 7.398 | 0.0000 | -0.0215 | 0 | 3 | +0.000 | promising |
| B5 | 615 | -67.903 | 0.0000 | -16.713 | 0.0000 | -0.0382 | 0 | 5 | -0.000 | B>>A |
| B6 | 836 | -80.413 | 0.0000 | -20.020 | 0.0000 | -0.0453 | 0 | 5 | -0.000 | B>>A |
| B7 | 1092 | -96.903 | 0.0000 | -40.726 | 0.0000 | -0.0279 | 0 | 5 | -0.000 | B>>A |
| B8 | 1382 | -137.119 | 0.0000 | -98.338 | 0.0000 | -0.0131 | 0 | 5 | -0.000 | B>>A |
| B9 | 43 | 28.340 | 0.0000 | 3.714 | 0.0002 | +0.0455 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: persample_konjnd_gentle_seed4_packed | 10125 | 0.9321 | 0.9327 | 0.7691 | 0.0489 | 0.9603 | 0.361 |
| B: v_compression_persample_2026-05-18 | 10125 | 0.9316 | 0.9321 | 0.7684 | 0.0462 | 0.9602 | 0.362 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9998 | 250.153 | 0.0000 | 2066.353 | 0.0000 | +0.0001 | 5 | 1 | +208.460 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### KADIK10k 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2417 | 0.2589 | 0.1697 | 0.0553 | 0.3083 | 0.966 |
| B1 | [0.10, 0.20) | 910 | 0.2588 | 0.2604 | 0.1795 | 0.0429 | 0.3203 | 0.966 |
| B2 | [0.20, 0.30) | 1111 | 0.1858 | 0.1963 | 0.1280 | 0.0324 | 0.2156 | 0.981 |
| B3 | [0.30, 0.40) | 1291 | 0.2207 | 0.2312 | 0.1542 | 0.0442 | 0.2618 | 0.973 |
| B4 | [0.40, 0.50) | 1013 | 0.2507 | 0.2515 | 0.1772 | 0.0385 | 0.2979 | 0.968 |
| B5 | [0.50, 0.60) | 919 | 0.2070 | 0.2368 | 0.1442 | 0.0424 | 0.2554 | 0.972 |
| B6 | [0.60, 0.70) | 936 | 0.2312 | 0.2474 | 0.1602 | 0.0427 | 0.2834 | 0.969 |
| B7 | [0.70, 0.80) | 985 | 0.2724 | 0.2758 | 0.1902 | 0.0386 | 0.3183 | 0.961 |
| B8 | [0.80, 0.90) | 1699 | 0.4522 | 0.4519 | 0.3175 | 0.0394 | 0.5354 | 0.892 |
| B9 | [0.90, 1.00] | 486 | 0.1723 | 0.1823 | 0.1178 | 0.0412 | 0.2053 | 0.983 |

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
| B0 | 705 | -87.422 | 0.0000 | -21.839 | 0.0000 | -0.0056 | 0 | 3 | -0.000 | promising |
| B1 | 910 | -24.248 | 0.0000 | -9.877 | 0.0000 | -0.0016 | 0 | 0 | -0.000 | tied |
| B2 | 1111 | -157.663 | 0.0000 | -29.271 | 0.0000 | -0.0074 | 0 | 3 | -0.000 | promising |
| B3 | 1291 | 14.721 | 0.0000 | 2.599 | 0.0093 | +0.0016 | 0 | 0 | +0.000 | tied |
| B4 | 1013 | -16.501 | 0.0000 | -4.144 | 0.0000 | -0.0021 | 0 | 0 | -0.000 | tied |
| B5 | 919 | 67.065 | 0.0000 | 21.214 | 0.0000 | +0.0041 | 0 | 0 | +0.000 | tied |
| B6 | 936 | 56.242 | 0.0000 | 17.602 | 0.0000 | +0.0032 | 0 | 0 | +0.000 | tied |
| B7 | 985 | 8.642 | 0.0000 | 3.426 | 0.0006 | +0.0007 | 0 | 0 | +0.000 | tied |
| B8 | 1699 | 22.501 | 0.0000 | 13.938 | 0.0000 | +0.0011 | 0 | 0 | +0.000 | tied |
| B9 | 486 | 44.271 | 0.0000 | 3.612 | 0.0003 | +0.0016 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: persample_konjnd_gentle_seed4_packed | 3000 | 0.8896 | 0.9026 | 0.7136 | 0.0437 | 0.9174 | 0.431 |
| B: v_compression_persample_2026-05-18 | 3000 | 0.8893 | 0.9018 | 0.7130 | 0.0457 | 0.9173 | 0.432 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9998 | 102.892 | 0.0000 | 1366.580 | 0.0000 | +0.0001 | 3 | 0 | +51.446 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.2054 | 0.4356 | 0.1554 | 0.0345 | 0.2183 | 0.900 |
| B1 | [0.10, 0.20) | 34 | 0.4840 | 0.5087 | 0.3446 | 0.0294 | 0.6270 | 0.861 |
| B2 | [0.20, 0.30) | 185 | 0.3243 | 0.3349 | 0.2213 | 0.0324 | 0.4041 | 0.942 |
| B3 | [0.30, 0.40) | 493 | 0.4704 | 0.4806 | 0.3256 | 0.0345 | 0.5689 | 0.877 |
| B4 | [0.40, 0.50) | 677 | 0.5359 | 0.5421 | 0.3746 | 0.0458 | 0.6286 | 0.840 |
| B5 | [0.50, 0.60) | 705 | 0.4696 | 0.4929 | 0.3255 | 0.0539 | 0.5544 | 0.870 |
| B6 | [0.60, 0.70) | 809 | 0.1854 | 0.2184 | 0.1262 | 0.0408 | 0.2312 | 0.976 |
| B7 | [0.70, 0.80) | 67 | 0.3473 | 0.4367 | 0.2282 | 0.0746 | 0.4015 | 0.900 |
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
| B0 ⚠ | 29 | 50.993 | 0.0000 | 96.986 | 0.0000 | +0.0167 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | 14.618 | 0.0000 | 48.418 | 0.0000 | +0.0069 | 0 | 0 | +0.000 | tied |
| B2 | 185 | -10.905 | 0.0000 | -13.712 | 0.0000 | -0.0028 | 0 | 0 | -0.000 | tied |
| B3 | 493 | 22.984 | 0.0000 | -17.555 | 0.0000 | +0.0017 | 0 | 0 | -0.000 | tied |
| B4 | 677 | 62.156 | 0.0000 | 54.796 | 0.0000 | +0.0017 | 0 | 0 | +0.000 | tied |
| B5 | 705 | 21.592 | 0.0000 | 8.637 | 0.0000 | +0.0003 | 0 | 0 | +0.000 | tied |
| B6 | 809 | -57.535 | 0.0000 | -2.513 | 0.0120 | -0.0010 | 0 | 0 | -0.000 | tied |
| B7 | 67 | -0.762 | 0.4460 | -80.123 | 0.0000 | -0.0004 | 0 | 0 | -0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: persample_konjnd_gentle_seed4_packed | 1008 | 0.8544 | 0.9051 | 0.6497 | 0.0456 | 0.8913 | 0.425 |
| B: v_compression_persample_2026-05-18 | 1008 | 0.8080 | 0.8648 | 0.5935 | 0.0456 | 0.8505 | 0.502 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9900 | 146.893 | 0.0000 | 485.097 | 0.0000 | +0.0408 | 5 | 0 | +122.411 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: persample_konjnd_gentle_seed4_packed | 600 | 0.8131 | 0.8198 | 0.6467 | 0.0550 | 0.8812 | 0.573 |
| B: v_compression_persample_2026-05-18 | 600 | 0.8183 | 0.8248 | 0.6527 | 0.0550 | 0.8856 | 0.565 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9983 | -73.679 | 0.0000 | -181.450 | 0.0000 | -0.0044 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 107.16s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
