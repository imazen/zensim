# V6-RESHIP median selection (2026-05-19, task #172)

5-seed CI at K=32 lr=5.66e-3, V6 recipe otherwise verbatim.

## Per-seed table

| bake | CID22 | KADID | TID | KonJND | AIC-3 | mono | tied | medRange | butter_p3 | PJND cc | all-band cc_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cc4v6r_s1 | 0.8506 | 0.6023 | 0.6764 | 0.2690 | 0.7806 | 0.9767 | 0.0000 | 76.36 | 1.756 | 1.09 | 2.18 |
| cc4v6r_s2 | 0.8521 | 0.6309 | 0.6918 | 0.1785 | 0.7810 | 0.9400 | 0.0000 | 75.59 | 1.739 | 0.94 | 1.58 |
| cc4v6r_s3 | 0.8503 | 0.6132 | 0.7068 | 0.3477 | 0.7763 | 0.9733 | 0.0000 | 76.18 | 1.706 | 1.12 | 2.19 |

## Median selection

Sorted by CID22 SROCC ascending. With n=3 seeds the median index is 1.

**Median bake: `cc4v6r_s1` (CID22 0.8506)**


## Gate verdicts for median bake `cc4v6r_s1`

| gate | observed | gate value | verdict |
|---|---:|---|:-:|
| mono ≥ 0.9378 | 0.9767 | ≥ 0.9378 | PASS |
| tied ≤ 5% | 0.0000 | ≤ 0.05 | PASS |
| medRange ≥ 50 | 76.36 | ≥ 50.0 | PASS |
| T63 butter_p3 < 2.5 | 1.756 | < 2.5 | PASS |
| PJND cc_std_median ≤ 5 | 1.09 | ≤ 5.0 | PASS |
| all-band cc_std max ≤ 5 | 2.18 | ≤ 5.0 | PASS |
| CID22 SROCC ≥ 0.84 (K=1 honest median) | 0.8506 | ≥ 0.84 | PASS |
| KonJND SROCC ≥ 0.196 (V6 ship floor) | 0.2690 | ≥ 0.196 | PASS |

**Overall ship decision: SHIP**

Median bake `cc4v6r_s1` passes all 8 gates. Recommend swapping the bake bytes in `zensim/weights/v_tuner_v6_2026-05-19.bin` with this seed.
