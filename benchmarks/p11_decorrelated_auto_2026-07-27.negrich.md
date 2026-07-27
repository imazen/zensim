# P11 decorrelated-auto (plain-mass arm) — 2026-07-27

Candidates 672 (924 − fold slots − BANDING − E1/E2); |r|>0.985 pairs 229; survivors **518**. Held-out univariate scorer: ext_cid22_train201 (train-legal).

BVLS arms on safesyn (≥0, standardized): 504 baseline 23/504 active; P11 survivors 27/518 active. Holdouts applied-only (never fit; CID22 MOS ban).

| corpus | 504-BVLS SROCC | P11-BVLS SROCC | Δ |
|---|---|---|---|
| ext_cid22val | 0.8753 | 0.8503 | -0.0250 |
| ext_aic3 | 0.7606 | 0.7730 | +0.0124 |
| ext_aic4 | -0.9137 | -0.9138 | -0.0001 |
| ext_csiq | 0.8306 | 0.8063 | -0.0243 |
| ext_live | 0.9244 | 0.9105 | -0.0139 |
| ext_konjnd_jpeg_val | -0.4741 | -0.4806 | -0.0065 |
| ext_sdr25 | -0.9734 | -0.9773 | -0.0039 |

Mean holdout Δ: -0.0088. MLP arm + the pathology-enriched diff (needs W2 kadis-924): PENDING.
Survivor mask: `p11_survivor_mask_2026-07-27.negrich.tsv`.
