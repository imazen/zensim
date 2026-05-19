# EXP-CROSS-CODEC-V2 — eval table

**Date:** 2026-05-19
**Substrate:** tighter equivalence parquet (gap ≤ 0.3, 30 levels, avif 2× row weight) at `/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet` (68,788 pairs).
**Recipe:** Tuner-v2 + cross-codec-eq parquet. Per-sample-α head, 372→128→128 identity.

## Mohammadi panel (SROCC aggregate, full panel in per-bake verdict.md)

| Bake | CID22 | AIC-3 | KADID | TID | KonJND | T=63 butter_max | T=63 butter_p3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ship (cc4_s1_w1.0)** | 0.880 | 0.806 | 0.800 | 0.822 | 0.327 | **5.52** | 2.16 |
| Tuner baseline | 0.879 | 0.813 | 0.770 | 0.748 | 0.235 | 8.07 | 3.03 |
| cc4v2_s1_w1_0 | 0.7826 | 0.7948 | 0.5775 | 0.6479 | 0.3848 | 5.016 | 1.971 |
| cc4v2_s1_w1_5 | 0.7479 | 0.7761 | 0.3131 | 0.4305 | 0.2359 | 5.511 | 2.267 |
| cc4v2_s1_w2_0 | 0.8237 | 0.8067 | 0.8069 | 0.8377 | 0.3511 | 1.152 | 0.536 |
| cc4v2_s2_w1_5 | 0.8263 | 0.7792 | 0.8044 | 0.8008 | 0.2111 | 10.120 | 3.817 |
| cc4v2_s3_w1_5 | 0.8328 | 0.7930 | 0.5894 | 0.5775 | 0.1171 | 3.187 | 1.342 |

## Gate evaluation

Strict gate: T=63 butter_max < 2.5
Relaxed gate: T=63 butter_max < 3.0

Secondary gates:
- CID22 SROCC ≥ 0.86 (within 0.02 of W=1.0 ship)
- KADID SROCC ≥ 0.70 (within 0.10 of W=1.0 ship)
- TID SROCC ≥ 0.72 (within 0.10 of W=1.0 ship)
- AIC-3 SROCC ≥ 0.78 (within 0.02 of W=1.0 ship)
