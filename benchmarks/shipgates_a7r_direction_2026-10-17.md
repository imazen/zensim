# Frozen A7r direction-agreement tier — 2026-10-17

All three candidates **FAIL**. Standard JPEG and ladder JXL fail for every candidate; diff-410 additionally fails AVIF-SVT on both grids. Every failure is a negative point difference D; every one-sided lower bound meets −0.05. Reachability and final targeting qualification remain separate, and the original strict A7r results remain on record.

Frozen spec SHA256: `b62a136965c4e1a236c12c8d158cd02733ca7a03fc40f0e932a484f91d4a6f9e`.

SSIMULACRA2 alone greedily selects up to four points from each q-sorted ladder, beginning at its lowest setting and requiring |Δmentor| ≥ 0.5 from the last selected point. Partial windows are retained; only ladders with no selected edge are excluded. There is no clamp exclusion. Here all 249 ladders contribute three edges: 747 total, no exclusions, no missing/nonfinite score cells. AVIF uses SVT only; rav1e is excluded. The standard SVT view replaces its legacy AVIF rows with the same 35 references’ SVT rows from the ladder instrument, preserving their RGB bindings; other standard rows are unchanged. The two grids remain separate. [Derivation and historical strict results](shipgates_encoder_monotonicity_2026-10-17.md).

An edge agrees iff Δscorer × Δmentor > 0; zero, missing or nonfinite scores fail. Family agreement is edge-weighted. D is candidate agreement minus CVVDP-FHD on identical edges. The paired bootstrap resamples whole ladders with replacement, carrying every scorer together: 10,000 draws, seed 20261017, canonical zenmetrics-stats `48665c8fb1bb3de5c99d792ae4ed4f3f1e854629`, serial Xoshiro256ss, Hyndman–Fan type 7. D’s lower bound is the 5th percentile; scorer intervals are descriptive two-sided 95% (2.5th/97.5th percentiles) from the same draws.

A family passes iff D ≥ 0 and lower5 ≥ −0.05, with at least 10 evaluable ladders. All four families on both grids must pass; this is an intersection-union test with no multiplicity adjustment. All eight families have 16–39 ladders, so none is insufficient.

| Candidate | Overall | Failed grid/family |
|---|---|---|
| diff-410 | FAIL | standard/avif-svt, standard/jpeg, ladder/avif-svt, ladder/jxl |
| diff+frag-820 | FAIL | standard/jpeg, ladder/jxl |
| W-C | FAIL | standard/jpeg, ladder/jxl |

## standard

| Candidate | Codec | Ladders / edges | Agreement | CVVDP-FHD agreement | D | Lower5 | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| diff-410 | avif-svt | 35 / 105 | 104/105 | 105/105 | -0.009523810 | -0.028571429 | FAIL |
| diff+frag-820 | avif-svt | 35 / 105 | 105/105 | 105/105 | 0.000000000 | 0.000000000 | PASS |
| W-C | avif-svt | 35 / 105 | 105/105 | 105/105 | 0.000000000 | 0.000000000 | PASS |
| diff-410 | jpeg | 22 / 66 | 65/66 | 66/66 | -0.015151515 | -0.045454545 | FAIL |
| diff+frag-820 | jpeg | 22 / 66 | 65/66 | 66/66 | -0.015151515 | -0.045454545 | FAIL |
| W-C | jpeg | 22 / 66 | 65/66 | 66/66 | -0.015151515 | -0.045454545 | FAIL |
| diff-410 | jxl | 33 / 99 | 99/99 | 99/99 | 0.000000000 | 0.000000000 | PASS |
| diff+frag-820 | jxl | 33 / 99 | 99/99 | 99/99 | 0.000000000 | 0.000000000 | PASS |
| W-C | jxl | 33 / 99 | 99/99 | 99/99 | 0.000000000 | 0.000000000 | PASS |
| diff-410 | webp | 16 / 48 | 48/48 | 48/48 | 0.000000000 | 0.000000000 | PASS |
| diff+frag-820 | webp | 16 / 48 | 48/48 | 48/48 | 0.000000000 | 0.000000000 | PASS |
| W-C | webp | 16 / 48 | 48/48 | 48/48 | 0.000000000 | 0.000000000 | PASS |

### Descriptive scorer intervals

| Scorer | Codec | Agreement | Descriptive95% interval |
|---|---|---:|---|
| diff-410 | avif-svt | 0.990476190 | [0.971428571, 1.000000000] |
| diff+frag-820 | avif-svt | 1.000000000 | [1.000000000, 1.000000000] |
| W-C | avif-svt | 1.000000000 | [1.000000000, 1.000000000] |
| CVVDP-FHD | avif-svt | 1.000000000 | [1.000000000, 1.000000000] |
| SSIMULACRA2 | avif-svt | 1.000000000 | [1.000000000, 1.000000000] |
| diff-410 | jpeg | 0.984848485 | [0.954545455, 1.000000000] |
| diff+frag-820 | jpeg | 0.984848485 | [0.954545455, 1.000000000] |
| W-C | jpeg | 0.984848485 | [0.954545455, 1.000000000] |
| CVVDP-FHD | jpeg | 1.000000000 | [1.000000000, 1.000000000] |
| SSIMULACRA2 | jpeg | 1.000000000 | [1.000000000, 1.000000000] |
| diff-410 | jxl | 1.000000000 | [1.000000000, 1.000000000] |
| diff+frag-820 | jxl | 1.000000000 | [1.000000000, 1.000000000] |
| W-C | jxl | 1.000000000 | [1.000000000, 1.000000000] |
| CVVDP-FHD | jxl | 1.000000000 | [1.000000000, 1.000000000] |
| SSIMULACRA2 | jxl | 1.000000000 | [1.000000000, 1.000000000] |
| diff-410 | webp | 1.000000000 | [1.000000000, 1.000000000] |
| diff+frag-820 | webp | 1.000000000 | [1.000000000, 1.000000000] |
| W-C | webp | 1.000000000 | [1.000000000, 1.000000000] |
| CVVDP-FHD | webp | 1.000000000 | [1.000000000, 1.000000000] |
| SSIMULACRA2 | webp | 1.000000000 | [1.000000000, 1.000000000] |

## ladder

| Candidate | Codec | Ladders / edges | Agreement | CVVDP-FHD agreement | D | Lower5 | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| diff-410 | avif-svt | 39 / 117 | 116/117 | 117/117 | -0.008547009 | -0.025641026 | FAIL |
| diff+frag-820 | avif-svt | 39 / 117 | 117/117 | 117/117 | 0.000000000 | 0.000000000 | PASS |
| W-C | avif-svt | 39 / 117 | 117/117 | 117/117 | 0.000000000 | 0.000000000 | PASS |
| diff-410 | jpeg | 39 / 117 | 114/117 | 105/117 | 0.076923077 | 0.034188034 | PASS |
| diff+frag-820 | jpeg | 39 / 117 | 112/117 | 105/117 | 0.059829060 | 0.017094017 | PASS |
| W-C | jpeg | 39 / 117 | 112/117 | 105/117 | 0.059829060 | 0.025641026 | PASS |
| diff-410 | jxl | 26 / 78 | 76/78 | 77/78 | -0.012820513 | -0.038461538 | FAIL |
| diff+frag-820 | jxl | 26 / 78 | 76/78 | 77/78 | -0.012820513 | -0.038461538 | FAIL |
| W-C | jxl | 26 / 78 | 76/78 | 77/78 | -0.012820513 | -0.038461538 | FAIL |
| diff-410 | webp | 39 / 117 | 117/117 | 117/117 | 0.000000000 | 0.000000000 | PASS |
| diff+frag-820 | webp | 39 / 117 | 117/117 | 117/117 | 0.000000000 | 0.000000000 | PASS |
| W-C | webp | 39 / 117 | 117/117 | 117/117 | 0.000000000 | 0.000000000 | PASS |

### Descriptive scorer intervals

| Scorer | Codec | Agreement | Descriptive95% interval |
|---|---|---:|---|
| diff-410 | avif-svt | 0.991452991 | [0.974358974, 1.000000000] |
| diff+frag-820 | avif-svt | 1.000000000 | [1.000000000, 1.000000000] |
| W-C | avif-svt | 1.000000000 | [1.000000000, 1.000000000] |
| CVVDP-FHD | avif-svt | 1.000000000 | [1.000000000, 1.000000000] |
| SSIMULACRA2 | avif-svt | 1.000000000 | [1.000000000, 1.000000000] |
| diff-410 | jpeg | 0.974358974 | [0.940170940, 1.000000000] |
| diff+frag-820 | jpeg | 0.957264957 | [0.923076923, 0.991452991] |
| W-C | jpeg | 0.957264957 | [0.923076923, 0.991452991] |
| CVVDP-FHD | jpeg | 0.897435897 | [0.837606838, 0.948717949] |
| SSIMULACRA2 | jpeg | 1.000000000 | [1.000000000, 1.000000000] |
| diff-410 | jxl | 0.974358974 | [0.935897436, 1.000000000] |
| diff+frag-820 | jxl | 0.974358974 | [0.935897436, 1.000000000] |
| W-C | jxl | 0.974358974 | [0.935897436, 1.000000000] |
| CVVDP-FHD | jxl | 0.987179487 | [0.961538462, 1.000000000] |
| SSIMULACRA2 | jxl | 1.000000000 | [1.000000000, 1.000000000] |
| diff-410 | webp | 1.000000000 | [1.000000000, 1.000000000] |
| diff+frag-820 | webp | 1.000000000 | [1.000000000, 1.000000000] |
| W-C | webp | 1.000000000 | [1.000000000, 1.000000000] |
| CVVDP-FHD | webp | 1.000000000 | [1.000000000, 1.000000000] |
| SSIMULACRA2 | webp | 1.000000000 | [1.000000000, 1.000000000] |

SSIMULACRA2 self-agreement is 100% by construction on all 747 edges, descriptive only. Exact counts, point differences, bounds, missing-score counts and descriptive intervals are in [the machine-readable table](shipgates_a7r_direction_2026-10-17.tsv).

## Pins and validation

- diff-410 model SHA256 `7cf4cfd92bb8824405db3ed27287bbec973a01eaeadf09b7ad037db92ab9a275`.
- diff+frag-820 model SHA256 `89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e`.
- W-C model SHA256 `f73059ff8f67ff177265d35dfd5ff5db61d8fc825a6b519cf6925dc8fc1d5e2a`.

RESULT SHA256 `9cde62c345247baacd3b1dfdc141d3f6893f9ef7a0e00c274af4497ba08008f5`; shared edge ledger SHA256 `79946700bb56f05f6a94b667acca01285a52ad7a77c423f4941d7399fdc76825`. [Artifact pointer](shipgates_2026-10-16.pointer.md) retains full input, source, canonical statistics and binary hashes. No new model scoring, encodes, training or human-label access. Native checks cover unequal edge counts per ladder, whole-ladder resampling, paired identical-scorer differences and deterministic seed replay. Original evidence remains immutable.

```text
run-heavy: done rc=0 2s | peak-RSS 0.21GiB | min-avail 26468MiB | peak-load 28.34
```

One CPU, 4 GiB cap, nice 19/ionice 3; no paper lock. The resource line describes this contended evaluation, not a performance qualification. JXL remains paused for priority generation.
