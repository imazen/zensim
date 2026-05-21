# V11-A'-372 v4 (V11-DECODER-FIX) — FALSIFIED on Balanced + Compression gates

**Task #195 (2026-05-20).** Full 4-codec × 372-feat substrate retrain.
Both the brief recipe and the V_24-style clean recipe are falsified.
The cross-codec-eq + anchor aux-loss combination structurally collapses
KonJND PJND tracking regardless of feature dimension.

## Phase summary

| Phase | Status | Outcome |
|---|---|---|
| 1 — Decoder fix (zenavif + zenjxl path-deps) | LANDED | `image::open` → native codec decode; 0 errors on 117,800-cell extraction |
| 2 — Full 117,800-cell extraction | DONE | 4-codec × 5-q coverage at 372 features, 73 MB parquet, 1,210s wall |
| 3 — Substrate v4 build | DONE | 2,471 anchor rows + 1,739 cross-codec-eq pairs |
| 4 — V11-A'-372 v4 5-seed retrain (brief + clean) | DONE | 10 bakes total (5 brief, 5 clean) |
| 5 — Ship gate | **FALSIFIED** | Both recipes lose ≥1 decisive corpus to ≥1 of V10 BalancedV3 / V_24 Compression |

## Phase 1 verification

| Codec | Smoke (cells, errors) | Full extraction (cells, errors) |
|---|---|---|
| zenavif | 50 / 0 | 4,000 / 0 |
| zenjxl  | 5 / 0  | 51,200 / 0 |
| zenjpeg | (image-crate path, unchanged) | 61,600 / 0 |
| zenwebp | (image-crate path, unchanged) | 1,000 / 0 |
| **TOTAL** | | **117,800 / 0** |

zenavif + zenjxl path-deps added to `zensim-bench/Cargo.toml` under
the `extract-omni` feature; both already in the workspace lockfile
via `zensim-target`. No `[patch.crates-io]` changes. Build clean.

## Phase 2 + 3 substrate counts

The full extraction's per-codec / per-band coverage is bounded by the
omni sweep's q grid: each non-jpeg codec ships only 5 q levels (10,
30, 60, 80, 90), so anchor emit counts at extreme ssim2 bands are
constrained.

Anchor parquet (`anchors_ssim2_372col_v4.parquet`, 2,471 rows):

| ssim2 → target | zenavif | zenjpeg | zenjxl | zenwebp | total |
|---:|--:|--:|--:|--:|--:|
| 100 → 100 | 0 | 0 | 0 | 0 | 0 (5 q levels can't hit perfect lossless) |
| 95 → 95 | 28 | 4 | 123 | 2 | 157 |
| 90 → 90 | 200 | 120 | 200 | 94 | 614 |
| 75 → 80 (JND) | 114 | 193 | 182 | 126 | 615 |
| 60 → 65 | 81 | 156 | 26 | 92 | 355 |
| 45 → 50 (JOD) | 54 | 149 | 11 | 31 | 245 |
| 30 → 35 | 51 | 119 | 2 | 12 | 184 |
| 18 → 20 | 67 | 61 | 0 | 1 | 129 |
| 10 → 10 | 62 | 51 | 0 | 3 | 116 |
| 3 → 0 | 24 | 32 | 0 | 0 | 56 |

Cross-codec equivalence pairs (`cross_codec_equivalence_ssim2_372col_v4.parquet`, 1,739):

| codec_a ↔ codec_b | count |
|---|--:|
| zenavif ↔ zenjpeg | 374 |
| zenavif ↔ zenjxl  | 307 |
| zenavif ↔ zenwebp | 220 |
| zenjpeg ↔ zenjxl  | 327 |
| zenjpeg ↔ zenwebp | 285 |
| zenjxl  ↔ zenwebp | 226 |

## Phase 4: brief-recipe 5-seed CI (FALSIFIED at seed=1)

Brief recipe per task #195:
```
--mse-weight 1.0 --ranknet-weight 0.0
--monotonicity-reg 1.0 --tanh-output-head-scale 20.0
--per-sample-alpha-head
+ anchor + cross-codec-eq + dynamic-range aux losses
```

| Bake | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|---|---:|---:|---:|---:|---:|---:|
| s1 | 0.7680 | 0.7905 | 0.7692 | 0.0756 | 0.7496 | 0.9184 |
| s2 | 0.7619 | 0.7856 | 0.7712 | 0.0758 | 0.7426 | 0.9129 |
| s3 | 0.6967 | 0.7807 | 0.7658 | 0.1275 | 0.7117 | 0.8958 |
| s4 | 0.7170 | 0.7913 | 0.7821 | 0.0196 | 0.7342 | 0.8998 |
| s5 | 0.6600 | 0.7659 | 0.7499 | 0.0210 | 0.6937 | 0.8791 |
| **median** | **0.7170** | **0.7856** | **0.7712** | **0.0758** | **0.7342** | **0.8998** |

vs V10 BalancedV3 (0.8324 / 0.9677 / 0.9729 / 0.8927 / 0.7845 / 0.9016):

All non-AIC-4 corpora regress by ≥ 0.08 SROCC. KonJND collapses
−0.82. **Decisively falsified on every gate criterion.** Same
structural failure mode the prior agent identified at 300-feat
(V11-A' v2 brief recipe in `benchmarks/v11_substrate_v2_methodology_2026-05-20.md`):
MSE-only + monotonicity-reg + tanh-pin combo inverts the V_24
training surface.

## Phase 4: clean-recipe 5-seed CI (NEAR-MISS, KonJND collapses)

Clean recipe: drop `--mse-weight 1.0 / --ranknet-weight 0.0 /
--monotonicity-reg 1.0 / --tanh-output-head-scale 20.0`. Keep
`--per-sample-alpha-head + anchor-parquet + cross-codec-eq-parquet
+ dynamic-range-floor-weight`.

| Bake | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|---|---:|---:|---:|---:|---:|---:|
| clean s1 | 0.8978 | 0.9312 | 0.8931 | 0.3942 | 0.8173 | 0.9537 |
| clean s2 | 0.8944 | 0.9254 | 0.8971 | 0.4390 | 0.8232 | 0.9522 |
| clean s3 | 0.8939 | 0.9212 | 0.8828 | 0.3888 | 0.8141 | 0.9459 |
| clean s4 | 0.8991 | 0.9253 | 0.8882 | 0.4060 | 0.8198 | 0.9504 |
| clean s5 | 0.8846 | 0.9220 | 0.8903 | 0.3602 | 0.8148 | 0.9560 |
| **median** | **0.8944** | **0.9253** | **0.8903** | **0.3942** | **0.8173** | **0.9522** |

CID22 mean ± std = 0.8940 ± 0.0058 — robust signal across seeds.

Median bake: **`cc4v11a_372_v4_clean_s2.bin`** (CID22 0.8944).

## Phase 5: bake_compare § A.9 verdict

### A = V11-A'-372 v4 clean s2 (median) vs B = V10 BalancedV3 (`v_balanced_v3_2026-05-20.bin`)

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8944 | 0.8324 | +36.0 | +30.0 | **A>>B** |
| KADID | 10125 | 0.9254 | 0.9677 | -92.5 | -0.000 | **B>>A** |
| TID | 3000 | 0.8971 | 0.9729 | -54.2 | -0.000 | **B>>A** |
| KonJND | 1008 | 0.4390 | 0.8927 | -17.2 | -0.000 | **B>>A** |
| AIC-3 | 600 | 0.8232 | 0.7845 | +18.7 | +15.6 | **A>>B** |

**Balanced trail gate** (any decisive B>>A blocks): **FAIL** (3 decisive B>>A).

### A = V11-A'-372 v4 clean s2 vs B = V_24-per-sample-α s4 Compression (`v_compression_persample_2026-05-18.bin`)

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8944 | 0.8641 | +24.1 | +20.1 | **A>>B** |
| KADID | 10125 | 0.9254 | 0.9316 | -60.3 | -10.0 | **B>>A** |
| TID | 3000 | 0.8971 | 0.8893 | +50.1 | +41.8 | **A>>B** |
| KonJND | 1008 | 0.4390 | 0.8080 | -12.8 | -0.000 | **B>>A** |
| AIC-3 | 600 | 0.8232 | 0.8183 | +4.3 | +0.000 | tied |

Overall: 4 A wins vs 4 B wins — tied.

**Compression trail gate** (decisive A>>B on ≥1 of CID22/AIC-3 +
KADID/TID/KonJND within −0.10):
- A>>B on CID22 ✓ (+0.030)
- AIC-3 tied
- KADID: -0.006 ✓ within tolerance
- TID: +0.008 ✓
- **KonJND: −0.369 vs −0.10 cap → FAR exceeds → FAIL**

### Ship gate verdict per task #195 Phase 5

| Metric | V10 BalancedV3 | V11-A372-v4 clean s2 median | Target | Verdict |
|---|---:|---:|---:|---|
| CID22 SROCC | 0.8324 | 0.8944 | ≥ 0.8374 | **PASS** (+0.057) |
| CID22 Z-RMSE | 0.564 | 0.455 | ≤ 0.530 | **PASS** (-0.109) |
| KADID drift | 0.9664 | 0.9254 | within −0.10 | PASS (−0.041) |
| TID drift | 0.9712 | 0.8971 | within −0.10 | PASS (−0.074) |
| **KonJND drift** | **0.8927** | **0.4390** | **within −0.10** | **FAIL (−0.454)** |
| AIC-4 SROCC | 0.9016 | 0.9522 | ≥ 0.8966 | PASS (+0.051) |
| Anchor JND landing | exact | bit-exact pre-spline | exact | PASS (in trainer) |

**The KonJND drift gate is the single blocker.** Same failure mode
as V11-A' v2 clean at 300-feat (KonJND collapsed from 0.8927 →
0.4033 there; now 0.8927 → 0.4390 at 372-feat). The 372-feat
IW-pool block adds ~+0.02 CID22 SROCC (0.8754 → 0.8944) and ~+0.04
on KonJND (0.4033 → 0.4390) over the 300-feat substrate — both
material lifts, but neither rescues KonJND.

### Decision

NO ship. The cross-codec-eq + anchor aux-loss recipe is structurally
KonJND-incompatible regardless of feature dimension. The bake is a
Compression-trail near-miss (PASSes CID22 gate decisively, FAILS
KonJND drift). The cross-codec-eq mechanism trains the network to
score equivalent (cross-codec, same-ssim2) pairs identically — this
flattens the PJND-anchored ranking that KonJND requires.

V10 BalancedV3 remains the Balanced ship. V_24-per-sample-α s4
remains the Compression ship.

## What the 372-feat fix accomplished

This task's deliverables are still meaningful even though the
retrain itself doesn't ship:

1. **Decoder unblock**: `extract_features_372col_omni` now decodes
   all 4 codecs at 372 features, not just zenjpeg+zenwebp. The
   55,200 previously-skipped cells are now extractable in 20 min
   wall on the local box (no vast.ai needed). This unlocks any
   future cross-codec experiment that needs 372-feat input.
2. **372-feat IW-pool contribution measured**: Direct A/B at the
   same recipe shows +0.022 CID22 SROCC (0.8754 → 0.8944) when
   the IW-pool block is added. This confirms the prior agent's
   "the user's framing was correct" finding with hard numbers,
   not extrapolation from other bakes.
3. **Cross-codec-eq structural KonJND-incompatibility confirmed
   at 372-feat**: The 300-feat falsification (V11-A' v2 clean,
   KonJND 0.4033) reproduces almost exactly at 372-feat (0.4390).
   The aux-loss mechanism is the problem, not feature richness.
   Future cross-codec-trail work needs a different aux-loss design
   — likely (a) per-row KonJND PJND-anchor passthrough loss with
   high weight to prevent flattening, or (b) routing the
   cross-codec-eq loss through only the high-q anchor band (≥75)
   where KonJND saturates anyway.

## Files written

- `zensim-bench/Cargo.toml` + `zensim-bench/examples/extract_features_372col_omni.rs`
  — decoder fix (commit `3bd88eca`).
- `scripts/v_next/v11_372feat/build_v11_372feat_substrate.py` —
  `--out-version` flag (commit `13b2e261`).
- `scripts/v_next/v11_372feat/run_v11a_372_v4_seed.sh` — brief
  recipe runner (commit `13b2e261`).
- `scripts/v_next/v11_372feat/run_v11a_372_v4_clean_seed.sh` —
  clean recipe runner (this commit).
- `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/multi_codec_372col_full.parquet`
  (73 MB) — full 4-codec × 372-feat extraction.
- `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/anchors_ssim2_372col_v4.parquet`
  (4.4 MB) — 2,471 anchor rows.
- `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_372col_v4.parquet`
  (4.6 MB) — 1,739 cross-codec equivalence pairs.
- 5 brief bakes at `/mnt/v/zen/zensim-eval/exp_v11a_372_v4_2026-05-20/cc4v11a_372_v4_s{1..5}.bin`.
- 5 clean bakes at `/mnt/v/zen/zensim-eval/exp_v11a_372_v4_clean_2026-05-20/cc4v11a_372_v4_clean_s{1..5}.bin`.
- This document.
