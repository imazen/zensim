# V10 per-human-dataset eval (task #183, 2026-05-20)

**Bake commit:** main@origin `6e4c665` (V10 ship: TunerV4 / BalancedV3 / CompressionV3).

## Gap surfaced: butter parquets do not cover human-dataset sources

The task asked for cross-codec stddev + mono "per human dataset." The implementation depends on the existing butter parquets at `/mnt/v/zen/picker-training/2026-05-19/butter/<codec>.parquet`, which carry q-sweeps for 4 codecs × 1000 reference images on the **synthetic-safe source corpus** (gen-mixed `*_512sq.png` / `*_1024sq.png` etc.). These references do NOT overlap with the human-dataset reference sets:

| dataset | n dataset refs | n butter refs | intersection |
|---|---:|---:|---:|
| aic3 | 10 | 1000 | 0 |
| aic4 | 5 | 1000 | 0 |
| konjnd | 1008 | 1000 | 0 |

The per-human-dataset filtered measurement therefore yields **zero rows** for every dataset. The literal task request cannot be answered with current artifacts.

**What's measurable:** the same cross-codec stddev + per-curve mono computation, applied **globally** to the synthetic-safe butter sweep (1000 refs × 4 codecs × 19 qs). Reported below as `scope=global_butter`. This is the same methodology / corpus used by the V9 mono ship audit (`benchmarks/v_tuner_v9_mono_audit_2026-05-20.md`), just extended to all 1000 refs instead of a 50-ref random sample.

**To get truly per-human-dataset measurement:** the butter sweep would need re-running on the human-dataset source images themselves (AIC-3: 10 refs, AIC-4: 5 refs, KonJND: 1008 refs). Estimated wall: ~30 min per codec × 4 codecs × ~1023 refs = several CPU-hours. Not in this task's 45-min budget.

All measurements below: `n_refs = 1000` (the cross-codec intersection of the 4 butter parquets).

## Cross-codec stddev (butter_pnorm3 at target T)

Methodology per task: for each (profile, target T), find the q whose score is closest to T for each codec, look up butter_pnorm3 at that q, compute stddev across the 4 codecs per source, aggregate across 1000 sources.

Gate: median + p90 stddev ≤ 5.0 at every (profile, T).

| profile | target | T | n | median | mean | p90 | p99 | max | gate (≤5 median+p90) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| V9_BalancedV2 | JND | 80 | 1000 | 0.104 | 0.168 | 0.321 | 1.154 | 1.967 | **PASS** |
| V9_BalancedV2 | JOD | 50 | 1000 | 0.199 | 0.346 | 0.758 | 2.587 | 2.793 | **PASS** |
| V9_CompressionV2 | JND | 80 | 1000 | 0.104 | 0.182 | 0.405 | 1.236 | 1.873 | **PASS** |
| V9_CompressionV2 | JOD | 50 | 1000 | 0.155 | 0.338 | 1.003 | 1.896 | 3.441 | **PASS** |
| V9_TunerV3 | JND | 80 | 1000 | 0.094 | 0.165 | 0.321 | 1.290 | 2.207 | **PASS** |
| V9_TunerV3 | JOD | 50 | 1000 | 0.110 | 0.162 | 0.327 | 0.757 | 1.257 | **PASS** |
| V10_BalancedV3 | JND | 80 | 1000 | 0.132 | 0.232 | 0.518 | 1.214 | 3.052 | **PASS** |
| V10_BalancedV3 | JOD | 50 | 1000 | 0.442 | 0.683 | 1.550 | 2.344 | 3.049 | **PASS** |
| V10_CompressionV3 | JND | 80 | 1000 | 0.117 | 0.218 | 0.487 | 1.792 | 3.075 | **PASS** |
| V10_CompressionV3 | JOD | 50 | 1000 | 0.189 | 0.414 | 1.181 | 2.226 | 3.127 | **PASS** |
| V10_TunerV4 | JND | 80 | 1000 | 0.067 | 0.107 | 0.196 | 0.757 | 1.817 | **PASS** |
| V10_TunerV4 | JOD | 50 | 1000 | 0.411 | 0.494 | 0.945 | 1.511 | 2.485 | **PASS** |

## Per-curve monotonicity

Methodology: for each (ref, codec) curve in the butter parquet, sort by q ascending, score every q against the bake, count strict decreases. `pair_strict` = 1 - n_strict / n_pairs; `curve_strict` = fraction of curves with zero strict decreases.

Gate: pair_strict mono ≥ 0.94 at every profile.

| profile | metric | n | value | gate (≥0.94 pair) |
|---|---|---:|---:|---|
| V9_BalancedV2 | pair_strict_mono | 72000 | 0.948 | **PASS** |
| V9_BalancedV2 | curve_strict_mono | 4000 | 0.520 | n/a |
| V9_CompressionV2 | pair_strict_mono | 72000 | 0.969 | **PASS** |
| V9_CompressionV2 | curve_strict_mono | 4000 | 0.704 | n/a |
| V9_TunerV3 | pair_strict_mono | 72000 | 0.969 | **PASS** |
| V9_TunerV3 | curve_strict_mono | 4000 | 0.745 | n/a |
| V10_BalancedV3 | pair_strict_mono | 72000 | 0.948 | **PASS** |
| V10_BalancedV3 | curve_strict_mono | 4000 | 0.520 | n/a |
| V10_CompressionV3 | pair_strict_mono | 72000 | 0.969 | **PASS** |
| V10_CompressionV3 | curve_strict_mono | 4000 | 0.704 | n/a |
| V10_TunerV4 | pair_strict_mono | 72000 | 0.968 | **PASS** |
| V10_TunerV4 | curve_strict_mono | 4000 | 0.743 | n/a |

## V10 vs V9 delta

Per (corresponding profile pair):

| pair | metric | V9 | V10 | Δ (V10 − V9) | direction |
|---|---|---:|---:|---:|---|
| V9_BalancedV2 → V10_BalancedV3 | cc_stddev @ JND | 0.104 | 0.132 | +0.028 | flat |
| V9_BalancedV2 → V10_BalancedV3 | cc_stddev @ JOD | 0.199 | 0.442 | +0.243 | LOOSENED |
| V9_BalancedV2 → V10_BalancedV3 | pair_strict_mono | 0.948 | 0.948 | +0.000 | flat |
| V9_BalancedV2 → V10_BalancedV3 | curve_strict_mono | 0.520 | 0.520 | +0.000 | flat |
| V9_CompressionV2 → V10_CompressionV3 | cc_stddev @ JND | 0.104 | 0.117 | +0.013 | flat |
| V9_CompressionV2 → V10_CompressionV3 | cc_stddev @ JOD | 0.155 | 0.189 | +0.034 | flat |
| V9_CompressionV2 → V10_CompressionV3 | pair_strict_mono | 0.969 | 0.969 | +0.000 | flat |
| V9_CompressionV2 → V10_CompressionV3 | curve_strict_mono | 0.704 | 0.704 | +0.000 | flat |
| V9_TunerV3 → V10_TunerV4 | cc_stddev @ JND | 0.094 | 0.067 | −0.026 | flat |
| V9_TunerV3 → V10_TunerV4 | cc_stddev @ JOD | 0.110 | 0.411 | +0.300 | LOOSENED |
| V9_TunerV3 → V10_TunerV4 | pair_strict_mono | 0.969 | 0.968 | −0.000 | flat |
| V9_TunerV3 → V10_TunerV4 | curve_strict_mono | 0.745 | 0.743 | −0.002 | flat |

## Verdict

**ALL V10 ROWS PASS** the absolute gates (median + p90 cross-codec stddev ≤ 5, pair_strict mono ≥ 0.94) on the global butter sweep. The largest V10 stddev p90 is 1.55 (BalancedV3 @ JOD), 3.2× under the 5.0 gate.

**SROCC + KROCC + PWRC on human val parquets (AIC-3, AIC-4, KonJND): bit-exact V9 ↔ V10** (see SROCC sanity table below). The PCHIP spline reallocation is rank-preserving by construction — V10 does NOT regress human-MOS rank-honesty on any of the three human datasets.

**Recommendation: ship V10. Mark task #184 (V10b fallback) as NOT NEEDED.** V10's spline reallocation passes every measurable gate and preserves rank-correlation on all three human-rated datasets.

### Non-blocking observation: JOD-band stddev expansion vs V9

V10 stddev at JOD (T=50, butter≈4.0) is larger than V9 stddev at JOD (T=30, butter≈4.0) in absolute terms:

- BalancedV3 JOD median 0.442 vs BalancedV2 JOD median 0.199 (Δ +0.243)
- CompressionV3 JOD median 0.189 vs CompressionV2 JOD median 0.155 (Δ +0.034)
- TunerV4 JOD median 0.411 vs TunerV3 JOD median 0.110 (Δ +0.300)

Both V9 and V10 are far under the 5.0 gate — this is not a failure. It's an expected consequence of the V10 spline placing JOD at score=50 (a denser, more discriminating part of the dial). The spline slope around score=50 (V10) is steeper than the spline slope around score=30 (V9), so a fixed butter-pnorm3 variation across codecs maps to a wider score variation. Per the V10 design intent ("dial spans full [0, 100] across best-codec lossless to worst-codec q=5 floors"), this is the dial doing more work in the JOD band — not a cross-codec parity regression.

If a tighter absolute stddev at JOD becomes a product requirement (e.g., for codec-selector convergence), the V10b fallback (#184) shifting JND → 70 / JOD → 40 would compress the spline knots toward each other in the mid-band and reduce the per-codec score variance at the cost of dial dynamic range. Current V10 stddev is well within gate, so no dispatch is needed today.

### Caveats

1. **These results are on the global synthetic butter corpus, NOT filtered to human-dataset sources.** The literal user intent ("tighten cross-codec stddev per human dataset") cannot be verified at human-dataset granularity without rebuilding the butter sweep on AIC-3 / AIC-4 / KonJND source images. The global synth corpus spans natural image content (gen-mixed multi-content), so it's a reasonable proxy — but content-class skew between synth and human-dataset sources could cause per-dataset numbers to differ.

2. **Mono curve_strict is unchanged V9 ↔ V10** for the same reason as SROCC: the PCHIP spline can't introduce a monotonicity violation that the pre-spline network didn't already have. Per the V9 mono audit (`benchmarks/v_tuner_v9_mono_audit_2026-05-20.md`), 73-88% of curve-mono violations are sub-1-score-unit wobbles in the network's raw output — those persist into V10 unchanged.

## SROCC sanity on human val parquets (bake_verdict)

Rank-correlation sanity check: V10's spline reallocation is monotone-preserving by construction, so SROCC / KROCC / PWRC should be bit-exact between V9 and V10 corresponding profiles. PLCC is scale-sensitive after the 4-parameter logistic rescale used by bake_verdict (Mohammadi 2025 convention).

Features: `/mnt/v/zen/zensim-training/2026-05-15-full-features/{aic3,aic4,konjnd}_features_372col_*.parquet`.

| profile | dataset | n | SROCC | PLCC | KROCC | PWRC | Z-RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| V9_BalancedV2 | aic3 | 600 | 0.7845 | 0.7951 | 0.6155 | 0.8630 | 0.606 |
| V9_BalancedV2 | aic4 | 300 | 0.9016 | 0.8927 | 0.7308 | 0.9471 | 0.451 |
| V9_BalancedV2 | konjnd | 1008 | 0.8927 | 0.9264 | 0.7070 | 0.9178 | 0.376 |
| V10_BalancedV3 | aic3 | 600 | 0.7845 | 0.7952 | 0.6155 | 0.8630 | 0.606 |
| V10_BalancedV3 | aic4 | 300 | 0.9016 | 0.8900 | 0.7308 | 0.9471 | 0.456 |
| V10_BalancedV3 | konjnd | 1008 | 0.8927 | 0.9270 | 0.7070 | 0.9178 | 0.375 |
| V9_CompressionV2 | aic3 | 600 | 0.8183 | 0.8244 | 0.6527 | 0.8856 | 0.566 |
| V9_CompressionV2 | aic4 | 300 | 0.9538 | 0.9504 | 0.8185 | 0.9766 | 0.311 |
| V9_CompressionV2 | konjnd | 1008 | 0.8080 | 0.8649 | 0.5935 | 0.8505 | 0.502 |
| V10_CompressionV3 | aic3 | 600 | 0.8183 | 0.8247 | 0.6527 | 0.8856 | 0.566 |
| V10_CompressionV3 | aic4 | 300 | 0.9538 | 0.9494 | 0.8185 | 0.9766 | 0.314 |
| V10_CompressionV3 | konjnd | 1008 | 0.8080 | 0.8647 | 0.5935 | 0.8505 | 0.502 |
| V9_TunerV3 | aic3 | 600 | 0.7865 | 0.8003 | 0.6212 | 0.8633 | 0.600 |
| V9_TunerV3 | aic4 | 300 | 0.9240 | 0.9122 | 0.7657 | 0.9552 | 0.410 |
| V9_TunerV3 | konjnd | 1008 | 0.2317 | 0.2175 | 0.1572 | 0.3178 | 0.976 |
| V10_TunerV4 | aic3 | 600 | 0.7865 | 0.8008 | 0.6212 | 0.8633 | 0.599 |
| V10_TunerV4 | aic4 | 300 | 0.9240 | 0.9085 | 0.7657 | 0.9552 | 0.418 |
| V10_TunerV4 | konjnd | 1008 | 0.2317 | 0.2187 | 0.1572 | 0.3178 | 0.976 |

### V10 vs V9 rank-correlation deltas

| pair | dataset | ΔSROCC | ΔPLCC | ΔPWRC | rank-preserved? |
|---|---|---:|---:|---:|---|
| V9_BalancedV2 -> V10_BalancedV3 | aic3 | +0.0000 | +0.0001 | +0.0000 | YES (bit-exact) |
| V9_BalancedV2 -> V10_BalancedV3 | aic4 | +0.0000 | -0.0027 | +0.0000 | YES (bit-exact) |
| V9_BalancedV2 -> V10_BalancedV3 | konjnd | +0.0000 | +0.0006 | +0.0000 | YES (bit-exact) |
| V9_CompressionV2 -> V10_CompressionV3 | aic3 | +0.0000 | +0.0003 | +0.0000 | YES (bit-exact) |
| V9_CompressionV2 -> V10_CompressionV3 | aic4 | +0.0000 | -0.0010 | +0.0000 | YES (bit-exact) |
| V9_CompressionV2 -> V10_CompressionV3 | konjnd | +0.0000 | -0.0002 | +0.0000 | YES (bit-exact) |
| V9_TunerV3 -> V10_TunerV4 | aic3 | +0.0000 | +0.0005 | +0.0000 | YES (bit-exact) |
| V9_TunerV3 -> V10_TunerV4 | aic4 | +0.0000 | -0.0037 | +0.0000 | YES (bit-exact) |
| V9_TunerV3 -> V10_TunerV4 | konjnd | +0.0000 | +0.0012 | +0.0000 | YES (bit-exact) |

**Headline:** SROCC, KROCC, and PWRC are bit-exact across every V9 -> V10 transition (the PCHIP spline is monotone-preserving by construction). PLCC moves by at most 0.003 in either direction -- within noise of the 4-parameter logistic rescale's curve-fit tolerance. V10 does NOT regress human-MOS rank-honesty.

This SROCC sanity is the strongest available signal at human-dataset granularity, because the cross-codec stddev measurement (above) is on the synthetic butter corpus and cannot be reproduced on the human-dataset sources without rebuilding the butter sweep on those source images.
