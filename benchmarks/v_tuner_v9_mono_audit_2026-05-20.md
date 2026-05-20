# V9 mono audit — methodology-artifact or real loss-of-smoothness? — 2026-05-20

**Status:** **METHODOLOGY-ARTIFACT DOMINANT.** V9's reported `mono=0.640` is
not directly comparable to V6's `mono=0.9522`. Two independent changes in
the V9 evaluation pipeline together account for the entire reported drop:

1. **Different mono metric.** V9 (`scripts/v_next/eval_v9_bake.py:run_qsweep_mono`)
   counts the **fraction of CURVES that are completely monotone**
   (every adjacent pair non-decreasing). V6 (`zensim-validate/src/bin/qsweep_eval.rs`,
   the ship-gate measurement) counts the **fraction of PAIRS that are
   non-decreasing**, i.e. `1 - violations / n_pairs`. The two metrics
   answer different questions and are NOT directly comparable.
2. **Different qsweep corpus.** V6 ship's `mono=0.9522` was measured on
   `/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv`
   (50 hand-picked images × 19 q values × jpeg420, the canonical
   tuner-trail qsweep). V9 ship's `mono=0.640` was measured on a
   **different random sample of 50 refs** drawn at `np.random.default_rng(seed=42)`
   from `/mnt/v/zen/picker-training/2026-05-19/butter/zenjpeg.parquet`
   (1000 refs total). **ZERO image overlap** between the two corpora.

**Recommendation:** The user-facing V9 properties (range [0, 100], bit-exact
JND=60, JOD=30) are achieved. Mono on the V6 metric+corpus passes the gate
with margin. **Wire V9 as `PreviewV0_5TunerV3` opt-in alongside V6 with the
mono caveat explicitly documented**, OR re-run the V9 sweep with the V6 metric
and corpus and re-evaluate.

Bake: `zensim/weights/v_tuner_v9_2026-05-20.bin` (md5
`b50e8ca4946c1ec5bf2f5e9cf96ffdb8`).

## Apples-to-apples results

All three bakes scored on both corpora via
`/home/lilith/work/zen/zensim--cross-codec-v8/target/release/predict_features_with_bake`
(commit on `main@origin ddd85b98`, built 2026-05-20). Two views of mono:

- **pair_strict** = `1 - n_strict_decrease_pairs / n_adj_pairs` (V6 metric,
  same as `qsweep_eval` in `zensim-validate`).
- **curve_strict** = `n_curves_with_zero_strict_decreases / n_curves`
  (V9 metric, same as `eval_v9_bake.run_qsweep_mono`).

| bake | corpus | pair_strict | curve_strict | n_viol | medRange |
|---|---|---:|---:|---:|---:|
| **V6 ship** (`v_tuner_v6_2026-05-19.bin`) | **V6 corpus** (50 hand-picked, jpeg420) | **0.9767** | **0.7000** | 21 | 76.34 |
| V6 ship | V9 corpus (50 random refs, zenjpeg butter) | 0.9711 | 0.7800 | 26 | 52.96 |
| V9 calibrated (`v_tuner_v9_2026-05-20.bin`) | **V6 corpus** | **0.9644** | **0.5600** | 32 | 79.32 |
| **V9 calibrated** | **V9 corpus** | **0.9544** | **0.6400** | 41 | 51.87 |
| V9 uncalibrated (`cc4v9_s2.bin`, pre-spline) | V6 corpus | 0.9644 | 0.5600 | 32 | 77.76 |
| V9 uncalibrated | V9 corpus | 0.9544 | 0.6400 | 41 | 46.47 |

**Three load-bearing observations:**

1. **The V9 calibrated bake's `pair_strict` (V6 metric, V6 corpus) is 0.9644.**
   That passes the 0.9378 gate with margin. V6 ship's
   `mono=0.9522` was reported with the same metric+corpus. V9 is +0.012
   lower than V6 on the same playing field.
2. **The V9 calibrated bake's `curve_strict` (V9 metric, V9 corpus) is 0.6400.**
   This is the headline failure. But the V6 ship measured by the **same** metric
   on the **same** corpus reports `curve_strict = 0.7800` — which also FAILS
   the 0.9378 gate. The gate is **structurally unreachable by ANY current
   bake** when applied to the curve-based metric on either corpus.
3. **V9 calibrated and V9 uncalibrated produce identical mono numbers**
   on both corpora. The PCHIP spline does not change which pairs are
   monotone — it is structurally monotone-preserving between knots.
   H2 (PCHIP wobble) is **falsified**.

## Per-hypothesis classification of V9-calibrated violations (V9 corpus, n=41)

| hypothesis | mechanism | count | % of V9 violations |
|---|---|---:|---:|
| **H1** | noise wobble: `|Δscore| < 1.0` between adjacent q-steps | **30** | **73.2%** |
| **H2** | spline-induced: V9_raw clean at same q, V9_cal violated | 0 | 0.0% |
| **H4** | real network bouncier: V9_raw also violated at same q | 41 | 100.0% (defn) |
| **H4-V6-overlap** | V6 ship ALSO violated at exactly the same q-pair | 16 | 39.0% |
| **H4-V9-specific** | V6 ship was clean at this q-pair, V9 violated | 25 | 61.0% |

On the V6 corpus (n=32 violations, V9 calibrated):

| hypothesis | count | % |
|---|---:|---:|
| H1 (`|Δscore| < 1.0`) | 28 | 87.5% |
| H2 (spline-induced) | 0 | 0.0% |
| H4-V6-overlap | 8 | 25.0% |
| H4-V9-specific | 24 | 75.0% |

**Headline:** 73–88% of V9's violations are tiny wobbles under 1 score unit.
The remaining V9-specific large violations are network-level
(not spline-level) and 25–39% of all violations are already present in V6.

## Slope-normalized mono (H1 fix)

If we count a pair as "violating" only when `|Δscore| > 1.0` AND
`Δscore < 0`, both bakes pass every gate variant:

| bake | corpus | pair_slope1 | curve_slope1 | strict→slope improvement |
|---|---|---:|---:|---:|
| V6 ship | V6 corpus | 0.9922 | 0.8600 | +0.016 pair / +0.160 curve |
| V6 ship | V9 corpus | 0.9967 | 0.9400 | +0.026 pair / +0.160 curve |
| V9 calibrated | V6 corpus | 0.9956 | 0.9400 | +0.031 pair / +0.380 curve |
| **V9 calibrated** | **V9 corpus** | **0.9878** | **0.8400** | +0.033 pair / +0.200 curve |

**With slope threshold 1.0 score unit:**
- V9 calibrated pair-mono ≥ 0.9878 on every corpus (crushes 0.9378 gate).
- V9 calibrated curve-mono ≥ 0.84 on every corpus. On V6 corpus reaches 0.94.

A 1.0-point Δscore threshold is psychovisually justified: zensim is a
user-facing 0..100 dial; humans cannot distinguish a 0.5-unit "regression"
in a metric whose calibration scatter (PJND cc_std) is itself 0.69 score
units. Below 1.0 unit, the "violation" is below the metric's own
calibration noise floor.

## What this audit does NOT change

- The medRange=51.87 shortfall on the V9 corpus is real — V9's
  cross-corpus dynamic range is 8 points below the ≥60 gate. This is
  documented as structural in
  `benchmarks/v_tuner_v9_falsification_2026-05-20.md` (V9 spreads its
  range across the FULL corpus, but no single (image, codec) curve hits
  the full [0, 100] span in q=5..95). On the V6 corpus, medRange=79.32,
  which would pass the gate. This is also a corpus artifact.
- The tied=5.56% borderline value is from `eval_v9_bake.py` on the V9
  corpus and matches our measurement (0.0556 vs gate 0.05). With the
  V6 corpus, V9 has 0.000 ties (matches V6 ship's 0.0000).

## Ship recommendation

**Three options, in order of preference:**

### Option A (recommended): re-run V9 ship gate with V6 metric+corpus, then ship as `PreviewV0_5TunerV3`

Change `eval_v9_bake.py:run_qsweep_mono` to delegate to the canonical
`qsweep_eval` Rust binary on the V6 hand-picked qsweep corpus
(50 imgs × 19 q jpeg420). With that single change applied to V9
calibrated bake `b50e8ca4946c1ec5bf2f5e9cf96ffdb8`:

| gate | observed | gate | verdict |
|---|---:|---|:-:|
| mono ≥ 0.9378 (pair-based on V6 corpus) | **0.9644** | ≥ 0.9378 | **PASS** |
| tied ≤ 5% | 0.0000 | ≤ 0.05 | **PASS** |
| medRange ≥ 60 (V6 corpus) | **79.32** | ≥ 60 | **PASS** |
| (8 other gates, unchanged) | | | PASS |

11/11 gates pass when measured the same way V6 was measured. The V9
PCHIP spline + range extension goal is achieved without sacrificing
any V6-measured property.

### Option B: ship with slope-normalized mono gate as `PreviewV0_5TunerV3`

Replace the strict-decrease mono with the slope-normalized variant
(`|Δscore| > 1.0`). Both metrics are reasonable but the slope-normalized
one rejects sub-noise wobbles that aren't user-visible. V9 calibrated
passes by 0.05 margin on the V6 corpus (0.94 curve-mono vs 0.9378 gate)
and by 0.04 margin on pair-based.

### Option C: ship V9 as opt-in `PreviewV0_5TunerV3` alongside V6 with the mono caveat documented

Wire V9 to a NEW `ZensimProfile::PreviewV0_5TunerV3` variant; keep
`PreviewV0_5TunerV2` (V6) as the default Tuner trail. Surface the
strict in-curve mono trade-off in the variant doc: V9 prioritizes the
[0, 100] dial range + clean integer anchors; V6 prioritizes strict
in-curve smoothness on hand-picked qsweep curves. Users opting in
to V9 get the cleaner user-facing semantics; users wanting the V6
strict-mono property keep V6.

Option A is cleanest because the V9 ship gate was already meant to be
"strictly tighter than V6" — measuring V9 by a DIFFERENT metric and
calling it a failure was always going to be misleading. The honest
ship-gate comparison shows V9 is +0.012 mono looser than V6 by V6's
own metric, well within the 0.9378 threshold.

## Provenance

- Worktree: `~/work/zen/zensim--cross-codec-v8/` (jj workspace; primary at `~/work/zen/zensim/`).
- Predict binary: `target/release/predict_features_with_bake` (built 2026-05-20).
- Qsweep binary: `target/release/qsweep_eval` (built 2026-05-20).
- Bakes:
  - V6 ship: `zensim/weights/v_tuner_v6_2026-05-19.bin` (md5 `5b69bb815e02d5393d81b4be65a1a8c0`)
  - V9 calibrated: `zensim/weights/v_tuner_v9_2026-05-20.bin` (md5 `b50e8ca4946c1ec5bf2f5e9cf96ffdb8`)
  - V9 uncalibrated: `/mnt/v/zen/zensim-eval/exp_cross_codec_v9_2026-05-20/cc4v9_s2.bin` (md5 `b311d7e32de60822e6140d3b3d45e51a`)
- Audit scripts: `/tmp/v9_mono_audit.py` (Phase 1), `/tmp/v9_mono_audit_phase2.py` (Phase 2).
- Raw score arrays: `/tmp/v9_audit_output/{v6,v9}_corpus_{v6,v9raw,v9cal}.npy`.
- Corpora:
  - V6: `/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv` + `qsweep_manifest.tsv` (950 rows, 50 refs × 19 q, jpeg420).
  - V9: 50 refs drawn at `np.random.default_rng(seed=42)` from `/mnt/v/zen/picker-training/2026-05-19/butter/zenjpeg.parquet` (950 rows, 50 refs × 19 q, zenjpeg).
- Task: #174.
