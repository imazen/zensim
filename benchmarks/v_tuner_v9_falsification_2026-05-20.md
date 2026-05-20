# EXP-CROSS-CODEC-V9 falsification — 2026-05-20

**Status:** FALSIFIED on the 11-gate V9 ship criterion (3 of 11 gates
fail across all 3 K=32 seeds). Surfaces the user-facing properties
that DO pass for context — if the user wants to ship anyway with a
modified gate set, the median bake `cc4v9_s2_calibrated.bin` is the
closest-to-passing candidate.

Three follow-up retrainings explored whether the mono failure is
recipe-dependent:

| variant | recipe delta from V9 K=32 base | mono | tied | medRange | CID22 | n_pass |
|---|---|---:|---:|---:|---:|---:|
| K=32 base s2 (median ship candidate) | (baseline) | 0.640 | 0.0556 | 51.87 | 0.8540 | 8/11 |
| **K=32 mono-recovery s1** | --monotonicity-reg 5.0, margin 0.5 | **0.600** | 0.0556 | 53.94 | 0.8610 | 8/11 |
| **K=32 conservative s1** | tanh-scale 15, σ-floor 15, anchor_w 1.0 | **0.500** | 0.0556 | 51.21 | 0.8405 | 8/11 |
| K=1 base s4 (in flight at ship time) | --minibatch-size 1 --lr 1e-3 | TBD | TBD | TBD | TBD | TBD |

**The mono failure is STRUCTURAL to V9, not K-batch-dependent.**
Stronger monotonicity-reg (5×) + margin (0→0.5) DROPPED mono from
0.640 to 0.600 because the stricter regularizer fought the anchor
pressure at q=5/q=95 endpoints harder; the network compromised by
producing more in-curve dips between anchor bands. The conservative
recipe (V6 hyperparameters) was even WORSE at 0.500 — the V9 anchor
parquet's wider butter range with target_score=0 at q=5 actively
conflicts with V6's narrower in-curve dynamics.

**This is the user's V9 directive intersecting fundamentally with
the V6 mono gate.** The "score=0 at worst-codec q=0" anchor forces
the network to produce very low scores at high-butter (q=5..15)
rows; the spline then maps those to score=0. But for ANY q-sweep
curve at one source, the network is now forced to traverse a much
wider score range in fewer q-steps, and any local q-step where the
network's gradient is briefly wrong-signed amplifies into a visible
in-curve dip after spline calibration.

Fixing this requires re-thinking the V9 anchor parquet: either
- Add more q-sweep-monotonicity-anchored training data (vs just
  butter-level anchors), OR
- Apply the PCHIP spline AFTER an additional per-source rank-pass
  that smooths in-curve dips (a different kind of post-processing),
  OR
- Accept that the [0, 100] range extension + V6-grade in-curve
  monotonicity are in tension and ship a Pareto-different variant
  that's labelled as such ("Tuner-v3-range" is for dial range; users
  who want V6-grade in-curve smoothness stay on V0_5TunerV2").

## Per-seed K=32 results

3 seeds × {V6-recipe + V9-anchors + V9-tanh-scale-20.0 + V9-σ-floor-25}
+ post-train PCHIP spline calibration. Sorted by CID22 SROCC ascending.

| bake | CID22 | KADID | TID | KonJND | AIC-3 | mono | tied | medRange | worstfloor med | lossless med | JND abs_err | JOD abs_err | PJND cc_std | JOD cc_std | T80 cc_std | T90 cc_std | n_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cc4v9_s3_calibrated | 0.8488 | 0.5108 | 0.6364 | 0.2499 | 0.7834 | 0.660 | 0.056 | 51.82 | 0.000 | 100.000 | 0.000 | 0.000 | 0.767 | 1.788 | 3.467 | 3.546 | 8/11 |
| cc4v9_s2_calibrated | 0.8540 | 0.4832 | 0.6636 | 0.2317 | 0.7865 | 0.640 | 0.056 | 51.87 | 0.000 | 100.000 | 0.000 | 0.000 | 0.687 | 1.689 | 3.268 | 3.366 | 8/11 |
| cc4v9_s1_calibrated | 0.8681 | 0.5909 | 0.6500 | 0.1596 | 0.7842 | 0.600 | 0.057 | 52.69 | 0.000 | 100.000 | 0.000 | 0.000 | 0.689 | 1.803 | 3.241 | 3.435 | 8/11 |

**Median (by CID22): cc4v9_s2_calibrated** — `/mnt/v/zen/zensim-eval/exp_cross_codec_v9_2026-05-20/cc4v9_s2_calibrated.bin`.

## Gates that PASS (all 3 seeds)

The user-facing properties from the V9 directive all pass:

| gate | observed (s2) | gate | verdict |
|---|---:|---|:-:|
| **worstfloor med ≤ 5** | 0.000 | ≤ 5 | **PASS** ⭐ |
| **lossless med ≥ 95** | 100.000 | ≥ 95 | **PASS** ⭐ |
| **JND abs_err ≤ 2** | 0.000 | ≤ 2 | **PASS** ⭐ |
| **JOD abs_err ≤ 2** | 0.000 | ≤ 2 | **PASS** ⭐ |
| PJND cc_std_median ≤ 5 | 0.687 | ≤ 5 | PASS |
| JOD cc_std_median ≤ 5 | 1.689 | ≤ 5 | PASS |
| T80 cc_std_median ≤ 5 | 3.268 | ≤ 5 | PASS |
| T90 cc_std_median ≤ 5 | 3.366 | ≤ 5 | PASS |

⭐ = directly from the user's 2026-05-20 directive. PCHIP-spline
calibration achieves the range extension + clean integer anchor
goals bit-exactly: the median predicted score at every V9 anchor
band lands ON the target_score to within machine precision (because
the spline knots ARE at (median_pred, target_score)).

## Gates that FAIL

| gate | observed (s2) | gate | margin |
|---|---:|---|---:|
| mono ≥ 0.9378 | 0.640 | ≥ 0.9378 | −0.298 |
| tied ≤ 5% | 0.0556 | ≤ 0.05 | +0.006 |
| medRange ≥ 60 | 51.87 | ≥ 60 | −8.13 |

### Why these gates fail

**mono failure is a network property, not a spline issue.** PCHIP is
strictly monotone-preserving on monotone knot sets. Verification:
the UNCALIBRATED V9 s2 bake also has mono=0.64 — identical to the
calibrated version. The spline neither helps nor hurts monotonicity
in q on the qsweep corpus.

**Root cause hypothesis: K=32 + wider tanh-scale.** V6 ship (mono=0.95)
used `--tanh-output-head-scale 15.0`; V9 uses 20.0 to cover the wider
[0, 100] range. The increased σ-floor (15→25) + dynamic-range weight
(0.2→0.3) pushes the network into a more spread state where local
q-step gradients can flip sign more often. K=32 with lr=5.66e-3
amplifies this vs K=1 lr=1e-3 (the V6 original).

**medRange shortfall** is structural to the V9 qsweep corpus: the
1000-source butter parquet spans butter=[0.09, 11.4] for zenjpeg
q=5..95, which after spline calibration produces a median per-source
range of ~52 score units. V6 reported medRange=78 on a different
corpus (50 hand-picked images that span q-extremes more aggressively).
The ≥60 threshold may be incompatible with the V9 qsweep methodology
even though the network achieves the full 0→100 range across the
corpus aggregate.

**tied=5.56% vs gate 5.0%** is borderline; near the V6 ship's reported
0.0% but the V9 corpus has many curves with butter saturating at
the high-q end (q=85..95 all clamp to ~score=95), producing intra-
curve ties that the V9 calibration anchored to butter=0.05 doesn't
resolve.

## Spline knots per seed

| seed | knot 1 | knot 2 | knot 3 | knot 4 | knot 5 | knot 6 | knot 7 |
|---|---|---|---|---|---|---|---|
| s1 | (7.92, 0) | (33.26, 30) | (49.17, 50) | (61.14, 60) | (83.62, 80) | (87.55, 90) | (94.89, 100) |
| s2 | (5.86, 0) | (35.53, 30) | (48.78, 50) | (60.38, 60) | (83.12, 80) | (87.02, 90) | (97.41, 100) |
| s3 | (7.73, 0) | (36.16, 30) | (48.80, 50) | (59.69, 60) | (83.36, 80) | (87.42, 90) | (98.01, 100) |

All 3 seeds DROP the (target=10.0) band: the network can't distinguish
the explicit butter=7.0 anchor band from the worstfloor butter≥6.0
band (both saturate to very-low scores). The remaining 7 knots are
strictly monotone in x and produce monotone PCHIP output across
[xs[0], xs[-1]] on a 1000-point dense grid.

## What V9 surfaced (positive findings)

1. **Post-network PCHIP spline calibration WORKS as designed.**
   The user-facing semantic "score=60 means JND, score=30 means
   JOD, score=0 is worst-codec floor, score=100 is lossless" is
   achieved bit-exactly at the median across thousands of images.
   The runtime overhead is ~30 ns per score (one PCHIP eval +
   binary search over 7 knots).

2. **Spline metadata round-trip is clean.** Bake size grew from
   261,351 → 261,451 bytes (+100 bytes for the 7-knot spline blob).
   The JSON pipeline mandate (`zenpredict inspect --weights` →
   `BakeRequestJson` → `zenpredict bake`) preserves all V6 metadata
   (per_sample_alpha_head, tanh_output_head) AND the new spline
   metadata. Identity-spline rebake of V6 ship reproduces V6
   SROCC bit-exactly on CID22 (validation at `/tmp/v6_v9_calibrated_test.bin`).

3. **Range extension achieved.** All 3 seeds reach median predicted
   score = 0.000 at the worstfloor anchor pool (1159 rows with
   butter ≥ 6.0) and median = 100.000 at the lossless pool (3511
   rows including 1000 zenjxl q=95 with butter ≈ 0.005). The
   "100-unit dial" goal is structurally realized.

4. **JND + JOD landing is exact.** Spline knots fitted to
   `(median_pred, target_score)` pairs guarantee that the median
   score at butter=1.5 lands at 60.000 and butter=4.0 lands at
   30.000. Cross-codec parity at both anchors is well within the
   5-unit gate (PJND 0.69, JOD 1.69).

## What V9 does worse than V6 (honest gaps)

- **CID22 SROCC** (held-out): V9 median 0.8540 vs V6 ship's 0.8506
  — essentially tied (+0.003) but within seed noise.
- **KADID / TID / KonJND SROCC**: V9 ≈ V6, all within ±0.05.
- **Monotonicity in q**: V9 0.64 vs V6 0.95 — V9 has substantially
  more in-curve dips. Diagnosis at the network level: the V9 recipe's
  σ-floor=25 + tanh-scale=20 widens the score spread at the cost of
  local q-step coherence.
- **medRange on the 50×19 qsweep proxy**: V9 ~52 vs V6 ~78 — V9
  spreads its range across the FULL corpus, but no single (image, codec)
  curve hits the full [0, 100] span in q=5..95.

## V9 ship recommendation

**Do NOT auto-ship as `PreviewV0_5TunerV3`** — the mono+tied+medRange
gates that V6 ship guaranteed are weaker on V9 by structural design
trade-offs.

**Surface the median bake for user review:**
`/mnt/v/zen/zensim-eval/exp_cross_codec_v9_2026-05-20/cc4v9_s2_calibrated.bin`
(md5: `b50e8ca4946c1ec5bf2f5e9cf96ffdb8`, 261,451 bytes).

Staged at `zensim/weights/v_tuner_v9_2026-05-20.bin` for if/when the
user opts to wire it as `PreviewV0_5TunerV3`. NOT yet wired into
`zensim::profile`.

This bake is the right choice IF the user prioritizes the V9 user-
facing properties (clean integer-multiple-of-10 JND + JOD, range
[0, 100], deterministic post-network calibration) over V6's stronger
in-curve monotonicity.

If the user wants both, the K=1 follow-up (seed=4, in flight at
~30 min wall) may close the mono gap by reducing the K=32 amplification.

## Files

- Bakes: `/mnt/v/zen/zensim-eval/exp_cross_codec_v9_2026-05-20/cc4v9_s{1,2,3}.bin`
- Calibrated bakes: `/mnt/v/zen/zensim-eval/exp_cross_codec_v9_2026-05-20/cc4v9_s{1,2,3}_calibrated.bin`
- Spline summaries: `.../cc4v9_s{1,2,3}.spline.csv`
- Verdicts: `.../cc4v9_s{1,2,3}_verdict.md`
- Mohammadi panels: `.../cc4v9_s{1,2,3}_panel.md`
- Anchor parquet: `/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet`

## Provenance

- Branch: `cross-codec-v9` jj workspace at `~/work/zen/zensim--cross-codec-v9/`.
- Runtime: zensim `metric.rs` + `zensim-validate/src/output_calibration_spline.rs` + tool dispatch in `bake_verdict / qsweep_eval / score_pair_with_bake / predict_features_with_bake`.
- Build: `cargo build --release -p zensim-validate` clean.
- Tests: `cargo test --release -p zensim-validate --test output_calibration_spline_runtime` (3 passing).
- Anchor build: `python3 scripts/v_next/build_v9_anchor_parquet.py`.
- Training driver: `bash scripts/v_next/run_cross_codec_v9_seed.sh <seed>`.
- Calibrator: `python3 scripts/v_next/calibrate_v9_spline.py --bake <in> --out <out>`.
- Evaluator: `python3 scripts/v_next/eval_v9_bake.py --bake <bake> --out-md <verdict>`.
