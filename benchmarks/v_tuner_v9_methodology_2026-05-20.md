# PreviewV0_5TunerV3 (V9 extended-range PCHIP-spline) methodology — 2026-05-20

**Status:** FALSIFIED on the V9 11-gate ship criterion (3 gates fail
across all K=32 seeds: mono, tied, medRange). User-facing properties
all PASS (range [0,100], JND=60 exact, JOD=30 exact). See
`benchmarks/v_tuner_v9_falsification_2026-05-20.md` for the full
falsification verdict + the staged candidate bake.

**Median candidate bake:** `zensim/weights/v_tuner_v9_2026-05-20.bin`
(md5 `b50e8ca4946c1ec5bf2f5e9cf96ffdb8`, 261,451 bytes, F32). NOT
wired into `zensim::profile` — staged for user review.

## Hypothesis (user directive 2026-05-20)

> "continue improving work so that range extends to q0 on the worst
> codec and q100 or lossless on the best, maintains a numeric-multiple
> of 10 value for jnd and also a multiple of 10 for jod, allowing input
> and output shaping and piecewise as needed"

V6 (`PreviewV0_5TunerV2`, commit `52295ed`) ships score=63 at PJND
following the CID22 paper convention. 63 is not a memorable round
number. The user-facing dial wants:

- score=60 at JND (clean multiple of 10)
- score=30 at JOD (clean multiple of 10)
- score=0 at worst-codec q=0 (range floor)
- score=100 at best-codec q=100 / lossless (range ceiling)

V9's contribution: extend the score range to a full [0, 100] span with
explicit user-facing semantic anchors, achieved via a post-network
PCHIP spline calibration layer (deterministic, monotone, fitted to
the trained network's median predictions).

## V9 design

### Anchor table (8 bands)

| butter_pnorm3 | target_score | semantic |
|---:|---:|---|
| 0.05  | 100 | lossless / q=100 best codec       |
| 0.30  |  90 | near-lossless                       |
| 0.60  |  80 | visually identical                  |
| 1.50  |  60 | **JND** (CID22 paper PJND)          |
| 2.50  |  50 | mildly noticeable                   |
| 4.00  |  30 | **JOD** (just objectionable)        |
| 7.00  |  10 | clearly distorted                   |
| 12.00 |   0 | worst-q floor                       |

Plus explicit q=5 worst-floor rows (butter ≥ 6.0, target_score=0) and
q=95 zenjxl-lossless rows (butter ≤ 0.10, target_score=100) to widen
the anchor pool at the extremes — because the butter parquets only
cover q=5..95 and don't reach butter=12.0.

Comparison with V6 / V8:

| butter | V6 | V8 | **V9** |
|---:|---:|---:|---:|
| 0.05 | — | — | **100** |
| 0.30 | 90 | — | 90 |
| 0.50 | — | 85 | — |
| 0.60 | — | — | 80 |
| 0.80 | 75 | — | — |
| 1.00 | — | 75 | — |
| 1.50 | 63 | — | **60** |
| 2.50 | 45 | 63 | 50 |
| 4.00 | 25 | 45 | **30** |
| 6.00 | 10 | — | — |
| 7.00 | — | — | 10 |
| 12.00 | — | — | **0** |

V6 had 6 bands with range [10, 90]. V8 had 4 bands with range [45, 85]
(V8 falsified for medRange collapse). V9 has 8 bands with range [0, 100]
and clean integer-multiple-of-10 JND + JOD.

### Architecture

V6 architecture (per-sample-α + tanh-pin) is preserved verbatim. V9
adds:

1. **Post-network PCHIP spline calibration** applied AFTER tanh-pin.
2. The spline is fit AFTER training:
   - Run inference on the V9 anchor parquet rows.
   - For each `target_score`, compute the MEDIAN predicted-raw.
   - Build PCHIP knots at `(median_pred, target_score)`.
   - Enforce strict monotonicity in BOTH x and y. If a band's median
     violates the canonical y-order, drop that band (network couldn't
     learn it discretely from neighbors).
   - Bake the spline knots as `zentrain.output_calibration_spline`
     metadata: `[u32 LE n_knots, n_knots × (f32 x, f32 y) LE]`.
3. The runtime reads the metadata and applies the spline at
   `apply_mlp_scoring` time. Linear extrapolation outside the knot
   range using the endpoint slope.

### Trainer changes from V6

```diff
-  --anchor-loss-weight 1.0
-  --anchor-target-score 63.0
-  --tanh-output-head-scale 15.0
-  --dynamic-range-floor-weight 0.2
-  --dynamic-range-sigma-threshold 15.0
+  --anchor-loss-weight 0.5
+  --anchor-target-score 60.0
+  --tanh-output-head-scale 20.0
+  --dynamic-range-floor-weight 0.3
+  --dynamic-range-sigma-threshold 25.0
```

Rationale:
- **anchor-loss-weight 0.5** (down from 1.0): 8 bands vs 6 means more
  anchor rows; reduce per-row pressure proportionally.
- **anchor-target-score 60** (down from 63): fallback target for
  non-targeted rows; should match the new V9 JND.
- **tanh-output-head-scale 20.0** (up from 15.0): widen active linear
  region to cover the full [0, 100] span without saturation issues.
  At scale=20, `y_pre ∈ [−60, 60]` maps to `[5, 95]` score units;
  the spline can extrapolate beyond that.
- **dynamic-range-floor-weight 0.3** + **sigma-threshold 25.0** (both
  up): encourage wider score spread to fully populate the [0, 100]
  range.

### K=32 SPEED-B path

Per `feedback_use_minibatch_32_for_aux_losses` and V6-RESHIP
verification (`benchmarks/v_tuner_v6_reship_2026-05-19.md`), K=32
lr=5.66e-3 is the verified-clean fast path. V9 uses it.

## Trainer command (per bake)

```sh
target/release/zensim_mlp_train \
    --group safesyn:/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet:1.0:0.0 \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 5.66e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 32 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 20.0 \
    --ranknet-weight 0.0 --mse-weight 1.0 \
    --monotonicity-reg 1.0 --monotonicity-margin 0.0 \
    --anchor-parquet /mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet \
    --anchor-loss-weight 0.5 --anchor-target-score 60.0 --anchor-step-p 0.30 \
    --cross-codec-eq-parquet /mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet \
    --cross-codec-eq-weight 1.0 --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.3 --dynamic-range-sigma-threshold 25.0 \
    --dynamic-range-step-p 0.05 --dynamic-range-probe-n 40 \
    --seed <S> --out cc4v9_s<S>.bin
```

Driver: `scripts/v_next/run_cross_codec_v9_seed.sh <seed>`.

## Calibration pipeline

After training each seed bake:

```sh
python3 scripts/v_next/calibrate_v9_spline.py \
    --bake cc4v9_s<S>.bin \
    --out cc4v9_s<S>_calibrated.bin \
    --spline-csv cc4v9_s<S>.spline.csv
```

The calibrator:
1. Loads the V9 anchor parquet (22,008 rows × 372 features × 8+2
   target bands).
2. Runs `predict_features_with_bake --bake-post raw` over all rows to
   get the network's predicted score (post-tanh-pin, pre-spline since
   the input bake has no spline yet).
3. Per-band MEDIAN predicted score becomes the spline knot x;
   target_score is y.
4. Drops bands whose medians violate canonical y-order (network
   inverted local ordering — happens for the worstfloor extreme
   where the network may saturate).
5. Builds PCHIP via Fritsch–Carlson; verifies monotonicity on a
   dense grid.
6. Round-trips the bake via `zenpredict inspect --weights` →
   `BakeRequestJson` → `zenpredict bake`, appending the spline
   metadata entry. Per CLAUDE.md "JSON pipeline mandate" — no
   ad-hoc ZNPR v3 emitters.

## Inputs (md5 / row count where computable)

- Train corpus: `canonical-2026-05-18/train/safesyn.parquet`
  (196,086 rows × 372 features; sha256 prefix `1ee0565fb6cb`).
- V9 anchor parquet:
  `/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet`
  (22,008 rows × 381 cols, 8 butter bands × 4 codecs × ~1000
  sources, plus explicit worstfloor + lossless rows).
- Cross-codec equivalence: same as V6/V8
  (`picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet`,
  68,788 pairs).

## Ship gate (V9-specific, extends V6 set)

V6 had 6 gates. V9 adds 5 more — range-extension and clean-anchor
accuracy — for a total of 11.

| gate | threshold | rationale |
|---|---|---|
| mono ≥ 0.9378 | post-spline | V6 floor; PCHIP guarantees monotone-preserving for monotone knots |
| tied ≤ 5% | post-spline | unchanged |
| medRange ≥ 60 | up from 50 | V9 targets the full [0, 100] range |
| **worstfloor med ≤ 5** | new | V9 wants score=0 at worst-codec q≥5 with butter≥6 |
| **lossless med ≥ 95** | new | V9 wants score=100 at zenjxl q=95 (butter≈0.005) |
| **JND abs_err ≤ 2** | new | |predicted − 60| at butter=1.5 band ≤ 2 |
| **JOD abs_err ≤ 2** | new | |predicted − 30| at butter=4.0 band ≤ 2 |
| PJND cc_std_median ≤ 5 | unchanged | cross-codec parity at JND |
| **JOD cc_std_median ≤ 5** | new | cross-codec parity at JOD |
| **T80 cc_std_median ≤ 5** | new | cross-codec parity at score=80 |
| **T90 cc_std_median ≤ 5** | new | cross-codec parity at score=90 |

A bake passes the V9 ship gate when ALL 11 sub-gates pass on the
median-of-3-seeds bake (sorted by CID22 SROCC).

## Median selection + ship decision

3 seeds sorted by CID22 SROCC ascending:

| seed | CID22 | KADID | TID | KonJND | AIC-3 | mono | tied | medRange | n_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s3 | 0.8488 | 0.5108 | 0.6364 | 0.2499 | 0.7834 | 0.660 | 0.0556 | 51.82 | 8/11 |
| **s2** | **0.8540** | **0.4832** | **0.6636** | **0.2317** | **0.7865** | **0.640** | **0.0556** | **51.87** | **8/11** |
| s1 | 0.8681 | 0.5909 | 0.6500 | 0.1596 | 0.7842 | 0.600 | 0.0567 | 52.69 | 8/11 |

**Median by CID22 SROCC: `cc4v9_s2_calibrated.bin`**.

**Ship decision: DO NOT auto-ship.** The mono+tied+medRange gates
that V6 ship guaranteed fail by structural design trade-offs
(K=32+wider tanh-scale needed to populate the [0, 100] range
amplifies in-curve dips on the qsweep proxy). The V9 user-facing
properties (range extension, clean integer JND+JOD, deterministic
post-network calibration) all pass with margin.

K=1 follow-up (seed=4, in flight) tests whether reducing K-batch
amplification closes the mono gap. If yes, V9-K1 ships in a
follow-up commit; if no, V9 remains a staged-but-unwired candidate
pending user direction.

## Predecessor docs

- V6 ship: `benchmarks/v_tuner_v6_methodology_2026-05-19.md`
- V6 reship K=32: `benchmarks/v_tuner_v6_reship_2026-05-19.md`
- V7 (empirical anchors): `benchmarks/v_tuner_v7_methodology_2026-05-19.md`
- V8 (4-band narrow): `benchmarks/v_tuner_v8_methodology_2026-05-19.md` (falsified)
- V9 anchor design: `benchmarks/v_tuner_v9_anchor_design_2026-05-20.md`
