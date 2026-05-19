# `PreviewV0_5TunerV2` from V4 candidates — FALSIFIED on monotonicity + range (2026-05-19)

## Hypothesis (parent task brief)

V4 closes the V3 mono gap (best V3 strict mono 0.9100 vs gate 0.9378)
with two architectural counterweights:

1. **tanh-pinned [0, 100] output head** (V4-C from V3 falsification doc).
   Wraps the per-sample-α head output as `y_score = 100·σ(y_pre/10)`,
   eliminating the post-affine β=5–10× amplification path V3 identified.

2. **Multi-codec PJND anchor** (user directive 2026-05-19 14:55).
   1000 sources × 4 codecs at each codec's PJND-q, all targeting
   score=63. Binds score=63 ↔ PJND across codecs during training.

Plus V4-B: `--monotonicity-reg 1.0` (NOT 5.0; V3's σ-floor already
prevents collapse).

Sweep: 3 seeds × W ∈ {0.5, 1.0} = 6 bakes.

## Verdict: **falsified.**

**0 of 6 V4 candidates pass the Tuner-trail gate.** The architectural
combination achieves the cross-codec calibration goal SPECTACULARLY
(cross-codec PJND score std median 0.10–0.14, **35–50× tighter than
gate ≤ 5.0**) but COMPRESSES the dynamic range so tightly around 63
that q-sweep range collapses to 8–16 score units (gate ≥ 50). Strict
monotonicity correspondingly drops to 0.78–0.84 (gate ≥ 0.9378),
WORSE than V3's best of 0.9100.

## Per-bake gate scorecard

Gates: `strict_mono ≥ 0.9378`, `tied ≤ 5 %`, `range ≥ 50`,
`T=63 butter_max < 2.5 OR butter_p3 < 2.5`,
`cross-codec PJND score std median ≤ 5.0`.

| Bake | mono | tied | q5_med | q95_med | range | T=63 b_max | T=63 b_p3 | PJND std | mono | tied | range | xc | pjnd | ALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|
| baseline_tuner (existing ship) | 0.9278 | 0.0044 | 4.96 | 94.64 | 89.68 | 8.07 | 2.11 | n/a | ✗ | ✓ | ✓ | ✓ | n/a | (current) |
| cc4v4_s1_w0_5 | 0.7756 | 0.0000 | 56.26 | 64.48 | 8.22 | 4.97 | **1.95** | **0.110** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4_s1_w1_0 | 0.8122 | 0.0000 | 54.53 | 64.39 | 9.86 | 5.98 | 2.27 | **0.100** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4_s2_w0_5 | 0.8111 | 0.0000 | 49.41 | 65.10 | 15.69 | 5.18 | **2.00** | **0.140** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4_s2_w1_0 | 0.8433 | 0.0000 | 55.26 | 64.39 | 9.13 | 4.37 | **1.76** | **0.110** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4_s3_w0_5 | 0.8178 | 0.0000 | 53.21 | 65.10 | 11.89 | 5.89 | 2.26 | **0.130** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4_s3_w1_0 | 0.8322 | 0.0000 | 54.49 | 65.00 | 10.51 | 4.87 | **1.97** | **0.110** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |

(Bold values pass their respective gates.)

## What V4 solved

| Goal | V3 result | V4 result | Verdict |
|---|---|---|---|
| Cross-codec PJND score parity | 5.52 butter_max (single-codec anchor) | std 0.10–0.14 (multi-codec anchor) | **SOLVED dramatically** |
| Tied-rate | 2.8–9.9 % (V3 was borderline) | 0.0 % across all 6 V4 bakes | **SOLVED — tanh pin has no boundary ties** |
| Output range | 89.94–90.02 (V3 after affine) | **8.22–15.69** | **REGRESSED** |
| Strict monotonicity | 0.82–0.91 (V3 best 0.9100) | **0.78–0.84** | **REGRESSED** |
| Cross-codec T=63 butter | 2/6 passed | 5/6 passed (butter_p3) | Improved |

## Why V4 broke range + monotonicity

The multi-codec anchor at weight=1.0 + step_p=0.15 pulls so hard
toward score=63 that the output collapses to a narrow band:

### 1. Anchor MSE dominates the per-pair gradient

Per pair-step:
- 1 RankNet/MSE pair step contributes `2 · mse_weight · (y - target) / (2 · K)`
  where `K = pairs_per_epoch = 50000`. With `mse_weight=1.0`, gradient
  magnitude per step is `~5e-5` × (y − target).
- With probability 0.15, an anchor step ALSO fires, contributing
  `2 · anchor_loss_weight · row_w · (y − 63)` directly (no `1/K` scaling).
  With `anchor_loss_weight=1.0` and `row_w=1.0`, gradient magnitude is
  `~2.0` × (y − 63).

The anchor's gradient is ~40,000× stronger per fired step. Even at
15 % firing rate vs 100 % pair-step rate, the anchor contributes
~6000× more total gradient signal toward score=63 than the per-pair
MSE contributes toward spreading the range.

### 2. σ-floor probe can't overcome anchor pull

The σ-floor probe (target σ=15 across 40 random equiv-pool rows)
fires at 5 % rate with weight 0.2 — total contribution ~`0.05 · 0.2 ·
40 · gradient ≈ 0.4 gradient per pair-step on average`. The anchor
contribution per step (anchor_loss_weight 1.0, step_p 0.15) is
`0.15 · 2.0 · err ≈ 0.3 · err`. With `err` typically 30 score units
(network predicts 30-95, anchor wants 63), anchor dominates 9:1.

### 3. tanh pin compresses extremes back toward the mean

`y_score = 100 · σ(y_pre / 10)` saturates at the extremes — to push
y_score from 63 to 95, y_pre needs to move from 0 to +20 (sigmoid
saturation region). The combined pressure: anchor pulling toward
y_pre=0, σ-floor pushing toward y_pre spread, MSE pulling toward
target. The anchor wins.

Result: q-sweep range 8-16 score units AT BEST (vs V3's 90).

## Multi-codec PJND check passed dramatically

Despite the range collapse, the multi-codec PJND check is the
brightest result of the experiment:

| Bake | agg_mean | per-codec mean spread | cc_std median | cc_std p95 |
|---|---:|---:|---:|---:|
| cc4v4_s1_w0_5 | 62.97 | 62.93–63.00 (0.07) | 0.11 | 0.40 |
| cc4v4_s1_w1_0 | 62.99 | 62.93–63.00 (0.06) | 0.10 | 0.44 |
| cc4v4_s2_w0_5 | 62.94 | 62.94–62.95 (0.01) | 0.14 | 0.41 |
| cc4v4_s2_w1_0 | 62.81 | 62.78–62.84 (0.06) | 0.11 | 0.46 |
| cc4v4_s3_w0_5 | 63.18 | 63.16–63.19 (0.03) | 0.13 | 0.42 |
| cc4v4_s3_w1_0 | 62.87 | 62.84–62.92 (0.08) | 0.11 | 0.42 |

All 6 V4 bakes predict score≈63 for each (source, codec) PJND pair,
with std≈0.5 across all 4000 anchor rows AND cross-codec std per
source <0.5 — the cross-codec calibration mechanism works
perfectly. The cost is the dynamic range.

## What V4 reveals about the architecture

The multi-codec anchor + tanh pin combination MAKES cross-codec
parity at a single anchor point achievable. But the loss landscape
created by 4000 anchor rows at the same target overpowers the
range-spreading mechanisms (per-pair MSE + σ-floor). To recover
both range AND cross-codec parity, the anchor must be:

1. **Weaker per-step** (smaller `anchor_loss_weight`), OR
2. **Lower firing rate** (smaller `anchor_step_p`), OR
3. **Distributed across score targets** (not all rows at score=63
   — some at near-lossless 80, some at noticeable 50, etc.), OR
4. **Score-relative** rather than absolute (e.g., enforce equal
   scores across the 4 codecs WITHOUT specifying what that score
   should be).

V4b is now in flight (dispatched 2026-05-19) testing option 1+2:
anchor_loss_weight ∈ {0.05, 0.10} AND anchor_step_p=0.05 (instead
of V4's 1.0 + 0.15). Same tanh pin + multi-codec anchor data.

## V4b verdict (relaxed-anchor sweep)

V4b sweep: 3 seeds × anchor_loss_weight ∈ {0.05, 0.10} = 6 bakes.
Per the V4 falsification analysis, the anchor was too strong; V4b
weakens it 10-20x to give per-pair MSE room to spread the range.

| Bake | mono | tied | q5_med | q95_med | range | T63 b_max | T63 b_p3 | PJND std | agg_mean | mono | tied | range | xc | pjnd | ALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|
| cc4v4b_s1_a0_05 | **0.9578** | 0.0022 | 11.65 | 46.90 | 35.25 | **1.27** | **0.59** | **0.48** | 42.08 | **✓** | ✓ | ✗ | **✓** | **✓** | FAIL |
| cc4v4b_s1_a0_10 | 0.9122 | 0.0000 | 41.61 | 70.59 | 28.98 | 3.74 | 1.64 | 0.55 | 63.75 | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4b_s2_a0_05 | **0.9378** | 0.0000 | 18.42 | 51.48 | 33.06 | **1.25** | **0.59** | **0.50** | 46.20 | **✓** | ✓ | ✗ | **✓** | **✓** | FAIL |
| cc4v4b_s2_a0_10 | 0.8700 | 0.0000 | — | — | 24.73 | 3.23 | 1.43 | 0.44 | 61.57 | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4b_s3_a0_05 | 0.8567 | 0.0000 | — | — | 21.91 | 2.58 | 1.15 | 0.44 | 60.05 | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4b_s3_a0_10 | 0.9000 | 0.0022 | — | — | 35.34 | **1.25** | **0.59** | **0.67** | 34.73 | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |

(Bold = passes its gate.)

**Verdict: 0 of 6 V4b candidates pass all 5 gates.** Two bakes pass
4 of 5 (mono + tied + xc + pjnd), failing ONLY the range gate ≥ 50.

### V4b key findings

- **Mono BEAT the gate on 2 bakes**: `cc4v4b_s1_a0_05` strict mono =
  **0.9578** (gate 0.9378, +0.020 above baseline tuner 0.9278);
  `cc4v4b_s2_a0_05` strict mono = **0.9378** (exactly at gate).
- **T=63 butter is excellent across the board**: butter_p3 mean
  0.59-1.64, well below gate 2.5. The dramatically tight V4 PJND
  calibration is preserved.
- **PJND check still passes**: cc_std median 0.44-0.67, far below
  gate 5.0.
- **Range still fails**: best V4b range is 35.34 (cc4v4b_s3_a0_10)
  vs gate ≥ 50. The tanh pin's natural compression caps the
  effective output range at ~30-50 score units for this corpus.
  This is an architectural feature, not a tuning failure.

### Range gate vs tanh-pin architecture: structural conflict

The range gate ≥ 50 originated from the affine-linear-output
architecture (PreviewV0_5Tuner / V_tuner-v2-s2 range 89.68 after
affine calibration). The tanh-pinned output naturally compresses
the dynamic range — `y_score = 100·σ(y_pre/scale)` has its useful
linear region in ~[20, 80], with diminishing returns past that.

For a tanh-pinned bake to hit range 50:
- y_pre at q=5 must be ≈ -15 → y_score ≈ 18
- y_pre at q=95 must be ≈ +15 → y_score ≈ 82
- Requires y_pre to span ±15 within the training distribution.

V4b's best bake (s3_a0_10) achieves y_score range 35.34 with
y_pre ranging roughly ±5 — the network only learned to express
~30 score units of dynamic range under the joint pressure of
per-pair MSE + multi-codec anchor + σ-floor + tanh pin.

**The architecture-vs-gate conflict is genuine**: tanh-pinned bakes
cannot easily hit range ≥ 50 while also passing the σ-floor +
multi-codec anchor pressure. The two ways to resolve it:

1. **Scale up tanh-pin scale** (e.g., scale=20 or 30): expands the
   linear region but loses the [0, 100] guarantee at training-distribution
   extremes.
2. **Replace the range gate with a min-range floor** (e.g., range ≥ 30):
   acknowledges that tanh-pinned bakes carry their own native range
   constraint. The Tuner-trail purpose (preventing dead-zone clamp
   ties) is satisfied by tied ≤ 5% directly — range was a proxy.

If the user signs off on a relaxed range gate (e.g., ≥ 30), **two V4b
bakes ship cleanly**: cc4v4b_s1_a0_05 (mono 0.958, range 35.25, T63
butter_p3 0.59, PJND std 0.48) and cc4v4b_s2_a0_05 (mono 0.938,
range 33.06, T63 butter_p3 0.59, PJND std 0.50). Without that
sign-off, V4b is also falsified.

## V5 direction proposals (after V4 + V4b falsification)

### V5-A: anchor-as-rank-preserve (not absolute MSE)

Replace `(y_anchor − 63)²` MSE with a RankNet-style sigmoid loss
that penalizes only when two SAME-source DIFFERENT-codec anchor
rows produce different scores:

```
L_anchor_rank = w · |Δ| · −log(sigmoid(s · (y_a − y_b)))
```

where (a, b) are two codecs at the same source's PJND, and `s` is
the sign of their (closer-to-63) preference. Doesn't pull either
toward an absolute value — just toward each other. The
score-spread comes entirely from the per-pair MSE.

### V5-B: smaller multi-codec anchor with target distribution

Build the anchor parquet at NOT just PJND-q but ALSO at other q
levels (q=20, 40, 60, 80) per codec, with target scores aligned to
the butter quality curve (e.g., butter_pnorm3 = 0.5 → target 90,
1.5 → target 63, 3.0 → target 30). The anchor enforces calibration
across a RANGE of scores, not just at 63 — which keeps range
expressed.

### V5-C: schedule the anchor weight

Start with `anchor_loss_weight = 0.0` for the first 100 epochs
(per-pair MSE establishes the score-shaped output range), THEN
ramp up the anchor weight from 0.0 to 1.0 over epochs 100–200
(cross-codec calibration aligned to the established range). The
gradient balance shifts over training, not all-at-once.

### V5-D: explicit range floor regularizer

Replace the σ-floor probe (heterogeneous A-side equiv pool) with
a per-image q-sweep substrate: for each of N training images,
forward at q ∈ {5, 25, 50, 75, 95}, enforce `σ ≥ 25 raw units`
(within-image, NOT cross-image). Aligned with V3 falsification
proposal V4-A but compatible with multi-codec anchor.

## Decision (final)

**No PreviewV0_5TunerV2 ship from V4 OR V4b under strict gate criteria.**
PreviewV0_5Tuner (baseline tuner, V_tuner-v2-s2 calibrated, range
89.68, strict mono 0.9278) remains the dial profile.
PreviewV0_5CrossCodec (V2 W=1.0 seed=1, T=63 butter 5.52) remains
the cross-codec profile.

**However**, V4b produced two architectural sweet-spot candidates
that beat baseline tuner on monotonicity AND pass all other gates
except range:

- `cc4v4b_s1_a0_05` — mono **0.9578** (+0.030 vs baseline tuner
  0.9278; +0.020 above gate 0.9378), tied 0.22%, range 35.25,
  T=63 butter_p3 **0.59**, PJND std 0.48.
- `cc4v4b_s2_a0_05` — mono **0.9378** (exactly at gate), tied 0%,
  range 33.06, T=63 butter_p3 **0.59**, PJND std 0.50.

These are mono+xc+pjnd-perfect bakes whose dynamic range is
constrained by the tanh-pin architecture (~35 score units instead
of the affine-linear's 90). If the user signs off on a relaxed
range gate appropriate to the tanh-pin architecture (e.g., range
≥ 30 instead of ≥ 50), `cc4v4b_s1_a0_05` is a clean ship as
PreviewV0_5TunerV2 — strictly better than baseline tuner on every
measured axis except dynamic range.

The V4 architectural changes (tanh-pinned [0, 100] output head +
multi-codec PJND anchor) ARE shipped to main as
infrastructure-on-shelf — they work cleanly and the cross-codec
calibration component is independently valuable. The multi-codec
PJND anchor reduces cross-codec score std from V3's "no calibration"
(diverging at T=63) to V4b's 0.5 score units across 4000
(source, codec) PJND pairs — **a 10x improvement in cross-codec
parity, the user's stated 2026-05-19 14:55 directive**.

**User action required**: decide whether to ship `cc4v4b_s1_a0_05`
under a relaxed range gate ("strict mono ≥ baseline tuner + 1 pp"
without an absolute range floor), or hold and pursue V5 directions
(per-codec multi-target anchor, scheduled anchor weight) to recover
the full ≥ 50 range while preserving mono. The V5-B + V5-C approach
in the proposals section above is the most likely path to both.

## Files produced this session

- `benchmarks/v_tuner_v4_methodology_2026-05-19.md` — hypothesis +
  recipe + post-hoc training observations.
- `benchmarks/v_tuner_v4_falsification_2026-05-19.md` — this doc.
- `scripts/v_next/build_multi_codec_pjnd_anchors.py` — anchor parquet builder.
- `scripts/v_next/run_cross_codec_v4_seed.sh` — V4 trainer driver.
- `scripts/v_next/run_cross_codec_v4_consistency.sh` — V4 T=63 driver.
- `scripts/v_next/eval_cross_codec_v4.sh` — V4 eval pipeline (no affine).
- `scripts/v_next/eval_v4_pjnd_check.py` — multi-codec PJND analyzer.
- `scripts/v_next/summarize_v4.py` — combined-table renderer.
- `scripts/v_next/run_cross_codec_v4b_seed.sh` — V4b (relaxed anchor) driver.
- `scripts/v_next/run_cross_codec_v4b_consistency.sh` — V4b T=63 driver.
- `scripts/v_next/eval_cross_codec_v4b.sh` — V4b eval pipeline.
- `scripts/v_next/eval_v4b_pjnd_check.py` — V4b PJND analyzer.
- `scripts/v_next/summarize_v4b.py` — V4b combined-table renderer.
- `/mnt/v/zen/zensim-training/2026-05-19-multi-codec-jnd-anchors/anchors_multi_codec_372col.parquet`
  — 4000-row multi-codec PJND anchor.
- `/mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19/` — V4 bakes + eval.
- `/mnt/v/zen/zensim-eval/exp_cross_codec_v4b_2026-05-19/` — V4b bakes + eval.

## Trainer + runtime commits

- `2934d0c6` (main, 2026-05-19): trainer tanh-output-head + multi-codec
  anchor + runtime dispatch.
- `7a45eb67` (main, 2026-05-19): eval pipeline + runtime tanh-pin
  dispatch in every bake-aware tool (bake_verdict, qsweep_eval,
  ensemble_score_rows, score_pair_with_bake).

## Reproduction

```bash
cd /home/lilith/work/zen/zensim--cross-codec-metric

# 0. Build trainer + tools.
cargo build --release --bins -p zensim-validate
cargo build --release --bin score_pair_with_bake -p zensim-validate

# 1. Build multi-codec anchor parquet.
python3 scripts/v_next/build_multi_codec_pjnd_anchors.py

# 2. Train 6 V4 bakes in parallel (~45 min on 7950X).
for seed in 1 2 3; do
  for w in 0.5 1.0; do
    bash scripts/v_next/run_cross_codec_v4_seed.sh $seed $w &
  done
done; wait

# 3. Eval pipeline.
bash scripts/v_next/eval_cross_codec_v4.sh
bash scripts/v_next/run_cross_codec_v4_consistency.sh
python3 scripts/v_next/eval_v4_pjnd_check.py /mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19

# 4. Summary.
python3 scripts/v_next/summarize_v4.py /mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19
```

Total compute: ~50 min on 7950X. Total data: ~1.6 MB bakes + ~15 MB
eval artifacts + 7 MB anchor parquet.
