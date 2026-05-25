# Codec-Target Metric Goals

The zensim codec-target metric is a **user-facing quality dial**.
Users type a target score; the codec hits it. Every goal below
derives from that use case. Rank correlation with human MOS
(SROCC) is a means, not an end — a metric can have perfect SROCC
and still be a broken dial (clamped range, non-monotone, codec-
dependent).

## G1 — Full-dial dynamic range

The score distribution on a representative multi-codec corpus
must span the usable dial:

| Measure | Threshold |
|---|---|
| Score p5 (across the corpus) | ≤ 25 |
| Score p95 | ≥ 85 |
| No flat zone wider than 5 score units | Verified by: no 5-unit bin with > 20% of all rows |

**Why:** v10 was clamped at ~55 for everything below butter=3.
A codec binary-searching for "score=30" got stuck. v11 fixed this
with tanh_scale=30; future bakes must not regress it.

**Trainer lever:** `--dynamic-range-floor-weight`,
`--dynamic-range-sigma-threshold`, `--tanh-output-head-scale`.

## G2 — JND semantic anchor

The visually-lossless threshold (KonJND PJND mean) must land at
a declared integer:

| Measure | Threshold |
|---|---|
| Mean score at KonJND PJND pairs | 60 ± 5 |
| Std across KonJND refs at PJND | ≤ 10 |

**Why:** "score 60 = just-noticeable difference" is the user-
facing semantic contract. Codecs targeting "visually lossless"
binary-search to score ≥ 60.

**Trainer lever:** anchor loss at score 60
(`--anchor-parquet`, `--anchor-loss-weight`), konjnd aggregation
head (`--konjnd-aggregation-weight`).

## G3 — Strict monotonicity

Higher codec quality must produce higher scores. Measured on a
50-image × 19-q JPEG sweep (q5 → q100):

| Measure | Threshold |
|---|---|
| Strict monotonicity rate | ≥ 93% |
| Tied-pair rate | ≤ 5% |

**Why:** a codec binary-search that encounters a reversal
(score(q=80) < score(q=75)) oscillates or picks the wrong q.

**Trainer lever:** `--monotonicity-reg`,
`--monotonicity-margin`.

## G4 — Cross-codec equivalence

The same perceptual quality across codecs must produce the same
score. Measured on matched-quality (butter ≤ 2.5) pairs across
≥ 3 codec families (JPEG, WebP, JXL) on 10+ images:

| Measure | Threshold |
|---|---|
| p50 \|Δscore\| across codecs at matched quality | ≤ 1.5 |
| \|Δscore\| / dial span | ≤ 2.5% |

**Why:** "score 70 from JPEG" should mean the same as "score 70
from JXL." A picker that routes between codecs based on score
needs this.

**Trainer lever:** `--cross-codec-eq-weight`,
`--cross-codec-rank-preserve-weight`.

## G5 — KonJND rank fidelity

The metric must rank visually-lossless pairs correctly:

| Measure | Threshold |
|---|---|
| KonJND-1k validation SROCC | ≥ 0.70 |
| KonJND-1k PWRC | ≥ 0.65 |

**Why:** KonJND is the only corpus measuring PJND thresholds.
v11 ship scores 0.285 (broken); YJ-AT retrain scores 0.666
(better). Target 0.70 is the floor for "dial works at the
lossless boundary." 0.85 is aspirational.

**Trainer lever:** `--konjnd-aggregation-weight`,
`--konjnd-aggregation-step-p`.

## G6 — Band coverage vs ssim2

No 10-band bin (B0..B9, width-10 on the 0-100 scale) where the
metric loses to ssim2 by more than 0.10 SROCC on any held-out
corpus:

| Measure | Threshold |
|---|---|
| max(ssim2_srocc − zensim_srocc) across (corpus, band) | ≤ 0.10 |

**Why:** the original training goal (CLAUDE.md 2026-05-10) was
"match-or-exceed fast-ssim2 across all quality bands." A metric
that wins aggregate but loses B1 by 0.15 creates a dead zone.

**Trainer lever:** per-band weighting (not yet implemented —
the trainer applies uniform loss across the score range). Future
work: stratified sampling or per-band loss weighting.

## G7 — Compression-corpus rank (advisory)

CID22 aggregate SROCC is the gold-standard generalization check.
This goal is **advisory, not blocking** (per CLAUDE.md 2026-05-14
ship policy):

| Measure | Threshold |
|---|---|
| CID22 aggregate SROCC | ≥ 0.85 (advisory) |
| CID22 PWRC | ≥ 0.88 (advisory) |

**Why:** a bake that drops CID22 by 0.005 while gaining +0.05
on B0/B1 IS the winning trade. CID22 is informative, not
determinative.

**Trainer lever:** training-target choice (mix_cv40_iw60 vs pure
ssim2 vs other), input feature transforms (this session's YJ
finding), CID22-train subset weighting.

## Priority order

When goals conflict (e.g., cross-codec equivalence trades
against monotonicity), resolve in this order:

1. **G1 (dynamic range)** — a clamped dial is unusable
2. **G3 (monotonicity)** — a non-monotone dial is unreliable
3. **G2 (JND anchor)** — the semantic contract
4. **G4 (cross-codec)** — the picker contract
5. **G5 (KonJND rank)** — the calibration anchor
6. **G6 (band coverage)** — the "no dead zones" guard
7. **G7 (CID22 rank)** — advisory generalization check

## Validation policy — `--val-policy goals`

The current trainer uses `--val-policy min` (worst-corpus SROCC)
to select the "best epoch" checkpoint. This is wrong:

- It optimizes for a SINGLE stat (SROCC) on a SINGLE corpus
  (whichever is worst, usually konjnd_dense which oscillates
  with cyclic LR) — violating the "SROCC-only verdicts BANNED"
  principle.
- It ignores G1 (dynamic range), G2 (JND anchor), G3 (mono),
  G4 (cross-codec) entirely — these are never measured during
  training, only post-hoc.
- The YJ-AT retrain showed the pathology: best val_min was
  epoch 10 (transient post-init) while every corpus improved
  through epoch 299 — except konjnd_dense which oscillated
  and dragged the min-policy down.

**Replace with `--val-policy goals`:** at each validation
checkpoint, compute a weighted pass/fail score against G1–G7:

```
goal_score = (
    w1 * dial_range_ok(features, sweep)     # G1: p5 ≤ 25 ∧ p95 ≥ 85
  + w2 * jnd_anchor_ok(konjnd_preds)        # G2: |mean - 60| ≤ 5
  + w3 * mono_rate(sweep)                   # G3: strict_mono ≥ 0.93
  + w4 * cross_codec_ok(eq_pairs)           # G4: p50|Δ| ≤ 1.5
  + w5 * konjnd_srocc(konjnd_preds)         # G5: SROCC ≥ 0.70
  + w6 * band_coverage(val_corpora)         # G6: max gap ≤ 0.10
  + w7 * cid22_srocc(cid22_preds)           # G7: advisory ≥ 0.85
)
```

Each term is 0.0–1.0 (soft gate: linear ramp from threshold to
target). Weights follow the priority order: w1 > w3 > w2 > w4 >
w5 > w6 > w7. The trainer selects the epoch with the highest
`goal_score`, not the epoch with the best worst-corpus SROCC.

**Why this is better:**
- Validates the actual properties the dial needs (range, mono,
  anchors) during training, not just rank correlation
- A konjnd_dense oscillation doesn't kill the checkpoint if
  mono + range + cross-codec are all passing
- The epoch-10 artifact disappears: early epochs have bad mono
  and no dial range, so goal_score is low even if SROCC peaks

**Implementation shape** (in `zensim-validate/src/mlp_train.rs`):
- At each val checkpoint, score the current weights against a
  small held-out q-sweep fixture (50 images × 19 q, ~1000
  forward passes, <1s on CPU) for G1/G3.
- Score against the existing val corpora for G5/G7.
- Score against the anchor parquet for G2.
- Score against the cross-codec-eq parquet for G4.
- The per-epoch cost is ~2s (vs ~0.5s for the current SROCC-
  only val). Acceptable for 300 epochs.

**Starter weights** (tunable per experiment):

| Goal | Weight | Rationale |
|---|---:|---|
| G1 dynamic range | 3.0 | Broken dial is unusable; highest priority |
| G3 monotonicity | 2.5 | Non-monotone dial is unreliable |
| G2 JND anchor | 2.0 | Semantic contract |
| G4 cross-codec | 1.5 | Picker contract |
| G5 KonJND rank | 1.0 | Calibration anchor |
| G6 band coverage | 0.5 | Guard rail |
| G7 CID22 rank | 0.5 | Advisory |

## Measurement

All goals are measured by `bake_verdict` + the 50-image × 19-q
JPEG sweep (`qsweep_eval`). A single "ship-readiness" command
should emit a pass/fail table for G1–G7.

## Current v11 ship scorecard

| Goal | v11 ship | Pass? |
|---|---|---|
| G1 p5 ≤ 25 | p5 = 28 | ✗ marginal |
| G1 p95 ≥ 85 | p95 ≈ 93 | ✓ |
| G2 JND = 60 ± 5 | mean 60 | ✓ |
| G3 mono ≥ 93% | 92.78% | ✗ marginal |
| G4 p50 \|Δ\| ≤ 1.5 | 1.37 | ✓ |
| G5 KonJND SROCC ≥ 0.70 | 0.285 | ✗ |
| G6 max band gap ≤ 0.10 | TBD | ? |
| G7 CID22 ≥ 0.85 | 0.860 | ✓ |

v11 passes 3/7 cleanly, marginal on 2, fails G5 decisively.
The YJ-AT retrain lifts G5 from 0.285 → 0.666 (closer to the
0.70 floor) but drops G7 to 0.816.
