# Monotone by construction — exploration (2026-10-22)

Question from the owner: can a metric that is correct **by construction** be competitive with
B (`diff+frag-820`, the E33/E34 `base-c` recipe)?

"By construction" here means the network guarantees, for every input and not only on data:

1. `raw(0⃗) = pin` bit-exactly and `raw ≤ pin` (B already has these through `--nonneg-distance`);
2. `raw` is **non-increasing in every input**, so more of any measured damage never raises the score.

B does not have (2). Its first layer has 67,779 negative weights against 22,582 positive. The
LIVE color-noise inversion (`benchmarks/gauntlet_live_color_trace_2026-10-18.md`) is this
mechanism: positive damage passes through negative first-layer weights and switches distance
units off.

This is an exploration record, not a registered experiment. It selects on E21, which is a
TRAIN-population cross-dataset statistic (LODO over KADID TRAIN+SELECT, TID2013, KonFiG
TRAIN+VAL and CID22-A(25)), already exposed by E33/E34/V40. Any candidate it produces needs a
separately registered assessment before it can claim more than "competitive on E21".

## Facts established before any arm ran

- **Domain.** All 420 read features are ≥ 0 with no NaNs on every fit, dev and coverage table and
  on all four E21 assessment tables (266,375 training/dev rows; the 42,021-row coverage pool).
  The 410 fragility products are therefore ≥ 0 as well (fragility ∈ (0, 1]). Monotonicity in
  each input is monotonicity on the whole data domain.
- **Zero-bias monotone networks are linear.** With `b1 = 0`, `W1 ≥ 0` and non-negative inputs,
  every ReLU unit is active on every row, so `raw = pin − c·x`. `--nonneg-distance
  --monotone-cbc` at any width is E34 arm L's class. E34 measured L (A family, 10 seeds) at
  **−0.0226** against `base-a`, with KADID −0.017, TID2013 −0.035, KonFiG −0.037.
- **Capacity inside the monotone class needs thresholds.** `--nonneg-thresholds` (this branch)
  projects `b1 ≤ 0` instead of freezing it at 0. `ReLU(b1) = 0` keeps (1) exact; with
  `W1 ≥ 0` the model is a non-negative sum of hinges `max(0, a·x − t)`, `a ≥ 0`, `t ≥ 0`:
  monotone, convex, piecewise-linear. Within-pair rank loss gives `b1` an exactly zero gradient
  while every unit is active, so thresholds start at per-unit pre-activation quantiles
  `U[0, 0.9)` of 4,096 training rows (own RNG stream).
- **Harness parity.** This workspace's trainer (flag off) and the `v2_lodo_mlp.py` strict route
  reproduce E34's `base-c` KADID seed-0 cell: training curve identical at every logged epoch and
  all 7,869 held-out predictions bit-identical to E34's stored predictions. The scorer
  (`scripts/rev4_featpot/byc_cells.py score`) reproduces E34's published point estimates for
  w-c (+0.00213532) and k-c (−0.00069957) against `base-c`.

## Arms (C family; every other recipe element is B's)

| Arm | Tokens | Class | Guarantee |
|---|---|---|---|
| control | — | E34 `base-c` stored predictions (seeds 0–9) | pin only |
| Lc | `H1:mono` | linear, `W1 ≥ 0` | monotone + pin |
| T | `mono:thr` | sum of hinges, `W1 ≥ 0`, `b1 ≤ 0`, H128 | monotone + pin |
| Ts | `thr` | B plus thresholds, signed `W1` | pin only (capacity control) |

## Decision rules (fixed before any arm result)

- Screen: seeds 0–2 × four folds, paired with `base-c`'s same seed indices. Statistic: E21 point
  estimate (mean over the four sources of signed SROCC difference, averaged over seeds) with
  the seed-unit SE.
- **Competitive on E21** = E33's point form: mean Δ ≥ −0.002 and every dataset Δ ≥ −0.005.
- An arm that is competitive on the 3-seed screen goes to seeds 3–9, then E34's full statistic
  through the `panel e34` owner and the full-data label-free gates (identity, dial C1–C6,
  G-STEER). An arm that misses by more than 2 seed-unit SEs on the screen is reported as
  measured and not extended.
- Every result is reported, including failures. No threshold above changes after a result.

## Results

(filled in as cells land)
