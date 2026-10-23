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
| R | `mono:thr:rp32` | 64 hinges + 32 tied ramp pairs (H128), `W1 ≥ 0` | monotone + pin, non-convex |

**Amendment before any arm result (2026-10-22 ~07:55Z), from a methodology review.** T tests
*convex* monotonicity, so a T loss falsifies convexity, not "monotone by construction". R is the
non-convex monotone arm: each tied pair `w2·(ReLU(z − t₁) − ReLU(z − t₂))`, `t₂ ≥ t₁ ≥ 0`, is a
bounded saturating ramp in the existing wire format; `certify_monotone_bake` verifies the
structure from the bake bytes. B's fragility products are already a narrow content gain under
`W1 ≥ 0` (effective weight `a_k + b_k·f_k ≥ 0`, `f_k ∈ (0, 1]`); a richer content-gated
partial-monotone arm (free content network × monotone damage terms) is a later step.
Because E21 is leave-one-dataset-out, it cannot see damage-mixture shift (the LIVE color-noise
failure came from families the coverage mask excluded). Two checks are added beside it for any
competitive arm: pair classes by distortion type (E34 §4.6 owner), and a label-free
damage-mixture test (adding a second distortion to a distorted image must not raise the score).

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

Cells: `~/tmp/byc/cells/<arm>/without_<fold>_s<seed>/` on dev (train.log, `byc_record.json` with
bake and prediction hashes, held-out `pred_<fold>.tsv`). Arm T used the `8ba33c7a` trainer
(`/var/tmp/byconstruction-target/release` at the time), its predictions the `4024c31b`
`bake_dial_refit`, which was re-checked bit-identical on E34's `base-c` KADID seed-0 cell.
Each T cell: 984–1013 s single-threaded on a contended box, peak RSS 0.99 GiB.

### Arm T (`mono:thr`, convex monotone): not competitive on E21

| | E21 Δ vs `base-c`, seeds 0–2 |
|---|---|
| mean (seed-unit SE) | **−0.0205 (0.0029)**; per seed −0.0154, −0.0255, −0.0207 |
| KADID / TID2013 / KonFiG / CID22-A | −0.0192 / −0.0423 / −0.0267 / **+0.0060** |
| absolute SROCC | 0.8714 / 0.8590 / 0.7904 / 0.8860 |

That is close to E34's linear L (−0.0226, A family). **Correction (same session, after
inspecting the bakes):** T did not use its convex capacity. Every T cell, and the full-data T
fit, ends with **one** live hidden unit out of 128 (two in one cell), sometimes with a small
threshold: the model is at most a single hinge `pin − c·max(0, a·x − t)`, essentially E34 L's
linear class with a dead zone. The other units' output weights sit at the `w2 ≤ 0` clamp.
Hinges are convex (they accelerate distance with damage); the training loss, through its
MSE leg on the teacher groups, wants a *saturating* response, so it pushes hinge output weights
positive and the clamp kills them. So T measures the linear monotone class on this data, not
convex monotone capacity, and the ramp arm R (saturating, still monotone) is the real capacity
test. By the rule above T is not extended. The loss has a specific shape.

**Pair classes** (`byc_cells.py pairclass`, an independent numpy implementation of E34 §4.6's
classes; concordance on target-ordered pairs, exact target ties excluded, cross-source pairs
for KADID a fixed 2M sample; mean over seeds 0–2; Δ = T − `base-c`):

| Dataset | same ladder | cross-type, same reference | cross-source |
|---|---|---|---|
| KADID | **+0.0067** | −0.0133 | −0.0120 |
| TID2013 | **+0.0053** | −0.0205 | −0.0234 |
| KonFiG | **+0.0015** | +0.0041 | −0.0132 |
| CID22-A (codecs) | −0.0006 | −0.0019 | **+0.0032** |

Within a ladder (same reference, same distortion) the monotone model orders as well or better,
up to +0.22 within-reference SROCC on KADID's contrast-change type (`byc_cells.py bytype`).
What it loses is calibration of severity *between* distortion types and contents on the
synthetic-distortion sets. On codec data it is neutral to slightly better.

**Development curves** (TRAIN-internal reference-disjoint dev groups, 12 paired cells, epoch 0 →
119): B's recipe loses human-dev SROCC while it keeps fitting the metric teachers; T does not.

| Dev group | T | `base-c` | T − `base-c` at 119 |
|---|---|---|---|
| human | 0.8122 → 0.8204 (+0.0082) | 0.8201 → 0.8092 (−0.0108) | **+0.0112** |
| CID22 teacher | +0.0028 | +0.0054 | −0.0041 |
| SafeSyn teacher | +0.0027 | +0.0035 | −0.0007 |

**Why convex monotone cannot calibrate across types.** A non-negative sum of hinges
`Σ c·ReLU(a·x − t)`, `a ≥ 0`, cannot express `max(x₁, x₂)`, which is monotone and convex:
`max(x₁, x₂) = x₁ + ReLU(x₂ − x₁)` needs a signed first-layer row. Max-like pooling is how a
metric avoids counting one artifact several times through redundant features; with only
non-negative rows, a distortion type that lights many correlated features is over-weighted
relative to one that lights few. That is the measured cross-type loss. Monotone classes that
do allow max-pooling (min-max networks; certifiable signed constructions) are the next arms.

### Damage-mixture test (label-free): B inverts, monotone does not

`scripts/byconstruction/mixture_test.py` (generate / score / report). 48 TRAIN-eligible 256-px
references from E34's inventory (`p4-inventory-r2/references`; E34's 128 augmentation and 8
development origins excluded; sha256 order within category), 4 base distortions D1 (Gaussian
blur σ 1.2, luma noise σ 4, chroma noise σ 8, posterize 12 levels) × 6 added distortions D2
(luma noise σ 4/8/16, chroma noise σ 8/16/32, BT.601 YCbCr, fixed seeds): 1,152 D1 → D2∘D1
steps, every one of which raises RGB error against the reference. Scored with
`score_pair_with_bake --bake-post raw` (`526261c3` build). An inversion is a score that rises
when D2 is added. Images and scores: `/mnt/v/output/zensim/byconstruction-2026-10-22/mixture/`.

| Model | Inversions / 1,152 | > 1 point | Max rise |
|---|---|---|---|
| B (production `89cb4861`) | **148** | 131 | **+44.2** |
| E34 `base-c` cell (KADID held out, seed 0) | 110 | 107 | +42.9 |
| T cell (same fold and seed) | **0** | 0 | −0.10 (smallest drop) |

B's worst cell, blur σ 1.2 then chroma noise σ 8, inverts on **48 of 48** references: mean B
score 15.0 for the blurred image, 33.8 after adding chroma noise (median rise +17.1, max +44.2).
Example 9940: B 26.5 → 67.3 while RGB MSE rises 6.0 → 80.3; T 66.8 → 33.1. This is the LIVE
color-noise failure mode reproduced systematically on TRAIN images, without labels. The
monotone model's guarantee is in feature space; on these pixel mixtures it also held end to
end.

Across all 12 matched LODO cells (4 folds × seeds 0–2; E34's stored `base-c` dense bakes vs T's):

| | Inversions / 1,152 per cell | Max rise per cell |
|---|---|---|
| `base-c` (B's recipe) | 86–171 | +24.7 to +61.5 |
| T (convex monotone) | 0–4 | ≤ +4.03 |

T's only inversions are posterize-12 followed by *light* noise (luma σ 4 or chroma σ 8), at most
3 references per cell: dithering. Light noise breaks up posterization contours and the
banding-type features fall, and dithered banding often looks better, so these are plausibly
correct. B's inversions are blur followed by chroma noise, with no such rationale. T's
guarantee is in feature space; a feature that falls when damage is added (here, banding under
dither) is the only route to an end-to-end rise.

### Arm R (`mono:thr:rp32`, saturating monotone): uses its capacity, no E21 gain

`4024c31b` trainer. 12 cells, 2006 s for the batch on a contended box, peak RSS 1.04 GiB.

| | E21 Δ vs `base-c`, seeds 0–2 |
|---|---|
| mean (seed-unit SE) | **−0.0225 (0.0024)**; per seed −0.0181, −0.0265, −0.0230 |
| KADID / TID2013 / KonFiG / CID22-A | −0.0168 / −0.0391 / −0.0299 / −0.0043 |

Unlike T, R keeps its structure: 13–29 live ramp pairs per cell (`live_units.py`), plus exactly
one live plain hinge (the other 63 hinges die, as in T). Saturation is used and does not move
E21: the missing ingredient is not the response shape.

Pair classes (Δ vs `base-c`): same ladder KADID +0.0070, TID +0.0085, KonFiG +0.0016, CID22-A
−0.0001; cross-type KADID −0.0109, TID −0.0172, KonFiG −0.0004, CID22-A −0.0019; cross-source
KADID −0.0098, TID −0.0226, KonFiG −0.0170, CID22-A −0.0031.

**Development curves revise the overfitting reading.** R's human-dev SROCC falls over training,
0.8080 → 0.7903 (−0.0177; `base-c` −0.0108), while teacher dev rises (+0.0016 SafeSyn, +0.0027
CID22). T's flat human-dev curve came from collapsing to a linear model, not from monotonicity:
with capacity, a monotone model also trades human ranking for teacher fit. The drift is a
capacity × teacher/human-recipe effect, not a sign-freedom effect.

Mixture test: 4–9 inversions per cell, max +5.22, all posterize followed by light noise
(dithering), the same cases as T.

### Where the monotone deficit lives (post hoc)

The type set below was chosen after reading the per-type table, so this is descriptive, not a
test. Removing the "benign global change" distortion types (KADID 7/8 saturation, 16 brighten,
17 darken, 18 mean shift, 20 non-eccentricity patch, 25 contrast change; TID2013 16 mean
shift, 17 contrast change, 18 saturation change; `scripts/byconstruction/benign_types.py`):

| Arm | KADID: all types → without | TID2013: all types → without |
|---|---|---|
| T | −0.0192 → −0.0090 | −0.0423 → −0.0129 |
| R | −0.0168 → −0.0062 | −0.0391 → −0.0114 |

Roughly 60–70% of the deficit on these two sets comes from distortions that move many damage
features while humans object little; a signed model can subtract the explained part, a
monotone one cannot. KonFiG (−0.027 / −0.030) has none of these types and keeps its deficit.

### The exposed LIVE color-noise case (report-only)

`manfishing`, `wn/img41` (severe, human 1 − DMOS/100 = −0.0024) vs `wn/img73` (milder, 0.3097),
`score_pair_with_bake --bake-post raw`. These examples are exposed and stay out of fitting and
selection (CLAUDE.md, 2026-10-18 Known Bug); this is a diagnostic read only.

| Model | severe img41 | milder img73 | ordered correctly |
|---|---|---|---|
| B (`89cb4861`) | 49.71 | −13.91 | no (matches the trace's 49.7143 / −13.9146) |
| `base-c` KADID-held-out cell s0 | 1.72 | −21.65 | no |
| T same cell | −207.42 | −68.62 | yes |
| R same cell | −171.78 | −46.56 | yes |
| T full-data, packed f16 + spline | −163.72 | −60.27 | yes |

### Label-free product gates: the full-data monotone model passes all of them

Full-data T, seed 0, packed by the E33 output stage (`v2_confirm_fit.py --pack-production
--e33-output-stage`, 776 s, peak RSS 0.99 GiB; `production-f16.bin` 21,215 B because only one
hidden unit is live). `bake_dial_refit certify-monotone`: certified, monotone. Gates through
`scripts/byconstruction/gates.py` (E33/E34 owners and binaries; outputs
`/mnt/v/output/zensim/byconstruction-2026-10-22/gates/`), against E34's `base-c` seed 0:

| Gate | Monotone T (full) | `base-c` s0 |
|---|---|---|
| C1 dial monotonicity, standard / ladder | **0.9970 / 0.9914** | 0.9954 / 0.9882 |
| C2–C6 (both grids) | pass | pass |
| G-DIAL (p5 ≤ 25, p95 ≥ 85, mono ≥ .93) | pass (p5 13.8 / −9.9, p95 97.5 / 95.2) | pass |
| NEARID N1–N3 | pass, **140/144** monotone ladders | pass, 130/144 |
| G-STEER (D12 exact gradients, 135 cases) | **135/135** | 130/135 |
| Identity at every SIMD tier (`e33_gates.identity`, C5 as registered) | 62 sources pass at all 10 permutations (620 rows) | (E34 gate record) |


The model is effectively linear, which is E34 L's class (E34's L also cleared G-STEER), so the
steering pass is expected, but it is now attached to a certified-monotone bake with a measured
mixture-test record.

### Arms added after T/R (definitions recorded before their results)

| Arm | Tokens | Class | Certified guarantee |
|---|---|---|---|
| E5 / E20 | `mono:thr:fr32:fb5` / `…:fb20` | 64 monotone hinges + 32 sign-free ramp pairs, summed amplitude ≤ 5 / 20 raw units | pin; raising any input raises the score by ≤ 5 / 20 |
| G | `mono:thr:gate` | monotone hinges; each fx1 (direct, product) pair projected onto `{a ≥ 0, a/s_d + b/s_p ≥ 0}` | pin; non-increasing in every damage feature for every fragility in (0, 1] |
| Lc | `H1:mono` | linear, `W1 ≥ 0`, C family | monotone (E34 L in B's family) |
| Ts | `thr` | B + trainable thresholds, signed `W1` | pin only (capacity control) |

G is the content-gated partial-monotone class: T forced both weights ≥ 0, so a unit's gain on
damage could only grow with fragility; G lets it rise or fall with content while staying
non-negative. The gate cone keeps a 2⁻⁸ margin and a direct-weight floor so f16 packing
cannot cross it; `certify_monotone_bake` checks it from the bake's `zensim.derived_inputs`
declaration (product factors must be `pjnd_fragility` slots). Smoke: a 2-epoch real-data G
cell's dense bake certifies monotone with 5,535 negative product weights.

The same decision rule applies to each: E33 point form on the 3-seed screen, then seeds 3–9
and the full-data gates for any arm that meets it.

**Per-type calibration bias** (`scripts/byconstruction/type_bias.py`; mean pooled prediction rank
minus pooled human rank, seeds 0–2; negative = rated worse than humans). Largest T − `base-c`
gaps:

| Set | Type | T | `base-c` |
|---|---|---|---|
| KADID | contrast change | −0.132 | −0.023 |
| KADID | color shift | −0.006 | +0.098 |
| KADID | color quantization | −0.059 | +0.003 |
| KADID | non-eccentricity patch | −0.086 | −0.036 |
| TID2013 | high-frequency pattern noise | −0.217 | −0.095 |
| TID2013 | local block-wise | +0.040 | +0.150 |
| TID2013 | denoising | −0.087 | −0.003 |

The deficit is two-sided: the monotone model over-penalizes types that light up many damage
features (contrast change, pattern noise, denoising), while B's recipe over-forgives some
others (color shift, block-wise) where T is closer to the human rank. "Benign global change"
is part of the story, not all of it.

### Arm E5 (`mono:thr:fr32:fb5`, certified budget 5): budget fully used, no E21 gain

`526261c3` trainer, 1,910 s for 12 cells, peak RSS 1.03 GiB.

| | E21 Δ vs `base-c`, seeds 0–2 |
|---|---|
| mean (seed-unit SE) | **−0.0194 (0.0027)** |
| KADID / TID2013 / KonFiG / CID22-A | −0.0158 / −0.0403 / −0.0231 / +0.0014 |

Every cell certifies (`certify-monotone` on its dense bake) with 5–21 live free ramps and a
certified free amplitude of 3.47–4.98 of the 5-unit budget: the optimizer wants more
non-monotone room than 5 points, and 5 points buys nothing measurable on E21. Pair classes
match T (same ladder KADID +0.0067, TID +0.0057; cross-type −0.0110 / −0.0189; cross-source
−0.0099 / −0.0226). Mixture test: 3–8 inversions per cell, max rise +5.01 (dithering cases; the
budget bound plus the feature-space-to-pixel gap).

### Information vs calibration: an oracle per-type calibration (diagnostic upper bound)

`scripts/byconstruction/oracle_type_calibration.py` gives each distortion type its own
least-squares affine map from prediction to human score, using the labels themselves (an
oracle, not a model), applied identically to the arm and to `base-c`; then pooled SROCC Δ:

| Arm | KADID raw → calibrated | TID2013 | KonFiG | CID22-A |
|---|---|---|---|---|
| T | −0.0192 → **+0.0015** | −0.0423 → −0.0054 | −0.0267 → −0.0153 | +0.0060 → +0.0078 |
| R | −0.0168 → **+0.0025** | −0.0391 → −0.0081 | −0.0299 → −0.0159 | −0.0043 → −0.0025 |
| E5 | −0.0158 → **+0.0019** | −0.0403 → −0.0058 | −0.0231 → −0.0144 | +0.0014 → +0.0055 |

Within each distortion type, across content and severity, the monotone models carry as much
human ranking information as B's recipe (slightly more on KADID). The E21 gap is almost
entirely how *different distortion types* are placed on one severity scale, which B obtains
from sign contrasts between correlated damage features. KonFiG keeps about −0.015. CID22-A,
whose "types" are codecs, is positive in raw form for T and E5.

So, on this feature set: coordinate-wise monotonicity and cross-type severity calibration
conflict; ranking within a distortion type does not suffer. For codec targeting, where the
types are codecs (CID22-A), the monotone models are at parity or better.

**Arm M (`mono:thr:mp32`), recorded before its result.** The oracle above says the monotone
deficit is cross-type severity calibration. The standard monotone way to stop one artifact
being counted several times through redundant features is to pool with a maximum instead of a
sum, and `W1 ≥ 0` hinge sums cannot express `max(x₁, x₂)`. M's first 64 hidden units form 32
certified max pairs `c·max(a·x, b·x − t)` (u: row `a ≥ 0`, bias 0; v: row `b − a`, `b ≥ 0`,
bias `−t ≤ 0`; tied `c ≤ 0`), the other 64 stay monotone hinges. A 2-epoch real-data smoke keeps
31 live max pairs and certifies monotone. Queue order (serial, one heavy job at a time): E20, G,
M, Lc, Ts.

### Prior evidence: data and inputs alone do not remove the inversion (registered, 2026-10-18)

[color_noise_repair_2026-10-18.md](color_noise_repair_2026-10-18.md) added the color/noise
coverage families that `cf98` excludes, restored fine-scale X/B inputs, and did both, on B's
signed class, as single models and as 3-seed ensembles. Every composition still scores the
severe LIVE `wn/img41` above the milder `wn/img73` (severe 0.26–62.6 vs milder −31.3 to −5.95).
The monotone bakes here order that pair correctly by construction. Coverage is a complement to
the guarantee, not a substitute for it.

### Arm E20 (`mono:thr:fr32:fb20`): the certified budget trades E21 for bounded inversions

`526261c3` trainer, 1,915 s for 12 cells, peak RSS 1.02 GiB. Every dense bake certifies; the
free amplitude is 19.67–20.00 in 9 cells and 5.3–6.6 in 3.

| Non-monotone budget ε (certified) | E21 Δ vs `base-c` (seed-unit SE) | Mixture inversions / cell | Max rise |
|---|---|---|---|
| 0 (T) | −0.0205 (0.0029) | 0–4 | +4.0 (dithering) |
| 5 (E5) | −0.0194 (0.0027) | 3–8 | +5.0 (dithering) |
| 20 (E20) | **−0.0154 (0.0026)** | 4–37 | +7.3 |
| unbounded (`base-c`) | 0 | 86–171 | +24.7 to +61.5 |

E20 per dataset: KADID −0.0110, TID2013 −0.0364, KonFiG −0.0167, CID22-A +0.0026. Pair
classes vs `base-c`: cross-type KADID −0.0078 (T −0.0133), TID −0.0160 (T −0.0205).

In one E20 cell (TID held out, seed 0) the free room reproduces B's failure mode: blur σ 1.2
then chroma noise σ 8 inverts on 26 of 48 references (max +7.3, inside the certificate's 20).
The non-monotone room that buys E21 is spent partly on the same inversions B makes; the
certificate bounds their size.

### Correction: the logged human-dev decline is not a clean memorization signal

The trainer's per-epoch dev SROCC (`compute_light_panel_subsampled` over all rows of a group)
is pooled across the human dev group's sources, each with its own label scale; E21 scores each
held-out dataset separately. Testing it directly: every cell also saved `refit/best.bin`, the
trainer's best dev-geomean3 epoch (the harness's `best_dev` rule).
`byc_cells.py predict-best` + `score --rule best` and `scripts/byconstruction/epoch_rules.py`:

| Model | mean E21 pooled SROCC, last epoch → best-dev epoch |
|---|---|
| `base-c` | 0.8722 → 0.8642 (**−0.0080**) |
| T | 0.8517 → 0.8518 |
| E5 | 0.8528 → 0.8533 |
| E20 | 0.8569 → 0.8529 |
| R | 0.8497 → 0.8461 |

Selecting by dev hurts B's recipe on held-out datasets; later epochs are what make it better
there. Under the `best_dev` rule the arm deltas shrink (T −0.0124, E5 −0.0109, E20 −0.0113, R
−0.0181) only because the control gets worse. The registered last-epoch comparison above
stands, and the earlier statements that R's or B's dev decline shows memorization are
withdrawn: what the logged metric shows is a shift in pooled cross-source ranking on the
in-population dev references, not an E21 loss.

## Where this stands (09:40Z; G, M, Lc, Ts still running)

**Can a metric that is correct by construction be competitive with B?**

- *Product gates:* yes. The certified-monotone full-data model passes every label-free gate
  run here (C1–C6, G-DIAL, N1–N3, identity at every tier, G-STEER 135/135 where `base-c` gets
  130/135), with higher dial monotonicity.
- *Robustness:* strictly better. 0–9 mixture inversions per cell (all dithering) against
  86–171 with rises to +61 for B's recipe; the exposed LIVE pair is ordered correctly. A
  registered repair with more coverage and inputs did not achieve this on B's signed class.
- *Codec calibration:* at parity. CID22-A (codec types) is +0.006 for T in raw form.
- *E21:* not yet. Best certified-monotone arm −0.0194 (E5); with a certified non-monotone
  budget of 20 raw units, −0.0154. The deficit is cross-type severity calibration on synthetic
  distortion sets: with an oracle per-type affine calibration the monotone arms are at parity on
  KADID (+0.002) and within −0.005 to −0.008 on TID.

**Mechanism.** B places distortion types on one severity scale with sign contrasts between
correlated damage features (half of its damage features have a positive derivative on a
typical row). The same contrasts produce its inversions. Coordinate-wise monotone combiners
built from these features cannot do that calibration; response shape (saturating ramps) does
not help, and a certified non-monotone budget buys it back slowly while spending part of the
room on the same inversions.

**Not supported by the data:** that sign freedom causes memorization (the logged dev decline
is pooled across sources; selecting by it lowers B's E21 by 0.008).

### A feature-level lead for the cross-type deficit (descriptive)

T's KADID-held-out seed-0 cell has one live unit (bias 0), so its distance splits exactly into
per-input contributions (`scripts/byconstruction/feature_contributions.py`). Share of total
distance by feature family, global-change types vs all KADID rows:

| Types | v2 ringing | other over-represented families | mean distance |
|---|---|---|---|
| contrast change | 15.0% (all types 7.2%) | hf_loss 9.0% (6.7%), transducers ~1.5% (0.9%) | 68.9 |
| saturation 1+2 | 12.6% | iw_ssim 8.1% (2.6%), iw_mse 3.3% (0.9%) | 129.1 |
| mean shift / brighten / darken | 16.4% | det family slightly up | 67.9 |
| all types | 7.2% | — | 86.1 |

The ringing feature (dilated-edge form) responds strongly to global tone and saturation
changes, which a ringing detector should not; on saturation the IW-weighted SSIM/MSE features
are about 3× over-represented. Tone-normalized versions of these features are the concrete
feature-side experiment for letting a monotone combiner calibrate across types. Not tested
here.

**Arm FR (`mono:thr:freering`), recorded before its result; post hoc.** Same as T, but the 10
v2 ringing direct inputs and their 10 fragility products (20 of 820 inputs;
`scripts/byconstruction/masks/free_ringing.tsv`) are sign-free via the existing
`--monotone-feature-mask`. Guarantee: monotone in the other 800 inputs. The feature was chosen
after reading contributions on KADID assessment rows, so FR is exploratory and cannot by itself
support a claim. It tests whether the cross-type deficit sits in a few mis-specified features.
Queued after Ts.

### Arm G (`mono:thr:gate`, content-gated monotone): no change

`4112601a` trainer. E21 Δ **−0.0202 (0.0030)**; KADID −0.0178, TID −0.0429, KonFiG −0.0261,
CID22-A +0.0060 (T: −0.0205). Dense bakes certify monotone through the gate rule. Like T, each
cell keeps one live unit, now a linear model whose damage gains move with fragility in both
directions (product weights down to −1.99, inside the cone). Pair classes match T (cross-type
KADID −0.0123, TID −0.0205). Content-dependent gain is not what the monotone class lacks; the
deficit is cross-type. Mixture test: 0–4 inversions per cell, max +4.49, dithering cases only.

### Arm M (`mono:thr:mp32`, certified monotone max pairs): best exactly-monotone arm

`d2b7ce30` trainer. Every dense bake certifies monotone (1–2 live signed max pairs each).

| | E21 Δ vs `base-c`, seeds 0–2 |
|---|---|
| mean (seed-unit SE) | **−0.0144 (0.0016)**; per seed −0.0133, −0.0123, −0.0175 |
| KADID / TID2013 / KonFiG / CID22-A | −0.0147 / −0.0252 / −0.0183 / +0.0008 |

Paired with T by seed: +0.0021, +0.0132, +0.0032. In 10 of 12 cells exactly two hidden units are
live, i.e. the whole model is `pin − c·max(a·x, b·x − t)`: one learned maximum between two
non-negative damage detectors. That single pooled pair recovers a large part of the cross-type
deficit:

| Pair class, Δ vs `base-c` | T | M |
|---|---|---|
| TID2013 cross-type | −0.0205 | **−0.0120** |
| TID2013 cross-source | −0.0234 | **−0.0152** |
| KADID cross-type | −0.0133 | **−0.0101** |
| KonFiG cross-source | −0.0132 | **−0.0088** |
| same ladder (all sets) | +0.0015 to +0.0067 | +0.0016 to +0.0064 |

Max-pooling is the monotone mechanism for cross-type calibration, as predicted from the
`max(x₁, x₂)` argument, and training keeps only one pair alive.

**Mixture test: feature-space monotonicity is not pixel-space monotonicity.** M has 1–32
inversions per cell (max +6.54), now including blur followed by noise. Some features are not
monotone in pixel damage: `hf_loss` (detail lost) falls when noise adds high-frequency energy
back to a blurred image, and a max lets the score follow whichever detector dominates. The
rises are small next to B's (+61), but a pixel-level guarantee needs damage-monotone features
as well as a monotone combiner; the mixture test is the practical check.

### Arm Lc (`H1:mono`, linear monotone in B's family)

E21 Δ **−0.0204 (0.0029)**; KADID −0.0186, TID −0.0423, KonFiG −0.0266, CID22-A +0.0060 — the
same as T, confirming T was the linear class in effect. (E34's L in the A family: −0.0226.)
12 cells in 68 s: width 1 is cheap.

### Seed stability (`scripts/byconstruction/seed_spread.py`)

Mean E21 pooled SROCC over the four sources, seeds 0–2:

| Model | mean | per seed | sd |
|---|---|---|---|
| `base-c` (B's recipe) | 0.8722 | 0.8665 / 0.8769 / 0.8733 | 0.0053 |
| E34 `w-c` (H32) | 0.8775 | 0.8758 / 0.8785 / 0.8782 | 0.0015 |
| T / Lc / G / E5 / E20 / R | 0.8497–0.8569 | — | 0.0008–0.0015 |
| M | 0.8579 | 0.8532 / 0.8646 / 0.8558 | 0.0060 |

The sign-constrained arms are 4–6× more seed-stable than B's recipe; M's single max pair brings
the variance back (which pair is found depends on the seed). E34's narrow `w-c` is both more
stable and higher than B's own width: plain capacity reduction is a cheap stabilizer.

**Seed ensembles** (`seed_ensemble.py`; mean raw prediction over seeds 0–2; averaging
certified-monotone models stays monotone): `base-c` 0.8722 → 0.8760, M 0.8579 → 0.8599, E20
0.8569 → 0.8578, T 0.8517 → 0.8519. Ensemble gaps vs `base-c`: M −0.0161, E20 −0.0182, T
−0.0241. B's recipe gains more from ensembling, so this does not close the gap.

### Arm Ts (`thr`, B plus thresholds, signed W1): parity

E21 Δ **−0.0008 (0.0012)**; KADID −0.0004, TID −0.0036, KonFiG +0.0016, CID22-A −0.0006 (per seed
+0.0004, −0.0032, +0.0005). 12–15 live units per cell (B's own bake has 87). Thresholds and unit
count do not matter for the signed class; the 0.02 between the monotone arms and B is the sign
structure.

### Arm FR (`mono:thr:freering`, monotone in 800 of 820 inputs): one feature carries both the gap and the inversion

`d2b7ce30` trainer, 12 cells in 757 s. Exploratory (ringing was chosen after reading KADID
assessment rows).

| | E21 Δ vs `base-c`, seeds 0–2 |
|---|---|
| mean (seed-unit SE) | **−0.0039 (0.0029)**; per seed −0.0011, −0.0097, −0.0009 |
| KADID / TID2013 / KonFiG / CID22-A | −0.0046 / −0.0153 / −0.0043 / **+0.0085** |

Pair classes vs `base-c`: cross-type KADID −0.0028, TID −0.0056 (T: −0.0133, −0.0205);
cross-source KADID −0.0022, TID −0.0086; same ladder still ≥ 0 except CID22-A −0.0002. Freeing
the sign of one feature family (20 inputs) recovers about 80% of the E21 gap. The learned
ringing weights are large and negative (to −22): the model subtracts ringing.

**And the inversion comes back through the same feature.** Mixture test: 2–41 inversions per
cell, max rise +26.2, dominated by blur σ 1.2 then chroma noise σ 8 (up to 27 of 48 references)
— B's signature failure. Feature dump (`score_pair_with_bake --dump-features-to`, B's read set):
adding chroma noise to the blurred image raises the ringing slots sharply, e.g. reference 9940
slot values 0.0012 → 0.0582 and 0.0019 → 0.0349 (references 3318 and 5050 behave alike).

**Mechanism.** The v2 ringing feature responds to benign global tone and saturation changes
*and* to chroma noise. Calibrating the benign changes requires subtracting it; the same
subtraction forgives chroma noise. That one conflation explains most of the cross-type
calibration gap and B's blur + chroma-noise inversion (and plausibly the LIVE color-noise case).
*Correction (arm Bnr, below): this holds for FR, whose only free features are ringing. It does not
hold for B: B retrained without ringing keeps the same inversion, so B's comes from its sign
freedom over correlated features, not from ringing alone.*
The fix is at the feature level: a ringing measure that does not respond to global tone or
saturation changes (or the two components as separate features, with the global-change part
treated as a non-damage descriptor). Then a monotone combiner can use the damage part without
subtracting anything.

**Arm DR (`mono:thr:dropring`), recorded before its result.** T with the same 20 ringing
inputs *dropped* (zeroed every step via `--monotone-feature-mask … --monotone-strict`): exactly
monotone over the remaining 800, certificate passes, no subtraction route. Tests whether
ringing forced to count as damage is what over-penalizes the global-change types. Smoke:
certified, 20 zero input rows, no negative weights. Queued after `mp64`.

**Why ringing conflates the two** (`zensim/src/feature_v2.rs`, `idx::RINGING`): per pixel,
`saturate(|src − dst|, C_RING_ERR=0.05) · saturate(activity, C_ACTIVITY) ·
(1 − saturate(|∇src|, C_RING_EDGE=0.02))`, mean-pooled. The error term is the raw per-pixel
difference with no local mean removed, so a global tone, contrast or saturation change makes it
uniformly large off edges, and chroma noise raises it in X/B. **Proposed feature fix (not
done; a feature-system change for the owner):** compute the ringing error on a local-mean-removed
(high-pass) difference, as a new append-only slot, so low-frequency tone shifts stop
registering while noise and true ringing still do. The IW-weighted SSIM/MSE slots, 3×
over-represented on saturation changes, are the next candidates for the same check. Any such
change means new feature ids or a formula revision and re-extraction of every training table.

### Feature probe: ringing confirmed flawed; IW-SSIM and hf_loss not confirmed

`scripts/byconstruction/feature_probe.py` (24 of the mixture references; B's computed Rev5
vector via `score_pair_with_bake --dump-features-to`; median over references of each family
summed over its 12 cells; outputs `/mnt/v/output/zensim/byconstruction-2026-10-22/feature-probe/`):

| Family | mean shift +8 | contrast ×0.85 | saturation ×1.25 | saturation ×0.8 | blur σ1.2 | luma noise σ4 | chroma noise σ8 | blur + luma noise |
|---|---|---|---|---|---|---|---|---|
| ringing | **0.3336** | 0.2308 | 0.1869 | 0.1596 | 0.1389 | 0.0829 | 0.2031 | 0.1710 |
| iw_ssim | 0.0168 | 0.0533 | 0.0481 | 0.0408 | 0.2856 | 0.0344 | 0.2892 | 0.3256 |
| iw_mse | 0.2579 | 0.1837 | 0.0422 | 0.0304 | 0.1989 | 0.0218 | 0.0512 | 0.2137 |
| hf_loss | 0.0653 | 0.5329 | 0.0440 | 0.4889 | 0.8970 | 0.2048 | 0.4639 | 0.9086 |
| mse | 0.2580 | 0.1787 | 0.0369 | 0.0254 | 0.1536 | 0.0218 | 0.0508 | 0.1702 |

- **Ringing is flawed**: a barely visible +8 luma shift is its largest response in the probe,
  2.4× a visible blur and 4× luma noise; contrast and saturation edits also exceed blur.
- **IW-SSIM** responds weakly to global edits; the earlier share-of-distance excess on
  saturation was not a feature defect.
- **hf_loss** falls when luma noise is added to the blurred image on only 6 of 24 references;
  the hypothesis that it drives M's blur + noise inversions is not supported.
- `mse` / `iw_mse` respond to a mean shift by definition (they measure it); not a defect.

### Full-data M: every product gate passes, best dial numbers in this study

Full-data M seed 0 (packed by the E33 output stage; `certify-monotone`: monotone, 1 live max
pair). Gates in `/mnt/v/output/zensim/byconstruction-2026-10-22/gates-M/`:

| Gate | M (full) | `base-c` s0 |
|---|---|---|
| C1 dial monotonicity, standard / ladder | **0.9984 / 0.9935** | 0.9954 / 0.9882 |
| C2–C6, G-DIAL (both grids) | pass | pass |
| NEARID N1–N3 | pass, **143/144** monotone ladders | pass, 130/144 |
| G-STEER (D12 exact gradients) | **135/135** | 130/135 |
| Identity at every SIMD tier | 62 sources × 10 permutations pass | — |

GitHub: imazen/zensim#72 records the ringing feature defect found here.

### Feature probe, all families (`feature_probe.py --all-families`)

Median over the 24 references, each v2 family summed over its 12 cells (outputs
`feature-probe/probe_all.json`). Clipping caveat: `mean_shift+8` and `saturation_x1.25` can clip
at 255, which creates real plateaus and contours; `contrast_x0.85` and `saturation_x0.8` do not.

| Family | +8 shift | contrast ×0.85 | sat ×1.25 | sat ×0.8 | blur σ1.2 | luma noise σ4 | chroma noise σ8 |
|---|---|---|---|---|---|---|---|
| ringing | 0.334 | **0.231** | 0.187 | **0.160** | 0.139 | 0.083 | 0.203 |
| banding | 0.011 | 0.011 | 0.198 | 0.011 | 0.132 | 0.196 | 1.269 |
| blockiness | 0.008 | 0.010 | 0.109 | 0.018 | 0.135 | 0.148 | 0.619 |
| hf_gain | 0.038 | 0.022 | 0.543 | 0.035 | 0.275 | 0.288 | 1.194 |
| hf_loss | 0.065 | 0.533 | 0.044 | 0.489 | 0.897 | 0.205 | 0.464 |
| pjnd_transducer | 0.705 | 0.494 | 0.376 | 0.314 | 0.357 | 0.151 | 0.407 |
| transducer_lowk | 0.801 | 0.575 | 0.398 | 0.330 | 0.399 | 0.169 | 0.425 |
| mse | 0.258 | 0.179 | 0.037 | 0.025 | 0.154 | 0.022 | 0.051 |
| edge_width | 0.000 | 0.000 | 0.036 | 0.038 | 0.181 | 0.003 | 0.913 |

Only **ringing** responds to clipping-free global edits more than to a visible blur, against
its purpose (filed as imazen/zensim#72, with this check added as a comment). banding,
blockiness and hf_gain spike only on the clipping-prone saturation-up edit; the transducers'
response to contrast reduction tracks a visible change; hf_loss under contrast/saturation
reduction and mse under a mean shift are what those measures define. Not filed.

### Arm DR (`mono:thr:dropring`): dropping ringing does not help

E21 Δ **−0.0216 (0.0031)**; KADID −0.0192, TID −0.0447, KonFiG −0.0280, CID22-A +0.0053 (T −0.0205;
FR −0.0039). Ringing forced to count as damage is not what over-penalizes the global-change
types; the mse-type, transducer and hf_loss families do that. What ringing gives a signed model
is an accidental **global-change detector**, which it subtracts to discount those other
features (FR's large negative ringing weights). Fixing ringing alone would remove that signal. DR's mixture
test: 0–3 inversions per cell, max +4.26, dithering only (like T).

**Revised design.** (1) Make ringing measure ringing (high-pass error), closing the chroma-noise
loophole. (2) Add an explicit global-change descriptor, zero at identity and blind to noise —
e.g. per channel, the fraction of the error explained by a global affine map `dst ≈ a·src + b`
(≈1 for mean shift, contrast or saturation, ≈0 for noise or blur) — and let it *reduce*
damage gains through the existing gated-product structure (gain ≥ 0). The model stays monotone
in every damage feature, forgives global changes explicitly, and noise cannot exploit the
descriptor because noise is not affine-explainable. Posted to imazen/zensim#72 as a correction.

**Prior art for the revised design (zenpapers corpus).** The decomposition is the classical
one: SSIM factors local similarity into luminance, contrast and structure, the structure term
being invariant to local affine luminance/contrast changes (Wang et al. 2004,
doi:10.1109/TIP.2003.819861; multiscale 2003, doi:10.1109/ACSSC.2003.1292216); CW-SSIM is
designed to be insensitive to luminance and contrast changes (2009, doi:10.1109/TIP.2009.2025923);
MAD separates detection-based from appearance-based judgment (Larson & Chandler 2010,
doi:10.1117/1.3267105). Applied here: compute the damage bank on the distorted image after
removing the best global (or low-frequency) affine tone map per channel — the residual damage —
plus a small set of global-change magnitude features, and combine all of them monotonically.
Global changes are then discounted by a small learned weight on their own terms, with nothing
subtracted.

**Arm DG (`mono:thr:dropglobal`), rule fixed before its result.** Exactly monotone (strict
mask) with every damage family dropped whose probe response to a mild or clipping-free global
edit exceeds its response to a visible σ1.2 blur: mse, masked_mse, iw_mse, pjnd_transducer,
transducer_lowk, transducer_highk, ringing (`masks/global_sensitive.tsv`). The selection uses
only the label-free probe, not assessment labels. A cheap proxy for "structure-only residual
damage"; if it beats T clearly, the affine-residual feature design is worth building.

### Arm DG (`mono:thr:dropglobal`): removing the global-sensitive families does not help

E21 Δ **−0.0229 (0.0032)**; KADID −0.0198, TID −0.0478, KonFiG −0.0304, CID22-A +0.0064 (T −0.0205,
DR −0.0216). The mse-type, transducer and ringing families still carry information a monotone
model needs (for noise, for example); removing them costs more than their over-penalization of
global changes. The calibration cannot be fixed by subtracting families from the bank: it needs
the same measures recomputed on the affine-residual difference, or an explicit discount signal.

### Arm M64 (`mono:thr:mp64`, all units max pairs)

E21 Δ **−0.0131 (0.0023)**; KADID −0.0166, TID −0.0215, KonFiG −0.0150, CID22-A +0.0005 (per seed
−0.0111, −0.0177, −0.0107). Certified monotone; 1–2 live max pairs in 11 of 12 cells (one cell
keeps all 128 units). Mixture test 1–6 inversions per cell, max +7.56. Within noise of M (−0.0144).

### Arm Bnr (`fx1nr`): B without ringing, signed

Owner question (13:00Z): is ringing essential, and what does doing it right cost? Bnr is B's signed
recipe (`sel:c6a94f8337b0@h32:H128:cv16:cf98:fx1nr`) with the 10 ringing ids removed from both legs:
400 direct inputs plus 400 fragility products (`scripts/rev4_featpot/byc_fx1nr_declaration.json`,
sha256 `923da70c…`). Everything else, including the signed first layer, is B's.

E21 Δ **−0.0003 (0.0009)**; KADID −0.0010, TID −0.0023, KonFiG +0.0041, CID22-A −0.0019 (per seed
+0.0008, +0.0002, −0.0020); E33 point form passes. B does not need ringing for its rank
performance: the signed model finds the same global-change discount through other features.

Mixture test: 72–178 inversions per cell (`base-c` 86–171), max rise +53.1 (`base-c` +61.5). The
worst combination is still blur σ1.2 + chroma noise σ8, which inverts on 28–48 of 48 references per
cell, followed by blur + chroma σ16, blur + luma σ4 and posterize + chroma noise. Removing ringing
does not remove B's inversion; sign freedom does. In the monotone class, ringing's only value was
as a subtractable global-change signal (FR −0.0039 vs DR −0.0216).

### Cost of the ringing fix, and of dropping ringing

Bench: `byc_score_timing` (new, `zensim-validate/src/bin/`) times `BakeScorer::compute` (score only,
no diffmap) on one deterministic textured pair with uniform ±3 noise. Settings: single thread
(`RAYON_NUM_THREADS=1`), pinned to core 2, default dispatch on dev (Zen 5). Each round is the median
of 40 calls (512²) or 8 calls (2048²) after 3 warmups; 12 rounds alternate the order of the two arms
(`scripts/byconstruction/ring_timing.sh`). Paired Δ is the median of the per-round ratios, with a
95% bootstrap CI (5,000 resamples). Raw rounds and hashes are in
`benchmarks/byconstruction_timing_2026-10-22/`.

**High-pass ringing prototype** (`RUSTFLAGS=--cfg byc_hp_ring`, all five ringing sites in
`zensim/src/feature_v2_kernels.rs`). The ringing error becomes `(src − mean4(src)) − (dst − mean4(dst))`
over the 4-neighbourhood the kernel already loads, instead of `src − dst`, so a global offset or
a smooth tone map no longer counts as ringing. Both builds score B's served bake (`89cb4861`):

| Size | Baseline median | High-pass median | Paired Δ (95% CI) |
|---|---|---|---|
| 512² | 5.979 ms | 6.021 ms | +0.78% (+0.62..+0.92) |
| 2048² | 87.016 ms | 87.263 ms | +0.24% (−0.39..+0.71) |

**Dropping ringing from the model** (baseline binary only; `base-c` kadid_s0 bake `7bd14f56` vs
Bnr kadid_s0 bake `3b7af590`):

| Size | `base-c` median | Bnr median | Paired Δ (95% CI) |
|---|---|---|---|
| 512² | 6.017 ms | 6.017 ms | −0.03% (−0.08..+0.09) |
| 2048² | 86.658 ms | 86.739 ms | +0.07% (−0.11..+0.41) |

Ringing is computed in the shared gradient kernel whether or not a bake reads it, so dropping it
saves nothing measurable, and fixing it costs under 1% single-threaded.

Limits:
- Load average was 3–8 during the runs, decaying from my own finished jobs; no foreign build or
  training ran.
- Tested on one synthetic content type, one host and one dispatch tier. Not measured
  multi-threaded.
- The prototype changes only the extraction kernel, not the spatial/diffmap sites in
  `feature_v2.rs`.
- It changes ringing values (512² timing pair: 88.2805 → 88.2424), so it is a new arithmetic
  revision. No stored table or bake is valid with it.
- Whether it improves accuracy is not measured; that needs re-extracted training tables and refits.
- The cost of the affine-residual design is not measured; whether each design streams is in the
  next section. (An earlier version of this note said a coarse-scale fit would avoid a pre-pass.
  It would not: the streaming walk builds coarse rows from fine rows as they arrive.)

### Can the fixes stream?

The Rev5 scorer is a single streaming pass. `StripPlaneProducer` (`zensim/src/feature_v2_stream.rs`)
keeps rolling row windows per scale and builds each coarser scale from finer rows as they arrive.
It emits 128-row kernel strips top to bottom, and per-strip accumulators merge in a fixed reduction
tree, so strip geometry does not change the result. The timings above ran through this walk: the
raw 2048² profile (`perf.data`, see the timing README) has samples in `StripPlaneProducer`,
`RollingPlane` and `stream_phase_b`.

| Design | Streams? |
|---|---|
| High-pass ringing | Yes, unchanged: the same 4-neighbourhood inside the same strip halo. |
| Global-change descriptor used as a non-negative gate | Yes, in the same pass. The affine fit `dst ≈ a·src + b` per channel, and the share of the error it explains, need only five sums (Σs, Σd, Σs², Σd², Σs·d) that add across strips in the existing reduction tree. The descriptor is final when the walk ends, which is when the model runs anyway. |
| Damage bank recomputed on the global affine residual | Not in one pass. The fit must exist before the damage kernels run, and every scale's rows arrive top to bottom. It needs a moment pre-pass. |
| Damage bank on a local affine residual (window means and variances, as in SSIM's structure term) | Yes, within the existing halo. |

Notes:
- The steering map would need the gate descriptor's gradient as an extra term; whether the map's
  retained per-strip data covers it is not checked.
- A pre-pass keeps memory bounded: `ImageSource` gives random access to borrowed rows, so it can
  accumulate moments without materializing planes. Its cost is not measured. It would repeat the
  sRGB→XYB conversion, which took 13.2% of all samples in a single-thread 2048² profile of this
  scorer (`benchmarks/byconstruction_timing_2026-10-22/perf_2048_symbols.txt`; about 8% of the
  samples are the benchmark's own image synthesis). A subsampled pre-pass converts proportionally
  fewer pixels. In reference-fed encoder loops the reference pyramid is already cached, so only the
  distorted side would need it.
- A fit taken from the strips seen so far is not an option: the top of the image would use a
  different fit than the bottom.
- Whether a local residual can share the SSIM pass's local moments is not checked.

### Ringing corrected (`ZENSIM_RING_FORM=hpm`): implementation, cost, retraining

Owner direction (2026-10-23): correct ringing with performance parity, train and test, then look for the next
feature improvement.

**What changed** (`08930023`, `980fde3f`). `RingForm` in `zensim/src/feature_v2.rs` is a per-process
measurement arm, selected with `ZENSIM_RING_FORM` and modelled on `ZENSIM_HF_GAIN`. `raw` (the default)
keeps every stored value. `hpm` measures the ringing error as `(src − mu1) − (dst − mu2)`, where `mu1`/`mu2`
are the SSIM local means the dense kernel already holds for the strip. The three saturations are unchanged.
The form is a const generic on the canonical gradient kernels, so the raw instantiations compile to the
previous code. The diffmap and both attribution passes follow the form. Local refinement refuses `hpm`,
because a local change moves the local means over the blur radius rather than the ±1 dilation it patches.
The first version (`hp4`, `08930023`) used the 4-neighbour mean instead; it cost about eight vector
operations per 8 pixels and is superseded.

Checks:
- B's 512² timing pair scores 88.280549 under `raw` with the old and new binaries, and 88.281888 under `hpm`.
- Vector against scalar canonical body is bit-identical on scalar, v3, v4 and v4x for both forms
  (extended `gradient_canon64_vec_matches_scalar_on_every_compiled_tier`). New tests cover offset
  invariance, noise response and an independent f64 per-pixel sum. `cargo test --release -p zensim
  --all-features`: 907 passed, 0 failed, 27 ignored. CI-exact clippy clean.
- Rev5 extraction under `raw` with the new extractor reproduces the stored bank on all 576 requested
  slots (600 safesyn rows across jpg/webp/avif/jxl, 600 cid22_a25 rows). Under `hpm`, every one of the
  eight training/assessment sets differs from the stored bank in exactly the 12 ringing slots.

**Cost, and a 9% speedup found on the way.** All timings: `byc_score_timing` (B's served bake, deterministic
textured pair), paired rounds with alternating order, pinned core(s), dev (Zen 5); raw rounds in
`benchmarks/byconstruction_timing_2026-10-23/`.

1. The first form (`hp4`, 4-neighbour mean, `08930023`) added about eight vector operations per 8 pixels: +0.97%
   at 512² and +0.55% at 2048² single-thread against the old binary (folding three divisions into one saved nothing).
2. The `hpm` form measured +3.0% (512²) and +3.5% (2048²) against the old code, and that was robust: five lld
   link-order shuffles (`--shuffle-sections`, SD 0.1–0.2 points), uniform function alignment 2^5–2^7, and
   environment padding all left it in place. The phase timer (`ZENSIM_FOLD_TIMING`) put all of it in the dense
   kernel (+10–14%), whose machine code is byte-identical; exact counts showed the same instructions (−0.3%), +4.5%
   cycles and +40% op-cache misses. Bisection on the source found the trigger: passing the `mu` planes to the
   gradient call at all (a literal `None` there was at parity across shuffles), even though the dense kernel runs
   first and reads the same data.
3. Mechanism: `Lanes16F64` (`[f64; 16]`, 8-byte aligned) holds the dense kernel's accumulators, which it adds in
   64-byte chunks every few pixels; where the array falls relative to a cache line depends on the caller's frame.
   `#[repr(align(64))]` on `Lanes16F64` and `LanesF64` (`ec764cd7`; values unchanged, scores bit-identical, full
   suite passes) made the **old** code 9.32% faster at 2048² (CI −9.63..−8.98) and removed the sensitivity.

Final measurement at `ec764cd7`, five lld link-order shuffles per comparison (`scripts/byconstruction/final_shuffle.sh`,
mean paired Δ, SD over seeds), single thread unless noted:

| Comparison | 2048² | 512² | 64² (seed 1) | 2048², 8 threads (seed 1) |
|---|---|---|---|---|
| shipped code → corrected, `hpm` | **−8.45% (0.38)** | **−7.45% (0.10)** | −1.40% (CI −1.49..−1.28) | −6.49% (CI −6.85..−4.90) |
| shipped code → corrected, `raw` | −8.85% (0.21) | | | |
| aligned old code → corrected, `hpm` (the ring form's own cost) | +0.49% (0.21) | +0.57% (0.19) | +1.05% (CI +0.75..+1.23) | +0.07% (CI −0.34..+1.69) |

So the corrected scorer is faster than the shipped one at every measured size and thread count, because of the
alignment fix; the ring form by itself costs about 0.5% (1% at 64²), positive on every shuffle seed. Single host,
one synthetic content type, x86 v4x dispatch only.

Lesson for timing this code: whole-walk comparisons between two builds can move 3–10% with nothing changed in the
kernel that moves. Use several `--shuffle-sections` seeds, exact `perf stat` counts and the phase timer before
crediting or blaming a change. Uniform function alignment does not randomize placement (relative order is kept).

**Training root.** `scripts/byconstruction/ring_bank.sh` re-extracted every set of the v40r4 `v2e29` root
(kadid train/select, tid2013, konfig train/val, cid22_a25, cid22_train, safesyn; 1,037 s on 24 threads,
peak RSS 7.05 GiB) and the E15 coverage-pool selection (44,000 pairs) under `hpm` with the pinned
decoders. `scripts/byconstruction/ring_root.py build` wrote a derived root
(`/mnt/v/output/zensim/byconstruction-2026-10-22/ringhpm/root`). In every table, all 564 non-ringing
measured columns are bit-identical to the stored root, and only the 12 ringing columns change. The
derived feature-set id `basic+peaks+v2@w1825/rev5_localwin_ringhpm#36c3f3af` is appended to the registry.
Strict admission (Python and the Rust trainer) accepts it only on declarations that record
`byc_derived_from`.

**Feature probe under `hpm`** (`feature_probe.py --all-families`, same 24 references; ringing summed over its
12 cells, median over references):

| | +8 shift | contrast ×0.85 | sat ×1.25 | sat ×0.8 | blur σ1.2 | luma noise σ4 | chroma noise σ8 |
|---|---|---|---|---|---|---|---|
| ringing, raw | 0.334 | 0.231 | 0.187 | 0.160 | 0.139 | 0.083 | 0.203 |
| ringing, `hpm` | **0.017** | 0.130 | 0.075 | 0.063 | 0.143 | 0.082 | 0.185 |

Every other family reproduces the earlier probe values exactly.

**Retrained on the derived root** (`scripts/byconstruction/ring_arms.sh`, trainer `35e90254`, 12 E21 cells per arm,
seeds 0–2; scored against E34 `base-c` by `byc_cells.py score`; mixture test on the same 1,152 steps, scored under
`ZENSIM_RING_FORM=hpm` with the `35e90254` scorer, which reproduces all 1,344 stored raw Bnr mixture scores exactly
under `raw`):

| Arm | E21 Δ (SE) | KADID | TID2013 | KonFiG | CID22-A | Mixture inversions / cell (max rise) |
|---|---|---|---|---|---|---|
| Bhpm: B's signed recipe | **−0.0002 (0.0023)** | −0.0037 | −0.0026 | +0.0045 | +0.0009 | 78–184 (+53.8) |
| FRhpm: monotone, ringing sign-free | −0.0042 (0.0029) | −0.0067 | −0.0027 | −0.0131 | +0.0058 | 6–70 (+35.8) |
| Mhpm: certified monotone max pairs | −0.0158 (0.0017) | −0.0159 | −0.0302 | −0.0183 | +0.0014 | 2–35 (+8.0) |
| for reference: Bnr | −0.0003 (0.0009) | −0.0010 | −0.0023 | +0.0041 | −0.0019 | 72–178 (+53.1) |
| for reference: FR (raw ringing free) | −0.0039 (0.0029) | −0.0046 | −0.0153 | −0.0043 | +0.0085 | 2–41 (+26.2) |
| for reference: M (raw ringing) | −0.0144 (0.0016) | | | | | 1–32 (+6.5) |

- B's recipe is at parity on the corrected feature (E33 point form passes) and keeps its inversion: blur σ1.2
  then chroma noise σ8 inverts on 48 of 48 references in every cell, as in `base-c` and Bnr.
- In the certified-monotone class the correction alone does not move E21: Mhpm −0.0158 against M −0.0144, under one
  seed-unit SE apart. Its mixture behaviour is unchanged too (worst: blur σ1.2 then luma σ4, small rises).
- Freeing the corrected ringing still recovers most of the monotone gap (T −0.0205) and still opens the
  chroma-noise route (blur + chroma σ8 inverts on up to 38 references). The high-pass error still responds to
  noise, so a sign-free ringing input can still be used to forgive it. Inside the monotone class ringing has to
  count as damage; the sign-free channel needs a signal noise cannot drive.


### Next feature: a global affine-change descriptor (`byc_globaff`)

**Why this one.** Under `hpm` the probe flags one remaining family by the same criterion (response to a
clipping-free global edit above a visible blur): the JND transducers, which also use the raw per-pixel error
(`pjnd_transducer` contrast ×0.85 0.494 vs blur 0.357; +8 shift 0.705). Error-magnitude families measure global
changes by definition, and DG showed dropping them costs more than it saves. FR and FRhpm show the monotone class
recovers most of its gap when it gets a sign-free signal it can subtract, and that ringing is the wrong such signal
because noise drives it. So the next feature is an explicit descriptor that a global change drives and noise does
not: per scale and XYB channel, the share of the mean squared error that a least-squares global fit
`dst ≈ a·src + b` explains, and that explained error (24 values). It streams in one pass (five additive sums per
channel; see "Can the fixes stream?").

**Implementation (research only).** `zensim-bench/examples/byc_globaff.rs` decodes with the shared decode owner,
converts with zensim's own XYB conversion and builds the 4-scale pyramid with zensim's own 2× downscale (two-pass f64
moments). It is a sidecar at research ids f1867–f1890, registered as
`basic+peaks+v2@w1891/rev5_localwin_ringhpm_globaff#d56ad477` (hash from the new `byc_fsid` bin, which reproduces the
registered `#36c3f3af`); zensim cannot compute these ids, so such bakes cannot be served. Training reads them from
a widened derived root (`ring_root.py globaff`); predictions use `byc_predict` (zenpredict, the trainer's row
layout), which reproduces all 7,869 stored `bake_dial_refit --score-units` predictions of a Bhpm cell bit for bit;
the mixture test uses a table of dumped feature vectors plus the descriptor, which reproduces the pixel-route
scores of a Bhpm cell exactly on all 1,344 images. 271,590 training pairs plus the pool: 482 s on 20 threads.

**Label-free check on the probe images** (median share at scale 0, X / Y / B, 24 references):

| Edit | Share |
|---|---|
| +8 luma shift | 0.37 / **0.997** / 0.22 |
| contrast ×0.85 | 0.23 / **0.997** / 0.16 |
| saturation ×1.25 | **0.97** / 0.35 / **0.94** |
| saturation ×0.8 | **0.99** / 0.41 / **0.97** |
| blur σ1.2 | 0.17 / 0.16 / 0.15 |
| luma noise σ4 | 0.003 / 0.000 / 0.004 |
| chroma noise σ8 | 0.003 / 0.14 / 0.002 |

(At the coarsest scale chroma noise leaves a tiny Y error that is mostly "explained", share 0.85 but explained
energy 4e-6; that is why both numbers are inputs.)

**Arms** (`globaff_arms.sh`, `globaff_arms2.sh`; trainer and scorers `58a4d100`; 12 E21 cells each):

| Arm | Class | E21 Δ (SE) | KADID | TID2013 | KonFiG | CID22-A | Mixture inversions / cell (max rise) |
|---|---|---|---|---|---|---|---|
| **Tghpm** | monotone in all 820 damage inputs, descriptor sign-free | **−0.0101 (0.0024)** | −0.0066 | −0.0236 | −0.0175 | +0.0071 | **2–7 (+7.1)** |
| Thpm (control) | monotone, no descriptor | −0.0221 (0.0032) | −0.0192 | −0.0452 | −0.0296 | +0.0055 | 0–4 (+4.2) |
| Bghpm | B's signed recipe plus the descriptor | −0.0031 (0.0017) | −0.0010 | −0.0033 | −0.0044 | −0.0037 | 70–190 (+61.4) |

- The descriptor recovers about half of the monotone class's E21 gap (Thpm −0.0221 → Tghpm −0.0101) **without** the
  chroma-noise route: Tghpm's only inversions are posterize then light noise (dithering, at most 3 references per
  cell), the same benign pattern as T. Sign-free ringing recovered more (FRhpm −0.0042) but inverts blur + chroma
  noise on up to 38 of 48 references. Tghpm has the best E21 of the arms that keep inversions bounded, ahead of the
  certified-monotone M64 (−0.0131, 1–6 inversions) and M (−0.0144, blur + luma σ4 on up to 24 references); its
  0.003 margin over M64 is about one seed-unit SE.
- Tghpm's guarantee is weaker than M's: it is monotone in every damage input, but the descriptor is free, so the
  monotonicity is not certified end to end. The descriptor is built to fall when unexplained error is added, and
  on the 1,152 mixture steps that held.
- Adding the descriptor to B's signed recipe does not help (−0.0031, E33 point form fails) and leaves B's
  inversions (blur + chroma σ8 on 48 of 48 references).
- Remaining gap: TID2013 −0.0236 and KonFiG −0.0175. Not tested: combining the descriptor with max pairs (the
  free-mask route refuses `mp`), gating products instead of sign-free inputs, and a served (streaming) version.

### AIC2026 TRAIN: how far image features go toward SROCC 0.99 (2026-10-23)

**Goal (owner, 2026-10-23):** convex, monotone scoring across every distortion type and dataset, and SROCC above
0.99 on AIC2026 TRAIN, with less compute and without memorization or overfitting.

**Population.** AIC2026 TRAIN is WG1's `public_train` split (document m112107, preliminary public-split JND
reconstruction, CSV SHA-256 `bd5ea3c5…`, tower IQA library `jpeg-aic2026`): **3,920 PTC crops of 28 sources**,
label `JND_score` (lower is better; quality SROCC uses −JND). `public_test` is the other 1,848 crops of 14 sources;
28 further sources have no labels. The first half of this study used the owner's separately supplied overlay
`jnd_reconstruction_2026-10-19.csv` (`JND_curve`) over all 42 labelled sources before the split was found; those
results are kept below and labelled "42-source pool". `JND_curve` and `JND_score` are two reconstructions of the
same study and agree at SROCC **0.9835** on `public_train` (0.986 on `public_test`); their difference grows with the
distortion level (σ 0.07 JND at level 1, 0.50 at level 20; `JND_score` is higher at the top).

**Role and exposure.** The owner's goal names the split TRAIN, so this branch fits on it, and **every AIC2026
number reported is either a frozen model (no fit on AIC2026) or out of fold** (sources split into folds; a source's
rows are predicted by a model fitted without any row of that source); in-sample numbers are never reported. The
42-source pool phase fitted the `JND_curve` labels of the 14 `public_test` sources inside its folds, so `public_test`
informed this study's design choices and is not a clean test for them; it is read only to assess frozen candidates.
The 28 unlabelled sources and every other protected population stay untouched. Ledger: `docs/DATA_SPLITS.md`,
2026-10-23. All 5,810 crop files on dev were SHA-256-equal to the 2026-10-19 admission before any use.

Tooling: `mixture_table.py build-pairs` (Rev5 vector + descriptor table; the table route reproduces all 5,768
stored B scores bit for bit), `aic_train.py` (zero-shot, out-of-fold fits, `analyze`), `jnd_band_proto.py` (band
feature prototypes), `aic_rust_cv.py` (the same folds through the Rust trainer), `byc_cells.py --aic-weight` (E21
cells that also fit the split). Data: `/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train/` on dev
(`folds_train/`: the split, its 7 source folds, pairs lists and the WG1 CSV).

**`public_train`, WG1 `JND_score`:**

| Model | Fit on AIC2026 | Global SROCC | Within source | Within ladder |
|---|---|---|---|---|
| B (`diff+frag-820`) | no | 0.9091 | 0.9558 | 0.9995 |
| full-data M (certified monotone) | no | 0.9108 | 0.9508 | 0.9998 |
| full-data T (convex monotone) | no | **0.9115** | 0.9494 | 0.9997 |
| NNLS linear on fx1g (844 inputs) | out of fold, 7 folds × 3 | 0.934 | 0.959 | 0.999 |
| additive hinges, **convex** monotone | out of fold | 0.939 | 0.963 | 0.997 |
| additive hinges + ramps, monotone | out of fold | 0.939 | 0.967 | 0.999 |
| gradient boosting, monotone | out of fold | 0.953 | 0.964 | 0.996 |
| gradient boosting, unconstrained | out of fold | 0.957 | 0.967 | 0.997 |
| fx1g + all 75 supplied peer metrics, monotone boosting | out of fold | 0.964 | — | — |
| fx1g + all 75 supplied peer metrics, unconstrained boosting | out of fold | **0.968** | — | — |

**Through the Rust trainer** (`aic_rust_cv.py`, AIC2026 alone, global rank loss on the split, 60 epochs × 20k pairs,
same 7 folds; `58a4d100` trainer): certified convex monotone **T 0.9527**, convex monotone max pairs M 0.9512, signed
S 0.9566 (within source 0.968 / 0.968 / 0.969). The servable convex monotone class matches monotone boosting.
**Memorization gap:** T fitted on all of `public_train` scores 0.9743 on it in sample, 0.022 above its out-of-fold
0.9527 (source-level offsets it can learn but not transfer). Even in sample this class does not reach 0.99.

The last two rows are the measured information ceiling of every full-reference measurement available here (our 844
inputs plus 75 published metrics, DVIFM excluded because its constants were trained on AIC2026): **0.968, 0.022
below the target**, and the two label reconstructions agree with each other only at 0.9835.

**E21 cells that also fit `public_train`** (`aic_arms.sh`; `byc_cells.py --aic-weight W`): every E21 cell adds an
AIC2026 group (global rank loss over the split, research replay) without one of the 7 source folds and predicts that
fold; seeds 0–2 cover every fold (`aicscore` pools the held-out predictions; a fold held out by two cells takes their
mean). E21 Δ is against `base-c` as everywhere in this record; mixture test as above (1,152 steps per cell).

| Arm | Class | AIC weight | E21 Δ (SE) | KADID / TID / KonFiG / CID22-A | AIC `public_train` OOF | Mixture inversions / cell (max rise) |
|---|---|---|---|---|---|---|
| Tghpm (no AIC leg) | monotone in damage inputs, descriptor sign-free | — | −0.0101 (0.0024) | −0.007 / −0.024 / −0.018 / +0.007 | — | 2–7 (+7.1) |
| **Tga** | same | 32 | **−0.0089 (0.0038)** | −0.012 / −0.023 / −0.011 / +0.010 | **0.932** (within source 0.958) | **1–4 (+4.1)**, dithering only |

| Ma | certified convex monotone max pairs (descriptor constrained) | 32 | −0.0172 (0.0053) | −0.017 / −0.039 / −0.019 / +0.007 | 0.934 (within source 0.958) | 2–5 (+6.3) |
| Bga | B's signed recipe | 32 | −0.0031 (0.0049) | −0.006 / −0.005 / +0.002 / −0.003 | 0.949 (within source 0.968) | 52–211 (+62.6) |

The signed class takes the AIC leg best (Bga: E21 at parity with `base-c`, AIC 0.949) and keeps B's inversions
(blur + chroma noise); the monotone arms give up about 0.015 on AIC and 0.006–0.014 on E21 to stay inversion-free.

**Candidate rule, committed (04743622) before the results of Ta, Ts1a, Mgfa and Tga128.** One combined arm becomes
the frozen candidate: among arms with at most 10 mixture inversions per cell and E21 Δ ≥ −0.010 against `base-c`, the
highest AIC public_train out-of-fold SROCC; ties (within 0.002) go to the better E21 Δ. Its full-data fit
(`aic_full.sh`, seed 0) is then read once on public_test and once in sample on public_train. No other model is read on
public_test.

Tghpm's 12 cells (no AIC leg) score 0.907 on `public_train` zero-shot (range 0.904–0.909, as B's 0.909): the AIC leg
is worth +0.025 out of fold. Per codec, Tga beats B on 13 of 14 codecs out of fold (JPEG XL modular 0.801 → 0.833,
JPEG 2000 0.890 → 0.932, WebP 0.908 → 0.950; VVC 0.978 → 0.967), and it is more even than the AIC-only T, which
loses VVC (0.908), PNG and HEVC: the E21 legs act against fitting the five common codecs.

**Mohammadi 2025 panel per dataset and split** (`aic_panel.py`, native `panel --batch`; JSON
`aic2026train/panel_mohammadi.json`): for each E21 set, out of sample = the LODO cells trained without it (seeds 0–2,
mean), in sample = a seed-0 cell trained with it. In-sample minus out-of-sample SROCC, B's recipe (`base-c`): TID2013
0.940 vs 0.901, KonFiG 0.875 vs 0.817, CID22-A 0.881 vs 0.880; Tga: TID2013 0.890 vs 0.879, KonFiG 0.819 vs 0.807, CID22-A
0.892 vs 0.890. The monotone models gain less from seeing a dataset than B's recipe does. AIC2026 public_train, T: in
sample 0.974 / out of fold 0.953 (PLCC 0.964 / 0.933, Z-RMSE 0.265 / 0.359).

**AIC2026 test data used (counted from the commands run, 2026-10-23).** Before the split was found, the 14 public_test
sources were inside the 42-source pool with the overlay labels: 1 frozen-model read, 10 descriptive label reads, 32 Python
source-fold runs and 4 Rust trainer runs that fitted and scored them. After the split was found, the official public_test
labels were read for frozen B once (SROCC 0.9255, with the reconstruction-agreement check), once more for the panel on the
same predictions, and once for per-source label statistics that were then left off the dashboard. No fit, model choice
or candidate read used them.

**42-source pool (`JND_curve` overlay; the first phase).**

**Zero-shot (no AIC2026 fit), global quality SROCC / mean within reference / mean within ladder:**

| Model | Global | Within reference | Within ladder |
|---|---|---|---|
| B (`diff+frag-820`) | 0.9113 | 0.9672 | 0.9993 |
| full-data T (convex monotone) | **0.9153** | 0.9625 | 0.9994 |
| full-data M (certified monotone max pairs) | 0.9104 | 0.9605 | 0.9995 |

The monotone models are at parity with B on these codec ladders, as on CID22-A. Every model orders each ladder
almost perfectly; the whole problem is placing different codecs and contents on one JND scale.

**Out-of-fold fits on B's fx1 inputs plus the descriptor (844 inputs), mean of 3 fold draws:**

| Model | Guarantee | Global | Within reference |
|---|---|---|---|
| NNLS linear | monotone | 0.950 | 0.971 |
| additive hinges (NNLS) | **convex** monotone | 0.962 | 0.973 |
| additive hinges + ramps | monotone | 0.963 | 0.975 |
| gradient boosting, monotone constraints | monotone | 0.965 | 0.975 |
| gradient boosting | none | 0.969 | 0.977 |

**The ceiling is information, not data or learner.**

- Learning curve (monotone boosting): 21 training references 0.961, 28: 0.964, 35: 0.966, 41: 0.966. More references
  like these do not help.
- Target transforms (log, square root, rank) change nothing (0.962–0.967).
- New feature families measured in Python (`jnd_band_proto.py`: masking-normalised Laplacian band errors at 7 scales,
  tile-max/top-10% pooling, band-energy gain vs loss): the best single feature is luma scale-2 band-energy loss,
  top-tile pooled, 0.932 (B's own best single input, `det_4th` luma scale 2, is 0.929; B itself 0.911). Alone the
  families reach 0.955–0.960; added to fx1 they reach 0.966 (monotone) and 0.971 (unconstrained).
- Each of 70+ supplied peer columns added to fx1 gains at most +0.005 (AIMOS, FSIM, PieAPP, butteraugli). **All 75
  supplied peers together with fx1 reach 0.974 (monotone boosting) / 0.977 (unconstrained)**; the peers alone 0.967 /
  0.974. (DVIFM columns excluded: their author constants were trained on AIC2026.)
- No reference-content descriptor (mean luminance, chroma, flat-area fractions, band energies) correlates with the
  per-source residual beyond |ρ| = 0.23 over 42 sources.

**The labels support 0.99.** The supplied pointwise confidence intervals are honest: (pointwise − curve)/pointwise
σ has SD 1.13. The curve's own σ is 0.073 JND, about the pointwise σ over √7, as a 7-point fit should be.
Resampling the curve labels from their intervals, independently or shared along each ladder, gives SROCC 0.998. So
the label noise is about 0.07 JND, while the best out-of-fold error is about 0.33 JND (RMSE).

**Where the error is** (monotone boosting, `aic_train.py analyze`):

| Oracle (applied to out-of-fold predictions) | Global SROCC |
|---|---|
| none | 0.9655 |
| remove the mean residual of each codec | 0.9672 |
| remove the mean residual of each source | 0.9723 |
| remove the mean residual of each (source, codec) ladder | 0.9834 |
| perfect shape within each ladder, ladder offsets kept | 0.9678 |

98% of discordant pairs are cross-source, and 62% are pairs whose labels differ by less than 0.25 JND (discordant
34% of the time). SROCC 0.99 needs discordance to fall from 8.2% to about 3.5%: the prediction error has to approach
the label noise. The largest single errors are ladders on a few contents: JPEG / JPEG 2000 / JPEG XL on a white
product shot (S55, up to 1.3 JND too lenient: block quantisation of faint shading in a flat background), most codecs
on a saturated fish on bokeh (S45), and JPEG / JPEG AI on S70 (too harsh).

**Compute.** Dropping every scale-0 slot (three quarters of the pixels) leaves the monotone linear and convex
additive models unchanged on the 42-source pool (0.950 → 0.950, 0.962 → 0.962); monotone boosting loses 0.006. On
`public_train` (7 folds × 3): NNLS 0.9344 → 0.9350, convex additive 0.9389 → 0.9373, monotone boosting 0.9529 → 0.9505. B's
own selection reads only 41 scale-0 slots of its 410 direct inputs (`fx1gs1` = fx1g without them: 387 direct, 369
products). Exact per-call user-space instructions of `BakeScorer::compute` (`perf stat`, slope between two iteration
counts so setup cancels, one thread, the `byc_score_timing` synthetic pair, two 1-epoch timing-only bakes over B's fx1
read set and over fx1 without scale 0):

| Size | fx1 read set | fx1 without scale 0 | Change |
|---|---|---|---|
| 512² | 95,488,070 | 75,092,745 | −21.4% |
| 2048² | 1,459,946,055 | 1,100,277,015 | −24.6% |

Less than the pixel share because the planner gates v2 per scale (`v2_scales`) but the v1 basic family as one switch:
a profile of the scale-1+ bake at 2048² still spends 35% of samples in the basic SSIM blurs (`fused_vblur_ssim`,
`fused_blur_h_ssim`, `box_blur_v_copy`) and 10% in `memmove`, at full resolution. A per-scale basic gate is the next
engine change if the E21 arms show scale 0 can go. Wall time was not measured (the box was running training).

A first attempt (`ComputeSet::v1_scales` from the declared ids; the streaming walk skips a scale with no v1-era read
and no v2-era work) is bit-identical on every requested slot in its own test, at the default revision and at Rev5,
and passes all 454 library tests. With all features on it fails 6 tests, all for one reason: the scalar walk skips
the dead scale while the attribution, session and refinement walks still compute it, so their full feature vectors
(unread slots included) disagree; one refinement golden also moves on a read slot (f488, local 1.0e-2 vs full 2.1e-5).
Profile plans needed every v1 scale kept (a profile's linear tail reads the v1 prefix undeclared). Not landed: every
walk must honour the skip first. Patch (261 lines, SHA-256 `0e3195eb…`):
`/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train/v1_scale_skip_attempt_2026-10-23.patch` on dev.

## Summary (all arms, E21 screen, seeds 0–2, Δ vs `base-c`)

| Arm | Guarantee | E21 Δ (SE) | Mixture inversions / cell (max rise) |
|---|---|---|---|
| Ts: B + thresholds | pin only | −0.0008 (0.0012) | — |
| Bhpm: B, ringing `hpm` | pin only | −0.0002 (0.0023) | 78–184 (+53.8) |
| Bghpm: B + descriptor | pin only | −0.0031 (0.0017) | 70–190 (+61.4) |
| Bnr: B without ringing | pin only | −0.0003 (0.0009) | 72–178 (+53.1) |
| FR: ringing sign-free | monotone in 800 of 820 | −0.0039 (0.0029) | 2–41 (+26.2) |
| FRhpm: `hpm` ringing sign-free | monotone in 800 of 820 | −0.0042 (0.0029) | 6–70 (+35.8) |
| **Tghpm: descriptor sign-free** | **monotone in all 820 damage inputs** | **−0.0101 (0.0024)** | **2–7 (+7.1)** |
| **M64: max pairs** | **certified monotone** | **−0.0131 (0.0023)** | 1–6 (+7.6) |
| **M: max pairs + hinges** | **certified monotone** | **−0.0144 (0.0016)** | 1–32 (+6.5) |
| Mhpm: M, ringing `hpm` | certified monotone | −0.0158 (0.0017) | 2–35 (+8.0) |
| E20: budget 20 | ≤ +20 certified | −0.0154 (0.0026) | 4–37 (+7.3) |
| E5: budget 5 | ≤ +5 certified | −0.0194 (0.0027) | 3–8 (+5.0) |
| G: content-gated | certified monotone | −0.0202 (0.0030) | 0–4 (+4.5) |
| Lc: linear | certified monotone | −0.0204 (0.0029) | — |
| T: thresholds | certified monotone | −0.0205 (0.0029) | 0–4 (+4.0) |
| Thpm: T, ringing `hpm` | certified monotone | −0.0221 (0.0032) | 0–4 (+4.2) |
| DR: ringing dropped | certified monotone | −0.0216 (0.0031) | 0–3 (+4.3) |
| R: ramps | certified monotone | −0.0225 (0.0024) | 4–9 (+5.2) |
| DG: global-sensitive dropped | certified monotone | −0.0229 (0.0032) | — |
| `base-c` (B's recipe) | pin only | 0 | 86–171 (+61.5) |

Full-data certified-monotone models (T, M) pass every label-free product gate; M: C1
0.9984 / 0.9935, NEARID 143/144, G-STEER 135/135, identity at all tiers.

**Answer to the question.** A metric that is monotone by construction is competitive with B
on the product contract (dial, identity, steering) and far more robust (inversions bounded to a
few points, no blur + chroma-noise failure), and at parity on codec data (CID22-A). It is not yet
competitive on the E21 cross-dataset rank statistic: the best certified-monotone arm is −0.013.
That gap is cross-type severity calibration on synthetic distortion sets; with oracle per-type
calibration the monotone models match B. Max-pooling recovers about a third of it inside the
monotone class. Inside the monotone class most of the rest runs through one conflated feature
(v2 ringing, imazen/zensim#72): freeing its sign alone recovers it (FR −0.0039), because ringing
doubles as a global-change detector. B itself does not depend on ringing (Bnr −0.0003, the same
inversions): its sign freedom finds that discount in other features and forgives chroma noise
with it. The high-pass ringing fix costs under 1% single-threaded. Closing the remaining gap by construction needs a feature
change: damage measured on the affine-residual difference plus explicit global-change terms
(the SSIM luminance/contrast/structure decomposition applied to this bank).

**Update 2026-10-23.** Ringing is corrected (`ZENSIM_RING_FORM=hpm`, research arm; raw stays the default and every
stored value is unchanged) and retrained. The correction does not change any model's E21 beyond noise (Bhpm −0.0002,
Mhpm −0.0158, Thpm −0.0221), and it does not remove B's inversion: that comes from sign freedom. The next feature, a
global affine-change descriptor that noise cannot drive, gives the monotone class a safe way to discount global
changes: Tghpm reaches −0.0101 with 2–7 mixture inversions per cell (dithering only), the best E21 of any arm that
keeps inversions bounded. It is a research sidecar today; serving it needs the descriptor in the streaming walk (five
additive sums per channel, see "Can the fixes stream?"). Timing the correction found a 9% speedup for the shipped
scorer (64-byte-aligned lane accumulators), so the corrected scorer is 7–8.5% faster than the shipped one; the ring
form itself costs about 0.5%.
