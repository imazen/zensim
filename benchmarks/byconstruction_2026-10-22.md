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
