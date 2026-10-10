# E33 — reference-only inputs may only scale differences (registration)

Status: **REGISTERED 2026-10-09 by owner approval, before any E33
implementation, fit or data read. Not yet authorized to launch** (launch needs
the implementation, smokes and freeze in §10). Drafted on zensim
`main@origin` `ecc87ac0496a` (draft commit `919634b2eccd`, rebased from local `24eaceef1cc1`). The
machine-readable form is
[e33_registration_2026-10-09.json](e33_registration_2026-10-09.json).

## 0. Owner approval (verbatim, 2026-10-09)

> "approved, all of it"

Given in reply to the draft summary (`E33_DRAFT_DONE.md`). It approves this
registration as drafted, including every listed open choice:

| Choice | Registered value |
|---|---|
| Near-identity bar (N1/N2) | 99.0 |
| Tail length factor (§8 step 3) | 2 |
| Runtime guard (§9.3) | no cell slower by a paired CI wholly above +2% |
| Tie rule (§9.4) | A wins ties over C (C needs the improvement test) |
| Candidate (§9.2) | full-data seed index 0; seeds 1–2 report-only |
| Steering non-regression bar | ≥ 128 of 135 (qualification still needs 135) |

The coordinator has corrected the readiness artifact's "plain MLP" claim
(§2). The rules below are unchanged from the draft.

## 1. Owner decisions this rests on (verbatim, 2026-10-09)

> "Shouldn't we ensure inputs that describe only the reference image or distort
> image only have a multiplicative or exponential impact with no constant sum?
> Anchors have a history of being a problem, right? And output shaping is
> presumably already done? And for internal numbers, doesn't 0 mean no
> difference, and positive numbers mean difference, with us refitting to
> reverse and start from 100?"

> "Would multiplying heads work for our case? Or feeding the scalar into a 2nd
> mlp to combine?" … "can we constrain during training some features from being
> involved in non-mulitplicative operations with other values?" … "So do you
> suggest adding features that are duplicates but fragility scaled?"

Coordinator answer, accepted: keep each difference `d` and add `d × fragility`
(fragility can approach 0, so a pure product would erase sensitivity on
textured references). "Yes I approve" covers drafting this registration and
closing the zenpredict graph PR (imazen/zenanalyze#89, closed).

## 2. A correction this draft depends on

The owner-facing proposal (claude.ai artifact "by_v2fy Rev5 Readiness",
section "Proposal: next candidate") says the production model "is a plain MLP
with bias terms" and that `--nonneg-distance` would be new. **That is wrong.**
Production seed 0 is head `N`, and `v2_lodo_mlp.py` appends
`--nonneg-distance` for head `N`
(`scripts/rev4_featpot/v2_lodo_mlp.py`, `train_command`). The production
cell's own log confirms it
(`fitv2d1-20261007/…__N/full_s0/train.log`: `nonneg_distance: true,
nonneg_pin: 100.0`, `scaler_mean zeroed`, ReLU, `feature_transforms: None`).
The frozen V40 control is the same recipe.

What follows from that:

* The production network is already `raw = 100 − g(x)`, `g ≥ 0`, `g(0) = 0`,
  no hidden biases, no feature transforms. The September cost of the
  constraint (CID22 0.890 → 0.882 at one seed; best-of-all §5.5: −0.0091 CID22
  at k=3) is **already paid** by the control. E33 does not re-measure it.
* The near-identity gap comes from **one** remaining violation of the
  zero-means-no-difference rule: the ten reference-only `pjnd_fragility`
  inputs enter `g` like differences. For a perfect copy, `g(0, f) > 0`, so the
  raw identity value is 87.58–97.59 by reference before the spline
  (NEARID). With those ten inputs zeroed, raw identity is the pin, and the
  current spline maps it to 99.81 (NEARID). The pin lies above the spline's
  top knot (the packed calibration rows span raw 15.81–92.18 in `pack.log`),
  so the linear upper extrapolation decides the identity score.
* So Arm A is "the control without the ten fragility inputs", and Arm C is
  "the control with fragility allowed only inside products with differences".
  Both inherit `--nonneg-distance` unchanged.

The artifact text is the coordinator's; this draft does not edit it. The
correction is listed for the owner in `E33_DRAFT_DONE.md`.

## 3. Hypotheses

* **H-A.** Removing the additive reference-only path keeps human rank within
  the E21 as-good tolerances, and makes the near-identity scale continuous up
  to 100.
* **H-C.** Letting fragility act only as a per-reference gain on differences
  (`d × f`) recovers whatever rank information the additive path carried,
  without any constant offset at identity.
* **H-T.** With the output spline pinned at the identity and given a strictly
  increasing tail, floor ties disappear, so the four heavy-JPEG steering cases
  and the C2 floor ties are no longer tied by construction.

A structural fact both arms share: with zero hidden biases and no transforms,
`g` is positively homogeneous of degree 1 in its inputs. In A, and in C for a
fixed reference (fragility fixed), scaling every difference by `t ≥ 0` scales
`g` by `t`. Reference-only information can then change *how fast* distance
grows with difference, never the distance at zero difference. That is the
"multiplicative impact with no constant sum" the owner asked for. `g` stays
convex in the standardized inputs, a restriction the control already has.

By_v2fy has **no** distorted-only inputs: all 410 non-fragility inputs are
`Form::Difference` (registry, below). Nothing analogous is needed for the
distorted side.

This ablation removes no extraction work (fragility comes out of the same v2
pass). It qualifies under the playbook's exception because it settles a
consequential product issue: identity, near-identity addressability and floor
ties.

## 4. Data, roles and exposure

SDR only. HDR legs (E29 hc4, E31 uh4) are **not** added here. Arms and control
use one recipe on the D1 human population (owner decision D1, 2026-10-07):
KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL, CID22-A(25). AIC-3 is excluded
from fitting, packing, assessment and control discovery. The teacher legs
(SafeSyn, CID22-train SSIMULACRA2 oracle), the coverage leg (`cv16:cf98`) and
their weights are unchanged. Use the same fit-data archive as the V40 control,
`data_sha 9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008`.
It must be byte-identical; anything else is a launch refusal.

Reads, all already-exposed populations:

| Read | Population | Purpose |
|---|---|---|
| E21 assessment | the four D1 sources, each withheld from its fold's fit (13,817 observations per rotation, 10 rotations × 3 labels) | primary human rule |
| Label-free gates | 38-image identity probe, 2,000-row negative-tail probe, standard C2 grid (4,318 flat / 9,411 ladder rungs), NEARID's 24 TRAIN references × 27 rungs, the 135 G-STEER engineering cases (TRAIN/SELECT/diagnostic roles as in STEERFIX's `INPUT_RECEIPT.json`) | identity, ties, near-identity, steering |
| Calibration | `cid22_fit.parquet`, the existing TRAIN calibration population of `bake_dial_refit pack` | output spline |

**Not read:** KADID TERMINAL, AIC-3/AIC-4/SDR25, CID22 gold/human, KonJND,
sealed/T0, HDR VAL, UPIQ, external NITS/LIVE/MCIQA. External panels need a
separate exposure authorization; E33 does not request one. The exposure ledger
entry is written at assessment time, not now.

## 5. Arms

All arms: head `N` (`--nonneg-distance`, pin 100), hidden 128, human weight
32, coverage `cv16:cf98`, 120 epochs × 50,000 pairs, final epoch 119, the V40
seed streams (`v2_common.seeds`), `ZENSIM_MAX_TIER=v3`, formula revision 5. No
feature transforms, no `--identity-rows`, no anchor loss, no post-result
selection of features, seeds or epochs.

| Arm | Spec | Direct inputs | Products | Input width |
|---|---|---|---|---:|
| Control (fresh, §6) | `sel:59f0bbc2f290@h32:H128:cv16:cf98` | 410 differences + 10 fragility | none | 420 |
| **A** | `sel:3b7bd5ebe929@h32:H128:cv16:cf98` | 410 differences | none | 410 |
| **C** | `sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1` | 410 differences | 410 × `d·f` | 820 |

`sel:` ids are the existing `v2_common.selection_id` (first 12 hex of the
SHA-256 of the sorted comma-joined columns). The 420 list reproduces the
control's `59f0bbc2f290`. The 410 list's full SHA-256 is
`3b7bd5ebe929bb5d2549132078b4a7665ee91f1a15e5d5c1b76a800f3ed396b9`. `fx1` is
a new recipe token (§10) naming the product declaration below. Arm C has **no
raw fragility input**.

### 5.1 Pairing table (Arm C)

Fragility is `pjnd_fragility`, v2 signal 21, `Form::ReferenceOnly`,
`1 − saturate(mean |∇src|, 0.02) = 0.02 / (G + 0.02)`, so `f ∈ (0, 1]` and it
is finite for every input (`saturate` maps NaN to 0 via `f64::max`). It is
computed on every kept cell; `transducers_luma_only` is off in this recipe, and
ADJUDICATE measured all ten slots nonzero per row.

Each difference pairs with the fragility slot of **its own scale and channel**.
Basic and v2 share the cell index `scale·3 + channel` (X=0, Y=1, B=2); the v2
fragility ID of cell `k` is `372 + 29k + 21`.

| Cell | Fragility ID | Basic difference IDs | v2 difference IDs | n |
|---|---:|---|---|---:|
| s0 Y | 422 | 13–25 | 401–421, 423–429 | 41 |
| s1 X | 480 | 39–51 | 459–479, 481–487 | 41 |
| s1 Y | 509 | 52–64 | 488–508, 510–516 | 41 |
| s1 B | 538 | 65–77 | 517–537, 539–545 | 41 |
| s2 X | 567 | 78–90 | 546–566, 568–574 | 41 |
| s2 Y | 596 | 91–103 | 575–595, 597–603 | 41 |
| s2 B | 625 | 104–116 | 604–624, 626–632 | 41 |
| s3 X | 654 | 117–129 | 633–653, 655–661 | 41 |
| s3 Y | 683 | 130–142 | 662–682, 684–690 | 41 |
| s3 B | 712 | 143–155 | 691–711, 713–719 | 41 |

The fragility IDs equal NEARID's counterfactual list. The JSON holds all 410
rows (ID, signal, scale, channel, fragility ID). Family, form, direction and
positions follow from the rules stated there.

**Input order.** Positions 0–409 hold the 410 differences in ascending ID
order. Position `410 + i` holds `d_i × f(cell(d_i))`.

**Arithmetic.** Each product is the IEEE f32 multiply of the two
f32-stored factors (the same f32 values the declared storage round trip
gives, which ADJUDICATE verified bit-for-bit against canonical extraction). The
trainer, `bake_dial_refit densify/predict` and serving all compute it through
one shared owner (§10).

**Unmatched differences.** A difference has no partner when no `pjnd_fragility`
slot exists at its `(scale, channel)` in the computed layout, for example a
`PerScale`, `Flat` or `NativeXb` block. Then it enters as `d` only, with no
product, and the declaration lists it explicitly as unpaired. A partner is
taken from the layout whether or not it is a direct input. For this keep list,
**0 of 410 are unpaired.** Pairing across scales or channels, or pairing with
any other ReferenceOnly signal (`grad_src_mean`, `luma_mean_ref`), is outside
`fx1`.

### 5.2 Why no `d × f²`

Decision: **`d × f²` is not in E33.**

1. `d·f = d / (1 + G/0.02)` is already the divisive-normalization masking form.
   For each hidden unit, `w·d + w′·d·f = d·(w + w′f)`, a reference-dependent
   gain affine in `f`. Across 128 ReLU units the network combines many such
   gains. `f²` adds curvature in the gain only.
2. On (0, 1], `f` and `f²` are strongly collinear. For `f` uniform on (0, 1],
   corr(f, f²) = √15/4 ≈ 0.968 (analytic). The TRAIN distribution of `f` was
   not measured; preparation reports the TRAIN correlation of `f` and `f²`
   descriptively. It does not reopen this choice.
3. Width and capacity: first-layer weights are 420×128 = 53,760 (control),
   410×128 = 52,480 (A), 820×128 = 104,960 (C), and would be 1,230×128 =
   157,440 with `f²`, trained on the same human rows (three D1 sources per
   fold) plus the teacher and coverage legs.
4. One product arm keeps one structural question per arm.

An `f²` arm becomes a separate later registration (E33b) only if C is adopted
and a named question needs it.

## 6. Control: fresh, not reused

Brief rule: reuse the frozen V40 control (`V40_CONTROL_PINS.json`, SHA-256
`4d7acfc887603b123f9631f38435df5477b6e628a372596cb8beb6128bddc84c`, program
`de44ea9b1ff85031265dfa1d60731de47f177d6816fb0880f85d28f38c405517`) only if the
program, data and recipe pins are identical.

**They will not be identical for Arm C.** C needs product inputs in the
trainer, in `bake_dial_refit densify/predict` (the E21 prediction path) and in
serving. That is a new program pin for both fitting and assessment. Arm A by
itself could run on program `de44ea9b` (a `sel:` subset is already supported).
That would compare A and C against controls built by different programs,
though, and leave A-vs-C confounded with the program change.

Registered decision, made before any E33 fit:

* One E33 program carries the `fx1` extension. One **fresh matched control**
  (40 cells: kadid / tid2013 / konfig / cid22_a25 × seed indices 0–9) runs
  under it and serves both arms.
* **Parity pre-check** before any arm fit: the fresh control cell `kadid_s0`
  must be bit-identical to V40 control `kadid_s0` (bake
  `2bc475deeb9a6135096e5890f5abf44670edd3e525d5d73d18215a8ccba63b9e`, result
  `62f2e3aa80a32eebf7e1f063b0cd5fc291785db3945e40aec51c4c93f1fbf74e`), under
  one Rayon thread and the v3 tier. A mismatch is recorded and the fresh
  control still runs. V40 cells never replace fresh cells after the fact. This
  follows `CONTROL_DECISION.md` (2026-10-07).
* The E33 assessment program must reproduce the V40 control predictions bit
  for bit on the 40 frozen V40 bakes. That positive control checks that the
  extended predictor did not change the old path.

## 7. Exactness: identity is 100 by construction for A and C

Chain: caller feature vector → (C: products) → feature transforms (none) →
scale-only standardization → layer 0 (no bias) → ReLU → layer 1 (weights ≤ 0,
bias = pin) → output spline.

Per-column evidence that every A/C input is exactly 0 on a perfect copy at
Rev5. Every method is required, and each is recorded per column in the
preparation receipt:

| # | Method | Covers |
|---|---|---|
| E1 | Registry: `feature_defs` gives `Form::Difference` for all 410 IDs (`v1` builder for basic; `V2` table entries for v2), `Form::ReferenceOnly` for the 10 fragility IDs | the claim being tested |
| E2 | Rev5 arithmetic gates: exact Difference identity on every tier (`featcanon_rev5_parity`, `feature_invariants`; v4x/v4/v3/scalar/wasm128/NEON evidence in the Rev5 worklog), rerun on the E33 build | arithmetic |
| E3 | Measurement on the E33 build: for each of the 410 IDs, the stored f32 value is ±0.0 on the 38 identity-probe references and the 24 NEARID identity rungs, at v4x, v4, v3 and scalar through the canonical extraction owner (ADJUDICATE measured 15,960 consumed positions on the frozen build: only the ten fragility slots were nonzero) | real images |
| E4 | Products: `±0 × f = ±0` exactly for finite `f`; `f ∈ (0, 1]` is finite by construction (§5.1). Checked by asserting every product column is ±0.0 on the E3 identities | C only |
| E5 | Transforms: the E33 recipe declares none. The trainer must **refuse** any feature transform under `fx1` or under the A spec, not just warn (the 2026-09-06 `winsor_p99` lesson) | 0→0 |
| E6 | Scaler: `--nonneg-distance` zeroes the mean; scale `σ ≥ 1e-8`, so `±0/σ = ±0` | standardization |
| E7 | Network: `b1 = 0` and `ReLU(±0) = 0` give `h = 0`, so `raw = 0 + pin = 100` (`to_bits` equality, `tests/nonneg_distance.rs`, at f32/f16/i8). `--protect-last` keeps the last layer f32; 100 is exact in f16 as well | raw = pin |
| E8 | Served: feature inference on the 38 identity vectors and 24 NEARID identities returns raw `== 100.0` bitwise for every A/C full-data seed, through `BakeScorer`, and the calibrated score `== 100.0` (§8) | C5 |

Any E3–E8 failure means the construction is broken. The arm is then
**INCOMPLETE** (an implementation defect to fix and re-register), not a model
result. No anchor rows and no identity rows are used. The calibration knot at
the pin (§8) is a construction constraint, not a fitted anchor.

`EDGE_WIDTH_CHANGE` is filled from the adjacent coarser scale, so it may be
constant at s3. Preparation records a per-column TRAIN nonzero census for the
410 differences and 410 products (features only, no labels). A column that is
exactly 0 on every TRAIN row is a dead input, which `pack` prunes; its product
is dead too. Dead columns are reported, never folded or dropped by hand.

## 8. Output stage: identity knot plus strictly increasing tail

Today `pack` fits up to 19 PCHIP knots on the packed network's raw outputs
over the TRAIN calibration rows. Its monotone filter allows equal `y`, so flat
segments can appear. The runtime (`score_math::pchip_eval_capped`, unchanged)
extrapolates above the top knot linearly, capped at 100. Below the bottom knot
it extrapolates linearly down to a floor `ys[0] − (ys[n−1] − ys[0])` (the
production seed-0 floor is −16.906). A PCHIP endpoint slope can also come out 0
(`pchip_endpoint` clamps when `d·s0 ≤ 0`), which makes the whole lower tail
flat.

E33 adds one opt-in calibration mode to the existing `pack` owner. Defaults
stay byte-identical for every other bake.

1. **Knots.** The existing `fit_spline_knots` binning (percentile edges
   1…99, bin medians, `neg_tail = true`), with the filter made strict: accept a
   knot only if `x > x_last + 1e-7` and `y > y_last + 1e-6`.
2. **Identity knot.** Require `x_last < pin` and `y_last < 100`, then append
   `(pin, 100)`. If either fails, `pack` refuses. Since `raw ≤ pin` for every
   input, the upper branch is hit only at `raw == pin`, where it returns
   exactly 100.
3. **Tail knot.** Prepend `(x_t, y_t)` with `x_t = x_0 − 2·(pin − x_0)` and
   `y_t = y_0 − s_0·(x_0 − x_t)`, where `s_0 = (y_1 − y_0)/(x_1 − x_0)` is the
   fitted bottom secant. The first two segments are then collinear (up to f32
   knot rounding), so the stored endpoint slope at `x_t` equals `s_0 > 0`. Below
   `x_t` the score falls linearly at that slope, and the unchanged OOD floor
   engages only at `x_floor = x_t − (100 − y_t)/d_0`. The factor 2 is fixed
   now and is not retuned after results.
4. Then the existing pack steps: quantize before calibration, prune dead
   columns, neg-tail materiality check, and evaluate final packed bytes.

Checks on every E33 full-data candidate, all through the serving owner:

| # | Check | Pass |
|---|---|---|
| K1 | stored PCHIP derivatives | `> 0` at every knot except possibly the identity knot; report `d_{n−1}` |
| K2 | dense monotonicity | 10⁶ points log-spaced in `g = pin − raw` over `[1e-6, pin − x_floor]`, plus `g = 0` and each knot ± 1 ulp: served score strictly decreasing in `g` at f64 for every consecutive pair |
| K3 | identity | `spline(pin) == 100.0` and E8 |
| K4 | floor never engaged | 0 evaluated rows with `raw ≤ x_floor` across every E33 population (calibration rows, C2 grid, negative-tail probe, identity probe, NEARID, every G-STEER forward). Report the count with `raw < x_0` and `min raw` relative to `x_floor` |
| K5 | existing guards | C1 mono ≥ .93, C3, C4, C6 and G-DIAL (p5 ≤ 25, p95 ≥ 85, mono ≥ .93) evaluated as in the release gate map |

K4 is the honest limit of a runtime-free tail. `g` is unbounded, so some input
far outside every registered population can still reach the floor. The floor
stays as the OOD safety net. A served tail that is strictly increasing for
*every* input needs a runtime change: for example a declared `asinh`
continuation, `floor + span·asinh((linear − floor)/span)`, the form STEERFIX
used as a diagnostic. That would cost Arm A its "today's serving code"
property, so it is a documented alternative, not part of E33.

Trainer bakes carry no output spline: the trainer logs no calibration fit for
these recipes, and the spline is added only by `pack`. So the E21 assessment
ranks raw `pin − g`, and the output stage cannot change E21. The output-stage
effect alone is measured by re-packing the frozen production dense
intermediates (control s0–s2, `e6995b10…` for s0) with the E33 mode, locally
and without refitting. That arm is report-only.

## 9. Decision rules (fixed before any fit)

### 9.1 Human rank: E21 as-good, per arm against the fresh control

Formulas exactly as E32 §"Seed-paired statistic", ten inferential seed units:
`delta(s,k)` = arm signed SROCC minus paired control signed SROCC, with source
equal weights `d_s = mean_k delta(s,k)`, `Mean Δ = mean_s d_s`,
`SE = sd(d_s, ddof=1)/√10`.

* Mean Δ ≥ −0.002.
* Every source's ten-seed mean Δ ≥ −0.005.
* W2 Δ > −2·SE: per KADID and TID2013 source/seed panel, the mean of the three
  lowest signed distortion-type SROCCs, arm minus control, averaged over the two
  sources within seed.

As-good is a retention test. It says nothing about improvement. Missing cells,
nonfinite metrics or zero-SE degeneracy mean INCOMPLETE.

### 9.2 Label-free gates, on full-data seed 0 of each arm

The candidate is full-data seed index 0. That matches the production
composition precedent; seeds 1–2 are reported and never used to choose.
Comparator: frozen production seed 0 (`f803b74c…`) as served.

| Gate | Rule | Seed-0 production today |
|---|---|---|
| **N1 near-identity ceiling** | on every one of the 24 NEARID references, both one-pixel ±1 rungs serve ≥ 99.0 | 88.69–97.67: fails |
| **N2 no gap** | the highest nonidentical served score on each reference ≥ 99.0 (no reference leaves a band below 100 unreachable) | max 97.73: fails |
| **N3 ladder order** | nonincreasing ladders ≥ 122 of 144 (seed 0's count). Every reversal, recrossing and 99/98/95/90 crossing reported, no smoothing | 122/144 |
| **C2 ties** | standard grid tied ≤ 0.05 (flat and ladder), unchanged bar and corrected comparator (`5788652e`) | 0.0065 / 0.036 |
| **C5 identity** | by construction (§7 E8, §8 K3): all 38 raw identities served at exactly 100.0 | fails 38/38 |
| **G-STEER** | 135 cases, M2 ≥ .99 and M3f ≥ .70, no exclusions, served path with STEERFIX's runtime fix; pass count ≥ 128. Case-level changes reported | 128/135 |
| **Output stage** | K1–K5 (§8) | n/a |
| **Runtime** | measured (§9.3); guard below | n/a |

The 99.0 bar in N1/N2 is a registered choice. The dial contract calls
[97.5, 100] identity, and a target of 98 has to be reachable by near-lossless
encodes; one changed LSB on one pixel of a 512-px image is far below
visibility. The only evidence near it is a diagnostic: on the frozen model with
the ten fragility inputs zeroed, a one-pixel change on the worst reference
scored 99.737 (NEARID). That is not a prediction for refit arms.

### 9.3 Runtime, measured not assumed

Exact layer sizes: first-layer weights 53,760 (control), 52,480 (A),
104,960 (C), each into 128 hidden units. A consumes 410 extracted IDs; C and
control consume 420. All three run the same extraction, because fragility comes
out of the same v2 pass.

Measure with the existing SPEEDQ/COSTCMP owner (`speedq_run.py`, warm
whole-call medians, first-32-clean paired rounds, quiet-box gates, pointwise
paired 95% CIs): candidate vs production seed 0 at 64², 256² and 1024²,
tiers v4x and v3, one thread, plus fresh-process peak RSS at 1024² and model
bytes. **Guard:** no cell is classified slower, meaning a paired CI wholly
above +2%. Report every cell.

### 9.4 Verdict

1. An arm is **eligible** if it passes 9.1 and every gate in 9.2–9.3.
2. Both eligible: **adopt A**, unless C beats A on the E32-style improvement
   test over the same cells (`d_s` = mean over sources of C − A signed SROCC;
   Mean > +0.002 and one-sided `p < 0.05`, Student t, df 9). Then adopt C.
   A wins ties because it needs no serving change.
3. Exactly one eligible: adopt it.
4. Neither eligible: **keep the control** (production seed 0). Record every
   result.
5. "Adopt" means the arm becomes the next production candidate and enters full
   qualification: every release gate, the KADID TERMINAL decision and serving
   review. E33 qualifies nothing.

Any other disposition is a separately recorded owner decision. E33's verdict
stays as registered. Stopping rule: all cells complete, no early stop on good
or bad interim results. Infrastructure failures rerun the same cell identity
and are recorded. No best-seed, best-fold or post-result rule choice.

### 9.5 Report-only (no rule)

* High-quality human slice: per source, signed SROCC within the top 20% of
  human quality, arm vs control, seed-paired. This tests the full944
  observation (`product_train_2026-09-14`): ≥91 SROCC fell 0.5174 → 0.2369 when
  a feature set with reference-dependent identity values was trained under
  `--nonneg-distance`, on a 57-pair slice. The hypothesis: an additive
  per-reference offset dominates `g` for near-lossless pairs, so they rank by
  reference rather than by distortion. The slices are small, so this is
  descriptive.
* Per-seed and per-source deltas, W1, raw `g` tails, seeds 1–2 full-data
  gates, the control re-pack (§8), TRAIN `f`/`f²` correlation, dead-column
  census.

## 10. Implementation plan (not executed)

Product columns are computed in **zensim input prep, declared in bake
metadata**. There is no zenpredict graph (PR #89 closed) and zenpredict is
unchanged. Arm A needs none of this; it runs on today's serving code.

1. **Declaration.** A JSON `fx1` file holds the ordered direct IDs and the
   pairing table of §5.1. The trainer, bake writer, densify/predict and serving
   read the same bytes; its SHA-256 is pinned in the packet.
2. **zensim serving.** A new metadata key, `zensim.derived_inputs` (v1: kind
   `product`, factor A a declared `Difference` ID, factor B a `ReferenceOnly`
   ID at the same scale/channel). It is parsed in `bake_metadata.rs`.
   `mark_feature_reads` / the planner add the referenced IDs to the read set.
   `BakeScorer` computes products after gather and before standardization. Load
   refuses unknown kinds, IDs outside the layout and cross-cell pairs.
   `zentrain.feature_ids` lists only the 410 direct IDs. Old runtimes already
   refuse such a bake, because the declared ID count must equal the caller
   width (`with_metadata`); a test pins that refusal. If this needs any new
   public item, it goes through a PR assigned to lilith; the plan keeps it
   crate-private.
3. **Steering.** For a product, the sensitivity to `d_i` is
   `w_i + f·w_{410+i}`. `f` is reference-only, so its replay delta is exactly 0
   (the existing ReferenceOnly rule). Extend `steering_support` and the
   attribution chain. Gate: finite-difference agreement on the 135 cases, as
   in STEERFIX.
4. **Trainer** (`zensim_mlp_train`). A `--derived-inputs <fx1.json>` flag. The
   loader retains the referenced fragility columns (compact rows otherwise drop
   unkept columns) and appends product columns before the scaler.
   `refuse_nonfinite_kept` covers referenced columns. Transforms are refused
   under the flag. `--nonneg-distance` is unchanged. The bake writes the
   declaration.
5. **`bake_dial_refit`.** densify, predict and pack carry and apply
   `zensim.derived_inputs` through the zensim owner, with no second
   implementation. Pack gets the §8 calibration mode (`--identity-knot`,
   `--tail-extend 2`, strict filter) and its refusals.
6. **`v2_lodo_mlp.py` / `v2_common.recipe_of`.** Add the `fx1` token, which
   only adds `--derived-inputs`. A's spec needs no change.
7. **Tests**, before any fleet job: `raw(identity) == pin` by `to_bits` for A
   and C bakes at f32/f16; trainer/predict/serving product parity, bit-identical
   on TRAIN rows; old-runtime refusal; K1–K3 on a synthetic spline including a
   forced-zero endpoint case; steering finite differences for C; the V40-path
   predictor parity of §6; `just clippy`, `just lint-scripts`,
   `cargo test -p zensim --all-features`.

### 10.1 Fleet packet

zenfleet only, through the existing V40 launcher/postfit owners (`launch.py`,
`fit_cell_exec.py`, `harvest_fit_cells.py`, `postfit.sh`, `score.py`).
Jobsets:

| Jobset | Cells |
|---|---:|
| `fite33-control` | 40 (4 folds × seeds 0–9) |
| `fite33-a` | 40 |
| `fite33-c` | 40 |
| `fite33-full` | 6 (A and C, full-data seeds 0–2, `pack` in-cell with the §8 mode) |
| **total fits** | **126** |

Plus three local re-packs of the frozen production dense intermediates (no
fit) and the local label-free gate runs.

**Placement rehearsal** (the V40 caps lesson: empty `hosts` maps blocked
placement, with 147 refusals and 0/40 cells run). Before authorization:
* `jobset_caps.json` entries for all four jobsets use the approved standard
  placement map from the V40 capsfix: home fleet workers named by CPU, no
  private addresses in tracked files. The landed launch/freeze refusals
  (empty map, unregistered alias, invalid slot count) must pass.
* Run a dry placement of one bounded smoke cell per jobset through the real
  launcher against the live caps.
* Run first-epoch smokes per arm on the slowest worker class, recording
  epoch time and peak RSS under the 6 GiB / no-swap / one-CPU envelope. C's cap
  is set from its smoke before freeze, never raised mid-run.
* Verify artifacts land after the first cell, before scaling.

**Budget.** Measured on V40, training loop only (epoch-119 `t=` in each
cell's log, one value per cell): control 40 cells median 795 s, range
509–1399 s, total 9.51 cell-hours; E32's 462-input arm median 930 s, total
10.54 cell-hours. The `fitv2d1` full-data seed-0 cell logged 1,261 s of
training. **E33's own per-cell costs are not measured.** C (820 inputs) and A
(410) get their numbers from the first-epoch smokes, which go into the
authorization record before launch. No projection is registered here.
Registered per-cell wall cap: 4,200 s for control and A (3 × the 1,399 s
V40 maximum). C's cap is 3 × the maximum its smokes imply, fixed at freeze.
A cell over its cap is stopped and diagnosed, not silently retried. The home
fleet has no cloud cost.

Freeze before launch: program/assessment archives and binary hashes, the
`fx1` declaration hash, data archive `9c3eff1b…`, fit specs, caps, smoke
receipts, the parity pre-check result and this registration's commit.

## 11. What E33 does not claim

It establishes SDR rank retention against a matched control on already-exposed
D1 populations, plus label-free identity, near-identity, tie and steering
properties on TRAIN probes. It does not establish perceptual accuracy at the
top of the scale (no admitted near-lossless human data), HDR behaviour,
untouched external generalization, or qualification.
