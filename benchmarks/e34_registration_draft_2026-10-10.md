# E34 — linear-along-repair training for the model-limited G-STEER failures (registration DRAFT)

Status: **DRAFT, 2026-10-10. Not registered and not approved.** No fit, no fleet job and no label read was done for it.
The characterization reads only stored, label-free G-STEER outputs. Drafted on zensim `main@origin` `41cfd432`.
Next steps: xhigh review, then owner approval or rejection of the whole registration and of every open choice (§8).

## 0. Scope

G-STEER is the last model failure on the release gates.
- **Population:** PRODQUAL-B's registered 135 cases (GSTEER143 ruling R1). Bars are M2 ≥ .99 and M3f ≥ .70; qualification needs 135/135.
- **Current counts** (QUAL-A, E33): A 129, C 127, production seed 0 128. Full-data seeds 1–2: A 124 / 128, C 127 / 131.
- **Settled already:** QUAL-A and STEERPATH settled spatial cost and memory; STEERFIX and E33C_STEER showed the code replays exactly.

E34 asks one question: **can a training or architecture change inside the existing one-hidden-layer `N` model class
make the served model linear enough along block repairs to pass G-STEER, while retaining human rank?**

## 1. The failures, from existing evidence only

### 1.1 What M2 measures

From `zensim/src/metric/steerfix_packet.rs::engineering_packet`, for each block of one image:
- **true gain:** the score change from fully repairing that block (reference pixels pasted in), `F(f+Δf) − F(f)`, with `Δf` the exact extracted feature change.
- **linear gain:** `s·Δf`, with `s` the central-difference sensitivities at the base (step `max(|f|·1e-3, 1e-5)`, `bake.rs::fd_gradient_into`).
- **M2** = Spearman(linear, true) over the blocks; **M3f** = Spearman(served map's `refinement_gain`, true).

Two facts follow.

1. **M2 doesn't depend on the output spline.** It is monotone, so it can't change the ranks of the true gains, and it scales
   every linear gain by the same positive slope. STEERFIX and E33C_STEER measured served, pre-floor and smooth-floor M2
   identical on every failing case. **M2 is a property of the network alone.**
2. **The network is concave in its standardized inputs.**
   - Head `N` projects `w2 ≤ 0`, `b1 = 0` and `b2 = pin` after every step (`zensim-validate/src/mlp_train/mod.rs::nonneg_project`).
   - So `raw = pin − Σ_j |w2_j|·ReLU(z_j)`, which is concave and piecewise linear. Products in C are linear in `d` for a fixed reference.
   - **Consequence:** with an exact supergradient, `true ≤ linear` must hold on every block. The difference between them is
     the curvature collected where hidden units switch along the repair segment.

### 1.2 Pass counts (stored outputs)

| Model | Seed 0 | Seed 1 | Seed 2 | Dense seed 0 |
|---|---:|---:|---:|---:|
| A (`7cf4cfd9…` and seeds) | 129 | 124 | 128 | 127 |
| C (`89cb4861…` and seeds) | 127 | 127 | 131 | 127 |
| Production (`f803b74c…`) | 128 | — | — | — |

Sources: QUAL-A `steer-s0-{a,c,seed0}.json`; E33 `gates/steer-s{1,2}-{a,c}.json` and `steer-s0-{a,c}-dense.json`.
- I recomputed M2 from the stored per-block rows: the maximum difference from the stored values is 2.6e-7.
- Script: `benchmarks/e34_draft_2026-10-10/characterize.py` (reads stored JSON only).
- Outputs, under `/mnt/v/output/zensim/e34-draft-2026-10-10/`:
  - `per_case.json` `6a6ba56b…`
  - `per_case.tsv` `90f7f3da…`
  - `seed0_failure_table.md` `4627dc39…`

### 1.3 Seed-0 failures

Content labels are from the reference pixels (TRAIN). "Without worst block" is M2 recomputed after dropping the single
block with the largest rank displacement. Rel. L2 is ‖linear − true‖ / ‖true‖ over the case's blocks in served units.
"true > linear" counts blocks that break the concavity inequality in raw units (E33C_STEER's PreFloor rows; A and C only).

| Model | Case | Block | Image (origin) | Level | Blocks | M2 | M3f | Without worst block | Rel. L2 | true > linear | Base |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | broad-157-b8 | 8 | 9380 product photo on white | 10 | 704 | .9795 | .9295 | .9814 | .037 | 277/704 | 92.8 |
| A | broad-220-b16 | 16 | 8384 text screenshot | 10 | 192 | .9868 | .9522 | .9901 | .037 | 107/192 | 90.5 |
| A | broad-10-b32 | 32 | 2010 dense photo | 10 | 56 | .9682 | .9699 | .9764 | .221 | 39/56 | 88.1 |
| A | broad-10-b64 | 64 | 2010 dense photo | 10 | 16 | .9353 | .9324 | .9643 | .426 | 6/16 | 88.1 |
| A | broad-136-b64 | 64 | 7066 line art | 10 | 16 | .9794 | .9794 | .9964 | .050 | 7/16 | 86.8 |
| A | broad-76-b64 | 64 | 6068 text scan | 13 | 16 | .9882 | .9971 | .9964 | .230 | 0/16 | 81.8 |
| C | broad-157-b8 | 8 | 9380 product photo on white | 10 | 704 | .9893 | .9030 | .9913 | .042 | 389/704 | 93.3 |
| C | broad-160-b32 | 32 | 9380 product photo on white | 13 | 48 | .9843 | .9690 | .9996 | .034 | 32/48 | 83.8 |
| C | broad-202-b32 | 32 | 8206 web screenshot, mostly white | 13 | 40 | .9662 | .9670 | .9820 | .165 | 29/40 | 84.9 |
| C | broad-34-b32 | 32 | 1054 architecture photo, flat sky | 13 | 48 | .9502 | .9468 | .9562 | .360 | 8/48 | 79.0 |
| C | broad-136-b64 | 64 | 7066 line art | 10 | 16 | .9882 | .9853 | .9964 | .022 | 16/16 | 88.2 |
| C | broad-202-b64 | 64 | 8206 web screenshot | 13 | 12 | .8601 | .9371 | .9273 | .220 | 7/12 | 84.9 |
| C | broad-31-b64 | 64 | 1054 architecture photo | 10 | 12 | .7273 | .7273 | .8182 | .351 | 0/12 | 90.6 |
| C | broad-34-b64 | 64 | 1054 architecture photo | 13 | 12 | .9720 | .9720 | .9909 | .378 | 1/12 | 79.0 |
| P | broad-206-b32 | 32 | 8206 web screenshot | 17 | 40 | .9698 | .9525 | .9826 | .114 | — | 57.5 |
| P | broad-206-b64 | 64 | 8206 web screenshot | 17 | 12 | .9441 | .8531 | .9545 | .146 | — | 57.5 |
| P | dog-256-20 | 32 | M3 fixture (dog, JPEG q20) | — | 64 | .9872 | .9857 | .9887 | .200 | — | 41.7 |
| P | I01/I21/I41/I61_10_05 | 8 | KADID JPEG | — | 3,072 | 0 | 0 | — | — | — | −16.9 |

Production's four KADID JPEG failures are floor ties (served scores −16.906, STEERFIX "CODE"). E33's identity
knot and strictly increasing tail remove them for A and C. **Every A and C failure is M2, and every one is "MODEL" by
E33C_STEER's criteria.** Floor and replay don't change them.

### 1.4 What the failures share (51 failures over the seven served models)

- **Block size.**
  - Failure rates: b64 23/168 (13.7%), b32 12/385, b16 6/168, b8 10/224 (6 of those 10 are non-floor).
  - At b64 a case has 12–16 blocks. M2 ≥ .99 then allows Σd² ≤ 2.9 at n = 12 and ≤ 6.8 at n = 16: at most one
    adjacent rank swap at n = 12, three at n = 16.
  - Dropping the single worst block lifts 12 of the 23 b64 failures above .99, against 6 of 24 at b8–b32.
- **But it is not only small-n.** At every block size, failing cases have higher linearization error (median Rel. L2,
  served) than passing ones:

  | Block | Failing | Passing |
  |---:|---:|---:|
  | 8 | .042 | .018 |
  | 16 | .082 | .018 |
  | 32 | .155 | .017 |
  | 64 | .126 | .021 |

  - R5STEER2 measured the curvature directly. Interpolating feature rows, 9 of 10 failing cases pass at 1% of the
    repair and fail at the full repair. The owner case's relative L2 grows from .02 at 1% to .19 at 100%.
- **Content.** Three of the eight broad references carry 28 of the 44 broad failures:

  | Origin | Content | Failures |
  |---|---|---:|
  | 7066 | line art | 10/84 |
  | 8206 | mostly white screenshot | 9/84 |
  | 9380 | product on white | 9/84 |
  | 6068 | uniform text scan | 2/84 |
  | 8384 | text screenshot | 2/84 |

  The failure-prone images mix large flat areas with concentrated detail, so their block gains are very uneven.
- **Distortion level.** Failures by JXL level: 10 → 21/224, 13 → 13/224, 17 → 10/224, where level 10 is the mildest
  (median base score 91.4 at level 10, 80.6 at 13, 38.9 at 17; seed-0 A, C and production). The median base score of failing broad cases is 85.3, against 80.4 for
  passing ones.
- **Seeds.**
  - 36 distinct (case, block) pairs fail in at least one model. Only 9 fail in two or more, and the most frequent
    (broad-136-b64) fails in 5 of 7.
  - Counts per model range from 124 to 131.
  - Failures sit near the bar and move with seed and packing: dense C passes two of packed C's failures and fails two
    others.
- **Gradient estimate.**
  - In 12 of the 14 seed-0 A/C failures, at least one block has true > linear in raw units. A concave network can't
    produce that with an exact supergradient.
  - So the central-difference sensitivities aren't supergradients there: hidden units sit within one step of a kink
    at the base.
  - R5STEER2 saw the same thing as step sensitivity: 2 of 10 cases recover with another step, 8 of 10 don't.
- **Training revision** (R5STEER2, uniform 3-seed ensembles, same `N` head and spec):

  | Weights | Arithmetic | Broad passes | Owner case (I61, seed index 2, b32) |
  |---|---|---:|---|
  | Rev4-trained | Rev4 | 92/96 | Rel. L2 .004 at full repair |
  | Rev4-trained | Rev5 | 93/96 | M2 .9997 |
  | Rev5-trained | Rev5 | 85/96 | Rel. L2 .19, M2 .888 |

  Training on Rev5 tables produced much more curved networks under the same head; Rev5 arithmetic with Rev4 weights
  stays linear. Averaging three seeds did not fix the Rev5 weights.
- **Not a floor or identity-knot effect.** A/C bases are 79–93, far above their floors (−304 for C). No A/C failure touches the identity knot.

**Not in existing evidence: which inputs or hidden units drive each misranking.** Stored rows hold per-block scores,
not per-feature deltas or activation patterns. Preparation item P3 (§7) measures this, report-only, after the arms are frozen.

## 2. Hypotheses

| # | Hypothesis | For | Against |
|---|---|---|---|
| H1 | **Curvature:** hidden units switching along repair segments, uneven across blocks, cause the misranking | Rel. L2 4–9× higher in failures at every block size; R5STEER2 1% vs full-repair; b64 (largest Δf) fails most | b8 cases fail too (n = 704) |
| H2 | **Kinks near the base:** units within one finite-difference step of a kink make `s` a poor linearization | true > linear in 12/14 seed-0 A/C failures; R5STEER2 step sensitivity 2/10 | 8/10 R5STEER2 cases persist under both alternative steps, so H2 alone explains few cases |
| H3 | **Content:** flat + concentrated-detail images with mild distortion expose H1 | 28/44 broad failures on 3/8 references; level 10 fails 2× level 17 | 3 references are too few to separate content from reference identity |
| H4 | **Gate discreteness at b64:** 12–16 blocks leave room for one to three adjacent swaps | 12/23 b64 failures pass with one block dropped | A gate property, not a model one; E34 changes no threshold |
| H5 | **The Rev5 training data, not the head, sets curvature** | At Rev5 arithmetic under one head, Rev4 weights pass 93/96 and Rev5 weights 85/96 | Which part of the Rev5 tables does it is unknown |
| H6 | **Seed noise near the bar:** any fix must shift the whole distribution | 27/36 failing pairs fail in one model only; counts 124–131 | — |

H1 and H2 have one root, the density of ReLU kinks in the operating region. The arms attack it directly.

## 3. Arms

**Families.** Each arm is fitted in **A's family** (`sel:3b7bd5ebe929@h32:H128:cv16:cf98`, 410 differences) and in
**C's family** (`…:fx1`, + 410 products). Exception: L is fitted in A's family only.

Everything else is exactly the E33 recipe:
- head `N`, pin 100, hidden 128, human weight 32, coverage `cv16:cf98`;
- 120 epochs × 50,000 pairs, final epoch 119, the V40 seed streams, `ZENSIM_MAX_TIER=v3`, Rev5;
- the fit data archive `9c3eff1b…`, byte-identical;
- the E33 output stage (identity knot, `--tail-extend 2`) at pack.

D1 fits run through the Rust trainer `zensim_mlp_train` (launched by `v2_lodo_mlp.py`), not `train_hybrid`. The arms
extend that owner; zenpredict v3 and serving are unchanged.

| Arm | Change | New data | Serving change |
|---|---|---|---|
| **Control-A, Control-C** | none; fresh fits under the E34 program | — | — |
| **K** (linear-along-repair penalty) | training loss term on steering-shaped pairs | steering augmentation table (§5) | none |
| **M** (monotone in distortion) | first-layer sign projection on nonnegative worse-is-higher columns | none | none |
| **L** (linear ceiling, A family only) | M on every kept column, hidden width 1 | none | none (a 1-unit bake) |

### 3.1 Arm K: a penalty for nonlinearity along repairs

**Pairs.** A steering pair is `(x, x′)`:
- `x` is a training pair's base feature row; `x′` is the same pair with one block fully repaired, extracted by the
  canonical Rev5 extractor;
- both in the arm's input layout (C: products built by the shared `fx1` owner).

**Residual.** Per pair, in raw units, `r = (F(x′) − F(x)) − ∇F(x)·(x′ − x)`.
- `∇F` is the **exact** activation-pattern gradient of the one-hidden-layer network, `Σ_j w2_j·1[z_j > 0]·W1_j / σ`.
  It is not a finite difference. Ties at `z = 0` follow the trainer's forward convention.
- Under `N`, `r ≤ 0` always. Its magnitude is the curvature crossed (§1.1).

**Loss.** Per image `i` with sampled blocks `k`:
`L_K = λ · mean_i [ Σ_k r_ik² / (Σ_k (F(x′_ik) − F(x_i))² + 1e-6) ]`.
- That is the squared relative L2 of §1.4, made scale-free per image. It reads no label.
- Gradients flow through `F(x′)`, `F(x)` and `∇F(x)·Δ` (piecewise linear in the weights) by the existing hand-written backprop.

**Per epoch:** 4,096 steering pairs (uniform over images, then blocks), added to the unchanged 50,000 rank pairs.
Under `nonneg_project` the arm keeps raw ≤ pin and raw(0) = pin exactly.

**λ (fixed rule, chosen on a population disjoint from G-STEER).**
- Pilot: λ ∈ {0.1, 1, 10}, one full-data seed-0 fit per value and family (6 fits).
- Each pilot model runs the G-STEER instrument on the **steering dev panel** (§5), 96 cases.
- Choose the λ with the most dev passes (M2 ≥ .99 and M3f ≥ .70); ties go to the smaller λ.
- The 40 LODO cells and full-data seeds 0–2 then run at that λ. No other value is fitted or read, and the 135-case
  population isn't touched until §4.

### 3.2 Arm M: monotone in distortion

**Column set.** `S_M` = the direct inputs whose registry `Direction` (`feature_defs`) is `HigherIsWorse` **and** whose
minimum over every D1 fit row is ≥ 0. For C it also includes each product whose difference is in `S_M` (`f ∈ (0, 1]`).
- Only feature columns are read for this; no label column.
- Preparation (P1) writes the census and the resulting list; nothing in the rule depends on results.

**Projection.** After every Adam step, `W1[:, S_M] ← max(W1[:, S_M], 0)`, next to the existing `nonneg_project`.
- With `w2 ≤ 0`, the score is then nonincreasing in every `S_M` column.
- A unit whose weights on the other columns are 0 can never switch for inputs ≥ 0.

**Mechanism:** fewer kinks can be crossed along a repair, since a repair lowers every `S_M` column it touches.

If the census puts **every** direct input in `S_M`, M can no longer switch units on the TRAIN domain and becomes a
linear model there. The registration then records "M = L in effect", and both still run as written.

### 3.3 Arm L: the linear ceiling (falsification anchor, A family)

- Keep list = `S_M` from P1; hidden width 1 (a new `H1` recipe value; `recipe_of` currently allows 8–512); M's projection on every column.
- For inputs ≥ 0 the one unit is always active, so `raw = pin − c·x` with `c ≥ 0`. **M2 = 1 on every block** whose base
  and repaired rows are nonnegative, and P2 checks that per case.
- L is adoptable under the same rule (§4.4), last in priority. Its real job is to measure how much human rank perfect
  steerability costs in this feature set.

### 3.4 Fit packet and cost

| Jobset | Cells |
|---|---:|
| Control-A, Control-C (LODO 4 folds × seeds 0–9) | 80 |
| K-A, K-C | 80 |
| M-A, M-C | 80 |
| L-A | 40 |
| full-data seeds 0–2 for Control-A/C, K-A/C, M-A/C, L-A | 21 |
| K λ pilot (full-data seed 0, 3 λ × 2 families) | 6 |
| **total** | **307** |

**Measured basis** (E33 `results-cells`, epoch-119 training time per cell; `e33_cell_costs.txt` `338b1df3…`):

| E33 arm | Median | Range | Total for 40 cells |
|---|---:|---|---:|
| A | 773 s | 407–1,684 | 8.40 h |
| C | 1,522 s | 782–2,854 | 16.92 h |
| control | 804 s | — | 9.77 h |
| full-data | 1,581 s median | — | 6 cells |

E33's 126 cells (37.8 cell-hours) ran in 3.7 h of wall time on the home fleet (`LAUNCH_LOG.md`, 07:58–11:42Z).

**Projection, to be replaced by first-epoch smokes before freeze, as in E33:**
- M equals its family control (one clamp per step).
- K adds 4,096 two-row forwards and backwards to roughly 100,000 rank-pair rows per epoch; the smoke measures the real overhead.
- L is cheaper than A.
- At E33's per-cell costs, the jobsets come to about 96 cell-hours before K's overhead: 4 A-family and 3 C-family LODO
  jobsets at 8.4 and 16.9 h each, plus 27 full-data and pilot cells at E33's 1,581 s full-data median.
- E33 completed 10.2 cell-hours per wall hour, so that is roughly 10 h of wall time on tower, i265, i270, r3500 and
  r3800x.
- These figures are arithmetic on E33's measurements, not measurements of E34.

**Caps and placement:**
- Wall caps follow E33's rule: 3 × the maximum the smokes imply, fixed at freeze.
- Placement uses the V40 capsfix standard map, a dry placement rehearsal and verified artifacts after the first cell.
- zenfleet only (`launch.py`, `fit_cell_exec.py`, `harvest_fit_cells.py`, `postfit.sh`). The home fleet has no cloud cost.

**Local work:**
- the augmentation and dev-panel extraction;
- G-STEER on the 135 cases for the 21 full-data models, and on the 96-case dev panel for the 6 pilot models, about
  7–14 min per model with the existing instrument;
- E21 assessment;
- the label-free gates.

## 4. Evaluation and decision rule (fixed before any fit)

### 4.1 Each arm against its own family's fresh control

- **Parity pre-check.** Control-A `kadid_s0` must equal E33 A `kadid_s0`, and Control-C equal E33 C, bit for bit, under
  one thread and v3. A mismatch is recorded and the fresh controls still run.
- **E21 as-good.** E33 §9.1 formulas and seed units, arm vs fresh family control on the four D1 sources:
  - Mean Δ ≥ −0.002;
  - every source ≥ −0.005;
  - W2 Δ > −2·SE.
  - Missing or nonfinite means INCOMPLETE.

### 4.2 G-STEER (primary), full-data seed 0

- PRODQUAL-B's 135 cases, M2 ≥ .99 and M3f ≥ .70, no exclusions, served path, with the same instrument, settings and
  pins as QUAL-A.
- **Arm pass: 135/135.**
- Seeds 1 and 2 are run and reported. Their counts must not be lower than the family control's count for the same seed
  (non-inferiority, guarding against a lucky seed 0).
- The 135 cases are read **once** per model, after every fit is complete. They are never used to choose λ, seeds,
  epochs or arms.

### 4.3 Non-regression gates (full-data seed 0, as registered in QUAL-A / E33)

- C1 mono ≥ .93, C2 ties ≤ .05, C3, C4, C5 exact 100 (E33 §7 E3–E8), C6;
- G-DIAL p5 ≤ 25, p95 ≥ 85, mono ≥ .93;
- N1 and N2 ≥ 99.0; N3 ≥ 122/144;
- E33 output stage K1–K5;
- runtime: no cell slower than the family control by a paired CI wholly above +2% (SPEEDQ/COSTCMP owner; 64², 256², 1024²; v4x and v3; 1 thread);
- spatial cost ≤ 3× uncached and map memory ≤ 128 B/pixel + 64 MiB (QUAL-A chain, 1024² and 2048²).

### 4.4 Verdict, per family

1. An arm is **eligible** if it passes 4.1, 4.2 and every gate in 4.3.
2. **Several eligible:** take the first in the order **M, K, L** (fewest moving parts first). A later arm is taken
   instead only if it beats the earlier one on the E32/E33 improvement test over the same cells: Mean > +0.002 and
   one-sided `p < 0.05`, Student t, df 9.
3. **None eligible:** the family keeps its E33 model. Every result is recorded.
4. **A vs C.** E34 doesn't choose between the families. The owner's open choice from E33 stays open, now with each
   family's E34 outcome attached.
5. **Meaning of "adopt."** "Adopt" means the next production candidate for that family, entering full qualification
   (every release gate, KADID TERMINAL decision, serving review). E34 qualifies nothing.

Infrastructure failures rerun the same cell identity. There is no early stop and no post-result rule change.

### 4.5 Report-only

- Every arm's 135-case per-case M2/M3f, Rel. L2 and true > linear counts (the §1 table, for all arms and seeds).
- The dev-panel results.
- L's M2 = 1 check (P2).
- The high-quality human slice as in E33 §9.5.
- Per-source deltas.
- K's penalty value per epoch.
- M's active-constraint fraction.

## 5. Data exposure

Reads, by role (`docs/DATA_SPLITS.md`):

| Read | Population | Role | Purpose |
|---|---|---|---|
| Fit | D1 human (KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL, CID22-A(25)), SafeSyn and CID22-train teachers, KADIS coverage; archive `9c3eff1b…` | as E33 | training |
| E21 | the same four D1 sources, each withheld from its fold | already exposed (E33, V40) | §4.1 |
| Steering augmentation (K) | imazen-26 TRAIN origins (last digit even), family-disjoint from every exclusion below; pixels and features only | TRAIN | §3.1 penalty |
| Steering dev panel (K's λ) | 8 more imazen-26 TRAIN origins, family-disjoint from the augmentation and the exclusions, built by the broad recipe (longest side 256, JXL scalar 10/13/17, blocks 8/16/32/64; 96 cases) | TRAIN, pixels only | §3.1 pilot |
| G-STEER | PRODQUAL-B's 135 cases | as registered (96 TRAIN, 27 historical M3, 12 KADID SELECT) | §4.2, once |
| Label-free gates | identity probe 38, NEARID 24, instrument grids, negtail probe | as E33 | §4.3 |
| Census (M, L) | feature columns of the D1 fit rows | features only | §3.2 |

**Exclusions from the augmentation pool and the dev panel:**
- the 8 broad origins (1054, 2010, 6068, 6610, 7066, 8206, 8384, 9380) and their `split_map_family` families;
- the M3 fixtures (city, dog, girl) and the KADID owner/JPEG references (I01, I21, I41, I61);
- the 38 identity-probe and 24 NEARID references;
- the instrument-grid references;
- any image within dHash distance 10 of the above, by the existing near-duplicate audit with contextual review.

The augmentation pool must also be checked against SafeSyn's references: I could not establish here whether SafeSyn
uses any of the 8 broad origins. Preparation P4 reports it, and if the 8 origins are in SafeSyn that is recorded, not hidden.

**Augmentation construction:**
- **References:** 128 eligible origins, chosen by k-means on reference-only features (k = 128, centroid-nearest,
  singletons kept), rendered at longest side 256.
- **Distortions:** four teacher codecs (`moz`, `jxl`, `webp`, `avif` from `TEACHER_CODECS`) × three qualities spanning
  q20–q80 by the codec's own scale.
- **Repairs:** blocks 8/16/32/64, up to 16 sampled blocks per (image, size) by a fixed seed. That is about 98,000 repair
  rows and 1,536 base rows.
- **Cost:** extraction at 256² costs about 2.2 ms per row (E33C_RUNTIME diagnostic at 256²), so about 4 CPU-minutes.
- Pixels and features only; no label exists for these rows. Codec variety keeps K from fitting the dev and G-STEER
  panels' JXL artefacts.

**Not read:** KADID TERMINAL, AIC-3/AIC-4/SDR25, CID22 gold and human, KonJND, sealed/T0, HDR VAL, UPIQ, and external
panels. The exposure-ledger entry is written at assessment time.

## 6. Falsification plan

Steering is judged **not learnable in this model class** if all of the following hold:
- no family has an eligible arm;
- both K and M fail G-STEER at seed 0, **and** their mean Rel. L2 on the 135 cases is not at least 30% below the family
  control's (report-only threshold for interpretation, not a gate);
- L-A reaches 135/135 (expected by construction) but fails E21 as-good.

That would mean every point this class reaches on the steerability–rank frontier either keeps rank or steers, not both.

What would follow, each a separate registration:
- **(a)** a smooth-activation class: a C¹ hinge with `h(0) = 0` and `h ≥ 0` (keeping `N`'s identity and nonnegativity),
  which needs a zenpredict activation and serving change;
- **(b)** steering that re-scores candidate blocks exactly instead of linearizing, which changes the cost model;
- **(c)** an owner review of the M2 bar where the gate has 12 blocks. Raised as a question, not changed here.

**Other outcomes:**
- If L-A fails E21 as-good only narrowly and K or M moves Rel. L2 a lot without reaching 135, the class is near the
  frontier, and a stronger K (larger λ) is the natural follow-up registration.
- If K or M reaches 135 and keeps rank, the hypothesis is confirmed for that family.

## 7. Preparation (after approval, before freeze; no fit)

- **P1.** Feature-only census of the D1 fit rows: per-column minimum, maximum and registry `Direction`. Fixes `S_M` and L's keep list.
- **P2.** L's M2 = 1 check: list the cases with any negative input, where linearity isn't guaranteed.
- **P3.** Report-only diagnostic on the E33 A/C seed-0 failures (stored models, label-free). For each case: the hidden
  units that switch along each repair, and the input columns that dominate those switches. Done after the arms are
  frozen, so it can't steer arm design.
- **P4.** Exposure checks:
  - the augmentation and dev-panel builds;
  - the dHash audit;
  - the SafeSyn reference intersection;
  - the k-means selection receipt.
- **P5.** Trainer:
  - `--steer-pairs <parquet> --steer-lambda λ` (K);
  - `--w1-nonneg-mask <tsv>` (M, L) on the plain `N` path, refused on any other head;
  - an `H1` recipe value.
  - Tests: exact-gradient residual against a finite-difference check on a toy net, projection invariants (raw(0) = pin
    exact at f32/f16), refusal combinations; default argv byte-identical.
- **P6.** Smokes per arm and family on the slowest worker class, caps, dry placement, parity pre-check, freeze record.

## 8. Open questions for the owner

1. **G-STEER bar for adoption.** 135/135 at seed 0, plus seeds 1–2 non-inferior to the control (§4.2). Or should seeds
   1–2 also reach 135, given the observed ±4 spread?
2. **Families.** A and C both (307 fits), or only one? C costs about twice as much per cell.
3. **L.** Adoptable last in priority (as drafted), or report-only?
4. **K's λ pilot** on a new dev panel of 8 imazen-26 TRAIN origins (§5), or a single fixed λ with no pilot?
5. **Augmentation codecs.** All four teacher codecs (as drafted), or JXL only like the broad panel?
6. **The M2 bar at b64** tolerates one adjacent swap at 12 blocks (§1.4). Intended? E34 doesn't change it.

## 9. What E34 does not claim

E34 tests training and architecture changes inside the current one-hidden-layer `N` class against matched controls on
already-exposed D1 populations and label-free TRAIN probes. It doesn't qualify a model, doesn't measure human
perception of steering, doesn't pick between A and C, and doesn't change any threshold or rule.
