# E34 — linear-along-repair training for the model-limited G-STEER failures (registration DRAFT, revision 2)

Status: **DRAFT revision 2, 2026-10-10. Not registered and not approved.** No fit, no fleet job and no label read was
done for it. Drafted on zensim `main@origin` `41cfd432`.
- Revision 2 answers the xhigh review `E34_DRAFT_REVIEW.md` (FIX-FIRST): arm M dropped (P1), P2-a–d fixed, the P3s
  addressed.
- Next: re-review, then the owner approves or rejects the whole registration and every open choice (§8).

## 0. Scope

G-STEER is the last model failure on the release gates.
- **Population:** PRODQUAL-B's registered 135 cases (GSTEER143 ruling R1). Bars are M2 ≥ .99 and M3f ≥ .70;
  qualification needs 135/135.
- **Current counts** (QUAL-A, E33): A 129, C 127, production seed 0 128. Full-data seeds 1–2: A 124 / 128, C 127 / 131.
- **Settled:** spatial cost and memory (QUAL-A, STEERPATH); exact code replay (STEERFIX, E33C_STEER).

E34 asks: **can a training or architecture change inside the one-hidden-layer `N` model class make the served model
linear enough along block repairs to pass G-STEER while retaining human rank?** It tests three points of that class,
not the whole class (§6).

**Adaptive design, disclosed.** The arms were designed after characterizing these 135 cases (§1). No number is tuned
on them (λ comes from a disjoint dev panel; W and L are fixed here; the 135 are read once per model, after all fits),
but the design is informed by them. §4.5 therefore also reports every full-data model on the disjoint dev panel.

## 1. The failures, from existing evidence only

### 1.1 What M2 measures

From `zensim/src/metric/steerfix_packet.rs::engineering_packet`, for each block of one image:
- **true gain:** the score change from fully repairing that block (reference pixels pasted in), `F(f+Δf) − F(f)`, with
  `Δf` the exact extracted feature change.
- **linear gain:** `s·Δf`, with `s` the central-difference sensitivities at the base (step `max(|f|·1e-3, 1e-5)`,
  `bake.rs::fd_gradient_into`).
- **M2** = Spearman(linear, true) over the blocks; **M3f** = Spearman(served map's `refinement_gain`, true).

Two facts follow.

1. **M2 doesn't depend on the output spline** (monotone: it keeps the true gains' ranks and scales every linear gain
   by one positive slope; STEERFIX and E33C_STEER measured identical served, pre-floor and smooth-floor M2). **M2 is a
   property of the network alone.**
2. **The network is concave in its standardized inputs.**
   - Head `N` projects `w2 ≤ 0`, `b1 = 0` and `b2 = pin` after every step
     (`zensim-validate/src/mlp_train/mod.rs::nonneg_project`), and the scaler mean is zeroed.
   - So `raw = pin − Σ_j |w2_j|·ReLU(z_j)` is concave and piecewise linear. Products in C are linear in `d` for a
     fixed reference.
   - With an exact supergradient, `true ≤ linear` must hold on every block. The gap is the curvature collected where
     hidden units switch along the repair.

### 1.2 Pass counts and the evidence script

| Model | Seed 0 | Seed 1 | Seed 2 | Dense seed 0 |
|---|---:|---:|---:|---:|
| A (`7cf4cfd9…` and seeds) | 129 | 124 | 128 | 127 |
| C (`89cb4861…` and seeds) | 127 | 127 | 131 | 127 |
| Production (`f803b74c…`) | 128 | — | — | — |

**Sources:** QUAL-A `steer-s0-{a,c,seed0}.json`; E33 `gates/steer-s{1,2}-{a,c}.json` and `steer-s0-{a,c}-dense.json`.

**Method:**
- Every number in §1 that comes from stored rows is produced by the committed
  `benchmarks/e34_draft_2026-10-10/characterize.py` into `summary.json`. It reads stored JSON only.
- Populations: the seven served models (A and C seeds 0–2, production seed 0). Dense bakes appear only in the
  pass-count table above.
- Recomputed M2 matches the stored values within 2.6e-7.

**Outputs** (`/mnt/v/output/zensim/e34-draft-2026-10-10/`):

| file | sha256 |
|---|---|
| `per_case.json` | `6a6ba56b…` |
| `per_case.tsv` | `90f7f3da…` |
| `summary.json` | `81d18131…` |

### 1.3 Seed-0 failures

Content labels are from the reference pixels (TRAIN).
- **Without worst block:** M2 after dropping the block with the largest rank displacement.
- **Rel. L2:** ‖linear − true‖ / ‖true‖ over the case's blocks, in served units.
- **true > linear:** blocks breaking the concavity inequality in raw units (E33C_STEER's PreFloor rows; A and C only).

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

Production's four KADID JPEG failures are floor ties (served −16.906; STEERFIX "CODE"), removed for A and C by E33's
identity knot and tail. **All 44 A and C failures are M2 misses** (A seed 1 broad-76-b64 also misses M3f), all "MODEL"
by E33C_STEER's criteria.

### 1.4 What the failures share (51 failures over the seven served models; 47 excluding the floor ties)

- **Block size.**
  - Failures: b64 23/168, b32 12/385, b16 6/168, b8 10/224 (6 non-floor).
  - At b64 a case has 12–16 blocks. M2 ≥ .99 allows Σd² ≤ 2.86 at n = 12 and ≤ 6.8 at n = 16, i.e. one or three
    adjacent swaps.
  - Dropping the most-displaced block lifts 12 of 23 b64 failures above .99, against 6 of 24 at b8–b32.
- **But it is not only small n.** At every block size, failing cases (non-floor) have a higher median linearization
  error (Rel. L2, served) than passing cases. The ratio ranges from 3.1× to 10.0× (all panels):

  | Block | Failing (n) | Passing (n) | Ratio | Broad-only ratio |
  |---:|---:|---:|---:|---:|
  | 8 | .042 (6) | .013 (214) | 3.1× | 2.3× |
  | 16 | .082 (6) | .018 (162) | 4.7× | 4.7× |
  | 32 | .160 (12) | .016 (373) | 10.0× | 8.9× |
  | 64 | .126 (23) | .021 (145) | 5.9× | 5.9× |

  From `summary.json` (`rel_l2_by_block`); revision 1's "4–9×" table did not reproduce.
- **Curvature, by analogy.**
  - R5STEER2 measured curvature directly on the E24 `by_v2fy` Rev5 bakes, not on E33's A and C.
  - Interpolating feature rows, 9 of 10 failing cases pass at 1% of the repair and fail at the full repair. The owner
    case's relative L2 grows from .02 at 1% to .19 at 100%.
- **Content.** Three of the eight broad references carry 28 of the 44 broad failures:

  | Origin | Content | Failures |
  |---|---|---:|
  | 7066 | line art | 10/84 |
  | 8206 | mostly white screenshot | 9/84 |
  | 9380 | product on white | 9/84 |
  | 6068 | uniform text scan | 2/84 |
  | 8384 | text screenshot | 2/84 |

  The failure-prone images mix large flat areas with concentrated detail.
- **Distortion level.**
  - Failures by JXL level: 10 → 21/224, 13 → 13/224, 17 → 10/224. Level 10 is the mildest.
  - Median base score is 91.4 at level 10, 80.6 at 13 and 38.9 at 17 (seed-0 A, C and production).
  - Failing broad cases have a median base of 85.3, against 80.4 for passing ones.
- **Seeds.** 36 distinct (case, block) pairs fail somewhere; only 9 fail in two or more models (broad-136-b64 in 5 of
  7). Counts range 124–131. Dense C passes two of packed C's failures and fails two others.
- **Gradient estimate.**
  - In 12 of 14 seed-0 A/C failures at least one block has true > linear in raw units. A concave network can't produce
    that with an exact supergradient.
  - It is not f32 rounding: the review measured the largest excess at 0.4–4.9% of each case's largest gain.
  - So the central-difference sensitivities aren't supergradients there: units sit within one step of a kink.
  - R5STEER2 saw it as step sensitivity, 2 of 10 cases.
- **Training revision (by analogy, E24 `by_v2fy` bakes, uniform 3-seed ensembles, same `N` head and spec;
  R5STEER2):**

  | Weights | Arithmetic | Broad passes | Owner case (I61, seed index 2, b32) |
  |---|---|---:|---|
  | Rev4-trained | Rev4 | 92/96 | Rel. L2 .004 at full repair |
  | Rev4-trained | Rev5 | 93/96 | M2 .9997 |
  | Rev5-trained | Rev5 | 85/96 | Rel. L2 .19, M2 .888 |

  Averaging three seeds did not fix the Rev5 weights. E33's A and C are Rev5-trained.
- **Not a floor effect:** A/C bases are 79–93; C's floor is −304.

**Not in existing evidence:** which inputs or hidden units drive each misranking. Stored rows hold per-block scores
only. Preparation P3 (§7) measures it, report-only, after the arms are frozen.

## 2. Hypotheses

| # | Hypothesis | For | Against |
|---|---|---|---|
| H1 | **Curvature:** uneven unit switching along repairs causes the misranking | Rel. L2 3.1–10.0× higher in failures at every block size; R5STEER2 1% vs full repair (by analogy); b64 (largest Δf) fails most | b8 cases fail too (n = 704) |
| H2 | **Kinks near the base** make `s` a poor linearization | true > linear in 12/14 seed-0 A/C failures; R5STEER2 step sensitivity 2/10 | 8/10 R5STEER2 cases persist under both alternative steps |
| H3 | **Content:** flat + concentrated-detail images with mild distortion expose H1 | 28/44 broad failures on 3/8 references; level 10 fails 2× level 17 | 3 references can't separate content from identity |
| H4 | **Gate discreteness at b64:** 12–16 blocks allow 1–3 adjacent swaps | 12/23 b64 failures pass with one block dropped | A gate property; E34 changes no threshold |
| H5 | **Training data sets curvature** | `by_v2fy` analogy: Rev4 weights 93/96 vs Rev5 weights 85/96 at Rev5 arithmetic | Not shown on A/C; the cause inside the Rev5 tables is unknown |
| H6 | **Seed noise near the bar:** a fix must shift the whole distribution | 27/36 failing pairs fail in one model only; counts 124–131 | — |

H1 and H2 have one root, the density of ReLU kinks in the operating region. K attacks curvature directly, W reduces the
number of kinks, and L removes them.

## 3. Arms

**Families.** Each arm is fitted in **A's family** (`sel:3b7bd5ebe929@h32:H128:cv16:cf98`, 410 differences) and in
**C's family** (`…:fx1`, + 410 products). L runs in A's family only.

The recipe is exactly E33's except for each arm's single change:
- head `N`, pin 100, hidden 128, human weight 32, coverage `cv16:cf98`;
- 120 epochs × 50,000 pairs, final epoch 119, V40 seed streams, `ZENSIM_MAX_TIER=v3`, Rev5;
- archive `9c3eff1b…`, byte-identical;
- E33 output stage at pack.

D1 fits run through `zensim_mlp_train` (launched by `v2_lodo_mlp.py`); the arms extend that owner. zenpredict v3 and
serving are unchanged.

| Arm | Single change | New data | New trainer code | Families |
|---|---|---|---|---|
| **Control** | none; fresh under the E34 program | — | — | A, C |
| **W** (narrow) | hidden width 32 instead of 128 (`H32`, an existing recipe value) | — | none | A, C |
| **K** (linear-along-repair penalty) | training loss on steering-shaped pairs | augmentation table (§5) | yes | A, C |
| **L** (linear ceiling) | hidden width 1 and W1 ≥ 0 on every input | — | yes (`N`-path projection, `H1`) | A |

### 3.1 Arm W: fewer kinks

Hidden width 32, nothing else: the program's other registered `--nonneg-distance` width (rev4_featpot prereg H32/H128),
not tuned; `recipe_of` already accepts it. **Mechanism (H1):** a quarter of the switching hyperplanes, so fewer lie
between a base row and its repair, while W stays a genuine ReLU network. **What W tests against:** each remaining kink
carries more weight, so per-crossing curvature may grow.

### 3.2 Arm K: a penalty for nonlinearity along repairs

**Pairs.** A pair is `(x, x′)`: a base row and the same pair with one block fully repaired, from the augmentation table
(§5). Both rows are in the arm's input layout (C: products through the shared `fx1` owner) and were extracted at
`ZENSIM_MAX_TIER=v3`, the fit archive's tier. Repairs whose row equals the base row bit for bit are dropped at build.

**Residual,** in raw units: `r = (F(x′) − F(x)) − ∇F(x)·(x′ − x)`, with `∇F` the exact activation-pattern gradient
`Σ_j w2_j·1[z_j > 0]·W1_j / σ` (not a finite difference; ties at `z = 0` as in the forward). Under `N`, `r ≤ 0`.

**Loss,** per image `i` with its sampled blocks `k`:
`L_K = λ · mean_i [ Σ_k r_ik² / D_i ]`, where `D_i = Σ_k (F(x′_ik) − F(x_i))²` is **detached** (no gradient flows
through it).
- An image with `D_i < 1e-10` is skipped in that step and counted in the log (stored raw b8 gains reach 5e-2).
- It is §1.4's squared relative L2, scale-free per image, and reads no label. Every term is piecewise linear in
  `w2` and `W1`, so the hand-written backprop extends directly.

**Sampling,** per epoch: 512 images (uniform over the 1,536 bases) × 8 blocks (uniform within the image, all four
sizes) = 4,096 pairs, beside the unchanged 50,000 rank pairs. Eight blocks per image stabilize the ratio (numerator and
denominator use the same eight). `nonneg_project` is unchanged, so raw ≤ pin and raw(0) = pin stay exact.

**λ** is picked on a population disjoint from G-STEER:
- Pilot: λ ∈ {0.1, 1, 10}, one full-data seed-0 fit per value and family (6 fits), each run on the 96-case **dev panel**.
- The λ with the most dev passes (M2 ≥ .99 and M3f ≥ .70) wins; ties go to the smaller λ.
- **The chosen pilot fit is K's full-data seed 0.** It is not refitted. Seeds 1–2 and the 40 LODO cells run at that λ.

### 3.3 Arm L: the linear ceiling (A family)

- **Change:** hidden width 1 (a new `H1` recipe value; `recipe_of` allows 8–512 today) and W1 ≥ 0 projected after every
  step on **all 410 inputs**.
- **Why every input:** all 410 are registered `difference` / `higher_is_worse`. The review's synthetic census found
  every one ≥ 0 on 150 pairs. P1 verifies the minimum is ≥ 0 on every D1 fit row; the rule doesn't read the
  unvalidated registry `Direction` field.
- **What follows:** with `b1 = 0`, `w2 ≤ 0` and nonnegative inputs the one unit is always active, so
  `raw = pin − c·x`, `c ≥ 0`: linear on the nonnegative domain. **M2 = 1 up to finite-difference rounding** (f32
  forward, steps down to `1e-5`; P2 compares `s` with `c`). **M3f ≥ .70 is not guaranteed.**
- **Role:** the anchor for how much human rank perfect linearity costs here. It is adoptable last (§4.4), or report-only
  if the owner prefers (§8).

**Prior evidence on sign-constrained and linear scores** (other heads, feature sets and populations, so not a
prediction of E34's numbers):

| Record | Model | Result |
|---|---|---|
| `v46_v46b_monotone_cbc_pareto_2026-05-26.md` | V46b, `monotone_cbc` | CID22 −0.035, KADID −0.12 vs V39 (0.9251) |
| `v47_masked_monotone_2026-05-27.md` | masked-strict | KADID 0.803 vs 0.925 |
| `minmax_monotone_2026-07-16.md` | positive-weight `monotone_cbc` MLP | imazen-26 0.032 |
| same | masked-monotone linear | imazen-26 0.862, "the empirical monotone ceiling for a linear score" |
| May sign census (`feature_sign_mask_2026-05-26.tsv`) | 372 v1 features | 300 `pin_geq0`, 72 `free` (correlation sign flips across distortion types); 33 of A's 130 v1 inputs are `free` |

L constrains those 33 too. That is its expected rank cost.

### 3.4 Dropped: arm M (monotone in distortion)

Revision 1's M projected W1 ≥ 0 on `HigherIsWorse` columns with nonnegative TRAIN minima. **Every input meets both
conditions** (all 410 differences; C's products are nonnegative too), so with `b1 = 0` no unit could switch: M was L's
linear class at width 128 (review P1). A partial M (the May sign-safe subset, 97 of A's 130 v1 inputs) is not admitted
either: that subset comes from a correlation census, not value signs; the v2 inputs have no census; and the prior
record above shows rank costs with no measured steering benefit. **W takes M's place** as the second nonlinear
intervention, with a cleaner mechanism and no new code.

### 3.5 Fit packet and cost

| Jobset | Cells |
|---|---:|
| Control-A, Control-C (LODO 4 folds × seeds 0–9) | 80 |
| W-A, W-C | 80 |
| K-A, K-C | 80 |
| L-A | 40 |
| Full-data: Control-A/C seeds 0–2 (6), W-A/C seeds 0–2 (6), K-A/C seeds 1–2 (4), L-A seeds 0–2 (3) | 19 |
| K λ pilot (3 λ × 2 families, full-data seed 0; the chosen one is K's seed 0) | 6 |
| **Total** | **305** |

**Measured basis:** E33 `results-cells`, epoch-119 training time per cell (`e33_cell_costs.txt` `338b1df3…`):
- A: median 773 s, 40 cells 8.40 h;
- C: median 1,522 s, 40 cells 16.92 h;
- full-data: A mean 1,054.9 s, C mean 2,156.6 s (3 cells each).

**Projection** (arithmetic on E33's measurements, replaced by first-epoch smokes before freeze):

| Part | How priced | Cell-hours |
|---|---|---:|
| LODO | 4 A-family jobsets × 8.40 h + 3 C-family × 16.92 h (W and L priced at their family's cost, which is conservative) | 84.36 |
| Full-data and pilot | 14 A cells × 1,054.9 s + 11 C cells × 2,156.6 s | 10.69 |
| **Total, before K's overhead** | | **95.05** |
| **If K's per-cell time doubled** | K is 29.78 of those hours | **124.8** |

**Wall time:** E33 ran 37.77 cell-hours in 3.73 h (10.12 per hour) on up to 19 slots, including loaned r5600g and i134
(i134 is an agent host since 10-10). Slot-proportional to the 15 home-fleet slots (tower, i265, i270, r3500, r3800x):
7.99 per hour, so about **11.9 h**, or 15.6 h if K doubles. Unmeasured.

**Caps and placement:** as E33 (3 × the smoke-implied maximum; V40 capsfix map; dry placement; artifacts verified
after the first cell). zenfleet only, no cloud cost.

**Local work:** augmentation and dev-panel extraction; G-STEER (135) on 21 full-data models and the dev panel on those
21 plus the 6 pilots (about 7–8 min per model, from QUAL-A and E33 timestamps); E21; the label-free gates.

## 4. Evaluation and decision rule (fixed before any fit)

### 4.1 Each arm against its own family's fresh control

- **Parity pre-check:** Control-A/C `kadid_s0` bit-identical to E33 A/C `kadid_s0` (one thread, v3); a mismatch is
  recorded and the fresh controls still run.
- **E21 as-good** (E33 §9.1 formulas and seed units, arm vs fresh family control, four D1 sources):
  - Mean Δ ≥ −0.002;
  - every source ≥ −0.005;
  - W2 Δ > −2·SE.
  - Missing or nonfinite: INCOMPLETE, which means not eligible.

### 4.2 G-STEER (primary), full-data seed 0

- PRODQUAL-B's 135 cases, M2 ≥ .99 and M3f ≥ .70, no exclusions, served path, same instrument, settings and pins as
  QUAL-A.
- **Arm pass: 135/135.**
- Seeds 1 and 2 are run. Each must be no lower than the family control's count for the same seed (ties pass).
  - This guard is weak: control counts move 124–131.
  - §8 offers the owner a stronger form: each ≥ max(the control's count, 133).
- The 135 cases are read once per model, after every fit. They never choose λ, seeds, epochs or arms.

### 4.3 Non-regression gates (full-data seed 0, as registered in QUAL-A / E33)

- C1 mono ≥ .93, C2 ties ≤ .05, C3, C4, C5 exact 100 (E33 §7 E3–E8), C6;
- G-DIAL p5 ≤ 25, p95 ≥ 85, mono ≥ .93;
- N1 and N2 ≥ 99.0, N3 ≥ 122/144;
- output stage K1–K5;
- runtime: no cell slower than the family control by a paired CI wholly above +2% (SPEEDQ/COSTCMP; 64², 256², 1024²;
  v4x and v3; 1 thread);
- spatial cost ≤ 3× uncached; map memory ≤ 128 B/pixel + 64 MiB (QUAL-A chain, 1024² and 2048²).

### 4.4 Verdict, per family: sequential incumbent

1. An arm is **eligible** if it passes 4.1, 4.2 and every gate in 4.3.
2. **Order: W, K, L.** That is fewest moving parts first: W changes one recipe value, K adds code and data, L changes
   the model class.
   - **Incumbent** := the first eligible arm in that order.
   - Each later eligible arm **replaces the incumbent only if it passes the improvement test against the current
     incumbent** over the same cells: Mean Δ > +0.002 and one-sided `p < 0.05`, Student t, df 9 (E32/E33).
   - An arm that isn't eligible never becomes or displaces the incumbent.
3. **No eligible arm:** the family keeps its E33 model. Every result is recorded.
4. **A vs C:** E34 doesn't choose between the families. The owner's open choice from E33 stays open, with both
   families' E34 outcomes attached.
5. **Meaning of "adopt":** the next candidate for that family, entering full qualification (every release gate, the
   KADID TERMINAL decision, serving review). E34 qualifies nothing.

Infrastructure failures rerun the same cell identity. There is no early stop and no post-result rule change.

### 4.5 Report-only

Every arm's per-case M2, M3f, Rel. L2 and true > linear on the 135 cases, all seeds; **the dev panel on all 21
full-data models** (the held-out steering read, §0); L's `s` vs exact coefficients (P2) and M3f; W's and K's Rel. L2
change vs control; K's penalty and skipped-image count per epoch; the high-quality human slice (E33 §9.5); per-source
deltas.

## 5. Data exposure

| Read | Population | Role | Purpose |
|---|---|---|---|
| Fit | D1 human (KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL, CID22-A(25)), SafeSyn and CID22-train teachers, KADIS coverage; archive `9c3eff1b…` | as E33 | training |
| E21 | the same four D1 sources, each withheld from its fold | already exposed (E33, V40) | §4.1 |
| Steering augmentation (K) | imazen-26 origins passing the eligibility rule below; pixels and features only | TRAIN | §3.2 |
| Steering dev panel | 8 more eligible origins, built by the broad recipe (longest side 256, JXL scalar 10/13/17, blocks 8/16/32/64; 96 cases) | TRAIN, pixels only | λ pilot; §4.5 |
| G-STEER | PRODQUAL-B's 135 cases | as registered (96 TRAIN, 27 historical M3, 12 KADID SELECT) | §4.2, once |
| Label-free gates | identity probe 38, NEARID 24, instrument grids, negtail probe | as E33 | §4.3 |
| Census (L) | feature columns of the D1 fit rows | features only | §3.3 |

**CID22-A(25)** is inherited from E33's archive. Per `docs/DATA_SPLITS.md` (2026-09-22), `844297.png` (CID22-A) is the
same picture as `3316926_opo25u.png` (sealed CID22-B), which is why B is counted as B(23). This adds no new exposure;
it is stated so the fit population is described fully.

**Eligibility** (`docs/DATA_SPLITS.md` §2a). An origin qualifies for the augmentation pool or the dev panel only if
both assignments agree, at the pinned corpus revision `187fbf338ce08e8e6654db7f04ddae58d5263da2`:
- `zenmetrics/scripts/picker/origin_split.py::split_of(origin) == TRAIN`;
- its `manifests/split_map_family.tsv` family is TRAIN.

Families are kept disjoint between the augmentation pool and the dev panel.

**Exclusions**, from both: the 8 broad origins (1054, 2010, 6068, 6610, 7066, 8206, 8384, 9380) and their families;
the M3 fixtures (city, dog, girl); KADID I01/I21/I41/I61; the 38 identity-probe, 24 NEARID and instrument-grid
references; **the E21 human references** (KADID, TID2013, KonFiG, CID22-A), since E21 is an evaluation population; and
anything within dHash 10 of these (existing near-duplicate audit, contextual review).

P4 also reports whether SafeSyn's references include any of the 8 broad origins. I couldn't establish that here.

**Augmentation construction:**
- **References:** 128 eligible origins by k-means on reference-only features (k = 128, centroid-nearest, singletons
  kept), longest side 256.
- **Distortions:** four teacher codecs (`moz`, `jxl`, `webp`, `avif`) × three qualities spanning q20–q80 on each
  codec's scale.
- **Repairs:** blocks 8/16/32/64, up to 16 per (image, size) by a fixed seed. That is 98,304 repair rows and 1,536
  bases, extracted at v3.
- **Cost:** at about 2.2 ms per 256² extraction (E33C_RUNTIME diagnostic), about 4 CPU-minutes.
- No label exists for these rows.

**Not read:** KADID TERMINAL, AIC-3/AIC-4/SDR25, CID22 gold and human, KonJND, sealed/T0, HDR VAL, UPIQ, external
panels. The exposure-ledger entry is written at assessment time.

## 6. Falsification plan

**Registered conclusion:** "these interventions (W at H32, K at its piloted λ, L) do not reach a point that both steers
and keeps rank". It is drawn when all of these hold:
- no family has an eligible arm;
- W and K both fail G-STEER at full-data seed 0 in both families;
- L-A reaches 135/135 but fails E21 as-good.

E34 tests three points of the one-hidden-layer `N` class. It cannot show the whole class unlearnable.

**Interpretive criterion, reported beside the conjunction and not part of it:** whether W's or K's mean Rel. L2 on the
135 cases is at least 30% below its family control's. It separates "the intervention made the model more linear but
not enough" from "it didn't make it more linear".

**What follows, each a separate registration:**
- **(a)** a smooth-activation class: a C¹ hinge with `h(0) = 0` and `h ≥ 0`. This needs a zenpredict activation and a
  serving change.
- **(b)** steering that re-scores candidate blocks exactly, which changes the cost model.
- **(c)** an owner review of the M2 bar at 12 blocks, raised as a question, not changed here.
- **(d)** if the interpretive criterion shows K became more linear without passing, a stronger-λ K.

**Other outcomes:** L-A missing E21 only narrowly means the frontier is close. W or K reaching 135 and keeping rank
confirms the hypothesis for that family. L-A missing 135 (on M3f) means the gate fails even a linear model, pointing at
(c) or at M3f's definition.

## 7. Preparation (after approval, before freeze; no fit)

- **P1.** Feature-only census of the D1 fit rows: per-column minimum and maximum for the 410 differences. L's rule
  needs every minimum ≥ 0. Any column below 0 is reported and L's linearity is stated only for nonnegative rows.
- **P2.** For L: compare `s` with the exact coefficients `c` per case. List the cases with any negative input row.
- **P3.** Report-only diagnostic on the E33 A/C seed-0 failures (stored models, label-free): the units that switch
  along each repair and the columns that dominate those switches. Done after freeze.
- **P4.** Exposure receipts: eligibility (both assignments, pinned revision); exclusions and dHash audit (including
  the E21 references); the SafeSyn intersection; k-means; the augmentation and dev-panel builds with tier and extractor
  pins.
- **P5.** Trainer:
  - **K:** `--steer-pairs <parquet> --steer-lambda λ`.
  - **L:** extend the **existing** `monotone_cbc` / `monotone_feature_pin` projection owner to the plain `N` path,
    with `monotone_feature_pin = None` meaning "all columns". It is wired today only on `--per-sample-alpha-head`
    (`capabilities.rs`), which `--nonneg-distance` refuses, so it can't be reused as is and a second projection must
    not be written. Add an `H1` recipe value and trainer acceptance of width 1.
  - **W:** no code.
  - **Tests:**
    - K's exact-gradient residual against finite differences on a toy network;
    - the L projection keeps raw(0) = pin exact at f32 and f16;
    - refusal combinations;
    - default argv byte-identical.
- **P6.** L serving smoke: an `in×1×1` `N` bake at f16 passes identity (C5, E3–E8), the steering instrument and the
  served-path gates. Then smokes per arm and family on the slowest worker class, caps, dry placement, parity pre-check
  and the freeze record.

## 8. Open questions for the owner

1. **G-STEER bar:** 135 at seed 0 with seeds 1–2 non-inferior (as drafted; weak), 135 on all three seeds, or seeds
   1–2 each ≥ max(the control's count, 133)?
2. **Families:** both (305 fits, about 95 cell-hours before K's overhead), or one? C costs about 2× per cell.
3. **W:** keep it (as drafted), or drop it and run K alone as the only nonlinear intervention (saves 86 fits)?
4. **L:** adoptable last (as drafted) or report-only, given the prior record of sign-constrained and linear scores (§3.3)?
5. **K's λ:** a pilot on the 8-origin dev panel, or one fixed λ?
6. **Augmentation codecs:** all four teacher codecs, or JXL only like the broad panel?
7. **The M2 bar at b64:** at 12 blocks it tolerates one adjacent swap. Intended?
8. **Hosts:** is the r5600g loan available? That sets wall time (about 12 h on the five fleet hosts, unmeasured).

## 9. What E34 does not claim

E34 tests three changes inside the one-hidden-layer `N` class against matched controls on already-exposed D1
populations and label-free TRAIN probes. It does not qualify a model, measure human perception of steering, pick
between A and C, show the whole class unlearnable, or change any threshold or rule.
