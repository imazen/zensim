# E34 — linear-along-repair training for the model-limited G-STEER failures (registration DRAFT, revision 3)

Status: **DRAFT revision 3, 2026-10-11. Not registered and not approved.** No fit, no fleet job and no label read was
done for it. Base: zensim `main@origin` `cdaca0ee`. Tables and history: [appendix](e34_registration_draft_appendix_2026-10-11.md).

## 0. Scope and revision history

G-STEER is the last model failure on the release gates: PRODQUAL-B's registered 135 cases (GSTEER143 ruling R1),
bars M2 ≥ .99 and M3f ≥ .70, qualification 135/135. Counts: A 129, C 127, production seed 0 128; full-data seeds 1–2:
A 124 / 128, C 127 / 131. Spatial cost and memory (QUAL-A, STEERPATH) and exact code replay (STEERFIX, E33C_STEER)
are settled.

E34 asks: **can a training or architecture change inside the one-hidden-layer `N` model class make the served model
linear enough along block repairs to pass G-STEER while retaining human rank?** It tests three points of that class.

- **Revision 2** (review FIX-FIRST → READY-FOR-OWNER): arm M dropped (it was the linear ceiling L), arm W added, P2/P3 fixed.
- **Revision 3 adopts the owner's 2026-10-11 IQA evaluation guidance** (`IQA_EVAL_GUIDANCE_2026-10-11.md`). The
  owner asked to wait for it before approving. What changed, and why:
  - **Source-clustered intervals for every claim.** The characterization's comparative claims (§1.4) now carry
    `zenmetrics-stats` intervals clustered by source picture, with the Student interval beside the percentile one
    (G < 30). One claim, the distortion-level effect, no longer holds under the Student interval and is now descriptive.
  - **The decision rule is restated as source-clustered tests with Holm** (§4). E21 as-good becomes a paired,
    reference-clustered non-inferiority test at the E33 tolerance; the sequential-incumbent improvement tests become
    paired reference-clustered tests; all nine form one Holm family fixed now. The seed-unit t test is reported, not used.
  - **Pair classes** (same ladder, cross-codec, cross-source) are reported for every human-agreement comparison (§4.6).
  - **One percentile rule** (Hyndman–Fan type 7) with B, seed and rule recorded; **all statistics from
    `zenmetrics-stats`**, through zensim's owners after the pin bump (P7). E34 reads no raw forced-choice data (§5).
  - **AIC2026 labels** join the never-read list.

**Adaptive design, disclosed.** The arms were designed after characterizing these 135 cases. No number is tuned on
them (λ comes from a disjoint dev panel; W and L are fixed here; the 135 are read once per model, after all fits), but
the design is informed by them. §4.7 also reports every full-data model on the disjoint dev panel.

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

### 1.2 Pass counts and evidence

| Model | Seed 0 | Seed 1 | Seed 2 | Dense seed 0 |
|---|---:|---:|---:|---:|
| A (`7cf4cfd9…` and seeds) | 129 | 124 | 128 | 127 |
| C (`89cb4861…` and seeds) | 127 | 127 | 131 | 127 |
| Production (`f803b74c…`) | 128 | — | — | — |

Sources: QUAL-A `steer-s0-{a,c,seed0}.json`; E33 `gates/steer-s{1,2}-{a,c}.json` and `steer-s0-{a,c}-dense.json`.
`characterize.py` (stored JSON only) writes `per_case.json` `6a6ba56b…` and `summary.json` `81d18131…`: pass counts,
failure counts by block, panel, reference and level, drop-one rescues, seed overlap and medians. The seed-0 failure
table, the concavity counts and the dense comparison (appendix A1, A2) came from the same files outside it
(review-verified). All 44 A/C failures are M2 misses, all "MODEL" by E33C_STEER; production's four floor-tied KADID
JPEG failures are removed for A and C by E33's output stage.

### 1.3 Clustered intervals for the comparative claims

Rows: the seven served models × 135 cases. Cluster: the source picture (8 broad origins, 3 M3 fixtures whose renditions
are one picture, 4 KADID references; G = 15, or G = 8 on the broad panel alone). `zenmetrics-stats`
`resample::cluster_bootstrap`, `Design::OneWay`, `weighted_mean` (rev `1e48c1ea`); B 10,000, seed 20261011,
Hyndman–Fan type 7, level 0.95. Student interval `estimate ± t(G−1)·se·√(G/(G−1))`, t from scipy (the crate has no
public t quantile; P7). Floor ties are excluded from every rate. Raw: `e34-rev3-2026-10-11/clustered.json`.

| Claim | G | Estimate | Percentile 95% | Student 95% | Reading |
|---|---:|---:|---|---|---|
| failure rate b64 − b8–b32, all panels | 15 | +0.106 | [0.021, 0.197] | [0.007, 0.205] | holds |
| same, broad panel | 8 | +0.095 | [0.002, 0.181] | [−0.020, 0.211] | descriptive |
| failure rate level 10 − level 17, broad | 8 | +0.049 | [0.004, 0.094] | [−0.010, 0.109] | **descriptive; not established** |
| Rel. L2 geometric-mean ratio failing / passing, b8 | 15 | 3.32× | [2.46, 7.44] | [1.63, 6.77] | holds (8,809 finite) |
| same, b16 | 15 | 4.28× | [2.92, 7.49] | [2.54, 7.24] | holds |
| same, b32 | 15 | 7.42× | [4.25, 12.86] | [4.00, 13.75] | holds |
| same, b64 | 15 | 6.45× | [3.31, 15.86] | [2.65, 15.69] | holds |

Ratios are exp of the clustered log-ratio statistic; draws with no failing case at a block size are dropped and counted.
Revision 2's 3.1–10.0× were ratios of medians without intervals (appendix A2).

### 1.4 What the failures share

- **Block size:** b64 fails more than b8–b32 (above). At b64 a case has 12–16 blocks, so M2 ≥ .99 allows one to three
  adjacent swaps, and dropping one block rescues 12 of 23 b64 failures (A2).
- **Curvature:** failing cases have 3.3–7.4× the linearization error of passing ones at every block size, with intervals
  excluding 1 (above). R5STEER2 shows the same on `by_v2fy` bakes, by analogy (A3).
- **Content (descriptive, 8 broad sources):** three references (line art, a mostly white screenshot, a product on
  white) carry 28 of 44 broad failures (A2). This names pictures and claims nothing about new images.
- **Distortion level:** milder JXL levels fail more often on the point estimate, but the Student interval includes
  zero. Not established.
- **Seeds:** failures move with seed and packing; 27 of 36 failing pairs fail in one model only (A2).
- **Gradient estimate:** 12 of 14 seed-0 A/C failures have a block with true > linear in raw units, which a concave
  network can't produce with an exact supergradient (A2).
- **Training revision (by analogy):** Rev4-trained `by_v2fy` weights are far more linear than Rev5-trained ones under
  one head (A3).

**Not in existing evidence:** which inputs or hidden units drive each misranking (P3).

## 2. Hypotheses

| # | Hypothesis | For | Against |
|---|---|---|---|
| H1 | **Curvature:** uneven unit switching along repairs causes the misranking | Rel. L2 3.3–7.4× higher in failures at every block size, clustered intervals excluding 1; b64 fails more (+0.106, interval excludes 0); R5STEER2 (by analogy) | b8 cases fail too (n = 704) |
| H2 | **Kinks near the base** make `s` a poor linearization | true > linear in 12/14 seed-0 A/C failures; R5STEER2 step sensitivity 2/10 | 8/10 R5STEER2 cases persist under both alternative steps |
| H3 | **Content:** flat + concentrated-detail images with mild distortion expose H1 | 28/44 broad failures on 3/8 references (descriptive) | 8 sources; the level effect's Student interval includes 0 |
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
- **Role:** the anchor for how much human rank perfect linearity costs here. It is adoptable last (§4.7), or report-only
  if the owner prefers (§8).

**Prior evidence** on sign-constrained and linear scores (V46b, V47-strict, min-max monotone, the May sign census) is in
appendix A4. It is historical pooled SROCC, neither pair-class split nor clustered, so it is context, not evidence.

### 3.4 Dropped: arm M

Revision 1's M (W1 ≥ 0 on nonnegative `HigherIsWorse` columns) covered all 410 inputs, so it was L's linear class;
a partial, sign-safe M is not admitted either. Reasons: appendix A5.

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

**Caps and placement:** as E33 (3 × the smoke-implied maximum; V40 capsfix map; dry placement; artifacts checked
after cell one). zenfleet only.

**Local work:** augmentation and dev-panel extraction; G-STEER (135) on 21 full-data models; the dev panel on 26
distinct models (21 full-data + 5 unchosen pilots); 7–8 min per model (QUAL-A, E33 timestamps); E21; gates.

## 4. Evaluation and decision rule (fixed before any fit)

### 4.1 Statistics conventions

- **Owner:** every interval, p-value and Holm decision comes from `zenmetrics-stats` (rev ≥ `1e48c1ea`, pinned at
  freeze) through zensim's owners after the pin bump (P7). Harness scripts compute none of them; the existing Python
  E21 scorer is an independent reference only.
- **Bootstrap:** `Design::OneWay(reference image)`, labels unique across datasets; B 10,000; seed 20261011 plus a
  fixed per-statistic offset; Hyndman–Fan type 7; level 0.95. Every result records rule, B and seed.
- **Few sources:** the Student interval `estimate ± t(G−1)·se·√(G/(G−1))` is reported beside the percentile one for
  every statistic with G < 30 (TID2013 25, CID22-A 25 and KonFiG TRAIN+VAL's few sources; KADID TRAIN+SELECT has 65).
- **Seeds** are not resampled. Each statistic averages seeds 0–9 inside every draw; E33's seed-unit t is reported beside it.

### 4.2 E21 as-good: a paired, reference-clustered non-inferiority test

- **Statistic:** `Δ` = the mean over the four D1 datasets (equal weights) of [mean over seeds 0–9 of the arm's signed
  SROCC minus the paired fresh family control's], on each dataset's withheld-fold rows, recomputed on each draw's
  copy-aware expanded rows (`panel::spearman`). A draw leaving a dataset with fewer than 3 references is non-finite
  and dropped; `n_finite` is recorded.
- **Hypothesis:** H0: Δ ≤ −0.002 (E33's tolerance as the margin). One-sided p from the bootstrap of `Δ + 0.002`:
  `p_zero/2` if its estimate is > 0, else `1 − p_zero/2`.
- **Guards** (point estimates, as E33, each reported with its clustered intervals): every dataset's Δ_d ≥ −0.005;
  W2 Δ > −2·se, with se now the clustered bootstrap se of W2.
- Missing or nonfinite results: INCOMPLETE, p = 1.

### 4.3 Improvement tests

Paired, same statistic, arm minus the incumbent it would replace. H0: Δ ≤ 0; one-sided p as above. A replacement also
needs the point estimate Δ > +0.002 (E33's practical bar).

### 4.4 One Holm family, fixed now

Nine hypotheses, all computed after every fit, whether or not the verdict consults them:

| # | Test | Family |
|---|---|---|
| 1–3 | E21 as-good: W, K, L | A |
| 4–5 | E21 as-good: W, K | C |
| 6–8 | improvement: K over W, L over W, L over K | A |
| 9 | improvement: K over W | C |

`multiple::holm_reject` at α = 0.05; adjusted p from `multiple::holm`. An INCOMPLETE entry enters with p = 1.

### 4.5 G-STEER and non-regression gates (full-data seed 0)

- **G-STEER:** PRODQUAL-B's 135 cases, M2 ≥ .99 and M3f ≥ .70, served path, QUAL-A's instrument and pins. **Pass:
  135/135.** Seeds 1–2 each no lower than the control's same seed (ties pass; weak, §8). A count on a fixed panel, so
  no interval decides it; the pass rate's clustered interval is reported. Read once per model, after all fits.
- **Gates (registered bars, unchanged):** C1 mono ≥ .93, C2 ties ≤ .05, C3, C4, C5 exact 100 (E33 §7 E3–E8), C6;
  G-DIAL p5 ≤ 25, p95 ≥ 85, mono ≥ .93; N1, N2 ≥ 99.0; N3 ≥ 122/144; K1–K5; spatial ≤ 3× uncached; map memory ≤ 128
  B/pixel + 64 MiB. Runtime keeps its registered SPEEDQ/COSTCMP paired-round CI (timing rounds, not images): no cell
  slower than the family control by a CI wholly above +2%.

### 4.6 Pair classes (report-only)

For every arm vs its control and dataset: sign agreement by pair class from `pairs::class_agreement`, with clustered
intervals: **same ladder** (same reference, same distortion type or codec), **cross-codec** (same reference, other
type or codec) and **cross-source**. Labels: KADID/TID2013 distortion type, KonFiG distortion, CID22-A codec. The
pooled SROCC in 4.2 is mostly cross-source calibration; these classes show where an arm gains or loses. Whether a
class should gate is an owner question (§8).

### 4.7 Verdict, per family: sequential incumbent

1. **Eligible** = Holm rejects its E21 H0 (4.2/4.4), the guards hold, G-STEER and every gate pass (4.5).
2. **Order W, K, L.** Incumbent := the first eligible arm. A later eligible arm replaces the incumbent only if Holm
   rejects that improvement H0 and its point Δ > +0.002. A non-eligible arm never enters.
3. **None eligible:** the family keeps its E33 model. **A vs C:** not chosen here. **Adopt** = enter full
   qualification. Infrastructure failures rerun the same cell; no early stop; no post-result rule change.
4. **Report-only:** per-case M2, M3f, Rel. L2 and true > linear for every arm and seed; the dev panel on all 21
   full-data models (the held-out steering read; G = 8, descriptive); L's `s` vs exact coefficients and M3f; W's and K's Rel. L2 ratio vs
   control with clustered intervals; K's penalty log; the high-quality slice (E33 §9.5) with clustered intervals and pair classes;
   per-dataset deltas.

## 5. Data exposure

| Read | Population | Role | Purpose |
|---|---|---|---|
| Fit | D1 human (KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL, CID22-A(25)), SafeSyn and CID22-train teachers, KADIS coverage; archive `9c3eff1b…` | as E33 | training |
| E21 | the same four D1 sources, each withheld from its fold | already exposed (E33, V40) | §4.2 |
| Steering augmentation (K) | imazen-26 origins passing the eligibility rule below; pixels and features only | TRAIN | §3.2 |
| Steering dev panel | 8 more eligible origins, built by the broad recipe (longest side 256, JXL scalar 10/13/17, blocks 8/16/32/64; 96 cases) | TRAIN, pixels only | λ pilot; §4.7 |
| G-STEER | PRODQUAL-B's 135 cases | as registered (96 TRAIN, 27 historical M3, 12 KADID SELECT) | §4.5, once |
| Label-free gates | identity probe 38, NEARID 24, instrument grids, negtail probe | as E33 | §4.5 |
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
- **Distortions:** four teacher codecs (`moz`, `jxl`, `webp`, `avif`) × three qualities over q20–q80.
- **Repairs:** blocks 8/16/32/64, up to 16 per (image, size), fixed seed: up to 98,304 repair rows, 1,536 bases, at v3.
- **Cost:** about 2.2 ms per 256² extraction (E33C_RUNTIME diagnostic), so about 4 CPU-minutes. No labels exist.

**Not read:** AIC2026 labels, KADID TERMINAL (and every TERMINAL segment), AIC-3/AIC-4/SDR25 labels, CID22 gold and
human, KonJND, sealed/T0, HDR VAL, UPIQ, external panels. The exposure-ledger entry is written at assessment time.

**No raw forced-choice data.** KonFiG's D1 labels are the authors' reconstructed JND scale from boosted triplets; E34
reads only those values. The not-sure-as-half rule, the dataset's worker screening and `fit_observer_slope` apply to no
E34 read. Any forced-choice read added later must apply them.

## 6. Falsification plan

**Registered conclusion:** "these interventions (W at H32, K at its piloted λ, L) do not reach a point that both steers
and keeps rank". It is drawn when all of these hold:
- no family has an eligible arm;
- W and K both fail G-STEER at full-data seed 0 in both families;
- L-A reaches 135/135 but fails E21 as-good.

E34 tests three points of the one-hidden-layer `N` class. It cannot show the whole class unlearnable.

**Interpretive criterion, beside the conjunction and not part of it:** W's or K's geometric-mean Rel. L2 ratio to its
family control on the 135 cases is ≤ 0.70 with both clustered upper bounds (percentile, Student; G = 15) below 1. It
separates "more linear but not enough" from "not more linear".

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
- **P7.** Statistics owners (guidance §4): bump zensim's `zenstats` pin to `zenmetrics-stats` ≥ `1e48c1ea` (the
  migration lane); implement 4.2–4.6 on it in a Rust owner (`panel evaluate` clustered comparisons or `bake_verdict`),
  with the existing Python E21 scorer as an independent reference reproducing E33's recorded seed-unit numbers. Ask
  zenmetrics for a public Student-interval helper (the t quantile now comes from scipy) and a dataset-stratified design.
- **P6.** L serving smoke: an `in×1×1` `N` bake at f16 passes identity (C5, E3–E8), the steering instrument and the
  served-path gates. Then smokes per arm and family on the slowest worker class, caps, dry placement, parity pre-check
  and the freeze record.

## 8. Open questions for the owner

1. **G-STEER bar:** 135 at seed 0 with seeds 1–2 non-inferior (as drafted; weak), 135 on all three seeds, or seeds
   1–2 each ≥ max(the control's count, 133)?
2. **Families:** both (305 fits, about 95 cell-hours before K's overhead), or one? C costs about 2× per cell.
3. **W:** keep it (as drafted), or drop it and run K alone as the only nonlinear intervention (saves 86 fits)?
4. **L:** adoptable last (as drafted) or report-only, given the prior record of sign-constrained and linear scores (appendix A4)?
5. **K's λ:** a pilot on the 8-origin dev panel, or one fixed λ?
6. **Augmentation codecs:** all four teacher codecs, or JXL only like the broad panel?
7. **The M2 bar at b64:** at 12 blocks it tolerates one adjacent swap. Intended?
8. **Hosts:** is the r5600g loan available? That sets wall time (about 12 h on the five fleet hosts, unmeasured).
9. **Margin and power (new):** clustered non-inferiority at −0.002 with Holm can fail an arm that truly equals its
   control if the clustered se is large; E33's seed-unit SE was 0.00075, and the clustered se is unmeasured. Keep the
   margin, widen it, or authorize one ledgered read of E33's stored E21 predictions (already exposed) to measure the
   clustered se before freeze?
10. **Pair classes (new):** report-only (as drafted), or should same-ladder and cross-codec agreement also gate?
11. **Holm family (new):** one family of nine (as drafted), or separate families for eligibility and improvement?

## 9. What E34 does not claim

E34 tests three changes in the one-hidden-layer `N` class against matched controls on already-exposed D1 populations
and label-free TRAIN probes. It doesn't qualify a model, measure perception of steering, pick A or C, show the class
unlearnable, or change any threshold or rule.
