# E34 freeze preparation — revision 5, INCOMPLETE / NOT FROZEN

This candidate retains revision 4 below. The owner approved its defaults as
recorded in the preparation brief. It cannot be frozen until the pending
receipts and qualifications listed here exist. No registered gate is changed.

| Binding | SHA-256 or state |
|---|---|
| Revision 4 registration | `d5716d80db1043be8e0f4991250a732764bc72aa37d1fea78754f892b61a73a3` |
| Appendix | `aeac526d454d8130e75e09294d5382111dff23aaf9e24095068d7a42dd54389a` |
| E33 fit contract | `3bb4705b8a07307b3dec3956f43d99ae649ea333b6c95817ac389167713ec4d7` |
| E33 data archive | `9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008` |
| P1 feature census | `747149eeb7e1175cf7e3b61d2c1bb7f2e1ad87c04cbc530c289952ae592f77c3` |
| Original 135-case packet | `b2af8465520139da6ebeede71c0c73c688d0d579e52e512cbf53f8181aec5ed9` |
| Family map capture | `9d07a0f63ef5fa167c5333535010f44b4ab9a087f04e560521b6d1aa1961820c` |
| Origin split owner capture | `58e992526d501596c6912d3163df7f5c7ef6d2b0e215fb00d928b5de9866e700` |
| P4 source preflight | `45e6c65c9b6c474504db37ea38cb08909f6c9b939090f6f3d41e58756e0b1ac5` |
| P4 eligibility, dHash, SafeSyn intersection, selection and builds | `PENDING — official SDR origins 1444 and 1458 absent` |
| P5 complete augmentation-builder validation | `PENDING` |
| P6 first-epoch and serving gates; E34 worker image/jobset | `PENDING` |
| P7 statistics implementation and parity | `PENDING — public helper review` |
| One-time E33 noise exposure and measured SEs | `PENDING — NOT READ` |

P2 currently binds an unfitted engineering fixture, not an adopted or qualified
L bake. The definitive L coefficient comparison and serving gates remain open.
See [preparation evidence](e34_preparation_2026-10-15.md).

---

# E34 — linear-along-repair training for the model-limited G-STEER failures (registration DRAFT, revision 4)

Status: **DRAFT revision 4, 2026-10-11. Not registered and not approved.** No fit, no fleet job and no label read was
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
  - **Source-clustered intervals for every claim.** The characterization's comparative claims (§1.3) now carry
    `zenmetrics-stats` intervals clustered by source picture, with the Student interval beside the percentile one
    (G < 30). One claim, the distortion-level effect, no longer holds under the Student interval and is now descriptive.
  - **The decision rule is restated as source-clustered tests with Holm** (§4). E21 as-good becomes a paired,
    reference-clustered non-inferiority test at the E33 tolerance; the sequential-incumbent improvement tests become
    paired reference-clustered tests; all nine form one Holm family fixed now. The seed-unit t test is reported, not used.
  - **Pair classes** (same ladder, cross-codec, cross-source) are reported for every human-agreement comparison (§4.6).
  - **One percentile rule** (Hyndman–Fan type 7) with B, seed and rule recorded; **all statistics from
    `zenmetrics-stats`**, through zensim's owners after the pin bump (P7). E34 reads no raw forced-choice data (§5).
  - **AIC2026 labels** join the never-read list.
- **Revision 4** (review round 3, FIX-FIRST, text only; no new data, fits or cost):
  - **The eligibility and improvement tests keep the training-seed variance** E33's rule measured: a two-way
    reference × seed bootstrap, arm and control paired on the seed index, with the Student rule at df 9 deciding
    beside the bootstrap p (§4.1–4.3). Revision 3 averaged seeds inside each draw, which left training noise out.
  - **The Student standard is applied like for like in §1.3.** The b64 rate effect and the b8/b16 Rel. L2 readings
    become descriptive, because the pictures that carry them are too few; the Rel. L2 claim is now the within-case
    comparison, and a column gives the pictures on each side (§1.3, §1.4, H1).
  - Round-3 P3s: the one-sided p is stated directly, the seed offsets are listed, KonFiG's 7 references and their
    draw effects are given, the pair-class inputs are pinned, and the test is named as E34's.

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
are one picture, 4 KADID references). `zenmetrics-stats` `resample::cluster_bootstrap`, `Design::OneWay`,
`weighted_mean` (rev `1e48c1ea`); B 10,000, seed 20261011, Hyndman–Fan type 7, level 0.95. Student interval
`estimate ± t(G−1)·se·√(G/(G−1))`, t from scipy (the crate has no public t quantile; P7). Floor ties are excluded from
every rate. Raw: `e34-rev4-2026-10-11/clustered.json` `c8a1ba8f…` (revision 3's seven rows reproduce bit for bit).

**Like for like:** a claim holds only if both intervals exclude the null **and** G counts the pictures that can inform
it. "Pictures a / b" are those with rows on each side of the contrast.

| Claim | Pictures a / b | G | Estimate | Percentile 95% | Student 95% | Reading |
|---|---|---:|---:|---|---|---|
| failure rate b64 − b8–b32, all panels | 8 / 15 | 15 | +0.106 | [0.021, 0.197] | [0.007, 0.205] | **descriptive:** 7 pictures have no b64 case and add only small-block rows |
| same, broad panel (like for like) | 8 / 8 | 8 | +0.095 | [0.002, 0.181] | [−0.020, 0.211] | **descriptive** |
| failure rate level 10 − level 17, broad | 8 / 8 | 8 | +0.049 | [0.004, 0.094] | [−0.010, 0.109] | **descriptive** |
| Rel. L2 ratio failing / passing models, **within case**, all blocks | 10 | 10 | 6.38× | [3.47, 11.41] | [3.08, 13.22] | **holds**; failing models larger in 31 of 32 cases |
| same, b8 | 2 | 2 | 2.75× | [1.38, 10.89] | [0.00, 1.6e6] | descriptive (3 cases) |
| same, b16 | 4 | 4 | 3.30× | [1.62, 13.63] | [0.48, 22.58] | descriptive (6 cases) |
| same, b32 | 8 | 8 | 8.06× | [4.66, 13.92] | [4.06, 15.98] | holds (10 cases) |
| same, b64 | 6 | 6 | 8.79× | [5.00, 19.47] | [3.25, 23.78] | holds (13 cases) |

**Within case:** a (case, block) that some served models fail and others pass, floor ties excluded. Its value is the
mean log Rel. L2 of the failing models minus the passing models, so picture, distortion and block are fixed and only
the model differs; ratios are exp of the clustered mean. Revision 3's between-picture ratios (3.3–7.4×, G = 15) rest
on 2 failing pictures at b8 and 4 at b16, so they are descriptive context now (appendix A7).

### 1.4 What the failures share

- **Block size (descriptive):** b64 fails more than b8–b32 on the point estimate, in 6 of 8 broad pictures (A7), but
  the like-for-like Student interval includes zero. The mechanism doesn't rest on the rate: at b64 a case has 12–16
  blocks, so M2 ≥ .99 allows one to three adjacent swaps, and dropping one block rescues 12 of 23 b64 failures (A2).
- **Curvature:** on the same case, the models that fail have 6.4× the linearization error of the models that pass
  (31 of 32 cases; Student [3.1, 13.2] over 10 pictures). Per block it holds at b32 and b64; b8 and b16 rest on 2 and
  4 pictures and are descriptive (above). R5STEER2 shows the same on `by_v2fy` bakes, by analogy (A3).
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
| H1 | **Curvature:** uneven unit switching along repairs causes the misranking | Same case, failing models have 6.4× the Rel. L2 of passing ones (31/32 cases, Student [3.1, 13.2], 10 pictures; holds at b32 and b64); R5STEER2 (by analogy) | b8 cases fail too (n = 704); b8/b16 and the b64 rate effect are descriptive |
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
- **Both noise sources are resampled.** `Design::TwoWay(reference image, training seed)`, the pigeonhole bootstrap:
  references (labels unique across datasets) and seeds 0–9 are drawn together. Seeds cross references, since every
  seed's model scores every reference. Each row holds the arm's and the control's predictions for the same image and
  seed index, so one draw resamples both alike (paired). The pigeonhole variance errs large, the safe side for a
  retention test. The reference-only (`OneWay`) se and E33's seed-unit SE are reported beside it, so each source shows.
- **Draws:** B 10,000; type 7; level 0.95; seed 20261011 + offset. Offsets: test k of §4.4 uses 100·k; its dataset
  guards 100·k + d (d = 1 KADID, 2 TID2013, 3 KonFiG, 4 CID22-A); its W2 se 100·k + 5; its reference-only se
  100·k + 6. §4.6 uses 1000 + 10·k + c (k = the arm's eligibility test, c = 1 same ladder, 2 cross-codec, 3
  cross-source); §4.7's report-only intervals use 2000, 2001, … in the order listed. Every result records rule, B,
  seed and finite draws.
- **One-sided p** for `θ` (= Δ + 0.002 in 4.2, Δ in 4.3): the bootstrap p is (1 + #{θ* ≤ 0}) / (B_finite + 1), read
  from the draws. It equals `p_zero/2` whenever the lower tail is the smaller one, the only case where p < 0.5. P7
  asks the crate for it.
- **Few levels, like for like with §1.3.** The Student p, from t(G − 1) on θ̂ / (se·√(G/(G−1))) with G the smallest
  level count among the resampled factors, is computed beside the bootstrap p, and **the larger of the two decides**.
  For every pooled test G = 10 (the seeds), so df 9, as in E33's seed-unit test.
- **Few-source datasets:** KonFiG has 7 references (E33's 756 rows = 7 × 108), TID2013 25, CID22-A 25, KADID
  TRAIN+SELECT 65 (122 in all). A draw with fewer than 3 copy-aware KonFiG references is non-finite, dropped and
  counted: 2.6% of draws (binomial, 122 draws at 7/122). KonFiG's reference count also changes in about 85% of draws.
  The stratified design requested in P7 removes both.

### 4.2 E34's eligibility test on the E21 statistic

This is E34's own test. E21's registered meaning for other experiments is unchanged, and **E33's point form** (mean
Δ ≥ −0.002, every dataset ≥ −0.005, W2 Δ > −2·SE with E33's seed-unit SE) is reported beside it.
- **Statistic:** `Δ` = the mean over the four D1 datasets (equal weights) of [the arm's signed SROCC minus the paired
  control's on the same seed index, averaged over the draw's seed copies]. Each SROCC is on the dataset's
  withheld-fold rows, recomputed on the draw's copy-aware expanded rows (`panel::spearman`).
- **Hypothesis:** H0: Δ ≤ −0.002 (E33's tolerance as the margin), p per 4.1.
- **Guards** (point estimates, as E33, each with its interval): every dataset's Δ_d ≥ −0.005; W2 Δ > −2·se, with se
  the two-way se of W2.
- Missing or nonfinite results: INCOMPLETE, p = 1.

### 4.3 Improvement tests

Paired, same statistic and design, arm minus the incumbent it would replace on the same draw and seed index. H0:
Δ ≤ 0; p per 4.1. A replacement also needs the point estimate Δ > +0.002 (E33's practical bar).

### 4.4 One Holm family, fixed now

Nine hypotheses, all computed after every fit, whether or not the verdict consults them:

| # | Test | Family |
|---|---|---|
| 1–3 | eligibility (4.2): W, K, L | A |
| 4–5 | eligibility (4.2): W, K | C |
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

For every arm vs its control and dataset: the paired difference (arm − control, same draw) in sign agreement per pair
class, with its interval: **same ladder** (same reference, same distortion type or codec), **cross-codec** (same
reference, other type or codec) and **cross-source**. Labels: KADID/TID2013 distortion type, KonFiG distortion,
CID22-A codec. Pinned inputs: `pairs::PairTable` under `Design::OneWay(reference)`; `target_tie` 0 (exact target
ties excluded and counted); `orientation` fixed per dataset to E21's sign, never `Auto`; no `family` labels. The pair
walk takes reference × codec, not seeds, so seeds 0–9 are averaged inside each draw and the seed-unit se of the
per-seed differences is reported beside. The
pooled SROCC in 4.2 is mostly cross-source calibration; these classes show where an arm gains or loses. Whether a
class should gate is an owner question (§8).

### 4.7 Verdict, per family: sequential incumbent

1. **Eligible** = Holm rejects its eligibility H0 (4.2/4.4), the guards hold, G-STEER and every gate pass (4.5).
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
- L-A reaches 135/135 but fails its eligibility test (4.2).

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
  migration lane); implement 4.1–4.6 on it in a Rust owner (`panel evaluate` clustered comparisons or `bake_verdict`):
  the two-way reference × seed design with arm and control paired on the seed index, the one-sided p from the draws,
  and the Student rule. The existing Python E21 scorer is an independent reference reproducing E33's recorded
  seed-unit numbers. Ask zenmetrics for a public Student helper (t quantile and tail; the t quantile now comes from
  scipy), a one-sided bootstrap p on `Interval`, and a dataset-stratified design.
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
9. **Margin and power:** the test now counts both noise sources (4.1), which makes it stricter than E33's. Arithmetic
   at the expected value, not a measurement: an arm that truly equals its control clears Holm's first step
   (p ≤ .05/9) only if its two-way se ≤ 0.00060 under the Student rule at df 9 (≤ 0.00067 at the fifth step, ≤ 0.00104
   unadjusted). E33's seed-unit SEs alone were 0.00075 (A) and 0.00126 (C); E33's A (Δ −0.00089) would have failed on
   seed noise alone (one-sided p 0.085 at df 9). So eligibility will mostly mean "measurably better" or "much less
   noisy". Keep the margin, widen it, or authorize one ledgered read of E33's stored E21 predictions (already exposed)
   to measure the two-way se, and each component, before freeze?
10. **Pair classes (new):** report-only (as drafted), or should same-ladder and cross-codec agreement also gate?
11. **Holm family:** one family of nine (as drafted), or separate families (eligibility and improvement, or one per
    model family)? Separate families are less strict; with question 9 this sets how often any arm can be eligible.

## 9. What E34 does not claim

E34 tests three changes in the one-hidden-layer `N` class against matched controls on already-exposed D1 populations
and label-free TRAIN probes. It doesn't qualify a model, measure perception of steering, pick A or C, show the class
unlearnable, or change any threshold or rule.
