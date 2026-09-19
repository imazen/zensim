# DVIFM Phase 2d — preregistration (2026-09-19)

Preregistered before any Phase-2d fit, adaptive tuning, or holdout read.
Committed BEFORE the CID22 49-ref human-score access described in §4
(exposure entry is part of the same commit). Scope this run: **D → C → A**
only; Parts B, E, F of the 2026-09-19 dispatch are deferred and are not
started here.

Output root: `/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/`
(`$OUT` below). Phase disk cap 60 GB; `/mnt/v` must keep ≥80 GB free.

## 0. Standing verdict vocabulary

Same as screen2b/2c:

- **ADVANCE** — the arm beats its control on the registered composite and on
  every required leg beyond threshold `2·SD/√5` (5 seeds → `2·SD/√5 ≈
  0.894·SD`), with sign count ≥ 4/5 on paired per-seed deltas.
- **INFO-NOT-USEFUL** — real signal (beats its permuted twin) but does not
  beat the control, or wins some legs and loses others without a composite
  win.
- **NEGATIVE** — does not beat the control, or fails the mechanism check
  (no separation from the permuted twin).
- **INCOMPLETE** — evidence missing or under-budgeted (an arm that has not
  plateaued, a leg that could not be scored, a cut arm). Never silently
  folded into a pass.

For Part D's standalone replication claim the same scale applies per
corpus, stated in §5.

## 1. Frozen references (read-only)

Frozen Rev3 leaders (recovery of 2026-09-15, read-only):

- `R915_basic228_h128_ens5` — profile `basic228` (feature ids 0..227),
  hidden 128, five members seeds 17101/17103/17107/17111/17113.
- `R915_y60_h32_ens5` — profile `y60` (60 registered ids), hidden 32,
  same five seeds.

Recorded under `/var/tmp/zensim-validation-2026-09-15/recovery/`:

- `fits/R915_<profile>_s<seed>.raw.bin` + `.raw.bin.spec.json` — the
  spec.json embeds the exact trainer argv and the sha256 + row count of
  every input table. The argv is the reproduction contract.
- `dedup-tables/{safesyn,cid22,codec,human}_{fit,development}.parquet` —
  168,142 fit rows after exact within-leg dedup (`input_table_row` indexes
  the pre-dedup `tables/` row).
- Weights recorded in argv: train `safesyn 1.0168526508775275`,
  `cid22 1.0115735134169879`, `human 0.5041192364219411`,
  `codec 0.6060902647942771` (all `withinref`, human `rank`, others
  `both`); development-val `safesyn 0.5`, `cid22 2.0`, `human 1.0`,
  `codec 1.0`. 120 epochs × 50,000 draws, `--pair-sampling uniform`,
  `--target-scale 1`, `--mse-weight 1`, `--early-stop-patience 0`,
  `--val-policy mean`, `--val-aggregate geomean3`, `--out-dtype f32`,
  `--no-auto-eval`. Seed i uses `--seed <seed> --init-seed <seed>
  --sample-seed (i)·1e9` — the disjoint streams (i = 1..5).
- Environment: `RAYON_NUM_THREADS=1 ZENSIM_FORMULA_REV=3
  ZENSIM_SAMPLE_DIGEST=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
- Post-fit: `bake_dial_refit densify` for serving; calibration exists under
  `calibrated/` but Part A decisions are rank-based on TRAIN development
  legs, so the comparison consumes raw fits + densified bakes only (same
  as 2b/2c; recorded in the report).

## 2. Part A — leader-recipe arms (exact)

### 2.1 Leader reproduction gate (A.1)

Re-run the recorded `R915_basic228_h128_s17101` argv verbatim against the
recorded `dedup-tables/` (sha-verified against the spec.json table
hashes) under the recorded environment. **Gate:** the produced
`.raw.bin` must be bit-identical to the frozen one, else STOP →
`~/tmp/devin/dvifm2d_BLOCKED.md` (a stated measured difference with a
proven cause is the only alternative the dispatch allows).

### 2.2 w986 re-extraction of the R915 estate (A.2)

- Source rows: the original pairs TSVs
  (`/var/tmp/zensim-validation-2026-09-14/baseline-recovery/*-pairs.tsv`
  lineage for cid22/safesyn; the human/codec leg TSVs as recorded in the
  recovery manifests) — every dedup-table row is re-extracted in dedup
  order.
- Route: `extract_features_372col --full-986 --dvifm-spec <spec>`
  (SDR native route, `ZENSIM_FORMULA_REV=3`), one run per input plane
  required by the arms (`xyb_y` default; `ycbcr_y`, `ycbcr_cb`,
  `ycbcr_cr` after Part C lands).
- Join key = the extracted f0..f943 columns: a dedup row joins an
  extracted row iff the 944 base columns are **bit-identical** and
  `ref_basename` matches. This makes the bit-identity check (first 956
  columns vs the R915 tables, i.e. f0..f943 + csfw f944..f955) intrinsic
  to the build: any row that fails to join is reported, not silently
  dropped. If the join is not 100%, STOP → BLOCKED (extraction era moved).
- Timing: measure wall time on the first 1,000 rows per plane, record
  projected job length in the progress log before running the rest.
- Outputs per plane: `<leg>_<split>.w986.<plane>.parquet` +
  `_MANIFEST.json` (`build_commit`, extractor sha256, spec sha256,
  input TSV sha256s, row-key hash = sha256 of the joined row-key list).

### 2.3 Arms

Identical recipe/seeds/streams/budget; only input ids differ:

| arm | table | keep-features | hidden |
|---|---|---|---|
| `basic228` (control = leader) | w986 xyb_y | 0..227 | 128 |
| `basic228+dvifm30[xyb_y]` | w986 xyb_y | 0..227 ∪ 956..985 | 128 |
| `basic228+perm30` | w986 xyb_y permuted | 0..227 ∪ 956..985 (row-permuted) | 128 |
| `basic228+dvifm30[ycbcr_y]` | w986 ycbcr_y | 0..227 ∪ 956..985 | 128 |
| `basic228+perm30[ycbcr]` | w986 ycbcr_y permuted | same permuted | 128 |
| `basic228+dvifm90[ycbcr]` | joined 3-plane w1046 | 0..227 ∪ 956..1045 | 128 |
| `basic228+perm90[ycbcr]` | joined 3-plane permuted | same permuted | 128 |
| `y60` (control) | w986 xyb_y | y60 ids | 32 |
| `y60+dvifm30[xyb_y]` | w986 xyb_y | y60 ∪ 956..985 | 32 |
| `y60+perm30` | permuted | permuted | 32 |
| `y60+dvifm30[ycbcr_y]` | w986 ycbcr_y | y60 ∪ 956..985 | 32 |
| `y60+perm30[ycbcr]` | permuted | permuted | 32 |
| `y60+dvifm90[ycbcr]` | joined 3-plane | y60 ∪ 956..1045 | 32 |
| `y60+perm90[ycbcr]` | joined permuted | permuted | 32 |

5 seeds each → 70 fits at the leader budget (120 epochs × 50,000 draws).
Permutation contract (same as 2b/2c): one recorded row permutation of the
DVIFM columns within each leg table, fit and development legs permuted
independently with distinct recorded seeds; the same permutation is reused
for every seed of that arm.

Joined 3-plane research table layout (documented, never silently
column-mixed): `f0..f943` (base 944, asserted bit-identical across the
three plane extractions) + `f944..f955` (csfw, identical) +
`f956..f985` = DVIFM of the `ycbcr_y` extraction +
`f986..f1015` = DVIFM of the `ycbcr_cb` extraction +
`f1016..f1045` = DVIFM of the `ycbcr_cr` extraction → width 1,046,
`--max-features 1046`. The per-plane spec sha256s are recorded in the
table `_MANIFEST.json`.

Trainer stamps: every emitted bake carries `zentrain.formula_revision=3`
and `feature_set_id` in its ZNPR metadata (the driver's responsibility;
asserted by re-reading the emitted bytes).

### 2.4 Decision statistics (TRAIN development legs only)

Per arm × seed: `predict_features_with_bake` over each development leg's
feature bin → `panel` (canonical owner): pooled signed SROCC,
per-reference batch SROCC (min_len 3, `Orientation::Auto` semantics as
pinned in the 2b/2c driver), per-family and per-ladder SROCC where the
leg has family labels, codec leg split per codec. The registered
composite = `geomean3` over the four dev legs with the recorded weights
(safesyn 0.5, cid22 2.0, human 1.0, codec 1.0) — matching
`--val-aggregate geomean3`.

Paired per-seed Δ(+dvifm − control) and Δ(+dvifm − +perm): mean, SD,
sign count, `2·SD/√5` threshold, plus the within-reference / local
ordering panels. Verdicts per §0.

Planned comparisons:

- `+dvifm30[xyb_y]` vs control (does the leader recipe rescue it).
- `+dvifm30[xyb_y]` vs `+perm30` (capacity control).
- `+dvifm30[ycbcr_y]` vs `+dvifm30[xyb_y]` (colour space effect).
- `+dvifm90[ycbcr]` vs `+dvifm30[ycbcr_y]` (what chroma adds).
- `+dvifm90[ycbcr]` vs control (the native arm's verdict).
- Each dvifm arm vs its own permuted twin (mechanism check).

## 3. Part C — native Y′CbCr planes (implementation contract)

- `input_plane ∈ {xyb_y, ycbcr_y, ycbcr_cb, ycbcr_cr}` added as a field of
  the training-gated `research::DvifmSpec` (default `xyb_y`) and
  `dvifm::DvifmParams`; surfaced in the spec JSON as top-level
  `"input_plane"` (absent = `xyb_y` → byte-identical default behaviour and
  identical spec hash handling for existing specs).
- Conversion: full-range BT.709 on the gamma-encoded sRGB input,
  computed in the same streaming strip machinery that feeds
  `dvifm_push_rows_walk` (a side-converter next to
  `convert_source_to_xyb_into_slices`, same owner
  `feature_v2_stream.rs`/`streaming`): per sRGB triple
  `Y′ = 0.2126 R′ + 0.7152 G′ + 0.0722 B′`,
  `Cb = (B′ − Y′) / 1.8556`, `Cr = (R′ − Y′) / 1.5748`.
  Ranges: `Y′ ∈ [0,1]`, `Cb,Cr ∈ [−0.5,0.5]`.
- Per-plane `DvifmNorm` applied by the existing `normalize_row` rule
  `(v − min)/scale`: `ycbcr_y → {min 0, scale 1}`,
  `ycbcr_cb → {min −0.5, scale 1}`, `ycbcr_cr → {min −0.5, scale 1}`,
  `xyb_y → DVIFM_NORM_SDR` (unchanged). HDR route keeps `DVIFM_NORM_PU`
  on `xyb_y` only; a non-`xyb_y` input plane on the HDR front-end is a
  hard error (out of scope this phase).
- Same 5-level pyramid, 5×5 blocks, visibility, 30 features/plane.
  Per-plane block caches: the existing `--dvifm-block-stats` output is
  per spec (hence per input plane); three runs = three caches — the
  machinery for all three Y′CbCr planes is kept general (Part B/E/F
  can reuse it).
- No new registered slots: a plane variant lands in the same
  `f956..f985` under a distinct spec sha256 + `feature_set_id`.
- Bit-identity gates: `input_plane = xyb_y` and dvifm-off are asserted
  bit-identical to main in a fixture test;
  `scripts/dvifm_parity_fixture.py` gains a `ycbcr` section (closed-form
  sRGB triples → expected Y′/Cb/Cr plane values + feature vectors),
  fixture ≤30 KB.
- Constants are per-plane from that plane's own block cache; XYB
  constants are never reused on Y′CbCr data.

## 4. CID22 49-ref human set — A/B exposure protocol

Split: the 49 references of `/mnt/v/dataset/cid22/CID22/original/`,
sorted by basename, Fisher–Yates shuffled with recorded seed
`20260919` → first 25 refs = **CID22-A** (fit-allowed for Part D
constants only), last 24 = **CID22-B** (sealed).

Ledger (committed with this prereg, BEFORE any MCOS value is read):
appended to `docs/DATASET_HISTORY.md` and the exposure-ledger section of
`docs/DATA_SPLITS.md`: date, user direction quote, the 25 A-ref names,
what is fitted (≤ ~90 scalar constants + a throwaway per-domain affine —
no MLP, no feature selection, no checkpoint selection), and the
consequence: any artefact consuming CID22-A-derived constants may quote
CID22 only on subset B, labelled `CID22-B(24)`.

Permitted reads:

- CID22-A rows: MCOS + pixels, for Part D constant fitting only.
- CID22-B rows: exactly ONE frozen read after all Part D variants are
  frozen — SROCC/KROCC/PLCC vs MCOS per variant + the comparator table of
  §5.4. No iteration afterwards.

Before this commit, only membership columns (`reference_img`,
`distorted_img`, `encoder`, `setting`) were read — never `MCOS`.

## 5. Part D — standalone DVIFM replication (runs first)

### 5.1 Model (convex by construction)

Per plane `p ∈ {Y′, Cb, Cr}`, per level `l ∈ 0..4`:

- block error `e_b = m_b^{P_{p,l}}` (m = max |ref−dist| over the 5×5
  block — the existing `BlockRec.m`);
- visibility `v_b = max(vis(C_s), vis(C_d))` with `vis` = the existing
  `pool_block` visibility (a = 1/C₀ knee, β negative exponent, ς
  smoothness, c_hi ceiling);
- pooled `s_{p,l} = mean_b(v_b · e_b)` — exactly the existing F1
  quantity, so scoring a fitted spec reuses the standard extraction;
- level error `E_{p,l} = s_{p,l}^{1/P_{p,l}}` (the learned Lp exponent);
- plane error `E_p = Σ_l w_{p,l} · E_{p,l}`, `w_{p,·}` on the simplex;
- total error `E = c_Y·E_Y′ + c_C·E_Cb + c_C·E_Cr`, `(c_Y, c_C, c_C)` on
  the simplex;
- reported score `= 100·exp(−λ·E)`, `λ > 0` fitted inside the same
  objective (monotone; does not affect SROCC/KROCC, only PLCC/MAE).

Parameter count: per-(plane,level) `{g, P, C₀, β, ς}` = 75, per-plane
level-weight simplexes 12 free, channel simplex 2 free, λ 1 → **90
parameters** for the 3-plane model. Luma-only (`ycbcr_y` plane): 25 +
4 + 1 = **30 parameters** (the talk's 29 + the score-scale λ).
XYB-Y ablation: same 30-parameter head on the `xyb_y` plane.

Convexity assertions (tested): all `w`, `c` ≥ 0 and each simplex sums to
1; `E` is non-decreasing in every `s_{p,l}` and in every `m_b`; `E = 0`
at identity (all `m_b = 0`); no bias term exists. A unit test checks
monotonicity under random parameters.

Loss (preregistered): MSE between `100·exp(−λ·E)` and the per-domain
min-max-normalised quality target (normalisation constants recorded per
domain: CID22-A MCOS already 0..100 quality-oriented; TID MOS 0..9 →
`/9·100`; KADID DMOS 1..5 → `(5−DMOS)/4·100`; imazen26 ssim2 0..100
already quality-oriented). Optimiser: Adam over the cached block
records (Phase-2 `dvifm_cache.py` lineage for IO; new
`tools/fit_standalone.py` under `$OUT/tools/` for the convex head —
not a second trainer, it is the constants fitter).

### 5.2 Crux: (a = 1/C₀, b = β) per (plane, level)

For each (plane, level) of the fitted variant: coarse log-grid
`C₀ ∈ [1e-4, 0.3]` (≥12) × `β ∈ [0.2, 1.4]` (≥10) holding all other
parameters at their current best → ≥8 multi-start Adam refinements from
the best grid cells → alternate (per-level constants ↔ weights) until
the fit-domain loss stops improving (relative gain < 1e-4 over an outer
sweep or 3 sweeps, whichever first). The full loss surface per
(plane, level) is committed to `$OUT/surfaces/` as CSV (grid values +
final refined path), and the report states per level: sharpness of the
optimum, agreement across levels, chroma vs luma knee location, and the
fit-loss delta vs the Phase-2 constants (`dvifm-local-fitted-final.json`
= the K1 set).

### 5.3 Fit domains and variants

Domains (all TRAIN-side or ledgered):

| domain | rows | target | refs |
|---|---|---|---|
| `cid22a` | CID22-A subset (~2,2xx rows) | MCOS | 25 refs |
| `tidkadid` | TID JPEG(10)+JP2K(11) all 25 refs; KADID JPEG(10)+JP2K(09) train-digit refs {0,2,4,6,8} | MOS/DMOS→quality | 25 + 40 |
| `imazen26` | wlin7 store subset, train-digit origins {0,2,4,6,8} (cap ~6 GB/plane of block cache) | `score_ssim2` | ≤212 origins |

Variants: `luma` (ycbcr_y only), `native3` (Y′+Cb+Cr), `xyb-y`
(ablation, xyb_y plane), each fitted on each domain separately and on
`cid22a+tidkadid` pooled (the talk's human-data analogue). Prior
constants `K1 = dvifm-local-fitted-final` scored without refit as the
"Phase-2 constants" control.

Block caches per (domain, plane) via `--dvifm-block-stats` under the
spec's `input_plane`; grid search and Adam consume caches only — pixels
are not re-read during tuning.

### 5.4 The single frozen CID22-B read (after all fits are frozen)

On CID22-B (24 sealed refs), one descriptive table: each fitted variant +
`K1` + comparators — SROCC, KROCC, PLCC vs MCOS, plus paired bootstrap
over the 24 references for the Δ-vs-`fast-ssim2` confidence interval.
Comparators scored on the same B rows through their public Rust surfaces:

- `fast-ssim2` (extractor `--audit-ssim2` audit channel),
- `ZensimProfile::B` and `::D` (shipped embedded bakes via
  `predict_features_with_bake` on the w944 B-table),
- `R915_basic228_h128_ens5` + `R915_y60_h32_ens5` (frozen
  `calibrated/*.bin` members via `ensemble_score_rows`).

Also reported (TRAIN-side development, may be read before B): KADID dev
refs last-digit {1,3,5} JPEG+JP2K rows, KonFiG `originsplit_val` rows —
per-domain SROCC for every fitted variant. AIC2026 `proposal-DVIFM*` /
`proposal-DVIFM-0.2*` columns: a single descriptive agreement table
(metric-vs-metric, no human labels — recorded as an AIC2026 exposure in
the ledger after the fact; never a fit target).

Standalone verdict vocabulary: **COMPETITIVE** if CID22-B SROCC ≥
`fast-ssim2` SROCC − 0.02; **WEAKER-BUT-REAL** if ≥ 0.85 but below that
band; **NOT-COMPETITIVE** otherwise. Chroma verdict: paired CID22-A
fit-domain + B-read Δ(native3 − luma) with bootstrap CI.

## 6. Hard rules (restated for this run)

TRAIN-only fitting/selection; CID22-B and all T0/secret holdouts
untouched except the single §5.4 read; DVIFM stays default-OFF and
unqualified regardless of outcome; `jj` small commits, no push, no
rewrites of foreign commits; `run-heavy` for every heavy job (≤8 jobs,
one at a time); `.workongoing` refreshed ≤2 min; scratch only
`~/tmp/devin/`; output cap 60 GB, `/mnt/v` ≥80 GB free; records:
`benchmarks/dvifm_screen2d_2026-09-19.{md,json}` + `.pointer.md`,
`board_discussion_sets.json` append (role `train-development`, 2-space
indent), `dvifm_block_gates_2026-09-19.md` outcome update;
`cargo fmt --all -- --check`, `just clippy`, `just lint-scripts` clean;
the five pre-existing `bake_surface.rs` failures untouched.

## 7. Order and cut rules

D first (cheapest, most decisive) → C arms → A arms. If wall time or
disk forces a cut: cut A before C, never D. Fewer arms run at full
seeds/budget rather than shrunken versions of more.

## Amendment 1 (2026-09-19, post-stop fitting correction)

Supervisor stop of the first Part-D fit run (20:45Z) found the standalone
fit was measuring the wrong thing, plus one data bug found while applying
the mandated orientation check. This amendment replaces the §5.2 fitting
procedure and records a target-column correction. Original text above is
unchanged; where they conflict, this amendment governs. **No
decision-relevant result had been read**: every number produced before the
stop was fit-domain (CID22-A / TID / KADID-train fit losses and surfaces —
themselves invalid, now discarded to `.bak`); CID22-B labels remain sealed
(zero reads of `human_score` on `cid22b.tsv` — it is emitted blanked).

### A1.1 Measured defects being fixed

1. **Scale-confounded grid.** `grid_sweep_level` scored each (C₀, β) cell
   as `100·exp(−λ·E)` at the *current* λ — λ was not refit per cell, so the
   surface ranked cells by how well they rescale E to a fixed λ rather than
   by masking quality. Measured on `cid22a.tsv`: target variance 172
   (sd 13.1) vs init MSE 908 and logged "best" grid cells 775–894 — 4.5–5×
   *worse* than predicting the mean. `tidkadid.tsv`: variance 751, init MSE
   3,777 (see A1.3 — half that pool was also inverted). The K1 control on
   the same rows, which refits λ, reached MSE 64.7 — confirming the fixed-λ
   objective, not the model class, was the failure.
2. **Every grid optimum on the grid edge.** C₀ = 0.3 (upper bound) and
   β = 1.267–1.400 (upper bound) on every level/variant logged (cid22a
   luma l0/l1/l2, xyb-y l0/l1, native3 l0, tidkadid luma l0). A bounded grid
   whose optimum is always the corner is not identifying an optimum.
3. **`native3` (the primary arm) crashed** — `ZeroDivisionError` in
   `fit_variant` because the grid's best cell β = 1.4 is a logit infinity
   under the [0.2, 1.4] bounded parameterisation. It was never rerun.

### A1.2 New objective and search (replaces §5.1 loss tail + §5.2)

- **Output map fitted per evaluation.** Every loss evaluation — sanity
  gate, every grid cell, every Adam objective call, every accept/reject,
  every reported MSE — first fits `ŷ = A·exp(−λ·E) + B` with **A > 0**
  (score monotone in −E; identity stays top of scale). For a given λ, A
  and B are the closed-form least-squares solution on basis `exp(−λE)`;
  if the unconstrained fit gives A ≤ 0 the constrained optimum is the
  boundary A→0⁺, i.e. ŷ = ȳ, MSE = var(y) — the cell is honestly recorded
  as "no better than constant". λ is selected by golden-section search on
  log λ over [1e-9, 1e9], ≤ 40 evaluations. Gradients for Adam use the
  envelope-theorem form: (A, B, λ) held at their argmin while differentiating
  w.r.t. level/head parameters.
- **Per-cell records.** Each grid cell records the refit MSE **and** the
  scale-free rank criteria of −E vs y (SROCC and KROCC). Selection is by
  refit MSE; all three surfaces are published per (plane, level, sweep) in
  `surfaces/<dom>_<var>/grid_*.csv` (`c0,beta,mse_refit,srocc,krocc`).
- **Sanity gate before any grid.** With the map refit, the init parameters
  must beat the constant predictor (MSE < var(y)) on each fit set; both
  numbers are printed. If not, the fit stops with diagnostics (per-subset
  means, SROCC(−E, y)) before fitting anything.
- **Widened grid + parameterisation.** Grid: C₀ ∈ [1e-4, 3] log-spaced ×
  18 points, β ∈ [0.05, 3.0] log-spaced × 18 points (the β spacing choice
  is recorded; log covers the 60× range uniformly). Parameterisation bounds
  strictly contain the grid: g ∈ (0.2, 2.0) logit (unchanged), β ∈
  (0.01, 10) logit, C₀ = exp(raw) with raw clamp [ln 1e-8, ln 1e4],
  ς = 1 + softplus (unchanged). Grid endpoints encode interiorly
  (lo + ε, hi − ε); no grid value is a logit infinity. If a grid optimum
  lands on an edge, that axis is extended once (8 points, same spacing);
  if it still lands on the edge the edge optimum is **reported as the
  finding** (e.g. "β wants > 3" / "the knee sits above every observed
  contrast = masking effectively off at this level"), never silently
  clamped.
- Everything else in §5.2 stands: ≥8 multi-start Adam refinements per
  (plane, level), alternating with the head; convex-by-construction
  assertions unchanged (A > 0 keeps the score non-increasing in E and
  E = 0 at identity ⇒ score = A + B, the fitted top of scale); sharpness
  (loss increase at ±1 grid step per axis), cross-level agreement and
  luma-vs-chroma knees reported from the surfaces.
- `eval-consts` (K1 control) uses the same map family: constants and
  uniform level weights as given, (A, B, λ) fitted by the same procedure —
  a 3-scalar head, still no MLP/selection.
- Artefact schema → `dvifm-standalone-fit-v2` (adds `map {A,B,lambda}`,
  per-level edge/sharpness diagnostics; `score` accepts v1 artefacts via
  `lambda` → map (100, 0, λ)).

### A1.3 Target orientation finding and correction (KADID legs)

The mandated pre-pooling orientation check (per
`scripts/canonical_corpus/check_target_orientation.py` ground truths)
measured, on `kadid_train.tsv` (400 rows, refs {0,2,4,6,8}) and
`kadid_dev.tsv` (250 rows, refs {1,3,5}):

- stored `human_score` ≡ `(5 − dmos)/4` exactly (max |err| = 0 vs
  `kadid10k/dmos.csv` join);
- KADID's `dmos.csv` column is **quality-oriented** (a MOS in disguise —
  Appendix F of the 2026-08-03 campaign): its mean on the joined train
  rows falls 4.434 → 4.245 → 3.904 → 2.211 → 1.384 across levels 1→5;
- hence `corr(human_score, quality) = −1.0` — **both KADID legs were
  stored exactly inverted**, and `tidkadid.tsv` / `pooled_*` pooled 400
  inverted rows with 250 correct ones.

TID verified correct: `human_score = MOS/9` exactly (max |err| = 0 vs
`mos_with_names.txt`), corr = +1.0. CID22-A = MCOS/100 (quality-oriented
by definition). imazen26 = `score_ssim2` (metric oracle, quality-oriented).
KonFiG val = `1 − q/3.2` (already quality-oriented per Appendix L).

**Correction**: all KADID-derived rows re-emitted as
`human_score = (dmos − 1)/4` — the canonical quality transform — identical
row order (block caches join on row order and stay valid). Old TSVs renamed
`.bak`, never deleted; manifests updated with `target_orientation` verdicts.
The range question is thereby answered: `tidkadid` spanning 5–97.5 while
CID22-A spans 28–92 is a genuine data property (the JPEG/JP2K ladder runs
to more severe distortion than CID22's operating range), *not* a scale
mismatch — after correction all sets sit on the shared 0–100 quality axis
`quality = human_score·100` with no further per-set affine.

No refit result from the inverted targets is carried forward; the `.bak`
surfaces/fits from the stopped run are retained for audit only.
