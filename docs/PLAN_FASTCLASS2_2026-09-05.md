# PLAN — FASTCLASS2: a 156-or-156+cheap model with 944-class RANK

**PRE-REGISTERED 2026-09-05, BEFORE ANY FIT.** Arms, seeds, shapes, sets,
gates, bars and the selection rule below are frozen at the commit that
introduces this file. Results append to
[`benchmarks/fastclass2_campaign_2026-09-05.md`](../benchmarks/fastclass2_campaign_2026-09-05.md);
nothing in §1–§7 here is edited after a number exists except in a section
explicitly labelled AMENDMENT, with its reason and timestamp.

USER DIRECTIVE (2026-09-05, verbatim): *"improve all fundamental things like
model matrix shapes and naming as needed to find a high performance but
944944mlp competitive 162 or 162 plus cheap model, you can improve the
kernel"*.

---

## 1. THE TARGET, stated as a number before anything is fit

The bar is the FAIR board's **replicated** 944 leaders
([`replication_wave_2026-09-05.md`](../benchmarks/replication_wave_2026-09-05.md)
§4c.1), k-mean over their own seed groups. Read from their stored fullevals by
this lane (not re-derived):

| leader | k | composite | CID22 | KonJND \|·\| |
|---|--:|--:|--:|--:|
| `W10L9PH` | 6 | 0.859350 | 0.884850 | 0.460850 |
| `W11J` | 7 | 0.859286 | 0.887357 | 0.462129 |
| `A5_r4` | 2 | 0.859450 | 0.882850 | 0.476950 |

**COMPETITIVE means, per axis, at k = 3, mean-vs-mean:** composite ≥ 0.8593,
CID22 ≥ 0.8849, KonJND ≥ 0.4609 — the *weakest* of the three leaders on each
axis, so the target is "inside the leader band", not "beats the best draw".

**And the incumbent fast class is already past two of those three.** Read the
same way from the fastclass wave's own k=3 cells:

| fast-class cell | k | composite | CID22 | KonJND |
|---|--:|--:|--:|--:|
| `FC_C0` (= published A4b, 156+free) | 3 | 0.857167 | 0.886233 | 0.356067 |
| `FC_D2` (+hf-withinref) | 3 | **0.865700** | **0.890500** | 0.431500 |
| `FC_D3` (+kon+hf-withinref) | 3 | **0.864500** | **0.886300** | 0.432200 |

So the honest statement of the problem, before any fit: **the fast class
already clears the composite and CID22 bars and misses ONLY KonJND, by
0.029.** This plan is therefore not "close a broad rank gap"; it is "hold
composite/CID22 while moving KonJND ~0.03, at ≤1.25× the 156 walk".

⚠ **ERA CAVEAT, load-bearing.** The fast-class cells are read on
`ext944-era2r4-2026-09-01`; `W11J`/`W10L9PH` on
`ext944-canonical-2026-08-01` + `sdr-pure-2026-08-28`. Those are different
extractor eras and the shift between eras is model-specific, not a constant
([`eval372_current_root_2026-08-30.md`](../benchmarks/eval372_current_root_2026-08-30.md)).
**The table above is therefore NOT yet an apples-to-apples comparison, and no
conclusion is drawn from it until §6.2 closes the era** by re-scoring the
leader bakes on the era2r4 root with `bake_verdict --features-root`.

---

## 2. FUNDAMENTAL 1 — IDENTITY, and the measurement that localises it

**Measured 2026-09-05, before any fit**, on the 39-row 944-pools identity probe
`/mnt/v/output/zensim/dfree-2026-09-05/probes/identity_probe_944pools_2026-09-05.parquet`
(the D+free lane's instrument, re-read by this lane; no new extraction):

| keep set | n slots | identity-NONZERO inside the set | max \|value\| | max spread over the 39 refs |
|---|--:|--:|--:|--:|
| 156 basic | 156 | 36 | 8.103e-04 | 7.882e-04 |
| 228 (+peaks) | 228 | 60 | 4.754e-03 | 4.587e-03 |
| **265 (+free)** | 265 | **64** | **6.883e-01** | **2.607e-01** |
| 289 (+class-C) | 289 | 64 | 6.883e-01 | 2.607e-01 |
| 944 (all) | 944 | 286 | 1.0 | 8.003e-01 |

**The whole of the fast class's identity contamination above 5e-3 is four
slots — f926, f931, f936, f941 = `LUMA_MEAN_REF`, one per scale.** All 33
other raw-moment slots (f733..f922) are identity-ZERO and so are all 24
class-C slots. `LUMA_MEAN_REF` is the only *reference-absolute* statistic in
the free set: it reads the reference's own mean luma, so it structurally
cannot vanish at ref == dist.

### 2.1 The fix taken: SELECTION, no kernel change

New slice `scripts/sota944/slice_basic156_free_nolumaref.txt` — **261
coordinates = the 265 set minus those four**. The producing walk is unchanged
(`V1PoolsMode::Peaks` + `V1FreeExtras::RawMoments` still emits all 37); the
model simply does not read four of them, and dead-column pruning drops the
lines at pack time, so W4 is *identical to the 265 arm by construction*, not
merely similar.

### 2.2 The fix NOT taken, and why it is registered rather than requested

A difference-form emit (e.g. `LUMA_MEAN_REF - LUMA_MEAN_DST`) is a KERNEL
change and belongs to the kernel lane. It is **not requested** because the
measurement above says the slots are droppable at a cost this lane can price
(arm S261 vs S265 is exactly that price), and because the D+free lane already
measured the whole raw-moment half to be worth ~3 % of the free set's CID22
gain and NEGATIVE on 8 of 12 corpora
([`d_free_id100_2026-09-05.md`](../benchmarks/d_free_id100_2026-09-05.md) §6).
If S261 loses materially to S265 the request becomes justified; that is a
result, not an assumption.

### 2.3 Identity is not fully fixed by dropping four slots — stated up front

At 4.75e-3 the 228/261 identity vector is still image-dependent, so the
identity *dial* is still not a scalar. The second half of the fix is the
**id100 anchor chain**
([`d_id100_2026-09-04.md`](../benchmarks/d_id100_2026-09-04.md)): fit the
output spline on an anchor that CONTAINS identity rows targeted at 100. For an
MLP that is `bake_dial_refit pack --anchor <anchor ∪ identity rows>`, since
`pack` takes exactly one anchor parquet. Arm **ID** (§4) builds it.

---

## 3. FUNDAMENTAL 2 — MATRIX SHAPES, and what a shape can and cannot cost

Two facts fix the design:

1. **`--hidden` was never reachable from the fast-class recipe.** Every
   `A3b`/`A4b`/`FC_*` bake carries the trainer's default `--hidden 128`; the
   wave script never exposed it. This plan adds `WR4_HIDDEN` (a no-op when
   unset, §5).
2. **Depth ≥ 2 is UNBUILDABLE on the plain path for this class** — the
   fastclass wave's E1 arm died on the trainer's own refusal (its §7.6):
   `--keep-features` is implemented on the plain `n→hidden→1` path and on
   `--per-sample-alpha-head` only. So a 2-layer 156+free student **requires**
   the per-sample-α head, which is a HEAD change as well as a DEPTH change.
   A depth-1 α-head control is therefore mandatory, and is an arm.

**The W4 consequence, registered before measuring:** hidden width changes the
FORWARD pass, not the WALK. The input SET changes the walk. So the ≤1.25× bar
is expected to be governed by the set, and width is expected to be nearly free
— that expectation is TESTED in §6.4 (`ZEN_S2_EXTRACT_ONLY` separates the
two) and is not assumed. It is also why the width grid extends UP to 256 and
not only down: if width is free, the small end is a size question and the
large end is the rank question.

---

## 4. THE ARMS (frozen — none added, dropped or renamed after this line)

**Base recipe = `FC_D3`**, i.e. `benchmarks/wave_r4_2026-09-01/train_156_student.sh
distill <seed>` with `WR4_KON_WITHINREF=1 WR4_HF_WITHINREF=1`, on the
`ext944-era2r4-2026-09-01` root, teacher
`foldapp2_views/safesyn_distill_hya_r4.parquet`. Chosen, not tuned: D3 is the
fastclass wave's only KonJND result that clears a significance test (variance
F = 75.9, p < 0.05; KonJND 0.4322 ± 0.0079 where the control swings
0.2998–0.4327), and its composite 0.8645 is second of seven by 0.0012. Every
arm below is that recipe VERBATIM plus the listed env delta.

**The base choice has a MEASURED cost, registered here so no later section can
present it as a discovery.** `product_composite` is a weighted mean of CID22
(1.00), imazen-26 real-codec (0.50), non-photo (0.30), KonJND (0.20), AIC-3
(0.10), AIC-4 (0.05) — **`hfnlproxy` is not in it**, and on that axis the
within-ref arms crater:

| k=3 cell | composite | CID22 | KonJND | **hfnlproxy** |
|---|--:|--:|--:|--:|
| `FC_C0` (uniform pairing) | 0.857170 | 0.886216 | 0.356085 | **0.757237** |
| `FC_D3` (kon+hf within-ref) | 0.864504 | 0.886294 | 0.432199 | **0.427064** |
| leaders `W10L9PH` / `W11J` | 0.8594 / 0.8593 | 0.8848 / 0.8874 | 0.4609 / 0.4621 | **0.7384 / 0.7014** |

So D3 buys +0.076 KonJND and +0.007 composite and pays **−0.330 hfnlproxy**.
`hfnlproxy` is in this repo's circularity-excluded set (its target is an ssim2
self-target, so a score there is agreement with ssim2, not a win over it) and
the non-circular near-lossless axis `hfnl_cid22band` barely moved for the same
arms (fastclass §7.4b: D3 pooled −0.0133, within-image −0.0080). That is why
D3 is still the base. **But hfnlproxy is carried as a REPORTED axis in every
results table of this campaign**, and a candidate whose hfnlproxy sits at
~0.43 is stated as such next to leaders at ~0.70–0.74; it is not hidden behind
a composite that cannot see it.

**The pairing factor therefore costs this lane ZERO fits**: `FC_C0` and
`FC_D3` are both already k=3 on disk at S265/H128, so uniform-vs-within-ref at
the control cell is already measured and is read, not re-run.

**Seeds 4004, 4005, 4006 for every arm (k = 3), matched across arms** so the
seed effect is a paired nuisance and the powerful test is paired-by-seed —
the fastclass wave's §7.4 lesson, adopted, not re-learned.

### Phase A — SET × WIDTH factorial, depth 1, plain path (30 fits)

| set | slice file | n |
|---|---|--:|
| **S156** | `scripts/sota944/slice_basic156.txt` | 156 |
| **S228** | `scripts/sota944/slice_basic156_peaks.txt` | 228 |
| **S261** | `scripts/sota944/slice_basic156_free_nolumaref.txt` (NEW, §2.1) | 261 |
| **S265** | `scripts/sota944/slice_basic156_free.txt` — **the control** | 265 |
| **S289** | `scripts/sota944/slice_basic156_free_classc.txt` | 289 |

widths **H32**, **H128** (`--hidden` default = the control) → 5 × 2 × 3 = **30**.

### Phase B — WIDTH extension, depth 1, plain (6 fits)

**H256** × {the two best Phase-A sets by k-mean composite} × 3 seeds.
*(Decision rule frozen here; the SETS it picks are read from Phase A.)*

### Phase C — HEAD / DEPTH at the CONTROL cell `S265/H128` (9 fits)

**AMENDMENT, registered 2026-09-05 20:30 UTC**, with Phase A at 2 of 30 cells
scored and only the CONTROL read. Phase C was registered "at the Phase-A/B
winner"; it now runs at the **control cell** `S265/H128`. Two reasons, and the
arms are unchanged:

1. **It makes C a clean single-variable experiment.** `P1α − control` at the
   same set and width isolates the HEAD, which is the comparison C exists to
   support; run at a winner it would confound head with set/width.
2. **It removes a dependency on this lane's own reading of Phase A**, so the
   whole queue A → C → ORACLE runs unattended instead of waiting on a
   judgement call in the middle of the night. Nothing about which arms run,
   or their seeds, changes.

If Phase A's winner turns out to differ materially from the control, a
winner-cell repeat of the best C arm is an ADDITION, reported as such.

| arm | env delta | what it isolates |
|---|---|---|
| **P1α** | `WR4_ALPHA_HEAD=1` | the HEAD alone, depth 1 — the control that makes P2 readable |
| **P2α** | `WR4_ALPHA_HEAD=1 WR4_N_HIDDEN_LAYERS=2` | depth 2 (`n→H→H/2→heads`), the arm the fastclass wave could not build |
| **SKIP** | `WR4_SKIP=1` | input→output linear skip; a shape lever with *zero* walk cost |

### Phase C2 — the KonJND mechanisms the alpha head UNLOCKS (conditional, ≤6 fits)

**AMENDMENT, registered 2026-09-05 20:15 UTC.** At registration time the only
arm cell read was `S265/H128/p` seed 4004, and it is the CONTROL — it exists to
reproduce the incumbent (gate G1) and carries no information about any arm. The
motivation below is the trainer's own documentation and the fastclass wave's
§7.6, not a Phase A number.

`--konjnd-aggregation-*` and `--pjnd-passthrough-*` are, per this repo's own
CLAUDE.md, *"only wired on the per-sample-α head"* — the head Phase C is the
first thing in this model class ever to build. The α head's own doc says it was
designed for exactly this axis: *"Lets the model assign α per-pair so photo-like
inputs (CID22-shaped) pull α toward rank-dominant while JND-step-grid inputs
(KonJND-shaped) pull α toward pool-dominant."* And the gap this campaign is
chasing is KonJND alone.

**CONDITION (frozen):** run C2 only if Phase C's `P1α` reaches a k=3 mean KonJND
within 0.02 of the plain control's — i.e. only if the head is not itself a
regression. Otherwise C2 is NOT RUN and is reported as blocked by its
precondition, never as a null.

**ARMS:** `KA` = `P1α` + `--konjnd-aggregation-*` at the trainer's own default;
`PP` = `P1α` + `--pjnd-passthrough-*` at its default. k = 3 each. Neither is
tuned; a swept version is out of scope for this campaign.

### Phase C3 — the A7r lever, which is also alpha-head-only (conditional, ≤3 fits)

**AMENDMENT, registered 2026-09-05 20:40 UTC**, same standing as C2: Phase A
had 4 of 30 cells scored and only the CONTROL had been read.

G6 established that A7r — a *ladder-ordering* property of the weights — is the
binding ship clause. The trainer has a regularizer aimed at exactly that,
`--monotonicity-reg` (*"penalises a predicted pair whose ordering disagrees
with the target ordering via a quadratic hinge"*), and a correct-by-
construction mode, `--monotone-cbc` (*"bounded [0,100] + monotone↓ in
distortion BY CONSTRUCTION (codec goals G1+G3)"*). **Both are wired ONLY on
`--per-sample-α-head`**, so Phase C is the first thing in this model class that
can reach either.

**ARM `MR`:** `P1α` + `--monotonicity-reg` at the trainer's own documented
default margin, k = 3. Same precondition as C2 (run only if `P1α` is not itself
a KonJND regression ≥0.02).

**`--monotone-cbc` is NOT an arm.** It additionally requires
`--tanh-output-head-scale > 0` and a per-feature sign-mask TSV, and this repo's
CLAUDE.md records that without the mask it *"collapses the dial by
mis-constraining the ~72 sign-flip features"*. The mask that exists
(`benchmarks/feature_sign_mask_2026-05-26.tsv`) is a **372**-layout artifact and
this class is 944-layout, so using it would be a silent index mismatch.
Registered as the named follow-up with its prerequisite stated; not run.

**EXECUTION ORDER (not an arm change).** Phases run C → A-ORACLE → C2/C3 → B,
because C is the only phase that can move either of the two axes the campaign
is actually short on. The ARMS are exactly as frozen; only the order they are
fit in changes.

### Phase A-ORACLE — the COMPUTE ceiling of this recipe (3 fits)

**AMENDMENT, registered 2026-09-05 20:10 UTC, while Phase A was on its first
cell and before ANY arm number existed.** Reason: G4's result changed what the
KonJND gap can be attributed to, and this arm is the cheapest thing that turns
the attribution from an argument into a measurement.

G4 established that the leaders train on a **pools-ZEROED** compute set — their
`f156..371` are structural zeros — so whatever KonJND skill they carry above
the fast class comes from **v2 (f372..719) + append (f720..923)**, blocks the
fast walk does not compute at all. The fast class's own cheap slice of that
territory is the 24 class-C slots, and the fastclass wave already measured them
at KonJND +0.0239 (t = 0.40, not significant). So the standing hypothesis is
that the KonJND deficit is a **COMPUTE gap, not a shape gap** — and Phase A as
registered cannot separate the two, because every one of its cells is
compute-restricted.

**The arm:** the base recipe VERBATIM with `WR4_KEEP` unset — i.e.
`--max-features 944` with **no** `--keep-features`, on the same pools root, same
seeds. Same recipe, same data, same era, same build; the ONLY difference is
that the model may read all 944 coordinates.

`ORACLE − S265` is then exactly "what the missing compute is worth under this
recipe", with none of the recipe/root/era confounds that comparing against
`W10L9PH`/`W11J` carries. If ORACLE's KonJND lands near the leaders' 0.478, the
deficit is compute and no shape in Phase A/B/C can close it; if ORACLE lands
near the fast class's 0.432, the deficit is the recipe and the leaders' margin
comes from somewhere this campaign has not looked. Either answer is a result.

It is NOT a ship candidate — it prices at the full 944 walk and fails W4 by
construction. It is an instrument.

### Phase D — the DIAL chain on the selected candidate (0 fits)

`bake_dial_refit pack --neg-tail --anchor <anchor944_pools_dial ∪ n_id identity
rows @ target 100>`, i.e. the id100 chain of §2.3, applied to the §7-selected
cell and its two seed siblings. Re-splining only; no retrain.

**Total 45 fits + 1 gate fit.** Budget ~8 min/fit (the fastclass wave's
measured wall time for this identical recipe/data/build) ⇒ ~6 h, **serial,
local, `run-heavy --mem 16G --jobs 8`, never two at once** (machine-safety
rule; sibling lanes are live on this box).

### What is NOT an arm, stated so it cannot be read as an omission

* **Any KonJND corpus/teacher change.** Nine mechanisms have failed on that
  axis across two lanes (fastclass §7.10) and the measured cause is that the
  quantity is absent from the training signal. This lane changes SHAPE and
  SET; it does not re-try data levers.
* **A kernel change.** Registered in §2.2; the kernel lane owns it.
* **`--ema-decay` / `--dro-eta` / `--listwise-weight` / `--hard-pair-frac`.**
  They are read ONLY on the per-sample-α path, so P1α/P2α make them *reachable*
  for the first time in this class — but turning one on would confound the
  head/depth reading. Registered as the named follow-up.
* **Any CID22 human-MOS training target, any default flip without §8's gate.**

---

## 4b. AMENDMENT — THE SERVABLE LANE (registered 2026-09-05 20:30 UTC, on the kernel lane's finding)

The kernel lane (`8817f379`, `perf(kernel): fast-class extraction — two
bit-exact defects fixed, cost map, and the free-extras serving gap`) measured
something that reorders this campaign's priorities:

> **No `V1FreeExtras` slot is reachable from `Zensim::compute` today** —
> `feature_v2.rs:7532` hard-codes `free_extras: Off` and `fold_engine.rs:158`
> truncates the emitted vector to 372. `wide_bake_v2_read` exists, is tested,
> and is **dead code**.

**Consequence, stated bluntly: of this campaign's five sets, only two can be
served, and only at a layout none of the 944 arms is trained on.**

| set | trains? | **serves through `Zensim::compute`?** |
|---|---|---|
| S156 (`f0..155`) | yes | **only at the v1-372 layout** |
| S228 (`f0..227`, `V1PoolsMode::Peaks` — the mode `D` already resolves to) | yes | **only at the v1-372 layout** |
| S261 / S265 (`+f733..941`) | yes | **NO** — needs a 944-layout scoring path |
| S289 (`+f377..696`) | yes | **NO** — same |

So a **SERVABLE lane** is added, and it runs BEFORE the remaining 944 arms:

**`{S228, S156} × {H128, H32} × k=3`, plus `SFULL372` (unrestricted 372,
k=3) = 15 fits**, trained at the **v1-372 layout** by
`benchmarks/fastclass2_campaign_2026-09-05/train_372_student.sh`.

**It is NOT the same recipe, and the differences are stated, not buried.** The
372 layout has no version of three of the 944 recipe's legs — `tbig_hf` (so
the D2/D3 within-ref lever has nothing to act on), the two distillation
teachers, and `konjnd_bpg_{train,val}` (replaced by the older 20,160-row
`konjnd-dense`, train-only at the same 1.2 weight, with no val twin so no
train==val). Everything else carries over verbatim: epochs, pairs/epoch, group
weights, loss modes, the 34 `f0..155` transforms, `--coarse-decay`.

**ERA, stated as instructed.** Training tables are the v1pre-era
`canonical-2026-05-21` set; their masked/IW blocks are the known-drifted
pre-fix ones and **a ≤228 slice never reads them** (`f0..227` is basic+peaks,
and the basic block never drifted). Eval is on the current
`2026-08-30-full-features-372` default; the flip lane's own 372 era A/B puts
the rank skew at **≤ 7e-4**.

### 4b.1 THE SERVING-PATH MARGIN — pre-registered before any set is compared

The 944 arms become a *"would it be worth building the serving path"*
measurement. Building it is internal-only work (the profile's bake declares its
`feature_set_id`/block profile; `Zensim::compute` derives compute set + layout
from it via `ComputeSet::from_block_profile`, emits the bake's caller width,
and the linear-tail/`raw_distance` semantics are re-decided explicitly), **zero
public-API delta**, gated by bit-exact tests against the stored pools-944
tables on the same pixels. It is justified **only if**, at k = 3 seed-means and
**within the 944 lane** (so era and recipe are held fixed), the best of
{S261, S265, S289} beats **S228** by

* **CID22 ≥ +0.0069** — the replication wave's own measured per-model
  bootstrap CI half-width — **or** **product composite ≥ +0.0070**, the fair
  board's median best-of-k inflation (i.e. larger than the selection noise the
  board already carries),

**AND** is no worse than S228 on **KonJND** or on **A7r**.

If the margin is met, the design is REGISTERED with its gates; whether it is
also implemented tonight depends on the gates finishing first. **The ship
decision is made on a SERVABLE candidate either way.** If a public-API change
turns out to be required, it is registered and this lane stops there.

### 4b.2 W4 is measured LAST, and the baseline is expected to move

The kernel lane measures the fast walk at **6.5 ms @576²/1T** (944-full 16.2
ms) with the **front end** (XYB convert + downscale) at a third of it, and a
separate lane is working that now. So the 156-walk denominator will move.
W4 is therefore run at the END, with the two-control protocol, **on the same
binary as its baseline** — never against a number quoted from an earlier build.

## 5. OWNER EXTENSIONS (all no-ops when unset)

`benchmarks/wave_r4_2026-09-01/train_156_student.sh`, same additive pattern
the fastclass lane used for its four levers:

| env | flag | why |
|---|---|---|
| `WR4_HIDDEN=<n>` | `--hidden <n>` | the shape axis; omitted when unset, so the argv (and the embedded `zentrain.repro`) is byte-identical to a pre-2026-09-05 run |
| `WR4_ALPHA_HEAD=1` | `--per-sample-alpha-head` | the only path on which `--keep-features` composes with depth ≥ 2 |
| `WR4_SKIP=1` | `--skip-connection` | zero-walk-cost shape lever |

`benchmarks/wave_r4_2026-09-01/score_arm.sh`: `BIN` becomes
`"${ZL_BIN:-<the same default>}"` — behaviour byte-identical for every existing
caller, and a lane pinning its own build no longer has to edit a committed
script (repo rule against hardcoded per-lane paths).

---

## 6. GATES — run and read BEFORE any arm

* **G1 — control equivalence.** `train_156_student.sh` with every new lever
  unset, seed 4005, on THIS lane's build, must reproduce `FC_D3_s4005`'s
  MODEL: `bake_verdict --full-json` `srocc_signed` equal on all 12 corpora and
  `product_composite` equal to printed precision. Gate is on the MODEL, not
  the bytes (this lane's binary is not the wave-r4 pin; the replication wave's
  CTL-A established the same distinction). **If G1 fails the wave STOPS** —
  every Δ would be measuring the build.
* **G2 — block profile.** Every arm's packed bake must report the block
  profile its slice implies (`v1_basic` 156/156; `v1_peaks` 72/72 for
  S228+; masked/IW 0), so W4 is inherited from a measured walk rather than
  assumed.
* **G3 — feature-set id.** Every arm's bake carries `zentrain.feature_set_id`,
  and every input set used is registered in
  `benchmarks/feature_sets_registry.json` (fundamental 3).
* **G4 — era, closed. AMENDMENT (2026-09-05, written after G4 ran and before
  any arm was read; the change is a CORRECTION of the method, not of a
  result).** As registered, G4 said "re-score the leaders on the era2r4 root".
  That would have been the registered wrong-regime silent-mis-score: the
  leaders train on `ext944-canonical-2026-08-01`, whose registered set is
  `basic+v2+append+append2@w944/ext944` — **pools ZEROED** — while the fast
  class's root is the pools-LIVE
  `basic+peaks+masked+iw+v2+append+append2@w944/era2r4`. Feeding a
  pools-zeroed-trained model live `f156..371` is not an era A/B.

  So each family is read on **its own COMPUTE at the SAME ERA**: the leaders on
  `ext944-era2r4-2026-09-01/foldapp2_views`
  (`basic+v2+append+append2@w944/era2r4` — same compute as their native set,
  era2r4 era), the fast class on the pools root it trains on. Era is then the
  only thing that changed for the leaders and regime purity holds for both.
  Runner: `benchmarks/fastclass2_campaign_2026-09-05/g4_era_leaders.sh`.

  **RESULT — the era moves the bar UP, so §1's target is raised, not lowered:**

  | leader | k | composite | CID22 | KonJND | hfnlproxy |
  |---|--:|--:|--:|--:|--:|
  | `W10L9PH` @ era2r4 | 6 | **0.8636** | **0.8877** | **0.4783** | 0.6863 |
  | `W11J` @ era2r4 | 7 | **0.8626** | **0.8908** | **0.4782** | 0.6696 |
  | *(their native-era values, for contrast)* | | 0.8594 / 0.8593 | 0.8848 / 0.8874 | 0.4609 / 0.4621 | 0.7384 / 0.7014 |

  **THE COMPETITIVE BAR, restated on the closed era** (the weaker leader per
  axis): composite ≥ **0.8626**, CID22 ≥ **0.8877**, KonJND ≥ **0.4782**. The
  incumbent `FC_D3` reads 0.8645 / 0.8863 / 0.4322 — **already past the
  composite bar, within 0.0014 of the CID22 bar (inside the ~0.0069 per-model
  CI half-width), and −0.046 on KonJND**, which is a LARGER gap than §1's
  cross-era −0.029. hfnlproxy 0.4271 vs 0.67–0.69 is the second gap.
* **G6 — A7r is a REPORTED AXIS on every arm. AMENDMENT registered 2026-09-05
  20:35 UTC**, when Phase A had scored 4 of 30 cells and no arm number beyond
  the CONTROL had been read. Reason: the ship rule's second clause turned out
  to be the binding one, and it is measurable per arm for ~5 s of CPU, so it
  becomes data instead of a single end-of-campaign verdict.

  MEASURED at registration time, on the 944 ladder instrument
  (`dial_grid_944col_ladder.parquet` + `dialcells_ssim2_ladder.tsv`,
  `--floor-rule resolvable`), A7r = the number of the 5 codecs whose
  floor-representability fraction is below the mentor's own:

  | bake | class | A7r (codecs failing, 0 = pass) | contract | C1 mono |
  |---|---|--:|---|--:|
  | **shipped Profile D** (372, ADD156 linear) | 372 additive | **0** | PASS | 0.9931 |
  | `Fctl_id100negrich` (156 slice, 944 linear) | 944 additive | 2 | PASS | 0.9879 |
  | `Fpeaks_id100negrich` (228 slice, 944 linear) | 944 additive | 4 | PASS | 0.9628 |
  | `Ffree_id100negrich` (265 slice, 944 linear) | 944 additive | 4 | PASS | 0.9615 |
  | `W11J_s4013` (944-full MLP leader) | 944 MLP | 4 | PASS | 0.9902 |
  | `FC_D3_s4004` (the fast-class incumbent) | 944 MLP | **5** | FAIL (C5) | 0.9398 |

  **Only the shipped 372 additive passes anywhere, and no 944-width model of
  any class does.** A7r is a *weights* property — a monotone output spline
  cannot reorder a ladder — so it cannot be repaired by the dial chain, which
  is exactly what the d_peaks lane measured at 372 (*"the raw pre-spline model
  is already inverted at the same step — lever is in the fit, not the
  spline"*). The ship rule stands unrelaxed; A7r is now reported per arm so the
  campaign can say whether any SET or SHAPE moves it, rather than only that the
  incumbent fails it.

* **G5a — the W4 instrument has no class-C arm, stated before it matters.**
  `ssim2_speed_bar.rs` carries `add156_156basic` / `peaks156_no_raw` /
  `free156_peaks_raw` and no 289 arm. If S289 wins selection, an additive
  `ZEN_HY_CLASSC` arm is added to the instrument and MEASURED; if it does not,
  S289's W4 is reported as a COMPOSITION of the two published marginals
  (free set +0.8–1.6 %/1T, class-C +1.3–1.5 % native over it) and labelled as
  composed, never as measured. Neither path is at risk of the 1.25× bar:
  the largest published composition is ~+4 %.
* **G5 — W4 protocol.** `zensim-bench/benches/ssim2_speed_bar.rs` with the
  candidate loaded via `ZEN_HY_*`, **1T and 8T**, min over ≥10 process starts,
  `ZEN_S2_WALL_S` sized for the image size, plus `ZEN_S2_EXTRACT_ONLY=1` to
  split walk from forward pass. `add156_156basic` is the ≤1.25× denominator.
  A reading where the stable `fast_ssim2` arm falls below a plausible floor
  for its size is DISCARDED as harness degeneration
  (`profile_d_notax_2026-09-01.md` §4), not selected by `min()`.

---

## 7. SELECTION — the owner ranks, this lane does not

`freeze_check --select --seed-group --min-k 2 --floor-basis all` over every
Phase A/B/C cell. Seed-group means, never a best cell: the fastclass wave's
§7.7 showed the cell-ranking rule picks a lucky draw on a class with this
seed spread. Reported alongside, never as the primary: the paired-by-seed Δ
vs the S265/H128 control with the owner's paired bootstrap
(`paired_perref_boot.py`, reference-clustered), and the k-mean table of §1's
three axes.

---

## 8. THE SHIP RULE — frozen, and deliberately harder than the goal

Install as `ZensimProfile::D` **only if all four hold**:

1. **Full D dial gate**: G-ADDR CONTRACT **6/6** on the candidate's own
   registered probes, and the per-codec floor rule resolvable PASS.
2. **Rank ≥ today's D on CID22 with CI** — today's Profile D reads CID22
   0.8633.
3. **W4 ≤ 1.25×** the 156 walk at **1T AND 8T**.
4. The flip discipline of
   [`d_ship_flip_2026-09-05.md`](../benchmarks/d_ship_flip_2026-09-05.md) §6:
   weights ≤ 30 KB + manifest + `profile.rs` + the two-control W4 protocol +
   tests/clippy/fmt/public-API zero delta + CHANGELOG/docs/annotations/ledger
   + board promote/regen/gates.

**944-competitiveness (§1) is the GOAL, not the gate.** A candidate that beats
today's D and passes the contract ships even if it misses the KonJND bar; a
candidate that hits the KonJND bar and fails the contract does NOT. If neither
holds, this lane PROPOSES with the table and flips nothing.
