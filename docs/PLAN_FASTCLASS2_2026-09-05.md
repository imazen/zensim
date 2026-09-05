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

### Phase C — HEAD / DEPTH at the Phase-A/B winner (set\*, H\*) (9 fits)

| arm | env delta | what it isolates |
|---|---|---|
| **P1α** | `WR4_ALPHA_HEAD=1` | the HEAD alone, depth 1 — the control that makes P2 readable |
| **P2α** | `WR4_ALPHA_HEAD=1 WR4_N_HIDDEN_LAYERS=2` | depth 2 (`n→H→H/2→heads`), the arm the fastclass wave could not build |
| **SKIP** | `WR4_SKIP=1` | input→output linear skip; a shape lever with *zero* walk cost |

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
* **G4 — era, closed.** The three leader bakes re-scored on the era2r4 root
  with `bake_verdict --features-root`, so §1's comparison becomes
  apples-to-apples. If a leader REFUSES to score there, it is reported as NOT
  COMPARABLE, never silently dropped.
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
