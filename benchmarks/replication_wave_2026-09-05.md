# REPLICATION WAVE — registration (2026-09-05)

**Registered BEFORE any fit.** Nothing below is a result; the Results section is empty
until bakes land. Written in the `replicate` sibling workspace (`claude-replicate`).

**Motivation as briefed.** The FAIR board
([`fair_gauntlet_2026-09-04.md`](fair_gauntlet_2026-09-04.md) §3) found that *"8 of the
top 8 combined-fair rows are UNREPLICATED (k=1)"* and that best-of-k inflates the
composite by a median +0.0066 / max +0.0222. The wave was to add seeds to those eight
single draws and decompose the spread into subset coverage vs initialisation.

**What the first hour measured instead: half that premise is an artifact, and the other
half survives.** Both findings are below, with the defect fixed at its owner.

---

## 1. DEFECT FOUND AND FIXED — an output-path flag was splitting one recipe into k=1 pieces

`gauntlet.seed_group_key` (the board's owner) and `freeze_check::seed_group_key` (its
Rust mirror) hash the repro argv with the seed and output-path flags removed:

```
SEED_GROUP_DROP_FLAGS = {--seed, --init-seed, --sample-seed, --out, --output, -o, --bake-out, --manifest}
```

**`--dump-checkpoints-dir` was not in that set**, and every run passes it a per-run
directory *whose name embeds the seed*. The normalized argv of two seeds of one recipe
therefore differed in exactly one token:

```
- /mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4031_ckpts
+ /mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4033_ckpts
```

That is the ONLY difference — verified by diffing the normalized argv of
`lstar2_4031_e060` vs `lstar2_4033_e070`, `LSTAR_s4021_packed` vs `LSTAR_s4022_packed`,
and `w11_s4013_e060` vs `w11_s4014_e050`. So each seed got its own 12-hex key and
reported **k=1 (UNREPLICATED)** when its true k is 3.

A full-board scan for the general case (every flag whose value is an absolute path, not
already dropped, whose basename carries a 4-digit token) found **one** real instance:
`--dump-checkpoints-dir`, on 12 of the 14 top UNREPLICATED cells. (`--bounds-tsv`
matched the regex only via the `2026` in a filename — a false positive, left alone.
`--anchor-parquet`, `--gram`, `--head`, `--keep-features`, `--slice-file`,
`--transforms-tsv`, `--emit-fit-npz` are inputs or non-seed-carrying and stay.)

### Fix

Added to **both** owners, with the reason in a comment at each site:

* `scripts/v_next/gauntlet.py` — `SEED_GROUP_DROP_FLAGS`
* `zensim-validate/src/bin/freeze_check.rs` — `SEED_GROUP_DROP_FLAGS: [&str; 9]`

**Gates:**

* new Rust regression test
  `seed_group_output_path_flag_carrying_the_seed_does_not_split_a_recipe` — asserts the
  two real board argvs share a key, that the normalizer drops the flag *and* its value,
  and carries a **negative control** (a genuine `--epochs` difference must still
  separate two recipes, so the fix cannot merge unrelated runs). 7/7 `seed_group` tests
  pass.
* `scripts/verify_seed_group_parity.py` — **PASS**, 436 fullevals, 0 excluded,
  **82 k≥2 groups** (78 before the fix).

### Blast radius, measured on the live board (436 fullevals)

| | groups | k=1 | k≥2 | max k |
|---|--:|--:|--:|--:|
| before | 109 | 31 | 78 | 12 |
| after | 101 | 19 | 82 | 12 |

Eight of the ten top-scoring combined-fair cells move from k=1 to **k=3**:

| cell | board composite | k before | k after | true seeds |
|---|--:|--:|--:|---|
| `w11_s4013_e060` | 0.8663 | 1 | **3** | 4012, 4013, 4014 |
| `LSTAR_s4021_packed` | 0.8648 | 1 | **3** | 4021, 4022, 4023 |
| `LSTAR3_s4043_packed` | 0.8638 | 1 | **3** | 4041, 4042, 4043 |
| `lstar2_4033_e070` | 0.8636 | 1 | **3** | 4031, 4032, 4033 |
| `lstar4023_e070` | 0.8624 | 1 | **3** | 4021, 4022, 4023 |
| `w11_s4014_e050` | 0.8616 | 1 | **3** | 4012, 4013, 4014 |
| `lstar2_4031_e060` | 0.8607 | 1 | **3** | 4031, 4032, 4033 |
| `LSTAR2_s4032_packed` | 0.8606 | 1 | **3** | 4031, 4032, 4033 |
| `A3b_s4004` | 0.8598 | 1 | 1 | 4004 — genuinely unreplicated |
| `LSTAR3_s4041_packed` | 0.8598 | 1 | **3** | 4041, 4042, 4043 |

### Two of the fair board's "top 8 k=1" rows were never single draws either — they are ENSEMBLES

`HYA_w084` and `BAL_E1_s4010_s4006` carry `model.kind == "ensemble"` (2 members each).
By the owner's own clause 1 an ensemble is an evaluation FUNCTION over members, not a
training replicate, so `seed_group_key` returns `None` and they are **UNGROUPABLE**, not
UNREPLICATED. The fairness TSV distinguishes them correctly (`seed_group` = `-`) but its
`k` column prints `1` for both states, which is what makes them read as single draws in
a sorted view. **Not a defect claim — an observation**, recorded because it is the other
half of why "8 of the top 8" overstated the problem. A distinct `k` rendering for
UNGROUPABLE (e.g. `—`) would remove the ambiguity; left to the board's owner.

---

## 1b. SECOND DEFECT — the seed-split READER shipped before its WRITER (found pre-fit, fixed)

`--init-seed` / `--sample-seed` landed 2026-09-04 (`0698e2f4`) precisely so a study can
hold the drawn subset fixed and vary the init, or the reverse. **The trainer never
recorded which values it used.** Its `zentrain.repro` block emitted only
`"seed": args.seed`, and `--seed` **defaults to 1**, so every split-seed run would have
recorded `seed: 1` with no split keys — while `gauntlet.seed_identity` (and its Rust
mirror) decide k by reading `repro.init_seed` / `repro.sample_seed` with a fallback to
`repro.seed`.

Consequence had this wave run as launched: all 14 arm fits share the identity `"1"`,
`build_seed_groups` collapses them into ONE draw, and the study reports **k=1** — the
exact quantity it exists to measure. Caught by inspecting the emission site before the
third fit; the runner was stopped, partial outputs deleted, and nothing was promoted.

**Fix:** `"init_seed": args.init_seed, "sample_seed": args.sample_seed` added to the
repro block — additive, `null` on a legacy `--seed`-only run so the fallback still
governs every existing bake and nothing already embedded changes meaning.

**Gate:** `zensim-validate/tests/repro_records_split_seeds.rs` (2 tests, mutation-verified
— deleting either emission fails). Runtime behaviour of the reader was already covered by
`freeze_check::seed_identity_reports_the_pair`.

**Verification in this wave:** CTL-B is a split-seed run; its embedded repro must carry
`init_seed: 4021, sample_seed: 4021`. That is an empirical check on the writer, stronger
than the source gate, and it is a launch precondition for the arm fits.

---

## 2. What survives, and what does not

**DOES NOT SURVIVE — "the leaderboard is largely single draws."** After the fix there
are exactly **2** groupable UNREPLICATED combined-fair recipes, and one of them
(`sota944_FS_PILOT16_s2501`) has composite 0.0000. The genuine count is **one**:
`A3b_s4004`. Every other cell at the top of the fair board is already k=2..9.

**SURVIVES — best-of-k inflation.** Recomputed over the 18 combined-fair k≥2 groups the
fix produces: **median +0.0061, max +0.0223, min +0.0013**; k-spread median 0.0141, max
0.0445. The fair board's +0.0066 / +0.0222 over 14 verified-fair recipes is reproduced,
on a larger and correctly-grouped population. **Best-of-k reporting is still the real
problem; "unreplicated leaders" was mostly a grouping bug.**

### Corrected top of the fair board (k-mean composite, k≥2 groups)

| # | representative (best seed) | k | k-mean | best-of-k | spread |
|--:|---|--:|--:|--:|--:|
| 1 | `LSTAR_s4021_packed` | 3 | 0.861467 | 0.864800 | 0.005400 |
| 2 | `LSTAR3_s4043_packed` | 3 | 0.860367 | 0.863800 | 0.006300 |
| 3 | `A5_r4_s4005` | 2 | 0.859450 | 0.865600 | 0.012300 |
| 4 | `W10L9PH_s4004_packed` | 6 | 0.859350 | 0.866400 | 0.011800 |
| 5 | `W10L9_s4003` | 2 | 0.858850 | 0.860200 | 0.002700 |
| 6 | `A1foldapp2_r4_s4004` | 2 | 0.858500 | 0.860100 | 0.003200 |
| 7 | `W11J_s4013_packed` | 3 | 0.857667 | 0.861900 | 0.007200 |
| 8 | `W10L9PBR_s4005_packed` | 3 | 0.856833 | 0.858100 | 0.002600 |
| 9 | `W10L9P_s4010_packed` | 9 | 0.856256 | 0.862000 | 0.017500 |
| 10 | `LSTAR2_s4032_packed` | 3 | 0.853567 | 0.860600 | 0.016200 |

`LSTAR` and `LSTAR3` — invisible as recipes before the fix — are the corrected top two.
The k=6 and k=9 groups (`W10L9PH`, `W10L9P`) remain the best-estimated, and the top four
k-means span 0.0021 on a median k-spread of 0.0141: **still not distinguishable.**

---

## 3. The wave, revised (and why)

Re-fitting seeds for recipes that already carry 3–9 draws would buy nothing. The budget
goes where the measurement is actually missing.

**Coverage question DROPPED before any fit.** The subset-quality study (`92caf565`,
[`subset_quality_study_2026-09-04.md`](subset_quality_study_2026-09-04.md)) already
answered it: whole-run coverage is SATURATED (every row drawn ~17×; between-seed
relative spread 1e-4..1e-6 while scores move 28–940× more), no coverage descriptor beats
a luck control, and lucky seeds are **not** better covered. **The sample seed changes the
ORDER rows are visited, not which rows.** So `zentrain.sample_coverage` is *reported*
(it is embedded automatically) and is **never** offered as an explanation. The axis that
remains open for intentional design is **order/curriculum**, not inclusion.

`--stratified-bands` was a silent no-op on every board bake until `37622802`. It is
reachable now and unmeasured — **it is NOT enabled anywhere in this wave**; a replication
must replay its recipes exactly.

### Arms

Every recipe here was trained with legacy `--seed S`, which the trainer maps to
`init_seed = sample_seed = S` (`zensim_mlp_train.rs:3933,3949`). Existing draws are
therefore all on the **diagonal**.

| arm | init seed | sample seed | what varies |
|---|---|---|---|
| **D** (existing, on the board) | S | S | both, confounded |
| **S** | fixed at S₀ | 5001, 5002 | ORDER / curriculum only |
| **I** | 5011, 5012 | fixed at S₀ | INITIALISATION only |

The diagonal draw at S₀ is a legitimate member of both arms (init = sample = S₀), so
each arm reaches **k=3** with 2 new fits.

**Anchor S₀ is the recipe's LOWEST seed, not its best** — anchoring at the
best-composite seed would re-introduce exactly the best-of-k selection this wave exists
to measure. Pre-registered anchors: `LSTAR` → 4021, `LSTAR3` → 4041, `A5_r4` → 4004.

### Fits — and the two recipes that had to be substituted, with the measurement that decided it

**Every arm member must share ONE data era, ONE trainer build, and ONE pack recipe**, or
an S-vs-I difference cannot be told from a pipeline difference. Two checks were run
before committing the budget.

**Pack-recipe byte-verification.** The board's cells are `_packed` bakes, so a new cell
is only comparable if it is packed identically. The family script's invocation
(`bake_dial_refit pack --neg-tail --anchor <ROOT>/anchor944_dial.parquet --target-col
target_score --verify <ROOT>/ext_cid22val.parquet --verify-col human_score
--verify-scale 100`, `scripts/w12u_lodestar_wave.sh`) was replayed against each
recipe's stored raw bake and the result compared to the stored packed bake:

| recipe | repro sha256[:16] | stored sha256[:16] | verdict |
|---|---|---|---|
| `LSTAR_s4021` | `2fc927829b932dc0` | `2fc927829b932dc0` | **BYTE-IDENTICAL** |
| `LSTAR3_s4041` | `fc74a0ba8f81641c` | `fc74a0ba8f81641c` | **BYTE-IDENTICAL** |
| `W11J_s4013` | `0b130233e811012b` | `0b130233e811012b` | **BYTE-IDENTICAL** |
| `A5_r4_s4004` | `0c99cf0ea949913d` | `8c9cb5fb86cc0c0d` | DIFFERS |
| `A3b_156_s4004` | `d194675e41b4d6ac` | `5f1d7a2512c02b32` | DIFFERS |

**Data-era check.** The two that differ are from the wave-r4 lane: their repro argv names
a different trainer binary (`/mnt/v/zen/cargo-targets/waver4/release/zensim_mlp_train` —
another workspace's target dir, pinned to no commit this lane can verify) and their
`--group` roots are **`ext944-era2r4-2026-09-01`**, the era-2 × radius-4 extractor tables,
not the canonical `ext944-canonical-2026-08-01` + `sdr-pure-2026-08-28` roots every other
candidate uses. Their pack recipe is not committed anywhere and could not be recovered.

**Decision.** `A5_r4` (corrected rank 3) is replaced by **`W11J`** (corrected rank 7,
k=3) — same canonical roots, same trainer path, committed wave script, pack byte-verified.
`A3b` is KEPT, because it is the one genuinely-k=1 recipe on the fair board, but its
stored diagonal cell is **NOT pooled**: its arm gets a diagonal re-run on this build so
all three members share a pipeline. Its results are reported separately from the
canonical-root recipes and are never mixed with them.

| # | recipe | root | fits | arms |
|---|---|---|--:|---|
| 0–1 | `LSTAR` — **CTL-A** legacy `--seed 4021`, **CTL-B** `--init-seed 4021 --sample-seed 4021` | canonical | 2 | equivalence controls |
| 2–5 | `LSTAR` `75b7da973ad4` (corrected rank 1), S₀=4021 | canonical | 4 | S×2, I×2 |
| 6–9 | `LSTAR3` `797d8ce9e932` (rank 2), S₀=4041 | canonical | 4 | S×2, I×2 |
| 10–13 | `W11J` `ecb3b9b18a42` (rank 7), S₀=4013 | canonical | 4 | S×2, I×2 |
| 14–16 | `A3b` — the one true k=1, S₀=4004 | **era-2 r4** | 3 | D-rerun + S×2 |

**17 fits.** New seeds are pre-registered: sample arm 5001/5002, init arm 5011/5012.

**Budget.** ~120 epochs/fit; measured ~7.5 s/epoch on an idle box (~15 min/fit) and up to
~21 s/epoch under self-inflicted CPU contention. Serial, local, under `run-heavy`
(`--jobs 12`, peak RSS 6.8 GiB/fit) — **never two fits at once** (machine-safety rule).
Expected 4–11 h; it runs overnight. The home fleet was considered and rejected: the
944 feature tables live on this box's `/mnt/v`, which tower cannot see, so staging would
cost more than it saves. Progress streams to
`/mnt/v/output/zensim/replication-2026-09-05/logs/PROGRESS.txt`, one line per fit start
and end with `rc` and elapsed seconds; every fit writes a `.done` marker on every exit
path, so the runner is resumable and a late wake-up costs nothing.

### Replay fidelity

Each recipe is replayed from its embedded `zentrain.repro.argv` **verbatim**, with only:
`argv[0]` → this workspace's trainer, `--out` / `--dump-checkpoints-dir` → the wave's
output root, and the seed flags per arm. All inputs verified present on disk. Trainer
pinned to one build for the whole wave (`zensim--replicate` @ the wave commit); the
build id is recorded per fit and every bake embeds `zentrain.repro` +
`zentrain.sample_coverage` automatically.

---

## 4. Pre-registered analysis (owner tools only)

1. **Harvest inline** — `bake_verdict --full-json` + fulleval + M3a per bake as it lands
   (`scripts/harvest_bakes.sh`).
2. **Group ranking** — `freeze_check --select --seed-group` over the wave plus the
   existing diagonal cells. The owner ranks; nothing is re-derived here.
3. **Group table** — mean, spread (min–max), k, per-seed values, and per-seed
   `sample_coverage` from the sidecar (reported, never explanatory — §3).
4. **Decomposition** — for each of the top-3 recipes, per-axis (composite, CID22,
   KonJND, AIC-3) spread of arm S vs arm I, with **paired-bootstrap CIs from the owner**
   (`panel --batch`, index-set resamples — never scipy in a loop).
5. **Pre-registered read.** The pilot in the subset-quality study found order-spread >
   init-spread on CID22 (0.0078 vs 0.0017) and the reverse on KonJND/AIC-3 at n=3 with
   no CIs. This wave's k=3-per-arm on real leaders is the confirmatory test. A
   difference is called only when the CI excludes zero; n=3 per arm per recipe is small
   and the honest outcome may be "not resolved" — that is a reportable result, not a
   failure.
6. **Promotion** — `scripts/promote_fulleval.py`, discussion set
   `2026-09-05-replication`. Both boards regenerated + `gauntlet_gates.sh` (blocked on
   the `d3a948ca` recovery — §6).

---

## 4b. CONTROLS — both gates PASS (measured 2026-09-05, fits 1–2 of 17)

The two controls are the wave's preconditions, and both are closed before any arm fit
was interpreted.

**CTL-A reproduces the stored model EXACTLY.** `LSTAR_s4021` replayed verbatim from its
embedded argv on the pinned binary, scored with `bake_verdict --regime 944 --full-json`
against the stored raw bake:

| | |
|---|---|
| corpora compared | 12 (aic3, aic4, cid22, csiq, hfnlproxy, imazen26, kadid, konjnd, live, nonphoto, sdr25, tid) |
| `srocc_signed` mismatches | **0 of 12**, equal to all 12 printed decimals |
| `product_composite` | `0.8648336661364353` on both, to all 16 digits |

So the trainer's behaviour is unchanged by everything that landed between 2026-08-28 and
`34b4899f` (this lane's repro-block addition included), and new arm draws are the same
population as the existing diagonal cells. Bake bytes are NOT identical — the coverage
sidecar and a different `--out` in the embedded argv both changed since — which is why
the gate is on the MODEL, not the file.

**CTL-B confirms the seed-flag equivalence.** `--init-seed 4021 --sample-seed 4021` vs
legacy `--seed 4021`: **0 of 12 mismatches, identical composite to 16 digits.** So
`--seed X` ≡ `--init-seed X --sample-seed X` on a real 944 recipe, and the legacy
diagonal draw is a legitimate member of both arm S and arm I.

**The writer fix is confirmed on a real bake, not just a probe.** CTL-B's embedded repro
reads `"init_seed":4021, "sample_seed":4021, "seed":1` — and that `seed:1` is exactly the
collapse §1b describes: without the two new keys this bake, and every arm member, would
have reported seed identity `"1"`.

Wall clock: 505 s and 477 s. The 17-fit wave is ~2.4 h, serial, on this box.

---

## 4c. RESULTS (all 17 fits landed, 0 nonzero rc; 2026-09-05)

Wall clock 336–576 s/fit, 2 h 11 m total, serial, local. Trainer binary sha256
`9eff0caf…` re-checked unchanged at harvest.

### 4c.1 What moved at the top once the leaders were replicated: they went DOWN

Fair-board k-mean composite, k≥2 groups, before vs after the wave's arms joined:

| recipe | k before | k-mean before | rank before | k after | k-mean after | rank after |
|---|--:|--:|--:|--:|--:|--:|
| `LSTAR` | 3 | 0.861467 | **1** | 7 | **0.856414** | **8** |
| `LSTAR3` | 3 | 0.860367 | **2** | 7 | **0.856843** | **6** |
| `W11J` | 3 | 0.857667 | 7 | 7 | **0.859286** | **3** |

**Two of the three corrected leaders fall out of the top five when they are
actually replicated.** The new top three — `A5_r4` (k=2, 0.859450),
`W10L9PH` (k=6, 0.859350), `W11J` (k=7, 0.859286) — span **0.0002** on k-spreads
of 0.010–0.012, i.e. still not distinguishable. Best-of-k inflation over the 18
combined-fair k≥2 groups rose from median +0.0061 to **median +0.0070, max
+0.0223**, and median k-spread from 0.0141 to **0.0160**: adding honest draws
makes the best-of-k premium bigger, not smaller, which is what a real
selection-on-noise effect does.

### 4c.2 ORDER vs INIT — the decomposition

Per-axis spread over k=3 models per arm, against the owner's own bootstrap CI
half-width (median over the arm's models). Nothing is recomputed: every SROCC
and CI is read from a `bake_verdict --full-json` fulleval.

| recipe | axis | spread(ORDER) | spread(INIT) | CI half | call |
|---|---|--:|--:|--:|---|
| LSTAR | cid22 | 0.019251 | 0.011077 | 0.0066 | **ORDER > INIT** |
| LSTAR3 | cid22 | 0.013457 | 0.001897 | 0.0067 | **ORDER > INIT** |
| W11J | cid22 | 0.014305 | 0.002049 | 0.0069 | **ORDER > INIT** |
| LSTAR | konjnd | 0.124678 | 0.125075 | 0.0733 | init > order (0.0004 — noise) |
| LSTAR3 | konjnd | 0.059300 | 0.038778 | 0.0745 | UNRESOLVED |
| W11J | konjnd | 0.055952 | 0.037722 | 0.0740 | UNRESOLVED |
| LSTAR | aic3 | 0.017685 | 0.012590 | 0.0349 | UNRESOLVED |
| LSTAR3 | aic3 | 0.016536 | 0.013042 | 0.0345 | UNRESOLVED |
| W11J | aic3 | 0.013511 | 0.016117 | 0.0342 | UNRESOLVED |
| LSTAR | composite | 0.017478 | 0.012575 | — | ORDER > INIT |
| LSTAR3 | composite | 0.010431 | 0.003253 | — | ORDER > INIT |
| W11J | composite | 0.003213 | 0.006620 | — | init > order |

**CID22 is the verdict: ORDER dominates INIT, 3 of 3 recipes**, and it is the
only axis where the effect clears the noise floor — the order spread is 2–7×
the per-model CI half-width on all three, while the init spread is *inside* it
on LSTAR3 and W11J. That **confirms the subset study's n=3 pilot read
(0.0078 vs 0.0017) on real leaders with k=3 per arm.**

**The pilot's KonJND / AIC-3 "flip" does NOT survive.** On both axes the
per-model CI half-width (0.073–0.075 KonJND, 0.034 AIC-3) is larger than either
arm's spread on essentially every cell: 5 of 6 are UNRESOLVED and the sixth
(LSTAR KonJND) separates the arms by **0.0004** on a 0.073 CI. The honest
statement is *not resolved at k=3*, not "init dominates there".

Secondary axes split (csiq/live/nonphoto/imazen26 give 4 ORDER, 3 init, 5
UNRESOLVED across the three recipes) — consistent with one real effect on the
human-MOS axis and noise elsewhere.

### 4c.3 A3b — the one genuinely-k=1 recipe (era-2, reported separately)

k=3 arm S (re-run diagonal + 2 order seeds), scored at its **native**
`ext944-era2r4-2026-09-01` root: CID22 mean **+0.885487**, spread 0.009957 (CI
half 0.0069); KonJND −0.337323, spread 0.088567 (CI half 0.0790); AIC-3
+0.804181, spread 0.013933. Same shape as the canonical three — an order-spread
on CID22 that clears its CI, and a KonJND spread that does not.

It has **no arm I** (registered that way) and WAS **not promoted to the board**:
`bake_verdict` correctly REFUSES to score it on the canonical folded root (it
structurally uses 72 caller lines in `f156-371` that the folded root feeds as
structural zeros — the registered wrong-regime class), and `run_full_eval.sh`
has no `--features-root` passthrough, so a board cell for it cannot be produced
without changing that owner. Named gap, not a silent omission. This also
explains a first attempt that packed it against the CANONICAL anchor and read
CID22 0.6424; with its native anchor the same bakes read 0.8808–0.8908.


> **CLOSED 2026-09-05 (owner lane `claude-selectfix`).** `run_full_eval.sh` no
> longer hard-codes a root: it takes the features root **FROM THE BAKE**
> (`zensim_validate::feature_set::resolve_features_root`, reached through
> `bake_verdict --print-features-root`, keyed on the feature-set registry).
> All four A3b cells — the three replication arms plus the original
> `A3b_s4004` — are now full board cells read on `ext944-era2r4-2026-09-01`,
> **with M3a**, in discussion set `2026-09-05-a3b-native`:
>
> | cell | CID22 | KonJND | composite | M3a |
> |---|---:|---:|---:|---:|
> | `A3b_s4004` (original) | 0.8908 | 0.3540 | 0.8598 | 0.7489 |
> | `A3b__D__i4004_p4004_packed` (diagonal re-run) | 0.8908 | 0.3540 | 0.8598 | 0.8618 |
> | `A3b__S__i4004_p5001_packed` | 0.8808 | 0.3733 | 0.8553 | 0.8652 |
> | `A3b__S__i4004_p5002_packed` | 0.8849 | 0.2847 | 0.8496 | 0.8776 |
>
> The original already had a verdict on the native root but **no M3a**, so it
> was UNMEASURED and therefore not selectable — that is the half the missing
> fulleval cost. Note the diagonal pair: `A3b_s4004` and `A3b__D__…` are the
> same recipe on the same seeds (i4004/p4004) **packed twice** — identical
> CID22 to 16 digits and composite differing at the 9th decimal, but M3a
> **0.7489 vs 0.8618**. Attribution density is sensitive to the pack (the two
> bakes have different sha256), which is worth knowing before reading any
> single-cell M3a as a property of a RECIPE.
>
> A3b is still k=1 on distinct seeds and is now subject to the replication
> floor registered the same day (campaign appendix **E.4 AMENDMENT**): its
> cells are listed and ranked, never selected.

### 4c.4 `freeze_check --select --seed-group`

Over the 14 promoted wave cells plus the 99 combined-fair board cells (113
inputs). The arms DO join their diagonals now: `LSTAR` k=7/13 cells, `LSTAR3`
k=7/7, `W11J` k=7/11.

**The owner's pick is `62df0d51a60e` = `W10L9_s4003_packed`, k=1, UNREPLICATED**,
8.00/8 floors, selection_composite 0.9841. Worth stating plainly: the registered
selection rule's PRIMARY key is the profile floor count, so **a single draw can
still win it** — the `--seed-group` flag makes the k visible and ranks groups by
their mean, but it does not stop an unreplicated cell from topping the table
when its floor count is highest. The three replicated wave groups rank 4, 8 and
11 on that rule with 6.14–7.14 mean floors.

### 4c.5 `sample_coverage` — reported, not explanatory

Every wave bake embeds `zentrain.sample_coverage` (band edges, per-group band
pair counts, duplicate-pair rate, near-threshold share, and a replay `digest`).
It is recorded and readable per seed. It is **not** offered as an explanation of
anything above: the subset-quality study (`92caf565`) measured whole-run
coverage as saturated, with no coverage descriptor beating a luck control and
lucky seeds no better covered. The axis this wave finds live is **ORDER**, which
is what the sample seed actually changes.

## 5. What this wave does NOT do

* It does not enable `--stratified-bands`, or change any recipe.
* It does not re-fit `W10L9PH` (k=6) or `W10L9P` (k=9) — already the best-estimated
  groups on the board.
* It does not replicate `HYA_w084` or `BAL_E1_s4010_s4006` (ensembles — not training
  replicates), nor `sota944_FS_PILOT16_s2501` (composite 0.0000).
* It does not re-derive any statistic the board or `freeze_check` already owns.

## 6. Open, blocking, and honest

**RESOLVED — the board was regenerated.** `d3a948ca` was re-landed as `be604c12`
before this wave's endgame; both boards were rebuilt on it (NOT-SHIPPABLE badges
verified present) and again after promotion. Original note kept below for the
record.

**Board regeneration was HELD.** `d3a948ca` (G-ADDR board coverage: NOT-SHIPPABLE badges,
`--graft-gaddr`) was dropped from `origin/main` by a sideways push and is **not an
ancestor of this workspace**. Both `summer_gauntlet_fair.html` and
`summer_gauntlet.html` were regenerated at 19:14 local **before** that was known, from a
tree without `d3a948ca` — so the live boards currently carry **0 NOT-SHIPPABLE badges**
where the previous files had them. The HTML is a derived artifact and fully
regenerable; no source or verdict was touched, and `fairness_tiers_2026-09-04.tsv` was
left intact (the new run wrote a separate `…2026-09-05.tsv`). **Both boards must be
regenerated once `~/tmp/recover_d3a948ca_READY.md` appears**, after
`jj git fetch` + rebase. Recorded here rather than quietly re-run.

**No further push until the wave completes.** The trainer resolves
`trainer_head_at_train` at RUNTIME (`jj log -r @-` in the crate dir), so advancing `main`
mid-wave would stamp later fits with a different head than earlier ones. **The authoritative pin is the BINARY, not the commit**: every one of the 17 fits is run
by one `zensim_mlp_train` whose sha256 is
`9eff0caf83853f2f3298cbc4559c599bc9a5186ff87cc07e5eef06d49e311913`
(built from `34b4899f`), recorded at
`/mnt/v/output/zensim/replication-2026-09-05/TRAINER_PIN.sha256` with a copy of the
binary beside it, and re-checked at harvest. A `trainer_head_at_train` stamp on a fit
records whatever `@-` was when that fit started, which is NOT the same thing once `main`
advances — read the binary hash, not the stamp.

---

## 7. Owner defects found by running this wave — four, all fixed and gated

Every one was invisible until a recipe was actually replayed and re-grouped.

| # | defect | owner(s) | how found | measured effect |
|---|---|---|---|---|
| 1 | `--dump-checkpoints-dir` not in `SEED_GROUP_DROP_FLAGS` (its value embeds the seed) | `gauntlet.py` + `freeze_check.rs` | diffing normalized argv of two "distinct" leaders | k≥2 groups 78→82; 8 of the 10 top combined-fair cells k=1 → true k=3 |
| 2 | `zentrain.repro` never recorded `init_seed`/`sample_seed` | `zensim_mlp_train.rs` | reading the emission site before the third fit | a k-arm split-seed study would report **k=1** (all arms identity `"1"`) |
| 3 | `argv[0]` (the trainer's build path) part of the recipe key | `gauntlet.py` + `freeze_check.rs` | this wave's arms would not group with their own diagonal | 32 distinct `argv[0]` on the board; groups 101→98, both merges genuine |
| 4 | a cell with NO recorded seed counted as a distinct draw; and `init == sample` counted as a second draw | `freeze_check.rs` (4a) + both (4b) | the cross-owner parity gate, immediately after fix 3 | two seedless cells reported k=2 off zero seeds; CTL-A/CTL-B inflated LSTAR's k 7→8 |

All four are the same shape: **a value that is not part of the recipe was being
treated as part of the recipe's identity, or a non-draw was being counted as a
draw.** Both directions corrupt `k`, and `k` is the number every seed-group
statistic divides by.

**Carry forward.** `run_full_eval.sh` hard-codes its `--features-root` per
regime, so a bake trained on a non-default root (era-2 × radius-4 here) cannot
be given a board cell at all. That is the next one in this family: it is not a
wrong number, it is a missing row, which is harder to notice.
