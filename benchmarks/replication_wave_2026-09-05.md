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

## 5. What this wave does NOT do

* It does not enable `--stratified-bands`, or change any recipe.
* It does not re-fit `W10L9PH` (k=6) or `W10L9P` (k=9) — already the best-estimated
  groups on the board.
* It does not replicate `HYA_w084` or `BAL_E1_s4010_s4006` (ensembles — not training
  replicates), nor `sota944_FS_PILOT16_s2501` (composite 0.0000).
* It does not re-derive any statistic the board or `freeze_check` already owns.

## 6. Open, blocking, and honest

**Board regeneration is HELD.** `d3a948ca` (G-ADDR board coverage: NOT-SHIPPABLE badges,
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
