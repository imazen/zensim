# Era-2 blast radius + retrain-wave registration

**REGISTERED, NOT LAUNCHED.** This is the third and last of the flip
prerequisites (rank preservation and the gate re-pin enumeration are done —
`era2_rank_preservation_2026-08-31.md` and `era2_perf_break_2026-08-31.md`
§27/§28). Nothing here is a decision; it is the cost sheet the decision needs.

---

## 0. The framing that matters: the radius decides the size of this

The rank lane established that the components are **separable and radius
dominates** — every model's worst-corpus delta is identical to five decimals
with the tile on or off, and the tile moves a composite by ~1e-5 against
radius's ~4e-3. That splits the blast radius cleanly in two, and **the cheap
half is the one that is shippable today**:

| | tiling (F3) | accumulation (F6) | radius 4 (F1) |
|---|---|---|---|
| test re-pins | **5 goldens** | **0** | 5 goldens (same set) |
| stored-feature re-extraction | **not required** | **not required** | **required** |
| retrain | **not required** (5/6 models pass) | not yet rank-checked | **required** (4/6 fail, all upper bounds) |
| fleet cost | none | none | **the whole 924 re-extraction** |

**Tiling + accumulation have essentially no data-side blast radius.** Radius 4
has all of it.

---

## 1. Tier 1 — mandatory, and small

Re-pinned by re-capturing. Applies to tiling and to radius 4 (same set), and to
**nothing** for the accumulation (§28.3).

| artifact | how it is re-pinned |
|---|---|
| `GOLDEN_SYNTHETIC`, `GOLDEN_REAL`, `GOLDEN_NONTIGHT` in `zensim/tests/v1_golden_bytes.rs` | `cargo run --example capture_v1_golden` |
| `fold_backed_fixtures_match_golden` | same capture (it reads the same three) |
| `hardcoded_reference_scores` in `zensim/tests/cross_platform.rs` | 8 scores, re-captured from the same fixtures |

That is the entire test-side cost. **Every internal-consistency gate survives**
— cross-engine, cross-entry-path, cross-rayon-pool-size, cross-tier, the
fold-vs-v1 family — which is what makes the flip a single step rather than an
incremental migration.

---

## 2. Tier 2 — conditional, and it is the expensive one

Required **only for components that fail the utility bar**, i.e. radius 4 for
4 of 6 models. Not required for tiling (5/6 pass at the production tile width)
or the accumulation (rank-checkable as of §28, not yet checked).

### 2.1 What must be re-extracted

Every stored `feat_*` table is an **era-1 extraction**. Under a radius change
they are stale as *training* inputs (as *scoring* inputs the rank lane already
measured what the shift costs). Canonical set, from `~/work/zen/DATA_PROVENANCE.md`:

| dataset | rows | where |
|---|---:|---|
| 11 local legs (cid22val / aic3 / aic4 / csiq / live / kadid / tid / safesyn / cid22t201 / konjnd / sdr25) | 149,195 | `/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/` |
| bigcodec `tbig_924_full` | 5,742,660 | `/mnt/v/output/zensim/tbig-924-2026-07-27/` |
| bigcodec 21 split views | 5,742,660 | same root, `bigcodec/<dataset>/<split>_924.parquet` |
| `kadis700k_924` | 699,999 | `/mnt/v/zen/zensim-training/kadis-924-2026-07-27/` |
| `kadis_negrich_924` | 167,034 | same dir |
| eval instruments `corruption_grid_924col` + `dial_grid_924col` | 2,016 + 4,817 | `/mnt/v/output/zensim/v2-eval-924-2026-07-27/` |

**The bigcodec table dominates**: 5.74 M rows is ~97 % of the re-extraction,
and it is the one that needs the fleet. The 11 local legs (149 k) and the two
eval grids (6.8 k) are a local-box job.

**Append-only discipline holds**: an era-2 re-extraction is a NEW dated root
(`ext944-era2-<date>/`), never an in-place overwrite of the 2026-07-27 tables.
The era-1 roots stay for every pre-flip verdict to remain reproducible — the
same rule the `_INVERTED_` KADID files follow.

### 2.2 What must be re-scored, not re-extracted

**378 `*.fulleval.json` board cells** at
`/mnt/v/output/zensim/reports/fulleval/`. These are verdicts over stored
feature tables, so they are a **rescore, not a re-extraction** — the cheap
path `run_full_eval.sh`'s `ZENSIM_M3_REUSE=1` already exists for exactly this.
They must be re-run against the era-2 tables and **published under a new era
stamp**, not overwritten: a board that mixes era-1 and era-2 cells is the
`score_zensim` era-mixing hazard the memory registry already records.

### 2.3 The retrain wave

Registered, unlaunched. Scope is set by the rank lane's 2 PASS / 4 FAIL:

| # | arm | why |
|---|---|---|
| W-R4-1 | **C944 flagship** — retrain at radius 4 on era-2 tables | it already PASSES as an era-1 bake; the retrain is to confirm the pass is not luck and to bank the RSS win |
| W-R4-2 | **B** | fails; era-1-trained, so its FAIL is an upper bound |
| W-R4-3 | **W-LIN ×2** | same |
| W-R4-4 | **BHdr** | fails both the radius bar AND the tiling composite clause by 3.2e-6 — the one model that constrains both components |
| W-R4-5 | **ADD156** | newly passes; cheap, and it is the sparse-additive control |

**Gate:** the same §21.1 bar, re-run on the retrained bakes. A retrain that
still fails moves radius 4 from "upper bound" to "measured cost", which is the
information the decision actually needs.

---

## 3. What this does NOT cover

* **Downstream consumers outside this repo.** `zenmetrics`' fleet metric names
  and any `score_zensim` column already written are era-stamped by the memory
  registry's own rule; changing them is a zenmetrics decision and is
  deliberately not proposed here.
* **Published crates.** `zensim` 0.2.7 on crates.io is unaffected — the era is
  an unreleased-main change, and a release carrying it is a separate decision
  with its own semver question.
* **The accumulation's rank check.** Wired as of §28 and cheap (the rank lane
  says a 70 s re-run); it is not part of this wave because it does not need
  one.
