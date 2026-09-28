# joint-core — build a shared TRAIN pair corpus (v1 and v2) usable by every DVIFM/geometry/zgeom lane

## What was asked

Brief: `../briefs/joint_core_build_prompt.md` (see also
`docs/PLAN_JOINT_CORE_SET_2026-09-19.md`, already permanent in the repo).
Build one canonical, provenance-tracked TRAIN pair corpus ("joint-core") that
every later DVIFM-family lane (transplant, zgeom, geometry, verdict) could
extract features over without re-deriving admission/kernel/split logic each
time — assembling fresh renditions across bands/classes/legs, reproducing
the frozen leader recipe on the smaller core as a sanity gate, and testing
whether the core is big enough to detect a 30-column permutation cost.

## Verdict (from `../reports/CORE_DONE.md` — v1 — and the transplant lane's step-0, which grew it to v2)

**v1 (joint-core-v1, 52,963 pairs): DONE with a measured too-small finding.**
Gates 1/2/4 pass (leader reproduction gap −0.0068 vs frozen R915, explained
by ~3.3x smaller train pool; convergence flat-to-rising; coverage matches
target bands/classes). **Gate 3 (30-column permutation-cost detectability)
FAILS by the plan's own rule**: cost +0.0021 is indistinguishable from the
~0.003 seed-noise floor — *the core is too small for feature screening*.
Per the work order this is reported as a measured outcome, not an unearned
pass. DVIFM constants fit (native3, Y'CbCr) on this corpus: SROCC
0.7296→0.8401 at 3,154 rows in 1448.5s (over the ≤10-min gate); a 788-row
budget variant hits 206.1s (inside the gate) at SROCC 0.8190 — row count is
the sanctioned lever.

**v2 (joint-core-v2, 105,614 pairs = v1 52,963 + v2reused 8,331 + v2fresh
44,320), built by the transplant lane's step 0:** growing the core past
~s61k **clears** the gate-3 detectability floor on all five seeds (see
`../transplant-x4x5x7/README.md` for the full table). This is why every
later mechanism-transplant/zgeom/geometry lane runs on v2, not v1.

## Code in this directory

| file | role | which corpus |
|---|---|---|
| `assemble_features.py` | assembles the fresh-rendition feature/pair rows | v1 |
| `assemble_core.py` | assembles the v1 core table from admitted legs | v1 |
| `select_core.py` | v1 pair/rendition selection (band/class/origin/codec/q coverage targets) | v1 |
| `fit_core.py` | the leader-reproduction fit driver (gate 1/2) | v1 |
| `fit_forms.py` | constant-form (gate/curve/prior) fitting used by the verdict lane's X2 step | v1 (shared with `../verdict-x1x2/`) |
| `dvifm_equiv.py` | DVIFM histogram/per-block equivalence check (HIST vs BLOCK_BIN vs BLOCK_TRUE vs BLOCK_F32) — validates the constants-fit pipeline's exactness | v1 |
| `fit_repro.sh` | reproduction driver tying select→assemble→fit together | v1 |
| `select_core_v2.py` | v2 selection: merges v1 rows (cohort=v1) with v2-reused-leg growth rows | v2 — **see AT-RISK below** |
| `assemble_core_v2.py` | v2 assembly: merges generated fresh-cell rows into the v1-union growth table, verifies paths/coverage, emits the extraction input + provenance table | v2 |

## AT-RISK: `select_core_v2.py` is uncommitted in the main checkout

Unlike every other file in this preservation pass, `select_core_v2.py`'s
**source location** (`/home/lilith/work/zen/zensim/tools/joint_core/select_core_v2.py`
in the main checkout, not a lane workspace) is an **uncommitted addition** as
of this pass (`jj status` in the main checkout shows `A tools/joint_core/select_core_v2.py`
in the current working-copy commit, which has no description set). It sits
in the same shared working directory that a concurrently active session was
using while this preservation pass ran. The copy in this directory is safe
(it is a plain file copy, not a git/jj reference), but the **original**
in-tree file is not yet part of any finalized commit — recommend a follow-up
session commit it (or fold it into a joint-core-v2 commit) from the main
checkout directly.

## Rust / repo state (v1)

Committed directly to the **main checkout's own history**:
- `mqrmtrrz` / `333b07ab` — "joint-core-v1 build pipeline + tooling" — adds
  `tools/joint_core/{assemble_core,assemble_features,dvifm_equiv,fit_core}.py`,
  `fit_repro.sh`, `select_core.py`; Rust:
  `zensim-bench/examples/{core_variant_gen,flag_confirm}.rs` (new),
  `zensim-bench/examples/{extract_features_372col,shared/zen_decode}.rs`,
  `zensim/src/feature_v2.rs`.
- `mywlzowt` / `55b9703d` — "joint-core-v1 record — built, gates measured,
  core too small" — the benchmark record commit (docs/data only).

v2's Rust (`assemble_core_v2.py`'s consumer-side, and the growth-rendition
plumbing) landed inside the transplant lane's commit — see
`../transplant-x4x5x7/README.md`. Both are unpushed but fully committed
local history.

## How to re-run

**v1:** `select_core.py` (band/class/origin/codec/q targets) →
`assemble_features.py` (fresh renditions) → `assemble_core.py` (merge into
the core table) → `fit_core.py` (gate 1/2 leader-reproduction fit) →
`fit_repro.sh` ties these together. `dvifm_equiv.py` independently validates
the constants-fit pipeline's HIST/BLOCK_BIN/BLOCK_TRUE exactness claims.

**v2:** `select_core_v2.py` (merges v1 rows + v2reused growth) →
[generate v2fresh cells via the transplant lane's `gen_extract_inputs.py`,
see `../transplant-x4x5x7/`] → `assemble_core_v2.py` (merge fresh cells into
the growth table, emit `pairs_core.tsv` + `pairs_provenance.tsv` +
`plan/coverage_report.json`).

Inputs consumed: admitted TRAIN-origin legs only (fresh_imazen26, cid22
train-origin, human, fresh_safesyn, hdr, konfig train-origin — see
`../reports/CORE_DONE.md`'s coverage table for exact percentages); no
eval-corpus or holdout labels are read by this lane.

## Artifacts (reference by path)

- `/mnt/v/output/zensim/joint-core-v1/` — `_MANIFEST.json` (build commits,
  32 per-input sha256s, cluster/seed rules, kernel provenance per leg),
  `audit/`, `bench/`, `cache/`, `cells/`, `coverage_report.json`, `dists/`
  (11 GB total)
- `/mnt/v/output/zensim/joint-core-v2/_MANIFEST.json` +
  `plan/coverage_report.json` (built by the transplant lane)
- Committed benchmark record (already permanent in the repo, not copied
  here): `benchmarks/joint_core_v1_2026-09-20.{md,json,pointer.md}`,
  `docs/DATA_SPLITS.md` addenda (v1 and v2)
