# Task: DVIFM — PHASE 2b: the POWERED screen (control arm, within-image metric, data-scale curve)

Repo `/home/lilith/work/zen/zensim` (jj-colocated), everything through Phase 2 is on `main@origin` (`84d2a70e`).
Read first: `benchmarks/dvifm_screen_2026-09-19.md` — especially "Supervisor review" — and
`benchmarks/dvifm_screen_prereg_2026-09-19.md`, `docs/DATA_SPLITS.md` (+ current override), `docs/FULL_EVAL.md`,
`../DATA_PROVENANCE.md`, `CLAUDE.md`. `/home/lilith/.claude/CLAUDE.md` and `/home/lilith/work/zen/CLAUDE.md` bind you.
Reuse the Phase-2 tooling and output layout (`/mnt/v/output/zensim/dvifm-screen-2026-09-19/`); new results go
under `/mnt/v/output/zensim/dvifm-screen2b-2026-09-19/`. Work autonomously; a supervisor verifies from artifacts.

## Why this exists
Phase 2 was NEGATIVE at 8,000 TRAIN rows, but the DVIFM arm had 25–30% LOWER train loss and 0.003–0.005 lower
dev SROCC (overfitting signature), there was no dimensionality control, and the decision metric (pooled SROCC)
was not the hypothesis (LOCAL / within-image ordering). User ruling 2026-09-19: comparisons must be
apples-to-apples with the same seeds and budgets, and a SUFFICIENTLY LARGE budget for every arm. This screen
settles whether the family carries transferable information.

## Preregister FIRST (commit `benchmarks/dvifm_screen2b_prereg_2026-09-19.md` before any fit)
Arms (identical trainer, head h128, loss, LR cycle, pairs/epoch, rows, row order; only the input columns differ):
  A `basic228` · B `basic228+dvifm30` (fitted-local constants as on main) ·
  C `basic228+perm30` = the SAME 30 DVIFM columns with rows permuted by one fixed, recorded permutation applied
    independently within the fit rows and within the dev rows (same marginals, zero information) ·
  D `y60` speed control.
Seeds: 10 paired (init seed s, sample seed s+10000), same ten for every arm and cell.
Budgets: CYCLE-ALIGNED — the LR schedule is a 50-epoch cosine cycle, so E ∈ {50, 100}; evaluate the final-epoch
  checkpoint (lr≈0). Do not use E=40.
Data-scale axis: TRAIN fit sizes N ∈ {2,000, 8,000, N_max}, nested subsets (2k ⊂ 8k ⊂ N_max) drawn with one
  recorded seed, stratified the way the existing segment tooling stratifies. N_max = the largest TRAIN-role,
  human-labelled, admitted row set you can assemble under the CURRENT DATA_SPLITS rules with source-disjoint
  dev families (state exactly which corpora/rows, with admission hashes). Target ≥ 30,000 rows; if the admitted
  human TRAIN estate is smaller, use all of it and say so. Extract w986 for the new rows through the SAME
  extraction owner (`--full-986`), Parquet + `_MANIFEST.json` with `build_commit` and input sha256s.
  Dev: the same 3,125-row dev segment as Phase 2 for every cell, plus (reported separately) any additional
  source-disjoint TRAIN-dev rows that N_max makes available.
  NEVER: CID22 human scores / the 49-reference gold set, AIC-3, AIC-4, AIC2026, KonJND validation rows, secret
  holdouts — not for fitting, selection, or "a look".
Metrics per fit on dev: pooled signed SROCC; WITHIN-REFERENCE rank metrics from the existing Rust panel owner
  (the registered within-image / local-ordering panels — find the owner in docs/FULL_EVAL.md / WAVE_PLAYBOOK; do
  not write a new statistic); fit-side SROCC and final train loss (to report the generalisation gap).
Primary decision metric: the within-reference panel. Pooled SROCC is secondary.
Decision rule, evaluated at N_max and E=100 on the 10 paired seeds (report mean paired Δ, SD of paired Δ, and
the sign count; seed spread is not a CI — also give a paired bootstrap over dev references if the panel owner
supports it):
  ADVANCE        : Δ(B−A) > 2·SD/√10 on the primary metric AND Δ(B−C) > 0 AND pooled SROCC Δ(B−A) ≥ −SD/√10.
  INFO-NOT-USEFUL: Δ(B−C) > 2·SD/√10 but B does not beat A.  (the columns carry signal basic228 already has)
  NEGATIVE       : B ≈ C (|Δ(B−C)| ≤ 2·SD/√10) — the columns behave like noise.
  INCOMPLETE     : any arm's dev metric still rising by more than SD/√10 between E=50 and E=100 at N_max → add
                   E=150 for ALL arms and re-evaluate once.
Also report Δ(B−A) and Δ(B−C) as a function of N (the learning-curve table) — the trend is the evidence about
whether more data would change the answer.

## Execution
- ~240 fits of ~1–3 min each. Run them as ONE run-heavy job that fans out ≤8 single-threaded fits at a time
  (`~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- <driver> 2>&1 | tee ~/tmp/devin/dvifm2b_fits.log`); the
  driver must be resumable (skip a cell whose RESULT exists) and write one JSON per fit. Extraction is a separate,
  earlier heavy job — never two heavy jobs at once. Check `free -h` and `df -h /mnt/v` first (keep ≥80 GB free;
  cap this phase at 40 GB).
- Extend the existing screen driver (`scripts/lib/feature_screen_ceiling.py` + `scripts/run_full_eval.sh`); do
  not create a second trainer, extractor, statistic or report pipeline. The permuted arm is a column transform
  of the extracted table, recorded with its permutation seed and a sha256 of the permuted table.
- Record: `benchmarks/dvifm_screen2b_2026-09-19.{md,json}` (+ `.pointer.md` for anything >30 KB), MISSING list
  first, git commit + exact commands, the full per-seed table, learning-curve table, verdict per the rule above,
  and an update to the outcome section of `benchmarks/dvifm_block_gates_2026-09-19.md` and
  `board_discussion_sets.json` (role `train-development`; keep the file's 2-space indent and append only — do
  NOT re-serialise the whole file).

## Hard rules
`jj` only, small commits, **DO NOT PUSH**, never rewrite commits you did not create. `cargo fmt -p <crate>`
before each commit; `cargo fmt --all -- --check`, `just clippy`, `just lint-scripts` clean at the end. No public
API additions; never relax or `#[ignore]` tests (5 pre-existing `zensim-validate/tests/bake_surface.rs` failures
stay untouched). Do not commit `__pycache__`. Scratch in `~/tmp/devin/` only — never `/tmp`, `/run`, `/dev/shm`.
Never delete caches or generated data. Local box only: no cloud, fleet or tower. Refresh
`/home/lilith/work/zen/zensim/.workongoing` (`<UTC ts> devin-dvifm <activity>`) every ≤2 min and before long
commands. Touch no other repo; no GitHub writes; no household names/MACs/LAN IPs; no `pgrep -f`.

## Reporting
`~/tmp/devin/dvifm2b_progress.log` per step; terminal file `~/tmp/devin/dvifm_PHASE2B_DONE.md` (commits, N_max
and its provenance, verdict with numbers, learning-curve table, what was NOT done) or
`~/tmp/devin/dvifm2b_BLOCKED.md`. Do not start optimisation, X/B channels, or frozen EVAL. A measured negative
is a good outcome; an unearned positive is the worst one.
