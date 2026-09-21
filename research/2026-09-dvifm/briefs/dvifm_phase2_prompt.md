# Task: DVIFM block-visibility — PHASE 2: the preregistered TRAIN screen

Repo: `/home/lilith/work/zen/zensim` (jj-colocated). Phase 1 (kernel, registration f956..f985, gates) is on
`main@origin` at `7fbea901`; read `benchmarks/dvifm_block_gates_2026-09-19.md` and
`~/tmp/devin/dvifm_PHASE1_DONE.md` first. A supervising session verifies from artifacts. Work autonomously.

## Read before anything
- `/home/lilith/work/zen/zenpapers/docs/iqa-methods/dvifm-zensim-worker-brief.md` — section "Screen" is your
  assignment; `dvifm-zensim-feature-design.md` §"How the weights and the constants get learned" + §"Screening".
- zensim: `CLAUDE.md`, `docs/DATA_SPLITS.md` (and its current override), `docs/FULL_EVAL.md` (feature screens),
  `docs/PRODUCTION_PRIORITIES_2026-09-15.md` (P2 + "Rules for the next executing agent"),
  `../DATA_PROVENANCE.md`, and the csfw screen precedent commit `2eebd76a` (`git show`).
- `/home/lilith/.claude/CLAUDE.md` + `/home/lilith/work/zen/CLAUDE.md` rules bind you.

## Fairness contract (user ruling 2026-09-19 — non-negotiable)
An earlier feature study was ruled INVALID because its arms were short-budget and not comparable. So:
1. Arms differ in exactly ONE thing (the 30 DVIFM columns). Same trainer binary, head shape, loss, LR cycle,
   epochs, pairs/epoch, TRAIN rows, row order, fit/dev families.
2. Identical init seeds AND identical sampling seeds across arms; ≥3 paired seeds; report per-seed paired
   differences, never bare means. Seed spread is not a confidence interval — say so.
3. Budget = what the SLOWEST-converging arm needs. Before the screen, run one convergence probe per arm and
   show the dev curve is flat (within seed noise) over the last 20% of epochs. If an arm has not plateaued,
   raise the budget for ALL arms. A screen whose arms have not both plateaued is INCOMPLETE, not negative.
4. Selection and every constant (C0 quantiles, F2 bin centres, fitted g/P/C0/β/ς) come from TRAIN fit rows only;
   the advancement decision uses source-disjoint TRAIN dev families. NEVER touch CID22 human scores, the CID22
   49-reference gold set, AIC-3, AIC-4, AIC2026, KonJND val, or any secret holdout — not for fitting, not for
   selection, not "just to look".
5. Write the preregistration (hypothesis, arms, seeds, budget + how it was set, advancement rule, disk budget
   for the block-stats cache, what counts as negative/incomplete) to
   `benchmarks/dvifm_screen_prereg_2026-09-19.md` and COMMIT it BEFORE running any training.

## Work, in order
1. Wire the training-only block-stats side output of the canonical extractor (behind `training`; 18 f32/block;
   do not build a second extractor). Size the TRAIN subsample to a stated disk budget (≤40 GB under
   `/mnt/v/output/zensim/dvifm-screen-2026-09-19/`; check `df -h /mnt/v` first — it is ~96% full, leave ≥80 GB
   free). Choose the subsample by the existing clustered/stratified TRAIN tooling, not at random; record how.
2. First-screen constants from the cache: g=1, P=1, β=0.65, ς=4; per level C0 = TRAIN 10th percentile of C̃, F2
   bin centres = TRAIN {10,30,50,70,90}% quantiles of ln min(C̃_ref, C̃_dist). Bake with provenance.
3. Re-extract the TRAIN subsample with the family ON under the new feature-set identity (w986), through the
   existing extraction owner. Persist tables as Parquet with a `_MANIFEST.json` carrying `build_commit`,
   input sha256s and row keys.
4. Screens, in the brief's order: (1) five-minute mechanism check (`scripts/run_full_eval.sh --stage
   feature-screen`); (2) basic228 vs basic228+30, with Y60 as the speed control; (3) same with the local band;
   (4) fit g,P,C0,β,ς per level on the cache (Python is fine for this invention step), re-bake, rescreen the
   better band — run (4) once even if (2) fails. Only a survivor goes to five-seed confirmation; do NOT run
   frozen EVAL assessment — stop and report instead.
5. Record every outcome, failures included: `benchmarks/dvifm_screen_2026-09-19.{md,json}` (git commit,
   commands, seeds, per-seed table, convergence evidence, cost line: DVIFM is +31.6 ms at 1024² vs the 50 ms
   p95 bar, so state what gain would justify optimisation), the ledger, and `board_discussion_sets.json` with role
   `train-development`. Files >30 KB do not go in git — put them under the output dir and commit a
   `.pointer.md`.

## Hard rules
- `jj` only for writes; small commits; **DO NOT PUSH**; never rewrite commits you did not create; no
  `jj workspace add` that shares the main `target/` dir (Phase 1 clobbered fingerprints that way — if you need
  a second build, give it its own `CARGO_TARGET_DIR` under `~/tmp/devin/`).
- Heavy work only via `~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- <cmd> 2>&1 | tee ~/tmp/devin/<n>.log`,
  ONE heavy job at a time, never truncated with head/tail. Check `free -h` first. This is the local box only:
  no cloud, no fleet launches, no tower.
- Never write to `/tmp`, `/run`, `/dev/shm`. Never delete caches or generated data (rename to `.bak`).
- No public API additions (use `pub(crate)` / `#[cfg(test)]`; Phase 1's one `pub fn` was narrowed by the
  supervisor). Never relax or `#[ignore]` a test. Known pre-existing failures to leave alone: 5 tests in
  `zensim-validate/tests/bake_surface.rs`.
- Run `cargo fmt -p <crate>` on what you touch before each commit (`cargo fmt --all -- --check` must be clean;
  Phase 1 left 32 unformatted hunks). `just clippy` + `just lint-scripts` clean at the end.
- Refresh `/home/lilith/work/zen/zensim/.workongoing` (`<UTC ts> devin-dvifm <activity>`) every ≤2 min of
  activity and before each long command.
- Touch no other repository. No GitHub issues/PRs/comments. No household names, MACs or LAN IPs in files.
- Do not use `pgrep -f`/`pkill -f`.

## Reporting
Append to `~/tmp/devin/dvifm2_progress.log` at each step. At the end — or when genuinely blocked — write
`~/tmp/devin/dvifm_PHASE2_DONE.md`: commits, the preregistered verdict per screen (ADVANCE / NEGATIVE /
INCOMPLETE with the numbers), convergence evidence, what was NOT done. If blocked, write
`~/tmp/devin/dvifm2_BLOCKED.md` with what you tried. A measured negative is a good outcome; an unearned
positive is the worst one.
