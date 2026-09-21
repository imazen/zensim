# Task: DVIFM — PHASE 2c: is it a SUBSTITUTE, and does it matter on CODEC distortions?

Repo `/home/lilith/work/zen/zensim` (jj-colocated); Phase 2b is on `main@origin` (`8bff0ff7`). Read
`benchmarks/dvifm_screen2b_2026-09-19.md` + its prereg, `docs/DATA_SPLITS.md` (+ override), `docs/FULL_EVAL.md`,
`../DATA_PROVENANCE.md`, `CLAUDE.md`; `/home/lilith/.claude/CLAUDE.md` + `/home/lilith/work/zen/CLAUDE.md` bind you.
Reuse the 2b driver (`scripts/lib/feature_screen_ceiling.py` matrix mode) and its extraction; outputs under
`/mnt/v/output/zensim/dvifm-screen2c-2026-09-19/`. Work autonomously; a supervisor verifies from artifacts.

## Why
2b verdict INFO-NOT-USEFUL: the 30 DVIFM columns beat their permuted twins 10/10 (real within-image signal) but
lose to basic228 10/10 when ADDED to it (redundant). Two questions that result does not answer:
 Q1 SUBSTITUTION — DVIFM is planned as an i16 integer kernel that could be cheap. Does the fast tier gain from
    it? i.e. is `y60+dvifm30` closer to `basic228` than `y60` is, and does `dvifm30` alone carry how much?
 Q2 CODEC DISTORTIONS — 2b used KADID/TID/KonFiG human rows (mostly synthetic distortions). The product is codec
    targeting, and block-peak visibility is a codec-artifact hypothesis. Does the 2b ordering hold on the
    TRAIN-role codec panel (JPEG/WebP/AVIF/JXL sweeps with the registered two-reference proxy labels used by the
    Sept-13 scale study — find that panel's owner and admission; labels are PROXIES, say so everywhere)?

## Preregister FIRST (commit `benchmarks/dvifm_screen2c_prereg_2026-09-19.md` before any fit)
Same trainer, head h128, loss, LR cycle, pairs/epoch, final-epoch checkpoint, E ∈ {50,100}, the SAME ten paired
seeds as 2b, the same fixed permutation rule for control arms.
Q1 arms on the 2b human rows (N_max=8,327 fit; same 3,125-row dev + dev2): `y60`, `y60+dvifm30`, `y60+perm30`,
   `dvifm30` alone, `perm30` alone, `basic228` (reuse 2b fits where the cell is byte-identical — cite the hash;
   otherwise refit).
   Report paired Δ(y60+dvifm − y60), Δ(y60+dvifm − y60+perm), and the GAP CLOSED
   = Δ(y60+dvifm − y60) / Δ(basic228 − y60) on the within-reference primary metric and pooled SROCC, with SD,
   sign counts, and the paired bootstrap over dev references.
   Rule: SUBSTITUTE-CANDIDATE if Δ(y60+dvifm − y60) > 2·SD/√10 on the primary metric AND Δ vs the permuted arm
   > 2·SD/√10; else NOT-A-SUBSTITUTE.
Q2 arms on the codec panel: `basic228`, `basic228+dvifm30`, `basic228+perm30`, `y60`, `y60+dvifm30`. Requires
   w986 extraction of that panel's TRAIN rows through the SAME extraction owner (Parquet + `_MANIFEST.json` with
   `build_commit`, input sha256s). Source-disjoint fit/dev split by origin, recorded. If the panel is large,
   subsample by the existing stratified tooling to ≤40k fit rows and ≤10k dev rows and record the rule; state the
   extraction wall time you measured on the first 500 rows BEFORE committing to the size. Report per codec as
   well as pooled (JPEG is where an 8-lattice blockiness hypothesis should show first). Same decision rules as
   2b (ADVANCE / INFO-NOT-USEFUL / NEGATIVE / INCOMPLETE) plus the Q1 rule for the y60 pair.
NEVER use CID22 human scores / the 49-ref gold set, AIC-3, AIC-4, AIC2026, KonJND validation, KonFiG test, KADID
terminal refs {7,9}, or any secret holdout. Proxy-labelled codec rows must be TRAIN-role under the current split
registry — prove it with the admission record or stop and write BLOCKED.

## Execution and records
One heavy job at a time through `~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- … 2>&1 | tee
~/tmp/devin/<n>.log` (extraction, then the resumable ≤8-way fit fan-out). `df -h /mnt/v` must stay ≥80 GB free;
cap 40 GB. The 2b audit initially failed because checkpoint bakes lacked `zentrain.formula_revision` /
`feature_set_id` — stamp them as part of the driver, not post hoc.
Record `benchmarks/dvifm_screen2c_2026-09-19.{md,json}` (+ `.pointer.md`; nothing >30 KB in git), MISSING list
first, commands + commit, per-seed tables, verdict per question; append (2-space indent, append-only) to
`board_discussion_sets.json` role `train-development`; update the outcome section of
`benchmarks/dvifm_block_gates_2026-09-19.md`.

## Hard rules
`jj` only, small commits, **DO NOT PUSH**; never rewrite commits you did not create. `cargo fmt --all -- --check`,
`just clippy`, `just lint-scripts` clean at the end. No public API additions; never relax/`#[ignore]` tests (5
pre-existing `zensim-validate/tests/bake_surface.rs` failures untouched). No second trainer/extractor/statistic/
report pipeline. Scratch only in `~/tmp/devin/` (never `/tmp`, `/run`, `/dev/shm`); never delete caches/generated
data; local box only. Refresh `/home/lilith/work/zen/zensim/.workongoing` (`<UTC ts> devin-dvifm <activity>`)
every ≤2 min. Touch no other repo; no GitHub writes; no household names/MACs/LAN IPs; no `pgrep -f`. Do not start
optimisation, X/B channels or frozen EVAL.

## Reporting
`~/tmp/devin/dvifm2c_progress.log`; terminal `~/tmp/devin/dvifm_PHASE2C_DONE.md` (commits, verdict per question
with numbers, gap-closed table, per-codec table, what was NOT done) or `~/tmp/devin/dvifm2c_BLOCKED.md`.
