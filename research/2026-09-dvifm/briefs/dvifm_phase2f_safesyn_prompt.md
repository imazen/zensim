# Task: DVIFM — PHASE 2f: learn the constants from SAFESYN instead of the CID22 test set

Repo `/home/lilith/work/zen/zensim` (jj-colocated). Prerequisite: Phase 2d's replication must be VALIDATED — the
standalone convex Y′CbCr model fitted on CID22-A with its one frozen read on the sealed CID22-B already recorded in
`benchmarks/dvifm_screen2d_2026-09-19.md`. If that read has not happened, STOP and write
`~/tmp/devin/dvifm2f_BLOCKED.md`. Read 2d's record + prereg + Amendment 1, `docs/DATA_SPLITS.md` (+ override),
`../DATA_PROVENANCE.md`, `CLAUDE.md`. `/home/lilith/.claude/CLAUDE.md` + `/home/lilith/work/zen/CLAUDE.md` bind you.
Outputs: `/mnt/v/output/zensim/dvifm-screen2f-2026-09-19/`.

## User direction (2026-09-19)
"after we validate we can replicate the results, we will try to learn values from safesyn rather than the cid test
set." Goal: the same DVIFM constants (per plane and level: g, P, C₀ = 1/a, β = b, ς), fitted on an ALL-TRAIN corpus
with no holdout exposure at all, and a like-for-like comparison against the CID22-A-fitted constants.

## Fit domain
SafeSyn, the leaders' own tables: `/var/tmp/zensim-validation-2026-09-15/recovery/tables/safesyn_{fit,development}.parquet`
(141,054 fit / 38,758 development rows; `ref_basename` + `human_score`; target is SSIMULACRA2-derived — a METRIC
TEACHER, not human MOS: say so on every table and in the record, and never call a SafeSyn-fitted result a human
result). READ-ONLY: never modify anything under `/var/tmp/zensim-validation-2026-09-15/`.
Pixels: resolve each row's reference/distorted bytes through the existing owner (SafeSyn is bitstream-only on
`/mnt/v` — decode through `zencodec`/the existing extraction path; do NOT re-encode anything, do NOT use any
non-imazen codec). If a row's pixels cannot be resolved, drop it and report the count.

## Disk reality — this is the binding constraint, plan it FIRST
Measured: a full DVIFM block cache costs ~971 KB per row per plane (Phase 2d, `cache/cid22a_ycbcr_y.bin`). SafeSyn
at 180k rows × 3 planes would be ~520 GB. `/mnt/v` has ~102 GB free and Phase 2d already holds 60 GB of caches.
So:
1. **Cap blocks per row at extraction time.** Add a per-row block cap (deterministic stride subsample, cap and
   seed recorded in the cache header/manifest) to the cache writer — the fitter already stride-subsamples for the
   Adam step, so this only moves that decision earlier. Target ≤ 40 KB/row/plane.
2. **Subsample rows** by the existing stratified/clustered tooling (not at random) to a stated budget; record the
   rule and the seed. Start from ≤ 20,000 fit rows + ≤ 6,000 development rows and say what that costs.
3. **Prove the subsample is adequate before trusting it:** refit the CID22-A constants from a capped cache and
   compare, per (plane, level), against 2d's full-cache constants and the fit-domain loss/SROCC. Report the
   difference; if a constant moves more than its ±1-grid-step sharpness, raise the cap and repeat.
4. Keep the whole phase ≤ 25 GB and leave ≥ 80 GB free on `/mnt/v`; check `df -h /mnt/v` before each stage. Do not
   delete any existing cache — if you need room, STOP and report; the supervisor mirrors to Tower and decides.

## Compactness (user 2026-09-19: "don't use storage like that, we need to be compact")
Read `docs/FITTED_CONSTANT_GUARDS_2026-09-19.md` §"Cache and storage discipline" and follow it. In particular:
- **A (C̃, m) 2-D histogram per (plane, level)** — 256×256 bins in the integer log domain, a few hundred KB per
  domain — supports the WHOLE C₀ × β grid exactly, because each cell's loss is a sum of per-block terms. Build the
  grid stage on histograms, not on per-block records, and verify on one small domain that the histogram grid
  reproduces the per-block grid to within the ±1-step sharpness.
- Keep per-block records only for the parameters that need per-block gradients, only on the capped subsample, and
  **quantised** (f16, or i16 in the kernel's integer domain) — check the quantisation against a full-precision
  cache on one small domain.
- One domain live at a time: fit it, write its constants/surfaces, release its cache before extracting the next.
- Whole phase ≤ 25 GB, ≥ 80 GB free on `/mnt/v`, `df` checked before each stage. If the design needs more, redesign.

## Fits and comparison
Same fitter and protocol as 2d (per-cell output-map refit; 18×18 log-grid over C₀ × β with edge extension; ≥8
multi-start Adam refinements per (plane, level), alternating with the convex head; convexity asserted; loss
surfaces and ±1-step sharpness committed). Variants: `native3` (Y′CbCr, all three planes) ONLY — **no luma-only arms** (user, 2026-09-19: "we can stop doing luma only work"; chroma replicated as the real gain: +0.027 SROCC on CID22-A, +0.028 on TID/KADID, while Y′-vs-XYB-Y did not replicate).
Report side by side, per plane and level: SafeSyn-fitted vs CID22-A-fitted constants — do they agree? where do they
differ, and by more than the sharpness? Include the fit-domain loss of each constant set evaluated on the other's
domain (transfer matrix), and the ssim2-teacher caveat.

## Judging — no new holdout exposure without asking
Judge the SafeSyn-fitted constants on: the SafeSyn development rows; the TRAIN-side development legs used in 2d;
the KADID development refs {1,3,5}; KonFiG originsplit-val. Report SROCC, KROCC, PLCC.
**Do NOT read CID22-B, AIC-3/4, AIC2026, KonJND val, KonFiG test, KADID terminal refs or any secret holdout.** 2d
already spent the single preregistered CID22-B read. If the SafeSyn-fitted model beats the CID22-A-fitted model on
every TRAIN-side judge above, write that conclusion and STOP — the supervisor asks the user whether to spend a
second, separately ledgered CID22-B read.

## Preregister first
`benchmarks/dvifm_screen2f_prereg_2026-09-19.md`, committed before any fit: the row/block subsample rule and seeds,
the cap-adequacy gate, the arms, the judges, and the decision rule (which constant set becomes the default and on
what evidence). Record everything — including "SafeSyn-fitted is worse" — in
`benchmarks/dvifm_screen2f_2026-09-19.{md,json}` + `.pointer.md` (nothing >30 KB in git), MISSING list first,
commands + commit, append-only 2-space-indent `board_discussion_sets.json` entry (role `train-development`), and an
outcome line in `benchmarks/dvifm_block_gates_2026-09-19.md`.

## Target rule (user, 2026-09-19: "don't clip ssim2 scores!")
Never clip a signed teacher target. SSIMULACRA2 goes negative on badly damaged pairs; `clip01` pins those rows at
a floor with no gradient (measured: 7.2% of the imazen26 leg) and biases every constant fitted against them. Use
the RAW SIGNED value; if a loss needs a bounded target use a strictly monotone, invertible squash and record it.
Drop a row only when the teacher is undefined, never because it is negative. Re-check any table you inherit for a
clamped target before fitting on it, and say in the record which tables were clamped.

## Rules
`jj` only, small commits, **DO NOT PUSH**; never rewrite commits you did not create; never delete caches/generated
data. One heavy job at a time through `~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- … 2>&1 | tee
~/tmp/devin/<n>.log`; scratch only `~/tmp/devin/`, never `/tmp`. `cargo fmt --all -- --check`, `just clippy`,
`just lint-scripts` clean at the end; no public API additions; never relax/`#[ignore]` a test (the 5 pre-existing
`zensim-validate/tests/bake_surface.rs` failures stay). Refresh `/home/lilith/work/zen/zensim/.workongoing`
(`<UTC ts> devin-dvifm <activity>`) every ≤2 min; another session may overwrite it — ignore that. Touch no other
repo; no GitHub writes; no household names/MACs/LAN IPs; no `pgrep -f`.

## Reporting
`~/tmp/devin/dvifm2f_progress.log` per step; terminal `~/tmp/devin/dvifm_PHASE2F_DONE.md` with the constants
comparison table, the cap-adequacy evidence, the judge table and what was NOT done — or
`~/tmp/devin/dvifm2f_BLOCKED.md`.
