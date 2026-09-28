# Task: DVIFM — PHASE 2e: Y′CbCr DVIFM + XYB zensim (early and late fusion, separate passes), reverse build

Repo `/home/lilith/work/zen/zensim` (jj-colocated). Phase 2d (standalone replication D, native-plane features C,
leader-recipe arms A) is done — read `~/tmp/devin/dvifm_PHASE2D_DONE.md`, `benchmarks/dvifm_screen2d_2026-09-19.md`
and its prereg, `docs/PREREG_SCALES_PLANES_2026-09-19.md` (legs L5, L6), `docs/PLAN_SPATIAL_STEERING_DVIFM_2026-09-19.md`,
`docs/DATA_SPLITS.md` (+ override + the CID22-A exposure entry), `../DATA_PROVENANCE.md`, `CLAUDE.md`.
`/home/lilith/.claude/CLAUDE.md` + `/home/lilith/work/zen/CLAUDE.md` bind you. Reuse 2d's tables, block caches,
driver and recipe verbatim. Outputs: `/mnt/v/output/zensim/dvifm-screen2e-2026-09-19/`.
If 2d's Part D or Part A did not complete, STOP and write BLOCKED saying what is missing.

## User direction (2026-09-19, verbatim)
"deprioritize the 3x and 1.5x and try yuv dvifm with zyb zensim, separate passes too."
"we should consider testing a reverse, with dvifm yuv features, all of them, and only adding in subsets of zensim"
(zyb = XYB. The 1.5×/3× scale work is NOT part of this run.)

## Preregister FIRST (`benchmarks/dvifm_screen2e_prereg_2026-09-19.md`, committed before any fit)
Recipe = 2d Part A's reproduced leader recipe: same tables, 120×50,000 draws, seeds 17101/17103/17107/17111/17113,
same sampler streams, same checkpoint rule; five paired seeds; threshold 2·SD/√5 and sign count ≥4/5; decision on
the TRAIN development legs (per leg + registered composite + within-reference panels + codec leg per codec) via
the existing Rust panel owner. Holdouts untouched (CID22-B stays sealed — 2d already spent its one read; AIC-3/4,
AIC2026, KonJND val, KonFiG test, KADID terminal refs, secret sets: never).

### E1 — EARLY fusion, separate extraction passes
One head over [XYB zensim columns ‖ Y′CbCr DVIFM-90 columns]. The two families are extracted by their own passes
(zensim's XYB walk; DVIFM's own Y′CbCr binomial pyramid) — no shared planes, and say so in the record. Arms:
`basic228(xyb)+dvifm90(ycbcr)` vs `basic228` vs `basic228+perm90`; `y60(xyb)+dvifm90(ycbcr)` vs `y60` vs
`y60+perm90`. Reuse 2d-C fits where a cell is byte-identical (cite the hash); otherwise fit.

### E2 — LATE fusion, fully separate passes/models
score = M( w·S_dvifm + (1−w)·S_zensim ), w ∈ [0,1] (convex), M one monotone map (the existing spline owner). S_dvifm
= 2d-D's standalone convex 3-plane model, frozen. S_zensim ∈ {the 2d-A `basic228` control ensemble, the `y60`
control ensemble}, frozen. Fit ONLY w and M on the TRAIN fit rows; judge on the development legs. Report w, the
gain over each component alone, and against E1's early-fusion arm at the same inputs. Also report the
"skip" operating points the separate-pass design allows: DVIFM alone, zensim alone, both — accuracy vs measured
extraction time (zenbench records from the speed matrix + the DVIFM cost gate; measured numbers only).
Note for the record: late fusion preserves DVIFM's exactly-additive block map for steering; early fusion does not.

### E3 — the REVERSE build
Base = DVIFM-90 (Y′CbCr, 2d-D tuned constants) in a head by itself. Add zensim XYB feature SUBSETS by the
registry's own family / compute-token grouping (from `feature_defs` / `feature_set_id` — do not invent groups):
the Y60 fast set; the remaining basic228 families (ssim, edge/artifact-detail, mse, hf energy/magnitude — per
scale where the registry splits them); peaks; masked; IW; the v2 append blocks; csfw.
Step 1: base alone, and base + each single family, each with a size-matched permuted control of THAT family's
columns. Step 2: greedy forward selection by family, at most four steps, stop when the paired gain < 2·SD/√5.
Head size rule stated up front (h32 for ≤120 inputs, h128 above). Deliverable: the accuracy-vs-extraction-cost
frontier, and the smallest zensim subset that with DVIFM-90 matches (within 2·SD/√5) the `basic228` control and
the frozen Rev3 rich ensemble on the development legs — or the statement that none does.

Order: E2 (cheapest: two scalars on frozen outputs) → E1 → E3. If wall time forces a cut, cut E3's step 2 first.

## Execution and records
One heavy job at a time via `~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- … 2>&1 | tee ~/tmp/devin/<n>.log`;
≤2 concurrent leader-recipe fits inside one job; resumable driver; bakes stamped with `zentrain.formula_revision`
+ `feature_set_id` in the driver. `df -h /mnt/v` ≥80 GB free; this phase ≤30 GB. Extend the existing owners only.
Record `benchmarks/dvifm_screen2e_2026-09-19.{md,json}` + `.pointer.md` (nothing >30 KB in git), MISSING list
first, commit + exact commands, per-seed tables, verdict per part; append-only 2-space-indent entry in
`board_discussion_sets.json` (role `train-development`); update `benchmarks/dvifm_block_gates_2026-09-19.md`.

## Hard rules
`jj` only, small commits, **DO NOT PUSH**; never rewrite commits you did not create; if `main` moved, do not
rebase — the supervisor does. `cargo fmt --all -- --check`, `just clippy`, `just lint-scripts` clean at the end. No
public API additions; never relax/`#[ignore]` tests (5 pre-existing `zensim-validate/tests/bake_surface.rs` failures
untouched). Scratch only `~/tmp/devin/` (never `/tmp`, `/run`, `/dev/shm`); never delete caches/generated data;
local box only. Refresh `/home/lilith/work/zen/zensim/.workongoing` (`<UTC ts> devin-dvifm <activity>`) every ≤2
min. Touch no other repo; no GitHub writes; no household names/MACs/LAN IPs; no `pgrep -f`. No optimisation work,
no 1.5×/3× scales, no Y′CbCr-mode zensim kernels, no production wiring in this run.

## Reporting
`~/tmp/devin/dvifm2e_progress.log`; terminal `~/tmp/devin/dvifm_PHASE2E_DONE.md` (commits; E2 w and gains; E1
verdicts; E3 frontier table; what was NOT done) or `~/tmp/devin/dvifm2e_BLOCKED.md`.
