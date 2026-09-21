# Task: DVIFM — PHASE 2d: the LEADING-MODEL recipe, and where the constants are learned

Repo `/home/lilith/work/zen/zensim` (jj-colocated). Read the 2b/2c records in `benchmarks/dvifm_screen2{b,c}_*`,
`benchmarks/recovery_completion_2026-09-15.md` (§"Training, chronology and qualification"), `docs/DATA_SPLITS.md`
(+ the current override at its top), `docs/DATASET_HISTORY.md` (latest entries), `../DATA_PROVENANCE.md`,
`docs/FULL_EVAL.md`, `CLAUDE.md`. `/home/lilith/.claude/CLAUDE.md` + `/home/lilith/work/zen/CLAUDE.md` bind you.
Outputs: `/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/`. Work autonomously; a supervisor verifies artifacts.

## User direction (2026-09-19, verbatim)
"use the training recipie from prior leading models. for optimizing the coefficient and exponents you should
consider learning from cid22 test as they did, and compare it to other data sets like imazen26 crops with ssim2
as oracle"

Screens 2/2b/2c used a generic h128 recipe on the 8,327-row human TRAIN estate, where EVERY arm overfits. That is
not how the leading models are trained. Redo the comparison the way the leaders were built.

## Part A — the leading recipe, exactly
The frozen Rev3 leaders are `R915_basic228_h128_ens5` / `R915_y60_h32_ens5`:
`/var/tmp/zensim-validation-2026-09-15/recovery/` — `fits/*.raw.bin.spec.json` carry the full trainer argv + input
hashes; `tables/{safesyn,cid22,codec,human}_{fit,development,calibration}.parquet` (168,142 fit rows after
dedup; human leg rank-only; SafeSyn + CID22-train legs carry signed same-buffer SSIMULACRA2 targets), 120 epochs ×
50,000 draws, seeds 17101/17103/17107/17111/17113, disjoint sampler streams, dev checkpoint selection as recorded.
READ-ONLY: never modify anything under `/var/tmp/zensim-validation-2026-09-15/`.
1. Reproduce ONE control fit (`basic228`, seed 17101) from the recorded argv on the recorded tables and show it
   matches the frozen `.raw.bin` result (bit-identical, or state the measured difference and why). If you cannot
   reproduce the leader, STOP and write BLOCKED — nothing downstream means anything otherwise.
2. Re-extract the SAME rows (fit + development; same row keys, same order) at w986 with DVIFM on, through the same
   extraction owner and Rev3 arithmetic the leaders used; verify the first 956 columns are bit-identical to the
   R915 tables for the ids those models consume (if not, stop and report — it means the extraction era moved).
   Measure wall time on the first 1,000 rows and state the projected job length before running the rest (one
   run-heavy job, ≤8 threads). Parquet + `_MANIFEST.json` (`build_commit`, input sha256s, row-key hash).
3. Arms, identical recipe/seeds/streams/budget, only the input ids differ: `basic228` (control = the leader),
   `basic228+dvifm30`, `basic228+perm30` (one fixed recorded row permutation within each leg, fit and development
   separately), `y60/h32`, `y60+dvifm30/h32`, `y60+perm30/h32`. Five seeds each (the leaders' five). That is 30
   fits at ~7–14 min each: run ≤2 concurrent inside ONE run-heavy job, as the recovery did; resumable driver.
4. Decision on the TRAIN development legs only, per leg and as the registered composite, using the existing Rust
   panel owner: paired per-seed Δ(+dvifm − control), Δ(+dvifm − +perm), SD, sign counts, within-reference / local
   ordering panels, and the codec leg per codec. Same verdict vocabulary as 2b (ADVANCE / INFO-NOT-USEFUL /
   NEGATIVE / INCOMPLETE), preregistered with thresholds in `benchmarks/dvifm_screen2d_prereg_2026-09-19.md`
   BEFORE any fit. With five seeds the threshold is 2·SD/√5 and the sign count must be ≥4/5.

## Part B — where the DVIFM coefficients and exponents (g, P, C₀, β, ς per level) are learned
Fit the constants four ways with the SAME fitting tool (`tools/fit_params.py` lineage from Phase 2, same
optimiser, epochs, priors, linear 30→1 head), then run Part A's `+dvifm30` arms once per constant set (if wall time forces a cut, run Part A for the two constant sets that differ most and say so):
  K1 `kadid-tid` — the Phase-2 fitted-local constants already on main (human TRAIN rows).
  K2 `imazen26-ssim2` — imazen-26 CROPS with SSIMULACRA2 as the oracle. Use the canonical copy only
     (`png-v3` + manifest; NEVER `/mnt/v/imazen-26*` quarantined paths — see DATA_PROVENANCE), TRAIN-side origins
     under the origin even/odd split, codec-distorted through imazen codecs only (existing sweep outputs — do
     not encode anything new with foreign tools), targets = our `fast-ssim2`. Size to ≤20 GB of block cache.
  K3 `cid22-human-A` — **CID22 49-reference human scores, as the DVIFM authors did — under a recorded, limited
     exposure:** split the 49 references by one recorded seed into A (25 refs) and B (24 refs). Fit constants on
     A ONLY. B stays sealed: no fitting, no selection, no looking, except the single frozen read below.
     BEFORE touching any CID22 human score, append an exposure entry to `docs/DATASET_HISTORY.md` and the
     exposure ledger section of `docs/DATA_SPLITS.md`: date, user direction (quote above), which 25 refs, what is
     fitted (≈25 scalar constants + a throwaway linear head — no MLP, no feature selection, no checkpoint
     selection), and the consequence: any model consuming K3 constants has CID22-A exposure and may only quote
     CID22 on subset B, labelled as such. Commit that entry first.
  K4 `hdr_v3mix` — the subset B/BHdr liked (user 2026-09-19: "consider the other training subset bhdr liked").
     B's 80% "cid" head was a lasso trained ONLY on `hdr_v3mix` (7,410 zenjxl HDR-PQ renditions, 38 TRAIN origins,
     target cvvdp-mix = 0.5·clip01(ssim2/100) + 0.5·clip01((JOD−6)/4)) and it generalised to CID22 better than
     the SDR-corpus heads (`benchmarks/profile_b_methodology_2026-07-12.md` §heads; DATA_SPLITS row "hdr_v3mix
     @944 (hdr944-leg)", manifest `/mnt/v/output/zensim/hdr944-leg/_MANIFEST.json`; the 372-era table is
     `/mnt/v/output/zensim-multicodec-probe/hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet`). Use the 7,410
     TRAIN-origin rows only; the 3,900 val-origin rows (20 origins) are this leg's development set. DVIFM must be
     extracted through the HDR route (PQ, PU-normalised `DvifmNorm` — Phase 1 tested it bit-identical) from the
     original HDR sources + renditions; if the source pixels are not on disk, say so in MISSING and do K4 on
     whatever TRAIN-side subset is — do not re-encode. This is a NEW-REGIME leg: never column-mix its feature
     table with the SDR tables.
     K4 is used twice: (i) as a fourth constants-fit domain alongside K1–K3, and (ii) as a training subset in its
     own right — add arms trained ONLY on `hdr_v3mix` TRAIN rows with B's actual recipe for that head (lasso
     λ=0.002 on raw features, per the methodology doc; reproduce that head first and show it matches): `b-cid-head`
     (its 35 selected features' parent set) vs the same + dvifm30 vs + perm30, judged on the hdr_v3mix val origins
     AND on the SDR development legs of Part A (transfer is the point — B's finding was that this subset transfers).
  The ONLY permitted read of CID22-B and of the other holdouts in this phase: after ALL of Part A/B is frozen and
  recorded, report the throwaway linear-head SROCC of each constant set (K1/K2/K3/K4) on CID22-B as a single
  descriptive table, no iteration afterwards. AIC-3, AIC-4, AIC2026, KonJND validation, KonFiG test, KADID
  terminal refs, secret holdouts: untouched.
Report: the four constant sets side by side (do they agree? per level), the fit-domain loss of each set on the
other three fit domains (cross-domain transfer matrix — TRAIN-side data plus CID22-A only), and Part A's verdict
per constant set.

## Part C — DVIFM in its NATIVE colour space, CHROMA INCLUDED (user 2026-09-19: "are you also testing dvifm with
## its native binomial scales in yuv?" and "uv is key on the paper, and dvifm beat all SOTA … on cid22 and many others
## on srocc and kendall")
Everything so far ran the talk's native 5-level BINOMIAL pyramid but on zensim's XYB **Y** plane only. The method
is **Y′CbCr, BT.709 for SDR, all three channels; per-channel errors combined linearly, Cb and Cr weighted equally,
luma weighted higher** (`zenpapers/docs/iqa-methods/aic4-proposal-metrics.md` §1). A Y-only XYB test is not a test
of DVIFM. Chroma is MANDATORY here, not optional.
- `input_plane ∈ {xyb_y (current default), ycbcr_y, ycbcr_cb, ycbcr_cr}` as a crate-private field of the DVIFM
  spec / `DvifmParams`. Y′CbCr = full-range BT.709 from the gamma-encoded sRGB input (state the matrix); Y′ on
  [0,1]; Cb/Cr centred, then the spec's (c − min)/range rule. HDR keeps the PU route and is out of scope for C.
  Same pyramid, blocks, visibility and 30 features per plane. No new registered slots: a variant lands in the SAME
  f956..f985 columns under a distinct spec hash + feature-set identity; the research table for the 3-plane arm
  is three extractions joined on row key with documented column names — never silently column-mixed. Toggle-off
  and the default `xyb_y` spec stay bit-identical to main (assert). Extend `scripts/dvifm_parity_fixture.py` and
  the fixture test to the Y′CbCr planes (fixtures ≤30 KB).
- Constants per plane from that plane's own block cache (contrast units differ; never reuse XYB constants).
- Arms on the Part A recipe, five seeds, each with its own permuted control: `+dvifm90[ycbcr y+cb+cr]` (THE native
  arm), `+dvifm30[ycbcr_y]`, `+dvifm30[xyb_y]`, on both `basic228/h128` and `y60/h32`. Comparisons to report:
  native-90 vs control; native-90 vs Y′-30 (what chroma adds); Y′-30 vs XYB-Y-30 (what the colour space changes).

## Part D — STANDALONE DVIFM, the way its authors use it (do this FIRST; it is the cheapest and most decisive)
DVIFM is reported as a standalone metric that matches CVVDP and beats VMAF on CID22 and others — not as 30 extra
inputs to an MLP trained on SSIMULACRA2 proxies. Our screens asked a different question. Replicate the actual
claim before anything else:
- Model = the talk's: per plane p ∈ {Y′,Cb,Cr}, per level l: s_{p,l} = mean_b(v_b·m_b^P) then a learned Lp
  exponent; level weights positive summing to one; channel combination linear with w_Cb = w_Cr and a separate
  w_Y; monotone map to the score scale. Parameters: g, P, visibility (C₀/a, β/b, ς), L per (plane, level) or
  shared as the talk's 29-parameter luma-only count implies — state your parameterisation and its count; keep it
  ≤ ~100 parameters for the 3-plane model. Fit with the Phase-2 block-cache tooling (Adam over cached block
  records; no pixels re-read), optimising a rank-aware or MSE-to-MOS loss — preregister which.
- Fit data, mirroring theirs within our ledgered limits: CID22-A (25 refs; Part B's exposure entry covers this —
  commit the ledger entry BEFORE any CID22 score is read), TID2013 + KADID-10k TRAIN refs JPEG/JPEG2000 subsets.
  Variants: luma-only (their 29-parameter configuration) and 3-plane.
- ONE frozen read per variant on CID22-B (24 sealed refs): SROCC, KROCC (Kendall), PLCC, with a paired bootstrap
  over the 24 references against our `fast-ssim2`, zensim `B`, `D`, and the two frozen Rev3 ensembles scored on
  the same B rows through their public Rust surfaces, plus the XYB-Y-only standalone as the ablation. No
  iteration after that read. This is the replication number: does a ≤100-parameter block-visibility metric with
  chroma reach or beat SSIMULACRA2 on held-out CID22 references? Also report KADID dev refs {1,3,5} and KonFiG
  originsplit-val (TRAIN-side development) for all of them.
- Report what chroma buys (3-plane vs luma-only, paired), the fitted channel and level weights, and fit time.
- **Convex by construction** (user 2026-09-19): every mixing weight is NON-NEGATIVE and each mix sums to one —
  level weights (softmax/simplex parameterisation), channel weights (w_Y, w_C, w_C) on the simplex; the score is
  non-decreasing in every block error and zero at identity. No negative weight, no bias that can lower a score,
  anywhere. Assert it in a test on the fitted parameters and on random parameters (monotone in each s_{p,l}).
- **The visibility constants are the crux** (user: "tuning the contrast distortion sensitivity constant and neg
  exponent are crucial"): a = 1/C₀ (the contrast-sensitivity knee) and b = β (the negative exponent), per
  (plane, level). Do NOT trust a single Adam run from one init. For each (plane, level): a coarse log-grid over
  C₀ ∈ [1e-4, 0.3] × β ∈ [0.2, 1.4] (≥ 12×10) with the other parameters at their current best, then multi-start
  Adam refinement (≥ 8 starts from the best grid cells), alternating with the remaining parameters until the
  fit-domain loss stops improving. Commit the loss surface per (plane, level) as a small table/figure data file:
  how sharp is the optimum, do neighbouring levels agree, does chroma want a different knee than luma, and how
  much does the final CID22-A/KADID/TID fit loss change between the Phase-2 constants and the tuned ones? This is
  all fit-domain (TRAIN-side + CID22-A) work — the sealed CID22-B read still happens exactly once, after.
If Part D shows the standalone 3-plane model is competitive, say so plainly at the top of the DONE file — it
changes the plan (a cheap integer DVIFM profile becomes the goal rather than an add-on feature family).

## Part E — the REVERSE build: DVIFM first, zensim as the add-in (user 2026-09-19)
Base = ALL DVIFM Y′CbCr features (the 90, with Part D's tuned constants). Add zensim feature SUBSETS by family and
measure what each adds, instead of adding DVIFM to a full zensim set:
- Families (use the registry's own family/compute-token grouping from `feature_defs` / `feature_set_id`; do not
  invent groupings): e.g. the Y60 fast set, the rest of basic228 by family (ssim, edge/artifact-detail, mse, hf
  energy/magnitude, per scale), peaks, masked, IW, v2 append blocks, csfw.
- Procedure on the Part A leader recipe (five seeds; h32 for ≤ ~120 inputs, h128 above — state the rule): base-90
  alone; base-90 + each single family (one-step forward screen, each with a size-matched permuted control of that
  family's columns); then greedy forward selection by family for at most four steps, stopping when the paired gain
  is < 2·SD/√5. Report the accuracy-vs-extraction-cost frontier (use the measured per-family costs from the
  speed-matrix / feature-plan records; do not estimate) — the question is the smallest zensim subset that, with
  DVIFM-90, matches `basic228` and the frozen Rev3 rich ensemble on the TRAIN development legs.

## Part F — Y′CbCr as an ALTERNATIVE MODE for zensim itself, and 1.5× / 3× scales (user 2026-09-19)
"test yuv as an xyb alternative mode and how much we actually lose; also 1.5x and 3x scales". Research-only,
crate-private, toggle-off bit-identical, distinct feature-set identities, never column-mixed:
- F1 colour mode: compute the EXISTING zensim feature kernels (start with the basic228 set) on BT.709 Y′CbCr planes
  in place of XYB (Y′→Y slot, Cb→X slot, Cr→B slot; per-plane normalisation stated), same pyramid and kernels.
  Train with the Part A recipe/seeds → report how much is lost (or gained) vs XYB per development leg, AND the
  measured extraction-time difference (zenbench, interleaved, 1T, 64²/256²/1024²/4096²; the XYB conversion is the
  thing being avoided — measure it, never estimate).
- F2 scales: add 1.5× and 3× downsampled planes (so ladders like {1, 1.5, 2, 3, 4, …}) — for the DVIFM pyramid
  (extra octave-fraction levels) and, separately, for zensim's scale set. Use a principled resampler for
  non-integer factors (Mitchell via the imazen `zenresize` crate — read its local README/source first; no foreign
  resamplers), identical on both sides of a pair. Arms: dyadic control vs +1.5× vs +3× vs +1.5×+3×, one change at a
  time, Part A recipe, five seeds, fairness contract: identical seeds/streams, budget set by the slowest-converging
  arm, an arm that has not plateaued is INCOMPLETE. Include the label-free stability gates on TRAIN images before
  training (1-px shift, 8×8 codec-grid phase 0..7, centre-crop vs full) — a variant worse than the control on
  those is reported as such.
Parts E and F are large. They are LAST in the order of work; do as much as the box allows, in order, and list the
rest as NOT done. Do not shrink seeds or budgets to fit more in — do fewer arms properly.

## Order of work — THIS RUN = D → C → A only
D (standalone replication, convex, tuned a/b) → C (native planes + chroma as features) → A (leader recipe arms).
Parts B, E and F are the NEXT dispatches: do NOT start them in this run. They are described here so that your
spec fields, research-table layout, block caches and driver extensions do not preclude them (e.g. keep per-plane
block caches for all three Y′CbCr planes; keep the permuted-control and family-subset machinery general). Finish
D, C and A properly — full seeds, full budgets — then write the DONE file. If wall time or disk forces a cut, cut A
before C, and never D.

## Execution and records
One heavy job at a time via `~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- … 2>&1 | tee ~/tmp/devin/<n>.log`.
`df -h /mnt/v` ≥80 GB free at all times; this phase ≤60 GB (check first; if it cannot fit, shrink K2, not Part A).
Stamp `zentrain.formula_revision` + `feature_set_id` into every bake in the driver. Extend the existing owners
(`feature_screen_ceiling.py`, `run_full_eval.sh`, the extraction example, `zensim_mlp_train`, the Rust panel) — no
second trainer/extractor/statistic/report. Record `benchmarks/dvifm_screen2d_2026-09-19.{md,json}` + `.pointer.md`
(nothing >30 KB in git), MISSING list first, commit + exact commands, per-seed tables; append-only 2-space-indent
entry in `board_discussion_sets.json` (role `train-development`); update the outcome section of
`benchmarks/dvifm_block_gates_2026-09-19.md`.

## Hard rules
`jj` only, small commits, **DO NOT PUSH**; never rewrite commits you did not create. `cargo fmt --all -- --check`,
`just clippy`, `just lint-scripts` clean at the end. No public API additions; never relax/`#[ignore]` tests (5
pre-existing `zensim-validate/tests/bake_surface.rs` failures untouched). Scratch only `~/tmp/devin/` (never
`/tmp`, `/run`, `/dev/shm`); never delete caches/generated data; local box only — no cloud/fleet/tower. Refresh
`/home/lilith/work/zen/zensim/.workongoing` (`<UTC ts> devin-dvifm <activity>`) every ≤2 min. Touch no other repo;
no GitHub writes; no household names/MACs/LAN IPs; no `pgrep -f`. No optimisation, X/B channels or production
wiring.

## Reporting
`~/tmp/devin/dvifm2d_progress.log`; terminal `~/tmp/devin/dvifm_PHASE2D_DONE.md` (commits; leader-reproduction
evidence; verdict per arm family and constant set with numbers; constants table; transfer matrix; the single
CID22-B table; what was NOT done) or `~/tmp/devin/dvifm2d_BLOCKED.md`.
