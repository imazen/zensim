# SESSION-RESUME — read this first after every compact

**Last updated: 2026-08-04 (consolidation wave).** The current era is the
**SOTA-944 model campaign** — pre-registered, five seed/lever waves + two
ensemble waves, all appended in place in the one authority doc:
[`benchmarks/sota944_campaign_2026-08-03.md`](benchmarks/sota944_campaign_2026-08-03.md).
Everything before it (372-era, 720/924-era) is historical context, era-tagged —
never compare numbers across eras without the doc's era-bridge notes.

## Current true state (2026-08-04)

- **★ 2026-08-05: Profile `C` SHIPPED (user-gated)** — the wave-11
  battery-selected `W10L9_s4003_packed` (k=8-confirmed corrected-mix recipe,
  appendix K.R) is `ZensimProfile::C` (`zensim-c`), weight
  `zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin` (sha `1a2c8d52…`,
  first PRUNED shipped bake, caller 944 / internal 667). **`B` remains the
  default** — C-vs-B is a stated trade (C: CID22/LIVE/CSIQ/nonphoto/dial-mono/
  M3a + corruption head; B: KonJND + HF-NL). No crates.io publish (separately
  gated). Repro + provenance: `docs/PROFILE_C_REPRODUCTION_2026-08-05.md`;
  distribution `s3://zentrain/profiles/C-2026-08-05/` + Tower.
- **Regime = 944** (folded+append+append2). Canonical data roots + grids are
  resolved by `bake_verdict --regime 944` itself (test-pinned; see entry
  point 1). REGIME PURITY is absolute: never column-mix 944 rows with
  720/372 parquets.
- **Campaign standing result:** the frozen 5-row bar (CID22 > 0.8924, KonJND
  ≥ 0.43, nonphoto ≥ 0.90, HF-NL-proxy, dial) has been cleared row-by-row but
  **never by one artifact**. Registered levers (seed scale n=23, near-top
  anchor, coherence, wave-4 combos) were honest nulls; the stabilized
  single-model ceiling is **`C_co3a_s1301` CID22 0.89067** (KonJND 0.405,
  nonphoto 0.905, HF-NL +0.251, dial 95.9%/0%).
- **Seed ENSEMBLES moved both blockers:** wave 5 `W5_E1_k2` CID22 **0.89425**
  (first CID22-bar pass in 64+ draws; paired bootstrap P(Δ>0)=0.968), wave 6
  `W6_GE2_trio` KonJND **0.4543** (the KonJND blocker broken; binding row is
  now CID22). Ensembles are evaluation functions, not shippable bakes — M3a
  not computable for them; distillation (wave-6 arm F) is the ship route and
  is **in flight** as of this writing.
- **Freeze decision = the USER'S, pending:** stabilized ~0.891 with better
  secondary axes vs. the unstable 924-era 0.8924 peak (EM4 — which fails the
  campaign's own HF-NL row; see the doc's Corrections section). Freeze bars +
  owner map: `freeze_check` (zensim-validate) +
  `benchmarks/decision_surface_audit_2026-07-31.md`.
- **G-RANGE on 944 MLPs** (2026-08-04 addendum): the gate tool now evaluates
  every bake class, and it surfaced that no 944 MLP candidate carries an
  output spline — dial packaging (`bake_dial_refit add-spline`) is required
  before that bar row can be judged on a freeze candidate.
- **Selection frame = the BALANCED profile (user-directed, 2026-08-04):** the
  user lowered the bar to surface candidates balanced across bands, datasets
  and uses — registered as `freeze_check --profile balanced-2026-08-04`
  (campaign doc AMENDMENT 8 + its RESULTS section: floors, composite, classes,
  full pass matrix, frontier, trade cards; §1 stays the freeze bar). Headline:
  **0/172 board cells pass all 8 floors** — classic-IQA breadth (CSIQ/LIVE ≥
  0.83) is the 944 era's binding balance axis; wave-7's `H_co3abpg_s2507` is
  the frontier-top single (kon 0.459 ∧ breadth ∧ nonphoto ∧ M3a 0.866 GOLD,
  missing only CID22 by 0.0045), and the packaging pass showed all raw-unit
  dial-mono numbers are unit-flattered (no packaged cell holds ≥93% in dial
  units — see the doc's unit caveat).

## ⇒ 2026-08-25: LAN era + the refinement plan (read after this file)

- **Fleet + storage are LAN-local** (user directive 2026-08-08; store =
  SeaweedFS on tower, `ZEN_STORE` defaults to LAN since 08-10, buckets keep
  their R2 names, R2 = cold/user-gated rundown). The operator cheat-sheet,
  the discrepancy list (production enroll script still R2-pinned; `fleet
  status` reads 0 on LAN — use `pool_progress.py`) and the program order live in
  [`docs/PLAN_LAN_ERA_REFINEMENT_2026-08-25.md`](docs/PLAN_LAN_ERA_REFINEMENT_2026-08-25.md).
  **The DONE test for the whole program** (user directive 2026-08-25, <4k chars):
  [`docs/GOAL_PRODUCTION_READINESS_2026-08-25.md`](docs/GOAL_PRODUCTION_READINESS_2026-08-25.md).
- **Wave-12 (appendix AD, 2026-08-21) is pre-registered, data-gate OPEN, and
  was never launched** — no `W12_s*` bake exists. It is the first compute to
  start (plan §4 B1; amend AD.7 by measurement — the box is 60 GiB again).
- **HDR phase-2 corpus (appendix S)**: encode drained 99.888 % on 08-07, score
  waves declared on R2 and never harvested, orientation gate PENDING
  (`/mnt/v/output/hdrgrid-2026-08-06/_MANIFEST.json`). Plan §4 B4.
- **jxl loop has NO secant controller** (power law exp 1.0 / clamp 2.0, k2
  18/27 · k3 24/27); the secant/bracket/per-tile arms are plan §5. **AV1
  steering** moves to the λ-side (rdmult) channel across zenrav1e /
  zenav1-svt / zenav1-aom behind one harness — plan §6.

## ⇒ 2026-08-25 GOAL RUN progress (docs/GOAL_PRODUCTION_READINESS_2026-08-25.md)

Per-criterion state (committed shas; `git -C <repo> merge-base --is-ancestor <sha> origin/<branch>` to verify):
- **C3 MODELS (SDR): CONVERGED.** Wave-12 (avif944 leg) run to completion — Profile C stands 8/8;
  best full-weight seed 7/8 (F1 CID22). The registered half-weight (w=0.5522) follow-up found the
  first 8/8 wave-12 seed (`W12hw_s4203`: CID22 0.8881>C, AVIF-dial 0.9673, M3a 0.8836) — a LATERAL
  trade with C (worse CSIQ/KonJND/nonphoto), G-AC3 not met, C still selected. avif944 adopted as a
  standing mix leg @ w≈0.55. zensim `AD.R`/`AD.R.1` (`bf15ed9e`,`1ad91786`). HDR models: NOT started
  (needs the HDR corpus, criterion 2).
- **C4 LOOPS: all 4 main image codecs OWN their SDR target loop (per-codec-ownership directive).**
  jxl `vardct/zensim_loop.rs` (+`JXL_ZENSIM_SECANT`, `1ed4ee72`); zenavif `target_quality.rs`
  (bracketed secant/bisection + `q0_head` zenpredict seed — EXEMPLAR); zenwebp `encoder/zensim_target.rs`
  (VERIFIED 2026-08-26: real one-pair secant + anchor seed + per-segment diffmap overrides, most advanced);
  **zenjpeg NEW `target_quality.rs` (`277b1efb` on origin/main) — `search_target` + `encode_with_target`,
  injected-scorer (zensim deps zenjpeg ⇒ cycle ⇒ MUST inject), 9 unit tests + a REAL-CODEC production
  gate (`79935f20`, tests/target_quality_real.rs, feat=zencodec: encode→decode→fast_ssim2, MEASURED
  6/6 target convergence, k2 1/6 k3 2/6 — a Zq seed would cut iters).** Dep-cycle finding
  + per-codec table: plan "CRITERION-4 STATUS". TODO: gainmap loop (HDR); zenpredict Zq autotune seed per
  codec; production gates (census/dial-mono/RD-under-independent-judge/perf); Program-D per-encoder
  λ-side steering. `zensim-target` (`7e17945e`) is now the shared-algo reference, not the owner.
- **C5 PERF (x86 SIMD): CLOSED.** Survey: jxl+zenrav1e already dual-arch; only zenavif `unpremultiply8`
  was NEON-only → AVX2 tier shipped (bit-identical + ~3.3-3.6×). zenavif `b92880e3`, zensim `9afa10f8`,
  `benchmarks/simd_x86_gap_survey_2026-08-25.md`.
- **C2 DATA: HDR corpus scoring LIVE (2026-08-26).** imazen-26 ID audit CLEAN (`78b60142`). HDR:
  98,805 hdrgrid encode blobs salvaged R2→local; the score jobs were ALREADY DECLARED on R2 (resume,
  not re-declare) — `hdrgrid-sf-gpu` (ssim2-gpu+iwssim-gpu, hdr:true) was 0/0, now RUNNING. First-cell
  GATE PASSED (693-pair chunk, 0 errors); scaled to **2 GPU boxes**: .27 (GTX 1060 6GB, r7900x≡lianli ssh-aliased) on **sf ssim2/iwssim** +
  **node-2/i134 (RTX 3070 8GB, .148) on sf2 butteraugli** — parallel, different metrics, no lease
  contention; sequencers via `lan_gpu_sequence.sh`. 3 CPU boxes (i265+r5900xt+tower) on `sf-cpu`
  features. sf/sf2-small DONE (r5900xt's 2GB DID do butteraugli-small — 687/687). NOTE: ssim2-gpu
  failed 15/453 HDR images on sf-huge (3.3% tail, investigate). Producing (GPU ~10 chunks + CPU 1263 in ~20 min). Recipe + resume:
  plan "HDR CAMPAIGN — EXECUTION LOG". TODO: drain → reassign 6GB boxes to `-huge` → `sf2`/butteraugli →
  `writeback_scores.py` → `_MANIFEST`+orientation gate+Tower mirror → HDR model wave; dHash+eye follow-up; curation.
- **C7 DOCS:** jxl comment + zenmetrics SeaweedFS doc-truth fixed; plan/survey/campaign records current.
- **C1 FLEET: 4 boxes BUSY (2026-08-26).** A1 enroll LAN fix landed; A3 cred distribution DONE (R2 creds
  pushed to r7900x/lianli/r5900xt/i265 — authorized). ssim2+butteraugli GPU-only (`ZEN_REQUIRE_GPU=1`, proven on the ONE GTX 1060 6GB .27; the
  GTX 1050 2GB is too small→CPU); cvvdp/zensim/features on CPU (i265). Drain/stall monitor armed
  (`/home/lilith/tmp/hdr-fleet-monitor.sh`). Note: the vast GPU-score launchers are cloud-only; LAN
  scoring uses the direct-manifest docker worker (recipe in plan). **C6 BROWSER:** located, not extended.
  **C8 ZENPICKER:** blocked on C4 autotune/gates.

Next: HDR GPU buckets drain → reassign to `-huge` + run `sf2`/butteraugli + harvest; then gainmap loop /
zenpredict Zq autotune. Repo gotchas: zenmetrics=`master`; verify pushes with `origin/<branch>` (git) not
`main@origin` (jj); after `bookmark set main -r @`+push verify `@-` (push auto-advances @ to an empty child).

## THE three entry points (a newcomer starts here)

1. **Evaluate any bake, correctly, with one command:**
   `bake_verdict --bake X.bin --regime 944` — resolves the ext944 features
   root, 944 dial/corruption grids, kadis-944 per-pair source, and the frozen
   12-corpus campaign list; a bare run cannot silently omit a corpus. Add
   `--fulleval out.json` for the schema-complete dashboard JSON, or run
   `scripts/run_full_eval.sh <bake> <name> 944` to also measure M3/M3a.
   (`scripts/sota944_verdict.sh` is the campaign's thin wrapper over the same
   preset.)
2. **See every model compared:** the summer-gauntlet board at
   `/mnt/v/output/zensim/reports/fulleval/summer_gauntlet.html` (rebuild:
   `scripts/v_next/bandwise_dashboard.py --fulleval-dir …/fulleval`; every
   regen must pass `scripts/v_next/gauntlet_gates.sh <html>`).
3. **Understand the science:** the campaign doc above (bar, arms, corrections,
   ensemble waves, addenda) + [`docs/TOP_MODELS_COOKBOOK.md`](docs/TOP_MODELS_COOKBOOK.md)
   (the 372-era roster + pitfall list — era banner at top).

## Reading order on resume

1. **This file** (~1 min)
2. **[`benchmarks/sota944_campaign_2026-08-03.md`](benchmarks/sota944_campaign_2026-08-03.md)** —
   the era authority: frozen bar, every wave's results, corrections, addenda.
3. [`docs/TOP_MODELS_COOKBOOK.md`](docs/TOP_MODELS_COOKBOOK.md) — validated
   science + exact reproduction of the (372-era) top models + the pitfall list.
4. [`docs/MODEL_SELECTION_SCORECARD.md`](docs/MODEL_SELECTION_SCORECARD.md) —
   the five-gate exam (RANK/DIAL/STEER/RD/TARGET) every ship candidate takes.
5. `CLAUDE.md` — rules + methodology (★924-parquets, ★E-M campaign, the
   NO-DUPLICATE-IMPLEMENTATIONS owner table, tool inventory).
6. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — how a number chains
   back to bytes; [`docs/DATA_SPLITS.md`](docs/DATA_SPLITS.md) +
   [`docs/DATASET_HISTORY.md`](docs/DATASET_HISTORY.md) — corpus law.
7. [`benchmarks/INDEX.md`](benchmarks/INDEX.md) → prior experiments; run
   `TaskList` for open work.

## Doc pointers (updated 2026-08-04)

- Campaign era: `benchmarks/sota944_campaign_2026-08-03.md` (authority) ·
  plan `docs/PLAN_SOTA944_CAMPAIGN_2026-08-01.md` · B lineage
  `benchmarks/profile_b_methodology_2026-07-12.md` · era bridge + backfill
  `benchmarks/backfill944_2026-08-01.md` + `backfill944_bigcodec_2026-08-02.md`.
- Shipped defaults: `ZensimProfile::B` (SDR) + BHdr; the 372-era candidates
  and their scorecards live in the cookbook; swaps remain user-gated.
- External reads / HDR domains: `scripts/external_reads/README.md`
  (seven-domain runner, `--from-stored` rescores in ~11 s).
- Eval-panel law: `docs/EVAL_PANEL_REQUIREMENT.md` (rank+dial two-panel
  mandate) · `docs/FULL_EVAL.md` (fulleval schema + 924/944-era eval slices).
- Historical: `docs/HISTORY-2026-05-v0x-era.md` (V0_x era) ·
  `benchmarks/best_per_day_summer_2026.md` (per-day 372-era champions) ·
  `zenanalyze/everything.md` is frozen-HISTORICAL (its own banner).

(CONTEXT-HANDOFF files are banned; durable facts live in the docs above. The
IQA literature corpus is `~/work/zen/zenpapers` — search it before designing
features or metrics.)
