# SESSION-RESUME — read this first after every compact

**Last updated: 2026-08-04 (consolidation wave).** The current era is the
**SOTA-944 model campaign** — pre-registered, five seed/lever waves + two
ensemble waves, all appended in place in the one authority doc:
[`benchmarks/sota944_campaign_2026-08-03.md`](benchmarks/sota944_campaign_2026-08-03.md).
Everything before it (372-era, 720/924-era) is historical context, era-tagged —
never compare numbers across eras without the doc's era-bridge notes.

## Current true state (2026-08-04)

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
  (campaign doc AMENDMENT 8: floors, composite, classes, and the full
  pass/frontier results; §1 stays the freeze bar). Headline: 0/145 cells pass
  all 8 floors — classic-IQA breadth (CSIQ/LIVE ≥ 0.83) is the 944 era's
  binding balance axis, and wave-7's arm H is the first cell family to cross
  it while holding KonJND.

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
