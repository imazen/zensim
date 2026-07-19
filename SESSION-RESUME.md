# SESSION-RESUME — read this first after every compact

**Last updated:** 2026-07-18 (five-gate scorecard era). Prior snapshots live in git history —
their durable facts are in the linked docs, not here.

## Reading order on resume

1. **This file** (~1 min)
2. **[`docs/TOP_MODELS_COOKBOOK.md`](docs/TOP_MODELS_COOKBOOK.md)** — the validated science +
   exact reproduction of every top model + the new-model loop + the pitfall list. THE entry
   point for model work.
3. **[`docs/MODEL_SELECTION_SCORECARD.md`](docs/MODEL_SELECTION_SCORECARD.md)** — the
   five-gate exam (RANK/DIAL/STEER/RD/TARGET), SDR + HDR, with the steer-mass pre-screen.
4. `CLAUDE.md` — rules + methodology (note: its V0_x-era status sections are historical;
   the cookbook supersedes them for current state).
5. `docs/DATA_SPLITS.md` + `docs/DATASET_HISTORY.md` — corpus law + poison ledger.
6. `benchmarks/INDEX.md` → the July-2026 docs when you need the evidence chain.
7. `TaskList` for open work.

## Current state (2026-07-18)

- **Shipped default = `ZensimProfile::B`** (linear-372 + dial spline; A deprecated behind
  `deprecated-profiles`). The A↔B history: B leads human-MOS holdouts; A led ssim2-agreement
  on codec sweeps (`benchmarks/ab_dial_monotonicity_2026-07-05.md`).
- **Ship candidates (swap user-gated, scorecard complete):**
  `Ebothg_scr0.5_dial` {CID22 0.879, nonphoto 0.906, **HF-NL 0.712** (best ever), LIVE 0.959,
  dial 0.985} and `winner_dial` {CID22 0.894, best jxl RD} — both weak on KonJND (0.27–0.34
  vs B 0.55). `ADD156` = exact-gradient/3.6KB runner-up. Verdicts:
  `benchmarks/sdr_scorecard_2026-07-18.md` + `screen_retrain_2026-07-18.md`.
- **The July-18 correction set** (older docs predate these — trust the cookbook):
  additive-mislabel (`additive_vs_mlp_correction`), M2=1.0-for-all + ModelSensitivity
  (`mlp_diffmap_coherence`), the `DiffmapResult::score()` legacy-V0_2 bug (fixed `834b4387`),
  zenjpeg inert Zq passes (worktree q-correction), RD probe results (`rd_probe_results`),
  LIVE-R2/CSIQ/PIPAL as first-class FR holdouts.
- **HDR:** shipped BHdr steer-mass 0.435; hdrmix-shaped lineage is a steering dead-end;
  steerable families exist (hdrbroadplh1 0.963, hdriwmix 0.762). Gate ≥0.5 before training
  (`benchmarks/hdr_steer_screen_2026-07-18.md`). PU-coherence + HDR-RD legs specced, unbuilt.
- **Feature-v2 ("perfectable features") campaign OPEN**: bound `hf_gain` at extraction, fix
  the IW `1/n`-vs-`Σw` divergence, spatializable-by-construction redesign — versioned opt-in
  regime, never a mutation of frozen v1 (all parquets/bakes depend on v1 byte-stability).
- **Literature**: search `~/work/zen/zenpapers` (+ `/mnt/v/input/papers/`) before designing
  features/metrics.
- **Worktrees live**: `jxl-encoder--zensim-diffmap-rd` + `zenjpeg--zensim-diffmap-rd` carry
  the RD-probe wiring (env-selectable profiles/maps, q-correction, probe binaries).

## Standing open items (beyond TaskList)

- Re-seed both codecs' legacy-V0_2 distance/starting-q tables against real scoring.
- KonJND/PJND lever for the Eboth family (data-mass tuning provably doesn't reach it).
- HDR G-RD + PU-coherence builds; then rank-vs-steer showdown for the steerable HDR families.
- Full-corpus RD phase before any probe-derived constant lands in source.
