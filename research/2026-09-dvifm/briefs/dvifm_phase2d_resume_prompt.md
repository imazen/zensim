# RESUME Phase 2d — with a mandatory fitting correction (supervisor, 2026-09-19 20:45Z)

You are resuming your own Phase-2d run in `/home/lilith/work/zen/zensim`. The original brief is
`~/tmp/devin/dvifm_phase2d_prompt.md` — it still binds you in full (read it again). Your committed work so far:
`a9861715` (prereg + CID22-49 A/B exposure ledger) and `bf8a2dbf` (native Y′CbCr input planes). Your outputs are
under `/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/` (28 block caches, pairs, tools/fit_standalone.py,
tools/metrics.py, surfaces/, logs/). Do NOT redo extraction. The supervisor stopped the run because the standalone
fit is measuring the wrong thing; fix it, amend the prereg, rerun.

## What is wrong (measured from your logs)
1. **The grid is scale-confounded.** `grid_sweep_level` scores each (C₀, β) cell with the output map
   `100·exp(−λ·E)` at the CURRENT λ — λ is not refit per cell. The target variance on `cid22a.tsv` is 172
   (sd 13.1); init MSE is 908 and the "best" grid cells are 775–894, i.e. 4.5–5× WORSE than predicting the mean.
   `tidkadid.tsv`: variance 751, init MSE 3,777. So the surface currently ranks cells by how well they rescale E to
   a fixed λ, not by how well they model masking. The C₀/β surfaces written so far are not interpretable.
2. **Every grid optimum sits on the grid edge** — C₀ = 0.3 (the upper bound) and β = 1.267–1.400 (the upper
   bound) on every level/variant logged (cid22a luma l0/l1, xyb-y l0/l1, native3 l0, tidkadid luma l0).
3. **`native3` (THE primary arm) crashed** with ZeroDivisionError in `fit_variant` (`fit_standalone.py:510`):
   the edge value β = 1.4 is a logit infinity in the bounded parameterisation. It was never rerun.

## Required fix (then an amendment, then rerun)
- **Per-cell map refit.** For every grid cell and every candidate the refinement evaluates, fit the output map
  in closed form / by 1-D search before scoring: map family `ŷ = A·exp(−λ·E) + B` with A > 0 (so the score stays
  monotone and identity is the top of the scale), A and B by linear least squares for a given λ, λ by golden-section
  on log λ (≤ 40 evaluations). Record, per cell, the refit MSE AND the scale-free rank criteria of −E vs y
  (SROCC and Kendall KROCC). Select by refit MSE; publish all three surfaces.
- **Sanity gate before any grid:** with the map refit, the init parameters must beat the constant predictor
  (MSE < var(y)) on each fit set; print both numbers. If not, stop and diagnose (units, orientation, row join)
  before fitting anything.
- **Widen the grid and the bounds together.** C₀ ∈ [1e-4, 3] log-spaced (≥16 points); β ∈ [0.05, 3.0] (≥16
  points); the parameterisation bounds must contain the grid with interior clamping (lo+ε, hi−ε) so no grid value
  is a logit infinity. If an optimum still lands on an edge, extend once more; if it still does, REPORT the edge
  optimum as the finding (e.g. "β wants > 3 / the knee wants to sit above every observed contrast = masking
  effectively off at this level") — do not silently clamp.
- **Target orientation per dataset.** Before pooling TID/KADID with CID22 or with each other, confirm each
  set's orientation and scale with `check_target_orientation.py` and map all to one 0–100 quality scale; record
  the mapping. `tidkadid.tsv` spans 5–97.5 while CID22-A spans 28–92 — say whether that is intended.
- Keep everything else from the brief: convex by construction (simplex level/channel weights, monotone, zero at
  identity — assert it), log-grid then ≥8 multi-start refinements per (plane, level), alternating with the head,
  loss surfaces committed as small data files. Report how sharp each optimum is (loss increase at ±1 grid step in
  each axis), whether neighbouring levels agree, and luma vs chroma knees.
- **Order:** native3 first (primary arm), then luma, then xyb-y, then the TID/KADID domain. Then the single sealed
  CID22-B read per the prereg, then Part C, then Part A.

## Prereg amendment (commit BEFORE rerunning any fit)
Append a dated "Amendment 1" section to `benchmarks/dvifm_screen2d_prereg_2026-09-19.md`: what changed (per-cell
map refit, widened grid/bounds, orientation mapping), why (the three measured defects above, with the numbers),
and that no decision-relevant result had been read (fit-domain only; CID22-B labels still sealed). Do not delete
the original text.

## Rules (unchanged)
jj only, small commits, DO NOT PUSH; one heavy job at a time through `~/work/zen/scripts/run-heavy --mem 16G
--jobs 8 -- … 2>&1 | tee ~/tmp/devin/<n>.log`; scratch only `~/tmp/devin/`; never delete caches/generated data
(rename to `.bak` if replacing surfaces); refresh `.workongoing` every ≤2 min (another session's marker may
overwrite yours — ignore it, per the user). Progress lines → `~/tmp/devin/dvifm2d_progress.log`; terminal file
`~/tmp/devin/dvifm_PHASE2D_DONE.md` or `~/tmp/devin/dvifm2d_BLOCKED.md`. Put the corrected knee/exponent tables and
the sharpness numbers at the TOP of the DONE file.
