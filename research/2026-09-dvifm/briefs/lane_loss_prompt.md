# LANE `loss` — make the constants fit identifiable, and fit them on SafeSyn

Read `~/tmp/devin/LANE_PREAMBLE.md` first; it binds you. Then `docs/PLAN_DVIFM_VERDICT_2026-09-20.md` §6 and §7 —
this lane IS those two sections. Outputs `/mnt/v/output/zensim/dvifm-loss-2026-09-20/` (≤20 GB).
Reuse `/mnt/v/output/zensim/joint-core-v1/tools/` and `/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/tools/
fit_standalone.py` (the convex fitter) — extend, do not rewrite from scratch.

The 2026-09-19 fits are uninformative about the masking exponent: one grid step changes the loss by <0.3% at 13 of
15 cells, and phase-2's "psychovisual" β≈0.65 was just its own prior. Fix the fitting, in this order, each change
measured separately so we learn which one mattered:

1. **Within-reference ranking loss.** Replace pooled MSE as the OBJECTIVE with a pairwise loss over same-reference
   pairs (hinge on score difference vs label difference, or a differentiable Spearman surrogate — state which and
   why). Keep pooled MSE and the refit map as reported diagnostics only. Report, per (plane, level), the new
   one-step sharpness for C₀ and β: this is the test of whether the objective was the problem.
2. **Tied parameters first.** Fit ONE β shared across all levels and Cb tied to Cr; then untie level-by-level only
   where the development leg improves by more than seed noise (nested comparison, report each step).
3. **Prior restored.** Fit free AND prior-pulled (β→0.65, λ chosen on the development leg, never on the fit loss);
   report both. Ship the prior-pulled one when the development leg cannot separate them.
4. **Weber-like contrast axis.** Add a contrast definition `C = band amplitude / (local mean + ε)` (state ε and the
   local-mean window; reuse the existing box-mean machinery) as an alternative to today's globally-normalised band
   amplitude range. Refit. This is the only variant whose exponent is comparable to published masking slopes, so
   report its β against Legge & Foley 0.62 / Watson 0.7 explicitly.
5. **Domains: SafeSyn alone, SafeSyn-majority, and the two human domains.** SafeSyn =
   `/var/tmp/zensim-validation-2026-09-15/recovery/tables/safesyn_{fit,development}.parquet` (READ-ONLY; 141,054 /
   38,758 rows; **signed** targets to −744 — never clip; if a step cannot take negatives, drop those rows and
   report the count). Build (C̃,m) histograms per (plane,level) for the grid stage — the joint-core lane measured
   them equal to the per-block grid sum to 4e-16 — and capped f16 per-block records only for gradients.
   Then report the constants for CID22-A, TID/KADID and SafeSyn side by side with profile intervals, and mark a
   constant **portable only where all three intervals overlap**.

Record in the §2 schema (value, interval, one-step sharpness, mode gate|off|curve, prior, agreeing domains,
detectors tripped). Deliverable: `benchmarks/dvifm_constants_2026-09-20.{md,json}` + a `constants-v1.json` spec
that the i16 kernel lane can consume, plus a one-paragraph answer to: *do we now have an identifiable masking
exponent, and is it psychovisually plausible?* Terminal file `~/tmp/devin/LANE_LOSS_DONE.md`, progress
`~/tmp/devin/lane_loss.log`.
