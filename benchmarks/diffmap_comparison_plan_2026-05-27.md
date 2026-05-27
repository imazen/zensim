# Plan: which diffmap drives jxl-encoder best? (#38, 2026-05-27)

User goal: an **excellent diffmap for jxl-encoder**. jxl-encoder's zensim loop
(`vardct/zensim_loop.rs`) already produces a per-pixel zensim diffmap
(`compute_with_ref_and_diffmap_linear_planar`) and uses it for sum-preserving
per-tile quant redistribution (`compute_tile_dist` → tiles with above-average
diffmap error get more bits). The question: is there a *better* diffmap than
the current v47-ssim2 (#32) one?

## Candidates

1. **#32 — v47-strict-QAT (ssim2-derived target)** — the shipped `Profile::A`,
   currently driving the loop. Baseline.
2. **#37 — v47-CVVDP (cvvdp-target)** — same arch/recipe, trained on
   `cvvdp_log_norm` instead of ssim2-derived human_score (training now).
   Swappable in the loop by pointing the scorer at the cvvdp bake. CVVDP is a
   full CSF-based visual-difference model, so its per-pixel view may localize
   perceptual error better than ssim2's.
3. **#33 — structural activity-relative signal** — the multi-scale,
   activity-relative per-tile excess (corruption-detection spike). NOT a
   trained-MLP diffmap; would need a per-region map injected into the loop's
   redistribution. Bigger lift; evaluate after 1 vs 2.

The diffmap is fundamentally the metric's per-pixel error view (SSIM + optional
edge/HF/MSE, weighted by trained weights, optional contrast mask + sqrt — see
`zensim/src/diffmap.rs` `DiffmapOptions`). So #32 vs #37 is a clean swap (same
loop, different bake); #33 is a new construction.

## Methodology — the CIRCULARITY guard (load-bearing)

**You cannot judge a zensim-driven encode by zensim** — driving the loop with
diffmap X and scoring the output with metric X is rigged (the encoder
optimizes exactly what you measure). Every diffmap-vs-diffmap RD comparison
MUST be judged by an **independent** panel:

- Loop driven by **zensim-v47-ssim2 (#32)** → judge by butteraugli + cvvdp +
  dssim + (real) ssim2 (all independent of the v47 MLP).
- Loop driven by **zensim-v47-cvvdp (#37)** → judge by butteraugli + ssim2 +
  dssim, and **NOT** cvvdp (that would be circular).
- The fair cross-comparison metric is one independent of BOTH: **butteraugli
  3-norm** and **dssim** are independent of both ssim2-MLP and cvvdp-MLP.
- Report the full RD curve (bytes vs each independent metric) at matched
  bitrates, on a multi-content corpus (photo + screen + the q5–q60 low band,
  per CLAUDE.md), not a single quality point.

A diffmap is "better" if, at equal bytes, the output scores better on the
INDEPENDENT panel (especially butteraugli + dssim) — i.e. it spent bits where
they reduced *genuinely* perceptible error, not just its own metric's error.

## Steps

1. **Baseline RD** (#32 diffmap): encode a multi-content corpus across the
   distance band with the loop ON (zensim-v47), persist bytes + decoded
   output. Score each output with the independent panel (zen-metrics:
   butteraugli, ssim2, dssim, cvvdp). Build RD curves. (Persist encodes +
   diffmaps per the ML-data-pipeline rule — encodes are expensive.)
2. **CVVDP-diffmap RD** (#37): swap the loop's scorer to the cvvdp bake
   (one-line, gated behind an env/feature for the experiment — do NOT change
   the shipped `Profile::A` pin), re-encode the same corpus, score with the
   independent panel (excl. cvvdp). Compare RD vs baseline.
3. **Verdict #32 vs #37**: which diffmap gives better butteraugli/dssim RD at
   equal bytes? If cvvdp wins decisively + the cvvdp bake's panel (bake_verdict)
   is acceptable, it's the better diffmap → consider shipping it as the loop's
   metric (and/or as a sibling profile). Honest report per the V41 caveat
   (cvvdp-target may rank worse on CID22 but localize better for encoding).
4. **#33 as a diffmap** (if 1–3 motivate it): build a per-region
   activity-relative map, inject as an alternative `tile_dist`, repeat the RD
   comparison. The structural signal is corruption-tuned; its value for
   *honest* RD redistribution is unproven — measure, don't assume.
5. Retune the loop's `ZENSIM_DISTANCE_TARGETS` calibration for whichever
   diffmap wins (re-seed per `examples/zensim_calibration_seed.rs`).

## Notes / risks

- The loop's redistribution is RELATIVE + sum-preserving, so the diffmap's
  *spatial shape* (where error concentrates) matters more than its absolute
  scale. A diffmap that's flat (no spatial discrimination) gives no RD benefit.
- Compute cost: each RD point is a full loop encode (butteraugli-loop is
  ~0.2–2.5 s/image). A 20-image × 5-distance × 2-diffmap grid = 200 encodes —
  feasible locally; persist everything.
- Keep the shipped `Profile::A` pinned to v47-ssim2 during the experiment;
  the cvvdp-diffmap swap is experiment-only until the RD verdict is in.
