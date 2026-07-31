# PLAN #69 — Closed-loop steering value: controller × redistribution study

Pre-registration (2026-07-29, per ITERATION_PROTOCOL discipline). Runs happen
only after these gates are frozen. Substrate: jxl-encoder `f195c8c0` (C3b
harness: arms/targets/stats/probe) + zensim `7de38ab0`+ (fused compare).

## The question

C3b measured that a PROVEN-coherent steering map (M3a 0.85-0.99 vs ΔS) does
NOT improve loop target-hitting — and on v47A the no-map baseline beat both
model maps (med |err| 0.244 vs fold 0.594 vs attr 0.807). The probe showed
the tile signal is strong and the fused scalar tracks decode better than the
fold arm's. So the binding constraint is the LOOP: how per-tile values become
qf adjustments (allocation → clamp-at-0 → normalization/blend → redistribution
→ damped controller). #69 asks: **which loop-design change lets map fidelity
convert into target-hitting?**

## Pre-registered hypotheses (each = one arm; all share the C3b harness)

- **H1 signed redistribution.** The clamp-at-0 + sign-blind L4 tile fold was
  designed for the fold's unsigned map. The attribution map is SIGNED and
  sum-preserving; clamping discards exactly the "refining HERE loses score"
  information. Arm: redistribution that respects signed mass (negative tiles
  push qf coarser proportionally, not via clamp-to-clean).
- **H2 controller separation.** t=75 cells were overshoot-dominated; the
  damped controller and the map fight over the global level. Arm: controller
  owns the MEAN qf step (scalar error only); the map steers only the
  ZERO-SUM residual across tiles.
- **H3 magnitude steering.** C3b normalized tile ratios; attribution
  magnitudes are score-units per tile. Arm: per-tile step sizes proportional
  to query_rect magnitude (capped), not rank/ratio-normalized fields.
- **H0 (control).** baseline / fold / attr exactly as C3b, unchanged.

## Gates (frozen now)

- **G1 (value):** an H-arm beats BASELINE (not just fold) on median
  |achieved−target| at equal iteration budget on ≥2 of 3 targets, decoded-
  judged, per bake. C3b's lesson: beating the fold is not the bar — the
  no-map controller is.
- **G2 (no-regression):** bytes at equal achieved score within +2% of
  baseline's (steering must not buy accuracy with size).
- **G3 (breadth):** n ≥ 27 cells/arm/bake (widen fixtures beyond
  city/dog/girl@576 — add ≥6 more refs incl. nonphoto), targets {70, 80, 88}
  (avoid the t=92 clamp saturation C3b measured; t=75→70 to reduce the
  overshoot-dominated cells).
- **G4 (perf pricing):** if any H-arm passes G1+G2, run its attr-stale
  variant (C3b proved staleness ≈ free) to price the single-pass ≤1.1×
  endpoint before any C-phase perf work restarts.
- Failure to pass G1 in all arms = the honest conclusion "per-tile steering
  adds no target-hitting value over a good scalar controller in this loop" —
  recorded, program pivots to the diffmap's OTHER consumer (zenjpeg-class
  per-block-probe codecs, where region-incremental queries are the contract).

## Budget + placement

One fixture-widening pass + 4 arms × 2 bakes × 27 cells ≈ a few hours of
576² encodes on this box (nice'd) — no fleet needed. All work in the
jxl-encoder diffmap-rd workspace (the C3b substrate); zensim stays read-only
unless H3 needs a query helper (additive only).

## Post-study register — task #70 productization (added 2026-07-31)

#69 RAN and concluded (jxl `d17cf7ce`; verdict `cc6e9575`; results doc:
jxl-encoder `benchmarks/zensim_attr_loop69_2026-07-29.md` + 3 TSVs). **H3
magnitude steering is the one loop rule with value** — v47A t70 med |err|
0.306 vs baseline 1.867 at bytes ratio 0.990; the only arm winning nonphoto
on both bakes; staleness free (G4). The ratio-normalized family (C3b attr /
H1 / H2) never beats the plain damped controller; on shippedB-linear ALL
arms fail. Follow-up work is task #70; the spec is recorded here so it
survives independent of any session task store:

1. **`ZENSIM_H3_GAIN` sweep** — #69 ran only the registered default 10.0.
   Sweep {2.5, 5, 10, 20, 40} on the #69 matrix. Pre-register the selection
   gate BEFORE running: best all-median |achieved−target| subject to G2
   (bytes at equal achieved within +2% of baseline).
2. **Single-pass ≤1.1× perf endpoint for H3** — staleness proven free twice
   (C3b + #69 G4), so implement the stale-scalar single-pass fused compare
   (C3a's ranked lever 3) and re-run the H3 arm at the final perf state;
   loop-quality gates must hold unchanged.
3. **The shippedB story, honestly** — the linear bake gains nothing from any
   steering arm. If H3 ships, it ships for MLP-class dials only, documented.
4. **924-class loop steering (longer term)** — needs the extractor-side
   retention hooks (C3a deviation 2; the extractor session's domain).

Substrate: jxl-encoder diffmap-rd harness (`d17cf7ce`) + the zensim
fused-compare API.
