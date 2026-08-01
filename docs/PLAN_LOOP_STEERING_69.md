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

### Loop-efficiency study (2026-07-31) — reshapes items 1–2 above

Pre-registered characterization, jxl-encoder `c57e634c..7c36807f`
(`benchmarks/zensim_diffmap_efficiency_2026-07-31.md` + 3 TSVs; supervisor
re-derived 5 endpoint families from the TSVs, 5/5 exact). Findings that
change #70's design space:

- **The loop is budget-limited, not tolerance-limited**: at the stock 6-iter
  budget, τ=0.25 is unreachable in 20/27 cells (baseline) / 17/27 (h3);
  doubling to 12 iters cuts the median floor 3.6–7× (baseline 0.955→0.268,
  h3-mag 0.649→**0.090**). Mechanism: nonphoto seeds land ~20 points high
  and the ≤1.35× controller clamp needs 10+ iterations to close that. The
  gain sweep (item 1) should co-sweep budget/clamp or it will measure the
  clamp, not the gain.
- **The loop emits the LAST iterate, not the best**: h3's judged error
  worsens past its sweet spot (0.59@k6 → 1.02@k12) because overshoot doesn't
  self-correct. Best-so-far emission (track the iterate whose internal score
  is closest to target; emit that) is a cheap, likely-large win — new
  sub-item for #70.
- **Early stop at looser τ does NOT save bytes on v47A** (+0.5% to +3.9%,
  landing above target — the loop approaches from above); only the linear
  bake saves. Don't sell tolerance-stopping as a byte saver.
- **The internal fused score is a trustworthy stop signal on photos**
  (judged-vs-internal med ±0.13 at every budget; nonphoto worst ±1.5) —
  further de-risks the stale/single-pass endpoint (item 2).
- **Bytes-targeting works as an outer loop** (transplanted damped controller
  on a new `JXL_ZENSIM_QF_GLOBAL_SCALE`): median 4 full encodes to within
  5%, 6 to 2%; 1% often unreachable in 8 (17/27) with deliberately-untuned
  damping. Quality at fixed size is path-independent (med +0.21 vs the
  quality-run at equal bytes) — size dialing doesn't cost quality.
- Cost context: ~36 ms/compare baseline, ~47.5 ms h3 + one-time ~196 ms
  model gradient at iteration 0 (576²).

### #70 status (2026-08-01) — items 1-2 + the emission sub-item landed

- **Item 1 (gain sweep): DONE** by the metric-matrix study (jxl
  `0eb31edc`, `benchmarks/zensim_loop_metric_matrix_2026-07-31.md`):
  pre-registered gate winner = **gain 20, clamp 1.35** (F1 tie 13/27 at
  k3; bytes tie-break 0.976; none disqualified; the clamp axis is dead at
  k3, +1 cell at k6).
- **Best-so-far emission sub-item: SHIPPED** (`JXL_ZENSIM_EMIT_BEST=1`,
  jxl `14f67aec` + results `4888b56a`,
  `benchmarks/zensim_emit_best_2026-07-31.md`): at k6 the within-2 census
  is unchanged exactly as trace-priced (base 18/27, h3 21/27) with
  medians improved (h3 0.926→0.745); at k12 the diagnosed overshoot
  regime is CURED — h3 med |err| 0.747→**0.382**, census 22→**25/27**
  (base 0.750→0.432, 23→25/27); bytes neutral-to-saving (med ratio
  1.0000 k6 / 0.992-0.993 k12). Extended budgets now help instead of
  hurting; h3 k12+emit-best is the best arm measured in the series.
- **Item 2 (stale-scalar single-pass): SHIPPED, additive** — zensim
  `326185e9` (`AttributionSession` +
  `Zensim::compute_with_ref_score_and_attribution_stale`; the combine
  folds IN-STRIP with the previous compare's coefficient packs — no
  retention, no second sweep) + jxl `1d3866e4`
  (`JXL_ZENSIM_SINGLEPASS=1`, default off, R0 sha `12cf08e0` preserved).
  Exactness, not tolerance: score bit-identical on every call; map
  BITWISE-equal to the fresh combine given matching packs (same-pair
  test + a planes(B)×coeffs(A) reference construction).
  - **Perf endpoint (the ≤1.1× bar): NOT met — measured floor 1.84×
    @576² / 1.40× @1152²** (marginal stale map 3.4 / 13.8 ms vs fold
    marginal 1.9 / 9.9 ms; the FRESH fused path measured 6.29× / 2.86×
    on the same interleaved quiet-box runs, load ~2.4 — so the lever cut
    the marginal 3.4× / 2.0×). Decomposition: the in-strip fold itself
    is fold-marginal-class (~0.5-0.7 ms @576²; the C1 ordering problem
    is solved); the floor sits in the per-scale **window SPREAD**
    (`box_spread_sum_preserving_f32`: 2.2-2.7 ms @576², 7.5-7.8 @1152²)
    + SAT/trim (0.5-0.6 / 1.7-1.9 ms) + the forced sd-plane store.
    Ranked remaining levers: (1) parallel spread — H pass is bitwise-safe
    row-parallel, V pass needs per-band running-sum re-init (f32-noise
    class; both fused paths shift together, stale==fresh exactness
    preserved) — est. lands ~at-bar @576² and PASSES @1152²; (2)
    in-strip clipped-window scatter of the window slots (kills the
    spread pass entirely; changes summation order → needs its own
    tolerance gate); (3) lazy/band-parallel SAT build.
  - **Loop-quality gate: staleness free, FOURTH consecutive
    measurement** (h3 g20 c1.35, k6, 27 cells, decoded-judged):
    all-cells med |err| 0.926 → **0.801** (improved 0.125; per-cell
    delta med −0.002, IQR [−0.092, +0.026]; only 2/27 cells move >0.5,
    both improvements, both nonphoto; worst regression +0.33 sc_gui
    t88); per-target Δ t70 +0.008 / t80 +0.143 / t88 0.000; within-2
    21/27 → 20/27. Loop cost −11 % (med ms/compare 66.2 → 58.9). The
    fresh arm reproduced the emit-best study's h3 row exactly (zensim
    `326185e9` is byte-transparent to the deployed fresh loop).
- **Item 3 (shippedB story)**: unchanged — no steering arm helps the
  linear bake; H3-class steering ships for MLP-class dials only.
- **Item 4 (924-class loop steering)**: unchanged — blocked on
  extractor-side retention hooks (that session's domain).
