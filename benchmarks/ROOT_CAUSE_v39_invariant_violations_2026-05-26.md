# Root cause: how a metric that violates basic invariants shipped (2026-05-26)

## Symptom
`ZensimProfile::A` (V39, the shipped bake) returns scores **>100** and
ranks **heavier degradation as higher quality** on synthetic content.
`cross_platform::score_sanity_checks` caught one slice (mandelbrot blur)
but was outside the acceptance loop.

## Evidence (instrumented repro; table below)
128×128, score = `Zensim::new(prof).compute(src, dst).score()`:

| profile | content | identical | blur r=1 | blur r=5 |
|---|---|--:|--:|--:|
| **V0_2** (linear, no MLP) | mandelbrot | 100 | 49.8 | −28.3 |
| V0_2 | checkerboard | 100 | 7.9 | −153.6 |
| V0_2 | color_blocks (smooth) | 100 | 32.8 | −68.3 |
| **A / V39** (MLP+spline) | mandelbrot | 100 | **124.7** | **158.9** |
| A / V39 | checkerboard | 100 | 95.8 | **165.1** |
| A / V39 | color_blocks (smooth) | 100 | 68.6 | **131.0** |
| A / V39 | value_noise | 100 | **−107.5** | **111.0** |

V0_2 is monotone (more blur → lower score) and bounded ≤100 on every
case. V39 inverts (blur5 > blur1 > identical) and exceeds 100 on every
case, smooth content included — so this is NOT an out-of-distribution
fractal corner case; it's a general property of the bake.

## Localization
- **Feature extraction is fine** — V0_2 consumes the same features and
  stays monotone.
- **The spline is monotone by construction** (fit with strict-increasing
  knots), so it cannot reorder. Yet score(blur5) > score(blur1) >
  score(identical). Therefore the **trained MLP's own output is
  inverted**: network(blur5) > network(blur1) > network(identical) on
  this content.
- **`extrapolate_score: true`** (metric.rs:2113) then lets the spline's
  linear extrapolation flow through unclamped, turning the inverted
  network output into >100 / <0.

Two distinct defects: (1) the MLP ranks degradation backwards
off-manifold; (2) the output is unbounded. Clamping only hides (2) — a
clamped V39 still ranks blur5 ≥ blur1 ≥ identical, pinned at 100.

## THE FUNDAMENTAL FLAW (why it shipped)

A linear weighted-distance metric (V0_1/V0_2) is **monotone and bounded
by construction**: more distortion → larger feature distances → larger
weighted sum → lower score, and the `100−A·d^B` map is ≤100. Moving to
an **unconstrained MLP** discarded those structural guarantees — and the
pipeline never replaced them with a check. Specifically:

1. **The optimization + acceptance signal is SROCC** (rank correlation)
   on natural-photo human-MOS corpora (CID22/KADID/TID/AIC/KonJND).
   SROCC is **rank-only and scale-invariant** — it is mathematically
   incapable of detecting out-of-[0,100] output, self-identity
   violations, or monotonicity inversions on any content. A bake can
   maximize the acceptance metric while being a broken function of its
   inputs.
2. **Validation content is narrow** — natural photos with realistic
   codec artifacts only. No synthetic content (screen/line-art/noise —
   which CLAUDE.md explicitly says matter for web) and no
   controlled-degradation monotonicity sweeps (apply N levels of one
   distortion, assert ordering). The MLP's off-manifold behavior was
   never exercised, so its inversion went unseen.
3. **The one test that encoded invariants** (`score_sanity_checks`:
   blur < identical) was (a) only on mandelbrot, (b) NOT part of the
   bake-acceptance gate (`bake_verdict` never runs it), and (c)
   silently red since V39 shipped because the dev loop ran
   `cargo test --lib` (it's an integration test under `tests/`).

Root enabling decision: **adopt a flexible MLP, validate it solely by
rank-correlation on a narrow content distribution, and add no
invariant/axiom gate.** SROCC has no incentive to preserve the metric
axioms, and nothing else checked them.

## Fix direction (NOT clamping)
1. **Add an invariant gate to bake acceptance** — for every bake, across
   synthetic + natural content: assert output ∈ [0,100]; self-identity
   is the max; monotonic under a controlled degradation ladder (blur,
   noise, quantization at increasing strength). Fail the bake on
   violation. This is the missing structural guard; it would have
   blocked V39. (A new `tests/metric_invariants.rs` should encode this gate.)
2. **Investigate / constrain the MLP's monotonicity** — why does the
   network invert off-manifold? Candidates: standardizer extrapolation
   on out-of-training-range features, the per-sample-α head, or simply
   an unconstrained surface. Options: monotonicity-regularized training,
   a degradation-ladder loss term, or architectural monotonicity
   constraints. A quality metric should be monotone by design or proven
   monotone by test.
3. **Wire the invariant gate into CI + `bake_verdict`** so SROCC is
   never again the sole acceptance signal (consistent with the existing
   "SROCC-only verdicts BANNED" rule — extended from stats panels to
   structural invariants).

## Mechanism confirmed (2026-05-26) — see deep-dive doc
Quantified in `docs/METRIC_INVARIANTS_MECHANISM_AND_REDESIGN_2026-05-26.md`.
Three compounding factors, all measured:
1. **Off-manifold inputs:** synthetic+blur drives features to **up to
   106–135σ** from the training mean (rms 12–18σ across all 372);
   identical content stays at max|z|=3.8. The MLP is evaluated 100σ
   outside anything it trained on.
2. **Unconstrained MLP inverts there** — LeakyReLU net has no global
   shape constraint; off-manifold it ranks more-blur as higher-pinned
   (provable: the spline is monotone, so the observed score inversion
   must originate pre-spline). Linear V0_2 on the same features stays
   monotone — defect is in the net, not the features.
3. **~1000× extrapolating spline:** 3 knots in a 0.033-wide input window,
   endpoint derivative ≈1024 score/unit, **linear (unbounded)
   extrapolation**. The whole [0,100] dial is manufactured from a
   ~0.13-wide post-pin band — narrower than the net's off-manifold
   excursion — so off-distribution the spline runs entirely in its
   unbounded regime.

## Status
Investigation + redesign analysis complete; shipped V39 untouched per the
user's accept decision. No clamp applied (would mask defect 1). The
redesign doc specifies principled detection (metamorphic tests, OOD
z-score/Mahalanobis flag, static gain analysis, IBP/MILP certification)
and a guarantee-by-construction architecture (non-negative dissimilarity
features → partial-monotone net → bounded squash → clamped-extrapolation
calibration). Lowest-risk first increment: the invariant gate
(`tests/metric_invariants.rs` + `bake_verdict --invariants`).
