# V46 / V46b — monotone-cbc CHARACTERIZED PARETO LIMIT (naive approach)

Two end-to-end training runs with `--monotone-cbc` enabled, isolating
the constraint cost from the recipe-choice cost.

## Bake panel

| Corpus | V46 (1-layer) | V46b (V32-faithful 2-layer) | V39 (shipped) |
|---|---:|---:|---:|
| CID22 SROCC | 0.7688 | 0.8447 | 0.8793 |
| KADID SROCC | 0.7569 | 0.8053 | 0.9251 |
| TID SROCC | 0.7943 | 0.7924 | 0.9317 |
| KonJND SROCC | 0.4253 | **0.4918** | 0.4197 |
| AIC-3 SROCC | 0.6886 | 0.7598 | 0.8023 |
| AIC-4 SROCC | 0.7718 | 0.8799 | 0.9051 |
| G1 dial p5..p95 | −4..70 (range 74) | **−21..−7 (range 14)** | −90..97 (range 187) |
| G7 (≥0.85) | FAIL | SOFT (0.89 score) | PASS |
| Goals weighted | 0.256 | 0.247 | 0.714 |

## What V46b reveals (the load-bearing finding)

V46b matches V32's hyperparams exactly + adds `--monotone-cbc` →
isolates the constraint's pure cost. V46b's SROCC delta vs V39 is
SMALL (CID22 −0.035, AIC-4 −0.025; KonJND WINS by +0.072). On
rank-based metrics the constraint is cheap.

**But V46b's dial range collapses to 14 score-units** (p5=−21, p95=−7).
The bake is unusable as a codec-target metric — a user typing
"score=70" cannot find any encode that maps to it, because the bake's
output range is [−21, −7].

## Why the dial collapses (structural)

With `encoder w1 ≥ 0`, `rank_w ≤ 0`, `α ≡ 1`:
- `h = LeakyReLU(w1·x + b1) ≥ 0` (encoder weights non-negative ⇒
  hidden activations non-negative for any non-negative input).
- `y_pre = rank_w · h + rank_b ≤ rank_b` (rank_w ≤ 0, h ≥ 0 ⇒ their
  product ≤ 0, so y_pre is upper-bounded by the bias).
- After tanh pin `100·σ(y_pre/scale)`, max output is `100·σ(rank_b/30)`.
  For `rank_b ≈ 30`, that caps at 73. For max=99 we'd need
  `rank_b ≈ 150`, but the MSE loss against [0,100] targets pulls
  rank_b down to natural-baseline scale (~30).

## Deeper reason: feature semantics

Natural images have `h > 0` from non-distortion features (shape,
brightness, edges, peaks). The naive "all encoder ≥ 0" interprets
every feature as distortion, so even the reference image has
`h > 0` ⇒ `y_pre < rank_b`. The dial loses its upper anchor.

The constraint is too coarse: it assumes every feature is a
"distortion magnitude" feature, but the 372-feature set mixes
distortion features (SSIM-derived, masked SSIM, peaks) with
structural features (color statistics, shape descriptors).

## What to fix (paths forward, NOT in this session)

1. **Per-feature sign mask** (preferred): identify the K distortion
   features (SSIM family); apply encoder-≥0 ONLY to those rows of w1.
   Leave the structural-feature rows unconstrained. Same correctness
   guarantee on the constrained subset; full expressivity on the rest.
   Pre-requisite: documented per-feature semantic labels (which
   features ARE monotone in distortion).

2. **Feature centering**: in `compute_scaler_from_groups`, subtract
   the per-feature MEDIAN computed on a "natural baseline" subset
   (reference images, q=100 encodes). The encoder then sees
   `h ≈ 0` at baseline and `h > 0` only when input deviates from
   baseline. This decouples h's magnitude from feature shape.

3. **Learned shape offset head**: add an unconstrained
   `y_shape = w_shape · x + b_shape` term and compose
   `y = y_rank_monotone + y_shape`. y_shape gives the dial its
   upper anchor; y_rank_monotone gives the structural guarantee
   on the distortion-monotone subset. Monotonicity holds as long
   as |y_shape|'s gradient is dominated by y_rank_monotone's in
   the distortion features.

## Decision

V46 / V46b DO NOT ship. V39 stays at `ZensimProfile::A` with its
existing known-limits (off-manifold inversion on synthetic OOD
content, characterized in
`zensim/tests/metric_invariants.rs::v39_known_limit_violations`).

The `--monotone-cbc` machinery (commits fa5c699 + bf92de5) remains
in the trainer for the next iteration, paired with one of the
three fixes above. The synthetic-data correctness test
(`zensim-validate/tests/monotone_cbc_projection.rs`) still passes
and proves the projection mechanism works as designed — the
issue is the constraint's semantic match to the feature set,
not the projection itself.

## Bakes (kept on disk for future analysis)

- `/mnt/v/output/zensim/bakes/v46_monotone_cbc_real_recipe_seed17_2026-05-26.bin` (261 KB)
- `/mnt/v/output/zensim/bakes/v46b_v32faithful_monotone_2026-05-26.bin` (257 KB)
- Verdicts: same paths, `.verdict.md` extension.
