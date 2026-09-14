# Pixel-response adjoints: a corrected premise, not a qualified map

September 14, 2026. **Synthetic correctness evidence only. No model qualifies.**
The historical zero-mass argument does not rule out a pixel gradient contracted
with a repair direction. The existing SIMD blur can implement the reflected
transpose through boundary weighting, without a new blur or SSIM kernel.
Three retained Rust tests establish the prerequisite for investigating a shared
response map. No production arithmetic or public API changes.

[Structured evidence](pixel_adjoint_2026-09-14.results.json) accompanies the
[served packet](http://localhost:3300/zensim/reports/pixel-adjoint-2026-09-14/pixel_adjoint_2026-09-14.md).
Registration, logs and measured source/binary are retained in
`~/work/zensim-validation-2026-09-14/adjoint-response/`.

## Why revisit this question

The [shared-moment experiment](shared_moment_response_2026-09-14.md) established
accurate finite reference-repair predictions, but its dense8 cost remains 28.1
scalar calls for basic228/H32. Recomputing overlapping neighborhoods per query
does not meet the three-call budget. Sharing the derivative computation across
queries is a distinct hypothesis.

The [July 29 C2b record](attribution_map_c1_2026-07-29.md) and the August 5 Claude
campaign memory rejected a 50/50 pixel/window mass allocation on eight measured
cells. They also dismissed the adjoint of residual signals because its columns
sum to zero, and treated the observed gap as a general first-order floor.
The measured spread failures stand. The broader mathematical dismissal does not:
an unweighted derivative is different from its product with a repair direction.

## Boundary-correct transpose

Let `K` be the radius-five, reflect-101 box operator. It is not symmetric under
the ordinary pixel inner product at boundaries. Let `D` be diagonal, with weight
one in the interior, one-half on an edge, and one-quarter at a corner. Detailed
balance gives `D K = K^T D`, hence:

`K^T z = D K (D^-1 z)`.

The test implementation scales the input, calls the existing SIMD
`box_blur_1pass_into`, then scales the output. Independent f64 gather/scatter
enumerates every reflected window sample. Signed fields and corner impulses
cover 8×8, 17×9 and 65×97 planes. Radius five stays within their dimensions;
no claim covers the owner's different clamping behavior for oversized radii.

Maximum forward/transpose vector error is **2.08370e-7**, and maximum normalized
adjoint pairing error is **2.32494e-7**, both below the preregistered 2e-5 bars.
Corner controls reject using plain `K` as its transpose in all three geometries.
This is mathematical-operator agreement within f32 rounding, not differentiation
of discrete floating-point rounding steps or a cross-hardware benchmark.

## Zero derivative sum does not imply zero repair response

For residual energy `F(d) = ||(I-K)d||²`, the pixel gradient is
`g = 2(I-K)^T(I-K)d`. Since `K 1 = 1`, `sum(g)=0`: adding a constant to the
whole distorted plane leaves its residual unchanged. But a repair toward zero
has direction `-d`, and `g·(-d) = -2F(d)`, generally nonzero.

| Plane | Energy F | Gradient sum | Repair-direction derivative |
|---|---:|---:|---:|
| 8×8 | 11.512299 | −5.47e-7 | −23.024599 |
| 17×9 | 51.987883 | 4.84e-8 | −103.975765 |
| 65×97 | 2056.909348 | 2.38e-5 | −4113.818696 |

All normalized identities pass the registered 2e-5 tolerance. Crucially, the
actual finite change from replacing the entire plane by zero is **−F**, not
−2F. The test explicitly distinguishes a correct derivative from an exact
finite-removal predictor. It neither weakens nor satisfies the model's finite
response gates.

## Canonical edge-feature directional checks

For reference `a` and distorted `d`, use the existing artifact/detail signal
`max(±((1+|d-Kd|)/(1+|a-Ka|)-1),0)` with p-root pools for p=1,2,4,8.
Differentiate the smooth active branch and apply `(I-K)^T` to its signal
derivative. Reference quantities are fixed. Both artifact and detail branches
are checked using varying-amplitude checkerboards and scaled distortions,
with a positive margin from the absolute-value and activation kinks.

Finite differences evaluate the existing SIMD blur and production
`fused_vblur_features_ssim` signal/pooling kernel on prepared means. The
independent f64 pooled root first agrees with that kernel within 2e-6. Whole
field, corner and interior perturbations at steps .001 and .0005 produce
**144 directional checks**, all passing the unchanged tolerance of 1% relative
or 2e-5 absolute. Maximum absolute directional error is **1.36695e-5**.
An earlier reference-pooling-only pass is preserved separately; the reported
final checks use the production pooling owner.

These are plane-level synthetic kernel checks, not model evaluation through
the public pixel API. They do not cover activation kinks, hard-max ties,
SSIM derivatives, complete HDR/color/resampling chains, model heads, or native
codec perturbations. No corpus, training, calibration, EVAL or TEST was accessed.

## Decision

Keep the tests and correct the current attribution documentation's general
dismissal. Do not repeat the failed 50/50 support-allocation experiment or claim
that a true derivative recovers exact finite removal. No serving default changes.

The next implementation must connect the needed feature derivatives and
color/scale chain to the complete model through the existing Rust owners,
then compare local directions and finite reference repairs separately on the
admitted TRAIN witnesses. Measure complete cost before advancing to native
JXL interventions. Scalar assessment, target bounds/seeds, corruption, HDR,
native RD and p95/memory qualification remain open; all shipping gates stand.
