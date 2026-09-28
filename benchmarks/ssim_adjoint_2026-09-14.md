# SSIM derivatives and a public-API color/scale control

September 14, 2026. **Synthetic correctness evidence; no shipping qualification.**
The Rev3 SSIM pixel derivative passes against the existing fused kernels, and
an independent MSE model passes through public pixel scoring with the color
and scale chain included. These are prerequisites for a shared response map;
they do not establish finite reference-repair accuracy or encoder utility.

[Structured results](ssim_adjoint_2026-09-14.results.json) accompany the
[served packet](http://localhost:3300/zensim/reports/ssim-adjoint-2026-09-14/ssim_adjoint_2026-09-14.md).
Registered protocol, source patches, failed harness logs, passing runs and
measured binary remain in `~/work/zensim-validation-2026-09-14/ssim-adjoint/`.
All implementation additions are test-only. No scoring formula, production
arithmetic, public API, model default, or shipping gate changes.

## Differentiate the existing Rev3 expression

The [preceding study](pixel_adjoint_2026-09-14.md) verified the reflected box
transpose and smooth edge derivatives. This study adds SSIM partials in the
existing `ssim_form` owner. It calls the existing scalar SSIM function for the
forward value and gates derivatives against its finite differences. It does
not introduce another SSIM scoring kernel.

For fixed reference mean `a`, distorted mean `m`, square-sum moment `s`, and
squared-error moment `e`, define `q=(m-a)²`, `L=min(q,1)`,
`E=max(e-q,0)`, and `V=max(s-a²-m²,0)+C2`. The existing Rev3 dissimilarity is
`L+(1-L)E/V`, with an outer nonnegative clamp. Differentiate each active branch
with respect to `m`, `s`, and `e`, keeping reference quantities fixed.

The actual retained production signals supply p-root chain factors for
p=1,2,4,8. If the resulting moment derivatives are `z_m`, `z_s`, and `z_e`,
then the distorted-plane gradient is:

`K^T(z_m) + 2d K^T(z_s) + 2(d-r) K^T(z_e)`.

`K^T` uses the previously tested weighted existing SIMD blur. Reference `r`
and distorted `d` here are samples on one XYB scale. This differentiates the
mathematical formula evaluated at retained f32 moments, not floating-point
rounding steps. Clamp kinks select an inactive-branch zero; smooth finite
differences do not claim differentiability at those kinks.

## Measured checks

| Check | Count | Result |
|---|---:|---|
| Scalar moment partials, active and clamped branches | 24 | Pass; worst error uses 0.493% of its tolerance |
| Complete SSIM plane directions through fused H/V kernels | 216 | Pass; max absolute derivative error 5.77838e-6 |
| Public pixel-score color/scale derivative comparisons | 24 | Pass; max absolute derivative error 6.04977e-7 |
| Public MSE feature reconstruction across four images | 48 | Pass; max feature error 6.85452e-12 |

These are derivative errors, not target-score accuracy measurements.
The registered derivative tolerance is the larger of 2e-5 absolute and 1%
relative; no tolerance changed after measurement. The feature-reconstruction
bar is 2e-6 absolute. SSIM plane tests use 8×8, 17×9 and 65×97, three contrast/
offset cases including saturated luminance, whole/corner/interior perturbations,
four p-root pools, and steps .001/.0005. Identity signals and selected derivatives
are exactly zero on all three planes. Scalar partial controls include synthetic
moment-level clamp branches, separate from image-derived cases.

The first revision-isolated run failed because the new test omitted the
completion marker required by `run_at_revision`. The parent correctly refused
to credit the child despite its zero exit. An initial text-edit attempt made
no change and repeated that failure. Adding the marker fixes the wrapper;
both wrapper execution and explicit Rev3 bodies pass. Initial logs remain
preserved. No mathematical change was needed to pass the registered tests.

## Independent public scoring control

A synthetic, unfitted ZNPR Rust bake has twelve nonzero, signed weights: one MSE
feature per channel and scale. `BakeScorer::compute` serves opaque linear-f32
sRGB images at 17×9, 65×97, 128×96 and 97×65. Values stay inside the display gamut;
alpha remains one and the source is not HDR.

The analytic directional chain uses the existing opsin constants and the
cube-root/opponent transform. It explicitly carries reflected minimum-size
padding and zero stride-padding derivatives, then calls the existing 2×
downscale kernel at each scale, including dropped odd tails. All twelve MSE
features per image reconstruct the public outputs. The complete model's
predicted derivative matches central differences of actual public pixel scores
for whole, corner and interior perturbations at both registered step sizes.

This isolates color/scale composition from SSIM conditioning. It does not
establish a complete SSIM-model gradient: the two controls remain separate.
No training data, corpus images, calibration, EVAL or TEST was accessed.

## Next step and limits

Assemble complete-model response using the verified SSIM, edge and color/scale
components, with explicit remaining MSE/HF/peak behavior and the existing model
sensitivity owner. Validate local derivatives and finite repairs separately
against the admitted TRAIN witnesses; include complete preparation/map/query
cost before advancing to native JXL interventions.

Hard-max ties, remaining feature families, gamut clipping/HDR, actual ensemble
behavior, finite-repair accuracy, runtime/memory and native RD are not qualified
by these tests. The full product goal remains active and no candidate is promoted.
