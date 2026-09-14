# Complete local120 composition: registered derivative gate fails

September 14, 2026. **Synthetic only; no trained-model or shipping result.**
The combined known linear model passes only **1 of 24** public pixel-derivative
comparisons. All 120 consumed base features reconstruct correctly. The first
failure exhibits both edge branch changes and sensitivity of SSIM finite
differences to numerical precision. The prototype is archived and removed;
the planned frozen ensemble, real images, finite repairs and runtime
qualification were not run. No production arithmetic or API changes.

[Structured evidence](complete_gradient_2026-09-14.results.json) and the
[served packet](http://localhost:3300/zensim/reports/complete-gradient-2026-09-14/complete_gradient_2026-09-14.md)
preserve the failed gate and diagnosis. Source, binary and complete outputs
remain under `~/work/zensim-validation-2026-09-14/complete-gradient/`.

## Registered composition and stop

The [SSIM and color/scale prerequisites](ssim_adjoint_2026-09-14.md) passed
separately. This prototype combines all ten local basic features per channel
and scale: SSIM and artifact/detail p1/2/4 pools, plus MSE. It retains actual
canonical band moments, uses public pixel-feature roots, merges SSIM and edge
mean-derivative fields, and calls the existing reflected box transpose three
times per channel/scale. Direct edge and MSE terms complete each plane gradient.

The pullback includes all four 2× scales, dropped odd tails, reflected minimum
padding, zero stride padding, and the in-gamut linear-RGB color Jacobian.
Reference quantities are fixed, alpha is one, and pooled roots equal to zero
select zero coefficients. HF, hard maxima and other undeclared terms are not
represented; the proposed local120 ensemble is admitted only if every declared
input belongs to the supported set.

The first control is an unfitted Rust linear bake with known signed f32 weights
on all 120 local inputs. Analytic composition uses those exact weights; central
differences call public `BakeScorer::compute`. Four deterministic fixtures at
17×9, 65×97, 128×96 and 97×65 use whole/corner/interior perturbations and steps
.001/.0005. The preregistered tolerance is max(2e-5 absolute, 1% relative).

Only one comparison passes. Maximum derivative error is **.00810755**.
Maximum consumed-feature reconstruction error is **8.67e-19**, so this failure
does not come from a different base feature regime. Public score secants closely
match the known weights dotted with public feature secants; the head is linear.
The planned three-member local120/H128 ensemble remains **unrun**. Its looser
f32-score derivative tolerance was registered separately but never used to
rescue this failed prerequisite. No fitting or corpus/EVAL/TEST access occurred.

## First-failure diagnosis

On the 17×9 whole-field direction, predicted derivative is **+.00159854**.
Public pixel-score secants are **−.00650901** at step .001 and **−.000284985**
at step .0005. The base score is .02960814. These are synthetic-model
derivative units, not target-score qualification errors.

An independently accumulated basis-feature gradient reproduces the combined
prediction within 2.68e-7. The diagnosis evaluates all 120 feature contributions
on the same fixture, retaining the original failed steps and adding diagnostic
steps .01/.005/.0001. The added steps do not replace the registered gates.

The existing centered-f64 `stable_ssim_plane` reference evaluates SSIM on the
same perturbed canonical XYB pyramids. Its output remains f32, with f64 pooling
in this diagnostic. No reference values enter production scoring or the
predicted gradient, and no new SSIM value kernel is introduced.

| Step | Sum of absolute SSIM derivative discrepancies, canonical | Same, reference | Edge activation switches | Residual sign switches |
|---|---:|---:|---:|---:|
| .01 | .023985 | .024069 | 4,825 | 598 |
| .005 | .007878 | .006610 | 2,501 | 416 |
| .001 | .011606 | .000580 | 357 | 19 |
| .0005 | .034939 | .000998 | 167 | 8 |
| .0001 | .091667 | .002626 | 35 | 1 |

The discrepancy sums cover the 36 SSIM coordinates, unweighted. Branch counts
compare the plus/minus endpoints across all channels and scales, not distinct
original RGB pixels. At .0005, the maximum single-coordinate SSIM derivative
discrepancy falls from .0046073 to .00008361 using the reference. MSE contributions
remain small in this first-case decomposition.

This provides evidence of numerical sensitivity at small steps and branch
crossing at the tested perturbations. Nonlinear pooling can also affect a finite
secant. It does **not** prove the complete gradient correct, attribute every
error to one cause, or establish violation of the scalar SSIM accuracy contract.
The reference does not silently turn the failed public-API comparison into a
pass. An initial diagnosis edit made no source change; its unchanged replay is
preserved separately and not presented as a reference run. Original predictions
and public secants exactly match the later enriched diagnostic output.

## Consequence for the next experiment

Keep the failed stop and the existing component tests. The next correctness
check needs to distinguish mathematical pullback agreement, finite changes
across branch boundaries, and the resolution of f32 score differences. Compare
independent forward/reverse directional propagation before inferring a coding
defect from these finite differences. Explicitly measure branch margins and
retain actual public finite-repair scores as their own acceptance evidence.

Do not tune models or remove features to hide this diagnostic failure. Do not
infer a universal impossibility of useful gradients, or claim the unmeasured
ensemble/RD/performance stages passed. The complete production goal and all
shipping gates remain open.
