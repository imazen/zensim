# Complete local120 derivatives agree; finite repair prediction fails

September 14, 2026. Synthetic diagnostics only; no model qualifies.
Independent forward and reverse propagation agree in all 12 combined and 120
feature-basis controls. On the frozen local120/H128 ensemble they also agree,
but a gradient contraction predicts the wrong sign for 6 of 8 partial repairs.
This closes the proposed fixed-gradient finite-repair shortcut for these cases.
It does not establish that gradients cannot guide smaller native interventions.

[Structured results](gradient_duality_2026-09-14.results.json) and the
[served packet](http://localhost:3300/zensim/reports/gradient-duality-2026-09-14/gradient_duality_2026-09-14.md)
retain source, registration, execution and failures. The measured binary is
retained locally under `~/work/zensim-validation-2026-09-14/gradient-duality/`.

## Contract and controls

This is a separately registered diagnostic, following the
[failed public-gradient gate](complete_gradient_2026-09-14.md). That original
1/24 result remains failed. Algebraic agreement does not replace it.
The forward path propagates RGB directions through the color Jacobian,
reflection and padding, all four scales, blurred moment tangents, verified
SSIM partials, active edge branches and MSE. It does not use reverse gradient
fields or the transpose helper. Both paths intentionally share primal kernels
and the separately verified scalar partials. Thus this is independent chain
propagation, not two independently implemented feature definitions.

The fixed-branch convention chooses zero at inactive edge kinks. One exact-zero
edge occurs at 65x97; none in the other three geometries. No exact-zero residual
occurs. Agreement under that convention does not establish unique derivatives
at kinks or derivatives of floating-point rounding.

The signed linear control passes 12/12 combined comparisons (maximum difference
9.834e-8) and 120/120 first-fixture feature-basis comparisons (1.279e-7).
The registered algebra tolerance is max(2e-5, .001 times the larger magnitude).
Only after these checks passed were the three previously frozen model members
loaded, with hashes and declared local120 input IDs verified. Their weights are
uniform one-third; the public BakeScorer owns complete-head sensitivities.
No model parameter, calibration, dataset or gate changed.

## Frozen model and actual public repairs

`PT914U_local120_h128_plain_ens3` passes all 12 algebra comparisons, maximum
difference 2.432e-5 within the registered relative tolerance. Only 5/24 actual
public score secants at steps .001/.0005 meet the older diagnostic tolerance;
maximum discrepancy is 4.504. Consumed base feature reconstruction differs by
at most 8.674e-19.

Finite repairs replace a rectangle with its reference pixels and execute the
complete public pixel scorer. The prediction contracts the base RGB gradient
with that actual reference-minus-distorted direction. Each repair is unique;
the two epsilon records do not double its sample count.

| Geometry | Rectangle | Actual score gain | Gradient prediction |
|---|---|---:|---:|
| 17x9 | corner 4x4 | -28.583 | +21.036 |
| 17x9 | middle third | -21.005 | +4.106 |
| 65x97 | corner 4x4 | -1.474 | -.264 |
| 65x97 | middle third | -40.544 | +.284 |
| 128x96 | corner 4x4 | -1.472 | +.136 |
| 128x96 | middle third | -47.390 | +1.626 |
| 97x65 | corner 4x4 | -3.390 | +.259 |
| 97x65 | middle third | -52.672 | -.024 |

All eight partial repairs decrease this model's score; six predictions have
the opposite sign. Maximum absolute gain error is 52.648 points. These are
synthetic model-score responses, without human or independent quality judges;
a score decrease is not proof of perceptual regression. The four whole-image
repairs reach the public identity override of 100 and are recorded separately.

## Decision

Keep the accurate finite-response references and these failing witnesses.
Do not use one fixed gradient as a qualified finite-repair estimator or repeat
algebra checks as a substitute for resolving its measured product failure.
The accurate shared-moment method still fails the dense runtime budget; useful
native spatial allocation also remains unqualified. Smaller native interventions
would need their own explicit semantics, witnesses and unchanged product gates.

The experimental implementation is archived and removed; all previously retained
component tests and production arithmetic remain unchanged. One initial Rust
JSON assignment compilation failure is retained; it preceded the science runs.
Clippy passes on the restored tree. No corpus, fitting, EVAL, TEST, native RD,
independent quality judgment or performance qualification ran in this packet.
