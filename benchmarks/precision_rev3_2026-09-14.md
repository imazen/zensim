# Rev3 precision versus finite-neighborhood response — September 14, 2026

The [family diagnosis](spatial_diagnosis_2026-09-14.md) found large spatial
prediction errors but did not establish their numerical cause. Extend the
existing test-only precision instrument to an explicitly declared formula
revision and its matching luminance form. Legacy specs without a revision
still require revision 1. Mismatched revision/form refuses before pixel access.
No new SSIM kernel, production feature formula, model, API or score path.

The existing centered f64 reference now also returns the local means it
already computes. The same diagnostic uses them for independent f64 edge
ratios and mean/L4/L2 pooling on the exact canonical f32 XYB pyramid. All
served SSIM and six basic edge slots, and their weighted repair contributions,
reconstruct against the immutable Rust BakeScorer records at the existing
absolute 1e-12 tolerance. Edge reconstruction is revision 3-only, preserving
the historical detail-root era distinction. Reference values never replace
features in a served model.

## Preregistered probes and results

First select the largest absolute SSIM-family prediction-error block for
basic228/H32 on report row2445, screenshot row5324 and photographic control
row396. Then, before follow-up measurements, add the two largest absolute
overall disabled-map-error blocks for the report and screenshot. These are
five blocks on three already admitted TRAIN development pairs. No new source
admission, fitting, calibration, EVAL or TEST access.

| Row / bounds | Served SSIM linear contribution | f64-reference SSIM | Served edge linear contribution | f64-reference edge |
|---|---:|---:|---:|---:|
| 2445 / [0, 16, 16, 32] | +2.0889429 | +2.0887096 | +3.1281399 | +3.1281638 |
| 5324 / [0, 8, 8, 16] | +2.5512803 | +2.5511836 | +0.2552152 | +0.2552057 |
| 396 / [96, 224, 128, 256] | +0.6082575 | +0.6082312 | +0.1091789 | +0.1091704 |
| 2445 / [0, 48, 16, 64] | +1.3858708 | +1.3856658 | +1.2752458 | +1.2752551 |
| 5324 / [24, 0, 32, 8] | +1.8889096 | +1.8888435 | +6.9579130 | +6.9580026 |

The largest absolute discrepancy is .000234 score points for SSIM and .000090
for the basic edge family. These are weighted feature changes under the frozen
base sensitivities, not predictions from a newly trained or corrected model.
On the actual worst screenshot block, SSIM and edge arithmetic corrections
are −.000066 and +.000090 points, against an observed score increase +6.9272
and disabled-map prediction −4.6206. This does not support numerical precision
as the explanation for that large failure. The finite response of neighborhoods
and feature interactions remains the consequential problem.

The direct reference isolates arithmetic on the existing pyramid. It does not
validate the perceptual suitability of the feature formula, the RGB-to-XYB
conversion, sampling geometry, hard-max/L8/HF response, or model nonlinearity.
Its evidence must not be generalized beyond the checked SSIM/basic-edge terms.

## The synthetic locality tolerance does not generalize

Every base SSIM signal stays within the existing synthetic absolute accuracy
bound 1e-3. But all five probes exceed the existing synthetic outside-support
movement tolerance 2e-5. Preserve the failures and their magnitudes:

| Row / bounds | Maximum base-signal error | Maximum outside-support movement | Changed outside-support signals |
|---|---:|---:|---:|
| 2445 / [0, 16, 16, 32] | 0.000160999 | 9.44436e-05 | 12177 |
| 5324 / [0, 8, 8, 16] | 0.000112773 | 2.26274e-05 | 6637 |
| 396 / [96, 224, 128, 256] | 0.000332177 | 0.000115544 | 5304 |
| 2445 / [0, 48, 16, 64] | 0.000160999 | 3.42429e-05 | 11662 |
| 5324 / [24, 0, 32, 8] | 0.000112773 | 4.24013e-05 | 4855 |

The changed-signal count includes any nonzero rounding difference; it is not a
count of material visual errors. The direct centered reference has exactly
zero movement outside the mathematical support. Independent centered and raw
f64 algebra agree within 3.04e-12. The existing exact f64 kernel passes its
2e-10 +2e-6 relative signal/pool and locality comparisons to that reference.
The original synthetic Rev3 locality and accuracy tests still pass. That is
weaker coverage than a real-image guarantee; do not quote their tolerance as
universally established. No threshold was loosened and no existing result
was overwritten to hide the distinction.

The arithmetic issue and the large repair-map failure need separate treatment.
Record stronger real-image locality coverage, but do not repeat a slow SSIM
kernel replacement as a proposed cure for these order-one map errors.
For these cases, the measured SSIM/basic-edge arithmetic contribution is tiny.

## Next implementation question

A reference-pixel repair changes the blurred distorted mean in every overlapping
window, not only the point signal owned by the rectangle. Frozen-signal removal
also cannot predict newly created extrema. The existing map representation
approximates these finite changes, and bins1 versus8 already gave unchanged
ranks on these aligned rectangles. The next useful prototype should account
for the finite neighborhood response using the existing retained extraction
owners, then evaluate the complete model on predicted feature changes through
its Rust surface. Inspect existing local recomputation/cache work before adding
another implementation. Register error and cost limits, keep the original
six-model/23-pair comparison, and require actual runtime benefit. More capacity,
arbitrary coefficient suppression or full-resolution map storage is not an
evidence-based next step for these failures.

## Verification and artifacts

Three initial SSIM probes, their three edge extensions and the five final
witness probes finish. Every existing SSIM numerical report field is unchanged
under the edge extension, and the first three final witnesses match the prior
edge report exactly. The existing direct-window/all-form analytic test and
Rev3 retained accuracy/locality controls pass. Wrong revision, wrong luminance
form and missing-revision-at-Rev3 controls refuse before nonexistent pixel
paths; the legacy revision 1 spec remains accepted. Final Clippy, formatting
and script lint results are retained. An intermediate compile caught one
remaining consumer of the extended test-reference tuple; its destructuring
was updated without changing arithmetic, and the final build passes.

Artifacts: `~/work/zensim-validation-2026-09-14/precision-rev3/` contains immutable
protocols/specs, source and binary hashes, old and final reports, build/refusal
logs and [compact results](precision_rev3_2026-09-14.results.json). The served
[previous A/B failure comparison](http://localhost:3300/zensim/reports/spatial-diagnosis-2026-09-14/gallery/index.html)
remains unchanged. These are diagnostic references, not new EVAL rows or new
model scores. No candidate qualifies. The full production goal remains active.
