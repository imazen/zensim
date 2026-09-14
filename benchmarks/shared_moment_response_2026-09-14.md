# Shared window moments: response accuracy passes, dense cost fails

September 14, 2026. **TRAIN development only; no candidate qualifies.**
Sharing retained window moments preserves accurate finite repair predictions,
but the tested dense8 implementation still costs 28.1 scalar calls for
basic228/H32 on a 1024² image. The advancement budget is three calls. The
prototype is archived and removed; production feature arithmetic is unchanged.

[Structured results](shared_moment_response_2026-09-14.results.json) and the
[served evidence packet](http://localhost:3300/zensim/reports/shared-moment-response-2026-09-14/shared_moment_response_2026-09-14.md)
contain registrations, source patches, predictions, controls and timings.
Complete artifacts and measured binaries remain in
`~/work/zensim-validation-2026-09-14/moment-response/`.

## Method and admission

The preceding [tile study](tile_response_2026-09-14.md) recomputed blurred
features for every affected tile. Here, retain the four actual canonical
V-blurred moments per scale/channel: reference mean, distorted mean,
mean square sum, and mean squared error. Reuse the verified row-local working
pyramid for each reference-rectangle repair.

For reference `a`, old distorted sample `d`, new sample `d'`, and `δ=d'-d`,
the three raw moment changes are `δ`, `δ(d'+d)`, and `δ(d'+d-2a)`.
Small f64 summed-area tables cover only the changed bounding rectangle.
Explicit reflected-window intersections and multiplicities give moment
updates, which are added to retained base moments and rounded to f32.

The existing fused feature kernel pools prepared old/new moments with radius
zero and a flattened single row. No new SSIM expression is implemented.
Local pooled-sum differences update canonical whole-plane sums; existing
finalizers retain the original scale pixel counts. New local maxima combine
with retained maxima outside the affected rectangle. Complete frozen ensembles
execute through public `BakeScorer::score_features_with_identity`.

This is an approximate response instrument, not a public spatial API or a new
feature era. Moment rounding and reduction grouping differ from full replay.
The source bound is expanded by the existing blur radius before repooling.

The registration reuses exactly six frozen models, 23 admitted TRAIN development
pairs and 1,626 rectangles. Source/family admission, original and decoded bytes,
pixels, model bytes, and public-pixel witness hashes are verified. Predictions
precede comparison with repaired-score witnesses. No fit, calibration, EVAL,
TEST or terminal access occurs. Seven unavailable original nominations remain
unavailable. Narrow models still use full228 extraction in this instrument.

## Correctness failure, diagnosis and repair

The first synthetic 17×9 corner repair failed: predicted score 8.569971 versus
public pixel score 8.493832, an error of .076139 above the registered .05 bar.
The largest feature errors were maximum slots. The new experimental helper
passed `(x,x+1),(y,y+1)` to `MaxRemoval::add`, whose existing contract uses
inclusive source-coordinate extrema. This incorrectly retained repaired pixels
touching the query's right or bottom boundary.

An independent direct-exclusion test reproduced the defect: a 5×4 signal plane
queried outside `[0,0,3,2]` returned 22 instead of 21. Correcting singleton
footprints to `(x,x),(y,y)` passes all 315 rectangles, including empty and full
queries. The production owner already used the correct convention; it did not
need an arithmetic fix. A comment now makes its inclusive contract explicit.

With that helper corrected, the unchanged synthetic public-score gate passes
across six geometries, empty/identity/border/seam repairs, odd dimensions and
reflected padding. Working pyramids and restoration match full RGB replay.
Retained base features equal canonical extraction. Reflected sums match direct
121-sample enumeration, and f64 moment identities pass independent algebra
checks. These tests use the same measured binary as the real matrix. Wrong-role
and wrong-revision controls reject before pixel access.

Initial failed logs and source patch are preserved. Repaired-state oracle
substitutions proposed during diagnosis were unnecessary once the endpoint
defect was demonstrated and the original gates passed; they were not run.

## Fixed accuracy matrix

All 138 model/image checks pass the registered rank ≥ .99, maximum error ≤ .05,
p95 error ≤ .01, and zero material wrong-sign gates. Every base score is exact.
The observed maximum below .01 also bounds p95; this is not a p95 estimate.

| Frozen ensemble | Minimum signed rank | Maximum gain error, points |
|---|---:|---:|
| basic228/H128 | .999820 | .003697 |
| basic228/H32 | .999899 | .002588 |
| y60/H32 | .999812 | .000635 |
| local120/H128 | .999555 | .001349 |
| basic156/H128 | .999555 | .001012 |
| basic192l8/H128 | .999891 | .001429 |

These 9,756 predictions concern ideal reference replacement. They do not
establish native encoder utility, better human-quality agreement, or qualification
of the existing production maps. Their prior failures remain recorded.

## Complete cost, including the fine grid

The fixed matrix finishes in 2.52 seconds. Its basic228 sweeps cost 6.32–20.00
scalar calls in initial feasibility timing. None passes the three-call budget.
Setup includes retained moments/extrema, the canonical base, working storage,
and bookkeeping; query totals include pyramid changes, moment tables, signal
pooling, complete inference, restoration, and destruction of query buffers.

After accuracy passed, the already registered dense8 cost stage reused the exact
binary on TRAIN row4589, JXL q5, at 1024²: 16,384 eight-pixel rectangles and six
frozen ensembles. The existing JXL owner defaults to bin8 and 8-aligned queries;
this finest-grid workload does not assert the image's native transform partition.

| Component | Time |
|---|---:|
| Shared setup | 150.60 ms |
| Moment updates and feature pooling | 1,216.39 ms |
| Pyramid updates | 66.79 ms |
| Restoration | 26.56 ms |
| H32 complete ensemble inference | 86.17 ms |
| H128 complete ensemble inference | 191.95 ms |

Complete basic228/H32 cost is **1.557 seconds / 55.431 ms = 28.09 scalar calls**;
H128 costs **1.663 seconds / 55.369 ms = 30.03 calls**. Totals retain shared
diagnostic overhead and subtract the other models' separately timed inference.
The earlier tile prototype took 3.144 seconds for H32 on this workload, but these
separate runs are not a controlled paired speedup measurement.

Dense8 accuracy is **unmeasured**: the saved large-image witnesses use block64.
No native encoding or RD check ran here. Single-run feasibility timings are
not latency p95 or a controlled memory qualification. No sparse extraction
speedup is claimed for the narrow models.

## Decision and remaining work

Reject this implementation as the dense steering path. Retaining base moments
removes repeated blur work but still repools overlapping neighborhoods and runs
a complete nonlinear head per query. Accurate finite responses now have both
exact and approximate reference implementations; another capacity or tile-size
sweep does not address this measured work requirement.

The next consequential question is whether shared pixel-response computation
can preserve useful spatial direction and finite-response fidelity within the
budget. Compare any such method against the preserved public-pixel witnesses
and then actual native interventions, without weakening gates. Keep exact
finite response as diagnostic evidence; do not silently substitute a coarse
grid or frozen-gradient surrogate for its claim.

Scalar/human-quality assessment, calibrated attainable targeting, corruption,
native spatial RD, HDR scope, p95/memory and frozen qualification remain required.
No candidate, default, public API or advertised capability is promoted.
