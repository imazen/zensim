# Finite band response: exact repairs, excessive dense-map cost

September 14, 2026. **TRAIN development mechanism evidence; no model qualifies.**
The test-only prototype reproduces every saved public-API repair gain, but its
dense rectangle sweep is too expensive to advance. Production source is restored
exactly to `fcc86e39`; no public API, model, scalar or map behavior changes.

[Structured results](band_response_2026-09-14.results.json) accompany this record.
The served [artifact packet](http://localhost:3300/zensim/reports/band-response-2026-09-14/band_response_2026-09-14.md)
contains the protocol, registration, predictions, canonical rank results,
hash-bound witnesses, source patch, measured source and controls. Complete local
artifacts, including the measured binary, are in
`~/work/zensim-validation-2026-09-14/band-response/`.

## Question and preregistered procedure

The [precision study](precision_rev3_2026-09-14.md) found SSIM/basic-edge numerical
errors much smaller than the observed finite-repair prediction errors. July's
[C2b experiment](attribution_map_c1_2026-07-29.md#c2b--bleed-allocation-measured-negative--perf-floor-2026-07-29)
had already rejected pure `I-K` mass allocation and a 50/50 residual spread.
Reusing the actual nonlinear feature owners is a different hypothesis.

The prototype lives only in `feature_v2`'s existing test module. It caches the
canonical `V1BasicSums` for each 32-row band, channel and scale. Each query copies
the base distorted RGB image, replaces the requested rectangle with reference
pixels, and constructs its four-level pyramid through `PrecomputedReference`.
It compares full raw rows against the base pyramid, recomputes every band whose
input halo contains a changed row using `fold_v1_one_band`, and merges old/new
sums in canonical band order. Existing finalizers produce all 228 basic/peak
features, and public `BakeScorer::score_features_with_identity` runs the complete
frozen ensemble. This eliminates the frozen-signal removal approximation and
the frozen feature-gradient approximation together.

This is an executable finite-reference-repair method, not an oracle substitution:
only base reference/distorted pixels, rectangle geometry and model bytes enter
prediction. Saved repaired scores are opened by a separate comparison step.
It is nevertheless an internal extraction prototype, **not a public pixel or
spatial API**, and a reference repair is not a native codec quality step.

The predeclared accuracy bar is maximum base/repaired score error `1e-5` and
signed rank at least `.99` in every nondegenerate case. Dense feasibility requires
a complete sweep within three uncached scalar calls before advancement. The
initial timings cannot establish p95 or a sparse-query speedup. No gate changed.

## Admission and accuracy

Reuse exactly the [23 admitted TRAIN pairs](spatial_coverage_2026-09-14.md) and
[six frozen ensembles](spatial_diagnosis_2026-09-14.md), including all original
rectangles and the seven original unavailable nomination cells. Admission,
parent registrations, model bytes, original/decoded file hashes and exact RGB
pixel hashes are verified before execution. No fit, calibration, EVAL, TEST or
terminal access occurs. No new dataset or feature era is created.

The 1,626 distinct RGB repairs produce **9,756 complete ensemble repair gains**.
All 138 model/image checks have rank **1.0**, zero base-score error, and exactly
zero error in score gain versus prior full public-pixel rescoring. Reconstructing
the absolute repaired score from the old base plus delta differs by at most
`4.44e-16` from the saved decomposition's floating-point addition.

For basic228/H32, the report at row 2445 moves from the old map's `.583982` rank
to `1.0`; screenshot row 5324 moves from `.519512` to `1.0`. These results support
the finite-response mechanism, not a new scalar-quality or native-RD claim.
They do not separately identify the benefit of feature response versus complete
head inference. Existing production maps and their 24 failed checks are unchanged.

The exact measured binary also passes 4,560 synthetic public-feature comparisons
at five odd, sub64 and multistrip geometries. Empty, corner, opposite-border,
band-seam and full-identity edits exercise invalidation and pixel identity.
Reused-band and fully rebuilt-band features are exactly equal; public features
agree within `1e-12`, and served scores agree exactly. Wrong-role and wrong-revision
controls refuse before pixel access. An initial synthetic bake used comma-separated
IDs instead of the owner's whitespace format; it was correctly refused and fixed.

## Cost and disposition

One serial process finishes the full matrix in 7.67 seconds. Query measurements
include RGB copying, full pyramid construction, changed-row scanning, replay,
finalization and complete ensemble inference. Buffer destruction and diagnostic
bookkeeping remain in total time. The shared extraction computes all 228 features
even for narrower models; their timings are not optimized subset measurements.

| Model | Accuracy checks | Dense sweep / scalar, range | Band work alone / scalar, range |
|---|---:|---:|---:|
| basic228/H128 | 23/23 | 7.33–57.32× | 5.20–35.86× |
| basic228/H32 | 23/23 | 7.22–57.46× | 5.25–35.97× |
| y60/H32 | 23/23 | 11.14–133.97× | 8.22–83.89× |
| local120/H128 | 23/23 | 7.67–62.76× | 5.54–36.97× |
| basic156/H128 | 23/23 | 7.09–57.82× | 5.09–36.18× |
| basic192l8/H128 | 23/23 | 7.11–57.16× | 5.06–35.76× |

These are **initial feasibility ratios**, using one scalar control per model/image,
not controlled performance qualification or a p95 claim. Every sweep exceeds
the advancement bar, including its band component alone. Eliminating full-pyramid
work alone would therefore not close the measured gap. No sparse-query benchmark
or sparse API is claimed. The prototype is archived as source, patch and executable;
it is removed from the working implementation to avoid permanent research cruft.

## Consequence for the next experiment

Do not repeat capacity, family-removal, numerical-precision or support-spreading
sweeps to solve this same failure. Complete finite feature response plus full
inference has now matched the observed repairs. The unresolved problem is obtaining
that response cheaply enough for a dense map.

Investigate bounded spatial recomputation through the existing kernel/retention
owners: interior signal removal plus recomputation of the changed blur boundary,
including newly created extrema and nonlinear pool finalization. Preserve global
feature normalization, canonical reference geometry and a measured error budget
against this exact implementation. First establish the work/memory bound and
mathematical support; do not assume cropped-image scoring is equivalent or that
f32 recurrence differences disappear outside the support. Register any approximation
and its accuracy/runtime bars before observing results. Scalar assessment,
corruption, bounded targeting, native RD, HDR and frozen qualification remain open.
