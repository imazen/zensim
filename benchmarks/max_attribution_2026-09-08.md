# Candidate max-removal rectangle estimates

Registered after `88d2716f`, before implementation or new image results.
The prior independent reference proved that maxima outside a rectangle can
be queried from four prefix/suffix projections. Unlike an additive density,
this handles ties and the next maximum exactly under frozen-signal removal.

Public delta on `ScoredAttribution`: add `refinement_gain(x0, y0, x1, y1)`
and `unsupported_refinement_feature_ids()`. The current caller is the
complete-candidate rectangle audit in `extract_features_372col`; the recovered
JXL variable-tile controller follows after intervention qualification.
Keep `attribution()` and `unsupported_feature_ids()` as the existing density
and density-coverage contract. No public signatures or scalar arithmetic change.

For max terms, remove a pyramid sample only if its entire logical source
footprint is inside the requested rectangle. Footprints follow floor-halved
2x2 downsampling, reflect-101 minimum-image extension, and ownership of SIMD
padding by the nearest extended-image edge. Project signal maxima onto the
minimum/maximum source coordinates of those footprints. Four prefix/suffix
lookups give the exact outside maximum, with O(width+height) retained space.
The query adds `-s * (global_max - outside_max)` to the existing density sum.
Full-image queries remove every max sample; partial coarse footprints remain.

This explicitly freezes the extracted signals. It does not model the actual
blur response to replacing pixels, max emergence elsewhere, nonlinear score
curvature or gate/clamp crossing. Those require finite pixel and codec checks.
Keep an ID unsupported if its retained maximum fails bit-for-bit parity with
the complete served feature. Density coverage remains unchanged.

Gates: exhaustive synthetic rectangles/ties and independently expanded
source footprints; all 36 max features through the public Rust surface on
every SIMD tier; empty/inverted/clipped, padded/tiny/odd images and all scales;
exact existing density/scalar/feature replay on 264 training pairs for the
fixed blend and F/D endpoints; explicit new query coverage; meaningful
core/serving/HDR tests, Clippy, lint and API snapshots. Only then measure
actual pixel interventions, recording failures separately from mechanism.
No holdouts, new model fits or codec-gain claims belong to this mechanism gate.

Preregistration, fixed inputs and subsequent evidence:
`/mnt/v/output/zensim/max-attribution-2026-09-08/`.

## Mechanism and serving result

The new Rust surface passes 14,268 source-footprint rectangle checks,
including explicit coordinate-coded pixels through the real reflection owner,
odd dimensions, all scales and padding. Ties remain non-additive. A one-bit
retained-feature mismatch leaves coverage missing. All 36 max and 36 L8 terms
pass through the public candidate surface on ten SIMD configurations.

The 264-pair blend/F-mean/D-endpoint replay produces 792 exact old feature,
scalar, pixel and density records. Each arm evaluates 275,176 new rectangle
queries; every new refinement coverage list is empty. Full-image max
reconstruction has maximum absolute error 3.56e-14 for the blend. Its max
correction is nonzero on every nonidentity full-image query and on
7,255/6,397/4,801/2,641/967 grid queries at sizes 8/16/32/64/128. D has no
max correction. Density-only coverage intentionally still reports max IDs.
No scalar model, feature arithmetic or score scale changes.

## Actual pixel screen — improvement, still failed

Extend the existing `diffmap_block_coherence` owner to load complete ensembles
and emit hashed, per-block JSON. Its seven refusal controls and single-bake
versus one-member-ensemble control pass. The first build caught a digest-format
trait mismatch; the corrected build and its evidence are retained separately.

Use all eight registered FIT origins, the nearest available distances to
0.5/2/8 (0.5, 1.6168174743652344, 7.731237411499023), and all blocks at
8/16/32/64 pixels. These 96 cells perform 25,248 actual reference-block pixel
replacements. All input/model hashes and base scores match the prior canonical
audit; every refined scalar runs through the complete `BakeScorer::ensemble`.

| Block | Mean density M3a | Mean finite query M3f | Mean M2 |
|---|---:|---:|---:|
| 8 | 0.69748 | 0.71774 | 0.99498 |
| 16 | 0.75974 | 0.80074 | 0.99402 |
| 32 | 0.83283 | 0.91043 | 0.99456 |
| 64 | 0.85393 | 0.94980 | 0.98189 |

Mean noninferiority passes at every size, but the screen remains **FAIL**:
M2 >=0.99 on 80/96 cells; M3f >=0.70 on 81/96. The 15 spatial failures are
nine screenshot, five document and one graphics cells, concentrated at small
blocks and distance 0.5. Individual regressions remain visible; averages do
not hide them. M2 is a local-linearization diagnostic, not a mathematical
upper bound: the finite query can outperform it on particular cells.
Keep its registered bar unchanged. No native codec/RD gate is passed here.

## Saved-data diagnosis identifies the next correction

The same Rust coherence owner now accepts `--refinement-analysis INPUT --json
NEW_OUTPUT`, reusing its statistics and refusing malformed/count-inconsistent
or numerically inconsistent reports. All 96 saved-statistic replays and four
malformed-input controls pass; no pixel/model computation is repeated.

Replace one predicted component at a time with its measured, sensitivity-
weighted feature delta. Using observed max changes closes **0/15** spatial
failures. Using observed non-max changes closes **15/15**. Mean correlations
with oracle non-max terms become 0.98205/0.98795/0.99234/0.97989 by block size;
oracle max means remain close to the actual finite query. These oracles are
diagnostic only and are never supplied to a runtime controller.

Next isolate the non-max error into basic SSIM, residual/HF and L8 terms using
the existing per-feature intervention owner, then correct the responsible
finite-removal/blur approximation. Do not spend another campaign tuning max
ownership or rerunning unchanged encodes. Preserve the max primitive, the
failed per-cell gates, and the separate coarse-block nonlinear-score failures.
Native JXL wiring/RD, corruption honest protection, HDR and full qualification
remain incomplete.

Verification: 463 core/golden/fold/SIMD/allocation/input tests and 16 candidate
serving tests pass (six ignored), as do CI-exact Clippy, 605-script lint and
the public API snapshot check. Compatibility comparison against the parent
finds exactly the two preregistered feature-gated methods, no removed/changed
items or traits, and no default supported API change. Reduced basic-only and
candidate-without-threads builds pass. The snapshot changes are committed
with the code. Performance and memory qualification remain unmeasured for
this complete query path.
