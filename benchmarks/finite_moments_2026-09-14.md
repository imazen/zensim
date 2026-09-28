# Base-image finite moment refinement — September 14, 2026

Registered before implementation or measurement, following the positive
[curvature oracle](moment_curvature_2026-09-14.md). Test the actual correction
using only the source and current reconstruction. No repaired-image features
may enter prediction. Keep all model bytes, scalar features, scores, gradients,
old density and max estimates fixed.

Concrete API delta: add opt-in
`BakeScorer::with_finite_moment_refinement(bool)` (default false). Its current
callers are the existing finite-block coherence example and extraction-path
benchmark; their process flags select this explicit scorer option. Both direct
attribution and prepared steering use the same owner. Retain a private binned
moment integral per active L2/L4/L8 feature in `ScoredAttribution`; add its
finite-minus-linear correction in `refinement_gain`. No second scorer, feature
era or optimizer. Preserve old calls exactly when disabled. Record enabled
mode with every result. Sampling geometry must use its existing projection.

For each feature F=(mean v^p)^(1/p), estimate removed mass from its base-image
integral, with the existing window spread and sum-preserving scale projection.
Let r be removed mass divided by n F^p. Add
`-g F [1-(1-r)^(1/p)-r/p]` to the previous refinement estimate. Bound the
fraction to [0,1] because this models removal of frozen nonnegative mass;
this is not clipping the scalar score. Identity has no correction. Skip only
zero-sensitivity or zero-moment features. Preserve the chosen bin geometry;
small bins cost more memory. Initial implementation is an opt-in prototype,
not a claim of acceptable speed or native encoder usefulness.

Verify independent finite-power arithmetic, zero/full removal, tiny fractions,
signed sensitivities, bin queries, scalar/feature/old-density parity, odd and
small geometries, repeated prepared calls and enabled/disabled behavior.
Then compare all four frozen basic/peak ensembles on the same five admitted
TRAIN JXL cases (twenty cases, 588 repairs) with their original reports. No
new training, calibration, EVAL or TEST access. The primary mechanism bar is
all five basic228/H32 M3f>=.70, unchanged M2, finite and supported results.
Report all failures and per-case effects; no changed gate or qualified model.

Measure enabled and disabled complete prepared calls with the existing paired
benchmark at 1024²/2048², one pinned worker, >=30 accepted rounds, plus memory.
Preserve flags/build/model hashes. Summary-only timing cannot qualify p95.
If the mechanism fails, retain diagnostic evidence and avoid further storage
optimization. If it succeeds but is expensive, optimize the shared retained
moment accumulation before recommending this path. Corruption, identity
ordering, human rank, broader coverage and native targeting/RD remain required.

Artifacts: `~/work/zensim-validation-2026-09-14/finite-moments/`.

Before first pixel measurement, extend the registered check to repeat all
twenty cases with bin 8 as well as bin 1, matching the benchmark's prepared
grid. This is 1,176 repairs total. The example also computes an uncorrected
base map to assert feature/score/sensitivity/density parity and retain the max
contribution separately from root corrections. Those extra base comparisons
are counted; they do not supply repaired-image information to the predictor.

## Measured accuracy

The base-image-only correction passes the primary mechanism gate. Complete
basic228/H32 passes all five M2/M3f cases at both bin 1 and bin 8. All forty
cases finish (1,176 actual pixel repairs); every block is finite and supported.
The extra uncorrected base calls are separately counted in each report.

| Ensemble | Complete spatial pass before / 5 | After / 5, both bins |
|---|---:|---:|
| basic156_h128 | 2 | 2 |
| basic192l8_h128 | 4 | 4 |
| basic192max_h128 | 1 | 1 |
| basic228_h32 | 4 | 5 |

| basic228/H32 TRAIN row | Original M3f | Finite moments M3f, bin 8 |
|---|---:|---:|
| 396 | 0.8247 | 0.9400 |
| 2436 | 0.8754 | 0.9528 |
| 4597 | 0.9167 | 0.9667 |
| 5316 | 0.6739 | 0.9200 |
| 6577 | 0.9629 | 0.9815 |

Unlike the preceding oracle, this predictor never reads repaired-image features.
They are used only to measure outcomes. Actual base scores, features and
sensitivities match the uncorrected public call exactly; its additive density
is also exact. Every bin-1 uncorrected refinement value matches the preceding
immutable report, and all recorded repair scores and M2 values are unchanged.
Both bins give identical observed M3f ranks on this packet. H128 heads retain
their separate M2 failures; the correction does not repair head nonlinearity.

## Paired latency and memory

The existing owner measured thirty accepted one-call rounds per arm/geometry,
interleaved on pinned CPU 8 with one worker and fixed textured RGB8 inputs.
Its reliability flag is clear. No p95 is inferred from these summaries.
Each timed call builds the score/map against a cached reference and makes one
rectangle query. Reference preparation and isolated query latency are not timed.

| Ensemble | Prepared median 1024² ms, off → on | Prepared median 2048² ms, off → on |
|---|---:|---:|
| basic156_h128 | 59.82 → 91.97 | 236.40 → 376.57 |
| basic192l8_h128 | 62.37 → 109.70 | 247.41 → 452.28 |
| basic192max_h128 | 83.37 → 116.08 | 328.65 → 467.40 |
| basic228_h32 | 84.22 → 131.97 | 338.19 → 544.58 |

For basic228/H32 the extra preparation cost is approximately 57% at 1024²
and 61% at 2048². This is too expensive to recommend as the shipping spatial
path. The previous uncorrected map was already costly relative to scalar
scoring. The opt-in remains disabled by default; no latency gate is declared
passed. Reuse base signal powers and merge retained accumulation passes before
expanding this representation or promoting a default.

| Size | Finite moments | Total process peak RSS KiB | Below worker memory cap |
|---|---|---:|---|
| 1024² | False | 105672 | True |
| 1024² | True | 127144 | True |
| 2048² | False | 369640 | True |
| 2048² | True | 453848 | True |

These four isolated runs use the largest declared basic/peak profile here,
basic228/H32, with thirty calls per process and bin 8. Process RSS includes
inputs, code and reference/scratch/maps, so it bounds worker storage on these
inputs. It is not incremental allocation or native codec RSS, and does not
qualify other geometries, bins or models.

## Checks and remaining work

Independent finite-power arithmetic and prepared-session reuse checks pass
across even, odd and tiny geometries, identity transitions, and serial/threaded
calls. The broader attribution suite passes 37 tests with two existing ignored
diagnostics; the existing revision driver also passes. Two HDR retained-feature
parity tests pass. The minimal custom-profile feature build, CI-exact Clippy,
API snapshots/check and 196 applicable semver checks pass. Scalar extraction
and model inference are unchanged; the new public builder is additive.

The failed intermediate unit-test setup used unsorted feature IDs and was
correctly refused by model admission; the fixture was sorted without relaxing
that guard. The original logs are retained. No scientific fit or verdict was
changed by this fixture correction.

No model is qualified. Basic228/H32 still has the preceding human-ranking loss
and four distorted scores above 100. Corruption composition, broader TRAIN
spatial coverage, frozen EVAL, native attainable-bound targeting and spatial
RD, HDR serving qualification and p95 remain outstanding. The next performance
work is shared retained accumulation and signal-power reuse with these exact
map results as controls. Full [results and hashes](finite_moments_2026-09-14.results.json)
are also in the served artifact packet.
