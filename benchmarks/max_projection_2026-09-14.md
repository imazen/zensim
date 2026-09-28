# Exact maximum-projection reduction: correctness passes, latency unqualified

September 14, 2026; parent c45632a0. This changes retained map construction,
not feature definitions, model weights, score calibration, or spatial semantics.
No training, corpus, EVAL, TEST, or terminal data were accessed. Existing model
quality, finite-repair, and native steering failures remain failures.

The preceding native latency study identified basic228 map cost as a practical
bottleneck. A complete prepared basic228/H32 ensemble profile attributes 29.80%
of sampled user cycles to `retain_max_removals`. The proposed optimization keeps
every maximum feature and its original f32 signal expression.

## What changed and why it is equivalent

A scale column owns the same source-x footprint in every row; a row owns the
same source-y footprint in every column. Thus a left/right maximum projection
can first reduce each column across rows, then scatter its maximum through the
column's source footprint. Top/bottom projections can first reduce each row.
This is separability of maximum over finite nonnegative signals, including ties.
The existing prefix/suffix maxima and rectangle query rules are unchanged.

The production helper now uses contiguous column accumulators and row reductions
through the existing SIMD dispatch owner. Source-footprint scatter updates fall
from four per pixel per active maximum to two per row plus two per column.
Signal processing still visits every relevant pixel; total extraction is not
O(width + height). Temporary scratch is three f32 columns, O(scale width), reused
by successive rows; no new full-resolution signal planes are retained. The
historical per-pixel implementation exists only under cfg(test).

## Correctness evidence

- 3,584 exact projection comparisons: native cascade plus all 27 supported
  sampling contracts, four geometries, four scales, and eight sensitivity masks.
  Includes triangle, Mitchell, RobidouxSharp, 1.5x/2x/3x sampling, preserved Y,
  direct prime-divisor schedules, odd sizes, reflected tiny images, zeros and ties.
- The existing independent reflected-source enumeration passes 14,268 rectangle
  queries. It supplies a separate check of the underlying ownership/query rules.
- Every tested production retention call compares all four finalized projection
  arrays, feature IDs, and sensitivities bit-for-bit against the historical path.
- A public API differential test routes one prepared worker through the old
  implementation and another through the new implementation. A call counter
  verifies that the old branch actually executes. It checks complete scores,
  consumed features, sensitivities, coverage, binned density and refinement
  queries, with identity/distorted/identity-restoration behavior through reuse.
  Both frozen basic228 three-member ensembles (H32 and H128) pass 15,540 rectangle
  comparisons each on synthetic 17x9, 97x131 and 256x257 inputs, at Rev3, with
  parallel execution disabled/enabled and finite-moment refinement off/on.
  A deterministic all228 linear fixture supplies a repository-contained control.

The focused attribution suite passes 37 tests (two unrelated diagnostics ignored),
and the serving revision/corruption suite passes 12. Final frozen-model checks,
the two projection tests without default features, CI-exact Clippy, formatting,
and script lint pass. No public API or model arithmetic era changes. These tests
do not establish cross-platform timing, HDR model qualification, or correctness
of the existing finite-repair approximation against real interventions.

## Complete-path timing: preserve observations, reject qualification

Use the unchanged canonical `extract_paths_bench`, one CPU, one Rayon worker,
Rev3, the frozen basic228/H32 three-member ensemble and explicit uniform weights.
Two synthetic sizes, 30 one-call rounds each, four fixed ABBA blocks. A is the
previous verified binary; B changes only projection construction. Equal-length
binary paths, source/model/binary hashes, raw chronological observations and
before/after process snapshots are retained. There are 240 observations.

The canonical prepared benchmark creates a worker and prepares the reference
outside each round's timed closure. Consequently these timings may include cold
comparison scratch. They do not represent a warmed worker reused indefinitely.
Old/new implementations are in separate alternating process blocks, not paired
inside individual rounds. Whole-path timing replaces the planned separate
microbenchmark; this method change was registered before timing.

Each cell below is median / p95 milliseconds, with linear sample quantiles:

| Block | 1024x1024 | 2048x2048 | Flagged checks |
|---|---:|---:|---:|
| A, first | 77.21 / 77.97 | 336.71 / 338.22 | 3 |
| B, first | 71.86 / 72.60 | 288.14 / 290.32 | 8 |
| B, second | 71.53 / 72.69 | 290.77 / 292.16 | 4 |
| A, second | 84.19 / 84.74 | 336.77 / 338.50 | 3 |

**Zero of four runs meet this study's quiet criterion.** All observations remain,
including all 18 flagged checks; no repeat campaign or selective deletion was
used to obtain clean output. The owner's `unreliable=false` permits a limited
number of flagged checks per group and does not meet our all-clean requirement.
The 1MP baseline also drifts materially between repetitions. These numbers are
descriptive, not a qualified speedup or a passed runtime gate.

A second reused-worker CPU profile completes 40 calls with 1,429 samples and no
lost samples; the new row helper accounts for 16.16% of sampled cycles. The old
profile has 1,662 samples, none lost. Profiles are diagnostic distributions,
not an independently qualified wall-clock comparison. The first attempted perf
launcher failed before work because of a missing library; that failure is kept.

## Decision and remaining product work

Retain the bit-exact projection implementation and regression tests. Reduced
scatter work is established structurally, and complete public outputs agree;
a latency improvement is still unqualified. This does not promote any model or
make the previously failing prepared maps suitable for native spatial steering.

The next product work remains closing the independent finite-intervention/native
steering accuracy gap and selecting the quality/runtime frontier using TRAIN.
Do not repeat this timing campaign simply to get a quiet run; any qualification
campaign needs a separately justified, controlled measurement plan. No native
image performance generalization follows from these synthetic results.

[Machine-readable results](max_projection_2026-09-14.results.json) record the
checks and evidence hashes. Local immutable packet: `~/work/zensim-validation-2026-09-14/max-projection/`.
The served packet is `/zensim/reports/max-projection-2026-09-14/` on the existing
report server. Binaries and raw perf captures remain local; no model qualifies.
