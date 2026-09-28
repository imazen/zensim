# Row-fused finite moments — September 14, 2026

Preregistered before implementation or measurement. The preceding finite-moment
prototype passes its small TRAIN mechanism screen but has excessive map cost.
Test an arithmetic-preserving implementation change: compute the signed edge
ratio once per channel/row, generate its artifact/detail L2/L4/L8 powers together,
and feed row buffers to the existing bin accumulator. Avoid six ratio passes
and six full-scale edge-power planes. Keep SSIM spreading and all scalar
arithmetic, model artifacts, coefficients, geometry, bin interpolation and
summation order unchanged. No feature ablation or new public API.

Refactor the existing bin owner to accept rows without changing its fold order.
Check fused row powers against the preceding per-plane implementation including
near-zero/signed inputs. Re-run bin/attribution, finite-moment/session and HDR
parity checks. Compare every scientific field of all forty preceding public-API
TRAIN reports at bins 1/8 (1,176 repairs), retaining failing head M2 verdicts.
Same admitted five JXL pairs, same four ensembles; no new fit, calibration,
source admission, EVAL or TEST access. Any changed scientific output blocks
acceptance of this optimization until explained; do not loosen tolerances.

Benchmark complete prepared calls, enabled/disabled, at 1024 squared and 2048
squared with the existing owner and all four ensembles, one pinned worker and
at least thirty accepted rounds. Re-run the saved previous binary under matching
conditions, bracketing the new binary, to distinguish implementation effects
from machine drift. Pin binaries, model hashes, flags, and per-arm rounds.
Summary-only timing does not qualify p95. Reject a statistically unconvincing
optimization or a material disabled-path regression; retain negative evidence.
Measure isolated total process RSS for basic228/H32 enabled at both geometries.
Do not promote the option or any model: human ranking, identity, corruption,
broader spatial/codec/HDR coverage and full frozen qualification remain open.

Artifacts: `~/work/zensim-validation-2026-09-14/moment-rows/`.

## Outcome: reject the implementation change

All forty reports reproduce the preceding JSON exactly: 1,176 real TRAIN pixel
repairs, all base scores/features/gradients/density and finite rectangle values.
Basic228/H32 still passes five small cases; the other models retain their M2
failures. No fit, calibration, EVAL or TEST was used. This proves the arithmetic
preservation claim, not broader model quality or native encoder value.

The fused edge loop uses vector division/multiplication in both dispatched
x86 implementations; binary disassembly is retained. Nevertheless, complete
prepared calls improve only modestly. Each process interleaves all four
ensembles with the correction off/on at both sizes, with thirty accepted
single-call rounds and clear owner reliability flags. Previous-binary runs
bracket the candidate; these are not paired samples across binaries.

| Enabled ensemble | 1024² old-before / candidate / old-after median ms | 2048² old-before / candidate / old-after median ms |
|---|---:|---:|
| basic156 | 91.26 / 90.21 / 91.89 | 375.16 / 359.07 / 376.40 |
| basic192l8 | 109.05 / 106.88 / 109.08 | 452.09 / 426.52 / 452.45 |
| basic192max | 115.68 / 113.61 / 115.45 | 467.23 / 449.73 / 467.20 |
| basic228 | 131.67 / 128.85 / 131.15 | 544.20 / 517.63 / 543.85 |

Enabled median cost falls 1.2–2.1% at 1024² and 3.7–5.7% at 2048² against the
bracketing old binaries. Disabled map medians instead rise 0.3–1.8%, consistently
across the sixteen comparisons. This is a measured tradeoff, not a universal
speedup. Given the much larger unresolved map-cost gap, retain the evidence
and reject this added implementation complexity. Production `attribution.rs`
is restored exactly to the preceding pushed version; no new runtime flag or
alternative scoring path remains. The candidate source, patch and binaries
are retained outside the repo for reproduction.

The three timing processes take 384 seconds total. Summary/mean confidence
intervals and all control arms remain in the linked results. No p95 is inferred
from medians or confidence intervals. RSS uses the existing separate thirty-call
map-construction mode, which does not query rectangles. Total process peaks
for the row candidate at bin 8 are:

| Size | Disabled KiB | Enabled KiB |
|---|---:|---:|
| 1024² | 105992 | 127540 |
| 2048² | 369596 | 457204 |

All four bound the declared worker cap on these inputs. They do not establish
incremental RSS, native codec memory, other models, fine bins or HDR costs.

## Verification retained

The candidate passes 36 tests in `attribution::tests` (two existing ignored),
three finite-moment/session tests, two HDR retained-feature
checks, minimal custom-profile build and CI-exact Clippy. The new row-power
reference test covers narrow/tail widths, identity, tiny, signed and large
edge changes; it lives with the archived rejected candidate.

The existing isolated SIMD integration owner now additionally checks enabled
finite moments against scalar scores/features and uncorrected density, plus
finite queries and the direction of signed curvature. Its ten dispatch
permutations pass on both the rejected candidate and the restored shipping
implementation. This extra production correctness coverage is retained.
No public API delta or scalar formula change occurs in this experiment.

[Structured results and evidence hashes](moment_rows_2026-09-14.results.json).

## Profile the larger costs before another optimization

A hardware-cycle profile of the restored shipping binary uses the existing
RSS/map-construction mode: basic228/H32, finite moments enabled, bin 8, 2048²,
one pinned worker, one reference preparation and thirty complete map calls.
It includes startup and **no rectangle queries**. The initial protocol label
incorrectly mentioned one query; source inspection corrected that label,
retaining the original metadata and unchanged command. This is a synthetic
implementation diagnostic, not a latency or quality qualification run.

The system `/usr/bin/perf` captures 3,230 samples at 199 Hz with zero reported
lost samples. The earlier user-local `perf` fails to load its Python library;
its failure log is retained. Sampled instruction-pointer/self-cycle shares are:

| Symbol / work | Self-cycle share |
|---|---:|
| `BinAccum::add_scale_plane` | 20.65% |
| `retain_max_removals` | 17.63% |
| horizontal SSIM blur inner kernel | 12.50% |
| vertical SSIM/features inner kernel | 11.98% |
| `moment_signal_plane` | 7.52% |
| box-spread kernel plus its sampled closure | 8.96% |
| memory copy | 4.93% |

The optimized release binary has unreliable reconstructed caller frames in
parts of the stack, so only flat instruction-pointer attribution is used;
no inclusive/call-tree percentages are claimed. Raw stack samples remain in
the private experiment directory. The flat symbol report and its command are
retained as reviewable evidence.

Next optimize regular dyadic bin folding at the existing `BinAccum` owner:
its coarse branch repeatedly computes clipped footprints and bin intersections
even when a power-of-two scale footprint fits completely inside an aligned
bin. Preserve accumulation order and handle partial edge cells explicitly;
retain arbitrary-bin and fractional-sampling behavior. Register and benchmark
that change against the restored baseline, including disabled controls. Max
retention and repeated extraction/map work are separate measured targets.
Do not start another feature fit to explain this implementation cost.

Historical guidance also remains explicit: the September 9 memory permits
registered bounded approximation when it earns speed; exact replay here is
only the control for an arithmetic-preserving optimization. The newest local
root-transcript file by modification time contains last text events from
September 8, not a new September 14 scientific ruling. The later September 13
data/corruption instructions remain authoritative. This was a selected-memory
and timestamp check, not a claim to have completed the full memory audit.
