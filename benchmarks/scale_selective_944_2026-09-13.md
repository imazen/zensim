# Scale-selective Rev3 944 and prime-divisor experiment

Registered before implementation/results, September 13 (after the capability study).
Concrete callers: Rust BakeScorer, native extractor, feature-screen owner, existing
extraction benchmark and finite block-repair instrument. No new public Rust items.

Keep canonical IDs and Rev3 arithmetic. Derive actual v2-era scale work from
read IDs, including adjacent gradient dependencies of EDGE_WIDTH_CHANGE; keep
legacy masked/IW scale selection. Retain SIMD kernels and their reduction order.
Check consumed values against unrestricted extraction at every scale, odd/small
images, sequential/threaded paths and HDR. Structural fills must match plan
coverage. Feature removal must result in measured skipped work.

Add an explicit experimental sampling metadata version for direct-from-original
XYB planes at divisors [1,2,4,8], [1,3,5,7], [1,2,3,5]. Triangle, Mitchell and
RobidouxSharp use the pinned zenresize main owner. Independent direct sampling
separates scale position from cascaded filtering. Dimensions floor after one
reflection pad sufficient for four valid levels. Keep signed floating lobes.
Unknown metadata and mismatched reference caches fail. SDR sampling only.

For integer divisor d and block period B, sampling origins visit B/gcd(B,d)
phases before repeating, under an exact integer grid. Odd primes avoid the
power-of-two common factors; they do not guarantee detection. Floor-rounded
output dimensions change the actual step to padded_length/output_length;
measure actual coefficient phase/support rather than assuming exact d.
Aliasing is governed by filter attenuation and the reduced Nyquist limit,
not number theory alone. Spatial ownership uses actual resizer tap geometry,
explicitly an approximation tested by finite repairs, not an exact derivative.

Audit masked/IW as normalized weighted pools: nonnegative weights, identity,
constant-signal response, scale-invariance under common weight scaling, and
bounds by the signal extrema. V1 activity-proxy IW and v2 bounded activity IW
are distinct from GSM mutual-information IW-SSIM; neither name proves parity
with that paper. Source: https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf .
Independent f64 reference tests must examine actual SIMD values and tails.

Use the previous frozen 19,958-row admission and source splits. Native per-scale
masked/IW screens can reuse those feature values only after current canonical
parity checks. Fresh direct-scale values require a new explicit producer identity
and extraction. Paired seeds 5101/5103/5107, H128 frequent checkpoints (32 epochs),
raw MAE and signed SROCC through the Rust panel. Separate human/codec-proxy/class
labels; no protected terminal data or model qualification. Start with triangle
on all three direct schedules; benchmark all three filters before broadening
training. Inspect each masked/IW scale and a pooled control. Stop after this
bounded comparison, recording failures, actual runtime, and limited source
coverage. No fraction of feature count is a speedup claim.

Pre-fit precision correction: the independent f64 coefficient check rejected
an unquantized interpretation of zenresize's public float API. Its streaming
owner rounds source rows to IEEE binary16 in a ring buffer, then filters to
f32. Direct sampling v2 therefore explicitly includes one source-plane f16
rounding per resized level; level 0 is unchanged f32. The v1 cascade already
had this behavior. Signed lobes survive, but this is not full-f32 convolution.
A separate quantized f64 oracle checks actual SIMD results with 2e-6 absolute
tolerance. The unquantized error is measured separately, not hidden by relaxing
the arithmetic gate. This precision limit may matter near identity; no model
is qualified or existing contract silently reinterpreted by this experiment.

Frozen fit count: native 12 layouts × 3 tasks × 3 seeds = 108. Four separate
legacy masked+IW scale additions and four v2 masked+IW scale additions use
basic228 as baseline. Full944/619/800 are broader controls. Three Triangle
direct schedules each test basic228/full944/coarse-v2-476 = 81 additional fits.
Total 189 fits, each 32 epochs with checkpoint selection every epoch. Filter
runtime checks cover all nine direct contracts; only Triangle is trained in
this bounded comparison, so no trained Mitchell/RobidouxSharp verdict is implied.

## Mathematical contracts and SIMD choices

The new behavior is private planning, not a new feature arithmetic revision.
The 348/append/append2 SIMD kernels run only on selected scales; default full
extraction retains their formulas and reduction order. Neighbor gradients are
retained for edge-width ratios, including the last scale's copied ratio.
The basic/peak backbone and source pyramid remain shared work. This is not
arbitrary per-signal kernel dispatch or a promise that every unread column
removes a pass. Legacy masked/IW share one scale-local activity/pooling chain.

For v2 activity a >= 0 and c = 0.01:

    t = a / (a + c)
    w_mask = 1 - t
    w_iw = t + 0.001
    pool(e,w) = sum(w*e) / sum(w)

The two weights satisfy w_mask + w_iw = 1.001. Their weighted numerator sum
must therefore equal 1.001 times the unweighted signal sum (within reduction
rounding). Independent f64 tests exercise the dispatched bounded-MSE pools,
identity, constant error, positivity/extrema, this partition identity, and
non-multiple-of-eight row tails. A common positive weight multiplier cancels.
This establishes those identities, not that activity is mutual information or
that the complete 944-feature family is perceptually optimal. Earlier claims
that the standalone IW helpers are paper-faithful have been corrected.

All nine direct schedules/filters are tested against independent separable
f64 convolution after explicitly modeled binary16 source rounding. Comparing
with an unquantized oracle is a required negative control. On the fixed
97x131 texture, maximum unquantized differences are 0.000121–0.000155 XYB;
the quantization-aware gate is 0.000002 absolute. Those numbers bound this
fixture, not every image. Binary16's roundoff can erase sufficiently small
coarse-plane differences and must be considered before near-lossless release
qualification. No custom SIMD resize kernel or private benchmark-only API is used.

For an exact integer grid, divisors 3/5/7 each visit every phase of an 8/16/32
pixel codec block. Divisors 2/4/8 visit progressively fewer phases. The actual
resizer step is padded_length/floor(padded_length/divisor), so odd dimensions
also move phase; kernels overlap neighboring phases rather than point-sampling.
A 3x reduction has Nyquist frequency 1/6 cycle per original pixel, versus 1/4
for 2x: better phase coverage does not recover frequencies removed by filtering.
All direct schedules keep full-resolution planes, so their coarse-scale tests
can retain impulse/aliasing evidence at scale0. Cross-scale gradient features
remain ratios in sampled-pixel units with existing stabilizers, not physical
per-octave derivatives. Retraining is required for each sampling contract.

## Data, fitting and interpretation

This extends the later September 13 capability study, not the earlier tiny
T2 proxy packet. Each sampling contract freshly extracts the same 19,958 rows:
11,125 human (KADID plus train-only TID), 620 codec-proxy and 8,213 deduplicated
corruption pairs. Current source-family admission excludes validation families
8414/8434 and co-locates the 7004/7058 derivatives. Human fit/dev/test contain
7,000/1,000/3,125 rows. Codec fit/dev/test contain 373/121/126 rows from
18/6/5 source families; test has six origins because of the grouped derivatives.
Corruption fit/dev/inner-test contain 5,504/1,482/1,227 rows from 8/2/2 origins.
The inner corruption test has 82 honest and 1,145 corrupt examples. No fixed
corruption validation origins, KADID terminal, CID22 gold, AIC or T0 admission.

Human data span 25 distortion types and five severities; codec data include
JPEG, WebP, AVIF-SVT and JXL, five knob settings per available source/codec and
identity anchors. Codec labels are signed SSIMULACRA2 proxies, not human
judgments or target-controller errors. Corruption targets are 0/100 class
labels with balanced training groups, not calibrated probabilities or a quality
dial. The full owner summaries retain per-family errors and threshold-50
false alarms/misses. The class imbalance makes correlation a poor headline
for corruption; compare the actual misses and honest-image failures too.

All fits use the same Rust H128 trainer, 32 epochs, 8,192 sampled pairs per
epoch, MSE weight 1, no early termination, paired initialization and sampling
seeds, and dev checkpoint selection every epoch. The default 50-epoch learning
rate cycle is unchanged. This short-budget comparison tests useful attainable
accuracy; it does not establish an asymptotic feature ceiling. Previous capacity
and data-volume controls still apply. Seed spans are not confidence intervals;
codec and corruption source diversity remain particularly limited.

Inference and evaluation use final Rust bakes through BakeScorer, Rust feature
prediction and the existing Rust panel with `--raw-errors`. No remapped MAE is
substituted for score error. Masked and IW are added together at each scale,
separately for legacy and v2 definitions; this does not isolate masked from IW.
All native scale subsets retain the canonical Rev3 values. Direct contracts
change the input planes and require separate training; they are not transparent
substitutions into a model trained on native dyadic planes.

Reproduction uses the existing owner, for each of the four committed recipes:

```sh
../scripts/run-heavy --mem 16G --jobs 8 scripts/run_full_eval.sh \
  --stage feature-screen benchmarks/scale_selective_944_2026-09-13.json \
  <fresh-output-directory> --ceiling-stage prepare
# Repeat with --ceiling-stage fit, then audit, then report on that directory.
```

Artifacts are retained under `~/work/zensim-validation-2026-09-13/scales944/`:
per-contract INPUTS/RESULT/SUMMARY, row-joined Parquet, original pair hashes,
producer metadata, recipes, binary hashes, final bakes, pixel and spatial audits,
raw panels, and execution manifests. `CODEC_PROVENANCE.json` preserves verified
original encoder-binary and parent-manifest hashes. Separate JPEG/WebP/JXL
commit identities were absent from that historical packet; this is not a claim
about current codec performance. A SIGTERM (exit 143) interrupted the first
prime357 preparation; its partial directory and failed stage remain recorded.
A fresh preparation resumed the unchanged recipe; no partial table was fitted.

## Completed accuracy and spatial results

All 189 fits, native pixel audits and spatial instruments completed. These
are three-seed medians of raw MAE, with distinct units in each column.
The compact result JSON preserves seed min/max; full owner summaries retain
family-level panels. No new model or feature set is qualified.

| Sampling | Layout | Human | Codec proxy | Corruption class |
|---|---|---:|---:|---:|
| native | basic228 | 7.332 | 14.871 | 17.585 |
| native | full944 | 7.199 | 13.150 | 8.936 |
| native | fine_y_v2_half | 7.316 | 11.790 | 8.573 |
| native | basic_v2_all | 7.344 | 13.300 | 8.746 |
| native | v1_iwmasked_s0 | 7.268 | 14.311 | 18.236 |
| native | v2_iwmasked_s0 | 7.398 | 13.593 | 16.447 |
| native | v1_iwmasked_s1 | 7.374 | 14.575 | 18.160 |
| native | v2_iwmasked_s1 | 7.164 | 14.560 | 17.065 |
| native | v1_iwmasked_s2 | 7.357 | 13.807 | 17.933 |
| native | v2_iwmasked_s2 | 7.239 | 14.641 | 16.843 |
| native | v1_iwmasked_s3 | 7.458 | 14.880 | 17.745 |
| native | v2_iwmasked_s3 | 7.304 | 13.819 | 17.380 |
| dyadic | basic228 | 7.448 | 12.284 | 18.496 |
| dyadic | full944 | 7.353 | 10.225 | 8.666 |
| dyadic | fine_y_v2_coarse | 7.256 | 11.045 | 16.926 |
| prime357 | basic228 | 7.397 | 11.923 | 20.715 |
| prime357 | full944 | 7.250 | 11.436 | 8.579 |
| prime357 | fine_y_v2_coarse | 7.344 | 12.063 | 16.894 |
| prime235 | basic228 | 7.451 | 13.797 | 18.038 |
| prime235 | full944 | 7.286 | 11.525 | 8.952 |
| prime235 | fine_y_v2_coarse | 7.213 | 13.877 | 14.486 |

Native v2 masked+IW at scale1 (half resolution) gives the lowest median human
MAE in this screen, 7.164 versus basic228's 7.332. That small advantage does
not establish superiority across content, and its corruption MAE remains
17.065 versus full944's 8.936. Legacy masked+IW alone does not recover the
wide feature set's corruption benefit at any scale. The broader 619 layout
retains half/quarter/eighth v2 work and is a more useful scalar candidate:
7.316 human, 11.790 codec-proxy, 8.573 class MAE. It has spatial failures.

At threshold 50, native full944 has 0/82 honest-image false alarms for each
seed, with 14/1,145 corruption misses for each. Basic228 has 15/23/26 false
alarms and 61/21/17 misses (seeds 5101/5103/5107). Native619 has 1/0/0 false
alarms and 17/11/15 misses. These are inner tests on only two corruption
origins, not a deployment false-positive-rate estimate.

For full944, direct Triangle 1/3/5/7 slightly improves human and class median
MAE over direct 1/2/4/8, but worsens codec-proxy MAE 10.225 -> 11.436. Direct
1/2/3/5 worsens that codec error to 11.525 too. Coarse476 at 1/2/3/5 improves
class MAE 16.926 -> 14.486 and human MAE 7.256 -> 7.213 versus its direct
dyadic control, while worsening codec error 11.045 -> 13.877. Prime phase
coverage is therefore not a free improvement. Human seed spans overlap;
this screen supports retaining primes as explicit experiments, not switching
the default pyramid or attributing differences solely to block misalignment.

Spatial checks use actual reference-block replacements, block size32, seven
retained JXL/noise/swap/aliasing cases per bake. PASS requires supported
refinement, finite values, M2 >= 0.8 and M3f >= 0.9; the global R/B swap is
reported but diagnostic-only. Those are this screen's thresholds, distinct
from the older gallery's M2 >= 0.99 / M3f >= 0.70 display thresholds.

Human native basic228 passes 12/18 non-swap checks. Native619 fails all18;
native800 passes2/fails16. Each native v2 masked+IW addition passes6/fails12.
Full944 and legacy masked+IW read sets are UNSUPPORTED because legacy
weighted features lack refinement support. Every direct coarse476 layout
fails all18 human checks (and all18 each for codec/class fits). Correct
sampling coordinates and finite maps do not establish useful steering.
Fix the v2 retained spatial failures before qualifying one of these scalar
candidates. These tests do not run native codec RD/target loops, and there
is no new target-error-bar or HDR qualification claim.

## Measured runtime and completed verification

Quiet before/after measurements use the same final human seed5101 bakes, warm
BakeScorer.compute, one synthetic 1024x1024 pair, 15 interleaved samples per
arm, on Linux with a Ryzen 9 9950X3D (Rust 1.98.1). CPU affinity is core0 for ST and cores0–7 for MT8. No training/build runs
overlap; the process snapshot is retained. Both binaries and model bytes are
SHA256-pinned; embedded git hashes alone do not identify a dirty build. The
benchmark reports `unreliable=false` for every run; some arms have round drift,
and raw min/max/MAD are retained. These are local microbenchmarks, not corpus
latency or memory qualification. Values below are milliseconds, median [min,max].

| Native layout | Before ST | After ST | Before MT8 | After MT8 |
|---|---:|---:|---:|---:|
| basic228 | 27.188 [27.138,27.358] | 26.505 [26.309,29.618] | 5.479 [5.338,5.811] | 5.491 [5.231,5.619] |
| full944 | 58.272 [58.169,58.459] | 58.726 [58.565,58.846] | 18.770 [18.403,19.449] | 19.160 [18.607,19.853] |
| fine_y_v2_half | 51.691 [51.517,55.623] | 20.725 [20.588,20.833] | 17.843 [17.309,18.309] | 7.521 [7.023,8.131] |
| basic_v2_all | 51.666 [51.573,52.810] | 52.065 [51.914,53.573] | 17.848 [17.419,18.734] | 18.205 [17.614,18.656] |
| v2_iwmasked_s0 | 51.674 [51.520,51.850] | 45.148 [44.875,46.118] | 17.602 [17.269,18.817] | 15.636 [14.851,16.357] |
| v2_iwmasked_s1 | 51.602 [51.523,51.876] | 26.162 [25.946,26.584] | 17.677 [17.350,18.668] | 7.685 [7.321,8.030] |
| v2_iwmasked_s2 | 51.664 [51.543,52.768] | 21.494 [21.381,21.607] | 17.737 [17.412,18.585] | 5.794 [5.417,6.500] |
| v2_iwmasked_s3 | 51.669 [51.511,55.780] | 20.529 [20.394,20.614] | 17.834 [17.398,18.947] | 5.424 [5.100,5.727] |
| v1_iwmasked_s0 | 32.534 [32.450,34.570] | 31.886 [31.692,32.070] | 6.753 [6.482,6.942] | 6.812 [6.483,7.138] |
| v1_iwmasked_s1 | 28.428 [28.399,28.649] | 27.717 [27.552,27.808] | 5.973 [5.545,6.099] | 5.852 [5.436,6.042] |
| v1_iwmasked_s2 | 27.508 [27.460,27.685] | 26.794 [26.749,26.909] | 5.667 [5.397,5.873] | 5.542 [5.442,5.765] |
| v1_iwmasked_s3 | 27.279 [27.232,28.262] | 26.573 [26.450,27.397] | 5.537 [5.276,5.701] | 5.477 [5.212,6.018] |

The 619 layout now costs 20.725ms versus 51.691ms ST (60% less), and 7.521ms
versus 17.843ms MT8 (58% less). This is actual extraction work removed, with
consumed features preserved. Coarse v2 masked+IW rows also become substantially
cheaper; full944 remains a control with about 0.8% ST/2.1% MT8 higher medians.
Within-scale family dispatch and scratch allocation remain conservative.

All three kernels below use equivalent all-live diagnostic 944 producers and
exact sampling metadata. This prices the complete public-API path, including
resize and extraction; the diagnostic scalar head is identical across filters.
It does not assign predictive accuracy to untrained Mitchell/Robidoux models.
Values are medians; full dispersion is in the compact JSON.

| Direct schedule | Kernel | ST ms | MT8 ms |
|---|---|---:|---:|
| dyadic | triangle | 83.276 | 25.139 |
| dyadic | mitchell | 102.707 | 30.721 |
| dyadic | robidouxsharp | 102.611 | 29.164 |
| prime357 | triangle | 73.236 | 21.806 |
| prime357 | mitchell | 89.816 | 24.950 |
| prime357 | robidouxsharp | 89.970 | 25.650 |
| prime235 | triangle | 91.380 | 26.214 |
| prime235 | mitchell | 113.363 | 30.612 |
| prime235 | robidouxsharp | 113.501 | 30.505 |

Triangle is cheapest in this comparison. Mitchell/RobidouxSharp add about
23–24% ST cost, with a smaller and noisier MT8 difference. Prime357 is cheaper
than direct dyadic at matched feature width, but direct-from-original filtering
still costs more than the native pyramid: trained full944 costs 73.034ms ST
at prime357 versus 58.726ms native. Prime235 increases sampled area and is
more expensive. No default filter or pyramid change follows from this study.

Every one of 189 bakes passed pixel re-extraction audits: 2,709 pair evaluations,
zero consumed-feature delta. Native full extraction is byte-identical across
19,958x944 values (CSV SHA256
`bd01507e26b143fbf194bd16d95f9d49e4fcc78246d421359467ec690b3941a8`).
The 1,323 spatial records include 756 native and 189 per direct schedule,
with 69,552 actual block replacements; unsupported results remain visible.

Local checks passed: all-feature library suite (466 passed, eight intentionally
ignored performance probes at that run), six dense-layout/public-API tests,
Rev3 weighted-pool algebra and every scale mask, SDR/HDR consumed parity,
all nine direct filter/schedule convolution references, serial/threaded pyramid
parity, ten Rev3 feature invariants, legacy identity expectation, ten registry
tests, the cross-feature-build serving matrix, public API snapshot check,
CI-exact clippy, script lint and
format checks. Added tests after the full suite were run directly; logs and
hashes are retained in `validation/RESULT.json`. An old identity test asserted
pre-Rev3 residue; it now correctly distinguishes Rev3's exact zero v1 block
from legacy residue, without changing production identity behavior.

Cached individual fits took approximately 19–47 seconds in the bounded
concurrent run. Fresh extraction per contract took about two to two-and-a-half
minutes. A single feature experiment can fit the requested five-minute loop;
the complete 189-fit campaign is a larger batch. Those elapsed times are
orchestration observations, not uncontended training benchmarks.

Keep native619 as a scalar research candidate, retain full944 as the corruption
control, and prioritize fixing spatial repair prediction and expanding source
coverage. Prime schedules remain explicit test arms; per-scale masked+IW alone
is not the corruption solution. Binary16 precision needs separate near-identity
assessment before a sampled model can qualify for the end-user target dial.

The served report is `/zensim/reports/scale-selective-944-2026-09-13/index.html`
on the existing development gallery service. Compact results and all four
recipes are committed beside this report. Detailed bakes, tables, raw panels,
spatial records and reproducibility manifests remain in the shared artifact
folder; no shipped profile or model was replaced.
