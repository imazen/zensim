# Preserve scores while reducing horizontal SSIM blur cost

The v4x horizontal SSIM kernel now stages full sixteen-row groups with a
cache-line pad between physical rows when width is divisible by 256. Logical
width, mirror indices, horizontal tile boundaries and running-sum/FMA order
remain unchanged. Both the full and reference-cached three-output forms use
the same private arithmetic body. Other widths and remainder rows retain the
contiguous path. No feature, model, calibration or public API changed.

## Evidence and measured scope

Native `perf` sampling of the existing six-arm 1024² complete-scoring benchmark
places 69.04% of user-cycle samples in the fused horizontal SSIM kernel, with
hot instructions around strided stores. This is a mixed-workload profile,
not a candidate-only attribution of cost. The row-pitch hypothesis differs
from the historical negative experiment that staggered whole plane bases.
The intervention also makes each staged group exactly sixteen rows; cache
effects and resulting compiler specialization have not been separated.

Forty randomized single-call rounds per geometry, before and after, on the
9950X3D with CPU 8 pinned and one worker; optimized non-native-CPU builds:

| Surface | 1024² before → after mean ms | 2048² before → after mean ms |
|---|---:|---:|
| Complete A/D blend | 88.40 → 42.20 | 194.94 → 141.44 |
| Complete D member | 76.41 → 25.34 | 136.20 → 89.77 |
| fast-ssim2 control | 68.48 → 68.01 | 287.79 → 291.16 |

See the saved JSON for full member/B/D controls and exact unrounded values.
Blend observations after the change span 41.203–46.040 ms at 1 MP and
139.206–145.705 ms at 4 MP, within the absolute 50/200 ms budgets in this
screen. The blend still costs more than 1.25 times D in the same build.
Pooled-feature cost remains consequential even after removing common work.

Widths 576/1152, where staging stays off, have separate 40-round controls.
The first 576² single-call run fails dispersion (5.54–7.67% MAD/median in
some arms) despite passing mean ratios; retain it as rejected evidence.
A preregistered repeat uses nine base calls per sample through the same
benchmark owner for both original and padded kernels. Its dispersion and
regression controls pass. This is throughput evidence, not latency tails.
The original 1152² control already passes both checks. Every admitted arm
stays within the preregistered 5% regression tolerance after
normalizing to fast-ssim2; the largest normalized increase is 4.50% for D at
1152². Preserve that increase as a bounded regression observation, not proof
of zero cost. It remains below the experiment's retention threshold.

These are engineering measurements. Background activity/resource advisories
prevent strict quiet-machine release admission. The benchmark retains
means/MAD/extrema, not raw rounds or p95; no p95 is invented. Complete scalar
performance, cached/spatial/HDR latency and incremental per-worker RSS remain
unqualified. Each staged arena is proportional to sixteen rows, not image
height (390 KiB at width 1024); combined-process peak RSS is not a per-worker
memory measurement. No quality or spatial gate inherits a pass from timing.

## Correctness and validation

The original arithmetic body matches the new strided body after normalizing
physical stride back to width; only addressing and the staging wrapper differ.
Tests compare every output bit over 175 width/height/radius combinations,
including signed inputs, vector tails and both sides of the rem-ring limit.
The cached form also preserves its unused mu1 buffer.

All 1,320 saved image audits and 792 spatial audits reproduce byte-for-byte,
including complete feature CSVs, calibrated scores, pixel hashes and density
hashes. The A/D blend's 252 incomplete nonidentity maps remain explicitly
incomplete; padding does not implement their missing pooled integrands.
No train fit, validation selection, terminal evaluation or new encoding occurs.

The library and selected golden/fold/SIMD/attribution/allocation/invariant tests
pass 465 tests with six ignored. All sixteen complete BakeScorer surface tests,
including HDR and ensemble/corruption composition, pass. Root CI-exact Clippy,
scoped formatting and 605-script lint pass. Clippy required the equivalent
`is_multiple_of` spelling; the final and initially measured benchmark `.text`
sections are byte-identical, with both binaries and section hashes retained.
`ZEN_S2_CALLS` now supports these explicit throughput controls and refuses
zero/malformed sizes or ambiguous single-call requests. Its normal benchmark
iteration jitter and per-call normalization remain active. The measured
single-call binary also reproduces byte-for-byte from its saved source.

## Next action and reproducibility

Retain this arithmetic-preserving optimization. Profile the remaining full-pool
cost before choosing a further compute repair or a cheaper competitive model;
complete A attribution still needs its missing integrands. Corruption honest
protection and useful spatial RD/targeting across all four codecs remain open.
The post-change native profile now places 31.05% of mixed-workload cycles in
the fused horizontal kernel, 15.32% in fused vertical SSIM and 12.24% in the
single-plane horizontal box blur. Inspect those existing owners next; these
mixed-arm percentages do not isolate A's marginal full-pool cost.
The successful scalar preference screen remains valid because all scores and
features reproduce exactly; the full model is not qualified.

Artifact: `/mnt/v/output/zensim/extraction-profile-2026-09-08/`; Windows mirror
`~/work/zensim-validation-2026-09-08/extraction-profile/`. Registrations precede
the profile, repair and non-trigger controls. Preserve original/final sources,
packed model identities, binaries, perf samples, disassembly, all timing and
process records, exact replay outputs and validation logs. The prior failed
benchmark-gate experiment remains in its separate immutable packet.
