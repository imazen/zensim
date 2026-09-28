# Rev3 maps, native color and HDR optimization — September 14, 2026

Implementation and TRAIN-only correctness/throughput evidence following the
[revised priorities](../docs/REV3_OPTIMIZATION_AND_COLOR_PLAN.md).
No new model fit, EVAL/TEST access or product promotion. The existing product
qualification verdicts remain unchanged.

## Changes delivered

- **Basic/L8 map fusion:** one retained-plane combine shares loads, edge-ratio
  division and powers. Const-generic specialization removes L8 work when unused.
  Separate f32 accumulation rounding points and f64 L8 coefficients remain.
  Exact hard-max row/column projections already existed and are preserved.
- **Native SDR peer audit:** the existing fast-ssim2 consumes canonical linear
  u16/f32-derived samples from the scorer's shared conversion. Original pixels
  and transformed float buffers have separate hashes. Arbitrary ICC stays with
  the existing decoder/CMS owner; this does not add a second color transform or
  SSIM algorithm. The SDR contract still clips to the sRGB display gamut.
- **HDR common primaries:** PQ, HLG and absolute-linear ImageSource paths now
  convert declared primaries into the opsin matrix's linear-sRGB basis without
  an SDR clamp. HLG luminance uses the source basis. The existing opsin/PU
  domain handling remains; this is not an unrestricted negative-light model.
- **Exact PQ16 lookup:** all 65,536 native codes reuse the existing PQ EOTF in a
  lazily initialized 256 KiB table. Peak, black and reflection stay per comparison;
  no interpolated approximation or second EOTF formula. Float PQ remains the
  direct-formula reference. Startup initialization is outside steady-state timings.
- **Explicit input/revision admission:** folded wide public extraction refuses mixed
  process/requested arithmetic revisions. Basic-only plans retain cross-revision
  serving. HDR rejects invalid display parameters and unsupported alpha.
  The native HDR datagen extractor uses zenpng16 and validates requested JXL
  output, metadata and completeness before producing a feature table.

HDR semantics are now **`hdr-common-primaries-v2`**. Old affected HDR features and
calibrations must be regenerated/reassessed. Raw `compute_pu_linear` callers
still supply absolute RGB already in the opsin linear-sRGB basis. The datagen
CLI requires `--input-contract hdr-common-primaries-v2-bt2020-pq10000`;
untagged references are an explicit datagen declaration, not a bit-depth inference.
Its manifest records the input era and extraction revision, but does not replace
canonical per-file hash admission. Arbitrary HDR ICC is refused by that tool.

## Correctness evidence

Nested native decoder/input tests passed 24 checks, and the HDR datagen test
passed. The root library suite passed **455 tests**, with 8 ignored. Public integration
checks passed: attribution across SIMD tiers (1 matrix test), dense layout
round trips (6), and native HDR/color/revision handling (4).

- Every PQ16 code in all three RGB positions at 100/1,000/4,000/10,000-nit peaks
  matches the existing float decoder **bit for bit**. Public u16 versus equivalent
  float-code extraction also matches exactly.
- Fused L8 versus the retained original two-pass implementation matches bits on
  scalar/SIMD tails, odd lengths, signed/zero and extreme coefficients, with
  nonzero destination accumulators. Public max/L8 models exercise both Rev1
  and Rev3 over all ten host dispatch permutations.
- HDR tests cover independent f64 colorimetry/display-light references, P3 and
  BT.2020, values above SDR white, negative gamut intermediates, PQ/HLG native16,
  invalid displays, scalar/cached extraction, and mixed-revision refusal.
- Native SDR tests check low-bit visibility and canonical C0-continuous sRGB
  conversion against an independent f64 reference. The canonical curve uses
  `linear-srgb`'s C0 constants; the initial textbook IEC reference was mismatched.
  The 2e-7 tolerance was not widened to conceal that mismatch.
- Native datagen tests preserve u16 low bits, accept declared untagged/matching
  BT.2020 PQ, and refuse HLG/P3 conflicts and 8-bit references.

The **214-pair admitted native SDR TRAIN replay** is complete. Public feature
CSV bytes and prior scalar/spatial audit measurements are unchanged. All 33
identities score 100. Native peer range: −62.19838 to 100; matched legacy peer:
−62.07116 to 100. Original source/decoded hashes are verified; the native audit
adds transformed float hashes. Historical q85 color quarantines and head misses
remain unresolved and are not reclassified by this audit.

Two testing corrections matter when interpreting the evidence. A wide raw
extraction test initially mixed explicit Rev3 with process Rev1 and produced
invalid large discrepancies; this combination now refuses. Independently,
minute f64-versus-f32 gamut-rounding differences on a 17×9 fixture can be amplified
by higher-order small-scale moments (one feature: 0.002000686 versus0.002535798).
Tests therefore separately bound matrix error against input magnitude and replay
an identical f32 representation through public paths. **Arbitrary full944 HDR
cross-representation stability is not established.**

## Performance

Existing `extract_paths_bench`, complete three-member Rust ensembles, Rev3,
one worker pinned to CPU8, 30 measured rounds, one comparison per round. No
`target-cpu=native`. Reference preparation and benchmark input construction are
outside measurements. Prepared timings include the existing half-image query.
Inputs are synthetic 1024²/2048² or separately hash-bound real TRAIN images.
SDR-trained weights are throughput instruments on HDR, not calibrated HDR models.
Raw rounds, registrations, binary hashes and resource flags are retained.
Builds started from `190d0340` plus the recorded working changes; the source
commit alone does not identify a benchmark binary.

PQ16 optimization, synthetic 1024², ABBA order; milliseconds per comparison:

| Profile | Before block0 p50/p95 | After block1 p50/p95 | After block2 p50/p95 | Before block3 p50/p95 |
|---|---:|---:|---:|---:|
| basic228/H128 ensemble | 71.26 /73.24 | 34.77 /35.07 | 34.15 /34.42 | 71.43 /72.48 |
| Y60/H32 ensemble | 62.17 /62.84 | 25.47 /25.84 | 25.26 /25.54 | 62.25 /62.77 |
| basic156/H128 ensemble | 71.36 /73.65 | 34.68 /34.95 | 34.21 /34.57 | 71.31 /72.24 |

Basic228 steady-state latency is about52% lower; Y60 about59% lower. Block0 was
flagged unreliable (6 resource waits); the other blocks were not (2/1/3 waits).
The large difference repeats in both directions, but this is a local synthetic
throughput result, not a quiet-qualified corpus/product claim. A final optimized
build reconfirmed p50 34.11ms for basic228,25.22ms for Y60 and34.10ms for basic156
(30 rounds,2 resource waits, not flagged unreliable); see `pq-shipping.json`.

Basic228 map fusion on synthetic inputs:

| Size | Before p50/p95 ms | After p50/p95 ms | p50 change |
|---|---:|---:|---:|
| 1024² | 73.34 /74.16 | 71.62 /72.97 | −2.34% |
| 2048² | 289.13 /292.32 | 280.25 /283.83 | −3.07% |

Nonpeak synthetic controls moved within about 1.3%. Real TRAIN before/after
blocks drifted, including unchanged controls: peak cases changed 34.18→38.16,
52.11→55.46,74.49→76.06,8.78→8.80 and 32.90→36.18ms. The native-before run was
unreliable. **No general map speedup is qualified from those real-image runs.**

Additional synthetic 1024² format measurements, basic228 ensemble:

| Input | Scalar p50/p95 ms | Prepared p50/p95 ms |
|---|---:|---:|
| Native SDR16 | 19.27 /19.70 | 73.00 /73.93 |
| Linear-float P3 SDR | 18.42 /18.71 | 73.63 /74.87 |
| Absolute-linear BT.2020 HDR | 30.88 /31.16 | Unsupported |

These format runs used the pre-PQ-lookup build; only PQ16 changed with the lookup.
No whole-image RGB8 projection is used for native measurements. HDR prepared
mode explicitly refuses rather than benchmarking an SDR substitute.

## Recovery and remaining product work

Both original supervision legs now have complete current Rev3/full944 TRAIN
caches: **CID22 17,611 pairs/201 references; SafeSyn 196,086 pairs/3,218 source
paths**, zero failures. SafeSyn retains 11,591 negative fresh SSIM2 targets,
range −743.8610 to 100, plus original oracles. These are legacy-RGB8 SDR controls.
The [recovery record](baseline_recovery_2026-09-14.md) binds their hashes.
Source-family fit/development/calibration admission is pending; no new model
has been fitted and competitive recovery has not yet been demonstrated.

Remaining priorities are concrete:

1. Avoid the second basic/peak retained extraction in prepared maps: retain
   signals during planned extraction, then combine after sensitivities exist.
2. Admit disjoint TRAIN families and recover one fast and one richer peak model
   using the established supervision/training/calibration recipe.
3. Implement native HDR prepared maps on the shared retention/attribution owner,
   establish matched HDR supervision and assess a shared SDR/HDR weight set.
4. Qualify frozen Rust compositions with the complete composite, scatter/tails,
   codec-attainable bounds, 1/2/3 shots and native spatial intervention/RD evidence.

These are not satisfied by color unit tests or a fast scalar benchmark. No
all-purpose HDR model or improved product qualification is claimed here.

## Reproduction and evidence

Artifact root: `/var/tmp/zensim-validation-2026-09-14/rev3-native-optimization/`.
Served report: `/zensim/reports/rev3-native-optimization-2026-09-14/index.html`.
`REGISTRATION.json`, `PERFORMANCE.json`, `DEPTH_REGISTRATION.json`,
`DEPTH_PERFORMANCE.json`, `PQ_LUT_REGISTRATION.json`, `PQ_LUT_PERFORMANCE.json`,
`NATIVE_ADMISSION.json` and `NATIVE_VERIFIED.json` bind the experiments.
The replay bundle retains benchmark binaries, reviewed source patch and original map-combine source, commands,
raw samples, audits and validation logs. Large TRAIN caches stay in their
original artifact directory; their verification receipts are included.

Local gates: CI-exact root clippy, touched nested extractor clippy, script lint,
public API snapshot check, scoped formatting and the tests above. One hidden
instrumentation function was added for a concrete native peer caller; no new
public model or feature API was invented. CI is not on the critical path.
