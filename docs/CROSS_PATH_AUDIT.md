# Cross-path reuse and exactness audit

Baseline implementation: `d15a5486`. Scores and maps must retain their existing
bits during this refactor. No public API or model changes are included.

## Routes and reuse boundaries

- `Zensim::compute` selects the streaming fold for supported model plans. Serial
  and parallel folds share the same canonical strip reduction order.
- `compute_with_ref` and `compute_with_ref_into` feed the fold from a cached
  reference, with a buffered fallback when the cache cannot feed it. The latter
  also reuses caller-owned scratch.
- RGB and planar linear diffmaps share the scale-band implementation. This refactor
  also shares final score selection and the mask/square-root/crop sequence. The
  diffmap score-selection step evaluates its serving plan once. The canonical
  scoring entry retains its own plan validation. The original crop order and
  allocation fast path are preserved.
- Planar canonical diffmap scoring currently materializes interleaved RGBA. Removing
  that allocation requires a compatible row-source design; the present ImageSource
  trait returns borrowed packed rows. It is not a free interface substitution.

## Boundaries deliberately retained

Scalar and SIMD extraction use the same fold planning and result scoring owners;
threaded extraction keeps the canonical strip reduction order. The refactor does
not replace those kernels or their reduction order.

Cached and uncached fold entry points validate different inputs and expose different
failure behavior. Their feature-width checks remain at their respective entry
points. Unifying those checks would require changing fallback/error behavior, not
just removing duplication.

The uncached diffmap entry can recognize identical source bytes and return the
identity score and zero map. A cached reference cannot perform that byte comparison.
This distinction remains explicit at the entry points.

## Existing mismatch

Before any production refactor, profile A produced different score bits between
interleaved and planar linear diffmaps on a 65×63 checkerboard with strided planes:
`4635359462469191635` versus `4635359462376882550`. The new cross-path test exposed
this; PreviewV0_2 also showed score mismatches, and legacy maps showed rounding
differences at 65×63 and 257×193. Changing either result conflicts with preserving every existing score.
The audit records this limitation rather than silently changing the score contract.

## Validation design

`zensim/tests/cross_path.rs` exercises subminimum images, odd dimensions, poisoned row
padding, masking/sqrt combinations, cached-reference scratch reuse and SIMD tiers.
`zensim/examples/path_audit.rs` supplies output fingerprints and zenbench timings for the
compute, cached, scratch, map and planar routes. Scalar token dispatch and explicit
thread counts are separate axes. The map contract also runs every available token
permutation, checking complete output lengths before comparing values. Memory observations use fresh-process maximum RSS,
including fixture and reference setup; timings exclude that setup and fingerprinting.

| Mode | Entry point | Threads | Dispatch |
|---|---|---:|---|
| scalar | compute | 1 | scalar tokens |
| serial | compute | 1 | automatic |
| parallel | compute | 8 | automatic |
| cached | compute_with_ref | 1 | automatic |
| scratch | compute_with_ref_into | 1 | automatic |
| map_serial / map_parallel | cached RGB diffmap | 1 / 8 | automatic |
| planar_serial / planar_parallel | cached linear planar diffmap | 1 / 8 | automatic |

For revision 7, the serial compute mode exercises the streaming fold with rolling
planes. Cached-reference setup is intentionally absent from that mode. The matrix
is not a claim about every thread count, architecture or image distribution. Runtime
measurements cover this x86 CPU only; ARM/WASM/i686 compilation does not establish
runtime performance on those architectures.

The timing matrix uses all five shipped profiles, 65×63 and 1024×768 checkerboard
pairs, and nine execution modes. Each cell interleaves fresh candidate, pre-refactor
slim and research processes. The first pass has eight process pairs per comparison,
with twelve zenbench rounds per process, plus three fresh-process RSS observations
per binary. The reported ratio is the median paired log-time ratio; intervals are
cellwise 95% bootstrap intervals. The exploratory non-regression margin is 2%, not
an exact-runtime guarantee. Inconclusive cells receive one fixed follow-up batch of 24 additional process pairs;
all original samples remain in the combined report. A separate RSS-only matrix uses
2048×1536 inputs with three processes per binary and route.

The research adapter loads the same stripped revision-7 bakes through its custom
profile API. Its RSS therefore includes that adapter's bake storage; slim embeds
the bakes. Both binaries use the same fixture, timing harness and locked metric
dependencies. Timings exclude fixture/reference creation; peak RSS includes it.

## Results from the frozen audit

All 135 measured profile/size/mode cells retained identical output fingerprints
against both controls. The split gate retained 25,175 identical rows, four prelisted
expected differences, zero unexpected differences and zero missing rows. The full
suite passed 486 tests with zero failures and 15 existing ignored tests; all nine
compile configurations, clippy and the unchanged public API delta passed.

The combined exploratory timing report has 180 comparisons: 166 meet the 2%
upper confidence bound and 14 remain inconclusive. None has a lower confidence
bound above 2%. This does not establish universal performance parity. A separate
fixed-block confirmation completed all 896 blocks and 2,688 invocations for those
14 comparisons, with 64 blocks per case and A/A controls. Every correctness check
passed, but all 14 comparisons were invalidated by the strict resource gate:
522 unclean observations (332 during execution, 190 at boundaries). Of these,
272 exceeded 5% CPU load and 401 reported heavy processes; these counts overlap.
RAM and temperature remained within limits. The run exited 2 and does not resolve
any of the 14 timing uncertainties. Its estimates are retained as invalid evidence,
not used to claim non-regression. A quieter run is required; changing the threshold
or discarding contaminated samples would change the experiment's contract.

Fresh-process RSS is not byte-identical. The largest median ratio against the slim
baseline is 1.053097 on PreviewV0_2's 65×63 scratch route (5,712 versus 5,424 KiB).
At 2048×1536, the largest ratio is 1.011516 on A's parallel planar route (209,748
versus 207,360 KiB). The largest ratio against research is 1.003426 on E's parallel
map route at that size (244,824 versus 243,988 KiB). These are observed process peaks,
not allocation counts, and the research bake-loading adapter affects that comparison.

Resource-wrapper observations: initial matrix `rc=0 1999s | peak-RSS 0.13GiB |
min-avail 23845MiB | peak-load 2.53`; follow-up `rc=0 2551s | peak-RSS 0.10GiB |
min-avail 23842MiB | peak-load 2.89`; large-memory matrix `rc=0 78s | peak-RSS
0.31GiB | min-avail 23607MiB | peak-load 2.61`. Confirmation: `rc=2 4538s | peak-RSS 0.09GiB | min-avail 23461MiB | peak-load 2.60`.
Per-process RSS above comes from
`/usr/bin/time -v`; wrapper peaks describe the supervised runs.

The confirmation launcher registers verified ancestor task IDs using zenbench's
existing launcher environment. Version 0.1.11 excludes ancestor process leaders
but can mistake their threads for rival benchmarks. An initial invalid probe is
retained; no resource gate or timing acceptance criterion was disabled.

Full provenance, commands and artifact locations are in the
[measurement record](../benchmarks/cross_path_2026-10-28.md).
