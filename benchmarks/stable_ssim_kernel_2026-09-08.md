# Stable SSIM moment kernel

Registered after `61fb3360`. The preceding direct-window diagnostic proved
nonlocal floating-point changes in production SSIM on actual training images.
This continuation implements the correction mechanism in the existing Rust
SSIM owner. **No existing feature era or model selects it yet.** The next
required change is versioned extraction and serving integration, followed by
fresh features, fitting/validation and complete inference measurements.

## Algorithm and numerical gates

`ssim_form::stable_ssim_plane` forms four moments from the exact f32 XYB
samples, performing products and all accumulation in f64: source mean,
distorted mean, summed second moments, and squared pairwise error. A
separable row ring implements one reflect-101 box with O(width*radius)
scratch. It forms the error variance directly, avoiding covariance
subtraction in the SSIM numerator. The existing four luminance forms are
supported. Output rounds once to f32; legacy kernels are untouched.

The independent direct, centered f64 window reference is moved from the
existing streaming diagnostic into the SSIM owner and shared by that
diagnostic and synthetic tests. It retains its separate raw-algebra control.
The candidate uses a sliding recurrence; the reference recomputes each
window directly. They are independent algorithms with an explicit parity bar.

Registered per-signal and pooled mean/L2/L4 accuracy:
`absolute error <= 2e-10 + 2e-6*abs(reference)`. The same bound applies to
out-of-support movement. Require finite values on the tested range and exact
synthetic identity. Keep every prior numerical diagnostic field unchanged;
no new candidate scores are inferred from substituting reference features.

All four previously registered real cases pass, before and after their
pixel intervention, at four scales and three channels (96 plane checks).
Maximum base-signal discrepancy is 1.490e-8; maximum pooled discrepancy
across both phases is 3.191e-10. Out-of-support movement is exactly zero in
these cases, versus the legacy diagnostic's approximately 0.001 peak.
Every prior numerical diagnostic field remains exact.

Synthetic tests cover five geometries, radii 0/1/5 and four luminance forms:
60 distortion configurations, 60 exact-identity controls and 15 analytic
constant-offset controls. Tiny/reflected dimensions, scalar/vector tails,
output extent and scratch reuse pass. An **isolated ignored test** exercises
all ten supported SIMD-token permutations; all 60 distortion output arrays
are bit-identical. It must run alone because dispatch changes are global.

## Cost and optimization

The initial correct scalar implementation is too expensive to add to model
inference. Its inner boundary lookup performs a modulo even for in-range
coordinates, and it lacks the repository's existing SIMD dispatch. Register
and apply an in-range lookup plus `archmage::autoversion`. The complete real
numerical report remains byte-for-byte equivalent after JSON parsing.

Three planes, radius five, one worker pinned to CPU 8 on Ryzen 9 9950X3D;
release build without target-cpu=native, two warmups and 30 measured rounds:

| Geometry | Scalar mean | Optimized mean | Optimized min–max | Kernel scratch |
|---|---:|---:|---:|---:|
| 1024² | 28.730 ms | 9.578 ms | 9.275–11.974 ms | 384 KiB |
| 2048² | 112.230 ms | 37.816 ms | 37.588–38.657 ms | 768 KiB |

These are kernel microbenchmarks on fixed synthetic XYB planes. They do not
include conversion, pyramid construction, other feature families, model
inference, maps or codec work. They are not full-inference, RSS or strict
quiet/p95 qualification. The artifact retains every sample and run-heavy
resource records, including the rejected expensive initial mechanism.

## Required integration

Use an explicit feature-arithmetic revision and metadata refusal contract.
Replace the old SSIM moment work where possible; simply adding this pass to
all old passes risks the complete-model latency budget. The current fused
v1 band owner is `fused_vblur_features_ssim`, with callers in the streaming,
folded-feature and attribution owners. Preserve one canonical signal source
for basic/peak/weighted SSIM pools and retained spatial planes. Do not silently
reinterpret `sigma12` as error energy: existing weighted-pool and v2 consumers
still rely on covariance semantics. Explicitly reject any new-era route until
its complete extraction and spatial contract is implemented.

Pay particular attention to fixed 32-row band/halo geometry and full/cached/
streaming parity when integrating the recurrence. Bind scalar features and
retained signals to the same arithmetic. Check all active feature IDs and
all affected routes before refreshing training data; do not relabel old
bakes or combine old and corrected feature columns. Reuse this validated
kernel and reference rather than implementing another precision candidate.

The full goal remains incomplete: numerical correction is not yet selected
by a served model, and corruption protection, native spatial RD, actual
codec targeting, HDR and complete release qualification remain required.
No scalar model or public API changes here.

Artifacts: `/mnt/v/output/zensim/stable-ssim-kernel-2026-09-08/`, mirrored in
the September 8 `stable-ssim-kernel` validation delivery.

Final verification: the source-bound final test binary reproduces the entire
numerical report byte-for-byte and passes the isolated ten-tier check. Its
fresh 30-round timing receipt measures 9.432/38.028 ms at 1024²/2048²
(min–max 9.143–11.680 / 37.916–38.590 ms). The preceding optimized samples
remain separate; no favorable runs are selected or pooled across binaries.
433 library tests pass (seven ignored), as do CI-exact Clippy, scoped
formatting and the 605-script lint. The numerical and SIMD ignored tests
were explicitly run; other ignored tests are not claimed as passed.

## September 9 integration registration

Reuse decision: inspected the current sibling fast-ssim2 sources and its later
August 31 notes. `blur/gaussian.rs` implements an f32 recursive Gaussian;
`simd_ops.rs` forms SSIM from f32 covariance subtraction. Neither provides the
stable reflect-101 box moments required here. Keep one correction owner in
`zensim::ssim_form`; do not change the independent SSIMULACRA2 judge.

Authorized API delta, registered before editing: add `FormulaRevision::Rev3`
for the existing training extraction and `BakeScorer` callers. Rev3 inherits
Rev2 and replaces the v1 SSIM signal with stable moments. Metadata value 3
and `ZENSIM_FORMULA_REV=3` must agree for pixel serving, including maps.
All basic, peak, masked and IW SSIM pools must consume the same corrected
signal. Existing default/model bytes stay Rev1; old bakes cannot be relabeled
as evidence of a refit. Reuse existing pooling/reduction and fixed band geometry.

Acceptance: independent numerical controls; fused versus retained and folded
versus streaming consumed-feature parity; weighted-pool controls; reject bake/
process mismatch even when both select Clamp. Run the existing candidate
surface on the corrected path before new fitting. Changed-era data must be
re-extracted. Existing product scorecard remains the release contract.

## September 9 integration — LANDED (issue #61)

The registration above described what integration would require. This section
records what integration actually did, and what it did not.

### Landed

`FormulaRevision::Rev3` selects `stable_ssim_plane` for the v1 SSIM signal.
`fused_vblur_features_ssim` forms the plane once per band and every consumer
reads it: the basic/peak tiers inside the fused kernel, the retained `sd`
plane (diffmap, attribution, folded replay), and the weighted pools through
three new reducers — `simd_ops::ssim_signal_inline_both` / `_mask` / `_iw`.
Those three copy their weights, `.max(0)` floor, d/d2/d4 tiers and f64
accumulation order verbatim from the legacy trio, so the only difference
between a legacy pool and its Rev3 counterpart is where `d_raw` comes from.
The shipped Rev1/Rev2 reducers are byte-for-byte unchanged.

Under Rev3 the streaming path also STOPS V-blurring `sigma1_sq` / `sigma12`
for the masked/IW block. That is not only a saving: running it would hand the
weighted pools a different `d_raw` than the basic pools already consumed,
which is the single-signal property this era exists to establish.

### Route contract

Rev3's support is exactly one reflect-101 box of `blur_radius`, so
`blur_passes != 1` — which selects `process_strip_channel`'s separate
blur+reduce fallback, whose halo is `passes * radius` — is NOT served.
`ssim_form::check_route` returns `ZensimError::ModelForwardFailed` naming both
the route and the revision, from every fallible entry that builds a
`ZensimConfig` (profile-based entries in `metric`, `attribution`, `diffmap`,
`corruption_head`, plus the two `pub fn`s that take a caller-built config).
The refusal is a `Result` on the caller's thread; the strip walk keeps only a
`debug_assert!` recording that the gate ran.

Observed while auditing that gate, NOT acted on: the `blur_passes != 1`
fallback calls `box_blur_1pass_into` — one pass — while `process_scale_bands`
sizes its halo at `passes * radius`. Whatever `blur_passes = 2` or `3` means
today, it is not "two or three box passes" in the streaming path. That is a
pre-existing question for whoever owns that route.

### Measured locality, on the integrated route

Not the standalone kernel: the real banded strip walk, retaining
`AttrScaleRetention::sd` at four scales and three channels. Fixture is a
deterministic document/screenshot pair (flat paper, hard glyph edges, one
smooth patch, near-lossless distortion) at 192x288, so scale 0 spans three
128-row strips and scale 1 spans two. A rectangle of the distorted image is
replaced with REFERENCE pixels; every signal whose own reflect-101 window
contains no changed sample must be unchanged.

| revision | out-of-support signals moved | peak abs delta |
|---|---:|---:|
| 1 (shipped) | 8,293 | 4.886e-4 |
| 3 | **0** (serial and rayon) | 0 |

The revision-1 row is a committed test of its own
(`locality_fixture_reproduces_out_of_support_movement_on_the_shipped_revision`)
so the revision-3 row cannot pass on an inert fixture.

Retained planes against the whole-plane canonical kernel: 220,320 signals,
worst |delta| **5.821e-11**, inside the registered
`2e-10 + 2e-6*|reference|` acceptance. Non-zero because the strips genuinely
re-seed the vertical recurrence; within bound because that re-seeding is
stable.

### Revision agreement

`check_pixel_revision` compares the REVISION, not only the luminance form it
selects. Revisions 2 and 3 both select `Clamp`, so the form comparison alone
would have served revision 2 coefficients against revision 3 pixels. A Rev3
process refuses bakes declaring 1 or 2 and serves one declaring 3; the shipped
process refuses a bake declaring 3, so an old bake relabelled `3` cannot stand
in for a refit. Fold-vs-streaming parity was re-established by re-running the
two existing `folded720_v1_*` gates in a Rev3 process, not by copying them.

Revision-specific tests own their process (`active_revision` is a `OnceLock`):
`ssim_form::run_at_revision` / `rerun_tests_at_revision` re-execute the test
binary with `ZENSIM_FORMULA_REV` set, and require a sentinel (or a test count)
from the child so a filter that matched nothing fails instead of passing
vacuously. That guard fired for real: the fold-parity wrapper first reported
`matched 0 tests`, because those gates are `training`-feature-gated.

### Cost, MEASURED — and it is a problem

Paired A/B through the repo's existing interleaved instrument
(`zensim/benches/extract_paths_bench.rs`), driven by
`scripts/bench/rev3_cost_ab.sh`. ONE binary built once (a rebuild alone has
moved a 2304 squared timing ~10% here), alternating revision-1 / revision-3
blocks so drift is shared, two blocks per revision, `RAYON_NUM_THREADS=1`,
`taskset -c 8`, plain release without target-cpu=native. `ZENSIM_FORMULA_REV=1`
and `=3` are the same byte length. `fast_ssim2` is revision-independent and
serves as the cross-block anchor. Medians of the two blocks per revision.

Quality of the measurement first, because it decides whether the rest is
readable: the revision-1 A-A replicate spread is **0.06%** at 1024 squared and
**1.26%** at 2048 squared on the anchor, and the anchor moves **-3.4% / -0.3%**
between revisions. 2048 squared is the trustworthy geometry (arm cv 0.8-11%
versus 9-14% at 1024 squared).

| arm @2048 squared | rev 1 | rev 3 | delta |
|---|---:|---:|---:|
| `fast_ssim2` (anchor) | 294.05 ms | 293.28 ms | -0.3% |
| `buf_v1_372` | 183.62 ms | 226.07 ms | +23.1% |
| `fold944_full` | 265.06 ms | 336.48 ms | +26.9% |
| `fold944_off` | 224.61 ms | 297.42 ms | +32.4% |
| `fold372_full` | 130.45 ms | 196.77 ms | +50.8% |
| `buf_v1_228` | 115.47 ms | 176.44 ms | +52.8% |
| `fold228_classc` | 88.97 ms | 154.34 ms | +73.5% |
| `fold228_peaks` | 86.66 ms | 152.44 ms | +75.9% |
| `fold228_moments` | 86.81 ms | 152.85 ms | +76.1% |
| `fold156_basic` | 82.31 ms | 154.24 ms | +87.4% |

The ordering is the mechanism: the f64 pass is a fixed addition, so the LEANER
the walk the larger the relative cost. `fold156_basic` does the least other
work and pays +87%; `fold944_full` does the most and pays +27%.

**The clause this breaks.** At revision 1, `fold944_full` (265.1 ms) is FASTER
than `fast_ssim2` (294.1 ms). At revision 3 it is **336.5 ms — slower**. The
scorecard's scalar-performance row requires "<= fast-ssim2 p95 on the same
inputs". As integrated, this correction flips zensim from beating the
independent judge to losing to it at 2048 squared. That is reported, not
softened: the dead-moment removal below is required work, not headroom.

What these numbers are NOT: they are extraction arms, not complete inference,
and they cannot be read against the absolute 50/200 ms bar. That bar is written
for a QUALIFIED CANDIDATE, and no revision-3 candidate exists because nothing
has been refit. The revision delta is what this measurement can attribute.

**Memory** (`/usr/bin/time -v` max RSS, one arm per process via the bench's
`ZEN_XP_RSS` mode, `scripts/bench/rev3_rss.sh`) is a non-issue:

| arm | size | rev 1 | rev 3 | delta | rev 3 bytes/pixel |
|---|---|---:|---:|---:|---:|
| `buf_v1_372` | 1024 | 48,904 KB | 49,648 KB | +744 KB | 48.5 |
| `fold372_full` | 1024 | 34,704 KB | 35,464 KB | +760 KB | 34.6 |
| `fold944_full` | 1024 | 53,960 KB | 54,968 KB | +1,008 KB | 53.7 |
| `buf_v1_372` | 2048 | 180,780 KB | 181,996 KB | +1,216 KB | 44.4 |
| `fold372_full` | 2048 | 77,688 KB | 80,048 KB | +2,360 KB | 19.5 |
| `fold944_full` | 2048 | 116,692 KB | 118,248 KB | +1,556 KB | 28.9 |

0.7-3.0 MB incremental, 0.7-3.0%, every arm far under the 128 bytes/pixel
clause. That matches the design: one strip-sized f32 signal plane plus the
kernel's O(width x radius) f64 row ring, ~1.8 MB at 2048 wide.

**A discarded run, and what it measured.** The first A/B was thrown out for
contention: a stray diagnostic process survived a `pkill -x` (the kernel
truncates process names at 15 characters, so the pattern matched nothing) and
a follow-up `kill` hit the wrong pid. Comparing the discarded revision-1 block
against the clean one is itself a result: medians agree to ~3% at 1024 squared
and ~0.5% at 2048 squared, because zenbench's exclusive lock made the second
process SLEEP rather than compete — contention doubled wall-clock (1286 s ->
648 s) without moving the timings. The discard was therefore conservative
rather than necessary, and is kept as
`rev3-cost-ab2-CONTENDED-DISCARDED`. "Contended or incomplete timing cannot
pass" is not a standard one gets to evaluate after seeing the numbers.

### Cost headroom, and why the obvious version of it is NOT free

Under revision 3 on the v1 strip path, `sigma_sq` and `sigma12` are computed
and thrown away: `ssim_dissim` is bypassed, `store_sigma` is false there, and
the masked/IW block's two sigma V-blurs are already skipped. That is 2 of the 4
planes in `fused_blur_h_ssim` plus 2 of the 4 running V-blur accumulators,
produced for no consumer. The folded path writes them through `store_sigma`
into the same dead end.

**The one-line version does not work.** `blur::fused_blur_h_mu` already exists
and computes exactly the two planes that are still needed — but it is NOT
bit-equivalent to `fused_blur_h_ssim`'s mu output. VERIFIED by reading
`blur.rs`'s own `h_entries_are_bit_exact_at_a_degenerate_last_column_tile`:
`fused_blur_h_mu`'s SCALAR TAIL carries a known, era-locked 1-2 ulp divergence
from its own vector body (`sum += add - rem` there vs `(sum + add) - rem` in
the group loops), deliberately left unfixed because correcting it would move
v1's shipped bytes; `fused_blur_h_ssim`'s tail is already consistent.

So swapping the H blur would move `mu1`/`mu2` at ragged heights, and through
them the activity map, the masked/IW weights and the edge features — i.e. slots
OUTSIDE `v1ssimstable`'s registered 132. `rev3_moves_exactly_the_registered_slots`
would fail, correctly. Recovering the H-side saving means a two-plane blur with
`fused_blur_h_ssim`'s tail semantics, not a call-site substitution; the V-side
saving (skipping the two accumulators when the sigma planes have no consumer)
is independent of that and does not touch `mu`.

### Registered spatial cells, replayed at both revisions

`scripts/bench/rev3_spatial_replay.sh` re-runs all 23 registered coherence
cells from `nonmax-diagnosis-2026-09-08/COMMANDS.json` through the in-tree
`diffmap_block_coherence` example, at revision 1 and revision 3, same bakes,
same rectangles, same block sizes. 46 runs, 0 non-zero exits.

**Read the caveat before the numbers.** Those bakes were fit against revision-1
features. Scoring them at revision 3 is a cross-era measurement by
construction, so the driver arms the `cross-revision-diagnostic` bypass;
`BakeScorer` refuses it otherwise. All 23 revision-3 runs carry the
"CROSS-REVISION DIAGNOSTIC" stderr line and none of the revision-1 runs do, so
every result is self-identifying. **What this measures is what the extraction
change does to a FIXED model. It is not model quality and not a
qualification.** A revision-3 candidate does not exist.

| bar | revision 1 | revision 3 |
|---|---:|---:|
| M2 >= 0.99 | 16 / 23 | **19 / 23** |
| M3a >= 0.70 | 7 / 23 | **20 / 23** |

M3a (attribution density) is where the correction shows, and it shows exactly
where the diagnosis predicted — the SMALL-block cells, which is where a
per-pixel signal that moves outside its own support does the most damage:

| cell | M3a rev 1 | M3a rev 3 |
|---|---:|---:|
| `row220-b8` | 0.1761 | 0.5821 |
| `row223-b8` | 0.2826 | 0.5701 |
| `row73-b8` | 0.3098 | 0.7540 |
| `row94-b8` | 0.3396 | 0.8438 |
| `row199-b8` | 0.4073 | 0.8122 |
| `row220-b16` | 0.3054 | 0.7500 |
| `row94-b16` | 0.3728 | 0.7509 |
| `row73-b16` | 0.4210 | 0.8186 |

M2 (linear coherence) is roughly flat and MIXED, not uniformly better: three
cells cross the bar, and four move down — `row199-b32` 0.9845 -> 0.9370,
`row202-b8` 0.9924 -> 0.9742, `row160-b32` 0.9998 -> 0.9988, `row34-b32`
0.9368 -> 0.9333 (that one fails at both revisions). Reporting the regressions
matters more than the headline: a fixed revision-1 model priced against
revision-3 features has no reason to improve monotonically, and it did not.

This is consistent with the issue's own finding that a saved-data oracle
replacing only the SSIM predictions resolved its failing cells, and it is the
first evidence that the numerical correction — not an oracle — moves the
spatial screen. It is not evidence that the spatial bars are MET by a
revision-3 candidate; only a re-extraction and a refit can say that.

### NOT done, and load-bearing

- **No model is trained on corrected features.** Every stored table and every
  shipped bake is a revision 1 artifact. The default stays revision 1.
- The `ZENSIM_FORMULA_REV` research pin does NOT make a built-in profile's
  own bake revision-checked — that contract lives in `BakeScorer`. A built-in
  profile declares no revision, so it IS revision 1, and pinning revision 3
  prices those coefficients against another era's features. Refusing would
  break extraction (the same call emits the features), so the profile path
  warns once per process on stderr. Use the pin to EXTRACT; use `BakeScorer`
  to score.
- Registered spatial cells have not been replayed on the corrected path, so
  nothing here says the 15 failing coherence cells now pass.
- No codec RD, HDR or product qualification.
