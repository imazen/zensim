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
| 3, exact f64 second pass (SUPERSEDED, see "Fusion") | 0 (serial and rayon) | 0 |
| 3, fused (SERVED) | 11,163 (serial and rayon) | **4.277e-6** (bound 2e-5) |

The revision-1 row is a committed test of its own
(`locality_fixture_reproduces_out_of_support_movement_on_the_shipped_revision`)
so the revision-3 rows cannot pass on an inert fixture.

Read the fused row honestly: it moves MORE out-of-support signals than the
shipped revision, because the fused form runs on the same f32 sliding sums and
those stay path-dependent. What the direct error moment removes is the
cancellation that made each of those movements large — the peak drops ~114×,
from 4.886e-4 to 4.277e-6. That is the bounded-error acceptance the user
directed ("bounded error is fine, speed above minor flaws"); the exact-zero row
is what it cost +27–87% of extraction to get, and no data was ever extracted at
it.

Retained planes against the whole-plane exact f64 kernel: 220,320 signals,
worst |delta| **3.150e-4** for the fused form (bound 1e-3), versus 5.821e-11
for the superseded exact second pass (bound `2e-10 + 2e-6*|reference|`).
All-equal windows: 216,222 identity windows, 15,135 non-zero, worst residue
**3.689e-6** (bound 1e-5); the exact form left 8.18e-15. These three numbers
are printed by the three `rev3_*` controls in `streaming.rs` under
`ZENSIM_FORMULA_REV=3` and are the registered acceptance.

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

> **SUPERSEDED.** Everything from here to "Registered spatial cells" measures
> the EXACT f64 second-pass form of revision 3, which is no longer served. It
> is kept because each negative result below is what ruled the alternatives
> out; the served form and its cost are in "Fusion" below.

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

### Can the f64 be dropped? NO — measured, 2x2 ablation

The f64 pass is roughly 70% of the cost regression, so the obvious escape is
to ask whether the PRECISION is doing the work or whether the FORMULATION is.
The kernel changed both at once — f64 accumulation, and forming the error
variance directly from a `(a-b)^2` moment instead of recovering it from
`var1 + var2 - 2*cov` — so the shipped kernel cannot answer this about itself.

`ssim_form::ablation_plane` mirrors the kernel's algorithm with the two knobs
independent (`streaming::tests::precision_ablation_separates_f64_from_the_direct_error_form`).
The `f32` mode rounds after every operation, which models f32 EXACTLY rather
than approximately: for `+ - * /` on f32-representable operands, computing in
f64 and rounding once is correctly rounded, since 53 bits exceed the 2p+2 = 50
needed at p = 24. The `(f64, direct)` arm is asserted BIT-IDENTICAL to
`stable_ssim_plane`, so the mirror is not a second implementation drifting.

Same fixture, same reference replacement, same out-of-support definition as
the locality control, over the real XYB pyramid planes:

| arm | out-of-support moved | peak abs delta | max err vs direct-f64 windows |
|---|---:|---:|---:|
| f64, direct `(a-b)^2` (SHIPPED) | **0** | 0 | 1.894e-8 |
| f32, direct `(a-b)^2` | **39,468** | 1.752e-5 | 5.440e-4 |
| f64, cov subtraction | 3,172 | 1.164e-10 | 1.894e-8 |
| f32, cov subtraction (legacy form) | 32,855 | 5.298e-4 | 3.378e-3 |

**The f64 is load-bearing; the formulation is not, for locality.** f32 with the
direct error form moves MORE signals than f32 with covariance subtraction. What
f64 alone buys is the large step — 32,855 to 3,172 moved, peak 5.3e-4 to
1.2e-10 — and the direct form then closes the remainder, 3,172 to 0.

That is the right mechanism on reflection: locality is a property of whether
the sliding sum is REVERSIBLE, which is precision. The reformulation fixes
dynamic range, and that shows up in the accuracy column instead — at f32 the
direct form is 6x more accurate (5.440e-4 vs 3.378e-3) while still being
non-local. Both changes are needed and neither substitutes for the other.

**Consequence for cost: speed cannot be bought with precision.** The remaining
levers are structural, not arithmetic:

1. **Remove the duplicate traversal.** The kernel walks the plane a second
   time with its own H recurrence, V ring and boundary mirroring, over data the
   fused pass already streams. Folding the moments into that walk pays for one
   traversal instead of two. It cannot reuse `mu1`/`mu2` — those are f32 and
   must stay f32 to remain outside the registered blast radius — so the f64
   means must be accumulated independently either way.
2. **Vectorize it.** The inner loop is `[f64; 4]` per pixel (array-of-structs)
   with a per-pixel `mirror()` branch in the hot path. `#[autoversion]` is
   applied but that shape will not vectorize. Splitting interior from boundary
   and going struct-of-arrays is what would let it reach f64x4/f64x8.

### Tiling makes locality STRUCTURAL, not numerical (van Herk / Gil-Werman)

The ablation above says the f64 is load-bearing *for the sliding recurrence*.
It is load-bearing there because the recurrence has unbounded history: the
window sum at `x` is a function of every sample the running sum has passed
over. Resetting periodically bounds that but does not remove it — a change
still perturbs the rest of its own tile.

The two-level decomposition removes it outright. With tiles of exactly the
window diameter `D`, and per-tile `prefix`/`suffix` running sums:

```text
window(x) = suffix[x] + prefix[x + D - 1]
```

`suffix[x]` reads only `x ..= tile end`; `prefix[x + D - 1]` reads only
`next tile start ..= x + D - 1`. Both ranges lie INSIDE the window. So the sum
is a deterministic function of exactly the window's own samples, in an order
fixed by the window's offset against the tile grid — identical window contents
give a bit-identical sum **at any precision**.

MEASURED (`precision_ablation_separates_f64_from_the_direct_error_form`), same
fixture, same out-of-support definition:

| arm | moved | peak abs delta | max err vs direct-f64 windows |
|---|---:|---:|---:|
| f64 slide, direct (SHIPPED Rev3) | 0 | 0 | 1.894e-8 |
| f32 slide, direct | 39,468 | 1.752e-5 | 5.440e-4 |
| f64 slide, cov subtract | 3,172 | 1.164e-10 | 1.894e-8 |
| f32 slide, cov subtract (Rev1 form) | 32,855 | 5.298e-4 | 3.378e-3 |
| **f64 TILED, direct** | **0** | 0 | 1.894e-8 |
| **f32 TILED, direct** | **0** | 0 | 1.588e-4 |
| **f32 TILED, cov subtract** | **0** | 0 | 4.962e-4 |

Every tiled arm is exactly local, at f32, in both formulations. Three
consequences:

1. **Locality stops costing precision.** It is a property of the traversal, not
   of the accumulator width.
2. **The direct `(a-b)^2` form still earns its place — on ACCURACY.** 1.588e-4
   versus 4.962e-4 at f32, a 3x gap, because it removes the dynamic-range
   problem from the error term.
3. **f32 tiled is 21x more accurate than what ships today** (1.588e-4 versus
   Rev1's 3.378e-3) while also being exactly local. It remains 4 orders worse
   than f64 because `variance_sum = E[a^2+b^2] - abar^2 - bbar^2` still
   cancels; that term needs f64 or a centered formulation, and the direct error
   moment is the only one of the four that does not.

Both f64 arms show the SAME 1.894e-8 error, which is the single rounding of the
f64 result to the f32 output — i.e. f64 tiled is exactly as accurate as f64
sliding, while also shortening the serial dependency from the row length to
`D` and making tiles independent (so the prefix/suffix passes vectorize across
tiles instead of being latency-bound).

So there are two candidate kernels, and the conservative one is already a speed
win: **f64 tiled** matches Rev3's accuracy with a shorter dependency chain, and
**f32 tiled** is cheaper again if 1.6e-4 is acceptable for a signal whose job
is spatial steering. Neither has been benchmarked yet; that is the next step,
and no speed claim is made here.

### Periodic stability resets: bound the damage, never remove it

Re-seeding the sliding recurrence exactly every `reset` positions (horizontally
and vertically) is the cheap version of the idea — it keeps the single pass and
just stops a change propagating past its own tile. MEASURED, f32 with the
direct error form:

| reset period | out-of-support moved | peak abs delta |
|---|---:|---:|
| none | 39,468 | 1.752e-5 |
| 64 | 5,971 | 4.411e-6 |
| 32 | 4,732 | 4.098e-6 |
| 16 | 2,159 | 1.088e-6 |
| 8 | 1,129 | 1.088e-6 |
| 4 | 390 | 2.701e-7 |
| tiled (van Herk) | **0** | 0 |

It asymptotes toward zero without reaching it, which is the structural
prediction: a reset bounds how far a change travels, but inside its own tile it
still travels. Reset-every-4 already costs more than the recurrence saves —
re-summing 11 samples every 4 positions — and still leaves 390 moved signals.
Only the decomposition reaches zero.

### The tiled kernel: exactly local, and ~1.8x too slow

`ssim_form::stable_ssim_plane_tiled` is a real (non-test) implementation,
`#[autoversion]`-dispatched, with the same `O(width x radius)` scratch bound as
the sliding kernel. It is BIT-IDENTICAL to `stable_ssim_plane` across the
kernel controls' geometries, radii and luminance forms
(`tiled_kernel_matches_the_sliding_kernel_within_the_registered_bound` asserts
the registered bound and measures worst delta 0.000e0), and exactly local at
f32 as well as f64.

Paired interleaved timing, both arms in one process alternating every round,
three planes, radius 5 (`stable_kernel_traversal_ab`):

| geometry | sliding | tiled | ratio |
|---|---:|---:|---:|
| 1024^2 | 9.199 ms | 21.143 ms | 2.30x |
| 2048^2 | 39.155 ms | 70.608 ms | 1.83x |

Four implementations were measured, and the progression is the finding:

| shape | 1024^2 ratio |
|---|---:|
| row-sized prefix/suffix, per-moment traversals, `x % d` in the inner loop | 3.44x |
| tile-local access, row-sized storage | 1.88x |
| tile-local storage, moments recomputed per pass | 1.99x |
| tile-local storage, moments hoisted | 1.83x (at 2048^2) |

**It plateaus at ~1.8x, and that plateau is the answer.** The remaining gap is
arithmetic, not memory: van Herk does ~12 adds per pixel (prefix, suffix,
combine) against the recurrence's 8 (add, subtract). The advantage it was
supposed to buy — a dependency chain of `D` instead of the row length — buys
nothing, because the sliding recurrence carries FOUR INDEPENDENT moment chains
and an out-of-order core already has enough instruction-level parallelism to
hide the latency. The recurrence was never latency-bound.

Not selected. Kept in tree because it is the only construction that makes
locality precision-independent, which is what any f32 variant would need.

### Which moments need the f64? (per-moment ablation)

The four moments have very different magnitudes — `a`, `b` and `a^2+b^2` are
order 0.5 while `(a-b)^2` is order 1e-6 on near-lossless content — so their f32
drift differs by orders of magnitude and they need not share a width. Bit `k`
set means moment `k` accumulated in f32, sliding recurrence, direct form:

| mask | moved | peak abs delta | max err vs f64 windows |
|---|---:|---:|---:|
| `0001` source mean `a` | **0** | 0 | 3.357e-4 |
| `0010` distorted mean `b` | 23,186 | 7.614e-6 | 3.487e-4 |
| `0100` `a^2+b^2` | 29,114 | 9.358e-6 | 4.511e-4 |
| `1000` error `(a-b)^2` | 7,533 | **5.867e-8** | **4.343e-7** |
| `1011` (only `a^2+b^2` in f64) | 26,166 | 1.118e-5 | 4.059e-4 |
| `1001` source mean + error | 7,482 | 5.867e-8 | 3.355e-4 |

Two things to read carefully:

- **The `0001` zero is CONDITIONAL, not free.** The reference plane is identical
  in both phases of this measurement, so drift in the source-side accumulator
  is common-mode and cancels in the comparison. That property does hold for the
  steering use case — one reference, many distorted candidates — but it buys
  locality only, and costs accuracy (3.357e-4).
- **The error moment is nearly free numerically but not exactly local.** In f32
  it moves 7,533 signals at a peak of 5.867e-8 — 9,000x below what ships today
  (5.298e-4) — while keeping f64-grade accuracy (4.343e-7).

So half the moments could be f32 IF "local" is defined as bounded below the f32
output's own noise rather than exactly zero bits. That is a change to the
acceptance bar and a product decision, not an implementation one; it is
recorded here, not taken. The distorted mean and `a^2+b^2` need f64 either way.

**Net: there is no cheap sliding kernel.** Per-moment precision does not unlock
one.

### A vectorisation attempt that was based on a misread, and failed

Disassembling around the dispatched `v4` body showed a predominantly scalar
instruction mix (44 `vaddsd` against 18 `vaddpd`, 39 `vmulsd` against 2
`vmulpd`), which suggested the four moments — exactly one AVX2 `f64x4` — were
not being vectorised, and that the `mirror` branch and the data-dependent
`if add != rem` were what stopped LLVM forming the vector.

Both obstacles were removed without touching the arithmetic: the interior of a
row needs no reflection, so the loop splits into boundary / interior /
boundary; and the guard became a branchless select, which chooses the same
value it always did (when `add == rem` the old sum wins, which is the point —
`(s + a) - a` is not exactly `s` in f64).

It was BIT-IDENTICAL, and it was SLOWER: 2048^2 went 38.5 -> 41.9 ms and
1024^2 went 9.2 -> 10.0 ms, a 7-9% regression. Reverted;
`stable_ssim_plane` is unchanged.

Two things to carry forward, the second more important than the first:

- The branchless select is a real cost, not a wash. On the flat content this
  kernel is for, `add == rem` is common and the branch is well predicted, so
  removing it replaces a predicted branch with unconditional work.
- **The premise was not soundly measured.** `perf` is broken on this box
  (missing `libpython3.10.so.1.0`), and the `#[arcane]` wrapper means the
  kernel body is a closure whose disassembly could not be cleanly isolated from
  its neighbours in a fixed address window — the instruction counts were
  identical before and after a restructure that certainly changed the source,
  which is itself evidence the window was not measuring only the loop. The
  scalar-codegen claim should be treated as UNVERIFIED until a working profiler
  or `cargo asm` on the isolated symbol confirms it.

Fix the profiler before optimising this kernel again.

### Where the cost actually is

Three things have now been measured and none of them is the cost:

- the dead sigma moments (V-side removal: <=1%, inside build noise);
- the accumulator width (f64 is load-bearing for the sliding recurrence);
- the traversal structure (tiling is exactly local but 1.8x slower).

What remains is the thing none of them touch: **there is a SECOND TRAVERSAL of
the plane at all.** The stable kernel re-walks data the fused H/V pass already
streams, with its own boundary handling and its own row ring. Folding the two
new moments into that existing walk is the only lever left that removes work
rather than re-shaping it. It cannot reuse `mu1`/`mu2` — those are f32 and must
stay f32 to remain outside the registered blast radius — so the f64 means still
have to be accumulated separately; the saving is the traversal, the loads and
the boundary work, not the arithmetic.

### The V-side sigma guard: TRIED, MEASURED, REVERTED

Under revision 3 the `sigma_sq` / `sigma12` moments have no reader on the v1
strip path, so the obvious first move is to stop accumulating them in the
V-blur. That was implemented across all four tier variants — 18 vector/scalar
accumulation runs plus the neon/wasm/scalar variant's array-based ring-fill and
slide — behind `need_sigma = store_sigma || stable_sd.is_empty()`, derived from
existing parameters so revisions 1 and 2 are untouched by construction. All 452
lib tests passed, including the folded-vs-streaming bit-parity gates and the
cross-revision blast-radius gate.

**It bought nothing measurable, and it is reverted.** Re-running the same
paired A/B on the rebuilt binary, with the revision-1 arms as cross-build
anchors (the change cannot affect them) and `fast_ssim2` as an absolute anchor
(the change cannot even reach it):

| arm @2048 squared | rev1 before -> after | rev3 before -> after |
|---|---:|---:|
| `fold944_full` | 265.06 -> 264.97 ms (0.0%) | 336.48 -> 332.74 ms (-1.1%) |
| `fold156_basic` | 82.31 -> 80.58 ms (-2.1%) | 154.24 -> 148.16 ms (-3.9%) |
| `fast_ssim2` | 294.05 -> 290.41 ms (-1.2%) | 293.28 -> 287.18 ms (-2.1%) |

`fast_ssim2` moved -1.2% / -2.1% between builds and the change cannot touch it,
so the revision-3 movements are inside the cross-build noise floor. The
differential attributable to the guard is <=1%. Twenty branch sites in the
crate's hottest kernel is not a fair price for a number that cannot be
distinguished from a rebuild.

**What that tells you, and it is the useful part.** The V accumulators are
register adds over plane data the H pass has just written and left cache-hot;
removing them saves almost nothing. The dead moments cost their money in the
**H pass, which writes two full strip planes**. So the whole saving lives on
the side that is NOT a call-site substitution — see below. Do not re-attempt
the V-side guard.

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

### Fusion: the second traversal is gone, and so is the cost

The lever "Where the cost actually is" pointed at. `blur::fused_blur_h_ssim`
already accumulated `Σa`, `Σb`, `Σ(a²+b²)` and `Σab` per column window; under
revision 3 the `Σab` plane has NO consumer (the dissimilarity is bypassed on
every route that serves the stable signal), so that plane now carries
`Σ(a−b)²` instead — the same two FMAs, on `a−b` — behind an `err: bool` that
`fused_blur_h_ssim` / `fused_blur_h_ssim3` read ONCE per call from
`ssim_form::active_revision()` and thread through all ten tier bodies (18
accumulation sites: the initial window fill and the slide, in v4x, v4, v3 and
the generic tail). `fused::fused_vblur_features_ssim` then forms

```text
loss     = f(mu1 − mu2)                      (the revision-2 Clamp luma form)
ve       = E[(a−b)²] − (mu1 − mu2)²          direct error variance, ≥ 0
d        = loss + (1 − loss) · ve / (var1 + var2 + C2)
```

from the same four f32 planes it always V-blurred (`ssim_form::ssim_direct16`
/ `ssim_direct8`, ten call sites), and the seven v2 dense-kernel sites
(`feature_v2::ssim_d_local` / `ssim_d_local_v`) do the same, so the v2 SSIM
family moves with the v1 family (era `v2ssimstable`, registered alongside
`v1ssimstable`; `Rev3.moved_slots` is the upper bound the 944-layout gate
enforces). The exact kernel `ssim_form::stable_ssim_plane` and its thread-local
scratch are off the served path; it survives as the reference the bounds are
measured against.

**What is given up, exactly.** The f32 sliding sums are still path-dependent,
so locality is bounded, not structural: 11,163 out-of-support signals move by
at most 4.277e-6 (the shipped revision: 8,293 by 4.886e-4, ~114× larger). The
direct moment kills the cancellation, which was the large term; the recurrence
drift it leaves is 1e-6-class. Against the exact kernel the fused plane is
within 3.150e-4 (bound 1e-3), and an all-equal window leaves ≤ 3.689e-6
(bound 1e-5). Those three numbers ARE the registered acceptance
(`streaming.rs`: `REV3_LOCALITY_PEAK_BOUND`, `REV3_ACCURACY_BOUND`,
`IDENTITY_BOUND`), printed by the three `rev3_*` controls under
`ZENSIM_FORMULA_REV=3`.

#### Single-thread cost: parity

Same instrument as before (`scripts/bench/rev3_cost_ab.sh`, one binary,
alternating revision blocks, two blocks per revision, `RAYON_NUM_THREADS=1`,
`taskset -c 8`, `ZEN_XP_WALL_S=300`, no target-cpu=native), on a quiet box
with exactly one bench process — a first attempt sat behind an orphaned bench
binary holding zenbench's exclusive lock and was discarded before it timed
anything. Raw report: `benchmarks/rev3_fused_cost_st_2026-09-09.json`.

| arm @2048 squared | rev 1 | rev 3 FUSED | delta | rev 3 EXACT (superseded) |
|---|---:|---:|---:|---:|
| `fast_ssim2` (anchor) | 291.62 ms | 288.64 ms | -1.0% | 293.28 ms |
| `buf_v1_372` | 178.58 ms | 153.95 ms | **-13.8%** | 226.07 ms (+23.1%) |
| `buf_v1_228` | 111.73 ms | 107.10 ms | -4.1% | — |
| `fold372_full` | 128.63 ms | 123.73 ms | -3.8% | — |
| `fold228_classc` | 86.16 ms | 83.97 ms | -2.5% | — |
| `fold228_moments` | 85.72 ms | 83.94 ms | -2.1% | — |
| `fold228_peaks` | 85.16 ms | 83.81 ms | -1.6% | — |
| `fold944_full` | 263.30 ms | 259.86 ms | **-1.3%** | 336.48 ms (+27%) |
| `fold156_basic` | 80.84 ms | 80.86 ms | 0.0% | 154.24 ms (+87%) |
| `fold944_off` | 223.25 ms | 223.85 ms | +0.3% | — |

The anchor moved -1.0% between revisions, so every fused delta inside ±1.5%
is parity. `fold944_full` at revision 3 (259.9 ms) is again under
`fast_ssim2` (288.6 ms) — the scorecard's "≤ fast-ssim2" clause holds. The one
real saving, `buf_v1_372` -13.8%, is the masked/IW block of the strip walk
(`streaming.rs`, `if need_ssim && stable`): under revision 3 the weighted
pools read the retained `stable_sd` plane inline, instead of V-blurring
`sigma1_sq` and `sigma12` per strip and re-deriving the covariance — two
`box_blur_v_from_copy` sweeps and a plane of arithmetic that the shipped
revision still pays. `fold372_full` -3.8% is the folded walk's share of the
same thing. That block was written for the exact form and is unchanged by the
fusion; the fusion's own contribution is the disappearance of the +27-87%.

At 1024 squared the picture is the same (anchor -1.4%; `buf_v1_372` -11.8%,
`fold944_full` -0.7%, `fold228_classc` +3.5% is the noisiest arm at that
geometry, cv 9-14% as recorded above).

**The legacy path paid nothing for the branch.** Revision-1 arms on the fused
build versus the pre-fusion build (the `err` bool is the only change that can
reach them): `fold944_full` 265.06 → 263.30 ms, `fold156_basic` 82.31 →
80.84 ms, anchor 294.05 → 291.62 ms — all inside the cross-build movement of
the anchor. The unswitch is on a value read once per call; LLVM hoists it out
of every tier body (see the `cargo asm` note below).

#### `cargo asm`: the branch is resolved outside the loops

`cargo asm -p zensim --lib --release --features custom-profiles,feature-regime-v2,threads,training fused_blur_h_ssim_inner_v4x`
(cargo-show-asm 0.2.62; the `_v4x_body::<MU1>` and `_v4x_strided` helpers are
inlined into it, so this one symbol is the whole AVX-512 H pass; 4,443 lines).
Splitting it into labelled blocks and keeping the FMA-bearing ones: every
accumulation loop is present TWICE — one body whose FMAs are fed by `vsubps`
(the `(a−b)²` form: e.g. `.LBB672_188`, 4 FMA / 3 `vsubps`, back-edge) next
to one with two FMAs and no `vsubps` at all (the `Σab` form: `.LBB672_186`,
2 FMA / 0 `vsubps`, back-edge), for the fill loop and the slide loop of each
row-group variant. There is no 8-bit `test`/`cmp` anywhere in the function:
the `err` flag is consumed once, outside the loops, and each loop body is a
straight-line copy specialised on it. That is what "unswitched" means here,
and it is why the revision-1 arms did not move.

#### Eight threads: no regression the instrument can resolve

Same driver with `THREADS=8` (`RAYON_NUM_THREADS=8`, `taskset -c 8-15`, one
CCD), `ZEN_XP_WALL_S=240`. Run twice: ONE block per revision
(`benchmarks/rev3_fused_cost_mt_2026-09-09.json`, table below), then a
TWO-block replication (`benchmarks/rev3_fused_cost_mt2_2026-09-09.json`)
whose point is the A-A spread, reported after the table.

| arm @2048 squared, 8 threads | rev 1 | rev 3 FUSED | delta |
|---|---:|---:|---:|
| `fast_ssim2` (anchor) | 292.13 ms | 293.25 ms | +0.4% |
| `buf_v1_372` | 38.03 ms | 36.28 ms | -4.6% |
| `fold228_moments` | 31.30 ms | 31.11 ms | -0.6% |
| `fold228_classc` | 31.44 ms | 31.62 ms | +0.6% |
| `fold156_basic` | 23.59 ms | 23.78 ms | +0.8% |
| `fold944_off` | 93.68 ms | 94.72 ms | +1.1% |
| `fold228_peaks` | 28.72 ms | 29.12 ms | +1.4% |
| `fold944_full` | 109.10 ms | 111.92 ms | +2.6% |
| `fold372_full` | 55.69 ms | 57.90 ms | +4.0% |
| `buf_v1_228` | 21.88 ms | 23.39 ms | +6.9% |

At 1024 squared (5-26 ms arms) the spread is wider in both directions:
`buf_v1_372` -7.2%, `fold944_full` +6.8%, `fold228_peaks` +14.2%, anchor
-0.3%.

**The two-block replication** (medians of two blocks per revision, 2048
squared): `fold944_full` 114.02 -> 115.13 ms (+1.0%), `fold372_full` 56.09
-> 59.62 ms (+6.3%), `buf_v1_372` 38.42 -> 36.60 ms (-4.7%), `buf_v1_228`
24.11 -> 24.34 ms (+0.9%), `fold156_basic` 25.52 -> 25.57 ms (+0.2%), anchor
307.62 -> 304.86 ms (-0.9%). The single-block arms that looked like movement
did not replicate: `buf_v1_228` went from +6.9% to +0.9%, `fold944_full`
from +2.6% to +1.0%, `fold372_full` from +4.0% to +6.3%.

What the replication actually measures is the instrument. The revision-1 A-A
spread between its two blocks at eight threads is **6.8-17.2%** per arm at
2048 squared (`fast_ssim2` 304.2 vs 325.5 ms, `fold944_full` 111.1 vs
118.7 ms, `fold156_basic` 24.4 vs 28.6 ms) — block 2 of revision 1 ran slower
on EVERY arm including the revision-independent anchor, and no competing
process was identified afterwards. Single-threaded the same A-A spread is
0.06-1.3%. So at eight threads this box resolves ±7-10% at best, every fused
delta above is inside that, and `fold372_full` +6.3% sits inside its own
revision-1 pair's 6.9% spread. The eight-thread verdict is "no regression the
instrument can resolve", and the instrument is the thing to fix before a
threaded claim finer than that is made: the run-to-run drift is a box
property (one CCD, eight rayon workers, whatever else the kernel scheduled
there), not a revision property.

The only revision-3-specific work on the v1 strip path that is NOT
arithmetic-neutral is the retention copy of the canonical plane
(`bufs.stable_sd.extend_from_slice`, one inner-strip memcpy per channel per
scale); that is memory traffic and would show under eight threads before it
shows under one. It is in the round-two list below regardless.

`fast_ssim2` at eight threads is 292 ms — the same as at one — and 81 ms at
1024 squared against 68 ms single-threaded: it does not scale with the pool
here, so as a threaded anchor it only says the box was steady, not that the
arms were.

#### Peak memory: unchanged

`/usr/bin/time -v` max RSS of the same binary, one arm per process, pinned to
cpu 8 (`scripts/bench/rev3_rss.sh`; table committed as
`benchmarks/rev3_fused_rss_2026-09-09.tsv`):

| arm | size | rev 1 | rev 3 FUSED | delta | bytes/pixel (rev 3) |
|---|---:|---:|---:|---:|---:|
| `fold944_full` | 2048² | 116,400 KB | 117,244 KB | +844 KB | 28.6 |
| `fold372_full` | 2048² | 77,512 KB | 78,112 KB | +600 KB | 19.1 |
| `buf_v1_372` | 2048² | 180,964 KB | 180,864 KB | -100 KB | 44.2 |
| `fold944_full` | 1024² | 53,904 KB | 54,296 KB | +392 KB | 53.0 |

The retained canonical plane is one inner strip per channel per scale, and
that is what the +0.4-0.8 MB is; every arm stays far under the scorecard's
128 bytes/pixel clause. The exact second pass's thread-local f64 scratch is
gone with it.

#### Where the time goes now (perf, fused revision 3)

`perf record -F 2999 -g` on `fold944_full` at 2048 squared, one pinned core,
ten iterations, `--no-children` self time, functions ≥ 2%
(`benchmarks/rev3_fused_perf_top_2026-09-09.txt`):

| self | function |
|---:|---|
| 26.3% | `blur::box_blur_v_copy_inner_v4x` |
| 13.1% | `fused::fused_vblur_ssim_inner_v4x` |
| 9.6% | `libc memmove` (AVX-512 erms) |
| 9.0% | `blur::fused_blur_h_ssim_inner_v4x` |
| 8.7% | `feature_v2::dense_block_kernel_era2_entry_v4x` |
| 8.5% | `blur::box_blur_h_inner_v4x` |
| 2.6% | `color::srgb_to_positive_xyb_planar_inner_v4x` |
| 2.6% / 2.5% | `feature_v2::append_block_kernel_entry_{nocross,cross}_v4x` |
| 2.4% | `simd_ops::edge_diff_channel_inline_both_inner_v4` |
| 2.4% / 2.3% | `feature_v2::gradient_block_kernel_entry_{bandvis,plain}_v4x` |
| 2.1% | `simd_ops::ssim_signal_inline_both_inner_v4` |
| 2.0% | `simd_ops::abs_diff_into_inner_v4` |

`ssim_form::stable_ssim_plane` — 22% of the exact form's profile — is absent;
the SSIM arithmetic that replaced it is inside `fused_vblur_ssim` and
`fused_blur_h_ssim`, whose shares are what they were at revision 1. What is
left on the table is not revision-3 work at all, and it is the round-two
list: **`box_blur_v_copy` at 26%** is the activity-map blur plus the fold's
v2-phase whole-window V sweeps (`feature_v2` around the `want_v2` block:
four `box_blur_v_from_copy` calls on planes the H pass just wrote);
**`memmove` at 9.6%** is retention and attribution copies (`bufs.stable_sd`
per strip, the `attr` plane copies); **`box_blur_h` at 8.5%** is the activity
chain's `box_blur_h_into_abs_diff` re-blurring `src` that the fused H pass
already blurred into its `mu1` plane — bit-equality of the two H blurs is the
thing to test before that one is a substitution. None of these move the
registered SSIM slots; all of them are v1/v2 plumbing shared with revision 1.

### Registered spatial cells, replayed at both revisions

`scripts/bench/rev3_spatial_replay.sh` re-runs all 23 registered coherence
cells from `nonmax-diagnosis-2026-09-08/COMMANDS.json` through the in-tree
`diffmap_block_coherence` example, at revision 1 and revision 3, same bakes,
same rectangles, same block sizes. 46 runs, 0 non-zero exits — run once for
the exact second-pass form and again for the FUSED form
(`benchmarks/rev3_spatial_cells_2026-09-09.json`,
`benchmarks/rev3_fused_spatial_cells_2026-09-09.json`); both columns are
below, and the bounded form lands on the same counts.

**Read the caveat before the numbers.** Those bakes were fit against revision-1
features. Scoring them at revision 3 is a cross-era measurement by
construction, so the driver arms the `cross-revision-diagnostic` bypass;
`BakeScorer` refuses it otherwise. All 23 revision-3 runs carry the
"CROSS-REVISION DIAGNOSTIC" stderr line and none of the revision-1 runs do, so
every result is self-identifying. **What this measures is what the extraction
change does to a FIXED model. It is not model quality and not a
qualification.** A revision-3 candidate does not exist.

| bar | revision 1 | revision 3, exact (superseded) | revision 3, FUSED |
|---|---:|---:|---:|
| M2 >= 0.99 | 16 / 23 | 19 / 23 | **19 / 23** |
| M3a >= 0.70 | 7 / 23 | 20 / 23 | **20 / 23** |

M3a (attribution density) is where the correction shows, and it shows exactly
where the diagnosis predicted — the SMALL-block cells, which is where a
per-pixel signal that moves outside its own support does the most damage:

| cell | M3a rev 1 | M3a rev 3 exact | M3a rev 3 FUSED |
|---|---:|---:|---:|
| `row220-b8` | 0.1761 | 0.5821 | 0.5781 |
| `row223-b8` | 0.2826 | 0.5701 | 0.5700 |
| `row73-b8` | 0.3098 | 0.7540 | 0.7527 |
| `row94-b8` | 0.3396 | 0.8438 | 0.8440 |
| `row199-b8` | 0.4073 | 0.8122 | 0.8175 |
| `row220-b16` | 0.3054 | 0.7500 | 0.7505 |
| `row94-b16` | 0.3728 | 0.7509 | 0.7510 |
| `row73-b16` | 0.4210 | 0.8186 | 0.8176 |

The fused column tracks the exact one to a few thousandths on every cell —
the bounded error is invisible at the scale the spatial screen reads.

M2 (linear coherence) is roughly flat and MIXED, not uniformly better: three
cells cross the bar, and some move down. Exact form: `row199-b32` 0.9845 ->
0.9370, `row202-b8` 0.9924 -> 0.9742, `row160-b32` 0.9998 -> 0.9988,
`row34-b32` 0.9368 -> 0.9333 (that one fails at both revisions). Fused form:
`row199-b32` 0.9845 -> 0.9415, `row199-b64` 1.0000 -> 0.9930, `row202-b8`
0.9924 -> 0.9858, and four more by ≤ 0.001 (`row136-b8`, `row139-b32`,
`row160-b32`, `row202-b32`); on M3a only `row160-b32` (0.9611 -> 0.9467) and
`row13-b32` (-0.0008) move down. Reporting the regressions matters more than
the headline: a fixed revision-1 model priced against revision-3 features has
no reason to improve monotonically, and it did not.

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
- The spatial-cell replay ("Registered spatial cells, replayed at both
  revisions") measures a FIXED revision-1 model on corrected pixels; it says
  nothing about a refit model, and there is no refit model.
- No codec RD, HDR or product qualification.
