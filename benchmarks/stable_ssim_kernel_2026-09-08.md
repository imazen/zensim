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
