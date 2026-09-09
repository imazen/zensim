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
