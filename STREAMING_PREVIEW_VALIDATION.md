# Scale-0-Only Streaming Preview — Empirical Validation

**Question:** For zenjpeg's `target-zq` per-strip AQ controller (#113), is a scale-0-only streaming proxy from zensim a useful gradient signal, given that scale 0 carries only ~6% of the trained-weight mass and the higher scales (1-3) carry ~94%?

**Answer:** Yes — strongly. Ship Option B (Issue #16) with explicit scale-0-proxy semantics.

**Branch:** `worktree-agent-a17b8d1d3e50c7b79` (worktree of `imazen/zensim`).
**Validation example:** `/home/lilith/work/zen/zensim/zensim-regress/examples/scale0_correlation.rs`
**Run log:** `/tmp/scale0_corr.log`

## Methodology

* **Corpus:** 6 synthetic images (2 RGB-noise gradients, 2 mandelbrots at different zoom, 2 color-block grids) + 24 real photos sampled from `/mnt/v/input/zensim/sources` (the project's standard training-source pool, downscaled to ≤768 px on the long side). Total: **30 images**.
* **Distortion:** A block-quantization stand-in for codec output. For each image, 5 distortion strengths were applied (8×8 blocks, step ∈ {4, 12, 32, 64, 128}) producing 150 (image, distortion) pairs. This is the same `jpeg_like` shape as `score_dump.rs` but with stronger per-pixel deltas to avoid quantization saturation at low quality.
* **Canonical metric:** `Zensim::compute(&src, &dst)` — `PreviewV0_1` profile (the embedded one), 4 scales, 228 weights, raw_distance and final score both reported.
* **Scale-0 proxy:** `dot(features[0..39], WEIGHTS_PREVIEW_V0_1[0..39])`. This is the dot product over scale-0 basic features only (3 channels × 13 features) — exactly what a `StreamingScale0` consumer could compute strip-by-strip without buffering for higher scales.
* **Both intra-image SROCC** (across the 5 distortion strengths — gradient direction the controller uses) and **cross-image SROCC** at fixed distortion (calibration question) are reported.
* **Sub-Q2:** A 1024×1024 mandelbrot was sliced into K-row windows (K ∈ {32, 64, 128}); each window's scale-0 proxy was computed independently for two distortion strengths (low: step=12; high: step=48). The signal-to-noise ratio of `mean(proxy_high − proxy_low)` to `stddev(proxy_low)` across windows was used as the per-window stability metric.

## Sub-question 1 — Scale-0 proxy vs canonical correlation

### Intra-image SROCC (gradient direction — the controller's most-load-bearing case)

| Bucket | n | Intra SROCC (raw_dist vs proxy) |
|---|---|---|
| Perfect (1.000) | 23 / 30 | All distortion strengths ranked identically by both signals |
| 0.900 | 6 / 30 | One pair-swap (typically the d8s64↔d8s128 saturation tail) |
| Undefined (NaN) | 1 / 30 | `blocks8_512` — the synthetic block image is unaffected by the distortion (all 5 distortions produce raw_distance = 0). Both signals agree perfectly; SROCC is undefined due to ties. (`blocks16_768` is an alias of this; counted once.) |

**Mean intra-image SROCC ≥ 0.9 on every non-trivial image. 23/30 are exactly 1.0.** This is the metric that matters most for #113's PI controller — when the controller increases AQ aggressiveness, scale-0 and canonical move in the same direction every time on real content.

### Cross-image SROCC at fixed distortion (calibration)

| Distortion | n | SROCC (proxy vs raw_dist) |
|---|---|---|
| d8s4   | 30 | 0.9119 |
| d8s12  | 30 | 0.9048 |
| d8s32  | 30 | 0.9479 |
| d8s64  | 30 | 0.9119 |
| d8s128 | 30 | 0.9640 |
| **Mean** | | **0.9281** |

### Pooled (all 150 pairs)

| Stat | Value |
|---|---|
| Pooled SROCC | **0.9778** |
| Pooled Pearson | 0.9453 |

Note on Pearson < SROCC: the proxy is a monotonic but slightly non-linear transform of the canonical raw_distance (because the trained weights at scales 1–3 saturate at different rates than scale 0). For a PI controller this is fine — the controller cares about sign and ranking, not absolute level.

## Sub-question 2 — Per-window stability

For a 1024×1024 mandelbrot at distortion d8s12 (low) vs d8s48 (high):

| K (rows) | Windows | Sign-agreement (delta > 0) | Mean Δ | Stddev(low) | SNR |
|---|---|---|---|---|---|
| 32  | 32 | 25 / 32 | 3.96e-1 | 1.245e-1 | **3.18** |
| 64  | 16 | 13 / 16 | 3.98e-1 | 1.211e-1 | **3.29** |
| 128 | 8  |  7 /  8 | 4.05e-1 | 1.054e-1 | **3.84** |

The "missing" sign-agreements at K=32 and K=64 (7 of 32 windows at K=32) are all in **flat black regions at the top/bottom of the mandelbrot where the source has zero local structure** — block-quantization there is a no-op (delta = 0, not negative). They are not noise. If we exclude windows where `proxy_low < 0.2` (a flat-content threshold), sign-agreement is **100%** at all K.

**Conclusion:** SNR ≥ 3.0 at K=32 means the signal-from-AQ-strength is 3× the across-window content noise even at the smallest reasonable window. The K=64 case (which is what zenjpeg #113 specs at 4:2:0) has SNR=3.29. This is comfortably enough headroom for a bounded PI controller scaling AQ in [0.5, 2.0].

## Caveats and limits

1. **Absolute calibration is bad.** Pooled Pearson is 0.94, not 0.99 — the proxy is *not* interchangeable with canonical for a "did we hit target zensim score" decision. The encoder still needs the canonical at iteration boundaries (or the buffered Option-A flavor of streaming) for that. Scale-0 is a *direction-of-travel signal between iterations*, exactly what #113 says it needs.
2. **Color-flat content gives a zero-signal proxy.** Same is true for canonical zensim — neither signal is a substitute for thinking when the input has no spatial detail. The controller's saturation counter (Layer 4 in #113) handles this case.
3. **The proxy is missing 94% of weight mass.** For images dominated by coarse-scale artifacts (large-scale ringing, blockiness at scales 2–3), scale-0 will *under-estimate* damage. None of the 30-image corpus tripped this — the SROCC stayed > 0.9 across photos and synthetic — but it's worth a sanity bench on 4:2:0 chroma noise, which is more scale-2/3-dominated, before relying on scale-0 alone for chroma AQ.
4. **The per-window numbers are computed by re-running `compute()` on independent strips** — not via a shared accumulator. The actual streaming implementation will produce slightly different per-window numbers because shared blur/Gaussian context spans strip boundaries. The existing `process_scale_bands` machinery already handles this internally; the API just needs to expose per-strip output.

## Recommendation

**Ship Option B with explicit scale-0-proxy semantics.** The validation answers all three open questions in #113 affirmatively:

* (Q1 from #113 "Open questions") Predicted vs measured per-strip score correlation: scale-0 proxy SROCC ≥ 0.9 intra-image on 29/30 corpus images. Controller has plenty of signal between iterations. R² ≈ 0.89 from Pearson.
* (Q2 from #113) Smallest-stable-window: **K=64 has SNR > 3 — comfortably stable**. K=32 also workable. K=128 not required.
* (#16's Option A vs B): scale-0-only is a strict subset of canonical — no algorithmic surprises, since the existing `process_scale_bands` already produces per-strip `StripChannelAccum`s for scale 0.

### Minimal API (sketch)

The proposed surface in the task brief is fine; I'd refine slightly:

```rust
// in zensim::streaming — gated behind a public re-export, NOT a new feature flag.

pub struct StreamingScale0Proxy<'a> {
    reference: &'a PrecomputedReference, // pyramid built once on source
    weights: &'a [f64; 39],              // WEIGHTS_PREVIEW_V0_1[0..39] by default
    // Per-channel running scale-0 StripChannelAccum (already exists, currently pub(crate))
    accums: [StripChannelAccum; 3],
    cumulative_dot: f64,
    rows_consumed: usize,
}

impl<'a> StreamingScale0Proxy<'a> {
    /// `weights` defaults to the active profile's WEIGHTS[0..39]; callers may
    /// override (e.g. zenjpeg may want a Y-only weight slice for chroma-blind control).
    pub fn new(reference: &'a PrecomputedReference, weights: Option<&'a [f64; 39]>) -> Self;

    /// Push a strip of distorted XYB-linear-planar pixels. Returns the
    /// scale-0 proxy contribution from THIS strip only. The strip rows must
    /// align to a 16-row scale-0 boundary (this is the existing STRIP_INNER).
    pub fn push_distorted_strip_linear_planar(
        &mut self,
        planes: [&[f32]; 3],
        strip_y: usize,
        rows: usize,
        stride: usize,
    ) -> Option<WindowContribution>;

    /// Cumulative scale-0 proxy so far. NOT comparable to canonical zensim — this
    /// is a controller signal, not a quality metric.
    pub fn current_proxy(&self) -> f64;
}

pub struct WindowContribution {
    pub strip_y: usize,
    pub rows: usize,
    /// Dot product of this window's scale-0 features against the weight slice.
    /// Encoder should treat this as a relative AQ-gradient signal, not as a score.
    pub window_proxy: f64,
}
```

Things deliberately omitted from the sketch above:
* No `block_diffmap` per window in v1. The diffmap logic in `diffmap.rs` fuses across scales; a scale-0-only diffmap is a different question and adds API surface that #113 doesn't yet need (the controller adjusts strengths from a scalar, not a per-block map).
* No `finalize() -> ZensimResult`. That is what Option A buffered streaming is for. Option B is explicitly the streaming-control variant. If a caller wants both they pair `StreamingScale0Proxy` with a separate finalize-only `compute_with_ref_into` at strip end.

### Validation test that should ship with the API

Add a regression test in `zensim/tests/` that:
1. Runs `Zensim::compute_all_features` on a 1024×1024 corpus image to extract the canonical scale-0 dot.
2. Runs `StreamingScale0Proxy` strip-by-strip on the same image; sums `window_proxy` across all windows.
3. Asserts the streamed sum equals the canonical dot within FP epsilon (1e-5 relative).

This is a tighter contract than #16's "score within 1e-4" because the scale-0 proxy is fully strip-bandable today — there's no buffered-vs-streamed approximation to defend.

### What this does NOT validate

* Real codec (mozjpeg / jpegli) distortion shapes. Block-quantization is a stand-in and likely *easier* than real codec output (no DCT-domain ringing, no chroma upsampling artifacts). I expect SROCC numbers to slip 2–5 points on real codec output, but stay well above the 0.9 intra-image threshold the controller needs. **Re-run `scale0_correlation.rs` against actual zenjpeg output before shipping #113 PR-D.**
* Chroma-dominant distortion. The corpus is luma-dominant; the proxy may underweight chroma damage that lives in scales 2–3. Spot-check before shipping.

## TL;DR

* Intra-image SROCC ≥ 0.9 on 29/30 corpus images (23/30 = 1.0).
* Pooled SROCC = 0.978, Pearson = 0.945, n = 150.
* K=64 window SNR = 3.3 — stable enough for a bounded PI controller.
* Ship Option B. The minimal API in #16 is the right shape; just be explicit that the streaming output is a *proxy* (controller signal) not a *score* (quality target). Add the strip-band accumulator-equivalence test.
