# Zensim as a JXL Encoder Optimization Target

Reference document for replacing or supplementing Butteraugli with zensim's diffmap in jxl-encoder-rs's quantization loop.

## Context: Why Butteraugli Is Hard to Beat

Butteraugli isn't just a metric bolted onto libjxl — it's co-evolved with the codec:

- **Shared XYB color space.** Butteraugli's perceptual model directly informed JXL's choice of XYB as its native representation. The opsin absorbance matrix, cube-root transfer, and opponent-color decomposition are shared infrastructure.
- **Quantization tables tuned for Butteraugli.** The AC coefficient quantization matrices, the distance-to-global-scale mapping, and the per-channel quality multipliers (x_qm_scale, b_qm_scale) were all calibrated against Butteraugli feedback.
- **AC strategy cost model.** The entropy vs pixel-domain-loss tradeoff in AC strategy selection uses a cost function whose constants were tuned with Butteraugli as the quality oracle.
- **Gaborish + EPF co-design.** The 5x5 pre-sharpening (Gaborish) and the edge-preserving filter (EPF) are tuned to produce reconstructions that score well on Butteraugli, not necessarily on other metrics.

Any replacement metric fights the encoder's DNA. The zensim loop's +7% file size at effort 7 isn't a metric problem — it's a mismatch between what zensim asks for and what the encoder can efficiently deliver.

### Known Butteraugli Failures (from libjxl issue tracker)

Despite co-evolution, Butteraugli-driven optimization causes documented problems:

- **Blurring regression (#3530, #3754).** PR #2836 changed AC strategy selection to use pixel-space Butteraugli feedback, causing excessive smoothing. Alakuijala admitted: "The blurring degradations are likely due to optimized use of the filtering strength — now it is L2 + butteraugli tuned instead of tuning by the human eye."
- **Color shifts (#414).** Blue-channel detail loss from aggressive S-cone spatial frequency exploitation. Butteraugli's model says high-frequency blue is invisible; human viewers disagree for saturated blues.
- **Non-photographic content.** libjxl 0.8.2 produces better visual quality than 0.10.2 for artwork, screenshots, and graphics — a regression caused by Butteraugli-metric-chasing.
- **Alakuijala's own admission:** "I always make the quality-related decisions only by my eyes and more or less ignore the metrics."

These are opportunities. A better optimization signal could fix these regressions.

## Current State

### Branch: `worktree-zensim-strategy-refine` (jxl-encoder-rs)

11 commits ahead of main. Key progression:

```
569e845  feat: add zensim quantization loop (alternative to butteraugli)
4ee54ec  feat: add SSIM2 quantization loop (alternative to butteraugli)
bf0e922  feat: diffmap-guided AC strategy refinement in zensim loop
05de829  feat: use DiffmapWeighting::Trained for zensim loop diffmap
3ea92b9  feat: optimize zensim loop with planar API and calibrated scaling
405ef12  fix: zensim loop NaN safety + tuning improvements
3ffa136  refine: zensim loop sum-preserving redistribution + stacking support
f4c4b1e  docs: add zensim stacking benchmark results (2026-03-07)
ebf65c0  feat: add Δss2 column to quality_compare report
```

### Benchmark Results (2026-03-07)

6 CLIC 1024 images x 3 distances x 5 modes:

| Mode | Description | Δ Size | Δ Butteraugli | Δ SSIM2 |
|------|-------------|--------|---------------|---------|
| e7 | baseline (no loop) | — | — | — |
| e8-bfly | butteraugli loop, 2 iters | **-2.6%** | **-6.9%** | -0.05 |
| e7-zen2 | zensim loop only, 2 iters | +7.0% | -4.0% | **+1.63** |
| e8-zen2 | bfly 2 iters + zensim 2 iters | -0.6% | -5.3% | +0.42 |
| e8-zen4 | bfly 2 iters + zensim 4 iters | +0.3% | -0.1% | +0.66 |

**Key observations:**
- Butteraugli loop wins on RD efficiency (the encoder is tuned for it).
- Zensim alone gives the best SSIM2 improvement (+1.63) but inflates files 7%.
- Stacking (bfly then zensim) shows diminishing returns: 4 zensim iters barely outperform 2.
- Zensim redistributes bits toward perceptually important areas (SSIM2 rises) but fights the encoder's butteraugli-tuned quantization structure.

## Encoder Architecture (jxl-encoder-rs)

### Encode Pipeline

```
sRGB input
  → linear RGB (gamma expansion)
  → XYB (opsin absorbance + cube-root + opponent decomposition)
  → adaptive quantization (per-block masking → quant_field)
  → AC strategy selection (cost model: entropy + pixel-domain loss)
  → forward DCT (per-strategy: 8x8, 16x16, 32x32, 64x64, mixed)
  → coefficient quantization (quant_field × dequant_matrix × qm_mul)
  → entropy coding (ANS + histogram clustering)
  → bitstream assembly
```

Optional refinement loops insert between adaptive quantization and final encoding:

```
  → [butteraugli loop] (effort 8+, 2-4 iterations)
  → [zensim loop] (feature-gated, 2-4 iterations)
```

### DistanceParams

```rust
pub struct DistanceParams {
    pub distance: f32,       // target perceptual distance (1.0 = visually lossless)
    pub global_scale: i32,   // quantization scale (0-255 range)
    pub quant_dc: i32,       // DC-specific quantization
    pub scale: f32,          // global_scale / 65536
    pub inv_scale: f32,      // 1.0 / scale
    pub scale_dc: f32,       // DC scale factor
    pub x_qm_scale: u32,    // X channel matrix scale (2-5)
    pub b_qm_scale: u32,    // B channel matrix scale (2-5)
    pub epf_iters: u32,      // edge-preserving filter iterations (0-3)
}
```

Distance-to-quantization mapping (effort >= 5):
```
global_scale = 0.39 / distance
quant_dc = DC_QUANT / max(0.5*distance, min(distance, effective_dist))
```

### quant_field

Per-block (8x8) quantization values. Two representations:

- **Float field:** `Vec<f32>`, values ~0.3-1.5. Used in optimization loops.
- **Integer field:** `Vec<u8>`, values 1-255. Derived via `clamp(round(qf_float * inv_scale + 0.5), 1, 255)`. Written to bitstream.

The loop modifies the float field; `SetQuantField` recomputes `global_scale` from its median/MAD and converts to integer.

### AC Strategy

19 strategies, gated by effort level:

| Effort | Strategies |
|--------|-----------|
| 5 | DCT8, DCT16x8, DCT8x16, DCT16x16, DCT4x4, DCT2x2, IDENTITY |
| 6 | + DCT4x8, DCT8x4, AFV0-3, DCT32x16, DCT16x32 |
| 7+ | + DCT32x32, DCT64x32, DCT32x64, DCT64x64 |

**Cost model** for selection:
```
total_cost = entropy_cost × entropy_mul + pixel_domain_loss
pixel_domain_loss = IDCT_error_8th_power_norm × channel_multiplier
```

Constants are distance-scaled:
```rust
let ratio = (distance + 0.137) / 1.137;
let info_loss_mul = 1.2 * ratio.powf(0.337);
let zeros_mul = 9.309 * ratio.powf(0.510);
let cost_delta = 10.833 * ratio.powf(0.367);
```

These constants were calibrated against Butteraugli. Recalibrating them against zensim could improve RD efficiency for zensim-guided encoding.

### Reconstruction Pipeline (for metric feedback)

After quantization, the encoder reconstructs the image for comparison:

1. Inverse DCT (per-strategy)
2. Chroma-from-luma interpolation (ytox/ytob)
3. Gaborish smoothing (5x5 blur, compensates pre-sharpening)
4. EPF sharpness filter (per-block adaptive)
5. XYB → linear RGB (reverse color transform)

The reconstructed linear RGB planes feed directly into zensim's planar API.

### Effort Levels and Loop Gating

- **Effort 7 (Squirrel, default):** No iterative loop. AC strategy search + adaptive quant.
- **Effort 8 (Kitten):** Butteraugli loop enabled, 2 iterations.
- **Effort 9+ (Tortoise/Glacier):** Butteraugli loop 4 iterations + Viterbi LZ77.

Zensim loop is feature-gated (`--features zensim-loop`) and can stack after butteraugli.

## Zensim Integration API

### Precomputed Reference Pattern

```rust
let zensim = Zensim::new(ZensimProfile::latest());

// Precompute once from source (linear RGB planes)
let precomputed = zensim.precompute_reference_linear_planar(
    [&src_r, &src_g, &src_b],
    width, height, stride,
)?;

// In each loop iteration, compare against reconstruction
let dm_result = zensim.compute_with_ref_and_diffmap_linear_planar(
    &precomputed,
    [&recon_r, &recon_g, &recon_b],
    width, height, padded_width,
    DiffmapOptions {
        weighting: DiffmapWeighting::Trained,
        masking_strength: Some(4.0),
        sqrt: true,
        include_edge_mse: true,
    },
)?;

let score = dm_result.score();        // 0-100 global quality
let diffmap = dm_result.diffmap();    // [width × height] per-pixel error
```

### Planar Buffer Requirements

- **Layout:** Row-major, separate R/G/B f32 planes
- **Stride:** Elements per row, >= width (encoder may pad for SIMD alignment)
- **Value range:** [0, 1] — clamped internally (reconstruction can produce out-of-range values from quantization error, gaborish overshoot)
- **Color space:** Linear light sRGB (not gamma-encoded, not wide gamut)

### DiffmapResult

```rust
impl DiffmapResult {
    pub fn result(&self) -> &ZensimResult   // full comparison result
    pub fn score(&self) -> f64              // convenience: 0-100
    pub fn diffmap(&self) -> &[f32]         // width × height, row-major, no padding
    pub fn width(&self) -> usize
    pub fn height(&self) -> usize
    pub fn into_parts(self) -> (ZensimResult, Vec<f32>, usize, usize)
}
```

Diffmap values: `[0, +∞)`. Zero = perceptually identical. Typical range for lossy compression: `[0, 0.3]`.

### DiffmapOptions

```rust
pub struct DiffmapOptions {
    pub weighting: DiffmapWeighting,     // channel combination strategy
    pub masking_strength: Option<f32>,   // contrast masking: 2.0-8.0 typical
    pub sqrt: bool,                       // compress dynamic range
    pub include_edge_mse: bool,          // add edge artifact/detail loss/MSE features
}

pub enum DiffmapWeighting {
    Trained,              // derive from profile's 228 weights (default)
    Balanced,             // [X:15%, Y:70%, B:15%], equal scale blend
    Custom([f32; 3]),     // user-specified [X, Y, B], auto-normalized
}
```

### ZensimResult Conversions

```rust
impl ZensimResult {
    pub fn score(&self) -> f64              // 100 - 18 × d^0.7
    pub fn raw_distance(&self) -> f64       // pre-mapping distance
    pub fn features(&self) -> &[f64]        // all 228 features
    pub fn approx_ssim2(&self) -> f64       // MAE ~4.4 points, r=0.974
    pub fn approx_dssim(&self) -> f64       // MAE ~0.00129, r=0.952
    pub fn approx_butteraugli(&self) -> f64 // MAE ~1.65 units, r=0.713
}
```

### Memory Budget

PrecomputedReference stores 3 f32 planes × 4 pyramid scales × ~1.33x (geometric sum):

| Resolution | PrecomputedReference |
|-----------|---------------------|
| 1024×1024 | ~16 MB |
| 1920×1080 | ~33 MB |
| 3840×2160 | ~133 MB |
| 7680×4320 | ~532 MB |

### Performance

| Resolution | Zensim (MT) | Zensim (ST) | Butteraugli (C++) |
|-----------|-------------|-------------|-------------------|
| 1024×1024 | ~40 ms | ~130 ms | ~600 ms |
| 1920×1080 | ~23 ms | ~170 ms | ~389 ms (ST) |
| 3840×2160 | ~171 ms | ~499 ms | ~2,446 ms |

Zensim's box blur is O(1) per pixel vs Butteraugli's IIR Gaussian (~60% of its runtime). SIMD via archmage (AVX-512/AVX2 dispatch).

## Zensim's 228 Weights

### Feature Layout

4 scales × 3 channels (XYB) × 19 features = 228 weights.

Per channel per scale:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | ssim_mean | Mean-pooled SSIM error |
| 1 | ssim_4th | L4-pooled SSIM error |
| 2 | ssim_2nd | L2-pooled SSIM error |
| 3 | art_mean | Edge artifact (ringing/banding), mean |
| 4 | art_4th | Edge artifact, L4 |
| 5 | art_2nd | Edge artifact, L2 |
| 6 | det_mean | Detail loss (blur/smoothing), mean |
| 7 | det_4th | Detail loss, L4 |
| 8 | det_2nd | Detail loss, L2 |
| 9 | mse | Raw MSE in XYB |
| 10 | hf_energy_loss | High-freq energy lost (blur detection) |
| 11 | hf_mag_loss | L1-robust version of energy loss |
| 12 | hf_energy_gain | High-freq energy added (ringing/sharpening) |
| 13 | ssim_max | Peak SSIM error |
| 14 | art_max | Peak edge artifact |
| 15 | det_max | Peak detail loss |
| 16 | ssim_l8 | L8-pooled SSIM (near-worst-case) |
| 17 | art_l8 | L8-pooled artifact |
| 18 | det_l8 | L8-pooled detail loss |

### 4-Scale Pyramid

| Scale | Resolution | Blur Window | What It Captures |
|-------|-----------|-------------|-----------------|
| 0 | 1× (full) | 11×11 | Fine detail, JPEG blocking, slight blur |
| 1 | 1/2× | 22×22 | Mid-frequency structure, banding |
| 2 | 1/4× | 44×44 | Coarse structure, posterization |
| 3 | 1/8× | 88×88 | Global structure, color shifts |

Downscaling: 2×2 box averaging (O(1) per pixel).

### What the Diffmap Uses vs Discards

The global score uses all 19 features. The diffmap only projects a subset:

| Feature | In Diffmap? | Notes |
|---------|------------|-------|
| ssim (0-2) | Yes (always) | Core spatial signal |
| edge artifact (3-5) | Yes (if include_edge_mse) | Ringing/banding detection |
| detail loss (6-8) | Yes (if include_edge_mse) | Blur detection |
| mse (9) | Yes (if include_edge_mse) | Raw pixel error |
| hf_energy_loss (10) | **No** | Blur vs clean: critical for encoder |
| hf_mag_loss (11) | **No** | Robust blur detection |
| hf_energy_gain (12) | **No** | Ringing/sharpening: critical |
| peak features (13-18) | **No** | Outlier sensitivity |

**This is the primary gap.** The encoder can't distinguish "I blurred detail away" from "this area is clean" without HF features, and can't detect ringing without HF energy gain. These are exactly the signals that let Butteraugli guide AC strategy and filtering.

### Diffmap Weight Derivation

`trained_multiscale_weights()` in `diffmap.rs`:

1. For each scale and channel, sum absolute values of the included feature weights.
2. Normalize across all scales/channels to sum to 1.0.

Result (PreviewV0_1 defaults):
- Scale 0: ~6% blend weight, Y-dominant (~99.3%)
- Scales 1-3: ~28-35% each, increased chroma contribution at coarser scales.

This derivation is lossy — it collapses 228 independently trained weights into ~12 blend coefficients. For encoder use, independently trained diffmap weights would be more appropriate.

## How the Loops Work

### Butteraugli Loop (butteraugli_loop.rs)

Per iteration:
1. `SetQuantField` — convert float field to integer via current params.
2. `transform_and_quantize_into()` — forward DCT + quantize.
3. Reconstruct — IDCT + gaborish + EPF → XYB.
4. Butteraugli comparison — produces L16-normed per-tile distances.
5. Adjust quant_field — asymmetric pow-based:
   - Iterations 0-1: `pow=0.2`, adjust both good and bad blocks.
   - Iterations 2+: `pow=0.0`, only increase quality of bad blocks.
6. Enforce deviation bounds from initial field.
7. Last iteration: compare-only (no adjustment).

AC strategy is **fixed** throughout the butteraugli loop.

### Zensim Loop (zensim_loop.rs)

Per iteration:
1. Same encode-reconstruct cycle.
2. Convert XYB reconstruction → linear RGB planes.
3. `compute_with_ref_and_diffmap_linear_planar()` — produces per-pixel diffmap + global score.
4. Compute per-tile L4 norms from diffmap (not L16 — diffmap already has 11×11 spatial smoothing).
5. **Sum-preserving redistribution:**
   ```
   ratio = tile_dist[i] / avg_tile_dist
   qf[i] *= 1 + K_ALPHA × (ratio - 1)      // K_ALPHA = 0.10 (conservative)
   renormalize qf to preserve original sum   // file-size-neutral
   ```
6. Enforce deviation bounds.
7. **AC strategy refinement** (iterations 0-1 only):
   - Scan multi-block transforms (DCT16+) where `tile_dist > 1.3 × target_distance`.
   - `split_one_level()` — DCT64→DCT32→DCT16→DCT8.
   - Recompute tile distances at finer granularity.

Key differences from butteraugli loop:
- **L4 norm** (not L16) — appropriate since diffmap is pre-smoothed.
- **Sum-preserving** — renormalizes quant_field to maintain file size budget.
- **Modifies AC strategy** — can split multi-block transforms where errors are concentrated.
- **Conservative α=0.10** — only 10% of the error ratio applied per iteration.
- **Budget-neutral anchoring** — uses target_distance, not measured distance.

### Stacking

Both loops can run sequentially. Butteraugli first (refines quant_field for RD), then zensim (redistributes for perceptual quality). Feature-gated independently.

```rust
// In encoder.rs
#[cfg(feature = "butteraugli-loop")]
if self.butteraugli_iters > 0 {
    params = self.butteraugli_refine_quant_field(...);
}

#[cfg(feature = "zensim-loop")]
if self.zensim_iters > 0 {
    params = self.zensim_refine_quant_field(..., &mut ac_strategy, ...);
}
```

### Why the Zensim Loop Inflates Files

The +7% size penalty (e7-zen2) has a specific cause: **the quant_field redistribution asks for quality the encoder can't deliver cheaply.**

When zensim's diffmap identifies a perceptually important area, it increases that block's quality (lower qf). But the encoder's quantization tables, dequant matrices, and entropy coder were calibrated so that Butteraugli-important areas are cheap to encode. Zensim-important areas may require more bits because the encoder's infrastructure doesn't have matching optimization.

The sum-preserving redistribution (commit 3ffa136) mitigates this by keeping total quantization budget constant. But redistribution itself has a cost: moving bits from Butteraugli-efficient areas to Butteraugli-inefficient areas reduces overall coding efficiency.

## Paths to Improvement

### 1. Include HF Features in the Diffmap

The diffmap currently discards features 10-18 (HF energy, peak statistics). These are the signals the encoder needs most:

- **hf_energy_loss** (blur detection): tells the encoder "detail was smoothed away here."
- **hf_energy_gain** (ringing detection): tells the encoder "artifacts were introduced here."
- **Peak features** (max, L8): catch the worst local errors without spatial averaging.

The HF residuals are already computed during `process_scale_bands` — they just aren't forwarded to the diffmap accumulation. This is ~10 lines in `streaming.rs`.

### 2. Train Diffmap Weights Independently

Current diffmap weights are derived from the 228 global weights by absolute-value summation. This loses information — the global score can weight ssim_mean differently from ssim_4th, but the diffmap collapses them.

Better: train a separate `DiffmapProfile` that optimizes for **spatial accuracy** (does high diffmap at pixel X predict human-visible error at pixel X?) rather than global correlation. The `zensim-validate` training infrastructure already supports custom objectives.

### 3. Retune AC Strategy Cost Model Constants

The cost model constants (`info_loss_mul`, `zeros_mul`, `cost_delta`) were calibrated for Butteraugli. With zensim as the quality oracle, these constants should be re-derived:

```
For each candidate AC strategy:
  encode with that strategy
  measure zensim diffmap over the tile
  compute cost = entropy + zensim_loss × new_info_loss_mul
  choose minimum-cost strategy
```

This is expensive (requires per-strategy zensim evaluation) but could be approximated by running the full search once on a training corpus and fitting new constants.

### 4. Train Encoder-Aware Diffmap Weights

Instead of training weights for measurement accuracy, train for **encoder convergence**: given the current diffmap, does following its gradient actually reduce the global score on the next iteration?

Training loop:
1. Encode image at distance d.
2. Compute diffmap with candidate weights.
3. Redistribute quant_field proportionally to diffmap.
4. Re-encode with new quant_field.
5. Measure global score improvement.
6. Optimize weights to maximize score improvement per iteration.

This trains the diffmap to point in directions the encoder can actually follow efficiently.

### 5. Calibrate Diffmap to JND Units

Butteraugli's distance scale (0.0 = lossless, 1.0 = visually lossless) gives the encoder a meaningful quality target. Zensim's diffmap values are unitless SSIM-derived quantities with no fixed perceptual interpretation.

Calibrating the diffmap to JND units would let the encoder allocate bits proportionally to visibility, reducing waste on invisible errors and concentrating effort on perceptible ones.

### 6. Address the Encoder's Butteraugli Bias

Several encoder parameters have Butteraugli baked in:

- **Dequant matrices** — optimized for Butteraugli visual quality.
- **x_qm_scale / b_qm_scale** — per-channel quality multipliers.
- **EPF strength** — tuned to improve Butteraugli score.
- **Gaborish kernel** — pre-sharpening compensates for Butteraugli's blur sensitivity.
- **CfL (Chroma-from-Luma)** — chroma prediction residuals sized for Butteraugli's color model.

A full zensim-native encoder would revisit all of these. In the short term, the most impactful target is the dequant matrices and per-channel quality multipliers — these directly control how bits are distributed across XYB channels.

## Key Files

### jxl-encoder-rs (worktree-zensim-strategy-refine branch)

| File | Role |
|------|------|
| `jxl_encoder/src/vardct/zensim_loop.rs` | Zensim quantization loop |
| `jxl_encoder/src/vardct/butteraugli_loop.rs` | Butteraugli quantization loop (reference) |
| `jxl_encoder/src/vardct/ssim2_loop.rs` | SSIM2 quantization loop (third option) |
| `jxl_encoder/src/vardct/encoder.rs` | Loop integration point, feature gates |
| `jxl_encoder/src/vardct/ac_strategy.rs` | AC strategy selection + split_one_level() |
| `jxl_encoder/src/frame.rs` | DistanceParams, quant_field management |
| `jxl_encoder/src/xyb.rs` | XYB color transform |
| `jxl_encoder/src/gaborish.rs` | Pre-sharpening filter |
| `jxl_encoder/src/epf.rs` | Edge-preserving filter |
| `benchmarks/zensim_stacking_2026-03-07.csv` | Stacking benchmark results |
| `docs/ac_strategy_cost_model.md` | AC strategy cost model reference |

### zensim (main branch)

| File | Role |
|------|------|
| `zensim/src/streaming.rs` | Core pipeline, planar API, diffmap computation |
| `zensim/src/diffmap.rs` | DiffmapOptions, weighting, contrast masking |
| `zensim/src/metric.rs` | Feature extraction, 228-weight scoring |
| `zensim/src/profile.rs` | Weight arrays, profile parameters |
| `zensim/src/color.rs` | XYB conversion, gamut mapping |
| `zensim/src/blur.rs` | O(1) box blur |
| `zensim/src/fused.rs` | Fused V-blur + feature extraction |
| `zensim/src/mapping.rs` | Bidirectional mapping to SSIM2, DSSIM, butteraugli |

### Training Infrastructure

| File | Role |
|------|------|
| `zensim-validate/src/main.rs` | Weight training pipeline (4,500+ lines) |
| `coefficient/examples/generate_zensim_training.rs` | Synthetic training data generation |
| `.claude/worktrees/training-algorithms/` | Training algorithm experiments |

Training optimizers: coordinate descent, CMA-ES, RankNet pairwise, FISTA proximal gradient.

## Gotchas

- **Disable zensim's internal parallelism** when used inside the encoder: `zensim.with_parallel(false)`. The encoder already parallelizes at group level; nested rayon causes contention.
- **Reconstruction can produce out-of-range linear RGB.** Negative values near black (quantization error), >1 from gaborish overshoot. The planar API clamps to [0, 1] — this is correct (displays clamp to gamut).
- **NaN safety.** The zensim loop skips non-finite diffmap values in L4 norm computation. If all pixels are non-finite, tile distance falls back to 0.
- **approx_butteraugli() has MAE ~1.65.** Do not use it as a distance anchor — causes +55% file size inflation. Use target_distance directly.
- **Minimum image size:** 8×8 pixels. Below this, zensim errors.
- **The old google/butteraugli repo is a different metric** from libjxl's internal butteraugli. 7+ years of divergence. Scores are not comparable.
