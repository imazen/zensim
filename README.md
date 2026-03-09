[![CI](https://github.com/imazen/zensim/actions/workflows/ci.yml/badge.svg?style=for-the-badge)](https://github.com/imazen/zensim/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/zensim.svg?style=for-the-badge)](https://crates.io/crates/zensim)
[![docs.rs](https://img.shields.io/docsrs/zensim?style=for-the-badge)](https://docs.rs/zensim)
[![codecov](https://codecov.io/gh/imazen/zensim/branch/main/graph/badge.svg?style=for-the-badge)](https://codecov.io/gh/imazen/zensim)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue?style=for-the-badge)](LICENSE-MIT)

# zensim

Fast psychovisual image similarity metric. Combines ideas from SSIMULACRA2 and butteraugli — multi-scale SSIM + edge + high-frequency features in XYB color space, with trained weights and AVX2/AVX-512 SIMD throughout.

## Quick start

```rust
use zensim::{Zensim, ZensimProfile, RgbSlice};

let z = Zensim::new(ZensimProfile::latest());
let source = RgbSlice::new(&src_pixels, width, height);
let distorted = RgbSlice::new(&dst_pixels, width, height);
let result = z.compute(&source, &distorted)?;
println!("{}: {:.2}", result.profile(), result.score()); // higher = more similar
```

### With imgref (default feature, supports stride)

```rust
use zensim::{Zensim, ZensimProfile};

let source: imgref::ImgRef<rgb::Rgb<u8>> = imgref::Img::new(&src_pixels, width, height);
let distorted: imgref::ImgRef<rgb::Rgb<u8>> = imgref::Img::new(&dst_pixels, width, height);
let z = Zensim::new(ZensimProfile::latest());
let result = z.compute(&source, &distorted)?;
```

`imgref::ImgRef` carries width, height, and stride in one type — no separate dimension arguments, and stride-padded buffers work automatically.

### RGBA

RGBA images are composited over a checkerboard before comparison, so alpha differences produce visible distortion:

```rust
use zensim::{Zensim, ZensimProfile, RgbaSlice};

let z = Zensim::new(ZensimProfile::latest());
let source = RgbaSlice::new(&src_rgba, width, height);
let distorted = RgbaSlice::new(&dst_rgba, width, height);
let result = z.compute(&source, &distorted)?;
```

## Batch comparison

When comparing one reference against many distorted variants, precompute the reference to skip redundant XYB conversion and pyramid construction:

```rust
use zensim::{Zensim, ZensimProfile, RgbSlice};

let z = Zensim::new(ZensimProfile::latest());
let source = RgbSlice::new(&ref_pixels, width, height);
let precomputed = z.precompute_reference(&source)?;
for dst_pixels in &distorted_images {
    let dst = RgbSlice::new(dst_pixels, width, height);
    let result = z.compute_with_ref(&precomputed, &dst)?;
    println!("score: {:.2}", result.score());
}
```

Saves ~25% per comparison at 4K, ~34% at 8K (break-even at 3-7 distorted images per reference).

## Score semantics

100 = identical. Higher = more similar. Score mapping: `100 - 18 × d^0.7` where `d` is the per-scale weighted feature distance (compressive — more resolution at the high-quality end where it matters most).

Scores are calibrated from 0 to 100 on our training data (344k synthetic pairs, q5–q100 across 6 codecs). Extreme distortions can produce scores below 0; the mapping is uncalibrated outside the training range.

`ZensimResult` provides:

| Method | Description |
|--------|-------------|
| `score()` | Similarity score (higher = more similar, typically 0–100) |
| `raw_distance()` | Weighted feature distance before nonlinear mapping (lower = better) |
| `dissimilarity()` | `(100 - score) / 100` — 0 = identical |
| `approx_ssim2()` | Approximate SSIMULACRA2 score (MAE 4.4 pts, r = 0.974) |
| `approx_dssim()` | Approximate DSSIM value (MAE 0.00129, r = 0.952) |
| `approx_butteraugli()` | Approximate butteraugli distance (MAE 1.65, r = 0.713) |
| `features()` | Raw feature vector for diagnostics |
| `mean_offset()` | Per-channel XYB mean shift `[X, Y, B]` |

The `mapping` module provides bidirectional interpolation tables between zensim scores and SSIM2, DSSIM, butteraugli, libjpeg quality, and zenjpeg quality — calibrated on 344k synthetic pairs across 6 codecs.

Results are deterministic for the same input on the same architecture. Cross-architecture scores (AVX2 vs scalar vs AVX-512) may differ by small ULP.

## Profiles

Each `ZensimProfile` variant bundles weights and parameters that affect score output. A given profile produces approximately the same scores across versions, but profiles may be removed in future major versions as the algorithm evolves.

| Profile | Weights | Training data | SROCC |
|---------|---------|---------------|-------|
| `PreviewV0_1` | 228 | 344k synthetic pairs (6 codecs, q5–q100) | 0.9936 (5-fold CV) |
| `PreviewV0_2` | 228 | 218k concordance-filtered pairs (Nelder-Mead) | 0.9960 |

`ZensimProfile::latest()` returns `PreviewV0_2`.

## Input requirements

- **Color space:** All inputs must be **sRGB-encoded** (gamma ~2.2). This is what you get from standard JPEG, PNG, and WebP decoders. If your pixels are linear-light (gamma 1.0), use `PixelFormat::LinearF32Rgba` via `StridedBytes` — zensim will apply the correct transfer function internally.
- **Wide gamut:** Display P3 and BT.2020 inputs are accepted via `ColorPrimaries` on `StridedBytes` — gamut-mapped to sRGB internally. Passing wide-gamut data as sRGB will produce incorrect scores (the metric sees the wrong colors).
- **Pixel formats:** `RgbSlice` (sRGB u8), `RgbaSlice` (sRGB u8 + alpha), `imgref::ImgRef` (sRGB u8, with stride), `StridedBytes` (any of: `Srgb8Rgb`, `Srgb8Rgba`, `Srgb8Bgra`, `Srgb16Rgba`, `LinearF32Rgba`), or implement the `ImageSource` trait directly.
- **Alpha:** RGBA inputs are composited over a checkerboard so alpha differences produce visible distortion. Supports `Straight` and `Opaque` alpha modes.
- **Dimensions:** Both images must be the same width × height, minimum 8×8.

## Performance

Pure-computation benchmarks (no I/O), synthetic gradient images, AMD Ryzen 9 7950X 16C/32T (WSL2). All implementations receive pre-allocated pixel buffers.

### SSIMULACRA2

Threading: zensim and ssimulacra2-rs use rayon (all cores). C++ libjxl and fast-ssim2 are single-threaded. `zensim_st` is zensim with `.with_parallel(false)` for a fair single-threaded comparison.

| Resolution | zensim | zensim_st | C++ libjxl (FFI) | fast-ssim2 | ssimulacra2-rs |
|------------|-------:|----------:|-----------------:|-----------:|---------------:|
| 512x512 | **8 ms** | 11 ms | 45 ms | 39 ms | 251 ms |
| 1280x720 | **14 ms** | 40 ms | 163 ms | 150 ms | 529 ms |
| 1920x1080 | **23 ms** | 90 ms | 389 ms | 338 ms | 997 ms |
| 2560x1440 | **37 ms** | 161 ms | 683 ms | 604 ms | 2,358 ms |
| 3840x2160 | **171 ms** | 499 ms | 2,033 ms | 1,390 ms | 3,763 ms |

Even single-threaded, zensim is **3–4x faster** than fast-ssim2 and **4x faster** than C++ libjxl. Multi-threaded zensim is **12x faster** than C++ libjxl at 4K.

### Butteraugli

Both butteraugli implementations are single-threaded. butteraugli-rs is the imazen pure-Rust port of libjxl's butteraugli.

| Resolution | C++ libjxl (FFI) | butteraugli-rs |
|------------|----------------:|---------------:|
| 512x512 | 72 ms | 60 ms |
| 1280x720 | 304 ms | 253 ms |
| 1920x1080 | 705 ms | 581 ms |
| 2560x1440 | 1,219 ms | 1,027 ms |
| 3840x2160 | 2,446 ms | 2,584 ms |

Benchmarks are in `zensim-bench/` — run with `cargo bench -p zensim-bench --bench bench_compare`.

## Design

- **XYB color space** — cube root LMS, same perceptual space as ssimulacra2/butteraugli
- **Modified SSIM** — ssimulacra2's variant: drops the luminance denominator, uses `1 - (mu1-mu2)²` directly. Correct for perceptually-uniform values where dark/bright errors should weigh equally.
- **4-scale pyramid** — 1×, 2×, 4×, 8× via box downscale (ssimulacra2 uses 6)
- **O(1)-per-pixel box blur** — 1-pass default with fused SIMD kernels
- **228 trained weights** — optimized on 344k synthetic pairs across 6 codecs (mozjpeg, zenjpeg, zenjpeg-xyb, zenwebp, zenavif, zenjxl)
- **AVX2/AVX-512 SIMD** throughout via [archmage](https://crates.io/crates/archmage), with safe scalar fallback

### Feature layout (per channel per scale)

19 features per channel per scale, all scored:

**Basic features (13):**

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | ssim_mean | Mean SSIM error |
| 1 | ssim_4th | L4-pooled SSIM error (emphasizes worst-case) |
| 2 | ssim_2nd | L2-pooled SSIM error |
| 3 | art_mean | Mean edge artifact (ringing, banding) |
| 4 | art_4th | L4-pooled edge artifact |
| 5 | art_2nd | L2-pooled edge artifact |
| 6 | det_mean | Mean detail lost (blur, smoothing) |
| 7 | det_4th | L4-pooled detail lost |
| 8 | det_2nd | L2-pooled detail lost |
| 9 | mse | Mean squared error in XYB |
| 10 | hf_energy_loss | High-frequency energy loss (L2 ratio) |
| 11 | hf_mag_loss | High-frequency magnitude loss (L1 ratio) |
| 12 | hf_energy_gain | High-frequency energy gain (ringing/sharpening) |

**Peak features (6):**

| Index | Feature | Description |
|-------|---------|-------------|
| 13 | ssim_max | Maximum SSIM error |
| 14 | art_max | Maximum edge artifact |
| 15 | det_max | Maximum detail lost |
| 16 | ssim_l8 | L8-pooled SSIM error (near-worst-case) |
| 17 | art_l8 | L8-pooled edge artifact |
| 18 | det_l8 | L8-pooled detail lost |

Total: 4 scales × 3 channels × 19 features = 228 weights. `FeatureView` provides named access to all features.

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `avx512` | yes | Enable AVX-512 SIMD paths |
| `imgref` | yes | `ImageSource` impls for `imgref::ImgRef<Rgb<u8>>` and `ImgRef<Rgba<u8>>` (stride-aware) |
| `training` | no | Expose metric internals for weight training/research |
| `classification` | no | Error classification API (`classify()`, `DeltaStats`, `ErrorCategory`) |

## Workspace crates

| Crate | Description |
|-------|-------------|
| `zensim` | Core metric library |
| `zensim-regress` | Visual regression testing — checksum management, tolerance specs, remote reference storage, amplified diff images, side-by-side montages, and sixel terminal display. See [zensim-regress/README.md](zensim-regress/README.md). |
| `zensim-validate` | Training and validation CLI for weight optimization |
| `zensim-bench` | Performance benchmarks (vs C++ libjxl, fast-ssim2, ssimulacra2-rs, butteraugli) |

### Visual diff images (zensim-regress)

`zensim-regress` generates amplified difference images and comparison montages for debugging visual regressions:

```rust
use zensim_regress::diff_image::*;

// Amplified diff: abs(expected - actual) * amplification_factor
let diff = generate_diff_image(&expected, &actual, 10);

// Side-by-side montage: expected | diff | actual (with border)
let montage = create_comparison_montage(&expected, &actual, 10, 2);

// Raw RGBA byte variants also available
let diff = generate_diff_image_raw(&exp_bytes, &act_bytes, w, h, 10);
```

Auto-save montages on checksum mismatch with `.with_diff_output()`, or display directly in sixel-capable terminals (foot, WezTerm, mintty). See [zensim-regress/README.md](zensim-regress/README.md) for full API docs.

## MSRV

Rust 1.89.0 (2024 edition).

## License

[MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE)

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.
