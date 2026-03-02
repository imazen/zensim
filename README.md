[![CI](https://github.com/imazen/zensim/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/zensim/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/zensim.svg)](https://crates.io/crates/zensim)
[![docs.rs](https://docs.rs/zensim/badge.svg)](https://docs.rs/zensim)
[![codecov](https://codecov.io/gh/imazen/zensim/branch/main/graph/badge.svg)](https://codecov.io/gh/imazen/zensim)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# zensim

Fast psychovisual image similarity metric. Combines ideas from SSIMULACRA2 and butteraugli — multi-scale SSIM + edge + high-frequency features in XYB color space, with trained weights and AVX2/AVX-512 SIMD throughout.

## Quick start

```rust
use zensim::{Zensim, ZensimProfile, RgbSlice};

let z = Zensim::new(ZensimProfile::latest());
let source = RgbSlice::new(&src_pixels, width, height);
let distorted = RgbSlice::new(&dst_pixels, width, height);
let result = z.compute(&source, &distorted)?;
println!("{}: {:.2}", result.profile, result.score); // 0-100, higher = more similar
```

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
    println!("score: {:.2}", result.score);
}
```

Saves ~25% per comparison at 4K, ~34% at 8K (break-even at 3-7 distorted images per reference).

## Profiles

Each `ZensimProfile` variant bundles all parameters that affect score output — weights, blur config, and score mapping. Same profile name = same scores forever. The crate accumulates profiles; old ones never change or get removed.

| Profile | Training data | SROCC |
|---------|---------------|-------|
| `GeneralV0_1` | 12k synthetic pairs | — |
| `GeneralV0_2` | 163k synthetic pairs | 0.9857 |

`ZensimProfile::latest()` returns the most recent general-purpose profile.

## Design

- **XYB color space** — cube root LMS, same perceptual space as ssimulacra2/butteraugli
- **Modified SSIM** — ssimulacra2's variant: drops the luminance denominator, uses `1 - (mu1-mu2)²` directly. Correct for perceptually-uniform values where dark/bright errors should weigh equally.
- **13 features per channel per scale** — SSIM (3 pooling norms), edge artifact/detail loss (3 norms each), MSE, and 3 high-frequency energy features
- **4-scale pyramid** — 1×, 2×, 4×, 8× via box downscale (ssimulacra2 uses 6)
- **O(1)-per-pixel box blur** — 1-pass default with fused SIMD kernels
- **156 trained weights** — optimized on 149.5k synthetic pairs across 4 codecs and 11 quality levels
- **AVX2/AVX-512 SIMD** throughout via [archmage](https://crates.io/crates/archmage), with safe scalar fallback

### Feature layout (per channel per scale)

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

Total: 4 scales × 3 channels × 13 features = 156 features, combined with trained weights into a single distance, then mapped to a 0–100 score.

## Input requirements

- **Color space:** sRGB (8-bit, 16-bit, f16) or linear f32.
- **Pixel formats:** `RgbSlice` (sRGB u8), `RgbaSlice` (sRGB u8 + alpha), `StridedBytes` (any of: `Srgb8Rgb`, `Srgb8Rgba`, `Srgb8Bgra`, `Srgb16Rgba`, `SrgbF16Rgba`, `LinearF32Rgba`), or implement the `ImageSource` trait directly.
- **Alpha:** RGBA inputs are composited over a checkerboard so alpha differences produce visible distortion. Supports `Straight`, `Premultiplied`, and `Opaque` alpha modes.
- **Dimensions:** Both images must be the same width × height, minimum 8×8.

## Score semantics

Scores range 0–100. 100 = identical. `ZensimResult::raw_distance` gives the weighted feature distance before nonlinear mapping (lower = more similar).

Results are deterministic for the same input on the same architecture. Cross-architecture scores (AVX2 vs scalar vs AVX-512) may differ by small ULP.

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `avx512` | yes | Enable AVX-512 SIMD paths |
| `training` | no | Expose metric internals for weight training/research |
| `imgref` | no | `ImageSource` impls for `imgref::ImgRef<Rgb<u8>>` and `ImgRef<Rgba<u8>>` |
| `f16` | no | Enable `SrgbF16Rgba` pixel format (IEEE 754 half-precision via `half` crate) |
| `zenpixels` | no | `ZenpixelsSource` for any interleaved format via `zenpixels` conversion (coming soon) |

## MSRV

Rust 1.85.0 (2024 edition).

## License

[MIT](LICENSE)

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.
