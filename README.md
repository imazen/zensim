[![CI](https://github.com/imazen/zensim/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/zensim/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/zensim.svg)](https://crates.io/crates/zensim)
[![docs.rs](https://docs.rs/zensim/badge.svg)](https://docs.rs/zensim)
[![codecov](https://codecov.io/gh/imazen/zensim/branch/main/graph/badge.svg)](https://codecov.io/gh/imazen/zensim)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# zensim

Fast psychovisual image similarity metric. Combines ideas from SSIMULACRA2 and butteraugli — multi-scale SSIM + edge/texture features in XYB color space, with trained weights and AVX2/AVX-512 SIMD throughout.

## Quick start

```rust
// Compare two sRGB images
let result = zensim::compute_zensim(&src_pixels, &dst_pixels, width, height)?;
println!("score: {:.2}", result.score); // 0-100, higher = more similar
```

```rust
// Compare two RGBA images (composited over checkerboard)
let result = zensim::compute_zensim_rgba(&src_rgba, &dst_rgba, width, height)?;
```

## Batch comparison

When comparing one reference against many distorted variants, precompute the reference to skip redundant XYB conversion and pyramid construction:

```rust
let precomputed = zensim::precompute_reference(&ref_pixels, width, height)?;
for dst in &distorted_images {
    let result = zensim::compute_zensim_with_ref(&precomputed, dst, width, height)?;
    println!("score: {:.2}", result.score);
}
```

Saves ~25% per comparison at 4K, ~34% at 8K (break-even at 3-7 distorted images per reference).

## Design

- **XYB color space** — cube root LMS, same perceptual space as ssimulacra2/butteraugli
- **4-scale pyramid** — 1×, 2×, 4×, 8× downscale covers most perceptual effects
- **O(1)-per-pixel box blur** — cascade approximates Gaussian; no FFT needed
- **13 features per channel per scale** — SSIM (mean, 4th-power, 2nd-power pooling), edge artifacts/detail loss, MSE, variance loss, texture loss, contrast increase
- **Trained weights** — optimized on 149.5k synthetic image pairs across 4 codecs and 11 quality levels
- **SIMD** — AVX2 and AVX-512 paths via [archmage](https://crates.io/crates/archmage), with safe scalar fallback

## Input requirements

- **Color space:** sRGB. Future versions may support additional input color spaces.
- **Pixel format:** `[u8; 3]` (RGB) or `[u8; 4]` (RGBA with straight alpha). RGBA inputs are composited over a checkerboard so alpha differences produce visible distortion.
- **Dimensions:** Both images must be the same width × height, minimum 8×8.

## Score semantics

Scores range 0–100. 100 = identical. `ZensimResult::raw_distance` gives the weighted feature distance before nonlinear mapping (lower = more similar).

Results are deterministic for the same input on the same architecture. Cross-architecture scores (AVX2 vs scalar vs AVX-512) may differ by small ULP.

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `avx512` | yes | Enable AVX-512 SIMD paths |
| `training` | no | Expose metric internals for weight training/research |

## MSRV

Rust 1.85.0 (2024 edition).

## License

[MIT](LICENSE)

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.
