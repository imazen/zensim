# zensim [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zensim/ci.yml?style=flat-square)](https://github.com/imazen/zensim/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zensim?style=flat-square)](https://crates.io/crates/zensim) [![lib.rs](https://img.shields.io/crates/v/zensim?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zensim) [![docs.rs](https://img.shields.io/docsrs/zensim?style=flat-square)](https://docs.rs/zensim) [![license](https://img.shields.io/crates/l/zensim?style=flat-square)](https://github.com/imazen/zensim#license)

Perceptual image similarity in 22 ms at 1080p. 18x faster than C++ SSIMULACRA2 at 4K.

Built on the same psychovisual foundations as SSIMULACRA2 and butteraugli — multi-scale SSIM, edge artifacts, detail loss, and high-frequency features in XYB color space — but with trained weights, fused SIMD kernels, and multi-threaded computation.

## Speed

AMD Ryzen 9 7950X 16C/32T (WSL2), synthetic gradient images, no I/O, pre-allocated buffers. zensim and ssimulacra2-rs use rayon (all cores); C++ libjxl, fast-ssim2, and butteraugli-rs are single-threaded. Enabling rayon for fast-ssim2 and butteraugli-rs made them slower at small sizes due to thread-pool overhead, so they're benchmarked single-threaded. Median of 100 samples via criterion.

### SSIMULACRA2 implementations

| Resolution | zensim | zensim (1 thread) | C++ libjxl (FFI) | fast-ssim2 | ssimulacra2-rs |
|------------|-------:|------------------:|-----------------:|-----------:|---------------:|
| 1280x720 | **14 ms** | 39 ms | 249 ms | 111 ms | 545 ms |
| 1920x1080 | **22 ms** | 89 ms | 377 ms | 350 ms | 1,056 ms |
| 3840x2160 | **91 ms** | 366 ms | 1,674 ms | 1,364 ms | 3,980 ms |

### Butteraugli implementations (single-threaded)

| Resolution | C++ libjxl (FFI) | butteraugli-rs |
|------------|----------------:|---------------:|
| 1280x720 | 269 ms | 83 ms |
| 1920x1080 | 647 ms | 154 ms |
| 3840x2160 | 2,688 ms | 906 ms |

Single-threaded zensim is 4x faster than C++ libjxl SSIMULACRA2. Multi-threaded at 4K: 18x.

Reproduce: `cargo bench -p zensim-bench --bench bench_compare` (C++ libjxl FFI requires a local libjxl build; set `LIBJXL_DIR` or let the build script auto-clone it)

## Correlation with human perception

Spearman rank-order correlation (SROCC) on raw perceptual distance against three independent human-rated image quality databases. Higher is better; 1.0 = perfect agreement with human rankings. None of these datasets were used for training.

| Dataset | Pairs | SROCC | KROCC |
|---------|------:|------:|------:|
| [CID22](https://cloudinary.com/labs/cid22) (codec compression) | 4,292 | 0.8676 | 0.6786 |
| [TID2013](https://www.ponomarenko.info/tid2013.htm) (24 distortion types) | 3,000 | 0.8427 | 0.6657 |
| [KADID-10k](https://database.mmsp-kn.de/kadid-10k-database.html) (25 distortion types) | 10,125 | 0.8192 | 0.6139 |

Weights trained on 218k concordance-filtered synthetic pairs (6 codecs, q5-q100, Nelder-Mead optimization). Training-set SROCC: 0.9960. See `zensim/src/profile.rs` for weights.

<details>
<summary>Reproduce these numbers</summary>

Download the datasets ([instructions below](#downloading-evaluation-datasets)), then:

```bash
# CID22 — expects CID22_validation_set.csv + original/ and compressed/ dirs
cargo run --release -p zensim-validate -- --dataset ./datasets/cid22 --format cid22

# TID2013 — expects mos_with_names.txt + reference_images/ and distorted_images/
cargo run --release -p zensim-validate -- --dataset ./datasets/tid2013 --format tid2013

# KADID-10k — expects dmos.csv + images/
cargo run --release -p zensim-validate -- --dataset ./datasets/kadid10k --format kadid10k
```

Look for `Raw dist corr: SROCC=...` in the output — that's the raw distance SROCC reported above. The `SROCC (Spearman)` line uses mapped scores, which are lower for KADID and TID due to score clamping at 0 (35% of KADID scores clamp).

</details>

## Quick start

```rust
use zensim::{Zensim, ZensimProfile, RgbSlice};

let z = Zensim::new(ZensimProfile::latest());
let source = RgbSlice::new(&src_pixels, width, height);
let distorted = RgbSlice::new(&dst_pixels, width, height);
let result = z.compute(&source, &distorted)?;
println!("score: {:.2}", result.score()); // 100 = identical, higher = better
```

Also accepts `RgbaSlice` (composited over noise background), `imgref::ImgRef` (with stride), `ZenpixelsSource` (with `zenpixels` feature), and `StridedBytes` for BGRA, 16-bit, linear float, and wide gamut (Display P3, BT.2020) inputs. See [docs.rs](https://docs.rs/zensim) for the full `ImageSource` trait.

### zenpixels integration

With the `zenpixels` feature, pass any `PixelSlice` or `PixelBuffer` directly:

```toml
[dependencies]
zensim = { version = "0.2", features = ["zenpixels"] }
```

```rust
use zensim::{Zensim, ZensimProfile, ZenpixelsSource};

let source = ZenpixelsSource::try_from_slice(&pixel_slice)?;
let distorted = ZenpixelsSource::try_from_slice(&other_slice)?;
let result = Zensim::new(ZensimProfile::latest()).compute(&source, &distorted)?;
```

Format mapping is automatic: RGBX/BGRX becomes opaque, premultiplied alpha is un-premultiplied, color primaries are forwarded. HDR (PQ, HLG) and grayscale are rejected with `UnsupportedFormat`.

## What the score means

100 = identical. Higher = more similar. The score is a compressive mapping (`100 - 18 × d^0.7`), giving more resolution at the high-quality end where it matters most.

Each `ZensimResult` also provides approximate translations to other metrics:

| Method | What it returns |
|--------|-----------------|
| `score()` | Zensim similarity (0-100) |
| `raw_distance()` | Feature distance before mapping (lower = better) |
| `approx_ssim2()` | SSIMULACRA2 estimate (MAE 4.4 pts, Pearson r = 0.974) |
| `approx_dssim()` | DSSIM estimate (MAE 0.00129, Pearson r = 0.952) |
| `approx_butteraugli()` | Butteraugli estimate (MAE 1.65, Pearson r = 0.713) |

The `mapping` module has bidirectional interpolation tables — including JPEG quality. These are median values from 344k synthetic pairs across 6 codecs (source: `zensim/src/mapping.rs`):

| Zensim | ≈ SSIM2 | ≈ DSSIM | ≈ JPEG quality |
|-------:|--------:|--------:|:---------------|
| 98 | 96.50 | 0.000017 | ~q95 |
| 90 | 89.41 | 0.000278 | ~q60 |
| 80 | 80.51 | 0.001119 | ~q30 |
| 70 | 71.40 | 0.002356 | — |

JPEG quality mapping accuracy is ±7 quality units MAE — individual images vary widely.

## Regression testing

[**zensim-regress**](zensim-regress/README.md) tracks pixel output across platforms and dependency updates. Hash-based checksums for fast exact matches; perceptual comparison with forensic evidence when hashes diverge. Amplified diff images, error classification, architecture-specific tolerances, CI manifests, and HTML reports.

```rust
use zensim_regress::checksums::{ChecksumManager, CheckResult};

let mgr = ChecksumManager::new("tests/checksums".as_ref());
let result = mgr.check_pixels("resize", "bicubic", "200x200",
    &pixels, width, height, None).unwrap();
assert!(result.passed(), "{result}");
```

Run with `UPDATE_CHECKSUMS=1` to create baselines. See the [zensim-regress guide](zensim-regress/README.md) for the full workflow.

## Batch comparison

Compare one reference against many distorted variants. Precomputing the reference skips redundant XYB conversion and pyramid construction — saves ~25% per comparison at 4K.

```rust
let precomputed = z.precompute_reference(&source)?;
for dst_pixels in &distorted_images {
    let dst = RgbSlice::new(dst_pixels, width, height);
    let result = z.compute_with_ref(&precomputed, &dst)?;
}
```

## How it works

228 features — 19 per channel (X, Y, B) per scale (1x, 2x, 4x, 8x) — scored by trained weights:

- **SSIM** (mean, L2, L4 pooling) — structural similarity in XYB, using ssimulacra2's modified formula (no luminance denominator)
- **Edge artifacts** (mean, L2, L4) — ringing, banding, blockiness
- **Detail loss** (mean, L2, L4) — blur, smoothing, texture destruction
- **MSE** in XYB color space
- **High-frequency features** — energy loss, magnitude loss, energy gain
- **Peak features** — per-feature max and L8-pooled (near-worst-case)

Computed in XYB (cube-root LMS) with O(1)-per-pixel box blur and fused AVX2/AVX-512 SIMD kernels via [archmage](https://crates.io/crates/archmage). Safe scalar fallback on all platforms.

## Profiles

Each `ZensimProfile` bundles weights and score mapping parameters. Scores from a given profile stay stable across crate versions.

| Profile | Training | SROCC |
|---------|----------|------:|
| `PreviewV0_1` | 344k synthetic, 5-fold CV | 0.9936 |
| `PreviewV0_2` | 218k concordance-filtered, Nelder-Mead | 0.9960 |

`ZensimProfile::latest()` returns `PreviewV0_2`. Results are deterministic for the same input on the same architecture; cross-architecture scores (AVX2 vs scalar vs AVX-512) may differ by small ULP.

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `avx512` | yes | AVX-512 SIMD paths |
| `imgref` | yes | `ImageSource` impls for `imgref::ImgRef<Rgb<u8>>` and `ImgRef<Rgba<u8>>` |
| `training` | no | Expose metric internals for weight training |
| `classification` | no | Error classification API (`classify()`, `DeltaStats`, `ErrorCategory`) |
| `zenpixels` | no | `ImageSource` adapter for zenpixels `PixelSlice`/`PixelBuffer` |

## Downloading evaluation datasets

To reproduce the SROCC numbers above, you need the three human-rated datasets. All are freely available for research use.

**TID2013** — [ponomarenko.info/tid2013.htm](https://www.ponomarenko.info/tid2013.htm)

25 reference images, 3,000 distorted (24 distortion types × 5 levels). Download the RAR archive, extract so you have `mos_with_names.txt`, `reference_images/`, and `distorted_images/` in the same directory.

*N. Ponomarenko et al., "Image database TID2013: Peculiarities, results and perspectives," Signal Processing: Image Communication, 2015. [DOI: 10.1016/j.image.2014.10.009](https://doi.org/10.1016/j.image.2014.10.009)*

**KADID-10k** — [database.mmsp-kn.de/kadid-10k-database.html](https://database.mmsp-kn.de/kadid-10k-database.html)

81 reference images, 10,125 distorted (25 distortion types × 5 levels). Download from [OSF](https://osf.io/xkqjh/). Expected structure: `dmos.csv` and `images/` directory in the same parent.

*H. Lin, V. Hosu, D. Saupe, "KADID-10k: A Large-scale Artificially Distorted IQA Database," QoMEX 2019. [DOI: 10.1109/QoMEX.2019.8743252](https://doi.org/10.1109/QoMEX.2019.8743252)*

**CID22** — [cloudinary.com/labs/cid22](https://cloudinary.com/labs/cid22)

49 validation reference images, 4,292 distorted (6 codecs, medium-to-lossless quality). Download the validation set. Expected structure: `CID22_validation_set.csv`, `original/`, and `compressed/` in the same directory. CC BY-SA 4.0.

*Jon Sneyers et al., "CID22: A Large-Scale Subjective Quality Assessment for Lossy Image Compression," 2024.*

## Workspace

| Crate | Description |
|-------|-------------|
| [`zensim`](https://crates.io/crates/zensim) | Metric library |
| [`zensim-regress`](https://crates.io/crates/zensim-regress) | Visual regression testing ([guide](zensim-regress/README.md)) |
| `zensim-bench` | Comparative benchmarks |
| `zensim-validate` | Evaluation and training CLI (internal) |

## MSRV

Rust 1.89.0 (2024 edition).

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs* | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | **zensim** · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sub>* as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

[zenjpeg]: https://crates.io/crates/zenjpeg
[zenpng]: https://crates.io/crates/zenpng
[zenwebp]: https://crates.io/crates/zenwebp
[zengif]: https://crates.io/crates/zengif
[zenavif]: https://crates.io/crates/zenavif
[rav1d-safe]: https://crates.io/crates/rav1d-safe
[zenrav1e]: https://crates.io/crates/zenrav1e
[zenavif-parse]: https://crates.io/crates/zenavif-parse
[zenavif-serialize]: https://crates.io/crates/zenavif-serialize
[zenjxl]: https://crates.io/crates/zenjxl
[jxl-encoder]: https://crates.io/crates/jxl-encoder
[zenjxl-decoder]: https://crates.io/crates/zenjxl-decoder
[zentiff]: https://crates.io/crates/zentiff
[zenbitmaps]: https://crates.io/crates/zenbitmaps
[heic]: https://crates.io/crates/heic
[zenraw]: https://crates.io/crates/zenraw
[zenpdf]: https://crates.io/crates/zenpdf
[ultrahdr]: https://crates.io/crates/ultrahdr
[mozjpeg-rs]: https://crates.io/crates/mozjpeg-rs
[webpx]: https://crates.io/crates/webpx
[zenflate]: https://crates.io/crates/zenflate
[zenzop]: https://crates.io/crates/zenzop
[zenresize]: https://crates.io/crates/zenresize
[zenfilters]: https://crates.io/crates/zenfilters
[zenquant]: https://crates.io/crates/zenquant
[zenblend]: https://crates.io/crates/zenblend
[fast-ssim2]: https://crates.io/crates/fast-ssim2
[butteraugli]: https://crates.io/crates/butteraugli
[resamplescope-rs]: https://crates.io/crates/resamplescope-rs
[codec-eval]: https://crates.io/crates/codec-eval
[codec-corpus]: https://crates.io/crates/codec-corpus
[zenpixels]: https://crates.io/crates/zenpixels
[zenpixels-convert]: https://crates.io/crates/zenpixels-convert
[linear-srgb]: https://crates.io/crates/linear-srgb
[garb]: https://crates.io/crates/garb
[zenpipe]: https://crates.io/crates/zenpipe
[zencodec]: https://crates.io/crates/zencodec
[zencodecs]: https://crates.io/crates/zencodecs
[zenlayout]: https://crates.io/crates/zenlayout
[zennode]: https://crates.io/crates/zennode
[ImageResizer]: https://imageresizing.net
[Imageflow]: https://github.com/imazen/imageflow
[imageflow-dotnet]: https://www.nuget.org/packages/Imageflow.AllPlatforms
[imageflow-node]: https://www.npmjs.com/package/@imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[archmage]: https://crates.io/crates/archmage
[magetypes]: https://crates.io/crates/magetypes
[enough]: https://crates.io/crates/enough
[whereat]: https://crates.io/crates/whereat
[zenbench]: https://crates.io/crates/zenbench
[cargo-copter]: https://crates.io/crates/cargo-copter

## License

[MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE)

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.
