<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zensim

Perceptual image similarity in 22 ms at 1080p. 18x faster than C++ SSIMULACRA2 at 4K.

Built on the same psychovisual foundations as SSIMULACRA2 and butteraugli — multi-scale SSIM, edge artifacts, detail loss, and high-frequency features in XYB color space — but with trained weights, fused SIMD kernels, and multi-threaded computation.

**Interactive chart exploration**: <https://imazen.github.io/zensim/> — scatter zensim / fast-ssim2 / butteraugli against human MOS across CID22 / KADID / TID / AIC corpora, filter by codec + version, with per-band SROCC tables and step-5 (20-bin) breakdowns.


## Correlation with human perception

Full Mohammadi 2025 stat panel against three independent human-rated image quality databases that v0.3 did NOT train on. KADID-10k and TID2013 are excluded because v0.3's recovery-phase-4 retrain included them as training groups — they're no longer fair holdouts. On the **CID22** codec-compression holdout, profile `A` reaches SROCC ≈ 0.86 (fast-ssim2: 0.89) while spending the full 0–100 dial with JND landing at score 60 — the property that matters for codec quality targeting.


## Quick start

```toml
[dependencies]
zensim = "0.3"
```

```rust
use zensim::{Zensim, ZensimProfile, RgbSlice};

// Pick a profile explicitly for pinned reproducibility:
let z = Zensim::new(ZensimProfile::A);
// Or use `ZensimProfile::latest_preview()` for whatever current preview
// ships (rotates as new previews land), or `ZensimProfile::codec_target()`
// for the stable codec-target contract. `PreviewV0_2` is the linear
// general-ranking profile; `PreviewV0_1` is the 0.2.x-compatible linear one.
// `src_pixels` / `dst_pixels` are `&[[u8; 3]]` — interleaved, sRGB-encoded
// (gamma, NOT linear) 8-bit RGB. `width`/`height` are `usize`. See "Input
// format" below: getting sRGB-vs-linear wrong silently corrupts every score.
let source = RgbSlice::new(&src_pixels, width, height);
let distorted = RgbSlice::new(&dst_pixels, width, height);
// `compute` returns `Result<ZensimResult, zensim::ZensimError>` — the `?`
// propagates a dimension-mismatch / too-small / too-large error.
let result = z.compute(&source, &distorted)?;
println!("score: {:.2}", result.score()); // 100 = identical, higher = better
```

Also accepts `RgbaSlice` (composited over a noise background), `imgref::ImgRef` (with stride), `ZenpixelsSource` (with `zenpixels` feature), and `StridedBytes` for BGRA, 16-bit, linear float, and wide gamut (Display P3, BT.2020) inputs. See [docs.rs](https://docs.rs/zensim) for the full `ImageSource` trait.

### Input format (read this — wrong input silently corrupts the score)

The `0..100` score is only meaningful if the pixels you pass match the contract zensim assumes. There is **no format auto-detection** for the `RgbSlice` fast path: if you hand it linear bytes where it expects sRGB, or planar bytes where it expects interleaved, it computes a perfectly valid-looking but wrong score — no error is raised. The contract:

- **Color encoding: sRGB-encoded (gamma), NOT linear.** `RgbSlice` / `RgbaSlice` / the `Srgb8*` and `Srgb16Rgba` `StridedBytes` formats all expect **display-encoded sRGB** values — the bytes a PNG/JPEG decoder gives you. zensim linearizes internally before the XYB conversion. If your data is already linear light, do **not** feed it as sRGB; use `StridedBytes` with `PixelFormat::LinearF32Rgba` (linear 32-bit float RGBA) instead. (Display P3 reuses the sRGB transfer function, so `Srgb8*` formats linearize it correctly; SDR BT.2020 technically wants BT.1886 — for exact results linearize externally and use `LinearF32Rgba`. Set primaries via `StridedBytes::with_color_primaries`.)
- **Channel order: interleaved, not planar.** `RgbSlice` takes `&[[u8; 3]]` laid out `R,G,B, R,G,B, …` (one `[u8; 3]` per pixel), `RgbaSlice` takes `&[[u8; 4]]` as `R,G,B,A, …`. **Planar** input (all R, then all G, then all B) is **not** accepted by these types — you must interleave it first, or describe it some other way. The `[[u8; 3]]` / `[[u8; 4]]` element type also pins it to exactly 3 / 4 bytes per pixel, **tightly packed** (no per-row padding) — for row padding use `StridedBytes` (below).
- **Dimensions: `width` and `height` are `usize`** (not `u32`). Both `RgbSlice::new(data, width, height)` and the underlying `ImageSource::{width,height}` use `usize`.
- **Both images must have identical dimensions**, and each must be non-zero. `compute` returns `ZensimError::DimensionMismatch` if they differ, `ZensimError::ImageTooSmall` if either dimension is `0`. (Since 0.3.0, sub-64px images down to 1×1 are reflect-padded internally and score normally — only empty inputs are rejected.)

The infallible constructors (`RgbSlice::new`, `RgbaSlice::new`, `StridedBytes::new`) **panic** if `data.len()` is too short for `width × height` (or the stride is invalid). For untrusted sizes use the `try_*` variants — `RgbSlice::try_new(data, width, height) -> Result<RgbSlice, ZensimError>` etc. — which return `ZensimError::InvalidDataLength` / `ZensimError::InvalidStride` / `ZensimError::ImageTooLarge` instead of panicking.

### Return type and errors

```rust
pub fn compute(
    &self,
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> Result<ZensimResult, zensim::ZensimError>
```

`ZensimError` is a `#[non_exhaustive]` enum (so match it with a `_` arm) — the variants `compute` can return are `DimensionMismatch`, `ImageTooSmall`, and `ImageTooLarge` (dimensions exceed the configured `max_pixels` cap — **120 MP by default** since #49; tighten it with `Zensim::with_max_pixels`, or pass `with_max_pixels(usize::MAX)` to opt out for trusted input — or `width × height` overflows `usize` on 32-bit / wasm32). HDR-flagged sources (`ImageSource::is_hdr` returns `true`) are refused with `HdrInputRequiresPuPath` — score HDR via the PU21 front-end (`Zensim::compute_pu_linear`, fed absolute-luminance linear RGB in cd/m²) instead. On success, `ZensimResult::score()` is the `0..100` similarity; `raw_distance()`, `approx_ssim2()`, `approx_dssim()`, and `approx_butteraugli()` are also available (see "What the score means").

### Strided / padded rows

When rows are not tightly packed (SIMD-aligned padding, a sub-region crop of a larger buffer, decoder output with row guards), use `StridedBytes`, where `stride` is the **byte** distance between the start of consecutive rows:

```rust
use zensim::{StridedBytes, PixelFormat};

// e.g. 8-bit RGB where each row is padded to `row_stride` bytes (≥ width*3):
let src = StridedBytes::new(&bytes, width, height, row_stride, PixelFormat::Srgb8Rgb);
// `try_new(..) -> Result<_, ZensimError>` returns InvalidStride / InvalidDataLength
// instead of panicking. `with_alpha_mode(.., AlphaMode)` / `with_color_primaries(..)`
// set alpha handling and gamut; default alpha mode is `AlphaMode::Unknown`.
let result = z.compute(&src, &dst)?;
```

With the `imgref` feature (on by default), `imgref::ImgRef<'_, rgb::Rgb<u8>>` and `ImgRef<'_, rgb::Rgba<u8>>` implement `ImageSource` directly and honor their pixel stride — pass an `ImgRef` straight to `compute`. (The element type must be `rgb::Rgb<u8>` / `rgb::Rgba<u8>`; the `ImgRef` stride is in pixels.)

### Cancellation

A single `compute` / `compute_with_ref` call is **not** interruptible mid-computation — the `zensim` metric library exposes no `Stop`-token parameter, and a single comparison is typically tens of milliseconds (~22 ms at 1080p). Cancellation is at the granularity of *your* loop: when comparing one reference against many distorted variants (see "Batch comparison"), check your own cancellation flag between `compute_with_ref` calls. The `enough` cooperative-cancellation crate is used by the `zensim-target` codec-targeting CLI (to bound its binary-search loop) and `zensim-regress`, not by the core metric API.

### zenpixels integration

With the `zenpixels` feature, pass any `PixelSlice` or `PixelBuffer` directly:

```toml
[dependencies]
zensim = { version = "0.3", features = ["zenpixels"] }
```

```rust
use zensim::{Zensim, ZensimProfile, ZenpixelsSource};

let source = ZenpixelsSource::try_from_slice(&pixel_slice)?;
let distorted = ZenpixelsSource::try_from_slice(&other_slice)?;
let result = Zensim::new(ZensimProfile::A).compute(&source, &distorted)?;
```

Format mapping is automatic: RGBX/BGRX becomes opaque, premultiplied alpha is un-premultiplied, color primaries are forwarded. HDR (PQ, HLG) and grayscale are rejected with `UnsupportedFormat`.

## Target-score CLI (`zensim-target`)

The [`zensim-target`](https://github.com/imazen/zensim/blob/main/zensim-target/README.md) workspace crate is the
runtime side of the "user-facing quality dial" goal. Given an input
image and a target zensim score, it picks the codec quality knob via
binary search:

```bash
cargo run --release -p zensim-target -- input.png \
    --target 70 --codec zenjpeg --output out.jpg
# codec=Jpeg  target=70.0  achieved=69.46  knob=78.44  bytes=62234  iters=5  converged=true
```

Supported codecs: `zenjpeg`, `zenwebp`, `zenavif` (wired and
demonstrated); `zenpng` (lossless, single probe); `zenjxl`
(encode-only in v0.1, decode plumbing pending). Demo matrix at
[`benchmarks/zensim_target_demo_2026-05-18.md`](https://github.com/imazen/zensim/blob/main/benchmarks/zensim_target_demo_2026-05-18.md):
33 / 36 cells converged within ±1.5 score units, median 5 iterations.

`zensim-target` is **AGPL-3.0-only** because it links the AGPL zen
codec crates; the core `zensim` library stays MIT/Apache.

## What the score means

100 = identical. Higher = more similar. Under profile `A`, the MLP output passes through a monotone PCHIP dial spline calibrated so the dial tracks degradation monotonically (identity ≈ 97.7; byte-identical inputs short-circuit to exactly 100). Under the linear profiles (`PreviewV0_1`/`PreviewV0_2`), the score is a compressive mapping (`100 - 18 × d^0.7`), giving more resolution at the high-quality end.

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

[**zensim-regress**](https://github.com/imazen/zensim/blob/main/zensim-regress/README.md) tracks pixel output across platforms and dependency updates. Hash-based checksums for fast exact matches; perceptual comparison with forensic evidence when hashes diverge. Amplified diff images, error classification, architecture-specific tolerances, CI manifests, and HTML reports.

```rust
use zensim_regress::checksums::{ChecksumManager, CheckResult};

let mgr = ChecksumManager::new("tests/checksums".as_ref());
let result = mgr.check_pixels("resize", "bicubic", "200x200",
    &pixels, width, height, None).unwrap();
assert!(result.passed(), "{result}");
```

Run with `UPDATE_CHECKSUMS=1` to create baselines. See the [zensim-regress guide](https://github.com/imazen/zensim/blob/main/zensim-regress/README.md) for the full workflow.

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

Each `ZensimProfile` bundles weights and score-mapping parameters. Scores from a given profile stay stable across crate versions. The published crate ships three profiles; the historical / experimental research profiles are preserved (bit-identically) in the unpublished `zensim-experimental` crate.

| Profile | Kind | CID22 SROCC | Bake |
|---------|------|------:|------|
| `A` | 372-input MLP, per-sample-α + monotone PCHIP dial spline | **0.87** | 27 KB v47-strict-QAT |
| `PreviewV0_2` | linear, 228 Nelder-Mead weights | 0.87 | none (linear) |
| `PreviewV0_1` | linear, 228 weights (0.2.x back-compat) | 0.86 | none (linear) |

`ZensimProfile::codec_target()` and `latest_preview()` both return `A` — the canonical production codec-target the zen codecs dial against. The deprecated `latest()` also returns `A`. To load your own bake, construct `ZensimProfile::Custom { params, name }` via [`ProfileParams::builder()`](https://docs.rs/zensim/latest/zensim/profile/struct.ProfileParams.html). Results are deterministic for the same input on the same architecture; cross-architecture scores (AVX2 vs scalar vs AVX-512) may differ by small ULP.

The historical `PreviewV0_4` / `PreviewV0_5*` SOTA-trail variants, `A_Phone`, and `LinearBounded` live in the **`zensim-experimental`** crate (not published), each rebuilt through the `Custom` extension point — e.g. `Zensim::new(zensim_experimental::preview_v0_5_tuner_v4())`. zenpredict (the MLP runtime) is MIT/Apache-2.0 — no AGPL transitive obligation on default builds.

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `avx512` | yes | AVX-512 SIMD paths |
| `threads` | yes | Multi-threaded computation via rayon (disable for wasm / single-threaded) |
| `imgref` | yes | `ImageSource` impls for `imgref::ImgRef<Rgb<u8>>` and `ImgRef<Rgba<u8>>` |
| `training` | no | Expose metric internals for weight training |
| `classification` | no | Error classification API (`classify()`, `DeltaStats`, `ErrorCategory`) |
| `zenpixels` | no | `ImageSource` adapter for zenpixels `PixelSlice`/`PixelBuffer` |
| `custom-profiles` | no | `ZensimProfile::Custom` + `ProfileParams::builder()` for externally-defined bakes |
| `streaming_strips_oom` | no | Un-ignores the ~500 MB 80 MP streaming OOM-relief integration test |


## Workspace

| Crate | Description |
|-------|-------------|
| [`zensim`](https://crates.io/crates/zensim) | Metric library |
| [`zensim-regress`](https://crates.io/crates/zensim-regress) | Visual regression testing ([guide](https://github.com/imazen/zensim/blob/main/zensim-regress/README.md)) |
| `zensim-experimental` | Historical / research profiles via the `Custom` extension point (unpublished) |
| `zensim-validate` | Evaluation and training CLI (internal) |
| `zensim-bench` | Comparative benchmarks (standalone root, sibling-dep) |
| `zensim-target` | Target-score codec CLI (standalone root, AGPL, sibling-dep) |

## MSRV

Rust 1.93.0 (2024 edition).

## License

[MIT](https://github.com/imazen/zensim/blob/main/LICENSE-MIT) OR [Apache-2.0](https://github.com/imazen/zensim/blob/main/LICENSE-APACHE)

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | **zensim** · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
