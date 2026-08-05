# zensim [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zensim/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zensim/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zensim?style=flat-square)](https://crates.io/crates/zensim) [![lib.rs](https://img.shields.io/crates/v/zensim?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zensim) [![docs.rs](https://img.shields.io/docsrs/zensim?style=flat-square)](https://docs.rs/zensim) [![MSRV](https://img.shields.io/badge/MSRV-1.93-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/zensim?style=flat-square)](#license)

Perceptual image similarity in 22 ms at 1080p. 18x faster than C++ SSIMULACRA2 at 4K.

Built on the same psychovisual foundations as SSIMULACRA2 and butteraugli — multi-scale SSIM, edge artifacts, detail loss, and high-frequency features in XYB color space — but with trained weights, fused SIMD kernels, and multi-threaded computation.

**Interactive chart exploration**: <https://imazen.github.io/zensim/> — scatter zensim / fast-ssim2 / butteraugli against human MOS across CID22 / KADID / TID / AIC corpora, filter by codec + version, with per-band SROCC tables and step-5 (20-bin) breakdowns.

<!-- crates.io:skip-start -->
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
<!-- crates.io:skip-end -->

## Correlation with human perception

Full Mohammadi 2025 stat panel against three independent human-rated image quality databases that v0.3 did NOT train on. KADID-10k and TID2013 are excluded because v0.3's recovery-phase-4 retrain included them as training groups — they're no longer fair holdouts. On the **CID22** codec-compression holdout, the default profile `B` reaches SROCC ≈ 0.876 and the deprecated `A` ≈ 0.86 (fast-ssim2: 0.89), both spending the full 0–100 dial with JND landing near score 60 — the property that matters for codec quality targeting. **The detailed per-corpus / per-band tables below are profile `A`'s** (the prior default); `B`'s full held-out panel is in [`benchmarks/profile_b_methodology_2026-07-12.md`](benchmarks/profile_b_methodology_2026-07-12.md), and regenerating these tables for `B` before release is a tracked follow-up.

<!-- crates.io:skip-start -->
Higher SROCC + PLCC + KROCC + PWRC is better; lower OR + Z-RMSE is better.

### CID22 — codec compression artifacts (n=4,292, sacred holdout)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| **zensim v0.3** | 0.860 | 0.853 | 0.673 | 0.045 | 0.909 | 0.523 |
| fast-ssim2 (SSIMULACRA2) | **0.890** | **0.888** | **0.706** | 0.042 | **0.935** | **0.460** |
| cvvdp (ColorVideoVDP) | 0.821 | 0.825 | 0.624 | 0.042 | 0.884 | 0.565 |
| iwssim (Wang & Li 2011) | 0.784 | 0.793 | 0.594 | 0.052 | 0.853 | 0.610 |

### AIC-3 CTC — JPEG-AIC compression at JND levels (n=600)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| zensim v0.3 | 0.776 | 0.788 | 0.607 | 0.042 | 0.854 | 0.616 |
| fast-ssim2 | **0.797** | **0.809** | **0.629** | 0.057 | **0.872** | 0.588 |
| cvvdp | 0.792 | 0.803 | 0.626 | 0.042 | 0.866 | **0.595** |
| iwssim | 0.774 | 0.791 | 0.606 | 0.045 | 0.854 | 0.612 |

### AIC-4 sample — JPEG-AIC reconstructed JND, 6 codecs (n=300)

| Metric | SROCC |
|---|---:|
| zensim v0.3 | 0.928 |
| fast-ssim2 | baselines pending |
| cvvdp | baselines pending |
| iwssim | baselines pending |

The AIC-4 baseline scores for ssim2/cvvdp/iwssim haven't been folded
into our standard panel yet — the raw per-pair metric scores live at
`/mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG-AIC_metric_scores.csv`
but the SROCC computation against the reconstructed-JND target isn't
in the panel doc. v0.3's AIC-4 SROCC of **0.928** is from
`bake_verdict` on the 300-pair val parquet at
`canonical-2026-05-18/val/aic4.parquet`.

### CID22 per-band SROCC (10 width-10 bins on the human-MOS scale)

Per `CLAUDE.md` "10-band reporting rule": the primary release gate is the per-band picture, not the aggregate. Below-PJND bands (B3–B5) are the hard ones because the human-MOS scores in those bands are noisy and bunched. Bands B0–B2 have ≤ 1 sample on CID22 and are omitted.

| Band | range | n | **v0.3** | ssim2 | cvvdp | iwssim |
|---|---|--:|---:|---:|---:|---:|
| B3 | [0.30, 0.40) | 57 | 0.051 | **0.134** | 0.148 | 0.096 |
| B4 | [0.40, 0.50) | 266 | 0.230 | 0.289 | 0.260 | 0.210 |
| B5 | [0.50, 0.60) | 615 | 0.273 | **0.389** | 0.290 | 0.193 |
| B6 | [0.60, 0.70) | 836 | 0.287 | **0.417** | 0.336 | 0.210 |
| B7 | [0.70, 0.80) | 1092 | **0.408** | 0.397 | 0.310 | 0.283 |
| B8 | [0.80, 0.90) | 1382 | 0.500 | **0.501** | 0.319 | 0.413 |
| B9 | [0.90, 1.00] | 43 | **0.220** | 0.112 | 0.081 | 0.134 |

Per-band read: ssim2 wins B5–B6 (where most CID22 mass sits); v0.3 wins B7 (good-quality region) and B9 (near-lossless tail). v0.3 essentially matches ssim2 on B8 (the dominant band). cvvdp + iwssim are weakest across every band — they're stronger as aggregate metrics than per-band rank predictors here.

CID22 per-band Z-RMSE (lower better):

| Band | n | **v0.3** | ssim2 | cvvdp | iwssim |
|---|--:|--:|--:|--:|--:|
| B3 | 57 | 0.950 | **0.947** | 0.990 | 0.989 |
| B4 | 266 | 0.959 | **0.947** | 0.962 | 0.965 |
| B5 | 615 | 0.959 | **0.921** | 0.954 | 0.971 |
| B6 | 836 | 0.957 | **0.908** | 0.941 | 0.972 |
| B7 | 1092 | **0.912** | 0.907 | 0.947 | 0.954 |
| B8 | 1382 | 0.866 | 0.866 | 0.947 | 0.909 |
| B9 | 43 | **0.937** | 0.940 | 0.952 | 0.854 |

ssim2 has the tightest Z-RMSE in the mid bands (B4–B6) — this is the ssim2-target training-bias caveat from CLAUDE.md materializing. v0.3 and ssim2 are tied on B7–B8. v0.3 wins the noisy tails (B9 near-lossless).

### Per-corpus headline

- **CID22**: ssim2 wins SROCC by 0.03; v0.3 is second. Note that CLAUDE.md's "SROCC-only verdicts BANNED" caveat applies — older trainers used ssim2-derived targets, which biases SROCC measurements toward ssim2-shaped surfaces. v0.3's Z-RMSE (0.523) trails ssim2's 0.460.
- **AIC-3**: 4-way tie within 0.02 SROCC; ssim2 nominally best.
- **AIC-4**: v0.3 SROCC = 0.928. cvvdp / ssim2 / iwssim baselines on AIC-4 haven't been computed into our panel doc yet (raw scores at `JPEG-AIC_metric_scores.csv`; SROCC against reconstructed-JND target is a TODO).
- **None of the four hits all three holdouts** — v0.3 trades 0.03 CID22 SROCC for full 0-100 dial coverage + JND@60-bit-exact + per-source PJND tracking (the dial properties that matter for codec targeting). See [`docs/CODEC_TARGET_METRIC.md`](docs/CODEC_TARGET_METRIC.md).

v0.2 (default-on linear profile through zensim 0.2.x): 228 linear weights × basic+peak features, trained on 218k concordance-filtered synthetic pairs via Nelder-Mead.

`A` (**deprecated** since 0.3.0 — the prior default MLP profile, superseded as [`codec_target()`](docs/CODEC_TARGET_METRIC.md) by `B` below; behind the default-on `deprecated-profiles` feature): 372-input MLP (372 → 128 → 64 + per-sample-α head + tanh-output pin) with a monotone 7-knot PCHIP dial spline. 27 KB packed bake (`v47-strict-QAT`, f16 + zerobias, file `zensim/weights/v47_strict_qat_native_2026-05-27.bin`). Masked-monotone by construction (W1 ≥ 0 on the 300 sign-safe features, rank_w ≤ 0, α ≡ 1): 0 inversions, 0 above-identity, identity = 97.69. Trained on 5 groups (safesyn 196k + cid22_train 17.6k + kadid 10.1k + tid 3k + konjnd_dense 20.2k) via one QAT-native pass. Held-out SROCC: CID22 0.866, KADID 0.793, TID 0.793, KonJND 0.419, AIC-3 0.768, AIC-4 0.885. Methodology: [`benchmarks/v0_qat_native_methodology_2026-05-27.md`](benchmarks/v0_qat_native_methodology_2026-05-27.md).

`C` (added 2026-08-05, research-grade; `B` stays the default): a 944-input MLP over the extended folded feature regime — CID22 0.8867, LIVE 0.9604, CSIQ 0.9331, dial monotonicity 99.3% in dial units. It consumes the 944-feature layout rather than the standard 372 pipeline, so it is scored via `Zensim::compute_folded720_append2_features` (`feature-regime-v2` feature) + `score_features_with_profile` — see the `ZensimProfile::C` rustdoc for the contract and [`docs/PROFILE_C_REPRODUCTION_2026-08-05.md`](docs/PROFILE_C_REPRODUCTION_2026-08-05.md) for full provenance + reproduction.

### v0.2 → v0.3 rough score equivalence

Both profiles span 0..100 but use the dial differently. v0.2's linear formula `100 − 18·|d|^0.7` floor-clamps below moderate distortion (38k of 68k cross-codec pairs land at v0.2 ≤ 5, while v0.3 spreads them across 28..50). v0.3 uses the full 0-100 dial with JND landing at exactly score 60. Rough lookup on 68,788 matched cross-codec pairs (Spearman v0.2 ↔ v0.3 = 0.88):

| v0.2 target | v0.3 median (p25 → p75) | rough quality region |
|--:|--:|---|
| 10 | 52.6 (47.8 → 55.6) | low-q, dial floor |
| 20 | 55.2 (52.4 → 58.2) | sub-PJND |
| 30 | 58.9 (55.0 → 60.5) | approaching JND |
| 40 | 61.0 (58.9 → 63.6) | just past JND |
| 50 | 65.8 (64.3 → 66.9) | mid-PJND |
| 60 | 68.8 (66.4 → 72.5) | comfortable quality |
| 70 | 78.4 (76.5 → 80.3) | good compression |
| 80 | 90.0 (87.6 → 91.1) | near-lossless |
| 90 | 96.6 (95.1 → 97.3) | visually lossless |
| 100 | 100 (exact, byte-identical short-circuit) | lossless |

The mapping is non-linear because v0.2's clamp at low quality compresses the 0-30 range into a single floor; v0.3 differentiates that region. For users targeting "score 70" in v0.2 code, the v0.3 equivalent is roughly **score 78**.

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
<!-- crates.io:skip-end -->

## Quick start

```toml
[dependencies]
zensim = "0.3"
```

```rust
use zensim::{Zensim, ZensimProfile, RgbSlice};

// Pick a profile explicitly for pinned reproducibility:
let z = Zensim::new(ZensimProfile::B);
// `B` is the deterministic linear-ensemble default for SDR content; `BHdr`
// is its HDR (absolute-nits) counterpart. Or use
// `ZensimProfile::codec_target()` / `latest_preview()` for the stable
// codec-target contract (both return `B`). `A` (the v47 MLP) is deprecated
// (behind the default-on `deprecated-profiles` feature).
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
let result = Zensim::new(ZensimProfile::codec_target()).compute(&source, &distorted)?;
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

100 = identical. Higher = more similar. Every published profile (`A`, `B`, `BHdr`) routes its raw MLP/linear-ensemble output through a monotone PCHIP dial spline calibrated so the dial tracks degradation monotonically (identity ≈ 97.7 for `A`; byte-identical inputs short-circuit to exactly 100 for all profiles).

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

Each `ZensimProfile` bundles weights and score-mapping parameters. Scores from a given profile stay stable across crate versions. The published crate ships five selectable profiles (`A` deprecated, `PreviewV0_1` / `PreviewV0_2` retained for 0.2.7 compatibility, `B` / `BHdr` current); the historical / experimental research profiles are preserved (bit-identically) in the unpublished `zensim-experimental` crate.

| Profile | Kind | CID22 SROCC | Bake |
|---------|------|------:|------|
| `B` (**default** — `codec_target()`) | 372-input linear ensemble (35-weight lasso) + dial spline, SDR content | **0.8764** | 7.3 KB ens-Pline-cid80 |
| `BHdr` | linear ensemble on PU-linear (absolute-nits) features + dial spline, HDR content only | n/a — HDR-only (UPIQ-HDR \|SROCC\| 0.7313) | 11.7 KB hdr-lasso0.001-shaped |
| `A` (**deprecated** — behind `deprecated-profiles`) | 372-input MLP, per-sample-α + monotone PCHIP dial spline | **0.8657** | 27 KB v47-strict-QAT |
| `PreviewV0_1` / `PreviewV0_2` | 228-weight linear (no MLP), `100 − 18·d^0.7` mapping — 0.2.7-compat | — | embedded weight arrays |

`ZensimProfile::codec_target()` and `latest_preview()` both return `B` — the canonical production codec-target the zen codecs dial against (the deprecated `latest()` also returns `B`). `A` (the prior default, the v47 MLP) is now `#[deprecated]` and lives behind the default-on `deprecated-profiles` feature — build with `--no-default-features` to drop it. To load your own bake, construct `ZensimProfile::Custom { params, name }` via [`ProfileParams::builder()`](https://docs.rs/zensim/latest/zensim/profile/struct.ProfileParams.html). Results are deterministic for the same input on the same architecture; cross-architecture scores (AVX2 vs scalar vs AVX-512) may differ by small ULP.

`ZensimProfile::PreviewV0_1` / `PreviewV0_2` (the linear profiles that shipped in 0.2.7) are RETAINED as first-class, non-deprecated, selectable variants — the 0.3.0 line reverted their removal (commit `493c91cd`) to preserve semver compatibility with 0.2.7; see the CHANGELOG `[Unreleased]` **Restored** entry. `B` is the current deterministic-linear default for SDR content.

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

<!-- crates.io:skip-start -->
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
<!-- crates.io:skip-end -->

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
