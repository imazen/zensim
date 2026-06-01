# zensim [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zensim/ci.yml?style=flat-square)](https://github.com/imazen/zensim/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zensim?style=flat-square)](https://crates.io/crates/zensim) [![lib.rs](https://img.shields.io/crates/v/zensim?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zensim) [![docs.rs](https://img.shields.io/docsrs/zensim?style=flat-square)](https://docs.rs/zensim) [![license](https://img.shields.io/crates/l/zensim?style=flat-square)](https://github.com/imazen/zensim#license)

Perceptual image similarity in 22 ms at 1080p. 18x faster than C++ SSIMULACRA2 at 4K.

Built on the same psychovisual foundations as SSIMULACRA2 and butteraugli — multi-scale SSIM, edge artifacts, detail loss, and high-frequency features in XYB color space — but with trained weights, fused SIMD kernels, and multi-threaded computation.

**Interactive chart exploration**: <https://imazen.github.io/zensim/> — scatter zensim / fast-ssim2 / butteraugli against human MOS across CID22 / KADID / TID / AIC corpora, filter by codec + version, with per-band SROCC tables and step-5 (20-bin) breakdowns.

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

Full Mohammadi 2025 stat panel against three independent human-rated image quality databases that v0.3 did NOT train on. KADID-10k and TID2013 are excluded because v0.3's recovery-phase-4 retrain included them as training groups — they're no longer fair holdouts. Higher SROCC + PLCC + KROCC + PWRC is better; lower OR + Z-RMSE is better.

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

`PreviewV0_3` (the MLP profile shipping in zensim 0.3.x and now the canonical [`codec_target()`](docs/CODEC_TARGET_METRIC.md)): 372-input MLP (372 → 128 → 128 identity passthrough) with per-sample-α head + tanh-output pin + 7-knot PCHIP spline. 54 KB packed bake (i8 + zerobias + lz4, md5 `cac9416124a5e5f8ff577bc78e15ea1f`, file `zensim/weights/v_tuner_v11_2026-05-24.bin`). Trained on 5 groups (safesyn 196k + cid22_train 17.6k + kadid 10.1k + tid 3k + konjnd_dense 20.2k) with a konjnd-aggregation aux loss for per-source PJND calibration. Methodology: [`benchmarks/v_tuner_v11_methodology_2026-05-24.md`](benchmarks/v_tuner_v11_methodology_2026-05-24.md).

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

## Quick start

```rust
use zensim::{Zensim, ZensimProfile, RgbSlice};

// Pick a version explicitly for pinned reproducibility:
let z = Zensim::new(ZensimProfile::PreviewV0_3);
// Or use `ZensimProfile::latest_preview()` for whatever current preview
// ships (rotates as new previews land), or `ZensimProfile::codec_target()`
// for the stable codec-target contract.
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
let result = Zensim::new(ZensimProfile::PreviewV0_3).compute(&source, &distorted)?;
```

Format mapping is automatic: RGBX/BGRX becomes opaque, premultiplied alpha is un-premultiplied, color primaries are forwarded. HDR (PQ, HLG) and grayscale are rejected with `UnsupportedFormat`.

## Target-score CLI (`zensim-target`)

The [`zensim-target`](zensim-target/README.md) workspace crate is the
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
[`benchmarks/zensim_target_demo_2026-05-18.md`](benchmarks/zensim_target_demo_2026-05-18.md):
33 / 36 cells converged within ±1.5 score units, median 5 iterations.

`zensim-target` is **AGPL-3.0-only** because it links the AGPL zen
codec crates; the core `zensim` library stays MIT/Apache.

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

Each `ZensimProfile` bundles weights and score-mapping parameters. Scores from a given profile stay stable across crate versions. The published crate ships three profiles; the historical / experimental research profiles are preserved (bit-identically) in the unpublished `zensim-experimental` crate.

| Profile | Kind | CID22 SROCC | Bake |
|---------|------|------:|------|
| `A` (alias: deprecated `PreviewV0_3`) | 372-input MLP, per-sample-α + monotone PCHIP dial spline | **0.87** | 27 KB v47-strict-QAT |
| `PreviewV0_2` | linear, 228 Nelder-Mead weights | 0.87 | none (linear) |

`ZensimProfile::codec_target()` and `latest_preview()` both return `A` — the canonical production codec-target the zen codecs dial against. The deprecated `latest()` also returns `A`. To load your own bake, construct `ZensimProfile::Custom { params, name }` via [`ProfileParams::builder()`](https://docs.rs/zensim/latest/zensim/profile/struct.ProfileParams.html). Results are deterministic for the same input on the same architecture; cross-architecture scores (AVX2 vs scalar vs AVX-512) may differ by small ULP.

The historical `PreviewV0_4` / `PreviewV0_5*` SOTA-trail variants, `A_Phone`, `PreviewV0_1`, and `LinearBounded` live in the **`zensim-experimental`** crate (not published), each rebuilt through the `Custom` extension point — e.g. `Zensim::new(zensim_experimental::preview_v0_5_tuner_v4())`. zenpredict (the MLP runtime) is MIT/Apache-2.0 — no AGPL transitive obligation on default builds.

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

Rust 1.93.0 (2024 edition).

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
