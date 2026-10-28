# zensim [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zensim/ci.yml?style=flat-square)](https://github.com/imazen/zensim/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zensim?style=flat-square)](https://crates.io/crates/zensim) [![lib.rs](https://img.shields.io/crates/v/zensim?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zensim) [![docs.rs](https://img.shields.io/docsrs/zensim?style=flat-square)](https://docs.rs/zensim) [![license](https://img.shields.io/crates/l/zensim?style=flat-square)](https://github.com/imazen/zensim#license)

> **Repository history.** This repository continues the published `zensim` crate from 0.2.7.
> The research, training and evaluation history written after 0.2.7 lives in
> [imazen/zensimx](https://github.com/imazen/zensimx).

A fast perceptual image similarity metric: 100 means identical, lower means more visible
damage. Use it to compare an image against a reference, or to steer an encoder to a target
quality.

```rust
use zensim::{RgbSlice, Zensim, ZensimProfile};

let z = Zensim::new(ZensimProfile::latest());
let result = z.compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h))?;
println!("{:.2}", result.score());
```

Encoders that search for a target score use `ZensimProfile::codec_target()` and score many
candidates against one reference with `precompute_reference` and `compute_with_ref`.

## Profiles

| Profile | Name | Use |
|---|---|---|
| `E` (`latest()`) | `zensim-e` | General purpose: compression, noise, blur, colour and contrast changes |
| `ECodec` (`codec_target()`) | `zensim-e-codec` | Codec output (JPEG, WebP, AVIF, JPEG XL); less reliable on other distortions |
| `A` | `zensim-a` | Legacy, for callers whose thresholds are calibrated to it |
| `PreviewV0_1`, `PreviewV0_2` | | The linear profiles of zensim 0.1 and 0.2 |

`E` and `ECodec` are monotone networks over multi-scale SSIM, edge, detail-loss and
high-frequency features in the XYB colour space: more damage never raises the score. Severe
damage can score below 0. Their scores are bit-identical across x86 SIMD tiers and thread
counts. Training data and evaluation details: [docs/PROFILES.md](docs/PROFILES.md).

## Accuracy

Spearman rank correlation with human ratings on the held-out evaluation references of nine
datasets, scored once after the models were frozen. Six of these datasets contributed other
references to training; NITS-IQA, IQA-ODI and VCL@FER were never trained on. SSIMULACRA2 and
butteraugli were trained on none of them:

| Dataset | `E` | `ECodec` | SSIMULACRA2 | Butteraugli (3-norm) |
|---|---:|---:|---:|---:|
| CID:IQ | 0.865 | 0.742 | 0.757 | 0.592 |
| CID:OA | 0.889 | 0.813 | 0.847 | 0.577 |
| Colourlab | 0.711 | 0.477 | 0.604 | 0.373 |
| IQA-ODI | 0.865 | 0.828 | 0.854 | 0.806 |
| IRCCyN/IVC | 0.940 | 0.935 | 0.949 | 0.912 |
| LIVE-MD | 0.894 | 0.910 | 0.913 | 0.827 |
| MDID | 0.870 | 0.692 | 0.787 | 0.572 |
| NITS-IQA | 0.706 | 0.724 | 0.777 | 0.706 |
| VCL@FER | 0.930 | 0.904 | 0.945 | 0.877 |

These datasets mix compression with other distortions; `ECodec` trades accuracy on the
latter for better ranking of codec output.

## Speed and memory

One thread, 95th-percentile time per comparison (AMD Ryzen 9 9950X3D, AVX-512, 2026-10-28):

| | 256² | 1024² | 2048² | 4096² | Peak memory, 4096² |
|---|---:|---:|---:|---:|---:|
| zensim `E` | 0.98 ms | 11.2 ms | 41.5 ms | 151 ms | 158 MiB |
| fast-ssim2 0.8.2 | 4.2 ms | 70.6 ms | 312 ms | 1,362 ms | 2,468 MiB |
| butteraugli 0.9.3 | 5.5 ms | 90.9 ms | 418 ms | 1,945 ms | 2,646 MiB |

With eight threads `E` takes 3.2 ms at 1024² and 10.1 ms at 2048². Method and the full grid:
[docs/PROFILES.md](docs/PROFILES.md#speed-and-memory).

## Input

sRGB 8-bit RGB or RGBA (`RgbSlice`, `RgbaSlice`, `imgref`), 16-bit and linear `f32` through
`StridedBytes`, Display P3 and BT.2020 primaries (clipped to sRGB), and `zenpixels` buffers
with the `zenpixels` feature. Both images must have the same size; images smaller than 64
pixels are reflect-padded.

## Features

`avx512`, `threads` and `imgref` are on by default. `training` exposes the linear-model
internals, `classification` the error-classification API used by `zensim-regress`, and
`zenpixels` the `ZenpixelsSource` adapter.

## License

MIT OR Apache-2.0.
