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
| `A` (`latest()`, `codec_target()`) | `zensim-a` | General purpose, the default: compression, noise, blur, colour and contrast changes |

`A` is a monotone network over multi-scale SSIM, edge, detail-loss and
high-frequency features in XYB: more damage never raises the score. Severe damage
can score below 0. Scores are bit-identical across x86 SIMD tiers and thread counts.
Training data and evaluation details: [docs/PROFILES.md](docs/PROFILES.md). How it works and how it was
built: [docs/MODEL_A_REPORT.md](docs/MODEL_A_REPORT.md).
Explore its features, score response and weights in the browser: [imazen.github.io/zensim](https://imazen.github.io/zensim/).

The 0.1/0.2 linear profiles are available with the `legacy-models` feature for score continuity with 0.2.7.

## Accuracy

Spearman rank correlation with human ratings on held-out references, read after
the adopted model was frozen. MDID contributed different references to training;
the other eight collections below were not trained on. The peer metrics were
trained on none of these collections.

| Dataset | `A` | SSIMULACRA2 | Butteraugli (3-norm) |
|---|---:|---:|---:|
| CID:IQ | 0.868 | 0.757 | 0.592 |
| CID:OA | 0.911 | 0.847 | 0.577 |
| Colourlab | 0.735 | 0.604 | 0.373 |
| IQA-ODI | 0.867 | 0.854 | 0.806 |
| IRCCyN/IVC | 0.919 | 0.949 | 0.912 |
| LIVE-MD | 0.902 | 0.913 | 0.827 |
| MDID | 0.927 | 0.787 | 0.572 |
| NITS-IQA | 0.734 | 0.777 | 0.706 |
| VCL@FER | 0.950 | 0.945 | 0.877 |

A separate [registered frozen-A assessment](docs/FINAL_READ_A_2026-11-05.md) is complete on all eight populations.
KADID TERMINAL failed both registered limits against two research models. CLIC 2022 majority agreement is
0.747 for A, 0.736 for SSIMULACRA2 and 0.688 for butteraugli; the record gives intervals and exposure disclosures.

## Input

sRGB 8-bit RGB or RGBA (`RgbSlice`, `RgbaSlice`, `imgref`), 16-bit and linear `f32` through
`StridedBytes`, Display P3 and BT.2020 primaries (clipped to sRGB), and `zenpixels` buffers
with the `zenpixels` feature. Both images must have the same size; images smaller than 64
pixels are reflect-padded.

`compute_hdr` decodes declared linear, PQ or HLG input to absolute light and uses
the anchored cube-root front end at a fixed 203 cd/m² SDR white, retaining brighter
light without clipping. Details: [HDR input](docs/PROFILES.md#hdr-input).

## Features

`avx512`, `threads` and `imgref` are on by default. `training` exposes the linear-model
internals, `classification` the error-classification API used by `zensim-regress`, and
`zenpixels` the `ZenpixelsSource` adapter.

## License

MIT OR Apache-2.0.
