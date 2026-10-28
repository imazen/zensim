# Profiles: models, training data and evaluation

Each `ZensimProfile` names a model embedded in the crate. The tables below map every profile
to its bake file and to the research record it came from. The research history (training
runs, evaluation panels, ablations) lives in [imazen/zensimx](https://github.com/imazen/zensimx);
the record for the revision-7 models is
[`benchmarks/byconstruction_2026-10-22.md`](https://github.com/imazen/zensimx/blob/main/benchmarks/byconstruction_2026-10-22.md).

| Profile | Bake | SHA-256 | Features |
|---|---|---|---|
| `E` | `weights/rev7_general_h128_2026-10-27.bin` | `e460a214c1e0c4f48bc7e3bb4705df548f9f172807cc63f39ba17cb1d94e05a0` | 410 revision-7 damage features + 410 fragility products |
| `ECodec` | `weights/rev7_codec_h128_2026-10-27.bin` | `f7ecd60299482210fe33586e8d4674fe9ba163f17f2dbaf54c6e99d5ef44af63` | the same 820 inputs |
| `A` | `weights/v47_strict_qat_native_byid_2026-09-06.bin` | `de0ddb3dc78b8b0a4fc99645f949acb73e7e08bec9ed4d86793605d882e9176a` | 372 revision-1 features |
| `PreviewV0_1`, `PreviewV0_2` | linear weights in `src/profile.rs` | | 228 revision-1 features |

## The revision-7 models (`E`, `ECodec`)

**Architecture.** A monotone network over 410 damage features (13 basic SSIM, edge and
high-frequency signals and 28 bounded "v2" signals, at the Y channel of the full-resolution
scale and the X, Y and B channels of three coarser scales) plus 410 fragility products: each
damage feature times `0.02 / (mean |∇ reference| + 0.02)`, so the same damage counts more on
flat content. The first layer's weights are non-negative and its biases non-positive, so no
input can raise the score by growing. 128 hidden units feed 32 tied max-pairs, and a monotone
spline calibrates the output against SSIMULACRA2 scores of the CID22 training references.
Weights are stored as f16.

**Training.**

- `E` (research name `MlicXref256e59`) starts from the licence-clean `Mlic` mix: synthetic
  codec ladders scored by SSIMULACRA2 (141,054 pairs), SSIMULACRA2 scores of CID22's training
  references, coverage pairs for light, spatial and newer distortion families, and the training splits of KADID-10k, TID2013, KonFiG-IQA,
  CID22 (the 25 training references), PIPAL, MDID, LIVE-MD, CID:IQ, CID:OA, IRCCyN/IVC,
  Colourlab and BAPPS. It was then continued for 60 epochs on the JPEG AIC-2026 public
  training set (weighted ×256) together with a rehearsal leg that keeps each older human
  dataset's own cross-reference order; the checkpoint after epoch 59 ships.
- `ECodec` (`MlicAICOnlye29`) continues the same `Mlic` model on the JPEG AIC-2026 public
  training set alone; the checkpoint after epoch 29 ships. It ranks compression artifacts
  better than `E` and generalises worse to other distortion types.

The AIC-2026 labels are distributed to JPEG AIC members only and are not part of this
repository; results on that set are not published here.

**Evaluation.** Both models were frozen before one read of the held-out evaluation splits
(references never used in training or selection). Spearman rank correlation with human
ratings:

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

NITS-IQA, IQA-ODI and VCL@FER were never trained on; the other six contributed different
references to training. The peers are zero-shot on all nine. Two-alternative agreement with
human choices: FG-IQA `E` 0.759 / `ECodec` 0.761, PieAPP 0.585 / 0.574, BAPPS 0.667 / 0.667.

**Bake preparation.** The shipped files are the evaluated artifacts (SHA-256 `7642eff4…` for
`E`, `63a1c5d3…` for `ECodec`) with two metadata entries removed: the training-provenance
block (`zentrain.repro`) and the sampler statistics (`zentrain.sample_coverage`). The weights,
feature declaration, formula revision, derived-input declaration and calibration spline are
unchanged; both versions score identically on every surface of the split parity run.

## Speed and memory

Measured 2026-10-28 on an AMD Ryzen 9 9950X3D (AVX-512), on the research crate this one was
cut from, which produces bit-identical scores: 95th-percentile time per comparison from 32
paired rounds under a quiet-machine gate, and peak resident memory from one fresh process per
measurement.

| Threads, size | `E` | `ECodec` | fast-ssim2 0.8.2 | butteraugli 0.9.3 |
|---|---:|---:|---:|---:|
| 1, 64² | 0.21 ms | 0.21 ms | 0.36 ms | 0.59 ms |
| 1, 256² | 0.98 ms | 0.99 ms | 4.16 ms | 5.55 ms |
| 1, 1024² | 11.2 ms | 11.2 ms | 70.6 ms | 90.9 ms |
| 1, 2048² | 41.5 ms | 41.1 ms | 312 ms | 418 ms |
| 1, 4096² | 151 ms | 150 ms | 1,362 ms | 1,945 ms |
| 8, 1024² | 3.2 ms | 3.2 ms | 76.4 ms | 34.5 ms |
| 8, 2048² | 10.1 ms | 10.1 ms | 311 ms | 205 ms |
| 8, 4096² | 63.5 ms | 59.6 ms | 1,370 ms | 900 ms |

Peak memory at 4096², one thread: `E` 158 MiB, fast-ssim2 2,468 MiB, butteraugli 2,646 MiB.
Reusing a reference with `precompute_reference` saves 8–11% per call at one thread; at eight
threads the uncached path is faster from 1024² up.

## Parity with the research crate

The crate was cut from imazen/zensimx on 2026-10-28. `examples/split_parity_check.rs` prints the
score, raw distance and a hash of the features and diffmaps for 231 generated image pairs,
every profile and every scoring entry point; zensimx's `split_parity_golden` example prints the
same lines. The two outputs were identical (6,930 rows) on an Intel Core Ultra 7 265K, and
`tests/profiles.rs` keeps 56 of those scores as golden values.
