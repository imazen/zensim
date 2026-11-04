# Profiles: models, training data and evaluation

Each `ZensimProfile` names a model embedded in the crate. The tables below map every profile
to its bake file and to the research record it came from. The research history (training
runs, evaluation panels, ablations) lives in [imazen/zensimx](https://github.com/imazen/zensimx);
the record for the revision-7 models is
[`benchmarks/byconstruction_2026-10-22.md`](https://github.com/imazen/zensimx/blob/main/benchmarks/byconstruction_2026-10-22.md).

| Public profile | Research / development name | Status |
|---|---|---|
| `A` (`zensim-a`) | research `F`, `mixF3` (zensimx adoption record, 2026-11-04) | Adopted general model; `latest()` |
| `LegacyA` (`zensim-legacy-a`) | earlier 0.3.0 development `A` | Preserved scores; never shipped in 0.2.7 |
| `E` (`zensim-e`) | research `E` | Unchanged; pending an owner decision |
| `ECodec` (`zensim-e-codec`) | research `ECodec` | Unchanged; pending an owner decision; `codec_target()` |

The published 0.2.7 API had only `PreviewV0_1` and `PreviewV0_2`; neither is renamed.
Research names stay unchanged.

| Profile | Bake | SHA-256 | Features |
|---|---|---|---|
| `A` | `weights/rev7_mix_h128_2026-11-03.bin` | `d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a` | 492 revision-7 damage features (with full-resolution chroma) + 492 fragility products |
| `E` | `weights/rev7_general_h128_2026-10-27.bin` | `e460a214c1e0c4f48bc7e3bb4705df548f9f172807cc63f39ba17cb1d94e05a0` | 410 revision-7 damage features + 410 fragility products |
| `ECodec` | `weights/rev7_codec_h128_2026-10-27.bin` | `f7ecd60299482210fe33586e8d4674fe9ba163f17f2dbaf54c6e99d5ef44af63` | the same 820 inputs |
| `LegacyA` | `weights/v47_strict_qat_native_byid_2026-09-06.bin` | `de0ddb3dc78b8b0a4fc99645f949acb73e7e08bec9ed4d86793605d882e9176a` | 372 revision-1 features |
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

## The adopted mix (`A`)

**What it is.** `A` (research profile `F`, recipe `mixF3`, "the TID2013 + MDID mix on F3's recipe") shares `E`'s
architecture and adds the full-resolution chroma signals of the round-3 chroma root to its inputs:
492 damage features and 492 fragility products, with the derived-input declaration `fx2` embedded in
the bake. Its base fit keeps the SafeSyn and CID22 SSIMULACRA2 legs, the coverage pool, BAPPS and the
chroma-teacher leg of the round-3 recipe and narrows the human-rating legs to TID2013's training
split and MDID, which share the recipe's MOS weight equally; PIPAL, LIVE-MD, CID:IQ, CID:OA,
IRCCyN/IVC and Colourlab are not trained on. It was then continued for 60 epochs on the JPEG AIC-2026
public training set (weighted ×256, 48 percent of the continuation's draws) with the base legs
rehearsed; the checkpoint after epoch 59 ships (seed 0 of three; the other seeds agree on every
evaluation within their intervals). The owner adopted it on 2026-11-04 after the E36 candidate
comparison (zensimx `benchmarks/e36_subset_test_2026-11-01.md`). The complete recipe, with every
command, trainer build and hash, is `recipes/MixF3-s0.json` and `docs/RECIPE_MIXF3.md` in the private
training repository, where it replays in the release sandbox.

**Evaluation.** The same frozen held-out read as the table above, `E` repeated for comparison. MDID
contributed different references to `A`'s training; the other eight collections were not trained
on.

| Dataset | `A` | `E` | SSIMULACRA2 | Butteraugli (3-norm) |
|---|---:|---:|---:|---:|
| CID:IQ | 0.868 | 0.865 | 0.757 | 0.592 |
| CID:OA | 0.911 | 0.889 | 0.847 | 0.577 |
| Colourlab | 0.735 | 0.711 | 0.604 | 0.373 |
| IQA-ODI | 0.867 | 0.865 | 0.854 | 0.806 |
| IRCCyN/IVC | 0.919 | 0.940 | 0.949 | 0.912 |
| LIVE-MD | 0.902 | 0.894 | 0.913 | 0.827 |
| MDID | 0.927 | 0.870 | 0.787 | 0.572 |
| NITS-IQA | 0.734 | 0.706 | 0.777 | 0.706 |
| VCL@FER | 0.950 | 0.930 | 0.945 | 0.877 |

Two-alternative agreement with human choices: FG-IQA `A` 0.757 (`E` 0.759), PieAPP 0.583 (0.585).
On CLIC 2021's validation questions (learned-codec pairs, read once as a confirmation) `A` agrees
with the raters on 0.644, SSIMULACRA2 on 0.626 and butteraugli's 3-norm on 0.606. On UPIQ's HDR pairs
through the PU21 front end `A` reaches a pooled rank correlation of 0.78 against `E`'s 0.79. One
case of the label-free map-steering gate (an 8-pixel map-rank case on a text screenshot) scores 0.69
against its 0.70 bar; the JPEG XL loop shows no byte cost from it.

**Bake preparation.** The shipped file is the evaluated seed-0 artifact (packed bake SHA-256
`185f43f7…`) with the two provenance entries removed, as for `E`; SHA-256 `d5e2a56d…`. The stripped
copy scored identically on all 1,386 surface and pair rows of the split parity run of 2026-11-04.

## HDR input

`Zensim::compute_hdr(source, distorted, encoding)` scores HDR pairs. `HdrEncoding::Linear` takes
`LinearF32Rgba` pixels in absolute cd/m²; `HdrEncoding::Pq { peak_nits }` and `HdrEncoding::Hlg { peak_nits,
ambient_lux }` take `Srgb16Rgba` code values (or `LinearF32Rgba` signal values) and decode them through a display
model. Declared primaries are converted without clipping light to the SDR range, and a PU21 front-end (100 cd/m²
maps to 1.0) feeds the same revision-7 feature walk as SDR input. The current profiles were trained on SDR content;
an HDR-trained profile is in development.

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

An initial paired timing on an Intel Core Ultra 7 265K
(2026-10-28; 1024×768, one pinned thread, the two crates' `split_timing` examples alternating
for 12 rounds of 10 calls) gave median ratios of slim to research of 1.003 (`E`), 1.006
(`ECodec`), 0.998 (`LegacyA`) and 1.008 (`PreviewV0_2`); `E` took 12.75 ms per call there.
Reusing a reference with `precompute_reference` saves 8–11% per call at one thread; at eight
threads the uncached path was faster from 1024² up in that research measurement.
These observations do not establish parity for every route. The broader
[cross-path audit](CROSS_PATH_AUDIT.md) records scalar, streaming, cached, threaded
and map measurements, including inconclusive comparisons and RSS differences.

## Parity with the research crate

For public `A` (research `F` / `mixF3`), run the research `split_parity_golden` example with
the adopted bake as its general model; it prints those rows under the first bake's label (`E`),
so relabel them `A` before the diff. For the original profiles, relabel the research `A` rows
`LegacyA`; leave `E`, `ECodec` and preview rows unchanged. Compare each matching bake separately
so the adopted model's temporary `E` label cannot collide with the original `E` model.

The crate was cut from imazen/zensimx on 2026-10-28. `examples/split_parity_check.rs` prints the
score, raw distance and a hash of the features and diffmaps for 231 generated image pairs,
every profile and every scoring entry point; zensimx's `split_parity_golden` example prints the
same lines. The expanded gate on 2026-10-28 checked 25,179 rows: 25,175 were
identical, four had prelisted score differences, and none had unexpected differences
or missing golden rows. The four exceptions are E/ECodec `compute_all_features`
scores whose research reference used a process-wide revision pin; their feature
hashes match. The exception list and golden data were unchanged by this audit.
`tests/profiles.rs` retains profile score goldens.
