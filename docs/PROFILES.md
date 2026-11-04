# Profiles: models, training data and evaluation

Each `ZensimProfile` names a model embedded in the crate. The tables below map every profile
to its bake file and to the research record it came from. The research history (training
runs, evaluation panels, ablations) lives in [imazen/zensimx](https://github.com/imazen/zensimx);
the record for the revision-7 models is
[`benchmarks/byconstruction_2026-10-22.md`](https://github.com/imazen/zensimx/blob/main/benchmarks/byconstruction_2026-10-22.md).

| Public profile | Research / development name | Status |
|---|---|---|
| `A` (`zensim-a`) | research `F`, `mixF3` (zensimx adoption record, 2026-11-04) | Adopted general model; `latest()` and `codec_target()` |
| `PreviewV0_1`, `PreviewV0_2` | published 0.1/0.2 linear profiles | Unchanged |

The published 0.2.7 API had only `PreviewV0_1` and `PreviewV0_2`; neither is renamed.
Research names stay unchanged.

| Profile | Bake | SHA-256 | Features |
|---|---|---|---|
| `A` | `weights/rev7_mix_h128_2026-11-03.bin` | `d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a` | 492 revision-7 damage features (with full-resolution chroma) + 492 fragility products |
| `PreviewV0_1`, `PreviewV0_2` | linear weights in `src/profile.rs` | | 228 revision-1 features |

The revision-7 research models `E` and `ECodec` stay in [imazen/zensimx](https://github.com/imazen/zensimx) and are not published.

## The adopted mix (`A`)

**What it is.** `A` (research profile `F`, recipe `mixF3`, "the TID2013 + MDID mix on F3's recipe")
is a monotone network over 492 revision-7 damage features, including full-resolution chroma,
and 492 fragility products. Its 128 hidden units feed 32 tied max-pairs and a calibrated
monotone output spline. The derived-input declaration `fx2` is embedded in the bake. Its base fit keeps the SafeSyn and CID22 SSIMULACRA2 legs, the coverage pool, BAPPS and the
chroma-teacher leg of the round-3 recipe and narrows the human-rating legs to TID2013's training
split and MDID, which share the recipe's MOS weight equally; PIPAL, LIVE-MD, CID:IQ, CID:OA,
IRCCyN/IVC and Colourlab are not trained on. It was then continued for 60 epochs on the JPEG AIC-2026
public training set (weighted ×256, 48 percent of the continuation's draws) with the base legs
rehearsed; the checkpoint after epoch 59 ships (seed 0 of three; the other seeds agree on every
evaluation within their intervals). The owner adopted it on 2026-11-04 after the E36 candidate
comparison (zensimx `benchmarks/e36_subset_test_2026-11-01.md`). The complete recipe, with every
command, trainer build and hash, is `recipes/MixF3-s0.json` and `docs/RECIPE_MIXF3.md` in the private
training repository, where it replays in the release sandbox.

**Evaluation.** Frozen held-out references, read after model selection. MDID
contributed different references to `A`'s training; the other eight collections were not trained
on.

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

Two-alternative agreement with human choices: FG-IQA `A` 0.757, PieAPP 0.583.
On CLIC 2021's validation questions (learned-codec pairs, read once as a confirmation) `A` agrees
with the raters on 0.644, SSIMULACRA2 on 0.626 and butteraugli's 3-norm on 0.606. One
case of the label-free map-steering gate (an 8-pixel map-rank case on a text screenshot) scores 0.69
against its 0.70 bar; the JPEG XL loop shows no byte cost from it.

**Bake preparation.** The shipped file is the evaluated seed-0 artifact (packed bake SHA-256
`185f43f7…`) with the training-provenance and sampler-statistics entries removed; SHA-256 `d5e2a56d…`. The stripped
copy scored identically on all 1,386 surface and pair rows of the split parity run of 2026-11-04.

## HDR input

`Zensim::compute_hdr(source, distorted, encoding)` scores HDR pairs. `HdrEncoding::Linear`
takes `LinearF32Rgba` pixels in absolute cd/m². `Pq { peak_nits }` and
`Hlg { peak_nits, ambient_lux }` take 16-bit or float code values and decode them through
the declared display model. Declared primaries are converted without clipping absolute light.

The fixed front end divides linear nits by **203 cd/m²** (BT.2408 reference white,
also the Ultra HDR SDR-base convention), then uses the SDR cube-root XYB opsin without
clipping values above 1.0. There is no public white-point or front-end knob.
An SDR image shown as HDR at that white therefore follows the same feature arithmetic,
with small floating-point differences from linearizing and rescaling the input.

The [research continuity record](https://github.com/imazen/zensimx/blob/41ea3ecfc0da147e2b4525b3225f52ea3dc44c6a/benchmarks/hc1_continuity_2026-11-04.tsv)
reports a maximum absolute SDR/HDR score gap of **0.0003** for the adopted seed-0 model
on its 24-pair PNG panel. The public test checks 14 synthetic SDR pairs in both thread
modes (28 comparisons), with a measured maximum gap of **0.00055716912091697** and an
asserted bound of **1e-3 per comparison**. This is continuity evidence, not an HDR
accuracy claim: `A` was fitted to SDR content. PU21-era HDR numbers in older records
describe a different front end and do not apply to this public HDR path.

All **166** existing HDR fixture cases reproduce the research `BakeScorer`'s anchored
cube-root scores bit for bit: linear, PQ and HLG; sRGB and BT.2020 primaries; packed
and padded rows; float and 16-bit containers; both thread modes. The fixtures include
identity and nonidentity pairs. See [golden values](../zensim/tests/fixtures/hdr_a_cube_root_203.tsv)
and [reference source and commands](../benchmarks/hdr_cube_root_203_reference_2026-11-04.md).

## Parity with the research crate

For public `A` (research `F` / `mixF3`), run the research `split_parity_golden` example with
the adopted bake as its general model; it prints those rows under the first bake's label (`E`),
so relabel them `A` before the diff. Preview names stay unchanged; exclude the research development `A` rows.
Compare only profiles retained by the public crate, using each matching bake separately.

The crate was cut from imazen/zensimx on 2026-10-28. `examples/split_parity_check.rs` prints the
score, raw distance and a hash of the features and diffmaps for 231 generated image pairs,
every profile and every scoring entry point; zensimx's `split_parity_golden` example prints the
same lines. The expanded gate on 2026-10-28 checked 25,179 rows: 25,175 were
identical, four had prelisted score differences, and none had unexpected differences
or missing golden rows. The four exceptions are E/ECodec `compute_all_features`
scores whose research reference used a process-wide revision pin; their feature
hashes match. The exception list and golden data were unchanged by this audit.
`tests/profiles.rs` retains profile score goldens.

### Historical public-name rename parity (2026-11-04)

Before the research-only profiles were withdrawn, on the Ryzen 9 9950X3D all 70 profile/pair score values were bit-identical before
and after the public-name change: 14 each for adopted `A` (formerly `F`), `LegacyA`
(formerly development `A`), `E`, `ECodec` and `PreviewV0_2`. The committed 70 historical
golden constants were not changed. The revision-1 historical goldens already use
a rounding tolerance: on this machine two legacy and thirteen preview values
differ from those reference bits both before and after the rename. This check
compares the two builds on the same host, retaining the historical test contract.

[Per-case score bits](../benchmarks/profile_rename_parity_2026-11-04.tsv) and
[commands and source identities](../benchmarks/profile_rename_parity_2026-11-04.md).
The research crate was not run for this naming-only check.

### Withdrawal parity (2026-11-04)

Removing the two research-only variants and selecting `A` for `codec_target()` left
all 56 checked SDR scores bit-identical: 14 cases each for `A`, `LegacyA`,
`PreviewV0_1` and `PreviewV0_2`, compared with a frozen export of `d97076e1` on the
same Ryzen 9 9950X3D. The same 56 values also remain bit-identical after the
HDR cube-root port. [Per-case score bits](../benchmarks/public_profiles_withdrawal_2026-11-04.tsv).
All 42 retained historical score constants are unchanged. The revision-7 cross-path,
HDR and profile checks passed (2, 2 and 8 tests respectively).
