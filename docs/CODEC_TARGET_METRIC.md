# Codec-target metric — the one user-facing quality dial

**Current integration guide, checked against source on 2026-09-07.**
`ZensimProfile::codec_target()` and `latest_preview()` return **B**.
D is available explicitly; the September 5 change replaced D’s calibration,
not the `codec_target()` alias. The implementation in
[`profile.rs`](../zensim/src/profile.rs) determines routing and embedded bytes.

**User ruling, later 2026-09-07:** no consumers have calibrated to B, C or D;
all may be improved/replaced. The mapping below records today's implementation,
not a frozen score contract. New models must be fully executable and servable
in Rust through a zensim surface API, and evaluation must use that API,
including all heads, corruption gates, splines and final composition.

## What the user controls

A user chooses a target such as **80**. The encoder chooses its own quality
parameter, measures the decoded result, and corrects it within its work budget.
A picker may choose the codec as well. JPEG quality, WebP quality and JXL
distance are internal controls with different scales and directions; users
should not need to translate between them or select a research model.

The product objective is small output at the requested perceived quality, with
low total latency and memory use. An undershoot loses quality; an overshoot
spends extra bytes. A fast metric matters on every correction pass, and a good
starting estimate saves the entire encode/decode/score pass. A fast model
forward alone does not establish a fast encoder loop.

The intended score contract is:

- **100 represents identity**, and a distortion must not score above a perfect
  copy. Test the model path as well as the API’s identical-input shortcut.
- **Negative scores are valid.** Keep them in scoring, reports and chart axes.
  There is no prescribed negative depth such as −50. Each codec’s lowest
  configurable, distinct outputs must remain representable.
- The usable range must retain resolution at both ends, especially near
  lossless. A monotone calibration cannot repair an inversion in raw predictions.
- A given target should imply consistent perceived quality across images,
  content types and codecs. Human near-threshold evidence and independent
  judges test that claim; agreement with a teacher metric alone does not.
- A steering map must help the encoder improve the same scalar it targets.
  Its value is measured in real bytes, achieved quality and passes.

These are requirements, not a claim that every available profile meets them.
In particular, **“60 = visually lossless/JND” is an old calibration convention,
not a validated universal promise for the current profiles**. The current
exam is [`MODEL_SELECTION_SCORECARD.md`](MODEL_SELECTION_SCORECARD.md), including
G-ADDR. Its floor rule is named in every comparison; the September ladder
ruler uses `resolvable` with margin 0.5 against SSIMULACRA2’s per-codec result.
See the [ladder measurement](../benchmarks/board_ladder_ruler_2026-09-06.md).

## Current variant → backing bake mapping

All filenames below are relative to `zensim/weights/`. Hashes are SHA-256
prefixes checked from those files on 2026-09-07. Each bake has a same-stem
TOML under [`weights/manifests/`](../zensim/weights/manifests/).

| Public variant | Current bake | Role |
|---|---|---|
| **B** (`zensim-b`) | `b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin` · `a96a5a66…` | Default SDR scorer; `codec_target()` and `latest_preview()` |
| **BHdr** (`zensim-b-hdr`) | `bhdr_linear_shaped_cvvdpmix_byid_2026-09-06.bin` · `8461ac55…` | Default HDR route through the absolute-luminance entry points |
| **C** (`zensim-c`) | `c_sdr_purity944_2026-08-29.bin` · `61ebc456…` | Available SDR candidate; not the default |
| **CHdr** (`zensim-c-hdr`) | `c_hdr_l1t1944_2026-08-29.bin` · `0a437d99…` | Available HDR candidate; BHdr remains the default |
| **D** (`zensim-d`) | `d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin` · `cd1098b4…` | Fast SDR profile; 28 declared feature IDs, 1,420-byte bake |
| **A** (`zensim-a`) | `v47_strict_qat_native_byid_2026-09-06.bin` · `de0ddb3d…` | Deprecated profile |
| `PreviewV0_1`, `PreviewV0_2` | In-source weight arrays | Historical compatibility profiles |

C/CHdr/D use the default-on `candidate-profiles` feature; A uses
`deprecated-profiles`. The September 6 explicit-ID conversion of A/B/BHdr/D
preserved served scores relative to their prior bakes. See
[dense bake flip](../benchmarks/dense_bake_flip_2026-09-06.md) and the
[build-feature serving fix](../benchmarks/dense_serving_ungate_2026-09-06.md).

**C and CHdr can now be served through `Zensim::compute`.** The older
“944-wide bakes cannot be served” limitation was closed by the feature-plan
work. Their remaining defect is different: the serving `append2_dst_activity`
toggle disagrees with training. Changing that behavior and densifying those
two profiles remain open in [D3](OPEN_DECISIONS_2026-09-06.md#d3-profile-c--chdr-serving-toggle-append2_dst_activity).

Do not identify feature semantics by vector width. Bakes declare IDs; producer
and consumer must also agree on feature revision and decoder/extraction
provenance. [`FEATURE_SET_IDS.md`](FEATURE_SET_IDS.md) owns that contract.

## Integration and real callers

For ordinary SDR integration, construct the metric with
`Zensim::new(ZensimProfile::codec_target())` and score source and reconstruction
using the same pixel/color/alpha interpretation. Use the HDR absolute-luminance
API for HDR; sending HDR through an SDR pixel path does not establish a common
SDR/HDR scale.

Existing callers and their algorithm owners, checked in the sibling source:

| Consumer | Entry point / owner | What it controls |
|---|---|---|
| zenjpeg | `Quality::Zq` / `ZqExplicit`; `zenjpeg/zenjpeg/src/encode/zq.rs` | Starting quality, correction passes and optional block-artifact bound |
| zenwebp | `LossyConfig::with_target_zensim`; `zenwebp/src/encoder/zensim_target.rs` | Content-specific starting quality, secant correction, asymmetric tolerances |
| zenavif | `TargetMetric::Zensim`; `zenavif/src/target_quality.rs` | Encode/decode/score target search using `codec_target()` |
| jxl-encoder | `jxl-encoder/jxl-encoder/src/vardct/zensim_loop.rs` and `perceptual_loop.rs` | Scalar target and quantization-field refinement; pin the configured driver when comparing models |
| zensim-target | [`target_search`](../zensim-target/src/lib.rs) | Cross-codec search with a bounded iteration budget and best-probe result |

The codec-native JPEG/WebP paths use a quality floor and separate overshoot
and undershoot handling. Their defaults permit best-effort output; inspect
achieved quality or request the codec’s strict failure policy when needed.
The `zensim-target` helper instead reports symmetric target tolerance and a
`converged` result. Do not silently equate those success conditions in an eval.

**CLI default aligned during the September 7 cleanup:** `zensim-target`'s
library, CLI default and `--profile default` all use `codec_target()`.
Use `--profile tuner-v4` to reproduce the previous CLI default. See its
[checked usage](../zensim-target/README.md). This change does not establish
that every sibling consumer uses the same model.

The product loop is: predict an initial setting → encode/reconstruct → score →
accept or correct. Reuse source-side work where the actual scoring API supports
it. Keep the best valid candidate and record failures to reach the target;
an exhausted search budget is not evidence of success. Lossless codecs do not
have a lossy quality dial.

In-encoder per-block RDO is a separate cadence from an image-level correction
loop. Do not insert a full image comparison into thousands of RDO decisions;
see the dated [feasibility study](RDO_LOSS_FEASIBILITY_2026-05-24.md).

## Evaluate the delivered output

For every candidate, report target error (including undershoot separately),
target-hit rate, bytes, number of passes, total encode/decode/score time and
peak memory, per codec and content class. Include tails and failure examples;
a pooled median curve can hide an individual source’s inversion.

Compare bytes at equal **independently judged** quality. A model’s own score
cannot establish its RD improvement. Keep human holdouts distinct from
teacher-agreement corpora, and grade the high-fidelity range independently.
The [scorecard](MODEL_SELECTION_SCORECARD.md),
[wave playbook](WAVE_PLAYBOOK.md), and
[reproducibility contract](REPRODUCIBILITY.md) own the evaluation workflow.

**Research selection is not release qualification.** `freeze_check --select`
can select a recipe with incomplete G-ADDR coverage or failed codec floors.
The addressability owner’s `Verdict::shippable()` requires both tiers to pass;
other product gates must pass too. The September 6 board record demonstrates
this distinction on the selected fast-class recipe. D has the strongest
recorded five-codec floor result among the compared zensim families, while
stronger-rank MLP candidates retain floor and/or near-threshold tradeoffs.
No new common-dial winner is declared by this guide.

## Reproducibility and chronology

A public profile name is not a frozen model hash. Pin the bake, extractor
revision, decoder versions, calibration and target-predictor lineage for a
reproducible experiment. A dial calibration change may require rescoring data
and retraining/recalibrating experimental starting-quality predictors even
when rankings are unchanged. This is an experiment-reproducibility concern;
the user confirms there is no deployed consumer calibration to B/C/D to preserve.

- **May 2026:** Balanced/Compression/Tuner experiments and provisional JND/JOD
  anchors; those variant names and integration examples are historical.
- **July 12:** the codec-target alias became B.
- **August 29:** C rotated to the purity-trained bake; CHdr was added.
- **September 5:** D’s id100+negative-tail calibration replaced its prior dial;
  the proposed peaks model did not ship.
- **September 6:** A/B/BHdr/D switched to explicit-ID bakes without changing
  served scores; the board adopted the floor-dense ladder ruler.

Training lineage remains in [B’s methodology](../benchmarks/profile_b_methodology_2026-07-12.md),
[BHdr’s record](../benchmarks/bhdr_improvement_split_lineage_2026-07-12.md),
[C’s purity wave](../benchmarks/sdr_pure_retrain_wave_2026-08-28.md),
[CHdr’s wave](../benchmarks/hdr944_retrain_wave_2026-08-28.md), and
[D’s dial change](../benchmarks/d_ship_flip_2026-09-05.md).
The older version of this guide is retained in Git history; its May numbers
must not be treated as current product guarantees.
