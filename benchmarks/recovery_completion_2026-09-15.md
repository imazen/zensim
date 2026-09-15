# Rev3 recovery, prepared maps and native HDR — September 15, 2026

**Later September 15 work order:** the
[controlling production plan](../docs/PRODUCTION_PRIORITIES_2026-09-15.md)
turns the measured blockers below into the next agent's tasks. It changes no
result, frozen artifact or gate in this report.

The implementation and frozen recovery assessment are complete. The two recovered
models beat matched B/D on the registered SDR composite, and prepared maps are
faster. **Neither model qualifies as an all-purpose replacement.** Both fail
CID22-band non-regression, codec addressability, targeting tails, independent
native spatial RD and HDR human ranking. No default or shipped profile changed.

This supersedes the unfinished-work list in the September 14
[optimization report](rev3_native_optimization_2026-09-14.md). It does not erase
the failed earlier rewrite or change its measurements. The immutable
[registration](completion_2026-09-15.md) precedes fitting and public assessment.
The artifact packet is `recovery-completion-2026-09-15/` beside the served
summer gauntlet; its report, frozen models, full verdicts, commands, receipts,
raw panels, codec outputs and replay archive are downloadable. Local authority:
`/var/tmp/zensim-validation-2026-09-15/`.

## Results on the same assessment populations

The composite is the existing registered multi-panel score, not MAE selection.
It includes the canonical ranking/band/panel treatment. Full verdicts retain
outlier ratios, percentile envelopes, scatter geometry, clumping, saturation,
signed predictions and per-pair values. The board renders all nine scatter
panels for both candidates. Original imazen/nonphoto/proxy targets retain their
source definitions; those are not newly collected human judgments.

| Model | Composite | CID22 SROCC | JPEG M3a | Dial p5 / p95 | Monotonic |
| --- | --- | --- | --- | --- | --- |
| Fast Y60/H32 ×5 | 0.846044 | 0.858900 | 0.889278 | -15.00 / 93.90 | 98.71% |
| Basic228/H128 ×5 | 0.875950 | 0.882889 | 0.840711 | -15.89 / 93.94 | 98.20% |

Matched B is **0.838237**, matched D **0.830760** on the same nine-corpus
assessment. The preceding Y60/H32 rewrite was **0.757601**, full944/H128
**0.548957**. B/D use their own Rev1 features; feeding them Rev3 caches would
invalidate the comparison. Complete composite coverage is six of six required
panels. The complete assessment covers six EVAL corpora plus published CID22
(4,292), AIC3 (600) and AIC4 (300) TEST rows where no EVAL exists, a 9,593-row
five-codec floor-dense ladder, and 27 JPEG repair cells per composition.

A higher aggregate does not establish non-regression in every important band:

| CID22 band | Matched B | Fast | Basic228 |
| --- | --- | --- | --- |
| B0-B6 | 0.702600 | 0.691606 | 0.709376 |
| B7 | 0.385247 | 0.360898 | 0.385961 |
| B8-B9 | 0.509198 | 0.458658 | 0.479244 |

## What changed in the implementation

The existing basic/peak extraction walk now retains the signals and final
statistics needed after complete model sensitivities are known. Map assembly
reuses them instead of extracting them again. Inactive X/B channels are not
copied. This applies to the declared basic/peak route; wide and fractional
feature plans retain their existing owners and fallback. There is no new
feature arithmetic or Python serving path.

`BakeScorer::prepare_steering_hdr(source, encoding, bin)` binds the reference
and native viewing/input contract to the existing steering session. PQ16, HLG
and absolute linear float retain their precision and declared primaries. The
same model composition, score, finite sensitivities and rectangle refinement
are used. Basic/peak IDs below 228 are supported; unsupported feature families,
fractional HDR sampling and incompatible contracts refuse explicitly. Hard
maxima remain non-additive rectangle terms: summing separate tile estimates is
not an estimate for their union. Optional finite L2/L4/L8 removal retains the
existing frozen-signal/model approximation, not an exact prediction of a codec edit.

Candidate reference pyramids now follow canonical natural-width geometry,
including odd widths and tiny reflection padding. Public legacy reference
construction is unchanged. Mixed session entrypoints clear retained validity
so a previous comparison cannot supply stale signals.

The native JXL candidate adapter had checked density-only coverage and rejected
max terms, despite their support in `ScoredAttribution::refinement_gain`.
Candidate transform tiles now query that existing complete owner and check
`unsupported_refinement_feature_ids`; named-profile controls retain their
original additive query. This correction was registered after the observed
refusal; models, controller settings, requests and TRAIN families stayed fixed.
Both full native matrices were rerun, including TRAIN seed calibration. All
**1,782** old/new fast-model records match encoded/decoded hashes and scores
exactly. Neither fit nor model choice used this repeated EVAL.

## Runtime and memory

Ryzen 9 9950X3D, release without target-cpu=native, one pinned worker,
30 accepted interleaved rounds per cell, same final five-member ensemble bytes
before/after. All numbers below are milliseconds per complete comparison at
1024×1024, not milliseconds per megapixel. Prepared rows exclude one-time
reference preparation and include score, map, maxima and **opt-in finite
moment refinement**. Scalar rows include reference work. Before/after input
geometries and model compositions match.

| Input | Model | Scalar p95 | Prepared p50 / p95 | Previous prepared p50 | Median reduction |
| --- | --- | --- | --- | --- | --- |
| rgb8 | Fast Y60/H32 ×5 | 9.77 | 26.57 / 26.95 | 36.12 | 26.4% |
| rgb8 | Basic228/H128 ×5 | 18.79 | 91.08 / 91.73 | 106.96 | 14.8% |
| sdr16 | Fast Y60/H32 ×5 | 11.15 | 27.72 / 28.17 | 37.65 | 26.4% |
| sdr16 | Basic228/H128 ×5 | 19.58 | 91.97 / 92.45 | 108.84 | 15.5% |
| linear-p3 | Fast Y60/H32 ×5 | 10.72 | 27.32 / 27.84 | 37.55 | 27.2% |
| linear-p3 | Basic228/H128 ×5 | 19.25 | 91.31 / 92.08 | 108.48 | 15.8% |
| hdr-pq16 | Fast Y60/H32 ×5 | 25.30 | 35.42 / 35.83 | new API | — |
| hdr-pq16 | Basic228/H128 ×5 | 34.57 | 100.26 / 101.28 | new API | — |
| hdr-linear2020 | Fast Y60/H32 ×5 | 23.15 | 34.23 / 34.82 | new API | — |
| hdr-linear2020 | Basic228/H128 ×5 | 31.61 | 98.78 / 99.75 | new API | — |

At 2048×2048 RGB8, scalar p95 is **37.35 / 79.40 ms**, prepared
**116.49 / 374.04 ms** (fast/rich). The fast model satisfies the absolute
scalar latency bars in this instrument. The richer model's prepared/scalar
ratio exceeds 3×; the fast model also slightly exceeds 3× at 2048×2048.
D/fast-SSIM2 relative-latency release clauses were not measured in this run.
Neither partial evidence nor the absolute timing pass completes G-PERF.
Five admitted real TRAIN pairs were also measured in both arms; all raw cases,
p50/p95 values, process records and tool hashes are in `performance/RESULT.json`.
Two initial SDR16 prepared runs were rejected by the strict resource gate;
retained retries passed under the same 30-round rule. No rejected sample was
relabeled reliable or selected for its speed.

Whole-process peak RSS for 2048×2048 prepared RGB8 was **283 / 546 MiB**;
linear BT.2020 HDR **427 / 674 MiB**. These include caller input buffers,
reference, scratch and maps. They are conservative totals, not isolated
incremental allocations; the incremental memory release clause is not claimed
from these totals. Retention trades memory for eliminating duplicate work.

## Native color, HBD and HDR evidence

Both final ensembles pass **214 native SDR pairs** and **495 native HDR
pairs**, checking canonical consumed features, cached/pixel scalar scores,
prepared HDR score/coverage, finite rectangle gains and identity. HDR audit
selects one minimum-area reference variant per admitted TRAIN source family
and all 15 producer/quality variants. Native SDR inputs and numerical tests
include 16-bit samples and linear P3 floats; this is no longer an RGB8-only
implementation exercise. Low-bit PQ/HLG sensitivity, independent P3/BT.2020
matrices, odd/tiny dimensions, serial/parallel paths, errors and session reuse
are covered by the native tests.

ICC bytes are not interpreted by Zensim itself. Inputs in different ICC profiles
must first be transformed through the color-management owner into a declared
common linear space; simply relabeling them sRGB is invalid. Native primary
conversion then uses that declaration. These checks establish native precision
and declared-primary behavior, not arbitrary ICC-profile corpus certification.

The corrected HDR extractor restores all 72 live peak columns and preserves
actual PQ cICP metadata in PNG/JXL. The JXL requested pixel descriptor selects
storage; authoritative primaries/transfer come from codestream cICP. Treating
the descriptor's defaults as authoritative previously mislabeled source color.
Old all-zero-peak or mislabeled HDR caches were not relabeled or reused.

Corrected TRAIN extraction contains **7,425 pairs / 495 reference variants /
33 source families**, with original reservations preserved. References are
195 BT.709-PQ and 300 P3-PQ variants. Original exclusions remain excluded;
no secret holdouts were read. Both native CVVDP and PU-SSIM2 were recomputed on
all 7,425 pairs, zero failures. New zenmetrics CLI contract:
`score-pairs --hdr --hdr-common-primaries`, footer
`common-bt709-pq-native-v3`. It refuses a legacy feature sidecar and unsupported
fallbacks. The former CPU CVVDP route could reach the 8-bit shell; the new
explicit route calls the existing native HDR scorer.

The CVVDP judge models a **BT.709 HDR display with reference-measured peak**;
primary conversion preserves wide/negative linear values before the display's
own gamut clipping. This is not P3-display qualification. Fresh judge outputs
retain the exact producing binary hash and complete decoder/input receipts.

The shared-model hypothesis was assessed on TRAIN before exposing human EVAL:
CVVDP rank was 0.9616 / 0.9176 and PU-SSIM2 0.8912 / 0.8867. These promising
TRAIN transfer results did not justify asserting a shared calibration. The
frozen SDR-trained models were then assessed unchanged on native UPIQ EXRs,
using **zenexr and the upstream Rust EXR crate**, through the Rust public API.

| UPIQ native HDR, 380 conditions / 30 references | SROCC | Raw PLCC | Negative predictions |
| --- | --- | --- | --- |
| Matched BHdr | .753314 | .733426 | 12 |
| Fast | .696017 | .665968 | 148 |
| Basic228 | .704373 | .576420 | 162 |

Both regress against BHdr. Signed ranges are fast −127.14..91.29 and rich
−214.08..91.40. No predictions exceed 100. The valid numerical HDR path and
its calibrated perceptual quality are separate questions: **this experiment
rejects shipping these SDR weights as the unified HDR model**. No UPIQ-driven
refit, spline change, seed selection or gate relaxation followed.

## Codec targeting and spatial value

Bounds were measured before scored steering and kept hidden from runtime.
Nine canonical TRAIN families calibrate per-codec seeds; eight original
validation families assess all frozen requests at 1/2/3 shots, with midpoint
controls. The existing attainable-witness rules distinguish impossible or
uncertain requests from misses. JPEG/JXL/WebP scalar targeting includes
1,884 cells and 48 bound cells. AVIF and HDR targeting are not qualified here.

| Model | Codec | Shots | Median error | p95 error | Worst error | Attainable n |
| --- | --- | --- | --- | --- | --- | --- |
| Basic228/H128 ×5 | jpeg | 1 | 4.428 | 13.367 | 18.622 | 50 |
| Basic228/H128 ×5 | jpeg | 2 | 0.860 | 5.107 | 12.110 | 50 |
| Basic228/H128 ×5 | jpeg | 3 | 0.637 | 3.815 | 6.003 | 50 |
| Fast Y60/H32 ×5 | jpeg | 1 | 4.257 | 12.957 | 17.702 | 51 |
| Fast Y60/H32 ×5 | jpeg | 2 | 0.952 | 5.490 | 6.540 | 51 |
| Fast Y60/H32 ×5 | jpeg | 3 | 0.437 | 2.945 | 5.021 | 51 |
| Basic228/H128 ×5 | jxl | 1 | 1.175 | 19.502 | 24.549 | 52 |
| Basic228/H128 ×5 | jxl | 2 | 0.283 | 10.644 | 21.393 | 52 |
| Basic228/H128 ×5 | jxl | 3 | 0.283 | 3.078 | 8.984 | 52 |
| Fast Y60/H32 ×5 | jxl | 1 | 0.919 | 20.589 | 28.516 | 54 |
| Fast Y60/H32 ×5 | jxl | 2 | 0.568 | 8.693 | 17.093 | 54 |
| Fast Y60/H32 ×5 | jxl | 3 | 0.266 | 2.330 | 13.473 | 54 |
| Basic228/H128 ×5 | webp | 1 | 2.866 | 23.068 | 26.869 | 52 |
| Basic228/H128 ×5 | webp | 2 | 0.937 | 15.373 | 21.852 | 52 |
| Basic228/H128 ×5 | webp | 3 | 0.624 | 7.502 | 17.814 | 52 |
| Fast Y60/H32 ×5 | webp | 1 | 2.637 | 25.029 | 31.110 | 55 |
| Fast Y60/H32 ×5 | webp | 2 | 0.749 | 13.728 | 17.484 | 55 |
| Fast Y60/H32 ×5 | webp | 3 | 0.531 | 11.786 | 17.484 | 55 |

Release bars remain median/p95 <=2/8, <=1/3, <=0.5/1, with three-shot
maximum <=3 and the registered undershoot constraints. Both models fail the
necessary p95 clauses. Bounds, all requests/dispositions, fixed-request
coverage, class panels, shot costs and midpoint controls are retained.

Both complete native JXL runs use scalar/neutral/active controls and independent
SSIM2/Butteraugli judges. There are **756 exact neutral ladder pairs** across
smoke/TRAIN/EVAL and both models. Two judges cover all **2,304 decoded pairs**.
The fast active three-shot p95 is 1.969; richer 1.902, both above 1. Sparse
matched-quality interpolation reveals content-class regressions on both
judges. Example richer active-vs-scalar arithmetic savings: photo −1.08%
Butteraugli / −0.41% SSIM2, screen −1.25% Butteraugli, graphic −1.80% SSIM2.
A negative arithmetic saving also rules out nonnegative geometric saving;
these already establish failure without an invented aggregate. No extrapolated
or self-judged benefit is called a pass. Source-bootstrap uncertainty and
matched-quality confirmation remain required for a positive release claim.

## Training, chronology and qualification

All fitting, transforms, normalization, checkpoint selection and spline
calibration used TRAIN only. Two exact feature plans (Y60/H32, basic228/H128),
five fixed seeds (17101, 17103, 17107, 17111, 17113), disjoint sampler streams,
120×50,000 draws per fit. Canonical source-family unions and exact reference
pixel duplicates were admitted before fitting; earlier reserved TRAIN test
segments remain reserved. The SafeSyn/CID22 full Rev3 TRAIN caches and admitted
human/modern-codec legs yield 168,142 fit rows after exact within-leg dedup.
All five samplers cover every usable fit row. The human leg is rank-only;
SafeSyn and CID22 retain signed same-buffer SSIM2 targets. Existing Rust
training/packing/calibration owners produce float models with negative-tail
splines and identity anchors; 2,021 TRAIN calibration anchors, 21 at identity.
All five final members have equal 0.2 weight, with no seed pruning.

Ten fits completed in 3,301 seconds wall time at two concurrent jobs. Per-fit
fast 416–442 seconds, richer 767–821 seconds. These recovery runs deliberately
restore broad training; they are not advertised as under-five-minute feature
experiments. Sixteen mathematically undefined small-group development
statistics are explicitly null with their paths recorded, not dropped rows.

`FROZEN.json` binds final bytes before the nine-corpus public assessment;
`HDR_TRAIN_DECISION.json` predates the UPIQ read. Earlier public exposures are
linked, published TEST is used only where no EVAL exists, and secret holdouts
remain untouched. Failed launches and noisy timing attempts are preserved
separately. Replays to correct strict JSON composition fields or adapter wiring
never changed model bytes or selected a recipe from evaluation results.

The existing `freeze_check --qualify` reports **failed** for both models, not
“not evaluated.” It records G-RANK/G-RD/G-TARGET and addressability failures.
The legacy EVAL manifests lack the decoder declaration required by formal table
qualification, despite retained producing binaries/pixel receipts; that field
is not fabricated. Separate legacy negative-tail/identity probe inputs were
not admitted to this new arithmetic lane, so corresponding G-ADDR contract
clauses remain not measured. The fresh real-pixel identity/parity tests are
reported as their own evidence, not silently substituted for a different probe.
No qualified corruption companion is attached. Its threshold, honest-codec
false positives and real bug detection remain a separate required product leg.

Implementation checks passed: complete root zensim tests with custom profiles,
training, threads, corruption-head and feature-regime-v2; no-default library
suite; explicit Rev3 native HDR tests; scoped clippy; HDR extractor test/lint;
zenmetrics native-primary and CLI-contract tests; JXL candidate example build
and lint, full native controls; API snapshot check and semver check. One new
public method, no breaking API delta. The historical per-bake test was updated
to the September 13 owner: Rev2 changes SSIM, HF gain and global contrast.
It independently checks HF saturation r2=r1/(1+r1), preserving within-revision
parity; other cross-instantiation basics allow four f64 ULPs.

The next product work is now narrowly identified: TRAIN-only HDR supervision
under the corrected contract, cheaper peak/moment map construction, complete
native codec interventions and stronger train-calibrated targeting, formal
input/probe admission, and a independently qualified corruption companion.
Reuse these Rust owners and frozen failures; do not restart feature science or
retune against these public results. These recovered artifacts are executable
research candidates, not new production defaults.
