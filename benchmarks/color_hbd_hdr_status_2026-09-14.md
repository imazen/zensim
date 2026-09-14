# Color, high bit depth, HDR, and the September 13–14 discussion

Status at source revision `16be227e44794a3cb25d68c8c9a95077596ad0c9`:
**core numerical checks pass; end-to-end color/HBD/HDR qualification of the
new Rev3 models is incomplete. No new model is shippable.** This review changes
no model, score, dataset role, or acceptance threshold. Fresh checks below use
synthetic fixtures and existing packaged weights, not protected source data.

## What is correct, and what is not yet established

| Area | Current implementation/evidence | Remaining limitation |
|---|---|---|
| Declared SDR primaries | The public source API supports sRGB, Display P3 and BT.2020. Fresh color/format and real-JPEG gamut tests pass. | The caller must supply the correct interpretation; this does not prove embedded profiles survive file decoding. |
| Embedded ICC in recent science | The TRAIN audit found 33 P3 PNG references affecting 213 fitting pairs; retained decoded samples match raw untransformed RGB8. | The shared PNG evaluation adapter drops ICC metadata. Earlier product-packet color semantics are limited; exact replay does not repair this. |
| Wide-gamut clipping | `GamutMapping::Preserve` distinguishes destructive clipping in the real zenjpeg fixtures. | Default `Clip` can hide gamut loss. Preserve uses a scalar unclamped conversion, and shipped profiles were not trained on its out-of-gamut distribution. It is not a calibrated substitute. |
| High-bit-depth SDR | `Srgb16Rgba` and linear-f32 input paths exist. P3/BT.2020 format equivalence passes. | Those public format fixtures originate in 8-bit samples. They do not establish sub-8-bit sensitivity or real 10/12/16-bit codec behavior. |
| Recent file-based HBD evaluation | The shared adapter admits higher-depth decoded buffers through `RowConverter` into `RGB8_SRGB`. | Precision is reduced to eight bits. These results cannot qualify banding, low-bit errors, or near-lossless HBD steering at native precision. |
| HDR transfer and extraction | PU absolute-luminance APIs, explicit PQ/HLG display parameters, and candidate `BakeScorer::compute_hdr` exist. Fresh routing, retained-feature, extreme-input and transfer checks pass. | Finite values and path parity are implementation evidence, not human-quality calibration or codec-loop qualification. |
| HDR primaries | The HDR row reader decodes transfer into nits and feeds the PU front end; it takes RGB primaries as-is. | It does not perform the SDR path's declared-primaries conversion. Cross-primaries HDR equivalence/color correctness remains unestablished; do not infer it from SDR P3 tests. |
| Fractional sampling and HDR | `BakeScorer::compute_hdr` explicitly refuses sampling-v1 plans. | The new fractional sampling contract is SDR-only. HDR needs a separately defined and tested PU sampling contract. |
| New Rev3 models | Nine frozen candidates have six populated SDR evaluation panels. | No current Rev3 HBD/HDR human panel, complete native HDR steering qualification, or qualified HDR runtime claim is supplied by those panels. |

P3 is wide-gamut SDR here: these profiles use the sRGB transfer curve. It is
not synonymous with HDR. High bit depth is also independent of dynamic range.
Keeping these three axes separate is necessary for interpretable results.

Source owners: `zensim/src/source.rs`, `zensim/src/metric/bake.rs`,
`zensim/src/feature_v2_stream.rs`, `zensim/src/transfer.rs`, and
`zensim-bench/examples/shared/zen_decode.rs`. The latter preserves neither PNG
ICC in `DecodedRgb8` nor native HBD precision in its output contract. Original
receipts must remain immutable when a corrected extraction era is introduced.

## Fresh correctness checks

All checks below ran locally with the locked dependency graph, default features
plus `custom-profiles`, and the existing resource limiter. No corpus integration
test, human label, TEST or TERMINAL segment was opened.

| Existing test suite | Passed | Scope |
|---|---:|---|
| `icc_coverage` | 22 | Declared primaries, format equivalence, determinism, gamut behavior, larger row-parallel fixtures. |
| `gamut_real_codec` | 4 | Synthetic wide-gamut cards encoded/decoded with zenjpeg, including clipping detection. |
| `pu_entry` | 5 | Public absolute-luminance API layout/stride/padding agreement, ordering and validation. |
| `linear_srgb_equiv` | 3 | Reference-formula comparison with `linear-srgb`, including all 65,536 u16 values. This reproduces the formula; it is not an exhaustive public decoder test. |
| Library filter `hdr` | 23 | HDR dispatch/refusal, fixed-weight routing, feature retention and extreme-input behavior. |
| Library filter `transfer::tests` | 8 | Existing sRGB/PQ/HLG reference-value and display-model checks. |

Total: **65 passed, zero failed**. These are targeted engineering checks, not
a complete test suite or a perceptual-quality verdict. Commands and exact log
hashes are in the accompanying `CHECKS.json`; logs are downloadable beside this
report. No product inference code changed for this review.

## Historical HDR evidence, with chronology preserved

- June 1: the stored UPIQ report completed 380 HDR comparisons on 30 references.
  Its best reported zensim-PU SROCC was .694, below the .740 PU-SSIM bar.
  That is historical model evidence, not a current Rev3 result.
- July: subsequent memories correct apparent feature ceilings caused by
  clipping/constant inputs, and an adaptively selected HDR improvement lost
  significance after correction across seven arms (reported maxT p=.221).
  Improved synthetic ramp behavior also coexisted with worse human ranking.
- July 27: the streaming HDR implementation added explicit Linear/PQ/HLG
  inputs. Its record explicitly described primaries as taken as-is. The current
  code still warrants that limitation; transfer tests do not resolve it.
- August 28: HDR944 HF re-anchoring raised the stored validation-band median
  from 81.06 to 93.87 and fraction scoring at least 88 from .002 to .967.
  This showed historical dial reach, not newly measured shot-count accuracy.
  A later same-population comparison reversed an apparent retrain advantage:
  HF SROCC delta was −.0093, with stored 95% interval [−.0176, −.0013].
  Earlier comparisons had used different bands/targets. The author-panel
  discussion is explicitly a prediction, not an actual expert review.
- September 8: all 76 imazen-26 HDR PNG companions were bound to the existing
  `variant/png-v3` source branch; documented family roles are 38 TRAIN,
  20 validation and 18 terminal. **This inventory does not authorize reading
  the terminal files.** HDR development images already exist; obtaining EXRs
  is not the prerequisite. The selected EXR owner is `zenextras/zenexr` wrapping
  Rust `exr`; its stored parity record covers 124,609,944 f32 samples.
- September 13–14: the later prohibition on all TEST/TERMINAL access governs
  further work. None of the historical HDR populations was reopened here.
  Historical UPIQ exposure cannot supply a fresh independent qualification.

References: `upiq_pu_validation_2026-06-01.md`,
`hdr_streaming_gates_2026-07-27.md`, `hdr944_retrain_wave_2026-08-28.md`,
`hdr944_author_panel_2026-08-28.md`,
`claude_memory_chronology_2026-09-14.md`, and
`docs/DATASET_HISTORY.md`'s September 8 HDR/EXR entries. Later corrections
take precedence over earlier favorable headline tables.

## Discussion: the last two days

1. **The revised feature regimes now have real SDR evaluation.** The nine
   frozen MT913 models were scored on KADID (3,125), KonJND (404), KonFiG
   (436), imazen26 (6,953), nonphoto (6,142), and near-lossless proxy (7,717).
   The last three overlap and are metric-labelled instruments, not independent
   human studies. All nine remain NO-SHIP; canonical composite coverage is
   3/6 and its value remains null. Wide/local models have negative near-lossless
   proxy rank in several cases; narrow models compress its high-quality dial.
   Missing HDR measurements must not be inferred from their SDR tables.
2. **Feature/scale work produced mechanisms and negative results, not a ceiling.**
   September 13 records cover per-scale 944/IW planning, reduced chroma,
   fractional kernels and cost profiles. Some early screens have historical
   internal TEST segments: their tables are preserved for chronology and are
   not reusable development data. Later strict TRAIN studies are separate.
   September 14 quarter-B training improved human/codec rank
   (.83150→.83582 / .92442→.93274), but increased codec scatter outliers
   (76→77), worsened other tails and lowered peer-map association. Neither
   quarter-B nor quarter-XB passes its advancement screen.
3. **Spatial correctness is not native quality benefit.** Derivative, finite
   response and map-parity work improved implementation evidence. Native
   constrained/local/TV experiments still expose peer-quality conflicts.
   Direct triangle/Mitchell/RobidouxSharp filters reduced focal coarse-B phase
   span by 68–71%, while mean span rose 15–24%; full-resolution period-control
   failures remained. No native rate–distortion win follows from those checks.
4. **A real Rev3 corruption pilot now runs through Rust.** Calibration detects
   154/154 severe proxies with 0/792 honest nonidentity activations; TRAIN
   development detects 267/267 with 2/1,666 honest activations. All 272 native
   JXL attempts preserve acceptance and maps. But 213 P3 fitting pairs were
   explicitly excluded, mobile corruption coverage is missing, and an AVIF
   q85 cast remains unresolved. EVAL advancement is blocked. The JPEG heatmap
   false alarm drops 55.67→0; the gallery retains both false alarms.
5. **Runtime claims remain bounded.** Several useful loops finish in minutes,
   but latest quarter-chroma timing suites failed quiet-run acceptance. Neither
   coefficient count nor noisy p95 establishes a shipping performance profile.

The accompanying source-document index covers the September 13–14 benchmark
records without inventing new evaluation rows. The board's new discussion set
selects the nine measured MT913 candidates and matched baselines; later TRAIN
experiments retain their separate linked reports and galleries.

## Next work in order

1. Carry native color metadata and precision through the existing decoder,
   extraction and public-score owners with an explicit versioned contract.
   Check actual encoded ICC/CICP too; fixing only reference interpretation
   does not establish what the original encoder did.
2. Define the intended common HDR comparison primaries/display contract before
   correcting or restricting the HDR route. Verify cross-primaries equivalence,
   PQ/HLG/linear equivalence, native-depth low-bit errors and gamut-clipping
   detection. Reuse existing conversion owners; do not add a parallel CMM.
3. Restore admitted P3/mobile TRAIN coverage, resolve the q85 provenance and
   review recoverable versus catastrophic labels before another head fit.
4. Freeze eligible candidates and separately gate native-precision SDR HBD,
   wide-gamut and HDR quality, attainable targeting, native spatial outcomes,
   and latency/memory. Never reopen prohibited data to fill a missing panel.

Until those steps pass, advertise only the measured SDR experimental scope.
