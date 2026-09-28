# September 13–14 discussion: color, high bit depth and HDR

Updated against `9620e39ef2ab9ba4e22b92ea744453b72b7909fe` on September 14.
**Native decoding has improved; end-to-end color/HBD/HDR model qualification
is still incomplete. None of the nine new Rev3 models is qualified to ship.**
This update incorporates the decoder work completed after the earlier status
review. It reports existing measurements; it adds no training or evaluation.

## Current correctness and evaluation status

| Area | Established | Still missing |
|---|---|---|
| Color profiles | The native decoder now retains source ICC/CICP and decoded-pixel metadata separately. Public SDR sources support declared sRGB, Display P3 and BT.2020 primaries. | The feature extractor still requests legacy RGB8. Retaining ICC does not apply it. The affected P3 rows do not yet have a color-correct extraction and public-score audit. |
| High bit depth (HBD) | Native u16 samples survive decoding; synthetic PNG tests retain actual low-bit differences. The inspected AVIF set includes 21 ten-bit files retained as Rgb16. Public u16 and linear-f32 scoring inputs exist. | The current file-based feature path still reduces samples to RGB8. Native-depth banding, low-bit sensitivity, near-lossless target accuracy and spatial steering have not been qualified for the new models. |
| HDR | Public absolute-luminance scoring, PQ/HLG transfer handling and the candidate HDR route exist. The recorded numerical/routing checks pass. | HDR extraction takes primaries as-is; cross-primaries equivalence is unestablished. Fractional sampling-v1 explicitly refuses HDR. There is no current Rev3 HDR perceptual or native steering qualification. |
| New model evaluation | All nine frozen MT913 candidates have six populated SDR panels, compared with matched B/D and SSIMULACRA2. | All remain NO-SHIP. Composite coverage is 3/6 and the composite value is null. SDR results cannot fill missing HBD/HDR evidence. |

P3 here is wide-gamut SDR, not HDR. Bit depth, gamut and dynamic range are
separate dimensions; a ten-bit sRGB AVIF is not evidence of HDR support.

## What the latest decoder work actually establishes

The native inspection covers **215 files from 214 admitted TRAIN-fit pairs**.
All 215 reproduce their old RGB8 pixel hashes exactly, while the new result
also retains native buffers and source metadata. That proves compatibility
with the earlier decoding output; it does not prove that output had the right
color interpretation.

The affected P3 population is 33 references and 213 pairs, including identities.
None of the 180 stored reconstructions retains the source's 520-byte ICC.
That is **not proof of an encoder bug**: encoding might correctly have converted
to sRGB. Original encoder input/conversion provenance is still needed.
The current common-profile identifier does not recognize that ICC, so guessing
its transform from the profile description would be incorrect.

Six JPEG outputs retain an XYB source ICC while their decoded descriptor has
unknown primaries/transfer. A source profile must not be blindly reapplied to
RGB already transformed by a codec. JXL reports authoritative sRGB CICP. These
cases require explicit source-versus-decoded interpretation, not one blanket
ICC rule. The AVIF q85 color cast remains unresolved; retained BT.709/sRGB,
full-range metadata does not locate its cause.

The latest decoder change passed 17 synthetic codec tests, one inspection-list
unit test and five CLI refusal checks. The earlier status review recorded 65
passing color/HDR engineering checks. Their receipts have different scopes and
revisions; none is a human-quality evaluation. This update rechecks the saved
evidence and current source rather than presenting those checks as newly run.

Default SDR gamut clipping measures the sRGB display result and can hide
destructive gamut loss. `GamutMapping::Preserve` can expose that loss, but
shipped models were not trained on its out-of-gamut distribution. It is not a
qualified replacement score or an HDR solution.

## Discussion of the last two days

1. **The feature rewrite now has actual SDR evaluation.** The nine frozen
   candidates cover y40/y60 H32/H128, local120, selected619, full944 H128/H256
   and linear60. Each has KADID (3,125 pairs), KonJND (404), KonFiG (436),
   imazen26 (6,953), nonphoto (6,142) and near-lossless proxy (7,717) results.
   The last three overlap and use metric labels; they are not three independent
   human studies. High-quality score compression, negative proxy rank in
   several wider/local models, tails and incomplete gates prevent shipment.
   The discussion dropdown selects these nine plus matched B/D and the peer.

2. **Feature/scale experiments have not established a useful ceiling.** The
   recent work covers per-scale 944/IW availability, reduced chroma, fractional
   filters and runtime profiles. In the later strict TRAIN quarter-chroma
   study, adding quarter-resolution B improves human/codec rank, but codec
   scatter outliers increase from 76 to 77 and peer-map association worsens.
   Neither quarter-B nor quarter-XB advances. Quiet timing acceptance failed,
   so those runs do not establish a shipping latency advantage.

3. **Spatial implementation evidence is ahead of spatial product evidence.**
   Constrained/local/TV studies expose conflicts between improving a model's
   own spatial response and improving peer quality. Triangle, Mitchell and
   RobidouxSharp reduce the focal coarse-B phase span by 68–71%, but mean
   span rises 15–24% and full-resolution period-control failures remain.
   These results do not establish a native rate–distortion win.

4. **The thresholded Rev3 corruption head is a real Rust pilot, with blockers.**
   TRAIN calibration detects 154/154 severe proxies with 0/792 honest
   nonidentity activations; TRAIN development detects 267/267 with 2/1,666
   honest activations. All 272 native JXL attempts preserve maps. The false
   alarms include a JPEG heatmap dropping from 55.67 to 0. P3/mobile coverage,
   the q85 case and recoverable-versus-catastrophic labels remain unresolved.
   Those are TRAIN results; the pilot has not advanced to EVAL.

5. **Color/precision admission is the newest completed step.** It gives us
   the native samples and metadata needed to correct extraction. It does not
   retrospectively repair earlier features, training or model evaluations.

The linked TRAIN reports retain complete comparisons and failure galleries.
They stay distinct from the measured EVAL candidates in the gauntlet.

**Use the full gauntlet for this comparison.** The older fair-only filter
requires a held-out CID22 result and excludes all 12 comparison members from
its file. Its `g_split`/LEGACY badge here reflects missing required evidence;
it is not proof that the six admitted SDR panels are absent or contaminated.
Do not access prohibited CID22 data merely to satisfy that historical filter.
The report's comparison link opens the full board with the exact 12 members.

## Chronology matters for HDR

The June UPIQ result was historical zensim-PU SROCC .694 versus PU-SSIM .740,
not a new-model result. July corrections identified clipping/constant-input
artifacts and removed significance from an adaptively selected improvement.
August re-anchoring improved score reach, but a later matched-population
comparison reversed the apparent ranking advantage. These later corrections
govern; favorable earlier summaries cannot qualify the current models.

September's inventory already identifies HDR PNG companions in imazen-26;
obtaining EXRs is not the prerequisite. The chosen EXR owner is zenexr in
zenextras, wrapping Rust exr. Historical inventories and reports do not authorize
opening TEST/TERMINAL data. No such data is accessed for this update.

## Next work, in dependency order

1. Connect the existing native decoder and CMS owners to extraction and the
   public scorer under an explicit color/precision contract. Audit scalar,
   cached-feature and spatial parity; refuse unresolved interpretation. Preserve
   the old extraction era so corrected and historical tables cannot be mixed.
2. Resolve original P3 encoder conversion and the q85 case; restore admitted
   mobile TRAIN coverage and refit only on the matching extraction era.
3. Define the HDR comparison primaries and display parameters, then verify
   PQ/HLG/linear equivalence, cross-primaries behavior and native-depth errors.
4. Freeze candidates trained on TRAIN only, then gate on admitted EVAL only:
   perceptual panels, outlier envelopes, clumping/saturation, per-image attainable
   target ranges, 1/2/3-shot errors, native spatial quality and measured runtime.
   Missing evidence stays incomplete; TEST/TERMINAL remains untouched.

The immediate product blocker is a trustworthy end-to-end input contract.
More model fitting on the old RGB8 path cannot resolve the color/HBD/HDR gaps.
