# September 13–14: feature overhaul, color, HBD and HDR — current assessment

Evidence reviewed through zensim `c1819e2936553ad2154e5c2bb85f6de7f2c4bbf0`
and zenjpeg `8f703a6eeaad080045aa4c1b3713a25df2e15529` on September 14.
**Native SDR input correctness has progressed; none of the nine revised models
is qualified to ship. Native HBD and HDR product evaluation remains incomplete.**
This is an assessment and gauntlet publication, with no new training or EVAL.

## Current status

| Area | Evidence established | Limits and remaining work |
|---|---|---|
| Color profiles | The opt-in `sdr-native-clip-v1` extractor uses native decoded metadata, preserves known SDR codes/primaries and applies the existing full CMS for arbitrary ICC. Public Rust scoring owns declared-primary conversion, alpha and display clipping. JPEG XYB decoded-output metadata was fixed at the codec owner. | Original P3 encoder input conversion is unproven. The AVIF q85 color cast is unresolved. Default sRGB display clipping can hide gamut loss; this is not a wide-gamut-loss qualification. Legacy RGB8 remains the default and earlier model evidence retains that input era. |
| High bit depth | Known sRGB-transfer u16 codes survive into the public scorer; linear samples use f32. Synthetic low-bit differences that RGB8 erases affect public features. The admitted file inventory includes 21 ten-bit AVIFs decoded as Rgb16. | Exact code retention is not proof of calibrated low-bit sensitivity. ICC conversion has bounded numerical error, not guaranteed preservation of every low bit. There is no current native-depth banding, near-lossless target-error, perceptual or spatial RD qualification for these models. |
| HDR | Existing public absolute-luminance paths and PQ/HLG transfer checks have recorded passing engineering tests. HDR PNG companions already exist in the imazen-26 inventory. | HDR extraction takes primaries as supplied, without a common-primaries conversion. Fractional sampling-v1 refuses HDR; the new native SDR adapter also refuses HDR deliberately. No new Rev3 HDR perceptual, targeting or spatial qualification exists. |
| Model quality | Nine frozen MT913 candidates each have six populated SDR EVAL panels through the public Rust path, with matched B/D and SSIMULACRA2 controls. | All remain NO-SHIP. Composite coverage is 3/6; the unavailable composite remains null. Input correctness and TRAIN audits do not repair or replace those EVAL results. |
| Spatial correctness | On 214 admitted TRAIN pairs, native372 base/head scores match pixel/cache/f32 paths; accepted-head maps match the perceptual branch. A separate native944 ensemble has exact scalar/cache/f32/spatial score parity. | Ensemble member-by-member canonical consumed-feature comparison is still missing. The no-head audit's zero feature-delta field does not establish that comparison. Finite queries and score parity do not establish native rate–distortion benefit. |

P3 is wide-gamut SDR here. Ten-bit sRGB is HBD SDR. Neither is evidence of HDR.

## What changed after the earlier status report

The decoder-only report was accurate at `9620e39e`: native buffers were retained
but extraction still requested RGB8. The later `c1819e29` implementation adds
an explicit native SDR extraction and scoring path. Read the earlier report as
historical evidence, not the current implementation status.

The native serving audit uses **214 TRAIN-fit pairs / 215 files**, including
213 P3-derived pairs and the unresolved q85 control. It is not EVAL and its
diagnostic row-ID column must not be used as a training quality label.

- Native372 base plus integrity head: **214/214** exact pixel/cached/stored-f32
  composed scores. Companion-declared canonical features have zero difference.
  Of these pairs, 213 are accepted, with **30,136** unchanged map queries;
  q85 remains the sole integrity rejection.
- Native944 two-member ensemble: **214/214** exact scalar/cached/stored-f32/
  spatial scores, matching scalar/spatial returned features and **626,249**
  finite refinement queries. An independent canonical comparison covering every
  member's consumed inputs remains a qualification prerequisite.
- All **33 identities** retain score 100. Legacy RGB8 CSV and audit outputs
  remain byte-identical on the 214 pairs.
- **23 synthetic tests**, one JPEG-owner regression and five CLI refusal cases
  pass in the saved final receipts. These are engineering checks, not human
  quality evaluations. The earlier 65 color/HDR checks have their own revisions
  and scopes; do not combine their counts into a model qualification score.

Native interpretation changes 181 scalar outputs by **−26.09 to +7.27 points**
relative to legacy. These are material input-era effects, not measured quality
gains: the weights were fitted in the earlier era. Generic transfer/gamut
conversion attempts failed numerical checks and were rejected without widening
tolerances. Known codes now stay native; arbitrary ICC retains the independent
P3 colorimetry bound of 0.0005 per linear component through the existing CMS.

None of the 180 P3-derived reconstructions retains the source ICC, but that
alone does not identify an encoder error: a correct conversion to sRGB could
remove it. Original encoding provenance must settle that question before these
cases become clean corruption negatives.

## Discussion set: the last two days

The latest EVAL discussion selects **nine MT913 feature-regime candidates plus
matched B, matched D and SSIMULACRA2**. It covers y40/y60 H32/H128, local120,
selected619, full944 H128/H256 and linear60. Every candidate has KADID (3,125),
KonJND (404), KonFiG (436), imazen26 (6,953), nonphoto (6,142) and near-lossless
proxy (7,717) rows. The last three overlap and use metric labels; they are not
independent human studies. High-quality compression, negative proxy rank in
several wide/local models, tails and incomplete gates still prevent shipment.
Matched B/D also have all six rank panels. The SSIMULACRA2 peer row has rank
results for the three human panels only; absent peer panels are not filled in.

The separately linked TRAIN discussions cover the actual subsequent science:

- **Feature/scale and runtime:** quarter-resolution B improves human/codec
  rank, but codec scatter outliers rise from 76 to 77 and peer-map association
  worsens. Quarter-B and quarter-XB do not advance. Quiet timing failed, so
  this does not establish a latency advantage or a feature-set ceiling.
- **Spatial response:** constrained/local/TV studies show that improving a
  model's own map consistency can worsen independent quality ordering.
  Triangle, Mitchell and RobidouxSharp reduce the focal coarse-B phase span
  68–71%, but mean span rises 15–24% and period-control failures persist.
  No native RD win follows from those measurements.
- **Corruption head:** the Rust Rev3 thresholded pilot detects 154/154 severe
  calibration proxies and 267/267 development proxies; honest nonidentity
  activations are 0/792 and 2/1,666. All 272 recorded native JXL attempts retain
  accepted steering maps. False alarms, mobile/P3 coverage and unresolved labels
  block EVAL. A corruption activation remains an error condition, not ordinary
  spatial quality control.
- **Color/depth:** native decoding, explicit color interpretation and native
  public scoring are now implemented and audited within the stated scope.
  Matching training-era admission/refits and native-depth EVAL are still pending.

Use the **full gauntlet** comparison. Its older fair-only filter requires CID22
and excludes these 12 members; that is missing historical-filter evidence,
not missing six-panel SDR evaluation or proof of contamination. Do not access
prohibited holdouts to satisfy that filter. The report provides an exact
12-member comparison link and links every recent TRAIN study separately.

## HDR chronology and next dependencies

June's historical UPIQ result was zensim-PU SROCC 0.694 versus PU-SSIM 0.740.
July corrections exposed clipping/constant-input artifacts and adaptive
selection. August score re-anchoring improved reach, but a later matched-panel
comparison reversed the apparent rank advantage. Those later corrections
govern; none qualifies the current Rev3 models. HDR PNGs are already available;
obtaining EXRs is not the blocker. The EXR owner remains zenextras/zenexr using
the Rust exr crate.

1. Close complete ensemble canonical-feature auditing; explicitly admit the
   new native input/audit era into downstream TRAIN manifests.
2. Resolve original P3 conversion and q85, restore correctly interpreted mobile
   corruption coverage, then refit/calibrate exclusively on matching TRAIN data.
3. Establish HDR comparison primaries, luminance/display and gamut semantics;
   verify PQ/HLG/linear equivalence and native-depth error behavior.
4. Freeze complete Rust models and gate only on admitted EVAL: composite
   coverage, rank, scatter envelopes/clumping/saturation, corruption false
   alarms, image-specific attainable targets and 1/2/3-shot error tails,
   native spatial RD, latency and memory. Never access TEST/TERMINAL segments.

No model promotion, new EVAL data or performance claim accompanies this update.
