# Native SDR input contract — September 14, 2026

Later September14 amendment: the existing audit now supports native linear-float
fast-ssim2 through shared scorer conversion. All 214 admitted pairs replay with
unchanged public features/scores/spatial measurements;33 identities score 100.
See [implementation and evidence](rev3_native_optimization_2026-09-14.md).
Historical native-peer refusal descriptions below refer to the earlier build.

Registered before implementation and new corpus reads. Concrete caller:
`extract_features_372col` and its existing complete-candidate audit need to
score native-depth, color-interpreted samples through the public Rust surface.
The existing RGB8 contract remains explicit and unchanged. Add opt-in
`--input-contract sdr-native-clip-v1`, requiring fresh audited explicit pairs
and zero tolerated failures. No named model is advanced by this change.

The native contract compares stored raster coordinates on an sRGB display:
decode native samples; resolve decoded color semantics; convert to linear
sRGB f32 with the existing zenpixels-convert/CMS owners; use the public
StridedBytes input and its existing display clipping/alpha behavior. Preserve
source metadata and native/presented pixel hashes in the audit. Source ICC and
codec-converted RGB are distinct: JPEG XYB output must be labelled at the
decoder owner, without applying the source XYB ICC a second time. Fix this
metadata behavior without changing public signatures or decoded samples.

Untagged PNG uses an explicitly recorded sRGB assumption. Unsupported gamma,
conflicting color signals, unknown decoded interpretation, and HDR inputs
must be refused, never silently retagged or quantized. This SDR contract does
not claim gamut-loss or HDR qualification. The legacy RGB8 peer audit must
not silently quantize native inputs to run a different comparison.

Verify with synthetic imazen encodes: actual u16 low bits, ICC versus known
primaries equivalence, source-versus-decoded JPEG metadata, unchanged legacy
pixels, complete canonical-feature versus candidate pixel/cached scoring,
spatial parity and explicit refusals. Reuse the existing audit rather than
adding another scorer. Any subsequent real-data run must use a separately
hash-bound TRAIN admission; historical features and pixel hashes stay intact.

This is an input/extraction era, not a feature-formula revision or a trained
model qualification. Refit and freeze eligible models on matching TRAIN
features before native color/HBD EVAL, target-loop or spatial claims.

## Implemented contract and numerical correction

The initial all-f32 conversion proposal above failed its numerical checks.
The generic RowConverter transfer/gamut route differed from the independent
transfer reference and produced a P3 discrepancy of .00454 in linear RGB on
the synthetic interior-color card. Using the full CMS for named sRGB instead
still introduced lookup-table error on low-bit samples. Failed checks are
retained; no tolerances were widened to admit those routes.

The implemented contract therefore keeps known sRGB-transfer samples as exact
RGBA u16 codes, and known linear samples as RGBA f32, with declared sRGB/P3/
BT.2020 primaries. The existing public ImageSource path owns their transfer,
primaries conversion and display clipping. Mechanical packing uses the same
neutral transfer/primaries on both sides and applies no guessed color curve.
Arbitrary ICC uses the existing MoxCms f32 transform after code normalization;
the result enters the public linear-sRGB surface. Its independent f64 P3
colorimetry check retains the unchanged .0005 linear-component error bound;
that bounded ICC check is separate from exact native u16-code retention.

The codec-owned XYB fix describes decoded u8 as sRGB and selected float output
as linear RGB while retaining the source ICC. No second XYB transform, decoder,
ICC parser, scoring kernel or public library API was introduced. Unsupported
transfer/primaries, explicit HDR, conflicting PNG authorities and unresolved
gamma/chromaticities fail. Untagged PNG explicitly assumes sRGB. Raster
orientation is recorded; this contract does not auto-orient source images.

The native audit uses schema `canonical-feature-audit-v2` and records the
input contract, native/source hashes, color interpretation, presented format,
endianness and complete candidate composition. Historical RGB8 audit-v1 stays
unchanged. Cached identity additionally requires matching presented format,
primaries, alpha and gamut policy; equal RGB codes labelled P3 and sRGB do not
constitute identical images. A regression checks this against public scoring.
The RGB8-only SSIMULACRA2 audit is refused with native input.
New wide extraction manifests also record the separate native input era.

## Completed TRAIN input and serving audit

The existing native-color admission authorizes 214 TRAIN-fit pairs / 215 files:
213 P3 pairs, including 33 identities, plus the unresolved AVIF q85 control.
All family/role authorities and file hashes were checked before scoring. The
TSV's `human_score` field carries the original row ID solely as an audit join
key; these files are **not a training target table**. No new fit or EVAL run
was performed. Original images, labels and earlier evidence remain unchanged.

| Check on final executable | Result |
|---|---|
| Native full372 extraction, existing Rev3 base plus integrity head | 214/214 complete; pixel/cached/stored-f32 composed scores exactly equal; maximum consumed-feature difference zero. |
| Prepared integrity path | 213 accepted, with 30,136 map queries equal to the perceptual branch; q85 row 5171 remains rejected. |
| Native full944 extraction and existing y70/y80 two-member ensemble | 214/214 complete; pixel/cached/stored-f32/spatial scores exactly equal; 626,249 finite refinement queries; no unsupported density IDs for this composition. |
| Native identities | All 33 retain score 100. |
| Historical compatibility | Old and final executables produce byte-identical legacy372 CSVs and complete JSONL audits for all 214 pairs. |
| Synthetic tests | 23 pass: native low bits, public-feature sensitivity to differences lost by RGB8, independent ICC colorimetry, translucent alpha, interpretation-aware identity, format coverage and refusals. One JPEG-owner regression also passes. |
| CLI preflight | Five refusal cases precede source access and emit no CSV/audit. Incompatible head/canonical width is now rejected before reading pairs. |

The first attempted full944-plus-head audit correctly rejected all rows because
this head's caller contract is 372 columns. Neither that native attempt nor its
legacy control emitted a partial CSV/audit. The completed runs use canonical372
for the head and full944 for the separate ensemble; no feature truncation or
head retagging was used. The final executable exactly reproduces all successful
initial audits after the preflight improvement.

Native interpretation changes 181 scalar outputs, with native-minus-legacy
differences ranging from −26.09 to +7.27 points. This is evidence that input
semantics matter, **not a measured quality improvement**: these frozen weights
were trained in the earlier input era. The q85 activation is still unresolved,
not newly classified as a confirmed encoder bug. Native extraction completion
does not establish the original P3 encoder input conversion or restore those
pairs as clean negatives automatically.

The compact results JSON and served replay bundle bind manifests, source,
executables, commands, feature/audit files and verification logs. No quiet
latency benchmark, native rate–distortion win, HDR qualification or model
advancement is claimed.

## Next dependency

Trace the original P3 encoder input/conversion and q85 failure through their
existing owners. Generate and label missing mobile corruption coverage with
the explicit native color contract, and build matching TRAIN features before
refitting. The new audit schema must be explicitly admitted by downstream
training manifests; do not let a historical RGB8 receipt stand in for it.
HDR still needs its own primaries/display contract and qualification.
