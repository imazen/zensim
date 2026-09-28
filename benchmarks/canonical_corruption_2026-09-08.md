# Canonical corruption input packet — September 8, 2026

**Generation/extraction complete; model training admission incomplete.** No new
model has been fitted or qualified. The September 6 nonlinear HGB result and
Rust ZCTH parity supersede the earlier linear-head separability hypothesis.
The measured historical HGB detection at matched 0.5% ladder FP was 98.52%;
that historical split and its foreign anchors do not qualify a canonical model.
See [registration](../docs/CANONICAL_CORRUPTION_2026-09-08.md),
[theory study](corruption_head_theories_2026-09-06.md) and
[Rust serving record](corruption_head_serving_2026-09-06.md).

## What changed

The existing native `m3_fixture_gen` has a private `corruption` mode. It uses
the unchanged pure generator from codec-corpus commit
`8e10d4d765667c1c49d74413878fc4bfb46dcf8d`, without the historical image driver.
The previously checked-out `3e7a8a22` branch omitted later real-bug families.
Native zenpng handles source/output PNGs; zenjpeg
`3d4ad0d77ecb55e78e67828538abb87068180d96` encodes and independently decodes two
YCbCr 4:2:0 q10/q20 anchors per source. Sources retain their original dimensions.
Every saved PNG roundtrips to exactly the packed RGB8 pixels that were written.

The existing Python corpus owner has an explicit canonical JSON mode. It checks
both canonical split owners, source bytes, family identities, tool/source
hashes, all row keys and every feature's finiteness. Missing or duplicated
rows stop the run. Partial Parquets cannot acquire a corpus COMPLETE marker.
The separate historical TSV mode retains its old pixel generation behavior.
No library API, new feature flag, classifier or generator implementation added.

## Data and provenance

The frozen targeting sources provide 12 training and 8 validation origins,
one distinct canonical family each; their families and origins do not overlap.
All source bytes are the existing longest-side-256 variants. Family manifest
SHA is `9d07a0f63ef5fa167c5333535010f44b4ab9a087f04e560521b6d1aa1961820c`;
canonical imazen-26 commit is `187fbf338ce08e8e6654db7f04ddae58d5263da2`.

| Retained catalog | Train | Validation |
|---|---:|---:|
| Origins / families | 12 | 8 |
| Rows: 713 attempts + 2 anchors per origin | 8,580 | 5,720 |
| Changed-pixel positive labels | 8,030 | 5,552 |
| Inert attempts, labeled noncorrupt | 526 | 152 |
| Unique source/pixel pairs | 7,757 | 5,375 |
| Duplicate pairs with exact feature/label agreement | 823 | 345 |
| Additional honest native JXL/AVIF rows | 456 | 304 |

Raw IDs are explicitly 0..371, formula revision 1, libm root form. The pinned
extractor SHA is
`d7076578ea9519e400ba89527b6ba6b80a56e517f8560334c2d6267965d49677`.
The packet preserves Cargo metadata/lockfile, build logs, 686 local source-file
hashes, exact producer source archive, codec revisions and binary bytes. The
optional generator dependencies do not change feature arithmetic.

The honest supplement uses only scalar bound records: 21 JXL distances and 17
AVIF CQ values per source. Every emitted bitstream SHA/length is checked, then
the current pinned native extractor independently decodes it. Prior model
scores are provenance only. `pairs-tsv` preserves numeric row/origin keys;
the initial positional-mode extraction dropped the extra key columns and is
retained as an unaccepted instrument attempt. The keyed outputs have complete
identity checks and are the only accepted supplement. No new encodes for this
supplement; 760 independent bitstream decodes in the accepted extraction.

All 40 dataset JPEG encodes/anchor decodes succeeded. This is a small input
packet, not evidence of coverage across all image sizes or HDR/color/alpha
contracts. Unsupported source metadata, bit depths, transparency and animation
are refused by this private normalized-sRGB generator.

## Validation and limits

- Root CI-exact Clippy and 605-script lint pass; fixture example Clippy passes.
  Four constant-chunk lint errors (two pre-existing) were mechanically repaired
  after the corpus completed. Final code reproduces all 715 engineering pixel
  hashes and every feature CSV byte. Earlier and current JPEG q10/q20/q80 and
  resize controls retain exact bytes; anchor q10/q20 match that native encoder.
- 18 corpus controls: one valid packet and 17 injected failures; 12 CLI
  rejections. Failed runs never emit a corpus completion marker; an existing
  output directory's sentinel remains intact. The alpha fixture is refused at
  the earlier metadata check, so it does not independently exercise the alpha
  pixel branch.
- Odd 205×256 geometry and a 410×512 engineering catalog complete, including
  all 713 attempts and two anchors. No larger geometry was used to fit a model.
- Full 372-column finiteness, exact keyed coverage, all retained PNG hashes,
  split disjointness and 1,168 duplicate pixel/feature/label checks pass.
- **C10 remains failed on the raw retained catalog:** the existing sampled
  `(f0,target)` duplicate screen reports 2.54% train and 21.29% validation.
  No threshold was relaxed. These are audit tables; fitting must deduplicate
  by source/pixel identity and preserve the removed-case accounting.

Still required before fitting: training-view deduplication and class weighting,
the existing T0 reference-content audit, current public-surface consumed-feature
parity, and an exact fit/calibration/evaluation partition registration. Suggested
bounded development split: two of the three training origins per class for
fitting and the third for probability calibration; all eight validation origins
remain evaluation-only. Register actual identities and estimator settings first.
Do not use the training script's historical broad-negative defaults: they include
held-out content and namespace-prefixed random source splits.

The existing HGB and exporter are the next owners. Its full composition must be
evaluated through `BakeScorer`. Its gate is discontinuous and currently maps a
flagged positive dial score to zero; current native map adapters explicitly
reject corruption companions. Neither this dataset nor historical classifier
accuracy resolves that spatial/negative-tail product contract.

## Artifacts

`/mnt/v/output/zensim/canonical-corruption-2026-09-08/` contains raw PNGs, native
anchor bitstreams, Parquets, source/producer manifests, exact binaries and
verification recipes. `RESULT_COMPLETE.json` describes only this input packet
and keeps `model_qualified: false`. Windows-readable copy:
`~/work/zensim-validation-2026-09-08/canonical-corruption/`.

| Accepted table | SHA-256 |
|---|---|
| `train.parquet` | `5f615821bb7c3054718e73362d90d156eb867e0a39b3f68b40f62b57cbc7aec5` |
| `validate.parquet` | `afba5adfbdbb1418b7d78b50ae1129339031e6d5f4446bffd14273d709a6914a` |
| `honest-train-keyed.parquet` | `cf2e86afae8a20e22cc018439cd5fbecc116908fc2cbe41a5ea56c80ec44982e` |
| `honest-validate-keyed.parquet` | `9eb396219408b9de1810c5ebf317fa7151f57204aa699c7f3f1a0ace387400ca` |
