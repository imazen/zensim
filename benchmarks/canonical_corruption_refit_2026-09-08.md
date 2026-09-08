# Canonical corruption refit preparation — September 8, 2026

The canonical single-fit path is implemented and its data is prepared, but
**no canonical image-data candidate has been fit or qualified**. The remaining
content-admission gap is 30 UPIQ HDR reference EXRs. Existing local EXR owners
delegate to `image::open`; no native EXR decoder was found. A narrowly scoped
exception for reference fingerprinting has been requested, not assumed.

## Source admission

The existing `check_holdout_overlap` now has a strict native PNG mode using
zenpng, zenpixels-convert and zenresize. It shares the historical dHash bit
ordering and preserves the legacy path. Its decode/resample era is explicitly
different; no legacy hash equivalence is claimed. Exact nonzero source counts,
complete decoding, fresh outputs and file hashes are mandatory. The sidecar
retains every input fingerprint and every pair at distance <=16. Strict flags
remain <=10; an audit does not automatically approve or exclude content.

All 12 canonical training origins were checked against these **182 SDR
reference entries**: CID22 49, AIC-3 10, AIC-4 5, SDR25 5, CSIQ 30, LIVE 29
and UPIQ SDR 54. Repeated references across corpora remain named coverage,
not independent observations. No human scores were used by the audit.

There are **zero strict flags**, with minimum distance 14. All four looser
matches were reviewed: origin 6068 is a two-column paper, while its matches
are a portrait and stacked gift boxes; origin 6610 is an illustrated old book,
while its LIVE/UPIQ match is a red barn photograph. No source was excluded.
This remains a crop-blind hash screen, not proof against every transformation.

LIVE's BMP originals revealed a missing format arm in the existing
`verify_bitstream_decode --decode-list` tool. BMP now routes to the already
shared native decoder. All 29 original BMP/native PNG pairs have exact RGB
identity, verified independently through the extractor audit. The final
formatted/linted producer reproduces all 29 PNG files byte-for-byte. The
failed prior run is retained; it is not counted as decoded coverage.

## Prepared data and fitting contract

The explicit `train_corruption_head.py --canonical-manifest` mode reads the
four hash-bound canonical tables and the complete Rust audit, retaining source
and pixel identities. It has no implicit broad-honest/negrich input list and
no random splitting of prefixed source names. All feature values are checked;
the 372-column f32 values are promoted to f64 for fitting.

| Role | Origins | Unique rows | Corruptions | Honest negatives |
|---|---:|---:|---:|---:|
| Fit | 8 | 5,504 | 5,179 | 325 |
| Probability calibration | 4 | 2,709 | 2,546 | 163 |
| Evaluation | 8 | 5,679 | 5,353 | 326 |

The 15,060 raw rows yield 13,892 unique source/pixel pairs; 1,168 duplicate
attempts are retained only in the raw packet. The fit/calibration/evaluation
origin and family sets do not overlap. Complete 372-feature duplicate counts
are 4/2/5, with **zero conflicting labels**; repeated feature values across
different source pairs are not automatically discarded.

The old C10 surrogate `(f0,label)` repeats 342/221/201 times across the full
roles despite different complete feature vectors. The existing Parquet
validator now accepts an explicit `duplicate_key_columns` contract, checking
**every row for zero duplicates or missing keys**. These views use origin,
geometry and both decoded-pixel hashes. The old sampled diagnostic is retained
and the default contract is unchanged. Raw-table C10 failures remain recorded.
All three prepared Parquets pass their explicit full-key contract.

Before fitting, the trainer requires a complete hash-bound content-admission
receipt covering all protected reference groups, including the 30 EXRs. Its
`--prepare-only` mode never fits. HGB settings, source weighting, seeds,
calibration sources and development bars are fixed in the
[registration](../docs/CANONICAL_CORRUPTION_REFIT_2026-09-08.md).

Actual fitting will export the same single estimator it calibrates, verify
raw-tree/probability/fire-set parity with `corrhead_parity`, then evaluate all
pixel pairs through the complete Rust `BakeScorer` audit. Reports explicitly
identify the canonical fit. `SCREEN.json` records the preregistered head bars;
neither that screen nor a completed fit qualifies the whole product. The
legacy CV-ensemble path and both legacy exporters retain their original ASTs
apart from the new-mode dispatch and documentation correction.

## Verification and artifacts

Fourteen content-cluster tests, CI-exact root Clippy, exact BMP-tool Clippy,
formatting and script lint pass. Native controls accept exact/resized matches
(distance 0/0), distinguish an unrelated image (33), and reject nine damaged
or incomplete inputs. Eight trainer controls refuse before fitting. Six
Parquet controls distinguish valid declared keys from duplicate/missing keys
and preserve the old default C10 refusal.

A separate **800-row invented numeric fixture** exercises the actual weighted
single-fit helper and ZCTH exporter: 100 trees / 3,916 nodes, raw error 0 ulp,
probability error 0, and zero fire-set disagreements in Rust. This tests the
implementation; it is not a trained image-quality candidate or evidence of
competitive perceptual accuracy.

Packet: `/mnt/v/output/zensim/canonical-corruption-refit-2026-09-08/`.
Share directory: `~/work/zensim-validation-2026-09-08/canonical-corruption-refit/`.
`HOLDOUT_INPUTS_COMPLETE_SDR.json`, `native-sdr-audit.hashes.json`,
`CLOSE_PAIR_REVIEW.json`, `PREPARE_MANIFEST.json`, `prepared-verified/`, the
control recipes, sources, binaries and logs preserve the evidence. The
preparation manifest deliberately has no usable admission hash yet.

| Artifact | SHA-256 |
|---|---|
| Native audit binary | `a7fdb7df0768786990c83658e3581f22eb86e7633fb8a2a082c37537f4ddb482` |
| Final BMP decode-list binary | `ce36e3cc5d6350647fdbbaec7dd82fb102f1e253619bdd0bbb302eca1db82b62` |
| Fit Parquet | `3c8c679f781867c1b65950adae20f4fbee5a5d05863082992afc997a1f8e2e32` |
| Calibration Parquet | `83712144f9a58d9d73a58d7e1cd71a2587254cef7c081640f7722fae6b9501b9` |
| Evaluation Parquet | `f53e4c22e5cd55f4e567931f6fd6bd01c72371cb3cf14768405f9d049fb9a696` |

Next: finish EXR reference admission, then execute the registered three-seed
canonical fit and its Rust evaluation. If that permission remains pending,
continue independent native spatial-allocation work; do not repeatedly rebuild
the same admitted SDR packet or silently waive HDR reference coverage.
