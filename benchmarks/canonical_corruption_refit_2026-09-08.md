# Canonical corruption refit — September 8, 2026

**Later user-directed D-regime experiment:** a newly trained f0..227 head
serves with D's existing extraction, detects all tested RGB swaps, but fails
honest-output protection. [Newer result and stopped cost sweep](canonical_corruption_d228_2026-09-08.md).
This supersedes the proposed extraction-plan extension below. The admission
receipt and the first all-372 experiment remain intact.

**Earlier all-372 result: content admission is complete; the first canonical head fits
and has exact Rust feature-row parity, but complete pixel serving is refused.**
Seed 4101 produces 100 trees / 6,100 nodes in a 204,790-byte ZCTH. Across 13,892
rows, raw decision error is 0 ulp, probability error is 0 and the fire set has
zero disagreements. The pixel audit then refuses the head's 372 declared
features because D's fast extraction plan does not compute all of them. No
`SCREEN.json` or `COMPLETE.json` exists for this seed; seeds 4103/4107 have not
run. This is a serving-plan incompatibility, not a measured quality verdict.

The user directed **`zenextras/zenexr` over the Rust
`exr` crate**, explicitly authorizing that dependency. This replaces the
unpushed custom zenbitmaps port; its tests, source and pixels remain preserved
in `native-exr-port-2026-09-08/`. The replacement matches all 98 saved fixtures
and all 30 reference outputs exactly; [validation and decoder contract](../../zenextras/benchmarks/zenexr_validation_2026-09-08.md).
The subsequent reference screen and contextual review are recorded below. This decoder
work is separate from HDR training inputs: imazen-26 already has 76 HDR PNGs
and 1,140 scale variants. [Confirmed source binding](../docs/TARGET_STEERING_PROTOCOL_2026-09-08.md#existing-hdr-png-inputs--user-correction-september-8).

## Completion of the remaining reference audit

The same audit owner adds `--native-linear`: PNG/EXR decoded through zenpng and
zenexr, linear BT.709 luminance, per-image peak normalization and a fixed log
transform, Lanczos f32 to 9×8, shared dHash comparison order. The
[registration](../docs/CANONICAL_CORRUPTION_REFIT_2026-09-08.md) predates results.
The existing untagged training PNGs require an explicit, per-file recorded
sRGB interpretation matching their training recipe; default unknown-transfer
refusal remains. No HDR units or perceptual-quality claim follows from this
content fingerprint. Original EXR metadata remains in the audit.

All 12 sources × 30 HDR references complete with **zero strict <=10 flags**,
minimum distance 15. Two <=16 matches involve origin 6068, a two-column paper:
reference 14 is nighttime buildings and a statue/fountain; reference 15 shows
people with laptops in a room opening onto a bright terrace. Visual review
finds distinct content; no exclusion. The original source-family split stays
fixed. The hash remains crop-blind and is not a proof against every transform.

The full earlier 12 × 182 SDR audit reproduces byte-for-byte, both TSV and JSON.
Three unit tests cover bit order, exposure invariance, unrelated data and
invalid numerics/geometry. Two CLI exact-image positives and nine refusals pass;
Clippy and formatting pass. A complete admission receipt binds 445 files,
including original references, the prior SDR review, new HDR review, input
identities, registration, controls and decoder binary. Human scores and distorted
holdout images were not read. The fit manifest changes only the admission pin.

New packet: `/mnt/v/output/zensim/canonical-corruption-refit-final-2026-09-08/`.
`CONTENT_ADMISSION.json`, `FIT_MANIFEST.json`, `hdr-audit.hashes.json`,
`HDR_CLOSE_PAIR_REVIEW.json`, `CONTROLS.json` and `fit/seed-4101/` retain the
complete result and failed pixel-audit log. Earlier preparation artifacts below
remain immutable. `thiserror` and its derive companion move from 2.0.19 to
2.0.20 to meet the pinned zenexr dependency; no other locked package changes.

Then-proposed next step (superseded by the later D-regime user request): extend
the existing `BakeScorer::plan` using supported complete-model
requirements, preserving revision and unsupported-feature rejection. Verify
pixel/cache/stored-feature parity and cost before rerunning the registered fit.
Do not truncate the head, weaken coverage checks, alter the feature schema or
claim validation from the feature-array parity result alone.

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

The earlier preparation stage ended awaiting EXR reference admission. The
later completion and first-fit serving refusal above supersede that pending
status; do not rerun the admission packet without changed inputs or code.
