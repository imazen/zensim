# Honest map-arm coverage and decoder contract — September 8, 2026

The expanded D228 corruption head still fails honest-output protection.
The experiment also exposes a consequential mismatch between native JXL
targeting and canonical independent decoding. No model is qualified and no
validation or terminal family was evaluated.

## Registered expansion and results

The existing corpus owner now imports retained native target bound outputs
through a strict manifest. It checks canonical source/family roles, existing
content admission, all bound cells, encoder records, bitstream hashes and
lengths, extractor/base identities and exact extraction coverage. New rows
receive fresh global keys, full canonical 372-column extraction and independent
Rust pixel/cache audit. The trainer now refuses differing features on duplicate
pixel pairs instead of silently discarding the second producer's values.

The supplement adds 608 attempts on the eight already admitted fit origins:
336 JXL (21 distances, neutral and active maps) and 272 AVIF (17 CQ settings,
both map arms). No new encode occurs. Deduplication leaves **402 new unique
honest pairs**, increasing the fit role from 5,504 to 5,906 unique rows
(5,179 positives, 727 honest). Calibration remains exactly 2,709 unique rows
on its original four sources; validation records are unchanged and unscored.

One seed, 4101, uses the unchanged cost-4 HGB/scaler/calibration recipe and
ZCTH v2 f32 input contract. All **8,615 training array rows** match sklearn
and Rust at zero raw ULP, zero probability delta and zero fire disagreements.
The complete pixel/cache audit covers **9,644 training attempts**. Calibration:

| Check | Result |
|---|---|
| Corruption detection | 2,513 / 2,546 = 98.7038% |
| Real-bug detection | 150 / 155 = 96.7742% |
| Strict below-native-q20 ordering | 2,521 / 2,546 = 99.0181% |
| Honest outputs lowered | 2 / 163 = 1.2270% — FAIL |

The previous cost-4 head lowered three honest calibration outputs. The new
head still fails both zero-native-lowering and <=1% overall lowering, so it
does not advance to validation. This expands codec variation on existing fit
families; it is not a new independent-source generalization estimate.

## Native JXL decoder discrepancy

The supplement retains previous decoded hashes as provenance and measures
current decoded pixels separately. **All 336 JXL attempts differ**, while all
272 AVIF attempts agree exactly. Comparing the same frozen D score through
the current Rust surface with the native bound score gives:

| Codec | Median absolute drift | p95 | Maximum |
|---|---:|---:|---:|
| JXL | 0.185157 | 1.807285 | 2.425388 |
| AVIF | 0.00000103 | 0.00000356 | 0.00000377 |

The largest JXL drift is source 6068, neutral map, distance 0.02186724:
native score 100 versus canonical 97.574612. This exceeds the new three-shot
target tolerance and prevents transferring old native hit counts to the
canonical decoding contract. The input packet itself consistently uses the
canonical decoder; it does not mix native and canonical feature values.

Source inspection finds a concrete candidate cause. Native
`jxl-encoder/examples/zensim_diffmap_rd.rs::decode_jxl_srgb_u8` decodes through
upstream `jxl` into f32 and manually clamps/rounds to u8. Canonical extraction
uses zenjxl's zencodec adapter, selecting zenjxl-decoder U8 output. That decoder
defaults to blue-noise dithering of U8 samples; F32 is undithered. The two paths
also use different decoder packages, so dithering is a **hypothesis requiring
a same-decoder on/off control**, not yet a demonstrated sole cause.

Next establish exact decoded-pixel agreement for the declared native JXL
product path, retaining old byte/score evidence under its original decoder era.
Then address remaining honest source/geometry coverage and qualify actual
targeting/spatial value. Do not remove dithering or change the target scale
merely to recover historical numbers.

## Reproduction and controls

Artifacts: `/mnt/v/output/zensim/corruption-honest-map-2026-09-08/`.
`SUPPLEMENT.json`, `REGISTRATION.md`, `data/FIT_MANIFEST.json` and preserved
source snapshots bind the input and fit. `DECODER_SCORE_DIAGNOSTIC.json` holds
all 608 paired decoded-hash/score comparisons. `fit/seed-4101/` preserves the
head, parity vectors, full audit and failed screen.

Nine malformed-input/immutable-output controls refuse before completion.
A separate real duplicate-pixel control changes one feature value and is
refused before training. Final importer replay reproduces feature CSV,
supplement pixel audit and Parquet byte-for-byte. Python compilation and
605-script lint pass. No Rust inference changed in this experiment; the
previous pinned Rust binaries and their precision/serving tests are reused.
