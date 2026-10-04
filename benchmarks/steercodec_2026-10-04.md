# STEERCODEC — the steering map as a per-block allocator for zenjpeg, JPEG XL and zqi (2026-10-04)

Owner, 2026-10-04: "Eval as a jxl and zenjpeg steering map and zqi." Development diagnostic on unlabelled stimuli; block swaps of two
independent encodes stand in for per-block quantization decisions, so nothing here is an encoder rate-distortion result.

## Setup

* Images: 8 CLIC 2025 tuning photos, centre 512×512 crops (`steercodec_2026-10-04/src_manifest.json`), 4,096 8×8 blocks each.
* Codecs, each at q30, q60, q85: zenjpeg 4:4:4 (`zenmetrics sweep`, image `exec-cvvdp-0a61830d`), JPEG XL (zenjxl, same
  image), zqi lossy (imazen/zqi b5e35ef5, default XYB, `steercodec_2026-10-04/zqi_roundtrip`). Swaps: q30→60 and q60→85.
* Models (Rev4, prepared steering): the R7 full-data confirm bakes of by_v2fy and v2 + basic at cv16:cf98, seed 0
  (`byv2fy-full-s0` sha256 `802c6369…`, `v2basic-full-s0` `3db5b626…`), densified and stamped Rev4. Tool
  `diffmap_block_coherence` sha256 `39eae0f6…` (zensim ce55429c), block 8, `ZENSIM_REPAIR_SOURCE` = the hi decode, with the
  neighbour-exact engine off (x0) and on (x1, `ZENSIM_NEIGHBOUR_EXACT=1`).
* Per case: M2 (gradient linearization vs true change) and M3f (map `refinement_gain` vs true single-block change, SROCC over all
  4,096 blocks). Allocation: upgrade 25 % of blocks of the lo decode to the hi decode, chosen by the map, by the oracle (true
  single-block zensim change), at random, or by the lowest map gain (anti); each composite's share of the full upgrade's gain,
  judged by zensim (the same bake), SSIMULACRA2 and butteraugli 3-norm (zenmetrics `score-pairs`). Medians over the 8 images.
  Harness `steercodec_2026-10-04/steer_codec.py`; summary `steercodec_2026-10-04/summary_rev4.json`; raw per-block JSONs and
  scores on tower `output/zensim/steercodec-2026-10-04/`.

## Results (Rev4)

Share of the full upgrade's gain captured by 25 % of blocks — map / oracle / random / anti:

| codec | swap | model | x | M3f | zensim | SSIMULACRA2 | butteraugli |
|---|---|---|---|---:|---|---|---|
| zenjpeg | 30→60 | by_v2fy | 0 | 0.772 | 0.59 / 0.62 / 0.22 / 0.02 | 0.43 / 0.43 / 0.19 / 0.10 | 0.34 / 0.38 / 0.19 / 0.10 |
| zenjpeg | 30→60 | by_v2fy | 1 | 0.828 | 0.59 / 0.62 / 0.22 / −0.00 | 0.42 / 0.43 / 0.19 / 0.09 | 0.37 / 0.38 / 0.19 / 0.10 |
| zenjpeg | 60→85 | by_v2fy | 0 | 0.783 | 0.57 / 0.53 / 0.21 / 0.03 | 0.39 / 0.35 / 0.15 / 0.07 | 0.32 / 0.28 / 0.16 / 0.08 |
| zenjpeg | 60→85 | by_v2fy | 1 | 0.886 | 0.50 / 0.53 / 0.21 / −0.00 | 0.33 / 0.35 / 0.15 / 0.06 | 0.26 / 0.28 / 0.16 / 0.07 |
| JPEG XL | 30→60 | by_v2fy | 0 | 0.740 | 0.79 / 0.84 / 0.22 / −0.05 | 0.63 / 0.62 / 0.21 / 0.04 | 0.68 / 0.64 / 0.19 / 0.03 |
| JPEG XL | 30→60 | by_v2fy | 1 | 0.793 | 0.79 / 0.84 / 0.22 / −0.07 | 0.63 / 0.62 / 0.21 / 0.04 | 0.67 / 0.64 / 0.19 / 0.03 |
| JPEG XL | 60→85 | by_v2fy | 0 | 0.827 | 0.64 / 0.66 / 0.20 / −0.02 | 0.55 / 0.55 / 0.18 / 0.04 | 0.47 / 0.45 / 0.16 / 0.03 |
| JPEG XL | 60→85 | by_v2fy | 1 | 0.848 | 0.64 / 0.66 / 0.20 / −0.02 | 0.55 / 0.55 / 0.18 / 0.05 | 0.48 / 0.45 / 0.16 / 0.04 |
| zqi | 30→60 | by_v2fy | 0 | 0.824 | 0.56 / 0.56 / 0.20 / 0.01 | 0.38 / 0.36 / 0.18 / 0.07 | 0.32 / 0.29 / 0.17 / 0.08 |
| zqi | 30→60 | by_v2fy | 1 | 0.861 | 0.55 / 0.56 / 0.20 / −0.00 | 0.36 / 0.36 / 0.18 / 0.08 | 0.30 / 0.29 / 0.17 / 0.08 |
| zqi | 60→85 | by_v2fy | 0 | 0.879 | 0.46 / 0.47 / 0.18 / 0.01 | 0.34 / 0.35 / 0.16 / 0.07 | 0.28 / 0.27 / 0.15 / 0.06 |
| zqi | 60→85 | by_v2fy | 1 | 0.893 | 0.46 / 0.47 / 0.18 / 0.01 | 0.34 / 0.35 / 0.16 / 0.07 | 0.26 / 0.27 / 0.15 / 0.06 |

v2 + basic rows are in the summary JSON; they track by_v2fy within 0.00–0.08 (largest gap: JPEG XL 30→60 zensim share 0.71 vs 0.79).
M2 ≥ 0.9987 in every case.

## Reading

* On all three codecs the map's top quarter captures 46–79 % of the full upgrade's zensim gain, between 0.05 below and 0.04
  above the single-block oracle (x0), and 2.3–3.6× random; the lowest-gain quarter captures ~0. SSIMULACRA2 and butteraugli,
  which never saw the map, agree: under them the map's share is within −0.04 to +0.04 of the zensim oracle's (below it only on
  zenjpeg 30→60 butteraugli, 0.34 vs 0.38, and zqi 60→85 SSIMULACRA2, 0.34 vs 0.35).
* The single-block oracle is not the best joint allocator: on zenjpeg 60→85 the map (0.57) beats it (0.53), and the
  independent judges show the same order. Blocks interact; the smoother frozen-density ranking allocates jointly better.
* The neighbour-exact engine raises M3f by 0.02–0.10 (it ranks single-block repairs more like the oracle) but does not raise
  the allocation shares, and on zenjpeg 60→85 lowers them (zensim 0.57 → 0.50, SSIMULACRA2 0.39 → 0.33). For top-quarter
  allocation it is not an improvement; its value is per-block accuracy (target control, small budgets, sign).
* by_v2fy steers as well as v2 + basic on all three codecs.

## Results (Rev5, 2026-10-04 23:50 UTC)

Same images, decodes, swaps and harness with the Rev5 full-data bakes, seed 0 (`byv2fy5` = `~/tmp/rev5bakes/byv2fy-full-s0.bin`
sha256 `690b2709…`, `v2basic5` `51c724a8…`; R5CONFIRM, zensim 2206946f), `ZENSIM_FORMULA_REV=5`, tools rebuilt at zensim 70a5066e
(`diffmap_block_coherence` `28e22ee2…`, `serve_custom_bake` `0243f071…`). The SSIMULACRA2 and butteraugli sidecars were recomputed
over all 1,608 pairs (Rev4 and Rev5 composites). Summary `steercodec_2026-10-04/summary_rev5.json`; rows (sha256 `51a0bc44…`),
per-block JSONs and scores on tower `output/zensim/steercodec-2026-10-04/` (`summary_rev5_full.json`, `swap/`, `score-rev5/`).

by_v2fy, Rev4 → Rev5, share of the full upgrade's gain at 25 % of blocks, map / oracle (independent judges):

| codec | swap | x | M3f | SSIMULACRA2 | butteraugli |
|---|---|---|---|---|---|
| zenjpeg | 30→60 | 0 | 0.772 → 0.764 | 0.43/0.43 → 0.44/0.42 | 0.34/0.38 → 0.34/0.38 |
| zenjpeg | 60→85 | 0 | 0.783 → 0.789 | 0.39/0.35 → 0.39/0.35 | 0.32/0.28 → 0.31/0.29 |
| zenjpeg | 60→85 | 1 | 0.886 → 0.885 | 0.33/0.35 → 0.34/0.35 | 0.26/0.28 → 0.29/0.29 |
| JPEG XL | 30→60 | 0 | 0.740 → 0.737 | 0.63/0.62 → 0.65/0.63 | 0.68/0.64 → 0.69/0.64 |
| JPEG XL | 60→85 | 0 | 0.827 → 0.818 | 0.55/0.55 → 0.56/0.55 | 0.47/0.45 → 0.48/0.45 |
| zqi | 30→60 | 0 | 0.824 → 0.827 | 0.38/0.36 → 0.38/0.37 | 0.32/0.29 → 0.32/0.30 |
| zqi | 60→85 | 0 | 0.879 → 0.885 | 0.34/0.35 → 0.35/0.35 | 0.28/0.27 → 0.28/0.28 |

Rev5 steers like Rev4 on all three codecs: every independent-judge share moves by at most 0.03 (largest: zenjpeg 60→85 x1
butteraugli +0.03), M3f by at most 0.01, M2 ≥ 0.9968. The Rev4 neighbour-exact loss on zenjpeg 60→85 (zensim share 0.57 → 0.50)
is absent at Rev5 (0.55 → 0.55). Random and anti shares are unchanged (0.15–0.21 and ≤ 0.11).

Not measured: budgets other than 25 %, block sizes other than 8×8, byte-equalised allocation, and any
real encoder integration (zenjpeg AQ, zqi `aq_gamma`, JXL's adaptive quantization field).
