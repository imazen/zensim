# Profile compatibility report — V0_2 vs V0_4

Bake: `v04_mlp_v5znpr2_20260430T044620.bin`

All scores use the standard `100 - 18·d^0.7` mapping, clamped at 0.
Distances below are profile-internal raw weighted distances, lower = more similar.

## Cross-profile correlation

Pearson + Kendall τ between V0_2 and V0_4 raw distances on the SAME pairs.
τ = 1 means rank-equivalent; τ < 1 means the profiles disagree on relative ordering.

| Dataset | n | r(d_V02, d_V04) | τ(d_V02, d_V04) | r(s_V02, s_V04) | mean Δscore | median Δscore | σ Δscore |
|---------|--:|:----:|:----:|:----:|:----:|:----:|:----:|
| KADIK10k | 10125 | 0.6834 | 0.7051 | 0.8255 | +63.32 | +59.79 | 34.42 |
| TID2013 | 3000 | 0.8276 | 0.7362 | 0.8942 | +54.43 | +51.08 | 27.02 |
| CID22 | 4292 | 0.9095 | 0.8610 | 0.1339 | +24.36 | +23.41 | 8.51 |

## Pairwise SROCC against human MOS

For each (reference, A, B) triplet sharing a reference image, compute the
signed differences `(human_A − human_B)` and `(distance_A − distance_B)` and
report `|SROCC|` between them. This is the codec-A-vs-B prediction skill —
what downstream codec gates actually need.

Per the Cloudinary CID22 paper (Table 6 in Sneyers et al. 2023), pairwise
correlation is generally HIGHER than absolute correlation (Table 3) since
relative comparisons within a reference cancel image-content variance.

**CID22 is the relevant column** for codec evaluation. KADID and TID2013
contain mostly non-compression distortions (paper p. 2: <5%% of KADID images
are compression-relevant; TID2013 similar). High SROCC there shows the MLP
can rank synthetic distortions but doesn't validate codec performance.

| Dataset | n triplets | abs SRCC V0_2 | abs SRCC V0_4 | **pairwise SRCC V0_2** | **pairwise SRCC V0_4** | pairwise τ V0_2-vs-V0_4 |
|---------|--:|:--:|:--:|:--:|:--:|:--:|
| KADIK10k | 627750 | 0.8192 | 0.9397 | **0.8034** | **0.9136** | 0.8266 |
| TID2013 | 169725 | 0.8427 | 0.9446 | **0.8525** | **0.9403** | 0.8614 |
| CID22 | 186086 | 0.8676 | 0.8927 | **0.7976** | **0.7121** | 0.8357 |

## V0_2 score distribution across the corpus

Step-5 buckets across the full V0_2 score range. Distortion sweeps must cover
the low-q regime with the same density as high-q (CLAUDE.md sweep rule).
A bucket with n=0 means no pairs in that V0_2 score band — the corpus doesn't
exercise that regime, NOT that V0_4 has been calibrated there.

| V0_2 bucket | n | V0_4 score p10 | p25 | median | p75 | p90 | mean Δ |
|:--:|--:|:--:|:--:|:--:|:--:|:--:|:--:|
| [-100, -95) | 658 | -2.9 | 12.1 | 25.7 | 35.3 | 43.1 | +122.86 |
| [-95, -90) | 62 | 0.4 | 25.5 | 37.9 | 49.1 | 54.2 | +126.32 |
| [-90, -85) | 83 | 25.2 | 36.1 | 42.3 | 49.7 | 55.2 | +127.82 |
| [-85, -80) | 103 | 10.2 | 28.0 | 40.3 | 48.7 | 53.5 | +119.57 |
| [-80, -75) | 100 | 13.8 | 28.0 | 41.9 | 52.2 | 55.8 | +116.52 |
| [-75, -70) | 123 | 13.8 | 27.7 | 41.0 | 51.1 | 57.4 | +110.11 |
| [-70, -65) | 143 | 12.3 | 24.3 | 40.6 | 53.3 | 56.8 | +106.58 |
| [-65, -60) | 148 | 13.6 | 30.4 | 46.5 | 54.2 | 61.0 | +105.01 |
| [-60, -55) | 176 | 13.6 | 29.6 | 43.5 | 54.6 | 58.5 | +98.93 |
| [-55, -50) | 172 | 16.6 | 34.8 | 48.9 | 58.5 | 69.9 | +99.57 |
| [-50, -45) | 188 | 19.6 | 36.8 | 48.6 | 57.8 | 64.6 | +94.53 |
| [-45, -40) | 187 | 21.6 | 40.4 | 52.1 | 59.7 | 66.2 | +93.51 |
| [-40, -35) | 198 | 22.8 | 40.5 | 53.2 | 63.6 | 71.4 | +89.69 |
| [-35, -30) | 224 | 26.1 | 46.6 | 57.0 | 64.0 | 74.1 | +87.64 |
| [-30, -25) | 216 | 26.9 | 48.8 | 57.3 | 65.2 | 80.1 | +83.83 |
| [-25, -20) | 274 | 33.9 | 52.0 | 58.7 | 65.6 | 84.3 | +81.88 |
| [-20, -15) | 273 | 33.3 | 53.9 | 60.8 | 68.0 | 93.2 | +78.65 |
| [-15, -10) | 315 | 42.4 | 55.5 | 63.2 | 71.0 | 100.0 | +77.17 |
| [-10, -5) | 349 | 41.3 | 56.8 | 66.5 | 78.9 | 100.0 | +75.40 |
| [-5, 0) | 370 | 44.9 | 59.7 | 68.2 | 78.8 | 100.0 | +71.87 |
| [0, 5) | 394 | 45.7 | 62.2 | 72.2 | 90.7 | 100.0 | +70.86 |
| [5, 10) | 372 | 40.8 | 65.0 | 72.6 | 85.6 | 100.0 | +65.55 |
| [10, 15) | 380 | 56.7 | 70.7 | 77.8 | 100.0 | 100.0 | +66.66 |
| [15, 20) | 404 | 59.1 | 74.0 | 81.3 | 100.0 | 100.0 | +64.80 |
| [20, 25) | 425 | 58.9 | 76.1 | 88.2 | 100.0 | 100.0 | +62.32 |
| [25, 30) | 426 | 69.6 | 79.1 | 95.2 | 100.0 | 100.0 | +60.47 |
| [30, 35) | 451 | 72.6 | 83.4 | 100.0 | 100.0 | 100.0 | +57.25 |
| [35, 40) | 524 | 76.0 | 87.2 | 100.0 | 100.0 | 100.0 | +54.98 |
| [40, 45) | 555 | 82.8 | 100.0 | 100.0 | 100.0 | 100.0 | +52.84 |
| [45, 50) | 567 | 83.6 | 100.0 | 100.0 | 100.0 | 100.0 | +48.34 |
| [50, 55) | 637 | 89.3 | 100.0 | 100.0 | 100.0 | 100.0 | +44.89 |
| [55, 60) | 748 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +41.05 |
| [60, 65) | 894 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +36.68 |
| [65, 70) | 1066 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +32.20 |
| [70, 75) | 1300 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +27.34 |
| [75, 80) | 1410 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +22.48 |
| [80, 85) | 1376 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +17.67 |
| [85, 90) | 631 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +12.95 |
| [90, 95) | 170 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +8.17 |
| [95, 100) | 5 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +4.85 |

**599 pairs at V0_2 = -100 (clamped)** — V0_4 score for these: p10 = -3.6, median = 25.0, p90 = 41.3.
V0_2's score mapping has flattened these to a single value; V0_4's response on these
pairs is the only signal left for distinguishing 'bad' from 'completely broken'.

## Per-(codec, quality_level) score bias (V0_4 − V0_2)

Median V0_4-minus-V0_2 score delta grouped by `dataset:codec:quality_level`.
For KADIK/TID, level = distortion strength 01..05 (01 = mildest, 05 = harshest).
For CID22, level = encoder quality. Sorted by |median Δ| so the most-disagreeing
(codec, level) cells surface first. Large positive Δ = V0_4 more lenient; negative = V0_4 harsher.

| dataset:codec:level | n | median Δ | p25 Δ | p75 Δ |
|---|--:|:----:|:----:|:----:|
| KADIK10k:kadid:17:q=05 | 81 | +135.43 | +129.19 | +139.75 |
| KADIK10k:kadid:25:q=01 | 81 | +133.58 | +122.67 | +143.75 |
| KADIK10k:kadid:23:q=05 | 81 | +133.14 | +123.51 | +144.11 |
| KADIK10k:kadid:16:q=04 | 81 | +127.97 | +102.76 | +140.25 |
| KADIK10k:kadid:16:q=05 | 81 | +127.23 | +122.91 | +131.90 |
| KADIK10k:kadid:23:q=04 | 81 | +125.96 | +111.55 | +135.97 |
| KADIK10k:kadid:24:q=04 | 81 | +123.26 | +115.69 | +131.39 |
| KADIK10k:kadid:08:q=05 | 81 | +123.04 | +105.38 | +130.44 |
| TID2013:tid:07:q=5 | 25 | +119.62 | +101.35 | +128.64 |
| KADIK10k:kadid:07:q=05 | 81 | +117.86 | +93.82 | +128.99 |
| KADIK10k:kadid:04:q=05 | 81 | +117.63 | +105.22 | +125.10 |
| KADIK10k:kadid:04:q=04 | 81 | +116.07 | +91.47 | +128.68 |
| TID2013:tid:03:q=5 | 25 | +114.83 | +105.39 | +125.88 |
| KADIK10k:kadid:24:q=03 | 81 | +113.87 | +102.88 | +126.43 |
| KADIK10k:kadid:23:q=03 | 81 | +111.86 | +95.56 | +121.21 |
| TID2013:tid:17:q=4 | 25 | +111.64 | +106.10 | +118.04 |
| KADIK10k:kadid:15:q=05 | 81 | +110.51 | +103.17 | +121.99 |
| KADIK10k:kadid:25:q=05 | 81 | +107.77 | +97.68 | +115.56 |
| KADIK10k:kadid:08:q=04 | 81 | +107.38 | +82.30 | +124.17 |
| KADIK10k:kadid:24:q=05 | 81 | +107.11 | +100.57 | +114.46 |
| TID2013:tid:17:q=5 | 25 | +104.63 | +98.44 | +108.74 |
| TID2013:tid:20:q=5 | 25 | +102.18 | +82.29 | +120.18 |
| TID2013:tid:13:q=5 | 25 | +100.91 | +91.66 | +112.26 |
| KADIK10k:kadid:04:q=03 | 81 | +100.08 | +76.44 | +119.36 |
| KADIK10k:kadid:25:q=02 | 81 | +98.74 | +90.40 | +106.46 |
| TID2013:tid:05:q=5 | 25 | +97.96 | +89.40 | +108.96 |
| KADIK10k:kadid:24:q=02 | 81 | +97.94 | +85.67 | +107.51 |
| KADIK10k:kadid:23:q=02 | 81 | +95.73 | +87.98 | +106.44 |
| KADIK10k:kadid:17:q=04 | 81 | +94.55 | +83.04 | +104.56 |
| KADIK10k:kadid:15:q=04 | 81 | +92.64 | +86.03 | +100.15 |
| TID2013:tid:14:q=5 | 25 | +91.68 | +77.84 | +100.99 |
| TID2013:tid:07:q=4 | 25 | +90.26 | +78.15 | +98.20 |
| TID2013:tid:23:q=5 | 25 | +89.83 | +79.12 | +103.86 |
| TID2013:tid:10:q=5 | 25 | +88.94 | +83.71 | +95.32 |
| TID2013:tid:24:q=5 | 25 | +88.14 | +82.21 | +93.11 |
| TID2013:tid:21:q=5 | 25 | +87.68 | +79.94 | +91.08 |
| TID2013:tid:03:q=4 | 25 | +87.36 | +78.70 | +96.23 |
| KADIK10k:kadid:10:q=05 | 81 | +87.11 | +79.05 | +94.99 |
| TID2013:tid:09:q=5 | 25 | +86.91 | +80.54 | +94.02 |
| KADIK10k:kadid:16:q=03 | 81 | +86.90 | +74.88 | +94.02 |

### Codec-aggregated medians (collapses level — read with caution)

| dataset:codec | n | median Δ | p25 Δ | p75 Δ |
|---|--:|:----:|:----:|:----:|
| KADIK10k:kadid:23 | 405 | +110.10 | +91.78 | +128.91 |
| KADIK10k:kadid:24 | 405 | +106.13 | +89.12 | +119.59 |
| KADIK10k:kadid:25 | 405 | +92.94 | +60.90 | +114.43 |
| KADIK10k:kadid:04 | 405 | +85.97 | +51.31 | +118.41 |
| KADIK10k:kadid:16 | 405 | +84.77 | +39.18 | +123.67 |
| KADIK10k:kadid:07 | 405 | +77.35 | +58.63 | +107.96 |
| TID2013:tid:15 | 125 | +76.75 | +68.21 | +84.80 |
| KADIK10k:kadid:15 | 405 | +74.66 | +63.04 | +96.71 |
| KADIK10k:kadid:20 | 405 | +72.35 | +62.37 | +82.23 |
| KADIK10k:kadid:08 | 405 | +71.77 | +43.66 | +110.67 |
| KADIK10k:kadid:05 | 405 | +71.63 | +60.69 | +83.03 |
| KADIK10k:kadid:21 | 405 | +70.15 | +58.43 | +80.36 |
| TID2013:tid:17 | 125 | +69.33 | +57.12 | +105.62 |
| TID2013:tid:14 | 125 | +67.90 | +49.52 | +83.00 |
| TID2013:tid:03 | 125 | +66.13 | +53.68 | +94.35 |
| TID2013:tid:07 | 125 | +64.48 | +53.59 | +94.97 |
| TID2013:tid:13 | 125 | +62.68 | +49.62 | +84.04 |
| KADIK10k:kadid:06 | 405 | +58.89 | +47.82 | +75.47 |
| TID2013:tid:12 | 125 | +58.39 | +44.39 | +72.87 |
| KADIK10k:kadid:11 | 405 | +56.10 | +44.30 | +67.45 |

## Top-30 largest score disagreements

Pairs where V0_2 and V0_4 score the SAME comparison most differently. Useful for spotting failure modes.

| dataset | reference | codec | quality | V0_2 score | V0_4 score | Δ |
|---|---|---|---|--:|--:|--:|
| KADIK10k | I70.png | kadid:16 | 04 | -99.36 | 96.00 | +195.37 |
| KADIK10k | I77.png | kadid:16 | 04 | -91.46 | 100.00 | +191.46 |
| KADIK10k | I39.png | kadid:16 | 04 | -84.91 | 100.00 | +184.91 |
| KADIK10k | I49.png | kadid:16 | 04 | -82.83 | 100.00 | +182.83 |
| KADIK10k | I03.png | kadid:17 | 05 | -89.67 | 86.94 | +176.62 |
| KADIK10k | I36.png | kadid:16 | 04 | -68.74 | 100.00 | +168.74 |
| KADIK10k | I36.png | kadid:16 | 05 | -100.00 | 65.35 | +165.35 |
| KADIK10k | I81.png | kadid:25 | 01 | -63.87 | 100.00 | +163.87 |
| KADIK10k | I79.png | kadid:24 | 02 | -77.71 | 85.40 | +163.12 |
| KADIK10k | I49.png | kadid:16 | 05 | -100.00 | 62.55 | +162.55 |
| KADIK10k | I46.png | kadid:25 | 01 | -62.15 | 100.00 | +162.15 |
| KADIK10k | I19.png | kadid:25 | 01 | -62.00 | 100.00 | +162.00 |
| KADIK10k | I31.png | kadid:25 | 01 | -60.79 | 100.00 | +160.79 |
| KADIK10k | I28.png | kadid:16 | 04 | -78.30 | 82.23 | +160.52 |
| KADIK10k | I52.png | kadid:08 | 04 | -100.00 | 60.02 | +160.02 |
| KADIK10k | I35.png | kadid:17 | 05 | -100.00 | 59.54 | +159.54 |
| KADIK10k | I12.png | kadid:25 | 01 | -58.53 | 100.00 | +158.53 |
| KADIK10k | I61.png | kadid:16 | 04 | -65.35 | 92.96 | +158.31 |
| KADIK10k | I70.png | kadid:25 | 01 | -57.75 | 100.00 | +157.75 |
| KADIK10k | I50.png | kadid:25 | 01 | -57.11 | 100.00 | +157.11 |
| KADIK10k | I63.png | kadid:16 | 04 | -71.49 | 84.80 | +156.29 |
| KADIK10k | I70.png | kadid:17 | 05 | -100.00 | 55.85 | +155.85 |
| KADIK10k | I26.png | kadid:25 | 01 | -55.77 | 100.00 | +155.77 |
| KADIK10k | I46.png | kadid:17 | 05 | -100.00 | 55.04 | +155.04 |
| KADIK10k | I38.png | kadid:23 | 04 | -100.00 | 55.01 | +155.01 |
| KADIK10k | I67.png | kadid:25 | 01 | -54.07 | 100.00 | +154.07 |
| KADIK10k | I23.png | kadid:17 | 05 | -100.00 | 54.00 | +154.00 |
| KADIK10k | I25.png | kadid:23 | 05 | -100.00 | 53.49 | +153.49 |
| KADIK10k | I47.png | kadid:25 | 01 | -53.45 | 100.00 | +153.45 |
| TID2013 | i04.BMP | tid:05 | 5 | -53.34 | 100.00 | +153.34 |

## Visually-lossless calibration (KonJND-1k)

1008 source images split into 504 JPEG and 504 BPG (no overlap). For each
source, the Probabilistic Just-Noticeable-Difference (PJND) threshold is the
mean file index where observers report just noticing the compression artifact
([Lin, Hosu, Saupe, IEEE T-CSVT 2022](https://ieeexplore.ieee.org/document/9802742)).
Pairs below are at `round(mean PJND)` per source — the canonical near
visually-lossless anchor.

Cloudinary CID22 paper Table 4 publishes the same anchor for nine reference
metrics. Numbers below place V0_2 and V0_4 on the same external scale; mean ±
stdev that's a tight band means the metric agrees with the human PJND notion
of visually-lossless (low cross-source variance), regardless of where the mean
lands on the 0-100 score scale.

### JPEG subset (n = 504)

| metric | mean | stdev |
|---|--:|--:|
| V0_2 raw distance | 2.1120 | 0.4455 |
| V0_4 raw distance | -6.0101 | 1.2995 |
| V0_2 score | 69.76 | 4.45 |
| V0_4 score | 100.00 | 0.00 |

Cloudinary Table 4 reference values for JPEG at PJND:
- SSIMULACRA 2: 63.10 ± 4.65
- DSSIM ×1000: 3.817 ± 1.297
- Butteraugli 3-norm: 1.699 ± 0.229
- MS-SSIM ×100: 99.22 ± 0.38
- VMAF: 91.86 ± 1.90
- PSNR-Y: 36.70 ± 3.79
- PSNR-HVS: 39.96 ± 1.79

### BPG subset (n = 504)

| metric | mean | stdev |
|---|--:|--:|
| V0_2 raw distance | 1.8835 | 0.4378 |
| V0_4 raw distance | -6.1630 | 1.5266 |
| V0_2 score | 72.12 | 4.51 |
| V0_4 score | 100.00 | 0.00 |

Cloudinary Table 4 reference values for BPG at PJND:
- SSIMULACRA 2: 65.38 ± 5.10
- DSSIM ×1000: 3.357 ± 1.267
- Butteraugli 3-norm: 1.528 ± 0.192
- MS-SSIM ×100: 99.21 ± 0.40
- VMAF: 90.05 ± 2.25
- PSNR-Y: 39.61 ± 2.98
- PSNR-HVS: 40.31 ± 1.78

