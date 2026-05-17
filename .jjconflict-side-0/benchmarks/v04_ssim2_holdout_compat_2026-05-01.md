# Profile compatibility report — V0_2 vs V0_4

Bake: `v04_mlp_ssim2_holdout_20260501T045510.bin`

All scores use the standard `100 - 18·d^0.7` mapping, clamped at 0.
Distances below are profile-internal raw weighted distances, lower = more similar.

## Cross-profile correlation

Pearson + Kendall τ between V0_2 and V0_4 raw distances on the SAME pairs.
τ = 1 means rank-equivalent; τ < 1 means the profiles disagree on relative ordering.

| Dataset | n | r(d_V02, d_V04) | τ(d_V02, d_V04) | r(s_V02, s_V04) | mean Δscore | median Δscore | σ Δscore |
|---------|--:|:----:|:----:|:----:|:----:|:----:|:----:|
| KADIK10k | 10125 | 0.9789 | 0.8797 | 0.7829 | -46.21 | -52.33 | 53.36 |
| TID2013 | 3000 | 0.9743 | 0.8539 | 0.7868 | -45.35 | -50.57 | 57.32 |
| CID22 | 4292 | 0.9811 | 0.9045 | 0.4473 | +22.67 | +22.70 | 9.77 |

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
| KADIK10k | 627750 | 0.8192 | 0.8432 | **0.8034** | **0.8217** | 0.9758 |
| TID2013 | 169725 | 0.8427 | 0.8401 | **0.8525** | **0.8485** | 0.9681 |
| CID22 | 186086 | 0.8676 | 0.8893 | **0.7976** | **0.8158** | 0.9769 |

## V0_2 score distribution across the corpus

Step-5 buckets across the full V0_2 score range. Distortion sweeps must cover
the low-q regime with the same density as high-q (CLAUDE.md sweep rule).
A bucket with n=0 means no pairs in that V0_2 score band — the corpus doesn't
exercise that regime, NOT that V0_4 has been calibrated there.

| V0_2 bucket | n | V0_4 score p10 | p25 | median | p75 | p90 | mean Δ |
|:--:|--:|:--:|:--:|:--:|:--:|:--:|:--:|
| [-100, -95) | 658 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -0.21 |
| [-95, -90) | 62 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -7.80 |
| [-90, -85) | 83 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -12.90 |
| [-85, -80) | 103 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -17.42 |
| [-80, -75) | 100 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -22.73 |
| [-75, -70) | 123 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -27.89 |
| [-70, -65) | 143 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -32.40 |
| [-65, -60) | 148 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -37.66 |
| [-60, -55) | 176 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -42.67 |
| [-55, -50) | 172 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -47.51 |
| [-50, -45) | 188 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -52.46 |
| [-45, -40) | 187 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -57.38 |
| [-40, -35) | 198 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -62.69 |
| [-35, -30) | 224 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -67.49 |
| [-30, -25) | 216 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -72.45 |
| [-25, -20) | 274 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -77.33 |
| [-20, -15) | 273 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -82.54 |
| [-15, -10) | 315 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -87.59 |
| [-10, -5) | 349 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -92.38 |
| [-5, 0) | 370 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -97.23 |
| [0, 5) | 394 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -101.34 |
| [5, 10) | 372 | -100.0 | -100.0 | -100.0 | -100.0 | -100.0 | -105.51 |
| [10, 15) | 380 | -100.0 | -100.0 | -100.0 | -100.0 | -74.2 | -105.86 |
| [15, 20) | 404 | -100.0 | -100.0 | -100.0 | -92.7 | -60.5 | -107.76 |
| [20, 25) | 425 | -100.0 | -100.0 | -100.0 | -63.6 | -36.1 | -102.73 |
| [25, 30) | 426 | -100.0 | -100.0 | -88.3 | -48.0 | -15.4 | -98.85 |
| [30, 35) | 451 | -100.0 | -100.0 | -72.5 | -40.2 | 2.3 | -94.86 |
| [35, 40) | 524 | -100.0 | -83.6 | -45.8 | -10.3 | 29.4 | -77.91 |
| [40, 45) | 555 | -90.0 | -57.2 | -24.5 | 6.1 | 52.5 | -63.41 |
| [45, 50) | 567 | -70.4 | -40.7 | 1.7 | 28.3 | 73.0 | -47.81 |
| [50, 55) | 637 | -46.6 | -10.5 | 28.3 | 57.7 | 100.0 | -27.81 |
| [55, 60) | 748 | -15.9 | 23.2 | 63.3 | 100.0 | 100.0 | -3.67 |
| [60, 65) | 894 | 20.2 | 100.0 | 100.0 | 100.0 | 100.0 | +20.54 |
| [65, 70) | 1066 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +26.63 |
| [70, 75) | 1300 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +26.50 |
| [75, 80) | 1410 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +22.47 |
| [80, 85) | 1376 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +17.67 |
| [85, 90) | 631 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +12.95 |
| [90, 95) | 170 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +8.17 |
| [95, 100) | 5 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +4.85 |

**599 pairs at V0_2 = -100 (clamped)** — V0_4 score for these: p10 = -100.0, median = -100.0, p90 = -100.0.
V0_2's score mapping has flattened these to a single value; V0_4's response on these
pairs is the only signal left for distinguishing 'bad' from 'completely broken'.

## Per-(codec, quality_level) score bias (V0_4 − V0_2)

Median V0_4-minus-V0_2 score delta grouped by `dataset:codec:quality_level`.
For KADIK/TID, level = distortion strength 01..05 (01 = mildest, 05 = harshest).
For CID22, level = encoder quality. Sorted by |median Δ| so the most-disagreeing
(codec, level) cells surface first. Large positive Δ = V0_4 more lenient; negative = V0_4 harsher.

| dataset:codec:level | n | median Δ | p25 Δ | p75 Δ |
|---|--:|:----:|:----:|:----:|
| TID2013:tid:04:q=5 | 25 | -141.85 | -148.43 | -131.52 |
| TID2013:tid:04:q=4 | 25 | -139.62 | -147.99 | -127.05 |
| TID2013:tid:02:q=5 | 25 | -139.22 | -142.75 | -131.75 |
| TID2013:tid:06:q=3 | 25 | -130.29 | -136.20 | -123.73 |
| KADIK10k:kadid:13:q=03 | 81 | -127.73 | -134.98 | -119.12 |
| TID2013:tid:01:q=4 | 25 | -126.81 | -134.20 | -120.17 |
| KADIK10k:kadid:11:q=03 | 81 | -124.81 | -131.95 | -105.10 |
| KADIK10k:kadid:12:q=03 | 81 | -124.23 | -133.00 | -105.24 |
| TID2013:tid:17:q=3 | 25 | -123.15 | -127.91 | -113.08 |
| TID2013:tid:05:q=3 | 25 | -122.55 | -141.24 | -112.46 |
| TID2013:tid:05:q=4 | 25 | -121.70 | -128.78 | -112.12 |
| TID2013:tid:19:q=4 | 25 | -119.47 | -133.43 | -112.08 |
| KADIK10k:kadid:11:q=04 | 81 | -118.04 | -131.26 | -107.11 |
| TID2013:tid:19:q=3 | 25 | -115.20 | -131.45 | -93.54 |
| KADIK10k:kadid:12:q=04 | 81 | -114.49 | -129.64 | -102.23 |
| KADIK10k:kadid:14:q=04 | 81 | -112.91 | -125.54 | -95.50 |
| KADIK10k:kadid:13:q=02 | 81 | -112.81 | -125.52 | -93.11 |
| KADIK10k:kadid:06:q=04 | 81 | -112.48 | -120.15 | -100.69 |
| TID2013:tid:22:q=4 | 25 | -110.81 | -121.30 | -101.53 |
| TID2013:tid:21:q=3 | 25 | -110.06 | -117.66 | -86.90 |
| TID2013:tid:06:q=4 | 25 | -110.04 | -122.77 | -99.56 |
| TID2013:tid:22:q=3 | 25 | -109.68 | -118.52 | -94.27 |
| KADIK10k:kadid:13:q=04 | 81 | -108.38 | -124.04 | -96.42 |
| KADIK10k:kadid:15:q=02 | 81 | -108.34 | -114.71 | -101.62 |
| TID2013:tid:03:q=2 | 25 | -108.05 | -121.20 | -84.04 |
| KADIK10k:kadid:04:q=02 | 81 | -106.77 | -115.89 | -82.93 |
| TID2013:tid:18:q=5 | 25 | -106.67 | -114.98 | -78.90 |
| TID2013:tid:09:q=4 | 25 | -105.98 | -112.97 | -96.40 |
| TID2013:tid:07:q=3 | 25 | -105.47 | -119.53 | -91.49 |
| TID2013:tid:03:q=3 | 25 | -105.13 | -114.48 | -98.53 |
| TID2013:tid:02:q=4 | 25 | -104.52 | -116.22 | -88.65 |
| KADIK10k:kadid:14:q=03 | 81 | -104.22 | -120.35 | -72.67 |
| KADIK10k:kadid:11:q=02 | 81 | -104.05 | -117.64 | -69.50 |
| TID2013:tid:07:q=2 | 25 | -103.75 | -119.31 | -58.83 |
| KADIK10k:kadid:03:q=05 | 81 | -101.71 | -111.51 | -82.28 |
| KADIK10k:kadid:17:q=04 | 81 | -101.60 | -107.25 | -91.72 |
| KADIK10k:kadid:16:q=03 | 81 | -101.14 | -109.07 | -82.59 |
| KADIK10k:kadid:01:q=04 | 81 | -101.13 | -116.18 | -81.86 |
| TID2013:tid:01:q=5 | 25 | -100.95 | -109.13 | -89.95 |
| TID2013:tid:21:q=4 | 25 | -100.67 | -112.20 | -89.85 |

### Codec-aggregated medians (collapses level — read with caution)

| dataset:codec | n | median Δ | p25 Δ | p75 Δ |
|---|--:|:----:|:----:|:----:|
| KADIK10k:kadid:13 | 405 | -102.51 | -124.04 | -69.81 |
| KADIK10k:kadid:11 | 405 | -101.37 | -121.44 | -64.26 |
| TID2013:tid:06 | 125 | -98.09 | -122.69 | -76.44 |
| KADIK10k:kadid:12 | 405 | -92.67 | -119.38 | -37.11 |
| TID2013:tid:01 | 125 | -88.49 | -117.16 | +6.36 |
| TID2013:tid:19 | 125 | -86.62 | -115.30 | -28.62 |
| TID2013:tid:15 | 125 | -84.26 | -91.49 | -76.51 |
| KADIK10k:kadid:06 | 405 | -81.95 | -105.15 | -48.89 |
| KADIK10k:kadid:15 | 405 | -79.64 | -101.80 | -43.12 |
| TID2013:tid:04 | 125 | -78.74 | -135.16 | +18.08 |
| KADIK10k:kadid:14 | 405 | -76.04 | -112.91 | +12.53 |
| KADIK10k:kadid:07 | 405 | -75.34 | -101.80 | -34.12 |
| KADIK10k:kadid:21 | 405 | -73.16 | -97.37 | -28.56 |
| KADIK10k:kadid:05 | 405 | -72.60 | -90.35 | -45.08 |
| KADIK10k:kadid:25 | 405 | -70.32 | -91.69 | -38.49 |
| TID2013:tid:03 | 125 | -69.95 | -101.43 | -39.26 |
| TID2013:tid:07 | 125 | -68.50 | -104.10 | -29.91 |
| TID2013:tid:05 | 125 | -64.50 | -116.47 | +18.22 |
| TID2013:tid:22 | 125 | -61.17 | -103.92 | +0.00 |
| KADIK10k:kadid:20 | 405 | -59.51 | -93.97 | +1.79 |

## Top-30 largest score disagreements

Pairs where V0_2 and V0_4 score the SAME comparison most differently. Useful for spotting failure modes.

| dataset | reference | codec | quality | V0_2 score | V0_4 score | Δ |
|---|---|---|---|--:|--:|--:|
| TID2013 | i08.BMP | tid:04 | 5 | 55.00 | -100.00 | -155.00 |
| TID2013 | i09.BMP | tid:04 | 4 | 53.92 | -100.00 | -153.92 |
| TID2013 | i12.BMP | tid:04 | 4 | 53.05 | -100.00 | -153.05 |
| TID2013 | i13.BMP | tid:04 | 5 | 52.89 | -100.00 | -152.89 |
| TID2013 | i03.BMP | tid:04 | 4 | 51.73 | -100.00 | -151.73 |
| TID2013 | i10.BMP | tid:04 | 4 | 51.59 | -100.00 | -151.59 |
| TID2013 | i04.BMP | tid:05 | 3 | 51.43 | -100.00 | -151.43 |
| TID2013 | i16.BMP | tid:04 | 4 | 51.22 | -100.00 | -151.22 |
| TID2013 | i14.BMP | tid:04 | 5 | 50.88 | -100.00 | -150.88 |
| TID2013 | i04.BMP | tid:04 | 3 | 50.24 | -100.00 | -150.24 |
| TID2013 | i05.BMP | tid:04 | 5 | 50.21 | -100.00 | -150.21 |
| TID2013 | i21.BMP | tid:04 | 5 | 49.91 | -100.00 | -149.91 |
| TID2013 | i04.BMP | tid:02 | 4 | 50.61 | -99.01 | -149.62 |
| TID2013 | i13.BMP | tid:02 | 5 | 48.94 | -100.00 | -148.94 |
| TID2013 | i19.BMP | tid:04 | 5 | 48.69 | -100.00 | -148.69 |
| TID2013 | i06.BMP | tid:04 | 5 | 48.43 | -100.00 | -148.43 |
| TID2013 | i15.BMP | tid:04 | 4 | 48.00 | -100.00 | -148.00 |
| TID2013 | i20.BMP | tid:04 | 4 | 55.50 | -92.49 | -147.99 |
| TID2013 | i14.BMP | tid:02 | 5 | 47.93 | -100.00 | -147.93 |
| TID2013 | i08.BMP | tid:02 | 5 | 47.55 | -100.00 | -147.55 |
| TID2013 | i11.BMP | tid:04 | 5 | 47.33 | -100.00 | -147.33 |
| TID2013 | i16.BMP | tid:05 | 3 | 47.14 | -100.00 | -147.14 |
| TID2013 | i01.BMP | tid:04 | 5 | 47.03 | -100.00 | -147.03 |
| TID2013 | i02.BMP | tid:04 | 4 | 46.99 | -100.00 | -146.99 |
| TID2013 | i17.BMP | tid:04 | 4 | 56.84 | -90.11 | -146.96 |
| TID2013 | i25.BMP | tid:04 | 4 | 49.83 | -94.97 | -144.80 |
| KADIK10k | I74.png | kadid:11 | 04 | 45.26 | -99.51 | -144.78 |
| TID2013 | i01.BMP | tid:04 | 4 | 62.72 | -81.91 | -144.63 |
| TID2013 | i23.BMP | tid:04 | 5 | 44.07 | -100.00 | -144.07 |
| TID2013 | i05.BMP | tid:02 | 5 | 51.38 | -92.62 | -144.00 |

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
| V0_4 raw distance | -17.6807 | 7.5031 |
| V0_2 score | 69.76 | 4.45 |
| V0_4 score | 99.26 | 5.42 |

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
| V0_4 raw distance | -21.7857 | 7.2419 |
| V0_2 score | 72.12 | 4.51 |
| V0_4 score | 99.57 | 4.50 |

Cloudinary Table 4 reference values for BPG at PJND:
- SSIMULACRA 2: 65.38 ± 5.10
- DSSIM ×1000: 3.357 ± 1.267
- Butteraugli 3-norm: 1.528 ± 0.192
- MS-SSIM ×100: 99.21 ± 0.40
- VMAF: 90.05 ± 2.25
- PSNR-Y: 39.61 ± 2.98
- PSNR-HVS: 40.31 ± 1.78

