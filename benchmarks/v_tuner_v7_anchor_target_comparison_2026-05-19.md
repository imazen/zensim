# V6 rule-of-thumb vs V7 empirical anchor targets (2026-05-19)

V6 (`PreviewV0_5TunerV2`, commit `1dd61fc`) shipped with hand-set anchor targets per butter band. Only the 1.5 anchor (CID22 PJND, 63) had empirical grounding. V7 replaces the other 5 with medians from the canonical ssim2 + cvvdp score parquets at `/mnt/v/zen/zensim-training/canonical-2026-05-18/scores/`.

Per-band aggregation: union all (codec, image, q) within `butter_pnorm3 ∈ [band ± 0.5]`, lookup ssim2_gpu and cvvdp_imazen_v0_0_1 by (basename, codec, q), normalize cvvdp via `-log(10 - cvvdp)` and min-max to [0, 100] using the score parquet's global min/max, then per-(codec, band) median.

## Aggregate per-band targets (median across codecs)

`target_score` (V7) = median ssim2 alone, since the V6 rule-of-thumb numbers are calibrated to ssim2's 0-100 range and PJND lands at ~63 in ssim2 per CID22 paper. cvvdp_log_norm (safesyn-corpus normed) shown as parallel signal — it lives in ~10-35 across the compression-product distortion regime, structurally below ssim2.

| butter_pnorm3 | V6 rule | empirical ssim2 (used as V7 target) | empirical cvvdp (norm) | Δ (ssim2 − V6) |
|---:|---:|---:|---:|---:|
| 0.3 | 90.0 | 87.58 | 40.45 | -2.42 |
| 0.8 | 75.0 | 85.57 | 37.37 | +10.57 |
| 1.5 | 63.0 | 77.69 | 32.48 | +14.69 |
| 2.5 | 45.0 | 62.41 | 23.22 | +17.41 |
| 4.0 | 25.0 | 55.14 | 17.34 | +30.14 |
| 6.0 | 10.0 | 34.64 | 17.03 | +24.64 |

## Per-codec per-band empirical medians

### zenjpeg

| butter_pnorm3 | V6 rule | ssim2 median | ssim2 n | cvvdp median (raw) | cvvdp median (norm) | cvvdp n | joint target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 90.0 | 86.53 | 13 | 9.9868 | 40.45 | 15 | 86.53 |
| 0.8 | 75.0 | 83.75 | 35 | 9.9682 | 34.93 | 43 | 83.75 |
| 1.5 | 63.0 | 79.43 | 42 | 9.9451 | 31.51 | 51 | 79.43 |
| 2.5 | 45.0 | 66.20 | 25 | 9.8359 | 24.64 | 29 | 66.20 |
| 4.0 | 25.0 | 55.14 | 9 | 9.5279 | 18.01 | 11 | 55.14 |
| 6.0 | 10.0 | 34.64 | 4 | 9.4486 | 17.03 | 5 | 34.64 |

### zenwebp

| butter_pnorm3 | V6 rule | ssim2 median | ssim2 n | cvvdp median (raw) | cvvdp median (norm) | cvvdp n | joint target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 90.0 | 87.15 | 13 | 9.9801 | 37.88 | 22 | 87.15 |
| 0.8 | 75.0 | 82.86 | 34 | 9.9626 | 33.92 | 64 | 82.86 |
| 1.5 | 63.0 | 75.38 | 43 | 9.9136 | 28.67 | 71 | 75.38 |
| 2.5 | 45.0 | 58.63 | 19 | 9.7418 | 21.80 | 37 | 58.63 |
| 4.0 | 25.0 | nan | 0 | 9.4165 | 16.68 | 3 | 16.68 |
| 6.0 | 10.0 | nan | 0 | nan | nan | 0 | nan |

### zenavif

| butter_pnorm3 | V6 rule | ssim2 median | ssim2 n | cvvdp median (raw) | cvvdp median (norm) | cvvdp n | joint target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 90.0 | 89.17 | 36 | 9.9917 | 43.33 | 54 | 89.17 |
| 0.8 | 75.0 | 87.39 | 41 | 9.9858 | 39.98 | 72 | 87.39 |
| 1.5 | 63.0 | 75.95 | 8 | 9.9597 | 33.44 | 21 | 75.95 |
| 2.5 | 45.0 | 55.21 | 10 | 9.6754 | 20.36 | 17 | 55.21 |
| 4.0 | 25.0 | 26.41 | 6 | 9.3254 | 15.77 | 6 | 26.41 |
| 6.0 | 10.0 | 19.97 | 3 | 8.9563 | 13.03 | 10 | 19.97 |

### zenjxl

| butter_pnorm3 | V6 rule | ssim2 median | ssim2 n | cvvdp median (raw) | cvvdp median (norm) | cvvdp n | joint target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 90.0 | 88.01 | 34 | 9.9868 | 40.45 | 43 | 88.01 |
| 0.8 | 75.0 | 87.80 | 29 | 9.9854 | 39.81 | 41 | 87.80 |
| 1.5 | 63.0 | 87.85 | 28 | 9.9874 | 40.77 | 32 | 87.85 |
| 2.5 | 45.0 | 87.80 | 23 | 9.9907 | 42.62 | 25 | 87.80 |
| 4.0 | 25.0 | 85.91 | 15 | 9.9826 | 38.72 | 17 | 85.91 |
| 6.0 | 10.0 | 86.28 | 4 | 9.9859 | 40.06 | 4 | 86.28 |
