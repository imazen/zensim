
## Synthetic q-sweeps: predicted_distance vs gpu_ssim2 ground truth

|SROCC| within each SSIM2 band (lower = worse local ranking).
MAE = post-calibration absolute error in SSIM2 units within the band.
Higher MAE = the metric's slope isn't faithful inside that band.

### |SROCC| vs ssim2_score within each band

| metric \ band | ≤ 0 | 0–25 | 25–40 | 40–60 | 60–75 | 75–90 | ≥ 90 |
|---|--:|--:|--:|--:|--:|--:|--:|
| V0_4-smooth | 0.9327 | 0.8334 | 0.8379 | 0.9412 | 0.9500 | 0.9781 | 0.8586 |
| V0_5 | 0.9751 | 0.9151 | 0.8705 | 0.9461 | 0.9468 | 0.9767 | 0.8671 |
| V0_4-smooth-konjnd-train | 0.8710 | 0.7020 | 0.7474 | 0.8854 | 0.8973 | 0.9116 | 0.7348 |
| V0_6 dct_hf | 0.9632 | 0.9074 | 0.8972 | 0.9616 | 0.9698 | 0.9841 | 0.8631 |

### Pair count per band (constant across metrics):

| band | ≤ 0 | 0–25 | 25–40 | 40–60 | 60–75 | 75–90 | ≥ 90 |
|---|--:|--:|--:|--:|--:|--:|--:|
| pairs | 13183 | 16239 | 17016 | 38020 | 45686 | 60964 | 26981 |
| % of training | 6.0% | 7.4% | 7.8% | 17.4% | 20.9% | 28.0% | 12.4% |

### Per-band-calibrated MAE (in SSIM2 units; lower = more faithful slope)

| metric \ band | ≤ 0 | 0–25 | 25–40 | 40–60 | 60–75 | 75–90 | ≥ 90 |
|---|--:|--:|--:|--:|--:|--:|--:|
| V0_4-smooth | 7.54 | 3.95 | 2.27 | 1.70 | 1.13 | 0.77 | 0.95 |
| V0_5 | 5.13 | 2.41 | 1.97 | 1.77 | 1.16 | 0.73 | 0.94 |
| V0_4-smooth-konjnd-train | 8.63 | 5.94 | 3.16 | 2.50 | 1.76 | 1.59 | 1.41 |
| V0_6 dct_hf | 6.26 | 2.58 | 1.65 | 1.39 | 0.86 | 0.69 | 0.92 |

## Human-MOS holdouts: predicted vs human, bucketed by SSIM2 ground truth

|SROCC| of predicted_distance vs human_score within each SSIM2 band.


### Per-band |SROCC| vs human_score (averaged across KADID/TID/CID22)

| metric \ band | ≤ 0 | 0–25 | 25–40 | 40–60 | 60–75 | 75–90 | ≥ 90 |
|---|--:|--:|--:|--:|--:|--:|--:|
| V0_4-smooth | 0.590 | 0.333 | 0.125 | 0.004 | 0.466 | 0.566 | 0.337 |
| V0_5 | 0.605 | 0.399 | 0.292 | 0.063 | 0.508 | 0.570 | 0.331 |
| V0_4-smooth-konjnd-train | 0.331 | 0.149 | 0.026 | 0.020 | 0.393 | 0.320 | 0.311 |
| V0_6 dct_hf | 0.523 | 0.396 | 0.207 | 0.031 | 0.505 | 0.593 | 0.194 |

References (same image pairs):
| ref SSIMULACRA 2 | 0.530 | 0.203 | 0.157 | 0.012 | 0.463 | 0.479 | 0.362 |
| ref Butteraugli 3-norm | 0.075 | 0.193 | 0.031 | 0.107 | 0.214 | 0.354 | 0.366 |

Holdout pair count per band:
| band pairs | 1206 | 389 | 335 | 684 | 969 | 843 | 74 |
