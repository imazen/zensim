/tmp/v07_band_analysis.py:91: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  rho = abs(stats.spearmanr(h, p_).correlation)
/tmp/v07_band_analysis.py:100: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  rho_s2 = abs(stats.spearmanr(h, s2).correlation)
/tmp/v07_band_analysis.py:101: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  rho_bu = abs(stats.spearmanr(h, bu).correlation)

## A) Per-band SROCC (synthetic) | predicted_distance vs SSIM2 score
higher = predicted ranks consistent with ground truth in that band

| band | n | V0_7 | V0_7-control |
|---|--:|--:|--:|
| ≤ 0 | 16161 | 0.9205 | 0.9157 |
| 0–25 | 30274 | 0.8267 | 0.7571 |
| 25–40 | 29540 | 0.7767 | 0.7118 |
| 40–60 | 67366 | 0.9367 | 0.8715 |
| 60–75 | 85467 | 0.9281 | 0.8363 |
| 75–90 | 82510 | 0.9582 | 0.9105 |
| ≥ 90 | 28888 | 0.7834 | 0.7490 |

## B) Per-band SROCC vs human MOS, by SSIM2 band, by dataset
v04_distance column = our predicted distance (column name kept for compat)


### Dataset: KADIK10k
| band | n | V0_7 | V0_7-control | V0_6 | V0_5 | V0_4-smooth | ref SSIM2 | ref Butter |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ≤ 0 | 697 | 0.6649 | 0.7170 | 0.6188 | 0.5998 | 0.6092 | 0.4956 | 0.0060 |
| 0–25 | 194 | 0.3902 | 0.3908 | 0.4130 | 0.4301 | 0.3281 | 0.1422 | 0.3420 |
| 25–40 | 157 | 0.3585 | 0.4291 | 0.1894 | 0.3650 | 0.0986 | 0.1977 | 0.1643 |
| 40–60 | 209 | 0.2441 | 0.2731 | 0.3021 | 0.3796 | 0.1980 | 0.1928 | 0.1122 |
| 60–75 | 119 | 0.3201 | 0.4418 | 0.3636 | 0.3759 | 0.2726 | 0.3495 | 0.2097 |
| 75–90 | 76 | 0.1474 | 0.2221 | 0.1382 | 0.1754 | 0.1830 | 0.2199 | 0.1040 |
| ≥ 90 | 48 | 0.2032 | 0.2100 | 0.2015 | nan | nan | nan | nan |

### Dataset: TID2013
| band | n | V0_7 | V0_7-control | V0_6 | V0_5 | V0_4-smooth | ref SSIM2 | ref Butter |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ≤ 0 | 509 | 0.6899 | 0.7082 | 0.6125 | 0.6774 | 0.6098 | 0.6160 | 0.0946 |
| 0–25 | 195 | 0.3579 | 0.3323 | 0.3888 | 0.3752 | 0.3302 | 0.3250 | 0.0317 |
| 25–40 | 174 | 0.2181 | 0.1953 | 0.3142 | 0.3246 | 0.2722 | 0.2073 | 0.1677 |
| 40–60 | 269 | 0.1877 | 0.1171 | 0.2826 | 0.2590 | 0.2423 | 0.1416 | 0.1096 |
| 60–75 | 220 | 0.1170 | 0.0684 | 0.1513 | 0.1221 | 0.0876 | 0.1364 | 0.3743 |
| 75–90 | 132 | 0.0768 | 0.1780 | 0.1619 | 0.1240 | 0.1393 | 0.1211 | 0.0707 |
| ≥ 90 | 1 | — | — | — | — | — | — | — |

### Dataset: CID22
| band | n | V0_7 | V0_7-control | V0_6 | V0_5 | V0_4-smooth | ref SSIM2 | ref Butter |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ≤ 0 | 0 | — | — | — | — | — | — | — |
| 0–25 | 0 | — | — | — | — | — | — | — |
| 25–40 | 4 | — | — | — | — | — | — | — |
| 40–60 | 206 | 0.4628 | 0.3194 | 0.3974 | 0.4201 | 0.4580 | 0.4102 | 0.3104 |
| 60–75 | 630 | 0.6518 | 0.6541 | 0.6385 | 0.6375 | 0.6379 | 0.6441 | 0.4175 |
| 75–90 | 635 | 0.6229 | 0.6090 | 0.6416 | 0.6432 | 0.6221 | 0.6406 | 0.3797 |
| ≥ 90 | 25 | 0.0915 | 0.0400 | 0.0908 | 0.1362 | 0.0615 | 0.0254 | 0.0185 |
