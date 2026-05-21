# Gate A — castleCSF Mode A vs baseline (V_22 MLP)

Generated: 2026-05-21T17:49:00.054474Z

Methodology: V_22-mix-LARGE MLP weights applied to features extracted with `--acumen-mode-a` (HF band-energy features scaled by per-(scale, channel) castleCSF Mode A weights at ppd=56, peak=100 cd/m², ambient=5 cd/m²) vs same MLP applied to legacy unmodulated features.

| Corpus | Variant | n | SROCC | PLCC | KROCC | Z-RMSE | Δ-SROCC |
|---|---|--:|--:|--:|--:|--:|--:|
| kadid | baseline | 10125 | 0.8192 | 0.7679 | 0.6228 | 0.6406 | |
| kadid | acumen | 10125 | 0.8107 | 0.7593 | 0.6141 | 0.6507 | **-0.0085 ↓** |
| tid | baseline | 2950 | 0.8434 | 0.8554 | 0.6670 | 0.5180 | |
| tid | acumen | 2950 | 0.8417 | 0.8528 | 0.6650 | 0.5223 | **-0.0017 ↓** |
| aic3 | baseline | 600 | 0.7961 | 0.7664 | 0.6273 | 0.6424 | |
| aic3 | acumen | 600 | 0.7896 | 0.7608 | 0.6202 | 0.6490 | **-0.0066 ↓** |
| cid22 | baseline | 4292 | 0.8675 | 0.8594 | 0.6784 | 0.5113 | |
| cid22 | acumen | 4292 | 0.8599 | 0.8528 | 0.6693 | 0.5223 | **-0.0076 ↓** |

## Summary

- **acumen wins**: 0 of 4 corpora
- **acumen losses**: 4
- **ties**: 0

  - kadid: Δ-SROCC = -0.0085
  - tid: Δ-SROCC = -0.0017
  - aic3: Δ-SROCC = -0.0066
  - cid22: Δ-SROCC = -0.0076
