# UPIQ validation panel — 2026-06-01

n = 4159 conditions joined (380 HDR, 3779 SDR). Stats via `zensim_validate::panel` (the `panel` binary); no stat math reimplemented. JOD truth = `upiq_subjective_scores.csv`.

SROCC (rank) per stratum; PLCC after 4-param logistic; PWRC + Z-RMSE from the full panel. `Δpub` = ALL-SROCC minus published full-set baseline (drift check).

| metric | SROCC all | SROCC HDR | SROCC SDR | PLCC all | PWRC all | Z-RMSE all | Δpub |
|---|--:|--:|--:|--:|--:|--:|--:|
| PU_PieApp | 0.9452 | 0.8748 | 0.9497 | 0.9580 | 0.9955 | 0.2869 | 0.0002 |
| PU_FSIM | 0.8413 | 0.7185 | 0.8553 | 0.9009 | 0.9850 | 0.4341 | 0.0003 |
| HDRVDP2_2 | 0.8149 | 0.8117 | 0.8201 | 0.8507 | 0.9802 | 0.5256 | -0.0001 |
| PU_SSIM | 0.6960 | 0.7395 | 0.6948 | 0.7138 | 0.9485 | 0.7004 | 0.0000 |
| PU_PSNR | 0.6671 | 0.5485 | 0.6818 | 0.6719 | 0.9401 | 0.7407 | 0.0071 |
| HDRVQM | 0.7741 | 0.8772 | 0.7741 | 0.8254 | 0.9751 | 0.5646 | — |
| FSIM | 0.8160 | 0.4568 | 0.8552 | 0.8844 | 0.9820 | 0.4667 | — |
| PSNR | 0.5332 | 0.4606 | 0.7020 | 0.5639 | 0.8588 | 0.8259 | — |

**Bar for zensim-HDR:** clear PU-SSIM / HDR-VDP-2 decisively, approach PU-PieAPP (SROCC 0.945). Watch the HDR column — the highlight band is where SDR-tuned metrics collapse.
