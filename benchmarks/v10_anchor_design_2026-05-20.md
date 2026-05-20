# V9 vs V10 anchor target comparison (EXP-CROSS-CODEC-V10, 2026-05-20)

V10 reallocates the zensim score-space per user direction 2026-05-20. Lossless = 100. JND = 80 (was 60 in V9). JOD = 50 (was 30 in V9). Borderline (q=0 worst codec) = 0. Below 0 = pathological / unreasonable (no clamp; linear extrapolation).

## Per-band target table

| butter_pnorm3 | V9 target | **V10 target** | semantic |
|---:|---:|---:|---|
| 0.05 | 100.0 | **100.0** | lossless / q=95-100 best codec |
| 0.3 | 90.0 | **95.0** | near-lossless |
| 0.6 | 80.0 | **90.0** | visually identical |
| 1.5 | 60.0 | **80.0** | **JND** (PJND threshold) |
| 2.5 | 50.0 | **65.0** | mildly noticeable |
| 4.0 | 30.0 | **50.0** | **JOD** (just objectionable) |
| 5.5 | — | **35.0** | 3x-DPI resize-out — usable at scale |
| 7.0 | 10.0 | **20.0** | clear artifacts even at scale |
| 9.0 | — | **10.0** | very degraded |
| 12.0 | 0.0 | **0.0** | worst-q floor / borderline unacceptable |

## V10 design rationale

- **Lossless = 100, JND = 80, JOD = 50, q=0-floor = 0.** Wider perceptibility band (50 score units between JOD and JND) gives the user-facing dial more resolution where compression product decisions live.
- **Below 0 = pathological.** V10 removes the [0, 100] hard clamp in apply_mlp_scoring AND in the bake-aware tools' default post mode. The PCHIP spline extrapolates linearly below xs[0] / above xs[-1] using the endpoint slope, so the worst codec at q=0 (butter >> 12) maps to a negative score. This signals 'unreasonable distortion' rather than collapsing to a tie block.
- **11 bands instead of V9's 8.** Denser sampling of the perceptibility curve (5.5 and 9.0 added). Tradeoff: max_distance widened to 0.5 so each per-band anchor pool stays similar in size.

## Realizability note

Butter parquets (`/mnt/v/zen/picker-training/2026-05-19/butter/<codec>.parquet`) only cover q=5..95 and don't reach butter=12.0 (max butter per codec: zenjpeg=11.4, zenwebp=5.8, zenavif=8.7, zenjxl=9.3).

V10 adapts by:

- Widening `max_distance` from V9's 0.4 -> 0.5 to allow the 12.0 band to claim the closest-available high-butter row.
- Adding explicit "worstfloor" anchor rows: every row with butter >= 6.0 from the q=5..95 butter parquets, target_score=0.
- Adding explicit "lossless" anchor rows: every row with butter <= 0.1, target_score=100.

The post-network PCHIP spline calibration applied AFTER training, with unclamped linear extrapolation, lets the spline output flow through to the final score uninhibited. Reasonable codec output lands in [0, 100]; pathological output extrapolates negative.

## Per-codec row counts (V10 build)

| codec | band | target_score | emitted | filtered | filter% |
|---|---:|---:|---:|---:|---:|
| zenjpeg | 0.05 | 100.0 | 662 | 338 | 33.80% |
| zenjpeg | 0.3 | 95.0 | 839 | 161 | 16.10% |
| zenjpeg | 0.6 | 90.0 | 907 | 93 | 9.30% |
| zenjpeg | 1.5 | 80.0 | 968 | 32 | 3.20% |
| zenjpeg | 2.5 | 65.0 | 992 | 8 | 0.80% |
| zenjpeg | 4.0 | 50.0 | 374 | 626 | 62.60% |
| zenjpeg | 5.5 | 35.0 | 131 | 869 | 86.90% |
| zenjpeg | 7.0 | 20.0 | 44 | 956 | 95.60% |
| zenjpeg | 9.0 | 10.0 | 20 | 980 | 98.00% |
| zenjpeg | 12.0 | 0.0 | 0 | 1000 | 100.00% |
| zenjpeg | worstfloor | 0.0 | 287 | 0 | 0.00% |
| zenjpeg | lossless | 100.0 | 2 | 0 | 0.00% |
| zenwebp | 0.05 | 100.0 | 402 | 598 | 59.80% |
| zenwebp | 0.3 | 95.0 | 788 | 212 | 21.20% |
| zenwebp | 0.6 | 90.0 | 888 | 112 | 11.20% |
| zenwebp | 1.5 | 80.0 | 969 | 31 | 3.10% |
| zenwebp | 2.5 | 65.0 | 976 | 24 | 2.40% |
| zenwebp | 4.0 | 50.0 | 321 | 679 | 67.90% |
| zenwebp | 5.5 | 35.0 | 5 | 995 | 99.50% |
| zenwebp | 7.0 | 20.0 | 0 | 1000 | 100.00% |
| zenwebp | 9.0 | 10.0 | 0 | 1000 | 100.00% |
| zenwebp | 12.0 | 0.0 | 0 | 1000 | 100.00% |
| zenavif | 0.05 | 100.0 | 899 | 101 | 10.10% |
| zenavif | 0.3 | 95.0 | 975 | 25 | 2.50% |
| zenavif | 0.6 | 90.0 | 997 | 3 | 0.30% |
| zenavif | 1.5 | 80.0 | 1000 | 0 | 0.00% |
| zenavif | 2.5 | 65.0 | 992 | 8 | 0.80% |
| zenavif | 4.0 | 50.0 | 965 | 35 | 3.50% |
| zenavif | 5.5 | 35.0 | 830 | 170 | 17.00% |
| zenavif | 7.0 | 20.0 | 287 | 713 | 71.30% |
| zenavif | 9.0 | 10.0 | 1 | 999 | 99.90% |
| zenavif | 12.0 | 0.0 | 0 | 1000 | 100.00% |
| zenavif | worstfloor | 0.0 | 775 | 0 | 0.00% |
| zenavif | lossless | 100.0 | 2 | 0 | 0.00% |
| zenjxl | 0.05 | 100.0 | 1000 | 0 | 0.00% |
| zenjxl | 0.3 | 95.0 | 1000 | 0 | 0.00% |
| zenjxl | 0.6 | 90.0 | 998 | 2 | 0.20% |
| zenjxl | 1.5 | 80.0 | 1000 | 0 | 0.00% |
| zenjxl | 2.5 | 65.0 | 996 | 4 | 0.40% |
| zenjxl | 4.0 | 50.0 | 612 | 388 | 38.80% |
| zenjxl | 5.5 | 35.0 | 77 | 923 | 92.30% |
| zenjxl | 7.0 | 20.0 | 18 | 982 | 98.20% |
| zenjxl | 9.0 | 10.0 | 18 | 982 | 98.20% |
| zenjxl | 12.0 | 0.0 | 0 | 1000 | 100.00% |
| zenjxl | worstfloor | 0.0 | 97 | 0 | 0.00% |
| zenjxl | lossless | 100.0 | 1000 | 0 | 0.00% |