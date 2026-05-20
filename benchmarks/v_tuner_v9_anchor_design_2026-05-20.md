# V6 vs V8 vs V9 anchor target comparison (EXP-CROSS-CODEC-V9, 2026-05-20)

V9 extends the score range to a clean [0, 100] span with the JND at score=60 (clean multiple of 10) and JOD at score=30 (clean multiple of 10). V6 (ship) used 6 bands [10, 90] range with PJND at 63; V8 (falsified) used 4 bands [45, 85] range with PJND at 63.

## Per-band target table

| butter_pnorm3 | V6 rule | V8 chosen | **V9 chosen** | semantic |
|---:|---:|---:|---:|---|
| 0.05 | — | — | **100.0** | lossless / q=100 best codec |
| 0.3 | 90.0 | — | **90.0** | near-lossless |
| 0.5 | — | 85.0 | — | (intermediate) |
| 0.6 | — | — | **80.0** | visually identical |
| 0.8 | 75.0 | — | — | (V6 band, dropped in V9) |
| 1.0 | — | 75.0 | — | (intermediate) |
| 1.5 | 63.0 | — | **60.0** | **JND** (CID22 paper PJND) |
| 2.5 | 45.0 | 63.0 | **50.0** | mildly noticeable |
| 4.0 | 25.0 | 45.0 | **30.0** | **JOD** (just objectionable) |
| 6.0 | 10.0 | — | — | (V6 band, dropped in V9) |
| 7.0 | — | — | **10.0** | clearly distorted |
| 12.0 | — | — | **0.0** | worst-q floor |

## V9 design rationale

- **Score range [0, 100] is the user-facing dial.** Below 30 → broken; 30-60 → noticeable; 60-90 → good; 90-100 → near-lossless.
- **JND at score=60 (multiple of 10)** instead of V6/V8's 63. The 60 is a memorable round number; 63 was a CID22 paper convention. Output spline calibration absorbs the underlying butter-to-score mapping difference.
- **JOD at score=30 (multiple of 10).** Below 30 = definitely objectionable, by user-facing convention.
- **8 bands instead of V6's 6 / V8's 4.** Denser coverage of the perceptibility curve. Tradeoff: more anchor rows = more anchor pressure; --anchor-loss-weight reduced 1.0 → 0.5 to compensate.

## Realizability note

Butter parquets (`/mnt/v/zen/picker-training/2026-05-19/butter/<codec>.parquet`) only cover q=5..95 and don't reach butter=12.0 (max butter per codec: zenjpeg=11.4, zenwebp=5.8, zenavif=8.7, zenjxl=9.3).

V9 adapts by:

- Widening `max_distance` from V8's 0.3 → 0.4 to allow the 12.0 band to claim the closest-available high-butter row even when it's noticeably below 12.0.
- Adding explicit "worstfloor" anchor rows: every row with butter ≥ 6.0 from the q=5..95 butter parquets, target_score=0. These widen the anchor pool at the low-score end.
- Adding explicit "lossless" anchor rows: every row with butter ≤ 0.1, target_score=100. These widen the anchor pool at the high-score end. (zenjxl q=95 has butter ≈ 0.005 across the entire 1000-source corpus and dominates this pool.)

The post-network PCHIP spline calibration applied AFTER training corrects for the residual mismatch between the anchor's nominal `target_score` and the network's actual predicted distribution at each anchor butter level.

## Per-codec row counts (V9 build)

| codec | band | target_score | emitted | filtered | filter% |
|---|---:|---:|---:|---:|---:|
| zenjpeg | 0.05 | 100.0 | 479 | 521 | 52.10% |
| zenjpeg | 0.3 | 90.0 | 775 | 225 | 22.50% |
| zenjpeg | 0.6 | 80.0 | 880 | 120 | 12.00% |
| zenjpeg | 1.5 | 60.0 | 966 | 34 | 3.40% |
| zenjpeg | 2.5 | 50.0 | 988 | 12 | 1.20% |
| zenjpeg | 4.0 | 30.0 | 336 | 664 | 66.40% |
| zenjpeg | 7.0 | 10.0 | 37 | 963 | 96.30% |
| zenjpeg | 12.0 | 0.0 | 0 | 1000 | 100.00% |
| zenjpeg | worstfloor | 0.0 | 287 | 0 | 0.00% |
| zenjpeg | lossless | 100.0 | 2 | 0 | 0.00% |
| zenwebp | 0.05 | 100.0 | 150 | 850 | 85.00% |
| zenwebp | 0.3 | 90.0 | 693 | 307 | 30.70% |
| zenwebp | 0.6 | 80.0 | 861 | 139 | 13.90% |
| zenwebp | 1.5 | 60.0 | 967 | 33 | 3.30% |
| zenwebp | 2.5 | 50.0 | 967 | 33 | 3.30% |
| zenwebp | 4.0 | 30.0 | 229 | 771 | 77.10% |
| zenwebp | 7.0 | 10.0 | 0 | 1000 | 100.00% |
| zenwebp | 12.0 | 0.0 | 0 | 1000 | 100.00% |
| zenavif | 0.05 | 100.0 | 878 | 122 | 12.20% |
| zenavif | 0.3 | 90.0 | 938 | 62 | 6.20% |
| zenavif | 0.6 | 80.0 | 978 | 22 | 2.20% |
| zenavif | 1.5 | 60.0 | 1000 | 0 | 0.00% |
| zenavif | 2.5 | 50.0 | 989 | 11 | 1.10% |
| zenavif | 4.0 | 30.0 | 928 | 72 | 7.20% |
| zenavif | 7.0 | 10.0 | 244 | 756 | 75.60% |
| zenavif | 12.0 | 0.0 | 0 | 1000 | 100.00% |
| zenavif | worstfloor | 0.0 | 775 | 0 | 0.00% |
| zenavif | lossless | 100.0 | 2 | 0 | 0.00% |
| zenjxl | 0.05 | 100.0 | 1000 | 0 | 0.00% |
| zenjxl | 0.3 | 90.0 | 1000 | 0 | 0.00% |
| zenjxl | 0.6 | 80.0 | 995 | 5 | 0.50% |
| zenjxl | 1.5 | 60.0 | 1000 | 0 | 0.00% |
| zenjxl | 2.5 | 50.0 | 990 | 10 | 1.00% |
| zenjxl | 4.0 | 30.0 | 562 | 438 | 43.80% |
| zenjxl | 7.0 | 10.0 | 15 | 985 | 98.50% |
| zenjxl | 12.0 | 0.0 | 0 | 1000 | 100.00% |
| zenjxl | worstfloor | 0.0 | 97 | 0 | 0.00% |
| zenjxl | lossless | 100.0 | 1000 | 0 | 0.00% |