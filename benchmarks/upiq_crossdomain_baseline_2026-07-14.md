# UPIQ cross-domain dial-alignment baseline

- B bake: `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` | BHdr bake: `bhdr_linear_shaped_cvvdpmix_2026-07-12.bin`
- SDR: n=3779 (u8-shell extraction) | HDR: n=380 (PU-linear extraction)

## Rank (|SROCC| dial vs JOD)
- SDR pooled: 0.8262
- HDR pooled: 0.7536
- SDR/live: 0.8945 (n=779)  [CLEAN leg]
- SDR/tid2013: 0.7804 (n=3000)  [training-overlap: guard-grade]
- HDR/narwaria: 0.7834 (n=140)
- HDR/korshunov: 0.9175 (n=240)

## Seam (residual vs pooled isotonic dial=g(JOD), dial points)
- SDR: mean +0.79, median -0.86, p95|.| 31.75
- HDR: mean -7.85, median -10.27, p95|.| 34.81
- **SEAM (mean_HDR − mean_SDR): -8.64 dial points**

## Equal-JOD band means (dial per domain; alignment = rows match)
| JOD band | SDR mean dial (n) | HDR mean dial (n) | Δ (HDR−SDR) |
|---|---|---|---|
| [-3,-2) | 18.3 (689) | 6.0 (42) | -12.3 |
| [-2,-1) | 32.5 (853) | 19.7 (154) | -12.8 |
| [-1,-0.5) | 50.5 (572) | 37.5 (91) | -13.1 |
| [-0.5,0.25) | 60.6 (790) | 64.2 (93) | +3.7 |
