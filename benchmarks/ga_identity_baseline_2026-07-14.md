# G-A / R1 sub-domain identity report

- B: `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` (native SDR path)
- BHdr: `bhdr_linear_shaped_cvvdpmix_2026-07-12.bin` (203-nit PQ re-encode → PU-linear path)
- n = 3779 SDR pairs (UPIQ images; JOD unused — no holdout burn)

## Aggregate (n=3779)
- Δ = BHdr − B: mean +0.76, median +1.95
- |Δ|: p50 10.12, p95 36.69, max 86.35
- rank agreement SROCC(B, BHdr): 0.8476
- **GATE (p95 ≤ 2, SROCC ≥ 0.99): FAIL**

## live (n=779)
- Δ = BHdr − B: mean -0.07, median +1.59
- |Δ|: p50 6.48, p95 27.17, max 80.52
- rank agreement SROCC(B, BHdr): 0.9216
- **GATE (p95 ≤ 2, SROCC ≥ 0.99): FAIL**

## tid2013 (n=3000)
- Δ = BHdr − B: mean +0.98, median +2.05
- |Δ|: p50 11.33, p95 37.66, max 86.35
- rank agreement SROCC(B, BHdr): 0.8134
- **GATE (p95 ≤ 2, SROCC ≥ 0.99): FAIL**

## Worst 10 |Δ|
- tid2013/25/i25_17_5.png: B 33.6 vs BHdr -52.7 (Δ -86.3)
- live/06/i06_01_5.png: B 10.3 vs BHdr -70.2 (Δ -80.5)
- tid2013/16/i16_24_5.png: B 25.3 vs BHdr -53.0 (Δ -78.3)
- tid2013/08/i08_17_5.png: B 32.7 vs BHdr -45.6 (Δ -78.2)
- tid2013/05/i05_17_5.png: B 32.6 vs BHdr -42.3 (Δ -74.9)
- live/01/i01_01_5.png: B 7.8 vs BHdr -67.0 (Δ -74.8)
- live/07/i07_01_5.png: B 3.1 vs BHdr -71.5 (Δ -74.5)
- tid2013/25/i25_17_3.png: B 59.2 vs BHdr -12.7 (Δ -71.9)
- tid2013/19/i19_17_5.png: B 32.3 vs BHdr -38.8 (Δ -71.1)
- live/27/i27_01_5.png: B 6.8 vs BHdr -64.0 (Δ -70.7)
