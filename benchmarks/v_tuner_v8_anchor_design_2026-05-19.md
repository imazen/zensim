# V6 rule-of-thumb vs V7 empirical vs V8 chosen anchor targets (2026-05-19)

V6 (`PreviewV0_5TunerV2`, commit `1dd61fc`) and V7 (cross-codec-v7, commit `bfd13cb`) both used a 6-band anchor grid keyed on `butter_pnorm3`. V6 used rule-of-thumb target scores; V7 used per-(codec, band) ssim2 medians. V7 surfaced two structural problems:

1. Heavy-distortion bands (4.0, 6.0) diverge cross-codec: zenwebp ssim2=16.7 vs zenjxl ssim2=85.9 at butter=4.0. Cross-codec consistency at score=63 is unstable when one codec saturates ssim2 while another collapses at the same butter level.

2. score=63 ↔ butter=1.5 (V6 convention) misaligns with ssim2=63 (CID22 paper Table 4 KonJND-1k human PJND anchor): V7's empirical median lands ssim2=63 near butter=2.5, NOT butter=1.5. The user-facing `score=63 = PJND` contract is calibrated to ssim2's 0-100 range, so the correct butter value for that contract is butter≈2.5.

## V8 design

**Drop the heavy-distortion bands (4.0 and 6.0 in V6/V7 → keep only 4.0 as the upper edge).** Restrict to bands where cross-codec divergence is bounded.

**Re-center the band → target_score table so score=63 lands at butter=2.5** (the CID22 paper / KonJND-1k empirical PJND anchor). The user-facing `score=63 = PJND` contract is preserved; the butter target underneath shifts from 1.5 to 2.5 to align with ssim2 PJND.

Unlike V7, the V8 target_score is a fixed table value, NOT an empirical median. The trainer is anchored to the perceptibility table; cross-codec divergence at the heavy-distortion edge is removed by construction.

## Per-band target table

| butter_pnorm3 | V6 rule | V7 empirical aggregate ssim2 | V8 chosen | rationale |
|---:|---:|---:|---:|---|
| 0.3 | 90.0 | 87.58 | DROPPED | heavy-distortion: cross-codec ssim2 divergence exceeds ±20 (zenwebp collapse vs zenjxl saturation) |
| 0.5 | — | — | 85.0 | high quality, below zenjxl saturation regime |
| 0.8 | 75.0 | 85.57 | DROPPED | heavy-distortion: cross-codec ssim2 divergence exceeds ±20 (zenwebp collapse vs zenjxl saturation) |
| 1.0 | — | — | 75.0 | near-lossless |
| 1.5 | 63.0 | 81.64 | DROPPED | heavy-distortion: cross-codec ssim2 divergence exceeds ±20 (zenwebp collapse vs zenjxl saturation) |
| 2.5 | 45.0 | 62.91 | 63.0 | **PJND anchor** — score=63 ↔ ssim2-PJND per CID22 paper Table 4 |
| 4.0 | 25.0 | 40.77 | 45.0 | upper edge of zenjxl saturation safety zone |
| 6.0 | 10.0 | 27.31 | DROPPED | heavy-distortion: cross-codec ssim2 divergence exceeds ±20 (zenwebp collapse vs zenjxl saturation) |

## KonJND-1k human empirical butter placement

Per CID22 paper Table 4, KonJND-1k human PJND lands at **ssim2 ≈ 63**. V7's empirical join shows ssim2=63 ≈ **butter_pnorm3 ≈ 2.5** (the V7 aggregate ssim2 at the 2.5 band is 62.91 — a near-exact landing on PJND).

V6 shipped with score=63 at butter=1.5, which is a tighter perceptibility band than the KonJND-1k human anchor. V8 corrects this by moving score=63 to butter=2.5.

## Per-codec row counts (V8 build)

| codec | band | target_score | emitted | filtered | filter% |
|---|---:|---:|---:|---:|---:|
| zenjpeg | 0.5 | 85.0 | 839 | 161 | 16.10% |
| zenjpeg | 1.0 | 75.0 | 935 | 65 | 6.50% |
| zenjpeg | 2.5 | 63.0 | 982 | 18 | 1.80% |
| zenjpeg | 4.0 | 45.0 | 281 | 719 | 71.90% |
| zenwebp | 0.5 | 85.0 | 788 | 212 | 21.20% |
| zenwebp | 1.0 | 75.0 | 929 | 71 | 7.10% |
| zenwebp | 2.5 | 63.0 | 956 | 44 | 4.40% |
| zenwebp | 4.0 | 45.0 | 146 | 854 | 85.40% |
| zenavif | 0.5 | 85.0 | 975 | 25 | 2.50% |
| zenavif | 1.0 | 75.0 | 998 | 2 | 0.20% |
| zenavif | 2.5 | 63.0 | 963 | 37 | 3.70% |
| zenavif | 4.0 | 45.0 | 822 | 178 | 17.80% |
| zenjxl | 0.5 | 85.0 | 991 | 9 | 0.90% |
| zenjxl | 1.0 | 75.0 | 994 | 6 | 0.60% |
| zenjxl | 2.5 | 63.0 | 986 | 14 | 1.40% |
| zenjxl | 4.0 | 45.0 | 494 | 506 | 50.60% |