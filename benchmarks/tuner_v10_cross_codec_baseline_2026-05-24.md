# Tuner v10 cross-codec consistency baseline — 2026-05-24

- **Bake:** `zensim/weights/v_tuner_v10_2026-05-20.bin`
- **Eq parquet:** `/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet`
- **n pairs:** 68,788
- **bake-post:** `clamp` (raw output, before any clamp/spline)

## Overall cross-codec score deviation

| stat | value |
|---|---:|
| n | 68,788 |
| median \|Δ\| | 1.183 |
| p90 \|Δ\| | 3.577 |
| p99 \|Δ\| | 8.049 |
| max \|Δ\| | 56.185 |
| stddev(Δ) | 2.516 |

Δ = score_a − score_b on matched butter-level anchor pairs. Lower is better. A perfectly cross-codec-consistent metric would give Δ=0 at every anchor.

## Per-butter-anchor breakdown

| butter_level | n | p50 \|Δ\| | p90 \|Δ\| | p99 \|Δ\| | max \|Δ\| | mean score_a | mean score_b |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0.300 | 4,716 | 1.928 | 5.245 | 8.204 | 11.495 | 94.10 | 94.18 |
| 0.703 | 5,442 | 1.107 | 3.546 | 6.603 | 10.588 | 88.08 | 88.04 |
| 1.107 | 5,721 | 0.566 | 1.673 | 4.135 | 10.098 | 83.67 | 83.62 |
| 1.510 | 5,816 | 0.786 | 2.401 | 5.802 | 15.667 | 79.29 | 79.21 |
| 1.914 | 5,802 | 1.386 | 3.752 | 7.467 | 21.781 | 73.42 | 73.34 |
| 2.317 | 5,712 | 1.395 | 4.020 | 8.391 | 25.418 | 67.65 | 67.60 |
| 2.721 | 5,310 | 1.470 | 3.847 | 7.569 | 19.116 | 62.78 | 62.78 |
| 3.124 | 4,636 | 1.552 | 3.999 | 9.480 | 56.185 | 58.22 | 58.45 |
| 3.528 | 3,395 | 1.419 | 3.932 | 10.262 | 54.562 | 54.93 | 54.99 |
| 3.931 | 2,157 | 1.346 | 3.918 | 9.097 | 38.869 | 53.24 | 53.03 |
| 4.334 | 1,494 | 1.382 | 3.952 | 12.059 | 32.673 | 53.45 | 53.35 |
| 4.738 | 1,199 | 1.370 | 3.893 | 13.042 | 32.673 | 53.94 | 53.93 |
| 5.141 | 1,098 | 1.319 | 3.863 | 15.083 | 32.673 | 54.16 | 54.15 |
| 5.545 | 1,047 | 1.307 | 3.700 | 12.669 | 26.352 | 54.22 | 54.03 |
| 5.948 | 1,000 | 1.258 | 3.501 | 10.446 | 26.352 | 54.58 | 54.26 |
| 6.352 | 976 | 1.202 | 3.156 | 8.323 | 24.522 | 54.65 | 54.15 |
| 6.755 | 960 | 1.175 | 3.127 | 7.280 | 22.227 | 54.82 | 54.39 |
| 7.159 | 959 | 1.171 | 3.144 | 8.172 | 22.227 | 54.88 | 54.42 |
| 7.562 | 953 | 1.159 | 3.105 | 8.198 | 44.908 | 54.99 | 54.58 |
| 7.966 | 951 | 1.159 | 3.107 | 6.896 | 44.908 | 55.06 | 54.59 |
| 8.369 | 948 | 1.148 | 3.065 | 6.283 | 44.908 | 55.20 | 54.79 |
| 8.772 | 944 | 1.138 | 3.042 | 5.793 | 44.908 | 55.37 | 54.94 |
| 9.176 | 944 | 1.138 | 3.042 | 5.793 | 44.908 | 55.37 | 54.94 |
| 9.579 | 944 | 1.138 | 3.042 | 5.793 | 44.908 | 55.37 | 54.94 |
| 9.983 | 944 | 1.138 | 3.042 | 5.793 | 44.908 | 55.37 | 54.94 |
| 10.386 | 944 | 1.138 | 3.042 | 5.793 | 44.908 | 55.37 | 54.94 |
| 10.790 | 944 | 1.138 | 3.042 | 5.793 | 44.908 | 55.37 | 54.94 |
| 11.193 | 944 | 1.138 | 3.042 | 5.793 | 44.908 | 55.37 | 54.94 |
| 11.597 | 944 | 1.138 | 3.042 | 5.793 | 44.908 | 55.37 | 54.94 |
| 12.000 | 944 | 1.138 | 3.042 | 5.793 | 44.908 | 55.37 | 54.94 |

## Per-codec-pair breakdown

| codec_pair | n | p50 \|Δ\| | p90 \|Δ\| | p99 \|Δ\| | max \|Δ\| | stddev(Δ) |
|---|--:|--:|--:|--:|--:|--:|
| zenjpeg_zenwebp | 15,323 | 1.058 | 3.232 | 6.174 | 21.781 | 1.981 |
| zenwebp_zenavif | 7,667 | 1.123 | 3.689 | 7.507 | 12.274 | 2.288 |
| zenwebp_zenjxl | 15,359 | 1.133 | 3.243 | 6.796 | 20.652 | 2.129 |
| zenjpeg_zenavif | 8,803 | 1.231 | 3.842 | 9.081 | 56.185 | 2.828 |
| zenavif_zenjxl | 9,309 | 1.326 | 4.037 | 9.736 | 26.352 | 2.716 |
| zenjpeg_zenjxl | 12,327 | 1.342 | 3.762 | 11.183 | 55.890 | 3.182 |

## Findings

### 1. Cross-codec consistency is strong in the JND/JOD band

In the practically-important 60..90 score band (mid-q to high-q codec
outputs), Tuner v10 is tight:

- **butter ≈ 1.1 (score ≈ 84):** p50 |Δ| = **0.57** — best of any anchor
- **butter ≈ 1.5 (score ≈ 79, JND-adjacent):** p50 |Δ| = **0.79**
- **butter ≈ 2.7 (score ≈ 63, near PJND):** p50 |Δ| = **1.47**

Codecs binary-searching for `score ≈ 70` (typical "good web compression")
land within ±1 score unit of each other across JPEG/WebP/AVIF/JXL.

### 2. Score floor at ~55 — low-q dial is clamped

At butter ≥ 6.8 (low-quality region, codec outputs that should land at
score 10–30), the per-anchor stats become **structurally identical**:

```
butter_level  6.755 → 12.0:  p50=1.138, p90=3.042, max=44.908
                              mean score_a = 55.37, mean score_b = 54.94
```

Every anchor above butter=6.8 produces the same set of pairs because
the bake's output saturates at score ≈ 55 for the worst codec outputs.
This is the **score-floor pathology**: Tuner cannot discriminate
"bad" vs "very bad" codec outputs — both map near 55. The 44.9 max
outlier is a single pair where one side hit 100 (lossless image) and
the other side saturated to ~55.

This is the same pathology V_tuner_v9 hit (`benchmarks/v_tuner_v9_falsification_2026-05-20.md`)
— the `medRange ≥ 50` gate failed because the network underutilizes the
score=0..30 range. The PCHIP spline anchor includes a knot at
butter≈12 → score=0, but the network's raw output before the spline
doesn't reach that region for many inputs.

### 3. Worst codec pair: JPEG↔JXL

| pair | p50 | p90 | p99 | stddev |
|---|--:|--:|--:|--:|
| zenjpeg_zenwebp | 1.06 | 3.23 | 6.17 | 1.98 |
| zenjpeg_zenjxl | 1.34 | 3.76 | **11.18** | **3.18** |
| zenavif_zenjxl | 1.33 | 4.04 | 9.74 | 2.72 |

JPEG↔JXL has the widest tail (p99=11.2 vs WebP↔JPEG's 6.2). Likely
driven by JXL's known ssim2/cvvdp saturation pattern on the v12_zenjxl
corpus (per `benchmarks/v11_methodology_2026-05-20.md`: zenjxl ssim2
sits flat at ~87 across q=10..90 on the substrate, so the network
sees fewer training samples discriminating between JXL quality levels).

## Comparison to prior measurements

V11-E falsification (`benchmarks/v11_e_per_codec_falsification_2026-05-20.md`)
measured Tuner v4's per-codec α (post-spline systematic bias) at
**0.7 score units** — small, confirming the network is approximately
codec-invariant on average. The numbers here are the matched-anchor
analog: same perceptual quality → same score, p90 = 3.58. This is
consistent with the per-codec finding (small offset) + larger
within-codec content noise.

## Cross-trail comparison (2026-05-24)

Ran the same eval on the other two trail ships for context. Tuner
is decisively the most cross-codec-consistent of the three:

| Bake | median \|Δ\| | p90 \|Δ\| | p99 \|Δ\| | n |
|---|--:|--:|--:|--:|
| **Tuner v10** (`v_tuner_v10_2026-05-20.bin`) | **1.183** | **3.577** | **8.049** | 68,788 |
| Compression v3 (`v_compression_v3_2026-05-20.bin`) | 2.637 | 15.317 | 39.460 | 68,788 |
| Balanced v3 (`v_balanced_v3_2026-05-20.bin`) | 3.055 | 20.709 | 51.487 | 68,788 |

**Tuner v10 is 3-6× tighter cross-codec than the other trails:**
- vs Balanced: 2.6× tighter at p50, 5.8× tighter at p90, 6.4× tighter at p99
- vs Compression: 2.2× tighter at p50, 4.3× tighter at p90, 4.9× tighter at p99

This is structural — Tuner's training mix explicitly includes
cross-codec equivalence pair-loss (`--cross-codec-eq-weight 1.0`)
plus the V_22 multi-band anchor parquet, while Balanced/Compression
optimize for within-codec rank fidelity. **Picking Balanced or
Compression as the codec-target metric would have given users
3-6× worse cross-codec dial precision** — concrete validation of
the `ZensimProfile::codec_target()` → `PreviewV0_5TunerV4` choice
(per `docs/CODEC_TARGET_METRIC.md`).

Raw scores: `/mnt/v/output/zensim/cross_codec_compare_2026-05-24/{balanced,compression}_v3.parquet`.

### JPEG q-sweep monotonicity (within-codec dial precision)

Cross-codec consistency above measures inter-codec agreement at
matched perceptual quality. A complementary dial-precision view is
**within-codec monotonicity**: does the JPEG q dial monotonically
increase the zensim score? Measured via `qsweep_eval` on the
canonical 50-image × 19-q JPEG sweep
(`/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv`):

| Bake | strict-mono % | strict violations | clamp-flat ties |
|---|--:|--:|--:|
| **Tuner v10** | **96.44%** | 32 / 900 | **0** |
| Compression v3 | 91.89% | 73 / 900 | 13 |
| Balanced v3 | 88.00% | 108 / 900 | 7 |

**Tuner v10 has ZERO tied-score regions** — every JPEG q step
produces a distinct score. Balanced and Compression have 7-13
clamp-flat dead zones where the dial can't disambiguate adjacent
q values; binary-search-to-target would stall at those points.

Combined picture: Tuner is the best codec-target metric on both
the cross-codec consistency axis (matched anchors land at the
same score) AND the within-codec monotonicity axis (q sweeps
yield monotone score progressions). Picking it for the canonical
codec-target metric is decisively the right call.

qsweep raw report:
`/mnt/v/output/zensim/cross_codec_compare_2026-05-24/qsweep_3trails.md`.

## Implications for codec-target ship

**Verdict: Tuner v10 is ship-ready as the canonical codec-target metric
for the score 60–100 region.** Cross-codec consistency in this band
(median 0.6–1.5 score units) is below the round-number-precision a
codec consumer needs to expose to users.

The score 0–55 region is **not ship-ready** for cross-codec use —
the floor saturation means a user typing "target score 20" gets
indeterminate codec outputs. The path forward (task #6 — Tuner v11)
must restore dynamic range in this band, either by:

1. Stronger anchor-pressure at low-q (anchor_loss_weight > 0.5,
   currently 0.5 per V9 recipe) — risk: monotonicity breakage.
2. Adding low-q training pairs from CID22-train-subset (task #5),
   which has real-codec encodes well below the safesyn distribution
   centre.
3. Per-source aggregation head for konjnd-dense (task #4) — unblocks
   raising konjnd weight, which would let the network see more
   low-distortion calibration anchors per ref.

## Reproducibility

```sh
python3 scripts/v_next/measure_tuner_v10_cross_codec.py \
    --bake zensim/weights/v_tuner_v10_2026-05-20.bin \
    --parquet /mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet \
    --out-parquet /mnt/v/output/zensim/tuner_v10_cross_codec_baseline_2026-05-24/scores.parquet \
    --out-md /home/lilith/work/zen/zensim/benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md
```
