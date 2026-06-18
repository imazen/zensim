# AIC-3 CTC EPFL — zensim V0_16 vs fast-ssim2 vs dssim (n=600)

> ⚠️ **SUPERSEDED — V0_2 mislabel** (correction landed 2026-05-12 evening).
> The "V0_16" SROCC numbers here are actually V0_2 (linear) outputs — `zenmetrics batch --metric zensim`
> defaults to `ZensimProfile::latest() == PreviewV0_2`, not the V0_4 MLP path.
>
> **Canonical replacement docs**:
> - [`cycle_6_finals_2026-05-12.md`](./cycle_6_finals_2026-05-12.md) — cross-corpus aggregate + per-codec scorecard
> - [`aic_per_codec_v0_16_2026-05-12.md`](./aic_per_codec_v0_16_2026-05-12.md) — AIC-3/AIC-4 per-codec TRUE V0_16
>
> **TRUE V0_16 on AIC-3** (via `dataset_metric_baseline --v04-bake`): **0.7990** vs fast-ssim2 0.7965 (**+0.0025**).
>
> This doc is kept for historical record only; do not cite the V0_16 numbers below.

**Dataset**: `/mnt/v/dataset/aic3_ctc_epfl/`
**Source CSV**: `info_with_bitrates.csv`
**n**: 600 (10 references × 6 codecs × 10 quality levels)
**Codecs**: AVIF, HM, JPEG-1, JPEG-2000, JPEGXL, VVC
**Human label**: `score.jnd` (reconstructed JND from subjective study;
more negative = worse quality; range [-2.5, -0.25])

## Aggregate |SROCC| vs human JND (n=600)

| Metric | |SROCC| | Note |
|---|---:|---|
| **fast-ssim2-gpu** | **0.7970** | the reference baseline |
| **zensim V0_16 (current ship)** | **0.7962** | -0.0008 vs ssim2 (effective tie) |
| dssim-gpu | 0.7884 | structural baseline |
| bpp | 0.6334 | from CSV |

**V0_16 effectively ties fast-ssim2 in aggregate (-0.0008)**. This is on
a low-q-heavy corpus that V0_16 has never seen — same shipping-bar
result as on the canonical CID22 benchmark.

## Per-codec |SROCC| (V0_16 vs fast-ssim2-gpu)

| Codec | n | zensim V0_16 | fast-ssim2-gpu | dssim | Δ (V−S) | Winner |
|---|---:|---:|---:|---:|---:|---|
| AVIF       | 100 | 0.8106 | **0.8183** | 0.8039 | -0.0077 | ssim2 |
| HM         | 100 | 0.7795 | **0.7838** | 0.7712 | -0.0043 | ssim2 |
| **JPEG-1** | 100 | **0.8497** | 0.8446 | 0.8510 | +0.0051 | **V0_16** |
| JPEG-2000  | 100 | 0.7658 | **0.7671** | 0.7597 | -0.0013 | ssim2 (within noise) |
| **JPEGXL** | 100 | **0.8507** | 0.8399 | 0.8319 | +0.0107 | **V0_16** |
| VVC        | 100 | 0.7999 | **0.8063** | 0.7918 | -0.0064 | ssim2 |

V0_16 wins on JPEG-1 and JPEGXL. ssim2 wins on AVIF, HM, JPEG-2000, VVC.
The largest gaps are V0_16's JPEGXL win (+0.0107) and ssim2's AVIF win
(-0.0077). Within noise on JPEG-2000.

**Important note on dssim**: on JPEG-1, dssim actually slightly beats both
V0_16 and ssim2 (0.8510 vs 0.8497 vs 0.8446) — a 3-way tie within ~0.006.

## Comparison to AIC-4 (n=300, sibling doc)

| Corpus | n | V0_16 | reference (paper SSIMULACRA2) | Δ |
|---|---:|---:|---:|---:|
| AIC-3 | 600 | 0.7962 | 0.7970 (fast-ssim2-gpu) | -0.0008 |
| AIC-4 | 300 | 0.9107 | 0.9125 (paper SSIMULACRA2) | -0.0018 |
| CID22 (paper Table 3 ref) | 250 | ~0.89 | 0.8895 | ≈ +0.002 |

On all three low-q-heavy public corpora, V0_16 is within ±0.002 of
fast-ssim2 in aggregate — consistent with CLAUDE.md goal #1
(match-or-exceed across all bands).

## Why this matters

The cycle-7 plan flagged AIC datasets as critical for low-q (B0/B1)
human-judgment coverage that CID22 underweights. The data now shows:

- V0_16 matches fast-ssim2 within noise on both AIC-3 and AIC-4.
- V0_16 has codec-specific wins (JPEG-1, JPEGXL on AIC-3; JPEG-1,
  JPEG-XL, VVC on AIC-4).
- V0_16 has codec-specific losses (AVIF, HM, VVC on AIC-3; AVIF,
  JPEG-AI on AIC-4).
- The pattern suggests V0_16 is good at the codecs whose distortions
  most resemble JPEG's training distribution (JPEG-1, JPEG-XL where XL
  inherits JPEG's structure) and weaker on modern HEVC/AV1-derived
  codecs (AVIF, HM, VVC).

This is actionable: any cycle-7 retraining should densify the AVIF/HM/VVC
distortion sampling to close those per-codec gaps.

## Reproducibility

```bash
python3 /home/lilith/work/zen/zensim/scripts/v_next/export_aic3_to_parquet.py \
    --metrics-tsv /tmp/aic3_dssim/scored_dssim.tsv \
    --metrics-tsv /tmp/aic3_dssim/scored_zensim.tsv \
    --metrics-tsv /tmp/aic3_dssim/scored_ssim2.tsv \
    --out /tmp/aic3_dssim/aic3_ctc_epfl.parquet
```

## Status

- AIC-3 dssim-gpu: ✅ DONE
- AIC-3 ssim2-gpu: ✅ DONE
- AIC-3 zensim CPU: ✅ DONE
- AIC-3 butteraugli-gpu: in flight (~400/600 at tick close)
- AIC-4 dssim/ssim2/butter-gpu: queued (single GPU sequence)
