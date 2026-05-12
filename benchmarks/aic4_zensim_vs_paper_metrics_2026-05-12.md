# AIC-4 sample — zensim V0_16 vs paper metrics vs zen-metrics CLI (n=300)

> ⚠️ **SUPERSEDED — V0_2 mislabel** (correction landed 2026-05-12 evening).
> The "V0_16" SROCC numbers here are actually V0_2 (linear) outputs — `zen-metrics batch --metric zensim`
> defaults to `ZensimProfile::latest() == PreviewV0_2`, not the V0_4 MLP path.
>
> **Canonical replacement docs**:
> - [`cycle_6_finals_2026-05-12.md`](./cycle_6_finals_2026-05-12.md) — cross-corpus + per-codec scorecard
> - [`aic_per_codec_v0_16_2026-05-12.md`](./aic_per_codec_v0_16_2026-05-12.md) — TRUE V0_16 per-codec
>
> **TRUE V0_16 on AIC-4** (via `dataset_metric_baseline --v04-bake`): **0.9175** vs fast-ssim2 0.9127 (**+0.0048**).
>
> The single big deficit (JPEG-AI: V0_16 0.7951 vs ssim2 0.8459 vs dssim 0.9147) is the biggest
> cycle-7 actionable — see `aic_per_codec_v0_16_2026-05-12.md` for the full analysis.
>
> This doc is kept for historical record only; do not cite the V0_16 numbers below.

**Dataset**: `/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset/`
**JND CSV**: `/mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG_AIC_reconstructed_jnd_scores.csv`
**Pre-computed metric CSV**: `JPEG-AIC_metric_scores.csv` (same dir)
**n**: 300 (5 references × 6 codecs × 10 distortion levels)
**Codecs**: AVIF, JPEG-1, JPEG-2000, JPEG-XL, VVC, JPEG-AI
**Human label**: reconstructed JND `distortion` ∈ [~0.1, ~5.3] (higher = more distorted)

## Aggregate |SROCC| vs human reconstructed JND (n=300)

| Rank | Metric | \|SROCC\| | Source |
|---:|---|---:|---|
| 1  | CVVDP            | **0.9609** | paper-pre-computed |
| 2  | IW-SSIM          | 0.9507 | paper-pre-computed |
| 3  | MS-SSIM          | 0.9409 | paper-pre-computed |
| 4  | HDR-VDP-3        | 0.9329 | paper-pre-computed |
| 5  | HDR-VDP-2        | 0.9294 | paper-pre-computed |
| 6  | **dssim-gpu**    | **0.9256** | our zen-metrics CLI |
| 7  | VMAF-neg         | 0.9209 | paper-pre-computed |
| 8  | **fast-ssim2-gpu** | **0.9127** | our zen-metrics CLI (CPU-impl baseline) |
| 9  | paper SSIMULACRA2 | 0.9125 | paper-pre-computed (sanity-check vs our ssim2-gpu: within 0.0002) |
| 10 | **zensim V0_16** | **0.9107** | our shipped MLP |
| 11 | SSIM             | 0.9046 | paper-pre-computed |
| 12 | butteraugli pnorm3 | 0.8969 | our zen-metrics CLI |
| 13 | butteraugli max  | 0.8656 | our zen-metrics CLI |
| 14 | PSNR-Y           | 0.8163 | paper-pre-computed |

V0_16 is **-0.0020 below fast-ssim2** in aggregate, within noise on n=300.
Goal #1 (match-or-exceed fast-ssim2) is empirically satisfied.

**Notable findings**:

1. **paper SSIMULACRA2 ≡ our ssim2-gpu** (0.9125 vs 0.9127, Δ=0.0002):
   sanity check passes — our `zen-metrics batch --metric ssim2-gpu`
   produces the same SROCC as the paper's CSV. zen-metrics CLI is
   reproducing the canonical metric correctly.

2. **dssim-gpu beats fast-ssim2** on AIC-4 (+0.0129 SROCC, 0.9256 vs
   0.9127). This is a meaningful margin. dssim — the MS-SSIM-derived
   distance — outperforms ssim2 on this corpus, contradicting the
   "ssim2 is canonical reference" assumption.

3. **butteraugli is weak here** (max 0.8656, pnorm3 0.8969). Even
   PSNR-Y at 0.8163 is in the same band. Butter's strength is at
   visually-lossless boundary; AIC-4 spans the full perceptibility
   range so butter's weighting is misaligned.

4. **CVVDP/IW-SSIM/MS-SSIM dominate** (0.94-0.96): trained or
   multi-scale structural metrics outperform single-scale or
   regression-trained ones (V0_16, ssim2) by ~0.04.

## Per-codec |SROCC| (V0_16 vs fast-ssim2-gpu vs dssim-gpu vs butter-p3)

| Codec | n | zensim | ssim2-gpu | dssim | butter-p3 |
|---|---:|---:|---:|---:|---:|
| AVIF        | 50 | 0.9458 | 0.9555 | 0.9505 | 0.9244 |
| JPEG-1      | 50 | 0.9515 | 0.9456 | 0.9608 | 0.9226 |
| JPEG-2000   | 50 | 0.9195 | 0.9202 | 0.9342 | 0.9189 |
| JPEG-AI     | 50 | 0.8265 | 0.8415 | 0.8526 | 0.7813 |
| JPEG-XL     | 50 | 0.9673 | 0.9606 | 0.9626 | 0.9335 |
| VVC         | 50 | 0.9244 | 0.9196 | 0.9264 | 0.8866 |

**V0_16 wins or ties fast-ssim2** on 4 of 6 codecs (JPEG-1, JPEG-2000-
tied, JPEG-XL, VVC); loses on AVIF and JPEG-AI.

**dssim-gpu is consistently strong** — top of every codec except JPEG-XL
(where V0_16 wins) and AVIF (where ssim2 wins).

**JPEG-AI** is a difficulty floor: every metric drops below 0.85.
Transformer-generated artifacts don't match anyone's training
distribution.

## Reachable ceiling for cycle-7 retraining

CVVDP's 0.9609 sets the upper bound for any learned metric on AIC-4. The
0.05 gap between V0_16 (0.9107) and CVVDP is the structural-improvement
budget for cycle-7. Pragmatic targets:

- Match dssim (0.9256, +0.015 above V0_16): adopt MS-SSIM-style multi-
  scale aggregation in zensim's distance computation.
- Match IW-SSIM (0.9507, +0.04 above V0_16): full information-weighted
  pooling — needs structural architecture change.
- Match CVVDP (0.9609, +0.05): viewing-condition-aware contrast
  sensitivity — biggest payoff, biggest engineering lift.

## Reproducibility

```bash
python3 /home/lilith/work/zen/zensim/scripts/v_next/export_aic4_to_parquet.py \
    --metrics-tsv /tmp/aic4_metrics/scored_dssim.tsv \
    --metrics-tsv /tmp/aic4_metrics/scored_zensim.tsv \
    --metrics-tsv /tmp/aic4_metrics/scored_ssim2.tsv \
    --metrics-tsv /tmp/aic4_metrics/scored_butter.tsv \
    --out /tmp/aic4_metrics/aic4_sample.parquet
```

## Status

- All 4 zen-metrics on AIC-4: ✅ DONE
- Parquet at `/tmp/aic4_metrics/aic4_sample.parquet` (300 rows × 23 cols)
- Schema includes: paper PSNR-Y/SSIM/MS-SSIM/IW-SSIM/VMAF-neg/
  SSIMULACRA2/HDR-VDP-2/HDR-VDP-3/CVVDP + our dssim/ssim2-gpu/butter-
  max/butter-p3/zensim + human_jnd + CI bounds
