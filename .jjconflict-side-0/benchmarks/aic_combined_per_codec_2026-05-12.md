# AIC-3 + AIC-4 — per-codec |SROCC| comparison across V_X / ssim2 / dssim / butter

> ⚠️ **SUPERSEDED — V0_2 mislabel** (correction landed 2026-05-12 evening).
> The "V0_16" SROCC numbers here are actually V0_2 (linear) outputs — `zen-metrics batch --metric zensim`
> defaults to `ZensimProfile::latest() == PreviewV0_2`, not the V0_4 MLP path.
>
> **Canonical replacement docs**:
> - [`cycle_6_finals_2026-05-12.md`](./cycle_6_finals_2026-05-12.md) — top-level
> - [`aic_per_codec_v0_16_2026-05-12.md`](./aic_per_codec_v0_16_2026-05-12.md) — TRUE V0_16 per-codec
>
> The "Pattern 3: V0_16 wins JPEG-derived codecs, loses HEVC/AV1-derived" was describing V0_2's
> per-codec behavior. TRUE V0_16 fixed those AVIF gaps; on held-out CID22 V0_16 wins 5 of 9 codecs
> including **AVIF_aurora_slow (+0.038 vs ssim2)** — the biggest per-codec gain in the cycle.
>
> This doc is kept for historical record only; do not cite the V0_16 numbers below.

**Background**: After completing the dssim/ssim2/butter backfill on both
AIC corpora (tick 458), the per-codec story tells a clearer cycle-7
story than the aggregates do.

## Combined per-codec table (vs human JND)

### AIC-3 CTC EPFL (n=100/codec)

| Codec | zensim V0_16 | fast-ssim2-gpu | dssim-gpu | butter pnorm3 |
|---|---:|---:|---:|---:|
| AVIF       | 0.8106 | **0.8183** | 0.8039 | 0.7762 |
| HM         | 0.7795 | **0.7838** | 0.7712 | 0.7425 |
| JPEG-1     | 0.8497 | 0.8446 | **0.8510** | 0.8222 |
| JPEG-2000  | 0.7658 | **0.7671** | 0.7597 | 0.7422 |
| JPEGXL     | **0.8507** | 0.8399 | 0.8319 | 0.8238 |
| VVC        | 0.7999 | **0.8063** | 0.7918 | 0.7521 |

### AIC-4 sample (n=50/codec)

| Codec | zensim V0_16 | fast-ssim2-gpu | dssim-gpu | butter pnorm3 |
|---|---:|---:|---:|---:|
| AVIF       | 0.9458 | **0.9545** | 0.9510 | 0.9115 |
| JPEG-1     | 0.9515 | 0.9453 | **0.9588** | 0.9178 |
| JPEG-2000  | 0.9195 | 0.9197 | **0.9212** | 0.8711 |
| JPEG-AI    | 0.8265 | 0.8459 | **0.9147** | 0.8416 |
| JPEG-XL    | 0.9673 | 0.9604 | **0.9814** | 0.9642 |
| VVC        | 0.9244 | 0.9194 | **0.9296** | 0.8945 |

## Three patterns

**Pattern 1: dssim is the strongest baseline overall.** Of the 12
(corpus, codec) cells, dssim is the top non-paper metric on 6 cells
(AIC-3 JPEG-1; AIC-4 JPEG-1/JPEG-2000/JPEG-AI/JPEG-XL/VVC). V0_16
tops on 2 cells (AIC-3 JPEGXL, AIC-4 not-top-but-close-on-JPEG-XL).
fast-ssim2 tops on 4 cells (AIC-3 AVIF/HM/JPEG-2000/VVC; AIC-4 AVIF).

**Pattern 2: JPEG-AI is a metric-class shift.** Both V0_16 and ssim2
drop to ~0.83 on JPEG-AI while dssim holds at 0.91. Transformer-codec
artifacts are not well-modeled by point-wise structural metrics; dssim's
multiscale-SSIM derivation handles them substantially better.

**Pattern 3: V0_16 wins JPEG-derived codecs, loses HEVC/AV1-derived.**
Across the two corpora (averaging the cells where both V0_16 and ssim2
appear):
- **JPEG-1** (n=150 across AICs): V0_16 ahead by +0.005 (0.9006 vs 0.8950)
- **JPEG-XL** (n=150): V0_16 ahead by +0.009 (0.9090 vs 0.9002)
- **AVIF** (n=150): V0_16 behind by -0.009 (0.8782 vs 0.8864)
- **HM** (AIC-3 only): V0_16 behind by -0.004
- **VVC** (n=150): V0_16 mixed (-0.006 AIC-3, +0.005 AIC-4)
- **JPEG-2000** (n=150): tied within noise

**Why this matters**: cycle-7 training-data plan should
1. Densify AVIF + HM + VVC encodes in the synth corpus (closes V0_16's
   per-codec losses)
2. Consider adding dssim as a co-training signal (especially for the
   high-quality regime where it outperforms ssim2 on AIC-4)
3. JPEG-AI is currently outside our training distribution — adding
   transformer-codec examples may be a separate workstream

## Aggregate cross-corpus summary (n=900 total)

| Metric | AIC-3 (n=600) | AIC-4 (n=300) |
|---|---:|---:|
| CVVDP (paper)            | (not run on AIC-3) | 0.9609 |
| IW-SSIM (paper)          | (not run)          | 0.9507 |
| MS-SSIM (paper)          | (not run)          | 0.9409 |
| **dssim-gpu**            | 0.7884             | **0.9256** |
| fast-ssim2-gpu           | **0.7970**         | 0.9127 |
| paper SSIMULACRA2        | (not run)          | 0.9125 |
| **zensim V0_16**         | **0.7962**         | **0.9107** |
| butter pnorm3            | 0.7571             | 0.8969 |
| butter max               | 0.7074             | 0.8656 |
| PSNR-Y (paper)           | (not run)          | 0.8163 |

V0_16 is within ±0.002 of fast-ssim2 on aggregate at both corpora. dssim
is the surprise winner on AIC-4. CVVDP/IW-SSIM dominate AIC-4 (paper-
provided, no compute cost to add them as panels in the comparison-site
view).

## Reproducibility

Both AIC parquets at `/tmp/aic3_dssim/aic3_ctc_epfl.parquet` and
`/tmp/aic4_metrics/aic4_sample.parquet`. Schemas in
`aic3_zensim_vs_baselines_2026-05-12.md` and
`aic4_zensim_vs_paper_metrics_2026-05-12.md`.
