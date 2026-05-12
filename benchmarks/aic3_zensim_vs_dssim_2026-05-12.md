# AIC-3 CTC EPFL — zensim V0_16 vs dssim vs ssim2 (in flight)

**Dataset**: `/mnt/v/dataset/aic3_ctc_epfl/`
**Source CSV**: `info_with_bitrates.csv`
**n**: 600 (10 references × 6 codecs × 10 quality levels)
**Codecs**: AVIF, HM, JPEG-1, JPEG-2000, JPEGXL, VVC
**Human label**: `score.jnd` (reconstructed JND from subjective study; more negative = worse quality)

## Aggregate SROCC vs human JND (n=600)

| Metric | SROCC | Note |
|---|---:|---|
| **zensim V0_16 (current ship)** | **+0.7962** | parquet column `score_zensim` |
| dssim-gpu (sign-flipped)        | +0.7884 | `score_dssim` (lower = better; sign-flipped here for direction match) |
| bpp                              | +0.6334 | from CSV |
| q (encoder param)                | +0.0467 | weakly correlated; expected — q means different things per codec |

V0_16 wins by **+0.0078 SROCC** over dssim in aggregate. This is on a dataset
that's heavily B0/B1 (low-q regime), exactly where the cycle-7 plan is
looking for a structural improvement.

## Per-codec SROCC vs human JND

| Codec | n | zensim V0_16 | dssim (flipped) | Δ |
|---|---:|---:|---:|---:|
| AVIF       | 100 | **+0.8106** | +0.8039 | +0.0067 |
| HM         | 100 | **+0.7795** | +0.7712 | +0.0083 |
| JPEG-1     | 100 | +0.8497 | **+0.8510** | -0.0013 |
| JPEG-2000  | 100 | +0.7658 | +0.7597 | +0.0061 |
| JPEGXL     | 100 | **+0.8507** | +0.8319 | +0.0188 |
| VVC        | 100 | **+0.7999** | +0.7918 | +0.0081 |

V0_16 wins on 5 of 6 codecs; loses to dssim only on JPEG-1 by 0.0013 (within
noise). Biggest win is JPEGXL (+0.019).

## Why this matters

CID22's MOS distribution skews B2/B3 (high quality). AIC-3 carries
**reconstructed JND scores spanning the perceptibility threshold**: JND ≈
[-2.5, -0.25], with negative values indicating sub-PJND distortion levels.
This is the regime where ssim2 (and by extension our V0_16 trained against
ssim2) had been suspected to underperform.

The data shows V0_16 holds up on AIC-3 — meaningfully matches/exceeds dssim
across codecs. dssim is a structural metric (multiscale SSIM-like), zensim
is the trained MLP; on this corpus they're effectively tied with zensim
slightly ahead.

## Status

- AIC-3 dssim-gpu: **DONE** (600/600)
- AIC-3 zensim CPU: **DONE** (600/600)
- AIC-3 ssim2-gpu: in flight
- AIC-3 butteraugli-gpu: queued
- AIC-4 sample: zensim DONE; dssim/ssim2/butter queued
- Next: when ssim2-gpu finishes, add the zensim-vs-fast-ssim2 comparison
  on this corpus to this doc. Per CLAUDE.md goal #1 (match-or-exceed
  fast-ssim2 on every band), AIC-3 is an important cross-check that
  was previously not measurable.

## Reproducibility

```bash
python3 /home/lilith/work/zen/zensim/scripts/v_next/export_aic3_to_parquet.py \
    --metrics-tsv /tmp/aic3_dssim/scored_dssim.tsv \
    --metrics-tsv /tmp/aic3_dssim/scored_zensim.tsv \
    --out /tmp/aic3_dssim/aic3_ctc_epfl.parquet

python3 -c "
import pyarrow.parquet as pq
from scipy.stats import spearmanr
df = pq.read_table('/tmp/aic3_dssim/aic3_ctc_epfl.parquet').to_pandas()
for col in ['score_dssim', 'score_zensim']:
    rho, _ = spearmanr(df[col], df['human_jnd'])
    print(f'{col}: SROCC={rho:+.4f}')
"
```
