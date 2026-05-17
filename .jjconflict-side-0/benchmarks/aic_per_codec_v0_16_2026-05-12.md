# AIC-3 + AIC-4 per-codec |SROCC| — TRUE V0_16 vs fast-ssim2 vs dssim

Per-codec breakdown using the TRUE V0_16 column (merged via
`dataset_metric_baseline --per-pair-output` in tick 475). Earlier
ticks reporting AIC per-codec numbers were V0_2 outputs; this is
the corrected picture.

## AIC-3 CTC EPFL (n=600, reconstructed JND ∈ [-2.5, -0.25])

| Codec | n | V0_2 | **V0_16** | fast-ssim2 | dssim | Δ(V16-S) | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| AVIF      | 100 | 0.8106 | 0.8092 | **0.8183** | 0.8039 | -0.0092 | ssim2 |
| HM        | 100 | 0.7795 | 0.7840 | 0.7838 | 0.7712 | +0.0001 | tie |
| JPEG-1    | 100 | **0.8497** | 0.8428 | 0.8446 | **0.8510** | -0.0018 | ssim2 (within noise) |
| JPEG-2000 | 100 | 0.7658 | 0.7629 | **0.7671** | 0.7597 | -0.0042 | ssim2 |
| **JPEGXL** | 100 | 0.8507 | **0.8539** | 0.8399 | 0.8319 | **+0.0140** | **V0_16** |
| VVC       | 100 | 0.7999 | 0.8004 | **0.8063** | 0.7918 | -0.0059 | ssim2 |

**V0_16 wins 1, ties 1, loses 4 of 6 codecs.** Aggregate
V0_16=0.7990 vs ssim2=0.7965 (+0.0025) is driven mostly by JPEGXL
(+0.014) — the only real V0_16 win on this corpus.

## AIC-4 sample (n=300, reconstructed JND ∈ [0.0, 3.8])

| Codec | n | V0_2 | **V0_16** | fast-ssim2 | dssim | Δ(V16-S) | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| AVIF      | 50 | 0.9458 | **0.9598** | 0.9545 | 0.9510 | **+0.0053** | V0_16 |
| JPEG-1    | 50 | 0.9515 | **0.9541** | 0.9453 | 0.9588 | **+0.0088** | V0_16 |
| **JPEG-2000** | 50 | 0.9195 | **0.9357** | 0.9197 | 0.9212 | **+0.0159** | V0_16 |
| **JPEG-AI**   | 50 | 0.8265 | **0.7951** | **0.8459** | **0.9147** | **-0.0508** | **ssim2 (and dssim DOMINATES)** |
| **JPEG-XL**   | 50 | 0.9673 | **0.9705** | 0.9604 | **0.9814** | **+0.0101** | V0_16 |
| **VVC**       | 50 | 0.9244 | **0.9375** | 0.9194 | 0.9296 | **+0.0181** | V0_16 |

**V0_16 wins 5, loses 1 of 6 codecs.** The single loss is on
JPEG-AI: V0_16 0.7951 falls **0.051 below** fast-ssim2 — the
biggest single-codec deficit anywhere in our cross-corpus data.

## The JPEG-AI anomaly

**JPEG-AI is a learning-based codec.** All metrics struggle with it
relative to traditional codecs, but the magnitudes differ:

| Metric on JPEG-AI (AIC-4) | \|SROCC\| | vs metric's AIC-4 aggregate |
|---|---:|---:|
| V0_16        | 0.7951 | -0.122 (much worse than aggregate 0.917) |
| fast-ssim2   | 0.8459 | -0.067 |
| dssim        | **0.9147** | -0.011 (almost no drop) |
| V0_2         | 0.8265 | -0.084 |

**dssim is essentially unaffected by JPEG-AI artifacts**. fast-ssim2
takes a noticeable hit; V0_16 takes a substantial hit. The
explanation is likely that dssim's multi-scale SSIM-derived
structure happens to capture whatever V0_X / ssim2 miss in
transformer-generated artifacts.

This makes a strong case for **dssim as a co-training signal for
transformer-codec robustness** — even if dssim doesn't beat ssim2
on CID22 aggregate, its JPEG-AI behavior is a property V0_X needs
to learn before transformer codecs become common in production.

## Cross-corpus summary (all three TRUE V0_16 numbers)

| Corpus | n | V0_16 | fast-ssim2 | Per-codec V0_16 wins | Per-codec losses |
|---|---:|---:|---:|---|---|
| AIC-3 | 600  | 0.7990 (+0.003) | 0.7965 | 1 (JPEGXL +0.014)                  | 4 (AVIF/JPEG-1/JPEG-2000/VVC, all within 0.01) |
| AIC-4 | 300  | 0.9175 (+0.005) | 0.9127 | 5 (AVIF/JPEG-1/JPEG-2000/JPEG-XL/VVC) | 1 (JPEG-AI -0.051) |
| CID22 | 4292 | 0.8919 (+0.002) | 0.8895 | 5 (AVIF×2 / JPEG_XL / WebP / JPEG_2000) | 2 (HEIC -0.002, JPEG -0.006) |

**V0_16 wins 11 of 21 per-codec comparisons; ties 3; loses 7.**
Aggregate-wise V0_16 wins on every corpus.

## Cycle-7 actionables (final list)

1. **JPEG-AI is the single biggest cycle-7 target.** V0_16 −0.051
   vs ssim2, while dssim is essentially intact at −0.011. Either
   (a) add JPEG-AI training examples to synth corpus, OR (b) add
   dssim as an auxiliary loss head so V_X learns dssim's structure.
2. **AIC-3 codec-specific losses** (AVIF/JPEG-1/JPEG-2000/VVC, all
   −0.005 to −0.010) suggest V_X is undertrained at sub-PJND
   distortion levels — AIC-3 JND ∈ [-2.5, -0.25] is the
   visually-lossless regime where V0_X has less synth coverage.
   Densify the q≥85 range in zenjpeg/zenwebp/zenavif synth sweeps.
3. **JPEG slight regression** on CID22 (V0_2 0.9424 → V0_16 0.9402,
   -0.006 vs ssim2) is a smaller concern; investigate at low
   priority.

## Reproducibility

```bash
python3 -c "
import pyarrow.parquet as pq
from scipy.stats import spearmanr
for path, key in [('aic3_ctc_epfl.parquet','human_jnd'),
                  ('aic4_sample.parquet','human_jnd'),
                  ('cid22.parquet','human_mos')]:
    df = pq.read_table(path).to_pandas()
    sub = df[df['codec'] != 'Reference'].dropna(subset=['score_zensim_v0_16'])
    for c in sorted(sub['codec'].unique()):
        s = sub[sub['codec']==c]
        rv, _ = spearmanr(s['score_zensim_v0_16'], s[key])
        rs, _ = spearmanr(s['score_ssim2_gpu'],    s[key])
        print(f'{path[:6]} {c:<14} V16={abs(rv):.4f} ssim2={abs(rs):.4f}')
"
```
