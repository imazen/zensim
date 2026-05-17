# AIC dataset inventory — for zensim cycle following CID22

**Location work**: 2026-05-12 per user directive "locate the AIC datasets
mentioned in the first paragraph of the CID22 paper".

The CID22 paper (Sneyers et al. 2023, JPEG WG1 `wg1m99012`) references
several JPEG AIC efforts. Below: status, what's downloadable, and what
to ingest for our cycles.

## TL;DR — what to download first

**Priority 1 — AIC-3 CTC (EPFL MMSPG)**: this is the dataset shown in
CID22 paper Figure 17 (paper p. 25). The only publicly-released AIC
test corpus with per-image subjective scores in the "high to near-
visually lossless" range. **~1.5 GB FTP**.

**Priority 2 — AIC-4 Sample Dataset**: newer (2024/2025), available via
official JPEG portal `aicdb.jpeg.org`. PTC scores + metric scores on
GitHub. Same quality regime as CID22.

**Priority 3 — BTC-PTC-24**: GitHub-hosted raw boosted-triplet-comparison
responses from a 2024 AIC-3 follow-up study. Useful for methodology
reproduction (paper Table 2 / Figure 6 analysis).

## Detailed inventory

### 1. AIC-3 CTC dataset (EPFL MMSPG)

- **Source**: Testolina et al. "JPEG AIC-3 Dataset" (QoMEX 2023)
- **Landing page**: https://www.epfl.ch/labs/mmspg/downloads/jpeg-aic3-dataset/
- **Download**: FTP `tremplin.epfl.ch:21` user `jpeg-aic3@epfl.ch`
  password `.L:p*O` → path `2023-01/JPEG AIC-3 Dataset/`
- **Size**: ~1.5 GB
- **Layout**:
  - `original/` — 10 source images (PNG)
  - `decoded/` — 500 distorted PNGs (10 originals × 10 quality levels × 5 codecs)
  - `*_info.csv` — JND scores + 95% CIs per pair
  - Naming: `<CODEC>_<IMG_NUMBER>_<WxH>_<QUALITY>.png`
- **Codecs**: JPEG, JPEG 2000, HEVC Intra, VVC Intra, JPEG XL, AVIF
- **License**: CC0
- **Used for CID22 paper Figure 17** (cross-validation against AIC-3 CTC)

**Ingestion plan**: FTP to `/mnt/v/dataset/aic3_ctc_epfl/`. Test that
our `fast-ssim2` reproduces the paper's per-metric SROCC on this set
(Figure 17 numbers). If it does, the set becomes a **second
independent CID22-style held-out** for our V_NEXT bakes.

### 2. AIC-4 Sample Dataset

- **Portal**: https://aicdb.jpeg.org/
- **ZIP**: https://aicdb.jpeg.org/JPEG_AIC-4_Sample_Dataset.zip
- **Contents**: `full_resolution_images/` + `PTC_images/` (cropped)
- **Scores GitHub**: https://github.com/jpeg-aic/JPEG-AIC-4-datasets
  - `JPEG_AIC_reconstructed_jnd_scores.csv`
  - `JPEG-AIC_metric_scores.csv`
- **Codecs**: AVIF, JPEG, JPEG 2000, JPEG XL, VVC, **JPEG AI** (new)
- **Newer than CID22**: same quality regime but updated codec mix
- **Ingestion plan**: download to `/mnt/v/dataset/aic4_sample/`. Same
  pipeline as AIC-3 CTC.

### 3. AIC-3 BTC-PTC-24 (Boosted Triplet Comparison study)

- **GitHub**: https://github.com/jpeg-aic/dataset-btc-ptc-24
- **Contents**: Amazon Mechanical Turk crowdsourced responses (2024-01-04
  to 2024-01-10), demographics + per-triplet selections
- **Use**: methodology reproduction for our Goal 2 work — exercises
  the TSBPC protocol the CID22 paper describes on pages 4–14
- **Not an image dataset**; it's a response dataset. Pairs with
  AIC-3 CTC images.

### 4. AIC-HDR2025

- **GitHub**: https://github.com/jpeg-aic/AIC-HDR2025
- **Scope**: HDR fine-grained quality variant
- **Relevance to us**: low priority unless we move into HDR territory.
  zensim is currently SDR-focused.

### 5. AIC-1, AIC-2 (no public images)

- **AIC-1**: ISO/IEC TR 29170-1:2017, "Guidelines for image coding
  system evaluation". https://www.iso.org/standard (paid). No bundled
  test images — methodology document only.
- **AIC-2**: ISO/IEC 29170-2:2015, "Evaluation procedure for nearly
  lossless coding". https://www.iso.org/standard/66094.html (paid).
  Annex B defines the flicker test. No public test images.

These two are **standards, not datasets**. Skip unless we need to
implement the AIC-1/AIC-2 measurement protocols ourselves.

### 6. CTC documents (procedural)

- AIC-3 CTC PDF: https://ds.jpeg.org/documents/jpegaic/wg1n100334-097-ICQ-JPEG_AIC_CTC_for_subjective_quality_assessment.pdf
- AIC documentation index: https://jpeg.org/aic/documentation.html

Procedural, not data. Useful when running our own subjective tests
(zensim issue #25, the crowdsourced eval web app).

## Use-case mapping

| Need | Best AIC dataset |
|---|---|
| Second held-out (after CID22 49-ref) | **AIC-3 CTC (EPFL)** — 500 pairs, 5 codecs |
| Newer codec coverage (JPEG AI) | **AIC-4 Sample** |
| TSBPC methodology repro (CID22 Goal 2 §7) | **BTC-PTC-24** |
| HDR quality | AIC-HDR2025 |
| Standards-procedure (no data) | AIC-1/AIC-2 ISO docs (paid) |

## Pending user authorization

- Download AIC-3 CTC to `/mnt/v/dataset/aic3_ctc_epfl/` (~1.5 GB, FTP from EPFL, CC0)
- Download AIC-4 Sample to `/mnt/v/dataset/aic4_sample/` (ZIP from aicdb.jpeg.org)
- Clone BTC-PTC-24 + AIC-HDR2025 GitHub repos (small)

These give us 2-3 NEW datasets disjoint from CID22 to validate V_NEXT
bakes against. Combined with the existing CID22 49-ref + KADID + TID +
KonJND, we'd have 6-7 independent SROCC checks per bake.
