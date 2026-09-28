# Rev4 E1b — is zensim's human deficit a cross-codec ordering problem? (2026-09-23)

Preregistered: `benchmarks/rev4_e1b_prereg_2026-09-23.md`. Scripts: `benchmarks/rev4_e1b_2026-09-23/`. Full run `/var/tmp/rev4-e1b/e1b_full.json` (sha256 `1ac7fc97d24d8588…`), B = 2000, seed 20260923, reference-clustered (forced choice: source-image-clustered, 5 images). Every accuracy is `acc_response` from `panel --pairwise` (sha256 `f11857c2…`, E1's binary). `**` CI entirely < 0, `*` CI entirely > 0. Δ = zensim − best peer of that pair class.

**This re-tests E1's exploratory lead on the same data it came from** (prereg §1). A confirmation here would mean the lead survives a fixed rule, not that it replicates.

## Decision (prereg §6)

| model | primary | family members separate (i) | best peer fixed (ii) | CID22 E1 codec def. (iii) | unit classes (primary) |
|---|---|---|---|---|---|
| B (served default; CSIQ: 07-07 bake/08-30 root; FC: 07-07 bake v1) | **REFUTED** | UNRESOLVED | REFUTED | REFUTED | cid22a NONSPEC, csiq NONSPEC, jpeg_aic_family CONF; members aic3 CONF, aic4crop CONF, fc_btc_native CONF; AIC-4 full CONF; TID MISSING |
| C = W10L9PH_s4004_packed | **UNRESOLVED** | UNRESOLVED | UNRESOLVED | UNRESOLVED | cid22a CONF, csiq NONE, jpeg_aic_family NONE; members aic3 NONE, aic4crop NONE, fc_btc_native NONE; AIC-4 full NONE; TID NONE |
| D (MT914_matched_D; CSIQ D_shipped@dguard2) | **UNRESOLVED** | UNRESOLVED | UNRESOLVED | UNRESOLVED | cid22a CONF, csiq NONE, jpeg_aic_family NONE; members aic3 NONE, aic4crop NONE, fc_btc_native MISSING; AIC-4 full NONE; TID MISSING |
| Rev3 fast R915_y60_h32_ens5 | **REFUTED** | REFUTED | REFUTED | UNRESOLVED | cid22a NONSPEC, csiq MISSING, jpeg_aic_family NONE; members aic3 NONE, aic4crop NONE, fc_btc_native MISSING; AIC-4 full NONE; TID MISSING |
| Rev3 rich R915_basic228_h128_ens5 | **UNRESOLVED** | UNRESOLVED | UNRESOLVED | UNRESOLVED | cid22a NONE, csiq MISSING, jpeg_aic_family NONE; members aic3 NONE, aic4crop NONE, fc_btc_native MISSING; AIC-4 full NONE; TID MISSING |
| PreviewV0_2 (CID22/AIC-3 May-era site parquet; AIC-4 pixel read) | **UNRESOLVED** | UNRESOLVED | UNRESOLVED | UNRESOLVED | cid22a CONF, csiq MISSING, jpeg_aic_family NONE; members aic3 NONE, aic4crop NONE, fc_btc_native MISSING; AIC-4 full NONE; TID MISSING |

CONF = cross deficit (CI < 0) with same-codec tie/win; NONSPEC = same-codec deficit (CI < 0) not demonstrably smaller than the cross one; NONE = no deficit; MISSING = no rows.

## Reading

- **B (served default): REFUTED.** On the JPEG-AIC family its deficit is cross-codec-specific: AIC-3 Δx −0.041, AIC-4 crop
  −0.054, forced choice −0.016, all CI < 0, with same-codec ties at the ceiling. It is not cross-specific elsewhere. On CSIQ it
  loses as much on JPEG/JPEG2000 ladders as across them (Δs −0.062 [−0.078, −0.045], Δx −0.061; 07-07 bake era, not the
  served bytes). On CID22-A its only deficit is same-codec (Δs −0.0026 vs IW-SSIM, CI < 0); cross pairs tie SSIMULACRA2.
  With the family members counted separately, B becomes UNRESOLVED (3 CONF vs 2 NONSPEC).
- **C (`W10L9PH_s4004`): UNRESOLVED.** Only CID22-A shows the pattern: Δx −0.0089 [−0.0130, −0.0049] vs SSIMULACRA2, a
  same-codec tie, and DiD −0.0088 (CI < 0). AIC-4 crop vs CVVDP at `standard_fhd` is −0.020 [−0.0415, +0.0020] on 5 references
  and just misses. AIC-3, forced choice and CSIQ tie. On the AIC-4 full-resolution rendering C beats SSIMULACRA2 (+0.0076*).
  **To resolve:** a second multi-codec corpus with enough references (the full AIC-4 set rather than the 5-reference sample),
  CVVDP at its documented display on AIC-3 and CID22-A (E2b), and forced-choice peers other than SSIMULACRA2.
- **Same-codec pairs are near the ceiling on JPEG-AIC.** Every metric except butteraugli max orders ≥ 99.8 % of same-codec
  AIC-3/AIC-4 pairs correctly, and forced-choice same-codec accuracy is identical for all three scorers. The same-codec "tie"
  there cannot tell a model that understands a ladder from one that only orders it monotonically.
- **Descriptive: where the cross-codec errors are.** The deficits sit mostly in small-gap pairs (< 1 JND, or at most the median MOS gap): e.g. C on CID22-A −0.0157 small vs −0.0022 large, B on AIC-3 −0.067 vs −0.012.
  Large-gap pairs are ≥ 0.989 for the best peer on every JND unit. Pairs that involve JPEG dominate: C on CID22-A loses on JPEG vs
  WebP −0.025, AVIF −0.019, HEIC −0.017 and JXL −0.011. On AIC-4 crop, vs CVVDP-fhd, it loses on JPEG-1 vs JPEG-2000 −0.14,
  JPEG-2000 vs JPEG-XL −0.094, JPEG-1 vs VVC −0.086 and AVIF vs JPEG-1 −0.040. C is ahead of CVVDP-fhd on all five JPEG-AI pairs (CI > 0 on four).
- **Exploratory: the JPEG error has opposite signs.** On CID22-A (mozjpeg at web qualities) C over-rates JPEG (484 of 510
  wrong-where-SSIMULACRA2-right pairs). On AIC-3/AIC-4 (JPEG-1 near threshold) every zensim model under-rates JPEG-1
  (C: 85 of 94 on AIC-3 and 162 of 162 on AIC-4 crop). One JPEG offset cannot fix both. Cross-codec supervision needs pairs at
  matched quality in both regimes.
- **Where peers win:** SSIMULACRA2 leads cross pairs on CID22-A (0.9115) and CSIQ (0.9479). CVVDP-fhd leads AIC-4 crop (0.9136
  vs C 0.8936). IW-SSIM leads AIC-3 (0.9400, C 0.9397 ties). butteraugli 3-norm leads TID (TRAIN). zensim was trained on
  SSIMULACRA2 labels, so ties with SSIMULACRA2 are partly distillation.

## Decision inputs: Δx (cross), Δs (same), DiD = Δx − Δs

| unit | model | best peer cross / same | n pairs cross / same (refs) | Δx [CI] | Δs [CI] | DiD [CI] |
|---|---|---|---|---|---|---|
| CID22-A(25) | B | ssim2 / iwssim | 70420 / 9988 (25) | -0.0008 [-0.0081, +0.0062] | -0.0026 [-0.0059, -0.0006]** | +0.0018 [-0.0042, +0.0079] |
| CID22-A(25) | C | ssim2 / iwssim | 70420 / 9988 (25) | -0.0089 [-0.0130, -0.0049]** | -0.0001 [-0.0003, +0.0000] | -0.0088 [-0.0128, -0.0048]** |
| CID22-A(25) | D | ssim2 / iwssim | 70420 / 9988 (25) | -0.0129 [-0.0193, -0.0067]** | +0.0001 [+0.0000, +0.0003] | -0.0130 [-0.0197, -0.0068]** |
| CID22-A(25) | R915_fast | ssim2 / iwssim | 70420 / 9988 (25) | +0.0016 [-0.0018, +0.0049] | -0.0013 [-0.0027, -0.0003]** | +0.0029 [-0.0003, +0.0061] |
| CID22-A(25) | R915_rich | ssim2 / iwssim | 70420 / 9988 (25) | -0.0012 [-0.0045, +0.0024] | -0.0006 [-0.0017, +0.0000] | -0.0006 [-0.0034, +0.0028] |
| CID22-A(25) | V0_2 | ssim2 / iwssim | 70420 / 9988 (25) | -0.0102 [-0.0158, -0.0048]** | -0.0004 [-0.0012, +0.0000] | -0.0098 [-0.0149, -0.0048]** |
| CID22-A, E1 encoder-dir codec (sens. iii) | B | ssim2 / dssim | 82657 / 12507 (25) | -0.0028 [-0.0106, +0.0048] | -0.0037 [-0.0072, -0.0008]** | +0.0009 [-0.0063, +0.0079] |
| CID22-A, E1 encoder-dir codec (sens. iii) | C | ssim2 / dssim | 82657 / 12507 (25) | -0.0071 [-0.0109, -0.0031]** | -0.0004 [-0.0013, +0.0004] | -0.0067 [-0.0103, -0.0026]** |
| CID22-A, E1 encoder-dir codec (sens. iii) | D | ssim2 / dssim | 82657 / 12507 (25) | -0.0122 [-0.0187, -0.0060]** | +0.0017 [-0.0001, +0.0036] | -0.0139 [-0.0199, -0.0077]** |
| CID22-A, E1 encoder-dir codec (sens. iii) | R915_fast | ssim2 / dssim | 82657 / 12507 (25) | +0.0015 [-0.0019, +0.0049] | -0.0014 [-0.0038, +0.0008] | +0.0028 [-0.0008, +0.0067] |
| CID22-A, E1 encoder-dir codec (sens. iii) | R915_rich | ssim2 / dssim | 82657 / 12507 (25) | -0.0003 [-0.0036, +0.0036] | -0.0005 [-0.0033, +0.0019] | +0.0002 [-0.0033, +0.0042] |
| CID22-A, E1 encoder-dir codec (sens. iii) | V0_2 | ssim2 / dssim | 82657 / 12507 (25) | -0.0096 [-0.0150, -0.0044]** | -0.0003 [-0.0025, +0.0017] | -0.0093 [-0.0137, -0.0046]** |
| CSIQ JPEG+JPEG2000 | B | ssim2 / ssim2 | 749 / 600 (30) | -0.0614 [-0.0841, -0.0401]** | -0.0617 [-0.0783, -0.0450]** | +0.0003 [-0.0246, +0.0260] |
| CSIQ JPEG+JPEG2000 | C | ssim2 / ssim2 | 749 / 600 (30) | -0.0013 [-0.0160, +0.0147] | +0.0000 [+0.0000, +0.0000] | -0.0013 [-0.0160, +0.0147] |
| CSIQ JPEG+JPEG2000 | D | ssim2 / ssim2 | 749 / 600 (30) | +0.0000 [-0.0094, +0.0094] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0094, +0.0094] |
| AIC-3 CTC (decoded PNG, source res.) | B | iwssim / ssim2 | 13500 / 2700 (10) | -0.0412 [-0.0666, -0.0144]** | -0.0011 [-0.0022, +0.0000] | -0.0401 [-0.0657, -0.0129]** |
| AIC-3 CTC (decoded PNG, source res.) | C | iwssim / ssim2 | 13500 / 2700 (10) | -0.0003 [-0.0078, +0.0076] | +0.0000 [+0.0000, +0.0000] | -0.0003 [-0.0078, +0.0076] |
| AIC-3 CTC (decoded PNG, source res.) | D | iwssim / ssim2 | 13500 / 2700 (10) | -0.0029 [-0.0137, +0.0081] | +0.0000 [+0.0000, +0.0000] | -0.0029 [-0.0137, +0.0081] |
| AIC-3 CTC (decoded PNG, source res.) | R915_fast | iwssim / ssim2 | 13500 / 2700 (10) | -0.0020 [-0.0116, +0.0072] | +0.0000 [+0.0000, +0.0000] | -0.0020 [-0.0116, +0.0072] |
| AIC-3 CTC (decoded PNG, source res.) | R915_rich | iwssim / ssim2 | 13500 / 2700 (10) | -0.0075 [-0.0211, +0.0053] | +0.0000 [+0.0000, +0.0000] | -0.0075 [-0.0211, +0.0053] |
| AIC-3 CTC (decoded PNG, source res.) | V0_2 | iwssim / ssim2 | 13500 / 2700 (10) | -0.0054 [-0.0195, +0.0098] | +0.0000 [+0.0000, +0.0000] | -0.0054 [-0.0195, +0.0098] |
| AIC-4 crop (620×800, as shown) | B | cvvdp_fhd / ssim2 | 7500 / 1350 (5) | -0.0536 [-0.0812, -0.0305]** | -0.0007 [-0.0022, +0.0000] | -0.0529 [-0.0808, -0.0286]** |
| AIC-4 crop (620×800, as shown) | C | cvvdp_fhd / ssim2 | 7500 / 1350 (5) | -0.0200 [-0.0415, +0.0020] | +0.0000 [+0.0000, +0.0000] | -0.0200 [-0.0415, +0.0020] |
| AIC-4 crop (620×800, as shown) | D | cvvdp_fhd / ssim2 | 7500 / 1350 (5) | -0.0101 [-0.0324, +0.0107] | +0.0000 [+0.0000, +0.0000] | -0.0101 [-0.0324, +0.0107] |
| AIC-4 crop (620×800, as shown) | R915_fast | cvvdp_fhd / ssim2 | 7500 / 1350 (5) | -0.0228 [-0.0467, +0.0040] | +0.0000 [+0.0000, +0.0000] | -0.0228 [-0.0467, +0.0040] |
| AIC-4 crop (620×800, as shown) | R915_rich | cvvdp_fhd / ssim2 | 7500 / 1350 (5) | -0.0185 [-0.0376, +0.0012] | -0.0007 [-0.0022, +0.0000] | -0.0178 [-0.0369, +0.0019] |
| AIC-4 crop (620×800, as shown) | V0_2 | cvvdp_fhd / ssim2 | 7500 / 1350 (5) | -0.0248 [-0.0505, +0.0009] | +0.0000 [+0.0000, +0.0000] | -0.0248 [-0.0505, +0.0009] |
| AIC-4 full res. (secondary) | B | ssim2 / ssim2 | 7500 / 1350 (5) | -0.0305 [-0.0513, -0.0092]** | -0.0015 [-0.0030, +0.0000] | -0.0291 [-0.0484, -0.0092]** |
| AIC-4 full res. (secondary) | C | ssim2 / ssim2 | 7500 / 1350 (5) | +0.0076 [+0.0001, +0.0137]* | +0.0000 [+0.0000, +0.0000] | +0.0076 [+0.0001, +0.0137]* |
| AIC-4 full res. (secondary) | D | ssim2 / ssim2 | 7500 / 1350 (5) | +0.0175 [+0.0061, +0.0308]* | +0.0000 [+0.0000, +0.0000] | +0.0175 [+0.0061, +0.0308]* |
| AIC-4 full res. (secondary) | R915_fast | ssim2 / ssim2 | 7500 / 1350 (5) | +0.0069 [+0.0019, +0.0115]* | +0.0000 [+0.0000, +0.0000] | +0.0069 [+0.0019, +0.0115]* |
| AIC-4 full res. (secondary) | R915_rich | ssim2 / ssim2 | 7500 / 1350 (5) | +0.0123 [+0.0032, +0.0240]* | +0.0000 [+0.0000, +0.0000] | +0.0123 [+0.0032, +0.0240]* |
| AIC-4 full res. (secondary) | V0_2 | ssim2 / ssim2 | 7500 / 1350 (5) | +0.0051 [-0.0023, +0.0103] | +0.0000 [+0.0000, +0.0000] | +0.0051 [-0.0023, +0.0103] |
| TID2013 JPEG+JPEG2000 (TRAIN, descriptive) | C | butter_p3 / ssim2 | 625 / 499 (25) | -0.0064 [-0.0288, +0.0176] | +0.0000 [+0.0000, +0.0000] | -0.0064 [-0.0288, +0.0176] |
| JPEG-AIC forced choice btc_native | C | ssim2 / ssim2 | 610 / 2700 (5) | -0.0020 [-0.0072, +0.0018] | -0.0000 [-0.0000, +0.0000] | -0.0020 [-0.0071, +0.0019] |
| JPEG-AIC forced choice btc_native | B | ssim2 / ssim2 | 610 / 2700 (5) | -0.0164 [-0.0249, -0.0094]** | +0.0000 [+0.0000, +0.0000] | -0.0164 [-0.0249, -0.0094]** |
## Descriptive strata (cross pairs; Δ vs the unit's cross best peer, fixed)

**CID22-A(25)** — peer ssim2; gap threshold 11.413699999999992

| stratum | n | peer acc | B Δ [CI] | C Δ [CI] |
|---|---|---|---|---|
| gap_small | 35210 | 0.8263 | +0.0001 [-0.0120, +0.0119] | -0.0157 [-0.0225, -0.0089]** |
| gap_large | 35210 | 0.9966 | -0.0016 [-0.0055, +0.0010] | -0.0022 [-0.0052, +0.0002] |
| pair:AVIF vs HEIC | 7976 | 0.9115 | -0.0090 [-0.0197, +0.0003] | -0.0036 [-0.0107, +0.0030] |
| pair:AVIF vs JP2 | 8973 | 0.9189 | -0.0020 [-0.0224, +0.0169] | -0.0068 [-0.0149, +0.0021] |
| pair:AVIF vs JPEG | 10892 | 0.8975 | +0.0081 [-0.0017, +0.0180] | -0.0193 [-0.0270, -0.0119]** |
| pair:AVIF vs JXL | 10841 | 0.8907 | +0.0061 [-0.0088, +0.0192] | -0.0086 [-0.0183, +0.0000] |
| pair:AVIF vs WebP | 8973 | 0.9276 | -0.0152 [-0.0261, -0.0059]** | -0.0086 [-0.0167, -0.0021]** |
| pair:HEIC vs JP2 | 1800 | 0.9106 | +0.0011 [-0.0083, +0.0100] | +0.0017 [-0.0061, +0.0100] |
| pair:HEIC vs JPEG | 2184 | 0.9107 | +0.0119 [+0.0014, +0.0240]* | -0.0169 [-0.0242, -0.0097]** |
| pair:HEIC vs JXL | 2176 | 0.9200 | +0.0051 [-0.0042, +0.0151] | +0.0000 [-0.0069, +0.0069] |
| pair:HEIC vs WebP | 1800 | 0.9228 | +0.0056 [-0.0056, +0.0167] | +0.0061 [-0.0050, +0.0178] |
| pair:JP2 vs JPEG | 2457 | 0.9284 | -0.0106 [-0.0195, -0.0016]** | -0.0061 [-0.0163, +0.0025] |
| pair:JP2 vs JXL | 2448 | 0.9257 | -0.0167 [-0.0289, -0.0057]** | -0.0029 [-0.0167, +0.0094] |
| pair:JP2 vs WebP | 2025 | 0.9190 | +0.0020 [-0.0138, +0.0168] | -0.0079 [-0.0193, +0.0030] |
| pair:JPEG vs JXL | 2970 | 0.9441 | -0.0047 [-0.0131, +0.0040] | -0.0108 [-0.0192, -0.0017]** |
| pair:JPEG vs WebP | 2457 | 0.9015 | +0.0122 [+0.0033, +0.0212]* | -0.0248 [-0.0328, -0.0165]** |
| pair:JXL vs WebP | 2448 | 0.8979 | +0.0057 [-0.0028, +0.0134] | -0.0025 [-0.0122, +0.0077] |

**CSIQ JPEG+JPEG2000** — peer ssim2; gap threshold 0.29521328762953625

| stratum | n | peer acc | B Δ [CI] | C Δ [CI] |
|---|---|---|---|---|
| gap_small | 375 | 0.8960 | -0.1120 [-0.1549, -0.0720]** | -0.0027 [-0.0316, +0.0294] |
| gap_large | 374 | 1.0000 | -0.0107 [-0.0209, -0.0027]** | +0.0000 [+0.0000, +0.0000] |
| pair:JPEG vs jpeg2000 | 749 | 0.9479 | -0.0614 [-0.0841, -0.0401]** | -0.0013 [-0.0160, +0.0147] |

**AIC-3 CTC (decoded PNG, source res.)** — peer iwssim; gap threshold 1 JND

| stratum | n | peer acc | B Δ [CI] | C Δ [CI] |
|---|---|---|---|---|
| gap_small | 7200 | 0.8883 | -0.0672 [-0.1076, -0.0212]** | -0.0001 [-0.0137, +0.0146] |
| gap_large | 6300 | 0.9992 | -0.0116 [-0.0229, -0.0022]** | -0.0006 [-0.0019, +0.0003] |
| pair:AVIF vs HM | 900 | 0.9422 | -0.0078 [-0.0211, +0.0078] | -0.0033 [-0.0156, +0.0067] |
| pair:AVIF vs JPEG-1 | 900 | 0.9467 | -0.0722 [-0.1278, -0.0111]** | +0.0100 [-0.0211, +0.0389] |
| pair:AVIF vs JPEG-2000 | 900 | 0.9767 | -0.0089 [-0.0256, +0.0078] | -0.0044 [-0.0256, +0.0133] |
| pair:AVIF vs JPEGXL | 900 | 0.9667 | -0.0411 [-0.0833, +0.0011] | +0.0067 [-0.0056, +0.0200] |
| pair:AVIF vs VVC | 900 | 0.9756 | -0.0256 [-0.0811, +0.0089] | -0.0022 [-0.0144, +0.0122] |
| pair:HM vs JPEG-1 | 900 | 0.8722 | -0.0667 [-0.1167, -0.0122]** | +0.0167 [-0.0011, +0.0400] |
| pair:HM vs JPEG-2000 | 900 | 0.9078 | -0.0011 [-0.0156, +0.0133] | +0.0089 [-0.0044, +0.0278] |
| pair:HM vs JPEGXL | 900 | 0.9200 | -0.0578 [-0.1011, -0.0100]** | +0.0011 [-0.0167, +0.0178] |
| pair:HM vs VVC | 900 | 0.9494 | -0.0239 [-0.0722, +0.0033] | -0.0039 [-0.0228, +0.0178] |
| pair:JPEG-1 vs JPEG-2000 | 900 | 0.9300 | -0.0944 [-0.1478, -0.0344]** | +0.0022 [-0.0322, +0.0344] |
| pair:JPEG-1 vs JPEGXL | 900 | 0.9400 | +0.0022 [-0.0122, +0.0200] | -0.0044 [-0.0244, +0.0156] |
| pair:JPEG-1 vs VVC | 900 | 0.9267 | -0.0722 [-0.1456, +0.0022] | -0.0111 [-0.0478, +0.0289] |
| pair:JPEG-2000 vs JPEGXL | 900 | 0.9622 | -0.0778 [-0.1278, -0.0200]** | -0.0156 [-0.0522, +0.0167] |
| pair:JPEG-2000 vs VVC | 900 | 0.9411 | -0.0222 [-0.0711, +0.0100] | -0.0044 [-0.0156, +0.0056] |
| pair:JPEGXL vs VVC | 900 | 0.9433 | -0.0489 [-0.1100, +0.0178] | -0.0011 [-0.0233, +0.0233] |

**AIC-4 crop (620×800, as shown)** — peer cvvdp_fhd; gap threshold 1 JND

| stratum | n | peer acc | B Δ [CI] | C Δ [CI] |
|---|---|---|---|---|
| gap_small | 4081 | 0.8420 | -0.0887 [-0.1428, -0.0498]** | -0.0331 [-0.0741, +0.0069] |
| gap_large | 3419 | 0.9991 | -0.0117 [-0.0262, -0.0003]** | -0.0044 [-0.0092, +0.0000] |
| pair:AVIF vs JPEG-1 | 500 | 0.8980 | -0.1580 [-0.2000, -0.1020]** | -0.0400 [-0.0680, -0.0020]** |
| pair:AVIF vs JPEG-2000 | 500 | 0.9420 | +0.0060 [-0.0400, +0.0380] | -0.0240 [-0.0720, +0.0240] |
| pair:AVIF vs JPEG-AI | 500 | 0.8940 | +0.0240 [-0.0220, +0.0980] | +0.0380 [+0.0220, +0.0520]* |
| pair:AVIF vs JPEG-XL | 500 | 0.9400 | -0.0940 [-0.1640, -0.0140]** | -0.0220 [-0.0740, +0.0340] |
| pair:AVIF vs VVC | 500 | 0.9280 | +0.0240 [-0.0120, +0.0580] | -0.0140 [-0.0400, +0.0140] |
| pair:JPEG-1 vs JPEG-2000 | 500 | 0.8960 | -0.2300 [-0.2960, -0.1640]** | -0.1400 [-0.2080, -0.0760]** |
| pair:JPEG-1 vs JPEG-AI | 500 | 0.8800 | +0.0000 [-0.0080, +0.0080] | +0.0120 [+0.0060, +0.0180]* |
| pair:JPEG-1 vs JPEG-XL | 500 | 0.9220 | -0.0060 [-0.0140, +0.0020] | -0.0040 [-0.0220, +0.0120] |
| pair:JPEG-1 vs VVC | 500 | 0.8820 | -0.1720 [-0.2000, -0.1440]** | -0.0860 [-0.1120, -0.0460]** |
| pair:JPEG-2000 vs JPEG-AI | 500 | 0.8980 | +0.0120 [-0.0620, +0.0880] | +0.0560 [+0.0180, +0.1000]* |
| pair:JPEG-2000 vs JPEG-XL | 500 | 0.9560 | -0.1620 [-0.2240, -0.0980]** | -0.0940 [-0.1460, -0.0340]** |
| pair:JPEG-2000 vs VVC | 500 | 0.9340 | +0.0280 [+0.0020, +0.0540]* | +0.0300 [+0.0040, +0.0520]* |
| pair:JPEG-AI vs JPEG-XL | 500 | 0.8880 | +0.0380 [+0.0200, +0.0580]* | +0.0160 [+0.0060, +0.0280]* |
| pair:JPEG-AI vs VVC | 500 | 0.9260 | -0.0100 [-0.0580, +0.0440] | +0.0280 [-0.0100, +0.0660] |
| pair:JPEG-XL vs VVC | 500 | 0.9200 | -0.1040 [-0.1540, -0.0480]** | -0.0560 [-0.1100, +0.0060] |

**AIC-4 full res. (secondary)** — peer ssim2; gap threshold 1 JND

| stratum | n | peer acc | B Δ [CI] | C Δ [CI] |
|---|---|---|---|---|
| gap_small | 4081 | 0.8047 | -0.0507 [-0.0888, -0.0156]** | +0.0115 [-0.0002, +0.0208] |
| gap_large | 3419 | 0.9889 | -0.0064 [-0.0151, +0.0017] | +0.0029 [-0.0009, +0.0083] |
| pair:AVIF vs JPEG-1 | 500 | 0.8020 | -0.0860 [-0.1140, -0.0660]** | +0.0380 [+0.0120, +0.0700]* |
| pair:AVIF vs JPEG-2000 | 500 | 0.9200 | +0.0340 [-0.0040, +0.0700] | +0.0060 [-0.0260, +0.0460] |
| pair:AVIF vs JPEG-AI | 500 | 0.9120 | +0.0040 [-0.0660, +0.1060] | -0.0100 [-0.0220, +0.0020] |
| pair:AVIF vs JPEG-XL | 500 | 0.9300 | -0.0840 [-0.1240, -0.0380]** | +0.0240 [+0.0080, +0.0400]* |
| pair:AVIF vs VVC | 500 | 0.9300 | -0.0220 [-0.1060, +0.0340] | -0.0080 [-0.0260, +0.0080] |
| pair:JPEG-1 vs JPEG-2000 | 500 | 0.7940 | -0.1080 [-0.1420, -0.0760]** | -0.0020 [-0.0280, +0.0220] |
| pair:JPEG-1 vs JPEG-AI | 500 | 0.8800 | +0.0220 [+0.0020, +0.0560]* | -0.0040 [-0.0160, +0.0080] |
| pair:JPEG-1 vs JPEG-XL | 500 | 0.8760 | +0.0200 [-0.0180, +0.0500] | +0.0080 [-0.0140, +0.0280] |
| pair:JPEG-1 vs VVC | 500 | 0.7860 | -0.0580 [-0.1100, +0.0220] | +0.0340 [+0.0080, +0.0500]* |
| pair:JPEG-2000 vs JPEG-AI | 500 | 0.9360 | -0.0160 [-0.0920, +0.0600] | -0.0120 [-0.0320, +0.0080] |
| pair:JPEG-2000 vs JPEG-XL | 500 | 0.9180 | -0.0860 [-0.1080, -0.0520]** | +0.0000 [-0.0180, +0.0160] |
| pair:JPEG-2000 vs VVC | 500 | 0.9340 | -0.0200 [-0.0800, +0.0140] | +0.0240 [+0.0160, +0.0340]* |
| pair:JPEG-AI vs JPEG-XL | 500 | 0.9100 | +0.0300 [-0.0120, +0.0780] | -0.0060 [-0.0160, +0.0040] |
| pair:JPEG-AI vs VVC | 500 | 0.9100 | -0.0360 [-0.0720, +0.0240] | +0.0040 [-0.0080, +0.0160] |
| pair:JPEG-XL vs VVC | 500 | 0.8920 | -0.0520 [-0.0920, -0.0100]** | +0.0180 [+0.0100, +0.0260]* |

**TID2013 JPEG+JPEG2000 (TRAIN, descriptive)** — peer butter_p3; gap threshold 0.1766377778

| stratum | n | peer acc | C Δ [CI] |
|---|---|---|---|
| gap_small | 313 | 0.8914 | -0.0128 [-0.0571, +0.0350] |
| gap_large | 312 | 1.0000 | +0.0000 [+0.0000, +0.0000] |
| pair:JPEG vs JPEG2000 | 625 | 0.9456 | -0.0064 [-0.0288, +0.0176] |

**JPEG-AIC forced choice btc_native** — peer ssim2; gap threshold |dlevel_left - dlevel_right| <= 1 small

| stratum | n | peer acc | B Δ [CI] | C Δ [CI] |
|---|---|---|---|---|
| gap_small | 330 | 0.6379 | -0.0179 [-0.0290, -0.0121]** | -0.0021 [-0.0068, +0.0034] |
| gap_large | 280 | 0.7239 | -0.0149 [-0.0239, -0.0051]** | -0.0019 [-0.0085, +0.0044] |
| pair:AVIF vs JPEG-1 | 54 | 0.6465 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| pair:AVIF vs JPEG-2000 | 52 | 0.6144 | +0.0072 [-0.0019, +0.0305] | -0.0007 [-0.0022, +0.0000] |
| pair:AVIF vs JPEG-AI | 24 | 0.8875 | -0.0619 [-0.1808, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| pair:AVIF vs JPEG-XL | 56 | 0.5439 | -0.0419 [-0.0801, -0.0093]** | -0.0111 [-0.0525, +0.0194] |
| pair:AVIF vs VVC | 52 | 0.6513 | -0.0042 [-0.0123, +0.0000] | -0.0042 [-0.0123, +0.0000] |
| pair:JPEG-1 vs JPEG-2000 | 50 | 0.5383 | -0.0251 [-0.0566, -0.0064]** | -0.0078 [-0.0277, +0.0000] |
| pair:JPEG-1 vs JPEG-AI | 24 | 0.8726 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| pair:JPEG-1 vs JPEG-XL | 36 | 0.6668 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| pair:JPEG-1 vs VVC | 62 | 0.6927 | -0.0049 [-0.0182, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| pair:JPEG-2000 vs JPEG-AI | 12 | 0.9110 | -0.1402 [-0.1937, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| pair:JPEG-2000 vs JPEG-XL | 46 | 0.6058 | -0.0568 [-0.1554, +0.0386] | -0.0216 [-0.0582, +0.0000] |
| pair:JPEG-2000 vs VVC | 50 | 0.6898 | -0.0019 [-0.0064, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| pair:JPEG-AI vs JPEG-XL | 16 | 0.8717 | +0.0000 [+0.0000, +0.0000] | +0.0222 [+0.0000, +0.0764] |
| pair:JPEG-AI vs VVC | 34 | 0.9049 | +0.0000 [+0.0000, +0.0000] | +0.0098 [+0.0000, +0.0323] |
| pair:JPEG-XL vs VVC | 42 | 0.6003 | -0.0082 [-0.0208, +0.0000] | +0.0043 [+0.0000, +0.0167] |

## EXPLORATORY (not preregistered): which side does zensim over-rate in JPEG-vs-other errors?

Source `benchmarks/rev4_e1b_2026-09-23/direction.py` → `/var/tmp/rev4-e1b/direction.json` (sha256 `4e783bf2c08de6ac…`). Among cross pairs involving JPEG that the model orders wrong while the unit's cross best peer orders right: `JPEG over-rated` = humans judged the JPEG stimulus worse; `JPEG under-rated` = humans judged the other codec worse.

| unit | peer | B | C | D | R915_fast | R915_rich | V0_2 |
|---|---|---|---|---|---|---|---|
| CID22-A(25) | ssim2 | 206 over / 242 under | 484 over / 26 under | 281 over / 113 under | 131 over / 197 under | 237 over / 61 under | 486 over / 20 under |
| AIC-3 CTC (decoded PNG, source res.) | iwssim | 5 over / 386 under | 9 over / 85 under | 4 over / 97 under | 5 over / 92 under | 10 over / 77 under | 8 over / 139 under |
| AIC-4 crop (620×800, as shown) | cvvdp_fhd | 0 over / 310 under | 0 over / 162 under | 1 over / 151 under | 0 over / 188 under | 5 over / 143 under | 1 over / 198 under |
| AIC-4 full res. (secondary) | ssim2 | 0 over / 149 under | 5 over / 17 under | 1 over / 11 under | 1 over / 14 under | 3 over / 10 under | 2 over / 29 under |
| CSIQ JPEG+JPEG2000 | ssim2 | 52 over / 5 under | 8 over / 3 under | 4 over / 1 under | — | — | — |

