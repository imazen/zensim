# paper_holdout_2026-09-23 — peer metrics on the held-out human table

Devin-executed; Opus-reviewed 2026-09-23 (REVIEW_PAPER_MEASURE.md): PROMOTE WITH CORRECTIONS

Reviewer independently confirmed CID22-A(25) n=2,192/25 and SROCC B 0.8794, fast-ssim2 0.8736, DVours 0.8351, GMSD 0.8148, DVgate 0.7637; CSIQ GMSD 0.9570 vs fast-ssim2 0.9047; KonJND-504 |SROCC| GMSD 0.7842 vs fast-ssim2 0.5272.

## Purpose and protocol

Paper-fairness lane: score peer metrics (GMSD, three DVIFM-ish presets) on the
same corpora, pixels and statistics owner as the zensim board, against the two
references the board uses (fast-ssim2 = SSIMULACRA2 0.8.2, frozen zensim B).
Registered scope: CID22-A(25), CSIQ, AIC-3, AIC-4 (620x800 crops AND full
resolution), JPEG-AI SDR25, KonJND-1k JPEG-504.

- Statistics owner: `zen_stats.panel_batch` (`panel` binary) for the full panel
  (|SROCC| / signed / PLCC / KROCC); `benchmarks/ssim2_bar_2026-08-31/
  paired_perref_boot.py` (the ssim2-bar owner) for reference-clustered paired
  bootstrap — 10,000 draws, seed 20260901, unit = reference cluster.
- Pair lists: `build_pairs.py` in board row order; per-row reference + human
  label verified on join (never positional).
- Pixels: every arm scored the SAME decoded pixels; KonJND JPEG-504 distortions
  decoded once by zensim's decode owner (`verify_bitstream_decode`) to PNG;
  a raw-JPEG-path GMSD check gives identical scores (see Checks).
- Exposure: `EXPOSURE.json` (frozen-public-test-exposure-v1; CID22-B untouched,
  no secret holdouts read); ledger entries in docs/DATASET_HISTORY.md and
  docs/DATA_SPLITS.md (2026-09-23). All 4,292 CID22 pairs were scored for
  pixels (CID22-A subset = 2,192 pairs over 25 refs carries every statistic;
  no CID22-B statistic was computed. The 4,292-row `pairs/cid22.tsv` carries all 49 references' MCOS copied from the DVIFM-ish pair list; only the A(25) rows were correlated).
- Peer configs (best documented, no fitting here): GMSD = zenmetrics `gmsd`
  crate @9ecfb9d4 (libgmsd parity, float64, distance negated to quality).
  DVIFM-ish = our reimplementation of the published/talk DVIFM configuration
  (dvifmish @49aaf667, float path) in three presets: `talk-faithful-luma`
  (constants fitted on CID22-A + TID2013/KADID JPEG/JP2K subsets — therefore
  NOT scored on CID22), `ours-full-luma` (teacher-fitted on the 201 CID22
  *training* references), `serving-gate-ycbcr3` (joint-core-v1 TRAIN rows).
- Binaries: sha256 list in `BINARIES.sha256` (panel f11857c2, zenmetrics-gmsd
  319a9f2b, dvifmish-49aaf667 69cd2309, rev3_extractor 28cf588a,
  ensemble_score_rows 5b454d0f, verify_bitstream_decode dcba58ef, B by-id bake
  a96a5a66, preset JSONs).

## Headline findings (details in tables below)

- CSIQ: GMSD 0.9570 pooled beats B 0.9342 and ssim2 0.9047 (CI-separated:
  +0.052 [+0.044,+0.060] vs ssim2; +0.023 [+0.012,+0.034] vs B). Peer win.
- KonJND: GMSD 0.7842 |SROCC| vs ssim2 0.5272 (+0.257 [+0.213,+0.304]) and B
  0.5194 (+0.265 [+0.219,+0.313]) — the largest peer win in the lane.
- CID22-A(25): B 0.8794 > ssim2 0.8736 > DVours 0.8351 > GMSD 0.8148 > DVgate
  0.7637 (B−ssim2 not CI-separated; GMSD and DVgate are clearly below ssim2).
- AIC-3: ssim2 0.7970 > B 0.7650 (CI-separated −0.032) — a peer win over B on
  the board's own frozen row.
- AIC-4 crops / full: DVours 0.9232 / 0.9156 — top scored peer arm on both
  renderings (vs ssim2 0.9127/0.9054, B 0.8906/0.8947); DVours−B is
  CI-separated on crops but not full resolution in the pooled result (+0.032 [+0.020,+0.046], +0.022 [−0.001,+0.041];
  within +0.030/+0.026 both CI-separated). cvvdp_fhd ctx arm reaches 0.9609 on
  the crop rendering (context, not a peer-comparison row).
- SDR25: GMSD 0.9794 and DVours 0.9762 above ssim2 0.9580/B 0.9554 pooled, but
  per-reference within-image saturates (all arms = 1.0000) — pooled deltas are
  not CI-separated; treat as a near-ceiling corpus.
- B's frozen numbers here are its serving configuration; it is not a
  correlation-tuned metric — design-scope caveat applies to every arm equally.

## Checks (all pass)

- DVIFM-ish orientation sanity: SROCC(quality) − SROCC(−E) = 0.0 on every
  corpus × preset (monotone transform, expected).
- B era cross-check on AIC-4 crop: board 0.89063434 vs Rev3-era re-score
  0.89064501; between-vector SROCC 0.999924 (eras agree).
- GMSD KonJND decoder check: PNG-decoded vs raw-JPEG-path scores — signed
  SROCC identical (−0.7841837 both), max |Δscore| = 0.0.
- Bootstrap-owner regression: the 2026-09-23 owner edits produce
  byte-identical output to the committed owner (ffc14647) on CSIQ at
  BOOT=2000 (see `boot/regression_csiq_{orig,edited}.txt`).
- Within-image axis is unavailable on KonJND (504 pairs over 504 references —
  one pair per reference; the owner skips it).

## KonJND convention and caveat

Signed SROCC on the PJND axis is NEGATIVE for every quality-oriented arm
(e.g. GMSD-quality −0.7842, ssim2 −0.5272): in this table's encoding the
per-reference target (JPEG quality at the 50% visibility threshold) increases
as the measured distortion becomes more visible, so the axis behaves as a
distortion/visibility axis. The headline reports |SROCC| per board convention;
signed values are in Table H4. KonJND measures visibility thresholds, not
quality ranking — the caveat applies equally to all arms.

## Table H1 — pooled |SROCC| with reference-clustered 95% CI (10,000 draws, seed 20260901)

| corpus (n / refs) | fast-ssim2 (SSIMULACRA2) | zensim B (frozen) | GMSD | DVIFM-ish talk-faithful-luma | DVIFM-ish ours-full-luma | DVIFM-ish serving-gate-ycbcr3 |
|---|---|---|---|---|---|---|
| CID22-A(25) human (MCOS) (2192 / 25) | 0.8736 [0.8322, 0.9176] | 0.8794 [0.8400, 0.9210] | 0.8148 [0.7776, 0.8526] | not scored | 0.8351 [0.7829, 0.8824] | 0.7637 [0.6665, 0.8606] |
| CSIQ (866 / 30) | 0.9047 [0.8933, 0.9157] | 0.9342 [0.9236, 0.9434] | 0.9570 [0.9498, 0.9631] | 0.8526 [0.8393, 0.8671] | 0.9019 [0.8912, 0.9122] | 0.9165 [0.9084, 0.9246] |
| AIC-3 CTC (EPFL, full resolution as distributed) (600 / 10) | 0.7970 [0.6918, 0.9105] | 0.7650 [0.6730, 0.8761] | 0.7830 [0.6932, 0.8946] | 0.7106 [0.6153, 0.8478] | 0.7656 [0.6746, 0.9017] | 0.7353 [0.6667, 0.8487] |
| AIC-4 sample, 620x800 PTC crops (300 / 5) | 0.9127 [0.8775, 0.9594] | 0.8906 [0.8607, 0.9423] | 0.9094 [0.8789, 0.9747] | 0.8524 [0.7782, 0.9327] | 0.9232 [0.8881, 0.9635] | 0.8343 [0.7871, 0.9393] |
| AIC-4 sample, full resolution (300 / 5) | 0.9054 [0.8754, 0.9616] | 0.8947 [0.8623, 0.9453] | 0.8992 [0.8648, 0.9656] | 0.7948 [0.7261, 0.9295] | 0.9156 [0.8932, 0.9666] | 0.8499 [0.8147, 0.9483] |
| JPEG-AI SDR25 (50 / 5) | 0.9580 [0.9341, 0.9928] | 0.9554 [0.9403, 0.9909] | 0.9794 [0.9714, 0.9928] | 0.9355 [0.9020, 0.9964] | 0.9762 [0.9644, 0.9971] | 0.9139 [0.8587, 0.9854] |
| KonJND-1k JPEG-504 (504 / 504) | 0.5272 [0.4564, 0.5911] | 0.5194 [0.4502, 0.5810] | 0.7842 [0.7393, 0.8205] | 0.5836 [0.5157, 0.6441] | 0.5851 [0.5179, 0.6448] | 0.3307 [0.2469, 0.4097] |

## Table H2 — within-image (mean per-reference |SROCC|), reference-clustered 95% CI

| corpus | fast-ssim2 (SSIMULACRA2) | zensim B (frozen) | GMSD | DVIFM-ish talk-faithful-luma | DVIFM-ish ours-full-luma | DVIFM-ish serving-gate-ycbcr3 |
|---|---|---|---|---|---|---|
| CID22-A(25) human (MCOS) | 0.9642 [0.9553, 0.9721] | 0.9591 [0.9456, 0.9699] | 0.8767 [0.8585, 0.8938] | not scored | 0.9226 [0.9097, 0.9345] | 0.9317 [0.9204, 0.9421] |
| CSIQ | 0.9084 [0.8981, 0.9181] | 0.9320 [0.9199, 0.9430] | 0.9586 [0.9528, 0.9635] | 0.8615 [0.8499, 0.8735] | 0.9035 [0.8936, 0.9129] | 0.9214 [0.9135, 0.9295] |
| AIC-3 CTC (EPFL, full resolution as distributed) | 0.9521 [0.9332, 0.9690] | 0.9183 [0.8836, 0.9512] | 0.9431 [0.9297, 0.9556] | 0.9244 [0.9016, 0.9452] | 0.9512 [0.9339, 0.9670] | 0.9353 [0.9140, 0.9548] |
| AIC-4 sample, 620x800 PTC crops | 0.9405 [0.9084, 0.9697] | 0.9114 [0.8826, 0.9402] | 0.9606 [0.9357, 0.9805] | 0.9086 [0.8837, 0.9336] | 0.9415 [0.9070, 0.9671] | 0.9218 [0.8738, 0.9527] |
| AIC-4 sample, full resolution | 0.9358 [0.9043, 0.9673] | 0.9101 [0.8816, 0.9386] | 0.9523 [0.9167, 0.9762] | 0.9126 [0.8902, 0.9351] | 0.9360 [0.9059, 0.9611] | 0.9218 [0.8809, 0.9527] |
| JPEG-AI SDR25 | 1.0000 [1.0000, 1.0000] | 0.9976 [0.9927, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| KonJND-1k JPEG-504 | one row per reference | one row per reference | one row per reference | one row per reference | one row per reference | one row per reference |

## Table H3a — paired Δ vs fast-ssim2 (candidate − reference; ✓ CI above 0, ✗ CI below 0)

| corpus | axis | zensim B (frozen) | GMSD | DVIFM-ish talk-faithful-luma | DVIFM-ish ours-full-luma | DVIFM-ish serving-gate-ycbcr3 |
|---|---|---|---|---|---|---|
| cid22a | pooled | +0.0053 [-0.0156, +0.0279] | -0.0597 [-0.0858, -0.0326] ✗ | not scored | -0.0387 [-0.0938, +0.0140] | -0.1063 [-0.1770, -0.0436] ✗ |
| cid22a | within | -0.0051 [-0.0136, +0.0020] | -0.0875 [-0.1015, -0.0741] ✗ | not scored | -0.0416 [-0.0527, -0.0298] ✗ | -0.0325 [-0.0432, -0.0220] ✗ |
| csiq | pooled | +0.0293 [+0.0146, +0.0440] ✓ | +0.0522 [+0.0442, +0.0601] ✓ | -0.0520 [-0.0598, -0.0439] ✗ | -0.0029 [-0.0101, +0.0047] | +0.0118 [+0.0031, +0.0207] ✓ |
| csiq | within | +0.0236 [+0.0081, +0.0395] ✓ | +0.0502 [+0.0410, +0.0604] ✓ | -0.0469 [-0.0540, -0.0400] ✗ | -0.0049 [-0.0125, +0.0033] | +0.0130 [+0.0054, +0.0207] ✓ |
| aic3 | pooled | -0.0320 [-0.0614, -0.0025] ✗ | -0.0136 [-0.0685, +0.0303] | -0.0805 [-0.1369, -0.0202] ✗ | -0.0282 [-0.0820, +0.0093] | -0.0571 [-0.1428, +0.0233] |
| aic3 | within | -0.0338 [-0.0510, -0.0166] ✗ | -0.0090 [-0.0319, +0.0137] | -0.0277 [-0.0475, -0.0100] ✗ | -0.0009 [-0.0093, +0.0081] | -0.0168 [-0.0243, -0.0084] ✗ |
| aic4 | pooled | -0.0235 [-0.0435, -0.0069] ✗ | +0.0009 [-0.0304, +0.0269] | -0.0557 [-0.1366, -0.0009] ✗ | +0.0083 [-0.0036, +0.0187] | -0.0667 [-0.1010, -0.0201] ✗ |
| aic4 | within | -0.0291 [-0.0406, -0.0201] ✗ | +0.0201 [+0.0015, +0.0363] ✓ | -0.0319 [-0.0401, -0.0239] ✗ | +0.0010 [-0.0065, +0.0095] | -0.0187 [-0.0394, -0.0007] ✗ |
| aic4full | pooled | -0.0137 [-0.0471, +0.0096] | -0.0021 [-0.0356, +0.0309] | -0.0945 [-0.1826, -0.0260] ✗ | +0.0080 [-0.0122, +0.0261] | -0.0476 [-0.0877, -0.0135] ✗ |
| aic4full | within | -0.0257 [-0.0476, -0.0013] ✗ | +0.0165 [-0.0060, +0.0439] | -0.0231 [-0.0395, -0.0080] ✗ | +0.0002 [-0.0079, +0.0141] | -0.0139 [-0.0311, +0.0043] |
| sdr25 | pooled | -0.0034 [-0.0259, +0.0136] | +0.0166 [-0.0057, +0.0572] | -0.0181 [-0.0731, +0.0590] | +0.0140 [-0.0265, +0.0494] | -0.0352 [-0.0849, +0.0029] |
| sdr25 | within | -0.0024 [-0.0073, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| konjnd | pooled | -0.0077 [-0.0418, +0.0251] | +0.2572 [+0.2131, +0.3042] ✓ | +0.0567 [-0.0158, +0.1288] | +0.0578 [+0.0086, +0.1064] ✓ | -0.1966 [-0.2704, -0.1252] ✗ |

## Table H3b — paired Δ vs B (candidate − reference; ✓ CI above 0, ✗ CI below 0)

| corpus | axis | fast-ssim2 (SSIMULACRA2) | GMSD | DVIFM-ish talk-faithful-luma | DVIFM-ish ours-full-luma | DVIFM-ish serving-gate-ycbcr3 |
|---|---|---|---|---|---|---|
| cid22a | pooled | -0.0053 [-0.0279, +0.0156] | -0.0650 [-0.0986, -0.0305] ✗ | not scored | -0.0440 [-0.0986, +0.0113] | -0.1116 [-0.1971, -0.0410] ✗ |
| cid22a | within | +0.0051 [-0.0020, +0.0136] | -0.0824 [-0.0990, -0.0668] ✗ | not scored | -0.0364 [-0.0489, -0.0222] ✗ | -0.0274 [-0.0416, -0.0130] ✗ |
| csiq | pooled | -0.0293 [-0.0440, -0.0146] ✗ | +0.0228 [+0.0121, +0.0344] ✓ | -0.0813 [-0.0986, -0.0632] ✗ | -0.0322 [-0.0460, -0.0182] ✗ | -0.0175 [-0.0303, -0.0038] ✗ |
| csiq | within | -0.0236 [-0.0395, -0.0081] ✗ | +0.0266 [+0.0161, +0.0375] ✓ | -0.0705 [-0.0870, -0.0536] ✗ | -0.0285 [-0.0420, -0.0146] ✗ | -0.0106 [-0.0254, +0.0049] |
| aic3 | pooled | +0.0320 [+0.0025, +0.0614] ✓ | +0.0184 [-0.0296, +0.0645] | -0.0485 [-0.0966, +0.0002] | +0.0038 [-0.0459, +0.0475] | -0.0251 [-0.0977, +0.0459] |
| aic3 | within | +0.0338 [+0.0166, +0.0510] ✓ | +0.0248 [-0.0126, +0.0637] | +0.0061 [-0.0260, +0.0351] | +0.0329 [+0.0116, +0.0542] ✓ | +0.0170 [+0.0002, +0.0338] ✓ |
| aic4 | pooled | +0.0235 [+0.0069, +0.0435] ✓ | +0.0244 [-0.0027, +0.0535] | -0.0322 [-0.0936, +0.0099] | +0.0318 [+0.0200, +0.0456] ✓ | -0.0432 [-0.0795, +0.0044] |
| aic4 | within | +0.0291 [+0.0201, +0.0406] ✓ | +0.0492 [+0.0266, +0.0719] ✓ | -0.0028 [-0.0081, +0.0025] | +0.0301 [+0.0148, +0.0453] ✓ | +0.0104 [-0.0157, +0.0331] |
| aic4full | pooled | +0.0137 [-0.0096, +0.0471] | +0.0116 [-0.0430, +0.0619] | -0.0808 [-0.1515, +0.0019] | +0.0217 [-0.0008, +0.0414] | -0.0339 [-0.0899, +0.0162] |
| aic4full | within | +0.0257 [+0.0013, +0.0476] ✓ | +0.0422 [+0.0026, +0.0824] ✓ | +0.0026 [-0.0219, +0.0307] | +0.0259 [-0.0021, +0.0516] | +0.0117 [-0.0266, +0.0425] |
| sdr25 | pooled | +0.0034 [-0.0136, +0.0259] | +0.0201 [-0.0091, +0.0507] | -0.0147 [-0.0774, +0.0541] | +0.0175 [-0.0135, +0.0487] | -0.0318 [-0.0892, +0.0157] |
| sdr25 | within | +0.0024 [+0.0000, +0.0073] | +0.0024 [+0.0000, +0.0073] | +0.0024 [+0.0000, +0.0073] | +0.0024 [+0.0000, +0.0073] | +0.0024 [+0.0000, +0.0073] |
| konjnd | pooled | +0.0077 [-0.0251, +0.0418] | +0.2649 [+0.2191, +0.3134] ✓ | +0.0645 [-0.0034, +0.1332] | +0.0655 [+0.0182, +0.1121] ✓ | -0.1888 [-0.2607, -0.1168] ✗ |

## Table H4 — full panel (canonical zenstats panel): |SROCC| / signed SROCC / PLCC / KROCC

| corpus | arm | |SROCC| | signed | PLCC | KROCC | source |
|---|---|---|---|---|---|---|
| cid22a | fast-ssim2 (SSIMULACRA2) | 0.8736 | +0.8736 | 0.8635 | 0.6827 | `board table` |
| cid22a | zensim B (frozen) | 0.8794 | +0.8794 | 0.8723 | 0.6919 | `pp_B_cid22.tsv` |
| cid22a | GMSD | 0.8148 | +0.8148 | 0.8154 | 0.6133 | `gmsd_cid22a.parquet:gmsd_cpu_imazen_v0_1_0 (negated)` |
| cid22a | DVIFM-ish ours-full-luma | 0.8351 | +0.8351 | 0.8325 | 0.6423 | `ours-full-luma.tsv:quality` |
| cid22a | DVIFM-ish serving-gate-ycbcr3 | 0.7637 | +0.7637 | 0.7568 | 0.5676 | `serving-gate-ycbcr3.tsv:quality` |
| csiq | fast-ssim2 (SSIMULACRA2) | 0.9047 | +0.9047 | 0.8877 | 0.7327 | `board table` |
| csiq | zensim B (frozen) | 0.9342 | +0.9342 | 0.9109 | 0.7710 | `pp_B_csiq.tsv` |
| csiq | GMSD | 0.9570 | +0.9570 | 0.9525 | 0.8122 | `gmsd_csiq.parquet:gmsd_cpu_imazen_v0_1_0 (negated)` |
| csiq | DVIFM-ish talk-faithful-luma | 0.8526 | +0.8526 | 0.8470 | 0.6598 | `talk-faithful-luma.tsv:quality` |
| csiq | DVIFM-ish ours-full-luma | 0.9019 | +0.9019 | 0.8866 | 0.7275 | `ours-full-luma.tsv:quality` |
| csiq | DVIFM-ish serving-gate-ycbcr3 | 0.9165 | +0.9165 | 0.9083 | 0.7423 | `serving-gate-ycbcr3.tsv:quality` |
| csiq | ctx:butteraugli_max | 0.8441 | +0.8441 | 0.8315 | 0.6646 | `csiq_butteraugli_gpu.tsv:butteraugli_max_gpu (negated)` |
| csiq | ctx:butteraugli_p3 | 0.8146 | +0.8146 | 0.8087 | 0.6360 | `csiq_butteraugli_gpu.tsv:butteraugli_pnorm3_gpu (negated)` |
| csiq | ctx:iwssim | 0.9215 | +0.9215 | 0.9028 | 0.7526 | `csiq_iwssim_gpu.tsv:iwssim_gpu` |
| csiq | ctx:cvvdp_4k | 0.8959 | +0.8959 | 0.8830 | 0.7275 | `csiq_cvvdp.tsv:cvvdp_cpu_imazen_v0_1_0` |
| aic3 | fast-ssim2 (SSIMULACRA2) | 0.7970 | +0.7970 | 0.8091 | 0.6294 | `board table` |
| aic3 | zensim B (frozen) | 0.7650 | +0.7650 | 0.7750 | 0.5911 | `pp_B_aic3.tsv` |
| aic3 | GMSD | 0.7830 | +0.7830 | 0.7933 | 0.6128 | `gmsd_aic3.parquet:gmsd_cpu_imazen_v0_1_0 (negated)` |
| aic3 | DVIFM-ish talk-faithful-luma | 0.7106 | +0.7106 | 0.7263 | 0.5419 | `talk-faithful-luma.tsv:quality` |
| aic3 | DVIFM-ish ours-full-luma | 0.7656 | +0.7656 | 0.7836 | 0.5994 | `ours-full-luma.tsv:quality` |
| aic3 | DVIFM-ish serving-gate-ycbcr3 | 0.7353 | +0.7353 | 0.7460 | 0.5633 | `serving-gate-ycbcr3.tsv:quality` |
| aic3 | ctx:butteraugli_max | 0.7074 | +0.7074 | 0.7265 | 0.5413 | `aic3_butteraugli_heldout.tsv:butteraugli_max_gpu (negated)` |
| aic3 | ctx:butteraugli_p3 | 0.7571 | +0.7571 | 0.7667 | 0.5840 | `aic3_butteraugli_heldout.tsv:butteraugli_pnorm3_gpu (negated)` |
| aic3 | ctx:iwssim | 0.7735 | +0.7735 | 0.7907 | 0.6064 | `aic3_iwssim_heldout.tsv:iwssim_gpu` |
| aic3 | ctx:cvvdp_4k | 0.7918 | +0.7918 | 0.8034 | 0.6256 | `aic3_cvvdp_heldout.tsv:cvvdp_cpu_imazen_v0_1_0` |
| aic4 | fast-ssim2 (SSIMULACRA2) | 0.9127 | -0.9127 | 0.9001 | 0.7453 | `board table` |
| aic4 | zensim B (frozen) | 0.8906 | -0.8906 | 0.8795 | 0.7081 | `b_sdr_linear_cid80_inclwinsor_dense_dial@cur372.fulleval.json per_pair.aic4[:300]` |
| aic4 | GMSD | 0.9094 | -0.9094 | 0.8987 | 0.7450 | `gmsd_aic4.parquet:gmsd_cpu_imazen_v0_1_0 (negated)` |
| aic4 | DVIFM-ish talk-faithful-luma | 0.8524 | -0.8524 | 0.8424 | 0.6690 | `talk-faithful-luma.tsv:quality` |
| aic4 | DVIFM-ish ours-full-luma | 0.9232 | -0.9232 | 0.9079 | 0.7672 | `ours-full-luma.tsv:quality` |
| aic4 | DVIFM-ish serving-gate-ycbcr3 | 0.8343 | -0.8343 | 0.8229 | 0.6464 | `serving-gate-ycbcr3.tsv:quality` |
| aic4 | ctx:butteraugli_max | 0.8656 | -0.8656 | 0.8522 | 0.6864 | `aic4_butteraugli_gpu.tsv:butteraugli_max_gpu (negated)` |
| aic4 | ctx:butteraugli_p3 | 0.8969 | -0.8969 | 0.8834 | 0.7263 | `aic4_butteraugli_gpu.tsv:butteraugli_pnorm3_gpu (negated)` |
| aic4 | ctx:iwssim | 0.9533 | -0.9533 | 0.9522 | 0.8203 | `aic4_iwssim_gpu.tsv:iwssim_gpu` |
| aic4 | ctx:cvvdp_4k | 0.8906 | -0.8906 | 0.8797 | 0.7321 | `aic4_cvvdp.tsv:cvvdp_cpu_imazen_v0_1_0` |
| aic4 | ctx:cvvdp_fhd | 0.9609 | -0.9609 | 0.9599 | 0.8410 | `aic4_cvvdp_standard_fhd.tsv:cvvdp_cpu_imazen_v0_1_0_standard_fhd` |
| aic4full | fast-ssim2 (SSIMULACRA2) | 0.9054 | -0.9054 | 0.8931 | 0.7383 | `fast-ssim2 extractor audit (full resolution)` |
| aic4full | zensim B (frozen) | 0.8947 | -0.8947 | 0.8826 | 0.7160 | `zensim_b_aic4_full.tsv` |
| aic4full | GMSD | 0.8992 | -0.8992 | 0.8889 | 0.7298 | `gmsd_aic4full.parquet:gmsd_cpu_imazen_v0_1_0 (negated)` |
| aic4full | DVIFM-ish talk-faithful-luma | 0.7948 | -0.7948 | 0.7869 | 0.6087 | `talk-faithful-luma.tsv:quality` |
| aic4full | DVIFM-ish ours-full-luma | 0.9156 | -0.9156 | 0.9029 | 0.7565 | `ours-full-luma.tsv:quality` |
| aic4full | DVIFM-ish serving-gate-ycbcr3 | 0.8499 | -0.8499 | 0.8391 | 0.6653 | `serving-gate-ycbcr3.tsv:quality` |
| sdr25 | fast-ssim2 (SSIMULACRA2) | 0.9580 | -0.9580 | 0.9753 | 0.8384 | `board table` |
| sdr25 | zensim B (frozen) | 0.9554 | -0.9554 | 0.9793 | 0.8302 | `zensim_b.tsv` |
| sdr25 | GMSD | 0.9794 | -0.9794 | 0.9844 | 0.8743 | `gmsd_sdr25.parquet:gmsd_cpu_imazen_v0_1_0 (negated)` |
| sdr25 | DVIFM-ish talk-faithful-luma | 0.9355 | -0.9355 | 0.9535 | 0.8057 | `talk-faithful-luma.tsv:quality` |
| sdr25 | DVIFM-ish ours-full-luma | 0.9762 | -0.9762 | 0.9769 | 0.8727 | `ours-full-luma.tsv:quality` |
| sdr25 | DVIFM-ish serving-gate-ycbcr3 | 0.9139 | -0.9139 | 0.9318 | 0.7551 | `serving-gate-ycbcr3.tsv:quality` |
| sdr25 | ctx:butteraugli_max | 0.8845 | -0.8845 | 0.9435 | 0.7241 | `sdr25_butteraugli_gpu.tsv:butteraugli_max_gpu (negated)` |
| sdr25 | ctx:butteraugli_p3 | 0.9760 | -0.9760 | 0.9876 | 0.8792 | `sdr25_butteraugli_gpu.tsv:butteraugli_pnorm3_gpu (negated)` |
| sdr25 | ctx:iwssim | 0.9496 | -0.9496 | 0.9618 | 0.8041 | `sdr25_iwssim_gpu.tsv:iwssim_gpu` |
| sdr25 | ctx:cvvdp_4k | 0.8609 | -0.8609 | 0.8672 | 0.7306 | `sdr25_cvvdp.tsv:cvvdp_cpu_imazen_v0_1_0` |
| konjnd | fast-ssim2 (SSIMULACRA2) | 0.5272 | -0.5272 | 0.5601 | 0.3703 | `board table` |
| konjnd | zensim B (frozen) | 0.5194 | -0.5194 | 0.5356 | 0.3579 | `b_sdr_linear_cid80_inclwinsor_dense_dial@cur372.fulleval.json per_pair.konjnd[:504]` |
| konjnd | GMSD | 0.7842 | -0.7842 | 0.8290 | 0.5983 | `gmsd_konjnd.parquet:gmsd_cpu_imazen_v0_1_0 (negated)` |
| konjnd | DVIFM-ish talk-faithful-luma | 0.5836 | -0.5836 | 0.6655 | 0.4157 | `talk-faithful-luma.tsv:quality` |
| konjnd | DVIFM-ish ours-full-luma | 0.5851 | -0.5851 | 0.6017 | 0.4182 | `ours-full-luma.tsv:quality` |
| konjnd | DVIFM-ish serving-gate-ycbcr3 | 0.3307 | -0.3307 | 0.3421 | 0.2314 | `serving-gate-ycbcr3.tsv:quality` |
| konjnd | ctx:butteraugli_max | 0.2586 | +0.2586 | 0.4510 | 0.1677 | `konjnd_butteraugli_heldout.tsv:butteraugli_max_gpu (negated)` |
| konjnd | ctx:butteraugli_p3 | 0.0941 | -0.0941 | 0.2481 | 0.0798 | `konjnd_butteraugli_heldout.tsv:butteraugli_pnorm3_gpu (negated)` |
| konjnd | ctx:iwssim | 0.5704 | -0.5704 | 0.7174 | 0.4102 | `konjnd_iwssim_heldout.tsv:iwssim_gpu` |
| konjnd | ctx:cvvdp_4k | 0.0562 | -0.0562 | 0.1628 | 0.0427 | `konjnd_cvvdp_heldout.tsv:cvvdp_cpu_imazen_v0_1_0` |
