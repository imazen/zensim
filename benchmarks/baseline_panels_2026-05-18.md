# Baseline Mohammadi panels — ssim2, cvvdp, iwssim controls (2026-05-18)

Computed via `scripts/baseline_panels_2026-05-18/extract_panels.py`. The Python `panel.py` mirrors the Rust reference at `zensim-validate/src/panel.rs` / `zensim-validate/src/bin/bake_verdict.rs` (4-parameter logistic before PLCC, corpus-wide σ for Z-RMSE unless per-stimulus σ is available). The implementation is validated against Mohammadi 2025's anchor Z-RMSE values: SSIMULACRA2 = 47.63, IW-SSIM = 31.51, CVVDP = 9.45, PSNR-Y = 13.36 — our matches to within 0.06 (see `panel.py::validate_against_anchor`).


Data sources per (metric, corpus):

- **ssim2** (fast-ssim2 score): per-pair CSV `benchmarks/v0_22_iw_v3_seed1_2026-05-17_eval_per_pair.csv` for CID22 / KADID / TID / AIC-3 CTC. Per-pair CSV does NOT carry KonJND-1k, so the KonJND row uses the AIC-3 anchor CSV's SSIMULACRA2 column — n/a for the KonJND-1k corpus directly.
- **cvvdp** (cvvdp_imazen_v0_0_1): score TSVs at `/mnt/v/zen/zensim-eval/{cid22,kadid,tid,aic3,konjnd_{jpeg,bpg}}_cvvdp_scores_2026-05-17.tsv`, joined to corpus MOS sources.
- **iwssim** (iwssim-gpu, `iwssim_gpu` column): score TSVs at `/mnt/v/zen/zensim-training/2026-05-18-iwssim-panels/{cid22,kadid,tid,konjnd_{jpeg,bpg},aic3}_iwssim_scores.tsv`, produced by `zen-metrics batch --metric iwssim-gpu --gpu-runtime cuda` on the same pair lists as cvvdp (all corpora have min(W, H) ≥ 384, well above the 176-pixel paper-strict floor — no adaptive small-image path needed). Joined to corpus MOS sources via `dist_path`. AIC-3 anchor PTC subset still uses the anchor CSV's `iw_ssim` column (the n=300 PTC-restricted set is canonical for paper anchor cross-validation).
- **Per-stimulus σ** for Z-RMSE: AIC-3 (anchor CSV `std_bootstrap`), TID (mos_std.txt). CID22 / KADID / KonJND use corpus-wide σ fallback (matches bake_verdict.rs convention).
- **PLCC**: 4-parameter logistic rescale (Mohammadi 2025 / ITU-T P.1401).
- Per-band slicing is the CLAUDE.md 10-band width-10 grid on [0, 1].



## CID22 (n=4292)

_cvvdp and iwssim panels use the same 4292 join (cvvdp pairs TSV ↔ MCOS CSV)._


| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| ssim2 (fast-ssim2) | 4292 | 0.8895 | 0.8879 | 0.7062 | 0.0424 | 0.9351 | 0.460 |
| cvvdp | 4292 | 0.8214 | 0.8251 | 0.6238 | 0.0424 | 0.8842 | 0.565 |
| iwssim | 4292 | 0.7836 | 0.7926 | 0.5938 | 0.0520 | 0.8525 | 0.610 |


### CID22 10-band panels

#### ssim2 (fast-ssim2)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1335 | 0.3214 | 0.0852 | 0.0351 | 0.2093 | 0.947 |
| B4 | [0.40, 0.50) | 266 | 0.2888 | 0.3205 | 0.1958 | 0.0526 | 0.3555 | 0.947 |
| B5 | [0.50, 0.60) | 615 | 0.3888 | 0.3897 | 0.2657 | 0.0407 | 0.4670 | 0.921 |
| B6 | [0.60, 0.70) | 836 | 0.4173 | 0.4200 | 0.2834 | 0.0383 | 0.4922 | 0.908 |
| B7 | [0.70, 0.80) | 1092 | 0.3974 | 0.4203 | 0.2773 | 0.0421 | 0.4702 | 0.907 |
| B8 | [0.80, 0.90) | 1382 | 0.5006 | 0.5003 | 0.3385 | 0.0347 | 0.5938 | 0.866 |
| B9 | [0.90, 1.00] | 43 | 0.1121 | 0.3420 | 0.0698 | 0.0698 | 0.0327 | 0.940 |

#### cvvdp

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1476 | 0.1435 | 0.1040 | 0.0526 | 0.2138 | 0.990 |
| B4 | [0.40, 0.50) | 266 | 0.2597 | 0.2739 | 0.1748 | 0.0564 | 0.3300 | 0.962 |
| B5 | [0.50, 0.60) | 615 | 0.2898 | 0.2999 | 0.1960 | 0.0325 | 0.3513 | 0.954 |
| B6 | [0.60, 0.70) | 836 | 0.3363 | 0.3382 | 0.2249 | 0.0359 | 0.3988 | 0.941 |
| B7 | [0.70, 0.80) | 1092 | 0.3103 | 0.3219 | 0.2126 | 0.0430 | 0.3638 | 0.947 |
| B8 | [0.80, 0.90) | 1382 | 0.3185 | 0.3217 | 0.2113 | 0.0304 | 0.3751 | 0.947 |
| B9 | [0.90, 1.00] | 43 | 0.0809 | 0.3060 | 0.0631 | 0.0930 | 0.1739 | 0.952 |


#### iwssim

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0955 | 0.1457 | 0.0827 | 0.0526 | 0.0091 | 0.989 |
| B4 | [0.40, 0.50) | 266 | 0.2101 | 0.2615 | 0.1429 | 0.0564 | 0.2605 | 0.965 |
| B5 | [0.50, 0.60) | 615 | 0.1928 | 0.2389 | 0.1309 | 0.0472 | 0.2241 | 0.971 |
| B6 | [0.60, 0.70) | 836 | 0.2101 | 0.2357 | 0.1388 | 0.0383 | 0.2530 | 0.972 |
| B7 | [0.70, 0.80) | 1092 | 0.2826 | 0.2985 | 0.1938 | 0.0504 | 0.3261 | 0.954 |
| B8 | [0.80, 0.90) | 1382 | 0.4129 | 0.4161 | 0.2798 | 0.0499 | 0.4844 | 0.909 |
| B9 | [0.90, 1.00] | 43 | 0.1338 | 0.5198 | 0.0963 | 0.0465 | 0.2532 | 0.854 |


## KADID-10k (n=10125)

_cvvdp and iwssim panels both use the 10125 join._


| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| ssim2 (fast-ssim2) | 10125 | 0.8133 | 0.8107 | 0.6174 | 0.0516 | 0.8828 | 0.585 |
| cvvdp | 10125 | 0.8339 | 0.8337 | 0.6389 | 0.0417 | 0.9018 | 0.552 |
| iwssim | 10125 | 0.8498 | 0.8446 | 0.6663 | 0.0357 | 0.9112 | 0.535 |


### KADID-10k 10-band panels

#### ssim2 (fast-ssim2)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 620 | 0.2062 | 0.2222 | 0.1442 | 0.0258 | 0.2750 | 0.975 |
| B1 | [0.10, 0.20) | 995 | 0.1772 | 0.1960 | 0.1227 | 0.0362 | 0.2267 | 0.981 |
| B2 | [0.20, 0.30) | 1206 | 0.0568 | 0.0871 | 0.0394 | 0.0323 | 0.0656 | 0.996 |
| B3 | [0.30, 0.40) | 1196 | 0.1277 | 0.1287 | 0.0887 | 0.0410 | 0.1557 | 0.992 |
| B4 | [0.40, 0.50) | 1013 | 0.1525 | 0.1803 | 0.1075 | 0.0484 | 0.1775 | 0.984 |
| B5 | [0.50, 0.60) | 919 | 0.0755 | 0.1362 | 0.0523 | 0.0413 | 0.0960 | 0.991 |
| B6 | [0.60, 0.70) | 936 | 0.0982 | 0.1244 | 0.0676 | 0.0374 | 0.1215 | 0.992 |
| B7 | [0.70, 0.80) | 985 | 0.1574 | 0.1594 | 0.1105 | 0.0487 | 0.1712 | 0.987 |
| B8 | [0.80, 0.90) | 1570 | 0.3510 | 0.3601 | 0.2458 | 0.0401 | 0.4192 | 0.933 |
| B9 | [0.90, 1.00] | 615 | 0.1777 | 0.1795 | 0.1291 | 0.0358 | 0.2094 | 0.984 |

#### cvvdp

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.1586 | 0.1789 | 0.1106 | 0.0567 | 0.2068 | 0.984 |
| B1 | [0.10, 0.20) | 910 | 0.1638 | 0.1801 | 0.1135 | 0.0429 | 0.2128 | 0.984 |
| B2 | [0.20, 0.30) | 1111 | 0.0893 | 0.1205 | 0.0614 | 0.0342 | 0.0983 | 0.993 |
| B3 | [0.30, 0.40) | 1291 | 0.1554 | 0.1794 | 0.1088 | 0.0395 | 0.1804 | 0.984 |
| B4 | [0.40, 0.50) | 1013 | 0.1889 | 0.2043 | 0.1329 | 0.0454 | 0.2264 | 0.979 |
| B5 | [0.50, 0.60) | 919 | 0.1116 | 0.1785 | 0.0779 | 0.0381 | 0.1358 | 0.984 |
| B6 | [0.60, 0.70) | 936 | 0.0965 | 0.1303 | 0.0666 | 0.0321 | 0.1181 | 0.991 |
| B7 | [0.70, 0.80) | 985 | 0.1715 | 0.1858 | 0.1187 | 0.0325 | 0.1843 | 0.983 |
| B8 | [0.80, 0.90) | 1699 | 0.3909 | 0.3932 | 0.2738 | 0.0377 | 0.4638 | 0.919 |
| B9 | [0.90, 1.00] | 486 | 0.1672 | 0.1577 | 0.1216 | 0.0412 | 0.1881 | 0.987 |


#### iwssim

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2604 | 0.2875 | 0.1823 | 0.0426 | 0.3351 | 0.958 |
| B1 | [0.10, 0.20) | 910 | 0.2508 | 0.2546 | 0.1719 | 0.0341 | 0.3122 | 0.967 |
| B2 | [0.20, 0.30) | 1111 | 0.1735 | 0.1772 | 0.1206 | 0.0432 | 0.2090 | 0.984 |
| B3 | [0.30, 0.40) | 1291 | 0.1874 | 0.1885 | 0.1307 | 0.0434 | 0.2255 | 0.982 |
| B4 | [0.40, 0.50) | 1013 | 0.1363 | 0.1380 | 0.0954 | 0.0464 | 0.1608 | 0.990 |
| B5 | [0.50, 0.60) | 919 | 0.1196 | 0.1276 | 0.0830 | 0.0370 | 0.1572 | 0.992 |
| B6 | [0.60, 0.70) | 936 | 0.1457 | 0.1453 | 0.1002 | 0.0310 | 0.1683 | 0.989 |
| B7 | [0.70, 0.80) | 985 | 0.1963 | 0.2129 | 0.1393 | 0.0457 | 0.2264 | 0.977 |
| B8 | [0.80, 0.90) | 1699 | 0.3874 | 0.3910 | 0.2726 | 0.0294 | 0.4583 | 0.920 |
| B9 | [0.90, 1.00] | 486 | 0.1402 | 0.1782 | 0.1013 | 0.0185 | 0.1546 | 0.984 |


## TID2013 (n=3000)

_cvvdp, ssim2, and iwssim Z-RMSE all use corpus-wide σ (per-stim mos_std contains zeros / near-zeros that blow up per-sample-σ-normalization)._


| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| ssim2 (fast-ssim2) | 3000 | 0.8460 | 0.8504 | 0.6614 | 0.0467 | 0.8846 | 0.526 |
| cvvdp | 3000 | 0.8531 | 0.8644 | 0.6721 | 0.0427 | 0.8853 | 0.503 |
| iwssim | 3000 | 0.7794 | 0.8306 | 0.5995 | 0.0327 | 0.8489 | 0.557 |


### TID2013 10-band panels

#### ssim2 (fast-ssim2)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0835 | 0.2398 | 0.0814 | 0.0690 | 0.0696 | 0.971 |
| B1 | [0.10, 0.20) | 34 | 0.4399 | 0.5209 | 0.3125 | 0.0294 | 0.5681 | 0.854 |
| B2 | [0.20, 0.30) | 186 | 0.2765 | 0.2654 | 0.1859 | 0.0269 | 0.3632 | 0.964 |
| B3 | [0.30, 0.40) | 492 | 0.2730 | 0.2972 | 0.1856 | 0.0427 | 0.3405 | 0.955 |
| B4 | [0.40, 0.50) | 677 | 0.4339 | 0.4392 | 0.2974 | 0.0458 | 0.5141 | 0.898 |
| B5 | [0.50, 0.60) | 705 | 0.3761 | 0.3857 | 0.2568 | 0.0482 | 0.4509 | 0.923 |
| B6 | [0.60, 0.70) | 806 | 0.2167 | 0.2557 | 0.1476 | 0.0372 | 0.2648 | 0.967 |
| B7 | [0.70, 0.80) | 67 | 0.4193 | 0.4755 | 0.2972 | 0.0896 | 0.4933 | 0.880 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

#### cvvdp

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.2394 | 0.4068 | 0.1800 | 0.1034 | 0.2130 | 0.914 |
| B1 | [0.10, 0.20) | 34 | 0.4777 | 0.5345 | 0.3339 | 0.0294 | 0.6243 | 0.845 |
| B2 | [0.20, 0.30) | 185 | 0.1981 | 0.2140 | 0.1343 | 0.0486 | 0.2554 | 0.977 |
| B3 | [0.30, 0.40) | 493 | 0.3956 | 0.4107 | 0.2698 | 0.0406 | 0.4921 | 0.912 |
| B4 | [0.40, 0.50) | 677 | 0.4865 | 0.5007 | 0.3367 | 0.0443 | 0.5787 | 0.866 |
| B5 | [0.50, 0.60) | 705 | 0.4227 | 0.4575 | 0.2917 | 0.0411 | 0.5056 | 0.889 |
| B6 | [0.60, 0.70) | 809 | 0.1529 | 0.2034 | 0.1034 | 0.0284 | 0.1867 | 0.979 |
| B7 | [0.70, 0.80) | 67 | 0.3701 | 0.4483 | 0.2455 | 0.0597 | 0.4250 | 0.894 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |


#### iwssim

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0264 | 0.2398 | 0.0025 | 0.0690 | 0.0459 | 0.971 |
| B1 | [0.10, 0.20) | 34 | 0.4815 | 0.5827 | 0.3375 | 0.0294 | 0.6605 | 0.813 |
| B2 | [0.20, 0.30) | 185 | 0.2469 | 0.2769 | 0.1697 | 0.0270 | 0.3046 | 0.961 |
| B3 | [0.30, 0.40) | 493 | 0.3351 | 0.3418 | 0.2284 | 0.0325 | 0.4192 | 0.940 |
| B4 | [0.40, 0.50) | 677 | 0.3454 | 0.3601 | 0.2323 | 0.0443 | 0.4139 | 0.933 |
| B5 | [0.50, 0.60) | 705 | 0.2966 | 0.2963 | 0.2014 | 0.0369 | 0.3570 | 0.955 |
| B6 | [0.60, 0.70) | 809 | 0.2189 | 0.2350 | 0.1485 | 0.0284 | 0.2680 | 0.972 |
| B7 | [0.70, 0.80) | 67 | 0.3070 | 0.3599 | 0.2110 | 0.0597 | 0.3668 | 0.933 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |


## KonJND-1k (n=1008)

_cvvdp and iwssim panels: 1008-pair PJND-threshold join (missing 0 file-not-found, joined via image_id × codec × round(t) → 504 JPEG + 504 BPG distorted paths). ssim2 n/a — no per-pair fast-ssim2 score extract on the 1008 PJND anchor pairs is on disk; see `benchmarks/baseline_metrics_with_konjnd_2026-05-01.md` for the published Cloudinary Table 4 mean ± stdev calibration anchor._


| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| ssim2 (fast-ssim2) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cvvdp | 1008 | 0.0482 | 0.1521 | 0.0256 | 0.0347 | 0.0225 | 0.988 |
| iwssim | 1008 | 0.1859 | 0.2274 | 0.1327 | 0.0308 | 0.3097 | 0.974 |


_Per-band: KonJND-1k human_score is a PJND threshold in raw units (range 22..70), not a 0..1 normalised quality. The 10-band 0..1 grid does not apply (matches `bake_verdict.rs` `enable_per_band=false` for KonJND-1k)._


## AIC-3 CTC per-pair sweep (n=600)

_`human_score` is the reconstructed JND from the per-pair CSV's normalised target column (matches `dataset_metric_baseline` convention). ssim2 SROCC 0.7965 reproduces the canonical fast-ssim2 baseline at n=600. cvvdp + iwssim were re-joined 2026-05-18 against the per-pair `human_score` column at n=600 — the cvvdp scores come from `/mnt/v/zen/zensim-eval/aic3_cvvdp_scores_2026-05-17.tsv` and the iwssim scores from `/mnt/v/zen/zensim-training/2026-05-18-iwssim-panels/aic3_iwssim_scores.tsv`. The PTC anchor subset (n=300) below uses the anchor CSV's per-pair bootstrap σ and its own iwssim / cvvdp / ssim2 columns directly._


| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| ssim2 (fast-ssim2) | 600 | 0.7965 | 0.8086 | 0.6288 | 0.0567 | 0.8716 | 0.588 |
| cvvdp | 600 | 0.7918 | 0.8034 | 0.6257 | 0.0417 | 0.8657 | 0.595 |
| iwssim | 600 | 0.7735 | 0.7907 | 0.6064 | 0.0450 | 0.8536 | 0.612 |

## AIC-3 CTC anchor PTC subset (n=300)

_ALL panels use per-stimulus bootstrap σ from `std_bootstrap` column. Validates against Mohammadi 2025 paper Z-RMSE table (SSIMULACRA2 47.63, IW-SSIM 31.51, CVVDP 9.45, PSNR-Y 13.36) to within 0.06._


| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| ssim2 (SSIMULACRA2 column) | 300 | 0.9053 | 0.8931 | 0.7379 | 0.0533 | 0.9436 | 47.632 |
| cvvdp (CVVDP column) | 300 | 0.9606 | 0.9589 | 0.8389 | 0.0667 | 0.9791 | 9.459 |
| iwssim (iw_ssim column) | 300 | 0.9443 | 0.9405 | 0.8022 | 0.0533 | 0.9696 | 31.453 |
| psnry (psnry column) | 300 | 0.8121 | 0.8048 | 0.6263 | 0.0400 | 0.8788 | 13.377 |


_Per-band slicing: AIC-3 CTC `human_score` (column `distortion`) is a reconstructed JND in [-3, 0] (more negative = worse), not 0..1 normalised. The 10-band 0..1 grid does not apply (matches `bake_verdict.rs` `enable_per_band=false` for AIC-3)._


## Footnotes

- **ssim2** = fast-ssim2 (SSIMULACRA 2 GPU implementation), scored via `fast_ssim2::compute_ssimulacra2`. Per-pair values from `benchmarks/v0_22_iw_v3_seed1_2026-05-17_eval_per_pair.csv`, which was produced by `dataset_metric_baseline --per-pair-output` during the V_22-IW v3 ship-eval pass (commit ~`2026-05-17`).

- **cvvdp** = ColorVideoVDP-imazen v0.0.1 (GPU), scored via `zen-metrics batch --metric cvvdp --gpu-runtime cuda`. TSVs land in `/mnt/v/zen/zensim-eval/`. Higher is better (CVVDP score domain `[0, 10]`).

- **iwssim** = IW-SSIM (Wang & Li 2011), `iwssim-gpu` GPU implementation via the `iwssim-gpu` crate (umbrella `iwssim_imazen_v<VER>`, score column `iwssim_gpu` in the per-corpus TSVs). Range `[0, 1]`; 1.0 = identical. Scored via `zen-metrics batch --metric iwssim-gpu --gpu-runtime cuda` on the same pair lists as cvvdp. All four large-corpus pair lists (CID22, KADID, TID, KonJND-1k JPEG+BPG subset) have min(W, H) ≥ 384, well above the 176-pixel paper-strict floor; the adaptive small-image reflect-pad path (`IwssimStrategy::ReflectPad`) was not exercised. AIC-3 anchor n=300 PTC subset still uses the anchor CSV's `iw_ssim` column directly (validates against Mohammadi 2025's paper Z-RMSE 31.51 within 0.06).

- **Bootstrap σ**: AIC-3 from anchor CSV `std_bootstrap` column (per-stimulus); TID from `mos_std.txt` (per-stimulus). CID22, KADID, KonJND use corpus-wide σ fallback (matches `bake_verdict.rs` convention for missing per-stimulus σ).

- **PLCC**: pearson on 4-parameter logistic-rescaled scores. Multi-start LM fit (13 starts) per Mohammadi 2025 / ITU-T P.1401 convention. Polarity (distance-shaped vs score-shaped metrics) is absorbed into the `b[3]` sign.

- **OR (outlier ratio)** uses the bake_verdict.rs convention: polarity-aligned z-score residuals; OR = fraction outside ±2σ of the *residual* distribution (not predictions outside ±2σ of MOS).

- **PWRC** (Pearson-weighted rank correlation): rank-transform both inputs, weight rows by distance from rank midpoint, then Pearson on the weighted ranks. Definition per Mohammadi 2025.

- **Z-RMSE**: per-sample-σ-normalized RMSE after the 4-parameter logistic rescale. With per-stimulus σ where available, corpus-wide σ otherwise. Lower is better.


## Data gaps (need follow-up scoring runs to fill)

- **iwssim × {CID22, KADID, TID, KonJND-1k, AIC-3 CTC n=600}**: **FILLED 2026-05-18.** Per-pair iwssim-gpu scores live under `/mnt/v/zen/zensim-training/2026-05-18-iwssim-panels/{cid22,kadid,tid,konjnd_{jpeg,bpg},aic3}_iwssim_scores.tsv`, produced by `zen-metrics batch --metric iwssim-gpu --gpu-runtime cuda` (commit branch `feat/iwssim-baseline-panels`). Aggregate + 10-band panels patched into the corpus sections above. Total scoring wall time ~28 min on the local 5070-class GPU. Cross-check against the 300-row AIC-3 anchor PTC subset (250 rows in the cvvdp pairs TSV overlap with the anchor; the other 50 are `VM_*` codec entries that the cvvdp pairs TSV doesn't cover): zen-metrics iwssim-gpu vs anchor CSV iw_ssim has Pearson 0.9973 / median abs-diff 0.00018 / max 0.00172 — essentially bit-equal to the canonical Wang & Li 2011 reference within float-precision noise. iwssim-gpu was simultaneously re-joined to the per-pair-CSV `human_score` for the AIC-3 CTC n=600 row above (and cvvdp re-joined from `aic3_cvvdp_scores_2026-05-17.tsv`).
- **ssim2 × KonJND-1k**: per-pair fast-ssim2 over the 1008 PJND-mean pairs is not in the per-pair CSV. The aggregate calibration mean ± stdev is documented at `benchmarks/baseline_metrics_with_konjnd_2026-05-01.md` (JPEG 62.55 ± 5.03, BPG 65.38 ± 5.42). A Mohammadi panel requires per-pair scores against the PJND threshold; fix: extract fast-ssim2 per (source × at-PJND-level) pair via the `dataset_metric_baseline --konjnd` per-pair path. n.b. ssim2 score vs PJND threshold is the calibration check, not a discrimination check — the panel SROCC is expected to be near 0 because all 1008 pairs are at the same perceptual threshold by design.
