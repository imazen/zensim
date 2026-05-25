# Cross-bake L0 i8-quantization zero-out census

Date: 2026-05-25

Scanned `/home/lilith/work/zen/zensim/zensim/weights/` recursively. 33 bakes loaded, 8 skipped (non-v3 or parse error).

## Skipped

- `v0_15_2026-05-12.bin` — non-v3 bake: version=2
- `v0_16_2026-05-12.bin` — non-v3 bake: version=2
- `v0_17_2026-05-13.bin` — non-v3 bake: version=2
- `v0_4_2026-04-30.bin` — non-v3 bake: version=2
- `v0_5_2026-05-11.bin` — non-v3 bake: version=2
- `v0_7_seed0_2026-05-11.bin` — non-v3 bake: version=2
- `v0_7_seed1_tv10_2026-05-11.bin` — non-v3 bake: version=2
- `v0_8_tainted_2026-05-11.bin` — non-v3 bake: version=2

## Method

For each ZNPR v3 bake, the L0 layer (`in_dim == n_inputs`) is loaded. For I8 bakes the per-feature zero count is the actual count of `weights[i, o] == 0` over the `out_dim` output columns. For F32 / F16 bakes the **same** scheme `scale[o] = max_i |W[i, o]| / 127.0; q = round(W[i, o] / scale[o]).clamp(-128, 127)` is simulated on the f32 / f16 weights so f32 / f16 / i8 columns are comparable. A feature is **fully_zeroed** when zero_fraction == 1.0, **mostly_zeroed** when ≥ 0.5.

## Headline numbers

- 33 bakes × variable n_inputs = 9424 feature×bake observations.
- 72 feature×bake observations are **fully zeroed** at L0 (0.76%).
- 996 feature×bake observations are **mostly zeroed** (≥ 50% of out_dim columns) (10.57%).

## Per-bake totals

| Bake | n_in | L0 dims | dtype | family | fully_zeroed | mostly_zeroed | L1_sum |
|---|---:|---|---|---|---:|---:|---:|
| `v0_18_1_full218k_noship_2026-05-14.bin` | 228 | 228×384 | i8 | with-peaks-228 | 0 | 0 | 4096.50 |
| `v0_18_2026-05-13_ship.bin` | 228 | 228×384 | i8 | with-peaks-228 | 0 | 228 | 3895.70 |
| `v0_18_inflated_pre_v19_swap_2026-05-14.bin` | 228 | 228×384 | i8 | with-peaks-228 | 0 | 228 | 3895.70 |
| `v0_19_overcleaned_2026-05-14.bin` | 228 | 228×384 | i8 | with-peaks-228 | 0 | 0 | 3905.91 |
| `picker_zenavif_2026-05-19.bin` | 109 | 109×64 | f32 | custom | 18 | 18 | 865.76 |
| `picker_zenjpeg_2026-05-19.bin` | 109 | 109×64 | f32 | custom | 18 | 18 | 944.13 |
| `picker_zenjxl_2026-05-19.bin` | 109 | 109×64 | f32 | custom | 18 | 18 | 866.91 |
| `picker_zenwebp_2026-05-19.bin` | 109 | 109×64 | f32 | custom | 18 | 18 | 877.01 |
| `v05_ensemble_classifier_2026-05-18.bin` | 300 | 300×64 | i8 | extended-300 | 0 | 0 | 1702.10 |
| `v0_18_2026-05-13.bin` | 228 | 228×384 | i8 | with-peaks-228 | 0 | 228 | 3895.70 |
| `v0_18_zerobiased_lz4_2026-05-13.bin` | 228 | 228×384 | i8 | with-peaks-228 | 0 | 228 | 3895.70 |
| `v0_20_is_calibrated_2026-05-15.bin` | 228 | 228×128 | f32 | with-peaks-228 | 0 | 0 | 1730.16 |
| `v0_22_iw_v2_2026-05-16.bin` | 372 | 372×128 | f32 | extended-iw-372 | 0 | 0 | 2427.02 |
| `v0_22_iw_v2_calibrated_2026-05-16.bin` | 372 | 372×128 | f32 | extended-iw-372 | 0 | 0 | 2427.02 |
| `v_balanced_v11_2026-05-20.bin` | 300 | 300×128 | i8 | extended-300 | 0 | 0 | 4154.85 |
| `v_balanced_v11a_s1_spline_2026-05-20.bin` | 300 | 300×128 | f32 | extended-300 | 0 | 0 | 2430.72 |
| `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` | 300 | 300×128 | i8 | extended-300 | 0 | 0 | 4154.85 |
| `v_balanced_v2_2026-05-20.bin` | 300 | 300×128 | i8 | extended-300 | 0 | 0 | 4154.85 |
| `v_balanced_v3_2026-05-20.bin` | 300 | 300×128 | i8 | extended-300 | 0 | 0 | 4154.85 |
| `v_balanced_v3_per_codec_2026-05-20.bin` | 300 | 300×128 | i8 | extended-300 | 0 | 0 | 4154.85 |
| `v_compression_2026-05-18.bin` | 372 | 372×128 | i8 | extended-iw-372 | 0 | 0 | 4640.57 |
| `v_compression_persample_2026-05-18.bin` | 300 | 300×128 | i8 | extended-300 | 0 | 0 | 4618.09 |
| `v_compression_v2_2026-05-20.bin` | 300 | 300×128 | i8 | extended-300 | 0 | 0 | 4618.09 |
| `v_compression_v3_2026-05-20.bin` | 300 | 300×128 | i8 | extended-300 | 0 | 0 | 4618.09 |
| `v_compression_v3_per_codec_2026-05-20.bin` | 300 | 300×128 | i8 | extended-300 | 0 | 0 | 4618.09 |
| `v_cross_codec_2026-05-19.bin` | 372 | 372×128 | f32 | extended-iw-372 | 0 | 5 | 73.58 |
| `v_cross_codec_v2_2026-05-20.bin` | 372 | 372×128 | f32 | extended-iw-372 | 0 | 5 | 73.58 |
| `v_tuner_2026-05-18.bin` | 372 | 372×128 | f32 | extended-iw-372 | 0 | 2 | 53.03 |
| `v_tuner_v10_2026-05-20.bin` | 372 | 372×128 | f32 | extended-iw-372 | 0 | 0 | 24906.03 |
| `v_tuner_v11_2026-05-24.bin` | 372 | 372×128 | i8 | extended-iw-372 | 0 | 0 | 77216.39 |
| `v_tuner_v4_per_codec_2026-05-20.bin` | 372 | 372×128 | f32 | extended-iw-372 | 0 | 0 | 24906.03 |
| `v_tuner_v6_2026-05-19.bin` | 372 | 372×128 | f32 | extended-iw-372 | 0 | 0 | 19078.10 |
| `v_tuner_v9_2026-05-20.bin` | 372 | 372×128 | f32 | extended-iw-372 | 0 | 0 | 24906.03 |

## Top-30 consistently zeroed features (across bakes of matching n_inputs)

Each row is keyed by (n_inputs, feature_idx); a feature ranks higher when it's fully-zeroed in more of the bakes that share its n_inputs.

| n_in | f | label | block | n_fully/n_seen | n_mostly/n_seen | mean_zero_frac | mean_imp |
|---:|---:|---|---|---|---|---:|---:|
| 109 | 23 | `f23` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 24 | `f24` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 25 | `f25` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 27 | `f27` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 28 | `f28` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 29 | `f29` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 30 | `f30` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 31 | `f31` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 32 | `f32` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 33 | `f33` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 34 | `f34` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 40 | `f40` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 41 | `f41` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 54 | `f54` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 55 | `f55` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 56 | `f56` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 57 | `f57` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 109 | 58 | `f58` | custom | 4/4 | 4/4 | 1.000 | 0.0000 |
| 228 | 129 | `s2.X.peaks2` | peaks | 0/7 | 4/7 | 0.573 | 63.7351 |
| 228 | 90 | `s1.Y.peaks1` | peaks | 0/7 | 4/7 | 0.552 | 54.4630 |
| 228 | 51 | `s0.B.peaks0` | peaks | 0/7 | 4/7 | 0.530 | 77.5576 |
| 228 | 107 | `s1.B.basic12` | basic | 0/7 | 4/7 | 0.523 | 0.0134 |
| 228 | 155 | `s2.B.basic3` | basic | 0/7 | 4/7 | 0.520 | 25.9197 |
| 228 | 183 | `s3.X.basic12` | basic | 0/7 | 4/7 | 0.519 | 1.9864 |
| 228 | 6 | `s0.X.basic6` | basic | 0/7 | 4/7 | 0.519 | 0.0280 |
| 228 | 128 | `s2.X.peaks1` | peaks | 0/7 | 4/7 | 0.518 | 1.0321 |
| 228 | 78 | `s1.Y.basic2` | basic | 0/7 | 4/7 | 0.518 | 0.5555 |
| 228 | 116 | `s2.X.basic2` | basic | 0/7 | 4/7 | 0.517 | 21.2998 |
| 228 | 40 | `s0.B.basic2` | basic | 0/7 | 4/7 | 0.517 | 0.9863 |
| 228 | 179 | `s3.X.basic8` | basic | 0/7 | 4/7 | 0.516 | 0.1128 |

## Top-30 consistently survived features (zero in no bakes)

Features that are NEVER fully-zeroed across every bake of their n_inputs cohort, ranked by mean importance (descending).

| n_in | f | label | block | n_seen | mean_zero_frac | mean_imp |
|---:|---:|---|---|---:|---:|---:|
| 109 | 53 | `f53` | custom | 4 | 0.016 | 11653628.7495 |
| 109 | 49 | `f49` | custom | 4 | 0.008 | 3415806.3935 |
| 109 | 0 | `f0` | custom | 4 | 0.031 | 19999.4874 |
| 109 | 10 | `f10` | custom | 4 | 0.023 | 13231.2886 |
| 109 | 51 | `f51` | custom | 4 | 0.004 | 2439.7713 |
| 109 | 52 | `f52` | custom | 4 | 0.020 | 2420.6210 |
| 372 | 129 | `s1.Y.basic5` | basic | 11 | 0.181 | 2391.1406 |
| 372 | 90 | `s0.B.iw3` | iw | 11 | 0.160 | 1382.3764 |
| 372 | 51 | `s0.Y.masked1` | masked | 11 | 0.087 | 978.2328 |
| 372 | 12 | `s0.X.basic12` | basic | 11 | 0.235 | 833.1417 |
| 109 | 108 | `f108` | custom | 4 | 0.000 | 767.0794 |
| 109 | 75 | `f75` | custom | 4 | 0.023 | 702.3580 |
| 109 | 76 | `f76` | custom | 4 | 0.016 | 538.8562 |
| 109 | 74 | `f74` | custom | 4 | 0.008 | 517.6093 |
| 372 | 155 | `s1.B.basic0` | basic | 11 | 0.021 | 394.7013 |
| 300 | 129 | `s1.B.basic4` | basic | 11 | 0.124 | 378.2224 |
| 109 | 99 | `f99` | custom | 4 | 0.027 | 338.1461 |
| 300 | 12 | `s0.X.basic12` | basic | 11 | 0.070 | 322.5783 |
| 109 | 73 | `f73` | custom | 4 | 0.012 | 263.4037 |
| 109 | 7 | `f7` | custom | 4 | 0.016 | 256.5164 |
| 109 | 20 | `f20` | custom | 4 | 0.008 | 236.4119 |
| 300 | 51 | `s0.B.basic1` | basic | 11 | 0.114 | 226.4434 |
| 300 | 90 | `s1.X.peaks2` | peaks | 11 | 0.100 | 208.4462 |
| 109 | 44 | `f44` | custom | 4 | 0.004 | 201.8706 |
| 372 | 116 | `s1.X.masked4` | masked | 11 | 0.028 | 188.9490 |
| 228 | 12 | `s0.X.basic12` | basic | 7 | 0.512 | 187.3419 |
| 109 | 17 | `f17` | custom | 4 | 0.016 | 186.4285 |
| 372 | 38 | `s0.Y.basic7` | basic | 11 | 0.023 | 169.9801 |
| 372 | 77 | `s0.B.peaks2` | peaks | 11 | 0.043 | 159.7724 |
| 109 | 72 | `f72` | custom | 4 | 0.031 | 101.1404 |

## Per-block mean zero-fraction across all bakes

| block | n observations | mean zero_fraction |
|---|---:|---:|
| basic | 4524 | 0.157 |
| custom | 436 | 0.178 |
| iw | 792 | 0.023 |
| masked | 1584 | 0.044 |
| peaks | 2088 | 0.156 |

## Per-bake callouts

### Bakes with the largest fully-zeroed fraction at L0

| Bake | dtype | fully_zeroed / total | frac |
|---|---|---|---:|
| `picker_zenavif_2026-05-19.bin` | f32 | 18 / 109 | 0.165 |
| `picker_zenjpeg_2026-05-19.bin` | f32 | 18 / 109 | 0.165 |
| `picker_zenjxl_2026-05-19.bin` | f32 | 18 / 109 | 0.165 |
| `picker_zenwebp_2026-05-19.bin` | f32 | 18 / 109 | 0.165 |
| `v0_18_1_full218k_noship_2026-05-14.bin` | i8 | 0 / 228 | 0.000 |

### Bakes with the smallest fully-zeroed fraction at L0

| Bake | dtype | fully_zeroed / total | frac |
|---|---|---|---:|
| `v_tuner_v9_2026-05-20.bin` | f32 | 0 / 372 | 0.000 |
| `v_tuner_v6_2026-05-19.bin` | f32 | 0 / 372 | 0.000 |
| `v_tuner_v4_per_codec_2026-05-20.bin` | f32 | 0 / 372 | 0.000 |
| `v_tuner_v11_2026-05-24.bin` | i8 | 0 / 372 | 0.000 |
| `v_tuner_v10_2026-05-20.bin` | f32 | 0 / 372 | 0.000 |

## Spot-check: v_tuner_v11_2026-05-24 (v0.3 ship)

- dtype: `i8`, n_inputs: 372, L0: 372×128
- L0 fully-zeroed: **0** / 372 features (0.00%)
- feature 129 (`s1.Y.basic5`, block `basic`): zero_count=12/128 (0.094), L1 share 0.182%, importance 9392.9095
- across 11 bakes with n_inputs=372, feature 129 is fully-zeroed in **0 of 11** (0.0%).
