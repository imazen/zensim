# Linear projections over the 372 features — SDR + HDR probe (2026-07-03)

Mission: find the best modern LINEAR heads (372→1) for SDR and HDR,
exploiting the per-weight postprocessing toolkit that linear bakes make
cheap — BVLS sign/bound projection, zerobias pruning, f16 quantization,
monotone PCHIP output spline refit AFTER quantization (standard
pack-then-calibrate order per `benchmarks/standard_bake_packing_2026-05-27.md`)
— and establish whether deterministic linear fits dodge the MLP
seed-collapse mode (2026-07-03 HDR wide fan: 43.75% collapse, 7/16 seeds,
per `benchmarks/strategy_ablation_2026-07-02.md`).

Prior art: `benchmarks/v02_bvls_shaped_2026-05-28.md` — 8.6 KB BVLS linear
bake at CID22 0.824 / KonJND 0.594 (KonJND far above every MLP ship).

## Method

Script: `scripts/v_next/linear_projections_2026-07-03.py` (subcommands
`gram` / `fit` / `finalize`). Three phases:

1. **gram** — one streaming pass per training group accumulates raw
   moments (S = ΣxxT, s = Σx, q = Σx·y per target, Σy, Σy², n) in BOTH
   feature spaces: `raw` and `shaped` (per-feature Yeo-Johnson / Winsor /
   SignedCbrt / QuantileBins from
   `benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv`,
   applied with the same f32-parity code as the 2026-05-28 script).
   Targets are minmax01'd per group (q0.001/q0.999 clip) before
   accumulation. **No row subsampling anywhere — every fit uses all rows
   of its mix** (the Gram trick makes full-data exact fits cheap; the
   brief's suggested 300k stride subsample was unnecessary).
2. **fit** — every (mix × family × shaping) solves from the weighted
   per-group Grams (Grams are additive under mix weights; standardization
   is derived algebraically from the mix's weighted moments):
   - `ridge`: (G_z + λ·W·I)w = c_z, λ ∈ {1e-7…1e-2} (6 points)
   - `bvls`: Cholesky trick G_z = LLᵀ → `lsq_linear(Lᵀ, L⁻¹c_z, bounds, method="bvls")`
     — exact bounded LS **on all rows** via a 372×372 system. Sign mask =
     `benchmarks/feature_sign_mask_2026-05-26.tsv` (300 pin w≥0, 72 free).
   - `lasso`: covariance-form coordinate descent (fixed sweep order,
     200 max sweeps), λ ∈ {3e-5…2e-3} (6 points)
3. **finalize** — per candidate: zerobias τ sweep on the standardized
   weights → f16 round-trip (numpy f16 cast = exactly the weights the
   bake stores) → **PCHIP dial spline fit on the PACKED (pruned+f16)
   forward** over `multiband_anchor_dial100.parquet` `target_score`
   (quantize-then-calibrate; same knot logic as
   `scripts/v_next/pack_and_calibrate.py`) → single ZNPR v3 bake emit via
   `zenpredict bake` JSON pipeline (`compressed: true`, layer dtype f16;
   shaped variants carry `zentrain.feature_transforms` +
   `feature_transform_params` metadata so `predict_transformed`
   dispatches).

Selection axes are train-legal only: `bigcodec_valdigits` SROCC (SDR),
`hdr_zenjxl_v3_valdigits` SROCC (HDR), and konjnd-dense-norm-train
`pjnd_target` |SROCC| as a weak guard. CID22 / AIC-3 / AIC-4 / KonJND-val
were never used for fitting, λ/τ selection, or spline knots; they appear
only in the final `bake_verdict` panels, and **every baked candidate's
panel is reported below** (no cherry-picking; round 2 was pre-registered
before its panels were run).

### Runtime/Python parity

- `lp_canon-bvls-raw-tau0-f16` reproduces the 2026-05-28
  `v02_bvls_NO_shaping` bake through this fully independent pipeline:
  CID22 0.8239 vs 0.8240, KADID 0.7567 vs 0.7567, TID 0.7342 vs 0.7343,
  KonJND 0.5935 vs 0.5941, AIC-3 0.7472 vs 0.7472, AIC-4 0.7938 vs 0.7937
  (2026-05-28 bake re-verdicted today, same env).
- Python-side hdrval SROCC == baked-runtime hdrval (via `bake_verdict`
  pred-dump) to 4 decimals on both HDR finalists (+0.8798, +0.8859) —
  transforms + f16 + spline round-trip is exact.

## Data lineage (all local, read-only)

| group | path | rows used | target |
|---|---|--:|---|
| bigcodec | `/mnt/v/output/zensim-multicodec-probe/bigcodec_traindigits_2026-07-02.parquet` | 2,946,036 | `human_score` (ssim2-normalized) |
| safesyn | `/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet` | 196,086 | `human_score` |
| cid22_train | `.../canonical-2026-05-21/train/cid22_train_norm.parquet` | 17,611 | `human_score` (ssim2-anchored, NOT MOS — train-legal per DATA_SPLITS.md) |
| kadid | `.../train/kadid.parquet` | 10,125 | `human_score` (DMOS) |
| tid | `.../train/tid.parquet` | 3,000 | `human_score` (MOS) |
| konjnd_dense | `.../train/konjnd-dense-norm.parquet` | 20,160 | `human_score` (active-mix); `pjnd_target` for the guard axis only |
| hdr_v3 | `/mnt/v/output/zensim-multicodec-probe/hdr_zenjxl_v3_traindigits_2026-07-03.parquet` | 7,410 | `human_score` (ssim2-based, PU-linear features) |
| hdr_v3mix | `.../hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet` | 7,410 | `human_score` = 0.5·ssim2 + 0.5·(JOD−6)/4 (cvvdp mix; READ-ONLY per w8-agent ownership) |
| anchor (spline only) | `.../canonical-2026-05-21/train/multiband_anchor_dial100.parquet` | 2,000 | `target_score` (0–97.4 dial) |
| val: bigcodec_val | `.../bigcodec_valdigits_2026-07-02.parquet` | 147,067 | selection |
| val: hdr_val | `.../hdr_zenjxl_v3_valdigits_2026-07-03.parquet` | 3,900 | selection |
| val: hdr_valmix | `.../hdr_zenjxl_v3mix_valdigits_2026-07-03.parquet` | 3,900 | selection |

Zero rows dropped for non-finite features/targets in any group, either
feature space (gram log). kadis_cvvdp_train was skipped: scalar-CVVDP
targets are a falsified direction (V41 + 2026-05-27,
`feedback_cvvdp_scalar_target_dead_end.md`) and the hdr_v3mix target
already carries the useful CVVDP signal — documented deviation from the
brief's "optionally" clause.

## Data mixes swept

| mix | groups (train_w) |
|---|---|
| big | bigcodec 1.0 |
| canon | safesyn 1.0, cid22_train 1.5, kadid 0.5, tid 0.5 (= 2026-05-28 recipe) |
| w7sdr | canon + konjnd_dense 1.2 + bigcodec 0.25 (w7_guard train mix minus HDR) |
| hdr | hdr_v3 1.0 |
| hdrmix | hdr_v3mix 1.0 |
| w7lin | w7sdr + hdr_v3 1.0 (= w7_guard_s101.toml train mix exactly) |
| w8lin | w7sdr + hdr_v3mix 1.0 (= w8_hdrmix_cvmix train mix) |
| canonhdr15 / canonhdr40 | canon + hdr_v3mix 15.0 / 40.0 (round 2: no bigcodec) |
| canonkjhdr15 | canonhdr15 + konjnd_dense 1.2 (round 2) |

154 round-1 fits + 52 round-2 fits; full val table at
`/mnt/v/output/zensim-multicodec-probe/linear-probe/fits/table.json` and
`fit.log` / `fit_round2.log` alongside.

## Reference bars

| ref | size | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 | UPIQ | hdrval |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A = v47-strict-QAT (shipped Profile::A) | 27,316 | 0.8657 | 0.7933 | 0.7927 | 0.4185 | 0.7680 | 0.8854 | 0.694 | — |
| t1dro51_s31 (MLP finalist, SDR) | 48,011 | 0.8708 | — | — | 0.3109 | — | — | 0.6594 | 0.9017 |
| w7_s101 (MLP HDR-mix finalist) | 67,293 | 0.8639 | — | — | 0.3524 | — | — | 0.6798 | 0.8985 |
| v02-bvls NO-shaping (2026-05-28 linear) | 8,622 | 0.8240 | 0.7567 | 0.7343 | 0.5941 | 0.7472 | 0.7937 | 0.6586 | — |
| raw metrics on UPIQ: cvvdp 0.758 / iwssim-HDR 0.808 / ssim2-HDR 0.704 | | | | | | | | | |

(hdrval = SROCC on hdr_zenjxl_v3_valdigits via bake_verdict pred-dump,
measured today; MLP sizes = `stat -c%s` on the .bin.)

## Round 1 — full panel, all 19 bakes (finalists × τ, selected on val axes only)

All numbers from `bake_verdict` (SROCC per corpus) + `upiq_panel.py`
(PU-linear features). B = bake bytes (f16 + lz4). mono/G1/goal from the
built-in DIAL panel / goals scorecard. Verdicts:
`/mnt/v/output/zensim-multicodec-probe/linear-probe/verdicts/`.

| bake | B | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 | UPIQ | G1 | mono | goal |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| big-lasso0.001-raw-tau0 | 3939 | 0.7571 | 0.4159 | 0.7320 | 0.3595 | 0.7043 | 0.8310 | 0.5913 | 1.00 | 0.9566 | 0.457 |
| big-lasso0.001-raw-tau0.005 | 3651 | 0.7648 | 0.4161 | 0.7312 | 0.3575 | 0.7043 | 0.8285 | 0.5887 | 1.00 | 0.9560 | 0.456 |
| big-ridge1e-06-raw-tau0 | 4090 | 0.6997 | 0.6127 | 0.6965 | 0.3755 | 0.6676 | 0.8252 | 0.6513 | 1.00 | 0.9498 | 0.419 |
| big-ridge1e-06-raw-tau0.005 | 4098 | 0.7036 | 0.6490 | 0.7098 | 0.3679 | 0.6746 | 0.8473 | 0.6528 | 0.92 | 0.9507 | 0.400 |
| big-ridge1e-06-raw-tau0.02 | 4073 | 0.6903 | 0.5801 | 0.6096 | 0.3689 | 0.6576 | 0.8069 | 0.5448 | 0.87 | 0.9396 | 0.361 |
| canon-bvls-raw-tau0 | 3782 | 0.8239 | 0.7567 | 0.7342 | 0.5935 | 0.7472 | 0.7938 | 0.6587 | 1.00 | 0.9381 | 0.560 |
| canon-bvls-raw-tau0.005 | 3741 | 0.8280 | 0.7706 | 0.7508 | 0.6014 | 0.7428 | 0.7880 | 0.6809 | 1.00 | 0.9370 | 0.557 |
| canon-ridge1e-05-raw-tau0 | 4090 | 0.8465 | 0.8813 | 0.8606 | 0.2406 | 0.7654 | 0.8420 | 0.6877 | 1.00 | 0.9370 | 0.626 |
| canon-ridge1e-05-raw-tau0.005 | 4062 | 0.8310 | 0.8603 | 0.8405 | 0.1630 | 0.7661 | 0.8530 | 0.6910 | 1.00 | 0.9345 | 0.611 |
| canon-ridge1e-05-raw-tau0.02 | 3942 | 0.8225 | 0.4063 | 0.5836 | 0.4249 | 0.6872 | 0.8121 | 0.7425 | 1.00 | 0.9222 | 0.475 |
| hdr-lasso0.001-shaped-tau0 | 11636 | 0.8347 | 0.7505 | 0.7165 | 0.3741 | 0.7855 | 0.9022 | 0.7313 | 1.00 | 0.9234 | 0.642 |
| hdr-lasso0.001-shaped-tau0.005 | 11348 | 0.8378 | 0.7557 | 0.7193 | 0.2552 | 0.7889 | 0.9106 | 0.7234 | 1.00 | 0.9245 | 0.658 |
| hdrmix-lasso0.001-raw-tau0 | 3772 | 0.8689 | 0.5060 | 0.7810 | 0.4129 | 0.7979 | 0.9319 | 0.6488 | 1.00 | 0.9711 | 0.695 |
| hdrmix-lasso0.001-raw-tau0.005 | 3477 | 0.8661 | 0.5615 | 0.7820 | 0.3339 | 0.8024 | 0.9340 | 0.6446 | 1.00 | 0.9726 | 0.706 |
| hdrmix-ridge0.0001-raw-tau0 | 4049 | 0.8200 | 0.1405 | 0.0130 | 0.2628 | 0.8048 | 0.9236 | 0.2322 | 0.74 | 0.9307 | 0.572 |
| hdrmix-ridge0.0001-raw-tau0.005 | 4044 | 0.8232 | 0.2232 | 0.2937 | 0.1942 | 0.8200 | 0.9498 | 0.4075 | 0.77 | 0.9679 | 0.634 |
| hdrmix-ridge0.0001-raw-tau0.02 | 3907 | 0.7755 | 0.2611 | 0.4546 | 0.2302 | 0.7476 | 0.8546 | 0.4868 | 1.00 | 0.9396 | 0.522 |
| w8lin-bvls-raw-tau0 | 3754 | 0.6523 | 0.6650 | 0.6978 | 0.4783 | 0.6878 | 0.7686 | 0.5856 | 1.00 | 0.9166 | 0.436 |
| w8lin-bvls-raw-tau0.005 | 3722 | 0.6526 | 0.6634 | 0.6929 | 0.4865 | 0.6882 | 0.7674 | 0.5856 | 1.00 | 0.9200 | 0.437 |

Round-1 reading:

- **`canon-bvls-tau0.005` strictly upgrades the 2026-05-28 linear ship-candidate**
  on 5/7 axes at 43% of its size: CID22 0.8280 (+0.004), KADID 0.7706
  (+0.014), TID 0.7508 (+0.017), KonJND 0.6014 (+0.007), UPIQ 0.6809
  (+0.022); AIC-3 −0.004, AIC-4 −0.006. The zerobias prune (τ=0.005,
  86→75 active) is a free win here.
- **bigcodec mass poisons linear CID22.** big-* (CID22 0.69–0.76) and
  w8lin-* (0.65) despite bigval 0.90–0.93. The 2.9M-row group dominates
  any mix it enters at meaningful weight; a single linear head gets pulled
  to the ssim2-normalized multi-codec regime and away from human rank.
  (The MLPs absorb the same mix fine — capacity.) The konjnd guard axis
  mis-ranked here (w8lin-bvls guard 0.192 was the best, its CID22 the
  worst BVLS) — guard is necessary-not-sufficient.
- **`hdrmix-lasso0.001-tau0` (47 weights, trained on 7,410 HDR JXL rows
  with the cvvdp-mix target) hit CID22 0.8689** — above A (0.8657) and
  w7_s101 (0.8639) — plus AIC-3 0.7979 / AIC-4 0.9319 / mono 0.9711.
  KADID 0.506 is its hole (analytic distortions). This motivated round 2.
- **Dense HDR ridge is falsified** (hdrmix-ridge: TID 0.01–0.45,
  G1 0.74–0.77, KADID ≤0.26) — without sparsity the 7.4k-row HDR fit
  doesn't transfer.
- **f16 quantization is free** (bigval 0.9313→0.9311 worst case);
  **the pack-then-calibrate spline refit keeps G1 = 1.00** on every
  non-broken candidate.
- Shaping (Yeo-Johnson TSV) helped only the pure-HDR mix (UPIQ 0.7313 —
  see round 2 comparison); everywhere else it lost val + guard, matching
  the 2026-05-28 "shaping specializes to the fitting distribution"
  finding.

## Round 2 — canon+hdrmix blends (no bigcodec) + hdrmix λ densify

Pre-registered before any round-2 panel was run: mixes {canonhdr15,
canonhdr40, canonkjhdr15} × families {bvls, lasso5e-4, lasso1e-3} +
hdrmix-lasso{5e-4, 2e-3}, τ ∈ {0, 0.005}. Every bake's panel is below —
nothing withheld.

| bake | B | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 | UPIQ | G1 | mono | goal |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| canonhdr15-bvls-raw-tau0 | 3789 | 0.7929 | 0.7412 | 0.7310 | 0.6537 | 0.7262 | 0.8140 | 0.6681 | 1.00 | 0.9317 | 0.496 |
| canonhdr15-bvls-raw-tau0.005 | 3738 | 0.8001 | 0.7540 | 0.7404 | 0.6696 | 0.7225 | 0.8095 | 0.6679 | 1.00 | 0.9315 | 0.489 |
| canonhdr15-lasso0.0005-raw-tau0 | 3907 | 0.8060 | 0.8723 | 0.7836 | 0.3454 | 0.7459 | 0.8464 | 0.6812 | 1.00 | 0.9685 | 0.535 |
| canonhdr15-lasso0.0005-raw-tau0.005 | 3611 | 0.8056 | 0.8099 | 0.7310 | 0.3529 | 0.7402 | 0.8413 | 0.7148 | 1.00 | 0.9685 | 0.526 |
| canonhdr15-lasso0.001-raw-tau0 | 3858 | 0.8371 | 0.8652 | 0.7777 | 0.3320 | 0.7525 | 0.8542 | 0.6942 | 1.00 | 0.9747 | 0.581 |
| canonhdr15-lasso0.001-raw-tau0.005 | 3592 | 0.8394 | 0.8502 | 0.7632 | 0.3347 | 0.7532 | 0.8568 | 0.6913 | 1.00 | 0.9736 | 0.584 |
| canonhdr40-bvls-raw-tau0 | 3768 | 0.7734 | 0.7221 | 0.7272 | 0.6271 | 0.7155 | 0.8205 | 0.6688 | 1.00 | 0.9264 | 0.474 |
| canonhdr40-bvls-raw-tau0.005 | 3746 | 0.7721 | 0.7292 | 0.7335 | 0.6180 | 0.7206 | 0.8243 | 0.6683 | 1.00 | 0.9254 | 0.482 |
| canonhdr40-lasso0.0005-raw-tau0 | 3900 | 0.8138 | 0.8631 | 0.7668 | 0.3382 | 0.7448 | 0.8585 | 0.7095 | 1.00 | 0.9704 | 0.541 |
| canonhdr40-lasso0.0005-raw-tau0.005 | 3612 | 0.8169 | 0.8023 | 0.7247 | 0.3267 | 0.7398 | 0.8591 | 0.7069 | 1.00 | 0.9681 | 0.539 |
| canonhdr40-lasso0.001-raw-tau0 | 3831 | 0.8307 | 0.8541 | 0.7631 | 0.3241 | 0.7531 | 0.8667 | 0.6995 | 1.00 | 0.9745 | 0.573 |
| canonhdr40-lasso0.001-raw-tau0.005 | 3604 | 0.8276 | 0.7926 | 0.7074 | 0.2954 | 0.7534 | 0.8713 | 0.7018 | 1.00 | 0.9726 | 0.574 |
| canonkjhdr15-bvls-raw-tau0 | 3778 | 0.7972 | 0.7458 | 0.7339 | 0.6253 | 0.7318 | 0.8110 | 0.6682 | 1.00 | 0.9275 | 0.506 |
| canonkjhdr15-bvls-raw-tau0.005 | 3735 | 0.7980 | 0.7436 | 0.7310 | 0.6500 | 0.7264 | 0.8082 | 0.6686 | 1.00 | 0.9264 | 0.497 |
| canonkjhdr15-lasso0.0005-raw-tau0 | 3912 | 0.8184 | 0.8729 | 0.7828 | 0.3805 | 0.7433 | 0.8438 | 0.6763 | 1.00 | 0.9700 | 0.545 |
| canonkjhdr15-lasso0.0005-raw-tau0.005 | 3604 | 0.8148 | 0.8255 | 0.7286 | 0.3718 | 0.7479 | 0.8454 | 0.7127 | 1.00 | 0.9681 | 0.550 |
| canonkjhdr15-lasso0.001-raw-tau0 | 3877 | 0.8416 | 0.8661 | 0.7758 | 0.3785 | 0.7500 | 0.8499 | 0.6873 | 1.00 | 0.9745 | 0.582 |
| canonkjhdr15-lasso0.001-raw-tau0.005 | 3592 | 0.8480 | 0.8145 | 0.7348 | 0.3776 | 0.7469 | 0.8508 | 0.6732 | 1.00 | 0.9726 | 0.584 |
| hdrmix-lasso0.0005-raw-tau0 | 3852 | 0.8556 | 0.3841 | 0.5514 | 0.4866 | 0.7920 | 0.9356 | 0.6541 | 1.00 | 0.9728 | 0.675 |
| hdrmix-lasso0.0005-raw-tau0.005 | 3508 | 0.8436 | 0.4523 | 0.7497 | 0.4419 | 0.7858 | 0.9375 | 0.6662 | 1.00 | 0.9768 | 0.657 |
| hdrmix-lasso0.002-raw-tau0 | 3762 | 0.8740 | 0.7887 | 0.7949 | 0.3716 | 0.7905 | 0.9246 | 0.6876 | 1.00 | 0.9711 | 0.677 |
| hdrmix-lasso0.002-raw-tau0.005 | 3445 | 0.8764 | 0.8312 | 0.8115 | 0.3141 | 0.7912 | 0.9229 | 0.6698 | 0.99 | 0.9724 | 0.675 |

Round-2 reading:

- **`hdrmix-lasso0.002-tau0` (35 weights, 3,762 B) is the best linear
  CID22 ever measured here: 0.8740** — beats every MLP reference
  (t1dro51_s31 0.8708, w7_s101 0.8639, A 0.8657) — with KADID 0.7887 /
  TID 0.7949 (≈ A's 0.793/0.793), AIC-3 0.7905 (> A 0.768), AIC-4 0.9246
  (> A 0.8854), UPIQ 0.6876 (≈ A 0.6933), KonJND 0.3716 (> t1dro51
  0.3109 and > w7 0.3524, but < A 0.4185), G1 1.00, mono 0.9711,
  goal 0.677 (> A 0.622). The λ=0.002 sparsity fixed λ=0.001's KADID
  hole (0.506 → 0.789).
- Its τ=0.005 sibling (15 weights!, 3,445 B) pushes CID22 to **0.8764**
  and KADID/TID to 0.831/0.812, trading KonJND down to 0.314.
- **`canonhdr15-bvls-tau0.005` sets the KonJND record: 0.6696**
  (prior best 0.6014 this session; best MLP 0.4185) at CID22 0.8001.
  The sign-masked BVLS + 15×-weighted hdr_v3mix group is the recipe.
- canonhdr40 adds nothing meaningful over canonhdr15 (marginal AIC-4
  wins, consistent CID22/KonJND losses) — 15× is enough; 40× only
  dilutes canon. konjnd_dense in the mix (canonkjhdr15) ≈ wash.

## Winners vs references

| bake | B | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 | UPIQ | hdrval | mono | goal |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **SDR pick: lp_hdrmix-lasso0.002-raw-tau0-f16** | 3,762 | **0.8740** | 0.7887 | 0.7949 | 0.3716 | **0.7905** | **0.9246** | 0.6876 | 0.8815 | 0.9711 | **0.677** |
| **HDR pick: lp_hdr-lasso0.001-shaped-tau0-f16** | 11,636 | 0.8347 | 0.7505 | 0.7165 | 0.3741 | 0.7855 | 0.9022 | **0.7313** | 0.8859 | 0.9234 | 0.642 |
| PJND alt: lp_canon-bvls-raw-tau0.005-f16 | 3,741 | 0.8280 | 0.7706 | 0.7508 | 0.6014 | 0.7428 | 0.7880 | 0.6809 | 0.8462 | 0.9370 | 0.557 |
| KonJND record: lp_canonhdr15-bvls-raw-tau0.005-f16 | 3,738 | 0.8001 | 0.7540 | 0.7404 | **0.6696** | 0.7225 | 0.8095 | 0.6679 | 0.8474 | 0.9315 | 0.489 |
| A (shipped, MLP 27 KB) | 27,316 | 0.8657 | 0.7933 | 0.7927 | 0.4185 | 0.7680 | 0.8854 | 0.6933 | 0.8505 | 0.9747 | 0.622 |
| t1dro51_s31 (MLP 48 KB) | 48,011 | 0.8708 | — | — | 0.3109 | — | — | 0.6594 | 0.9017 | — | — |
| w7_s101 (MLP 67 KB) | 67,293 | 0.8639 | — | — | 0.3524 | — | — | 0.6798 | 0.8985 | — | — |
| v02-bvls (2026-05-28 linear) | 8,622 | 0.8240 | 0.7567 | 0.7343 | 0.5941 | 0.7472 | 0.7937 | 0.6586 | — | 0.9381 | 0.560 |

(hdrval for linear picks = Python == runtime, parity-verified; for
canon-bvls/canonhdr15-bvls rows taken from the fit table at the baked τ.)

## Determinism — the collapse mode is structurally absent

The 2026-07-03 MLP HDR wide fan collapsed at 7/16 seeds (43.75%,
auto-gated); `strategy_ablation_2026-07-02.md` documents the same mode at
1-in-3 / 1-in-5 rates in the SDR ablations. Linear fits have **no seed
axis at all** — ridge is a strictly convex solve, BVLS a deterministic
active-set method, the lasso CD uses a fixed sweep order — and the whole
pipeline was verified bit-stable end-to-end:

- **44/44 fits byte-identical** (`w`, `bias`, `mu`, `sd` arrays compared
  as raw bytes) across a FULL re-run: `gram --force` re-accumulated every
  moment from the parquets, `fit` re-solved (mixes big / hdrmix /
  canonhdr15 × ridge+bvls+lasso × raw+shaped; same `run-heavy --jobs 6`
  thread caps). `determinism_check.py` in the probe dir.
- **Re-baked artifacts sha256-identical** for both headline candidates
  (`lp_hdrmix-lasso0.002-raw-tau0-f16` `bdb5d63ea699…`,
  `lp_canonhdr15-bvls-raw-tau0.005-f16` `1400fa2f86e2…`) — fit → f16 →
  spline → JSON → `zenpredict bake` reproduces the exact bytes.

Caveat stated precisely: determinism was verified on this machine at
fixed BLAS thread count. Cross-ISA reduction-order drift (the AVX-512 vs
AVX2 divergence seen in the MLP cross-machine wave) is not covered by
this check — but there is no *collapse* axis: the solution is the unique
optimum of a convex problem, so any such drift is ULP-scale, not a
0.56-vs-0.85 seed lottery.

## Post-fit toolkit findings (what linear makes cheap)

- **zerobias τ=0.005 on standardized weights is usually free or better**
  (canon-bvls: +0.004 CID22 +0.014 KADID +0.007 KonJND +0.022 UPIQ;
  canonhdr15-bvls: +0.007 CID22 +0.016 KonJND). At τ=0.02 quality falls
  off (canon-ridge CID22 0.8465→0.8225, KADID 0.88→0.41). Sweep it, don't
  default it: on the sparsest fits it can flip KonJND either way
  (hdrmix-lasso0.001: 0.4129→0.3339; hdrmix-lasso0.002: 0.3716→0.3141).
- **f16 is free** (≤0.0002 val SROCC anywhere) and with lz4 makes every
  raw-feature linear bake **3.4–4.1 KB regardless of density** — weight
  count barely matters for size; it matters for generalization.
- **Spline-on-packed (quantize-then-calibrate) held G1=1.00 + goal-grade
  dial on every non-broken candidate**; the linear winners' mono
  (0.971–0.977) matches or beats A's 0.9747 and every one clears G3 ≥0.93.
- Sign-mask BVLS is *the* PJND lever: every KonJND ≥ 0.59 bake in this
  program (old + new) is a sign-masked BVLS fit.
- Per-corpus affine: unnecessary as a bake artifact — for a linear head,
  scaler + affine fold into `w` (rank-invariant); the monotone spline is
  the only nonlinearity that matters. Nothing to sweep.

## Falsified / negative results (do not retry without new evidence)

1. **bigcodec mass in a linear mix destroys CID22** (0.65–0.76 across
   big/w7sdr/w7lin/w8lin at any family). The MLP recipes carry the same
   group at 0.25 fine; a 372→1 head cannot. Linear mixes should hold the
   2.9M-row group OUT (or at ≪0.25 effective mass — untested).
2. **Dense (ridge) HDR-only fits don't transfer** (TID 0.01, KADID 0.14,
   G1 0.74). HDR fits need sparsity (lasso act ≤ 100) or the canon blend.
3. **konjnd-dense-train `pjnd_target` guard is a weak selector** — it
   correctly favored BVLS but mis-ranked w8lin-bvls (best guard 0.192,
   worst BVLS CID22 0.65). Keep val/konjnd (semi-holdout) in the verdict
   panel as the real check.
4. **Input shaping (Yeo-Johnson TSV) loses on every axis for SDR mixes**
   (confirms 2026-05-28) — EXCEPT pure-HDR, where `hdr-lasso0.001-shaped`
   is the UPIQ winner (0.7313 vs raw sibling 0.6488). The TSV was
   screened on safesyn; that it transfers to PU-linear HDR features at
   all is notable.
5. kadis_cvvdp scalar-target variant: not attempted (falsified direction,
   see Data lineage note).

## Honest verdict

**Where linear wins:**

- **KonJND / PJND discrimination — decisively.** 0.6696
  (canonhdr15-bvls-tau0.005) vs 0.4185 best-MLP. The f16 MLP heads lose
  the fine-weight precision PJND needs (v47 QAT trade); linear+BVLS keeps
  it. This axis alone justifies a linear sibling profile.
- **Size:** 3.4–4 KB vs 27–67 KB (7–18×), and 15–86 active features
  means proportionally cheaper feature extraction if a reduced-feature
  fast path ever lands.
- **Determinism:** no seed axis, no collapse gate, bit-reproducible
  artifacts. The 43.75%-collapse fan cost a day of fleet compute +
  gating machinery; a linear refresh is one deterministic minute.
- **CID22, surprisingly, at the top of this program:** 0.8740 (35
  weights) / 0.8764 (15 weights) beat A (0.8657) and the best MLP
  finalist t1dro51_s31 (0.8708). The cvvdp-mix HDR target on 7,410
  zenjxl-HDR rows is doing the work — it transfers to SDR codec
  distortions linearly. (Caveat: single holdout, n=4,292; the two
  hdrmix-lasso siblings bracket 0.869–0.876, so it's stable across
  adjacent λ/τ, but treat "beats every MLP" as ~+0.005-scale, not a
  blowout.)
- **AIC-4:** 0.9246–0.9375 across the hdrmix family vs A's 0.8854.
- **Dial:** mono 0.971–0.977 + G1 1.00 on the winners ≈ MLP-grade.

**Where linear loses:**

- **hdrval top-end:** best linear 0.8859 vs MLP 0.9017 (−0.016) — the
  MLPs are genuinely better rankers inside the HDR corpus distribution.
- **No single linear bake covers all axes.** The CID22 winner holds
  KonJND 0.37; the KonJND winner holds CID22 0.80–0.83. The MLP A is
  more balanced per-bake (0.8657/0.4185). Linear's answer is siblings
  (they're 3.7 KB each), not one head.
- **KADID/TID ceiling:** canon-ridge reaches 0.88/0.86 (above A!) but
  only by sacrificing KonJND; the balanced winners sit at 0.75–0.81.
- bigcodec val (raw multi-codec ssim2 rank): dense big-ridge reaches
  0.9313 but nothing CID22-competitive exceeds ~0.85 there.

**Picks:**

- **SDR: `lp_hdrmix-lasso0.002-raw-tau0-f16`** (35 weights, 3,762 B,
  sha256 `bdb5d63ea699…`) — CID22 0.8740, KADID 0.7887, TID 0.7949,
  KonJND 0.3716, AIC-3 0.7905, AIC-4 0.9246, UPIQ 0.6876, hdrval 0.8815,
  G1 1.00, mono 0.9711, goal 0.677.
- **HDR: `lp_hdr-lasso0.001-shaped-tau0-f16`** (50 weights, 11,636 B) —
  UPIQ 0.7313 (best of ANY bake measured in this program, incl. all MLPs;
  above raw ssim2-HDR 0.704, still below iwssim-HDR 0.808 / cvvdp 0.758),
  hdrval 0.8859, CID22 0.8347, AIC-4 0.9022.
- **PJND sibling (both domains): `lp_canonhdr15-bvls-raw-tau0.005-f16`**
  (70 weights, 3,738 B, sha256 `1400fa2f86e2…`) — KonJND 0.6696 record,
  CID22 0.8001, UPIQ 0.6679; or the more CID22-conservative
  `lp_canon-bvls-raw-tau0.005-f16` (0.8280 / 0.6014 / 0.6809).

None of these rotates a shipped profile today — that's a user decision
(the SDR pick's KonJND 0.3716 vs A's 0.4185 is a real regression on one
shipping axis, worth an explicit trade call). As sibling profiles or
regression-gate metrics they are strictly better than the 2026-05-28
linear candidate on every axis it was proposed for.

## Files / reproduction

- Script (committed): `scripts/v_next/linear_projections_2026-07-03.py`
  — `gram` (97 s full re-accumulation) → `fit` (108 s, 154 fits) →
  `finalize --keys … --taus …`.
- Probe dir: `/mnt/v/output/zensim-multicodec-probe/linear-probe/`
  — `bakes/*.bin` (41), `verdicts/*.md` (41), `panel_all.tsv`,
  `fits/*.npz` + `fits_run1/` (determinism pair), `grams/*.npz`,
  `gram.log` / `fit.log` / `fit_round2.log` / `finalize_*.log` /
  `panel*.log`, `eval_finalists.sh`, `hdrval_score.py`,
  `determinism_check.py`.
- Eval env: `bake_verdict` @ repo commit 5caf3b75 build (Jul 2),
  features-root `2026-05-15-full-features`, dial grid
  `dial_grid_372col_2026-05-29.parquet`; `upiq_panel.py` on
  `upiq_features_372_pulinear.parquet` (n=380).

## Deviations from the brief

1. No 300k subsample — Gram trick made full-data (2.9M-row) exact fits
   cheaper than any subsample; fits are additionally mix-reweightable
   for free.
2. kadis_cvvdp target variant skipped (falsified direction; documented
   above).
3. bake-level f16/zerobias used my numpy-exact replication of
   `pack_and_calibrate.py`'s order inside `finalize` (single bake emit)
   instead of shelling to the tool — `predict_features_with_bake` isn't
   built in this tree and a cargo build alongside the live w8 training
   was avoidable; parity was verified against the runtime instead
   (§ Runtime/Python parity).
4. Round-2 λ set widened from the pre-registered {bvls, lasso1e-3}
   families to also panel lasso5e-4 (chosen on val axes before any
   round-2 panel was run); all 22 round-2 panels reported.



## Ensembles + residual stack (2026-07-03 evening, follow-on mission)

### Convex linear ensembles — method

A convex blend of RAW-feature-space linear heads collapses to a SINGLE
372→1 linear layer (fold each head's scaler: v_k = w_k/sd_k; per-head
output z-normalized on the anchor set; blended weights = Σ α_k ṽ_k) — so
every "ensemble" here bakes as one tiny identity-scaler layer
(**816–1,119 bytes** f16+lz4) and inherits full determinism. Shaped heads
are excluded (incompatible input transform). Head pool (8, all raw):

| alias | head (fit key @ τ) | axis it owns |
|---|---|---|
| cid | hdrmix-lasso0.002 @0 | CID22 0.8740 |
| kon | canonhdr15-bvls @0.005 | KonJND 0.6696 |
| upq | canonhdr15-lasso0.0005 @0.005 | UPIQ-raw 0.7148 |
| kad | canon-ridge1e-05 @0 | KADID/TID 0.88/0.86 |
| cbv | canon-bvls @0.005 | 2026-05-28 recipe |
| hds | hdr-lasso0.001-raw @0 | pure-HDR ssim2 |
| pjt | pjnd-ridge1e-3 (NEW: konjnd-dense `pjnd_target` target) | guard 0.8096, bigval ≈0 |
| s3h | hdrmix300-lasso0.002 (NEW: f0..f299 only) | ties full head → IW-pool block (f300–371) contributes ~nothing to this fit |

(A 9th candidate — NNLS with ALL 372 weights pinned ≥0 — fit to exactly
zero active weights: full sign-pinning is infeasible for score
prediction on these features. Negative result, head dropped.)

α chosen ONLY on train-legal axes (cid22tr = ssim2-anchored
cid22_train_norm, bigcodec_val stride-7, hdr_val, hdr_valmix,
konjnd-dense-train pjnd_target |SROCC|), corner-normalized; 8,436
combos (pairs step .05, triples step .1, quads step .1) under 4
pre-registered scalarizations (S1 maximin, S2 triple-mirror,
S3 balanced, S4 cid-lean); plus two labeled extras registered BEFORE any
ensemble panel ran (S5 no-guard/no-pjt; P-line = fixed cid↔kon at
α∈{.3,.5,.7}); plus four **panel-informed frontier probes** run AFTER
seeing the P-line panels, labeled as such (cid80/cid85 + two 3-head
blends) to answer the triple-gate question directly. Every baked blend's
panel is reported below — nothing withheld.

### Panel — all 14 ensemble/cascade bakes

| bake | B | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 | UPIQ | G1 | mono | goal |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| casc-bvlsbase-lasso0.0005 | 3903 | 0.8118 | 0.8078 | 0.7786 | 0.4213 | 0.7459 | 0.8530 | 0.6671 | 1.00 | 0.9494 | 0.543 |
| casc-bvlsbase-lasso0.001 | 3859 | 0.8128 | 0.7749 | 0.7519 | 0.5140 | 0.7345 | 0.8463 | 0.6613 | 1.00 | 0.9456 | 0.525 |
| ens-F3-c45k25u30 | 900 | 0.8572 | 0.8261 | 0.7994 | 0.5236 | 0.7580 | 0.8595 | 0.6881 | 1.00 | 0.9704 | 0.606 |
| ens-F3-c55k25u20 | 905 | 0.8636 | 0.8197 | 0.8006 | 0.5380 | 0.7632 | 0.8661 | 0.6857 | 1.00 | 0.9709 | 0.614 |
| ens-Pline-cid30 | 821 | 0.8256 | 0.8021 | 0.7859 | 0.6558 | 0.7396 | 0.8308 | 0.6761 | 1.00 | 0.9496 | 0.546 |
| ens-Pline-cid50 | 816 | 0.8454 | 0.8104 | 0.7974 | 0.6335 | 0.7545 | 0.8489 | 0.6808 | 1.00 | 0.9560 | 0.593 |
| ens-Pline-cid70 | 829 | 0.8653 | 0.8065 | 0.8004 | 0.5862 | 0.7704 | 0.8748 | 0.6830 | 1.00 | 0.9672 | 0.626 |
| ens-Pline-cid80 | 823 | 0.8733 | 0.8017 | 0.7998 | 0.5439 | 0.7775 | 0.8906 | 0.6846 | 1.00 | 0.9711 | 0.644 |
| ens-Pline-cid85 | 832 | 0.8759 | 0.7989 | 0.7990 | 0.5135 | 0.7816 | 0.8992 | 0.6844 | 1.00 | 0.9709 | 0.654 |
| ens-S1maximin | 1095 | 0.4762 | 0.3692 | 0.1354 | 0.7556 | 0.5982 | 0.7916 | 0.4715 | 1.00 | 0.8775 | 0.394 |
| ens-S2triple | 1095 | 0.4762 | 0.3692 | 0.1354 | 0.7556 | 0.5982 | 0.7916 | 0.4715 | 1.00 | 0.8775 | 0.394 |
| ens-S3balance | 1103 | 0.5964 | 0.4359 | 0.2195 | 0.7195 | 0.6633 | 0.8466 | 0.5598 | 1.00 | 0.9049 | 0.416 |
| ens-S4cidlean | 1103 | 0.5891 | 0.4428 | 0.2207 | 0.7274 | 0.6531 | 0.8403 | 0.5682 | 1.00 | 0.9026 | 0.411 |
| ens-S5noguard | 1119 | 0.8793 | 0.7879 | 0.8055 | 0.3497 | 0.7821 | 0.8875 | 0.6403 | 1.00 | 0.9602 | 0.659 |

### Reading

- **`ens-Pline-cid80` (cid:0.80 + kon:0.20, 823 bytes, 95 active) is the
  strongest small model this program has produced.** vs shipped A
  (27,316 B MLP): CID22 0.8733 vs 0.8657 (+0.008), KADID 0.8017 vs
  0.7933 (+0.008), TID 0.7998 vs 0.7927 (+0.007), KonJND 0.5439 vs
  0.4185 (**+0.125**), AIC-3 0.7775 vs 0.7680 (+0.010), AIC-4 0.8906 vs
  0.8854 (+0.005), goal 0.644 vs 0.622 — **beats A on 7 of 9 tracked
  axes** at 1/33rd the size; loses UPIQ 0.6846 vs 0.6933 (−0.009) and
  mono 0.9711 vs 0.9747 (−0.004). `ens-Pline-cid85` pushes CID22 to
  0.8759 with KonJND 0.5135.
- **The CID22↔KonJND trade is strongly non-linear in α — in our favor.**
  Linear interpolation between the cid (0.8740/0.3716) and kon
  (0.8001/0.6696) corners predicts ~0.52 KonJND at the midpoint; the
  measured cid50 blend holds **0.6335**, and even at α_cid=0.85 KonJND
  is 0.5135. Mechanism: near PJND threshold the cid head is nearly flat
  (ties); the small kon component breaks those ties with the BVLS head's
  sign-stable ordering, while the cid head dominates the global
  ordering. This is exactly the "linear raw outputs have real dynamic
  range, so convex blends preserve rank" hypothesis, confirmed — and the
  spline-on-blend refit kept mono 0.947–0.971 / G1 1.00 on every P-line
  and F3 point.
- **Triple gate (CID22 ≥0.87 ∧ KonJND ≥0.50 ∧ UPIQ ≥0.70): 2 of 3 legs
  HIT, decisively — UPIQ leg fails.** cid80 = 0.8733 ✓ / 0.5439 ✓ /
  0.6846 ✗ (−0.015); cid85 = 0.8759 ✓ / 0.5135 ✓ / 0.6844 ✗. Pulling
  UPIQ up via the upq head (F3 blends) tops out at 0.6881 while giving
  back CID22 (0.8572–0.8636). The only ≥0.70-UPIQ candidates are the
  shaped pure-HDR head (0.7313, not raw-blendable) and specialist
  singles; on raw-space blends the UPIQ ceiling observed is ~0.69. A
  Pareto point holding all three was NOT found.
- **`pjt`-blends reach KonJND 0.7556** (S1/S2: upq:0.5+pjt:0.5) — above
  the G5 0.70 floor that two MLP architectures were falsified against
  (v42/v43) — but at catastrophic cost elsewhere (CID22 0.476, TID 0.135,
  AIC-3 0.598; G5's AIC-3 leg still fails). The pjnd_target head is a
  real PJND mechanism, not a metric artifact (guard 0.59 on the S1 blend
  transferred to val KonJND 0.7556) — worth a dedicated
  KonJND-specialist slot if one is ever needed, and a possible G5 route
  if its AIC-3 cost can be bought back.
- **The 4 blind scalarizations (S1–S4) all mis-fired** (CID22 0.48–0.60)
  — the guard axis over-rewarded pjt content and cid22tr normalization
  compressed the real CID22 differences. The informative ensembles came
  from the fixed P-line + the no-guard S5 + frontier probes. Lesson:
  with a self-owned axis in the pool (pjt owns guard), corner
  normalization makes that axis a magnet; exclude self-owned axes from
  scalarization or normalize on a pool without the owner.
- **2-stage cascade (BVLS base + sparse lasso correction on train
  residual): dominated by direct blends** — casc-lasso1e-3 = CID22
  0.8128/KonJND 0.5140 vs cid50's 0.8454/0.6335 at 1/5th the size.
  Falsified as a frontier-mover (the correction re-optimizes the same
  trade the blend explores, with less control).
- **`ens-S5noguard` (kad:0.30+hds:0.20+s3h:0.50, 1,119 B) is the best
  CID22 of the entire program: 0.8793** — +0.0085 over the best MLP
  (t1dro51_s31 0.8708), +0.0136 over A — with TID 0.8055, AIC-3 0.7821,
  AIC-4 0.8875, goal 0.659; its costs are KonJND 0.3497 and UPIQ 0.6403.
  Selected blind by the no-guard quality-axes scalarization (disclosure:
  its cid22tr 0.978 selection value was partly self-favored — the kad
  head's mix contains cid22_train — but CID22-val at 0.8793 is a genuine
  49-ref holdout number).

### Residual-stack prep (deterministic base + learnable correction)

Corpora for a future residual-MLP experiment (NOT trained here — the
local training slot is occupied by the w8 run). Base = the SDR pick
`lp_hdrmix-lasso0.002-raw-tau0-f16.bin` applied exactly as the bake
stores it (f16 weights, f32 scaler/math).

Definition: `residual_target = human_score − clip01(a·clip(linear_pred,
clamp_lo, clamp_hi) + b)` with (a, b) OLS-fit on each TRAIN file's
clamped preds and REUSED for its val file (no val-fit leakage); the
inner clip bounds the base to the anchor-observed raw domain (the dial
spline's trusted range), the outer clip01 bounds the affined base to
[0,1]; `linear_pred` (raw, un-clamped, un-affined) also stored per row
for audit. Runtime composition: `final = clip01(a·linear_base(x) + b) +
residual_mlp(x)` — a hard sum over four constants + one clip, so a
collapsed/degenerate residual head degrades toward the deterministic
base instead of toward a broken constant: **the collapse mode is
bounded by construction.** Dial spline then fits on the composed output
(same pack-then-calibrate order).

Files at `/mnt/v/output/zensim-multicodec-probe/linear-probe/residual/`
(+ `_MANIFEST.json` with base-bake sha256, per-file sha256, affine, and
residual ranges; parquet KV metadata carries the same):

Base bake sha256:
`bdb5d63ea699e9bf28c3f5f24a895f3bab105d74d5df9c211406d8637a301692`

Pred clamp domain (anchor min/max): [-4.9061, 1.1274]; composition clip01 bounds the affined base to [0,1], so residual_target ∈ [−1,1] **by construction** (observed: bigcodec [−0.79, +0.94], hdr [−0.55, +0.22]).

| file | rows | affine (a, b) | residual range | sha256 |
|---|--:|---|---|---|
| bigcodec_traindigits_residual_2026-07-03.parquet | 2,946,036 | (0.3405, +0.4095) | [-0.7880, +0.9349] | `80599586e33d528345b364eaf1c4f5cc4dd8ed18a615a0fa4163086d624d3a39` |
| bigcodec_valdigits_residual_2026-07-03.parquet | 147,067 | (0.3405, +0.4095) | [-0.7934, +0.9386] | `c1b895a59d2b7bc0506ef1941d396325800043e96d7f6cb0229927bded928542` |
| hdr_zenjxl_v3_traindigits_residual_2026-07-03.parquet | 7,410 | (0.8194, +0.0683) | [-0.5474, +0.2123] | `fa7d767adb94f96c58cd6c9a43c415b457f65a13ee07f9b5f5c6645e5ef6786e` |
| hdr_zenjxl_v3_valdigits_residual_2026-07-03.parquet | 3,900 | (0.8194, +0.0683) | [-0.5147, +0.2194] | `764a4b92765ae43ccc9cbcd67428258fc4537b75cda146dc8c0cfe97beb38438` |

Two earlier iterations were caught by the contract and fixed in the DATA (never by relaxing the contract): unclamped preds reached residual +158 on bigcodec valdigits (OOD extrapolation of the sparse HDR-fit base also squashed the OLS slope to 0.17), and clamp-only still left a +2.19 tail (rows where the base is confidently wrong at the clamp floor). Superseded files kept as `*.unclamped.bak` / `*.clamponly.bak`.

All four validated with
`validate_parquet.py --kind train --target-col residual_target --target-range=-1.001,1.001`
(the tight [−1,1] contract the clip01 composition guarantees):

- C1 footer / C2 372-feature completeness / C3 nulls / C4 NaN-Inf / C5 target
  present+in-range / C7 rows: **PASS on all four files**.
- `hdr_zenjxl_v3_valdigits_residual`: all checks PASS.
- `bigcodec_*_residual`: **C10 duplicate-(f0,target) FAIL at 18.80% / 2.05% —
  inherited verbatim from the source corpora** (source
  `bigcodec_traindigits_2026-07-02.parquet` fails C10 at the identical
  18.80%; the validator's own message documents this as the known
  modes_full no-op-knob duplication from 2026-07-02). Row-for-row
  derivation preserves it by design.
- `hdr_zenjxl_v3_traindigits_residual`: **C6 one all-constant feature col —
  also inherited verbatim** (source fails C6 identically; a feature that is
  constant across the 7,410-row HDR train split).

No new defects were introduced by the derivation; full log at
`linear-probe/residual_validate.log`.
