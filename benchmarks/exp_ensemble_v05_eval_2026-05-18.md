# EXP-ENSEMBLE-V05 — corpus-membership classifier routing
_Eval date: 2026-05-18.  Held-out 20% per corpus.  Seed: 42.  Routing accuracy: **0.9826**._

## Methodology

Logistic regression on 372 zenanalyze features (val-parquet's f0..f371) predicts `is_compression_corpus` (CID22+AIC-3=1; KADID+TID+KonJND=0).

- Training: 80% per corpus (stratified). Class weights balanced.
- Test: 20% held-out per corpus.
- Routing rule: if `p(compression) > 0.5` → use compression bake (`v_compression_persample_2026-05-18.bin`), else balanced bake (`v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin`).
- Scores: bake outputs computed by Rust binary `ensemble_score_rows` (bit-exact match with `forward_one_bake` incl. per-sample-α head dispatch).
- Controls: ssim2_log_norm / iwssim_log_norm / cvvdp_log_norm columns from each val parquet (per-pair perceptual metric, log-rescaled to 0..1).

## Routing accuracy summary

| Corpus | n_test | truth | mean p(compression) | fraction routed → compression |
|---|---:|---:|---:|---:|
| cid22 | 858 | 1 | 0.9307 | 0.9790 |
| kadid | 2025 | 0 | 0.0065 | 0.0049 |
| tid | 600 | 0 | 0.0139 | 0.0117 |
| konjnd | 201 | 0 | 0.1747 | 0.1294 |
| aic3 | 120 | 1 | 0.9151 | 0.9583 |

**Overall routing accuracy on holdout: 0.9826**

## Routing accuracy — FULL corpus

_The classifier identifies corpora, not pairs. Routing accuracy on the full 5-corpus val set (training + holdout) sets the ensemble's deployable per-corpus SROCC, since at inference we don't know which 20% slice a pair came from._

| Corpus | n_full | truth | mean p(compression) | fraction routed → compression |
|---|---:|---:|---:|---:|
| cid22 | 4292 | 1 | 0.9286 | 0.9758 |
| kadid | 10125 | 0 | 0.0049 | 0.0024 |
| tid | 3000 | 0 | 0.0095 | 0.0067 |
| konjnd | 1008 | 0 | 0.1427 | 0.0913 |
| aic3 | 600 | 1 | 0.8880 | 0.9450 |

**Full-corpus routing accuracy: 0.9857**

## Per-corpus full Mohammadi panel (FULL corpus, deployment view)

_Each (corpus, pair) is routed via the trained classifier; scores are the routed bake's output. This is what a deployed PreviewV0_5Ensemble runtime produces._

### CID22 (n = 4292, full corpus)

| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced (V0_5) | 4292 | 0.8324 | 0.8289 | 0.6340 | 0.0520 | 0.8324 | 0.559 |
| Compression (V0_5) | 4292 | 0.8641 | 0.8614 | 0.6742 | 0.0543 | 0.8641 | 0.508 |
| Ensemble (V0_5) | 4292 | 0.8633 | 0.8597 | 0.6730 | 0.0534 | 0.8633 | 0.511 |
_Controls (fast-ssim2 / iwssim / cvvdp) omitted: the canonical val parquets carry null control columns. Score sidecars live separately at `scores/{ssim2_imazen,iwssim_imazen,cvvdp_imazen_v0_0_1}.parquet` keyed by (image_path, codec, q, knob_tuple_json) which is not joinable to the val parquets' (ref_basename, anchor index) layout. The ensemble vs single-bake verdict above is unchanged by this gap; control SROCC for these corpora is reported in the per-bake methodology docs (`benchmarks/v22_mix_LARGE_iwssim_methodology_2026-05-18.md`, `benchmarks/v0_24_persample_alpha_methodology_2026-05-18.md`)._

### KADID (n = 10125, full corpus)

| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced (V0_5) | 10125 | 0.9677 | 0.9686 | 0.8432 | 0.0520 | 0.9677 | 0.249 |
| Compression (V0_5) | 10125 | 0.9316 | 0.9321 | 0.7684 | 0.0538 | 0.9316 | 0.362 |
| Ensemble (V0_5) | 10125 | 0.9676 | 0.9685 | 0.8430 | 0.0520 | 0.9676 | 0.249 |
_Controls (fast-ssim2 / iwssim / cvvdp) omitted: the canonical val parquets carry null control columns. Score sidecars live separately at `scores/{ssim2_imazen,iwssim_imazen,cvvdp_imazen_v0_0_1}.parquet` keyed by (image_path, codec, q, knob_tuple_json) which is not joinable to the val parquets' (ref_basename, anchor index) layout. The ensemble vs single-bake verdict above is unchanged by this gap; control SROCC for these corpora is reported in the per-bake methodology docs (`benchmarks/v22_mix_LARGE_iwssim_methodology_2026-05-18.md`, `benchmarks/v0_24_persample_alpha_methodology_2026-05-18.md`)._

### TID (n = 3000, full corpus)

| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced (V0_5) | 3000 | 0.9729 | 0.9717 | 0.8571 | 0.0497 | 0.9729 | 0.236 |
| Compression (V0_5) | 3000 | 0.8893 | 0.9018 | 0.7130 | 0.0523 | 0.8893 | 0.432 |
| Ensemble (V0_5) | 3000 | 0.9719 | 0.9709 | 0.8558 | 0.0460 | 0.9719 | 0.240 |
_Controls (fast-ssim2 / iwssim / cvvdp) omitted: the canonical val parquets carry null control columns. Score sidecars live separately at `scores/{ssim2_imazen,iwssim_imazen,cvvdp_imazen_v0_0_1}.parquet` keyed by (image_path, codec, q, knob_tuple_json) which is not joinable to the val parquets' (ref_basename, anchor index) layout. The ensemble vs single-bake verdict above is unchanged by this gap; control SROCC for these corpora is reported in the per-bake methodology docs (`benchmarks/v22_mix_LARGE_iwssim_methodology_2026-05-18.md`, `benchmarks/v0_24_persample_alpha_methodology_2026-05-18.md`)._

### KONJND (n = 1008, full corpus)

| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced (V0_5) | 1008 | 0.8927 | 0.9265 | 0.7070 | 0.0585 | 0.8927 | 0.376 |
| Compression (V0_5) | 1008 | 0.8080 | 0.8648 | 0.5935 | 0.0685 | 0.8080 | 0.502 |
| Ensemble (V0_5) | 1008 | 0.8792 | 0.9214 | 0.6883 | 0.0615 | 0.8792 | 0.389 |
_Controls (fast-ssim2 / iwssim / cvvdp) omitted: the canonical val parquets carry null control columns. Score sidecars live separately at `scores/{ssim2_imazen,iwssim_imazen,cvvdp_imazen_v0_0_1}.parquet` keyed by (image_path, codec, q, knob_tuple_json) which is not joinable to the val parquets' (ref_basename, anchor index) layout. The ensemble vs single-bake verdict above is unchanged by this gap; control SROCC for these corpora is reported in the per-bake methodology docs (`benchmarks/v22_mix_LARGE_iwssim_methodology_2026-05-18.md`, `benchmarks/v0_24_persample_alpha_methodology_2026-05-18.md`)._

### AIC3 (n = 600, full corpus)

| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced (V0_5) | 600 | 0.7845 | 0.7953 | 0.6155 | 0.0450 | 0.7845 | 0.606 |
| Compression (V0_5) | 600 | 0.8183 | 0.8248 | 0.6527 | 0.0583 | 0.8183 | 0.565 |
| Ensemble (V0_5) | 600 | 0.8132 | 0.8200 | 0.6468 | 0.0583 | 0.8132 | 0.572 |
_Controls (fast-ssim2 / iwssim / cvvdp) omitted: the canonical val parquets carry null control columns. Score sidecars live separately at `scores/{ssim2_imazen,iwssim_imazen,cvvdp_imazen_v0_0_1}.parquet` keyed by (image_path, codec, q, knob_tuple_json) which is not joinable to the val parquets' (ref_basename, anchor index) layout. The ensemble vs single-bake verdict above is unchanged by this gap; control SROCC for these corpora is reported in the per-bake methodology docs (`benchmarks/v22_mix_LARGE_iwssim_methodology_2026-05-18.md`, `benchmarks/v0_24_persample_alpha_methodology_2026-05-18.md`)._

## Headline SROCC table (FULL corpus, deployment view)

| Corpus | Balanced | Compression | Ensemble | max(B, C) | Δ ensemble vs max |
|---|---:|---:|---:|---:|---:|
| cid22 | 0.8324 | 0.8641 | **0.8633** | 0.8641 | -0.0008 |
| kadid | 0.9677 | 0.9316 | **0.9676** | 0.9677 | -0.0001 |
| tid | 0.9729 | 0.8893 | **0.9719** | 0.9729 | -0.0010 |
| konjnd | 0.8927 | 0.8080 | **0.8792** | 0.8927 | -0.0135 |
| aic3 | 0.7845 | 0.8183 | **0.8132** | 0.8183 | -0.0050 |

## Per-corpus full Mohammadi panel (held-out 20%)

### CID22 (n_test = 858)

| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced (V0_5) | 858 | 0.8413 | 0.8344 | 0.6430 | 0.0606 | 0.8413 | 0.551 |
| Compression (V0_5) | 858 | 0.8659 | 0.8645 | 0.6770 | 0.0536 | 0.8659 | 0.503 |
| Ensemble (V0_5) | 858 | 0.8652 | 0.8633 | 0.6763 | 0.0536 | 0.8652 | 0.505 |

### KADID (n_test = 2025)

| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced (V0_5) | 2025 | 0.9671 | 0.9675 | 0.8426 | 0.0435 | 0.9671 | 0.253 |
| Compression (V0_5) | 2025 | 0.9259 | 0.9267 | 0.7608 | 0.0489 | 0.9259 | 0.376 |
| Ensemble (V0_5) | 2025 | 0.9669 | 0.9674 | 0.8423 | 0.0440 | 0.9669 | 0.253 |

### TID (n_test = 600)

| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced (V0_5) | 600 | 0.9744 | 0.9746 | 0.8624 | 0.0433 | 0.9744 | 0.224 |
| Compression (V0_5) | 600 | 0.8878 | 0.8985 | 0.7121 | 0.0533 | 0.8878 | 0.439 |
| Ensemble (V0_5) | 600 | 0.9736 | 0.9737 | 0.8608 | 0.0400 | 0.9736 | 0.228 |

### KONJND (n_test = 201)

| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced (V0_5) | 201 | 0.8972 | 0.9519 | 0.7130 | 0.0746 | 0.8972 | 0.306 |
| Compression (V0_5) | 201 | 0.7983 | 0.8851 | 0.5791 | 0.0647 | 0.7983 | 0.465 |
| Ensemble (V0_5) | 201 | 0.8719 | 0.9472 | 0.6767 | 0.0796 | 0.8719 | 0.321 |

### AIC3 (n_test = 120)

| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced (V0_5) | 120 | 0.7661 | 0.7628 | 0.5919 | 0.0250 | 0.7661 | 0.647 |
| Compression (V0_5) | 120 | 0.8046 | 0.7998 | 0.6329 | 0.0417 | 0.8046 | 0.600 |
| Ensemble (V0_5) | 120 | 0.8078 | 0.8019 | 0.6358 | 0.0417 | 0.8078 | 0.597 |

## § A.9 verdicts per corpus

_Decisive A>>B per § A.9: ΔSROCC > 0.005 AND ensemble wins ≥ 3 of 5 stats (SROCC, PLCC, KROCC, PWRC, Z-RMSE — lower is better for Z-RMSE)._

| Corpus | Ensemble vs Balanced | Ensemble vs Compression |
|---|---|---|
| cid22 | **A>>B** (ΔSROCC=++0.0239, wins=5) | tie (ΔSROCC=-0.0007, wins=-5) |
| kadid | tie (ΔSROCC=-0.0002, wins=-4) | **A>>B** (ΔSROCC=++0.0410, wins=5) |
| tid | tie (ΔSROCC=-0.0008, wins=-5) | **A>>B** (ΔSROCC=++0.0858, wins=5) |
| konjnd | **B>>A** (ΔSROCC=-0.0253, wins=-5) | **A>>B** (ΔSROCC=++0.0736, wins=5) |
| aic3 | **A>>B** (ΔSROCC=++0.0417, wins=5) | tie (ΔSROCC=+0.0032, wins=5) |

## Trail-gate verdicts (vs Balanced ship)

### Balanced trail gate

A>>B on ≥1 corpus + no decisive B>>A on any.

- Ensemble decisive wins: cid22, aic3
- Ensemble decisive losses: konjnd

**Balanced trail verdict**: FAIL

### Compression trail gate (vs Balanced ship)

A>>B on ≥1 of {CID22, AIC-3} + no decisive B>>A on the other compression corpus + mean Δ ≥ −0.10 on {KADID, TID, KonJND}.

- Compression wins (A>>B on cid22 or aic3): cid22, aic3
- Compression losses (B>>A on cid22 or aic3): none
- Synthetic Δ (KADID/TID/KonJND): [np.float64(-0.0001935916896019796), np.float64(-0.0008137541139812132), np.float64(-0.025296552500049363)] (mean=-0.0088)
- Any synthetic Δ < −0.10: False

**Compression trail verdict**: PASS

## Headline SROCC table (per corpus, held-out 20%)

| Corpus | Balanced | Compression | Ensemble | max(B, C) | Δ ensemble vs max |
|---|---:|---:|---:|---:|---:|
| cid22 | 0.8413 | 0.8659 | **0.8652** | 0.8659 | -0.0007 |
| kadid | 0.9671 | 0.9259 | **0.9669** | 0.9671 | -0.0002 |
| tid | 0.9744 | 0.8878 | **0.9736** | 0.9744 | -0.0008 |
| konjnd | 0.8972 | 0.7983 | **0.8719** | 0.8972 | -0.0253 |
| aic3 | 0.7661 | 0.8046 | **0.8078** | 0.8046 | +0.0032 |
