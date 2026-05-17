# `benchmarks/` — INDEX

Methodology docs, falsification logs, sweep outputs, perf
benchmarks, and bake binaries. 76 markdown files as of 2026-05-16,
organized by theme + chronology. Each entry is one line:
file path + one-line summary.

If you arrived here cold, start with [RESEARCH.md](../RESEARCH.md)
first — this index is a deep dive for follow-up.

## Headline reference cards

These are the docs to read first for a given purpose:

| Purpose | Doc |
|---|---|
| **Most recent full-stat panel comparison across all bakes** | [`v0_20_all_bakes_stat_comparison_2026-05-15.md`](v0_20_all_bakes_stat_comparison_2026-05-15.md) |
| **Currently-shipped bake methodology** | [`v0_18_methodology_2026-05-13.md`](v0_18_methodology_2026-05-13.md) + [`v0_18_ship_reference_card_2026-05-14.md`](v0_18_ship_reference_card_2026-05-14.md) |
| **Runtime cost of extended + IW features** | [`extended_iw_runtime_perf_optimized_2026-05-15.md`](extended_iw_runtime_perf_optimized_2026-05-15.md) |
| **Falsification re-evaluation triage + results (today's methodology shift)** | [`falsification_reeval_triage_2026-05-15.md`](falsification_reeval_triage_2026-05-15.md), [`falsification_reeval_results_2026-05-15.md`](falsification_reeval_results_2026-05-15.md) |
| **Mohammadi 2025 stat-panel reproduction proof** | [`mohammadi_2025_verification_2026-05-14.md`](mohammadi_2025_verification_2026-05-14.md) |

## V_X bake methodology + ship decisions (chronological)

### Pre-V_18 (early experiments)

- [`v04_3norm_baseline_2026-05-01.md`](v04_3norm_baseline_2026-05-01.md) — V_4 butter-3norm baseline.
- [`v04_3norm_compat_2026-05-01.md`](v04_3norm_compat_2026-05-01.md), [`v04_ssim2_holdout_baseline_2026-05-01.md`](v04_ssim2_holdout_baseline_2026-05-01.md), [`v04_ssim2_holdout_calibration_2026-05-01.md`](v04_ssim2_holdout_calibration_2026-05-01.md), [`v04_ssim2_holdout_compat_2026-05-01.md`](v04_ssim2_holdout_compat_2026-05-01.md), [`v04_to_ssim2_anchors_2026-05-01.md`](v04_to_ssim2_anchors_2026-05-01.md), [`v04_calibrate_mapping_2026-05-01.md`](v04_calibrate_mapping_2026-05-01.md) — V_4 ssim2-target compatibility audit.
- [`profile_compat_v02_v04_2026-05-01.md`](profile_compat_v02_v04_2026-05-01.md) — V_2 (linear) vs V_4 (MLP) per-pair compatibility.
- [`baseline_metrics_with_konjnd_2026-05-01.md`](baseline_metrics_with_konjnd_2026-05-01.md) — KonJND-1k anchor alongside KADID/TID/CID22 for the first time.
- [`v0_6_eval_2026-05-11.md`](v0_6_eval_2026-05-11.md) — V_6 eval.

### V_18 ship lineage (2026-05-12 → 2026-05-14)

- [`v0_18_methodology_2026-05-13.md`](v0_18_methodology_2026-05-13.md) — **The shipped V_18 methodology doc.** Read this for the canonical recipe (3-way concat of base + cycle-14 s1 + cycle-14 s42, h=128, α/β calibration).
- [`v0_18_ship_reference_card_2026-05-14.md`](v0_18_ship_reference_card_2026-05-14.md) — Quick reference: V_18 ship's per-corpus + per-band numbers.
- [`v0_18_zerobiased_lz4_10band_2026-05-14.md`](v0_18_zerobiased_lz4_10band_2026-05-14.md) — V_18 zerobiased + LZ4 ship-form 10-band SROCC reproducing the F32 numbers.
- [`v0_18_repro_and_cross_corpus_analysis_REVERTED_2026-05-14.md`](v0_18_repro_and_cross_corpus_analysis_REVERTED_2026-05-14.md) — REVERTED: cross-corpus contamination audit reverted (loose d≤16 dHash false-positives).
- [`v0_18_1_full218k_noship_2026-05-14.md`](v0_18_1_full218k_noship_2026-05-14.md) — V_18.1: full 218k retrain. **No-ship** (−0.011 CID22).

### V_19 (KADID/TID-purge — reverted)

- [`v0_19_methodology_initial_failure_REVERTED_2026-05-14.md`](v0_19_methodology_initial_failure_REVERTED_2026-05-14.md) — V_19 retrain on KADID/TID-purged synth. **REVERTED** (CID22 −0.015).
- [`v0_19_REVERTED_2026-05-14.md`](v0_19_REVERTED_2026-05-14.md) — V_19 final REVERTED note.
- [`v0_19_smaller_retrain_2026-05-13.md`](v0_19_smaller_retrain_2026-05-13.md), [`v0_19_10band_2026-05-14.md`](v0_19_10band_2026-05-14.md) — V_19 component evals.

### V_20 cycle (input-shaping, IW-SSIM exploration, 2026-05-14 → 2026-05-15)

- [`v0_20_v0_21_design_2026-05-14.md`](v0_20_v0_21_design_2026-05-14.md) — V_20/V_21 design doc.
- [`v0_20_input_shaping_methodology_2026-05-15.md`](v0_20_input_shaping_methodology_2026-05-15.md) — **V_20 IS methodology**. 98 transforms (winsor_p99 / signed_cbrt / signed_sqrt / quantile_bins / log1p).
- [`v0_20_three_directions_summary_2026-05-15.md`](v0_20_three_directions_summary_2026-05-15.md) — D1 (concat) / D2 (ensemble) / D3 (tighter transforms) summary.
- [`v0_20_d2_ensemble_v18+is_2026-05-15.md`](v0_20_d2_ensemble_v18+is_2026-05-15.md) — **D2: V_18 ship + V_20 IS multi-bake ensemble** (live as `PreviewV0_4`).
- [`v0_20_4_runtime_mix_sweep_2026-05-15.md`](v0_20_4_runtime_mix_sweep_2026-05-15.md) — Runtime mix α sweep for the D2 ensemble.
- [`v0_20_l0_norms_2026-05-15.md`](v0_20_l0_norms_2026-05-15.md) — **GD-selection analysis**. Confirms IW + masked features ARE selected by gradient descent (Pearson correlation < 0.85 between weight maps).
- [`v0_20_extended_falsification_2026-05-15.md`](v0_20_extended_falsification_2026-05-15.md) — 300-feat extended bake. Falsified on CID22 SROCC (but per the new methodology, that gate is suspect — see `falsification_reeval_*`).
- [`v0_20_low_n_band_analysis_2026-05-15.md`](v0_20_low_n_band_analysis_2026-05-15.md) — Low-n band ceiling analysis. Empirical SROCC max per cell when n<100.

### V_20a IW-SSIM weighted pooling (Wang & Li 2011)

- [`v0_20a_iwssim_design_2026-05-14.md`](v0_20a_iwssim_design_2026-05-14.md) — Design doc for IW pooling.
- [`v0_20a_iw_perf_2026-05-14.md`](v0_20a_iw_perf_2026-05-14.md) — Per-pair compute cost of the IW pool.
- [`v0_20a_smoke_methodology_2026-05-14.md`](v0_20a_smoke_methodology_2026-05-14.md), [`v0_20a_sweep_methodology_2026-05-14.md`](v0_20a_sweep_methodology_2026-05-14.md) — Sweep methodology.
- [`v0_20a_path_a_falsification_2026-05-14.md`](v0_20a_path_a_falsification_2026-05-14.md) — **V_20a falsification doc** (CID22 catastrophic, TID/KADID wins — see "Methodology shift" in [`falsification_reeval_results_2026-05-15.md`](falsification_reeval_results_2026-05-15.md) for re-eval).
- [`v0_20a_ship_form_verification_2026-05-14.md`](v0_20a_ship_form_verification_2026-05-14.md) — Hypothetical ship form (not landed).

### V_20b distortion-manifold pre-training (Su 2023)

- [`v0_20b_distortion_manifold_design_2026-05-15.md`](v0_20b_distortion_manifold_design_2026-05-15.md) — Design doc. Falsified for CID22 transfer; KADID + TID win.

### V_20d JND-anchored output calibration (queued)

- [`v0_20d_jnd_calibration_design_2026-05-14.md`](v0_20d_jnd_calibration_design_2026-05-14.md) — Design doc. Not yet trained.

### Falsification re-evaluation (2026-05-15 methodology shift)

- [`falsification_reeval_triage_2026-05-15.md`](falsification_reeval_triage_2026-05-15.md) — **Catalogue of every SROCC-gated falsification**, classified by re-eval priority.
- [`falsification_reeval_results_2026-05-15.md`](falsification_reeval_results_2026-05-15.md) — **Full-panel re-eval of 8 falsified bakes**. All 8 confirmed against original SROCC-only verdict — but with the caveat that all bakes were trained against ssim2 targets, so the gate is structurally rigged toward V_18.

### Cycle outcomes (recovery cycle, 2026-05-11 → 2026-05-13)

- [`cycle_summary_2026-05-11.md`](cycle_summary_2026-05-11.md) — Overall cycle summary.
- [`cycle_6_finals_2026-05-12.md`](cycle_6_finals_2026-05-12.md) — Cycle 6 finals.
- [`cycle_7_dssim_outcomes_2026-05-12.md`](cycle_7_dssim_outcomes_2026-05-12.md) — Cycle 7 dssim co-training **falsified** (V_24/V_27 variants).
- [`cycle_8_konjnd_pareto_outcomes_2026-05-13.md`](cycle_8_konjnd_pareto_outcomes_2026-05-13.md) — Cycle 8 KonJND Pareto.
- [`cycle_9_lowq_boost_outcomes_2026-05-13.md`](cycle_9_lowq_boost_outcomes_2026-05-13.md) — Cycle 9 low-q row-weight boost **falsified**.
- [`cycle_9b_pair_boost_outcomes_2026-05-13.md`](cycle_9b_pair_boost_outcomes_2026-05-13.md) — Cycle 9b pair-resampling boost **falsified**.
- [`cycle_10_kadid_tid_outcomes_2026-05-13.md`](cycle_10_kadid_tid_outcomes_2026-05-13.md) — Cycle 10 KADID/TID hyperparam sweep.
- [`cycle_12_midq_boost_outcomes_2026-05-13.md`](cycle_12_midq_boost_outcomes_2026-05-13.md) — Cycle 12 mid-q boost ("first positive signal", falsified at multi-seed).
- [`cycle_14_per_band_tv_outcomes_2026-05-13.md`](cycle_14_per_band_tv_outcomes_2026-05-13.md) — Cycle 14 per-band TV (foundation of V_18 ship).
- [`pareto_2026-05-11.md`](pareto_2026-05-11.md) — Pareto front analysis.
- [`tv_smoothness_sweep_2026-05-10.md`](tv_smoothness_sweep_2026-05-10.md) — TV smoothness sweep across weights.

### Specific recovery-cycle variants

- [`v0_24_dssim_cotrain_v1_result_2026-05-12.md`](v0_24_dssim_cotrain_v1_result_2026-05-12.md), [`v0_24_v2_dssim_cotrain_2026-05-12.md`](v0_24_v2_dssim_cotrain_2026-05-12.md) — V_24 v1/v2 dssim co-training results.
- [`v0_26_konjnd_aligned_2026-05-12.md`](v0_26_konjnd_aligned_2026-05-12.md), [`v0_27_konjnd_dssim01_2026-05-12.md`](v0_27_konjnd_dssim01_2026-05-12.md) — V_26/V_27 KonJND + dssim variants.

## Champions + corpus comparison

- [`champion_2026-05-10.md`](champion_2026-05-10.md), [`champion_candidate_2026-05-10.md`](champion_candidate_2026-05-10.md) — Champion register entries.

### Per-corpus full-panel reports

- [`aic3_zensim_vs_baselines_2026-05-12.md`](aic3_zensim_vs_baselines_2026-05-12.md) — AIC-3 zensim vs baselines.
- [`aic4_zensim_vs_paper_metrics_2026-05-12.md`](aic4_zensim_vs_paper_metrics_2026-05-12.md) — AIC-4 zensim vs paper metrics.
- [`aic_combined_per_codec_2026-05-12.md`](aic_combined_per_codec_2026-05-12.md), [`aic_per_codec_v0_16_2026-05-12.md`](aic_per_codec_v0_16_2026-05-12.md) — Per-codec AIC.
- [`cid22_full_v0_16_vs_ssim2_2026-05-12.md`](cid22_full_v0_16_vs_ssim2_2026-05-12.md), [`cid22_per_codec_v0_16_2026-05-12.md`](cid22_per_codec_v0_16_2026-05-12.md) — V_16 CID22 reports.

### Statistical methodology

- [`mohammadi_2025_verification_2026-05-14.md`](mohammadi_2025_verification_2026-05-14.md) — Our logistic rescale reproduces Mohammadi 2025 anchor Z-RMSE values.

## Performance + optimization

- [`extended_iw_runtime_perf_2026-05-15.md`](extended_iw_runtime_perf_2026-05-15.md) — Original 4-permutation runtime cost benchmark.
- [`extended_iw_runtime_perf_optimized_2026-05-15.md`](extended_iw_runtime_perf_optimized_2026-05-15.md) — **Post-optimization**: combined Extended+IW dropped from +25 % → +12 % at 1024² via fused 2-mask SIMD + V-blur-only on H-blurred sigma buffers + IW-only mu1 bug fix.
- [`iw_perf_hotspots_2026-05-15.md`](iw_perf_hotspots_2026-05-15.md) — Profiling hotspots that drove the optimization.

## Paper-faithful spike

- [`iw_pyramid_spike_methodology_2026-05-15.md`](iw_pyramid_spike_methodology_2026-05-15.md) — Wang & Li 2011 steerable-pyramid weight estimator design.
- [`iw_pyramid_ab_results_2026-05-15.md`](iw_pyramid_ab_results_2026-05-15.md) — A/B vs spatial-variance weights. Pearson 0.838 — weights ARE different.

## Wire format / bake binary

- [`reorder_lz4_zstd_eval_2026-05-13.md`](reorder_lz4_zstd_eval_2026-05-13.md) — HU L2-reorder + LZ4 ship-form 5.2× compression.
- [`reorder_rle_eval_2026-05-13.md`](reorder_rle_eval_2026-05-13.md) — RLE sweep (dead end).
- [`v0_17_quantization_review_2026-05-13.md`](v0_17_quantization_review_2026-05-13.md) — F32 / F16 / I8 quantization review.
- [`zenpredict_compression_eval_2026-05-13.md`](zenpredict_compression_eval_2026-05-13.md) — zenpredict-side compression eval.
- [`zenpredict_rle_zerobias_eval_2026-05-13.md`](zenpredict_rle_zerobias_eval_2026-05-13.md) — zerobias + RLE eval.

## Data integrity audits

- [`coefficient_blocklist_patch_2026-05-12.md`](coefficient_blocklist_patch_2026-05-12.md) — coefficient blocklist patch.
- [`holdout_overlap_audit_2026-05-11.md`](holdout_overlap_audit_2026-05-11.md) — Holdout overlap audit (initial).
- [`dhash_threshold_revert_2026-05-14.md`](dhash_threshold_revert_2026-05-14.md) — **dHash d≤16 cleanup REVERTED** (false positives; d≤10 + user-eye verification is the new standard).

## Sweep verification

- [`jsmlp_codec_sweep_verify_2026-05-12.md`](jsmlp_codec_sweep_verify_2026-05-12.md) — JS MLP codec sweep verification.

## Reading order suggestions

**If you want to ship a new bake**:
1. [`v0_18_methodology_2026-05-13.md`](v0_18_methodology_2026-05-13.md) — canonical reference recipe
2. [`v0_18_ship_reference_card_2026-05-14.md`](v0_18_ship_reference_card_2026-05-14.md) — gate numbers
3. [`v0_20_input_shaping_methodology_2026-05-15.md`](v0_20_input_shaping_methodology_2026-05-15.md) — `--auto-transforms` default
4. [CLAUDE.md "Principled experiment workflow"](../CLAUDE.md) — 10-step protocol

**If you want to understand a falsification**:
1. [`falsification_reeval_triage_2026-05-15.md`](falsification_reeval_triage_2026-05-15.md) — catalogue
2. [`falsification_reeval_results_2026-05-15.md`](falsification_reeval_results_2026-05-15.md) — verdict reproduction
3. [CLAUDE.md "SROCC-only verdicts BANNED"](../CLAUDE.md) — why these were SROCC-gated and what replaces SROCC

**If you want to optimize runtime**:
1. [`extended_iw_runtime_perf_optimized_2026-05-15.md`](extended_iw_runtime_perf_optimized_2026-05-15.md) — latest numbers + cost breakdown
2. [`iw_perf_hotspots_2026-05-15.md`](iw_perf_hotspots_2026-05-15.md) — where the time goes
3. CLAUDE.md "Performance Optimization" — archmage SIMD patterns

**If you want to investigate the IW path**:
1. [`v0_20a_path_a_falsification_2026-05-14.md`](v0_20a_path_a_falsification_2026-05-14.md) — original V_20a falsification (SROCC-gated)
2. [`v0_20_l0_norms_2026-05-15.md`](v0_20_l0_norms_2026-05-15.md) — GD-selection confirmation across 4 bakes
3. [`iw_pyramid_spike_methodology_2026-05-15.md`](iw_pyramid_spike_methodology_2026-05-15.md) — paper-faithful pyramid path (untested for ship)
4. [`iw_pyramid_ab_results_2026-05-15.md`](iw_pyramid_ab_results_2026-05-15.md) — pyramid vs spatial-variance A/B
