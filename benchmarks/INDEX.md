# `benchmarks/` — INDEX

> **⚠ BANNER 2026-07-18:** this index STOPS at 2026-05-15 — its 'ship a new bake' reading order is obsolete (points at v0_18 methodology). For 2026-06/07 work (five-gate scorecard, RD probe, diffmap coherence, additive-vs-MLP correction, HDR steer screen) start from `docs/TOP_MODELS_COOKBOOK.md` or `ls benchmarks/*2026-0[67]*.md`.

> **★ RETROSPECTIVE 2026-07-26:** [`best_per_day_summer_2026.md`](best_per_day_summer_2026.md) — the **best model per calendar day** (2026-05-01 → 07-25), with verified bake paths, recipes, headline metrics, and the summer champions (best CID22 = winner_dial 0.894; best KonJND = cl_tfm 0.761; best HF-NL/dial = Ebothg_scr0.5_dial; shipped B/A/BHdr). Machine-readable twin: `/mnt/v/output/zensim/reports/best_per_day.json`. The one-stop map of every model-experiment day this summer.

Methodology docs, falsification logs, sweep outputs, perf
benchmarks, and bake binaries. 76 markdown files as of 2026-05-16,
organized by theme + chronology. Each entry is one line:
file path + one-line summary.

If you arrived here cold, start with [RESEARCH.md](../RESEARCH.md)
first — this index is a deep dive for follow-up.

> **★★ PROFILE D — THE GATING TAX, AND A W4 EXCEPTION THAT WASN'T 2026-09-01/02:** [`profile_d_notax_2026-09-01.md`](profile_d_notax_2026-09-01.md) — removes the **gating tax** (`feature-regime-v2` is now a default feature; `cargo semver-checks` clean, zero public API delta, new parity gate) and the one real **tier duplication** in the touched kernels (raw moments: 10 hand-duplicated per-tier sites → 2 generic helpers + 1 scalar pair; `dense_block_kernel` is ERA-LOCKED and untouched). No feature slots minted. **The measurement is the interesting half.** The first 162-cell both-tier sweep was CORRUPT and the doc says so in §4.3: a `zenbench` `compare` group given too little `max_wall_time` for its size **degenerates to a spuriously near-zero mean for every arm at once** — caught by cross-checking `fast_ssim2`, an arm whose cost cannot depend on zensim's thread count, reading 1.58 ms against 690–724 ms in adjacent rows. Three 2304² cells had **0-of-9, 1-of-9 and 5-of-9** valid starts. Independently, this lane's own `nice`-d concurrent builds contaminated the same tier — `nice` lowers priority, it does not isolate from a `taskset`-pinned sweep — swinging `fast_ssim2` **128.9–633.6 ms inside one cell**. **⇒ `min()` over process starts is only safe against noise that is one-directional; a harness that can report LOW defeats it, and `min()` will select the corrupted reading as "the best one".** Now a standing warning in `CLAUDE.md`. **§4.4 is the clean re-measurement**: per-size wall budget (8/15/**60**), a collection-time plausibility filter, an idle box, n=9 valid starts × 18 cells × both tiers, **0 corrupt reads and 0 retries in 54 invocations** — and **every cell passes the 1.25× W4 bar at 1T and 8T**, worst ratio 1.189×. **The a4bkon lane's recorded 1.44–1.46× FAIL at 1152²/8T does NOT reproduce** (1.143× `v4x`, 1.026× `v3`), and the contamination cuts both ways: this lane's own spurious 2.037× at 1152²/1T resolves to 1.079×. Recorded as a dated **APPENDIX C ADDENDUM** in the ssim2-bar doc — that verdict is NOT edited in place, and "nothing passes W1–W7" still stands (W1/KonJND remains the binding failure).

> **★★ PROFILE D SHIPS + THE PUBLISHED-CRATE SPEED CHECK 2026-09-01:** [`profile_d_and_published_speed_2026-09-01.md`](profile_d_and_published_speed_2026-09-01.md) — **closes the W7 gap the ssim2 bar and HYBRID docs both flagged** ("today's best 156-class candidate needs `custom-profiles`... no profile slot, no `from_block_profile`"): `ZensimProfile::D` now ships (behind default-on `candidate-profiles`) carrying ADD156 arm-A (spline-top-extended, rank-exact), and `ComputeSet::from_block_profile` (internal) derives its compute set automatically. Measures BOTH the default-build and `feature-regime-v2` forms rather than assuming either, and separately answers the user's "did we regress vs the crates.io-published profile A, I remember a 10" — a same-process, ASLR-controlled bench of the published `PreviewV0_2`/`A` against current main, `B`, `D`, and `fast-ssim2` as an opponent row, across 576²/1152²/2304² × 1/8/16T.

> **★★ THE ssim2 BAR 2026-08-31:** [`ssim2_replacement_bar_2026-08-31.md`](ssim2_replacement_bar_2026-08-31.md) — **the first head-to-head against SSIMULACRA2 in the project's history**, as ONE registered exam (W1–W7) with every threshold derived from a measured noise floor. Answers the user's "have we made real progress" per mission axis. Headlines: **SPEED WON** — zensim is faster at every size and thread count (576²/1T: fast-ssim2 21.7 ms vs the 944 walk 18.3 / the 372 fold 9.4 / the basic fold 7.4; 6.5× at 8T), and nobody had ever measured it. **RANK a TIE** — reference-clustered paired bootstrap on CID22 (49 refs, B=10,000): both 944 flagships tie ssim2 pooled *and* within-image, **shipped B is measurably WORSE within-image (−0.0079, CI excludes 0)**, ADD156 worse on both; CSIQ is the one strict win (+0.047). **DIAL split** — `W10L9P_s4005` beats ssim2 (mono 0.9947 vs 0.9930, 6 % vs 14 % of near-lossless ladders inverting) while B and ADD156 are the only arms that END ladders backwards. **HDR** — shipped `BHdr` beats ssim2's integrated-PU path 0.7536 vs 0.7044; the frozen HDR candidate-of-record loses to both. Plus the gates-and-goals audit (every gate measures us against ourselves; `peer_ssim2` scores 0.8979 on our own `balanced_composite`, above every model, by arithmetic) and the 10-row premise ledger incl. the traced 944 cost story. New instruments: `bake_verdict --dial-peer-scores`, `panel --per-group`, the `fast_ssim2` bench arm.

> **★★ THE HYBRID + the SPEED CLASS BAR 2026-09-01:** [`hybrid_candidate_2026-09-01.md`](hybrid_candidate_2026-09-01.md) — the ssim2 exam's **W4 amended twice** (both user directives): APPENDIX B adds the **8-thread** count, then **AMENDMENT B2** replaces the opponent bar with a **CLASS bar — ≤ 1.25× the 156-walk class at BOTH 1 and 8 T**, `fast-ssim2` kept as context. The 1.25 is derived: below ~1.10× is inside the bar arm's own ASLR spread, and the measured classes are 372 → 1.55–1.85×, 944 → 2.06–2.68×, so it cuts in the gap. **MEASURED 1 T** (per-start ratio, 5 ASLR starts, 6 arms interleaved in ONE process): bar **6.90 / 25.70 / 107.20 ms**, `zensim_B` 1.55–1.85×, `flagship_944off` 2.06–2.16×, `q7b_944pools` 2.42–2.58×, `hybrid_944pools` 2.57–2.68×, `fast_ssim2` 2.78–3.40×. **No 944 model can pass** — the gap is the extraction — so both flagships, `Q7b` and every 944 hybrid become **teachers/upper bounds**. The ensemble's SECOND forward costs **1–6 % of one compare**. **PART I:** a convex blend `w·W10L9PH + (1−w)·Q7b` **PASSES W1, which NEITHER parent does**, over a measured window **w ∈ [0.76, 0.86]** bounded below by LIVE and above by KonJND; **KonJND is SUPER-ADDITIVE** (parents 0.5006 / 0.5118, blend **0.5390**, ssim2 0.5272); the blend cuts Q7b's deepest q≥85 backwards step **91.3 → 30.4** and removes the flagship's tied rate. **W2 still fails, structurally** — the named near-lossless win survives to w ≈ 0.4–0.6 and W1 starts at 0.76, so the feasible regions are **disjoint**. **PART II — distillation into the 156 set:** the control did not reproduce `ADD156`, and every explanation was tested — clip +0.007, λ +0.06 and plateaus, solver a trade, **fit substrate NOTHING**, and **the TRAINING LEG +0.057, which reproduces the incumbent exactly** (196k canonical safesyn → CID22 0.8643 / KonJND 0.5406 at 31 coefficients). Over nine λ the teacher target beats the human one on CID22 (8/9, median +0.017) and CSIQ (+0.022) and **costs KonJND monotonically (−0.008 → −0.115)**. **The leg is ~7× the teacher and free of the KonJND cost** — the priced ask for the era-2/radius-4 wave. `SADD_BIGLEG` (156-class, 4,117 B) **ties `ADD156` and gains no clause** — its one clear edge is **KonJND 0.5432, above `peer_ssim2`'s 0.5272**; on the regime-matched grid both fail W3 (a first draft's W3-gain claim was a cross-grid comparison and is corrected in the doc). New/fixed instruments: `bake_verdict --ensemble-weights`, `bake_dial_refit predict --ensemble-weights` **plus a pruned-bake width bug** (it was prefixing a 944-row to 667 columns), `gram --max-feat`, `fit-lasso --anchor-prefix`, the four amended-W4 bench arms, and **LIVE + AIC-4 added to the paired bootstrap** (the exclusion was ROOT-scoped: 372 root max |Δ| 1.12, both 944 roots 0.0).

> **★ HF HUMAN DATA 2026-09-01:** [`hfhuman_2026-09-01.md`](hfhuman_2026-09-01.md) — **the JPEG-AIC study family, ingested as eval axes.** AIC-3 BTC/IPTC, AIC-4/PTC and JPEG-AI-SDR25 are **ONE study on TEN source images**, and three board axes already read it (`aic3`, `aic4`, and `sdr25` = `bake_verdict`'s selection comparator). **Registered split rule `jpeg-aic-family-holdout-2026-09-01`: HOLDOUT-ONLY family-wide, membership by CONTENT — no training leg exists or should be built** (training on any member contaminates all three axes at once and buys ≤900 rows on 10 images; the alternative partition is priced and rejected). What it DOES give is the exam's missing instrument: a non-circular human near-lossless axis at **n = 515,250 forced choices** (vs `hfnl_cid22band`'s 1,425). **VERDICT** (native reading): ssim2 acc 0.7302 against a majority-oracle **ceiling of 0.7346**; Q7b/W10L9P/ADD156/W10L9PH all **TIE**, shipped **B −0.0025 [−0.0037, −0.0014] BEHIND** — the CID22 verdict again, on 88× the judgments. **AND the corpus supplies its own robustness test**: BTC stimuli are *boosted* (2× magnified, ~1.8× amplified — measured), so `btc_displayed` and `btc_native` serve the IDENTICAL responses and differ only in the pixels — **14 of 36 verdicts FLIP**. What survives: **nobody strictly beats ssim2 under both readings** (zero WIN→WIN cells); **shipped B is BEHIND under both**; **Q7b never loses** on the pooled or cross-codec axis under either; **the 944 pair is the most reading-sensitive arm** (cross-codec −0.0020 → −0.0276). The one strict win — W10L9PH *and* W10L9P on the JPEG-AI leg, +0.0012 [+0.0002,+0.0025], two seeds — **reverses to a significant loss on the other reading**, and is reported as arm-dependent rather than claimed. Also: **`aic3`'s target is a DESIGN value** (`−0.25 × level` exactly on 600/600 rows, 121 interpolated) and **`sdr25` is a 50-row sub-slice of `aic4` with a different reconstruction**. Overlap audit clean (min d=12 over 1,221 refs, eye-verified). New statistic `zensim_validate::pairwise` + `panel --pairwise`. Artifacts: [`hfhuman_2026-09-01.pointer.md`](hfhuman_2026-09-01.pointer.md).

> **★ FAILURE PROFILES 2026-08-31:** [`failure_profiles_2026-08-31.md`](failure_profiles_2026-08-31.md) — **where each model HURTS**, not how it ranks. The statistic→production-situation mapping; the board-wide ladder-inversion measurement (`dial.zones`: inversions by codec × quality zone × content class, plus the worst ladders by reference image name, 322 of 379 cells); and the board's new *Failure profile* panel. Headlines: `q>=85` is the failing band (**189 of 322 models** carry a ladder that ends backwards there); avif is the worst codec and webp nearly clean; shipped **B** ranks **20.8 %** of `hf_nearlossless` references backwards where ADD156 ranks 0 %, and two board cells invert ~86 % of references while publishing positive pooled SROCCs. Artifacts: [`failure_profiles_2026-08-31.pointer.md`](failure_profiles_2026-08-31.pointer.md).

## Headline reference cards

These are the docs to read first for a given purpose:

| Purpose | Doc |
|---|---|
| **Canonical codec-target metric** (2026-05-24) | [`tuner_v10_cross_codec_baseline_2026-05-24.md`](tuner_v10_cross_codec_baseline_2026-05-24.md) + [`../docs/CODEC_TARGET_METRIC.md`](../docs/CODEC_TARGET_METRIC.md) |
| **Tuner v11 in-flight retrain** (task #6) | [`v_tuner_v11_methodology_2026-05-24.md`](v_tuner_v11_methodology_2026-05-24.md) |
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

## Sweep budgeting (cost before you launch)

- [`hdr_sweep_budget_2026-08-05.md`](hdr_sweep_budget_2026-08-05.md) + [`.tsv`](hdr_sweep_budget_2026-08-05.tsv) —
  multi-codec **HDR** sweep budget: measured `alpha + beta*pixels` per (codec, preset, quality)
  for zenav1-svt (p6/p10/p13 + HdrFork), zenjxl (e7/e1), jpeg+gainmap (ultrahdr), and zenavif
  (speed 4/10) on real 10-bit PQ sources across a 64x64 -> 7.08 MP ladder, plus per-pair GPU/CPU
  metric cost. **Headline: encode is minutes, scoring is hours** — the 76-source corpus encodes in
  6.9 CPU-h (~3-5 min of distributed wall) but takes 11.5-16.1 GPU-h to score, so sweep wall-clock
  is the metric queue and the QUALITY presets are effectively free. Weekend N = 283 (all-GPU
  metrics) to ~800 (cvvdp-on-CPU + ssim2 only).

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

**If you want to make EXTRACTION faster** (read this BEFORE re-profiling any kernel):
1. [`v2_block_cost_2026-08-31.md`](v2_block_cost_2026-08-31.md) — **what the v2-348 + append-204
   block actually costs, at the tier that ships.** `dense`'s `POOL_SIMD` path is `v4x`-only and
   valgrind masks AVX-512 out of CPUID, so callgrind cannot profile production; this is the wall-clock
   re-profile the predecessor asked for, via seven new `fold_timing` phases. **Every v2 feature kernel
   is flat in ns/px across a 16× pixel range; the plane passes degrade 1.8–3.7×**, so the block is
   64 % plane traffic at 2304² and 32 % at 576² — its composition inverts with size. `dense` is
   **13.5 % of the block on `v4x`**, not the 22–26 % the `v3` Ir profile shows. Production : consumption
   of the six shared planes = **1.84 : 1**. Includes four falsified levers (rayon band split,
   `STRIP_ROWS`, a bit-identical row-major V blur, bounds checks) and a measured era-2 retarget.
2. [`feature_cost_frontier_2026-08-31.md`](feature_cost_frontier_2026-08-31.md) — which FAMILIES a model
   class needs, and what each class's walk costs (the model-class Pareto front).
3. [`extraction_perf_and_buffered_removal_2026-08-30.md`](extraction_perf_and_buffered_removal_2026-08-30.md)
   — the callgrind family split and the levers rejected with reasons. Its Ir shares are the `v3` tier;
   read 1 above before quoting any of them as a wall share.
4. [`fold_footprint_2026-08-31.md`](fold_footprint_2026-08-31.md) — the working-set closed form and the
   per-thread L3 budget. [`fold_mt_scaling_2026-08-31.md`](fold_mt_scaling_2026-08-31.md) — thread scaling.
5. [`era2_perf_break_2026-08-31.md`](era2_perf_break_2026-08-31.md) — the byte-moving break. Its dense
   framing predates 1; the plane pipeline is the bigger prize.

**If you want to make TRAINING faster** (read this before re-profiling the trainer):
1. [`trainer_perf_2026-08-04.md`](trainer_perf_2026-08-04.md) — fresh profile: Adam 44 % (divider-bound,
   irreducible at fixed math), the trainer is RAM-bound not FLOP-bound, half of every lane's 11.88 GB was
   dead after standardization (fixed — see 2 below), `--minibatch-size 32` = 3.63× but changes the bake
   (gated + measured).
   Also: lianli is a same-ISA-tier drop-in node, tower is AVX2-only; zenfleet training not worth it
   below ~60–80 runs/wave.
2. [`trainer_mem_release_2026-08-04.md`](trainer_mem_release_2026-08-04.md) — the dead copy, REMOVED:
   each group is one flat buffer that standardization takes and transforms in place, so a lane holds
   one copy of the features, not two. Full-data (11-group, 779k-row) bit-identity gate. Includes the
   measured failure of the obvious fix — freeing per-row `Vec`s moved full-recipe peak RSS only
   ~0.4 GB, because scattered ~7.5 KB chunks freed out of interleaved glibc arenas never return to
   the OS. Gate memory claims on the real process, not an allocator probe.
3. [`trainer_perf_2026-08-04.tsv`](trainer_perf_2026-08-04.tsv) — raw counters, K-scan, gate output

**If you want to investigate the IW path**:
1. [`v0_20a_path_a_falsification_2026-05-14.md`](v0_20a_path_a_falsification_2026-05-14.md) — original V_20a falsification (SROCC-gated)
2. [`v0_20_l0_norms_2026-05-15.md`](v0_20_l0_norms_2026-05-15.md) — GD-selection confirmation across 4 bakes
3. [`iw_pyramid_spike_methodology_2026-05-15.md`](iw_pyramid_spike_methodology_2026-05-15.md) — paper-faithful pyramid path (untested for ship)
4. [`iw_pyramid_ab_results_2026-05-15.md`](iw_pyramid_ab_results_2026-05-15.md) — pyramid vs spatial-variance A/B

**Era-2 break — rank preservation (the flip gate)**:
[`era2_rank_preservation_2026-08-31.md`](era2_rank_preservation_2026-08-31.md) — the registered §21.1 bar executed across the roster on old-vs-new features: 7 arms × 6 models × 9 corpora. Tiling at the production width passes 5 of 6 and is bit-identical on 8 of 9 corpora *by construction* (the corpora are narrower than the tile); radius 4 passes 2 of 6 and dominates the combined verdict; the era-2 dense kernel is not wired and so is not measurable. Also closes bar clause 3 (the dial gates) for BOTH tiling and radius 4.

**Era-2 break — the user's decision surface**:
[`era2_drop_redefine_table_2026-08-31.md`](era2_drop_redefine_table_2026-08-31.md) — items E (drops) + F (redefinitions) as one table: what each buys, its per-model rank cost, and what it takes to ship. Backed by [`era2_perf_break_2026-08-31.md`](era2_perf_break_2026-08-31.md) and [`blur_radius_locality_branches_2026-08-31.md`](blur_radius_locality_branches_2026-08-31.md).
[`era2_blast_radius_2026-08-31.md`](era2_blast_radius_2026-08-31.md) — the era-2 flip's third prerequisite: what re-pins, what re-extracts, what retrains, and the registered (unlaunched) wave.
[`era2_fast_profile_subset_2026-08-31.md`](era2_fast_profile_subset_2026-08-31.md) — "what subset should a high-speed model be limited to?": the measured cost curve for five compute sets at 1/8/16T, what each gives up, and the recommended pair (`156` for a fast profile, `944peaks` free for 944 MLPs).

**Board currency (2026-08-31)**:
[`board_regen_2026-08-31.md`](board_regen_2026-08-31.md) — the KonJND JPEG-504 ruler reaches the gauntlet: the 17 diluted-ruler cells re-scored through the owners (all 378 board cells now read n=504), the ordering changes it causes (the KonJND leader was a ruler artefact; composite top-10 and the `--select` winner unchanged), badge verification end-to-end, curation of the W-LIN 7b candidate, and three pre-existing board-integrity defects found in passing (276 stale composites, peer rows leading the default sort on an unnormalised scale, one cell that is not valid strict JSON).

**Board fairness (2026-09-04)**:
[`fair_gauntlet_2026-09-04.md`](fair_gauntlet_2026-09-04.md) — the FAIR gauntlet: eight corrections applied (best-of-k seed groups, the `predict --ensemble` units defect, four era/circularity items that were registered but badged nothing, the free-40 precision skew, G-ADDR, the composite's own 37% ssim2-circularity, train==val), a seven-criterion fairness filter with per-row tiers (42 VERIFIED-FAIR / 55 FAIR-NOTED / 336 LEGACY), what moves at the top once best-of-k becomes a k-mean, and three defects found in passing (a registry field-scope collision, a Python/Rust mirror drift, and a render-gate whose last third never gated). Audit TSV: [`fairness_tiers_2026-09-04.pointer.md`](fairness_tiers_2026-09-04.pointer.md). Exam transcription: `ssim2_exam_scorecard_2026-08-31.json`.

**Dial addressability — the D lineage (2026-09-04)**:
[`d_id100_2026-09-04.md`](d_id100_2026-09-04.md) — `D-id100`: the registered-not-run §14.6(ii) build, executed. The shipped Profile D chain reproduced byte-exactly end to end (closing its manifest's stated provenance gap); the identity pin measured to be a **spline** property, not a fit one (eight real re-fits at 0.1–20 % identity data mass move the identity dial +0.0055 and cost −0.0125 CID22 — the identity Gram's `S`/`s`/`q` are exactly zero); 21 identity anchor rows give **CONTRACT 6/6 + REGRESSION 4/9** with byte-identical weights and zero pair-order flips, and the anchor's own unclamped `ssim2_gpu` column takes it to **7/9**. Includes a fixed owner defect (`fit_spline_knots`'s neg-tail dedup deleted genuinely-negative knots) and measured impossibility proofs for the two axes that remain. Artifacts: [`d_id100_bakes_2026-09-04.pointer.md`](d_id100_bakes_2026-09-04.pointer.md).

**Push clobber + the push guard (2026-09-05)**:
[`push_clobber_2026-09-05.md`](push_clobber_2026-09-05.md) — `origin/main` moved SIDEWAYS twice on 2026-09-04 (jj ops `db7c8ca86b69`, `0edf97e28a91`), dropping nine commits from six lanes with no error and no warning; the per-added-line audit separating the seven that were re-landed or superseded from the one that was genuinely lost (`d3a948ca`, the G-ADDR board coverage — 482 of 498 added lines absent, `cut_gaddr_negtail_probe.py` absent entirely, and the boards on `/mnt/v` already generated with it), the re-land (`2e5cdc8b`, diff byte-for-byte identical to the original), and the owner guard `scripts/safe_push.sh` (fetch → ancestor-assert → set → push → verify; refuses sideways with rc=3 and names every commit it would drop; 4-case self-test incl. the negative control, plus a retrospective control that replays the real clobber).
