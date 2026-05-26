# SESSION-RESUME — read this first after every compact

**Last updated:** 2026-05-25T (evening — V39 ship + dial/spline findings)

## Current state

**Shipped champion:** `PreviewV0_3` = `zensim/weights/v39_v32plus_spline_seed17_2026-05-25.bin`
- Architecture: 372→128→64→heads (2-layer, per-sample α, tanh pin, PCHIP spline)
- Recipe: V32 ranking (normalized [0,1] group targets, hybrid MSE 0.6 +
  RankNet 0.6) + multi-band anchor at weight 0.01 for post-training
  spline calibration. seed=17.
- Beats prior V0_3 (v_tuner_v11) on **5 of 6** held-out corpora
  (full panel) AND the G1 dial:
  CID22 0.8793 / KADIK 0.9251 / TID 0.9317 / KonJND 0.4197 / AIC-3 0.8023;
  G1 dial 1.00 (prior 0.69). **AIC-4 (n=300) is the exception** —
  V0_3 wins SROCC 0.9284 vs V39 0.9051 (V39 wins AIC-4 DS-AUC). Not
  literally "universal"; accurate for the 5 compression holdouts.
- **Goal status:** PASSES G1 + G7; soft-passes G8/G9; **FAILS G5**
  (KonJND HF-rank 0.42 < 0.70 — the acknowledged learning-metric HF
  zone). Full CODEC_TARGET_GOALS.md achievement NOT done; needs the
  unstarted HF-corpus acquisition.
- Bake bytes carry both `tanh_output_head` + `output_calibration_spline`.

**⚠ The old "0.885 AIC-3 ceiling / CVVDP gap needs CSF features" claim is
FALSIFIED** (see Findings below). The AIC-3 "gap" was a measurement
artifact; per-ref we already beat raw CVVDP. The real lever is
cross-ref absolute-scale calibration (the dial), not features.

## Findings & falsifications (2026-05-25 evening) — READ BEFORE RE-TRYING

1. **Broken dial from SROCC-chasing (the big one).** Bakes trained
   without an output calibration spline collapse the dial to a
   near-constant (~65, G1=0.00) even at high SROCC — UNUSABLE as a
   codec target. The shipped V0_3 had a spline; V13–V36 didn't.
   Fix: the PCHIP spline. Because SROCC is **rank-invariant under a
   monotone spline**, a well-ranking compressed bake + a multi-band-
   anchor spline = good rank AND a working dial (that's V39).
   Detail: `benchmarks/v5_vs_v03_comparison_2026-05-25.md`.
2. **Anchor weight causes rank collapse.** V37 used
   `--anchor-loss-weight 0.5` with per-row anchor targets → the anchor
   MSE fought RankNet and held-out CID22 SROCC collapsed 0.88→0.55.
   **The anchor is for SPLINE FITTING ONLY — use weight ≤ 0.01.** Any
   meaningful anchor weight destroys rank.
3. **Target-scale mismatch → training divergence.** 5-group MSE-only
   training diverges when group `human_score` scales differ
   (cid22_train raw MCOS [3-94], konjnd [-66,96], others [0,1]).
   Normalize all targets to a common scale.
4. **dynamic-range-floor overshoots** (V40): pushed output >100 →
   clamps to saturated top. Not a clean G1 fix on its own.
5. **CVVDP-emulator is a dead end** (V41): training toward CVVDP
   scores gives WORSE human-MOS (CID22 0.66). Emulating CVVDP's
   output ≠ having its accuracy.
6. **AIC-3 "0.80 vs 0.96 CVVDP gap" was an artifact.** CVVDP's 0.96
   was a 5-image subset; on the full 600-pair/10-ref set CVVDP is
   0.79 pooled / 0.93 per-ref. Our bake scores 0.9475 per-ref —
   ABOVE raw CVVDP (0.9342). Mixing real CVVDP in as an input
   feature: +0.004 (noise). **NOT feature-limited.** Detail:
   `benchmarks/aic3_cvvdp_feature_spike_2026-05-25.md`.
7. **DATA BUG: kadid/tid iwssim + ssim2_gpu corrupt.** `iwssim` was a
   100% copy of `human_score` (target leak); `ssim2_gpu` was joined
   ref-vs-ref (~0 corr). Shipped bakes SAFE (used `human_score`);
   multi-target `iwssim` experiments on kadid/tid INVALID. Fixed
   siblings written (`*_fixed_2026-05-25.parquet`), pending promotion.
   Detail: `benchmarks/DATA_INTEGRITY_kadid_tid_metric_columns_2026-05-25.md`.

## Read order on resume

1. This doc (`SESSION-RESUME.md`) — current state, ~2 min
2. `CLAUDE.md` — methodology + workflow + gotchas
3. `docs/CODEC_TARGET_GOALS.md` — the goal set (G1-G11)
4. `benchmarks/v5_vs_v03_comparison_2026-05-25.md` — V39 lineage + findings
5. `RESEARCH.md` — corpus map + workflow recipes

## What shipped (29 commits, 2026-05-25)

### Speed
- 9× per-epoch training speedup (SIMD encoder + parallel validation)
- f32 SIMD encoder ready (`simd_encoder_f32.rs`, 1.36× over f64)

### Architecture
- 2-layer MLP (372→128→64→heads) + skip connection
- Full forward/backward with exact h1_pre caching
- 2-layer bake format (3 BakeLayer entries in ZNPR v3)

### Validation
- Mohammadi 2025 exact-methodology eval (`scripts/mohammadi_eval.py`)
- Multi-stat panel (SROCC+PLCC+PWRC), `--val-policy goals`
- NaN safety gate, per-sample Z-RMSE, output spline fitting
- DisplayProfile struct + iPhone 14 Tier 1 calibration

### Tests: 111+ across 3 crates

## What to do next (priority order)

1. **Cross-ref absolute-scale calibration** — the real lever for the
   AIC-3 pooled residual (NOT CSF features — that's falsified, see
   Finding 6). The dial is near-constant on near-imperceptible
   distortions; per-ref/absolute-scale anchoring is the open problem.
2. **Promote the kadid/tid data-bug fix** — review `*_fixed_2026-05-25.parquet`,
   `mv` over originals, rebuild `_MANIFEST.json`, re-sync R2, then re-run
   any multi-target `iwssim` bake (those used the leaked label).
3. **Smoother spline** — V39's spline is coarse (3 knots, raw p5=-89.7
   pre-clamp). More anchor bands surviving the strict-mono filter → a
   smoother dial. The dynamic-range-floor lever overshot (Finding 4);
   needs tuning, not abandonment.
4. **iPhone 14 / phone CVVDP bake** — BLOCKED on data: no phone/TV/PPD
   CVVDP columns exist anywhere on disk. When the backfill lands:
   `--target-column cvvdp_phone_log_norm` + V39's spline recipe.
   `DisplayTarget` enum + profiles already shipped.
5. **Wire cross-codec-eq aux loss for 2-layer** — currently "not yet
   wired" for multi-layer/skip mode (blocks the G4 cross-codec goal
   and the dynamic-range-floor's eq-pool substrate at 2-layer).

DONE this session (don't redo): σ-weighted MSE infra, modular refactor
(mlp_train → arch.rs/goals.rs/utils.rs), f32 encoder, DisplayTarget +
iPhone Tier-1 profile, auto-bake_verdict-after-train, DS-AUC + goals
scorecard in bake_verdict.

## Key files

| File | What |
|------|------|
| `zensim-train-core/src/simd_encoder.rs` | SIMD f64 encoder (production) |
| `zensim-train-core/src/simd_encoder_f32.rs` | SIMD f32 encoder (ready) |
| `zensim-train-core/src/per_sample_alpha_head.rs` | Head forward/backward + bake |
| `zensim-validate/src/mlp_train/` | trainer (mod.rs + arch.rs + goals.rs + utils.rs) |
| `zensim-validate/src/bin/bake_verdict.rs` | eval: Mohammadi panel + DS-AUC + goals scorecard; auto-runs after every train |
| `zensim-validate/src/panel.rs` | LightPanel + ValAggregate + stats |
| `zensim/src/display.rs` | DisplayProfile + DisplayTarget enum (desktop/phone/tv) |
| `zensim/src/profile.rs` | `mlp_bake_preview_v0_3` → V39 (shipped) |
| `docs/CODEC_TARGET_GOALS.md` | Goal set (G1-G11) |
