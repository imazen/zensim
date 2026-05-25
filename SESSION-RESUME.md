# SESSION-RESUME — read this first after every compact

**Last updated:** 2026-05-25T14:00Z (marathon session)

## Current state

**Champion bake:** `prod_2layer_v3_anchor_2026-05-25.bin` (258 KB)
- Architecture: 372→128→64→heads (2-layer, per-sample α, tanh pin)
- Training: mix_cv40_iw60 target, 5 groups, anchor, LR=1e-3, L2=1e-4
- AIC-3 held-out SROCC: **0.885** (Mohammadi 2025 methodology)
- Located: `/mnt/v/output/zensim/bakes/prod_2layer_v3_anchor_2026-05-25.bin`
- Also: `benchmarks/prod_2layer_v3_anchor_2026-05-25.bin`

**Architecture ceiling confirmed at 0.885 AIC-3 SROCC** across 10
experiments. Gap to BUTTERAUGLI (0.893): 0.008. Gap to CVVDP (0.960):
0.075. The CVVDP gap requires CSF-aware feature engineering.

## Read order on resume

1. This doc (`SESSION-RESUME.md`) — current state, ~2 min
2. `CLAUDE.md` — methodology + workflow + gotchas
3. `docs/CODEC_TARGET_GOALS.md` — the goal set (G1-G11)
4. `benchmarks/marathon_results_2026-05-25.tsv` — experiment results
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

1. **σ-weighted MSE loss** — per-row σ for Z-RMSE optimization
2. **Modular refactor** — mlp_train.rs 10k lines → library modules
3. **Display-aware features (Tier 2)** — PPD as 373rd input
4. **f32 training pipeline** — wire f32 encoder into training loop
5. **Multi-bake ensemble** — combine v3 + codec-only bake strengths

## Key files

| File | What |
|------|------|
| `zensim-train-core/src/simd_encoder.rs` | SIMD f64 encoder (production) |
| `zensim-train-core/src/simd_encoder_f32.rs` | SIMD f32 encoder (ready) |
| `zensim-train-core/src/per_sample_alpha_head.rs` | Head forward/backward + bake |
| `zensim-validate/src/mlp_train.rs` | 10k-line trainer (needs refactor) |
| `zensim-validate/src/panel.rs` | LightPanel + ValAggregate + stats |
| `zensim/src/display.rs` | DisplayProfile struct |
| `scripts/mohammadi_eval.py` | Held-out AIC-3 evaluation |
| `scripts/arch_eval_matrix.sh` | Architecture comparison |
| `docs/CODEC_TARGET_GOALS.md` | Goal set (G1-G11) |
