# v_next status snapshot — 2026-05-07

Live status as I work toward a real V0_4 bake on the unified corpus.

## Merged today

| PR | Title | Notes |
|---|---|---|
| #28 | feat(v_next): unified parquet pipeline + score-quality analyzer + CID22 paper notes | 2.37M-row training data + analysis + docs |
| #29 | feat(zensim): V0_4 MLP runtime via zenpredict (selective merge of #24) | Ships V0_4 dispatch path with V0_2-equivalent placeholder bake |
| #30 | feat(v_next): trainer + bake + cross-codec sweep launcher | Standardization fix, ZNPR baker, vast.ai launcher |
| #24 | (closed, superseded by #28+#29+#30) | Trainer pipeline lives in zenanalyze/zentrain going forward |
| #27 | (already merged earlier) hardening | Brought `ZensimError` to `#[non_exhaustive]` |

`zensim` bumps to **0.3.0** on next release (queued in CHANGELOG).

## In flight

### Local training (RTX 5070)

`scripts/v_next/train_v_next_mlp.py` — 228 → 64 LeakyReLU → 1 MLP,
predicting `score_ssim2`, `mse + 0.5·ranknet` loss, source-disjoint
80/10/10 split, AdamW lr=3e-3, batch 16384, 50 epochs.

```
epoch   0  val_srocc=0.8239  val_krocc=0.6459
epoch   1  val_srocc=0.8644  val_krocc=0.6911
epoch   2  val_srocc=0.8892  val_krocc=0.7280
...
```

Trajectory healthy; expect plateau around `val_srocc ≈ 0.95`. Output:
`/mnt/v/zen/zensim-training/2026-05-07/runs/<ts>_v_next_ssim2_64h_full/`.

### Cross-codec sweeps on vast.ai

| Run | Codec | Boxes | Cells planned | Files at last check |
|---|---|---:|---:|---:|
| `sweep-v16w-2026-05-07` | zenwebp | 10 | 329,616 | 340 / 1962 |
| `sweep-v16a-2026-05-07` | zenavif | 5 | 41,202 | 0 / ~1962 (booting) |
| `sweep-v16j-2026-05-07` | zenjxl | 10 | 109,872 | 0 / ~1962 (booting) |

Total burn: ~25 boxes × ~$0.05/hr ≈ **$1.25/hr**. Source corpus
mirrored to `s3://zentrain/sweep-v16{w,a,j}-2026-05-07/sources/` from
the v15r 1024 px Lanczos3 corpus. Worker docker image
`ghcr.io/imazen/zen-metrics-sweep:0.6.3` with binary
`zen-metrics-0.6.8-linux-x86_64-gpu` (53× faster than 0.6.5 per
the v_next handoff).

## Score mapping question (for V0_4 bake follow-up)

The trainer outputs `pred ≈ score_ssim2 ∈ [0, 100]`, where 100 means
identical. zensim's API contract is the same:
`Zensim::compute(...).score()` returns 0..100, 100 = identical.

Currently `zensim::metric::apply_mlp_scoring` does:
```rust
let raw = predictor.predict(features)[0] as f64;
let score = distance_to_score_mapped(raw, params.score_mapping_a,
                                      params.score_mapping_b);
```
where `distance_to_score_mapped(d, a, b) = clamp(100 - a * d^b, 0, 100)`.

That mapping was designed for V0_2's "raw distance" semantics: features
weighted to a positive distance, larger-distance = worse-quality, then
mapped to 0..100. With V0_4's MLP trained directly on ssim2-scale
targets, `raw` is already in the score scale — the mapping
double-transforms it.

Three viable post-bake paths:

1. **Train target = `100 - ssim2`** (a "distance"). Then with `a=1,
   b=1`, `score = 100 - 1·d^1 = 100 - (100-ssim2) = ssim2`. ✓ Easiest;
   only needs trainer flag flip on the next run.

2. **Keep target = `ssim2`, post-process the bake**. Reset
   `score_mapping_a/b` for the V0_4 profile to identity-equivalent
   values via output_specs in ZNPR v3 (linear scale + offset on the
   model output before the score mapping).

3. **Branch in `apply_mlp_scoring`**. If a metadata flag in the bake
   says `score_is_direct = true`, skip `distance_to_score_mapped`
   entirely and clamp the prediction to [0, 100]. Tighter coupling
   but cleanest semantically.

Going with **option 1** for the first real bake — single-line trainer
change, no zensim runtime changes. Re-baking on `100 - ssim2` keeps
SROCC identical (monotone transform) but produces the correct absolute
score scale at runtime.

## Pending / blocked

- **TODO §4.4 follow-up** — Adversarial pairs file already exists
  (`adversarial_pairs_top_disagree.parquet`). Once V0_4 bake lands,
  re-run the analyzer to see if the new model resolves disagreements
  or maps them differently.

- **Multi-task supervision** (paper §6 recommendation: ssim2 primary
  + dssim/ba_p3 for q-band tails). Skip for first V0_4 bake; queue
  for V0_5.

- **Multi-scale invariance probe** (TODO §4.3). Paper uses 6 scales,
  zensim uses 4. Defer until cross-codec data settles.

- **zenpredict release prep** — Blocked by uncommitted work in
  `~/work/zen/zenanalyze/` (modified `zenpredict/src/feature_transform.rs`
  + 2 added benchmark files in `benchmarks/` totaling 3140+ line
  additions, predates this session).

- **zenmetrics knob-grid expansion** — Blocked by uncommitted work in
  `~/work/zen/zenmetrics/` (modified GPU pipeline files in
  `crates/{butteraugli-gpu,dssim-gpu,ssim2-gpu}/src/` + a v13
  Dockerfile).

Both repos' uncommitted changes look substantive; I'm not touching
them per the never-destroy-uncommitted-work rule. The cross-codec
sweep launched from the OLD (last-pushed) docker image successfully
on vast.ai, so the in-flight changes don't block this work — just
the release-prep + zenmetrics-edit work.
