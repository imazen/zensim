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

## v16 cross-codec sweep — FAILED, all workers destroyed

The 25-box vast.ai cross-codec sweep (v16w / v16a / v16j) produced
empty score columns on every cell. Workers reported "[done]" rows;
TSVs have the right header + chunk-key fields but `encoded_bytes /
encode_ms / decode_ms / score_*` are all blank.

The same `zen-metrics-0.6.8-linux-x86_64-gpu` binary running locally
on the same source-image directory produces working zenwebp output
("2/2 cells emitted; encode-fail=0"), so the binary itself is fine.
The failure is environment-side: most likely the
`ghcr.io/imazen/zen-metrics-sweep:0.6.3` docker image is missing
something the binary dlopens (libwebp / libaom / libjxl runtime), or
something in the worker's onstart pipeline got truncated.

Spent: \$0.64 of \$31.74 vast.ai credit. No retry attempted —
correct fix is to:

1. Spin up a SINGLE vast.ai box, drop into `docker exec` on the
   image, run the binary against one source image manually, look at
   stderr.
2. If it's a runtime lib gap, rebuild the docker image with the
   missing libs.
3. Run a 1-chunk smoke before scaling.

The chunk JSONLs (`/mnt/v/zen/zensim-training/2026-05-07/v16-chunks/`),
the chunks-uploaded R2 prefixes
(`s3://coefficient/jobs/sweep-v16{w,a,j}-2026-05-07/chunks.jsonl`),
and the source-mirrored prefixes
(`s3://zentrain/sweep-v16{w,a,j}-2026-05-07/sources/`) all stay in
place for a future retry.

## In flight

### Local training (RTX 5070) — DONE

`scripts/v_next/train_v_next_mlp.py` — 228 → 64 LeakyReLU → 1 MLP,
predicting `score_ssim2`, `mse + 0.5·ranknet` loss, source-disjoint
80/10/10 split, AdamW lr=3e-3, batch 16384, 50 epochs.

Final: best epoch 44, **val_srocc=0.9547**, **test_srocc=0.9814**.
Output:
`/mnt/v/zen/zensim-training/2026-05-07/runs/20260507T115414_v_next_ssim2_64h_full/`.

**Contamination audit (post-train):**
- Source-disjoint split verified by reproducing the seed=0 split
  bit-exactly: 783 train / 98 val / 98 test images, all mutually
  disjoint.
- 0 CID22-named files in the corpus (none of the validation refs
  could leak through).
- **Codec coverage caveat**: trainer default is `--sweeps v15r,v15rc`
  which is zenjpeg only (2.30M of the 2.37M rows). The v12_{zenavif,
  zenjxl, zenwebp} + v14_zenpng parquets are present and
  schema-compatible (300 features each) but were not loaded. The
  baked V0_4 is therefore a zenjpeg-tuned model. Cross-codec
  generalization remains unverified until the v16 sweep is rerun
  successfully.

### Cross-codec sweeps on vast.ai

See "v16 cross-codec sweep — FAILED" above. All workers destroyed.

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
