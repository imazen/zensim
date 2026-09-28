# steer-map — is the DVIFM field a better spatial-steering substrate than attribution?

## What was asked

Brief: `../briefs/lane_steer_prompt.md`. Three stages (S1/S2/S3): S1 makes
DVIFM's per-block field a first-class owned output (`block_field` +
`paint_block_field`); S2 compares it against the model's own attribution
gradient as a score-gain steering substrate via finite edits; S3 compares
both against two independent perceptual judges (ssim2, butteraugli) as a
severity field. Explicitly out of scope: any codec/RD loop, byte claims.

## Verdict (from `../reports/LANE_STEER_DONE.md` — copied verbatim)

Split verdict: DVIFM is **NOT** a better score-gain steering substrate — S2
shows attribution (the model's own gradient) wins decisively (SROCC
0.54-0.58 vs DVIFM's 0.10-0.21; DVIFM wins only 12-28% of pairs/size). But
DVIFM **IS** a better independent-perceptual-severity field — S3 shows DVIFM
much closer to the inter-judge agreement ceiling (DVIFM 0.327/0.338 vs
attribution 0.209/0.218 against ssim2/butteraugli; judges agree with each
other at 0.366; DVIFM wins 74%/68% of pairs) and it is local (pyramid leak
≤5% mean vs attribution's 57% mean / p90 79% — attribution is global-context).
Screen content is the weak cell for DVIFM vs judges (n=20, mixed).

Smallest actionable change if this is picked up again: retain the DVIFM
block cache in the steering session (the walk already runs) and consume
`query_eps_scale0` — no new extraction pass needed.

## Code in this directory

| file | role |
|---|---|
| `aggregate.py` | aggregates `pairs.jsonl` + `blocks.bin`/`rects.bin` sidecars into `aggregate.json`/`summary.json` — the S2/S3 comparison tables in the benchmark report come from this |

The S1/S2/S3 harness itself is Rust (`zensim-bench/examples/dvifm_steer_study.rs`,
see commit below) — this directory holds only the post-hoc Python aggregator.

## Rust / repo state

Committed directly to the **main checkout's own history** at `wtwzoykq` /
`7731c6209a11` — "dvifm-steer lane: S1 field owner + S2/S3 study + record".
Per the report this commit is a fileset-split of 8 files carrying this
lane's work, deliberately separated from other lanes' WIP that remained in
`@` at the time (`tools/joint_core/*`, `extract_features_372col.rs`).
Touches `zensim/src/dvifm.rs` (`block_terms` factored, `block_field` +
`paint_block_field`), `zensim/src/attribution.rs` (`query_rect_frac`,
`from_f64_canvas`), `zensim/src/feature_v2.rs`, `zensim/src/research.rs`
(`collect_dvifm_fields`, `DvifmFieldMap`/`DvifmLevelField`),
`zensim-bench/examples/dvifm_steer_study.rs`. Not pushed; fully committed
local history, not at risk.

## How to re-run

1. Build the S2+S3 harness: `zensim-bench/examples/dvifm_steer_study.rs`
   (requires the `training` feature; gated `required-features` in
   `zensim-bench/Cargo.toml`).
2. Run it to produce `pairs.jsonl` (482 rows), `blocks.bin`, `rects.bin`.
3. `python3 aggregate.py` reads those three files and emits
   `aggregate.json` + `summary.json` — the tables in
   `benchmarks/dvifm_steer_2026-09-20.md` come directly from this script's
   output.

Inputs consumed: 482 image pairs across photo/screen/doc/lineart/ai content
classes (SDR only — HDR excluded by design since the B bake and DVIFM's
default plane are SDR).

## Artifacts (reference by path)

- `/mnt/v/output/zensim/dvifm-steer-2026-09-20/` — `pairs.jsonl` (482 rows),
  `blocks.bin` + `rects.bin` (f32 sidecars, ~146 MB total), `summary.json`,
  `aggregate.json`
- Committed benchmark record (already permanent in the repo, not copied here):
  `benchmarks/dvifm_steer_2026-09-20.{md,json}`
