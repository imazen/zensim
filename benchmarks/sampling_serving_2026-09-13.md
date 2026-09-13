# Sampling filters through Rust serving and spatial steering

Registered September 13 before implementation/results. Concrete callers are
`BakeScorer` scalar/cache/spatial methods, the existing extractor audit,
`extract_paths_bench`, the bounded feature-screen stage and
`diffmap_block_coherence`. No new scorer, filter kernel or statistic owner.

Wire a closed versioned `zentrain.sampling` bake metadata contract:
`v1:{y|xyb}:{triangle|mitchell|robidouxsharp}:{3/2|2|3}`.
Y mode means full-resolution Y at level 0 (no finest X/B feature work),
then XYB at d,2d,4d. XYB mode means all three channels at d,2d,4d,8d.
The named filter is used at every transition; dimensions floor at each
transition. Small inputs are reflected before sampling to keep all four
levels valid. No standard bake changes; unknown/mixed sampling contracts
and mismatched cached references must fail. New features are re-extracted,
labelled and refit; old weights are never silently retagged.

Reuse canonical XYB conversion, the folded feature kernels and the existing
retained spatial integrands. Sampling initially materializes float pyramids;
benchmark that full allocation/conversion/resampling cost honestly against
the streaming box baseline. Keep signed float values, including cubic lobes.
HDR must either use its declared PU front end or refuse explicitly.

Spatial mass is pulled back through the actual resizer's coordinate/weight
tables using normalized squared tap weights (a positive, mass-preserving
error-ownership approximation, not an exact pixel derivative). Finite-max
ownership uses every nonzero tap's source support, composed across levels and
reflection. Neither rule proves finite-edit accuracy: test actual pixel
replacements through the same complete candidate API and retain failed cells.

Fit the fixed T2-only 264-pair packet with explicit SSIM2 proxy labels and the
existing reference-level roles. Use paired seeds 4004/4005/4006 when comparing
families. Record final pixel/cache feature and score parity, identities,
negative tails, per-reference rank and full panels. Benchmark1/4MP full
scalar serving, cached spatial maps and memory with paired quiet timing.
Check spatial interventions at 8/16/32 pixels on existing training examples
and phase-sensitive impulses/grids, including color-only errors. No protected
holdout or release qualification is implied. Record a negative result if
sampling cost or lost signal defeats the proposed speed/quality tradeoff.

## Serving contract and scope

Implemented after the registration above. Each recipe carries an explicit
producer identity, populated ID set, formula revision and sampling tag.
The trainer propagates the admitted sampling metadata into the final bake;
the final extractor audit calls `BakeScorer` on actual pixels. Untagged models
keep the existing pyramid. This is experimental opt-in behavior, not a new
named/default model or a release promotion.

All 18 sampled contracts are tested on odd and reflected small images for
scalar/cache/spatial feature, score and raw-distance parity. Tests also cover
identity, finite maps, signed-mass conservation, incompatible ensembles,
legacy-cache refusal and the SDR-only boundary. A legacy Zensim spatial
entry cannot consume a cache bound to a sampled BakeScorer. HDR and an
unmatched corruption companion are explicitly refused. A raw cached feature
slice has no runtime provenance; the existing table admission and public
pixel audit bind that part of the experiment.

Pyramids are initially materialized. Independent source/distorted pyramids
and float channels now use the existing Rayon pool for large inputs;
small inputs and single-thread execution retain sequential resizer reuse.
All 18 contracts have bit-exact serial/parallel pyramid checks. Reference
geometry is cached once; spatial projection presently uses a full logical
canvas before binning. Timing includes these costs. This implementation is
suitable for measuring the hypothesis; it is not a streaming-resizer speed
claim. The previous filter-only benchmark cannot substitute for these costs.

## Fixed spatial packet

Before inspecting spatial results, pinned seed 4004 for all 20 layouts and
8/16/32-pixel replacements on seven cases. Four use existing JXL decodes at
distance nearest 1.0 (actual 1.093362): photo 2010, document 6610, screen 8462
and graphic 9066. Three reuse canonical-corruption training PNGs on photo 2010:
16 salt/pepper impulses, a whole-image R/B swap, and the catalog's 2×
nearest-neighbor down/up aliasing artifact. No new Python image generator.
The aliasing case is a periodic resampling artifact, not an exhaustive codec
grid or phase sweep. File hashes and selections are in `SPATIAL_CASES.json`.

The existing Rust `diffmap_block_coherence` owns correlations. M2 compares
feature-space linearization against actual score changes; M3a uses additive
density; M3f uses `refinement_gain`, including finite-max correction. Each
actual block edit is scored through the complete candidate's public API.
Retain cells below the existing M2≥0.99 / M3f≥0.70 diagnostic bars, including
negative gains and constant/uninformative cells. These tests do not measure
native encoder rate/distortion benefit or reachable-target steering error.

## Geometry interpretation

For ideal rational spacing d=p/q and a codec block period B, the sampling
phase repeats after `Bq/gcd(p,Bq)` output samples. At B=8, d=2 repeats every
4 samples (one block); d=3 every 8 samples (three blocks); d=3/2 every 16
samples (three blocks). Thus 3× and1.5× redistribute alignment over successive
blocks; neither eliminates periodic alignment. Actual finite-image spacing
is input/output dimension ratio after flooring, so odd dimensions must also
be tested rather than assigned the ideal phase formula blindly.

An isolated impulse of amplitude a becomes a·h after a linear filter: peak
sensitivity scales with `max|h|`, energy with `sum(h²)` (products of the two
axis factors in 2D). Wider smoothing can erase small corruption cues even
when broad codec quality ranks well. Keeping finest Y preserves that route
for luma impulses, while color-only fine errors still depend on coarser XYB.
Cubic negative lobes can preserve or amplify some edge contrast; they do not
prove better corruption detection. The fixed intervention results below test
part of this tradeoff, and cannot certify unseen phases or corruption types.

## Integrated measurements

AMD Ryzen 9 9950X3D; pinned core 0 (ST) or 0–7 (MT8), 25 interleaved samples
per layout/geometry, zenbench median/MAD and paired intervals retained in JSON.
The host had no other sustained heavy compute. Automated host-load gating was
disabled for the opt-in sampling instrument because the local zenbench gate
counts its own runner; its lock/interleaving/statistics remain active. Some
groups report temporal drift, so comparisons use interleaved samples, not
independent wall-clock runs. Builds are excluded from the training budget.

All 60 H32 models fit and pass final BakeScorer pixel/cache audits in
**256.56 seconds total**; each three-seed layout takes 10.8–14.6 seconds.
Across them: 15,840 pixel/cache comparisons, 720 exact identity checks,
and no silently clipped negative scores. The smallest served score is
-25.911; distorted scores above 100: 21.

The table uses signed inner-test SROCC averaged over three paired seeds.
It covers only two previously examined T2 origins and SSIM2 proxy targets.
M3f passes are out of 21 cells per layout; every layout has failed cells.

| Layout | ST 1MP ms | MT8 1MP ms | MT8 spatial 1MP ms | Test SROCC | M3f≥.70 |
|---|---:|---:|---:|---:|---:|
| box_full228 | 26.54 | 5.57 | 44.81 | 0.9325 | 17/21 |
| box_y190 | 14.05 | 3.98 | 30.32 | 0.9853 | 16/21 |
| xyb_mitchell_2 | 37.90 | 10.29 | 36.44 | 0.8410 | 9/21 |
| xyb_mitchell_3 | 24.31 | 6.64 | 23.76 | 0.9627 | 1/21 |
| xyb_mitchell_3d2 | 55.04 | 12.83 | 52.37 | 0.9579 | 9/21 |
| xyb_robidouxsharp_2 | 37.84 | 12.81 | 36.97 | 0.8860 | 7/21 |
| xyb_robidouxsharp_3 | 24.31 | 6.65 | 23.98 | 0.9357 | 4/21 |
| xyb_robidouxsharp_3d2 | 55.04 | 12.74 | 51.37 | 0.9702 | 8/21 |
| xyb_triangle_2 | 25.31 | 7.36 | 29.54 | 0.8727 | 8/21 |
| xyb_triangle_3 | 15.50 | 4.99 | 18.10 | 0.9712 | 2/21 |
| xyb_triangle_3d2 | 39.16 | 9.61 | 42.45 | 0.9387 | 9/21 |
| y_mitchell_2 | 44.87 | 11.95 | 58.08 | 0.9755 | 10/21 |
| y_mitchell_3 | 32.09 | 8.88 | 44.85 | 0.9717 | 8/21 |
| y_mitchell_3d2 | 62.32 | 14.76 | 72.13 | 0.9811 | 14/21 |
| y_robidouxsharp_2 | 44.90 | 11.67 | 58.03 | 0.9782 | 13/21 |
| y_robidouxsharp_3 | 32.04 | 8.74 | 44.94 | 0.9812 | 8/21 |
| y_robidouxsharp_3d2 | 62.31 | 14.78 | 72.77 | 0.9784 | 16/21 |
| y_triangle_2 | 32.48 | 10.53 | 51.22 | 0.9738 | 15/21 |
| y_triangle_3 | 23.35 | 6.97 | 39.11 | 0.9805 | 5/21 |
| y_triangle_3d2 | 46.62 | 12.04 | 64.07 | 0.9812 | 16/21 |

The box/full-Y control is the cheapest scalar arm on both 1MP and 4MP
in this implementation. Parallelizing independent resizes materially improves MT8 sampling cost: for
example XYB/3× Triangle falls from 12.32 to 4.99 ms. Sampled cubic
pyramids still cost more than the Y190 control. Filter-only measurements
cannot stand in for integrated conversion, materialization and feature cost.
The compact [JSON](data/sampling_serving_2026-09-13.json) includes 4MP timing,
dispersion, seed ranges, raw proxy MAE and all 420 spatial cells.

Spatial mapping completed 144,960 actual replacements in 39 seconds.
All candidate refinement feature sets were supported, but coverage does
not imply accuracy: downsampling all XYB by 3× passes only 1–4 of 21 M3f
cells, versus 17/21 for box/full228 and 16/21 for box/Y190. Even the controls
have serious failures; neither is spatially qualified. The squared-tap
ownership approximation must earn its accuracy on finite edits.

Keep full-resolution luma in the next experiment. Prefer measuring coarse-only
expensive feature families before optimizing alternate cubic pyramids. No
new filter or trained model is promoted to a default by these results.

## Validation, artifacts and chronology

The earlier [filter-only report](fullres_y_subset_2026-09-12.md) correctly
said alternate pyramids were not yet served; this follow-up supersedes that
status. It also preserves the first integrated sequential-resize measurements
under `before-parallel-resize/`; those are diagnostic history, not the final
MT comparison. The raw-distance consistency correction precedes final timing.
Training and intervention scores do not change under the later parallelism:
pyramid bit parity is checked, and the fixed sub-65,536-pixel packet stays on
the same sequential path. All bake hashes and original instrument hashes are
retained; do not infer build identity from a dirty checkout's parent git hash.

Artifacts live at `~/work/zensim-validation-2026-09-13/sampling/`: recipes,
60 final bakes, extraction caches, final public-API audits, Rust panels,
source copies/hashes, immutable measurement binaries, timing JSON/logs,
420 intervention JSON/log pairs and the orchestration scripts. The directory
is under the existing work share. Large artifacts remain outside git; the
compact report data is committed. Exact commands and environment are in
`run_timing.py`, `run_matrix.py` and `run_spatial.py` alongside manifests.

The default-revision library/integration suite, sampling-specific Rev3 API
checks, serial/parallel tests, CI-exact clippy, script lint, no-default-feature
build and API-doc check pass. The first attempt to run the entire legacy
suite with global Rev3 was invalid for shipped-revision golden tests; that
log is preserved. Two new registry failures from that run were fixed by
explicitly recording and checking the producer's finest-channel selection.
No protected evaluation or publishing/release command was run.

The user's follow-up about useful coarse legacy extras is answered in the
[separate audit](coarse_legacy_features_2026-09-13.md). It motivates a bounded
coarse-family experiment, not deletion or relocation of existing baked inputs.

Final timing runs record process-group peak RSS of 0.23 GiB (ST), 0.50 GiB
(MT8 scalar) and 0.19 GiB (MT8 spatial). These are benchmark process peaks,
not per-layout memory attribution. No allocation-free claim is made.

A rebuilt final extractor re-audited all 60 bakes against all 264 pairs in
12.73 seconds: 15,840 comparisons, unchanged scores and consumed features.
The later legacy-profile guard rejects nonidentity sampled bakes installed
through Custom profiles; complete candidates use BakeScorer. This guard
does not alter sampling extraction or model arithmetic.

## Served A/B inspection gallery — September 13 follow-up

The existing gauntlet renderer now builds a static spatial gallery from this
packet, without rerunning inference or recomputing correlations:

```sh
python3 scripts/v_next/gauntlet.py \
  --spatial-gallery "$HOME/work/zensim-validation-2026-09-13/sampling" \
  --out "$HOME/work/zensim-validation-2026-09-13/spatial-gallery/index.html"
```

Use a fresh output path; existing HTML is never overwritten. Serve the whole
output directory, including `images/`, `data/` and `manifest.json`, and open
`index.html`. Source PNG hashes must match `SPATIAL_CASES.json`; the renderer
requires a successful matrix, unique cells, supported refinement and complete
block partitions. Original result hashes remain in the downloadable index.

The default view shows **235 of 420 cells** failing M2 ≥ 0.99 or M3f ≥ 0.70;
185 passing controls remain selectable. Filters cover model, case and block
size. A/B views, a wipe, exact pixel zoom, signed actual/predicted/error maps,
scatter points and block rows share a selected rectangle. The repair preview
copies that reference rectangle onto B; its displayed score comes from the
recorded Rust evaluation. This is inspection of development failures, not
new qualification evidence. Whole-image R/B is the only permutation in this
packet; additional channel operations need separately recorded checks.

Browser checks cover failure/control counts, channel filtering, A/B wipe,
block selection, exact repair pixels inside/outside the rectangle, true 4×
zoom, permalink reload and a 390-pixel viewport without page overflow. No
page errors or failed HTTP responses occurred. Generator negative controls
reject failed matrices, duplicate cells and incorrect PNG hashes before
creating output, and preserve existing HTML. Script lint and CI-exact clippy
pass; inference is unchanged.
