# Reachable targets and codec steering — September 8, 2026

User ruling: establish each image's codec range before judging steering; fit
codec-specific seeds/heuristics on imazen/imazen-26 **training** content; compare
1, 2 and 3 shots. JXL is the first diffmap integration, with JPEG and WebP also
in scope, followed by deeper AVIF work. The user still controls one score,
including negative scores. Do not rescale that dial per image or codec.

This supersedes the July 18 G-TARGET screen as a qualification procedure. The
September 7 midpoint/bisection measurements remain reproducible observations,
but their mixed feasible/infeasible median does not establish a model failure.
The old median-error <=2 bar was a research screen, not a perceptually justified
product tolerance. Report error distributions and hit rates at explicit bands;
no new universal perceptual error bar is established here.

## Preregistered instrument change

Owner: `zensim-target`, extending its existing search and `demo_matrix`.
Concrete caller: the imazen-26 calibration/steering experiment. Unpublished API
delta, registered before implementation: `TargetSpec` gains an optional
`SeedEstimate { knob, score_per_knob }`. The existing search uses the estimate
for shot 1, its training slope for shot 2, and a bracket-safeguarded measured
secant for later shots. An absent seed preserves midpoint behavior. Estimates
must be finite, in the codec range and have the correct slope sign. No new
metric inference API or second controller is introduced.

`demo_matrix` gains an explicit bound-sweep/calibration/evaluation mode. Fit
and evaluation execute in Rust; candidate scores always call `BakeScorer`,
named scores call `Zensim`. Calibration is a codec/model-specific curve table,
not a change to the zensim score scale. Freeze it before evaluating other
source families. A calibration artifact includes scorer, codec configuration,
binary, source/split and probe identities; mismatches must fail.

## Bounds before steering

1. Pin original bytes, rendered RGB interpretation, dimensions, encoder and
   decoder commits, codec configuration (including chroma, effort, resampling,
   filters, lossless mode), metric bytes/formula revision and map mode.
2. Encode both ends and an interior ladder over the **declared configuration's**
   full supported knob range. Score the actual reconstructions. Save encoded
   and decoded hashes, bytes, scores and measurement cost. Deduplicate
   saturation by encoded hash, not assumed q intervals. Inspect reversals and
   discontinuities. Lossless and alternative chroma/resampling are separate
   configurations, not unreported extensions of the same range.
3. The minimum/maximum observed scores are **attained bounds on the sampled
   family**, not a proof of extrema over every possible codec mode or every
   floating-point knob. Endpoint-only monotonicity is not assumed. Targets
   with a measured reconstruction in their tolerance band are witnessed
   feasible. Other fixed requests are outside the measured envelope or
   unwitnessed inside it; neither category is silently called impossible.
4. Evaluate the steering model on witnessed targets, including targets drawn
   from the ladder itself to exercise each image's usable range and gaps.
   Keep the original score coordinates. Report excluded/unwitnessed requests
   and coverage beside errors. A sparse grid cannot qualify range coverage.
   Separate G-ADDR still prevents a collapsed model dial from earning an easy
   targeting pass.
5. Bounds are an evaluation oracle only. The timed controller gets source
   pixels, target, frozen training calibration and its own prior probes. It
   must never receive evaluation q/score pairs, endpoint scores or an
   image-specific seed chosen from the bound sweep. Report oracle cost
   separately; do not conceal it inside a claimed one-shot production method.

## Calibration and measurements

- Canonical corpus moved from `imazen/codec-corpus` to `imazen/imazen-26` on
  August 23. Its August 27 `split_map_family.tsv` adds shared-content families;
  the old local README/digit-only tables alone are insufficient. Use both
  canonical origin membership and family separation; exclude disagreements
  for the initial experiment. Pin the latest manifest bytes. Read measured
  render URLs from the August 30 index; do not derive them by extension swap.
- First baseline: median training score at each native knob, followed by an
  explicitly recorded monotone running envelope; invert that curve per codec
  and scorer, with its local slope. Compare the same images/targets with midpoint search.
  This bounded baseline is not a replacement for the existing codec-specific
  content predictors. Fit future predictor/heuristic choices on train only;
  use source-family-disjoint validation for development. Reserve terminal
  test for a frozen, adequately covered candidate.
- Run budgets 1, 2 and 3 separately, preserving early stopping and actual work.
  Report median, p90/p95, worst and signed error, undershoot, hit rates at
  explicit tolerances, bytes, total latency, score/map latency and memory.
  Group by codec, content family, size and target band; uncertainty is by
  source family, never by treating repeated requests as independent images.
- An outer shot is a full encode/decode/score. Native JXL reconstructs inside
  one encode: report full encodes, internal reconstructions, scalar compares,
  map evaluations, terminal decode/verification and time **separately**.
  Do not rename K updates plus an initial compare as K complete shots.
- Diffmap trials must prove the candidate map was consumed and changed the
  intended quantization/RDO state. Compare active and neutral maps at the
  same target/budget, including an engagement trace. Use independent SSIM2
  and Butteraugli judges at matched quality/bytes; scalar accuracy alone is
  not evidence of better allocation. Re-score the emitted bitstream, not
  only the encoder's internal reconstruction.
- No default model or codec setting changes on a smoke result. Preserve failed
  experiments. Expand source/size coverage before a product qualification.

## Branch and implementation audit

Later September 8 correction: the [fleet and attribution reuse audit](../benchmarks/diffmap_reuse_audit_2026-09-08.md)
extends this initial inventory. JXL already has fused/binned/stale attribution
and H3 steering. JPEG's `zenjpeg--zensim-diffmap-rd` is a `jj` workspace at
`46a6ff30`, absent from a Git-only worktree listing and not merged into main.
AVIF's later `examples/zensim_cq_rd.rs` already implements folded-944/H3,
split-role maps and a hint-engagement probe; its August 7 report contains
August 29 negative results and must be read through its final sections.
The missing candidate surface must extend these owners, not recreate them.

Remote refs were refreshed September 8 without changing sibling checkouts.
The exact inventory and local dirty-state records are retained in the private
target-audit directory. Main branches were newer than several local HEADs.

| Codec | Existing owner and mechanism | Follow-through |
|---|---|---|
| JXL | `jxl-encoder/src/vardct/zensim_loop.rs`, `examples/zensim_diffmap_rd.rs`: global controller plus map-driven quant-field redistribution, map/model experiments and seed head | First native integration. August 27 independent-judge results and August 30 secant study precede September 7 reconstruction corrections. Rejected `issue103-{coordinate-control,full-distance-feedback,targeting-research,upsample-phase-order}` branches retain negative research. Do not repeat or merge them blindly. |
| JPEG | `zenjpeg/src/target_quality.rs`: float-native outer secant; `encode/zq.rs` behind `target-zq`: global correction plus per-block AQ feedback; `zq_seed.rs` and bucket anchors | Include real diffmap engagement; reconcile existing profile/seed identity and pass accounting before candidate comparisons. `feat/quality-zq-target`, `feat/zq-bucket-calibration`, `feat/zq-linear-f32`, `explore/perceptual-loops` preserve earlier work. |
| WebP | `src/encoder/zensim_target.rs` behind `target-zensim`: outer q correction plus per-segment map correction near the band | Map path is conditional, not guaranteed by enabling the feature. One-pass metrics can be NaN/optimistic, so independently measure final bytes. `recovered/zensim-target-extended-eval-a8516` and `spike/zenpicker-knobs` are historical experiments. |
| AVIF | `src/target_quality.rs`, `two_pass_zensim.rs` and `sb_pool.rs`; `cooptloop` branch connects rav1e research | Inventory now; deeper map integration after JXL, with JPEG/WebP included. Preserve July 12 negative two-pass results and the dirty rav1e trace file. |

Existing native map integrations often construct `ZensimProfile::Custom` or
older named profiles. `BakeScorer` currently exposes complete scalar serving,
not a complete composed-model map surface. Do not describe old per-bake
gradient mounting as support for every multi-head/ensemble/corruption model.
That serving gap must be closed and parity-tested before those candidates can
qualify in native map loops. Keep this distinct from the scalar baseline.

First execution and remaining native integration details:
[September 8 measurement record](../benchmarks/target_steering_bounds_2026-09-08.md).
The existing alternate JXL `zensim_backend.rs` bridge also clamps
`100 - score` to 0..100; the dedicated `zensim_loop.rs` uses another loss
conversion. Audit the actual selected route before claiming negative-target
support. The generic scalar controller measured here never clamps scores.
