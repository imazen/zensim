# Steering-compatible feature subsets and product API

## Result — API implemented; models remain unqualified

The useful fast subset is **local120**: SSIM, edge-artifact and edge-detail
mean/L2/L4 plus MSE on XYB at four native scales. It omits hard maxima, L8,
whole-plane HF ratios, legacy masked/IW and v2/append families. Its 100-feature
variant keeps only Y at full resolution and all three channels at coarser
scales. These are explicit canonical feature IDs, not reordered feature files.
The runtime actually skips omitted reductions and finest X/B work. SSIM+MSE48
also skips edge reductions, but gives up more human accuracy.

A local feature set alone does not guarantee finite-edit steering. A nonlinear
head can still reorder gains when a whole codec block is repaired. The linear
local120 + monotone spline control passes **96/96** broader development cases:
eight admitted training origins, three JXL distortion levels, four block sizes,
25,248 real repairs. Uniform five-seed nonlinear local120/fine-Y ensembles pass
84/96 and 87/96, respectively. Their human errors are lower, but the remaining
spatial failures are real; no cases are hidden by an aggregate pass.

| Candidate | Human test raw MAE | Test SROCC | Broad spatial passes |
|---|---:|---:|---:|
| local120, uniform five MLPs | 7.497 | 0.9341 | 84/96 |
| fine-Y local100, uniform five MLPs | 7.560 | 0.9342 | 87/96 |
| local120, linear + spline, lambda .01 | 10.188 | 0.8955 | 96/96 |

The linear control still scores the retained salt/pepper image **76.28**, R/B
swap **55.44**, and 2x nearest-neighbor aliasing **37.73**. These scores are
unacceptable. Its strong block-repair coherence is not corruption detection
or perceptual approval. Uniform ensembles are likewise not qualified: finite
repair failures already block promotion. The nonnegative-distance architecture
was tested with five seeds and rejected: it worsens human/corruption error and
still fails spatial cases. No named model, default profile, or target dial changed.

### Initial three-seed subset screen

| Subset | Human test MAE median [min, max] | Spatial passes / 18 |
|---|---:|---:|
| basic228 | 7.332 [7.318, 7.426] | 11/18 |
| basic156 | 7.745 [7.465, 7.827] | 12/18 |
| fine_y130 | 7.701 [7.497, 8.009] | 13/18 |
| local120 | 7.804 [7.594, 8.002] | 15/18 |
| fine_y_local100 | 7.649 [7.635, 7.989] | 13/18 |
| means84 | 8.006 [7.964, 8.457] | 14/18 |
| fine_y_means70 | 8.203 [8.028, 8.288] | 12/18 |
| local_means48 | 8.442 [8.433, 8.479] | 14/18 |
| fine_y_local_means40 | 8.479 [8.294, 8.639] | 14/18 |
| ssim_mse48 | 8.369 [8.311, 8.887] | 14/18 |
| fine_y_ssim_mse40 | 8.639 [8.478, 8.974] | 14/18 |
| mse_hf48 | 9.180 [8.939, 9.615] | 16/18 |
| mse12 | 12.648 [12.556, 12.793] | 15/18 |

### Evidence and scope

There are **195 unique MLP fits**, not 222: 117 initial fits, 60 constrained
controls and 18 additional finalist seeds; the five-seed finalist directory
reuses 27 original fits. Nine deterministic lasso controls and two uniform
compositions add no stochastic refits. Rust owns extraction, inference, fitting,
calibration, panel statistics and spatial interventions. Python only schedules
those tools. The linear controls use the existing Rust lasso/spline owner.

Fresh native Rev3 extraction covers the same 19,958 admitted pairs as the
preceding scale study. Human fit/dev/test sizes are 7,000/1,000/3,125;
57/8/25 source references are disjoint, TID contributes only training rows.
Codec proxy and corruption have separate objectives and tables: their errors
must not be compared as if they share the human scale. Original image/decoder,
source admission, extraction binary, formula, recipe, row-order and model hashes
are retained in the artifact manifests. No new data source or split was admitted.
Historical codec PNGs are capability probes, not current native codec RD evidence.

The test partition is now development evidence used during feature selection;
it is not a fresh terminal validation. CID22 gold, AIC and other T0 panels remain
untouched. The initial spatial panel is six non-swap cases at block32 plus a
whole-image R/B diagnostic. The broader panel uses four block sizes 8/16/32/64;
all 96 cells per candidate must meet M2>=.99 and M3f>=.70. The linear minimums
are M2=.993007 and M3f=.755245. Swaps still need scalar detection even though
whole-image swap map correlation is diagnostic-only.

### Serving and extraction changes

`BakeScorer::prepare_steering` owns a reusable source-bound session and returns
the complete score with `refinement_gain`. It refuses unsupported families and
corruption companions; a complete map is never fabricated by dropping scored
terms. Active ensemble members are checked; zero-weight members do not require
work. `with_parallel(false)` lets an outer codec worker own scheduling, including
finite-difference gradient work. Revision 1/2/3 basic/peak bakes can coexist in
one process with no environment mutation. Wider feature families retain their
explicit revision mismatch refusal.

The existing fused SIMD kernels now skip unused peak/HF reductions; SSIM+MSE
plans also skip edge reductions. Cached map assembly skips inactive channels,
unneeded reference HF scans, masked/IW pools and the unused B forward pass.
Native basic/peak scalar extraction reuses the reference pyramid through the
existing cached fold owner; incompatible small/odd/sampling caches retain the
canonical fallback. There is still a scalar extraction followed by map assembly.

An independent derivative check found and fixed a real cached-map bug: the
revision-3 saturating HF-gain feature used the derivative of a revision-1 ratio.
The canonical derivative owner now supplies coefficients for the active model.
Tests check four forms against finite differences and preserve Rev1 bits. The
pre-fix and intermediate optimization evidence is retained; later final API
replays supersede it. No new SSIM or resize kernel owner was introduced.

### Confirmed runtime and memory

AMD Ryzen 9 9950X3D, release without target-cpu=native; CPU 8 for ST and
CPUs 8–15 for eight threads. Scalar includes both input pyramids; map includes
the complete score and fresh sensitivities with an amortized reference cache,
bin 8. These are actual compact models/compositions, not extraction-only loops.
Each final timing has 30–40 measured rounds. Values below are means in ms;
JSON retains sample count, extrema, variance and median. No p95 is invented
from a mean. The independent zenbench process-name gate is disabled because
of its known self-heartbeat false positive; runs had no overlapping builds,
training or other task benchmarks. The run-heavy logs retain load and memory.

| Candidate | ST scalar 1MP / 4MP | ST score+map 1MP / 4MP | MT8 scalar 1MP / 4MP | MT8 score+map 1MP / 4MP |
|---|---:|---:|---:|---:|
| local120 | 16.03 / 76.97 | 56.54 / 222.12 | 6.03 / 30.97 | 20.95 / 90.11 |
| fine_y_local100 | 10.09 / 46.85 | 32.86 / 137.17 | 4.62 / 21.49 | 15.56 / 65.13 |
| linear_local120 | 15.81 / 76.82 | 54.86 / 219.88 | 5.92 / 30.80 | 19.12 / 88.16 |

On the same ST input/control run, D takes 16.63/79.86ms and fast-ssim2
66.83/291.96ms. Fine-Y local100 is **6.62x faster than fast-ssim2 at 1MP**
and 1.65x faster than D. That is a measured layout/model/input result, not
a universal corpus or hardware claim. The packed fast ensemble reduces 1MP
map time from 53.24 to 32.86ms compared with its unoptimized wide representation.

The 1MP map/scalar mean ratios remain 3.53x(local120), 3.26x(fine-Y),
3.47x(linear). The registered 3x p95 gate is not established; do not qualify
the runtime by reporting only the fast scalar. Four-megapixel mean ratios
are about 2.86–2.93x; they do not waive the 1MP bar. Further map work must
address the remaining scalar-plus-map passes, preserving full-score semantics.

Whole-process peak RSS for the prepared worker, including inputs/cache/model/
scratch/map, is 103.69/360.91MiB(local120), 85.63/288.79MiB(fine-Y),
103.31/360.49MiB(linear) at 1MP/4MP after 30 comparisons. This conservatively
fits the incremental 192/576MiB limits, though codec process RSS is separate.

Matched original nonnegative bakes remain timing controls in `models.tsv`;
they are not selected quality models. Before/after scalar ST at 1MP is
26.52→15.62ms(local120),14.68→9.90ms(fine-Y),26.46→14.18ms(SSIM+MSE48).
That comparison includes the new single-worker policy as well as skipped
reductions; basic228 also speeds up 26.40→16.47ms. Early map/control groups
sometimes stopped at 25 samples and remain descriptive intermediate evidence.
The final dense ST runs were repeated with `ZEN_XP_MIN_ROUNDS=30`; MT8 final
runs already met 30 samples. Command manifests distinguish every binary.

Cached H128 training took a median 19.28s per initial fit (range 16.34–35.09s);
the entire 195-fit search is not a single five-minute experiment. Reuse its
hash-verified tables for bounded follow-ups rather than repeating extraction.

### Validation

474 library tests pass, 8 remain ignored. Clippy, API snapshots, 196 semver
checks and the final serving matrix pass. AArch64 and Wasm compile checks
pass with existing warnings; these are not runtime measurements on those ISAs.
All eleven packed bakes pass 43 native human/codec/corruption pixel comparisons
each: consumed features and pixel-versus-cached scores have zero difference.
The prepared-session doctest passes. Complete model quality qualification
remains blocked by the explicitly failed/missing gates above.

### Remaining release gates

No artifact in this study is a shipping model. Keep the spatially sound linear
control and accurate nonlinear ensembles as explicit controls for the next fit.
The next meaningful model change must improve local corruption/aliasing response
and human quality while constraining finite-edit head curvature; adding a
scalar-only corruption companion would violate this steering contract.

Before promotion: pass every retained corruption and repair case; freeze a
five-seed candidate/composition; then evaluate untouched human panels and all
native JXL/AVIF/JPEG/WebP integrations with active/neutral controls. Witness each
image's codec bounds before model steering, keep those bounds hidden from the
runtime, and use training-only codec seeds for the registered 1/2/3-shot error
bars. Require measured nonnegative RD savings under every independent judge
and >=1% under one. HDR spatial is outside this SDR session contract and needs
its own qualified lane. Missing terminal/target/RD/HDR checks are not passes.

### Reproduction and artifacts

Artifact root: `~/work/zensim-validation-2026-09-13/steerable/`.
`screen/`, `nonnegative/`, `finalists/`, `linear/` and `ensemble/` retain every
fit, command, raw score panel and model hash. `broad-dense/` is the final packed-artifact API
replay; `broad/` and `broad-final/` and explicitly named intermediate/refused directories remain
historical evidence. `FINAL_RESULTS.json` collects results and artifact hashes.
The three adjacent JSON recipes are immutable registrations for the existing
feature-screen owner; `reproduce/` retains executed follow-up orchestration.
Use the existing feature-screen stages (see recipe commands/manifests),
`bake_dial_refit`, `ensemble_score_rows` and `diffmap_block_coherence` rather
than rebuilding a second evaluator from the summary numbers.

```sh
../scripts/run-heavy --mem 16G --jobs 8 scripts/run_full_eval.sh \
  --stage feature-screen benchmarks/steerable_subset_2026-09-13.json \
  <fresh-output-directory> --ceiling-stage prepare
# Continue with --ceiling-stage fit, then audit, then report.
```

`dense/RESULT.json` binds each compact artifact to its original wide bake.
The existing densifier keeps model weights at their original precision and
runs a 512-row bit-identity gate. All 22,250 saved uniform-ensemble predictions
are unchanged at six-decimal output precision; all 288 broad spatial scores,
M2/M3f values and verdicts are identical. Each dense MLP is 58–70 KB, down from
about 499 KB. `BakeScorer::ensemble(&models, None)` serves their uniform mean;
the linear control uses `BakeScorer::new`. Both prepare steering identically.
The gradient reuses the extraction planner's structural read proof to skip
unreachable input probes. Exhaustive finite-difference tests include dense IDs,
zero-weight ensemble members and nonfinite unused inputs.

## Chronological registrations

Registered September 13 after the scale-selective944 study, before new fits.
User request: omit features resistant to spatial steering, optimize actual
extraction/map cost thoroughly, deliver official APIs and qualifying fast models.

Start from canonical Rev3 native planes. No prime/fractional default change;
the previous direct-scale study did not establish a quality/cost advantage.
Retain all canonical IDs. Initially omit v1 masked/IW (unsupported refinement),
v2/append (retained repair failures) and compare the known basic228 control
with twelve narrower mean/moment/local-error/HF subsets. Independently vary
finest chroma, peaks/high moments and whole-plane HF ratios. A pure MSE control
bounds the easiest local error signal; it is not presumed perceptually adequate.

Reuse the exact source-family admission of the 19,958-pair capability corpus
with fresh canonical extraction. Three paired seeds, H128,32 epochs and
every-epoch dev selection. Fit separate human/codec-proxy/corruption objectives;
their scales are not interchangeable. Keep seed dispersion and family tails.
Screen actual reference-block repairs through the complete Rust BakeScorer.
Use release M2>=0.99 / M3f>=0.70, supported refinement required; global RB swap
is diagnostic-only for map correlation. These differ from the prior scale
screen's stricter M3f/looser M2 and may not be compared by pass-count alone.

Select on scalar quality, every spatial failure and actual cost. Confirm
finalists with five seeds and broader block sizes/content before terminal
human data. Reuse retained source-disjoint native codec instruments for final
witnessed-range/1,2,3-shot/spatial RD checks. Do not infer codec savings from
rectangle repair correlation or allow model selection on terminal panels.
The existing production scorecard remains the shipping bar. Failed or missing
gates must prevent a qualification claim, not be renamed or silently dropped.

Optimization must skip passes/accumulators rather than only removing inputs.
Preserve consumed Rev3 values against unrestricted extraction, including SIMD
tails, threading, odd geometry, identity and relevant HDR inputs. Benchmark
same final bakes before/after through public scalar and cached-reference map
APIs, quiet pinned ST/MT, 1MP/4MP and >=30 samples for product cost evidence.
No target-cpu=native. Preserve original measurements and negative controls.

Public API registration (concrete callers: codec loops, existing spatial
instrument, doctest/integration example): a prepared steering session on
BakeScorer that binds source/reference cache/scratch and rejects incomplete
refinement contracts before use. Per-model arithmetic must be honored for the
selected feature regime without process-global environment mutation. Keep
existing APIs and explicit research overrides compatible. Freeze exact public
signatures after the subset/dependency audit; update changelog and snapshots,
exercise mixed-model revisions in one process and concurrent workers, and
check semver. Coverage means executable terms, not certified steering quality.
No new public model name or default change before a final artifact qualifies.

Follow-up registration: all 18 local120 M3f checks pass, while three M2
checks fail. Test the existing Rust trainer's nonnegative-distance head
against basic228/local120/fine_y_local100/ssim_mse48, five paired seeds
5101/5103/5107/5113/5119. Same frozen admitted tables, with manifest hashes
verified on reuse; no new selection data. This architecture guarantees its
identity anchor and upper bound, but is not assumed to fix finite-edit ranks.
It is distinct from the unconstrained screen and retains every failed gate.

Exact public delta: BakeScorer::with_parallel(self, bool) -> Self;
BakeScorer::prepare_steering<'s,S:ImageSource>(&'s mut self, &'s S, usize)
-> Result<SteeringSession<'s,'a,S>,ZensimError>; re-export SteeringSession;
SteeringSession::compute(&mut self, &impl ImageSource, Option<&str>)
-> Result<ScoredAttribution,ZensimError>. The existing spatial instrument and
extraction benchmark exercise these entries; no new evaluator owner.

Linear control registration (after the nonnegative-head results): nine
deterministic Rust lasso fits on the same human fit rows, local120 /
fine_y_local100 / ssim_mse48 crossed with lambda 0, 0.01, 0.1. Use the
existing bake_dial_refit gram/fit-lasso/append-meta owners, raw156 prefix,
2,000 CD sweeps and the same fit-only anchor for the existing monotone spline.
This checks whether a single projection removes finite-head curvature error;
no five identical seed reruns for a deterministic solver. Report all arms.

Finalist registration: extend the unconstrained local120/fine_y_local100/
ssim_mse48 arms to seeds 5113 and 5119, retaining all three original paired
seeds without refitting. Same 32-epoch recipe and same frozen tables.
A fresh final API audit covers all five seeds, including the corrected HF
derivative on controls; the pre-fix spatial evidence is preserved separately.

Broader registration: uniformly average all five unconstrained human seeds
for local120 and fine_y_local100 through BakeScorer::ensemble; compare the
linear local120/lambda .01 spatial control. Reuse the September 8
max-attribution PIXEL_REGISTER's 24 pairs / eight FIT origins, three JXL
distances, all blocks at 8/16/32/64. Verify original file hashes and current
train admission. This is a development robustness gate, not terminal quality
or native codec rate/distortion qualification. The existing
ensemble_score_rows CLI accepts repeated --bake to evaluate the same uniform
public composition on the already admitted scalar tables.

Packing follow-up registration: profiling the actual ensembles exposed gradient
probes over 944 declared input positions despite 100/120 structural reads. Reuse
the canonical extraction read proof to skip unreachable probes; check against
exhaustive central differences and preserve nonfinite-input validation. Also
run the existing Rust `bake_dial_refit densify` owner on all eleven finalist
member/control artifacts, without retraining or requantization. Require its
512-row identity gate plus admitted real-row score parity and the complete
96-cell-per-composition spatial replay before distributing dense artifacts.
Preserve the original wide bakes and timings.
