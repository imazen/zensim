# Runtime profile experiments after the corruption clarification

## Results: several executable options, none qualified yet

The later [Claude-era gate review](#claude-era-gate-review-after-the-runtime-study)
below records the evidence still needed before extending or selecting these
profiles. In particular, these results are not full gauntlet evaluations.

The useful low-cost layout keeps local Y features at 1×, 1/2, 1/4 and 1/8,
with X/B features only at 1/8: **60 features**. Its five-H128 ensemble roughly
matches the prior large local120 ensemble on the observed human test panel,
while costing about half as much. The three-seed luma-only ensemble saves
only another 0.11 ms at 1MP but loses over one point of human MAE. Keeping
some coarse chroma is a much better observed tradeoff than deleting it all.

A mixture of the five coarse60 MLPs (weight 1/7 each) and its linear+spline
control (weight 2/7) passes **96/96** broad repair cells at raw human MAE
**7.898**, versus **10.188** for the previous consistent linear120 control.
It uses six distinct model forwards; duplicate linear members from the initial
mixture experiment were merged into the existing weighted Rust ensemble API.
All 33,375 saved mixture predictions agree at six-decimal TSV precision after
that change, and all final spatial cells were replayed with the served weights.

**The artifact responses are still unacceptable.** The passing mixture scores
salt/pepper **64.75**, R/B swap **32.95**, and 2× nearest-neighbor aliasing **47.95**
on the retained cases. Their previous user-reviewed problems are not solved by
passing block-repair coherence. These human-trained candidates have no attached
corruption catcher. Do not install them as shipping profiles or reinterpret
these scores as successful catastrophic detection.

### Quality and complete runtime

Raw human test errors use 3,125 KADID rows on 25 reference images, already used
as development evidence in the earlier study. They are not fresh terminal
validation, and small MAE/rank differences do not establish significance.
Selection used development error and repair results. The mixture weights were
selected on this same repair panel, so their pass needs independent confirmation.

All timings below are **means**, in milliseconds, at 1024² / 2048² pixels on
an AMD Ryzen 9 9950X3D. ST is one pinned core; MT8 uses eight pinned cores.
Each timing retains 30–35 samples, extrema, median and variance. No
`target-cpu=native`; cached map timing includes the score and fresh sensitivities.

| Candidate | Human MAE | SROCC | Repair passes / 96 | ST scalar 1MP / 4MP | ST score+map 1MP / 4MP | MT8 scalar / map at 1MP |
|---|---:|---:|---:|---:|---:|---:|
| Y-only40, three H128 | 8.534 | .9149 | 94 | 7.98 / 37.75 | 26.60 / 107.39 | 4.02 / 12.71 |
| Coarse60, five H128 | 7.487 | .9340 | 95 | 8.08 / 38.16 | 27.21 / 108.11 | 4.07 / 13.22 |
| Mid80, five H32 | 7.465 | .9327 | 90 | 8.43 / 39.69 | 28.12 / 112.41 | 4.13 / 13.30 |
| Fine100, five H32 | 7.494 | .9327 | 90 | 9.85 / 46.54 | 32.53 / 136.68 | 4.40 / 14.42 |
| Full120, five H32 | 7.526 | .9324 | 91 | 15.60 / 77.28 | 56.28 / 221.45 | 5.95 / 19.85 |
| Prior fine100, five H128 | 7.560 | .9342 | 87 | 9.85 / 46.66 | 33.01 / 136.32 | 4.41 / 14.84 |
| Prior full120, five H128 | 7.497 | .9341 | 84 | 15.71 / 77.67 | 56.62 / 221.70 | 5.93 / 20.55 |
| Linear80 + spline | 10.599 | .8948 | 96 | 8.37 / 39.72 | 27.59 / 111.62 | 3.98 / 12.81 |
| Prior linear120 + spline | 10.188 | .8955 | 96 | 15.48 / 77.44 | 55.39 / 221.26 | 5.77 / 18.90 |
| **Coarse60 weighted mixture** | **7.898** | **0.9313** | **96** | **8.14 / 38.14** | **27.23 / 108.81** | **4.11 / 13.24** |

Consult the adjacent [structured results](runtime_profiles_2026-09-13.results.json)
for exact values, all 13 compositions, unrounded ranks, failed cells, manifests,
MT8 4MP values and timing dispersion. The two other mixture fractions were
retained: 1/6 linear passes 93/96; 1/2 passes 94/96. More linear weight does
not monotonically improve finite-edit ordering. Linear60 passes 95/96.

On the same final scalar benchmark, D takes 16.29/80.12 ms and fast-ssim2
66.46/290.45 ms. Coarse60 is about 2.0× faster than D and 8.2× faster than
fast-ssim2 at 1MP in this fixture. Direct before/after runs on the *same*
coarse60 bytes measure 9.98→8.08 ms at 1MP and 46.05→38.16 ms at 4MP:
roughly 19%/17% less time from actually skipping coarse channel work.
The change keeps the shared XYB pyramid; conversion and pyramid cost remain.
Smaller heads reduce file sizes but barely change scalar latency at a fixed
layout. Scalar feature extraction is the more consequential cost here.

Coarse60's whole-process prepared-worker RSS is 80.55/266.48 MiB at 1MP/4MP;
the weighted mixture is 80.50/266.25 MiB, versus 85.61/289.11 MiB for prior
fine100. This includes inputs, cache, model, scratch and map, but not a codec.
The mixture's 1MP mean map/scalar ratio is still **3.35×**. The registered
p95 ratio is not established: this instrument retains summary statistics, not
a p95. Faster absolute maps do not waive that unresolved release requirement.

### Experimental scope and implementation

There are **56 new MLP fits**: 48 initial fits across human and codec-proxy
objectives, plus eight added human finalist seeds. Two new deterministic
linear fits and the mixtures add no stochastic replicas. Previous H128
ensembles and linear120 are inherited controls, not new fits. Cached initial
fits take a median 17.56 seconds (4.36–32.51 seconds). The entire campaign is
not one five-minute experiment.

The existing feature-screen owner verifies the prior input, table, arithmetic,
sampling and extractor identities before reusing the 19,958-row preparation.
The human task has 7,000 fit / 1,000 dev / 3,125 test rows on 57/8/25 disjoint
references; TID is fit-only. The separate 620-row codec-proxy task preserves
its origin-family split and SSIMULACRA2 targets. No new fit uses the retained
8,213-row corruption task's old binary labels. No new source or protected
CID22/AIC/T0 value was admitted. Old decoded ladders are capability probes,
not evidence for current native codec RD.

Private planner masks now skip coarse X/B gathers, feature kernels, retention
copies and reductions for local-only basic consumers. Y at all scales and
shared pyramid production remain complete. Plan unions restore required
channels for other consumers; wider families stay conservative. Canonical
feature arithmetic, named profiles and the public API remain unchanged.
Prepared steering already skips map channels with zero sensitivity.

All 477 library tests pass, eight diagnostics remain ignored. Tests cover
canonical retained SDR/HDR values, identity, impulses/grid patterns, odd sizes,
threading, cache reuse and union dependencies. CI-exact Clippy, script lint
(612 scripts), formatting and the native/portable serving matrix pass. Final
packed/weighted compositions pass **559** pixel comparisons (43 per model)
with zero consumed-feature difference and matching cached/pixel scores.
The final broad replay has 1,248 model/case cells and 328,224 real block repairs
across eight training origins, three JXL levels and four block sizes. The
prior broad runs are preserved separately, not counted as extra independent
examples. Initialization failures in early packing/audit commands remain in
logs; only `pixel-served` and `broad-served` are final completed evaluations.

### Reproduction and next work

Artifact root: `~/work/zensim-validation-2026-09-13/runtime-profiles/`.
`screen/`, `confirm-small/`, `confirm-coarse/`, `linear/`, `packed/`,
`weighted-scores/`, `pixel-served/`, `broad-served/` and `timing/` contain the
owner results. `served-models.json` binds member paths, hashes and weights.
`FINAL_RESULTS.json` binds the final reports; `tools/` preserves measured
binaries and `reproduce/` preserves the executed orchestration. The adjacent
recipes reproduce preparation/fit/audit/report through the existing
`run_full_eval.sh --stage feature-screen` owner and `--ceiling-stage` options.
The plot is `tradeoffs.svg` / `tradeoffs.png` in the artifact root.

Prioritize reviewed severity-aware corruption labels and a catcher calibrated
against valid low-quality encodes, then allow inactive companions through the
prepared steering API with explicit failure on activation. Preserve the fast
coarse60 and spatially consistent mixture controls while improving the
remaining map/scalar cost. Before promotion, freeze an entire composition and
run independent human panels, witnessed per-image codec bounds, calibrated
1/2/3-shot targeting and native JXL/AVIF/JPEG/WebP spatial RD; HDR has its own
outstanding model/steering qualification. This study establishes useful
achievable points, not a universal feature ceiling or a shippable model.

## Chronological registration


Registered before fitting, September 13. Continue the prior steerable-subset
study with three cheaper scale allocations: local Y at all four native scales,
plus X/B only at scale 3 or scales 2–3. Compare fine-Y local100 and local120.
Use canonical Rev3 local SSIM/edge mean/L2/L4 and MSE, with explicit IDs.
The feature planner must skip the actual unneeded channel work at coarse
scales; removing model inputs alone is not a runtime result. Keep full XYB
pyramid production for downstream dependencies and unchanged retained values.

Fit human and codec-proxy objectives separately using the exact admitted,
hash-verified tables of the preceding study. No new corruption classifier fit:
its catastrophic/recoverable labels need review under the later contract.
Retained corruption/aliasing pixel scores remain diagnostics, not categorical
catastrophic detection claims. Human targets and codec-proxy targets must not
be pooled or described as one calibrated scale. Existing development test
panels are reused selection evidence; no protected terminal data is opened.

Screen H32 for all five allocations and H128 for the three new allocations,
three paired seeds 5101/5103/5107, 32 epochs, every-epoch dev selection. Reuse
the prior H128 fine-Y/local120 controls rather than repeat them. The two new
seeds 5113/5119 confirm useful finalists. Uniform five-seed ensembles and the
existing deterministic linear+spline control distinguish head capacity from
spatial coherence. Retain failed and dominated options. Select by dev error,
runtime and every spatial failure; test-panel results remain descriptive.

Use the existing Rust trainer, BakeScorer prediction/pixel audit, densifier,
panel and real block-repair instrument. Benchmark final compact bytes through
scalar and prepared score+map APIs at 1MP/4MP, pinned ST/MT8, >=30 quiet samples,
without target-cpu=native. Record complete model/input/binary/command identities,
dispersion and memory. Compare against prior candidates and D/fast-ssim2 in
the same timing run. Screen retained repair cases; broaden finalists across
the existing 96 source/quality/block-size cells, M2>=.99 and M3f>=.70.

Deliver measured runtime options and their quality limits; do not install named
profiles or qualify a model from these development results. Target-loop/RD,
new corruption operating points, terminal human and HDR qualification remain
separate required evidence. Consult the later corruption activation contract
when interpreting the previous report's blanket companion restrictions.

Implementation changes are private planner/dispatch data and orchestration
options in the existing feature-screen owner; no new public API is planned.

Confirmation registration after the three-seed development read: five-seed
human ensembles for y_coarse60/H128, y_mid80/H32, fine_y_local100/H32 and
local120/H32. Reuse the original three fits and add only seeds 5113/5119.
The old fine-Y/local120 H128 ensembles remain accuracy controls. Luma-only
H128 is retained as a three-seed lower-cost diagnostic, not a promoted finalist.
The y_mid80 larger head changes median dev MAE only 7.090 to 7.064, whereas
y_coarse60 improves 7.416 to 7.128; this determines the chosen head sizes.
No test-panel results determine this choice. Add deterministic lambda .01
linear+spline controls for y_coarse60/y_mid80 using the existing fit owner,
same fit-only human rows, to distinguish feature loss from nonlinear steering.

After broad repair evaluation, coarse60's five-MLP ensemble passes 95/96;
its sole failure is M2=.9860 at a 64px block on origin 8206, while its linear
control fails a different cell. Register mixtures using the same features:
the five MLP members plus 1, 2 or 5 copies of the fitted linear member, giving
linear fractions 1/6, 2/7 and 1/2 through the existing uniform ensemble API.
No new fit, label or source is involved. Evaluate all 96 cells for each, with
dev error and complete runtime; choose the smallest linear fraction that
passes every cell if any does. This is development selection, not qualification.

## Claude-era gate review after the runtime study

September 13, in response to the user's request for rigor when extending toward
944 features. This is an evidence audit and a continuation protocol, not another
fit result or a retrospective change to the earlier acceptance rules.

Reviewed the original late-August/September user turns in the recovered Claude
transcript, relevant project memories, the cookbook/B methodology, the zenpapers
feature adjudications, and the current playbook/scorecard. The earlier
[transcript audit](science_workflow_audit_2026-09-07.md) indexes the private
sources. This follow-up does not claim to reread every historical tool result.
Later split, feature-serving, floor-ruler and corruption rulings take precedence
over old memory headlines. In particular, July's broad impossibility claims about
maps and September 5's width-based serving restrictions are not current contracts.

### What the existing evidence actually covers

The saved broad result contains two origins each of photo, document, graphic
and screen content, three JXL levels and four block sizes: 24 cells per class.
Coarse60 passes 24/24, 24/24, 24/24 and **23/24** respectively; the selected
mixture passes 24/24 in each. The screen failure must remain visible beside
the pooled 95/96. This panel is already content-stratified; its limitation is
only eight previously used training origins, small images and block restoration
rather than native codec allocation. The mixture was selected on this panel.
Neither another replay nor more block sizes supplies independent source evidence.

Audit source: `runtime-profiles/broad-served/RESULT.json` in the shared artifact
root, SHA256 `0dff3fb1dd0b8a80f306da815169d665c6176edc3b93cb18ac5057d8ef35f22d`.

The earlier scale-selective study used M2 >= .8 and M3f >= .9 for its diagnostic
screen; this runtime study uses M2 >= .99 and M3f >= .70. Their pass counts are
not directly comparable. M3f block-refinement evidence also does not silently
substitute for every historical M3/M3a instrument or for native RD.

The 944-versus-228 table is a matched three-seed, single-model training study.
The newer 60-feature headline is a five-member ensemble. Comparing those
headlines cannot isolate feature availability. Likewise, a corruption-trained
944 class predictor's error is not the human-trained 944 scorer's bug-detection
performance. The old class labels predate the revised severity contract.

The fair gauntlet currently has 174 rendered model rows and none of these
runtime candidates. Its existing gate script passed on this audit: both script
blocks, render/SSR/badges, compare-link negative controls, and strict JSON for
511 input fullevals. That verifies the existing board, not the new models'
scientific qualification. No new fulleval was manufactured from the partial
runtime report and no historical board values were overwritten.

### Requirements for the next feature comparison

1. **Match the experiment before attributing a gain.** Compare coarse60,
   local120, basic228, selected619 and full944 controls on the same admitted
   source rows, Rev3 arithmetic, training objective, head size, checkpoint
   cadence, packing and calibration. Vary added feature families/scales in a
   separate axis from head architecture or corruption composition. Retain B/D,
   the relevant historical MLP/additive controls and independent metric peers
   on compatible evaluation surfaces; their historical scalar numbers are not
   transferable to a new feature era. Exact extension IDs and the bounded fit
   matrix must be registered before running them.
2. **Use replicated recipes, with attributable randomness.** Three paired
   initialization/order streams for screening, five for finalists; report
   per-seed values, mean and spread, plus the actual final ensemble separately.
   Preserve sampler configuration and sequence/coverage evidence. The September
   5 subset study falsified the assumption that favorable seeds merely covered
   more rows: order mattered and coverage had saturated. Check current trainer
   behavior instead of repeating that old hypothesis. Reuse hash-compatible
   prior fits where these conditions match; do not retrain merely for new names.
3. **Separate development from confirmation.** These KADID development-test
   rows have already influenced research. Keep them labeled exposed. Report
   per-distortion, quality-tail and content-class errors with counts, not only
   pooled MAE/SROCC. Keep teacher-proxy axes out of independent-human superiority
   claims. For uncertainty over content, use paired reference-cluster resampling;
   image pairs and training seeds are not independent reference samples. Freeze
   noninferiority margins before new comparisons; an interval containing zero
   does not prove equivalence. Terminal CID22/AIC/T0 data stay out of iteration.
4. **Gate spatial capability before expensive promotion.** Require complete
   Rust refinement coverage and scalar/map/pixel parity for the exact packed
   composition. Re-evaluate controls and candidates on one instrument/rule,
   with photo, dense text/documents, graphics and UI, native codec block sizes,
   larger images, odd geometry and bin/rectangle alignment cases. Register
   additional admitted sources before scoring them. Full944 remains a scalar
   information control while legacy weighted refinement is unsupported; do not
   silently omit those features from its map. Cheap kill tests precede sweeps.
5. **Keep product gates intact.** Use the current
   [scorecard](../docs/MODEL_SELECTION_SCORECARD.md), G-ADDR ladder/resolvable
   ruler and [target protocol](../docs/TARGET_STEERING_PROTOCOL_2026-09-08.md).
   Identity, above-identity failures, negative tails, tied regions and every
   required codec floor are separate from rank. Establish per-image codec
   bounds before judging 1/2/3-shot targeting; keep bounds hidden from the
   controller and seeds calibrated only on training families. Spatial value
   requires active/neutral controls and bytes at equal independently judged
   quality in the existing native JXL/AVIF/JPEG/WebP owners. SDR results cannot
   qualify HDR. The revised corruption gate must not fire on ordinary low-quality
   encodes; frozen severity labels and false-activation counts precede a new fit.
6. **Measure actual cost and keep the gauntlet current.** Benchmark complete
   final compositions on the same quiet build, inputs and threads, retaining
   samples sufficient for p95, cache/map costs and incremental worker RSS.
   Bin-8 output does not imply that full-resolution feature work disappears.
   After each completed owner fulleval, update the existing gauntlet and its
   review set, annotations and fairness grouping; run its render/data/negative
   controls. Missing, unsupported, failed and qualified remain distinct. A
   cohort-average floor cannot erase an individual seed's failure, and a cached
   composite cannot survive changed component statistics without recomputation.

The next comparison must establish a useful accuracy/runtime/steering tradeoff,
not a feature-count winner. Current candidate disposition remains **unqualified**:
artifact responses fail, full gauntlet/target/RD/HDR coverage is missing, and the
p95 map/scalar and release memory requirements have not been established.
