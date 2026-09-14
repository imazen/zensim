# Minimal-latency and wide-model comparison

Registered September 13 after the strict train/eval ruling, before fitting.
Source checkout starts at a9a2cea0. No named model or default is changed.

## Fixed comparison

Eight configurations, five paired initialization seeds 6101/6103/6107/6113/6119
and order seeds +10000: Y40/H32,H128; coarse-Y60/H32,H128; local120/H128;
selected619/H128; full944/H128,H256. These span the measured cheapest extraction
layouts and the wide accuracy reference, with matched capacity controls.
Forty fits total; no best-of-five selection. All fit 32 epochs, 8192 pairs/epoch,
the existing both/rank+MSE objective and training-only checkpoint monitoring.
Eval never chooses epochs, transforms, calibration, mixture weights or seeds.
Report per-seed values, mean/spread and each complete uniform five-member
ensemble. Add the deterministic train-only linear60+spline control.

Train: canonical KADID TRAIN (5000 rows/40 refs) plus TID's approved all-train
3000 rows/25 refs. Eval: original canonical KADID SELECT (3125 rows/25 refs),
admitted as eval under its original split authority. This is an already exposed
eval population, not a fresh independent holdout. No historical screen/test
segment or mixed cache is opened or relabeled. No terminal/test data is read.
Pin original canonical-view hashes and source-family assignments. Fresh Rev3
Rust pixel extraction supplies all consumed features. Before fitting, verify
the KADID pixel/label row mapping against unchanged MSE feature coordinates in
the canonical view; any unresolved mapping mismatch blocks training.

## Gates and reporting

Use the existing feature-screen owner and final Rust BakeScorer surface.
Pixel/cache parity must hold. Evaluate every KADID distortion family and source;
report raw score error separately from the gauntlet's logistic-remapped MAE.
No independent human superiority is inferred from a teacher-proxy axis.

For spatial diagnosis, register eval-source cases before fitting, apply one
M2 >= .99 / M3f >= .70 rule, and retain unsupported full944 weighted terms as
unsupported. Seed audits and complete ensemble audits are separate. Source and
content coverage must accompany pass counts; a KADID repair check is neither
a screen-content gate nor native encoder RD. Existing current-incumbent bakes
may be compared on the same eval surface with their correct feature revision,
but historical trained-on-eval controls must remain annotated.

Benchmark final served compositions, uncached scalar and cached bin-8 map paths,
at 1MP/4MP, pinned ST and MT8, at least 30 accepted quiet interleaved rounds.
Record p95 if the existing owner can retain samples; otherwise leave the p95
gate unmeasured. Feature count, smaller bakes and mean latency cannot substitute
for measured complete-worker cost. Unsupported map paths are not timed as if
they represented the whole candidate.

Promote actual Rust-owner verdicts through the existing promotion tool into
the gauntlet. Add a review set, retain per-seed grouping and complete-ensemble
identity, annotate the exposed eval population and missing gates, regenerate
both views, and run the gauntlet render/data/compare-link checks. No copied
historical composite or fabricated full-eval passes. Current G-RANK/G-DIAL/
G-ADDR, corruption severity, target, native spatial RD, HDR and cost contracts
remain in force, limited to legally admitted eval populations. Missing evidence
stays incomplete and no model is qualified by this bounded experiment.

Stop after this fixed comparison and report the accuracy/latency frontier and
failures. Additional feature/head tuning requires a new training-only design.
Artifacts and exact commands live under
`~/work/zensim-validation-2026-09-13/minimal-top/`.

## Evaluation correction registered September 14

Before publishing: KADID contains decoded identities. Feature-only cached scoring
without that evidence differs from the public pixel API's exact-identity path.
Keep the fits frozen. Re-extract the admitted eval pairs with the existing Rust
native audit to obtain pixel hashes and identity flags; preserve earlier cached
panels as superseded diagnostics. Add an optional strict `pixels_identical`
column reader to the unpublished validation loader, consumed by the existing
`ensemble_score_rows` and `bake_verdict` corpus paths. Call the existing
`BakeScorer::score_features_with_identity`; do not synthesize scores in Python.
No supported zensim API change. Final seed/ensemble panels and gauntlet verdicts
must use this same identity-bearing eval view. A fixture must distinguish known
identity from zero features without identity proof and reject invalid flags.

## Final results, September 14

These supersede the initial feature-only score panels. All 100 verified identity pairs now execute the public Rust identity path. Training/checkpoints/model bytes are unchanged; the linear control additionally carries its known training feature-set metadata, with score parity verified.

| Configuration | Ensemble raw MAE | Signed SROCC | Scalar ST ms | Cached bin-8 map ST ms | Spatial cases |
|---|---:|---:|---:|---:|---|
| y40_h32 | 9.665 | 0.9091 | 8.07 | 26.23 | 4/4 pass |
| y40_h128 | 9.724 | 0.9135 | 8.08 | 26.38 | 4/4 pass |
| y60_h32 | 8.497 | 0.9290 | 8.17 | 26.64 | 4/4 pass |
| y60_h128 | 8.142 | 0.9323 | 8.18 | 26.87 | 4/4 pass |
| local120_h128 | 7.981 | 0.9328 | 15.78 | 55.87 | 4/4 pass |
| selected619_h128 | 7.337 | 0.9399 | 18.38 | unsupported | 4/4 INVALID |
| full944_h128 | 7.029 | 0.9433 | 53.97 | unsupported | unsupported |
| full944_h256 | 7.074 | 0.9430 | 54.09 | unsupported | unsupported |
| linear60 | 10.342 | 0.8965 | 8.09 | 26.27 | 4/4 pass |

MAE is in the served 0–100 score units with negative outputs preserved, without fitting an eval rescale. Every neural row is the uniform five-member ensemble. The full gauntlet separately reports its conventional logistic-remapped statistics; that diagnostic fit does not alter any served model.

Coarse60/H128 is the useful low-latency basis in this bounded study: 1.11 points more raw error than full944/H128, at 6.60× lower scalar latency. Moving from H128 to H32 barely changes latency and worsens coarse60 error. Local120 improves error by only 0.16 points at nearly twice the scalar cost; the seed spread limits that comparison. Doubling the full944 head does not improve this eval. These are candidates and negative findings, not universal feature ceilings or product qualification.

### Replicates

| Configuration | Seed mean raw MAE | Seed minimum–maximum |
|---|---:|---:|
| y40_h32 | 9.866 | 9.611–10.073 |
| y40_h128 | 9.978 | 9.395–10.752 |
| y60_h32 | 8.757 | 8.447–8.980 |
| y60_h128 | 8.564 | 7.824–8.997 |
| local120_h128 | 8.488 | 7.681–9.043 |
| selected619_h128 | 7.674 | 7.603–7.778 |
| full944_h128 | 7.447 | 7.268–7.543 |
| full944_h256 | 7.700 | 7.380–8.516 |

The adjacent [machine-readable results](minimal_top_2026-09-13.results.json) retain all five values, model hashes, timing dispersion and source hashes. Seed ranges are not confidence intervals, and four image sources do not establish broad spatial coverage. All per-source and distortion-family panels remain in `final-eval/`.

### Timing and spatial limits

AMD Ryzen 9 9950X3D; ST pinned to CPU 8, MT8 to CPUs 8–15. All four timing runs completed with at least 30 accepted interleaved rounds per model at 1024² and 2048² pixels, with zenbench not flagging the runs unreliable. Final whole compositions were measured through `BakeScorer`, including cached prepared bin-8 score+map. The input is the existing deterministic RGB8 texture fixture. Fast-SSIM2 ST takes 62.94 ms at 1024² in the same group; this is a timing comparison, not a human-quality comparison. The legacy D control had a process revision override and is excluded from scientific comparisons. p95 and complete-worker memory remain unmeasured.

The six compact compositions pass 24/24 complete-ensemble repair cells on the four registered KADID JPEG sources. Selected619 has nonfinite map predictions in all four cases and fails all 25 ensemble spatial/scalar parity audits; its finite correlation summary is invalid evidence. The gallery owner independently refuses these nonfinite blocks. The feature-screen gate now explicitly reports INVALID for such input, even when a correlation number is finite. Full944 remains unsupported for complete refinement. Both wide sets are refused by the prepared steering surface and are not given a misleading map timing.

### Gauntlet and release status

Forty seed rows, eight ensemble rows and the deterministic linear control were promoted from actual final `bake_verdict` JSON through `promote_fulleval.py`. The all-rows gauntlet has a **minimal / wide** review button. Both all/fair views pass script parsing, DOM/chart rendering, comparison-link behavior and strict JSON checks (560 source evaluations). Existing fairness rules exclude these KADID-only rows from the fair view because the required CID22 gate is absent. No fairness or product gate was waived; the conservative legacy KADID integrity label is explicitly annotated for this source-disjoint eval study.

Remaining before product qualification: broader admitted eval content (including screens), reviewed corruption categories/activation, scalar/map correctness and complete refinement for wider models, witnessed codec-specific attainable bounds, train-calibrated 1/2/3-shot targeting, native JXL/AVIF/JPEG/WebP spatial RD, HDR and p95/worker cost. No new corruption head, named profile or default was shipped.

### Validation and reproduction

15 split/control/spatial regression checks pass. Rust fixtures verify strict identity evidence, null/invalid rejection, unknown zero-feature handling, negative-score preservation and sequential/parallel identity row alignment. CI-exact Clippy, script lint and the API documentation check pass. Full native eval audit establishes the 100 identities and verifies the narrow scorer against all 3,125 original pairs. Packed model parity is preserved; no fitting used eval.

The recipe, source admissions, row-mapping negative control, tools, complete commands, original refusals and final evaluations are under the artifact root above. `final-eval/` is authoritative for score panels and verdicts; earlier `screen/audits`, `packed` score panels and `verdicts/` are retained feature-only diagnostics, superseded for served-score claims. The `served-audit/` parity failure and gallery refusal are retained. Scientific summary: `FINAL_RESULTS.json`. Download bundle: `/zensim/reports/minimal-top-2026-09-13/models-and-evidence.tar.gz` on the local gallery server.
