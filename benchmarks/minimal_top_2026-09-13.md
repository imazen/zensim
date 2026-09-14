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
