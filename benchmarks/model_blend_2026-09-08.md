# A/D blend: scalar screen passes, spatial coverage is incomplete

The [registered fit](model_preferences_2026-09-08.md#next-registered-model-experiment)
selects D weight 0.41: the first hundredth-grid weight that correctly orders all
1,647 independent-judge consensus pairs on the eight fit origins. The exact
ordered f64 weights are `[0.5900000000000001, 0.41]`, applied to the complete
calibrated generation-A and D scores. The four separate training-calibration
origins were not used to choose the weight.

All 824 calibration pairs also order correctly. Complete Rust pixel/cache
inference matches the independently calculated blend exactly on all 264 input
pairs. All twelve identities are exactly 100; no distorted score exceeds 100;
observed range is -24.4161067 to 97.4601024. This passes the registered base
preference screen. It is not held-out perceptual, floor, targeting or release
qualification. No network was trained and no validation/terminal input scored.

## Canonical serving and the newly measured spatial gap

The existing extractor audit accepts `--audit-ensemble a.bin,d.bin` and
`--audit-ensemble-weights <a>,<d>`. It calls `BakeScorer::ensemble`, preserving
member order and binding all file hashes plus weights in every audit row.
Weights must be explicit and pass the existing Rust convex-weight contract.
Single-bake audits preserve their scores and schema; exact A/D endpoint
controls and the previous single-model scores reproduce across all 264 rows.
The source/decoder/feature era remains that of the preceding canonical packet.

Ensemble audits additionally call the complete reference-cached attribution
surface at bin 8, requiring exact scalar score/feature equality and finite
maps. They report unsupported feature IDs rather than hiding missing terms.
This uses the existing combined extraction plan; no new features or scorer
implementation are introduced. The unpublished bench enables the existing
`custom-profiles` feature needed by those reference/attribution APIs. The first
build without that capability failed explicitly and is retained in the logs.

The blend and A-only endpoint each report missing pooled-feature terms on
**all 252 nonidentity images**. The D-only endpoint has no missing terms.
The missing sensitivities are in f156..371: v1 peaks and masked/IW pools.
Identity maps remain zero and all 792 ensemble spatial scores/features agree
exactly with their scalar counterparts. Thus the complete scalar is served,
but the density cannot yet represent all of its active inputs. **Do not wire
this blend into native steering as though the map were complete.**

Before committing to pooled-feature implementation, price this actual blend
against D and SSIM2 with the existing `ssim2_speed_bar` owner. A uses the full
v1 pool regime, so cheap D forwards do not make their shared extraction cheap.
The release performance bar remains binding. If the blend exceeds it, repair
cost or train a competitive model in a cheaper supported regime. If its cost
is acceptable, implement the missing pooled-feature attribution in Rust, with
feature-integral reconstruction and finite-block checks. Preserve explicit coverage reporting and the unchanged
scalar feature arithmetic. A zero local sensitivity is not proof that a finite
block edit cannot cross a max, clamp or gate. The historical H3 gain on named A
cannot qualify the current complete model through a partial map.

## Verification and reproducibility

The final software passes root CI-exact Clippy, the feature-exact nested
extractor Clippy, scoped formatting, sixteen existing surface tests (including
ensemble/corruption composition, HDR and cached attribution), and 605-script
lint. Fifteen invalid CLI configurations refuse before output creation, both
before and after enabling spatial auditing. Controls include missing/duplicate
members, missing/malformed/nonfinite/negative/wrong-sum weights and ambiguous
single-bake plus ensemble requests. No public inference API changed.

The two measured builds each execute five 264-row scalar audits: blend, A, D,
and A-only/D-only ensemble endpoints. The spatial build additionally evaluates
792 maps and 792 reference-cached comparisons. The fit reuses earlier judge
scores; no new encodes or judge evaluations occur. Timings are engineering
logs, not quiet-machine performance qualification. Previously failed corruption
heads are not attached to this base-only screen and remain required work.

Artifacts: `/mnt/v/output/zensim/model-blend-2026-09-08/`, mirrored to
`~/work/zensim-validation-2026-09-08/model-blend/`. `MODEL.json` pins both packed
member files, exact weights, source roles and the full weight-search trace.
`RESULT_FINAL.json` keeps scalar PASS and spatial INCOMPLETE distinct; `SPATIAL.json`
lists every missing feature. Both binaries, source copies, commands, hashes,
audits and refusal controls remain available. The final native instrument is
`extractor-spatial`; `extractor` preserves the first scalar-only build.
