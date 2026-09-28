# Complete candidate feature audit — September 14, 2026

Registered against `044ce1e1` before implementation. The current extractor
checks canonical features only for tree-companion declared IDs. A no-head
ensemble therefore writes a zero maximum without comparing member inputs.
Scalar/cache equality can conceal cancellation or saturation and is insufficient.

Concrete caller: the existing `extract_features_372col` candidate audit. Exact
API addition: hidden `BakeScorer::consumed_feature_ids() -> Result<Vec<u16>,
ZensimError>`, gated by `feature-regime-v2`. It validates the serving plan and
reuses the existing structural read-set owner, covering active ensemble members
and corruption companions. IDs are sorted canonical feature IDs, independent
of local gradients or observed data. Unknown/unservable plans refuse.

Inspection also found that the structural owner only reads layer-zero weights,
while min-max metadata replaces that network with feature-dependent pieces.
Extend that owner using the existing min-max parser; test a head reading a
feature absent from network weights. No new metadata parser or scorer.

The existing audit will compare all consumed canonical inputs with pixel-path
features using its existing absolute/relative tolerance, check finite values
and width, and record the measured IDs and maximum. Keep the native audit-v2
and legacy audit-v1 schemas; add an explicit complete-read-set audit marker.
Old records without that marker remain weaker evidence, never upgraded in place.

Verification: asymmetric dense IDs, zero-weight ensemble members, a replacement
min-max head, corruption companion coverage, and a negative control whose
consumed input changes despite unchanged final score. Run relevant feature-build,
API snapshot and local lint checks. Then replay the already admitted 214 native
TRAIN pairs with fresh outputs and pinned models/input era. No fitting, EVAL or
TEST/TERMINAL reads. Passing establishes serving correctness within scope, not
model quality, HDR qualification or spatial RD benefit.

## Completed evidence

The diagnostic is implemented through the existing planner and structural
read-set helper. The min-max fix uses the existing parser and replacement
piece weights, rather than the placeholder network, with the one-to-one
transform contract checked before a skip decision. A sparse dense-layout
regression reads masked f300 while the placeholder reads basic f13; canonical
feature, cached/pixel score and finite-difference checks pass. Inactive linear
corruption gates retain their input IDs, and zero-weight ensemble members do
not add unused IDs. The cancellation negative control rejects wrong consumed
features even when their final model scores are exactly equal.

| Frozen audit composition | Rows | Consumed IDs checked | Maximum canonical/pixel feature difference |
|---|---:|---:|---:|
| Native372 Rev3 base plus tree integrity head | 214 | 228 | 0 |
| Native944 y70/y80 ensemble | 214 | 80 | 0 |
| Legacy372 base plus tree head compatibility control | 214 | 228 | 0 |

All 642 rows carry `feature_audit_scope: complete-structural-read-set-v1` and
the actual sorted consumed IDs. Every prior audit field is unchanged, including
scores, identity decisions, head activations and spatial query evidence. All
three CSVs remain byte-identical to their earlier runs. The JSONL files add
the explicit scope and IDs; they are not byte-identical to earlier JSONL.
Old records and binaries are preserved. All 215 source file hashes and model
hashes were verified before replay; no dataset roles or labels changed.

This closes the previously missing independent canonical consumed-feature
comparison for the tested ensemble, including its 626,249 recorded refinement
queries. It does not establish feature parity for every possible composition,
quality improvement, target accuracy, a native RD win, or HBD/HDR qualification.
The min-max fix needs re-evaluation for any affected historical candidate;
historical metrics must not be silently relabelled as corrected results.

Next admit the native audit-v2/input era explicitly in the strict TRAIN
pipeline, resolve original P3 encoding/q85 interpretation and restore mobile
corruption coverage before matching refits. All model NO-SHIP verdicts remain.

Local checks: 14 candidate/revision tests, 13 planner tests and the extractor
cancellation regression pass; one ignored planner diagnostic remains unrun.
Root and targeted bench clippy, API snapshot verification, minimal feature
builds, script lint and gauntlet render gates pass. The supported API snapshot
is unchanged apart from its hidden-surface count; the new method is recorded
in the internal snapshot. These checks do not replace scientific qualification.
