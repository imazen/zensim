# Canonical corruption serving screen — September 8 preregistration

Before new instrument edits/results. Start with the existing frozen D model
SHA `cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6`
and historical nonlinear HGB w372 companion
`c95bd5f84c235e485df30f98c2009d6dbc7186618998df9e5de69ca0bc4dce15`.
No new fit yet: content admission remains incomplete. This is a development
screen, not held-out qualification of either model's historical training.

Extend only the existing `extract_features_372col` owner with optional private
CLI arguments `--audit-jsonl`, `--audit-bake`, `--audit-corruption-head`.
The sidecar records decoded packed RGB8 hashes, source file hashes, dimensions,
row keys and optional complete `BakeScorer` results. Decode through the current
shared native owner. The base bake and companion execute through the public
surface on actual decoded pixels and independently on canonical raw features.
Also record base-only cached score and head probability for interpretation.
Do not reconstruct the composition in Python. Preserve the ordinary extractor
CSV behavior/bytes when audit options are absent. No public API or new feature.
Add serde_json/sha2 to the existing zen-decode tool feature; the benchmark crate
already depends on zenpredict and enables zensim's corruption-head surface.

When a companion is present, compare every declared consumed feature with
canonical extraction: tolerance `1e-6 + 1e-5*abs(canonical)`, and final cached
versus pixel score tolerance `1e-4`. Report actual maxima, never only pass/fail.
Reject unsupported width, nonfinite values, missing/changed inputs, nonfresh
outputs, orphan arguments and any extraction failure. No absent audit row may
be counted complete. Record extraction, pixel comparisons, cached comparisons
and explicit auxiliary head evaluations separately; no maps or encodes here.

Run all 14,300 canonical catalog/anchor rows and the 760 keyed honest native
bitstreams, with source/split joins and exact expected rows. Preserve all raw
attempts, but compute diagnostic detection/FP using one source/pixel identity
per role, retaining removed counts and conflicting-label checks. Report by
source/content/family, honest codec/quality, inert identity, and native anchor
ordering. Keep negatives unchanged. The existing head's baked threshold is
frozen; no threshold tuning on validation.

This first screen asks whether the historical nonlinear gain transfers enough
to justify a canonical refit and where the complete composition fails. It
cannot qualify a model or waive C10, T0 content audit, a registered family-level
fit/calibration split, rank/floor/targeting gates, or spatial discontinuity
handling. Current native adapters reject corruption companions explicitly.

Before a new fit, replace the legacy trainer's CV-ensemble reporting/exported
single-fit mismatch with a canonical mode that evaluates the exact exported
artifact. Reuse its estimator factory, scaler, isotonic and ZCTH exporter.
Preregister exact identities, weights, hyperparameters and seeds separately.

## Identity-context amendment, before model/API edits

The first full screen stopped on exactly the 678 inert attempts: raw-feature
composition returned 0 and pixel scoring returned 100. This is the existing
documented distinction, not proof that the pixel override is wrong. Raw zero
features are not proof of pixel identity; the pixel path has that proof.
The first 715-row noninert-photo pilot had exact 0 feature and score deltas.
Preserve the failed full run and its strict rejection.

Add `BakeScorer::score_features_with_identity(features, width, height,
codec_hint, pixels_identical)` for the concrete extractor-audit caller and
cached feature consumers that retain verified pair identity. A true flag must
mean byte-identical decoded pixels in the same encoding and geometry, never
a zero feature row, score, source name or lossy hash. It returns 100 before
model inference, matching the existing pixel API; false delegates to the
unchanged raw-feature method. Route SDR/HDR pixel scoring through this same
identity-aware owner. Keep the old raw-feature API unchanged and add the API
snapshot/changelog. Test true and false on a firing corruption companion,
ordinary zero rows and negative scores. Audit compares both cached literal
model output and identity-aware output with explicit identity evidence; only
the latter is the complete pair score. Continue requiring exact expected
coverage and the already registered feature/score tolerances.

Before the final replay, also check the actual training-table precision:
round canonical features through f32, score with the same verified identity
context, require the same `1e-4` pixel-score tolerance and exact head fire set.
Retain the unrounded comparison and measured probability delta. The corpus
tables store f32, so testing only fresh f64 rows would miss a deployment edge.
