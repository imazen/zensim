# Public-API Ablation Report: `zensim`

**Date:** 2026-06-11
**Snapshot commit:** `c48a93e8` (regenerated on each `cargo test` run)
**Snapshot file:** `docs/public-api/zensim.txt`
**Method:** Read snapshot + source; org-wide grep for every candidate item excluding
the crate's own repo dir, `target/`, `.jj/`, `docs/public-api/`, and `*.txt` snapshots.
**Conservative bar:** flag only clear mistakes. If >10 % of items flagged, bar is too low.

**Grep command template:**
```
grep -r "<ITEM>" /home/lilith/work/ \
  --include="*.rs" \
  --exclude-dir=target \
  --exclude-dir=".jj" \
  --exclude-dir="zensim" \
  -l
```
Note: jj sibling workspaces (`_zensim-pu-panel`, `zensim--release-audit`) are zensim's own
worktrees and are excluded from "external usage" counts.

---

## Summary

| | Count |
|---|---|
| Default-features items | 835 |
| All-features items | 1,162 |
| Flagged A (non-breaking: add `#[doc(hidden)]` or `#[deprecated]`) | 10 |
| Flagged B (breaking: demote to `pub(crate)` or remove) | 4 |
| Total flagged | 14 |
| % of all-features surface | 1.2 % |

All 14 flagged items live in the `training` or `custom-profiles` feature sections. The
default-features surface is clean.

---

## Module: `zensim` (root — training-gated free functions)

These items appear only in the all-features snapshot. The `training` feature is documented
in `zensim/src/lib.rs` (lines 277-308) as an internal research surface that changes scores
and is not part of the stable metric API.

| item | class | evidence | proposed action | semver impact |
|---|---|---|---|---|
| `dissimilarity_to_score(f64) -> f64` | A | 0 external files (counterpart `score_to_dissimilarity` used by imageflow + zenmetrics; this inverse has zero external use) | Add `#[doc(hidden)]` or `#[deprecated]` | None |
| `compute_iw_weights(…) -> Vec<f32>` | B | 0 external files; `training`-gated; doc comment says "research experiments" | Demote to `pub(crate)` in next training-surface cleanup | Breaking within `training` feature only |
| `try_score_from_features(&[f64], &[f64]) -> Result<(f64, f64), ZensimError>` | B | 0 external files; `training`-gated; low-level feature-scoring plumbing | Demote to `pub(crate)` | Breaking within `training` feature only |

---

## Module: `zensim::cvvdp_features` (training-gated)

Module doc: "EX-4 extended feature modules — only meaningful inside the feature-extract pipeline;
the metric hot path never calls them." (`zensim/src/lib.rs` lines 233-241)

| item | class | evidence | proposed action | semver impact |
|---|---|---|---|---|
| `pub mod zensim::cvvdp_features` | B | 0 external files importing or calling it | Demote whole module to `pub(crate)` or remove `pub mod` | Breaking within `training` feature only |
| `CVVDP_FEATURE_COUNT: usize` | B | 0 external files | Covered by module demotion | Same |
| `extract_cvvdp_features(…) -> Vec<f32>` | B | 0 external files | Covered by module demotion | Same |

---

## Module: `zensim::xyb_lms_features` (training-gated)

Same rationale as `cvvdp_features` — per lib.rs comment, "only meaningful inside the feature-extract
pipeline."

| item | class | evidence | proposed action | semver impact |
|---|---|---|---|---|
| `pub mod zensim::xyb_lms_features` | B | 0 external files | Demote to `pub(crate)` | Breaking within `training` feature only |
| `XYB_LMS_FEATURE_COUNT: usize` | B | 0 external files | Covered by module demotion | Same |
| `extract_xyb_lms_features(…) -> Vec<f32>` | B | 0 external files | Covered by module demotion | Same |

Note: `cvvdp_features` and `xyb_lms_features` modules are already gated by
`#[cfg(feature = "training")]`. Demoting them to `pub(crate)` is the natural follow-through:
external code with `features = ["training"]` could currently reach these functions, but the
module doc explicitly says they are pipeline-internal.

---

## Module: `zensim::display` (default features)

| item | class | evidence | proposed action | semver impact |
|---|---|---|---|---|
| `DisplayCalibration::alpha: f32` (public field) | A | 0 external files reading or writing this field; jxl-encoder defines its own `DisplayCalibration` struct unrelated | Add `#[doc(hidden)]` to the three fields; provide getters if callers ever appear | None |
| `DisplayCalibration::beta: f32` (public field) | A | 0 external files | Same | None |
| `DisplayCalibration::ppd: f32` (public field) | A | 0 external files | Same | None |

`DisplayCalibration` itself (the type) is used by `ProfileParams` and should stay public.
Only the raw field access is the issue — direct field writes bypass any future invariant checks.

---

## Module: `zensim` (training-gated consts and types)

| item | class | evidence | proposed action | semver impact |
|---|---|---|---|---|
| `WEIGHTS: &[f64; 228]` | A | 0 external files outside own repo; `training`-gated; zenmetrics uses the named `WEIGHTS_PREVIEW_V0_1` / `WEIGHTS_PREVIEW_V0_2` instead | Add `#[doc(hidden)]` | None |
| `IwWeightConfig` struct (all fields pub) | A | 0 external files (zenmetrics uses `IwWeightKind` only in comments referencing GPU kernel doc strings, not as imported type) | Add `#[doc(hidden)]` on struct; move toward `pub(crate)` in next training cleanup | None |
| `IwWeightKind` enum | A | 0 external type uses | Add `#[doc(hidden)]` | None |

---

## Items NOT flagged (confirming conservative bar)

These were considered and kept:

| item | reason kept |
|---|---|
| `score_features_with_profile`, `score_features_with_profile_and_codec` | zenmetrics (`zensim-gpu/src/opaque.rs`, tests) |
| `score_to_dissimilarity` | imageflow_core + zenmetrics |
| `WEIGHTS_PREVIEW_V0_1`, `WEIGHTS_PREVIEW_V0_2`, `LINEAR_WEIGHTS_PREVIEW_*` | zenmetrics (`opaque.rs`, `pipeline.rs`) |
| `ProfileParams` + all public fields | coefficient/examples/spot_zensim_ba_weights.rs |
| `ProfileParamsBuilder` + all methods | Required by `custom-profiles` feature contract |
| `ZensimConfig` struct + fields | zenmetrics doc-comments reference field names; struct is `#[non_exhaustive]` |
| `BlurKernel`, `DownscaleFilter` | `ZensimConfig` field types — kept with ZensimConfig |
| `compute_iw_features: bool` field on `ProfileParams` | Part of ProfileParams public contract |
| `ZenpixelsSource` | `zenpixels` feature integration point |
| `UnsupportedFormat` | Error type needed when `zenpixels` feature is active |
| `CH_X`, `CH_Y`, `CH_B` consts | Feature-indexing constants used alongside `FEATURES_PER_*` |
| `FEATURES_PER_CHANNEL_BASIC` etc. | Training data dimension helpers — coherent group with `try_score_from_features` callers |

---

## Top 10 Highest-Confidence Ablations

Ranked by confidence (zero external hits + clear internal-only intent documented in source).

1. **`pub mod zensim::cvvdp_features`** (B) — Module doc + lib.rs comment: "only meaningful inside
   the feature-extract pipeline; the metric hot path never calls them." Zero external imports.
   Demoting the `pub mod` to `pub(crate)` prevents any external training code accidentally
   depending on a pipeline-internal feature shape that changes with experiments.

2. **`pub mod zensim::xyb_lms_features`** (B) — Identical rationale as cvvdp_features. Both modules
   were added together as EX-4 research, gated by `training`, and the crate comment is explicit
   about their scope.

3. **`compute_iw_weights`** (B) — `training`-gated; source doc says "for research experiments";
   no external callers in 18 months of codebase history. Belongs beside its consumers in
   `zensim-gpu` as a `pub(crate)` or re-exported with an opaque newtype if GPU ever needs it.

4. **`try_score_from_features`** (B) — Raw feature-vector scoring plumbing exposed under `training`.
   Only call sites are in the crate's own tests. The stable entry point for GPU backends is
   `score_features_with_profile[_and_codec]`, which wraps this.

5. **`dissimilarity_to_score`** (A) — The inverse of `score_to_dissimilarity` (which is used by
   imageflow and zenmetrics). The inverse is unused outside zensim itself. Hiding it avoids
   callers depending on the specific mapping math, which changes between profile versions.

6. **`DisplayCalibration::alpha`, `::beta`, `::ppd` fields** (A) — Public struct fields on an
   internal display-calibration type. No external reads or writes found. `#[doc(hidden)]` on
   the fields is the non-breaking fix; making them private is the clean fix if a breaking change
   is planned anyway.

7. **`IwWeightConfig`** (A) — Training-internal weight-tuning struct. Zero external type-level uses.
   Used exclusively as an argument to `compute_iw_weights` (also a candidate).

8. **`IwWeightKind`** (A) — Enum consumed only by `IwWeightConfig`. doc-comment mentions in
   zenmetrics are about GPU kernel constants, not the Rust enum type.

9. **`WEIGHTS: &[f64; 228]`** (A) — Unnamed alias shadowed by the more specific
   `WEIGHTS_PREVIEW_V0_1` / `WEIGHTS_PREVIEW_V0_2` that zenmetrics actually imports.
   Keeping a nameless `WEIGHTS` const creates confusion about which profile it represents
   (currently it points at V0.2 but has no version label in the name).

10. **`CVVDP_FEATURE_COUNT` + `XYB_LMS_FEATURE_COUNT`** (B, covered by module demotion) — Exported
    size constants for internal feature vectors. Their values are load-bearing only inside the
    feature-extract pipeline; exposing them invites external code to assume feature-vector shapes
    that change with experiments.
