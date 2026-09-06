# Feature system, phase 4 — dense layouts and the wire format

**Date:** 2026-09-05. **Lane:** `zensim--featsys2`.
**Design:** [`../docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md`](../docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md) §4.
**Plan + pre-registered gates:** [`../docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`](../docs/PLAN_FEATURE_SYSTEM_2026-09-05.md) phase 4.
**Predecessor:** [`feature_system_phase2_2026-09-05.md`](feature_system_phase2_2026-09-05.md).

---

## 1. What landed

`zensim::feature_layout::Layout` — a declared mapping from feature ids to
positions. `identity(width)`, `dense(&SlotSet)`, `width`, `walk_width`,
`slot_at`, `pos_of`, `ids`, `is_identity`, `gather`, plus `slots_of` /
`dense_slots_of` / `declared_layout` for resolving one out of a
`FeatureSetId` or a bake.

`Plan` now carries a `Layout` rather than a `layout_width: usize`, and gained
`walk_width()` — **the identity width the walk must emit**, which is the same
number as the layout width for every artifact that exists and is not for a
dense one.

`research::Request::dense()` and `Extraction::{layout_name, position_of}`;
`ZENSIM_RESEARCH_DENSE=1` on the extractor. The runtime gathers at the bake
boundary (`metric::forward_one_bake_with_codec`) when a bake declares a dense
layout, and takes today's exact code path when it does not.

Phase 1 landed this as `layout_width` + `LayoutBlocks` **and recorded why**:
every registered layout was the identity mapping, so a full map would have
been a type with no consumer and no test that could distinguish it from the
width it wrapped. `Layout::dense` is that consumer, and it arrived with three
real tests and three real defects.

---

## 2. Gate results

Run under `--features training,classification,custom-profiles,feature-regime-v2,threads`.

### G4.1 — a dense table and its `w944` equivalent are the same data

Proved three ways, at increasing distance from the type:

**(a) Values, in the research engine.**
`a_dense_layout_carries_the_same_values_as_its_sparse_equivalent`: a
`dense265` extraction and a `w944` one over the same pixels hold
bit-identical values on all 265 carried ids, located through *each layout's
own* `position_of` rather than by assuming positions. The dense arm carries
265 values and 265 provenance rows, all `populated`; the sparse arm carries
the same 265 plus **679** unpopulated positions.

**(b) Through a REAL bake, in the runtime.**
`zensim/tests/dense_layout_round_trip.rs` — table → bake → serve → same
vector. Two synthetic ZNPR v3 bakes built through the mandated JSON pipeline:
a 944-input one live only on the 265 ids, and a 265-input one declaring
`basic+peaks+moments@w265/era2r4#4fcef1d6` in `zentrain.feature_set_id`. Both
carry **distinct per-id weights**, so the forward pass is a checksum of *which
features arrived at which position* — a permutation or a truncation changes
the answer, which equal weights would have hidden. They score **bit-identically**
through `Zensim::compute` at 3 geometries.

**With a negative control**, because the positive result alone proves nothing:
a 265-input bake that does NOT declare its layout is served the identity
prefix (`f0..265` = `basic` + 37 slots of `masked`) and must score
DIFFERENTLY. It does. Without that assertion the test would pass just as
happily with an inert gather.

**(c) On real tables, through the extractor.** Four pairs, both arms:

| | layout | columns | producer id |
|---|---|--:|---|
| sparse | `w944` | 944 | `basic+peaks+moments@w944/era2r4#4fcef1d6` |
| dense | `dense265` | 265 | `basic+peaks+moments@w265/era2r4#4fcef1d6` |

**0 value mismatches over 1,060 cells.** Same compute tokens, same slot hash,
different declared width — which is exactly what `FEATURE_SET_IDS.md`'s
identity layer is for. File size 21,071 vs 29,898 bytes (**70.5 %**) — and
that number deserves a caveat: in CSV a structural zero costs 4 bytes, so the
honest saving is 679 columns, not 30 % of a well-compressed table. The point
of dense is that the wire format stops lying about what a row contains, not
that it compresses better.

### G4.2 — every existing artifact resolves and scores unchanged

* `every_legacy_width_is_the_identity_mapping` — all seven registered widths
  (156/372/504/720/924/944/956) satisfy `slot_at[i] == Some(i)` and
  `pos_of(i) == Some(i)` over their whole range, and `walk_width == width`.
  So declaring them moves no stored byte.
* `every_shipped_bake_resolves_to_an_identity_layout` — every bake in every
  shipped profile. The roster comes from `feature_plan::servability_census::
  shipped_profiles()`, the SAME `#[cfg]`-dependent list the servability census
  uses, rather than a second copy that could drift past a feature flag.
* `a_bake_declaring_a_dense_set_resolves_to_that_layout` — the **positive
  control**, without which the line above would pass if `declared_layout`
  always returned identity.
* The whole standing byte-gate suite is green (see §4).

### G4.3 — public API

**No new public type.** `Layout` is `pub(crate)`. `Request::dense()` and
`Extraction::{layout_name, position_of}` are additions to the already
`#[doc(hidden)]` `research` surface. `docs/public-api/zensim.txt` — the
SUPPORTED surface — is unchanged.

---

## 3. Three defects found by building it

None of these were reachable before a dense layout existed, and all three are
the same shape: **code that treats a position as an id**.

**3.1 The scoring path truncated where it must gather.**
`fold_engine::compute_fold_backed` sized its feature vector by
`plan.layout_width()`. For a dense plan that is the PACKED width (265), so the
walk was cut at 372 and the gather then read `f720..f941` off the end —
returning the structural fill for every raw-moment id. `Plan::walk_width()` is
now the single owner of *"how wide must the walk be for this layout to be
fillable"*, and both the scoring path and the identity short-circuit use it.

**3.2 `score_plan` compared caller widths.**
A dense bake declares 265 inputs, and `265 <= v1_width (372)`, so it took the
"narrow, nothing to plan" shortcut — which returns `None`, meaning no plan,
meaning a 372-wide walk. It now compares `Plan::walk_width()`. The shortcut
itself is worth keeping: it exists so the narrow non-skipping case stays on
the *identical* code path rather than an equivalent one.

**3.3 `ComputeSet::from_block_profile` reads POSITIONS as v1 slot indices.**
`caller_input_width() <= v1_total` selects its v1 branch, then
`bake_pool_need_from_model` interprets the live layer-0 columns as v1 slots.
Correct for an identity layout; wrong *by construction* for a dense one, where
position 228 is a raw-moment id and not a masked one.

MEASURED: for the `dense265` bake it derived `v1_pools: Full, free_extras:
Off`, whose `emit` does not contain the moment ids — so **`Plan::for_bake`
refused its own bake**, `score_plan` returned `None`, and the profile silently
fell back to a 228-wide walk. The failure was visible only as a wrong number.

Fixed by deriving a non-identity layout's compute in ID space
(`Plan::derive_with_layout`) while identity layouts keep
`from_block_profile` — the tested derivation, so **no served bake changes**.
`from_block_profile_agrees_with_the_id_space_derivation` is the standing
evidence for collapsing the two in phase 5 (and for retiring
`fold_engine::wide_bake_v2_read`, which exists only to serve
`from_block_profile`'s wide branch).

---

## 4. One hazard closed before it could bite

The first dense producer id came out
`basic+peaks+moments@w265/era2r4#3fb78648`. The real hash of that slot set is
`#4fcef1d6`.

Cause: `ComputeSet::feature_set_id` derives its hash from `populated_slots`
clipped to the width it is handed, and for a dense layout that width is the
PACKED count — so it clipped the family union to `0..265` and hashed a
*completely different* 265-member set.

That is worse than a cosmetic mislabel, because `declared_layout` would then
have **accepted** the wrong reconstruction: `slots_of`'s sparse reading of
`@w265#3fb78648` reproduces it, and it also has 265 members, so a bake
carrying that id would have had its input vector permuted into the wrong
features — silently, which is this whole design's characteristic failure.

Two changes close it:

* the producer id takes its COMPUTE TOKENS from the walk-width call and its
  HASH from `plan.emit`, which is already in id space
  (`the_dense_producer_id_carries_the_same_slot_hash_as_the_sparse_one`);
* `declared_layout` uses `dense_slots_of` — the STRICT reading (clip to the
  registry's full width, require the hash, require the packed count, refuse an
  identity range) — while `slots_of`'s two-reading form stays for callers
  reproducing a registered set. A sparse id can no longer be mistaken for a
  dense layout.

`Request::for_set` also stopped carrying its own copy of the reconstruction
(it clipped to `layout_width` unconditionally and so refused every dense id) —
`feature_layout::slots_of` is the one owner.

---

## 5. Standing gates, all green

| suite | tests | result |
|---|--:|---|
| `fold_engine_parity` | 14 | PASS |
| `v1_feature_width_pure_function` | 10 | PASS |
| `v1_golden_bytes` | 5 | PASS |
| `feature_set_id` | 9 | PASS |
| `feature_invariants` | 10 | PASS |
| `research_engine_parity` | 6 | PASS |
| `dense_layout_round_trip` | 1 | PASS |
| `zensim --lib` (`feature_*` + `research`) | 31 | PASS |

`cargo test --workspace` green; clippy `--all-targets --all-features` clean.

---

## 6. What is NOT done, stated plainly

* **No stored table or bake has been converted to a dense layout**, and none
  should be until a consumer wants one. The wire format is now *expressible*;
  retiring 944-with-structural-zeros is a decision about new artifacts, not a
  migration of old ones. Every legacy width remains a declared layout over the
  same ids, which is why nothing moved.
* **`Plan::union` of two DENSE layouts falls back to identity** at the wider
  walk width. A union of two different packings has no meaning (whose
  positions would it use?) and no caller — a profile's up-to-three bakes are
  all identity-laid-out. Registered here rather than half-implemented.
* **The parquet story is untouched.** The 70.5 % CSV figure above is not a
  compression claim; a columnar format with run-length encoding already stores
  679 zero columns cheaply. The win is that a dense row cannot be mistaken for
  a sparse one.
