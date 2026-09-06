# The cruft purge — retiring positional feature layouts

**Status:** pre-registered 2026-09-06, before any code in this lane. Gates below
are written first and are never edited to match a result. A gate that fails is
reported failed, not re-scoped.

**Design this executes:** [`FEATURE_SYSTEM_DESIGN_2026-09-05.md`](FEATURE_SYSTEM_DESIGN_2026-09-05.md)
(phases 1-5 landed; see [`PLAN_FEATURE_SYSTEM_2026-09-05.md`](PLAN_FEATURE_SYSTEM_2026-09-05.md)).
**Identity layer:** [`FEATURE_SET_IDS.md`](FEATURE_SET_IDS.md).
**Defects:** [`FEATURE_DEFECTS_AUDIT_2026-09-05.md`](FEATURE_DEFECTS_AUDIT_2026-09-05.md).

---

## 0. The ruling, and what it means concretely

**USER RULING (2026-09-06, verbatim):** *"get rid of the cruft and confusion,
the technical debt here is a huge problem for research and production serving
imo. a 372 layout where the bake skips features and features aren't computed is
a bad contract"*

That sentence names a specific artifact, and it exists. **MEASURED 2026-09-06**
(`bake_block_profile` over `zensim/weights/*.bin`; full output at
`/mnt/v/output/zensim/purge-2026-09-06/shipped_bake_block_profiles.txt`):

| shipped bake | profile | declares | layer-0 rows | caller lines READ | positions the runtime carries and nobody reads |
|---|---|--:|--:|--:|--:|
| `d_sdr_add156_id100_negrich_dial_2026-09-05` | **D (default)** | 372 | 372 | **28** | **344 (92.5 %)** |
| `d_sdr_add156_dense_dial_2026-08-31` | D (era-1) | 372 | 372 | 28 | 344 |
| `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07` | **B** | 372 | 372 | 95 | 277 (74.5 %) |
| `bhdr_linear_shaped_cvvdpmix_2026-07-12` | **BHdr** | 372 | 372 | 133 | 239 |
| `v47_strict_qat_native_2026-05-27` | **A** | 372 | 372 | 285 | 87 |
| `c_sdr_purity944_2026-08-29` | **C** | 944 | 667 | 667 | **277, via `FeatureTransform::Drop`** |
| `c_hdr_l1t1944_2026-08-29` | **CHdr** | 944 | 697 | 697 | **247, via `FeatureTransform::Drop`** |

Two distinct lies ride the same wire, and the ruling names both:

* **"the bake skips features"** — C and CHdr declare a **944** caller width and
  carry 247-277 `drop` transforms. The runtime must emit 944 positions so the
  bake can throw a quarter of them away.
* **"features aren't computed"** — D declares **372** and the runtime, correctly,
  does not compute `f156..371` at all (`V1PoolsMode::Off`). Those 216 positions
  are **written 0.0 because the layout says an id lives there and the plan does
  not populate it**. A consumer cannot tell that zero from a measured zero.

**And nothing declares what it reads.** MEASURED: **0 of 11** shipped bakes carry
a `zentrain.feature_set_id`. The mechanism landed in phase 4
(`feature_layout::declared_layout` + `dense_slots_of`) and no shipped artifact
uses it.

## 0b. THE CONTRACT (five lines)

1. **A bake declares the feature ids it reads.** By id, in its own metadata — not
   by a width, not by inference from which layer-0 rows happen to be nonzero.
2. **A table stores exactly the ids it holds.** An id it does not hold is
   **ABSENT** — no column, no zero.
3. **The runtime computes exactly the declared ids** (at kernel granularity) and
   **emits exactly those, dense, in ascending id order**.
4. **Research and production speak the same contract** — same ids, same layout
   type, same refusal, differing only in plan breadth and provenance output.
5. **A mismatch is a loud, named refusal.** Never a structural zero, never a
   positional prefix, never a silent `Drop`.

## 0c. Non-negotiable: shipped SCORES do not move

Every increment is byte-identical on shipped output. The wire, the API and the
code shape change; the numbers do not. Public API changes ARE authorized by the
ruling but are **batched under `CHANGELOG.md` "QUEUED BREAKING CHANGES"** for
0.3.0. **No crates.io publish in this lane.**

---

## 1. Standing gates — every increment, no exceptions

| # | gate | how |
|---|---|---|
| **P-BYTE** | No shipped byte moves. | `zensim/tests/v1_golden_bytes.rs` (incl. the non-tight fixture) + `fold_engine_parity.rs` + `feature_invariants` + a full `to_bits()` dump over the 20-cell parity matrix (`zensim/tests/common/parity_cells.rs`), diffed before/after. |
| **P-SCORE** | No shipped bake's prediction moves. | `bake_verdict --full-json` on a fixed bake + fixed root, before vs after: the diff is the **wall-time line only**. |
| **P-SERVE** | Servability census stays at **zero refusals**. | `feature_plan::servability_census` (no filesystem) + `serve_custom_bake --census` (filesystem tier, `zensim/weights` + `--fulleval-dir`). A new refusal is a gate FAILURE, never a known limitation. |
| **P-TEST** | `cargo test --workspace` green; `just clippy` (`-D warnings`) green; `cargo fmt` clean. | |
| **P-API** | Every public-API delta is **ENUMERATED here** and mirrored into `CHANGELOG.md` QUEUED BREAKING CHANGES **in the same commit**, with `cargo public-api` output attached. `cargo semver-checks` is RUN and its verdict recorded. | `just api-doc-check` regenerates `docs/public-api/*.txt`. |
| **P-OWNER** | One owner per task. Any converter is an extension of the canonical tool, never a new script. | grep + a test asserting any wrapper agrees with the owner. |
| **P-APPEND** | Append-only. No id renumbered, no registry entry edited or deleted. | the existing registry test. |

**P-PERF (reported, not gated):** the `zensim_D` bench arm must be unchanged or
better. The box is shared; **measure only when idle**, and check
`~/tmp/fastclass2_w4_deferred.log` before any pinned sweep. If the box is not
idle, report `NOT MEASURED` — never a number taken under contention (the
2026-09-01 own-process-contention finding).

---

## 2. Increments

Each lands as its own commit, each independently gate-clean.

### A. Inventory (measured)

Every legacy concept enumerated with `file:line` and classified **DELETE /
REPLACE-BY-PLAN / DEPRECATE-SHIM / KEEP**. Recorded in
`benchmarks/cruft_inventory_2026-09-06.md`. No code change.

**Gate A.1** — every concept in the ruling's scope appears with a class and a
count: regime enums and `--regime`, width literals (156/228/265/289/300/372/376/
504/720/924/944/956), structural-zero producers, `FeatureTransform::Drop`
skipping, `V1PoolsMode`/`V1FreeExtras`/`skip_unread_pools`, `from_block_profile`,
`wide_bake_v2_read`, `caller_input_width` positional arithmetic,
`prep_bake_input_f32` widening. A concept found later that is not in the
inventory is an inventory failure, recorded as such.

### B. Bakes declare their ids — `bake_dial_refit densify`

The owner tool gains a mode that rewrites a bake to the dense contract: layer-0
rows, scaler mean/scale, feature transforms and params, bounds and sparse
overrides all **permuted and packed to the read set**; `n_inputs ==
caller_input_width == |read set|`; **zero `Drop` transforms**; the dense
`zentrain.feature_set_id` stamped. Old bakes are **kept on disk** as retired
copies; `zensim/weights/manifests/` updated.

**Gate B.1 (prediction identity)** — for every converted bake, `bake_verdict
--full-json` before vs after differs **only in the wall-time line**. Not a
tolerance: a byte diff.
**Gate B.2 (score identity through the runtime)** — `Zensim::compute` on the
20-cell parity matrix returns bit-identical `score`, `raw_distance` and
`mean_offset` for the dense bake and its wide original.
**Gate B.3 (no Drop survives)** — every converted bake has zero `drop` transforms
and `n_inputs == caller_input_width`. Asserted over `zensim/weights/*.bin`.
**Gate B.4 (the declaration is READ, not decorative)** — a converted bake whose
`zentrain.feature_set_id` is removed must **refuse or differ**, proving the
runtime gathers by the declaration rather than by a positional prefix. A negative
control: an undeclared bake of the same width scores differently.
**Gate B.5 (retired copies)** — the pre-conversion bytes remain on disk with their
sha256 recorded.

### RESULTS — A and B, 2026-09-06

**Increment A: DONE.** `benchmarks/cruft_inventory_2026-09-06.md`. Gate **A.1
PASS** — every concept in scope carries a class and a count. The headline
INVERTS the obvious prior: the production positional layer is thin (**11** width
literals in `zensim/src`, every one already a `const`; **0** inline) while the
debt is in tests (**343**), consumers, and 105 `--regime` call sites.
`wide_bake_v2_read` has **1** production caller and **0** test callers.

**Increment B: the TOOL and the DECLARATION are DONE; the CONVERSION is
BLOCKED for the 944 class, and the consumer side is NOT YET WIRED.**

| gate | result |
|---|---|
| **B.1** prediction identity | **PASS on 9 of 11**, and the 2 exceptions are a NaN class, not a numeric one — 16 of 512 probe rows on `v47_strict_qat_native` and `bhdr_..._anchored2`, every one of them a dropped line whose own value was NaN. `fma(NaN, 0.0, acc)` is NaN, so a zero weight row still poisons the wide bake. Reported with a count, never silently allowed. |
| **B.2** score identity through `Zensim::compute` | **PASS on 8 of 11** (bit-identical served score on real pixels, `serve_custom_bake --census`). **FAIL on the three append2-bearing bakes** — see §5 of `benchmarks/dense_bake_contract_2026-09-06.md`. Cause MEASURED and it is a PRE-EXISTING defect, not densify's: `Plan::for_bake`'s identity-layout branch derives `append2_dst_activity: true` and its id-space branch derives `false`, and the canonical extractor defaults **false**. Shipped C and CHdr therefore serve on a BANDVIS formula their weights never saw, worth **0.87 / 0.31** zensim points. |
| **B.3** no `Drop` survives | **PASS** — every densified output has 0 `drop` transforms and `caller_input_width() == n_inputs()`, asserted in the tool before it writes. |
| **B.4** the declaration is READ | **PASS** — `an_explicit_feature_id_list_resolves_to_that_dense_layout` pins both halves: the declared bake gets the dense layout, and the SAME width with the declaration removed falls back to identity. Plus a strict-parse gate over duplicate / descending / unparseable / empty / out-of-range. |
| **B.5** retired copies | N/A yet — no shipped bake has been REPLACED. |

**Two blockers, both measured, neither hand-waved:**

1. **The `append2_dst_activity` skew** (above) blocks densifying C and CHdr AND
   blocks increment D, because both would adopt the honest `false` and move
   shipped scores. The fix is one line and it is **a user decision**, not a
   lane's. Registered in `CLAUDE.md` "Known Bugs".
2. **The consumer side still slices positionally.**
   `bake_runtime::score_row` — the DEDUP-M canonical dispatch every eval tool
   inherits — copies `row[..n_inputs]` and zero-pads. `zensim`'s runtime gathers
   by declared id; `zensim-validate`'s does not. **So a dense bake would be
   MIS-SCORED by `bake_verdict` and every sibling**, silently, by reading the
   first `|read set|` POSITIONS instead of the declared IDS. Swapping any shipped
   bake to dense before that is wired would be a data-corruption bug, so **no
   shipped bake was replaced.** This is increment B-2 and it is the next step.

The width-floor fix landed in the same pass closes the OTHER half of that hole
(a corpus narrower than the bake is now refused rather than zero-filled).

### C. Tables store exactly the ids they hold

A converter **at the canonical owner** (the `pack_*` / extract tool, never a new
script) rewrites a wide table to dense-by-id: drops absent-id columns, stamps
`feature_set_id` in `_MANIFEST.json`, preserves row order and every kept value
byte-for-byte.

**Gate C.1** — on a converted table, every kept column is **bit-identical**
(`to_bits()`) to its source column, and the row count and row order are equal.
**Gate C.2** — a bake scores **bit-identically** from the dense table and from
its wide source (this is phase 4's G4.1 applied to a real stored artifact rather
than a synthetic one).
**Gate C.3** — the dropped columns were **all-absent**, i.e. every dropped column
is a structural zero for the whole table, proven by a full-column scan, not
sampled. A column with any nonzero value is NEVER dropped; if one is found the
increment stops and reports it.
**Gate C.4** — `_MANIFEST.json` carries `feature_set_id`, `build_commit`, and
per-file sha256; the source root is **not modified and not deleted**.

Scope here: the eval roots, the eval instruments (dial/corruption grids), the
dial anchors, and the fast-class legs. **bigcodec and KADIS are REGISTERED as
fleet jobs** through the existing `JobKind::Feature` executor, **not run in this
lane** — they are millions of rows.

### B-2. The consumers gather by id (NEXT — the prerequisite for converting anything)

`bake_runtime::score_row` takes a caller-sized `&mut [f32]` scratch and fills it
positionally. It must instead take a per-bake row adapter that knows the bake's
declared layout: identity ⇒ today's copy, byte for byte; dense ⇒ a gather.
`score_row_minmax` has the same coupling (it indexes `transforms[i]` and `row[i]`
at layer-0 positions) and needs the same adapter.

**Gate B-2.1** — for every bake that exists today (all identity layouts) the
scratch fill is byte-identical and `scripts/verify_verdict_identity.sh` reports
**0 mismatches** on a fixed bake + root.
**Gate B-2.2** — a dense bake scored through `bake_verdict` agrees BIT-EXACTLY
with the same bake scored through `Zensim::compute` on the same pixels.
**Gate B-2.3** — a dense bake reaching an un-migrated scorer REFUSES rather than
slicing. No positional fallback survives that could quietly serve the wrong ids.

### D. Delete the superseded derivations

`ComputeSet::from_block_profile` and `fold_engine::wide_bake_v2_read` collapse
into `Plan::derive_with_layout`. This is phase 5's own named next step; its unit
evidence already exists (`from_block_profile_agrees_with_the_id_space_derivation`)
and phase 5 correctly refused to claim it without the census.

**Gate D.1** — the 445-bake servability census is re-run and is **identical**:
same SERVED count, same per-bake plan. Not just "still zero refusals" — the same
plan for the same bake.
**Gate D.2** — both symbols are **gone from the tree** (grep returns only
historical prose in `benchmarks/`), and every test that exercised them exercises
the id-space derivation instead.

### E. The runtime speaks ids

`Zensim` gains id-addressed feature access; positional accessors are **deprecated
with shims for one release**, not removed. `feature_set_id` gains the
**layout-free** id form (`<compute>/<era>#<hash>`) with every existing
`<compute>@w<layout>/<era>#<hash>` kept as a **registry alias** — with a dense
wire the layout component is redundant, and the user has authorized naming
changes.

**Gate E.1** — every existing registered id string still resolves, via alias, to
the same slot set and the same `slots_hash8`.
**Gate E.2** — the id-addressed accessor and the positional one return the same
value for every id in every identity layout.
**Gate E.3** — P-API: the delta is enumerated in §3 below before it lands.

### F. `--regime` becomes a derived print

`--regime N` resolves through the registry and **prints its derived meaning**
(phase 5 already does this); the remaining width-keyed branches become alias
lookups. Regime enums that no longer select behaviour are deleted or deprecated
per §3.

**Gate F.1** — every `--regime N` invocation in a committed script produces a
**byte-identical verdict** (modulo the wall-time line) before and after.
**Gate F.2** — the `--regime 944` silent-mis-scoring class stays structurally
unreachable (phase 5's G5.2, re-run).

### G. Docs

`FEATURE_SET_IDS.md` rewritten to the clean contract with the era zoo demoted to
a **migration appendix**; `CLAUDE.md`'s regime/pool/skip sections rewritten **in
place** to the new truth (dead guidance deleted, measured history left in
`benchmarks/`); cookbook + `DATA_PROVENANCE` pointers for converted tables;
ledger ROUND rows.

**Gate G.1** — no doc left in tree describes a deleted symbol as live. grep for
each deleted symbol across `*.md` returns only past-tense/benchmark prose.

---

## 3. Public API delta — enumerated BEFORE it lands

Pre-registered so that a delta appearing outside this list is a gate failure.
`docs/public-api/zensim.txt` is the supported surface. Everything here is
**QUEUED for 0.3.0**, batched, not shipped piecemeal, and **nothing is published**.

**Proposed REMOVALS (breaking; all currently in the supported surface):**

| symbol | why it goes |
|---|---|
| `feature_v2::FeatureRegime` (6 variants) | a regime is a width plus a set of structural zeros — exactly the concept being retired. Replaced by a declared feature set id. |
| `feature_v2::ZensimV2Result::regime()` | returns the above |
| `feature_v2::V1PoolsMode` (4 variants + `CARRIER_SLOTS`) | "which pools ran" is a `Plan` property, derived from the declared ids, not a caller-set mode |
| `feature_v2::V1FreeExtras` (3 variants) | same |
| `feature_v2::V2NewFeatureToggles::v1_pools` (field) | same |
| `feature_v2::ZensimV2Result::v1_pools()` | same |

**Proposed ADDITIONS:** an id-addressed feature accessor and the declared feature
set of a result. Named in the increment that lands them; each is additive.

**DEPRECATE-SHIM (kept for one release, `#[deprecated]`):** positional feature
accessors and any width-returning method a downstream may call.

**Decision rule if an increment needs surface not listed here:** it STOPS and
adds it to this list in its own commit first.

---

## 4. Sequencing, risk, and what this lane will NOT do

A → B → **B-2** → D are ordered (nothing may be CONVERTED before B-2, and D
needs B's census). C is independent. E and F need B-2. G is last.

**Both of D's symbols are additionally blocked on the `append2_dst_activity`
decision**, because collapsing `from_block_profile` into the id-space derivation
adopts `false` and moves shipped C / CHdr scores. That is the pivotal blocker of
this whole program and it is one line plus a decision.

**The largest risk is C**, because it rewrites stored artifacts. Mitigated by:
the source root is never modified, gate C.3 scans every cell of every dropped
column rather than sampling, and the dense output is proven to score
bit-identically before any consumer is repointed.

**Explicitly out of scope:** any kernel/arithmetic change (this is dispatch and
wire shape); any renumbering (append-only stands); flipping F4 or F5 (they stay
registered `Proposed`); any crates.io publish; bigcodec/KADIS conversion (fleet
jobs, registered not run); `zenanalyze-api` (frozen).
