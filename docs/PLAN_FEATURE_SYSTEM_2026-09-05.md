# Feature-system refactor — the phased plan and its pre-registered gates

**Design:** [`FEATURE_SYSTEM_DESIGN_2026-09-05.md`](FEATURE_SYSTEM_DESIGN_2026-09-05.md).
**Status:** phase 1 in progress (this lane). Phases 2+ registered, not started.

Gates are **pre-registered**: written here before the code, and never edited to
match a result. A phase that fails its gate is reported failed, not re-scoped.

---

## Standing gates — every phase, no exceptions

| # | gate | how |
|---|---|---|
| **G-BYTE** | No shipped byte moves. | `zensim/tests/v1_golden_bytes.rs` (incl. the non-tight fixture) + `fold_engine_parity.rs` (11 tests, 18 geometries × {serial, rayon} × pools 1/2/3/8/16) + a full `to_bits()` dump over the 20-geometry set, diffed before/after. |
| **G-API** | Zero public-API delta. | `just api-doc-check` (`ZEN_API_DOC=check`). A phase that needs public surface stops and registers it. |
| **G-TEST** | `cargo test --workspace` green; `just clippy` (`-D warnings`) green; `cargo fmt` clean. | |
| **G-OWNER** | No duplicate implementation. Every derivation added here has exactly one owner, and any pre-existing function computing the same thing becomes its caller. | grep + a test asserting the wrapper agrees with the owner. |
| **G-APPEND** | Append-only. No id renumbered, no registry entry edited or deleted. | a test over the committed registry JSON. |
| **G-SERVE** | **UNIVERSAL SERVABILITY.** Zero refusals across all four census populations — shipped profiles, shipped bakes, board bakes, registered producer sets — except LOUD, named refusals for genuinely unregistered ids/revisions. | `feature_plan::servability_census` (no filesystem) + `serve_custom_bake --census` (filesystem tier). A new refusal is a phase-gate FAILURE, never a known limitation. |

---

## Phase 1 — the registry, the types, the plan, and servability

**Lands in this lane.** Byte-identical to shipped output.

### 1a. The definition registry

* A committed `benchmarks/feature_defs_registry.json` keyed `family:block_local`
  carrying the *declarative* fields (`name`, `statistic`, `cost`, `form`,
  `direction`, `kernel`, `revisions`) — **per signal, not per slot**.
* A generated `zensim::features::defs` table expanding it across
  (scale, channel) using the existing geometry constants
  (`FEATURES_PER_CHANNEL_*`, `APPEND2_PER_SCALE`, `CSFW_PER_SCALE`,
  `NUM_SCALES`) and the existing `idx`/`idx_append`/`idx_append2`/`idx_csfw`
  constants. The generator has no independent copy of the layout arithmetic.

**Gates:**

* **G1.1** — for every registered `ComputeToken`, the registry's slot set equals
  `ComputeSet::populated_slots`'s for a plan with exactly that family on. Both
  directions, all 12 tokens.
* **G1.2** — `slots_hash8` over the registry's slots for each of the 20
  registered sets in `benchmarks/feature_sets_registry.json` equals the `sets`
  entry's recorded `slots_hash8`. Any mismatch fails with both values.
* **G1.3** — names are unique, `[a-z0-9_]+`, and stable: a committed snapshot of
  `id -> name` must not change for any existing id.
* **G1.4** — the id arithmetic round-trips: `def_at(id).unwrap()` reconstructs
  `id` from `(family, block_local, scale, channel)` for every id in every
  registered layout.

### 1b. `FeatureSet` / `Layout` / `Plan`

* `Layout` with the five legacy widths registered as identity mappings, plus
  `Layout::dense(&SlotSet)`.
* `FeatureRequest` + its three constructors.
* `Plan::derive`, `covers`, `missing`, `cost`.
* `ComputeSet::from_block_profile` becomes a thin wrapper over
  `Plan::derive(&FeatureRequest::for_bake(model)).compute`.

**WHAT ACTUALLY LANDED, and the one deliberate deviation.** `Plan::{derive,
for_bake, v1, covers, union, toggles}` and `PlanError` landed as specified, and
`from_block_profile` is now a live call site reached through `Plan::for_bake`
(it was `allow(dead_code)`). The `FeatureRequest` wrapper collapsed into
`Plan::derive(&SlotSet, width)` + `Plan::for_bake(&Model)` — the same two
constructors, without a struct that only ever carried its two arguments.

**`Layout` landed as `layout_width` plus `LayoutBlocks::for_width`, NOT as an
id→position map, and that is a decision rather than an omission.** Every
registered layout today is the IDENTITY mapping over its range (`slot_at[i] ==
Some(i)`), so a `Vec<Option<u16>>` + `HashMap` would be a type with no consumer
and no test that could distinguish it from the width it wraps — dead weight
carried into the hot path. What the plan genuinely needs from a layout is
*which optional blocks the declared width reaches*, and that is what
`LayoutBlocks` computes, as a NESTED chain (`csfw ⇒ append2 ⇒ append`) matching
the assertions the walk already makes. **The full id→position map lands with
phase 4**, where `Layout::dense` gives it its first real consumer and its first
real test. Recorded here so the gap is a scheduled decision, not a silent
shortfall.

**Gates:**

* **G1.5** — for every bake in the shipped profile set (A, B, D, BHdr, C) and
  every bake fixture the existing `from_block_profile` tests cover, the wrapper's
  `ComputeSet` is **field-for-field equal** to the pre-refactor function's. The
  existing tests (`from_block_profile_*`, 6 of them) must pass unmodified.
* **G1.6** — `Plan::derive` agrees with the live `score_pool_mode` on the
  `V1PoolsMode` axis for every shipped profile.
* **G1.7** — the five legacy `Layout`s are identity over their range: for each,
  `slot_at[i] == Some(i)` for all `i < width`.

### 1c. Servability

The runtime serves any bake whose read set `Plan::derive` can compute, at the
bake's declared layout width — closing the three hard-codings
(`feature_v2.rs:7574`, `fold_engine.rs:158`, `metric.rs:4750`).

**Gates:**

* **G1.8 (THE servability proof)** — a 944-declared bake whose read set is
  `basic+peaks+moments` scores through `Zensim::compute`, and its feature vector
  is **bit-identical** (`to_bits()`) to the stored pools-944 table's row on the
  same pixels, on every slot the plan populates.
  **STATUS: PARTIAL, and the blocker is F5, not the plan.** The SERVING half
  passed — the arms plan to the cheap walk, emit 265 / 289 slots at layout 944,
  and score (`the_campaign_free_set_arms_plan_to_the_cheap_walk`, and the 392
  944-wide bakes in the census). The bit-exactness half is split by the defect
  audit's own measurement: `basic+peaks` (`f0..227`, 228 of the 265) is
  **bit-identical between the two routes**, and the 37 raw-moment slots are
  **NOT** — 9.12 % of cells exceed the 2e-5 parity bar, worst 3.63e-3, from
  catastrophic cancellation in `Σs²/n − (Σs/n)²` plus two reduction
  granularities. That is **F5, a pre-existing route defect**, and this lane
  registered it rather than fixing it because fixing it is a byte change to a
  feature, which the phase's own G-BYTE forbids. **Phase 2b lands it**, and its
  window is closing: the cost is zero shipped bytes only while no shipped bake
  reads those slots. Claiming G1.8 green today would require either weakening
  the bar to a tolerance or fixing a feature inside a byte-identical phase —
  neither is acceptable, so it is reported PARTIAL.
* **G1.9 (the refusal is still a refusal)** — a bake reading slots no plan can
  compute is refused with `PlanError::Uncomputable` naming the missing slots.
  Not served with zeros.
* **G1.10 (nothing that scored changes)** — every currently-servable bake's
  score, `raw_distance`, `mean_offset` and full feature vector are bit-identical
  before and after. This is G-BYTE applied at the scoring entry.

**Also landed in phase 1 (from the defect audit, 2026-09-05):**

* **G1.11 (census)** — the four populations go to zero refusals. MEASURED:
  profiles 8/10 → **10/10**; shipped bakes 8 → **13/13**; board bakes 32/433 →
  **433/433**; registered producer sets 3/14 → **14/14 plannable**. Combined
  bake census: **445 SERVED, 0 REFUSED**, six declared widths.
* **G1.12 (default-build `C`)** — `ZensimProfile::C` scores a non-identical pair
  on a default build (`candidate-profiles` is in `default`), and the emitted
  vector is the bake's full declared width. The test that pinned the opposite
  is INVERTED, and its original concern kept as its own assertion.
* **G1.13 (defects modelled, not flipped)** — F4, F5 and F15 are attached to
  the slots the audit named, with F4's and F5's fixes registered as
  **Proposed** revisions carrying their migration cost. Nothing is applied;
  `proposed_revisions_are_distinguishable_from_landed_ones` keeps "modelled"
  and "applied" from blurring.
* **G1.14 (identity decomposition)** — the registry reproduces the audit's
  measured split (15 reference-only + 12 `PJND_FRAGILITY`) from its own `Form`
  declarations.

**Not in phase 1:** the research engine, provenance output, revision selection,
dense layouts in anger, any wire-format change, any public API.

---

## Phase 2 — the research engine

A named, complete entry taking a `Plan`, over the buffered walk and the `oracle`
accumulators, emitting values **+ per-feature provenance**.

**Gates:**

* **G2.1 (engine parity)** — for every plan both engines can serve, at the same
  revision, every shared id agrees **bit-exactly**. Generalizes
  `fold_engine_parity.rs` from one pair of walks to any plan.
* **G2.2 (thread invariance)** — research-engine output is bit-identical across
  rayon pool sizes 1/2/3/8/16, matching the standard the v1 fix
  (`v1_feature_width_pure_function.rs`) already holds the extractor to.
* **G2.3 (completeness)** — the research engine serves `FeatureRequest::everything()`
  at every registered width, and its emitted slot set equals the registry's.
* **G2.4 (provenance is checked, not asserted)** — each feature's reported
  owning kernel is verified by a perturbation probe: disabling that kernel must
  change that feature and no other.

### RESULTS — phase 2 LANDED 2026-09-05

Record: [`benchmarks/feature_system_phase2_2026-09-05.md`](../benchmarks/feature_system_phase2_2026-09-05.md).
Code: `zensim/src/research.rs`, `zensim/tests/research_engine_parity.rs`,
`zensim/examples/v2_ab_extract.rs` (`ZENSIM_AB_MODE=research`).

| gate | result |
|---|---|
| **G2.1** engine parity | **PASS.** 3 tests × the 20-cell shared matrix: v1 layout (7,440 bit comparisons vs `compute_extended_features`), 944 layout (18,880 vs the production append2 walk), full 956 (19,120 vs the CSFW walk). Plus a real-corpus check: 60 CID22 pairs, `research(everything@944)` vs production `foldapp2pools`, CSVs **byte-identical** (sha256 `253d864c…`). |
| **G2.2** thread invariance | **PASS.** `Request::everything()` bit-identical across rayon pools 1/2/3/8/16 **and** equal to the serial answer, over all 24 pool-sweep cells. |
| **G2.3** completeness | **PASS.** `Request::everything()` emits and populates all **956** registered slots; every position resolves to a registry definition (zero `unregistered_*` rows). |
| **G2.4** provenance checked | **PASS**, and it FOUND A DEFECT (below). Perturbation probe over 5 families × 20 cells: every position the narrowed plan still populates is bit-identical to the full walk's, and every unpopulated position is exactly `0.0`. |

**Measured cost — reported, not gated** (60 CID22 pairs at 512×512, one
binary, arms interleaved, 3 reps, min per arm, compute-only µs/pair):

| arm | ms/pair | vs production |
|---|--:|--:|
| production `foldapp2pools` (944, pools Full) | 55.7 | — |
| **research** `everything@944` | **55.8** | **+0.2 %** (inside the run-to-run spread of 55.7–60.0) |
| research `everything@956` (adds CSFW) | 58.0 | +4.1 % — the CSFW block's own work |
| research `basic+peaks+masked+iw@372` | 27.5 | **0.49×** — the narrow plan really does skip |

So the named entry costs nothing measurable over calling the walk by hand, and
a narrower plan is genuinely cheaper. **The research engine is not a slower
second implementation; it is the same walk with a plan and a manifest.**

**Two design points where the plan doc's sketch did not survive the code, both
for measured reasons** (full argument in `research.rs`'s module doc):

* **It is NOT the buffered walk.** `streaming::compute_multiscale_stats_
  streaming` takes no `V2NewFeatureToggles` and mentions `append_block` /
  `csfw_block` nowhere — it is structurally v1-only, i.e. it tops out at 372
  of 956 slots. A research engine that cannot compute two thirds of the
  registry is not comprehensive.
* **It is NOT oracle-backed by default.** `feature_v2::oracle`'s `Neumaier` /
  `Exact` accumulators produce DIFFERENT bits from the production reduction —
  that is their purpose as a ruler. Using them would make G2.1 unsatisfiable
  by construction. The oracle stays the separately-gated precision ruler.

**DEFECT FOUND AND FIXED by G2.4 — `Plan` described walks that cannot exist.**
`V2NewFeatureToggles` has exactly ONE layout/compute separation (`v1_only`);
there is no per-block layout-only flag, because `ComputeSet::from_toggles`
derives `append`/`append2`/`csfw` from the same `*_block` flags that set the
WIDTH, and hard-sets `v1_basic: true`. So a plan saying "compute append but
not CSFW at layout 956" produced `toggles()` with `csfw_block` on (from the
width), the walk computed CSFW, and `emit` — derived from the un-normalized
request — called those twelve positions structural zeros. The probe measured
`f944` at **0.0678** on a plan that declared it unpopulated. Fixed by
normalizing through the toggles the plan itself emits, so `compute ==
ComputeSet::from_toggles(plan.toggles())` **by construction**
(`normalization_is_a_fixed_point`), applied in all four constructors
(`derive`, `for_bake`, `v1`, `union`). `emit` only ever WIDENS, so nothing
that planned before stops planning and no served bake changes — the whole
servability census still passes. The missing capability (a per-block
layout-only flag) is REGISTERED, not built: it needs a walk change and this
lane's scope is dispatch. `a_wide_layout_computes_every_block_it_reaches` is
its negative gate.

**Also landed (not a gate, a duplication removal):** the parity geometry
matrix has ONE owner, `zensim/tests/common/parity_cells.rs`. It was a
`const` inside `fold_engine_parity.rs`, and its four-cell pool-sweep extension
was written out identically at **three** call sites with nothing checking the
three stayed equal.

---

### Phase 2b — land F5 while it is still free

F5 (the free-40 raw-moment route-parity skew) is registered as a **Proposed**
revision, and its migration cost is **zero shipped bytes today** because no
shipped bake reads the raw-moment tranche. That is not a permanent property:
the campaign's 265/289 free-set arms are exactly the bakes that would start
reading it.

**Gate G2b.1** — compensated accumulation makes the free route and the append
kernel agree within the 2e-5 parity bar on the audit's 773-pair real-pixel
population (the synthetic-only gate that MISSED it must be replaced, not
re-run). **G2b.2** — no bake that reads those slots exists at landing time, or
the phase stops and re-prices. Land it BEFORE a bake reads them.

---

## Phase 3 — revisions

Per-slot revision history wired to a selector, so a research extraction can
reproduce a prior era's semantics for the slots that era moved.

**Gates:**

* **G3.1** — for each registered era pair, the registry's "slots this era moved"
  set is verified against a re-extraction on a fixture: exactly the listed slots
  differ. A slot that moves and is not listed fails.
* **G3.2** — selecting the current revision is byte-identical to not selecting
  one.
* **G3.3** — the shipped runtime's revision is named, and a table built at a
  different one is reported by `feature_set::check`, not silently consumed.

**Prerequisite, honestly stated:** G3.1 needs a re-extraction per era pair. Some
eras' inputs are gone (safesyn's decode cache; `docs/DATASET_HISTORY.md` §3.32),
so those pairs will be `NOT MEASURABLE` and must be recorded as such rather than
inferred.

---

## Phase 4 — dense layouts and the wire format

Make `Layout::dense` the default for new tables and new bakes, retiring
"944-with-structural-zeros" as the wire format **for new artifacts only**.

**Gates:**

* **G4.1** — a dense-layout table and its `w944` equivalent produce bit-identical
  scores through the same bake.
* **G4.2** — every existing artifact still resolves and scores unchanged.
* **G4.3** — public API delta for any new wire type is registered and approved
  **before** it lands.

### RESULTS — phase 4 LANDED 2026-09-05

Record: [`benchmarks/feature_system_phase4_2026-09-05.md`](../benchmarks/feature_system_phase4_2026-09-05.md).
Code: `zensim/src/feature_layout.rs`, `zensim/tests/dense_layout_round_trip.rs`,
`Plan::{derive_with_layout, walk_width}`, `research::Request::dense()`,
`ZENSIM_RESEARCH_DENSE=1`.

| gate | result |
|---|---|
| **G4.1** dense == sparse | **PASS**, three ways. (a) values: a `dense265` extraction is bit-identical to its `w944` twin on all 265 carried ids, found through each layout's own index. (b) **through a real bake**: two synthetic ZNPR v3 bakes (265-input dense-declaring, 944-input sparse) with per-id checksum weights score BIT-IDENTICALLY through `Zensim::compute` at 3 geometries, with a negative control (an *undeclared* 265-input bake, served the identity prefix) scoring differently. (c) on real tables: 265 vs 944 columns, **0 mismatches over 1,060 cells**, same producer slot hash `#4fcef1d6`. |
| **G4.2** nothing existing changes | **PASS.** All seven legacy widths are identity mappings (`slot_at[i] == Some(i)`); **every shipped bake resolves to an identity layout** — gated, with a positive control proving `declared_layout` CAN return dense. The whole standing byte-gate suite is green. |
| **G4.3** public API | **No new public type.** `Layout` is `pub(crate)`; `research::Request::dense()` and `Extraction::{layout_name, position_of}` are additions to the already-`#[doc(hidden)]` research surface. The SUPPORTED surface (`docs/public-api/zensim.txt`) is unchanged. |

**Three defects found by building it**, all measured, all fixed:

1. **The scoring path truncated where it must gather.** `fold_engine::
   compute_fold_backed` sized its vector by `plan.layout_width()`. For a dense
   plan that is the PACKED width (265), so the walk was cut before the gather
   could reach f720–f941. `Plan::walk_width()` is now the one owner of "how
   wide must the walk be", and the scoring path uses it.
2. **`score_plan` compared caller widths.** A dense bake declares 265 inputs,
   which is `<= v1_width`, so it took the "narrow, nothing to plan" shortcut
   and then had its ids gathered out of a 372-wide vector. It now compares
   `Plan::walk_width()`.
3. **`ComputeSet::from_block_profile` reads POSITIONS as v1 slot indices.**
   That is correct for an identity layout and wrong by construction for a
   dense one — under `dense265`, position 228 is a raw-moment id, not a masked
   one. MEASURED: it derived `v1_pools: Full, free_extras: Off`, whose `emit`
   does not cover the moment ids, so **`Plan::for_bake` refused its own bake**
   and the profile silently fell back to a 228-wide walk. Non-identity layouts
   now derive in ID space; identity layouts still use `from_block_profile`, so
   no served bake changes. `from_block_profile_agrees_with_the_id_space_
   derivation` is the phase-5 evidence for collapsing the two.

**And one hazard closed before it could bite:** the first dense producer id
read `basic+peaks+moments@w265/era2r4#3fb78648` — `ComputeSet::feature_set_id`
derives its hash from `populated_slots` clipped to the width it is handed, and
for a dense layout that width is the packed count, so it hashed a *different*
265-member set. `declared_layout` would then have accepted that wrong
reconstruction, because it also has 265 members, and permuted the vector a bake
is served. The id now takes its tokens from the walk-width call and its hash
from `plan.emit`; `dense_slots_of` is the strict reading `declared_layout`
uses, separate from `slots_of`'s two-reading form.

---

## Phase 5 — consumer migration

`bake_verdict`, the trainer, the extractors and the manifest writers take
`FeatureRequest`/`Plan` directly; the width-keyed `--regime` flag becomes an
alias resolver (it already prints its resolved id).

**Gates:**

* **G5.1** — every `--regime N` invocation resolves to the same set it does
  today, proven by a byte-identical verdict on a fixed bake + root.
* **G5.2** — the `--regime 944` silent-mis-scoring bug is **structurally**
  unreachable: the plan refuses rather than under-computing. Reproduced against
  the recorded instance (shipped B, CID22 0.3862 vs its true 0.8764).

### RESULTS — phase 5 LANDED 2026-09-05

Record: [`benchmarks/feature_system_phase5_2026-09-05.md`](../benchmarks/feature_system_phase5_2026-09-05.md).

| gate | result |
|---|---|
| **G5.1** | **PASS.** A fixed bake on a fixed root gives a verdict **byte-identical apart from the wall-time line**, before and after the whole phase. `--regime N` now also PRINTS its derived meaning, resolved through the default root's own manifest and the registry (`--regime 372 means regime v1-372 = basic+peaks+masked+iw@w372 (372 slots, #d16a1091)`) — and says NOT ESTABLISHED rather than guessing when it cannot be derived. |
| **G5.2** | **PASS**, reproduced: `--regime 944` on shipped B refuses, naming the 49 caller lines in `f156-371` the folded root feeds as structural zeros. That is the pre-existing hand-specialised guard; the GENERAL form (`feature_set::check`'s `SlotsNotPopulated`, every block at every regime) now refuses by default too — restricted to the case where BOTH ids are STORED, because an inferred id is evidence about a name and not about bytes. |

**Scope change forced by the revision lane's measurement, and it is the more
important half of this phase.** F5's fix is not free: all three shipped 944
bakes read the full `GLOBAL_DMEAN/CGAIN/CLOSS` set (33 slots each), so a
GLOBAL `SHIPPED_REVISION` flip would move **22 of 33 inputs per bake**. So
**revision became a PER-BAKE declaration**: `V2NewFeatureToggles::
formula_revision` + `ComputeSet::formula_revision` carry it through the walk,
`feature_v2::bake_formula_revision` reads a bake's `zentrain.formula_revision`
stamp (absent = the shipped revision), `Plan::revisions_agree` +
`score_plan` refuse a mixed-revision profile (one walk computes one era), and
`bake_verdict` refuses a bake/build revision mismatch naming both.

MEASURED: two revisions coexist in one process, differing on **11 slots**, all
`GLOBAL_CGAIN`/`GLOBAL_CLOSS`, with `GLOBAL_DMEAN` never moving — narrower
than `paired_global_contrast`'s name suggests and correct, since the fix is to
the paired CONTRAST estimate. The default toggles reproduce `Rev1`
bit-for-bit, so nothing that scores today changes. **This is what lets `Rev2`
ship without touching Profile C until C is refit.**

**Honest scope:** only F5's half of `Rev2` is per-request. Its luma-form half
(`ssim_form::active_luma_form`) is a `OnceLock` inside the SIMD kernels, and
making that per-request is a kernel-dispatch change this lane does not own.
And `ComputeSet::from_block_profile` is NOT yet collapsed into
`Plan::derive_with_layout` — the unit evidence holds
(`from_block_profile_agrees_with_the_id_space_derivation`) and phase 4 already
routes dense layouts through the id-space form, but retiring it (and
`wide_bake_v2_read` with it) needs the 445-bake census re-run as evidence, not
just a unit gate. Not done, not claimed.

---

## Sequencing and risk

Phases 1 → 2 → 3 are strictly ordered (2 needs the plan; 3 needs the registry's
revision field populated, which 2's parity work exercises). Phase 4 needs 3
(a dense layout is only safe once revisions are explicit). Phase 5 can start
after 1 and interleave.

**The largest risk is phase 1c**, because it is the only phase that changes
runtime behaviour at all. It is mitigated by G1.10 being byte-identity rather
than tolerance, and by the change being strictly additive (a refusal becomes a
success; no success changes).

**The second risk is scope creep into the kernels.** This design touches
*dispatch*, not *arithmetic*. No kernel is restructured, no summation order
moves, `dense_block_kernel` is not touched. If a phase appears to need a kernel
change, it stops and registers it.
