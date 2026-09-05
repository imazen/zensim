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
* **G1.9 (the refusal is still a refusal)** — a bake reading slots no plan can
  compute is refused with `PlanError::Uncomputable` naming the missing slots.
  Not served with zeros.
* **G1.10 (nothing that scored changes)** — every currently-servable bake's
  score, `raw_distance`, `mean_offset` and full feature vector are bit-identical
  before and after. This is G-BYTE applied at the scoring entry.

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
