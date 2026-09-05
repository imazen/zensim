# The feature system — definitions, plans, two engines

**Status:** design, 2026-09-05. Phase plan + gates:
[`PLAN_FEATURE_SYSTEM_2026-09-05.md`](PLAN_FEATURE_SYSTEM_2026-09-05.md).
Identity layer (landed, consumed here unchanged):
[`FEATURE_SET_IDS.md`](FEATURE_SET_IDS.md).

**User directive (2026-09-05, verbatim):** *"can you refactor all of that into
something to be proud of? a comprehensive all-feature version for research, and
an optimized version with maximum performance for what is key, and a software
shape and contract that just works end to end for any feature subset or feature
revision"*

Three asks, and they are the three sections below: **comprehensive** (a research
engine that can compute every registered feature at every registered revision),
**optimized** (a production engine that computes exactly what a caller needs and
nothing else), **a contract that just works** (adding, subsetting or revising a
feature requires no change in any consumer).

---

## 0. What this is on top of

`FEATURE_SET_IDS.md` landed the **identity** layer on 2026-09-05: a feature set
is `<compute>@w<layout>/<era>#<slots-hash8>`, `zensim::feature_set_id` owns the
grammar and the hash, and `zensim_validate::feature_set` derives producer and
consumer ids and checks coverage. That work is not re-opened here. It answers
*"are these two vectors the same kind of thing?"*.

This design answers the three questions it deliberately did not:

| question | today | this design |
|---|---|---|
| *What IS slot 353?* | nothing in-tree can say — `feature_set_id` reasons about slot **families**, and per-slot naming exists only as a 13-entry array inside a `#[cfg(test)]` diagnostic (`streaming.rs:5878`) and four `pub mod idx*` constant blocks that give a **local** offset inside one block | a **definition registry**: one entry per slot id, generated from the `idx*` constants that already exist |
| *Given a consumer's needs, what must run?* | `ComputeSet::from_block_profile` (`feature_v2.rs:2056`) — and it is `#[cfg_attr(not(test), allow(dead_code))]`, i.e. **not on any runtime path** | a **`Plan`**, derived once, the only thing either engine dispatches on |
| *Can the runtime serve this bake?* | for a bake declaring > 376 inputs: **no**, with a generic error | a plan-coverage question with a specific answer |

---

## 0b. UNIVERSAL SERVABILITY — a hard contract requirement

**User directive (2026-09-05): *"also make sure everything can be served"*.**

**The contract.** Every bake whose FeatureSet consists of registered feature
ids at a supported revision MUST be servable by the production engine through
`Zensim::compute`, and by the research engine, **in any declared layout**. No
"trains fine, cannot be served" class may exist. A refusal is legitimate only
for genuinely unregistered ids or revisions, and it must be LOUD and name the
slots — never silent zeros, and never a truncated prefix.

**It was violated at scale, and the violation was invisible.** The feature
defect audit (`docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md` §4A, measured at
`3376baee`) censused four populations and found the same single error in all
four — `ModelForwardFailed { reason: "bake declares more input features than
the caller supplied" }`:

| population | n | SERVED before | SERVED after increment 1 |
|---|--:|--:|--:|
| shipped profiles (`ZensimProfile`, default features) | 10 | **8** | **10** |
| shipped bakes (`zensim/weights/**.bin`) | 11 / 13 | **8** | **13** |
| board bakes (behind the 467 fullevals) | 433 | **32** | **433** |
| registered producer sets | 14 | **3** | **14** plannable |

`ZensimProfile::C` and `CHdr` are SHIPPED, `candidate-profiles` is DEFAULT-ON,
and both were unservable on any non-identical pair. **The identity
short-circuit hid it**: `ref` vs `ref` returns `100.000000` before the model
runs, so the profile looked alive on the one pair anybody smoke-tests.

**The census is a gate, in two tiers, with one driver.**

* `zensim::feature_plan::servability_census` (unit tests) — no filesystem, so
  it runs everywhere: every shipped profile, every registered producer set, and
  the campaign's 265/289 free-set arms. Its report carries a BEFORE column
  derived from the REMOVED `prep_bake_input_f32` rule (`declared <= v1_width +
  4`), so the comparison is reproduced rather than recalled.
* `zensim/examples/serve_custom_bake.rs --census` — the filesystem tier
  (`zensim/weights`, `--fulleval-dir`). It drives the SAME `Zensim::compute`
  entry as the single-bake mode the fastclass2 campaign already documents:
  extended, not duplicated.

**Measured after increment 1: 445 bakes, 445 SERVED, 0 REFUSED**, spanning six
declared widths (156, 372, 504, 720, 924, 944). The plan's phases must keep
that list empty; a new refusal is a phase-gate failure, not a known limitation.

---

## 1. The measured problem

Not a tidiness argument. Five defects, each already recorded, each an instance
of the same missing layer.

**1.1 The runtime serves exactly one layout, by three independent hard-codings.**
`Zensim::compute` on the fold path:

* `feature_v2.rs:7574` builds its request as `V2NewFeatureToggles { v1_pools,
  v1_only: true, ..Default::default() }` — `free_extras` is pinned to the
  default (`Off`) with no way for a caller to ask otherwise;
* `fold_engine.rs:158` then does `features.truncate(v1_feature_width(config))`,
  discarding any layout wider than 372;
* `metric.rs:4750` (`prep_bake_input_f32`) refuses when the bake declares more
  than `features.len() + 4`, with `"bake declares more input features than the
  caller supplied"`.

Consequence, measured by the fastclass2 campaign: **a 372-layout bake serves; a
944-declared bake is refused** — *including* one whose entire read set is
`basic+peaks+moments`, which the fold already computes, at 944 positions, via
`V1FreeExtras::RawMoments`. The capability is present and unreachable. Three
hard-codings, each locally defensible, add up to "the product can only ever ship
one feature set".

**1.2 The one function that could have answered is dead.** `from_block_profile`
derives a `ComputeSet` from a bake, and `wide_bake_v2_read` (`fold_engine.rs:364`)
is called from nowhere else. Both are test-only. Its own doc records why —
wiring it in would replace a cached lookup with a per-call parse on the hot path.
That is a real objection to *that* function, and the answer is not to wire it in
but to **derive the plan once, where the bake is loaded**, and cache it beside
the bake metadata that is already cached (`cached_bake_metadata`).

**1.3 The derivation that IS live is a boolean, not a plan.** `score_pool_mode`
resolves one axis (`V1PoolsMode`) from the profile's bakes. Every other axis —
v2 blocks, append, append2, csfw, free tranches, scales — is fixed at the
request site. So the runtime can skip the masked/IW pools and nothing else, and
a new axis means a new special case at a new call site.

**1.4 Structural zeros are the wire format.** A "944" row is 944 f64s of which
216 may be zero *because the family was not computed* (`ext944`) or nonzero
*because it was* (`pools`) — same width, same slot numbering, different data.
`FEATURE_SET_IDS.md` §3.1 counts **seven** distinct feature sets called "944".
The identity layer makes them distinguishable; it does not make the wire format
self-describing. A layout should be a **declared mapping from ids to positions**,
and the legacy widths should be three such declarations over the same ids —
which costs nothing, changes no stored byte, and makes a dense subset expressible
for the first time.

**1.5 There is no revision axis at all.** Every era break in this repo
(`v1pre → v1cur → v1postc`, the pools flip, era-2) changed what a slot's value
IS while its id, name and position stayed put. The era token records *that* it
happened, per table. Nothing records **which slots a given era moved**, so
"is my number affected?" is answered by re-extracting and diffing — measured at
several hours per instance, several times in the last month.

---

## 2. Layer 0 — the definition registry

**One entry per slot id.** Generated, not hand-written: every field either comes
from a constant that already exists in `feature_v2.rs` or is a declaration this
design adds, and a test proves the generated table agrees with the existing
owners.

```rust
pub struct FeatureDef {
    pub id: u16,                    // the stable slot id — append-only, never renumbered
    pub name: &'static str,         // [a-z0-9_]+, unique, stable
    pub family: ComputeToken,       // the FEATURE_SET_IDS vocabulary — one owner
    pub block_local: u16,           // index within the block (the `idx*` constant)
    pub scale: u8,                  // pyramid scale 0..num_scales
    pub channel: Channel,           // X | Y | B | Scalar (per-scale slots)
    pub statistic: Statistic,       // Mean | L2 | L4 | L8 | Max | WeightedMean | Ratio | Bin
    pub cost: CostClass,            // Free | Cheap | Expensive
    pub form: Form,                 // Difference | ReferenceOnly | Distorted
    pub direction: Direction,       // HigherIsWorse | HigherIsBetter | Unsigned
    pub kernel: KernelId,           // the owning kernel — the "who computes this" link
    pub revisions: &'static [Revision],
}

pub struct Revision {
    pub era: &'static str,          // the FEATURE_SET_IDS era token this landed in
    pub commit: &'static str,       // the byte-changing commit
    pub note: &'static str,         // what changed and what it means for stored numbers
}
```

**Why each field is here, and not one more:**

* `family` + `block_local` + `scale` + `channel` — these four **reconstruct the
  id arithmetic** that is currently written out at every emit site
  (`scale * 39 + ch * 13 + fi` for basic, `scale * 3 * PER_CHANNEL + ch *
  PER_CHANNEL + local` for v2/append, `scale * PER_SCALE + local` for
  append2/csfw). Generating both directions from one table is what makes
  "ids, names and hashes cannot drift" true rather than aspirational.
* `form` is load-bearing and currently undocumented per slot. A
  **difference-form** feature is 0 on an identity pair; a **reference-only** one
  is not. `LUMA_MEAN_REF` being reference-only is exactly why the 944 identity
  probe is not the zero vector while the 372 one is — a fact that cost a lane a
  day and is currently prose in one benchmark doc. As a registry field it is a
  one-line query and a test.
* `direction` is the monotone expectation the dial gates already assume
  informally. Making it declarative lets the invariant probes (the audit lane's)
  be **generated over the registry** instead of hand-listed.
* `cost` is what makes the plan's cost model real rather than a guess: `Free`
  = falls out of an accumulator another slot already needs (the raw-moment and
  class-C tranches are exactly this), `Cheap` = shares a sweep, `Expensive` =
  its own pass.
* `revisions` is the missing era axis from §1.5, **per slot**. `f353 moved at
  era X` becomes a lookup instead of an extraction. A revision carries a
  `RevisionStatus`: **Landed** (the values moved) or **Proposed** (the defect
  is known and modelled, the fix is NOT applied). Modelling a fix is not making
  one — the point is that a fix can be scheduled with its blast radius already
  known.
* `defect` records a LIVE defect from
  `docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`, keyed by its id, so *"does
  anything this bake reads have a live defect?"* is a query over its read set.
  Three are modelled today, none flipped:
  * **F4** — v1's SSIM per-pixel dissimilarity has a `.max(0)` floor, no upper
    cap, and `num_m` carries no `C1`; `f313` reaches 5.8e6 against a
    photographic p99.9 of 0.48. Attached to the three `ssim_*` signals in BOTH
    the masked and IW blocks (72 slots, including the audit's two worst,
    `f241` and `f313`). Its fix is registered as a **Proposed** revision
    (`v1ssimcap`) because applying it changes v1's SHIPPED bytes on every
    high-chroma image and the migration is re-extract **and** re-verdict the
    whole 372 lineage. The bake-side winsor guard absorbs it today.
  * **F5** — the free-40 raw-moment route-parity skew (9.12 % of cells over
    the 2e-5 bar, catastrophic cancellation in `Σs²/n − (Σs/n)²`, and two
    reduction granularities). Registered as **Proposed** (`freecomp`) with the
    note that this is the CHEAPEST window: **no shipped bake reads those slots
    yet**, so the migration cost is zero shipped bytes today and rises the
    moment one does. Land it before a bake reads them, not after.
  * **F15** — `PJND_FRAGILITY` is nonzero on an identity pair. Declared
    `ReferenceOnly` because it IS computed from the reference; the defect is
    the VALUE, not the form.
* **The identity decomposition is pinned to the registry.** A 944-wide identity
  vector resolves into exactly 15 reference-only slots (correct by design —
  `grad_src_mean` at the 11 append cells the 944 walk computes, plus
  `luma_mean_ref` once per scale), 12 `PJND_FRAGILITY` slots (F15), and fp
  residue ≤ 1.12e-3. The registry reproduces the first two counts **from its
  own `Form` declarations** (`identity_nonzero_slots_decompose_exactly_as_the_audit_measured`),
  so the flag encodes what the identity probe measures rather than merely
  claiming to.

**The registry is append-only**, matching the feature-numbering directive
(2026-07-19: new features get new indices after all existing ones; deprecate,
never renumber). A retired slot gets `deprecated: true` and keeps its id
forever. A revision **appends** to `revisions`; it never edits history.

**It is generated.** The generator reads the `idx`/`idx_append`/`idx_append2`/
`idx_csfw` constants and the block geometry constants
(`FEATURES_PER_CHANNEL_*`, `APPEND2_PER_SCALE`, `CSFW_PER_SCALE`,
`NUM_SCALES`) — the existing owners — and emits the table. The declarative
fields (`form`, `direction`, `cost`, `kernel`, `revisions`) live in a committed
JSON keyed by `family:block_local`, i.e. **per signal, not per slot**, so
declaring a new v2 signal is one JSON entry rather than twelve.

---

## 3. Layer 1 — sets, unchanged

`zensim::feature_set_id` stays exactly as landed. The registry makes two of its
operations *derivable* rather than hand-maintained:

* `ComputeToken → SlotSet` is currently a `match` on ranges inside
  `ComputeSet::populated_slots`; with the registry it is a filter over
  `FeatureDef::family`. One derivation, and the existing function becomes its
  caller.
* `slots_hash8` is unchanged and stays THE hash owner.

---

## 4. Layer 2 — Layout, made explicit

```rust
pub struct Layout {
    name: &'static str,     // "w372", "w944", "dense", …
    width: usize,
    slot_at: Vec<Option<u16>>,   // position -> feature id
    pos_of: HashMap<u16, usize>, // feature id -> position
}
```

A layout is a **declared mapping from ids to positions**. Three facts follow,
none of which are true today:

1. **The legacy widths are declarations, not code paths.** `w372`, `w720`,
   `w924`, `w944`, `w956` are `Layout`s whose `slot_at[i] == Some(i)` over their
   registered range. Every stored table, every shipped bake, every board row is
   already in one of them. **No stored byte moves.**
2. **A structural zero is a layout property, not a data property.** A position
   whose id the plan does not compute is written `0.0` *because the layout
   declares that id lives there and the plan does not populate it* — which is
   exactly the distinction §1.4 says is missing, now representable.
3. **A dense subset becomes expressible.** `Layout::dense(&SlotSet)` packs a
   consumer's read set with no gaps. Nothing requires it yet; it is what makes
   "no more 944-with-zeros as the wire format" reachable without breaking the
   944 that exists.

The `Layout` type is what the code already gropes at with the
`layout_append` / `append_on` pair at `feature_v2.rs:8815` — LAYOUT (is the
block present in this width) is *already* separated from COMPUTE (is it being
computed) at that one site. This generalizes that distinction instead of
re-deriving it per block.

---

## 5. Layer 3 — the Plan, and the two engines

### 5.1 The request

```rust
pub struct FeatureRequest {
    pub want: SlotSet,        // which feature ids the caller needs
    pub layout: LayoutRef,    // how to lay them out
    pub revision: RevisionRef,// which era's semantics (default: current)
}
```

Three constructors cover every caller that exists:

* `FeatureRequest::for_bake(&Model)` — the read set from
  `block_profile`, the layout from `caller_input_width()`. **This is the
  servability fix.**
* `FeatureRequest::for_set(&FeatureSetId)` — reproduce a registered set, e.g.
  to re-extract a root.
* `FeatureRequest::everything()` — the research engine's default.

### 5.2 The plan

```rust
pub struct Plan {
    pub compute: ComputeSet,      // the EXISTING type — derived, not hand-set
    pub layout: Layout,
    pub emit: SlotSet,            // what will actually be populated
    pub scales: ScaleRange,
    pub cost: CostEstimate,
}

impl Plan {
    pub fn derive(req: &FeatureRequest) -> Result<Plan, PlanError>;
    pub fn covers(&self, want: &SlotSet) -> bool;
    pub fn missing(&self, want: &SlotSet) -> SlotSet;
}
```

`Plan::derive` is the **single** place that maps "what is wanted" to "what
runs". It replaces `ComputeSet::from_block_profile` (which stays as a thin
wrapper over `Plan::derive(&FeatureRequest::for_bake(model)).compute` until its
test callers migrate) and subsumes the live `score_pool_mode` axis.

The derivation is: for each `ComputeToken`, does `want` intersect the registry's
slots for that family? If so the family is on. The free tranches are the one
interesting case, and the registry makes it declarative rather than special: a
slot with `cost: Free` whose owning block is off is served by the free
accumulator, which is precisely what `V1FreeExtras` encodes today by hand at
`feature_v2.rs:8807`.

**Cost is part of the plan, not an afterthought.** `CostEstimate` is summed from
`FeatureDef::cost` plus the measured per-block map from the kernel lane, so
`Plan::cost` can be compared before running, and the named plans below are
first-class rather than folklore.

### 5.3 The two engines

Both take a `Plan`. That is the whole contract.

| | RESEARCH | PRODUCTION |
|---|---|---|
| entry | `research::extract(plan, src, dst) -> Extraction` | `Zensim::compute*`, plan-driven |
| covers | every registered feature, every registered revision | whatever `Plan::derive` can build |
| priority | correctness, provenance, determinism | speed |
| walk | buffered / oracle-backed, reference semantics | the fold, minimal plan |
| threads | thread-invariant by construction | thread-invariant, and gated |
| output | values **+ per-feature provenance** (which kernel, which revision, which accumulator) | values only |

**Parity is a gate, not a hope.** For any plan both engines can serve, at the
same revision, every shared id must agree **bit-exactly** (`to_bits()`). This is
the same discipline `fold_engine_parity.rs` already applies between the fold and
buffered walks (11 tests, 18 geometries × {serial, rayon} × pools 1/2/3/8/16) —
generalized from "one pair of walks" to "any plan".

The research engine is not new code so much as a **named, complete** entry to
machinery that exists: the buffered walk, the `oracle` module's `Neumaier` and
`Exact` accumulators (`feature_v2.rs:18826`, already the standing regression
ruler), and the full toggle set. What it adds is (a) an entry that takes a
`Plan`, (b) per-feature provenance, (c) the revision selector.

---

## 6. The end-to-end contract

**A table** carries `feature_set_id` + `revision` in `_MANIFEST.json` (the id
half already landed).

**A bake** declares the FeatureSet it reads — **by id, not by position**. Today
a bake declares a *width* and its read set is inferred from layer-0 weights;
that inference (`block_profile::used_caller_lines`) is sound and stays, but a
bake baked after this lands carries the id explicitly and the inference becomes
the fallback for legacy bakes.

**The runtime** builds the plan from the bake and serves it, or refuses with the
specific missing slots. `PlanError::Uncomputable { missing: SlotSet }` replaces
`"bake declares more input features than the caller supplied"`.

**Every consumer refuses mismatches loudly** — already true via
`feature_set::check` and `--require-feature-set-match`; the plan makes the
refusal actionable by naming the slots.

**"Just works" concretely means:**

| operation | cost |
|---|---|
| add a feature | one registry entry (id = next free) + one kernel hook. No consumer changes. |
| retire a feature | `deprecated: true`. Id is never reused. No consumer changes. |
| revise a feature | append a `Revision` + a new era token. The migration table says which slots moved. |
| use a subset | `FeatureRequest { want, .. }`. The plan computes only that. |
| a new layout | one `Layout` declaration. |

**Servability becomes a registry property.** "Can the runtime serve this bake?"
is `Plan::derive(...).is_ok()` — a question with a checked answer, rather than
three hard-codings and a dead function.

---

## 7. Migration — zero numeric change

**The shipped bytes are the golden set.** Every existing artifact keeps its
numbers, and the gate is byte-identity, not agreement-within-tolerance.

| today | after | what moves |
|---|---|---|
| `w372` / `w720` / `w924` / `w944` / `w956` widths | five registered `Layout`s, identity mappings | nothing |
| `v1pre` / `v1cur` / `v1postc` / `pools` / `era2r4` … eras | registered revisions, per affected slot | nothing |
| shipped A / B / D / BHdr / C bakes | unchanged bytes; each gets a derived `FeatureRequest` | nothing — proven by the golden gates + a full `to_bits()` dump |
| stored tables | unchanged; resolve through the existing alias/roots tables | nothing |
| `ComputeSet::from_block_profile` | wrapper over `Plan::derive` | its result, gated identical |
| `score_pool_mode` | one axis of `Plan` | its result, gated identical |

The one **behavioural** change in increment 1 is strictly additive: a bake the
runtime **refused** may now be **served**. Nothing that scored before scores
differently — that is the byte-identity gate.

---

## 8. Performance

The plan makes named plans first-class, each with a measured cost:

| named plan | slots | what it is |
|---|---|---|
| `basic` | 156 | Profile D's actual read set |
| `basic+peaks` | 228 | D-class with the peak pool |
| `basic+peaks+moments` | 265 | the free-set arm — **the plan that is currently unreachable at runtime** |
| `basic+peaks+moments+classc` | 289 | + the class-C tranche |
| `basic+peaks+masked+iw` | 372 | the v1 set, shipped B's read set |
| everything | 944 / 956 | the research default |

Cost comes from `FeatureDef::cost` summed over the plan plus the kernel lane's
measured per-block map (`benchmarks/kernel_fastclass_2026-09-05.md`) — **cited,
never estimated**. Where a plan has no measurement it reports `UNMEASURED`
rather than a number.

Two standing perf constraints this design must not break, both measured and
recorded in CLAUDE.md:

* **The plan must be derived once and cached.** `from_block_profile`'s own doc
  records that a per-call uncached parse would regress the hot path it exists to
  speed up. `Plan` is derived where `cached_bake_metadata` already caches, which
  is once per bake-bytes pointer.
* **Never trade a byte for a cycle silently.** The `dense_block_kernel` note is
  the precedent: a 1.17× @8T upper bound is not worth re-extracting every table
  and re-training every model, and that trade is the user's to make, not a
  lane's.

---

## 9. What this design deliberately does not do

* **No renumbering.** Ever. Append-only, per the 2026-07-19 directive.
* **No new public API in increment 1.** The types land `pub(crate)` /
  `#[doc(hidden)]`; promoting any of them is a registered, separately-approved
  step (`cargo public-api` delta must be empty — `just api-doc-check`).
* **`zenanalyze-api` is not touched.** It stays frozen and is the naming
  substrate (`[a-z0-9_]+`, 8 lowercase hex, `fold_hash`) that both the
  feature-set id and the definition registry's names obey.
* **No stored artifact is rewritten.**
* **The research engine is not made fast** and the production engine is not made
  comprehensive. That is the point of having two.
