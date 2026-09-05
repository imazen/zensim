# Feature-set identifiers — stop naming feature sets by a count

**Status:** design + landed owner (2026-09-05). Registry:
[`benchmarks/feature_sets_registry.json`](../benchmarks/feature_sets_registry.json).
Owner code: `zensim::feature_set_id` (grammar + hash) and
`zensim_validate::feature_set` (bake/root derivation + the compatibility check).

**User directive (2026-09-05):** *"We need a better way to identify feature-sets
than by count — we need a good naming system."*

---

## 1. Why the counts fail — measured, this week, not hypothetically

A count answers one question (how wide is the vector?) and is then used to answer
three (which slots carry values? which extractor produced them? what does a model
consume?). Every one of these is a MEASURED failure of that conflation:

| # | The failure | Record |
|---|---|---|
| 1 | **"944" is a LAYOUT width holding ≥2 native extraction eras.** A grid census over the 97 fair cells found 78 on `dial_grid_944col_2026-08-01` (build `ec3bdd6a`), 9 on `…_POOLS_2026-08-30`, 7 on `…_foldapp2_2026-09-01`. `gFOLD2 == gPOOLS with f156..371 zeroed`; `g0801 vs gFOLD2` bit-identical except `f720..923` (max abs 7.2e-9). Three files, one number. | `benchmarks/dial_addressability_gate_2026-09-04.md` §15.3; `~/tmp/gaddrinst_RESUME.md` finding 1 |
| 2 | **`f156..371` is zeroed at some 944 roots and LIVE at others.** `bake_verdict` already has to ask the ROOT (`root_declared_regime` → `…pools` / `…carriers`) because the width cannot tell it. | `bake_verdict.rs` R1b guard, 2026-08-30 |
| 3 | **"156+free" = 156 basic + 109 free (72 peaks + 37 raw-moment) — and reads a 944-wide vector.** The name adds two numbers that live in three different blocks of a fourth number's layout. | `benchmarks/free_features_2026-09-01.md` §1 |
| 4 | **"+classC" adds 24 slots at EXISTING 944 positions.** Nothing about the width changes; the populated set does. A count cannot express it at all. | `benchmarks/free_features_classC_2026-09-04.md` §1 |
| 5 | **A bake has three different "widths".** `n_inputs` (internal, post-prune: 667) ≠ `caller_input_width` (declared: 944) ≠ the slots actually consumed (e.g. 28 of 156 for ADD156). Quoting any one of them as "the feature set" is wrong two ways out of three. | `zensim-validate/src/block_profile.rs` module doc; `benchmarks/dead_column_pruning_2026-08-04.md` |
| 6 | **The identity probe is the zero vector at 372 but NOT at 944**, and the 944 grids carry 39 references where the canonical 372 grid carries 38. Same word ("the identity probe"), different instrument. | `~/tmp/gaddrinst_RESUME.md` findings 2–3 |
| 7 | **`--regime 944` silently mis-scores a 372 bake that uses `f156..371`.** Shipped **B** reads CID22 **0.3862** at `--regime 944` against its true **0.8764** at `--regime 372` — a plausible-looking number, no warning. | `CLAUDE.md` Known Bugs; campaign appendix U.R0 / W |
| 8 | **Profile D was pinned to "156" and the free set sat unused**, because "156" named a byte-count of weights rather than a compute set that could have carried 109 more slots for ~0. | `benchmarks/profile_d_notax_2026-09-01.md`, `free_features_2026-09-01.md` |
| 9 | **The v1-372 `f0..155` is NOT the 944 fold's `f0..155`.** Measured on 4,424 shared dial cells: 156 of 156 slots differ, max abs 1.0214. "The basic block" names two different quantities. | `~/tmp/gaddrinst_RESUME.md` finding 3 |

The common shape: **a number encodes ONE of {which slots, which era, which
width}, and gets read as all three.**

---

## 2. The identifier

Four orthogonal parts. The first three are the human handle; the fourth is the
identity.

```
  <compute>@w<layout>/<era>#<slots-hash8>
  │          │          │     │
  │          │          │     └─ 8 lowercase hex — hash of the ORDERED slot-id list
  │          │          │        actually populated (or, for a consumer, read)
  │          │          └─ registered extractor ERA token  [a-z0-9_]+
  │          └─ LAYOUT: the vector width the runtime emits
  └─ COMPUTE: `+`-joined registered block tokens, in registry order
```

Example: `basic+peaks+moments@w944/era2r4#a1b2c3d4`

### 2.1 COMPUTE — which kernels/slot families actually run

A closed, registered vocabulary. Each token names a slot family whose presence
changes the POPULATED set (not merely a value). Canonical order is the registry's
order, which is layout order with the two scattered free tranches last.

| token | family | slots (4-scale layout) | zensim source of truth |
|---|---|---|---|
| `basic` | v1 basic fold | `f0..155` (156) | `ComputeSet::v1_basic` |
| `carriers` | the 10 `fused944native` carrier slots | 178,190,196,226,231,237,243,303,321,333 | `V1PoolsMode::Carriers` (`CARRIER_SLOTS`) |
| `peaks` | v1 soft-peak pool | `f156..227` (72) | `V1PoolsMode::Peaks` |
| `masked` | v1 masked pool | `f228..299` (72) | `V1PoolsMode::Full` |
| `iw` | v1 IW pool | `f300..371` (72) | `V1PoolsMode::Full` |
| `v2` | v2-348 dense block | `f372..719` (348) | `ComputeSet::v2_blocks` |
| `append` | append-204 | `f720..923` (204) | `ComputeSet::append` |
| `append2` | append2 / BANDVIS | `f924..943` (20) | `ComputeSet::append2` |
| `csfw` | CSFW tier-1 | `f944..955` (12) | `ComputeSet::csfw` |
| `moments` | the free raw-moment tranche | 37 scattered slots in append/append2 | `V1FreeExtras::RawMoments`, `free_slot_indices` |
| `classc` | the class-C bounded-error tranche | 24 scattered slots (12 v2 `MSE` + 12 Y luminance-bin) | `V1FreeExtras::RawMomentsPlusBoundedErr`, `class_c_slot_indices` |
| `hdr` | RESERVED — the future HDR block | (append-only, above `csfw`) | not yet emitted |

`masked` and `iw` are separate tokens even though `V1PoolsMode::Full` turns both
on together, because a bake can read one without the other and the check in §4 is
per-slot.

**Sub-toggles are NOT in the name, deliberately.** `gradient`, `blockiness`,
`transducer_bank`, `transducers_luma_only`, `append2_dst_activity` change which
slots *inside* a family are populated. They are captured by the **hash** (a
different populated set is a different hash, so they cannot collide) and by the
registry entry's `notes`. The name is a handle; the hash is the identity. Adding
five more tokens to the short form would make every 944 name unreadable and buy
nothing the hash does not already guarantee.

### 2.2 LAYOUT — the emitted width

`w<N>`, `N` = the width of the vector the runtime emits / the table stores /
the model declares as `caller_input_width()`. Registered widths today: `w372`,
`w720`, `w924`, `w944`, `w956`.

**LAYOUT is not COMPUTE.** A `v1_only` request at 944 is still a 944-wide row with
`f372..` at the structural `0.0`; `basic+peaks+moments@w944` and
`basic+peaks+masked+iw+v2+append+append2@w944` are the same width and different
data. That distinction is exactly what the counts erased.

### 2.3 ERA — which extractor produced it

A registered `[a-z0-9_]+` token, one per extractor era, each pinned in the
registry to its `build_commit`, its canonical root path, and the prose label
`zensim_validate::eval_roots::era_of` already prints. The era is load-bearing
because **the shift between eras is model-specific, not a constant offset**:
0.00000 SROCC for a basic-only bake, |0.489| for one that leans on the drifted
masked/IW block (`benchmarks/eval372_current_root_2026-08-30.md`). A number read
on one era cannot be corrected into the other, only re-verdicted.

Registered eras: `v1pre`, `v1cur`, `v1postc`, `ext720`, `ext924`, `ext944`,
`pools`, `era2r4`, `unknown`.

**`v1postc` is the era the SHIPPED runtime extracts** (2026-09-05 root, build
`4fbd8ff8`). Option C (`56bbcda2`, 2026-08-30 15:43) stopped v1 pooling phantom
columns, and the `v1cur` root was built at `ea16c7ee` two hours EARLIER — so
both older 372 roots are one era behind the product. Unlike the `v1pre -> v1cur`
step, this one moves `f0..227`, which every 372 bake reads; MEASURED on five
shipped bakes the RANK shift is nevertheless <= 6.8e-4 SROCC, while the DIAL
moves much more (`benchmarks/d_peaks_372_postC_2026-09-05.md` SS3).

### 2.4 `#<slots-hash8>` — the content hash

8 lowercase hex of a 32-bit hash of the **sorted, de-duplicated slot-id list**.

* Algorithm: FNV-1a/64 over the canonical decimal-with-commas rendering of the
  sorted slot list, folded to 32 bits by `NamedFeature::fold_hash`'s exact rule
  (`(h >> 32) ^ (h & 0xffff_ffff)`).
* **ONE owner:** `zensim::feature_set_id::slots_hash8`. Every producer and
  consumer calls it; nothing re-derives it. (This is the same discipline
  `zenanalyze`'s `feature_qualified_names` uses — a committed TSV rather than an
  off-Rust re-derivation — for the same reason: a silent hash mismatch is worse
  than no hash.)
* **Set semantics, not order semantics:** the input is canonicalised (sorted +
  deduped) first, so two producers that emit the same slots in different internal
  orders agree. Ordering *within* the vector is the LAYOUT's job, not the hash's.

### 2.5 Reusing the `zenanalyze-api` substrate, not paralleling it

The FROZEN `zenanalyze-api` contract already owns per-feature identity:
`NamedFeature` = `name@hex8`, name charset `[a-z0-9_]+`, hex 8 lowercase digits,
and `fold_hash(u64) -> u32` as the mandatory fold. This design **reuses all four
of those decisions verbatim** rather than inventing a second scheme:

| zenanalyze-api | feature-set id |
|---|---|
| `name@hex8` — one string is the identity | `<compute>@w<layout>/<era>#<hash8>` — one string is the identity |
| name charset `[a-z0-9_]+` | compute tokens and era tokens use the same charset |
| 8 lowercase hex, strictly validated | same, strictly validated (rejects uppercase, 7 or 9 digits, non-hex) |
| `fold_hash` = `(h>>32) ^ (h & 0xffffffff)` | the same function, applied to the slot-list hash |
| `Provenance` carries config/descriptor hashes *beside* the name | `era` carries the extractor build class *beside* the compute+layout |
| `Offer::satisfies(&Request)` — coverage, by name | `FeatureSetId` + `SlotSet::covers` — coverage, by slot | 

**Nothing in `zenanalyze-api` changes.** A feature-set id is a coarser handle
than a `NamedFeature` (a whole vector, not one column) and lives in zensim; the
API crate stays frozen. The relationship is: a zensim feature-set is one
producer's answer to what a `Request` for 944 columns would be, at a width and an
era `zenanalyze-api` deliberately does not model.

---

## 3. Worked ids

### 3.1 Every meaning "944" has had

**Seven** distinct feature sets have been called "944" in this repo. All seven
are on the board, in a landed benchmark, or on disk as a canonical root; all
seven now have distinct ids.

| # | id | what it actually is | `f156..371` | populated | where |
|---|---|---|---|--:|---|
| 1 | `basic+v2+append+append2@w944/ext944` | the SOTA-944 campaign root | **ZEROED** | 728 | `ext944-canonical-2026-08-01`, build `ec3bdd6a` |
| 2 | `basic+v2+append+append2@w944/era2r4` | the era-2 wave-r4 zeroed VIEW | **ZEROED** | 728 | `ext944-era2r4-2026-09-01/foldapp2_views/`, regime `folded720append2` |
| 3 | `basic+peaks+masked+iw+v2+append+append2@w944/era2r4` | the era-2 wave-r4 ROOT itself | **LIVE (216)** | 944 | `ext944-era2r4-2026-09-01/`, regime `folded720append2pools`, build `75c09149` |
| 4 | `basic+peaks+masked+iw+v2+append+append2@w944/pools` | the 2026-08-30 all-pools era | **LIVE (216)** | 944 | `dial_grid_944col_POOLS_2026-08-30`, `r1b/wlin7-pools944` |
| 5 | `basic+carriers+v2+append+append2@w944/pools` | `fused944native` — ten carrier slots live | **LIVE (10)** | 738 | `r1b-a3fused-2026-08-30`, regime `fused944native-carriers` |
| 6 | `basic+peaks+moments@w944/era2r4` | the free-set arm, a.k.a. **"156+free"** | ZEROED except peaks | 265 | `benchmarks/free_features_2026-09-01.md` |
| 7 | `basic+peaks+moments+classc@w944/era2r4` | free set **+ class C**, a.k.a. **"+classC"** | ZEROED except peaks | 289 | `benchmarks/free_features_classC_2026-09-04.md` |

Rows 1 and 2 differ ONLY in era — same compute, same width, same populated
slots — and are measured bit-identical everywhere except `f720..923` (max abs
7.2e-9). **Rows 2 and 3 are the same era and live in the same directory tree**:
`ext944-era2r4-2026-09-01/` is `folded720append2pools` and its own
`foldapp2_views/` subdir is the zeroed derivation, whose manifest already has
to say *"NEVER column-mix with the folded720append2pools tables these came
from"* in prose. Rows 4 and 6 share the same width and **zero** of their
non-basic slots. No count distinguishes any pair here; every id does.

One more shape the registry has to model: `wlin7b-2026-08-30`'s manifest
records its regime as *"mixed by file: `*_pools944` = folded720append2pools,
`*_ctrl944` = folded720append2"*. A root whose feature set varies per FILE
resolves to no id at all, and must be addressed per file.

### 3.2 The rest of the fair board's sets

| id | legacy name | notes |
|---|---|---|
| `basic+peaks+masked+iw@w372/v1cur` | "372" | the default eval root since 2026-08-30 (`ea16c7ee`) |
| `basic+peaks+masked+iw@w372/v1postc` | "372" | `2026-09-05-full-features-372-postC` (`4fbd8ff8`) — the RUNTIME era, post-option-C. NOT the default; that is a user decision |
| `basic+peaks+masked+iw@w372/v1pre` | "372" | the 2026-05-15 stored root — masked/IW from the thread-dependent window |
| `basic@w372/v1cur` | "156" | Profile **D** / **ADD156**-class: 372-wide caller, basic-only reads |
| `basic+peaks+masked+iw+v2@w720/ext720` | "720" | `ext720-canonical-2026-07-22` |
| `basic+v2+append@w924/ext924` | "924" | `ext924-canonical-2026-07-27`, build `0b3d16b0` |
| `basic+peaks+masked+iw+v2+append+append2+csfw@w956/unknown` | "956" | CSFW tier-1, default-OFF, no canonical root yet |

---

## 4. What the id is FOR — the compatibility check

Two id flavours, both the same type:

* a **producer** id (an extractor, a features root, a dial grid) whose slot set is
  what it POPULATES;
* a **consumer** id (a bake) whose slot set is what it READS
  (`block_profile`'s structurally-used caller lines).

A read is sound iff:

1. `layout_width` agrees (a consumer cannot read a column that is not there —
   this already fails loud today via `FeatureLenMismatch`), **and**
2. `consumer.slots ⊆ producer.slots` — every slot the bake reads is actually
   populated, **and**
3. `era` agrees, or the difference is explicitly acknowledged.

**(2) is bug #7.** Shipped **B** reads 49 of `f156..371`; the `ext944` root
populates none of them; so `B ⊄ ext944` and the 0.3862 is refused instead of
printed. Today's guard (`block_profile::folded_root_conflict` + the R1b
`root_declared_regime` probe) is the same check hand-specialised to one block at
one regime; the id generalises it to every block at every regime, which is what
lets it also catch e.g. a `moments`-reading bake pointed at a `basic`-only table.

**(3) is the era trap.** Widths and slot sets can agree perfectly while the values
came from a different extractor — rows 1 and 2 of §3.1 — and the resulting SROCC
delta is model-specific. Era mismatch is a WARNING by default (it is often
intentional: an A/B across eras is a legitimate experiment) and a REFUSAL under
`--require-feature-set-match`.

---

## 5. The registry

`benchmarks/feature_sets_registry.json` — **append-only**, `_schema` header,
four sections:

* `compute_tokens` — the closed vocabulary, each with its slot range/derivation
  and the zensim symbol that owns it.
* `eras` — token → `{build_commit, root, label, notes}`.
* `sets` — the registered worked ids of §3, each with `compute`, `layout`, `era`,
  `slots` (compact range string, e.g. `"0-155,720-923"`), `slots_hash8`, and
  provenance.
* `aliases` — every legacy count/name → the id(s) it has meant. An alias with
  more than one target is **ambiguous by construction** and consumers must say so
  rather than pick one.

Append-only, like the feature numbering itself: a set that is superseded gets a
`superseded_by` field, never a deletion or a renumber.

### 5.1 The alias table

| legacy name | resolves to | ambiguous? |
|---|---|---|
| `156` | `basic@w372/v1cur` | no |
| `156+free` | `basic+peaks+moments@w944/era2r4` | no |
| `+classC` | `basic+peaks+moments+classc@w944/era2r4` | no |
| `372` | `basic+peaks+masked+iw@w372/v1cur`, `…/v1pre` **or** `…/v1postc` | **YES — 3 eras** |
| `720` | `basic+peaks+masked+iw+v2@w720/ext720` | no |
| `924` | `basic+v2+append@w924/ext924` | no |
| `944` | six ids — §3.1 | **YES — 6 meanings** |
| `956` | `basic+peaks+masked+iw+v2+append+append2+csfw@w956/unknown` | no (no root yet) |

`--regime 372|720|944` keeps working unchanged; the flag is now understood as
"select this alias's DEFAULT target", and the resolved id is printed so the
choice is visible rather than assumed.

---

## 6. Ownership + plumbing

| what | owner | status |
|---|---|---|
| the grammar, the token vocabulary, `SlotSet`, the hash | `zensim::feature_set_id` | **LANDED** |
| `ComputeSet` → COMPUTE tokens + populated slots | `feature_v2::ComputeSet::{compute_parts, populated_slots, feature_set_id}` (`pub(crate)`) | **LANDED** |
| the extractor-side entry point | `feature_v2::V2NewFeatureToggles::{feature_set_id, populated_slots}` (`#[doc(hidden)] pub`) | **LANDED** |
| a bake's CONSUMER id from its bytes | `zensim_validate::feature_set::bake_feature_set_ref` | **LANDED** |
| a features root's PRODUCER id | `zensim_validate::feature_set::root_feature_set_ref` | **LANDED** |
| the per-caller-line read set | `block_profile::{caller_line_norms, used_caller_lines}` (`profile()` now calls them — one derivation) | **LANDED** |
| the compatibility check | `zensim_validate::feature_set::check` | **LANDED** |
| `bake_verdict` prints both ids, reports every disagreement | `bake_verdict` | **LANDED** |
| `bake_verdict --require-feature-set-match` refuses | `bake_verdict` | **LANDED** |
| `--full-json` `feature_set` block | `bake_verdict::feature_set_block` | **LANDED** |
| `bake_block_profile` prints the id + read set (text and `--json`) | `bake_block_profile` | **LANDED** |
| trainer embeds `zentrain.feature_set_id` | `zensim_mlp_train` | **LANDED** |
| board shows the id, badges an inferred one | `scripts/v_next/gauntlet.py` (`feature_set_id_of`, `fsid()`) | **LANDED** |
| extractor writes `feature_set_id` into `_MANIFEST.json` | `scripts/canonical_corpus/*` manifest writers | **REGISTERED, not executed** — the writers are another lane's; the helper they need (`V2NewFeatureToggles::feature_set_id`) is landed and the registry's `roots` table covers every existing root meanwhile |

**Measured on shipped B, reproducing three published numbers from three
different records, independently:** the derived consumer id is
`basic+peaks+masked+iw@w372/unknown#9403d2a7` over **95** live caller lines,
**49** of them in `f156..371` and **23** in `f228..371` — matching campaign
appendix W's "49 structurally-used lines", CLAUDE.md's "23 of its 95 live
inputs in f228..371". At the `ext944` root the gate names all 49 as
READ-BUT-NOT-POPULATED, which is the mechanism behind that read's CID22
**0.3862** against its true **0.8764**.

### 6.1 What every new artifact MUST carry

* **a bake** — `zentrain.feature_set_id` metadata: the id of the tables it was
  TRAINED on (the producer id), not a re-derivation of its own reads. Its own
  consumer id is always derivable from the bytes, so it is not stored.
* **a features root / parquet dir** — a top-level `"feature_set_id"` string in
  `_MANIFEST.json`, beside the existing `"regime"` prose (which stays: it is the
  human sentence, the id is the machine key).
* **a verdict** — `--full-json` gains `feature_set.{bake,root,verdict}`; the
  existing `features_root` block stays.
* **a dial/corruption grid** — same as a features root.

### 6.2 What is deliberately NOT changed

* `zenanalyze-api` — FROZEN, consumed only.
* `--regime N` — kept, mapped to an alias. Nothing that works today stops working.
* `ComputeSet` stays `pub(crate)` (`era2_perf_break_2026-08-31.md` §26.1's
  recorded decision: "No new public type, no new public entry point" for the
  perf path). The feature-set id is the public handle; the compute set is not.
* Feature NUMBERING — append-only, unchanged. An id names a subset of the
  existing numbering; it never renumbers.
* No stored table, verdict or bake is rewritten. Legacy artifacts resolve through
  the alias table and are badged `inferred`.

---

## 7. Migration

### 7.1 What keeps working, unchanged

* **`--regime 372|720|944`** — kept verbatim, same defaults, same presets. The
  flag now means "select this alias's DEFAULT target", and the resolved id is
  printed so the choice is visible rather than assumed.
* **`--cross-regime`** and the `folded_root_conflict` guard — untouched. The id
  check runs *beside* it, not instead of it: the guard refuses, the id check
  reports (or refuses under `--require-feature-set-match`).
* **`_MANIFEST.json` `"regime"`** — stays. It is the human sentence; the id is
  the machine key, and the registry's `regime_strings` table maps the former to
  the latter so no existing root has to be rewritten.
* **the board's `regime` column** — stays, as the compact label. Every
  historical cell has a width and almost none has an id, so removing it would
  blank 450 rows to prove a point. The id rides in the chip tooltip and is
  marked `(inferred)` whenever it was derived from the width.
* **`n_inputs` / `caller_input_width` / `zenanalyze-api`** — all unchanged.
* **Feature NUMBERING** — append-only, unchanged. An id names a SUBSET of the
  existing numbering; it never renumbers, and a new block extends the token
  vocabulary (`hdr` is already reserved) exactly as a new block extends the
  numbering.

### 7.2 What every new artifact MUST carry

| artifact | carries | who writes it |
|---|---|---|
| a **bake** | `zentrain.feature_set_id` metadata = the PRODUCER id of the tables it TRAINED on | `zensim_mlp_train` (landed; refuses to stamp when the training groups span different sets) |
| a **features root / parquet dir** | a top-level `"feature_set_id"` string in `_MANIFEST.json`, beside `"regime"` | the canonical-corpus manifest writers (registered; `V2NewFeatureToggles::feature_set_id` is the helper) |
| a **dial / corruption grid** | same as a features root | same |
| a **verdict** | the `feature_set` block in `--full-json` | `bake_verdict` (landed) |
| a **board row** | `feature_set.bake` flows through to the chip | `gauntlet.py` (landed) |

A bake's own CONSUMER id is never stored — it is always derivable from the
bytes, and a stored copy could disagree with them.

### 7.3 The three steps

1. **Now:** ids are derived and printed everywhere; mismatches are reported.
   Legacy artifacts resolve through the registry's `roots` / `regime_strings`
   tables and are marked `inferred`. **An inferred id is evidence about the
   artifact's NAME, never about its BYTES.**
2. **Next artifact:** every new bake / root / grid carries a real id per §7.2.
   An artifact with a stored id never falls back to inference.
3. **When the fleet next re-extracts:** the manifest writers stamp
   `feature_set_id` at write time, and `--require-feature-set-match` becomes
   `bake_verdict`'s default. It is not the default today for one honest reason:
   every existing bake reads era `unknown`, so strict mode would refuse the
   entire board — which is a true statement about the artifacts, not a useful
   one about a given run.

An `unknown` era is never silently treated as a match — it is reported exactly
like a mismatch, because "we do not know which extractor made this" and "we
know it was a different one" have the same consequence for a published number.
It is reported at `note` rather than `MISMATCH` prominence only so the one
honest fact common to all 450 legacy cells does not drown the ones specific to
a run; `--require-feature-set-match` treats it as a refusal like any other.

### 7.4 Adding a new set, era or token

* **A new set** — append an entry to `sets` with its `slots`, and let the
  registry test compute the hash for you (it fails with the owner's value).
* **A new era** — append to `eras` with the `build_commit` and root, and add
  the root to `roots`.
* **A new token** — append to `compute_tokens` AND to
  `zensim::feature_set_id::ComputeToken::ALL` (in layout order, scattered
  tranches last). Both are needed: the enum is the closed vocabulary the parser
  validates against, the registry is where the slots live.
* **Never** edit or delete an existing entry — supersede it with a new one and
  point `superseded_by` at the replacement, the same discipline
  `benchmarks/eval_annotations.json` uses.
