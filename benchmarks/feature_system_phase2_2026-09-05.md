# Feature system, phase 2 — the RESEARCH engine and per-feature provenance

**Date:** 2026-09-05. **Lane:** `zensim--featsys2`.
**Design:** [`../docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md`](../docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md) §5.3.
**Plan + pre-registered gates:** [`../docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`](../docs/PLAN_FEATURE_SYSTEM_2026-09-05.md) phase 2.

User directive this serves (verbatim, 2026-09-05): *"a comprehensive
all-feature version for research, and an optimized version with maximum
performance for what is key, and a software shape and contract that just works
end to end for any feature subset or feature revision"* + *"also make sure
everything can be served"*.

---

## 1. What landed

`zensim::research` (`#[doc(hidden)] pub`, gated on `feature-regime-v2`):

* `Request` — `everything()`, `for_slots(want, width)`, `for_set(&FeatureSetId)`,
  `for_bake_bytes(&[u8])`, plus `.at_revision()`, `.with_era_label()`,
  `.with_parallel()`, and `.validate()` (plan + revision check with **no
  image**).
* `extract(&Request, src, dst) -> Result<Extraction, ResearchError>`.
* `Extraction` — `values()`, `provenance()`, `emitted()`, `feature_set_id()`,
  `layout_width()`, `build_commit()`, `manifest_json()`.
* `FeatureProvenance` — per emitted position: `id`, `name`, `family`, `scale`,
  `channel`, `statistic`, per-slot `cost`, `tranche`, `form`, `direction`,
  `kernel`, `revision_era` + `revision_commit`, `proposed_revision`, `defect`,
  `deprecated`, and **`populated`** (did the plan compute it, or is the value
  the layout's structural zero?).
* `RevisionRef::{Current, Named}` + `ResearchError::{RevisionUnavailable,
  RevisionUnregistered, Plan, Compute, UnreadableBake}` — every refusal names
  the slots, and the selector consults the revision lane's own arithmetic
  switch (§4c) rather than assuming the shipped default.

`zensim/examples/v2_ab_extract.rs` gained `ZENSIM_AB_MODE=research`
(env knobs `ZENSIM_RESEARCH_{SET,WIDTH,ERA,REVISION}`), which writes the CSV
**and** a `_MANIFEST.json` carrying the whole provenance table plus the
producer `feature_set_id`, the emitted slot set, and both build-commit
readings. It is an EXTENSION of that extractor — same pairs TSV, same `zen_io`
decode, same grouped ref-reuse, same NO-GRACEFUL-SKIPS abort — not a fork.

---

## 2. The two places the design's sketch did not survive the code

The design called the research engine *"buffered / oracle-backed, reference
semantics"*. Two of those three words are wrong, and the reasons are
measurements rather than preferences.

**It cannot be the buffered walk.** `streaming::compute_multiscale_stats_
streaming` — the BUFFERED path, whose name says "streaming" for an unrelated
reason (CLAUDE.md's naming trap) — has no `V2NewFeatureToggles` parameter and
contains **zero** occurrences of `append_block` or `csfw_block`. It is
structurally v1-only: 372 of the 956 registered slots. A research engine that
cannot compute two thirds of the registry is not comprehensive, so the
research engine drives the **fold**, which is the machinery that already
computes every registered block.

**It cannot be oracle-backed by default.** `feature_v2::oracle`'s `Neumaier`
(Kahan-Babuška) and `Exact` (Shewchuk expansion) accumulators produce
*different bits* from the production reduction — that is their entire purpose
as the standing precision ruler. Making them the research engine's arithmetic
would make **G2.1 unsatisfiable by construction**. They stay where they are,
gated separately by `era2_oracle_bounds_hold_for_every_pool_shape`.

What is left is the honest and, it turns out, the strong version: the research
engine is the **same walk**, reached through a `Plan` instead of a hand-built
toggle struct, carrying a manifest. Parity is then a property of the code
rather than of two implementations kept in step by hand.

---

## 3. Gate results

Run under `--features training,feature-regime-v2,threads` (`training` is NOT a
default feature — a plain `cargo test -p zensim` compiles none of these files
and reports a green run that proved nothing).

### G2.1 — engine parity, bit-exact on every shared id

`zensim/tests/research_engine_parity.rs`, over the shared 20-cell geometry
matrix (`common::parity_cells::CELLS`, the same set `fold_engine_parity.rs`
uses — by reference, not retyped):

| test | comparison | bit comparisons | result |
|---|---|--:|---|
| `..._at_the_v1_layout` | research `0..372@372` vs `Zensim::compute_extended_features` | 7,440 | PASS |
| `..._at_the_944_layout` | research `0..944@944` vs `compute_folded720_append_features_streaming` (`append2_block`, `v1_pools: Full`) | 18,880 | PASS |
| `research_everything_agrees_with_the_production_csfw_walk` | research `everything` vs the same walk `+ csfw_block` | 19,120 | PASS |

**Real-corpus confirmation, stronger than the synthetic gates:** 60 CID22
validation pairs (512×512, real PNG/JPEG decode through `zen_io`), extracted
twice through the same binary —

```
ZENSIM_AB_MODE=foldapp2pools                                → cost_P944.csv
ZENSIM_AB_MODE=research ZENSIM_RESEARCH_SET=everything
                        ZENSIM_RESEARCH_WIDTH=944           → cost_R944.csv
```

`sha256 = 253d864cfdb7421c3e8e7e8ee9d60d71918004542df0df781244c80aa574ad7c`
for **both** (1,119,084 bytes; `cmp` clean). Artifacts:
`/mnt/v/output/zensim/featsys-2026-09-05/`.

### G2.2 — thread invariance

`research_output_is_thread_invariant`: `Request::everything()` with
`with_parallel(true)` is bit-identical across rayon pool sizes **1/2/3/8/16**
and equal to the serial answer, at all **24** pool-sweep cells. Same standard
`v1_feature_width_pure_function.rs` holds the v1 extractor to.

### G2.3 — completeness

`everything_covers_the_whole_registry`: emits **956** values and 956
provenance rows, `emitted()` covers `0..956`, and **no** position falls
outside the registry (zero `unregistered_*` rows). Per-family census from the
smoke manifest:

| family | slots | | kernel | slots |
|---|--:|---|---|--:|
| v2 | 348 | | v2_dense | 288 |
| append | 204 | | v1_fused | 156 |
| basic | 156 | | v1_mask_iw | 144 |
| peaks / masked / iw | 72 each | | append | 132 |
| append2 | 20 | | v1_peaks | 72 |
| csfw | 12 | | v2_gradient | 60 |
| | | | free_raw_moments | 40 |
| | | | free_bounded_err | 36 |
| | | | append2 / csfw | 16 / 12 |

Cost: 523 `expensive`, 372 `cheap`, 61 `free`. Form: 916 `difference`, 28
`reference_only`, 12 `similarity`. Defects carried: **F4** 132 slots, **F5**
40, **F15** 12. Proposed revisions: `v1ssimcap` 132, `freecomp` 36. Revisions:
`v1postc` 372, `base` 584. *(F4 reads 132, not the audit's original 72 — the
revision lane corrected its blast radius by measurement in `8078830b`, and the
provenance reads the corrected registry rather than a copied number.)*

### G2.4 — provenance is CHECKED, and it found a defect

`dropping_a_family_perturbs_only_its_own_slots`, 5 families × 20 cells: every
position the narrowed plan still populates is **bit-identical** to the full
walk's, and every position it does not populate carries its declared
structural fill (`0.0`, or `1.0` on the twelve F15 slots — §4b).
`every_populated_slot_names_a_kernel_its_plan_runs` additionally pins that the
cheap free-set plan's 265 populated slots name only `v1_fused` / `v1_peaks` /
`free_raw_moments` — never the `append` kernel that owns the block those slots
LIVE in — and that every tranche slot's per-slot cost reads `free`.

---

## 4. THE DEFECT THE PROBE FOUND

**`Plan` described walks that cannot exist.**

`V2NewFeatureToggles` has exactly ONE layout/compute separation: `v1_only`,
which turns every v2-era kernel off while leaving the declared width alone.
There is **no per-block layout-only flag** —
`ComputeSet::from_toggles` (`feature_v2.rs:1907`) derives

```rust
let append  = t.append_block  && v2_blocks;
let append2 = t.append2_block && v2_blocks;
csfw:         t.csfw_block    && v2_blocks,
v1_basic:     true,                        // no toggle can turn it off
```

from the *same* `*_block` flags that decide the WIDTH. `Plan::toggles()` set
those flags from `LayoutBlocks::for_width`, so a plan that said *"compute the
append block but not CSFW, at layout 956"* emitted `csfw_block: true`, the
walk computed CSFW, and `plan.emit` — derived from the **un-normalized**
request — declared those twelve positions structural zeros.

MEASURED: `f944` (`csfw_w_global_dmean_s0_s`) came back at **0.06778112971292545**
on a plan that reported it unpopulated, at 1153×72.

**Fix:** a fixed point rather than a second rule. `Plan::normalized` resolves
`compute` *through the toggles the plan itself would emit*, so

```
compute == ComputeSet::from_toggles(plan.toggles())
```

**by construction**, and `emit` follows from that. Applied in all four
constructors (`derive`, `for_bake`, `v1`, `union`). Gates:
`normalization_is_a_fixed_point` (8 cases including the one that found it:
a 956 layout whose request deliberately skips the top block) and
`a_wide_layout_computes_every_block_it_reaches` — a **negative** gate that
pins the limitation so the day a per-block layout-only flag lands, it fails
and forces the plan to stop over-claiming.

**Blast radius: none.** `emit` only ever WIDENS, so no request that planned
before stops planning, and the servability census (`every_shipped_profile_is_
servable`, `every_registered_producer_set_is_plannable`, `the_campaign_free_
set_arms_plan_to_the_cheap_walk`) passes unchanged. No shipped bake produces a
partial-v2 plan — every shipped plan is all-or-nothing, which is exactly why
this sat undetected.

The missing capability — a per-block layout-only flag, so a 956-wide vector
could carry a computed append block beside a zeroed CSFW one — is
**REGISTERED, not built**: it needs a walk change and this lane's scope is
dispatch.

---

## 4b. A SECOND measured finding: the structural fill is not a constant

`ZensimV2Result`'s own doc says *"a v1-only 944 request is still a 944 row
with `f372..` at the structural 0.0"*. MEASURED at 64×64, `v1_only` +
`append2_block`, pools `Off`: of the **572** positions the walk leaves alone,
**560 are exactly `0.0` and twelve are exactly `1.0`** —

```
f393 f422 f451 f480 f509 f538 f567 f596 f625 f654 f683 f712   = 1.0
```

— one per (scale, channel) cell, and they are precisely
`v2_pjnd_fragility_*`, i.e. **exactly the twelve slots the defect audit
already tagged F15** (*"`PJND_FRAGILITY` is nonzero on an identity pair"*).
Same finaliser, second place: it returns `1.0` for its degenerate no-samples
case and runs whether or not the kernel that fills its accumulators did.

Not fixed here — kernel arithmetic is the revision lane's, and a fix moves
shipped bytes. What landed instead is
`research::nonzero_structural_fill_slots()`, **derived from the registry's
defect field** rather than hard-coded as twelve indices (so it follows the fix
when one lands), plus a gate that pins BOTH halves:
`the_structural_fill_value_is_zero_except_on_the_f15_slots` fails if a new
position becomes nonzero **or** if an F15 slot starts reading zero.

The general lesson, worth carrying: **"unpopulated" and "zero" are different
claims.** `Plan::emit` is the authority on which positions carry a computed
number; the byte at an unpopulated position is whatever its finaliser's
degenerate branch returns, which is a per-signal property.

---

## 4c. The two revision selectors COMPOSE — measured end to end

This lane owns the REQUEST-side selector (*which era is a caller allowed to
be told it got?*); the revision lane owns the ARITHMETIC selector
(`ssim_form::active_revision`, pinnable per process with
`ZENSIM_FORMULA_REV=1|2`, landed in `8078830b`). They meet in
`research::check_revision`, which asks the arithmetic switch rather than
assuming the shipped default.

**Under the shipped revision (Rev1)** a `v1ssimcap` request is REFUSED, and the
refusal names **exactly** the slots
`feature_defs::FormulaRevision::Rev2::moved_slots` declares — gated by
`the_refusal_names_exactly_the_registrys_moved_slots`, which compares against
that owner rather than a copied list. 132 slots at the full `0..372`; **36**
for a `basic`-only request, which is what the CLI demonstration below prints.

**Under `ZENSIM_FORMULA_REV=2`** the identical request SUCCEEDS, with no change
in this module — that is the point of the two selectors being separate.

**And the declared blast radius is the measured one.** Two extractions of the
same four pairs at `basic+peaks+masked+iw@372`, one per pinned revision:

| block | columns moved |
|---|--:|
| basic (`f0..156`) | 36 |
| peaks (`f156..228`) | 24 |
| masked (`f228..300`) | 36 |
| iw (`f300..372`) | 36 |
| **total** | **132** |

**132 measured = 132 declared**, max `|Δ|` 3.36e-1. And the manifest reports
which arithmetic ran, per table:

```json
"revision": "v1ssimcap",              // what was ASKED for
"formula_revision": "Rev2",           // what actually RAN
"formula_revision_eras": ["v1ssimcap", "freecomp"]
```

with the per-column `revision` field moving `v1postc → v1ssimcap` on exactly
the 132 affected columns. **A table can no longer be produced under a pinned
arithmetic era without saying so.**

The compatibility rule this needed is stated once, in
`research::signal_matches_era`, and the first draft got it wrong in a way
worth recording: it asked *"is this slot's era equal to the requested one?"*,
which refused **156 of 156** basic slots for `v1ssimcap` when only 36 carry
it, and still refused 120 unaffected slots under `ZENSIM_FORMULA_REV=2`. **An
era is a boundary in TIME, not a label every slot must wear** — a signal no
revision `x` touches computes the same quantity in `x`'s world as in any
other, so it is compatible with `x`. Corrected rule: *effective era equals
`x`, OR no revision entry of this signal names `x`.*

---

## 5. Cost — reported, not gated

The plan pre-registered *"cost is NOT a gate for the research engine but must
be reported"*. One binary, arms interleaved, 3 reps, min per arm, 60 CID22
pairs at 512×512, compute-only µs/pair (the extractor's own accumulator, net
of decode):

| arm | rep1 | rep2 | rep3 | **min** | vs production |
|---|--:|--:|--:|--:|--:|
| production `foldapp2pools` (944, pools Full) | 56.4 | 55.7 | 60.0 | **55.7** | — |
| research `everything@944` | 57.8 | 59.7 | 55.8 | **55.8** | **+0.2 %** |
| research `everything@956` | 59.7 | 59.8 | 58.0 | **58.0** | +4.1 % |
| research `basic+peaks+masked+iw@372` | 28.5 | 27.5 | 28.4 | **27.5** | **0.49×** |

The 944 arms differ by 0.1 ms against a 4.3 ms run-to-run spread within each
arm — **the named entry costs nothing measurable**. The +4.1 % at 956 is the
CSFW block's own work, not overhead. The 372 arm at 0.49× is the point of the
plan: a narrower request really does skip the work.

Log: `/mnt/v/output/zensim/featsys-2026-09-05/cost_run.log`.

---

## 6. Refusals, verified end to end

Every refusal is loud, names what it refused, and — since `Request::validate()`
— fires **once before any image is decoded**. (The first draft checked
per-pair and printed a 6 KB message listing all 372 slots' eras, four times on
a 4-pair smoke run; on a real corpus that is 200,000 messages, each after
paying for a decode.)

```
$ ZENSIM_AB_MODE=research ZENSIM_RESEARCH_REVISION=v1ssimcap …
ABORT: research: revision 'v1ssimcap' (registered) is not what this build
computes for 372 slot(s) — 0-371 compute(s) v1postc instead. No era other than
the current one can be reproduced until it is registered as a LANDED revision
(a Proposed one is a priced design, not an implementation).

$ … ZENSIM_RESEARCH_REVISION=nonsense …
ABORT: … revision 'nonsense' (NOT a registered era token — check the spelling) …

$ … ZENSIM_RESEARCH_SET=bogus …
ABORT: ZENSIM_RESEARCH_SET names "bogus", which is not a registered compute
token. Registered: basic, carriers, peaks, masked, iw, v2, append, append2,
csfw, moments, classc, hdr

$ … ZENSIM_RESEARCH_REVISION=v1postc …        # the CURRENT era, by name
wrote 4 rows x 372 features … (bit-identical to RevisionRef::Current)
```

The revision rule, stated once: a `Named(x)` request is servable iff every
requested slot's latest **LANDED** revision is `x`, or the slot has never been
revised. Reproducing a superseded era needs the superseded code, which this
build does not have — so it refuses instead of quietly serving current values
under an old name. When the REV2 lane registers a landed `rev2`, the same
selector follows it with no change here.

---

## 7. Duplication removed in the same pass

The parity **geometry matrix** now has ONE owner,
`zensim/tests/common/parity_cells.rs`. It was a `const` inside
`fold_engine_parity.rs`, and its four-cell pool-sweep extension
(`[(256,256), (96,320), (320,96), (577,385)]`) was written out identically at
**three** call sites with nothing checking the three stayed equal. A second
parity suite on its own hand-picked width list is precisely the drift the
no-duplicate rule exists to stop — both suites stay green while the union of
what they cover quietly shrinks.

## 8. Stale doc claims corrected in place (from the inventory sweep)

* `fold_engine_parity.rs` is **14 tests over 20 cells** (24 in the three
  pool-sweeping tests), not "11 tests over 18 geometries" — CLAUDE.md was
  stale on all three counts.
* **`folded_v1_only_matches_full_walk` has never existed in the tree.** Three
  live doc comments in `feature_v2.rs` (plus CLAUDE.md) cited it; the real
  cousin is `free_extras_are_pure_addition_to_the_v1_only_walk`
  (`feature_v2.rs:14942`). Fixed at the live sites; left in the historical
  campaign records, which are records.
* `pyramid_stride_has_no_phantom_columns` lives in `feature_v2.rs`, not
  `blur.rs`.
* `v1_golden_bytes.rs`'s determinism test says "Runs BOTH fixtures" and runs
  **three** (64×64 tight, 200×150 non-tight, and the real PNG pair).
* `training` is **not** a default feature, so a plain `cargo test -p zensim`
  compiles none of the three primary byte gates. Every gate invocation in this
  record names its feature set.
