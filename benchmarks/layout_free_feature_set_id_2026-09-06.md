# The layout left the feature-set id

Increment **E (id form)** of [`../docs/PLAN_CRUFT_PURGE_2026-09-06.md`](../docs/PLAN_CRUFT_PURGE_2026-09-06.md),
pre-registered in that plan's §3 before it landed (commit `2234ec66`).
Design: [`../docs/FEATURE_SET_IDS.md`](../docs/FEATURE_SET_IDS.md) §2, §2.2, §4.

---

## 1. What changed

Canonical form is **`<compute>/<era>#<hash8>`**. `PartialEq` and `Hash` are over
compute + era + slots-hash, so two ids differing only in `@w<N>` are EQUAL and
interchangeable in every map and set. `Display` emits the layout only when the
id carries one. **Every `@w<N>` string ever written still parses**, to an id
equal to its layout-free spelling.

The width moves to `zensim_validate::feature_set::FeatureSetRef::layout`
(`Option<usize>`) — a property of the ARTIFACT (a bake's `caller_input_width()`,
a table's column count) rather than of the set.

## 2. Why — measured, not argued

The cruft purge's increment B-2 read a densified shipped `B` and its wide twin
as `basic+peaks+masked+iw@w95/unknown#9403d2a7` and
`…@w372/unknown#9403d2a7`: **the same compute tokens and the same slots hash**,
because they are the same read set. Only the wire differs. `feature_set::check`
duly reported a `LayoutDiffers` mismatch on every dense-bake/wide-table pair —
inside a REFUSAL surface — for a difference that cannot make a read unsound,
since a dense bake GATHERS its ids out of a wider row. That is the design, not a
defect.

**The end-to-end demonstration.** `bake_verdict --full-json` on shipped dense
`B` at the default 372 root, before vs after this change, over the whole JSON
tree: **every statistic bit-identical**, and exactly five differing fields, all
inside `feature_set`:

| field | before | after |
|---|---|---|
| `feature_set.bake` | `basic+peaks+masked+iw@w95/unknown#9403d2a7` | `basic+peaks+masked+iw/unknown#9403d2a7` |
| `mismatches[0].kind` | **`LayoutDiffers`** | `EraUnknown` |
| `mismatches[0].detail` | *"bake declares w95 but the table is w372"* | the era note |
| `mismatches[1]` | the era note | *(gone — there is only one mismatch now)* |

The spurious mismatch is gone; the legitimate one (an unestablished era) stays.

## 3. `check` now reports a SHORTFALL, not a difference

`LayoutDiffers` fires only when **both** widths are known **and** the consumer
needs a wider row than the producer emits. A NARROWER consumer is the dense
design. The unsound direction is the other one, and it is the only one worth a
refusal surface's attention.

## 4. The alias contract, held on the committed registry

`benchmarks/feature_sets_registry.json` is **not edited** — it is append-only and
every key in it is the legacy `@w<N>` spelling. `Registry` builds a layout-free
index of those keys at load time (and **refuses a collision**, rather than
last-writer-wins), so a caller holding a canonical id resolves to the
append-only entry. This is what makes the change a RENAME rather than a
migration: no published id needs rewriting.

`every_legacy_at_w_key_resolves_from_its_layout_free_spelling` holds it on the
real registry: for every legacy key, the layout-free spelling resolves to the
SAME entry, both strings parse to EQUAL ids with equal compute/era/hash, and
when both spellings reconstruct a slot set the sets are equal.

**Dropping the hint never loses information — measured.** The canonical form
searches `zensim::feature_set_id::registered_layout_widths()` where the legacy
form pins one candidate, so it can only ever reconstruct MORE. Exactly one
registry entry today reconstructs ONLY from its layout-free spelling:
`basic+peaks+moments@w944/era2r4#0b476506`, a CONSUMER set (the 265 free set
minus the four `LUMA_MEAN_REF` slots). Its `@w944` records the WIRE width of the
tables it reads, while its slot set is the family union clipped to **924** — so
the legacy spelling pins the wrong candidate and fails, and the canonical one
finds it. The test asserts that count is exactly 1; a second is a finding to
record, not a number to bump.

Most registry sets reconstruct from neither spelling, and that is by design:
they are PINNED slot lists (the carriers set's scattered slots, the consumer
read sets), which is why the registry stores `slots` at all. Reconstruction is a
convenience for the family-union sets; the pinned list is the truth for the rest.

## 5. The width is still worth recording

A PRODUCER should keep stamping it (`FeatureSetId::from_slots_with_layout`) — it
is the party that knows, and it turns a reader's search into one check.
`zensim::feature_v2::ComputeSet::feature_set_id` and `research`'s producer id
both do. `every_registered_layout_width_is_a_candidate` holds the candidate list
and the registry in sync, so registering a set at a new width fails a gate
instead of silently becoming unreproducible from its canonical id.

## 6. Public API

Breaking, all three pre-registered: `FeatureSetId::new` and `from_slots` lose
their `layout_width` parameter, and `layout_width()` returns `Option<usize>`.
Additive: `new_with_layout`, `from_slots_with_layout`, `with_layout`,
`layout_free`, and `#[doc(hidden)] registered_layout_widths()`. Batched under
`CHANGELOG.md` QUEUED BREAKING CHANGES for 0.3.0; **nothing is published**.

`cargo semver-checks --manifest-path zensim/Cargo.toml` reports *"no semver
update required"* — and that verdict is **weaker than it sounds and is recorded
as such**: the working version is already `0.3.0` against a `0.2.7` baseline, so
the tool classifies the comparison as a major change and SKIPS all 254 checks
(`0 pass, 254 skip`). It confirms the version bump is sufficient for whatever
changed; it did not evaluate this change.
