# Feature system, phase 5 — consumer migration, and revision as a PER-BAKE declaration

**Date:** 2026-09-05. **Lane:** `zensim--featsys2`.
**Plan + pre-registered gates:** [`../docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`](../docs/PLAN_FEATURE_SYSTEM_2026-09-05.md) phase 5.
**Predecessors:** [`feature_system_phase2_2026-09-05.md`](feature_system_phase2_2026-09-05.md),
[`feature_system_phase4_2026-09-05.md`](feature_system_phase4_2026-09-05.md).
**Depends on:** the revision lane's `feature_rev2_2026-09-05.md` (F4/F5, `FormulaRevision`).

---

## 1. The requirement that changed the shape of this phase

The revision lane's F5 fix is **not free**. `bake_block_profile` shows all
three shipped 944 bakes — `c_sdr_mlp944_corrmix`, `c_hdr_l1t1944`,
`c_sdr_purity944` — reading the full `GLOBAL_DMEAN`/`CGAIN`/`CLOSS` set, 33
slots each. A GLOBAL flip of `SHIPPED_REVISION` to `Rev2` would move **22 of
33 inputs per bake** and silently re-price every one of them, so `Rev2` could
not ship until Profile C was refit.

That is the wrong shape for a contract the user asked to *"just work end to end
for any feature subset or feature revision"*. **Revision is therefore a
PER-BAKE declaration**: a bake declares the revision it was trained against
(`zentrain.formula_revision`; absent means `Rev1`, the registry default), the
plan computes the declared revision, and two bakes at different revisions
coexist in one process.

---

## 2. What landed

* **`V2NewFeatureToggles::formula_revision`** and `ComputeSet::formula_revision`
  — the revision travels with the REQUEST. Defaults to
  `ssim_form::active_revision()`, so every existing construction is
  byte-identical and `ZENSIM_FORMULA_REV` keeps working as the whole-process
  pin.
* **`feature_v2::bake_formula_revision(&Model)`** — the per-bake reader. An
  unrecognised value falls back to the shipped revision and the caller reports
  it; it is never guessed at silently.
* **`Plan::formula_revision()` / `Plan::revisions_agree()`**, and
  `fold_engine::score_plan` returns NO plan for a profile whose bakes disagree
  — one walk computes one arithmetic era, and unioning them would serve one
  bake the other's arithmetic.
* **`bake_verdict` REFUSES a revision mismatch** between the bake's
  declaration and the build's active revision, naming both.
* **`bake_verdict` refuses `SlotsNotPopulated` by default** when BOTH
  feature-set ids are STORED (not inferred) — see §4.
* **`--regime N` prints its DERIVED meaning**, resolved through the default
  root's own `_MANIFEST.json` regime and the registry, rather than hard-coded.

---

## 3. Gate results

### G5.1 — `--regime N` resolves to the same set it does today

A fixed bake (`b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07`) on a
fixed root (the default 372 root), before and after the whole phase:

```
diff <(grep -v '^Wall time:' verdict_before.md) \
     <(grep -v '^Wall time:' verdict_after3.md)   →  no output
```

**BYTE-IDENTICAL apart from the wall-time line** (0.28 s vs 0.26 s), which is
this repo's standing convention for a verdict comparison. Re-checked after the
per-bake revision work as well.

The flag now also says what it MEANS, derived rather than asserted:

```
bake_verdict: --regime 372 means regime v1-372 = basic+peaks+masked+iw@w372
  (372 slots, #d16a1091) — derived from
  /mnt/v/zen/zensim-training/2026-08-30-full-features-372's manifest, not hard-coded
```

and when it cannot be derived it says so instead of guessing:

```
bake_verdict: --regime 720 — its default root declares regime <none> and the
  registry has no entry for it, so the flag's meaning in COLUMNS is NOT
  established (never read that as a match)
```

*(The 372 root's manifest declares `v1-372 (extended+iw, num_scales=4, …)` —
the convention is `<registry key> (human prose)`, so the leading token is the
key. Both forms are tried rather than assumed.)*

### G5.2 — the `--regime 944` silent-mis-scoring bug

**Already structurally blocked**, and now blocked twice. Reproduced against the
recorded instance:

```
$ bake_verdict --bake b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin --regime 944
bake_verdict: REFUSING wrong-regime read: bake structurally uses 49 caller
  line(s) in f156-371, a block this folded root feeds as STRUCTURAL ZEROS
```

That is the pre-existing `folded_root_conflict` guard, which is this check
hand-specialised to ONE block at ONE regime. The general form —
`feature_set::check`'s `SlotsNotPopulated`, which covers **every block at
every regime** — now REFUSES by default too, with one deliberate restriction
(§4).

### Per-bake revision — the coexistence property

`zensim/tests/per_bake_revision.rs`, 2 tests:

* `two_revisions_coexist_in_one_process_and_differ_only_where_the_revision_moves`
  — two 944 extractions at two declared revisions, in ONE process, at 3
  geometries. They differ, every difference is inside the append block's
  `GLOBAL_*` set, and the DEFAULT toggles reproduce `Rev1` bit-for-bit so
  nothing that scores today changes.
* `the_revision_moves_only_the_global_contrast_pair` — **MEASURED blast
  radius: 11 slots**, all `GLOBAL_CGAIN` or `GLOBAL_CLOSS`, and
  **`GLOBAL_DMEAN` never moves**. Narrower than the name
  `paired_global_contrast` suggests, and correct: the fix is to the paired
  CONTRAST estimate, not the mean. Pinned in both directions.
* `mixed_revision_profiles_get_no_plan` (in `feature_plan`) — every shipped
  profile is single-revision, AND `revisions_agree` actually distinguishes,
  so the refusal in `score_plan` cannot be vacuous.

`bake_verdict` end to end:

```
$ ZENSIM_FORMULA_REV=2 bake_verdict --bake <a Rev1 bake> …
bake_verdict: REFUSING — … declares formula revision Rev1 and this build
  computes Rev2. The arithmetic differs on the slots that revision moves, so
  every number would be about a formula the bake was not trained on.
```

**A method note worth keeping.** The first draft of the blast-radius test
pinned the three `GLOBAL_*` slots at (scale 0, Y) and FAILED — not because the
revision was inert but because on that fixture `GLOBAL_CGAIN` at that cell is
exactly `0.0` in both revisions. A per-cell assertion about a value that can
legitimately be zero is a fixture-dependent test wearing a correctness test's
clothes. The gate now asserts over the whole vector.

---

## 4. The one deliberate restriction on the new refusal

`SlotsNotPopulated` refuses **only when BOTH feature-set ids are STORED**.

`FEATURE_SET_IDS.md` §2.3 is explicit that an INFERRED id is "evidence about
the artifact's NAME, never about its BYTES". Most roots on disk carry no
recorded id — the default 372 root's is inferred from the registry's root
table — so refusing on an inferred id would be refusing on a guess, and would
break every workflow that scores against them. When either side is inferred
this stays a report, and `--require-feature-set-match` escalates it exactly as
before. `--allow-unpopulated-slots` is the opt-out, mirroring
`--cross-regime`.

The pre-existing `folded_root_conflict` guard is KEPT rather than replaced: it
fires on inferred roots too, where the general check deliberately will not.

---

## 4b. The public-API delta, registered

The plan's **G-API** says "zero public-API delta — a phase that needs public
surface stops and registers it". This phase needs one line of it, so here it
is rather than buried:

`V2NewFeatureToggles` gains a public field, `formula_revision:
FormulaRevision`. `docs/public-api/zensim.txt` goes 538 → 539 lines. It is
ADDITIVE, and it has to be public because the struct is the `pub` request type
callers construct — a `pub(crate)` field would make the struct
unconstructible outside the crate entirely.

The struct is not `#[non_exhaustive]`, so an EXHAUSTIVE external struct
literal would break. Every in-tree caller (13 examples and benches) already
uses `..V2NewFeatureToggles::default()`, and the same was true of every field
this struct has previously gained — `append_block`, `append2_block`,
`csfw_block`, `free_extras`, `v1_pools`. So the established contract is
"construct with `..default()`", and this follows it. `FormulaRevision` itself
reaches consumers only through a `#[doc(hidden)]` re-export, so it is on the
internal surface.

Everything else in phases 2, 4 and 5 is `pub(crate)` or `#[doc(hidden)]`.

---

## 5. What is NOT done, stated plainly

**The luma-form half of `Rev2` is still process-global.** `Rev2` has two
halves. `paired_global_contrast` is a FINALISER parameter, so it threads
per-walk for free — and it is the half the shipped 944 bakes actually read,
which is why per-bake revision is useful today. `ssim_form::active_luma_form`
is a `OnceLock` read inside the SIMD kernels; making THAT per-request is a
change to the kernel dispatch, which is the revision lane's. So a per-bake
`Rev2` today gets rev2's global-contrast arithmetic and the process's luma
form. Stated in the toggle's own doc rather than papered over.

**`ComputeSet::from_block_profile` is not yet collapsed into
`Plan::derive_with_layout`.** The evidence for doing so landed in phase 4
(`from_block_profile_agrees_with_the_id_space_derivation`, which holds across
every shipped bake), and phase 4 already routes non-identity layouts through
the id-space derivation. Identity layouts still use `from_block_profile`
because it is the tested derivation and keeping it keeps the 445-bake census
on the SAME code rather than an equivalent one. Collapsing it — and thereby
retiring `fold_engine::wide_bake_v2_read`, which exists only to serve its wide
branch — needs the census re-run as evidence, not just the unit gate. **Not
done, and not claimed.**

**The `None`-plan shortcut in `compute_folded_v1_372_streaming_impl` stays.**
Its literal `V1PoolsMode::Full, v1_only: true, ..default()` is not a duplicate
to delete: phase 1 chose it deliberately so the narrow non-skipping case stays
on the *identical* code path rather than an equivalent one, which is a
stronger guarantee than equal output. Replacing it with `Plan::v1(...)` would
trade that guarantee for three fewer lines.
