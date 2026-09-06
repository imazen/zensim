# `--regime` becomes a derived, printed value — and re-arming the guard it broke

Increment **F** of [`../docs/PLAN_CRUFT_PURGE_2026-09-06.md`](../docs/PLAN_CRUFT_PURGE_2026-09-06.md)
(gates F.1, F.2). Owner: `zensim_validate::feature_set::derive_regime`.

---

## 1. The width was always a fact about the bake

`--regime N` names a vector width. The bake knows its own read set — literally,
since 2026-09-06, for the four shipped profiles that declare their ids — so the
narrowest registered layout that can carry it is **derivable**, and a derived
value cannot be typed wrong.

`bake_verdict` now prints it on every run:

```
bake_verdict: regime — DERIVED 372 (reads 95 ids, highest f369, incl. the f156..371 pool block); effective 372
bake_verdict: regime — DERIVED 944 (reads 667 ids, highest f941); effective 944
```

Candidates come from `zensim::feature_set_id::registered_layout_widths()`, so
the derivation cannot invent a width the registry does not know
(`the_derived_regime_is_always_a_registered_width`).

**The flag stays accepted**, because it also picks the corpora list and the
dial/corruption grids — which the bake alone cannot decide. It is deprecated
*as a width selector*, says so in `--help` and in a note on every run that
passes it, and a value that disagrees with the derivation is **refused rather
than obeyed**.

## 2. What is now refused, measured on the shipped artifacts

| invocation | before | now |
|---|---|---|
| `--bake <944 bake>` (bare) | scored at the **372** root by default | **REFUSED**: *"--regime 372 cannot carry this bake: it reads f941 and 372 stops at f371"* |
| shipped `B` `--regime 944` | refused (the pre-existing 944 guard) | refused |
| shipped `B` `--regime 720` | **allowed** — nothing checked 720 | **REFUSED**: 720 zeroes `f156..371` and `B` reads 49 lines there |
| `D` (basic-only) `--regime 720` | allowed | allowed — the board's ext720 era rows still work |
| `B` at a `…pools` root | allowed | allowed — that root has `f156..371` LIVE |
| `--regime 720` **with an explicit `--features-root`** | allowed | allowed |

Two asymmetries are deliberate and both are measured, not assumed:

* **A wider regime is not automatically wrong.** The board's era rows read
  372-class bakes at the ext720 root on purpose. Only a FOLDED regime plus a
  bake that reads `f156..371` is unsound, because that fold zeroes the block.
* **`--regime`'s number stops meaning a width once `--features-root` is
  explicit.** The frozen as-run LOO drivers
  (`scripts/external_reads/asrun/…/verdict_parallel.sh`) pass `--regime 720`
  for its corpora/grid preset while pointing at a 944 root; refusing on the
  flag's number would have broken provenance copies that must not change. The
  carry-check therefore fires only when the regime's PRESET root is in effect.

The folded-regime refusal is root-aware in exactly the way the pre-existing 944
guard is (a `…pools` / `…carriers` root has the block live), and it defers to
that guard at 944 rather than duplicating it — so it ADDS the same protection at
720/924, where nothing checked it before.

## 3. The flip had disarmed the guard that exists for this bug

**Found by running the matrix above, not by reading the code.**
`block_profile::profile` tabulates its family table (`f0_155`, `f156_371`,
`f372_719`, …) by indexing `caller_line_norms` — which is indexed by layer-0
**POSITION** — with ranges of feature **IDS**. Identical for an identity layout;
not for a dense one.

MEASURED on the real artifact: the densified shipped `B` reads `f3..f369` at
positions `0..94`, so the positional fold reported **`uses_f156_371: false`** for
a bake that reads **49** lines in that block — and `folded_root_conflict`, whose
entire job is to refuse that read at a folded root (the recorded CID22 **0.3862**
against its true **0.8764**), returned `None` and let it through.

So between increment 2A landing and this fix, `bake_verdict --regime 944` on a
dense shipped `B` was **not** refused. Caught in the same session and before any
use, and stated here rather than quietly fixed: densifying a bake disarmed a
guard, which is exactly the risk a positional/id confusion carries.

Fixed by scattering the norms into id space before tabulating (an identity bake
maps `j -> j`, so no existing profile moves). Gates:
`a_dense_bake_reports_the_families_its_declared_ids_land_in` and
`folded_root_conflict_fires_for_a_dense_pool_reading_bake`, each with a negative
control proving an undeclared bake of the same shape reports the opposite.

## 4. Scripts

**MEASURED: 220 `--regime` occurrences across the tree, of which 25 name 372 and
195 name 720/924/944.** Of the 25, only **two** are executable call sites
(`benchmarks/fastclass2_campaign_2026-09-05/run_fastclass2_372.sh`); the rest are
prose in records, which are not edited. Those two are dropped — `--regime 372`
sets no default their explicit `--features-root` does not already set.

**The 195 others are NOT deleted, and that is the measured answer rather than a
scope cut.** `--regime 944` selects the features root, the dial grid, the
corruption grid, the per-pair metric source AND the 12-corpus campaign list;
`--regime 720` swaps the grids and filters the corpora. Deleting the literal
changes what those scripts MEASURE, not merely how they spell it. The flag is
deprecated as a width selector, not as a preset.

`just lint-scripts`: **611 scripts checked, all runnable.**

## 5. Verdict identity

`bake_verdict --full-json` on shipped dense `B` at the default 372 root, before
this increment vs after: **zero differing fields** over the whole JSON tree.
CID22 SROCC **0.8821166166351724**, kon504 **|0.5193759178072009|** — unmoved
across all four increments of this lane.

## 6. Board

The gauntlet was regenerated from the unchanged `fulleval` dir
(479 bakes) and the result is **BYTE-IDENTICAL to the live board** —
23,729,925 bytes, sha256 prefix `41d8a8f9d5f6fdeb` on both — so no statistic on
it moved across any of this lane's four increments.
`scripts/v_next/gauntlet_gates.sh` passes: `node --check` on every extracted
script block, the DOM-shim render harness, sortable-table clicks, ECharts mounts
+ both themes + SSR of one option per panel kind, 3,646 registry-annotated cells
badged, and all 479 fulleval files strict-valid JSON.

Servability census after all four increments: **17 SERVED, 0 REFUSED** through
the production `Zensim::compute`, plus the five in-lib census tests
(`every_shipped_profile_is_servable`,
`from_block_profile_agrees_with_the_id_space_derivation`, …) green.
