# PLAN — make the LADDER instrument the board's operative dial ruler (2026-09-06)

**Lane:** `claude-ladderruler`, jj sibling workspace `~/work/zen/zensim--ladderruler`.
**Status of this file when pushed:** PRE-REGISTRATION. §1–§8 are frozen before any
re-grade ran. Results are §9 onward and are appended, never rewritten.

**User rules this serves.**
* 2026-09-04, verbatim: *"floor and ceiling dial addressability is crucial … any model
  that limits dial range cannot ship."*
* 2026-09-05, verbatim: *"i care that the lowest configurable settings per codec are
  representable, not that negative fifty is in that specifically."*
* 2026-09-05 (third), verbatim: *"for inversions, we should choose say ssim2 and butter
  and only flag true inversions where they agree, and we can then file or update tracking
  issues on codecs for when they are nonmonotonic."*
* Standing: no cruft, no confusion — one operative ruler, named, with the retired one
  kept readable and never silently mixed in.

---

## 1. The defect, stated as two facts that cannot both stay true

**Fact A — the board's dial column is the 2026-05-29 CANONICAL grid.** Every
`dial.addressability` block on the board was cut on one of six instruments, dominated by
`dial_grid_372col_2026-05-29_quarantined_v2` (4,424 cells, 106 ladders, ONE avif backend)
and `dial_grid_944col_2026-08-01` (4,817 cells). On the canonical grid `zenjpeg`'s three
lowest "settings" are **byte-identical** — q 0/5/10 are one bitstream — so its floor bar is
a vacuous `0.0000` that anything passes.

**Fact B — every current gate is defined on the 2026-09-05 LADDER instrument.**
`benchmarks/ladder_instrument_2026-09-05.md`: 5 codec families (including `avif-svt` AND
`avif-rav1e` as separate ladders), 39 references, a 66-step floor-dense q axis, saturation
flagged by encode hash, bytes and pixels persisted, both 372 and 944 widths, and registered
`peer_ssim2` bars. It is the ONLY instrument on which
* the operative `resolvable` floor rule (`A7r`) has a registered per-codec bar,
* the two-reference inversion rule (§18 of the gate doc) is measurable at all — the
  canonical grids carry butteraugli-`max` only, and the rule needs `pnorm3`,
* and jpeg's floor bar is real (`0.5385`, not `0.0000`).

**The wall.** `promote_fulleval.py --graft-gaddr` REFUSES to merge a ladder reading into
`dial.addressability`, and it is right to: the block's same-grid gate proves a G-ADDR read
was taken on the board's own grid, and two instruments in one column produce silently wrong
cross-cell comparisons. Measured refusal text, gate doc §17.7: *"`dial.mono_pct` differs
between the board (0.9946831135686942) and the G-ADDR read (0.9758792901923281) — the read
was NOT taken on the board's dial grid; refusing."* So the 97 ladder readings that exist
live in a sidecar directory the board cannot see, and `freeze_check --select`'s CONTRACT
veto cannot act on them.

**The fix is a SECOND column, not a relaxed gate.** Nothing about the same-grid gate is
loosened. The ladder reading gets its own block with its own instrument stamp.

---

## 2. What lands

1. **`dial_ladder`** — a new, top-level fulleval block carrying the ladder-instrument
   G-ADDR verdict verbatim, plus an instrument stamp (path, sha256, width, registered
   label, floor rule, floor margin, reference/mentor, value pins, tail pins) and the era
   decision that admitted or refused the cell.
2. **`dial.addressability` is NOT moved, renamed, or rewritten.** It stays exactly where
   it is and remains the canonical-grid read; the board SURFACES it under the name
   `dial_canonical` and never compares it across instruments. Rewriting 508 sha-gated
   blocks to rename a field would break `--graft-gaddr`'s idempotence check and the
   2026-09-05 regrade path for zero measurement gain.
3. **The board's operative ruler becomes `dial_ladder`** — the NOT-SHIPPABLE badge, the
   `floors ok` column, the G-ADDR panel and the `dial`/`dial.addressability` scoreboard
   column all read it when present and fall back to the canonical block when it is absent.
4. **`freeze_check --select`'s CONTRACT veto and A7r floor folding read the same block**,
   through ONE accessor, with an explicit flag to reproduce the canonical-only reading.

---

## 3. Instrument selection — width, then era, both MEASURED

### 3.1 Width
From the bake's CALLER width via `bake_block_profile --bake X --json`
(`caller_input_width`), never `n_inputs`: a dead-column-pruned bake reports `n_inputs` 28
or 228 while still taking 372 features. Then:

| caller width | ladder instrument |
|---|---|
| 372 | `dial_grid_372col_ladder.parquet` (`4c3874a78c469e15`) |
| 944 | `dial_grid_944col_ladder.parquet` (`0e8e5fb789bd21b2`) |
| anything else (720 / 924 / 504 / 156-native) | **NOT MEASURED** — "no ladder instrument at width W" |

Both ladder grids hold the SAME 9,593 distinct settings and are registered in
`benchmarks/dial_addressability_floor_2026-09-04.json` under BOTH the `distinct` and the
operative `resolvable` (margin 0.5) rules.

### 3.2 Era — MEASURED, not assumed
The two 944 instruments are **different feature sets**, measured here by the populated-slot
signature of each grid (first row group, sha8 of the sorted populated-slot list):

| grid | populated slots | populated ranges | slot-set sha8 |
|---|--:|---|---|
| `dial_grid_944col_ladder` (ladder, `foldapp2pools`) | **905** | `f0..719` live, then the append blocks | `b6811ae0` |
| `dial_grid_944col_POOLS_2026-08-30` | 905 | identical | `b6811ae0` |
| `dial_grid_944col_2026-08-01` (bake_verdict's DEFAULT 944 grid) | **689** | `f0..155` + `f372..`, **`f156..371` STRUCTURALLY ZERO** | `026c0aba` |
| `era2-rank-2026-08-31/dial_grid_944col_r4` (era2r4 foldapp2) | 689 | identical to the row above | `026c0aba` |

A bake trained where `f156..371` are always zero receives **zero gradient** on those 216
weights, so they keep their INIT values. Scoring it on a grid that POPULATES them
multiplies live data by untrained weights — the `--regime 944` silent-mis-scoring bug class
this repo already carries as a Known Bug. Therefore, for a 944 cell:

* the cell's ORIGINAL grid is the pools-era one (`694e16c4520a5d41`) ⇒ **era-matched, GRADE**;
* else `bake_block_profile`'s `uses_f156_371 == false` ⇒ **provably IMMUNE** (the columns
  the bake reads are the same in both feature sets) ⇒ **GRADE**, recorded as `era: immune`;
* else ⇒ **NOT MEASURED**, reason: `era mismatch: bake reads f156-371, which its own 944
  instrument zeroes (026c0aba) and the ladder-944 populates (b6811ae0)`.

At 372 the ladder grid populates all 372 slots (`73d2a5e8`), the same slot set as every
372 root, so no cell is refused on slot grounds. The ladder-372 instrument is a SINGLE
extraction era; that it differs from the 2026-05-29 canonical grid's era is the *point* of
the switch, is recorded on every cell's instrument stamp, and is why a `dial_ladder` value
is never compared with a `dial_canonical` one.

### 3.3 The refusals `bake_verdict` already owns
Width mismatch, formula-revision mismatch and `SlotsNotPopulated` all make `bake_verdict`
exit non-zero. Any non-zero exit becomes **NOT MEASURED with the tool's own refusal text as
the reason** — never a retry with a different flag, never a number.

---

## 4. ONE thing changes per cell — the instrument

Every other input is REPRODUCED from the cell's own recorded invocation so the before/after
diff is attributable to the ruler alone:

* bake path + sha, ensemble members and weights, `--features-root`, `--corpora`,
  `--regime`, `--cross-regime`, `--negtail-probe`, `--identity-probe`, `--floor-rule`
  (omitted ⇒ the registry's operative rule), `--gaddr-value-pins report`,
  `--gaddr-tail-pins product`.
* Sources, in order: the 2026-09-04 as-run log when one exists (97 cells), else the
  fulleval's own fields (`bake`, `bake_sha256`, `features_root.path`, `model.*`,
  `dial.addressability.*`).
* **An input that cannot be recovered is left ABSENT**, which yields NOT MEASURED on the
  rows that need it — never a guess. In particular no negative-tail or identity probe is
  ADDED to a cell that did not have one: adding coverage would confound "the ruler moved"
  with "more rows became measurable".
* `--gaddr-grid-truth` and `--reference-truth` are resolved from the REGISTRY by grid
  sha (`grid_floor_representability` / `inversion_truth.reference_tables`), a lookup, not
  a guess. The ladder is the only instrument carrying a `pnorm3` table, so the two-reference
  inversion rule becomes measurable on the board for the first time.

---

## 5. The board and the selector

**Gauntlet.** ONE accessor decides the operative block: `dial_ladder` if present, else
`dial.addressability`. The panel header names the instrument and its rule; the canonical
block ships alongside as `dial_canonical`, rendered on hover with an explicit
"different instrument — never compare across" note, and is never mixed into a column,
a sort, or a badge.

**`freeze_check`.** `gaddr_contract_fails` and `gaddr_codec_states` read through the same
one accessor. `--gaddr-block canonical` reproduces the pre-2026-09-06 reading byte-for-byte;
`ladder` forces the new block; `auto` (default) is ladder-then-canonical. `--floor-basis
legacy` keeps its existing meaning (no veto, no A7r floors) unchanged.

**The graft.** `promote_fulleval.py --graft-gaddr-ladder <gaddr.json> --graft-into <cell>`:
* sha-gated on the scorer bake exactly as `--graft-gaddr` is;
* REFUSES unless the reading's `grid_sha256` is one of the two REGISTERED ladder grids
  (read from the registry at run time, not hard-coded);
* writes only `dial_ladder` + `dial_ladder_source`; **`dial` is untouched** and that is
  asserted by the existing `_write_board_gated` allow-list;
* the same-grid gate on `--graft-gaddr` is NOT relaxed and NOT reused here — a ladder
  reading is a different instrument by construction, so the gate that proves "same grid"
  is replaced by one that proves "a registered ladder grid".

---

## 6. Pre-registered expectations (so a surprise is visible as a surprise)

1. **A7r will get much harder and the badge count will move DOWN, not up.** Gate doc
   §17.7 measured all 97 reconstructible cells FAILING A7r on the ladder, while the
   contract-fail count fell 47 → 43 on the same 97. A7r is a REGRESSION row; the badge is
   CONTRACT-driven. So: many more A7r fails, and a small DECREASE in NOT-SHIPPABLE badges.
   A badge count that goes UP would be a surprise and must be explained before publishing.
2. **`peer_ssim2` will pass the contract tier on the ladder** (gate doc §14.8 measured 6/6
   on the canonical grid; §9 of the ladder record re-derived its floor bars there).
3. **Shipped Profile D fails A7r on jpeg by exactly one ladder** (20/39 vs the mentor's
   21/39) — the single result the instrument was built to expose.
4. **A large fraction of 944 cells will be NOT MEASURED on era grounds.** `bake_verdict`'s
   default 944 grid is the folded-era one, so most 944 board cells were graded there.
5. **`--select`'s pick may move**, because the veto and the A7r floor fold now read a
   harder instrument. A pick that does NOT move is also a result and is reported as one.

## 7. Acceptance gates (all must hold before anything is published)

| # | gate |
|---|---|
| G1 | Every one of the 508 board cells is either GRADED on a ladder instrument or NOT MEASURED **with a recorded reason string**. No cell is silently absent. |
| G2 | No `dial_ladder` block exists whose `grid_sha256` is not a registered ladder grid. |
| G3 | No board cell's `dial` block changed. Byte-diff over all 508 before/after, restricted to `dial`, must be empty. |
| G4 | `promote_fulleval.py --self-test-graft-gaddr` still passes (the canonical graft is untouched), and the new ladder graft has its own fixture self-test including a NEGATIVE control (a non-ladder grid must be REFUSED). |
| G5 | `freeze_check --gaddr-block canonical` reproduces the pre-change `--select` output byte-for-byte. |
| G6 | `scripts/v_next/gauntlet_gates.sh` passes on BOTH emitted boards (`node --check` on every script block + the DOM-shim render harness). |
| G7 | The fair board is < 12 MiB (the registered size rule). |
| G8 | The served fair board returns HTTP 200 and a compare fragment on four named cells renders. |
| G9 | Every superseded dial number is scoped in `benchmarks/eval_annotations.json` to the CANONICAL instrument, so no reader treats a pre-switch value as a ladder value. |

## 8. Reversibility

* `--gaddr-block canonical` (freeze_check) and deleting the `dial_ladder` blocks restore
  the pre-change reading exactly; the canonical block was never touched, so the restore is
  a deletion, not a re-derivation.
* `scripts/gaddr_board_ladder.py grade --floor-rule distinct --gaddr-value-pins hard`
  reproduces the pre-ruling window on the ladder.
* Nothing in `zensim/weights/` is opened for writing. No gate bar is changed.

