# The board's operative dial ruler becomes the LADDER instrument (2026-09-06)

**Lane:** `claude-ladderruler`, jj sibling workspace `~/work/zen/zensim--ladderruler`.
**Pre-registration:** [`docs/PLAN_BOARD_LADDER_RULER_2026-09-06.md`](../docs/PLAN_BOARD_LADDER_RULER_2026-09-06.md),
pushed as `22ffc5d2` **before any cell was re-graded**. §1–§8 there are frozen; this
file is the measurement record.

**User rules this serves.** *"floor and ceiling dial addressability is crucial … any
model that limits dial range cannot ship"* (2026-09-04); *"i care that the lowest
configurable settings per codec are representable"* (2026-09-05); *"for inversions, we
should choose say ssim2 and butter and only flag true inversions where they agree"*
(2026-09-05); and the standing no-cruft rule — one operative ruler, named, with the
retired one kept readable and never silently mixed in.

---

## 1. The defect, and why a second column was the only honest fix

The board's `dial.addressability` column was cut on the **2026-05-29 CANONICAL grid**
(4,424 cells, four codec families, one AVIF backend) and on `bake_verdict`'s default
944 grid. Every current gate is defined on the **2026-09-05 FLOOR-DENSE LADDER**
(`benchmarks/ladder_instrument_2026-09-05.md`): 9,593 distinct settings, five codec
families including `avif-svt` **and** `avif-rav1e`, 66 floor-dense q steps, saturation
flagged by encode hash.

Three things are measurable **only** on the ladder:

1. the operative `resolvable` `A7r` rule's registered per-codec bars;
2. jpeg's floor bar as a real number — on the canonical grid zenjpeg emits **one
   bitstream for q 0..10**, so its three lowest "settings" are one setting sampled
   three times and the bar is a vacuous `0.0000` that anything passes;
3. the two-reference inversion rule — the canonical grids carry butteraugli-`max`
   only, and the rule needs `pnorm3`.

`promote_fulleval.py --graft-gaddr` **refuses** a ladder reading, and correctly: its
same-grid gate proves a G-ADDR read was taken on the board's own grid, and two
instruments in one column produce silently wrong cross-cell comparisons. Nothing was
relaxed. The ladder reading gets its own block, `dial_ladder`, and
`dial.addressability` was **not moved, renamed, or rewritten** — verified: **0 of 508
`dial` blocks changed** across the whole pass.

## 2. Instrument selection — width, then era, both MEASURED

**Width** comes from `bake_block_profile --json`'s `caller_input_width`, never
`n_inputs`: a dead-column-pruned bake reports 28 or 228 while still taking 372
features. 372 → `dial_grid_372col_ladder`, 944 → `dial_grid_944col_ladder`, anything
else → NOT MEASURED.

**Era** is a measured property of the grids, not an assumption. Populated-slot
signature (first row group, sha8 over the sorted populated-slot list):

| grid | populated | populated ranges | slot-set sha8 |
|---|--:|---|---|
| `dial_grid_944col_ladder` (`foldapp2pools`) | **905** | `f0..719` live + the append blocks | `b6811ae0` |
| `dial_grid_944col_POOLS_2026-08-30` | 905 | identical | `b6811ae0` |
| `dial_grid_944col_2026-08-01` — `bake_verdict`'s DEFAULT 944 grid | **689** | `f0..155` + `f372..`, **`f156..371` STRUCTURALLY ZERO** | `026c0aba` |
| `era2-rank-2026-08-31/dial_grid_944col_r4` | 689 | identical to the row above | `026c0aba` |
| `dial_grid_372col_ladder` | 372 | all | `73d2a5e8` |
| `dial_grid_372col_2026-05-29_quarantined_v2` | 372 | all | `73d2a5e8` |

A bake trained where `f156..371` are always zero receives **zero gradient** on those
216 weights, so they keep their INIT values; a grid that POPULATES them multiplies
live data by untrained weights. That is the `--regime 944` silent-mis-scoring bug
class this repo already carries. So a 944 cell is graded only when its own instrument
is already pools-era, **or** `block_profile.uses_f156_371 == false` makes it provably
immune. Otherwise: NOT MEASURED, with that sentence as the reason.

**The immunity is the common case, and it is measured, not assumed:** 359 of the 381
graded 944 cells are immune (weights exactly zero on `f156..371`), 22 are already
pools-era, and 34 are refused.

## 3. Coverage — every one of the 508 board cells accounted for

`scripts/gaddr_board_ladder.py grade` (the 2026-09-04 as-run replay covered 97 cells;
this reconstructs an invocation for every cell from the fulleval's own fields when no
log exists).

| | cells |
|---|--:|
| **GRADED on a ladder instrument** | **450** |
| — 944, immune | 359 |
| — 372 | 67 |
| — 944, already pools-era | 22 |
| — peer cells (`peer_ssim2`, `peer_butteraugli`) | 2 |
| **NOT MEASURED, each with a recorded reason** | **58** |
| — era mismatch (bake reads `f156..371`) | 34 |
| — no ladder instrument at that caller width (8×156, 8×720, 4×924, 1×504) | 21 |
| — reference metric with no per-cell table on this instrument (`peer_cvvdp`, `peer_iwssim`) | 2 |
| — bake not on disk (`D_shipped@dguard2`, a deleted sibling workspace path) | 1 |

Per-cell reasons: `/mnt/v/output/zensim/ladder-board-2026-09-06/ladder_board_summary.tsv`.
**No cell is silently absent** (acceptance gate G1).

### 3.1 ONE thing changes per cell, and the probe recovery is PROVEN

Every other input is reproduced from the cell's own recorded invocation, so the
before/after diff is attributable to the ruler alone. **No probe is ADDED to a cell
that lacked one** — that would confound "the ruler moved" with "more rows became
measurable".

Recovering the right probe for a cell with no as-run log is not a guess:
`measured.negtail` is a function of `(bake, probe)` **only** and never touches the
dial grid, so the correct probe is the one whose read reproduces the board's block
**bit-for-bit**. MEASURED on `W10L9PH_s4007_packed` — the board's canonical-grid read
and this lane's ladder read agree exactly (`frac_below_zero` 0.0265, `min`
−69.61934236224508, `p1` −22.014636705459125, `p5` 7.100433565953122). 14 probes were
recovered this way, **0 unrecovered**.

Two traps this surfaced, both fixed by measurement rather than by loosening the check:

* **`measured.identity` mixes probe reads with GRID properties.** `n_above_identity`
  and `n_grid_cells_total` count dial-grid cells — 4,817 on the POOLS grid, 9,593 on
  the ladder — so an identity block can never match across instruments. Found on
  `Ffree@dfreelane`, whose board block records `n_grid_cells_total: 4817`. Only the
  probe-scoped fields are compared.
* **This build emits C5/C6 as measured CHECKS but serialises no `measured.identity`
  dict**, while the board blocks (written by another lane's build) carry one. So
  "the identity probe was accepted" is read from the check states, which every build
  emits.

## 4. WHAT FLIPS — and the pre-registered expectation was right, once decomposed

The NOT-SHIPPABLE badge is drawn off a MEASURED CONTRACT-tier fail in the operative
block. Over all 508 cells it goes **63 → 75**. The plan (§6.1) predicted a small
DECREASE and required a rise to be explained before publishing. Decomposed, it is:

| | cells |
|---|--:|
| badge BEFORE (canonical ruler) | 63 |
| badge AFTER (ladder ruler) | **75** |
| of the +22 gained: **NEWLY-GRADED** cells that had no dial ruler at all before | **21** |
| of the +22 gained: TRUE flips on a cell that already had a canonical block | **1** (`A2b_l0.002`, C1) |
| lost, all with a canonical block | **10** |

**So the true ruler effect is −10 +1 = −9 badges. The +21 is first-ever coverage**
on cells the canonical pass never reached — 171 of 508 cells had a G-ADDR block
before, 450 have one now.

**On the 93 as-run cells this lane re-graded, the contract-fail count goes 46 → 42** —
the decrease §17.7 of the gate doc measured (47 → 43 on its 97), reproduced
independently on a different code path.

**Every one of the 21 new badges is on a LEGACY (20) or FAIR-NOTED (1) cell; every one
of the 10 lost badges is VERIFIED-FAIR.** So the fair board strictly *loses* badges
and the all-rows board gains coverage — a clean split, and not one that was designed
for.

### 4.1 The two rows that move, and why — measured, not reasoned

* **C2** (*no dial-grid cell out-scores a perfect copy*, bar ≤ 0.05) gets **easier**:
  `W10L9PH_s4007_packed` 0.09081 → **0.03060**, `LSTAR3_s4043_packed` 0.12420 →
  **0.04590**, `LSTAR3__I__i5012_p4041_packed` 0.06444 → **0.01998**. The ladder is
  floor-dense, so proportionally fewer of its cells sit near the ceiling clamp.
* **C1** (*monotonicity*, bar ≥ 0.93) gets **harder**: `A2b_l0.002` 0.97852 →
  **0.91712**. The ladder samples q 0..30 at step 1, so it holds far more
  near-flat adjacent pairs for a dial to wobble on.

Board-wide the C-row fail counts on the ladder are C3 43, C4 43, **C2 22, C1 20**,
C5 5. The switch is **directional per row, not uniformly stricter** — a reader who
takes "the ladder is a harder ruler" as a blanket statement will misread C2.

## 5. A7r on the ladder — and the rule the §9 tables were cut under

Per-codec floor representability over the 450 graded cells, under the **OPERATIVE
`resolvable`** rule (margin 0.5), whose mentor bars are `avif-rav1e` 0.6410 /
`avif-svt` 1.0000 / `jpeg` 0.6667 / `jxl` 0.9615 / `webp` 1.0000:

| codec | cells passing | median fraction |
|---|--:|--:|
| `avif-rav1e` | 11 (2.4 %) | 0.2821 |
| `avif-svt` | 38 (8.4 %) | 0.7692 |
| `jpeg` | 47 (10.4 %) | 0.4872 |
| `jxl` | 71 (15.8 %) | 0.7308 |
| `webp` | 115 (25.6 %) | 0.9487 |

Cells by how many codecs they clear: **293 clear none**, 91 clear one, 30 two, 23
three, 3 four, and **10 clear all five**.

**⚠ These are `resolvable` fractions and are NOT the `distinct` fractions
`benchmarks/ladder_instrument_2026-09-05.md` §9–§9.4 tabulates.** The two are
different quantities on the same cells and the difference changes verdicts: under
`distinct` that record measured shipped Profile D **failing jpeg by one ladder**
(0.5128 against a 0.5385 bar); under the operative `resolvable` rule the same
ADD156/D lineage passes all five. Always read `floor_rule` beside a fraction.

### 5.1 The key rows

| cell | contract | A7r | `avif-rav1e` | `avif-svt` | `jpeg` | `jxl` | `webp` |
|---|---|:--:|--:|--:|--:|--:|--:|
| **`peer_ssim2` — THE MENTOR** | **PASS** | pass | 0.6410 | 1.0000 | 0.6667 | 0.9615 | 1.0000 |
| `D_shipped_ctl@did100lane` | INCOMPLETE | **pass** | 0.6667 | 1.0000 | 0.6667 | 1.0000 | 1.0000 |
| `d_id100@did100lane` | INCOMPLETE | **pass** | 0.6667 | 1.0000 | 0.6667 | 1.0000 | 1.0000 |
| `d_id100_negrich@did100lane` | INCOMPLETE | **pass** | 0.6667 | 1.0000 | 0.6667 | 1.0000 | 1.0000 |
| `ADD156_safesyn_only_raw_lasso` | INCOMPLETE | **pass** | 0.6667 | 1.0000 | 0.6667 | 1.0000 | 1.0000 |
| `v47_strict_QAT_native@cur372` (Profile A) | **PASS** | fail | 0.3590 | 0.8462 | 0.5128 | 0.8462 | 1.0000 |
| `W10L9P_s4005_packed` | INCOMPLETE | fail | 0.4872 | 0.9231 | 0.5641 | 0.9231 | 0.9744 |
| `peer_butteraugli` | INCOMPLETE | fail | 0.2564 | 0.9487 | 0.7692 | 0.5385 | 0.9231 |
| `BOA_A_plain_s4004` | INCOMPLETE | fail | 0.1795 | 0.8205 | 0.5641 | 0.3846 | 0.9744 |
| `BOA_F_nonneg32_s4004` | INCOMPLETE | fail | 0.1282 | 0.6667 | 0.4359 | 0.3077 | 0.7949 |
| `fc2_372_S228_H128_s4004` (the `--select` pick) | INCOMPLETE | fail | 0.1795 | 0.8205 | 0.5641 | 0.3846 | 0.9744 |
| `fc2_944_S228_H32_s4004` | INCOMPLETE | fail | 0.2051 | 0.7949 | 0.4615 | 0.4231 | 0.8462 |
| `D_guard12_p999@dguard2` | — | — | NOT MEASURED — caller width 156 | | | | |

**The instrument endorses the ADD156/D lineage and nothing else on the board.** Four
D-lineage cells meet or beat the mentor on every codec (`jpeg` exactly at its bar,
`jxl` and `avif-rav1e` above it); `peer_ssim2` is the only other five-codec pass.
Every best-of-all and fast-class arm — including the registered `--select` winner —
fails at least three codecs, and `avif-rav1e` is where they collapse (0.13–0.21
against a 0.641 bar). **`peer_butteraugli` fails four of five**, which is a second
independent argument against it as a mentor and is consistent with gate doc §18's
finding that `peer_butteraugli_max` fails C1 under both inversion readings.

## 6. `freeze_check --select` — the pick does NOT move, and 10 vetoes lift

Over the **125 VERIFIED-FAIR** cells, `--select --seed-group --min-k 2 --floor-basis
all`:

| ruler | SELECTED |
|---|---|
| `--gaddr-block canonical` (pre-2026-09-06) | `11e243eb0b86` — k=3, 8/8 floors, mean composite 0.9853 |
| `--gaddr-block auto` (**operative**) | `11e243eb0b86` — identical |
| `--gaddr-block ladder` | `11e243eb0b86` — identical |

Members: `fc2_372_S228_H128_s4004/5/6`. Floors every seed clears: `bandtail, breadth,
cid22, dial, dialrange, hfnl, konjnd, nonphoto`.

**G5, proven against a pristine binary rather than argued:** `freeze_check.rs` was
reverted to the parent revision's own source, built into a separate
`CARGO_TARGET_DIR`, and run on the same 125 cells. `--gaddr-block canonical` is
**byte-identical** to that binary's output (empty diff).

**10 cells change selectability, all in the same direction (`NO — CONTRACT FAIL` →
`yes`)** — `W11J__I__i5011_p4013_packed`, `W10L9PH_s4007_packed`, `w11_s4013_e060`,
`LSTAR3_s4043_packed`, `LSTAR3__S__i4041_p5001_packed`, `lstar2_4031_e060`,
`LSTAR3__I__i5012_p4041_packed`, `w11_s4012_e080`, `LSTAR__I__i5012_p4021_packed`,
`LSTAR3__S__i4041_p5002_packed`. Their C2 genuinely PASSES on the operative
instrument (§4.1). **No fair cell gains a veto.** Worth stating plainly: switching
ruler can *admit* a candidate the previous ruler vetoed — the veto's
removal-only property holds within a ruler, not across one.

## 7. What landed

* **`scripts/gaddr_board_ladder.py`** — `grade` / `graft` over every board cell.
* **`promote_fulleval.py --graft-gaddr-ladder`** — writes `dial_ladder` +
  `dial_ladder_source` only; `dial` is not in the write allow-list. Its gate is the
  MIRROR of `--graft-gaddr`'s: where that proves *"this read was taken on the board's
  own grid"*, this proves *"on a REGISTERED ladder grid"*, resolved from
  `benchmarks/dial_addressability_floor_2026-09-04.json` at run time rather than
  hard-coded. Peer readings (no bake to sha) are name-gated against the matching
  `reference-metric` cell. Self-test with a **negative control** (a non-ladder grid
  must be REFUSED) — PASS; the pre-existing `--graft-gaddr` self-test still PASSes.
* **`freeze_check`** — `gaddr_block` is the ONE owner of which block the CONTRACT
  veto and the `A7r` floor fold read; `--gaddr-block auto|canonical|ladder`.
  `ladder` refuses to fall back, so a cell with no ladder read is NOT MEASURED
  rather than graded on another instrument. New test pins all three modes on a cell
  carrying BOTH blocks with OPPOSITE verdicts, plus both absence cases (40 tests pass).
* **`gauntlet.py`** — one owner picks the operative block; the scoreboard column, the
  NOT-SHIPPABLE badge, the `floors ok` column and the G-ADDR panel all read it. Every
  tooltip NAMES its ruler; a ladder-graded cell carries the canonical read beside it
  as **`dial_canonical`** — context only, labelled a different instrument, never
  sorted on, counted, or mixed into a column.
* **`benchmarks/eval_annotations.json`** — three append-only entries scoping every
  pre-switch dial number to the canonical instrument, recording the 58 NOT-MEASURED
  cells as *absent-not-failed*, and warning that the board's A7r fractions are
  `resolvable` and not the `distinct` ones §9 of the ladder record tabulates.

## 8. Acceptance gates

| # | gate | result |
|---|---|---|
| G1 | every cell graded or NOT MEASURED with a reason | **PASS** — 450 + 58 = 508 |
| G2 | no `dial_ladder` on a non-registered ladder grid | **PASS** — 0 |
| G3 | no board cell's `dial` block changed | **PASS** — 0 of 508 |
| G4 | both promoter self-tests, incl. the negative control | **PASS** |
| G5 | `--gaddr-block canonical` reproduces the pristine binary | **PASS** — byte-identical |
| G6 | `gauntlet_gates.sh` on both boards | **PASS** — rc=0 both |
| G7 | fair board under the registered 12 MiB cap | **PASS** — 12,528,928 B (11.9485 MiB), margin **53,984 B** |
| G8 | served board + named compare fragment | **PASS** — 200 on both; fragment resolves 4/4, no banner |
| G9 | superseded dial numbers scoped in the registry | **PASS** — 3 entries |

**G7 is the one to watch.** The margin fell from ~164 KB to **53,984 B**, because 279
cells gained a first-ever G-ADDR block. The registered trim levers (curated-set
per-pair stripping, note truncation) are untouched and available; the next lane that
adds a per-cell block should re-measure before assuming headroom.

**Two of the three new registry entries are deliberately DOCUMENTATION-ONLY**
(`{"manual": ...}` scope), and the reason is editorial before it is bytes: an
annotation that fires on 450 of 450 ladder-graded cells, or on 171 of 171 canonical
blocks, is a statement about the RULER and carries no information about the CELL — a
badge on every row is noise. The board says both things where a reader actually meets
the number: every G-ADDR tooltip NAMES its ruler, the `floors ok` tooltip names it
again, and a ladder-graded cell renders `dial_canonical` explicitly labelled a
different instrument. Only the genuinely per-cell fact — *this* cell has no ladder
reading, and here is why — keeps a machine scope. (It also bought 38,448 B of the
margin above; the editorial call and the byte saving happened to point the same way,
and the first was decided before the second was measured.)

## 9. Reproduce

```sh
cargo build --release -p zensim-validate --bin bake_verdict --bin bake_block_profile --bin freeze_check --bin panel
scripts/gaddr_board_ladder.py grade --out /mnt/v/output/zensim/ladder-board-2026-09-06 --jobs 8
scripts/gaddr_board_ladder.py graft --out /mnt/v/output/zensim/ladder-board-2026-09-06
# the retired window / retired value-pin grading, for audit
scripts/gaddr_board_ladder.py grade --out <other> --floor-rule distinct --value-pins hard

# selection under each ruler
freeze_check --select <fair cells> --seed-group --min-k 2 --floor-basis all \
    --gaddr-block canonical|auto|ladder

cd scripts/v_next && export ZEN_PANEL_BIN=<repo>/target/release/panel
python3 bandwise_dashboard.py --fulleval-dir /mnt/v/output/zensim/reports/fulleval \
  --fair-only --fairness-tsv /mnt/v/output/zensim/reports/fairness_tiers_2026-09-06.tsv \
  --out /mnt/v/output/zensim/reports/summer_gauntlet_fair.html
python3 bandwise_dashboard.py --fulleval-dir /mnt/v/output/zensim/reports/fulleval \
  --out /mnt/v/output/zensim/reports/summer_gauntlet.html
../../scripts/v_next/gauntlet_gates.sh <each html>
```

As-run artifacts (block storage, not git):
`/mnt/v/output/zensim/ladder-board-2026-09-06/{gaddr,logs,cells.json,ladder_board_summary.tsv}`.

## 10. Registered, NOT run

* **A ladder instrument at 720 / 924 / 504 / 156-native width** — 21 board cells have
  no ruler for want of one. Building each is a full encode+extract pass in that
  feature set.
* **A pools-era ladder read for the 34 era-mismatched 944 cells.** Needs either a
  ladder instrument in the folded (`026c0aba`) feature set or bakes whose
  `f156..371` weights are exactly zero. Not a re-run: a different instrument.
* **`peer_cvvdp` / `peer_iwssim` per-cell tables on the ladder.** The instrument
  scores ssim2, butteraugli-max and butteraugli-pnorm3; those two metrics were never
  run over its 9,593 cells.
* **`D_shipped@dguard2`'s bake** points into a deleted sibling jj workspace
  (`~/work/zen/zensim--dguard2`) — the committed-path-into-a-worktree fossil this
  repo's own linter rule warns about, found here on a board cell rather than a script.
