# `freeze_check --select` — the G-ADDR CONTRACT tier is now a hard pre-filter (2026-09-06)

Owner fix for registry entry [`select-rule-blind-to-dial-contract-2026-09-06`](eval_annotations.json)
(landed by [`best_of_all_2026-09-06.md`](best_of_all_2026-09-06.md) §5.14). Lane: sibling jj
workspace `zensim--selectfix`.

**Thesis, confirmed.** `freeze_check --select`'s PRIMARY key (the `balanced-2026-08-04`
profile's F1-F8 floor count) and its TIE-BREAK (`selection_composite`) were both **completely
blind to `dial.addressability`** — G-ADDR's CONTRACT tier (C1-C6), the absolute product bar
the user made a HARD ship gate on 2026-09-04 ("any model that limits dial range cannot ship").
On the best-of-all wave, `--select --seed-group --min-k 2 --floor-basis all` picked `A_plain`
(CONTROL, G-ADDR contract 4/6 — fails C5+C6) over five arms that reach 6/6 on every seed,
because floor count and composite have no way to see a contract failure at all.

## 1. The fix

Two additive changes to `zensim-validate/src/bin/freeze_check.rs`, both reading the
candidate's OWN `dial.addressability` block (`dial_addressability::to_json`'s JSON shape,
embedded by `bake_verdict` — never recomputed):

1. **CONTRACT pre-filter (hard veto).** Any candidate (or, under `--seed-group`, any GROUP
   with at least one MEMBER) that MEASURES a G-ADDR CONTRACT-tier (`C1`-`C6`) fail is now
   **absolutely unselectable** — independent of floor count or `selection_composite`. A
   `NOT MEASURED` contract row is never a fail (an absent `dial.addressability` block, or an
   all-unmeasured contract tier, both stay eligible — only a `state: "fail"` row disqualifies,
   the same "absence is not evidence of failure" rule the codebase already applies to an
   unmeasured M3a). Vetoed candidates are listed under their own **"NOT SELECTABLE —
   candidate(s): contract FAIL — not selectable"** heading (ungrouped path) or via
   `group_unselectable_reason`'s `"CONTRACT FAIL — not selectable (C5,C6)"` string
   (seed-group path), naming the failing rows.
2. **`A7r` per-codec floors folded into `--floor-basis all`.** The per-codec
   floor-representability breakdown (`dial.addressability.measured.codec_floor[]`) becomes
   one MORE floor per MEASURED codec: a codec `state: "pass"` credits a pass, `"fail"`
   credits a fail, and `"not_measured"` is excluded from BOTH numerator and denominator (a
   codec the instrument never touched can neither certify nor condemn). At the seed-group
   level this uses the SAME all-reps discipline as the existing F1-F8 intersection: a codec
   counts in the denominator only when EVERY representative measured it, and in the numerator
   only when additionally every representative passed it
   (`seed_group_a7r_credited_only_when_every_rep_measures_and_passes_it`).

`FloorBasis` gained a third value, **`--floor-basis legacy`**, which reproduces the
**pre-2026-09-06 `all` byte-for-byte**: F1-F8 only, no `A7r` floors, no CONTRACT veto — the
audit/reproduction escape hatch (same role as `--min-k 1` for the replication floor), never
for a real selection. `--floor-basis mean` keeps its existing, unrelated meaning (the k-seed
MEAN floor count, F1-F8 only, unaffected by `A7r`) but **still carries the CONTRACT veto** —
the veto is a product-safety gate, not a floor-counting convention, so only the explicit
`legacy` escape bypasses it.

Both effects are strictly additive: they can only REMOVE a candidate from selection, never
admit one the pre-fix rule refused (mirroring the existing replication-floor and
floor-basis-`all` amendments' own "only removes" invariant, verified for the veto by
`gaddr_contract_veto_blocks_a_higher_composite_seed_group`'s final assertions).

## 2. The failing-first test

`gaddr_contract_veto_blocks_a_higher_composite_seed_group` builds two synthetic seed groups
via the exact shape `dial_addressability::to_json` emits (a `gaddr_fixture` helper, never a
second implementation of the verdict): a **CONTROL** group (3 seeds, every seed measures a
`C5`+`C6` contract fail, mean M3a ≈0.92 — the higher tie-break) and a **PASS** group (3 seeds,
contract-clean, mean M3a ≈0.60 — the lower tie-break), both tied at 8/8 balanced floors so the
test isolates the veto from any floor-count confound. **Verified failing-first**: with the
veto stubbed out (`gaddr_vetoes` forced to `false`, the `group_unselectable_reason` check
short-circuited), the test fails exactly as the bug predicts —
`assertion `left == right` failed: PASS must be SELECTED — CONTROL is vetoed / left:
"36a170f355c0" (CONTROL) / right: "e57751f93504" (PASS)`. Restoring the fix makes it (and 4
companion tests — `gaddr_contract_veto_excludes_an_ungrouped_row`,
`gaddr_a7r_folds_into_all_reps_floor_count_but_not_mean_or_legacy`,
`seed_group_a7r_credited_only_when_every_rep_measures_and_passes_it`, and the fixture-sanity
assertions inside the failing-first test itself) pass. Full suite:
**39/39** (`zensim-validate --bin freeze_check`), 35 pre-existing unchanged + 4 new, all green;
`cargo clippy` clean (zero new warnings); `cargo fmt -p zensim-validate` clean.

## 3. Re-run over the best-of-all board cells

**Direct re-run over `/mnt/v/output/zensim/best-of-all-2026-09-06/verdicts/*.fulleval.json`**
(the exact `bestofall_gates.sh select` / `endgame_bestofall.sh do_select` invocation) is
**UNCHANGED — still selects `8ad90f29c3a8` (`A_plain`).** This is a DATA-availability fact of
this wave, not a defect in the fix: `do_board`'s own comment records that the wave's G-ADDR
was measured on a **different instrument** (the floor-dense LADDER grid) than the board's
`dial.addressability` (the canonical DIAL grid, where every contract row reads
`NOT MEASURED` for all 27 cells) — `promote_fulleval.py --graft-gaddr` correctly REFUSES to
merge them ("the read was NOT taken on the board's dial grid"), so none of these 27 fullevals
carry a usable contract reading in the field `--select` reads. Confirmed by direct inspection:
all 27 verdicts read `contract: "INCOMPLETE (not a pass)"` with zero measured fails.

**Demonstration re-run** (scratch copies only, `~/tmp/selectfix_demo_verdicts/`, never
touching the committed board): merging each cell's sibling `gaddr/gaddr_<name>.json` (the
peer-safe `--gaddr-json` ladder-instrument output) into a COPY's `dial.addressability` field
reproduces the registry's exact numbers — `A_plain`/`E_plainlad`/`G_anchorlad` measure `C5`+
`C6` fails on all 3 seeds each (contract 4/6); `B_nonneg`/`C_lad05`/`D_lad20`/`D_lad20m`/
`F_nonneg32`/`H_anchorlad` are clean (6/6) — and running `--select --seed-group --min-k 2
--floor-basis all` over the merged copies:

| | before (raw board data / `--floor-basis legacy`) | after (`--floor-basis all`, gaddr merged) |
|---|---|---|
| **SELECTED** | `8ad90f29c3a8` = **A_plain** (k=3, 8/8 floors, sel_comp 0.9850) | `510eaeb89d6e` = **F_nonneg32** (k=3, 7/8 floors, sel_comp 0.9836) |
| **A_plain / E_plainlad / G_anchorlad (9 cells)** | ranked #1-#8, all `selectable: yes` | all 9 flip to `NO — CONTRACT FAIL — not selectable (C5,C6)` |

`A_plain` still RANKS #1 by floor-count+composite (the fix is a selectability veto, not a
re-sort — pinned by the test's own sanity assertion), but the SELECTABLE winner search skips
every vetoed group and lands on `F_nonneg32`, the highest-ranked CONTRACT-clean recipe. Full
transcripts: `~/tmp/selectfix_boa_after_raw.txt` (direct, unchanged), `~/tmp/selectfix_boa_after_demo.txt`
(demonstration, merged). The demonstration copies and transcripts are scratch, not committed —
this doc is the durable record.

## 4. Re-run over the fair board (VERIFIED-FAIR tier, 125 cells)

`scripts/v_next/gauntlet.py --fairness-tsv` over the current 508-cell board
(`/mnt/v/output/zensim/reports/fulleval/`) resolves **125 VERIFIED-FAIR** cells (46
FAIR-NOTED, 337 LEGACY). Of those 125, **50 carry a MEASURED G-ADDR CONTRACT fail** in their
own embedded `dial.addressability` block (the `LSTAR`/`LSTAR2`/`LSTAR3`/`W10L9PBR`/`W10L9PB`/
`W10L9PH`/`HDR944_GH2b` families, mostly `C2`/`C3`/`C4`).

Running `--select --seed-group --min-k 2` over exactly these 125 paths, `--floor-basis legacy`
(= byte-for-byte the pre-fix rule, verified in §2) vs default `--floor-basis all`:

- **The WINNER is unchanged**: `11e243eb0b86` (`fc2_372_S228_H128`, k=3, 8/8 floors, sel_comp
  0.9853) is already rank #1 pre-fix and carries no contract fail or A7r data at all
  (`gaddr: — (not measured)`) — so on THIS pool the fix has nothing to override at the top.
- **9 lower-ranked groups (ranks 14, 15, 16, 17, 22, 23, 24, 25, 27) flip from `selectable:
  yes` to `NO — CONTRACT FAIL — not selectable`**: `4ec838fb58b9`/`7329e47afb88`/`215b6fd2d665`/
  `1779f777d767`/`5f9365c4e550`/`9facbc8a2223` (C2,C3,C4) and `2de29bc5ae93`/`277c17e7b0c6`/
  `167f924daedc` (C3,C4). None of them ranked above the winner, so the exit-code-bearing pick
  is unaffected on this pool today — but every one of those 9 was, before this fix, a
  candidate `--select` would have happily promoted to "selected" had it simply ranked one
  slot higher, with zero warning that it fails an absolute product gate. That silent gap is
  exactly what the best-of-all wave's `A_plain` fell into, and is now closed everywhere
  `--select` runs, not only on the wave that found it.

Full transcripts: `~/tmp/selectfix_fair_before.txt`, `~/tmp/selectfix_fair_after.txt`; row-level
diff script inline in this session's transcript (parses both seed-group markdown tables,
diffs the `selectable` column by group key).

## 5. What this does NOT change

- The `balanced-2026-08-04` profile's F1-F8 floors themselves — untouched, still exactly the
  registered §8.1 set. This fix adds a SEPARATE veto axis plus an opt-in `A7r` extension; it
  does not fold C1-C6 into F1-F8 (the registry's own sketched `fix_path` — "add C5/C6/A7r ...
  to the balanced profile's floor set" — considered that shape; a hard veto is stronger and
  was what the calling directive specified: dial addressability is a HARD gate, not one more
  tradeable floor a high CID22 can outvote).
- `--profile balanced-2026-08-04`'s direct (non-`--select`) PASS/FAIL table — unaffected;
  this fix is scoped to the `--select` k-seed rule.
- Ensembles' own ranked pool (`balanced_composite` alone, no `selection_composite`) — the
  `selectable`/TSV column now also reflects the veto for consistency, but ensembles have no
  "winner" mechanism in this tool today, so nothing about their ranking changes.
- No historical bake, verdict, or board cell was re-scored or re-derived; this is a selection
  RULE fix, read against G-ADDR numbers other tools already produced.

## 6. Registry

`benchmarks/eval_annotations.json` entry `select-rule-blind-to-dial-contract-2026-09-06`'s
`fix_path` now points here; a new entry `select-rule-contract-prefilter-fixed-2026-09-06`
(`kind: "annotated"`, documentation-only) records the fix landing per the registry's
append-only discipline ("supersede with a new entry and point `fix_path` at the resolution" —
never edit a finding's own `reason` after the fact).

## 7. Files touched

- `zensim-validate/src/bin/freeze_check.rs` — the fix + 4 new tests (`gaddr_contract_fails`,
  `gaddr_codec_states`, `gaddr_measured_floor_counts`, `gaddr_effective_counts`,
  `gaddr_vetoes`, `gaddr_fail_reason`; `SelectRow`/`SeedGroupRow` new fields; `FloorBasis::Legacy`;
  `rank_pool`/`group_unselectable_reason`/`floor_count` updated; CLI + usage text).
- `benchmarks/eval_annotations.json` — fix_path update + new FIXED entry.
- `docs/MODEL_SELECTION_SCORECARD.md`, `docs/WAVE_PLAYBOOK.md` (step 6), `CLAUDE.md` (E.4
  paragraph) — pointers to this doc + the new `--floor-basis legacy`/veto behavior.
- `CHANGELOG.md` — Fixed entry under zensim-validate (internal tooling; not published, but
  the changelog tracks workspace-wide fixes per repo convention).
- `benchmarks/INDEX.md` — banner entry.
