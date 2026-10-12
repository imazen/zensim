# REV4CANON worklog — final canonical Rev4 feature arithmetic

> Sections "Session log" (Task A) were written by the Devin SWE-2 lane. Its Task B / C1 / C3 numbers were
> NOT carried over: the lane was taken over by a Claude session on 2026-09-30, which re-measured every reported
> number on fresh builds (section "Takeover" below). Times are Mountain Time (`TZ=America/Denver date`).

Lane: `quarantine/devin/rev4canon`, workspace `~/work/zen/zensim--rev4canon`
(jj workspace `rev4canon`), base `main@origin` = `0badd366`.
Brief: `/home/lilith/tmp/zensim-paper/rev4/REV4CANON_brief.md`.
Rules: `DEVIN_COMMON.md` + `LANE_PREAMBLE.md` + `QUARANTINE.md`.
`CARGO_TARGET_DIR=/var/tmp/rev4canon/target`; scratch under `/var/tmp/rev4canon/`;
manifest `/home/lilith/tmp/devin/rev4_rev4canon_manifest.tsv`. Heavy commands go
through `~/tmp/devin/heavy`. Never push; the local quarantine bookmark is the
only bookmark this lane moves.

## Session log

### 2026-09-30 22:34Z — workspace claim + chain recreation (Task A.1)

- Wrote `.workongoing` (`devin-rev4canon`).
- `jj duplicate '3954d541::13587bed' -d @` → `f02ba22d` (dup of `3954d541`,
  change `xxwsrwzk`) + `96474324` (dup of `13587bed`, change `orqyortv`),
  auto-merged onto `0badd366` with 0 conflicts (as the re-review measured).
- Dropped the empty lane WIP commit (`afa834c2`, abandoned — it was empty);
  new empty `@` = `ymszzsxz` on top of the chain.

### 2026-09-30 — Task A.2/A.3: R1 `#[non_exhaustive]`, R3 doc wording, CHANGELOG resolution

- `FormulaRevision` → `#[non_exhaustive]` (`feature_defs.rs`); the one
  cross-crate exhaustive match (`zensim-validate bake_verdict` revision map)
  got a refusing `_` arm.
- R3 verbatim wording fixes in the `Rev4` doc comment.
- CHANGELOG `QUEUED BREAKING CHANGES … AWAITING USER APPROVAL` → `BREAKING —
  approved 2026-09-30, option 1`.

### 2026-09-30 22:40Z — Task A gates



## Takeover (Claude, 2026-09-30 ~20:40 MT onward) — re-measured numbers

All dumps, scripts and logs live under `/var/tmp/rev4canon/` (not committed; sizes). Coordinator decisions D1-D9 are in
`~/tmp/zensim-paper/rev4/REV4CANON_decisions.md`.



### C1 — XYB body vs tail
`color::tests::rev4_xyb_body_tail_bits_identical_across_tiers`: "REV4-XYB-BODY-TAIL permutations=10 all position-identical", child 13.35 ms. Runs in the
normal suite via self-re-exec (no `#[ignore]`; an in-process permutation raced 12 lib tests).

### C3 — tailhist
- Root cause: `TailEdges::bin` read `f64::to_bits`; a sign-bit-set value (tiny negative `art`/`det` from f32 rounding, NaN) sorted above every edge and went to the TOP
  bin; p99 then emitted the top edge (1.27) above the exact max (0.998). Fix: `bin_folded` folds non-positive/NaN into bin 0, applied at Rev4 only (`TailAccum::fold`).
  Rev1-Rev3 keep the legacy bins byte for byte (registry + Known Bugs entry in `CLAUDE.md`).
- Failing-then-passing: with the fold removed `rev4_tailhist_bin_folds_sign_bit_values` fails ("neg tiny: bin(-0.000000000001) must fold to 0"); with it, passes.
- Invariant p95 <= p99 <= max, 14 pairs x 48 cells = 672: Rev4 with fold 0 violating; fold removed 1 (crop64x64 det); old-canon binary 1; Rev3 (legacy) 1.
- Top-edge saturation with the shipped edges (no recalibration, decision D2a), % of cells per map, 12 cells/pair (ssim/art/det/mse), from the
  diagnostic ladder logs (`c2/diagfix`, `analyze_c2_v2.py`; the diagnostic build is not committed):
  kadid_train (257 pairs) 0.81/0/0/0; tid2013 (273) 0.24/0/0/0; konfig_train (327) 0/0/0/0; konjnd_bpg_train (260) 0/0/0/0; safesyn (257) 0.58/0/0/0;
  cid22_train (63, extra, not part of the gate) 0/0/0/0. Max 0.81% (limit 1%). Full-bank saturation is re-measured after re-extraction (REEXTRACT B3).
- Registry: era `c3negfold` (Proposed, commit "-"), Rev4 era tokens end `..., "tiercanon", "c3negfold"`; attached to the 8 Bin signals only.
  The `feature_set_id` era is CALLER-supplied (`Request::with_era_label`), so a producer must pass `tiercanon_c3negfold` (token charset `[a-z0-9_]`).
  Test `rev4_manifest_carries_c3negfold_and_era_labels_make_distinct_ids`. Auto-deriving the id era from the active eras is a follow-up, not done.
- dev4 (stretch, report-only per D5): see the report; no code change.
