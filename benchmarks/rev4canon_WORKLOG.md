# REV4CANON worklog — final canonical Rev4 feature arithmetic

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

- `just api-doc` (heavy, CARGO_TARGET_DIR=/var/tmp/rev4canon/target):
  `test public_api_surface_docs_are_current ... ok`, rc=0.
  `docs/public-api/zensim.internal.txt` now lists
  `#[non_exhaustive] pub enum feature_v2::FormulaRevision`.
- `cargo check -p zensim --all-targets`: rc=0 (2 pre-existing dead-code
  warnings in lib test: `DVIFM_BLOCK_F32`, `to_f32`).
- `cargo check -p zensim-validate --bin bake_verdict --bin tier_audit_features`: rc=0.
- `cargo test -p zensim --release --test featcanon_tier_parity --test featcanon_rev4_contract`:
  `featcanon_rev4_contract`: `test result: ok. 7 passed; 0 failed`;
  `featcanon_tier_parity`: `test result: ok. 3 passed; 0 failed`.
  Log: `/var/tmp/rev4canon/logs/task_a_tests.log`.
