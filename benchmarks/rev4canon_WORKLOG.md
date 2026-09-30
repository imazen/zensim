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

## Task B gates — rec64+c64 canon (2026-09-30)

Builds: oracle `tier_audit_features` + product `tier_audit_features` (release, /var/tmp/rev4canon/target{,-prod}).
Matrix script: /var/tmp/rev4canon/run_matrix.sh (14 configs x 12 TRAIN-role pairs, v3 tier).
Dumps: /var/tmp/rev4canon/dump{,-tier,-rev3,-prod}/. Logs: logs/matrix1.log, tier_parity_ext.log.

### Tier parity (new canon, all tiers vs v3)
- 14 pairs (12 review + crop256x256 + mosaic2x2_1024x768) x {v4x,v4,v3,scalar}:
  0/1825 slots differ on every pair — logs/tier_parity_ext.log.
  Sizes: 8x8,17x9,64x64,97x63,131x65,256x256,384x512,512x384,640x480,1024x768,2048x1536.

### Rev1-3 bit-identity gate
- prod.rev3 (ZENSIM_FORMULA_REV=3, no override) vs review's fix-commit binaries
  (/var/tmp/review-featcanon/rr/vec_review_fix_rev3_{v4x,v4,v3,scalar}):
  12 shared pairs x 4 tiers, 0 differing slots — logs/compare_refs.log section A + per-tier cmp.

### Product neutrality
- product-build prod == oracle-build prod bitwise, 12/12 pairs (dump-prod vs dump).

### Oracle-arm stability
- my c32_v3 == featacc's c32_v3 bitwise, 12/12 pairs (refactor preserved reviewed c32 candidate).
- my exact_v3 == featacc's exact_v3 bitwise, 12/12 (ruler unchanged).
- my prod_v3 == my c64_v3 bitwise, 12/12 (prod IS the c64 canon arm).
- c32-oracle vs old Rev4 canon (featacc prod_v3): differs on 10 families, 8110/21900 slots —
  canon::<LanesF32> is the reviewed *candidate*, NOT a bitwise replay of the era-2 mix
  (consistent with FEATACC's prod-vs-c32 nonzero families). See logs/diffcounts.log.

### Accuracy vs exact oracle (worst-of-12 max rel err, floor 1e-9) — family_tables.txt
new(prod) <= c32 on ALL 18 families; strict improvement on basic/peaks/masked/iw/v2/append/
csfw/gridblk/mapdev/z1max; ties elsewhere. prod == c64 column exactly.

### Slots changed vs old Rev4 canon — logs/diffcounts.log
new canon differs from old Rev4 on 18119/21900 slots (12 pairs); per-family table in log.

## Task C1 — XYB body-vs-tail (2026-09-30)
- Structural state: every production XYB converter entry threads `revision`
  (`convert_source_to_xyb_into_slices_chunked` -> `*_at_revision`), so Rev4
  runs `srgb_xyb_canon`/`linear_xyb_canon` — one scalar per-pixel form,
  position-independent by construction (no SIMD tail path exists at Rev4).
  The two raw `srgb_to_positive_xyb_planar_into` callers in feature_v2.rs
  (23486, 25197) are inside `mod tests` derivation-table helpers only.
- New test `color::tests::rev4_xyb_body_tail_bits_identical_across_tiers`
  (#[ignore], token-permutation): 3 Rev4 entry points x ~1504 colours x
  widths {9,13,17,23,29,31,33,37,41} x positions — all bits equal per colour
  vs its full-chunk reference, on all 10 x86 token permutations.
  Output: "REV4-XYB-BODY-TAIL permutations=10 all position-identical"
  (log: logs/xyb_tail_test.log)
