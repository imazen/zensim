# Rev5 pipeline worklog (coordinator side: banks, tables, fits)

Spec: `rev5_spec_2026-10-04.md`. Implementation lane: SWE-2, workspace `../zensim--rev5`, bookmark `quarantine/swe2/rev5`
(brief `~/tmp/zensim-paper/rev4/REV5F32_brief.md`). Times UTC.

## 2026-10-04 11:13 — bank driver and the subset-request identity gate

`scripts/rev4_featpot/rev5_bank.py` generalises the Rev4 re-extraction assembler (`/var/tmp/reextract/assemble_set.py`, never
committed): it asks the extractor only for `--restore-cuts basic,peaks,v2` (slots f0..227, f372..719), writes every other column
as NaN (absent, never a measured-looking zero), refuses a non-finite requested slot, and records the requested ranges in the
manifest (schema `rev5-featbank-v1`).

Gate at Rev4 (extractor sha256 `8c6f4c03…`, build 259045b0, era `tiercanon_c3negfold`, konfig_train first 40 pairs, native tier,
4 threads): `verify-against /var/tmp/rev4-featbank-r4` → 576 requested slots, **0 differing** (bitwise f64), absent columns all
NaN. So requesting the Rev5 subset does not change its values at Rev4.

Finding for the lane (addendum appended to its brief): the same request's `feature_set_id` at Rev4 is
`basic+peaks+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype+gmsbank+mapdev+z1max+gmsnative+dvifmgate…#f30d13cf`
— the research plan computes families nobody asked for. At Rev5 that request has to compute and declare only basic + peaks + v2.

## 2026-10-04 11:40 — instrument tables and fitter accept Rev5 banks

* `v2c_wide.py --revision 5 --bank <rev5 bank> --expect-era/--expect-fsid/--expect-binary/--expect-build --pad-to 1853`:
  a `BankProfile` replaces the hard-wired Rev4 pins. At Rev5 the requested slots (f0..227, f372..719) must be finite and every
  other slot NaN (both refused otherwise); Rev4 sidecar extras and the aux family are refused (they would mix revisions);
  tables are NaN-padded to the Rev4 tables' width so kept columns get the same first-layer initial weights. Rev4 behaviour is
  unchanged (`REV4_PROFILE` carries the old pins).
* Fitter (`v2_lodo_mlp.py`, `v2_confirm_fit.py`): `refuse_nonfinite_kept` refuses any keep list that reads a non-finite
  column in any training or validation table; `predict` densifies the bake (`bake_dial_refit densify`, bit-identical by its
  gate) when the table declares formula revision ≥ 5, because an identity-width bake multiplies NaN absent slots by zero
  weights and scores NaN. The densify helper moved from `external_sets.py`'s copy into `v2_common.dense_bake`.
* Tests: `Rev5Tables` (load + pad + bit-exact requested columns, five refusals, fitter guard, revision probe); full
  `test_v2c` 69 tests OK.

## 2026-10-04 11:55 — one extractor contract for E15 and the external sets

`e14_kadis_ordinal.extractor(revision)` is now the single owner of the extractor contract (binary + sha256, era, arguments,
environment, width, requested slots). Rev4 returns the existing pins unchanged; Rev5 reads `/var/tmp/rev5-extract/build_meta.json`
(written when the Rev5 extractor is built) and requests `basic,peaks,v2`. `apply_contract` keeps requested slots (must be
finite) and turns the extractor's structural zeros elsewhere into NaN. `e15_coverage.py extract|table --revision 5` and
`external_sets.py extract|table --revision 5` use it; their table manifests now carry the extraction's own feature-set id and
formula revision. With `--root <rev5 root>` every output lands under that root (the Rev5 root needs `e15/selection.parquet`
copied from the Rev4 root: same pairs). Test `ExtractorContract`.
