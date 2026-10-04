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
