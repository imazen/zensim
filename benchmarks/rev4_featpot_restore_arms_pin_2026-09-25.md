# Restore-cuts arms — input pin (2026-09-25, before any label read)

**POTENTIAL — ceiling, not a model score.** Pins the inputs named in `rev4_featpot_restore_cuts_amendment_2026-09-25.md` (commit `72b35b43439b`, 2026-09-25T15:57:29Z). Written 2026-09-25T15:59:05Z; this lane has read no label of any restore arm.

- `benchmarks/rev4_featpot_restore_arms_2026-09-25.json` (sha256 `6b920f2a619a105acea5915f563b57a968a1f635ac6598a60d7ccea41529251c`): family id ranges, arm column sets (A1, A1m, A1w, B2, B2m, B1, B1s, C8n, ALL), per-set sidecar pins (manifest and per-family file sha256, row counts, feature_set_id, build commit `ec5b1821a4c6`, binary sha256) for the nine D1/D2 populations, and the sha256 of the drop-list file.
- `benchmarks/rev4_featpot_restore_arms_droplists_2026-09-25.json` (sha256 `5addfec54c3ceaeba89925784cd6ef5f98beb33fb9facc160f3e7368a7026f72`): the 55 label-free A1w drop lists (8 D1 sets x {full, o0..o4} = 48, plus 7 D2 folds). **Correction to the amendment text:** it says the drop lists live in the arms JSON; they live in this companion file (the arms JSON exceeds the 30 KB record limit with them inline, and points to the companion by path and sha256). The rule is unchanged.
- Generator `scripts/rev4_featpot/restore_arms_pin.py`: reads feature columns only (`ref_basename` and `f0..f943` of `/var/tmp/rev4-featpot/admitted/POT_<set>_rev3_944.parquet`) and the restore manifests; the admitted tables contain no label column. Input hashes are in the arms JSON (`inputs`).
- Join contract: restore sidecars join to the admitted Rev3 tables strictly by `pair_key`; a missing or duplicate key fails the arm.
- Not yet pinned: Part B sidecars (C7 `features__csfw_dvifm.parquet`, C8), so B1, B1s, C8n and ALL stay MISSING.
