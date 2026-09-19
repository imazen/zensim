# Pointer: dvifm-screen2c-2026-09-19 evidence (not in git)

Record: `benchmarks/dvifm_screen2c_2026-09-19.{md,json}`
Protocol: `benchmarks/dvifm_screen2c_prereg_2026-09-19.md` (commit `4d6946db`)

All evidence >30 KB lives under:

    /mnt/v/output/zensim/dvifm-screen2c-2026-09-19/

Key artifacts:

- `recipe_q1.json` / `recipe_q2.json` — the executed recipes (sha256
  `448ac697…` / `320b732b…`; full hashes in each run's RESULT.json
  `identity.recipe_sha256`).
- `q1-human/` — Q1 run: `RESULT.json` (stage log, per-cell bake sha256s,
  argv, in-driver stamp records, `reuse.cells` — 40 byte-identical 2b
  bakes cited by hash), `SUMMARY.json`, `REPORT.md`, `cells/*.json`
  (120), `human_*.parquet` + `*.features.bin` + `_MANIFEST.json`
  (permutation leg seeds 6619/6620/6621), `audits/` (6 bakes × 33 dev
  pairs), `stamp_values/` (the two metadata values stamped per bake).
- `q2-codec/` — Q2 run: same layout, 100 cells, `audits/` (5 bakes × 5
  pairs); `_MANIFEST.json` carries `build`-time file sha256s and the
  codec permutation seeds (6619 train / 6620 eval).
- `segments/` — `codec-{train,eval}-{segment,admission}.json` (373/121
  rows, source-family-disjoint; six test-role families excluded) and
  `codec_panel_pinned.json` (input sha256 pins: family map, ladder +
  anchor manifests, per-codec pairs/label TSVs).
- `admit_2c.py` — the codec segment/admission builder.
- `specs/dvifm-local-fitted-final.json` — the pinned w986 DVIFM spec
  (sha `1b828398…`, same bytes as 2b).

Proxy caveat: every `q2-codec/` metric is against `score_ssim2`
SSIMULACRA2 PROXY labels, not human judgments.

Immutable evidence: do not delete; rename to `*.bak` if superseded.
