# Pointer: dvifm-screen-2026-09-19 evidence (not in git)

Record: `benchmarks/dvifm_screen_2026-09-19.{md,json}`
Protocol: `benchmarks/dvifm_screen_prereg_2026-09-19.md`

All evidence >30 KB lives under:

    /mnt/v/output/zensim/dvifm-screen-2026-09-19/

Key artifacts:

- `screen2-lap/run/`, `screen3-local/run/`, `screen4-fitted/run/` —
  per-screen `RESULT.json`, `SUMMARY.json`, `REPORT.md`, `audits/`,
  per-fit trainer logs and `*.bin.spec.json` (full argv + binary
  identity).
- `mech/run/` — mechanism-check run (subset extraction, fits, audit).
- `probe/run/`, `probe3/run/`, `probe4/run/` — 160-epoch convergence
  probe logs per arm (plateau rule evidence for E=40).
- `cache-lap.bin`, `cache-local.bin` (8.25 GB each, 11,125 rows) +
  `.index.jsonl` + `cache-*-features.csv` + `.manifest.json` —
  the training-only DVIFM block-statistics caches (18 f32/block,
  level-major, block-row-major).
- `specs/dvifm-{seed,seed-local,lap-derived,local-derived,
  local-fitted,local-fitted-final}.json` — every constants spec used,
  with provenance blocks; sha256s in the result JSON.
- `tools/` — `dvifm_cache.py` (reader), `derive_constants.py`,
  `fit_params.py` (Adam fit + replay parity gate), `cache_parity.py`.
- `probe4/run_probes.sh` — probe driver.

Immutable evidence: do not delete; rename to `*.bak` if superseded.
