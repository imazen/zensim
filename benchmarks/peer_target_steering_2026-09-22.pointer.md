# Bulky data for peer_target_steering_2026-09-22

Bulk outputs live outside the repository (shared disk ruling — `/home` nearly
full):

- `/var/tmp/paper-target/runs/peer-20260923T055844Z/` — the peer run:
  `ADMISSION.json`, `COMMANDS.json`, `COMPLETE.json`, `REPRODUCTION.json`,
  `EXCHANGE_RATE.json`, `MACHINE_STATE.jsonl`, `train.json`, `validate.json`,
  `compositions-rev1.json`, `demo_matrix` (binary, sha256
  `11138d4c2c5648fb5e4aba0e121fc57adc851bb00fda13240c457ae6a8cf363b`),
  `{rate,fit,eval,analysis}.log`, `rate/`, `fit/`,
  `eval/{measurements.jsonl,bounds.jsonl,bounds/,INPUTS.json,
  analysis_summary.json,analysis_summary.md,*.jxl artifacts}`.
- `/var/tmp/paper-target/reanalysis/{rev3,r915,rev1}/` — Sept-14/15 runs
  re-analyzed by the same analyzer (each with `COMPLETE`, `INPUTS.json`,
  `analysis_summary.{json,md}`, `measurements.jsonl`, `bounds.jsonl`).
- `/var/tmp/paper-target/analyzer-check/` — analyzer smoke run.
- `/var/tmp/paper-target/src/` — jxl-encoder snapshot @`ee80c785` (sibling tree
  left dirty upstream; archived for the record).
- `/var/tmp/paper-target/{build_and_run.sh,run_peer.py,reproduce_check.py,
  record_method_draft.md}` — lane driver scripts.

Record + companion JSON: `benchmarks/peer_target_steering_2026-09-22.{md,json}`
(committed). The JSON carries run/binary/analysis sha256s and the full
per-budget, per-position, per-kind and cross-model pairing tables.
