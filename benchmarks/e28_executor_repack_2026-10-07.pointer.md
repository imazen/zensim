# E28 v36 executor admission artifacts

Artifacts/evidence: `/mnt/v/output/zensim/e28-executor-repack-2026-10-07/`.
Working evidence: `/var/tmp/e28/ready4-v36/`. Jobset: `fitv2e28b-20261007`.
Fit-script producer: `8283d5186887b586a7554d9f3b25450e3ddb5ba0`.
Profile: `ce26183223f8f05979d7a95ee7916c3e0cc29849`.
Numerical producer unchanged: `5b2cc589429546b8c569509bc4fea3e04c5ad2af`.
Data producer unchanged: `a06541fd44b9570c4c8a2a06aeb4ce1c45b51955`.

| Artifact | SHA-256 |
|---|---|
| image-context/program.tar.gz | `9e7959f0e26f311c315d4a2b9a077131a1a0fe1e20a79b12c13e28abde8eeb23` |
| e28-lodo.tar.gz | `189be77e73bba87545e188009f483e20d42f67f2dc2d57858cf26d6b0147c897` |
| fit-manifest-fitv2e28b-20261007.json | `1a1a622a4563e4c46fa7bbab5f6bf68199e93721f26d3d96a88455d4ac362f06` |
| prepared-image.tar.gz | `e59d3dfe8cca959faeb380a1ffdb39afcfbcbae74e37b450372d11b9dcfe8a07` |
| frozen admission inventory | `71067920cbe93f05ef12d13317e4332af8f2ec73ca37b403d8bae670e30afe47` |

Image: `ghcr.io/imazen/zenfleet-worker:fit-e28-s2recipe-v36-w8dc42d4e`.
Local ID: `sha256:1cc1a8ee7bf5e1488838ea6b804511a2098854c1524333789f11c16cb9b9d813`. No registry/R2 publication or enqueue.

Executor evidence: `EXECUTOR_SMOKE_PASS.json`, `EXECUTOR_SHORT_PATH_PASS.json`,
`EXECUTOR_SHORT_OUTPUT.tar.gz`, `EXECUTOR_SHORT_TIME.txt`,
`REAL_s2o_PATH_PASS.json`, `REAL_s2m_PATH_PASS.json`, `REAL_s2o_TRAIN.log`,
`REAL_s2m_TRAIN.log`, `e28-v36-*.log` and matching container inspect files.
Production scripts and declared argv are unchanged; bounded owner constants
are supplied by the opt-in test hook in
`scripts/tests/e28_executor_smoke_site/sitecustomize.py`. Test driver:
`scripts/tests/e28_executor_smoke.py`; caller recipe: `e28-executor-image-smoke`.

Other checks: `REVIEWER_PROBE_PASS.json`, `ALL_FOLD_ADMISSION.json`,
`PROGRAM_INVENTORY_CHECK.json`, `PAYLOAD_DELTA.json`, `SHORT_SMOKE_PASS.json`,
`CAPS_CHECK.json`, `READ_ONLY_GATE_PASS.json`, `LAUNCH_GATE_CHECK.log`.
The program differs from v35 only in recipe and build metadata; data is
byte-identical. The bounded smoke and first-epoch checks do not constitute
a complete new 120-epoch trajectory or model qualification.
