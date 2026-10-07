# E28 prepared artifacts — 2026-10-07

Registration: [`e28_ssim2recipe_registration_2026-10-07.md`](e28_ssim2recipe_registration_2026-10-07.md), `82c9af81`.
Evidence and limits: [`E28_WORKLOG.md`](E28_WORKLOG.md).

Canonical local evidence: `/mnt/v/output/zensim/e28-2026-10-07/`.
NAS mirror: `/mnt/tower/output/zensim/e28-2026-10-07/`.
The final artifact set is `v32/` under that evidence root; superseded sets remain
archived for provenance. `COPY_MANIFEST.json` records each file's hash and size;
`SHA256SUMS` verifies the transferred payloads. No R2 upload or registry push
was performed. The lane's original working evidence is `/var/tmp/e28/v32/`.

| Artifact | SHA-256 |
| --- | --- |
| `v32/image-context/program.tar.gz` | `3d21e77fbfa9948479261578c5ff885e0ed0c78e58089ce9128f3bf764c2930e` |
| `v32/e28-lodo.tar.gz` | `ac8a52ceaa57b936ddd920595618d90f5d24a9d4910ab23d055879a83c738c51` |
| `v32/fit-manifest-fitv2e28-20261007.json` | `f5274de3da22c6bcfb93526461cc3b643cabe85a7ef9132ceeaf2cecd618d75e` |
| `v32/prepared-image.tar.gz` | `4be129671c1a42dea27e8e8f34a3dda554b63ab96bf73f734e9a265b2e309294` |

Prepared local tag: `ghcr.io/imazen/zenfleet-worker:fit-e28-s2recipe-v32-w8dc42d4e`.
Local Docker image ID: `sha256:a4dbeffa54a35d5a8396050222309353c84c970fcda305ea5e3642e66de8df0c` (not a published registry digest).
Source binary producer: `5b2cc589429546b8c569509bc4fea3e04c5ad2af`.
Curated-view/Python preparation producer: `be40ab351adc92542cc74e23ee532b7d487ca80d`.
The base image and full compiler/dependency records are in `v32/build_meta_e28.json`.
Python payload pins are in `v32/PYTHON_PINS.json`; the pre-fit teacher/grouping
pins are committed JSON in this directory and included in the program.

The declaration is 2 arms × 5 fixed source-held-out folds × 10 seeds = 100 cells,
with 420 kept columns, H128/head N, 120 epochs × 50,000 draws and final epoch 119.
Both local KADID/seed-0 full smokes passed. Each used one CPU and a 6 GiB cap,
and its recorded sampler digest matched the actual live training stream.
The 46-parameter NM smoke is nonconverged and remains POTENTIAL; no bake ships.
The complete scientific decision has not run.

The gate in `v32/e28_launch.sh` refuses without an explicit coordinator
`LAUNCH_AUTHORIZATION.json`; `AUTHORIZATION_REQUIRED.json` is only a false-valued
schema template. Local zenmetrics profile change `kplzuzlklxslosxqnyoovosutxwusntz`
(`4e4db22fed6b984d3e42399db50225fce0209343`) must be landed/published by the
coordinator before launch. The source lane likewise remains local for review.
