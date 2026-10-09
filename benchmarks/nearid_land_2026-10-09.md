# NEARID landing validation

The NEARID chain originally ending at `4828cc682e5a7c3c94d8489f1c73f184fd0f8063` is rebased onto STEERFIX at `ed8eabbcb0b017081c6b10f7ab37f0f6287e2ef9`. The rebased measurement/report tip is `123f3fc99afd098278772503f34dc5661d84943d`. Only the reporting script required conflict resolution. Both STEERFIX and NEARID helpers retain their original AST bodies, and STEERFIX's CLI remains unchanged within the combined dispatcher. The merged script SHA256 is `5b4514bc503d5d473c6a76a4eed2e63ba217c52c9db15fb2f9e32b9d59d74736`.

`scripts/reproduce_nearid_report.py` exercises the merged CLI against the three frozen scored-row files. All 1,944 rows reproduce `SUMMARY.json`, `scores.tsv` and `SUMMARY.md` byte for byte. Figures are regenerated; SVG timestamps are excluded from equality checks. This checks report reproduction, without rescoring models or opening pixels, labels or corpora. The independent NEARID review's score replay remains the evidence for score reproduction.

The command, run from the NEARID workspace, was:

```sh
TMPDIR=/home/lilith/tmp/nearid-land \
CARGO_TARGET_DIR=/home/lilith/tmp/nearid-land/target \
MPLCONFIGDIR=/home/lilith/tmp/nearid-land/matplotlib \
flock /home/lilith/tmp/zensim-paper/rev4/heavy.lock \
  /home/lilith/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
  just --justfile benchmarks/nearid.just --working-directory . \
  land-validate /mnt/v/output/zensim/nearid-land-2026-10-09
```

Scoped Rust format check, both report controls, `just lint-scripts` (918 scripts), and CI-exact `just clippy` all passed. Clippy uses `--workspace --all-targets --all-features --exclude zensim-wasm-tests -- -D warnings`. Cargo reported an existing future-incompatibility notice for proc-macro-error2 2.0.1. The wrapper measured `done rc=0 78s | peak-RSS 0.94GiB | min-avail 39542MiB | peak-load 15.30`; this is the combined validation process tree, not a model memory measurement.

Evidence lives at `/mnt/v/output/zensim/nearid-land-2026-10-09/`, mirrored to `/mnt/tower/output/zensim/nearid-land-2026-10-09/`. `MERGE_PRESERVATION.json` records the AST comparison; `reproduced/REPRODUCED.json` pins all three input and report hashes; full validation and per-check logs are retained. `MANIFEST.json` and `ARCHIVE_VERIFIED.json` record the all-file SHA256 mirror check. The original measurement evidence remains at `/mnt/v/output/zensim/nearid-2026-10-09/` and its NAS mirror.

No model bytes, serving defaults or test expectations changed. The shared main bookmark was not moved, and nothing was pushed. The owned landing Cargo target is removed after validation and verified evidence mirroring.
