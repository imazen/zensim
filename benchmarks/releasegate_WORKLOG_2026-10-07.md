# RELEASEGATE implementation evidence — 2026-10-07

Local implementation commit `148dea93ecb4b9acde89f5ddd99f8fbb185fa550`,
base main `b47599be`. No source push, fleet command, training, production
assessment, actual KADID TERMINAL label read or profile change.

[The gate map](release_gate_map_2026-10-07.md) covers the production scorecard,
Rev5 implementation/evaluation list, current D1–D3 decisions and D2 contract.
It names measuring owners, explicit inputs, existing numerical rules, status,
protected-data boundary and unresolved owner decisions. The current machine
qualification's incomplete scorecard coverage remains explicit. It includes
the pinned registered harvest and densify → quantize → TRAIN calibration →
complete-composition verdict chain; existing instruments are not gate passes.

[The read harness](../scripts/rev4_featpot/kadid_terminal_read.py) requires
coordinator authorization and a locally committed byte-bound receipt. It
checks full final-model budget/seeds/D1 input hashes via the canonical model
inspector, every non-label binding/gate artifact and ordered population before
reserving exposure. Its shared exclusive journal and locked/fsynced ledger
reservation prevent retries after loss of output or any failed confirmation.
It reuses `v2c_labels` and the canonical Rust `panel`, without adding inference
or statistics implementations. Added jj workspaces resolve the underlying Git
store with `jj --ignore-working-copy git root`; committed pin checks do not
snapshot payloads. A real added-workspace public registration read passed.

Synthetic validation used a real Rust panel, fixture inspector/model responses,
temporary local committed pins and synthetic authorization/ledger/labels only.
It covers both PASS and FAIL, negative signed orientation, the full metric and
scatter reports, loss of output, spent journal, missing authorization, changed
receipt/uncommitted pin/input, failed gate, decoded short fit, wrong seed/epoch/
population/bootstrap budget, malformed label adapter, direct protected binding
and symlinked source/binding. Python `io.open` and `builtins.open` tripwires
require zero synthetic sentinel opens for preflight refusals. A third full
10000-reference-resample fixture has nonzero paired SE; independent SciPy
Spearman resampling agrees with canonical Rust deltas and SE to12 decimal
places. No test expectation or release bar changed.

Final command:

```bash
TMPDIR=/home/lilith/tmp/releasegate \
ZEN_PANEL_BIN=/mnt/v/output/zensim/shippath11-2026-10-07/bin/panel \
  ../scripts/run-heavy --mem 16G --jobs 8 -- just releasegate-tests
```

All14 tests passed. Full log:
`~/tmp/releasegate/terminal-tests-final-14.log`.
Wrapper: `rc=0 14s | peak-RSS 0.17GiB | min-avail 46324MiB | peak-load 2.20`.
This is a synthetic software gate, not production quality or timing qualification.

Reused panel SHA-256:
`e9a8f460c76d729a898a508c900a17f6998ad2dcc1282e4d7ea5d841c1b7769d`.
Registered program inspector SHA-256:
`c4fe929ab8d115dc45fe39eae97665abed7e870c9ea50b53bd3be4c94f58383f`.
The inspector was additionally used on the retained SHIPPATH11 TRAIN-only
short model to inspect actual repro field shapes; its epoch001/2×128 budget
remains a refusal for this registered terminal route.

CI-exact `just clippy` passed with a private Cargo target directory,16GiB/8jobs:
`rc=0 55s | peak-RSS 0.93GiB | min-avail 44790MiB | peak-load 5.75`.
`cargo fmt --all --check`, scoped Python Ruff F checks and script lint passed.
No Rust source/API/format bytes changed. Full release corpus/platform checks
and the existing validation registry owner decision remain in the gate map;
this lane does not claim to resolve them.

Actual terminal authorization and spent journal were not created. The real
DATA_SPLITS exposure ledger is unchanged. Future execution must record the
read there through the harness, with final pre-read receipt and result hashes.
Production seed/ensemble choice, complete-companion prediction receipt,
original2000-stimulus mapping, actual release evidence and authorization remain
prerequisites; the delivered1952-key features cannot silently reduce D2's
registered population. An ensemble requires explicit member inspection support
before authorization; the current registered receipt handles one final bake.
