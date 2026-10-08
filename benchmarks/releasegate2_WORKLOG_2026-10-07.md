# RELEASEGATE round-two fixes — 2026-10-07

Source fix `32dc619d3e5f06e17c5a21ef8c630e0f85ba3c13`, based on current
main `8a0e3aea984fa8a205e5c6bb62ac2c7b41028778` plus rebased copies of the
reviewed RELEASEGATE commits. The originals remain on their review bookmark.
Local-only; no push, real terminal read, exposure-ledger mutation, fleet action
or serving switch. The fresh round-two workspace is removed after verification.

## Reviewed bugs and dispositions

All seven findings are closed in source and synthetic tests:

1. Receipt replacement: hash one retained buffer, match it to the exact
   committed object, then parse those verified committed bytes. A deterministic
   replacement after lookup cannot redirect the authorized label input.
2. Metadata admission: receipt, authorization, repository pin, source/code,
   models, qualification/report artifacts, bindings, population/predictions,
   ledger/journal and output use the trusted preparation boundary before
   read/hash/parse. Lexical and resolved ancestry are both required. Original
   corpus stores, bank/terminal/T0 populations and shared protected markers
   refuse, including symlinks. Receipt fields cannot widen the trusted roots.
3. Adapter completeness: absolute string path,64-lowercase-hex SHA, supported
   format, three nonempty distinct column names, JSON rows_key and optional
   complete usecols validate before reservation without label access. Dedicated
   terminal-only input; selection/pairs indirection remains forbidden.
4. Correlation direction: opt-in `panel --signed-quality --input … --json`
   returns canonical signed Spearman, Kendall tau-b and logistic fitted Pearson.
   The fit's mapping is restored to increasing predicted-quality direction;
   this is neither absolute fitted PLCC nor substitution of raw Pearson. Raw
   signed Pearson is retained separately. Within-reference means reduce the
   returned canonical signed correlations, min3, with explicit excluded counts;
   per-type values retain sign. Existing panel modes/statistic math are unchanged.
5. Parser errors: results/library/CLI expose fixed error categories only.
   Actual synthetic pandas conversion errors containing the private canary
   cannot appear in the result, exception message or CLI output/traceback.
   Durable reservation and spent state survive the failed read.
6. E30: requires reports.E30 state completed and a hash-bound complete registered
   four-source/10-seed report with40 control entries, zero missing cells and no
   adoption rule. A negative removal-cost report passes this completion check;
   its numerical deltas never gate D1. Other pre-terminal gates still require pass.
7. RD map: correct owner scripts/v_next/rd_probe_analyze_2026-07-18.py;
   --interventions asserts historical SHA cd1098b4… at line554. Parametrizing and
   binding the final production model remains a prerequisite, not a claimed pass.

[Updated gate map](release_gate_map_2026-10-07.md) owns the receipt/report contract,
trusted metadata locations, correlation form and remaining release prerequisites.
It remains below30KB. No original14 test method/assertion changed (AST verified);
fixture preparation now supplies a completed, intentionally negative E30 report.

## Verification

Artifacts: `/mnt/v/output/zensim/releasegate2-2026-10-07/`; full logs and hashes
in its VERIFY.json. Private build scratch: `~/tmp/releasegate2/`.

```bash
TMPDIR="$HOME/tmp/releasegate2" \
ZEN_PANEL_BIN=/mnt/v/output/zensim/releasegate2-2026-10-07/bin/panel \
../scripts/run-heavy --mem 16G --jobs 8 -- just releasegate-tests
```

23/23 synthetic tests pass, including the original14. Reviewer probes cover
receipt replacement through a full invocation, direct/symlinked terminal/T0
receipt/authorization/source/binding aliases, protected committed pin lookup,
missing/malformed label pins, reversed metrics and real label-parser CLI leakage.
Open tripwires assert zero sentinel opens and no spent record on preflight
refusal. Signed statistics agree with SciPy to12 decimal places conditional on
canonical emitted logistic predictions; the nonlinear case distinguishes fitted
from raw PLCC. Paired10000-reference bootstrap parity remains unchanged.
Wrapper: `rc=0 27s | peak-RSS 0.42GiB | min-avail 31537MiB | peak-load 2.61`.

17/17 Rust panel tests pass, including existing batch/per-group/pairwise/rendering
contracts and the new signed-quality regression. Wrapper:
`rc=0 1s | peak-RSS 0.96GiB | min-avail 33579MiB | peak-load 2.86`.
Legacy36-case cross-language panel parity passes at1e-9, conditional on the
canonical logistic fit. Wrapper:
`rc=0 8s | peak-RSS 0.10GiB | min-avail 32389MiB | peak-load 4.36`.
These checks do not independently validate the logistic optimizer or qualify
production quality/performance.

CI-exact just clippy passed (`--workspace --all-targets --all-features
--exclude zensim-wasm-tests -- -D warnings`), then passed on the final source.
Full run: `rc=0 49s | peak-RSS 0.93GiB | min-avail 30190MiB | peak-load 6.37`.
Final incremental check: `rc=0 0s | peak-RSS 0.17GiB | min-avail 32502MiB | peak-load 2.58`.
Full fmt check, Ruff F and script lint pass (851 runnable scripts).
All heavy checks/builds were serialized with16GiB/8jobs caps.

The new test panel is a native debug artifact, SHA-256
`a669bec560f7d376ed14e022bec95f8259a62be270809e7d830bb4d5e80967b0`.
It supports the opt-in signed route; the old v39 panel does not. Future final
receipts must pin a rebuilt canonical panel with this interface; the prediction
scorer remains independently pinned.
No binary was committed and no production receipt/authorization was generated.
The real DATA_SPLITS exposure ledger is unchanged; the actual shared spent
journal remains absent. Final fits/gates, complete companion predictions,
original2000-stimulus mapping and coordinator authorization remain prerequisites.
