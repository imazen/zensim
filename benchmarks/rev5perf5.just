# Byte-budget candidates; heavy recipes require run-heavy and the shared lock.
set working-directory := '..'
export TMPDIR := env('HOME') / 'tmp'

rev5perf5-baseline-checks:
    cargo fmt -p zensim -- --check
    just clippy
    just lint-scripts

rev5perf5-thread-parity:
    cargo test -p zensim --all-features ordered_batches_match_serial_with_reused_strided_scratch -- --nocapture

rev5perf5-tests:
    cargo test -p zensim --all-features

rev5perf5-build root label:
    just costcmp-build > "{{root}}/provenance/build-{{label}}.jsonl" 2>&1
    just speedq-freeze "{{root}}/provenance/build-{{label}}.jsonl" "{{root}}/provenance/instrument-{{label}}"

rev5perf5-parity binary dest:
    python3 scripts/demos/speedq_run.py parity --binary "{{binary}}" --dest "{{dest}}" --collect-legacy-failures

rev5perf5-python-tests:
    python3 -m unittest discover -s scripts/demos -p 'test_speedq*.py'
    python3 -m unittest discover -s scripts/demos -p 'test_costcmp*.py'

rev5perf5-checks:
    cargo fmt -p zensim -- --check
    cargo fmt --manifest-path zensim-bench/Cargo.toml -- --check
    just clippy
    just costcmp-clippy
    just lint-scripts
    just -f benchmarks/rev5perf5.just rev5perf5-python-tests
    cargo check -p zensim --no-default-features
    cargo check -p zensim --no-default-features --features feature-regime-v2
    just -f benchmarks/rev5perf5.just rev5perf5-tests

rev5perf5-budget mode root:
    python3 scripts/demos/costcmp_run.py {{mode}} --budget-grid --budget-binaries "{{root}}/provenance/budget-binaries.json" --binary "{{root}}/provenance/instrument-cap128" --dest "{{root}}/budget/{{mode}}" --parity "{{root}}/budget/parity/PREFLIGHT_PASS.json" --analyzer "{{root}}/provenance/paired-rounds-analyzer" --lock /home/lilith/tmp/zensim-paper/rev4/heavy.lock

rev5perf5-progress root:
    python3 scripts/demos/costcmp_run.py status --budget-grid --budget-binaries "{{root}}/provenance/budget-binaries.json" --dest "{{root}}/budget"

rev5perf5-rss-under-load root:
    python3 scripts/demos/costcmp_run.py rss --rss-under-load --budget-grid --budget-binaries "{{root}}/provenance/budget-binaries.json" --binary "{{root}}/provenance/instrument-cap128" --dest "{{root}}/budget/rss" --parity "{{root}}/budget/parity/PREFLIGHT_PASS.json" --lock /home/lilith/tmp/zensim-paper/rev4/heavy.lock

rev5perf5-byte-accounting:
    cargo test -p zensim --all-features rev5_job_budget_handles_exact_fit_and_overflow -- --nocapture

rev5perf5-rss-report-gates root:
    PYTHONPATH=scripts/demos python3 -c 'from pathlib import Path; from types import SimpleNamespace; import costcmp_scaling_report; costcmp_scaling_report.report(SimpleNamespace(raw_dir=Path("{{root}}"),out_json=Path("{{root}}/rss-report.json"),out_md=Path("benchmarks/rev5perf5_rss_2026-10-09.md"),budget_grid=True,rss_only=True))'
    python3 scripts/demos/check_costcmp_scaling_report.py --budget-grid --rss-only --raw-dir "{{root}}" --dest "{{root}}/rss-report-controls"
    just -f benchmarks/rev5perf5.just rev5perf5-python-tests

rev5perf5-report-gates root:
    PYTHONPATH=scripts/demos python3 -c 'from pathlib import Path; from types import SimpleNamespace; import costcmp_report; costcmp_report.report(SimpleNamespace(raw_dir=Path("{{root}}"),out_json=Path("{{root}}/report.json"),out_md=Path("benchmarks/rev5perf5_scaling_2026-10-09.md"),budget_grid=True))'
    python3 scripts/demos/check_costcmp_scaling_report.py --budget-grid --raw-dir "{{root}}" --dest "{{root}}/report-controls"
    just -f benchmarks/rev5perf5.just rev5perf5-python-tests

rev5perf5-archive root mirror:
    python3 scripts/demos/rev5perf_archive.py "{{root}}" "{{mirror}}"

rev5perf5-candidate root mib:
    python3 scripts/demos/rev5perf5_candidate.py candidate --root "{{root}}" --mib {{mib}}
    just -f benchmarks/rev5perf5.just rev5perf5-build "{{root}}" cap{{mib}}
    just -f benchmarks/rev5perf5.just rev5perf5-thread-parity
    just -f benchmarks/rev5perf5.just rev5perf5-parity "{{root}}/provenance/instrument-cap{{mib}}" "{{root}}/cap{{mib}}/parity"
    just -f benchmarks/rev5perf5.just rev5perf5-compare "{{root}}" {{mib}}

rev5perf5-compare root mib:
    PYTHONPATH=scripts/demos python3 -c 'from pathlib import Path; from rev5perf_report import compare_bits; from speedq_run import write; write(Path("{{root}}/provenance/parity-cap{{mib}}.json"), compare_bits(Path("{{root}}/uncapped"),Path("{{root}}/cap{{mib}}")))'

rev5perf5-inventory root:
    python3 scripts/demos/rev5perf5_candidate.py inventory --root "{{root}}"

rev5perf5-budget-parity root binary label:
    python3 scripts/demos/rev5perf5_candidate.py budget-parity --root "{{root}}" --binary "{{binary}}" --label "{{label}}"

rev5perf5-select root mib:
    python3 scripts/demos/rev5perf5_candidate.py select --root "{{root}}" --mib {{mib}}

rev5perf5-legacy-report raw root:
    PYTHONPATH=scripts/demos python3 -c 'from pathlib import Path; from types import SimpleNamespace; import costcmp_scaling_report; costcmp_scaling_report.report(SimpleNamespace(raw_dir=Path("{{raw}}"),out_json=Path("{{root}}/provenance/legacy-scaling-replay.json"),out_md=Path("{{root}}/provenance/legacy-scaling-replay.md"),budget_grid=False))'
