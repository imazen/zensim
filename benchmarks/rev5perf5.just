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
