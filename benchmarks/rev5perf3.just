# Packed row-prefix regression and frozen-output qualification.
set working-directory := '..'

rev5perf3-regress:
    cargo test -p zensim --all-features --lib source_rows_tests -- --nocapture

[positional-arguments]
rev5perf3-lint root:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo fmt -p zensim -- --check > "$1/provenance/fmt-final.log" 2>&1
    just --justfile benchmarks/rev5perf3.just rev5perf3-regress > "$1/provenance/padded-rows-final.log" 2>&1
    just clippy > "$1/provenance/clippy-final.log" 2>&1
    just speedq-clippy > "$1/provenance/speedq-clippy.log" 2>&1
    just lint-scripts > "$1/provenance/lint-scripts.log" 2>&1

[positional-arguments]
rev5perf3-verify root before:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$1/provenance"
    just --justfile benchmarks/rev5perf3.just rev5perf3-regress > "$1/provenance/padded-rows.log" 2>&1
    just --justfile benchmarks/rev5perf.just rev5perf-gate "$1" "$2" > "$1/provenance/gate.log" 2>&1
    just --justfile benchmarks/rev5perf.just rev5perf-validation "$1"
