# Packed row-prefix regression and frozen-output qualification.
set working-directory := '..'

rev5perf3-regress:
    cargo test -p zensim --all-features --lib source_rows_tests -- --nocapture

[positional-arguments]
rev5perf3-verify root before:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$1/provenance"
    just --justfile benchmarks/rev5perf3.just rev5perf3-regress > "$1/provenance/padded-rows.log" 2>&1
    just --justfile benchmarks/rev5perf.just rev5perf-gate "$1" "$2" > "$1/provenance/gate.log" 2>&1
    just --justfile benchmarks/rev5perf.just rev5perf-validation "$1"
