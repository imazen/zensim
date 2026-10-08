# Dedicated Rev5 profiling and bit-identity gates; invoke with --justfile.
set working-directory := '..'

# Invoke inside the brief's flock/run-heavy envelope. No timing gates change.
rev5perf-profile binary dest mode="perf":
    python3 scripts/demos/rev5perf_profile.py --binary "{{binary}}" --dest "{{dest}}" --mode "{{mode}}"

rev5perf-report *options:
    python3 scripts/demos/rev5perf_report.py {{options}}

rev5perf-kernel-test:
    cargo test -p zensim --all-features feature_v2_stream::tests
    cargo test -p zensim --all-features rev5_simd_edges_match_frozen_windows
    cargo test -p zensim --all-features --test featcanon_rev5_parity

# Supply fresh roots; invoke under the task's shared heavy-command lock.
[positional-arguments]
rev5perf-gate dest before:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$1/provenance"
    just speedq-build > "$1/provenance/build.jsonl" 2>&1
    just speedq-freeze "$1/provenance/build.jsonl" "$1/provenance/instrument"
    just speedq-parity "$1/provenance/instrument" "$1/parity" --collect-legacy-failures
    just --justfile benchmarks/rev5perf.just rev5perf-report --before "$2" --after "$1" --out "$1/frozen-parity.json" --parity-only

[positional-arguments]
rev5perf-timing root analyzer:
    #!/usr/bin/env bash
    set -euo pipefail
    just speedq-timing --binary "$1/provenance/instrument" --dest "$1/timing" \
        --parity "$1/parity/PARITY_STRICT_PASS.json" --analyzer "$2" \
        --only scalar-t8-1920x1080,v4-t8-1920x1080,v4-t8-4096x4096,v4-t2-64x64 \
        --arms by_v2fy_r3 by_v2fy_r4 by_v2fy_r5 zensim_B fast_ssim2 butteraugli

[positional-arguments]
rev5perf-validation root:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo fmt -p zensim -- --check > "$1/provenance/fmt.log" 2>&1
    cargo test -p zensim --all-features > "$1/provenance/all-features-test.log" 2>&1
    just clippy > "$1/provenance/clippy.log" 2>&1
    just speedq-clippy > "$1/provenance/speedq-clippy.log" 2>&1
    just lint-scripts > "$1/provenance/lint-scripts.log" 2>&1

[positional-arguments]
rev5perf-collect root before analyzer:
    #!/usr/bin/env bash
    set -euo pipefail
    just --justfile benchmarks/rev5perf.just rev5perf-gate "$1" "$2"
    just --justfile benchmarks/rev5perf.just rev5perf-validation "$1"
    just --justfile benchmarks/rev5perf.just rev5perf-timing "$1" "$3"
    just --justfile benchmarks/rev5perf.just rev5perf-report --before "$2" --after "$1" --out "$1/comparison.json"
    just --justfile benchmarks/rev5perf.just rev5perf-profile "$1/provenance/instrument" "$1/perf" perf
    just --justfile benchmarks/rev5perf.just rev5perf-profile "$1/provenance/instrument" "$1/callgrind" callgrind
    just --justfile benchmarks/rev5perf.just rev5perf-profile "$1/provenance/instrument" "$1/phases" phases

