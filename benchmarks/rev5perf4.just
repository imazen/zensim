# Rev5 scaling investigation; invoke under shared heavy.lock and run-heavy.
set working-directory := '..'
export TMPDIR := env('HOME') / 'tmp'

[positional-arguments]
rev5perf4-baseline root:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 -c 'from pathlib import Path; import subprocess; assert Path("zensim/src/feature_v2.rs").read_bytes() == subprocess.check_output(["jj","file","show","-r","2c5124b4","zensim/src/feature_v2.rs"]), "baseline requires the original runtime source"'
    cargo fmt -p zensim -- --check > "$1/provenance/fmt-before.log" 2>&1
    just clippy > "$1/provenance/clippy-before.log" 2>&1
    just costcmp-build > "$1/provenance/build-before.jsonl" 2>&1
    just speedq-freeze "$1/provenance/build-before.jsonl" "$1/provenance/instrument-before"

rev5perf4-profile binary dest mode calls="32":
    python3 scripts/demos/rev5perf_profile.py --binary "{{binary}}" --dest "{{dest}}" --mode "{{mode}}" --calls {{calls}} --only v4x-t8-1024x1024,v4x-t16-1024x1024,v4x-t32-1024x1024,v4x-t8-4096x4096,v4x-t16-4096x4096,v4x-t32-4096x4096

rev5perf4-batch-test:
    cargo test -p zensim --all-features ordered_batches_match_serial_with_reused_strided_scratch -- --nocapture

[positional-arguments]
rev5perf4-build root label:
    #!/usr/bin/env bash
    set -euo pipefail
    just costcmp-build > "$1/provenance/build-$2.jsonl" 2>&1
    just speedq-freeze "$1/provenance/build-$2.jsonl" "$1/provenance/instrument-$2"

rev5perf4-parity binary dest:
    python3 scripts/demos/speedq_run.py parity --binary "{{binary}}" --dest "{{dest}}" --collect-legacy-failures

rev5perf4-costcmp mode root:
    python3 scripts/demos/costcmp_run.py {{mode}} --scaling --binary "{{root}}/provenance/instrument-verified" --before-binary "{{root}}/provenance/instrument-before" --dest "{{root}}/scaling/{{mode}}" --parity "{{root}}/scaling/parity/PREFLIGHT_PASS.json" --analyzer /mnt/v/output/zensim/costcmp-2026-10-09/provenance/paired-rounds-analyzer --lock /home/lilith/tmp/zensim-paper/rev4/heavy.lock

rev5perf4-python-tests:
    TMPDIR=/home/lilith/tmp python3 -m unittest discover -s scripts/demos -p 'test_speedq*.py'
    TMPDIR=/home/lilith/tmp python3 -m unittest discover -s scripts/demos -p 'test_costcmp*.py'

rev5perf4-all-tests:
    cargo test -p zensim --all-features

rev5perf4-report root json markdown:
    PYTHONPATH=scripts/demos python3 -c 'from pathlib import Path; from types import SimpleNamespace; import costcmp_report; costcmp_report.report(SimpleNamespace(raw_dir=Path("{{root}}"),out_json=Path("{{json}}"),out_md=Path("{{markdown}}"),scaling=True))'

rev5perf4-after root:
    just -f benchmarks/rev5perf4.just rev5perf4-build "{{root}}" after
    just -f benchmarks/rev5perf4.just rev5perf4-profile "{{root}}/provenance/instrument-after" "{{root}}/after-phases" phases 3
    just -f benchmarks/rev5perf4.just rev5perf4-parity "{{root}}/provenance/instrument-after" "{{root}}/after/parity"

rev5perf4-gates root:
    python3 scripts/demos/rev5perf_report.py --before "{{root}}/before" --after "{{root}}/after" --out "{{root}}/provenance/frozen-parity.json" --parity-only
    just -f benchmarks/rev5perf4.just rev5perf4-checks

rev5perf4-checks:
    just clippy
    just costcmp-clippy
    just -f benchmarks/rev5perf4.just rev5perf4-all-tests

rev5perf4-affinity binary dest cpuset cells:
    python3 scripts/demos/rev5perf_profile.py --binary "{{binary}}" --dest "{{dest}}" --mode perf --calls 16 --cpuset "{{cpuset}}" --only "{{cells}}"

rev5perf4-heap binary dest:
    python3 scripts/demos/rev5perf_profile.py --binary "{{binary}}" --dest "{{dest}}" --mode heaptrack --only v4x-t8-4096x4096,v4x-t32-4096x4096

rev5perf4-profiles root:
    just -f benchmarks/rev5perf4.just rev5perf4-profile "{{root}}/provenance/instrument-before" "{{root}}/before-phases" phases 3
    just -f benchmarks/rev5perf4.just rev5perf4-profile "{{root}}/provenance/instrument-before" "{{root}}/before-perf" perf 16
    just -f benchmarks/rev5perf4.just rev5perf4-profile "{{root}}/provenance/instrument-after" "{{root}}/after-perf" perf 16
    just -f benchmarks/rev5perf4.just rev5perf4-affinity "{{root}}/provenance/instrument-before" "{{root}}/before-ccd1" 8-15 v4x-t8-1024x1024,v4x-t8-4096x4096
    just -f benchmarks/rev5perf4.just rev5perf4-affinity "{{root}}/provenance/instrument-before" "{{root}}/before-ccd0-smt" 0-7,16-23 v4x-t16-1024x1024,v4x-t16-4096x4096
    just -f benchmarks/rev5perf4.just rev5perf4-affinity "{{root}}/provenance/instrument-after" "{{root}}/after-ccd0-smt" 0-7,16-23 v4x-t16-1024x1024,v4x-t16-4096x4096
    just -f benchmarks/rev5perf4.just rev5perf4-heap "{{root}}/provenance/instrument-before" "{{root}}/before-heap"
    just -f benchmarks/rev5perf4.just rev5perf4-heap "{{root}}/provenance/instrument-after" "{{root}}/after-heap"

rev5perf4-trial root label:
    just -f benchmarks/rev5perf4.just rev5perf4-build "{{root}}" "{{label}}"
    just -f benchmarks/rev5perf4.just rev5perf4-profile "{{root}}/provenance/instrument-{{label}}" "{{root}}/{{label}}-phases" phases 7

rev5perf4-final-gates root:
    just -f benchmarks/rev5perf4.just rev5perf4-parity "{{root}}/provenance/instrument-final" "{{root}}/final/parity"
    python3 scripts/demos/rev5perf_report.py --before "{{root}}/before" --after "{{root}}/final" --out "{{root}}/provenance/final-frozen-parity.json" --parity-only
    just -f benchmarks/rev5perf4.just rev5perf4-checks

rev5perf4-final-profiles root:
    just -f benchmarks/rev5perf4.just rev5perf4-profile "{{root}}/provenance/instrument-final" "{{root}}/final-perf" perf 16
    just -f benchmarks/rev5perf4.just rev5perf4-heap "{{root}}/provenance/instrument-final" "{{root}}/final-heap"

rev5perf4-verify root:
    just -f benchmarks/rev5perf4.just rev5perf4-build "{{root}}" verified
    just -f benchmarks/rev5perf4.just rev5perf4-parity "{{root}}/provenance/instrument-verified" "{{root}}/verified/parity"
    just -f benchmarks/rev5perf4.just rev5perf4-checks

rev5perf4-verified-profiles root:
    just -f benchmarks/rev5perf4.just rev5perf4-profile "{{root}}/provenance/instrument-verified" "{{root}}/verified-phases" phases 3
    just -f benchmarks/rev5perf4.just rev5perf4-profile "{{root}}/provenance/instrument-verified" "{{root}}/verified-perf" perf 16
    just -f benchmarks/rev5perf4.just rev5perf4-heap "{{root}}/provenance/instrument-verified" "{{root}}/verified-heap"

rev5perf4-verified-parity root:
    python3 scripts/demos/rev5perf_report.py --before "{{root}}/before" --after "{{root}}/verified" --out "{{root}}/provenance/verified-frozen-parity.json" --parity-only

rev5perf4-feature-checks:
    cargo fmt -p zensim -- --check
    cargo fmt --manifest-path zensim-bench/Cargo.toml -- --check
    cargo check -p zensim --no-default-features
    cargo check -p zensim --no-default-features --features feature-regime-v2

rev5perf4-qualify root:
    just -f benchmarks/rev5perf4.just rev5perf4-feature-checks
    just -f benchmarks/rev5perf4.just rev5perf4-verified-profiles "{{root}}"

rev5perf4-report-controls root dest:
    python3 scripts/demos/check_costcmp_scaling_report.py --raw-dir "{{root}}" --dest "{{dest}}"

rev5perf4-progress root:
    python3 scripts/demos/costcmp_run.py status --scaling --before-binary "{{root}}/provenance/instrument-before" --dest "{{root}}/scaling"

rev5perf4-archive root mirror:
    python3 scripts/demos/rev5perf_archive.py "{{root}}" "{{mirror}}"

rev5perf4-report-gates root:
    just -f benchmarks/rev5perf4.just rev5perf4-report "{{root}}" "{{root}}/report.json" benchmarks/rev5perf4_scaling_2026-10-09.md
    just -f benchmarks/rev5perf4.just rev5perf4-report-controls "{{root}}" "{{root}}/report-controls"
    just -f benchmarks/rev5perf4.just rev5perf4-python-tests
