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

# Floor grid: six arms (uncapped, frozen 64/128/256 MiB, 128 MiB with floor 3,
# 128 MiB with floor 17 = original route) at 1024², 4096², 4608×4096,
# 6144×4096 and 8192×4096 (16/5/4/3/2 slots at 128 MiB). Internal --lock;
# the quiet gate and clean-round rule are unchanged.
rev5perf5-budget-timing root:
    python3 scripts/demos/costcmp_run.py timing --budget-grid --floor-arm --budget-binaries "{{root}}/provenance/budget-binaries-v3.json" --binary "{{root}}/provenance/instrument-floor3-grid" --dest "{{root}}/budget/timing" --parity "{{root}}/budget/parity-v3/PREFLIGHT_PASS.json" --analyzer "{{root}}/provenance/paired-rounds-analyzer" --lock /home/lilith/tmp/zensim-paper/rev4/heavy.lock

# Quiet-gated fresh-process RSS of the two floor arms over the floor grid.
rev5perf5-floor-rss root:
    python3 scripts/demos/costcmp_run.py rss --budget-grid --floor-arm --budget-binaries "{{root}}/provenance/budget-binaries-v3.json" --binary "{{root}}/provenance/instrument-floor3-grid" --dest "{{root}}/budget/rss" --parity "{{root}}/budget/parity-v3/PREFLIGHT_PASS.json" --lock /home/lilith/tmp/zensim-paper/rev4/heavy.lock

# Quiet-gated fresh-process RSS for the four measured arms (internal --lock).
rev5perf5-budget-rss root:
    python3 scripts/demos/costcmp_run.py rss --budget-grid --budget-binaries "{{root}}/provenance/budget-binaries.json" --binary "{{root}}/provenance/instrument-cap128" --dest "{{root}}/budget/rss" --parity "{{root}}/budget/parity/PREFLIGHT_PASS.json" --lock /home/lilith/tmp/zensim-paper/rev4/heavy.lock

rev5perf5-floor-preflight root:
    #!/usr/bin/env bash
    set -euo pipefail
    test -f zensim/src/feature_v2.rs
    trap 'python3 scripts/demos/rev5perf5_candidate.py floor-source --root "{{root}}" --slots 3' EXIT
    python3 scripts/demos/rev5perf5_candidate.py floor-source --root "{{root}}" --slots 17 --label floor17-grid
    just -f benchmarks/rev5perf5.just rev5perf5-build "{{root}}" floor17-grid
    python3 scripts/demos/rev5perf5_candidate.py floor-source --root "{{root}}" --slots 3 --label floor3-grid
    just -f benchmarks/rev5perf5.just rev5perf5-build "{{root}}" floor3-grid
    python3 scripts/demos/rev5perf5_candidate.py floor-inventory --root "{{root}}"
    python3 scripts/demos/costcmp_run.py parity --budget-grid --floor-arm --budget-binaries "{{root}}/provenance/budget-binaries-v3.json" --binary "{{root}}/provenance/instrument-floor3-grid" --dest "{{root}}/budget/parity-v3"
    for slots in 3 17; do
        just -f benchmarks/rev5perf5.just rev5perf5-parity "{{root}}/provenance/instrument-floor${slots}-grid" "{{root}}/floor${slots}-grid/parity"
        PYTHONPATH=scripts/demos python3 -c 'import sys; from pathlib import Path; from rev5perf_report import compare_bits; from speedq_run import write; r=Path(sys.argv[1]); write(r/f"provenance/parity-floor{sys.argv[2]}-grid.json", compare_bits(r/"uncapped", r/f"floor{sys.argv[2]}-grid"))' "{{root}}" "$slots"
    done

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
    PYTHONPATH=scripts/demos python3 -c 'from pathlib import Path; from types import SimpleNamespace; import costcmp_report; costcmp_report.report(SimpleNamespace(raw_dir=Path("{{root}}"),out_json=Path("{{root}}/report.json"),out_md=Path("benchmarks/rev5perf5_scaling_2026-10-09.md"),budget_grid=True,floor_arm=True))'
    python3 scripts/demos/check_costcmp_scaling_report.py --budget-grid --floor-arm --raw-dir "{{root}}" --dest "{{root}}/report-controls"
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
