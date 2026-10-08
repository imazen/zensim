# Imported by the root justfile; run these recipes through that file.
# SPEEDQ uses the existing synthetic matrix images and Rust scoring surfaces.
# Explicitly omit optional C++ oracle arms: only existing Rust peers are required.
speedq-build:
    ZENSIM_BENCH_SKIP_CPP_FFI=1 cargo bench --no-run --manifest-path zensim-bench/Cargo.toml --bench ssim2_speed_bar --features speedq,ssim2-rayon --message-format=json-render-diagnostics

speedq-clippy:
    ZENSIM_BENCH_SKIP_CPP_FFI=1 cargo clippy --manifest-path zensim-bench/Cargo.toml --bench ssim2_speed_bar --features speedq,ssim2-rayon -- -D warnings

[positional-arguments]
speedq-parity binary dest *options:
    #!/usr/bin/env bash
    binary=$1
    dest=$2
    shift 2
    exec python3 scripts/demos/speedq_run.py parity --binary "$binary" --dest "$dest" "$@"

speedq-test:
    python3 -m unittest discover -s scripts/demos -p 'test_speedq.py' -v

# Marker counts are progress only; the reporter validates qualification evidence.
[positional-arguments]
speedq-status raw_dir:
    python3 scripts/demos/speedq_run.py status --dest "$1"

# Run one traced segment; keep its destination separate from qualification data.
[positional-arguments]
speedq-diagnose *options:
    python3 scripts/demos/speedq_run.py diagnose "$@"

[positional-arguments]
speedq-mirror source dest:
    nice -n19 ionice -c3 rsync -a --no-owner --no-group "$1/" "$2/"

[positional-arguments]
speedq-timing *options:
    python3 scripts/demos/speedq_run.py timing "$@"

[positional-arguments]
speedq-rss *options:
    python3 scripts/demos/speedq_run.py rss "$@"

# Freeze the executable named by the successful Cargo JSON build receipt.
speedq-freeze build_log dest:
    python3 scripts/demos/speedq_run.py freeze --build-log "{{build_log}}" --dest "{{dest}}"
