# MAPPERF: same complete retained outputs, frozen models and measurement owner.
mapperf-bin-check:
    cargo test --locked -p zensim --lib --features custom-profiles attribution::tests::aligned_bin_fold_matches_independent_footprint_intersections -- --nocapture
    cargo test --locked -p zensim --lib --features custom-profiles attribution::tests::pixel_bin_unit_factor_matches_frozen_row_fold_bits -- --nocapture
    cargo test --locked -p zensim --lib --features custom-profiles attribution::tests::binned_ -- --nocapture

mapperf-runtime-build:
    CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target}" cargo bench --locked --manifest-path zensim-bench/addrfix-runtime/Cargo.toml --bench ssim2_speed_bar --features costcmp --no-run --message-format=json-render-diagnostics

mapperf-runtime-clippy:
    cargo clippy --locked --manifest-path zensim-bench/addrfix-runtime/Cargo.toml --bench ssim2_speed_bar --features costcmp -- -D warnings

mapperf-runtime mode root model:
    python3 scripts/rev4_featpot/mapperf.py {{quote(mode)}} --root {{quote(root)}} --model {{quote(model)}}

mapperf-progress root="/mnt/v/output/zensim/mapperf-2026-10-17":
    python3 -c 'from pathlib import Path; p=Path("{{root}}")/"provenance"; [(print(f"{f.name}: "+" | ".join(f.read_text(errors="replace").splitlines()[-3:]))) for f in sorted(p.glob("*.log"))]'

mapperf-local-refine-check:
    cargo test --locked -p zensim --lib --features custom-profiles local_refine:: -- --nocapture

mapperf-parity-check:
    just mapperf-bin-check
    just e34-d12-parity
    just mapperf-local-refine-check

mapperf-instrument-build:
    # Prepare the warm runtime first so matched timing can run during cache verification.
    CARGO_TARGET_DIR="$PWD/target" just mapperf-runtime-build > /mnt/v/output/zensim/mapperf-2026-10-17/provenance/urgent-v2-build.jsonl
    RUSTFLAGS='--cfg steerfix_instrument' cargo test --locked -p zensim --lib --features custom-profiles --no-run --message-format=json-render-diagnostics

mapperf-report root:
    python3 scripts/rev4_featpot/mapperf.py report --root {{quote(root)}}

mapperf-native-cache-prepare root:
    python3 scripts/rev4_featpot/mapperf.py native-cache-prepare --root {{quote(root)}}

mapperf-native-cache-check binary root model:
    ZENSIM_FORMULA_REV=5 ZENSIM_NEIGHBOUR_EXACT=1 RAYON_NUM_THREADS=1 STEERFIX_PACKET={{quote(root)}}/native-cache/{{quote(model)}}/job.json {{quote(binary)}} metric::bake::steerfix_packet::engineering_packet --exact --nocapture

mapperf-native-cache-compare root:
    python3 scripts/rev4_featpot/mapperf.py native-cache-compare --root {{quote(root)}}

mapperf-native-cache-full-compare binary root:
    python3 scripts/rev4_featpot/mapperf.py native-cache-full-compare --binary {{quote(binary)}} --root {{quote(root)}}

# Each nine-case native-cache chunk releases the shared locks for other lanes.
mapperf-native-cache-run binary root model:
    python3 scripts/rev4_featpot/mapperf.py native-cache-run --binary {{quote(binary)}} --root {{quote(root)}} --model {{quote(model)}}

mapperf-audit-build:
    RUSTFLAGS='--cfg steerfix_instrument' cargo test --locked -p zensim --lib --features custom-profiles --no-run --message-format=json-render-diagnostics

mapperf-audit binary root model:
    ZENSIM_FORMULA_REV=5 RAYON_NUM_THREADS=1 MAPPERF_AUDIT_ROOT={{quote(root)}} MAPPERF_AUDIT_MODEL={{quote(model)}} MAPPERF_STAGE_TIMING=1 {{quote(binary)}} metric::bake::mapperf_audit::declared_plan_and_stages --exact --nocapture

mapperf-stages root binary sha output:
    python3 scripts/rev4_featpot/mapperf_stages.py --root {{quote(root)}} --binary {{quote(binary)}} --sha256 {{quote(sha)}} --output {{quote(output)}}

mapperf-subset-check:
    cargo test --locked -p zensim --lib --features custom-profiles feature_plan::tests:: -- --nocapture
    just mapperf-local-refine-check

mapperf-subset-audit-build:
    just mapperf-subset-check
    just mapperf-audit-build

mapperf-stage-runtime-build:
    RUSTFLAGS='--cfg steerfix_instrument' cargo bench --locked --manifest-path zensim-bench/addrfix-runtime/Cargo.toml --bench ssim2_speed_bar --features costcmp --no-run --message-format=json-render-diagnostics

mapperf-stages-runtime root binary sha output cpu="10":
    python3 scripts/rev4_featpot/mapperf_stages.py --runtime --cpu {{quote(cpu)}} --root {{quote(root)}} --binary {{quote(binary)}} --sha256 {{quote(sha)}} --output {{quote(output)}}

mapperf-subset-oracle binary root:
    ZENSIM_FORMULA_REV=5 RAYON_NUM_THREADS=1 MAPPERF_AUDIT_ROOT={{quote(root)}} MAPPERF_AUDIT_MODEL=diff-410 MAPPERF_SUBSET_ORACLE=1 {{quote(binary)}} metric::bake::mapperf_audit::declared_plan_and_stages --exact --nocapture

mapperf-before-runtime-build root:
    cargo bench --locked --manifest-path {{quote(root)}}/source/before/zensim-bench/addrfix-runtime/Cargo.toml --bench ssim2_speed_bar --features costcmp --no-run --message-format=json-render-diagnostics

mapperf-native-remote-prepare root remote:
    python3 scripts/rev4_featpot/mapperf_native_remote.py prepare --root {{quote(root)}} --remote-root {{quote(remote)}}

mapperf-native-remote-run root binary sha:
    python3 scripts/rev4_featpot/mapperf_native_remote.py run --root {{quote(root)}} --binary {{quote(binary)}} --binary-sha256 {{quote(sha)}}

mapperf-frozen-bin-check binary:
    RAYON_NUM_THREADS=1 nice -n 19 ionice -c 3 {{quote(binary)}} attribution::tests::aligned_bin_fold_matches_independent_footprint_intersections --exact --nocapture --test-threads 1
    RAYON_NUM_THREADS=1 nice -n 19 ionice -c 3 {{quote(binary)}} attribution::tests::pixel_bin_unit_factor_matches_frozen_row_fold_bits --exact --nocapture --test-threads 1
    RAYON_NUM_THREADS=1 nice -n 19 ionice -c 3 {{quote(binary)}} attribution::tests::binned_ --nocapture --test-threads 1

# Match the ordinary D12 lib scope; packet instruments are caller-selected separately.
mapperf-frozen-map-check binary:
    RAYON_NUM_THREADS=1 {{quote(binary)}} metric::bake:: --skip metric::bake::steerfix_packet --skip metric::bake::mapperf_audit --nocapture --test-threads 1
    RAYON_NUM_THREADS=1 {{quote(binary)}} local_refine:: --nocapture --test-threads 1

mapperf-source-clippy:
    cargo clippy --locked -p zensim --lib --features custom-profiles -- -D warnings

# Samples the complete warmed prepared-map action; timing acceptance is separate.
mapperf-profile root model:
    python3 scripts/rev4_featpot/mapperf.py profile --root {{quote(root)}} --model {{quote(model)}}

# Validate one finished cell without fitting or implying a complete grid.
mapperf-cell-report root model geometry:
    python3 scripts/rev4_featpot/mapperf.py report --root {{quote(root)}} --model {{quote(model)}} --geometry {{quote(geometry)}}

mapperf-profile-stat root model:
    python3 scripts/rev4_featpot/mapperf.py profile-stat --root {{quote(root)}} --model {{quote(model)}}

# Queue behind the current job; release after one accepted cell or five minutes.
mapperf-lock-window root log:
    flock "$HOME/tmp/r7900x-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 1 -- python3 scripts/rev4_featpot/mapperf_lock_window.py --root {{quote(root)}} --log {{quote(log)}}

mapperf-lock-window-check:
    python3 scripts/rev4_featpot/test_mapperf_lock_window.py

# Run from this lane's owned workspace; preserve reference binaries first.
mapperf-clean-owned-targets artifacts:
    python3 scripts/rev4_featpot/mapperf_cargo_refs.py --workspace "$PWD" --artifacts {{quote(artifacts)}}
    python3 scripts/rev4_featpot/mapperf_cargo_refs.py --workspace "$PWD" --artifacts {{quote(artifacts)}} --check-only
    cargo clean --locked --target-dir "$PWD/target"
    cargo clean --locked --target-dir "$PWD/target-before"
    cargo clean --locked --manifest-path zensim-bench/addrfix-runtime/Cargo.toml --target-dir "$PWD/zensim-bench/addrfix-runtime/target"
