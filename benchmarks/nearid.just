# Invoke from repo root with just --justfile benchmarks/nearid.just --working-directory .
# All cargo/scoring commands must be enclosed by the coordinator's heavy.lock + run-heavy.
test:
    cargo test -p zensim --all-features --example serve_custom_bake nearid::tests
build:
    cargo build -p zensim --release --all-features --example serve_custom_bake
score program packet out model bake revision:
    ZENSIM_FORMULA_REV={{revision}} {{program}} --nearid {{packet}} {{out}} {{model}} {{bake}}
summary root:
    python3 scripts/prodqual_label_free.py --nearid-summary {{root}}
contact program packet image:
    {{program}} --nearid-contact {{packet}} {{image}}
register root:
    python3 scripts/prodqual_label_free.py --nearid-register {{root}}
report-test:
    python3 -m unittest discover -s scripts -p test_prodqual_label_free.py
verify:
    just clippy
    just lint-scripts
replay-verify root:
    python3 scripts/prodqual_label_free.py --nearid-verify {{root}}
# Entire pipeline inherits the caller's flock + run-heavy scope; children run sequentially.
pipeline root:
    #!/usr/bin/env bash
    set -euo pipefail
    root='{{root}}'
    mkdir -p "$root/bin" "$root/models"
    cp "$CARGO_TARGET_DIR/release/examples/serve_custom_bake" "$root/bin/serve_custom_bake"
    cp /mnt/v/output/zensim/adjudicate-2026-10-07/models/seed0.bin "$root/models/seed0.bin"
    cp zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin "$root/models/B.bin"
    cp zensim/weights/v47_strict_qat_native_byid_2026-09-06.bin "$root/models/A.bin"
    jj describe -m 'feat: add label-free near-identity serving instrumentation and replay checks'
    jj log -r @ --no-graph -T 'commit_id ++ "\n"' > "$root/BUILD_COMMIT.txt"
    jj bookmark set quarantine/codex/nearid -r @
    jj new -m 'docs: record measured near-identity panel'
    for model in seed0 B A; do
        printf '%s palette-sol NEARID scoring %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$model" > .workongoing
        revision=1
        if [[ "$model" == seed0 ]]; then revision=5; fi
        strace -f -e trace=openat -o "$root/$model.openat.log" env ZENSIM_FORMULA_REV="$revision" \
            "$root/bin/serve_custom_bake" --nearid "$root/PREREAD_FINAL.json" "$root" "$model" "$root/models/$model.bin" \
            2>&1 | tee "$root/$model.run.log"
    done
    MPLCONFIGDIR="$TMPDIR/matplotlib" python3 scripts/prodqual_label_free.py --nearid-summary "$root" 2>&1 | tee "$root/summary.log"
    python3 scripts/prodqual_label_free.py --nearid-verify "$root" 2>&1 | tee "$root/replay-verify.log"

# Verification before visual admission and measurement. Caller owns heavy.lock.
checks root:
    #!/usr/bin/env bash
    set -euo pipefail
    root='{{root}}'
    cargo fmt -p zensim
    cargo build -p zensim --release --all-features --example serve_custom_bake 2>&1 | tee "$root/build-source-policy.log"
    cargo test -p zensim --all-features --example serve_custom_bake nearid::tests 2>&1 | tee "$root/tool-tests-source-policy.log"
    MPLCONFIGDIR="$TMPDIR/matplotlib" python3 -m unittest discover -s scripts -p test_prodqual_label_free.py 2>&1 | tee "$root/report-tests-source-policy.log"
    just clippy 2>&1 | tee "$root/clippy-source-policy.log"
    just lint-scripts 2>&1 | tee "$root/lint-source-policy.log"
    "$CARGO_TARGET_DIR/release/examples/serve_custom_bake" --nearid-contact "$root/PREREAD_FINAL.json" "$root/train-panel.png" 2>&1 | tee "$root/train-panel.log"

# NAS mirror + all-file verification + owned inactive Cargo target cleanup.
archive:
    bash benchmarks/nearid_archive.sh

# Rebase validation uses frozen scored rows only, preserving the original evidence.
reproduce frozen out:
    python3 scripts/reproduce_nearid_report.py {{frozen}} {{out}}

land-validate root:
    #!/usr/bin/env bash
    set -euo pipefail
    root='{{root}}'
    cargo fmt -p zensim -- --check 2>&1 | tee "$root/fmt.log"
    just --justfile benchmarks/nearid.just --working-directory . reproduce /mnt/v/output/zensim/nearid-2026-10-09 "$root/reproduced" 2>&1 | tee "$root/reproduce.log"
    python3 -m unittest discover -s scripts -p test_prodqual_label_free.py 2>&1 | tee "$root/report-tests.log"
    just lint-scripts 2>&1 | tee "$root/lint-scripts.log"
    just clippy 2>&1 | tee "$root/clippy.log"
