# Run compute-heavy recipes through the environment's resource-limited runner.

fmt:
    cargo fmt -p zensim -p zensim-regress --check

check:
    cargo check --offline -p zensim --all-targets --all-features --message-format=short

test:
    cargo test --offline --workspace --features zensim/training,zensim/classification,zensim/zenpixels --no-fail-fast

clippy:
    cargo clippy --offline --workspace --all-targets --all-features -- -D warnings

path-test:
    cargo test --offline -p zensim --test cross_path --features training,classification,zenpixels -- --test-threads=1

path-audit *args:
    cargo run --offline --release -p zensim --example path_audit --features training,classification -- {{args}}

# Run inside the same resource/CPU placement wrapper as the executables.
path-audit-paired *args:
    python3 scripts/path_audit.py {{args}}

path-audit-report *args:
    PYTHONDONTWRITEBYTECODE=1 python3 scripts/path_audit_report.py {{args}}

path-audit-confirm-plan *args:
    PYTHONDONTWRITEBYTECODE=1 python3 scripts/path_audit_plan.py {{args}}

# Profile-serving validation; run each cargo command under the heavy-work lock
# and run-heavy, as with the other compute-heavy recipes above.
profile-test:
    cargo test --workspace --all-targets

profile-doc-test:
    cargo test --workspace --doc

profile-score-test:
    cargo test -p zensim --all-features --test profiles

# Populate the existing ignored classification PNG fixtures in a fresh checkout.
profile-fixtures:
    bash zensim/tests/fixtures/generate.sh
    cargo test -p zensim --test generate_bitops -- --ignored

# Requires an outer run-heavy systemd scope; CPU_IDS must name one CPU class.
profile-test-pinned cpu_ids:
    bash scripts/profile_test_cpuset.sh "{{cpu_ids}}" cargo test --workspace --all-targets
