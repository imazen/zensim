# Profile rename validation — 2026-11-04

Observed on `dev` (Ryzen 9 9950X3D). Source: rename `38b16050`, feature-gating and
fixture setup `c8dd10a7`, canonical-name and parity contracts `eb777fea`.

- `cargo test --workspace --all-targets`: 455 passed, 0 failed, 6 existing ignores;
  both custom benchmark targets completed.
- `cargo test --workspace --doc`: 14 passed, 0 failed, 9 existing ignores.
- `cargo test -p zensim --all-features --test profiles`: 8 passed, 0 failed.
- Scoped `cargo fmt --check` and all-target/all-feature clippy with `-D warnings`: exit 0.
- Exact same-host score preservation: 70/70 cases
  ([per-case bits](profile_rename_parity_2026-11-04.tsv)).

The all-target gate reaches two custom zenbench executables. A one-run owner exception
allowed this gate on dev. The CPU placement was `enforced_uniform` on CPUs
`8-15,24-31`, with a 16 GiB hard memory cap, no swap and 12 build/Rayon/OMP jobs.
Thread affinity alone was insufficient; the helper copies the resource wrapper's
finite memory cap into a job-specific CPU scope and runs Cargo as the original user.
The benchmark source, protocol, thresholds and existing test expectations were unchanged.

Payload under the exclusive resource wrapper:

```sh
bash scripts/profile_test_cpuset.sh 8-15,24-31 cargo test --workspace --all-targets
```

Complete stdout/stderr, failed setup attempts, other gate logs, command scripts,
source identities and SHA-256 manifest are preserved on dev at:

`/mnt/v/output/zensim/profile-a-gates-2026-11-04/`

Resolve/check the recorded observation with:

```sh
ls -l /mnt/v/output/zensim/profile-a-gates-2026-11-04/manifest.json
sha256sum /mnt/v/output/zensim/profile-a-gates-2026-11-04/test-all-targets-pinned.log
```

Pinned log SHA-256: `b4ab33b1f74d3cd979778d3dd1e5423195d91376b67f35af519c6f1a79badf9d`.

```text
run-heavy: done rc=0 82s | peak-RSS 1.24GiB | min-avail 37667MiB | peak-load 4.91
```

The benchmark output is retained as gate evidence, not as a performance comparison.
