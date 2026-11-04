# Phase 2 profile and HDR validation — 2026-11-04

Source commits: `fdce4ac2ec221f6b0d3d4966b74a8df4dcaf8abc` (withdraw E/ECodec) and `0b596aa0b331135e47ed314cdfe819822f077afa` (cube-root HDR).
Host: `dev`, Ryzen 9 9950X3D. The current public line at this checkpoint is
`A`, `LegacyA`, `PreviewV0_1`, `PreviewV0_2`; both selectors return `A`.

- Workspace all-target tests: 455 passed, 0 failed, 6 existing ignores; both custom
  benchmark targets completed under the same verified cpuset/memory wrapper.
- Workspace doctests: 14 passed, 0 failed, 9 existing ignores.
- All-feature contracts: 2 cross-path, 4 HDR and 8 profile tests passed.
- Scoped fmt checks and all-target/all-feature clippy with `-D warnings`: exit 0.
- 166/166 A HDR golden scores match the independent research BakeScorer bit for bit.
- 56/56 retained-profile SDR values match the frozen `d97076e1` baseline bit for bit.
- 28 SDR/HDR continuity comparisons at 203 cd/m²: maximum gap `0.00055716912091697`,
  below the unchanged `1e-3` assertion.

The all-target command reaches two custom zenbench executables. The same gate uses
CPUs `8-15,24-31`, with verified `enforced_uniform` cpuset placement, a 16 GiB hard
memory cap, no swap and 12 build/Rayon/OMP jobs. This is a correctness gate, not a
performance comparison; no benchmark source, protocol or threshold changed.

Commands run under the exclusive heavy-work lock and resource wrapper:

```sh
cargo fmt -p zensim -p zensim-regress --check
bash scripts/profile_test_cpuset.sh 8-15,24-31 cargo test --workspace --all-targets
cargo test --workspace --doc
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p zensim --all-features --test profiles --test hdr --test cross_path -- --nocapture
```

Complete output, setup/lint diagnostics, source identities, scratch oracle packages,
reference binary and checksums are preserved on dev at:

`/mnt/v/output/zensim/profile-a-phase2-2026-11-04/`

Resolve the observation with `ls -l` on `manifest.json` and verify the all-target log:

```sh
sha256sum /mnt/v/output/zensim/profile-a-phase2-2026-11-04/test-all-targets.log
```

Expected SHA-256: `a3334e554d352bc86ba75ad38e7e9a29f544dda2c41c73d7110e019a0fa4af5d`.

```text
run-heavy: done rc=0 120s | peak-RSS 1.43GiB | min-avail 36356MiB | peak-load 13.38
```

The original SIMD permutation test mutates process-wide token state; all scoring
tests now hold its testing lock. The legacy golden failure seen without this guard
was a test scheduling race. Goldens and tolerances were not changed.
