# zensim

Workspace with two published crates: `zensim` (the metric) and `zensim-regress` (visual
regression testing on top of it). They are released independently with separate semver.

This repository continues the published crates from 0.2.7. Model training, evaluation, feature
research and their records live in [imazen/zensimx](https://github.com/imazen/zensimx), from
which this crate was cut on 2026-10-28. Train and evaluate there; this repository only serves
frozen models.

## Scores are a contract

- A change that moves any profile's scores or features is deliberate and recorded in
  `CHANGELOG.md`, never a side effect of a refactor.
- `tests/profiles.rs` holds golden scores; the revision-7 profiles (`E`, `ECodec`) must stay
  bit-identical across x86 SIMD tiers and thread counts.
- For refactors of the extraction or scoring code, rerun the split parity gate: build
  `examples/split_parity_check.rs` here and zensimx's `split_parity_golden` on the same machine
  and compare the outputs with `diff` (`docs/PROFILES.md`, "Parity with the research crate").
- Shipping a new model: add its bake to `zensim/weights/`, list it in `zensim/Cargo.toml`'s
  `include`, add a profile, document it in `docs/PROFILES.md`, and add golden scores.

New public API goes through a pull request so the signatures are reviewed.

## Release process

Tag format: `zensim-v0.3.0` for the library, `zensim-regress-v0.4.0` for the regression crate.

Before any release:

1. `cargo semver-checks --manifest-path zensim/Cargo.toml` (and for `zensim-regress`) against
   the previous published version; fix violations or bump accordingly.
2. `cargo test --workspace --all-targets` and `cargo test --workspace --doc`.
3. `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
4. README.md is accurate, confirmed by the owner before publishing.
5. CI green on every platform in `.github/workflows/ci.yml` (started by hand).

Release steps, per crate: bump the version, `cargo update -w`, commit `release: <crate>
v<version>`, push, wait for CI, `git tag <crate>-v<version>`, push the tag, `gh release create`,
then `cargo publish --manifest-path <crate>/Cargo.toml`. Never publish without a matching
pushed tag and GitHub release.

## Known Bugs

- Cross-path audit at `d15a5486` (2026-10-28): profile A's interleaved and planar
  linear diffmap scores differ on the 65×63 checkerboard fixture with padded rows.
  Score bits: interleaved `4635359462469191635`, planar `4635359462376882550`.
  The audit also found PreviewV0_2 score mismatches and legacy map rounding
  differences at 65×63 and 257×193. Reproducer fixture: `zensim/tests/cross_path.rs`.
  This predates the reuse refactor; preserving
  every existing score prevents silently canonicalizing either path.
