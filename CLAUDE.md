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
- `tests/profiles.rs` holds golden scores; the revision-7 profile `A` must stay
  bit-identical across x86 SIMD tiers and thread counts.
- Public `compute_hdr` uses the anchored cube-root front end at 203 cd/m².
  `tests/hdr.rs` checks research-derived A goldens bit for bit and SDR/HDR
  continuity within 1e-3; changing the front end changes HDR scores deliberately.
- For refactors of the extraction or scoring code, rerun the split parity gate: build
  `examples/split_parity_check.rs` here and zensimx's `split_parity_golden` on the same machine
  and compare the outputs with `diff` (`docs/PROFILES.md`, "Parity with the research crate").
- Shipping a new model: add its bake to `zensim/weights/`, list it in `zensim/Cargo.toml`'s
  `include`, add a profile, document it in `docs/PROFILES.md`, and add golden scores.

New public API goes through a pull request so the signatures are reviewed.

`split_parity_check` requires the `training` feature (declared in Cargo.toml).
`legacy-models` is off by default. `training` enables it for custom linear scoring;
preview tests and weights follow the feature. Run all-target tests and clippy in
both default and `--features legacy-models` configurations.
Fresh checkouts also need the existing classification PNG fixtures: run the two
commands in `just profile-fixtures` under the resource wrapper before testing.
The PNGs are ignored build inputs; keep them out of commits.
On asymmetric Linux hosts, `just profile-test-pinned CPU_IDS` adds a verified
cpuset in a job-specific system scope, copying the finite run-heavy memory
cap and running Cargo as the original user. Ordinary taskset affinity does not
satisfy zenbench process admission.

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

- Resolved test-harness race (2026-11-04): SIMD token permutations are process-wide.
  Unguarded legacy golden checks could observe a tier switch during computation.
  Every scoring test in `tests/profiles.rs` now holds `lock_token_testing()`,
  matching `tests/hdr.rs`; golden values and tolerances were not changed.

- The historical cross-path audit at `d15a5486` also found PreviewV0_2
  score mismatches and map rounding differences at 65×63 and 257×193.
  Those preview contracts remain behind `legacy-models`; see the historical
  `docs/CROSS_PATH_AUDIT.md`.

- The 2026-11-05 cross-target audit reported preview-profile differences of up to
  0.14 score points between native and WASM. The previews remain behind
  `legacy-models`; this does not relax their test expectations. The same audit
  reported bit-identical A scores on all 231 compute rows.

- Resolved i686 fixture overflow (2026-11-06): two feature-plan tests multiplied
  an index by 2654435761 in usize. The offsets now use u64, retaining all
  20,480 existing 64-bit fixture offsets and unchanged assertions (`b737a353`).
