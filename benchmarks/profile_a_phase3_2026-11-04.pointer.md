# Final public-profile validation (2026-11-04)

Validated source: `788918cde82db14ec210c98d0359b768e445d162`; baseline: `64142bdc92f665d862a51515d5b3391bbdfc74c7`.

| Configuration | Passed | Failed | Existing ignores | Full log SHA-256 |
|---|---:|---:|---:|---|
| test-default | 428 | 0 | 6 | `8d474fa1a6e319d79bd897d0b87d67add49bb526075e2aeb90c43ca6200f0a5c` |
| test-legacy | 443 | 0 | 6 | `b18618213631d498b2738426633c748fb0aa10ce37f6995c49a2956ab28f24c3` |

Both all-target runs completed both custom benchmark executables. Commands:
`bash scripts/profile_test_cpuset.sh 8-15,24-31 cargo test --workspace --all-targets`
and the same command with `--features legacy-models`. Each ran through the
resource wrapper with a 16 GiB cap and 12 build jobs. These runs validate the
gate; they are not a paired performance comparison.

All-feature library/integration tests: 470 passed, 0 failed, 6 existing ignores.
Doctests in each configuration: 14 passed, 0 failed, 8 existing ignores.
Clippy with `-D warnings` passed in default, legacy-models and all-feature builds.
Both documentation builds and no-default-feature builds passed. Actionlint passed.
Rustdoc reports an existing generated redundant-link warning in zensim-regress;
the required documentation build completes.

## Score preservation

[42 retained SDR rows](profile_a_phase3_parity_2026-11-04.tsv) compare the frozen
Phase 2 baseline with the final default (14 A rows) and legacy-models (42 rows)
builds on the same machine. Every populated comparison is bit-identical.
The existing A and preview golden constants and tolerances are unchanged.
All 166 research-derived HDR cases and 28 continuity comparisons pass in each
configuration; the maximum continuity gap remains 0.00055716912091697.

[Oracle source](profile_a_sdr_preservation_2026-11-04.rs) prints the SDR rows.
Run it as a scratch Cargo binary with edition 2024, dependencies on the checked-out
zensim and bytemuck 1.25.0, and a `legacy-models` feature forwarding to
`zensim/legacy-models`. Set `ZENSIM_SDR_PAIRS` at build time to the absolute path
of `zensim/examples/split_parity/pairs.rs`. Run once without the feature and once
with it, using the same resource wrapper. The archived manifests and command
files preserve the exact run setup.

## Artifacts and removal measurement

Full logs, commands, oracle sources/manifests, failed attempts and per-file hashes
are archived on dev at `/mnt/v/output/zensim/profile-a-phase3-2026-11-04/` and
mirrored to `/mnt/tower/output/zensim/profile-a-phase3-2026-11-04/`.
The artifact manifest records sizes and SHA-256 hashes; this file is the pointer
for the logs that exceed 30 KB. CI remains manually dispatched and was not started.

The two removal commits delete 7,877 text lines and add 276 (net 7,601 removed).
Three files are removed: the two exclusive revision-1 feature-vector tests and
the 26,456-byte v47 bake. The generic by-ID gather and output-spline evaluator
remain because A uses them. Retired IDs f228–f371 are unassigned; no retained
feature is renumbered. A's shared 156 steering-map coefficients remain, while
the full 228-weight preview vectors and linear scorer are gated.
