# Part B clean-source builds (pass 1 and pass 2)

Both extractor binaries were built from clean `git archive` snapshots of fetched main/master (no sibling working copy), with `CARGO_HOME` and target under `/var/tmp/`. Neither is a stock main build: each carries a scratch 2-line `audit.rs` width-gate patch (pass 1 adds 1322; pass 2 adds 1322 and 1502). Main superseded that gate in `384d15e1` and REVIEW_PARTB (2026-09-25) shows unpatched main `abe694a0` reproduces every sidecar bit for bit, so the patches are not part of the landing. Cargo ran without `--offline`/`--locked` (git pins fetched); `cargo metadata` reports 0 git-source packages. `corruption-corpus` is a `[patch]` path to a `git archive` of codec-corpus `8e10d4d7`.

## Pass 1: `--full-rev4`, sidecars `features__rev4c1c4` + `features__csfw_dvifm`

- Binary `/var/tmp/partb/target/release/examples/extract_features_372col` sha256 `dd4db2eea31a27ab54315ea841793aa23bf76f6f4c871c22d8f5c8f9ee4434d3`.
- Build meta v2 `/var/tmp/partb/build_meta.json` sha256 `407e5f054981a008bd1974e53c04beed74ee137d67b01d532ce9367a96083768`; Cargo.lock `bf1130b4d9d50e7ee90772c2a8c9e0f264901c52b60a2b558fe8970d40e53080`; overlaid `zensim-bench/Cargo.toml` `55f7e43c46e140f618e1e88273373aefc4327cfa0477d53cef16d9d74216dace`; `rustc 1.98.1 (48a229cea 2026-09-01)`.
- **Correction 2026-09-25:** v1 build meta (`5ce5df9a8626665eced02a62febf50499c2e391b17f576558a1aa7665d7b35b0`, kept as `/var/tmp/partb/build_meta_STALE_5ce5df9a_codex_13z.json`) was written by the Codex lane at 2026-09-24T13:00Z before the build and recorded zenavif `a7c56be9`, zenjxl-decoder `814994a2`, zenmetrics `43abef94`, no codec-corpus. The binary was compiled from zenavif `b5be33bd`, zenjxl-decoder `623d0c36`, zenmetrics `ea16371c` and codec-corpus `8e10d4d7` (v2, regenerated from `/var/tmp/partb/src` and the build-time `snapshot_revs.tsv`). All 36 pass-1 manifest entries were repointed to v2; feature values are unaffected.

| repo | branch | commit |
|---|---|---|
| codec-corpus | main | `8e10d4d76566` |
| jxl-encoder | main | `b16116ba848d` |
| rav1d-safe | main | `e771c7b3e7bb` |
| zenanalyze | main | `b102fa5f3480` |
| zenavif | main | `b5be33bd244e` |
| zenbench | main | `53941021fd20` |
| zenjpeg | main | `8f703a6eeaad` |
| zenjxl | main | `4a2c021b4589` |
| zenjxl-decoder | main | `623d0c360735` |
| zenmetrics | master | `ea16371cfe32` |
| zenpixels | main | `e56f626b14e6` |
| zenpng | main | `cfccd88f77cc` |
| zenresize | main | `e3975fb9d6d6` |
| zensim | main | `6a7b88e507ef` |
| zenwebp | main | `8aa8a7858b97` |

## Pass 2: `--full-gmsbank`, sidecar `features__gmsbank`

- Binary `/var/tmp/partb2/target/release/examples/extract_features_372col` sha256 `730741b9577ba82059880345d125c7dbfd4c9bcd941d99e80464b9d2f4d90642`.
- Build meta `/var/tmp/partb2/build_meta.json` sha256 `3fc2c5ac164b9f9a207c903f75852a196a00afa1ecf1b3655f65f5ed4bf6ef0c`; Cargo.lock `2944a9ef860b6bb0bcd57a79e9dea44f711df598d917e9c233ec943584ee815a`; `rustc 1.98.1 (48a229cea 2026-09-01)`.
- Built from zensim `390c99a3c38c88d4f6d553bf44990a7ee983b14f` (a descendant of `26494c8a`, includes the C8 fix `8a0850be`).

| repo | branch | commit |
|---|---|---|
| jxl-encoder | main | `f738479d0709` |
| rav1d-safe | main | `e771c7b3e7bb` |
| zenanalyze | main | `6bbcacb11313` |
| zenavif | main | `b5be33bd244e` |
| zenbench | main | `53941021fd20` |
| zenjpeg | main | `8f703a6eeaad` |
| zenjxl | main | `4a2c021b4589` |
| zenjxl-decoder | main | `736506c86c11` |
| zenmetrics | master | `f1eef2aecf71` |
| zenpixels | main | `e56f626b14e6` |
| zenpng | main | `cfccd88f77cc` |
| zenresize | main | `e3975fb9d6d6` |
| zensim | main | `390c99a3c38c` |
| zenwebp | main | `8aa8a7858b97` |
