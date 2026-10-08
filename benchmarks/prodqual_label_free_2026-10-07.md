# Production qualification: label-free gates, 2026-10-07

Qualification is **not green**. Four existing test assertions still fail. Seed 1 and seed 2 also fall below the registered feature-inference identity band on all four synthetic identity probes. The final seed or ensemble is not frozen; this run neither selects a composition nor reads evaluation or human-label payloads.

Source under test: `6aa91cbcc95cb88288141c16b5a0d3298c24e35b`, based on `main@origin` at `c8c2475f52c8`. Review bookmark: `quarantine/codex/prodqual-a`. Nothing was pushed and the shared `main` bookmark was not moved. The gate map was read from `quarantine/codex/releasegate2:benchmarks/release_gate_map_2026-10-07.md`; it remains under review and is not included here.

## Inputs and metadata

All three explicit packed `fitv2d1-20261007` f16 seed files matched both the brief's pins and their canonical mirror before staging. Complete source paths and hashes are in the accompanying JSON and `model-pins.json`.

| Seed | Bytes | SHA-256 |
| --- | ---: | --- |
| 0 | 110201 | `f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400` |
| 1 | 109504 | `1bf8f3afbf0d4c760e3fcfb1e36fa3ca8c2c19f838b0f6719b8c681fa682fdec` |
| 2 | 109470 | `54118c54ce7fe90bf1e9dcec52f7f822caf54e93e5b1d1a95a2bbd37f741b688` |

The canonical `inspect_qualified_checkpoint` example was built from the tested source and passed on 3/3 files. Each declares FormulaRevision 5, qualified provenance, feature set `basic+peaks+v2@w1825/rev5_localwin#36c3f3af`, checkpoint epoch `119`, seven admitted tables, and no historical replay metadata. Its preserved executable SHA-256 is `1202eca2242187aa36c9a07ebdd67c93dafa4505ea61745504ef066e76d9945a`. Inspecting embedded provenance did not load those tables.

## Pixel, cache and dispatch audit

The existing `serve_custom_bake` example now has a private `--prodqual` mode. It loads explicit packed bytes through `Model::from_bytes`, `Request::for_bake_bytes` and public `BakeScorer` APIs. It compares direct pixels, canonical research extraction, feature inference, identity-aware feature inference and prepared steering caches. All runs use the same deterministic synthetic texture with geometries 64×64, 97×63, 131×65 and 255×129, dropping 0 through 6 low bits per channel in a nested distortion ladder. Fixture content was not adjusted to improve results.

Each seed reads 420/420 declared feature IDs. The audit records every consumed feature's f64 bits, including cached extraction, rather than assuming the first 420 positions of the wider feature namespace are the inputs. Across 28 pairs × 10 native token permutations × 3 seeds (840 native rows), there were zero consumed-feature mismatches, cache-feature mismatches, cache/pixel score mismatches or native tier row mismatches. The permutations exercise v4x, v4, v3 and scalar on the native host. All three seeds were also run on WASM128: 28 pairs each, zero cross-target row mismatches, zero WASM feature/cache mismatches, and finite outputs throughout.

Identical pixels score exactly 100 for 3/3 seeds on every geometry. No distorted pair exceeds identity. All 12 pixel-score ladders are nonincreasing; this is a **report-only** synthetic observation, not an evaluation-quality or perceptual ranking gate.

| Seed | Feature identity at 64×64 | 97×63 | 131×65 | 255×129 | Registered [97.5,100] band |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 98.36815975441085 | 98.2979819795809 | 98.32183293954 | 98.32761109030544 | Pass on 4/4 |
| 1 | 96.81424717298958 | 96.76696026546807 | 96.76900683558603 | 96.79637797063307 | Below band on 4/4 |
| 2 | 97.20547219784015 | 97.1752002250599 | 97.17313127727995 | 97.18860034504641 | Below band on 4/4 |

These are feature-only inference results; the identity-aware pixel path remains exactly 100. No threshold was changed. The full registered 38-image identity/negative-tail gate was not run. Serving these Rev5 bakes requires the process pin `ZENSIM_FORMULA_REV=5`; this change does not switch any serving default.

## Test and API state

| Gate | Result | Evidence |
| --- | --- | --- |
| Workspace all targets/all features, excluding wasm-tests, compile | Pass | `workspace-build-final.log` |
| Label-free workspace runtime subset | Four failing test targets | `workspace-tests.log` |
| Final rerun of the four affected targets | Same four assertions fail | `workspace-failures-final.log` |
| Release Rev5 parity | Pass, all three parity tests | `rev5-parity-final.log` |
| Feature invariants, per-bake revision, steering | Executed in runtime subset and passed | `workspace-tests.log` |
| Workspace doc tests | Pass | `doc-tests-final.log` |
| `just clippy` | Pass | `clippy-final.log` |
| `just api-doc-check` | Pass; public snapshots unchanged | `api-doc-check-final.log` |
| CI feature permutations | 27/27 configurations pass clippy and library tests | `feature-matrix-final.log` |
| Cross-build serving matrix | Eight comparison arms pass; zero named refusals | `serving-matrix.log`, `serving-matrix/` |
| Native and WASM production probes | Exact parity and cache audit pass | `summary-final.json`, raw `*-synthetic-final.json` |
| Scoped fmt check | Pass | `just prodqual-fmt-check` |

The exact unfiltered `cargo test --workspace --all-targets --all-features --exclude zensim-wasm-tests` runtime command was **not executed**. Its full compilation passed. The runnable subset uses lib/bins/tests/examples and five explicit caller-owned exclusions: `cid22_aggregate_srocc_matches_audit_reference`, `cid22_first_row_matches_bake_verdict_reference`, `parallel_matches_sequential_iwssim_log_target`, `parallel_matches_sequential_default_target_with_scale`, and `canonical_dial_grid_is_the_quarantined_v2_grid`. The first four read human-label payloads; the fifth hashes an evaluation dial payload. Harness-free benchmark mains were compiled only: their external image/model defaults and performance runs are outside this lane's admitted synthetic inputs. Existing optional historical-fixture tests that return when files are absent are not claimed as fixture coverage. No test expectation, threshold or ignore annotation was changed.

The unresolved assertions are:

- `zensim/tests/featcanon_rev4_contract.rs:438`, `served_paths_serve_rev4`: 1867 features versus expected 1825 (nested re-execution propagates the failure).
- `zensim/tests/research_engine_parity.rs:161`, `research_everything_agrees_with_the_production_walk`: at 1153×72, production width is 1825 and research width is 1867.
- `zensim-validate/tests/bake_surface.rs:364`, `formula_revision_is_selected_per_bake_and_unknown_or_mixed_revisions_refuse`: the test expects revision 5 to be unknown, while the loader accepts it.
- `zensim-validate/tests/feature_set_match.rs:131`, `basic_only_bake_compatibility_respects_partial_producers`: basic coverage for `basic+v2@w720/rev5_localwin#62adfc93` is 1 versus expected 0. This failure is already recorded in the project Known Bugs; its projection omits basic IDs 0–12 and 26–38.

The label-free workspace run's `openat` trace records 33 unique external attempted paths: metadata manifests, nonexistent sidecar probes and historical packed bake bytes. No external corpus parquet, CSV, image or evaluation payload appears in that trace. This is a file-open audit of the selected process tree, not a general proof about arbitrary programs.

## Private feature-guard correction

The first feature matrix found nine configurations failing both clippy and library tests: `training` without `feature-regime-v2` compiled the palette module, which referenced an absent research module. Commit `e83fefd01f1e3de47d0da98c6cc9b8714b237c3a` adds an inner `cfg(feature = "feature-regime-v2")` guard to the private palette module. Its callers are in research. The correction adds no public API and changes no model or metric arithmetic.

The full matrix then passes 27/27 configurations. `baseline-unchanged.json` compares initial and final native evidence: zero changes in pixel, cached, feature-only and identity-aware score bits or consumed feature bits for all 840 rows. The final probe additionally checks cached feature bits against canonical extraction.

## Reproduction and evidence

Tracked machine results: [prodqual_label_free_2026-10-07.json](prodqual_label_free_2026-10-07.json). Local evidence root: `/mnt/v/output/zensim/prodqual-a-2026-10-07/`. Mirror: `/mnt/tower/output/zensim/prodqual-a-2026-10-07/`. Raw feature bits, binaries and logs remain outside git. `program-pins.json`, `_MANIFEST.json`, `SHA256SUMS`, `mirror-verification.json` and `cleanup-receipt.json` record the pins and storage checks.

Builds and test commands use `TMPDIR=/home/lilith/tmp/prodqual-a`, `CARGO_TARGET_DIR=/home/lilith/tmp/prodqual-a/target`, `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, and `/home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 --`. Repeatable commands are in the justfile: `prodqual-workspace-build`, `prodqual-workspace-tests`, `prodqual-workspace-failures`, `prodqual-rev5`, `prodqual-feature-matrix`, `prodqual-serving-matrix`, `prodqual-synthetic`, `prodqual-wasm-build`, `prodqual-wasm-synthetic`, `prodqual-training-only`, `prodqual-fmt-check` and `prodqual-mirror`. The ordinary commands are `cargo test --workspace --doc`, `just clippy` and `just api-doc-check`. Canonical inspection uses `cargo build -p zensim -p zensim-validate --all-features --example inspect_qualified_checkpoint`, then that binary with each pinned model path. Full command echoes and run-heavy resource lines are preserved in logs; the JSON maps each gate to its command and log.

Native builds use Rust 1.99.0 (b940084d7), LLVM 23.1.1. WASM follows the repository CI pin: Rust 1.98.1, wasm32-wasip1, `-C target-feature=+simd128`, Wasmtime 40.0.1. WASM has only the staged evidence directory preopened as `/inputs`. The final native synthetic run records `run-heavy: done rc=0 90s | peak-RSS 0.91GiB | min-avail 45903MiB | peak-load 15.00`; the final WASM run records `run-heavy: done rc=0 8s | peak-RSS 0.21GiB | min-avail 46680MiB | peak-load 11.87`. These are observations for these commands, not performance qualifications.

Coverage is limited to serial legacy-RGB8 SDR, four small geometries and one procedural texture class. HDR, large images, performance, human-quality evaluation, final composition and the full canonical identity gate remain unqualified by this lane.

All 81 evidence payloads were SHA-256 verified on the mirror before cleanup, including three randomly chosen files. The seven pinned executables were also verified before deleting only this lane’s inactive Cargo target directory. Its measured apparent size was 7,939,303,046 bytes; the cleanup receipt records filesystem free bytes before and after. Evidence remains locally and on the mirror. The initial archive transfer reported ownership-attribute errors; content hashes passed. The repeatable mirror recipe disables ownership/group preservation for this archive mount.
