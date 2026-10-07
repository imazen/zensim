# MAINFIX investigation, 2026-10-07

**Coordinator sign-off is required before changing these three test contracts.** No Rust implementation, serving behavior or test expectation was changed. The reported width failures concern a comprehensive research request after the palette registry expanded; the production walk still emits 1825. Revision 5 is correctly known under the registered Rev5 spec. Its process pin is also required by that spec.

A concrete [unapplied patch](mainfix_proposed_test_contracts_2026-10-07.patch) is ready for review. It applies cleanly against this workspace, but has not been applied or tested. The requested full `cargo test -p zensim --all-features` run on a fixed tree remains pending the decision. Nothing is green by assumption.

## History and reproduced causes

Fresh workspace base and unchanged Rust build: `6497a365c9540412fa1423a922cbb2a38f126468` (`main@origin` when this lane started). Diagnostic driver/recipes and initial proposal: `7dc9bb52fd3e31be44104957c9a4f406aafc7eab`. Review bookmark: `quarantine/codex/mainfix`. Nothing was pushed; the shared `main` bookmark was not moved.

The introducing commits were identified from `jj log` and exact `jj diff` hunks, rather than inferred from commit subjects. Saved hunks are `palette-introduction.diff`, `palette-private-dispatch.diff` and `rev5-introduction.diff`. `history-pins.json` records full commit IDs. All seven relevant implementation files and the three failing test files are byte-identical between current main and the tested PRODQUAL source `6aa91cbc`; see `prodqual-source-comparison.json`.

| Failure | Fresh baseline observation | Cause and introducing commit |
| --- | --- | --- |
| `featcanon_rev4_contract::served_paths_serve_rev4` | `research::extract(Request::everything())` has 1867 values, assertion expects 1825 | `2022da95ebd2a3a184b28029a1966db7cec3d531` appends the 42 research palette slots and makes training research width 1867 |
| `research_engine_parity::research_everything_agrees_with_the_production_walk` | At 1153×72, production emits 1825, but the test expects research `full_width()` = 1867 | Same palette commit; the test equates comprehensive registered research width with production width |
| `bake_surface::formula_revision_is_selected_per_bake_and_unknown_or_mixed_revisions_refuse` | Loader accepts `"5"`, the negative control says it must be unknown | `560de601006da590369cfea034d9481efd969bf6` adds the enum, metadata parser and process switch for the registered Rev5 revision |

The palette chain landed through merge `0a8a7ef851e211419f94ebf9896f08de42dd4dd9`; the precise source introduction is `2022da95`. The later private-dispatch refactor `44df60621948` changes the palette's internal token ownership, not the intended research/serving separation.

`research::Request::everything` is documented as every registered slot. With `training`, `research::full_width()` includes the palette tail; without `training`, it stops at the palette base, 1825. The failure at `featcanon_rev4_contract.rs:438` is specifically this research extraction, after the served-call loop succeeds. It is not a 1867-value production result. In the second failing test, the actual production vector is measured as 1825. `Plan::for_bake` explicitly rejects any palette read. No serving-width regression was found in these investigated paths.

The existing `palette_research` integration gate passes on current main. It checks sparse palette structural fills, 42-value dense palette output, private identity, tiers, strided rows, superseded-revision refusal and bit parity of the legacy vector when palette is appended. Its legacy reads are basic/peak/v2; this does not claim all legacy families were exercised by that particular test. The broader production-vs-research test still fails before its full bit comparison, so that gate remains open pending approval.

## Rev5 metadata and the process pin

[Rev5 spec, section 1](rev5_spec_2026-10-04.md) registers the known revision, its supported basic/peak/v2 scope, the same process model as Rev4, `ZENSIM_FORMULA_REV=5`, and refusal of cross-revision mixes. Addendum B records the landing at `60174678682753458b3aeb9c403ee87ca0d157b7` and freezes its arithmetic. Rejecting revision 5 as unknown would contradict this contract.

Metadata chooses the plan's Rev5 arithmetic. It does **not** remove the process agreement guard for pixel scoring: `metric/bake.rs::check_pixel_revision` calls `ssim_form::refuse_rev4_mix` before narrow-plan exemptions. A differing requested/process revision is refused when either is revision 4 or later. This prevents process-owned formula gates from being silently mixed with the bake's revision. Feature-only inference can load a known revision without proving pixel-serving compatibility.

The existing, unchanged `serve_custom_bake` example was built from current main and run against all three pinned production f16 bakes, using the same checked-in `v1_golden_real_ref.png` on both sides. This identity-only fixture avoids seed-comparison or quality evidence. The driver uses exactly the brief's model SHA-256 pins before any scoring; no corpus/label loader is involved.

| Seed | Process revision unset | `ZENSIM_FORMULA_REV=5` |
| --- | --- | --- |
| 0 | Pixel and identity calls refused with cross-revision error, exit 1 | Pixel and identity 100, emitted vector 720, exit 0 |
| 1 | Same refusal, exit 1 | Pixel and identity 100, emitted vector 720, exit 0 |
| 2 | Same refusal, exit 1 | Pixel and identity 100, emitted vector 720, exit 0 |

All six expected outcomes were observed and reproduced through the justfile recipe. The declared caller width is 420; the emitted canonical vector reaches slot 719 and has width 720. These are different quantities. Raw outputs and exact model/program/fixture hashes are in `revision-pin/` and `revision-pin-just/`. The preserved serving executable SHA-256 is `03a91d0d788de6638566af2f3886231aab9c4f2d7a9aeb4cce9f431cb8e154b6`.

The existing `rev5_bake_serves_supported_reads_and_refuses_unsupported` unit gate passes too: supported reads serve in a Rev5 process, scores/maps agree, unsupported families refuse, and a Rev4 bake refuses in a Rev5 process. Current behavior matches the spec. **No change to revision selection or the process pin is proposed.** This is a contract/mechanics probe, not a production quality qualification or composition choice.

## Approval-ready test proposal

The patch changes only the three affected tests:

1. Check comprehensive Rev4 research width against `research::full_width()`, separately retain an explicit 1825-slot legacy request, and assert every legacy feature bit matches the comprehensive extraction.
2. Assert the complete production walk has its explicit legacy width 1825. Keep the comprehensive research width assertion and every production feature's bit comparison across the existing geometry matrix. Do not append palette to production to satisfy the test.
3. Replace the obsolete unknown-revision `"5"` negative control with `"999"`, retain the other malformed/unknown and mixed-ensemble cases, and add explicit known-revision-5 loading.

The owner-decision `basic_only_bake_compatibility_respects_partial_producers` test is untouched and was not run in this lane. No expectation or threshold has been relaxed, and no ignore or skip was added. The proposal remains unapplied because the lane instruction requires a coordinator decision for these stale expectations. Its tests have **not** been run on the proposed tree.

## Verification and remaining work

- The fresh baseline invocation of the three affected targets fails with the same three assertions (`baseline-targets.log`, rc 101).
- Existing Rev5 serving contract and all four palette integration tests pass (`contract-gates.log`, rc 0).
- Current-source serving example build passes (`serving-probe-build.log`, rc 0).
- All three packed bakes exhibit the expected unpinned refusal and pinned identity behavior in both driver invocations (`revision-pin*.log`, rc 0).
- Python syntax check, scoped `cargo fmt -p zensim --check` and `git apply --check` pass. The proposal stays unapplied.

After coordinator approval, apply the reviewed test patch, rerun the three affected targets, then run the requested **full** `cargo test -p zensim --all-features` including integration tests. Also run the applicable clippy/API checks. The full fixed-tree suite was not run in this lane because no authorized fixed test tree exists. The known owner-decision validation failure remains separate.

## Evidence and reproduction

[Machine record](mainfix_investigation_2026-10-07.json). Evidence root: `/mnt/v/output/zensim/mainfix-2026-10-07/`. Mirror: `/mnt/tower/output/zensim/mainfix-2026-10-07/`. Program/model/fixture bytes, complete command logs, exact history diffs, source pins and the unapplied patch remain outside git except the small proposal and report. `_MANIFEST.json`, `SHA256SUMS`, `mirror-verification.json` and `cleanup-receipt.json` document preservation and storage checks.

Recipes: `just mainfix-baseline`, `just mainfix-contracts`, and `just mainfix-revision-probe <program> <fixture> <fresh-result-dir> <seed0> <seed1> <seed2>`. The build command is `cargo build -p zensim --all-features --example serve_custom_bake`. The baseline was run directly with the exact cargo command now recorded in `mainfix-baseline`. Each driver execution refuses to overwrite prior case logs/results.

Builds use `TMPDIR=/home/lilith/tmp/mainfix`, `CARGO_TARGET_DIR=/home/lilith/tmp/mainfix/target`, `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, and `/home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 --`. Toolchain: Rust 1.99.0 (b940084d7), LLVM 23.1.1. The baseline records `run-heavy: done rc=101 140s | peak-RSS 1.26GiB | min-avail 32046MiB | peak-load 14.09`; the existing contract gates record `run-heavy: done rc=0 28s | peak-RSS 1.97GiB | min-avail 43210MiB | peak-load 9.52`. These are command resource observations, not performance claims.

All 48 evidence payloads were hash-verified on the mirror before cleanup, including three randomly chosen files. All six preserved executable hashes were verified, then only this lane’s inactive Cargo target was removed (measured apparent size 3,388,354,762 bytes). The archive mirror recipe is `just mainfix-mirror <evidence> <destination>`. Small diagnostics, records and the unlanded workspace remain available for review.
