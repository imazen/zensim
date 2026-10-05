# SHIPPATH worklog — 2026-10-05 UTC

MISSING: no production-qualified by_v2fy bake. This lane closes only the Rev5 TRAIN teacher table admission plumbing. It does not run human/T0 evaluation, full fitting, fleet jobs or publishing.

Base c95d3ac30ec068599ec68a21cf8d119919425a17; own jj workspace shippath; bookmark quarantine/codex/shippath. Governing brief: ~/tmp/zensim-paper/rev4/SHIPPATH_brief.md. Read local CLAUDE, production priorities, playbook, splits, provenance index and Rev5 spec including 03:25 UTC correction.

Before smoke: use only safesyn and CID22 oracle TRAIN fit/internal-development tables, four files from frozen v2c5/main/real. No protected labels. Preserve original Parquet bytes and reference order; write a fresh admission view at /var/tmp/shippath/teachers. Declare the actual producing executable as the decoder era, together with legacy-rgb8; do not invent individual decoder commits. Append the missing registered Rev5 producer slots. Run the canonical Rust trainer without historical replay, head N/H128/420 IDs, bounded 1-epoch/500-pair smoke with init 1101/sample 101; this is plumbing validation and is not a research recipe reproduction or a qualified model.


## Outcome and commands

The complete gate/owner map is [shippath_gate_map_2026-10-05.md](shippath_gate_map_2026-10-05.md); measured summary is [shippath_smoke_2026-10-05.json](shippath_smoke_2026-10-05.json). The map is MISSING-first; it distinguishes table admission from full model qualification and records every next owner chunk.

Admission view command is in [FEATURE_SET_IDS](../docs/FEATURE_SET_IDS.md#rev5-train-teacher-admission-view--october-5-2026). Scratch logs retain actual resource caps and exits.

Exact smoke command (bounded predeclared plumbing check):

```bash
TMPDIR=/var/tmp/shippath ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- target/debug/zensim_mlp_train --group safesyn:/var/tmp/shippath/teachers/safesyn_fit.parquet:1:0:withinref,both --group safesyn_development:/var/tmp/shippath/teachers/safesyn_dev.parquet:0:0.5:withinref,both --group cid22:/var/tmp/shippath/teachers/cid22_fit.parquet:1:0:withinref,both --group cid22_development:/var/tmp/shippath/teachers/cid22_dev.parquet:0:2:withinref,both --target-column human_score --target-scale 1 --hidden 128 --epochs 1 --pairs-per-epoch 500 --init-seed 1101 --sample-seed 101 --pair-sampling uniform --max-features 1853 --keep-features /var/tmp/shippath/keep420.txt --mse-weight 1 --early-stop-patience 0 --val-policy mean --val-aggregate geomean3 --out-dtype f32 --log-every 1 --no-auto-eval --nonneg-distance --out /var/tmp/shippath/smoke.bin
```

All four table admissions show explicit Rev5 identities, no issues, qualified_provenance=true and historical_replay=null. Canonical zenpredict::Model metadata reader verifies the same admission embedded in the output, producer identity and revision5. Original Parquet SHA equals copied SHA for each file; no feature/target/order changed. Original roots were not written. Rehashed recorded extraction binary equals c649e810…. Original decoder dependency commits remain unknown; new decoder era binds the actual executable and legacy-rgb8 route. The smoke does not solve historical input/encoder provenance limitations.

Canonical densify/predict/profile checks:

```bash
TMPDIR=/var/tmp/shippath ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- target/debug/bake_dial_refit densify --in /var/tmp/shippath/smoke.bin --out /var/tmp/shippath/smoke-dense.bin
TMPDIR=/var/tmp/shippath ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- target/debug/bake_dial_refit predict --bake /var/tmp/shippath/smoke-dense.bin --corpus /var/tmp/shippath/teachers/cid22_dev.parquet --score-units --out /var/tmp/shippath/cid22-dev-predictions.tsv
TMPDIR=/var/tmp/shippath ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- target/debug/bake_block_profile --bake /var/tmp/shippath/smoke-dense.bin --json --lines
```

Densification: 979628→225318bytes, 512 probe predictions BIT-IDENTICAL. Dense input count420, caller width720, H128, two layers, all420 declared IDs exactly match candidate list. Consumer basic+v2/rev5_localwin#62adfc93, populated producer basic+peaks+v2@w1825/rev5_localwin#36c3f3af. Embedded qualified admission survives. 3785/3785 oracle TRAIN-development predictions finite. No research quality/statistical-performance conclusion.

Completed checks (every heavy command wrapped as above):

- `cargo build --locked -p zensim-validate --bin zensim_mlp_train --bin bake_dial_refit --bin freeze_check`: pass, 78s wrapper, debug/optimized.
- `cargo test --locked -p zensim-validate --lib feature_set:: -- --test-threads=2`: 7passed, including Rev5 registered slots, absent-ID rejection, mixed revision/sampling/replay, missing decoder.
- `cargo test --locked -p zensim-validate --test feature_set_match`: 10passed, including real Rust slot hash and layout-free registry resolution.
- `python3 scripts/tests/test_rev5_teacher_admission.py`: 7passed, including byte-preserving copy, altered table/bank/receipt/chunk, human-path and sealed-path refusals before output.
- CI-exact `just clippy`: pass, 48s wrapper. No public API changed; no semver/API snapshot gate required for these private Python additions/registry/test edits.
- Initial `cargo fmt --all --check`: failed on newly written test formatting; `cargo fmt --all` corrected it; final check passed.
- Initial `just lint-scripts`: 812 scripts pass; final rerun includes the new test and is recorded below.

All source/model/table/binary hashes and actual argv retained in scratch and portable SHIPPATH_assets. Source execution used a dirty own workspace (recorded in admission receipt), and the final local commit binds the reviewed code. No full-data fit, protected label, EVAL/holdout read, full evaluation, selection, qualification, default change, fleet job or push.

Final closure checks: `just lint-scripts` checked 813 scripts, all runnable; final `cargo fmt --all --check` exited 0. Reviewed the complete diff and verified all report file:line references resolve in this workspace. Portable evidence will be finalized under `/home/lilith/tmp/zensim-paper/rev4/SHIPPATH_assets` before the DONE report is written last.

Local implementation commit: `5e8c28b0d8d84b2c2fb745943cc5d36ab10928d1`. All smoke receipt script hashes match the reviewed source in that commit. Evidence bundle created at `/home/lilith/tmp/zensim-paper/rev4/SHIPPATH_assets` with raw logs/admissions, model bytes and profiles, source snapshots, teacher-only bank manifests and frozen source receipts. Final manifest binds every copied artifact; no full corpora are copied. A second local closure commit records this completed evidence step; bookmark remains local and no push is performed. DONE is the final output-file write after commit/evidence verification.
