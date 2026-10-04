# HDRCORR worklog — 2026-10-04

Brief: `~/tmp/zensim-paper/rev4/HDRCORR_brief.md`. Own jj workspace `/home/lilith/work/zen/zensim--hdrcorr`, name `hdrcorr`, parent **4abdd866**, change **okywtzxt**, bookmark `quarantine/codex/hdrcorr`. Local-only; no push. Workspace retained for coordinator landing.

## Frozen scope and results

Read project CLAUDE/AGENTS, DATA_SPLITS, WAVE_PLAYBOOK, September15 priorities, HDR plans and canonical corruption activation/refit/serving owners. Read the provenance index and respected source roles. Explicit brief authorizes this jj workspace; coordinator checkout untouched. No T0/UPIQ labels or distorted images, corruption validation examples, fits, feature selection, calibration or threshold tuning. Decisions: `/home/lilith/tmp/zensim-paper/rev4/HDRCORR_decisions.md`.

Completed all six candidate recipes and shipped B/BHdr via production BakeScorer. Native HDR TRAIN 7,425 and registered VAL 3,900, zero score refusals/nonfinite values. Candidates remain Rev4. B/BHdr require own Rev1 processes (exact mixed-Rev4 refusal measured). codec_target HDR routing selects BHdr; raw B SDR weights on HDR are an extra labeled control.

Canonical corruption TRAIN: 9,036 raw attempts, 8,213 distinct source/pixel pairs, 7,725 catalog positives and 480 honest encodes, plus 8 inert identity controls. Exact coverage, finite scores, deduplication and honest q10/q20 cohorts enforced. Both existing tree heads refuse all six candidates at bake.rs:767. Rev4-compatible fitted companion and human HDR metrics remain MISSING. No qualification/default flip.

Main report: [hdrcorr_2026-10-04.md](hdrcorr_2026-10-04.md); [compact numerical summary](hdrcorr_2026-10-04.summary.json). All per-reference stats, raw scores, metadata, command records, refusals, tests and provenance remain in `/var/tmp/hdrcorr` and portable `~/tmp/zensim-paper/rev4/HDRCORR_assets/evidence.tar.gz`. Full-population density PNG/PDF exported there and visually checked.

## Input and serving provenance

- `/var/tmp/hdrcorr/FROZEN.json`: all eight unmodified byte hashes; `/var/tmp/hdrcorr/PROVENANCE.json`: production library/source, tool/lock/head hashes, compiler, platform, dependency pins and final fail-closed tool control. Root zenpredict 05de3cbc; bench existing pin 9a9be820. Native decoder and canonical panel exact binary hashes pinned; producing commits unknown. No speed claim.
- TRAIN admission `/var/tmp/zensim-validation-2026-09-15/hdr-corrected/ADMISSION.json`; 495 references/33 families, PQ cICP, native PNG16/JXL16. Truth `/var/tmp/zensim-validation-2026-09-15/hdr-corrected/judges/{cvvdp,ssim2}.parquet`; exact row IDs/source/knob checks. See report for declared mix formula.
- VAL authority `/mnt/v/output/zensim/hdr944-leg/hdr_v3mix944_valdigits_2026-08-03.parquet`, 300 references/20 origins. `prepare_val.py` exact old-feature identity match; no old-feature prediction; byte-identical aliases only. Historic truth era explicitly retained. `hdr-val-admission.json` pins authority and join.
- Corruption authority `/mnt/v/output/zensim/canonical-corruption-2026-09-08/{train,honest-train-keyed}.parquet`; canonical TRAIN only. Honest native bitstreams decoded with copied zen-owned verifier. PNG raw RGB pixel digests and source/bitstream hashes checked before serving. `corruption-metadata.json` and decode TSVs retain provenance.
- `INPUTS_VERIFIED.json`: 7,948 pre-pinned hashes unchanged; 4,200 VAL files separately hashed after scoring, timing disclosed.
- Final HDR helper changed failure-exit handling after full runs; earlier scoring binary and final binary hashes separately recorded. `HDR_SCORER_CONTROL.json`: expected rc1 for 16 mixed-revision baseline refusals; candidate outputs bit-identical on 8 rows.

## Replay commands

Run from this workspace. Use `TMPDIR=/var/tmp/hdrcorr`, `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. All heavy build/scorer/statistical processes ran through `run-heavy --mem 16G --jobs 8 --` (wrapper nice priority), thread caps as below. Scratch preparers and exact metadata are archived. Fresh scoring output paths are required because owners refuse overwrite.

### hdr-train-command.json

```bash
env TMPDIR=/var/tmp/hdrcorr ZENSIM_FORMULA_REV=4 RAYON_NUM_THREADS=1 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- zensim-bench/target/release/examples/hdr944_extract --ref-root /mnt/v/output/imazen-26-hdr-grid-2026-06-14 --input-contract hdr-common-primaries-v2-cicp-pq10000 --threads 8 --out /var/tmp/hdrcorr/hdr-train-candidates.tsv --pairs /var/tmp/zensim-validation-2026-09-15/hdr-v2/pairs-0.tsv --enc-root /mnt/v/output/zenmetrics/datagen-2026-06-23-hdr/enc/zenjxl --pairs /var/tmp/zensim-validation-2026-09-15/hdr-v2/pairs-1.tsv --enc-root /mnt/v/output/zenmetrics/datagen-2026-07-03-hdr-hq/enc/zenjxl --score-bake /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s0.bin --score-bake /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s1.bin --score-bake /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s2.bin --score-bake /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s0.bin --score-bake /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s1.bin --score-bake /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s2.bin
```

### hdr-val-command.json

```bash
env TMPDIR=/var/tmp/hdrcorr ZENSIM_FORMULA_REV=4 RAYON_NUM_THREADS=1 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- zensim-bench/target/release/examples/hdr944_extract --ref-root /mnt/v/output/imazen-26-hdr-grid-2026-06-14 --input-contract hdr-common-primaries-v2-cicp-pq10000 --threads 6 --out /var/tmp/hdrcorr/hdr-val-candidates.tsv --pairs /var/tmp/hdrcorr/hdr-val-0.tsv --enc-root /mnt/v/output/zenmetrics/datagen-2026-06-23-hdr/enc/zenjxl --pairs /var/tmp/hdrcorr/hdr-val-1.tsv --enc-root /mnt/v/output/zenmetrics/datagen-2026-07-03-hdr-hq/enc/zenjxl --score-bake /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s0.bin --score-bake /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s1.bin --score-bake /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s2.bin --score-bake /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s0.bin --score-bake /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s1.bin --score-bake /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s2.bin
```

### hdr-train-baselines-command.json

```bash
env TMPDIR=/var/tmp/hdrcorr ZENSIM_FORMULA_REV=1 RAYON_NUM_THREADS=1 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- zensim-bench/target/release/examples/hdr944_extract --ref-root /mnt/v/output/imazen-26-hdr-grid-2026-06-14 --input-contract hdr-common-primaries-v2-cicp-pq10000 --threads 2 --out /var/tmp/hdrcorr/hdr-train-baselines.tsv --pairs /var/tmp/zensim-validation-2026-09-15/hdr-v2/pairs-0.tsv --enc-root /mnt/v/output/zenmetrics/datagen-2026-06-23-hdr/enc/zenjxl --pairs /var/tmp/zensim-validation-2026-09-15/hdr-v2/pairs-1.tsv --enc-root /mnt/v/output/zenmetrics/datagen-2026-07-03-hdr-hq/enc/zenjxl --score-bake /home/lilith/work/zen/zensim--hdrcorr/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin --score-bake /home/lilith/work/zen/zensim--hdrcorr/zensim/weights/bhdr_linear_shaped_cvvdpmix_byid_2026-09-06.bin
```

### hdr-val-baselines-command.json

```bash
env TMPDIR=/var/tmp/hdrcorr ZENSIM_FORMULA_REV=1 RAYON_NUM_THREADS=1 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- zensim-bench/target/release/examples/hdr944_extract --ref-root /mnt/v/output/imazen-26-hdr-grid-2026-06-14 --input-contract hdr-common-primaries-v2-cicp-pq10000 --threads 6 --out /var/tmp/hdrcorr/hdr-val-baselines.tsv --pairs /var/tmp/hdrcorr/hdr-val-0.tsv --enc-root /mnt/v/output/zenmetrics/datagen-2026-06-23-hdr/enc/zenjxl --pairs /var/tmp/hdrcorr/hdr-val-1.tsv --enc-root /mnt/v/output/zenmetrics/datagen-2026-07-03-hdr-hq/enc/zenjxl --score-bake /home/lilith/work/zen/zensim--hdrcorr/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin --score-bake /home/lilith/work/zen/zensim--hdrcorr/zensim/weights/bhdr_linear_shaped_cvvdpmix_byid_2026-09-06.bin
```

### SDR corruption production serving

```bash
env TMPDIR=/var/tmp/hdrcorr RAYON_NUM_THREADS=1 ZENSIM_FORMULA_REV=4 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- target/release/examples/serve_custom_bake --pairs /var/tmp/hdrcorr/corruption-pairs.tsv --shard 0/2 /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s0.bin /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s1.bin /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s2.bin /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s0.bin /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s1.bin /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s2.bin > /var/tmp/hdrcorr/corruption-candidates-0.tsv
```

```bash
env TMPDIR=/var/tmp/hdrcorr RAYON_NUM_THREADS=1 ZENSIM_FORMULA_REV=4 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- target/release/examples/serve_custom_bake --pairs /var/tmp/hdrcorr/corruption-pairs.tsv --shard 1/2 /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s0.bin /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s1.bin /home/lilith/tmp/chromaq/bakes_full/byv2fy-full-s2.bin /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s0.bin /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s1.bin /home/lilith/tmp/chromaq/bakes_full/v2basic-full-s2.bin > /var/tmp/hdrcorr/corruption-candidates-1.tsv
```

```bash
env TMPDIR=/var/tmp/hdrcorr RAYON_NUM_THREADS=1 ZENSIM_FORMULA_REV=1 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- target/release/examples/serve_custom_bake --pairs /var/tmp/hdrcorr/corruption-pairs.tsv /home/lilith/work/zen/zensim--hdrcorr/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin /home/lilith/work/zen/zensim--hdrcorr/zensim/weights/bhdr_linear_shaped_cvvdpmix_byid_2026-09-06.bin > /var/tmp/hdrcorr/corruption-baselines.tsv
```

### Existing head attachment probes

For each head path in PROVENANCE.json, run the final serve helper with `--head-probe HEAD` plus the six bakes at env Rev4; run the same two baseline paths at env Rev1. Store keyed outcome TSVs. No mutation of any head.

### Canonical assessment and exports

```bash
env TMPDIR=/var/tmp/hdrcorr OPENBLAS_NUM_THREADS=1 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 /var/tmp/hdrcorr/assess_hdr.py train candidates
env TMPDIR=/var/tmp/hdrcorr OPENBLAS_NUM_THREADS=1 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 /var/tmp/hdrcorr/assess_hdr.py train baselines
env TMPDIR=/var/tmp/hdrcorr OPENBLAS_NUM_THREADS=1 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 /var/tmp/hdrcorr/assess_hdr.py val candidates
env TMPDIR=/var/tmp/hdrcorr OPENBLAS_NUM_THREADS=1 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 /var/tmp/hdrcorr/assess_hdr.py val baselines
env TMPDIR=/var/tmp/hdrcorr OPENBLAS_NUM_THREADS=1 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 scripts/v_next/corruption_gate_eval.py --score-table-report --metadata /var/tmp/hdrcorr/corruption-metadata.json --scores /var/tmp/hdrcorr/corruption-candidates-0.tsv --scores /var/tmp/hdrcorr/corruption-candidates-1.tsv --out-json /var/tmp/hdrcorr/corruption-candidates.json
env TMPDIR=/var/tmp/hdrcorr OPENBLAS_NUM_THREADS=1 /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 scripts/v_next/corruption_gate_eval.py --score-table-report --metadata /var/tmp/hdrcorr/corruption-metadata.json --scores /var/tmp/hdrcorr/corruption-baselines.tsv --out-json /var/tmp/hdrcorr/corruption-baselines.json
python3 /var/tmp/hdrcorr/verify_inputs.py
python3 /var/tmp/hdrcorr/plot_hdr.py
python3 /var/tmp/hdrcorr/write_report.py
```

`assess_hdr.py` calls existing scripts.lib.zen_stats panel_batch_indexed/scatter over the full population, not new correlation math. Report generator only formats stored statistics. Duplicate/finiteness/coverage/tie orientation checked by Python fixtures.

## Implementation and completed checks

Existing example `hdr944_extract` gained repeated --score-bake, native BakeScorer scoring and keyed fail-closed refusals; extractor/audit contract retained. Existing serve helper gained feature-gated --head-probe. Existing corruption_gate_eval owner gained TRAIN-only scored-table descriptive evaluation. No library/public API change. docs/DATA_SPLITS adds the explicit exposure ledger.

All passed:
- `cargo test -p zensim --lib --features custom-profiles,corruption-head corruption_head -- --test-threads=2` (27 tests).
- `python3 scripts/tests/test_integrity_admission.py` (10 tests; meaningful AUC tie/orientation, dedup, exact coverage and forbidden-role fixture).
- Standalone HDR native decoder test (low-bit precision and conflicting transfer refusal), log test-hdr.log.
- `cargo test -p zensim --lib --features custom-profiles,corruption-head b_routes_to_hdr_weights_on_the_nits_path -- --test-threads=1` (1).
- CI recipe `just clippy`; standalone `cargo clippy --manifest-path zensim-bench/Cargo.toml --example hdr944_extract --features hdr944-extract -- -D warnings`; `just lint-scripts` (810 scripts).
- Format/whitespace and scoped board/report checks recorded below before commit.

## Local board and handoff

Append-only discussion set 2026-10-04-hdrcorr-rev4, role train-development, report URL /zensim/reports/hdrcorr-2026-10-04/. Local report files only; no deployment. Build a study-only gauntlet using an empty fulleval directory and absent loop/HFNL summaries. This deliberately avoids protected stored full-verdict labels and creates no qualification rows. Full compare-set gates require at least two EVAL rows and cannot pass this zero-row board; report any scoped failure explicitly. Browser check the local report and verify all measured tables/density are visible.

`HDRCORR_DONE.md` is written only after artifacts, verification and the local jj commit are complete. Leave workspace in place for coordinator; never push.

### Actual final checks and board limitations

- `cargo fmt --all --check` and `rustfmt --edition 2024 --check zensim-bench/examples/hdr944_extract.rs`: passed.
- Native HDR test command: `cargo test --manifest-path zensim-bench/Cargo.toml --example hdr944_extract --features hdr944-extract -- --test-threads=2`.
- Build commands: `cargo build --release -p zensim --example serve_custom_bake --features custom-profiles,corruption-head`; `cargo build --release --manifest-path zensim-bench/Cargo.toml --example hdr944_extract --features hdr944-extract` (both capped by run-heavy).
- Scoped board CLI attempt refused empty fullevals at `scripts/v_next/gauntlet.py:1068`. Invoked existing `gauntlet.build_html([], out, loop_targeting=None, hfnl_axis=None)` directly; no stub qualification rows or target files read.
- `gauntlet_gates.sh`: script parse PASS, full DOM/model gate FAIL as expected for zero EVAL rows: no SVG model plots, no ECharts model mounts/options, too few scoreboard rows to test (0). Compare-set/strict EVAL gates cannot complete. This is not a full board gate pass.
- Actual Chromium scoped report check PASS: table row counts `[8,8,8,8,8,5,16,8,11]`, full density loaded, MISSING-first disclosure, no JS page exceptions. Study entry/link visible after opening TRAIN disclosure. Initial assertion tested hidden disclosure text before opening; corrected harness to open the actual disclosure; page content unchanged. Screenshot visually inspected. `BROWSER_VERIFIED.json` retains exact checks; `browser_check.mjs` harness archived.
- Parent production library files unchanged; only examples, existing diagnostic owner/tests, report/discovery and split ledger changed. All final failure text has owner file:line.

- Registry report URL also served transiently on loopback http://127.0.0.1:18784 for the Chromium check; report and study disclosure both passed. Server stopped after checks. No public publication.
- Final added-line whitespace check passed; all eight frozen bake hashes reverified and all four native HDR refusal sidecars empty. Discovery entry preserves every existing set unchanged. `FINAL_CHECKS.json` records these assertions.
