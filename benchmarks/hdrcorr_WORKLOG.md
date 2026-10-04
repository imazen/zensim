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

## HDRTEACH follow-on — 2026-10-04

Same existing workspace and local bookmark; parent HDRCORR commit09f6032f. No push,
model fitting, feature extraction or UPIQ dataset read. User explicitly authorizes
same 7,425 TRAIN / 3,900 registered VAL native pairs. Scratch `/var/tmp/hdrteach`.

Before any HDR-VDP-3 scores, `PREREGISTRATION.json` froze ppd60, quality task,
RGB BT.709 absolute nits, led-lcd-srgb emission, surround none, age24, all upstream
reference options and the agreement rule: exact-reference average tied midranks
may differ by <=1 position. Undefined groups (<2 or nonfinite) fail closed.
TRAIN uses fresh CVVDP JOD; VAL retains historic JOD and mixed labels separately.

### Native owner and build

zenmetrics commit348bde5bf96d635383965d80d9adb8f9c195fd14; live checkout remains
read-only. Live sibling ultrahdr-rs resolution requires unavailable zenjpeg^0.9.0.
Documented `scripts/ci/lock.sh --check --rev <commit>` exports tracked sources and
23 pinned sibling commits, passing unchanged. Initial default export copied ignored
scratch too broadly; stopped only our copy job, then used the tracked-revision route.
Lean CLI build hit existing CVVDP display feature gating; added cpu-cvvdp, no edits.

```bash
# In zenmetrics; LOCK_SNAP_DIR and TMPDIR are task-owned scratch paths.
env TMPDIR=/var/tmp/hdrteach LOCK_SNAP_DIR=/var/tmp/hdrteach/pinned-snap \
  ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
  scripts/ci/lock.sh --check --rev 348bde5bf96d635383965d80d9adb8f9c195fd14
# From pinned-snap/work/zenmetrics, isolated CARGO_TARGET_DIR:
cargo build --locked --release -p zenmetrics-cli --no-default-features \
  --features sweep,png,jxl,hdr,cpu-hdrvdp,cpu-cvvdp
cargo test --locked --release -p hdrvdp --test v3_reference -- --test-threads=1
```

Both cargo commands use run-heavy; sequential builds/tests capped at8 jobs. A briefly
started two-job reference-test build was suspended while the eight-job CLI build
finished, then resumed. All21 synthetic VDP3 golden cases and invalid-input checks
pass. These reference-port fixtures do not establish universal MATLAB bit equality.
CLI binary sha2561278c6935ad4ebbb624e6213c56ad672602b05433b288ea1f9274dbf02dee721;
lock sha256eff5add1a4714fdf8857bf6449cd8fb125972378e2d2a5ff851d8682d7e7a59a.
Native dispatch transports absolute RGB as f32 then widens to the f64 VDP3 metric;
no PQ-to-u8 scoring branch is taken. Declared cICP primaries are converted explicitly.

### Fixed-condition scoring and durable sharding

```bash
zenmetrics score-pairs --metric hdrvdp3 --hdr --hdr-common-primaries \
  --gpu-runtime cpu --hdrvdp3-ppd 60 --hdrvdp3-task quality \
  --hdrvdp3-input rgb-bt709 --hdrvdp3-emission led-lcd-srgb \
  --hdrvdp3-age 24 --hdrvdp3-surround none --group-by-ref \
  --pairs-tsv <keyed-native-shard.tsv> --out-parquet <fresh-output.parquet>
```

Opaque key `hdrteach://<original-role>/<row_id>` prevents row-order or basename joins.
Original row identities and ref/dist hashes are restored in final Parquet; VAL's
bit-identical producer aliases stay documented by HDRCORR admission. Native inputs
verified:12,120 files, all match frozen prior hashes. Pilots: smallest0.1411s;
largest3072x2490 took200.9132s, peak RSS3.23GiB. Timing is capacity planning, not a
benchmark claim. Disjoint reference shards use local/r7900x/i265/r5900xt, same binary;
two fixed shared PQ/JXL probe scores exactly equal across all four CPU hosts.

Every host uses run-heavy16G/nice19, <=8 single-thread workers, reserving128MiB+
550B/pixel per active worker within14GiB; three pairs per durable shard. Inputs are
SHA-checked remotely before scoring. Full exact coverage and finiteness are mandatory.
`--fail-on-bogus` is inappropriate per tiny shard: majority-identity/narrow-range and
nominal mean bounds can reject valid q_jod; the reference formula allows negatives.
Final admission preserves negatives and requires finite values<=10 and nonconstant
per-set outputs. Raw owner runtime="unknown"/absent viewing footer are retained honestly;
explicit CPU dispatch and complete viewing/binary provenance are added to final tables.

### Existing table and panel owners

`build_hdr_train_parquets.py --teacher-manifest` is a strict metric-only join over
admitted identities and pinned raw shards. It refuses missing/extra/duplicate keys,
q/codec/knob disagreement, mixed binary hashes and changed sources; never loads features
or changes roles. Its flag and rank residuals use scipy average midranks.
`hdr_route_panel.py --teacher-parquet` calls the canonical Rust panel in signed
SROCC-only mode (`--stats srocc`), with no logistic calibration fit. It emits all
per-reference/family records, worst keyed rows and full-population raw density plots.
Behavioral integration fixtures include shuffled scores/truth, ties, reversed ranking,
original VAL target retention and eight fail-closed controls.

### HDRTEACH urgent owner fleet restriction

Owner excluded r5900xt/dev and restricted allowed hosts to tower (Docker only), i265, i270, r3500, r3800x. Stopped only owned process groups on r5900xt/dev/r7900x; retained completed outputs (114/2098/141 rows). Verified no task scorer/runner on r5900xt after stop. i265 remains running unchanged. Reassigned every unfinished exact original key via /var/tmp/hdrteach/MIGRATION.json to tower/r3500/r3800x; i270 unreachable. Native two-row parity probes PASS on all three replacements. Hard memory and worker caps are recorded in decisions. Full11,325 scope and fixed viewing/agree rule remain unchanged. ETA contingency proposal is recorded, not executed.

Allowed-fleet queue rebalance: tower drained its active native children to successful completion; validated four real native outputs before writing recovery markers. No active scorer was cancelled in this rebalance. Only unstarted keys moved to serial stage2 on i265/r3500 (219/207 rows); initial queues must complete before these launch. Tower restarted Docker-only with all prior outputs retained and old execution plan/log preserved. REBALANCE.json /REBALANCE_AUDIT.json preserve custody and prove11325 unique original keys/3975chunks. Current pixel-capacity forecast ~6.97h remaining, no reduction applied.

Later observed capacity moved only the unstarted r3500 stage2 to tower Docker-only stage2 (207rows,804,389,376pixels), preserving exact keys/chunk IDs. Old unstarted r3500 plan disabled. No active native scorer interrupted. Final serial stages: i265 and tower; coverage remains3975chunks/11,325unique rows. REBALANCE2.json pins custody.

i265 initial native queue COMPLETE2,822rows, wrapperexit0 (16,226s). Started serial stage2 only after initial COMPLETE/native exit, with hard24G/reservation22GiB/max8single-threaded workers;27,400MiB system available at launch. Original condition/binary/keys preserved. Initial whole-run626.5rows/h is a resolution-mixed operational rate, not a benchmark.


2026-10-05T00:47UTC: r3800x drain completed with remote native-count0 and paused-task-PID proof. Recovered21 real rows from native exit0/0NaN logs and exact Parquet identities, retained all546 completed chunks. Terminated only paused scheduler, resumed1810-row plan. Reassigned246 unstarted rows/1,004,525,568pixels to serial i265 stage3 (24G/22GiB/max8) after stage2 COMPLETE. Tower initial2311 rows completed exit0;207-row serial Docker stage2 launched. First launch used incorrect run-heavy positional syntax, exited127 before any score; retained failure log and corrected to --mem/--jobs. Full-set coverage audit PASS3975chunks/11,325unique original keys. Updated forecast1.83hours remaining, resolution/load-sensitive; rates i265530.2/tower659.4/r3500309.6/r3800x407.8rows/hour. The755-row reduction proposal remains unexecuted; no native/science settings changed.


2026-10-05T01:05UTC: i265 serial stage2 COMPLETE (219rows), wrapper exit0. Started246-row stage3 only after that completion, with unchanged binary/view/row identities and hard24G/max8 single-thread workers. All allowed queues continue full scope.


2026-10-05T01:45:44UTC: i265 all3serial stages COMPLETE3287rows; r3500 COMPLETE1357; r3800x COMPLETE1810. Final whole-run observed rates502.0/287.1/401.2rows/hour respectively (resolution-mixed). Tower2380/2518at531.8rows/hour;138rows remain full scope, pixel forecast1.11hours. All original12120nativefileSHA reverifiedPASS before final admission; /var/tmp/hdrteach/verify-inputs-final.log. No subset run.


## Full-set completion and admission — 2026-10-05T02:26UTC

All 11,325 native rows /3,975 shards completed; every log exit0/0NaN, fixed CLI flags, input hash and exact unique key verified. TRAIN7,425/495refs/33families; VAL3,900/300/20; no dropped rows or scope reduction. Final allowed-host averages: i265502.02, towerDocker490.63, r3500287.11, r3800x401.16 rows/hour; scoring ETA0. Completed labels from dev/r5900xt/r7900x before the urgent rule remain with honest actual-host provenance. No owned scoring/runner or task Docker container remains on any prior/current host (CLOSURE_CHECK PASS).

Tower Docker-only actual owners/fixture/panel/independent audit completed at CPU1/hard2G, no network. Signed pooled SROCC TRAIN0.8283190440865288 / VAL0.8211975657571554; mean within-reference0.9988197386046849 /0.982051282051282, all795refs defined. Preregistered agree7390/7425(99.5286%) and3696/3900(94.7692%);35/204 disagree rows retained. Historic VAL mixed target auxiliary pooled0.862939 (not flag-defining). TABLE_AUDIT PASS all native/CVVDP bindings, roles, view JSON/inputSHA, midranks/flags and every per-reference signed-Spearman; max independent SciPy delta2.22e-16/1.11e-16. Behavioral fixture PASS shuffle/ties/signed reverse/boundary/negative/historic preservation and8fail-closed controls. Project just clippy and just lint-scripts PASS; native21synthetic golden cases plus invalid inputs PASS; bounded all-host2-row native probe exactly equal. No corpus-wide MATLAB parity claim.

Measured report lists worst rows/references/families and native examples with complete top3reference ladders per role; all-row score density counts checked. UPIQ calibration disclosure retained: paper sections3and5 explicitly useUPIQ>4000SDR/HDR images; no UPIQ images/labels read. Future student cannot claim UPIQ-independent testing. No model fitted, filtered, selected, promoted, default/API/bake/feature regime changed or repository pushed. Original HDRCORR outputs unchanged.

Scoped offline study board uses actual board owner and append-only TRAIN-development registry, without protected EVAL/bakes. Gate1 JavaScript parse PASS; full gate2 FAIL (no model SVG/ECharts/sortable EVAL rows), so full board qualification is MISSING. Scoped real browser PASS eight tables counts[2,2,10,10,10,10,10,10], both plot assets loaded, MISSING first, rule visible, study link served, zero runtime exceptions. Loopback server/Chrome stopped after checking; no publication/deployment.


HDRTEACH closure: archive member validation PASS 17622 members, 3975 native shards, 92899223 bytes, SHA256 `194d32efe411cd0f49dd03ae0db62f7ca7ab3b46e414f7252d4c21750f696c74`. Streamed sequential validation after replacing linked-file entries with regular portable files; all member SHA256 values match. Earlier slow/link verification logs retained as administrative evidence. Final source lint PASS811scripts and added-line whitespace PASS; scoped final browser rerun PASS, server/Chrome stopped. Canonical teacher targets and source owners were unchanged by packaging. Stable outputs `/mnt/v/output/zensim/hdrteach-2026-10-04/`; external assets/decisions under `~/tmp/zensim-paper/rev4/`; no push.


E26 Part A (2026-10-05): HDRCORR/HDRTEACH rebased onto main@origin8dc22f22. Main integrity audit admission, companion handling and exposure/history entries preserved; our metric diagnostics and head probe remain additive. Full414301-byte HDRCORR summary moved unchanged to tower output/zensim/hdrcorr-2026-10-04; pointerSHA0875932dadaddf361f24458c3cc9a1217e9c2a21ed471167931bf45fcf0b97b7. Every changed/new file in both commits is<=30000bytes unless already larger on main. Admission15tests, HDR teacher fixture, serving example compilation, Rust corruption tests, CI-exact just clippy and just lint-scripts passed. Checks/logs in /var/tmp/e26/partA-*. No push; original8db29245 preserved under quarantine/codex/hdrcorr-pre-e26.
