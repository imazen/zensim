# Production Rev5 release gates — 2026-10-07

**Blocked:** no harvested, qualified D1 production model is available. This map
and the synthetic D2 harness are ready; neither replaces B nor qualifies a
model. No KADID TERMINAL label was opened. Work starts from zensim main
`b47599be`, after SHIPPATH landed at `3f273924` and zenmetrics `e056defc`.

The controlling contracts are [Rev5 spec](rev5_spec_2026-10-04.md),
[production scorecard](../docs/MODEL_SELECTION_SCORECARD.md),
[current split ledger](../docs/DATA_SPLITS.md),
[D2 registration](kadid_terminal_registration_2026-10-07.md),
[activation contract](../docs/CORRUPTION_ACTIVATION_2026-09-13.md), and
[target protocol](../docs/TARGET_STEERING_PROTOCOL_2026-09-08.md).
[Research status §16](rev5_research_status_2026-10-05.md) fixes launch order:
E28 completes, E30 is assessed, then production fits. Earlier five-source
research, R7 confirmation and stamped Rev5 bakes do not become qualified D1
evidence. R7 sets are spent; AIC-4's R7 read is possibly contaminated by AIC-3.
D1 excludes every AIC-family training source. D3 makes only UPIQ-380 HDR TRAIN;
its remainder stays T0 and it cannot independently test a student trained on
HDR-VDP-3, which was calibrated on UPIQ.

`done` below applies to the named implementation/evidence only. `ready` means
the instrument can be used after explicit admission and freezing. Every
model-dependent pass must name the exact final packed composition, companion,
threshold, decoder, populations and evaluator. Missing/stale/failed evidence
blocks replacement. No commands below authorize data exposure or fleet work.

## Gate inventory

Owner commands are relative to the repository unless a crate path is named.
`$BV`, `$FC`, `$PANEL`, `$DM` denote hash-pinned release binaries
`bake_verdict`, `freeze_check`, `panel`, `diffmap_block_coherence`; `$FINAL`
denotes immutable packed final bytes, `$HEAD` the ZCTH v4 companion and
`$EVAL` an explicitly admitted Rev5 root. Every optional corpus/grid path must
be explicit; historical defaults may reach labels that are not authorized.

| Gate / owner command | Required inputs and pass rule | Status for production | Protected reads |
|---|---|---|---|
| **E30 / D1 population:** `python3 scripts/rev4_featpot/e30_four_source.py score --root "$D1" --results "$E30_RESULTS" --control-root "$E24" --control-pins "$CONTROL_PINS" --out "$E30_REPORT"` | Completed, hash-bound report for 40 nA3 cells/four folds/seeds0–9 and pinned E24 controls. Report E21 deltas as removal cost: **no numerical pass/adoption rule and not a gate on D1**. Externals report-only; no AIC fold. | **blocked on E30 completion and report binding**; E28 complete on current main. D1 remains fixed even if reported deltas are negative. | Only D1 design TRAIN populations; no AIC labels. |
| **Table provenance:** canonical strict fit/harvest; `$INSPECTOR "$FINAL"`; `$BV … --fulleval …`, `$FC --qualify --fulleval …` | All seven per-table admissions; exact D1 four-source receipt, feature-set/revision/decoder, row-selection/key/order hashes, producer/sampling/tool/recipe identities. No inferred/unknown leg or historical replay. Epoch119 and 120×50000 budget, registered seeds. Every companion independently admits. | **done plumbing; blocked on registered fits**. Short smokes cannot qualify/install as full cells. | TRAIN only; metadata guards refuse protected sources/bindings. |
| **Rust surface / final identity:** `serve_custom_bake --corruption-head "$HEAD" "$FINAL" "$REF" "$DIST"`; pixel/cache audit through existing extractor and BakeScorer | Complete composition runs through public Rust surface; spline, ensemble/routing, head/threshold all included. Exact consumed IDs and tree raw/fire parity; complete pixel/cache/prepared precision contract, identity100. No Python inference or revision stamp as admission. | **ready; blocked on final composition choice and fits**. Earlier serving proofs use research weights. | Explicit TRAIN probes only. |
| **G-RANK / board axes:** `$BV --bake "$FINAL" --features-root "$EVAL" --corpora "$AUTHORIZED_CORPORA" … --fulleval "$VERDICT"`; `$PANEL --input "$PAIRS" --json --per-group`; `$PANEL --input "$PAIRS" --json --scatter` | Full human aggregate and each supported content aggregate ≥SSIM2, incumbent CID22-band performance, no collapse. Preserve per-corpus signed SROCC, KROCC, logistic PLCC and raw Pearson, PWRC/OR/Z-RMSE, bands, within-reference means, coverage, raw scatter/tails/saturation and counts. Existing freeze selection floors F4/F5/F7/F8 are reported; selection is separate from qualification. | **blocked on final model, authorized exposure and complete compatible labeled instrument**. E24/E25 are research evidence; NITS loss remains disclosed. **Owner decision:** freeze exact supported content aggregates, human populations and any currently undefined no-collapse/band comparison rule before reading. | Yes for human EVAL/T0; prior exposure and authorization required. Exposed D1/LODO and R7 results cannot be called fresh tests. |
| **G-DIAL:** `$BV … --dial-grid "$GRID" --gaddr-grid-truth "$DIAL_TRUTH"` | Registered standard-grid p5≤25, p95≥85, monotonicity≥.93; quantize before TRAIN-only calibration, evaluate final bytes and all ties/order. Replacement grid is July's 4424 rows, not unavailable May pixels. | **ready features; blocked on final model and bound mentor/grid truth**. | Synthetic/metric grid can be TRAIN; any human binding needs separate exposure. |
| **G-ADDR / five codec floors:** `$BV … --dial-grid "$LADDER" --gaddr-grid-truth "$DIAL_TRUTH" --gaddr-json "$ADDR" | CONTRACT and REGRESSION pass; A7r for avif-rav1e, avif-svt, jpeg, jxl, webp all meet registered mentor representability on identical cells. Floor-dense distinct-encode ladders, duplicates and encoded/pixel hashes accounted. `freeze_check` verifies same-scorer ladder hash and each codec floor. | **blocked on Rev5 ladder registration/mentor binding and final measurement**. 9593 features prepared. Changed table SHA needs a new canonical registration row; do not transfer old gate receipt. | No human labels in delivered feature table; mentor truths require explicit source admission. |
| **Negative tails / identity:** `$BV … --negtail-probe "$TAIL" --identity-probe "$IDENTITY"` | C1 mono≥.93, C2 tied≤.05, C3 some all-negative-truth score<0, C4 deepest probe<0, C5 feature-inference identity band[97.5,100], pixel identity exactly100, C6 no distortion above identity. No dial clamp. Original 2000 tails and38 identities retained. | **ready features; blocked on final composed model/peer truth**. | No protected label required on registered synthetic probes. |
| **G-STEER:** `$DM "$REF" "$DIST" --bake "$FINAL" --block 8 --json "$OUT"`; `scripts/m3a_sweep.sh --bake "$FINAL" --bin "$DM" --grid full --label "$NAME" --logdir "$OUTDIR"` | M2≥.99, M3≥.70; complete finite-repair feature/scalar/density replay, local_refine neighbor-exact or explicit refusal. Broad96 (24×8/16/32/64), owner12 (4×3 seeds), KADID JPEG8×8, full27 M3/M3a fixtures. Identity and geometry/padding/stride cases included. | **ready base-pair features; blocked on final composition support/measurements**. Research93/96 broad is not full pass (minimum M2 .957894737). M3 wrapper does not forward companions; extend existing owner before complete-composition use. | TRAIN spatial fixtures; no terminal labels. |
| **STEERCODEC / CHROMAQ:** registered `benchmarks/r5steer_2026-10-04/run2.py`, `benchmarks/chromaq_2026-10-04/analyze.py "$SWEEP444" "$SWEEP420" "$SCORES" "$OUT"`; codec-owned `zensim_diffmap_rd`/`zensim_cq_rd` | Rev5 spec requires rerun broad/owner/JPEG, CHROMAQ and JXL/zenjpeg/zqi map-guided vs oracle/random quality-swap allocations at equal block share, judged by fixed zensim, SSIM2 and butteraugli on identical decoded bytes. Preserve signed interventions and honest low-quality anchors. | **blocked on final bytes and parametrizing historical drivers** (their hardcoded research models must not be run as production tests). **Owner decision:** no additional CHROMAQ, zqi or equal-share acceptance threshold is registered; report diagnostics, obtain rule before assessment. | Registered TRAIN scenes only; any expanded population requires admission. |
| **G-RD / spatial value:** codec owners `jxl-encoder/examples/zensim_diffmap_rd.rs --native-interventions`, `scripts/v_next/rd_probe_analyze_2026-07-18.py --interventions "$NATIVE_PACKET"`; JPEG/AVIF/WebP owners in target protocol | Active/neutral/intervention controls in each supported codec; vs strong scalar controller ≥0% geometric-mean byte savings on every independent judge, ≥1% on one, no content aggregate regression. Overlapping judged-quality intervals, source-bootstrap uncertainty, all emitted bytes/pixel/quantizer/judge and cost pins. Own-score improvements alone fail this gate. | **blocked on final-model experiments/binding**. Analyzer `--interventions` currently asserts historical model SHA `cd1098b4…` (line554); parameterize and bind the final production model before using it for this gate. No sibling edits/runs authorized here. | TRAIN fit/calibration and separately authorized EVAL source families; not human terminal labels by default. |
| **G-TARGET:** `cargo run --release --manifest-path zensim-target/Cargo.toml --example demo_matrix -- --source-manifest "$SOURCES" --compositions "$COMPOSITIONS" --calibration "$TRAIN_CAL" --budgets 1,2,3 --codecs "$CODECS" --out "$FRESH"` plus native codec owners | Witness each attainable range before requests, hide bounds from runtime; TRAIN-fit seeds. 1shot median≤2/p95≤8/undershoot>8≤5%; 2shot≤1/≤3/>3≤5%; 3shot≤.5/≤1/max≤3/>1≤1%. Per codec/configuration/SDR-HDR lane; same requests,100% dispositions, uncertain/unattainable separate, count native reconstructions/map work. | **blocked on feasibility registration, final composition and native controller measurements**. Use demo_matrix’s admitted bounds/compositions route; legacy --source/--bake demo mode is not the complete exam. | TRAIN calibration, authorized EVAL pixels/judges. Human/protected bindings only by separate authorization. |
| **Integrity / ZCTH v4:** `python3 scripts/v_next/train_corruption_head.py --refit-admission-manifest "$TRAIN_REFIT_PIN" --out-dir "$FRESH"`; `$BV … --corruption-head "$HEAD" --corruption-head-threshold 0.9 --corruption-grid "$REVIEWED_GRID"`; `python3 scripts/v_next/corruption_gate_eval.py --integrity-admission "$ADMISSION" --audit-jsonl "$AUDITS" --out-json "$REPORT"` | Header/numeric/admission digest validates; TRAIN source-table/declaration/role/decoder and selections match. Activation separate from lowering: zero honest native activations/lowering, overall≤1%, unique detection≥95%, real bugs≥90%, legacy belowq20≥99%,100% tested non-inert RGB swaps. Reviewed catastrophic/recoverable/ambiguous strata and source-matched anchors per activation contract. | **done companion TRAIN admission/parity**, **blocked on final composition and untouched source-family EVAL/severity screen**. [SHIPPATH6](shippath6_WORKLOG.md) refit has identical numeric sections and seven unchanged TRAIN gates; this is not EVAL qualification. **Owner decision:** freeze unresolved stratified bars/anchors before new evaluation, never universalize catalog positives. | TRAIN companion record only; EVAL corruption labels/source screen require explicit admission. No default protected reference-list rehash. |
| **Rev5 correctness:** `cargo test -p zensim --release --all-features --test featcanon_rev5_parity`; feature_invariants, per_bake_revision, legacy tier audit and steering tests | Bit parity v4x/v4/v3/scalar/wasm128/NEON, fixed virtual lanes/FMA; stable moments two-pass reference within1e−6 relative and positive crop; exact Difference/Similarity identity, computed ReferenceOnly; Rev1–4 bytes unchanged. Per-family accuracy vs exact no worse than Rev3. Freeze60174678 output arithmetic. | **done landed arithmetic gates** with native/WASM/i686/NEON evidence in [Rev5 worklog](rev5_WORKLOG.md); final artifact/build regression rerun still **ready**. No values changed here. | Synthetic and explicitly admitted TRAIN audit pairs only. |
| **Runtime/memory / work census:** `cargo bench -p zensim --bench extract_paths_bench` with `ZEN_XP_EXTERNAL_MODELS` and pinned tier/thread/geometry; existing zenbench owner matrix | Quiet release9950X3D, no target-cpu=native,30 accepted paired rounds; v4x/v3,1thread64²/256²/1MP/4MP vsRev3/4, fixed+per-pixel fit; disclose MT scaling. Uncached complete p95≤50/200ms at1024²/2048²,≤1.25×D and≤SSIM2; cached score+map≤3×uncached; peak incrementalRSS≤128B/pixel+64MiB/worker, codecRSS separate. Work census no duplicate plane/family/channel work, no warm allocations/zero fills; certified three consecutive kernel attempts gain<2%. | **blocked on quiet-box qualification and final composition**. Landed timing and stop rule provisional; warm census passed only registered research configurations. Rev5 scalar/WASM **speed** requirement withdrawn; scalar/WASM correctness retained. “Scalar score” cost row is complete score path on supported SIMD, not a renewed scalar-tier speed requirement. | No protected data necessary; admit/pin explicit benchmark sources. |
| **Input/serving / bake-format / API:** `cargo test --workspace --all-targets --all-features --exclude zensim-wasm-tests`; `cargo test --workspace --doc`; `just clippy`; `just api-doc-check`; feature-permutation/serving tests; `cargo semver-checks` for release | ZNPR v3 canonical loader, declared IDs/revision and all admission/provenance survive pack; legacy ZCTH1/2/3 bytes/readers unchanged, v4 rejects tampering/numerical-contract mismatch. Final pixel/cache/prepared/HDR and named-profile binding; supported transfer/primaries/luminance/alpha/depth/ICC/stride/geometry matrix, explicit unsupported errors. No stale public snapshot, supported API break or serving switch. | **ready checks; blocked on final shipping binding and existing validation registry owner decision** (`basic_only_bake_compatibility_respects_partial_producers`, Known Bugs). Do not weaken expectation. Release also needs explicit owner README/release approval, platform CI including WindowsARM/macOSIntel/i686, and zenpredict v3 publication prerequisite. | Tests must use synthetic/explicit TRAIN fixtures; opt-in corpus gates never scan terminal labels. |
| **HDR scope:** native `BakeScorer::compute_hdr`/prepared ingress checks; `scripts/hdr/hdr_route_panel.py … --parquet "$ADMITTED_HDR"`; HDR rank/dial/spatial/RD/target/cost owners | Correct native PQ/cICP/common-primary absolute nits and precision. SDR fits may demonstrate API transfer, not human HDR accuracy. HDR judges on interpreted native pixels; same release rows per supported HDR lane. One public score needs registered alignment/qualified mixed TRAIN or justified metadata head. | **blocked on owner supported-scope decision and independent human HDR/display study**. E26hd4 within-ref passes but pooledHDR degrades; E27 neither arm passes. D3 UPIQ TRAIN and teacher agreement are not independent validation. No “HDR qualified” claim; no silent restriction of supported inputs. | Registered HDR VAL already exposed; fresh human HDR needs new authorization; restUPIQ remainsT0. |
| **KADID TERMINAL / D2:** `python3 scripts/rev4_featpot/kadid_terminal_read.py --receipt "$COMMITTED_PIN" --authorization "$AUTH" --output "$NEW_RESULT"` | Exactly2000 original stimuli; report signedSROCC,KROCC,PLCC,within-refSROCC,per-typeSROCC,full scatter. Production signedSROCC delta vsB≥−2SE, paired **reference** bootstrap10000 draws, fixed seed in pin; delta vsresearchRev4by_v2fy≥−.005. Single final frozen composition, after every pre-terminal gate passes. Failures spend set too. | **ready harness; blocked on fits, release gates, complete stimulus mapping and coordinator authorization/pre-read committed pin**. Final seed/ensemble/research comparator identity is an **owner decision before labels**; no best-on-terminal selection. | **Yes, exactly one authorized read. Forbidden in this lane.** |

The current `freeze_check::qualification_report` covers surface/provenance,
ladder identity, CONTRACT/REGRESSION, five floors and five product artifacts.
It omits integrity, HDR scope, runtime/memory, supported inputs and some spec
diagnostics. Terminal authorization requires coordinator-reviewed supplemental
evidence for all rows; its PASS alone is incomplete. No new bar is introduced.

## Harvest to serving candidate: fixed identities and command chain

Preparation lives at `/mnt/v/output/zensim/shippath11-2026-10-07` (`$A`).
[SHIPPATH11 pointer](shippath11_READY_2026-10-07.pointer.md) owns the inventory.

| Input | SHA-256 |
|---|---|
| Production `fit-manifest-fitv2d1-20261007.json` | `236eb2feb7861c05e5bc437b1063674266dca13f7c46a865c5b5e5c0e6025e84` |
| E30 manifest | `acf29dcb9c061870ee23c892e2143e56a91e8d5730f0fa0599cf5a824a1a0282` |
| `image-context/program.tar.gz` | `6d491eb165a00b70297de0864ec2e749818ca81491207a7da3fa6e36c2028cd0` |
| `d1-fit-data.tar.gz` | `8f2dae5e024670c02bbb81e5ed18254a5132f4423fc672f6aeb1059560bba084` |
| D1 decision | `1baaa0ae980757d69e351240cb9c0c4d3c5369bde2eb52addafcd49845dbe94a` |
| Fresh D1 frozen view | `4e1c0d997acea3a985f21c858ced0231d832a6ee27f34568dbc98ac450653999` |
| ZCTH v4 `$HEAD=/var/tmp/shippath6/refit/head.zcth` | `568380f1bd2fe6202ebcc5a194078d7885035f466abe93ada3a4b3ea8f4a26f2` |

Image `ghcr.io/imazen/zenfleet-worker:fit-d1-e30-v39-w925f9783329f`, image ID
`sha256:f9a7ff6436fd9e9d6d8fbb1f108b615753f65dc2ff1974dd2e22f956383b5e5c`,
worker build `925f9783329f`. Program inventory pins every binary/source file;
the trusted fit contract is
`benchmarks/shippath_qualified_fit_contract_2026-10-07.json`. No fleet command
was run by RELEASEGATE. Launchers retain their authorization-file refusal.

Harvest registered cells using `$A/committed-tools/harvest_fit_cells.py` and the
coordinator's existing explicit manifest/IDs/ledger/blob-prefix/endpoint/scratch/
rescue arguments, adding **exactly**:

```text
--program-archive /mnt/v/output/zensim/shippath11-2026-10-07/image-context/program.tar.gz
--checkpoint-inspector /mnt/v/output/zensim/shippath11-2026-10-07/bin/inspect_qualified_checkpoint
```

Do not use `--allow-local-smoke`. This verifier admits actual selected/packed
metadata, seven table pins,120×50000 budget,epoch119,seeds and receipt/freeze/
decision against the trusted program contract before installation. The old
running E28 harvest owner has a separately recorded budget-verification gap;
do not substitute it for this production verifier. Coordinator supplies the
transport values; copying a guessed endpoint or shell command is not a pin.

Each `confirm/cells/sel:59f0bbc2f290@h32:H128:cv16:cf98__N/full_s{0,1,2}`
contains `result.json`, `fleet_receipt.json`, original selected `refit/last.bin`,
`keep_features.txt`, development curve/log, canonical dense model and
`refit/production-f16.bin` plus pack log. `result.json` binds selected SHA,
packed SHA,dense SHA,TRAIN calibration SHA,deployment order and admissions;
fleet receipt binds exact program/data/job/cell/blob. These are full-data
TRAIN-development reports, never a new confirmatory label read or seed choice.

The registered executor already performs this chain through
`v2_production_pack.pack_production`; preserve its artifacts. For an explicit
replay into **fresh output outside all immutable inputs**, use the pinned
`$A/bin/bake_dial_refit`:

```bash
"$A/bin/bake_dial_refit" densify --in "$CELL/refit/last.bin" --out "$FRESH/dense.bin"
# Require its BIT-IDENTICAL probe gate, exact420 IDs and preserved admission.
"$A/bin/bake_dial_refit" pack --in "$FRESH/dense.bin" --out "$FRESH/production-f16.bin" \
  --dtype f16 --zerobias-bulk 0 --protect-last --neg-tail \
  --anchor "$D1/wide/main/real/cid22_fit.parquet" --target-col human_score --verify none
"$A/bin/inspect_qualified_checkpoint" "$FRESH/production-f16.bin"
```

The Rust pack command quantizes bulk f16, protects final f32, **then** calibrates
on the admitted CID22 oracle TRAIN fit table (SHA
`5726053da43c9bc0bf33a26ad99c3b9fdde64de42425e685cc84527c606d5999`).
`--verify none` does not grant qualification: require canonical metadata,
ordered TRAIN-development inference, pixel/cache parity and all final gates.
Never calibrate on human EVAL, terminal labels or the development curve.
No revision stamping is needed or allowed to repair provenance.

After owner freezes one serving composition (seed/ensemble, head,threshold,
calibration and supported scope), bind exact bytes/instruments and authorize
the required EVAL exposures. Use direct `bake_verdict` with the full composition
and **all optional input paths explicit**. Example command shape:

```bash
"$BV" --bake "$FINAL" --corruption-head "$HEAD" --corruption-head-threshold 0.9 \
  --features-root "$EVAL" --corpora "$AUTHORIZED_CORPORA" \
  --dial-grid "$GRID" --negtail-probe "$TAIL" --identity-probe "$IDENTITY" \
  --corruption-grid "$REVIEWED_GRID" --perpair-metrics "$PEERS" \
  --gaddr-grid-truth "$DIAL_TRUTH" \
  --gaddr-json "$ADDR" --fulleval "$VERDICT" --output "$REPORT"
"$FC" --qualify --fulleval "$VERDICT"
```

Verify owner flags with `--help`, preflight complete discovered metadata via
`--print-input-paths`/`assessment_identity.py` before hashing, and pin executable
SHA; this is a command template, not an authorized invocation. The SHIPPATH7
`run_full_eval.sh --stage identities` route only collects composition/instrument
identities and exits before scoring. It refuses instrument/head transport in
scoring stages, so it cannot silently serve as this full composition command.
Run ladder as a separate verdict with `--dial-grid "$LADDER"` and bound
ladder truths. `--dial-peer-scores` is a `label=path` peer-only mode and refuses
`--fulleval`; `--gaddr-grid-truth` supplies candidate mentor truth. Rank/dial
verdict, finite spatial, native RD/targeting, integrity EVAL and
runtime/HDR/input packets remain separate owner measurements.

## D2 receipt, authorization and exposure protocol

The new [harness](../scripts/rev4_featpot/kadid_terminal_read.py) orchestrates
the existing Rust statistics owner via `lib.zen_stats` (full panel, signed batch
SROCC, indexed paired resamples, `--per-group`, `zenstats::scatter`) and the
existing original-label reader `v2c_labels.load_label_rows`. It consumes
precomputed, label-free predictions from the pinned **Rust production surface**;
it does not implement inference, fit or calibration. A canonical prediction
receipt must bind scoring byte identities and complete composition. Python
feature forwards cannot replace that receipt.

Create the final receipt **only after harvest and all pre-terminal gates**.
Schema `kadid-terminal-final-model-v1` has:

- `design_line: by_v2fy-rev5-d1-20261007`, `population_rows:2000`,
  `bootstrap_resamples:10000`, fixed integer `bootstrap_seed`,
  `orientation:quality`, registration SHA and `code_sha256` for
  `kadid_terminal_read.py`, `v2c_labels.py`, `zen_stats.py`, `_terminal_bound_io.py`.
- D1 `human_sources:[kadid,tid2013,konfig,cid22_a25]`,
  `fit_jobset:fitv2d1-20261007`, `selected_epoch:119`, `seed_index:0|1|2`.
  The canonical inspector independently checks actual Rev5 qualification,
  epoch, seven admissions, sampler, decoded budget/seeds and D1 input hashes.
  An ensemble needs a separately reviewed extension to inspect/bind every
  member; this first harness deliberately refuses a manifest in place of a
  production model. It never chooses the best seed on terminal scores.
- Explicit `{path:absolute,sha256}` specs: `population`, `predictions`,
  `prediction_receipt`, `scorer`, `panel`, `inspector`, `qualification`;
  `models:{production,profile_b,research_by_v2fy}`; `bindings:[spec,…]` for
  companion bytes/sidecars, bank receipts/declarations, decoded-input audit,
  source selection/order and harvest receipts. `composition.primary` equals
  `models.production`, with every routing/member/head/threshold detail frozen.
  ProfileB bytes must equal shipped `codec_target()`; research comparator is
  the owner-chosen **Rev4** research bake on its correct Rev4 arithmetic.
- `qualification` JSON names the same `composition` and production
  `model_sha256`; `gates` maps each name in the script's `GATES` to
  `{state:pass,artifact:{path,sha256}}`. Coordinator reviews these owner
  artifacts and supplemental unresolved rows before authorizing. This is an
  authorization boundary. Separately, `reports.E30` is
  `{state:completed,artifact:{path,sha256}}`: complete registered four-source
  report with no rule/adopt fields; numerical deltas never veto D1.
- `labels` is the original-label adapter spec `{path,sha256,format,
  ref_col,dist_col,label_col}` plus `rows_key`/`usecols` if needed. Dedicated
  TERMINAL-only original manifest; no `select` or `via_pairs`, no discovery.
  The SHA must be64 lowercase hex; JSON needs a nonempty `rows_key`, and
  `usecols`, if present, must include all adapter columns. Its SHA comes from
  an existing receipt or the coordinator's pre-read pin;
  **do not hash actual terminal labels to prepare this file**.

Population TSV has exactly `source_row_id,pair_key,ref_basename,distortion_type,
ref_path,dist_path,ref_pixels_sha256,dist_pixels_sha256`. Original2000 distinct
stimulus paths/IDs, physical order and decoded byte pins are mandatory; repeated
bank `pair_key` is permitted for collapsed stimuli. SHIPPATH7's1952-key feature
projection is **not** an automatic2000-row read population. A label-free mapping
from original source IDs and frozen bank receipts must recover every original
stimulus, including identity; if absent, preparation stays blocked. Predictions
TSV has exactly `source_row_id,production,profile_b,research_by_v2fy`, all finite,
same order. Receipt schema `kadid-terminal-surface-predictions-v1` contains
`labels_read:false`, `surface:"zensim::BakeScorer / Zensim::codec_target"`
and exact copies of population/predictions/scorer/models/composition/bindings.
Existing `score_pairs_tuner --profile b --ensemble production=… --ensemble
research_by_v2fy=…` serves standalone candidates through Rust and zenpng; it
does **not** attach the tree head. Complete-companion scoring must extend that
existing owner and prove decoder/pixel parity before a production prediction
receipt is authorized. `serve_custom_bake --pairs` likewise does not attach a
head and uses legacy image decoding. Do not silently use either incomplete
path for the final composition. This measured source limitation remains explicit.

Commit the final receipt JSON locally before any label read. Coordinator file
`kadid-terminal-authorization-v1` must have `authorize_once:true`, same
`design_line`, exact `receipt_sha256`, nonempty `coordinator_message`, full
40-hex `pre_read_commit`, `receipt_repo_path`, absolute canonical `ledger` and
shared `journal`. Parse **the verified committed bytes**, never reopen the
live receipt after matching its hash. No live authorization/receipt exists here.

Bound-I/O details: [round-three record](releasegate3_WORKLOG_2026-10-07.md).

Before label hashing, the harness validates authorization, committed pin, all
non-label specs/bindings, gate artifacts, exact model inspector metadata,
prediction receipt/schema/order and original2000 population. Every metadata
path must lie lexically and after resolution under source/preparation roots:
repo, `~/tmp/zensim-paper/rev4`, `/mnt/v/output/zensim`, `/var/tmp/rev4-featpot`.
Original corpus stores, banks, terminal/T0 populations and shared protected
markers refuse before handle-bound hash/parse. Stage label-free artifacts outside those
roots; receipt fields cannot authorize metadata aliases into them. Exposure appends to `docs/DATA_SPLITS.md` under
`flock`, writes/fsyncs shared exclusive journal
`~/tmp/zensim-paper/rev4/KADID_TERMINAL_SPENT.json`, and fsyncs ledger reservation
**before the first label open**. Duplicate reads refuse even if output is lost,
another workspace is used, confirmation fails, label hash fails or a statistic
fails. The journal is never deleted to retry this design line.

Only then are labels hashed/read and joined by exact original ref/dist paths;
coverage must be exactly2000 with no nonfinite/drop. Canonical panel
`--signed-quality` fixes higher-is-better direction for SROCC and Kendall tau-b.
PLCC is four-parameter logistic fitted Pearson with the fitted mapping restored
to increasing predicted-quality direction; signed raw Pearson remains separate.
Within-ref means reduce canonical `srocc_signed` results (min3); no Auto
polarity. Per-type SROCC also stays signed. Legacy panel modes are unchanged. Reference draws use NumPy default_rng seed
fixed in receipt; shared drawn reference indices for both models, all stimuli
of each selected reference retained,10000 draws, sample SD(ddof1) of paired
SROCC deltas as SE. Confirmation uses the registered inequalities inclusively.
Failure is final; no model/seed/rule change follows from this read. Result SHA
and PASS/FAIL append to ledger. ERROR retains the spent record and emits a
fixed error category only; parser contents/tracebacks cannot cross result/CLI boundaries.

Synthetic positive and negative fixtures run the real Rust panel10000 paired
resamples; authorization, commit/hash, model/gate, symlink and spent-state
tripwires prove zero sentinel opens on refusals. Synthetic models/inspector
responses are orchestration fixtures, not scientific admission. Run:

```bash
TMPDIR="$HOME/tmp/releasegate" ZEN_PANEL_BIN="$A/bin/panel" \
  ../scripts/run-heavy --mem 16G --jobs 8 -- just releasegate-tests
```

This lane adds implementation/documentation. The real exposure ledger is
unchanged; only synthetic fixture ledgers were written. The real TERMINAL spent journal is absent; no authorization
is issued, gate claimed passed for production, or shipped profile changed.
