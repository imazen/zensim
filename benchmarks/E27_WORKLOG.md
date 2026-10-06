# E27 worklog — registered HDR leg forms, 2026-10-05

Registration070b8247 is copied byte-for-byte from main before any E27 fit.
Implement on accepted-for-review E26 landing chain7d02b458; no push.
E26LAND_REVIEW.md was absent at first reads; inspect and fix it before smoke
or launch readiness. Existing scientific E26 records and cells are immutable.

Fixed arms hp4 (pooled cross-reference rank) and ha4 (within-reference MSE
plus rank), score=10*q_jod without clipping, unchanged7390 agreement TRAIN
rows,420 by_v2fy IDs,base/headN/seeds0–9/five folds/120 epochs/last119,
nominal4 using E26 acceptance weighting. No HDR development or calibration.
Reuse the E26 native banks, teacher authority and approved dense+stamp
serving bindings; fresh E27 data root, content-addressed pack and outputs.

Decision fixed by registration: SDR E21 as-good; pooled and within-reference
SROCC against each teacher >=-2SE; pooled HDR-VDP-3 gain >2SE; choose passing
arm with larger pooled HDR-VDP-3 gain. E26 hd4, external SDR and full scatter
geometry are report-only. No E27 VAL read for fitting or smoke selection.

Prepare program/image/data and one full cell smoke only. Existing launch
owners v2_loop/score_chain, allowed host-scoped fillers, no tail_trim, no
new filler/worker. Do NOT enqueue until coordinator confirms E26 landed and
matching zenmetrics pins pushed. E27_READY.md is the readiness receipt.

Compatibility gate: the merged historical wrapper now passes the explicit
replay option, unsupported by E26's actual old fleet-v2 trainer. A current
trainer therefore needs frozen E26 trajectory/weight/prediction parity
before the E27 fit; this is software equivalence, no scientific retuning.
Pack complete runtime dependencies, including the merged safe-path owner;
actual binary inventory/build provenance must agree without copied v25b prose.


E26 landing review arrived and found no defects requiring a fix; full report
E26LAND_REVIEW.md inspected before smoke. Its archived short-fit comparisons
preserve groups, weights, trajectories and predictions across the merge.

Compatibility-plan correction, before any E27 fit: actual6b28576f trainer
--help confirms --historical-replay is supported. Earlier unsupported-option
assumption was wrong. E27 retains the exact E26 deployed binaries and shared
program data. The provisional new-binary pack was never fitted or queued and
is retained separately. No new binary or full E26 replay is needed; actual
CLI support and independent short-fit parity resolve the compatibility gate.
Program metadata derives from the original fleet-v2 owner record plus actual
v28 member checks, rather than the erroneous v25b copied descriptors.


Tool-discovery incident: an overly broad rg --files lookup under /var/tmp
reached _sealed directory enumeration; child pairs/raw directories refused
access. No label payload/file contents were opened by that filename lookup.
No fit/selection data came from it. Subsequent lookup is scoped to named
existing tool directories. Record the unintended boundary attempt; do not
claim no protected-path contact. The study's protected-label restriction
and unqualified research status are unchanged.


Implementation/preparation verified: hp4 routes to pooled rank; ha4 to existing
withinref,both/MSE1; hd4/hd16 and historical/strict paths retain their owners.
All167 Python tests pass (after correcting an initially wrong test-only panel
path); existing E26 decision recomputation is exact, 20 Rust sampling tests
pass, CI-exact clippy and lint-scripts pass. Synthetic full raw/cell plotting
mechanics pass; no registered HDR VAL was read during preparation.

Data3f008d2f preserves every56 original E26 payload;100-cell manifest8baf0da9
matches first hp4/kadid/s0 smoke and declares6GiB/1thread throughout. Program
v29a53912d8 packs exact E26 binaries with merged pinned Python owners and the
required v2c_wide dependency. Build metadata records actual binaries and the
original producer. Tower image3de660ba and i265 image2548f87e differ in image-ID
representation but every filesystem layer/runtime config field is identical.

Zenmetrics profile v2r5hdr27 local jj change vktmpnnrytkrukrqxknyttnrlsosowzu
(commit344fa081b0f0a915ea2cc8a5878547e9383d4828); lane did not push. Full pins
are in the preparation pointer and E27_decisions.md. Source inventory was
frozen before the fit (44c8c2bc); all packed pins still match. E27's full hp4
smoke runs only through the existing fit-cell executor in one i265 Docker
container, no fleet queue, worker/filler or tail controller launch.

Preparation archive on Tower hash-verifies35 files/657741466bytes, manifest
262542438bd17c6d2d8500deb349c7db0dfdb6ff45b7a2a178c02c774e456a60.
Large files remain external. A partial archive attempt (cross-filesystem
hard-link refusal) was retained, then completed with copies and full hash gates.
Launch script's missing-authorization test fails closed before all external
writes and leaves fleet_queue byte-identical. Await both coordinator confirmations.


Decision fixtures now include nonzero paired seed variance; the10-test HDR
suite also rejects a positive pooled gain below2SE. Scientific rule/source
program unchanged. The report-only geometry now includes all50 canonical
pooled-vs-within cell points as well as all raw model-row observations.
Future HDR panel hashes match the existing3900-row E26 native/cache proof,
VAL bank manifest and approved cache predictor (metadata-only prep reads).

One-row native/cache smoke uses the same admitted E26 TRAIN pair and420-ID
features. Original corpus paths are not present on Tower; exactly one PNG
and one JXL were copied with SHA receipts to a new Docker-only probe mount,
not a new corpus or label population. The11.3MB staged pixels preserve original
bytes. No HDR VAL prediction/label read or model assessment during prep.


Full hp4/kadid/s0 smoke PASS on i265:120epochs x50000pairs, final119 despite
SDR development best epoch0;2532.6seconds at final epoch;7869 predictions;
pooled rank, effective4.287408376091277, HDR dev0, SIMDv3. Actual HDR intake
rows7390/roleTRAIN/populationagree-only/fixed10*q_jod. Harvest owner verifies
program/data/argv/bake/blob receipts. Source bakedcf1a8fc; dense000daffe;
stamped4b15f7e0, immutable source preserved. Existing dense512-row gate and
approved6c3a9400 stamp PASS. Old table admission remains formula_revision null /
qualified_provenance false, as inherited E26; no qualification upgrade.

Native smoke first refused because invocation omitted ZENSIM_FORMULA_REV=5 and
thus used Rev1. Original keyed refusal/log retained under native-smoke. Correct
Rev5 invocation under native-smoke-r2 PASS:cached=native74.63024139404297,
absolute delta0, exactly the same admitted TRAIN pixel hashes and stamped bake.
No program/data/model correction or refit. Smoke archive30files/15122297bytes,
manifest6a98966ce124b50002b7422863c06d5ac507b121b889c3af6c8cc7f13f0c5a36.

Readiness requires both coordinator confirmations and all smoke/binding/native
receipts before image publication/upload/queue mutation. No authorization exists;
E27 is absent from the byte-identical queue. No owner or tail controller started.
Report-only external SDR and full registered HDR commands are prepared for after
harvest; neither assessment ran during preparation. No source push.


## E27 fleet launch — coordinator, 2026-10-05 19:21 UTC

Coordinator launched fitv2e27-20261005 at19:21:36Z, top of the existing queue.
E26 landed as caee5bed; zenmetrics profile344fa081 is pushed/verified on origin.
E27_REVIEW.md found no code defects (168Python/20sampling/25trainer tests and
short historical/hd4/hp4/ha4 fits preserve the registered mechanics). Original
readiness records remain immutable snapshots of pre-launch state.

Two launch deviations are explicit. Tower lacked ghcr credentials; coordinator
saved the image, verified identical RootFS/Env/Entrypoint/Cmd, and published from
dev using the i265 image representation2548f87e. This is artifact publication by
the coordinator, not authorization for fleet training on dev. Program/data and
fit identity unchanged. IMAGE_PUBLISH_RECEIPT.json is the authoritative receipt.

My launcher omitted jobset_caps.json even though E26 had required this actual
runtime envelope. Manifest memory hints alone were insufficient. Coordinator
added6g and host caps tower5/i2653/i2703/r3500 2/r3800x2 to the existing owner.
Verified launch_v2 consumes that entry; no filler/worker restart by the lane.
Existing v2_loop/score_chain controllers1713829/1713830 run; no tail_trim.
Status, authorization, publisher and cap receipts are recorded in LAUNCH_RECEIPT.
Authorization prose says nominal19:25; actual status timestamp19:21:36 and the
coordinator's19:21 message govern launch timing. No changed scientific rule.

All100 verified/installed cells and posted SDR decision are prerequisites for
registered HDR VAL and report-only external SDR panels. No early HDR panel or
E27 verdict. Landing rebase onto main@origin follows final scientific records;
no source push. Keep original cells and E26/preparation evidence immutable.

## Post-harvest invocation preflight — 2026-10-05

The prepared HDR invoker pointed its exact pinned cache predictor at the Tower
NFS mount, which is noexec. Copied the same bytes to /var/tmp/e27/bin and verified
e4a411209ff5 unchanged; no rebuild, fit, bank, model or rule change. Explicitly
pin process revision5 to match every serving stamp and the canonical panel
c9c610b8 from bin-v2. The existing one-TRAIN-row wire and smoke bound bake give
74.63024139404297, bit-identical to the original native/cache smoke (delta0).
No VAL payload read for this preflight. Preserved original command and recorded
both hashes in PANEL_INVOCATION_FIX.json; the corrected invocation is used only
after all100 installed cells and the posted SDR decision.

Launch-records archive:21 files/48994bytes, verified manifest
893507d4b82cf736bfb3f318462ff6841252bcd380007f5b49afa35aa76cf320
at output/zensim/e27-2026-10-05/launch-records. Runtime snapshot confirms actual
5/3/2/2 workers on Tower/i265/r3500/r3800x, 6GiB, one fit thread. The snapshot's
manually typed19:36Z timestamp was ahead of actual time; the local corrected
receipt uses its original file mtime, with before/after hashes in
LAUNCH_ARCHIVE_TIMESTAMP_CORRECTION.json. Immutable launch archive retained.
Await owner watches100 fleet receipts plus posted E27 SDR result. No new
fleet controller, filler, worker, tail trim or source push.

The final HDR invoker also explicitly caps ZENSIM_MAX_TIER=v3, matching the
frozen fit profile and native TRAIN smoke. Repeat of the same single TRAIN wire
is still74.63024139404297, delta0; no VAL payload read. Original invocation and
first path-fix receipt retained; final hash in PANEL_INVOCATION_FIX and pointer.
Launch-record lint:820 scripts checked, all runnable.

## E27 registered assessment completed — 2026-10-06

MISSING: human HDR judgment/current qualification, complete original encoder
provenance, strict trainer table admission for the historical program, and
shipping/runtime/product gates. This study establishes no new shipping default.

All100 arm cells were receipt/hash verified and installed; last119 throughout.
The existing score owner posted its complete-set SDR result; the canonical
waiter completed at04:29:17Z. No new filler/worker or tail_trim was launched.
Registered HDR VAL covers all3900 rows/300 references for each of50 controls,
50 hp4,50 ha4 and50 report-only E26 hd4 cells. Fresh dense512-row prediction
gates and the approved Rev5 stamps bind all200 serving bakes; source hashes
remain unchanged. Native/cache proof, bank, predictor and statistical-owner
hashes match the pre-registered pins. No dropped/nonfinite panel rows.

Verdict: retain the control; neither E27 arm passes. hp4 is SDR as-good
(meanDelta+0.000609925836325429, SE0.0011019651341445987) and pooled HDR-VDP-3
improves+0.12304812916115589 (SE0.006596765718227003), but pooled CVVDP
declines-0.0889260400361078 (SE0.002981641513570055), failing its >=-2SE guard.
ha4 fails SDR (meanDelta-0.03880942362030192, SE0.0014751554799661538) and
within-reference teacher noninferiority plus pooled CVVDP. The full original
rule is applied without threshold changes; no passing-arm tie occurs.

Both report-only panels are complete: E26 hd4 and external NITS/LIVE/MC-IQA
(all50 paired cells and every existing category); raw density retains195000
model-row observations per composition and all50 cell points. Repeated model
rows are not independent examples. Full means, paired deltas/SE, all SDR sources
and all external categories are in
[e27_result_summary_2026-10-05.json](e27_result_summary_2026-10-05.json).
Original HDR VAL/native proof reuse is explicit; no new all-model pixel replay
or human qualification is claimed. Legacy admission formula_revision null /
qualified_provenance false persists; only serving binding comes from the stamp.

Tower final-assessment archive hash-verifies3629 files/662966352bytes, manifest
38c04362dd80cb9d20e1bcde882d7250ae1eca02651c36adbc748464f7ade26a.
The [final pointer](e27_final_2026-10-05.pointer.json) retains all100 immutable
arm cells,50 controls,50 E26 report cells,200 full panel predictions/bindings,
external inputs/results, ledger, proof and runtime receipts. Preparation,
readiness and launch snapshots remain unchanged. All large new artifacts stay
on Tower. Landing rebase/validation follows; no source push.

## E27 landing validation — 2026-10-06

Rebased only the E27 chain onto main@origin ec4768cf; original fit tip6413f8f6
remains pinned separately. All five program Python source pins and all core
source files are unchanged from the reviewed fit source. Historical, strict,
hd4/hd16, hp4/ha4 and forbidden-VAL actual-loader tripwires retain their tests.
The complete168-test Python suite passes, as do20sampling/25trainer/29pack/
41verdict/15research/36serving/2native/API-doc checks; fmt, lint and both clippy
commands pass. One pre-existing verdict and three pre-existing serving tests
remain ignored, unchanged; no new ignore or relaxed threshold. Logs and
LANDING_VALIDATION.json are preserved with hashes in the landing archive.
No program rebuild/refit, scientific-rule change, source push or shipping
qualification. The final rebased tip is reported in E27_DONE.md for review.

The [full SDR diagnostic record](e27_sdr_report_2026-10-05.json) preserves all
source/seed control values and auxiliary W1/W2/W3/W4 panels. Its inherited E13
improvement-rule metadata/adopt flags are report-only; E27 uses E21 as-good
from e27_sdr_decision.json and the registered four HDR guards. No altered rule.
