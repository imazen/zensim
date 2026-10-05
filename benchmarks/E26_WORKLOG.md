# E26 worklog

Part A landed as-is at main ef31614a77db7788d8aea34715d8f1030e5e0ab9,
as confirmed by the coordinator. Part B is its direct child; no rebase.

Before any fit: fixed q_jod transform and explicit native research caller
registered in the original E26 record. At that checkpoint no fit had started. Full decision
rule, arms, feature IDs, seeds, folds, and VAL population remain unchanged.

Initial placement concern (resolved below): the shared queue is not host scoped: host_filler.sh invokes
kids_pick.py with QUEUE=fleet_queue; kids_pick.py reads unqualified triples.
It could serve prohibited hosts if such fillers ran; the coordinator later confirmed that none run. E26 will not enter this
queue until its allowed-host restriction is enforceable. Existing workers
are owned by other lanes and are not stopped or restarted.

Coordinator resolved placement: only the five allowed consumers run, all
single-thread cells; existing tower capacity explicitly accepted. No extra
filler or worker will be started/restarted. Transform and rank-only leg
committed before fitting at 8e8f3170.

Native parity: exact 420-ID Rev5 research extraction matches canonical
720 HDR features bitwise on 16x16 (padding) and 65x97 pairs; SDR extraction
refuses the HDR sources. Existing native decoder test passes. Hidden API
snapshots regenerated and api-doc-check passes; supported API unchanged.

Production blocker discovered by one exact E24-control/TRAIN-pair probe:
selected E24 bake refuses native Rev5 scoring. Its immutable trainer log
records null revision and unknown feature identities despite Rev5 table
sidecars. Fitting/scoring stopped before the first E26 fit; coordinator
request and exact proof paths recorded in E26_decisions.md. Native feature
extraction and admission checks continue; no cross-revision bypass.

Actual loader/packer tests: synthetic receipts exercise negative target
retention, row reversal, VAL role, disagreement, duplicate row ID, wrong
revision/read set/transform/source pin, changed table/manifest, dev leg and
confirmation/all packing refusals. Three tests pass; never fit fixtures.

2026-10-05 UTC — coordinator authorized receipt-bound Rev5 serving binding
for all 50 immutable E24 controls and all 100 E26 arm cells: existing
dense_bake (512-row BIT-IDENTICAL gate), then pinned bake_stamp_revision.
Per-bake source/dense/stamped hashes and gate receipts go in a new directory.
The trainer admission's null revision / qualified_provenance false remains
an explicit historical gap; the stamp owns serving revision. One control
probe gives native/cache score 74.35427856445312, bit-identical.

Native extraction complete: TRAIN7390 (627s), VAL3900 (326s), exact420
requested IDs, original hashes/order, absent NaNs. No E26 grid launched.
First Docker executor smoke stopped at table admission, before fitting:
null feature_set_id is a malformed trainer declaration. Native research
intentionally has no family-token shorthand for channel-subset plans; retain
that null under source_bank_feature_set_id, explicit IDs and per-slot
provenance rather than inventing an identity. Existing SDR trainer gap and
registered controls/rule are unchanged. Failed archive/logs remain preserved.

Independent review E26_REVIEW.md: fix P2 by checking fit-only record shape,
pinned sidecar, TRAIN/E26/Rev5/agree-only7390/read-set contract BEFORE any fit
or keys payload hashing/open. Later membership, row-order, targets and hashes
still gate. Actual loader/packer tests4/4 pass with io.open tripwires; the
reviewed pre-fix actual loader fails the new forbidden_val tripwire.
Program/profile/image must be rebuilt; coordinator push confirmation gates
fleet enqueue. No filler or worker starts/restarts.

Before first successful fit, freeze report-only steering reference: first
admitted TRAIN ref (source1066,1200x1600), its full registered quality ladder,
control and each arm's without_kadid_s0 cell. This is descriptive only and
cannot change selection or the registered SDR/HDR rule.

Report-only corruption panel frozen before first successful fit: unchanged
HDRCORR9036-row canonical TRAIN packet, all12 TRAIN origins and honest
q10/q20/native cohorts; matched without_kadid seeds0,1,2 for control,hd4,
hd16. Existing production serve_custom_bake and corruption_gate_eval owners
score/report all9036 rows, preserve deduplication and negative scores, with
no threshold fitting or integrity activation claim. This descriptive subset
is not the50-cell HDR/SDR adoption panel.

Native control steering:15/15 fixed TRAIN rows pass exact420-ID features
and cached/native/prepared score parity; max feature and score error0.
Identity100/refinement0 pass, unsupported refinement IDs empty. Existing
HDR audit mode now uses the research planned walk atRev5 rather than the
unsupported legacy append2/full-pool walk; previous revisions keep that
legacy path. Native extraction pin f0a81037 remains untouched; separate
audit executable is pinned in evidence.

Program v28=7e8ac08b, data r2=2eb85985, image digest b99cf76e, profile
08ae0dfa pushed by coordinator. Full120-epoch Docker smoke runs before
enqueue. Existing launcher has E26-only6g/AVX2 and RAM-safe concurrency
envelope; other jobsets and all filler PIDs unchanged. Own worker starts
and restarts remain forbidden. New actual loader tests4/4 and pre-fix
negative control, native tests2/2, root and native strict Clippy and
lint-scripts813 all pass. Final scientific panels still pending.

2026-10-05 04:17UTC: full120-epoch smoke PASS,42m38s, selected119 and
7869 heldout predictions. Existing harvest_fit_cells.verify_blob verifies
all hashes/identities/tier. Source4c3b106b; identical dense/stamp owner then
new-student cached/native TRAIN score74.24549102783203, bit-exact.
Full100-cell jobsetfitv2e26r2-20261005 enqueued top of sharedqueue via the
existing owners. No filler/worker directly launched or restarted. Actual
initial reachable slots5/3/2/2;i270 unavailable. Resource envelope is
E26-only; existing fillers' status text still reports requested free slots,
so actual Docker census, not that message, gives the started count.

Whole3900-row native VAL/control cached-production parity PASS: maximum
delta0, all rows finite, exact original order, pixel-identical population0.
Existing native extractor now emits an ordered pixel-identity census in its
manifest. The frozen bank/feature-extraction executable stays immutable.
HDR panel requires this hash-bound whole-population proof and the same
cache predictor binary; identity shortcuts cannot silently contaminate the
fast path. Core trainer/program/data/thresholds unchanged. Full native proof
took677s at2 threads,0.82GiB peak; no human interpretation of these labels.

Native final census-source checks: two native tests pass, strict native Clippy passes. Durable tower native-bank copy verified98 files/260964434bytes including extraction work manifests. Full100 fits still running; final scientific panels remain pending. Existing fillers only, no worker restarts.

First5 cells canonical harvest PASS. Actual fleet without_kadid_s0 matches full smoke result science and selected weights exactly (runtime/path metadata excluded by canonical compare_science). First5 arm cells have dense/stamp receipts, originals intact. Frozen hd4 steering15/15 passes exact canonical/cached/prepared features and scores, same final audit executable as control. Corruption hd4 three seeds score all9036 frozen TRAIN rows; canonical report retains8213 unique/7725 positives/480 honest and qualified=false. Remaining fits and final50-cell decisions pending.

All reachable hosts measured at06:25UTC: tower5 slots2.49cells/h, i2653slots4.33, r35002slots1.10, r3800x2slots3.88;24 complete/verified/bound. ETA~6.44h remaining plus final panels, provisional from hd4 observed service times. Ledger timestamps are pass-level: service times use consecutive ledger uploads per worker (first from pass start). Live FitCell claim renewal confirmed; no custom lease controller. Full100 unchanged, no failures observed.

Coordinator landing instruction: main advanced to0eb01905 (SHIPPATH
strict-admission chain). Leave running v28 and its pinned experiment owners
unchanged through fit/panel completion. Afterwards rebase E26 records onto
main@origin, merge strict/historical/HDR behavior and retain their tests,
including the seven overlapping files named by the coordinator. Coordinator
reviews the merge; this lane does not push. At07:03UTC27 cells have completed,
with no failures observed. E26_DONE remains unwritten until all work is done.

At08:53UTC50/100 ledger completions (48hd4,2hd16); first hd16
canonical harvest PASS, v3/last119. Immutable experimental owner sources
archived and hash-verified on tower:272 files/10308748 bytes under
output/zensim/e26-2026-10-05/frozen-owner-source, including the original
7ad72c87 native extractor source separately from the final proof owner.
This preserves producer/scoring evidence across the later SHIPPATH merge;
binary build provenance remains in build_meta_e26_v28.json. No rebase yet.

At09:23UTC54/100 canonical-verified and dense/stamp-bound cells, all
v3/last119, no failures. Measured capacities: tower2.48cells/h,
i2654.33, r35001.10, r3800x3.98. First hd16 service times i265
2440–2456s, r3800x1776–1826s. Provisional remaining~3.87h (fluid fit
ETA13:16UTC; slow-host tail and final panels add time), full100 unchanged.
Per-cell intervals retained in FLEET_TIMING_0922.json.

At10:31UTC70/100 ledger completions;68 canonical-verified/stamped.
All three frozen steering compositions now pass15/15 TRAIN rows with
zero canonical/cached/prepared feature and score errors, no unsupported
refinement/density IDs. Hd16 corruption seeds0–2 scored all9036 frozen
TRAIN rows through production BakeScorer; canonical report COMPLETE,
8213 unique/7725 positive/480 honest, qualified=false. Registered final
SDR/HDR panels still wait for all100; no population or rule changes.

Coordinator incident report (resumed2026-10-05): tail_trim PID3635590
was stopped at approximately16:19UTC after the last two cells
(hd16 aic3 seeds8/9) repeatedly lost their holders about every2 minutes
for roughly4 hours. Once tail_trim stopped, their holders survived and
both fits completed. This supersedes the earlier smooth-service ETA;
retain the incident and completed outputs. Root cause is not independently
diagnosed here. All100 cells are now installed; the existing SDR owner
reports both arms as_good. No filler/worker restart by this lane.

Packed executable provenance erratum (final audit): earlier "exact v25b
Rust binaries" prose was inaccurate. Immutable v28 embedded files inventory
and tar members agree on trainer6b28576f/predictor81ec2207/panelc9c610b8:
the existing fleet-v2 set, covered as old_set by the canonical13-check
ALL_IDENTICAL record benchmarks/rev4_featpot_canon_binary_parity_2026-10-01.json.
Copied binary_mix descriptors instead name v25b605d20e0/56da0529/f76b85a7.
Do not rewrite frozen pack or its metadata; supplemental receipt
PROGRAM_BINARY_PROVENANCE_FINAL.json records actual hashes, source records
and bounded parity evidence. No direct100-cell HDR equivalence claim.
Registered populations, recipe, weights, seed pairing and adoption rule
remain unchanged; human/product qualification remains missing.

Final registered E26 result: hd4 is the lowest passing weight; both arms pass
SDR as-good and the two within-reference HDR conditions over all50 matched
cells. Hd4 HDR-VDP-3 delta+0.0014655677655677746, SE0.0003411001471767929;
CVVDP delta+0.0015161172161172077, SE0.0003392474364166754. Hd16 deltas
+0.0014747252747252793 / +0.0015007326007325662, SEs0.0003378213879858726 /
0.00033559844516359565. Pooled HDR-VDP-3 falls(-0.044864262128246485 hd4,
-0.12476287879994075 hd16); it remains reported-only under the preregistration.
Human HDR, encoder RD/spatial and full product qualification remain MISSING.
All150 full VAL panels retain3900 rows/300 references and per-bake bindings;
whole native/cache control parity is bit-identical with0 pixel-identical rows.
Full records: output/zensim/e26-2026-10-05 (tower); large JSONs are external.
V28's actual packed Rust executables are the existing fleet-v2 set
6b28576f/81ec2207/c9c610b8, not the v25b hashes in copied binary_mix prose.
The embedded inventory is correct; supplemental provenance erratum records
actual hashes and bounded canonical parity evidence without rewriting the pack.

Final fits:100 completed; last ledger upload17:05:36UTC,12.81h since
04:17 launch. Wall capacities including draining/churn: tower1.722675367
cells/h(22),i2654.327934547(36),r35001.107453779(10),r3800x3.968582059(32).
Coordinator tail_trim intervention supersedes earlier provisional ETA.


Landing merge and checks complete (2026-10-05): rebased E26 onto fetched
main@origin6874a981 (SHIPPATH9), preserving strict/historical admission,
immutable-output/checkpoint protections, teacher/ordinal propagation and
E26 native HDR leg. v2c_wide keeps both full-recipe/assessment actions and
hdr-leg; rev5_bank keeps both label-free assessment manifest and HDR mode.
P2 manifest/record admission still precedes any HDR candidate payload open.
Frozen v28 program/data/image and all completed numerical evidence unchanged.
Original source history retained at quarantine/codex/hdrcorr-e26-frozen;
merged owner hashes and logs are in the landing validation pointer.

Final checks PASS:162 Python tests (including four E26 actual-loader/packer
read tripwires and historical/strict/checkpoint/corruption-admission paths),
25 trainer/29 pack-refit/41 verdict Rust tests,15 research tests,36 serving
revision tests,2 native HDR extractor tests; CI-exact root Clippy, native
Clippy -D warnings, formatting, API snapshots and820 script lint. One
verdict and three serving corpus tests remain explicitly ignored/opt-in.
Two stale assertions inherited from main were repaired: capacity fit tuples
now carry declared epochs; historical curated receipt checks ordered row
selection and changed targets as well as row counts. No runtime behavior
change. Initial Python missing-binary failures and API-count mismatch are
retained; rebuilt merged trainer/panel and regenerated snapshots pass.
No push; coordinator reviews the merge. Full results remain research-only.
