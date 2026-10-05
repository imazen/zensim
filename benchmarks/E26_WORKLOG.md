# E26 worklog

Part A landed as-is at main ef31614a77db7788d8aea34715d8f1030e5e0ab9,
as confirmed by the coordinator. Part B is its direct child; no rebase.

Before any fit: fixed q_jod transform and explicit native research caller
registered in the original E26 record. No E26 fit has started. Full decision
rule, arms, feature IDs, seeds, folds, and VAL population remain unchanged.

The existing shared fleet queue is not host scoped: host_filler.sh invokes
kids_pick.py with QUEUE=fleet_queue; kids_pick.py reads unqualified triples.
It serves prohibited hosts as well as allowed hosts. E26 will not enter this
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
