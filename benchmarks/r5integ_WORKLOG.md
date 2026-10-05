# R5INTEG worklog — 2026-10-04

Brief: `/home/lilith/tmp/zensim-paper/rev4/R5INTEG_brief.md`. Dedicated jj workspace
`/home/lilith/work/zen/zensim--r5integ`, parent `9dfc50b8`, bookmark
`quarantine/codex/r5integ`; local only, no push. The explicit brief authorizes
this workspace despite the default no-new-worktrees rule. Scratch and TMPDIR:
`/var/tmp/r5integ`. No sealed data, fleet operation, canonical validation-pair processing or label read.
The existing content-admission guard rehashes its protected reference files as
provenance checks; no protected reference decoding, feature extraction,
fingerprinting or human-label use occurs. `ADMISSION_REHASH_SCOPE.json` lists
these hash-only reads explicitly.

Read local AGENTS/CLAUDE, DATA_SPLITS, WAVE_PLAYBOOK, session notes, canonical
corruption/refit/serving/activation registrations, Rev5 frozen-arithmetic spec,
and the relevant provenance and dataset-history records. Reuse the existing
canonical refit, ZCTH exporter, research extractor, public BakeScorer,
corruption_gate_eval and corrhead_parity owners.

## Registered scope before fitting

Use exactly the canonical packet's 12 TRAIN origins: fit
2010/1054/6068/6610/7066/9380/8206/8384; calibrate
1214/6064/9066/8462. This is the explicitly requested historical canonical
inner split, not the later product packet's split. All eight canonical validation
origins remain excluded. The retained c95bd5 Rev1 control is the historical September6 theory fit,
not a fit of this September8 packet; no same-training reproduction is claimed.
Historical binary catalog labels remain historical;
no output-dependent relabeling and no catastrophic-disposition qualification.

Full extraction: 9,036 attempts (8,580 catalog + 456 native JXL/AVIF), production
research owner, explicit requested IDs and era. Rev4 requests basic/peaks/masked/
IW/v2, head f0..371; Rev5 basic/peaks/v2, head f0..227. Wide CSV columns outside
the explicit populated-ID set are structural zero, never an admitted family.
The full-width feature-set string describes the layout; its producer manifest's
populated IDs describe actual extraction. Rev5 arithmetic is frozen at 60174678;
no reachable feature arithmetic changes in this lane.

The first Rev4 research/public audit refused exactly 526 pixel-identical attempts:
the legacy production identity shortcut returns zero features, research computes
the walk. Preserve `extract-rev4.log`. Subsequent full raw extraction includes all
attempts for provenance; fit and nonidentity audits exclude verified identity
pairs before weighting, as recommended by CLAUDE's September25 identity note.
Identities are separately guarded by public score-100 tests, not counted as head
false alarms. The projection records raw/unique exclusion counts. No changed-pixel
corruption or honest low-quality row is removed.

Recipe: existing fit-only source-weighted StandardScaler; clipping +/-8; balanced
HGB, 100 iterations, <=31 leaves, early stopping off; equal per-origin weighting,
honest multiplier 1; calibration-only weighted isotonic; fixed P>0.9. Seed4101
predeclared as in the registered f32 repair. No threshold/seed/cost selection.
A fixed Rev4 D228 diagnostic (same inputs/objective) measures removing f228..371
without conflating that delta with Rev5 arithmetic; it cannot select the head.

## Owner changes and initial controls

Extend canonical manifest mode with an explicit TRAIN-only revision-refit schema,
matching research manifest/binary/revision/ID contract, f32 payload hashes, fixed
canonical roles and the identity policy. Legacy manifest behavior is retained.
Extend the existing writer from revisions1–3 to1–5, preserving legacy bytes.
The existing audit accepts research-w1825 companions and records revision/count/
f32 hashes on RGB8 audits too; matching widths and all consumed-feature checks
remain strict.

A new Rev4 composition regression exposed the additional blanket pixel refusal
in `BakeScorer::check_pixel_revision`: compatible companions attached but compute
refused. Remove that obsolete refusal because plan() already unions and checks
companion reads. Retain sampled-Rev4, revision-mismatch and unsupported-family
refusals. The regression verifies an extra masked f300 against research extraction;
Rev4/Rev5 tests exercise active/inactive public composition and prepared maps.

Initial checks: 29 corruption-head tests pass. Synthetic exporter: seven Rust
parity cases (two legacy + five v3 revisions), unchanged legacy v1/v2 bytes,
three invalid contracts refused. Admission boundary tests refuse EVAL and illegal
regimes before feature payloads. Initial clippy passed before final changes;
final checks/results are appended after measurement.

## Replay

All heavy commands use `/home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 --`,
explicit TMPDIR, and pair threads4. No timing or speed claim from contended runs.
Build root tools with `CARGO_TARGET_DIR=/var/tmp/r5integ/target-root`; standalone
extractor with `/var/tmp/r5integ/target-bench`, `training,zen-decode`. The standalone
workspace has no tracked lockfile: initial --locked refused; offline resolution
created its lock, which is pinned in provenance. Root lock remains unchanged.

Scratch replay owners: `prepare_inputs.py`, `extract_all.py`, `prepare_refit.py`,
`refit_chain.py`. Exact extraction/refit commands and hashed manifests are stored
beside outputs. Final artifacts, summary, checks, source/binary hashes and local
commit will be recorded below before DONE is written.

## Measured results and delivery

9036 raw attempts become 8510 nonidentity attempts, then 8205 unique pixel pairs:
5499 fit, 2706 calibration. Positives: 7725 historical corruption pairs; honest:
456 native codec pairs and 24 q10/q20 anchors. No canonical validation pair is
decoded or scored. Existing admission receipts rehash 445 bound files (347 PNG,
60 EXR, 29 BMP, seven JSON, one Markdown, one executable), including protected
reference bytes; the scope file discloses these provenance-only reads.
All 456 newly decoded native pixel hashes match the retained HDRCORR deliveries.
The before/after pixel-refusal repair CSVs match byte-for-byte for both revisions.

Calibration denominators are 2546 positives, 160 honest pairs, 155 real_bug pairs.
The same fixed Rev4 all-372 head accompanies all six requested bakes:

| Composition | Detection | Honest activation/lowering | real_bug detection | Below q20 |
|---|---:|---:|---:|---:|
| Rev4 byv2fy-full-s0 | 2517/2546 | 1/160 | 143/155 | 2528/2546 |
| Rev4 byv2fy-full-s1 | 2517/2546 | 1/160 | 143/155 | 2530/2546 |
| Rev4 byv2fy-full-s2 | 2517/2546 | 1/160 | 143/155 | 2530/2546 |
| Rev4 v2basic-full-s0 | 2517/2546 | 1/160 | 143/155 | 2529/2546 |
| Rev4 v2basic-full-s1 | 2517/2546 | 1/160 | 143/155 | 2528/2546 |
| Rev4 v2basic-full-s2 | 2517/2546 | 1/160 | 143/155 | 2531/2546 |
| Rev4 D228 diagnostic, byv2fy s0 | 2526/2546 | 3/160 | 149/155 | 2534/2546 |
| Rev5 D228, stamped byv2fy s0 SMOKE ONLY | 2524/2546 | 3/160 | 149/155 | 2530/2546 |
| Rev1 c95bd5 + shipped D | 2389/2546 | 16/160 | 106/155 | 2523/2546 |
| Rev1 c95bd5 + shipped B | 2389/2546 | 16/160 | 106/155 | 2516/2546 |

Rev4 all-372: 98.860958% detection, 0.625% honest activation. Rev5 smoke:
99.135899% detection, 1.875% honest activation. All newly fitted heads detect
5179/5179 fit positives with 0/320 fit honest activations. Pooled TRAIN:
Rev4 all-372 7696/7725 positives and 1/480 honest activations; Rev4 D228
7705/7725 and 3/480; Rev5 D228 7703/7725 and 3/480. These are TRAIN screens,
not held-out evaluation or production qualification.

The controlled slot delta removes 144 masked/IW f228..371 reads, retaining
f0..227 (basic and v2). On the same Rev4 byv2fy s0, removal adds 9/2546
detected positives, 6/155 real_bug detections, 2/160 honest activations and
6/2546 below-q20 positives. Rev5 arithmetic versus the Rev4 D228 diagnostic
changes detection by -2/2546 and below-q20 by -4/2546, with equal honest
activation and real_bug counts. No new selection rule or model promotion.

All new fits fail the registered zero-native-activation and zero-native-lowering
bars. D228 additionally fails both <=1% pooled-honest calibration bars. Recall,
real_bug and below-q20 bars pass for new heads. Rev1 controls fail honest, recall
and real_bug bars; B also fails below-q20. No guards are relaxed. All heads have
zero activation on q10/q20 anchors and on the lowest observed native bands
(JXL distance25 and AVIF knob255); codec knob scales are not equated. Full
per-codec/per-knob denominators and counts are in the summary/report artifacts.
Rev4 false activation is origin6064 JXL distance0.01; D228 and Rev5 also
activate origin6064 AVIF knobs160/176. Honest rows retain their honest labels.

The matrix glob additionally matched three v2basicU bakes. Their measurements
are retained as supplemental only: same 2517/2546 detection and 1/160 honest
activation; below-q20 counts2529/2528/2532. They did not participate in fitting
or selection. Across all 13 compositions, maximum consumed-feature, pixel/cache
score, stored-f32 score and stored-f32 head-probability deltas are all zero.
Each of three exporter parity checks covers 8205 unique pairs: raw and
probability 0 ULP, zero fire disagreements. No timing/performance claim.

Actual artifact probes: 84 matched public attachments (six Rev4 bakes and one
Rev5 smoke bake, each on 12 TRAIN origin identities), finite scores and identity
score100. Both cross-revision artifact probes exit1 with the revision refusal at
zensim/src/metric/bake.rs:772. Rev5 prepared probes: one inactive pair accepts
56 map queries with unchanged base maps; three active pairs return typed
REJECTED_INTEGRITY and zero map queries. Rev4 all-372 prepared probes refuse
all four pairs at zensim/src/metric/bake.rs:1448: masked/IW has no spatial
refinement. Sampled Rev4 remains refused at bake.rs:921. No universal prepared
support is claimed.

Full R5CONFIRM bakes at /home/lilith/tmp/rev5bakes remain absent. The Rev5 smoke
bake is a copy stamped by the existing bake_stamp_revision owner; its weights
are Rev4 byv2fy s0, not a new Rev5 fit. Reviewed catastrophic/recoverable labels
are missing. No default, release, head or profile promoted.

Final checks: 29 corruption-head tests, three Rev5 full-vector parity tests,
10 admission tests, seven synthetic format/parity cases (512 rows each; legacy
bytes unchanged; invalid0/6/native5 refused), just clippy, standalone extractor
Clippy, and just lint-scripts (811 scripts) pass. Targeted cargo fmt for zensim
and rustfmt for the extractor pass. The broad standalone fmt check reports
pre-existing differences in untouched bench examples; none were rewritten.
Standalone Clippy initially exposed four pre-existing extractor diagnostics;
mechanical fixes preserve NaN handling, checked division and private manifest
arity. The exact measured extractor binary is retained separately; these later
changes affect only unrequested DVIFM histogram code and a lint annotation.
Trainer hashes embedded in fitted heads pin fit-time source; subsequent early
admission/report guards and factoring identical calibration bars do not change
the numerical recipe. Existing saved SCREEN results equal the factored owner.

Replay extends the scripts above with attach_matrix.py, refresh_reports.py,
probe_heads.py and make_summary.py. Revision env is process-pinned; commands,
input/order/feature-set hashes, audit JSONL, raw CSV, projected Parquet, NPZ,
owner reports, binaries, locks and failure/success logs are retained in the
R5INTEG_assets evidence archive with a SHA256 index and duplicate mapping.
The committed ARTIFACTS pointer contains all three head byte lengths and hashes;
heads exceed30KB and are external. The committed summary is identical to the
external measured summary. DONE is written only after artifact verification
and the local commit, with the commit ID recorded there.

## R5INTEG2 follow-up — registered 2026-10-04 before fitting

Coordinator follow-up authorizes a new TRAIN-development regime at both
revisions: f0..227 plus f372..719, exactly 576 reads (228 basic/peak + 348 v2),
caller width1825. No masked/IW reads. The recipe, historical TRAIN fit/calibration
split, seed4101, threshold and all gates remain fixed. Reuse the pinned matching
Rev4/Rev5 research payloads; no source image re-extraction or relabeling.
`benchmarks/r5integ2_REGISTRATION.json` records the contract and composition/
prepared probes before any fitting. Independent 6064 SSIMULACRA2/Butteraugli
checks are diagnostic evidence, never training labels or threshold selection.
Scratch remains `/var/tmp/r5integ/r5integ2`; no sealed data or fleet change.

Correct the previous 64,362-byte committed summary: copy its exact bytes to
`/mnt/tower/output/zensim/r5integ-2026-10-04/r5integ_2026-10-04.summary.json`,
verify SHA256 `310f18bc1a541cd8c6df7752f4daba3db9b0432a6da6fd36d0da5474e10799c4`,
then replace the same repo filename with a 406-byte JSON pointer. The original
external summary/evidence and this worklog remain intact. All new evidence over
30KB is external; commit pointers and bounded factual notes only.

### Review corrections and bounded layout — before the second export

P1: prepared steering now forwards its complete serving plan at Rev1–5 before
removing the companion for finite-difference probes. The old conditional
excluded Rev1–4; Rev5 takes the same branch as before. Added explicit revision
regressions using base f13, companion-only peak f159, caller widths372/1825,
active/inactive controls, repeated admission and detached map parity; linear
companion admission is also covered. P2: audit_report admits the refit manifest
and every input origin/role before payload hashing, Parquet or audit reads.
The TRAIN-only schema requires the exact canonical fit/calibration lists and
empty evaluate. Direct tests include the reviewer's synthetic3311 anchors and
positive, early-read tripwires, and successful historical-schema evaluation.
13 Python admission tests and27 corruption-head library tests pass.

The first registered576 fits used width1825 and pass all seven TRAIN gates.
Actual private compute-plan inspection exposed unrelated Rev4 optional kernel
activation caused by that wide layout. Register width720 separately in
r5integ2_LAYOUT_REGISTRATION.json before a bounded-layout rerun; no feature,
recipe, label or threshold changes. Complete basic reads can still add
full-resolution X/B moments omitted by a base plan; zero extra extraction is
not established. The rerun must have bit-identical NPZ feature/raw/probability
arrays to the first fits. Pin the serving extractor separately from the
immutable fit-feature producer; this allows the corrected prepared runtime
without rewriting extraction provenance. All first-fit evidence is retained.

### R5INTEG2 completed measurements — 2026-10-04

The review fixes are committed in79ae0eb3; the explicit bounded-prefix research
audit fix is ded4a5c5. Frozen7a9dbacb replays confirm both linear/tree silent
acceptance at Rev1–4 (scalar f159 approximately0.148902, headP=1; prepared
f159=0); Rev5 returns CorruptionDetected. All five revised per-revision
regressions cover caller372/1825, active/inactive controls, repeated errors,
companion-only f159 and linear heads. Companion detachment from finite-difference
probes remains. Rev5's four original prepared audit records are identical
before/after the fix. P2's direct audit_report tests reject synthetic3311
validation anchors/positive and bad role/origin/schema before any payload,
Parquet or audit read, while historical evaluation still succeeds.

Final basic-v2-576 heads have caller720: Rev4=209374bytes,
SHA3368dd75296e09fa556debb1ed9ba7b7d65117a0c88d9fc45b2840edd4a37a47;
Rev5=209310bytes,
SHAae789b5f2e8a598f4152caa2666be663e736561a5653a91e6a1b6a31c475c32b.
Both unchanged seed4101 fits retain exact first-fit NPZ feature/raw/probability
arrays for8205 rows; Rust exported decisions0ULP, probability delta0 and no
fire disagreement. Both calibration screens:2536/2546 detected,0/160 honest
activation/lowering, real_bug149/155 (Rev4) or150/155 (Rev5). No labels changed.

All six full Rev5 R5CONFIRM bakes arrived and their SHA256/provenance and actual
Rev5 declarations pass. Repeat every six-bake Rev5 composition with both new576
and prior D228 heads. Repeat nine Rev4 new576 compositions (six primary plus
three original supplemental v2basicU), retaining original Rev1 B/D and prior
R5INTEG controls. Across21 full compositions plus a separately flagged smoke,
8510 rows each have maximum consumed-feature/scalar-cache/stored-f32 score
errors0. New576 heads pass all seven existing TRAIN calibration gates on every
bake; below-q20=2537..2539/2546 at Rev4 and2539..2540/2546 on actual Rev5.
Prior Rev5 D228 retains2524/2546 detection and3/160 honest activations/lowering,
failing all four specificity guards on each full bake. No promotion follows.

Prepared audits cover22 full compositions/110 rows:67 accept (3752 exact base
map queries),43 typed CorruptionDetected with zero queries. Each new576 head
accepts all four original honest probes and rejects preregistered TRAINpositive
index2, selected by lowest metadata index without inspecting predictions.
The prior D228 controls preserve their honest false activations and reject
correctly. Main six+six bake identity probes=144, with two cross-revision
refusals. Actual private plan inspection covers all nine Rev4 and six Rev5:
Rev4 byv2fy adds full-resolution X/B moments; v2basic/U adds no kernels.
Rev5 byv2fy adds X/B moments plus Peaks; v2basic adds Peaks. No unrelated family
is added at caller720. The assumption of zero runtime work is disproven; fit
columns reused existing extraction. Retain width1825 and earlier failed
bounded-audit/plan checks as diagnostics, not qualification evidence.

Independent judges: requested exec-cvvdp Docker image digestc1a97ef6..., CPU
zenmetrics-cli0.6.0 binarySHA37c3bc11..., seven keyed rows each, zero NaNs.
Origin6064 JXLd0.01 SSIMULACRA2=96.65016051457665, Butteraugli max=
0.11993369460105896, pnorm3=0.03428146946973271; AVIF160/176 SSIMULACRA2=
74.61134204762597/67.22642850810986, max=4.041477203369141/4.670391082763672.
Native and delivery PNG metrics agree exactly. Reference196x256 RGB8 has no
alpha/ICC/CICP/gamma/sRGB/EXIF chunk. JXL source8bit, no alpha, CICP1/13/0/full,
ICC536bytes (retained hash); AVIF8bit444, no alpha, CICP1/13/6/full. Legacy decode
via native codec registry and packed RGB8 projection matches retained delivery
pixel hashes; no inspector color conversion. Reference transfer is unknown;
legacy path assumes sRGB. Inspector binary is pinned, original producing commit
unknown (same disclosed HDRCORR inspector). Judge image binary/digest pin the
executed tool; local CLI source explains routes but is not asserted identical
to image source. Image AVIF-decode override is unset. New native metrics are
independent judges, not proof that shared codec implementations are infallible.

JXL changes550/50176 RGB8 pixels by at most1 code; mean absolute code difference
0.00371359481292517. A Rust literal reference-self forward through the old base
weights yields95.1181640625 at Rev4 and stamped Rev5, while public identity=100;
JXL base=93.6163558959961/93.6163330078125. This supports plausibility of the learned
base's93.6 without relabeling the honest pair. A preliminary ordinary identity
feature audit hit its existing zero-feature identity shortcut mismatch at f422;
no tolerance was relaxed. The explicit literal-forward diagnostic distinguishes
learned inference from the identity shortcut and passes at both revisions.

Checks pass:34 corruption tests,36 revision-contract tests,three native Rev5
whole-vector tests,13 Python admission tests,two standalone extractor audit
tests, explicit artifact plan and literal-forward diagnostics, CI just clippy,
standalone release extractor Clippy, just lint-scripts (811), root fmt and
extractor audit rustfmt. Subsequent additions are private diagnostic tests and
factual docs only; no feature-kernel, API, recipe, label or default change.
All canonical fit attempts use the existing content-admission hash-only guard,
including protected reference hashes as disclosed by original R5INTEG evidence;
no evaluation feature/pixel scoring, labels or sealed files were admitted.

Full summary is external157,769bytes (final pointer records exact size/hash); never
commit it. benchmarks/r5integ2_ARTIFACTS.json pins the external heads/summary,
full evidence scratch and archived handoff. All earlier R5INTEG_DONE and its
external summary/head evidence remain intact. Final report is written last,
after packaging, verification and the final local quarantine bookmark commit.

Final archive: /mnt/tower/output/zensim/r5integ-2026-10-04/r5integ2/evidence.tar.zst,
SHA2568c6595fe75b33884a38c6e8dbfd9ce19a48215cda49207f150ea36790ff180c0.
838 indexed files (8,156,720,700 original bytes) verify against every archive
member hash. Handoff/index/source pins, summary and both heads have verified
secondary copies beside the tower archive. The source/results commit is e445e549;
the final documentation/pointer commit is recorded in R5INTEG2_DONE.md.

### R5INTEG3 preregistration — 2026-10-04

Before fitting, register by-v2fy-420 at `benchmarks/r5integ3_REGISTRATION.json`:
exact 420 semantic IDs from the costset2 candidate JSON, caller width720,
Rev4+Rev5, unchanged seed4101/recipe/split/gates. The existing Rust metadata
owner verified all six Rev4/Rev5 by_v2fy full bakes against those exact IDs;
receipt/log pinned under `/var/tmp/r5integ/r5integ3/DECLARED_IDS.json`.
Use original immutable research columns, with matching producer pins and the
already corrected serving extractor. No fresh extraction for fitting.
The Rev5 fit/attach base is now actual R5CONFIRM byv2fy-full-s0, not smoke.
Register15 full compositions (9 Rev4 including supplemental U,6 Rev5),
the same five TRAIN prepared pairs and12-origin identity probes, plus exact
private ComputeSet equality for all15. Retain failures without tuning.
This is TRAIN development and cannot qualify a production head.
Receipt orchestration correction: Rust harness prefixed the first of six ID
lines; receipt parsing now accepts the prefix. The first registration commit
held only this worklog; the next commits the actual registration and pinned
six-bake receipt. Both precede either fit; no scientific setting changed.

R5INTEG3 fixed-fit and plan/probe results: both420-ID heads fit on the same
5499 fit/2706 calibration unique pairs. Rev4 calibration2532/2546 detections,
0/160 honest activation/lowering,147/155 real bugs; Rev5 calibration2534/2546,
0/160,148/155. Both fitted base compositions pass all seven unchanged gates.
Rust parity on8205 rows/head: exact raw margins/probabilities and fire sets.
Private plan equality PASSES all15 registered full compositions. In particular
Rev5 by_v2fy retains full_res_xb=false and v1_pools=Off; Rev4's inherent Peaks
mode is identical with/without head. No added family/channel/scale flags.
Prepared15 compositions/75 rows:60 accepted,15 typed CorruptionDetected,
3360 exact base-map queries, zero for rejections. Identity180 and cross-revision
refusal2 pass.14 admission tests,34 corruption tests,36 revision-contract
tests, clippy, lint-scripts and fmt pass. Full8510-row matrix remains running.

R5INTEG3 full matrix COMPLETE:9 Rev4 full compositions and6 actual Rev5 full
R5CONFIRM compositions, all7/7 gates. Detection2532/2546 (Rev4),2534/2546
(Rev5),0/160 honest activation/lowering,147/155 and148/155 real bugs.
Composed below q20 Rev4 2533–2537/2546; Rev5 2536–2537/2546. Each audits8510
TRAIN attempts. Feature, scalar/cache and stored-f32 score/probability errors
are zero. Exact fit/calibration recipe and TRAIN roles unchanged; no new fit
extraction. The two matching full-s0 audits reuse the fitter's identical
composition/pair packet;13 other matrix compositions were scored anew.
The420-ID heads lose4/2 calibration detections versus the576-read heads but
remove the extra scale-0 X/B work (and Rev5 Peaks), as exact private plans
establish. Retain both fixed heads and old controls; no tuning or selection.

R5INTEG3 archival verification COMPLETE:516 members,4224170695 uncompressed
bytes,1311489822-byte zstd archive at
`/mnt/tower/output/zensim/r5integ-2026-10-04/r5integ3/evidence.tar.zst`, SHA256
`aa7272e771cbd982fd69568fedd2e615e5cb9cdfcaceb3acb7afc0ee136fec4d`.
An initial unverified bundle included a live run-heavy status file; retained
under `unverified-attempt-1/`. The final wrapper uses a separate scratch TMPDIR
and excludes ephemeral tmp.* files. Every final member hash matches its index.
Large summary3912179 bytes SHA256
`2187ddcb0bda0d99a45a445372b736d39e8b8d690a0f2dd72214acf51889e329`
and both heads have verified tower mirrors; only small pointer JSON is committed.
`benchmarks/r5integ3_ARTIFACTS.json` binds heads, summary, full archive/handoff.
Coordinator landed the earlier chain as4e3e23de; this branch stays on2d5e7ec0
plus R5INTEG3 commits for landing its net diff. No push/rebase/integration.
R5INTEG3_DONE.md will be written last after the final local commit and checks.
