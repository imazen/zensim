# E29 four-source amendment — draft before fitting, 2026-10-07

Coordinator review/registration is required. This local draft does not launch
E29. It supersedes the five-source/control/statistical clauses of
[e29_hdr_consensus_registration_2026-10-07.md](e29_hdr_consensus_registration_2026-10-07.md)
and implements the coordinator's E29 brief. No E29 fit preceded this draft.

D1 sources are kadid, tid2013, konfig, cid22_a25, with their approved
KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL and CID22-A25 roles.
AIC-3 and its holdout family are excluded from training, preprocessing,
packing, scoring, control discovery and all new reads. Seeds 0..9 × four
LODO folds give 40 cells per arm. SafeSyn/CID22 teacher and coverage inputs,
420 by_v2fy IDs, Rev5, head N, H128, h32, cv16/cf98, optimizer, preprocessing,
seed streams and 120 epochs × 50,000 draws/final epoch 119 remain fixed.

Retain both hb4 and hc4; neither may be dropped. Use exactly the existing
7,390 E26/E27 corrected HDR TRAIN agreement rows and native Rev5 features.
No new teacher runs, HDR development group, HDR VAL/confirmation reads or
UPIQ ingestion occur during preparation. hb4 averages each teacher's pooled
mid-rank normalized as (rank-1)/(7390-1), then uses pooled rank loss. hc4
uses only cross-reference pairs with matching teacher order and both raw
JOD differences >= 0.05 in absolute value. Its target order is that agreed
order; no normalization changes the threshold. Nominal HDR weight is 4
in the unchanged E26 reference-count acceptance convention.

E29 may launch only after all 40 E30 nA3 cells are complete and pinned.
Control is nA3, never E30's historical E24 comparison control. Before fitting,
freeze complete final-119 result/bake pins and exact parity of labels/roles,
ordered SDR feature bits, teachers/coverage, transformations/scalers,
loss weights, optimizer, sampler/RNG/seed streams and fit program/binaries.
A strict-admission, pair-list or program change affecting baseline training
invalidates reuse. If exact reuse cannot be proved, register exactly one
fresh matched 40-cell control before fitting, after E30 completes; no mixing
or outcome-dependent baseline choice. Preparation smokes are explicit local
jobs, cannot install as full-budget cells and provide no adoption evidence.

The inferential units are ten seeds. For every SDR or HDR endpoint first
average paired arm-minus-control deltas equally across all four folds within
seed; report mean and sample_sd(ddof=1)/sqrt(10). Never treat 40 cells as
independent or combine source SEs in quadrature. Missing/nonfinite metrics,
undefined distortion-type correlations, incomplete cells and zero SE mean
INCOMPLETE. SDR E21 guards remain mean >= -0.002, every source's ten-seed
mean >= -0.005 and W2 > -2 paired-seed SE. W2 uses each arm/control panel's
worst three signed distortion-type SROCCs for KADID/TID separately; average
the two paired source differences within seed, then reduce over ten seeds.

At later registered assessment on the unchanged 3,900-row hdr_v3mix VAL,
pooled and within-reference SROCC against HDR-VDP-3 and historic CVVDP must
each be >= -2 paired-seed SE. Pooled gain must exceed +2 paired-seed SE
against BOTH teachers, as explicitly required by the coordinator's new brief
(the original E29/E27 text required improvement against HDR-VDP-3 only).
Both teachers and all SDR guards are conjunctive. Adopt the passing arm with
larger pooled HDR-VDP-3 gain; exact ties adopt none. No passing arm retains
the control. Borda VAL, hp4 history and external SDR are report-only.
This is teacher-agreement research, not independent human HDR qualification.

Freeze admission metadata/key populations before any payload hash/read;
strict four-source SDR admission and native HDR subset provenance stay
explicit, without relabelling the latter as a full RGB feature-family table.
Freeze data/program/image/manifest hashes, build_commit, binary identities,
exact declared grid, role receipts and actual fit-cell-exec smoke receipts.
Stage the identical prepared root locally for the scorer and mirror evidence
and artifacts to tower. Missing control/registration pins block launch.

Preparation incident addendum (before any full E29 fit): a new import tripwire
initially imported the legacy HDR panel before installing its mocks. The old
module executed its legacy CLI on import and read the 22,860-row
`hdrgrid_mc944_t2_val.parquet` default (952 columns), including teacher targets
and features. It printed target swing diagnostics and failed before any student
prediction because the predictor executable was absent. No fit, target, pair
threshold, control choice or decision gate used that table; the registered
3,900-row hdr_v3mix VAL and confirmations were not opened. The unintended
legacy HDR VAL read violates the preparation restriction and is disclosed for
coordinator review. It cannot be undone or labelled unexposed.

The panel CLI is now guarded by `main`; import and absent-control HDR panel
tripwires intercept reads/processes before execution. Rebuilt program/image
pins replace earlier smoke-only versions, which are preserved. No full fit or
new assessment is authorized by this corrective addendum. Incident receipt:
`/mnt/v/output/zensim/e29-2026-10-07/UNINTENDED_EXPOSURE.json`.

Coordinator control decision (2026-10-07, before any scientific arm fit):
E29, E31 and E32 use one shared fresh matched control under the combined v40
program. Its 40 cells preserve the E30 nA3 by_v2fy / N / Rev5 D1 recipe,
four folds, ten seeds, 120 epochs × 50,000 draws, final epoch 119. This
supersedes the exact-E30 reuse option and solo E29 control proposal above.
A full control-recipe baseline parity cell is a validation check, not a
choice between controls. A mismatch is recorded; the shared fresh control
still runs. No E30 cell may replace it based on results. The combined package
and image are coordinator work later; this lane lands code and tests only.
