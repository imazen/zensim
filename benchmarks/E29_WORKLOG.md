# E29 preparation — 2026-10-07

Missing: combined v40 package/image review and shared matched-control completion; completed matched-control pins and full hb4/hc4 fits;
publication/queue enrollment by the coordinator; SDR/HDR assessment and product
qualification. An unintended legacy HDR VAL import read violated the preparation
restriction and remains an explicit review exception. This lane prepared and
tested plumbing only, with no push or launch.

The amendment was committed as `788cfb50` before any E29 smoke. It retains hb4
and hc4, the raw two-teacher 0.05 agreement threshold, nominal HDR weight four,
E21 SDR guards and equal four-fold paired composites over ten seeds. The new
brief explicitly requires significant pooled gains against both teachers.
Missing/nonfinite metrics and zero SE are INCOMPLETE. AIC is excluded.

Implementation uses the existing strict `v2_lodo_mlp.py`/`zensim_mlp_train`
stack, preparation packer, statistics owner and HDR panel. The explicit hc4
pair-list entry is research tooling in the unpublished validation crate. The
ordinary trainer entry and inactive pair-list RNG remain unchanged. The coordinator has chosen one shared fresh matched control for E29/E31/E32
under combined v40. This supersedes the solo control proposal. Exact E30
reuse is now rejected before checkpoint or label opens. The lane runs one
full control-recipe validation cell to test baseline numerical parity; it is
not part of the shared scientific control and cannot change its selection.

## Data and exposure

Prepared root: `/mnt/v/output/zensim/e29-2026-10-07/v2e29`.
It copies only the frozen D1 inventory and existing native HDR TRAIN authority.
D1 roles remain those of [DATA_SPLITS](../docs/DATA_SPLITS.md). Four human
sources are kadid, tid2013, konfig, cid22_a25; teacher/coverage inputs and ordered
420 feature IDs retain SHIPPATH admission. Metadata and label-free key
populations precede all payload hashes and reads, including lower-owner checks.

Existing E26/E27 authority contributes 7,390 agreement-only TRAIN rows over
495 references. Native features are bit-preserved; only targets change. hb4
uses pooled normalized midrank Borda targets. hc4 contains all 16,140,412
lexicographically ordered unique cross-reference pairs passing both raw JOD
thresholds and matching signs. An independent audit compares all feature IEEE
bits, recomputes targets using NumPy unique-count ranks and checks every pair,
including completeness against the full eligible universe.

One unintended legacy HDR VAL read occurred during an added import test: the
unguarded legacy panel opened all 22,860 rows/952 columns of
`hdrgrid_mc944_t2_val.parquet`, printed target swings and failed before student
prediction. No fit used those rows. This violates the preparation read limit
and requires disclosure in coordinator review; see `UNINTENDED_EXPOSURE.json`.
No new teacher run, AIC discovery/read, registered 3,900-row HDR VAL,
confirmation or protected human label read occurred. D1 table payloads transported into image smokes include
only approved training/internal-development populations; held-out full human
tables were packed after admission but not scored. Existing HDR TRAIN teacher
columns were read solely for the registered targets/pair list. Metadata and
selected model artifacts from the exact 40 E30 nA3 cells were independently
verified and pinned after they completed; no E30 label payload was opened.

Native HDR's historical subset has no full-family feature-set identity. HDR
checkpoints explicitly remain `qualified_provenance=false`, with Rev5/subset
provenance and pair-list digest retained. Seven SDR admissions remain strict.
The research inspector expects this limitation; the ordinary qualified inspector
still refuses it. No serving model or human HDR qualification is claimed.

## Earlier solo preparation pins and executor evidence

The coordinator superseded solo E29 packaging with the combined v40 package.
The following archives and image are as-run preparation evidence, not launch
authority or pins for the future combined package. No new solo package or
image is built after that decision.

[Artifact pins](/mnt/v/output/zensim/e29-2026-10-07/PINNED_ARTIFACTS.json)
record producer commits, image identity, binaries, manifests and every smoke.

- Data: `9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008`
  (70 pinned files, 21 table payloads; 19 SDR and two HDR target variants).
- Program: `44ca24737293b404e29e150ebce78632cc21d5de7b6416c3058fb53c2100ef58`
  (31 pinned files; zenmetrics profile commit `f15cb9202532daf97b5ec882380088b7800abf66`).
- Local image: `ghcr.io/imazen/zenfleet-worker:fit-e29-consensus-v41-w925f9783329f`;
  ID `sha256:3b85136b13c37d30ae74d30a767bed49dca1e4d13643b26ad0ce4ae59335cb15`.
  Its saved image and build log are retained. No registry push occurred.
- Full-budget arm manifest: `592754b6968bfa2c1b94c8fd1f59cd8834e7cdf1e5320944503afda2a4de5217`;
  80 cells, four folds × ten seeds × two arms. The proposed control manifest
  has 40 cells; every destination is outside admitted input roots.
- E30 completed-cell pins: `ecf82b68142946e7be1642affbdb006fcc013f94b9d549044ee3a9a3cad84aeb`.
  All 40 selected checkpoints pass the original program-bound trusted contract
  and canonical inspector at epoch 119/120, 50,000 draws per epoch. This pins
  completion, not an E29 exact-parity control approval.

Three distinct declared local jobs (base/hb4/hc4, 2 epochs × 128 draws) ran
inside the image through `/usr/local/bin/fit-cell-exec`, including fresh archive
extraction, SHA verification and `/var/tmp/rev4-featpot` linking to the
hash-named `/scratch/fit-cell/.../rev4-featpot`. All three returned verified
blobs with epoch 1 selected. hb4 and hc4 additionally ran their unchanged
registered argv (120 × 50,000), reached epoch 0 and were deliberately stopped
by the test driver; those are incomplete smoke runs, never scientific cells.

The panel CLI is now protected by a main guard; importing it cannot trigger
its legacy data read. Import and absent-control HDR panel tripwires record zero
payload opens/process calls. The original failure log is preserved.

The owner harvest accepts the short blobs only with `allow_local_smoke=True`.
Twelve re-signed negative blobs (budget, false epoch119, admission, mode bypass)
are refused; normal installation also refuses all three short blobs.
Clean-environment `e24_rev5.py e29-score --preflight-only` passes against the
staged root. Full assessment requires complete frozen controls/cells before
any labels; its absent-control tripwire records zero table reads.

Actual container cap was one CPU and six GiB, no swap, tier v3 and one Rayon
thread. `declare-fits` retains its historical 2-GiB/four-thread packing hint;
`jobset_caps.json` separately records the intended actual envelope. Bounded
base/hb4/hc4 `/usr/bin/time -v` maximum RSS was respectively 1,058,412 /
970,724 / 1,059,196 KiB. The corresponding cgroup peaks were 2,027,532,288 /
1,917,644,800 / 2,339,270,656 bytes. hc4's full-argv first-epoch cgroup peak
was 2,373,599,232 bytes. These are smoke measurements, not full-fit forecasts.
Outer Docker wrapper: `run-heavy: done rc=0 13s | peak-RSS 0.03GiB |
min-avail 31852MiB | peak-load 2.43` for that hc4 first-epoch run. Full pair
universe audit: `run-heavy: done rc=0 36s | peak-RSS 3.49GiB |
min-avail 30538MiB | peak-load 2.66`.

## Validation and review

Evidence logs preserve successful and earlier failed attempts. E29/E26 Python
suite passes 19 tests, sampler suite 22, SHIPPATH admission regressions 27 and
zenmetrics fit-tool suite 53. Release build and clippy with warnings denied
pass; script lint checks 847 runnable scripts. These validate the exercised
plumbing, not scientific success or full product qualification.

The complete earlier artifact tree remains on tower. After every original
file was byte-verified, the user authorized removal of only this lane's
superseded local extraction stages, duplicated negative blobs and Cargo/source
caches. `CLEANUP_RECEIPT.json` records 15,980,081,152 allocated bytes removed.
`ARCHIVE_MIRROR_RECEIPT.json` on tower preserves the complete pre-cleanup
11,020-file proof; the final local mirror receipt covers the retained subset.
The local prepared root, one data/program/image archive, binaries, manifests
and evidence remain available. No unrelated cache or artifact was removed.
