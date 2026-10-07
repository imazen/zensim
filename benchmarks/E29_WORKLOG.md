# E29 preparation — 2026-10-07

Missing: coordinator registration of the four-source amendment and the single
matched-control choice; completed matched-control pins and full hb4/hc4 fits;
publication/queue enrollment by the coordinator; SDR/HDR assessment and product
qualification. This lane prepared and tested plumbing only, with no push or launch.

The amendment was committed as `788cfb50` before any E29 smoke. It retains hb4
and hc4, the raw two-teacher 0.05 agreement threshold, nominal HDR weight four,
E21 SDR guards and equal four-fold paired composites over ten seeds. The new
brief explicitly requires significant pooled gains against both teachers.
Missing/nonfinite metrics and zero SE are INCOMPLETE. AIC is excluded.

Implementation uses the existing strict `v2_lodo_mlp.py`/`zensim_mlp_train`
stack, preparation packer, statistics owner and HDR panel. The explicit hc4
pair-list entry is research tooling in the unpublished validation crate. The
ordinary trainer entry and inactive pair-list RNG remain unchanged. The new
program cannot establish E32's exact-binary reuse parity with E30 v39: exactly
one 40-cell matched control is proposed, with its manifest frozen before full
fitting. No fresh control was fitted in this lane.

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

No new teacher run, AIC discovery/read, HDR VAL, confirmation or protected human
label read occurred. D1 table payloads transported into image smokes include
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

## Pins and executor evidence

[Artifact pins](/mnt/v/output/zensim/e29-2026-10-07/PINNED_ARTIFACTS.json)
record producer commits, image identity, binaries, manifests and every smoke.

- Data: `9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008`
  (70 pinned files, 21 table payloads; 19 SDR and two HDR target variants).
- Program: `4153dea74b21db8cf680bc67e95b405578d35dadbaec53e5dc86117093e12e22`
  (31 pinned files; zenmetrics profile commit `95edcc31bc7f17d6be80573c62f0b0f32f3ac4ef`).
- Local image: `ghcr.io/imazen/zenfleet-worker:fit-e29-consensus-v40-w925f9783329f`;
  ID `sha256:b73ee13c60322503eac1d3dbf409ff0dea8ca08b0bfe639c090c79ed9f5ab756`.
  Its saved image and build log are retained. No registry push occurred.
- Full-budget arm manifest: `16dbf77ac5e469362728a9c805360810447b493d857c7320e46ac330a72f8f52`;
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

The owner harvest accepts the short blobs only with `allow_local_smoke=True`.
Twelve re-signed negative blobs (budget, false epoch119, admission, mode bypass)
are refused; normal installation also refuses all three short blobs.
Clean-environment `e24_rev5.py e29-score --preflight-only` passes against the
staged root. Full assessment requires complete frozen controls/cells before
any labels; its absent-control tripwire records zero table reads.

Actual container cap was one CPU and six GiB, no swap, tier v3 and one Rayon
thread. `declare-fits` retains its historical 2-GiB/four-thread packing hint;
`jobset_caps.json` separately records the intended actual envelope. Bounded
base/hb4/hc4 `/usr/bin/time -v` maximum RSS was respectively 1,015,280 /
1,075,972 / 1,039,772 KiB. The corresponding cgroup peaks were 2,472,161,280 /
1,967,529,984 / 2,254,950,400 bytes. hc4's full-argv first-epoch cgroup peak
was 2,811,379,712 bytes. These are smoke measurements, not full-fit forecasts.
Outer Docker wrapper: `run-heavy: done rc=0 12s | peak-RSS 0.03GiB |
min-avail 30489MiB | peak-load 1.46` for that hc4 first-epoch run. Full pair
universe audit: `run-heavy: done rc=0 31s | peak-RSS 3.48GiB |
min-avail 28938MiB | peak-load 2.43`.

## Validation and review

Evidence logs preserve successful and earlier failed attempts. E29/E26 Python
suite passes 18 tests, sampler suite 22, SHIPPATH admission regressions 27 and
zenmetrics fit-tool suite 53. Release build and clippy with warnings denied
pass; script lint checks 847 runnable scripts. These validate the exercised
plumbing, not scientific success or full product qualification.

Fresh workspaces `zensim--e29` and `zenmetrics--e29` retain local
`quarantine/codex/e29` bookmarks for review. Source commits are not pushed.
The artifact tree, image archive, prepared root and logs are mirrored to
`/mnt/tower/output/zensim/e29-2026-10-07/`; the mirror receipt verifies every
file and records three selected artifact hashes. No cleanup discards evidence.
