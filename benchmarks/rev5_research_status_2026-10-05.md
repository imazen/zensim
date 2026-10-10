# zensim Rev5 / by_v2fy research — status and plan (2026-10-05 20:40 UTC, coordinator hand-off)

Written because the coordinator (Claude) is out of usage; work continues in SWE-2 (devin) and Codex-sol lanes. Everything
below is measured or on main unless marked OPEN/PLAN. Main = `scripts/safe_push.sh` only. Feature-potential page (owner view):
https://claude.ai/artifact/FP3ZGtzRa4thGi91yKMErZ (v50).

## 1. Where things stand

| Item | State | Record |
|---|---|---|
| Rev5 arithmetic (by_v2fy families only; f32 local windows, 16 lanes, FMA, stable moments, exact identity) | landed, FROZEN at 60174678 | `benchmarks/rev5_spec_2026-10-04.md` |
| E24 Rev5 retrain vs Rev4 by_v2fy | as good (+0.0011 ± 0.0010); external NITS −0.0040 ± 0.0016 | spec Addendum C |
| E25 Rev4-trained weights served at Rev5 | practically identical to Rev4 (Δ ~1e-6); retrain differences were retraining variance | spec, `benchmarks/e25_2026-10-04/` |
| Steering at Rev5 | Rev4 weights at Rev5: broad 93/96, owner 12/12; Rev5 retrain curvier (85/96) — not a gradient defect | `r5steer*`, R5STEER2 |
| STEERCODEC / CHROMAQ at Rev5 | steering like Rev4 on zenjpeg/JXL/zqi; chroma agreement slightly better | `steercodec_2026-10-04.md`, `chromaq_2026-10-04.md` |
| Integrity companions | by_v2fy-420 heads (zero added extraction) pass all 7 TRAIN gates at Rev4/Rev5; ZCTH v4 provenance; prepared-steering companion-read bug fixed (Rev1–4) | f614ffa9, bf965c5f, 06ea889a |
| Recommended Rev5 artifact | Rev4-trained full-data by_v2fy stamped Rev5 + Rev5 by_v2fy-420 head — **research artifact only**: featpot fits carry historical-replay table admission, `freeze_check` fails them by design | spec correction 2026-10-05 03:25 |
| SHIPPATH (qualified path) | chunks 1–4 landed: qualified Rev5 table admission for all recipe tables; strict trainer route (refuses human legs without an owner decision record); epoch-dump provenance; ZCTH v4; Rev5 assessment feature tables (15 tables, 26,911 rows, features only); protected-path guards | b68273a0 … 6874a981; `benchmarks/shippath*` |
| HDR teacher (HDRTEACH) | HDR-VDP-3.0.7 labels for 7,425 TRAIN + 3,900 VAL; within-ref agreement with CVVDP 0.9988/0.982; UPIQ cannot test a student (HDR-VDP-3 calibrated on it) | `benchmarks/hdrteach_2026-10-04.md` |
| E26 (HDR-VDP-3 within-ref rank leg) | registered rule adopts hd4 (within-ref +0.0015 ± 0.0003) BUT pooled HDR-VDP-3 SROCC −0.045 (hd4) / −0.125 (hd16): not a shipping candidate | merge caee5bed; `E26_DONE.md` |
| E27 (HDR leg forms keeping cross-image comparability: hp4 pooled rank, ha4 within-ref MSE+rank) | **RUNNING** on the fleet since 19:21 UTC (fitv2e27-20261005, 100 cells, ~5–6 h); pre-launch review clean | registration 070b8247; `/var/tmp/e27/` |

## 2. Open owner decisions (asked, not yet answered)

1. **Human-data role for the production fit** — may the five design-released sources (KADID TRAIN+SELECT, TID2013, KonFiG
   TRAIN+VAL, CID22-A 25 refs, AIC-3) train the qualified Rev5 by_v2fy? Request: `~/tmp/zensim-paper/rev4/SHIPPATH_decisions.md`;
   the strict route needs a `shippath-human-role-decision-v1` JSON record. Blocks SHIPPATH chunk 5 (full qualified fit + gates).
2. **KADID TERMINAL confirmatory read** (2,000 pairs, never read) — register and pin before any label is opened.
3. Answered 2026-10-05: PID 760429 killed (not a worker); disk: remove /home temporary/generated items already on tower, keep
   what the next research stage uses (hand-off lane DISKCLEAN below).

## 3. Running lanes (herdr workspace w17)

* `hdrcorr-sol` (Codex gpt-6.1-sol): owns E27 — after all 100 cells harvest and the SDR score posts
  (`==== E27 RESULT` in `/var/tmp/fitv2/status.log`), run `/var/tmp/e27/HDR_PANEL_AFTER_HARVEST.sh` and
  `EXTERNAL_SDR_AFTER_HARVEST.sh`, apply the registered rule, write `E27_DONE.md`, rebase records onto main (no push).
* `shippath-sol` (Codex): idle; next work blocked on decision 1.
* SWE-2 hand-off lanes (this document's §5).

Fleet notes: E26/E27 use a 6 GB per-cell envelope via `/var/tmp/fitv2/jobset_caps.json` (consumed by `launch_v2.sh`); tower's
docker lacks ghcr credentials (push images from dev); E26's last two cells were stuck 4 h while tail_trim ran — do not use
tail_trim until diagnosed (it is omitted from E27).

## 4. Next steps (in order)

1. E27 verdict → if an arm passes, it is the HDR leg candidate; land records (independent review first, merge commit).
2. SSIM2-recipe experiment (owner question 2026-10-05, §6) — research brief then registration (E28), then fleet run.
3. Decision 1 → SHIPPATH chunk 5: full qualified by_v2fy fit through the strict route, then the gate map
   (`benchmarks/shippath_gate_map_2026-10-05.md`): G-RANK, G-DIAL, G-ADDR/codec floors, tails/identity, G-STEER, G-RD, G-TARGET,
   integrity, HDR, runtime. Decision 2 provides the confirmation read.
4. Rev5 quiet-box speed qualification (zenbench, 64²–4 MP, v4x/v3, vs Rev4/Rev3; MT scaling) — now unblocked (PID 760429 gone).
5. tail_trim diagnosis (fleet tooling): why E26's last holders kept dying.

## 5. Hand-off lanes (SWE-2, started 2026-10-05)

* `DISKCLEAN` — `~/tmp/zensim-paper/rev4/DISKCLEAN_brief.md`: remove only /home items with a verified byte-identical copy on tower;
  keep current-research items; ledger with sha256s.
* `SSIM2RECIPE` — `~/tmp/zensim-paper/rev4/SSIM2RECIPE_brief.md`: verify the SSIMULACRA2 tuning recipe from primary sources, map our
  data, draft the E28 registration (no fits).

## 6. Owner question: can we use SSIMULACRA2's training recipe? (preliminary answer; SSIM2RECIPE verifies)

Recipe as stated by the owner: TID2013, KADID-10k, KonFiG-IQA (F boosting) and the CID22 training data; Nelder-Mead simplex;
minimise MSE and maximise Kendall and Pearson correlation.

* **Data — mostly yes, and we already train on most of it.** by_v2fy's human union is KADID (TRAIN+SELECT), TID2013, KonFiG
  (TRAIN+VAL), CID22-A (25 refs) and AIC-3. TID2013 is train-only by ruling; KADID TERMINAL stays held out; KonFiG overlaps MCL-JCI
  (R7a guard). Whether our KonFiG tables include the F-boosted condition needs checking.
* **CID22 training data — use SSIMULACRA2 as a teacher, as the owner suggests.** We have the CID22 training references but not
  their human scores (the public human scores are the 49-reference validation set, which is our holdout). Scoring the CID22
  training pairs with our own SSIMULACRA2 (fast-ssim2 / zenmetrics, imazen code, allowed as a teacher) gives a label that encodes
  SSIM2's own fit to the CID22 training humans. Our existing CID22 oracle teacher leg may already be this — check which metric its
  labels are.
* **Objective — partly new.** Our trainer uses within-reference pairwise rank (a differentiable Kendall surrogate) plus MSE on
  teacher legs. A pooled Pearson term is new and directly addresses E26's lesson (cross-image comparability).
* **Optimiser — Nelder-Mead fits SSIM2's ~100-weight formula, not our MLP.** It is derivative-free and scales poorly past a few
  hundred parameters. Two faithful options: (a) an SSIM2-style low-parameter head (linear or monotone over by_v2fy's 420 features)
  fitted by Nelder-Mead/Powell on MSE + (1−Kendall) + (1−Pearson) — cheap, interpretable baseline; (b) keep the MLP and add pooled
  Pearson and a Kendall surrogate to its loss.
* Proposed E28 (to register before fitting): arms (a) and (b) above with the SSIM2 data mix incl. CID22-train SSIM2 teacher; E21
  as-good rule on LODO plus external sets; report Kendall/Pearson/MSE per source.

## 7. Updates (2026-10-05 22:00 UTC)

* **SSIM2 recipe — verified** (`~/tmp/zensim-paper/rev4/SSIM2RECIPE_DONE.md`, draft `E28_registration_DRAFT.md`). Corrections to §6:
  the objective's weights are unpublished (libjxl `tools/ssimulacra2.cc:286-291`: MSE on CID22-train only, Kendall on all four
  sets, Pearson at a lower weight); SSIM2 tunes ~113 parameters (108 weights + remap); KonFiG "F boosting" is the flicker-boosted
  triplet reconstruction (KonFiG Exp I), which our `konfig_*` tables do NOT carry (they carry the nominal design grid
  `1 − q_jnd/3.2`) — reconstructing it is a port of the authors' MATLAB Thurstonian fit over raw data we hold; the CID22-train leg
  (201 refs, 17,611 pairs) already carries self-computed fast-ssim2 labels, i.e. the owner's suggestion is the standing design, and it
  is self-distillation of SSIM2's fit, not CID22 human signal; Nelder–Mead suits ≤~128 grouped weights, Powell beyond.
* **E27:** 24/100 done at 21:51 UTC (~10 cells/h; completion ~05:30 UTC Oct 6).
* **DISKCLEAN:** freed 4.3 GB (one byte-identical parquet); ~187 GB kept pending owner decision because no byte-identical tower copy
  exists (`~/tmp/zensim-paper/rev4/DISKCLEAN_DONE.md`). Anomaly to check: `tbig_720_full.parquet` and two `tbig-join-out` parquets
  have the same size but different sha256 locally vs tower — possible silent corruption on one side.

## 8. Updates (2026-10-07 09:05 UTC)

* **E27 result** (`~/tmp/zensim-paper/rev4/E27_DONE.md`, registration 070b8247): neither arm passes; the E24 Rev5 control stays.
  hp4 (pooled rank on 10 × q_jod) keeps SDR as good (+0.0006 ± 0.0011) and lifts pooled HDR-VDP-3 SROCC 0.846 → 0.969, but pooled CVVDP
  falls 0.930 → 0.841 (fails non-inferiority); ha4 (within-ref MSE + rank) fails SDR (−0.039 ± 0.0015). The two teachers agree within
  references (0.998) but only 0.828 pooled, so a single-teacher pooled leg makes the model that teacher's cross-image calibration. Next:
  a consensus leg (pairs where both teachers agree on cross-image order) or human HDR data (UPIQ needs owner approval).
* Loop plan and state: `~/tmp/zensim-paper/rev4/LOOP.md` (owner directive 2026-10-07).
* **Disk (2026-10-07 09:35 UTC):** owner-approved move of the INUSE-audit `MOVE_TO_TOWER` set finished — 124.4 GB (72 dirs + 647
  strictly unreferenced probe root files) to `tower:/mnt/user/coefficient/archive/mntv-2026-10-07/<same relative path>`, each item
  sha256-manifest-verified on both sides before local deletion (ledger `~/tmp/zensim-paper/rev4/MNTV_MOVE_LEDGER.tsv`). Plus the zensr
  move (55.2 GB, `ZENSR_MOVE_LEDGER.tsv`). NVMe now 122 GB free. Not moved (owner decision pending): decoded-image caches (~183 GB, encodes
  durable), 3 ASK items; `datasets/*`, `dataset/*`, `input/papers` stay local by owner rule. E28 registered (82c9af81), E29 registered
  (982f3983), KonFiG F-scale reconstruction lane running.
* **KonFiG flicker-boosted (F) JND scale reconstructed** (2774d1d1; `scripts/canonical_corpus/konfig_fscale.py`, output
  `/mnt/v/dataset/konfig-iqa/derived/konfig_fscale_trainval_2026-10.parquet`, TRAIN+VAL sources only; held-out test sources dropped
  before parsing). Oracle: the authors' unmodified MATLAB under GNU Octave agrees within 0.005 JND over all 637 values. Kendall τ between
  our SSIMULACRA2 and the F scale = 0.772 (SSIM2's published Part-A figure 0.767); the design-grid label we train on scores 0.584 on the
  same cells. Scale is distortion-oriented and not level-monotone by design (23/49 sequences, reproduced by the oracle). E28 keeps its
  registered design-grid KonFiG label; an F-scale variant needs its own registration after E28.
* **E28** implementation ready (`E28_READY.md`): s2o/s2m smokes pass (KADID held-out SROCC 0.878 / 0.769), legacy paths byte-identical,
  285 Rust + 175 Python tests pass; nm diagnostic non-convergent within its fixed budget (diagnostic only). Pre-launch review running.

## 9. Owner decisions (2026-10-07) and follow-up

* **D1** production human data = KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL, CID22-A; AIC-3 stays in the JPEG-AIC holdout family
  (with AIC-4 and SDR25). The R7 AIC-4 read is flagged as possibly contaminated (R7 confirm fits trained on AIC-3). E30 (d61eac11)
  measures the cost of dropping the AIC-3 leg (no AIC-3 label read). The strict route is being extended to the four-source record.
* **D2** KADID TERMINAL: registered (`benchmarks/kadid_terminal_registration_2026-10-07.md`); read once on the final qualified model.
* **D3** UPIQ-380 re-designated T2 HDR training data (DATA_SPLITS). An HDR human-leg experiment will be registered after ingestion.
* **D4** disk: decodes with locally durable encodes deleted (ladder decodes kept); the two KADIS PNG sets were MOVED to tower instead of
  deleted because their R2 source (`s3://zentrain/kadis-700k-gpu/distorted/`) no longer exists; gen-* encodes and ~/tmp leftovers moved
  to tower (verified); `~/tmp/aic2026` unzip deleted after its audit products were archived; tbig copies being compared with R2.
* **New owner request:** dominant-colour (top-N, N = 2–8) shift features — implementation lane `palette-sol`; potential experiment E32 to be
  registered before any fit.

## 10. Updates (2026-10-07 12:55 UTC)

* **E28 (SSIM2 recipe).** The focused re-review closed both original admission findings and found one more: an already-prepared fit
  root containing an extra `confirm`/`HDR` directory reached the CID22 teacher checksum before being refused. Fixed in the E28
  quarantine chain (`ab0b395f`: the shared prepared-root admission walks the root for forbidden directories before any checksum;
  regression fails 4/4 without the fix, 16/16 tests pass with it). The fix changes a packed program payload, so the package is being
  re-pinned (v35). Then: re-review, land, push the zenmetrics profile, launch. No E28 fit has run.
* **tail_trim root cause** (read-only diagnosis): `zenfleet-worker`'s SIGTERM claim release deletes the R2 claim without checking that
  this worker still owns it, so a stale ex-owner deletes a newer owner's claim and the cell loops. Fix (ownership-conditional delete,
  owner-checked renew) is being implemented with tests that reproduce the E26 loop; tail_trim stays off until it lands.
* **D4 complete.** tbig: R2, tower and `/mnt/v/zen/tbig-720-2026-07-22` agree (sha `3fc1ef81…`). The stale `~/tmp/tbig_720_full.parquet`
  is the same file with one ~1 MiB zeroed region at offset 28.9 GB (a dropped download chunk), so it is a corrupt copy; it was archived
  to tower rather than deleted. The ~/tmp join files are being hashed against tower: identical ones deleted, divergent ones archived.
* **E32 (palette shift)** draft registration exists (palette_v2 after a sign-inversion fix found by a synthetic darkening test; four-source
  folds per D1). Extraction is running; registration follows the lane's report and review.

## 11. Updates (2026-10-07 13:45 UTC)

* **E28** v35 package built: only `e28_recipe.py` and build metadata changed; the data archive is byte-identical to v34; the reviewer's
  25-case admission probe refuses all 25 with zero label-bearing opens; the installed-image smoke matches v34 exactly. Independent
  re-review running; then land, push the zenmetrics profile, launch.
* **Worker claim fix** implemented locally (ownership-checked release and renewal, seven regression tests that fail on the old code,
  61 crate tests and clippy pass). Independent review running. It currently sits on unreviewed D1 profile commits and will be rebased
  onto `master` before landing.
* **Palette (top-N colour shift) features** built: 42 features (N = 2–8, six signals each), extracted for 254,778 bank rows, verified
  independently, serving reads refused. Measured limits: hue direction agrees on 97/168 synthetic cases; the features are blind to
  high-frequency chroma loss (the CHROMAQ HF ladder barely moves them). Review running; E32 registration follows it.
* **Main test fix.** SHIPPATH7 registered a by_v2fy projection producer that broke three registry tests on main. Two are fixed
  (`47a2e1d4`); the third (`zensim-validate` `basic_only_bake_compatibility_respects_partial_producers`) needs an owner decision on its
  expected value and is logged in CLAUDE.md Known Bugs.
* **Disk:** D4 and the tbig cleanup are complete. All three divergent ~/tmp tbig copies were the canonical files with 1–2 MiB zeroed
  regions (dropped download chunks); they are archived on tower, not used.

## 12. Updates (2026-10-07 14:20 UTC)

* **E28 landed** on main (`f13b695c`, merge; 184 Python and 285 zensim-validate tests pass on the merged tree) and the zenmetrics fit
  profile is on master (`2659b6c2`). **The first fleet launch failed**: all 100 cells refused at admission because the admission check
  rejected any symlink in the data root's absolute path, and the fit executor deliberately reaches that root through a link. The local
  image smokes never went through the executor, so they missed it. The jobset was stopped and dequeued within ten minutes; no fit ran.
  Fix `8283d518` (only components below the data root may not be links; regression reproduces the fleet error) is under review, and the
  v36 package is being built with a mandatory smoke through the executor's real entry point. Relaunch as `fitv2e28b-20261007`.
* **Worker claim fix**, round 2: both review findings fixed (same-owner renewal collision no longer drops a live chunk; malformed
  timestamps never authorize delete), rebased onto `master` alone. Re-review running.
* **Fleet capacity:** i270 currently refuses the fleet's SSH key and presents a changed host key, so it is out until checked; E28 runs on
  tower, i265, r3500 and r3800x (12 concurrent cells).

## 13. Updates (2026-10-07 14:50 UTC)

* **E28 is training.** Relaunched as `fitv2e28b-20261007` at 14:40 UTC with the reviewed admission fix (zensim main `0f2946bf`,
  zenmetrics `ce261832`). The v36 package was verified through the image's real executor entry point before launch. Twelve cells run at
  a time (tower 5, i265 3, r3500 2, r3800x 2); trainers confirmed running, no cell failures. 100 cells in total.
* **Worker claim fix landed** on zenmetrics master (`f26c61cb`) after two review rounds: a shutting-down worker no longer deletes a
  claim it doesn't own, and a renewal collision with the same worker no longer drops a live cell. The best-effort window between the
  ownership read and the delete remains and is documented. Fit images built after this pick it up; E28's image predates it.
* **SHIPPATH (D1 production fit + E30)** review found two admission/harvest ordering defects; round 11 is fixing them.
  **Palette round 2** (no public API change, semantic verifier, final E32 text) is under re-review.

## 14. Updates (2026-10-07 15:25 UTC)

* **E28 progress:** 18 of 100 cells done and harvested by 15:10 UTC, about 15–20 minutes per cell on 12 slots, no failures. A spot-checked
  cell ran the full registered budget (120 epochs, 50,000 pairs, selected epoch 119).
* **Palette features landed** on main (`0a8a7ef8`) after two review rounds: 42 research-only features (top N = 2–8 colour clusters, six
  shift signals each), no change to the supported public API, a verifier that checks the instrument's identity and column map before
  any join. **E32 is registered** on main (`benchmarks/e32_palette_registration_2026-10-07.md`): one primary arm (by_v2fy + 42 palette
  features), four-source LODO, seed-level paired statistics, control = E30's 40 nA3 cells when exact parity holds. E32 launches only
  after E30 completes. Measured limits stay in the registration: hue sign agrees on 97/168 synthetic cases, the features barely see
  high-frequency chroma loss, and the 32×32 sample lattice can miss systematic edits.
* **zenmetrics CI:** the fleet worker image workflow has failed since 2026-09-27 (deploy manifest missing the inherited lints table).
  A verified fix is committed; GitHub rejected the push with an internal error three times, so it will be retried.

## 15. Updates (2026-10-07 16:05 UTC)

* **E28:** 51 of 100 cells done at 15:51 UTC; every installed cell so far passes an independent budget audit (120 epochs, 50,000 pairs,
  epoch 119) — the running harvest doesn't check the budget itself, so the audit runs again before the verdict.
* **AIC-3 in E28 (for the owner):** E28 uses the same five-source exploratory design as E21–E27, so its s2o arm trains on AIC-3 when
  AIC-3 isn't the held-out source and its aic3 fold reads AIC-3 labels. D1 keeps AIC-3 out of the production model, and the §3d family
  rule calls the JPEG-AIC family "never a training input". E28 stays research-only: its models are never read on AIC-4 or SDR25 and
  never ship, and any recipe it supports must be re-checked on the four-source D1 route before production. If the owner wants
  exploratory runs to stop using AIC-3 as well, future registrations (E29 onward) will switch to the four-source design.
* **SHIPPATH round 11** fixed both review findings (admission now checks populations before any payload read; harvest binds the
  registered budget and admission identities, and smoke runs can no longer install as full cells). Its v39 image carries the reviewed
  worker claim fix. Under review.
* **UPIQ-380 ingested** for D3: 380 HDR pairs on 30 references (fit 330 / development 50 by reference hash), Rev5 by_v2fy features
  through the same HDR route as E26/E27, admitted from dataset metadata before the label file was opened; no other UPIQ data read.
  E31 (one human HDR rank arm) is drafted. Under review.

## 16. Updates (2026-10-07 16:25 UTC)

* **E28:** 74 of 100 cells done at 16:21 UTC; all installed cells pass the budget audit. Verdict expected within the hour.
* **SHIPPATH landed** in both repos after its second review (zensim `3f273924`, zenmetrics `e056defc`): the D1 four-source strict route,
  E30's 40 cells and the three production fits, with admission that checks populations before any payload read and a harvest that
  binds the registered budget, so a short smoke can never install as a full cell. Launch order: E30 after E28 completes; the production
  fit after E30's verdict, because E30 is the registered check of dropping the AIC-3 leg.
* **UPIQ-380:** the review confirmed the data (all 380 pairs re-extracted bit-identically, labels and split correct) and found three
  defects to fix before E31 can be registered: malformed extraction options fall back to the legacy label reader; the extractor checks
  feature-ID count, not identity; and the draft trained on AIC-3. E31 is being rewritten as a four-source design with E30's nA3 cells
  as its control.
* **CI snapshot fix** landed: a lock-check snapshot is now 0.5 GB instead of 18 GB.

## 17. Updates (2026-10-07 17:35 UTC)

* **E28 verdict: no.** Neither SSIM2-recipe arm passes and the by_v2fy control is retained. s2o (SSIMULACRA2's pooled
  within-dataset Kendall/Pearson terms added to the current mix): signed −0.0018 ± 0.0012, TID2013 −0.0054 (breaks the per-source
  guard), pooled KROCC −0.0027 and PLCC −0.0019 — no recipe signal. s2m (SSIMULACRA2's own data mix, without SafeSyn and the coverage
  leg): signed −0.0360, KADID −0.1416. So the answer to "can we use SSIMULACRA2's recipe?" is that we can, and on by_v2fy it doesn't
  help; dropping SafeSyn/coverage hurts a lot. E28b (F-scale KonFiG label) isn't worth running on this evidence. Records on main
  (`296b6702`); tower archive verified file by file.
* **E30 running** since 17:06 UTC (40 four-source cells, 12 at a time). It's report-only and not a gate on D1; the production fit
  (three seeds) launches once E30's report is recorded, per the D1 ledger.
* **UPIQ-380 landed and E31 registered** (`6aeddf43`) after two review rounds: four-source D1 design, E30's nA3 cells as control,
  one human HDR rank arm. E29 (HDR teacher consensus) is being amended to the same four-source design and implemented.
* **Release prep:** a production gate map and the one-time KADID TERMINAL read script (refuses without a final-model receipt and an
  explicit authorization; tested only on synthetic data) are in review.

## 18. Updates (2026-10-07 18:10 UTC)

* **E30: dropping the AIC-3 human leg costs nothing measurable.** Four-source cells versus the five-source control: signed
  +0.0003 ± 0.0015, every held-out source at or above +0.0002, W2 −0.0029 ± 0.0078. All 40 cells audited at the registered budget.
  E30 is report-only; D1 stands either way, and now it stands with evidence. Records on main (`ccbf0ec7`).
* **The D1 production fit is running** (`fitv2d1-20261007`, three full-data seeds, by_v2fy at Rev5 on KADID TRAIN+SELECT, TID2013,
  KonFiG TRAIN+VAL and CID22-A), launched 18:00 UTC. Each seed is densified, packed to f16 and calibrated inside the fit; harvest is
  bound to the registered budget and admission identities.
* **Next experiments:** E29 (HDR teacher consensus), E31 (UPIQ-380 human HDR leg) and E32 (palette colour shift) are being built
  as four-source designs that reuse E30's 40 cells as their control; none launches before its implementation is reviewed.
* **Release prep:** the KADID TERMINAL read script is in its second round (the first review found two ways it could open the
  wrong file); speed qualification passed its Rev4/Rev5 parity preflight and is moving to timing.

## 19. Updates (2026-10-07 18:40 UTC)

* **Production fit:** one of three seeds done at 18:20 UTC; the other two are training.
* **E29 / E31 / E32 control.** Both E31 and E32 stopped correctly at their registered parity check: E30's trainer refuses the new
  inputs (the UPIQ table identity, the palette IDs) before fitting, so no arm can run on E30's exact program and E30's cells can't be
  reused as-is. Using the registrations' own fallback, decided now before any arm fit: one extended program carries all three
  experiments' extensions, and one fresh matched control (E30's four-source recipe run under that program, 40 cells) serves all
  three. The extension must first be shown not to move the baseline (a control cell is expected bit-identical to E30's). Recorded in
  `benchmarks/e29_e31_e32_shared_control_decision_2026-10-07.md`.
* **Disk:** the shared NVMe is down to about 21 GB free. My E29 lane accounts for most of the last drop and is trimming to its
  pinned artifacts; the two largest consumers belong to other sessions (`~/tmp/downstream-0.9.30` 190 GB,
  `/mnt/v/output/imazen-26-compat` 51 GB).

## 20. Updates (2026-10-07 19:10 UTC)

* **The D1 production fit is complete.** Three seeds (0–2), all audited at the registered budget (120 epochs, 50,000 pairs, epoch
  119, the four D1 sources, 420 features), each densified, packed to f16 (~110 KB) and TRAIN-calibrated. Packed model hashes match
  their records; tower mirror verified (`/mnt/tower/output/zensim-production-d1-2026-10-07/`).
* **Owner decision needed before any evaluation gate:** which composition ships — one seed, or an ensemble. No rule for choosing
  among the three seeds was registered, and the scorecard requires freezing the model bytes before gates read evaluation data
  (choosing after looking would be adaptive selection). Label-free gates (serving surface, Rev5 parity, API/format, runtime) run
  on all three seeds meanwhile.
* **E29 incident:** an import in a new unit test ran an unguarded legacy CLI that read a legacy HDR VAL panel (teacher-scored,
  22,860 rows). No fit or decision used it; recorded in the exposure ledger; the module is now guarded.

## 21. Updates (2026-10-07 19:35 UTC)

* **Import hazard closed on main** (`a9a5ca0a`): after the E29 incident, the review found five more legacy scripts that read an
  evaluation or protected panel just by being imported — including one that reads the sealed hidden KADIS panel. They now refuse
  import, the incident module's CLI only runs as a script, and a tripwire test proves importing any of the six reads nothing.
* **E29** needs a second round (its Rust trainer path loaded HDR data before validating the row keys; a scoring sign issue).
  **E31/E32** trainer extensions are being written, each with a proof that the baseline path is unchanged. The shared fresh
  control waits for all three.
* **Production model:** label-free gates are running on all three seeds; evaluation gates wait for the owner to choose the final
  composition (one seed or an ensemble).
* **Speed qualification:** Rev4/Rev5 score parity recorded across tiers and threads (`ced5090f`); timing runs wait for a quiet box.

## 22. Updates (2026-10-07 20:05 UTC)

* **E31 and E32 trainer extensions keep the baseline bit-identical.** Each lane reran a full-budget control cell (KADID held out,
  seed 0) under its extended trainer; the model bytes match E30's cell exactly (after stripping run-specific metadata). The
  extensions are in cross-review; then they land with E29's and the combined package (shared control + arms) gets built.
* **Owner items now open:**
  1. Final production composition (one seed or an ensemble) — evaluation gates wait for it.
  2. E31 provenance: the producer of the legacy UPIQ HDR JOD file is unknown; E31's registration requires recovering it or an
     explicit owner decision to accept the gap before the E31 arm is fit. E29 and E32 don't depend on it.
  3. Whether exploratory runs should also drop AIC-3 (new registrations already use the four-source design).
  4. The `zensim-validate` partial-producer test expectation (Known Bugs).

## 23. Updates (2026-10-07 20:35 UTC)

* **Production label-free gates: not green.** All three packed models load and score through the public Rust surface, and their
  scores and features are bit-identical across ten dispatch permutations and WASM. But on the registered feature-only identity
  probe (an image against itself, band [97.5, 100]) **seed 0 passes (≈98.3) and seeds 1 and 2 fail (≈96.8, ≈97.2)** on all four
  probe sizes; the pixel path scores exactly 100 for all three. This reads no labels. It bears on the owner's composition choice.
* **Three integration tests fail on main** (two from the palette landing, which I verified without the integration tests; one on
  Rev5 revision handling), and serving the Rev5 bakes needs a process environment pin. Logged in Known Bugs; fix in progress.
* **E29/E31/E32:** all three reviews found that the native trainer entry reads data before fully admitting each group's role,
  weight and row keys. One integration lane is building a single shared admission owner for the combined program, then the
  combined package (shared control + E29 + E32; E31's arm waits for the owner's provenance decision).
* **KADID TERMINAL script:** the metadata race is closed (0 protected opens in 12,000 stressed attempts); round 4 fixes the label
  and scorer-binary reopen-by-name and hard-link aliases.

## 24. Updates (2026-10-07 21:05 UTC)

* **The three failing integration tests are stale contracts, not code defects.** The palette landing legitimately widened the
  research feature registry to 1,867 slots (production still returns 1,825, and palette stays refused at serving), and Rev5
  became a known revision by design. A proposed test patch keeps every assertion and makes each one check the current contract;
  it's under a strictness review before it lands.
* **Serving Rev5 needs `ZENSIM_FORMULA_REV=5` by design** — the Rev5 spec forbids mixing revisions in one process, so a product
  integration (imageflow and others) must set the pin when it adopts the new model. Not a bug; it belongs in the release notes.
* **E29 round 2 passes review.** It feeds the combined v40 program now being integrated with E31/E32.
* **Disk:** reclaimed ~69 GB of inactive build output from finished zensim lanes (binaries archived to tower first).

## 25. Updates (2026-10-07 21:35 UTC)

* **Main's zensim tests are green again** (`32b58f7b`): the three stale test contracts were updated after a strictness review
  found no weakening; the full zensim suite passed 900 tests (664 library, 227 integration across 38 targets, 9 doctests).
* In flight: the combined v40 program (E29 + E31 + E32 with one shared native admission check), the fourth round of the KADID
  TERMINAL read script, review of the label-free production gate report, and speed qualification (waiting for a quiet box).
* Still waiting on the owner: final production composition, E31 provenance, AIC-3 in exploratory runs, and the zensim-validate
  partial-producer test.

## 26. Owner decisions (2026-10-07 21:50 UTC)

* **Production ships seed 0 alone** — frozen in `benchmarks/production_composition_2026-10-07.json` before any evaluation-reading
  gate. The evaluation release gates now run on that model.
* **AIC-3 is fine for feature experiments** (feature-ceiling style); production stays on D1's four sources.
* **UPIQ provenance accepted** (file traced to zenmetrics' 2026-06-09 UPIQ-PU work from the official UPIQ release) → E31 can run
  with the shared control in the v40 package.
* i270 host keys accepted; stale workspaces and inactive build folders being removed.

## 27. Housekeeping per the owner (2026-10-07 22:20 UTC)

* **Stale workspaces removed:** 42 in zensim and 11 in zenmetrics (each snapshotted first; unpushed work kept on local
  `stale-ws/<name>` bookmarks). Four zensim workspaces whose working copies were stale were left in place so no unsnapshotted edit
  is lost: `gaddrinst`, `steercheck`, `zensim--gmsbank`, `zensim--gmsd-chroma`.
* **Inactive Cargo target folders deleted:** 171 folders, ~739 GB, under `~/work` and `~/tmp` (active lane workspaces and a
  folder another session was writing to were kept; top-level executables archived to
  `/mnt/tower/output/target-binaries-2026-10-07/` first). Free space on the shared NVMe: 726 GB.
* **i270:** new host keys accepted. The box is currently booted into Windows, so the fleet can't use it until it boots Ubuntu.

## 28. Updates (2026-10-08 00:35 UTC)

* **Landed** (`1cd8a888`): the production gate map, the KADID TERMINAL read harness after four review rounds (no open P1/P2;
  three minor P3s are being closed before any real read, which still needs the owner's explicit go), and the label-free
  production gate report. Full verification passed (zensim 900 tests, zensim-validate, Python incl. 43 terminal-harness tests,
  clippy, API).
* **v40** (shared control + E29 + E31 + E32, 200 full-budget cells) is ready and under independent review; it launches on the
  fleet after review.
* **Production evaluation gates on seed 0** are finishing; results next.
* **Speed qualification:** score parity done; timing waits for a quiet box, planned once the review and gate runs finish.

## 29. Updates (2026-10-08 01:00 UTC)

* **i270 is back in Ubuntu** (owner OK; no Windows session was logged in) and available to the fleet (3 slots) for the v40 launch.
* **The last failing zensim-validate test is resolved** (`5cd12253`, owner left the call to the coordinator): the by_v2fy
  projection root and the research-only palette family are listed as known partial producers, and the test requires the missing
  basic coverage to be reported for them.
* **Independent process review:** a Codex reviewer (herdr tab `process-review`) is auditing today's process and results against
  the evidence; its report lands as `~/tmp/zensim-paper/rev4/PROCESS_REVIEW.md`.

## 30. Release gates on the frozen seed-0 model (2026-10-08 01:25 UTC)

**Seed 0 does not qualify yet.** Fails: the full 38-image feature-identity gate (all 38 between 92.2 and 97.5 against the
[97.5, 100] band; pixel identity is exactly 100), standard-grid ties 0.076 (bar ≤ 0.05), and 7 of 135 steering cases (JPEG
4/8). Passes: dial calibration, negative tails, pixel identity, integrity-head compatibility. Blocked for missing inputs or
rules: human rank axes (no admitted independent human root), five codec floors, RD, targeting, integrity class, HDR, speed. The
earlier four-image identity pass was too narrow. A lane is now determining, for each failure, whether the harness or the model
is wrong; the model stays frozen and no bar changes.

An independent Codex review of the day's process and results found the headline experiment claims supported within their
scope, flagged that legacy composite error bars ignore fold covariance and that E30 shows "no detectable cost", not
equivalence, and listed process fixes (one shared admission invariant for every entry point, an end-to-end rehearsal before
any fleet launch, scoped landing receipts, a release matrix). v40 needs fixes before launch.

## 31. Speed qualification and gate adjudication (2026-10-08 11:30 UTC)

**SPEEDQ (frozen seed 0, `f803b74c`; landed `4f111a90`).** 192/192 size × tier × thread cells, 6144 paired rounds, on
dev (9950X3D). The release criterion "Rev5 at least as fast as Rev4 everywhere" **fails**: 94 cells faster, 80 slower,
18 inconclusive (pointwise paired 95% CIs). Slower: the scalar tier in every cell (35–75%); 1920×1080 at ≥4 threads on
the SIMD tiers (4–24%); v4 at 2048²/4096² with several threads; 64²/128² at 1–2 threads (8–14%, higher fixed cost:
v4x 1-thread α +795 µs vs −1381 µs). Rev5's per-pixel cost is lower (v4x 1 thread β 21.8 vs 32.6 ns/px). Peak RSS at
4096², v4x, 32 threads: Rev5 199,636 KiB, Rev4 246,856 KiB. Rev4/Rev5 score and all 420 consumed features are
bit-identical across 384 cells. Method changes during the run, both recorded in the report: ssimulacra2_rs timed only
in the first 63 segments (coordinator; it was ~58% of round time and has no tier dispatch); owner-approved rule
2026-10-08 keeps the first 32 clean rounds of at most 64 instead of discarding a segment with any flagged round (74
segments, 230 rounds excluded). Report: `benchmarks/rev5_speedq_2026-10-07.md`; full JSON on `/mnt/v` via pointer.
A bit-identical speed fix for the slow classes is in progress (REV5PERF lane).

**ADJUDICATE (seed-0 gate failures from §30).** C2 ties: a harness defect (matching NaN placeholders compared as
different); with the comparator fixed, the unchanged model passes the unchanged 0.05 bar (flat 0.0065, ladder 0.036);
patch under review. C5 identity: a real model property: on raw feature vectors of byte-identical pairs every candidate
scores below 97.5 because ten reference-only `pjnd_fragility` inputs are nonzero (the "all-zero features" note is wrong);
zeroing those ten inputs returns every model to the band. Served pixel identity is exactly 100. G-STEER: seven real
failures; the packed spline floor erases network responses on heavy-JPEG cases; the dense model still fails two.
Both C5 and G-STEER dispositions await the owner.

## 32. Landed 2026-10-08 13:30 UTC (main `079042fb`)

- **C2 tie check fix** (`5788652e`): matching NaN placeholders at the same slot now compare equal; finite epsilon stays
  1e-5. The frozen seed-0 model passes C2 unchanged (standard 28/4318 = 0.0065). The wrong "identity gives all-zero
  features" note is corrected.
- **Rev5 speed fixes** (runtime `ee9e5b55`): bit-identical (384/384 strict Rev4/Rev5 score and feature bits; 4,560/4,560
  extra reviewer cells). Re-timed worst cells: scalar 1920×1080 t8 +73% → +0.5% (inconclusive); v4 1920×1080 t8
  +27.5% → −8.5%; v4 4096² t8 +4.9% → −26.6%; v4 64² t2 +8.5% → +0.4% (inconclusive). The other 188 cells are being
  re-timed in a full rerun (SPEEDQ2); the criterion stays unestablished until then.
- **KADID TERMINAL read hardening** (RELEASEGATE5/6, tip `2ccd1ea5`): device-identity admission, replaced-ledger refusal,
  per-transaction locks, pinned acceptance inputs, reservation required before PASS. The read now requires the separate
  preparation/exposure filesystem layout written in the committed requirements; the current shared layout refuses.
- Landing checks on the combined tip: zensim 902 passed (27 ignored), bake_verdict 50, releasegate tests 66, lint, clippy.

## 33. V40 completed SDR results (2026-10-09)

All four V40 jobsets completed postfit: control 40, E29 80, E32 40, E31 40.
The 200 unique fits cover the shared control and four research arms, each
with four D1 source folds and ten paired seed indices. The fresh matched
V40 control is frozen; no E30 control substitution was made at assessment.

- **hc4: AS-GOOD.** All registered SDR guards pass.
- **hb4: NOT AS-GOOD.** KonFiG signed delta −0.008259966112729855 fails
  the each-source ≥ −0.005 guard.
- **palette: NOT AS-GOOD.** Signed delta −0.002956562407006813 fails
  the mean ≥ −0.002 guard; all four source deltas are negative. The
  registered improvement test records `adopt=false`.
- **uh4: AS-GOOD.** All registered SDR guards pass.

[Exact statistics and pins](v40_result_summary_2026-10-09.json) retain each
original decision object, including SE, n, per-source and W2 deltas, guard
outcomes, seed arrays, and E32's t/df/one-sided p. The [plain results record](v40_result_summary_2026-10-09.md)
explains the registered tolerances. The [artifact pointer](v40_results_2026-10-09.pointer.md)
records the complete decisions and tower mirror, with three randomly selected
files verified by SHA-256.

These are SDR results on already exposed four-source D1 design populations,
with each assessed source excluded from its corresponding fit. AS-GOOD means
registered SDR retention against the matched control; it does not establish
HDR performance, an untouched external-test result, or an improvement that
changes the frozen production composition. The uh4 fit used admitted UPIQ
TRAIN data, but no optional HDR, external or UPIQ report was opened. Those
reports still require separate exposure authorization and a freeze. KADID
TERMINAL remains unopened by this lane. The [exposure ledger](../docs/DATA_SPLITS.md#exposure-ledger--2026-10-09-completed-v40-sdr-assessments)
records the assessed roles and counts.

## 34. Landed 2026-10-09 and open owner decisions (main `cb5a6600`)

All results below are on the frozen seed-0 production model `f803b74c` unless stated.

- **Speed (SPEEDQ3, `462f7fe5`; then REV5PERF2 `c989a2d4`, REV5PERF3 `cb7e0777`, REV5PERF4 `cb5a6600`).** Full 192-cell
  rerun: 171 faster, 0 slower, 21 inconclusive versus Rev4 (pointwise paired 95% CIs); the owner accepted "0 slower" as
  meeting the speed requirement (2026-10-09). REV5PERF4 fixed multi-thread scaling: v4x 32 threads 1024² 14.06 → 7.55 ms,
  4096² 213.6 → 138.6 ms (serving A 8.80 / 177.9, B 8.20 / 176.4). Every change is bit-identical (384/384 strict parity;
  reviewer sweeps across tiers, thread counts and shapes). REV5PERF3 fixed a pre-existing panic on legally padded rows.
  Cost: REV5PERF4 raises peak RSS at 4096² from 201 MB to 550 MB at 16–32 threads (width × min(threads, 16), no cap).
- **Cost versus peers (COSTCMP, `713d2d73`).** v4x, 1 thread: 1 MP Rev5 22.4 ms, A 45.5, fast-ssim2 main (`09ec3e7c`,
  0.9.0) 60.7; 4096² 332 / 709 / 1458. Peak RSS at 4096², 1 thread: Rev5 191,664 KiB, A 505,364, fast-ssim2 main
  1,839,488. fast-ssim2 has no AVX-512 path. (Multi-thread comparisons predate REV5PERF4.)
- **Steering (STEERFIX, `ed8eabbc`).** Of the seven G-STEER failures, four were a code defect (the packed spline floor erased
  the signal; steering now uses pre-calibration sensitivities below the floor, and Rev5 defaults to neighbour replay) and three
  are model limits (broad-206 b32/b64, dog-256-20). Served scores bit-identical (55,515/55,515); the served gate stays 128/135
  because floor-tied scores cannot pass the rank bars.
- **Near-identity (NEARID, `f1b82519`).** No changed image scores 98 or above for seed 0 (max 97.73), B (96.23) or A
  (97.39); one changed pixel scores 88.69–97.67 for seed 0. Cause: the ten reference-only `pjnd_fragility` inputs enter
  additively; neutralizing them puts all 24 identities at 99.81. The owner's design direction (2026-10-09): internal 0 means
  no difference, reference-only inputs only scale differences, no anchors. Proposed E33 (registration pending owner approval):
  `--nonneg-distance` at Rev5 without those ten inputs; a gated head only if needed; smooth monotone tail instead of the floor.
- **zenpredict graphs (imazen/zenanalyze PR #89, Opus lane).** ZNPR v4 static op graphs; 0 mismatches against the old runtime
  on 260 fixtures, the production bake and 504 Rev4 bakes; zensim suite passes; review MERGE-READY pending the owner's choice
  of how `layers()` behaves on graph files. zenanalyze main also took the magetypes 0.9.30 migration (`a10e5581`, bit-identical).

Open owner decisions: (1) zenpredict `layers()` option a/b/c; (2) V40 HDR-side reports (HDR VAL exposure); (3) E33
registration; (4) zenanalyze scalar-tier feature bug and a batch of doc corrections (zenanalyze and zenpredict); (5) a byte
budget for REV5PERF4's memory growth.

## 35. Owner-authorized V40 HDR validation (2026-10-09)

The owner approved the registered HDR-side reports (verbatim “yes, to all 3
still open”); the exact E29 population and six-object exposure freeze were
committed before any validation payload read. E29 completed on the original
3,900-row / 300-reference hdr_v3mix VAL bank against the frozen V40 control.
**hc4 passes the registered both-teacher HDR improvement/retention gates
and SDR guards; E29 selects hc4.** hb4 fails pooled CVVDP improvement and
the previously recorded KonFiG SDR guard. [Exact decisions/statistics](v40_hdr_result_summary_2026-10-09.json),
[descriptive Borda panels](v40_hdr_borda_report_2026-10-09.json), and
[complete evidence/tower pins](v40_hdr_results_2026-10-09.pointer.md) retain
the registered endpoints and ten-seed equal-four-fold reduction.

This is synthetic HDR teacher agreement, not independent human HDR evidence
or a production composition change. E31 remains unrun: its packet UPIQ
fit/development and registered external-video routes exceed the approval;
packet HDR mode supports E29 only. Clarification was requested. No UPIQ
development, external, terminal, AIC-family, T0 or sealed data was opened.
See the [exposure ledger](../docs/DATA_SPLITS.md#exposure-ledger--2026-10-09-owner-authorized-v40-hdr-validation-reports).

## 36. E31 expanded owner-authorized HDR reports (2026-10-09)

The owner approved E31's registered additional reads (verbatim “approved,
all of it”). Approval and an exact seven-object freeze were committed
before payload reads. UPIQ TRAIN fit (330 / 26 references) and TRAIN
development (50 / 4) now have complete report-only panels for forty uh4
models and forty matched controls each: pooled/study/reference signed ranks
and raw scatter geometry. No new statistic, fit, selection or adoption rule.
[Per-cell readout](v40_e31_hdr_results_2026-10-09.md),
[exact summary](v40_e31_hdr_result_summary_2026-10-09.json),
[evidence/tower pins](v40_e31_hdr_results_2026-10-09.pointer.md).
All 178 archived files match the tower mirror. These are training/development
reports, not independent human HDR qualification or a production change.

HDR-VDC/AVT remain blocked: retained Rev2/944 tables are incompatible with
Rev5/420, and the archived drivers deleted decoded frames. The coordinator
confirmed no newer cache/archive is known and prohibited re-extraction
without fresh owner sign-off. No external payloads/labels were opened.
Terminal, AIC-family, T0 and sealed populations remain unopened. The
four-source SDR result and the previously registered E29 outcome are unchanged.

## 37. E33 registered results (2026-10-10)

E33 ran exactly as registered (amendment A1 recorded before any fit): fresh 40-cell control, Arm A (410
differences, no reference-only inputs) and Arm C (A plus 410 same-cell difference × fragility products),
40 LODO cells each, plus full-data seeds 0–2 of A and C. All 126 cells completed on the home fleet with no
wall-cap stop and no poisoned cell, and the program's own harvest owner verified every one.

- **E21 (9.1): A and C are both AS-GOOD** against the fresh control. A: mean Δ −0.00089 (SE 0.00074);
  C: mean Δ +0.00374 (SE 0.00126), every source positive. C beats A on the registered improvement test
  (mean +0.00463, t = 3.98, df 9, one-sided p = 0.0016).
- **Label-free gates (9.2), full-data seed 0:** both candidates pass N1, N2 and N3 (one-pixel rungs
  ≥ 99.39 for A and ≥ 99.64 for C, where production seed 0 scores 88.69–97.67), C2 ties, C5 identity
  (exactly 100.0 on all 620 source × tier rows) and the output stage (K1–K3 at pack; K4 0 rows with raw ≤ x_floor on every registered population (calibration 12,163,
  negative-tail 2,000, identity 38 plus the 620-row proof, standard 4,424, ladder 9,593, NEARID 648 and 55,515
  G-STEER forwards); C1/C3/C4/C6/G-DIAL). G-STEER: A 129/135 passes; **C 127/135 fails** the ≥ 128 bar.
- **Runtime (9.3):** A is not slower in any cell. **C is slower** at 64² (+26–27%) and 256² (+6–7%) on
  v4x and v3, so it fails the guard.
- **Verdict (9.4.3): adopt A** as the next production candidate. That means full qualification, the KADID
  TERMINAL decision and serving review; E33 qualifies nothing. C's E21 advantage does not apply because C
  is ineligible. Owner, verbatim 2026-10-10: "i think it might matter, dont write off c". The registered
  verdict stands; C stays under investigation (its steering failures are diagnosed with the STEERFIX method,
  no threshold change). Report-only seeds 1–2 G-STEER: A 124 and 128, C 127 and 131.

[Result summary](e33_result_summary_2026-10-10.md), [exact JSON](e33_result_summary_2026-10-10.json),
[artifact pointer](e33_results_2026-10-10.pointer.md). Exposure: [DATA_SPLITS](../docs/DATA_SPLITS.md)
(E33 exposure receipt). The populations are the same already-exposed D1 design sources V40 read; no
untouched-test, HDR or external claim follows.
