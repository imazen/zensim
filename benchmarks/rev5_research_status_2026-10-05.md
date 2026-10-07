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
