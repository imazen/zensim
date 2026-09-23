# Rev4 E2a — which display model is behind every CVVDP value we used (2026-09-23)

Lane E2a of `docs/REV4_EXPERIMENTS_2026-09-23.md` (answers the precondition of Q2). Analysis of stored
files, manifests, logs and code history only. **No scoring, no training, no human labels read** (hence
no prereg). Worklog with every command and hash: `benchmarks/rev4_e2a_WORKLOG.md`. Machine-readable
twin: `benchmarks/rev4_e2a_cvvdp_display_2026-09-23.json`. Dataset-paper quotes (Deliverable 3):
`benchmarks/rev4_e2a_viewing_conditions_p{1,2}_2026-09-23.md`.

## Headline

1. **Every CVVDP value that has ever been a zensim training target at SDR was computed at
   `standard_4k` (75.40 ppd, 200 cd/m²), with two exceptions: the May-2026 phone bakes.** The
   `zensim-b-phone` / `ZensimProfile::A_Phone` bake was trained on CVVDP at the imazen preset
   `modern_oled_phone_indoor` (109.97 ppd, 400 cd/m², *not* parity-checked against pycvvdp), and a
   first attempt used `iphone_14_pro` (159.61 ppd). So the premise "CVVDP was only ever a teacher at
   one display" is almost true, but not quite.
2. **The falsified CVVDP-teacher recipes (V_22-CVVDP 2026-05-17, V12/V13/V14 2026-05-20, V41 2026-05-25,
   v47 2026-05-27) all used `standard_4k`,** by construction: until zenmetrics `088f4bf5`
   (2026-05-25T23:50-06:00) the CLI hard-coded `STANDARD_4K` photometry *and* geometry and had no
   display flag. Those records do not state a display; the code path does.
3. **Every HDR CVVDP teacher value before 2026-08-06 (Profile B's `hdr_v3mix` cid head, `hdr_kadis_mix`)
   used a 1000-cd/m² linear HDR display with 4K-desktop geometry (75.40 ppd),** not a display matched to
   any human study. Later hdrgrid values use a measured per-reference peak (medium confidence).
4. **Every board `peer_cvvdp` value is `standard_4k`,** including AIC-3/AIC-4/SDR25, whose organisers
   configure CVVDP as `standard_fhd` (37.84 ppd). The only non-4K board row is the unmerged
   `peer_cvvdp_aicfhd` (AIC-4 only). AIC2026's CVVDP level placement is also `standard_fhd`
   (quoted from its paper).
5. **Column names cannot certify a display.** Measured: the phone, iPhone and 4K runs of the same 8
   pairs all write `cvvdp_imazen_v0_0_1`, with values that differ by up to 1.41 JOD (iPhone vs 4K)
   (`/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25/sanity8_*.tsv`). CPU `cvvdp` on origin/master has no
   display input at all; the `_<display>` column suffix exists only in unmerged `d71922bd`.
6. **No human dataset we use was collected at 75 ppd.** Where a paper states geometry, it is 24–60 ppd
   (KonJND 24.3, LIVE 26.8–33.5, CID22 nominal 46.9, UPIQ SDR 51 stated, Korshunov 60.3, Narwaria 56.5).
   KADID, TID, CSIQ, KonFiG, AIC-3, AIC-4 and SDR25 give no usable viewing distance.

## Deliverable 1 — inventory: CVVDP value → display → evidence

Confidence: **high** = code path verified at the producing commit (or its bounded commit range) plus
a matching command/manifest; **medium** = code default with the producing build unrecorded;
**unknown** = not established. "Display from column name" was never used as evidence.

| id | used as | values (column / file) | display | evidence | confidence | producing commit |
|---|---|---|---|---|---|---|
| T1 | teacher/training substrate | May-2026 fleet CVVDP backfill `cvvdp_imazen_v0_0_1` (s3://zentrain/cvvdp-backfill-2026-05-15-half/cvvdp_imazen/, 711k rows; local cache /mnt/v/zen/zensim-training/2026-05-15-cvvdp-r2/, 11,695 chunk parquets; -> canonical-2026-05-21/scores/cvvdp_imazen_v0_0_1.parquet, cvvdp_iwssim_LARGE*, 2026-05-17-cvvdp*/) | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | zenmetrics 088f4bf5^ (all builds before 2026-05-25T23:50-06:00): CLI cvvdp hard-coded CvvdpParams::PLACEHOLDER (display STANDARD_4K) + DisplayGeometry::STANDARD_4K ppd; no display flag existed; image tags 0.6.4-cvvdp-* (zenmetrics commits 72485d30/0a0118c0, 2026-05-15); parquet schema carries no display field | high | pre-088f4bf5 (image 0.6.4-cvvdp-libcuda2; exact commit not recorded) |
| T2 | teacher (V_22-CVVDP, V_22-mix, v47, v13/v14, V41, cvvdp_target_probe) | `cvvdp_score` / `cvvdp_log_norm` / `mix_cv*_iw*` in canonical-2026-05-21 train/{safesyn,kadid,tid}.parquet and 2026-05-17-cvvdp/*_features_iwssim_cvvdp_372col.parquet (local cvvdp-gpu backfill 2026-05-16/17: safesyn 196,086, KADID 10,125, TID 3,000) | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | zenmetrics 088f4bf5^ (all builds before 2026-05-25T23:50-06:00): CLI cvvdp hard-coded CvvdpParams::PLACEHOLDER (display STANDARD_4K) + DisplayGeometry::STANDARD_4K ppd; no display flag existed; v0_22_cvvdp_methodology_2026-05-17.md:66-78 (backfill steps, cvvdp-gpu CLI fix bd9ba04b) | high | local build after bd9ba04b, before 088f4bf5 |
| T3 | teacher (cid22_train rows) | canonical-2026-05-21/train/cid22_train.parquet `cvvdp_score`/`cvvdp_log_norm`/mix (task #7, 2026-05-24, 17,611 pairs from the 201 train-only CID22 refs) | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | zenmetrics 088f4bf5^ (all builds before 2026-05-25T23:50-06:00): CLI cvvdp hard-coded CvvdpParams::PLACEHOLDER (display STANDARD_4K) + DisplayGeometry::STANDARD_4K ppd; no display flag existed; manifest entry 3 schema_note dates the backfill 2026-05-24 (before 088f4bf5 on 05-25 23:50) | high | pre-088f4bf5 |
| T4 | teacher (V12 anchors / cross-codec eq; v12 falsification) | `score_cvvdp_imazen_v0_0_1` in 2026-05-20-v11-substrate/multi_codec_372col_full.parquet -> 2026-05-20-v12-cvvdp-substrate/{anchors_cvvdp_372col*,cross_codec_equivalence_cvvdp_372col}.parquet | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | zenmetrics 088f4bf5^ (all builds before 2026-05-25T23:50-06:00): CLI cvvdp hard-coded CvvdpParams::PLACEHOLDER (display STANDARD_4K) + DisplayGeometry::STANDARD_4K ppd; no display flag existed; column produced by the May 18-19 omni sweeps (DATA_PROVENANCE cvvdp-v15rc-2026-05-18 / omni-multi-codec-2026-05-19, image v22) | high | pre-088f4bf5 |
| T5 | teacher (zensim-b-phone / `ZensimProfile::A_Phone` bake zensim_b_phone_oled_2026-05-26.bin) | /mnt/v/output/zensim/iphone14-cvvdp-2026-05-25/{kadid,tid}_cvvdp_phone.tsv (column `cvvdp_imazen_v0_0_1`) -> {kadid,tid}_phone_cvvdptgt*.parquet + modern_oled_anchor.parquet | modern_oled_phone_indoor (109.97 ppd; 400 cd/m2, black 0.0005, refl 0.3979; sRGB) — imazen preset, NOT parity-checked vs pycvvdp | zensim_b_phone_oled_methodology_2026-05-26.md:65 (command `zenmetrics batch --metric cvvdp --display-model modern_oled_phone_indoor`, features gpu-cvvdp); at 088f4bf5 `--metric cvvdp` was the GPU kind and honoured the flag; sanity8_{std4k,ip14,phone}.tsv values differ per display under the SAME column name | medium-high (display known only from the file name + methodology doc; column name identical to 4K runs) | 088f4bf5 or later, before 41047c97 (2026-06-30) |
| T6 | teacher (iPhone-14 dial parquets, first zensim-b-phone attempt) | /mnt/v/output/zensim/iphone14-cvvdp-2026-05-25/{kadid,tid}_cvvdp_iphone14.tsv -> *_iphone14_cvvdptgt.parquet | iphone_14_pro (159.61 ppd; 1025 cd/m2, black 0.0004, refl 0.3979; sRGB) — parity-checked | scripts/v_next/build_iphone14_cvvdp_dial_parquets.py:4-5 docstring (`--display-model iphone_14_pro`); file name only; same column name | medium | 088f4bf5 or later |
| T7 | teacher: Profile B cid head (80%) and w8_hdrmix_cvmix; hdr_v3mix | /mnt/v/output/zenmetrics/datagen-2026-06-23-hdr/sidecars/zenjxl/cvvdp.parquet (`cvvdp_imazen_v0_0_1`, 7,980 rows) -> hdr_zenjxl_v3mix_{train,val}digits_2026-07-03.parquet `score_cvvdp` -> cvvdp-mix `0.5*clip(ssim2/100)+0.5*clip((JOD-6)/4)` | STANDARD_HDR_LINEAR photometry with y_peak=1000 (linear EOTF, contrast 1e6 -> black 0.001, E_ambient 10 -> refl 0.0159) + standard_4k geometry (75.40 ppd) | log/zenjxl.score.log: `zenmetrics score-pairs --metric cvvdp --hdr --hdr-transfer pu-rescale ... --gpu-runtime cuda` (2026-06-23, before 41047c97 so `cvvdp` = GPU kind); code at 0619afcc main.rs:1222-1234 builds DisplayTarget::hdr(HDR_DISPLAY_PEAK_NITS=1000); HDR score-pairs --hdr cvvdp -> DisplayTarget::hdr(HDR_DISPLAY_PEAK_NITS) / HdrScorer(HDR_PEAK_NITS); both constants = 1000.0 at a7fb1f35, ee6f2f0d, 0619afcc, f3832a55 and 6471f4d7^ (git grep); geometry STANDARD_4K (cvvdp_gpu.rs:116) | medium-high (binary commit not recorded; every candidate commit 06-03..08-06 uses the same 1000-nit target) | build between a7fb1f35 and 6471f4d7 (probably 0619afcc) |
| T8 | teacher: hdr_kadis_mix_*digits_2026-07-13 (cvvdp-mix HDR target) | kadis-hdr-2026-07-13 sidecars cvvdp.parquet (11,400 rows; s3://codec-corpus/kadis-hdr-2026-07-13/sidecars-merged/) | STANDARD_HDR_LINEAR photometry with y_peak=1000 (linear EOTF, contrast 1e6 -> black 0.001, E_ambient 10 -> refl 0.0159) + standard_4k geometry (75.40 ppd) | DATA_PROVENANCE: zenmetrics master f3832a55, `score-pairs --hdr --hdr-transfer pu-rescale` x {...cvvdp...}; after 41047c97 `cvvdp` = CPU -> umbrella HdrScorer(HDR_PEAK_NITS=1000) STANDARD_HDR_LINEAR; HDR score-pairs --hdr cvvdp -> DisplayTarget::hdr(HDR_DISPLAY_PEAK_NITS) / HdrScorer(HDR_PEAK_NITS); both constants = 1000.0 at a7fb1f35, ee6f2f0d, 0619afcc, f3832a55 and 6471f4d7^ (git grep); geometry STANDARD_4K (cvvdp_gpu.rs:116) | medium-high | f3832a55 |
| T9 | teacher: hdrgrid372/hdrgrid944 legs (Appendix-Q cvvdp-mix target) | /mnt/v/output/hdrgrid-2026-08-06/harvest-2026-08-26/scores.parquet `cvvdp_cpu_imazen_v0_1_0` (49,715 rows) | STANDARD_HDR_LINEAR photometry with y_peak = MEASURED reference content peak (per ref), standard_4k geometry (75.40 ppd) | hdrgrid _MANIFEST.json runs.score_cpu 'cvvdp DEFERRED per appendix AA measured-peak hold' and runs.diffmap 'must carry measured-ref-peak semantics, not static-1000'; measured peak since 6471f4d7 (2026-08-06); harvest build e461f96d is after 6471f4d7. The image/commit of the wave that actually wrote cvvdp is not in the manifest (pinned images 9093cc23/a7cf7df9 predate 6471f4d7) | medium | unknown (post-6471f4d7 wave inferred from the manifest hold) |
| T10 | teacher: KADIS-700k weak labels (`cvvdp/10 primary`, DATA_SPLITS §3); kadis_cvvdp_{train,val}.parquet; multicodec_profile_probe_2026-06-30 `cvvdp_w1` | kadis700k_canonical_gpu_2026-07-01.parquet `score_cvvdp_cpu_imazen_v0_1_0` (700k) | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | CPU `cvvdp` (column cvvdp_cpu_imazen_*) on every origin/master commit: umbrella run_metric -> cpu_dispatch.rs CvvdpParams::default()=PLACEHOLDER=STANDARD_4K, Cvvdp::new -> DisplayGeometry::STANDARD_4K (pipeline.rs:145-146); no display input until unmerged d71922bd; README_gpu.md:94-95 config METRICS=...,cvvdp,... (no display setting exists for CPU cvvdp) | high | image zenmetrics-sweep:kadis-persist (commit not recorded) |
| T11 | teacher: sdrcodec_pl203 cvvdp-mix; 2026-06-24 unified store | /mnt/v/zen/zensim-training/2026-06-24/unified/* `cvvdp_imazen_v0_0_1` (datagen-2026-06-23 SDR; build_commit 1c3760e0) | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | zenmetrics jobexec/compare/sweep SDR scoring -> run_metric umbrella path, fixed STANDARD_4K (main.rs comment at 2601-2606; jobexec.rs:147-154); SPLIT runs use score-pairs per metric with no display flag recorded (zenmetrics docs/SCORING_DATA_2026-06-24.md) | medium-high | 1c3760e0 (manifest) |
| T12 | training store (not used as a teacher in any record found) | fill4 sidecar `score_cvvdp` (s3://zentrain/fill4-6codec-2026-07-01/canonical/fill4metrics_sidecar_2026-07-02.parquet, 4.2M rows) and avifgen harvest-2026-08-26 scores.parquet cvvdp | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | zenmetrics jobexec/compare/sweep SDR scoring -> run_metric umbrella path, fixed STANDARD_4K (main.rs comment at 2601-2606; jobexec.rs:147-154) | medium-high | not recorded in the entries read |
| E1 | board peer row `peer_cvvdp` (cid22/kadid/tid/aic3/konjnd/csiq/live/aic4/sdr25) | refmetrics/{cid22,aic3_*_heldout,konjnd_*_heldout,csiq,live,aic4,sdr25}_cvvdp.tsv (`cvvdp_cpu_imazen_v0_1_0`) + {kadid,tid}_cvvdp_gpu.tsv (`cvvdp_imazen_v0_0_1`) | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | CPU `cvvdp` (column cvvdp_cpu_imazen_*) on every origin/master commit: umbrella run_metric -> cpu_dispatch.rs CvvdpParams::default()=PLACEHOLDER=STANDARD_4K, Cvvdp::new -> DisplayGeometry::STANDARD_4K (pipeline.rs:145-146); no display input until unmerged d71922bd; kadid/tid: run_gpu_metrics.sh passes no --display-model -> cvvdp-gpu batch/score-pairs without --display-model -> DisplayTarget::default() = STANDARD_4K (metrics/cvvdp_gpu.rs:65-70); empirical: aic4_cvvdp.tsv reproduces pycvvdp standard_4k (SROCC 0.8906 = 0.8906, CVVDPFIX_DONE.md) | high | local builds 2026-07-15 (CPU cid22, GPU kadid/tid), 07-18 (aic3/konjnd), 08-27 (csiq/live/aic4/sdr25) |
| E2 | board peer row `peer_cvvdp_aicfhd` (unmerged, AIC-4 only) | refmetrics/aic4_cvvdp_standard_fhd.tsv (`cvvdp_cpu_imazen_v0_1_0_standard_fhd`) | standard_fhd (37.84 ppd; 200 cd/m2, black 0.2, refl 0.3979; sRGB) | .meta sidecar: command `zenmetrics batch --metric cvvdp --display-model standard_fhd`, code d71922bd (unmerged); max \|d\| 0.00027 JOD vs organisers | high | d71922bd (../zenmetrics--cvvdpfix) |
| E3 | per_pair.kadis.cvvdp in candidate fulleval rows (reference column) | KADIS-720/700k metric parquet cvvdp | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | docs/FULL_EVAL.md:254-263; source = T10 | high | as T10 |
| E4 | UPIQ-HDR evaluation rows (cvvdp@1000/4000/6000/10k) | /mnt/v/output/zenmetrics/upiq-pu/panel_cvvdp_gpu{,_4000,_6000,_10k}.tsv | STANDARD_HDR_LINEAR photometry, y_peak 1000 / 4000 / 6000 / 10000 per file, standard_4k geometry (75.40 ppd) — NOT UPIQ's 1920x1080 SIM2 viewing geometry | zenmetrics benchmarks/pu_integrated_upiq_2026-06-09.md Addenda 4-5; examples/upiq_hdr_score.rs peak arg -> HdrScorer::new; Cvvdp::new -> STANDARD_4K geometry | medium-high (geometry from code default, not recorded in the run) | not recorded |
| E5 | corruption gate peer column (2026-05-28) | /mnt/v/output/zensim/corruption_gate_results/corruption_multimetric_2026-05-28.tsv cvvdp (`cvvdp_imazen_v0_0_1`) | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | corruption_corpus_multimetric_2026-05-28.md:29 `zenmetrics compare` -> compare.rs:122 run_metric (fixed STANDARD_4K) | high | not recorded (post-088f4bf5 build) |
| E6 | dial/corruption refmetrics (2026-08-27) | refmetrics/{dialgrid,corruption}_cvvdp.tsv (`cvvdp_cpu_imazen_v0_1_0`) | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | CPU `cvvdp` (column cvvdp_cpu_imazen_*) on every origin/master commit: umbrella run_metric -> cpu_dispatch.rs CvvdpParams::default()=PLACEHOLDER=STANDARD_4K, Cvvdp::new -> DisplayGeometry::STANDARD_4K (pipeline.rs:145-146); no display input until unmerged d71922bd | high | local build 2026-08-27 |
| E7 | AIC-3 CVVDP-feature spike input + comparator (2026-05-25) | 600 AIC-3 pairs scored `zenmetrics batch --metric cvvdp --gpu-runtime cuda` | standard_4k (75.40 ppd; 200 cd/m2 peak, black 0.2, refl 0.3979; sRGB EOTF) | aic3_cvvdp_feature_spike_2026-05-25.md:10; no --display-model in the command; batch path pre-088f4bf5 or default target after | high | pre- or at 088f4bf5 |
| E8 | AIC-3 organisers' CVVDP (PTC 5-image subset, '0.96') | AIC-3 dataset-provided CVVDP column | unknown (not established in this lane; see Deliverable 3 for the AIC CTC display) | aic3_cvvdp_feature_spike_2026-05-25.md:22-24 names the column but not its display | unknown | external |
| E9 | AIC-4 organisers' CVVDP (JPEG-AIC_metric_scores.csv, 0.9609) | /mnt/v/repos/iqa-tools/jpeg-aic__JPEG-AIC-4-datasets/JPEG-AIC_metric_scores.csv CVVDP | standard_fhd (ColorVideoVDP 0.4.2, `-d standard_fhd`) | AIC-4 CTC v2.0 wg1n101246 §4 (per CVVDPFIX_DONE.md); reproduced to 0.00005 JOD by pycvvdp 0.4.2 standard_fhd | high | external |
| E10 | AIC2026 level placement + JND_CVVDP column (no human labels) | /mnt/v/datasets/aic2026/metrics_{fullres,cropped}.csv CVVDP / JND_CVVDP | standard_fhd | AIC2026 paper (arXiv:2607.22783) converted text /mnt/v/input/papers/ce/ceb244cf…md line 150: 'CVVDP scores were computed using the standard Full HD SDR display configuration (cvvdp -d standard_fhd)' | high | external |
| E11 | jxl-encoder diffmap RD loop + cvvdp judge (L12-04, 2026-05-27) | jxl-encoder examples/zensim_diffmap_rd.rs `--metric cvvdp` loop/judge | standard_4k (code default) | jxl-encoder src/api.rs:2518/2662 default DisplayModel::STANDARD_4K 'used when no' display is configured (current HEAD); 2026-05-27 build not verified | medium | not recorded |
| E12 | pycvvdp reference worker column `cvvdp_pycvvdp_v054` | scripts/sweep/pycvvdp_worker.py outputs (R2 not audited; none in the 2026-06-24 corpus per CVVDPFIX_DONE.md) | standard_4k by default (`--display-name` default); argument order swapped (predict(ref,dist)) before unmerged 27c69beb | origin/master pycvvdp_worker.py:40,71,193 | high for the code default; whether any stored values exist is unknown | 3c99f833 onward |
| E13 | cvvdp_gpu_mode_probe_2026-08-05.tsv | benchmarks/cvvdp_gpu_mode_probe_2026-08-05.tsv | n/a — performance probe (wall time / GPU memory), no CVVDP values stored | file header: metric scale rc wall_s peak_gpu_util_pct ... | high | n/a |

Code-path facts the table relies on (all `git show origin/master:` or `git show <commit>:` in zenmetrics;
details and line numbers in the worklog, Step 2):
- before `088f4bf5`: `CvvdpParams::PLACEHOLDER` (display `STANDARD_4K`) + `DisplayGeometry::STANDARD_4K`
  ppd, no flag (verified at `0422c968` and `088f4bf5^`);
- `088f4bf5`..`3a7dda0d`/`41047c97` (2026-05-25..06-25/30): `--metric cvvdp` is the GPU kind and
  honours `--display-model`; default `standard_4k`;
- from `3a7dda0d`/`41047c97` (2026-06-25/30, "cvvdp/iwssim score on native CPU by default" / "unsuffixed=CPU, add cvvdp-gpu"): unsuffixed `cvvdp` = CPU port, no display input; `cvvdp-gpu` honours the
  flag but writes `cvvdp_imazen_v0_0_1` for any display;
- `run_metric` (jobexec, compare, fleet sweeps): fixed `STANDARD_4K`;
- HDR: `STANDARD_HDR_LINEAR` photometry, `y_peak` = 1000 (constant `HDR_DISPLAY_PEAK_NITS`/`HDR_PEAK_NITS`
  at every commit a7fb1f35..6471f4d7^) or, from `6471f4d7` (2026-08-06), the measured reference peak;
  geometry always `STANDARD_4K`.

Not found: any zensim training target or board row computed at a display matched to a human dataset.

## Deliverable 2 — display catalogue (our port and pycvvdp)

ppd, black and reflected luminance recomputed from pycvvdp 0.5.4's `display_model.py` formulas
(`/var/tmp/rev4-e2a/catalogue.py`; the pycvvdp venv itself no longer runs, see worklog). "conf." =
listed in `crates/cvvdp-conformance/src/displays.rs` on origin/master (pass/fail per cell is in zenmetrics
`CVVDP_CONFORMANCE.md`; the unmerged cvvdpfix run reports cpu 403/403, gpu 400/403 within 1e-3 over 13
displays). "conf. (cvvdpfix)" = the unmerged 13-display matrix at pycvvdp 0.5.7.

| preset | in pycvvdp 0.5.4 | ppd | distance m | diag in | resolution | peak cd/m² | black cd/m² | refl cd/m² | EOTF | E_amb lux | conf. | conf. (cvvdpfix) |
|---|---|--:|--:|--:|---|--:|--:|--:|---|--:|---|---|
| `standard_4k` | yes | 75.40 | 0.7472 | 30 | 3840x2160 | 200 | 0.2 | 0.3979 | sRGB | 250 | yes | yes |
| `standard_hdr_pq` | yes | 75.40 | 0.7472 | 30 | 3840x2160 | 1500 | 0.0015 | 0.01592 | PQ | 10 | yes | yes |
| `standard_hdr_hlg` | yes | 75.40 | 0.7472 | 30 | 3840x2160 | 1500 | 0.0015 | 0.01592 | HLG | 10 | yes | yes |
| `standard_hdr_linear` | yes | 75.40 | 0.7472 | 30 | 3840x2160 | 1500 | 0.0015 | 0.01592 | linear | 10 | no | no |
| `standard_hdr_linear_dark` | yes | 75.40 | 0.7472 | 30 | 3840x2160 | 1500 | 0.0015 | 0 | linear | 0 | yes | yes |
| `standard_hdr_linear_zoom` | yes | 25.23 | 0.2500 | 30 | 3840x2160 | 10000 | 0.01 | 0.01592 | linear | 10 | no | no |
| `standard_fhd` | yes | 37.84 | 0.6000 | 24 | 1920x1080 | 200 | 0.2 | 0.3979 | sRGB | 250 | yes | yes |
| `standard_hmd` | yes | 13.15 | 3.0000 | None | 1440x1600 | 100 | 0.1 | 0 | sRGB | 0 | no | no |
| `standard_phone` | yes | 120.56 | 0.4000 | 6 | 2400x1080 | 500 | 0.05 | 0.3979 | sRGB | 250 | yes | yes |
| `sdr_4k_30` | yes | 60.55 | 0.6000 | 30 | 3840x2160 | 100 | 0.1 | 0.3979 | sRGB | 250 | yes | yes |
| `sdr_fhd_24` | yes | 37.84 | 0.6000 | 24 | 1920x1080 | 100 | 0.1 | 0.3979 | sRGB | 250 | no | no |
| `htc_vive_pro` | yes | 13.15 | 3.0000 | None | 1440x1600 | 133.3 | 0.1 | 0 | sRGB | 0 | yes | yes |
| `iphone_12_pro` | yes | 159.61 | 0.5080 | 6.1 | 2532x1170 | 825 | 0.0004 | 0.3979 | sRGB | 250 | no | no |
| `iphone_14_pro` | yes | 159.61 | 0.5080 | 6.1 | 2532x1170 | 1025 | 0.0004 | 0.3979 | sRGB | 250 | yes | yes |
| `iphone_14_pro_vert` | yes | 159.61 | 0.5080 | 6.1 | 1170x2532 | 1025 | 0.0004 | 0.3979 | sRGB | 250 | no | no |
| `iphone_14_pro_hdr` | yes | 159.61 | 0.5080 | 6.1 | 2532x1170 | 1590 | 0.0004 | 0.01592 | HLG | 10 | no | no |
| `iphone_14_pro_hdr_vert` | yes | 159.61 | 0.5080 | 6.1 | 1170x2532 | 1590 | 0.0004 | 0.01592 | HLG | 10 | no | no |
| `ipad_pro_12_9` | yes | 92.39 | 0.5080 | 12.9 | 2732x2048 | 600 | 0.37 | 0.3979 | sRGB | 250 | no | no |
| `macbook_pro_16` | yes | 98.79 | 0.6350 | 16 | 3072x1920 | 500 | 0.37 | 0.3979 | sRGB | 250 | no | no |
| `lg_oled_2017_sdr` | yes | 120.41 | 2.5654 | 64.5 | 3840x2160 | 272 | 0.014 | 0.1592 | sRGB | 100 | no | no |
| `lg_oled_2017_hdr` | yes | 120.41 | 2.5654 | 64.5 | 3840x2160 | 754 | 0.038 | 0.1592 | sRGB | 100 | no | no |
| `eizo_CG3146` | yes | 75.19 | 0.7341 | 31.063 | 4096x2160 | 300 | 0.1 | 0 | sRGB | 0 | no | no |
| `65inch_hdr_pq_4knit` | no | 92.22 | 1.9800 | 65 | 3840x2160 | 4000 | 0.004 | 0.007958 | PQ | 5 | no | yes |
| `65inch_hdr_pq_2Knit` | no | 92.22 | 1.9800 | 65 | 3840x2160 | 2000 | 0.002 | 0.007958 | PQ | 5 | no | yes |
| `65inch_hdr_pq_1Knit` | no | 92.22 | 1.9800 | 65 | 3840x2160 | 1000 | 0.001 | 0.007958 | PQ | 5 | no | yes |
| `lg_oled_2026_hdr_pq` | no | 102.63 | 2.2001 | 64.9 | 3840x2160 | 3000 | 0.0005 | 0.007958 | PQ | 5 | no | yes |
| `modern_oled_phone_indoor` | no | 109.97 | 0.3500 | 6.1 | 2532x1170 | 400 | 0.0005 | 0.3979 | sRGB | 250 | no | no |

Notes. `modern_oled_phone_indoor` is the only imazen-only preset (`display_models_imazen.json`); it is
excluded from conformance because pycvvdp 0.5.4 cannot name it (pycvvdp accepts `config_paths`, so a
parity check is possible but has not been run). The `65inch_hdr_pq_*` and `lg_oled_2026_hdr_pq` presets
ship upstream from pycvvdp 0.5.7. HMD presets use a 3 m nominal distance with a 110° diagonal FOV.
The HDR scoring paths in Deliverable 1 do **not** use a named preset: they use `STANDARD_HDR_LINEAR`
with an overridden peak and 4K geometry, a configuration no conformance cell covers.

## Deliverable 3 — documented viewing conditions (summary; quotes in the two part files)

| dataset | setting | stated geometry | ppd | nearest preset (ppd) |
|---|---|---|---|---|
| CID22 | crowd (Subjectify), desktop/laptop | DSBQS: 1 image px = 1 CSS px ≈ 0.0213° (authors: "only an approximation"); pairwise protocol upscaled to screen height | 46.9 nominal (DSBQS); pairwise not derivable | none; between standard_fhd 37.84 and sdr_4k_30 60.55 |
| KonJND-1k | crowd (AMT), card-calibrated physical size | 640 px shown 13.797 cm wide, workers asked to sit at 30 cm | 24.3 derived | none (below every preset) |
| KADID-10k | crowd | not stated ("variable screen resolutions") | — | none justified |
| TID2013 | lab + internet | 19"+ LCD/CRT, 1152×864, distance "comfortable" | — | none justified |
| CSIQ | lab, 4 calibrated LCDs | distance not stated | — | none justified |
| LIVE R2 | lab, 21" CRT 1024×768 | 2–2.5 screen heights | 26.8–33.5 derived | none (below every preset) |
| KonFiG-IQA | crowd (AMT) | not stated; boosted conditions pre-zoomed 2× | — | none justified |
| AIC-3 (Testolina QoMEX'23, 10 refs) | crowd, expert viewers | ≥1920×1080, DPR 1; distance not stated | — | none justified |
| AIC-3 BTC/PTC (DCC'25, 5 refs) = AIC-4 sample stimuli | crowd (AMT) | not stated; BTC = 2× zoom + 2× amplification | — | none justified |
| AIC-4 CTC | (metric configuration, not the human study) | SDR metrics at 37.84 ppd / 200–203 cd/m²; HDR at 56.55 ppd / 1000 cd/m² | 37.84 stated | `standard_fhd` = the CTC's own CVVDP setting |
| SDR25 | crowd, AIC-3 interfaces | not stated | — | as AIC-4 (CTC convention) |
| UPIQ alignment study | lab | SDR 32" 2560×1440 at 90 cm, stated 51 ppd (our arithmetic gives 56.8 — unexplained); HDR 50 ppd after 3.2× upscale | 51 stated / 56.8 derived | none; custom |
| └ Korshunov 2015 | lab, 47" SIM2 FHD, ≤4000 cd/m² | 3.2 picture heights | 60.3 derived | ppd ≈ sdr_4k_30, luminance HDR → custom |
| └ Narwaria 2013 | lab (via the authors' HVEI 2014 paper; OE 2013 not retrieved) | 3 H, 47" SIM2 1080p, 4000 cd/m² | 56.5 derived | custom (≈ CTC HDR 56.55) |

Caveat: the AIC-4 CTC's 37.84 ppd configures the *metrics*; the AIC human studies do not document a
viewing distance. Choosing `standard_fhd` for AIC sets reproduces the organisers' metric convention, not a
measured human geometry. `DATA_SPLITS.md` calls the AIC-3 CTC set 10 references; Mohammadi 2025's AIC-3
evaluation is the 5-source study. Which study produced our AIC-3 labels should be pinned before a
geometry is assigned (helper finding, not resolved here).

## Deliverable 4 — proposed E2b design (proposal only; nothing run)

**Arms.** Score CVVDP with the parity-verified CPU port at a geometry × luminance grid chosen to
separate the two effects, plus the program's named presets:

| arm | ppd | peak | why |
|---|--:|--:|---|
| `standard_4k` | 75.40 | 200 | control (every past value) |
| `sdr_4k_30` | 60.55 | 100 | program list; lower ppd and peak together |
| `standard_fhd` | 37.84 | 200 | program list; AIC/AIC2026 organiser setting; geometry-only change vs 4K |
| `sdr_fhd_24` | 37.84 | 100 | added: pairs with `standard_fhd` to isolate peak luminance at fixed geometry |
| `standard_phone` | 120.56 | 500 | program list |
| `iphone_14_pro` | 159.61 | 1025 | program list |
| `modern_oled_phone_indoor` | 109.97 | 400 | program list; **exploratory until a pycvvdp parity check with `config_paths` passes** |

Report-only extra arms (never selected): custom geometries at the dataset-documented ppd (KonJND 24.3,
LIVE 30 = mid of 26.8–33.5, CID22 46.9, UPIQ-SDR 51 and 56.8), each added to
`display_models_imazen.json` and parity-checked against pycvvdp via `config_paths` first.

**Selection data (the only data used to choose).** TRAIN-role views per DATA_SPLITS §8:
`ext_kadid_train` (40 refs / 5,000 rows), TID train (12 refs / 1,440), `konfig_originsplit_train`.
Label all three as memorised by several zensim eras (irrelevant to CVVDP itself, relevant to any later
comparison with zensim). None has documented viewing geometry, so the selected display is the one that best
fits crowd/unstated conditions; say so.

**Statistic.** Per leg, pooled SROCC and within-image (per-reference) SROCC through `zen_stats.panel`;
paired difference Δ = display − `standard_4k` with reference-clustered bootstrap, B = 2000, seed 20260923.
KonFiG: triplet ordering accuracy on its `q_jnd` scale through the same owner.

**Proposed decision rule (for the coordinator to preregister).** A display *beats `standard_4k`* only if,
on **both** KADID-train and TID-train, pooled Δ ≥ +0.010 SROCC **and** the 99% cluster CI of Δ excludes 0
(Bonferroni over the 6 challenger arms, α = 0.05/6 ≈ 0.008), and KonFiG-train accuracy Δ is not
significantly negative. If several pass, take the largest mean Δ over the two SROCC legs. If none
passes, CVVDP's showing as a teacher stands and E2c does not run. The +0.010 margin is a proposal: it is
above the +0.0065 peak-only effect measured on UPIQ (L11-04) and far below the +0.070 AIC-4 geometry
effect (CVVDPFIX), so a real geometry effect clears it and a luminance-only nudge does not.

**Report, not select.** Each held-out set at its documented geometry where one exists (CID22-A(25) at
46.9 custom; KonJND-504 at 24.3 custom; LIVE at 30 custom; UPIQ-HDR at custom SIM2-class HDR presets,
60.3/56.5 ppd, 4000 cd/m²); AIC-3/AIC-4/SDR25 at `standard_fhd` (organiser convention, flagged as not a
measured human geometry); CSIQ at every arm (no documented geometry). Every read goes in the exposure
ledger. Plus each arm's behaviour in the E1 quality bands.

**Prerequisites.** (a) Land `d71922bd` (CPU `--display-model` + `_<display>` column suffix) or score with
`cvvdp-gpu` into display-named files with a `.meta` sidecar per file — never rely on the column name.
(b) Rebuild a working pycvvdp venv (the `scripts/cvvdp_goldens/.venv` interpreter is gone) for the
custom-preset parity checks. (c) Pin which AIC-3 study our labels come from.

**Cost.** Scoring only: ~(5,000 + 1,440 + KonFiG-train) pairs × 7 arms on selection legs, plus the
report legs; the CPU port measured ~315 ms/MP (L08). No training.

## Caveats

- Producing zenmetrics commits are recorded for few columns (T7 log, T8/T11 manifests, E2 .meta). Where a
  commit is missing, the display comes from the code default over the bounded commit range; the table's
  confidence column says which.
- `profile_b_methodology_2026-07-12.md` records `hdr_v3mix` traindigits sha256 prefix `31e08c70…`; the file
  on disk (mtime 2026-07-03 11:23, unchanged since creation) hashes to `1bbae34c09c01dbb…`. Unresolved;
  the display attribution (T7) does not depend on it.
- The hdrgrid cvvdp wave's image is not in its manifest (T9, medium).
- E8 (AIC-3 organisers' CVVDP) display not established here.
- Deliverable 3 quotes were gathered by a helper agent; four were re-verified against the sources (listed
  in the worklog), the rest are as quoted with locators.
