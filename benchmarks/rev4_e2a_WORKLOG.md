# Rev4 lane E2a worklog — CVVDP display inventory (2026-09-23)

Lane: `e2a`. Workspace `../zensim--rev4-e2a` (jj, parent `main@origin` ba922d83). Analysis of existing
files and code only; no scoring, no training, **no human labels read** (no prereg needed per brief).
Bulky intermediates: `/var/tmp/rev4-e2a/`.

## Step 1 — display catalogue (Deliverable 2)

pycvvdp venv `zenmetrics/scripts/cvvdp_goldens/.venv` is broken (its uv interpreter symlink target
`~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu` no longer exists), so pycvvdp could not be
imported. Instead `/var/tmp/rev4-e2a/catalogue.py` re-implements pycvvdp 0.5.4 `display_model.py`
formulas (read from the venv's installed source): geometry `get_ppd` (lines 433-525: height from
diagonal+aspect, `pix_deg = 2*deg(atan(0.5*width_m/res_x/dist))`, `ppd=1/pix_deg`; fov_diagonal branch
for HMDs, distance 3 m), photometry `load` (154-197: `contrast = Y_peak/min_luminance` else `contrast`
else 500; `k_refl` default 0.005) and `get_black_level` (372-376: `Y_refl = E_ambient/pi*k_refl`,
`Y_black = Y_peak/contrast`); EOTF from `color_spaces.json` (default colorspace `sRGB`).

Command: `python3 /var/tmp/rev4-e2a/catalogue.py <pycvvdp054 display_models.json> > /var/tmp/rev4-e2a/catalogue.tsv`
Inputs (sha256):
- `zenmetrics/crates/cvvdp/data/display_models.json` 653979727e809fcd9e44424aff2615d54edde026d9feb17d8f07b7f0ff61b517 (identical md5 in cvvdp-gpu/ and ../zenmetrics--cvvdpfix)
- `zenmetrics/crates/cvvdp/data/display_models_imazen.json` be7ac75d7c6063f068a389bddcb0f9ba49274bc6bbd76fe9f712102a8d0c860f
- pycvvdp 0.5.4 `vvdp_data/display_models.json` 1921c53cc3d69478d35d5a56d768db3c3aa9e1465cfa044f84d1edbd3bdeb779 (22 presets)
Outputs: `catalogue.py` 88fa1346…fe71, `catalogue.tsv` e4e7340b19947f9cbe86a8be487d91d181d496614e6614ca36e27b8e0cb633b3.
Cross-checks against independently recorded values: standard_4k 75.40 ppd, standard_fhd 37.84 ppd,
Y_refl 0.3979 (CVVDPFIX_DONE.md and aic4_cvvdp_standard_fhd.tsv.meta); modern_oled_phone_indoor 109.97
(zensim_b_phone_oled_methodology_2026-05-26.md:32). All match.
Conformance membership: `crates/cvvdp-conformance/src/displays.rs` on zenmetrics origin/master (266967e9)
lists 9 displays; the unmerged `../zenmetrics--cvvdpfix` version lists 13 (adds 65inch_hdr_pq_{1K,2K,4k}nit,
lg_oled_2026_hdr_pq, which pycvvdp 0.5.7 ships).

## Step 2 — producing code paths (zenmetrics, read via `git show origin/master:` and history)

- `088f4bf5` (2026-05-25T23:50-06:00) "feat(cli): --display-model flag for cvvdp scoring". Its message:
  "The CLI cvvdp scoring path was pinned to STANDARD_4K viewing geometry + photometry for every score
  ever computed". Verified in the parent: `git show 088f4bf5^:crates/zen-metrics-cli/src/metrics/cvvdp_gpu.rs`
  lines 230/269 construct with `CvvdpParams::PLACEHOLDER`, line 275 `ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree()`;
  `git show 088f4bf5^:crates/cvvdp-gpu/src/params.rs:1490` PLACEHOLDER `display: DisplayModel::STANDARD_4K`;
  same at `0422c968` (2026-05-15, the backfill-era scorer) params.rs:279. `088f4bf5^` main.rs has 0 matches
  for `display.model`.
- origin/master today: `batch`/`score-pairs` honour `--display-model` only for `cvvdp-gpu`
  (main.rs:1567, 2592 warn "only affects cvvdp-gpu; ignored"); default `DisplayTarget::default()` =
  STANDARD_4K photometry+geometry (metrics/cvvdp_gpu.rs:65-70). The GPU column name is
  `cvvdp_imazen_v0_0_1` whatever the display (see Step 3 for a measured instance).
- CPU `cvvdp` (column `cvvdp_cpu_imazen_v0_1_0`) goes through the umbrella `run_metric` →
  `zenmetrics-api/src/cpu_dispatch.rs:136-152` with `CvvdpParams::default()` = PLACEHOLDER = STANDARD_4K
  (`crates/cvvdp/src/params.rs:1550`, 738-742). No display input exists on origin/master; the flag and
  `_<display>` column suffix arrive only in unmerged `d71922bd` (../zenmetrics--cvvdpfix).
- `zenmetrics jobexec` / fleet SDR scoring uses `run_metric` (jobexec.rs:147-154) → fixed STANDARD_4K.
- HDR scoring: `DisplayTarget::hdr(peak)` (cvvdp_gpu.rs:110-117) = STANDARD_HDR_LINEAR photometry with
  `y_peak = peak`, geometry STANDARD_4K (75.4 ppd). Peak = measured reference content peak since `6471f4d7`
  (2026-08-06, "measure display nits from the reference ... never config"); before that a configured
  constant (`HDR_DISPLAY_PEAK_NITS`, 1000 per L11-04 and zenmetrics `hdr_cvvdp_faithful_2026-06-03.md`).
- `scripts/sweep/pycvvdp_worker.py` (origin/master) `--display-name` default `standard_4k`,
  column `cvvdp_pycvvdp_v054`, `metric.predict(ref, dist)` (swapped args; fix 27c69beb unmerged).
- `run_gpu_metrics.sh` (`/mnt/v/output/zensim/reports/refmetrics/run_gpu_metrics.sh`, 2026-07-15):
  `zenmetrics batch --metric cvvdp-gpu --pairs … --output …` for kadid and tid, **no --display-model**.

## Step 3 — stored values: files, columns, hashes (read headers/metadata only, never label columns)

- Board peer files `/mnt/v/output/zensim/reports/refmetrics/*cvvdp*.tsv`: sha256 list in
  `/var/tmp/rev4-e2a/refmetrics.sha256` (e.g. aic4_cvvdp.tsv 4042d578…d9c3, cid22_cvvdp.tsv 913a454a…7076,
  kadid_cvvdp_gpu.tsv 3aeb63d7…702c, aic4_cvvdp_standard_fhd.tsv f7bb935f…06aa, run_gpu_metrics.sh
  2ba18adb…745c). Last-column names: command H6 below.
- iPhone/phone runs `/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25/`: kadid_cvvdp_iphone14.tsv 57fc1dce…c149,
  kadid_cvvdp_phone.tsv 8422a3eb…ed4, tid_cvvdp_iphone14.tsv 9ccddc9e…b5, tid_cvvdp_phone.tsv 62286351…f9;
  sanity8_{std4k,ip14,phone}.tsv 6d19b8a3…/9a6d1705…/79d194b6… — same column `cvvdp_imazen_v0_0_1`, values differ.
- HDR: datagen-2026-06-23-hdr/sidecars/zenjxl/cvvdp.parquet fbc68ef7…d4 (schema image_path,codec,q,knob_tuple_json,
  cvvdp_imazen_v0_0_1; 7,980 rows; no key-value metadata); producing command in `log/zenjxl.score.log`
  (`zenmetrics score-pairs --metric cvvdp --hdr --hdr-transfer pu-rescale … --gpu-runtime cuda`).
  hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet 1bbae34c09c01dbb… (record says 31e08c70…; mismatch noted).
  Builder `scripts/hdr/build_hdr_train_parquets.py` at zensim 0f567ad7 reads `sidecars/zenjxl/cvvdp.parquet`
  from `--datagen` default `/mnt/v/output/zenmetrics/datagen-2026-06-23-hdr`.
- UPIQ panels `/mnt/v/output/zenmetrics/upiq-pu/panel_cvvdp_gpu{,_4000,_6000,_10k}.tsv` 7aa78f47…/c2e1a200…/d1e3f725…/93138160….
- AIC2026 `metrics_fullres.csv` 786e1647…010951 (columns CVVDP, JND_CVVDP).
- Manifests read: canonical-2026-05-21/_MANIFEST.json (entries 0-17), 2026-05-15-cvvdp-r2/_MANIFEST.md,
  2026-05-17-cvvdp/_MANIFEST.md, hdrgrid-2026-08-06/_MANIFEST.json, kadis700k canonical README_gpu.md,
  DATA_PROVENANCE.md §2026-06-24, §KADIS-700k, §kadis-hdr, §hdrgrid.
- Excluded: benchmarks/cvvdp_gpu_mode_probe_2026-08-05.tsv (timing/memory only, no CVVDP values).

## Step 4 — Deliverable 3 (helper agent) and its verification

A read-only helper (general-purpose agent, opus) wrote `/var/tmp/rev4-e2a/d3_viewing_conditions.md`
(sha256 682e9f6fe91638913f2b757bf0a4a278c542101f74f8332656ee5f698d54e221), committed split as
`rev4_e2a_viewing_conditions_p{1,2}_2026-09-23.md`. Re-verified by the lane:
- `grep -c '0.0213 degrees' /mnt/v/input/papers/5b/5b69d93b*.md` → 1 (CID22)
- `grep -c 'viewing distance to 30 cm' /mnt/v/input/papers/c2/c20a173c*.md` → 1 (KonJND-1k)
- `pdftotext …/68/6845a362….pdf - | tr '\n' ' ' | grep -o '.\{60\}pixels per degree.\{60\}'` → "The viewing
  distance was 90 cm for both the SDR display (51 pixels per degree)" (UPIQ)
- `grep -o '3\.2 picture heights' /mnt/v/input/papers/17/17c3f5f8….md` → found (Korshunov)
- UPIQ SDR derived ppd from 32" 2560×1440 @ 0.9 m: 56.76 (python one-liner) — reproduces the helper's 56.8.
- AIC2026 display (lane's own read): `/mnt/v/input/papers/ce/ceb244cf….md:150` "CVVDP scores were computed using
  the standard Full HD SDR display configuration ( cvvdp -d standard_fhd )".

## Step 5 — headline recompute commands (run 2026-09-23, from ~/work/zen) and their outputs

H1 `python3 -c "import math;f=lambda d,r,D:1/(2*math.degrees(math.atan(0.5*(d*0.0254*r[0]/math.hypot(*r))/r[0]/D)));print(round(f(30,(3840,2160),0.7472),2),round(f(24,(1920,1080),0.6),2),round(f(6.1,(2532,1170),0.35),2),round(f(6.1,(2532,1170),20*0.0254),2))"`
   → `75.4 37.84 109.97 159.61` (standard_4k, standard_fhd, modern_oled_phone_indoor, iphone_14_pro)
H2 `git -C zenmetrics log -1 --format='%h %ad %s' --date=iso-strict 088f4bf5`
   → `088f4bf5 2026-05-25T23:50:36-06:00 feat(cli): --display-model flag for cvvdp scoring (iPhone-14 unblock)`
H3 `git -C zenmetrics show 088f4bf5^:crates/zen-metrics-cli/src/metrics/cvvdp_gpu.rs | grep -n 'STANDARD_4K.pixels_per_degree'`
   → `275:    let ppd = cvvdp::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();`
H4 `git -C zenmetrics grep -h 'HDR_DISPLAY_PEAK_NITS: f32' 0619afcc -- crates` → `pub const HDR_DISPLAY_PEAK_NITS: f32 = 1000.0;`
H5 sanity8 paste/awk (see DONE file) → header `cvvdp_imazen_v0_0_1` ×3; `max ip14-4k diff 1.4111`
H6 last column of every refmetrics *cvvdp*.tsv → only `aic4_cvvdp_standard_fhd.tsv` carries a display suffix.
H7 `grep -n -o 'cvvdp -d standard_fhd' /mnt/v/input/papers/ce/ceb244cf….md` → `150:cvvdp -d standard_fhd`
