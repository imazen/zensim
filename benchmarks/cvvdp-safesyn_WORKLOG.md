# Lane `cvvdp-safesyn` worklog — Rev4 E2b display selection + SafeSyn CVVDP labels (2026-09-23)

Lane: `cvvdp-safesyn`. Brief `/home/lilith/tmp/zensim-paper/rev4/CVVDP_SAFESYN_brief.md`
(+ `CVVDP_SAFESYN_FLEET_addendum.md` — Part 2 runs on the zenfleet job system behind a
FLEET_READY/FLEET_GO review gate). Workspaces: `../zensim--cvvdp-safesyn` (jj,
`main@origin` a14e563e, bookmark `quarantine/devin/cvvdp-safesyn`) and
`../../zenmetrics--cvvdp-safesyn` (jj, `master@origin` 266967e9, bookmark
`quarantine/devin/cvvdp-safesyn`). Scratch: `/var/tmp/cvvdp-safesyn/` (build target
`/var/tmp/cvvdp-safesyn/target-zenmetrics`). Manifest tsv:
`/home/lilith/tmp/devin/rev4_cvvdp-safesyn_manifest.tsv`. Quarantine: nothing is
pushed; nothing here is evidence until audited (`~/tmp/devin/QUARANTINE.md`).

## Setup (2026-09-23 ~03:30 Denver)

- zenmetrics lane: `jj duplicate` copies of cvvdpfix trio onto `master@origin` 266967e9:
  - `27c69beb` (worker arg order) → `6b81258e`
  - `d71922bd` (CPU `--display-model` + `_<display>` column suffix) → `72a7f309`
  - `dc1fca78` (docs) → `5375d523` ← bookmark `quarantine/devin/cvvdp-safesyn`
  - originals in `../zenmetrics--cvvdpfix` untouched.
- Build (queued on the shared `heavy.lock` behind another lane):
  `CARGO_TARGET_DIR=/var/tmp/cvvdp-safesyn/target-zenmetrics ~/tmp/devin/heavy --mem 20G --jobs 8 -- cargo build --release -p zenmetrics-cli --features jobexec`
  log `/var/tmp/cvvdp-safesyn/logs/build-zenmetrics.log`.
- Input hashes (sha256sum, this session):
  - `/mnt/v/output/zensim/reports/refmetrics/kadid_pairs.tsv` ceaf324f2bc646045adba0bfca5a299d27d0eded8e929d64f3d543bcc9e3b040
  - `/mnt/v/dataset/tid2013/tid_pairs_ab.tsv` e6a790e8f766ad5e484c3a66cfd9707f3d23270832b393f98a16cadc72cde032
  - `/mnt/v/output/zensim/konfig944/build/konfig_pairs.tsv` 44bde2d85f86f1da3d7a2a7b02bad919ae1164b2d221181037963598e63703d4
  - `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_kadid_train_2026-08-29.parquet` 1fb8ff9bf335f215c22e7e2f3b63c3a9706d92151236d7a664f5cae1293b3cce
  - `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_tid_train_2026-08-29.parquet` 269c7a8c614f6842cc4a3f5c0d2d938919d1596664424df20cc77d0f9d72e9ea
  - `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/konfig_originsplit_train_944.parquet` b919b9ad458a593baf443c57afb2c64c1e19a45f7042fada0146687a01574683
  - `/mnt/v/output/zensim/v2-backfill-2026-07-20/aic4_pairs.tsv` 955a9601e94c0877a00c822dbbbd17ad4ae11544dd6d550b9ed79d1ad9d983ff
  - `/mnt/v/output/zensim/reports/refmetrics/aic4_cvvdp.tsv` 4042d578c0f15b89d03301a46edc65d19f5e446412651a19d56707f11817d9c3 (300 rows, col `cvvdp_cpu_imazen_v0_1_0`)
  - `/mnt/v/output/zensim/reports/refmetrics/aic4_cvvdp_standard_fhd.tsv` f7bb935f965c9abc983565211ea1425af20cb5eb7c642af5989f6fb410dd06aa (300 rows, col `cvvdp_cpu_imazen_v0_1_0_standard_fhd`)
  - SafeSyn verified cache `/var/tmp/zensim-validation-2026-09-14/baseline-recovery/`: `safesyn-train944.parquet` 6044fdc8cf4f646cda6457a9e73e54edd7d9fed57dfd95f1c47e1f4091220c68, `SAFESYN_ADMISSION.json` 9ced0f04adecfcabebf05c136c05f1c4849bfbc9f798734af642c4d4ee70fd4b, `safesyn-train-pairs.tsv` 5a53976070a5e21b2bb7fe0d05f58b510b93e141dd9207dd3e2cd337fd15cd3b
- Display preset check: `modern_oled_phone_indoor` is in `display_models_imazen.json`
  (resolution 2532×1170, 0.35 m, 6.1", 400 cd/m² peak, 0.0005 black, 250 lux) → both
  `DisplayModel::by_name` and `DisplayGeometry::by_name` resolve; pycvvdp parity status TBD.

## Preregistration

`benchmarks/cvvdp-safesyn_prereg_2026-09-23.md` committed BEFORE any human-label read
in this lane (E2a Deliverable 4 verbatim + operational bindings). Label columns have
not been read at the time of that commit.

(Further entries appended chronologically.)

## Coordinator update ~10:30Z — rebase onto new master (b02812ae)

`master@origin` advanced to `b02812ae` ("bench(cvvdp): video_vs_ssim2"): now
contains the cvvdpfix trio natively plus V5 SIMD/rayon CVVDP, conformance-v2 vs
pycvvdp 0.5.7 (four new HDR PQ displays), and video scoring. Instruction: drop
the trio copies, keep only lane jobexec work, rebuild, redo Part 0.

- `jj rebase -s @ -d master@origin`: my change `a544a65b` → `b373b48a` on
  `b02812ae`; the trio copies (`6b81258e`/`72a7f309`/`5375d523`) left behind —
  upstream ancestors verified (`git merge-base --is-ancestor` for 27c69beb,
  d71922bd, dc1fca78 → all true). Lane diff vs base: `job.rs` +34 (test only),
  `jobexec.rs` +~330 (`cvvdp@<display>` arms + `ZEN_JOBEXEC_PIXEL_HASH`
  stamping + tests). `jj bookmark set` refused the sideways move (guard); the
  lane-local bookmark was repointed via forget+create.
- Part 0 stored-score check (panel via `ZEN_PANEL_BIN=~/work/zen/zensim/target/
  release/panel`): stored `aic4_cvvdp_standard_fhd.tsv` column vs its
  `human_score` → **SROCC 0.9608955655 ≈ 0.9609** ✓; board `aic4_cvvdp.tsv`
  (`cvvdp_cpu_imazen_v0_1_0`) → SROCC 0.8905597718 ≈ 0.8906. No AIC labels
  re-read — scores only.
- Tests on rebased tree (`--features jobexec`, dev profile):
  `cargo test -p cvvdp` → 227 pass/0 fail; `-p zenmetrics-cli cvvdp` → 13 pass;
  `jobexec` filter → 18 pass (incl. new `cvvdp_display_of_parses_only_the_at_
  form`, `cvvdp_at_unknown_display_is_refused`, `cvvdp_at_standard_4k_matches_
  the_plain_cvvdp_row`, `cvvdp_at_standard_fhd_emits_the_display_column`);
  `-p zenfleet-core` → 144 pass (incl. new
  `cvvdp_display_metric_string_is_distinct_work_and_cpu_routed` — plain `cvvdp`
  JobId differs from `cvvdp@standard_fhd`, CpuHeavy routing, empty caps).
- Executor pixel-hash integration completed: `mk_row_px` stamps
  `reference_pixels_sha256`/`distorted_pixels_sha256` (flat RGB8 sha256) into
  every per-variant SDR ScoreFile row once the variant decoded (feature rows,
  `cvvdp@` rows, generic score rows, post-decode error rows); fetch/decode
  error rows stay unstamped (no buffer). HDR `run_score_file_hdr` untouched —
  nits decode, not RGB8. Env gate `ZEN_JOBEXEC_PIXEL_HASH=1`, off by default.
- Release rebuild queued on `heavy.lock` (~23 min behind other lanes so far):
  same `cargo build --release -p zenmetrics-cli --features jobexec` command;
  binary lands at `/var/tmp/cvvdp-safesyn/target-zenmetrics/release/zenmetrics`.

## Post-rebase Part 0 on fresh binary (b373b48a / base b02812ae) — ~05:0xZ

- Fresh release binary `/var/tmp/cvvdp-safesyn/target-zenmetrics/release/zenmetrics`
  (64,428,664 B): sha256 `ebffd5b8e3843d8e1b4987262904a6f43b5789f34fa019e3af799fb1b73b9d45`.
- `batch --metric cvvdp` (no flag) over `aic4_pairs.tsv` (300 pairs, `--jobs 8`):
  all 300 scores **bit-identical** to board `aic4_cvvdp.tsv` `cvvdp_cpu_imazen_v0_1_0`
  (identical=300 differ=0). Column name unchanged → no V5 drift on default path.
- `batch --metric cvvdp --display-model standard_fhd`: all 300 **bit-identical** to
  stored `aic4_cvvdp_standard_fhd.tsv` (`cvvdp_cpu_imazen_v0_1_0_standard_fhd`) →
  zero drift vs the cvvdpfix-era scores. SROCC from stored scores stays 0.9609.
- Throughput smoke: ~15.2s wall / 300 pairs @ 8 jobs ≈ 19.7 pairs/s (default),
  15.5s ≈ 19.3 pairs/s (fhd). vs historical ~1-4 pairs/s — V5 SIMD/rayon landed.
- Blob plan DONE: `s3://codec-corpus/safesyn-rev2-2026-09-06/` already held
  199,304 objects covering 196,030/196,086 pairs; uploaded the 57 missing
  objects (ref `Verkehrstote Deutschland 1953-2012_512sq.png` + its 56 admitted
  variants; the space-named ref was skipped by the rev2 uploader). Coverage now
  100%; spot-check sha256 byte-identical to `SAFESYN_VERIFIED.json` audit
  (`q5.jpg` = c49189b0…). URI scheme: `sources/<ref>.png`,
  `images/<src>/<codec>/<file>` — the `declare-scorefiles --full-uri` route.
- `modern_oled_phone_indoor` parity vs pycvvdp 0.5.7 (config_paths injection of the
  imazen preset, merged display_models.json in /var/tmp/cvvdp-safesyn/pycfg/):
  8 KADID pairs, max |Δ| = 0.0001 JOD (pycvvdp torch/cu130 vs our CPU V5).
  Display is CHECKED, not unchecked — upgrades the lane's weakest preset.
