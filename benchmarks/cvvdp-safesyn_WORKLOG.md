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
