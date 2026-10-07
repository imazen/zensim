# UPIQ-380 ingestion policy — frozen before label conversion

Owner decision D3 (2026-10-07) admits exactly UPIQ's HDR subset as T2 TRAIN.
All other UPIQ is T0. [Dataset documentation](https://www.cl.cam.ac.uk/research/rainbow/projects/upiq/)
and local `/mnt/v/datasets/upiq/README.md` identify Narwaria and Korshunov as
HDR. Their original EXR directory filenames establish the reference/distortion
pairs without opening a label file: 140 + 240 pairs, 10 + 20 references.
The metadata-only initial record is
`/home/lilith/tmp/upiq-2026-10-07/metadata-membership.json`, SHA-256
`e21673d352ad26ac1d9ed494c9ca0480b22c8cb206c34a9f852fc47d54ad5496`.

Original inputs: `/mnt/v/datasets/upiq_extracted/upiq_dataset/images/`,
restricted to explicitly listed Narwaria/Korshunov EXRs. No SDR image payload
is opened. Existing HDR-only JOD derivative:
`/mnt/v/output/zenmetrics/upiq-pu/upiq_cid_jod.csv`, SHA-256
`e0b23f539d46845f6cc4a65591474bafbf7eb4faf02da7175fb342caa5df80ac`.
All 380 identifiers bind the metadata coordinates. This derivative's producer
commit is unknown; it does not acquire qualified provenance by being copied.
The original mixed subjective/objective CSVs are excluded and unopened.

Before label conversion, split by original reference EXR SHA-256 interpreted
as an unsigned integer modulo 5: remainder zero is TRAIN development, all
others TRAIN fit. No label-dependent rebalance. Both slices retain T2 TRAIN;
development is report-only at final epoch 119, never a generalization test,
selection or calibration input. Target representation is fixed as
`100 + 10 * JOD`, no clipping; JOD zero maps to reference score 100. No label
range sets the transform, and E31's proposed rank-only loss uses its order.

`scripts/rev4_featpot/upiq380.py` owns metadata admission and v2 table creation.
`upiq_pu_score --training-allowlist JSON --allowlist-sha256 SHA --out TSV`
extracts through the existing zenexr BT.709 absolute-linear-nits decoder and
`research::extract_hdr`, the production HDR/PU fold used by E26/E27, at process
`ZENSIM_FORMULA_REV=5`. Original file hashes, the complete allowlist and exact
by_v2fy 420 IDs are bound before decoding. Requested features are f64; absent
slots are NaN. This explicit slot subset does not invent a family-token ID.
The output manifest records actual input reads and research provenance.

`just upiq380-check`, `just upiq380-rust-check`, `just upiq380-clippy` and
`just upiq380-build` provide the local checks/build. All heavy commands run
through run-heavy. There is no model fit or fleet smoke in ingestion. Every
future fleet smoke must exercise actual fit-cell-exec staging, inventory
verification and FIT_ROOT binding.

Round-two review exposed two admission defects: malformed extraction options
could enter the legacy label reader, and Rust accepted a sorted 420-slot set
without exact ID identity. Both are fixed locally in `a7625c14`; seven Python
and seven Rust tests pass. Twenty actual-binary refusals include all eleven
original cases, exact-ID refusal before image metadata and eight malformed
CLI refusals before synthetic label access. The original frozen binary
reproduces both CLI defects and the wrong-ID path access using only scratch
tripwires. The prior data remains immutable; round-two extraction retains
the historical 1825-column transport explicitly despite newer research slots.

E31's final proposal is
[e31_upiq380_registration_2026-10-07.md](e31_upiq380_registration_2026-10-07.md).
It supersedes the five-source draft: four D1 folds, seeds 0–9, final119,
E30 nA3 control with exact-parity reuse or one frozen matched fresh control.
AIC-3 is excluded from every fit/development/preprocessing/packing/evaluation
or control-discovery read. Launch waits for all 40 E30 nA3 cells and exact pins.
The one nominal-weight-4 human rank arm retains E21 numerical SDR guards with
E32's seed-paired reduction. UPIQ TRAIN/development and UPIQ-calibrated
HDR-VDP-3 cannot qualify independent human HDR accuracy. Frozen HDR-VDC/AVT
video reads are report-only. The independent Squintly HDR study and its
separately registered decision rule remain missing. The unrecovered label
producer requires explicit owner disposition before fitting; see
[upiq380_label_provenance_2026-10-07.md](upiq380_label_provenance_2026-10-07.md).
