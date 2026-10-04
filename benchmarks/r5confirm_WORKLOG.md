# R5CONFIRM worklog — 2026-10-04

MISSING: none for the six requested artifacts and serving smoke gates. No model-selection or quality-qualification claim.

Authorized by `~/tmp/zensim-paper/rev4/R5CONFIRM_brief.md`. Workspace `../zensim--r5confirm`, bookmark
`quarantine/codex/r5confirm`, parent `9dfc50b8` (descendant of frozen Rev5 arithmetic `60174678`). Local commits only.
This builds artifacts for STEERCODEC/CHROMAQ and the integrity companion; no model-selection or held-out-label read.

## Data ownership and admission

The Rev4 confirm pack `cf9b83179376c78471905c8e97e3280b3d45ea008123f9febdbd660f9046f99d`
was made by `scripts/rev4_featpot/v2c_pack.py --kind confirm` (R2 and DATA_SPLITS transport ledger;
284 members, including the previously missing freeze record). R7's grid owner is `v2_setcompare.py`;
its A/B cells use `v2_confirm_fit.py`, full `human_all`, head N, recipe `@h32:H128:cv16:cf98`.
CHROMAQ's Rev4 final models use each cell's `refit/last.bin`, the canonical `v2_common.dense_bake`
(`bake_dial_refit densify`, bit-identical prediction gate) and `bake_stamp_revision`.

Pixels-only extraction used `rev5_bank.py` and existing verified extractor
`/var/tmp/rev5-extract/target/release/examples/extract_features_372col`, SHA-256
`c649e810f2c3e1d2811e623adc5c7600e952229a7a14db7db9d71ebd9cbbde8b`, build
`1a9d5a8a17ffb57b87b3600ee2b3116011598e45` (descendant of `60174678`). Six confirm sets were
added under `/var/tmp/rev5-featbank` using keys/pixels only; no `_sealed` path was opened.
Feature identity `basic+peaks+v2@w1825/rev5_localwin#36c3f3af`; requested 576 slots
(f0–227, f372–719), remaining slots absent/NaN, wide tables padded to 1853 as at Rev4.

Extended existing owners: Rev5 freeze admits only `main/real` (Rev4 still requires all families/variants),
pack inventory names the actual revision and refuses mixed revisions, R7 grid can restrict entries/seeds.
The CLI now binds its module name before the verifier import, preserving the selected Rev5 width/profile.

`/var/tmp/r5confirm/build_pack.sh` runs confirm → verify → freeze → the same packer. All 30 audit gates passed:
features cast bit-identically, row/key order preserved, training target admission matches, confirm targets all zero,
confirm keys carry no labels, receipt hashes unchanged. Freeze SHA-256
`1611e7667aae2ed17c2d47008b727b7472eba2571de6913de1a6b697e34b84f3`.
Pack `/var/tmp/r5confirm/rev5-confirm.tar.gz`, SHA-256
`1707879d64581d60450e2d6978988f2d77007fc0814e1b513ac6a1076cfe35c1`, 389,904,514 bytes, 38 file
members plus `input_inventory.json`. Contains `main/real` safesyn/cid22/human_all fit/dev, manifests/keys,
receipts/keep list/freeze, and six features-only confirm tables with manifests/keys.
Rows: safesyn fit/dev 141054/38757; cid22 12163/3785; human_all 11970/2447;
confirm cid22_b 2100, aic4 300, konjnd_jpeg_select 404, konjnd_jpeg_terminal 100, csiq 865, mcljci 5000.
Coverage data is carried by pinned program v25b (same keys/rungs as Rev4, features re-extracted at Rev5),
SHA-256 `6bf584ac70579bdf9a0242b7ccfd688ff0182cb5c75e35085ba71967207f8f5e`; not duplicated in this pack.
The six fit/dev tables have identical reference/target row order to R7; teacher key sidecars are identical
(R7 has no human_all key sidecar). Check `/var/tmp/r5confirm/r7_training_parity.json`.
Full inventory and pin checks: `/var/tmp/r5confirm/{input_inventory,data_summary,r7_grid_parity}.json`.

## Fleet launch — 21:54:32 UTC

Jobset `fitv2r5confirm-20261004`, six cells: R7 A/B head N, full-data seeds 0–2. All six argv equal R7
byte-for-byte except `--root /var/tmp/rev4-featpot/v2c5`.
Program `/var/tmp/fitv2/program-v2r5-v25b.tar.gz`, verified SHA-256
`16716aef5492c44ca0ab0efc9bb92860ff7b5aedae143166498f2afc773bdece`;
image `ghcr.io/imazen/zenfleet-worker:fit-v2r5-v25b-w8dc42d4e`.
`/var/tmp/r5confirm/launch.sh` follows `setcmp_launch.sh`: declare-fits, control, manifest/data upload,
queue insertion, v2_loop/score_chain/tail_trim. Queue inserted SECOND after `fitv2e24b-20261004`.
Only existing host fillers place workers; no container launched manually, no herdr restart.
The operational tail_trim copy filters its existing LAN owner to Tower/i265/i270/r3500/r3800x and omits dev;
source/adapted hashes and exact changes in `/var/tmp/r5confirm/tail_trim_provenance.json`.

## Validation

`TMPDIR=/var/tmp/r5confirm ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 -m unittest scripts.tests.test_v2c`:
71 synthetic tests passed (64 s), including Rev5 subset freeze, mixed-revision refusal and confirm feature-identity refusal.
`just lint-scripts`: 811 scripts, all runnable. CI-exact `just clippy` through run-heavy: exit 0 (54 s).
Production serving examples built from this workspace at the pinned parent, release,
features `custom-profiles,feature-regime-v2,threads,training` (65 s, exit 0).
The recorded release builds, extraction, table verification, packing and final HDR probe build used
run-heavy with 16G/8 jobs. The HDR probe links the same production library and asserts revision 5,
SDR/HDR identity = 100, prepared/scalar score equality and finite density/rectangle gains.

Two setup attempts failed before the successful pack: first confirm started before MCL-JCI extraction completed;
second verifier import reset the CLI profile to Rev4. Both failures are recorded; no failed archive was declared or uploaded.
Logs `/var/tmp/r5confirm/{extract,pack,test_v2c,build_serving,clippy,lint_scripts,launch}.log`.

## Worker placement — 22:44:45 UTC

The existing Tower filler placed workers after the E24 jobset's unclaimed work was exhausted.
Six task claims subsequently observed on `Tower-q926`–`Tower-q931`; snapshot
`/var/tmp/r5confirm/initial_claims.json`. No manually launched containers.

## Harvest, export and serving — 23:14 UTC

All six cells completed on the allowed Tower workers and were verified/installed by the existing
`harvest_driver_v2.py` → `harvest_fit_cells.py` owner. Three by_v2fy cells installed incrementally;
three v2+basic cells in the final harvest. All receipts report tier v3. Finalizer validates program/data/argv,
job/blob identity, receipt/result/source-bake hashes, frozen receipts, Rev5 coverage pin and finite
features-only predictions. Every result uses `epoch_rule=last`, `epochs=120`, `selected_epoch=119`
(the final zero-based epoch). No checkpoint was selected from held-out labels.

`/var/tmp/r5confirm/finalize_bakes.py` calls the canonical `v2_common.dense_bake` (its BIT-IDENTICAL
prediction gate) then `bake_stamp_revision <dense> 5 <output>`. No hand-written bake serializer.
Final artifacts: `~/tmp/rev5bakes/`, six models plus `SHA256SUMS`, `provenance.json`, `serving_gates.json`.
Provenance contains program/data/manifest/cell/result/receipt/blob/source/dense/final SHA-256s,
seed streams, final epoch, trainer/predictor/panel binaries, worker identities and serving binary/source/fixture pins.

| Bake | Bytes | SHA-256 |
|---|---:|---|
| byv2fy-full-s0.bin | 216448 | `690b270936b1b361d50573b9572e6c34f8edfe73fb21ec197dbcc9705ae7a314` |
| byv2fy-full-s1.bin | 207907 | `c043e3f447ead371fc6e94c4d20b4940f6f5d26001d9a2c4d9b2b3e588fc1cf2` |
| byv2fy-full-s2.bin | 216899 | `38741b490416f977461783f03c6da680e637af80f9f3e24018495a4cb65e59be` |
| v2basic-full-s0.bin | 254428 | `51c724a8cb3cb11c312224ac564c1795678196af1971ea3af4c3dff5d58953d3` |
| v2basic-full-s1.bin | 259244 | `c803a37204ac4c6def7d06fc1f51e71c98f30bd0df593b4c1faa2ea86a3c5ed1` |
| v2basic-full-s2.bin | 262918 | `98e113c49e29d0ea95a44f01249fcda087af676a9c60fa5b51f3282d3eb1e850` |

Provenance SHA-256 `1ab3d1afbdfda4af1448ec13f331a83feb4db75ad2a865827d3ad72e7d70e07e`.
Serving gates SHA-256 `05a46e3e4de939203f530c3e940ca1af3411c07fb203a9bb2a67e3b649e1cf9a`.
`sha256sum -c SHA256SUMS`: all six OK.

Serving command `/var/tmp/r5confirm/run_serving.sh`, run-heavy 16G/8 jobs, environment
`ZENSIM_FORMULA_REV=5 ZENSIM_NEIGHBOUR_EXACT=1 ZENSIM_PREPARED_STEERING=1 RAYON_NUM_THREADS=1`:

- Production `serve_custom_bake --pairs`: 18 finite scores (three SDR pairs × six bakes); the natural-image
  identity pair scores exactly 100 for all six. Two unlabelled CHROMAQ JPEG distortions and identity,
  centre crops 97×83 RGB8 sRGB; source/crop hashes in fixture provenance.
- Native Rust HDR probe links the SAME built production library. Each exact exported model declares Rev5,
  serves SDR and native HDR (97×83 opaque LinearF32Rgba, byte stride 1552, HdrEncoding::Linear),
  has direct identity score exactly 100, and prepared SDR/HDR scores exactly equal scalar scores.
  All density entries and rectangle gains are finite. The generated HDR fixture is a serving smoke, not an HDR-quality assessment.
- `diffmap_block_coherence --bake --block 32 --json`, prepared steering and neighbour-exact enabled:
  six successful reports, 12 block interventions/rectangle queries per bake (72 total), all scalar deltas,
  density gains and finite-refinement gains finite. Density/refinement unsupported-ID lists empty in all six reports.
- No serving refusal; therefore no refusal file:line to report. No rank, model-selection or encoder RD verdict.

HDR non-identity scores (synthetic fixture): by_v2fy seeds 0–2 = 93.60839080810547, 91.03465270996094,
87.06578826904297; v2+basic = 92.69828796386719, 92.03557586669922, 90.76319885253906.
Detailed receipts `/var/tmp/r5confirm/{serving_pairs_scores,serving_hdr_scores}.tsv`, six
`coherence_*.json`, `{harvest_incremental,harvest_final,densify_incremental,finalize,serving}.log`.
The existing score_chain and tail_trim completed; no herdr restart, manual container launch or push.
