# One frozen B evaluation against supplied JND reconstruction (2026-10-19)

The owner requested one evaluation of `jnd_reconstruction_2026-10-19.csv`.
Default **B / zensim-b** is the frozen diff+frag-820 bake
`89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e`.
The scorer source is main `8d6aa6cc8143746ea5d7ee27df9c118e74230a76`.

Results on this data are not published here. Curve and available pointwise
labels are evaluated separately. Original pairs were scored once through
`score_pairs_tuner --profile b --threads 2` and `Zensim::compute`.

Native `panel evaluate` owns all rank, per-codec and within-reference statistics;
`panel rank-interval` delegates the 10,000 reference-bootstrap draws to zenstats.
Confidence intervals describe resampling references, not propagated uncertainty
in the reconstructed labels. The provided reconstruction is not raw observer data.
JND rises with distortion; quality correlations use **−JND**.
PLCC uses the existing four-parameter logistic statistical mapping, without
changing the saved B predictions. No training, model selection, corruption
head, new encode or JND calibration was performed.

Admission binds the original RGB8 reference and distorted PNGs by SHA-256
and verifies pair dimensions. No depth conversion or resampling
was needed. CSV SHA-256:
`5bdbf5402c41b2cccf22b62ad352160b88f952a53bea1cce6ae4ef14c5890e61`.
The sparse pointwise table is an exact ID join from the one saved prediction
population. The initial report adapter refused extra score IDs in that subset;
the failed report remains preserved, and only table/statistical reporting was
repeated. Pixels were not rescored.

Artifacts, checked on **dev**:
`/mnt/v/output/zensim/jnd-reconstruction-2026-10-19-b/`.
Native executable/runtime identities and the pre-read freeze:
`/mnt/v/output/zensim/jnd-reconstruction-2026-10-19-b-runtime/`.
Matching NAS paths are under `/mnt/tower/output/zensim/`.
`_MANIFEST.json` binds the final files; `_MIRROR_VERIFIED.json` records full
SHA-256 readback of 34 assessment files and 12 runtime files from the NAS.
Original labels/predictions stay outside Git.

Execution resource receipts are retained with the assessment artifacts.

## Owner-requested matched peer comparison (2026-10-20)

The comparison uses the original admitted curve and pointwise populations.

| Metric | Curve SROCC | Pointwise SROCC | Curve PLCC | Pointwise PLCC |
|---|---:|---:|---:|---:|
| DVIFM (proposal) | 0.6574823375 | 0.6323718520 | 0.6407661925 | 0.6338429951 |
| DVIFM 0.2 (proposal) | 0.8818749291 | 0.8670391190 | 0.8414741030 | 0.8399897263 |
| DVIFM 0.2 use_chroma (proposal) | 0.7785456847 | 0.7728807655 | 0.7293202653 | 0.7285206407 |

Curve and pointwise populations match the original admission.
B predictions are unchanged, bit-for-bit equal to the first report. Peers use
the supplied `metrics_cropped.csv` (SHA256
`3fb7208165c039cdc4bdea2e1244ea52020b60019c9c96f4b00a9b9967a7ad99`),
not fresh runs of current metric implementations. DVIFM variants retain the
supplier's column identities. Original crop files were
SHA256-equal to the frozen B image admission before joining exact distorted
and reference IDs. Bitrate is not an identity key. No new pixel scoring,
training, model selection or JND calibration. Small rank differences are
descriptive; paired significance was not assessed.

Comparison artifacts, checked on dev:
`/mnt/v/output/zensim/jnd-reconstruction-2026-10-19-peers/`;
full SHA-verified mirror under `/mnt/tower/output/zensim/`.
`SUMMARY.json`, native `report/report.json`, `suite.json`, `ADMISSION.json`
and final `_MANIFEST.json` retain values, inputs and native ownership.
The initial preflight rejected dots in metric IDs before statistics; it is
preserved in the adjacent `.preflight.bak` directory. Final IDs use `v02`.

Native reporting and pixel-identity resource receipts accompany the artifacts.
[Supplied dataset](https://doi.org/10.18419/DARUS-6156).

## Owner-selected reproduced Rust score tables (2026-10-20)

**Missing: DVIFM 0.3-rs.** No DVIFM table was present in the selected
score directory when inspected; its path was requested. The historical
DVIFM proposal columns above do not measure DVIFM 0.3-rs.

The owner directed this comparison to retained native Rust score tables.
These reproduce SSIMULACRA2, GMSD and both butteraugli p-norm-3 and max
without substituting the dataset's proposal column. Exact file SHA256s and
column names are in `suite.json` and `ADMISSION.json`.

Results on this data are not published here.

Both admitted populations and original B statistics are unchanged. Crop
hashes and native score-table pixel paths match the B admission. No new pixel
scoring, training, model selection or JND calibration.

Artifacts checked on dev:
`/mnt/v/output/zensim/jnd-reconstruction-2026-10-20-reproduced/`;
full SHA256-verified mirror under `/mnt/tower/output/zensim/`.
Native reporting and admission resource receipts accompany the artifacts.

## Complete native DVIFM 0.3-rs comparison (2026-10-20)

The previously missing DVIFM 0.3-rs leg is now measured. One fresh batch
used `dvifmish` source `31c4e256`, the native full-color `port-lap-v0.3`
preset and the admitted matched pairs, four scoring threads. The resulting
TSV now sits beside the owner-selected reproduced Rust metric TSVs.
Every raw and JND value is exactly equal to the retained Rust `f24a08a`
result for the admitted pairs. No fit, parameter selection or JND calibration.
B predictions are unchanged.

| Metric | Curve SROCC | Pointwise SROCC | Curve PLCC | Pointwise PLCC |
|---|---:|---:|---:|---:|
| DVIFM 0.3-rs (port-lap-v0.3) | 0.9570774651 | 0.9330864427 | 0.9494058223 | 0.9364946319 |

Original crop hashes and native score-table pixel paths match the first
B admission for both populations. DVIFM's frozen preset provenance states that
the author constants were trained on AIC2026; this is not an unseen-corpus
test for DVIFM. No claim of paired statistical significance.

Full results checked on dev:
`/mnt/v/output/zensim/jnd-reconstruction-2026-10-20-dvifm03/`;
matching fully SHA256-verified archive under `/mnt/tower/output/zensim/`.
Exact DVIFM binary, source commit, preset, Cargo.lock, input pairs and
parity receipt are retained under `dvifm-runtime/`; full native statistics
are in `report/report.json`. `suite.json` binds every score input.

DVIFM scoring resource receipt: `run-heavy: done rc=0 32s | peak-RSS
0.07GiB | min-avail 36946MiB | peak-load 19.74`. This is the contended
batch including image I/O, not a comparative performance benchmark.
