# One frozen B evaluation against supplied JND reconstruction (2026-10-19)

The owner requested one evaluation of `jnd_reconstruction_2026-10-19.csv`.
Default **B / zensim-b** is the frozen diff+frag-820 bake
`89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e`.
The scorer source is main `8d6aa6cc8143746ea5d7ee27df9c118e74230a76`.

| Supplied target | Rows | Quality SROCC | 95% source-bootstrap interval | Logistic PLCC | KROCC |
|---|---:|---:|---:|---:|---:|
| JND_curve (primary) | 5,768 | 0.9112672374 | [0.8821988787, 0.9368212923] | 0.8881425161 | 0.7425199285 |
| JND_pointwise (available observations) | 2,050 | 0.8972082446 | [0.8686565614, 0.9214329945] | 0.8857370772 | 0.7142464295 |

Both populations cover 42 references. All 5,768 original pairs were scored
exactly once through `score_pairs_tuner --profile b --threads 2` and
`Zensim::compute`. Every score is finite; no paired row was dropped.
Native `panel evaluate` owns all rank, per-codec and within-reference statistics;
`panel rank-interval` delegates the 10,000 reference-bootstrap draws to zenstats.
Confidence intervals describe resampling references, not propagated uncertainty
in the reconstructed labels. The provided reconstruction is not raw observer data.
JND rises with distortion; quality correlations use **−JND**.
PLCC uses the existing four-parameter logistic statistical mapping, without
changing the saved B predictions. No training, model selection, corruption
head, new encode or JND calibration was performed.

Admission binds all 5,810 original RGB8 PNGs (42 references and 5,768 distorted
images) by SHA-256 and verifies pair dimensions. No depth conversion or resampling
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

Scoring resource receipt: `run-heavy: done rc=0 177s | peak-RSS 0.15GiB |
min-avail 10633MiB | peak-load 4.41`. This is the whole evaluation process,
not a quiet benchmark or a per-image memory claim. Native assessment completed
in 1s, peak-RSS 0.01GiB; source-bootstrap calls completed in 3s and 1s,
peak-RSS 0.01GiB each. No comparative runtime claim.


## Owner-requested matched peer comparison (2026-10-20)

| Metric | Curve SROCC | Pointwise SROCC | Curve PLCC | Pointwise PLCC |
|---|---:|---:|---:|---:|
| zensim-b (diff+frag-820) | 0.9112672374 | 0.8972082446 | 0.8881425161 | 0.8857370772 |
| SSIMULACRA2 | 0.9075910932 | 0.8907853818 | 0.8840582700 | 0.8794177084 |
| Butteraugli (proposal) | 0.9379678144 | 0.9236748528 | 0.9156786733 | 0.9055576203 |
| DVIFM (proposal) | 0.6574823375 | 0.6323718520 | 0.6407661925 | 0.6338429951 |
| DVIFM 0.2 (proposal) | 0.8818749291 | 0.8670391190 | 0.8414741030 | 0.8399897263 |
| DVIFM 0.2 use_chroma (proposal) | 0.7785456847 | 0.7728807655 | 0.7293202653 | 0.7285206407 |
| GMSD | 0.9079580431 | 0.8800895537 | 0.8741040690 | 0.8658162975 |

Same 5,768/2,050 rows and 42 references; zero missing/dropped scores.
B predictions are unchanged, bit-for-bit equal to the first report. Peers use
the supplied `metrics_cropped.csv` (SHA256
`3fb7208165c039cdc4bdea2e1244ea52020b60019c9c96f4b00a9b9967a7ad99`),
not fresh runs of current metric implementations. DVIFM variants retain the
supplier's column identities. Every one of the 5,810 original crop files was
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

Native reporting: `run-heavy: done rc=0 10s | peak-RSS 0.03GiB |
min-avail 36279MiB | peak-load 25.70`. This is statistical reporting, not
metric execution timing. Pixel identity verification completed in 4s,
peak-RSS 0.09GiB. [Supplied dataset](https://doi.org/10.18419/DARUS-6156).
