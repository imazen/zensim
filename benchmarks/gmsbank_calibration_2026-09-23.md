# GMSBANK C8 pixel-only calibration (2026-09-23 local, 2026-09-24 UTC)

No human labels or fitted metric targets were read. This is the units conversion frozen by `benchmarks/gmsbank_prereg_2026-09-23.md`, not a performance tuning step. The input tables were `/var/tmp/rev4-featbank/bank/{cid22_train,safesyn}/keys.parquet` (SHA256 `c99a0887705b17bff05bdbfc55b89f9b4f060bc94a009302543f15d9a0127219` and `12d48d7fc02afd5026067a348ea20a3896ea6de9a94bd99a25ec2db071c2b28f`). Only pixel paths, dimensions, digests and pair keys were read.

## Method

- `LegacyRgb8` decoded pixels, their SHA256 digests, and `SHA256(ref_digest || dist_digest || "legacy-rgb8")` were asserted for every selected pair before calibration. The reference pixel classifier is fixed in the design note. At most 32 references per corpus/content/size stratum, in SHA256(ref_group) order, and at most 2 distortions per reference in pair-key order were sampled. Empty strata stay empty.
- On co-sited interior pixels, both the GMSD gamma-luma 2×-box Prewitt magnitude and zensim scale-1 XYB Y central-difference magnitude had to exceed `1e-6`. Each pair contributed one median across its reference and distorted eligible sites; the pooled median equal-weights pairs.
- `R = pooled median(m_Prewitt/m_XYB_Y)`. `c_mid = 0.0026/R²`; the bank constants are `c_mid·4^(k−2)`, k=0..4. No fitting, resampling or hypothesis test was performed.

## Result

- Selected pairs: **652**; eligible co-sited sites: **175,324,296**; nonempty strata: **12/32**.
- Pooled ratio R: p25 **1.0395610843687322**, median **1.1044535823324719**, p75 **1.184353457359629**. This quartile spread describes content variation in the deterministic conversion.
- `c_mid = 0.0021314660107855971`. Pinned source literals in `zensim/src/gmsbank_constants.rs`: `0.00013321662567409982, 0.00053286650269639927, 0.0021314660107855971, 0.0085258640431423883, 0.034103456172569553`.

| corpus/content/size | n pairs | p25 R | median R | p75 R |
|---|---:|---:|---:|---:|
| cid22_train/line_art/large | 0 | — | — | — |
| cid22_train/line_art/medium | 8 | 1.11159 | 1.16303 | 1.22469 |
| cid22_train/line_art/small | 0 | — | — | — |
| cid22_train/line_art/tiny | 0 | — | — | — |
| cid22_train/mixed/large | 0 | — | — | — |
| cid22_train/mixed/medium | 64 | 1.10297 | 1.1491 | 1.17818 |
| cid22_train/mixed/small | 0 | — | — | — |
| cid22_train/mixed/tiny | 0 | — | — | — |
| cid22_train/photo/large | 0 | — | — | — |
| cid22_train/photo/medium | 64 | 1.00575 | 1.05673 | 1.12014 |
| cid22_train/photo/small | 0 | — | — | — |
| cid22_train/photo/tiny | 0 | — | — | — |
| cid22_train/screen/large | 0 | — | — | — |
| cid22_train/screen/medium | 42 | 1.11843 | 1.2046 | 1.26152 |
| cid22_train/screen/small | 0 | — | — | — |
| cid22_train/screen/tiny | 0 | — | — | — |
| safesyn/line_art/large | 64 | 1.17301 | 1.20238 | 1.22423 |
| safesyn/line_art/medium | 64 | 1.07749 | 1.18421 | 1.21425 |
| safesyn/line_art/small | 0 | — | — | — |
| safesyn/line_art/tiny | 0 | — | — | — |
| safesyn/mixed/large | 64 | 1.05674 | 1.10095 | 1.13939 |
| safesyn/mixed/medium | 64 | 1.05551 | 1.09159 | 1.17596 |
| safesyn/mixed/small | 0 | — | — | — |
| safesyn/mixed/tiny | 0 | — | — | — |
| safesyn/photo/large | 64 | 0.974921 | 1.05504 | 1.08717 |
| safesyn/photo/medium | 64 | 0.980005 | 1.03308 | 1.08211 |
| safesyn/photo/small | 0 | — | — | — |
| safesyn/photo/tiny | 0 | — | — | — |
| safesyn/screen/large | 26 | 0.97682 | 1.10245 | 1.15576 |
| safesyn/screen/medium | 64 | 1.04045 | 1.08086 | 1.16048 |
| safesyn/screen/small | 0 | — | — | — |
| safesyn/screen/tiny | 0 | — | — | — |

## Recompute and raw outputs

- Working directory for each command: `/home/lilith/work/zen/zensim--gmsbank`. Every heavy step ran through `/home/lilith/tmp/devin/heavy --mem 16G --jobs 8 --` with `CARGO_TARGET_DIR=/var/tmp/gmsbank/target`.
- `python3 scripts/gmsbank/calibration_select.py refs` produced 3,419 reference paths in `/var/tmp/gmsbank/calibration/refs.tsv`, SHA256 `8d4b9135ea4ca326066206cf92bd45171bd9d9f8f15d5448653b534425d4ffd1`.
- `/var/tmp/gmsbank/target/release/examples/gmsbank_decode_dump classify /var/tmp/gmsbank/calibration/refs.tsv /var/tmp/gmsbank/calibration/classes.tsv` decoded and classified all 3,419 references, exit 0; classes SHA256 `824c8e2b9be0e11fd1c8a37a6f447ee4195c39561a70ff97817dc2ee57209e5d`, log SHA256 `36c31bdbc234101fbbdf49d829e486dfbe8cdf2aa2db941299881f072dac492a`.
- `python3 scripts/gmsbank/calibration_select.py select` ran 2026-09-23 23:53:42.210517–23:53:43.005504 UTC, exit 0; `/var/tmp/gmsbank/calibration/pairs.tsv` SHA256 `f6f201193100c537a633ee81b09587554e40ac8359011f1d452850bc4ff870bb`, `selection.json` SHA256 `e01e1f09b4688a7a77e92b28a367549ddb5975191d0ae2656d700402f9d7a84f`. Actual log line: `pairs 652`.
- `/var/tmp/gmsbank/target/release/examples/gmsbank_decode_dump dump /var/tmp/gmsbank/calibration/pairs.tsv /var/tmp/gmsbank/calibration` ran 2026-09-23 23:53:53.618151–23:57:34.634208 UTC through heavy, exit 0; `planes.tsv` SHA256 `aa49fbe114acb464c9d88316c28b7bd1e0b1ee8a17b6def9a742b6d4d034a6d3`, log SHA256 `36084cadc58dad99e9a6ddccbbcee113acc848c632437864aaf88ba2dbe58ded`.
- `GMSBANK_CALIB_DIR=/var/tmp/gmsbank/calibration CARGO_TARGET_DIR=/var/tmp/gmsbank/target cargo test -p zensim --release --all-features --lib gmsbank_calibration_scale1_y_dump -- --nocapture` ran 2026-09-23 23:59:43.288624–2026-09-24 00:03:39.648453 UTC through heavy, exit 0. Actual line: `test result: ok. 1 passed; 0 failed; 0 ignored`; `ratios.tsv` SHA256 `767197dfae605bb05eb078fa27b064dc456cbcb7a1e645531f0ee30fb1338c40`, `xyb8.tsv` SHA256 `5898620078ecc4c64e7f50bcaa5637f399fa1c80ed546cea0bce3e0a3fd53376`, log SHA256 `1a176d6e46375deddd2838c0d68b8261621da828963fe7b3c90e9662996149cb`.
- `python3 scripts/gmsbank/calibration_report.py` ran 2026-09-24 00:04:00.516023–00:04:00.569607 UTC, exit 0. Raw `/var/tmp/gmsbank/calibration/report.json` SHA256 `a0f0d183ce47f0f9cb46c15d3ee792b15cce029a9b6f8c11269ddae1dad2c5fb`; log SHA256 `78294c1ef448d5fe5432895acd8bd6f90ef9027ca0853c5b5e234d8f7cceacf7`. Its first JSON line contains the reported n, quartiles, constants and site count.
- Exact command/UTC/output-hash records are `/var/tmp/gmsbank/command_records/calibration_{select,dump,ratios2,report}.json`. Large RGB8 and XYB plane dumps stay only under `/var/tmp/gmsbank/calibration/`.

## Limit

The selected TRAIN bank contains only medium and large images under the preregistered size cutoffs, so the tiny/small strata have n=0. The content and corpus coverage above is reported without inventing replacements or changing thresholds. The calibration measures unit conversion; it does not establish a human-quality benefit.
