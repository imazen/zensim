# C8 chroma calibration — 2026-09-24 (quarantined)

Preregistered in commit9016272c before measuring; TRAIN pixels only, no human labels.
1520 pair/geometry records:652 native plus deterministically selected box-derived records.
28/32 selection strata filled; all tiny/small strata present, four CID22 large strata absent.
Empty eligible-site sets are omitted from that ratio only; no zero replacement. Y constants unchanged.

| Ratio | Valid records | Empty | p25 | p50 | p75 | Middle C |
|---|---:|---:|---:|---:|---:|---:|
| b_gradient | 1344 | 176 | 264.83148026426119 | 325.87881104709379 | 412.84240983917562 | 0.0013183046665446051 |
| b_value | 1520 | 0 | 15.004406857006508 | 26.65765722714572 | 74.781558547804536 | 0.77396038285061741 |
| x_gradient | 1428 | 92 | 266.94017191086641 | 371.45455482843846 | 705.57657463772625 | 0.00101465093400699 |
| x_value | 1344 | 176 | 306.09344748271388 | 608.05310492452668 | 2030.3898238778916 | 0.0014875777316638384 |

Each bank is middleC ×4^(k−2), k=0..4; source literals are in `zensim/src/gmsbank_constants.rs`.
The large B value stabilizer follows the frozen ratio method; it was not retuned after measurement.
Use the detailed preregistration for opponent coefficients, centering, floors, equal-record pooling, derivative geometry and source grouping.

Bulk: `/var/tmp/gmsd-chroma/c8/calibration/` (large planes/RGB: `/var/tmp/gmsd-chroma/remote-r5600g/` (14,566 files, 4.4 GB, copied from the former remote tree, which has since been removed) and its mirror on the tower at `output/zensim/gmsd-chroma-2026-09-24/remote-r5600g/` (sha256 spot-checked)).
`chroma_report.json` SHA256 `a6aecdf036d89b7afef10e9c8aaa6c935836fbc69ed12950e56f41eabcf9c633`.
`chroma_ratios.tsv` SHA256 `181882b37c3de040426be2d087224d1ec43ad459fa280a1d6b543faaef0702be`.
`selection.json` SHA256 `5472cfa075f1e6a75ba2d2463485999befa67cfdfc18346cc0ca89074669def7`.
Report owner: `scripts/gmsbank/calibration_report.py --chroma-dir /var/tmp/gmsd-chroma/c8/calibration`.
Producer: test-only `gmsbank_chroma_calibration_dump`, retained source snapshot `c8/src`; run and dependency provenance in lane worklog.
