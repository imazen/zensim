# E14 KADIS ordinal ladder table (pointer)

Teacher-free ordinal coverage leg for the featpot instrument (design log E14; `scripts/rev4_featpot/e14_kadis_ordinal.py`).
Not committed (99 MB); rebuilt deterministically from the pinned inputs below, packed into fit program v19 as
`data/e14/kadis_ordinal.parquet` (+ `.manifest.json`).

- File: `/var/tmp/rev4-featpot/v2c/e14/kadis_ordinal.parquet`, sha256 `b41d31519577c0ab64a92fe1fc157fd80fcd844be60c7f84b7b3f2c66334f4e6`
- 11,495 rows, 2,598 ladders (reference × type × sign), 24 KADIS types × 100 train references (`source_id % 10 < 8`,
  lowest sha256(source_filename) per type); 187 pixel-identical and 318 single-rung rows dropped
- Inputs: KADIS-700k GPU canonical `c9a6fd56…` (keys + `distorted_url`; persisted PNGs fetched from R2, never regenerated),
  references `/mnt/v/datasets/kadis700k/refs`
- Features: f0–f1824 from the r4 bank extractor (`/var/tmp/reextract/target/release/examples/extract_features_372col`,
  sha256 `8c6f4c03…`, zensim 259045b0 + era-label patch; ZENSIM_FORMULA_REV=4, sqrt root, era tiercanon_c3negfold,
  restore-cuts prefix,mapdev,z1max,gmsnative,dvifmgate, legacy-rgb8), feature_set_id `…#d57e9571` (= the v2c bank);
  f1825–f1852 NaN (texgain/satsign not extracted; v2_lodo_mlp refuses keep lists that read them with this leg)
- Target: −|dist_param|, ranked only within a ladder (`withinref,rank`)
- Rebuild: `python3 scripts/rev4_featpot/e14_kadis_ordinal.py select|extract|table --root /var/tmp/rev4-featpot/v2c`
