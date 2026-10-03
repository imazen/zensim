# E15 coverage pool (pointer)

Teacher-free ordinal coverage ladders for the featpot instrument (design log E15; `scripts/rev4_featpot/e15_coverage.py`).
Not committed (370 MB); rebuilt deterministically; packed into fit program v21 as `data/e15/coverage_pool.parquet`
(+ `.keys.parquet`, `.parquet.manifest.json`).

- Pool: `/var/tmp/rev4-featpot/v2c/e15/coverage_pool.parquet`, sha256 `6b00349c8aca6613aeb1591f8411e738e3c70798274c7df9017dfbc8844848b3`; keys sha256 `bc225a115ab8505738a5c17ced6d4fc592a9a38ac6d9ec98661f4e8f0898addf`
- 42,021 rows, 9,594 ladders; 702 pixel-identical and 1,277 single-rung rows dropped
- KADIS train references (`source_id % 10 < 8`): the 20 rule-compliant types × 400 (lowest sha256(source_filename) per type;
  E14's 100 are a nested subset); excluded 6/9/10/15 (third-party generated). Persisted distortions from R2, never regenerated.
- Our numpy TID-style types on 400 further train references each: `lbw` local block-wise (32 fixed blocks per reference, level
  L applies the first 2/4/8/16/32; uniform blocks at local mean ± 32–96), `cab` lateral chromatic aberration (red right, blue
  left by 1/2/3/5/8 px, edge-replicated). Written as PNG.
- Families (rows): noise 8,000 · spatial 8,000 · blur 5,643 · light 5,600 · colour 5,178 · new 4,000 · contrast 3,600 · quantize 2,000
- Features: f0–f1824, the r4 bank extractor and arguments (as E14), feature_set_id `…#d57e9571`; f1825+ NaN (refused on read)
- Rebuild: `python3 scripts/rev4_featpot/e15_coverage.py select|generate|extract|table --root /var/tmp/rev4-featpot/v2c`
