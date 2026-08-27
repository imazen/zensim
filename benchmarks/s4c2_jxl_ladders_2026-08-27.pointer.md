# S4+C2 jxl ladder target table (2026-08-27)

- **Data**: `/mnt/v/zen/zensim-training/s4c2-2026-08-27/jxl_ladders_9pt.parquet`
  (157,395 ladders; train 80,745 / val 48,370 / test 28,280) + `_MANIFEST.json`
  (input sha256s of the three zenjxl_lossy canonicals, flag histograms).
- **Generator**: `scripts/canonical_corpus/derive_jxl_ladders.py` (this repo;
  landed in commit b9d0b5ae — note 3dadcec3 is an EMPTY describe artifact whose
  message named this pointer before the file existed; this commit corrects it).
- **Shape**: per-(split, origin_id, variant_name, cell) 9-point q→zensim ladder;
  raw knots (q/score/bytes) as list columns; q_seed + local dscore/dlogq slope +
  flag at t∈{70,80,88}; monotonicity diagnostics (n_inversions, max_inversion).
- **Load-bearing honesty**: the source grid is 9 points, max q90 — the
  pre-registration's "q-dense" was WRONG. t88 is 45.6% `above` (unreachable by
  q90) — treat as CENSORED at fit, never extrapolate. t70 has 32,507 `below`
  (already ≥70 at q5: the easy-content class the prior exists to catch).
  Holdout (corpus9 / dial-39, id + dHash) is enforced at FIT, keyed on the
  carried origin_id / content_source / content_image_sha.
- **Ruling context**: plan doc "S4+C2 DESIGN RULING (2026-08-27)" — zensimA-proxy
  targets first; frozen proxy-gap gate (≤2 q-steps median, ≥90% slope-sign
  agreement) on the dial-39 probe before any fit ships.
