# fairness_tiers_2026-09-04.tsv — pointer

Per-row fairness audit for the 2026-09-04 fair-gauntlet pass: every board cell -> tier
-> failing criteria -> seed-group `k` and its mean/spread/min/max -> matched registry
ids. **433 rows, 322,477 bytes** — over the 30 KB git limit, so it lives in block
storage with this tracked pointer (ML Data Pipeline Discipline §7b).

| | |
|---|---|
| path | `/mnt/v/output/zensim/reports/fairness_tiers_2026-09-04.tsv` |
| sha256 | `24029c180ffd0a2cff9a86dba45b00d462dd9b2ac6090d5bdac0372bf22b6965` |
| rows | 433 (one per `*.fulleval.json` on the board) |
| bytes | 322,477 |
| produced by | `bandwise_dashboard.py --fulleval-dir … --fairness-tsv <path>` (owner: `gauntlet.write_fairness_tsv`) |
| written from | the same `gauntlet.fairness_of` decisions the board renders, so the file and the page cannot disagree |
| record | `benchmarks/fair_gauntlet_2026-09-04.md` |
| browser | http://localhost:3300/zensim/reports/fairness_tiers_2026-09-04.tsv |

Columns: `name tier fails k seed_group composite composite_k_mean composite_k_spread
composite_k_min composite_k_max cid22_signed cid22_k_mean konjnd_signed konjnd_k_mean
sample_coverage curated regime gaddr_pass gaddr_fail annotations notes`.

`sample_coverage` reads **NOT MEASURED** on all 433 rows today — no bake carries
`zentrain.sample_coverage` yet (the ownerfix lane is adding it). It is never a zero.

Regenerate with the Reproduce block of the record. The TSV is derived, not primary: it
can be rebuilt from the fullevals + the annotations registry at any time.
