# fairness_tiers_2026-09-06.tsv — pointer

Re-issued fairness audit (board hygiene lane, 2026-09-06) over the CURRENT fulleval
population — the criteria are unchanged from `benchmarks/fair_gauntlet_2026-09-04.md`
§2 (`gauntlet.fairness_of`, mechanical, nothing re-derived); this is a re-run over more
rows, not a new mechanism. Record: [`fair_gauntlet_2026-09-06.md`](fair_gauntlet_2026-09-06.md).

| | |
|---|---|
| path | `/mnt/v/output/zensim/reports/fairness_tiers_2026-09-06.tsv` |
| sha256 | `eded6d3227e299d6276bd4559180d9a8ceb7b5a1e4751bdfb6d95a336d1886e7` |
| rows | 508 (one per `*.fulleval.json` on disk at generation time) |
| bytes | 902,912 |
| produced by | `bandwise_dashboard.py --fulleval-dir … --fair-only --fairness-tsv <path>` (owner: `gauntlet.write_fairness_tsv`, unchanged) |
| written from | the same `gauntlet.fairness_of` decisions the fair board renders, so the file and the page cannot disagree |
| browser | http://localhost:3300/zensim/reports/fairness_tiers_2026-09-06.tsv |

**Never overwrites `fairness_tiers_2026-09-04.tsv`** (untouched, sha256 unchanged,
mtime still 2026-09-05 19:58 — that file itself already grew from the 433 rows its own
`.md` documents to 479 on disk before this pass touched anything; flagged, not fixed,
in the record above). The immediate predecessor of this file
(`fairness_tiers_2026-09-06.tsv`, 481 rows, written 02:39 the same day by an earlier
pass) is preserved alongside this one as
`fairness_tiers_2026-09-06_pre508.tsv.bak` (renamed, not deleted, per the
never-delete-generated-data rule).

Columns: unchanged from the 2026-09-04 pointer — `name tier fails k seed_group
composite composite_k_mean composite_k_spread composite_k_min composite_k_max
cid22_signed cid22_k_mean konjnd_signed konjnd_k_mean sample_coverage curated regime
gaddr_pass gaddr_fail annotations notes`. `sample_coverage` still reads NOT MEASURED on
every row (unchanged since 2026-09-04 — the ownerfix lane has not landed it).

Regenerate with the Reproduce block of `fair_gauntlet_2026-09-04.md` (unchanged
commands), pointing `--fairness-tsv` at a freshly dated path.
