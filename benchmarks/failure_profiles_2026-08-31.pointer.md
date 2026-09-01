# Failure-profile artifacts (2026-08-31) — block-storage pointer

Block storage: `/mnt/v/output/zensim/failure-profiles-2026-08-31/` (236 MB — the
per-cell fresh verdicts under `zones/` dominate; never committed).

| file | what |
|---|---|
| `failure_inventory_2026-08-31.tsv` | 4,550 rows — per board cell x corpus: signed SROCC, per-reference mean, **frac_refs_backwards**, OR, Z-RMSE, PWRC, train==val, worst usable band + its n and span |
| `ladder_inversions_2026-08-31.tsv` | 7,728 rows — per board cell x split x zone: rung pairs, material inversions + rate, flat, codec-saturated, ladders, ladders with an inversion, **ladders ending backwards**, inversion magnitude med/max |
| `worst_ladders_2026-08-31.tsv` | 3,477 rows — the worst individual ladders **by reference image name** (image, codec, content class, zone, endpoint delta, deepest backwards step) |
| `dial_zones_measure_log.json` + `dial_zones_sweep.log` | as-run record: which dial grid reproduced each cell byte-identically, and the explicit reason for each of the 57 cells not measured |
| `nearlossless/` | the JXL q=99.9 rung probe — per-cell dial predictions for 5 models + the as-run script |
| `zones/` | the 322 fresh `bake_verdict --full-json` verdicts the graft copied `dial.zones` from |

Board snapshot before the pass:
`/mnt/v/output/zensim/reports/summer_gauntlet_pre_failprof_2026-08-31.html`;
fulleval snapshot + `SHA256SUMS`:
`/mnt/v/output/zensim/reports/fulleval-snapshots/pre-failure-profiles-2026-08-31/`.

sha256 of each table is in the directory's `_MANIFEST.json`. Method + findings:
[`failure_profiles_2026-08-31.md`](failure_profiles_2026-08-31.md).
