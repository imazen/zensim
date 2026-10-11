# Percentile-rule re-run of the recorded bake_compare decisions (2026-10-11)

**Question.** zenmetrics-stats (formerly zenstats) changed `bootstrap_ci_delta`,
and with it `decisive`, from order-statistic percentiles (`col[⌊0.025·n⌋]`,
`col[⌈0.975·n⌉ − 1]`) to Hyndman–Fan type 7 (linear interpolation, numpy's
default). This is the owner decision of 2026-10-11: one rule across the crate.
Does that change any recorded `bake_compare` decision?

**Answer: no.** Over 300 (pair × corpus × band-or-aggregate) cells, no agree
count and no decision differs between the two rules. No recorded decision is
borderline under the switch.

## Setup

- **Recorded reports.** 14 `bake_compare` reports from 2026-05-18, all with
  B = 1000, seed 42 and 10 bands (`benchmarks/bake_compare_*.md`,
  `exp_v22_hybrid_*`, `v24_*`). 10 can run. Four (`v24_hybrid_nin_s2_*`,
  `v24_hybrid_no_nin_s4_*`) used bakes in `/tmp` that no longer exist.
  The exposure-ledger entry for this run (`docs/DATA_SPLITS.md`, 2026-10-11)
  says 12 runnable; the correct count is 10.
- **Binaries.** One source tree (zensim main `8ef81d0f`), built twice:
  - against zenstats `3e16cb45`, the order-statistic rule
    (`bake_compare_orderstat`, sha256 `6bcbbc6d…`);
  - against zenmetrics `67b2ba74`, type 7 (`bake_compare_type7`, `4829d404…`).

  The bootstrap draws are identical (same RNG, seed and B); only the
  percentile summary differs.
- **Data, as registered in the exposure ledger.**
  - CID22 (4,292) and AIC-3 (600): read in full, as recorded.
  - KADID, TID and KonJND: **SELECT views only** (3,125 / 1,080 / 807 rows),
    because the full tables contain TERMINAL segments and no 372-column TRAIN
    view exists. Neither the full tables nor any `*_terminal_*` file was opened.
    The scratch feature root symlinked the SELECT views under the file names
    `bake_compare` expects.
- **Scripts.** `percentile_rule_rerun_2026-10-11/run.sh` (the 20 runs),
  `compare.py` (the comparison) and `pairs.json` (bake paths: recorded paths
  where they exist, otherwise `zensim-experimental/weights/<basename>`). Outputs,
  both binaries and SHA256SUMS are in
  `/mnt/v/output/zensim/percentile-rule-rerun-2026-10-11/` (15 MB).

## Results

From `percentile_rule_rerun_2026-10-11/compare_result.txt`:

- **Cells compared:** 300. **Cells whose agree counts or decision differ: 0.**
- **CI bounds that change sign relative to 0: 3.** Each is a lower bound that
  goes from exactly 0 to slightly negative (an outlier-ratio Δ in band B0 of
  KADID / TID, B5 of TID). A bound of exactly 0 never counted as excluding
  zero, so no vote changes.
- **Bound shift as a fraction of the CI width (1,781 intervals).**

  | bound | median | p99 | max |
  |---|---:|---:|---:|
  | lower | 0.0001 | 0.0008 | 0.025 |
  | upper | 0.0001 | 0.0014 | 0.0041 |

## Reproduction of the recorded reports (separate from the rule switch)

- **CID22 and AIC-3 SROCC for A and B reproduce exactly** in all 20 corpus
  aggregates.
- **Aggregate decisions match in 18 of 20.** The two that differ, under *both*
  rules:
  - `bake_compare_5grp_cvL05_vs_largeiwssim` on AIC-3: recorded "promising", now `Tied`;
  - `bake_compare_exmix3_vs_persample` on CID22: recorded "tied", now `PromisingNotDecisive`.

  PWRC was redefined in zenstats on 2026-05-26, after these reports. The
  recorded PWRC columns differ from the re-run's (for example CID22 PWRC_A
  0.9191 recorded vs 0.9780 now for exmix3), and PWRC feeds the decisive rule.
  That is the likely cause; it was not isolated.
- The KADID / TID / KonJND decisions are on SELECT rows only and are not
  comparable with the recorded full-table runs.

## Labels

- The 14 recorded reports were computed under the order-statistic rule and the
  2026-05-18 PWRC definition, and stay as recorded.
- The 10 re-run here show no decision that depends on the percentile rule.
- The 4 whose bakes are gone are not re-verified.
