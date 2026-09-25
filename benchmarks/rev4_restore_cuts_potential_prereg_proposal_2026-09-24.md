# Restored-cut potential-arm preregistration proposal (2026-09-24)

**Proposal only for the `featbank-potential` owner.** This lane (`restore-cuts`) reads no human labels and fits
nothing. The coordinator must merge this into the potential preregistration BEFORE that owner reads labels for
these arms. Inputs: the promoted f32 bank `/var/tmp/rev4-featbank/bank/` (R0 = the bank's existing-family
columns; C7 f956..985 and the Part B C1..C4/C8 sidecars are joined the same way) plus this lane's sidecars
`features__restore_{mapdev,z1max,gmsnative,dvifmgate}.parquet` (written under
`/var/tmp/restore-cuts/bank/<set>/`, manifests `_MANIFEST_restore.json`), joined strictly by `pair_key`. Producer
commit, binary sha256 and feature-set identity are in each manifest and must be pinned by the potential owner;
a missing or duplicate join fails.

The cost of a family is **measured and reported, never a gate** (user ruling 2026-09-24; the D5 clause "benefit
pays its measured runtime" is superseded by the 09-24 amendment). Cost numbers: `benchmarks/restore-cuts_cost_2026-09-24.md`.

## Arms (each on the same R0 baseline, each with a size-matched independently permuted control)

| Arm | Columns added to R0 | Size-matched negative control | What it tests |
|---|---|---|---|
| R0 | none | baseline only | reference |
| A1 | `mapdev` f1502..1561 (60) | 60 independently permuted columns | COST_CUTS_AUDIT A1: map-deviation columns, judged on human-label bars (the earlier rejection was judged on an SSIMULACRA2-dominated aggregate) |
| A1m | `mapdev` slots `mse_dev`, `hfsq_src_dev`, `hfsq_dst_dev` only (36) | 36 permuted | the record's named "MSE and HF-energy" subset, no HF-magnitude maps |
| A1w | width-matched replacement: R0 with 60 of its weakest-by-prereg-rule columns dropped, +`mapdev` | the same, permuted | the width-matched replacement arm the gmsd record skipped (the potential owner fixes the drop rule before reading labels) |
| B2 | `z1max` f1562..1789 (228) | 228 permuted | ungated 5x5 block-max pooling as the two-surface hybrid: R0's global 228 surface + block-max 228 surface |
| B2m | `z1max` slots pooled from the codec-sensitive maps only (`edge_art_*`, `edge_det_*`, `mse`, and the `ssim_*` peaks; 12 cells x 12) | 144 permuted | tests whether the record's codec/human concentration survives a narrower block |
| B1 | C7 f956..985 (curve, already in the bank) + `dvifmgate` f1820..1824 (5) | 5 permuted | curve vs gate form: the two forms share F2; only F1 differs |
| B1s | C7 with its five F1 columns REPLACED by the five gate F1 columns (30 columns total) | 5 permuted replacements | pure form swap at equal width |
| C8n | R0 + C8 (f1322..1501, revised) + `gmsnative` f1790..1819 (30) | 30 permuted | whether dropping native X/B gradient slots lost anything |
| ALL | A1 + B2 + B1 + C8n additions | equal-width permuted control | complementarity |

Permutations are made inside the same TRAIN fold and reference-group protocol as the potential owner, seed
`20260923`, never across train/evaluation boundaries; the control keeps column count and marginal distributions.
Use the existing `zen_stats` or `panel`/`bake_verdict` statistic owner for SROCC, ties and reference-clustered
bootstrap CIs (10,000 resamples, seed `20260924`, the owner's percentile-CI convention). Do not write a new
statistic implementation.

## Data roles

Unchanged from `docs/DATA_SPLITS.md` (rulings D1/D2/D4, 2026-09-23): D1 in-sample potential sets, D2 LODO folds,
confirmation sets pixels-only. The restore-cuts lane read no label of any set.

## Decision rule

The D5 adoption bar minus the superseded runtime clause: stability-selection frequency >= 0.6; nested-CV SROCC gain
>= +0.005 with a reference-clustered CI excluding zero on at least two human sets; the non-negative-distance head
preserves the dial contract. A gain over the permuted control is necessary diagnostic evidence, not alone an
adoption pass. Report each arm's SROCC per leg (human legs first), with the C8 and cost columns beside them.
No interpretation is a claim until the potential owner runs the preregistered analysis and its review promotes it.
