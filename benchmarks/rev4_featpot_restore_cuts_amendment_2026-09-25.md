# Rev4 POTENTIAL amendment — restore-cuts arms (2026-09-25)

**POTENTIAL — ceiling, not a model score.** Supplements preregistration `d2169f5b`, addenda `044f00dc` / `bf553563`, the C1-C4 amendment, the cost amendment `61d3c07a050d` and the csfw/C7/B1 amendment `28daeec17d59`. Written at 2026-09-25T15:57:29Z by Claude Sonnet lane `featbank-potential`, **before this lane read any label for these arms and before any restore-cuts column value was opened**. The A1w drop rule below is fixed here, before anything is computed from the data.

## Authority and source

Coordinator instruction, 2026-09-25: fold the restore-cuts families (landed on zensim main: implementation `384d15e1`, ids pinned and exposure ledger `abe694a0`) into the candidate preregistration. Reviewed arm proposal: `benchmarks/rev4_restore_cuts_potential_prereg_proposal_2026-09-24.md` on main (sha256 `aaa0c239885d8afc08d33e57be19be675fa9d5e538a2eecd87a32da6fa4ada95`). Families, all default-off, layout width 1825 (registry `zensim/src/feature_defs.rs` at main `abe694a0`; slot id = `base + (scale*3+channel)*per + block_local` for per-channel families):

| family | ids | layout |
|---|---|---|
| mapdev | f1502-f1561 (60) | 12 cells (scale*3+channel) x 5 signals: 0 mse_dev, 1 hfsq_src_dev, 2 hfsq_dst_dev, 3 hfabs_src_dev, 4 hfabs_dst_dev |
| z1max | f1562-f1789 (228) | 12 cells x 19 signals: 0-12 the 13 BASIC (ssim_mean, ssim_4th, ssim_2nd, edge_art_{mean,4th,2nd}, edge_det_{mean,4th,2nd}, mse, var_loss, tex_loss, contrast_inc), 13-18 the 6 peaks (ssim_max, edge_art_max, edge_det_max, ssim_l8, edge_art_l8, edge_det_l8) |
| gmsnative | f1790-f1819 (30) | X cell 15 signals (f1790-f1804), then B cell 15 (f1805-f1819) |
| dvifmgate | f1820-f1824 (5) | f1gate_l0..l4 |

Sidecars: `/var/tmp/restore-cuts/bank/<set>/features__restore_{mapdev,z1max,gmsnative,dvifmgate}.parquet` with `_MANIFEST_restore.json` (feature_set_id, build commit `ec5b1821a4c6`, binary sha256 `4ea8f3331fd585ae5e82e71b773a67f7a02dd8548d89ec1320c01298ee025ed5`), joined **strictly by `pair_key`**; a missing or duplicate key fails the arm. The exact per-set file hashes are pinned in `benchmarks/rev4_featpot_restore_arms_2026-09-25.json` (committed before any label read).

## Arms (each on the same R0 = Rev3 944 baseline; every arm has a size-matched permuted control)

| arm | columns | control (size-matched) | inputs beyond restore sidecars |
|---|---|---|---|
| A1 | mapdev f1502-1561 (60) | the same 60, permuted | none |
| A1m | mapdev locals {0,1,2} in all 12 cells (36): ids 1502+5c+{0,1,2}, c=0..11 | the same 36, permuted | none |
| A1w | R0 minus 60 columns chosen by the **drop rule below**, plus mapdev (60) | the same drop, plus the 60 mapdev columns permuted | none |
| B2 | z1max f1562-1789 (228) | the same 228, permuted | none |
| B2m | z1max block-locals {3,4,5,6,7,8,9,13,14,15,16,17,18} in all 12 cells (156): ids 1562+19c+l (the reviewed corrected set: edge_art x5, edge_det x5, mse, ssim_max, ssim_l8) | the same 156, permuted | none |
| B1 | C7 f956-985 (30) + dvifmgate f1820-1824 (5) | the 5 gate columns permuted (C7 real) | C7 sidecar (Part B `features__csfw_dvifm.parquet`) |
| B1s | C7 with its five F1 columns (f956, f962, f968, f974, f980; `level*6+0`) replaced by dvifmgate f1820-1824 (30 columns) | the 5 replacement columns permuted | C7 sidecar |
| C8n | R0 + C8 (f1322-1501, revised) + gmsnative f1790-1819 (30) | the 30 gmsnative columns permuted (C8 real) | C8 sidecar (Part B) |
| ALL | A1 + B2 + B1 + C8n additions | every added family permuted as its own block (equal total width) | C7 and C8 sidecars |

**Availability:** A1, A1m, A1w, B2, B2m depend only on the restore sidecars and can run now. B1, B1s, C8n, ALL need Part B sidecars that do not yet exist; until they do they are **MISSING**, and no substitute is derived.

**Control construction (primary):** the arm's added columns are permuted jointly across pair rows within each reference group, moving rows by unique `pair_key` (collapsed stimulus copies stay together), seed 20260923, never across a fit/evaluation boundary — the same instrument as the C1-C4 and csfw/C7 amendments. The proposal's "independently permuted columns" variant is registered as a **secondary** control for the deterministic models (BVLS, lasso) only, reported next to the primary; it breaks within-family correlation, so it is a harsher null and does not replace the primary. Either control's incremental delta versus R0 must have a paired reference-clustered 95% CI including 0, else that dataset's instrument is declared insensitive for the family.

## A1w drop rule (fixed before any computation on the data; features only, no label)

"Weakest" is defined **label-free** so the arm cannot select on the labels it is evaluated against. Per fit population (each outer training fold's rows, the full in-sample fit rows, and each D2 fold's training rows) compute on the fit rows' R0 feature values only: (1) constant (zero-variance) R0 columns are dropped first, in ascending id order; (2) then repeatedly find the pair of remaining columns with the highest |Pearson r| and drop the member with the **higher column id**, until exactly 60 columns are dropped in total (ties broken by lower id pair). The dropped-id lists are computed from feature columns only, committed with their input hashes in `benchmarks/rev4_featpot_restore_arms_2026-09-25.json` before any label read, and reused for every inner fold, model and seed of that population. Meaning: A1w tests replacing the 60 most redundant R0 columns with mapdev at equal width (944).

## Everything else (unchanged and inherited)

- **Sets, folds, roles:** the same D1 in-sample sets and D2 LODO folds as the C1-C4 amendment; the forbidden and secret sets stay unread. The restore-cuts lane read no label; this lane reads labels of these arms only after the pin note is committed.
- **Models and statistics:** the 50-lambda lasso, sign-masked BVLS (sign masks from the registry `direction`; `Unsigned`/`Undeclared` columns are free), H32/H128 non-negative-head MLP, 5x5 Latin-square seeds, 5 outer/4 inner reference folds, 200 half-reference stability subsamples, `panel`/`zen_stats` as the only statistic owner. **Bootstrap:** 2,000 reference-clustered resamples, seed 20260923, as in the potential prereg — the proposal's 10,000 / seed 20260924 is NOT used, so restore arms stay comparable with every other arm in the run.
- **Decision bar:** D5 as amended: (1) nested-CV gain >= +0.005 SROCC with a paired reference-clustered 95% CI excluding zero on >= 2 human sets; (2) stability frequency >= 0.6; (4) no non-negative-head dial-contract regression. **No cost clause**; cost is measured and reported (`benchmarks/restore-cuts_cost_2026-09-24.md` on main), never a gate. A gain over the permuted control is necessary diagnostic evidence, not an adoption pass by itself.
- **Eras:** deterministic BVLS/lasso arms run locally (dev-box era of their own). **All MLP fits of these arms go to the fleet AVX2 era** with the other potential-run MLP grids; no restore-arm MLP cell runs on the dev box, and no dev-box AVX-512 cell enters any cross-arm comparison.
- **Cells for the fleet lane:** the arm/column/sidecar/drop-list spec is in `benchmarks/rev4_featpot_restore_arms_2026-09-25.json`; the per-arm MLP cell grid follows the existing potential grid (sets x H32/H128 x 5 outer + full view x 5 seeds, real arm plus control).
