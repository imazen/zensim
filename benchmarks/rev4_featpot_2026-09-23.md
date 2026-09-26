# Rev4 feature-bank potential baseline status (quota stopped, 2026-09-24)

**POTENTIAL — ceiling, not a model score.** The preregistration is `d2169f5b9c69`; the promoted-bank/GMSD addendum was committed as `044f00dc043f` at 2026-09-23T22:59:18Z, before this resumed lane read any bank label value. The promoted Rev3 f32 bank at `/var/tmp/rev4-featbank/bank/` is the only feature input. Targets come exclusively from the role-allowed bank `labels__*.parquet` files through `scripts/rev4_featpot/data.py`; the adapter verifies pinned manifests and joins on `pair_key` plus `source_row_id`.

## MISSING

- **Erratum (2026-09-26, landing review):** the "2/480 registered fit cells" denominator in the first item below is an arithmetic error. The registered grid is **960** (8 sets × 2 arms × 2 heads × 6 views × 5 seeds), so the two results counted there are 2/960; the same applies to the "478/480" in the JSON companion. Nothing else in this record changes.
- The full H32/H128 non-negative-head 5×5 Latin baseline is still missing. Two H32 outer-fold/seed results (AIC-3 and KADID TRAIN) exist, 2/480 registered fit cells, with one outer-fold family-importance receipt. A qualified H128 two-epoch smoke passed but is not a 60-epoch result. The batch launcher stopped at the coordinator's quota gate.
- The quota gate stopped the CI, P0 sham and E1/pairwise runners between cells. At the stop: 21/32 deterministic cells have reference CIs, 14/16 shams are complete and exactly zero, and 17/32 E1/pairwise cells are complete. All eight 200-draw stability sets, all 16 paired positive controls and both 7×7 R0 matrices are complete. The frozen inventory is `/var/tmp/rev4-featpot/baseline_summary.json` (SHA-256 `20e41de1e613ae38b3d31e3f2da962862a68c8873470a2b2fbd16a5bb91202bd`).
- Incumbent/peer LODO rows and JPEG-vs-other residual analysis remain undone. MCL-JCI is excluded pending D3. C1–C4 and P1–P3 candidate arms wait for the later coordinator message and reviewed Part B sidecars.

## Measured so far

The 7-fold R0 BVLS and lasso LODO rotations, with 2,000 reference-clustered intervals, are recorded in `benchmarks/rev4_featpot_lodo_2026-09-23.md`. Their source models and raw prediction arrays remain under `/var/tmp/rev4-featpot/lodo/LODO_*`. The per-cell fit/CI, positive-control, sham, stability and E1 result receipts are under `/var/tmp/rev4-featpot/`, with source hashes and commands in `benchmarks/featbank-potential_WORKLOG.md` and its `2026-09-24` continuation. No fitted bake has entered `zensim/weights/` or the board.

The BPG target is **SSIMULACRA2/100 oracle units** (negative values are possible), not human accuracy. CID22-A(25) is **human MCOS/100**, not an oracle. CID22 TRAIN and SafeSyn raw 0–100 SSIMULACRA2 oracle columns are outside this baseline. No raw units are pooled without each source's fit-only q0.001/q0.999 transform.

The requested `/home/lilith/tmp/zensim-paper/rev4/FEATBANK_POTENTIAL_BASELINE_DONE.md` records this partial quota-stop result with MISSING work first. The exact resume commands are in the HANDOFF section of `benchmarks/featbank-potential_WORKLOG_2026-09-24b.md`. No featpot runner remains active.

## Deterministic R0 baseline (32 fit cells complete)
These are POTENTIAL diagnostics on the promoted Rev3 bank. Scores are owner `panel` SROCC, with reference-grouped nested CV. The gap is in-sample minus nested. All 32 stored fit scores, and all 14 LODO diagonal scores, were recomputed exactly from stored predictions and role-allowed bank labels by `scripts/rev4_featpot/recompute_scores.py` (log SHA-256 `dab84065cb7f33d580883314e6e524c6c29e47332aca2f64c3a0595280d06123`; exact final output `"audited_fit_cells": 32`). The BPG row measures an oracle target, not human accuracy. The CI receipts are still filling and remain in `/var/tmp/rev4-featpot/fits/`.
| Set | Rows / refs | BVLS nested | BVLS in-sample | Gap | Lasso nested | Lasso in-sample | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| kadid_train | 5,000 / 40 | 0.9307 | 0.9544 | +0.0237 | 0.9369 | 0.9493 | +0.0123 |
| tid2013 | 3,000 / 25 | 0.9326 | 0.9636 | +0.0309 | 0.9251 | 0.9258 | +0.0007 |
| konfig_train | 327 / 6 | 0.8415 | 0.9975 | +0.1560 | 0.7125 | 0.8573 | +0.1448 |
| konjnd_bpg_train | 8,060 / 403 | 0.9937 | 0.9957 | +0.0020 | 0.9960 | 0.9964 | +0.0004 |
| cid22_a25 | 2,192 / 25 | 0.8163 | 0.9814 | +0.1651 | 0.8720 | 0.9406 | +0.0685 |
| aic3 | 600 / 10 | 0.6379 | 0.9922 | +0.3543 | 0.7145 | 0.8543 | +0.1399 |
| kadid_select | 3,125 / 25 | 0.9110 | 0.9547 | +0.0437 | 0.9156 | 0.9255 | +0.0099 |
| konfig_val | 436 / 8 | 0.7520 | 0.9968 | +0.2448 | 0.7835 | 0.8593 | +0.0758 |

## R0-minus-basic positive controls (all 16 complete)
The paired delta is `R0 − (R0 minus f0..f227)` in nested SROCC, with B=2,000 reference-clustered bootstrap intervals. A CI excluding zero marks this particular instrument as sensitive. All other dataset/model pairs are insensitive; no candidate-family conclusion follows on them.
| Set | BVLS delta [95% CI] | Sensitive | Lasso delta [95% CI] | Sensitive |
|---|---:|:---:|---:|:---:|
| kadid_train | +0.0030 [+0.0002, +0.0065] | yes | +0.0040 [+0.0015, +0.0063] | yes |
| tid2013 | +0.0040 [-0.0024, +0.0146] | no | +0.0013 [-0.0010, +0.0035] | no |
| konfig_train | -0.0165 [-0.0507, +0.0163] | no | -0.0034 [-0.0115, +0.0251] | no |
| konjnd_bpg_train | +0.0002 [-0.0008, +0.0011] | no | +0.0000 [-0.0003, +0.0004] | no |
| cid22_a25 | -0.0066 [-0.0311, +0.0156] | no | +0.0029 [-0.0061, +0.0139] | no |
| aic3 | +0.0497 [-0.0311, +0.1324] | no | +0.0000 [+0.0000, +0.0000] | no |
| kadid_select | +0.0035 [-0.0016, +0.0083] | no | +0.0050 [+0.0007, +0.0090] | yes |
| konfig_val | +0.0475 [-0.0484, +0.1872] | no | -0.0061 [-0.0101, +0.0000] | no |
