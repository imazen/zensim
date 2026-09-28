# Rev4 baseline LODO status (2026-09-24)

**POTENTIAL — ceiling, not a model score.** The table below is a quarantined diagnostic transfer ceiling for the promoted Rev3 944-feature bank. It is never a shipped-model or held-out confirmation score.

## MISSING

- The H32/H128 MLP LODO rotations, incumbent/peer comparisons and JPEG-vs-other residual analysis have not run. The BVLS and lasso transfer matrices alone do not satisfy the full §3b protocol.
- MCL-JCI remains excluded pending D3. Candidate-family and GMSD arms wait for reviewed Part B sidecars. CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID TERMINAL, LIVE and secret holdouts remain unread.
- The source-level label orientation evidence and peer display configurations still need the datasets lane's independent checks before any decision-rule conclusion.

## R0 BVLS seven-fold transfer

Each row held out one D2 source dataset, trained on the other six with equal **total source weight**, and evaluated on the listed view. Each training source's target was transformed by its own training-row q0.001/q0.999 affine-to-[0,1] rule before pooling. The score and 95% CI are from owner `panel` SROCC with 2,000 reference-clustered resamples (seed 20260923).

| Held-out source | Evaluation view | Rows | Refs | Target units | Transfer SROCC | 95% CI |
|---|---|---:|---:|---|---:|---|
| KADID TRAIN | KADID TRAIN | 5,000 | 40 | human quality, native | 0.827519 | [0.802724, 0.853558] |
| TID2013 | TID2013 | 3,000 | 25 | human quality, native | 0.744398 | [0.713715, 0.776488] |
| KonFiG TRAIN | originsplit VAL | 436 | 8 | human-derived `1−q_jnd/3.2` | 0.808984 | [0.726607, 0.908598] |
| KonJND BPG TRAIN | BPG VAL | 2,020 | 101 | **SSIMULACRA2 oracle /100; not human accuracy** | 0.945197 | [0.937737, 0.952602] |
| CID22-A(25) | CID22-A(25) | 2,192 | 25 | **human MCOS/100; not an oracle** | 0.704333 | [0.648893, 0.762834] |
| AIC-3 CTC | AIC-3 CTC | 600 | 10 | CTC design-JND ladder, not MOS | 0.724077 | [0.656304, 0.809003] |
| KADID SELECT | KADID SELECT | 3,125 | 25 | human quality, native | 0.865404 | [0.847925, 0.882233] |

Command, from `/home/lilith/work/zen/zensim--featbank-potential` (2026-09-24 00:08:13–00:12:21 UTC, exit 0):

```bash
CARGO_TARGET_DIR=/var/tmp/rev4-featpot/target CARGO_HOME=/var/tmp/rev4-featpot/cargo_home ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- python scripts/rev4_featpot/lodo_bvls.py --arm r0 > /var/tmp/rev4-featpot/lodo_r0_bvls.log 2>&1
```

Actual final output: `{"result": "/var/tmp/rev4-featpot/lodo/LODO_r0_bvls/result.json", "sha256": "7d7d9427737d79e3d1cf522a04a4b86c15a4de21f4d633fdfc9cc7619b070af1"}`. The log SHA-256 is `d147d25911d30ec84d8538db6b087355342c2b691da9dc1ba3dd7b0c334aab59`; its seven `heldout` lines are the table's numerical source. All fit NPZ files and source Grams live only under `/var/tmp/rev4-featpot/lodo/LODO_r0_bvls/`.

CI command (2026-09-24 00:16:00–00:19:18 UTC, exit 0): `CARGO_TARGET_DIR=/var/tmp/rev4-featpot/target CARGO_HOME=/var/tmp/rev4-featpot/cargo_home ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- python scripts/rev4_featpot/lodo_ci.py --arm r0 > /var/tmp/rev4-featpot/lodo_r0_bvls_ci.log 2>&1`. Actual final output: `{"output": "/var/tmp/rev4-featpot/lodo/LODO_r0_bvls/ci.json", "sha256": "687231495ac2da11b26cdc7bf3cce34fa2f1b11037c27fba38f7cb3ee42d8e33"}`; log SHA-256 `51899dea1895d8dcb5161ca13fd63df27b1ede7b1216624c76a932aee51367f1`. Its seven `heldout` lines are the CI source.

The seven source populations are now LODO-exposed; KonFiG VAL and BPG VAL are LODO-evaluation-exposed. The exact exposure receipt is in `docs/DATA_SPLITS.md`. The BPG target is stored in SSIMULACRA2/100 units and can be negative; CID22-A is human MCOS/100. These raw units were never pooled directly.

## R0 lasso seven-fold transfer

The same seven folds use a six-source inner CV to select one point on the 50-λ path. The table reports SROCC and B=2,000 reference-clustered 95% intervals. The source-level target transform and equal-source weighting follow the BVLS rotation above.

| Held-out source | Evaluation view | Rows | Refs | Transfer SROCC | 95% CI |
|---|---|---:|---:|---:|---|
| KADID TRAIN | KADID TRAIN | 5,000 | 40 | 0.704278 | [0.678265, 0.731877] |
| TID2013 | TID2013 | 3,000 | 25 | 0.779976 | [0.753764, 0.804064] |
| KonFiG TRAIN | originsplit VAL | 436 | 8 | 0.835011 | [0.773241, 0.911077] |
| KonJND BPG TRAIN | BPG VAL (SSIMULACRA2 oracle/100) | 2,020 | 101 | 0.991330 | [0.989484, 0.992945] |
| CID22-A(25) | CID22-A(25) (human MCOS/100) | 2,192 | 25 | 0.830964 | [0.773269, 0.888730] |
| AIC-3 CTC | AIC-3 CTC (design-JND ladder) | 600 | 10 | 0.797790 | [0.704440, 0.904421] |
| KADID SELECT | KADID SELECT | 3,125 | 25 | 0.727378 | [0.694958, 0.760083] |

The lasso LODO command was `CARGO_TARGET_DIR=/var/tmp/rev4-featpot/target CARGO_HOME=/var/tmp/rev4-featpot/cargo_home ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- python scripts/rev4_featpot/lodo_lasso.py --arm r0 > /var/tmp/rev4-featpot/lodo_r0_lasso.log 2>&1` (exit 0). Its exact final output names `/var/tmp/rev4-featpot/lodo/LODO_r0_linear/result.json`, SHA-256 `0c01528806375c912ea25e9ed5eb4cd42d32b372d5980e5d93ae2b9c3807f109`; log SHA-256 `6d2981771c80209c17f5875bf83d8600fc650f16d005a3cb06114d1f888f89fd`. The CI command was the same heavy prefix with `python scripts/rev4_featpot/lodo_ci.py --arm r0 --model linear > /var/tmp/rev4-featpot/lodo_r0_lasso_ci.log 2>&1` (exit 0); actual final output names `ci.json`, SHA-256 `cbf0b3c701d1bc31c09bd9b974d5a2bb67d1a75d09a75927fff0f5ffd28fdcf2`; log SHA-256 `d3a0aebba55a55688363eb6deaa592e1a82d2433e26b3adfd3c6760264c6f29c`. Those seven `heldout` lines and the CI JSON are the table's numerical source.

The lasso `family_fold_frequency` recorded in the result is basic 0/7, peaks 4/7, masked/IW 1/7, v2 5/7, append 7/7 and append2 0/7. This records a selection pattern within the R0 transfer fit; it is not a candidate-family adoption result.

## R0 cross-evaluation matrices (completed 2026-09-24)
Rows are seven six-source fits; columns are the D2 evaluation views. Only the diagonal is held-out source transfer. Off-diagonal cells may evaluate on rows from a training source; the JSON receipts record `source_in_fit` and `eval_rows_seen_in_fit` separately. KonFiG and BPG columns use reference-disjoint VAL views. Values are signed owner `panel` SROCC; each of the 49 cells per model has a B=2,000 reference-clustered CI in the raw JSON. Different target units were never pooled in evaluation.

### BVLS matrix
| Fit without / eval | kadid_train | tid2013 | konfig_train | konjnd_bpg_train | cid22_a25 | aic3 | kadid_select |
|---|---:|---:|---:|---:|---:|---:|---:|
| kadid_train | 0.8275 | 0.8302 | 0.8494 | 0.9400 | 0.8711 | 0.8622 | 0.8891 |
| tid2013 | 0.9003 | 0.7444 | 0.8219 | 0.9382 | 0.8886 | 0.8485 | 0.9033 |
| konfig_train | 0.9091 | 0.8530 | 0.8090 | 0.9509 | 0.8845 | 0.8519 | 0.9108 |
| konjnd_bpg_train | 0.8979 | 0.8463 | 0.8627 | 0.9452 | 0.8674 | 0.8822 | 0.9094 |
| cid22_a25 | 0.9048 | 0.8468 | 0.8458 | 0.9219 | 0.7043 | 0.8574 | 0.9049 |
| aic3 | 0.9195 | 0.8735 | 0.8759 | 0.9575 | 0.8814 | 0.7241 | 0.9236 |
| kadid_select | 0.8877 | 0.8309 | 0.8547 | 0.9374 | 0.8756 | 0.8574 | 0.8654 |

Raw matrix: `/var/tmp/rev4-featpot/lodo/LODO_r0_bvls/transfer_matrix.json` SHA-256 `1e69ccee1f30922b6e34fceb9eadffe7691e58f835e2e702039f5f62a679e25e`; heavy log SHA-256 `251b255e7e5a5e59da2055127b158697b3840e104443f4ab7d0058c40785ccfd`. The heavy run ended with `rc=0 20s` and final output `"cells": 49`.

### lasso matrix
| Fit without / eval | kadid_train | tid2013 | konfig_train | konjnd_bpg_train | cid22_a25 | aic3 | kadid_select |
|---|---:|---:|---:|---:|---:|---:|---:|
| kadid_train | 0.7043 | 0.7200 | 0.7692 | 0.9876 | 0.7937 | 0.7947 | 0.7274 |
| tid2013 | 0.7914 | 0.7800 | 0.8255 | 0.9894 | 0.8368 | 0.7958 | 0.8037 |
| konfig_train | 0.8007 | 0.7836 | 0.8350 | 0.9895 | 0.8263 | 0.7981 | 0.8118 |
| konjnd_bpg_train | 0.8147 | 0.7902 | 0.8461 | 0.9913 | 0.8328 | 0.7788 | 0.8235 |
| cid22_a25 | 0.8339 | 0.7984 | 0.8641 | 0.9920 | 0.8310 | 0.7859 | 0.8403 |
| aic3 | 0.8554 | 0.8333 | 0.8779 | 0.9935 | 0.8574 | 0.7978 | 0.8601 |
| kadid_select | 0.7043 | 0.7200 | 0.7692 | 0.9876 | 0.7937 | 0.7947 | 0.7274 |

Raw matrix: `/var/tmp/rev4-featpot/lodo/LODO_r0_linear/transfer_matrix.json` SHA-256 `734926c31f2af9d9387fc5c39b7d0a4b97f5d096b43d831448412202e1211f36`; heavy log SHA-256 `2946574187cb51f49bdef5f755adf3a6522efda4459d9692a3bc278943dc31c4`. The heavy run ended with `rc=0 20s` and final output `"cells": 49`.
