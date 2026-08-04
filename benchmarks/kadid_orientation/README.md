# KADID orientation — the measurements behind campaign APPENDIX F (2026-08-04)

Determination, method and decision rules: `benchmarks/sota944_campaign_2026-08-03.md`
**REGISTERED APPENDIX F** (registration) and **APPENDIX F RESULTS** (F.R1–F.R9).
Ledger row: `docs/DATASET_HISTORY.md` §3.20. Registry:
`benchmarks/eval_annotations.json` (`kadid-ext-root-inverted`,
`kadid-ext-trained-inverted-model`, `kadid-e1-gate-unsigned`).

**The finding in one line:** `ext720`/`ext924`/`ext944` `ext_kadid.parquet` store
`human_score = (5 − dmos)/4`, the inverse of the canonical `(dmos − 1)/4`, so every
KADID SROCC measured on an ext root is sign-flipped and every model trained on one
learned to rank KADID backwards. TID is clean on every root.

## Files

| file | what it is |
|---|---|
| `kadid_signed_all_2026-08-04.tsv` | every board fulleval (n=188): stored `rank.kadid.srocc_signed`, the value **vs TRUE quality**, the eval regime, and which KADID table the bake TRAINED on (parsed from the embedded `zentrain.repro.argv`). `NO-REPRO` = predates the repro mandate. This is the list `kadid-ext-trained-inverted-model` scopes over; regenerable by negating `rank.kadid.srocc_signed` in the stored fullevals. |
| `kadid_per_distortion_type_2026-08-04.txt` | T4 — signed SROCC vs true quality per KADID distortion type (all 25, n=405 each, full 10,125 rows) for `winner_dial`, shipped **B** and `H_co3abpg_s2507`. Forward pass: `predict_features_with_bake --bake-post raw`; statistics: `panel --batch --stats full`. The era models are positive on 25/25; the 944 incumbent is positive on exactly the 8 compression+noise types and negative on all 17 analytic types. |

## Re-running the gate

```sh
cargo build --release -p zensim-validate --bin panel
ZEN_PANEL_BIN=target/release/panel \
  python3 scripts/canonical_corpus/check_target_orientation.py --all-roots
```
Exit 1 while the three ext KADID tables remain unrebuilt. `SKIPPED` means *not checked*,
never *passed*.
