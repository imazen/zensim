# CANONTAB worklog — v2-canon wide tables + the confirmatory pipeline

Lane brief: `~/tmp/zensim-paper/rev4/CANONTAB_brief.md`. Bookmark `quarantine/claude/canontab` (local only). Times MT.
Scratch and outputs: `/var/tmp/canontab/`. Sealed labels (`/var/tmp/rev4-featbank/_sealed/`) were never opened, listed or copied.

## 2026-10-01 02:51 — start

- Workspace `zensim--canontab` from `main@origin` (`d029bebc`); no push, `main` not moved.
- Read: amendment R1–R3, `docs/DATA_SPLITS.md` roles, `v2_{common,wide,lodo_mlp,compare,tune}.py`, `restore_data.py`, the Rev4
  bank manifests (`/var/tmp/rev4-featbank-r4/<set>/_MANIFEST.json`: schema `rev4-featbank-r4-v1`, width 1825, float64,
  Rev4, era `tiercanon_c3negfold`, `feature_set_id …@w1825/tiercanon_c3negfold#d57e9571`, build `259045b0`, binary `8c6f4c03…`).
- Brief vs reality (recorded, not silently resolved):
  - "SSIMULACRA2 targets taken from the pinned R915 tables by pair_key": the R915 dedup tables (`…/recovery/dedup-tables/*.parquet`)
    carry `ref_basename, human_score, f0.., source_family, input_table_row` and **no `pair_key`** (`input_table_row` indexes the
    table itself, not the bank). The v2 builder already takes the targets from the old bank's `labels__ssim2_oracle.parquet`
    by `pair_key` under the committed teacher pin and uses the R915 tables only for the fit/dev reference split. v2c does the
    same, and `verify` checks the R915 tables' targets against the bank's per reference (gate `teacher`).
  - Human sources are stimulus-level (kadid_train has 5,000 stimuli on 4,880 keys). v2c takes the stimulus rows from
    `data.load` exactly as v2 does and attaches Rev4 features by `pair_key`; "row counts equal the Rev4 bank" is therefore
    checked as Σ `n_stimuli` over the non-identical Rev4 keys, and distinct-key order equals the Rev4 key order.
  - csiq is 865 rows in the bank (the brief says 866).

## Instrument finding (affects the confirmatory read, not the exploratory runs)

`lib.zen_stats` panel `srocc` is a MAGNITUDE (`srocc_signed` carries the sign); `v2_compare.py` reads `srocc`. The exploratory
instrument is therefore blind to target orientation: a model whose scores anti-correlate with the human label scores the same
as a correct one. `v2_confirm_read.py` reads `srocc_signed` against the quality-oriented label (declared orientation per set),
and a mutation test (orientation table forced to QUALITY) breaks the planted-effect test, as it must.

## 2026-10-01 03:30–04:05 — canon main tables built and gated

- SafeSyn Rev4 finished 03:30; `v2c_wide.py build --legs all --family main` → `/var/tmp/canontab/v2c` (98 s, peak RSS 5.5 GiB); confirm tables
  (all four variants, label-free); `verify`: all gates ok (`wide/verify.json`). SafeSyn fit 141,054 + dev 38,757 rows = Rev3 v2's counts.
- Equivalence check (one-off, code exercise with the OLD peers, output deleted): the aux builder reproduced Rev3 v2's aux
  `human_score, f944–f947` columns bit for bit for real/p1/p3 on kadid, aic3, cid22_a25 (same seeds, D7 permutation, R3 oracles).
- Gate cell: `v2_confirm_fit.py --spec r0 --head N --seed-index 0`, `ZENSIM_MAX_TIER=v3`, 120 epochs, 706 s, best epoch 28; run twice
  (`--dest` runA/runB): all six prediction files byte-identical; the bakes differ in 6 bytes (embedded output path and timestamp), so
  `selected_bake_sha256` differs while predictions do not.
- `v2_compare.py --calibration` re-run on a symlinked root: `calibration.json` byte-identical to the coordinator's.
- Coordinator decisions applied: matched null only (no `--eval-variant`), labels named by the pin, R2.1 primary test (short list + Holm).
