# Rev4 E4 worklog (2026-09-23)

Every step: exact command, inputs with sha256, outputs with sha256, and the output lines each
number came from. Bulky intermediates live under `/var/tmp/rev4-e4/`.

## 0. Workspace

`cd ~/work/zen/zensim && jj workspace add ../zensim--rev4-e4 -r main@origin` → parent
`ba922d8390f54e63ed96c05d6d0c90dffb15842b`.

Binaries (a fresh `cargo build` was queued behind the shared heavy lock, which was held by
multi-hour jobs; per CLAUDE.md "reuse verified binaries … record their hashes"):
`/home/lilith/work/zen/zensim/target/release/{ensemble_score_rows,panel}` (built 2026-09-19)
copied to `/var/tmp/rev4-e4/bin/`:
```
5b454d0fc88cdac719aa580d2b5eea5bde24cb8fe743e73e061ff62f54b42c81  ensemble_score_rows
f11857c27563bfc0ca242d07ef52ba78526b9545d02a3ab9f0754e4a34d05688  panel
```
Validity of the scorer is established by exact reproduction (step 2), not by provenance.

## 1. Scoring input for the 944 ladder grid

`ensemble_score_rows` requires a `human_score` column; the 372 instrument ships a
`_dummytarget` copy, the 944 one does not. Built one (zeros, zstd — the binary is compiled
without snappy):
```
python3 (pyarrow): read dial_grid_944col_ladder.parquet (0e8e5fb7…), append human_score=0.0, write zstd
→ /var/tmp/rev4-e4/dial_grid_944col_ladder_dummytarget.parquet
4a9a3c521e2f9f9e1a3f65f898d6406535e01b5e29ec6dae5fc5a582f55a13d6
```

## 2. Ladder predictors (non-human)

```
ESR_BIN=/var/tmp/rev4-e4/bin/ensemble_score_rows nice -n 19 ionice -c 3 \
  python3 scripts/rev4_e4_agreement.py ladder --out /var/tmp/rev4-e4/ladder_predictors.json
```
Inputs: `cells.json` afeedacc…, `reference_truth_ladder_pnorm3.tsv` 257b124f…, the two grids
above, per-cell `gaddr/<name>.json` (C1 stored value). Output
`/var/tmp/rev4-e4/ladder_predictors.json` `8f845a59a277045f37baf981a9b6f63ec512d69f92b4e5973cd60fd77350ce3b`;
log `/var/tmp/rev4-e4/logs/ladder2.log`, last line:
```
cells=448 reproduced=448 not_reproduced=0 []
```
i.e. `1 − inv_dial/pairs` from the dumped per-cell scores equals the board's stored C1
(`mono_agree`, agree reading, pnorm3 @ 0.05) with |Δ| = 0 on every graded bake cell. Pair
counts on the instrument: 9,411 adjacent pairs, reference `unknown` 0, two-reference-agree
pairs 5,232, butteraugli-material 5,619, ssim2-material 7,416, agree-excluding-floor 4,878.

MAIN (944, "immune" era): 359 cells, 19 lineages. P_dis median 0.009938837920489297
(min 0, max 0.3572); P_dial median 0.01806. These set the prereg gate threshold τ.

## 3. Prereg + exposure ledger

Committed before any human-label statistic: `benchmarks/rev4_e4_prereg_2026-09-23.md`,
`docs/DATA_SPLITS.md` "Exposure ledger — 2026-09-23: rev4 step-1 e4".
