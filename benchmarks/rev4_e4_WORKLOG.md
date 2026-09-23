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

## 4. Outcomes (after the prereg commit d50f365e)

```
ZEN_PANEL_BIN=/var/tmp/rev4-e4/bin/panel ESR_BIN=/var/tmp/rev4-e4/bin/ensemble_score_rows \
  nice -n 19 ionice -c 3 python3 scripts/rev4_e4_agreement.py outcomes
```
Extra binary: `bake_verdict` (primary checkout build, 2026-09-19) → `/var/tmp/rev4-e4/bin/bake_verdict`
`367d81a7133173a5c5f84a623a378d64dfd10841f8e2a36227d100845e4d1e5b`, used only for
`--print-features-root`.

- First attempt (the prereg's resolved-root-only rule, plus a stdout/stderr parse bug) resolved
  no root for 357 cells and was discarded unanalysed, as `/var/tmp/rev4-e4/outcomes_try1_unread.json`.
  Fixing the parse left 61 cells unresolvable: 37 had a cosmetic `regime: 720` on 944-input
  bakes, and 24 had no repro metadata.
- Final rule (deviation 1 in the record): candidate roots are tried in `eval_roots` order after
  the bake-resolved one. The first root whose TRAIN-role KADID SROCC reproduces the stored value
  to ≤ 1e-6 is used, and CID22-A is scored only there.
- CID22-A files: `ref_basename ∈ A(25)` via pyarrow `filters=` into
  `/var/tmp/rev4-e4/cid22A/<hash>_cid22A.parquet`, 2,192 rows, 25 refs. B rows are never handed out.

Output `/var/tmp/rev4-e4/outcomes.json` `c5f8fc4c1d2242486e631f5a8e9755d041ca77094c03028e464a56ff5bbc36a2`, last log line:
```
cells=426 kept=421 excluded=5 ['ADD156_safesyn_only_raw_lasso', 'bhdr_linear_shaped_cvvdpmix', 'cl_tfm_corruption_LQ_MLP_s13', 'v02_bvls_NO_shaping', 'v47_strict_QAT_native']
```
(all five are 720-width SECONDARY rows whose `@cur372` twin is kept). Recheck:
```
$ python3 -c "import json;d=json.load(open('/var/tmp/rev4-e4/outcomes.json'))['cells'];k=[r for r in d if 'excluded' not in r];print(len(d),len(k),max(r['kadid_abs_err'] for r in k),sorted({r['cid22A_n'] for r in k}))"
426 421 6.132225160992988e-08 [2192]
```
Root counts: ext944-canonical 355, 372-postC 29, 372 2026-08-30 26, foldapp2_views 5,
era2r4 3, 2026-05-15 2, r1b-pools944 1.

## 5. Analysis (preregistered)

```
ZEN_PANEL_BIN=/var/tmp/rev4-e4/bin/panel nice -n 19 ionice -c 3 python3 scripts/rev4_e4_agreement.py analyze
```
→ `/var/tmp/rev4-e4/analysis.json` `ee67d70224d71e7598391320f1169a7ca7be6fe2bc8b5d6d9f8c35fc92ea41df`
(plus `table_MAIN.json` `cae2fdeb…`). A rerun to `analysis_rerun.json` is byte-identical: `cmp` printed nothing, and the log says `IDENTICAL`.
Headline extraction and its actual output:
```
$ python3 -c "import json;a=json.load(open('/var/tmp/rev4-e4/analysis.json'));m=a['MAIN'];print(m['verdict']);[print(y,round(r['rho_C']['point'],3),round(r['partial_P_dis']['point'],3),[round(x,3) for x in r['partial_P_dis']['ci_bonf']],round(r['partial_P_bu']['point'],3),[round(x,3) for x in r['partial_P_bu']['ci95']],round(r['delta_Combo_minus_C']['point'],3),[round(x,3) for x in r['delta_Combo_minus_C']['ci95']]) for y,r in m['outcomes'].items()];print(m['gate_view']['tables'])"
['ADOPT, circularity unresolved', ['D2_floors'], []]
H1_cid22A 0.593 -0.199 [-0.434, 0.217] 0.004 [-0.228, 0.227] -0.229 [-0.401, -0.012]
H2_aic3 0.658 -0.167 [-0.553, 0.351] -0.052 [-0.302, 0.225] -0.235 [-0.506, 0.048]
H3_konjnd504 0.597 0.272 [-0.189, 0.537] 0.075 [-0.118, 0.271] 0.006 [-0.15, 0.162]
H4_csiq 0.46 0.408 [-0.049, 0.685] 0.029 [-0.211, 0.337] 0.128 [-0.15, 0.361]
D1_contract -0.38 0.159 [-0.083, 0.443] 0.446 [0.151, 0.561] 0.158 [-0.01, 0.38]
D2_floors 0.179 0.697 [0.359, 0.82] 0.231 [-0.122, 0.554] 0.401 [0.087, 0.605]
{'agreement P_dis > tau': {'reject_failed': 97, 'reject_ok': 76, 'keep_failed': 64, 'keep_ok': 122}, 'composite C_A < median': {'reject_failed': 115, 'reject_ok': 64, 'keep_failed': 46, 'keep_ok': 134}}
```
Columns: outcome, ρ(C_LOO,Y), partial P_dis, its Bonferroni CI, partial P_bu, its 95% CI,
Δ(Combo−C), its 95% CI.

## 6. Exploratory (not preregistered)

```
ZEN_PANEL_BIN=/var/tmp/rev4-e4/bin/panel nice -n 19 python3 scripts/rev4_e4_explore.py
```
→ `/var/tmp/rev4-e4/explore.json` `175198876d93df9e97d68d5e3e90bb255de579ce1a1fdf946ef027049570c6c5`.
Leave-one-lineage-out partial P_dis→D2 0.644–0.727; P_dis→CSIQ 0.126–0.504 (min when
sota944_C dropped); P_bu→D1 0.391–0.496. D1-only gate: P_dis>τ reject 28/46 failures, 145/313
non-failures; P_bu>median(0.0169) reject 45/46, 131/313. Encoder-attributed rungs per cell
0 / 3 / 15 (min / median / max).

## 7. Records

`benchmarks/rev4_e4_agreement_gate_2026-09-23.{md,json}` (JSON = compact copy of analysis.json +
explore.json, 19.7 KB).
