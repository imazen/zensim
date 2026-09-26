# Rev4 feature-bank potential preregistration (2026-09-23)

**POTENTIAL — ceiling, not a model score.** This record is fixed before this lane reads any human-label values. It covers the baseline R0 run from existing f64 Rev3 caches. Candidate arms wait for `FEATBANK_EXTRACT_DONE.md` and review; MCL-JCI waits for `DATASETS_DONE.md` and D3.

## Inputs and admission

The following inventories were obtained by sha256 streaming and Parquet footer inspection (CSV newline count); no target column was decoded. Keys and labels are inline in these older files, and no f32 bank sidecar exists yet. Rows shown are whole-file counts, not admitted population counts. `/var/tmp/rev4-featpot/input_inventory.json` records the same full hashes and paths.

| Source | Rows | SHA-256 | Intended population / guard |
|---|---:|---|---|
| `ceiling/final/human_fit.parquet` | 7,000 | `da8fa67109527470f73152a46665b364f4418d5201083ad28c145b3b4ad51e58` | KADID TRAIN and TID; split by registered reference identity |
| `ceiling/final/human_dev.parquet` | 1,000 | `a0d4fe5dbd21062a13430505f096b4dd7f7780fc8f9306a24876c06a45c29282` | remaining KADID TRAIN refs; TRAIN role despite the historical `dev` filename |
| `rev3-public-human-eval/features-rev3/ext_kadid.parquet` | 3,125 | `3b6c2fc87409e5126e35d416ed7d8cc7fa21b35f49cacbff3b3d8bc1e9de8399` | KADID SELECT only |
| `rev3-public-human-eval/features-rev3/ext_cid22val.parquet` | 4,292 | `b97af5bc5364319c32dd9bb1a30577b89d1bc5fe3c32e58727ae51be9bc350c8` | mixed A/B file: admit **CID22-A(25) rows only** by a pinned ref allowlist before projecting `human_score`; never materialize B targets |
| `rev3-public-human-eval/features-rev3/ext_aic3.parquet` | 600 | `d091abdc7cf2fd54e643afc375d74f735480727b48437bfe4aa8074506cd142e` | AIC-3 CTC only |
| `rev3-public-human-eval/features-rev3/ext_konfig.parquet` | 436 | `d89aa8fdba2981fcd18fc04a3b2fc5747f1f79befda240197fd352690b0f483c` | suspected originsplit VAL; verify origin IDs before admission |
| `konfig944/build/konfig_944.csv` | 1,090 | `8b0f80f7ccf08eb8965b27df3cd56b29d6a64ccaa58bf581c69fa9423ca0b267` | candidate source for TRAIN origin view; require Rev3 identity/parity check |
| `ext944-dstact-run-2026-08-06/konjnd_bpg_train_944.csv` | 8,060 | `5fb3de72f1e6e24c35bf972840a9752f01cab8fc6dce2dc5f4f5eaf0baee9b5d` | KonJND BPG TRAIN only; require Rev3 identity/parity check and verify whether target is human or teacher |
| `ext944-dstact-run-2026-08-06/konjnd_bpg_val_944.csv` | 2,020 | `449b250db0c373ca81acabdb04944e11e9c264f31ac177953a7a46bcf22c2e80` | KonJND BPG internal reference validation only; same guards |

The two `ceiling/final` files reside under `/home/lilith/work/zensim-validation-2026-09-13/`. The `ext_*` Parquets reside under `/home/lilith/work/zensim-validation-2026-09-14/`. The CSVs reside under `/mnt/v/output/zensim/`. The f64 baseline identity is `basic+peaks+masked+iw+v2+append+append2@w944/ceiling_rev3#b782e349`, formula revision 3. Before pooling any CSV with these Parquets, verify its feature-set identity and arithmetic era, not just its width. Missing or incompatible populations become MISSING, not silently replaced. Additional input files require an appended prereg amendment committed before their first label read.

Data roles follow `docs/DATA_SPLITS.md` ruling of 2026-09-23. TRAIN: KADID TRAIN, TID2013, KonFiG originsplit_train, KonJND BPG half. D1 diagnostic fit: CID22-A(25), AIC-3 CTC, KADID SELECT, KonFiG originsplit_val. D2 provisional LODO folds: KADID, TID, KonFiG, KonJND-BPG, CID22-A, AIC-3, KADID SELECT. CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID TERMINAL, LIVE and secret holdouts are excluded. Every first held-out label read gets a preceding `docs/DATA_SPLITS.md` exposure receipt.

## Arms and estimands

R0 = existing Rev3 944; R0 − F for every existing ledger family F; R0 + each C1–C4; R0 + all C; basic228. Only R0 and its **R0 − basic** positive control run in this baseline phase; family permutations are negative controls as each family is admitted. Every result is an in-sample potential ceiling. Per (arm, model, dataset), report in-sample fit, 5-outer/4-inner reference-grouped nested CV, their gap, and LODO transfer where admitted. Folds are assigned by a fixed reference-hash order with seed 20260923; every variant of a reference remains in one fold. Inner tuning and standardization see fit-fold references only. No outer or LODO test target influences fitting, hyperparameters, transforms or checkpoint selection.

Models: `zensim_validate::gram_lasso` / `bake_dial_refit gram|fit-lasso` lasso path, 50 log-spaced lambdas from lambda-max to lambda-max*1e-4; `gram_lasso::box_cd` BVLS with `benchmarks/feature_sign_mask_2026-05-26.tsv` bounds; `zensim_mlp_train --nonneg-distance` H32 and H128, 60 epochs, inner-fold checkpoint selection. New IDs' BVLS sign masks come from registry `direction` and must be recorded in an amendment before candidate fits. Per-reference raw grams must sum to whole-set raw moments to f64 round-off before use. Lasso lambda uses the inner 1-SE rule; BVLS has fixed bounds. The existing `bake_dial_refit gram --target-minmax01` owner supplies per-dataset q0.001/q0.999 affine-to-[0,1] normalization, fitted on **training references of each fold only**. LODO gives each source dataset equal total weight; a held-out dataset's labels are used only for evaluation. Native orientation: quality increases for KADID, TID, KonFiG, CID22-A, AIC-3, KADID SELECT; audit against the independent orientation checks before LODO. KonJND BPG target type remains to be verified before inclusion.

## Statistics, seeds, controls

Report SROCC, KROCC, PLCC, within-reference pairwise ordering accuracy and E1-band tail SROCC via `panel` / `scripts/lib/zen_stats.py` (the existing statistics owner), never local reimplementations. Paired reference-clustered bootstrap: B=2,000, seed=20260923, using `panel --batch` resample manifests. Every cell reports rows, references, the CI and all per-seed values. Lasso/BVLS are deterministic, one fit per fold. Lasso stability uses 200 reference-level half-samples (seed 20260923), records path entry order and the lambda-1SE selected set; family frequency counts subsamples with at least one selected ID. MLP permutation importance uses outer folds only.

MLP init seeds: `1101, 1103, 1107, 1109, 1117`. Sample seeds: `101, 100000101, 200000101, 300000101, 400000101` (raw-stream offsets, not ordinary independent RNG seeds). For outer fold o=0..4 and replicate i=0..4, pair init[i] with sample[(i+o) mod 5], a 5x5 Latin square; in-sample uses o=0. Preflight all sample windows with `subset_sim --require-disjoint-sampler-windows` at the actual 60-epoch draw budget before training. Report mean/min/max and each seed value, with sample-order and initialization spreads separately; never best-of-k.

Negative control per family: within-reference permutation of that family's columns, matched in size; its paired delta CI must include 0. Positive control: R0 against R0 − basic (f0..f227 removed); its delta CI must exclude 0. A failed control makes that dataset **instrument insensitive**, with no family conclusion. Controls use the same folds, seeds and statistics as the corresponding arm.

**Decision rule (copied verbatim from plan §3):**

> 1. its nested-CV potential gain (R0 + C vs R0, or R0 vs R0 − F for existing families) has a paired
>    reference-clustered 95% CI excluding zero on **≥ 2 human sets**, and a point gain ≥ +0.005 SROCC;
> 2. stability-selection frequency ≥ **0.6** (preregistered) on those sets;
> 3. it is within its §2.4 budget, measured;
> 4. under the non-negative-distance MLP head, the dial-contract gates (monotone ladders, identity 100) do
>    not regress.

> A family that meets 1-3 only on TRAIN-role sets is "TRAIN-supported, unconfirmed".

All fitted artifacts go under `/var/tmp/rev4-featpot/fits/POT_*` or `/var/tmp/rev4-featpot/lodo/LODO_*`; no model is packed into `zensim/weights`, scored on the board, or used to choose a shipped recipe.
