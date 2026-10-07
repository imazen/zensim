# E28 — the SSIMULACRA2 tuning recipe applied to by_v2fy (registered 2026-10-07 09:20 UTC, before any E28 cell exists)

Drafted by the SSIM2RECIPE lane (`~/tmp/zensim-paper/rev4/SSIM2RECIPE_DONE.md`, primary-source verification), reviewed and
registered by the coordinator with the amendments at the end. No E28 cell exists at registration time.

Owner question (2026-10-05, verbatim): "Ssim2 was trained on tid2013, kadid-1k, Konfig (f boosting), and teh CID22
training data, using nelder-mead simplex, minimzing mse and maximizing kendall and pearson correlation. Is that recipie
something we can do? Since we lack the CID22 train data, we can calculated ssim2 for it and use that instead."

Verified recipe (primary sources, see SSIM2RECIPE_DONE.md): 108 map weights + ~5 score-map constants tuned by
Nelder–Mead simplex on CID22-train + TID2013 + KADID-10k + KonFiG-IQA Experiment I **flicker-boosted** reconstruction;
objective = minimise MSE on the CID22 training set and maximise Kendall correlation with all four sets, plus Pearson
correlation **at a lower (unpublished) weight**; validation on the CID22 49-ref validation set. Our CID22-train leg
already carries our own SSIMULACRA2 labels (`labels__ssim2_oracle`, `peer_ssim2.implementation = "fast-ssim2"` in the
admission audit) — the owner's proposed substitution is already the standing teacher.

## Arms (everything not named is exactly the E24/E25 by_v2fy instrument)

Same 420-column by_v2fy read set (`sel:59f0bbc2f290`), same Rev5 tables, same five LODO folds
(kadid / tid2013 / konfig / cid22_a25 / aic3), same seeds 0–9, head N, recipe suffix `@h32:H128:cv16:cf98` where a
bake exists. Control: the E24 Rev5 by_v2fy cells, seed-paired (the E26/E27 precedent).

* `s2o` — **SSIM2 objective on the current data mix.** The R1 legs (safesyn + cid22 teachers, `withinref,both`;
  four-source human leg, `withinref,rank`) plus two new pooled terms on every leg whose target is on a shared
  cross-image scale (safesyn, cid22, kadid, tid): pooled cross-reference rank pairs (the group's `ref_ids` restriction
  lifted for a fixed share `p_pool` of its draws) and a pooled Pearson term `w_p · (1 − ρ)` evaluated on each draw
  batch. Fixed constants: `p_pool = 0.5`, `w_p = 0.5` (Pearson at the published "lower weight" relative to Kendall;
  no numeric weights were ever published, so these are our fixed choice, not a search).
* `s2m` — **SSIM2 data mix + objective.** As `s2o`, but the training legs are exactly SSIM2's four: kadid_train,
  tid2013, konfig originsplit_train, cid22 fit (ssim2-oracle). SafeSyn and cid22_a25/aic3 legs are dropped (they have
  no SSIM2-recipe counterpart); the human leg's rank supervision switches from within-reference-only to the pooled
  objective of `s2o` (SSIM2's Kendall/Pearson were pooled per dataset, not within-reference).
* `nm` — **SSIM2-style low-parameter head.** A grouped linear head: the 420 features reduced to a fixed ≤128-weight
  map (weights shared within each by_v2fy feature family, one free scale per family block, fixed from the family's
  declared structure — not fit per-feature), then the fixed two-stage score remap `s → 100·(1 − s)` with a 4-parameter
  monotone cubic fit alongside the weights. Fitted by Nelder–Mead simplex (Powell registered as the fallback optimiser
  if NM fails its convergence check within the cell budget) minimising `J = MSE(cid22_fit) + mean_legs(1 − τ) +
  0.5·mean_legs(1 − ρ)` over the `s2m` legs. This arm produces no bake; it is a POTENTIAL diagnostic like the
  BVLS/lasso lane (`lodo_bvls.py`), and its outputs are labelled POTENTIAL, never a model score.

KonFiG label caveat (registered, not silently fixed): our konfig tables carry the design grid `human_score = 1 −
q_jnd/3.2`, not the flicker-boosted reconstructed scale SSIM2 used (`scores.csv`'s `quality` column is DCR-derived —
verified `quality = 1 − 0.25·mean_dcr` to CSV precision — not the triplet reconstruction). The raw EXP_I F-condition
responses (87,580 triplets) and the authors' MATLAB reconstruction exist at `/mnt/v/dataset/konfig-iqa/`; porting the
reconstruction is a separate registered step if `s2m`/`nm` show the recipe helps. This deviation is declared in the
result either way.

## Decision rule (fixed before any fit)

Seed-paired over the 50 cells per bake arm:

1. **As good:** E21's rule against the control (signed mean Δ ≥ −0.002, every source Δ ≥ −0.005, W2 > −2 SE).
2. **Recipe signal:** pooled KROCC and PLCC against the held-out source's labels each improve over control by
   more than 2 SE in the mean over the five folds.
3. Adopt the passing arm with the larger mean signed Δ; if both pass, prefer `s2o` (keeps safesyn). If neither
   passes, the SSIM2 recipe does not transfer to by_v2fy and the result is recorded, not iterated, without a new
   registration. `nm` is never adopted — it only bounds what a ~100-parameter head can reach under the SSIM2
   objective; its fold SROCC/KROCC deltas vs control are reported.

Reported, not ruled on: per-source SROCC/KROCC/PLCC/MSE for every fold (the recipe's own metrics), external NITS /
LIVE / MCIQA seed-paired Δ, `mcljci_k40`-style KonFiG-contamination caveat retained for any MCL-JCI read (R7a),
pooled-vs-within-reference scatter geometry, and the `nm` arm's fitted weight table. Any change after a fit starts is
a new registration.

## Implementation owner map (no new trainer/scorer)

- `zensim-validate/src/mlp_train/mod.rs` — `GroupLossMode` and the pair-draw paths; the pooled terms land here
  (`ref_ids: None` cross-reference pairs already exist per group; the `p_pool` share and a differentiable pooled
  Pearson batch term are the additions).
- `zensim-validate/src/bin/zensim_mlp_train.rs` — `--group` spec parsing (the MODE field) and new flags
  (`--pooled-rank-share`, `--pooled-pearson-weight`), recorded in repro metadata.
- `zensim-train-core/src/stats.rs` / `zenmetrics/crates/zenstats/src/panel.rs` (`kendall_tau`) — the stat owners for
  ρ/τ inside the `nm` objective; a new `e28_simplex.py` driver computes `J` per fold in-process (never a per-eval
  `panel` subprocess).
- `scripts/rev4_featpot/v2_lodo_mlp.py` + `v2_common.py` — cell driver and leg tables; `s2m`'s leg list is a
  `SOURCES`/`TEACHERS` redeclaration, pinned by a new `benchmarks/e28_teacher_pin_*.json` before any cell runs.
- The `nm` fitter is new code under `scripts/rev4_featpot/` (scipy 1.18.1 `minimize`), mirroring
  `linear_probe.py`/`candidate_linear.py`'s admitted-table input path; features come only from the admitted
  `POT_*` tables, never the sealed bank label files.

## Coordinator amendments (part of the registration)

1. **Pooled pairs never cross legs.** `p_pool` cross-reference pairs and the pooled Pearson batch term are drawn and evaluated within a
   single leg (one dataset or teacher table), never across legs whose label scales differ. E27 showed that a pooled term against one
   label scale makes the model adopt that scale's cross-image calibration; within-leg pooling keeps each comparison on one scale.
2. **`nm` grouping is pinned before fitting.** The exact feature-to-weight grouping (≤128 weights, derived from the declared by_v2fy
   family structure), the remap parameterisation, the NM convergence check and the Powell fallback trigger are committed as
   `benchmarks/e28_nm_grouping_*.json` before any `nm` fit; no grouping or remap choice may be made after seeing any fold result.
3. **KonFiG deviation is declared.** E28 trains on the design-grid KonFiG label, not SSIM2's flicker-boosted reconstruction; the result
   is read as "SSIM2's objective and data mix on our features", not a reproduction of the SSIM2 fit.
4. Fleet execution uses the existing pipeline (pinned program/data packs, `jobset_caps.json` memory envelope, no `tail_trim`).

## Result (2026-10-07, appended after the registered assessment)

All 100 cells completed at the registered budget (independent audit 100/100). **Neither arm passes; adopted: none; the
by_v2fy control is retained.**

| Arm | Signed Δ (mean ± SE) | Worst source | Pooled KROCC Δ | Pooled PLCC Δ | As-good | Recipe signal |
|---|---|---|---|---|---|---|
| s2o | −0.0018 ± 0.0012 | tid2013 −0.0054 | −0.0027 ± 0.0016 | −0.0019 ± 0.0017 | no (tid2013 < −0.005) | no |
| s2m | −0.0360 ± 0.0033 | kadid −0.1416 | −0.0368 ± 0.0032 | −0.0333 ± 0.0045 | no | no |

Reading: adding SSIMULACRA2's pooled within-dataset Kendall/Pearson terms to the current mix (s2o) does not improve
pooled ranking and costs a little on TID2013; SSIMULACRA2's own data mix without SafeSyn and the coverage leg (s2m) is
much worse, mostly on KADID. Research-only (POTENTIAL); this run used the pre-D1 five-source exploratory design.
Records: `benchmarks/e28_result_summary_2026-10-07.json`, `benchmarks/e28_final_2026-10-07.pointer.md`.
