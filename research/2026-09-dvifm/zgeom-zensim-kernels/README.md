# zgeom-zensim-kernels — box vs binomial pyramid, and 5x5 block-peak pooling, on zensim's OWN 228 features

## What was asked

Brief: `../briefs/lane_zgeom_prompt.md` (+ `lane_zgeom_plan_copy.md`). Two
questions left open by the transplant lane, run on the same joint-core-v2
corpus (105,614 pairs) with the same paired-seed + permuted-control
discipline:

- **Z2** — replace zensim's own 2x2 box-filter pyramid decimate with
  binomial `[1,2,1]` / `[1,3,3,1]` kernels, on zensim's own 228-column
  `basic+peaks` surface. Measure dev accuracy (5 seeds), three label-free
  stability gates (1px shift, codec phase 0-7, crop-vs-full), and extraction
  cost.
- **Z1** — replace global-moment pooling with 5x5 block-peak x two-state-gate
  pooling, at matched column count (228), vs an ungated block-peak
  decomposition arm and row-permuted controls.

## Verdict (from `../reports/LANE_ZGEOM_DONE.md` — copied verbatim, this is the long one, read it in full before citing a number)

| question | measured | verdict |
|---|---|---|
| Z2 dev accuracy | bin121 −0.0023 (4/5 neg, t=−3.26); bin1331 −0.0013 (**5/5 neg**, t=−3.70) vs box2 | **NO adoption case — box2 stays** |
| Z2 stability | binomials strictly MORE shift-stable (worst-feature \|Δln\| 2.53→0.71/0.70); zero fails on any gate | trades accuracy for shift-stability, doesn't rescue the loss |
| Z2 cost | +48%/+65% β (t1), 2.2x/2.6x (t8) | **UPPER BOUND, not a fair comparison** — box2 dispatches through production AVX-512 `incant!`; the binomial arms are unSIMD'd scalar research code. Direction (binomial costs more) is robust; magnitude is not. |
| Z1 replacement (gate) | z1gate −0.1808 vs box2 (5/5 neg, t=−66.6) | **catastrophic — NOT a better pooling.** The p10 two-state gate admits ~10% of blocks; the masked-off 90% holds most of the distortion signal. |
| Z1 gate mechanism | z1gate − z1gateperm = +0.678 (5/5) | block signal is real; the gate is what kills it |
| Z1 ungated block-peak | z1max +0.0012 vs box2 (4/5 pos, t=+1.60), wins concentrated on codec (+0.0139, 5/5) and human (+0.0126) legs | **mild positive — a mechanism finding, not yet an adoption case (n=5 seeds)** |

Four-line verdict from the report: (1) no binomial-downsampler adoption case;
(2) gated block pooling rejected, ungated block-peak is a real small
positive; (3) next step would be a decision lane on `z1max` alone with more
seeds plus a block+global hybrid; (4) **nothing ships** — `zgeom` stays
research-only behind `feature-regime-v2`.

## Code in this directory

| file | role |
|---|---|
| `build_tables_z2.py`, `extract_z2.sh` | box2/bin121/bin1331 kernel extraction over joint-core-v2 |
| `run_fits_z2.py` | Z2 paired-seed fit driver |
| `build_tables_z1.py`, `extract_z1.sh` | z1max/z1gate (+ permuted controls) block-pooling extraction |
| `run_fits_zgeom.py`, `drive_batch.sh` | shared paired-seed fit driver/batcher for both Z1 and Z2 |
| `score_fits_zgeom.py` | parses best-val geomean3 SROCC from all 35 fit logs |
| `z1_c0.py` | derives the p10/p25/p50 gate threshold (`c0`) per channel x scale from TRAIN |
| `analyze_cost.py` | the `α + β·pixels` cost fit (with the fairness caveat documented in the report — box2 is SIMD, binomial arms are scalar) |
| `analyze_stability.py` | the 1px-shift / codec-phase / crop-vs-full stability gate analysis (192 pairs x 11 transforms x 3 kernels) |
| `run_measurements.sh` | top-level driver tying extraction → fits → cost → stability together |

## Rust / repo state

Same workspace and same commit as `transplant-x4x5x7`: `zensim--transplant`,
commit `syysxxqu` / `59cea473`. Adds `zensim/src/zgeom.rs` (new),
`zensim-bench/examples/zgeom_cost.rs` + `zgeom_stability.rs` (new). Working
copy clean as of this preservation pass (see `transplant-x4x5x7/README.md`
for the full at-risk status — there is none).

## How to re-run

1. `extract_z2.sh` / `build_tables_z2.py` extract the box2/bin121/bin1331
   arms; `extract_z1.sh` / `build_tables_z1.py` extract z1max/z1gate/
   z1maxperm/z1gateperm — all over joint-core-v2 (100,997 train-leg rows),
   at formula rev 3, identical row order across arms.
2. `z1_c0.py` derives the gate threshold from TRAIN before the z1gate
   extraction.
3. `run_fits_z2.py` / `run_fits_zgeom.py` (via `drive_batch.sh`) run the 35
   total fits (5 seeds x 7 arms) — see `../reports/LANE_ZGEOM_DONE.md` for
   the exact per-arm/per-seed tables.
4. `score_fits_zgeom.py` parses best-val geomean3 SROCC from every fit log.
5. `analyze_cost.py` fits `α + β·pixels` per kernel at 1/8 threads — **read
   the fairness caveat in the report before citing these numbers**: box2 is
   production SIMD, the binomial kernels are scalar research code with
   per-tap `reflect_101` and per-decimate heap scratch.
6. `analyze_stability.py` runs the three label-free stability gates.

Inputs consumed: joint-core-v2 (`../transplant-x4x5x7/README.md`'s
artifact), train-leg rows only; HDR rows (4,290) intentionally excluded
(SDR-only path); `cid22b_unsealed` (holdout) never touched.

## Artifacts (reference by path)

- `/mnt/v/output/zensim/zgeom-2026-09-21/` — `cost/`, `dev/`, `extract/`
  (18 CSVs + manifests), `features/`, `fits/` (incl. `runs{,_z2,_z1}.json`,
  `logs_{z2,z1}/*.log`, and `logs_{z2,z1}/scripts/*.sh` — the generated
  per-{arm,seed} invocation scripts, not copied here), `perm/`, `reports/`
  (`fit_results.json`), `specs/c0_z1.json`, `stability/` (`stability.csv`,
  `stability_gates.json`)
- Committed benchmark record (already permanent in the repo, not copied
  here): `benchmarks/zgeom_2026-09-21.{md,json}`, and the registry entries
  in `benchmarks/feature_sets_registry.json` (5 new `zgeom_*` eras/sets,
  append-only)
