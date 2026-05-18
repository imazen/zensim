# EX-MIX3 — 3-way (cv + iw + sm) target mix supervision — methodology + results

**Date:** 2026-05-18
**Status:** IN-FLIGHT (15-job train matrix running)
**Branch:** `feat/ex-mix3-target`
**Workspace:** `/home/lilith/work/zen/zensim--ex-mix3`

## Hypothesis (Step 1 of principled experiment workflow)

1. **Hypothesis**: Adding ssim2_log_norm to the cvvdp+iwssim training-target mix
   produces a bake whose output is less ssim2-shape-biased than V_22-LARGE+iwssim
   (training on `mix_cv40_iw60` = 0.4·cvvdp + 0.6·iwssim) while preserving full
   Mohammadi panel agreement on CID22/KADID/TID/KonJND/AIC-3. Per CLAUDE.md
   "SROCC-only verdicts BANNED + ssim2-target training bias" (2026-05-15): a
   target that mixes three independent metrics should produce a more general
   surface than any 2-metric blend.

2. **Falsification**: If all 3 target variants fail to Pareto-beat V_22 noLARGE
   on ≥4-of-6 panel-stats agreement across CID22 + KADID + TID + KonJND + AIC-3,
   the 3-way blend direction is dead. Sub-falsification: if ssim2 contribution
   produces ssim2-favored SROCC on CID22 (the documented bias), the verdict
   stands but cite the bias explicitly.

3. **Cost ceiling**: 3 variants × 5 seeds × ~50 min ≈ 12.5 h serial (3.5 h at
   concurrency=2 + EX-DUAL co-tenant on box). Budget 4 h wall.

4. **Ship form**: If a variant Pareto-wins, repack via `zenpredict repack`
   (i8 + zerobias + lz4) — DO NOT bump crate version (per user direction
   2026-05-17). Add to candidate pool; next bake that beats it ships first.

## Reporting panel (Step 2)

| Corpus | Role | When inspected |
|---|---|---|
| safesyn (training) | Training group | continuously (loss/val_srocc each epoch) |
| KADID (train+val) | Training + held-out validation | continuously |
| TID (train+val) | Training + held-out validation | continuously |
| CID22 | Gold-standard validation | END of run only (per CID22-mid-experiment-leak rule) |
| KonJND-1k | Held-out PJND anchor | END of run only |
| AIC-3 | Held-out low-q anchor | END of run only |

Stat panel: Mohammadi 2025 full set (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE)
at aggregate + 10-band level. Bake-compare § A.9 decisive rule per corpus.

## Coverage gate (input data audit)

ssim2 coverage check across canonical training parquets at
`/mnt/v/zen/zensim-training/canonical-2026-05-18/train/`:

| Corpus | Rows | ssim2_log_norm coverage | Triple (cv+iw+sm) coverage | Action |
|---|--:|---:|---:|---|
| safesyn | 196,086 | 100.0% | 100.0% | KEEP |
| kadid | 10,125 | 100.0% | 100.0% | KEEP |
| tid | 3,000 | 100.0% | 100.0% | KEEP |
| konjnd-dense | 20,160 | **0.0%** | 0.0% | **DROP** (no ssim2 scores) |
| cvvdp_iwssim_LARGE | 73,300 | **0.0%** | 0.0% | **DROP** (sidecar doesn't cover these rows) |

Coverage gate: <50% triple-coverage → DROP (not zero-fill; zero-fill is a
falsification antipattern per CLAUDE.md). EX-MIX3 trains on 3 groups
(safesyn + kadid + tid), 209,211 total rows. The V_22-LARGE+iwssim baseline
trained on 5 groups (282,519 rows including LARGE + konjnd). The closer
apples-to-apples baseline is V_22 noLARGE 372-feat (4 groups, 210,219 rows
— EX-MIX3 differs only by dropping konjnd which was at weight 0.02).

## Mix targets (3 variants)

All computed from existing `cvvdp_log_norm`, `iwssim_log_norm`,
`ssim2_log_norm` columns (all are safesyn-anchored normalized, range ≈ 0..100):

| Variant | Formula |
|---|---|
| `mix_cv33_iw33_sm33` | `(1/3)·cvvdp_log_norm + (1/3)·iwssim_log_norm + (1/3)·ssim2_log_norm` |
| `mix_cv30_iw40_sm30` | `0.30·cvvdp + 0.40·iwssim + 0.30·ssim2` (slight iw bias, matches V_22 preference) |
| `mix_cv40_iw40_sm20` | `0.40·cvvdp + 0.40·iwssim + 0.20·ssim2` (conservative ssim2 add) |

Build script: `scripts/v_next/add_mix3_target.py` → emits parquets at
`/mnt/v/zen/zensim-training/2026-05-18-mix3/`.

Schema preserved: 372 features (f0..f371), all 4 target columns added
(mix_cv33_iw33_sm33 replaces the nullable canonical column; the 2 variant
columns appended).

### Statistics of new columns (safesyn / kadid / tid)

| corpus | column | range | mean |
|---|---|---|---|
| safesyn | mix_cv33_iw33_sm33 | [0.672, 99.487] | 44.007 |
| safesyn | mix_cv30_iw40_sm30 | [0.806, 99.539] | 43.073 |
| safesyn | mix_cv40_iw40_sm20 | [0.806, 99.692] | 38.899 |
| kadid | mix_cv33_iw33_sm33 | [33.724, 76.493] | 41.262 |
| kadid | mix_cv30_iw40_sm30 | [30.499, 71.792] | 37.811 |
| kadid | mix_cv40_iw40_sm20 | [20.742, 71.793] | 29.528 |
| tid | mix_cv33_iw33_sm33 | [24.868, 63.157] | 39.292 |
| tid | mix_cv30_iw40_sm30 | [22.464, 57.658] | 35.891 |
| tid | mix_cv40_iw40_sm20 | [15.314, 59.528] | 31.646 |

The KADID/TID ranges sit higher than safesyn because the safesyn-anchored
normalization of ssim2_log_norm assumes safesyn's score distribution. RankNet
is rank-only, so the absolute scale shift doesn't matter for training — only
the per-group rank ordering does.

## Training recipe

Per (variant, seed): one `zensim_mlp_train` invocation. All hyperparams match
V_22 noLARGE 372-feat verbatim except `--target-column`:

| Setting | Value |
|---|---|
| Groups | `safesyn:safesyn.parquet:1.0:0.0`, `kadid:kadid.parquet:0.3:1.0`, `tid:tid.parquet:0.3:1.0` |
| Hidden | 128 |
| Epochs | 300 (no early-stop) |
| pairs-per-epoch | 50,000 |
| lr | 1e-3 cosine to 0, period 50 |
| L2 | 1e-5 |
| leaky-α | 0.01 |
| val-policy | min |
| max-features | 372 |
| minibatch-size | 256 |
| PWRC | on, sensory threshold 5.0 |
| NiN | β=0.1, p=1.0, q=2.0 |
| target-scale | 100.0 (matches V_22 noLARGE recipe — rank loss is scale-invariant) |
| out-dtype | f32 |

Seeds: 1, 2, 3, 4, 5. Variants: cv33_iw33_sm33, cv30_iw40_sm30, cv40_iw40_sm20.

Trainer: `/home/lilith/work/zen/zensim/target/release/zensim_mlp_train`
(prebuilt; build inspection deferred but signature matches V_22 noLARGE).

## Per-pair / per-ref audit

ssim2_log_norm here is used as a TARGET column (single scalar regression target,
contributes to MOS comparisons), NOT as a feature. The per-ref-features-are-noise
antipattern (memory: feedback_per_ref_features_are_noise.md) does NOT apply.

## Baselines for bake_compare

- **V_22 noLARGE 372-feat seed=3**: `/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18/v22_372feat_noLARGE_s3_h128.bin`
  Closest apples-to-apples comparator (4 groups, 372 features, mix_cv40_iw60).
- **V_22-LARGE+iwssim 300-feat seed=3**: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128.bin`
  Current strongest mix bake (5 groups, 300 features). EX-MIX3 is 372-feat —
  bake_compare across feature widths may not work; fallback is bake_verdict
  side-by-side.

## Results

### 5-seed CI (TBD — fills after training matrix completes)

| Variant | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---|---|---|---|---|
| V_22 noLARGE s3 (baseline, single seed) | 0.8558 | 0.9336 | 0.8904 | 0.8369 | 0.8107 |
| EX-MIX3 cv33_iw33_sm33 | TBD | TBD | TBD | TBD | TBD |
| EX-MIX3 cv30_iw40_sm30 | TBD | TBD | TBD | TBD | TBD |
| EX-MIX3 cv40_iw40_sm20 | TBD | TBD | TBD | TBD | TBD |

### Pareto verdict

TBD — applies bake_compare § A.9 decisive rule per corpus per band.

## Honest gaps

Even before training completes, these gaps will materialize:

1. **Coverage gate dropped LARGE + konjnd-dense.** V_22-LARGE+iwssim's lift
   came from LARGE's content-class breadth (per the V_22 methodology). EX-MIX3
   sees a smaller training corpus (209k vs 282k rows). If EX-MIX3 underperforms
   V_22-LARGE+iwssim, the corpus shrink — not the 3-way blend — is the prime
   suspect; the apples-to-apples comparator is V_22 noLARGE.

2. **ssim2-normalization is safesyn-anchored.** kadid and tid rows use ssim2
   scores computed on those corpora but normalized by safesyn statistics.
   This is consistent with how `mix_cv40_iw60` is built across corpora, but
   it means kadid/tid mix targets are NOT "kadid-anchored" or "tid-anchored"
   — they live on a synthetic-corpus reference scale.

3. **Mid-experiment leak risk.** Per principled-workflow Step 2, CID22 is
   opened LAST. The verdict-aggregator script runs ONLY after all 15 trains
   complete. No mid-flight CID22 inspection.

## Lineage

- Build script: `scripts/v_next/add_mix3_target.py`
- Trainer driver: `scripts/v_next/run_ex_mix3_seed.sh`
- Matrix driver: `scripts/v_next/run_ex_mix3_all.sh`
- Eval aggregator: `scripts/v_next/eval_ex_mix3_all.sh`
- Output parquets: `/mnt/v/zen/zensim-training/2026-05-18-mix3/{safesyn,kadid,tid}.parquet`
- Output bakes: `/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/exmix3_<variant>_s<seed>_h128.bin`
- Training logs: same dir as bakes, `.log` + `.stdout` per seed

## Open / TBD

1. Validate bake_compare across feature-width mismatch (372 vs 300) — likely fails;
   plan B is to evaluate via bake_verdict full-panel side-by-side.
2. After all 15 trains finish, run `eval_ex_mix3_all.sh` and append 5-seed CI tables.
3. If a variant Pareto-wins: `zenpredict repack` packed bake + record path.
4. If all variants falsified: document root cause (coverage shrink vs 3-way blend) and
   close the experiment with a CLAUDE.md learnings entry.
