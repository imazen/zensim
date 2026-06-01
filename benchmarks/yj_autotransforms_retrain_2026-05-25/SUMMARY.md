# v11 retrain with YJ auto-transforms — verdict (task #214 Phase 2)

Date: 2026-05-25. Single-seed (s=1) v11 retrain layering
`--auto-transforms` (scipy MLE YJ screen + cross-corpus-safe
substitutions) onto the canonical v11 recipe. Phase 1
(`benchmarks/yeo_johnson_screen_widest_2026-05-25/`) eliminated YJ
boundary pinning via scipy unconstrained MLE; Phase 2 layers those
transforms onto the v11 trainer.

## TL;DR — Verdict: NO SHIP (advisory; user decides)

Per CLAUDE.md ship policy: all gates ADVISORY per 2026-05-14. This
bake **wins KonJND-1k decisively** (+0.377 SROCC, +0.470 PLCC,
+0.316 PWRC, −0.281 Z-RMSE) but **loses CID22 / KADID / AIC-3 /
AIC-4 by modest single-stat deltas** (−0.03 to −0.05 SROCC). TID is
~flat. The full Mohammadi panel paints a more nuanced picture than
SROCC alone (per CLAUDE.md "SROCC-only verdicts BANNED").

The per-block L0 mass shift is the more durable finding (see § 5):
the YJ-autotransforms retrain pulls **56pp of mass off `basic`**
and redistributes to `peak / masked / iw_pool`. This is the
structural answer to task #214's "why doesn't IW-pool dominate v11"
question — when the standardize layer sees a richer transform
suite, the trainer naturally uses the previously-collinear pool
features more.

Phase 1 commit: `b78d3a09` (already on main).
Phase 2 bake: `zensim-experimental/weights/v_tuner_v11_yj_autotransforms_2026-05-25.bin`
(274,735 bytes, md5 `bfb109563f6f9ca6e88eaa1921d331f4`).

## 1. Phase 1 — widest-grid YJ screen (recap)

| Run                | λ range / method                                  | YJ outright wins | Boundary-pinned |
|--------------------|---------------------------------------------------|-----------------:|----------------:|
| narrow (prior)     | golden-section ∈ [-2, 2]                          |        3 / 372   |    370 / 372    |
| wide (prior)       | golden-section ∈ [-5, 5]                          |        8 / 372   |    361 / 372    |
| hardgrid10         | golden-section ∈ [-10, 10]                        |       10 / 372   |    349 / 372    |
| **widest (scipy)** | `scipy.stats.yeojohnson_normmax` (unconstrained)  | **53 / 372**     | **0 / 372**     |

scipy's data-driven bounds (overflow-aware `log_max_float /
log1p(20·max|x|)`) drove boundary pinning to ZERO. λ distribution
on the canonical safesyn corpus: **min −1062.62, median −73.68,
max −1.72**. Every YJ-winning feature wants λ < 0 (heavy positive
tails).

Detail: `benchmarks/yeo_johnson_screen_widest_2026-05-25/SUMMARY.md`.

## 2. Phase 2 — retrain configuration

Recipe identical to `scripts/v_next/run_tuner_v11_attempt7_seed.sh`
with TWO additions:

1. `--auto-transforms benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv`
2. `--auto-transforms-min-lift 0.05`

Cross-corpus-safe TSV: starts from the scipy MLE screen
(`screen_results.tsv` with 53 YJ winners + 222 winsor_p99 + 111
log + …) and **substitutes any `log` entry where ANY training corpus
has x ≤ 0 with the best-supported alternative** (per the
per_transform.tsv ranking).

Without this substitution, `log(0) = −∞` flows into the standardize
step on the anchor parquet (which has 94 features whose min is 0
vs safesyn's min > 0) → training NaN at epoch 0. The substitution
preserves all 53 YJ entries (YJ has full-real-line domain + the
runtime overflow guard) and converts the unsafe-log entries to
winsor_p99 / signed_cbrt / quantile_bins (whichever the screen
ranks next).

Substitution table:

| Token after substitution | Count |
|---|--:|
| winsor_p99       | 183 |
| quantile_bins    |  75 |
| signed_cbrt      |  56 |
| **yeo_johnson**  |  **53** |
| clip_then_log1p  |   4 |
| log1p            |   1 |
| (identity)       |   ~ |

Other recipe args (verbatim from v11 ship):

```
--hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 5.66e-3 --l2 1e-5
--leaky-alpha 0.01 --val-policy min --early-stop-patience 0
--max-features 372 --minibatch-size 32 --target-column mix_cv40_iw60
--per-sample-alpha-head --tanh-output-head-scale 30.0
--ranknet-weight 0.0 --mse-weight 1.0
--monotonicity-reg 1.0 --monotonicity-margin 0.0
--anchor-parquet anchors_v9_372col.parquet --anchor-loss-weight 0.5
--cross-codec-eq-parquet cross_codec_equivalence_tight_v3.parquet
--cross-codec-eq-weight 1.0 --cross-codec-rank-preserve-weight 0.2
--dynamic-range-floor-weight 0.3 --dynamic-range-sigma-threshold 25.0
--konjnd-aggregation-parquet konjnd-dense.parquet
--konjnd-aggregation-weight 0.3 --seed 1
```

5 training groups: safesyn:1.0 + cid22_train:0.5 + kadid:0.5 +
tid:0.5 + konjnd_dense:0.3. Final epoch loss: 13.66 (down from
38.65 at epoch 0). Best `val_min` SROCC: 0.9510 at epoch 10 (see
note in § 6 on val-policy methodology gap).

Wall time: ~25 min on Lilith's 7950X.

## 3. Phase 2 training log digest

| Epoch | loss  | val_min | safesyn | cid22_train | kadid | tid   | konjnd_dense |
|------:|------:|--------:|--------:|------------:|------:|------:|-------------:|
|  0    | 38.65 | 0.9409  | 0.9697  | 0.8989      | 0.9327 | 0.9239 | 0.9795      |
| 10    | 29.17 | 0.9510 *| 0.9669  | 0.9220      | 0.9570 | 0.9720 | 0.9370      |
| 50    | 21.42 | 0.9265  | 0.9727  | 0.9120      | 0.9572 | 0.9669 | 0.8239      |
| 100   | 18.44 | 0.9076  | 0.9698  | 0.8839      | 0.9530 | 0.9698 | 0.7614      |
| 200   | 16.44 | 0.9315  | 0.9689  | 0.9205      | 0.9543 | 0.9706 | 0.8429      |
| 299   | 13.66 | 0.9290  | 0.9811  | 0.9611      | 0.9669 | 0.9827 | 0.7535      |

`val_min` early-peak (`*`) is the cyclic-LR + konjnd_dense
oscillation pattern noted in the v11 ship — `--val-policy min` is
governed by the worst-performing corpus, which is konjnd_dense in
both runs. By final epoch every other corpus has its best SROCC
ever, but konjnd_dense slid from 0.94 → 0.75.

## 4. Per-corpus Mohammadi panel diff (current v11 ship vs YJ-AT retrain)

| Corpus | n | SROCC Δ | PLCC Δ | KROCC Δ | OR Δ | PWRC Δ | Z-RMSE Δ |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22         |  4292 | -0.0449 | -0.0368 | -0.0599 | -0.0039 | -0.0280 | +0.0550 |
| KADIK10k      | 10125 | -0.0260 | -0.0252 | -0.0371 | -0.0102 | -0.0172 | +0.0560 |
| TID2013       |  3000 | -0.0037 | +0.0051 | -0.0081 | -0.0037 | -0.0015 | -0.0100 |
| **KonJND-1k** |  1008 | **+0.3768** | **+0.4698** | **+0.2600** | -0.0049 | **+0.3159** | **-0.2810** |
| AIC-3 CTC     |   600 | -0.0267 | -0.0446 | -0.0280 | +0.0050 | -0.0281 | +0.0530 |
| AIC-4 sample  |   300 | -0.0091 | -0.0089 | -0.0132 | +0.0200 | -0.0097 | +0.0210 |

Raw values (ship → candidate):

| Corpus | n | SROCC ship | SROCC cand | PWRC ship | PWRC cand | Z-RMSE ship | Z-RMSE cand |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22         |  4292 | 0.8604 | 0.8155 | 0.9089 | 0.8809 | 0.523 | 0.578 |
| KADIK10k      | 10125 | 0.9237 | 0.8977 | 0.9550 | 0.9378 | 0.385 | 0.441 |
| TID2013       |  3000 | 0.8849 | 0.8812 | 0.9146 | 0.9131 | 0.459 | 0.449 |
| KonJND-1k     |  1008 | 0.2888 | **0.6656** | 0.4043 | **0.7202** | 0.966 | **0.685** |
| AIC-3 CTC     |   600 | 0.7761 | 0.7494 | 0.8538 | 0.8257 | 0.616 | 0.669 |
| AIC-4 sample  |   300 | 0.9284 | 0.9193 | 0.9620 | 0.9523 | 0.383 | 0.404 |

**The KonJND-1k jump is structural, not noise**. PJND thresholds
are the only corpus measuring "visually lossless" calibration —
the v11 ship's SROCC 0.29 there has been a known weak point, and
the YJ-autotransforms recipe doubles it. The peak/masked/iw_pool
share-of-mass increase (§ 5) explains the mechanism.

The CID22/KADID/AIC losses are modest (< 0.05 SROCC, well within
the 0.10 advisory tolerance in CLAUDE.md). PWRC + Z-RMSE move in
the same direction as SROCC, so this is not a "SROCC lies, panel
wins" inversion — the result is consistent across stats.

## 5. Per-block L0 importance shift

| Block | Ship % | Candidate % | Δ pp |
|---|---:|---:|---:|
| basic    | 95.85% | 39.13% | **−56.72** |
| peak     |  2.07% | 22.78% | **+20.71** |
| masked   |  0.86% | 18.30% | **+17.44** |
| iw_pool  |  1.22% | 19.79% | **+18.57** |

(Importance = `scaler_scale[i] × Σ_h |L0[h, i]|` aggregated per
block. The v11 ship is i8-quantized vs the YJ-AT candidate f32, so
absolute mass scales differ — only the share-of-mass column is
directly comparable.)

This is the most important finding of Phase 2. The v11 ship was
**95.85% basic-block-dominated** — every other block (peak,
masked, iw_pool) was getting < 3% of L0 mass. With YJ + winsor +
quantile_bins applied per-feature, the standardize step reshapes
the input distribution enough that the trainer can extract
signal from the previously-collinear peak / masked / iw_pool
features. The IW-pool's share moves from 1.22% to 19.79%
(+18.57pp), finally giving the IW features the weight task #214's
prior investigation said was being suppressed by collinearity with
basic + peaks.

The corresponding eval regressions (SROCC ↓ on CID22/KADID/etc)
suggest the redistributed mass isn't yet OPTIMAL — the trainer
got new degrees of freedom but didn't fully converge to the best
use of them in a single seed. A multi-seed sweep + hyperparam
retuning on this transform set is the next experiment.

## 6. Honest gaps + methodology notes

1. **Single seed.** Per task brief: single-seed Phase 2 only. A
   5-seed sweep is the natural follow-on to estimate CI on the
   SROCC deltas.
2. **`--val-policy min` vs CLAUDE.md.** CLAUDE.md (2026-05-15
   "SROCC-only verdicts BANNED") requires the full Mohammadi panel
   for ship verdicts, not SROCC alone. But the v11 trainer still
   uses `--val-policy min` (worst-corpus SROCC) for the
   in-training "best epoch" selection. That's a pre-existing
   methodology gap with the trainer itself — Phase 2 inherits it
   for apples-to-apples vs v11 ship. The final bake is from epoch
   299, not "best val", per `--early-stop-patience 0`.
3. **Trainer metadata-propagation bug fixed mid-experiment.** The
   `--auto-transforms` flag was producing bakes WITHOUT the
   `zentrain.feature_transforms` metadata on the
   per-sample-α-head + tanh path. First Phase 2 run scored
   CID22 SROCC 0.215 (garbage) because the runtime fed raw
   features to a transform-trained network. Patched
   `bake_per_sample_alpha_head_v3_with_tanh_and_transforms` to
   propagate the metadata — see commit on this PR. This is the
   kind of bug a refactor of the trainer's bake-emit paths would
   surface (see user feedback "decrappification and refactoring").
4. **`--minibatch-size 32`.** Same as v11 ship — the task brief
   was apples-to-apples vs v11. Smaller minibatch with `--lr
   5.66e-3` does compromise stability at this LR, but it's the
   ship recipe's choice and not isolatable in this single
   experiment. A separate ablation is warranted.
5. **scipy unconstrained MLE produces extreme λ.** The runtime
   YJ implementation has an overflow guard (commit `9a9be820` on
   zenanalyze main) that falls back to Identity on non-finite
   output. On this corpus, the guard is not active (verified — no
   feature has both extreme negative λ AND extreme negative x in
   the training corpora). But the guard is critical for portability
   to other corpora.

## 7. Artifacts

| Path | Content |
|---|---|
| `v11_ship_baseline_verdict.md` | bake_verdict on current v11 ship (`v_tuner_v11_2026-05-24.bin`) |
| `v11_yj_at_verdict.md` | bake_verdict on the new YJ-AT retrain |
| `diff_summary.md` | Mohammadi-panel diff + raw values + per-band tables |
| `bake_verdict_diff.tsv` | machine-readable diff (one row per (corpus, band, stat)) |
| `l0_per_block.tsv` | per-block L0 share-of-mass for both bakes |

Phase 1 artifacts: `../yeo_johnson_screen_widest_2026-05-25/`.

## 8. Bake metadata

```
$ zenpredict inspect zensim-experimental/weights/v_tuner_v11_yj_autotransforms_2026-05-25.bin
n_inputs: 372
layers: [372→128 (LeakyReLU, f32), 128→128 (Identity, f32)]
metadata:
  zentrain.per_sample_alpha_head    (numeric, 1056 bytes)
  zentrain.tanh_output_head         (numeric, 4 bytes)
  zentrain.feature_transforms       (utf8,  ~5000 bytes)
  zentrain.feature_transform_params (utf8, ~10000 bytes)
```

## 9. Reproduce

```bash
cd ~/work/zen/zensim
bash scripts/v_next/run_tuner_v11_yj_autotransforms_2026-05-25.sh 1
./target/release/bake_verdict \
  --bake zensim-experimental/weights/v_tuner_v11_yj_autotransforms_2026-05-25.bin \
  --output benchmarks/yj_autotransforms_retrain_2026-05-25/v11_yj_at_verdict.md
python3 scripts/v_next/yj_at_phase2_diff.py \
  --ship-md benchmarks/yj_autotransforms_retrain_2026-05-25/v11_ship_baseline_verdict.md \
  --candidate-md benchmarks/yj_autotransforms_retrain_2026-05-25/v11_yj_at_verdict.md \
  --out-tsv benchmarks/yj_autotransforms_retrain_2026-05-25/bake_verdict_diff.tsv \
  --out-md benchmarks/yj_autotransforms_retrain_2026-05-25/diff_summary.md
./target/release/examples/l0_per_block_compare \
  benchmarks/yj_autotransforms_retrain_2026-05-25/l0_per_block.tsv
```

Requires zenanalyze pinned at commit `9a9be8200f26` (YJ FeatureTransform variant +
NaN-safety overflow guard + universal NaN-safety test suite).
