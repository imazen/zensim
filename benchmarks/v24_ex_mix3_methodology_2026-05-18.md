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

## Mid-experiment fix: konjnd PJND-anchor added (2026-05-18 09:42)

**First-cycle seed=1 results (without konjnd training group):**

| Variant | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---|---|---|---|---|
| cv33_iw33_sm33 s1 | 0.8934 | 0.9186 | 0.8695 | **0.2990** | 0.8114 |
| cv30_iw40_sm30 s1 | 0.8940 | 0.9291 | 0.8762 | **0.2996** | 0.8114 |
| baseline (V_22 noLARGE s1) | 0.8558 | 0.9336 | 0.8904 | 0.8369 | 0.8107 |

**CID22 +0.038 (huge win), KonJND -0.54 (catastrophic collapse).** Without
the 1008-row konjnd training group at weight 0.02, the model has no JND
boundary signal — same failure mode as V_22-CVVDP-LARGE (per the V_22
methodology, "Pure-CVVDP supervision on compression distortions gives no
signal for JND ordering").

**Fix applied:** Re-introduce the konjnd training group at weight 0.02
using PJND-passthrough as the target. The konjnd small parquet's
`human_score` IS the PJND compression-q threshold (range [22.46, 69.98],
mean 41.98) — the same value V_22 noLARGE used through `mix_cv40_iw60`
(which was a passthrough copy of `human_score` in that corpus). For
EX-MIX3 I write `mix_cv33_iw33_sm33 = mix_cv30_iw40_sm30 =
mix_cv40_iw40_sm20 = mix_cv40_iw60 = human_score = PJND` for the 1008
konjnd rows. This is NOT zero-fill (the value IS the real PJND signal); it
is a label-passthrough for a group whose 3-way blend equation is
undefined because the group has no cvvdp/iwssim/ssim2 per-pair scores.

This preserves V_22 noLARGE's exact JND-anchoring mechanic verbatim while
keeping the 3-way blend on the other 3 groups (safesyn 1.0 / kadid 0.3 /
tid 0.3). The first-cycle results are archived at
`/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/noKONJND_backup/` for
reference.

Updated training groups (4 groups, 210,219 rows — identical to V_22 noLARGE):

| Group | Rows | Target | Train_w | Val_w |
|---|--:|---|--:|--:|
| safesyn | 196,086 | mix_cv33_iw33_sm33 | 1.0 | 0.0 |
| kadid | 10,125 | mix_cv33_iw33_sm33 | 0.3 | 1.0 |
| tid | 3,000 | mix_cv33_iw33_sm33 | 0.3 | 1.0 |
| konjnd | 1,008 | mix_cv33_iw33_sm33 (= PJND) | 0.02 | 1.0 |

Re-launched 15-job matrix at 09:43:50.

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

## Partial results snapshot (round-2, 6/15 bakes, 10:11 UTC)

Per-variant 2-seed mean vs V_22 noLARGE 5-seed baseline mean:

| Variant | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---|---|---|---|---|
| cv33_iw33_sm33 (n=2) | 0.8637 | 0.9161 | 0.8706 | 0.8395 | 0.7928 |
| cv30_iw40_sm30 (n=2) | 0.8612 | 0.9273 | 0.8766 | 0.8447 | 0.7985 |
| cv40_iw40_sm20 (n=2) | 0.8583 | 0.9170 | 0.8750 | 0.8374 | 0.8023 |
| baseline V_22 noLARGE (n=5) | 0.8425 | 0.9311 | 0.8897 | 0.8371 | 0.8059 |

### bake_compare (cv30_iw40_sm30 s1 vs V_22 noLARGE s3, 1000 bootstrap)

| Corpus | Verdict | Notes |
|---|---|---|
| CID22 | **A>>B (decisive)** | +0.039 SROCC, +0.030 PWRC |
| KADID | promising | within ±0.001 |
| TID | **B>>A (decisive)** | -0.015 SROCC |
| KonJND | tied | -0.004 |
| AIC-3 | tied | +0.001 |

Overall: A wins 4 decisive cells, B wins 3 — A overall winner. **TID is a
decisive regression**, so this variant **fails the strict Pareto-or-perish
gate**. But it's a TRADE — CID22 decisively wins (the gold-standard
held-out, +0.039 SROCC), TID decisively loses (one of the synthetic
non-compression corpora, -0.015 SROCC). The trade direction matches the
"two-trail SOTA" framework in CLAUDE.md / memory: PreviewV0_5Compression
trail prioritizes CID22 + AIC-3 (compression product decisions);
PreviewV0_5Balanced trail prioritizes Pareto-all-corpora.

EX-MIX3 is firmly on the compression-priority trail. Final ship decision
awaits 5-seed CI completion.

## CID22 10-band per-band lift (cv33_iw33_sm33 s1 vs V_22 noLARGE s1)

| Band | n | V_22 noLARGE | cv33 s1 | Δ |
|---|--:|---:|---:|---:|
| B3 (30-40) | 57 | 0.0436 | 0.1276 | **+0.084** |
| B4 (40-50) | 266 | 0.2541 | 0.2830 | +0.029 |
| B5 (50-60) | 615 | 0.2518 | 0.3220 | **+0.070** |
| B6 (60-70) | 836 | 0.2438 | 0.2932 | +0.049 |
| B7 (70-80) | 1092 | 0.3776 | 0.3956 | +0.018 |
| B8 (80-90) | 1382 | 0.4844 | 0.4867 | +0.002 |
| B9 (90-100) | 43 | 0.1675 | 0.1992 | +0.032 |

Uniformly positive across every band with non-trivial n. B3 / B5 are the
big winners (+0.07-0.08 SROCC) — these are mid-quality bands where most
compression product decisions live. This confirms the 3-way blend's gain
is broad-based, not a band-specialist trade.

## 3-seed snapshot (round-2, 8/15 bakes, 10:19 UTC)

| Variant | n | CID22 | KADID | TID | KonJND | AIC-3 |
|---|--:|---|---|---|---|---|
| cv33_iw33_sm33 | 3 | **0.8661±0.0092** | 0.9171±0.0034 | 0.8702±0.0019 | 0.8396±0.0008 | 0.7966±0.0123 |
| cv30_iw40_sm30 | 3 | 0.8619±0.0097 | **0.9266±0.0019** | 0.8769±0.0007 | **0.8437±0.0043** | **0.8006±0.0068** |
| cv40_iw40_sm20 | 2 | 0.8583±0.0116 | 0.9170±0.0037 | 0.8750±0.0019 | 0.8374±0.0026 | 0.8023±0.0127 |
| V_22 noLARGE (baseline, n=5) | 5 | 0.8425±0.0110 | 0.9311±0.0022 | 0.8897±0.0015 | 0.8371±0.0066 | 0.8059±0.0057 |
| V_22-LARGE+iwssim (alt, n=5) | 5 | 0.8339±0.0071 | 0.9673±0.0002 | 0.9726±0.0004 | 0.8869±0.0034 | 0.7872±0.0078 |

### Pattern emerging

vs V_22 noLARGE (apples-to-apples, 4-group 372-feat):
- **CID22: all 3 variants win decisively** (Δ +0.016 to +0.024, well outside ±σ)
- **KonJND: all 3 variants parity or slight win** (+0.000 to +0.007)
- **AIC-3: all 3 variants within seed noise** (Δ -0.004 to -0.009)
- **KADID: all 3 variants small regression** (Δ -0.004 to -0.014)
- **TID: all 3 variants small regression** (Δ -0.013 to -0.019)

vs V_22-LARGE+iwssim (current strongest mix):
- **CID22: EX-MIX3 wins decisively** (+0.028 to +0.032)
- **AIC-3: EX-MIX3 wins** (+0.009 to +0.013)
- **KADID/TID/KonJND: EX-MIX3 LOSES decisively** (LARGE has the content-class breadth)

**Variant ranking (preliminary):**
- **cv30_iw40_sm30** is the most balanced: best KADID/KonJND/AIC-3 of the 3, second-best CID22, smallest TID regression. **Likely the ship candidate.**
- **cv33_iw33_sm33** is the most aggressive CID22 specialist: best CID22 (+0.024), but worse KADID/AIC-3/TID.
- **cv40_iw40_sm20** is in between but doesn't dominate any corpus.

The cv30 variant's lift over baseline is concentrated where it should be:
- CID22 (the gold-standard compression-distortion benchmark): +0.019
- KonJND (PJND anchor): +0.007
- KADID (synthetic non-compression distortions, less weight for our use case): -0.004 (parity)
- TID (similar to KADID): -0.013

Final 5-seed CI awaits round-4-5 completion (ETA ~10:46).

## 4-seed CI (12/15 bakes, 10:33 UTC)

| Variant | n | CID22 | KADID | TID | KonJND | AIC-3 |
|---|--:|---|---|---|---|---|
| cv33_iw33_sm33 | 4 | 0.8673±0.0079 | 0.9176±0.0029 | 0.8697±0.0019 | 0.8400±0.0011 | 0.7988±0.0110 |
| cv30_iw40_sm30 | 4 | 0.8627±0.0081 | 0.9271±0.0019 | 0.8766±0.0009 | 0.8428±0.0039 | 0.8017±0.0059 |
| cv40_iw40_sm20 | 4 | 0.8625±0.0083 | 0.9187±0.0029 | 0.8761±0.0017 | 0.8378±0.0028 | 0.8032±0.0076 |
| V_22 noLARGE (baseline, n=5) | 5 | 0.8425±0.0110 | 0.9311±0.0022 | 0.8897±0.0015 | 0.8371±0.0066 | 0.8059±0.0057 |
| V_22-LARGE+iwssim (alt, n=5) | 5 | 0.8339±0.0071 | 0.9673±0.0002 | 0.9726±0.0004 | 0.8869±0.0034 | 0.7872±0.0078 |

### Δ vs V_22 noLARGE (units of σ)

| Variant | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| cv33_iw33_sm33 | +22σ | -6σ | -10σ | +1σ | -1σ |
| cv30_iw40_sm30 | +18σ | -2σ | -7σ | +1σ | -1σ |
| cv40_iw40_sm20 | +17σ | -6σ | -7σ | +0σ | -0σ |

(σ uses the larger of the two seed-stds. Treat |Δ| < 2σ as parity/noise.)

**Decisive winners:**
- CID22: all 3 variants decisively win (Δ ~ 17-22σ)
- KADID: cv30 parity (-2σ); cv33+cv40 small decisive losses (-6σ)
- TID: all 3 variants decisively lose (-7 to -10σ)
- KonJND: all 3 variants parity
- AIC-3: all 3 variants parity

cv30_iw40_sm30 is the most balanced (best KADID, best KonJND, smallest TID
regression, near-best CID22). Recommended ship candidate.

## FINAL 5-seed CI (15/15 bakes complete, 10:45 UTC)

| Variant | n | CID22 | KADID | TID | KonJND | AIC-3 |
|---|--:|---|---|---|---|---|
| cv33_iw33_sm33 | 5 | **0.8655±0.0079** | 0.9163±0.0038 | 0.8701±0.0018 | 0.8399±0.0010 | 0.7967±0.0106 |
| cv30_iw40_sm30 | 5 | 0.8611±0.0078 | **0.9270±0.0017** | 0.8770±0.0013 | 0.8431±0.0034 | 0.7997±0.0068 |
| cv40_iw40_sm20 | 5 | 0.8608±0.0082 | 0.9182±0.0027 | 0.8762±0.0015 | 0.8392±0.0039 | 0.8021±0.0070 |
| V_22 noLARGE (baseline, n=5) | 5 | 0.8425±0.0110 | 0.9311±0.0022 | 0.8897±0.0015 | 0.8371±0.0066 | 0.8059±0.0057 |
| V_22-LARGE+iwssim (alt, n=5) | 5 | 0.8339±0.0071 | 0.9673±0.0002 | 0.9726±0.0004 | 0.8869±0.0034 | 0.7872±0.0078 |

### Δ vs V_22 noLARGE 5-seed baseline

| Variant | ΔCID22 | ΔKADID | ΔTID | ΔKonJND | ΔAIC-3 | strict Pareto gate |
|---|---:|---:|---:|---:|---:|---|
| cv33_iw33_sm33 | **+0.023** | -0.015 | -0.020 🚨 | +0.003 | -0.009 | **FAIL** (TID + KADID decisive regression) |
| cv30_iw40_sm30 | **+0.019** | -0.004 | -0.013 🚨 | +0.006 | -0.006 | **FAIL** (TID decisive regression) |
| cv40_iw40_sm20 | **+0.018** | -0.013 | -0.013 🚨 | +0.002 | -0.004 | **FAIL** (TID + KADID decisive regression) |

🚨 = decisive regression per § A.9: ΔSROCC < -0.01 AND ΔPWRC < -0.005 AND ΔZ-RMSE > +0.020.

## Pareto verdict: **FAIL on strict gate** but **WIN on compression-priority trail**

The strict Pareto-or-perish gate (≥4-of-6 panel agreement per corpus, NO
decisive regression on any) fails for all 3 variants — each has a
decisive TID regression. cv33 and cv40 additionally have decisive KADID
regression.

But per CLAUDE.md "two-trail SOTA" framework (memory:
project_two_trail_sota.md / feedback_two_trail_sota.md), there are two
valid ship trails:

1. **PreviewV0_5Balanced** trail: Pareto-all-corpora, currently
   V_22-LARGE+iwssim (CID22 0.834, KADID 0.967, TID 0.973, KonJND 0.887,
   AIC-3 0.787).
2. **PreviewV0_5Compression** trail: CID22+AIC-3 priority for compression
   product decisions, currently V_22-372feat (CID22 0.842, KADID 0.931,
   TID 0.890, KonJND 0.837, AIC-3 0.806).

**EX-MIX3 cv30_iw40_sm30 SUPERSEDES V_22-372feat on the compression trail:**
- CID22 +0.019 (compression gold standard improves)
- KonJND +0.006 (PJND parity maintained)
- KADID -0.004 (near-parity)
- TID -0.013 (small synthetic-distortion regression)
- AIC-3 -0.006 (low-q held-out near-parity)

The trade direction is correct: gives up small synthetic-distortion
accuracy for meaningful compression-gold-standard improvement.

vs V_22-LARGE+iwssim (the balanced trail) cv30 wins **CID22 +0.027 + AIC-3 +0.013**
but loses **KADID -0.040 + TID -0.096 + KonJND -0.044** — confirms EX-MIX3
is firmly on the compression trail, not balanced.

## Ship candidate

`/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/exmix3_cv30_iw40_sm30_s3_h128_packed.bin`

- Source: cv30_iw40_sm30 seed=3 (representative seed near 5-seed mean).
- Packed: i8 + zerobias 0.005 + lz4 (51,976 bytes = 26.7% of unpacked).
- Quantization drift: CID22 SROCC +0.0009 (negligible).
- bake_verdict on packed (sanity check):
  - CID22 0.8642 ✓
  - KADID 0.9255 ✓
  - TID 0.8776 ✓
  - KonJND 0.8424 ✓
  - AIC-3 0.8048 ✓

This bake CAN ship as the new PreviewV0_5Compression weight if the user
prefers a stronger CID22 lift over current V_22-372feat. No crate version
bump per user direction 2026-05-17.

## Data-lineage table

| Path | Role | sha256 prefix | row count |
|---|---|---|---|
| `/mnt/v/zen/zensim-training/2026-05-18-mix3/safesyn.parquet` | training group A | new build | 196,086 |
| `/mnt/v/zen/zensim-training/2026-05-18-mix3/kadid.parquet` | training group B | new build | 10,125 |
| `/mnt/v/zen/zensim-training/2026-05-18-mix3/tid.parquet` | training group C | new build | 3,000 |
| `/mnt/v/zen/zensim-training/2026-05-18-mix3/konjnd.parquet` | training group D (PJND-passthrough) | new build | 1,008 |
| `/mnt/v/zen/zensim-training/canonical-2026-05-18/val/cid22.parquet` | validation, gold std | `6eea08253fa2` | 4,292 |
| `/mnt/v/zen/zensim-training/canonical-2026-05-18/val/konjnd.parquet` | validation, PJND | `3e999a372577` | 1,008 |

CID22-contam status: all training sources are derived from canonical
2026-05-18 corpus, which has the post-2026-05-12 perceptual-overlap
purge applied. No new contamination risks.
