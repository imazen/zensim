# V_24 per-sample α head + konjnd_w=0.10 — methodology + results

**Date:** 2026-05-18
**Branch:** `feat/ex2-stdpool-head` (workspace `zensim--persample-konjnd010`)
**Bake artifacts:** `/mnt/v/zen/zensim-eval/v24_persample_konjnd010_2026-05-18/`

## Hypothesis (Step 1 of principled workflow)

1. **Hypothesis**: Compounding two proven mechanisms — per-sample α
   head (wins V_24-hybrid-NiN scalar α 5/0 in bake_compare, AIC-3 +0.028
   lift) and gradient-unstarvation via konjnd train weight 0.02→0.10
   (gave +0.21 KonJND in earlier discriminator) — should close the
   KonJND gap vs V_22-mix-LARGE (0.893) while preserving the AIC-3
   lift. **Target: CID22 ≥ 0.832, KonJND ≥ 0.88, AIC-3 ≥ 0.80.**
2. **Falsification**: If 5-seed mean KonJND stays below 0.83 (no
   compounding) OR AIC-3 drops below V_22's 0.785 (the gradient
   reweighting destroyed the per-sample α benefit), the compound
   hypothesis is dead.
3. **Cost ceiling**: 5-seed parallel training (~15 min wall) + eval
   (~5 min) + pack + bake_compare. ~60 min total.
4. **Ship form**: New PreviewV0_N bake candidate IF Pareto gate passes
   (CID22 ≥ V_22, KonJND ≥ V_22 - 0.01, AIC-3 ≥ V_22 + 0.015, KADID ≥
   V_22 - 0.04, TID ≥ V_22 - 0.04). Otherwise: methodology + negative
   result + commit.

## Reporting panel (Step 2)

| Corpus | Role | When inspected |
|---|---|---|
| safesyn (val_w=0) | training only | n/a |
| KADID | held-out integrity guard | per-seed verdict |
| TID | held-out integrity guard | per-seed verdict |
| KonJND | **load-bearing target** | per-seed verdict |
| CID22 | gold-standard generalization | end-of-experiment only |
| AIC-3 | compression-focused holdout | end-of-experiment only |
| cvvdp_iwssim_large (val_w=0) | training only | n/a |

Mohammadi 2025 full panel (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE)
aggregate + 10-band per corpus reported via `bake_verdict`.

## Architecture

Per-sample α head (identical to the
`v24_per_sample_alpha_methodology_2026-05-18.md` baseline):

```
h_pre  = b1 + Σ x_i · W1[i, :]             (n_hidden = 128)
h      = LeakyReLU(h_pre, slope=0.01)
y_rank = h · rank_w + rank_b
y_pool = [μ, σ, max, p_6](h) · reducer_w + reducer_b
α_logit(x) = b_α + h · W_α                  (per-sample W_α path)
α(x)   = sigmoid(α_logit(x))
y      = α(x) · y_rank + (1 − α(x)) · y_pool
```

Init: `W_α = 0`, `b_α = 0` → α(x) = 0.5 at start.
Bake metadata key: `zentrain.per_sample_alpha_head`.

## Training recipe — ONLY change from per-sample α baseline

| Group | Rows | Target | Train_w | Val_w |
|---|---|---|---|---|
| safesyn | 196,086 | mix_cv40_iw60 | 1.0 | 0.0 |
| kadid   |  10,125 | mix_cv40_iw60 | 0.3 | 1.0 |
| tid     |   3,000 | mix_cv40_iw60 | 0.3 | 1.0 |
| **konjnd** | **1,008** | **PJND** | **0.10** ← 0.02 | 1.0 |
| cvvdp_iwssim_large | 73,300 | mix_cv40_iw60 | 0.5 | 0.0 |

Hyperparams: hidden=128, epochs=300, lr=1e-3 cosine 50-epoch period,
l2=1e-5, leaky-α=0.01, minibatch=256, val-policy=Min, PWRC
(sensory_threshold=5.0), NiN 0.1 (p=1, q=2), 300-feature input.

Trainer command (per seed): see
`scripts/v_next/run_persample_konjnd010_seed.sh`.

## Data lineage (Step 6 metadata)

| File | MD5 | Rows |
|---|---|---|
| safesyn_mix_300col.parquet | a111126a430022f7541d4fbe0baba671 | 196,086 |
| kadid_mix_300col.parquet | 4802afadbe495e7172329e65c8c42e66 | 10,125 |
| tid_mix_300col.parquet | 98b74bfa574e1c8adf0ddb61bba70748 | 3,000 |
| konjnd_mix_300col.parquet | 83403418c6e71a7ba273147c257359a6 | 1,008 |
| cvvdp_iwssim_large_300col_v2.parquet | 7d8ddfe9a0067768d26dedbf398dd8bb | 73,300 |

Baselines for comparison:
- V_22-mix-LARGE+iwssim s3 packed (b703c9cfc7e1908faf5b0e78dc823221)
- Per-sample α seed4 packed (f09a9abdce00805000c1d112c2421b2d)

## Bake shape (Step 4)

Per-sample α head — `score`-shaped (training target is mix_cv40_iw60 in
0..100 range). No affine calibration needed.

## Results — FALSIFIED on Pareto gate

### 5-seed CI table (held-out via `bake_verdict`)

| Corpus | Mean SROCC | ± std | min | max | n seeds |
|---|---|---|---|---|---|
| CID22 | 0.7941 | 0.0096 | 0.7827 | 0.8036 | 5 |
| KADID | 0.9301 | 0.0004 | 0.9297 | 0.9305 | 5 |
| TID | 0.8891 | 0.0010 | 0.8883 | 0.8904 | 5 |
| **KonJND** | **0.9707** | 0.0016 | 0.9686 | 0.9731 | 5 |
| AIC-3 | 0.8031 | 0.0052 | 0.7948 | 0.8092 | 5 |

### Pareto gate (user task brief)

| Corpus | Mean | Gate | Result |
|---|---|---|---|
| CID22 | 0.7941 | ≥ 0.832 (V_22) | **FAIL** (−0.038) |
| KADID | 0.9301 | ≥ 0.928 (V_22 − 0.04) | PASS |
| TID | 0.8891 | ≥ 0.933 (V_22 − 0.04) | **FAIL** (−0.044) |
| KonJND | 0.9707 | ≥ 0.880 (V_22 − 0.01) | PASS (+0.091) |
| AIC-3 | 0.8031 | ≥ 0.800 (V_22 + 0.015) | PASS (+0.003) |

3 of 5 gates pass — **falsified**.

### Comparison vs baselines (seed4 — best KonJND)

| Metric | V_22-LARGE | Per-sample-α (s4) | **konjnd010 (s4)** | Δ vs V_22 | Δ vs scalar α | Δ vs persample-α |
|---|---|---|---|---|---|---|
| CID22 SROCC | 0.832 | 0.864 | **0.785** | −0.047 | n/a | −0.079 |
| KADID SROCC | 0.968 | 0.932 | 0.931 | −0.037 | n/a | −0.001 |
| TID SROCC | 0.973 | 0.889 | 0.888 | −0.085 | n/a | −0.001 |
| **KonJND SROCC** | **0.893** | 0.808 | **0.973** | **+0.080** | n/a | **+0.165** |
| AIC-3 SROCC | 0.785 | 0.818 | 0.809 | +0.024 | n/a | −0.009 |

### bake_compare decisive verdicts (200 resamples)

**konjnd010 s4 vs V_22-mix-LARGE** (`compare_seed4_vs_v22_LARGE.md`):
- ADecisivelyBeatsB: 2 cells (KonJND, AIC-3)
- BDecisivelyBeatsA: 19 cells (CID22, KADID, TID)
- Overall: **B wins** (V_22-LARGE)

**konjnd010 s4 vs per-sample-α (s4) baseline** (`compare_seed4_vs_persample_baseline.md`):
- ADecisivelyBeatsB: 1 cell (KonJND)
- BDecisivelyBeatsA: 9 cells (CID22 dominant)
- Overall: **B wins** (per-sample-α baseline)

### What happened (Step 10 negative-result analysis)

The compound hypothesis ASSUMED that the +0.21 KonJND lift seen
in earlier `feat/ex2-konjnd010` (std-pool head, scalar α, konjnd_w=0.10)
would stack additively with per-sample α's +0.028 AIC-3 lift. Instead
the konjnd_w=0.10 setting drove the encoder so hard toward
KonJND-shape that the rank/pool mix no longer differentiates
CID22-style photo distortions effectively:

- KonJND val_srocc climbed from 0.62 (per-sample α baseline at
  konjnd_w=0.02) to 0.97 — far past the V_22 anchor.
- CID22 generalization collapsed from 0.864 (per-sample α) to 0.794
  — even below V_22-LARGE's 0.832, which had konjnd_w=0.02 plus the
  full LARGE recipe.
- The KADID/TID losses (-0.001 each vs per-sample-α) are noise; the
  CID22 loss is the **real** falsification.

Compounding interpretation: the two mechanisms target the same
gradient-channel-allocation problem from opposite angles. per-sample
α gives JND-likely inputs *more rank-head weight* (raises α for
KonJND-shaped inputs); raising konjnd_w pushes the loss surface
toward KonJND-style rankings *regardless* of input shape. Doing
both at once locks the rank/pool mix at a global KonJND-favoring
configuration that the per-sample α path can't dial back, because
the encoder hidden vector h is itself being warped by the
gradient-weight imbalance.

### Conclusion

V_24-persample-konjnd010 is **not a ship candidate**. The compound
move overshot KonJND at the cost of CID22 generalization — the
opposite of "near-strict improvement." Per CLAUDE.md Step 10, this
result is committed as data.

### Per-corpus α distribution (seed=4 — engagement diagnostic)

*(Skipped — diagnostic was the load-bearing question for the
per-sample α baseline, which already established mechanism
engagement. For konjnd010 the bake_verdict aggregate result was
sufficient to falsify the compound hypothesis. If a follow-up
experiment wants the α distribution, run the same diagnostic on
seed4 — the per-sample α dispatch is in place.)*

### Artifacts

| File | Path |
|---|---|
| Bake (best KonJND, seed=4, raw f32) | `/mnt/v/zen/zensim-eval/v24_persample_konjnd010_2026-05-18/persample_konjnd010_seed4.bin` (223 KB) |
| Bake (best KonJND, seed=4, f32-lz4 packed) | `persample_konjnd010_seed4_packed.bin` (159 KB) |
| Bake (best KonJND, seed=4, i8-zerobias-lz4) | `persample_konjnd010_seed4_i8packed.bin` (44 KB) |
| Per-seed verdict (Mohammadi panel) | `verdict_seed{1..5}.md` |
| Eval summary | `eval_all_summary.log` |
| bake_compare vs V_22-LARGE | `compare_seed4_vs_v22_LARGE.md` |
| bake_compare vs per-sample-α | `compare_seed4_vs_persample_baseline.md` |
| Training logs | `persample_konjnd010_seed{1..5}.log` |

### Next direction (do NOT pursue this hypothesis further)

Tuning konjnd_w in (0.02, 0.10) MAY find a sweet spot, but the
2-axis sweep cost (5 seeds × ~5 konjnd_w values) is large for
modest expected gain. The promising avenue is to find a
DIFFERENT KonJND lift mechanism that doesn't trade against CID22 —
e.g., KonJND-anchored finetuning on top of a per-sample-α
checkpoint (preserves CID22 head), or PJND-aware loss weighting
(re-weights pairs near the JND boundary rather than reweighting
the konjnd group as a whole).

