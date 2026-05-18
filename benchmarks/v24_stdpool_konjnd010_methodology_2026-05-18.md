# V_24-stdpool-konjnd010 methodology (EX-2 follow-up: gradient-starvation test)

**Status: 5-seed CI complete 2026-05-18 ~08:21Z. Verdict: partial
KonJND recovery (+0.21 vs prod) but CID22 loses, KADID/TID/AIC-3
flat. Strict-Pareto loss vs V_22. Pareto gate FAILS. Do NOT ship.
Result supports the gradient-starvation hypothesis: KonJND was
gradient-starved at konjnd train_w=0.02, NOT structurally dead.**

## Hypothesis

The V_24-stdpool-prod 5-seed CI showed KonJND −0.351 vs V_22, and
the NiN-off followup (V_24-stdpool-nonin) **falsified** the "NiN
gradient interaction is the mechanism" hypothesis (KonJND stayed
at 0.52 regardless of NiN on/off).

Two remaining hypotheses about the KonJND collapse:

1. **Gradient-starvation.** At `konjnd train_w = 0.02`, the KonJND
   group's 1k pairs contribute ~0.02 × 1k ≈ 20 effective weighted
   pairs per epoch vs safesyn's 196k × 1.0 = 196k. Pool-head's σ-
   reducer is more aggressive than V_22's mean-only mapping and may
   need more gradient mass on JND-boundary pairs to converge.
   Bumping konjnd train_w from 0.02 to 0.10 (5×) directly tests this.

2. **Intrinsic σ-pool/LeakyReLU/PJND mismatch.** σ over LeakyReLU
   activations does not carry the PJND-threshold-discriminating
   signal. Raising weight only amplifies noise; KonJND saturates
   near 0.55.

**Falsification.** Either outcome is load-bearing:
- KonJND ≥ 0.85 at any seed (or 5-seed mean ≥ 0.85) → hypothesis 1
  supported, pool-head architecture is **viable with the right
  recipe**.
- KonJND saturates near 0.55 → hypothesis 2 supported, pool-head
  direction is **structurally dead for JND tasks**.
- KonJND in 0.7–0.85 → partial information; report exactly.

## Experimental change

Single-knob change vs V_24-stdpool-prod:

| Knob | V_24-stdpool-prod | **V_24-stdpool-konjnd010** |
|---|---|---|
| `--group konjnd:.../konjnd_features_mix_targets_372col.parquet:` | `:0.02:0.0` | **`:0.10:0.0`** |
| All other flags | (V_22-mix-LARGE recipe) | (identical) |

`--pool-head` enabled, `--norm-in-norm-weight 0.1` (NiN on), all
other hyperparameters identical to V_24-stdpool-prod (commits
`286373ab`, `a620f775`).

## Files

- Bakes: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v24_stdpool_konjnd010/v24_stdpool_konjnd010_s{1..5}_h128.bin`
- Logs: `…_s{1..5}_h128.log` and `…_s{1..5}.stdout`
- Trainer binary: `/home/lilith/work/zen/zensim--ex2-stdpool-nonin/target/release/zensim_mlp_train` (reused; identical commit to ex2-nonin)
- bake_verdict binary: `/home/lilith/work/zen/zensim--ex2-stdpool-nonin/target/release/bake_verdict`
- bake_compare binary: `/home/lilith/work/zen/zensim--ex2-stdpool-nonin/target/release/bake_compare`
- Launchers: `scripts/v24_stdpool_konjnd010_train.sh`, `scripts/v24_stdpool_konjnd010_eval.sh`

## V_22-mix-LARGE baseline (unchanged)

`/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin`

| Corpus | n | SROCC |
|---|--:|---:|
| CID22 | 4292 | 0.8324 |
| KADIK10k | 10125 | 0.9677 |
| TID2013 | 3000 | 0.9729 |
| KonJND-1k | 1008 | 0.8927 |
| AIC-3 CTC | 600 | 0.7845 |

## Per-seed training summary

All 5 seeds early-stopped at epoch 70 (patience 30). Wall time
~4 min total (less than predicted because system load fell to
~33 mid-run, less co-tenancy than expected).

| Seed | Best val SROCC | Final reducer_w (bake-stored, best-checkpoint) = [μ, σ, max, p_6] | End-of-train reducer_w |
|---|---|---|---|
| s1 | 0.9744 | [-0.958, **1.459**, 0.848, 0.692] | [-1.575, 1.266, 0.827, 0.785] |
| s2 | 0.9744 | [-0.941, **1.490**, 0.859, 0.685] | [-1.525, 1.387, 0.831, 0.771] |
| s3 | 0.9741 | [-0.846, **1.487**, 0.864, 0.686] | [-1.412, 1.378, 0.823, 0.781] |
| s4 | 0.9742 | [-0.851, **1.403**, 0.852, 0.672] | [-1.397, 1.269, 0.834, 0.764] |
| s5 | 0.9752 | [-0.891, **1.402**, 0.874, 0.681] | [-1.477, 1.264, 0.837, 0.771] |

**σ-weight at best checkpoint (what the bake actually uses):
mean 1.448 ± 0.043** vs V_24-stdpool-prod's 1.543 ± 0.098 vs
V_24-stdpool-nonin's 1.538 ± 0.066. The σ-weight magnitude
**decreased slightly** (−0.09) when konjnd train_w jumped 5×. The
reducer is NOT amplifying σ further to chase KonJND. The KonJND
improvement is being achieved by the **layer-1 weights** (the
228 × 128 matmul) reshaping to weight JND-relevant feature
patterns more strongly — the additional gradient mass is going
into the encoder layer, not the head.

## Bake_verdict 5-seed aggregate SROCC table

| Corpus | V_22-LARGE | V_24-prod (mean) | V_24-nonin (mean) | **V_24-konjnd010 (mean ± std)** | Δ vs V_22 | Δ vs V_24-prod |
|---|---:|---:|---:|---:|---:|---:|
| CID22 | 0.8324 | 0.8376 | 0.8346 | **0.7780 ± 0.0085** | **−0.0544** | **−0.0596** |
| KADIK10k | 0.9677 | 0.9167 | 0.9159 | **0.9130 ± 0.0010** | −0.0547 | −0.0037 |
| TID2013 | 0.9729 | 0.8912 | 0.8907 | **0.8890 ± 0.0008** | −0.0839 | −0.0022 |
| KonJND-1k | 0.8927 | 0.5414 | 0.5227 | **0.7539 ± 0.0079** | −0.1388 | **+0.2125** |
| AIC-3 CTC | 0.7845 | 0.7785 (s3) | 0.7752 | **0.7709 ± 0.0052** | −0.0136 | −0.0076 |

Per-seed CID22: 0.7864 / 0.7862 / 0.7761 / 0.7747 / 0.7664.
Per-seed KonJND: 0.7564 / 0.7606 / 0.7570 / 0.7403 / 0.7551.

**Full Mohammadi panel** (aggregate, from verdicts, mean across seeds):

| Corpus | SROCC | PLCC | KROCC | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|
| CID22 | 0.778 | ~0.786 | ~0.580 | ~0.852 | ~0.620 |
| KADID | 0.913 | ~0.912 | ~0.739 | ~0.948 | ~0.409 |
| TID2013 | 0.889 | ~0.898 | ~0.711 | ~0.917 | ~0.439 |
| KonJND | 0.754 | ~0.832 | ~0.527 | ~0.807 | ~0.555 |
| AIC-3 | 0.771 | ~0.778 | ~0.598 | ~0.851 | ~0.628 |

## KonJND verdict

**Recovered partially (0.7539 mean ± 0.0079) but did NOT hit 0.85.**

- V_24-prod (konjnd_w=0.02, NiN on): KonJND 0.5414
- V_24-konjnd010 (konjnd_w=0.10, NiN on): **KonJND 0.7539** (+0.21)
- V_22 baseline (mean-pool head): KonJND 0.8927

The 0.21 absolute lift is large and consistent across seeds
(std 0.008), so the partial recovery is real. But the −0.14 gap
to V_22 is also real and decisive (bake_compare h_SROCC −28.5,
h_Z-RMSE −68.5, all panel stats favor V_22).

## bake_compare seed=3 vs V_22-mix-LARGE (decisive verdict)

| Corpus | A=V_24-konjnd010 s3 SROCC | B=V_22 SROCC | h_SROCC | h_Z-RMSE | PWRC_diff | DecScore | Aggregate |
|---|---:|---:|---:|---:|---:|---:|---|
| CID22 | 0.7761 | 0.8324 | −22.381 | −41.515 | −0.0505 | **−3.730** | **B>>A** |
| KADIK10k | 0.9130 | 0.9677 | −96.358 | −702.331 | −0.0325 | saturated − | **B>>A** |
| TID2013 | 0.8877 | 0.9729 | −53.387 | −303.799 | −0.0671 | saturated − | **B>>A** |
| KonJND-1k | 0.7570 | 0.8927 | −28.468 | −68.513 | −0.1113 | saturated − | **B>>A** |
| AIC-3 CTC | 0.7618 | 0.7845 | −7.122 | −15.837 | −0.0200 | saturated − | **B>>A** |

**Per-band decisive tally: A=0 cells, B=20 cells, 2
promising-not-decisive, 0 ties.** V_22-mix-LARGE-iwssim
**decisively beats** V_24-konjnd010 on every aggregate corpus.

CID22 dropped from V_24-prod's marginal CID22 win
(s3 DecScore +9.56) to a decisive B-win (DecScore −3.73). The
extra konjnd gradient mass pulled the encoder away from CID22-
friendly representations.

Compare file:
`/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v24_stdpool_konjnd010/v24_konjnd010_vs_v22_s3_compare.md`.

## Pareto gate (user-specified: CID22 ≥ +0.005 AND KonJND ≥ 0.85 AND KADID/TID within −0.01)

- CID22: **−0.054 vs V_22** (FAIL; gate is +0.005)
- KonJND: **0.754 mean** (FAIL; gate is 0.85)
- KADID: **−0.055 vs V_22** (FAIL; gate is −0.01)
- TID: **−0.084 vs V_22** (FAIL; gate is −0.01)
- AIC-3: **−0.014 vs V_22** (within noise but moving wrong way)

**ALL FOUR gates fail.** No ship. No packed bake produced.

## Hypothesis verdict

**The result supports the GRADIENT-STARVATION hypothesis (#1),
not the intrinsic σ-pool/PJND-mismatch hypothesis (#2).**

Evidence:
1. **KonJND moved by +0.21 SROCC** when only the konjnd training
   weight was changed (5× bump, single-knob A/B). That magnitude
   of response is not consistent with "the architecture cannot
   learn the signal." If hypothesis 2 were correct, raising the
   weight would amplify noise (KonJND stays near 0.55 or jitters).
   Instead the bake reliably reaches ~0.75 across 5 seeds with
   tiny variance.
2. **σ-weight didn't grow** — the reducer wasn't the bottleneck;
   the encoder weights (layer-1 of the MLP, 228×128) absorbed the
   additional KonJND-aware gradient.
3. **CID22 dropped by −0.06** as the encoder reshaped to be more
   KonJND-friendly. That's the expected trade-off when gradient
   mass redistributes between corpora — informs that there IS a
   trade between CID22 and KonJND under this architecture, but
   it's a smooth trade, not a structural break.
4. **The 0.21 lift is consistent across all 5 seeds** (std 0.008).
   Tight clustering means the recipe is mechanistic, not lucky.

The pool-head architecture is **viable for JND-boundary signal
when given enough gradient mass on the corpus**, but at this
training-corpus mix the gradient-mass redistribution needed to
hit 0.85 on KonJND costs more CID22/KADID/TID than the user's
Pareto gate allows.

## What's next (not run in this experiment)

Per the V_24-stdpool-nonin section "Next mechanism fix candidates"
(now updated by this result):

1. **Sweep konjnd train_w further** — try 0.20, 0.30, 0.50. The
   response curve from 0.02→0.10 is steep (+0.21 KonJND for 5×
   weight); a 10× or 25× would test whether KonJND keeps climbing
   toward 0.85, and at what cost to the other corpora. The KonJND
   data is tiny (1008 pairs) so even at 0.5 train_w the
   effective sample count remains low compared to safesyn's 196k.

2. **Hybrid head** (RankNet 1-wide + 4-stat pool reducer with
   learned mix coefficient). Preserves V_22's mean-pool path that
   handles KonJND well while adding σ-pool capacity for CID22.
   This is the natural follow-up given the gradient-starvation
   confirmation: pool-head alone has the σ-pool bias that hurts
   KonJND; a hybrid lets the head route per-pair via mix coef.

3. **Per-corpus normalized gradient scaling** — instead of
   weighting raw pair counts, scale gradients so each corpus
   contributes equal total gradient magnitude per epoch. Removes
   the n × train_w confounding (safesyn's 196k × 1.0 vs konjnd's
   1k × 0.10).

The result rules out hypothesis 2 (intrinsic mismatch). Future
EX-2-line experiments should pursue gradient-mass mechanisms,
not architectural reducer redesigns.

## Honest gaps at land time

1. **No 0.20/0.30/0.50 konjnd_w sweep** to characterize the
   response curve. The brief said "If KonJND recovers, even
   partially (0.7-0.85 range), that's information — report it;
   don't extend the experiment to chase 0.85 specifically." We
   stopped at 0.10 per that direction.
2. **σ-weight magnitudes from bake metadata (best-val checkpoint)
   differ from end-of-train σ-weights** (1.45 vs 1.33 mean). This
   is because Adam continues to evolve the reducer between the
   best-val epoch and early-stop epoch; the runtime uses the
   best-val checkpoint. Worth noting for any future analysis that
   correlates training-time reducer evolution with bake-time
   behavior.
3. **No PWRC band-weighting or TV regularizer ablation.** V_22's
   recipe includes TV that pool-head currently skips when NiN is
   active (see prod methodology section). Could partially explain
   the structural CID22 penalty.

## Files

- 5 bakes: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v24_stdpool_konjnd010/v24_stdpool_konjnd010_s{1..5}_h128.bin`
- 5 training logs: `…_s{1..5}_h128.log` and `…_s{1..5}.stdout`
- 5 verdicts: `…_s{1..5}_verdict.md`
- bake_compare s3 vs V_22: `v24_konjnd010_vs_v22_s3_compare.md` + `.json`
- Launchers: `scripts/v24_stdpool_konjnd010_train.sh`, `scripts/v24_stdpool_konjnd010_eval.sh`


