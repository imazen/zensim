# V43 — G5 regime-routed 2-bake ensemble {V39, KonJND-HF-specialist}

**Date:** 2026-05-26
**Goal:** CODEC_TARGET_GOALS.md **G5 (KonJND-1k val SROCC ≥ 0.70)** without
regressing the other 5 corpora — via a 2-bake ensemble that routes
near-lossless/HF inputs to a KonJND-specialist and mid-fidelity inputs
to the shipped V39.
**Verdict:** **FALSIFIED offline.** No router/combiner clears
`KonJND ≥ 0.70 AND CID22/KADIK/TID/AIC-3/AIC-4 each within −0.01 of V39`.
Nothing wired into the runtime (the wiring gate only fires on an
offline pass, per the task brief). Characterized negative.

## Hypothesis / falsification (Step 1)

- **Hypothesis:** a 2-bake ensemble routing HF (near-lossless, KonJND-
  like) inputs to a KonJND-specialist and everything else to V39 gets
  **both** KonJND ≥ 0.70 AND the other 5 ≈ V39 (within −0.01).
- **Falsification:** if no router (score-gate OR learned classifier) and
  no specialist (w002/w005/w01) clears the bar across a swept threshold
  grid, the regime is not separable and the hypothesis is dead.
- **Cost ceiling:** reuse existing v42 specialist bakes + offline
  combiner; no new training unless the offline test passes.
- **Ship form:** runtime profile variant **only if** offline clears the
  bar.

## Inputs (no new bakes trained — v42 specialists reused)

| Bake | Path | Standalone aggregate SROCC (this scoring path) |
|---|---|---|
| **V39** (general, PreviewV0_3) | `zensim/weights/v39_v32plus_spline_seed17_2026-05-25.bin` | cid22 0.8793 · kadid 0.9251 · tid 0.9317 · **konjnd 0.4197** · aic3 0.8023 · aic4 0.9051 |
| **w002** (V39 recipe + konjnd-agg w=0.02) | `/mnt/v/output/zensim/bakes/v42_konjnd_agg_w002_2026-05-26.bin` | cid22 0.874 · konjnd 0.574 · aic3 0.790 |
| **w005** (… w=0.05) | `/mnt/v/output/zensim/bakes/v42_konjnd_agg_w005_2026-05-26.bin` | cid22 0.732 · konjnd 0.842 · aic3 0.569 |
| **w01** (… w=0.10) | `/mnt/v/output/zensim/bakes/v42_konjnd_agg_w01_2026-05-26.bin` | cid22 0.283 · konjnd 0.854 · aic3 0.065 |

Per-pair scores dumped via `ensemble_score_rows` (bit-exact with the
runtime `forward_one_bake`, incl. per-sample-α head) against the
372-feature val parquets at `2026-05-15-full-features/`, the same root
V39's baseline was measured on. Scores at
`/mnt/v/output/zensim/g5_ensemble_2026-05-26/scores/{corpus}_{bake}.tsv`.

## Bake-shape diagnostic (Step 4) — the load-bearing finding

**Both bakes' RAW output is a near-flat band** on this scoring path:

| Corpus | V39 raw [min,max] (std) | w01 raw [min,max] (std) |
|---|---|---|
| cid22 | [49.650, 49.879] (0.040) | [42.666, 43.197] (0.084) |
| konjnd | [49.713, 49.846] (0.020) | [42.672, 43.265] (0.151) |
| aic3 | [49.659, 49.833] (0.036) | [42.697, 43.141] (0.092) |

The dial we see at runtime (G1=1.00) comes from the PCHIP spline /
per-corpus logistic rescale, **not** from the raw output spread. The
bakes carry rank in a 0.2-wide raw band. **Consequence:** blending two
near-flat bands in raw score space (`g·spec + (1−g)·v39`) scrambles
within-corpus rank — so any soft/continuous blend destroys the
specialist's KonJND ordering. **Only a hard switch** (pick one bake's
full ordering per pair) preserves rank. This kills option (b)'s blend
forms and forces a hard router.

## Option (b): score-gate on V39 prediction level — FALSIFIED

`combine = sigmoid((v39_pred − center)/width)·spec + (1−g)·v39`, swept
center∈{40..85}, width∈{3..20}, all 3 specialists. Best "others-within-
−0.01" passer reaches **KonJND ≈ 0.48–0.51** — far below 0.70. `max(V39,
spec)` control leaves KonJND at 0.420 (specialist doesn't produce higher
raw scores on KonJND pairs). Root cause: V39's raw level is ~constant
per corpus (≈49.8 everywhere), so the gate cannot distinguish KonJND
pairs from CID22/AIC-3 near-lossless pairs — the regime is not separable
in prediction-level space. Log: `sweep_w01.log`.

## Option (a): learned 372-feature router (konjnd-vs-rest) — CLOSE, still FAILS

1-hidden-layer MLP (64u) on f0..f371, label=1 for KonJND pairs, trained
on an 80% stratified split, evaluated on the held-out 20% AND full
corpus. Hard route to specialist when `p(konjnd) > tau`.

**Held-out 20%, w01 specialist, tau=0.5 (the principled report):**

| Corpus | route% | V39 SROCC | Ensemble SROCC | Δ |
|---|--:|--:|--:|--:|
| cid22 | 1.0% | 0.8857 | 0.8580 | **−0.0276** ✗ |
| kadid | 0.3% | 0.9223 | 0.9225 | +0.0002 ✓ |
| tid | 0.3% | 0.9298 | 0.9293 | −0.0005 ✓ |
| **konjnd** | 90.5% | 0.4570 | **0.7014** | **+0.2444** ✓ (≥0.70) |
| aic3 | 0.0% | 0.7912 | 0.7912 | +0.0000 ✓ |
| aic4 | 0.0% | 0.8879 | 0.8879 | +0.0000 ✓ |

KonJND clears 0.70, and 4 of 5 others are within −0.01 — **only CID22
breaks** (−0.028). On the FULL corpus at tau=0.5: KonJND 0.734, but
CID22 −0.0142 **and** AIC-3 −0.0157 break the bar.

**tau sweep (FULL corpus, w01) — the unavoidable crossover:**

| tau | cid22 | konjnd | aic3 | KonJND≥0.70? | others_ok? |
|--:|--:|--:|--:|:--:|:--:|
| 0.50 | 0.865 | 0.734 | 0.787 | ✓ | ✗ (cid22/aic3) |
| 0.60 | 0.868 | 0.677 | 0.787 | ✗ | ✗ |
| 0.70 | 0.871 | 0.606 | 0.794 | ✗ | ✓ |
| 0.80 | 0.874 | 0.493 | 0.802 | ✓ standalone-diluted | ✓ |

The tau where KonJND first clears 0.70 (≤0.55) is always below the tau
where CID22+AIC-3 recover (≥0.70). **No tau passes both.** w005 behaves
identically; w002 is too weak (KonJND maxes at 0.469 even fully routed).

## Refinement attempts (experiment-rigor extension) — all fail

`g5_router_refined_2026-05-26.py`, w01: (R1) soft-blend on routed pairs
and (R3) continuous prob-weighted mix both **collapse KonJND to
0.05–0.12** — confirming the flat-raw-band finding (blends scramble
rank). (R2) high-tau hard route reproduces the crossover above.

## Mechanism — fully quantified

At tau=0.5 (FULL), only **26 of 4292 CID22 pairs (0.61%)** and **2 of 600
AIC-3 pairs** misroute to the specialist — but those 26 CID22 pairs sit
at mid-to-high quality (MCOS 0.47–0.83), i.e. the near-lossless pairs
whose **features overlap the KonJND regime**. The specialist ranks them
by its CID22-broken ordering (standalone 0.28), creating rank inversions
that drag aggregate CID22 SROCC down −0.014 (FULL) / −0.028 (held-out).
The damage is **specialist-independent** (same misroutes for w002/w005/
w01) and **mode-independent** (hard route is the only rank-preserving
mode, and it can't avoid the false positives).

## Why the regime is not cleanly separable

KonJND-1k is a JPEG/BPG dataset of **near-visually-lossless** pairs. Its
feature signature overlaps CID22's and AIC-3's HF (near-lossless) tails.
A router loose enough to capture ≥90% of KonJND (needed for SROCC ≥0.70)
necessarily captures ~0.6% of CID22 and ~0.3% of AIC-3 — and at CID22's
n=4292 and the specialist's catastrophic CID22 rank, even that tiny
false-positive rate exceeds the −0.01 bar. This is the Mohammadi 2025
HF-overlap problem (G5 doc §): "learning-based metrics systematically
underperform … near-lossless because their training data rarely includes
near-lossless pairs" — the specialist that fixes KonJND is, by
construction, the one that mis-orders CID22's near-lossless pairs.

## Verdict & what would be needed

**FAIL the bar.** Best achievable held-out: KonJND 0.70 at CID22 −0.028
(or FULL: KonJND 0.73 at CID22 −0.014, AIC-3 −0.016). A 2-bake
regime-routed ensemble does **not** close G5 cleanly with the available
specialists. Not shipped; not wired into the runtime.

Open directions (NOT pursued — each needs new evidence per Step 10):
1. A KonJND-specialist that does NOT crater CID22 — i.e. break the v42
   Pareto tension at the training step (the specialist is the problem,
   not the router). Per the v42 doc this needs a decoupled aggregation-
   only head / gradient decoupling, not a recipe-weight tweak.
2. A 3-way ensemble where the "HF" bake is good on CID22's HF tail too
   (so misroutes are cheap) — requires an HF bake that isn't KonJND-
   overfit.
3. The structural fix the G5 doc itself flags: acquire near-lossless HF
   training pairs so the GENERAL bake's KonJND rank improves, removing
   the need for a specialist at all.

## Artifacts

- Combiner scripts: `scripts/v_next/g5_regime_gate_ensemble_2026-05-26.py`
  (score-gate), `scripts/v_next/g5_classifier_router_2026-05-26.py`
  (learned router), `scripts/v_next/g5_router_refined_2026-05-26.py`
  (R1/R2/R3 refinements).
- Per-pair scores + logs: `/mnt/v/output/zensim/g5_ensemble_2026-05-26/`.
- V39 baseline verdict: regenerated via `bake_verdict`; numbers match
  `benchmarks/v5_vs_v03_comparison_2026-05-25.md` FRESH 2026-05-26 rerun
  (KonJND 0.4197, CID22 0.8793).
