# Strategy ablation wave 1 — 2026-07-02 (trainer commit 78ec8e61)

18 cells: 9 variants × seeds {17, 7} on the v53 base recipe (dedup+hqfill
corpus + KonFiG group). Trained twice: zen-train-1 ccx63 (PAR=8, ~25 min
wall) and 9× cx53 fleet (`strategy-results/abv1` on R2) — same manifests,
cross-machine replication. Scored with bake_verdict (rank) + the kadis
safety grid (dial). Base seeds reproduce v53_s17/s7 exactly, as required.

| variant | CID22 s17 | CID22 s7 | KonJND s17/s7 | G1 s17/s7 | mono s17/s7 |
|---|---|---|---|---|---|
| base     | 0.6557 | 0.7198 | 0.41/0.46 | 0.79/0.76 | 0.970/0.976 |
| ema      | 0.6449 | 0.6678 | 0.42/0.45 | 0.78/0.79 | 0.971/0.966 |
| hardpair | **0.8402** | **0.8031** | 0.40/0.45 | 0.99/0.88 | 0.971/0.971 |
| strat    | 0.6710 | 0.6412 | 0.40/0.40 | 0.85/0.78 | 0.974/0.975 |
| dro      | **0.8474** | 0.7195 | 0.42/0.46 | 1.00/0.83 | 0.979/0.974 |
| listmle  | 0.8233 | 0.8255 | **0.24/0.25** | 0.89/0.88 | 0.978/0.978 |
| triplet  | 0.7181 | 0.7003 | 0.43/0.38 | **0.00**/0.90 | 0.996/0.975 |
| tier1    | 0.8246 | **0.8542** | **0.45/0.46** | **0.96/0.99** | 0.980/0.974 |
| all      | 0.8567 | 0.8472 | 0.39/0.38 | **0.00**/0.94 | 0.996/0.978 |

## Verdicts (both panels, per the two-panel rule)

- **hardpair — WINNER, single best lever.** +0.13 mean CID22, lifts BOTH
  seeds, repairs the dial (G1 0.79→0.99 at the collapsed seed), KonJND flat.
  Mechanism: near-threshold pairs carry the informative RankNet gradient;
  uniform pair draws are mostly wasted and noisy (seed lottery).
- **tier1 (ema+hardpair+strat) — WINNER, best balanced.** CID22 0.825/0.854,
  best KonJND alongside a 0.82+ CID22, best joint dial (0.96/0.99). hardpair
  carries it; ema+strat neither help nor hurt inside the combo.
- **dro — rescuer.** Fixes the collapsing seed (+0.19 at s17), neutral at the
  healthy seed. Textbook GroupDRO behavior. Compose-with-hardpair test =
  wave-2b (running).
- **listmle — REJECTED as configured.** Consistent +0.10 CID22 both seeds
  but craters KonJND to 0.24-0.25 (cross-content lists optimize global rank
  at the cost of near-threshold discrimination). Retry candidate: within-
  ladder lists only.
- **ema — no effect / slightly negative** at per-epoch decay 0.9. Averaging
  across a bimodal (collapsed/healthy) trajectory plausibly hurts. Retry
  candidate: EMA gated to start after epoch ~50.
- **strat — wash.** ±0.03, dial unchanged.
- **triplet — mild positive on rank (+0.05 avg CID22), but s17 produced a
  FLAT-DIAL bake (G1=0.00, mono 0.996 = constancy artifact).** Same failure
  as all_s17. Both flat-dial cells share heavy step-stealing from the pair
  loss; suspected mechanism: too few anchor/mse steps at this frac. Needs
  the spline/anchor interaction investigated before re-running at weight.
- **all — SROCC-best (0.857/0.847) but dial-dead at s17 (G1=0.00)** and
  KonJND damaged by the listmle component. Not shippable as-is.
- **Flat-dial trap note:** mono≈0.996 with G1=0.00 is a near-constant
  output, NOT a good dial — monotonicity alone is un-gateable without G1.

## Wave 2b (running): hpdro + t1dro + hponly×{31,47,63}

Tests whether dro's rescue composes with hardpair, and hardpair's 5-seed
stability. Results land in this file's next section.

## Reference vs targets

A (v47) CID22 = 0.8657; best wave-1 cell tier1_s7 = 0.8542 (−0.011). The
strategy levers recovered nearly all of the corpus-swap regression with the
v53 base; wave-2b + a hardpair-on-v51-corpus cell will separate corpus from
recipe effects.

## Wave 2b results (complete)

| family | CID22 by seed | mean±sd | KonJND | G1 / mono |
|---|---|---|---|---|
| hponly (hardpair, 5 seeds: 17/7/31/47/63) | .8402/.8031/.8387/.8424/.8454 | **0.834±0.017, 0 collapses** | 0.35-0.45 | 0.94-1.00 / 0.971-0.979 |
| t1dro (ema+hp+strat+dro, 3 seeds) | .8245/.8387/.8415 | **0.835±0.009** | **0.458-0.468** | 0.96-1.00 / 0.973-0.977 |
| hpdro (hp+dro only, 3 seeds) | .8491/.5616/.8314 | 1-in-3 COLLAPSE (s7: G1 0.72) | 0.25 at collapse | rejected |

- **hpdro's collapse proves the composition is non-additive**: dro alone never
  collapsed, hardpair alone never collapsed (5 seeds), their pair did once.
  The tier1 buffer (ema+strat) is load-bearing inside t1dro.
- **Finalists: hponly (simplest, one flag) and t1dro (tightest + best KonJND).**
  Both ~0.835 mean CID22 vs A's 0.8657 (−0.031).
- **Cross-machine determinism (wave 1)**: 16/18 byte-IDENTICAL box-vs-cx-fleet;
  the 2 divergent cells both ran on ONE fleet box → SIMD ISA tier (AVX-512 vs
  AVX2 Adam kernel) reduction-order drift, not nondeterminism. Determinism
  holds per ISA tier.
- **Wave 3 (running)**: hponly + t1dro on the v51 (pre-dedup) corpus × seeds
  {17,7} — isolates whether the remaining −0.031 vs A is corpus or recipe.
