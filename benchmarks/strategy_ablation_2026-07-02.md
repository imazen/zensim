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

## Wave 3 (corpus isolation) — v51 corpus + finalist strategies

| cell | CID22 | KonJND | G1/mono |
|---|---|---|---|
| hponly51_s17 | 0.8391 | 0.4440 | 1.00/0.978 |
| **hponly51_s7** | **0.8767 — BEATS A (0.8657)** | 0.3184 (below A) | 1.00/0.975 |
| t1dro51_s17 | 0.8485 | 0.4295 | 0.99/0.975 |
| t1dro51_s7 | 0.8594 | 0.3796 | 0.99/0.975 |

- **Corpus effect confirmed**: same strategies on the v51 (pre-dedup) corpus
  gain ~+0.02 mean CID22 over the dedup+hqfill corpus — the knob-no-op
  duplicates act as regularization-by-upweighting; dedup is cleaner data but
  (as-is) worse training. Reconciling (dup-aware sampling weights on the
  dedup corpus?) is future work.
- **hponly51_s7 is the first cell above Profile-A on CID22** with a perfect
  dial — but its KonJND (0.318) is below A's 0.4185, so the pre-registered
  KonJND-no-regression gate is NOT yet passed. t1dro trades a little CID22
  for better KonJND, same pattern as wave-2b.
- Graduation seeds (s31/47/63 × both finalists) running — 5-seed stats,
  then full-panel + SDR25 + KonJND gate on the family winner.
- Wave-3's first launch PREFLIGHT-FAILED all 4 cells in 8s: the w3 manifests
  (generated from the pre-contract v51 base) lacked per-input contracts —
  the gate did its job; fix was declaring reality, not loosening.

## Graduation battery — t1dro51 five seeds (2026-07-03)

**CID22 five seeds**: .8485/.8594/.8708/.8483/.8625 → mean 0.8579 ± 0.0095,
zero collapses (hponly51 collapsed 1-in-5 at s47 → t1dro51 is the family
winner). Two seeds above A (0.8657); mean −0.008 below.

**SDR25 (the honest post-ssim2 holdout)** — scored via the feature path
(sdr25_features_372col + bake forward; path validated by A's feature-path
number reproducing its runtime 0.9036 to 4dp):

| bake | pooled SROCC | within-image |
|---|---|---|
| Profile-A | 0.9036 | 0.9976 |
| ssim2 | 0.958 | 1.000 |
| t1dro51 s17/s7/s31/s47/s63 | 0.9551/0.9642/0.9694/0.9640/0.9609 | 1.0000 ×5 |

**t1dro51 ≥ ssim2 on SDR25 at 4/5 seeds and > A at 5/5 (+0.06)** — the
pre-registered "SDR25 ≥ 0.958" RD gate is PASSED (A fails it at 0.904).

Pre-registered gate status (t1dro51):
- SDR25 ≥ ssim2: **PASS** (4/5 seeds; all 5 > A)
- dials (G1/mono): **PASS** (0.99+/0.973+ all seeds)
- CID22-49 ≥ 0.8854: OPEN (mean 0.858; best 0.871; A 0.8657 also fails this)
- KonJND ≥ A 0.4185: MIXED (0.31-0.44 across seeds)

Join gotcha recorded: ZENSIM_DIAL_PRED_OUT re-keys rows as (group ordinal,
local index); joining on assumed source order scrambles rows (caught because
A's per-pair values matched runtime exactly while ranks didn't; fixed with a
value-recovered permutation). Scoreboard tooling should join on echoed keys.

Recipe of record (t1dro51): v51 base + ema_decay=0.9, hard_pair_frac=0.5,
hard_pair_max_delta=0.05, stratified_bands=10, dro_eta=0.5, trainer 78ec8e61.
