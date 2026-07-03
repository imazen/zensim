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

## Three-way head-to-head (t1dro51_s31 vs Profile-A vs ssim2, per-pair, 2026-07-03)

Report: /mnt/v/output/zensim/reports/2026-07-03_compare_t1dro51_s31 (viewer
http://172.23.240.1:3300/zensim/reports/). ssim2 scored per-pair fresh via
`zenmetrics batch` (CID22 sanity: 0.8894 = its known number; A = 0.8657 exact).

| corpus | t1dro51_s31 | ProfileA | ssim2 |
|---|---|---|---|
| CID22-49 | 0.8708 | 0.8657 | **0.8894** |
| AIC-3 | **0.8013** | 0.7680 | 0.7965 |
| SDR25 | **0.9694** | 0.9036 | 0.9580 |
| KonJND | 0.3109 | 0.4185 | **0.4784** |

Candidate > ssim2 on 2/3 compression T0 holdouts (AIC-3, SDR25); ssim2 keeps
CID22-49 (+0.019) and KonJND. Candidate > A on 3/4. Tool:
scripts/v_next/metric_compare_report.py (ssim2 per-pair TSVs in the probe dir).

## Banding investigation (user report, 2026-07-03) — RESOLVED: corpus structure

The horizontal bands visible in the scatter pages are the SUBJECTIVE axes'
own quantization, not a metric artifact: CID22 has 542 distinct MCOS values
across 4,292 pairs (KonJND: 739/1,008), while every metric's prediction axis
is essentially continuous (t1dro51_s31: 4,011 unique values @3dp, top-20
modes hold 1.5%; ssim2: 4,070; A: 3,888 — no f16/spline quantization grid
detected, median inter-mode gap 0.006-0.05). Identical bands appear in the
ssim2 charts. Visual proof:
`/mnt/v/output/zensim/reports/2026-07-03_ssim2_baseline/banding_diagnosis.png`.

ssim2 now has its own standalone page in the viewer (same format/corpora,
KADID/TID omitted as in-sample): `2026-07-03_ssim2_baseline`.

## Ridge investigation (follow-up to banding) — NOT spline artifacts

Density ridges in the prediction axis were tested against each bake's PCHIP
dial-spline knot outputs (decoded from `zentrain.output_calibration_spline`:
18 (raw_x, dial_y) pairs; raw axis spans only 47.95..48.18 — the dial does
live in the spline) with a 2000-draw permutation null: mean ridge→knot
distance 1.58 vs null 1.39, p=0.75 (candidate) / 1.62 vs 1.41, p=0.70 (A) —
**no knot alignment**. Decisive cross-check: two independently trained models
place ridges at near-identical score positions (≈59.5/60.1, 63.9/64.2, ~72,
~76-77) ⇒ the ridges are CORPUS structure — CID22's fixed per-codec quality
ladders cluster true quality, and any accurate metric reproduces those
clusters. Full scatter-anatomy: horizontal bands = MCOS quantization (542
distinct values/4292); vertical ridges = quality-ladder clustering; no output
quantization (4k+ unique preds); no spline artifacts.

## Wave 4 final verdicts (t1dro51 base; base = 0.8485/0.8594 CID22, 0.430/0.380 KonJND)

| lever | CID22 s17/s7 | KonJND s17/s7 | verdict |
|---|---|---|---|
| delta003 (tight band) | **0.8700**/0.8515 | 0.376/0.427 | CID22 lever (+0.02 on weak seed) |
| delta008 (wide band) | 0.8549/0.8446 | **0.4642**/0.417 | KonJND lever (family best, ~free) |
| frac07 | 0.8541/0.8379 | 0.411/0.415 | wash — keep 0.5 |
| triplet on stack | 0.6473/0.5590 | 0.345/0.264 | FALSIFIED stacked (mild alone — interaction) |
| kagg 0.02/0.05 (head ACTIVE) | 0.4947-0.5574 | **0.097-0.227** | FALSIFIED — destroys both axes; α collapses to 1.0; replicates+worsens the v42 Pareto finding on this stack |

- **The mining band δ is a per-deployment CID22↔KonJND dial** — the wave's
  actionable positive.
- First w4 kagg run was a SILENT NO-OP (weight without pool parquet) —
  caught because cells reproduced base byte-identically; now a hard panic
  (122dc1e8). The corrected run is the honest falsification above.
- KonJND beyond ~0.46 on this architecture still looks representation-bound
  (consistent with the 2026-05-26 G5 characterization); remaining unexplored
  levers: EXP_II interior-pivot triplets (data exists, model deferred),
  konjnd-scoped hard-pair mining (needs per-group frac plumbing).

## Falsification audit (user challenge 2026-07-03: "misunderstandings or data errors?")

- **kagg verdict RETRACTED — INVALID-AS-TESTED (scale mismatch, config-error
  class).** The aggregation loss compares PINNED RAW output (mod.rs ~7620:
  `pin_forward(y)` per-ref means) against RAW PJND targets (22.5-70), but on
  the v47/t1dro recipe family the learned raw output spans only ~0.23 units
  around 48 (dial-in-the-spline, decoded this session). The MSE gradient is
  ~100x the signal scale → mutual destruction + α→1.0 — exactly what was
  measured. v42-era recipes had raw outputs on the PJND scale, hence their
  trade instead of destruction. Valid retest = compare on the DIAL output or
  map targets into raw-band units (design change, queued). The wave-4
  "FALSIFIED" row for kagg is superseded by this entry.
- triplet-on-stack: downgraded to "replicate before trusting" (2 seeds, no
  confirmed mechanism). listmle KonJND crater: stands (2-seed replicated +
  mechanism; within-ladder retest queued). ema/strat washes: stand. hpdro
  collapse: low-n caveat (1 event / 3 seeds).

## Dial-continuity vs the ridge artifact (user question 2026-07-03)

Measured on the densified dial grid (t1dro51_s31, 115 ladders, 4,702 adjacent
steps): JND zone (q70-90) median adjacent step 1.13pt, 2.6% flat, 1.7%
backwards; near-lossless median 0.21pt with 31% sub-0.1pt steps — EXPECTED
(sub-JND grid spacing; the truthful dial barely moves across visually
identical configs) — 3.0% backwards; full-range 2.3% backwards. Verdict: the
corpus-ridge artifact does NOT reduce dial continuity — between-ladder
response is smooth where targeting operates. True precision bounds: codec
action-space quantization (targeting-layer nearest-achievable) and the 2-3%
inversion tail (TV/within-ladder levers queued).

## Profile B (HDR-capable) — w5 first results (2026-07-03)

User directive: prep the HDR-capable candidate as **ZensimProfile::B** (new
slot; A unchanged). Corpus: imazen-26 HDR grid, PU21-fed 372 features,
dedup-by-content, LSD splits (train 3,420 / val 1,800 / test held).

| cell | hdr_val SROCC/PLCC | CID22 | AIC-3 | AIC-4 | KonJND |
|---|---|---|---|---|---|
| w5_hdronly_s17 (smoke) | 0.830 / 0.087 | — | — | — | — |
| w5_hdrmix_s17 (B seed) | **0.9694 / 0.9516** | 0.8544 | 0.7937 | 0.9041 | 0.4031 |

- SDR→HDR structure transfer is large (+0.14 SROCC vs hdr-only training);
  the SDR panel holds at same-seed parity → G-HDR-SDR-PANEL passes at s17.
- PU21 features carry real HDR signal (the existential check).
- Pre-registered remaining gates: G-HDR-SDR-CONSISTENCY (SDR-range content
  through HDR path == SDR path; harness queued in zenmetrics), multi-seed
  stability (s7/s31 launched), HDR dial continuity, UPIQ ≥ 0.694.

## Profile B v1 multi-seed + UPIQ + the v3 pivot (2026-07-03)

Multi-seed hdrmix: hdr_val 0.9694/0.9712/0.9694 (s17/s7/s31, PLCC ~0.954,
zero collapses). SDR panel: s17 0.8544 / s7 0.8470 (parity) but **s31 CID22
0.7976** (−0.073 vs its SDR base) — G-HDR-SDR-PANEL is seed-dependent;
held-out-val selection picks s17 as the B v1 candidate.

UPIQ-HDR (within-harness, shell features → bakes): B v1 0.6546, SDR-bake
0.6594, A 0.6459 — vs cvvdp 0.758 / iwssim-HDR 0.808 / ssim2-integrated
0.704. The documented ~0.05 u8-shell penalty reproduces at bake level →
**the shell features are the UPIQ bottleneck, not training.**

v3 pivot (docs-were-stale dig, user-directed): zensim's PU-linear path
already runs the FULL extended+iw feature engine (shared process_scale_bands;
only the zenmetrics feature branch + registry doc lagged). Landed:
`Zensim::compute_pu_linear_extended_features` (width-parity + identity
tests) + `score-pairs --hdr-features-pu-linear` (zenmetrics 6f591dd5) +
fleet BIN_KEY/EXTRA_ARGS params + merge_v3_shards.py. v3 re-extraction
fleets launched (v3june 4 + v3hq 6 cx boxes) + UPIQ local. v2
(anchored shell) is OBSOLETE — superseded by v3 before implementation.

## Profile B v3 first verdicts (PU-linear features, 2026-07-03)

| | B v1 s17 (shell) | B v3 s17 (pu-linear) | B v3 s7 |
|---|---|---|---|
| UPIQ-HDR | 0.6546 | **0.6819** | 0.6504 |
| KonJND | 0.4031 | **0.4481** (> A 0.4185) | 0.1221 |
| CID22-49 | 0.8544 | 0.8439 | **0.5753 COLLAPSED** |
| AIC-3 | 0.7937 | 0.7695 | 0.5276 |
| hdr_val (v3 set, incl. HQ zone) | — | 0.9009 | 0.8884 |

- Direction CONFIRMED: transfer-invariant features lift the human-MOS HDR
  holdout (+0.027) and KonJND (+0.045, now above A). Bake (0.682) still
  below its raw integrated input score (0.693) → hdr group weight is the
  principled lever (not UPIQ-fishing). Gap to cvvdp 0.758 = CSF mechanism
  (task #5), now buildable on the PU-linear path.
- **Stability: 1/2 seeds SDR-collapsed** (s7; hdr_val fine — the collapse is
  SDR-side). Seed fan s31/47/63 launched; selection remains held-out-val.
- Consistency harness green on the PU path (identity + rank).

## B v3 seed fan COMPLETE — selection-blindness discovered (2026-07-03)

| seed | CID22 | KonJND | UPIQ | best-val |
|---|---|---|---|---|
| s17 | 0.8439 | 0.4481 | 0.6819 | 0.8357 |
| s7 | **0.5753 COLLAPSE** | 0.1221 | 0.6504 | 0.8347 |
| s31 | 0.8398 | 0.2848 | 0.6412 | 0.8361 |
| s47 | 0.8416 | 0.3511 | 0.6639 | 0.8319 |
| s63 | **0.5420 COLLAPSE** | 0.1786 | 0.6548 | 0.8349 |

- **2/5 SDR-collapse rate on the v3 recipe**, and — the load-bearing finding —
  **the held-out-val selection metric CANNOT distinguish collapsed from
  healthy seeds** (Δ ≤ 0.0014 across all five). The val geomean's components
  don't measure the CID22/KonJND-shaped anchor structure that collapses.
  This extends the kb25/s7 lesson: a val group is only a guard if it spans
  the failure mode. NEXT WAVE MUST: add an anchor-shaped component to the
  selection geomean (audit which groups compose val(geomean3) first) before
  any further v3 recipe iteration; until then, seed selection for B v3 is
  by full panel, not by val metric.
- B v3 candidate by full panel: **s17** (UPIQ 0.6819, KonJND 0.4481 > A,
  CID22 0.8439). Healthy-seed SDR is tight (~0.84); HDR-side metrics carry
  the seed variance.

## Collapse instrumentation (2026-07-03, user directive)

**Audit — no training-val axis sees the collapse.** Per-group val SROCC at
each seed's SELECTED epoch (w6 fan): every gap between healthy and collapsed
is ≤0.016, and kadid/tid INVERT (collapsed seeds score higher). Root cause:
all val targets are metric-anchored (cid22_train = ssim2-anchored); the
collapsed bakes remain excellent metric predictors — what collapses is the
relation to HUMAN anchors (verdict CID22-MOS 0.84→0.54, AIC-3-JND, KonJND-
PJND all crater together). The ssim2-shaped-surface trap as an instability.

**Fix, verified on labeled data (2 collapsed + 3 healthy bakes):**
1. `konjnd_anchor` val-only guard group (canonical val/konjnd.parquet, mean
   PJND, human-derived, training-legal via konjnd_dense semi-status;
   val_w=4.0): per-bake |SROCC| separates healthy {0.285-0.448} from
   collapsed {0.122-0.179}, margin +0.106 — moves the selection mean by
   ~0.06 vs the 0.001 healthy jitter. w7 5-seed fan retraining under it.
2. runcells POST-TRAIN COLLAPSE GATE: auto bake_verdict floors (CID22<0.75
   or KonJND<0.20 → rc=9 + `<cell>-COLLAPSED` status row) — every future
   cell self-reports, fleet or local, no human in the loop.
3. CSF-feature direction claims CORRECTED (falsified v39-era AIC-3 spike;
   task deleted): the cvvdp gap is calibration + corpus, not CSF features.

## w7 guard verdict on the collapse seed (2026-07-03)

s7 under the konjnd_anchor guard STILL collapses (CID22 0.4452, AIC-4
0.3603, guard 0.1217) — **steering falsified**: no healthy epoch existed to
select; the whole trajectory is in the diverged basin (training-dynamics,
not late drift). The guard's real role is VISIBILITY (selection value
0.4962 collapsed vs 0.5100 healthy — ~3x the blind spread, directional but
not a hard discriminator); the runcells post-train verdict gate is the
REJECTION layer (fires decisively: 0.4452 < 0.75 → rc=9). Operating policy:
seed-fan + auto-reject; collapse-rate REDUCTION is a recipe-stability
question (first candidate: the w8 cvvdp-mix target, since ssim2-shaping is
the suspected basin driver).

## w7 guarded fan COMPLETE — instrumentation verdict (2026-07-03)

| seed | w6 CID22 | w7 CID22 | w7 KonJND | w7 sel-val |
|---|---|---|---|---|
| s17 | 0.8439 | 0.8378 | 0.3679 | 0.5100 |
| s7 | 0.5753 C | **0.4452 C** | 0.1217 | 0.4962 |
| s31 | 0.8398 | 0.8415 | 0.3556 | 0.4742 |
| s47 | 0.8416 | 0.8479 | 0.3232 | 0.5193 |
| s63 | 0.5420 C | **0.5182 C** | 0.2972 | 0.4866 |

1. **Basins are SEED-DETERMINISTIC**: the same 2/5 seeds collapse in both
   w6 (no guard) and w7 (guard) — initialization decides the basin;
   reproducible, not run-noise.
2. **Selection-steering falsified** (no healthy epoch exists on a collapsed
   trajectory) AND **cross-seed selection-ranking falsified** (healthy s31
   0.4742 < both collapsed seeds). The guard group in selection is neutral-
   to-mildly-useful in-training but is NOT a seed discriminator.
3. **The rejection layer is the instrument that works**: runcells post-train
   verdict gate fires on both collapsed seeds via the CID22<0.75 floor
   (note: s63's KonJND 0.297 is above the 0.20 floor — CID22 is the
   load-bearing floor; the OR-condition stands).
4. Healthy-seed quality under the guard: CID22 unchanged (~0.842), KonJND
   ~equal on average. The guard costs nothing.

**Operating policy (final)**: train seed fans; the gate auto-rejects
collapsed cells (rc=9); pick among survivors by full panel. Collapse-RATE
reduction = recipe-level work — first probe is w8 (cvvdp-mix target,
attacking the suspected ssim2-shaping basin driver), in flight.
