# P1 — full-data (KADIS-700k 720) lever + weak-zone attack (2026-07-25, autonomous run)

Autonomous experiment log (user: "proceed autonomously, keep track of experiments +
results in a file, don't stop to ask, keep moving"). Every experiment pre-registers a
hypothesis + kill band before unblinding, per `docs/ITERATION_PROTOCOL.md`.

## Grounding (what's already settled — do NOT re-litigate)

- **720-vs-372 is SETTLED** (`benchmarks/e2_optimal_model_720_vs_372_2026-07-23.md`):
  the appended v2-348 block does **not** beat v1 for RANK. Two instruments converge —
  multi-seed MLP (compression holdouts −0.002…−0.007, tight σ; FR inconclusive at n≈800)
  AND deterministic linear BVLS (compression Δ=−0.000 flat, FR/analytic materially worse,
  5.2× active weights = overfit). LOO: **banding actively harmful (+0.401)**, peak
  compression-only, transducer mildly harmful; basic/masked/iw/gms/ringing carry the
  model but do *v1's job*, not beating it.
- **So v2's justification is DIFFMAP COHERENCE, not RANK** (SESSION-RESUME: v2 = 100%
  spatializable at ~0 compression cost). That's a P2 question (coherent diffmap), separate.
- **Ship candidates (Ebothg/winner_dial) beat B everywhere EXCEPT KonJND** (0.27–0.34 vs
  B 0.55); docs say data-mass provably can't reach KonJND → needs a mechanism.
- **The #1 product blocker = diffmap≠scalar** (closed-loop can't steer).

## What the full KADIS-700k 720 uniquely enables (this run's questions)

The E2 verdict trained on ssim2-**compression** mass (tbig+safesyn+cid201) + kadid/tid
guards. It never had massive **analytic-distortion** mass. KADIS-700k 720 =
`/mnt/v/zen/zensim-training/kadis-720-2026-07-24/kadis700k_720.parquet` (699,999 cells,
720 feat + 7 metric targets, 24 IQA distortion families, split key `source_id`). Open
questions it can answer:

- **Q1 (weak zone):** does adding KADIS-700k analytic mass lift the weak corpora
  (KonJND, CSIQ, LIVE) **without poisoning** the compression holdouts (CID22, imazen26)?
- **Q2 (v2 rescue):** with analytic training mass, do the v2 analytic families
  (gms/ringing/transducer/blockiness) finally pay off (partially overturn E2)?
- **Q3 (KonJND mechanism):** KADIS near-threshold / severity range + `score_*` targets
  — is there a mechanism (not just data mass) that recovers KonJND?
- **Q4 (mix target):** KADIS `score_cvvdp`/`score_iwssim` as a mix target for the
  documented CID22-recovery lever.

Instrument: **deterministic linear BVLS** (`linear_projections_2026-07-03.py`, seed-free —
E2 proved MLP too seed-noisy on the small corpora). Verdict: `bake_verdict --regime 720`.

## Poison gate (MUST pass before KADIS touches training)

KADIS refs vs CID22-49 holdout — confirm perceptual-disjointness (dHash d≤10 + name).
Bans in force: CID22-49 human MOS never trains; KADIS is a METRIC-anchored analytic guard
(ssim2/mix target), never human. See `docs/DATASET_HISTORY.md` poison ledger.

## Experiments

| id | hypothesis | kill band | result |
|---|---|---|---|
| E-K0 | poison gate: KADIS ⟂ CID22-49 | any d≤10 dup → STOP | _pending_ |
| E-K1 | reproduce E2 lin372 baseline | must match E2 (CID22 0.794) | _pending_ |
| E-K2 | +KADIS ssim2 guard lifts weak zones, holds compression | CID22 drop >0.01 → KADIS poisons dial | _pending_ |
| E-K3 | +KADIS unlocks v2 (720+KADIS ≥ 372+KADIS) | flat/worse → v2 stays no-win | _pending_ |

## Results (running)

### E-K1 — kbase baseline (720 shaped, clean recipe, KonJND truly held-out)
`kbase` = t720_{safesyn 1.0, cid201 1.5, kadid 0.5, tid 0.5}; NO konjnd-in-train, NO bigcodec.
468/720 active weights. Bake `bakes/p1kadis/kbase_720.bin`.

| corpus | SROCC | note |
|---|--:|---|
| CID22 (gold) | **0.7983** | compression-human holdout |
| imazen26 real-codec | 0.8259 | compression ssim2 |
| imazen26 non-photo | 0.8240 | non-photo ssim2 |
| AIC-4 | 0.8697 | JND |
| AIC-3 | 0.7325 | JND |
| LIVE-R2 | 0.6200 | general-FR |
| CSIQ | 0.5460 | general-FR |
| **KonJND-1k** | **0.1269** | near-threshold — THE weak zone |
| KADID/TID | 0.049 / 0.338 | *train guards (memorization, ignore)* |

**Methodology finding:** E2's `twinsdr` trains on `ext_konjnd_jpeg_val` (the held-out set)
— a KonJND verdict leak. But it's **benign**: E2's konjnd-in-train linear got 0.05–0.18,
my clean held-out kbase gets 0.127 — same ballpark; the linear model can't fit KonJND
whether trained on it or not. My `kbase` is the honest baseline. (Fix twinsdr separately.)

### E-K0 — poison gate (KADIS ⟂ CID22-49)
Name-disjoint (0 intersection). dHash-64 on the **49-ref holdout** vs 140k KADIS refs:
d≤5=32, d≤10=747 (696 distinct KADIS refs). The matches are dHash false-positives on
flat/smooth refs (VAL `2887497` ~ BOTH `sunset-787012` AND `texture-1590106` at d=5 —
impossible for a real dup). Per policy (never auto-quarantine on dHash; d≤10 needs eye
review), **conservatively EXCLUDED all 696 flagged KADIS refs** (0.5% of 140k) → training
provably disjoint at d>10. Flagged list: `kadis-720-.../kadis_cid22_flagged.txt`. **PASS
(gated by exclusion).**

### E-K2 — +KADIS analytic-severe guard (50k clean, w=0.5) vs kbase — **STRONG WIN**
`kbase_kadis` bake `bakes/p1kadis/kbase_kadis_720.bin` (469/720 active).

| corpus | class | base | +KADIS | Δ |
|---|---|--:|--:|--:|
| CID22 (gold) | comp, GATED | 0.7983 | 0.8109 | **+0.013** |
| imazen26 real | comp ssim2 | 0.8259 | 0.8538 | +0.028 |
| imazen26 non | non-photo | 0.8240 | 0.8520 | +0.028 |
| AIC-3 / AIC-4 | JND | 0.733/0.870 | 0.748/0.872 | +0.015/+0.003 |
| KonJND | near-threshold | 0.1269 | 0.2304 | **+0.104** |
| CSIQ | analytic-FR | 0.5460 | 0.8880 | **+0.342** |
| LIVE-R2 | analytic-FR | 0.6200 | 0.9261 | **+0.306** |

**Kill band (CID22 drop >0.01 = poison): PASS — CID22 went +0.013.** KADIS does NOT poison
the dial; it HELPS every holdout.

**Interpretation (honest):** CSIQ/LIVE are the SAME analytic distortion families as KADIS
(blur/noise/jpeg/jp2k/contrast) → +0.3 is largely *distribution match* (legit but expected;
CSIQ/LIVE stop being "novel-distortion generalization" once KADIS trains). **The clean
generalization win is the OUT-of-KADIS-distribution holdouts** — CID22 (compression, gated)
+0.013, imazen26 (compression) +0.028, KonJND (near-threshold) +0.104 — all up, all on
disjoint distortion types. That's the real finding: the full KADIS analytic mass improves
zensim's generalization *including* on compression + near-threshold, not just analytic.
**CSIQ/LIVE overlap gate: d≤5=0 → CLEAN (no ref-leak); the +0.3 is genuine distribution-match, not contamination. KADIS dHashes cached at /mnt/v/datasets/kadis700k/kadis_dhash.npz.**

### E-K2b — KADIS weight sweep (w=0.5/1.0/2.0) — **Pareto, robust to weight**
| corpus | base | w0.5 | w1.0 | w2.0 | reading |
|---|--:|--:|--:|--:|---|
| CID22 (gold) | 0.798 | **0.811** | 0.809 | 0.804 | peaks w0.5, +0.006 even at w2.0 |
| imazen26_rc | 0.826 | 0.854 | 0.856 | 0.854 | flat +0.028 |
| KonJND | 0.127 | 0.230 | 0.245 | **0.262** | monotone ↑ (+0.135 at w2.0) |
| CSIQ | 0.546 | 0.888 | 0.910 | **0.924** | monotone ↑ (matched dist) |
| LIVE | 0.620 | 0.926 | 0.933 | **0.937** | monotone ↑ |
| AIC-3 | 0.733 | 0.748 | 0.750 | 0.752 | ↑ |

**Every weight beats baseline on every corpus.** w0.5 = max CID22; w1.0 = clean balance
(CID22 +0.011, KonJND +0.118, CSIQ 0.910). The KADIS guard is a Pareto improvement, not a
compression↔analytic trade — analytic mass helps the compression + near-threshold holdouts too.

### E-K3 — does KADIS need v2? (372 vs 720, ±KADIS) — **YES: the full data RESCUES v2**
| corpus | 372 base | 372+K | Δ372 | 720 base | 720+K | Δ720 |
|---|--:|--:|--:|--:|--:|--:|
| CID22 | 0.787 | 0.775 | **−0.012** | 0.798 | 0.811 | **+0.013** |
| imazen26_rc | 0.796 | 0.810 | +0.014 | 0.826 | 0.854 | +0.028 |
| KonJND | 0.187 | 0.155 | **−0.031** | 0.127 | 0.230 | **+0.103** |
| CSIQ | 0.657 | 0.799 | +0.142 | 0.546 | 0.888 | +0.342 |
| LIVE | 0.843 | 0.904 | +0.061 | 0.620 | 0.926 | +0.306 |

**KADIS HELPS at 720 but HURTS at 372 on CID22 (−0.012) + KonJND (−0.031); and helps
2–5× MORE at 720 on the FR corpora.** Mechanism: v2's analytic families (gms/ringing/
transducer/blockiness) give the model capacity to ABSORB the analytic training signal
without cannibalizing v1's compression capacity. With v1-only (372), KADIS pulls the fit
toward analytic at the expense of compression/near-threshold; with v2 (720), the v2 block
routes it → Pareto.

## CONCLUSION (reframes the feature-v2 program)

**E2's "v2-348 does not beat v1" verdict was CONDITIONAL on compression-only training.**
It lacked analytic training mass, so v2's analytic families had nothing to represent.
Add the full KADIS-700k analytic mass and **v2 becomes load-bearing** — 720+KADIS is a
clean Pareto win over every prior arm (CID22 0.811 > 0.798 base > E2's 720 no-KADIS;
KonJND 0.230 vs 0.127; FR ~0.9), while **372+KADIS is NOT** (hurts CID22/KonJND). So:
1. **v2 keeps its RANK justification** (not just diffmap coherence) once analytic data exists.
2. **KADIS-700k is a first-class training corpus** (metric-anchored analytic guard), not just
   the corruption-head negatives — it's a Pareto lever, ref-disjoint from CID22 (gated), no leak.
3. **KonJND is reachable** — the "data-mass can't reach it" claim was about the WRONG data
   (compression); analytic-severe + near-threshold mass nearly doubles it (0.127→0.262).

**Next (queued):** apply KADIS to the ship recipe (does it lift the Ebothg/winner_dial ship
candidates' KonJND without CID22 cost?); KADIS mix-target (cvvdp/iwssim) for CID22; full-700k
mass vs 50k. Bakes: `bakes/p1kadis/`. Tool groups/mixes: `t720_kadis`, `kbase[_kadis[12]]`.

### E-K2c — KADIS 50k→200k scale (w0.5) — more data = same trend as more weight
| corpus | 50k | 200k | Δ |
|---|--:|--:|--:|
| CID22 | 0.811 | 0.805 | −0.005 (still +0.007 vs base) |
| imazen26_rc | 0.854 | 0.855 | +0.001 |
| KonJND | 0.230 | 0.267 | **+0.036** |
| CSIQ | 0.888 | 0.925 | +0.037 |
| LIVE | 0.926 | 0.937 | +0.010 |

Data-volume is the same lever as weight: more KADIS → more KonJND/FR, tiny CID22 cost, always
above baseline. **Operating point:** 50k@w0.5 max-CID22; scale KADIS up when KonJND/FR matter more.

**⚠ tool gotcha:** `gram --force` at one `ZLIN_NFEAT` clobbers the other's cached grams (cache is
keyed by group, not N_FEAT). E-K3's 372 grams clobbered the 720 set → restore with a 720
`gram --force` before any 720 twin. (Fix: key the gram cache on N_FEAT — queued.)
