# Rev4 E4: two-reference agreement as a selection gate, a retrospective (2026-09-23)

**Question (Q4).** Does a candidate's disagreement with the two-reference rule on the TRAIN-side
ladder predict its held-out human outcome and its dial-contract outcome better than the
registered composite does? The two-reference rule applies where SSIMULACRA2 and
butteraugli-pnorm3 agree on an order and the candidate reverses it.
Prereg: [rev4_e4_prereg_2026-09-23.md](rev4_e4_prereg_2026-09-23.md), committed before any
human label was read. The numbers are in [rev4_e4_agreement_gate_2026-09-23.json](rev4_e4_agreement_gate_2026-09-23.json).
Commands and hashes are in [rev4_e4_WORKLOG.md](rev4_e4_WORKLOG.md).

## Verdict

- **Held-out human outcomes: negative.** Disagreement predicts none of CID22-A(25), AIC-3,
  KonJND-504 or CSIQ beyond the composite: every partial-SROCC CI at the preregistered Bonferroni
  level contains zero. Adding agreement to the composite makes CID22-A prediction *worse*:
  Δ = −0.229, 95% CI [−0.401, −0.012].
- **TRAIN-side floor outcome (D2, A7r floors): strongly predicted, but circular.**
  - The partial SROCC is +0.697, Bonferroni CI [+0.359, +0.820].
  - This one outcome triggers the preregistered rule. The rule's mechanical label is
    **"ADOPT, circularity unresolved"**: the butteraugli-only arm does not hold on D2
    (+0.231, 95% CI [−0.122, +0.554]).
  - D2 is not a held-out outcome. It is measured on the same ladder instrument, with ssim2 as
    the floor mentor.
- **Answer to the program's decision rule** ("predicts held-out outcomes the composite misses"):
  **not met, so record a negative result.** The floor finding restates what the existing A7r and
  C1 gates already measure. It is not a new selection signal.
- Agreement never becomes a training signal. Nothing here changes that.

## 1. What was measured

- **Candidates.** The graded cells of the ladder board of 2026-09-06.
  - MAIN: 359 cells at width 944 in the "immune" era. All were scored on the one 944 ladder
    instrument (`0e8e5fb7…`, 9,593 cells, 9,411 adjacent pairs).
  - SECONDARY: 60 deduplicated width-372 cells, scored on the 372 ladder instrument.
  - Excluded: the 22 pools-era cells and the two peers.
- **Predictors.** Per-cell ladder scores come from `ensemble_score_rows`. They reproduce the
  board's stored C1 (`mono_agree`) with |Δ| = 0 on **448/448** graded cells, so the predictor
  side is the board's own measurement.
  - P_dis: the exact disagreement rate over the 5,232 pairs where both references agree
    materially.
  - P_dial: 1 − `mono_agree`.
  - P_bu: butteraugli-only, over 5,619 pairs.
  - P_s2: ssim2-only, over 7,416 pairs.
  - P_dis_nofloor: P_dis without each ladder's bottom three pairs.
- **Outcomes.**
  - H1 CID22-A(25): SROCC recomputed from A-only rows. CID22-B was never scored.
  - H2 AIC-3, H3 KonJND-504 and H4 CSIQ: the stored per-candidate SROCCs.
  - D1: a measured contract failure in C2–C6. C1 is excluded because it is P_dial.
  - D2: the A7r mean represented fraction over the five codecs.
- **Root identity.** Each cell's CID22-A rows were scored on the feature root whose TRAIN-role
  KADID SROCC reproduces the cell's stored KADID to ≤1e-6. This held for 421/426 cells,
  including all 359 MAIN cells (355 on `ext944-canonical-2026-08-01`).
- **Composite.** C_A uses the registered `balanced_composite` weights, with CID22 re-cut to A(25)
  and the CID22-49 band-tail term omitted. C_LOO drops the outcome's own term.
- **Statistics.** All SROCCs come from `panel`.
  - The bootstrap is lineage-clustered: 19 lineages in MAIN (the effective n), B = 4000,
    seed 20260923.
  - Partial ρ(−P, Y | C_LOO) is computed from the three owner SROCCs of each resample.

## 2. MAIN results (n = 359 cells, 19 lineages; every row uses all 359)

Signs: positive means "lower disagreement / higher composite goes with a better outcome".
The table uses 95% CIs; the Bonferroni CIs are in the JSON.

| outcome | ρ(C_LOO, Y) | ρ(−P_dis, Y) | partial P_dis \| C [95%] | partial P_dis Bonferroni | partial P_bu \| C [95%] | Δ(Combo − C) [95%] |
|---|---|---|---|---|---|---|
| H1 CID22-A(25) | **+0.593** [+0.329, +0.772] | −0.009 | −0.199 [−0.372, +0.150] | [−0.434, +0.217] | +0.004 [−0.228, +0.227] | **−0.229 [−0.401, −0.012]** |
| H2 AIC-3 | **+0.658** [+0.400, +0.811] | +0.006 | −0.167 [−0.453, +0.238] | [−0.553, +0.351] | −0.052 [−0.302, +0.225] | −0.235 [−0.506, +0.048] |
| H3 KonJND-504 | **+0.597** [+0.300, +0.724] | +0.318 | +0.272 [−0.107, +0.472] | [−0.189, +0.537] | +0.075 [−0.118, +0.271] | +0.006 [−0.150, +0.162] |
| H4 CSIQ | **+0.460** [+0.165, +0.746] | +0.437 | +0.408 [+0.019, +0.616] | [−0.049, +0.685] | +0.029 [−0.211, +0.337] | +0.128 [−0.150, +0.361] |
| D1 no contract fail | **−0.380** [−0.485, −0.053] | +0.070 | +0.159 [−0.037, +0.393] | [−0.083, +0.443] | +0.446 [+0.151, +0.561] | +0.158 [−0.010, +0.380] |
| D2 floors (A7r) | +0.179 [−0.150, +0.602] | **+0.708** | **+0.697 [+0.438, +0.787]** | **[+0.359, +0.820]** | +0.231 [−0.122, +0.554] | **+0.401 [+0.087, +0.605]** |

How to read the table:

- The composite predicts every held-out human outcome, with CIs above zero, even after its own
  term is removed. Disagreement adds nothing on CID22-A or AIC-3; on those two its partial
  effect is negative.
- On KonJND-504 and CSIQ the partial effect is positive but does not survive the correction.
  - The CSIQ 95% CI does exclude zero.
  - That effect is fragile. Leave-one-lineage-out (exploratory) moves the CSIQ partial between
    +0.126 and +0.504, and removing `sota944_C` takes most of it.
- The composite does not see the dial.
  - Its association with D1 is negative: higher-composite lineages (LSTAR, W10) carry most of
    the 46 contract failures.
  - Its association with D2 is null.
- The alternative predictors (P_dial, P_s2, P_dis_nofloor) behave like P_dis. None reaches the
  Bonferroni level on a human outcome. P_dis_nofloor keeps D2 at +0.685, Bonferroni
  [+0.339, +0.812].

## 3. Gate view (prereg §7; descriptive)

- Threshold: τ = the median P_dis = 0.00994.
- "Failed" means D1 = 0, or a held-out mean percentile rank in the bottom tertile.

| gate on MAIN (n = 359) | reject & failed | reject & ok | keep & failed | keep & ok |
|---|--:|--:|--:|--:|
| agreement: P_dis > τ | 97 | 76 | 64 | 122 |
| composite: C_A < median | 115 | 64 | 46 | 134 |

The composite gate rejects more of the failures and fewer of the good candidates.

- **Caveat:** "failed" partly uses held-out ranks that the composite contains, which
  tilts this comparison toward the composite.
- **Exploratory, contract failures only (D1):**
  - P_dis > τ rejects 28 of 46 failures, and rejects 145 of 313 non-failures.
  - A butteraugli-only gate (P_bu > its median 0.0169) rejects 45 of 46 failures, and rejects
    131 of 313 non-failures.
  - The failures cluster in a few lineages, so this is a lead, not evidence.

## 4. Circularity

- **ssim2 is on both sides of D2.** SSIMULACRA2 labels trained most of these candidates. It is
  also one reference in the rule, and it is the mentor in A7r's `resolvable` floor rule.
- **D2 also shares the instrument.** A7r checks strict ordering in the bottom rungs of the same
  ladder that P_dis is counted on.
- **Removing the overlap does not remove the effect.** Excluding those rungs leaves +0.685, but
  both quantities still measure agreement with ssim2's ordering on the same images and codecs.
- **The sensitivity arm fails on D2.** Removing ssim2 from the predictor (P_bu) drops D2 to
  +0.231, with a CI that spans zero. The strong D2 association is therefore best read as ssim2
  consistency measured twice.
- **The human-outcome null is not a circularity artefact.** The composite that wins on those
  outcomes puts 0.80 of its 2.45 weight (33%) on ssim2-labelled axes (imazen26 + nonphoto). It still
  beats agreement on CID22-A and AIC-3, where ssim2 plays no part in the labels.

## 5. SECONDARY (width 372; 60 cells; 15 lineages)

- Same pattern as MAIN. The composite leads on AIC-3 (+0.690) and CSIQ (+0.810).
- No human outcome has a partial CI that excludes zero.
- D2 is again strongly predicted: partial +0.749, Bonferroni [+0.405, +0.979]. This time every
  predictor arm holds, including P_bu at +0.707.
- Adding agreement hurts CSIQ prediction: Δ −0.507 [−0.703, −0.131].
- Gate: P_dis > τ rejects 51 of 60 cells, because the 372 lineages sit above the 944-derived τ.
  The composite gate is better calibrated here.

## 6. Deviations from the prereg

1. **Root selection.** The prereg said "resolved root + KADID check". Twenty-four cells had no
   bake-resolvable root, and 37 carried a cosmetic `regime: 720` on 944-input bakes. For every
   cell, the root was chosen as the first candidate in `eval_roots` order whose TRAIN-role KADID
   reproduces the stored value to 1e-6. CID22-A was scored only on that root. The choice reads
   no held-out label. A first run with the unmodified rule excluded 61 cells; its output was
   never analysed (`outcomes_try1_unread.json`).
2. **Binaries.** `ensemble_score_rows`, `panel` and `bake_verdict` were reused from the primary
   checkout build of 2026-09-19. A fresh build was queued behind the shared heavy lock. Validity
   rests on the exact C1 (448/448) and KADID (421/421) reproductions, not on provenance.
3. **D1 coverage.** C3–C6 are `not_measured` on most cells (C3/C4 on 351 of 450, C5/C6 on 434),
   so D1 = 1 means "no measured failure", not "passes". This follows the prereg definition, and
   it is a real limitation.
4. **Combo ranks** are fixed on the full population rather than recomputed inside each resample.

## 7. Caveats and fairness

- **Mixed eras in the composite inputs.** The stored imazen26 and nonphoto terms come from three
  file eras (n = 6,953, 7,869 and 10,025). They enter only the composite, and they are
  ssim2-labelled axes.
- **KonJND** is on the correct JPEG-504 ruler on all 359 MAIN cells.
- **Peers are not candidates here.** `peer_ssim2` has zero disagreement with itself by
  construction.
- **Every candidate is a zensim bake.** "Where peers win" does not apply to this lane.
- **Effective n is 19 lineages.** The CIs are wide, and a small real effect on KonJND or CSIQ
  cannot be excluded.
- **The contract and floor outcomes use the board's own ruler.** That is the ssim2-mentored
  `resolvable` rule.
