# E29 — an HDR leg that learns the two teachers' consensus cross-image order (registered 2026-10-07 09:40 UTC, before any E29 fit)

Controlling registration: [the four-source amendment](e29_four_source_amendment_2026-10-07.md)
and [shared fresh control decision](e29_e31_e32_shared_control_decision_2026-10-07.md)
supersede this original record’s population, control, arm-retention and inference
clauses. The original text below is retained as registration history; full fits
use four folds, both hb4/hc4, the fresh matched V40 control and both-teacher gates.

E27 (`benchmarks/e27_hdr_pooled_registration_2026-10-05.md`; result in `benchmarks/rev5_research_status_2026-10-05.md` §8) showed the
two HDR teachers agree within references (0.998) but only 0.828 pooled: a pooled leg on HDR-VDP-3 alone (hp4) lifted pooled HDR-VDP-3
SROCC 0.846 → 0.969 while pooled CVVDP fell 0.930 → 0.841. E29 trains the cross-image order only where the teachers do not disagree.

## Arms (everything not named here is exactly E27's hp4)

Same TRAIN population (the 7,390 HDR TRAIN `agree = true` rows), same Rev5 native HDR features for by_v2fy's 420 IDs, base recipe
`sel:59f0bbc2f290@h32:H128:cv16:cf98` head N at Rev5, seeds 0–9 × five LODO folds, nominal HDR weight 4 (E26 weighting), 120 epochs,
final epoch 119. Teacher scores per TRAIN row: HDR-VDP-3 `q_jod` (HDRTEACH) and the fresh corrected-TRAIN CVVDP JOD (HDRCORR).

* `hb4` — **Borda consensus target.** Each teacher's TRAIN scores are converted to pooled mid-ranks in [0, 1] over the 7,390 rows; the
  target is their mean; the HDR leg uses E27's pooled rank loss on that target (pairs across references within the HDR leg).
* `hc4` — **agreement-filtered pairs.** Pooled rank pairs across references, kept only when both teachers order the pair the same way
  with |Δq_jod| ≥ 0.05 and |ΔCVVDP JOD| ≥ 0.05; target order = the agreed order. Requires a pair-list group in the existing trainer;
  if the implementation finds that infeasible without changing other paths, `hc4` is dropped and recorded BEFORE any fit (only `hb4` runs).

Control: the E24 Rev5 by_v2fy cells (seed-paired). E27 hp4 is reported alongside, not ruled on.

## Decision rule (fixed now; identical to E27's so the results compare)

On the registered hdr_v3mix VAL (3,900 pairs, 300 references; VAL role; never trained, calibrated or used for selection), seed-paired over
the 50 cells per arm:

1. SDR as good (E21: signed mean ≥ −0.002, every source ≥ −0.005, W2 > −2 SE).
2. Pooled and within-reference SROCC against both HDR-VDP-3 and historic CVVDP VAL labels each ≥ −2 SE.
3. Pooled SROCC against HDR-VDP-3 improves by more than 2 SE.
4. Adopt the passing arm with the larger pooled HDR-VDP-3 gain; exact ties adopt none; if none pass, no HDR leg from this line.

Reported, not ruled on: pooled gain against CVVDP, pooled agreement with the Borda consensus on VAL, external SDR sets. Execution: the
existing pipeline (pinned packs, `jobset_caps.json` memory envelope, no `tail_trim`), launched after E28 completes on the fleet.
