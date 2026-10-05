# E27 — an HDR teacher leg that keeps HDR scores comparable across images (registered 2026-10-05 18:00 UTC, before any E27 fit)

E26 (`benchmarks/e26_hdr_teacher_registration_2026-10-05.md`; result `~/tmp/zensim-paper/rev4/E26_DONE.md`) adopted hd4 by its
registered rule, but that rule only gated **within-reference** HDR SROCC, which the control already scores at 0.998 (a saturated
measure). The report-only **pooled** HDR-VDP-3 SROCC fell by 0.045 ± 0.010 (hd4) and 0.125 ± 0.014 (hd16): the within-reference rank
leg reorders images against each other. For a one-target-score product, cross-image comparability matters, so E26's hd4 is not a
shipping candidate. E27 tests two leg forms that constrain cross-image levels.

## Arms (everything not named here is exactly E26's)

Same teacher (HDR-VDP-3 `q_jod`, HDRTEACH labels), same TRAIN population (7,390 `agree = true` rows), same fixed target
transform (score = 10 × q_jod), same Rev5 native HDR features for by_v2fy's 420 IDs, base recipe
`sel:59f0bbc2f290@h32:H128:cv16:cf98` head N at Rev5, seeds 0–9 × the five LODO folds, weight 4 in E26's weighting convention.

* `hp4` — the HDR leg uses a **pooled** rank loss: pairs are drawn across references within the HDR TRAIN leg (not only within one
  reference), target order from 10 × q_jod.
* `ha4` — the HDR leg uses the existing teacher-leg form: **within-reference MSE plus rank** on 10 × q_jod (absolute level and order).

Control: the E24 Rev5 by_v2fy cells (same seeds/folds), seed-paired, with E26's revision binding (dense + stamp). E26 hd4 is reported
alongside, not ruled on.

## Decision rule (fixed now)

On the registered hdr_v3mix VAL (3,900 pairs, 300 references; VAL role; never trained, calibrated or used for selection), seed-paired
over the 50 cells per arm:

1. SDR not worse: E21's as-good rule (signed mean ≥ −0.002, every source ≥ −0.005, W2 > −2 SE).
2. HDR not worse anywhere: pooled and within-reference SROCC against both HDR-VDP-3 and historic CVVDP VAL labels each ≥ −2 SE.
3. HDR better: pooled SROCC against HDR-VDP-3 improves by more than 2 SE.
4. Adopt the arm that passes 1–3 with the larger pooled HDR-VDP-3 gain; if neither passes, by_v2fy gets no HDR leg from this line.

Reported, not ruled on: external SDR sets, E26 hd4 on the same panels, pooled-vs-within scatter geometry. Any change after a fit
starts is a new registration.
