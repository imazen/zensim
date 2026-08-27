# HDR freeze — predicted author verdicts from the zenpapers corpus (2026-08-28)

The user's directive: "reading zenpapers, predict what each author's verdict
would be based on the methodologies in their papers." Every row below is an
EXTRAPOLATION from documented methodology — labeled prediction, not quotation.
Grounding: `zenpapers/docs/iqa-methods/evaluation-statistics.md` (§2 mapping,
§3 Krasula, §5 cross-dataset) and `subjective-scaling-jod.md` (JOD/UPIQ),
with the primary works those docs trace.

Question under review: freeze `HDR944_L1T1_s4005_hfpack` (incumbent) vs
`HDR944R_t2_s4003_hfpack` (retrain), and the legitimacy of the swing-FIDELITY
amendment to the HDR-route gate panel.

## The panel

**1. Mohammadi · Sneyers · Saupe · Ascenso** (compression-metric evaluation;
the P.1401-mapped SROCC/PLCC/OR/PWRC/Z-RMSE panel + the HF/MF split of
arXiv:2509.13150). Their §2 doctrine: raw-scale comparisons are the
false-negative trap — "a metric can be perfectly rank-correct yet have a
compressive response curve"; each metric gets its own monotone map, fit once
globally. ⇒ **Predicted: the SDR panel's raw-swing g1 comparison would draw
their §2 criticism; per-candidate monotone re-anchoring is standard, so the
amendment is legitimate in form. Verdict: incumbent on the pooled panel —
WITH a demand for the HF/MF split reported beside it (where t2's HF-band
SROCC 0.716 vs 0.591 is a real advantage they would not let disappear).**

**2. Krasula (+ Le Callet; Pinheiro · Korshunov · Ebrahimi HDR evaluation).**
Their §3 doctrine: pairwise CI-grounded discrimination — C0 (different-vs-
similar AUC) explicitly "rewards a metric for NOT flagging differences that
humans cannot perceive." Over-swing is precisely flagging imperceptible
differences. ⇒ **Predicted: the fidelity amendment is their C0 logic in gate
form — endorsed. Verdict: incumbent; plus the sharpest criticism of our
evidence: the HDR val targets are METRIC-derived, so no C0/C1 with human CIs
exists on the HDR route at all — they would prescribe a small human pairwise
study before any HDR ship claim.**

**3. Mantiuk (+ Perez-Ortiz; UPIQ/JOD; HDR-VDP/cvvdp lineage).** JOD is a
fixed perceptual unit; a metric whose scale stretches ×1.9 against ground
truth is mis-scaled by definition; §5(ii): a ranking that flips across
datasets is overfit. ⇒ **Predicted: amendment endorsed (scale fidelity IS the
JOD philosophy); the t2 sihdr sign-flip reads as the §5(ii) overfit signal —
disqualifying. Verdict: incumbent; demand: bootstrap CIs on every headline
delta (their standard practice).**

**4. VQEG / ITU-T P.1401 (the standards corpus).** Map-then-compare is
mandatory; monotone maps preserve SROCC; comparing unmapped dynamic ranges
across differently-scaled candidates is invalid. ⇒ **Predicted: the
amendment is REQUIRED, not merely legitimate; the pre-amendment absolute
swing bar was the methodological error. Verdict: whatever the post-mapping
statistics say — the incumbent, per E.4.**

**5. Zhang · Katsenou · Bampis · Krasula · Li · Bull (VMAF line).**
Per-database reporting; cross-validation discipline; suspicious of any bar
adjusted after seeing results. ⇒ **Predicted: amendment plausible but they
would demand a HELD-OUT confirmation (the bar was amended post-hoc — confirm
the fidelity readings on data not used to set it). Verdict: incumbent,
provisional on that confirmation.**

**6. The dataset-construction school (Lin · Hosu · Saupe KADID/KADIS;
Ponomarenko TID).** Hygiene-first: splits, leakage, scale. ⇒ **Predicted:
applaud the family-aware split + census-clean verification; flag that t2's
target (era-B zensim) is a MODEL-derived target — metric-on-metric
circularity they habitually warn against; t1's cvvdp-mix is closer to
accepted metric-anchored practice. Verdict: incumbent (target-hygiene
grounds), independent of the swing debate.**

## Panel synthesis (prediction)
6/6 lean **incumbent**; the amendment is endorsed by 5 (required by one) with
one demand for held-out confirmation. The panel's recurring demands on US:
(i) report the HF/MF split beside any freeze [have: HF-band SROCCs];
(ii) bootstrap CIs on the headline deltas [run below];
(iii) held-out confirmation of the amended fidelity bar [run below];
(iv) the honest gap none of our instruments close: NO HUMAN pairwise CIs on
the HDR route — every HDR target is metric-derived. A small human pairwise
study (Krasula C0/C1 form, ~9 scenes × few conditions) is the registered
future lever if HDR claims need human grounding.

## The panel's demands — RUN (2026-08-28)

**(ii) Bootstrap CI (Mantiuk-row demand):** paired bootstrap (B=2000, shared
index sets, canonical panel_batch_indexed) on the SHARED t1-val HF band
(n=3,036): **HF-band SROCC delta (t2 − incumbent) = −0.0093, 95% CI
[−0.0176, −0.0013] — significant, and it REVERSES the earlier table**: the
"t2 discrimination lead (0.716 vs 0.591)" compared each candidate on its OWN
band with different targets and n (3,036 vs 238) — a cross-band artifact this
exercise caught. On a shared instrument the incumbent leads HF discrimination
too.

**(iii) Held-out confirmation (VMAF-row demand):** first attempt INVALID and
recorded as a lesson — raw swing-ratios against the t2-val leg's era-B
targets (span −147…95; svt target swing 247) mixed scales, the exact
map-before-compare error P.1401 §2 warns about, now caught in our own check.
The valid same-scale form — scene-split stability on t1-val (half-A/half-B):
**incumbent in-band on all 3 codecs in BOTH halves (0.91–1.36); t2
out-of-band on jpeg-gainmap (1.69 / 2.38) and zenjxl (1.46 / 1.46) in BOTH
halves.** The amendment's readings are scene-stable.

**(iv) stands as the honest gap:** no human pairwise CIs exist on the HDR
route (all targets metric-derived); the Krasula-form small human study is the
registered future lever for any human-grounded HDR claim.

## Bottom line
The predicted panel is 6/6 incumbent; our own follow-through on its demands
strengthened that verdict (the one apparent t2 advantage was a cross-band
artifact; the fidelity readings are scene-stable; the sole invalid check was
ours, caught and recorded).
