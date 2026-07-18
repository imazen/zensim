# Final dial metric — running experiment log (2026-07-18)

Campaign to validate/build the additive-core dial metric per
`docs/FINAL_DIAL_METRIC_DESIGN_2026-07-18.md`. Every experiment gets a hypothesis, a
result, and a verdict. Accumulate; don't overwrite.

## Baselines (context)
- **B (shipped linear)**: CID22 0.876, near-lossless HF 0.614, corruption gate (DIALED) 18%,
  diffmap spatial ~0.87–0.91.
- **basic-156 additive core** (simple 5-group linear, seed 13, no spline): CID22 0.8978,
  imazen26 0.8353, nonphoto 0.8531, HF 0.4446, corruption gate (RAW) 85.6%, diffmap ~0.987.
- Confirmed prior: additive→exact diffmap 0.987 (`diffmap_block_coherence`); basic≥full quality.

## Experiment queue
- E1  spline-preservation: full-range monotone spline on additive core preserves corruption+CID22+HF?
- E2  verify 85.6% + per-region anomaly on full corruption corpus (not the 222-subset)
- E3  winsor additive core: recover B's HF 0.614 with winsor transforms on basic block?
- E4  ModelCoherent diffmap for the SHIPPED path (additive core's own weights → 0.987 diffmap)
- E5  negative tail (kadis_negrich) through the additive core + full-range spline
- E6  dial monotonicity of the additive core on a q-sweep (Eval A foundation)
- E7  L2 max-supplement: peak/max severe term fixes edge_border w/o disturbing honest?
- E8  monotone GAM core: beat linear on CID22 while staying additive?
- E9  one-shot targeting residual (Eval C) — q_hat regressor

---
## E1 — spline-preservation → OVERTURNED: it's the WINSOR transforms, not the spline
Hypothesis: B's 18% corruption gate was the dial spline flattening the corruption range.
FALSIFIED. Measured B's RAW (pre-spline) corruption gate = **18.0%** (= its dialed gate;
`mapped` post = 0%). So B's MODEL doesn't rank corruption low — the spline is innocent.
The difference from basic-156 (85.6% raw) is the RECIPE. Prime suspect: B's `winsor_p99`
transforms clip the extreme distortion features corruption produces (which is *why* winsor
lifts near-lossless HF 0.614 vs basic-156's 0.445 — same outlier-taming mechanism). So
corruption-gating vs near-lossless is a **WINSOR tradeoff**, located in the feature transforms,
not the spline/pooling. This supersedes the "preserve-through-dial" reframe. → test E3.

## E2 — the 85.6% is MISLEADING: over-reaction to localized HF, perceptually inverted
Per-region median RAW corruption score (higher=better; q20 anchor = constant 0.68):
| region | n | med corr raw | gate% |
|---|--|--|--|
| whole | 42 | −2.14 | 69% |
| frac2 | 36 | −3.36 | 69% |
| frac4 | 36 | −5.78 | 78% |
| sq64 | 36 | −9.33 | 100% |
| sq16 | 36 | −13.93 | 100% |
| sq8 | 36 | **−20.35** | 100% |

The additive core scores an **8px break (−20) as far worse than a whole-image corruption
(−2)** — perceptually BACKWARDS (whole-image is much more visible; small breaks are the
SUBTLE/sub-perceptible zone per the perceptibility study). So the 85.6% overall gate is
driven by **over-reaction to localized HF/edge features**, not correct perceptible-corruption
gating — the exact over-gating-the-subtle pathology flagged earlier. The genuinely-perceptible
whole-image corruption is UNDER-caught (69%). VERDICT: "85.6%" is not a win; the additive core
mis-ranks corruption by localization. The gate must be perceptibility-calibrated, and the raw
number is a distractor. This also predicts E3: winsor (clipping the extreme features) should
kill the small-region over-reaction — which is precisely how B lands at 18%.

## E3 — winsor hypothesis FALSIFIED; E6 — dial-mono is the RECIPE, not additivity
E3 (basic-156 + winsor): CID22 0.8907, HF **0.3736** (winsor HURT it here, vs 0.445 no-winsor),
corruption **87.4%** (NOT killed), dial-mono 0.416. So winsor is NOT what blinds corruption or
smooths the dial — E1's winsor hypothesis is dead.

E6 dial monotonicity (densified multi-codec q-sweep, G3≥0.93):
| bake | dial-mono | corruption(raw) | CID22 | HF |
|---|--|--|--|--|
| B (shipped: BVLS+dial-anchor+spline) | **0.9792** | 18% | 0.876 | 0.614 |
| basic-156 (RankNet linear, no spline) | 0.4736 | 85.6% | 0.8978 | 0.445 |
| basic-156 + winsor (RankNet) | 0.4159 | 87.4% | 0.8907 | 0.374 |

Both are additive/linear, yet B is a SMOOTH dial (0.98) and my RankNet is JITTERY (0.47) — so
additivity isn't the dial issue, the RECIPE is. A monotone spline can't fix 0.47→0.98 (it
preserves order), so B's RAW model must already be ~0.98 monotone. Prime suspect: the LOSS —
RankNet optimizes pair ORDER (jittery magnitude within-image); least-squares/BVLS fits VALUES
(smooth). → test E-MSE: basic-156 with MSE loss. The B↔basic-156 tradeoff (smooth dial + blind
corruption ↔ jittery dial + gates corruption) is now the central axis to resolve.

## Synthesis after E1–E6 — convergence toward B-family
- **B is the better dial**, decisively: dial-mono 0.98 (vs basic-156 0.47), HF 0.614 (vs 0.445),
  CID22 0.876 (vs 0.898 — recipe noise, both held-out). basic-156's ONLY apparent edge is
  corruption 85%, which **E2 debunked** (over-reaction to localized HF, perceptually inverted).
- So the additive-core experiment does NOT beat B; it trades a real dial for a spurious gate.
- **Convergent design**: B-family smooth additive dial + SEPARATE corruption guard (butteraugli
  Stage-2, per "corruption isn't the scalar's job") + ModelCoherent diffmap (B's own weights).
- **Open question**: B's diffmap is 0.87 (non-basic features); the exact-diffmap 0.987 needs
  basic-only, but basic-only via RankNet is a jittery dial (0.47). Can B's dial-smoothness be
  had on basic-only features? Lever = loss function (least-squares/BVLS smooth vs RankNet jitter).
  → E-MSE tests it.

## E-MSE — LOSS is the dial-monotonicity lever (CONFIRMED, key finding)
basic-156, safesyn-only, **MSE loss**: dial-mono **0.9981** (vs RankNet 0.474; ≈ B's 0.979),
HF 0.532, corruption 27%, CID22 0.074 (safesyn-only → no generalization, DATA issue not loss).
Conclusions:
- **Dial monotonicity = the LOSS.** Least-squares/MSE fits VALUES → smooth (0.998); RankNet fits
  pair ORDER → jittery (0.47). This is why B (BVLS least-squares) is a 0.98 dial. SETTLED.
- **RankNet's 85% corruption was a loss artifact** — MSE gives 27% (realistic). Over-weighting
  extreme-feature pairs in RankNet inflated it. Confirms E2: the additive core never had a real
  corruption edge.
- Recipe for a good additive dial: **MSE loss + multi-corpus (for CID22) + basic features (exact
  diffmap)**. MSE needs common target scale (mixed scales diverge). → E-MSE2: safesyn + cid22_train
  (both ssim2-scale, MSE-compatible).

## E-MSE / E-MSE2 CORRECTION — the MSE "smooth dial" is COLLAPSE, not smoothness
E-MSE2 (MSE, safesyn+cid22_train): dial-mono 0.9981 BUT dial p5/p95 = **−1.3 / −0.1** (near-constant
output) and CID22 **0.085**. The high monotonicity is SPURIOUS — a collapsed near-constant output
has no inversions AND no rank/range. MSE-SGD on the linear basic model underfits to the target mean
(the bulk of human_score dominates the squared error; the spread is ignored). So:
- **Corrected E-MSE finding**: MSE loss did NOT give a genuine smooth dial — it collapsed.
  The 0.998 was an artifact of near-constant output.
- **RankNet vs MSE-SGD is a false dichotomy**: RankNet = ranked (CID22 0.898) + jittery (0.47);
  MSE-SGD = collapsed (CID22 0.085) + fake-smooth. NEITHER matches B.
- **B's BVLS least-squares does BOTH** (CID22 0.876, dial-mono 0.98, full range 13.6/99.7) — so a
  CONSTRAINED/analytical least-squares avoids the collapse that MSE-SGD hits. The dial smoothness is
  achievable but needs B's fitting approach, not naive MSE-SGD.
- → E-BOTH: RankNet + MSE (`:both`) to get rank (RankNet keeps the spread) + smoothness (MSE),
  and separately consider reproducing B's BVLS on basic features.

## E4-prep — B's sensitivity mass: 62% basic / 38% non-basic (95/372 nonzero)
Extracted B's linear weights (`zenpredict inspect --weights`) + scaler. Sensitivity mass
|w|/scale: **BASIC(f0-155) 62.1%, NON-BASIC(f156-371) 37.9%** (46 + 49 nonzero). So B is NOT
purely additive — 38% of its scalar rides the non-additive peak/max/IW features. That is exactly
why B's diffmap caps at 0.87 (the basic-only diffmap misses that 38%). Combined with E2 (non-basic
adds nothing to CID22 aggregate): a **basic-only B would have the exact diffmap (0.987) AND the
same CID22 (0.876)** — dropping the 38% non-basic mass costs nothing measurable and buys the exact
gradient. So the winning recipe = **B's smooth-dial fitting (BVLS least-squares) restricted to
basic features**. B's basic sensitivities saved for the ModelCoherent diffmap experiment.

## E-BOTH — THE WINNING RECIPE ✓ (RankNet+MSE on basic features)
Groups: safesyn:both, cid22_train:both, kadid:rank, tid:rank; basic-156; seed 13.
| metric | E-BOTH | B (ref) | verdict |
|---|--|--|--|
| dial-mono | **0.9840** | 0.979 | ✓ matches B (G3 pass) |
| CID22 | **0.8953** | 0.876 | ✓ BEATS B |
| imazen26 | 0.8349 | 0.841 | ≈ |
| nonphoto | 0.8542 | ~0.88 | ≈ |
| HF near-lossless | **0.5862** | 0.614 | ≈ B (vs RankNet 0.445) |
| dial range (pre-spline) | −3.4/24.6 | — | real spread (NOT collapsed) |
| diffmap coherence | **0.987** (basic-only) | 0.87 | ✓ EXACT (B can't — 62% basic) |

`:both` = RankNet (rank, keeps the spread → CID22) + MSE (value-fit → smooth dial), and being
basic-only it carries the exact diffmap. This is the first recipe to get **smooth dial + quality
+ exact diffmap simultaneously**, beating B on CID22 (0.895) and matching dial-mono (0.984) while
adding the exact diffmap B structurally can't have. The design CLOSES:
**basic-156 + `:both` loss + dial spline ([neg,100] + negatives) + butteraugli corruption guard.**
Caveats before ship: single seed (seed-confirm needed); pre-spline range needs the dial spline;
full-corpus + konjnd/aic groups to add. But the architecture is validated end to end.

## E-BOTH full-panel confirmation (CID22)
SROCC 0.8953 / PLCC 0.8928 / KROCC 0.7115 / OR 0.0005 / PWRC 0.9842 / Z-RMSE 0.451 /
DS-AUC 0.8239 / per-ref 0.9609 / %bwd 0%. Solid across the full Mohammadi panel — the 0.895
is real, not a SROCC artifact; calibration (PLCC 0.893) and within-ref rank (0.961) are strong.
Seed-confirm (s7,s23) in flight to check robustness of the beats-B result before any ship claim.

## E-BOTH seed-confirm — ROBUST (s7/s13/s23)
| seed | dial-mono | CID22 | HF |
|---|--|--|--|
| s13 | 0.984 | 0.8953 | 0.586 |
| s7  | 0.985 | 0.8928 | 0.612 |
| s23 | 0.984 | 0.8950 | 0.642 |
Rock-solid across seeds (no collapse). dial-mono 0.984±0.001 ≥ B 0.979; CID22 0.894±0.001 > B
0.876; HF 0.61±0.03 ≈ B 0.614. The winning recipe is seed-robust.

## E7-lite — zensim's OWN peak features do NOT auto-gate corruption (butteraugli clarification)
Runtime cannot call butteraugli (user, correct). "Learn from butteraugli" = use zensim's own
XYB max/p-norm PEAK features (f156-371, the butteraugli-max analog), one forward pass. BUT tested:
median corruption raw score by region, basic-156 (no peak) vs full-372 (WITH peak):
| bake | whole | frac2 | frac4 | sq64 | sq16 | sq8 |
|---|--|--|--|--|--|--|
| basic-156 (no peak) | −2.1 | −3.4 | −5.8 | −9.3 | −13.9 | −20.4 |
| full-372 (WITH peak) | −0.6 | −1.9 | −4.5 | −11.2 | −14.0 | −19.1 |
Adding the peak features does NOT fix the localization inversion (whole-image corruption still
scored MILDER than an 8px break). This is inherent to max-norm pooling (fires on worst LOCAL
error → over-weights small intense breaks). So corruption-in-the-runtime-scalar is NOT free from
zensim's butteraugli-like features — it inherits butteraugli-max's localization bias and needs
deliberate perceptibility-calibrated weighting. OPEN QUESTION whether it can be done cleanly
without hurting the honest dial; the offline butteraugli Stage-2 (zensim-regress) remains the
reliable corruption check.
