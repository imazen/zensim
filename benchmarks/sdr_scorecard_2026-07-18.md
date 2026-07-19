# SDR five-gate scorecard + ship recommendation (2026-07-18)

Part 1 of the three-part campaign (winner spline → scorecard → SDR decision). Process:
`docs/MODEL_SELECTION_SCORECARD.md`. Candidates: **winner_dial** (`Ebothg_hfgain_winsor_dial`
— the 156→128→1 `:both` MLP + surgical hf_gain winsor + the NEW `bake_dial_refit add-spline`
dial), **ADD156** (`ADD156_safesyn_only_raw_lasso`, genuinely-additive basic-156, 3.6 KB),
**B** (shipped, `b_sdr_linear_cid80_inclwinsor_dense_dial`).

**add-spline validation (the new tool's gate):** SROCC IDENTICAL on all 10 corpora pre/post
spline (monotone rank invariance, byte-level check); dial raw[−3.2,24.2] → **[14.6,95.3]**
(G1 pass), mono 0.980; steering unchanged through the spline (M2 1.000 / M3 0.759 on the
spot-check pair); zenjpeg then targets its dial for real (40→40.49 in 2 passes, 55→55.57 in 3).

## The scorecard

| gate | winner_dial | ADD156 | B shipped |
|---|---|---|---|
| **G-RANK** CID22 | **0.8939** | 0.8634 | 0.8764 |
| LIVE-R2 | 0.9600 | **0.9602** | 0.8970 |
| CSIQ | **0.9584** | 0.9024 | 0.9342 |
| PIPAL (GAN axis) | **0.6241** | 0.4940 | 0.5650 |
| imazen-26 non-photo | 0.8548 | **0.8628** | 0.8606 |
| AIC-3 | **0.8041** | 0.7773 | 0.7774 |
| **KonJND (PJND/HF axis)** | 0.3352 | 0.4462 | **0.5466** |
| **HF near-lossless** | 0.5872 | 0.4581 | **0.6142** |
| **G-DIAL** mono / p5 / p95 | 0.980 / 14.6 / 95.3 ✓ | 0.985 / 9.5 / 95.1 ✓ | 0.979 / 13.6 / 99.7 ✓ |
| **G-STEER** M3 (fold) | 0.759 (signed) | **0.849** (abs) | 0.660 ceiling (abs) |
| **G-RD** jxl photos (s2/bt/zs) | **+4.5/+0.9/+3.6%** | +3.3/**+2.3**/+4.1% | +4.3/**−0.7**/+0.9% (Trained map) |
| G-RD zenjpeg photos | +0.5/+0.2/+1.2% | +0.5/+0.2/+1.1% | ≡ baseline |
| G-RD zenjpeg screens | +0.7/+1.4/+0.0% | **+1.3/+1.4/+0.0%** | ≡ baseline |
| **G-TARGET** own-dial residual / passes | 1.75 / 4 ✓ | 1.72 / 4 ✓ | 0.85 / 4 ✓ |
| OOD max \|score\| | 100.0 | 100.0 | 96.2 |
| size | 83 KB | **3.6 KB** | 7.3 KB |

(jxl SCREEN G-RD is negative for ALL zensim drivers (−1…−21%, n=2 images) — a shared gap
addressed in Part 2, not a discriminator. B's jxl row used its shipped Trained map; its model
map fixes the butteraugli regression at an ssim2 cost — see `rd_probe_results_2026-07-18.md`.)

## Recommendation

**winner_dial for the SDR compression/dial slot** — it wins 5 of 8 held-out rank corpora
(including ALL THREE new FR compression benchmarks: LIVE-R2 0.960, CSIQ 0.958, PIPAL 0.624),
passes every dial gate, carries a near-best deployable steer map (0.759), delivered the best
jxl photo RD (+4.5% ssim2, positive on all judges), and its zenjpeg loop converges on its own
dial (1.75 ≤ 2 in ≤4 passes).

**The cost, stated loudly:** KonJND **0.335 vs B's 0.547** (−0.21) and HF near-lossless
**0.587 vs 0.614** (−0.027) — the PJND/near-lossless axis, the metric's standing weak zone
and precisely where B is strongest. This is a REAL trade, not noise. Per the trails
framework: winner_dial is the compression/dial-trail candidate; **B stays the
near-lossless-conservative alternative** (and the incumbent until the user calls the swap —
profile swaps remain user-gated per standing preference). ADD156 is the exact-diffmap /
tiny-footprint runner-up (best steer 0.849, best zenjpeg-screen row, 3.6 KB) — the fallback
wherever a fixed exact-gradient map or minimal size outranks +0.03 CID22.

**Not resolved by this scorecard:** the KonJND/HF gap (needs its own training lever — G5
history says naive weight tuning fails); the jxl screen-driving gap (Part 2); re-seeding both
codecs' legacy-V0_2 calibration tables against real-B/winner scoring.

Data: `/mnt/v/output/zensim/rd-target-eval-2026-07/` (incl. `winner_dial_aq` rows +
re-judged panel); sidecars next to each bake; probe tooling per the scorecard doc.
