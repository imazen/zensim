# Screen-mass retrain — probe-driven tuning iteration (2026-07-18)

Part 2 of the tune-and-pick campaign: the RD probe's screen column (every zensim jxl driver
NEGATIVE on screens) used as a *measurable training objective*. Recipe = the Ebothg winner
EXACTLY (groups/seed/surgical hf_gain winsor — bounds extracted from the bake's own
metadata) + ONE new group: `bigcodec` (imazen-26 diverse real-codec, train-origin digits,
2,946,036 rows, `:both` loss, [0,1] target verified) at weight w. Trainer:
`zensim_mlp_train`, seed 13; dial via `bake_dial_refit add-spline` (same anchor as P1);
runner: `screen_retrain_2026-07-18.sh` (recorded in the P1 memory).

## w = 0.5 — rank panel vs the winner (both dialed, full 10-corpus sidecars)

| corpus | winner | scr0.5 | Δ |
|---|--|--|--|
| CID22 | 0.8939 | 0.8793 | −0.0146 |
| **imazen-26 real-codec** | 0.8329 | **0.8926** | **+0.0597** |
| **imazen-26 non-photo** | 0.8548 | **0.9062** | **+0.0514** |
| **HF near-lossless** | 0.5872 | **0.7116** | **+0.1244** |
| KonJND | 0.3352 | 0.2707 | −0.0645 |
| AIC-3 / AIC-4 / CSIQ / PIPAL / LIVE-R2 | — | — | all within ±0.007 |
| dial mono / p5 / p95 | 0.980 / 14.6 / 95.3 | **0.985** / — / 95.4 | + |

**The headline is HF near-lossless 0.712** — it beats B's 0.614 (the previous
best-of-any-architecture) by +0.098 and closes the axis P1 flagged as the winner's biggest
weakness. The imazen-26 lifts (+0.05–0.06) are the intended screen/diverse effect. Costs are
real: CID22 −0.015 (0.879 — still above B's 0.876) and KonJND −0.065 (0.271 — the PJND axis
worsens; it is anti-correlated with the bigcodec mass in this recipe family).

## w = 0.5 — the deciding G-RD screen column (jxl, bytes at equal judged score)

| driver | class | ssim2 | butteraugli | zensim |
|---|---|--|--|--|
| winner | photo | +4.5% | +0.9% | +3.6% |
| **scr0.5** | photo | +3.3% | +1.0% | **+4.3%** |
| winner | screen | −11.2% | −20.8% | −4.8% |
| **scr0.5** | screen | **−5.9%** | −19.7% | **−2.2%** |

**Verdict: the model retrain closes roughly HALF the model-side screen gap** (ssim2-judged
−11.2 → −5.9; zensim-judged −4.8 → −2.2) while holding photos — but the butteraugli-judged
screen loss (~−20%) is unmoved by the model. Attribution: the remaining screen deficit is
substantially the LOOP side — jxl's photo-seeded `zensim_targets` distance table + loop
behavior on screen content — which no scalar retrain can fix. (n=2 screen images: directional,
per the probe-scale caveat.)

## What this iteration validates

1. **The probe works as a tuning objective**: one recipe change, aimed at one probe column,
   moved that column in the predicted direction while the anti-gaming judges + rank panel
   guarded the rest. This is the scorecard's G-RD/G-TARGET used as an optimizer, as designed.
2. **The screen gap decomposes**: ~half model (fixed by data mass), remainder loop-side
   (needs the distance-table re-seed + possibly per-class tables — already queued as the
   calibration follow-up in `rd_probe_results_2026-07-18.md`).
3. **A new Pareto point exists**: scr0.5 = {CID22 0.879, nonphoto 0.906, HF-NL 0.712, dial
   0.985} — for products weighting diverse/near-lossless content over the last 0.015 CID22
   and KonJND, it is arguably the strongest bake in the program. KonJND remains this family's
   structural weakness (G5 history).

## w = 1.0 dose-response — saturation; w = 0.5 is the operating point

| corpus | winner | scr0.5 | scr1.0 |
|---|--|--|--|
| CID22 | 0.8939 | 0.8793 | 0.8761 |
| imazen-26 real-codec | 0.8329 | 0.8926 | 0.9004 |
| imazen-26 non-photo | 0.8548 | 0.9062 | 0.9144 |
| HF near-lossless | 0.5872 | 0.7116 | 0.7161 |
| KonJND | 0.3352 | 0.2707 | 0.2359 |
| **LIVE-R2** | 0.9600 | 0.9593 | **0.9343** |
| CSIQ / PIPAL / AIC-3 | ≈flat | ≈flat | ≈flat |
| dial mono | 0.980 | 0.985 | 0.980 |

Doubling the mass buys +0.008 nonphoto / +0.005 HF-NL (diminishing) while the costs
accelerate: KonJND −0.035 further and **LIVE-R2 −0.025** (the classic FR compression holdout
starts paying). **Verdict: w = 0.5 is this iteration's output** — `Ebothg_scr0.5_dial.bin`
({CID22 0.879, nonphoto 0.906, HF-NL 0.712, LIVE 0.959, dial 0.985}). The remaining screen-RD
deficit is loop-side (distance-table re-seed), and KonJND remains the family's structural
weakness — both explicitly out of scope for data-mass tuning.
