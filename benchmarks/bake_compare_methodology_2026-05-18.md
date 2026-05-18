# bake_compare — smoke test result + methodology, 2026-05-18

## What this is

This doc captures the first end-to-end smoke run of the new
`bake_compare` binary at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/bake_compare.rs`.
The binary implements § A.9 of `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md`
verbatim and is now the canonical "A vs B" ship-decision gate for
every subsequent V_X experiment.

Per the V_X principled workflow (Step 2 — decide reporting panel
upfront), every new ship-candidate methodology doc going forward
MUST include a `bake_compare A vs PRIOR_SHIP` table — see
`v0_18_methodology_2026-05-13.md` template, "Ship Comparison vs
Prior" section (added 2026-05-18).

## Inputs

| Bake | Path |
|---|---|
| **A** | `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin` |
| **B** | `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_s3_h128_packed.bin` |

- A: V_22-mix-LARGE+iwssim (the new ship candidate)
- B: prior V_22-mix konjnd@0.02 ship (the current reference)

Both are h=128, seed=3, packed (zerobias-quantized). They share the
same training corpus mix; A adds the IW-SSIM auxiliary supervision
described in `benchmarks/iw_pyramid_spike_methodology_2026-05-15.md`.

## Command

```sh
./target/release/bake_compare \
    --a /mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin \
    --b /mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_s3_h128_packed.bin \
    --output benchmarks/bake_compare_smoke_2026-05-18.md \
    --json benchmarks/bake_compare_smoke_2026-05-18.json
```

Wall time: **113 s** (5 corpora × 2 bakes × 1000 bootstrap resamples).

## Headline result

| Cells | Count |
|---|---:|
| `ADecisivelyBeatsB` | **16** |
| `BDecisivelyBeatsA` | 0 |
| `PromisingNotDecisive` | 4 |
| `Tied` | 9 |
| `Noisy` (n < 30) | 1 |

**Overall winner: `A` — 16 decisive A wins vs 0 B wins.**

### Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Verdict |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8324 | 0.8292 | 0.559 | 0.579 | 0.9006 | 0.8927 | +2.76 | +28.3 | +1.382 | promising |
| KADIK10k | 10125 | 0.9677 | 0.9059 | 0.249 | 0.416 | 0.9804 | 0.9440 | +95.4 | +633.9 | +79.46 | **A>>B** |
| TID2013 | 3000 | 0.9729 | 0.8860 | 0.236 | 0.448 | 0.9832 | 0.9146 | +54.5 | +310.4 | +54.46 | **A>>B** |
| KonJND-1k | 1008 | 0.8927 | 0.8931 | 0.376 | 0.219 | 0.9178 | 0.9204 | -0.2 | -193.9 | -0.00 | promising |
| AIC-3 CTC | 600 | 0.7845 | 0.7907 | 0.606 | 0.598 | 0.8630 | 0.8592 | -2.78 | -6.3 | +0.00 | tied |

A decisively beats B on KADID and TID at aggregate. CID22 trends
A's way but agree_A=3 (needed ≥4) so it lands as "promising not
decisive" — that's exactly the case the rule was designed to catch
(SROCC + Z-RMSE + PWRC all in A's favor but bootstrap CIs of OR /
KROCC / PLCC didn't all clear zero).

### Per-band decisive verdict highlights

**CID22**:
- B3..B6 (low-mid quality, n=57..836): A wins on h_SROCC every
  band except B3 (B3 ties; h marginal). B5 and B6 land "promising"
  (h_SROCC > 6, but Z-RMSE didn't help).
- B7..B8 (mid-high quality, n=1092 / 1382): **B wins** h_SROCC by
  -5.8 / -2.8 — this is exactly the band where the prior ship
  (B) is calibrated. A's IW-SSIM auxiliary loss pulls the
  predictions away from the ssim2-shape that CID22 MOS rewards in
  this band. Doesn't clear the 4-condition rule either way, lands
  as "tied" since bootstrap CIs cross zero.
- B9 (n=43): tied.

**KADIK10k**:
- **A decisively beats B on B0..B8 (9 bands out of 10).**
- B9 ties (h_SROCC +3.5 but Z-RMSE +1.4 < 1.96 → fails decisive).
- DecScore peaks at +18.2 in B8 (n=1699).

**TID2013**:
- **A decisively beats B on B2..B6 (5 bands).**
- B0 noisy (n=29 < 30 by 1). B1 ties (n=34, marginal stats).
- B7 ties (n=67, h_SROCC = -0.69, no signal either way).

**KonJND-1k**: promising at aggregate, no per-band (corpus uses
raw mean_threshold scale, not 0..1 normalized MOS).

**AIC-3 CTC**: ties at aggregate. h_SROCC = -2.78 (favors B), but
h_Z-RMSE = -6.3 reversed (favors A actually), and the bootstrap
CIs all cross zero. This is exactly the case § A.9 flags:
"single-stat win, no decisive."

## DecisiveScore aggregate

| Corpus | Aggregate DecScore |
|---|---:|
| CID22 | +1.382 |
| KADIK10k | **+79.464** |
| TID2013 | **+54.463** |
| KonJND-1k | -0.000 |
| AIC-3 CTC | +0.000 |

Per § A.9 line 168: practical cutoff is `|DecisiveScore| > 7.84`
(= 1.96 × 4 × 1). KADID and TID both clear it by an order of
magnitude in A's favor. CID22 is below threshold.

## Edge cases that the run revealed

1. **Polarity alignment is load-bearing.** First run shipped with
   the MRR formula taking raw signed SROCC against MOS. Both
   bakes are score-shaped (`spearman(scores, humans) > 0`) so
   they didn't differ in sign, but the original
   `polarity_factor`-only path produced negative `h_SROCC` even
   when A was the higher-SROCC bake. Root cause: `atanh(r_A) -
   atanh(r_B)` is direction-correct only when both `r > 0`; my
   first version flipped only B against A, leaving A's
   orientation as whatever the trainer happened to use. Fixed by
   independently sign-flipping each bake to score-orientation
   before computing `r_AZ`, `r_BZ`, `r_AB`.

2. **`r_AB = 1` (perfect agreement) divides by zero.** § A.9
   doesn't say what to do — we clamp `r` strictly inside
   `[-0.9999, 0.9999]` before atanh and before
   `(1 - r) / denominator`. Any bake pair correlated to 1e-4 of
   perfect has effectively zero stat power anyway.

3. **`r_AB` for MRR-on-Z-RMSE.** § A.9 says "replace SROCC_* with
   `1 - Z-RMSE_* / σ_max`" but is silent on what `r_AB` should
   be in that universe. Z-RMSE isn't pair-rankable, so we reuse
   the SROCC-based `r_AB`. This is the same approximation as the
   Mohammadi 2025 worked example (the two metrics' agreement is
   on rankings, not absolute calibration).

4. **AIC-3 + KonJND skip per-band.** Both use non-`[0,1]`-normalized
   target scales (JND step grid for AIC-3, raw mean_threshold for
   KonJND). Aggregate-only is the load-bearing read on those, same
   convention as `bake_verdict`.

5. **DecisiveScore goes to ±0.000 when sign factors are 0.** Added
   a `signum_or_zero` helper so the product isn't accidentally
   inverted by 0.0_f64.signum() returning a NaN-clean ±0 — the
   sign factors collapse the score to 0 when calibration or PWRC
   tie exactly, which is the right behavior (DecScore is a "how
   confident" scalar, not "which side").

## What lands in the methodology doc template

The template `benchmarks/v0_18_methodology_2026-05-13.md`
gains a new mandatory section (added 2026-05-18):

```markdown
## Ship comparison vs prior (bake_compare § A.9)

`bake_compare --a <this_bake> --b <prior_ship>` must be run and
its summary table pasted here. The decisive-band tally is the
ship-go/no-go signal. If A loses any decisive band on CID22 or
AIC-3 that the prior ship won, flag it explicitly.

[paste output of bake_compare's "Cross-corpus aggregate summary"
and "Decisive-band totals" sections]
```

This is a HARD requirement starting with the next ship candidate
after V_22-mix-LARGE+iwssim.

## Reproducibility

- Bootstrap seed: 42 (default)
- Resamples: 1000 (per § A.9 step 4)
- Bands: 10-band B0..B9 width-10 grid
- Features parquet root:
  `/mnt/v/zen/zensim-training/2026-05-15-full-features/`

Re-run is deterministic for a given (seed, resamples, panel
input). The xoshiro256** RNG is seeded per-resample, so
resample k always draws the same row sample regardless of how
rayon schedules across cores.

## JSON output schema (for site / dashboards)

`bake_compare --json results.json` writes a structured form:

```jsonc
{
  "a_bake": "...path...",
  "a_label": "...",
  "b_bake": "...",
  "b_label": "...",
  "bands_mode": 10,
  "bootstrap_resamples": 1000,
  "seed": 42,
  "corpora": [
    {
      "name": "cid22",
      "display": "CID22",
      "n_total": 4292,
      "enable_per_band": true,
      "aggregate": {
        "n_band": 4292,
        "panel_a": { "n": 4292, "srocc": 0.8324, "plcc": ..., "pwrc": ..., "z_rmse": ... },
        "panel_b": { ... },
        "r_ab": 0.9236,
        "h_srocc": 2.764, "p_srocc": 0.0057,
        "h_z_rmse": 28.327, "p_z_rmse": 0.0,
        "pwrc_diff": 0.0079,
        "ci_delta": [[lo, hi], ...x6],
        "agreement_a": 3, "agreement_b": 0,
        "decisive_score": 1.382,
        "decision": "PromisingNotDecisive"
      },
      "per_band": [{ "label": "B3", "range": "[0.30, 0.40)", "outcome": {...} }, ...]
    }, ...
  ],
  "aggregate_counts": { "a_decisively_beats_b": 16, ... },
  "overall_winner": "A"
}
```

Suitable for the interactive comparison site at
<https://imazen.github.io/zensim/> (when that ships).
