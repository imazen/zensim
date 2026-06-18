# zensim-b-phone (modern_oled_phone_indoor) — methodology, 2026-05-26

A zensim metric that emulates **CVVDP at a realistic modern-phone
display** (everyday indoor SDR viewing), with a working user-facing
dial. Supersedes the prior interim phone slot (a desktop-bake copy)
and the first phone-CVVDP attempt that shipped a broken dial.

## Hypothesis / falsification

- **Hypothesis**: A zensim bake trained on phone-condition CVVDP
  scores (KADID + TID scored under `modern_oled_phone_indoor`) with
  the V39 rank-then-spline recipe, where the spline anchor is built
  FROM the phone-CVVDP training distribution itself, gives (a) a
  working dial (G1 p5≤25 ∧ p95≥85) and (b) held-out phone-CVVDP
  tracking SROCC ≥ 0.85.
- **Falsification**: if the dial stays broken (G1=0) or held-out
  phone-CVVDP SROCC < 0.85, the approach is wrong.
- **Result**: PASS. Held-out phone-CVVDP pooled SROCC = **0.9342**,
  G1 = **1.00** (p5=17.7 p95=96.6).

## CORRECTION 1 — realistic display model (zenmetrics)

Registered `modern_oled_phone_indoor` in
`crates/cvvdp-gpu/data/display_models_imazen.json` (committed in
zenmetrics `8f46ed6`; this session verified + used it):

| param | value |
|---|---|
| resolution | 2532 × 1170 |
| diagonal | 6.1" |
| viewing distance | 0.35 m (hand-held) |
| **computed ppd** | **109.97** (`DisplayGeometry::pixels_per_degree`) |
| y_peak | 400 nit (indoor SDR auto-brightness setpoint) |
| y_black | 0.0005 nit (OLED native) |
| ambient | 250 lux → y_refl ≈ 0.398 nit → effective contrast ≈ 1000:1 |
| colorspace | sRGB / BT.709 |

Sanity check (8 KADID pairs spanning the distortion range), JOD under
`standard_4k` (75.4 ppd) vs `modern_oled_phone_indoor` (109.97 ppd):

```
distorted           std_4k JOD   phone JOD   delta
I02_02_05.png           5.339       5.871    +0.532
I57_01_04.png           7.562       8.237    +0.675
I70_13_04.png           8.386       8.615    +0.229
I39_24_02.png           7.291       7.493    +0.202
I61_11_02.png           9.524       9.700    +0.176
I16_20_04.png           8.875       8.963    +0.088
I06_06_01.png           9.766       9.787    +0.021
I23_03_01.png          10.000      10.000    +0.000   (identical pair)
mean: std=8.343  phone=8.583  mean delta=+0.240
```

The two display models give **consistently different JOD** — the new
preset is registered and active. Direction note: the phone gives
*higher* JOD (artifacts slightly LESS visible) because at 109.97 ppd
vs 75.4 ppd each artifact subtends a *smaller* visual angle and falls
further into the CSF high-frequency rolloff. (Higher ppd ≠ "zoomed
in"; it means more pixels per degree, i.e. each pixel is smaller.)
The 400-nit-vs-200-nit luminance lift partly offsets this but does
not reverse it.

## Phone-CVVDP scoring

`zenmetrics batch --metric cvvdp --display-model modern_oled_phone_indoor
--gpu-runtime cuda` on the row-aligned pairs TSVs (verified 0
ref_basename mismatches vs the canonical KADID/TID train parquets):

| corpus | n | phone-CVVDP JOD p5/p50/p95 |
|---|---|---|
| KADID | 10125 | 5.56 / 8.93 / 9.96 (min −3.12) |
| TID   | 3000  | 6.81 / 9.27 / 9.96 |

The zenmetrics build excluded `jxl-encoder` (a sibling repo broken at
HEAD on `crate::vardct::butteraugli_loop` / `perceptual_loop`) via
`--no-default-features --features 'png,gpu-cvvdp,gpu-cuda'` — KADID/TID
are 8-bit PNG so only png decoding is needed. Default-feature builds
are blocked until that sibling repo compiles; not in scope here.

## Dial transform (JOD → 0..100)

V12 monotone piecewise-linear band map (the canonical CVVDP-JOD →
dial map shared by the zensim anchor family), same transform the
canonical `cvvdp_log_norm`-adjacent dial uses; monotone so SROCC vs
raw phone-CVVDP is preserved. Resulting dial spread:

| corpus | dial p5/p50/p95 |
|---|---|
| KADID | 13.7 / 58.1 / 95.9 |
| TID   | 24.6 / 64.4 / 95.5 |

## CORRECTION 2 — phone-CVVDP-derived spline anchor

The prior attempt used a DESKTOP-CVVDP-derived anchor
(`anchors_cvvdp_372col_continuous.parquet`, target_score p50≈93.6),
mismatched with the phone-CVVDP dial distribution → broken dial
(G1=0.00, p5=96 p95=138). Fix: build the anchor FROM the phone-CVVDP
data — `modern_oled_anchor.parquet`, 1959 rows stratified across 20
uniform dial bins, `target_score` = phone-CVVDP dial (p5=7.2 p50=51.1
p95=95.0, min=0 max=100, every band populated). 372 features +
`anchor_weight` + `target_score`.

## Training (V39 recipe)

`zensim_mlp_train --hidden 128 --n-hidden-layers 2 --per-sample-alpha-head
--epochs 200 --lr 0.001 --l2 0.0001 --seed 17 --target-column human_score
--max-features 372 --mse-weight 0.6 --ranknet-weight 0.6 --monotonicity-reg 1.0
--minibatch-size 32 --auto-transforms <yeo_johnson screen> --anchor-parquet
modern_oled_anchor.parquet --anchor-loss-weight 0.01 --anchor-step-p 0.05`.

Held-out slice: 20% of refs by ref_basename hash (KADID 14 holdout
refs / 67 train; TID 6 / 19). Training never sees the holdout refs.

### Spline-fit fix (the actual dial repair)

The bake came out **distance-shaped**: raw output is NEGATIVELY
correlated with the dial target (anchor SROCC(raw, target) ≈ −0.93).
Both the in-trainer spline fitter (`mlp_train/mod.rs:7553`) and
`calibrate_v9_spline.py` assume a SCORE-shaped bake (raw ↑ as target
↑) and drop nearly every band for a distance-shaped one → degenerate
3–5-knot spline → broken dial.

Also: `--tanh-output-head-scale 30` saturated the network output to a
0.29-wide window (~51.1–51.4), destroying spline resolution. The
shipped variant drops the tanh pin (`--tanh-output-head-scale 0`,
script `train_zensim_b_phone_oled_notanh_2026-05-26.sh`) so the raw
output keeps usable per-band separation.

The PCHIP runtime `apply` only requires xs (pred) strictly increasing;
ys (target) may DECREASE. New calibrator
`scripts/v_next/calibrate_phone_spline_bidir.py` strips the degenerate
spline, re-scores the anchor (raw), auto-detects the dominant target
direction, and builds a monotone-**decreasing** 11-knot spline
(pred 51.076→51.304 ⇒ target 95.8→5.1), injected via the canonical
`zenpredict inspect`→`bake` JSON pipeline. Spline metadata: ZNPR v3
`zentrain.output_calibration_spline`.

## Results (held-out)

**Phone-CVVDP tracking** (bake output vs held-out phone-CVVDP — the
real "is it a good phone-CVVDP emulator" metric, `--bake-post clamp`):

| corpus (holdout) | n | SROCC | PLCC | dial p5/p50/p95 |
|---|---|---|---|---|
| KADID | 1750 | +0.9343 | +0.819 | 15.5 / 58.8 / 94.5 |
| TID   | 720  | +0.9301 | +0.870 | 17.8 / 66.4 / 97.8 |
| **pooled** | 2470 | **+0.9342** | +0.825 | 15.5 / 60.7 / 96.6 |

**bake_verdict G1 dial scorecard** (held-out CID22/KADID/TID/KonJND/AIC-3
pooled, logistic-rescaled dial space):

```
| G1 dynamic range | pooled p5≤25 ∧ p95≥85 | p5=17.7 p95=96.6 | 1.00 |
```

Prior broken attempt: p5=96 p95=138 (G1=0.00). **Fixed.**

Held-out human-MOS SROCC (NOT the goal metric — this bake emulates
phone-CVVDP, not human MOS; reported for transparency): CID22 0.701,
KADID 0.792, TID 0.816, AIC-3 0.749, KonJND 0.196 (PJND-threshold
scale, not a clean ranking corpus). These are below the desktop V39
ship (CID22 0.88) by construction — phone viewing de-weights the
high-frequency artifacts the desktop-viewed MOS corpora penalize.

## Artifacts

- bake: `zensim/weights/zensim_b_phone_oled_2026-05-26.bin` (ZNPR v3,
  236 KB) — wired into `DisplayTarget::Phone` via
  `profile::mlp_bake_cvvdp_phone_interim`.
- scores: `/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25/{kadid,tid}_cvvdp_phone.tsv`
- dial parquets + anchor: `/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25/`
  (`{kadid,tid}_phone_cvvdptgt.parquet`, `modern_oled_anchor.parquet`)
- train log: `/mnt/v/output/zensim/bakes/zensim_b_phone_oled_notanh_seed17_2026-05-26.train.log`
- verdict: `/mnt/v/output/zensim/bakes/zensim_b_phone_oled_notanh_seed17_cal_2026-05-26.verdict.md`

## Honest gaps

- The pre-spline raw output has low dynamic range (most rows cluster
  near pred≈51.2; only severe-distortion rows separate). The dial
  works because the monotone spline stretches that narrow band, but
  per-pair resolution in the high-quality tail (dial 80–100) is
  coarse. A higher-capacity head or a wider-spread training target
  could improve this.
- Safesyn (196k rows) was NOT scored — no row-aligned phone-CVVDP
  pairs TSV exists for it and building one (path reconstruction for
  196k encoded variants) is a separate infrastructure job. KADID+TID
  (13k) was sufficient for a working bake; safesyn is a volume-expansion
  follow-up.
- One seed (17). A small seed sweep would tighten the SROCC CI.
