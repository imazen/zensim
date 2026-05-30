# A_Phone evaluated on compressed images — 2026-05-29

Per the request "we want A_Phone to be evaluated once compressed."
`ZensimProfile::A_Phone` (bake `zensim_b_phone_oled_2026-05-26.bin`,
phone-CVVDP-anchored) run through `bake_verdict` against the held-out
rank corpora AND the densified multi-codec compressed dial grid.

Reproduce:
```bash
ZENSIM_DIAL_GRID=/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29.parquet \
  ./target/release/bake_verdict --bake zensim/weights/zensim_b_phone_oled_2026-05-26.bin
```

## Rank (held-out corpora) — A_Phone vs v47/Profile::A

| corpus | A_Phone SROCC | v47 SROCC | A_Phone Z-RMSE | v47 Z-RMSE | winner |
|---|--:|--:|--:|--:|---|
| CID22 | 0.7011 | **0.8657** | 0.695 | 0.512 | v47 (by 0.16) |
| TID2013 | **0.8163** | 0.7927 | 0.562 | 0.577 | **A_Phone** |
| AIC-3 CTC | 0.7492 | **0.7680** | 0.660 | 0.620 | v47 (slight) |
| AIC-4 sample | **0.8930** | 0.8854 | **0.435** | 0.481 | **A_Phone** |

A_Phone is **not uniformly worse** — it wins TID and AIC-4 outright
(rank + calibration). It loses CID22 badly (−0.16 SROCC). CID22 is the
desktop-viewing MOS gold standard for codec distortions; A_Phone is
anchored to a *phone-OLED* CVVDP target, so a lower correlation with
desktop MOS is the expected display-model difference, not a regression.

## Dial (on the compressed grid) — A_Phone reads as a phone-display dial

| metric | A_Phone | v47 | note |
|---|--:|--:|---|
| inversions (>0.5pt) | 0.0775 | 0.0208 | A_Phone fails G3 (0.07) — mostly webp 0.137 |
| flat / clamp | 0.0163 | 0.0163 | tie — both clean |
| monotonicity | 0.9225 | 0.9792 | A_Phone fails G3 (0.93) |
| **dial p5 / p95** | **53.0 / 98.8** | 15.0 / 94.4 | **A_Phone floors at 53 — G1 fails** |

Per-codec `score @worst→@best` (worst quality = jpeg q0 / max-distance):

| codec | A_Phone worst→best | v47 worst→best |
|---|---|---|
| avif | 43.8 → 94.5 | 9.5 → 94.4 |
| jpeg | 60.7 → 94.7 | 17.9 → 92.6 |
| jxl | 55.4 → 77.9 | 29.0 → 74.9 |
| webp | 50.8 → 94.3 | 8.3 → 94.4 |

**A_Phone's dial floors high on codec sweeps — but this is a property of
its anchor, not provably a bug.** A_Phone's output spline was calibrated
on `modern_oled_anchor.parquet` (per
`benchmarks/zensim_b_phone_oled_methodology_2026-05-26.md`), which is
**KADID-only** phone-CVVDP (KADID's ~95%-non-compression synthetic
distortions — blur, noise, color, geometric — scored under the
`modern_oled_phone_indoor` display model). The spline's low knots were
therefore fit to *harsh synthetic* distortions, which span the full
[7, 99] dial range on that anchor. Codec artifacts are more
phone-forgiving than KADID's harshest distortions, so on the codec
sweep the dial only descends to ~53 (jpeg q0 → 60.7, avif q0 → 43.8).

Whether jpeg q0 *should* read ~60 on a phone is **not determinable from
the data we have** — there are no phone-CVVDP scores on codec sweeps
(the anchor is KADID-only). It's plausible that q0 blocking is genuinely
visible on a phone and 60 is too lenient; it's also plausible the phone
display model rates it that high. The honest statement: A_Phone's dial
range on *codec* distortions is narrower than its range on *synthetic*
distortions, by construction of its anchor.

## Conclusions

1. **A_Phone is a phone-display QUALITY ESTIMATE, not a codec-target
   dial.** It fails G1 (can't represent "score 30" — nothing scores below
   53) and G3 (monotonicity 0.92 < 0.93). Do not use it to binary-search
   a codec to a target score. v47/Profile::A remains the codec-target
   dial.
2. **A_Phone's rank skill is real and display-specific** — it beats v47
   on TID + AIC-4, loses on CID22. The phone-vs-desktop split is the
   mechanism, not a defect.
3. **The resolving experiment (not yet run):** score the codec dial-grid
   images under `modern_oled_phone_indoor`
   (`zen-metrics batch --metric cvvdp --display-model
   modern_oled_phone_indoor`) to get phone-CVVDP ground truth ON codec
   distortions. Then either (a) the truth confirms ~53-60 for q0 → A_Phone
   is correctly calibrated and the narrow codec dial range is real phone
   behavior, or (b) the truth says q0 ≈ 20 → A_Phone's KADID-only anchor
   under-represents codec distortions and the spline should be refit on a
   codec-inclusive phone-CVVDP anchor (rank-invariant under a monotone
   spline, so it costs nothing on the TID/AIC-4 wins). Until that runs,
   "recalibrate A_Phone" is premature — the anchor at
   `/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25/modern_oled_anchor.parquet`
   is KADID-only. Regardless of the outcome, A_Phone is **not** the
   codec-target dial — v47/Profile::A is.
