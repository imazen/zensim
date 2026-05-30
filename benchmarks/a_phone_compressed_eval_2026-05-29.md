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

**A_Phone's dial floors high by design — but probably too high.** A
phone-OLED profile *should* score compressed images more leniently than
a desktop reference (smaller screen + viewing distance mask artifacts),
so a compressed dial range is expected. But **jpeg q0 → 60.7** (severe
8×8 blocking) and **avif q0 → 43.8** are implausibly high even for phone
viewing — a user can see q0 JPEG blocking on any display. The dial isn't
using its low range. This is a **calibration finding**: A_Phone's output
spline maps its (well-ranked) raw output into [53, 99] instead of
[~10, 99].

## Conclusions

1. **A_Phone is a phone-display QUALITY ESTIMATE, not a codec-target
   dial.** It fails G1 (can't represent "score 30" — nothing scores below
   53) and G3 (monotonicity 0.92 < 0.93). Do not use it to binary-search
   a codec to a target score. v47/Profile::A remains the codec-target
   dial.
2. **A_Phone's rank skill is real and display-specific** — it beats v47
   on TID + AIC-4, loses on CID22. The phone-vs-desktop split is the
   mechanism, not a defect.
3. **Next lever (needs the phone-CVVDP anchor):** refit A_Phone's output
   calibration spline on compressed phone-CVVDP data so the dial uses the
   full [~10, 99] range. SROCC is rank-invariant under a monotone spline,
   so this costs nothing on the rank wins and fixes G1. This is the
   deferred "A_Phone compressed recalibration" task — it requires
   locating the phone-CVVDP anchor parquet (not in the canonical val set
   yet).
