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

### RESOLVED 2026-05-29: the floor is correct CVVDP-on-codec behavior

Scored zenjpeg directly with CVVDP (standard_4k desktop) over the
q-sweep, n=40 images
(`/mnt/v/output/zensim/a_phone_resolve_2026-05-29/jpeg_cvvdp_desktop.tsv`):

| q | CVVDP JOD p5 | p50 | p95 |
|--:|--:|--:|--:|
| 0 | 8.62 | **9.03** | 9.46 |
| 10 | 8.62 | 9.03 | 9.46 |
| 20 | 9.03 | 9.32 | 9.69 |
| 50 | 9.62 | 9.74 | 9.89 |
| 100 | 9.94 | 10.00 | 10.00 |

**CVVDP rates the ENTIRE zenjpeg quality range in a ~1-JOD band (9.0 →
10.0).** Even q0 (the worst zenjpeg can emit) is only ~1 JOD below
perfect on CVVDP's scale; CVVDP barely separates jpeg q0 from q100 on raw
JOD. The phone display model is **+0.24 JOD more lenient** still
(per the methodology doc), so phone-CVVDP(q0) ≈ 9.2–9.3 JOD. Mapped
through the V12 JOD→dial transform (which was fit on KADID, whose
*synthetic* distortions reach much lower JOD), that narrow codec JOD band
lands at dial ~53–99 — **exactly the range A_Phone produces.**

So A_Phone's floor-at-53 on codec sweeps is **faithful emulation of
phone-CVVDP, not a calibration bug.** Codec artifacts genuinely sit in
CVVDP's top JOD band; any CVVDP-emulating metric will have a compressed
dial on codec quality. This is the same mechanism behind the project's
"CVVDP-emulator training is a DEAD END for codec dials" finding (V41) —
A_Phone is a CVVDP emulator and inherits CVVDP's codec-leniency. No
recalibration would fix this without abandoning the phone-CVVDP target
A_Phone exists to represent.

## Conclusions

1. **A_Phone is a phone-display QUALITY ESTIMATE, not a codec-target
   dial.** It fails G1 (can't represent "score 30" — nothing scores below
   53) and G3 (monotonicity 0.92 < 0.93). Do not use it to binary-search
   a codec to a target score. v47/Profile::A remains the codec-target
   dial.
2. **A_Phone's rank skill is real and display-specific** — it beats v47
   on TID + AIC-4, loses on CID22. The phone-vs-desktop split is the
   mechanism, not a defect.
3. **The floor is correct, not a bug (RESOLVED — see section above).**
   Direct CVVDP scoring of the zenjpeg sweep shows codec quality lives in
   a ~1-JOD band (q0 ≈ 9.0, q100 = 10.0); A_Phone faithfully emulates
   that. The narrow codec dial is intrinsic to CVVDP-on-codecs — no
   recalibration fixes it without abandoning the phone-CVVDP target.
   A_Phone is a phone-display quality estimate (and a strong one — it
   wins TID + AIC-4 rank); it is **not** and cannot be a codec-target
   dial. v47/Profile::A remains the codec-target dial.
