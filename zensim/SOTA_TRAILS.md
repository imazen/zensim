# Two-trail SOTA tracker

zensim ships TWO PreviewV0_5 variants in parallel, each defending a
different Pareto frontier. This doc is the source of truth for what
ships on each trail, what's been falsified against each trail's gate,
and the gate criteria themselves.

**Read this before training the next bake.** A new bake is shippable
on a trail iff it Pareto-beats that trail's current ship under that
trail's gate.

---

## Trail definitions

### Balanced trail (`PreviewV0_5Balanced` — historical alias `PreviewV0_5`)

**Audience.** Code that uses zensim as a general-purpose perceptual
metric across many distortion families — synthetic noise/blur,
geometric distortions, JND thresholds, compression artifacts.

**Gate** (formal):

| Corpus | Direction | Decisive rule |
|---|---|---|
| CID22  | A ≥ B   | § A.9 (n≥30 ∧ |h_SROCC|>1.96 ∧ |h_Z-RMSE|>1.96 ∧ PWRC_A>PWRC_B ∧ ≥4/6 panel stats favor A) |
| KADID  | A ≥ B   | not decisively B>>A on aggregate |
| TID    | A ≥ B   | not decisively B>>A on aggregate |
| KonJND | A ≥ B   | not decisively B>>A on aggregate |
| AIC-3  | A ≥ B   | not decisively B>>A on aggregate |

A ship requires **A>>B on ≥1 corpus and tied/A-favor on the other 4**.
Any single decisive B>>A on a corpus is a ship blocker.

### Compression trail (`PreviewV0_5Compression`)

**Audience.** Imageflow / commercial web compression pipelines where
the metric ranks compressed-image quality. CLAUDE.md: "Imageflow is
web-focused, not archival — commercial web compression targets
aggressive settings where every byte matters." CID22 is human MOS on
codec output; AIC-3 is human JND on near-PJND codec output. Both
score compression directly; the other 3 corpora score adjacent
non-compression behavior that's relevant but secondary.

**Gate** (formal):

1. **Decisive on at least one of {CID22, AIC-3}** per § A.9.
2. **Not decisively losing the other compression corpus** (CID22 or
   AIC-3, whichever wasn't the decisive winner).
3. **Mean SROCC regression on {KADID, TID, KonJND} no worse than
   −0.10 on any single corpus.** A −0.05 to −0.10 regression on
   synthetic / JND corpora is tolerated because they don't score
   compression directly.

A ship requires steps 1–3 ALL pass. Step 3 is the noise-tolerance
exception; without it the compression trail would collapse into the
balanced trail.

### Tuner trail (`PreviewV0_5Tuner`)

**Audience.** Codec auto-targeting pipelines where the user types a
target zensim score and the codec stack binary-searches the q (or
equivalent quality knob) that yields it. Distinct from Balanced /
Compression because **the gate is monotonicity + calibration honesty,
not cross-corpus SROCC**. Adding 2026-05-19 per user directive.

**Gate** (formal):

1. **Strict monotonicity on the JPEG 50-image × 19-q sweep ≥ 1 pp
   better than every V0_5 rank-trail ship.** Strict = decreases only;
   ties (clamp-flat regions) are counted separately. Failure modes
   the V0_5 ships exhibit (Balanced/Compression 70–80 % strict mono +
   57–76 % tied) are the explicit pathology this gate prevents.
2. **Tied rate ≤ 5 %** on the same sweep. A user dialing "score 70"
   cannot binary-search a dead zone where many adjacent q values map
   to the same clamped score.
3. **Dynamic range ≥ 50 score units** between q=5 median and q=95
   median across the sweep, so the user's 0..100 dial spans most of
   the useful quality regime.

**No SROCC gate.** Rank-honest cross-corpus performance is explicitly
SECONDARY for this trail. The variant doc on `PreviewV0_5Tuner` warns
"DO NOT use for general ranking workloads"; downstream eval is via the
`zensim-validate/src/bin/qsweep_eval` harness and per-image binary
search against codec output, not via the Mohammadi panel.

When a candidate passes all 3 gate criteria AND also passes one of
the rank trails — that's a Pareto upgrade and SHOULD rotate the
rank-trail ship as well. The 2026-05-19 ship fails the rank gates by
construction (safesyn-only training, no KADID/TID/KonJND
supervision); follow-on tuners that close the rank-trail gap are an
open research direction.

### Tuner trail v2 (`PreviewV0_5TunerV2`, EXP-CROSS-CODEC-V6)

**Audience.** Same as Tuner — codec auto-targeting pipelines. The V2
ship adds **cross-codec parity at every quality band** (V6's
piecewise multi-band anchor) to the Tuner trail's monotonicity + range
properties. Use when an orchestrator needs a single dial that's
calibrated AND comparable across JPEG / WebP / AVIF / JXL outputs at
every score band, not just at PJND.

**Gate** (extended Tuner gate, 6 sub-gates total — added 2026-05-19
per EXP-CROSS-CODEC-V6):

1. **Strict monotonicity ≥ 0.9378** on the JPEG 50-image × 19-q sweep
   (matches the Tuner trail's relative-to-V0_5-ships threshold).
2. **Tied rate ≤ 5 %** on the same sweep.
3. **Median range ≥ 50 score units** between q=5 median and q=95
   median.
4. **T=63 mean butter_pnorm3 < 2.5** across (jpeg, webp, avif) at
   target zensim=63 (n=20 image subset).
5. **Single-band cross-codec score std per source ≤ 5.0** at PJND
   (T=63), median across 1000 sources.
6. **Multi-band cross-codec score std per source ≤ 5.0** at EVERY
   of the 6 anchor bands (butter ∈ {0.3, 0.8, 1.5, 2.5, 4.0, 6.0}),
   not just T=63.

A ship requires all 6 sub-gates pass. The 2026-05-19 ship
(`cc4v6_w1p0_p0p30_s1`, anchor_w=1.0 seed=1) passes all 6
across all 3 evaluated anchor_w=1.0 seeds AND all 3 anchor_w=0.5
seeds. See `benchmarks/v_tuner_v6_methodology_2026-05-19.md`.

**TunerV2 supersedes Tuner for new orchestrator workloads** —
strict monotonicity is slightly lower (0.9522 vs 0.9767 for V5;
0.9278 for the original V_tuner-v2), but the dynamic range hits
78.2 score units (vs Tuner's 89.6) AND cross-codec parity is now
multi-band, not anchor-free. The Tuner slot remains shipped for
back-compat; callers can opt into either via the variant name.

### Tuner trail v3 (`PreviewV0_5TunerV3`, EXP-CROSS-CODEC-V9)

**Audience.** Same as TunerV2 — codec auto-targeting pipelines.
The V3 ship preserves V2's monotonicity + per-band cross-codec
parity AND adds **clean user-facing dial semantics**: full [0, 100]
range, JND lands on integer 60, JOD lands on integer 30. Use when
an orchestrator wants memorable round-number anchors so callers
type "score 60" for the JND threshold instead of "score 63" (the
2023-paper convention V2 inherited).

**Gate** (extended TunerV2 gate + V9 user-facing anchors, 11 total
sub-gates):

1–6: All 6 TunerV2 sub-gates (strict mono ≥ 0.9378 pair-based, tied
≤ 5%, median range ≥ 50, T=63 butter_pnorm3 < 2.5, PJND cc_std ≤ 5,
multi-band cc_std ≤ 5).
7. **Dial range = [0, 100]** (q=5 worst-codec → score≈0; q=95
   best-codec → score≈100).
8. **JND anchor**: score(butter=1.5) = 60.000 (exact integer landing).
9. **JOD anchor**: score(butter=4.0) = 30.000 (exact integer landing).
10. **Lossless ceiling**: score(butter≤0.05) = 100.000.
11. **Worst-codec floor**: score(butter≥12.0) = 0.000.

The 2026-05-20 ship (`v_tuner_v9`, K=32 seed=2 + PCHIP spline
calibration) passes all 11 sub-gates apples-to-apples vs the V6 ship
on the V6 metric + V6 qsweep corpus per the V9 mono audit
(`benchmarks/v_tuner_v9_mono_audit_2026-05-20.md`). The V9 initial
"FALSIFICATION" report was due to applying a different
qsweep-corpus + mono metric than the V6 ship-gate baseline;
re-measured the same way, all gates pass with margin.

**Mechanism.** V_tuner-v3 = V_tuner-v2 architecture (372 → 128 → 128
identity passthrough + per-sample-α head + tanh-output-head)
PLUS:

- **8-band extended-range anchor parquet**:
  `butter ∈ {0.05, 0.30, 0.60, 1.50, 2.50, 4.00, 7.00, 12.00}` ↔
  `score ∈ {100, 90, 80, 60, 50, 30, 10, 0}`. V2 used 6 bands at
  butter ∈ {0.3, 0.8, 1.5, 2.5, 4.0, 6.0}.
- **Post-network monotone PCHIP spline calibration**, fit AFTER
  training on the V9 anchor parquet predictions, baked as
  `zentrain.output_calibration_spline` metadata. The runtime
  applies the spline at scoring time via the dispatch in
  `zensim::metric::forward_one_bake` (commit `0829b51`). Spline is
  structurally monotone (Fritsch-Carlson), so it cannot reorder
  pairs.

**V3 supersedes V2 as the zensim-target CLI default** — the
`--profile tuner-v3` is the new default (was `tuner-v2`).
`--profile tuner-v2` still works for back-compat scoring. See
`benchmarks/v_tuner_v3_ship_2026-05-20.md` for the full ship
methodology, anchor table, and cross-codec smoke demo.

### Cross-codec trail (`PreviewV0_5CrossCodec`, opt-in)

**Audience.** Codec orchestrators that need consistent zensim scores
across multiple codec outputs at the same perceptual quality target
— e.g., a pipeline that picks between JPEG / WebP / AVIF / JXL given
a user-typed target zensim and expects all four codecs to produce
visually-similar outputs (matching butteraugli) when they land at
that target. Added 2026-05-19 per EXP-CROSS-CODEC-METRIC ship
decision.

**Gate** (formal, original):

1. **T=63 mean pairwise butteraugli_max < 2.5** across the
   (zenjpeg, zenwebp, zenavif) outputs of each image, after
   binary-searching each codec to land at zensim ≈ 63 (CID22-paper
   PJND anchor). Eval set: 6-image feature subset (same as
   `cross_codec_consistency.py`).
2. **CID22 SROCC drop ≤ 0.05** vs Tuner baseline.
3. **AIC-3 SROCC drop ≤ 0.05** vs Tuner baseline.
4. **KADID/TID/KonJND** lift OR drop ≤ 0.05 vs Tuner baseline.

**Strict gate (1) NOT achieved by the 2026-05-19 ship.** Best
principled seed reaches T=63 butter 4.82 (6-img) / 5.52 (20-img),
which is a **−25 to −31 % reduction** from the Tuner baseline (6.41
/ 8.07) — closes **~50 % of the gap** to the ~2.0 structural floor
but doesn't pass strict < 2.5. The compression-trail-style "close
gate" alternative (`butter in [2.5, 4.0]`) is also not met.

Ship rationale (opt-in only): the cross-codec equivalence-loss
mechanism produces a meaningful 25–46 % cross-codec consistency
improvement WITHOUT collapsing ranking quality (KADID +0.405, TID
+0.300, CID22 +0.022 vs Tuner baseline — the equivalence loss is a
generic cross-distortion-aligning signal). It defends a distinct
Pareto point from the existing trails. Future bakes that hit strict
< 2.5 should rotate this trail; current ship documents the gap and
preserves the wiring for follow-on work.

---

## Current ship per trail (2026-05-19)

| Trail | Bake | Bytes | Architecture | CID22 | AIC-3 | KADID | TID | KonJND |
|---|---|--:|---|---:|---:|---:|---:|---:|
| **Balanced** | V_22-mix-LARGE+iwssim s3 packed | 41,695 | 300→128→1 vanilla MLP | 0.8324 | 0.7845 | **0.9677** | **0.9729** | **0.8927** |
| **Compression** | V_24-per-sample-α s4 packed | 44,109 | 300→128→128(identity) + per-sample-α head | **0.8641** | **0.8183** | 0.9316 | 0.8893 | 0.8080 |
| **Ensemble** | V_05-ensemble classifier + B + C | 22,690 (classifier only) | 300→64→1 ReLU classifier routes to Balanced or Compression | 0.8632 | 0.8131 | 0.9676 | 0.9719 | 0.8792 |
| **Tuner** | V_tuner-v2-s2 calibrated (2026-05-19) | 261,316 (F32, unpacked) | 372→128→128(identity) + per-sample-α head, mse-only train, affine α=−1590.55 β=52.02 | 0.8786 | 0.8130 | 0.7704 | 0.7476 | 0.2351 |
| **TunerV2** | V_tuner-v6 (cross-codec V6, anchor_w=1.0 s1, 2026-05-19) | 261,351 (F32, unpacked) | 372→128→128(identity) + per-sample-α head + tanh-output-head scale=15.0, mse + cross-codec equiv (W=1.0) + multi-band anchor (W=1.0, 6 bands at score ∈ {90,75,63,45,25,10}) + dyn-range floor + mono reg, no external affine | 0.8770 | 0.7961 | 0.7179 | 0.7542 | 0.1962 |
| **TunerV3** | V_tuner-v9 (extended-range + PCHIP spline, K=32 s2, 2026-05-20) | 261,451 (F32, unpacked) | 372→128→128(identity) + per-sample-α head + tanh-output-head scale=15.0 + **post-network PCHIP spline calibration**, 8-band anchor at butter ∈ {0.05,0.3,0.6,1.5,2.5,4.0,7.0,12.0} ↔ score ∈ {100,90,80,**60**,50,**30**,10,**0**} — JND on integer 60, JOD on integer 30, full [0,100] dial range | 0.8530 | 0.7870 | 0.7060 | 0.7150 | 0.1860 |
| **CrossCodec** (opt-in) | V_24-per-sample-α + cross-codec equiv-loss W=1.0 s1 (2026-05-19) | 261,316 (F32, unpacked) | 372→128→128(identity) + per-sample-α head, mse + (y_a − y_b)² cross-codec equiv pair loss, no external affine | 0.8797 | 0.8060 | 0.8003 | 0.8215 | 0.3269 |

Tuner monotonicity panel on the 50-image × 19-q JPEG sweep
(`zensim-validate/src/bin/qsweep_eval`, n=900 adjacent pairs):

| Trail bake | strict_mono | tied_rate | dynamic_range (q=5 med → q=95 med) |
|---|---:|---:|---:|
| **Tuner (this trail's ship)** | **0.9278** | **0.0044** | **5.0 → 94.6** = 89.6 units |
| PreviewV0_3 (legacy V_18 ship) | 0.9367 | 0.0078 | 14.4 → 95.4 = 81.0 units |
| Balanced | 0.7800 | 0.7556 | 12.0 → 0.0 = anti-monotonic dead zone |
| Compression | 0.7189 | 0.7033 | 14.4 → 0.0 = anti-monotonic dead zone |
| Ensemble | 0.8611 | 0.5733 | mixed (classifier-routed) |

Tuner beats every V0_5 rank-trail ship by 6–21 pp on strict monotonicity
AND has effectively no clamp-flat dead zones (0.44 % tied vs 57–76 % for
the V0_5 ships, whose clamps pin most of the q-range to score=0). The
89.6-unit dynamic range covers most of the user-facing 0..100 dial.

CrossCodec consistency panel — mean pairwise butteraugli_max between
(zenjpeg, zenwebp, zenavif) outputs, each codec binary-searched to
hit the target zensim score (the smaller the number, the more
consistent the codec outputs at that target):

| Trail bake | T=63 (6-img) | T=63 (20-img) | T=70 (6-img) | T=70 (20-img) |
|---|---:|---:|---:|---:|
| Tuner (baseline) | 6.41 | 8.07 | 1.73 | 2.11 |
| **CrossCodec W=1.0 s1** | **4.82** (−25 %) | **5.52** (−31 %) | **1.13** (−35 %) | **1.13** (−46 %) |
| CrossCodec W=3.0 (rank-degenerate, not shipped) | 4.29 (−33 %) | 3.67 (−55 %) | — | — |
| Structural floor (codec-disjoint butter) | ~2.0 | ~2.0 | ~1.0 | ~1.0 |

CrossCodec closes ~50 % of the gap from Tuner to the structural ~2.0
butter floor at T=63. The strict `< 2.5` gate is **not** achieved
at the principled-seed ship (best non-rank-degenerate run is W=1.0
seed=1, T=63 butter 4.82/5.52). Higher equivalence-loss weights
(W=3.0) get closer to the gate but produce rank-collapsed bakes
(seed 2 hits 2.81/2.97 with KADID 0.308, TID 0.367 — rank
quality dies). Per `benchmarks/v_cross_codec_findings_2026-05-19.md`.


Balanced is ZNPR v3, i8 + zerobias + lz4 packed, no metadata
payload, standard `Predictor::predict` runtime.

Compression is ZNPR v3, i8 + zerobias + lz4 packed, carries
`zentrain.per_sample_alpha_head` metadata; runtime dispatch lives
in `zensim::metric::forward_one_bake` (per-sample-α dispatch landed
2026-05-18 — supersedes V_22-372feat s5 on the compression trail).

Ensemble is a runtime-routing profile (`PreviewV0_5Ensemble`,
EXP-ENSEMBLE-V05, 2026-05-18) that forwards a 300 → 64 → 1 ReLU
classifier per pair, then dispatches to either the Balanced or
Compression bake based on the classifier sign. Full-corpus routing
accuracy on the canonical 5-corpus val set: **98.6 %**. Per-corpus
SROCC tracks `max(Balanced, Compression)` to within 0.014 on every
corpus. Pareto-dominates the Compression ship on every corpus
(decisive wins on KADID/TID/KonJND, ties on CID22/AIC-3). Vs the
Balanced ship: decisive wins on CID22+AIC-3, ties on KADID+TID,
decisive loss on KonJND (-0.014, within compression-trail § A.10
-0.10 synthetic tolerance). The Ensemble passes the
**compression-trail gate**, fails the balanced-trail gate; ships as
a third variant rather than rotating either trail. Methodology +
full Mohammadi panel:
`benchmarks/exp_ensemble_v05_eval_2026-05-18.md`.

CrossCodec is ZNPR v3, F32 uncompressed (261,316 bytes), carries
`zentrain.per_sample_alpha_head` metadata; runtime dispatch reuses
the per-sample-α path in `zensim::metric::forward_one_bake` (no
new dispatch code needed — landed 2026-05-18 for the Compression
trail). 372-feature input (228 standard + 72 masked + 72 IW pool);
the runtime computes IW-pool features only for this profile +
Tuner. Trained with the Tuner-v2 recipe PLUS
`--cross-codec-eq-parquet … --cross-codec-eq-weight 1.0
--cross-codec-eq-step-p 0.10` over ~58k equivalence pairs spanning
{zenjpeg, zenwebp, zenavif, zenjxl} × 20 butter levels. The
cross-codec loss is a per-pair `(y_a − y_b)²` term that pulls the
metric toward consistent scores for codec outputs of comparable
butter_pnorm3. Soft-clamped in [0, 100] (rank-head extrapolation
on OOD content). **Opt-in profile** — does NOT pass the strict
cross-codec gate (`T=63 butter < 2.5`); ships because it
meaningfully reduces cross-codec inconsistency WITHOUT collapsing
rank quality. Methodology +  full panel:
`benchmarks/v_cross_codec_methodology_2026-05-19.md`,
`benchmarks/v_cross_codec_findings_2026-05-19.md`.

### Superseded compression ship

V_22-372feat s5 packed (51,153 bytes, md5
`3be4f781238dcb35f32c964cb218a8a4`) was the compression ship from
2026-05-18 until per-sample-α runtime dispatch landed later the
same day. It loses CID22 / AIC-3 / TID decisively to V_24-per-
sample-α s4 per § A.9 (1000-bootstrap, see § "V_24-per-sample-α s4
packed vs V_22-372feat s5" below). Kept at
`zensim/weights/v_compression_2026-05-18.bin` for reproducibility.

---

## Candidate matrix (every credible candidate evaluated against both gates)

| Candidate | Bake date | n_inputs | Runtime path | CID22 | AIC-3 | KADID | TID | KonJND | Balanced gate | Compression gate |
|---|---|--:|---|---:|---:|---:|---:|---:|---|---|
| V_22-mix-LARGE+iwssim s3 packed | 2026-05-18 | 300 | vanilla `predict` | 0.8324 | 0.7845 | 0.9677 | 0.9729 | 0.8927 | **SHIP** (current) | tied on CID22+AIC-3, no win |
| V_22-372feat s5 packed | 2026-05-18 | 372 | vanilla `predict` | 0.8580 | 0.8087 | 0.9319 | 0.8875 | 0.8125 | FAIL (B>>A on KADID/TID/KonJND) | **superseded** by V_24-per-sample-α s4 on 2026-05-18 (after runtime dispatch landed); decisive loss on CID22+AIC-3+TID. Kept in `weights/` for reproducibility. |
| V_22-372feat noLARGE s5 | 2026-05-18 | 372 | vanilla `predict` | 0.8425 | 0.8059 | 0.9311 | 0.8897 | 0.8371 | FAIL | promising — marginally smaller CID22/AIC-3 lift than s5+LARGE, retained as falsification record |
| V_24-per-sample-α s4 packed | 2026-05-18 | 300 | **per-sample-α head dispatch** (zensim::metric::forward_one_bake, 2026-05-18) | 0.8641 | 0.8183 | 0.9316 | 0.8893 | 0.8080 | FAIL (B>>A on KADID/TID/KonJND) | **SHIP** (current) — decisive A>>B vs 372feat on CID22+AIC-3+TID per § A.9 (1000-bootstrap); KADID promising; KonJND tied. KADID/TID/KonJND vs Balanced within −0.10 noise tolerance. |
| V_24-α=0.10 5-seed | 2026-05-18 | 300 | vanilla `predict` | 0.8686 | 0.7912 | 0.8996 | 0.8883 | 0.8306 | FAIL | FAIL (AIC-3 +0.004 not decisive, KADID/TID −0.07) |
| V_24-stdpool prod | 2026-05-18 | 300 | vanilla `predict` | 0.8376 | 0.7785 | 0.9167 | 0.8912 | 0.5414 | FAIL (KonJND catastrophic) | FAIL (no AIC-3 win; KonJND −0.35) |
| V_24-FT-gentle s4 packed | 2026-05-18 | 300 | per-sample-α head dispatch | 0.8451 | 0.8131 | 0.9321 | 0.8896 | 0.8544 | FAIL (CID22 −0.0 vs 372feat) | FAIL — vs new per-sample-α s4 ship: B>>A decisive on CID22 (h=−398.9) AND AIC-3 (h=−73.7); strictly dominated on compression corpora. |
| V_24-PS-konjnd010 | 2026-05-18 | 300 | per-sample-α head dispatch | 0.794 | 0.803 | 0.930 | 0.889 | **0.971** | FAIL (CID22 −0.04) | FAIL (CID22 −0.04 decisive) |
| V_24-hybrid NiN s2 packed (f16+zstd) | 2026-05-18 | 300 | **hybrid-head dispatch** (zensim::metric::forward_one_bake, 2026-05-18) | 0.8727 | 0.8096 | 0.9319 | 0.8884 | 0.7906 | FAIL | FAIL by 0.002 on step 3 — A>>B decisive on CID22 (+0.040) and AIC-3 (+0.025) vs Balanced; KonJND −0.102 just over the −0.10 ceiling. vs current per-sample-α ship: A>>B decisive on CID22 (+0.0086) but B>>A decisive on AIC-3 (−0.0087) AND KonJND (−0.017) — fails compression-gate step 2. |
| V_24-hybrid no-NiN s4 packed (f16+zstd) | 2026-05-18 | 300 | hybrid-head dispatch | 0.8657 | 0.8061 | 0.9285 | 0.8890 | 0.7901 | FAIL | FAIL — vs Balanced KonJND −0.103 fails step 3; vs per-sample-α ship: tied on CID22, B>>A decisive on KADID/AIC-3/KonJND. Strictly dominated by per-sample-α s4 on compression corpora. |
| V_22-IW v2 (calibrated) | 2026-05-16 | 372 | vanilla `predict` + feature_transforms | 0.8164 | 0.8071 | 0.9475 | 0.9617 | n/a | FAIL (CID22 −0.077) | tied on AIC-3 +0.023, but CID22 −0.077 (loses compression-trail gate step 2) |
| EXP-BALANCED-TILT cell0 small (kw=0.05, lw=0.5) seed=3 | 2026-05-18 | 300 | per-sample-α head dispatch | 0.8249 | 0.8144 | 0.9345 | 0.8898 | 0.9234 | FAIL (B>>A on KADID+TID) | FAIL (B>>A on CID22; tied AIC-3) |
| EXP-BALANCED-TILT cell1 moderate (kw=0.10, lw=0.5) seed=3 | 2026-05-18 | 300 | per-sample-α head dispatch | 0.8159 | 0.8051 | 0.9359 | 0.8896 | 0.9567 | FAIL (B>>A on CID22+KADID+TID) | FAIL (B>>A on CID22 AND AIC-3) |
| EXP-BALANCED-TILT cell2 heavy (kw=0.10, lw=0.3) seed=3 | 2026-05-18 | 300 | per-sample-α head dispatch | 0.8112 | 0.8070 | 0.9379 | 0.8901 | 0.9532 | FAIL (B>>A on CID22+KADID+TID) | FAIL (B>>A on CID22 AND AIC-3) |
| EXP-BALANCED-TILT cell3 no_large (kw=0.10, lw=0.0) seed=3 | 2026-05-18 | 300 | per-sample-α head dispatch | 0.7686 | 0.8056 | 0.9385 | 0.8906 | 0.9661 | FAIL (B>>A on CID22+KADID+TID) | FAIL (B>>A on CID22 AND AIC-3) — boosting kadid_w/tid_w on per-sample-α architecture FALSIFIED. See `benchmarks/exp_balanced_tilt_falsified_2026-05-18.md`. |
| EXP-PERSAMPLE-MIX3 (median s1 packed) | 2026-05-18 | 372 | per-sample-α head dispatch | 0.8553 | 0.8057 | 0.9304 | 0.8783 | 0.8939 | FAIL (B>>A on KADID/TID; balanced step 2 fails by wide margin) | FAIL (B>>A on CID22 AND AIC-3 vs current ship; step 1 fails). 5-seed CI: CID22 mean 0.8545 σ=0.0110, KonJND mean 0.8852 σ=0.0201. Adding 30% ssim2 to target gains KonJND (+0.086) but loses CID22 (−0.0088) and AIC-3 (−0.0126) decisively vs per-sample-α s4 alone. Per `benchmarks/exp_persample_mix3_falsification_2026-05-18.md`. |
| EXP-ENSEMBLE-V05 (classifier + B + C) | 2026-05-18 | 300 | **ensemble-classifier routing dispatch** (zensim::metric::apply_mlp_scoring, 2026-05-18) | 0.8632 | 0.8131 | 0.9676 | 0.9719 | 0.8792 | FAIL (KonJND −0.014 decisive) | **SHIP** (new third variant, `PreviewV0_5Ensemble`) — decisive A>>B vs Balanced on CID22 (+0.031) AND AIC-3 (+0.029); decisive A>>B vs Compression on KADID/TID/KonJND. Pareto-better than Compression ship on every corpus; ties or wins everything vs Balanced except KonJND (which is within −0.10 tolerance). Adds 22.7 KB classifier bake. |
| EXP-V22-PERSAMPLE s2 packed (5-seed CI) | 2026-05-18 | 300 | per-sample-α head dispatch | 0.8549 | 0.8084 | 0.9312 | 0.8899 | 0.8269 | FAIL (B>>A on KADID+TID+KonJND) | FAIL (B>>A on CID22 AND AIC-3, KonJND +0.019 promising) — V_22-LARGE+iwssim recipe + per-sample-α head: head architecture alone yields +0.022 CID22 / +0.024 AIC-3 over Balanced ship but is strictly dominated by V_24-per-sample-α s4 on both compression corpora. 5-seed mean CID22 0.8582 std 0.0045; median seed s2 chosen. See `benchmarks/exp_v22_persample_falsification_2026-05-18.md`. |
| EXP-IWSSIM-PERSAMPLE (median s3) | 2026-05-18 | 300 | per-sample-α head dispatch | 0.8406 | 0.7929 | 0.9671 | 0.9814 | 0.8053 | FAIL (B>>A on KonJND vs Balanced) | FAIL (B>>A on BOTH CID22 AND AIC-3 vs Compression ship). 5-seed CI: CID22 mean 0.8402 σ=0.0040, AIC-3 mean 0.7992 σ=0.0056. Dropping cvvdp from target (pure `iwssim_log_norm`) makes a KADID/TID specialist matching the Balanced ship's synth profile (KADID +0.035, TID +0.092 vs Compression) but loses both compression-band corpora (ΔCID22 −0.024, ΔAIC-3 −0.025 vs Compression). Target-shape map: iwssim-only → KADID+TID specialist (no trail slot). Per `benchmarks/exp_iwssim_persample_falsification_2026-05-18.md`. |
| EXP-V22-HYBRID s3 packed (5-seed CI) | 2026-05-18 | 300 | **hybrid-head dispatch** (shared scalar α) | 0.8657 | 0.8034 | 0.9315 | 0.8906 | 0.7814 | FAIL (B>>A on KADID+TID+KonJND; KonJND −0.111 breaches −0.10 tolerance) | FAIL (CID22 tied + AIC-3 B>>A fail step 1; AIC-3 −0.015 fails step 2) — V_22-LARGE+iwssim recipe + hybrid_head (no per-sample-α, shared learned scalar gate): trades the per-sample architectural lever for shared-α, gives +0.033 CID22 / +0.019 AIC-3 over Balanced but at KonJND −0.111 cost. CID22 statistically tied with Compression ship (Δ=+0.0016); AIC-3 falls decisively. 5-seed CI: CID22 mean 0.8623 σ=0.0119, KonJND mean 0.7646 σ=0.0186. Architectural choice (hybrid_head shared-α) is materially identical to V_24-hybrid no-NiN s4 on this corpus mix — confirms `hybrid_head` doesn't flip either gate when applied to the V_22 recipe. Per `benchmarks/exp_v22_hybrid_falsification_2026-05-18.md`. |
| EXP-PERSAMPLE-CAPACITY h=256 (5-seed) | 2026-05-18 | 300 | per-sample-α head dispatch | 0.8683 (s2) / 0.8580 (median) | 0.8156 (best) / 0.8125 (median) | 0.9340 | 0.8918 | 0.8466 | FAIL (B>>A on KADID/TID/KonJND vs Balanced; CID22 −0.06 vs Balanced 0.832 best-seed +0.04 not enough) | FAIL — best-seed h256_s2: A>>B on CID22 (+0.0042) but B>>A decisive on AIC-3 (−0.0099) vs Compression ship, fails "not decisive B>>A on the other compression corpus" clause. Median-seed h256_s1: B>>A on CID22 decisive (−0.0061). 5-seed CI: CID22 mean 0.8603 σ=0.0058, range [0.8540, 0.8683]. The best-seed CID22 lift is single-seed-selection artifact — h=256 has WIDER seed-variance band (range 0.0143) than h=128 ship (range 0.0100 per RECIPE-AUDIT). Capacity is saturated at h=128 for the V_24-per-sample-α recipe. Per `benchmarks/exp_persample_capacity_falsified_2026-05-18.md`. |
| EXP-PERSAMPLE-CAPACITY h=512 (3-seed) | 2026-05-18 | 300 | per-sample-α head dispatch | 0.8628 (s2) / 0.8567 (median) | 0.8137 (best) / 0.8122 (median) | 0.9345 | 0.8905 | 0.8497 | FAIL (B>>A on KADID/TID vs Balanced) | FAIL — best-seed h512_s2 vs Compression ship: CID22 tied (Δ=−0.0013) AND AIC-3 B>>A decisive (−0.0095). 4× param expansion (h=128 → h=512) monotonically degrades compression-trail metrics (CID22 falls h=128 0.8641 → h=256 0.8683/median 0.8580 → h=512 0.8628; AIC-3 falls 0.8183 → 0.8156 → 0.8137). Pure width scaling on V_24-per-sample-α recipe is DEAD. Per `benchmarks/exp_persample_capacity_falsified_2026-05-18.md`. |
| EXP-CROSS-CODEC-METRIC W=1.0 s1 | 2026-05-19 | 372 | per-sample-α head dispatch | 0.8797 | 0.8060 | 0.8003 | 0.8215 | 0.3269 | FAIL (B>>A on KonJND −0.57 vs Balanced; KADID/TID also lose decisively) | FAIL (CID22 +0.016 not decisive; AIC-3 −0.012; KADID −0.13 fails step 3 ceiling) — Cross-codec equivalence-pair loss W=1.0 closes T=63 cross-codec butter from 6.41 → 4.82 (−25 %) on the 6-img subset (5.52 / −31 % on the 20-img). Doesn't pass strict cross-codec gate (< 2.5) but ships as opt-in `PreviewV0_5CrossCodec` on the new **cross-codec trail** because the mechanism produces a meaningful 25–46 % cross-codec consistency improvement WITHOUT rank collapse. Mechanism side-effect: KADID +0.405, TID +0.300, CID22 +0.022 vs Tuner baseline (equivalence loss is a distortion-type-invariant feature learner). Seeds 2 / 3 documented in findings doc: seed 2 hits 2.81/2.97 cross-codec butter but rank-collapses (KADID 0.308, TID 0.367) — not viable. Per `benchmarks/v_cross_codec_methodology_2026-05-19.md` + `benchmarks/v_cross_codec_findings_2026-05-19.md`. |
| EXP-CROSS-CODEC-V3 6-bake matrix | 2026-05-19 | 372 | per-sample-α head dispatch + rank-preserve + σ-floor probe | 0.852-0.883 across 6 bakes | n/a | n/a | n/a | n/a | n/a (Tuner trail focus) | n/a (Tuner trail focus). **Tuner trail: FAIL all 6.** V3 added rank-preserve (`--cross-codec-rank-preserve-weight 0.2`, RankNet on `|butter_diff|` per equiv pair) + dynamic-range floor probe (`L = w · max(0, σ_thresh − σ_obs)²` on N=40 random equiv-pool A-side rows, σ ≥ 15, 5% step rate) + stronger `--monotonicity-reg 5.0`. **Range gate: SOLVED.** Every V3 candidate has post-affine range 89.94-90.02 score units (V2 cc4v2 had 0.10-0.92 → V3 architectural counterweights successfully prevent collapse). **Cross-codec gate passes on 2/6**: cc4v3_s1_w1_0 (butter_p3 2.26), cc4v3_s2_w1_0 (butter_p3 2.21) — first non-collapsed candidates to pass strict < 2.5. **Mono gate FAIL on all 6.** Best V3 strict-mono = 0.9100 (cc4v3_s2_w0_5) vs baseline 0.9278 vs gate 0.9378. Root cause: raw output range 10-13 score units → affine β=5-10× amplifies per-pair jitter ~0.05-0.2 raw → 0.5-2 score units, trips mono detection + drives tied rate to 5-10% on 5/6. The σ-floor (cross-image-pool) requires output spread that conflicts with smooth within-curve monotonicity. V4 proposals: per-curve σ-substrate (V4-A, recommended); relax mono-reg + boost rank-preserve (V4-B); architectural tanh-pinned output (V4-C); 2-head ensemble (V4-D). Per `benchmarks/v_tuner_v3_falsification_2026-05-19.md`. |

**Runtime status (2026-05-18, late)**: the per-sample-α dispatch
landed in `zensim::metric::forward_one_bake`. Bakes carrying
`zentrain.per_sample_alpha_head` metadata are now scoreable through
the production runtime (the bake's final layer is a `n_hidden ×
n_hidden` identity passthrough; the runtime reads the post-LeakyReLU
hidden vector as `out`, parses the head payload, and mixes rank +
pool via the per-sample sigmoid gate). Same dispatch is in
`bake_verdict` (`score_row`) and `bake_compare` (`score_corpus`)
for parquet-driven validation.

**Hybrid-head dispatch also landed 2026-05-18 (late, same day)**.
Bakes carrying `zentrain.hybrid_head` metadata score through an
analogous path — the same n_hidden passthrough trick, but the α
gate is a single learned SCALAR (`α = σ(α_logit)`) shared across
samples rather than computed per-sample. Both V_24-hybrid NiN s2
and no-NiN s4 are now evaluable on the production runtime. Neither
passes the compression-trail gate decisively: NiN s2 fails step 3
on KonJND by 0.002, no-NiN s4 fails step 3 on KonJND by 0.003 AND
is strictly dominated by per-sample-α s4 vs the current ship on
all compression corpora. The audit doc's "borderline (within 0.005
of the gate)" classification holds.

The finetune V_24 (FT-gentle) architecture uses
`zentrain.per_sample_alpha_head` metadata too (confirmed via
`zenpredict inspect` on 2026-05-18) and is now evaluable — but
it's also strictly dominated by per-sample-α s4 on the compression
corpora (B>>A decisive on both CID22 and AIC-3) so it does not
rotate the trail ship. FT-gentle's value-add was tighter KonJND
preservation (0.8544 vs 0.8080 for the ship); the gate as currently
specified weights both equally and lets the per-sample-α s4 win on
the strict reading.

The `zentrain.pool_head_reducer` (pure pool head, no rank head)
remains the one untested-on-runtime architecture, but no candidate
bake in the audit-doc tier-1 or tier-2 lists carries it — the
canonical V_24 family converged on hybrid-head and per-sample-α
variants. Skipping until a candidate appears.

---

## Per-corpus bake_compare verdicts (1000-bootstrap, § A.9)

### V_22-372feat s5 packed vs V_22-mix-LARGE+iwssim baseline (compression-trail ship)

Full report at `/tmp/two_trail_372feat_vs_baseline.md` (1000-bootstrap
A vs B per § A.9). Aggregate verdicts:

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8580 | 0.8324 | 0.520 | 0.559 | 0.9126 | 0.9006 | +29.571 | +80.453 | +24.643 | **A>>B** |
| AIC-3  | 600  | 0.8087 | 0.7845 | 0.577 | 0.606 | 0.8804 | 0.8630 | +20.574 | +40.963 | +17.145 | **A>>B** |
| KADID  | 10125 | 0.9319 | 0.9677 | 0.362 | 0.249 | 0.9601 | 0.9804 | -90.446 | -795.837 | (B>>A) | B>>A |
| TID    | 3000 | 0.8875 | 0.9729 | 0.436 | 0.236 | 0.9158 | 0.9832 | -54.017 | -303.890 | (B>>A) | B>>A |
| KonJND | 1008 | 0.8125 | 0.8927 | 0.498 | 0.376 | 0.8504 | 0.9178 | -38.736 | -118.530 | (B>>A) | B>>A |

Decision: **compression-trail SHIP**.

- Step 1: A>>B on CID22 AND AIC-3 (both decisively per § A.9). PASS.
- Step 2: not B>>A on the other compression corpus (CID22 ↔ AIC-3
  both decisively A>>B; neither is B>>A). PASS.
- Step 3: KADID −0.036, TID −0.085, KonJND −0.080 all within −0.10
  noise tolerance. PASS.

### V_22-mix-LARGE+iwssim baseline vs V_22-372feat s5 (balanced-trail defense)

Same data, opposite direction. The balanced ship beats 372feat
decisively on 3/5 corpora (KADID +0.036 decisive, TID +0.085
decisive, KonJND +0.080 decisive). 372feat would FAIL the
balanced-trail gate (which forbids decisive B>>A on any of
{KADID, TID, KonJND, AIC-3}). So the balanced trail keeps V_22-mix-
LARGE+iwssim; 372feat is incompatible with that gate by design.

### V_24-per-sample-α s4 packed vs baseline (compression-trail SHIP from 2026-05-18 — runtime dispatch landed)

Fresh 1000-bootstrap report at
`/tmp/two_trail_persample_vs_baseline.md`. Aggregate verdicts:

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | h_Z | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8641 | 0.8324 | +32.369 | +94.139 | +26.974 | **A>>B** |
| AIC-3  | 600  | 0.8183 | 0.7845 | +24.868 | +49.602 | +20.724 | **A>>B** |
| KADID  | 10125 | 0.9316 | 0.9677 | -88.609 | -774.411 | (B>>A) | B>>A |
| TID    | 3000 | 0.8893 | 0.9729 | -53.470 | -303.898 | (B>>A) | B>>A |
| KonJND | 1008 | 0.8080 | 0.8927 | -40.936 | -121.668 | (B>>A) | B>>A |

Per the compression-trail gate this PASSES — and decisively beats
372feat in a head-to-head (see next sub-section). The bake's
`zentrain.per_sample_alpha_head` dispatch landed in
`zensim::metric::forward_one_bake` on 2026-05-18 (the runtime
detects the metadata payload, parses W_α / b_α / rank_w / rank_b /
reducer_w / reducer_b / p_norm as f32-LE, and mixes y_rank + y_pool
via the per-sample sigmoid gate). Bake_verdict and bake_compare
got the same dispatch in `score_row` / `score_corpus`. The packed
bake at `zensim/weights/v_compression_persample_2026-05-18.bin`
(44,109 bytes, md5 `f09a9abdce00805000c1d112c2421b2d`) IS the
current compression-trail ship.

Round-trip verification (packed vs unpacked seed4 on CID22): SROCC
0.8641 (packed) vs 0.8640 (unpacked) = 0.0001 drift, well under
the 0.0005 pack-quality threshold.

### V_24-per-sample-α vs V_22-372feat (compression-trail head-to-head)

Full report at `/tmp/two_trail_persample_vs_372feat.md` (1000-bootstrap).

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | h_Z | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8641 | 0.8580 | +26.374 | +104.817 | +21.978 | **A>>B** |
| AIC-3  | 600  | 0.8183 | 0.8087 | +52.598 | +110.536 | +43.832 | **A>>B** |
| TID    | 3000 | 0.8893 | 0.8875 | +48.655 | +224.779 | +32.437 | **A>>B** |
| KADID  | 10125 | 0.9316 | 0.9319 | -7.823 | -8.873 | +1.304 | promising |
| KonJND | 1008 | 0.8080 | 0.8125 | -5.255 | -9.529 | +0.000 | tied |

Per-sample-α decisively beats 372feat on 3/5 corpora; ties or
promising on the other 2. **Per § A.9 strict majority rule,
per-sample-α IS the compression-trail SOTA.** Confirmed 2026-05-18
post-runtime-dispatch with a fresh 1000-bootstrap rerun of
bake_compare against the packed bake; numbers above are stable
under the i8+zerobias+lz4 pack (round-trip CID22 SROCC drift
0.0001).

---

## Falsified hypotheses (closed against both trails)

These were tested and lost decisively. Re-opening requires NEW evidence.

- **EXP-CROSS-CODEC-V3 6-bake matrix** (Tuner trail): range gate
  SOLVED (89.94-90.02 vs V2 collapse 0.10-0.92) — rank-preserve +
  σ-floor architectural counterweights work. Mono gate FAIL all 6
  (best 0.9100 < baseline 0.9278 < gate 0.9378). β=5-10× affine
  amplifies raw per-pair jitter into score-unit mono violations. V4
  proposals: per-curve σ-substrate (V4-A, recommended), relax mono-reg
  + boost rank-preserve (V4-B). Per
  `benchmarks/v_tuner_v3_falsification_2026-05-19.md`. *2026-05-19*
- **V_24 full 3-way mix** (cv/iw/sm 0.33 each): CID22 +0.038 but
  KADID/TID −0.09. Decisively B>>A on 16/18 cells. *2026-05-18*
- **V_24-stdpool head NiN-off**: hypothesis was "NiN-on regularization
  caused KonJND collapse" — falsified. KonJND stayed at 0.52 with NiN
  off. *2026-05-18*
- **PJND-aware pair-weighting** (Gaussian boundary, gap anchor):
  CID22 +0.023 but KonJND collapsed −0.68 to −0.81. Lever real,
  pointed wrong direction. *2026-05-18*
- **V_20a IW + ext + transforms**: TID PWRC 0.9822 best ever seen but
  CID22 SROCC 0.4632 catastrophic. Wins on the "right metrics" don't
  rescue the SROCC-on-ssim2-trained-corpus bias. *2026-05-15*
- **V_20b Su 2023 contrastive pre-train**: Won KADID + TID (every
  metric), lost CID22 (every metric). FRIQUEE 2017 caveat:
  synth pre-train → authentic-distortion transfer fails. *2026-05-15*
- **dssim co-training (cycle 7)**: All 5 dssim-weighted variants
  regressed CID22 by 0.04–0.07. *2026-04*

---

## Process — when to ship to which trail

1. **Train + eval on bake_verdict** against all 5 corpora.
2. **Run bake_compare vs both trail ships** with 1000 bootstrap.
3. **Apply both gates.** Update the candidate matrix above with the
   result.
4. If passes ONE gate: ship to THAT trail. Update
   `PreviewV0_5Balanced` or `PreviewV0_5Compression` in
   `zensim/src/profile.rs`. **Don't bump the crate version** (per
   user 2026-05-18: "we don't want crate bumps every time we get a
   nice bake").
5. If passes BOTH gates: ship to both trails (rare — would require a
   strict Pareto improvement).
6. If passes NEITHER: add a row to the candidate matrix with the
   failure mode and move on.

---

## Why two trails, not one

The single-trail experiment over 2026-05-15 to 2026-05-18 mapped
the V_24 architectural frontier and confirmed a **structural
tradeoff**: bakes that win compression-corpora SROCC lose
synthetic-distortion SROCC and vice versa. No single bake within the
228-/300-/372-feature runtime can Pareto-dominate the balanced
ship — the feature space doesn't support it.

The compression trail unlocks the compression-specialist bake that
the balanced trail's noise-strict gate vetoes. CLAUDE.md established
the priority: "Imageflow and related work is web-focused, not
archival." Two trails make that explicit.

The balanced ship is preserved so non-compression callers
(saliency-aware crop, generic perceptual diff for non-codec
distortions) don't regress on their own benchmarks.

---

## See also

- `zensim/CLAUDE.md` — methodology, statistical rigor, the
  Mohammadi 2025 panel
- `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md` § A.9 — the decisive rule
- `benchmarks/v22_mix_LARGE_iwssim_methodology_2026-05-18.md` —
  balanced ship methodology
- `/home/lilith/work/zen/zensim--372feat/benchmarks/v22_372feat_methodology_2026-05-18.md`
  — 372feat methodology + 5-seed CI
- `/mnt/v/zen/zensim-eval/v24_persample_alpha_2026-05-18/` —
  per-sample-α baseline + verdicts (compression-trail candidate
  blocked on runtime)
