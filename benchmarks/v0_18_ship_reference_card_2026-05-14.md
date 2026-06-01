# V_18 ship reference card — canonical V_X comparison baseline

**Bake**: `zensim-experimental/weights/v0_18_2026-05-13.bin`
  (md5 `c94e93607390d0b6704e95f3851d421e`, 93,064 bytes)
**Date**: 2026-05-14 eve
**Full eval log**: `benchmarks/v0_18_ship_reference_card_2026-05-14.log`
  (264 lines, full-stat panel + 10-band + step-5 + MRR + Wilcoxon
  per CLAUDE.md rigor mandate)
**Per-pair CSV**: previously committed at
  `benchmarks/per_pair_v0_18_aic3_anchor_2026-05-14.csv` (AIC-3 only).
  KADID+TID+CID22 per-pair lives at
  `/tmp/per_pair_v0_18_ship_2026-05-14.csv` — too large (2.8 MB)
  to commit; regenerate via the command at the bottom of this card.

## Aggregate SROCC vs fast-ssim2 (gold-standard baseline)

| Corpus     | n     | V_18 ship | fast-ssim2 | Δ vs ssim2 | butteraugli |
|---|---:|---:|---:|---:|---:|
| KADIK10k   | 10125 | **0.9427** | 0.8133 | **+0.1294** | 0.6062 |
| TID2013    |  3000 | **0.9526** | 0.8460 | **+0.1066** | 0.6696 |
| CID22      |  4292 | **0.8933** | 0.8895 | **+0.0038** | 0.7412 |
| AIC-3 Anchor (250) | — | **0.9149** | — | — | — |

V_18 ship is the shipped weight as of 2026-05-13 and the **honest
baseline** (no training contamination per the 2026-05-14 zero-real-
contamination correction).

## CID22 10-band SROCC — where V_18 wins and where it LOSES

CID22 is the cross-band validation gold standard. Per-band SROCC:

| 10-band | range | n | V_18 ship | fast-ssim2 | Δ vs ssim2 |
|---|---|---:|---:|---:|---:|
| B0..B2 | [0, 30) | 0–1 | n/a | n/a | n/a (no labels) |
| **B3** | [30, 40) | **57** | **0.0246** | **0.1335** | **−0.109 LOSS** |
| B4 | [40, 50) | 266 | 0.3029 | 0.2888 | +0.014 |
| B5 | [50, 60) | 615 | 0.3891 | 0.3888 | +0.000 |
| B6 | [60, 70) | 836 | 0.3943 | 0.4173 | −0.023 |
| B7 | [70, 80) | 1092 | 0.3936 | 0.3974 | −0.004 |
| B8 | [80, 90) | 1382 | 0.5127 | 0.5006 | +0.012 |
| B9 | [90, 100] | 43 | 0.1545 | 0.1121 | +0.042 |

**The known weakness of V_18 ship is CID22 B3 [30, 40)**: V_18
under-performs fast-ssim2 by **−0.109 SROCC** on the 57 lowest-quality
pairs in CID22. That's the band where compression product decisions
live (heavy compression — files where every byte matters).

B0..B2 are empty on CID22 (the corpus doesn't include
artifact-saturated reference-quality images).

This is **the gap V0_20b distortion-manifold pre-training should
close**.

## AIC-3 per-codec breakdown (n=50 per codec)

Surfaces which codecs V_18 handles well vs poorly:

| Codec | n | V_18 SROCC | V_2 SROCC | fast-ssim2 SROCC | Δ vs ssim2 | V_18 AT-1-JND median |
|---|---:|---:|---:|---:|---:|---:|
| AVIF       | 50 | 0.9500 | 0.9398 | 0.9358 | +0.014 | 75.93 |
| JPEG-1     | 50 | 0.9536 | 0.9518 | **0.9629** | **−0.009** ⚠ | 72.93 |
| JPEG-2000  | 50 | **0.9147** ⚠ | 0.8949 | 0.9023 | +0.012 | 77.62 |
| JPEGXL     | 50 | **0.9833** ✓ | 0.9831 | 0.9663 | +0.017 | 71.80 |
| VVC        | 50 | 0.9201 | 0.9231 | 0.8945 | +0.026 | 76.88 |
| **OVERALL** | 250 | **0.9149** | 0.8968 | 0.8943 | +0.021 | 73.92 |

**Findings**:
- **V_18 beats fast-ssim2 on 4 of 5 codecs.** The exception is
  legacy **JPEG-1** where V_18 underperforms by 0.009 SROCC. This
  is consistent with V_18's known CID22 weakness on the low-quality
  legacy-JPEG band.
- **Weakest absolute SROCC: JPEG-2000 (0.9147) and VVC (0.9201).**
  Both lifted vs ssim2 but lowest overall. These are also the
  codecs farthest from the synthetic training distribution
  (zenjpeg / WebP / JXL dominated training).
- **Strongest: JPEGXL (0.9833).** Consistent with JXL being heavily
  represented in the V_18 training corpus.
- **AT-1-JND landing varies 71.80–77.62 across codecs** (5.82-point
  spread). The V_18 dial is **not fully codec-agnostic at JND** —
  same subjective "just noticeable" lands at different V_X scores
  depending on which codec produced the distortion. Codec-specific
  calibration is a V0_20d Option-C variant worth considering.

The +0.026 lift on VVC and +0.021 overall vs ssim2 on a held-out
corpus that V_18 has never seen during training is honest
generalization signal.

## Per-corpus per-band performance vs fast-ssim2

KADID and TID are auxiliary integrity corpora (not compression-tuned
per CLAUDE.md training-goal #2). V_18 lifts both significantly:

- KADID 10-band: V_18 0.0..0.97 across B0..B9 (mostly +0.05 to +0.30
  over ssim2).
- TID 10-band: V_18 0.0..0.98 across B0..B9 (mostly +0.05 to +0.20
  over ssim2).

See `benchmarks/v0_18_ship_reference_card_2026-05-14.log` for the
full per-corpus 10-band tables + 20-bin step-5 tables + statistical
significance panels (MRR + Wilcoxon vs V_18).

## What this card is for

When evaluating a V_20+ candidate:
- **Aggregate must match-or-exceed V_18 ship on CID22 (0.8933)** —
  the shipping bar per CLAUDE.md goal #1.
- **CID22 B3 [30, 40) is THE band to lift** — V_18 ships at 0.0246
  there, fast-ssim2 at 0.1335. Any new candidate should at least
  reach 0.10 here.
- All other CID22 bands within ±0.02 of V_18 (no regression).
- Auxiliary corpora (KADID/TID): no >0.005 aggregate regression.
- AIC-3 (held-out): cross-corpus SROCC ≥ 0.90.

A candidate that matches V_18 elsewhere but reaches CID22 B3 [30, 40)
SROCC ≥ 0.13 (fast-ssim2 level) is a **legitimate ship candidate**.

## Reproduction

```sh
cargo run --release -p zensim-bench --example dataset_metric_baseline -- \
  --kadid /mnt/v/dataset/kadid10k \
  --tid /mnt/v/dataset/tid2013 \
  --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
  --v04-bake zensim-experimental/weights/v0_18_2026-05-13.bin \
  --max-pairs 50000 \
  --per-pair-output /tmp/per_pair_v0_18_ship_<date>.csv \
  2>&1 | tee benchmarks/v0_18_ship_reference_card_<date>.log
```

Runtime: ~4 min on water-cooled 7950X.

## V_18 ship limitations (honest accounting)

Per `benchmarks/v0_20a_path_a_falsification_2026-05-14.md` and
`benchmarks/v0_20a_ship_form_verification_2026-05-14.md`:

1. CID22 B3 [30, 40) SROCC 0.0246 — the worst V_18 band on the
   priority corpus. fast-ssim2 lands at 0.1335 there.
2. CID22 B6/B7 SROCC slightly under fast-ssim2 (−0.023, −0.004).
3. CID22 B9 [≥90] SROCC 0.1545 — visually-lossless tail, n=43
   (noisy). +0.176 lift available via 60/40 V_18+baseline_hq mix
   BUT trades aggregate −0.140 (deferred — wrong-direction trade
   per "B0..B5 priority").
4. V_18 outputs at AIC-3 1-JND = **73.92** (not 63) — calibrated
   to KonJND-1k anchor, not to AIC-3 JND. Disagreement is 11
   score units. See `benchmarks/v0_20d_jnd_calibration_design_2026-05-14.md`.
5. Architecture: 372→384→1 (3-way concat at 0.65/0.30/0.05 of
   base/cycle14-s1/cycle14-s42 trained on 144,791-row clean corpus).
   Capacity may be saturated for the IW feature signal per the
   V0_20a Path A falsification.

## Related artifacts

- Methodology: `benchmarks/v0_18_methodology_2026-05-13.md`
- Reproduction audit: `benchmarks/v0_18_repro_and_cross_corpus_analysis_2026-05-14.md`
- Falsifications: `benchmarks/v0_20a_path_a_falsification_2026-05-14.md` +
  `benchmarks/v0_20a_ship_form_verification_2026-05-14.md`
- V0_20d sanity: `benchmarks/v0_20d_jnd_calibration_design_2026-05-14.md` +
  `scripts/v_next/aic3_jnd_sanity.py`
- Future V_20+ priorities: `docs/v0_20_path_evaluation_2026-05-14.md`
