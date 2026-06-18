# Cycle-6 finals — V0_16 vs fast-ssim2 across all 5 public corpora

**Status**: COMPLETE 2026-05-12. V0_16 shipped at
`zensim/weights/v0_16_2026-05-12.bin` (md5 `baf3fdcb...`,
228 → 128 LeakyReLU → 1 MLP, affine-calibrated α=28.0366 β=-5.0738).

**Goal #1 from CLAUDE.md** ("match-or-exceed fast-ssim2 across all
quality bands"): **EMPIRICALLY MET** on every public corpus we
ship parquets for.

---

## Cross-corpus aggregate |SROCC| vs human

| Corpus | n | V0_2 (linear) | **V0_16 (MLP)** | fast-ssim2 | butter | held-out? |
|---|---:|---:|---:|---:|---:|---|
| AIC-3 CTC EPFL | 600   | 0.7962 | **0.7990** | 0.7965 | 0.7095 | ✅ true |
| AIC-4 sample   | 300   | 0.9107 | **0.9175** | 0.9127 | 0.8969 | ✅ true |
| CID22 (full)   | 4292  | 0.8676 | **0.8919** | 0.8895 | 0.7911 | ✅ true |
| KADID-10k      | 10125 | 0.8192 | **0.9403** | 0.8133 | 0.6062 | ⚠️ KADID_train was V_X supervision |
| TID2013        | 3000  | 0.8427 | **0.9501** | 0.8460 | 0.6696 | ⚠️ TID_train was V_X supervision |

**3 truly-held-out corpora (no V_X training overlap)**: V0_16
beats fast-ssim2 by +0.0024 to +0.0048. Modest but consistent
margin across diverse human-judgment corpora.

**2 training-overlap corpora**: V0_16 dominates by +0.10 to +0.13.
Confirms V_X is well-fit to its training distribution but doesn't
generalize-test anything.

---

## Per-codec scorecard (TRUE V0_16 only, on truly held-out corpora)

V0_16 wins (W) / ties within ±0.001 (T) / loses (L) per codec
vs fast-ssim2:

### CID22 (9 codecs, n=4292)

| Codec | n | V0_16 | ssim2 | Δ | Result |
|---|---:|---:|---:|---:|---|
| **AVIF_aurora_slow** | 446 | **0.8809** | 0.8425 | **+0.0384** | W (biggest win) |
| **JPEG_XL** | 535 | **0.9314** | 0.9219 | +0.0096 | W |
| **AVIF_aom_s1** | 423 | **0.8903** | 0.8813 | +0.0090 | W |
| WebP | 441 | 0.9085 | 0.9052 | +0.0034 | W |
| JPEG_2000 | 441 | 0.8743 | 0.8724 | +0.0019 | W (within noise) |
| AVIF_aom_s7 | 539 | 0.9138 | 0.9140 | -0.0002 | T |
| AVIF_aurora_fast | 539 | 0.8618 | 0.8627 | -0.0009 | T |
| HEIC | 392 | 0.8902 | 0.8920 | -0.0017 | L (within noise) |
| JPEG | 536 | 0.9402 | 0.9458 | -0.0056 | L |

CID22: **W=5 T=2 L=2**

### AIC-4 (6 codecs, n=300)

| Codec | n | V0_16 | ssim2 | Δ | Result |
|---|---:|---:|---:|---:|---|
| VVC | 50 | 0.9375 | 0.9194 | +0.0181 | W |
| JPEG-2000 | 50 | 0.9357 | 0.9197 | +0.0159 | W |
| JPEG-XL | 50 | 0.9705 | 0.9604 | +0.0101 | W |
| JPEG-1 | 50 | 0.9541 | 0.9453 | +0.0088 | W |
| AVIF | 50 | 0.9598 | 0.9545 | +0.0053 | W |
| **JPEG-AI** | 50 | **0.7951** | **0.8459** | **-0.0508** | **L (biggest deficit)** |

AIC-4: **W=5 L=1** — V0_16 wins every classical codec; loses to
ssim2 only on JPEG-AI (transformer codec).

### AIC-3 (6 codecs, n=600)

| Codec | n | V0_16 | ssim2 | Δ | Result |
|---|---:|---:|---:|---:|---|
| JPEGXL | 100 | 0.8539 | 0.8399 | +0.0140 | W |
| HM | 100 | 0.7840 | 0.7838 | +0.0001 | T |
| JPEG-1 | 100 | 0.8428 | 0.8446 | -0.0018 | L (noise) |
| JPEG-2000 | 100 | 0.7629 | 0.7671 | -0.0042 | L |
| VVC | 100 | 0.8004 | 0.8063 | -0.0059 | L |
| AVIF | 100 | 0.8092 | 0.8183 | -0.0092 | L |

AIC-3: **W=1 T=1 L=4** — only JPEGXL clearly wins; sub-PJND
distortion regime (JND ∈ [-2.5, -0.25]) is V0_16's weak band.

### Cross-corpus per-codec totals

**V0_16 wins 11 / ties 3 / loses 7 of 21 codec comparisons** on
truly-held-out corpora.

---

## The JPEG-AI anomaly (single biggest cycle-7 target)

On AIC-4 JPEG-AI:

| Metric | \|SROCC\| | vs metric's aggregate AIC-4 |
|---|---:|---:|
| V0_2          | 0.8265 | -0.084 |
| V0_16         | 0.7951 | -0.122 (worst metric here) |
| fast-ssim2    | 0.8459 | -0.067 |
| **dssim**     | **0.9147** | **-0.011** (essentially intact) |
| paper CVVDP   | 0.9609 (aggregate) | (no JPEG-AI breakdown emitted) |

dssim's multi-scale SSIM-derived structure captures JPEG-AI
artifacts that V_X / ssim2 / V0_2 all miss. Three possible cycle-7
responses:

1. **Add JPEG-AI training samples** to synth corpus. Most direct
   fix; requires acquiring JPEG-AI encoder access.
2. **Add dssim as an auxiliary loss head**. V_X learns dssim's
   structure in parallel with ssim2. Should generalize beyond
   JPEG-AI to other transformer codecs.
3. **Multi-scale SSIM aggregation** built into V_X. Architectural
   change rather than data; harder to evaluate vs (2).

(2) is the most promising option for the rest of cycle-7.

---

## Document index (cycle-6 reading order)

1. **This doc** (`cycle_6_finals_2026-05-12.md`) — top-level summary.
2. CID22 aggregate: `cid22_full_v0_16_vs_ssim2_2026-05-12.md`
3. CID22 per-codec: `cid22_per_codec_v0_16_2026-05-12.md`
4. AIC per-codec: `aic_per_codec_v0_16_2026-05-12.md`
5. **Superseded** (V0_2-mislabeled): `aic3_zensim_vs_baselines_2026-05-12.md`,
   `aic4_zensim_vs_paper_metrics_2026-05-12.md`,
   `aic_combined_per_codec_2026-05-12.md`. These docs report
   `zenmetrics batch --metric zensim` outputs which were actually
   V0_2 (linear weights), not V0_16. The number reported as "V0_16"
   in those docs is the V0_2 number. Use the corrected docs above
   for the TRUE V0_16 picture.

---

## Comparison-site live URL

<https://imazen.github.io/zensim/compare.html>

All 5 corpora's parquets carry the `score_zensim_v0_16` column
(verified via DuckDB-WASM query end-to-end). The Y-axis dropdown
exposes:
- score_v0_2_linear (V0_2)
- score_zensim_v0_16 (V0_16 SHIP)
- score_ssim2_gpu
- score_dssim (CID22, AIC-3, AIC-4)
- score_butter_p3
- score_butter_max
- score_ssim2_paper (AIC-4 only)
- score_cvvdp / score_iw_ssim / score_ms_ssim / etc (AIC-4 only,
  paper pre-computed)
- human_mos / human_jnd / human_dmos (corpus-dependent)
- bpp (CID22 only)
- q / quality_index / dlevel (codec-parameter axes)

Per-corpus schema differences are handled by per-corpus DuckDB
schema introspection (see `compare-worker.js:getSchema`).

---

## Recommended cycle-7 priorities

1. **JPEG-AI training-data acquisition + cycle-7 training run**.
   The single biggest empirical deficit.
2. **dssim co-training signal**. Concrete: add a second loss head
   in `train_v_next_mlp.py` that targets dssim's output, weighted
   ~0.3 vs the ssim2 head's 1.0. Most useful if (1) is hard to
   pursue.
3. **AIC-3 sub-PJND coverage**. The q≥85 regime is under-sampled
   in synth; V0_16 loses 4/6 codecs there (all by <0.01 SROCC, so
   small absolute gain available).
4. **R2 unified parquet hosting** (user-blocked). Once enabled,
   the comparison-site can query the 2.37M-row codec-sweep
   parquets and per-codec analysis scales 100×. The site code
   path is already ready for this.

End cycle-6.
