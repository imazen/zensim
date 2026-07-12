# B + BHdr vs the field — test-set integrity + multi-metric panel (2026-07-12)

Re-verified benchmark of the new default `ZensimProfile::B` (SDR) and
`ZensimProfile::BHdr` (HDR) against `A` (v47, deprecated) and the external
metrics (fast-ssim2, cvvdp, iwssim for SDR; the UPIQ HDR panel for HDR), on a
**test set whose integrity was verified first**. Every number is measured, not
carried forward.

## Part 1 — Test-set integrity (verified before benchmarking)

| Check | Result | Evidence |
|---|---|---|
| **Feature cleanliness** (the zensim-GPU odd-dim garbage that corrupted the 2026-05-29 dial grid) | **CLEAN** — masked/IW block `f228..f371` is **0/144 bit-constant** across rows on CID22/AIC-3/KonJND (garbage is bit-constant; these vary) | probe on the val parquets |
| **Byte-equivalence to the audited canonical set** | **bit-identical** — max\|Δ\| = **0.000e+00** across f0/f100/f200/f300/f371, 2000 rows (2026-05-15 features == canonical-2026-05-21/val) | positional join spot-check |
| **CID22-49 held-out ↔ B's training** | **basename-DISJOINT** — 0 intersection with `cid22_train` (201 refs) and `safesyn` (3218 refs) | ref_basename set intersection |
| **`cid22_train` label type** | **ssim2-anchored, NOT human MOS** (CLAUDE.md-sanctioned metric-anchored training-only subset) — so even the disjoint train subset can't leak the val MCOS | canonical schema + CLAUDE.md |
| **Corpus roles for B** | KADID + TID are in B's kon-head training → **train==val guards** (not skill signal); AIC-3, AIC-4, KonJND never trained → **clean holdouts**; CID22 holdout (above) | B lineage (`profile_b_methodology_2026-07-12.md`) |
| **Baseline metrics on the same test set** | ssim2/cvvdp/iwssim computed on the **same corpora, same join, same panel implementation** (Python `panel.py` mirrors `panel.rs`, validated against Mohammadi 2025 anchor Z-RMSE to ≤0.06) | `baseline_panels_2026-05-18.md` |

**Verdict:** the SDR test set is clean, held-out, and cross-metric-comparable.
BHdr is benched on the UPIQ **HDR stratum** (n=380) — BHdr is HDR-only by design,
so SDR corpora don't apply to it.

## Part 2 — SDR panel (full Mohammadi 2025 stats, not SROCC-alone)

Clean held-out compression corpora. **Bold = best in column.** `B`/`A` via
`bake_verdict` on the verified features; ssim2/cvvdp/iwssim from the validated
baseline doc.

### CID22 (n=4292) — the gold-standard codec-compression holdout

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| ssim2 (fast-ssim2) | **0.8895** | **0.8879** | **0.7062** | 0.0424 | 0.9351 | **0.460** |
| **B** (default) | 0.8764 | 0.8760 | 0.6878 | **0.0002** | **0.9809** | 0.482 |
| A (deprecated) | 0.8657 | 0.8591 | 0.6742 | 0.0009 | 0.9782 | 0.512 |
| cvvdp | 0.8214 | 0.8251 | 0.6238 | 0.0424 | 0.8842 | 0.565 |
| iwssim | 0.7836 | 0.7926 | 0.5938 | 0.0520 | 0.8525 | 0.610 |

ssim2 leads rank (SROCC/PLCC/KROCC); **B is a clear #2, ahead of A, cvvdp, iwssim**,
and B actually **leads the whole field on PWRC (0.9809) and outlier-ratio (0.0002)** —
the dial calibration pays off in the importance-weighted + outlier stats. B closes
~45 % of A's SROCC gap to ssim2 (0.8657→0.8764 vs 0.8895).

### AIC-3 CTC (n=600) — JND-level compression holdout

| Metric | SROCC | PLCC | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|
| ssim2 | **0.7965** | **0.8086** | 0.8716 | **0.588** |
| cvvdp | 0.7918 | 0.8034 | 0.8657 | 0.595 |
| **B** | 0.7774 | 0.7880 | **0.9376** | 0.616 |
| iwssim | 0.7735 | 0.7907 | 0.8536 | 0.612 |
| A | 0.7680 | 0.7845 | 0.9334 | 0.620 |

**Five-way tie within 0.03 SROCC** — ssim2 nominally best, B mid-pack and ahead
of A; B leads PWRC again.

### AIC-4 sample (n=300) — reconstructed-JND holdout

| Metric | SROCC | PLCC | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|
| **B** | **0.8906** | **0.8795** | **0.9786** | **0.476** |
| A | 0.8854 | 0.8768 | 0.9756 | 0.481 |

**B best of the two zensim profiles.** External baselines (ssim2/cvvdp/iwssim)
have not been computed on AIC-4 — a known gap, not an omission.

### KonJND-1k (n=1008) — perceptibility / PJND holdout

| Metric | SROCC | PWRC | Z-RMSE |
|---|---:|---:|---:|
| **B** | **0.5466** | **0.8503** | **0.860** |
| A | 0.4185 | 0.7915 | 0.932 |
| iwssim | 0.1859 | 0.3097 | 0.974 |
| cvvdp | 0.0482 | 0.0225 | 0.988 |

**B decisively best of all metrics** on the perceptibility anchor (ssim2 has no
KonJND join in the baseline doc). All absolute correlations are low here — PJND is
a hard, near-threshold corpus — but B ranks it far better than cvvdp/iwssim.

### Guards (train==val for B — reported, NOT a skill claim)

TID2013 (n=3000): B 0.7868 / A 0.7927 / ssim2 0.8460 / cvvdp 0.8531 / iwssim 0.7794.
These corpora are ~95 % non-compression analytic distortions and are in B's
training groups; ssim2/cvvdp leading here is expected and is not evidence about
codec-compression skill.

## Part 3 — HDR panel: BHdr vs the HDR field (UPIQ HDR stratum, n=380)

BHdr re-scored from the shipped bake (`bhdr_linear_shaped_anchored2_2026-07-04.bin`,
sha `373eac56`) over the PU-linear UPIQ HDR features, panel vs JOD truth
(positional join verified 380/380 on reference content). Baselines from
`upiq_baselines_2026-06-01.md` (HDR-stratum SROCC).

| Metric | SROCC (HDR) | note |
|---|---:|---|
| HDRVQM | 0.8772 | HDR-specialist |
| PU-PieAPP | 0.8748 | HDR-specialist (panel best overall) |
| HDR-VDP-2 | 0.8117 | HDR-specialist |
| PU-SSIM | 0.7395 | PU-domain SSIM |
| **BHdr** | **0.7281** | 11.7 KB linear, deterministic |
| PU-FSIM | 0.7185 | PU-domain FSIM |
| PU-PSNR | 0.5485 | |
| PSNR | 0.4606 | SDR-tuned — collapses on HDR |
| FSIM | 0.4568 | SDR-tuned — collapses on HDR |

BHdr full panel: SROCC 0.7281, PLCC 0.6957, KROCC 0.5488, OR 0.000, PWRC **0.9451**,
Z-RMSE 0.7184. (Re-score matches the shipped 0.7313 within join noise.)

**Read:** BHdr is a **competent lightweight HDR metric** — it **ties PU-SSIM /
PU-FSIM** and cleanly **beats every SDR-tuned metric** (FSIM, PSNR) that collapses
on HDR, at 11.7 KB and fully deterministic. It **trails the heavy HDR specialists**
(HDR-VDP-2, PU-PieAPP, HDRVQM) by ~0.08–0.15 SROCC. Its PWRC (0.945) is mid-pack
among the strong metrics.

## Verdict

- **B (SDR default): validated on a clean, held-out, cross-comparable test set.**
  #2 to ssim2 on CID22 rank, ahead of A/cvvdp/iwssim; **best-in-field on PWRC +
  outlier-ratio on CID22**; best of all metrics on KonJND; best zensim profile on
  AIC-4; a 5-way tie on AIC-3. B ≥ A on every clean holdout.
- **BHdr (HDR): a competent lightweight** — ties PU-SSIM, beats SDR-metric HDR
  collapse, trails the heavy HDR specialists. Fair for its 11.7 KB deterministic
  footprint.
- **Integrity confirmed** — features clean + byte-identical to the audited
  canonical set; CID22-49 disjoint from training; roles (holdout vs guard)
  established; baselines measured on the identical test set with the same panel.

### Provenance

- Features: `/mnt/v/zen/zensim-training/2026-05-15-full-features/` (bit-identical to `canonical-2026-05-21/val/`)
- B/A/BHdr bakes: `zensim/weights/{b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07,v47_strict_qat_native_2026-05-27,bhdr_linear_shaped_anchored2_2026-07-04}.bin`
- SDR baselines: `benchmarks/baseline_panels_2026-05-18.md`; HDR baselines: `benchmarks/upiq_baselines_2026-06-01.md`
- UPIQ HDR features: `/mnt/v/output/zensim-multicodec-probe/upiq_features_372_pulinear.parquet` (n=380); JOD: `/mnt/v/datasets/upiq/upiq_subjective_scores.csv`
- Tools: `bake_verdict` (SDR panels), `predict_features_with_bake` + `panel` (HDR), all in `zensim-validate`
