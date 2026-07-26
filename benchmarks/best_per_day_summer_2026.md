# Best model per calendar day — summer 2026 retrospective (2026-05-01 .. 2026-07-25)

**What this is.** One row per calendar day that ran a zensim model/bake experiment,
naming the **best model of that day**, its **verified bake path on disk**, the recipe
reference, the headline metrics claimed that day, and one line on why it won / what it
tested. Built 2026-07-26 by reading all 325 dated `benchmarks/*2026-0[5-7]*.md` docs
(one experiment-day each) and cross-referencing against **1,565 `.bin` files** across every
bake root. Machine-readable twin: `/mnt/v/output/zensim/reports/best_per_day.json`.

**How "best of day" is chosen.** Primary tiebreak is **CID22 SROCC** (the gold human-MOS
holdout, higher = better, validation-only forever). When CID22 is not the day's axis — a
dial-tuning day, a steer/coherence day, an ssim2-north-star day — the day's own stated
winner is used and the axis is noted. Falsified / reverted / no-ship days record the day's
best *candidate* plus its verdict.

## ⚠ Integrity caveats (read before citing any number)

1. **Eval regimes differ across eras — compare within an era.** May used a 6-corpus panel
   (KADID/TID/CID22/KonJND/AIC); late-May added the quarantined dial grid + Mohammadi
   10-band; July added FR holdouts (LIVE-R2/CSIQ/PIPAL) + HF-near-lossless + imazen-26.
   The same bake reads differently across panels — e.g. shipped **B is CID22 0.876 on the
   07-18 scorecard and 0.882 on the 07-23 720-regime scoreboard**. A CID22 of 0.8934 in
   May (V0_18, vs-ssim2 eval) is *not* directly comparable to 0.8939 in July (winner_dial,
   FR-holdout panel).
2. **The auto-generated `<bake>.verdict.md` sidecars lie for wide bakes.** Next to any
   504/720-feature bake they score against a stale **372-only** parquet (zero-padded),
   reporting wrong CID22 (e.g. `ebothg_m504` sidecar says 0.2955, the doc says 0.884). The
   benchmark `.md` doc numbers are authoritative for those bakes — never the sidecar.
3. **Historical bakes live in four places.** Not just `/mnt/v/output/zensim/bakes/`:
   `/mnt/v/zen/zensim-eval/` (491 May-era ship/experiment bakes),
   `zensim/zensim-experimental/weights/` (the V0_x historical ships),
   git-tracked `zensim/benchmarks/*.bin` (mtime `2026-06-18` = a bulk re-touch, **not**
   the creation date), and `/mnt/v/output/zensim-multicodec-probe/`. Every path below was
   verified to exist.

## Summer champions

| role | model | verified bake | headline |
|---|---|---|---|
| **Best CID22 / rank (SDR)** | **winner_dial** (`Ebothg_hfgain_winsor_dial`) | `/mnt/v/output/zensim/corr-lq/Ebothg_hfgain_winsor_dial.bin` | CID22 **0.894**, LIVE 0.960, CSIQ 0.958, PIPAL 0.624, dial 0.980, steer M3 0.759 |
| **Best HF-near-lossless + best all-around dial** | **Ebothg_scr0.5_dial** | `/mnt/v/output/zensim/screen-retrain-2026-07-18/Ebothg_scr0.5_dial.bin` | HF-NL **0.712** (best ever), CID22 0.879, nonphoto 0.906, LIVE 0.959, dial **0.985** |
| **Best KonJND (any eval)** | **cl_tfm** (corruption+LQ MLP) | `/mnt/v/output/zensim/corr-lq/cl_tfm.bin` | KonJND **0.761** — FIRST to clear the G5 0.70 floor; corruption 100%, CID22 0.883; HF-NL dead (~0) |
| **Best KonJND (tiny/linear, May-era)** | **v02-bvls NO-shaping** | `/mnt/v/output/zensim/bakes/v02_bvls_NO_shaping_2026-05-28.bin` | KonJND 0.594, CID22 0.824, 8.6 KB / 86 active |
| **Best diffmap-coherent / steer (SDR)** | **ADD156** (`ADD156_safesyn_only_raw_lasso`) | `/mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin` | steer M3 **0.849** (best), CID22 0.863, exact fixed gradient, **3.6 KB** |
| **Best coherent-regime RANK (ship direction)** | **foldMLP+bigcodec+KADIS (E-K5)** | `/mnt/v/output/zensim/bakes/p1kadis/foldmlp_bigcodec_kadis_720.bin` | CID22 **0.8713** on the foldable (no iw/masked) regime; closes most of the gap to non-coherent 0.879; single-seed, M3 unmeasured |
| **SHIPPED default (`ZensimProfile::B`)** | `b_sdr_linear_cid80_inclwinsor_dense_dial` | `zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` | CID22 0.876, KonJND **0.547** + HF-NL **0.614** (best of ship candidates), dial 0.979 |
| **SHIPPED Profile A (deprecated 07-12)** | `v47-strict-QAT-native` | `zensim/weights/v47_strict_qat_native_2026-05-27.bin` | CID22 0.8657, KonJND 0.418, identity 97.7 exact, native QAT packing |
| **SHIPPED HDR (`ZensimProfile::BHdr`)** | `bhdr_linear_shaped_cvvdpmix` | `zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin` | UPIQ 0.7536, steer-mass 0.435 (shaped lineage = a steering dead-end per 07-18 audit) |

**The through-line.** May was a fast MLP ship cadence (V0_4 → V0_18 → V_22 → V_24/V6 →
V10 → v_tuner_v11 → V39) that peaked around CID22 0.89 but repeatedly tripped over
contamination and off-manifold axiom violations. Late-May pivoted to **monotone-by-
construction** correctness (v47-strict-QAT, the shipped Profile A). July restarted from a
**linear-projection** foundation (→ shipped Profile B, 07-07), then climbed back to
MLP-class rank with the **Ebothg** family (07-18) while confronting the two open product
problems that remain unsolved at the cutoff: **KonJND** (only cl_tfm's 0.761 clears the
floor, at the cost of a dead HF-NL) and **diffmap↔scalar coherence** (the top rankers use
non-spatializable pools; the coherent foldable+KADIS regime reaches only ~0.87 CID22).

## Per-day table

Legend: `verdict` = shipped / candidate / falsified / no-ship / eval / ablation.
`†` = bake verified on disk. Metrics are as-claimed in the source doc (see caveat 1 on
cross-era comparison).

| date | best model of the day | verified bake † | CID22 | other headline | verdict / why |
|---|---|---|---|---|---|
| 2026-05-01 | V0_4 MLP ssim2_holdout (→ V0_5) | `synthetic-v2/runs/v04_mlp_ssim2_holdout_20260501T045510.bin` | 0.889 (abs) | pairwise 0.816 | shipped as V0_5; best pairwise of 3 V0_4 bakes |
| 2026-05-10 | CHAMPION h192x128_ep300 (2-hidden MLP, TV10) | `benchmarks/h192x128_ep300_safesyn218k_kt_2026-05-10.bin` | 0.880 | kadid 0.931, non-mono 4.6% | day's winner (+0.042 agg); first past V0_2 smoothness floor; appears abandoned |
| 2026-05-11 | V0_8 (h128 TV15 KonJND-aligned) | `zensim-experimental/weights/archive/v0_8_tainted_2026-05-11.bin` | 0.895 (tainted; honest 0.891) | — | shipped, then **retroactively invalidated** (11.8% contamination) |
| 2026-05-12 | V0_16 (purged 144,791-row corpus) | `zensim-experimental/weights/archive/v0_16_2026-05-12.bin` | 0.892 | aic4 0.918, non-mono 2.3% | first HONEST clean-data ship; beats ssim2 on all 3 holdouts |
| 2026-05-13 | V0_18 (I8 rebake of V0_17 3-way concat) | `zensim-experimental/weights/v0_18_2026-05-13.bin` | **0.893** | −73.8% size | FIRST to clear the aspirational 0.8934; V0_19 single-MLP no-ship |
| 2026-05-14 | V_18 ship (revalidation) | `zensim-experimental/weights/v0_18_2026-05-13.bin` | 0.893 | repro 0.8912 | reconfirm; V0_18.1 no-ship, V0_19 reverted, V0_20a falsified |
| 2026-05-15 | D2 ensemble (V_18 × V_20-IS-calibrated, runtime mix) | `benchmarks/v0_20_input_shaping_seed1_calibrated_2026-05-15.bin` | 0.894 | +0.0003 vs V_18 | no-ship (marginal, not productized) |
| 2026-05-16 | V_22-IW v2 (log-target retrain) | `zensim-experimental/weights/v0_22_iw_v2_2026-05-16.bin` | 0.816 | kadid 0.948, tid 0.962 | shipped as **additive** PreviewV0_5; 3/4-corpora win |
| 2026-05-17 | V_22 mix cv40/iw60 (seed3 h128) | `zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_iw60_s3_h128.bin` | 0.888 | 5-seed 0.885; aic3 0.836 | ship-candidate; best of day's CVVDP family |
| 2026-05-18 | V_24 per-sample-α s4 (PreviewV0_5Compression) | `zensim-eval/v24_persample_alpha_2026-05-18/persample_seed4.bin` | 0.864 | konjnd 0.808 | shipped; ~10 same-day exps failed to beat it |
| 2026-05-19 | V6 ship cc4v6_w1p0_p0p30_s1 (TunerV2) | `zensim-eval/exp_cross_codec_v6_2026-05-19/cc4v6_w1p0_p0p30_s1.bin` | 0.877 | dial-mono 0.952 | shipped; only bake in v2→v8 lineage to pass every dial gate |
| 2026-05-20 | V10 trio: BalancedV3 + TunerV4 | `zensim-experimental/weights/v_balanced_v3_2026-05-20.bin` (+ `v_tuner_v10`) | 0.832 / 0.854 | balanced konjnd 0.893; tuner mono 96.4% | shipped (2, PCHIP dial-honesty recal); V11/12/13 falsified |
| 2026-05-21 | Recovery Phase-3 champion (h64 meanpol s5) | `exp_recovery_champion_2026-05-21/seeds/cc4_recovery_meanpol_h64_s5.bin` | 0.850 | konjnd 0.369 | **FALSIFIED** (KonJND < 0.793 gate); no PJND signal in train set |
| 2026-05-24 | Tuner v11 "a8" (PreviewV0_3 = latest()) | `zensim-experimental/weights/v_tuner_v11_2026-05-24.bin` | 0.860 | konjnd 0.285, mono 95.4% | shipped; first MLP+spline `latest()`; culmination of 8-attempt arc |
| 2026-05-25 | V39 (V32 ranknet + multi-band-anchor spline, seed17) | `zensim-experimental/weights/v39_v32plus_spline_seed17_2026-05-25.bin` | 0.879 | konjnd 0.420, aic4 0.905, zrmse 0.493 | shipped PreviewV0_3; V41 cvvdp-target failed at 0.66 |
| 2026-05-26 | V39 (defect accepted) + zensim_b_phone_oled | `…/v39_v32plus_spline_seed17…` (+ `bakes/zensim_b_phone_oled_seed17_2026-05-26.FINAL.bin`) | 0.879 | phone-oled cvvdp-SROCC 0.934, dial G1 1.00 | V39 ships despite off-manifold axiom violations (user accepted); v42/43/44/45/46 fail to displace |
| 2026-05-27 | **v47-strict-QAT-native (Profile A)** | `zensim/weights/v47_strict_qat_native_2026-05-27.bin` | 0.866 | konjnd 0.419, identity 97.7, mono 0.943 | **shipped Profile A**; fixed V39's identity/blur defect; native QAT packing |
| 2026-05-28 | v02_372feat Cell 5 (pure-linear, ssim2 target) — *runner-up: v02-bvls NO-shaping* | `bakes/v02_372feat_cell5_2026-05-28.bin` (+ `bakes/v02_bvls_NO_shaping_2026-05-28.bin`) | 0.870 / 0.824 | cell5 zrmse 0.494 4.8KB; **bvls konjnd 0.594** | candidate; bvls = KonJND champion of the tiny/linear family |
| 2026-05-29 | v02_372feat mix_ss_iw (50/50 ssim2+iwssim linear) | `bakes/v02_372feat_mix_ss_iw_2026-05-29.bin` | **0.874** | konjnd 0.464 | no-ship (CID22 specialist); Cell5 stays all-rounder; A is only bake passing both G3 sub-gates |
| 2026-05-30 | v47_plus_large (JXL-data augment) | `bakes/v47_plus_large_2026-05-30.bin` | 0.732 (Δ−0.134) | jxl-dial worse | **FALSIFIED**; near-lossless-only JXL data poisons; v47 stands |
| 2026-06-01 | Profile A via PU21/HDR front-end (eval) | `zensim/weights/v47_strict_qat_native_2026-05-27.bin` | — | hdr-band 0.694 (bar 0.740) | eval only; PU front-end validated, bar not cleared |
| 2026-06-13 | Profile A HDR baseline reconfirm (plan/eval) | `zensim/weights/v47_strict_qat_native_2026-05-27.bin` | — | hdr-band 0.693 | data-unblock + methodology; no HDR bake trained |
| 2026-06-30 | probe_372 (multi-codec ssim2 MLP) | `zensim-multicodec-probe/probe_372.bin` | 0.883 | konjnd 0.562, dial-mono 0.72 | no-ship (rank-only, **dial broken** — fails G3) |
| 2026-07-02 | hponly51_s7 (hardpair RankNet MLP) | *none-on-disk (fleet-trained, R2)* | 0.877 | konjnd 0.318 | candidate; first cell to beat A's CID22 via hardpair mining |
| 2026-07-03 | lp_ens-S5noguard (deterministic linear ensemble) | `zensim-multicodec-probe/linear-probe/bakes/lp_ens-S5noguard-tau0-f16.bin` | 0.879 | dial-mono 0.960 | candidate; beats every MLP finalist, no seed-collapse; ens-Pline-cid80 → Profile B next day |
| 2026-07-04 | Profile B v1 + BHdr v1 (both SHIPPED as new enum variants) | `zensim/weights/b_sdr_linear_cid80_anchored_2026-07-04.bin` (+ `bhdr_linear_shaped_anchored2`) | 0.873 | konjnd 0.544; bhdr upiq 0.731 | shipped; first day `ZensimProfile::B`/`BHdr` exist |
| 2026-07-05 | b_sdr_linear_cid80_dense_dial (B, winsor + dense-dial fix) | `zensim/weights/b_sdr_linear_cid80_dense_dial_2026-07-05.bin` | 0.876 | konjnd 0.547, dial pass | shipped; fixes raw −1131 tail + dead-zone; A↔B knob trade noted |
| 2026-07-07 | **b_sdr_linear_cid80_inclwinsor_dense_dial (THE shipped B)** | `zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` | 0.876 | konjnd 0.547, HF-NL dial 96.1 | **shipped default**; fixes near-lossless dial pinning at 0 MOS cost |
| 2026-07-11 | B as real-encoder dial (validation) | `…/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` | — | \|ρ\| 0.953, MOS-resid ±6.3, η² 0.582 (best-in-field, n=678k) | validation; B = best per-image dial |
| 2026-07-12 | BHdr v2 (bhdr_linear_shaped_cvvdpmix) — promoted then audited | `zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin` | 0.845 (OOD) | upiq 0.754 | shipped BHdr; same-day audit downgraded UPIQ significance (WY p=0.221) |
| 2026-07-14 | B-negatives dial-unblock (rank-identical to B) | `reports/b_negatives/b_sdr_linear_cid80_ssim2anchored_dense_dial_2026-07-14.bin` | 0.876 | negatives 52/2016 (was 0) | candidate; day's 5 BHdr-retrain experiments all **falsified** |
| 2026-07-15 | mlp_2L_diverse_H128 (2-layer diverse MLP, "best output") | `reports/b_negatives/mlp_2L_diverse_H128_2026-07-15.bin` | 0.881 | nonphoto 0.950, konjnd 0.509, dial 0.968 | candidate (user-gated); 2nd hidden layer breaks CID22↔nonphoto trade |
| 2026-07-16 | SPLIT: depth_v2_s13 (best ranker, dial broken) + k24_champion (best dial) | `depth-iter/depth_v2_s13.bin` (+ `minmax-sweep/k24_champion.bin`) | 0.888 / 0.831 | depth nonphoto 0.962 (dial 0.55 FAIL); minmax dial 0.979 PASS + imazen26 0.880 | rank-sibling / dial-candidate; min-max = first monotone-by-construction dial above A on ssim2 |
| 2026-07-17 | **cl_tfm (corruption+LQ MLP, s13)** | `corr-lq/cl_tfm.bin` | 0.883 | **konjnd 0.761** (first past G5 0.70), corruption 100%, HF-NL ~0 | candidate; beats B on CID22 + corruption + KonJND; HF-NL architecturally dead |
| 2026-07-18 | **winner_dial** (`Ebothg_hfgain_winsor_dial`) + Ebothg_scr0.5_dial + ADD156 | `corr-lq/Ebothg_hfgain_winsor_dial.bin` (+ `screen-retrain-2026-07-18/Ebothg_scr0.5_dial.bin`, `corr-lq/ADD156_safesyn_only_raw_lasso.bin`) | **0.894** / 0.879 / 0.863 | winner: LIVE 0.960 CSIQ 0.958 steer 0.759; scr0.5: **HF-NL 0.712**; ADD156: **steer 0.849** | ship candidates (user-gated); five-gate scorecard climax; KonJND −0.21 vs B is the stated trade |
| 2026-07-19 | dec_extlumacoh (coherence-maxed append-only) | `v2-ab-2026-07-19/dec_extlumacoh.bin` | 0.650 | coherence 1.00 (v1 baseline 0.53) | ablation; primary A/B **KILLed** naive full-v2 swap (v1 0.618 > v2 0.576); deltas = seed noise |
| 2026-07-23 | ebothg-504 (basic-156 ++ v2-348 MLP + bigcodec) | `bakes/top5/ebothg_m504.bin` | 0.884 | imazen26 0.930, dial 0.960 | candidate (gate-passing); winner-504 (0.892) tops raw CID22 but FAILS dial; bigcodec stabilises the v2 dial |
| 2026-07-24 | ideal_p0p2_L0p003_F0p005 (foldable BVLS, dial-mono reg) | `signedpow-clean-2026-07-24/ideal_p0p2_L0p003_F0p005.bin` | 0.787 | dial 0.958, coherence 0.58 | resolution-pick (disputed); "price of clean"; same-day fold_extraction contradicts its foldability |
| 2026-07-25 | **foldMLP+bigcodec+KADIS (E-K5)** — coherent-regime frontier | `bakes/p1kadis/foldmlp_bigcodec_kadis_720.bin` | 0.8713 | konjnd 0.285, csiq 0.863, live 0.838 | frontier candidate; best CID22 of the diffmap-coherent regime; linear sibling foldcanon+KADIS 0.815 |

**Non-model days** (infra / perf / data only, no bake): 2026-06-10 (PU21 SIMD bench),
2026-07-06 (jxl-encoder fix + HF corpus), 2026-07-10 (winsor-corpus analysis),
2026-07-13 (kadis-hdr corpus gen + GPU fleet), 2026-07-20 (tower-vs-local perf + backfill),
2026-07-21 (v2 ref-reuse SIMD perf), 2026-07-22 (corpus consolidation + eval backfill).

## Coverage

- **39 distinct model-experiment days** covered (42 per-day entries incl. the 07-18 trio
  and 05-28 pair); **7 non-model days** noted; range 2026-05-01 → 2026-07-25.
- **41 of 42** per-day bakes verified present on disk; the 1 exception (07-02 `hponly51_s7`)
  was fleet-trained and lives only in R2, flagged `none-on-disk`.
- **All 9 champion bakes verified on disk.** 1,565 `.bin` files scanned across 6 roots.
- Gaps in the doc record: **2026-05-22 / 05-23** have no benchmark doc (idle or covered
  elsewhere); several June weeks are sparse (the HDR/PU + multicodec-probe stretch).

## Provenance

Each pick's `recipe_ref` names the source doc and, where present, the `.spec.json` (exact
`argv`) or `weights/manifests/*.toml`. The `.spec.json` sidecars carry the reproduction
command for the modern (07-18+, 07-23, 07-25) bakes; the May-era ships trace through
`zensim-experimental/weights/` + their methodology docs. Full machine-readable record with
per-metric claims: `/mnt/v/output/zensim/reports/best_per_day.json`.
