# Provenance + reproducibility — best results of the 2026-07-02..04 campaign

Canonical record for every ship-grade artifact the strategy/linear/HDR
campaign produced. Every claim below is committed evidence; every artifact is
rebuildable from pinned inputs. Repo commits referenced are on
`imazen/zensim@main` unless marked zenmetrics.

## The candidates (panel |SROCC| vs human; same harness, references reproduce
## their known numbers: A CID22 0.8657, ssim2 0.8894/0.4784/0.958)

| artifact | bytes | sha256 (16) | CID22 | AIC-3 | AIC-4 | KonJND | UPIQ | SDR25 | dial |
|---|--:|---|--:|--:|--:|--:|--:|--:|---|
| **Profile B (SDR) SHIPPED: b_sdr_linear_cid80_dense_dial** | 12,988 | `b78adb15b55d5c91` | 0.8763 | 0.7774 | 0.8906 | 0.5474 | 0.6846 | — | ALL gates PASS: inv 0.026, dead-zone 0.0005, mono 0.974, G-RANGE 0 extrapolating (winsor + dial-top-extend) |
| its winsor-only predecessor (archived) | 12,891 | `b92b0b7a384242d6` | 0.8763 | 0.7774 | 0.8906 | 0.5474 | 0.6846 | — | dead-zone 0.0563 FAIL (superseded) |
| its raw pre-winsor twin | 823 | `7b326ac56a05c240` | 0.8733 | 0.7775 | 0.8906 | 0.5439 | 0.6846 | — | f155 tail exposed (superseded) |
| CID22 record: lp_ens-S5noguard-tau0-f16 | 1,119 | `620b47e405289f26` | **0.8793** | 0.7821 | 0.8875 | 0.3497 | 0.6403 | — | mono 0.9602 |
| **Profile B-HDR: lp_hdr-lasso0.001-shaped-anchored2-f16** | 11,684 | `373eac56e7a07d6d` | 0.8347* | 0.7855* | 0.9022* | 0.3741* | **0.7313** | — | knots [25.9, 92.8] (data ceiling) |
| PJND sibling: lp_canonhdr15-bvls-raw-tau0.005-f16 | 3,738 | `1400fa2f86e2154d` | 0.8001 | 0.7225 | 0.8095 | **0.6696** | 0.6679 | — | mono 0.9315 |
| best MLP (SDR): w3_t1dro51_s31 | ~48K | `c2ffc04452a61de6` | 0.8708 | 0.8013 | 0.9154 | 0.3109 | 0.6594 | 0.9694 | mono 0.973+ |
| best MLP (HDR): w7_guard_s101 | ~67K | `9f0d6f3293cd7939` | 0.8639 | 0.7761 | 0.8861 | 0.3524 | 0.6798 | 0.9538 | — |

**Routing (2026-07-04):** `ZensimProfile::B` auto-routes to the BHdr weights
on the PU-linear (nits) entry points — typed dispatch on the declared input
domain, never pixel-value sniffing (rejected: threshold seams at 5-10pt
cross-model scatter). `BHdr` = explicit unrouted handle.

*HDR pick's SDR numbers are informational only — it is INVALID on SDR content
(rank 0.72 on the SDR dial grid, wild extrapolation); routing is mandatory.

## Reproducibility, per artifact class

### Linear fits + ensembles (all `lp_*`) — deterministic, byte-exact
- **Builder**: `scripts/v_next/linear_projections_2026-07-03.py` (fit/finalize/
  ensemble subcommands; Gram-matrix exact full-data solves — no SGD, no seed).
  Landed at zensim `87b3ee25`; ensembles + residual at `ee850a95`.
- **Determinism proof**: 44/44 refits byte-identical across a full pipeline
  re-run from parquet (fresh Gram accumulation); re-baked artifacts
  sha256-identical (`linear-probe/determinism_check.py`, bake_sha_run1.txt).
- **Fit inputs** (all pinned, local + R2):
  - SDR-champion target corpus: `hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet`
    (7,410 rows; cvvdp-mix target = 0.5·clip01(ssim2)+0.5·clip01((JOD−6)/4));
    built by `scripts/hdr/build_hdr_train_parquets.py --mix-target` from the
    v3 PU-linear datagen dirs (below).
  - canonical corpora: `/mnt/v/zen/zensim-training/canonical-2026-05-21/`
    (R2 `s3://zentrain/canonical-2026-05-21/`, shas in its `_MANIFEST.json`).
  - bigcodec mm6: `bigcodec_mm6_traindigits_2026-07-02.parquet` (1,559,919
    rows train / 914,418 val; 4-metric join; R2
    `s3://zentrain/strategy-fleet-2026-07-02/derived/`).
- **Fit stores**: `linear-probe/fits/*.npz` (w, bias, mu, sd, space) — the
  exact solved weights; bakes re-emit from these via the JSON pipeline
  (`zenpredict bake`), never hand-rolled bytes.
- **Ensembles**: convex weights over named heads, selected ONLY on
  training-legal val axes (documented per-ensemble in
  `benchmarks/linear_projections_2026-07-03.md`); cid80 = cid.80+kon.20,
  S5 = kad.30+hds.20+s3h.50.
- **Anchored dial siblings**: `scripts/v_next/shared_anchor_refit.py` (SDR:
  canonical `multiband_anchor_dial100.parquet`; HDR: v3-val at
  human_score×100) and `scripts/v_next/hdr_anchor_dense_refit.py`
  (anchored2: +1,760 train>0.85 rows, Q-Q top knots to the 92.8 data
  ceiling). Rank invariance verified: SROCC 1.000000 pre/post.
- **Winsor tail guard (SDR B final step)**: `scripts/v_next/winsorize_bake.py`
  attaches 372 `winsor_p99` transforms with fit-corpus (hdr_v3mix) [p0.1,p99.9]
  bounds to the anchored bake (fully-zero-constant f25/f64 floored to [0,1e-9]).
  Was an ad-hoc inline command through 2026-07-05; now committed + proven
  byte-identical to the shipped `b92b0b7a`.
- **Runnable end-to-end repro**: `scripts/reproduce_b.sh` chains
  ensemble→anchor→winsor and asserts byte-identity to shipped B (`b92b0b7a`)
  + CID22 0.8763 + G-RANGE PASS. Routing design (why B dispatches SDR/HDR):
  `docs/PROFILE_B_ROUTING_DESIGN_2026-07-05.md`.

### HDR corpus (the v3 PU-linear extraction)
- **Sources**: imazen-26 HDR subset — 76 sources × scales/crops = 1,140
  PQ-PNG renditions at `/mnt/v/output/imazen-26-hdr-grid-2026-06-14`.
- **Encodes**: zenjxl --hdr only. June grid (q 5..95, 7,980 cells,
  R2 `s3://codec-corpus/picker-sweep-2026-06-22/datagen-2026-06-23-hdr/`) +
  near-lossless top-up (q90..100, 9,120 cells, `datagen-2026-07-03-hdr-hq/`).
  zenjxl floors q<15 (q5≡q15 byte-identical) → dedup-by-content in the
  corpus builder.
- **Features**: PU-linear integrated path —
  `Zensim::compute_pu_linear_extended_features` (zensim `14d4140f`, width-
  parity + identity tests) via `zenmetrics score-pairs --hdr
  --hdr-features-pu-linear` (zenmetrics `6f591dd5`); extracted by 10 cx
  boxes (`hdr_score_fleet.sh`, runs v3june/v3hq, zero gate failures) into
  `s3://zentrain/hdr/runs/v3{june,hq}/`; merged by
  `scripts/hdr/merge_v3_shards.py`.
- **Targets**: ssim2 (omni inline) + cvvdp (7,980 June + 9,120 HQ, verified
  0% NaN) + butteraugli/dssim/iwssim sidecars (June + HQ complete).
- **Built corpora** (`build_hdr_train_parquets.py`, LSD origin splits, test
  digits {7,9} never touched): `hdr_zenjxl_v3_{train,val}digits_2026-07-03`
  = 7,410/3,900 rows, shas `31e08c70…`/`63688cb0…`; mix variant
  `hdr_zenjxl_v3mix_*` same rows, mix target.

### MLP references
- **t1dro51**: full kit at `scripts/reproduce_t1dro51.sh` (pinned trainer
  commit 78ec8e61, R2 inputs, 5 recorded bake shas, per-SIMD-tier caveat).
- **w7_guard_s101**: manifest `zensim/weights/manifests/w7_guard_s101.toml`
  (trainer-commit stamped; konjnd_anchor guard group; deterministic given
  ISA tier). Trained on the wide fan (box run, ctlfan machinery).

### Instruments (what makes the numbers trustworthy)
- **Collapse gate**: runcells post-train `bake_verdict` floors (CID22<0.75 ∨
  KonJND<0.20 → rc=9 + `<cell>-COLLAPSED` row). Justification: family-latent
  6.25% collapse floor (control fan, `05ae0210`); v3 amplification 43.75%.
- **Consistency harness**: zenmetrics
  `crates/zenmetrics-api/tests/hdr_sdr_consistency.rs` — SDR-range content
  through both paths; fixed-anchor seam 6.4pt, SDR-anchored ≤0.92pt.
- **Spline parity**: validate-side upper extrapolation capped at 100
  (zensim `5d4978db`) — matches the product runtime; dial p95 artifacts
  (321/504) were this bug.
- **Dial alignment**: shared-anchor refits give cross-model MAE 5.05pt in
  the mutual dial zone (40-95); boundary seam ≤0.92pt; out-of-domain =
  router territory (measured invalid, rank 0.72).
- **UPIQ harness**: `scripts/hdr/upiq_panel.py` (380 EXR pairs, JOD);
  SDR25: value-recovered permutation join (reconstruction
  `scripts/v_next/reconstruct_sdr25_jnd.py`, trap-verified semantics).

## Falsified this campaign (do not retry without new evidence)
MLP stabilization via target choice (w8: cvvdp-mix collapsed 2/2 while making
linear the champion), via selection guard (w7: same seeds collapse; steering
impossible — no healthy epoch exists), via seed selection-value ranking
(healthy s31 below collapsed seeds); SDR residual corrections at every λ
(v1 ssim2-target AND v2 mix-target — ensembles dominate the composed Pareto);
2-stage linear cascades; CSF-feature engineering (v39-era, re-confirmed);
input shaping outside pure-HDR; bigcodec mass in linear CID22 fits.

## Where everything lives
- Docs: this file; `benchmarks/linear_projections_2026-07-03.md` (all 55+
  bakes, ensembles, residual, attribution);
  `benchmarks/strategy_ablation_2026-07-02.md` (MLP campaign + collapse
  program); `docs/PLAN_HDR.md`; reports viewer
  `http://172.23.240.1:3300/zensim/reports/` (`2026-07-03_linear_vs_all` is
  the cross-metric page).
- Artifacts: `/mnt/v/output/zensim-multicodec-probe/linear-probe/`
  ({bakes,fits,residual}, panel_all.tsv, verdicts).
- R2: `s3://zentrain/strategy-fleet-2026-07-02/` (SDR inputs+bakes),
  `s3://zentrain/hdr/runs/v3*/` (HDR feature shards),
  `s3://codec-corpus/picker-sweep-2026-06-22/datagen-*hdr*/` (encodes).

## Profile B as a codec quality knob — gate measurements (2026-07-04)

**G1 SDR25: CLOSED — B 0.9574 vs A 0.9036** (n=50, value-recovered
permutation path; ssim2 = 0.9580). B beats A on 8/9 axes; UPIQ (−0.009)
is the sole A-favoring axis.

**G2 dial equivalence: MEASURED.**
- Spline: 18 knots, raw-x [−1.974, 1.138] (a real input range — the linear
  raw output is meaningful, unlike the MLP's 0.23-wide band), dial-y
  [0, 95.89]; above the top knot the product runtime extrapolates linearly
  capped at 100.
- Per-codec dial (115 ladders): monotonicity B = 93.9 (avif) / 99.4 (jpeg)
  / 98.3 (jxl) / 99.7 (webp) — all above the 93% gate; median ceilings
  88.9–96.4; jxl 95-reach 93.9% (A: 72.7%).
- **q-targeting simulation** (pick q minimizing |score−target|): median
  landing error 0.2–3.4pt across codecs × targets {30,50,70,85,90} — at
  parity with A overall, B better at T≥85 on jpeg/jxl/avif.
- **A↔B dial correspondence** (for migrating stored targets): B30≈A42.4,
  B50≈A59.6, B70≈A74.8, B85≈A88.4, B90≈A91.7 (medians; scales converge at
  the top). B is more pessimistic at low quality.

**Remaining blocker — RESOLVED 2026-07-05 (w11) as instrument corruption,
not a B gap.** The "3/24 webp ladders cap below 50" finding was measured on
dial-grid rows whose masked/IW-block features are extraction garbage (see
the corrected §webp below): on the quarantined grid (9 corrupt ladders
removed) **every honest webp ladder tops at ≥81 under B — ceiling p10 =
83.7 (was 9.4), min 81.0, mono 0.9952 — webp is knob-READY like
jpeg/jxl/avif.** The webp p90 landing errors at T≥70 were the same
artifact.

G3 (regress baselines per consumer) + G4 (no zensim-gpu B kernel) are
engineering items, unchanged. G5 closed (shipped bytes = anchored sibling).

## webp trio — FINAL correction (w11, 2026-07-05): dial-grid corruption, not B blindness

**Both earlier adjudications of this section were wrong about the cause.**
The trio's −80/−81/−12 came from the DIAL GRID's stored features, and those
rows are extraction garbage: masked/IW-block features (f228..f371) sit at
34..489 (f235 = 269.8/274.7/46.4), **bit-constant across each ladder's 40 q
values** — impossible for distorted-side features — while fresh CPU
extraction (`extract_features_372col`) on the same webp-inspect decodes
gives 0.003..0.025. A systematic screen flags **9 of 115 ladders** (8/24
webp + 9059ec…×jpeg); all flagged images hit odd dimensions in the scale
pyramid, and the grid's own provenance notes the zensim-gpu odd-dim
pathology (NaN cells dropped — these produced non-NaN garbage instead).

**The shipped B on FRESH pixels is sane and correctly ordered**
(q90/95/100): a9143 (clean by every judge) = 88.6/90.8/**92.1**; a06b =
74.8/78.5/81.5; c37e = 66.8/82.8/79.0 — vs ssim2 87.6-91.2 / 60.6-67.9 /
36.0-61.6 and A 90.6-93.0 / 64.3-72.5 / 41.3-71.0. No −80s exist on honest
inputs; grid-B was compared against fresh-ssim2/butter. What stands from
the earlier reads: a06b/c37e genuinely carry severe webp artifacts
(butter-max 5.3-21.5) and BOTH B and A are optimistic there vs butteraugli
(ssim2-family pooling dilutes localized artifacts — a metric-family
characteristic, not an OOD failure).

**The corpus refit (w11) was executed anyway and FALSIFIED**: every
safesyn-slice mass 0.02..1.0 × λ × τ costs held-out CID22 (0.842-0.857 vs
0.8733, gate ≥0.87) and usually KonJND; the ssim2-anchored cid22tr
selection axis moves UP while real CID22 moves DOWN (the ssim2-target trap,
now measured on a selection axis). Full tables:
`benchmarks/linear_projections_2026-07-03.md` §w11. **B-slot unchanged.**
Bottom-floor mitigation: bounds garbage-input damage at 0 (knob shows
"unreachable" instead of −80) with ZERO corruption-gate cost on this bake
(224/672 pass both ways), but cannot restore knob usability on corrupt
inputs. Real-input fragility (bigcodec_val heavy tails f155/f52/f216, 0.95%
below raw −2, min −938) is documented in §w11 — the falsified refit fixed
it but paid too much; a floor decision is product-side and open.

**Quarantined dial grid** (the honest instrument until a v2 rebuild):
`dial_grid_372col_2026-05-29_quarantined.parquet` (4,457 rows, sha256
`b5d27f212fc6b00c…`) next to the canonical file. Every dial number measured
on the 9 corrupt ladders since 2026-05-29 (any bake, incl. A) is
garbage-input scoring.

**Unrepresentable targets under B (≈ codec physics, shared with A, per the
ceiling-median comparison)**: >94 unreachable for 50% of (image,codec),
>99 for 90% (extrapolation past the 95.89 knot reaches ~100 for the top
decile); <2 unreachable for 50% (encode floors — easy content floors at
~89 on jpeg, so LOW targets are unrepresentable there under ANY metric);
per-ladder interior gaps median 12-18pt (low-q end) bound worst-case
landing error at ~gap/2, matching the T=30 p90 simulation errors.
Artifacts: webp-inspect/{pairs.tsv,*.parquet}, decoded PNGs.

## f155 tail forensics + winsorize verdict (2026-07-05)

Two distinct bugs behind the tail behavior (page:
reports/2026-07-05_f155_tails): (1) the dial grid's odd-dim GPU-extractor
corruption (9/115 ladders, instrument-side — quarantined; zensim-gpu fix +
grid v2 = open zenmetrics work); (2) f155's REAL heavy tail on tiny dark
screen-content renditions (val max 14,532 vs fit-corpus p99.9 = 0.479;
264/290 of B's sub-−5 raw rows; scale-pyramid degeneracy at ~50-100px).
**Winsorize verdict: clamp features to fit-corpus [p0.1,p99.9] — provably
bounds the linear raw output (inside knot domain), zero tails, free-or-
BETTER on the SDR axes.** SHIPPED 2026-07-05 (commit below).

**The fix used the mechanism that already existed — no new code.**
`zenpredict::FeatureTransform::WinsorP99` ("clip to [p1,p99], preserves
rank within bounds", + the `WinsorThen*` stack family, `QuantileBins`,
`clamp_inclusive`, `YeoJohnson`) has been in `zenpredict/src/
feature_transform.rs` since 2026-05-14/17, flows through the
`zentrain.feature_transforms` ZNPR metadata, is applied by
`Predictor::predict_transformed`, and is auto-dispatched by EVERY consumer
via `has_nontrivial_feature_transforms()` (zensim `metric.rs`) /
`has_transforms` (validate `bake_runtime.rs`). **The shipped BHdr bake
ALREADY carries 183 `winsor_p99` + 75 `quantile_bins` + 56 `signed_cbrt` +
53 `yeo_johnson` transforms** — which is exactly why only B SDR had the
tail: `ens-Pline-cid80` was the ONE bake baked in raw space with no
transforms at all. The fix: re-bake B SDR with 372 `winsor_p99`
transforms (fit-corpus [p0.1,p99.9] params), keeping its weights / scaler
/ spline — `b_sdr_linear_cid80_winsor_2026-07-05.bin` (12.9 KB, sha
`b92b0b7a…`). Measured on the SHIPPED ENSEMBLE through the real runtime
(`bake_verdict`, which dispatches `predict_transformed`): raw min
−1131→−1.86 (580 sub−5 rows → 0), CID22 0.8732→**0.8763**, KonJND
0.5434→**0.5474**, AIC-3 flat, identity=100 preserved. UPIQ −0.007 (HDR,
BHdr's domain, not an SDR-B gate). The `PROFILE_B` slot now points at the
winsor bake; a regression test
(`profile_b_tests::both_b_bakes_carry_winsorizing_transforms`) fails loud
if a future re-bake reverts to the raw `predict` path. The earlier
"feature_bounds section + inline clamp in the forward" plan was
ABANDONED — it reinvented (in the wrong layer) what `winsor_p99` already
does, and would have forced `bake_runtime.rs` to duplicate the clamp
(the two-code-paths-diverge trap). No zenpredict edit was needed.

## f155-tail vs GPU-odd-dim-bug reconciliation (2026-07-05, verified)

The zenmetrics odd-dim extractor bug (fixed `aade882e` on zenmetrics master —
`div_ceil`→floor pyramid heights, GPU/CPU mismatch on odd heights at any pyramid
scale) is a SEPARATE issue from the f155 tail, on SEPARATE files. Verified:

- **The f155 tail is REAL, on CLEAN data.** My winsor work used
  `bigcodec_valdigits/traindigits_2026-07-02.parquet` (non-hqdedup) — the agent's
  blast-radius screen found these CLEAN (0/147,067 and 0/2,946,036 row violations),
  CPU-extracted (group count matches the canonical-picker CPU sweep). I
  re-verified: f155 VARIES with distortion (n_distinct 51/105 rows on
  o_8301.scale54x96), not bit-constant like the GPU bug's dial-grid signature
  (269.8 across all 40 q). So f155=14,532 is a genuine degenerate-statistics heavy
  tail on tiny renditions (54×96 → ~6-px-wide pyramid scale), correctly extracted.
  **The winsor fix is correctly characterized.**
- **B's training data is CLEAN.** hdr_v3mix (B's cid head, 80%) + hdr_v3 (BHdr):
  0% masked>5, 0% IW>5, 0 bit-constant refs. HDR renditions are fixed-size so the
  odd-dim bug never touched them. B is built on clean data; its tail exposure was
  inference-only on tiny renditions.
- **The GPU bug's actual casualties are OTHER files:** the dial grid (9/115
  ladders — already quarantined), `bigcodec_hqdedup_{train,val}digits` (CORRUPT,
  82/4585 + 5/2745), jxl-hqfill-A (56/1650), and the `bigcodec_hqfill*` group
  (likely corrupt, unscreened). **CAVEAT (agent finding):** the corrupt
  `bigcodec_hqdedup_*` pair was used as the `bigcodec` train/val group by the
  **w11-era ABLATION bakes** (manifests `ab_base_s7.toml` etc.) — those ablations
  are compromised and should be re-run on the CLEAN `bigcodec_*digits` or
  re-extracted data. This does NOT affect shipped Profile B (trained on the
  verified-clean hdr_v3mix + canonhdr15). Full per-dataset table:
  zenmetrics `docs/ZENSIM_GPU_ODDDIM_CORRUPTION_2026-07-05.md`.

**B vs ssim2 training (for interpreting the KADIS color divergence):** B trains to
predict **ssim2 + cvvdp (a 0.5/0.5 mix), dominantly on COMPRESSION renditions**
(hdr_v3mix HDR-JXL 80% + canonhdr15 SDR sweeps 20%); no CID22-49 MOS. SSIMULACRA2
is hand-tuned to **human MOS on CID22-201 + KADID + TID** (KADID = the color/denoise
distortion families). So on KADIS color distortions ssim2 is human-tuned and B is
out-of-domain — the divergence favors ssim2 there; on compression (B's domain) they
agree.

## Measured: linear-B vs best-MLP (t1dro51) — dial + safety (2026-07-05)

Ran `bake_verdict` (rank + DIAL panels) + `bake_outlier_gate` on both, same
dial grid (`dial_grid_..._quarantined`, 106 curves/4 codec families) + same
147k bigcodec_val safety corpus.

**RANK** (held-out): linear-B CID22 **0.8763** / KonJND **0.5474** / AIC-3 0.7774;
t1dro51 CID22 0.8708 / KonJND 0.3109 / AIC-3 **0.8013**. Linear wins CID22
(+0.006) + KonJND (+0.24); MLP wins AIC-3 (+0.024).

**DIAL** — the MLP is CLEANER, not the linear:
| | inversions (G3≤0.07) | flat/clamp dead-zone (G3≤0.05) | G1 dyn-range |
|---|--:|--:|--:|
| linear-B (winsor) | 0.0262 ✓ | **0.0563 ✗** | p5=50.2 (fail) |
| linear-B (pre-winsor) | 0.0269 ✓ | 0.0623 ✗ | — |
| **MLP t1dro51** | 0.0246 ✓ | **0.0000 ✓** | p5=48.6 (fail) |

- The linear's ~5.6% dead-zone is INHERENT (pre-winsor 6.23% → winsor slightly
  REDUCED it to 5.63%; NOT a winsor artifact — my first hypothesis, falsified by
  measuring the pre-winsor bake). A 372→1 linear collapses many near-lossless
  feature deltas to identical scores; the MLP's 128→64 hidden capacity resolves
  them (0 flat). So on the dial POINT metrics the MLP wins.
- G1 (reach to p5≤25) FAILS for BOTH on this grid (p5≈48-50).
- The provenance table's "0 dead" note for the 823B bake is STALE vs this grid
  (measured 6.23%).

**What linear actually wins on the dial: RELIABILITY, not point-quality.** The MLP
is a seed lottery — 43.75% of v3 seeds COLLAPSE to G1=0.00 (flat/dead dial, mono
0.996 constancy artifact); t1dro51 s31 is a HEALTHY survivor. Linear is Gram-exact
deterministic, 0 collapse — its (imperfect) dial is GUARANTEED, not drawn.

**SAFETY (G-RANGE tail):** linear-B post-winsor PASSES (raw [-1.86,1.12], 0
extrapolating); pre-winsor FAILED (-1131 tail). The MLP is NOT bounded by
construction — leaky_relu hidden reaches 33,709 / final 3,006 on the same tail
rows — BUT its tail is POSITIVE, which the dial caps at 100; the linear's was
NEGATIVE (below the bottom knot → uncapped downward → the -80 dial). So the acute
failure was linear+negative-tail-specific; the winsor guard fixes the input side.
Z-RMSE (calibration) vs human: AIC-3 linear-B 0.616 vs MLP **0.579** (MLP better).
xmetric vs 5 peers (KADIS-700k): linear-B diverges 9%, concentrated on color
distortions (out-of-domain), sides with iwssim — characterized, not a bug.

**Bottom line:** linear-B wins CID22/KonJND rank + determinism + size (12.9KB vs
48KB); t1dro51 wins AIC-3/AIC-4 rank + dial cleanliness (0 dead-zone) + AIC-3
calibration. Neither dominates. Since A has NOT shipped (crates.io still at 0.2.7
= old PreviewV0_2; A/B/BHdr all unreleased in 0.3.0), the 0.3.0 SDR-default choice
among {A=v47-MLP, t1dro51-MLP, linear-B} is fully open.

## Best-of-both probe: linear rank + MLP-clean dial (2026-07-05)

QUESTION: can linear-B get the MLP's clean dial without losing its rank? Diagnosed
the linear's dial dead-zone (5.63%, the only dial gate it failed) — 100% of it is
near-lossless PILE-UP at the top spline knot (y=95.9 at raw 1.138), while
near-lossless dial configs reach raw 2.8. The linear's RAW output is distinct on
every near-lossless pair (0 raw-flat); only the spline cap flattens them. So it's a
pure calibration fix, RANK-INVARIANT (monotone spline).

FIX (`scripts/v_next/dense_dial_refit_b.py`): extend the spline top monotonically
to y=100 over the OOD-high near-lossless raw range (near-lossless → 100 by
definition). Result — dense-dial-B vs shipped-B, same weights+winsor, spline only:

| model | CID22 | KonJND | inversions (≤0.07) | dead-zone (≤0.05) | dial verdict |
|---|--:|--:|--:|--:|---|
| shipped-B (linear) | 0.8763 | 0.5474 | 0.0262 ✓ | **0.0563 ✗** | 1 gate FAIL |
| **dense-dial-B** | **0.8763** | **0.5474** | 0.0386 ✓ | **0.0005 ✓** | **ALL PASS** |
| MLP t1dro51 | 0.8708 | 0.3109 | 0.0246 ✓ | 0.0000 ✓ | ALL PASS |

**dense-dial-B is a strict improvement over shipped-B** (identical held-out rank —
CID22/KonJND to 4dp — now passes ALL dial gates). It keeps the linear's CID22+KonJND
+ determinism + 13KB AND matches the MLP on dial-gate compliance. The one residual:
inversions rose 0.0262→0.0386 (still ✓) — the flat cap was HIDING near-lossless rank
noise that the extension exposes; the MLP's higher capacity resolves that noise
(0.0246). So best-of-both on the DIAL is achieved (all gates pass, rank intact); the
MLP keeps only a marginal inversions edge + its AIC-3/AIC-4 RANK win.

**What is NOT solved: the rank Pareto.** dense-dial-B still trails the MLP on AIC-3
(0.777 vs 0.801) / AIC-4 (0.891 vs 0.915); the MLP still trails on KonJND (0.31 vs
0.55). The dial fix is orthogonal to this. Closing the AIC-3/AIC-4 gap without losing
KonJND is the open "best of both on rank" question (levers: ensemble the MLP's
AIC-strong head as a hard-routed component; a KonJND-recovered MLP — but MLP+KonJND
is a documented G5 Pareto limit).

**SHIPPED 2026-07-05.** The hand-placed extension was replaced by a PRINCIPLED one
and rotated in as `ZensimProfile::B`. Key correction over the first cut: **extend the
winsor bake's EXISTING spline top — do NOT rebuild the spline from the anchor.**
Rebuilding from the balanced anchor lifts the bottom knot off the real-content raw
floor (multiband winsorized preds reach only −1.37; real bigcodec content reaches
−1.86), so low-quality content extrapolates DOWNWARD — the wild-negative tail winsor
exists to prevent (outlier gate caught it: 33% below-knot, then 4.4% after a partial
fix). `scripts/v_next/dense_dial_refit_b.py` therefore reads the archived winsor
bake, keeps its bottom + in-distribution knots VERBATIM (domain already reaches
−1.974), and appends ONLY the OOD-high top: `score(r)=100−(100−y0)·exp(−k·(r−x0))`
with decay `k=3.31` fit by ROBUST regression `log(100−target) ≈ log A − k·raw` on the
600 near-lossless multiband-anchor rows (target>70), from the top knot (1.138, 95.9).
No hand values, no eval-grid dependence (real content tops at raw ~1.12 so the
endpoint can't come from data — the saturation shape IS the principle; identity=100
is held by the `is_identical` short-circuit, not the spline top). Shipped bake
`b_sdr_linear_cid80_dense_dial_2026-07-05.bin` (12,988 B, sha `b78adb15`,
deterministic re-run byte-identical), reproduced by `scripts/reproduce_b.sh`. All 5
profile tests pass (identity, B→BHdr routing, winsor-transform guard). The winsor-only
predecessor is in `zensim/weights/archive/`. **Result vs both prior points:** strictly
better than the winsor-only bake (dead-zone 0.0563 FAIL → 0.0005 PASS) AND the rank
noise the rebuild introduced is gone (inversions 0.026 ≈ the winsor bake's 0.026 ≈ the
MLP's 0.025) — because only the OOD tail moved, the in-distribution rank is byte-for-
byte the winsor bake's. G-RANGE PASS (0 extrapolating, both tails). The rank Pareto
(AIC-3/AIC-4 vs KonJND) is untouched and remains the open "best of both on rank"
question.
