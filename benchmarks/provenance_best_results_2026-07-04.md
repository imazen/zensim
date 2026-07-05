# Provenance + reproducibility — best results of the 2026-07-02..04 campaign

Canonical record for every ship-grade artifact the strategy/linear/HDR
campaign produced. Every claim below is committed evidence; every artifact is
rebuildable from pinned inputs. Repo commits referenced are on
`imazen/zensim@main` unless marked zenmetrics.

## The candidates (panel |SROCC| vs human; same harness, references reproduce
## their known numbers: A CID22 0.8657, ssim2 0.8894/0.4784/0.958)

| artifact | bytes | sha256 (16) | CID22 | AIC-3 | AIC-4 | KonJND | UPIQ | SDR25 | dial |
|---|--:|---|--:|--:|--:|--:|--:|--:|---|
| **Profile B (SDR): lp_ens-Pline-cid80-anchored-f16** | 823 | `7b326ac56a05c240` | 0.8733 | 0.7775 | 0.8906 | 0.5439 | 0.6846 | — | mono 0.9711, 0 dead, p95≤100 post-5d4978db |
| its pre-anchor twin (tau0) | 823 | `1cddfe5e14d81128` | same ranks (spline is rank-invariant; SROCC identical to 1.000000) | | | | | | knots >100 (superseded) |
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

**Remaining blocker (the only knob-quality gap found): the webp high-target
tail.** 3/24 webp ladders cap below score 50 under B (A gives ~62+):
`a06b91d3d8419aad_513x769`, `a9143f4b78fe5a13_513x769`,
`c37e9ae52fbab790_1022x818` — driving p90 landing errors of 5.3–8.9pt at
T≥70 on webp (A: 0.7–1.4). OPEN: visually inspect those encodes to decide
whether B is honestly penalizing webp artifacts A misses, or blind — the
answer decides between "document as codec-specific caveat" and "ensemble
tweak". Non-webp codecs have NO blocker: B is knob-ready for jpeg/jxl/avif
targeting on these measurements.

G3 (regress baselines per consumer) + G4 (no zensim-gpu B kernel) are
engineering items, unchanged. G5 closed (shipped bytes = anchored sibling).

## webp trio adjudicated + unrepresentable-target map (2026-07-05)

**The 3 webp ladders are a B-SPECIFIC feature-OOD blindness, not honest
penalty**: fresh independent scoring (webp-inspect/) at q90-100 — ssim2 says
61-91, butteraugli agrees (1.2-7.9), A says 32-64 (low but sane), **B says
−80/−81/−12** (raw pred below the bottom knot → deliberately-uncapped
downward extrapolation). Two of three share the odd 513×769 shape; the fit
corpus (HDR-mix, 7,410 rows) lacks this content class. FIXES (open): refit
the ensemble with such SDR content in-corpus (real fix); knob-context floor
at bottom-knot y (mitigation — turns −80 into 0, honest "unreachable");
until then webp is caveated in the migration plan.

**Unrepresentable targets under B (≈ codec physics, shared with A, per the
ceiling-median comparison)**: >94 unreachable for 50% of (image,codec),
>99 for 90% (extrapolation past the 95.89 knot reaches ~100 for the top
decile); <2 unreachable for 50% (encode floors — easy content floors at
~89 on jpeg, so LOW targets are unrepresentable there under ANY metric);
per-ladder interior gaps median 12-18pt (low-q end) bound worst-case
landing error at ~gap/2, matching the T=30 p90 simulation errors.
Artifacts: webp-inspect/{pairs.tsv,*.parquet}, decoded PNGs.
