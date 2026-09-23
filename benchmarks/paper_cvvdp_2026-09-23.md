Devin-executed; Opus-reviewed 2026-09-23 (REVIEW_PAPER_MEASURE.md): PROMOTE WITH CORRECTIONS

# Display-matched CVVDP for the zensim paper — 2026-09-23

Peer-fairness lane of the zensim companion paper. Every CVVDP number on the
board was computed for pycvvdp's default display, `standard_4k` (30-inch 4K at
0.7472 m, 75.40 pixels per degree). This record scores CVVDP at each study's
**documented** display instead, through the same instruments (the in-tree CPU
port behind `zenmetrics batch --metric cvvdp`, the peer-row builder, the
canonical `panel`). No zensim model is scored here.

Reviewer independently confirmed SROCC standard_4k → study display: AIC-4 crop 0.8906 → 0.9609, AIC-3 0.7918 → 0.8246, KonJND 0.0562 → 0.4193, CID22-A 0.8197 → 0.8173.

The Opus hand-off bookmark originally pointed to `b5cedde4`, with a 26-line record stub and 16-line pointer. Its later `0a5e8e1b` target is tree-identical to the earlier Opus commit `7bfda9ec`, but **not** to bookmarked hand-off `b5cedde4`; the latter differs by 42 deleted lines across two files. Both states remain recoverable.

<!-- RESULTS -->

## What changed when the display matched the study

Pooled SROCC vs the human labels, `panel` owner, and paired bootstrap
(B = 2000, seed 20260923) of the display-change delta. "Cluster" resamples
whole references; "pairs" resamples individual pairs (the UPIQ record's
convention — reported alongside because per-reference counts are small).

| set | board display (4k) | documented display | Δ pooled SROCC | cluster CI95 | pair CI95 |
|---|---|---|---|---|---|
| AIC-3 CTC (n=600, 10 refs) | 0.7918 | **0.8246** (fhd) | +0.0329 | [−0.0117,+0.0851] | [+0.0180,+0.0487] |
| AIC-4 sample (n=300, 5 refs) | 0.8906 | **0.9609** (fhd) | +0.0703 | [+0.0054,+0.1519] | [+0.0461,+0.1010] |
| SDR25 (n=50, 5 refs) | 0.8609 | **0.9464** (fhd) | +0.0855 | [−0.0218,+0.1671] | [+0.0157,+0.1806] |
| CID22-A (n=2192, 25 refs) | 0.8197 | **0.8173** (css 46.91) | −0.0024 | [−0.0391,+0.0281] | [−0.0106,+0.0057] |
| KonJND-1k JPEG (n=504) | 0.0562 | **0.4193** (kon 24.29) | +0.3631 | [+0.2864,+0.4243] | same |

KonJND is the big one: at the paper's default display CVVDP is near-zero on
KonJND (0.056); at the study's own calibrated geometry it reaches 0.419 —
still below every other arm on this corpus, but an honest baseline. CID22-A
is flat (−0.0024): its documented CSS-pixel geometry is close enough to 4k
that nothing moves, so the board number needs no display correction.

AIC-4 full-resolution rendering (the PTC files before cropping) at fhd lands
at 0.9606 vs 0.9609 for the crops — Δ = −0.0003, CI95 [−0.0058,+0.0065]:
crop and full-resolution encodes are statistically identical under the
metric. The organisers' published per-file CVVDP for those full-resolution
files is reproduced by our CPU port to max |Δ| = 5.3e-4 JOD (n = 300; see
parity below), so the recorded GPU rung and the CPU port agree on this
corpus at print precision.

## UPIQ-HDR: peak and geometry

UPIQ's own subjective protocol ran on a 47-inch full-HD HDR SIM2 monitor.
The board carries recorded `cvvdp-gpu` runs at `standard_4k` geometry with
display peaks 1000–10000. Scoring the same 380 pairs at each study's
documented geometry (Narwaria 56.55 px/deg ≈ 3.0H; Korshunov 60.32 ≈ 3.2H)
and documented 4000 cd/m² peak:

| configuration | pooled SROCC | Narwaria | Korshunov |
|---|---|---|---|
| recorded GPU, 4k @ 1000 peak | 0.7580 | 0.6641 | 0.9645 |
| recorded GPU, 4k @ 4000 peak | 0.8153 | 0.7819 | 0.9692 |
| recorded GPU, 4k @ 6000 peak | 0.8245 | 0.7817 | 0.9691 |
| recorded GPU, 4k @ 10000 peak | 0.8309 | 0.7807 | 0.9686 |
| CPU port, 4k @ 6000 | 0.8245 | 0.7817 | 0.9691 |
| CPU port, 4k @ 10000 | 0.8310 | 0.7807 | 0.9686 |
| **documented geometry @ 4000** | **0.8289** | **0.7992** | **0.9682** |
| documented geometry @ 6000 | 0.8321 | 0.7940 | 0.9679 |
| documented geometry @ 10000 | 0.8353 | 0.7864 | 0.9678 |

Paired deltas (cluster CI95 / pair CI95):

- recorded 4k @4000 → documented @4000: **+0.0135** [−0.0052,+0.0420] / [+0.0036,+0.0253]
- recorded 4k @6000 → documented @6000: +0.0076 [−0.0044,+0.0239] / [+0.0007,+0.0155]
- recorded 4k @10000 → documented @10000: +0.0043 [−0.0044,+0.0137] / [−0.0018,+0.0108]
- recorded 4k @10000 → documented @4000: −0.0021 [−0.0194,+0.0125] / [−0.0116,+0.0083]

Reading: at matched peak the documented geometry is the better
configuration (+0.004 to +0.014 pooled), but the unclamped 10000-nit run at
4k geometry is ≈ the documented 4000-nit configuration (−0.002) — the
display peak matters more than the geometry on this set, and the recorded
10000 row was already a fair stand-in. Per-study, the documented geometry
lifts Narwaria (+0.017 vs recorded 4k @4000) and slightly costs Korshunov
(−0.001); the pooled gain is Narwaria-driven.

## Parity (our scoring vs stored/organisers' values)

| check | n | max \|Δ\| | >1e-3 |
|---|---|---|---|
| AIC-4 crop, default 4k vs board table | 300 | 0 | 0 |
| AIC-4 crop, fhd vs recorded cvvdpfix fhd | 300 | 0 | 0 |
| AIC-3, default 4k (head 60) vs board | 60 | 0 | 0 |
| SDR25, default 4k vs board | 50 | 0 | 0 |
| CID22-A, default 4k (head 200) vs board | 200 | 1.0e-4 | 0 |
| KonJND, default 4k (head 100) vs board | 100 | 4.3e-4 | 0 |
| fhd via `--display-geometry` vs `-d standard_fhd` | 50 | 0 | 0 |
| CPU port vs recorded GPU, UPIQ 4k @6000 | 380 | 8.7e-4 | 0 |
| CPU port vs recorded GPU, UPIQ 4k @10000 | 380 | 8.7e-4 | 0 |
| AIC-4 **full-res** fhd vs organisers' full-res CSV | 300 | 5.3e-4 | 0 |
| AIC-4 **crop** fhd vs organisers' full-res CSV | 300 | 0.129 | 288 |

The last line is the control, not a failure: the crop and the
full-resolution file are different rendered stimuli, so they must differ
(they do — and per the paired delta above the metric still scores them
identically at fhd). The 5.3e-4 agreement on the full-resolution files is
the external-implementation anchor.

CID22-B (24 sealed references) was scored pixels-only (`cid22B_css` table,
sha256 `b892e7f7…`) with no label column and is **not** used for any
label-bearing statistic.

## Optimized-binary verification (r5900xt, 2026-09-23 ~10:4xZ)

Per user instruction the lane re-scored every job it could with the optimized
build from the r5900xt cvvdpvideo workspace
(`zenmetrics` sha256 `373e7156…`, `upiq_hdr_score` sha256 `9d729491…`;
their `master` head `b02812ae`, which adds magetypes kernels for the video hot
loops while the still path is required bit-identical):

- **All 9 SDR `batch` tables bit-identical** (aic3_fhd, sdr25_fhd,
  aic4full_fhd and all six `par_*` preset tables; ~1950 rows, 0 diffs).
- The r5900xt binary's CLI predates this lane's `--display-geometry` flag, so
  the four custom-geometry jobs (par_fhd_geom_equals_preset, cid22A_css,
  cid22B_css_pixels_only, konjnd_kon) cannot run on it (rc=2). Their tables
  remain produced by the lane build; the fhd-preset rows they parallel are
  bit-identical under both binaries.
- The r5900xt `upiq_hdr_score` example is the production `HdrScorer` route and
  **reproduces the recorded GPU UPIQ tables to max |Δ| = 2.9e-6 JOD** — a
  stronger CPU/GPU parity anchor than the lane example's 8.7e-4.
- The lane's explicit-geometry CPU example deviates from `HdrScorer` by
  ≤8.7e-4 JOD at matched peak and 4k geometry (mean −1.5e-4). The deviation is
  **rank-identical (Spearman 1.0)**, so every SROCC and delta in this record
  is unaffected; absolute UPIQ JOD values in the doc-geometry tables carry a
  ≤9e-4 instrument caveat. The example's 5th positional arg (geometry) is
  silently ignored by the older r5900xt example, so its "doc-geometry" opt
  outputs were discarded rather than trusted.

## Peer rows for the board

`scripts/v_next/build_peer_fullevals.py --refmetrics-dir /var/tmp/paper-cvvdp/refmetrics`
produced (bulky, not committed — `/var/tmp/paper-cvvdp/fulleval/`):

- `peer_cvvdp_aicfhd` — aic4 0.9609, aic3 0.8246, sdr25 0.9464, aic4_fullres 0.9606
- `peer_cvvdp_studydisplay` — aic4 0.9609, aic3 0.8246, sdr25 0.9464, cid22A 0.8173, konjnd 0.4193, upiq_hdr 0.8289
- `peer_cvvdp_4k_cid22A` — cid22A 0.8197 (same-population 4k comparator)
- `peer_cvvdp_upiq_display` — the nine UPIQ cells above

<!-- TABLES -->

## Where each display comes from (primary sources)

| study | documented viewing condition | CVVDP configuration used | pixels per degree |
|---|---|---|---|
| AIC-3 CTC, AIC-4 sample, SDR25 | JPEG AIC CTC v2.0 (wg1n101246 §4): ColorVideoVDP 0.4.2 `-d standard_fhd` — 37.84 px/deg, 200 cd/m², black 0.2, reflected 0.3979. This is the metric-evaluation display the organisers use; the AIC-3 CTC paper (QoMEX 2023) itself records only crowd-sourced expert viewing. | `--display-model standard_fhd` | 37.84 |
| CID22 (Sneyers, Ben Baruch, Vaxman) | DSBQS (the absolute-score protocol): images at `dpr1`, "one image pixel corresponds to one CSS pixel, which theoretically corresponds to a visual angle of 0.0213 degrees (though in practice this may only be an approximation)"; desktop/laptop only. TSBPC (the pairwise part): images upscaled to fill the screen height — no fixed geometry. Crowd-sourced; "large differences in viewing conditions will inevitably remain". | CSS reference pixel = 96 dpi at 28 in: `--display-model standard_4k --display-geometry 1920,1080,22.947,0.7112` (photometry unchanged) | 46.91 |
| KonJND-1k (Lin et al., TCSVT 2022) | Crowd-sourced; credit-card calibration of pixel density; every 640×480 image shown at 13.797 cm × 10.347 cm (the size on a 13.3-inch 1366×768 screen); workers asked to sit 30 cm away. | `--display-model standard_4k --display-geometry 1366,768,13.3,0.30` | 24.29 |
| UPIQ-HDR (Narwaria 2013; Korshunov 2015) | UPIQ's `pix_per_deg`: 56.5487 (Narwaria), 60.3186 (Korshunov). Korshunov documents a 47-inch full-HD SIM2 display, 0.001–4000 cd/m², at 3.2 picture heights; 47-inch 1920×1080 at 3.0H / 3.2H reproduces both UPIQ values to 4 decimals. | `upiq_hdr_score` example, CPU cvvdp, `STANDARD_HDR_LINEAR` photometry at the stated peak, geometry `1920,1080,47,1.75582` / `1920,1080,47,1.87288` | 56.55 / 60.32 |

Photometry is left at the board's 200 cd/m² standard office display for every
SDR set: no SDR study documents it. The only knob changed is geometry, which
enters the still-image metric solely through pixels per degree (verified in
`crates/cvvdp/src/pipeline.rs`, `with_geometry`).

## Provenance

- Scoring: `/var/tmp/paper-cvvdp/run1.sh` → `run_scores.py`; binaries
  `/var/tmp/paper-cvvdp/bin/{zenmetrics,upiq_hdr_score,panel}` (sha256 in
  `scores/manifest.json` and `analysis.json`: zenmetrics `ca154a6a…`,
  upiq_hdr_score `531b11af…`).
- Score tables + shard logs: `/var/tmp/paper-cvvdp/scores/` (manifest binds
  every output sha256 to its input pairs table).
- Analysis: `benchmarks/paper_cvvdp_2026-09-23/analyze.py` →
  `/var/tmp/paper-cvvdp/analysis.json` (parity, stats, deltas; all stats via
  the canonical `panel` through `scripts/lib/zen_stats.py`).
- UPIQ geometry derivation: `upiq_verify.py` → `upiq_verify.json` (the
  47-inch geometry reproduces UPIQ's published `pix_per_deg` to 4 decimals).
- Validation caveat: the trailing `cargo clippy` in the build wrapper failed
  on a pre-existing style lint (`chunks_exact` at
  `zenmetrics-api/src/cpu_dispatch.rs:1065`). Scoring itself completed
  (wrapper rc=0) and every score table passed manifest checks, but the lint
  gate is recorded as failed — the affected file is not in the cvvdp code
  path used here.
