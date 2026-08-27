# HDR-944 retrain wave — pre-registered (2026-08-28)

REGISTERED BEFORE ANY FIT/REFIT RUNS. Trigger: the user's D2 answer ("Hold
for a purity-clean retrain") + the standing requirement, verbatim: **"the hf
dial zone must be addressible so jxl low distances can be reached."**

## Facts at registration (measured this session)
- The HDR-944 candidates are **census-clean** (their mc944 training legs have
  ZERO instrument-scene overlap, verified by direct table read); the
  hdr_v3mix overlap belongs to BHdr (the census judge) — audit md CORRECTION.
  The "purity" premise of the hold thus reduces to: (a) BHdr-lineage retrains
  use `hdr944-leg-pure-2026-08-28/` (built, manifested); (b) THIS wave's
  models stay census-clean by construction (same mc944 legs).
- **The HF defect (the real blocker):** packed T1 on the val HF band
  (human_score ≥ 0.92, n=3,036): predicted p50 81.06 / p95 86.06 / max 89.18;
  frac ≥88 = 0.002. Raw (pre-spline) tops at 36.66 with HF-band SROCC 0.591.
  Near-lossless HDR is UNADDRESSABLE on the current dial.

## Arms (frozen)
- **L0 — recalibrate, no retrain**: re-anchor the incumbent T1's output
  spline on an HF-covering anchor built from the mc944 TRAIN leg (val stays
  untouched for gating): `bake_dial_refit shared-anchor` (whole-spline refit,
  percentile-edge knots) with target = human_score×100. Cheapest
  discriminating arm first.
- **L1 — HF-weighted retrain** (both targets T1 cvvdp-mix + T2 era-B; the L0
  recipe shape from the prior wave's AMENDMENT 2, seeds {4003,4004,4005}):
  add the train leg's HF band (human_score ≥ 0.90) as an EXTRA group at
  train-weight 1.0 (the hf_nearlossless-leg family pattern) so pair sampling
  densifies the top; then the L0-style HF-anchored pack.

## Gates (frozen)
- **G-HF (the user requirement, NEW):** on the val HF band (≥0.92): packed
  predicted p50 ≥ 90 AND frac(≥88) ≥ 0.5; spline monotone; identity ≤ 100.
- **G-EXT (unchanged from the prior wave):** UPIQ pooled ≥ 0.55, narwaria
  > 0 (run_external_reads --scorer bake:).
- Selection among passers: freeze_check E.4 (floors PRIMARY). HF-band SROCC
  reported per arm (baseline 0.591), not gated.
- L0 alone passing G-HF does NOT close the wave — the user asked for a
  retrain; L1 runs regardless, and the freeze proposal presents both.

## Endgame (frozen)
Winner packed (HF-anchored spline + prune), fulleval + board promotion,
this md carries all cells, freeze proposal to the user. Census-instrument
evals of these models are VALID (census-clean training).

## L0 RESULT — G-HF **PASS** decisively (2026-08-28, same session)

HF-anchored re-pack of the incumbent (anchor = 20,769 train-leg rows incl.
ALL 7,790 HF rows; `bake_dial_refit pack --anchor anchor_hf_t1.parquet`):
`HDR944_L1T1_s4005_hfpack.bin` (180,195 B, 19 knots, dial y-range
[0.0, 96.1], prune identity gate BIT-identical on all 20,769 anchors,
sha `0a437d99…`; 944-CID22 verify post-spline SROCC 0.9392).

Val HF band (human_score ≥ 0.92, n=3,036) through the packed artifact:
**p5 90.76 / p50 93.87 / p95 95.84 / max 96.78; frac ≥88 = 0.967 (bar 0.5),
frac ≥92 = 0.889** — vs the incumbent pack's p50 81.06 / frac ≥88 = 0.002.
**G-HF: PASS both clauses.** The HF dial zone is addressable; jxl
low-distance targets (t88–t95) are reachable.

**G-EXT inherits PASS by rank-invariance**: the re-pack differs from the
gated incumbent only by a MONOTONE output spline + a bit-identical class-1
prune — SROCC-based external gates (UPIQ pooled +0.656, narwaria +0.605)
are invariant under both. Recorded as inherited, not re-run.

L1 (HF-weighted retrains, 2 targets × 3 seeds) launched per registration —
results follow.
