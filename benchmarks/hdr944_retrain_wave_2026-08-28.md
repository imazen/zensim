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

## JXL LOW-DISTANCE REACHABILITY — MEASURED YES (2026-08-28, user question "are d 0.1 to d 1 jxl reachable")

Val zenjxl ladder (270 cells/rung) through the L0 hfpack; d = the public
`quality_to_distance` mapping (q≥90 → d=(100−q)/10):

| d | q | dial p50 | | d | q | dial p50 |
|---|---|---|---|---|---|---|
| 1.0 | 90 | 92.53 | | 0.4 | 96 | 94.55 |
| 0.8 | 92 | 93.06 | | 0.2 | 98 | 95.16 |
| 0.6 | 94 | 93.63 | | 0.0 | 100 | 95.39 |

d0.1 ≈ target 95.3; the whole band sits below the dial top (max 96.8).
Controllability: per-scene monotone 99.9% (1888/1890 steps); adjacent-rung
separation median 0.54 dial points, 99.9% positive — every 0.2-distance step
is distinguishable (loop tol ±0.5 ⇒ ~0.2d resolution; ±0.25 ⇒ ~0.1d). The
dial p50 tracks the leg's own cvvdp-mix target p50 within ~0.3 at every rung.
Under the incumbent pack this band read 81–86, unreachable — the L0 fix is
what makes it addressable.

## PEER METRICS on the same instrument (user request: "ssim2 and butter and cvvdp as peers")

Same 2,160 val zenjxl cells (q≥86), scores joined from the post-drain
harvest by encode_sha (`harvest-2026-08-27/scores.parquet`; join 2,160/2,160):

| q | d | zensim hfpack p50 | ssim2 p50 | butter_max p50 | cvvdp JOD p50 |
|---|---|---|---|---|---|
| 90 | 1.0 | 92.53 | 86.46 | 1.998 | 9.973 |
| 92 | 0.8 | 93.06 | 87.54 | 1.858 | 9.979 |
| 94 | 0.6 | 93.63 | 88.41 | 1.682 | 9.983 |
| 96 | 0.4 | 94.55 | 89.96 | 1.410 | 9.990 |
| 98 | 0.2 | 95.16 | 90.85 | 1.230 | 9.993 |
| 100 | 0.0 | 95.39 | 91.24 | 1.146 | 9.994 |

Controllability in the d1.0→d0 band (per-scene, 1,350 adjacent steps):

| | monotone | frac step>0 | median step |
|---|---|---|---|
| **zensim hfpack** | **99.9%** | **0.999** | 0.54 |
| ssim2 | 99.3% | 0.966 | 0.94 |
| cvvdp | 98.5% | 0.985 | 0.0041 JOD (ceiling-saturated: 9.97→9.99) |
| butter_max | 91.7% | 0.896 | 0.157 |

Reading: ssim2 is a strong peer with the largest per-step magnitude; cvvdp
discriminates but against a saturated ceiling (tolerance bands become
sub-0.01-JOD — impractical as a dial up here); butteraugli-max is the
noisiest orderer in this band. The hfpack peers at the top on both
controllability axes. These rows accompany every freeze candidate table.

## L1 RESULTS (2026-08-28) — all six retrains PASS G-HF and G-EXT; sihdr flag

Training: 6/6 done (2 targets × seeds {4003,4004,4005}; HF group engaged —
in-train hdrmc_hf SROCC climbing through 0.92-0.97). All packed with the
HF anchors (identity gates BIT-identical; T1 dial tops 96.3-96.4, T2 93.4
with a negative-capable bottom).

| candidate | G-HF p50 / frac≥88 | UPIQ pooled | narwaria | korshunov | CID22-944 verify | HF-band SROCC |
|---|---|---|---|---|---|---|
| **L0 hfpack (incumbent+re-anchor)** | 93.87 / 0.967 | **+0.666** | +0.643 | **+0.928** | **0.9392** | 0.591 |
| R t1_s4003 | 93.83 / 0.979 | +0.643 | +0.684 | +0.832 | 0.9096 | 0.619 |
| R t1_s4004 | 93.82 / 0.979 | +0.632 | +0.687 | +0.828 | 0.9012 | 0.623 |
| R t1_s4005 | 93.92 / 0.979 | +0.626 | +0.684 | +0.816 | 0.8846 | 0.623 |
| R t2_s4003 | 92.30 / 1.000 | +0.659 | **+0.702** | +0.875 | 0.8991 | **0.716** |
| R t2_s4004 | 92.25 / 1.000 | +0.631 | +0.699 | +0.825 | 0.8861 | 0.692 |
| R t2_s4005 | 92.29 / 1.000 | +0.636 | +0.683 | +0.826 | 0.8744 | 0.710 |

All 7 PASS both frozen gates (G-HF: p50≥90 ∧ frac≥88≥0.5; G-EXT: pooled
≥0.55 ∧ narwaria >0). The retrains buy HF discrimination (+0.03 t1-band,
t2-band 0.69-0.72) and narwaria (+0.04-0.06) at SDR-CID22 cost (−0.03..−0.05
vs L0's 0.9392). **Diagnostic flag (not a registered gate): the retrains flip
`sihdr pooled` NEGATIVE (−0.41..−0.54) where L0 reads +0.358** — the
HF-weighting inverted the SI-HDR out-of-domain ordering; recorded for the
freeze decision. jxl d-ladder: t1_s4003 ≡ L0's curve (92.5→95.4 across
d1.0→d0, band fully addressable); t2_s4003 maps lower (86.5→91.4, also
addressable on its own scale).

freeze_check refused selection pending M3a (required-measured, appendix E.4)
— the coherence instrument is running for all 7; selection follows.

## SELECTION (registered E.4 rule) + ENDGAME — 2026-08-28

M3a measured for all 7 (coherence instrument, owner run_full_eval).
**SELECTED: `HDR944_L1T1_s4005_hfpack`** — floors 5/8 (top, tied with the
three t2 retrains), tie-break selection_composite **0.8853** (t2 retrains
0.795-0.814; t1 retrains 4/8 floors), M3a 0.7642, sdr25 0.9667. sha256
`0a437d9927dd63dc…`, 180,195 B, dial [0, 96.1]. The L0 arm (incumbent + HF re-anchor)
beats the retrains under the registered rule: the retrains' HF-discrimination
and narwaria gains do not offset their floor/composite/SDR-CID22 costs — and
the retrains carry the sihdr sign-flip flag. Runner-up:
`HDR944R_t2_s4003_hfpack` (5/8, M3a 0.8122, HF-band SROCC 0.716 — the
discrimination lead; kept packaged for a future HF-precision wave). All 7
fullevals + M3a artifacts on the board dir; every cell in this md.
**Freeze remains USER-GATED — proposal presented with peers + d-ladder.**

## GATE-SCORECARD CORRECTION + SEMANTICS AUDIT (2026-08-28, the user's challenges)

**Gates (the user is right).** The E.4 selection and the GATE scorecard
disagree, and the earlier recommendation presented only the E.4 lens. Gate
facts: the selected L0 pack reads `g1_dynamic_range` **0.696** and
`weighted_goal` 0.420 against the t2 retrain's **g1 = 1.0** and 0.453 (dial
reach 77 vs 107; shipped-B reads g1 = 1.0, weighted_goal 0.610). The two
lenses trade: E.4 (floors + balanced composite) favors L0 via its CID22-family
strength; the gate panel favors t2_s4003 via dynamic range. NOTE the gate
panel runs on the SDR 944 dial grid — an SDR-route instrument applied to
HDR-route bakes — so both readings are cross-domain and neither is a clean
HDR product gate; the honest state is BOTH tables side by side, no
recommendation. **The freeze recommendation is WITHDRAWN pending the user's
read**; the freeze remains user-gated.

**Column semantics (verified from bytes, per the user's warning):** neither
leg's `human_score` is human — the t1 leg's is the cvvdp-mix (half ssim2 by
construction; range [0, 0.9999], `zensim_score` column all-null) and the t2
leg's is era-B zensim/100 (range [−2.34, 0.956] — the negative tail is the
fingerprint). Every G-HF band, anchor, and "target" in this wave is therefore
METRIC-derived; computations were internally consistent, and all labels in
this doc now carry that reading. The picker-lineage axes' `human_score` is
ssim2-derived (the axis definition; peer provenance states it).

## PEERS EVERYWHERE — status (user directive: all graphs, no model skipped)

Peer rows (ssim2/butteraugli/cvvdp/iwssim) now cover **8 of the board's rank
axes** from stored data only: cid22/kadid/tid/aic3/konjnd (refmetrics
per-pair tables) + imazen26/nonphoto/hfnlproxy (fill4+hqfill+avifgap sidecar
join through the identity-gated slice-selection reproduction; coverage
71%/71%/100%). Notable: on hfnlproxy the classics collapse (butter 0.507,
cvvdp 0.434, iwssim 0.327) — peer context for the HF-NL weak zone. Peers
carry `per_pair` (scatter matrix) and ride every rank-driven chart.
**Remaining, registered as scoring jobs (stored data does not exist):**
csiq / live / aic4 / sdr25 axes, the dial-grid ladder (peer dial curves) and
the corruption grid (peer ordering) — each needs (ref,dist) pair
reconstruction + `zenmetrics batch` (ssim2/butter GPU, cvvdp CPU).
