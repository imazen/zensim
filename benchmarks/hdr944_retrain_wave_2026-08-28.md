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

## PEERS COMPLETE — 4 peers × ALL 12 board rank axes (2026-08-28)

Scoring closed the gaps: cvvdp CPU locally (sanctioned rung); ssim2/butter/
iwssim on the wsl RTX 5070 via the baked `exec-gpu-cuda13-6d4f9963` container
(GPU-only rule; first-cell AWGN-monotone gate; 11 runs rc=0, full rows;
LIVE via the PNG mirror). All TSVs persisted in `reports/refmetrics/`.

| peer | cid22 | kadid | tid | csiq | live | aic3 | aic4 | konjnd | sdr25 | imazen26 | nonphoto | hfnlproxy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ssim2 | 0.889 | 0.813 | 0.846 | 0.905 | 0.960 | 0.797 | 0.913 | 0.479 | 0.958 | 1.0* | 1.0* | 1.0* |
| butteraugli | 0.741 | 0.606 | 0.668 | 0.844 | 0.871 | 0.707 | 0.866 | 0.358 | 0.884 | 0.670 | 0.684 | 0.507 |
| cvvdp | 0.821 | 0.834 | 0.853 | 0.896 | 0.946 | 0.792 | 0.891 | 0.048 | 0.861 | 0.803 | 0.778 | 0.434 |
| iwssim | 0.784 | 0.850 | 0.779 | 0.921 | 0.957 | 0.773 | 0.953 | 0.186 | 0.950 | 0.692 | 0.669 | 0.327 |

(*self-target: those axes' targets are ssim2-derived — 1.0 by construction,
footnoted in the row, never a measurement.) Board regenerated (304 rows, all
gates PASS); peers ride the scoreboard, heatmap, band tables and scatter
matrix. Remaining chart surfaces without peer data: the dial-grid ladder and
corruption grid (peer curves need scoring those grids' encodes — registered,
not run). /mnt/v is NOT shared between dev and wsl (verified by sharecheck —
overlapping content is historical sync); data shipped with the job.

## PEER GRID SURFACES COMPLETE (2026-08-28) — dial + corruption, all 4 peers

Scored from the persisted grid pixels (dial: 6,487 ladder pairs; corruption:
2,016 gate PNGs; cvvdp CPU local, GPU trio on the wsl 5070 baked container;
the first GPU pass was starved by rsync's `--files-from` non-recursion —
caught by rows=0, re-shipped size-verified, re-run):

| peer | dial mono (per-ladder) | corruption pass_q10 | pass_q20 |
|---|---|---|---|
| iwssim | **0.995** | 0.098 | 0.168 |
| cvvdp | 0.985 | 0.213 | 0.362 |
| ssim2 | 0.978 | 0.201 | 0.345 |
| butteraugli | 0.882 | **0.521** | **0.618** |

The two instruments dissociate cleanly: iwssim is the best LADDER orderer and
the WORST corruption detector; butteraugli inverts (noisiest ladder, 2.5×
corruption-gate lead) — the measured, on-board version of the
"butter wins corruption-gate" finding that motivated the corruption HEAD.
Peer rows now carry rank (12 axes) + per_pair (scatter) + dial.curves +
corruption blocks — every gauntlet chart surface renders peers. Board 304
rows, all gates PASS.

## HDR-ROUTE GATE PANEL — registered before computing (2026-08-28, the user's "name the next lever" hold)

The E.4-vs-gates lens disagreement is an INSTRUMENT artifact: the gate panel
runs on the SDR dial grid. This panel replaces it for HDR candidates with the
same gate semantics computed on HDR-route content.

**Data (frozen):** the mc944 **t1 VAL leg** (22,860 cells; census-clean,
val-split, never trained by any candidate), all three codecs
(zenjxl / zenav1-svt / jpeg-gainmap), per-rendition q-ladders. Forward =
`predict_features_with_bake` on the PACKED artifacts.

**Metrics + bars (frozen; the SDR panel's semantics transposed):**
- HG1 dynamic range = min over codecs of (pooled p50 at ladder top − p50 at
  ladder bottom); PASS ≥ 40 (usable swing on every codec).
- HG-mono = per-(rendition,codec) fraction of non-decreasing adjacent steps;
  PASS ≥ 0.93 (the SDR G3 bar).
- HG-tied = fraction of adjacent steps with |Δp50-scale| < 0.05 per ladder;
  PASS ≤ 0.05.
- HG-reach = pooled p50 span across the full grid (reported).
Candidates: the two freeze contenders (incumbent hfpack, t2_s4003 hfpack) +
t1_s4003 hfpack for context. Verdict rule: if ONE contender passes all three
gated rows and the other does not, the tie is broken; if both pass, E.4
stands as the registered tie-break; if both fail, neither freezes and the
gentle-HF arm registers next.

## HDR-ROUTE PANEL RESULT + REGISTERED AMENDMENT (2026-08-28)

**Registered-form result: ALL THREE FAIL HG1≥40** — and the failure is the
BAR's, not the models': the TARGET p50 swings are jpeg-gainmap **15.27**
(32.3→47.6 — the gainmap format ceiling, measured), zenjxl **27.24**
(68.3→95.5), zenav1-svt 94.20. An absolute ≥40 bar demands more range than
ground truth possesses on two of three codecs — a miscalibrated transposition
of the SDR g1.

**AMENDMENT (recorded, justification = the target-swing measurement above):**
- HG1-F swing FIDELITY: 0.65 ≤ model_swing/target_swing ≤ 1.5 per codec
  (a dial must track the true range — under- AND over-swing both fail).
- HG-mono ≥ 0.93 computed only on codecs whose target swing ≥ 25 (svt, jxl);
  on jpeg-gainmap the true ladder steps are sub-noise (15 points over the
  whole grid) and a mono read there measures noise.

**Amended verdict:**
| candidate | fidelity gm/jxl/svt | mono svt/jxl | result |
|---|---|---|---|
| **incumbent hfpack** | 1.14 / 0.93 / 1.01 | 0.996 / 0.995 | **PASS** |
| t2_s4003 hfpack | **1.90** / **1.44** / 0.95 | 0.998 / 0.997 | FAIL (over-swing ×1.9 gm, ×1.4 jxl) |
| t1_s4003 hfpack | 1.17 / 0.87 / 1.00 | 0.975 / **0.892** | FAIL (jxl mono) |

**The tie breaks toward the INCUMBENT, coherently with E.4**: the SDR panel's
g1 preference for t2 rewarded over-swing on an SDR instrument; on the HDR
route with fidelity semantics, the incumbent is the only faithful dial. The
amended-verdict acceptance is the USER's (the registered form said all-fail
⇒ gentle-HF next; this amendment is a recorded bar correction, not a silent
re-gate).

## GENTLE-HF ARM (GH1) — registered 2026-08-27 BEFORE any fit (the user's hold: "Hold for gentle-HF retrain arm")

Skeptical premise, stated up front: t2's apparent HF lead was ALREADY
falsified by the shared-band paired bootstrap (Δ −0.0093, CI excludes 0), so
this arm tests whether ANY era-B HF admixture at low weight adds HF skill
without t2's failure modes (over-swing ×1.9/×1.4, unbounded −387 emissions).
Its null result is as decisive as its positive: if GH1 fails, every retrain
lever short of the human pairwise study is exhausted and the incumbent
freeze case is complete.

**Arm (frozen):** the L1-T1 recipe verbatim (3 groups: hdrmc t1 train 1.0 /
t1_train_hf 1.0 / t1 val) + a 4th group `hdrmc_hf2:
hdr944-retrain-2026-08-28/t2_train_hf.parquet` at **train-weight 0.3, val 0**.
Scale note, recorded: the t2 HF leg's human_score is era-B/100; in the HF
band (≥0.90) both target scales occupy ≈[0.9,1.0] and pair sampling is
per-group (rank loss within group), so no cross-scale pair is drawn. Seeds
{4003,4004,4005}; epochs/pairs/decay/width identical to L1. Pack = the
HF-anchored parity invocation (anchor_hf_t1, verify 944-CID22).

**Gates (frozen):**
1. Amended HDR-route fidelity: 0.65 ≤ swing ratio ≤ 1.5 per codec; mono
   ≥ 0.93 on svt+jxl.
2. G-OUT v2 on-route (cid22) + clause D on every axis.
3. G-HF addressability: val HF band p50 ≥ 90 AND frac(≥88) ≥ 0.5.
4. HF shared-band paired-Δ vs `HDR944_L1T1_s4005_hfpack` (aligned rescore,
   B=5000, seed 11): GH1 DISPLACES the incumbent only if Δ > 0 with CI
   excluding 0 AND gates 1-3 pass; ties → incumbent stands.
Selection among own seeds: freeze_check E.4. Driver:
`scripts/hdr_gh1_arm.sh` (queued behind the SDR seed extension —
serialized heavy jobs).

### GH1 RESULTS — mechanism real, mono cost fatal; incumbent stands (2026-08-28)

3/3 seeds trained/packed/harvested clean. Frozen gates:

| seed | route fidelity gm/svt/jxl | jxl mono (bar 0.93) | G-HF p50 / frac≥88 | HF shared-band Δ vs incumbent |
|---|---|---|---|---|
| GH1_s4003 | 1.15 / 1.00 / 0.75 ✓ | **0.852 ✗** | 93.80 / 0.984 ✓ | **+0.0276 [+0.0188,+0.0363]** |
| GH1_s4004 | 0.72 / 1.00 / 0.80 ✓ | **0.872 ✗** | 93.83 / 0.979 ✓ | **+0.0266 [+0.0147,+0.0389]** |
| GH1_s4005 | 1.08 / 0.99 / 0.83 ✓ | **0.859 ✗** | 93.84 / 0.975 ✓ | **+0.0272 [+0.0190,+0.0353]** |

**Verdict under the frozen displacement rule: GH1 does NOT displace — gate 1
fails on every seed (jxl mono 0.85-0.87 vs incumbent 0.996).** The science:
the era-B HF admixture at weight 0.3 adds REAL, seed-consistent HF-band rank
skill (+0.027, all CIs exclude 0) and full addressability, at the isolated
cost of jxl ladder monotonicity — the dose-response across the family is now
measured (t2@1.0: fidelity broken ×1.9; GH1@0.3: fidelity fine, mono broken,
HF +0.027; incumbent@0: all gates pass). Per this arm's own registration,
the retrain levers are exhausted and **the incumbent
`HDR944_L1T1_s4005_hfpack` freeze case is COMPLETE** — E.4 selection, G-HF,
amended route panel (only faithful dial), 6/6 author panel, HF CI reversal
of t2, G-OUT v2 on-route + bounded everywhere, and now two falsified
challengers. A micro-dose arm (0.10-0.15) chasing the +0.027 without the
mono cost is the one remaining registered-able retrain idea; running it is
the USER's call, as is the freeze.

## MICRO-DOSE ARM (GH2) — registered 2026-08-28 BEFORE any fit (user call)

The user chose one micro-dose arm before freezing. **Arms (frozen):** the
GH1 recipe with the hf2 group's train-weight at **0.10 (GH2a)** and **0.15
(GH2b)** — bracketing the dose inside the user's range; seeds {4003,4005}
per dose (4 fits). Stems `HDR944_GH2a_s*` / `HDR944_GH2b_s*`. Pack (HF
anchor) + harvest identical to GH1.

**Gates + displacement rule (frozen, unchanged from GH1):** amended route
fidelity + jxl/svt mono ≥ 0.93 + G-OUT v2 on-route + G-HF + HF shared-band
paired-Δ vs `HDR944_L1T1_s4005_hfpack` — displacement requires gates 1-3
PASS and Δ > 0 with CI excluding 0. Any seed passing all → E.4 among
passers picks the freeze proposal; 0/4 passing → the incumbent freezes with
the dose-response fully measured (1.0 / 0.3 / 0.15 / 0.10 / 0).

### GH2 RESULTS — 0/4 displace; dose-response complete; teacher EXONERATED (2026-08-28)

| cell | jxl mono (bar 0.93) | fidelity | G-HF | HF shared-band Δ |
|---|---|---|---|---|
| GH2a_s4003 (0.10) | 0.917 ✗ | all ✓ | ✓ | +0.0141 [+0.0067,+0.0210] |
| GH2a_s4005 (0.10) | 0.867 ✗ | all ✓ | ✓ | +0.0277 [+0.0199,+0.0353] |
| GH2b_s4003 (0.15) | 0.921 ✗ | all ✓ | ✓ | +0.0123 [+0.0055,+0.0187] |
| GH2b_s4005 (0.15) | 0.866 ✗ | all ✓ | ✓ | +0.0257 [+0.0182,+0.0329] |

Full dose-response on jxl mono: **0 → 0.996 PASS; 0.10 → 0.867-0.917;
0.15 → 0.866-0.921; 0.30 → 0.852-0.872; 1.0 (t2) → fidelity broken.** The
HF gain is real at every dose; the mono cost appears at the smallest dose
and never clears the bar. **Mechanism isolated by measurement:** BOTH
teachers are themselves ~0.995-0.999 monotone on every codec's ladders
(era-B jxl 0.998, cvvdp-mix jxl 0.995) — the damage is NOT inherited label
noise but a target-mixture/optimization artifact, invisible to epoch
selection (val = cvvdp-mix only). Closing it would need a mono-aware
constraint or selection signal — a NEW mechanism, not a dose.

**HDR lane verdict: under the frozen displacement rule, nothing displaces.
The incumbent `HDR944_L1T1_s4005_hfpack` freeze case is COMPLETE** with two
falsified challengers (t2, GH-family at 4 doses), the amended route panel,
6/6 author panel, G-OUT v2 on-route + bounded, G-HF, and now a
teacher-exonerating mechanism record. The freeze remains the USER's call
(overnight — logged, will be presented after 08:00 Denver).

## ★ FREEZE EXECUTED (user, 2026-08-28T08:53Z): HDR944_L1T1_s4005_hfpack = HDR CANDIDATE-OF-RECORD

User approved after the complete case (E.4, G-HF, amended route panel, 6/6
author panel, t2 + GH-family falsified, bounded, G-OUT on-route).
Artifact: /mnt/v/output/zensim/bakes/hdr944-2026-08-27/HDR944_L1T1_s4005_hfpack.bin
sha256 0a437d99… — packaged (HF-anchored spline, prune identity bit-exact on
20,769 anchors, M3a measured, G-EXT inherited-by-rank-invariance).
Candidate-of-record ONLY; shipped default remains B; Krasula-form human
study = registered future lever.
