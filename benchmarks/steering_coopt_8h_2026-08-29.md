# 8-hour steering co-optimization program — REGISTERED 2026-08-29 ~05:0xZ (pre-fit)

User directive (verbatim anchors): 8 hours, nomad cluster, "expand your
dataset as needed and experiment with improving and cooptimizing map,
secant, and loop steering", "2 and 3 shot approaches matter the most",
"buttloop jxl to be improved upon with zensim substituted", "zenjpeg can
afford 2 or more shots", "freeze of the best candidate for sdr and hdr
first, as Profile C", "determine how much precision and accuracy matter
here, for this task, and if we are solving the right problem in RD",
"keep a readable artifact up to date and avoid blocking questions".

## Phase 0 (hour 0) — freezes + infrastructure
- **P0.1 Profile C (SDR) := north-anchor** (W10L9PH_s4004_packed, sha
  61ebc456…) — user-authorized profile freeze; embedded weights swap +
  pinning test + docs.
- **P0.2 Profile CHdr := aurora-anchor** (HDR944_L1T1_s4005_hfpack, sha
  0a437d99…) — the HDR analogue (BHdr-parallel naming), same mechanics.
- **P0.3 Research artifact live** (claude.ai artifact, updated per phase).

## The question set (frozen)
- **Q1 (2/3-shot supremacy per codec):** which {seed, secant/controller,
  map} composition minimizes k2/k3 |err| and bytes on jxl / avif /
  zenjpeg (2+ shots sanctioned)? Arms per codec: blind vs fitted-seed;
  own-map vs river-lantern-map (split-role) vs no-map; existing
  controllers only (no new update rules this window).
- **Q2 (beat buttloop):** jxl beats-butter harness re-run with the zensim
  arm = Profile-C-frozen scorer (+ best map from Q1); bar = the committed
  butter-loop reference cells at equal encode budget. Gates: match or beat
  byte-parity target-hit rate.
- **Q3 (co-optimization additivity):** are seed + map gains additive or
  interacting? Factorial read on the Q1 grid.
- **Q4 (precision/accuracy + right-problem):** decompose k3 error into
  scorer-calibration bias (peer-anchored honest deltas), map allocation
  error, controller residual. Determine where added precision stops
  paying against the ±2 product tolerance and the 1-pt dial quantum.
  THEN the RD-framing question: |err|-to-target is nearly saturated
  (avif k3 0.180) — test whether the REAL objective should be
  bytes-at-achieved-quality + cross-codec consistency (same target ⇒
  same truth across codecs), measured from the same cells.
- **Q5 (dataset expansion, nomad):** dense per-codec (image × param)
  encode+score censuses — ~150-200 sources × dense ladders × 4 codecs ×
  {zensim-C, ssim2, butteraugli} — targeting the measured jpeg/webp
  top-zone truth gap; lands as a loop-proxy corpus v2 + honest-anchor
  refit + fitted per-codec seed tables (S1-style, the svt precedent).

## Compute plan
- LOCAL (serialized): loop harness arms (avif ctrl h3 + split-role build
  + R0 gate + anchor-lantern-on-avif; jxl beats-butter zensim arm;
  zenjpeg 2/3-shot census via its search_target owner).
- NOMAD (8 nodes ready: r7900x/i265/r3500 always-on + 4 intermittent +
  dev): the Q5 census fleet job. Sanctioned path: existing fleet job
  machinery; no hand-rolled orchestration.
- Freezes/reports never wait on fleet completion; artifact updates at
  each phase boundary.

## Non-goals this window
New controller math; trainer surgery (map-aware loss stays registered
owner work); any Profile B default flip (B remains shipped default);
publishes.

## P0.1/P0.2 EXECUTED (2026-08-29 ~05:2xZ)
Profile C (SDR) := north-anchor (`c_sdr_purity944_2026-08-29.bin`,
149,343 B, sha 61ebc456…) — fn `mlp_bake_c_purity944`, pinning test
updated, width test holds (944/667). **NEW `ZensimProfile::CHdr`** :=
aurora-anchor (`c_hdr_l1t1944_2026-08-29.bin`, 180,195 B, sha 0a437d99…,
944/697) — additive variant (enum is non_exhaustive), BHdr-parallel,
BHdr remains shipped HDR default. 8/8 profile tests green incl. both
sha pins + identity + end-to-end folded944.

## Hour 1-2 log

**Q5 fleet is LIVE on 6 LAN nodes** (r7900x/i265/r3500/r5900xt/r5600g/i134;
dev+wsl excluded to protect the local loop lane). Bring-up found and fixed
three real infrastructure defects, each caught by a designed gate:
(1) stale `:exec` image — capability gate excluded all 61,812 jobs
(anti-wedge invariant); fixed by local musl rebuild + push. The jxl
feature is OMITTED from the new image: **sibling drift** — zenjxl@e79179e
no longer compiles against jxl-encoder main (`api::ErrorClass` moved) —
so the 13,311 zenjxl cells wait for the sibling fix (flagged, not-mine);
48,501 jpeg/webp/avif cells proceed. (2) `source_fetch` failures — the
executor resolves `$ZEN_CORPUS_PREFIX/<image_path>` VERBATIM, and plan
cells carried absolute local paths; re-declared with basenames (q5c).
(3) docker socket + root-owned buildx state — image builds ride
`sudo docker` with the user config (the established lane). Smoke gate
PASSED on q5c (done rows, real output_shas, encoded blobs persisting)
before any scale-up.

**Q1-avif SETTLED (same-binary tri-arm):** scalar 0.392/19, own-map
0.291/18 (median for a hit — a trade, not a win), pair 0.409/18.
**No map arm beats the scalar loop on avif.** With the earlier
within-binary result (north-anchor 0.180/24 vs gray-tower 0.336/23),
the avif ship-shape is: Profile-C scalar loop, no map. SUBSTRATE
REGRESSION FLAG: concurrent zenavif encoder changes moved the scalar
census 0.180→0.392 between binaries — loop targetability degraded by
those changes; flagged to the owning lane.

**In flight:** Q2 beatbutter (north-anchor, fresh substrate; G-BB1
substrate gate will speak first) + zenjpeg k4/k5 census extension.

## Hour 2-3: Q2 + zenjpeg verdicts

**Q2 (beat-buttloop) — DECIDED on fresh same-substrate cells, north-anchor
both arms:** inner zensim loop (h3, k3) med |err| **0.404 (23/27)** vs the
buttloop-structure outer (score-target re-encode search, zensim judge)
**1.739 (15/27)** at 4 encodes — a 4.3× error advantage at equal-or-lower
budget (inner compares are cheaper than full re-encodes). k2 inner 0.817
(19/27) also beats outer. bin8 ≡ bin1 BYTE-identical (binned attribution
integration is free); clamp 2.5 vs default 2.0 = +1 hit at equal median
(marginal, not adopted). The committed 2026-08-07 G-BB4 already held for
gray-tower; today's data extends it to the frozen Profile C on the current
substrate: **the butter loop's structure is the dominated part — "buttloop
with zensim substituted" loses to the native inner controller.**

**zenjpeg "2 or more shots" — the floor is the KNOB, not the controller:**
arm B (fitted seed) k2 1.905/14, k3 1.383/17, **k4 1.096/19, k5 1.064/20**
— a hard plateau at ~1.06, which is zenjpeg's integer-q rung spacing at
these targets (the addressability audit's effective-quantum finding,
manifesting as a shots-invariant error floor). More shots cannot beat the
rung; the lever is codec-side sub-q granularity (per-block quant scaling),
registered as owner work. Judge-era caveat: zenjpeg census judges via its
git-pinned zensim (self-consistent within-census); Profile-C re-pin is a
registered follow-up.

## Q4 VERDICT — precision, accuracy, and whether RD target-hitting is the right problem

All terms measured this window (native units, k3-class budgets):

| error term | measured size | vs product quanta (dial 1.0, tolerance ±2) |
|---|---|---|
| controller residual (jxl inner) | 0.404 med | 2.5× under quantum |
| controller residual (avif) | 0.29-0.39 | ~3× under quantum |
| controller residual (zenjpeg) | **1.06 floor** | AT quantum — and it is the KNOB's rung spacing, not controller error |
| controller residual (svt-HDR, S1) | 1.51 | inside tolerance |
| scorer-truth calibration (top anchors) | **0.9-2.2** per codec | 1-2 quanta — THE BINDING TERM |
| cross-codec truth-span divergence | webp top 90.8 vs jxl 97.0 ssim2 | the "same target, same quality" contract is the gap |
| map steering effect on \|err\| | negative or neutral everywhere measured | maps move BYTES (−1..−5.4%), at quality cost |

**Answers.** (1) *How much precision matters:* controller precision is
SOLVED — every codec's loop sits at or under the dial quantum, and
0.1-point score reporting is over-precise against the ±2 product
tolerance and ssim2's own ~±0.3-0.5 per-cell noise. Further controller
refinement pays ≈ nothing; zenjpeg's floor is a codec-side knob-quantum
problem (sub-q per-block scaling), not a loop problem. (2) *How much
accuracy matters:* accuracy — calibration to model-independent truth and
its CONSISTENCY across codecs — is 3-5× the controller residual and is
now the whole ballgame for scoring quality. (3) *Are we solving the right
RD problem:* partly no. Target-|err| is saturated below useful quanta;
the objectives that still buy product value, in order: **(a) cross-codec
calibration consistency** (truth-anchored refit — exactly what the Q5
census feeds), **(b) bytes-at-achieved-truth** (steering/maps re-gated as
BD-rate-at-fixed-truth with paired ssim2/butter CIs — the measured maps
already cut bytes but were judged on the wrong axis), **(c) knob
granularity floors** (zenjpeg). REGISTERED PROPOSAL for the next steering
study: replace the \|err\| gate with "bytes ≤ baseline − X% at
truth-quality non-inferior (paired CI)" — the maps' measured behavior
(byte savings at small quality cost) becomes a candidate WIN under the
right objective instead of a loss under the wrong one.

## Hour 3-4: Q3 additivity + the parity inversion — the co-optimization picture completes

**Q3 (seeds compose):** avif fitted S1 seeds (trace-derived, t70→cq129 /
t80→106 / t88→46, committed in zenavif): **k1 (2 encodes) seeded 0.656
(19/27) vs blind 0.881 (17/27)** — +26% median at the tightest budget,
≈ blind k2. With svt (17.6→3.3), zenjpeg (+46-48%), and jxl's committed
seed work, the family law is measured everywhere: **fitted seeds are the
2-shot lever; the inner controller is the 3-shot lever; they compose.**

**The parity inversion (the window's sharpest finding):** re-scoring the
map arms at self-judged QUALITY PARITY (|Δachieved| ≤ 0.5 vs scalar):
avif own-map **+2.16%** bytes, avif pair **+3.06%** — the naive "−5.4%
byte savings" was under-shooting targets, not efficiency. jxl's pair, by
the same analysis: **−0.41% at parity AND better error** — it alone
passes the re-gated objective. **Ship-shape by codec: jxl =
anchor-lantern (scorer + companion map); avif = Profile-C scalar + S1
seeds; zenjpeg = seeded search (floor = knob quantum, codec-side lever
registered); svt-HDR = S1-seeded qp staircase (already wired).**
Truth-judged (peer-CI) confirmation of the parity numbers rides on the
census metric stage (armed, auto-declares on encode convergence).

## Extension hours (user directives): repos synced, aom Zq SHIPPED, avif rule falsifications

- **All repos synced** — the other machine's zenjxl fix landed (compiles);
  zenmetrics patched for zenjpeg's new unpublished zenanalyze 0.2.0 dep;
  jxl-capable executor image pushed; fleet cycled — all 61,812 cells
  claimable (6 nodes).
- **aom backend Zq (user directive, premature ruling superseded):**
  `zenav1-aom/crates/aom-target` — the svt ruling pattern taken further:
  encoder AND judge dependency-injected (`trial(qindex)` closure; crate
  has zero codec/metric deps; pure-Rust whole-frame encoder swaps in
  later, no loop change). Census harness injects aomenc/aomdec CLI
  (matrix-roundtrip-gated) + the frozen Profile-C judge. **Phase-A census
  CLOSED: blind k2 3.497 (9/27), k3 1.476 (19/27)** — comparable-judge
  numbers on the instrument where "ruled premature" stood this morning.
  All three AV1 backends (rav1e-class via zenavif, svt, aom) now have
  measured Zq loops. Better-than-ssim2-integration criteria met:
  decoded-pixel Profile-C judging + seeded-capable inner search + census
  discipline vs an outer ssim2 re-encode bisection.
- **avif steering-rule bug hunt:** mechanism found (arith-mean-after-clamp
  renorm drift, screen-concentrated), fix arm FALSIFIED (zerosum 0.491/17
  vs legacy 0.291/18) — per-SB redistribution of any rule fights the
  controller on screen content; the avif map lane needs TRAINED maps
  (registered owner work). Separate bug note: scalar t70 unreachable on
  screen crops (achieved floors ~50) on the current substrate.
