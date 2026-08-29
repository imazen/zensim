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
