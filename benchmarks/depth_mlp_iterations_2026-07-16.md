# Depth-MLP vs B — iterations 1→2 (2026-07-16)

Pursuing the one **measured** open lever from the SSIM-explosion investigation:
a 2-hidden-layer MLP breaks the CID22↔non-photo trade a linear head (B) can't.
Done via the **Rust trainer** (`zensim_mlp_train --manifest`), not the Python
`blend_lib` that first found it — the candidate is reproducible + provenance-gated.

**Status: candidate. Profile B is NOT swapped** (`include_bytes!` untouched); a
swap is user-gated.

## The verdict (depth_v2, 3-seed mean, held-out)

| corpus | metric | depth_v2 | B | Δ |
|---|---|--:|--:|--:|
| CID22 | pooled | 0.8876 | 0.8764 | **+0.011** |
| non-photo | pooled | 0.9616 | 0.8606 | **+0.101** |
| AIC-3 | pooled | 0.8039 | 0.7774 | **+0.027** |
| AIC-4 | pooled | 0.9300 | 0.8906 | **+0.039** |
| KonJND | pooled | 0.4967 | 0.5466 | −0.050 |
| **HF near-lossless** | **per-ref** | **+0.948** | **+0.488** | **+0.460** |
| HF near-lossless | pooled | 0.157 | 0.614 | −0.458 |

**depth_v2 beats B on 4/5 pooled held-out corpora**, and wins the HF codec-dial
metric (per-ref) decisively. The two B-wins are both the known-hard regimes:
KonJND PJND (G5 Pareto limit — both bakes fail the 0.70 floor) and cross-image
near-lossless *scale*.

## The HF "crater" is a pooled-metric artifact (the session's recurring theme)

Iteration 1 (no HF training) cratered HF: pooled 0.102, **per-ref −0.924, 100%
of references ranked BACKWARDS** — the §8.39 near-lossless inversion. Iteration 2
added `hf_nearlossless_train` with **`:withinref`** (draw RankNet pairs within
one image, not across). Result:

| | HF pooled | HF per-ref | %backwards |
|---|--:|--:|--:|
| depth_v1 | 0.102 | −0.924 | 100% |
| depth_v2 | 0.157 | **+0.948** | **0%** |
| B | 0.614 | +0.488 | 21% |

Within-ladder pairing **completely fixed the within-image ranking** (−0.924 →
+0.948, better than B). The pooled SROCC stays low because it measures
cross-image near-lossless *scale* — all-near-lossless, genuinely ambiguous, and
not what a codec binary-searching one image needs. The near-lossless ladder
moves ~0.92 ssim2 pts within-image vs ~6 between-image, so pooled is dominated by
the between-image scale the model gets wrong. Same pooled-vs-per-ref confound as
the IW-pooling work, the AIC-3 0.79/0.93 split, and the r7 HF finding. `bake_verdict`
reports both precisely because of this.

**KonJND** also improved under `:withinref` (+0.020, 0.477 → 0.497) — PJND is a
per-image threshold, so cross-image pairing had been teaching between-image scale.

## Recipe (reproducible)

`zensim_mlp_train --manifest zensim/weights/manifests/depth_v2.toml --seed N`.
- **arch** 372 → 128 → 64 → 1, LeakyReLU(0.01), identity out; winsor_p99
  auto-transforms; PCHIP dial spline.
- **groups** safesyn / cid22_train / kadid / tid (rank) + konjnd_dense
  (**withinref**) + hf_nearlossless (**withinref**) + bigcodec-120k-slice +
  bigcodec_val + kadis-60k-slice.
- vs the Python blend candidate: Rust arch is 128→64 (not 128→128), RankNet loss
  (not smooth_l1), + konjnd/HF withinref (blend had neither). The *finding*
  (depth breaks the trade) reproduces; the bytes are not claimed identical.

## Infrastructure (owner-extensions, no duplication)

- `bake_verdict --json` — machine-readable panel (the dashboard never parses the report).
- `scripts/v_next/bake_compare_dashboard.py` — theme-aware comparison; pooled +
  **per-ref** panels. Every number from `bake_verdict`.
- manifest group `within_ref` / `loss_mode` fields.
- trainer logs `withinref`/`loss` per group (closed a serde silent-ignore footgun;
  confirmed `withinref=true` on both ladder groups on iteration 2's first run).

Dashboard: `/mnt/v/output/zensim/depth-iter/dash/depth_2026-07-16.html`
(browser: http://localhost:3300/zensim-depth-dashboard/depth_2026-07-16.html).

## ⚠ THE DECIDING GATE: depth wins rank, FAILS the dial (two-panel verdict)

The rank wins above are only half the eval. The codec-dial panel (G1 range /
G3 monotonicity on the densified multi-codec q-sweep) is the other mandatory
half — and it is where depth loses decisively:

| bake | monotonicity (G3 ≥ 0.93) | dial range p5/p95 (G1) |
|---|--:|--:|
| B | **0.979** ✅ | **13.6 / 99.7** ✅ |
| A | 0.978 ✅ | 16.7 / 94.5 ✅ |
| depth_v2 | **0.550** ❌ | **−17.7 / 8.5** ❌ |
| depth_v3 | 0.526 ❌ | −18.8 / 9.5 ❌ |

depth_v2 has **45% dial inversions** (score goes *backwards* as codec quality
rises on nearly half of adjacent-q pairs) and **no usable 0–100 range**. For the
primary use case — "user types zensim 85, codec binary-searches the q that hits
it" — this dial is unusable. A monotone output spline cannot fix it: the raw
output is non-monotonic *in codec quality*, so there is no monotone remap.

**So depth_v2 is NOT a drop-in B replacement.** It is a decisively better
*ranker* (4/5 pooled + HF per-ref) with a broken *dial*. In the SOTA_TRAILS
framing it is a **rank-trail** candidate (like `PreviewV0_5Compression`), not the
dial-bearing Profile B. B and A are deliberately dial-optimized (linear /
masked-monotone + spline) at a rank cost; depth is the opposite trade.

This is exactly the "a bake can win the rank panel and be a broken dial" case
`CLAUDE.md`'s TWO-PANEL rule exists for — caught by the panel, not by a
rank-only view.

## ⚠ RETRACTION (2026-07-16, same day): "fundamental" was overclaimed from n=1

The "CONVERGED VERDICT" below declared the rank↔dial tension **fundamental** from
a SINGLE maximally-constrained experiment (depth_v4). That is the same
premature-conclusion error this session kept catching elsewhere — two endpoints
do not define a frontier, and I never swept the constraint or used the right
tool. Specifically depth_v4:

- used `monotone_strict`, which **DELETES the 72 non-sign-safe features** — a
  naive, lossy way to get monotonicity (all-positive weights + fewer inputs);
- set `monotonicity_reg = 1.0`, the **maximum**, never 0.05/0.1/0.3;
- so it measured the *most-constrained* corner and called the line to it
  "fundamental".

**The forgotten solutions** (monotonicity is a solved problem):
1. **Partial monotonicity via `monotone_pin_during_training` (KEEP-72)** — pin
   the 300 sign-safe features, keep the 72 free ones *expressive*. The trainer
   already has this; depth_v4 used the delete-them flag instead.
2. **A constraint-strength sweep** — the rank↔dial trade is a *frontier*, and a
   point on it (dial passes at small rank cost) would falsify "fundamental".
3. **Monotone-by-construction architectures that don't lose expressiveness** —
   UMNN (integrate a positive-derivative net), deep lattice networks, Sill
   min-max monotone nets, mixed-activation constrained monotonic nets. These are
   universal approximators *for monotone functions*; depth_v4's all-positive +
   dropped-features method is the lossy special case.
4. **Two-head decoupling** — expressive rank head + monotone dial head on a
   shared trunk (the dial being monotone encodes a *true* constraint, so it
   costs nothing real).

**depth_v5 is mapping the actual frontier** (KEEP-72 + reg ∈ {0.0, 0.1, 0.3}).
Verdict below is SUSPENDED pending that measurement. Do not cite "fundamental".

### depth_v5 frontier — reg0p0 (reg=0 corner) result + MECHANISM (2026-07-16)

reg0p0 (KEEP-72, `monotonicity_reg=0.0`, 160 epochs, seed 13) is the **worst
corner** (no soft monotonicity penalty). Its held-out panel **collapsed** — far
worse than depth_v4, and this pinned down *why* depth doesn't beat B:

| bake | held-out CID22 (MOS) | train cid22 (ssim2) | dial mono | note |
|---|---|---|---|---|
| B (linear) | ~0.876 | — | monotone | ship |
| depth_v4 (strict, reg=1.0) | 0.7247 | — | — | constrained corner |
| v5_smoke (KEEP-72, short run) | 0.6992 | — | — | sane |
| **depth_v5 reg0p0 (KEEP-72, 160ep, reg=0)** | **0.1185** | **0.985** | 0.84 | **collapsed** |

**This is NOT a bake_verdict bug** — `v5_smoke` (same KEEP-72 config, fewer
epochs, same scorer) reads a sane 0.70. The collapse is a training-dynamics +
selection failure, and the mechanism is the valuable part:

1. **Raw output collapsed to a narrow band.** Best-epoch spline fit:
   `anchor pred [20.84, 48.60] target [0, 97.37]` — the net's raw predictions
   span only ~28 units, stretched ~3.5× by the dial spline. Overfit noise inside
   that band gets amplified into rank noise.
2. **Model selection was blind to the collapse.** Every val-selection group is
   **ssim2-anchored AND also trained on** — cid22_train has the *highest*
   val_w=2.0 *and* train_w=1.0; the one held-out val group (`bigcodec_val`) is
   also ssim2-anchored. So `val(geomean3)` peaked at 0.939 (train ssim2-SROCC)
   while held-out human-MOS CID22 sat at 0.12, invisibly. This is the
   "held-out val group REQUIRED — train==val selection hides collapse" hazard.
3. **Capacity × monotone constraint = ssim2-overfit away from MOS.** depth_v2
   (unconstrained) trains on the same ssim2 targets and gets CID22-MOS 0.88 — its
   ssim2-shaped ranking happens to align with MOS. The monotone constraint
   removes that natural solution and the high-capacity net spends its capacity
   on an ssim2-fitting solution that maximizes train/val ssim2-SROCC but drifts
   *far* from MOS.

**The insight this yields: B's linearity is REGULARIZATION, not a limitation.**
Low capacity can't overfit the ssim2 training target, so B's ssim2-shaped
ranking stays close to human MOS. Adding capacity *under the dial (monotone)
constraint* buys ssim2-ranking you don't need and costs MOS-ranking you do. To
beat B you need capacity that helps MOS without ssim2-overfit — which requires
MOS training data (we have none beyond the holdouts) or capacity control that
is ~what B already is.

reg0p0 is the zero-regularization corner; **reg0p1/reg0p3 (soft penalty) test
whether regularization pulls the solution back toward B.**

### Full frontier verdict — DECISIVE (all 3 reg points, seed 13)

| reg (`monotonicity_reg`) | held-out CID22 | dial mono | dial G1 (range) | dial G3 (mono≥.93) |
|---|---|---|---|---|
| 0.0 | 0.1185 | 0.842 | ✗ | ✗ |
| 0.1 | 0.2960 | 0.717 | ✗ | ✗ |
| 0.3 | **0.4372** | 0.450 | ✓ | ✗ |
| **B (shipped)** | **~0.876** | monotone | ✓ | ✓ |

The frontier is **monotonic in reg but converges nowhere near B**:

- **Rank climbs with reg** (0.12 → 0.30 → 0.44) — soft ordering pressure
  partially counteracts the ssim2-overfit collapse, but the best point (0.44) is
  **half of B's 0.876**.
- **Dial monotonicity FALLS with reg** (0.84 → 0.72 → 0.45) — counterintuitive
  until you see why: `monotonicity_reg` penalizes *cross-image* training-pair
  disagreement, and satisfying that pushes the 72 free features harder, which
  breaks *within-q-ladder* (dial) monotonicity. So reg buys cross-image rank at
  the cost of the dial. **No reg setting gets both**; G3 fails at every point.
- Every full run also stays collapsed on the other held-out axes (AIC-3 ≤ 0.08,
  non-photo ≤ 0.08, HF near-lossless per-ref negative).

**CONCLUSION (now measured across the frontier, not asserted from one point):**
depth-under-monotone — KEEP-72, any reg — **does not beat B and is not shippable
as a dial**. It collapses (0.12) or at best reaches half of B (0.44) on held-out
MOS, and never clears the G3 monotonicity gate. The retraction was right to
demand the sweep; the sweep confirms depth loses, for the *measured* reasons
above (capacity overfits the ssim2 target away from MOS; the dial collapses; and
model selection on ssim2-anchored val groups is blind to both). This is the
honest end of the "beat B by adding depth" pursuit.

Reports (comprehensive `bake_verdict --html --compare B`):
`/mnt/v/output/zensim/zensim-reports/depth_v5_reg0p{0,1,3}.html`.

### The "good data" hypothesis — TESTED, and it does not beat B either (2026-07-16)

The mechanism above pointed at human-MOS data as the lever to make capacity help
MOS instead of overfitting ssim2. Tested with KonFiG human triplets
(ordered-probit NLL, 541,895 responses). Two findings:

1. **A silent-no-op trainer bug, caught + fixed.** The first run (`depth_v6` =
   depth_v2 plain 2-layer + `--triplet-weight 0.5`) baked **byte-identical** to
   depth_v2 (md5 `b1aecd40`, sha `d8a139a6`): the triplet step lives ONLY in
   `train_mlp_per_sample_alpha_head`, so the plain path loaded the pool, logged
   it, and threw the flags away. A byte-identity check caught it before it
   shipped as a false "triplet beats B." Fixed with a fail-loud guard
   (`4691cf2a`) mirroring the `--monotonicity-reg`/`--mse-weight` guards — the
   REPRODUCIBILITY §5 "adjacent check missing" pattern.

2. **Clean A/B on the arch where triplet fires: it HURTS.** Matched runs on the
   dial-viable `per_sample_alpha + tanh` base (learned α, no monotone_cbc), the
   only difference being `--triplet-weight 0.5`:

   | corpus | psa base | +triplet(0.5) | Δ |
   |---|---|---|---|
   | CID22 | 0.8559 | 0.8456 | −0.010 |
   | AIC-3 | 0.7900 | 0.7645 | −0.026 |
   | AIC-4 | 0.9263 | 0.9152 | −0.011 |
   | KonJND | 0.4805 | 0.2787 | **−0.202** |
   | dial mono | 0.929 | 0.937 | +0.008 |

   Triplet at w=0.5 regresses every rank corpus (KonJND badly). BUT the weight
   sweep {0.1,0.2,0.3} shows the response is **non-monotonic** — w=0.1 lifts CID22
   to 0.8734 (+0.017, ≈B) and AIC-3 to 0.8059 (+0.016) while w=0.3 lifts KonJND to
   0.6424; w=0.2/0.5 hurt. So the earlier "hurts" (w=0.5 only) was premature — the
   good-data signal DOES help at low weight, at a dial-monotonicity cost
   (0.929→0.85). The non-monotonic bouncing (0.846–0.873 around base 0.856) is
   within the single-seed noise band, so a low-weight sweep {0.02,0.05,0.075} +
   seed confirmation is needed before this is a finding, not noise.

**Pareto summary of everything tried this session:** plain depth (CID22 0.890,
dead dial) · per_sample_alpha+tanh (CID22 0.856, viable dial mono 0.929) ·
monotone-depth frontier (collapses) · KonFiG triplet (hurts). **B (CID22 0.876,
full dial) sits on the frontier — nothing tried beats it on both rank AND dial.**
Beating B needs something not yet tried — most likely genuinely more human-MOS
data at scale (KonFiG is 1,220 stimuli), not another architecture or loss on the
data we have. `docs/PLAN_BEAT_A.md` (Claude-authored, not human-confirmed) queues
AIC-3's 420k triplets + SDR25-as-holdout; that regime shift is a user call.

## ~~CONVERGED VERDICT: the rank↔dial tension is fundamental (depth_v4 proved it)~~ — RETRACTED, see above

The natural question — *can depth keep its rank wins AND get a monotone dial?* —
was tested directly. **depth_v4** = depth's data + A/v47's full dial machinery
(per-sample-α head, masked-monotone with the 300/72 sign mask, tanh-pin,
monotonicity-reg 1.0). Result (seed 13; the effect is ~0.15, far above seed
noise):

| | dial monotonicity | dial range | CID22 | non-photo |
|---|--:|--:|--:|--:|
| depth_v2 (unconstrained) | 0.550 ❌ | −17.7/8.5 ❌ | 0.888 | 0.962 |
| **depth_v4 (constrained)** | **0.971 ✅** | **22.8/95.3 ✅** | **0.725** | **0.861** |
| B | 0.979 ✅ | 13.6/99.7 ✅ | 0.876 | 0.861 |

The dial machinery **fixed the dial** (0.55 → 0.971, both gates pass) and in the
same move **destroyed the rank** — CID22 crashed −0.16, non-photo lost its entire
+0.10 advantage. depth_v4 is now *strictly worse than B*: it only matches B's
dial while losing B's rank.

**So the depth advantage and a monotone dial are the same coin's two sides.**
depth_v2's rank wins *come from* the unconstrained 2-layer capacity that also
makes its output non-monotone in codec quality. Constrain that capacity for the
dial (monotone weights, sign mask, tanh-pin) and the rank advantage goes with it.
You get depth_v2's rank *or* depth_v4's dial — the middle ground is B.

**Conclusion.** The depth lever produces a better **ranker**, not a better
**dial metric**. Its home is codec *selection* / RD-loop *ranking* (where per-ref
and pooled rank matter and the 0–100 dial is irrelevant) — a **rank-trail
sibling** in the SOTA_TRAILS framing (like `PreviewV0_5Compression`), NOT a
replacement for the dial-bearing Profile B. B stays the shipped quality-dial
metric. This is the honest, measured end of the "beat B" pursuit: B is not beaten
*as a dial*; it is beaten *as a ranker* by a model that can't also be a dial.

## Honest gaps / next

- **KonJND pooled −0.050** — the one genuine pooled loss. G5 is a characterized
  Pareto limit (both bakes fail 0.70); pushing it via the aggregation head costs
  CID22/non-photo per prior work. Iteration 3 tries a mild konjnd-weight bump
  (recover without breaking the 4 wins).
- **HF pooled cross-image scale** — depth wins per-ref; recovering the pooled
  scale needs the near-lossless dial calibration B has. Lower priority (per-ref
  is the codec-dial metric).
- **Not a swap yet** — a swap needs the dial panel (G1/G3 monotonicity) + size
  (2-layer f32 is ~big; f16 repack) + the full methodology doc gates per the
  ship policy. This doc is the numbers half.

## MULTI-AXIS REFRAME (2026-07-16 eve — user "eval on imazen26", "nonphoto/ssim2 matters")

Everything above measured rank on CID22 (human MOS) + the panel. Adding
**imazen-26** as a first-class gate (G-IM26: ssim2-agreement over 962k held-out
real-codec cells across 4 lossy codecs, origin {7,9}) does not soften the "depth
is a better ranker" conclusion — it makes it **dominance**:

| model | CID22 | imazen-26 (ssim2) | nonphoto | dial |
|---|---|---|---|---|
| **B** (shipped linear) | 0.876 | 0.841 | 0.861 | viable ✓ |
| A_v47 | 0.866 | 0.862 | 0.878 | viable ✓ |
| no_tfm | 0.868 | 0.865 | 0.877 | viable ✓ |
| **base_tfm** (psa+tanh+tfm) | 0.856 | 0.948 | 0.952 | viable ✓ (G1) |
| **depth_v2** | **0.890** | **0.956** | **0.961** | dead (mono 0.55) |

- **depth_v2 DOMINATES B on every rank axis** (+0.013 CID22, +0.114 imazen-26,
  +0.101 nonphoto) — the rank/selection-trail champion, outright. Dial stays
  non-monotone (shared-anchor refit fails; a monotone spline can't fix
  non-monotone-in-q), so it's a *ranker*, confirming the SOTA rank-trail role.
- **WHY the gap:** B is linear and *excluded* from bigcodec (it poisons a linear
  model's CID22 per DATASET_HISTORY), so it never learned the real-codec ssim2
  surface → 0.841. The MLP absorbs bigcodec via capacity → both axes. Held-out.
- **base_tfm is a dial-viable BETTER all-rounder than B** once ssim2-agreement is
  weighted: imazen-26 +0.107, nonphoto +0.091, dial G1 ✓ (mono 0.929), −0.02 CID22.
- **The feature transforms are LOAD-BEARING here** (base_tfm 0.948 vs no_tfm
  0.865) — the earlier "less is more" was an artifact of ignoring nonphoto/ssim2.

So "B is at the achievable frontier" was **CID22-only**. On the multi-axis
picture the depth/psa MLPs beat B decisively. B still owns a clean dial — but
base_tfm has that too. Ship path: **depth_v2 as the rank/selection metric**;
**base_tfm (or a both-axis-sweep variant) as a candidate dial** that beats B once
ssim2-agreement counts. Both-axis train sweep RESULT (cid22/bigcodec weights): FAILED — no variant beat
base_tfm. Raising cid22 weight *hurt* held-out CID22 (0.856→0.841→0.818) because
cid22_train is ssim2-anchored (more weight = more ssim2-overfit = worse human
MOS — the session's recurring trap); ↑both craters (0.530). So no single
dial-viable bake matches B's CID22 AND wins imazen-26 via weight tuning — the
dial is a genuine B-vs-base_tfm tradeoff. A true both-axis dial winner would need
a 2-head (B's linear CID22 head + an MLP ssim2 head), not weight tuning.
