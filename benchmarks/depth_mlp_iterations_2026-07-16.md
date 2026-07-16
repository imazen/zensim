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
(browser: http://172.23.240.1:3300/zensim-depth-dashboard/depth_2026-07-16.html).

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
