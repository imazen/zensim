# Spatial steering built on DVIFM ideas — plan (2026-09-19)

Status: PLAN. Nothing here is measured. It sits under
[production priority P3](PRODUCTION_PRIORITIES_2026-09-15.md) ("make JXL spatial
guidance deliver independent quality/byte value") and does not replace its three
questions, its controls, or its exit criteria. User direction 2026-09-19: "also a
plan for spatial steering with dvifm ideas".

Mechanism source: `../zenpapers/docs/iqa-methods/dvifm-zensim-feature-design.md`.
Kernel on main: `zensim/src/dvifm.rs` (f956..f985, opt-in, default off).
Screens so far: `benchmarks/dvifm_screen{,2b,2c}_2026-09-19.md`.

## Why DVIFM is a better steering substrate than what we steer with today

Today's steering map is the attribution density of a learned scalar model over
globally pooled features. P3 records the consequences: rectangle terms are hard
max and non-additive ("summing tile gains does not estimate a union"), predicted
rectangle gain has to be checked against actual finite-edit gain, native JXL RD
is negative for the Rev3 controls, and steering refuses feature ids ≥ 228.

DVIFM has four structural properties that attack exactly those points:

1. **Additive over blocks, exactly.** A level's score is `mean_b(v_b · m_b^P)`.
   The contribution of any rectangle is the sum of its blocks' terms. The union
   of two tiles is the sum. No attribution approximation, no finite-difference
   probe: the map *is* the metric.
2. **The map is the quantity an encoder controls.** `m_b` is the peak absolute
   error in a 5×5 block of a band; `v_b ∈ (0,1]` says how visible an error of
   that size is given the masking contrast on the more revealing side. `v_b` is
   a direct "how much error can this block absorb" signal — an adaptive
   quantisation field in perceptual units, per band.
3. **Reference-side visibility needs no encode.** `v_ref,b` depends only on the
   source. With tied curves `max(v_ref, v_dist) = v(min(C_ref, C_dist))`; before
   the first encode `v(C_ref)` is an upper bound on block tolerance that costs
   one pyramid of the source. That is a zero-shot initial quant field.
4. **The grid is coprime with every codec grid.** 5-px blocks against 8/16/32/64
   codec blocks coincide only every 40 px (80 on the low-pass plane), so an
   encoder cannot hide error in a lattice-aligned blind spot, and the map has no
   preferred phase relative to the codec's partition.

And, if the integer definition lands (user direction: no f64, i16 domain, integer
log-mapped tables), a fifth: **bit-identical on every architecture**, so a
steering decision is reproducible across the fleet and inside a codec's RDO loop.

## What we build, in order

Each step has a gate. A failed gate stops the line and is recorded; later steps
are not started on hope.

### S0 — prerequisites (in flight)

- Standalone DVIFM in BT.709 Y′CbCr, all three planes, convex by construction,
  with tuned visibility knee `a = 1/C₀` and exponent `b = β` (Phase 2d, Part D).
  **Gate:** the one frozen read on the sealed CID22-B references is at least
  competitive with SSIMULACRA2 (paired bootstrap over references). Steering with
  a metric that does not rank images is pointless; if D fails, this plan stops
  at S0 and the reason is recorded.
- Integer (i16) kernel (`~/tmp/devin/dvifm_phase3_perf_prompt.md`), so the map
  is cheap enough to compute per shot. **Gate:** map + score ≤ the existing
  prepared-steering budget at 1024² (measured, zenbench, interleaved).

### S1 — the map owner

Extend `attribution.rs` / `diffmap` (the existing map owners — no second map
pipeline): emit, per band and per plane, the block field `ε_b = v_b · m_b^P`
painted over each block's pixels as `ε_b / N_b`, plus the two factors `v_b`,
`m_b` separately. The design doc already specifies this and that it folds
exactly.
**Gates:** (a) Σ map = score contribution, exactly (integer) or ≤1e-6 (float
oracle); (b) rectangle query = sum of block terms for arbitrary rectangles,
including ones that cut blocks (define the rule: area-proportional, stated);
(c) identity → all-zero map; (d) streaming/strip invariance, bit-identical.

### S2 — P3 question 1, which DVIFM should answer by construction

"Does predicted rectangle gain track the actual finite-edit Zensim gain?" For a
DVIFM score the prediction is exact for edits that change only that rectangle's
blocks — but band pyramids leak: an edit in a rectangle changes coarser-level
blocks outside it. Measure that leak instead of assuming it away.
**Experiment (TRAIN images, existing `diffmap_block_coherence` + native
intervention instruments):** replace rectangle R of the distorted image with the
reference (the ideal edit), predict Δscore from the map, measure actual Δscore.
Report prediction error by rectangle size (16…256 px) and by pyramid level, and
the same for the current attribution-density map as the control.
**Gate:** DVIFM's median relative prediction error is below the current map's at
every rectangle size, and unions of two disjoint rectangles are additive within
the measured leak bound.

### S3 — P3 question 2: does the map agree with independent judges?

Per-block agreement between the DVIFM map and independent spatial judges on
TRAIN codec pairs: SSIMULACRA2's per-scale error maps and Butteraugli's diffmap
through our own `fast-ssim2` / `butteraugli` crates (two-reference truth — a
block is "truly bad" only where both agree), plus CVVDP's map where a stored one
exists. Rank agreement of block-level severity within an image, per codec.
**Gate:** within-image block-rank agreement with the two-reference truth is at
least that of the current steering map, per codec, with JPEG and JXL reported
separately. Text/screen content reported on its own (the known weak class for
maps; DVIFM's edge-discounted contrast exists precisely for it).

### S4 — zero-shot quant field from the reference alone

`v(C_ref)` per block and band → a per-block tolerance; map band tolerances onto
the codec's control (JXL: per-group/AQ distance field via the existing
`zensim_diffmap_rd` seam; JPEG: per-block AQ in zenjpeg; AVIF stays out of scope
while the backend hold stands). Each codec owns its own loop; this crate only
supplies the field (per-codec loop-ownership rule).
**Experiment:** matched-quality byte cost and matched-byte quality vs the codec's
own AQ and vs a flat field, judged by independent metrics, not by DVIFM itself
(SSIMULACRA2 ∧ Butteraugli agreement; CVVDP where available). Neutral/active and
scalar-controller controls retained, as P3 requires.
**Gate:** bytes saved at matched two-reference quality, with a per-source-family
breakdown and no content class regressing beyond its own uncertainty. A saving
judged only by DVIFM does not count.

### S5 — closed loop: magnitude steering with an exact map

The only loop value measured so far is magnitude steering (H3). With an additive
map the controller is simple: after shot k, the blocks with the largest
`ε_b` are where bits buy the most score; blocks with `v_b·m_b^P ≈ 0` and high
`v_ref` headroom are where bits can be withdrawn. One secant step on the global
quality target (existing per-codec secant), one redistribution step on the
field. Shot budget stays at the registered three.
**Gate:** the P3 exit — independent-judge RD and coherence on frozen assessment —
at equal encodes against the scalar controller. Report map construction and
query cost, shot count and bytes, per source family.

### S6 — what zensim keeps

The reverse-build experiment (Phase 2e, Part E) decides the scalar: DVIFM-90 plus
the smallest zensim family subset that matches `basic228` on TRAIN development.
If a learned head sits on top of DVIFM terms, steering still uses the DVIFM map
(exactly additive) and treats the head's correction as a per-image scalar gain,
so the map stays exact. A head that needs spatially varying corrections is a
separate, later question.

## Risks stated up front

- **Pyramid leak (S2)** may make rectangle predictions poor at coarse levels. The
  local band `G_l − B²G_l` leaks less than the Laplacian (shift sensitivity 0.4%
  vs 13.9% in the design doc's toy); it is the fallback, not an assumption.
- **Peak pooling is sparse.** `m_b` is a hard max: an encoder can lower one pixel
  and move the block. That is good for artifacts (ringing, blocking peaks) and
  weak for low-amplitude texture loss. S3 will show it as disagreement with the
  judges on blurred texture; the design's soft-peak ablation is the fix to try.
- **Convexity is a constraint on the score, not on the encoder's problem.** RD
  allocation against an additive, monotone map is well-posed; it is not claimed
  to be convex in the codec's parameters.
- **Tuning on CID22-A** exposes those 25 references; every steering result is
  judged on TRAIN codec pairs and independent metrics, never on CID22.

## Ownership and dispatch

Fable: this plan, gate review, go/no-go between steps. Bounded execution
(S1 map owner, S2/S3 measurement, the i16 kernel) goes to Devin swe-2 with
preregistered gates; mundane lanes to Sonnet. One heavy job at a time on the box.
