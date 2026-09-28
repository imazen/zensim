# LANE `zgeom` — the two questions the transplant result did NOT answer

You are the `transplant` lane, continuing. Your `joint-core-v2` (105,614 pairs, permutation gate clear from 61k up),
your arm machinery and your permuted-control discipline are exactly what these need. `~/tmp/devin/LANE_PREAMBLE.md`
still binds you (shared heavy lock via `~/tmp/devin/heavy`, no push, no holdout reads). Outputs
`/mnt/v/output/zensim/zgeom-2026-09-21/` (≤10 GB). Report to `~/tmp/devin/LANE_ZGEOM_DONE.md`, progress
`~/tmp/devin/lane_zgeom.log`.

Your X4 tested 5×5 block pooling **appended** to the 944-feature surface, and it landed at −0.0007 against its
permuted control. That answers "does block pooling add anything on top of everything zensim already has" — no. It
does NOT answer either question below.

## Z1 — block pooling as a REPLACEMENT, not an addition
Append-tests live in the saturated regime: with 944 columns already present, a redundant statistic cannot show a
gain even if it is a better statistic. So swap instead of add, at matched feature count:
- Baseline arm: zensim's existing pooling over its own XYB residual planes (`basic228` as-is).
- Replacement arm: the SAME planes and scales, with the global-moment pooling for the pooled-statistic slots
  REPLACED by 5×5 block-peak × two-state-gate pooling, at the same or FEWER columns than it removes. State exactly
  which slots you replaced and the column count each side.
- Control arm: the same replacement with the block statistics row-permuted.
Leaders' recipe, 5 paired seeds, development legs; report per-seed paired differences and sign counts. The question
is whether block pooling is a BETTER pooling than global moments, not whether it is an extra one.

## Z2 — box vs binomial for zensim's OWN pyramid (prereg leg P1)
The geometry lane measured, on DVIFM's model: box2 fails the 1-px shift gate (+0.0146) and drifts +0.054 over codec
phases where binomial kernels drift ≈0, and `bin1331` gained +0.0110 SROCC for +44% single-thread cost (+5% at 8
threads). None of that was measured on ZENSIM's features, which is what would actually change. So:
- Arms: `basic228` on today's 2×2-box scales (baseline) · the same features on `bin121` scales · on `bin1331`
  scales. Same recipe, seeds, rows, row order; identical kernel both sides of every pair; new feature-set identity
  per arm (never column-mix across kernels).
- Report: development-leg accuracy per seed; the three label-free stability gates (1-px shift, codec-grid phase
  0..7, crop-vs-full) for zensim's own features; and the measured extraction cost at 256²/1024²/2048², 1 thread and
  8, with `α + β·pixels`.
- State the consequence plainly in the record: adopting a new downsampler changes every existing bake's inputs, so
  it is an era break that invalidates all trained weights and every historical table. The decision needs the
  accuracy gain on OUR features, the cost, and the stability gain side by side — produce exactly that table.

Order: Z2 first (it is the one that could change production), then Z1. If either shows a gain that beats its
control, say what the smallest shippable change would be; if not, record the negative and stop. Do not adopt
anything, do not touch a default, do not push.
