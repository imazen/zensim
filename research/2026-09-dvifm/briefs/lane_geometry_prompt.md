# LANE `geometry` — sampling kernels × band × block size, on BOTH dvifm and zensim pooling

Read `~/tmp/devin/LANE_PREAMBLE.md` first; it binds you (shared heavy lock, no push, no holdout reads).
Then `docs/PLAN_DVIFM_VERDICT_2026-09-20.md` §4 (kernel theory) and §5 (block-edge contrast), and the existing
brief `~/tmp/devin/planar_filter_cost_prompt.md` — this lane SUPERSEDES that brief and widens it.
Outputs `/mnt/v/output/zensim/geometry-2026-09-21/` (≤10 GB). Data: `joint-core-v1`
(`/mnt/v/output/zensim/joint-core-v1/`), TRAIN/development legs only.

Nobody has ever measured these three axes inside a metric. Measure them, on both pooling designs, and report cost
and accuracy together. Every arm is one change from its baseline; identical kernel on both sides of a pair always.

## Axis A — planar sampling kernel (the downsample step)
`box2` (zensim's 2×2 average) · `bin121` (DVIFM's [1 2 1], the current default) · `bin1331` · `mitchell`
(via the imazen `zenresize` crate — read its local README/source first, no foreign resamplers) · `lanczos3`.
Report for each: taps/output px, passes, live planes, halo rows for streaming — and the MEASURED cost.

## Axis B — band construction
`lap` (G−E(G↓), needs the expand pass) · `local` (G−B²G, no expand, no lattice) · `boxres` (zensim's 11×11
box-mean residual, O(1)/px via running sums).

## Axis C — block size and grid
n ∈ {3, 5, 7} (odd, coprime with codec grids) plus n ∈ {4, 8} as NEGATIVE CONTROLS (grid-aligned), and a
(2,2)-offset second grid averaged with the first as an optional variant. Report grid-phase sensitivity per n.

## Applied to BOTH designs
1. **DVIFM pooling** (5×5 block peak × two-state gate visibility) on its own binomial pyramid — the A×B×C matrix.
2. **zensim pooling transplant**: zensim's XYB decomposition with block pooling at each n — this is where block
   size interacts with our own features. Coordinate with the `transplant` lane (pane w10:p7) so you do not both
   build the same arm: it owns X4/X5/X7 on the accuracy side; you own the GEOMETRY sweep and the cost side.
   If a needed arm already exists there, reuse its table rather than re-extracting.

## FLATTENED DESIGN (supervisor, 2026-09-21) — 3 stages, ~20 runs, NOT the 150-cell factorial

The full A×B×C×design cross-product is far too much. Run a screening design instead, one factor at a time from a
fixed baseline, and only spend the expensive protocol on survivors.

**Baseline** = `bin121` + `local` band + n=5, DVIFM pooling, i16 kernel.

**Stage 1 — cheap screen, NO training (13 arms, label-free).** From the baseline change exactly one thing:
kernels `box2 / bin1331 / mitchell / lanczos3` (4 arms) · bands `lap / boxres` (2) · block sizes `n=3 / n=7`
plus the grid-aligned negative controls `n=4 / n=8` (4) · zensim-pooling-at-n=5 (1) · the (2,2)-offset second
grid (1), plus the baseline itself. For each: the three label-free stability gates (1-px shift, codec-grid phase
0..7, centre-crop vs full) and a COST SCREEN at 1024² only, 1 pinned thread, ≥20 interleaved rounds. That is the
whole of axes A, B and C at first order, and it is hours not weeks.
Advance an arm only if it is not worse than the baseline on stability by more than the baseline's own round-to-round
spread; report all 13 either way.

**Stage 2 — accuracy on survivors only (≤6 cells).** Take at most the 2 best kernels, the best band and the best
block size, plus the zensim-pooling arm, and fit the standalone gate model with the histogram grid stage only (no
multi-start refinement), 3 paired seeds, development-leg SROCC/KROCC with per-seed paired differences against the
baseline. If two winners are on different axes, test ONE 2×2 interaction between them — not all pairs.

**Stage 3 — full protocol on ONE cell.** The single recommended cell (plus the baseline as its control) gets the
complete cost measurement: 64²/256²/1024²/2048²/4096², 1 thread and 8 threads, ≥30 paired rounds, `α + β·pixels`
with both terms, peak RSS.

Drop anything that does not serve the four required answers. `lanczos3` may be screened in Stage 1 for stability
only and then excluded with its reason recorded (it rings; a ringing reference makes the metric measure its own
resampler). If Stage 1 shows an axis is flat — e.g. block size does not matter once the grid is coprime — say so
and do not carry that axis into Stage 2.

## What to measure per arm (apply the stage that arm is in)
- **Cost**: zenbench interleaved (never criterion), ≥30 paired rounds, 1 pinned thread and 8 threads, sizes
  64²/256²/1024²/2048²/4096², built WITHOUT `-C target-cpu=native`, competing processes recorded. Fit and report
  `α + β·pixels` with BOTH terms. Memory via `/usr/bin/time -v`. Report the i16 kernel where it supports the cell
  (`zensim/src/dvifm_int.rs`, `DvifmParams::int16()`), f64 otherwise; say which.
- **Stability, label-free, TRAIN images**: 1-px shift of the pair; codec-grid phase 0..7 on an 8×8 DC-step overlay;
  centre-crop vs full. Report relative feature spread per cell. This is where `box2` is expected to lose and where
  n=4/8 should show alignment sensitivity — confirm or refute.
- **Accuracy**: fit the standalone gate model per cell on the core's fit rows, 3 paired seeds, and report the
  development-leg SROCC/KROCC with per-seed paired differences against the `bin121`+`local`+n=5 baseline. Cheap
  because the histogram grid stage is seconds; do not run the full multi-start refinement for every cell — state
  the protocol you use and keep it identical across cells.

## Deliverable
`benchmarks/geometry_matrix_2026-09-21.{md,json}` (+ `.pointer.md`; nothing >30 KB in git): the three tables
(cost with α and β, stability, accuracy), and a one-paragraph recommendation naming (a) the cheapest cell that is
not worse than the baseline, (b) the best cell at any cost, (c) whether block size matters at all once the grid is
coprime, and (d) whether zensim's box scales cost it measurable stability against binomial. Say plainly which
numbers are measured and which are operation counts. Terminal file `~/tmp/devin/LANE_GEOMETRY_DONE.md`, progress
`~/tmp/devin/lane_geometry.log`.
