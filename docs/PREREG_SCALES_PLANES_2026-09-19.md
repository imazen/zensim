# Preregistration — alternate scales, planes, colour space and pooling (2026-09-19)

Status: PREREGISTERED PLAN. Nothing here is run yet. Execution: bounded Devin swe-2 dispatches with their own
per-leg preregistrations; Fable reviews gates. Companions: [spatial steering plan](PLAN_SPATIAL_STEERING_DVIFM_2026-09-19.md),
[joint training core + pyramid tradeoff](PLAN_JOINT_CORE_SET_2026-09-19.md) (2026-09-19: one compact core set for
both zensim and DVIFM; no luma-only arms; the six-arm Z/D/H tradeoff study lives there),
[fitted-constant guards](FITTED_CONSTANT_GUARDS_2026-09-19.md).

## Why

- zensim converts to XYB once at full resolution, then 2×2-box-averages XYB f32 planes (4 dyadic scales,
  11×11 box-mean residual, global pooling, one fixed-phase 8-lattice blockiness slot). SSIMULACRA2 downsamples
  linear RGB first. Nothing pools block-wise; nothing is on a grid coprime with codec grids.
- AIC2026 (eval-only, label-free) showed crop-vs-full score swings up to 35 points and codec offsets of
  ±2–6 at matched CVVDP-JND. The DVIFM toy shows box scales alias 1.6× across codec-grid phases at level 3.
- **The Sept 13 scale-selective study is NOT evidence against non-dyadic scales** (user ruling 2026-09-19).
  Its own record: 32 epochs × 8,192 pairs, Triangle kernel only, proxy labels, direct contracts change the
  input planes. It set no asymptotic ceiling and its arms were not budget-matched to convergence.

## Fairness contract (applies to every leg)

1. Arms differ in exactly one thing. Same trainer binary, head shape, loss, LR cycle, epochs, pairs/epoch.
1a. Use the LEADING MODELS' recipe (user 2026-09-19): the R915 tables, 120×50,000 draws, seeds
   17101/17103/17107/17111/17113 — reproduce one leader fit before any arm. A generic h128 on the 8,327-row human
   TRAIN estate overfits in every arm (measured, `benchmarks/dvifm_screen2b_2026-09-19.md`).
1b. Every added-feature arm has a size-matched PERMUTED-column control; budgets are aligned to the LR cycle.
2. Same init seeds AND same sampling seeds across arms, ≥3 paired seeds to screen, 5 to confirm; report
   paired differences per seed, never bare means.
3. Budget is set by the SLOWEST-converging arm: run a convergence probe on each arm first (dev curve flat
   within seed noise over the last 20% of epochs); the screen budget = max over arms, applied to all.
   A leg whose arms have not both plateaued is INCOMPLETE, not negative.
4. Same TRAIN rows, same fit/dev source-disjoint families, same row order; each arm re-extracted under its
   own registered feature-set identity. Selection on TRAIN dev only. CID22 human, AIC-3/4, AIC2026, secret
   holdouts never select.
5. Cost is part of the result: 1T ms at 64²/256²/1024²/4096² and B/px for every arm (zenbench, interleaved).
6. Advancement rule declared before the run: paired dev gain > 2× seed SD on the registered composite AND
   no regression beyond seed SD on local-ordering and dial panels AND label-free stability not worse.

## Label-free stability gates (free; run before any training, TRAIN images only)

S1 1-px global shift of the pair; S2 codec-grid phase 0..7 on a synthetic 8×8 DC-step overlay;
S3 centre-crop vs full; S4 2:3 and 3:4 resize of both sides (score should move smoothly).
Report per-feature relative spread. A variant worse than the current extractor on S1–S3 gets no budget.

## Legs, cheapest first

**Tier 1 — appended families, existing slots bit-stable**
- L1 DVIFM block pooling, Y, 5×5 on a binomial 5-level pyramid (in flight; brief in zenpapers).
- L2 band construction inside L1: talk Laplacian vs local band vs zensim ρ.
- L3 block size n ∈ {3,5,7} with n=4 and n=8 as negative controls (tests the coprime-grid claim directly).
- L4 DVIFM in its native BT.709 Y′CbCr with ALL THREE planes (chroma is key to the method) vs Y′-only vs XYB-Y.
- L5 standalone DVIFM (convex by construction; tuned visibility knee a=1/C₀ and exponent b=β per plane/level)
  fitted on human data as its authors did — CID22-A 25 refs under a ledgered exposure, one frozen read on the
  sealed 24 — then the REVERSE build: DVIFM-90 as the base, zensim feature families added by subset.

- L6 (user, later 2026-09-19: "try yuv dvifm with xyb zensim, separate passes too") Y′CbCr DVIFM together with
  XYB zensim, two ways: EARLY fusion — the DVIFM-90 columns and the XYB zensim columns in one head (each family
  extracted by its own pass; no shared pyramid); LATE fusion — the standalone convex DVIFM score and the zensim
  score computed as fully separate passes/models and combined by a convex 2-weight blend (plus one monotone map)
  fitted on TRAIN. Late fusion keeps DVIFM's exact additive map for steering and lets either pass be skipped.
  Compare early vs late vs each alone, same recipe/seeds, permuted control for the early arm.

**Tier 2 — pyramid changes (every existing slot changes → one batched era break, TRAIN re-extract per arm)**
- P1 downsample kernel: 2×2 box vs binomial [1 2 1].
- P2 order: XYB-then-downsample vs linear-RGB-downsample-then-XYB.
- P3 scale schedule, the fair redo of Sept 13: dyadic {1,2,4,8} vs {1,2,3,5} vs {1,3,5,7} vs a 1.5× ladder
  {1,1.5,2.25,3.4,5}, each with a principled resampler (Mitchell), all kernels not just Triangle, full
  budget per the fairness contract.
- P4 planes: Y-only vs XYB vs XYB+luma-only extra scale.
- P5 colour mode (user 2026-09-19): the existing zensim kernels on BT.709 Y′CbCr in place of XYB — how much
  accuracy is actually lost, and how much extraction time the skipped XYB conversion buys (measured, all sizes).
- P6 **DEPRIORITISED (user, later 2026-09-19)** — runs last, after every other leg. 1.5× and 3× scales added to the ladder — for the DVIFM pyramid and, separately, for
  zensim's scale set: dyadic vs +1.5× vs +3× vs both, Mitchell via `zenresize`, identical on both sides.

**Tier 3 — interactions, winners only:** best Tier-1 pooling × best Tier-2 pyramid (2×2), five seeds, then
frozen assessment through the existing EVAL owners, with AIC2026 monotonicity/offset/crop panels as
label-free secondary reads.

## Time budget

Tier 1 rides the DVIFM lane (days). Tier 2 needs one TRAIN re-extract per arm (~10 arms) plus
convergence-matched training: schedule as a LAN zenfleet job set, not a single-box afternoon; do not start it
until the budget for ALL arms of a leg is reserved — a half-run leg is the Sept 13 failure again.
