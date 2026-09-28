# LANE geometry — DONE (rev 2, fair-cost re-measurement) 2026-09-21

Flattened 3-stage screen executed end-to-end; the cost axis was then
re-measured under the supervisor's fairness protocol after pass 1 was found
to compare vectorized-dispatch vs scalar-path. All numbers measured through
~/tmp/devin/heavy (no outside-lock deviation needed). No push. Geometry jj
workspace `lkqrrmzv` + merge `lkzlwlmx` (dvifm3 stack → i16 serving kernel).

## Recommendation (unchanged)

**`bin1331.local.n5`** — binomial [1 3 3 1] pyramid decimate/expand kernel,
local band, n=5 block, DVIFM pooling.

## Fair-cost protocol conformance

1. Every timed kernel = generic row body over `F64x8Backend` behind the
   same `#[magetypes]`/`incant!` tier dispatch — no scalar fallback in a
   measured path. `#[inline(always)]` on the helper chain folds vector ops
   into the featured tier entry (required — generic-backend monomorphs
   otherwise compile featureless: codebase-wide call-per-op, verified on
   production `level_push_row`/`vblur3`/i16 rows too).
2. Resolved tier reported at startup: **v4x (AVX-512+VBMI2/GFNI/VNNI)**.
3. Proof per kernel — disassembly: entry_scalar 0 zmm/0 ymm; v3 ~4703 ymm;
   v4/v4x ~3592 zmm + ~268 ymm. Forced-scalar control (`--force-tier
   scalar`, same binary): auto faster for every kernel (+2.1%..+6.7%,
   monotone in taps); serving arms invariant ±0.5% (negative control).
4. `dvifm_int.rs`/`DvifmParams::int16()` merged in; i16 reported alongside
   f64 (i16 = serving path, 32 lanes/AVX-512 reg).
5. Block sizes: production `scan_block_row` is scalar n-generic → geometry
   n-generic scalar scan is the SAME dispatch shape → n≠5 IS cost-comparable
   (kept, not dropped).
6. Old table marked SUPERSEDED in the record, preserved with reason at
   `cost_superseded/ cost3_superseded/ stage{1,3}_superseded.*` +
   call-per-op reference binary output `cost_callperop/cost_t1.json`.
7. Accuracy/stability NOT re-run (were fair).

## Measured evidence

Stage 1 stability (unchanged — was fair): PASS = bin1331, mitchell,
lanczos3, lap, boxres, n3, @2_2, baseline. FAIL = box2 (+0.0146 shift1),
n4, n7, n8 (grid controls), xyb (+0.054 codec phases, +0.023 shift1).

Stage 1 FAIR cost @1024², t1, 20 rounds (auto=v4x):
baseline 32.13ms; box2 0.99×; bin1331 1.04×; @2_2 1.01×; boxres 1.06×;
lap 1.00×; n4 1.04×; n3 1.27×; mitchell 1.28×; lanczos3 1.51×;
xyb 1.04×; n7 0.96×; n8 0.91× (cheaper, unstable).
Serving: f64stream 1.42×; **i16stream 0.94×**.
t8 batch-of-8: bin1331 0.99×, mitchell 1.02×, lanczos3 1.04× — bandwidth
saturated; i16stream 0.45× per batch.
Forced-scalar margins: +4.0% bin121, +3.6% bin1331, +6.7% mitchell,
+5.1% lanczos3, +2.1..4.7% rest; serving ±0.5%.

Stage 2 accuracy (unchanged): pooled ΔSROCC — xyb +0.0126 (stab-failed),
**bin1331 +0.0110**, mitchell +0.0079, bin1331.boxres +0.0064 (sub-additive,
rejected), boxres +0.0053, n3 +0.0032.

Stage 3 FAIR (bin1331 + baseline + serving; 64²..4096², 30 rounds):
- t1 β: bin1331 45.02 vs bin121 43.69 ns/px (**+3.0%**, was +29% artifact);
  1024² 31.33 vs 30.30ms; 4096² 749.9 vs 727.8ms; α −8.95/−8.73ms;
  R²≥0.999.
- t8 β: 95.24 vs 95.82 ns/px per batch-of-8 (**−0.6%**, CI crosses zero
  ≥1024²); per-image ≈ ÷8.
- f64stream t1: β 44.31 (≈ f64 geometry + ~1ms pump α).
- **i16stream: β 28.76 t1 (−34% vs f64 geometry), 30.77 t8 (~2.6× faster
  than f64 batch); 4096² 482.6ms t1 / 516.7ms per 8-batch.**
- peak RSS 2.33 GiB t1 / 16.31 GiB t8.

Call-per-op reference (pre-fix binary, provenance): bin1331 2.20×, mitchell
3.95×, lanczos3 4.77×, boxres/xyb 2.3× — quantifies the shipped-codegen
inlining gap (bonus finding for production vectorization work).

## Four required answers

(a) Cheapest not-worse: `bin121.lap.n5` (1.00×, flat, unfitted); among
fitted cells the baseline itself.
(b) Best at any cost: `box2.boxres.n5.xyb` (+0.0126) DISQUALIFIED by
stability; best eligible = `bin1331.local.n5` (+4.0% @1024² t1, parity t8).
(c) Block size once coprime: nearly flat — keep n=5 (n3 +0.0032 accuracy
doesn't cover +27% cost; n4/n8 fail stability as designed).
(d) Zensim box scales cost measurable stability: YES — +0.054 ln worst
codec phase vs ≈0 for binomial kernels.

## Caveats

- Geometry research path f64-only; no bin1331-i16 kernel exists (would be a
  new definition, not a measurement). i16 arm = production cell shape.
- i16 arm measured on today's shipped (call-per-op) codegen — fixing
  generic-backend inlining production-wide would widen its advantage.
- Scalar islands by design (h-decim, block scan, pooling, allocs) bound the
  cell-level SIMD margin to ~2–7%.
- konfig_val negative for ALL cells (systematic leg effect).
- xyb arm is 4 levels/36 features — not a 5-level DVIFM configuration.
- Report: `zensim--geometry/benchmarks/geometry_matrix_2026-09-21.{md,json,
  pointer.md}`; artifacts `/mnt/v/output/zensim/geometry-2026-09-21/`
  (stage{1,2}.json+md regenerated with fair cost; stage3_fair.*).
