# geometry-matrix — sampling-kernel x band x block-size matrix for DVIFM + zensim pooling

## What was asked

Brief: `../briefs/lane_geometry_prompt.md`. A flattened 3-stage screen over
DVIFM's own geometry knobs: pyramid decimate/expand kernel (box2, bin121,
bin1331, mitchell, lanczos3, boxres, xyb, plus grid controls n3/n4/n7/n8/
@2_2), band mode (Laplacian vs Local), and block size, answering four
questions: (a) cheapest not-worse config, (b) best-at-any-cost config, (c)
whether block size matters once made coprime with the sampling grid, (d)
whether zensim's own box-filter pyramid has measurable stability cost vs the
binomial/other kernels tested here.

Note: stage-1 cost was measured a first time, found to be comparing
vectorized-dispatch vs a scalar path (unfair), and **re-measured** under a
fairness protocol (every kernel a generic row body over the same
`#[magetypes]`/`incant!` tier dispatch, resolved tier + disassembly proof
recorded, forced-scalar negative control). All numbers in the report and
below are the fair (re-measured) ones; the superseded first-pass numbers are
kept in the artifacts, marked SUPERSEDED, never cited as current.

## Verdict (from `../reports/LANE_GEOMETRY_DONE.md` — copied verbatim)

**Recommendation (unchanged across both cost passes): `bin1331.local.n5`** —
binomial [1 3 3 1] pyramid decimate/expand, local band, n=5 block, DVIFM
pooling.

Four required answers:
- (a) Cheapest not-worse: `bin121.lap.n5` (1.00x, flat, unfitted); among
  fitted cells, the baseline itself.
- (b) Best at any cost: `box2.boxres.n5.xyb` (+0.0126 accuracy) is
  DISQUALIFIED by stability; best eligible = `bin1331.local.n5` (+4.0%
  @1024² t1 fair cost, parity at t8 batch-of-8).
- (c) Block size once coprime with the grid: nearly flat — keep n=5 (n3's
  +0.0032 accuracy doesn't cover its +27% cost; n4/n8 fail stability by
  design as grid controls).
- (d) Zensim's own box-filter pyramid has measurable stability cost: YES —
  the xyb arm shows +0.054 ln worst codec-phase shift vs ≈0 for the
  binomial kernels.

Fair-cost headline numbers (t1, 1024²): baseline (box2) 32.13 ms; bin1331
1.04x; i16-serving stream **0.94x** (faster than f64 box2 despite being the
integer serving kernel). Call-per-op (unfixed codegen) reference showed
inflated ratios up to 4.77x for the same kernels — this quantifies the
production inlining defect the follow-on `simd` lane then fixed (see
`../reports/LANE_SIMD_DONE.md`).

## Code in this directory

| file | role |
|---|---|
| `run_lane.sh` | top-level 3-stage driver |
| `tools/stage1_analyze.py` → `stage1_analyze.py` | stage-1 kernel x geometry stability + cost analysis |
| `stage2_analyze.py` | stage-2 pooled accuracy analysis (ΔSROCC per arm) |
| `stage3_analyze.py` | stage-3 fair-cost re-measurement (bin1331 + baseline + serving, 64²..4096², 30 rounds) |
| `make_view.py` | renders the comparison views used in the benchmark report |
| `fit_geometry.py` | the geometry-lane fit driver (lives under `tools/joint_core/fit_geometry.py` in the workspace — same dir name as the joint-core lane's tools, but this file is geometry-specific) |

## Rust / repo state

Workspace `zensim--geometry` (`/home/lilith/work/zen/zensim--geometry`),
three commits, working copy clean (`usoslxkz`/`c2f2344f`, empty — nothing at
risk), unpushed, not merged to main:
- `lkqrrmzv` / `ef722289` — "sampling-kernel x band x block-size matrix on
  dvifm+zensim pooling" — adds `zensim/src/dvifm/geom.rs` (new),
  `zensim-bench/examples/geometry_matrix.rs` (new), touches
  `zensim/src/{dvifm,research}.rs`, `zensim-bench/Cargo.toml`,
  `tools/joint_core/fit_geometry.py` (new), `tools/joint_core/fit_core.py`.
- `lkzlwlmx` / `82fe761a` — "fair-cost re-measurement — vectorized dispatch
  for all kernels, i16 serving arm, SUPERSEDED old cost table" — touches
  `zensim/src/{dvifm/geom,dvifm,feature_v2,research}.rs`,
  `zensim-bench/examples/geometry_matrix.rs`.
- `porqknnm` / `f35dbcc8` — a follow-on **`simd`** lane in the same
  workspace: "fold dvifm+dvifm_int generic helper chains into featured
  `incant!` entries" — the root-cause fix for the call-per-op codegen defect
  this lane's cost measurement surfaced. See `../reports/LANE_SIMD_DONE.md`
  for its own verdict (measured 44-48% additional speedup on top of the
  int16 kernel, bit-exact before/after). Its brief is
  `../briefs/lane_simd_prompt.md`.

## How to re-run

1. `run_lane.sh` drives all three stages end to end via the
   `zensim-bench/examples/geometry_matrix.rs` harness.
2. `stage1_analyze.py` reads the stage-1 output and produces the fair-cost
   comparison (resolved-tier + disassembly proof are printed by the Rust
   harness itself, not this script).
3. `stage2_analyze.py` reads the stage-2 output for pooled ΔSROCC per arm
   (5 paired dev seeds).
4. `stage3_analyze.py` reads the stage-3 fair re-measurement (64²..4096²,
   30 interleaved paired rounds, t1 and t8-batch).
5. `make_view.py` renders the final comparison tables/views.
6. `fit_geometry.py` is the underlying per-arm SROCC fitter these stages
   call into.

Inputs: same joint-core-v1/v2 TRAIN pairs used by the sibling lanes; no
holdout reads.

## Artifacts (reference by path)

- `/mnt/v/output/zensim/geometry-2026-09-21/` — `cache/`, `cells.json`,
  `cost3_fair/`, `cost3_superseded/` (kept, marked superseded, not deleted),
  `cost_callperop/` (the unfixed-codegen reference run), `cost_fair/`,
  `cost_fair_scalar/`
- Committed benchmark record (already permanent in the repo, not copied
  here): `benchmarks/geometry_matrix_2026-09-21.{md,json,pointer.md}`
  (in the `zensim--geometry` workspace's own commit — not yet merged to the
  main checkout's `benchmarks/`); the follow-on simd audit:
  `benchmarks/simd_inlining_audit_2026-09-21.{md,json}`
