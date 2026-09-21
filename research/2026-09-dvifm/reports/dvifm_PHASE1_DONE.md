# DVIFM Phase 1 — DONE (2026-09-19)

Scope executed: the worker brief's "What to build" + "Gates before any
screen" only. No TRAIN extraction, no training, no block-stats cache
fitting, no screening. The family is opt-in, default-off, additive-only.

## Commits (all on top of supervising tip stkzxxpq / b684c2fe; none pushed)

| change | commit | content |
|---|---|---|
| nnopxxqx | a4abefa5 | scalar DVIFM kernel (`dvifm.rs`): 5-level binomial pyramid, reflect-101, target-shape expand lattice, Laplacian+Local band modes, 5×5 blocks, corner-3×3 extrema, log-domain visibility, F1+5 F2 = 30 features; streaming `DvifmAccum`; baked SDR/PU norms with recompute tests |
| yuymwvmo | 0bf63a20 | numpy parity fixtures + test (`scripts/dvifm_parity_fixture.py`, fixture 28.5 KB) |
| quntvqrr | 65b9165a | registration: `ComputeToken::Dvifm` appended (no bit renumber), `Replication::Flat`, `KernelId::Dvifm`, `dvifm_block` toggle (default OFF, requires csfw→append2→append), `Folded720Dvifm` regime, walk hook on scale-0 Y strips (lazy accum from strip-info dims), `Phase::DvifmKernel`, registry w986 (slots_hash8 `685eb6ef`, verified by the zensim-validate sync test) |
| xsqlswzu | 03a3f8b6 | streaming parity tests |
| tzyqruwn | 5344eff1 | SIMD tier parity test |
| lrsvvxwq | e177b6f8 | byte-stability test |
| rstxtrry | 7d7c018c | bench arms `fold956_csfw`/`fold986_dvifm` |
| opunyxpv | 7f2b556b | gates doc + research-parity test widened to 986 |

Gates doc: `benchmarks/dvifm_block_gates_2026-09-19.md` (MISSING list first).

## Gate results (measured)

| gate | result |
|---|---|
| numpy reference parity | **PASS** — all pyramid planes + 30 features ≤1e-6, both band modes |
| streaming ≡ whole-plane | **PASS bit-identical** — pump pushes 1..128 rows incl. non-multiples of 5/16; served walk tail `to_bits`-equal to `dvifm_features_stream` at 150×170, 67×83, 131×129, 128×128, 40×50 (sub-64 reflect-pad path), serial+parallel; HDR/PU route 96×101 bit-identical |
| SIMD ≡ scalar | **PASS bit-identical** — v3/v4/v4x all `to_bits`-equal to scalar on this AVX-512 host, both band modes, lane-remainder widths 125/97/6/128. neon/wasm128 unexercisable here |
| byte stability toggle ON | **PASS** — f0..f955 `to_bits`-identical 956→986, serial+parallel |
| byte stability toggle OFF | **PASS** — pump hook is None when off; v1_golden_bytes 5/5, fold_engine_parity 10/10, feature_invariants 13/13, v1_width 10/10; registry sync green |
| cost: toggle-off vs pre-DVIFM binary | **PASS (no regression)** — two-binary A/B, 2 alternating blocks × 30 paired rounds, pinned cpu 8: shared arms −0.1..−3.5% at 2048² (anchor +0.34%); 1024² group contaminated by −8.6% anchor drift — reported not smoothed |
| cost: toggle-on (fold956→fold986) | **MEASURED: +31.6 ms (+66%) at 1024², +120.5 ms (+54%) at 2048²** (medians; cv 0.3–1.0%) — real cost of a 5-level f64 pyramid; not claimed cheap |
| memory toggle-on | **+0.5 MB at 1024² (51.4→51.8 B/px), +1.3 MB at 2048² (27.2→27.6 B/px)** — `/usr/bin/time -v` peak RSS, ZEN_XP_RSS arms |
| `just clippy` (CI-exact, workspace, all features) | **PASS** `-D warnings` clean |
| `just lint-scripts` | **PASS** 624 scripts |
| `cargo test -p zensim -p zensim-validate --features feature-regime-v2,training,custom-profiles` | 523 lib + all binaries green **except the 5 known pre-existing `zensim-validate/tests/bake_surface.rs` failures** (named out-of-scope in the work order; verified unchanged-by-me) |

## Spec ambiguities resolved

- **f956 free?** YES — `csf_tier1_gates_2026-07-28.md` remainder #1 records the
  chroma-tier claim f956..f979 recommended CLOSED (2026-07-29); `hdr` registry
  token has `slots: null`. Documented in the gates doc.
- **Replication shape** — the design's own pyramid is 5 levels × 6 signals =
  flat 30; added `Replication::Flat` (scale-0 attributed), rather than
  stretching PerScale.
- **Expand lattice** — the Python `_up` builds the zero lattice at the
  TARGET shape (h,w), not 2⌈h/2⌉ — odd-dim reflect domains differ; ported
  exactly.
- **`peak`** is a saturating weighted mean, not logsumexp; `contrast_g` has
  an `edge` flag (corner-3×3 extrema at block edges, full range inside).
- **`visibility_smooth`**: c≤0 → v=1 (NaN handled via partial_cmp, no
  float-cmp lint).
- **Accumulator dims** — taken from the producer's strip info (post-pad),
  so sub-64 images are pumped at the padded 64² the walk actually sees
  (tested at 40×50).

## NOT done (deliberately — next dispatch)

- TRAIN extraction / screening / model fitting; `DvifmAccum::block_cache`
  remains dead training-side surface.
- SEED-1.0 constants unfit — the family is NOT a trained DVIFM score.
- HDR cost, cross-arch (neon/wasm) measurement, `v1_pools=Full`
  cross-binary arm.
- 1024² toggle-off re-measurement under a quieter box would tighten the
  "no change" claim; current evidence shows no regression within the
  contaminated bound.

## Evidence dirs

`~/tmp/devin/dvifm-ab/` (A/B logs, run.meta, binary.sha256,
competing_procs.txt, report.txt), `~/tmp/devin/dvifm-rss/rss.tsv`,
`~/tmp/devin/dvifm_progress.log`.
