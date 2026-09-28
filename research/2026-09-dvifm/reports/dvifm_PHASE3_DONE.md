# dvifm PHASE 3 — DONE (Int16 kernel built; passes both cost bars; f64 stays oracle-only)

Repo: `/home/lilith/work/zen/zensim--dvifm3` (jj workspace of
`/home/lilith/work/zen/zensim`, created because devin-core holds the main
checkout with a live marker loop). All commits sit on top of the
devin-core stack that contained `PLAN_DVIFM_VERDICT_2026-09-20.md`.
**Not pushed.** Box shared with devin-core's extraction + a Tower rsync —
recorded everywhere it mattered, never smoothed.

## Commits (oldest → newest)

| commit | what |
|---|---|
| `3dc4e78a` | `bench: dvifm step-0 stage split — scan 37%, pyramid 50%, pool ~12%` |
| `943781e8` | `dvifm: step-1 exact shortcuts + step-2 Int16 serving kernel` (one jj change — the step-1 describe stayed open and absorbed step-2; described honestly) |
| `fa0b60f0` | `dvifm: step-2b — int16 monotone test, i686 warning fixes, deviation record` |
| `0d3f8c00` | `dvifm: step-3 cost gate — ZEN_XP_ARMS bench filter + i16-vs-f64 verdict` |

`cargo fmt --all -- --check` clean; `just clippy` (CI-exact, `-D
warnings`) clean; `just lint-scripts` clean (625 scripts).

## Step-0 split table (as committed)

Added ms this box (32 paired rounds, pinned): +3.24 ms @256², +49.72 ms
@1024², +192.05 ms @2048² (phase-1 quiet-box: +31.6/+120.5 — both kept).

| stage | 1024² share | calls/iter @1024² |
|---|--:|--:|
| normalise f32→f64 | ~3% | 2048 |
| pyramid (blur/down/expand/sub) | ~50% | 24960 |
| block scan max/min | ~37% | 394 |
| per-block math | ~10–14% | 55390 |
| ring/alloc | <2% (below the 10% skip bar) | — |

Transcendentals per block: **26** (powf=17, ln=3, exp=4, ln_1p=2) —
the count that made the integer LUT design the obvious lever.
`esum3`/`zrow` = 0: the fitted default runs Local bands only.

## What landed

**Step 1 (bit-identical f64)**: g=1/P=1 bypasses, `VisBaked` hoisted
constants, tied-curve merged-visibility shortcut (monotonicity-checked,
NaN falls back to two evals), pump tap borrows replacing per-call
`.to_vec()` clones, spare-row pool. All 43 pre-existing dvifm tests
green throughout.

**Step 2 (the primary deliverable — built first, not gated)**
`zensim/src/dvifm_int.rs`: Q14 quantiser (round-half-up, NaN→0), integer
`[1 2 1]/4` pyramid via the two-step rounding average, `i16` bands
proven to fit `[-16384,16384]` (`band_fits_i16`), integer block scan,
log-indexed LUTs (`lzc` + 6 mantissa bits + 4 frac bits, 705
entries/level) carrying v + five hats in Q15 plus φ/`m^P` in Q14, exact
`i64` accumulation, one f64 division at emit. `DvifmVis{Curve,Gate,Off}`
honored by the f64 oracle too; `DvifmParams::int16()` = fitted constants
+ Gate at fitted knees = the serving definition; `default()` stays the
ExactF64 oracle; `served()` applies `ZEN_DVIFM_MATH=i16` once per
process (the pre-bake selector); explicit `DvifmSpec.math` wins.
`DvifmAccumSel` picks the kernel once at construction — no per-row
dispatch. Research cache emits 17-i32 `records_int` (no soft-peak —
pooled features never read it; documented). Bench spec loader accepts
`math:"i16"` + per-level `vis`, backwards-compatible defaults.

**Parity (exact, not tolerance)**: `scripts/dvifm_int_ref.py` is an
independent integer implementation; the fixture
`dvifm_int_parity_2026-09-20.txt` (4 sizes × {gate,curve} ×
{local,laplacian}) passes `to_bits`-exact including pyramids, block
records, table indices and sums. Strip-size, SIMD-tier, accum-selector
and block-cache replay are all bit-identical; i686 `cross` run: 9/9
int16 tests green — the integer definition is architecture-stable.

**Step-2b deviation** (`benchmarks/dvifm_int16_deviation_2026-09-20.md`,
260 TRAIN-side synthetic pairs, 6 sizes 512²..1024², 3 spec arms):
- Quantisation alone (i16+curve vs oracle): p50 rel 3e-4..6e-3, p99 ≤
  0.21, **Spearman ≥ 0.99987 on all 30 features** — near-lossless.
- Serving (i16+gate vs oracle): F2 bins identical to the curve arm (hats
  carry no `v` — f64 semantics); gate F1 = 0.27–0.58× oracle mean,
  Spearman 0.70–0.80. The fitted c0 sits ≈p10 of block contrast so the
  gate admits ~10% of blocks — a deliberately sparser statistic,
  recorded as such. Knee is free per level if a future screen wants
  gate≈curve behaviour.
- LUT error over the full u16 domain: curve-vis ≤1.1e-4 abs, hats ≤
  9.3e-3, φ/m^P ≤ 1.2e-3, gate exact except one 4-Q14-unit knee cell at
  L4 (2.4e-4 contrast wide).
- Invariants: identity→0, monotone-in-noise (new test), thread-count
  byte-identical (RAYON 1 vs 8 on the par_iter extractor), toggle-off
  and ExactF64 outputs unchanged bit-for-bit.

**Step-3 cost gate** (`benchmarks/dvifm_int16_cost_2026-09-20.md` —
two-binary A/B, 4 alternating blocks × 30 paired rounds, cpu28, sizes
64²..4096²):

| size | +f64 ms | +i16 ms | i16/f64 |
|--:|--:|--:|--:|
| 64² | 0.175 | 0.235 | 1.34 (LUT-bake fixed cost) |
| 256² | 3.020 | 2.126 | 0.70 |
| 1024² | 46.891 | 31.800 | 0.68 |
| 2048² | 194.149 | 122.501 | 0.63 |
| 4096² | 774.006 | 483.285 | 0.62 |

Fits: f64 = −0.28 + 46.16 ns/px; i16 = +0.89 + 28.77 ns/px. RSS: +15.9
B/px (f64) vs +4.1 B/px (i16) — ~4× less working set. Anchor drift: B
blocks ran 5–10% faster box (recorded, not smoothed; i16 stays ~0.7×
even anchor-normalised).

**Bar verdicts (added ms, p95 ≤ 50 @1024² / ≤ 200 @2048²)**
- **Int16: PASS** — 31.8/122.5 ms medians; worst-round bounds
  37.7/152.3 ms still under both bars.
- **ExactF64: AT/OVER** — 46.9/194.1 ms medians sit within 6%/3% of the
  bars; tail crosses both (worst-round bounds 88.7/321.3 ms).

## Tests run (final state)

- `cargo test --features feature-regime-v2,training dvifm` — **52/52
  green** (incl. all 9 new int16 tests).
- `cargo test --features feature-regime-v2 dvifm` — 42/42 green.
- `cross test --target i686-unknown-linux-gnu` — 9/9 int16 green.
- `just clippy`, `just lint-scripts`, `cargo fmt --all -- --check` —
  clean. The 5 pre-existing `zensim-validate` bake_surface failures were
  left alone per the prompt.

## NOT done

- No second fitted screen — the phase-2 verdict stands; this phase only
  established the integer serving definition and its cost. Whether the
  gate-F1 sparser statistic ranks better/worse in a re-screen is open
  (the block cache now emits in the served int domain, so fitting can
  happen there directly).
- `peak` (soft-peak field) intentionally absent from the i32 record —
  pooled features never read it; if a fitter ever wants it in the int
  domain it needs a defined integer form first.
- Neon/wasm measured only via unit-tier parity (bit-exactness), not
  wall-clock; i686 got the cross run, no perf numbers there.
- HDR/YCbCr input-plane paths exist (`DvifmInputPlane`) but were not
  exercised in the cost gate (SDR XYB-Y serving path only).
- 4096² shows no strip/streaming change — the pump architecture is
  unchanged; only arithmetic changed.

## Evidence

`~/tmp/devin/dvifm3-ab/` (A/B blocks, rounds JSONs, RSS, analyze.py,
binary.sha256, competing_procs, run.meta), `~/tmp/devin/dvifm3-dev/`
(pair TSV, three spec arms, three CSVs + manifests, deviation_report.json,
584 MB int block cache), `~/tmp/devin/dvifm3_progress.log`,
`benchmarks/dvifm_perf_2026-09-19.md`,
`benchmarks/dvifm_int16_deviation_2026-09-20.md`,
`benchmarks/dvifm_int16_cost_2026-09-20.md`.
