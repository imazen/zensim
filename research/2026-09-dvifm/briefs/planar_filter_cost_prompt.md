# Task: planar filter matrix — accuracy AND measured cost (DVIFM pyramid + zensim scales)

Repo `/home/lilith/work/zen/zensim` (jj-colocated). Run this only when the box is QUIET — no other fit/bench/
training job (check `uptime`, `free -h`, and that no `fit_standalone.py`/`zensim_mlp_train` is running). Read
`benchmarks/dvifm_block_gates_2026-09-19.md`, `benchmarks/speed_matrix_2026-09-18.md` (protocol to copy),
`docs/FITTED_CONSTANT_GUARDS_2026-09-19.md`, `docs/PREREG_SCALES_PLANES_2026-09-19.md`, `CLAUDE.md`.
`/home/lilith/.claude/CLAUDE.md` (Performance Optimization; the sweep discipline) and `~/work/zen/CLAUDE.md` bind you.
Outputs: `/mnt/v/output/zensim/planar-filters-2026-09-19/` (≤10 GB). Work autonomously; a supervisor verifies artifacts.

## Why
Nobody has measured the cost of the plane-building filters themselves. Phase 1 measured the whole DVIFM family
(+31.6 ms at 1024², +120.5 ms at 2048², f64, Y only) with no breakdown, and zensim's own 2×2 box scales have never
been isolated from the rest of extraction. Every later choice — which band, which downsampler, whether the local
band replaces the Laplacian, whether an i16 kernel is worth it — needs cost per filter, measured, not argued.

## Part 1 — the stage split (do this first; it decides what is worth optimising)
Instrument the existing DVIFM path (`Phase::DvifmKernel` + the zenbench arms `fold956_csfw`/`fold986_dvifm`) to
report time in: (a) normalise/convert input, (b) downsample chain, (c) expand + subtract (band construction),
(d) block scan max/min, (e) per-block math, (f) ring/alloc. Sizes 256²/1024²/2048², 1 pinned thread, ≥30
interleaved rounds, dispersion reported. Commit the table before touching any filter.

## Part 2 — the filter matrix
Implement each as a crate-private option behind the existing spec (no new public API, no new registered slots):

**Downsamplers** (identical on both sides of a pair; reflect-101 borders throughout)
| id | kernel | note |
|---|---|---|
| `box2` | 2×2 average | zensim's production scale step |
| `bin121` | [1 2 1]⊗[1 2 1]/16, ↓2 | DVIFM's own (current default) |
| `bin1331` | [1 3 3 1]/8 per axis, ↓2 | wider, still shift/add-friendly |
| `mitchell` | Mitchell–Netravali, ↓2 | 4-tap reference quality (via `zenresize` — read its local README/source first; no foreign resamplers) |
| `lanczos3` | ↓2 | 6-tap upper bound on cost/quality |

**Band constructions** (on the chosen downsampler)
| id | band | note |
|---|---|---|
| `lap` | G_l − E(G_{l+1}), E = zero-insert + 4·B | DVIFM's own; needs the expand pass |
| `local` | G_l − B²·G_l | no expand, no 2^l lattice; the design's first ablation |
| `boxres` | s_k − 11×11 box mean | zensim's residual; O(1)/px via running sums |

Report for every (downsampler × band) cell: taps per output pixel, passes over the plane, extra planes held live,
halo rows needed for streaming, and the MEASURED time and peak RSS.

## Part 3 — measurement protocol (non-negotiable)
Sizes 64², 256², 1024², 2048², 4096²; 1 thread pinned (`taskset`) and 8 threads; zenbench interleaved (never
criterion), ≥30 paired rounds, `nice -n19 ionice -c3`, build WITHOUT `-C target-cpu=native`, competing processes
recorded. Fit and report `total = α + β·pixels` with BOTH terms per cell — a ms/MP number alone is not a result.
Memory via `/usr/bin/time -v`. Report f64 (today's kernel) and, where the kernel already supports it, f32; do not
implement the i16 kernel here.

## Part 4 — accuracy side, so cost is not compared in a vacuum
For each cell, on TRAIN images only: the label-free stability gates (1-px shift of the pair; 8×8 codec-grid phase
0..7; centre-crop vs full) reported as relative feature spread, and the DVIFM feature correlation against the
`lap`+`bin121` reference. No training, no screens, no holdout reads (never CID22-B/AIC/AIC2026/KonJND-val/
KonFiG-test/KADID terminal). If a cheaper filter matches the reference features within the shift-gate noise, that
is the headline.

## Records
`benchmarks/planar_filter_cost_2026-09-19.{md,json}` (+ `.pointer.md`; nothing >30 KB in git), MISSING list first,
git commit + exact commands + host state, the stage-split table, the matrix with α and β, the stability table, and
a one-paragraph recommendation naming the cheapest cell that is not worse than the reference. Preregister nothing
(this is measurement, not selection), but state explicitly which numbers are measured and which are op counts.

## Scope note (user, 2026-09-19)
No luma-only arms anywhere in this phase — all DVIFM work is 3-plane Y′CbCr. Y′-vs-XYB-Y as a single plane is
settled as equivalent (CID22-A +0.006 for XYB-Y, TID/KADID −0.003 — opposite signs, both small).

## Rules
`jj` only, small commits, **DO NOT PUSH**; heavy work through `~/work/zen/scripts/run-heavy --mem 16G --jobs 8 --
… 2>&1 | tee ~/tmp/devin/<n>.log`, one at a time; scratch only `~/tmp/devin/`; `cargo fmt --all -- --check`,
`just clippy`, `just lint-scripts` clean; no public API; never relax/`#[ignore]` a test (5 pre-existing
`zensim-validate/tests/bake_surface.rs` failures stay); refresh `.workongoing` every ≤2 min; touch no other repo;
no GitHub writes; no `pgrep -f`. Report once: `~/tmp/devin/planar_filter_DONE.md` (or `_BLOCKED.md`), progress to
`~/tmp/devin/planar_filter_progress.log`.
