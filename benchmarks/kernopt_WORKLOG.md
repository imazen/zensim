# KERNOPT worklog — optimize the kernels the winning models run, bit-exact

Lane: SWE-2 herdr tab (devin-swe2-kernopt), 2026-10-03. Workspace:
`zensim--rev4vec` on anon change over `main@origin` 3110f90f. Local bookmark
`quarantine/swe2/kernopt` (never pushed). Scratch `/var/tmp/kernopt`,
`CARGO_TARGET_DIR=/var/tmp/kernopt/target`.

Goal: reduce serving cost of the two winning feature sets — **v2+basic**
(504 cols: f0–f155 + f372–f719) and **by_v2fy** (420 cols, ids in
`benchmarks/costset2_2026-10-03.candidate_ids.json`) — through
`BakeScorer::compute` (scalar) and `prepare_steering`/`SteeringSession::compute`
(prepared map), at Rev3 and Rev4, 1 MP and 4 MP, single thread. Bit-exact at
every revision and tier.

## Profile methodology

- Bakes: `/var/tmp/steercheck/gate3/{rev3dense,rev4dense}/human-{v2basic,by_v2fy}-h128-full-s5101.bin`.
  Metadata TLV-verified: v2basic declares 504 ids (0–155, 372–719), by_v2fy
  declares 420 ids (13–25, 39–155, 401–429, 459–719); revision stamped per dir.
- Harness: `extract_paths_bench` (`ZEN_XP_RSS=bake`, `ZEN_XP_MODELS=<manifest>`,
  `ZEN_XP_SIZES=1024` or `2048`, `ZEN_XP_PREPARED=1` for the map cell),
  `RAYON_NUM_THREADS=1`.
- Profiler: valgrind `--tool=callgrind`, `ZENSIM_MAX_TIER=v3` (valgrind cannot
  run AVX-512; a new `apply_tier_cap_from_env` in the bench mirrors
  `featcost::apply_tier_cap`). Profiles in `/var/tmp/kernopt/cg/*.out`,
  annotate script `analyze.sh`, tables `ranking_all.txt`.
- Numbers below are **self** Ir shares (callgrind_annotate default, threshold
  99.5), % of total program Ir for 3 iterations of the cell.

## Pre-change kernel ranking (self Ir %, 2026-10-03, v3 tier)

### Scalar score — v2basic

| kernel | r3 1MP | r3 4MP | r4 1MP | r4 4MP |
|---|---|---|---|---|
| `VWin::slide64x8` (f64 vblur window, canon) | – | – | 23.39 | 23.68 |
| `fused_vblur_ssim_inner_v3` (era2 f32) / `..._canon64_vec_v3` (r4) | 21.74 | 21.81 | 12.74 | 12.79 |
| `fused_blur_h_ssim_inner_v3` / `..._rec64_rows_v3` | 16.99 | 17.06 | 14.41 | 14.52 |
| `dense_block_kernel_era2_entry_v3` / `..._canon64_vec_v3` | 14.36 | 14.28 | 12.88 | 12.92 |
| `box_blur_v_copy_inner_v3` (f32, both revs) | 12.11 | 12.13 | 6.83 | 6.89 |
| `box_blur_h_inner_v3` (f32, both revs) | 8.49 | 8.52 | 4.79 | 4.84 |
| `gradient_block_kernel_entry_v3` / `..._canon64_vec_v3` | 6.86 | 6.85 | 6.90 | 6.92 |
| `srgb_to_positive_xyb_planar_inner_v3` / `srgb_xyb_canon_vec_v3` | 4.77 | 4.78 | 2.69 | 2.71 |
| `stream_phase_b` self (fold/emit driver) | 3.18 | 3.20 | 1.85 | 1.88 |
| memset+memcpy | 4.03 | 4.04 | 2.56 | 2.11 |

### Scalar score — by_v2fy

| kernel | r3 1MP | r3 4MP | r4 1MP | r4 4MP |
|---|---|---|---|---|
| `VWin::slide64x8` | – | – | 21.49 | 21.93 |
| `fused_vblur_ssim_inner_v3` / `canon64_vec_v3` | 18.90 | 19.22 | 11.77 | 11.88 |
| `fused_blur_h_ssim_inner_v3` / `rec64_rows_v3` | 14.80 | 15.03 | 13.29 | 13.46 |
| `dense_block_kernel` (era2/canon64) | 12.54 | 12.61 | 11.90 | 12.00 |
| `box_blur_v_copy_inner_v3` | 10.53 | 10.69 | 6.29 | 6.39 |
| `box_blur_h_inner_v3` | 7.41 | 7.50 | 4.43 | 4.48 |
| `gradient_block_kernel` (era2/canon64) | 6.78 | 6.45 | 6.37 | 6.42 |
| `srgb_to_xyb` (rev variant) | 8.33 | 8.46 | 4.97 | 5.05 |
| memcpy | 3.04 | 3.95 | 1.81 | 2.05 |

### Prepared map — v2basic

| kernel | r3 1MP | r3 4MP | r4 1MP | r4 4MP |
|---|---|---|---|---|
| `fused_blur_h_ssim_inner_v3` / `rec64_rows_v3` | 17.20 | 18.84 | 18.28 | 19.61 |
| `VWin::slide64x8` | – | – | 13.89 | 14.95 |
| `fused_vblur_ssim_inner_v3` / `canon64_vec_v3` | 10.63 | 11.61 | 7.81 | 8.34 |
| memcpy (`__memcpy_avx_unaligned_erms`) | 9.18 | 7.02 | 6.50 | 4.22 |
| `attr_pass_b_main_entry_v3` | 8.32 | 9.04 | 5.90 | 6.30 |
| `dense_block_kernel` | 6.79 | 7.35 | 7.65 | 8.16 |
| `box_blur_v_copy_inner_v3` | 5.73 | 6.25 | 4.06 | 4.35 |
| `retention_pass_b_all_scales` (scalar driver) | 5.03 | 2.12* | 3.57 | 3.84 |
| `attr_pass_b_grad_entry_v3` | 4.40 | 3.87* | 3.15 | 3.26 |
| `box_blur_h_inner_v3` | 4.02 | 3.83 | 2.85 | 3.06 |
| `srgb_to_xyb` (rev variant) | 3.76 | 7.56 | 2.66 | 3.00 |
| `gradient_block_kernel` | 3.24 | 3.09* | 4.10 | 4.37 |
| `BinAccum::add_scale_plane` | 1.71 | 1.30 | 1.21 | 1.30 |

### Prepared map — by_v2fy

| kernel | r3 1MP | r3 4MP | r4 1MP | r4 4MP |
|---|---|---|---|---|
| `fused_blur_h_ssim` (era2/rec64) | 14.71 | 16.45 | 16.31 | 17.46 |
| `VWin::slide64x8` | – | – | 12.37 | 13.30 |
| `fused_vblur_ssim` (era2/canon64) | 9.08 | 10.14 | 6.99 | 7.44 |
| memcpy | 8.69 | 7.84 | 6.43 | 5.16 |
| `attr_pass_b_main_entry_v3` | 7.13 | 7.91 | 5.28 | 5.62 |
| `srgb_to_xyb` | 6.45 | 7.56 | 4.77 | 5.36 |
| `dense_block_kernel` | 5.83 | 6.44 | 6.85 | 7.28 |
| `box_blur_v_copy_inner_v3` | 4.90 | 5.46 | 3.62 | 3.88 |
| `retention_pass_b_all_scales` | 4.29 | 2.38 | 3.17 | 3.42 |
| `attr_pass_b_grad_entry_v3` | 3.87 | 4.14 | 2.86 | 2.94 |
| `box_blur_h_inner_v3` | 3.45 | 3.83 | 2.55 | 2.72 |
| `BinAccum::add_scale_plane` | 2.94 | 3.27 | 2.17 | 2.32 |

(* some by-size percentages for prepared r3 4MP taken from the truncated top-14
extract; full tables in `/var/tmp/kernopt/cg/ranking_all.txt` and the raw
`.out` files.)

## Cross-cell kernel ranking (what to attack, in order)

1. **`VWin::slide64x8` + `fused_vblur_ssim_canon64_vec_v3`** — the Rev4 f64
   vertical-blur+SSIM fusor. 33–40% combined self of Rev4 scalar cells,
   20–23% of Rev4 prepared. slide64x8 alone is the single largest kernel in
   every Rev4 cell.
2. **`fused_blur_h_ssim_*` (h-blur+SSIM row kernel)** — #1 or #2 self cost in
   all 16 cells: 13.3–19.6%. Rev3 uses `inner_v3` (f32 era2), Rev4 uses
   `rec64_rows_v3` (f64).
3. **`dense_block_kernel`** (era2 f32 / canon64 f64) — 5.8–14.4%.
4. **`box_blur_v_copy` + `box_blur_h`** (f32, shared by both revs) — combined
   ~10.7–20.6% scalar, ~8.4–10% prepared.
5. **`gradient_block_kernel`** — 3.1–6.9%.
6. **Prepared only:** `attr_pass_b_main_entry_v3` + `attr_pass_b_grad_entry_v3`
   (~9–12% combined), `retention_pass_b_all_scales` scalar driver,
   `BinAccum::add_scale_plane`, and 4–9% raw memcpy (`FoldRetention::copy_strip`
   & friends — buffer-reuse candidates).
7. **`srgb_to_positive_xyb` / `srgb_xyb_canon_vec_v3`** — 2.6–8.5% (larger
   share in by_v2fy where everything else shrank).

All listed kernels are already `_v3` arcane bodies — the remaining wins are
memory-level: less copying, fused passes, better blocking, bounds-check
removal, and any scalar residue inside the `*_v3` bodies.

## Optimization log

### K1–K4 (Rev4 canon64 lane-parallel bodies) — landed, parity green

- **K1 `VWin::slide64x8`**: the f64x8 vertical window slide was a plain
  `#[inline]` method; rustc emitted it out-of-line and every `f64x8` op inside
  became a `call` to a `core::arch` shim (`__mm256_add_pd` etc.) that cannot
  inline back into the `#[target_feature]` caller. `#[inline(always)]` alone
  did NOT fix it (verified in disasm — call persisted). Fixed by turning the
  body into `macro_rules! vwin_slide64x8` expanded textually inside the two
  `#[magetypes]` canon bodies (`fused_vblur_ssim_canon64_vec`,
  `fused_vblur_edge_canon64_vec`), so the ops compile in-region. Same
  `state + add − rem` per-column recurrence, `chunks_exact` windows for the
  vector part, scalar `width % 8` tail. Also pre-sliced the add/remove rows
  once per call (bounds checks carried by the chunk iterators).
- **K2 `fused_blur_h_ssim_rec64_rows`**: hoisted the eight per-lane source
  and destination row slices out of the x loop (`rec64_rows8_mut` helper for
  the split borrows); gathers/stores are now row-local and provable. Tail
  rows still run `fused_blur_h_rec64_row` verbatim.
- **K3 `fused_vblur_ssim_canon64_vec` + `fused_vblur_edge_canon64_vec`**:
  whole-plane `src`/`dst` slicing at entry, per-row window-state slices cut
  to `full * 8`, row-local vector loads/stores; scalar tail unchanged
  (`vblur_ssim_elem_canon` verbatim).
- **K4 `dense_block_kernel_canon64_vec`**: pre-sliced the seven input planes
  to `height * width` plus per-row full-vector-region slices; the scalar
  remainder still calls `dense_elem_canon` verbatim. 263 `slice_index_fail`
  call sites → 0.

**Result (callgrind, `v2basic.r4.scalar.1024`, 3 iters):** 5.38 B Ir →
3.51 B Ir (−34.7%). New self-cost top 5: vblur canon64 21.5%, dense canon64
17.6%, rec64 rows 13.3%, gradient canon64 10.6%, box_blur_v 10.5%.
`slide64x8` no longer exists as a symbol; all four arcane bodies have zero
`__mm256`/`slice_index_fail` call sites (verified via objdump).
Tier-parity gates all pass (`canon64` + `rec64_rows` filter tests,
to_bits-identical on every compiled tier, odd/tiny/specials).

### Residual check-site census (post K1–K4, `panic_bounds_check`/`slice_index_fail` call sites per body)

| body | v3 | v4 | v4x |
|---|---|---|---|
| box_blur_h_inner | 27 | 66 | 97 |
| fused_blur_h_ssim_inner | 20 | 54 | 94 |
| fused_vblur_ssim_inner | 61 | — | — |
| box_blur_h_into_abs_diff_inner | 40 | 103 | 103 |
| attr_pass_b_grad_entry | 53 | 53 | 53 |
| attr_pass_b_main_entry | 26 | 26 | 26 |
| dst_y_edge_mask_entry | 49 | 49 | 49 |
| downscale_2x_inner | ~ | 37 | 37 |
| fused_blur_h_ssim_rec64_rows | 23 | — | — |
| dense_block_kernel_era2_entry | 18 | 18 | 18 |


### K5 — era2 (Rev3) pre-slices: applied, ~0 measurable gain

Same whole-plane slice pattern applied to `fused_vblur_ssim_inner_{v3,v4,v4x}`,
`fused_blur_h_ssim_inner_{v3,v4,v4x}`, `box_blur_h_inner_{v3,v4,v4x}`,
`box_blur_v_copy_inner_{v3,v4,v4x}` and the fused-era2 block kernels in
`feature_v2.rs`. Post-change `slice_index_fail` counts in those bodies are
unchanged-ish (LLVM was already folding the hot checks; it cannot prove
`idx * width + col_base + 8 <= height * width` with both `idx` and
`col_base` runtime-varying, and the residual panic pads sit on cold paths —
verified in disassembly, all hot loops branch-free of checks). Kept: the
slices are cheap (one compare per plane per call) and document the length
contract; no measurable Ir delta in the r3 cells.

**Strided-extent bug found by smoke run (fixed).** The first build
truncated `input`/`output` in `box_blur_h_v4x_strided` and all six planes
in `fused_blur_h_ssim_v4x_strided` to `height * width` — but those bodies
are *strided* (tile pitch `stride = width + 16` when `width % 256 == 0 &&
height >= 16`), so the truncation cut live rows and panicked at
`blur.rs:1173` on the uncapped v4x path only (callgrind runs were v3-capped
and never touched it). Fix: slice extent is now
`height.saturating_sub(1) * stride + width` — the true access bound for
`(row_base + ro) * stride + idx`, `idx <= width - 1`. New tile-path
geometries `(256,32)`, `(512,17)`, `(256,40)` added to `RING_GEOM` and
`FUSED_GEOM`; every prior geometry maxed at width 592, so the strided tile
path had *no* unit coverage. Lesson recorded for the NEON/x86 gate notes:
v3-capped profiling plus odd-size-only tests = the strided path can only
be caught by a production-width smoke run.

### K6/K7 — prepared path: analyzed, below the 3% bar

- `attr_pass_b_main_entry_v3` (7.8% of r4 prepared cell): ~9 bounds-check +
  2 overflow compares per 8-px chunk ≈ 15% of the kernel ≈ **1.2–1.5% of the
  cell** — below bar.
- `attr_pass_b_grad_entry_*` (3.1–4.4%): same pattern, smaller ceiling.
- `FoldRetention::copy_strip` ≈ 5.1% inclusive (8 memcpys per strip):
  redirecting the blur/stats kernels to write retained planes directly would
  remove ~3.5 of the 8 copies (mu1/act/pyr copies are pinned by the
  activity-halo `gridblk_strip_wide` read, the rolling-plane eviction, and
  the `process_strip_channel` staged path) ≈ **2.5–3% of the cell at best**,
  against real halo/eviction plumbing risk. Declined per the stop rule.
- `retention_pass_b_all_scales` (3.6–5.0%): scalar coefficient math inlined
  into the driver — semantic, untouched.
- `fused_blur_h_ssim_rec64_rows_v3` prepared-vs-scalar 2.2× weight is real
  work (phase B re-walks the strips at wider windows), not redundant compute;
  inner loop already tight (252 insns, 1 branch).

### Before/after wall time — `extract_paths_bench`, bake arm

ST, `taskset -c 8 nice -n 19`, `RAYON_NUM_THREADS=1`, **native tier (v4x,
AVX-512)** — production dispatch, NOT the v3 profiling cap. 3 rounds per
arm interleaved BEFORE/AFTER, median of per-invocation ms. BEFORE =
pre-change build on parent `3110f90f` (sha256
`12c089b8705f…` — preserved `/var/tmp/kernopt/bench/`); AFTER = this tree.
Load during run: avg ~2 (idle).

| cell | 1 MP before | 1 MP after | ratio | 4 MP before | 4 MP after | ratio |
|---|---:|---:|---:|---:|---:|---:|
| v2basic r4 scalar | 183.3 | 151.7 | **1.208×** | 786.8 | 664.7 | **1.184×** |
| v2basic r4 prepared | 293.8 | 259.4 | **1.133×** | 1196.1 | 1068.5 | **1.119×** |
| by_v2fy r4 scalar | 90.6 | 74.3 | **1.219×** | 404.9 | 344.5 | **1.175×** |
| by_v2fy r4 prepared | 148.3 | 130.7 | **1.135×** | 628.1 | 565.0 | **1.112×** |
| v2basic r3 scalar (anchor) | 50.8 | 50.9 | 0.998 | 218.0 | 218.1 | 1.000 |
| v2basic r3 prepared (anchor) | 134.4 | 134.1 | 1.002 | 516.1 | 514.6 | 1.003 |
| by_v2fy r3 scalar (anchor) | 29.9 | 29.8 | 1.003 | 127.4 | 127.2 | 1.002 |
| by_v2fy r3 prepared (anchor) | 78.1 | 78.5 | 0.995 | 302.4 | 301.1 | 1.004 |

Anchors (untouched code) sit within ±0.5 % — every Rev4 gain is 20–40× the
noise floor and every AFTER round beat every BEFORE round in its cell. The
r4/r3 scalar ratio improved from ~3.6× to ~3.0× (v2basic 1 MP); callgrind
(v3 cap) showed −35 % Ir on the same path. Rev4's remaining premium is the
LanesF64 widening math (`vcvtps2pd`/`vaddpd` per element), the eight
memory-resident f64×8 accumulators in vblur, and the prepared path's
retention copies — all semantic per the current formula revision.

### Remaining split (r4 scalar 1 MP, self-cost, post-change)

vblur canon64 21.5 %, dense canon64 17.6 %, rec64 13.3 %, gradient canon64
10.6 %, box_blur_v 10.5 %, box_blur_h 7.3 %, srgb_xyb 4.1 %. All at or near
their semantic floors under the bit-exact constraint — next lever would
need a numeric revision (drop the f64 canonical lanes or reassociate the
pairwise tree), which is exactly the documented stop condition.
