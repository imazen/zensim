# PLAN — fast-class extraction kernel (156 + cheap), 2026-09-05

**Pre-registered before any measurement or code.** Lane: KERNEL. The parallel
CAMPAIGN lane owns the MODEL side (which 156+cheap head competes with a 944
MLP on rank); this lane owns the EXTRACTION KERNEL that serves it.

User directive (2026-09-05, verbatim): *"… find a high performance but 944-mlp
competitive 156 or 156 plus cheap model, you can improve the kernel"*.

Mission constraint (memory `user_zensim_mission`, 2026-08-31, verbatim): *"we
can do an era break in order to push performance higher, remember the point of
zensim is to be extremely fast, and as good or better than ssim, and be good at
hdr."* — so a byte-moving lever is **presentable as an era-break candidate**,
never silently flipped, and era breaks are **batched**.

---

## 0. Scope

**In scope:** the cost of producing the fast class — v1 basic `f0..155`, plus
the cheap extras that ride the same walk (`V1PoolsMode::Peaks` = `f156..227`;
`V1FreeExtras::RawMoments` and `::RawMomentsPlusBoundedErr` = the 40 + 24 free
v2-layout slots), through `Zensim::compute_folded720_append_features_streaming`
and through the product entry `ZensimProfile::D` on a **default** build.

**Out of scope:** model training, rank, dial, any `bake_*` tool, the 944 pool
block's own cost (already priced, `pools_full_extraction_2026-08-30`), HDR
front-end, GPU.

**Non-negotiable:** no shipped feature byte moves in this lane without a
registered era break. A lever that is only reachable through an era break gets
**measured and registered, not flipped**.

---

## 1. What is already measured — do NOT re-derive

Sources: `benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md`,
`era2_perf_break_2026-08-31.md`, `fold_engine_2026-08-31.md`,
`fold_mt_scaling_2026-08-31.md`, `fold_footprint_2026-08-31.md`,
`profile_d_notax_2026-09-01.md`, `free_features_2026-09-01.md`,
`free_features_classC_2026-09-04.md`, `d_ship_flip_2026-09-05.md`.

Facts this plan builds on and will not re-measure:

- The **rem-ring** (H-blur redundant-gather removal) landed: −8.29 % buffered
  v1-372, −5.57 % zeroed-944 fold, −6.22 % live-pools fold at 576²/v3.
- **H-tile PACKING** landed and is worth 1.15× @5 MP / 1.73× @21 MP at 1T;
  **restricting a loop range instead bought nothing** (1.06× @2304², 0.96 ×
  @4608²). *Locality comes from LAYOUT, not loop bounds.* Do not re-attempt an
  x-range refactor.
- **Row banding of phase A LOSES** (+15.6 % / +5.5 %) — the activity closure is
  `±2·BLUR_RADIUS` = 62 % redundancy on a 32-row band.
- **Fusing the activity abs-diff into the H-blur LOSES** (+1.04 % 944-full,
  +2.01 % 372-only) — post-rem-ring the H blur gathers one strided column and
  fusing makes it two.
- **`map_init` per-band scratch LOSES** (allocating parallelism: 7.75 → 10.00 ms
  at 3T); persistent per-band scratch slots are the fix and shipped.
- **`fused_vblur_ssim` fission is RETIRED** — 27 of its 28 spill loads are
  per-column-group setup, only 1 load / 0 stores inside the innermost loop.
- **`dense_block_kernel` is the MT ceiling and ERA-LOCKED.** Band/row-partial
  merging is bit-exact only under `POOL_SIMD` (v4x) AND `width % 8 == 0` at
  every scale; neither holds generally (−2 ulps with a scalar tail, 13 ulps with
  per-pixel pools). Amdahl UPPER bound on fixing it: **1.17× @8T, 1.23× @16T**
  — against re-extracting every 944 table and re-training every 944 model.
- **Y-channel imbalance has no free fix** — X and B are the *small* channels, so
  overlapping their phase B with Y's phase A shortens nothing.
- `V1PoolsMode::Peaks` **costs the same as `Off`** to compute (the peak
  accumulators are the fused V-blur's unconditional tier).
- Raw moments: **+0.8–1.6 %**; the 944 *layout* alone is inside noise.
- **`feature-regime-v2` is default-on since 2026-09-01**, so `ZensimProfile::D`
  gets the fold on a plain `cargo add zensim` build.

## 1b. The two measurement hazards that invalidate results here

Both are measured, both bind this lane:

- **ASLR, ±10 % at 2304², bimodal (~334 / ~360 ms), not noise.** Protocol:
  ONE binary + RUNTIME arms; **identical-BYTE-LENGTH env values** (`156` /
  `15c` / `15f` / `15x` — three characters, deliberately); arms interleaved;
  min of N walks in a process; min over ≥15 process starts with ASLR ON; a
  **bit-identical control arm** carried throughout. `setarch -R` is a second
  opinion, never primary.
- **zenbench can degenerate under a tight wall budget** and report a
  spuriously *near-zero* mean for every arm at once. `min()` does not protect
  against a harness that reads LOW. Therefore: scale `ZEN_S2_WALL_S` with size
  (≥60 s at 2304²), and **validate every reading at collection time** against a
  stable reference arm's plausible floor.
- **Own-process contention is real even niced.** No concurrent `cargo`, no
  other lane's fits, during a pinned sweep. The sweep script self-checks load
  before each cell and refuses rather than producing a contaminated number.

---

## 2. H1 — the cost map (deliverable 1)

**Question:** where does the fast walk's time go, by kernel, and how does that
change with size / threads / tier?

**Instrument:** `ZENSIM_FOLD_TIMING=N` (already in tree,
`zensim/src/fold_timing.rs`, 21 phases incl. `Producer`, `ProdConvert`,
`ProdDownscale`, `AWall`/`ABusy`, `BWall`/`BBusy`, `BandBusy`, `BlurHWall`,
`BlurBandBusy`, `Between`, `MeanOffset`, `DenseKernel`, `GradKernel`,
`AppendKernel`, `PhaseAV2Planes`, `PhaseAAppendPlanes`) driven through
`zensim/examples/foldapp_stream_bigpair.rs`. `occ = busy / (wall × threads)` is
the occupancy column — a phase with `occ ≈ 1/threads` is serial.

**Grid:** arms `156` / `15c` / `15f` / `15x` / `372` / `944full`
× sizes 576² / 1152² / 2304² × threads 1 / 8 / 16 × tiers native(v4x) and
`ZEN_S2_CAP_V3=1`(v3/AVX2). Callgrind (`--no-default-features`-style v3 build)
for the **serial instruction** attribution only, with the tier caveat stated on
every number.

**Pre-registered reporting rule:** every published cell carries size, threads,
tier, arm, and the control-arm reading. No "ms/MP" without the intercept, and
per CLAUDE.md the intercept here is **negative** (per-pixel cost RISES with
size) — so the linear model is reported as *failing*, not as a fixed-cost
saving.

**Success = the map exists and is honest.** H1 has no pass/fail bar; it is the
map every lever below is judged against.

---

## 3. Levers, with pre-registered gates

**Universal admission gate (binding, all levers):**

- **G-EXACT.** `cargo test --workspace` green, and specifically
  `fold_engine_parity` (11 tests, `to_bits()` over 18 geometries × {serial,
  rayon} × pools 1/2/3/8/16), `v1_golden_bytes` (incl. the non-tight fixture),
  `folded_v1_only_matches_full_walk`, `v1_feature_width_pure_function`,
  `free_extras_are_pure_addition_to_the_v1_only_walk`, plus a **new
  before/after `to_bits()` dump comparison** over the 20-geometry set via
  `ZENSIM_BIGPAIR_DUMP` on every arm. Zero differing bits.
- **G-PERF.** Landed only if measured **≥ 2 %** at some (size, threads, tier)
  cell, with the **bit-identical control arm reading 1.000 ± the cell's own
  measured noise floor**. A cell inside the noise band is *unestablished*, not
  a result. CIs reported.
- **G-API.** `cargo public-api` ZERO delta unless the lever is an explicit,
  named API addition.
- **G-CLEAN.** clippy + fmt on touched files.

A lever failing G-PERF is **reverted and recorded as falsified**, not parked.

| # | Lever | Hypothesis | Pre-registered gate / decision rule |
|---|---|---|---|
| **L1** | **`V1PoolsMode::Off` disables the band-local self-blur** — `pools_mode_for_need` never returns `Off`, and the `self_blur` predicate requires `Full \| Peaks`, so an `Off` request falls back to phase A's four strip-wide H planes. The bench arm `add156_156basic` and bigpair `156off` both request `Off`. | The instrument is pricing a walk production never runs, and pricing it **too high**. | Measure `Off` vs `Peaks` A/B at equal env-byte-length. If `Off` is slower by >2 %, this is an **instrument defect**: fix the arms to the production mode, restate any affected published fast-class number, and tell the campaign lane. No product code change is implied. |
| **L2** | **Producer** — `Phase::Producer`'s own doc says "serial by construction today", but `feature_v2_stream.rs` has `rayon::join` (sides) + `par_iter_mut` (channels) and `ADVANCE_ROWS` was raised by the fold-MT lane. | The CLAUDE.md "producer with no rayon" claim is stale; the real residual is the strip-emission cursor, not conversion. | First **settle the fact** from `fold_mt_scaling`. Publish the `Producer` / `ProdConvert` / `ProdDownscale` occupancy from H1. Only if `Producer` `occ` is low AND its wall share ≥ 10 % at some cell does an implementation follow; otherwise record "already parallel" and **correct CLAUDE.md**. |
| **L3** | **Width dependence after H-packing** — packing shipped; what remains at 2304 / 4608 for the *fast* arms specifically (H1 measured phase-A width dependence on the 944 walk). | The fast walk's H blur may still be width-bound at 4608. | Report `BlurHWall` / `BlurBandBusy` vs width for `156`/`15f`. Implement only under G-PERF. Any tile-shape change must be **packing**, never loop-range restriction (falsified). |
| **L4** | **Does `v1_only` actually skip everything at runtime?** Source says `if v2_blocks { dense … grad … }` and `v2_blocks = !v1_only`, and phase A's v2 planes are gated. | The skip is structural and complete; BANDVIS/CSFW/append cost the fast class nothing. | **Verify, don't assume:** `DenseKernel`, `GradKernel`, `AppendKernel`, `CsfwKernel`, `PhaseAV2Planes`, `PhaseAAppendPlanes` must all read **exactly zero** ns on the `156` arm. A non-zero reading is a bug to fix. This is a correctness check with a numeric bar, not a perf lever. |
| **L5** | **Cheap-slot marginal cost** — peaks (claimed zero-marginal) and raw moments (+0.8–1.6 %) and class-C. | The claims hold at every cell in the H1 grid, not just the one they were measured at. | Publish `15c`→`15f`→`15x` marginals across the grid. **Bar: ≤ +2 % on the 156 walk for the whole cheap set at every cell.** If a cell exceeds it, find the redundant pass; if none exists, report the honest number and revise the claim. |
| **L6** | **`#[rite]` / dispatch audit** — a nested `#[arcane]` in the hot path is a perf bug (memory `feedback_archmage_dispatch`); generic SIMD helpers must be `#[inline(always)]` (a 5.3× regression from an un-inlined one is already on record here). | The fast path has no nested `#[arcane]` and no un-inlined generic SIMD helper. | Static audit + `nm` on a release build: zero `raw_moments_*` / helper symbols surviving. A finding is fixed; no finding is reported as clean. |
| **L7** | **Allocation audit** — zero allocations per compare after warm-up on the fast path. | `V2Scratch` reuse is complete. | A counting-allocator test asserting **0 allocations** on the 2nd..Nth `compute_folded720_append_features_streaming` with a reused `V2Scratch`. A non-zero count is a defect to fix. |
| **L8** | **Thread policy for the product loop** — intra-image threading vs pair-level parallelism (the zenmetrics pattern: fold `.with_parallel(false)`, parallelise across pairs). | For the fast class at product sizes, pair-level wins. | Measure aggregate throughput (pairs/s) both ways at 8T/16T. Deliverable is a **documented recommended thread policy**, not code. |

**Ranking:** L1 and L4 first (cheapest, and both are potential *defects* rather
than optimisations — a wrong number in an instrument is worse than a slow
kernel). Then L5, L7, L6 (cheap, bounded). Then L2, L3 (real work, gated on
H1). L8 last.

---

## 4. Extension point for new cheap slots (deliverable 3)

The free-features lane established the pattern: generic helper pairs
(`raw_moments_accumulate{8,16}` / `raw_moments_finish{8,16}<T: F32x{8,16}Backend>`)
plus one scalar pair, `#[inline(always)]`, replacing what had been 6 vector +
4 scalar hand-copies per tier. `#[rite]` does **not** apply — it resolves
`#[target_feature]` from a concrete token, and these are generic over a backend
trait.

**This lane's obligations:**

1. Verify the pattern still holds after any change here.
2. Add a test that a **new dummy cheap slot** costs ≤ 1 % and is emitted
   bit-exactly **in both engines** (fold and buffered).
3. Document in `docs/FEATURE_SET_IDS.md` how a new cheap slot gets its id —
   it names a SUBSET of the existing append-only numbering and never renumbers,
   so a new slot extends the compute-token vocabulary and changes `slots_hash8`.

---

## 5. Era-break policy for this lane

If a lever is only reachable by moving bytes, it is **registered, not flipped**:
what changes, by how much (max |Δ| and fraction of cells over tolerance), which
tables and which models it invalidates, and the re-extraction / re-training wave
it implies. Batched with any other pending era break. Every era break still owes
determinism, thread-invariance, declared numerical equivalence, and
rank-preservation gates. `dense_block_kernel` restructuring is the known
example and its Amdahl ceiling (1.17× @8T) is already below what its
re-extraction wave would cost — this lane does **not** propose it.

---

## 6. Deliverables

1. The cost map (H1), per kernel × size × threads × tier, with occupancy.
2. The lever table with measured deltas, CIs, and the control-arm reading.
3. Landed bit-exact levers, each with its `to_bits()` proof.
4. Registered era-break levers with their numbers — none flipped.
5. A `zensim-bench/benches/ssim2_speed_bar.rs` arm for **"D + cheap"** so the
   campaign lane's W4 is one command. Note the bench today has **no
   `ZensimProfile::D` arm at all** (its `zensim_B` arm is the buffered B walk),
   and its `add156_156basic` arm requests `V1PoolsMode::Off` — see L1.
6. Doc corrections in place (CLAUDE.md producer claim, if L2 confirms stale).
7. `cargo test --workspace` green, clippy/fmt clean on touched files,
   `cargo public-api` zero delta, CHANGELOG lines.

**Artifacts:** `/mnt/v/output/zensim/kernel-2026-09-05/`.
**Record:** `benchmarks/kernel_fastclass_2026-09-05.md`.

---

## 7. Budget

Measurement is serialized behind box idle (the campaign lane's extraction and
fits have priority; the sweep script self-checks load and refuses rather than
producing a contaminated number). Levers are implemented in ranked order and
each is landed or falsified before the next begins — no parked work.

---

# LANE 2 — the front end (pre-registered 2026-09-05, before any measurement or code)

Lane 1 closed with a map whose headline is *"for the fast class the FRONT END is
the biggest phase, and it is not the part anybody has been optimising"*:
producer **32.5 %** of the 6.509 ms `156` walk, `ProdConvert` alone **22.0 %**,
`ProdDownscale` **10.5 %**. Lane 1 declined L2 on the grounds that the producer
is *already parallel* (sides `rayon::join`, cascade 6-way) — which settles
**scheduling** and says nothing about the **per-pixel work inside a chunk**.
Lane 2 takes that inside.

## L2.0 Stage split (deliverable 1) — no pass/fail bar

`ProdConvert` is one timer over five distinguishable stages: the `[u8;3]`
gather + `srgb_u8_to_linear` LUT expansion, the 3×3 opsin matrix, the three
`cbrt_midp` calls, the XYB mix + positive shift, and the buffer plumbing
(`rgb_buf` staging copy, `Vec<&mut [f32]>` chunk collections, the pad spread).
Instrument: **callgrind Ir attribution**, which is deterministic and therefore
**valid on a loaded box** — the one measurement in this lane that does not have
to wait for an idle window. Tier caveat stated on every number: valgrind cannot
execute AVX-512, so the profile is the **`v3`/AVX2** tier
(`X64V3Token` = AVX2+FMA+BMI1/2) and kernel *ratios* may shift at `v4x`.
Wall-clock confirmation of any landed lever follows the §1b ASLR protocol on an
idle box.

## L2.x levers, with pre-registered gates

Universal admission gate is §3's (G-EXACT / G-PERF / G-API / G-CLEAN),
unchanged. **G-PERF is ≥ 2 % of the fast walk at some cell with the
bit-identical control arm reading 1.000 ± that cell's own measured noise
floor.** A lever inside the noise band is *unestablished*, not a result. A
lever that fails is reverted and recorded as falsified.

| # | Lever | Hypothesis | Pre-registered gate / decision rule |
|---|---|---|---|
| **L9** | **The `rgb_buf` staging copy is a per-chunk `Vec` allocation.** `streaming.rs:1299` does `Vec::with_capacity(rows*width)` inside `process_chunk`, then memcpys every row into it, then hands it to the kernel. At 1152²/64-row chunks that is a **221 kB** buffer per chunk (past glibc's 128 kB mmap threshold ⇒ fresh pages + faults + `munmap`), and it is 3 more `Vec`s for the `p{0,1,2}_chunks` collections. | These are a large share of the **~175 per-walk allocations lane 1 could not locate** (§3.2), and the mmap/first-touch is real time at ≥1152². | Count first: extend `fastclass_alloc_steady_state` attribution to name the producer. Then remove what is removable **bit-exactly** — the three chunk-collection `Vec`s become indexed `par_chunks_mut` zips (same boundaries, same body). The staging copy is only removable bit-exactly if the kernel sees the **same element count**; see L10. Land under G-PERF, or record the allocation win alone with the wall-clock delta stated as unestablished. |
| **L10** | **Per-row kernel calls are NOT bit-exact** and this must be established before anyone tries it. The kernel's vector body uses `cbrt_midp` and its scalar remainder uses `cbrtf_fast` — *different cube roots*. A call over `rows·width` elements has remainder `(rows·width) % 16`; per-row calls have remainder `width % 16` on **every** row. | Equal only when `width % 16 == 0`. Otherwise per-row conversion moves bytes. | **Prove it by measurement, not by reading:** a test that converts the same pixels in one bulk call and in per-row calls and compares `to_bits()`, at a width with `width % 16 == 0` and at one without. If the non-aligned case differs, per-row conversion is registered as an **era break**, never flipped, and the staging copy stays for non-aligned widths. |
| **L11** | **`cbrt_midp` carries sign/zero handling this call site cannot need.** The kernel's own comment states `mixed >= K_B0 ≈ 0.0038` — "no zeros, denormals, NaN, or infinities". `cbrt_midp` still does sign-extract, `abs`, and a zero-select per call, ×3 per 16 pixels. | A positive-domain cube root is bit-exact here and strictly less work. | Bit-exactness is the gate: the positive-domain form must reproduce `cbrt_midp` `to_bits()`-exactly **over the whole reachable input domain** (all 2²⁴ `mixed` values in `[K_B0, matrix max]`, brute-forced), not on samples. It lives in **zensim** — archmage is a different repo and is not to be touched. Land under G-PERF. |
| **L12** | **The de-interleave is 48 scalar LUT loads + 3 stack arrays per 16 pixels.** `r_arr[i] = srgb_u8_to_linear(p[0])` etc., then `from_array`. | LLVM may already turn this into shuffles (CLAUDE.md's fixed-array pattern) — or the 256-entry LUT gather may block it. | **Read the disassembly before proposing anything** (`objdump` on the release binary), per the `fused_vblur_ssim` lesson: locate the cost against loop structure first. A LUT lookup is bit-exact under any correct implementation, so a SIMD form is admissible; a polynomial is not (era break). Report clean if LLVM already vectorised it. |
| **L13** | **`ProdDownscale` is 10.5 %** and `downscale_2x_into` is called 6× per scale per produce with a `par_iter_mut` fan-out of 6. | Unknown; not yet split. | Report its Ir share and whether it is bound by the 2×2 average or by the plane plumbing. Implement only under G-PERF. |
| **L14** | **`linear-srgb` batch API.** `srgb_u8_to_linear` already routes to `linear_srgb::default`; the crate also ships batch/SIMD forms. | A batch call may beat the scalar-per-component form. | Admissible **only** if it is bit-identical on **all 256** u8 inputs against the current scalar entry point, proven by an exhaustive test. Otherwise era break, registered not flipped. |

## L2 obligations carried from lane 1

* Finish lane 1's **NOT MEASURED** cells when the box goes idle: L1's 8T half
  (`Off` vs `Peaks`), and the 8T/2304² + both-tier sweep of the shipped `D`
  path through the new `zensim_D` bench arm. Publish as MEASURED into lane 1's
  tables, or leave them NOT MEASURED **with the reason**.
* Keep lane 1's ratchets green: `fastclass_alloc_steady_state` (bar 200/walk —
  *lower it, never raise it*) and
  `self_blur_sizing_predicate_matches_the_strip_loop_predicate`.

## Measurement discipline specific to this lane

The box is carrying the campaign lane's fits and extractions. **Callgrind Ir is
load-immune and is therefore the primary instrument for the stage split.**
Every wall-clock cell goes through `scripts/kernel_fastclass_sweep.sh`, which
self-checks load and emits `SKIPPED_BOX_BUSY` rather than a contaminated
number. No cell is published without its control-arm reading. No idle-attached
waiting: the sweep is a script that gates itself.
