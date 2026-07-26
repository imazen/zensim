# Streaming-only foldapp — C0 design note (2026-07-26)

Chunk C0 of `STREAMING_FOLDAPP_PLAN_2026-07-26.md` (plan commit `6676aaf2`).
Study sources: `zensim/src/{streaming,fused,blur,feature_v2}.rs`,
`zensim/tests/streaming_strips.rs`, `benchmarks/v2_append_block_2026-07-26.md`,
`benchmarks/fold_extraction_2026-07-24.md`, archmage/magetypes/
safe_unaligned_simd 0.9.28/0.2.5 sources + doc-site examples.

## 1. What the study established (parity-load-bearing facts)

1. **The v2 walk is already kernel-strip-tiled.** Every per-pair plane the
   dense/gradient/append/fold kernels read (`mu1/mu2/ssq/s12/activity`) is
   computed per 128-row kernel strip (`STRIP_ROWS`) from a **wide buffer** of
   `strip_h + 2*HALO_P` (=148) rows gathered by `gather_strip_halo` — real rows,
   `reflect_101` only at the TRUE plane top/bottom. The blur pass
   (`fused_blur_h_ssim` + `box_blur_v_from_copy` + activity chain) runs **wholly
   inside the wide buffer**. `box_blur_v_from_copy`'s f32 running-sum state
   initializes at the wide buffer's top — so per-kernel-strip values are a pure
   function of the wide-buffer contents. **Reproduce the wide-buffer contents ⇒
   reproduce every one of these planes bit-for-bit.** This is the exact
   mechanism the plan's G-PARITY sentence names.
2. **The moments cache is a replay, not a source of truth.**
   `compute_ref_moments_channel` / `compute_ref_activity_into` fill the cache by
   replaying the identical strip walk. Deleting the cache deletes duplicated
   machinery, not canonical semantics. Cross activities (`act_x`/`act_b`) are
   likewise strip-replayed — per-kernel-strip recomputation in the streamed walk
   is bit-identical by construction.
3. **Exactly two computations in the 924 path are NOT strip-tiled today:**
   - `bs2 = blur(src²)` (`compute_ref_s2blur_into`): whole-plane
     `box_blur_1pass_into`. Its V pass carries one f32 running sum from plane
     row 0 to the bottom — a streamed walk cannot reproduce that accumulation
     history without carrying whole-plane V-state across strips.
   - `blockiness_sparse` pass 1: **column-outer** iteration (for each lattice
     column, all rows) into a single running f64 — a row-ordered walk cannot
     reproduce that f64 association.
4. **Pyramid + XYB are trivially streamable.** `downscale_2x` is a stateless
   2×2 box with floor dims (odd trailing row/col dropped; in-place vs
   `downscale_2x_into` produce identical values — already relied on by
   `build_v2_ref_scales`'s doc). XYB conversion is per-pixel over row chunks.
5. **Accumulators are the merge surface.** `DenseAccum` / `GradientAccum` /
   `AppendAccum` / `V1BasicSums` are f64 sums (+`WeightedSum` pairs)
   accumulated per kernel strip **in strip order** by
   `compute_channel_scale_v2_with_fold`; `finish_channel_scale` /
   `finish_append` / `V1BasicSums::finalize_into` are pure f64 finalizes of
   (accums, n). f64 addition is order-sensitive, so parity requires preserving
   the per-(scale,channel) kernel-strip accumulation ORDER — not building
   per-outer-strip partials and merging (that reassociates).
6. **`fold_v1_basic_bands` needs no changes.** It consumes wide-buffer-local
   H-planes + src/dst with global coordinates (`y0`, `HALO_P`, plane height)
   and replays v1's 32-row bands internally. Identical wide buffer ⇒ identical
   `V1BasicSums`. The v1-basic parity class vs the frozen v1 path is untouched.
7. **Scale walk:** always `NUM_SCALES=4`; scale dims halve by floor; sub-64
   inputs reflect-pad to 64 BEFORE the pyramid on both sides (same helper both
   sides). `edge_width` is the only cross-scale value: it needs per-(scale,ch)
   `(Σgrad_src, Σgrad_dst, n)` — global sums, order-preserved, so it finalizes
   after the walk with bit-identical math.

## 2. Two pre-chunks: make the materialized path streamable FIRST

To keep G-PARITY a strict bitwise statement ("streamed == materialized, all
924"), the two non-streamable orderings change canonically on the CURRENT
path before any streaming code lands. Both changes hit every entry path
identically, so all existing path-vs-path parity tests
(`append_ref_paths_bit_identical`, `folded720_ref_paths_bit_identical`,
`append_first720_bit_stable`, …) stay green — none pins absolute v2/append
values.

- **P1 — `blockiness_sparse` goes row-ordered.** New canonical form: two
  accumulators, `sum_v` (vertical-step terms, now iterated y-outer over all
  rows × lattice columns) and `sum_h` (horizontal-step terms, y-outer over
  lattice rows × all columns — already row-ordered today), combined
  `sum_v + sum_h` at the end. Value shift: f64 reassociation only (~1e-16
  rel) on BLOCKINESS (1 of 29 v2 slots; 12 of 720 values). Documented in the
  commit + this file; the strip-fed variant then reproduces the new order
  exactly (running `sum_v`/`sum_h` across kernel strips, row-ordered).
- **P2 — `bs2` becomes kernel-strip-tiled.** `compute_ref_s2blur_into`
  (cache fill AND pair replay) re-tiled to the walk's own geometry:
  per 128-row kernel strip, square + `box_blur_h` + `box_blur_v_from_copy`
  inside the same `gather_strip_halo(HALO_P)` wide buffer every other plane
  uses (V window ±5 of strip rows reads H(sq) rows [ks0−5, ks1+5) ⊂ wide
  buffer — HALO_P=10 suffices). Value shift: f32 ULPs on `bs2` ⇒ ULP-scale
  shifts on the 5 σ-split append lanes (MSCN_DIFF_MEAN/L2, CONTRAST_GAIN/
  LOSS, TEXTURE_DISSIM; ≤60 of 204 append slots). Pre-sanctioned by
  `benchmarks/v2_append_block_2026-07-26.md` ("Strip-tile the bs2
  fill/replay … do it BEFORE any corpus freezes append values"); the append
  block has no trained consumer. Bonus: (a) kills the two 48 MB whole-plane
  temps on the replay path (835.7 → ~740 MB @12 MP per the bench doc's
  estimate), (b) `bs2` and `ssq` get MATCHED accumulation tiling, so the
  kernel's `var₂ = (ssq − bs2) − mu2²` subtracts same-rounding-class
  operands — strictly better conditioning than today's mixed tiling.

After P1+P2, **every** input to every kernel in the 924 walk is a pure
function of kernel-strip wide-buffer contents + global row coordinates, and
every pooled value is a ks-ordered f64 accumulation ⇒ the streamed walk can
be bit-exact by construction.

## 3. Streamed architecture (C1/C2): per-scale-cursor producer, one pass

One walk object per pair (`source`, `distorted` both stream — no prepared
reference, no cache):

- **Rolling planes:** 2 sides × 3 channels × 4 scales of row-windowed plane
  buffers (`Vec<f32>` + `lo_row`/`hi_row`, linear rows, memmove-compacted;
  contiguous slices for kernels — no ring indexing). Scale 0 fills from
  `convert_source_to_xyb_into`-style per-row-chunk conversion of a
  `SubsetView`; scale s+1 fills by `downscale_2x_into` of scale-s row pairs.
- **Per-scale kernel-strip cursors:** scale s processes its next 128-row
  kernel strip as soon as rows `[ks0 − HALO_P, ks1 + HALO_P)` (clamped to the
  true plane, where `gather_strip_halo`'s reflect handles the rest) are
  present in the rolling window. This decouples production chunk size from
  kernel-strip alignment — no outer-strip-must-be-multiple-of-1024 constraint,
  and retention per scale is O(148 + slack) rows at that scale's width.
- **Advance loop:** produce a chunk of scale-0 rows (both sides, 3 channels)
  → cascade downscales → for each scale in order 0..4, process every ready
  kernel strip → raise `lo_row`s to `min(next ks window start, next downscale
  input row)`. Kernel strips of a given (scale, ch) therefore execute in
  EXACTLY today's order (ks 0,1,2,… per scale), interleaved across scales —
  the interleave is irrelevant to parity because accumulators are
  per-(scale,ch).
- **Per kernel strip, two phases** (both 3-channel-parallel under `threads`,
  preserving `parallel_matches_serial_exactly`):
  - *Phase A (blur):* per channel, run today's blur pass over the wide
    window. Interior kernel strips read the rolling planes IN PLACE (the
    wide window is a contiguous slice — `run_blur_pass_inner` already takes
    explicit src/dst slices); only true-edge strips gather into
    `src_wide`/`dst_wide` for reflection. Also fills per-ks `bs2` (P2 form)
    and each channel's activity.
  - *Phase B (kernels):* per channel, `dense_block_kernel` +
    `gradient_block_kernel` + `append_block_kernel` + `fold_v1_basic_bands`,
    accumulating into per-(scale,ch) global accums. Y's append reads X/B
    activity from THEIR phase-A scratches (read-only) — this also removes
    today's pair-path double-computation of the X/B activity chains
    (computed once per ks instead of once in the channel walk + once in the
    whole-plane replay provisioning). `(B, scale 0)` skip unchanged.
  - Blockiness (P1 form): per-ks row-ordered lattice sampling from the
    rolling src/dst planes into running per-(scale,ch) `sum_v`/`sum_h`.
- **Finalize (after the last strip):** per (scale,ch):
  `finish_channel_scale`, `finish_append`, `V1BasicSums::finalize_into`, then
  the edge-width cross-scale chain exactly as today (same f64 ops on
  identical grads). No plane survives the walk.

### Geometry / alignment invariants (checked by C1's unit gate)

- Kernel strips are 128-row tiles of the FULL plane at each scale, rooted at
  row 0 — identical to today (`y0 = 0, 128, …`; last strip truncated at
  `h_s`).
- The wide window for ks k at scale s is rows
  `[max(0, 128k − 10), min(h_s, 128(k+1) + 10))` of the scale-s plane, plus
  `reflect_101` rows at true edges — `gather_strip_halo` semantics verbatim.
- Downscale needs input rows `[2j, 2j+2)` for output row j — rolling windows
  advance in even steps at every scale by construction (production chunk =
  multiples of 16 scale-0 rows keeps all four scales' parities aligned;
  exact chunk size chosen in C1, correctness does not depend on it).
- Gradient kernel needs ±1 row (⊂ HALO_P window). Fold bands need ±5
  (32-row-aligned bands ⊂ 128-row strips, `HALO_P ≥ 5`). Append kernel
  reads only strip rows of `ref_y`(=src Y plane), `bs2`, cross activities.

### C1 unit gate (producer parity)

For fixture images (including odd dims, h<128, h≈one-strip, tall-thin):
materialize today's pyramid (`build_v2_ref_scales` + dst walk), and assert
the producer's per-ks wide windows are **byte-equal** to
`gather_strip_halo` output from the materialized planes, at every scale ×
channel × side × kernel strip.

## 4. Memory budget (G-RAM ≤ 250 MB @ 12 MP, heaptrack)

@ 4000×3000 (12 MP): rolling planes ≈ 6 × Σ_s R_s·(w/2^s)·4 B with
R_s ≈ 148 + slack ≈ 192 rows → ~35 MB. `V2Scratch` strips (3 ×
13 × 148·w·4 B) ≈ 92 MB (unchanged; interior in-place windows may drop the
2 wide buffers → −14 MB). Decoded inputs (driver-owned) ≈ 72 MB. Total
≈ **200 MB** — inside the 250 MB gate with margin (v1stream reference:
153.9 MB). At 80 MP (8000 wide), rolling planes ≈ 70 MB + scratch ≈ 185 MB
— O(width), record-only per the plan. The 1.03 GB (cached) / 835.7 MB
(replay) materialized-path numbers disappear with the machinery.

## 5. CPU budget (G-CPU ≤ 62 ms/pair aic3-100 1T; target ≤ 59.5)

Component estimate against today's measurements: streamed pair ≈ fold-720
pair path 54.2 (includes per-pair ref XYB + pyramid + ref-side blur chains,
all of which the streamed walk performs strip-locally) + append marginal
(kernel ≈ 2-3, per-ks bs2 ≈ 2-3, luma/cross ≈ 1; minus the X/B activity
double-computation the pair path pays today) ≈ **58-61 expected**. Risks and
levers if over: (i) in-place interior windows (skip 2×148-row memcpy per
ks×ch×scale); (ii) convert/downscale rolling so margin rows are produced
once (structural in the cursor design — overlap waste is zero, unlike v1's
independent-strip 2× conversion overhead); (iii) `fused_blur_h_ssim` variant
emitting sq_h is explicitly NOT attempted first (§A.14/§A.16 register-
pressure lessons — do not widen tuned kernels casually). Honest-stop
threshold per the plan: miss by >15% (>71.3) with measurements committed.

## 6. What gets deleted at C5 (G-SIMPLER inventory)

- `V2PreparedReference.moments` (`V2RefMoments` mu1/activity/bs2 planes),
  `fill_ref_moments`, `compute_ref_moments_channel`,
  `compute_ref_s2blur_into` (whole-plane form replaced in P2, deleted with
  the cache), `has_cached_moments`.
- `run_blur_pass_strip_cached_ref` + `run_blur_pass_strip_cached_ref_fold`
  and every `moments_for`/`have_s2_cache` branch in the walk.
- `V2Scratch::{append_act_x, append_act_b, append_bs2, append_tmp_a,
  append_tmp_b, append_sized_for}` + `ensure_append` (the pair-path replay
  planes) and `compute_ref_activity_into`.
- `prepare_v2_reference_with_moments`, `prepare_v2_reference_with_moments_
  append` (public, `feature-regime-v2`-gated; in-repo consumers only:
  `v2_ab_extract`, `v2_speed_baseline` — updated same commit).
- The materialized folded/append walk path itself
  (`compute_folded720_[append_]with_ref_impl` re-pointed to streaming; the
  scale-outer materialized body remains only for the plain-v2
  (`V2Bounded`) research path, which is out of this plan's scope).
- Driver: `ZENSIM_AB_MOMENTS` handling for fold/foldapp modes; foldapp
  becomes a cache-free pair call.
- Equivalent gates recreated for the streaming entries: path-parity
  (pair vs scratch-reuse vs parallel), sub-64, identity-zeros, bounds,
  (B,0) skip, first-720-bit-stability vs plain fold.

Net: the entire "reference cache" concept exits the folded/append path;
`V2PreparedReference` keeps only `scales` for the plain-v2 path.

## 7. Chunk mapping (updated from the plan's C1-C6)

- **C1a = P1** (blockiness canonical order), **C1b = P2** (bs2 strip-tiled)
  — small, individually-committed, full suite green.
- **C1** producer + unit gate (§3). **C2** walk port + full-924 bitwise
  parity tests vs the materialized path (fixtures + real aic3 pairs).
- **C3** blockiness strip-fed hardening (lattice alignment across ks
  boundaries, odd sizes) — the variant itself lands with C2.
- **C4** gates: G-PARITY sweep, G-CPU aic3-100, G-RAM heaptrack 12 MP +
  80 MP synthetic (`make_pair` pattern from `tests/streaming_strips.rs`).
- **C5** switchover + deletions (§6) + CHANGELOG. **C6** workspace cleanup +
  final bench doc + honest residuals.

## 8. SIMD discipline notes (from the crate/doc study)

No new SIMD kernels are required: the port composes existing tuned kernels
(`fused_blur_h_ssim`, `box_blur_h`, `box_blur_v_from_copy`,
`downscale_2x_into`, dense/gradient/append kernels) — all already
`#[arcane]`+`incant!`-dispatched with per-tier bodies. Any new hot loop that
does emerge follows the house pattern: `#[magetypes(...)]` generic body over
`GenericF32x8<Token>` (or `F32x8Backend` generics like the existing
kernels), `#[arcane]` entry + `incant!` dispatch, scalar tail mirroring the
SIMD formula, `safe_unaligned_simd`-style reference-based loads (via
magetypes `partition_slice`/`from_array`), benchmarked WITHOUT
`-C target-cpu=native`. Register-pressure lessons §A.14/§A.16 stand: no
widening of the dense/append kernels.
