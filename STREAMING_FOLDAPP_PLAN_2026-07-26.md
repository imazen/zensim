# Streaming-only folded+append extraction — plan (2026-07-26)

> **STATUS (2026-07-26): COMPLETE — C0-C6 done.** Chunks C0-C4 landed
> `e421f28a..d9e4f9f9`; G-CPU measured 1.33× vs the 1.15× gate (floor
> ≈1.27× — the honest cache-free price; the gate's v1-derived anchor
> under-estimated what the v2 moments cache amortizes), G-PARITY byte-
> exact on fixtures + 100 real pairs, G-RAM 221 MB @ 12 MP (4.7× under
> the cached path). The user accepted the CPU trade ("Delete now") and
> the C5 switchover shipped: folded/append extraction is STREAMING-ONLY,
> the reference-cache machinery is deleted (net −880 lines), plain-v2
> keeps its own moments cache. Record:
> `benchmarks/streaming_foldapp_gates_2026-07-26.md` (+ C5 addendum),
> design note `docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md`.

**Goal (user directive):** port the v1 streaming-strips architecture to the
folded-720+append (924) walk and make it the ONLY path — eliminating the
materialized-pyramid + reference-cache machinery (V2PreparedReference
moments/bs2, pair-path replay planes) as a maintenance surface.

## Why (measured, this host, aic3-100 1-thread unless noted)

Peak heap at 12 MP: v1 streaming **153.9 MB** vs foldapp 835.7 MB (no cache)
/ 1.03 GB (cached). Full table: `benchmarks/v2_append_block_2026-07-26.md`.

Ref-cache value A/B (compute ms/pair, decode 2.04 s subtracted, medians):

| path | pair / per-strip | cached ref | cache buys |
|---|--:|--:|--:|
| v1 full (372) | 57.2 | 48.6 | −15.1% |
| v1 streaming (score) | 65.4 | 55.0 | −15.9% |
| fold-720 (v2 moments) | 54.2 | 49.55 | −8.6% |
| foldapp (v2 moments) | 83.9* | 54.1 | (*not comparable) |

\* foldapp-without-cache today re-runs whole-plane bs2/activity replays per
pair — a bolt-on, not a native cache-free design. v1's numbers are the
honest guide: **a well-integrated cache-free path costs ≤ ~15% even at
8–10 variants/reference, and streaming+full-ref matched the materialized
pair path (55.0 vs 57.2).** That is the case for deleting the cache.

## Design directives

1. **Streaming-only.** Per-strip XYB conversion + per-strip pyramid
   production feeding the existing v2 strip kernels; per-strip
   accumulator merge (v1's `ScaleAccumulators::merge` precedent). No
   full-image f32 planes anywhere in the folded/append path; no
   V2PreparedReference/moments/bs2 cache; delete the pair-path replay
   planes (`V2Scratch::append_*`).
2. **v1 is the blueprint.** `streaming.rs` (`compute_multiscale_stats_
   streaming_strips[_with_ref]`, STRIP_INNER band tiling, per-strip
   pyramid downscale, byte-exactness by tiling bands against the
   full-image plane layout) + `fused.rs` + `blur.rs`. Study before
   designing.
3. **Every append signal streams.** activity/cross-channel act (per-strip
   chains, all 3 channels within a strip), bs2 (per-strip square+blur —
   its whole reason for being cached disappears when it's computed
   in-pass per strip), ref_y (strip of the raw ref), global sums +
   dev2 (plain accumulators), edge_width (per-scale grad means — scale
   interleave). `blockiness_sparse` needs a strip-fed variant (row-ordered
   8-lattice sampling). (B, scale 0) skip stays.
4. **SIMD discipline:** archmage `#[arcane]`/`#[rite]`/`incant!` +
   `#[magetypes]` generic kernels, `safe_unaligned_simd` for any new
   load/store shapes. Study: `cargo read archmage` / `magetypes` /
   `safe_unaligned_simd` (extracted at `~/.cache/cargo-read/*-0.9.28`,
   `safe_unaligned_simd-0.2.5`), the archmage doc site
   (`~/work/archmage/docs/site/content/{archmage,magetypes}/`), the 7
   production examples (`.../magetypes/examples/`), and this repo's own
   §A.14/§A.16 lessons in `feature_v2.rs` (register pressure, inline
   collapse, POOL_SIMD) — do NOT widen tuned kernels casually.

## Acceptance gates (all measured, committed to benchmarks/)

- **G-PARITY:** streamed folded-720+append output vs the current
  materialized path: byte-exact for the v2+append blocks when the strip
  producer feeds bit-identical strip planes (the current walk is already
  strip-tiled — reproduce `gather_strip_halo` semantics from streamed
  input); v1-basic fold keeps its existing band-replay parity class.
  Full existing suite green throughout (144 lib + integration).
- **G-CPU:** aic3-100 batch (8–10 variants/ref), 1-thread: streamed
  foldapp ≤ 1.15× today's CACHED foldapp (54.1 ms/pair → ≤ ~62); target
  ≤ 1.10×. v1's streaming+ref≈pair result says this is reachable;
  per-strip ref rebuild is the honest price of cache deletion.
- **G-RAM:** heaptrack peak heap at 12 MP ≤ 250 MB (v1stream class);
  large-image test in the `streaming_strips_oom` pattern.
- **G-SIMPLER:** net-negative diff on cache machinery: moments/bs2 cache,
  `ensure_append`, replay planes, `prepare_v2_reference_with_moments_
  append` all deleted (or hard-deprecated) after parity+perf gates pass.

## Chunks (each: commit + push + gates green + next-chunk note)

- **C0 study:** read the three crates + doc site + v1 streaming/fused/blur
  + `feature_v2.rs` walk; commit a short design note (strip geometry
  covering 4 scales + halos for blur/gradient/append; scale-interleave
  plan for edge_width; accumulator merge plan).
- **C1 strip producer:** per-strip XYB + pyramid producer (margins per
  v1's `strip_inner=256/margin=128` math scaled to the v2 walk's
  BLUR_RADIUS/HALO_P needs), feeding strip planes bit-identical to
  `gather_strip_halo` output. Unit-gated against the materialized planes.
- **C2 walk port:** folded+append walk consumes the producer: dense/
  gradient/append kernels + `fold_v1_basic_bands` per strip, per-strip
  accumulator merge, edge_width via per-scale interleave.
- **C3 blockiness:** strip-fed `blockiness_sparse` variant, parity-gated.
- **C4 gates:** G-PARITY/G-CPU/G-RAM measured + committed benchmark doc.
- **C5 switchover:** entries become streaming-only; cache machinery
  deleted; drivers/tests updated; CHANGELOG.
- **C6 cleanup:** workspace forget + rm, final bench doc, honest residual
  list.

## Honest-stop rule

If any chunk reveals a wall (e.g., bit-parity impossible without
reproducing padded-width semantics, or G-CPU misses by >15%), STOP and
ship the investigation + measurements + recommendation as a docs commit —
do not force it.
