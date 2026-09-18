# What zensim can show today — September 18, 2026

**User direction (Sept 18):** Squintly is deferred. Get zensim to something
(a) useful enough and (b) offering an opt-in feature set that is a good research
foundation; find what is worth talking about and build demos/benchmarks for it.
This does not change the [production plan](PRODUCTION_PRIORITIES_2026-09-15.md),
its gates, or any frozen artifact. It selects what to *show* from measured
evidence, and lists the gaps that block showing more.

Sources: the production plan, the
[recovery result](../benchmarks/recovery_completion_2026-09-15.md), the
Sept 7–15 Codex session transcript, and three read-only inventories (public API,
measured evidence, existing demos) taken at `1dea2044`. Every number below
already exists in a dated record; none is new.

## Two tiers

**(a) Default tier — a fast scalar metric you can drop in.** What a user gets
from `Zensim::new(ZensimProfile::codec_target())` with default features:
score, cached reference, strided/u16/f32/P3/BT.2020 input, diffmap, streaming
strips, cancellation, wasm, plus `zensim-regress` (publishable, builds from a
bare clone). crates.io is still 0.2.7 (`PreviewV0_2`); B/BHdr/C/CHdr/D are
unreleased.

**(b) Research tier — opt-in.** `BakeScorer` (runtime ZNPR bakes, ensembles,
splines — default features), `custom-profiles` (attribution, `prepare_steering`,
`prepare_steering_hdr`, `refinement_gain`), `corruption-head`, `training`,
the doc-hidden `research` extraction engine, Rev2/Rev3 arithmetic, and the
unpublished `zensim-target` / `zensim-validate` / `zensim-bench` tools.

## Worth talking about (measured, with source)

| Claim | Evidence | Source |
|---|---|---|
| Faster than fast-ssim2, single thread, one process, interleaved | 1024²: B 1.62×, buf228 3.05×, D 4.1×, coarse60 8.2× | `benchmarks/crates_io_speed_bar_2026-09-10.json`, `benchmarks/runtime_profiles_2026-09-13.md` |
| Rev3 ensembles: 9.8 ms (fast) / 18.8 ms (rich) per 1024² scalar p95 | 30 interleaved rounds, pinned core | recovery result |
| CID22 rank tie with SSIMULACRA2 | 0.8927 vs 0.8894, paired bootstrap over 49 references | `benchmarks/ssim2_replacement_bar_2026-08-31.md` |
| CSIQ strictly better than ssim2 | +0.047, CI [+0.038, +0.056] | same |
| Fewer near-lossless dial inversions than ssim2 | 6% vs 14% of q≥85 ladders | same |
| Corruption gate beats every peer on one held-out corpus | 100% vs butteraugli-max 72.5%, ssim2 31.1%, dssim 23.0% (July bake, one source family) | `benchmarks/corruption_gate_fair_comparison_2026-07-17.md` |
| Targeting beats a fixed-q baseline at equal encodes | JXL 45/50 vs 7/50 within ±1 at three shots | `benchmarks/target_steering_bounds_2026-09-08.md` |
| Identity is exactly 100; nothing scores above identity; signed tails | D-id100 contract 6/6, 0/4,424 cells above identity | `benchmarks/d_id100_2026-09-04.md` |
| D lineage is the only one clearing all five codec floors | 4 of 450 board cells + ssim2 itself | `benchmarks/board_ladder_ruler_2026-09-06.md` |
| Scalar memory 19–54 B/px | `/usr/bin/time -v`, scalar path only | `benchmarks/stable_ssim_kernel_2026-09-08.md` |
| Native 16-bit / linear P3 / PQ16 / HLG input with parity audits | 214 SDR + 495 HDR pairs | recovery result |
| Pure-Rust train → bake → serve, byte-reproducible bakes | A_plain reproduction 0.889 in ~190 s/seed | `benchmarks/cleanup_training_reproduction_2026-09-07.md` |

## Do not claim

README speed table ("18× at 4K" measures ~9×, "4× single-thread" ~2.3×; wrong
CPU named). KADID/TID anything. KonJND as a win. Rev3 near-lossless ordering
(B8–B9 regresses vs B). Three-shot targeting tails. JXL spatial byte savings
for Rev3 (negative). AVIF targeting/steering (unmeasured). Any HDR result for
current weights (UPIQ .696/.704 vs BHdr .753). Incremental memory. The
integrity head as qualified (31% activation on ambiguous cases, one known
miss). Anything as "shipped".

## Gaps that block showing more

1. **No Rev3-vs-peer speed head-to-head.** The harness exists
   (`zensim-bench/benches/ssim2_speed_bar.rs`, `ZEN_S2_ENSEMBLE`); it has never
   been pointed at the Sept 15 frozen ensembles.
2. **No speed × accuracy picture.** Results span four model generations;
   nothing puts PreviewV0_2 / B / D / C / Rev3 fast / Rev3 rich and the peers on
   one latency-vs-CID22 chart.
3. **Nobody can look at a diffmap.** No colored heatmap renderer, no CLI that
   shows where two images differ.
4. **The only public page (`site/`) is three months stale** (V0_x bakes).
5. **Research-tier warts:** arithmetic revision is a process-wide env var
   (`ZENSIM_FORMULA_REV`) instead of a property the bake selects; no Rev3 or
   corruption-head bake ships in the package; steering refuses feature IDs ≥ 228;
   five source files in an MIT/Apache crate carry AGPL headers
   (`score_math.rs`, `feature_layout.rs`, `serving.rs`, `research.rs`,
   `fold_timing.rs`) — a licensing question for the user, not an agent.
6. **README/lib.rs drift:** speed table, A-era accuracy table, HDR guidance that
   predates in-crate PQ/HLG decode, missing feature flags, no mention of
   steering or `BakeScorer` beyond one paragraph.

## Demo and benchmark set (existing owners only)

| # | Deliverable | Owner extended | Status |
|---|---|---|---|
| 1 | Diffmap heatmap example + served gallery | `zensim/examples/diffmap_heatmap.rs`, `scripts/demos/diffmap_gallery.py`, `just demo-diffmap` | done (`d58f15a9`); JPEG only, Profile B, TRAIN-role imazen-26 sources; served under `zensim/demos/diffmap-heatmap-2026-09-18/` |
| 2 | Speed matrix: PreviewV0_2, B, D, C, Rev3 fast, Rev3 rich vs fast-ssim2 / ssimulacra2-rs / butteraugli; 64² / 256² / 1024² / 2048² / 4096²; 1T and MT; α + β·pixels fit | `ssim2_speed_bar` (zenbench) | next |
| 3 | Speed × accuracy Pareto page from #2 + existing fulleval CID22 rows (with bootstrap CIs) | `gauntlet.py` data, one new static page | after #2 |
| 4 | Targeting demo: `zensim-target` vs fixed-q on TRAIN images, per codec | `demo_matrix` | after #3 |
| 5 | README correction batch | — | needs user approval |

## Findings from building the demos

* **`compute_with_diffmap` scores an identical pair 96.20, `compute` scores it
  100.** Verified Sept 18 on a 900×675 PNG against itself, Profile B. On a
  distorted pair the two paths agree to full precision; only `compute` has the
  identity short-circuit. Registered in `CLAUDE.md` Known Bugs.
* **`DiffmapResult` documents typical photo values as `[0, 0.3]`; measured
  maximum over 18 JPEG q20/50/80 cells is 0.084** with default `Trained`
  weighting. The demo uses a measured shared ceiling of 0.06. The doc comment
  is unchanged pending the README/doc correction batch.
* One absolute heat scale cannot serve photos and flat graphics together:
  photo q20 texture clips at 0.06 while clipart p99 is 0.009.
