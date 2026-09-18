# Cross-generation speed matrix — 2026-09-18

Every zensim generation and three peer metrics, interleaved in one process per
arithmetic revision, across five geometries, at one thread and at sixteen.

This is a SPEED measurement and nothing else. It says nothing about which
metric is right, which profile ranks better, or which one anyone should ship.
A profile that is fast and wrong is still wrong; the ranking evidence lives in
`bake_verdict` panels, not here.

## How to reproduce

```
just bench-speed-matrix
```

which builds both fast-ssim2 feature configurations through `run-heavy`, runs
`scripts/demos/speed_matrix_run.sh`, and renders this file's tables with
`scripts/demos/speed_matrix_report.py`. The raw zenbench rounds — the actual
evidence — are written to the `raw` directory named in the recipe; this report
is a view of them and can be regenerated from them at any time.

The prose you are reading is `benchmarks/speed_matrix_2026-09-18.notes.md`.
The tables below it are generated. Editing the generated `.md` by hand is
pointless: the next run erases it.

## Two processes, and why

`ssim_form::active_revision` is a process-global `OnceLock`. Setting
`ZENSIM_FORMULA_REV=3` changes the pixel kernels for everything in the
process, so a single-process matrix cannot hold both the revision-3 frozen
ensembles and the revision-1 named profiles honestly. Measured, not assumed:
under `ZENSIM_FORMULA_REV=3` zensim prints

> ZENSIM_FORMULA_REV pins Rev3 pixels, but a built-in profile's bake is
> revision-1 coefficients. Its FEATURES are the pinned revision; its SCORE
> prices them with weights fit against another extractor and is not a served
> score.

and the timing moves with the arithmetic — `zensim_B` at 64&sup2; ran 177.8 µs
at revision 1 and 165.8 µs at revision 3 in the smoke runs. So:

- **`1t-rev1` / `mt16-rev1`** — the named profiles (`PreviewV0_2`, `B`, `C`,
  `D`) and the three peers.
- **`1t-rev3` / `mt16-rev3`** — the two frozen revision-3 ensembles, plus the
  anchor.
- **`fast_ssim2` runs in both.** It is revision-independent, so it is the
  cross-process bridge. If its medians disagree between the two processes at
  the same size and thread count, the box moved between them and the two
  halves are not comparable. Check that before reading any cross-process
  comparison. Within a process, every number is interleaved and therefore
  paired; across processes, nothing is.

One trap worth recording, because it bit this run: `BakeScorer`'s narrow-plan
fast path does **not** refuse a revision mismatch. The revision-3 ensembles
load and score happily inside a revision-1 process, silently, at revision-1
pixels. The first attempt at this sweep did exactly that and was discarded.
The driver now names its arms explicitly per process instead of relying on a
refusal that does not come.

## What is inside each timed region

Every arm starts from the same `[u8; 3]` sRGB pixel pair, produced by
`test_pair(n, n)` — byte-identical to the generator
`zensim/benches/extract_paths_bench.rs` and `fold_pools_bench` use, so this
instrument and those feed their kernels the same pixels.

| arm | inside the timed region | outside |
|---|---|---|
| `fast_ssim2` | `compute_ssimulacra2(Img<&[u8;3]>, …)` — every conversion | nothing |
| `butteraugli` | `butteraugli(Img<&[RGB8]>, …, &Params::default())` — every conversion | the zero-cost `bytemuck::cast_slice` view |
| `ssimulacra2_rs` | `Vec` clone (its by-value API forces one), sRGB→linear→XYB, the metric | the u8→f32 sRGB widening |
| `zensim_*` (named profiles) | `Zensim::new(profile).compute(&RgbSlice, &RgbSlice)` — router, extraction, forward, output spline | `Zensim::new` itself, hoisted once |
| `rev3_*_ens5` | `BakeScorer::ensemble(..).compute(..)` — one extraction serving all five members, all five forwards, calibration, composition | model parsing, hoisted once |

`ssimulacra2_rs` is the one arm whose contract differs, and the difference cuts
both ways: it is *not* charged for the u8→f32 widening the other arms' inputs
never need, and it *is* charged for a `Vec` clone its API forces (2 × 48 MiB of
memcpy at 4096&sup2;). `zensim-bench/benches/bench_compare.rs` draws the line in
exactly the same place; this arm is a copy of that decision, not a second
opinion about it. Read its 4096&sup2; column with that in mind.

`zensim_V0_2` is the **in-tree** implementation of the profile the published
0.2.x line defaults to. It is not the published 0.2.7 binary. That opponent
lives in `zensim-bench/benches/crates_io_speed_bar.rs` behind the
`crates-io-0-2-7` feature, which pulls the real crate; the two are not
interchangeable.

## Method

- zenbench, randomized round-robin interleaving. Arms inside a group share the
  box's thermal and neighbour state rather than accumulating it onto whichever
  ran second, which is the bias criterion's isolated back-to-back runs bake in.
- `ZEN_S2_SINGLE_CALL=1`: one call per round, so the raw rounds are genuine
  per-call latencies and p95 means something. A batched mean is not a
  percentile, and the report refuses to emit one where the rounds were batched.
- 32 rounds per arm per size (the standing bar is ≥30).
- Release build, `lto = "thin"`, `codegen-units = 1`, **no**
  `-C target-cpu=native` — runtime SIMD dispatch is what users get.
- Scoring thread rows: 1T on `taskset -c 2`, 4T on `0-3`, 8T on `0-7` — all
  within CCD0, the 96 MiB-L3 die, so the 1→4→8 scaling column is one cache
  regime throughout. 16T is `0-15` and necessarily spans **both** dies: that
  factor carries a cache change as well as more cores, and is not comparable to
  the other three.
- 4T and 8T drop `butteraugli` and `ssimulacra2_rs`, to keep the run's wall
  time down — they are the two slowest arms by a wide margin and each extra
  column costs more than all five zensim arms combined. **That is the only
  reason.** An earlier draft of this file justified the omission by asserting
  neither has a thread pool; the 16T column in the tables below falsifies that
  outright (butteraugli 2.19× at 4096&sup2;, ssimulacra2_rs 1.44×), so the
  claim is struck rather than quietly dropped. Both are present at 1T and 16T;
  their 4T and 8T cells are blank because they were not measured, and nothing
  is interpolated into them.
- Every threaded row uses the `ssim2-rayon` build so fast-ssim2's Gaussian blur
  is threaded too; an MT row measured against a single-threaded opponent is not
  an MT comparison. The 1T row uses the plain build. For the zensim arms that
  build difference is irrelevant — the feature only touches fast-ssim2 — so
  their scaling factors are clean. For `fast_ssim2` itself the 1T→NT factor
  compares its non-threaded build against its threaded one, which is exactly
  the question "does turning its threading on help?" Measured answer at
  4096&sup2;: no. 1363 / 1341 / 1339 / 1391 ms at 1 / 4 / 8 / 16 threads.

## The anchor row doubles as a per-column validity check

Every thread configuration is its own process, so the `xNT x1T` scaling
columns are cross-process comparisons like the revision ones. They get the same
guard, for free: because `fast_ssim2`'s threading buys nothing (above), its
`x1T` entries **should all read about 1.00×**. Where they do, that column's
process saw the same box as the 1-thread process and its scaling factors can be
trusted. Where they do not, it did not.

One column fails that check here: revision 3, 8 threads, where the anchor reads
0.81× (1724 ms against 1389 ms at 1T) and 27 of 32 rounds are flagged noisy.
That run was contended. Its two ensemble arms' scaling factors are inflated and
should not be quoted.
- `nice -n19 ionice -c3`, not `run-heavy`: its cgroup scope makes zenbench's
  own resource gating refuse the run, which is why `just rev3-cost` says the
  same thing.

## Frozen bakes

The two revision-3 ensembles are the September 15 frozen recovery bakes, five
equal-weighted (0.2) members each. The driver verifies every member's sha256
against that directory's own `FROZEN.json` and refuses to benchmark bytes that
do not match.

`rev3_fast_y60_ens5` — `R915_y60_h32_s{17101,17103,17107,17111,17113}.bin`:

```
4545e49e378d641218683782e8af575fd3d074580b49714a7b90193ebe8c5ce5
2a2693ca2083bb4a88eb729ae0451fa9a130ab8f0317db5da585f9037ca7a573
051d44a22afb947d6b5ac5114589310f46d5ae0098e32c19901be848d3345c01
cf76d47ce45ab9f03e8a8f99d7d1abaa13685d10cbf16cf27ad204b3beae9a28
1e04d875b9ed21302cbe6b396dbfc134eb0e1bc524c31ff44d9eb8c1034631ed
```

`rev3_rich_basic228_ens5` — `R915_basic228_h128_s{17101,17103,17107,17111,17113}.bin`:

```
17d17b20f78ad9dc7b2c90a1c991488581c3c23f0130a2422bd93194c4fac099
b1a62c7bc97d789bf1f3175418bba976ea004623ea1778f85c2e85f8d9e250e0
9c48aaddce98a02847d6514db263c35981bb4b7ba7918a719769902ed9152815
4e6bb14c981b0999351fecd2f6a0f6ba6b7e0f92d6f7e615ad311c8034020a5f
5f5fc2b25d85d125f3cd625807e3760c47d11c0460995b9ab1a705e02ad7d09b
```

## Reading the fit — and why it fails

Each arm gets an ordinary least-squares `time = α + β·pixels` over the
geometries it was measured at, with α in milliseconds (fixed per-call
overhead) and β in milliseconds per megapixel (per-pixel work). Both are
reported because neither alone is meaningful: at 64&sup2; α dominates and at
4096&sup2; β does, so a "ms/MP" number with no intercept is miscalibrated at
whichever end matters.

**The honest result is that no arm is linear over 64&sup2;–4096&sup2;.** Every
global fit is marked `NO`, and several intercepts come out *negative* — which
is physically impossible as a fixed per-call cost and is the tell. The cause
is not noise: per-pixel cost **rises with size** as the working set leaves
cache, so a straight line through a 4096× pixel range is dragged steep by the
largest point and undershoots everything below it.

The linearity verdict is judged on the **relative** residual, not the absolute
one. An absolute test passes trivially once the largest size is seconds and the
smallest is microseconds — that test initially reported every arm as cleanly
linear *including the ones with negative intercepts*, which is exactly the kind
of tidy-looking wrong number this report exists not to publish.

So read the **marginal cost table** instead. Each cell is
`(t_i − t_{i−1}) / (MP_i − MP_{i−1})` — a measured difference between adjacent
sizes, no fit, no extrapolation. Reading across a row shows where per-pixel
cost grows, which is the thing the single fit was hiding.

## Cross-process validity

The revision-1 and revision-3 halves are separate processes, so nothing bridges
them except `fast_ssim2`. The report computes that anchor's drift per size and
**refuses** the cross-process comparison where it exceeds 5%, rather than
printing a caveat under a number nobody will re-read.

Measured here: drift is 0.3–3.6% at 1024&sup2; and above, and ±8–21% at
64&sup2; and 256&sup2;. Cross-process reading is therefore sound at 1024&sup2;,
2048&sup2; and 4096&sup2;, and **not** sound at the two small sizes — at those
geometries a single call is hundreds of microseconds and process-level
scheduling noise swamps the signal. Within a process every arm is interleaved
and therefore paired, at every size.

## Feature extraction

`zensim/benches/extract_paths_bench.rs` is the existing owner of the
extraction question and `scripts/demos/speed_matrix_extract.sh` only drives it.
The arm set is whatever that bench defines: `buf_v1_228`, `buf_v1_372`,
`fold156_basic`, `fold228_peaks`, `fold228_moments`, `fold228_classc`,
`fold372_full`, `fold944_off`, `fold944_full`, and the `fast_ssim2` anchor.

**There is no arm for the y60 / coarse-Y plan** the revision-3 fast ensemble
reads. That regime is planned from the bake's declared feature IDs at serve
time; it is not one of this bench's enumerated walks, and none of the `fold*`
arms is it. Rather than pick the nearest-looking one and mislabel it, the
matrix leaves the cell empty. The closest available measurement of that walk is
the end-to-end `rev3_fast_y60_ens5` arm in the scoring half, which prices the
extraction *and* five forwards together.

Extraction pinning follows `benchmarks/k4_st_mt_2026-09-10.md` so the numbers
are comparable to that record: 1T = cpu 8, 4T = cpus 8-11, 8T = cpus 8-15 (all
one CCD), 16T = cpus 0-15 (both). Note this is a **different die** from the
scoring half's 1T core (cpu 2, on CCD0): the two halves' absolute 1T numbers
are on different L3s and should not be differenced against each other.

Both revisions are run. The September 10 record was revision 1 only, and
revision 3 took fused-kernel work after it.

## What this run actually found

Read the tables, not this list — but these are the results that surprised the
people who asked for the matrix.

**`fast_ssim2`'s threading buys nothing.** With the `ssim2-rayon` feature
compiled in and 4, 8 or 16 threads given to it, 4096&sup2; costs 1363 / 1341 /
1339 / 1391 ms. Flat. Every zensim scaling factor in the matrix is therefore
also, incidentally, a measurement of how much of the gap is threading: at 16
threads `zensim_B` is 0.08× the anchor, against 0.49× at one thread.

**Both other peers do thread, contrary to what this file claimed before the
run.** At 4096&sup2;/16T, butteraugli is 2.19× its own 1T time and
`ssimulacra2_rs` is 1.44×. The earlier assertion that neither had a thread pool
was written from assumption and is struck above.

**zensim is not uniformly ahead.** At 64&sup2; `zensim_B` is *slower* than
`fast_ssim2` (1.15×) and `zensim_C` is 4.3× slower; the fixed per-call cost of
the fold/plan/forward stack dominates when there are almost no pixels to
amortise it over. zensim's advantage is a large-image advantage, and it only
becomes decisive from about 256&sup2; up.

**`PreviewV0_2` — the published 0.2.x profile — is still competitive.** It
beats `B` at every size measured, and at 64&sup2; and 256&sup2; it is the
fastest arm in the matrix outright, ahead of `D`. `D` overtakes it only from
1024&sup2; up (28.4 ms vs 29.6 ms there, widening to 355 vs 421 at
4096&sup2;). Newer is not automatically faster, and the profile most users are
currently running is not the slow one.

**`zensim_C` is the expensive one.** 0.80× the anchor at 4096&sup2;/1T against
`B`'s 0.49× and `D`'s 0.26×, and it scales worst of the four (2.4-2.7×). The
944-wide dense bake is priced accordingly.

**Per-pixel cost is not constant, and the direction differs by implementation.**
Across 64&sup2;→4096&sup2; the marginal cost of `fast_ssim2` climbs 40→85
ms/MP and butteraugli's 60→127, while `zensim_B` (28→41→39→41) and `zensim_D`
(19→28→21→21) flatten or fall back after 1024&sup2;. The streaming fold holds
its per-pixel rate as the working set leaves cache; the peers' pyramid
implementations do not.

**Revision 3 is not free, and not uniform.** Its fused kernels make
`buf_v1_372` 8-21% cheaper (most often 15-20%) and `fold372_full` 2-10%
cheaper, reproducibly, across every thread count and every size ≥1024&sup2;
where the anchor certifies the comparison. `buf_v1_228`, `fold156_basic`,
`fold228_*` and `fold944_off` are unchanged within ±2%. The saving is
specific to the 372-wide paths, not a general speedup.

**The 944 folds still do not scale.** `fold944_full` reaches 2.46× on sixteen
cores; `buf_v1_372` reaches 5.92× on the same box in the same matrix. That
reproduces the September 10 finding on a different day with 32 rounds instead
of 12, and it is still the clearest single target in the extraction path.
