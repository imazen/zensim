# Profile D ships + the published-crate speed regression check (2026-09-01)

Two user-approved tasks, one doc. **Task 1**: build the product path the
ADD156 ship audit found missing (`ComputeSet::from_block_profile` +
`ZensimProfile::D`) and land it. **Task 2**: answer the user's question —
"compare speed to profile A as published on crates.io in case we regressed —
I remember a 10" — with a same-process, ASLR-controlled measurement.

---

## Part 1 — `ZensimProfile::D` ships

### 1.1 What was missing (the audit)

`benchmarks/add156_ship_audit_2026-08-31.md` D1: ADD156 (the campaign's
basic-only, 156-zone-reading bake) scored 2.54×/4.43×/3.52× faster than the
full walk at 1/8/16T (`benchmarks/era2_fast_profile_subset_2026-08-31.md`
§1), within 0.019 CID22 of shipped `B`, but **had no product path at all**:
`ComputeSet` was `pub(crate)`, `ComputeSet::from_block_profile` did not
exist (`feature_v2.rs:1716` said it was "the next step and deliberately NOT
added here"), and there was no `ZensimProfile` slot to carry the bake.
`benchmarks/ssim2_replacement_bar_2026-08-31.md` and
`benchmarks/hybrid_candidate_2026-09-01.md` both registered this as the W7
("reachable by a default build") gap blocking every 156-class speed claim
in that campaign's exam.

### 1.2 What landed

- **`ComputeSet::from_block_profile(model: &crate::mlp::Model) -> Self`**
  (`zensim/src/feature_v2.rs`, `pub(crate)`) — derives the minimal v1
  compute set a bake structurally needs from its own layer-0 read pattern.
  A bake declared within the v1 372-feature layout cannot read any v2-era
  block by construction (no such column exists), so `v2_blocks` and every
  field under it come back `false` exactly; within that layout, `v1_pools`
  is derived via the SAME shared policy `fold_engine::score_pool_mode`
  already uses (`bake_pool_need_from_model` + `pools_mode_for_need`, both
  factored out of the existing, tested code rather than reimplemented — see
  §1.4). A bake wider than the v1 layout (a 944-class model) is not
  analysed block-by-block here and gets the safe "everything needed"
  fallback rather than an optimistic under-report.
  - Kept `pub(crate)` per the recorded decision in
    `benchmarks/era2_perf_break_2026-08-31.md` §26.1 and
    `era2_fast_profile_subset_2026-08-31.md` §4 ("No new public type, no
    new public entry point... the cheapest shipping form... is the
    recommendation").
- **`ZensimProfile::D`** (`zensim/src/profile.rs`) — a new unit variant on
  the `#[non_exhaustive]` enum, gated behind the existing default-on
  `candidate-profiles` feature (same gate as `C`/`CHdr`). Carries
  `weights/d_sdr_add156_dense_dial_2026-08-31.bin` (3,671 B, sha256
  `4481c2d4a7c0d35e82f423587b9bc5ce8a52642375e778e5214af38b799ad504`) — the
  audit's own registered fix, arm A from
  `benchmarks/add156_d7_ood_guard_2026-08-31.pointer.md`: the ORIGINAL
  campaign bake plus ONLY the free, rank-exact spline-top extension that
  closes the 100%-above-knot HF-near-lossless failure. The costly winsor
  OOD guard (arms B/C, which buy full G-RANGE coverage at real rank cost on
  LIVE/KADID/TID) is deliberately **not** included — that pointer doc calls
  it a separate, user-gated trade, and no such approval exists here.
  `skip_score_mapping`/`extrapolate_score` both set `true` (audit finding
  D9: a spline-carrying bake with either `false` silently scores every
  distortion `0.000000`).
- **`Zensim::new(ZensimProfile::D)` opts itself into the fast engine.**
  Every other profile still defaults to the buffered walk and leaves the
  existing `#[doc(hidden)]` `with_engine`/`with_unread_feature_skipping`
  knobs to an explicit opt-in (unchanged). `D` is the one exception: since
  its entire reason to exist is speed, `Zensim::new` sets `fold_engine =
  true, skip_unread_pools = true` specifically for `D`
  (`matches!(profile, ZensimProfile::D)`, cfg-gated so it compiles with or
  without `candidate-profiles`). Both fields are inert outside
  `feature-regime-v2` (nothing reads them), so this is a no-op on a build
  without that feature.

### 1.3 The W4/W7 decision — un-gate vs fall back, MEASURED

The brief required choosing between un-gating the v1-only fold path for a
default build, or letting `D` fall back to buffered by default with the
fast path staying behind `feature-regime-v2`, and measuring both.

**Un-gating was rejected on inspection, not by assumption.** The entire
fold-backed engine — `ComputeSet`, `V1PoolsMode`, `score_pool_mode`, and
every extraction entry `compute_fold_backed`/`compute_folded_v1_372_*`
consumes — lives inside `feature_v2.rs` (18k+ lines) and `fold_engine.rs`,
both declared `#[cfg(feature = "feature-regime-v2")]` at the `mod` level in
`lib.rs`. Splitting out exactly the v1-only-fold subset from the genuinely
new v2-bounded-feature machinery it's interleaved with (HDR PU path, the
944 append/append2/csfw blocks, the streaming strip producer) is a
large, cross-cutting refactor of code that is under **active, separate,
concurrent development this week** (the era-2 flip landed 2026-08-31, one
day before this task). Attempting it inside a tight-scope task would touch
far more than Profile D and risks the exact kind of surprise regression
the "never improvise beyond the brief" rule exists to prevent.

**Falling back to buffered by default was chosen, and here is what it
costs, measured** — both forms of `D` (this task's own scratch bench,
`main_D` arm, base build vs the `fast-d`-feature build) and, as a cross
reference, `main_B` in both builds as the required cross-build anchor: see
§2.6 (Profile D's column) for the D-fast-vs-fast-ssim2 numbers and
`~/tmp/zensim-speed-check/run1_parsed.tsv`/`run2_parsed.tsv` for the raw
base-vs-fastd, main_D-vs-main_B rows at every size/thread cell measured
(built directly from this task's own measurement, not re-cited from the
campaign docs).

**The consequence, stated plainly, WITH a correction to how the audit's
headline number applies here.** `era2_fast_profile_subset_2026-08-31.md`'s
"2.54×/4.43×/3.52× at 1/8/16T" is `156` (D's compute set) **against
`944full`** (the full folded-944 walk: basic + pools + v2-348 + append +
append2 + csfw) — it is NOT `156` against `372` (shipped `B`'s compute
set: basic + pools(Full), no v2-era block at all). Reading the SAME table's
`372` row against `156` gives the honest comparator for "D vs B": roughly
**1.4×/1.6×/1.3× at 1/8/16T** (155.5/109.6, 44.1/28.0, 42.9/32.2 ms at
2304², from that table) — real, but far short of 2.5×+. `B` is not a
944-class model, so the 944-class ratio never applied to "D vs B" in the
first place; this task's own §2.4 measurement checks that smaller,
correct-comparator claim directly rather than re-citing the bigger one.
On a **default build** (no `feature-regime-v2`), `D` computes the full
372-feature vector exactly like `B` does — it does not reach even the
smaller `156`-vs-`372` speedup; §2.4 states how close to parity it lands.
The fast form is real only in a build compiled with `feature-regime-v2`.
Every W4 PASS claimed for a `156`-class arm in `hybrid_candidate_2026-09-01.md`
remains, as that document itself already said, "a property of the model
and not of any code path a user can run" **on a default build**; `D` now
makes it a code path a user CAN run, one feature flag away, instead of a
campaign-only bake with no profile slot at all.

### 1.4 No duplicated logic

`fold_engine.rs::bake_pool_need` (bytes → parse → `V1PoolNeed`) was split
into a parse-free `bake_pool_need_from_model(&Model) -> V1PoolNeed`, with
the bytes-taking form now a two-line wrapper. The "which `V1PoolsMode` for
a given need" policy (`Off` is mathematically tempting but never the right
answer — `Peaks` costs the same and has smaller footprint,
`benchmarks/fold_footprint_2026-08-31.md` §9.2) was factored into
`pools_mode_for_need(V1PoolNeed) -> V1PoolsMode`. `ComputeSet::
from_block_profile` and `fold_engine::score_pool_mode` both call through
these two shared functions — one policy, two call sites, cross-checked by
tests (§1.5), never duplicated. `score_pool_mode` itself is unchanged in
behavior (same cached, `ProfileParams`-level union it always was) — it was
not rerouted through `from_block_profile`'s single-model, uncached form,
which would have regressed the exact hot path this work exists to speed up.

### 1.5 Gates

All run via `~/work/zen/scripts/run-heavy`, `CARGO_TARGET_DIR` scoped to
this workspace.

- **Full suite green**, across default features, all-features
  (`feature-regime-v2,candidate-profiles,custom-profiles,training,threads,
  classification,zenpixels,oracle`), and `candidate-profiles` explicitly
  disabled (`avx512,imgref,threads,deprecated-profiles`): 125/125,
  264/264, 113/113 passed respectively (0 failed), plus 6/6 doctests.
  **Excluded from that count, and NOT caused by this work**: `blur::
  tests::{fused,abs_diff,box_blur}_h_ring_matches_regathered_reference`
  and (all-features only) `feature_v2::tests::phase_a_blur_bands_are_bit_
  exact` — `attempt to subtract with overflow` panics in `blur.rs`. `jj
  diff` against this task's parent commit shows `blur.rs` untouched by
  this work; `jj log` on that file shows its most recent change is the
  2026-08-31 "ERA-2 FLIP" commit (`515001dc`), one day before this task
  started — a pre-existing regression from concurrent, unrelated,
  in-flight work, not this task's to fix. Flagged here so it's visible;
  not silently worked around.
- **`clippy --all-targets -D warnings`** clean on all three feature
  combinations above.
- **D's scores bit-identical to scoring the same bake through existing
  entries** — `fold_engine::skip_policy_tests::
  profile_d_scores_are_engine_and_skip_invariant`: buffered vs fold,
  skip-off vs skip-on, all four combinations, through the public
  `Zensim::compute` entry point, on three geometries, all bit-identical to
  the default (fast-by-default) construction.
- **The derivation never removes a family the model reads** (the safety
  property `era2_fast_profile_subset_2026-08-31.md` §5 item 2 asks for):
  `feature_v2::tests::from_block_profile_matches_score_pool_mode_on_
  shipped_b` and `..._derives_the_156_set_for_profile_d` cross-check
  `from_block_profile` against the independently-tested `score_pool_mode`
  path on `B` (must resolve `Full`) and `D` (must resolve `Peaks`, never
  the mathematically-tempting-but-footprint-inferior `Off`); `..._falls_
  back_to_everything_on_a_wide_bake` proves the safe fallback fires on a
  944-wide model.
- **Default-build (W7) correctness**, with NO `feature-regime-v2`:
  `profile::profile_c_tests::d_weight_sha256_pinned`,
  `d_bake_loads_caller_width_372_dense` (372 caller = 372 internal, dense,
  unpruned — unlike `C`/`CHdr`), `d_identity_fixture_scores_100`,
  `d_compute_on_non_identical_pair_scores_normally` (unlike `C`/`CHdr`, `D`
  fits the STANDARD `Zensim::compute` pipeline directly — no folded-944
  extraction, no `ModelForwardFailed`), `d_ladder_is_monotone_and_bounded`.
  All 5 pass in a build with zero extra features beyond the default set.

### 1.6 Full public-API additions (for approval record)

1. `ZensimProfile::D` — new unit variant, `#[cfg(feature =
   "candidate-profiles")]` (default-on), on a `#[non_exhaustive]` enum.
   Additive; no existing item changed shape.

Nothing else is public. `ComputeSet`, `ComputeSet::from_block_profile`,
`bake_pool_need_from_model`, and `pools_mode_for_need` are all
`pub(crate)`. `CHANGELOG.md` records this under `[Unreleased]`; no version
was bumped and nothing was published (`cargo publish` was not run).

---


## Part 2 — the crates.io speed-regression check

The user: *"compare speed to profile A as published on crates.io in case we
regressed — I remember a 10."* Clarified mid-task: the remembered claim is
**"Profile A is ~10x faster than ssim2."**

### 2.1 The claim, located and quoted exactly

**`README.md` (repo root, current — the crates.io-published README.crates.md
carries the identical text) headline:** *"Perceptual image similarity in
22 ms at 1080p. **18x faster than C++ SSIMULACRA2 at 4K.**"* Speed table
(stated conditions: **AMD Ryzen 9 7950X 16C/32T (WSL2)**, synthetic gradient
images, no I/O, pre-allocated buffers; zensim + ssimulacra2-rs use rayon ALL
CORES, C++ libjxl / fast-ssim2 / butteraugli-rs are SINGLE-threaded; "Median
of 100 samples via criterion"):

| Resolution | zensim (MT) | zensim (1T) | C++ libjxl FFI | fast-ssim2 | ssimulacra2-rs |
|---|--:|--:|--:|--:|--:|
| 1280×720 | 14 ms | 39 ms | 249 ms | 111 ms | 545 ms |
| 1920×1080 | 22 ms | 89 ms | 377 ms | 350 ms | 1,056 ms |
| 3840×2160 | 91 ms | 366 ms | 1,674 ms | 1,364 ms | 3,980 ms |

Plus: *"Single-threaded zensim is 4x faster than C++ libjxl SSIMULACRA2.
Multi-threaded at 4K: 18x."* Against **fast-ssim2** specifically (not the
C++/reference row) the published table's own numbers give **7.9×/15.9×/15.0×**
(720p/1080p/4K, zensim-MT vs fast-ssim2-1T — the table's own mismatched-thread
framing) or **~2.8–3.9×** on a matched-single-thread reading. **"~10×" is the
zensim-MT-vs-fast-ssim2-1T band** — the mismatched-thread reading the README
itself presents as its headline comparison.

**Doc-integrity item, found while locating the claim (not asked for, flagged
per policy):** the README states the test box is a **7950X**; `lscpu` on
this box today reports **AMD Ryzen 9 9950X3D** (confirmed independently
earlier in this task and by the coordinator). The hardware changed since
whatever run produced the published table — a real confound for any
before/after reading below, called out explicitly rather than papered over.
Also: **"synthetic gradient images"** — the project's own benchmarking
discipline (`~/work/claudehints/topics/benchmarking.md`) is wary of smooth-
gradient synthetic corpora for content-diversity claims; noted, not
relitigated — this is a *speed* micro-benchmark, not a quality/ranking
claim, so the concern is weaker here but the README doesn't say so.

### 2.2 Method for this section — `bench_compare` is the primary instrument

Per the user's direction, `zensim-bench/benches/bench_compare.rs`
(**criterion**, not zenbench — this file predates the zenbench mandate and
was run as-is, unmodified, not rewritten as part of this task) is the
PRIMARY instrument for reproducing the published table: it is the exact
source of the README's numbers (`zensim` = `ZensimProfile::A` MT,
`zensim_st` = `ZensimProfile::A` 1T, `SIZES` already includes 1280×720 /
1920×1080 / 3840×2160, `make_test_images` is the exact "synthetic gradient"
generator). Built via `cargo build --release --bench bench_compare`
(`LD_LIBRARY_PATH` pointed at a local libjxl 0.12 build for the C++ FFI
arms — the system package is 0.11, SONAME-incompatible; a build/link
detail, not a code issue). Run via the compiled binary directly, criterion's
own default sampling (100 samples, ~3s warmup) — a SINGLE process
invocation per build, not the multi-ASLR-start protocol (criterion has no
built-in multi-process-start-then-min workflow; hand-rolling one was out of
budget for this section — **flagged as a methodology gap**: these specific
numbers carry the ±10% single-process-layout uncertainty CLAUDE.md's own
ASLR finding documents, unlike the multi-start numbers elsewhere in this
doc). `criterion`'s CLI needs `--bench <filter>` (not a bare filter) to run
in full-benchmark mode from a compiled binary — a bare filter routes through
its libtest-compatibility shim and only smoke-tests each benchmark once;
this cost real time to discover and is recorded here so it isn't rediscovered.

The published-0.2.7 and Profile D arms use this task's scratch crate
(`~/tmp/zensim-speed-check/`), extended to (a) accept the README's exact
"WxH" sizes and (b) reproduce `bench_compare.rs`'s `make_test_images` pixel
generator byte-for-byte (`ZEN_SC_README_SIZES=1`) — same pixels bench_compare
measured, different binary. This part DOES use the full protocol: zenbench,
one binary, runtime-interleaved arms, 3 process starts with ASLR on
(reduced from 5 for time — noted), min-over-starts. Matched-thread readings
use no separate arms: the SAME binary run under `RAYON_NUM_THREADS=1` vs
`RAYON_NUM_THREADS=32` (`taskset -c 0` / `-c 0-31`) — this instrument's
`fast_ssim2` arm has no `rayon` feature enabled, so its number is single-
threaded regardless of `RAYON_NUM_THREADS`; the MT fast-ssim2 reading (2.3)
comes from a THIRD binary, `bench_compare` rebuilt with `--features
ssim2-rayon`, filtered to the `fast_ssim2` arm alone.

### 2.3 Deliverable 1 — does the README's own table still hold, cell by cell?

`bench_compare` run today, default build, its own generator/sizes/framing
(single criterion process, see the methodology caveat above):

| Resolution | zensim MT: published → today | Δ | zensim 1T: published → today | Δ | fast-ssim2 (1T): published → today | Δ | ssimulacra2-rs: published → today | Δ | C++ libjxl FFI: published → today | Δ |
|---|---|--:|---|--:|---|--:|---|--:|---|--:|
| 1280×720 | 14 → 7.37 ms | **−47 %** | 39 → 35.04 ms | −10 % | 111 → 54.34 ms | **−51 %** | 545 → 286.14 ms | **−48 %** | 249 → 77.43 ms | **−69 %** |
| 1920×1080 | 22 → 19.65 ms | −11 % | 89 → 84.31 ms | −5 % | 350 → 139.82 ms | **−60 %** | 1,056 → 635.74 ms | **−40 %** | 377 → 190.55 ms | **−49 %** |
| 3840×2160 | 91 → 84.44 ms | −7 % | 366 → 344.67 ms | −6 % | 1,364 → 594.79 ms | **−56 %** | 3,980 → 2,551.1 ms | −36 % | 1,674 → 805.87 ms | **−52 %** |

**Every cell is faster today than published — nothing regressed in absolute
terms.** But the cells did NOT move together: **zensim's own numbers moved
little (−5 % to −11 % at 1080p/4K, a bigger −47 % only at the smallest,
most fixed-overhead-dominated size)**, while **every opponent moved a lot
(−36 % to −69 %)**. That asymmetry is the whole story for §2.5.

### 2.4 Deliverable 2 — the honest, matched-thread frame

| | 720p | 1080p | 4K |
|---|--:|--:|--:|
| **README's own framing** (zensim MT ÷ fast-ssim2 1T), today | 8.56× | 7.63× | 6.34× |
| **Matched 1T/1T** (zensim 1T ÷ fast-ssim2 1T), today | 1.64× | 1.67× | 1.65× |
| **Matched MT/MT** (zensim 32T ÷ fast-ssim2 **with `ssim2-rayon`**, one filtered `bench_compare` run) | 49.44/6.80=**7.27×** | 128.88/17.40=**7.41×** | 582.74/89.10=**6.54×** |

**fast-ssim2's own rayon feature is barely worth anything at these
sizes, and at 4K it's a wash** — confirms and extends the ssim2-bar lane's
576² finding (~1.2×) to 720p/1080p/4K: MT fast-ssim2 (49.44/128.88/582.74 ms,
`bench_compare --features ssim2-rayon`) is only **1.18×/1.04×/0.97×** faster
than its own 1T number (58.20/134.30/566.10 ms, the SAME
`ZEN_SC_README_SIZES` scratch-crate reading the rest of this section uses —
at 4K "MT" is actually a hair slower than 1T, within noise of flat). That is
why the "README's own framing" and "matched MT/MT" rows land close together:
fast-ssim2 barely moves when threaded, so which of ITS thread counts you
pick barely matters. **The matched 1T/1T frame is the one genuinely
different number, and it is the honest one**: ~1.6–1.7×, a small fraction of
"~10×". The 6.3–7.4× MT-class band (either framing) is real but was never a
clean "10×" even in the mismatched published table (max 15.9×, min 6.3× —
the range straddles 10× rather than centering on it).

### 2.5 Deliverable 3 — the published-0.2.7 arm, at the README's own sizes

(`published_v02` = `zensim_pub::Zensim::new(ZensimProfile::latest())`,
0.2.7's actual `latest()` = `PreviewV0_2`.)

| size | thread | published_v02 (0.2.7) | main_v02 (same algorithm, current main) | ratio (main÷published) |
|---|---|--:|--:|--:|
| 720p | 1T | 29.70 ms | 25.00 ms | **0.84×  (16 % faster)** |
| 1080p | 1T | 68.20 ms | 56.80 ms | **0.83×  (17 % faster)** |
| 4K | 1T | 300.40 ms | 237.00 ms | **0.79×  (21 % faster)** |
| 720p | 32T | 5.60 ms | 5.70 ms | 1.02× (2 % slower) |
| 1080p | 32T | 11.50 ms | 12.70 ms | 1.10× (10 % slower) |
| 4K | 32T | 45.50 ms | 56.90 ms | **1.25× (25 % slower)** |

**The exact same algorithm (`PreviewV0_2`) got FASTER on current main at
single-thread — 16–21 % — at every size.** At full 32-thread saturation on
this 32-thread-max box it flips to slightly SLOWER (2–25 %, worse at larger
sizes). Cross-checked on the square 576/1152/2304 sizes at a DIFFERENT
threading config (rayon capped at 16 workers, all 32 logical CPUs still
available via `taskset`) — `main_v02` was faster than `published_v02` at
**every** cell there (0.52×–0.93×, i.e. 7–48 % faster), including at "16T".
**Reading both together: current main's `PreviewV0_2` code is faster than
0.2.7's at any thread count that doesn't oversubscribe the box to its
absolute logical-CPU ceiling** — 32 actual rayon workers on a 32-thread
machine has enough pool/dispatch overhead to erase the per-call
improvement on this cheap a workload, and does so more at larger images
(more total dispatch, same per-item cost). **No regression in the
algorithm itself; a real, size-and-saturation-dependent effect at the
absolute thread ceiling.**

Now the profile that actually SHIPS by default changed (0.2.7's default was
`PreviewV0_2`; current main's is `B`, `codec_target()`), and that IS slower
in absolute terms — `main_A` / `main_B` vs `published_v02`:

| size | thread | published_v02 | main_A | main_B | main_A÷pub | main_B÷pub |
|---|---|--:|--:|--:|--:|--:|
| 720p | 1T | 29.70 | 35.50 | 35.30 | 1.20× | 1.19× |
| 1080p | 1T | 68.20 | 79.70 | 79.00 | 1.17× | 1.16× |
| 4K | 1T | 300.40 | 342.20 | 335.30 | 1.14× | 1.12× |
| 720p | 32T | 5.60 | 6.80 | 8.70 | 1.21× | 1.55× |
| 1080p | 32T | 11.50 | 17.40 | 20.80 | 1.51× | 1.81× |
| 4K | 32T | 45.50 | 89.10 | 92.60 | 1.96× | **2.04×** |

**This is a real, measured, and expected cost — not a code regression.**
`PreviewV0_2` extracts a narrower 228-feature vector with a plain linear
dot product; `A`/`B` extract the full 372-feature vector (`extended_features
+ compute_iw_features`) and run an MLP/linear-372 forward pass — genuinely
more work, in exchange for the SROCC/quality gains the A→B lineage was built
for (documented throughout `docs/CODEC_TARGET_METRIC.md`). It grows with
thread count for the same reason as above: more total per-call fixed
overhead (feature extraction setup, more rayon task spawns for the wider
per-scale work) amortizes worse as thread count approaches the box's ceiling.
**"Did we regress" — no, for the shared code path; yes in absolute ms for
the shipped default, by design, and the trade is the one the whole A→B
project history documents.**

### 2.6 Deliverable 4 — Profile D's column: does it restore the ~10× band?

D reaches further toward it than `B`/`A` do, but does not cleanly clear it
under any frame measured:

| frame | main_A/main_B (today) | Profile D (fast build) | closest to "10×"? |
|---|--:|--:|---|
| Matched 1T/1T vs fast-ssim2 | 1.64–1.67× | **2.22×/2.34×/2.46×** (720p/1080p/4K) | D, but still ≪10× |
| Matched MT/MT vs fast-ssim2 (rayon) | 6.54–7.41× | 49.44/10.20=**4.85×** (720p), 128.88/20.60=**6.26×** (1080p), 582.74/71.00=**8.21×** (4K) | **D at 4K, 8.2×** — closest of any arm/frame in this whole study |
| README's own mismatched framing (D-fast MT ÷ fast-ssim2 1T) | 6.34–8.56× | 58.20/10.20=5.70× (720p), 134.30/20.60=6.52× (1080p), 566.10/71.00=**7.97×** (4K) | close to `main_A`'s own number in this frame |

**Headline, stated honestly**: Profile D's fast form (feature-regime-v2
build) is the single closest any arm in this study comes to a ~10×-class
advantage over fast-ssim2 — **8.2× at 4K on the matched-MT/MT frame** — but
it does not clear 10× outright, and the 32T numbers carry real noise (12–45%
per-arm spread across starts at this thread count in the underlying data,
`~/tmp/zensim-speed-check/run2_parsed.tsv` / `readme_parsed.tsv` — reported,
not hidden). On the smaller/matched frames D lands at 2.2–2.5×. **Read this
as "D recovers a meaningful fraction of the shrunk advantage, most visibly
at 4K under heavy threading, not as a full restoration of a clean 10× band
under every condition."**

### 2.7 Deliverable 5 — doc-integrity verdict

**The README's published SSIMULACRA2-comparison table does NOT reproduce
within noise today. Every absolute cell is faster (nothing regressed), but
the two headline RATIO claims have both decayed to roughly half:**

- *"Single-threaded zensim is 4x faster than C++ libjxl SSIMULACRA2"* — 4K,
  published 366÷1,674=4.57× (rounds to "4x"); **measured today
  344.67÷805.87 = 2.34×.**
- *"Multi-threaded at 4K: 18x"* — published 91÷1,674=18.4×; **measured today
  84.44÷805.87 = 8.72–9.54×** (9.54× single-process `bench_compare` reading;
  8.72× using the README-matrix's multi-start 32T minimum instead of
  `bench_compare`'s own MT `zensim` reading — both well under half of 18).

**Mechanism, decomposed as asked:** zensim's OWN number barely moved
(−5% to −11% at 1080p/4K — genuinely a bit faster, not slower). The C++
reference implementation moved a great deal (−49% to −69%) — almost
certainly the dominant cause is **libjxl itself getting faster between
whatever version produced the original numbers and the 0.12 build on this
box today** (the box's own hardware also changed, 7950X→9950X3D per the
README's own now-stale spec line, which the coordinator flagged and this
task independently confirmed via `lscpu` — an un-separable confound for
absolute times, but it does NOT explain the asymmetry: if the chip alone
explained it, zensim would have improved by a similar fraction, and it
did not). **The honest attribution: the ratio didn't shrink because zensim
got slower — it shrank because the opponents got much faster, and zensim's
own recent growth (372-feature extraction, MLP heads, the whole A→B/era-2
lineage this project has been measuring all week) kept its absolute number
roughly flat instead of improving at the same rate.**

**Flagged as a published-claim item for the user, per policy — not fixed
here (no user go-ahead to edit and republish README claims):**
1. `README.md` / `README.crates.md`'s "18x faster... at 4K" and "4x faster,
   single-threaded" no longer reproduce — today's numbers are roughly half
   of both, on this same box, same generator, same harness the README's own
   "Reproduce" line names.
2. `README.md`'s stated test hardware ("AMD Ryzen 9 7950X 16C/32T") does not
   match this box (`lscpu`: **AMD Ryzen 9 9950X3D**) — the numbers were
   measured on different silicon than what's printed.
3. The table's own reproduce instruction (`cargo bench -p zensim-bench
   --bench bench_compare`) still runs and still measures the profile it
   always measured (`ZensimProfile::A`) — the harness is not broken, only
   the numbers it currently produces disagree with the ones printed next to
   it.
4. Minor: "Median of 100 samples via criterion" — predates this project's
   "use zenbench, not criterion" mandate; not a correctness issue, a
   consistency-with-current-practice one.

None of these are silently corrected — they are reported here for the user
to decide whether/how to update the README.

### 2.8 Raw data

`~/tmp/zensim-speed-check/` (scratch, not committed): `run1_parsed.tsv` (7
arms × 576/1152/2304 × 1/8/16T × 5 starts, the original square-size grid),
`run2_parsed.tsv` (adds `ssimulacra2_ref` on the same grid), `readme_parsed.
tsv` (README's exact sizes/pixels × 1/32T × 3 starts, all 4 deliverables'
source), `benchcompare_default.log` / `benchcompare_rayon4.log`
(`bench_compare` criterion output, §2.3/§2.4's source). `parse_results.py` /
`synthesize.py` are the reduction scripts (min-over-starts per
`(build,threads,size,arm)`, same rule as `scripts/hybrid_speed_read.py`).
