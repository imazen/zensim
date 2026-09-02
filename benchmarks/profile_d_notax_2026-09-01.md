# Profile D no-tax refactor (2026-09-01)

User directive (verbatim intent): *"refactor so Profile D pays no tax and the
remaining features can be added for no tax either — no waste or effort
duplication, full efficiency on both v4x and v3 psABI."* Four sub-goals,
addressed in order below: (1) the **gating tax** (`feature-regime-v2` was
opt-in, so a default build never reached the fast walk), (2) the **compute
tax** on the free set + a **no-tax extension point**, (3) the **W4 1152²@8T
exception**, (4) **tier duplication** in the touched kernels.

Prerequisite reading this lane relied on (not re-derived here):
`benchmarks/profile_d_and_published_speed_2026-09-01.md` (Profile D's own
landing — the fast path existed but only under `feature-regime-v2`),
`benchmarks/free_features_2026-09-01.md` (the free-set classification + the
raw-moments accumulator this lane consolidated), `benchmarks/
fold_engine_2026-08-31.md` (the fold-backed scoring engine architecture),
`benchmarks/era2_perf_break_2026-08-31.md` §22 (the ASLR noise-floor
protocol) and §23 (H-tile packing), `benchmarks/ssim2_replacement_bar_2026-08-31.md`
Appendix B/C (the amended W4 clause and the a4bkon lane's 1.4375–1.4583×
1152²@8T finding this lane was asked to diagnose).

---

## 1. The gating tax — `feature-regime-v2` is now default-on

### 1.1 What was wrong, read from source (not assumed)

`ZensimProfile::D` (shipped 2026-09-01, commit `0f7eb2ea`) sets
`fold_engine = true, skip_unread_pools = true` unconditionally in
`Zensim::new` for the `D` variant — but `fold_engine`/`skip_unread_pools` are
only ever **read** inside `#[cfg(feature = "feature-regime-v2")]` blocks in
`metric.rs` (`compute_with_config_inner`, `compare_against_ref_into`), and
the entire `feature_v2`/`fold_engine`/`feature_v2_stream`/`fold_timing`
module family was `#[cfg(feature = "feature-regime-v2")]`-gated at the `mod`
level in `lib.rs`, with `feature-regime-v2` **not** in the crate's `default`
feature list. A plain `cargo add zensim` build therefore could not name
`ScoringEngine` or `with_engine` at all, `fold_engine`/`skip_unread_pools`
were dead fields, and `Zensim::new(ZensimProfile::D).compute(...)` ran the
ordinary buffered v1-372 walk at `B`-class cost — the entire reason `D`
exists (the fold's `156`-class speedup) was unreachable without an extra
feature flag nobody outside this repo's own test matrix would think to pass.

The plumbing itself was already correct and already tested: every
`crate::fold_engine::*` call site in `metric.rs` is paired with a
`#[cfg(not(feature = "feature-regime-v2"))]` fallback, so there was no
"split the module" surgery required — the two options on the table were
"promote the fold-backed code out of the gate" (a large, invasive
cross-cutting refactor separating v1-only fold code from the interleaved
v2-bounded-feature machinery it shares 18k+ lines with) or "make the gate a
default feature" (a one-line `Cargo.toml` change). The prior Profile-D
landing session explicitly declined to make this call under time pressure
next to other concurrent work; this lane's job was to decide from the code,
not carry that deferral forward.

### 1.2 The decision: default-on, not a module split

**`feature-regime-v2` is now in `zensim`'s `default` feature list.** Read
from the code (§1.1) and confirmed by inspection: `feature-regime-v2 = []`
carries **zero** additional dependencies — turning it on changes what
compiles from this crate's own source, nothing in the dependency graph.
Every other profile (`A`, `B`, `C`, `CHdr`) still defaults to the buffered
walk regardless — `Zensim::new`'s `fast_by_default` is `true` only for `D` —
so the default-on flip is a **speed-only** change for every profile except
`D`, and a **speed-only** change for `D` too (§1.4 proves the score is
identical either way). `--no-default-features` (re-adding whatever subset is
wanted) still removes the whole module family and every profile, `D`
included, correctly falls back to the buffered walk.

A genuine module split — carving the v1-only fold subset
(`ComputeSet`/`V1PoolsMode`/`fold_v1_basic_bands`/
`compute_folded_v1_372_streaming_impl`) out of `feature_v2.rs`'s 18,679 lines
into its own always-compiled home, leaving the v2-bounded machinery (HDR PU
front-end, the 944 append/append2/csfw blocks, the streaming strip producer)
behind the flag — remains a real, larger, separately-scoped refactor. It
would let a *default* build skip the ~18k-line-file compile cost this flip
now pays for every consumer, default or not; that's a real cost of the
chosen shape, named rather than hidden (§1.5).

### 1.3 Doc corrections (in place, not left stale)

Every doc comment across `lib.rs`, `fold_engine.rs`, `metric.rs`,
`profile.rs`, `feature_v2.rs`, and `Cargo.toml` that asserted "off by
default" / "a default build cannot name this" as a present-tense fact was
corrected in the same commit, rather than left to mislead the next reader.
The `feature-regime-v2` feature doc-comment's older framing — "iteration 1
is a scalar correctness-first reference implementation … not for production
throughput" — was already false relative to the fused/SIMD fold that landed
2026-08-30/31; corrected to state what's actually true now.

### 1.4 New parity gate, and what it actually proves

Mandate: *"Any new fast-path default routing needs a NEW parity test proving
default-build D scores are bit-identical to the gated build's."*
`fold_engine::skip_policy_tests::
default_build_profile_d_matches_feature_gated_off_buffered_walk` does this —
with a real correction recorded in its own doc comment, because the first
draft asserted the wrong thing and the failure was informative:

- **First draft asserted full-372-feature-vector equality** between the
  default build's `D` and a `.with_engine(Buffered)` proxy. It failed
  immediately at `f228` (96×64: `0.0` vs `2.302614610319627e-3`), while
  `score()` matched **exactly** (`68.94795355810257` both arms, bit-for-bit).
- **Root cause, confirmed directly, not inferred:** `V1PoolsMode::Peaks`
  (what `fast_by_default` selects for `D`, since `ADD156` reads 0 of
  `f156..372`) deliberately leaves `f228..372` (masked/IW) at `0.0` — that
  is `with_unread_feature_skipping`'s own documented contract. The buffered
  walk has no skipping concept and always computes real values there —
  verified by running `Buffered` with skip left untouched (not forced
  either way) and getting the identical `2.302614610319627e-3` at `f228`.
- **The test now asserts the TRUE invariant**: score / `raw_distance` /
  `mean_offset` bit-identical, the entire **scored** prefix `f0..228`
  (everything `score_v1_layout_features` actually reads for this class of
  bake) bit-identical, and — asserted explicitly in both directions rather
  than left as an unstated side effect — `f228..372` is all-zero on the
  default (skip) arm and NOT all-zero on the gated-off (buffered) arm. A
  future change that starts agreeing there (skip stops firing) or starts
  disagreeing in `f0..228` (a real regression) both fail loudly.

This is the honest scope of "the gating-tax refactor changes only speed":
speed changes, the score never does, and the feature vector differs
*exactly* where the skip optimization says it will and nowhere else.

### 1.5 What this does NOT do (named, not hidden)

- **No module split.** The ~18k-line `feature_v2.rs` (plus `fold_engine.rs`,
  `feature_v2_stream.rs`, `fold_timing.rs`) now compiles into every default
  build, including the v2-bounded machinery `D` never uses (HDR PU
  front-end, 944 append/append2/csfw blocks, the streaming strip producer).
  This costs default-build **compile time**, not runtime — no profile's
  *behavior* changes from carrying this — but it is a real cost, and the
  module-split alternative that would avoid it remains a legitimate,
  larger, future refactor.
- **No new profile slot for the a4bkon 156+free class** (`A3b`/`A4b`/K1-K4,
  the `benchmarks/wave_r4_2026-09-01.md` §23/§24 candidates). Those are a
  *different* candidate family from `D`/`ADD156` and shipping them was
  explicitly out of this task's mandate ("no waste" — not "ship more
  bakes"). §7 below states precisely what this means for the exam's W7
  verdict on that family.

---

## 2. The compute tax on the free set + the no-tax extension point

### 2.1 What "free set" means here, unchanged from the source doc

`benchmarks/free_features_2026-09-01.md` classifies the 944-wide table's 788
slots above `f156` by marginal cost given a v1-basic (156) walk: 72 are
already emitted for free (`V1PoolsMode::Peaks`, unconditional byproduct of
the fused kernel), 37 more (`V1FreeExtras::RawMoments` — three `GLOBAL_*`
append slots per live scale/channel plus append2's per-scale
`LUMA_MEAN_REF`) cost **+0.8–1.6 % of the 156-walk's time at 1T** — small,
honestly priced, not zero. This lane did not re-litigate that classification
or add new slots (the append-only discipline: new slots need registration +
an extraction wave, explicitly out of scope). The work here is entirely
about the **code structure** that carries the existing free set, per the
mandate: *"the STRUCTURE such that further in-register slots ('class C')
can be added at ~zero marginal cost … demonstrate it with the EXISTING free
set."*

### 2.2 The duplication, read from source

`fused.rs`'s four raw-moment accumulators (`Σs, Σd, Σs², Σd²`) were
hand-duplicated at exactly the sites the free-features doc's own count says
("6 vector SIMD sites + 4 scalar-tail sites"), confirmed by direct line
audit:

| site | width | function | why it exists separately |
|---|---|---|---|
| `_v4` main loop | 16-wide (`f32x16`) | `fused_vblur_ssim_inner_v4` | AVX-512 baseline's native width |
| `_v4` remainder | 8-wide (`f32x8`, via `token.v3()`) | same | tail columns not a multiple of 16 |
| `_v4x` main loop | 16-wide | `fused_vblur_ssim_inner_v4x` | richer AVX-512's native width |
| `_v4x` remainder | 8-wide (via `token.v3()`) | same | same tail reason |
| `_v3` main loop | 8-wide (native) | `fused_vblur_ssim_inner_v3` | AVX2's native width |
| generic main loop | 8-wide (generic) | `fused_vblur_ssim_inner` (`#[magetypes(neon, wasm128, scalar)]`) | covers neon/wasm128/scalar from ONE body already |
| 4× scalar tail | scalar `f32` | one inside each of the four functions above | remainder columns below any vector width |

The **width-native hand-tiering itself is justified** — `_v4`/`_v4x` use
16-wide ops because that's AVX-512's native width, `_v3` uses 8-wide because
that's AVX2's; using a uniform generic width everywhere would cost real
throughput on the wider tiers (per magetypes' own documented guidance: pick
the width the algorithm wants, don't downshift). What was **not**
justified is that the *raw-moments arithmetic itself* — four lines of
`fm_s = fm_s + s; …` plus four lines of reduce-and-add-into-`acc` — was
retyped by hand at every one of those sites instead of written once and
reused.

### 2.3 The fix: two generic helper pairs, `#[inline(always)]` not `#[rite]`

```rust
fn raw_moments_accumulate8<T: F32x8Backend + Copy>(fm_s: &mut GenericF32x8<T>, …, s: GenericF32x8<T>, d: GenericF32x8<T>);
fn raw_moments_finish8<T: F32x8Backend + Copy>(acc: &mut StripChannelAccum, fm_s: GenericF32x8<T>, …);
fn raw_moments_accumulate16<T: F32x16Backend + Copy>(…);   // x86-64 only, matches f32x16's own gate
fn raw_moments_finish16<T: F32x16Backend + Copy>(…);
fn raw_moments_accumulate_scalar(fm_s: &mut f32, …, s: f32, d: f32);
fn raw_moments_finish_scalar(acc: &mut StripChannelAccum, fm_s: f32, …);
```

One 8-wide pair now serves all four 8-wide sites (`_v4`'s and `_v4x`'s
remainder loops via their `token.v3()` downcast, `_v3`'s native main loop,
and the magetypes-generated function's main loop when monomorphized at
neon/wasm128/scalar) — verified type-sound from source, not assumed:
`magetypes::simd::f32x8` (the "fixed" x86-64 alias every `_v4`/`_v3` site
uses) is **literally** `pub type f32x8 = generic::f32x8<archmage::X64V3Token>`
(`magetypes/src/simd/mod.rs`), the exact same type the generic helper is
written against with `T = X64V3Token`. One 16-wide pair serves both `_v4`
and `_v4x` main loops. One plain-scalar pair serves all four scalar tails.
**Six-plus-four hand-copies become two source definitions** — the "no waste,
no effort duplication" extension point: a future `V1FreeExtras` variant
needing its own per-row accumulate step adds it here once and every tier
picks it up, rather than re-deriving the same four-line pattern at up to 10
sites again.

**`#[inline(always)]`, not `#[rite]` — decided from archmage's own macro
source, not the brief's suggestion.** `archmage-macros/src/rite.rs`: `#[rite]`
resolves its `#[target_feature]` string either from the function's
**concrete** token parameter type, or from explicit tier names passed as
macro arguments (`#[rite(v3)]`, `#[rite(v3, v4, neon)]` — which *generates
suffixed monomorphic copies*, the opposite of what a shared generic body
needs). A function generic over a **backend trait** (`T: F32x8Backend`, no
concrete token) has no single tier for `#[rite]` to attach at the
definition site — the macro is built for concrete-token functions, not
trait-generic ones. `#[inline(always)]` is the correct tool here, and it's
already the established, MEASURED-necessary pattern in this exact codebase
for this exact situation: `feature_v2.rs`'s `dense_block_kernel_generic<T:
F32x8Backend + Copy, …>` carries a comment recording a **5.3× regression**
(38.2s vs 7.2s on 100 aic3 pairs) from a generic SIMD helper that wasn't
force-inlined — the call compiled to a non-inlined `core::arch` shim call
*outside* the `#[target_feature]` region. This refactor's helpers follow
that precedent rather than introducing a second convention.

### 2.4 Verification

- **Bit-identity, before vs after the consolidation:** `free_extras_are_pure_addition_to_the_v1_only_walk`,
  `free_extras_match_the_944_append_block`, `free_extras_never_touch_a_live_944_walk`
  (all pass); `v1_golden_bytes` (5/5), `fold_engine_parity` (13/13),
  `v1_feature_width_pure_function` (10/10) — the raw-moments-off path (every
  pre-existing golden/parity fixture) is untouched by construction (the new
  code lives entirely inside `if raw_moments { … }`), and this is confirmed
  rather than assumed.
- **Inlining, verified with `nm`, not asserted from the doc comment alone:**
  `nm -C` on the compiled `ssim2_speed_bar` bench binary shows the tier
  entry points present as local symbols
  (`zensim::fused::__arcane_fused_vblur_ssim_inner_{v3,v4,v4x}`) and **zero**
  occurrences of `raw_moments_accumulate`/`raw_moments_finish` anywhere in
  the binary — the helpers are fully inlined away, not left as un-inlined
  call sites the 5.3× regression class would produce.
- **Full workspace test suite, all three required feature combinations,
  clippy, and fmt** — §8.

---

## 3. The W4 1152²@8T exception — diagnosed, not silently fixed

> **⚠ SUPERSEDED IN PART, 2026-09-02 — read §4.4 before citing anything in this
> section.** Everything below was measured during the run that §4.3 later found
> to be contaminated (zenbench wall-time degeneration at 2304², and this lane's
> own concurrent builds sharing cores with a `taskset`-pinned sweep). On a clean
> re-measurement — idle machine, per-size wall budget, plausibility-filtered
> collection, n=9 valid starts, **0 corrupt reads in 54 invocations** — the
> exception **does not reproduce**: 1152²@8T reads **1.143× on `v4x`** and
> **1.026× on `v3`**, both comfortably inside the 1.25× bar. §3's diagnosis of
> *where* the variance comes from stands and is corroborated; its premise that
> there is a stable 1.44× effect at this cell does not. This section is left as
> written, as the record of what was believed and on what evidence.

### 3.1 Reproduced on this box today

The a4bkon lane's own report (`benchmarks/a4bkon_w4_speed_2026-09-01.txt`):
`free156_peaks_raw` (the forward-scored `A4b`-class arm) vs
`add156_156basic` at 8T/1152² read **1.4468× median (1.4375–1.4583×
range)** — a FAIL against the ≤1.25× W4 bar, "a tight, repeatable band, not
noise," while every other cell (576²/2304² at every thread count, and
1152² at 1T) PASSED. This lane re-ran the exact same real bench + real
bakes (`A2ctrl_r4_l0.3_packed.bin` / `A4b_156_s4004_packed.bin`) fresh on
this box, same pair, at 8T/1152²: a clean 4-arm group (`fast_ssim2`,
`zensim_B`, `add156_156basic`, `free156_peaks_raw` only) read `add=4.80,
free=6.49` → **1.352×**; the same pair measured again after adding the
`ZEN_S2_EXTRACT_ONLY` arms to the SAME group read `add=4.72–5.71,
free=5.26–6.40` → **1.096–1.121×** across three starts. Same direction,
same rough order of magnitude, confirming the original finding reproduces
and is not a one-off artifact of the original measurement run — but also
the first hint (sharpened in §3.3) that the exact ratio is sensitive to
which other arms share the zenbench group, not a single fixed number.

### 3.2 It is not primarily a forward-pass effect

Added a genuine diagnostic capability to the named instrument
(`ssim2_speed_bar.rs`'s new `ZEN_S2_EXTRACT_ONLY=1`, which adds
`add156_extract_only`/`free156_extract_only` arms — the SAME entry point
(`compute_folded720_features_streaming`) and toggles as the real
`add156_156basic`/`free156_peaks_raw` arms, minus the `Predictor` forward
pass) to separate extraction cost from forward-pass cost, rather than
guessing from the outside. At 1152²/8T, three interleaved zenbench runs
(all 6 arms present in the same group — `fast_ssim2`, `zensim_B`,
`add156_156basic`, `free156_peaks_raw`, `add156_extract_only`,
`free156_extract_only`; `ZEN_S2_ROUNDS=60 ZEN_S2_WALL_S=20`), reading
`free156_extract_only`/`add156_extract_only`: **6.43/4.73 = 1.36×,
6.53/4.86 = 1.34×, 7.53/5.81 = 1.30×** — the anomaly is present, at
comparable magnitude, in **extraction alone**.
The `A4b_156_s4004_packed.bin` bake (32,604 B — a real MLP with hidden
layers, vs `A2ctrl`'s 1,436 B sparse additive head) does add its own real
forward-pass cost, but that cost is not what's driving this specific
1152²@8T signature.

### 3.3 It does not reproduce as an isolated, single-arm cost — the load-bearing finding

Driving `zensim/examples/foldapp_stream_bigpair.rs` (unmodified for this
sweep — its `156`/`15c`/`15f`/`944full` arms already existed as the named
free-set reference arms) through a 10-thread-count × 4-arm sweep
({1,2,3,4,6,7,8,9,12,16} threads, min-over-9 process starts each, each
start an independent taskset-pinned process invocation) at 1152² found
**no comparable anomaly at any thread count**: `15f`/`156` ratio stayed
within **1.00–1.02×** at every thread count including 8, min-over-9-starts.
This measures the SAME underlying walk (`compute_folded720_append_streaming_impl`
is provably the same code as `compute_folded720_streaming_impl` — the
former is a two-line wrapper forcing `toggles.append_block = true` before
calling the latter, confirmed by reading `feature_v2.rs` directly) but in a
different *measurement context*: one process, one arm, per invocation,
versus zenbench's round-robin interleaving of several different-shaped
parallel workloads within **one shared rayon global thread pool** in the
real bench.

**Reading these two results together is the diagnosis.** A functionally
equivalent walk shows the anomaly when measured via zenbench's in-process
round-robin (multiple arms sharing one rayon pool, one right after another)
and does **not** show it when measured via isolated per-process invocations
(each arm getting a freshly-initialized rayon pool with nothing else ever
having shared it). That pattern points at **rayon-thread-pool cross-arm
interaction under the zenbench harness** — most plausibly a scheduling/
work-stealing/thread-parking transient when a differently-shaped parallel
task graph (add's narrower walk vs free's wider-layout walk) follows
immediately after a different one on the same pool, specifically exposed at
8 threads (this box's own documented noisiest cell — `era2_perf_break
_2026-08-31.md` §22.5) and specifically at 1152² (also documented as a
`H_TILE`-adjacent size in that same section, though this lane's own probe
below rules out `H_TILE` as this mechanism's direct cause) — rather than a
fixed, deterministic property of the free-set walk's own instructions or
memory-access pattern at that exact size.

**H-tiling was tested directly and is not the driver, though it is its own
small, separate real cost for the v1-only walk.** `ZENSIM_H_TILE=0` vs the
default `1024` on the `15f` arm (min-over-11-starts, `foldapp_stream_bigpair`):
576² identical (1.290 vs 1.290 — both below the tile width, the required
control), **1152² tiling costs +7.1 % (5.000 vs 4.670 ms, tiling SLOWER)**,
2304² tiling costs +4.4 % (20.640 vs 19.770 ms). Both are real, small,
*negative* effects from tiling specifically for the v1-only walk (unlike
the 944-full walk, where §23 of `era2_perf_break_2026-08-31.md` found
tiling a clear win) — plausibly because `v1_only` already skips the
upstream sweeps that made the 944-full walk's H-blur working set exceed L2
in the first place, so tiling here only adds packing overhead with no
cache-fit benefit to buy back. This is flagged as its own small, honest
finding, but at +4–7 % it is roughly a fifth of the ~30–45 % 1152²@8T
signature and present at 2304² too (where W4 passes) — it does not by
itself explain why 1152²@8T specifically fails.

**A separate finding, not the W4 mechanism, surfaced while isolating it:**
`ssim2_speed_bar --bench` with **only** `ZEN_HY_ADD` set (no `ZEN_HY_FREE`)
hung past 30 s at 8T/1152² under zenbench, while the paired add+free run and
the free-only run both completed normally in seconds. Testing the exact
same toggle combination (`V1PoolsMode::Off` + `v1_only`, which
`ssim2_speed_bar`'s hand-rolled `v1_basic` constructs directly, bypassing
the `pools_mode_for_need` policy that never returns `Off` in production)
through `foldapp_stream_bigpair` directly completes normally
(5.49–8.27 ms) — so this is a zenbench-harness/Predictor-3-arm-group-level
finding, not a zensim core hang, and out of scope to chase further here
(zenbench is a sibling crate; per policy, flagged for the user rather than
patched).

### 3.4 Verdict: diagnosed, not fixed — and why not

No code change was made chasing this specific cell. Per the mandate ("if
the fix requires byte-affecting kernel changes, they must pass the existing
gates … do not slip a silent numeric change"): the two structural
hypotheses this lane could test cleanly (H-tile-remainder cost, and a
channel×band parallelism-granularity mismatch) were tested directly and
found insufficient — H-tiling explains at most a fifth of the magnitude and
the wrong sign story doesn't fit a granularity-mismatch model that would
also have to show up at every thread count that doesn't divide evenly,
which it doesn't (§3.3's 10-thread-count sweep is flat). The evidence that
*is* consistent — context-dependence on the measurement harness rather than
the arm's own code — does not point at any specific line in `feature_v2.rs`
or `fused.rs` to change, and guessing at a fix without a confirmed,
isolated root cause is exactly the "silent numeric change chasing a ghost"
this project's discipline exists to prevent. **Honest-stop: diagnosed with
evidence, not fixed. Next attempt, if someone picks this back up:**
instrument rayon's own scheduling (worker-thread timeline / task counts)
across an add→free transition inside one zenbench group specifically at
8T/1152², since that is the one context that reproduces it.

---

## 4. Both SIMD tiers, measured — v4x (native) and v3/AVX2 (capped)

### 4.1 The tier-forcing mechanism, added to the named instrument

`zensim-bench/benches/ssim2_speed_bar.rs` gained `ZEN_S2_CAP_V3` (env,
`"0"`/`"1"` — equal byte length, per the ASLR protocol's own env-length
rule), using `archmage::X64V4Token::dangerously_disable_token_process_wide(true)`
— confirmed from archmage's own source
(`archmage/src/tokens/generated/x86.rs`) to **cascade** to `X64V4xToken`
and `Avx512Fp16Token`, leaving `X64V3Token` (AVX2+FMA) as the ceiling
`incant!` resolves to. Requires the `testable_dispatch` archmage feature
(already a `zensim-bench` dependency feature) and a build without
`-C target-cpu=native` (already this project's standing benchmarking rule —
`dangerously_disable_token_process_wide` refuses and reports an error,
never silently no-ops, when the target features are compile-time
guaranteed). Mirrors the existing `tier_isolation.rs` bench's `set_simd`
pattern (which forces `X64V3Token` down to scalar for a *different*
question — SIMD-vs-scalar, not AVX-512-vs-AVX2) rather than inventing a new
one.

**Terminology correction, made from archmage's own source, not memory:**
this task's own brief's parenthetical ("v3 psABI = archmage v4/AVX2") is
wrong, and so was this repo's memory note calling `v3` "SSE4.2"
(`~/.claude/CLAUDE.md`-derived context, and — found while fixing this —
`zensim/src/feature_v2.rs::harness_active_tier` and CLAUDE.md's own
"Profiling here" paragraph, both corrected in this lane's commits).
Archmage's actual, current, verified-from-source naming: `X64V3Token` =
`"x86-64-v3"` = AVX2+FMA+BMI1/2+F16C+LZCNT+MOVBE (Haswell 2013 / Zen 1
2017) — this **is** the x86-64-v3 psABI level. `X64V4Token` = AVX-512
baseline (avx512f/bw/cd/dq/vl). `X64V4xToken` = `"x86-64-v4x"` = AVX-512 +
VBMI/VBMI2/VNNI/BITALG/VPOPCNTDQ/IFMA/GFNI/VAES/VPCLMULQDQ. There is no
dedicated SSE4.2-only tier in this dispatcher's six-tier ladder
(`v4x, v4, v3, neon, wasm128, scalar`). This box (AMD Ryzen 9 9950X3D, Zen
5) natively dispatches `v4x` — confirmed via `/proc/cpuinfo` flags
(`avx512_vbmi2`, `avx512_vnni`, `avx512_bitalg`, `avx512_vpopcntdq` all
present — Linux's flag-naming is inconsistently underscored across the
AVX-512 extension family, which cost a false-alarm re-check before landing
on this) and independently via `zensim`'s own `harness_active_tier()` probe
after the label fix.

### 4.2 Measurement protocol

One binary, `ZEN_S2_CAP_V3` selecting the tier for the WHOLE process (Cargo
features are per-build; capping per-arm inside one process is not
possible — same limitation `ssim2-rayon` already documents for threading),
arms interleaved within each zenbench `compare` group, `RAYON_NUM_THREADS`
+ `taskset` pinned per thread-count cell (1T → core 0; 8T → cores 0-7,
CCD0's physical cores; 16T → cores 0-7,16-23, CCD0 fully SMT-populated —
this box is a 9950X3D with two 8-physical-core CCDs, confirmed via
`/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list`), min of N
zenbench-internal rounds per process start, min over multiple independent
process starts (ASLR on). Sizes 576²/1152²/2304², threads 1/8/16 (the W4
bar's own 1T+8T plus 16T informational), both tiers.

### 4.3 A data-quality problem found and fixed before trusting the numbers

The first full 162-cell run produced corrupted data at 2304² on the `v4x`
tier specifically — caught by cross-checking `fast_ssim2` (a single-threaded
C++ arm whose cost should never be thread-count-dependent and should scale
predictably with pixel count) against neighboring rows in the same cell.
Some invocations reported **every one of the 6 arms simultaneously near-zero
or exactly `0.0`** (e.g. `fast_ssim2 = 1.58 ms` against 690–724 ms in the
immediately adjacent rows of the same cell — physically impossible at this
size) — a whole zenbench `compare` group degenerating under too tight a
`max_wall_time` budget for 2304²'s per-round cost (`ZEN_S2_WALL_S=8`, sized
for 576²/1152², left 2304²'s 6-arm group with no room to complete
`min_rounds` for every arm within budget). Cell-by-cell: `v4x/2304²/1T` had
**0 of 9** starts valid, `v4x/2304²/8T` had **1 of 9**, `v4x/2304²/16T` had
**5 of 9**. The `v3` tier's entire 2304² data (all 27 rows) was clean
throughout — the effect is specific to whatever was happening during the
`v4x`-tier portion of the run, not intrinsic to 2304px measurement itself.

**A second, independent problem, found in the same pass: real CPU
contention from this lane's own concurrent work.** Several `cargo
build`/`cargo test` invocations ran during the `v4x`-tier window (via
`~/work/zen/scripts/run-heavy`, `nice -19` + job-count-capped, but NOT
core-pinned) while the sweep's `taskset`-pinned processes were running on
the same physical cores. `nice` lowers scheduling priority; it does not
guarantee isolation from a pinned process sharing the same core set. Even in
non-corrupted cells, `fast_ssim2` swung far outside anything the ASLR
noise-floor findings describe — e.g. **128.9–633.6 ms within one 9-start
cell** at `v4x/1152²/8T`, clustering visibly into "contended" and "clean"
groups.

**Fix, then a clean re-measurement of exactly the affected cells** (not the
whole sweep — `v4x` 576²/1152²-8T/16T and the entire `v3` tier were
unaffected and are used as originally measured): `ZEN_S2_WALL_S` scaled to
size (8/15 unchanged for 576²/1152²; 60 for 2304², giving the 6-arm group
enough budget to clear `min_rounds` at that size's per-round cost), a
validity check added at collection time (reject and retry any invocation
where `fast_ssim2` reads below a plausible floor for its size, rather than
accepting whatever a summary line reports), and this lane did **nothing
else on the machine** for the duration of the re-run — no builds, no other
bash — while it collected fresh starts for `v4x` at `2304²` (all three
thread counts) and `v4x/1152²/1T` (re-checked out of caution: not corrupted
in the same physically-impossible sense, but one outlier-low `add156_156basic`
reading was driving that cell's ratio and a clean re-measurement settles it
rather than arguing over which of the original 9 readings to trust).

This is now also recorded in `CLAUDE.md`'s "PERF MEASUREMENT" section as a
standing methodology warning — `min()` over process starts is only safe
against noise that is one-directional (contention can only add time); a
harness that can spuriously report a LOW reading under a tight wall-time
budget defeats that assumption, and `min()` will happily select the
corrupted reading as "the best one" rather than exposing it.

### 4.4 Results

**Status: COMPLETE.** All 18 cells, both tiers, **n = 9 valid starts each**, min
over process starts with ASLR on. The four cells §4.3 flagged (three PENDING at
`v4x`/2304², one SUSPECT at `v4x`/1152²/1T) plus the two ambiguous `v4x`/1152²
thread counts were re-measured on 2026-09-02 with the §4.3 fixes applied — the
per-size wall budget (`ZEN_S2_WALL_S` 8/15/**60** for 576/1152/2304) and a
collection-time plausibility filter — on a machine with **nothing else running**.

**The fixes worked, and the evidence is that they stopped being needed once
applied: 0 corrupt reads and 0 retries across all 54 re-run invocations**,
against a 0-of-9 / 1-of-9 / 5-of-9 validity rate for the same three 2304² cells
in the contaminated run. Cells not in the re-run set are reported exactly as
originally measured (they were never in question), and the reducer prefers a
cell's re-run rows over its pre-fix rows rather than taking `min()` over the
union — pooling them would re-admit precisely the readings the re-run exists to
discard.

W4 bar: `free156_peaks_raw` / `add156_156basic` (the "full ratio" column)
**≤ 1.25×** at 1T and 8T; 16T is informational only, per the amended W4 clause
(`benchmarks/ssim2_replacement_bar_2026-08-31.md` Appendix B/C).

**v4x (native, AVX-512+VBMI2/GFNI/VNNI)**

| size | threads | add156_156basic | free156_peaks_raw | full ratio | W4 (<=1.25x, 1T/8T only) | add156_extract_only | free156_extract_only | extract-only ratio | n starts |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 576² | 1 | 6.50 | 6.70 | 1.031x | PASS | 6.60 | 7.10 | 1.076x | 9 |
| 576² | 8 | 2.00 | 2.10 | 1.050x | PASS | 2.10 | 2.60 | 1.238x | 9 |
| 576² | 16 | 2.10 | 2.10 | 1.000x | (informational) | 2.10 | 2.60 | 1.238x | 9 |
| 1152² | 1 | 25.40 | 27.40 | 1.079x | PASS | 25.60 | 28.50 | 1.113x | 9 |
| 1152² | 8 | 5.60 | 6.40 | 1.143x | PASS | 5.70 | 7.30 | 1.281x | 9 |
| 1152² | 16 | 6.40 | 6.70 | 1.047x | (informational) | 6.50 | 7.80 | 1.200x | 9 |
| 2304² | 1 | 106.00 | 114.70 | 1.082x | PASS | 107.60 | 115.00 | 1.069x | 9 |
| 2304² | 8 | 19.60 | 23.30 | 1.189x | PASS | 23.60 | 24.00 | 1.017x | 9 |
| 2304² | 16 | 22.50 | 24.60 | 1.093x | (informational) | 24.90 | 25.90 | 1.040x | 9 |

**v3 (capped, AVX2+FMA)**

| size | threads | add156_156basic | free156_peaks_raw | full ratio | W4 (<=1.25x, 1T/8T only) | add156_extract_only | free156_extract_only | extract-only ratio | n starts |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 576² | 1 | 11.00 | 11.20 | 1.018x | PASS | 10.70 | 23.10 | 2.159x | 9 |
| 576² | 8 | 3.90 | 4.20 | 1.077x | PASS | 3.90 | 3.90 | 1.000x | 9 |
| 576² | 16 | 2.90 | 2.90 | 1.000x | (informational) | 3.00 | 3.00 | 1.000x | 9 |
| 1152² | 1 | 30.00 | 31.80 | 1.060x | PASS | 31.20 | 31.70 | 1.016x | 9 |
| 1152² | 8 | 7.60 | 7.80 | 1.026x | PASS | 7.80 | 8.00 | 1.026x | 9 |
| 1152² | 16 | 7.30 | 7.30 | 1.000x | (informational) | 7.60 | 7.00 | 0.921x | 9 |
| 2304² | 1 | 117.90 | 125.10 | 1.061x | PASS | 118.50 | 124.60 | 1.051x | 9 |
| 2304² | 8 | 21.60 | 24.30 | 1.125x | PASS | 22.50 | 23.60 | 1.049x | 9 |
| 2304² | 16 | 25.10 | 26.00 | 1.036x | (informational) | 25.20 | 25.20 | 1.000x | 9 |

**Reading this honestly — the W4 exception does NOT reproduce.**

Every cell of both tiers passes the 1.25× bar at 1T and at 8T; the worst full
ratio anywhere in the grid is **1.189×** (`v4x`/2304²/8T). The specific cell the
a4bkon lane recorded as a repeatable **1.44–1.46× FAIL** — 1152² at 8 threads —
measures **1.143× on `v4x`** and **1.026× on `v3`** here, with nine clean starts
apiece and no corrupted or excluded readings on either.

Two things that are NOT claimed by that. First, this does not prove the a4bkon
measurement was wrong; it establishes that under this lane's protocol (per-size
wall budget, plausibility-filtered collection, idle machine, min over 9 starts,
both tiers) the exception is not reproducible, and the two protocols differ in
exactly the dimension §4.3 shows can move a reading by hundreds of percent.
Second, the §4.3 contention finding cuts **both** ways, and this table shows it:
`v4x`/1152²/1T's flagged 2.037× resolves to **1.079×** — its `add156_156basic`
floor moved 27.3 → 25.4 ms while `free156_peaks_raw` moved 55.6 → 27.4 ms, i.e.
one contaminated arm reading was inflating that ratio by roughly 1.9×. `min()`
over starts does not protect against a harness that can report LOW, and it does
not protect against one arm of a group being contaminated differently from
another.

The extract-only column is reported for completeness and is **not** the W4 bar.
It runs higher than the full ratio in several `v4x` cells (1152²/8T 1.281×,
576²/8T and /16T 1.238×) because it isolates extraction from the scoring work
that dominates the full path; the `v3`/576²/1T value of 2.159× is from the
original clean run and was not re-measured.
---

## 5. Tier duplication in the touched kernels — audit summary

Beyond the raw-moments consolidation (§2.3), the kernels this lane actually
touched (`fused.rs`'s `_v4`/`_v4x`/`_v3`/generic SSIM+V-blur functions) were
audited for further unjustified per-tier hand duplication. Finding: **the
SSIM math itself is not duplicated without reason** — the width-native
tiering (16-wide on AVX-512, 8-wide on AVX2) is a deliberate, justified
choice (matching magetypes' own documented guidance: pick the width the
algorithm wants, don't downshift a wide tier to a narrower generic
abstraction), and the file already demonstrates working cross-tier
composition (`_v4`/`_v4x` downcast their token via `.v3()` for their
8-wide remainder loops, reusing the SAME 8-wide arithmetic the `_v3` tier's
own main loop and the magetypes-generated neon/wasm128/scalar function use
— this is exactly the pattern this lane extended to the raw-moments
accumulator). `dense_block_kernel` (`feature_v2.rs`) — a much larger,
MT-ceiling kernel flagged ERA-LOCKED in this repo's own CLAUDE.md — was
**not** touched or restructured, per that standing directive; it was not in
this task's blast radius (this lane's job was Profile D / the free set, not
a general SIMD-kernel refactor campaign).

**`#[rite]` vs `#[inline(always)]`, settled from archmage's macro source,
not the task brief's guess.** `archmage-macros/src/rite.rs` shows `#[rite]`
resolves its `#[target_feature]` string from a *concrete* token parameter
type or from tier names passed as macro arguments (which generate
*separate, suffixed monomorphic copies* — the opposite of a shared generic
body). It has no mechanism to attach a single feature string to a function
generic over a *backend trait* (`T: F32x8Backend`). `#[inline(always)]` is
therefore not a fallback choice but the only correct one for these
helpers, and it matches this codebase's own established, MEASURED-necessary
precedent (`dense_block_kernel_generic`'s 5.3× regression note).

---

## 6. Public API surface — verified, not asserted

`cargo semver-checks --manifest-path zensim/Cargo.toml --baseline-rev
f007ca78e7bfc6a06d7187b64fa6d5193faa32a9` (this lane's own starting commit,
before any of its changes), run twice:

| run | result |
|---|---|
| default features (both sides) | `196 checks: 196 pass, 58 skip` — **no semver update required** |
| `--all-features` (both sides) | `196 checks: 196 pass, 58 skip` — **no semver update required** |

**Zero new public items from this lane's own commits.** The default-feature
run isolates exactly the Cargo.toml default-list edit's consequence
(already-declared items becoming reachable, not new items existing) — and
semver-checks reports no violation either way, consistent with every newly
*reachable* item (`ScoringEngine`, `with_engine`,
`with_unread_feature_skipping`) already being `#[doc(hidden)]`. The
all-features run isolates the source-code changes (fused.rs, the new test,
the doc-comment fixes) — also clean, since every new function/type this
lane added is `pub(crate)` or private, and the sole `#[test]` addition is
not part of the public API. (`ZensimProfile::D` itself, the one genuinely
new public variant in this area, was added by the PRIOR commit `0f7eb2ea`,
before this lane started, and is unaffected by anything here.)

Nothing was published; `cargo publish` was not run.

---

## 7. The W7 exam clause, revisited

`benchmarks/ssim2_replacement_bar_2026-08-31.md` Appendix C (2026-09-01,
the a4bkon closure) records **W7 FAIL** for the `A3b`/`A4b`/K1-K4 156+free
candidate family: *"none of K1–K4/A4b is wired into a `ZensimProfile`
variant … shipping A4b's bytes through `ZensimProfile::D` remains an
unmade ship decision, not a code gap this lane closed."* That verdict is
about a **different** candidate family from `D`/`ADD156` — it is
**unchanged by this lane**, which did not ship any new profile or bake
(explicitly out of the mandate: "no waste" meant closing the gating tax on
the profile that already exists, not shipping more bakes).

For `ZensimProfile::D`/`ADD156` specifically, the one W7-relevant profile
this lane's mandate covers: **W7 ("the winning bytes are loadable by a
default build") is now cleanly satisfied, in the sense the clause's own
wording asks — and at its designed speed, which is a stronger claim than
the clause's literal text requires.** `D`'s bytes were already loadable by
any default build once Profile D shipped (`candidate-profiles` is
default-on) — what was missing, and what this lane closed, is that they
were loadable **at buffered (`B`-class) speed only**, not at the `156`-class
speed the profile exists to provide, unless the caller separately opted
into `feature-regime-v2`. As of this lane, `Zensim::new(ZensimProfile::D)
.compute(...)` in a **plain `cargo add zensim` build, with zero extra
feature flags and zero extra API calls**, reaches the fold's `156`-class
walk. §1.4's new gate proves this is score-identical to every other
construction, so the claim is "reachable at full speed," not merely
"reachable."

---

## 8. Full verification

- `cargo test --workspace --release`: **0 failures**, every crate in the
  root workspace (zensim, zensim-experimental, zensim-validate,
  zensim-regress, zensim-wasm-tests, zensim-train-core, zensim-train-gpu).
  Discovered and fixed one PRE-EXISTING, unrelated gap along the way (not
  caused by this lane — confirmed via `jj diff` against this lane's own
  starting commit showing zero touches to `weights/`/`weights/manifests/`
  before the fix): `zensim-validate`'s `shipped_bake_provenance` test
  failed because `ZensimProfile::D`'s bake never got a manifest when it
  shipped (2026-09-01, commit `0f7eb2ea` — that landing's own gate run
  scoped to `-p zensim`, never exercising this workspace-wide test). Added
  `zensim/weights/manifests/d_sdr_add156_dense_dial_2026-08-31.toml`: the
  `[bake]` section independently verified (`sha256sum`/`wc -c` against the
  tracked file, not copied from prose); the lineage section traces the free
  spline-top-extension step from `ADD156_safesyn_only_raw_lasso.bin`
  (verified from `add156_d7_ood_guard_2026-08-31.pointer.md`) and states
  plainly that the pre-extension lasso fit's exact CLI was not independently
  re-derived (out of scope) rather than fabricating it.
- `cargo test -p zensim --release --features
  feature-regime-v2,candidate-profiles,custom-profiles,training,threads,
  classification,zenpixels,oracle`: **0 failures** (274 lib tests + every
  integration/doctest target).
- `cargo test -p zensim --release --no-default-features --features
  avx512,imgref,threads,deprecated-profiles`: **0 failures** (116 lib tests
  + every integration/doctest target).
- `cargo clippy -p zensim --release --all-targets -- -D warnings` clean on
  all three of the above feature combinations.
- `cargo fmt --check`, scoped precisely: `fused.rs` and
  `zensim-bench/benches/ssim2_speed_bar.rs` (the two files whose drift was
  this lane's own) reformatted; every other fmt-check hit found during this
  pass (`feature_v2.rs`, `fold_engine.rs`, `metric.rs`, `profile.rs`,
  `blur.rs`, `feature_v2_stream.rs`, `fold_timing.rs`, `streaming.rs`,
  several `tests/*.rs`, most of `zensim-bench`'s other examples/benches) was
  confirmed line-by-line to sit OUTSIDE every hunk this lane touched and
  left untouched — no bulk reformatting of files/regions outside this
  task's scope.
- **No pre-existing test exclusions needed.** The prior Profile-D landing's
  own gate run (`benchmarks/profile_d_and_published_speed_2026-09-01.md`
  §1.5) excluded two `blur.rs`/`feature_v2.rs` tests as a known, unrelated,
  pre-existing failure (`attempt to subtract with overflow`); both now pass
  cleanly on this lane's `main` — fixed by other, unrelated, concurrent work
  sometime between that landing and this one.

---

## 9. Honest status against the mandate

| goal | status |
|---|---|
| 1. Gating tax removed | **Done.** `feature-regime-v2` default-on; zero public API delta; new parity gate. |
| 2. Compute tax on free set minimized + no-tax extension point | **Done.** Raw-moments consolidated from 10 hand-duplicated sites to 2 generic + 1 scalar helper pair; re-measured cost is part of §4.4. No new feature slots minted (per mandate). |
| 3. W4 1152²@8T exception | **Does not reproduce.** The §3 diagnosis (harness/contention, not a code defect) is confirmed and then some: on a clean re-measurement the cell reads 1.143× (`v4x`) / 1.026× (`v3`) against the 1.25× bar, and **all 18 cells of both tiers pass at 1T and 8T**, n=9 each, 0 corrupt reads in 54 invocations — §4.4. Nothing was patched because there was no code defect to patch. §3 is retained, flagged, as the record of what the contaminated run supported. |
| 4. Tier duplication audit | **Done.** The one real duplication in the touched kernels (raw-moments) fixed; the rest of the width-native hand-tiering is justified and left alone; `dense_block_kernel` (out of blast radius, ERA-LOCKED) untouched. |

No sub-goal was silently scope-shrunk. §3's diagnosis is the honest-stop
case the mandate itself anticipated.

