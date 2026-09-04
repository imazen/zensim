# Class-C free slots — 24 more 944 positions the 156 walk can emit, and what they cost

Continues `benchmarks/free_features_2026-09-01.md` §4, which classified all 944
slots by marginal cost given a v1-basic-only (156) walk, shipped the 109-slot
"free set" (72 peaks + 37 raw-moment), and named **class C** — values already
live in the fused kernel's registers, never emitted — as the next tranche.
Companion: `benchmarks/profile_d_notax_2026-09-01.md`, which built the generic
`raw_moments_accumulate{8,16}` / `_finish` helper-pair extension point
explicitly so a new `V1FreeExtras` variant would be "one function to add, not
up to 10 hand-copies."

**The question:** of the 236 slots the classification calls class C
(`C-raw` 49, `C-blur` 154, `C-xch` 33), which can actually be emitted with no
new plane, no new load and no new pass — and what does emitting them cost?

**The answer, up front:** **24 of them can. They cost ~1.3–1.5 % of the 156
walk's wall time at the native (AVX-512) tier and ~2.0–2.3 % at the v3/AVX2
tier, measured at 1T; that is not zero and this doc does not call it free.**
212 cannot, for two distinct reasons stated in §4. Separately, and not part of
this lane's tranche, real-pixel testing found that **9.12 % of the PREVIOUS
lane's free-40 cells miss its own 2e-5 route-parity bar on real corpus pairs**
(§6.2) — a synthetic-only gate had not seen it.

---

## 1. What is left in class C, and why only 24 of it is reachable

`benchmarks/free_features_2026-09-01/slot_classification.tsv`, read per slot:

| sub-class | slots | reads | status after this lane |
|---|--:|---|---|
| `C-raw` — v2-348 `MSE` | 12 | `s, d` | **LANDED** |
| `C-raw` — `GLOBAL_*` / `LUMA_MEAN_REF` | 37 | `s, d` | already shipped (`V1FreeExtras::RawMoments`) |
| `C-xch` — `LUM_{DARK,MID,BRIGHT}_ERR` at **Y** | 12 | `s, d, ref_y` (and on Y, `ref_y` **is** `s`) | **LANDED** |
| `C-xch` — the same trio at **X / B** | 21 | `s, d, ref_y` (a genuinely foreign plane) | REFUSED — §4.1 |
| `C-blur` | 154 | `mu1, mu2, ssq, s12` | REFUSED — §4.2 |

The two landed families share one per-pixel value, which is why they land
together and why the pair is cheaper than either alone would suggest:

```
mse_i = sat((s − d)², C_MSE)          C_MSE   = 0.01
t     = sat(ref_Y,   C_LUM_T)         C_LUM_T = 0.35     sat(x,c) = max(x,0)/(max(x,0)+c)
```

* **v2-348 `MSE`** (12 cells, `f377 406 435 / 464 493 522 / 551 580 609 /
  638 667 696`) `= clamp01(Σ mse_i / n)`.
* **append `LUM_DARK_ERR` / `LUM_MID_ERR` / `LUM_BRIGHT_ERR`** at Y (12 slots,
  `f739-741 / 790-792 / 841-843 / 892-894`) — the dark/bright Bernstein-weighted
  means of that same `mse_i`, with **mid DERIVED** from the partition of unity
  (`Σw_mid = n − Σw_dark − Σw_bright`), exactly as `finish_append` derives it.

`pd = s − d` is the register the `acc.mse` line one row above already holds, so
the bounded error adds one `max`, one multiply, one divide and one add per
row-vector. The luminance weights add a second divide plus four accumulators,
and **only on Y** — see §4.1.

**Why these two, and not other cheap candidates.** The v1 basic block already
carries an MSE, but an *unsaturated* one (`Σ(s−d)²/n`). Its saturated sibling is
a different statistic: `sat` compresses outliers, so on near-lossless content —
the zone the metric is documented weakest in — the unsaturated mean is carried
by a handful of pixels while the bounded mean is not. The luminance bins split
that same bounded error by reference luma, which is the axis an HDR/tone-aware
model has no other cheap access to.

**No slot is renumbered and none is invented.** All 24 are existing 944-layout
positions that a v1-only walk leaves at their structural zeros; this fills them
by a cheaper route. That is the append-only feature-numbering discipline
satisfied, not bent — a genuinely NEW feature would need indices past `f943`
and a layout/era decision, which §4.2 is exactly about.

---

## 2. Implementation

`V1FreeExtras::RawMomentsPlusBoundedErr` — a strict superset of `RawMoments`.
The kernel parameter changed from a bare `raw_moments: bool` to one `Copy`
`fused::FreeExtrasWork { raw_moments, bounded_err, lum_bins }`, so all five
fused-kernel signatures keep exactly one free-extras parameter.

Arithmetic goes through the extension point the profile-D lane built: three
source definitions (`bounded_err_accumulate{8,16,_scalar}`, plus the `lum_bins`
and `finish` siblings), `#[inline(always)]`, generic over a backend **trait**,
serving **10 call sites** — `_v4`/`_v4x` native f32x16 loops, their `token.v3()`
f32x8 remainder loops, `_v3`'s native f32x8 loop, the
`#[magetypes(neon, wasm128, scalar)]`-generated f32x8 loop, and four scalar
tails. Single-source across `v4x / v4(AVX2) / v3 / neon / wasm128 / scalar`.
(`#[rite]` still does not apply, for the reason the raw-moments block states:
it resolves `#[target_feature]` from a concrete token, and these are generic
over a backend trait.)

Accumulators are band-batched — vector-add every row, `reduce_add` once at the
band's last inner row — the same shape and the same precision argument as the
raw moments (bounded to `V1_BAND_ROWS` = 32 rows of f32 before the f64 upgrade).

**One owner for the luminance bins.** `lum_bins_from_weighted_sums` is now
called by BOTH `finish_append` and the free finalize, so §5's parity gate tests
the ACCUMULATION rather than comparing two hand-copied formulas — the
discipline `global_stats_from_raw_moments` already set.

`fold_engine::wide_bake_v2_read` now returns the cheapest covering
`V1FreeExtras` instead of a bool, so a bake whose live columns include these 24
still gets the cheap v1-only walk rather than the "compute everything" fallback.

---

## 3. Gates

Six new tests. **Two failed first and found real defects** — they are in the
list because they caught something, not because failing-first was performed.

| gate | what it holds | measured |
|---|---|---|
| `class_c_extras_are_pure_addition_to_the_free_walk` | every pre-existing slot **bit-identical** with the variant on vs off (156 basic + 72 peaks + the 40 raw-moment slots), the 24 are `+0.0` with it off, all finite | 7 geometries × 2 arms, exact `to_bits()` |
| `class_c_kernel_constants_match` | the kernel's f32 constants equal what the 944 kernel's own SIMD path splats | exact f32 equality |
| `class_c_integrands_match_the_f64_scalar_oracle` | the per-pixel integrands against the **f64** `saturate` the append kernel's scalar tail calls, over a range including negatives and the near-zero regime `sat` amplifies ~1/C; plus the partition of unity the mid bin is derived from | worst **relative** \|Δ\|: **1.11e-7** (mse), **8.20e-7** (weights) — at the f32 floor |
| `class_c_extras_match_the_944_walk` | each of the 24 vs the SAME slot from an entirely different kernel (dense v2 for `MSE`, append for the bins) | 9 geometries × 2 arms; worst \|Δ\| **7.48e-9** (`LUM_MID_ERR` s2 Y, 208×144) against a 2e-5 bar |
| `class_c_extras_are_thread_invariant_and_finite` | bit-identical across rayon pools 1/2/8/16 **and** vs the serial walk; every slot finite | 4 geometries, exact `to_bits()` |
| `class_c_slot_set_is_append_only_and_disjoint` | 24 slots, disjoint from the free 40, inside the 944 layout, count derived from `append_cell_active`, and the 4-scale list spelled out so a renumber shows as a review diff | 1..4 scales |

`attribution::tests::attribution_covers_expected_slots_per_width` gained four
probe rows — the 24 positions are decomposable by the attribution density, so a
steering loop that reads them gets a real map rather than a silently empty one.

**The two first-failures, stated plainly:**

1. `class_c_extras_are_pure_addition_to_the_free_walk` failed on slot **733**
   (`GLOBAL_DMEAN` s0 c0) moving `2.368e-2 → 0`. The raw-moments EMISSION gate
   was written `toggles.free_extras == V1FreeExtras::RawMoments`, so the moment
   a superset variant was requested it silently zeroed all 40 raw-moment slots.
   Fixed to `!= Off`. **A model reading those slots would have been served
   zeros with nothing failing** — exactly the failure class this gate exists for.
2. `class_c_kernel_constants_match` failed as an `as f64` round-trip
   (`0.009999999776482582 != 0.01`). Neither constant is representable in f32,
   so that equality is false for a *correct* implementation; the claim that
   means something is f32 equality against `C_MSE as f32`, which is the value
   the 944 kernel itself splats. Corrected rather than loosened.

Full suite: **268 lib tests + every integration target, 0 failed**; clippy clean;
default, `--no-default-features` and full-feature builds all warning-free (the
last also clears 2 pre-existing dead-code warnings of the same class).

---

## 4. What was refused, and why

### 4.1 The 21 `LUM_*_ERR` slots on X and B — a new load stream

The bin weight is `sat(ref_Y, C_LUM_T)`, a function of the **reference luma**
plane. On the Y channel that plane IS this channel's `src` — the append
kernel keeps `ref_y` a separate argument only to make the dependence explicit
(`csfw_block_kernel_generic`'s own note) — so on Y the weight is a pure
register carry. On X and B it is a foreign plane: no new *compute*, but a new
*load stream* into a per-channel kernel, on 2 of 3 channels, for every pixel.
That is outside the class-C definition this lane was scoped to ("no new plane,
load or pass"), so those 21 slots stay at their structural zeros. It is a cheap
follow-up for a lane that wants to price a stream — not a blocker, and not
something to smuggle in under a "free" heading.

### 4.2 The 154 `C-blur` slots — a different value, and a layout change

The classification is explicit that these read the V-blurred `mu1/mu2/ssq/s12`
which ARE in the fused kernel's registers, but that **a v1 band re-inits its
V-blur recurrence at the band buffer top while phase A runs it over the whole
strip window**, so the emitted number would not be the 944 table's number.
Two independent disqualifiers follow:

1. They would be genuinely NEW features, needing 154 NEW slot numbers past
   `f943` — a layout/regime width change, i.e. an era decision that needs the
   user, not a lane.
2. Their value is a function of the band tiling, so they could not pass this
   lane's own gate 5 (bit-identity vs the serial walk and across pool sizes) in
   the form a 944-trained model would expect, and they would need their own
   semantic definition and their own oracle rather than inheriting the 944 one.

Refused on both counts. Nothing about this is a limitation of the helper-pair
extension point; it is what the numbers are.

---

## 5. Cost

**Protocol** — `benchmarks/era2_perf_break_2026-08-31.md` §22.5, unchanged: ONE
binary, arms at RUNTIME (`ZENSIM_BIGPAIR_TOGGLES`), byte-identical env blocks
(**every arm name is exactly three characters**, and `ZEN_S2_CAP_V3` is passed
to every arm in a sweep), arms INTERLEAVED inside each process start, min of 7
inner walks per start, min over 15 process starts with ASLR on, CCD0-pinned.
Four arms:

* `156` — the real 156-wide production walk.
* `15c` — CONTROL: 944 layout, v1-only compute, no free accumulators.
* `15f` — `15c` + the raw moments (the previous lane's free set).
* `15x` — `15f` + the class-C bounded-error tranche. **`15x`/`15f` is this
  lane's marginal cost; `15x`/`15c` is what the whole free set costs.**

Runner: `scripts/freefeats_ab.sh` (extended with `FF_ARMS` + `ZEN_S2_CAP_V3`
forwarding — the owner extended, not forked). Analysis:
`scripts/freefeats_ab_analyze.py` (extended: it reports the two new ratios when
`15x` rows are present). Raw + summary:
`/mnt/v/output/zensim/classc-2026-09-04/{native,capv3}/ab_{raw,summary}.tsv`
(720 rows per tier = 15 starts × 4 arms × 9 size/thread cells). Commit
`a8b24c8e`. Box load during both sweeps was ~3.3–3.7 (other lanes active);
arms are interleaved inside a start, so load enters every arm equally, and the
`min`-of-`min` estimator is one-directional against it — but see §5.3.

### 5.1 Native tier (AVX-512 ceiling)

| size | thr | `15x`/`15f` (class-C alone) | `15x`/`15c` (whole free set) |
|---:|---:|---|---|
| 576 | 1 | **1.0134 [1.0101, 1.0185]** | **1.0134 [1.0101, 1.0185]** |
| 576 | 8 | 1.0155 [0.9924, 1.0465] | 1.0315 [0.9924, 1.0472] |
| 576 | 16 | 1.0000 [0.9930, 1.0213] | 1.0000 [0.9860, 1.0213] |
| 1152 | 1 | **1.0130 [1.0049, 1.0168]** | **1.0118 [1.0107, 1.0156]** |
| 1152 | 8 | 1.0160 [0.9903, 1.0319] | 1.0276 [1.0109, 1.0313] |
| 1152 | 16 | 1.0058 [0.9962, 1.0115] | **1.0136 [1.0038, 1.0194]** |
| 2304 | 1 | **1.0146 [1.0072, 1.0198]** | **1.0190 [1.0056, 1.0242]** |
| 2304 | 8 | **1.0253 [1.0167, 1.0372]** | 1.0083 [0.9847, 1.0200] |
| 2304 | 16 | 1.0076 [0.9967, 1.0148] | **1.0076 [1.0067, 1.0148]** |

**Bold = the 95 % CI excludes 1.0.** At 1T — the cleanest signal, tightest CIs —
the class-C marginal is **+1.3 % to +1.5 %**, consistently positive at all three
sizes. At 8T/16T most CIs straddle 1.0.

### 5.2 v3 tier (AVX2 ceiling, `ZEN_S2_CAP_V3=1`)

| size | thr | `15x`/`15f` | `15x`/`15c` |
|---:|---:|---|---|
| 576 | 1 | **1.0199 [1.0184, 1.0230]** | **1.0262 [1.0231, 1.0278]** |
| 576 | 8 | **1.0290 [1.0000, 1.0507]** | 1.0216 [0.9930, 1.0432] |
| 576 | 16 | 1.0000 [0.9740, 1.0400] | 0.9740 [0.9677, 1.0130] |
| 1152 | 1 | **1.0234 [1.0223, 1.0252]** | **1.0311 [1.0278, 1.0329]** |
| 1152 | 8 | 1.0182 [0.9688, 1.0219] | **1.0276 [1.0109, 1.0313]** |
| 1152 | 16 | **1.0344 [1.0106, 1.0380]** | **1.0495 [1.0196, 1.0550]** |
| 2304 | 1 | **1.0201 [1.0199, 1.0264]** | **1.0246 [1.0230, 1.0300]** |
| 2304 | 8 | 1.0229 [0.9960, 1.0712] | 1.0109 [0.9863, 1.0596] |
| 2304 | 16 | **1.0231 [1.0103, 1.0326]** | **1.0203 [1.0089, 1.0298]** |

At 1T the class-C marginal is **+2.0 % to +2.3 %** — about 1.5× the AVX-512
figure, which is the expected direction: the tranche's two `sat` evaluations are
divides, and the v3 tier does them 8 lanes wide instead of 16.

### 5.3 The honest verdict on cost

**It is not free.** The brief this lane was given expected "~zero marginal";
the measurement says **+1.3–1.5 % (AVX-512) / +2.0–2.3 % (AVX2) of the 156
walk at 1T**, and the number is published rather than the expectation. That is
the same order as the raw-moments tranche's own honestly-priced +0.8–1.6 %, and
for comparison `benchmarks/feature_cost_frontier_2026-08-31.md` prices the next
family jump (basic-only → shipped B's masked+IW arm) at 1.3–1.6×, not 1.02×.

Two caveats a reader should carry:

* **8T/16T are not resolved here.** Most threaded CIs straddle 1.0, and 8T is
  this box's documented worst-conditioned thread count. The 1T column is the
  claim; the threaded columns are reported, not asserted.
* **`15f`/`15c` reads ~1.000 at 1T on the native tier in this run**, where the
  previous lane measured +0.8–1.6 % for the same comparison on a different day
  and a different binary layout. Cross-run comparison of a <10 % effect across
  two BUILDS is exactly what §22.5 says not to trust; within THIS run all four
  arms are one binary, so `15x`/`15f` and `15x`/`15c` are sound and
  `15f`/`15c` should be read from the previous lane's run, not this one.

---

## 6. Real-pixel route parity — and a finding about the PREVIOUS tranche

The gates in §3 run on synthetic textured images. The question a model owner
actually has is different: **if a model trains on stored 944-walk values and is
served from the cheap route, do the numbers agree on real corpus pixels?**

Instrument: `zensim/examples/v2_ab_extract` gained mode **`foldapp2fast`** (the
v1-only compute set at the 944 layout, peaks live, every free extra on) beside
the existing `foldapp2pools` (the full 944 walk that built the stored root).
Both were run over the same **773 real pairs** — a 1-in-9 stride of
`r1b-pools944-2026-08-30/pairs/pairs_imazen26_png.tsv`, real zenavif/zenwebp/
zenjpeg codec output at real corpus geometries from `64×48` to `1024×1024`.
Artifacts: `/mnt/v/output/zensim/classc-2026-09-04/routeparity/`.

### 6.1 The class-C 24: parity holds on real pixels

| set | cells | over the 2e-5 bar | worst \|Δ\| | worst relative |
|---|--:|--:|--:|--:|
| **class-C (24)** | 18,552 | **0 (0.00 %)** | **9.81e-8** | **2.45e-6** |

All 18,552 cells non-zero on the 944 side, so the comparison is not vacuous.

**Free throughput datapoint from the same run:** the fast route extracted these
773 real pairs at **7.7 ms/pair** against the full 944 pools walk's
**18.8 ms/pair** — **2.44×** — for a vector carrying 289 usable slots
(156 basic + 72 peaks + 40 raw-moment + 24 class-C). That is a wall-clock ratio
from an extraction run, not a §5-protocol measurement; it is quoted as an
order-of-magnitude, not a benchmark.

### 6.2 The free-40: 9.12 % of its cells miss its own bar on real pixels

Measured in the same two files, so it is the same comparison the previous
lane's `free_extras_match_the_944_append_block` makes — just on real corpus
pixels instead of synthetic ones:

| set | cells | over the 2e-5 bar | worst \|Δ\| |
|---|--:|--:|--:|
| free-40 (raw moments) | 28,601 | **2,607 (9.12 %)** | **3.63e-3** |

By family: **`GLOBAL_CLOSS` 1,467 · `GLOBAL_CGAIN` 1,132 · `GLOBAL_DMEAN` 8 ·
`LUMA_MEAN_REF` 0.** Worst relative errors reach **~55×** where the true value
is ~1e-9. Worst \|Δ\| in the largest-25 %-by-pixel-count bucket is 3.63e-3 vs
1.60e-3 in the smallest — it grows with plane size.

The mechanism is consistent with exactly which slots diverge:
`global_stats_from_raw_moments` computes `gvar = Σs²/n − (Σs/n)²` and
`|Σs − Σd|/n`, both **catastrophic-cancellation** forms. The two routes stage
f32→f64 differently (the append kernel adds a whole row into f32 lanes and
reduces per row; the fused kernel accumulates 32 rows of f32 lanes and reduces
per band), so the cancellation error differs — and on a near-identical pair the
true value is at the f32 accumulation floor, which is where a 55× relative
error comes from. `LUMA_MEAN_REF`, the one slot of the 40 that is a plain mean
with no cancellation, has **zero** cells over the bar. The class-C values are
bounded means in [0, 2] with no cancellation, which is why they hold at 1e-7.

**This is a finding about the neighbouring lane's tranche, not this one's, and
it is reported rather than fixed** — the fix (an f64 or compensated
accumulation for `Σs²`, or a Welford-style form) is a decision for the slot
family's owner. What a reader must take from it: **a fast-class model that
reads `GLOBAL_CGAIN`/`GLOBAL_CLOSS` is trained on one route's values and served
another's, with disagreement up to 3.6e-3 absolute.** The 24 class-C slots and
`LUMA_MEAN_REF` do not have that problem; the peaks and basic block are
BIT-identical between the two routes (worst \|Δ\| exactly 0.0 over
773 × 228 cells).

### 6.3 A v1-only 944 walk does NOT leave every unreached slot at zero

Also measured in the same files, and pre-existing (it is identical on the
`RawMoments` route): **all twelve `PJND_FRAGILITY` slots** — `f393 422 451 /
480 509 538 / 567 596 625 / 654 683 712` — read a **constant 1.0** on 773/773
rows. `finish_channel_scale` produces that from zeroed accumulators; it is a
formula artifact, not a value. No other slot outside {basic, peaks, free-40,
class-C} is non-zero.

**Consequence for a training lane: slice to the free set explicitly.** A model
built from "whatever columns are non-zero" gets twelve constant-1.0 columns
that carry no information and do not exist in the stored 944 tables.
`scripts/sota944/slice_basic156_free_classc.txt` (289 coordinates) is the
slice that is correct.

---

## 7. Data — no extraction wave was needed, and that is a measured result

The lane brief expected a new dated root. **It is not needed, and building one
would have been waste.** The class-C slots are existing 944 positions, so any
genuinely non-folded 944 root already contains them, produced by the full walk.

MEASURED on `/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/`
(regime `folded720append2pools`, 14 corpora, 170,007 rows,
`build_commit ced6f52a`) — the same root the free-features lane's own quality
probe used: reading the 24 class-C columns of `ext_cid22val.parquet` gives
**4,292 / 4,292 non-zero on every one of the 24**, with sane ranges
(e.g. `f377` v2 `MSE` s0 c0: 2.62e-4 … 7.57e-2, mean 8.37e-3).

So the distillation lane can train on this slice **today**, on the existing
root, with no fleet wave and no new regime — and §6.1 is the evidence that what
it trains on is what the cheap route will serve. Regime purity is unchanged and
still binding: the `ext944-canonical` / `*_pure` roots zero `f156-371` and would
silently zero the peaks half of the free set; only a `folded720append2pools`
(or wider, genuinely non-folded) root is valid for this slice.

**Hand-off:** `scripts/sota944/slice_basic156_free_classc.txt` — 289
coordinates = `slice_basic156_free.txt`'s 265 plus the 24 class-C. Point any
`bake_dial_refit fit-lasso --slice-file` or `zensim_mlp_train --keep-features`
at it. The 156-only and 156+free slices are both still there and still correct
for a model that should not use the new set.

---

## 8. Reproduce

```sh
# gates
cargo test -p zensim --lib --release \
  --features custom-profiles,feature-regime-v2,threads,training -- --nocapture class_c

# cost, both tiers (each ~10 min on this box)
FF_ARMS="156 15c 15f 15x" scripts/freefeats_ab.sh /mnt/v/output/zensim/classc-2026-09-04/native
ZEN_S2_CAP_V3=1 FF_ARMS="156 15c 15f 15x" \
  scripts/freefeats_ab.sh /mnt/v/output/zensim/classc-2026-09-04/capv3
python3 scripts/freefeats_ab_analyze.py \
  /mnt/v/output/zensim/classc-2026-09-04/native/ab_raw.tsv \
  /mnt/v/output/zensim/classc-2026-09-04/native/ab_summary.tsv

# real-pixel route parity (773 pairs, both routes, ~30 s)
cargo build --release -p zensim \
  --features custom-profiles,feature-regime-v2,threads,training --example v2_ab_extract
for m in foldapp2pools foldapp2fast; do
  ZENSIM_AB_MODE=$m target/release/examples/v2_ab_extract \
    /mnt/v/output/zensim/classc-2026-09-04/routeparity/pairs_sub.tsv /tmp_out_$m.csv
done
```
