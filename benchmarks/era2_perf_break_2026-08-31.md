# era-2: the batched performance break — DESIGN

**Status: DESIGN, committed before code** (per the lane instruction). User
decision 2026-08-30, verbatim rationale: *"the point of zensim is to be
extremely fast, and as good or better than ssim, and be good at HDR."*

**Governing principle: ONE era, batched.** Everything byte-affecting that we
have measured and want ships in this single break. Never piecemeal — the
CHANGELOG QUEUED-BREAKING-CHANGES discipline, applied to feature bytes. A
second byte-moving change after this one costs a second re-extraction of every
944 table and a second retrain of every 944 model; that is the cost this doc
exists to make payable exactly once.

**Not in scope, deliberately:** layout or slot renumbering. `f0..f943` keep
their meanings and positions; the append-only discipline holds. This is an
**era** step (same questions, better-computed answers), not a **regime**
relayout. HDR features are not designed here — the break is only made
HDR-route-clean so a later HDR append lands on era-2 as its base.

---

## 1. What era-1 does today, and why it blocks perf

> **CORRECTION (2026-08-31, from the v2-block lane's wall-clock decomposition
> — `benchmarks/v2_block_cost_2026-08-31.md` §7).** The 23.2 % below is a
> **callgrind instruction count on the `v3` tier**, and callgrind masks
> AVX-512 out of CPUID, so it can never execute the shipping path: `POOL_SIMD`
> is `v4x`-only, and the Ir profile therefore prices the *scalar-pool* form of
> this kernel. Measured by wall clock on the shipping tier, `dense_block_kernel`
> is **13.5 % of the v2 block and 7.3 % of the 944 walk** at 2304², and every
> figure derived from the 23.2 % — including the **1.17× @8T / 1.23× @16T**
> Amdahl bound two paragraphs down — is scoped to `v3`, not to what ships.
> Keep reading §1 for the *structural* argument (why the kernel cannot be split
> bit-exactly), which is unaffected; discard the *magnitudes*. §22 is where
> that correction sent this lane next.

`dense_block_kernel_generic` (`feature_v2.rs:2190`) is **23.2 % of the 944-full
walk** (125,098,677 of 540,211,635 net Ir at 576², `v3` tier — see the
correction above) and gets 3-way channel parallelism only. It cannot be split further without changing bytes, because
two paths accumulate into the f64 accumulator **per pixel, across row
boundaries**:

1. **`weighted_pool_accumulate_scalar` inside the vector x-loop** (`:2352`) —
   the §A.14 register-pressure fix scalarises the 11 masked/IW/soft-peak
   weighted pools and adds them per lane, per pixel. This is the
   `POOL_SIMD == false` path: **every tier except `v4x`**.
2. **The `for x in width8..width` scalar tail** (`:2416`) — adds `sum_d`,
   `sum_d2`, … per pixel whenever `width % 8 != 0`.

Measured consequence (era-1 doc §14.1): a row-partial merge diverges by **−2
ulps** with a scalar tail and **13 ulps** with per-pixel pools; it is exactly 0
only when `POOL_SIMD` is on *and* `width % 8 == 0`. `POOL_SIMD` is `v4x`-only
by design, and 200×150 walks widths 200/100/50/25 so the tail fires at three of
four scales. There is no general bit-exact split.

The Amdahl upper bound on fixing it was **1.17× @8T / 1.23× @16T** *while
holding bytes*. The break removes that constraint, and §5 re-opens the other
levers that were rejected for byte stability alone — so the batched gain is the
point, not the dense change on its own.

---

## 2. The era-2 accumulation shape — FIXED VIRTUAL LANES

**User refinement (2026-08-30), adopted: the canonical semantics are the
SIMD-shaped accumulation, and the scalar path is made to MATCH SIMD — never
the reverse. No compensated or otherwise slower accumulator is ever forced
into a product path; the compensated/exact oracle (§11) is the RULER, never
the semantics.**

### 2.0 The definition

era-2 accumulation is defined at a **fixed virtual lane grouping of 8 `f64`
lanes**, independent of the physical tier:

```
for each accumulator:  lane[0..8] : f64            // NOT f32
term at pixel x  ->  lane[x mod 8] += term         // terms in increasing x
reduce:  acc = ((((((lane0+lane1)+lane2)+lane3)+lane4)+lane5)+lane6)+lane7
```

Three deliberate choices, each load-bearing:

* **8 lanes, not 16.** 8 `f64` is exactly one `f64x8` on AVX-512, two `f64x4`
  on AVX2/NEON, and 8 scalars on the fallback — every tier maps onto it without
  remainder. Choosing 16 would strand every non-AVX-512 tier.
* **`f64` accumulators, not `f32`.** This is the accuracy change that makes the
  break worth taking on quality grounds as well as speed: era-1 accumulates in
  `f32` lanes, whose ~`4.3e-6` relative error **dominates the entire error
  budget** (§10.3). Moving the accumulator to `f64` removes that term
  outright. Term *evaluation* stays `f32` (SIMD-shaped, per the refinement) —
  so the residual error is the per-term evaluation rounding, not the summation.
* **The reduction tree is part of the semantics**, written above explicitly, so
  no tier may choose its own.

**The scalar tail disappears as a special case.** Tail pixels `x ∈ [width8,
width)` land in `lane[x mod 8]` like every other pixel. There is no separate
tail accumulation path, which is what made era-1's grouping width-dependent.

### 2.1 How each tier realises it

| tier | physical | realisation |
|---|---|---|
| `v4x` (AVX-512) | `f64x8` | consumes the virtual lanes natively, 1:1 |
| `v4`/`v3` (AVX2/SSE) | `f64x4` ×2 | the same 8 partial sums, two registers |
| `neon`/`wasm128` | `f64x2` ×4 | the same 8 partial sums, four registers |
| `scalar` | `[f64; 8]` | an array of 8 running sums mirroring the virtual lanes — **which LLVM will likely auto-vectorise anyway; that is to be MEASURED, not assumed** |

An `f32x16` producer (v4x term evaluation) widens to two `f64x8` halves, both
added into the same 8 lanes — pixel `x` still lands in `lane[x mod 8]`.

### 2.2 Theorem (cross-tier bit-identity, by construction)

> **Claim.** Every tier produces bit-identical accumulator values.

*Proof.* All tiers (i) evaluate the same per-pixel formula on the same `f32`
inputs with the same operations — lane width does not change per-element
arithmetic; (ii) widen `f32→f64`, which is exact; (iii) add each term into
`lane[x mod 8]` in increasing `x`, so each lane sees the same terms in the same
order; and (iv) reduce by the same fixed tree. Every operation is IEEE-754
`+`, `−`, `*`, `/` or `max`, all correctly rounded and therefore
architecture-independent. ∎

### 2.3 What remains architecture-dependent — enumerated, then neutralised

This is the historical killer: v1's exact golden never held cross-vendor
(241–246 of 372 features diverged on every non-AMD class). Under fixed-lane
`f64` accumulation the enumeration is short, and it was checked in source
rather than assumed:

| source | status in the dense path |
|---|---|
| **FMA contraction** (`a*b+c` fused vs separate) | **CHECKED: zero `mul_add` in the dense kernel body.** Rust does not auto-contract without fast-math, so `a*b+c` is mul-then-add on every target. To be **gated**, not merely observed |
| **`rsqrt` / reciprocal estimates** (vendor-specific seed tables) | **already neutralised and documented in-tree**: the MSCN normalizer deliberately uses IEEE `sqrt` + `div` because "`rsqrt()` is the hardware estimate whose seed table is CPU-VENDOR-specific". Nothing in the dense formulas uses it |
| **Transcendentals** (`powf`, `exp`, `ln` — libm-dependent) | **none in the dense formulas**; they use only `+ − * / .max()` |
| **`f32→f64` widening** | exact by IEEE-754, no rounding |
| **`.max()` NaN semantics** | operands are non-NaN by construction here; a NaN would be a bug the oracle catches |
| **Horizontal reduce order** | fixed by §2.0, not left to the tier |

**So cross-ARCH bit-identity is a plausible consequence, not just cross-tier**,
and §4 gate 5 states what can and cannot be verified directly from this box.

### 2.4 Band parallelism on top

Unchanged from the earlier design and orthogonal to the lanes: each band keeps
its own 8-lane set, bands merge **lane-wise in band order**, then the fixed
tree reduces once. Band layout is a pure function of `(height, BAND)`, so §10.6
still holds.

---

## 2bis. The superseded shape (kept for the record)

**Invariant (the whole design in one line): every accumulator receives exactly
one add per BAND, and bands are a pure function of geometry.**

```
BAND = fixed compile-time constant (rows), independent of thread count
for each band b, in order:
    band_partial[b] = 0
    for each row y in band b, in order:
        row_local = 0                      // all ~35 fields
        row_local += <vector-lane reduction over width8 columns>
        row_local += <scalar tail terms, x in width8..width>   // FOLDED IN
        band_partial[b] += row_local
acc = fold(band_partial[0..n], in band order)
```

Three properties follow **by construction**, not by measurement:

* **Thread-count invariance.** The band layout depends only on `(height,
  BAND)`. Whether one thread or sixteen compute the bands, the merge is the
  same sequence of f64 additions in the same order. This retires the §3.27
  class of defect (the era-1 372 masked/IW block was a function of
  `RAYON_NUM_THREADS`) *structurally* rather than by testing for it.
* **Band-parallelism is exact against era-2's own reference.** The serial and
  parallel paths run identical code and differ only in which thread evaluates
  `band_partial[b]`; the fold is sequential in both.
* **Determinism.** No reduction order anywhere depends on scheduling.

### 2.1 The register-pressure constraint, and how it is solved

The reason the pools were scalarised in the first place stands: 11 weighted
pools = 22 lane accumulators in the original form, and 13 core + 16 pool +
constants does not fit the 16 vector registers of `v4`/`v3`/`neon`/`wasm128`.
`POOL_SIMD` (2026-07-21) got the pool block to 16 lanes by sharing one `Σw`
per family, and was still gated to `v4x` (32 registers) only.

**Design: two passes per row, with a row-local intermediate buffer** — the
user's earlier steer that multiple passes per row can beat one fused pass when
spills are the binding constraint.

| pass | live lane accumulators | reads | writes |
|---|---|---|---|
| **A** | ~13 core (`sum_d`, `sum_d2/3/4`, `art`, `det`, `mse`, `hf_*`, `pjnd_*`) | `src`, `dst`, `mu1`, `mu2`, `ssq`, `s12`, `activity` | 5 × row scratch (`d`, `art_i`, `det_i`, `mse_i`, `act`) |
| **B** | 16 pool (`POOL_SIMD` shape, now on every tier) | the 5 row scratches | — |

Row scratch is `5 × width × 4 B` = 11.5 KB at width 576, 23 KB at 1152 —
L1/L2-resident, allocated once per (strip, channel) and reused.

**The alternative — recompute in pass B instead of storing — is the A/B this
lane must measure**, not assume: it trades 5 stores + 5 loads per pixel for
re-deriving `d`/`art_i`/`det_i`/`mse_i` (division-heavy). Both are era-2-legal;
whichever wins ships. Neither is assumed here.

**The scalar tail folds into the row-local accumulators** in both passes: the
tail's terms are added into `row_local` (not into `acc`), so the "one add per
band" invariant holds at every width, `width % 8 != 0` included. This is the
part that makes the shape width-independent, and it is why era-2 is a real fix
rather than a `v4x`-only one.

### 2.2 What this does NOT change

Per-pixel *formulas* are untouched — era-2 sums **the same terms** as era-1, in
a different grouping. That is why §4's equivalence gate expects a small,
measurable drift rather than a different answer, and why the rank-preservation
gate expects ≈0.

---

## 3. The re-opened pile (rejected for byte stability ALONE)

Each still has to **win on measurement**. The break pays the era cost; it does
not pay a perf-regression cost.

| lever | why it was rejected | era-2 status |
|---|---|---|
| **art-L4 weighted sums inside the fused V-blur** (lane-2's "unshippable bit-exact") | `fused_vblur_ssim_inner` iterates column-group-major and folds into f64 in that order; `simd_ops::{edge_diff_channel_inline_both, ssim_channel_inline_both, build_inline_mse}` iterate row-major. Moving the math changes the f64 summation order. Everything the fused loop already computes (`ed`, `a4`, `dl4`) is therefore recomputed on purpose | **RE-OPENED.** The recompute is **+9,589,905 Ir (12.6 % of the pool gap)** for `edge_diff_channel_inline_both` alone. Must win or it does not ship |
| **Inner-rows-only V-blur write** in `box_blur_1pass_into` | not a byte issue — rejected on value (~0.4 % of the walk) | stays rejected; re-measure only if it becomes free alongside the above |
| activity-fusion (`box_blur_h_of_abs_diff`) | not a byte issue — measured **+1.04 % / +2.01 %**, a real loss | **stays rejected.** The break does not resurrect measured losses |
| row-parallel H-blur | not a byte issue — measured neutral | **stays rejected** |
| `fused_vblur_ssim` fission | not a byte issue — premise void (1 spill load in the innermost loop) | **stays retired** |

Only the first row is a genuine byte-stability rejection. The honest read of
this table is that the "rejected for byte stability" pile was **one item deep**,
and the break's value is therefore concentrated in §2 (dense becoming
band-parallelisable) plus whatever the MT program then buys.

---

## 4. Era-2 gates

**Backed by the proof chain in §10 (error analysis) and the scalar oracle in
§11.** No era-2 kernel ships without both its oracle comparison and its proven
bound written down here. Any measurement exceeding its proven bound is a
**BUG, and the lane stops** — a bound violation is never a reason to widen a
tolerance.

**Five** new gates, plus the existing suite. Gate 5 (cross-tier / cross-arch
bit-identity) is new with the virtual-lane refinement — see §12.4. Each is stated as what it proves, not
as a number to be tuned.

1. **Same-binary determinism, bit-exact.** Two computes of the same pair on one
   machine agree on all 944 slots. (Extends `v1_same_class_determinism_bitexact`
   to the fold.)
2. **Thread-count invariance, BY CONSTRUCTION.** Assert `RAYON_NUM_THREADS`
   1 / 2 / 3 / 8 / 16 produce bit-identical vectors. This is a *check* of a
   structural property (§2), not the mechanism that provides it.
3. **Numerical equivalence to era-1, within a MEASURED and DECLARED
   tolerance.** era-2 sums the same terms in a different grouping, so the drift
   is real and bounded but must be *measured*, never hand-waved as "should be
   ulps". Method: extract the same pair set under era-1 and era-2, report
   per-feature max abs and max rel drift and the distribution, then declare the
   tolerance from the measurement and gate on it. **A drift larger than the
   declared bound at any slot is a bug in the reshape, not a tolerance to
   widen.**
4. **Rank preservation.** Shipped B plus 2–3 roster bakes scored on era-1 vs
   era-2 features of the same pairs; SROCC deltas reported per corpus. Expect
   ≈0 — the era-1→era-3 (C) precedent moved cid22 by +0.000024 — but **verify,
   do not assume**. A material rank move means the reshape changed more than
   the grouping.

Existing gates that must stay green unchanged: `folded720_v1_pools_match_v1_path`,
`folded720_v1_basic_matches_v1_path`, `v1_372_bit_exact_to_fold_at_every_width`,
`streamed_parallel_matches_serial`, `folded_v1_only_matches_full_walk`.

**Note on the v1 gates:** they compare the fold to the buffered path. Both walks
consume the same kernels, so era-2 moves them *identically* and the relative
gates stay valid. The **absolute** gates (`v1_golden_bytes`) do move, and are
re-pinned as part of the era with the negative-control discipline used for
`GOLDEN_NONTIGHT` — including the non-tight fixture, which must stay in the set.

---

## 5. The era stamp — old and new extractions must never silently mix

Per `DATA_PROVENANCE` conventions:

* **`_MANIFEST.json` gains an `era` field** (`"era": "2"`) plus the
  `build_commit` that already exists. A manifest without an `era` field is
  era-1 by definition; that default is written down so absence is never
  ambiguous.
* **Fleet metric naming: a REGISTERED zenmetrics change, not an edit by this
  lane.** Another lane may hold zenmetrics, and the fleet metric names/defaults
  are a standing scope fence. The proposal to register: an explicit suffix on
  the folded metric names (e.g. `zensim-foldapp2pools-e2`) so a mixed harvest
  is impossible by construction rather than by convention. **This lane does not
  edit zenmetrics**; it hands the naming over as a registered request.
* **Annotations registry**: one `era-1-dense-grouping` entry marking every
  944-era table as prior-era, with the same *predicate* form used for
  `v1-372-era2-phantom-column-pooling` — except that here the predicate is
  unconditional (all widths, all tiers), which is simpler and should be said
  plainly.

---

## 6. Blast radius and follow-on waves — REGISTERED, not launched

**This lane registers this plan with priorities. It does not launch any wave.**

### 6.1 Re-extract (byte-affected)

| # | artifact | scale | priority |
|---|---|---|---|
| R1 | 11 canonical `ext924/944` legs (`ext924-canonical-2026-07-27`) | 149,195 rows | **P0** — the training substrate |
| R2 | eval instruments (`corruption_grid_924col`, `dial_grid_924col`) | 2,016 + 4,817 | **P0** — gates depend on them |
| R3 | era-3 372 eval root → era-2 (the 8 re-extractable corpora + kon504) | ~42 k rows | **P0** — ~1 min, already scripted |
| R4 | bigcodec `tbig_924_full` + 21 split views | 5,742,660 rows | **P1** — the big one; fleet |
| R5 | `kadis700k_924` + `kadis_negrich_924` | 867,033 rows | **P1** — fleet |
| R6 | svt/aom harvest features | in flight | **P1** — coordinate with the fleet lane; do not orphan a running harvest |
| R7 | the registered ~227 k B re-extraction | 227 k | folds in here — do it once, on era-2 |

### 6.2 Retrain (semantics-affected)

The 944 roster: inputs change meaning, not merely value, so every 944-trained
model is prior-era. **P0 for the ship candidates** (shipped B's 944 successors,
the C-class roster); **P2 for the historical board**, which should be annotated
rather than rebuilt.

### 6.3 Sequencing

R3 first (minutes, validates the pipeline end-to-end), then R1+R2 (hours,
unblocks gates), then R4/R5 on the fleet, then retrains. R6 is a coordination
item, not a scheduling one.

---

## 7. HDR-route cleanliness

The `hdr944` route shares these kernels — it differs only in the front end
(`FrontEnd::Hdr` → PU-XYB) before the identical streaming walk. So:

* **HDR inherits era-2 automatically**, and its tables take the same era step:
  the HDR canonical legs, `hdrgrid`, and any BHdr-facing eval slice are in the
  §6.1 blast radius and are enumerated there by inclusion.
* **The reshape must not special-case SDR.** The row scratch and two-pass shape
  live below the front-end split, so there is nothing HDR-specific to design —
  the requirement is only that no part of §2 keys on `FrontEnd`.
* **A future HDR-feature append lands ON era-2 as its base**, at `f944+`,
  append-only. Nothing here reserves or renumbers slots for it.

---

## 8. Sequencing with the fold-engine lane

That lane (score() / ref-cache / attribution / oracle) is running in parallel on
era-1 semantics. Its parity gates are **relative** — fold vs buffered — and
era-2 moves both walks identically, so **those gates stay valid under the
break**. What does move is any **absolute** pin: `v1_golden_bytes` and any
stored-vector fixture.

**Proposed protocol:** land era-2 kernels *after* the fold-engine lane pins its
parity stages, so its gates are written against a stable reference; then re-pin
the absolute goldens once, as part of the era commit. If that sequencing binds,
the alternative is to rebase its gates together with the era in one commit —
explicitly, never silently. Reported up rather than decided here.

---

## 9. Stage plan

| stage | content | gate |
|---|---|---|
| **S0** | this design, committed | — |
| **S0b** | the **scalar oracle** (§11) — L1 Neumaier reference + L2 exact accumulator, tier × geometry harness — landed BEFORE any kernel change, so era-2 is judged against exactness from its first commit | L1-vs-L2 self-check; era-1 kernels measured against it to establish the **pre-break baseline** deviation |
| **S1** | pass-A/pass-B reshape + folded tails, `POOL_SIMD` shape on every tier, serial only | era-1 suite green except the absolute goldens; §4 gates 1 + 3 measured; **§10.5 per-family bounds filled in with real numbers and every oracle measurement under its bound**; §10.7 (3) evaluated with B's actual weights |
| **S2** | band-parallel dense on the new shape | §4 gate 2; 944-full × 1/8/16T re-measured — **what is the new ceiling?** |
| **S3** | the re-opened art-L4 lever (§3), ship only if it wins | paired bench |
| **S4** | era stamp + gate re-pins + annotations + §6 registered plan | §4 gate 4 (rank preservation) |

Each stage is a verified commit. **S1 is where the byte change happens**, so
the equivalence measurement (§4.3) is reported there, before any of the perf
work that depends on it.

---

## 10. The error analysis — theory, proof, and the bounds each kernel must meet

Notation: `u₆₄ = 2⁻⁵³ ≈ 1.11e-16`, `u₃₂ = 2⁻²⁴ ≈ 5.96e-8`. For a sum of `n`
terms, `γₖ = ku/(1−ku)`, so `γₖ ≈ ku` to first order.

### 10.1 The standard model

**Recursive (sequential) summation** (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., Thm 4.1 / eq. 4.3): the computed sum `Ŝₙ` of
`x₁…xₙ` satisfies

```
|Sₙ − Ŝₙ|  ≤  γ_{n−1} · Σᵢ|xᵢ|   ≈  (n−1)·u·Σᵢ|xᵢ|        … (1)
```

**Blocked summation** into `b` blocks of size `m = n/b`, each block summed
recursively and the block totals then summed recursively:

```
|Sₙ − Ŝₙ|  ≤  γ_{m−1}·Σᵢ|xᵢ| + γ_{b−1}·Σᵢ|xᵢ|
           ≈  (m + b − 2)·u·Σᵢ|xᵢ|                        … (2)
```

**SIMD lane accumulation is interleaved blocked summation.** `L` lanes each
accumulate `n/L` terms (stride `L`), then the lanes are reduced. That is
exactly (2) with `m = n/L`, `b = L` — the interleaving changes *which* terms
land in which block, not the bound's shape.

### 10.2 Theorem (era-2 is no worse than era-1, worst case)

> **Claim.** For the same `n` terms, the era-2 grouping's worst-case bound is
> ≤ era-1's, with equality only at the degenerate block counts.

*Proof.* Define `f(b) = n/b + b − 2`, the coefficient in (2). `f` is convex on
`b ∈ [1, n]` with `f′(b) = 1 − n/b²`, so its minimum is at `b = √n`, where
`f = 2√n − 2`. At the endpoints `f(1) = f(n) = n − 1`, which is exactly the
sequential coefficient in (1). Hence for all `b ∈ [1, n]`,

```
f(b) = n/b + b − 2  ≤  n − 1
```

with equality iff `b ∈ {1, n}`. era-1 aggregates at `b = 1` (one running total,
every add landing in it); era-2 aggregates at `b = n_bands` with
`1 < n_bands < n`. Therefore era-2's coefficient is strictly smaller. ∎

**Verified numerically before this doc was committed** (the claim is small
enough to check exhaustively, so it was): `f(b) ≤ n−1` holds for every integer
`b ∈ [1, min(n, 4000)]` at `n ∈ {128, 576, 73728, 1000000}` — **0 violations** —
with the minimum landing at `√n` as derived (`n=73728`: `f(1)=f(n)=73727`,
`f(√n)=541`).

And the instance-wise caveat below was checked too, not merely asserted: over
six trials of the real shape (128 row-reduces + 512 tail terms, error measured
against an **exact rational** sum), blocked `b=4` beat sequential in **4 of 6**
— e.g. `2.71e-11 → 2.02e-12` on the best trial, `6.13e-12 → 2.07e-11` on the
worst. The bound is a worst-case statement; individual pairs can move either
way, which is exactly why §11's oracle measures rather than infers.

Two honesty notes that belong with the proof:

* This compares **worst-case bounds**, not instance-wise error. A specific pair
  can drift either way; the claim is that era-2 cannot be worse *in the bound*,
  and is typically tighter.
* The bound is on the distance to the **exact** sum. That is the right target —
  §11's oracle measures against exactly that, not against era-1.

### 10.3 Where the error actually lives (the term that dominates)

The f32 lane accumulation is **unchanged** between eras — same lanes, same
chunk count, same order. Only the f64 aggregation layer changes. Their relative
magnitudes at 576², scale 0, per row (`width = 576`, `L = 8`, chunks/lane
`= 72`; strip `height = 128`):

| layer | terms | unit roundoff | ≈ relative bound |
|---|---:|---:|---:|
| f32 lane accumulation (unchanged) | 72 per lane | `u₃₂` | `72 · 5.96e-8` ≈ **4.3e-6** |
| f64 aggregation, era-1 (`b = 1`) | 128 rows | `u₆₄` | `127 · 1.11e-16` ≈ **1.4e-14** |
| f64 aggregation, era-2 (`b = 4` bands of 32) | 32 + 4 | `u₆₄` | `34 · 1.11e-16` ≈ **3.8e-15** |

**The layer era-2 changes is ~10⁸ times smaller than the layer it does not
touch.** For the *core* accumulators, this is why the expected drift is far
below anything a score can see — and it is a statement about where the error
budget sits, not a hope.

### 10.4 The one family that genuinely changes accuracy class — stated plainly

The **weighted pools** (11 masked/IW/soft-peak pairs) are the exception, and it
must not be buried. On non-`v4x` tiers era-1 accumulates them **per pixel in
f64**; era-2 accumulates them **in f32 lanes, reduced per row** (the
`POOL_SIMD` shape). That is a *coarser* accumulation, and for those slots
era-2 is expected to be **less** accurate than era-1, not more:

```
era-1 pools (non-v4x): N = height·width = 73,728 f64 adds  →  ≈ 8.2e-12 rel
era-2 pools:           f32 lanes, 72 chunks/lane           →  ≈ 4.3e-6  rel
```

**This is not new territory, and that is the argument for accepting it.**
`POOL_SIMD` has shipped on `v4x` since 2026-07-21 with exactly these semantics,
a documented **5e-4** module tolerance, and the `pool_simd_drift_within_policy`
gate. era-2 therefore does not invent a numeric class — **it makes every tier
compute what production `v4x` already computes**, which also removes a real
cross-tier divergence that exists today. The bound to meet for pool slots is
the existing 5e-4 policy, re-measured under §11 rather than inherited on trust.

### 10.5 Per-family bounds — form now, numbers at S1

The absolute bound per slot is `(coefficient) · u · Σᵢ|xᵢ|`, so it needs the
empirical `Σ|x|` per slot family. Those come from the golden corpora at S1 and
get filled in here:

| family | n per accumulator | dominant layer | bound form | measured `Σ|x|` | abs bound | ulp bound |
|---|---|---|---|---|---|---|
| core sums (`sum_d`, `sum_d2/3/4`, `art`, `det`, `mse`) | `height·width` | f32 lanes | `72·u₃₂·Σ|x|` | *S1* | *S1* | *S1* |
| `hf_*` | `height·width` | f32 lanes | same | *S1* | *S1* | *S1* |
| `pjnd_*` | `height·width` | f32 lanes | same | *S1* | *S1* | *S1* |
| weighted pools (num + den) | `height·width` | **f32 lanes (changed class, §10.4)** | `72·u₃₂·Σ|x|`, policy 5e-4 | *S1* | *S1* | *S1* |
| band/row f64 aggregation | `n_bands` | f64 | `(m+b−2)·u₆₄·Σ|x|` | *S1* | *S1* | *S1* |

**The §11 oracle measurements must land under these bounds. If any slot
exceeds its bound, the analysis is wrong or the kernel is — either way the lane
STOPS and finds out which.** The doc will say which, explicitly, rather than
adjusting the table.

### 10.6 Theorem (determinism and thread-invariance, by construction)

> **Claim.** The era-2 accumulator is a pure function of (input planes, height,
> `BAND`) — independent of thread count, band→thread assignment, and completion
> order.

*Proof.* Let `B(h) = {b₀ … b_{k−1}}` be the band partition, determined solely by
`h` and the compile-time constant `BAND` (§2). (i) Each band partial `P_b` is
computed from input planes and `b` alone: bands read overlapping *inputs* but
write disjoint *partials*, and no `P_b` reads another. (ii) The merge is the
fixed sequence `acc = ((P₀ + P₁) + P₂) + …`, indexed by band, evaluated
sequentially. Composition of (i) and (ii) contains no term whose value or
position depends on scheduling. ∎

**Corollary.** Invariance holds for *all* thread counts, not merely the tested
ones — the test is a check on the structure, not the source of the property.
This retires the §3.27 defect class (era-1's 372 masked/IW block was a function
of `RAYON_NUM_THREADS`) structurally.

*Enforcing test:* `era2_thread_invariance_bitexact` (`RAYON_NUM_THREADS` ∈
{1, 2, 3, 8, 16}).

### 10.7 Score-level propagation

**Linear class (shipped B), exactly.** With standardisation `z_j = (f_j − μ_j)/σ_j`,
weights `w_j`, and the monotone output spline `g`:

```
|Δscore|  ≤  Lg · Σⱼ |wⱼ/σⱼ| · |Δfⱼ|                        … (3)
```

where `Lg = max|g′|` on the operating range. Both `w`/`σ` (from the bake) and
`Lg` (from the spline knots) are known constants, so (3) is computable exactly
— no estimation. It gets evaluated with shipped B's actual weights at S1.

**MLP class (the C roster), via Lipschitz composition.** For layers
`W₁…W_L` with 1-Lipschitz activations (ReLU/GELU both are):

```
|Δscore|  ≤  Lg · (Πᵢ ‖Wᵢ‖₂) · ‖Δf ⊘ σ‖₂                    … (4)
```

`‖Wᵢ‖₂` is the spectral norm (largest singular value), computed from the bake.
(4) is loose — it is an upper bound, not a prediction — which is the right side
to be loose on.

**The acceptance argument.** Both bounds get compared to the **dial's
materiality step**: the score is `[0,100]` and the campaign's materiality
threshold is 0.5 score points. The claim to be demonstrated at S1 is
`bound ≪ 0.5`, by orders of magnitude. **Then confirmed empirically**, because a
bound is not a measurement: same pairs, era-1 vs era-2, for shipped B + 2–3
roster bakes — reporting `max |Δscore|` and SROCC deltas on every eval corpus,
against the **pre-declared** pass thresholds

* `max |Δscore| ≤ 0.05` score points (10× below materiality), and
* `|ΔSROCC| ≤ 0.001` per corpus (the era-1→era-3 precedent moved cid22 by
  +0.000024, so this is generous by ~40×).

Exceeding either is a STOP, not a re-declaration.

---

## 11. The scalar oracle — the standing correctness instrument

**Test-only. Never in the product build.** Gated `#[cfg(any(test, feature =
"oracle"))]`; the `oracle` feature exists so benches and fuzz targets outside
`#[cfg(test)]` can reach it, and it is not in `default`.

### 11.1 Two levels, because "correct" must not itself be an approximation

| level | implementation | role |
|---|---|---|
| **L1 — reference** | plain scalar loops, unambiguous left-to-right order, f64 with **Neumaier** compensated summation (Kahan–Babuška–Neumaier, which unlike plain Kahan is correct when the running total is smaller than the addend) | the readable definition of the feature math; fast enough for every gate geometry |
| **L2 — ground truth** | **exact** accumulation: TwoSum/Shewchuk expansion, or fixed-point `i128` where the value range provably allows it (checked, not assumed) | the value the bound in §10 is measured *against*, so "correct" is exact rather than merely careful |

L1 is compared to L2 on the gate fixtures; if L1 ever drifts from L2 beyond its
own compensated bound, the oracle is broken and says so before it is used to
judge anything else.

### 11.2 What is gated

For **every SIMD tier × geometry class**:

* tiers: `v4x`, `v4`, `v3`, `neon`/`wasm128` (where the host permits), `scalar`
  — driven by forcing the `incant!` tier, so a tier is never skipped silently;
* geometries: **tight** (`96×64`, `208×144`, `592×80`), **non-tight**
  (`200×150`, `576×96`, `127×93`), **sub-64** (`48×40`), the **golden set**
  fixtures, plus **randomized fuzz geometries** (seeded, width/height drawn
  across the `%8`, `%16` and band-boundary classes so tails and partial bands
  are hit on purpose).

**Reported per slot family, worst case named** — a table of (family, tier,
geometry, max abs dev, max rel dev, ulps, bound, pass/fail), with the single
worst cell called out in prose. Not a bare "all passed".

### 11.3 Standing role

This oracle is **the regression instrument for all future perf work on these
kernels**, not a one-off for era-2. Any later change to the pooled-feature math
— fusion, fission, tier work, a new lane width — is measured against it before
it is measured against a bench. Recorded here so the next lane inherits the
instrument rather than rebuilding it, and so "we compared against the previous
implementation" is never again the strongest available correctness claim.

### 11.4 Why this ordering matters

era-1's correctness argument was *relative*: the fold matched the buffered path,
and the buffered path matched a golden captured from itself. Nothing in that
chain was ever compared to the exact answer. The oracle breaks the circularity —
after era-2, "correct" means "within a proven bound of the exact sum", and the
golden fixtures become a *convenience* for catching regressions fast rather than
the definition of right.

---

## 12. Math update for fixed virtual lanes — and the bound my first model got WRONG

### 12.1 The oracle earned its keep before a single kernel changed

The first version of §10.5's bound modelled **only the summation error**. Run
against the oracle, it failed immediately and correctly:

```
96x64 ch0 [tight, w%8==0] dispatched: slot sum_d (core) deviates
4.459455237082466e-5 from the EXACT sum, above its proven bound
2.2204967404769756e-5  (Σ|x| = 1.8626876582358037e1)
```

Measured coefficient `dev/(u₃₂·Σ|x|)` ≈ **40**; my model predicted **20**. The
missing factor is **term-evaluation rounding**: the SIMD path evaluates each
per-pixel term in `f32`, so every term already carries its own relative error
before it is ever summed. `ssim_d_local_v` alone is ~10 `f32` operations
(`2·μ₁·μ₂+c₁`, `μ₁²+μ₂²+c₁`, `s₁₂−μ₁μ₂`, `2·cov+c₂`, `ssq−μ₁²−μ₂²+c₂`, a
division, `1−local`, a `max`) with a division in the middle, and the SSIM form
is cancellation-prone.

**This is exactly why the oracle lands before the kernels** (§11.4): era-1's
correctness argument was relative and could never have surfaced this, because
both sides of a relative comparison share the same evaluation error. The
corrected model is below; the failure is recorded rather than quietly patched,
because a bound that was wrong once should be visibly re-derived.

### 12.2 The corrected bound

For a slot accumulating `n` terms, with `k_eval` `f32` rounding operations per
term, `L = 8` virtual lanes, and `m = n/L` terms per lane:

```
|dev|  ≤  [ k_eval·u₃₂                       ]·Σ|xᵢ|     (term evaluation, f32)
        + [ (m − 1)·u₆₄ + (L − 1)·u₆₄        ]·Σ|xᵢ|     (lane accum + reduce, f64)
        + O(u²)                                                        … (5)
```

The change from era-1 is entirely in the **second** bracket: era-1's
accumulation term is `(chunks_per_lane)·u₃₂ ≈ 72·5.96e-8 ≈ 4.3e-6`; era-2's is
`(m + L − 2)·u₆₄ ≈ (9216 + 6)·1.11e-16 ≈ 1.0e-12` at 576². **Six orders of
magnitude smaller.** The first bracket is unchanged between eras because the
term evaluation is unchanged — which is why era-2 is a large *accuracy*
improvement even though it is motivated by speed.

`k_eval` per family is a documented op count, not a fitted constant:

| family | dominant formula | `k_eval` (counted) |
|---|---|---|
| core `sum_d…sum_d4` | `ssim_d_local` + up to 3 self-multiplies | ~13 |
| core `art`/`det` | `bounded_sim` (5 ops) + subtract | ~7 |
| core `mse` | `saturate` (3 ops) | ~4 |
| `hf_*` | `bounded_excess_pair` (shared-denominator, 6 ops) | ~7 |
| `pjnd_*` | `pjnd_transducer` (4 ops) | ~5 |
| pools | the above **plus** `saturate(act)` and one multiply per pair | family + 4 |

The gate uses these with a stated safety factor and reports the measured
coefficient beside the bound, so drift *toward* the bound is visible long
before it crosses.

### 12.3 The blocked-vs-sequential theorem still applies

§10.2 is unchanged and now applies to the **virtual-lane** grouping directly:
`L = 8` lanes of `m = n/8` is blocked summation with `b = 8`, so its
coefficient is `n/8 + 6 ≤ n − 1` for all `n ≥ 8` — strictly tighter than
sequential, by the same convexity argument, with the added benefit that `L` is
now a *fixed constant* rather than a tier-dependent lane width. era-1's
coefficient varied with the physical tier (8 vs 16 lanes); era-2's does not,
which is the same property that gives §2.2 its bit-identity.

### 12.4 Gate 5 (new): cross-tier and cross-arch identity

* **Cross-tier, verifiable here:** `v3`, `v4`, `v4x` all summon on this box
  (Zen 4), plus the `scalar` fallback — four tiers, asserted bit-identical.
* **Cross-arch, NOT verifiable from this box:** `neon` (aarch64) and `wasm128`
  need CI or the localizer harness. The design's claim for them is a
  *consequence of the theorem* (§2.2) plus the enumeration (§2.3), and it is
  **declared as unverified-locally** rather than asserted — the CI matrix
  (`windows-11-arm`, `macos-*-intel`, `i686`) is where it becomes evidence.

**Honest statement of what changed:** era-1's goldens were known not to hold
cross-vendor (241–246 of 372 features diverged), and the policy response was
tolerances. era-2 aims to make the exact golden *true* rather than tolerated.
That is a claim to be earned on the CI matrix, not one this doc gets to assert.

### 12.5 The second thing the oracle caught: cancellation amplification

With the term-evaluation component added, `sum_d` passed and **`sum_d2`
failed** — deviation `4.719e-7` against a bound of `1.806e-7`. The measured
coefficient against the slot's own `Σ|x|` was **120**, far above any plausible
op count.

The cause is not the accumulation at all:

```
d = max(1 − local, 0)   with   local ≈ 1        ← a CANCELLING difference
```

so `d`'s **absolute** error is `~u₃₂·|local| ≈ u₃₂` and does **not** shrink as
`d` shrinks. The moments then inherit it amplified by the derivative,
`δ(dᵏ) = k·d^(k−1)·δd`, so their error is proportional to `Σ|d^(k−1)|` — not to
`Σ|dᵏ|`. Those differ enormously: at 96×64 ch0, `Σ|d| = 18.63` against
`Σ|d²| = 0.0659`, a factor of **283**. A bound written against a slot's own
`Σ|x|` understates every derived slot by about two orders of magnitude. The
same applies to every pool `num` (weight × d).

**Resolution — one uniformly-valid form instead of a per-slot patchwork.**
Every intermediate in these formulas is bounded by 1 in magnitude (`d`, `sal`,
the weights, `bounded_sim`, `saturate` all live in `[0,1]`), so a single `f32`
rounding anywhere in a term contributes at most `u₃₂·1` — **absolute**, not
relative to `|term|`. Hence

```
|dev| ≤ k_eval·n·u₃₂                                    (term evaluation, absolute)
      + [ (chunks + L)·u₃₂ + (rows + tail)·u₆₄ ]·Σ|xᵢ|  (accumulation)          … (6)
```

which is cancellation-safe by construction and needs no special case for
moments, pools or tails.

**(6) is deliberately LOOSE.** It is a proven upper bound, not a fitted one,
and worst-case FP bounds are always slack against measured error because real
roundings partially cancel. The *regression* signal is therefore not the assert
— it is the **reported deviation as a percentage of bound**, which moves long
before the bound is crossed.

**Result: gate green, worst case `18.93 %` of its proven bound** —
`pools_scalar`, 127×93 ch2 (non-tight, `w%8==7`), `ws_peak_ssim.den`,
`dev 4.53e-3` against bound `2.39e-2`. Healthy: neither absurdly slack nor
tuned to the data.

**Scoreboard for the oracle, before it has judged a single era-2 kernel: it
found two independent errors in my analysis** (the missing term-evaluation
component, then cancellation amplification), each of which would have shipped
as a silently-wrong "proven bound". That is the argument for §11.4 — a relative
comparison against the previous implementation could not have surfaced either,
because both sides share the same evaluation error.

---

## 13. MEASURED BEFORE BUILDING: the f64 virtual lanes cost 2.3×, and the design changes

§2.1 said the scalar realisation "will likely auto-vectorise anyway — that is
to be MEASURED, not assumed". Measured (`zensim/benches/era2_accum_shape.rs`,
576 × 128, accumulation isolated with the term evaluation stubbed so the lane
shape is 100 % of the work):

| shape | time | vs era-1 |
|---|---:|---:|
| `era1_f32_lanes` (8 f32 lanes, reduced per row) | **3.08 µs** | base |
| `era2_f64_chunked` (8 f64 virtual lanes, `chunks_exact(8)`) | **6.99 µs** | **+132 – 147 %** |
| `era2_f64_virtual` (8 f64 lanes, naive `x % 8` indexing) | **34.17 µs** | +780 – 915 % |

Two findings, one of them fatal to the design as written:

1. **Never write the lanes as `lane[x % 8] += …`.** The modulo indexing defeats
   vectorisation completely — **11× slower** than the chunked form. The
   structure must be `chunks_exact(8)` with a folded tail. This is an
   implementation requirement, not a style note.
2. **f64 virtual lanes cost 2.27× on the accumulation step.** f64 SIMD lanes
   are half as wide, and the accumulation does not get cheaper elsewhere to pay
   for it. In the real kernel accumulation is a minority of the work (the term
   evaluation is division-heavy), so 2.27× is an **upper bound** on the whole-
   kernel penalty rather than the penalty itself — but it is a real cost
   against a design whose entire justification is speed.

### 13.1 Is the accuracy the f64 lanes buy actually needed? — No

The f32 lane accumulation contributes ≈ **4.3e-6** relative error (§10.3). The
dial's materiality step is 0.5 points on a `[0,100]` scale, i.e. **5e-3**
relative. The error era-2's f64 upgrade would remove therefore sits about
**three orders of magnitude below anything a score can resolve**, and about
**100× below** the module's own long-standing 5e-4 pool policy.

So the f64 accumulators buy precision that no consumer can see, at 2.27× on the
step, in a break whose stated purpose is "extremely fast".

### 13.2 The revised design: 8 **f32** virtual lanes + an f64 band layer

The prize was never f64 — it was the **fixed grouping**. Fixing the grouping is
what gives cross-tier bit-identity (§2.2) and band-parallelism (§2.4); the
accumulator *type* is independent of both. So:

```
per row:   8 f32 lanes, fixed count (era-1 uses 8 OR 16 depending on tier)
           terms enter via chunks_exact(8); the tail folds into the same lanes
           reduce by a FIXED tree  ->  f64
per band:  f64 band partial += the row's f64 value        (one add per row)
merge:     band partials folded in band order             (one add per band)
```

This keeps **every** era-2 prize:

* **cross-tier bit-identity** — the lane count (8) and reduce tree are fixed, so
  `v4x` no longer accumulates 16-wide while `v3` accumulates 8-wide. IEEE f32
  `+` is correctly rounded on every arch, so §2.2's proof holds unchanged with
  `f32` substituted for `f64`;
* **band-parallelism** — the f64 band layer is exactly the §2.4 structure;
* **no width-dependence** — the tail folds into the lanes;
* **era-1-class accuracy AND era-1-class speed** — the f32 lane depth per row is
  unchanged, so §10.3's dominant term neither grows nor shrinks.

What it gives up is the accuracy improvement, which §13.1 shows nothing
consumes.

**This is a design change driven by measurement, and it is put up for
confirmation rather than taken unilaterally** — the user's refinement said
"e.g. 8 f64 virtual lanes", and this proposes 8 **f32** virtual lanes with an
f64 band layer on top for the same structural guarantees. If the f64
accumulators are wanted for reasons beyond the ones analysed here, the 2.27 %
… 2.27× cost is now quantified and the choice is informed either way.

### 13.3 Consequence for the break's size

Under §13.2 the byte change is much smaller than under f64 lanes: fix the lane
count at 8, fix the reduce tree, fold the tail, add the band layer. It remains
a **real** byte change (era-1's `v4x` accumulates 16-wide, and the tail
currently lands in the f64 running total), so the era and its blast radius
stand — but the reshape is a far smaller and lower-risk edit than the f64
rewrite, which is a second reason to prefer it.

---

## 14. S1 implementation findings (before the kernel)

### 14.1 The lane count was ALREADY 8 everywhere — the break is smaller again

`type V8<T> = GenericF32x8<T>` and `dense_block_kernel_generic<T: F32x8Backend>`
— so **every tier, `v4x` included, already accumulates in 8 f32 lanes.** The
"v4x accumulates 16-wide" concern in §13.2 was wrong: `v4x`'s only difference is
`POOL_SIMD` (pools in lanes vs per-pixel scalar), not lane width.

So the era-2 reshape reduces to four changes, not five:

1. `POOL_SIMD` pool accumulation on **every** tier (the register-pressure item);
2. the scalar tail **folded into the lanes** instead of the f64 running total;
3. the **f64 band layer** with band-order merge;
4. a **fixed reduction tree** — see §14.2, which is the one that turned out to
   matter most.

### 14.2 `reduce_add()` IS TIER-DEPENDENT — era-2 must not call it

The identity theorem (§2.2) requires the reduction order to be part of the
semantics. It is not, today. `GenericF32x8::reduce_add` delegates to
`T::reduce_add`, and the backends genuinely disagree:

| backend | f32x8 reduction order |
|---|---|
| `x86_v3` (AVX2) | `extractf128` + `add_ps` pairs lane `i` with lane `i+4` **first**, then a `shuffle`/`movehl` tree over the resulting 4 |
| `x86_v4` (AVX-512, f32-delegated) | delegates to the `v3` f32x4 path |
| `wasm128` | `(l0+l1) + (l2+l3)` — **adjacent** lanes first |
| `scalar` | `(a[0]+a[1]) + (a[2]+a[3])` — adjacent lanes first |

`(l0+l4)+(l1+l5)` and `(l0+l1)+(l2+l3)` are different groupings of the same
eight addends, so they round differently. **Every horizontal reduction in the
current kernels is therefore an unspecified, backend-dependent operation.**

**Consequence for S1, concrete:** era-2 extracts lanes with `to_array()` and
sums them in an explicitly written fixed order. The cost is one array
materialisation per reduction — negligible, since reductions happen once per
row per accumulator, not per pixel.

**Hypothesis worth flagging, NOT a claim:** this is a plausible contributor to
the historical cross-vendor golden failure (241–246 of 372 v1 features diverged
on every non-AMD class, all non-AMD classes agreeing with each other on one
alternative result set). A tier-dependent reduction tree would produce exactly
that signature — one result per *reduction shape*, not one per CPU. It has not
been isolated to this cause and should not be reported as solved until it is;
`v4`/`v4x` delegating to `v3` while `wasm`/`scalar` use adjacent-pairing is at
least consistent with the observed clustering.

**This is the third design-level error the "write the math out" discipline has
surfaced** — after the missing term-evaluation component and cancellation
amplification (§12.1, §12.5). All three were invisible to relative A/B testing:
a fold-vs-buffered comparison shares the same `reduce_add`, so it can never see
that the reduction order is unspecified.

### 14.3 `ERA2_BAND_ROWS` is SEMANTICS, not a tuning knob — earned the hard way

The first version of the structural gate asserted "band-merge == serial fold,
bit-exact". It **failed at 127×93** (`7.854278564453125` vs
`7.8542633056640625`), and the assertion was wrong rather than the design:

```
serial:  ((band0 + r32) + r33) + …        banded:  band0 + band1
```

Banding **is** a different grouping. This is the same blocking
non-associativity documented for the era-1 dense kernel (§14 of the era-1 doc)
— it does not go away because the shape is new. (576×128 passed by
coincidence, which is exactly why a single geometry is never enough.)

**Consequence, and it needs to be loud: `ERA2_BAND_ROWS` is part of era-2's
semantics.** Changing it changes output bytes and is an era decision, not a
perf tuning knob. It is deliberately not derived from thread count, image
height, or anything else that varies at runtime — that is precisely what makes
thread-invariance structural rather than tested.

**What is actually asserted** by `era2_band_merge_and_tail_are_structural`:

* bands may be **computed** in any order (the gate computes them in reverse, as
  an out-of-order worker pool would) provided they are **merged** in band
  index order — bit-identical;
* lane `j` holds exactly the terms at `x ≡ j (mod 8)` in increasing `x`, tail
  included — so the shape is width-independent, which era-1's was not.

**Fourth error caught by the write-it-down discipline**, after the missing
term-evaluation component, cancellation amplification, and the tier-dependent
`reduce_add`. Every one of them was invisible to relative A/B testing.

### 14.4 Pass structure is a perf knob and NOT semantics

The converse, and it is useful: splitting a row into several passes (to fit
register pressure on 16-register tiers) changes neither which terms an
accumulator receives nor their order — each accumulator still sees its terms in
increasing `x`. **Any pass split yields identical bytes.** So the split can be
tuned freely, even per tier, without touching the era. Only the *lane count*,
the *reduction tree* and the *band size* are semantics.

---

## 15. The `reduce_add` hypothesis: MEASURED on one vendor/tier pair

§14.2 flagged, as a hypothesis and explicitly not a claim, that the
tier-dependent `reduce_add` could be a contributor to v1's historical
cross-vendor golden failure. It is now measured on one pair.

**Method.** `zensim/examples/era2_vendor_probe.rs` — one binary, no
`target-cpu=native`, printing all 35 dense accumulator slots as raw f64 bits
for **both eras** across three geometry classes (tight 64×64, non-tight
200×150, scalar-tail 127×93), plus the tier the dispatcher actually selected.
Built once here, `scp`'d, run on both boxes, diffed.

| box | CPU | vendor | tier selected |
|---|---|---|---|
| dev (this box) | AMD Zen 4 | AuthenticAMD | **v4x (AVX-512)** |
| i134 (LAN) | Intel Core i5-13400F (Raptor Lake) | GenuineIntel | **v3 (SSE4.2)** |

**Result.**

| era | slots differing across the pair | per geometry |
|---|---:|---|
| **era-1** | **66 of 105** | 22/35 at each of 64×64, 200×150, 127×93 |
| **era-2** | **0 of 105** | 0/35 at every geometry |

era-1 diverges on **63 %** of dense accumulator slots between the two boxes.
era-2 is **bit-identical** — same binary, different vendor, different tier.

### 15.1 What this does and does not establish

**Does:** the era-2 shape (fixed 8-lane grouping + the written-out
`era2_reduce8` tree + tail folded into the lanes) produces byte-identical
output across a genuine vendor pair where era-1 does not. The hypothesis is
**MEASURED on this pair**, not merely plausible.

**Does not:** prove it for all vendors. This is **one pair**. `neon` and
`wasm128` remain unverifiable from this box (§12.4) and stay declared as such;
CI is where they become evidence.

**And the confound must be named:** vendor and tier are **not separable here**.
The AMD box selects `v4x` and the Intel box selects `v3`, so what was varied is
*(vendor, tier)* jointly. The mechanism is tier-dependent — `reduce_add`
resolves per backend — so the honest statement is that this is a **cross-tier**
result which a vendor difference happened to induce. That is exactly the shape
the historical failure had (all non-AMD classes agreeing with *each other* on
one alternative result set, i.e. clustering by reduction shape rather than by
CPU), which is why it is consistent with the hypothesis — but "consistent with"
is as far as one pair goes.

### 15.2 The consequence worth surfacing to the user

era-1's golden policy is a **tolerance** precisely because exactness never held
cross-vendor (241–246 of 372 v1 features diverged on every non-AMD class). If
era-2 is bit-identical across tiers, **re-tightening the golden gate from
tolerance back to exact becomes a user option** — recovering a materially
stronger correctness property that was given up in 2026-08. Registered as an
option, not taken: it needs the CI matrix (`windows-11-arm`, `macos-*-intel`,
`i686`) to confirm the `neon`/`wasm` half before anyone relies on it.

---

## 16. Four catches: the case for oracle-first, in one place

The user asked for the math written out. That demand — not the tests, not the
benches — is what found all four of the following. Each was a real defect in
the analysis or the instrument, each would have shipped silently, and **not one
of them was reachable by a relative A/B test**, because both sides of a
fold-vs-buffered comparison share the same evaluation error, the same
`reduce_add`, and the same grouping.

| # | what was wrong | how it surfaced | what it would have cost |
|---|---|---|---|
| **1** | The error bound modelled only **summation**, ignoring that the SIMD path evaluates each per-pixel term in `f32` so every term carries its own error *before* being summed | oracle gate failed immediately: `sum_d` deviated `4.459e-5` against a bound of `2.220e-5` — measured coefficient **40** vs a predicted 20 | a "proven bound" that was wrong by 2×, published as the era's acceptance criterion |
| **2** | **Cancellation amplification.** `d = max(1−local, 0)` with `local ≈ 1` cancels, so `d`'s *absolute* error does not shrink with `d`; the moments inherit it amplified by `k·d^(k−1)`, making their bound proportional to `Σ|d^(k−1)|`, not `Σ|dᵏ|` | with (1) fixed, `sum_d2` failed at coefficient **120**; `Σ|d|` is **283×** `Σ|d²|` at that geometry | every derived slot's bound understated by ~2 orders — the gate would have passed anything |
| **3** | **`reduce_add()` is tier-dependent.** `x86_v3` pairs lane `i` with `i+4` first; `wasm128`/`scalar` pair adjacent lanes. Every horizontal reduction in the kernels was an unspecified operation | reading the magetypes backends while writing §2.2's identity proof — the proof could not be completed without pinning the tree | era-2 would have shipped claiming cross-tier bit-identity while calling a function that does not have it. **Now measured: 66/105 slots diverge in era-1, 0/105 in era-2** (§15) |
| **4** | My own **test premise**: it asserted "band-merge == serial fold, bit-exact". Banding *is* a different grouping — the same blocking non-associativity already documented for era-1's dense kernel | the gate failed at 127×93 (`7.854278564453125` vs `7.8542633056640625`); 576×128 passed *by coincidence* | `ERA2_BAND_ROWS` would have been treated as a tuning knob. It is **semantics** — changing it changes bytes |

Two things worth drawing out.

**Catch 3 is the one that pays for the whole exercise.** It began as a proof
obligation nobody could discharge, became a documented hypothesis, and is now
a measured cross-vendor result that puts an exact golden gate back within
reach after it was abandoned as impossible. A relative A/B would have reported
"fold matches buffered" on both boxes and found nothing — because they match
*each other* while both diverge from the other machine.

**Catch 4 is the reason to keep writing tests that can fail.** It was my
assertion that was wrong, not the design, and only a geometry sweep caught it:
the geometry a lazier test would have used (576×128) passed by luck. One
geometry is never enough, and a test that passes on the first try deserves a
negative control before it is trusted.

---

## 17. The kernel was 9–10× slower, and the cause was my own trap note

First measured build of `dense_block_kernel_era2` (paired zenbench,
`era2_dense_ab`):

| geometry | era-1 dispatched | era-2 (first cut) | ratio |
|---|---:|---:|---:|
| 576×128 | 98.2 µs | **1007.5 µs** | **10.3× slower** |
| 1152×128 | 226.2 µs | **2001.9 µs** | **8.9× slower** |

Disqualifying for a break justified by speed — and the cause was **the exact
trap §13 wrote down and I then walked into**. §13 said "never modulo indexing;
the structure must be `as_chunks::<8>`". The first kernel instead used

```rust
while x < width { let n = (width - x).min(8); for k in 0..n { … } }
```

a **runtime** bound. LLVM cannot prove `n == 8`, so it declines to vectorise and
emits a scalar loop with bounds checks — the same class of failure as the
modulo form, reached by a different route. Writing the trap down is not the
same as not falling into it; only the bench caught it.

**Fix:** `as_chunks::<8>()` on all seven input planes, one closure shared by
the full chunks (`n = 8`, compile-time constant) and the zero-padded tail
(`n = width % 8`), so the two paths cannot drift.

**The fix is byte-neutral, and that is checked rather than assumed.** Oracle
deviations after the rewrite are *identical to the digit* — core `2.7564e-4`,
hf `3.7719e-5`, pjnd `1.5011e-4`, pools `4.3567e-3` — confirming §14.4's claim
that chunking and pass structure are perf knobs, not semantics.

Re-measurement of the corrected kernel is **queued behind the fold-engine
lane** on zenbench's box-wide exclusive lock (that lock is the system working
as designed — it prevents two lanes' benches from contaminating each other).
The flip stays blocked on that number: a break justified by speed does not ship
before its speed is known.

### 17.1 The ISA gap: plain Rust compiles to baseline SSE2

The `as_chunks` fix took era-2 from 10.3× to **4.4×** slower — better, still
disqualifying. The remaining factor was not shape at all:

**a plain Rust function compiles to baseline x86-64 (SSE2) regardless of what
the host supports.** era-1's kernel lives inside an `#[arcane]`/`incant!`
`target_feature` region, so LLVM emits AVX-512 there; mine did not, so it
emitted SSE2. Same source, ~4× less vector throughput.

Fix: wrap the identical body in `#[magetypes(v4x, v4, v3, neon, wasm128,
scalar)]` + `incant!`, so each tier compiles the **same source** inside its own
`target_feature` region. The token is unused — magetypes is present purely for
the ISA region, not for its types.

| stage | 576×128 | 1152×128 | vs era-1 |
|---|---:|---:|---:|
| era-1 dispatched | 103.5 µs | 235.3 µs | 1.00× |
| era-2, runtime chunk bound | 1007.5 µs | 2001.9 µs | 10.3× / 8.9× |
| era-2, `as_chunks::<8>` | 482.1 µs | — | 4.4× |
| era-2, `+ target_feature` | **232.1 µs** | **457.5 µs** | **2.24× / 1.94×** |

**Bit-identity survives per-tier compilation — verified, not assumed.** Oracle
deviations are unchanged to the digit, and the cross-vendor probe re-run under
the new build still reports **era-1 66/105 slots differing, era-2 0/105**. So
compiling one source into six `target_feature` regions did not reintroduce
divergence, which is the outcome §2.3's enumeration predicted (no FMA
contraction without fast-math, no transcendentals, no `rsqrt`).

### 17.2 Where the remaining 2× is, and the lever for it

**era-2 is still ~2× slower than era-1, so the flip stays blocked.** The gap is
now attributable rather than mysterious: era-1 runs **one pass**; era-2 runs
**two**, writing and re-reading four scratch planes per row. That round-trip is
the two-pass design's cost, and it was adopted to fit 16-register tiers.

**The lever is already licensed by §14.4: pass split is byte-neutral, so it can
differ per tier.** `v4x` has 32 registers and does not need the split — era-1's
own `POOL_SIMD` path proves a single fused pass fits there. So the next step is
a **single-pass body for `v4x`, two-pass for the 16-register tiers**, which
§14.4 guarantees produces identical bytes and which the oracle + vendor probe
can both confirm. That is the measurement that decides whether era-2 can reach
parity; until it does, the break does not ship.

**Status: the flip is correctly blocked on a number, not on an opinion.**

---

## 18. Era-2 becomes the umbrella (user directive, 2026-08-31)

> *"push forward with refactoring for speed, we can break feature calcs for a
> new era."*

Era-2 is now the single batched break for **every** byte-changing speed
refactor, not just the accumulation reshape. The batching rule is unchanged and
now matters more: a second byte-moving change after this one costs a second
re-extraction of every 944 table and a second retrain of every 944 model, so
everything measured-and-wanted goes in **once**.

Absorbed scope, ordered by **measured gain per unit of risk** — which is the
order they will be built:

| # | item | gain | risk | why that order |
|---|---|---|---|---|
| **A** | per-tier pass split (in flight) | closes the remaining **2.24×** era-2 deficit | **lowest** — §14.4 proves pass split is byte-neutral, so the oracle must show *zero* change | already licensed, already instrumented, and it is the thing standing between era-2 and parity |
| **B** | column tiling | per-thread hot set **4.43 → 1.02 MiB** at `Tw=512`, CCD1 occupancy **35.4 → 8.2 MiB** (2304²/16T) | **medium** — reorders the pooled f64 accumulation | the big one, and only safe *because* of the fixed-grouping discipline already built |
| **C** | `fold_v1` request flag | skips f0..372 entirely for models that read none of it (W-LIN class) | **low** — pure block-skipping, same shape as `v1_only` | smaller population, but nearly free once A and B are in |

### 18.1 Column tiling — how it composes with the lanes and bands

Priced by the footprint lane (`benchmarks/fold_footprint_2026-08-31.md`): halo
derived from the kernel chain (`box_blur_1pass_into` = H then V ⇒ two chained H
passes ⇒ **10 columns per side**, buffer `Tw + 20`), redundant work
**15.6 / 7.8 / 3.9 / 1.0 %** at `Tw = 128 / 256 / 512 / 2048`.

**`TILE_WIDTH` is SEMANTICS, exactly like `ERA2_BAND_ROWS`.** It reorders the
pooled f64 accumulation, so changing it changes bytes. It carries the same
"not a tuning knob" warning at its definition, for the same reason and with the
same failure mode if ignored.

**Two constraints fall out of the existing design, and they are what make
tiling safe rather than another grouping hazard:**

1. **`TILE_WIDTH` must be a multiple of 8.** A pixel's lane is `x mod 8` on the
   *global* x. For tile `t` at offset `k`, `x = t·Tw + k`, so
   `x mod 8 = k mod 8` **iff `Tw ≡ 0 (mod 8)`**. With that, lane assignment is
   tile-invariant and the §2.2 identity proof carries over untouched. Without
   it, tiling would silently permute which terms share a lane.
2. **The merge is a fixed function of (tile, band).** Defined as **tile-major,
   band-minor, both increasing** — which is also the natural *loop* order for
   tiling (walk a column strip down the whole image while its working set is
   hot), so the merge order and the loop order coincide and **no per-tile
   partial storage is needed**. Each tile's partial is order-free; the merge is
   sequential in tile index. §10.6's determinism theorem applies verbatim with
   `(height, BAND)` replaced by `(height, width, BAND, TILE_WIDTH)`.

Gated by the same instruments, no new ones: scalar oracle per tier × geometry
(with tile-boundary geometries added — widths that are and are not multiples of
`TILE_WIDTH`), the vendor probe on dev + i134, determinism, and
thread-invariance by construction.

### 18.2 Machine facts — corrected, and one old claim FALSIFIED

Superseding the numbers used earlier in this doc:

| | |
|---|---|
| CPU | **Ryzen 9 9950X3D**, asymmetric L3 |
| CCD0 | cpus 0-7, 16-23 → **96 MiB** (3D V-Cache) |
| CCD1 | cpus 8-15, 24-31 → **32 MiB** |
| `getconf LEVEL3_CACHE_SIZE` | reports 32 MiB — **wrong for half the machine** |
| per-thread budget (binding = CCD1) | **8 MiB at 8T, 4 MiB at 16T** |

**The "3.5× is the machine's own bound" claim is FALSIFIED** — it was the
fold's *own footprint*, not a hardware ceiling. Post-fix saturation is **5.85×
(big CCD) / 4.54× (small)**. Tiling is therefore designed against the **CCD1**
budget, and per-CCD saturation gets reported before and after.

### 18.3 Re-baselining — the old ratios in this doc are stale

The fold-engine, fold-MT and footprint lanes have all landed byte-neutral work
on these kernels since §17 was measured. The 1T fold is now **0.78× / 0.87×**
buffered — i.e. *faster* serially — so §17's era-1 baseline no longer describes
current main. Everything from here is measured after
`jj git fetch && jj rebase -d main@origin`, and the §17 table is retained only
as the record of how the 10.3× → 2.24× sequence was diagnosed.

---

## 19. Stage A result: the fused v4x hypothesis is FALSIFIED, and the real gap is named

### 19.1 Re-baseline on current main

The other lanes' byte-neutral work moved the baseline, so everything was
re-measured after `jj rebase -d main@origin`:

| geometry | era-1 dispatched | era-2 (two-pass) | ratio |
|---|---:|---:|---:|
| 576×128 | 106.8 µs | 226.1 µs | **2.12×** |
| 1152×128 | 231.8 µs | 454.9 µs | **1.96×** |

Close to §17's numbers, so the other lanes' work did not change the era-1/era-2
relationship — useful to know before attributing anything.

### 19.2 The fused hypothesis: measured, and wrong

§17.2 proposed a single fused pass for `v4x` on the reasoning that 32 SIMD
registers would hold 13 core + 16 pool accumulators and skip the scratch
round-trip. Implemented behind a `const FUSED: bool` and measured:

| geometry | era-1 | era-2 two-pass | era-2 **fused** |
|---|---:|---:|---:|
| 576×128 | 111.8 µs | 226.1 µs | **445.5 µs** |
| 1152×128 | 212.7 µs | 454.9 µs | **911.2 µs** |

**Fusing is 2× WORSE**, taking era-2 from 2.12× to 3.98× of era-1. The
arithmetic says why: 29 `Lanes8` accumulators is **232 live f32 values**, and
32 registers do not hold that — it spills. This is the same wall §A.14 hit when
it scalarised the pools originally, and exactly why `POOL_SIMD` was capped at
16 accumulators rather than 22. **The two-pass split is not a 16-register
concession; it is the right structure on every tier.**

Both instantiations are retained (set to two-pass) because
`era2_fused_and_two_pass_are_bit_identical` asserts §14.4's byte-neutrality
against both — across 6 geometries × 3 channels, **all 35 slots bit-identical**.
So the claim that pass structure is a perf knob and not semantics is now
confirmed in code, on the very configuration that turned out to be the slower
one.

### 19.3 The remaining 2.12× is attributable: `[f32; 8]` arrays vs `V8<T>`

With fusion ruled out and the ISA region already in place (§17.1), one
structural difference remains between the two kernels: era-1 accumulates in
**`V8<T>` magetypes SIMD types**; era-2 accumulates in **plain `[f32; 8]`
arrays** and relies on auto-vectorisation. era-1's `v4x` path already does
everything era-2's fused attempt tried — 13 core + 16 pool accumulators, one
pass, no scratch — and is fast, which isolates the difference to the
accumulator representation rather than to the pass structure or the ISA.

**Next lever, well-specified:** rewrite the era-2 body against `V8<T>` while
keeping every semantic (8 lanes — `V8` *is* 8 lanes; the explicit
`to_array()` + `era2_reduce8` tree, never `reduce_add()`; tail folded into the
lanes; band-order merge). The semantics are unaffected because `V8`'s
elementwise ops are the same IEEE operations in the same order, and all three
instruments verify it immediately: the oracle, the fused/two-pass gate, and the
vendor probe.

**The flip stays blocked.** Stage A did not close the gap; it falsified one
hypothesis, confirmed the byte-neutrality claim the whole per-tier strategy
rests on, and named the next lever with evidence. Column tiling (item B) and
`fold_v1` (item C) are unaffected by this result and remain queued behind it —
tiling in particular should be built on the *fast* kernel, not the slow one,
since its predicted win is a cache-footprint effect that a 2× compute deficit
would mask.

---

## 20. Stage B: the V8 rewrite closes the 2.12× — and the closure→macro step was worth 55×

### 20.1 Result

| geometry | era-1 dispatched | era-2 (V8, macros) | ratio |
|---|---:|---:|---:|
| 576×128 | 98.4 µs | **101.8 µs** | **+4.0 – +5.8 %** |
| 1152×128 | 202.7 µs | **210.1 µs** | **−7.1 – +16.2 %** (CI spans zero) |

**Parity.** era-2 is within ~4–6 % at 576² and statistically indistinguishable
at 1152². The gap that blocked everything downstream is closed.

### 20.2 The path, and the trap in the middle

| step | 576×128 | vs era-1 |
|---|---:|---:|
| runtime chunk bound | 1007.5 µs | 10.3× |
| `as_chunks::<8>` | 482.1 µs | 4.4× |
| `+ target_feature` region | 226.1 µs | 2.12× |
| `V8<T>` accumulators, **closures** | **5610.1 µs** | **36×** |
| `V8<T>` accumulators, **macros** | **101.8 µs** | **1.04×** |

The V8 rewrite made things **17× worse** before it made them better, and the
cause is documented in this very file's history: era-1's `dense_block_kernel`
carries a comment recording a **5.3×** regression when "the POOL_SIMD variant
pushed the body past LLVM's inline-cost threshold, the hint stopped being
honored and every V8 operator compiled into a CALL to a non-inlined
`core::arch` shim outside the feature region". My body had two large closures
(`terms`, `pools`); LLVM declined to inline them, so every `V8` operation
became an out-of-line call **outside** the `target_feature` region — the same
failure, at 36× instead of 5.3× because the body is bigger.

**Converting the two closures to `macro_rules!` — textual expansion, so there
is no inline-cost decision to lose — took it from 5610 µs to 101.8 µs, a 55×
step.** `#[inline(always)]` on the enclosing function is *not* sufficient when
the hot work sits inside closures it contains.

**This is the fifth catch**, and the first one the *code's own comments* had
already warned about. Reading era-1's kernel comment before writing the
replacement would have saved the detour; the bench found it in one run
regardless.

### 20.3 All three instruments re-verified on the V8 kernel

* **Oracle bounds** — deviations essentially unchanged, with one *improvement*:
  `hf` went **3.7719e-5 → 3.4382e-5**, because `bounded_excess_pair_v` uses a
  true division where my scalar helper used reciprocal-multiply. The V8 path
  now matches era-1's per-pixel term math exactly. core `2.7564e-4`, pjnd
  `1.5011e-4`, pools `4.3567e-3` unchanged; all under bound.
* **Vendor probe (dev AMD-v4x vs i134 Intel-v3)** — **era-1 66/105 slots
  differ, era-2 0/105**. Cross-tier bit-identity survives the rewrite, which is
  the load-bearing one: each tier now compiles its own `V8` operations, and
  they still agree to the bit.
* **Bit-identity gates** — `era2_fused_and_two_pass_are_bit_identical`,
  `era2_band_merge_and_tail_are_structural`,
  `era2_reduce_tree_is_fixed_and_explicit` all green.

### 20.4 One trap worth recording: the tail cannot be zero-padded

The V8 chunk loop is fed by `as_chunks::<8>`, but the row **tail** must be
masked, not zero-padded into a full-width add. Padded lanes are not
accumulator-neutral for the pool **weights**: `mask_w = 1 − saturate(act)` and
`saturate(0) = 0`, so every padded lane would contribute `mask_w = 1.0` (and
`iw_w = IW_WEIGHT_FLOOR`) to the weight denominators. `v8_add_first` therefore
adds only the first `n` lanes. The core term families *are* zero on padded
lanes (`d`, `art`, `det`, `mse`, `hf`, `pjnd` all evaluate to 0 from zero
inputs), which is exactly why this would have been easy to miss — the
pool denominators are the only place it shows.

### 20.5 Status

The 2.12× is closed, so tiling (item B) and `fold_v1` (item C) are unblocked
and will be built on **this** kernel. Still required before the flip: the
sibling lane's v2-348/append hand-off folded into the same break (reusing
`Lanes8`/`era2_reduce8` rather than growing a second set of primitives), rank
preservation, the blast-radius registration, and the gate re-pin enumeration.

---

## 21. Charter widened: "speed is king" (user directive, 2026-08-31)

Until now every era-2 lever had to reproduce the same mathematical quantity —
the break bought a *regrouping*, never a *redefinition*. That constraint is
lifted. The user's directive, in its own terms: **speed is king**; features may
be **dropped** if it helps; HDR wants **toggleable feature sets per route**;
**const generics** are in scope; **direct archmage intrinsics** are allowed
beyond magetypes where they pay; and **the canonical calculation of a feature
may change for speed, as long as the feature remains useful**.

**The bar moves from bit-identity to utility preservation — and utility has to
be *proven*, not asserted.** Bit-identity remains the standard *within* the
era (cross-tier, cross-vendor, thread-count); what is no longer required is
that era-2 reproduce era-1.

### 21.1 The redefinition bar — declared BEFORE anything is fitted

Any feature redefinition must clear all three, and the thresholds below are
registered **now**, before a single candidate exists, precisely so they cannot
be renegotiated against a result:

1. **A scalar oracle for the NEW definition.** The oracle is not optional
   because the definition changed — it is what makes "the new formula, computed
   exactly" a thing that exists to measure against. Same two levels (Neumaier
   L1, exact-expansion L2), same per-family bounds written out (§10.5), same
   "a bound violation is a bug, not a tolerance to widen".
2. **Rank preservation, pre-declared bar.** Shipped B + the 944 roster + the
   two W-LIN winners, scored on old-vs-new features over the **same pairs**:
   > **PASS iff no corpus loses more than `0.005` SROCC and the product
   > composite does not fall.**
   `0.005` is ~200× the era-1→era-3 (option C) precedent of `+0.000024` on
   cid22 and an order below the campaign's 0.5-score-point materiality step, so
   it is strict enough to catch a real utility loss and loose enough not to
   trip on reseeding noise. **A redefinition that fails is reverted, not
   renegotiated.**
3. **The dial gates**, wherever a redefinition could bend monotonicity —
   `bake_verdict`'s G1/G3/G4 on the densified multi-codec grid. A cheaper
   formulation that costs monotonicity is not cheaper.

### 21.2 Dropping features — the drop set, and how a drop is expressed

Input from the frontier lane: **peaks are free byproducts**, **masked+IW are
one pass group costing +33–36 %**, **v2-348+append roughly doubles
extraction**. The intra-v2 structural read (which slots are byproducts vs
which force their own sweep) comes from the v2-block lane
(`benchmarks/v2_block_cost_2026-08-31.md`) and is a **coordination input, not
something this lane re-derives**.

**A drop is a declared structural zero, never a renumbering.** The
append-only discipline is unconditional: `f156..371`'s precedent is exactly
this — slots preserved, zeroed, with a registered reason, so every existing
bake keeps reading the right columns. A drop set therefore ships as
(slot range → registered reason) in `benchmarks/eval_annotations.json`, and
`bake_block_profile` already reports which blocks a bake actually uses, so the
per-model cost of a drop is computable before it is taken.

**Proposal shape** (to be filled by measurement, not guessed): for each
candidate block, its **measured** extraction saving and its **measured** rank
cost per model class, so the user chooses on a table rather than an argument.

### 21.3 HDR toggles — generalize `V1PoolsMode`/`v1_only` into a compute set

The right shape is a **compute-set descriptor** on the request: which feature
families to compute, resolved per route. `V1PoolsMode` and the `#[doc(hidden)]`
`v1_only` are the two ad-hoc instances of this idea already in tree; era-2
replaces both with one descriptor rather than adding a third.

Frontier-lane input: **`BHdr` reads 28 masked / 17 IW; `c_hdr_l1t1944` reads
0 masked / 72 IW.** So the HDR route demonstrably does not need the whole pool
group, and a descriptor lets it say so.

**One thing to state rather than leave implicit: the fold currently falls back
to the BUFFERED path for declared-HDR input.** Era-2 does **not** change that
by itself — the descriptor makes the HDR *feature set* expressible, not the HDR
*route* fold-native. Whether HDR becomes fold-native is a separate decision
that belongs with the fold-engine lane, and this doc will not quietly imply it
has happened.

### 21.4 Const generics — yes, with the cost reported

For the compute-set descriptor and the tier/lane/tile parameters, monomorphising
removes branches from the hot loop. Stage B is a live warning about
monomorphisation-adjacent effects: the closure→macro finding shows how
violently code shape interacts with inlining here. So every const-generic
parameter added must report **code size and compile time alongside the speed
win** — more monomorphisation is also more I-cache pressure and more build
time, and this lane has already measured one case where the "obviously faster"
structure was 36× slower.

### 21.5 Direct archmage intrinsics — justified per site

The repo rule prefers `#[magetypes]` for uniform algorithms, and **stage B is
evidence for that rule, not against it**: the generic `V8<T>` types turned out
*faster* than plain `[f32; 8]` arrays. So raw intrinsics are reserved for
specific measured cases where even `V8` leaves something on the floor. Each
such site needs its measured delta, and the generic path stays as the reference
implementation for every tier not hand-written — which is also what keeps the
cross-tier bit-identity claim checkable.

### 21.6 Re-ordered plan (gain per unit of risk)

| # | item | gain | risk | status |
|---|---|---|---|---|
| ~~A~~ | pass split | — | — | **done**: fused falsified, byte-neutrality confirmed |
| ~~B~~ | `V8<T>` accumulators | **2.12× → 1.04×** | — | **done**, parity reached |
| **C** | column tiling | hot set 4.43 → 1.02 MiB/thread | medium | next; constraints derived (§18.1) |
| **D** | compute-set descriptor (subsumes `fold_v1`, `v1_only`, `V1PoolsMode`, HDR toggles) | skips whole blocks per route | low | replaces item C of §18 with the general form |
| **E** | drop set | up to ~2× (v2-348+append) | **highest** — the only item that can lose utility | needs §21.1's bar and the v2-block lane's read |
| **F** | redefinitions | unbounded, per candidate | highest | each one takes §21.1 individually |

**C and D first** — both are pure skipping/reordering with the existing
instruments, no utility question. **E and F last**, because they are the only
items where "faster" can mean "worse", and they are the ones that need the
user's judgement on a measured table rather than this lane's.

**Unchanged non-negotiables:** ONE batched era; oracle + written-out math +
declared bounds per kernel; determinism and thread-invariance by construction;
cross-tier and (where verifiable) cross-vendor bit-identity *within* the era;
the vendor probe on dev + i134; blast radius registered, not launched; HDR-route
cleanliness; and the flip blocked until parity and perf numbers exist.

---

## 22. Band-local phase A: MEASURED AND FALSIFIED at every band height

The v2-block lane's hand-off named this the era's biggest measured serial
prize — "**−65.8 ms: the block 1.49×, the whole walk 367.7 → ~302 ms
(1.22×)**", with ~275 ms (1.34×) projected. It was measured **by proxy** (the
v1 fold's band-local self-blur against phase A's strip-wide H blur, at
2.00 vs 4.94 ns/px). Built and measured directly, **it loses at every band
height tested**, and the reason is arithmetic that the proxy could not show.

### 22.1 What was built

`stream_phase_ab_banded` (`feature_v2.rs`) runs the phase-A plane chain and
the phase-B kernels together over one band of `ZENSIM_BAND_ROWS` rows at a
time, out of a `width * (B + 2·HALO_P)` scratch, instead of phase A
materialising ~13 strip-wide planes that phase B then re-reads. The band
window is exact, not conservative: `activity(y) ← abs_src(y ± R) ← mu1(y ± R)
← mu1_h(y ± 2R)` and `HALO_P = 2·BLUR_RADIUS`, so `± HALO_P` is precisely the
closure of the chain; the v1 fold's own `V1_BAND_OVERLAP = 5 ≤ HALO_P` and the
gradient kernel's ±1 row both fit inside it. `wide_h` is always
`strip_h + 2·HALO_P` (the producer mirror-gathers the halo even at the plane
edges), so the window never clips and every band is exactly `B + 2·HALO_P`
rows.

**The plumbing is proven, not asserted.** At `B = STRIP_ROWS` the band loop
degenerates to one band per strip — the same decomposition the strip-wide path
performs — and it reproduces the strip-wide 956-feature vector **bit-for-bit,
956 of 956 slots** at 1152². That is the control: any divergence at smaller
`B` is the band split, not a plumbing bug.

### 22.2 The measurement

**Protocol first, because it is what makes the numbers mean anything.** 2304²,
1T, `944full`, CCD0-pinned (`taskset -c 0-7,16-23`), arms selected by a
**runtime** flag so all four run from ONE binary, env values chosen to be the
**same byte length** in every arm (`ZENSIM_BAND_LOCAL=0|1`,
`ZENSIM_BAND_ROWS=032|064|128`) so the environment block cannot shift the
address space between arms, arms **interleaved** round-robin, **min of 7 walks
within each process**, and — the part that turned out to dominate everything —
**15 separate process starts per arm with ASLR ON, reported as the minimum
over those 15 layouts.** §22.5 is why.

`B=128` is the **control**: the band loop degenerates to one band per strip,
so it performs the identical decomposition and emits the identical bits, and
whatever it costs above the baseline is the price of the extra scratch buffer
and the loop plumbing, not of banding.

| shape | halo redundancy | MIN over 15 | p25 | median | max | vs base | **minus control** |
|---|---:|---:|---:|---:|---:|---:|---:|
| strip-wide (base) | 1.156× | **323.54** | 332.67 | 335.11 | 360.33 | — | — |
| banded `B=32` | 1.625× | **373.97** | 383.99 | 412.19 | 420.14 | +15.6 % | **+13.1 %** |
| banded `B=64` | 1.31× | **341.39** | 350.03 | 352.20 | 379.31 | +5.5 % | **+3.0 %** |
| banded `B=128` (control) | 1.156× | **331.56** | 333.48 | 335.34 | 358.75 | +2.5 % | 0 |

**Every band height that actually bands is worse, monotonically in the halo
redundancy.** Three other estimators agree on the sign:

| estimator | base | `B=32` | `B=64` | `B=128` |
|---|---:|---:|---:|---:|
| ASLR **off**, one layout | 334.68 | 415.75 (+24.2 %) | 375.45 (+12.2 %) | 356.50 (+6.5 %) |
| ASLR on, median of 9 | 355.46 | — | 351.87 (−1.0 %) | 335.46 (−5.6 %) |
| ASLR on, min of 15 (above) | 323.54 | 373.97 (+15.6 %) | 341.39 (+5.5 %) | 331.56 (+2.5 %) |

The ASLR-on **median** row is the one to distrust and it is included as the
counter-example: with a bimodal ±8 % layout distribution and n = 9 it reports
`B=128` — a bit-identical arm — as **5.6 % faster than the thing it is
identical to**. The min-over-layouts estimator puts the same arm at +2.5 %,
which is the honest reading of "the plumbing costs a little", and that is the
sanity check the estimator has to pass before any other row of the table is
worth reading.

**The plumbing is proven, not asserted.** At `B = STRIP_ROWS` the walk
reproduces the strip-wide 956-feature vector **bit-for-bit, 956 of 956 slots**
at 1152² (`ZENSIM_BIGPAIR_DUMP` writes every feature with its `to_bits()`).
At `B = 32`, 547 of 956 slots are still bit-identical and the divergence is
confined to near-zero cancelling features — the worst *relative* deltas
(f693 at 19 %, f692 at 16 %) sit on absolute values of 2.9e-5 and 2.1e-5, i.e.
absolute changes of ~5e-6 — which is the expected signature of the V-blur
running-sum restart, not of a plumbing error.

Under `ZENSIM_FOLD_TIMING` (which perturbs the cache and compresses the
spread) the phase attribution shows where it goes at `B=64`: `blur_h`
**129.40 → 117.70 ms** — the band-local H blur *is* more efficient per unit
work (129.40/1.156 = 111.9 vs 117.70/1.31 = 89.8, **1.25×**, the same
direction the proxy reported at 1.40×) — but it is doing **13 % more work**,
and `planesA` (41.12 → 40.24) and `planesApp` (18.65 → 20.01) do not improve
at all. The efficiency gain is real, and it is smaller than the tax.

### 22.3 Why the proxy over-promised — the arithmetic the ns/px numbers hide

The proxy compared the v1 fold's self-blur against phase A's strip-wide blur.
Those two do not have the same closure:

* the fold's self-blur produces **four H planes** and consumes them in the
  same band, so it needs `± V1_BAND_OVERLAP = ± 5` rows → **42 rows for 32
  kept, 1.31×**;
* phase A's chain ends at `activity = blur(|src − blur(src)|)`, **two chained
  blurs**, so it needs `± 2·BLUR_RADIUS = ± 10` → **52 rows for 32 kept,
  1.625×**, and that redundancy applies to the V blurs and the activity and
  `bs2` chains too, not only to H.

So "make phase A band-local" costs 1.41× the plane work that the proxy's
shape costs (1.625/1.156 against the strip form; 1.24× against the fold's own
1.31×). At the per-unit efficiency actually available (1.25×), the trade is a
wash at best — which is exactly the measured `B=64` row, and it gets worse as
`B` shrinks.

The second reason is footprint. The hand-off's target working set was
**~1.2 MiB** — "~21 rows of H planes and ~11 of V planes … ≈ 130 rows ×
width × 4 B". A 52-row band of the **13** planes the chain actually keeps is
**676 plane-rows = 6.2 MiB at 2304**, five times that, so the band-local form
never becomes L2-resident; it moves from *L3-and-DRAM* to *L3*. The 1.2 MiB
figure belongs to the **rolling row window**, not to a band.

### 22.4 And the rolling row window is predicted to lose, by an already-measured result

The rolling window is the shape that gets both properties the band cannot:
zero halo redundancy (each H row computed once) and the ~1.2 MiB footprint.
It also **requires** the V blur to advance one row at a time across the whole
plane, keeping one running sum per column in a `width`-sized array — which is
precisely the **row-major running-sum V blur** the v2-block lane already built,
proved bit-identical over 21 geometries × 3 radii, measured at **+9 % on
`planesA` at its best tile size (47.71 → 52.13 ms)**, and reverted (L3). Its
finding is the same mechanism that defeats the band: the column-major form
keeps its accumulator **in a register** across all 148 rows, and any shape
that round-trips that accumulator through memory pays more than the traversal
saves.

I did not rebuild it to re-derive that. **Both shapes that could deliver the
hand-off's prize are now measured: the band directly (this section, three
heights), and the rolling window's load-bearing component by the lane that
tried it.** The prize as scoped is not there.

### 22.5 Methodology — the noise floor is 10 %, and it is ASLR

This is the transferable part, and it invalidates more than this experiment.

The first three sweeps of this experiment said band-local **won by 6.6 %**,
then that `B=128` won by 9.4 %, then that everything was a wash. All three
were the same binary. The cause, isolated:

| condition (same binary, same env, 8 process starts) | results (ms, min of 11 walks each) | spread |
|---|---|---:|
| **ASLR off** (`setarch -R`) | 363.22 363.39 363.53 363.22 363.16 363.49 362.98 363.80 | **±0.13 %** |
| **ASLR on** | 335.48 335.11 357.60 340.53 328.81 361.47 361.97 357.51 | **10.1 %** |

**With ASLR disabled the 944 walk at 2304² is deterministic to ±0.13 %. With
ASLR enabled it is bimodal over a 10 % range**, landing near ~334 or ~360 and
rarely between. Identical binary, identical environment, identical work — only
the mmap base differs. The plane buffers are each `2304 × 148 × 4 B` =
1,363,968 B = exactly 333 pages, allocated at a fixed relative stride, so
where that whole block lands decides whether the streams conflict.

Things that were tried and are **not** the mechanism, each measured:

* **Transparent huge pages.** THP is `madvise` on this box and
  `AnonHugePages: 0 kB` in `smaps_rollup` in **both** the fast and slow
  states. Not it.
* **A controlled heap-base shift.** Leaking 0…512 pages immediately before
  the scratch is sized (`ZENSIM_BIGPAIR_HEAPSHIFT`) moved the walk by
  **nothing**: 327.29–328.27 ms across the whole sweep.
* **Staggering the planes against each other.** Giving plane *k* an extra
  `k × stagger` elements, swept over 64 B … 64 KiB, also moved **nothing**:
  326.95–328.54 ms. Both probes were removed; neither is a lever.
* **CCD placement.** The 9950X3D's asymmetric L3 (96 MiB on CCD0, 32 MiB on
  CCD1) was the first suspect. Pinning to one CCD does not remove the
  bimodality. Pinning is retained because it removes one variable for free.

And one trap worth naming, because it produced a confident wrong conclusion
mid-session: **the size of the environment block is itself a layout input.**
Adding `ZENSIM_PLANE_STAGGER=0` to the environment — a variable that provably
does nothing — flipped one build from 359 ms to 328 ms, and a later unrelated
edit flipped the sense of that same comparison. Any A/B whose arms are
selected by *presence* of an env var is comparing two address spaces. Hence
`ZENSIM_BAND_ROWS=032` rather than `=32` in §22.2.

**The protocol that follows from this**, and that any future perf claim in
this repo at this size has to meet:

1. **One binary, runtime-selected arms.** A before/after across two *builds*
   cannot distinguish a real change from a relayout; any edit reshuffles the
   binary's own layout by the same ~10 %.
2. **Byte-identical environment blocks** between arms.
3. **Interleave** the arms; never `base×N` then `arm×N`. Sequential blocks
   measure drift.
4. **Min of N walks inside a process** (the machine can only make a walk
   slower) — this removes *interference*, and it is not enough on its own.
5. **Min over ≥15 process starts with ASLR on** — this removes *layout*. It
   answers "what does this shape cost in its best placement", which is the
   only question that is stable enough to be worth answering.
6. **Carry a bit-identical control arm** whenever one exists (here `B=128`).
   Its measured delta is the plumbing/layout floor, and an estimator that
   reports the control as *faster than the thing it is identical to* is
   telling you it is not yet sound.

A `setarch -R` run is a useful adjunct — it is deterministic and fast — but it
samples exactly one arbitrary layout, so it can only ever be a second opinion.

### 22.6 Disposition

`stream_phase_ab_banded` and its `ZENSIM_BAND_LOCAL` / `ZENSIM_BAND_ROWS`
knobs are **reverted** — a losing second implementation of the plane pipeline
is a duplicate, and this repo's rule is to delete it rather than park it
(the same call the v2-block lane made on its row-major V blur). What is kept
is the instrument: `foldapp_stream_bigpair` gained `ZENSIM_BIGPAIR_TOGGLES`
(`944full` / `924` / `372`), `ZENSIM_BIGPAIR_ITERS` (median + **min**),
`ZENSIM_BIGPAIR_PARALLEL`, and `ZENSIM_BIGPAIR_DUMP` (all features with their
`to_bits()`, which is how the `B=128` bit-identity control was run).

**Where the plane pipeline can still be attacked, and why it is the other
axis.** The halo tax is what killed row banding: 20 rows out of 32 is **62 %**
redundancy. The same closure in the **column** direction is `± BLUR_RADIUS`,
so a 256-wide column tile pays `266/256` = **4 %**, and a 512-wide tile 2 %.
Column tiling is cheap exactly where row banding is expensive, and it cuts the
H blur's working set — the doc's own diagnosis of the 3.69× degradation
("16 rows × 6 planes × 2304 × 4 B = 884 KiB, which is the 1 MiB L2 on this
part") — along the axis that appears in that product. That is where this lane
goes next, and §22.2's estimator is what it will be judged on.

---

## 23. Column tiling: the H blur is a function of WIDTH, and tiling it is 1.20×@5 MP / 1.78×@21 MP

§22 killed row banding because the halo closure of the phase-A chain is
`±2·BLUR_RADIUS` — 20 rows out of a 32-row band, **62 % redundancy**. The same
closure in the **column** direction is `±BLUR_RADIUS`: a 1536-wide tile
re-blurs 10 of 1546 columns, **0.6 %**. That asymmetry is the whole reason this
axis works and the other one cannot.

### 23.1 The zero-code probe that pointed here

Hold the pixel count fixed at 5.31 MP and change only the aspect ratio
(2304², 1T, `setarch -R`, `ZENSIM_FOLD_TIMING`):

| shape | width | `blur_h` ms | `v2:planesA` ms |
|---|---:|---:|---:|
| 2304 × 2304 | 2304 | **104.99** | 38.62 |
| 1152 × 4608 | 1152 | **34.58** | 27.07 |
| 576 × 9216 | 576 | **30.95** | 26.41 |
| 288 × 18432 | 288 | **31.33** | 25.14 |

**Same work, same pixels: the H blur costs 3.4× more at width 2304 than at
width 1152, and is flat below that.** That is not a pixel-count effect and it
is not bandwidth — it is the kernel's own working set. It holds 16 rows ×
6 planes (src, dst, four outputs), which is `16 × 6 × 2304 × 4 B` = **884 KiB**
against the 1 MiB L2 on this part, and 442 KiB at 1152. `planesA` shows the
same shape more weakly (1.43×). The v2-block lane named this mechanism
(`benchmarks/v2_block_cost_2026-08-31.md` §2.3) and concluded
`H_BLUR_BAND_ROWS = 16` was already the smallest legal value — which is true,
and is why the fix has to come from the **other** dimension of that product.

### 23.2 What was built (a deliberate lower bound)

`fused_blur_h_ssim_column_tiled` (`feature_v2.rs`) runs the existing
`fused_blur_h_ssim` over column tiles of `ZENSIM_H_TILE` output columns, each
blurred with a `±BLUR_RADIUS` column halo whose outputs are discarded. It
**copies** each tile's two inputs in and its four outputs back out, so every
consumer downstream still sees full-width planes and nothing else in the walk
changes. Those copies are pure overhead that a stride-aware kernel would not
pay — **so every number below is a lower bound on the real win.**

Not bit-exact: the kernel's running sum along x restarts at each tile, so this
is an era-2 byte change. Tile boundaries are a pure function of `(width,
tile)`, so the result stays thread- and schedule-invariant — the era-2
contract. Control: at width 1152 with `tile = 1536` no tiling happens and the
956-feature vector is **bit-identical, 956/956**; with `tile = 1024` (two
tiles) 791/956 slots are still bit-identical and the largest *relative*
deviation is 0.143 on `f171`, whose absolute value is 4.0e-4 — the near-zero
cancelling-feature signature, same as §22.2.

### 23.3 The measurement

2304², 1T, `944full`, CCD0-pinned, one binary, runtime-selected arms with
byte-identical env values (`ZENSIM_H_TILE=0000|0512|1024|1536`), interleaved,
min of 7 walks per process, **min over 15 process starts** — the §22.5
protocol.

| tile | MIN over 15 | p25 | median | max | spread | vs untiled |
|---|---:|---:|---:|---:|---:|---:|
| off (`0000`) | 324.10 | 333.20 | 334.64 | 356.14 | **9.9 %** | — |
| 512 | 271.25 | 272.15 | 272.53 | 277.61 | 2.3 % | **1.195×** |
| 1024 | 271.64 | 271.88 | 271.96 | 277.18 | 2.0 % | 1.193× |
| 1536 | **270.57** | 270.89 | 271.06 | 279.37 | 3.2 % | **1.198×** |

Two results, not one:

1. **1.20× on the whole 944 walk**, flat across tile widths from 512 to 1536 —
   which is what §23.1 predicts, since anything at or below ~1152 fits.
   Under `ZENSIM_FOLD_TIMING` the attribution is unambiguous: **`blur_h`
   130.47 → 40.36 ms, a 3.2× on the single largest item in the walk.**
2. **Tiling also collapses the ASLR lottery.** The untiled arm spans 9.9 % over
   15 layouts; every tiled arm spans 2–3 %. Shrinking the working set removes
   most of the conflict cliff §22.5 documents — a second, independent benefit,
   and one that makes every future measurement here cheaper.

### 23.4 Size sweep — and the superlinear term this removes

Min over 7 process starts per cell, `tile = 1536`:

| size | MP | untiled | tiled | speedup | untiled ms/MP | tiled ms/MP |
|---|---:|---:|---:|---:|---:|---:|
| 576² | 0.33 | 7.36 | 7.36 | **1.000×** | 22.2 | 22.2 |
| 1152² | 1.33 | 62.52 | 62.47 | **1.001×** | 47.1 | 47.0 |
| 2304² | 5.31 | 325.40 | 270.72 | **1.202×** | 61.3 | 51.0 |
| 4608² | 21.2 | 2308.51 | 1298.79 | **1.777×** | 108.9 | 61.3 |

The two small buckets are **exactly unchanged**, as they must be: their width
is below the tile, so no tiling happens and the code path is the untiled one.
That is the regression check for the small end, not an argument by analogy.

The interesting column is ms/MP. Untiled, it climbs **22.2 → 108.9** — the
944 walk has a large superlinear term in image size. Tiled, it climbs
22.2 → 61.3, and from 2304² to 4608² (4× the pixels) the tiled walk grows
**4.80×** where the untiled grows **7.09×**. **Tiling does not just shift the
curve down, it removes most of the superlinear term**, which is why the win
grows with size — 1.20× at 5 MP, 1.78× at 21 MP, and by construction larger
still above that.

### 23.5 THE WIN IS THE PACKING, NOT THE COLUMN RESTRICTION

The obvious criticism of §23.2 is that it copies six planes per tile to avoid
touching four hand-written tier bodies, and that a kernel which simply took
the output column range would get the same locality **without** the copies.
That was this lane's own stated next step. It was built, and it is **wrong**.

`x0`/`x1` were threaded through all **16** H-blur bodies (four families —
`box_blur_h`, `box_blur_h_into_abs_diff`, `fused_blur_h_mu`,
`fused_blur_h_ssim` — × four tiers), 38 window-initialisers generalised from
`|i − r|` to `|x0 + i − r|` reflected at the PLANE's edge, 38 x-loops
re-based, and 48 rem-ring warm-up conditions changed from `x >= diam` to
`x >= x0 + diam`. Every existing entry point became a thin wrapper passing
`0..width`, and the **whole suite stayed green at 369 passed / 0 failed**,
including the v1 golden byte gates — so the generalisation is byte-neutral on
the packed path, exactly as designed.

Then both forms were put behind one knob in ONE binary and measured:

| cell | untiled | **packed** | range-in-place |
|---|---:|---:|---:|
| 1T 2304² | 354.28 | **281.90 (1.257×)** | 334.62 (1.059×) |
| 1T 4608² | 2440.69 | **1424.16 (1.714×)** | 2540.78 (**0.961×**) |
| 8T 2304² | 97.50 | **89.05 (1.095×)** | 95.57 (1.020×) |
| 8T 4608² | 701.29 | **522.35 (1.343×)** | 682.18 (1.028×) |

**Restricting the column range in place buys essentially nothing — and at
1T/4608² it is a net loss.** The copies are not overhead to be optimised
away; they *are* the optimisation.

The mechanism, once seen, is the one every packed GEMM already knows.
Restricting `x` to a tile does not change which cache lines the kernel walks:
the six planes are still full-width, so a 16-row group at tile width is
sixteen contiguous runs separated by `width × 4 B` (18 KiB at 4608²), and the
prefetchers see six strided streams exactly as before. **Staging the tile into
a compact `rows × (tile + 2R)` buffer is what makes the accesses dense** — the
copy in is one linear prefetch-friendly sweep, the kernel then runs entirely
inside a small hot buffer, and the copy out is another linear sweep. Locality
comes from the *layout*, not from the loop bounds.

**The x-range refactor was therefore deleted, not parked** — 16 kernel bodies
reverted to their committed form. It is recorded here because the hypothesis
was well-motivated, the experiment was cheap once the A/B knob existed, and
the negative result is the useful part: **anyone reaching for "tile the loop"
on this pipeline should reach for "pack the tile" instead.**

Same pass, same conclusion for the **activity chain**. §23.5 of the previous
revision claimed +1.5 % at 21 MP for tiling `box_blur_1pass_into`. Isolated
properly — H and A as independent arms in one binary, more repetitions — it is
a wash:

| cell | none | H only | H + A | A only |
|---|---:|---:|---:|---:|
| 1T 2304² | 345.61 | **274.25 (1.260×)** | 285.48 (1.211×) | 351.12 (0.984×) |
| 1T 4608² | 2537.03 | 1446.11 (1.754×) | **1404.08 (1.807×)** | 2454.07 (1.034×) |
| 8T 2304² | 115.10 | 110.41 (1.042×) | **105.07 (1.095×)** | 118.06 (0.975×) |
| 8T 4608² | 720.69 | **579.89 (1.243×)** | 587.07 (1.228×) | 735.91 (0.979×) |

The A-only column is 0.975–1.034× — noise. H+A against H alone is positive in
two cells and negative in two. **The activity tiling was removed**, and the
earlier +1.5 % is superseded: it was a two-cell reading of a wash.

### 23.6 The shipped shape, measured across threads AND sizes

What remains is exactly one change: the phase-A fused H blur, column-tiled
with packing, inside the rayon row band as well as the serial path (the axes
compose — the band gives a thread its rows, the tile keeps that band's
6-plane window dense; each worker owns its own thread-local arena).

Min over 5–7 process starts per cell, min of 3–15 walks per process,
CCD-pinned, arms interleaved, byte-identical env blocks:

| threads | 576² | 1152² | 2304² | 4608² |
|---:|---:|---:|---:|---:|
| **1** | 1.001× | 0.997× | **1.151×** | **1.733×** |
| **8** | 1.008× | 0.935× | 1.064× | **1.234×** |
| **16** | 0.989× | 1.018× | 0.944× | **1.109×** |

**Read the two left columns first — they are the control.** At 576² and 1152²
the tile (1536) is wider than the image, so no tiling happens and both arms
run the *identical code path*: those cells must be 1.000×, and what they
actually report is the measurement's own noise floor — **±0.3 % at 1T, ±1.8 %
at 16T, and as much as 6.5 % at 8T** (the 8T/1152² cell reads 0.935× on
identical code). Threaded cells carry a thread-placement lottery on top of
§22.5's layout one, and min-over-7 does not fully remove it.

Against that floor:

* **Established wins:** 1T from 2304² up (1.151× / **1.733×**, against a
  0.3 % floor), and 4608² at every thread count (1.234× @8T, 1.109× @16T).
* **Not established either way:** 2304² threaded (1.064× @8T, 0.944× @16T) —
  both inside their own cells' noise floor.
* **No established regression anywhere.** The 0.891× regression reported in
  the previous revision of this section was **the packed ACTIVITY chain, not
  the H blur**; with the activity tiling removed, the H-only arm is ≥1.04× at
  every 8T cell measured.

So the earlier "two thresholds, conditioned on width and on threading" is
withdrawn. One condition remains, and it is structural rather than tuned: the
tile does nothing when `width <= tile`, which is why the small buckets are
exactly the untiled code path.

### 23.7 What is next, in value order

1. **The zero-copy form, which is now the blocker for everything else.**
   §23.5 shows the remaining items (`fold` 28.3 %, `planesA` residue 19.1 %)
   cannot be reached by a copying tile. Two shapes do it:
   a **`stride` parameter** on the blur entry points (row length stays
   `width`, row base becomes `y * stride`), or a **column-slab walk** — copy
   only `src`/`dst`/`refy` columns into slab-width buffers once per slab and
   run phase A *and* phase B entirely at slab width, so the ~13 plane buffers
   are allocated slab-wide and **nothing is ever copied out** (the kernels
   accumulate; the planes never need to be full-width, because nothing outside
   the strip reads them). The slab form needs no kernel signature changes at
   all, at the cost of an x-offset argument for `blockiness_sparse_strip_wide`
   (whose lattice is in global x) and a per-slab X/B activity stash.
2. **Delete the copies from the H blur too.** `fused_blur_h_ssim_column_tiled` copies six planes
   per tile purely to avoid touching the four hand-written tier bodies. A
   `stride` parameter (row length stays `width`, row base becomes `y*stride`)
   removes every copy — and the workspace's own pixel-buffer rule already says
   multi-row functions must take stride natively, at no cost on the packed
   path. This is the *right* API, not just the faster one.
3. **The parallel path.** `fused_blur_h_ssim_banded`'s rayon arm is untouched;
   it row-bands across threads and each band still runs full-width. Tiling
   composes with it (tiles inside a band), and the shrunken working set should
   help contention more than it helps 1T.
4. **`TILE_WIDTH` is then a constant with provenance**, not a knob: 512–1536
   are within noise of each other at 2304², so the choice should be made on
   the 4608² and threaded cells, and re-derived per the workspace sweep rule
   rather than fixed at whatever won once at one size.

---

## 24. The packed column slab was measured BEFORE it was built — and its premise is false

The slab was specified in §23.7 and directed as the next build: pack
`src`/`dst`/`refy` into slab-width buffers once per slab, run phase A *and*
phase B at slab width, never copy out. Its stated targets were the two
remaining leaders at 4608² — **`fold` (28.3 %) and `planesA` (19.1 %)** — on
the reasoning that both "have the same width disease" as the H blur.

**One of those two does not.** The premise came from this lane's own report
and was never measured; the width probe that found the H blur's disease
(§23.1) measures every other phase for free, and it says:

Same 5.31 MP, 1T, `setarch -R`, untiled, only the aspect ratio changes:

| phase | w=2304 | w=1152 | w=576 | width-driven headroom |
|---|---:|---:|---:|---:|
| `blur_h` | 131.96 | 35.34 | 31.75 | **−100.2 ms (76 %)** — already taken |
| **`fold`** | **79.86** | **77.33** | **75.01** | **−4.85 ms (6.1 %)** |
| `v2:planesA` | 39.31 | 27.37 | 26.08 | −13.23 ms (33.7 %) |
| `v2:planesApp` | 18.81 | 16.07 | 15.67 | −3.14 ms (16.7 %) |
| `v2:dense` | 26.76 | 27.06 | 27.40 | none — **worse** at narrow width |
| `v2:gradient` | 18.41 | 19.42 | 21.22 | none — **15 % worse** at w=576 |
| `v2:append` | 15.82 | 15.85 | 16.04 | none |
| `v2:blockiness` | 4.66 | 4.69 | 4.64 | none |
| producer | 35.62 | 35.40 | 35.18 | none |

**The fold is not width-diseased: 6.1 %, ~4.9 ms at 5 MP.** It is 28 % of the
walk because it is 28 % of the *work*, not because its planes miss cache. Its
per-pixel cost does drift (15.04 ms/MP at 5 MP against 17.6 at 21 MP, 1.17×),
so a slab would recover something at 21 MP, but nothing like the H blur's
8.5×.

So the slab's whole available prize is **`planesA` + `planesApp` ≈ 16.4 ms of
a 302 ms tiled walk at 5 MP (5.4 %)**, against which it must pay: a three-plane
copy-in per slab, and **a measured penalty on the pointwise kernels**, which
get *slower* at narrow width — `gradient` +15 % at w=576, `dense` +2.4 %.
Those kernels are 66 ms at 5 MP, so a 5 % penalty there is −3.3 ms and eats a
fifth of the prize before the copies are counted. It also needs a
`stride`/`width` pair threaded through six phase-B kernels **and** the fold,
because a column slab's halo is interleaved in every row and cannot be sliced
off the way a row band's halo can — the "no kernel signature changes" claim in
§23.7 was wrong for exactly that reason.

**Not built.** A 5.4 % ceiling that arrives already reduced, behind a
signature change across seven kernels, is not the next move — and the same
probe says what is.

### 24.1 Where the fold's mass actually is: the v1 pools, 41.2 ms

Decomposing the fold by pool mode (2304², 1T, `setarch -R`, untiled):

| arm | `V1PoolsMode` | `fold` | Δ |
|---|---|---:|---:|
| `944full` | Full | **78.79** | — |
| `924` | Off | **37.59** | **−41.20** |
| `372` (`v1_only`, self-blur) | Full | 78.15 | — |

**The masked/IW/soft-peak pool pass is 41.2 ms — 52 % of the fold, and 13.6 %
of the whole tiled 5 MP walk.** That is the single largest remaining item in
the walk, it is *feature arithmetic* rather than layout, and no amount of
tiling or slabbing touches it. It is squarely an **item E/F question** (drop
set / redefinition): the 11 masked/IW pool pairs cost 13.6 % of every compare,
and whether each earns that is exactly the measured table the charter asks
for. Priced here so that table has a cost column before it has a utility one.

### 24.2 And a bit-identical dedup the fold is sitting on

`fused_vblur_features_ssim`'s own doc comment records that its `mu1_out` /
`mu2_out` / `ssq_out` / `s12_out` are "**the exact planes
`box_blur_v_from_copy` would produce**", bit-identically, because each
column's V blur is an independent scalar recurrence (with one stated caveat:
planes under 7 rows). In the **944** walk, phase A has *already computed those
four planes strip-wide* for the v2 kernels — and then every fold band V-blurs
the same four planes again over its own 42-row window (1.31× redundancy).

The 944 walk therefore V-blurs the same four moment planes **twice**. The
non-pool part of the fold is 37.6 ms at 5 MP, of which the V blur is a
substantial fraction, so the dedup is worth roughly 15–20 ms (5–6 % of the
tiled walk) and — unlike everything else in §23–24 — is a candidate to be
**bit-identical, needing no era at all**. The activity blur is NOT part of
this: v1 mirror-clamps it at its own band edges, which is the genuinely
different-semantics case the v2-block lane recorded as not removable.

Ranked against the slab's reduced 5.4 %, this is the better next build: bigger,
byte-neutral, and confined to one call site rather than seven signatures.

### 24.3 Cross-lane: numbers for the blur radius / locality / branches lane

For `benchmarks/blur_radius_locality_branches_2026-08-31.md` (not yet on
`main` at the time of writing), so it is not re-derived there:

* **Rolling-window locality overlaps §22 and the answer is negative.**
  Band-local phase A, measured at three band heights with a bit-identical
  control: `B=32` **+15.6 %**, `B=64` **+5.5 %**, `B=128` (control, 956/956
  bit-identical) +2.5 % — so **+13.1 % / +3.0 % / 0** net of plumbing, monotone
  in halo redundancy. The rolling row window needs the row-major running-sum V
  blur that the v2-block lane measured at **+9 %** and reverted (its L3).
* **Radius changes this lane's halo arithmetic directly.** Phase A's closure is
  `±2·BLUR_RADIUS` (activity = `blur(|src − blur(src)|)`), so a 32-row band
  costs `(32 + 4R)/32` — **1.625× at R=5, 1.50× at R=4, 1.375× at R=3**. Row
  banding does not become viable until R≈2 (1.25×), and even then it must beat
  a measured 1.25× per-unit efficiency gain, so **a radius cut does not rescue
  §22**. The column closure is `±R`, where the same cut is worth almost
  nothing (a 1536-wide tile goes 0.65 % → 0.26 % redundancy) — **column tiling
  is radius-insensitive; row banding is radius-limited.**
* **Branch behaviour at row tails/edges:** the H blur's rem-ring warm-up
  (`x >= diam`) and the `add_idx`/`rem_idx` mirror selects are the per-column
  branches in the packed tile path, and packing multiplies the number of tile
  edges by `width / tile`. That is measurable with the `ZENSIM_H_TILE` knob
  already on `main` (`0000` vs `0512`/`1536`) — a 3× change in edge count on
  the same binary.
* **Protocol, if these are measured at 2304²:** ASLR alone makes this walk
  bimodal over **10.1 %** (§22.5). Use `setarch -R`, one binary with runtime
  arms, byte-identical env-var lengths, interleaving, and an
  identical-code-path control cell; the noise floor is ±0.3 % at 1T and up to
  **6.5 % at 8T**.

### 24.4 Priced: phase A's four V sweeps are 22.05 ms, and the fold already has those values in registers

§24.2 guessed 15–20 ms for deduplicating the fold's V blur. Reading the kernel
first changes the direction of the dedup, and measuring it sizes the real one.

`fused_vblur_features_ssim` does not run a V-blur *pass* — it maintains the
four column recurrences in registers while it streams the features, and it
already exposes `mu1_out` / `mu2_out` / `ssq_out` / `s12_out` behind
`store_mu` / `store_sigma`. In the 944 walk with `V1PoolsMode::Full` it is
**already storing all four** into `FoldPoolScratch` for the pool replay. So
removing the fold's V blur would save only ALU, not a memory pass — the
§24.2 estimate was wrong in the useful direction: **the duplicate to remove is
phase A's, not the fold's.**

Priced with a temporary timing probe (skips the sweeps and lets the outputs be
garbage — timing only; the probe was removed in the same commit that recorded
this), 2304², 1T, `setarch -R`, untiled:

| arm | `v2:planesA` | Δ |
|---|---:|---:|
| all four V sweeps | **39.82** | — |
| skip `mu2` / `ssq` / `s12` | **23.13** | **−16.69** |
| skip all four | 17.77 | −22.05 |

**The four `box_blur_v_from_copy` sweeps are 22.05 ms — 55 % of `planesA`.**
Three of them (`mu2`, `ssq`, `s12`) are **16.69 ms = 5.5 % of the 302 ms tiled
walk at 5 MP**, and `planesA` is width-diseased, so the share grows with size
(it is 222.7 ms at 4608²).

`mu1`'s sweep (5.35 ms) must stay: the activity chain needs `mu1` on the
`±BLUR_RADIUS` halo rows *outside* the strip, which the fold — banded over
strip rows only — never produces.

**The build this implies**, ranked above both the slab and §24.2:

1. Run the v1 fold **before** the other phase-B kernels (it is already inside
   `stream_phase_b`, currently third of five).
2. Point its existing `mu2_out` / `ssq_out` / `s12_out` stores at the
   strip-wide plane buffers at the band's row offset instead of (or as well
   as) the band-local `FoldPoolScratch`, so phase A can skip those three
   sweeps entirely.
3. Keep `mu1`'s sweep for the activity halo.

**Bit-identity is plausible but not free**, and is the thing to gate rather
than assume: the fused kernel's outputs are documented bit-identical to
`box_blur_v_from_copy` *from the same inputs*, and `folded720_v1_pools_match_
v1_path` already compares those pool slots to v1's 372 with `to_bits()`. The
open question is the **plane edges**: the fold V-blurs a band buffer clamped
to the plane and mirrors at its edges, while phase A V-blurs the wide window
whose halo the producer reflect-pads. Interior strips have real rows on both
sides and must agree by construction; the top and bottom strips are where the
two mirror conventions have to be shown to coincide. That is a test, and it
is the first thing the build should write.

**And one obstacle, worked out here so the next pass does not hit it blind.**
Step 2 cannot simply hand the fold a strip-plane slice covering its whole band
BUFFER. The buffer is `[b0 − V1_BAND_OVERLAP, b1 + V1_BAND_OVERLAP)` and the
kernel stores at buffer-row coordinates, so adjacent bands' buffers overlap by
`2 × V1_BAND_OVERLAP = 10` rows — and those overlap rows are exactly the ones
each band's own V blur mirrors at its buffer edge, i.e. contaminated. Writing
whole buffers in ascending band order lets band `k+1` overwrite band `k`'s
last 5 GOOD rows with its own contaminated top halo; descending order breaks
the first 5 instead. This is the same overlap-corruption that killed in-place
row banding in §22.1, in a different costume.

Two ways out, with different prices:

* **Copy the inner rows out** after the fused call (3 planes × `inner_h` rows
  per band, read from a band scratch that is L2-hot). Needs no kernel change.
  It saves the H-plane *read* and the V-blur ALU but still pays the plane
  write, so it recovers roughly **half** of the 16.69 ms.
* **Give the fused kernel an inner-only store offset** (store row `y` to
  `out[(y − inner_start) · width]`), so each band writes a disjoint strip-plane
  range in place. Recovers the full 16.69 ms, but touches the store sites in
  all six `fused_vblur_ssim_inner` tier bodies — and §23.5's lesson says to
  price that against the copy, not assume the no-copy form wins.

**Not yet built** — it is specified here with its price, its obstacle, and two
costed exits, so the next pass starts from a number rather than a guess. That
is the same discipline that stopped the slab, and the ~16.7 ms (5.5 %) ceiling
should be weighed against §24.1: **the v1 pools are 41.2 ms, 2.5× larger, and
no layout change touches them.**

---

## 25. `ZENSIM_H_TILE` default-on: what flipping it costs and gains

Queued as the decision after the slab. The data is §23.6's table, re-read as a
flip decision.

| threads | 576² | 1152² | 2304² | 4608² |
|---:|---:|---:|---:|---:|
| **1** | 1.001× | 0.997× | **1.151×** | **1.733×** |
| **8** | 1.008× | 0.935× | 1.064× | **1.234×** |
| **16** | 0.989× | 1.018× | 0.944× | **1.109×** |

**What it costs: nothing below the tile width, structurally.** The tile is a
no-op when `width <= tile`, so at 576² and 1152² both arms execute the *same
instructions* — those cells are not "small regressions", they are the
measurement's noise floor (±0.3 % at 1T, ±1.8 % at 16T, up to 6.5 % at 8T),
and they bound how much any other cell in the same row can be trusted.

**What it gains:** 1.151× / 1.733× at 1T from 2304² up, and 1.234× / 1.109×
at 4608² on 8 and 16 threads. The 2304²-threaded cells (1.064× @8T, 0.944×
@16T) sit inside their own rows' floors and are unresolved in either
direction; resolving them needs n ≥ 15 process starts, and it is **not a
blocker**, because the worst case there is bounded by the floor that the
control cells measure.

**Recommendation: flip it on unconditionally as part of era-2, with no width
threshold.** A threshold would only re-express what the code already does for
free (`width <= tile` ⇒ untiled), and every cell where tiling actually runs is
a win or inside noise. It **cannot** flip standalone: the running sum along x
restarts per tile, so it is a byte change and rides the era-2 golden re-pin
with the oracle bounds, vendor probe and rank-preservation gate like
everything else in the break.

**On `TILE_WIDTH` as a constant.** 512 / 1024 / 1536 are within noise of each
other at 2304² (271.25 / 271.64 / 270.57), so the value is not critical, which
is itself the useful finding — but it should be *derived*, not picked: the
quantity that matters is the H blur's live window,
`H_BLUR_ROW_GROUP × 6 planes × tile × 4 B`, which wants to sit inside L2.
At the 1 MiB L2 on this part that is `tile <= 1365`; **1024** leaves headroom
on parts with 512 KiB L2 and is inside the measured-flat band. Fixing it at
1536 because 1536 is what the sweep happened to use would be exactly the
"hand-distilled from N=1" anti-pattern the workspace rules name.

### 24.5 The two exits, PRICED — exit 1 recovers 80–86 %, and my "about half" was wrong

Round 29's directive was to price the two exits rather than assume, which is
this lane's own rule and which §24.4's estimate had already broken ("recovers
roughly **half**"). Priced with a contained timing probe — phase A skips the
`mu2`/`ssq`/`s12` sweeps in both arms; **exit 1** additionally performs the
three inner-row copies a redirect would make, **exit 2** performs none.
Outputs are garbage in both arms by construction; this measures time. 1T,
`setarch -R`, CCD-pinned, `H_TILE` off so the sweeps are the only variable.

| | 2304² walk | Δ | `planesA` | `fold` | 4608² walk | Δ | `planesA` | `fold` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 330.48 | — | 39.27 | 80.18 | 1939.21 | — | 248.50 | 382.57 |
| **exit 1** (copy) | **318.51** | **−11.97 (−3.6 %)** | 22.42 | 81.62 | **1840.03** | **−99.18 (−5.1 %)** | 139.21 | 394.74 |
| **exit 2** (no copy) | 315.50 | −14.98 (−4.5 %) | 22.51 | 78.06 | 1824.38 | −114.83 (−5.9 %) | 139.28 | 382.17 |

**`planesA` falls 39.27 → 22.42 (−16.85 ms) and 248.50 → 139.21 (−109.3 ms)**,
reproducing §24.4's −16.69 ms independently. The copy shows up exactly where
it should — in `fold`, +1.44 ms at 2304² and +12.17 ms at 4608² — and costs
**3.0 ms / 15.7 ms** of walk.

**So exit 1 captures 80 % (2304²) and 86 % (4608²) of exit 2's win, for zero
kernel change.** My §24.4 estimate of "roughly half" was wrong because it
assumed the copy costs as much as the plane write it replaces; it does not —
the copy's source is an L2-hot band buffer, while the sweep it removes was a
cold full-plane *read* plus the same write.

**Decision: build exit 1.** Exit 2's extra 3.0 / 15.7 ms would cost an output
row-offset parameter threaded through all six `fused_vblur_ssim_inner` tier
bodies, and §23.5 is the standing precedent for not assuming the no-copy form
wins — there, the no-copy form lost outright. Exit 2 is registered as a
follow-up with its price attached, not as the primary.

**Two implementation facts that cost time to establish, recorded so they are
not re-derived:**

1. **§24.4's overlap obstacle is smaller than stated.** The fused kernel
   writes **only the inner rows** of its `mu1_out`/`mu2_out`/`ssq_out`/
   `s12_out` (its own comment says so, and the pool replay then reads exactly
   `inner_start .. inner_start + inner_h`). So bands write disjoint strip-plane
   rows even though their band *buffers* overlap by `2 × V1_BAND_OVERLAP`. The
   corruption I predicted only occurs if a caller hands the kernel a
   whole-buffer output slice **and** the kernel wrote halo rows — it does not.
2. **The blocker is a BORROW, not the algorithm.** `stream_phase_b` takes
   `scr: &ScratchV2Strip`, and its `src_win`/`dst_win` arguments are produced
   by `stream_windows_shared(.., scr)` — i.e. they may borrow `scr.src_wide` /
   `scr.dst_wide`. Taking `&mut scr` to write `mu2`/`ssq`/`s12` conflicts with
   that live immutable borrow. The fix is a **field-level destructure at the
   call site** (`let ScratchV2Strip { src_wide, dst_wide, mu2, ssq, s12, .. }`)
   with the window selection inlined there, so the raw windows and the three
   moment planes are disjoint field borrows. That, plus moving the fold to
   first among the phase-B kernels (it is currently third of five; the
   accumulators are independent, so the move is byte-neutral), is the whole
   build.

Also note the redirect only applies where the fold actually produces the
planes: `V1PoolsMode::Full` (⇒ `BandPoolWork::Full` ⇒ `store_sigma`), which is
the 944 product mode. Any other pool mode keeps phase A's sweeps, and the
parallel band path needs the per-band output slices chunked disjointly before
it can opt in.

---

## 26. Item D: the compute-set descriptor — landed internally, public surface PROPOSED

**What landed** (`ComputeSet` in `feature_v2.rs`, `pub(crate)`): one derivation
of *what a request computes*, replacing six ad-hoc locals that were recomputed
from `V2NewFeatureToggles` at the top of `foldapp_streaming_walk` and
re-derived again inside the strip loop. The walk now reads `compute.v2_blocks`
/ `.append` / `.append2` / `.self_blur_eligible()` / `.plane_needs(self_blur)`
instead of writing `&& v2_blocks` out at each site.

**Behaviour is unchanged, and that is gated rather than asserted.**
`compute_set_matches_legacy_derivation` sweeps **1,024 combinations** (256
toggle bit-patterns × 4 `V1PoolsMode` values) and checks each against the
legacy expressions written out verbatim, plus two invariants the old code only
implied: a `v1_only` request forces **every** v2-era block off whatever else
the caller set, and `append2_dst_activity` cannot outlive `append2`.

**Why this is the right home for two things the charter asks for.**

* **Per-model drops (item E).** §24.1 measured the v1 masked/IW/soft-peak pool
  pass at **41.2 ms — 13.6 % of the tiled 5 MP walk** — and the frontier lane
  measured its rank cost as **exactly 0 for the 944 MLPs** and **0.399 CID22
  for `B`**. A global drop is therefore wrong in both directions. The shipping
  form is `ComputeSet::from_block_profile(model)`: read the model's own block
  profile and compute only what it can read. That constructor is **not added
  here** — it needs a `zenpredict::Model`, which is a question about which
  crate owns the derivation, and it should be answered before it is coded.
* **HDR toggles.** The HDR append is a future regime; its blocks become
  **fields on this struct, append-only**, exactly as feature numbering is. The
  HDR front end already reaches the walk (`FrontEnd::Hdr` selects the BANDVIS
  deltas and the `hl_bins` lane), so the wiring exists — what was missing was a
  place to put "this request wants the HDR blocks" that is not another boolean
  on the public request type.

### 26.1 Proposed public surface — for approval, NOT taken

`V2NewFeatureToggles` remains the public request type and is **unchanged**.
Nothing below is implemented; it is listed so the API change can be approved
or refused as a unit, per the repo's API rule.

| item | shape | why |
|---|---|---|
| `pub struct ComputeSet` | promote the `pub(crate)` type | lets a caller state a compute set directly instead of encoding it in toggle booleans |
| `ComputeSet::from_toggles(V2NewFeatureToggles)` | already exists | the compatibility path; keeps every current caller working unchanged |
| `ComputeSet::from_block_profile(&Model)` | new | **item E's shipping form** — derives the drop set from the model, so the 13.6 % pool pass is skipped exactly when it is worth 0 |
| `compute_*_with_set(.., ComputeSet)` | one new entry per existing extraction entry | the only way to *use* a compute set that was not derived from toggles |

The cheapest version that unlocks item E without any of the above is to keep
everything `pub(crate)` and derive the compute set **inside** the existing
entry points from a model handle the caller already passes. That needs no new
public types at all, and is the recommendation.

### 26.2 What item D does NOT do

It does not change what is computed, and it is not a perf change — the six
locals it replaces were already correct. Its value is that **items E and F are
expressible**: before it, "drop the pool pass for this model" had no place to
live except another public boolean, and "add the HDR blocks" had the same
problem. The measured levers are in §23–§25 and
`benchmarks/era2_drop_redefine_table_2026-08-31.md`; this is the vehicle.
