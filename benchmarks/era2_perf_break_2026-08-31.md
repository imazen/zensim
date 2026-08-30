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

`dense_block_kernel_generic` (`feature_v2.rs:2190`) is **23.2 % of the 944-full
walk** (125,098,677 of 540,211,635 net Ir at 576²) and gets 3-way channel
parallelism only. It cannot be split further without changing bytes, because
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

## 2. The era-2 accumulation shape

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

Four new gates, plus the existing suite. Each is stated as what it proves, not
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
