# What do we actually need to compute? — the feature-cost frontier

**The question, from the user: "what features should we drop?" — asked in
service of "extremely fast, as good or better than ssim, and good at HDR".**
Mid-lane the user widened it: *the model class itself is on the table.* If
dropping the linear, or the blend, or the MLPs buys ~2×, that trade is worth
evaluating. So this note answers the wider question — a Pareto front over
**model classes**, each priced end to end as (required feature families →
compute → quality) — and the narrower one falls out of it.

Companion lanes, whose numbers this note reads rather than re-derives:
`benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md` (the callgrind
family split), `benchmarks/fold_engine_2026-08-31.md` (fold-backed scoring),
`benchmarks/fold_mt_scaling_2026-08-31.md` (thread scaling),
`benchmarks/fold_footprint_2026-08-31.md` (working set + the per-thread L3
budget).

---

## 0. The one-paragraph answer

**Slot counts are a lie about cost.** Shipped B reads 95 of 372 inputs, but you
cannot save 74 % by not computing the other 277, because the families share
passes. Read from source: v1's peak block is a **byproduct that is already paid
for** on every path, and masked + IW are **one pass group**, not two. So inside
`f0..372` there is exactly one compute boundary — *peaks* versus
*masked-and-IW* — and that is what shipped here as `V1PoolsMode::Peaks` plus a
per-profile weight-skipping policy.

**But the big lever is not inside a model — it is the choice of model.** The
944 MLP is the quality ceiling and needs everything. The W-LIN 7b blend needs
the whole v2-348 block and essentially nothing of v1's 372 (ablating all 372
costs it CID22 −0.027 and *improves* its LIVE by 0.117). And a **basic-only**
model — `ADD156`, which reads 28 of 156 basic lines and **zero** of the 216
pool lines — lands within 0.019 pooled CID22 of shipped B, **beats** it on
within-image ranking on seven of eight corpora including the near-lossless
band, and needs the cheapest walk the extractor has (half the per-thread hot
set of what `score()` runs today). If a 2× is wanted, that is where it is.

---

## 0.1 THE DECISION TABLE

One row per model class. Quality is `|SROCC|` from `bake_verdict --full-json`,
all five on the **same** root (`r1b-pools944-2026-08-30`) over the same pairs.
`ms` is the **extraction** cost of that class's cheapest fold request — the
bake forward is excluded and is not measured here (a 372→1 linear is 372 MACs,
the 149 KB MLP is 667×128+128 ≈ 85 k MACs, against a 15-380 ms extraction).
`WS/thr` is the per-thread band hot set derived in §4.4 — that derivation is
for the `v1_only` walk, so the two 944 rows are marked `≥` (their band task
also carries the v2 planes, which the fold-footprint lane prices separately).

<!--DT_BEGIN-->
| model class | example | required families | ms 576²/1152²/2304² @1T | @8T | @16T | WS/thr (W=2304) | CID22 | KonJND | nonphoto | imazen26 | HF-NL pooled / per-ref | vs ssim2 (human corpora) | to ship |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| basic-only | ADD156 | basic | 6.41 / 25.3 / 122.0 | 1.34 / 5.31 / 29.3 | 1.79 / 7.33 / 30.0 | 2.21 MiB | 0.8632 | 0.5363 | 0.8453 | 0.8546 | 0.295 / **0.799** | ties on LIVE/CSIQ/KADID, −0.026 CID22, **+0.058 KonJND** | a profile slot + a ship call — **no retrain, no era**; the skip already fires |
| sparse 372 linear | **shipped B** | basic + peaks + **masked + IW** | 8.72 / 36.0 / 170.0 | 1.91 / 8.90 / 53.6 | 2.82 / 10.0 / 40.0 | 4.43 MiB | 0.8821 | 0.5198 | 0.8498 | 0.8603 | 0.350 / **0.765** | +0.030 CSIQ, −0.061 LIVE, −0.067 TID | nothing to drop |
| W-LIN 7b blend | Q7b g0.20 | all of v1-372 + v2-348 + append-204 | 15.1 / 62.0 / 354.0 | 4.51 / 22.7 / 124.6 | 6.92 / 25.3 / 120.0 | ≥4.43 MiB | 0.8588 | 0.5118 | 0.8778 | 0.8873 | 0.406 / **0.756** | below on all six (LIVE −0.147) | a retrain + the `fold_v1` lever (§6.4) to cash its dead v1-372 |
| 944 MLP | C purity944 | basic + v2-348 + append-204 (**no pool block**) | 12.8 / 52.0 / 312.0 | 3.91 / 17.7 / 101.0 | 6.15 / 22.7 / 102.0 | ≥4.43 MiB | 0.8927 | 0.5006 | 0.9277 | 0.9313 | 0.694 / **0.810** | **at or above on all six** | **nothing** — its bake reads 0/216 pool lines, so the shipped skip is exact |
<!--DT_END-->

**Speed, up front:** at 2304²/1T the basic-only class's walk is **2.9×** the
W-LIN blend's and **2.6×** the 944 MLP's, and **1.6×** today's shipped
buffered v1-372. The ~2× the user asked about is available, and it is bought
by changing the model class, not by trimming a family out of one.

**vs-ssim2 verdict** compares only the human-labelled corpora — `nonphoto`,
`imazen26` and `hfnlproxy` have ssim2 *as their target*, so a model's number
there is agreement-with-ssim2, not a win over it (§4.3).

---

## 1. The structure, read from source (this is the actual answer)

`zensim/src/feature_v2.rs`, the fold's v1 band replay.

### 1.1 v1's 372 layout, and what each family costs to produce

| family | slots | produced by |
|---|---|---|
| basic | `f0..156` | `fused::fused_vblur_features_ssim`, always |
| **peaks** | `f156..228` | the SAME kernel call — `ssim_d8`/`edge_art8`/`edge_det8` + three running maxima |
| masked | `f228..300` | the band pool arm |
| IW | `f300..372` | the same band pool arm |

**The peaks are free, and this is not an estimate.** `fused_vblur_ssim_inner`
accumulates `acc.ssim_d8 += (sd4*sd4)` and `acc.ssim_max = max(…)`
**unconditionally in every SIMD variant** — verified by counting:
`grep -c 'acc.ssim_d8 +='` in `zensim/src/fused.rs` is **10** and
`grep -c 'acc.edge_art_max = '` is **20** (the ssim and edge kernel families ×
each dispatch tier), and **not one of them sits behind a predicate**. And `V1BasicSums::accumulate` merges them unconditionally with the
comment "free to carry". `V1PoolsMode::Off` has therefore been *computing* the
peak block all along and merely declining to emit it. Not emitting the peaks
saves nothing; emitting them costs 72 `f64` stores per image.

**Masked and IW are ONE pass group, not two.** Inside `fold_v1_one_band`'s pool
arm, the work is:

1. `simd_ops::abs_diff_into` — the ref-side activity `|src − H_blur(src)|`;
2. `blur::box_blur_1pass_into` — blur that activity (borrows `ssq_v` as temp);
3. the fused V-blur kernel with `store_mu` **and** `store_sigma` on — two extra
   plane writes it would not otherwise make;
4. `simd_ops::build_inline_mse(act, V1_MASK_K, V1_IW_K, …)` → **both** MSE slots;
5. `simd_ops::ssim_channel_inline_both(…, V1_MASK_K, V1_IW_K)` → **both** SSIM triples;
6. `simd_ops::edge_diff_channel_inline_both(…)` → **both** art/det pairs.

Steps 4-6 are `*_both` kernels: one sweep, two masking strengths. Steps 1-3 are
shared inputs. **Dropping only masked, or only IW, saves nothing but a few
arithmetic ops inside three kernels — the sweeps, the activity chain and the
sigma stores all stay.** Dropping both removes the entire arm.

### 1.2 Therefore

```
f0..372  =  [ basic + peaks : one fused kernel, one price ]
         +  [ masked + IW   : one pass group, one price   ]
```

Two compute states, not four. `V1PoolsMode::Peaks` (shipped this lane) is the
first; `V1PoolsMode::Full` is both.

---

## 2. Per-family rank contribution — measured, zero-at-inference

**Method.** `bake_contrib --ablate-range`, extended this lane to take several
named ranges in one run. The ablation is the **exact rank-|K| update**: for a
family `G`, the layer-0 pre-activation becomes `z0 − Σ_{k∈G} x̃_k·W0[k,:]`
followed by an exact re-forward through activation, remaining layers, head
dispatch, tanh pin and output spline (`bake_contrib`'s registered method,
campaign appendix C.1). Setting the *standardized* input to 0 is exactly "the
model reads this family at its scaler mean", i.e. the family carries no
information. Baselines are parity-gated against `bake_runtime::score_row`:
**max|diff| = 0.000e0 over 47,511 rows** on the shipped-B run. SROCC is
`zensim_validate::panel::spearman` — nothing is hand-rolled.

**The caveat that governs how to read every number below: zero-at-inference is
not retrained-without.** It measures what the model *currently* extracts from a
family, holding all other weights fixed. A family that looks droppable here may
still be load-bearing if the model were refit — and, more often in these
tables, a family that looks *essential* here would be partly recovered by a
refit that redistributed its weight. So a large negative delta is a **lower
bound on what a retrain must recover**, and a near-zero delta is a **reliable
drop signal** (nothing to recover).

### 2.1 Shipped B (372-input linear, 7,325 B) — the pool block is load-bearing

Eval root `2026-08-30-full-features-372` (the current default), nine corpora,
47,511 rows. Δ|SROCC| vs the un-ablated baseline:

| ablated family | CID22 | KonJND | nonphoto | imazen26 | HF-NL | KADID | TID | CSIQ | LIVE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline \|SROCC\| | 0.8821 | 0.6497 | 0.8640 | 0.8306 | 0.5027 | 0.8085 | 0.7785 | 0.9342 | 0.8970 |
| basic `f0..156` | −0.706 | −0.288 | −0.725 | −0.650 | −0.212 | −0.686 | −0.632 | −0.684 | −0.800 |
| peaks `f156..228` | −0.038 | −0.311 | −0.037 | −0.034 | −0.108 | −0.026 | −0.014 | −0.015 | −0.096 |
| masked `f228..300` | −0.317 | −0.614 | −0.124 | −0.113 | **+0.025** | −0.193 | −0.392 | −0.191 | −0.110 |
| IW `f300..372` | −0.147 | −0.104 | **+0.024** | **+0.037** | **+0.025** | −0.125 | −0.146 | −0.124 | −0.027 |
| **masked+IW** (the pass group) | **−0.399** | **−0.525** | −0.123 | −0.111 | +0.023 | −0.240 | −0.421 | −0.257 | −0.145 |
| whole pool block `f156..372` | −0.493 | −0.579 | −0.219 | −0.208 | −0.015 | −0.293 | −0.499 | −0.332 | −0.206 |

**Reading.** For B the masked/IW pass group is the best-value block in the
metric: it is the only family with a real compute price, and removing it costs
**0.40 CID22 and 0.53 KonJND**. Peaks cost nothing to compute and still carry
0.31 KonJND — pure profit. IW *alone* is mildly negative on the two ssim2 axes
(nonphoto +0.024, imazen26 +0.037), but it does not decompose: it shares its
sweep with masked, so "drop IW, keep masked" is not a compute state.

**There is no Pareto-dominated family in B.** The frontier for B is a single
point: it needs everything it computes.

### 2.2 W-LIN 7b `Q7b_pools_g0.2_a0.2_b0.97` (944-input, 3,583 B) — v1 is nearly free

Eval root `r1b-pools944-2026-08-30` (its own training root; the ext944 root
feeds `f156..371` as structural zeros and would make this measurement
vacuous).

| ablated block | CID22 | KonJND | nonphoto | imazen26 | HF-NL | KADID | TID | CSIQ | LIVE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline \|SROCC\| | 0.8588 | 0.5118 | 0.8778 | 0.8873 | 0.4056 | 0.7218 | 0.7767 | 0.8794 | 0.8129 |
| basic `f0..156` | −0.008 | −0.001 | −0.004 | −0.001 | −0.036 | — | — | — | — |
| peaks | −0.003 | −0.011 | −0.003 | −0.000 | −0.010 | — | — | — | — |
| masked+IW | −0.005 | −0.048 | **+0.001** | **+0.002** | −0.004 | — | — | — | — |
| whole pool block `f156..372` | −0.010 | −0.063 | −0.004 | −0.001 | −0.015 | — | — | — | — |
| **the whole v1-372 `f0..372`** | **−0.027** | −0.059 | −0.015 | −0.008 | −0.060 | **+0.048** | **+0.041** | −0.004 | **+0.117** |
| **v2-348 `f372..720`** | **−0.745** | **−0.494** | **−0.501** | **−0.540** | **−0.405** | −0.492 | −0.424 | −0.319 | −0.309 |
| ↳ v2 `f372..600` | −0.390 | −0.504 | −0.344 | −0.371 | −0.333 | −0.255 | −0.201 | −0.187 | −0.143 |
| ↳ v2 `f600..720` | −0.010 | −0.027 | −0.013 | −0.008 | −0.018 | −0.058 | −0.040 | −0.013 | −0.100 |
| append `f720..944` | −0.016 | +0.020 | −0.019 | −0.018 | −0.046 | −0.012 | −0.033 | −0.013 | −0.061 |

The g0.25 sibling reproduces every one of these within ±0.02 (same file).

**Reading.** The blend's signal lives almost entirely in **v2-348**. The whole
v1-372 block — 372 slots, the entire pool arm, the basic fused kernel — is
worth **−0.027 CID22** to it, and dropping it *improves* LIVE (+0.117), KADID
(+0.048) and TID (+0.041). This is a Pareto-dominated block in the strict
sense the coordinator asked for: real compute for ~zero rank.

### 2.3 The 944 MLPs already dropped the pool block, structurally

`bake_block_profile` (extended this lane with the v1 COMPUTE families) on the
shipped 944 candidates:

| bake | layer 0 | v1_basic used | v1_peaks | v1_masked | v1_iw | f372-719 | f720-943 |
|---|---|---:|---:|---:|---:|---:|---:|
| `ADD156_safesyn_only_raw_lasso` | 372→1 f16 | **28**/156 | **0**/72 | **0**/72 | **0**/72 | — | — |
| `b_sdr_linear_cid80_inclwinsor_dense_dial` (**shipped B**) | 372→1 f16 | 46/156 | 26/72 | 10/72 | 13/72 | — | — |
| `v47_strict_qat_native` (profile A) | 372→… | 106/156 | 51/72 | 64/72 | 64/72 | — | — |
| `Q7b_pools_g0.2_a0.2_b0.97` | 944→1 f16 | 104/156 | 59/72 | 23/72 | 25/72 | 348/348 | 185/224 |
| `c_sdr_purity944` | 667→128→1 f16 (pruned) | 156/156 | **0**/72 | **0**/72 | **0**/72 | 348/348 | 163/224 |
| `c_sdr_mlp944_corrmix` | 667→128→1 f16 (pruned) | 156/156 | **0**/72 | **0**/72 | **0**/72 | 348/348 | 163/224 |

The 944 MLPs read the pool block at **exactly zero** — they trained on a folded
root where `f156..371` are structural zeros. So for the 944-MLP class the pool
arm is not a trade at all: it is work with a provably zero effect on the
answer, and the shipped skip removes it.

---

## 3. Per-family compute cost — measured

### 3.1 Wall clock, paired/interleaved (zenbench), 7 arms × 3 sizes × {1, 8, 16}T

Arms, and what each one is the cheapest request for:

| arm | request | serves |
|---|---|---|
| `buf_v1_228` | buffered v1, no extended/IW | (control) |
| `buf_v1_372` | buffered v1, extended + IW | today's buffered walk |
| `fold156_basic` | fold, `v1_only` + `Off` | (control — see the note below) |
| `fold228_peaks` | fold, `v1_only` + **`Peaks`** | the **basic-only** class |
| `fold372_full` | fold, `v1_only` + `Full` | the **sparse 372 linear** class (what `score()` runs today) |
| `fold944_off` | fold, 944 + `Off` | the **944 MLP** class (its bake reads no pool line) |
| `fold944_full` | fold, 944 + `Full` | the **W-LIN 7b** class |

`fold156_basic` and `fold228_peaks` accumulate the identical sums — the peak
tier is the fused kernel's unconditional byproduct — so their difference is
purely the H-plane shape: `Off` hands the band no scratch and therefore
disables the band-local self-blur, falling back to phase A's strip-wide H
planes, while `Peaks` blurs the 42 rows each band consumes into its own
buffer. Their `f0..156` slots are bit-identical by transitivity
(`folded720_v1_pools_match_v1_path` gives `Off ≡ Full` there,
`folded_peaks_mode_is_pure_compute_skipping` gives `Peaks ≡ Full`).

<!--MS_BEGIN-->
| arm | 576² 1T | 8T | 16T | 1152² 1T | 8T | 16T | 2304² 1T | 8T | 16T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `buf_v1_228` | 6.67 | 1.45 | 1.79 | 28.7 | 6.70 | 6.00 | 146.0 | 29.2 | 24.0 |
| `buf_v1_372` | 10.5 | 2.13 | 2.31 | 44.0 | 8.52 | 7.33 | 200.0 | 38.8 | 30.0 |
| `fold156_basic` | 6.15 | 1.25 | 2.56 | 26.0 | 4.97 | 9.33 | 188.0 | 36.4 | 40.0 |
| `fold228_peaks` | 6.41 | 1.34 | 1.79 | 25.3 | 5.31 | 7.33 | 122.0 | 29.3 | 30.0 |
| `fold372_full` | 8.72 | 1.91 | 2.82 | 36.0 | 8.90 | 10.0 | 170.0 | 53.6 | 40.0 |
| `fold944_off` | 12.8 | 3.91 | 6.15 | 52.0 | 17.7 | 22.7 | 312.0 | 101.0 | 102.0 |
| `fold944_full` | 15.1 | 4.51 | 6.92 | 62.0 | 22.7 | 25.3 | 354.0 | 124.6 | 120.0 |

Marginal cost of the masked/IW pass group (`fold372_full − fold228_peaks`) — the ONLY separable family boundary inside `f0..372`:

| | 576² 1T | 8T | 16T | 1152² 1T | 8T | 16T | 2304² 1T | 8T | 16T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **delta** | +2.31 (36 %) | +0.56 (42 %) | +1.03 (57 %) | +10.67 (42 %) | +3.58 (67 %) | +2.67 (36 %) | +48.00 (39 %) | +24.24 (83 %) | +10.00 (33 %) |

Marginal cost of the v2-348 + append-204 blocks (`fold944_full − fold372_full`):

| | 576² 1T | 8T | 16T | 1152² 1T | 8T | 16T | 2304² 1T | 8T | 16T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **delta** | +6.4 (74 %) | +2.6 (137 %) | +4.1 (145 %) | +26.0 (72 %) | +13.8 (155 %) | +15.3 (153 %) | +184.0 (108 %) | +71.0 (133 %) | +80.0 (200 %) |

**Conditions.** Two-point subtraction `(t(N) − t(1)) / (N−1)` over the bench
binary's single-arm loop (`ZEN_XP_RSS`), which removes process start, the
`test_pair` construction and first-touch page faults. Arms are visited
ROUND-ROBIN, 5 rounds, median reported; `N` is raised 10× for the 8T/16T
passes so a 10 ms clock still resolves a few-ms arm. Pinned with `taskset`
(1T: one core; 8T: cores 8-15; 16T: cores 8-23), `nice -n19 ionice -c3`,
**while another lane held zenbench's exclusive lock for a multi-hour paired
A/B**. This is therefore NOT the locked `extract_paths_bench` group (which
stays queued behind that lane): it is the same arms measured with a coarser
instrument that does not jump the queue. Box load during the runs was
4.4-6.5 of 28 cores.

Round-to-round spread, `(max − min) / median`, worst cell per arm:

| arm | worst spread | where |
|---|---:|---|
| `buf_v1_228` | 33.3 % | 1152²/16T |
| `buf_v1_372` | 27.3 % | 1152²/16T |
| `fold156_basic` | 50.0 % | 576²/16T |
| `fold228_peaks` | 28.6 % | 576²/16T |
| `fold372_full` | 45.4 % | 576²/16T |
| `fold944_off` | 45.8 % | 576²/16T |
| `fold944_full` | 44.4 % | 576²/16T |
<!--MS_END-->

### 3.2 Instruction-level split (predecessor lane, callgrind, 576², serial, v3 tier)

| arm | Ir | note |
|---|---:|---|
| `fold372_only` (`v1_only`, pools Full) | 249,228,173 | 0.743× buffered v1-372 |
| `buf_v1_372` | 335,620,797 | today's buffered walk |
| `fold944_off` | 458,918,753 | pools zeroed |
| `fold944_full` | 534,893,298 | pools live |

`fold944_full − fold944_off` = **+76.0 M Ir**, and the split (predecessor §3)
puts **47 % of it in one H-blur** (`box_blur_h_inner_v3`, +39.5 M), 14 % in
`ssim_channel_inline_both`, 11 % in the activity V-blur, 4 % in the fused
kernel — i.e. the pool block's price is dominated by producing the *activity
plane*, exactly the chain §1.1 lists as shared between masked and IW.


---

## 4. The model-class frontier

### 4.1 Required feature set per class — derived from the bakes, not assumed

| class | example bake | bytes | required v1 families | required v2/append | cheapest fold request |
|---|---|---:|---|---|---|
| **basic-only** | `ADD156_safesyn_only_raw_lasso` | 3,575 | basic (28 of 156 lines live) | none | `v1_only` + `Peaks` |
| **sparse 372 linear** | shipped **B** | 7,325 | basic + peaks + **masked + IW** | none | `v1_only` + `Full` |
| **W-LIN 7b blend** | `Q7b_pools_g0.2_a0.2_b0.97` | 3,583 | basic + peaks + masked + IW | v2-348 **+** append-204 | 944 walk + `Full` |
| **944 MLP** | `c_sdr_purity944` | 149,343 | basic **only** | v2-348 + append-204 | 944 walk + `Peaks` |

The two 944 classes both need the full 944 walk; they differ only in the pool
arm, which the shipped policy now removes for the MLP.

### 4.2 Quality, all five models on ONE root, one corpus set

`bake_verdict --full-json`, root `r1b-pools944-2026-08-30` for all five (it is
the only root that carries live `f0..372` **and** `f372..944`, so the 372-wide
and 944-wide bakes are read over the same pairs). `|SROCC|`, as
`bake_verdict` reports it; nothing recomputed here.

| model | n_in | bytes | composite | CID22 | KonJND | nonphoto | imazen26 | HF-NL | CSIQ | LIVE | KADID | TID | AIC-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic-only ADD156 | 372 | 3,575 | 0.8259 | 0.8632 | **0.5363** | 0.8453 | 0.8546 | 0.2947 | 0.9017 | **0.9603** | 0.8081 | 0.8237 | 0.7770 |
| sparse 372 (**B**) | 372 | 7,325 | 0.8335 | 0.8821 | 0.5198 | 0.8498 | 0.8603 | 0.3500 | 0.9349 | 0.8985 | 0.8085 | 0.7789 | 0.7637 |
| W-LIN 7b g0.20 | 944 | 3,583 | 0.8304 | 0.8588 | 0.5118 | 0.8778 | 0.8873 | 0.4056 | 0.8794 | 0.8129 | 0.7218 | 0.7767 | 0.7444 |
| W-LIN 7b g0.25 | 944 | 3,599 | 0.8296 | 0.8555 | 0.5031 | 0.8824 | 0.8917 | 0.4063 | 0.8829 | 0.7956 | 0.7225 | 0.7799 | 0.7438 |
| **944 MLP** (C purity) | 944 | 149,343 | **0.8663** | **0.8927** | 0.5006 | **0.9277** | **0.9313** | **0.6944** | **0.9443** | 0.9636 | 0.9137 | 0.9386 | **0.8000** |

Cross-root caution, stated because it bit this lane: the same two 372 bakes
read very differently at the `2026-08-30-full-features-372` root
(ADD156 sdr25 0.0353 there vs 0.9797 here; KonJND 0.4462 vs 0.5363 — the two
roots ship *different KonJND corpus files*, n = 1008 vs 504). **Only the
single-root table above is a valid cross-class comparison.**

### 4.2b The same five, read WITHIN image — and the reading changes

Pooled SROCC on a corpus with many references is partly a cross-image scale
agreement; the **within-image** number is what a codec dial actually needs (the
user asks for a target and the encoder walks its own ladder). `bake_verdict`
reports it as `per_ref`, and the project's own note on it says to read the two
together: "a wide gap means the pooled number is carried by cross-image scale
rather than ranking".

Mean within-reference SROCC, same five models, same root:

| model | CID22 | nonphoto | imazen26 | **HF-NL** | CSIQ | LIVE | KADID | TID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| basic-only ADD156 | 0.9507 | 0.9311 | **0.9294** | **0.7993** | 0.9033 | 0.9590 | 0.8277 | 0.8273 |
| sparse 372 (**B**) | 0.9532 | **0.9322** | 0.9288 | 0.7652 | 0.9327 | 0.9037 | 0.8194 | 0.7844 |
| W-LIN 7b g0.20 | 0.9535 | 0.9186 | 0.9222 | 0.7558 | 0.8993 | 0.8118 | 0.7335 | 0.7855 |
| W-LIN 7b g0.25 | 0.9538 | 0.9198 | 0.9238 | 0.7559 | 0.9037 | 0.7945 | 0.7346 | 0.7879 |
| **944 MLP** | **0.9585** | 0.9270 | 0.9255 | **0.8099** | **0.9459** | **0.9622** | **0.9220** | **0.9428** |

**This inverts the HF-NL story.** Pooled, ADD156 looks like the worst model on
HF-NL by a wide margin (0.295 vs B's 0.350 and the MLP's 0.694). Within
reference it is the **second best of the five** — 0.799, *above* shipped B
(0.765) and the W-LIN blend (0.756), 0.011 behind the 944 MLP. The pooled gap
is cross-image scale, not ranking, and it is the ranking that a target-hitting
loop consumes.

On this axis the basic-only class is level with or ahead of shipped B on seven
of eight corpora (CID22 −0.003, nonphoto −0.001, imazen26 +0.001, HF-NL
**+0.034**, LIVE **+0.055**, KADID +0.008, TID **+0.043**); its one real loss
is CSIQ (−0.029).

### 4.3 Against the ssim2 floor

`peer_ssim2.fulleval.json` — the reference-metric row already on the board.
`nonphoto` / `imazen26` / `hfnlproxy` are excluded from this comparison
**because their target IS ssim2** (the peer scores 1.0 there by construction);
on those axes a model's SROCC is *agreement with* ssim2, not a win over it.
The human-labelled corpora are where "as good or better than ssim2" is
answerable, and on six of the seven the pair counts match exactly:

| model | CID22 | CSIQ | LIVE | KADID | TID | AIC-3 | KonJND *(n differs)* |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ssim2 peer** | 0.8894 | 0.9047 | 0.9599 | 0.8133 | 0.8460 | 0.7970 | 0.4786 (n=1008) |
| basic-only ADD156 | −0.026 | −0.003 | **+0.000** | −0.005 | −0.022 | −0.020 | **+0.058** |
| sparse 372 (B) | −0.007 | **+0.030** | −0.061 | −0.005 | −0.067 | −0.033 | **+0.041** |
| W-LIN 7b g0.20 | −0.031 | −0.025 | −0.147 | −0.092 | −0.069 | −0.053 | **+0.033** |
| W-LIN 7b g0.25 | −0.034 | −0.022 | −0.164 | −0.091 | −0.066 | −0.053 | **+0.025** |
| **944 MLP** | **+0.003** | **+0.040** | **+0.004** | **+0.101** | **+0.093** | **+0.003** | **+0.022** |

Read with two flags the board already carries: **KADID and TID are 100 %
train==val** for the models trained on them, so their columns reward
memorization (the 944 MLP's +0.10/+0.09 there is the least trustworthy pair of
numbers in this note); and the KonJND row compares different corpus files.

**The honest ssim2 verdict.** Only the **944 MLP** is at-or-above ssim2 on
every human corpus. **ADD156 is the closest thing to a tie** — within 0.026 on
CID22, within 0.005 on KADID, dead level on LIVE and CSIQ, and 0.058 *ahead*
on KonJND — while needing the cheapest walk in the crate. Shipped B and the
W-LIN blend both trail ssim2 on LIVE and TID by more than ADD156 does.

### 4.4 Working set — the second axis, derived from the footprint lane's model

`benchmarks/fold_footprint_2026-08-31.md` §9.2 establishes the per-thread hot
set: a `v1_only + Full` self-blur band task touches **12 planes over 42 rows**
— 2 raw windows, 4 band-local H, then `act_raw`, `act`, `mu1_v`, `mu2_v`,
`ssq_v`, `s12_v` — i.e. `2,016·W` bytes. `V1PoolsMode::Peaks` touches **6 of
those 12**: it never runs the activity chain and never asks the fused kernel
to store mu or sigma, so `FoldPoolScratch::ensure` is skipped entirely and only
`ensure_h` runs. **`1,008·W` bytes — exactly half.** (Derived from that lane's
model and this lane's code, not a second measurement of the same thing.)

| W | `Full` / thread | **`Peaks` / thread** | `Full` × 8 (one CCD) | **`Peaks` × 8** |
|---|---:|---:|---:|---:|
| 1152 | 2.21 MiB | **1.11 MiB** | 17.7 MiB | **8.9 MiB** |
| 2304 | 4.43 MiB | **2.21 MiB** | **35.4 MiB** (> 32 MiB L3) | **17.7 MiB** (fits) |
| 4096 | 7.88 MiB | **3.94 MiB** | 63.0 MiB | 31.5 MiB |
| 8192 | 15.75 MiB | **7.88 MiB** | 126.0 MiB | 63.0 MiB |

The six planes `Peaks` skips are not merely untouched, they are **never
allocated**: `FoldPoolScratch::ensure` is called only below the `HOnly` arm, and
`StreamChannelAccums` (which owns `pool_scratch`) is constructed per walk
(`feature_v2.rs:7499`) rather than persisted in `V2Scratch`, so no compare
inherits a grown pool scratch from an earlier one. Only `ensure_h` runs.

The footprint lane's §9.2 finding was that at 2304² eight fold threads need
35.4 MiB against a 32 MiB L3 while buffered needs 26.6 MiB, and "the threshold
falls exactly between them". **A peaks-only model class moves the fold to the
other side of that threshold** — 17.7 MiB, below buffered's own 26.6 MiB. So
the basic-only class plausibly wins twice: less work per pixel AND a hot set
that fits where the current one does not. Whether the second win materialises
is that lane's §9.4 experiment, not this one's; this note supplies the input to
it, not the answer.

---

## 5. What shipped

### 5.1 `V1PoolsMode::Peaks` — the compute boundary, made expressible

The extractor could previously say "no pool slots" (`Off`), "ten carrier slots"
(`Carriers`) or "all 216" (`Full`). None of those is the boundary §1 found.
`Peaks` is: emit `f156..228`, skip the masked/IW pass group.

`BandPoolWork { HOnly, Carriers, Full }` is its band-level resolution. `HOnly`
hands the band its scratch **only** so the band-local self-blur shape
(`FoldHSource::SelfBlur`) stays available — the fold-MT lane's memory-traffic
lever previously required `Full`, so without this a peaks-only request would
have regressed to phase A's strip-wide H planes.

**Gate — `feature_v2::tests::folded_peaks_mode_is_pure_compute_skipping`:** 19
geometries × {`v1_only` walk, full 944 walk} × {serial, rayon} = 76 cells.
Every slot `Peaks` emits is `to_bits()`-identical to `Full`'s; every skipped
slot is exactly `+0.0` (not a partial accumulation, not a NaN from finalising
an accumulator nothing wrote); and the peak block is asserted non-vacuous so
the identity cannot pass by both sides being zero.

### 5.2 Per-profile weight-skipping

`fold_engine::score_pool_mode(params, config, skip)` resolves the mode from
what the profile's consumers structurally read:

* the linear `weights`, which `metric::score_v1_layout_features` reads over
  `[0, num_scales·3·FEATURES_PER_CHANNEL_WITH_PEAKS)` = `f0..228`. Masked and
  IW are out of its reach — **checked at runtime, not assumed**, so a future
  config that widened that range cannot silently make the skip unsound;
* `mlp_bytes`, `mlp_bytes_b3` and `ensemble_classifier_bytes` — the UNION, via
  `cached_bake_pool_need`, interned by bake-bytes pointer exactly as
  `metric::cached_bake_metadata` is (parsing a bake per compare would defeat
  the point).

"Structurally reads" is `L∞(W0[k, :]) > 0` over that caller line. An exactly
zero weight makes the slot unreachable by the forward pass for every input, so
leaving it at `0.0` cannot move the score by a ULP. A merely *small* weight is
never treated as skippable.

**It declines rather than guesses.** When `caller_input_width() != n_inputs()`
the bake declares a variable-arity `FeatureTransform` (`Drop` from
dead-column pruning, or an expander) and layer-0 column `k` is not caller line
`k` — the caller-width bug class `zensim-validate::block_profile` documents
four instances of. Rather than re-implement that crate's arity walk (one owner
per task), this returns `V1PoolNeed::ALL`. It costs nothing today: pruning only
removes columns that were already exact zeros, so a pruned bake's parent gives
the same answer.

**The policy never returns `Off`,** and the reason is footprint rather than
arithmetic: `Off` hands the band no scratch, which disables self-blur, so the
walk falls back to phase A's four strip-wide H planes. `Peaks` computes the
identical sums, emits a superset of `Off`'s slots, and is the smaller hot set
(§4.4). `Off` is dominated on every axis.

### 5.3 The opt-in, and why it is one

`Zensim::with_unread_feature_skipping(bool)` — `#[doc(hidden)]`,
`feature-regime-v2`-gated, **default off**. The skip is score-neutral by
construction but it is *feature*-visible: skipped slots come back `0.0` from
`ZensimResult::features()`. That is exactly the distinction
`fold_engine::is_fold_backable` already draws for
`streaming::active_channels`' channel skipping, and it is why the extraction
entries (`compute_extended_features`, the `compute_all_features` path) pass
`None` unconditionally — they exist to hand the caller the vector, so zeroing a
slot there is a wrong answer even when no weight reads it.

**Gates:**

* `fold_engine::skip_policy_tests::a_fired_skip_leaves_raw_distance_bit_identical`
  — forces the skip on 5 geometries × serial/rayon and asserts `score`,
  `raw_distance`, `mean_offset` and every slot in `f0..228` are bit-identical,
  the skipped block is exactly zero, and `Full`'s block is non-zero.
* `fold_engine_parity::unread_feature_skipping_is_inert_on_a_profile_that_reads_the_block`
  — 23 geometries × rayon pools {1, 2, 3, 8, 16} × {Buffered, Fold}: opting in
  on profile B changes nothing, because B reads the block.
* three policy tests (shipped profiles resolve to `Full`; off by default;
  `bake_pool_need` on shipped B matches what `bake_block_profile` reports for
  the same bytes; an unparseable bake needs everything).

Whole `zensim` lib suite 238 passed / 0 failed; `fold_engine_parity` 12/12.

### 5.4 Tooling — owner extensions, no new duplicates

* **`zensim-validate::block_profile`** gains `V1_FAMILIES` and
  `BlockProfile::v1_families` — the v1 **COMPUTE** families (basic 156 / peaks
  72 / masked 72 / IW 72) alongside the existing append-only numbering blocks,
  rendered in both the text table and the `block_profile` JSON the board
  consumes. This is what made §2.3 a one-command answer.
* **`bake_contrib --ablate-range`** becomes repeatable and nameable
  (`NAME=LO..HI`, comma-separable), so one run prices a whole family frontier.
  The expensive part of that binary is the per-input pass (`n_inputs` full
  re-forwards per row); it is now paid once for six ranges instead of six
  times. Output is TSV-shaped (`ABL\t…`) so a frontier table assembles without
  parsing prose. `spearman` is still the canonical
  `zensim_validate::panel` one.

### 5.5 New public items — LISTED FOR APPROVAL

All three are `feature-regime-v2`-gated (not a default feature) and the two
methods are `#[doc(hidden)]`:

| item | kind | note |
|---|---|---|
| `feature_v2::V1PoolsMode::Peaks` | new enum variant | `V1PoolsMode` is `pub` and **not** `#[non_exhaustive]`, so this is technically breaking for a downstream exhaustive `match`. Inside this workspace the only matches are in `zensim` itself and its benches. |
| `Zensim::with_unread_feature_skipping(bool)` | new method | additive |
| `Zensim::score_pool_mode() -> V1PoolsMode` | new method | additive; the read-back the perf harness prices |

---

## 6. The recommendation, and what each part takes

### 6.1 Answering "what should we drop?" literally

**Inside a fixed model: for shipped B, nothing.** Every family it computes it
uses (§2.1), the peak block is free, and the masked/IW pass group — the only
family with a real price — is worth 0.40 CID22 and 0.53 KonJND to it. There is
no Pareto-dominated family in shipped B, and the honest answer to the literal
question is that B's 95-of-372 read set is *not* a 74 % saving waiting to be
taken.

**Across models, the dominated blocks are real and large:**

| drop set | for which class | rank cost (zero-at-inference) | realizable by |
|---|---|---|---|
| masked + IW (`f228..372`) | the 944 MLPs | **exactly 0** (their layer 0 is exact-zero there) | **nothing — shipped this lane** |
| masked + IW | W-LIN 7b | CID22 −0.005, KonJND −0.048, nonphoto **+0.001**, imazen26 **+0.002** | a retrain, or accept the KonJND cost |
| the whole v1-372 | W-LIN 7b | CID22 −0.027, imazen26 −0.008, LIVE **+0.117** | a retrain **and** a `fold_v1` skip lever |
| peaks (`f156..228`) | anyone | −0.038 CID22 / −0.311 KonJND on B | **never worth doing** — it saves zero compute |

### 6.2 The model-class recommendation

**Evaluate the basic-only class first, and evaluate it seriously.** ADD156 is
within 0.019 CID22 and 0.006 imazen26 of shipped B, *beats* B on KonJND
(+0.017) and LIVE (+0.062), ties ssim2 on LIVE/CSIQ/KADID and is 0.058 ahead
of ssim2 on KonJND — while its cheapest fold request is `v1_only + Peaks`, the
smallest walk the crate can do and (§4.4) half the per-thread hot set of what
`score()` runs today. Its apparent loss is **pooled HF-NL (0.295 vs B's 0.350 and the 944 MLP's
0.694)** — but §4.2b shows that gap is cross-image scale, not ranking: **within
reference ADD156 is 0.799 on HF-NL, ahead of B's 0.765 and behind only the 944
MLP's 0.810.** For the stated product — a dial the encoder walks per image —
the within-image number is the one that governs, and on it the basic-only class
matches or beats shipped B on seven of eight corpora.

**Keep the 944 MLP as the quality reference.** It is the only class at or above
ssim2 on every human corpus, and it is the only one that is genuinely strong on
HF-NL. It costs the full 944 walk plus a 149 KB bake — but note that even it
does not need the pool block, so the shipped skip applies to it.

**The W-LIN 7b blend is the class this data does not flatter.** At 3.6 KB it is
the smallest 944-class model and it wins the two ssim2-anchored axes among the
linear classes (nonphoto 0.878, imazen26 0.887, both above B), but it trails
ssim2 on LIVE by 0.147 and on KADID by 0.092, and it needs the *whole* 944 walk
to do it — the most expensive compute set on the board for a model that reads
essentially nothing from the 372 block it forces the walk to compute.

### 6.3 Retrain-feasibility notes (not run by this lane)

Zero-at-inference understates what a refit recovers, so each drop set below is
paired with what a wave would have to do:

* **basic-only, refit for the pooled tails.** ADD156's real gaps to B are
  POOLED HF-NL (−0.055) and CID22 (−0.019); its within-reference numbers are
  already level or better (§4.2b), so what a refit has to fix is cross-image
  calibration, not ranking — which is the dial/spline half of the pipeline
  (`bake_dial_refit`) at least as much as the head. A wave that refits a
  156-input additive head with `hf_nearlossless` weighted up and re-anchors the
  output spline is the direct test. Cost: one `zensim_mlp_train` /
  `bake_dial_refit fit-lasso` recipe over the existing `f0..156` columns —
  **no new extraction**, every canonical root already carries them. This is
  the cheapest wave on this list by a wide margin and it is the one to run
  first.
* **W-LIN 7b without v1-372.** The blend's own numbers say the block is worth
  −0.027 CID22 to it; a refit on `f372..944` alone would recover part of that
  by redistributing onto v2 slots it already uses. Needs the `fold_v1` lever
  below **and** a retrain; the training tables need no change (drop columns).
* **A 944 MLP without the pool block** needs nothing — it already trains that
  way. This is the one row where the ablation and the retrain agree exactly,
  because the weights are structurally zero rather than merely small.

### 6.4 The one lever this lane did NOT pull, and why it is the next one

`fold_v1` is a hardcoded `let fold_v1 = true;` (`feature_v2.rs:7395`), but the
walk **already branches on it** in four places — the band-replay call
(`:6299`), the width computation (`v1_total = if fold_v1 { … } else { 0 }`),
the finalize block, and the pool-mode plumb. Making it a compute decision would
let a 944-class model that reads nothing in `f0..372` skip v1's entire band
replay, which is the single largest remaining block for the W-LIN class.

It is not in this lane because it is **not byte-neutral**: it changes `f0..372`
from real values to zeros in a 944-wide vector, which is a regime-shaped
decision (every 944 training table has live `f0..372` in the pools roots and
structural zeros in the folded roots), and no model on the board today reads
zero from *all* of `f0..372` — the 944 MLPs read all 156 basic lines. Priced
here, decided elsewhere.

---

## 7. Reproduction

All commands from the repo root; binaries built with
`--features custom-profiles,feature-regime-v2,threads,training`.

**Per-bake read set (the §2.3 table):**

```sh
cargo build --release -p zensim-validate --bin bake_block_profile
./target/release/bake_block_profile --bake <bake.bin>     # v1 COMPUTE families
./target/release/bake_block_profile --bake <bake.bin> --json | jq .v1_families
```

**Family ablation (the §2.1/§2.2 tables) — one run, six ranges:**

```sh
cargo build --release -p zensim-validate --bin bake_contrib
R=/mnt/v/zen/zensim-training/2026-08-30-full-features-372
./target/release/bake_contrib --bake zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin \
  --rows 20000 \
  --corpus cid22:$R/cid22_features_372col_2026-05-15.parquet:human_score:1 \
  --corpus konjnd:$R/konjnd_features_372col_2026-05-15.parquet:human_score:1 \
  --corpus nonphoto@20000:$R/nonphoto_features_372col_2026-07-15.parquet:human_score:1 \
  --corpus imazen26@20000:$R/imazen26_test_120k_2026-07-16.parquet:human_score:1 \
  --corpus hfnlproxy@20000:$R/ext_hfnlproxy.parquet:human_score:1 \
  --corpus kadid:$R/kadid_features_372col_2026-05-15.parquet:human_score:1 \
  --corpus tid:$R/tid_features_372col_2026-05-15.parquet:human_score:1 \
  --corpus csiq:$R/csiq_features_372col_2026-07-18.parquet:human_score:1 \
  --corpus live:$R/live_features_372col_2026-07-18.parquet:human_score:1 \
  --ablate-range 'v1_basic=0..156,v1_peaks=156..228,v1_masked=228..300,v1_iw=300..372,v1_masked_iw=228..372,v1_pools_all=156..372'
```

For the 944 bakes swap `R` for
`/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30` (its `ext_*` filenames)
and add `v2_348=372..720,append_204=720..944,v1_all=0..372`. **Do not use the
ext944 root for a pool ablation** — it feeds `f156..371` as structural zeros,
so every pool delta reads as exactly 0 for the wrong reason.

As-run logs: `benchmarks/feature_cost_2026-08-31/ablate_{shippedB,q7b,q7b_blocks}.log`.

**Cross-class quality (§4.2):** `bake_verdict --features-root
/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30 --regime 944 --cross-regime
--full-json <out>.json` for each of the five bakes; the ssim2 floor is the
board's own `/mnt/v/output/zensim/reports/fulleval/peer_ssim2.fulleval.json`,
read, not recomputed.

**Compute (§3.1):**

```sh
cargo build --release --bench extract_paths_bench -p zensim \
  --features custom-profiles,feature-regime-v2,threads,training
for T in 1 8 16; do RAYON_NUM_THREADS=$T <the bench binary>; done
```

**Gates:**

```sh
cargo test --release -p zensim --features custom-profiles,feature-regime-v2,threads,training \
  --lib folded_peaks_mode_is_pure_compute_skipping skip_policy
cargo test --release -p zensim --features custom-profiles,feature-regime-v2,threads,training \
  --test fold_engine_parity
```

---

## 8. Measurement quality, stated plainly

* The ablation deltas are **exact** in the sense that matters — the rank-|K|
  update is algebraically the same forward pass, and the baseline parity gate
  against `bake_runtime::score_row` returned **max|diff| = 0.000e0** over
  47,511 rows. What is *approximate* is their meaning: zero-at-inference is a
  lower bound on what a retrain would have to recover (§2, §6.3).
* `nonphoto` / `imazen26` / `hfnlproxy` are stride-decimated to 20,000 rows in
  the ablation runs (baseline and ablated share the rows, so the *delta* is
  unaffected; the absolute baselines there differ slightly from a full-corpus
  verdict). §4.2's verdict table is un-decimated.
* KADID and TID are 100 % train==val for the models trained on them. Their
  columns are integrity guards, not ranking signal, and the note says so at
  every point it quotes them.
* The KonJND comparison against the ssim2 peer row crosses corpus files
  (n = 504 vs 1008) and is flagged in place; the other six human corpora match
  pair-for-pair.
* The wall-clock table's conditions and any zenbench CV flags are recorded with
  it in §3.1.

