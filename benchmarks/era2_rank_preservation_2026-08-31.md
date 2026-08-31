# era-2 rank preservation: the roster on old-vs-new features, against the registered bar

**Lane:** era-2 rank preservation — the third and last of the era-2 flip
prerequisites (`benchmarks/era2_perf_break_2026-08-31.md` §6, "what is still
missing"). The other two — the gate re-pin enumeration (§27) and the blast-radius
registration (§6) — were closed by the era-2 lane before this ran.

**What this is.** The measurement, not the argument. The era-2 lane registered
its own bar in §21.1 on 2026-08-31 **before any candidate existed**:

> **PASS iff no corpus loses more than `0.005` SROCC and the product composite
> does not fall.**

This lane executes that bar across the model roster and reports the result. A
FAIL is as valid an outcome as a PASS and is not softened anywhere below. Nothing
here computes a statistic: every SROCC and composite is read verbatim from a
`bake_verdict --full-json` fulleval (whose stats route through
`zensim_validate::panel` → `zenstats`), and
[`era2_rank_preservation_2026-08-31/era2_rank_table.py`](era2_rank_preservation_2026-08-31/era2_rank_table.py)
only arranges them.

**Method is the blur/radius lane's, deliberately reused rather than reinvented**
(`benchmarks/blur_radius_locality_branches_2026-08-31.md` §3.2): re-extract every
corpus per arm over byte-identical pairs TSVs with `v2_ab_extract`
(`ZENSIM_AB_MODE=foldapp2pools`) → `promote_ext944_canonical.py`
(`EXT944_MODE=folded720append2pools`) → `bake_verdict --regime 944 --full-json`.
The `foldapp2pools` regime (f156-371 LIVE) serves both model classes from one
root, which is what lets a 372-input bake and a 944-input bake be read on
**exactly the same pixels**.

---

## 0. The result in one page

**At the recommended production tile width (`ZENSIM_H_TILE=1024`, era-2 doc §25),
the tiling flip — the only byte-changing component of the break that is merged
on `main` today — is bit-identical on 8 of the 9 eval corpora and moves the 9th
by at most 2.0e-4 SROCC.** Five of six roster models PASS the bar outright. The
sixth, `BHdr`, fails only the bar's *composite* clause, and it fails it by
**3.2e-6** — its worst corpus loss is 4e-5, which is 125× inside the 0.005 corpus
threshold.

**The reason 8 of 9 corpora are bit-identical is structural, not lucky, and it is
also this measurement's main limitation.** Every H entry guards on
`tile > 0 && width > tile`, and the eval corpora are narrow: six of nine have a
maximum reference width of 512 px, and only AIC-3 has any reference wider than
1024. So at production width the eval panel can only *see* the flip on one
corpus. Two stress arms (`tile = 256` and `tile = 32`, the latter being §27's
gate-re-pin setting and therefore the maximum tile-edge density anyone has
proposed) were run for exactly this reason, and they bound the worst case.

**The bar's third clause — the dial gates — is SATISFIED BY CONSTRUCTION for the
tiling flip, and this lane proved it rather than leaving it open.** The dial grid
and the corruption grid were re-extracted per arm. Every dial-grid reference is
≤ 1024 px wide and the corruption grid is a single 576×576 reference, so at
tile = 1024 the two grid twins come back **byte-identical** (same sha256,
4,547,248/4,547,248 cells equal). At the tile = 32 stress setting the grid does
move and the dial panel was run: no model crosses either G3 bound.

**`BLUR_RADIUS` 4 (item F1) is not merged and is a separate question, and it
dominates the combined answer.** Radius 4 was measured on the same roster; it
passes on **two** models (`C944` and — new here — `ADD156`) and fails on four.
Composing it with the tile changes nothing: every model's worst-corpus delta is
identical to five decimals between "radius 4, tile off" and "radius 4, tile 1024".
So the two components are separable, and if the break carries radius 4 the
verdict is radius 4's verdict.

---

## 1. Deliverable 1 — what actually changes bytes, and what is merged

Read out of source at `9e52fb164c28725a6f12d911707b8caaeaac995e` (this
workspace's parent; the working tree is byte-identical to it for every arm below
except the radius arms, which patch and revert — control C4).

| # | component | where | status on `main` today | byte-changing? | measurable here? |
|---|---|---|---|---|---|
| 1 | **Column tiling, all H entries** | `zensim/src/blur.rs` `h_blur_tile_width()` (:2911) guarding `box_blur_h` (:600), `box_blur_h_into_abs_diff` (:1294), `fused_blur_h_mu` (:2000), `fused_blur_h_ssim` (:3016), `fused_blur_h_ssim3` (:3086) | **MERGED**, behind `ZENSIM_H_TILE`, **default `0` = off** | **YES** when `tile > 0 && width > tile` — the running x-sum restarts per tile | **YES** — runtime env, three widths measured |
| 2 | **Fixed 8 virtual lanes + fixed `era2_reduce8` tree** | `dense_block_kernel_era2*`, `feature_v2.rs:17068+` | **IN TREE, NOT WIRED.** Every call site is inside `pub(crate) mod tests` (:10906+) or behind `#[cfg(any(test, feature = "oracle"))]` (`harness_dense_slots` :17176, `bench_dense_era1/era2` :17253/:17273). The four production dense call sites (`:5137`, `:5215`, `:6528`, `:8584`) call era-1 `dense_block_kernel` unconditionally, with no flag | YES, by design | **NO** — there is no configuration of today's `main` that routes a feature extraction through the era-2 kernel |
| 3 | **`ERA2_BAND_ROWS` as semantics** | `feature_v2.rs:17102`, `pub(crate) const = 32` | used only inside (2) | YES (era-2 doc §14.3) | **NO** — follows (2) |
| 4 | **`TILE_WIDTH` as semantics** | the value of `ZENSIM_H_TILE`; not a const in tree | see (1); §25 derives 1024 from L2, does not freeze it | YES — each width is a distinct grouping | **YES** — measured at 1024 / 256 / 32 |
| 5 | **`BLUR_RADIUS` 5 → 4 (item F1)** | `feature_v2.rs:575` + `V1_BAND_OVERLAP` (:4429) + `metric.rs`/`profile.rs` constants + the `fold_engine.rs:89` fold gate | **NOT in the break** — it is a row on the E+F decision table, radius on `main` is 5 | YES | **YES**, by rebuild (`patch_radius.sh`, reverted after) |
| 6 | **V-plane redirect, exit 1 (item F4)** | — | **NOT LANDED.** Priced in `512c66c2`; nothing in `zensim/src` implements it. §24.5 additionally claims the candidate is *bit-identical* (the fused kernel's stores are documented identical to `box_blur_v_from_copy`), with only the plane top/bottom edges open | probably NO | **NO** — no code to measure |
| 7 | **Item D `ComputeSet`** | `feature_v2.rs:1688`, `pub(crate)`, 1,024-combination legacy-parity gate | MERGED, no public API | **NO** — byte-neutral derivation of the six locals it replaces | n/a |
| 8 | **Item E1, drop the v1 pool pass** | expressed *through* (7) | decision-surface row, not code | per-request, not global | not re-derived here — the E+F table carries the frontier lane's numbers (944 MLPs exactly 0; W-LIN −0.005 cid22; **B −0.399 cid22**) |

**So the break, as it stands today, has exactly one merged byte-changing
component: the tile.** Everything else is either not wired (2, 3), not landed
(6), byte-neutral (7), a per-request choice rather than a redefinition (8), or a
proposal on the decision table (5).

---

## 2. Extraction: seven arms, and the controls that make them comparable

Nine eval corpora — the set `bake_verdict` reads at `--regime 944` that has a
local pairs TSV, which includes **kon504** (`ext_konjnd_jpeg_val`, 504 rows, the
`--regime 944` KonJND leg). **20,516 pairs per arm.**

| arm | `ZENSIM_H_TILE` | `BLUR_RADIUS` | what it is |
|---|---|---|---|
| `era1` | unset | 5 | the control |
| `e2t1024` | `1024` | 5 | **the era-2 ship candidate** at §25's derived production width |
| `e2t256` | `0256` | 5 | intermediate tile-edge density |
| `e2t32` | `0032` | 5 | **maximum** tile-edge density — §27's gate-re-pin setting |
| `r4ctl` | unset | 4 | item F1 alone |
| `r4t1024` | `1024` | 4 | **the combined break** |
| `r4t32` | `0032` | 4 | radius 4 under max tile-edge density |

Full log: [`extraction_controls.txt`](era2_rank_preservation_2026-08-31/extraction_controls.txt).

### 2.1 Control C1/C2 — the era-1 arm reproduces the canonical root exactly

The `era1` arm's nine CSVs are **byte-identical** to the blur/radius lane's `r5`
extraction, and the nine promoted parquets have **matching sha256**. That lane in
turn measured its `r5` root at **19,367,104 / 19,367,104 cells identical** to the
canonical `/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30` root, max abs 0.
Three consequences:

1. Every delta below is attributable to the arm variable and to nothing else in
   the pipeline.
2. The blur lane's `r4` root is directly usable as an arm of this study — and
   control C3 confirms it: **my own radius-4 rebuild reproduces their `r4`
   extraction byte-for-byte on all nine legs.**
3. Incidentally, this is an independent byte-neutrality check on `ab49d4b7` —
   the commit that moved the tile from selected call sites onto all H entries —
   over 20,516 real corpus pairs with the tile off. It is byte-neutral.

Every model's `era1` verdict also reproduces its published values: `WLIN7b_g020`
comes back cid22 0.85885 / konjnd 0.51184 / csiq 0.87936 / live 0.81289 /
kadid 0.72176 / tid 0.77673 / aic3 0.74442 — all seven matching
`feature_cost_frontier_2026-08-31.md` §5 to four decimals — and B comes back
cid22 0.88211, its recorded runtime value.

### 2.2 Row alignment

`(ref_basename, human_score)` compared row-for-row across all seven arms plus the
blur lane's `r4` and `r5`: **0 mismatching (leg, arm) cells, 20,516 rows per
arm.** [`row_alignment_gate.txt`](era2_rank_preservation_2026-08-31/row_alignment_gate.txt).

### 2.3 The corpora are narrower than the production tile — the finding that shapes everything

| corpus | refs | min W | max W | refs > 1024 |
|---|---:|---:|---:|---:|
| cid22 | 49 | 512 | 512 | 0 |
| kadid | 81 | 512 | 512 | 0 |
| tid | 25 | 512 | 512 | 0 |
| csiq | 30 | 512 | 512 | 0 |
| live | 29 | 480 | 768 | 0 |
| konjnd (kon504) | 504 | 640 | 640 | 0 |
| sdr25 | 5 | 620 | 620 | 0 |
| aic4 | 5 | 620 | 620 | 0 |
| **aic3** | 10 | 560 | **2592** | **7** |

At `tile = 1024` the byte-compare comes out exactly as the guard predicts:
**eight legs byte-identical, `ext_aic3` differing on 420 of 600 rows** (the seven
refs > 1024 × 60 pairs each). The same holds in the score domain — at
`tile = 1024`, `per_pair.pred` is *unchanged for every pair of every corpus except
aic3*, for all six models
([`score_deltas.txt`](era2_rank_preservation_2026-08-31/score_deltas.txt); note
`bake_verdict` writes a 5,000-row subsample of `per_pair` for kadid, so that
block's `n` is 5,000 while the SROCC panel above uses all 10,125 rows).

**This is the honest limit of the production-width measurement: it exercises the
tile on one corpus of nine.** It is not a defect of the extraction — it is what
the corpora are. The stress arms exist to cover it, and §5 says what would close
it properly.

---

## 3. Deliverable 3 — the roster, both eras

Six models. Bake paths and sha256 are recorded in the run manifest
`/mnt/v/output/zensim/era2-rank-2026-08-31/_MANIFEST.json` (block storage; see
the pointer file).

| label | bake | n_inputs |
|---|---|---:|
| **B** | `zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` | 372 |
| **C944** | `zensim/weights/c_sdr_purity944_2026-08-29.bin` (the `C purity944` flagship) | 944 |
| **WLIN7b_g020** | `/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.2_a0.2_b0.97.bin` | 944 |
| **WLIN7b_g025** | `/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.25_a0.2_b0.97.bin` | 944 |
| **ADD156** | `/mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin` | 372 |
| **BHdr** | `zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin` | 372 |

No roster member was skipped. `BHdr`'s route **is** reachable — it is a
372-input shaped-linear bake and reads the same 944 root — but note what that
means: these are the **SDR** eval corpora, so this reads BHdr's SDR behaviour.
Its HDR panel (UPIQ / SI-HDR / the rest) is a different instrument owned by
`scripts/external_reads/run_external_reads.py` and is **not measured here**.

Signed SROCC as the panel reports it; several corpora carry a distortion-oriented
target and are canonically negative (aic4, kadid, konjnd, sdr25). Deltas are in
**magnitude**, and a sign flip would be a failure regardless of size — **no cell
in this study flipped sign**.

### 3.1 The table

Δ|SROCC| against each model's own era-1 row. `tile 1024` is the ship candidate;
`tile 256`/`tile 32` are stress bounds; `r4` columns are item F1, not part of the
break as it stands.

| model | corpus | era-1 | tile 1024 | tile 256 | tile 32 | r4 (tile off) | r4 + tile 1024 |
|---|---|---:|---:|---:|---:|---:|---:|
| **B** | cid22 | +0.88211 | +0.00000 | +0.00000 | +0.00000 | +0.00305 | +0.00305 |
|  | konjnd | -0.51980 | +0.00000 | -0.00006 | -0.00035 | +0.01144 | +0.01144 |
|  | csiq | +0.93493 | +0.00000 | +0.00000 | -0.00000 | -0.00859 | -0.00859 |
|  | live | +0.89852 | +0.00000 | -0.00001 | -0.00001 | -0.01249 | -0.01249 |
|  | kadid | -0.80851 | +0.00000 | -0.00000 | +0.00000 | -0.00968 | -0.00968 |
|  | tid | +0.77888 | +0.00000 | +0.00000 | -0.00001 | -0.00614 | -0.00614 |
|  | aic3 | +0.76367 | +0.00001 | -0.00004 | -0.00012 | -0.00084 | -0.00083 |
|  | aic4 | -0.89042 | +0.00000 | -0.00004 | -0.00010 | +0.00310 | +0.00310 |
|  | sdr25 | -0.95755 | +0.00000 | +0.00000 | -0.00019 | +0.00000 | +0.00000 |
|  | *composite* | *0.81997* | *+0.00000* | *-0.00001* | *-0.00006* | *+0.00400* | *+0.00400* |
|  | **BAR** |  | **PASS** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| **C944** | cid22 | +0.89272 | +0.00000 | -0.00002 | -0.00003 | +0.00194 | +0.00194 |
|  | konjnd | -0.50060 | +0.00000 | -0.00009 | -0.00014 | +0.01522 | +0.01522 |
|  | csiq | +0.94427 | +0.00000 | -0.00000 | -0.00000 | +0.00233 | +0.00233 |
|  | live | +0.96363 | +0.00000 | -0.00000 | +0.00001 | +0.00051 | +0.00051 |
|  | kadid | -0.91373 | +0.00000 | +0.00000 | +0.00000 | -0.00075 | -0.00075 |
|  | tid | +0.93862 | +0.00000 | +0.00000 | -0.00000 | +0.00062 | +0.00062 |
|  | aic3 | +0.79996 | +0.00002 | -0.00005 | -0.00004 | +0.00117 | +0.00113 |
|  | aic4 | -0.91439 | +0.00000 | +0.00000 | -0.00019 | +0.00164 | +0.00164 |
|  | sdr25 | -0.97695 | +0.00000 | +0.00000 | -0.00077 | +0.00202 | +0.00202 |
|  | *composite* | *0.82856* | *+0.00000* | *-0.00003* | *-0.00005* | *+0.00384* | *+0.00383* |
|  | **BAR** |  | **PASS** | **FAIL** | **FAIL** | **PASS** | **PASS** |
| **WLIN7b_g020** | cid22 | +0.85885 | +0.00000 | +0.00000 | -0.00000 | +0.00297 | +0.00297 |
|  | konjnd | -0.51184 | +0.00000 | -0.00018 | -0.00082 | +0.00441 | +0.00441 |
|  | csiq | +0.87936 | +0.00000 | -0.00000 | +0.00001 | -0.00275 | -0.00275 |
|  | live | +0.81289 | +0.00000 | -0.00001 | +0.00005 | +0.02138 | +0.02138 |
|  | kadid | -0.72176 | +0.00000 | -0.00000 | -0.00000 | -0.00125 | -0.00125 |
|  | tid | +0.77673 | +0.00000 | -0.00000 | -0.00001 | -0.00727 | -0.00727 |
|  | aic3 | +0.74442 | +0.00020 | +0.00013 | +0.00007 | +0.00489 | +0.00487 |
|  | aic4 | -0.85377 | +0.00000 | +0.00004 | -0.00005 | +0.00508 | +0.00508 |
|  | sdr25 | -0.90348 | +0.00000 | +0.00000 | +0.00000 | -0.00298 | -0.00298 |
|  | *composite* | *0.79878* | *+0.00001* | *-0.00001* | *-0.00012* | *+0.00340* | *+0.00340* |
|  | **BAR** |  | **PASS** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| **WLIN7b_g025** | cid22 | +0.85554 | +0.00000 | +0.00000 | +0.00002 | +0.00351 | +0.00351 |
|  | konjnd | -0.50310 | +0.00000 | -0.00014 | -0.00009 | +0.00513 | +0.00513 |
|  | csiq | +0.88293 | +0.00000 | -0.00000 | -0.00001 | -0.00248 | -0.00248 |
|  | live | +0.79562 | +0.00000 | +0.00001 | +0.00000 | +0.01498 | +0.01498 |
|  | kadid | -0.72249 | +0.00000 | -0.00000 | -0.00000 | -0.00074 | -0.00074 |
|  | tid | +0.77994 | +0.00000 | -0.00001 | -0.00001 | -0.00734 | -0.00734 |
|  | aic3 | +0.74376 | +0.00010 | +0.00016 | +0.00013 | +0.00469 | +0.00472 |
|  | aic4 | -0.85184 | +0.00000 | +0.00007 | -0.00010 | +0.00544 | +0.00544 |
|  | sdr25 | -0.89983 | +0.00000 | +0.00000 | +0.00000 | +0.00077 | +0.00077 |
|  | *composite* | *0.79491* | *+0.00001* | *-0.00001* | *+0.00001* | *+0.00391* | *+0.00391* |
|  | **BAR** |  | **PASS** | **FAIL** | **PASS** | **FAIL** | **FAIL** |
| **ADD156** | cid22 | +0.86324 | +0.00000 | +0.00000 | +0.00002 | +0.00672 | +0.00672 |
|  | konjnd | -0.53625 | +0.00000 | +0.00006 | -0.00001 | -0.00039 | -0.00039 |
|  | csiq | +0.90167 | +0.00000 | +0.00000 | +0.00000 | +0.00981 | +0.00981 |
|  | live | +0.96029 | +0.00000 | +0.00000 | -0.00000 | +0.00053 | +0.00053 |
|  | kadid | -0.80806 | +0.00000 | +0.00000 | +0.00000 | +0.00367 | +0.00367 |
|  | tid | +0.82369 | +0.00000 | -0.00000 | -0.00000 | +0.00145 | +0.00145 |
|  | aic3 | +0.77700 | +0.00000 | +0.00003 | +0.00003 | -0.00267 | -0.00267 |
|  | aic4 | -0.93308 | +0.00000 | +0.00000 | -0.00001 | +0.00353 | +0.00353 |
|  | sdr25 | -0.97974 | +0.00000 | +0.00000 | +0.00000 | -0.00240 | -0.00240 |
|  | *composite* | *0.81099* | *+0.00000* | *+0.00001* | *+0.00002* | *+0.00485* | *+0.00485* |
|  | **BAR** |  | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** |
| **BHdr** | cid22 | +0.84329 | +0.00000 | +0.00003 | +0.00083 | +0.00735 | +0.00735 |
|  | konjnd | -0.63967 | +0.00000 | +0.00001 | -0.00004 | +0.00184 | +0.00184 |
|  | csiq | +0.84427 | +0.00000 | -0.00001 | -0.00002 | +0.00711 | +0.00711 |
|  | live | +0.91050 | +0.00000 | -0.00001 | +0.00002 | -0.00839 | -0.00839 |
|  | kadid | -0.73393 | +0.00000 | -0.00000 | +0.00000 | +0.00058 | +0.00058 |
|  | tid | +0.75354 | +0.00000 | -0.00000 | -0.00003 | -0.00424 | -0.00424 |
|  | aic3 | +0.79263 | -0.00004 | -0.00010 | +0.00044 | -0.00373 | -0.00361 |
|  | aic4 | -0.88925 | +0.00000 | +0.00006 | -0.00037 | -0.00203 | -0.00203 |
|  | sdr25 | -0.88456 | +0.00000 | +0.00096 | -0.01623 | +0.00864 | +0.00864 |
|  | *composite* | *0.81107* | *-0.00000* | *+0.00002* | *+0.00063* | *+0.00536* | *+0.00537* |
|  | **BAR** |  | **FAIL** | **PASS** | **FAIL** | **FAIL** | **FAIL** |

Full per-corpus tables:
[`tile_rank_analysis.txt`](era2_rank_preservation_2026-08-31/tile_rank_analysis.txt),
[`combined_rank_analysis.txt`](era2_rank_preservation_2026-08-31/combined_rank_analysis.txt);
machine-readable
[`tile_rank_deltas.tsv`](era2_rank_preservation_2026-08-31/tile_rank_deltas.tsv),
[`radius_rank_deltas.tsv`](era2_rank_preservation_2026-08-31/radius_rank_deltas.tsv),
[`combined_rank_deltas.tsv`](era2_rank_preservation_2026-08-31/combined_rank_deltas.tsv).

---

## 4. Deliverable 4 — the verdict against the registered bar

The bar has two clauses in the pass predicate and they behave very differently at
this scale, so both are reported at full precision rather than rounded into
agreement.

### 4.1 The tiling flip at production width — the only thing actually up for the flip

| model | worst corpus Δ\|SROCC\| | composite Δ | verdict |
|---|---:|---:|---|
| B | **+0.00000** (nothing loses) | **+4.963e-07** | **PASS** |
| C944 | **+0.00000** | **+1.241e-06** | **PASS** |
| WLIN7b_g020 | **+0.00000** | **+1.464e-05** | **PASS** |
| WLIN7b_g025 | **+0.00000** | **+7.445e-06** | **PASS** |
| ADD156 | **+0.00000** | **0.000e+00** (exactly) | **PASS** |
| BHdr | −4.0e-05 (aic3) | **−3.226e-06** | **FAIL** |

**Overall: 5 PASS, 1 FAIL.** Every corpus clause passes for every model — the
largest single-corpus loss in the whole production-width arm is `BHdr`'s
**4.0e-05** on aic3, which is **125× inside** the 0.005 threshold. The one FAIL
is entirely the composite clause, which as registered carries **no tolerance**
("the composite does not fall"), and `BHdr`'s composite falls by **3.2e-6**.

Two details make the size of these numbers concrete. `ADD156`'s aic3 SROCC is
**exactly** `0.7769958141384993` in both arms even though 415 of its 600 aic3
predictions changed (max |Δscore| 1.18e-2) — the perturbation never reordered a
single pair, so its composite delta is exactly `0.0`. `B`'s aic3 SROCC moves
`0.7636656440679733` → `0.763672344329381`, which is where its `+4.963e-07`
composite comes from. That is the resolution this bar is being applied at.

For scale, using the bar's own reference points: 3.2e-6 is **13 %** of the
`+0.000024` era-1→era-3 precedent §21.1 cites as the size of a non-event, and
**1/1550** of the corpus clause it sits beside. It is also an *exact* quantity,
not a draw — same weights, same pairs, deterministic extraction; re-running
reproduces it. Whether a 3.2e-6 movement constitutes "the composite falling" is a
judgement for the bar's owner. **This lane does not renegotiate a registered
threshold, so it reports the FAIL.**

### 4.2 The stress arms

| arm | corpus clause | composite clause | overall |
|---|---|---|---|
| `tile 256` | passes everywhere (worst −1.8e-4, WLIN g020 konjnd) | fails on B, C944, WLIN g020, WLIN g025 (−1e-5 … −3e-5) | **2 PASS / 4 FAIL** |
| `tile 32` | **fails once**: BHdr sdr25 **−0.01623** | fails on B, C944, WLIN g020 | **2 PASS / 4 FAIL** |

The `tile 32` BHdr/sdr25 cell is the only genuine corpus-clause failure anywhere
in the tiling study, and it has a mechanism:
[`score_deltas.txt`](era2_rank_preservation_2026-08-31/score_deltas.txt) shows
the tile perturbation in **score points**, and BHdr amplifies it by more than an
order of magnitude over the rest of the roster — on sdr25 at `tile 32`,
max |Δscore| is **1.823** for BHdr against **0.054** for C944 and **0.045** for
B. sdr25 is also the smallest corpus (50 pairs, 5 refs), so it converts a given
score perturbation into the largest rank movement. Both facts point the same way
and neither is in play at production width, where BHdr's sdr25 predictions are
*bit-identical*.

### 4.3 Item F1 (`BLUR_RADIUS` 4) — measured on the same roster

| model | worst corpus Δ | which corpus | composite Δ | verdict |
|---|---:|---|---:|---|
| C944 | −0.00075 | kadid | +0.00384 | **PASS** |
| **ADD156** | **−0.00267** | **aic3** | **+0.00485** | **PASS** |
| B | −0.01249 | live | +0.00400 | FAIL |
| WLIN7b_g020 | −0.00727 | tid | +0.00340 | FAIL |
| WLIN7b_g025 | −0.00734 | tid | +0.00391 | FAIL |
| BHdr | −0.00839 | live | +0.00536 | FAIL |

The four models the blur/radius lane measured reproduce to the digit
(their §3.3: C944 kadid −0.0007, B live −0.0125, both W-LIN tid −0.0073). **The
new information is `ADD156`: it PASSES, so radius 4 now clears the bar on two
roster members rather than one**, and the second one is a 3.6 KB truly-additive
basic-156 model — a different class from the 944 MLP flagship, which makes the
pass less likely to be an accident of one architecture. `BHdr` fails, on live.

### 4.4 Composition — the two components are separable, and radius dominates

| model | r4, tile off | r4 + tile 1024 | difference |
|---|---:|---:|---|
| B | worst −0.01249, comp +0.00400 | worst −0.01249, comp +0.00400 | none to 5 dp |
| C944 | −0.00075 / +0.00384 | −0.00075 / +0.00383 | 1e-5 on composite |
| WLIN7b_g020 | −0.00727 / +0.00340 | −0.00727 / +0.00340 | none to 5 dp |
| WLIN7b_g025 | −0.00734 / +0.00391 | −0.00734 / +0.00391 | none to 5 dp |
| ADD156 | −0.00267 / +0.00485 | −0.00267 / +0.00485 | none to 5 dp |
| BHdr | −0.00839 / +0.00536 | −0.00839 / +0.00537 | 1e-5 on composite |

**Every verdict is unchanged by adding the tile to radius 4.** The isolation is
therefore clean and its shape is simple: the tile contributes ~1e-5 to a
composite and the radius contributes ~4e-3, so **whatever the break decides about
radius 4 IS the break's verdict**, and the tile rides along for free.

**Which subset ships, if the answer is "not all of it":**

* **Tiling alone at `tile ≥ 1024`** — passes on 5 of 6; the one failure is a
  3.2e-6 composite movement in BHdr and no corpus clause is violated anywhere.
  This is the shippable subset.
* **Tiling at `tile < 1024`** — starts costing composite on the 944-class models
  and, at `tile = 32`, costs BHdr a real corpus clause. If a smaller tile is ever
  wanted for footprint, it needs its own pass at that width; §25's own
  recommendation (derive the width from L2, which gives 1024) is on the right
  side of this.
* **Radius 4** — passes on C944 and ADD156, fails on B, both W-LINs and BHdr.
  It is not carried by this measurement.

### 4.5 The caveat that cuts the other way — every model was trained at era-1

**All six bakes were trained on era-1 features** (radius 5, tile off). Every
number above is an old-weights-on-new-features read. That is the **strict** form
of the test — it cannot flatter the change — but it is also its main limitation,
and the limitation is asymmetric: the losses sit exactly where a retrain would be
expected to recover most, because the weights are fitted to a summation grouping
and a support width that the candidate no longer has. **So the honest reading of
every FAIL above is "this is an upper bound on the cost, not an estimate of it."**

This matters much more for radius 4 (a support-width change, ~4e-3 composite
movements, four FAILs) than for tiling (a summation-grouping change at 1e-5).
A **radius-4 retrain is already registered** by the blur/radius lane (§3.4) and
is deliberately **not launched here** — retraining is a training decision, and
this lane's job was to run the bar.

---

## 5. Deliverable 5 — the dial clause

The blur/radius lane flagged (§3.5) that `bake_verdict`'s dial and corruption
panels read *stored* 944-feature grid parquets, so those blocks came back
byte-identical across all four of its radius roots, and it correctly refused to
read that as the clause being satisfied — it left clause 3 **OPEN**.

**The same is true here, empirically: `dial` and `corruption` are byte-identical
across all seven of my arms for all six models**
([`dial_clause.txt`](era2_rank_preservation_2026-08-31/dial_clause.txt)) — which
proves only that `bake_verdict` reads a grid that no arm re-extracted. That is
not an answer, so this lane went and got one: **the dial grid and the corruption
grid were rebuilt per arm** through their owners (`scripts/v_next/build_dial944.py`
with `ZENSIM_H_TILE` set, `scripts/v_next/build_corr944.py`), from the persisted
pixels, and compared cellwise.

### 5.1 Tiling: clause 3 is satisfied BY CONSTRUCTION at `tile ≥ 1024`

Every dial-grid **reference** is ≤ 1024 px wide — 512 (9 refs), 513 (9), 769 (4),
818 (7), 1022 (7), 1024 (3) — and the H entries guard on `width > tile`, so at
`tile = 1024` every call takes the untiled path. The corruption grid is a single
576×576 reference, untiled at every tile ≥ 576. Measured, not just argued:

| dial grid twin pair | cells | differing | max abs |
|---|---:|---:|---:|
| era-1 vs **tile 1024** | 4,547,248 | **0** | **0** — and the two parquets share sha256 `1bed24cf…` |
| era-1 vs tile 32 | 4,547,248 | 2,298,351 | 6.278e-01 |

**So for the tiling flip at the production width the dial panel's inputs do not
change, and its outputs therefore cannot. Clause 3 is satisfied by construction —
there is nothing to run.**

At `tile = 32` the grid does move, so the dial panel was run on that twin. **No
model crosses a G3 bound**: `mono_pct` moves by at most ±0.0023 and stays
0.9735–0.9943 against the ≥0.93 gate; `tied_pct` is unchanged for five models and
*improves* for C944 (0.0376 → 0.0393 — still far under the ≤0.05 gate);
`dynamic_range` moves ≤0.051 of 60–85 points; `reach` ≤0.22. Even at maximum
tile-edge density the dial gates hold.

### 5.2 Radius 4: clause 3 was OPEN, and this lane closed it

The blur/radius lane registered the radius dial twin and did not run it. Running
it: the radius genuinely moves both grids — **52.05 %** of dial cells
(max |Δ| 0.942) and **52.16 %** of corruption cells (max |Δ| 67.25) — and the
panel comes back with **no gate flip on any model**
([`dial_clause_r4.txt`](era2_rank_preservation_2026-08-31/dial_clause_r4.txt)):

| model | G3 mono_pct r5 → r4 | G3 tied_pct | G1 dynamic_range Δ | G4 reach Δ | corruption pass_q10 Δ |
|---|---|---|---:|---:|---:|
| B | 0.96661 → 0.96810 | 0 → 0 | **−1.296** | −0.438 | — |
| C944 | 0.99319 → 0.99341 | 0.03764 → 0.03488 | +1.299 | +0.672 | **−0.00595** |
| WLIN7b_g020 | 0.99426 → 0.99426 | 0 → 0 | +0.475 | +3.826 | +0.01637 |
| WLIN7b_g025 | 0.99405 → 0.99383 | 0 → 0 | +1.991 | +3.152 | +0.01042 |
| ADD156 | 0.99596 → 0.99447 | 0 → 0 | +1.145 | +1.101 | — |
| BHdr | 0.97490 → 0.97363 | 0 → 0 | +1.338 | +1.187 | — |

**Clause 3 does not block radius 4 on any model.** B is the only one that loses
dynamic range (−1.30 of 73.8) and reach; C944 gains both and loses 0.006 of
corruption `pass_q10`. The "—" cells are honest absences, not zeros:
`bake_verdict` emits no corruption block for a 372-input bake, in both arms
alike.

### 5.3 One incidental finding worth recording

Rebuilding the dial grid on today's tree at era-1 settings and comparing to the
**stored** `dial_grid_944col_2026-08-01.parquet` gives **104,107 of 4,547,248
cells differing (0.0023 %) at max |Δ| 7.2e-9**. That is era-1-vs-era-1 drift
across the month since that grid was built — well inside the repo's golden policy
`|Δ| ≤ max(1e-6, 1e-5·scale)`, so it is not a defect, but the canonical dial grid
is very slightly stale relative to HEAD and someone should know that before
treating its last digits as fixed.

---

## 6. What is NOT measured, and why

1. **The era-2 accumulation shape** — fixed 8 virtual lanes, the `era2_reduce8`
   tree, and `ERA2_BAND_ROWS` banding. **Not reachable.** The kernel exists and
   is gated (§11's oracle, §12.4's cross-tier identity, §14.3's band-merge
   structural gate all pass on it), but every call site is `#[cfg(test/oracle)]`
   and the production dense path calls era-1 unconditionally. **This lane cannot
   clear the bar for the largest semantic component of the break until that
   kernel is wired behind a runtime switch** (a `ZENSIM_ERA2_DENSE`-style env
   knob on the four production call sites would be enough, and would make the
   measurement a re-run of §2's arms, ~70 s per arm).
2. **The V-plane redirect (F4)** — not landed. §24.5's own reading is that it may
   need no rank gate at all if the edge case is bit-identical.
3. **imazen26 / nonphoto / hfnlproxy** — three of the campaign's twelve corpora.
   Their 944 slices come from the bigcodec test views rather than a local pairs
   TSV, so they are outside this extraction. **Listed as absent, not counted as
   passes** — the same scoping the blur/radius lane used.
4. **The HDR panel.** `BHdr` was scored on the SDR corpora only. Its HDR reads
   (UPIQ, SI-HDR, Korshunov, …) belong to
   `scripts/external_reads/run_external_reads.py` and were not run.
5. **Item E1's rank cost** is not re-derived here — the E+F table carries the
   frontier lane's numbers. E1 is a per-request compute-set choice expressed
   through item D, not a global redefinition, so it is a different shape of
   question from the one this bar asks.
6. **Any retrain.** None was launched. §4.5 says why the same-weights read is
   the strict form of the test and also its limitation.

---

## 7. Method appendix

**Box.** Ryzen 9 9950X3D, 16C/32T, 60 GiB, WSL2; extraction tier `v4x`. Nothing
in this note is a timing measurement, so the §22.5 ASLR protocol does not apply —
every number is a deterministic function of bytes.

**Commit.** `9e52fb164c28725a6f12d911707b8caaeaac995e`, in the sibling jj
workspace `zensim--era2-rank` with its own `CARGO_TARGET_DIR`. `main` did not
move under this run. The radius arms apply
`benchmarks/blur_radius_locality_branches_2026-08-31/patch_radius.sh <wt> 4` and
revert to 5; the revert is verified two ways (all four patched files byte-identical
to pre-patch copies, and a fresh cid22 extraction byte-identical to the `era1`
arm).

**Everything runs through the owner tools.** `v2_ab_extract` +
`promote_ext944_canonical.py` produce every feature table; `bake_verdict
--regime 944 --full-json` produces every statistic (routing to
`zensim_validate::panel` → `zenstats`); `build_dial944.py` / `build_corr944.py`
produce every grid twin. The one script this lane wrote,
`era2_rank_table.py`, reads `srocc_signed` and `composite` out of the fullevals
and arranges them — it computes nothing.

**Artifacts.** Feature roots (7 × 141 MB, not committed):
`/mnt/v/zen/zensim-training/era2rank-{era1,e2t1024,e2t256,e2t32,r4ctl,r4t1024,r4t32}-2026-08-31/`.
CSVs, verdicts, dial/corruption grid twins, and the run manifest:
`/mnt/v/output/zensim/era2-rank-2026-08-31/` (`_MANIFEST.json` carries
`build_commit`, per-arm env, per-leg parquet sha256, per-bake sha256, and grid
sha256). Pointer: [`era2_rank_preservation_2026-08-31.pointer.md`](era2_rank_preservation_2026-08-31.pointer.md).
Committed tabulations and drivers:
[`era2_rank_preservation_2026-08-31/`](era2_rank_preservation_2026-08-31/).
