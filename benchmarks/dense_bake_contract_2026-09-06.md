# The dense bake contract — and the train/serve skew it found on shipped C

Increment B of [`../docs/PLAN_CRUFT_PURGE_2026-09-06.md`](../docs/PLAN_CRUFT_PURGE_2026-09-06.md).
Tool: `bake_dial_refit densify`. Artifacts:
`/mnt/v/output/zensim/purge-2026-09-06/`.

---

## 1. What densify does

Rewrites a bake so that **`caller_input_width() == n_inputs() == |read set|`**,
with **zero `FeatureTransform::Drop`**, and a `zentrain.feature_ids` metadata
entry naming the ids it reads (ascending decimal, one per line). The runtime
resolves that declaration through `feature_layout::declared_layout` to a dense
`Layout` and **gathers** the walk's vector into it — instead of handing over a
wide vector the bake discards a quarter of.

`zentrain.feature_ids` is a **metadata convention**, not a format change: ZNPR v3
metadata is free-form utf8 key/value and `zenpredict` is untouched. It is
preferred over `zentrain.feature_set_id` because a family-token id can only name a
family UNION, so it cannot express Profile D's 28 scattered `basic` slots.
The parse is strict — strictly ascending, in range, no duplicates — and anything
else falls back to the identity layout, because a half-believed id list permutes
the vector a bake is served.

## 2. Why removing a weight-dead input row is bit-exact

`zenpredict`'s matmul is a **SAXPY over inputs**
(`zenanalyze/zenpredict/src/inference.rs`): `for i in 0..in_dim { for k { dst[k] =
fma(src[i], w[i][k], dst[k]) } }`, vectorized across the OUTPUT dimension. Removing
an input removes exactly the terms `fma(s, 0.0, acc)` from each accumulator, **in
place**, leaving every surviving term's order unchanged. Only class-1 pruning
(whole layer-0 row exactly `0.0`) is used; class 2 folds into the bias and reorders
one f32 sum, so it is off.

## 3. Two gates, and one of them corrected a doc claim

**DEAD-COLUMN GATE (the strong one).** Score the ORIGINAL bake with every dropped
line's value replaced by `0.0`; that must be bit-identical to the densified bake on
**every** probe row. This isolates "the columns are dead" from "the probe happened
to avoid the difference". **PASSED on all 11 shipped bakes, 512 rows each.**

**IDENTITY GATE (same input, both bakes).** Bit-identical on 9 of 11; on
`v47_strict_qat_native` (A) and `bhdr_linear_shaped_anchored2` **16 of 512 rows
differ, and every one of them is a NaN absorption**: the probe's ±2e3 domain drives
`log1p`/`yeo_johnson` out of domain on a DROPPED line, and `fma(NaN, 0.0, acc)` is
**NaN**, so the wide bake poisons its accumulator where the dense one has nothing to
poison it with.

**That falsifies a doc claim this repo relied on.** `zensim-validate/src/prune.rs`
said class-1 pruning is *"bit-identical for **every** input including NaN"*. It is
not; class 1 has the same NaN caveat class 2's doc already carried, and only class
2's was written down. Corrected in place, citing this measurement. Both only change
behaviour on input that was already garbage — the defect was the false universal
claim, and a gate believed it.

## 4. MEASURED: 8 of 11 densify with a bit-identical served score. Three do not.

`serve_custom_bake --census` through the production `Zensim::compute`, on one real
CID22 pair (`162520.png` vs `libjxl/e7_q30.png`):

| bake | declared: wide → dense | served, wide | served, dense | Δ |
|---|---|--:|--:|--:|
| `v47_strict_qat_native` (**A**) | 372 → 285 | 41.468519 | 41.468519 | **0** |
| `b_sdr_..._inclwinsor_dense_dial` (**B**) | 372 → 95 | 41.762495 | 41.762495 | **0** |
| `bhdr_..._cvvdpmix` (**BHdr**) | 372 → 133 | 52.296464 | 52.296464 | **0** |
| `d_sdr_add156_id100_negrich_dial` (**D**) | 372 → **28** | 47.610832 | 47.610832 | **0** |
| `b_sdr_..._dense_dial` (retired) | 372 → 95 | 41.750064 | 41.750064 | **0** |
| `b_sdr_..._anchored` (retired) | 372 → 95 | 41.700967 | 41.700967 | **0** |
| `bhdr_..._anchored2` (retired) | 372 → 50 | 46.376163 | 46.376163 | **0** |
| `d_sdr_add156_dense_dial` (retired) | 372 → 28 | 47.793587 | 47.793587 | **0** |
| `c_sdr_purity944` (**C**) | 944 → 667 | 47.743838 | 48.609764 | **+0.866** |
| `c_sdr_mlp944_corrmix` (retired) | 944 → 667 | 46.030840 | 46.108342 | +0.078 |
| `c_hdr_l1t1944` (**CHdr**) | 944 → 697 | 68.980064 | 68.669243 | **−0.311** |

Census: **11 SERVED, 0 REFUSED** in both arms.

## 5. THE FINDING — shipped Profile C and CHdr serve on a different BANDVIS formula than they were trained on

The three that move are exactly the three append2-bearing bakes, and the cause is a
single compute flag. `Plan::for_bake` derives COMPUTE two different ways depending
on the layout, and the two disagree:

```
c_sdr_purity944, IDENTITY layout w944  (from_block_profile -> `everything`)
  append2_dst_activity: true      v1_pools: Full
c_sdr_purity944, DENSE layout dense667 (Plan::derive_with_layout, id space)
  append2_dst_activity: false     v1_pools: Off
```

`v1_pools` is score-neutral here — `bake_block_profile` reports `f156_371: 216
cols, 216 exact0, **0 used**` for both C bakes. `append2_dst_activity` is not: it
selects the `BV_DSTACT` BANDVIS combine (`feature_v2.rs:3689, 3778`) and so changes
the **values** of `f924..f943`, which C reads.

**Which one is right? The canonical extractor says OFF.**
`zensim/examples/v2_ab_extract.rs:414` computes `dstact_on` from
`ZENSIM_APPEND2_DSTACT`, defaulting to **false**, and
`scripts/canonical_corpus/extract_944_canonical.sh` never sets it. So every
canonical 944 table — the data C and CHdr were trained and evaluated on — was
extracted with `append2_dst_activity = false`. CLAUDE.md's own adjudication says
the same: *"extraction stays toggle-OFF (both mask arms failed gates)"*.

**The runtime disagrees with the training data.** `ComputeSet::from_block_profile`
falls back to `everything` for any wide bake it cannot narrow, and `everything`
hard-sets `append2_dst_activity: true` (`feature_v2.rs:2163`). So the SHIPPED
runtime path for C and CHdr computes a BANDVIS formula their weights never saw.

This is **pre-existing** and has nothing to do with densify — densify EXPOSED it,
because the id-space derivation is honest about what the read set implies and the
`everything` fallback is a shrug. It is the same bug class the ruling names: what
is computed does not match what is declared.

**Blast radius, measured on one pair: 0.87 zensim points on C, 0.31 on CHdr.**
Both shipped. Not a rounding difference.

### 5.1 Consequence for this increment, stated honestly

Densifying C and CHdr **cannot** satisfy this lane's own pre-registered P-SCORE
gate ("no shipped bake's prediction moves"), because the honest derivation produces
different numbers than the buggy one. The two available moves both need a decision
that is not a lane's to make:

* **(a) Fix `everything`** — one line, `append2_dst_activity: true → false`. This
  moves shipped C and CHdr scores to their train-consistent values. It is very
  probably the right fix, and it is still a shipped-score change.
* **(b) Make the dense derivation reproduce `true`** — perpetuates the skew in a
  new place.

**So C and CHdr are NOT densified here.** They are reported, with the measurement,
as a registered blocker. The other nine bakes — including shipped **A**, **B**,
**BHdr** and **D** — densify with a bit-identical served score.

### 5.2 What was ruled out, by measurement rather than argument

* **Not the walk width.** `c_hdr_l1t1944` declares ids up to `f943`, so its dense
  `walk_width()` is 944 — identical to the wide arm — and it still moves. (The two
  667-input bakes reach only `f941`, so they walk 942; `LayoutBlocks::for_width`
  turns `append2` on above 924, so 942 and 944 agree on every block.)
* **Not the pools.** Both C bakes read 0 of the 216 `f156..371` lines.
* **Not the gather.** The eight non-append2 bakes gather 28–285 scattered ids out of
  a 372-wide walk and land bit-identical.
