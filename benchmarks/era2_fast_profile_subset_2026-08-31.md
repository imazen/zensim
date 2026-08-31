# What subset should a high-speed model be limited to?

**Decision table for the user.** Costs are MEASURED here on the era-2 shipped
build; quality numbers are the campaign's, attributed and not re-derived.
Nothing is shipped — this is the curve and the recommendation.

---

## 1. The measured cost curve

2304², min over 5 process starts per cell, `min` of 7 walks per process,
CCD-pinned, era-2 defaults (tile 1024 + fixed-lane accumulation). Peak RSS from
`smaps_rollup` at 1T.

| compute set | what it computes | 1T | 8T | 16T | peak RSS |
|---|---|---:|---:|---:|---:|
| **`944full`** (today) | basic + **pools(Full)** + v2-348 + append + append2 + csfw | 278.0 | 124.0 | 113.6 | 105,084 kB |
| `944carriers` | as above, pools reduced to the 10 carrier slots | 286.3 | 102.8 | 108.0 | 105,092 kB |
| **`944peaks`** | as above, **pool pass dropped**, free peaks kept | **246.8** | **101.8** | **96.6** | 98,320 kB |
| `372` | basic + pools(Full), **no v2-era block** | 155.5 | 44.1 | 42.9 | 89,828 kB |
| **`156`** | basic + **free peaks only**, no v2-era block | **109.6** | **28.0** | **32.2** | **82,672 kB** |

Speed-up against `944full`, and RSS saved:

| | 1T | 8T | 16T | RSS |
|---|---:|---:|---:|---:|
| `944peaks` | 1.13× | **1.22×** | 1.18× | −6.4 % |
| `372` | 1.79× | 2.81× | 2.65× | −14.5 % |
| **`156`** | **2.54×** | **4.43×** | **3.52×** | **−21.3 %** |

Three things the table says that the prose did not:

* **Dropping the pool pass is worth 1.13–1.22×**, which independently
  reproduces the campaign's "41.2 ms = 13.6 % of the tiled walk"
  (1/(1−0.136) = 1.157×).
* **`944carriers` is NOT a cheap middle.** At 1T it is *slower than
  `944full`* (286.3 vs 278.0) — the carrier subset still runs the masked/IW
  kernel at scales 0–1, so it pays the pass without amortising it. It only
  helps at ≥8T. **Do not offer it as the fast option.**
* **The basic-only class scales best**: its advantage grows from 2.54× at 1T
  to 4.43× at 8T, because what it removes (the v2 plane pipeline) is the part
  that contends for bandwidth across threads.

## 2. What each subset gives up

Quality is the campaign's measurement, not this note's. Cited, with the source
lane named; cells this lane cannot supply say so rather than being estimated.

| subset | 944 MLPs | W-LIN blend | shipped `B` | source |
|---|---|---|---|---|
| drop the pool pass (`944peaks`) | **exactly 0** | −0.005 CID22 / −0.048 KonJND | **−0.399 CID22 / −0.525 KonJND** | frontier lane |
| drop v2-348 + append (`372`, `156`) | **not applicable** — they are defined on it | **−0.745 CID22** | n/a (B is basic-class) | frontier lane |
| basic-only vs `B` (`156`) | n/a | n/a | **within 0.019 CID22**, and **beats `B` on within-image ranking on 7 of 8 corpora** | frontier lane |
| per-corpus deltas vs `B` and vs ssim2 | **NOT MEASURED HERE** — ATTACH from the frontier lane's per-corpus table | | | |

**The pool row is the whole reason item D exists.** The same 13.6 % is worth
*nothing* to a 944 MLP and 0.399 CID22 to `B`. There is no global right answer;
there is only a per-model one.

## 3. Recommendation

**Two points on the curve, not one, because they answer different questions.**

### 3.1 `156` — basic + free peaks — for a genuine fast profile

**2.54× / 4.43× / 3.52× at 1/8/16T, −21 % peak RSS**, paired with an
**ADD156-class model**, which lands **within 0.019 CID22 of `B`** and **beats
it on within-image ranking on 7 of 8 corpora**. Within-image ranking is what a
codec loop actually consumes, so on the loop's own criterion this is not a
downgrade at all.

Keep the peaks: they are **free byproducts** of accumulators the basic pass
runs anyway, so dropping them saves nothing and costs the peak-weighted slots.

### 3.2 `944peaks` — for callers that must keep a 944 MLP

**1.13–1.22× for exactly zero rank cost** on the 944 MLPs, because they read
the pool slots with zero weight. This one is free and needs no model change —
it is the "make the current thing faster" option, and it should be the
**default for any 944-MLP request** once the derivation in §4 exists.

### 3.3 Not recommended

`944carriers` (slower than full at 1T), and dropping peaks (saves nothing).
`372` is dominated: it costs the v2 block's quality *and* keeps the pool pass
that a 944 MLP does not want and `B` does.

## 4. The API shape — no new public types

Item D's `ComputeSet` already expresses all of this. The cheapest shipping
form, recommended over promoting anything:

```rust
// inside the existing entry points — the caller passes a model handle already
let compute = ComputeSet::from_block_profile(model);   // <- the one new fn
```

`from_block_profile` reads the model's own block profile — the same
`bake_block_profile` the dashboard already uses — and switches off any family
the model reads with zero weight. That makes §3.2 **automatic**: a 944 MLP
stops paying for the pool pass without anyone choosing a flag, and `B` keeps
it because `B` actually uses it.

For §3.1 the selection is a **profile**, not a flag: an ADD156-class profile
carries an ADD156 bake, and `from_block_profile` then derives `156`
by itself. **No new public type, no new public entry point.** The full
proposed-surface alternative (promote `ComputeSet`, add `compute_*_with_set`)
stays listed in era-2 §26.1 and is not needed for this.

## 5. What it would take to ship

1. **`ComputeSet::from_block_profile`** — needs a `zenpredict::Model`, so it is
   a decision about which crate owns the derivation. That is the only code.
2. **A gate that the derivation never removes a family the model reads.**
   Shape: for each shipped bake, assert the derived compute set's dropped
   families all have zero weight in the block profile. Cheap, and it is what
   makes the automatic form safe.
3. **A fast-profile bake.** `156` is only a product if an ADD156-class model is
   published behind it; today ADD156 is a campaign bake, not a profile.

### 5.1 Registered, NOT launched: the fast-profile retrain

**Is a retrain on the reduced set warranted? Probably yes, and it is cheap —
but it is not required to ship §3.1.**

* **Not required**, because ADD156 is *already* trained on the basic families
  only. Restricting the compute set does not change its inputs; it stops
  computing slots the model never reads. Its 0.019-CID22 gap to `B` is a
  property of the model, not of the compute set.
* **Warranted**, because every current bake was trained on **era-1** features
  and the era-2 flip has now moved them (`f372+` for the accumulation; the
  tiled H planes for any reference wider than 1024). A fast-profile bake
  trained *at era-2 on the reduced set* is the honest version of this product.

**Wave `W-FAST-1`** (registered, unlaunched): retrain an ADD156-class bake at
era-2 on the `156` compute set; gate on the §21.1 bar against `B`, plus the
within-image ranking comparison that is this profile's actual selling point.
It shares the era-2 re-extraction with `benchmarks/era2_blast_radius_2026-08-31.md`
§2.3 — if that wave runs, this arm is nearly free to add.
