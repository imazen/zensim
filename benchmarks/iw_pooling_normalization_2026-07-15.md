# The IW block is pooled with the wrong denominator — 144 of 372 features carry a per-reference scale error

**2026-07-15.** Found while answering "let's design the best models, with good
data this time". The answer to that question turns out to be upstream of
architecture: **the features are the data, and 144 of them are mis-pooled.**

Status: **the defect is proven from source and the direction is measured. The
shipped magnitude is NOT yet measured** (see §4 — one honest caveat, and it is
the caveat that gates the retrain decision). Nothing here has been fixed.

---

## 1. The defect

`zensim/src/iw_pool.rs` implements a real weighted mean:

```rust
#[allow(dead_code)] // tests-only reference implementation; hot path is fused into streaming
impl WeightedPool {
    /// Weighted mean: `(Σ w_i v_i) / Σ w_i`.
```

`zensim/src/streaming.rs::finalize` — **the shipped hot path** — pools the same
accumulators like this:

```rust
let one_over_n = 1.0 / self.n as f64;
...
iw_ssim[c * 3]     = self.iw_ssim_d[c]  * one_over_n;                 // MEAN
iw_ssim[c * 3 + 1] = (self.iw_ssim_d4[c] * one_over_n).max(0.0).powf(0.25);
iw_ssim[c * 3 + 2] = (self.iw_ssim_d2[c] * one_over_n).max(0.0).sqrt();
iw_art_4th[c]      = (self.iw_art4[c]  * one_over_n).max(0.0).powf(0.25);
iw_det_4th[c]      = (self.iw_det4[c]  * one_over_n).max(0.0).powf(0.25);
iw_mse[c]          = self.iw_mse[c]    * one_over_n;                  // MEAN
```

It accumulates `w·v` and divides by `n`. So:

```
  shipped = Σ(w·v)/n = (Σ(w·v)/Σw) · (Σw/n) = ref · mean_w
```

and the moment exponent carries through:

| IW feature | pooling | shipped/ref factor |
|---|---|---|
| `iw_ssim_mean`, `iw_mse` | mean | `mean_w` |
| `iw_ssim_2nd` | 2nd (`.sqrt()`) | `mean_w^0.5` |
| `iw_ssim_4th`, `iw_art_4th`, `iw_det_4th` | 4th (`.powf(0.25)`) | `mean_w^0.25` |

**`streaming.rs` never accumulates `Σw` at all** (`grep w_sum` finds only
unrelated row sums). It is not a wrong divisor — the quantity does not exist.
The fix requires a new accumulator, not a changed constant.

`Σ(w·v)/n` is not meaningless; it is *mean-of-weighted-values*. But it is not a
weighted mean, and it conflates the distortion being measured with the
reference's activity.

## 2. Why it is a **cross-image** error specifically

The weights derive from the **reference** only — a blurred reference-activity
map (`streaming.rs` step 2, "the per-pixel blurred reference-activity signal
shared by both mask and iw_weight"), and in the reference impl
`compute_iw_weights(ref_plane, …)` takes only the reference plane.

So `mean_w` is a **per-reference constant**. That means the error:

- **leaves within-image ranking exactly intact** — it is a monotone scaling
  applied identically to every distorted version of one reference;
- **corrupts cross-image ranking** — each reference gets a different scale.

Pooled SROCC is a cross-image statistic. Per-ref SROCC is not. So the defect is
*structurally invisible* to the per-ref view and lands entirely on the pooled
number — which is our headline metric and our stated gap vs ssim2 (CID22 0.8764
vs 0.8894). This is the same pooled-vs-per-ref axis the CID22 paper documents
(0.79 pooled / 0.93 per-ref for CVVDP) and that `bake_verdict` prints both of.

It also predicts an architecture asymmetry: the model receives a **product**
(signal × reference-property). A linear model cannot undo a product. An MLP can
approximate it *if* some other feature correlates with `mean_w`. Consistent
with — but NOT established by — the measured "2-layer diverse MLP beats linear B
on non-photo by +0.089" (`benchmarks/blend_2layer_methodology_2026-07-15.md`).
Do not cite that as evidence for this mechanism; it is a hypothesis this defect
makes, not a result.

## 3. Measured: `mean_w` varies a lot

`zensim/src/iw_pool.rs::tests::iw_mean_weight_spread_across_references`
(`#[ignore]`; `ZENSIM_IW_REF_DIR=<dir> cargo test -p zensim --release
iw_mean_weight -- --ignored --nocapture`).

60 CID22 reference images (`/mnt/v/dataset/cid22/CID22/original`), default
`IwWeightConfig`:

| stat | `mean_w` | `mean_w^0.25` |
|---|--:|--:|
| min | 0.001197 | 0.1860 |
| p25 | 0.003502 | 0.2433 |
| p50 | 0.004770 | 0.2628 |
| p75 | 0.006743 | 0.2866 |
| max | 0.018325 | 0.3679 |

`mean_w` spans **15.3×** across CID22's own references. Implied cross-image
scale error, by moment:

| moment | factor | spread |
|---|---|--:|
| mean (`iw_ssim_mean`, `iw_mse`) | `mean_w` | **15.3×** |
| 2nd | `mean_w^0.5` | **3.9×** |
| 4th | `mean_w^0.25` | **1.98×** |

Extremes: `1292115.png` (mean_w 0.001197) vs `1454613116.png` (0.018325).

## 4. The caveat that gates the decision — READ THIS BEFORE ACTING

**The 15.3× was measured with `iw_pool::compute_iw_weights`. The shipped path
does not call it.** `streaming.rs` derives weights inline from the blurred
activity map via `k_iw` ("all weights derived inline from activity. Mask plane
is NEVER materialized"). That is a different function, and it is the whole
reason the two implementations could drift apart unnoticed in the first place.

So what is proven vs not:

| claim | status |
|---|---|
| shipped divides IW accumulators by `n`, never `Σw` | **PROVEN** (source) |
| `streaming.rs` has no `Σw` accumulator | **PROVEN** (source) |
| the correct weighted mean exists and is dead code | **PROVEN** (source) |
| no test holds the two impls together | **PROVEN** (grep) |
| weights derive from the reference only ⇒ per-reference factor | **PROVEN** (source, both impls) |
| `mean_w` spans 15.3× on CID22 refs **under `iw_pool`'s weights** | **MEASURED** |
| the SHIPPED `k_iw` weights have that spread | **NOT MEASURED** |
| fixing it improves any SROCC | **NOT MEASURED** |

Next step is therefore **not** a retrain. It is: instrument `streaming.rs` to
also accumulate `Σw`, emit `mean_w` per image, and re-measure §3 against the
weights we actually ship. That is a small change and it settles the magnitude.

## 5. The cheap experiment (no re-extraction)

If the shipped factor is confirmed wide, the hypothesis can be tested **without
re-extracting any distorted image**, because `shipped = ref · mean_w^p` and
`mean_w` depends only on the reference:

1. compute `mean_w` once per reference image in a corpus (49 for CID22 val);
2. divide each IW feature in the EXISTING parquet by `mean_w^p` (p = 1, 0.5,
   0.25 per the §1 table) — pure arithmetic on stored columns;
3. refit the same linear head on corrected vs uncorrected features;
4. compare **pooled** CID22 SROCC (the prediction: pooled improves, per-ref is
   unchanged — a per-ref change would falsify the mechanism).

This turns a multi-day re-extract into an afternoon, and it is falsifiable in
the right direction: the mechanism predicts *which* statistic moves, not just
that something gets better.

## 6. Why this is also a duplication finding

CLAUDE.md permits a second implementation **only** when "the mirror exists to
solve a *specific measured* problem the owner can't, and a test fails the build
the moment they diverge. Without that test it is not a mirror, it is a fork with
a good story."

`iw_pool.rs` is a fork with a good story. Its comment says "hot path is fused
into streaming" — asserting agreement — and nothing checks it. The correct
implementation is the one marked `dead_code`.

Fifth instance of `docs/REPRODUCIBILITY.md` §5's pattern: **the careful half
gets built, the adjacent check does not, and the gap is invisible precisely
because the careful half looks like diligence.** A hand-written reference
implementation of a weighted mean *is* diligence — it just was never wired to
anything that would notice it disagreeing.

## 7. Relationship to §3.19 and the winsor guard

`docs/DATASET_HISTORY.md` §3.19 records the `1/n`-vs-`Σw` bug and the unbounded
`iw_art4`/`iw_det4` energies, calling the winsor guard "load-bearing and
sufficient" and the `Σw` switch "a scheduled full-retrain fix". Both stand.
What this doc adds:

- the **per-reference ⇒ cross-image-only** structure of the error, and the
  falsifiable pooled-vs-per-ref prediction that follows;
- the **per-moment** breakdown (the mean features are off by `mean_w`, not
  `mean_w^0.25` — a much larger factor than §3.19's "~1.5–2×" implies);
- that **no `Σw` accumulator exists**, so the fix is additive, not a one-liner;
- the **cheap test** in §5 that settles it without a full re-extract.

"Sufficient" in §3.19 meant sufficient to ship B. It was never a claim that the
features are right — and a winsor guard *clamps* these features, which is a
band-aid over the same wound.
