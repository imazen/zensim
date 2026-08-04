# Dead-column pruning in `bake_dial_refit pack` — 2026-08-04

**What shipped:** `pack` now removes layer-0 inputs that *cannot* change a
prediction, automatically, in the same pass as zerobias + dtype + spline refit.
On all three sota944 ship candidates that is **944 → 667 layer-0 inputs (277
dropped, 29.3 %)**, with **bit-identical predictions** and **byte-identical
verdicts**.

**Headline correction up front.** The motivating `bake_contrib` figure —
"73,128 B, ~44 % of the packed encoder" — is a **decompressed** measurement.
The ZNPR payload is LZ4-compressed and a run of zero `f16` weights compresses
to nearly nothing, so the *file* only shrinks by 382–478 B (0.2–0.3 %). The
real wins are **inference** (29.3 % fewer layer-0 SAXPY rows, which compression
cannot give you) and **resident footprint** (−73,128 B decompressed). Do not
quote the 73 KB as a file-size saving.

---

## 1. Why it is expressible at all — the format determination

The ZNPR v3 format needed **no wire change**. What it needed was a way to say
"input `k` is accepted from the caller and contributes nothing", and the
`zentrain.feature_transforms` metadata blob was already the right place: it is
line-aligned to the *raw* input width, and `Sinusoidal` had already established
that a transform's output arity need not be 1 (the loader cross-checks
`sum(output_arity) == n_inputs`, `zenpredict/src/model.rs:682-711`).

What v3 could express before this change:

| section | what it does | can it express an input-index map? |
|---|---|---|
| `feature_order` (v3.1, header @100) | **permutation** of inputs, `perm.len() == n_inputs` enforced (`composer.rs:203`), *inverted at load* (`model.rs:1458 apply_feature_order_inverse`) | **No** — it reorders for compressibility; the width is unchanged and the mapping is undone before inference. |
| `feature_bounds` | `n_inputs` × `(low, high)` OOD bounds, validated at `model.rs:655`; **not** applied in `forward()` | No — a report surface, not a gather. |
| `sparse_overrides` | `(output_idx, value)` post-processing, applied after the output specs (`predictor.rs:196`) | No — output side. |
| `output_specs` / `discrete_sets` | per-**output** clamp/activation/snap | No — output side. |
| `feature_transforms` + `_params` | per-raw-input transform, **variable arity** since `Sinusoidal` | **Yes** — this is the one. |

So the change is a new *token*, not a new section:
**`FeatureTransform::Drop`** (zenanalyze `88410ba6`), arity 0. A pruned bake
declares `drop` on the dead raw lines; `caller_input_width()` stays 944 while
`n_inputs()` becomes 667.

Arity 0 was *technically* already reachable as "`sinusoidal` with zero
frequencies" — `output_arity(&[]) == 0`, and `required_param_arity(Sinusoidal)`
returned `None` so the composer let it through. That spelling was deliberately
**not** used: the variant's own docs say "with `params = []` the variant
produces no output (arity 0); a bake-side validator SHOULD reject this"
(`feature_transform.rs`), so shipping bakes whose correctness depends on that
validator never being written would have been building on sand. The same commit
*implements* that rejection and points it at `drop`, so arity 0 now has exactly
one spelling.

### The accessor that makes it safe

`Model::caller_input_width()` = `feature_transforms().len()` when present, else
`n_inputs()`. **Size every feature vector by this.** The two diverge in both
directions once variable-arity transforms exist (`Sinusoidal` ⇒ larger, `Drop`
⇒ smaller). Feeding `n_inputs()` to a pruned bake raises
`FeatureLenMismatch` — loud, never a silent prefix — but the product runtime's
width dispatch had an `n_inputs < features.len()` PREFIX branch
(`zensim/src/metric.rs`) that would have truncated first, so every
"how many features do I feed" site in this repo was converted.

---

## 2. The three classes of "dead" — only two are prunable

This is the whole correctness story. `bake_contrib` measures what a bake is
*effectively ignoring on a corpus*; that is a strict superset of what is safe
to remove.

| class | test | prunable | guarantee |
|---|---|---|---|
| **1 weight-dead** | `W0[k,:]` is exactly `0.0` | yes | **bit-identical** — `fma(x̃, 0, acc) == acc` |
| **2 transform-forced-constant** | the bake's OWN transform maps every input to one constant `c` | yes | exact in real arithmetic; the contribution `x̃(c)·W0[k,:]` folds into `b0`, which reorders one `f32` sum, so *not* bit-identical |
| **3 inert on this corpus** | mean\|Δ\|≈0 under ablation, but the weight is live and no transform pins it | **NO** | the corpus merely never exercised it |

Class 3 is the trap: **in a corpus report it is indistinguishable from class 1**
and it is not mathematically dead. `zensim-validate/src/prune.rs` makes it
structurally unreachable rather than merely discouraged — **`prune::plan()`
takes no corpus statistic as an input at all.** Every decision comes from the
bake's own weights, transforms and scaler.

`tests/prune_classes.rs` carries a fixture with one input of each shape. The
load-bearing assertion is the negative one:

```rust
// input 3 is constant across every probe, has a live weight, no forcing transform
assert!(plan.keep.contains(&3), "class-3 input 3 MUST be retained");
// ...and it really is live: perturbing it moves the score
assert_ne!(base.to_bits(), moved.to_bits());
```

Plus: class 1 alone is asserted bit-identical; class 2 reproduces the full model
to fp tolerance; pruning is idempotent (safe to re-run `pack` on its own
output); an `n_inputs()`-sized vector is refused; and **class 2 is refused
outright on an `i8` layer 0** — removing a *nonzero* row can move the
per-output max-abs quantization scale and thus re-quantize every other weight.
Class-1 rows are all-zero and cannot hold the max, so they stay safe on i8.

### Class detection, both gates

`forced_constant()` requires **both**:

1. **structural** — the variant is in the winsor family and its clamp bounds
   are finite with `lo >= hi` (the runtime `clamp_inclusive` returns `lo` when
   `lo > hi`). `ClipThenLog1pThenWinsor`'s bounds are params `[1]`/`[2]`, not
   `[0]`/`[1]` — `[eps, q_lo, q_hi]`.
2. **empirical** — all 13 spread probes (incl. ±0, ±1e-30, ±FLT_MAX, ±inf)
   must produce bit-identical output from the *real* `apply_with_params`.

**Gate 2 is the one that decides, and that is not academic.** Two shapes pass
gate 1 and are *not* constant, because `WinsorP99` is a hand-rolled
`if x < lo {lo} else if x > hi {hi} else {x}` and **not** the `clamp_inclusive`
the stacked variants use:

* **`[0, 0]`** — `-0.0 < 0.0` is false and `-0.0 > 0.0` is false, so `-0.0`
  falls through the `else` and comes back out as `-0.0`. Not bit-constant.
  This is exactly the shape of the 24 `winsor_p99:[0,0]` columns in the
  sota944 bakes; they are pruned as class 1 anyway, so being strict costs
  nothing here.
* **`[7, 2]`** (`lo > hi`) — returns `lo` for small `x` and `hi` for large `x`.
  Two values. `clamp_inclusive` would have returned only `lo`, so the *stacked*
  variants behave differently from `WinsorP99` on the same params.

The family is not uniform; params-only reasoning mis-classifies both. Gate 2
alone, conversely, could false-positive on a transform that saturates across
the probe set. (Both cases are pinned by
`prune::tests::the_empirical_gate_rejects_winsor_cases_the_structural_gate_admits`.)

`NaN` is deliberately **not** probed: `clamp_inclusive` propagates NaN, so a
NaN feature is not mapped to the constant. That is the one respect in which
class 2 is weaker than class 1, and it only differs on input that was already
garbage (the unpruned bake returns NaN — no usable score). `--no-prune-constants`
restricts to class 1, which is bit-identical for *every* input including NaN.

---

## 3. The identity gate

Runs on **every** pack, on the anchor corpus, comparing pre- vs post-prune
scores through the full runtime dispatch (`forward_scored_6dec`, i.e. the same
head/pin/spline path as production):

* class 1 only ⇒ demand **exact bit equality** of every score;
* class 2 present ⇒ demand `|Δ| <= --prune-identity-tol` (default 1e-4) and
  report the worst case.

Either way it **refuses to write the bake** on failure.

---

## 4. Measured — the three sota944 ship candidates

Command (per bake), binary `bake_dial_refit`, anchor
`ext944-canonical-2026-08-01/anchor944_dial.parquet` (2035 rows):

```sh
bake_dial_refit pack --in <X>_dial.bin --out <X>_pruned.bin --neg-tail \
  --anchor anchor944_dial.parquet --target-col target_score \
  --verify ext_cid22val.parquet --verify-col human_score --verify-scale 100
```

| bake | layer-0 in | class 1 | class 2 | identity gate | file (`--no-prune`) | file (pruned) | Δ bytes |
|---|---|---|---|---|---|---|---|
| `H_co3abpg_s2507` | 944 → 667 | 277 | 0 | **PASS bit-identical**, 2035/2035 | 165,872 | 165,467 | **−405** (−0.24 %) |
| `C_em944_s31` | 944 → 667 | 277 | 0 | **PASS bit-identical**, 2035/2035 | 172,067 | 171,685 | **−382** (−0.22 %) |
| `C_co3a_s1307` | 944 → 667 | 277 | 0 | **PASS bit-identical**, 2035/2035 | 180,446 | 179,968 | **−478** (−0.26 %) |

Identical 277 on all three, consistent with `bake_contrib`'s finding that the
944 bakes are dead on the *same* inputs.

**All 277 are class 1.** `bake_contrib` decomposed its 277 as 216 structural
zeros + 39 never-populated + 22 winsor-clip; after `--zerobias-bulk 0.005` all
of them have exactly-zero weight rows, so the strictly stronger (bit-identical)
class applies to every one and class 2 never fires on these bakes. The 24
`winsor_p99` columns with params `[0,0]` that *would* have been class 2 are a
subset of the class-1 set — cross-checked by the verdict's
`model.feature_transforms` chip count: 64 non-identity before, 317 after, and
`277 + 64 − 24 = 317`.

Decompressed footprint removed: `277 × (128 × 2 B + 2 × 4 B) = 73,128 B` per
bake.

### `--no-prune` reproduces the shipped artifacts byte-for-byte

| bake | shipped `_packed.bin` sha256 | `pack --no-prune` sha256 |
|---|---|---|
| `H_co3abpg_s2507` | `6d801d13…a323d8fa` | `6d801d13…a323d8fa` ✓ |
| `C_em944_s31` | `5870046d…2f9b9b12` | `5870046d…2f9b9b12` ✓ |
| `C_co3a_s1307` | `6c147fd4…1d92c8bfef`* | `6c147fd4…1d92c8bfef`* ✓ |

*(full: `6c147fd429a82af03a91e84a5ce7dba422f955a889e8c6196860cf1d92c8bfef`)*

So the only behaviour change is pruning; the historical byte-reproduction claim
in `CLAUDE.md` remains true under `--no-prune`.

### Verdicts unchanged — `bake_verdict --regime 944`, full-JSON diff

Every pruned bake was re-verdicted with the identical invocation and its
`--full-json` diffed field-by-field against the shipped `_packed.full.json`.
**Six fields differ on each, all descriptive; zero measured statistics move.**

```
/bake                        (path)
/bake_sha256                 (new bytes)
/model/file_bytes            165872 -> 165467
/model/layers[0]/in          944 -> 667
/model/scaler/n              944 -> 667
/model/feature_transforms    len 64 -> len 317   (the drop chips)
```

Everything else — every SROCC / PLCC / KROCC / OR / PWRC / Z-RMSE, every band
table, every per-codec dial curve, every corruption stat, **every per-pair
prediction** — is byte-identical across all three bakes. Advisory CID22 SROCC
from the pack run itself: 0.8806 / 0.8869 / 0.8857, matching the `--no-prune`
arm exactly.

---

## 5. The inference win — MEASURED

`zensim-validate/examples/prune_forward_bench.rs` (zenbench, interleaved
round-robin so drift hits both arms equally), `C_em944_s31` pruned vs its
`--no-prune` twin, 256 feature rows per iteration through
`Predictor::predict_transformed`:

```
predict_transformed  4 rounds × 1 calls   ⚠ only 4 rounds
              mean ±mad ms   95% CI vs base
 ├─ unpruned   71.6 ±2.8ms   [69.3–73.9]ms
 ╰─ pruned     53.4 ±2.7ms   [-29.6% – -19.1%]
```

**−25.4 % wall time, 95 % CI [−29.6 %, −19.1 %]** — consistent with the 29.3 %
of layer-0 SAXPY rows removed (layer 0 dominates, but the transform pipeline,
layer 1 and per-call fixed cost do not shrink, so the total lands slightly
under the row fraction).

**Caveat, stated plainly:** the box was busy (a concurrent `zenpredict-bake`
bench held zenbench's exclusive lock for ~6 min, and 120 of 124 rounds were
discarded as noisy), so this is a 4-round result flagged by the harness. The
interval excludes zero by a wide margin and the magnitude matches the
structural prediction, but a quiet-box re-run would tighten it.

Why compression cannot deliver this: the forward kernel's
`if s == 0.0 { continue }` fast path does **not** fire on a dead column,
because the standardized input `(x − mean)/scale` is generically nonzero even
when the entire weight row is zero. An un-pruned bake genuinely pays 944 rows
to accumulate 277 rows of zeros.

## 6. Ordering inside `pack`

**zerobias → PRUNE → dtype/quantize → spline refit.**

Zerobias is what *creates* most weight-dead columns (a column of sub-τ weights
becomes exactly zero: L0 zeroed 54,722/120,832 on `C_em944_s31`), so the plan is
built on post-zerobias weights. The spline still lands last, fit on the final
packed net, so the QUANTIZE-then-CALIBRATE invariant is untouched — and because
class-1 pruning is bit-identical, the spline it fits is the same spline.

---

## 7. Reproduce

```sh
cargo build --release -p zensim-validate --bin bake_dial_refit --bin bake_verdict
cargo test  --release -p zensim-validate --test prune_classes

# pruned + un-pruned twin
D=/mnt/v/output/zensim/bakes/sota944/bakes
A=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
bake_dial_refit pack --in $D/C_em944_s31_dial.bin --out pruned.bin  --neg-tail \
  --anchor $A/anchor944_dial.parquet --target-col target_score --verify none
bake_dial_refit pack --in $D/C_em944_s31_dial.bin --out unpruned.bin --no-prune --neg-tail \
  --anchor $A/anchor944_dial.parquet --target-col target_score --verify none

# inference A/B
cargo run --release -p zensim-validate --example prune_forward_bench -- \
  unpruned.bin pruned.bin 256
```
