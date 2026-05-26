# Metric invariants: the V39 inversion mechanism, principled detection, and a redesign that can't regress off-distribution (2026-05-26)

Companion to `benchmarks/ROOT_CAUSE_v39_invariant_violations_2026-05-26.md`.
That doc establishes *what* broke; this one establishes *why* at the
mechanism level (with numbers), *how to detect* this class of failure in
general, *how to guarantee* it cannot occur (the math), and the concrete
architecture/training/gate changes for a generalizable zensim.

> Literature note: the monotone-network / Lipschitz / verification
> citations below are from training knowledge — web search was
> unavailable when this was written. The mathematics is self-contained
> and does not depend on citation precision; double-check exact venues
> before quoting them externally.

---

## 1. The mechanism, quantified

The runtime score is a composition of four stages:

```
score = spline( pin( net( standardize( φ(x,y) ) ) ) )
        └─ S4 ─┘ └S3┘ └S2┘ └──── S1 ────┘ └ S0 features
```

- **S0** `φ(x,y)` — extract 372 perceptual features for the pair.
- **S1** standardize — `z_i = (φ_i − mean_i) / scale_i` using the bake's
  per-feature scaler (`scaler_mean`/`scaler_scale`).
- **S2** `net` — 372→128→64→64 LeakyReLU MLP + per-sample-α head.
- **S3** `pin` — tanh output head `100·σ(y/30)`, bounds **[0,100]**.
- **S4** `spline` — PCHIP `output_calibration_spline`, **linearly
  extrapolating** outside its knots.

Each stage is individually reasonable on-distribution. Composed
*off*-distribution they multiply into an unbounded, order-inverting map.
Three measured facts:

### (A) Synthetic content lands 100+σ off the training manifold

V39's scaler was fit on natural-photo + codec-artifact pairs. Feeding it
a 128×128 Mandelbrot and its blurs (measured directly against the bake's
scaler):

| case | score | max\|z\| | n(\|z\|>3) | n(\|z\|>5) | n(\|z\|>10) | rms z |
|---|--:|--:|--:|--:|--:|--:|
| identical | 100.000 | 3.81 | 27 | 0 | 0 | 1.58 |
| blur r=1 | 124.675 | **106.58** | 77 | 46 | 36 | 11.77 |
| blur r=5 | 158.870 | **134.61** | 155 | 112 | 84 | 17.57 |

Blurred synthetic content drives individual features to **>100 standard
deviations** from the training mean; the whole 372-vector sits at
12–18σ rms. The network is being evaluated in a region of input space it
*never saw* — and `rms z` grows with blur, so "more distortion" reads as
"further off-manifold," which the unconstrained net maps the wrong way.

### (B) The unconstrained MLP inverts off-manifold

A LeakyReLU MLP is a piecewise-linear function with no global shape
constraint. On its training manifold it learned a sensible ranking; at
100σ out, its output is governed by whichever linear piece it lands in —
effectively arbitrary. Here it is monotone *increasing* in distortion.
This is provable from the runtime, not just asserted:

- The PCHIP spline is fit strictly-increasing, hence **monotone**, hence
  order-preserving.
- We observe `score(blur5) > score(blur1) > score(identical)`.
- A monotone S4 cannot create that order. Therefore the **pre-spline
  (post-pin) network output is itself inverted**:
  `pin(net(blur5)) > pin(net(blur1)) > pin(net(identical-region))`.

The linear `PreviewV0_2` consumes the *same* S0 features and stays
monotone+bounded on every content type — so S0 is fine and the defect is
entirely in S2 (+ amplified by S4). This is the IQA instantiation of the
well-documented fragility of learned perceptual metrics off their
training distribution (cf. R-LPIPS, adversarial-robust LPIPS).

### (C) The output stage is a ~1000× extrapolating amplifier

The shipped spline has **3 knots** at x ∈ {49.8297, 49.8467, 49.8631}
(width **0.0334**) mapping to y ∈ {0, 13.15, 28.53}. Top-segment slope
≈ 940/unit; the PCHIP endpoint derivative used for extrapolation is
≈ **1024 score-units per unit of pinned input**. Consequences:

- The post-pin band carrying the *entire* on-distribution dial is
  ~0.13 wide (≈[49.83, 49.96] maps to ≈[0,100]). This matches the
  CLAUDE.md V39 note "raw bake output is a ~0.2-wide flat band."
- Inverting the spline from observed scores: identical≈49.933,
  blur1≈49.957, blur5≈49.990 in pinned space — a 0.057-wide spread,
  amplified ~1000× into 100 / 124 / 159.
- Because S4 **extrapolates linearly with unbounded slope**, any pinned
  value above ~49.96 produces score >100 with **no ceiling**.

**The dynamic range of the dial is smaller than the network's
off-manifold excursion.** The "signal" (0.13-wide band) is narrower than
the noise the net emits off-distribution, so off-distribution the spline
operates entirely in its unbounded-extrapolation regime. That is the
deep reason a *tiny* network misbehavior becomes a 158.

---

## 2. Why the validation regime is structurally blind to all of this

The acceptance functional was `A(f) = SROCC(f(X_test), Y_test)` over
natural-photo MOS corpora. Two exact statements:

1. **SROCC is invariant under any monotone reparametrization.** For any
   strictly-increasing g, `SROCC(g∘f) = SROCC(f)`. So SROCC cannot see
   boundedness, scale, the tanh pin, or the spline *at all* — f and its
   spline-mangled cousin are indistinguishable to it. (PLCC/Z-RMSE see
   scale, but only on `X_test`.)
2. **A(f) depends on f only through its values — only their ranks — on
   `X_test`.** The metric axioms (bounded, self-identity-maximal,
   degradation-monotone) are properties of f on its *entire* domain.
   `X_test` (natural photos) is a measure-zero slice of that domain.
   Maximizing A leaves f free to be arbitrary everywhere else.

So the pathology isn't bad luck — it is the *generic* outcome of
optimizing a flexible function against a rank-only objective on a narrow
slice. The linear baseline escaped it only because its hypothesis class
(non-negative-weighted distance) is bounded+monotone *by construction*,
so the axioms held on the whole domain regardless of where it was
trained. The MLP discarded the hypothesis-class guarantee and the
pipeline never replaced it with a check.

---

## 3. Principled detection (what each method can and cannot prove)

Four layers, cheap→expensive, refutation→proof:

### 3.1 Metamorphic / property-based invariant tests (refute, cheap)
Sample many `(content, degradation)` and assert the axioms. Detects
violations on sampled points; **cannot prove absence**. Must span
content classes (synthetic fractal/checker/noise/screen/line-art +
natural) because the pathology lives where coverage is thin.
- **Boundedness:** `0 ≤ f(x,y) ≤ 100`.
- **Self-identity maximal:** `f(x,x) = 100` and `f(x,x) ≥ f(x,y)`.
- **Symmetry** (if intended): `f(x,y)=f(y,x)`.
- **Degradation monotonicity:** for a strength-parametrized degradation
  `D_t` (blur/noise/quantize), `f(x, D_t(x))` non-increasing in `t`,
  on a *ladder* of t. This is the single most diagnostic test — it
  caught the inversion immediately.

### 3.2 OOD-input detection (refute at runtime, cheap)
The scaler already yields z-scores; the training distribution's own tail
gives a threshold (e.g. 99.9th-percentile training `rms z` and
`max|z|`). Flag any input beyond it as "network is extrapolating."
- Diagonal z-score is the cheap proxy; the proper multivariate test is
  **Mahalanobis distance** `√((z)ᵀ Σ⁻¹ (z))` using the training feature
  covariance Σ (captures correlated-feature excursions the diagonal
  misses).
- This doesn't fix the metric; it lets the runtime **fall back to a
  guaranteed-safe metric** (the linear V0_2) or widen reported
  uncertainty when off-manifold. It would have flagged every blur case
  above (max|z| 106 vs identical's 3.8).

### 3.3 Static gain / Lipschitz analysis (conservative a-priori bound, cheap)
Upper-bound the composed sensitivity **without running on OOD data**:
`L_total ≈ (∏ₗ ‖Wₗ‖₂ · slopeₗ) · max|spline′|`. The product of per-layer
spectral norms times activation slopes bounds the net's Lipschitz
constant; the spline's max slope (here ~1024) multiplies it. If
`L_total × (plausible input excursion)` ≫ 100, the metric is
unbounded-prone — computable at bake time as a red flag. V39 fails this
trivially: the spline term alone is ~1024.

### 3.4 Certified bounds / monotonicity (prove over a box, expensive)
- **Interval Bound Propagation / CROWN** over a box of standardized
  inputs gives *provable* output bounds (proves boundedness over that
  box).
- **MILP encoding of the ReLU network** (Liu et al., certified-monotonic
  line) can *verify or refute* monotonicity in a given input direction
  over a box. Sound but costly — use as a periodic audit, not per-epoch.
- Cheap refutation companion: sample gradients `∂f/∂(distortion-dir)`;
  any positive sign refutes monotonicity instantly.

**Summary:** tests/OOD-flags *refute* (cheap, run every bake);
gain-analysis gives a *conservative a-priori* bound (cheap); IBP/MILP
*prove* over a bounded domain (expensive, periodic). None alone is
sufficient — but the cheap layers would have blocked V39 on day one.

---

## 4. Guarantee by construction — the math

Goal: a learned `f(x,y) ∈ [0,100]` that is (i) bounded, (ii)
self-identity-maximal, (iii) monotone non-increasing in distortion, with
all three holding on the **entire** input domain (so they generalize to
content never trained on), while keeping more expressivity than linear.

### 4.1 Reparametrize to non-negative dissimilarity features
Feed the network **dissimilarity features** `d = φ(x,y) ∈ ℝ_{≥0}^m`, each
coordinate 0 iff locally identical and increasing with local distortion
(per-subband error energy, `1−SSIM_local`, `|Δ|` statistics). Many of
zensim's 372 are already this shape; the rest are *similarity* features
`s` and map by `s ↦ (1−s)` or `s ↦ −log s`. Key property: **φ(x,x)=0**.
Then define

```
f(x,y) = 100 · S( g( d ) ) ,   d = φ(x,y) ≥ 0
```

with `g: ℝ_{≥0}^m → ℝ_{≥0}` a learned "total distortion" and `S` a fixed
squashing distortion→quality.

### 4.2 Boundedness (immediate)
Pick `S: [0,∞) → (0,1]` strictly decreasing with `S(0)=1`, e.g.
`S(u)=e^{−u}` or `S(u)=1/(1+u)`. Then `f = 100·S(g(d)) ∈ (0,100]` for
all inputs, with `f=100` iff `g(d)=0`. **No post-hoc extrapolating
spline.** If dial-linearity vs MOS is wanted, the calibration must be a
**bounded monotone** map `[0,100]→[0,100]` (monotone with *constant*
(clamped) extrapolation, or isotonic regression) — it can never leave
range. This kills the >100 defect by construction.

### 4.3 Self-identity maximal (immediate, given 4.1–4.2)
`φ(x,x)=0 ⟹ g(0)`. Require `g(0)=0` (pin the bias, or use
`g(d)=‖h(d)‖` with `h(0)=0`). Since `S` is decreasing and `g≥0`,
`f(x,x)=100·S(0)=100=max f`. Holds for **all** content — including the
100σ-out Mandelbrot — because it's an algebraic property of the form,
not a learned fact.

### 4.4 Monotonicity in distortion (the core)
Require `g` **non-decreasing in each coordinate of d** (more
dissimilarity in any subband ⟹ ≥ total distortion ⟹ ≤ quality).
Coordinate-wise monotonicity is achievable by construction; options in
increasing expressivity:

- **Non-negative weights + monotone activation** (Sill, *Monotonic
  Networks*, 1997/98): every weight ≥0 (parametrize `W=softplus(θ)`) and
  activations non-decreasing ⟹ network non-decreasing in every input.
  Min-max networks are universal approximators of monotone functions.
- **Deep Lattice Networks / partial monotonic functions** (You et al.,
  NeurIPS 2017): calibrators + lattices, monotone in chosen inputs, free
  in others, more expressive.
- **Unconstrained Monotonic NNs (UMNN)** (Wehenkel & Louppe, NeurIPS
  2019): `g` as the integral of a positive network — monotone, very
  expressive.
- **Certified-monotonic + penalty** (Liu et al., NeurIPS 2020): train
  unconstrained with a monotonicity regularizer, verify by MILP.

**Recommended: partial monotonicity.** Split features into a
**distortion** set (the monotone path) and a **content-descriptor** set
(reference texture/activity/luminance — a *free*, non-monotone context
pathway). The content pathway emits **non-negative per-distortion gains**
(softplus gating); `g = Σ_j gain_j(content) · ψ_j(d_j)` with `ψ_j`
monotone-increasing, `gain_j ≥ 0`. Then content can modulate *how much*
a given distortion costs (HF texture can mask or expose artifacts) while
`g` stays monotone in `d` for every fixed content. This is exactly the
Deep-Lattice "monotone-in-some-inputs" setting and **preserves the
nonlinearity that justified leaving linear** — without sacrificing the
axiom. Crucially, monotonicity then holds at 100σ off-manifold too,
because it's structural.

### 4.5 Lipschitz boundedness for graceful off-manifold behavior
Monotone+bounded still permits ugly magnitude swings off-manifold (bad
calibration). Bound the network's Lipschitz constant — **spectral
normalization** per layer (Miyato et al., 2018), or a 1-Lipschitz
architecture (GroupSort, Anil et al., 2019). Then off-manifold heavy
distortion ⟹ `g` large but L-controlled ⟹ `S(g)→0` ⟹ `score→0`, the
**correct** direction ("unknown heavy distortion looks bad"), instead of
exploding to 158. Off-manifold the metric *degrades gracefully toward 0*
rather than diverging.

### 4.6 The guarantee (informal theorem)
> If `f = 100·S(g(φ(x,y)))` where (a) `φ ≥ 0` and `φ(x,x)=0`; (b) `g` is
> non-decreasing in its inputs with `g(0)=0` (partial-monotone: free on
> content inputs, monotone on distortion inputs); (c) `S:[0,∞)→(0,1]`
> strictly decreasing with `S(0)=1`; and (d) any calibration is a
> bounded monotone `[0,100]→[0,100]` map — then `f` satisfies
> boundedness, self-identity-maximality, and degradation-monotonicity on
> the **entire** input domain, and any monotone calibration leaves SROCC
> unchanged. Data shapes `g` only *within* the monotone class; it cannot
> break the axioms.

This is precisely the property the linear baseline had for free, restored
without giving up nonlinear expressivity.

---

## 5. Concrete redesign for zensim

1. **Feature reparam (S0/S1).** Audit the 372 features; express each as a
   non-negative dissimilarity (`1−s` / `−log s` for similarity features,
   keep error-energies as-is). Tag each feature `distortion` vs
   `content-descriptor`. `φ(x,x)=0` must hold per coordinate.
2. **Architecture (S2).** Partial-monotone head: a free spectral-normed
   MLP over content descriptors emits `softplus` non-negative gains;
   combine with monotone `ψ_j(d_j)` via non-negative weights into scalar
   `g ≥ 0`. Replace tanh-pin-into-extrapolating-spline with
   `f = 100·e^{−g}` (or `100/(1+g)`).
3. **Calibration (S4).** If dial-vs-MOS linearity is needed, fit isotonic
   regression or PCHIP with **constant (clamped) extrapolation** on
   `[0,100]`, applied after the squash — provably in-range, rank-
   preserving. *Never* linear-extrapolating, never high-gain on a thin
   band.
4. **Training losses.** Keep the rank loss. **Add a degradation-ladder
   monotonicity loss**: on-the-fly augmentation ladders (blur/noise/
   quantize at increasing strength) over arbitrary images *including
   synthetic*, penalize `f(x,D_{t+1}) > f(x,D_t)`. With the monotone
   architecture this loss is ≈0 by construction (it becomes a cheap
   correctness assertion); without it, it's the soft route. Add spectral-
   norm regularization on the free pathway.
5. **The invariant gate (process fix).** New `tests/metric_invariants.rs`
   + a `bake_verdict --invariants` mode running §3.1 (bounds /
   self-identity / degradation-monotonicity across synthetic+natural) and
   §3.3 (static gain = ∏ spectral norms × max spline slope). **Fail the
   bake on violation.** Wire into CI. Extends the existing "SROCC-only
   verdicts BANNED" rule from stats panels to structural invariants:
   SROCC is never again the sole gate.

---

## 6. Costs / tradeoffs (honest)

- Monotone/partial-monotone nets are somewhat less expressive per
  parameter; matching V39's CID22 SROCC may need more params or careful
  content/distortion split. But linear already proves monotone+bounded is
  *attainable*; the MLP's job becomes "beat linear's SROCC *within* the
  monotone class," which is a well-posed, safe search.
- MILP verification is expensive → periodic audit only. Per-epoch relies
  on the ladder loss + spectral norm + sampled invariant tests.
- Removing the high-gain spline compresses the dial; the bounded monotone
  calibration restores range *and* rank without the extrapolation hazard.
- The squash `S` changes the loss landscape; rank loss should be computed
  on `g` (or pre-squash) where gradients are well-scaled, with the squash
  only at output.

## Status / next
Analysis + design only; shipped V39 untouched (accepted known-limit). The
lowest-risk first increment is the **process fix (§5.5 invariant gate)** —
it's independent of any retrain, would have blocked V39, and turns the
red `score_sanity_checks` into a first-class, content-broad gate.
Architecture work (§5.1–5.4) is the larger follow-on.
