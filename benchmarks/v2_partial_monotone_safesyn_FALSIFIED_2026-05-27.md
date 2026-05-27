# V2 partial-monotone-residual + safesyn — FALSIFIED (2026-05-27)

## Hypothesis (V1 → V2)

The V1 partial-monotone-residual probe (`monotone_subspace_probe.rs`,
`score = 100 − λm·(D_m(x_mono) − D_m(id)) − F_b(x_free)`, F_b ∈ [0, δ=25])
hit **CID22 SROCC 0.61** — far below V39's 0.879. V1 trained on
cid22_train + kadid + tid + konjnd only. **Hypothesis: the 0.61 is
data-starvation** (V1 omitted safesyn, 196k rows — the bulk of the
corpus). Falsification: if CID22 stays flat across the safesyn add, the
gap is the architecture, not the data.

## Result — FALSIFIED

Added `safesyn.parquet` (196,086 rows, ssim2-derived target, weight 1.0)
+ doubled pairs/epoch (30k→60k). Log:
`v2_partial_monotone_safesyn_FALSIFIED_2026-05-27.log`.

| corpus | V1 (no safesyn) | V2 (+safesyn, 2× ppe) | V39 ship | range (V2) |
|---|---|---|---|---|
| CID22 | 0.61 | **0.6254** | 0.879 | [−1151.8, 99.5] |
| KADID | — | 0.7392 | — | [−11206.8, 100.0] |
| TID | — | 0.7424 | — | [−4120.1, 99.8] |
| KonJND | — | **−0.3487** | — | [−1044.7, 99.6] |
| AIC-3 | — | 0.5646 | — | [−436.5, 99.7] |

Blur ladder (A1/A2/A3): **0 inversions, 0 above-identity, range to
−6001** — the axioms hold by construction, as designed.

196k extra rows + 2× epochs moved CID22 by **+0.015 (noise)** and drove
KonJND to **anti-correlated (−0.35)**. Data was not the gap.

## Root cause — a DESIGN contradiction, not an optimizer bug

The additive form is **self-defeating**:

    score = 100 − λm·ΔD_m − F_b,   ΔD_m ≥ 0 unbounded,   F_b ∈ [0, δ=25]

- The **resolution requirement** (a terrible distortion must score very
  negative, not cap at 99) forces ΔD_m to reach the thousands
  (observed: −11206 on KADID, −1151 even on CID22-compression).
- The **free head** is bounded to [0, 25]. So on every pair that isn't
  near-identity, `λm·ΔD_m ≫ δ` and the score order is **entirely**
  determined by −ΔD_m. SROCC(score) ≈ SROCC(−ΔD_m) = the rank quality
  of the **monotone backbone alone**.
- The monotone-on-300-pinned-features backbone ranks CID22 at ~0.625.
  The bounded free head is mathematically negligible against the
  unbounded mono term and **cannot** lift it.

The rising training loss (153→292 over 140 epochs) is the same effect:
the within-group RankNet margin |sb−sa| grows without bound as ΔD_m
scales, and a stable fraction of wrong-order pairs keep large margins.
Hard W≥0 projection + Adam adds churn (KonJND sign-flip) but is
secondary — the ceiling is structural: **"unbounded negative
resolution" via a monotone term directly defeats "free head refines
rank."**

## Reframe + next direction

The actual V39 defect is an **A2 violation** (a blurred image scored
*above* the identity = the reference itself), NOT a full-degradation-
monotonicity (A3) failure. Per the user's reframe, A3 is "human-authority"
monotonicity, not internal math — and the corruption corpus
(codec-corpus#7 / PR#8, built this session) is the held-out gate for the
property that actually matters: **broken < honest-LQ**.

So drop the monotone (A3) constraint; keep only **A1 (≤100) + A2
(identity is the unique max)**, which is achievable with an **expressive
non-negative penalty anchored at identity**:

    score = 100 − P(x),   P(x) ≥ 0,   P(identity) = 0,   P expressive

The natural form is a **learned-embedding distance** (deep metric):

    P(x) = λ · ‖φ(x) − φ(id)‖²,   φ = small MLP (no sign constraint)

- A1: P ≥ 0 ⇒ score ≤ 100, by construction.
- A2: P(id)=0 and P(x)>0 for x≠id (φ injective enough) ⇒ identity unique max.
- Panel: φ is unconstrained ⇒ free to match the expressive ~0.88 ceiling.
- A3 dropped as an absolute; the corruption corpus is the gate for
  "broken < honest-LQ" instead.

This decouples the two requirements V2 collided: the embedding distance
is unbounded-above (resolution) AND expressive (panel), and identity-
anchoring (not monotonicity) delivers the axiom that the V39 defect
actually broke. `residual_identity_probe` tried RBF/linear forms of this
and failed on *scale* (exp saturated; linear blew up) — the embedding-
distance form with explicit scale control is the untried, motivated
next rung. → V3 probe.
