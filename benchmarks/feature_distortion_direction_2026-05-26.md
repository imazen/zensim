# Per-feature distortion-direction analysis (2026-05-26)

Empirical basis for a **monotone-by-construction** zensim encoder — the
fix for the V39 correctness defect (blur scoring ≥ identity on OOD
content). Run by `zensim/tests/feature_distortion_direction.rs`
(`cargo test --release -p zensim --features training --test
feature_distortion_direction -- --nocapture`).

## Method

For each of the 372 features, measure Spearman correlation with
**increasing distortion** across 4 content types (color_blocks, checker,
mandelbrot, value_noise) × 3 distortion ladders (blur radius 0→8;
posterize 8→1 bits; additive noise amp 0→64) = 12 correlations/feature.
Classify by sign-safety:

- **pin W1 ≥ 0** — the feature never significantly *decreases* with any
  distortion (`min_corr ≥ −0.15`). Safe to constrain non-negative.
- **free** — the feature genuinely *flips sign* (`min < −0.15 AND
  max > 0.15`): rises for one distortion type, falls for another.
  Constraining it either way is wrong (this is what collapsed V46b).

## Result

| Class | Count | Block spread |
|---|--:|---|
| **pin W1 ≥ 0** (sign-safe) | **300 / 372** | basic + peaks + masked + IW |
| **free** (genuine sign-flip) | **72 / 372** | 40 basic, 16 peaks, 8 masked, 8 IW |
| pin W1 ≤ 0 | 0 | — |

**All 372 features are exactly 0 at identity** (mean = max = 0.0000) —
they are pure error/difference features (`ssim_mean = mean(d)`,
`artifact = max(0,+d)`, `detail_lost = max(0,−d)`,
`hf_energy_loss/gain`, …). So identity is the natural maximum: with
`W1 ≥ 0` on the sign-safe features, `h = LeakyReLU(b1)` at identity and
rises monotonically as any distortion pushes those features up.

The 72 "free" features are **distortion-type-selective / spread**
features (variance of the error map, the blur-vs-ringing directional
pair, p95/peak spreads) — they rise for some distortions and fall for
others, so they carry real signal but cannot be sign-pinned. They are
NOT cleanly the IW/masked block (my prior guess) — they're spread
across all four blocks, concentrated in `basic` (40).

Mask artifact: `benchmarks/feature_sign_mask_2026-05-26.tsv`
(`feat_idx, sign_mask ∈ {pin_geq0, free}, mean/min/max corr`).

## V39 (Profile::A) defect — quantified

Blur-ladder scores (radius 0 = identity):

| content | r0 | r1 | r2 | r3 | r4 | r5 | r6 |
|---|--:|--:|--:|--:|--:|--:|--:|
| color_blocks | 100 | 68.6 | **100** | 100 | 100 | 100 | 100 |
| checker | 100 | 95.8 | **100** | 100 | 100 | 100 | 100 |
| mandelbrot | 100 | **100** | **100** | **100** | **100** | **100** | **100** |
| value_noise | 100 | **−105** | **−31** | **61** | 100 | 100 | 100 |

- **5 adjacent-step inversions** on the 4-content × 6-step ladder.
- mandelbrot: flat at 100 at *every* blur radius — V39 cannot tell a
  blurred fractal from the original (violates axiom 2, self-identity is
  the *unique* max).
- value_noise: −105 → −31 → 61 → 100 as blur *increases* — fully
  inverted (violates axiom 3, degradation monotonicity).
- The ≤100 clamp (commit 24f9346) fixed axiom 1 (boundedness) only; the
  ties + inversions remain.

`LinearBounded` (the V0_2-style bounded squash): **0 inversions**,
strictly decreasing on all four contents — the correct-by-construction
target (but the weaker non-MLP metric).

## Fix recipe (next step)

Retrain the per-sample-α encoder with a **per-feature sign mask**:
- `W1[:, j] ≥ 0` for the 300 sign-safe columns,
- **either** drop the 72 free columns (`W1[:, j] = 0` → STRICT
  monotonicity guarantee, lose 19% of features),
- **or** keep them free (partial monotonicity — monotone in the 300
  core-error features, modulated by the 72; NOT a strict guarantee but
  retains all signal).

Combined with `rank_w ≤ 0` + the increasing tanh pin, the strict variant
is bounded + self-identity-max + degradation-monotone on the ENTIRE
input domain, by construction. Use softplus reparam on the constrained
weights (`w = softplus(θ)`) so they reach 0 without the dead-weight
collapse the hard clamp caused (v45 series). The V46b dial collapse is
explained: it pinned ALL 372 ≥ 0, mis-constraining the 72 flips; the
masked variant only pins the 300 that are genuinely sign-safe.

Open question the retrain answers: does the 300-feature strict variant
(72 dropped) retain a competitive Mohammadi panel, or is the partial
variant (72 kept free) the better trade? Measure both vs V39.
