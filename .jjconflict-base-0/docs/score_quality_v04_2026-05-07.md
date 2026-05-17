# V0_4 vs V0_2 score-quality audit on held-out val (2026-05-07)

98 source-disjoint images × 19 q levels × per codec/knob → 232,902 pairs.

## Accuracy vs ssim2 target (held-out)

| Metric  | V0_4 (trained) | V0_2 (linear) | Δ          |
|---------|:--:|:--:|:--:|
| SROCC   | 0.9547         | 0.9085        | +0.046     |
| KROCC   | 0.8421         | 0.7824        | +0.060     |
| PCC     | 0.9600         | 0.9241        | +0.036     |
| bias    | +0.59          | +8.56         | -7.97      |
| MAE     | 4.00           | 9.18          | -5.18      |

**Win for V0_4 on ranking and calibration.**

## Per-curve monotonicity (12,258 curves, 220,644 q-steps)

| Source      | non-monotonic q-steps | %      |
|-------------|:--:|:--:|
| ssim2 (GT)  | 11,210 | 5.08% |
| V0_2        | 10,723 | 4.86% |
| **V0_4**    | **18,227** | **8.26%** |

**V0_4 is over-jagged.** 8.26% per-step violations vs ssim2's 5.08%
ground-truth — the trained model amplifies q-direction noise instead
of tracking smooth codec RD curves. V0_2's linear scorer was actually
slightly *smoother* than the ground truth (4.86%), suggesting a
linearly-mixed feature combination naturally averages out per-q noise
that an MLP picks up.

## Hypothesis & follow-up

Likely causes for V0_4 bumpiness:
1. RankNet loss uses `max_pairs_per_group=64` random-sampled pairs —
   doesn't preferentially enforce neighbor-q consistency.
2. ReLU MLP creates piecewise decision boundaries that can flip on
   neighboring feature vectors.
3. No regularization terms penalizing per-q derivative sign flips.

Candidate fixes for V0_4.1 / V0_5:
- Add a TV-style monotonicity regularizer:
  `L_tv = mean(relu(score(q_i) - score(q_{i+1})))` for adjacent q in
  the same group.
- Lower learning rate + higher weight decay to reduce model
  flexibility (currently lr=3e-3, wd=1e-5).
- Try a wider but shallower model (228 → 32 → 1) to favor smoother
  decision boundaries.

## Decision

V0_4 still ships — calibration improvement (-7.97 bias, -5.18 MAE) is
worth the bumpiness regression for the typical "compare two encodes
of the same source" use case where absolute score matters more than
the q-derivative. But this is documented as a known V0_4 artifact;
V0_5 should target both axes.
